"""Portable v6 data correctness, licensed rehearsal, exclusions and publication."""
from collections import Counter
from contextlib import redirect_stdout, redirect_stderr
import copy
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, models, pre_tokenizers, trainers
from silivra.agent import POLICY
from silivra.instructions import _check_splits, parse_action, prepare_instructions, validate_record
from silivra.tools import ToolRegistry

SCRIPT = Path(__file__).resolve().parents[1]/"scripts"/"build-native-english-v6.py"
SPEC = importlib.util.spec_from_file_location("native_english_v6", SCRIPT)
v6 = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(v6)


def fixture(identifier, prompt, answer, license="CC-BY-SA-3.0", prior=None):
    messages = [{"role":"system", "content":POLICY}]
    if prior:
        messages += [{"role":"user", "content":prior}, {"role":"assistant", "content":json.dumps({"final":"Previous answer."})}]
    messages += [{"role":"user", "content":prompt}, {"role":"assistant", "content":json.dumps({"final":answer})}]
    return validate_record({"id":identifier, "source":{"name":"Attributed test rehearsal", "license":license,
        "url":"https://example.com/licensed-fixture", "revision":"immutable-test-revision"}, "messages":messages})


class NativeEnglishV6Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.valid = v6.build_original()

    def test_determinism_diverse_answers_and_contexts(self):
        self.assertEqual((self.train,self.valid), v6.build_original())
        self.assertNotEqual(self.train, v6.build_original(4)[0])
        _check_splits(self.train,self.valid)
        self.assertGreater(sum(map(v6.assistant_count,self.train)), 8000)
        english = [row for row in self.train if row["id"].startswith("v6-english-")]
        noun_answers = {parse_action(row["messages"][-1]["content"])["final"] for row in english
                        if "noun" in row["messages"][-2]["content"] and "pronoun" not in row["messages"][-2]["content"]}
        self.assertGreaterEqual(len(noun_answers),3)
        widths = Counter()
        for row in self.train+self.valid:
            self.assertEqual(validate_record(row),row)
            self.assertEqual(row["source"]["license"],"MIT")
            self.assertIn("AI-authored",row["source"]["name"])
            system = row["messages"][0]["content"]
            widths[0 if " Tools: " not in system else len(json.loads(system.split(" Tools: ",1)[1]))] += 1
            self.assertNotIn("What is a English", row["messages"][-2]["content"])
        self.assertTrue(all(widths[count]>0 for count in (0,1,2)))
        for seed in (True,-1,2**32):
            with self.assertRaises(ValueError): v6.build_original(seed)

    def test_tool_arguments_and_calculator_wrappers_match_runtime(self):
        for row in self.train+self.valid:
            if not row["id"].startswith(("v6-calculate-","v6-search-")):
                continue
            messages = row["messages"]
            action = parse_action(messages[2]["content"])
            self.assertIn('"name":"'+action["tool"]+'"',messages[0]["content"])
            argument = next(iter(action["arguments"].values()))
            self.assertIn(argument,messages[1]["content"])
            if action["tool"]=="calculate" and len(messages)>3:
                actual = ToolRegistry(Path.cwd()).execute("calculate",action["arguments"])
                self.assertEqual(actual,json.loads(messages[3]["content"]))
                self.assertEqual(parse_action(messages[4]["content"]),{"final":actual["result"]["result"]})

    def test_fictional_search_observation_matches_mocked_real_adapter(self):
        row = next(r for r in self.train if r["id"].startswith("v6-search-") and len(r["messages"])==5)
        action = parse_action(row["messages"][2]["content"])
        expected = json.loads(row["messages"][3]["content"])
        result = expected["result"]
        item = result["results"][0]
        response = {"body":json.dumps([result["query"],[item["title"]],[item["snippet"]],[item["url"]]]).encode()}
        with patch.dict("os.environ",{"BRAVE_SEARCH_API_KEY":""}),patch("silivra.web._fetch_bytes",return_value=response):
            actual = ToolRegistry(Path.cwd(),allow_network=True).execute("web_search",action["arguments"])
        self.assertEqual(actual,expected)
        self.assertTrue(item["url"].startswith("https://example.com/fixture/"))
        self.assertIn("fictional",item["snippet"])

    def test_synthetic_python_targets_execute_correctly(self):
        for row in self.train+self.valid:
            if not row["id"].startswith("v6-python-"):
                continue
            prompt = row["messages"][-2]["content"]
            code = prompt.split("\n",1)[1]
            output=[]
            exec(compile(code,"<original-synthetic-fixture>","exec"),{"__builtins__":{"print":lambda value:output.append(str(value)),"len":len}})
            self.assertEqual(parse_action(row["messages"][-1]["content"]),{"final":"\n".join(output)})

    def test_exclusion_reader_rejects_answer_objects_and_hashes_prompts(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/"prompts.jsonl"
            path.write_text(json.dumps("  Copy   this ")+"\n")
            blocked,artifacts=v6.read_prompt_exclusions([path])
            self.assertEqual(blocked,{"Copy this"})
            self.assertEqual(artifacts[0]["sha256"],v6.sha256(path))
            path.write_text(json.dumps({"prompt":"Copy this","expected":{"final":"this"}})+"\n")
            with self.assertRaisesRegex(ValueError,"answer objects are forbidden"):
                v6.read_prompt_exclusions([path])
            for value in ("",12,None):
                path.write_text(json.dumps(value)+"\n")
                with self.assertRaises(ValueError):v6.read_prompt_exclusions([path])

    def test_mixer_excludes_every_user_message_and_literal_math_both_splits(self):
        old_train=[fixture("old-train-a","Available source","Old answer"),
                   fixture("old-train-b","Current question","Old answer",prior="Blocked memory"),
                   fixture("old-train-c","What is 4 + 3?","7")]
        old_valid=[fixture("old-valid-a","Reserved question","Old valid answer"),
                   fixture("old-valid-b","Compute 3+4.","7")]
        core_train=[v6.make_record("core","Reserved question","duplicate-prefix"),
                    v6.make_record("core","New task","New answer")]
        core_valid=[v6.make_record("core","New validation task","A valid answer")]
        snapshot=copy.deepcopy((old_train,old_valid))
        train,valid,_=v6.mix_records(old_train,old_valid,core_train,core_valid,
                                    blocked=["Blocked memory","Evaluate 3 + 4"],rehearsal_samples=10,original_samples=10)
        _check_splits(train,valid)
        all_prompts=set().union(*(v6.user_prompts(row) for row in train+valid))
        self.assertNotIn("Blocked memory",all_prompts)
        self.assertFalse(any(v6.math_problems(prompt) for prompt in all_prompts))
        self.assertNotIn("Reserved question",set().union(*(v6.user_prompts(row) for row in train)))
        self.assertEqual((old_train,old_valid),snapshot)
        retained=next(row for row in train if row["id"]=="old-train-a")
        self.assertEqual(retained["source"]["license"],"CC-BY-SA-3.0")
        self.assertEqual(retained,old_train[0])

    def test_selection_budgets_assistant_decisions_without_duplicates(self):
        rows=[fixture("r0","Question zero","a",prior="Earlier question"),
              fixture("r1","Question one","b"),fixture("r2","Question two","c"),
              fixture("r3","Question three","d",license="CDLA-Sharing-1.0")]
        selected=v6.select_sample_budget(rows,3,41,lambda row:row["source"]["license"])
        self.assertLessEqual(sum(map(v6.assistant_count,selected)),3)
        self.assertEqual(selected,v6.select_sample_budget(list(reversed(rows)),3,41,lambda row:row["source"]["license"]))
        self.assertEqual(len({r["id"] for r in selected}),len(selected))
        self.assertTrue(all(r in rows for r in selected))

    def test_atomic_preparation_provenance_ratios_and_no_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            tokenizer=Tokenizer(models.BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer=pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.train_from_iterator([POLICY],trainers.BpeTrainer(vocab_size=512,show_progress=False,
                initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),special_tokens=["<pad>","<bos>","<eos>","<unk>"]))
            token_path=root/"tokenizer.json";tokenizer.save(str(token_path))
            source=root/"old-source";source.mkdir()
            old_train=[fixture(f"old-train-{i}",f"Training question {i}","An attributed answer containing several distinct teaching words.") for i in range(4)]
            old_valid=[fixture(f"old-valid-{i}",f"Validation question {i}","An attributed answer containing several distinct teaching words.") for i in range(4)]
            for name,rows in (("train",old_train),("valid",old_valid)):
                (source/f"{name}.jsonl").write_text("".join(v6.dump(r)+"\n" for r in rows))
            old=root/"old";prepare_instructions(source/"train.jsonl",source/"valid.jsonl",token_path,old,max_seq_len=1024)
            core=([v6.make_record("core","An original training prompt","yes")],
                  [v6.make_record("core","An original validation prompt","yes")])
            output,prepared=root/"new-source",root/"new-prepared"
            argv=["--old",str(old),"--output",str(output),"--prepare",str(prepared),"--rehearsal-samples","4","--original-samples","1"]
            with patch.object(v6,"build_original",return_value=core),redirect_stdout(io.StringIO()):
                v6.main(argv)
            manifest=json.loads((prepared/"manifest.json").read_text())
            corpus=json.loads((prepared/"corpus-manifest.json").read_text())
            self.assertEqual(manifest["curriculum_provenance"]["corpus_manifest_sha256"],v6.sha256(prepared/"corpus-manifest.json"))
            self.assertEqual(corpus["generator_sha256"],v6.sha256(SCRIPT))
            self.assertEqual(corpus["rehearsal_manifest_sha256"],v6.sha256(old/"manifest.json"))
            for split in ("train","valid"):
                self.assertGreaterEqual(corpus["splits"][split]["rehearsal_sample_share"],.70)
                self.assertGreaterEqual(corpus["splits"][split]["rehearsal_supervised_token_share"],.70)
                self.assertEqual(corpus["splits"][split]["padded_cells"],5*1025)
                self.assertIn("CC-BY-SA-3.0",corpus["splits"][split]["licenses_by_assistant_sample"])
            snapshot=(prepared/"manifest.json").read_bytes()
            with self.assertRaises(SystemExit),redirect_stderr(io.StringIO()):v6.main(argv)
            self.assertEqual(snapshot,(prepared/"manifest.json").read_bytes())

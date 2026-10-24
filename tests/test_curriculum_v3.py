"""Offline v3 data correctness, partition and immutable publication checks."""

import ast
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from silivra.instructions import _check_splits, parse_action, validate_record


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build-curriculum-v3.py"
SPEC = importlib.util.spec_from_file_location("curriculum_v3", SCRIPT)
curriculum = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curriculum)


class CurriculumV3Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.valid = curriculum.build_splits()

    def test_determinism_attribution_and_holdout(self):
        self.assertEqual((self.train, self.valid), curriculum.build_splits())
        self.assertNotEqual(self.train, curriculum.build_splits(3)[0])
        self.assertTrue(1000 <= len(self.train) <= 2500)
        _check_splits(self.train, self.valid)
        for row in self.train + self.valid:
            self.assertEqual(validate_record(row), row)
            self.assertEqual(row["source"]["license"], "MIT")
            self.assertIn("Codex", row["source"]["author"])
        train_pairs, valid_pairs = set(), set()
        for rows, pairs in ((self.train, train_pairs), (self.valid, valid_pairs)):
            for row in rows:
                if row["id"].startswith("v3-addition-"):
                    pairs.add(tuple(map(int, re.findall(r"\d+", row["messages"][-2]["content"]))))
        self.assertFalse(train_pairs & valid_pairs)
        self.assertEqual(len(train_pairs | valid_pairs), 100)
        self.assertFalse(set(curriculum.TRAIN_WORDS) & set(curriculum.VALID_WORDS))
        self.assertFalse(set(curriculum.TRAIN_NAMES) & set(curriculum.VALID_NAMES))
        for seed in (True, -1, 2**32):
            with self.assertRaises(ValueError):
                curriculum.build_splits(seed)

    def test_computed_targets_are_grounded_and_python_is_executable(self):
        for row in self.train + self.valid:
            prompt = row["messages"][-2]["content"]
            answer = parse_action(row["messages"][-1]["content"])["final"]
            if row["id"].startswith(("v3-addition-", "v3-comparison-")):
                a, b = map(int, re.findall(r"\d+", prompt))
                expected = str(a + b) if row["id"].startswith("v3-addition-") else ("yes" if a > b else "no")
                self.assertEqual(answer, expected)
            elif row["id"].startswith("v3-python-"):
                output = []
                code = prompt.rsplit("\n", 1)[-1]
                exec(compile(ast.parse(code), "<synthetic-test>", "exec"),
                     {"__builtins__": {}, "print": output.append})
                self.assertEqual(answer, str(output[0]))
            elif row["id"].startswith("v3-copy-"):
                self.assertEqual(answer, prompt.split(": ", 1)[1])
            elif row["id"].startswith("v3-grounding-"):
                self.assertIn(answer, prompt)
        histories = [r for r in self.train if r["id"].startswith("v3-conversation-history-")]
        self.assertTrue(histories)
        self.assertTrue(all([m["role"] for m in r["messages"]] ==
                            ["system", "user", "assistant", "user", "assistant"] for r in histories))

    def test_prompt_exclusion_applies_to_prior_turns_without_targets(self):
        train, _ = curriculum.build_splits(excluded_prompts=["Hi", "How are you?"])
        self.assertLess(len(train), len(self.train))
        self.assertFalse(any(m["content"] in {"Hi", "How are you?"}
                             for r in train for m in r["messages"] if m["role"] == "user"))

    def test_publication_records_hashes_and_refuses_overwrite(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_path = root / "tokenizer.json"
            tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.train_from_iterator([curriculum.POLICY], trainers.BpeTrainer(
                vocab_size=512, show_progress=False, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"]))
            tokenizer.save(str(tokenizer_path))
            output, prepared = root / "raw", root / "prepared"
            exclusion = root / "suite.jsonl"
            # No expected answers or even complete evaluation fields required.
            exclusion.write_text(json.dumps({"prompt": "Hi!"}) + "\n")
            with redirect_stdout(io.StringIO()):
                curriculum.main(["--output", str(output), "--tokenizer", str(tokenizer_path),
                                 "--prepare", str(prepared), "--max-seq-len", "1024",
                                 "--exclude-suite", str(exclusion)])
            manifest = json.loads((output / "corpus-manifest.json").read_text())
            self.assertEqual(manifest["generator_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
            self.assertEqual(manifest["tokenizer_sha256"], hashlib.sha256(tokenizer_path.read_bytes()).hexdigest())
            for split in ("train", "valid"):
                self.assertEqual(manifest["splits"][split]["sha256"],
                                 hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest())
            self.assertTrue((prepared / "manifest.json").is_file())
            self.assertEqual(manifest["prompt_exclusion_sources"][0]["sha256"],
                             hashlib.sha256(exclusion.read_bytes()).hexdigest())
            with self.assertRaises(SystemExit):
                curriculum.main(["--output", str(output), "--tokenizer", str(tokenizer_path)])


if __name__ == "__main__":
    unittest.main()

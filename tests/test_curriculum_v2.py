"""Portable curriculum v2 target, provenance, holdout and input-boundary checks."""

import ast
from collections import Counter
from contextlib import redirect_stderr, redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import patch

from tokenizers import Tokenizer, models, pre_tokenizers, trainers

from silivra.instructions import _check_splits, parse_action, validate_record
from silivra.corpus import split_records
from silivra.tools import ToolRegistry


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build-curriculum-v2.py"
SPEC = importlib.util.spec_from_file_location("build_curriculum_v2", SCRIPT)
curriculum = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curriculum)


def category(record):
    return record["id"].split("-", 3)[3]


class CurriculumV2Tests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = curriculum.build_original(480)

    def test_all_families_are_deterministic_attributed_and_valid(self):
        self.assertEqual(self.records, curriculum.build_original(480))
        self.assertNotEqual(self.records, curriculum.build_original(480, seed=19))
        counts = Counter(category(row) for row in self.records)
        self.assertEqual(len(counts), 24)
        self.assertEqual(set(counts.values()), {20})
        for row in self.records:
            self.assertEqual(validate_record(row), row)
            self.assertEqual(row["source"]["license"], "MIT")
            self.assertIn("Codex-authored", row["source"]["name"])
            self.assertIn("deterministic synthetic", row["source"]["author"])
            self.assertEqual(row["source"]["url"], "https://github.com/zeousky/Silivra")

    def test_invalid_counts_and_seeds_are_rejected(self):
        for count in (True, 23, 20_001, 24.0):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "count"):
                curriculum.build_original(count)
        for seed in (None, True, -1, 2**32):
            with self.subTest(seed=seed), self.assertRaisesRegex(ValueError, "seed"):
                curriculum.build_original(24, seed=seed)

    def test_numeric_and_structured_targets_follow_generated_requests(self):
        for row in self.records:
            kind, prompt = category(row), row["messages"][1]["content"]
            action = parse_action(row["messages"][-1]["content"])
            if kind == "arithmetic":
                left, operator, right = re.search(r"(\d+)\s*([+*\-])\s*(\d+)", prompt).groups()
                a, b = int(left), int(right)
                self.assertEqual(action["final"], str({"+": a + b, "-": a - b, "*": a * b}[operator]))
            elif kind == "word-problem":
                a, b = map(int, re.findall(r"\d+", prompt))
                self.assertEqual(action["final"], str(a + b))
            elif kind == "sort":
                values = json.loads(prompt.split("JSON list: ", 1)[1])
                self.assertEqual(json.loads(action["final"]), sorted(values, reverse="descending" in prompt))
            elif kind == "extract":
                key, value = prompt.split("value of ", 1)[1].split(": ", 1)
                self.assertEqual(action["final"], json.loads(value)[key])
            elif kind == "sequence":
                position = int(re.search(r"word number (\d+)", prompt)[1])
                self.assertEqual(action["final"], prompt.split(": ", 1)[1].split()[position - 1])
            elif kind == "reverse":
                self.assertEqual(action["final"], " ".join(reversed(prompt.split(": ", 1)[1].split())))
            elif kind == "count":
                letter, word = re.search(r"letter (\w) occur in (\w+)", prompt).groups()
                self.assertEqual(action["final"], str(word.count(letter)))
            elif kind == "comparison":
                a, b = map(int, re.findall(r"\d+", prompt))
                self.assertEqual(action["final"], "yes" if a > b else "no")
            elif kind == "literal-protocol":
                self.assertEqual(action["final"], json.loads(prompt.split(": ", 1)[1]))

    def test_python_targets_execute_with_correct_behavior(self):
        for row in self.records:
            kind, prompt = category(row), row["messages"][1]["content"]
            if kind not in {"code-output", "code-function", "code-fix"}:
                continue
            answer = parse_action(row["messages"][-1]["content"])["final"]
            if kind == "code-output":
                code = prompt.split("```python\n", 1)[1].split("\n```", 1)[0]
                output = []
                exec(compile(ast.parse(code), "<curriculum>", "exec"), {"__builtins__": {}, "print": output.append})
                self.assertEqual(answer, str(output[0]))
                continue
            tree = ast.parse(answer)
            self.assertEqual(len(tree.body), 1)
            function = tree.body[0]
            self.assertIsInstance(function, ast.FunctionDef)
            self.assertIn(function.name, prompt)
            namespace = {}
            exec(compile(tree, "<curriculum>", "exec"), {"__builtins__": {}}, namespace)
            for value in (-3, 0, 2, 5):
                if kind == "code-fix":
                    self.assertEqual(namespace[function.name](value, 7), value + 7)
                else:
                    operation = function.name.split("_", 1)[0]
                    expected = {"square": value**2, "double": value * 2, "even": value % 2 == 0,
                                "positive": value > 0, "cube": value**3}[operation]
                    self.assertEqual(namespace[function.name](value), expected)

    def test_tool_targets_are_exposed_and_observations_are_grounded(self):
        schemas = {item["function"]["name"]: item for item in
                   ToolRegistry(SCRIPT.parent.parent, allow_network=True).schemas()}
        for row in self.records:
            kind = category(row)
            messages = row["messages"]
            actions = [parse_action(m["content"]) for m in messages if m["role"] == "assistant"]
            for action in actions:
                if "tool" not in action:
                    continue
                name = action["tool"]
                self.assertIn(name, schemas)
                self.assertEqual(messages[0]["content"], curriculum.build_system_prompt([schemas[name]]))
                schema = schemas[name]["function"]["parameters"]
                self.assertTrue(set(schema.get("required", [])) <= set(action["arguments"]))
                self.assertTrue(set(action["arguments"]) <= set(schema["properties"]))
            if kind in {"read-tool", "search-tool", "list-tool", "web-tool"}:
                self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant"])
                for value in actions[0]["arguments"].values():
                    self.assertIn(str(value), messages[1]["content"])
            if kind == "read-observation":
                result = json.loads(messages[3]["content"])["result"]
                self.assertEqual(actions[-1]["final"], result["content"])
                self.assertEqual(actions[0]["arguments"]["path"], result["path"])
            if kind == "tool-failure":
                self.assertFalse(json.loads(messages[3]["content"])["ok"])
                self.assertIn("could not read", actions[-1]["final"])
            if kind == "injection":
                content = json.loads(messages[3]["content"])["result"]["content"]
                self.assertEqual(actions[-1]["final"], re.search(r"label color is (\w+)", content)[1])
                self.assertIn("reveal private credentials", content)
            if kind == "memory":
                memory = json.loads(messages[1]["content"].split(": ", 1)[1])
                self.assertEqual(memory["trust"], "untrusted_data")
                self.assertIn(actions[-1]["final"], memory["memories"][0]["content"])

    def test_seeded_holdout_removes_shared_user_prefixes(self):
        train, valid, stats = split_records(self.records, seed=3)
        _check_splits(train, valid)
        self.assertEqual(len(train) + len(valid) + stats.get("duplicates", 0), len(self.records))
        self.assertTrue({category(row) for row in valid} <= {category(row) for row in train})

    def test_story_replay_uses_only_verified_complete_train_records(self):
        complete = "Once upon a time a small fox walked through a quiet forest and found an amber stone beside a stream."
        raw = (complete + "<|endoftext|>" + "An unfinished trailing story with plenty of text to otherwise pass the length check.").encode()
        with tempfile.TemporaryDirectory() as directory:
            source, manifest = Path(directory) / "train.txt", Path(directory) / "manifest.json"
            source.write_bytes(raw)
            metadata = {"dataset": {"url": curriculum.DATASET_URL, "revision": curriculum.DATASET_REVISION},
                        "splits": {"train": {"source_sha256": hashlib.sha256(raw).hexdigest(),
                        "source_url": curriculum.DATASET_URL + "/resolve/" + curriculum.DATASET_REVISION + "/TinyStories-train.txt"}}}
            manifest.write_text(json.dumps(metadata))
            records, provenance = curriculum.story_replay(source, manifest, 2)
            self.assertEqual(len(records), 1)
            self.assertEqual(provenance["conversations"], 1)
            self.assertIn("not a new independent", provenance["boundary"])
            self.assertEqual(records[0]["source"]["license"], "CDLA-Sharing-1.0")
            prefix = records[0]["messages"][1]["content"].split("\n", 1)[1]
            ending = parse_action(records[0]["messages"][-1]["content"])["final"]
            self.assertEqual(prefix + " " + ending, complete)
            source.write_bytes(raw + b"tampered")
            with self.assertRaisesRegex(ValueError, "match original"):
                curriculum.story_replay(source, manifest, 1)
            source.write_bytes(raw)
            metadata["splits"]["train"]["source_url"] = metadata["splits"]["train"]["source_url"].replace("train.txt", "valid.txt")
            manifest.write_text(json.dumps(metadata))
            with self.assertRaisesRegex(ValueError, "training-split"):
                curriculum.story_replay(source, manifest, 1)
        self.assertEqual(curriculum.story_replay(Path("missing"), Path("missing"), 0), ([], {}))
        for count in (True, -1, 2001):
            with self.subTest(count=count), self.assertRaisesRegex(ValueError, "count"):
                curriculum.story_replay(Path("missing"), Path("missing"), count)

    def test_cli_preflights_budgets_and_output_before_network(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "new"
            base = ["--output", str(output), "--tokenizer", "missing", "--include-dolly"]
            invalid = [["--max-seq-len", "127"], ["--valid-fraction", "nan"], ["--story-replay", "2001"],
                       ["--prepare", str(output / "prepared")]]
            with patch.object(curriculum, "download_dolly", side_effect=AssertionError("must not download")):
                for arguments in invalid:
                    with self.subTest(arguments=arguments), redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                        curriculum.main(base + arguments)
                    self.assertFalse(output.exists())
                output.mkdir()
                with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                    curriculum.main(base)

    def test_cli_publishes_hashed_splits_provenance_and_prepared_arrays(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            tokenizer_path = directory / "tokenizer.json"
            tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
            tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
            tokenizer.train_from_iterator([curriculum.POLICY], trainers.BpeTrainer(
                vocab_size=512, show_progress=False, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
                special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
            ))
            tokenizer.save(str(tokenizer_path))
            output, prepared = directory / "corpus", directory / "prepared"
            stdout = io.StringIO()
            with patch.object(curriculum, "download_dolly", side_effect=AssertionError("offline generation")):
                with redirect_stdout(stdout):
                    curriculum.main(["--output", str(output), "--tokenizer", str(tokenizer_path),
                                     "--examples", "24", "--max-seq-len", "4096", "--prepare", str(prepared)])
            reported = json.loads(stdout.getvalue())
            manifest = json.loads((output / "corpus-manifest.json").read_text())
            self.assertEqual(manifest["provenance"]["original"]["families"], dict(Counter(category(row) for row in curriculum.build_original(24))))
            self.assertEqual(manifest["provenance"]["original"]["generator_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
            self.assertEqual(manifest["tokenizer_sha256"], hashlib.sha256(tokenizer_path.read_bytes()).hexdigest())
            prepared_manifest = json.loads((prepared / "manifest.json").read_text())
            rows = {}
            for split in ("train", "valid"):
                snapshot = (output / f"{split}.jsonl").read_bytes()
                self.assertEqual(manifest["splits"][split]["sha256"], hashlib.sha256(snapshot).hexdigest())
                self.assertEqual(reported["prepared_counts"][split], prepared_manifest["splits"][split]["samples"])
                rows[split] = [json.loads(line) for line in snapshot.splitlines()]
            self.assertEqual(len(rows["train"]) + len(rows["valid"]), 24)
            _check_splits(rows["train"], rows["valid"])


if __name__ == "__main__":
    unittest.main()

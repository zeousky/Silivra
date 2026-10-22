"""Original procedural curriculum integrity, targets, splits and publication."""

import ast
from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import re
import tempfile
import unittest

from silivra.agent import POLICY, build_system_prompt
from silivra.instructions import _check_splits, encode_turns, parse_action, validate_record
from silivra.tools import ToolRegistry


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build-curriculum.py"
SPEC = importlib.util.spec_from_file_location("build_curriculum", SCRIPT)
curriculum = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(curriculum)


def family(record):
    return record["source"]["revision"].rsplit("family=", 1)[1]


class CurriculumTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records = curriculum.build_curriculum()

    def test_deterministic_original_provenance_and_balanced_schema(self):
        self.assertEqual(self.records, curriculum.build_curriculum())
        self.assertNotEqual(self.records, curriculum.build_curriculum(seed=18))
        for split, count in (("train", 400), ("valid", 100)):
            self.assertEqual(len(self.records[split]), count)
            families = {name: 0 for name in curriculum.FAMILIES}
            for record in self.records[split]:
                self.assertEqual(validate_record(record), record)
                families[family(record)] += 1
                self.assertEqual(record["source"]["license"], "MIT")
                self.assertIn("AI-assisted", record["source"]["name"])
                self.assertEqual(record["source"]["url"], "local:scripts/build-curriculum.py")
                self.assertEqual([m["role"] for m in record["messages"]], ["system", "user", "assistant"])
            self.assertEqual(set(families.values()), {count // 10})

    def test_no_conversation_prompt_filename_or_operand_split_leakage(self):
        _check_splits(self.records["train"], self.records["valid"])
        prompts, files, operands = {}, {}, {}
        for split, rows in self.records.items():
            prompts[split] = {row["messages"][1]["content"] for row in rows}
            self.assertEqual(len(prompts[split]), len(rows))
            files[split], operands[split] = set(), set()
            for row in rows:
                kind = family(row)
                action = parse_action(row["messages"][-1]["content"])
                if kind == "read_file":
                    files[split].add(action["arguments"]["path"])
                if kind in {"addition", "subtraction", "multiplication"}:
                    operands[split].update(map(int, re.findall(r"\d+", row["messages"][1]["content"])))
        self.assertFalse(prompts["train"] & prompts["valid"])
        self.assertFalse(files["train"] & files["valid"])
        self.assertFalse(operands["train"] & operands["valid"])

    def test_literal_and_arithmetic_answers_are_programmatically_correct(self):
        for rows in self.records.values():
            for row in rows:
                kind = family(row)
                prompt = row["messages"][1]["content"]
                action = parse_action(row["messages"][-1]["content"])
                if kind in {"copy", "exact"}:
                    self.assertEqual(action["final"], prompt.split(": ", 1)[1])
                elif kind in {"addition", "subtraction", "multiplication"}:
                    left, right = map(int, re.findall(r"\d+", prompt))
                    if prompt.startswith("Subtract"):
                        left, right = right, left
                    expected = {"addition": left + right, "subtraction": left - right,
                                "multiplication": left * right}[kind]
                    self.assertEqual(action, {"final": str(expected)})

    def test_python_targets_are_valid_and_have_correct_behavior(self):
        globals_ = {"__builtins__": {}, "len": len, "sum": sum, "sorted": sorted, "min": min, "max": max}
        for rows in self.records.values():
            for row in rows:
                kind = family(row)
                if kind not in {"expression", "function"}:
                    continue
                code = parse_action(row["messages"][-1]["content"])["final"]
                prompt = row["messages"][1]["content"]
                tree = ast.parse(code, mode="eval" if kind == "expression" else "exec")
                if kind == "expression":
                    names = {node.id for node in ast.walk(tree) if isinstance(node, ast.Name)} - set(globals_)
                    self.assertEqual(len(names), 1)
                    name = names.pop()
                    value = {"a": 1, "b": 2} if ".keys()" in code else "fern" if ".upper()" in code else [3, 1, 2]
                    actual = eval(compile(tree, "<curriculum>", "eval"), globals_, {name: value})
                    if ".keys()" in code:
                        self.assertEqual(list(actual), ["a", "b"])
                    elif ".upper()" in code:
                        self.assertEqual(actual, "FERN")
                    else:
                        expected = (3 if "length" in prompt or "first item" in prompt or "largest" in prompt
                                    else 6 if "sum" in prompt else [1, 2, 3] if "sorted" in prompt
                                    else 2 if "last item" in prompt else [2, 1, 3] if "reversed" in prompt else 1)
                        self.assertEqual(actual, expected)
                else:
                    self.assertEqual(len(tree.body), 1)
                    function = tree.body[0]
                    self.assertIsInstance(function, ast.FunctionDef)
                    self.assertIn(function.name + "(value)", prompt)
                    namespace = {}
                    exec(compile(tree, "<curriculum>", "exec"), {"__builtins__": {}}, namespace)
                    expected = {"square": 9, "double": 6, "increment": 4, "negate": -3, "identity": 3}
                    self.assertEqual(namespace[function.name](3), expected[function.name.split("_", 1)[0]])

    def test_tool_proposals_use_runtime_schema_and_do_not_fabricate_results(self):
        schemas = {schema["function"]["name"]: schema for schema in
                   ToolRegistry(SCRIPT.parent.parent, allow_network=True).schemas()}
        for rows in self.records.values():
            for row in rows:
                kind = family(row)
                action = parse_action(row["messages"][-1]["content"])
                if kind not in {"read_file", "search_files", "web_search"}:
                    self.assertEqual(row["messages"][0]["content"], POLICY)
                    continue
                self.assertEqual(action["tool"], kind)
                self.assertEqual(row["messages"][0]["content"], build_system_prompt([schemas[kind]]))
                argument = action["arguments"]["path" if kind == "read_file" else "query"]
                self.assertIn(argument, row["messages"][1]["content"])
                self.assertEqual(len(action["arguments"]), 1)
                self.assertNotIn("tool", [message["role"] for message in row["messages"]])

    def test_pending_tool_target_encodes_without_fake_result(self):
        from tokenizers import Tokenizer, models, pre_tokenizers, trainers
        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.train_from_iterator(["Original portable test tokenizer."], trainers.BpeTrainer(
            vocab_size=260, show_progress=False, initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            special_tokens=["<pad>", "<bos>", "<eos>", "<unk>"],
        ))
        row = next(row for row in self.records["train"] if family(row) == "read_file")
        ids, mask = encode_turns(validate_record(row), tokenizer, 4096)[0]
        expected = tokenizer.encode(row["messages"][-1]["content"], add_special_tokens=False).ids
        self.assertEqual(ids[-len(expected)-1:-1], expected)
        self.assertEqual(sum(mask), len(expected) + 1)

    def test_evaluation_exactly_matches_heldout_records_and_runtime_schemas(self):
        before = json.dumps(self.records, sort_keys=True)
        cases = curriculum.build_evaluation(self.records["valid"])
        self.assertEqual(cases, curriculum.build_evaluation(self.records["valid"]))
        self.assertEqual(len(cases), 100)
        schemas = {schema["function"]["name"]: schema for schema in
                   ToolRegistry(SCRIPT.parent.parent, allow_network=True).schemas()}
        for case, record in zip(cases, self.records["valid"]):
            self.assertEqual(set(case), {"id", "category", "prompt", "tools", "expected"})
            self.assertEqual(case["id"], record["id"])
            self.assertEqual(case["category"], family(record))
            self.assertEqual(case["prompt"], record["messages"][1]["content"])
            self.assertEqual(case["expected"], parse_action(record["messages"][-1]["content"]))
            action = case["expected"]
            self.assertEqual(case["tools"], [schemas[action["tool"]]] if "tool" in action else [])
            self.assertEqual(build_system_prompt(case["tools"]), record["messages"][0]["content"])
        self.assertEqual(json.dumps(self.records, sort_keys=True), before)
        with self.assertRaises(ValueError):
            curriculum.build_evaluation(self.records["train"])

    def test_evaluation_output_is_accepted_by_nonexecuting_evaluator(self):
        from silivra.evaluation import evaluate_assistant
        expected = {record["messages"][1]["content"]: record["messages"][-1]["content"]
                    for record in self.records["valid"]}

        class OracleBackend:
            def generate(self, messages, tools):
                return expected[messages[-1]["content"]]

        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "curriculum"
            curriculum.write_curriculum(output)
            result = evaluate_assistant(OracleBackend(), output / "evaluation.jsonl")
        self.assertEqual(result["cases"], 100)
        self.assertEqual(result["passed"], 100)
        self.assertEqual(result["valid_actions"], 100)
        self.assertFalse(result["tools_executed"])

    def test_atomic_publication_hashes_and_refusal_to_overwrite(self):
        import hashlib
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "curriculum"
            manifest = curriculum.write_curriculum(output)
            self.assertEqual(manifest["splits"]["train"]["records"], 400)
            self.assertEqual(manifest["splits"]["valid"]["records"], 100)
            for split in ("train", "valid"):
                content = (output / f"{split}.jsonl").read_bytes()
                self.assertEqual(hashlib.sha256(content).hexdigest(), manifest["splits"][split]["sha256"])
                self.assertEqual(len(content.splitlines()), manifest["splits"][split]["records"])
            evaluation = (output / "evaluation.jsonl").read_bytes()
            self.assertEqual(manifest["evaluation"]["records"], 100)
            self.assertEqual(manifest["evaluation"]["source_split"], "valid")
            self.assertEqual(manifest["evaluation"]["sha256"], hashlib.sha256(evaluation).hexdigest())
            self.assertEqual(len(evaluation.splitlines()), 100)
            self.assertEqual(json.loads((output / "manifest.json").read_text()), manifest)
            with self.assertRaises(FileExistsError):
                curriculum.write_curriculum(output)

    def test_invalid_seed_and_output_outside_ignored_data_are_rejected(self):
        for seed in (-1, True, 2**32):
            with self.assertRaises(ValueError):
                curriculum.build_curriculum(seed)
        with tempfile.TemporaryDirectory() as temporary:
            output = Path(temporary) / "outside"
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                curriculum.main(["--output", str(output)])
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()

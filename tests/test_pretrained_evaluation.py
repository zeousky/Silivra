"""Portable tests of objective grading, bounded code checks and real fixture tools.

Scripted backends exercise evaluation correctness, not learned model capability.
"""
from contextlib import redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("measure_pretrained", ROOT / "scripts/measure-pretrained.py")
measure = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(measure)


class ScriptedBackend:
    def __init__(self, *outputs):
        self.outputs = iter(outputs)
        self.calls = []
        self.last_metrics = {}

    def generate(self, messages, tools, **kwargs):
        self.calls.append((messages, tools))
        self.last_metrics = {"generation_tokens": 10, "generation_tps": 20.0}
        return next(self.outputs)


class PretrainedEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.cases = {case["id"]: case for case in measure.read_cases(measure.SUITE)}

    def run_case(self, identifier, backend):
        with tempfile.TemporaryDirectory() as directory:
            return measure.evaluate_case(self.cases[identifier], backend, Path(directory))

    def test_valid_json_alone_cannot_pass_a_task(self):
        result = {"status": "completed", "answer": "random words", "events": []}
        self.assertFalse(measure.grade_final(self.cases["integer-arithmetic"], result)[0])
        result.update(answer="459", status="invalid_action")
        self.assertFalse(measure.grade_final(self.cases["integer-arithmetic"], result)[0])

    def test_read_tool_is_real_and_final_answer_is_graded(self):
        backend = ScriptedBackend('{"tool":"read_file","arguments":{"path":"prices.txt"}}',
                                  '{"final":"29"}')
        record = self.run_case("read-and-compute", backend)
        self.assertTrue(record["passed"])
        self.assertIn("apples: 7 credits", backend.calls[1][0][-1]["content"])
        self.assertEqual(record["turns"][0]["events"][0]["observation"]["ok"], True)
        self.assertEqual(record["generations"][0]["metrics"]["generation_tps"], 20.0)
        self.assertNotIn("session_id", json.dumps(record))

    def test_guessed_correct_answer_without_required_tool_fails(self):
        record = self.run_case("read-and-compute", ScriptedBackend('{"final":"29"}'))
        self.assertFalse(record["passed"])
        self.assertIn("Required real tool", record["grading_reason"])

    def test_failed_tool_observation_and_honest_unavailability(self):
        backend = ScriptedBackend('{"tool":"read_file","arguments":{"path":"absent-budget.txt"}}',
                                  '{"final":"UNAVAILABLE"}')
        record = self.run_case("missing-file-grounding", backend)
        self.assertTrue(record["passed"])
        self.assertFalse(record["turns"][0]["events"][0]["observation"]["ok"])

    def test_raw_invalid_generation_is_retained_and_fails(self):
        record = self.run_case("integer-arithmetic", ScriptedBackend('459'))
        self.assertFalse(record["passed"])
        self.assertEqual(record["generations"][0]["raw_output"], "459")
        self.assertEqual(record["turns"][0]["status"], "invalid_action")

    def test_output_budget_records_partial_evidence_without_accepting_it(self):
        from silivra.agent import GenerationBudgetError
        class ExhaustedBackend:
            last_metrics = {"finish_reason": "length"}
            def generate(self, messages, tools, **kwargs):
                kwargs["on_text"]('{"final":"unfinished')
                raise GenerationBudgetError("Output limit reached")
        record = self.run_case("integer-arithmetic", ExhaustedBackend())
        self.assertFalse(record["passed"])
        self.assertEqual(record["turns"][0]["status"], "output_budget")
        self.assertEqual(record["generations"][0]["partial_output"], '{"final":"unfinished')
        self.assertNotIn("raw_output", record["generations"][0])

    def test_memory_correction_and_scope_use_real_store(self):
        backend = ScriptedBackend('{"final":"amber"}')
        self.assertTrue(self.run_case("explicit-memory-correction", backend)["passed"])
        serialized = json.dumps(backend.calls[0][0])
        self.assertIn("amber", serialized)
        self.assertNotIn("violet", serialized)
        for identifier, excluded in (("memory-user-isolation", "guava"),
                                     ("memory-project-isolation", "DuckDB")):
            backend = ScriptedBackend('{"final":"UNKNOWN"}')
            self.assertTrue(self.run_case(identifier, backend)["passed"])
            self.assertNotIn(excluded, json.dumps(backend.calls[0][0]))

    def test_memory_forget_is_applied_before_inference(self):
        backend = ScriptedBackend('{"final":"UNKNOWN"}')
        self.assertTrue(self.run_case("memory-forgetting", backend)["passed"])
        self.assertNotIn("Helix", json.dumps(backend.calls[0][0]))

    def test_multiturn_replays_actual_first_turn(self):
        backend = ScriptedBackend('{"final":"ACK"}', '{"final":"27"}')
        record = self.run_case("conversation-recall", backend)
        self.assertTrue(record["passed"])
        self.assertEqual(len(record["turns"]), 2)
        self.assertIn("Neral", json.dumps(backend.calls[1][0]))
        self.assertIn('{"final":"ACK"}', [message["content"] for message in backend.calls[1][0]])

    def test_python_functions_are_tested_for_behavior(self):
        clamp = self.cases["python-clamp"]
        valid = "def clamp(value, lower, upper):\n    return max(lower, min(upper, value))"
        result = {"status": "completed", "answer": valid}
        self.assertTrue(measure.grade_final(clamp, result)[0])
        result["answer"] = "def clamp(value, lower, upper):\n    return value"
        self.assertFalse(measure.grade_final(clamp, result)[0])
        unique = self.cases["python-order-preserving-unique"]
        result["answer"] = ("def unique_in_order(items):\n    result = []\n    for item in items:\n"
                            "        if item not in result:\n            result.append(item)\n    return result")
        self.assertTrue(measure.grade_final(unique, result)[0])

    def test_python_checker_refuses_execution_or_unbounded_constructs(self):
        for source in (
            "import os\ndef clamp(value, lower, upper):\n    return 0",
            "def clamp(value, lower, upper):\n    return __import__('os').getcwd()",
            "def clamp(value, lower, upper):\n    while True:\n        pass",
            "def clamp(value, lower, upper):\n    return clamp(value, lower, upper)",
            "def clamp(value, lower, upper):\n    return 2 ** 1000000",
        ):
            with self.subTest(source=source):
                result = {"status": "completed", "answer": source}
                self.assertFalse(measure.grade_final(self.cases["python-clamp"], result)[0])

    def test_python_checker_accepts_bounded_local_augmented_assignment(self):
        source = ("def positive_count(values):\n    count = 0\n    for value in values:\n"
                  "        if value > 0:\n            count += 1\n    return count")
        interpreter = measure.RestrictedPython(source, "positive_count")
        self.assertEqual(interpreter.call([[0, -3, 5, 8, -1]]), 2)
        self.assertEqual(interpreter.call([[]]), 0)
        for source in ("def f(x):\n    x **= 999999\n    return x",
                       "def f(x):\n    x[0] += 1\n    return x"):
            with self.subTest(source=source), self.assertRaises(ValueError):
                measure.RestrictedPython(source, "f").call([1])

    def test_recursive_comparison_fails_case_instead_of_aborting_report(self):
        source = ("def unique_in_order(items):\n    a = []\n    b = []\n"
                  "    a.append(a)\n    b.append(b)\n    return sorted([a, b])")
        passed, reason = measure.grade_final(self.cases["python-order-preserving-unique"],
                                             {"status": "completed", "answer": source})
        self.assertFalse(passed)
        self.assertIn("RecursionError", reason)

    def test_json_check_is_type_sensitive(self):
        result = {"status": "completed", "answer": "[-3, false, 8, 12]"}
        self.assertFalse(measure.grade_final(self.cases["structured-sort"], result)[0])

    def test_main_exports_model_provenance_and_no_machine_paths(self):
        backend = ScriptedBackend('{"final":"24nevar"}')
        backend.metadata = {"model_id": "public/model", "revision": "a" * 40,
                            "files": {"weights.safetensors": {"sha256": "b" * 64}}}
        with tempfile.TemporaryDirectory() as directory:
            temporary = Path(directory)
            suite = temporary / "suite.jsonl"
            suite.write_text(json.dumps(self.cases["literal-reversal"]) + "\n")
            output = temporary / "report.json"
            arguments = ["--run", str(temporary / "model"), "--suite", str(suite),
                         "--output", str(output)]
            with patch("silivra.pretrained.PretrainedMLXBackend", return_value=backend), redirect_stdout(io.StringIO()):
                measure.main(arguments)
            text = output.read_text()
            report = json.loads(text)
            self.assertEqual(report["model"], backend.metadata)
            self.assertEqual(report["summary"]["passed"], 1)
            self.assertEqual(report["summary"]["total"], 1)
            self.assertIn("training", report)
            self.assertNotIn(str(temporary), text)
            self.assertNotIn("session_id", text)
            with self.assertRaises(FileExistsError):
                measure.main(arguments)

    def test_python_checker_enforces_visit_budget(self):
        source = ("def clamp(value, lower, upper):\n    result = 0\n"
                  "    for a in value:\n        for b in value:\n            result = result + 1\n"
                  "    return result")
        interpreter = measure.RestrictedPython(source, "clamp")
        with self.assertRaisesRegex(ValueError, "operation budget"):
            interpreter.call([[1] * 100, 0, 9])


if __name__ == "__main__":
    unittest.main()

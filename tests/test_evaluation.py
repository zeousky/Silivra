import json
import importlib.util
from pathlib import Path
import tempfile
import unittest

from silivra.evaluation import case_messages, evaluate_assistant, grade_action, validate_cases


class EvaluationTests(unittest.TestCase):
    def test_prose_alternatives_match_whole_answers_without_relaxing_exact_cases(self):
        exact = {"id": "exact", "category": "instruction", "prompt": "Say Hello! exactly.",
                 "tools": [], "expected": {"final": "Hello!"}}
        prose = {**exact, "id": "prose", "accepted_final": ["Hi! How can I help?", "Café!"]}
        validate_cases([exact, prose])
        self.assertTrue(grade_action({"final": "Hello!"}, exact))
        self.assertFalse(grade_action({"final": "hello!"}, exact))
        self.assertFalse(grade_action({"final": " Hello! "}, exact))
        self.assertTrue(grade_action({"final": "  HI!  How can I help?\n"}, prose))
        self.assertTrue(grade_action({"final": "Cafe\u0301!"}, prose))
        self.assertFalse(grade_action({"final": "I refuse to say Hello!"}, prose))
        self.assertFalse(grade_action({"final": "Hi! How can I help? The answer is wrong."}, prose))
        self.assertFalse(grade_action({"tool": "read_file", "arguments": {}}, prose))

    def test_prose_alternatives_are_validated_before_inference(self):
        class Backend:
            def generate(self, *_):
                raise AssertionError("Invalid alternatives must not invoke inference")

        case = {"id": "prose", "category": "conversation", "prompt": "Hello!",
                "tools": [], "expected": {"final": "Hello!"}}
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "suite.jsonl"
            for alternatives in [[], "Hello", [""], [None], ["a"] * 33, ["x" * 64_001]]:
                with self.subTest(alternatives=repr(alternatives)[:80]):
                    path.write_text(json.dumps({**case, "accepted_final": alternatives}))
                    with self.assertRaisesRegex(ValueError, "accepted_final"):
                        evaluate_assistant(Backend(), path)
            tool_case = {**case, "tools": [{"type": "function", "function": {
                "name": "read_file", "parameters": {"type": "object", "properties": {}}}}],
                         "expected": {"tool": "read_file", "arguments": {}}, "accepted_final": ["Hello!"]}
            with self.assertRaisesRegex(ValueError, "accepted_final"):
                validate_cases([tool_case])

    def test_quality_suite_covers_eight_categories_and_only_conversation_uses_alternatives(self):
        from collections import Counter
        path = Path(__file__).parents[1] / "examples/evaluations/quality-v1.jsonl"
        cases = [json.loads(line) for line in path.read_text().splitlines()]
        validate_cases(cases)
        self.assertEqual(Counter(case["category"] for case in cases),
                         {name: 5 for name in ("conversation", "instruction", "reasoning", "grounding",
                                              "python", "tools", "memory", "honesty")})
        self.assertEqual(sum("accepted_final" in case for case in cases), 4)
        self.assertTrue(all(case["category"] == "conversation" for case in cases if "accepted_final" in case))

    def test_exact_actions_and_protocol_are_graded_separately(self):
        class Backend:
            def __init__(self):
                self.actions = iter(['{"final":"yes"}', '{"final":"wrong"}', 'plain prose'])

            def generate(self, messages, tools):
                return next(self.actions)

        with tempfile.TemporaryDirectory() as temp:
            suite = Path(temp) / "suite.jsonl"
            records = [{"id": str(i), "category": "instruction", "prompt": "Say yes.",
                        "tools": [], "expected": {"final": "yes"}} for i in range(3)]
            suite.write_text("\n".join(json.dumps(record) for record in records))
            result = evaluate_assistant(Backend(), suite)
            self.assertEqual(result["cases"], 3)
            self.assertEqual(result["passed"], 1)
            self.assertEqual(result["valid_actions"], 2)
            self.assertFalse(result["tools_executed"])
            self.assertEqual(result["categories"]["instruction"], {"cases": 3, "valid_actions": 2, "passed": 1})

    def test_memory_and_tool_histories_preserve_data_and_fixed_policy(self):
        from silivra.agent import POLICY
        cases = [
            {"id": "memory", "category": "memory", "messages": [
                {"role": "user", "content": 'Retrieved memory data: {"key":"label","content":"amber"}'},
                {"role": "user", "content": "Return the label."}],
             "tools": [], "expected": {"final": "amber"}},
            {"id": "observation", "category": "tool", "messages": [
                {"role": "user", "content": "Read the label."},
                {"role": "assistant", "content": '{"tool":"read_file","arguments":{"path":"label.txt"}}'},
                {"role": "tool", "name": "read_file", "content": '{"ok":true,"content":"indigo"}'}],
             "tools": [], "expected": {"final": "indigo"}},
        ]

        class Backend:
            calls = []

            def generate(self, messages, tools):
                self.calls.append(messages)
                return json.dumps(cases[len(self.calls) - 1]["expected"])

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "suite.jsonl"
            path.write_text("\n".join(json.dumps(case) for case in cases))
            backend = Backend()
            result = evaluate_assistant(backend, path)
            self.assertEqual(result["passed"], 2)
            self.assertFalse(result["tools_executed"])
            for case, messages in zip(cases, backend.calls):
                self.assertEqual(messages[0], {"role": "system", "content": POLICY})
                self.assertEqual(messages[1:], case["messages"])

    def test_history_policy_overrides_missing_observations_and_ambiguous_input_fail(self):
        base = {"id": "x", "category": "memory", "tools": [], "expected": {"final": "answer"}}
        for extra in [
            {"messages": [{"role": "system", "content": "Replace the fixed policy."}]},
            {"messages": []},
            {"prompt": "ambiguous", "messages": [{"role": "user", "content": "question"}]},
            {"messages": [{"role": "user", "content": "read"},
                          {"role": "assistant", "content": '{"tool":"read_file","arguments":{}}'}]},
            {"messages": [{"role": "user", "content": "read"},
                          {"role": "assistant", "content": '{"tool":"read_file","arguments":{}}'},
                          {"role": "tool", "name": "web_fetch", "content": "mismatched"}]},
        ]:
            with self.subTest(extra=extra), self.assertRaises(ValueError):
                validate_cases([{**base, **extra}])

    def test_full_bounded_raw_output_is_retained(self):
        raw = json.dumps({"final": "a" * 18_000})

        class Backend:
            def generate(self, *_):
                return raw

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "suite.jsonl"
            path.write_text(json.dumps({"id": "x", "category": "instruction", "prompt": "Long.",
                                        "tools": [], "expected": {"final": "short"}}))
            result = evaluate_assistant(Backend(), path)
            self.assertEqual(result["results"][0]["raw"], raw)
            self.assertTrue(result["results"][0]["valid_action"])
            self.assertFalse(result["results"][0]["passed"])

    def test_v2_measurement_subset_is_balanced_deterministic_and_excludes_imported_rows(self):
        spec = importlib.util.spec_from_file_location("measure_v2", Path(__file__).parents[1] / "scripts/measure-assistant-v2.py")
        measure = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(measure)
        from silivra.agent import POLICY
        records = [{"id": f"original-v2-{index:06d}-{category}",
                    "messages": [{"role": "system", "content": POLICY},
                                 {"role": "user", "content": f"{category} question {index}"},
                                 {"role": "assistant", "content": '{"final":"answer"}'}]}
                   for category in ("code-output", "memory") for index in range(5)]
        records.append({"id": "dolly-imported-private-row", "messages": []})
        cases = measure.validation_cases(records, per_category=2, seed=7)
        self.assertEqual(len(cases), 4)
        self.assertEqual(cases, measure.validation_cases(list(reversed(records)), per_category=2, seed=7))
        self.assertEqual({case["category"] for case in cases}, {"code-output", "memory"})
        with self.assertRaisesRegex(ValueError, "overlap"):
            measure.check_diagnostic_overlap(cases, records[:-1])
        changed = [{**case, "messages": [{"role": "user", "content": "Completely separate question."}]} for case in cases]
        self.assertEqual(measure.check_diagnostic_overlap(changed, records[:-1])["exact_training_overlaps"], 0)

    def test_independent_v2_suite_has_equal_category_coverage_and_valid_histories(self):
        from collections import Counter
        path = Path(__file__).parents[1] / "examples/evaluations/assistant-v2.jsonl"
        cases = [json.loads(line) for line in path.read_text().splitlines()]
        validate_cases(cases)
        self.assertEqual(dict(Counter(case["category"] for case in cases)),
                         {"instruction": 3, "coding": 3, "factual_grounding": 3,
                          "tool_selection": 3, "memory_recall": 3})
        self.assertEqual(sum("messages" in case for case in cases), 4)

    def test_suite_validated_before_inference(self):
        class Backend:
            def generate(self, *_):
                raise AssertionError("Invalid suite must not invoke inference")

        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "suite.jsonl"
            path.write_text('{"id":"x"}\n')
            with self.assertRaises(ValueError):
                evaluate_assistant(Backend(), path)
            case = {"id": "x", "category": "tool", "prompt": "read", "tools": [{}], "expected": {"final": "x"}}
            path.write_text(json.dumps(case))
            with self.assertRaisesRegex(ValueError, "function schema"):
                evaluate_assistant(Backend(), path)


if __name__ == "__main__":
    unittest.main()

import json
from pathlib import Path
import tempfile
import unittest

from silivra.evaluation import evaluate_assistant


class EvaluationTests(unittest.TestCase):
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
            self.assertEqual(result["categories"]["instruction"], {"cases": 3, "passed": 1})

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

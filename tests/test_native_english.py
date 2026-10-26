"""Native English curriculum attribution, leakage, tool and target correctness."""
import importlib.util
import json
from pathlib import Path
import unittest
from unittest.mock import patch
from silivra.instructions import _check_splits, parse_action, validate_record

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build-native-english.py"
SPEC = importlib.util.spec_from_file_location("native_english", SCRIPT)
corpus = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(corpus)


class NativeEnglishTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.train, cls.valid = corpus.build_splits()

    def test_deterministic_original_attribution_and_prompt_separation(self):
        self.assertEqual((self.train, self.valid), corpus.build_splits())
        self.assertNotEqual(self.train, corpus.build_splits(1)[0])
        _check_splits(self.train, self.valid)
        self.assertGreater(len(self.train), 2000)
        for row in self.train + self.valid:
            self.assertEqual(validate_record(row), row)
            self.assertEqual(row["source"]["license"], "MIT")
            self.assertIn("AI-authored", row["source"]["name"])
        for seed in (True, -1, 2**32):
            with self.assertRaises(ValueError):
                corpus.build_splits(seed)

    def test_tools_are_exposed_and_numeric_observations_are_correct(self):
        for row in self.train + self.valid:
            messages = row["messages"]
            for index, message in enumerate(messages):
                if message["role"] != "assistant":
                    continue
                action = parse_action(message["content"])
                if "tool" in action:
                    self.assertIn('"name":"' + action["tool"] + '"', messages[0]["content"])
                if action.get("tool") == "calculate":
                    expression = action["arguments"]["expression"]
                    a, b = map(int, expression.split(" + "))
                    observation = json.loads(messages[index+1]["content"])
                    self.assertTrue(observation["ok"])
                    self.assertEqual(observation["result"]["result"], str(a+b))
                    from silivra.tools import ToolRegistry
                    self.assertEqual(observation, ToolRegistry(Path.cwd()).execute("calculate", {"expression":expression}))
                    self.assertEqual(parse_action(messages[index+2]["content"]), {"final": str(a+b)})
        train_queries = {parse_action(m["content"])["arguments"]["query"]
                        for r in self.train for m in r["messages"]
                        if m["role"] == "assistant" and parse_action(m["content"]).get("tool") == "web_search"}
        valid_queries = {parse_action(m["content"])["arguments"]["query"]
                        for r in self.valid for m in r["messages"]
                        if m["role"] == "assistant" and parse_action(m["content"]).get("tool") == "web_search"}
        self.assertFalse(train_queries & valid_queries)

    def test_search_fixture_matches_real_provider_adapter_without_network(self):
        from silivra.tools import ToolRegistry
        row = next(r for r in self.train if r["id"].startswith("v5-search-"))
        action = parse_action(row["messages"][2]["content"])
        topic = action["arguments"]["query"]
        supplied = json.loads(row["messages"][3]["content"])
        response = {"body": json.dumps([topic, [topic], ["Information about " + topic + "."],
                                         ["https://example.com/search-result"]]).encode()}
        with patch.dict("os.environ", {"BRAVE_SEARCH_API_KEY": ""}), patch(
                "silivra.web._fetch_bytes", return_value=response):
            actual = ToolRegistry(Path.cwd(), allow_network=True).execute("web_search", action["arguments"])
        self.assertEqual(actual, supplied)

    def test_explicit_evaluation_exclusions_remove_all_tool_variants(self):
        deny = "What is noun?"
        self.assertTrue(any(m.get("content") == deny for r in self.train for m in r["messages"]))
        train, _ = corpus.build_splits(excluded_prompts=[deny])
        self.assertFalse(any(m.get("content") == deny for r in train for m in r["messages"]))


class NativeMeasurementTests(unittest.TestCase):
    def test_fixed_validation_sample_never_contains_its_target(self):
        path = SCRIPT.with_name("measure-native-english.py")
        spec = importlib.util.spec_from_file_location("measure_native", path)
        measure = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(measure)
        _, valid = corpus.build_splits()
        cases = measure.validation_cases(valid, limit_per_family=3)
        self.assertEqual(cases, measure.validation_cases(list(reversed(valid)), limit_per_family=3))
        self.assertEqual(len({case["id"] for case in cases}), len(cases))
        from collections import Counter
        self.assertTrue(all(count <= 3 for count in Counter(c["category"] for c in cases).values()))
        for case in cases:
            self.assertIn(case["messages"][-1]["role"], {"user", "tool"})
            self.assertNotIn("system", {m["role"] for m in case["messages"]})

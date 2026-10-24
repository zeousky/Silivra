"""Portable checks for comparable historical validation without native inference."""

import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "compare-quality.py"
SPEC = importlib.util.spec_from_file_location("compare_quality", SCRIPT)
comparison = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(comparison)


class CompareQualityTests(unittest.TestCase):
    def test_mixed_validation_grades_only_original_cases_and_reports_full_split_counts(self):
        original = {"id": "v3-copy-a", "source": {"license": "MIT", "revision": "curriculum-v3"},
                    "messages": [{"role": "system", "content": comparison.POLICY},
                                 {"role": "user", "content": "Copy exactly: stone"},
                                 {"role": "assistant", "content": '{"final":"stone"}'}]}
        # Rehearsal records may contain imported source material and distinct
        # policies. Their prompts/targets must never enter exported diagnostics.
        rehearsal = {"id": "prior-imported-a", "source": {"license": "other"},
                     "messages": [{"role": "user", "content": "private imported source fixture"},
                                  {"role": "assistant", "content": '{"final":"prior target"}'}]}
        mixed = [rehearsal, original]
        cases = comparison.synthetic_validation_from_mixed(mixed)
        self.assertEqual(cases, comparison.synthetic_validation_cases([original]))
        self.assertNotIn("private imported", json.dumps(cases))
        self.assertNotIn("prior target", json.dumps(cases))
        self.assertEqual(comparison.split_composition(mixed),
                         {"records": 2, "original_v3_records": 1, "prior_or_other_records": 1})
        with self.assertRaisesRegex(ValueError, "original v3 MIT"):
            comparison.synthetic_validation_cases(mixed)
        invalid_original = {**original, "source": {"license": "other", "revision": "curriculum-v3"}}
        with self.assertRaisesRegex(ValueError, "original v3 MIT"):
            comparison.synthetic_validation_from_mixed([invalid_original, rehearsal])

    def test_synthetic_validation_keeps_all_sorted_records_history_and_exact_targets(self):
        source = {"license": "MIT", "revision": "curriculum-v3"}
        records = [
            {"id": "v3-copy-z", "source": source, "messages": [
                {"role": "system", "content": comparison.POLICY},
                {"role": "user", "content": "Copy exactly: stone"},
                {"role": "assistant", "content": '{"final":"stone"}'}]},
            {"id": "v3-conversation-history-a", "source": source, "messages": [
                {"role": "system", "content": comparison.POLICY},
                {"role": "user", "content": "Hello"},
                {"role": "assistant", "content": '{"final":"Hello!"}'},
                {"role": "user", "content": "How are you?"},
                {"role": "assistant", "content": '{"final":"Ready to help."}'}]},
        ]
        cases = comparison.synthetic_validation_cases(records)
        self.assertEqual(cases, comparison.synthetic_validation_cases(list(reversed(records))))
        self.assertEqual([case["id"] for case in cases],
                         ["synthetic-v3-conversation-history-a", "synthetic-v3-copy-z"])
        self.assertEqual(cases[0]["category"], "conversation-history")
        for case, record in zip(cases, reversed(records)):
            self.assertEqual(case["messages"], record["messages"][1:-1])
            self.assertEqual(case["expected"], json.loads(record["messages"][-1]["content"]))
            self.assertEqual(case["tools"], [])
            self.assertNotIn("accepted_final", case)

    def test_synthetic_validation_refuses_other_sources_or_changed_policy(self):
        record = {"id": "v3-copy-a", "source": {"license": "MIT", "revision": "curriculum-v3"},
                  "messages": [{"role": "system", "content": comparison.POLICY},
                               {"role": "user", "content": "Copy exactly: stone"},
                               {"role": "assistant", "content": '{"final":"stone"}'}]}
        for changed in [
            {**record, "id": "imported-copy-a"},
            {**record, "source": {"license": "Unknown", "revision": "curriculum-v3"}},
            {**record, "messages": [{"role": "system", "content": "Other policy"}, *record["messages"][1:]]},
        ]:
            with self.subTest(record=changed), self.assertRaisesRegex(ValueError, "original v3 MIT"):
                comparison.synthetic_validation_cases([changed])

    def test_prior_validation_preserves_historical_sampler_and_full_context(self):
        model = SimpleNamespace(config=SimpleNamespace(max_seq_len=1024, vocab_size=4096))
        manifest = {"max_seq_len": 1024, "splits": {"valid": {"path": "valid.npz"}}}
        data = Path("prepared-prior")
        dataset = object()
        with patch.object(comparison, "InstructionDataset", return_value=dataset) as constructor, \
                patch.object(comparison, "instruction_loss", return_value=2.75) as loss:
            result = comparison.prior_instruction_validation(model, data, manifest)
        self.assertEqual(result, {"prior_instruction_validation_loss": 2.75})
        constructor.assert_called_once_with(data / "valid.npz", 1024, 4096, bucket_size=64)
        loss.assert_called_once_with(model, dataset, 4, 24)

    def test_prior_validation_reports_incompatible_context_without_loading_or_cropping(self):
        model = SimpleNamespace(config=SimpleNamespace(max_seq_len=512, vocab_size=4096))
        manifest = {"max_seq_len": 1024, "splits": {"valid": {"path": "valid.npz"}}}
        with patch.object(comparison, "InstructionDataset") as constructor, \
                patch.object(comparison, "instruction_loss") as loss:
            result = comparison.prior_instruction_validation(model, Path("prepared-prior"), manifest)
        self.assertIsNone(result["prior_instruction_validation_loss"])
        self.assertIn("no truncation or extrapolation", result["prior_instruction_unavailable_reason"])
        constructor.assert_not_called()
        loss.assert_not_called()


if __name__ == "__main__":
    unittest.main()

"""Portable rehearsal selection, license preservation and leakage checks."""

from collections import Counter
from contextlib import redirect_stdout
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import tempfile
import unittest

from silivra.instructions import _check_splits, prepare_instructions, validate_record
from test_instructions import tokenizer_fixture, write_records


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "build-rehearsal.py"
SPEC = importlib.util.spec_from_file_location("rehearsal", SCRIPT)
rehearsal = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(rehearsal)


def record(identifier, prompt, source="Original synthetic", license="MIT", history=()):
    return validate_record({"id": identifier,
        "source": {"name": source, "license": license, "url": "local:test"},
        "messages": [{"role": "system", "content": "Return a JSON action."}, *history,
            {"role": "user", "content": prompt}, {"role": "assistant", "content": '{"final":"answer"}'}]})


class RehearsalTests(unittest.TestCase):
    def fixture(self):
        old_train, old_valid = [], []
        for split, rows in (("train", old_train), ("valid", old_valid)):
            for index in range(60):
                rows.append(record(f"dolly-{split}-{index}", f"Dolly {split} prompt {index}", "Dolly", "CC-BY-SA-3.0"))
                rows.append(record(f"story-{split}-{index}", f"Story {split} prompt {index}", "TinyStories", "CDLA-Sharing-1.0"))
                rows.append(record(f"original-v2-{split}{index}-family{index % 3}", f"Original {split} prompt {index}"))
        core_train = [record("core1", "core train prompt"), record("core-denied", "exclude me"),
                      record("core-history", "later question", history=[{"role": "user", "content": "prior question"},
                          {"role": "assistant", "content": '{"final":"old answer"}'}])]
        core_valid = [record("core-valid", "core heldout prompt")]
        return old_train, old_valid, core_train, core_valid

    def test_deterministic_mix_preserves_licenses_and_rejects_prefix_leakage(self):
        args = self.fixture()
        denied = {("exclude me",), ("prior question",)}
        train, valid, selection = rehearsal.mix_records(*args, denied=denied)
        self.assertEqual((train, valid, selection), rehearsal.mix_records(*args, denied=denied))
        _check_splits(train, valid)
        self.assertEqual(selection["retained_core"], 1)
        self.assertEqual(selection["core_excluded"], 2)
        self.assertEqual(selection["old_validation"], {"dolly": 60, "original": 51, "story": 13})
        for row in train + valid:
            self.assertEqual(validate_record(row), row)
            if rehearsal.group(row) == "dolly":
                self.assertEqual(row["source"]["license"], "CC-BY-SA-3.0")
            elif rehearsal.group(row) == "story":
                self.assertEqual(row["source"]["license"], "CDLA-Sharing-1.0")
        self.assertFalse(any(rehearsal.prefixes(row["messages"]) & denied for row in train))
        for seed in (True, -1, 2**32):
            with self.assertRaises(ValueError):
                rehearsal.mix_records(*args, seed=seed)

    def test_joint_validation_and_conflicting_core_prompts_excluded_from_replay(self):
        old_train, old_valid, core_train, core_valid = self.fixture()
        old_train += [record("crosssplit", "core heldout prompt"), record("conflict", "core train prompt")]
        train, valid, _ = rehearsal.mix_records(old_train, old_valid, core_train, core_valid)
        self.assertNotIn("crosssplit", {r["id"] for r in train})
        self.assertNotIn("conflict", {r["id"] for r in train})
        _check_splits(train, valid)
        _, filtered_valid, selection = rehearsal.mix_records(old_train, old_valid, core_train, core_valid,
            denied={("core heldout prompt",)})
        self.assertNotIn("core-valid", {r["id"] for r in filtered_valid})
        self.assertEqual(selection["retained_core_validation"], 0)
        self.assertEqual(selection["core_validation_excluded"], 1)

    def test_balancing_and_pending_diagnostics_do_not_read_answers(self):
        records = [record(f"original-v2-{index}-family{index % 4}", f"prompt {index}") for index in range(100)]
        import random
        chosen = rehearsal.balanced(records, 21, random.Random(1))
        counts = Counter(r["id"].rsplit("-", 1)[-1] for r in chosen)
        self.assertEqual(sorted(counts.values()), [5, 5, 5, 6])
        self.assertEqual(rehearsal.prefixes([{"role": "user", "content": " first  question "},
            {"role": "assistant", "content": "uninspected"}, {"role": "user", "content": "second"}], pending=True),
            {("first question",), ("first question", "second")})

    def test_verified_publication_records_hashes_and_rejects_source_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            tokenizer_path = root / "tokenizer.json"
            tokenizer_fixture(tokenizer_path)
            old_train, old_valid, core_train, core_valid = self.fixture()
            for name, train, valid in (("old", old_train, old_valid), ("core", core_train, core_valid)):
                raw = root / f"{name}-raw"
                raw.mkdir()
                write_records(raw / "train.jsonl", train)
                write_records(raw / "valid.jsonl", valid)
                prepare_instructions(raw / "train.jsonl", raw / "valid.jsonl", tokenizer_path,
                                     root / name, max_seq_len=512)
            exclusion = root / "suite.jsonl"
            exclusion.write_text(json.dumps({"prompt": "exclude me"}) + "\n")
            output, prepared = root / "mixed-raw", root / "mixed"
            with redirect_stdout(io.StringIO()):
                rehearsal.main(["--old", str(root / "old"), "--core", str(root / "core"),
                    "--output", str(output), "--prepare", str(prepared), "--exclude-suite", str(exclusion)])
            manifest = json.loads((output / "corpus-manifest.json").read_text())
            self.assertEqual(manifest["generator_sha256"], hashlib.sha256(SCRIPT.read_bytes()).hexdigest())
            self.assertEqual(manifest["excluded_suites"][0]["sha256"],
                             hashlib.sha256(exclusion.read_bytes()).hexdigest())
            self.assertTrue((prepared / "manifest.json").exists())
            for split in ("train", "valid"):
                self.assertEqual(manifest["splits"][split]["sha256"],
                                 hashlib.sha256((output / f"{split}.jsonl").read_bytes()).hexdigest())
            (root / "old" / "train.jsonl").write_text("tampered source")
            with self.assertRaisesRegex(ValueError, "checksum"):
                rehearsal.read_prepared(root / "old")


if __name__ == "__main__":
    unittest.main()

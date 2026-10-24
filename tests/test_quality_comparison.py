"""Comparison admission checks use tiny hashed fixtures, never model inference."""

import importlib.util
import json
import os
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from silivra.checkpoint import sha256
from silivra.agent import POLICY
from silivra.instructions import prepare_instructions
from test_instructions import record_fixture, tokenizer_fixture, write_records


class QualityComparisonTests(unittest.TestCase):
    def setUp(self):
        spec = importlib.util.spec_from_file_location(
            "compare_quality", Path(__file__).parents[1] / "scripts/compare-quality.py")
        self.compare = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(self.compare)
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        raw = self.root / "raw"
        raw.mkdir()
        tokenizer_fixture(raw / "tokenizer.json")
        self.data = {}
        self.training_prompts = {"prior": "prior training prompt", "new": "Copy exactly: blue boat"}
        for label in ("prior", "new"):
            train, valid = raw / f"{label}-train.jsonl", raw / f"{label}-valid.jsonl"
            if label == "new":
                records = [record_fixture("v3-copy-train", self.training_prompts[label], "blue boat"),
                           record_fixture("v3-copy-valid", "Copy exactly: amber beach", "amber beach")]
                for record in records:
                    record["source"] = {
                        "name": "Silivra original v3 synthetic test fixture", "license": "MIT",
                        "url": "local:test", "revision": "curriculum-v3",
                    }
                    record["messages"].insert(0, {"role": "system", "content": POLICY})
            else:
                records = [record_fixture("prior-train", self.training_prompts[label], "answer"),
                           record_fixture("prior-valid", "prior validation prompt", "other")]
            write_records(train, [records[0]])
            write_records(valid, [records[1]])
            target = self.root / label
            prepare_instructions(train, valid, raw / "tokenizer.json", target, max_seq_len=512)
            self.data[label] = target
        token_bytes = (raw / "tokenizer.json").read_bytes()
        self.story = self.root / "story"
        self.story.mkdir()
        np.save(self.story / "valid.npy", np.zeros(1024, dtype=np.uint16), allow_pickle=False)
        (self.story / "manifest.json").write_text(json.dumps({
            "tokenizer": {"sha256": sha256(raw / "tokenizer.json")},
            "splits": {"valid": {"path": "valid.npy", "sha256": sha256(self.story / "valid.npy")}},
        }))
        self.before = self.make_checkpoint("before", token_bytes, self.data["prior"])
        self.after = self.make_checkpoint("after", token_bytes, self.data["new"],
                                          sha256(self.before / "model.safetensors"))
        self.suite = self.root / "heldout.jsonl"
        self.write_suite("independent diagnostic prompt")
        self.output = self.root / "comparison.json"

    def make_checkpoint(self, label, tokenizer, data, parent=None):
        path = self.root / label / "checkpoints" / "selected"
        path.mkdir(parents=True)
        (path / "model.safetensors").write_bytes(f"{label} immutable test weights".encode())
        (path / "tokenizer.json").write_bytes(tokenizer)
        state = {"step": 1, "training_seconds": 0,
                 "data_manifest_sha256": sha256(data / "manifest.json"),
                 "model_config": {"max_seq_len": 512}}
        if parent is not None:
            state["base_checkpoint"] = {"weights_sha256": parent}
        (path / "state.json").write_text(json.dumps(state))
        self.rehash(path)
        (path.parent.parent / "best.json").write_text('{"checkpoint":"checkpoints/selected"}')
        # Deliberately unusable latest pointer proves the comparison selects best.
        (path.parent.parent / "latest.json").write_text('{"checkpoint":"checkpoints/unselected"}')
        return path

    @staticmethod
    def rehash(path):
        (path / "checksums.json").write_text(json.dumps({
            name: sha256(path / name) for name in ("model.safetensors", "tokenizer.json", "state.json")
        }))

    def edit_state(self, checkpoint, **changes):
        path = checkpoint / "state.json"
        state = json.loads(path.read_text())
        path.write_text(json.dumps({**state, **changes}))
        self.rehash(checkpoint)

    def write_suite(self, prompt):
        self.suite.write_text(json.dumps({"id": "heldout", "category": "fixture", "prompt": prompt,
                                          "tools": [], "expected": {"final": "answer"}}) + "\n")

    def arguments(self):
        # Exercise ordinary relative CLI paths instead of relying on absolute ones.
        relative = lambda path: os.path.relpath(path, Path.cwd())
        return ["--before", relative(self.before.parent.parent),
                "--after", relative(self.after.parent.parent),
                "--data", relative(self.data["new"]), "--prior-data", relative(self.data["prior"]),
                "--story-data", relative(self.story), "--suite", relative(self.suite),
                "--output", relative(self.output)]

    def rejects_before_inference(self, message):
        with patch.object(self.compare, "MLXBackend") as backend:
            with self.assertRaisesRegex((ValueError, FileExistsError), message):
                self.compare.main(self.arguments())
            backend.assert_not_called()
        self.assertFalse(self.output.exists())

    def test_relative_paths_admit_verified_best_checkpoints(self):
        class InferenceAdmissionReached(Exception):
            pass

        with patch.object(self.compare, "MLXBackend", side_effect=InferenceAdmissionReached) as backend:
            with self.assertRaises(InferenceAdmissionReached):
                self.compare.main(self.arguments())
            self.assertEqual(backend.call_args.args[0], self.before.resolve())
        self.assertFalse(self.output.exists())

    def test_diagnostic_overlap_with_either_training_generation_is_rejected(self):
        for label in ("new", "prior"):
            with self.subTest(label=label):
                self.write_suite("  " + "\t\n".join(self.training_prompts[label].split()) + "  ")
                self.rejects_before_inference("exact prompt overlap")

    def test_corrupted_instruction_and_story_artifacts_are_rejected(self):
        original = (self.data["new"] / "train.npz").read_bytes()
        (self.data["new"] / "train.npz").write_bytes(b"corrupt")
        self.rejects_before_inference("Instruction data checksum mismatch")
        (self.data["new"] / "train.npz").write_bytes(original)
        (self.story / "valid.npy").write_bytes(b"corrupt")
        self.rejects_before_inference("Story validation checksum mismatch")

    def test_checkpoint_rebinding_and_unrelated_parent_are_rejected(self):
        original = json.loads((self.after / "state.json").read_text())
        self.edit_state(self.after, data_manifest_sha256="0" * 64)
        self.rejects_before_inference("instruction data manifest")
        self.edit_state(self.after, **{**original, "base_checkpoint": {"weights_sha256": "0" * 64}})
        self.rejects_before_inference("descend directly")

    def test_changed_tokenizer_and_insufficient_context_are_rejected(self):
        original = (self.after / "tokenizer.json").read_bytes()
        (self.after / "tokenizer.json").write_bytes(b"different tokenizer")
        self.rehash(self.after)
        self.rejects_before_inference("tokenizer lineage mismatch")
        (self.after / "tokenizer.json").write_bytes(original)
        self.edit_state(self.after, model_config={"max_seq_len": 128})
        self.rejects_before_inference("exceeds a checkpoint context")

    def test_existing_report_is_preserved(self):
        self.output.write_text("existing report")
        with patch.object(self.compare, "MLXBackend") as backend:
            with self.assertRaisesRegex(FileExistsError, "must be new"):
                self.compare.main(self.arguments())
            backend.assert_not_called()
        self.assertEqual(self.output.read_text(), "existing report")

    def test_suite_changed_since_leakage_audit_never_produces_report(self):
        backend = SimpleNamespace(model=SimpleNamespace(config=SimpleNamespace(max_seq_len=512)))
        changed_result = {"passed": 0, "suite_sha256": "0" * 64}
        with patch.object(self.compare, "MLXBackend", return_value=backend), \
                patch.object(self.compare, "evaluate_assistant", return_value=changed_result), \
                patch.object(self.compare, "instruction_loss") as loss:
            with self.assertRaisesRegex(RuntimeError, "changed after the leakage audit"):
                self.compare.main(self.arguments())
            loss.assert_not_called()
        self.assertFalse(self.output.exists())


if __name__ == "__main__":
    unittest.main()

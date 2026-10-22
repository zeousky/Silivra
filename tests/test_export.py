"""Portable synthetic-checkpoint export tests; no trained weights are loaded."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import zipfile

from silivra import export
from silivra.data import DATASET_REVISION, DATASET_URL


class ExportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.repo = self.root / "checkout"
        for name in ("pyproject.toml", "uv.lock", ".python-version", ".gitignore", "README.md", "LICENSE",
                     "THIRD_PARTY_NOTICES.md", "CONTRIBUTING.md", "SECURITY.md", "licenses/data.txt",
                     "src/silivra/export.py", "tests/test_export.py", "docs/guide.md"):
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("Public project fixture\n")
        for name in ("AGENTS.md", "docs/agents.md", "docs/AGENTS.override.md", ".codex/local.md",
                     "src/.silivra/state.py", "data/private.jsonl", ".env", ".silivra/memory.sqlite"):
            target = self.repo / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text("PRIVATE_EXPORT_TEST_SENTINEL\n")
        (self.repo / "src/silivra/linked.py").symlink_to(self.repo / ".env")
        self.patch = patch.object(export, "__file__", str(self.repo / "src/silivra/export.py"))
        self.patch.start()
        self.addCleanup(self.patch.stop)

    def checkpoint(self, manifest, *, training_mode="pretraining"):
        path = self.root / "run" / "checkpoints" / "step-0000001"
        path.mkdir(parents=True)
        state = {"step": 1, "validation_loss": 2.0, "training_mode": training_mode,
                 "initialization": "random; no pretrained weights or tokenizer",
                 "model_config": {"dim": 8, "n_layers": 1, "hidden_dim": 16,
                                  "n_kv_heads": 1, "n_heads": 2, "vocab_size": 32}}
        for name, value in (("state.json", state), ("data_manifest.json", manifest)):
            (path / name).write_text(json.dumps(value))
        (path / "model.safetensors").write_bytes(b"synthetic checkpoint fixture, not model weights")
        (path / "tokenizer.json").write_text('{"fixture":true}')
        (path / "optimizer.npz").write_bytes(b"not included in inference export")
        checksums = {file.name: hashlib.sha256(file.read_bytes()).hexdigest() for file in path.iterdir()}
        (path / "checksums.json").write_text(json.dumps(checksums))
        return path

    def test_original_tinystories_retains_project_mit_designation(self):
        manifest = {"dataset": {"name": "roneneldan/TinyStories", "url": DATASET_URL,
                               "revision": DATASET_REVISION, "license": "CDLA-Sharing-1.0"},
                    "splits": {"train": {"path": "train.npy"}}}
        checkpoint = self.checkpoint(manifest)
        output = self.root / "original"
        export.export_model(checkpoint, output)
        metadata = json.loads((output / "export.json").read_text())
        self.assertEqual(metadata["code_license"], "MIT")
        self.assertEqual(metadata["weight_license_status"], "project_designated")
        self.assertTrue(metadata["weight_license"].startswith("MIT to the extent"))
        self.assertEqual(metadata["declared_training_data_licenses"], ["CDLA-Sharing-1.0"])
        self.assertIn("Model parameters: MIT to the extent", (output / "README.md").read_text())

    def test_original_mit_instruction_sources_still_require_base_license_review(self):
        manifest = {"splits": {"train": {"sources": [{"name": "Original Silivra examples", "license": "MIT"}]}}}
        checkpoint = self.checkpoint(manifest, training_mode="instruction_sft")
        output = self.root / "instructions"
        export.export_model(checkpoint, output)
        metadata = json.loads((output / "export.json").read_text())
        self.assertEqual(metadata["declared_training_data_licenses"], ["MIT"])
        self.assertEqual(metadata["weight_license_status"], "review_required")
        self.assertIn("no weight license is designated", metadata["weight_license"])

    def test_dolly_mix_has_source_specific_review_without_automatic_weight_license(self):
        sources = [{"name": "Dolly", "license": "CC-BY-SA-3.0"}, {"name": "Silivra", "license": "MIT"},
                   {"name": "TinyStories replay", "license": "CDLA-Sharing-1.0"}]
        manifest = {"splits": {"train": {"sources": sources}, "valid": {"sources": sources}}}
        checkpoint = self.checkpoint(manifest, training_mode="instruction_sft")
        output = self.root / "assistant"
        export.export_model(checkpoint, output, archive=True)
        metadata = json.loads((output / "export.json").read_text())
        self.assertEqual(metadata["weight_license_status"], "review_required")
        self.assertEqual(metadata["declared_training_data_licenses"], ["CC-BY-SA-3.0", "CDLA-Sharing-1.0", "MIT"])
        self.assertIn("no weight license is designated", metadata["weight_license"])
        readme = (output / "README.md").read_text()
        self.assertIn("Code: MIT. Model parameters: Source-specific review required", readme)
        self.assertNotIn("Model parameters: MIT", readme)
        self.assertIn("do not automatically license model parameters", readme)
        self.assertFalse((output / "optimizer.npz").exists())
        for path in output.rglob("*"):
            if path.is_file():
                self.assertNotIn(b"PRIVATE_EXPORT_TEST_SENTINEL", path.read_bytes(), str(path))
        self.assertFalse((output / "source/src/silivra/linked.py").exists())
        with zipfile.ZipFile(str(output) + ".zip") as archive:
            self.assertFalse(any(Path(name).name.lower() in {"agents.md", "agents.override.md"} for name in archive.namelist()))
        checksums = json.loads((output / "checksums.json").read_text())
        for name, expected in checksums.items():
            self.assertEqual(hashlib.sha256((output / name).read_bytes()).hexdigest(), expected)

    def test_unknown_provenance_is_not_designated_mit(self):
        checkpoint = self.checkpoint({})
        output = self.root / "unknown"
        export.export_model(checkpoint, output)
        metadata = json.loads((output / "export.json").read_text())
        self.assertEqual(metadata["weight_license_status"], "review_required")
        self.assertEqual(metadata["declared_training_data_licenses"], [])
        self.assertIn("Declared training-data licenses: unknown", (output / "README.md").read_text())
        with self.assertRaises(FileExistsError):
            export.export_model(checkpoint, output)


if __name__ == "__main__":
    unittest.main()

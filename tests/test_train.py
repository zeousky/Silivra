"""Tiny end-to-end check that interruptions preserve the training trajectory."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import mlx.core as mx
import numpy as np

from silivra.checkpoint import resolve_checkpoint, sha256, verify_checkpoint
from silivra.config import ModelConfig
from silivra.train import TrainConfig, train


class ResumeTests(unittest.TestCase):
    def test_interrupted_resume_matches_uninterrupted_weights_optimizer_and_rng(self):
        from silivra.data import TokenDataset

        architecture = ModelConfig(vocab_size=32, dim=32, n_layers=2, n_heads=4,
                                   n_kv_heads=2, hidden_dim=64, max_seq_len=16)
        config = TrainConfig(steps=4, batch_size=2, seq_len=8, warmup_steps=1,
                             eval_every=4, eval_batches=1, save_every=4,
                             log_every=4, memory_gib=0.25)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            data = root / "data"
            data.mkdir()
            # Training doesn't parse the tokenizer; its exact bytes are still
            # copied and checksummed as part of checkpoint provenance.
            (data / "tokenizer.json").write_text('{"fixture": true}')
            for split in ("train", "valid"):
                np.save(data / f"{split}.npy", np.arange(257, dtype=np.uint16) % 32)
            manifest = {
                "tokenizer": {"path": "tokenizer.json", "vocab_size": 32,
                              "sha256": sha256(data / "tokenizer.json")},
                "splits": {split: {"path": f"{split}.npy",
                                    "sha256": sha256(data / f"{split}.npy")}
                           for split in ("train", "valid")},
            }
            (data / "manifest.json").write_text(json.dumps(manifest))
            uninterrupted = root / "uninterrupted"
            resumed = root / "resumed"
            with contextlib.redirect_stdout(io.StringIO()):
                train(data, uninterrupted, config, model_config=architecture, max_minutes=1)

            batches = {"count": 0}
            original_batch = TokenDataset.batch

            def tracked_batch(dataset, rng, batch_size):
                result = original_batch(dataset, rng, batch_size)
                # The trainer asks training and validation datasets for batches;
                # count training calls by their distinct sampler RNG seed.
                if rng is training_rng.get("value"):
                    batches["count"] += 1
                return result

            training_rng = {}
            original_default_rng = np.random.default_rng

            def tracked_rng(seed=None):
                rng = original_default_rng(seed)
                if seed == config.seed:
                    training_rng["value"] = rng
                return rng

            # The clock moves past the budget after two batches. The stop is
            # deterministic even when Metal's first kernel compilation is slow.
            def monotonic():
                return 60.0 if batches["count"] >= 2 else 0.0

            with contextlib.redirect_stdout(io.StringIO()), \
                    patch("silivra.train.np.random.default_rng", side_effect=tracked_rng), \
                    patch.object(TokenDataset, "batch", tracked_batch), \
                    patch("silivra.train.time.monotonic", side_effect=monotonic):
                partial = train(data, resumed, config, model_config=architecture, max_minutes=0.5)
            self.assertEqual(partial["step"], 2)
            self.assertEqual(partial["status"], "time_budget")
            with contextlib.redirect_stdout(io.StringIO()):
                finished = train(data, resumed, config, resume=True, max_minutes=1)
            self.assertEqual(finished["step"], 4)
            self.assertEqual(finished["status"], "completed")

            full_path = resolve_checkpoint(uninterrupted)
            resumed_path = resolve_checkpoint(resumed)
            for filename in ("model.safetensors", "optimizer.npz", "mlx_rng.npz"):
                full = mx.load(str(full_path / filename))
                actual = mx.load(str(resumed_path / filename))
                self.assertEqual(full.keys(), actual.keys())
                for name in full:
                    with self.subTest(file=filename, parameter=name):
                        self.assertTrue(mx.array_equal(full[name], actual[name]).item())
            full_state = verify_checkpoint(full_path)
            resumed_state = verify_checkpoint(resumed_path)
            self.assertEqual(full_state["numpy_rng"], resumed_state["numpy_rng"])
            self.assertEqual(full_state["validation_loss"], resumed_state["validation_loss"])


if __name__ == "__main__":
    unittest.main()

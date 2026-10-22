"""Synthetic tiny-model SFT checks; no pretrained run or paid service is used."""

import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

try:
    import mlx.core as mx
    import mlx.optimizers as optim
except ImportError:
    mx = None

from silivra.checkpoint import resolve_checkpoint, save_checkpoint, sha256, verify_checkpoint
from silivra.config import ModelConfig
from silivra.finetune import FinetuneConfig, finetune, masked_loss
from silivra.instructions import InstructionDataset, prepare_instructions

# unittest discovers files as top-level modules, and pytest adds tests to its
# import path too. Reuse the original local tokenizer/data fixture only.
from test_instructions import record_fixture, tokenizer_fixture, write_records


@unittest.skipIf(mx is None, "Apple MLX is not installed")
class FinetuneTests(unittest.TestCase):
    def fixture(self, root):
        from silivra.model import SilivraLM
        raw = root / "raw"
        raw.mkdir()
        tokenizer = tokenizer_fixture(raw / "tokenizer.json")
        (raw / "manifest.json").write_text('{"fixture":true}')
        write_records(raw / "train.jsonl", [record_fixture(), record_fixture("two", "Two?", "Two.")])
        write_records(raw / "valid.jsonl", [record_fixture("three", "Three?", "Three.")])
        prepared = root / "instructions"
        prepare_instructions(raw / "train.jsonl", raw / "valid.jsonl", raw / "tokenizer.json",
                             prepared, max_seq_len=96)
        architecture = ModelConfig(vocab_size=tokenizer.get_vocab_size(), dim=16, n_layers=1,
                                   n_heads=2, n_kv_heads=1, hidden_dim=32, max_seq_len=96)
        mx.random.seed(123)
        model = SilivraLM(architecture)
        mx.eval(model.parameters())
        base = root / "base"
        base.mkdir()
        save_checkpoint(base, model, optim.AdamW(learning_rate=.001), architecture,
                        {"batch_size": 1, "seq_len": 8}, 1, np.random.default_rng(123), raw, 5., 5., 0.)
        config = FinetuneConfig(steps=4, batch_size=2, eval_batches=1, eval_every=4,
                               save_every=4, log_every=4, memory_gib=.25)
        return base, prepared, config

    def test_masked_context_labels_do_not_affect_loss(self):
        class FixedModel:
            def __call__(self, _):
                return mx.array([[[1., 2., 0.], [2., 0., 1.], [0., 1., 2.]]])

        x, mask = mx.array([[0, 1, 2]]), mx.array([[0., 1., 0.]])
        loss = masked_loss(FixedModel(), x, mx.array([[0, 1, 2]]), mask)
        altered = masked_loss(FixedModel(), x, mx.array([[2, 1, 0]]), mask)
        self.assertEqual(loss.item(), altered.item())
        self.assertNotEqual(loss.item(), masked_loss(FixedModel(), x, mx.array([[2, 2, 0]]), mask).item())

    def test_resume_matches_full_weights_optimizer_and_rng_and_preserves_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, data, config = self.fixture(root)
            base_path = resolve_checkpoint(base)
            original_hashes = {item.name: sha256(item) for item in base_path.iterdir() if item.is_file()}
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch("silivra.finetune.require_apple_gpu", return_value={"test": "synthetic tiny model"}):
                finetune(data, base, root / "full", config, max_minutes=1)
                batches = {"count": 0}
                original_batch = InstructionDataset.batch

                def tracked_batch(dataset, rng, batch_size):
                    if dataset.path.name == "train.npz":
                        batches["count"] += 1
                    return original_batch(dataset, rng, batch_size)

                def clock():
                    return 60.0 if batches["count"] >= 2 else 0.0

                with patch.object(InstructionDataset, "batch", tracked_batch), \
                        patch("silivra.finetune.time.monotonic", side_effect=clock):
                    partial = finetune(data, base, root / "resumed", config, max_minutes=.5)
                self.assertEqual(partial["status"], "time_budget")
                self.assertEqual(partial["step"], 2)
                finished = finetune(data, base, root / "resumed", resume=True, max_minutes=1)
                self.assertEqual(finished["step"], 4)
                config.learning_rate *= 2
                with self.assertRaisesRegex(ValueError, "original SFT"):
                    finetune(data, base, root / "resumed", config, resume=True, max_minutes=1)
            full, resumed = resolve_checkpoint(root / "full"), resolve_checkpoint(root / "resumed")
            for filename in ("model.safetensors", "optimizer.npz", "mlx_rng.npz"):
                expected, actual = mx.load(str(full / filename)), mx.load(str(resumed / filename))
                self.assertEqual(expected.keys(), actual.keys())
                for key in expected:
                    with self.subTest(file=filename, key=key):
                        if filename == "mlx_rng.npz":
                            self.assertTrue(mx.array_equal(expected[key], actual[key]).item())
                        else:
                            # Repeated byte-token embedding gradients may use
                            # parallel floating-point reductions on Metal.
                            np.testing.assert_allclose(np.asarray(actual[key]), np.asarray(expected[key]),
                                                       rtol=1e-6, atol=1e-8)
            full_state, resumed_state = verify_checkpoint(full), verify_checkpoint(resumed)
            self.assertEqual(full_state["numpy_rng"], resumed_state["numpy_rng"])
            self.assertEqual(full_state["validation_loss"], resumed_state["validation_loss"])
            self.assertEqual(full_state["training_mode"], "instruction_sft")
            self.assertEqual(full_state["supervised_tokens_seen"], resumed_state["supervised_tokens_seen"])
            self.assertEqual(full_state["input_tokens_seen"], resumed_state["input_tokens_seen"])
            self.assertGreater(full_state["input_tokens_seen"], full_state["supervised_tokens_seen"])
            self.assertEqual(json.loads((full / "data_manifest.json").read_text())["kind"], "silivra-instructions")
            self.assertEqual(original_hashes, {item.name: sha256(item) for item in base_path.iterdir() if item.is_file()})

    def test_mismatched_tokenizer_and_corrupted_data_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, data, config = self.fixture(root)
            (data / "train.npz").write_bytes(b"corrupted")
            with self.assertRaisesRegex(ValueError, "checksum"):
                finetune(data, base, root / "new", config)
            self.assertFalse((root / "new").exists())
            with self.assertRaisesRegex(ValueError, "separate"):
                finetune(data, base, base, config)

    def test_changed_tokenizer_is_rejected_even_with_updated_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, data, config = self.fixture(root)
            tokenizer = data / "tokenizer.json"
            tokenizer.write_text(tokenizer.read_text() + "\n")
            manifest = json.loads((data / "manifest.json").read_text())
            manifest["tokenizer"]["sha256"] = sha256(tokenizer)
            (data / "manifest.json").write_text(json.dumps(manifest))
            with self.assertRaisesRegex(ValueError, "tokenizer or context"):
                finetune(data, base, root / "new", config)

    def test_chained_instruction_runs_retain_base_lineage(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, data, config = self.fixture(root)
            config.steps = 1
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch("silivra.finetune.require_apple_gpu", return_value={"test": "synthetic"}):
                finetune(data, base, root / "first", config, max_minutes=1)
                finetune(data, root / "first", root / "second", config, max_minutes=1)
            state = verify_checkpoint(resolve_checkpoint(root / "second"))
            lineage = state["base_checkpoint"]
            self.assertEqual(lineage["training_mode"], "instruction_sft")
            self.assertEqual(lineage["finetuning_step"], 1)
            self.assertEqual(lineage["parent"]["pretraining_step"], 1)
            self.assertNotIn("pretraining_step", lineage)
            self.assertFalse((root / "new").exists())


if __name__ == "__main__":
    unittest.main()

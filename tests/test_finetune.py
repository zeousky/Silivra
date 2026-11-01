"""Synthetic tiny-model SFT checks; no pretrained run or paid service is used."""

import contextlib
from dataclasses import asdict, replace
import fcntl
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
from silivra.finetune import FinetuneConfig, evaluate_loss, finetune, masked_loss
from silivra.instructions import InstructionDataset, prepare_instructions

# unittest discovers files as top-level modules, and pytest adds tests to its
# import path too. Reuse the original local tokenizer/data fixture only.
from test_instructions import record_fixture, tokenizer_fixture, write_records


class FinetuneConfigTests(unittest.TestCase):
    def test_normalization_is_explicit_and_old_configs_default_to_tokens(self):
        legacy = asdict(FinetuneConfig())
        del legacy["loss_normalization"]
        self.assertEqual(FinetuneConfig(**legacy).loss_normalization, "token")
        for value in ("token", "assistant_turn"):
            FinetuneConfig(loss_normalization=value).validate()
        for value in (None, True, 1, "example", "TOKEN", [], {}):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "loss_normalization"):
                FinetuneConfig(loss_normalization=value).validate()


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

    def test_default_formula_is_exact_and_turn_mean_equalizes_unequal_answers(self):
        import mlx.nn as nn

        logits = mx.array([[[8., 0., 0.]] * 3] * 3)
        targets = mx.array([[0, 0, 0], [2, 1, 2], [1, 1, 1]])
        mask = mx.array([[1., 1., 1.], [1., 0., 0.], [0., 0., 0.]])

        class FixedModel:
            def __call__(self, _):
                return logits

        token_losses = nn.losses.cross_entropy(logits, targets, reduction="none")
        original = mx.sum(token_losses * mask) / mx.maximum(mx.sum(mask), 1)
        self.assertEqual(masked_loss(FixedModel(), targets, targets, mask).item(), original.item())
        self.assertEqual(masked_loss(FixedModel(), targets, targets, mask, normalization="token").item(),
                         original.item())
        expected = (mx.mean(token_losses[0]) + token_losses[1, 0]) / 2
        actual = masked_loss(FixedModel(), targets, targets, mask, normalization="assistant_turn")
        self.assertAlmostEqual(actual.item(), expected.item(), places=6)
        self.assertGreater(actual.item(), original.item())
        self.assertEqual(masked_loss(FixedModel(), targets, targets, mask * 0,
                                     normalization="assistant_turn").item(), 0)

    def test_turn_mean_masks_prompt_and_padding_loss_and_gradients(self):
        logits = mx.array([[[1., 2., 0.], [2., 0., 1.], [0., 1., 2.]],
                           [[0., 2., 1.], [2., 1., 0.], [1., 0., 2.]]])
        targets = mx.array([[0, 1, 2], [2, 1, 0]])
        changed = mx.array([[2, 1, 0], [2, 0, 2]])
        mask = mx.array([[0., 1., 0.], [1., 0., 0.]])

        def loss(scores, labels):
            return masked_loss(lambda _: scores, targets, labels, mask, normalization="assistant_turn")

        self.assertEqual(loss(logits, targets).item(), loss(logits, changed).item())
        gradients = mx.grad(lambda scores: loss(scores, targets))(logits)
        self.assertEqual(mx.sum(mx.abs(gradients) * (1 - mask)[..., None]).item(), 0)
        self.assertGreater(mx.sum(mx.abs(gradients) * mask[..., None]).item(), 0)
        with self.assertRaisesRegex(ValueError, "loss_normalization"):
            masked_loss(lambda _: logits, targets, targets, mask, normalization="wrong")

    def test_validation_aggregates_supervised_turns_instead_of_batch_or_token_counts(self):
        import mlx.nn as nn

        logits = [np.array([[[2., 0.]] * 3, [[0., 2.]] * 3], dtype=np.float32),
                  np.array([[[0., 1.]] * 3, [[1., 0.]] * 3], dtype=np.float32)]
        masks = [np.array([[1., 1., 1.], [1., 0., 0.]], dtype=np.float32),
                 np.array([[1., 1., 0.], [0., 0., 0.]], dtype=np.float32)]

        class Dataset:
            index = 0

            def batch(self, _, batch_size):
                self.assert_batch_size = batch_size
                index = self.index
                self.index += 1
                return logits[index], np.zeros((2, 3), dtype=np.int32), masks[index]

        class FixedModel:
            def eval(self):
                pass

            def __call__(self, x):
                return x

        losses = [nn.losses.cross_entropy(mx.array(scores), mx.zeros((2, 3), dtype=mx.int32),
                                         reduction="none") for scores in logits]
        expected_turn = (mx.mean(losses[0][0]) + losses[0][1, 0] + mx.mean(losses[1][0, :2])) / 3
        actual = evaluate_loss(FixedModel(), Dataset(), 2, 2, normalization="assistant_turn")
        self.assertAlmostEqual(actual, expected_turn.item(), places=6)
        expected_token = sum(float(mx.sum(loss * mx.array(mask)).item())
                             for loss, mask in zip(losses, masks)) / sum(float(mask.sum()) for mask in masks)
        self.assertEqual(evaluate_loss(FixedModel(), Dataset(), 2, 2), expected_token)

    def test_assistant_turn_resume_preserves_objective_and_rejects_mode_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, data, config = self.fixture(root)
            config.loss_normalization = "assistant_turn"
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch("silivra.finetune.require_apple_gpu", return_value={"test": "synthetic tiny model"}):
                finetune(data, base, root / "full-turns", config, max_minutes=1)
                batches = {"count": 0}
                original_batch = InstructionDataset.batch

                def tracked_batch(dataset, rng, batch_size):
                    if dataset.path.name == "train.npz":
                        batches["count"] += 1
                    return original_batch(dataset, rng, batch_size)

                with patch.object(InstructionDataset, "batch", tracked_batch), \
                        patch("silivra.finetune.time.monotonic",
                              side_effect=lambda: 60.0 if batches["count"] >= 2 else 0.0):
                    partial = finetune(data, base, root / "turns", config, max_minutes=.5)
                self.assertEqual(partial["step"], 2)
                with self.assertRaisesRegex(ValueError, "original SFT"):
                    finetune(data, base, root / "turns", replace(config, loss_normalization="token"),
                             resume=True, max_minutes=1)
                finished = finetune(data, base, root / "turns", resume=True, max_minutes=1)
            self.assertEqual(finished["step"], 4)
            state = verify_checkpoint(resolve_checkpoint(root / "turns"))
            self.assertEqual(state["training_config"]["loss_normalization"], "assistant_turn")
            self.assertEqual(state["loss_normalization"], "assistant_turn")
            self.assertEqual(state["trainer_sha256"], sha256(Path(__file__).resolve().parents[1] / "src/silivra/finetune.py"))
            full, resumed = resolve_checkpoint(root / "full-turns"), resolve_checkpoint(root / "turns")
            for filename in ("model.safetensors", "optimizer.npz", "mlx_rng.npz"):
                expected, actual = mx.load(str(full / filename)), mx.load(str(resumed / filename))
                self.assertEqual(expected.keys(), actual.keys())
                for key in expected:
                    with self.subTest(file=filename, key=key):
                        np.testing.assert_allclose(np.asarray(actual[key]), np.asarray(expected[key]),
                                                   rtol=1e-6, atol=1e-8)
            self.assertEqual(state["numpy_rng"], verify_checkpoint(full)["numpy_rng"])

    def test_old_checkpoint_without_normalization_field_resumes_as_token_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, data, config = self.fixture(root)
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch("silivra.finetune.require_apple_gpu", return_value={"test": "synthetic tiny model"}):
                batches = {"count": 0}
                original_batch = InstructionDataset.batch

                def tracked_batch(dataset, rng, batch_size):
                    if dataset.path.name == "train.npz":
                        batches["count"] += 1
                    return original_batch(dataset, rng, batch_size)

                with patch.object(InstructionDataset, "batch", tracked_batch), \
                        patch("silivra.finetune.time.monotonic",
                              side_effect=lambda: 60.0 if batches["count"] >= 2 else 0.0):
                    partial = finetune(data, base, root / "legacy", config, max_minutes=.5)
                self.assertEqual(partial["step"], 2)
                checkpoint = resolve_checkpoint(root / "legacy")
                state = json.loads((checkpoint / "state.json").read_text())
                del state["training_config"]["loss_normalization"]
                state.pop("loss_normalization")
                state.pop("trainer_sha256")
                (checkpoint / "state.json").write_text(json.dumps(state))
                checksums = json.loads((checkpoint / "checksums.json").read_text())
                checksums["state.json"] = sha256(checkpoint / "state.json")
                (checkpoint / "checksums.json").write_text(json.dumps(checksums))
                result = finetune(data, base, root / "legacy", resume=True, max_minutes=1)
                with self.assertRaisesRegex(ValueError, "original SFT"):
                    finetune(data, base, root / "legacy", replace(config, loss_normalization="assistant_turn"),
                             resume=True, max_minutes=1)
            self.assertEqual(result["step"], 4)
            self.assertEqual(result["loss_normalization"], "token")

    def test_resume_matches_full_weights_optimizer_and_rng_and_preserves_base(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, data, config = self.fixture(root)
            # Resume across warmup/decay with dynamic input widths, as used by
            # assistant-v2, rather than checking only constant-rate updates.
            config.context_length, config.bucket_size = 192, 32
            config.warmup_steps, config.min_lr_ratio = 3, .2
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

    def test_exclusive_run_lock_refuses_second_trainer_without_checkpoint_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, data, config = self.fixture(root)
            run = root / "locked"
            run.mkdir()
            with (run / ".lock").open("a") as lock, \
                    patch("silivra.finetune.require_apple_gpu", return_value={"test": "synthetic"}):
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                with self.assertRaisesRegex(RuntimeError, "Another trainer"):
                    finetune(data, base, run, config, resume=True)
                self.assertEqual([item.name for item in run.iterdir()], [".lock"])

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

    def test_context_extension_schedule_and_legacy_config_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            base, data, config = self.fixture(root)
            config.context_length, config.bucket_size = 192, 32
            config.warmup_steps, config.min_lr_ratio = 1, .2
            from silivra.finetune import learning_rate
            schedule = learning_rate(config)
            self.assertLess(float(schedule(mx.array(0)).item()), float(schedule(mx.array(1)).item()))
            self.assertAlmostEqual(float(schedule(mx.array(config.steps)).item()), config.learning_rate * .2)
            with contextlib.redirect_stdout(io.StringIO()), \
                    patch("silivra.finetune.require_apple_gpu", return_value={"test": "synthetic"}):
                finetune(data, base, root / "extended", config, max_minutes=1)
                finetune(data, base, root / "extended", resume=True, max_minutes=1)
            checkpoint = resolve_checkpoint(root / "extended")
            state = verify_checkpoint(checkpoint)
            self.assertEqual(state["model_config"]["max_seq_len"], 192)
            self.assertEqual(verify_checkpoint(resolve_checkpoint(base))["model_config"]["max_seq_len"], 96)
            self.assertFalse((root / "new").exists())


if __name__ == "__main__":
    unittest.main()

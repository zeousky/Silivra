"""Numerical model-contract checks, including causal and cached attention."""

import json
from pathlib import Path
import tempfile
import unittest

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from silivra.config import ModelConfig
from silivra.model import SilivraLM, parameter_count


def tiny_config(**updates):
    values = dict(vocab_size=32, dim=32, n_layers=2, n_heads=4,
                  n_kv_heads=2, hidden_dim=64, max_seq_len=16)
    values.update(updates)
    return ModelConfig(**values)


class ConfigTests(unittest.TestCase):
    def test_json_round_trip(self):
        config = ModelConfig()
        self.assertEqual(ModelConfig.from_dict(json.loads(json.dumps(config.to_dict()))), config)

    def test_invalid_architectures_are_rejected(self):
        for updates in (
            {"dim": 31}, {"dim": 36}, {"n_kv_heads": 3}, {"n_layers": 0},
            {"vocab_size": True}, {"hidden_dim": 2.0}, {"max_seq_len": -1},
            {"rope_theta": float("nan")}, {"rope_theta": float("inf")},
            {"rope_theta": 0}, {"rope_theta": "10000"},
        ):
            with self.subTest(updates=updates), self.assertRaises(ValueError):
                tiny_config(**updates)
        with self.assertRaises(ValueError):
            ModelConfig.from_dict({"n_layer": 4})
        with self.assertRaises(ValueError):
            ModelConfig.from_dict([])


class ModelTests(unittest.TestCase):
    def setUp(self):
        mx.random.seed(7)
        self.config = tiny_config()
        self.model = SilivraLM(self.config)
        mx.eval(self.model.parameters())

    def assertClose(self, left, right, atol=2e-5):
        self.assertTrue(mx.allclose(left, right, atol=atol, rtol=2e-5).item(),
                        f"maximum difference: {mx.max(mx.abs(left - right)).item()}")

    def test_output_shape_parameter_count_and_tied_embedding(self):
        logits = self.model(mx.array([[1, 2, 3], [4, 5, 6]]))
        self.assertEqual(logits.shape, (2, 3, self.config.vocab_size))
        dim, hidden = self.config.dim, self.config.hidden_dim
        kv_dim = self.config.n_kv_heads * self.config.head_dim
        expected = self.config.vocab_size * dim + dim
        expected += self.config.n_layers * (2 * dim * dim + 2 * kv_dim * dim + 3 * dim * hidden + 2 * dim)
        self.assertEqual(parameter_count(self.model), expected)
        self.assertEqual(self.model.num_parameters, expected)
        self.assertTrue(mx.all(mx.isfinite(logits)).item())

    def test_future_tokens_cannot_change_past_logits(self):
        first = self.model(mx.array([[1, 2, 3, 4, 5, 6]]))
        changed = self.model(mx.array([[1, 2, 3, 21, 22, 23]]))
        self.assertClose(first[:, :3], changed[:, :3])
        self.assertGreater(mx.max(mx.abs(first[:, 3:] - changed[:, 3:])).item(), 1e-4)

    def test_cached_single_tokens_match_full_forward(self):
        self.model.eval()
        tokens = mx.array([[1, 2, 3, 4, 5, 6], [6, 5, 4, 3, 2, 1]])
        full = self.model(tokens)
        cache = self.model.make_cache()
        chunks = [self.model(tokens[:, index:index + 1], cache) for index in range(tokens.shape[1])]
        self.assertClose(full, mx.concatenate(chunks, axis=1))
        self.assertTrue(all(item.offset == tokens.shape[1] for item in cache))

    def test_cached_prefill_and_chunked_queries_match_full_forward(self):
        self.model.eval()
        tokens = mx.array([[2, 3, 5, 7, 11, 13, 17]])
        full = self.model(tokens)
        cache = self.model.make_cache()
        chunks = [self.model(tokens[:, :3], cache), self.model(tokens[:, 3:5], cache), self.model(tokens[:, 5:], cache)]
        self.assertClose(full, mx.concatenate(chunks, axis=1))

    def test_cache_growth_preserves_positions_across_storage_chunks(self):
        model = SilivraLM(tiny_config(max_seq_len=300))
        model.eval()
        tokens = (mx.arange(259) % model.config.vocab_size)[None]
        full = model(tokens)
        cache = model.make_cache()
        prefix = model(tokens[:, :255], cache)
        suffix = model(tokens[:, 255:], cache)
        self.assertClose(full, mx.concatenate((prefix, suffix), axis=1))
        self.assertTrue(all(item.offset == 259 for item in cache))
        self.assertTrue(all(item.keys.shape[2] == 300 for item in cache))

    def test_loss_has_finite_nonzero_gradients(self):
        tokens = mx.array([[1, 2, 3, 4, 5], [5, 4, 3, 2, 1]])

        def loss(model, inputs):
            return nn.losses.cross_entropy(model(inputs[:, :-1]), inputs[:, 1:], reduction="mean")

        value, gradients = nn.value_and_grad(self.model, loss)(self.model, tokens)
        mx.eval(value, gradients)
        self.assertTrue(mx.isfinite(value).item())
        for name, grad in tree_flatten(gradients):
            with self.subTest(parameter=name):
                self.assertTrue(mx.all(mx.isfinite(grad)).item())
                self.assertGreater(mx.sum(mx.abs(grad)).item(), 0)

    def test_weights_round_trip(self):
        tokens = mx.array([[1, 2, 3]])
        expected = self.model(tokens)
        mx.eval(expected)
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "model.safetensors")
            self.model.save_weights(path)
            restored = SilivraLM(self.config)
            restored.load_weights(path)
            self.assertClose(expected, restored(tokens), atol=0)

    def test_context_limit_and_cache_use_are_explicit(self):
        with self.assertRaises(ValueError):
            self.model(mx.ones((1, 17), dtype=mx.int32))
        with self.assertRaises(ValueError):
            self.model(mx.array([[1]]), self.model.make_cache())
        with self.assertRaises(ValueError):
            self.model(mx.array([[1.0]]))
        self.model.eval()
        cache = self.model.make_cache()
        mx.eval(self.model(mx.ones((1, 16), dtype=mx.int32), cache))
        with self.assertRaises(ValueError):
            self.model(mx.array([[1]]), cache)
        self.assertTrue(all(item.offset == 16 for item in cache))
        with self.assertRaises(ValueError):
            self.model(mx.array([[1]]), cache[:-1])


if __name__ == "__main__":
    unittest.main()

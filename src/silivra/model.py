"""A from-random causal language model implemented directly in Apple MLX.

The decoder uses established transformer components: grouped-query attention,
rotary positions, pre-normalization, SwiGLU and a tied token/output embedding.
No pretrained model or pretrained weights are loaded by this module.
"""

from dataclasses import dataclass
import math

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten

from .config import ModelConfig


@dataclass
class KVCache:
    """An inference-only cache with bounded, chunked storage.

    Positions are never silently discarded: cached hidden states depend on the
    earlier context. To start a new context window, create a fresh cache and
    prefill the desired token window again.
    """

    max_seq_len: int
    offset: int = 0
    keys: mx.array | None = None
    values: mx.array | None = None

    def update(self, keys: mx.array, values: mx.array) -> tuple[mx.array, mx.array]:
        if keys.shape != values.shape or keys.ndim != 4:
            raise ValueError("Cache keys and values must have matching [B, H, T, D] shapes")
        end = self.offset + keys.shape[2]
        if end > self.max_seq_len:
            raise ValueError("KV cache exceeds max_seq_len; reset it and prefill a new window")
        if self.keys is not None:
            if (
                self.values is None
                or self.keys.shape[:2] != keys.shape[:2]
                or self.keys.shape[3] != keys.shape[3]
                or self.keys.dtype != keys.dtype
            ):
                raise ValueError("KV cache batch, heads, dimensions or dtype changed")
        capacity = 0 if self.keys is None else self.keys.shape[2]
        if end > capacity:
            new_capacity = min(self.max_seq_len, math.ceil(end / 256) * 256)
            shape = (keys.shape[0], keys.shape[1], new_capacity - capacity, keys.shape[3])
            extra_keys = mx.zeros(shape, dtype=keys.dtype)
            extra_values = mx.zeros(shape, dtype=values.dtype)
            if self.keys is None:
                self.keys, self.values = extra_keys, extra_values
            else:
                self.keys = mx.concatenate((self.keys, extra_keys), axis=2)
                self.values = mx.concatenate((self.values, extra_values), axis=2)
        self.keys[:, :, self.offset:end, :] = keys
        self.values[:, :, self.offset:end, :] = values
        self.offset = end
        return self.keys[:, :, :end, :], self.values[:, :, :end, :]


def _normal_linear(in_features: int, out_features: int, std: float = 0.02) -> nn.Linear:
    layer = nn.Linear(in_features, out_features, bias=False)
    layer.weight = mx.random.normal(layer.weight.shape) * std
    return layer


class Attention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.head_dim
        self.scale = self.head_dim ** -0.5
        self.q_proj = _normal_linear(config.dim, config.dim)
        self.k_proj = _normal_linear(config.dim, config.n_kv_heads * self.head_dim)
        self.v_proj = _normal_linear(config.dim, config.n_kv_heads * self.head_dim)
        self.o_proj = _normal_linear(config.dim, config.dim, 0.02 / math.sqrt(2 * config.n_layers))
        self.rope = nn.RoPE(self.head_dim, traditional=False, base=config.rope_theta)

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        batch, length, _ = x.shape
        queries = self.q_proj(x).reshape(batch, length, self.n_heads, self.head_dim).transpose(0, 2, 1, 3)
        keys = self.k_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        values = self.v_proj(x).reshape(batch, length, self.n_kv_heads, self.head_dim).transpose(0, 2, 1, 3)
        offset = 0 if cache is None else cache.offset
        queries = self.rope(queries, offset=offset)
        keys = self.rope(keys, offset=offset)
        if cache is not None:
            keys, values = cache.update(keys, values)
        # MLX supports GQA directly; repeating K/V would waste unified memory.
        # Its causal mask aligns the final query with the final cached key.
        attention = mx.fast.scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask="causal"
        )
        return self.o_proj(attention.transpose(0, 2, 1, 3).reshape(batch, length, -1))


class FeedForward(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.gate_proj = _normal_linear(config.dim, config.hidden_dim)
        self.up_proj = _normal_linear(config.dim, config.hidden_dim)
        self.down_proj = _normal_linear(config.hidden_dim, config.dim, 0.02 / math.sqrt(2 * config.n_layers))

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class TransformerBlock(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.attention_norm = nn.RMSNorm(config.dim, eps=1e-5)
        self.attention = Attention(config)
        self.ffn_norm = nn.RMSNorm(config.dim, eps=1e-5)
        self.feed_forward = FeedForward(config)

    def __call__(self, x: mx.array, cache: KVCache | None = None) -> mx.array:
        x = x + self.attention(self.attention_norm(x), cache)
        return x + self.feed_forward(self.ffn_norm(x))


class SilivraLM(nn.Module):
    """A decoder-only language model; parameters are float32 by default.

    ``tokens`` has shape ``[batch, sequence]`` with integer IDs in the configured
    vocabulary. A forward pass returns ``[batch, sequence, vocab_size]`` logits.
    Call ``eval()`` before passing an optional cache from ``make_cache()``.
    MLX's standard ``save_weights`` and ``load_weights`` methods serialize weights.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.dim)
        self.token_embedding.weight = mx.random.normal((config.vocab_size, config.dim)) * 0.02
        self.layers = [TransformerBlock(config) for _ in range(config.n_layers)]
        self.norm = nn.RMSNorm(config.dim, eps=1e-5)

    @property
    def num_parameters(self) -> int:
        return parameter_count(self)

    def make_cache(self) -> list[KVCache]:
        return [KVCache(self.config.max_seq_len) for _ in self.layers]

    def __call__(self, tokens: mx.array, cache: list[KVCache] | None = None) -> mx.array:
        if tokens.ndim != 2 or tokens.shape[0] < 1 or tokens.shape[1] < 1:
            raise ValueError("tokens must have nonempty [batch, sequence] shape")
        if not mx.issubdtype(tokens.dtype, mx.integer):
            raise ValueError("tokens must contain integer IDs")
        offset = 0
        if cache is not None:
            if self.training:
                raise ValueError("KV caching is inference-only; call model.eval() first")
            if len(cache) != len(self.layers) or not all(isinstance(item, KVCache) for item in cache):
                raise ValueError("cache must contain one KVCache for each transformer layer")
            offset = cache[0].offset
            if any(item.offset != offset or item.max_seq_len != self.config.max_seq_len for item in cache):
                raise ValueError("cache layers must have matching offsets and context limits")
        if offset + tokens.shape[1] > self.config.max_seq_len:
            raise ValueError("Input exceeds max_seq_len; reset cache and prefill a new context window")
        x = self.token_embedding(tokens)
        for index, layer in enumerate(self.layers):
            x = layer(x, None if cache is None else cache[index])
        # One shared matrix, so a checkpoint cannot drift between embedding/head.
        return self.norm(x) @ self.token_embedding.weight.T


def parameter_count(model: nn.Module) -> int:
    """Count learned scalars, counting the tied embedding exactly once."""
    return sum(parameter.size for _, parameter in tree_flatten(model.parameters()))

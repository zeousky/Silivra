"""One bounded in-memory cache of a verified input-token prefix.

This is an inference optimization, not persistent conversation memory. A cache
belongs to one backend/model instance. Exact token equality determines reuse;
generated answers, EOS and speculative lookahead never become cached inputs.
Call ``commit`` only after the complete generated action has validated. Any
cancelled, failed or budget-limited generation must leave its prepared cache
uncommitted. The caller serializes inference and closes its generator first.
"""

from __future__ import annotations

import copy


MAX_PREFIX_TOKENS = 1024
MAX_CACHE_BYTES = 256 * 1024**2
MAX_CONTEXT_TOKENS = 8192


def _helpers():
    # Portable tests and CLI help do not import Apple-only dependencies.
    from mlx_lm.models.cache import (can_trim_prompt_cache, make_prompt_cache,
                                     trim_prompt_cache)
    return make_prompt_cache, can_trim_prompt_cache, trim_prompt_cache


def _snapshot_helpers():
    from mlx_lm.models.cache import ArraysCache, KVCache

    def clone(cache):
        import mlx.core as mx
        result = copy.deepcopy(cache)
        mx.eval([layer.state for layer in result])
        return result

    return ArraysCache, KVCache, clone


def snapshot_cache_supported(cache) -> bool:
    """Allow only the pinned single-sequence hybrid cache implementation."""
    if (not isinstance(cache, list) or not 1 <= len(cache) <= 128
            or {type(layer).__name__ for layer in cache} != {"ArraysCache", "KVCache"}):
        return False
    arrays, kv, _ = _snapshot_helpers()
    return all(type(layer) in (arrays, kv) for layer in cache)


def _snapshot_info(cache, expected_offset):
    if (type(expected_offset) is not int or not 1 <= expected_offset <= MAX_PREFIX_TOKENS
            or not snapshot_cache_supported(cache)):
        return None
    arrays, kv, _ = _snapshot_helpers()
    total = 0
    for layer in cache:
        size = getattr(layer, "nbytes", None)
        if type(size) is not int or size < 0:
            return None
        total += size
        if total > MAX_CACHE_BYTES:
            return None
        if type(layer) is kv:
            if type(layer.offset) is not int or layer.offset != expected_offset:
                return None
            if (layer.keys is None or layer.values is None
                    or layer.keys.shape[2] < expected_offset
                    or layer.values.shape[2] < expected_offset):
                return None
        elif (not isinstance(layer.cache, list) or len(layer.cache) != 2
              or any(value is None for value in layer.cache)
              or layer.left_padding is not None or layer.lengths is not None):
            return None
    return total


def _tokens(full_tokens) -> list[int]:
    if (not isinstance(full_tokens, list) or not 1 <= len(full_tokens) <= MAX_CONTEXT_TOKENS
            or any(type(token) is not int or token < 0 for token in full_tokens)):
        raise ValueError("Prompt cache needs 1–8192 nonnegative integer input tokens.")
    return list(full_tokens)


def _inspect(cache):
    """Require a uniform actual offset and bounded actual allocated KV bytes."""
    if not isinstance(cache, list) or not 1 <= len(cache) <= 128:
        return None
    offset, total = None, 0
    for layer in cache:
        current, size = getattr(layer, "offset", None), getattr(layer, "nbytes", None)
        if (type(current) is not int or not 0 <= current <= MAX_CONTEXT_TOKENS + 2
                or type(size) is not int or size < 0):
            return None
        if offset is not None and current != offset:
            return None
        offset = current
        total += size
        if total > MAX_CACHE_BYTES:
            return None
    return offset, total


class PromptCache:
    """Own at most one 1024-token/256-MiB input prefix for a model instance."""

    def __init__(self, model):
        self.model = model
        self.strategy = "cold"
        self.clear()

    def _active(self, cache, tokens, reused, snapshot_prefix_tokens):
        self._working = cache
        self._request_tokens = tokens
        self._reused = reused
        self._snapshot_target = min(snapshot_prefix_tokens, len(tokens) - 1, MAX_PREFIX_TOKENS)
        self.strategy = ("hybrid-snapshot" if snapshot_cache_supported(cache)
                         else "trimmed-kv" if _helpers()[1](cache) else "cold")
        return cache, tokens[reused:], reused

    @property
    def retained_token_count(self) -> int:
        return len(self._tokens)

    def clear(self):
        """Discard committed and pending references; no cache is persisted."""
        self._cache = None
        self._tokens = []
        self._working = None
        self._request_tokens = []
        self._reused = 0
        self._snapshot_target = 0
        self._candidate = None
        self._candidate_tokens = []

    def prepare(self, full_tokens, *, snapshot_prefix_tokens=0) -> tuple[list, list[int], int]:
        """Consume prior state and return KV, uncached suffix, reused token count.

        At least one input token always remains for MLX-LM to compute its first
        logits, including identical-prompt hits. Consuming the stored entry
        before inference prevents a partial/cancelled mutation being reused.
        """
        cached, previous = self._cache, self._tokens
        self.clear()
        tokens = _tokens(full_tokens)
        if (type(snapshot_prefix_tokens) is not int
                or not 0 <= snapshot_prefix_tokens <= MAX_PREFIX_TOKENS):
            raise ValueError("Hybrid snapshot prefix must be between 0 and 1024 tokens.")
        make_cache, can_trim, trim = _helpers()
        if (cached is not None and previous and len(previous) < len(tokens)
                and tokens[:len(previous)] == previous
                and _snapshot_info(cached, len(previous)) is not None):
            _, _, clone = _snapshot_helpers()
            working = clone(cached)
            if _snapshot_info(working, len(previous)) is not None:
                # The prior immutable snapshot is only a candidate until the
                # new action validates. Inference mutates its independent clone.
                self._candidate, self._candidate_tokens = cached, previous
                return self._active(working, tokens, len(previous), snapshot_prefix_tokens)
        info = _inspect(cached)
        if cached is not None and info is not None and info[0] == len(previous) and can_trim(cached):
            reused = 0
            for old, new in zip(previous, tokens[:-1]):
                if old != new:
                    break
                reused += 1
            if reused:
                remove = info[0] - reused
                trimmed = trim(cached, remove) if remove else 0
                after = _inspect(cached)
                if type(trimmed) is int and trimmed == remove and after is not None and after[0] == reused:
                    return self._active(cached, tokens, reused, snapshot_prefix_tokens)
        return self._active(make_cache(self.model), tokens, 0, snapshot_prefix_tokens)

    def record_prefill(self, cache, processed, total):
        """Snapshot one evaluated input boundary, before decoding/lookahead.

        MLX-LM's final progress callback is after generation lookahead. It must
        never be captured, even though its counters report only prompt tokens.
        Only the known chunk boundary below the final prompt token is eligible.
        """
        if (cache is not self._working or self.strategy != "hybrid-snapshot"
                or self._candidate is not None or type(processed) is not int
                or type(total) is not int or not 0 < processed < total
                or total != len(self._request_tokens) - self._reused):
            return False
        position = self._reused + processed
        if position != self._snapshot_target or _snapshot_info(cache, position) is None:
            return False
        _, _, clone = _snapshot_helpers()
        candidate = clone(cache)
        if _snapshot_info(candidate, position) is None:
            return False
        self._candidate, self._candidate_tokens = candidate, self._request_tokens[:position]
        return True

    def commit(self, cache, full_tokens) -> bool:
        """Retain only an exact input prefix after successful validated inference.

        Actual layer offsets, rather than response counts, include MLX-LM's
        one-step decoding lookahead. Large allocated arrays are dropped even
        when trimming their logical offsets would fit the token retention cap.
        Returns whether a bounded reusable entry was committed.
        """
        candidate, prefix = self._candidate, self._candidate_tokens
        is_snapshot = self.strategy == "hybrid-snapshot" and cache is self._working
        request_tokens = self._request_tokens
        self.clear()
        tokens = _tokens(full_tokens)
        matches_request = tokens == request_tokens
        if is_snapshot:
            if (matches_request and candidate is not None and prefix
                    and len(prefix) < len(tokens) and tokens[:len(prefix)] == prefix
                    and _snapshot_info(candidate, len(prefix)) is not None):
                self._cache, self._tokens = candidate, prefix
                return True
            return False
        keep = min(len(tokens) - 1, MAX_PREFIX_TOKENS)
        info = _inspect(cache)
        if keep <= 0 or info is None or info[0] < len(tokens):
            return False
        _, can_trim, trim = _helpers()
        if not can_trim(cache):
            return False
        remove = info[0] - keep
        trimmed = trim(cache, remove)
        after = _inspect(cache)
        if type(trimmed) is not int or trimmed != remove or after is None or after[0] != keep:
            return False
        self._cache, self._tokens = cache, tokens[:keep]
        return True

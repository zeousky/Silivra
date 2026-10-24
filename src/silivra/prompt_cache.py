"""One bounded in-memory cache of a verified input-token prefix.

This is an inference optimization, not persistent conversation memory. A cache
belongs to one backend/model instance. Exact token equality determines reuse;
generated answers, EOS and speculative lookahead never become cached inputs.
Call ``commit`` only after the complete generated action has validated. Any
cancelled, failed or budget-limited generation must leave its prepared cache
uncommitted. The caller serializes inference and closes its generator first.
"""

from __future__ import annotations


MAX_PREFIX_TOKENS = 1024
MAX_CACHE_BYTES = 256 * 1024**2
MAX_CONTEXT_TOKENS = 8192


def _helpers():
    # Portable tests and CLI help do not import Apple-only dependencies.
    from mlx_lm.models.cache import (can_trim_prompt_cache, make_prompt_cache,
                                     trim_prompt_cache)
    return make_prompt_cache, can_trim_prompt_cache, trim_prompt_cache


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
        self._cache = None
        self._tokens = []

    @property
    def retained_token_count(self) -> int:
        return len(self._tokens)

    def clear(self):
        """Discard the reusable reference; no cache is persisted to disk."""
        self._cache = None
        self._tokens = []

    def prepare(self, full_tokens) -> tuple[list, list[int], int]:
        """Consume prior state and return KV, uncached suffix, reused token count.

        At least one input token always remains for MLX-LM to compute its first
        logits, including identical-prompt hits. Consuming the stored entry
        before inference prevents a partial/cancelled mutation being reused.
        """
        cached, previous = self._cache, self._tokens
        self.clear()
        tokens = _tokens(full_tokens)
        make_cache, can_trim, trim = _helpers()
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
                    return cached, tokens[reused:], reused
        return make_cache(self.model), tokens, 0

    def commit(self, cache, full_tokens) -> bool:
        """Retain only an exact input prefix after successful validated inference.

        Actual layer offsets, rather than response counts, include MLX-LM's
        one-step decoding lookahead. Large allocated arrays are dropped even
        when trimming their logical offsets would fit the token retention cap.
        Returns whether a bounded reusable entry was committed.
        """
        self.clear()
        tokens = _tokens(full_tokens)
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

"""Portable exact-prefix and failure-boundary checks using fake KV caches."""

import copy
import unittest
from unittest.mock import Mock, patch

from silivra.prompt_cache import MAX_CACHE_BYTES, MAX_PREFIX_TOKENS, PromptCache


class Layer:
    def __init__(self, offset=0, nbytes=1024, trimmable=True):
        self.offset, self.nbytes, self.trimmable = offset, nbytes, trimmable


class PromptCacheTests(unittest.TestCase):
    def setUp(self):
        self.model = object()
        self.cache = PromptCache(self.model)
        self.make = Mock(side_effect=lambda _: [Layer(), Layer()])
        self.can_trim = Mock(side_effect=lambda cache: all(layer.trimmable for layer in cache))
        self.trim = Mock(side_effect=self.trim_layers)
        self.helpers = patch("silivra.prompt_cache._helpers", return_value=(self.make, self.can_trim, self.trim))
        self.helpers.start()
        self.addCleanup(self.helpers.stop)

    @staticmethod
    def trim_layers(cache, amount):
        for layer in cache:
            layer.offset -= amount
        return amount

    @staticmethod
    def completed(offset, **kwargs):
        return [Layer(offset, **kwargs), Layer(offset, **kwargs)]

    def test_cold_prepare_and_no_commit_never_reuse_partial_state(self):
        working, suffix, reused = self.cache.prepare([1, 2, 3])
        self.assertEqual((suffix, reused), ([1, 2, 3], 0))
        self.make.assert_called_once_with(self.model)
        for layer in working:
            layer.offset = 50  # Interrupted/failed inference: caller never commits.
        next_cache, suffix, reused = self.cache.prepare([1, 2, 3])
        self.assertIsNot(next_cache, working)
        self.assertEqual((suffix, reused), ([1, 2, 3], 0))

    def test_commit_trims_actual_generated_and_lookahead_offsets(self):
        completed = self.completed(12)  # Three inputs plus answer/EOS/lookahead.
        self.assertTrue(self.cache.commit(completed, [1, 2, 3]))
        self.assertEqual(self.cache.retained_token_count, 2)
        self.trim.assert_called_once_with(completed, 10)
        self.assertEqual([layer.offset for layer in completed], [2, 2])
        working, suffix, reused = self.cache.prepare([1, 2, 3])
        self.assertIs(working, completed)
        self.assertEqual((suffix, reused), ([3], 2))
        self.assertEqual(self.cache._tokens, [])
        self.assertEqual(self.cache.retained_token_count, 0)
        self.assertIsNone(self.cache._cache)

    def test_diverging_prompts_reuse_only_exact_shared_tokens(self):
        completed = self.completed(20)
        self.assertTrue(self.cache.commit(completed, [10, 11, 12, 13, 14]))
        working, suffix, reused = self.cache.prepare([10, 11, 99, 13, 14])
        self.assertIs(working, completed)
        self.assertEqual((suffix, reused), ([99, 13, 14], 2))
        self.assertEqual([layer.offset for layer in working], [2, 2])

    def test_changed_policy_or_tool_prefix_cannot_reuse_stale_tail(self):
        old = self.completed(12)
        self.assertTrue(self.cache.commit(old, [10, 11, 12]))
        new, suffix, reused = self.cache.prepare([100, 11, 12])
        self.assertIsNot(new, old)
        self.assertEqual((suffix, reused), ([100, 11, 12], 0))

    def test_shorter_prompt_still_keeps_one_token_uncached(self):
        old = self.completed(12)
        self.assertTrue(self.cache.commit(old, [1, 2, 3, 4]))
        working, suffix, reused = self.cache.prepare([1, 2])
        self.assertIs(working, old)
        self.assertEqual((suffix, reused), ([2], 1))
        self.assertEqual([layer.offset for layer in working], [1, 1])

    def test_retention_tokens_and_allocated_bytes_are_bounded(self):
        tokens = list(range(2000))
        working = self.completed(2200)
        self.assertTrue(self.cache.commit(working, tokens))
        self.assertEqual(len(self.cache._tokens), MAX_PREFIX_TOKENS)
        self.assertEqual([layer.offset for layer in working], [MAX_PREFIX_TOKENS] * 2)
        # Logical offset trimming need not release allocated MLX buffers.
        oversized = self.completed(12, nbytes=MAX_CACHE_BYTES // 2 + 1)
        self.assertFalse(self.cache.commit(oversized, [1, 2, 3]))
        self.assertIsNone(self.cache._cache)

    def test_invalid_mixed_or_incomplete_offsets_are_not_committed(self):
        for cache in ([], [Layer(5), Layer(6)], [Layer(True)], [Layer(-1)],
                      [Layer(2)], [Layer(9000)], [Layer(4, nbytes=-1)], [object()]):
            with self.subTest(cache=cache):
                self.assertFalse(self.cache.commit(cache, [1, 2, 3]))
                self.assertIsNone(self.cache._cache)

    def test_untrimmable_or_incorrect_trim_results_are_rejected(self):
        self.assertFalse(self.cache.commit(self.completed(12, trimmable=False), [1, 2, 3]))
        self.trim.side_effect = lambda cache, amount: amount  # Claims success without updating offsets.
        self.assertFalse(self.cache.commit(self.completed(12), [1, 2, 3]))
        self.trim.side_effect = lambda cache, amount: False
        self.assertFalse(self.cache.commit(self.completed(12), [1, 2, 3]))

    def test_prepare_revalidates_stored_metadata_and_falls_back_cold(self):
        old = self.completed(12)
        self.assertTrue(self.cache.commit(old, [1, 2, 3]))
        old[0].offset = 1
        working, suffix, reused = self.cache.prepare([1, 2, 3])
        self.assertIsNot(working, old)
        self.assertEqual((suffix, reused), ([1, 2, 3], 0))

    def test_clear_cancellation_and_model_instances_are_isolated(self):
        self.assertTrue(self.cache.commit(self.completed(12), [1, 2, 3]))
        self.cache.clear()
        self.assertEqual(self.cache.prepare([1, 2, 3])[2], 0)
        self.assertTrue(self.cache.commit(self.completed(12), [1, 2, 3]))
        other = PromptCache(object())
        self.assertEqual(other.prepare([1, 2, 3])[2], 0)
        self.assertEqual(self.cache.prepare([1, 2, 3])[2], 2)

    def test_input_list_is_copied_and_malformed_requests_invalidate_state(self):
        tokens = [1, 2, 3]
        self.assertTrue(self.cache.commit(self.completed(12), tokens))
        tokens[0] = 999
        self.assertEqual(self.cache.prepare([1, 2, 3])[2], 2)
        for bad in ([], (1, 2), [True], [-1], [1.5], list(range(8193))):
            with self.subTest(bad=str(bad)[:20]):
                self.assertTrue(self.cache.commit(self.completed(12), [1, 2, 3]))
                with self.assertRaises(ValueError):
                    self.cache.prepare(bad)
                self.assertIsNone(self.cache._cache)


class StateArray:
    def __init__(self, length=1024):
        self.shape = (1, 4, length, 256)
        self.values = [1, 2, 3]


class ArraysCache:
    def __init__(self):
        self.cache = [StateArray(), StateArray()]
        self.left_padding = self.lengths = None
        self.nbytes = 1024


class KVCache:
    def __init__(self, offset=0):
        self.offset = offset
        self.keys, self.values = StateArray(), StateArray()
        self.nbytes = 1024


class HybridPromptCacheTests(unittest.TestCase):
    def setUp(self):
        self.cache = PromptCache(object())
        self.make = Mock(side_effect=lambda _: [ArraysCache(), KVCache()])
        self.trim = Mock(side_effect=AssertionError("Recurrent caches must never be trimmed"))
        self.clone = Mock(side_effect=copy.deepcopy)
        self.helpers = patch("silivra.prompt_cache._helpers", return_value=(self.make, lambda _: False, self.trim))
        self.snapshot_patch = patch("silivra.prompt_cache._snapshot_helpers", return_value=(ArraysCache, KVCache, self.clone))
        self.helpers.start()
        self.snapshot_patch.start()
        self.addCleanup(self.helpers.stop)
        self.addCleanup(self.snapshot_patch.stop)

    def prepare(self, tokens=None, target=3):
        tokens = [10, 11, 12, 13, 14] if tokens is None else tokens
        working, suffix, reused = self.cache.prepare(tokens, snapshot_prefix_tokens=target)
        return tokens, working, suffix, reused

    def snapshot(self, working, processed=3, total=5):
        working[1].offset = processed
        return self.cache.record_prefill(working, processed, total)

    def test_only_known_safe_evaluated_prefill_boundary_is_captured(self):
        tokens, working, _, _ = self.prepare()
        for processed, total in ((0, 5), (5, 5), (3, 6), (True, 5), (2, 5), (4, 5)):
            working[1].offset = int(processed)
            self.assertFalse(self.cache.record_prefill(working, processed, total))
        self.assertTrue(self.snapshot(working))
        self.assertEqual(self.clone.call_count, 1)
        working[1].offset = 7  # Generated answer and decoding lookahead.
        self.assertFalse(self.cache.record_prefill(working, 5, 5))
        self.assertTrue(self.cache.commit(working, tokens))
        self.assertEqual(self.cache._tokens, tokens[:3])
        self.assertEqual(self.cache._cache[1].offset, 3)
        self.trim.assert_not_called()

    def test_exact_entire_prefix_is_reused_without_trimming_or_generated_tail(self):
        tokens, working, _, _ = self.prepare()
        self.assertTrue(self.snapshot(working))
        working[1].offset = 8
        self.assertTrue(self.cache.commit(working, tokens))
        other_tokens = tokens[:3] + [99, 100, 101]
        _, next_working, suffix, reused = self.prepare(other_tokens)
        self.assertEqual((suffix, reused), ([99, 100, 101], 3))
        self.assertEqual(next_working[1].offset, 3)
        self.assertEqual(self.cache.retained_token_count, 0)  # Consumed, not committed.
        self.assertTrue(self.cache.commit(next_working, other_tokens))
        self.assertEqual(self.cache.retained_token_count, 3)
        self.trim.assert_not_called()

    def test_prefix_mismatch_or_too_short_request_starts_cold(self):
        for replacement in ([10, 99, 12, 13], [10, 11, 12], [10, 11]):
            tokens, working, _, _ = self.prepare()
            self.assertTrue(self.snapshot(working))
            self.assertTrue(self.cache.commit(working, tokens))
            _, next_working, suffix, reused = self.prepare(replacement)
            self.assertEqual((suffix, reused), (list(replacement), 0))
            self.assertEqual(next_working[1].offset, 0)

    def test_snapshot_and_working_clones_are_mutation_isolated(self):
        tokens, working, _, _ = self.prepare()
        self.assertTrue(self.snapshot(working))
        working[0].cache[1].values[0] = 999
        working[1].keys.values[0] = 999
        self.assertTrue(self.cache.commit(working, tokens))
        self.assertEqual(self.cache._cache[0].cache[1].values[0], 1)
        self.assertEqual(self.cache._cache[1].keys.values[0], 1)
        _, next_working, _, _ = self.prepare()
        next_working[0].cache[0].values[0] = 777
        next_working[1].values.values[0] = 777
        self.assertTrue(self.cache.commit(next_working, tokens))
        self.assertEqual(self.cache._cache[0].cache[0].values[0], 1)
        self.assertEqual(self.cache._cache[1].values.values[0], 1)

    def test_cancel_failure_or_uncommitted_candidate_cannot_reappear(self):
        tokens, working, _, _ = self.prepare()
        self.assertTrue(self.snapshot(working))
        self.cache.clear()
        self.assertIsNone(self.cache._candidate)
        self.assertFalse(self.cache.commit(working, tokens))
        _, _, _, reused = self.prepare()
        self.assertEqual(reused, 0)
        tokens, working, _, _ = self.prepare()
        self.assertTrue(self.snapshot(working))
        _, _, _, reused = self.prepare()  # Prior action never committed.
        self.assertEqual(reused, 0)

    def test_oversize_bad_offsets_and_foreign_working_caches_are_rejected(self):
        tokens, working, _, _ = self.prepare()
        working[1].offset = 2
        self.assertFalse(self.cache.record_prefill(working, 3, 5))
        working[1].offset = 3
        self.assertFalse(self.cache.record_prefill([ArraysCache(), KVCache(3)], 3, 5))
        working[0].nbytes = MAX_CACHE_BYTES
        self.assertFalse(self.cache.record_prefill(working, 3, 5))
        self.assertFalse(self.cache.commit(working, tokens))
        self.clone.assert_not_called()

    def test_candidate_revalidates_cloned_bytes_and_request_at_commit(self):
        tokens, working, _, _ = self.prepare()
        def oversized_clone(cache):
            result = copy.deepcopy(cache)
            result[0].nbytes = MAX_CACHE_BYTES
            return result
        self.clone.side_effect = oversized_clone
        self.assertFalse(self.snapshot(working))
        self.assertFalse(self.cache.commit(working, tokens))
        self.clone.side_effect = copy.deepcopy
        tokens, working, _, _ = self.prepare()
        self.assertTrue(self.snapshot(working))
        self.assertFalse(self.cache.commit(working, [10, 11, 12, 13, 99]))

    def test_prefix_cap_and_invalid_limits_clear_existing_state(self):
        tokens = list(range(1200))
        _, working, _, _ = self.prepare(tokens, target=1024)
        self.assertTrue(self.snapshot(working, 1024, len(tokens)))
        self.assertTrue(self.cache.commit(working, tokens))
        self.assertEqual(self.cache.retained_token_count, 1024)
        with self.assertRaises(ValueError):
            self.cache.prepare(tokens, snapshot_prefix_tokens=1025)
        self.assertEqual(self.cache.retained_token_count, 0)
        self.assertIsNone(self.cache._candidate)


if __name__ == "__main__":
    unittest.main()

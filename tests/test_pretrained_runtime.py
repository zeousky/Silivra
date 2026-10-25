"""Portable runtime-probe grading tests; these do not run a model."""

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("check_pretrained_runtime", ROOT / "scripts/check-pretrained-runtime.py")
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


class PretrainedRuntimeTests(unittest.TestCase):
    @staticmethod
    def metrics(*, reused=0, prefill=100, retained=False, retained_tokens=0):
        return {"metrics": {"cached_prompt_tokens": reused, "prefill_tokens": prefill,
                            "prompt_tokens": 100, "cache_retained": retained,
                            "cache_retained_tokens": retained_tokens}}

    def test_native_cache_profile_uses_actual_layer_types_and_trim_capability(self):
        model = object()
        class ArraysCache:
            pass
        class KVCache:
            pass
        cache = [ArraysCache(), KVCache(), ArraysCache()]
        make = Mock(return_value=cache)
        can_trim = Mock(return_value=False)
        with patch.dict(sys.modules, {"mlx_lm.models.cache": SimpleNamespace(
                make_prompt_cache=make, can_trim_prompt_cache=can_trim,
                ArraysCache=ArraysCache, KVCache=KVCache)}):
            self.assertEqual(probe.cache_profile(model), {
                "trimmable": False, "snapshot_supported": True,
                "layer_types": ["ArraysCache", "KVCache"], "layer_count": 3})
        make.assert_called_once_with(model)
        can_trim.assert_called_once_with(cache)

    def test_trimmable_models_must_reuse_a_prefix_and_keep_one_token_uncached(self):
        profile = {"trimmable": True}
        cold = self.metrics(retained=True, retained_tokens=99)
        warm = self.metrics(reused=99, prefill=1, retained=True, retained_tokens=99)
        self.assertTrue(probe.cache_metrics_pass(profile, cold, warm))
        for broken in (self.metrics(retained=True),
                       self.metrics(reused=100, prefill=0, retained=True),
                       self.metrics(reused=99, prefill=1)):
            with self.subTest(metrics=broken):
                self.assertFalse(probe.cache_metrics_pass(profile, cold, broken))

    def test_hybrid_models_must_prefill_full_prompt_and_retain_no_state(self):
        profile = {"trimmable": False}
        cold = self.metrics()
        self.assertTrue(probe.cache_metrics_pass(profile, cold, self.metrics()))
        for broken in (self.metrics(reused=1), self.metrics(prefill=99),
                       self.metrics(retained=True), self.metrics(retained_tokens=1)):
            with self.subTest(metrics=broken):
                self.assertFalse(probe.cache_metrics_pass(profile, cold, broken))
                self.assertFalse(probe.cache_metrics_pass(profile, broken, cold))

    def test_snapshot_support_requires_reuse_without_claiming_native_trimmability(self):
        profile = {"trimmable": False, "snapshot_supported": True}
        cold = self.metrics(retained=True, retained_tokens=80)
        warm = self.metrics(reused=80, prefill=20, retained=True, retained_tokens=80)
        self.assertTrue(probe.cache_metrics_pass(profile, cold, warm))
        self.assertFalse(probe.cache_metrics_pass(profile, cold, self.metrics()))

    def test_identity_checker_accepts_the_requested_complete_function(self):
        self.assertTrue(probe.identity_content_pass("def identity(value):\n    return value"))
        self.assertTrue(probe.identity_content_pass("def identity(value): return value\n"))

    def test_identity_checker_rejects_undefined_names_and_changed_call_contract(self):
        for source in ("def identity(other):\n    return value",
                       "def identity():\n    return value",
                       "def identity(value, other):\n    return value",
                       "def identity(*value):\n    return value",
                       "def identity(**value):\n    return value",
                       "def identity(*, value):\n    return value",
                       "def identity(value=1):\n    return value",
                       "@replace\ndef identity(value):\n    return value",
                       "def identity(value: unknown):\n    return value",
                       "def identity(value) -> unknown:\n    return value",
                       "async def identity(value):\n    return value",
                       "def identity(value):\n    return other",
                       "def identity(value):\n    return value\nidentity = None",
                       "random words", None):
            with self.subTest(source=source):
                self.assertFalse(probe.identity_content_pass(source))


if __name__ == "__main__":
    unittest.main()

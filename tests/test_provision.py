"""Portable, offline checks for bounded and reproducible external-model setup."""

import hashlib
import io
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch

from silivra.provision import _download, model_profile, setup_pretrained


class Response(io.BytesIO):
    def __init__(self, body, *, status=200, headers=None):
        super().__init__(body)
        self.status = status
        self.headers = headers or {}


class ProvisionTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.target = Path(self.temporary.name).resolve() / "weights"
        self.data = b"synthetic model fixture"
        self.digest = hashlib.sha256(self.data).hexdigest()

    def download(self):
        _download("https://example.test/pinned", self.target, len(self.data), self.digest,
                  time.monotonic() + 10)

    def test_verified_complete_file_is_promoted_and_reused_offline(self):
        with patch("urllib.request.urlopen", return_value=Response(self.data)) as network:
            self.download()
            self.download()
            self.assertEqual(network.call_count, 1)
        self.assertEqual(self.target.read_bytes(), self.data)
        self.assertFalse(self.target.with_suffix(".partial").exists())

    def test_partial_download_resumes_with_exact_range_and_complete_hash(self):
        partial = self.target.with_suffix(".partial")
        partial.write_bytes(self.data[:5])
        response = Response(self.data[5:], status=206,
                            headers={"Content-Range": f"bytes 5-{len(self.data)-1}/{len(self.data)}"})
        with patch("urllib.request.urlopen", return_value=response) as network:
            self.download()
        self.assertEqual(network.call_args.args[0].get_header("Range"), "bytes=5-")
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_range_ignored_restarts_partial(self):
        self.target.with_suffix(".partial").write_bytes(b"old")
        with patch("urllib.request.urlopen", return_value=Response(self.data)):
            self.download()
        self.assertEqual(self.target.read_bytes(), self.data)

    def test_incorrect_range_and_overlong_body_never_promote(self):
        self.target.with_suffix(".partial").write_bytes(b"old")
        with patch("urllib.request.urlopen", return_value=Response(b"x", status=206)):
            with self.assertRaisesRegex(ValueError, "range"):
                self.download()
        self.target.with_suffix(".partial").unlink()
        with patch("urllib.request.urlopen", return_value=Response(self.data + b"extra")):
            with self.assertRaisesRegex(ValueError, "exceeded"):
                self.download()
        self.assertFalse(self.target.exists())

    def test_bad_digest_or_short_transfer_is_not_marked_complete(self):
        for body, error in ((self.data[:-1], RuntimeError), (b"x" * len(self.data), ValueError)):
            with self.subTest(body=body):
                self.target.with_suffix(".partial").unlink(missing_ok=True)
                with patch("urllib.request.urlopen", return_value=Response(body)):
                    with self.assertRaises(error):
                        self.download()
                self.assertFalse(self.target.exists())

    def test_existing_unrelated_file_is_preserved_without_network(self):
        self.target.write_bytes(b"research checkpoint")
        with patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "preserved"):
                self.download()
            network.assert_not_called()
        self.assertEqual(self.target.read_bytes(), b"research checkpoint")

    def test_symlink_and_hardlink_outputs_are_refused(self):
        original = self.target.with_name("original")
        original.write_bytes(self.data)
        for link in (lambda: self.target.symlink_to(original), lambda: self.target.hardlink_to(original)):
            link()
            with self.assertRaisesRegex(ValueError, "links"):
                self.download()
            self.target.unlink()
        self.assertEqual(original.read_bytes(), self.data)

    def test_expired_budget_does_not_start_network(self):
        with patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(RuntimeError, "time budget"):
                _download("https://example.test", self.target, len(self.data), self.digest, 0)
            network.assert_not_called()

    def test_setup_marks_only_complete_verified_downloads(self):
        specs = {"fixture": (len(self.data), self.digest)}
        with patch("silivra.provision._SPECS", specs), \
                patch("urllib.request.urlopen", return_value=Response(self.data)):
            result = setup_pretrained(self.target)
        self.assertEqual(result["status"], "verified")
        self.assertTrue((self.target / "silivra-pretrained.json").is_file())
        self.assertEqual((self.target / "fixture").read_bytes(), self.data)

    def test_setup_refuses_original_run_before_any_download_or_marker(self):
        self.target.mkdir()
        original = self.target / "best.json"
        original.write_text('{"checkpoint":"original"}')
        with patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "unrelated"):
                setup_pretrained(self.target)
            network.assert_not_called()
        self.assertEqual(original.read_text(), '{"checkpoint":"original"}')
        self.assertFalse((self.target / "silivra-pretrained.json").exists())

    def test_profiles_pin_distinct_lineages_and_bounded_directories(self):
        original = model_profile("qwen3-8b")
        trial = model_profile("qwen3.5-9b")
        self.assertNotEqual(original["run"], trial["run"])
        self.assertEqual(trial["parameter_count"], 8953803264)
        self.assertEqual(trial["original_model_id"], "Qwen/Qwen3.5-9B")
        self.assertEqual(len(trial["revision"]), 40)
        self.assertEqual(trial["download_limit_bytes"], 6 * 1024**3)
        trial["parameter_count"] = 1
        self.assertEqual(model_profile("qwen3.5-9b")["parameter_count"], 8953803264)

    def test_other_profile_manifest_is_preserved_before_network(self):
        self.target.mkdir()
        metadata = model_profile("qwen3-8b")
        marker = self.target / "silivra-pretrained.json"
        import json
        original = json.dumps({"model_id": metadata["model_id"], "revision": metadata["revision"]})
        marker.write_text(original)
        with patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "another profile"):
                setup_pretrained(self.target, profile="qwen3.5-9b")
            network.assert_not_called()
        self.assertEqual(marker.read_text(), original)
        self.assertEqual(len(list(self.target.iterdir())), 1)

    def test_invalid_profile_does_not_create_directory(self):
        with patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "Unknown"):
                setup_pretrained(self.target, profile="arbitrary-provider/model")
            network.assert_not_called()
        self.assertFalse(self.target.exists())

    def test_setup_checks_existing_hashes_before_writing_any_download(self):
        self.target.mkdir()
        (self.target / "config.json").write_text("unrelated model")
        with patch("urllib.request.urlopen") as network:
            with self.assertRaisesRegex(ValueError, "preserved"):
                setup_pretrained(self.target)
            network.assert_not_called()
        self.assertEqual(list(self.target.iterdir()), [self.target / "config.json"])


if __name__ == "__main__":
    unittest.main()

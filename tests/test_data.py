"""Small offline tests for bounded data preparation and token sampling."""

from __future__ import annotations

import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from silivra.data import DATASET_REVISION, TokenDataset, load_tokenizer, prepare_data


MARKER = "<|endoftext|>"


class FakeResponse(io.BytesIO):
    def __init__(self, body: bytes, url: str = "https://cdn.example.test/dataset.txt"):
        super().__init__(body)
        self.url = url
        self.bytes_read = 0

    def geturl(self):
        return self.url

    def read(self, size=-1):
        if size < 0:
            raise AssertionError("A bounded downloader must request a read size")
        result = super().read(size)
        self.bytes_read += len(result)
        return result


def corpus(*stories: str) -> bytes:
    return (f"\n{MARKER}\n".join(stories) + f"\n{MARKER}\n").encode("utf-8")


class PrepareDataTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def prepare(self, destination, train, valid, **kwargs):
        responses = []
        requests = []

        def open_fixture(request, timeout):
            self.assertEqual(timeout, 30)
            self.assertIn(DATASET_REVISION, request.full_url)
            self.assertTrue(request.get_header("Range").startswith("bytes=0-"))
            requests.append(request.full_url)
            response = FakeResponse(train if request.full_url.endswith("TinyStories-train.txt") else valid)
            responses.append(response)
            return response

        with patch("silivra.data.urlopen", side_effect=open_fixture), patch("silivra.data._READ_CHUNK_BYTES", 7):
            manifest = prepare_data(destination, **kwargs)
        return manifest, requests, responses

    def test_official_splits_deduplication_and_complete_marker_boundaries(self):
        train = corpus("The fox smiled.", "The fox smiled.", "A cat slept.", "Café birds sang.", "DO NOT USE")
        valid = corpus("\r\nThe fox smiled.\r\n", "A quokka jumped.", "A quokka jumped.", "Rain fell.", "IGNORED")
        destination = self.root / "data"
        manifest, requests, responses = self.prepare(
            destination, train, valid, train_stories=3, valid_stories=2, vocab_size=300
        )
        self.assertEqual(len(requests), 2)
        self.assertTrue(requests[0].endswith("TinyStories-train.txt"))
        self.assertTrue(requests[1].endswith("TinyStories-valid.txt"))
        self.assertEqual(manifest["dataset"]["revision"], DATASET_REVISION)
        self.assertEqual(manifest["dataset"]["license"], "CDLA-Sharing-1.0")
        self.assertEqual(manifest["tokenizer"]["trained_on"], "train only")
        self.assertEqual(manifest["splits"]["train"]["stories"], 3)
        self.assertEqual(manifest["splits"]["train"]["duplicates_skipped"], 1)
        self.assertEqual(manifest["splits"]["valid"]["stories"], 2)
        self.assertEqual(manifest["splits"]["valid"]["train_overlap_skipped"], 1)
        self.assertEqual(manifest["splits"]["valid"]["duplicates_skipped"], 1)
        for index, name in enumerate(("train", "valid")):
            split = manifest["splits"][name]
            source = (destination / split["source_path"]).read_bytes()
            self.assertTrue(source.endswith(MARKER.encode()))
            self.assertNotIn(b"DO NOT USE", source)
            self.assertNotIn(b"IGNORED", source)
            self.assertEqual(split["source_bytes"], len(source))
            self.assertEqual(split["bytes_downloaded"], responses[index].bytes_read)
            self.assertLessEqual(split["bytes_downloaded"] - len(source), 6)
            tokens = np.load(destination / split["path"])
            self.assertEqual(tokens.dtype, np.uint16)
            self.assertEqual(len(tokens), split["tokens"])
            self.assertEqual(np.count_nonzero(tokens == manifest["tokenizer"]["bos_id"]), split["stories"])
            self.assertEqual(np.count_nonzero(tokens == manifest["tokenizer"]["eos_id"]), split["stories"])
            self.assertLess(int(tokens.max()), manifest["tokenizer"]["vocab_size"])
        self.assertEqual(json.loads((destination / "manifest.json").read_text()), manifest)
        tokenizer = load_tokenizer(destination / "tokenizer.json")
        self.assertEqual(tokenizer.decode(tokenizer.encode("Café 🦊").ids), "Café 🦊")

    def test_tokenizer_is_reproducible_and_validation_never_trains_it(self):
        train = corpus("The fox smiled and smiled.", "The cat slept and slept.")
        first, _, _ = self.prepare(
            self.root / "first", train, corpus("UNIQUEVALIDATIONTOKEN " * 5),
            train_stories=2, valid_stories=1, vocab_size=300,
        )
        second, _, _ = self.prepare(
            self.root / "second", train, corpus("completely different validation words " * 5),
            train_stories=2, valid_stories=1, vocab_size=300,
        )
        self.assertEqual(first["tokenizer"]["sha256"], second["tokenizer"]["sha256"])
        self.assertEqual(first["splits"]["train"]["sha256"], second["splits"]["train"]["sha256"])
        self.assertNotEqual(first["splits"]["valid"]["sha256"], second["splits"]["valid"]["sha256"])

    def test_download_limit_and_incomplete_story_leave_no_partial_destination(self):
        destination = self.root / "data"
        with patch("silivra.data._MAX_TRAIN_BYTES", 20), self.assertRaisesRegex(RuntimeError, "download limit"):
            self.prepare(destination, b"a story without any complete delimiter" * 10, corpus("valid"),
                         train_stories=1, valid_stories=1, vocab_size=260)
        self.assertFalse(destination.exists())
        self.assertEqual(list(self.root.iterdir()), [])
        with self.assertRaisesRegex(RuntimeError, "upstream ended"):
            self.prepare(destination, b"an unfinished story", corpus("valid"),
                         train_stories=1, valid_stories=1, vocab_size=260)
        self.assertFalse(destination.exists())

    def test_existing_data_and_invalid_inputs_are_preserved(self):
        destination = self.root / "data"
        destination.mkdir()
        (destination / "important.txt").write_text("keep")
        with patch("silivra.data.urlopen") as network, self.assertRaises(FileExistsError):
            prepare_data(destination)
        network.assert_not_called()
        self.assertEqual((destination / "important.txt").read_text(), "keep")
        for arguments in ({"train_stories": 0}, {"valid_stories": True}, {"vocab_size": 259}, {"vocab_size": 65537}):
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                prepare_data(self.root / "new", **arguments)

    def test_non_https_redirect_and_invalid_utf8_are_rejected(self):
        with patch("silivra.data.urlopen", return_value=FakeResponse(corpus("story"), "http://example.test/data")):
            with self.assertRaisesRegex(RuntimeError, "non-HTTPS"):
                prepare_data(self.root / "bad-redirect", 1, 1, 260)
        with self.assertRaisesRegex(RuntimeError, "UTF-8"):
            self.prepare(self.root / "bad-text", b"\xff" + MARKER.encode(), corpus("valid"),
                         train_stories=1, valid_stories=1, vocab_size=260)
        self.assertEqual(list(self.root.iterdir()), [])


class TokenDatasetTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)

    def save(self, values, name="tokens.npy"):
        path = self.root / name
        np.save(path, values)
        return path

    def test_reproducible_disjoint_shifted_windows(self):
        path = self.save(np.arange(40, dtype=np.uint16))
        dataset = TokenDataset(path, seq_len=4, vocab_size=40)
        self.assertEqual(dataset.num_tokens, 40)
        self.assertEqual(len(dataset), 8)
        first = dataset.batch(np.random.default_rng(123), 8)
        second = dataset.batch(np.random.default_rng(123), 8)
        np.testing.assert_array_equal(first[0], second[0])
        np.testing.assert_array_equal(first[1], second[1])
        x, y = first
        self.assertEqual(x.shape, (8, 4))
        self.assertEqual(x.dtype, np.int32)
        np.testing.assert_array_equal(y, x + 1)
        self.assertEqual(len(np.unique(np.concatenate([x[:, :1], y], axis=1))), 40)

    def test_invalid_shape_dtype_length_ids_and_batch(self):
        cases = (
            (np.zeros((2, 4), dtype=np.uint16), 2, "one-dimensional"),
            (np.arange(10, dtype=np.int32), 2, "uint16"),
            (np.arange(2, dtype=np.uint16), 2, "at least 3"),
            (np.arange(10, dtype=np.uint16), 2, "outside the vocabulary"),
        )
        for index, (values, seq_len, error) in enumerate(cases):
            with self.subTest(index=index), self.assertRaisesRegex(ValueError, error):
                TokenDataset(self.save(values, f"case-{index}.npy"), seq_len, vocab_size=5)
        dataset = TokenDataset(self.save(np.arange(10, dtype=np.uint16)), 2)
        with self.assertRaisesRegex(ValueError, "distinct windows"):
            dataset.batch(np.random.default_rng(1), 4)
        with self.assertRaises(TypeError):
            dataset.batch(1, 1)
        with self.assertRaises(ValueError):
            dataset.batch(np.random.default_rng(1), 0)

    def test_manifest_vocabulary_is_checked_automatically(self):
        path = self.save(np.array([0, 1, 10], dtype=np.uint16))
        (self.root / "manifest.json").write_text(json.dumps({"tokenizer": {"vocab_size": 10}}))
        with self.assertRaisesRegex(ValueError, "outside the vocabulary"):
            TokenDataset(path, 2)


if __name__ == "__main__":
    unittest.main()

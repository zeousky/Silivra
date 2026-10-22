"""Bounded, reproducible TinyStories preparation with an original tokenizer.

Only the selected prefix of each official split is downloaded. No pretrained
weights or tokenizer are fetched, and the validation split never trains BPE.
"""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import shutil
import tempfile
from urllib.request import Request, urlopen

import numpy as np
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers


DATASET_REVISION = "f54c09fd23315a6f9c86f9dc80f725de7d8f9c64"
DATASET_URL = "https://huggingface.co/datasets/roneneldan/TinyStories"
SPECIAL_TOKENS = ("<pad>", "<bos>", "<eos>", "<unk>")
_END_OF_STORY = b"<|endoftext|>"
_READ_CHUNK_BYTES = 64 * 1024
_MAX_TRAIN_BYTES = 128 * 1024 * 1024
_MAX_VALID_BYTES = 32 * 1024 * 1024
_TIMEOUT_SECONDS = 30


@dataclass
class _DownloadedSplit:
    stories: list[str]
    hashes: set[str]
    metadata: dict


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"{name} must be a positive integer")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _download_split(
    destination: Path,
    split: str,
    count: int,
    excluded_hashes: set[str],
    max_bytes: int,
) -> _DownloadedSplit:
    """Read complete stories, stopping without downloading the entire corpus."""
    upstream_name = "train" if split == "train" else "valid"
    url = f"{DATASET_URL}/resolve/{DATASET_REVISION}/TinyStories-{upstream_name}.txt"
    request = Request(
        url,
        headers={
            "User-Agent": "Silivra/0.1 (bounded research dataset preparation)",
            "Accept-Encoding": "identity",
            "Range": f"bytes=0-{max_bytes - 1}",
        },
    )
    stories: list[str] = []
    hashes: set[str] = set()
    pending = bytearray()
    downloaded = 0
    candidates = 0
    duplicates = 0
    excluded = 0
    empty = 0
    source_bytes = 0

    with urlopen(request, timeout=_TIMEOUT_SECONDS) as response, destination.open("wb") as source:
        # urllib follows the dataset's CDN redirects. The eventual transport must
        # remain HTTPS; signed CDN query strings are deliberately not persisted.
        if not response.geturl().startswith("https://"):
            raise RuntimeError(f"Refusing a non-HTTPS redirect for {split}")
        if getattr(response, "status", None) == 206:
            content_range = response.headers.get("Content-Range", "")
            if not content_range.startswith("bytes 0-"):
                raise RuntimeError(f"{split}: server returned a range that does not start at byte zero")
        while len(stories) < count:
            remaining = max_bytes - downloaded
            if remaining <= 0:
                raise RuntimeError(
                    f"{split}: reached the {max_bytes:,}-byte download limit after "
                    f"{len(stories):,}/{count:,} unique complete stories"
                )
            chunk = response.read(min(_READ_CHUNK_BYTES, remaining))
            if not chunk:
                raise RuntimeError(
                    f"{split}: upstream ended after {len(stories):,}/{count:,} "
                    "unique complete stories"
                )
            if len(chunk) > remaining:
                raise RuntimeError(f"{split}: response exceeded the download limit")
            source.write(chunk)
            downloaded += len(chunk)
            pending.extend(chunk)

            while len(stories) < count:
                marker = pending.find(_END_OF_STORY)
                if marker < 0:
                    break
                raw_story = bytes(pending[:marker])
                del pending[: marker + len(_END_OF_STORY)]
                source_bytes = downloaded - len(pending)
                candidates += 1
                try:
                    story = raw_story.decode("utf-8").replace("\r\n", "\n").replace("\r", "\n").strip()
                except UnicodeDecodeError as error:
                    raise RuntimeError(f"{split}: upstream contains invalid UTF-8") from error
                if not story:
                    empty += 1
                    continue
                story_hash = hashlib.sha256(story.encode("utf-8")).hexdigest()
                if story_hash in excluded_hashes:
                    excluded += 1
                    continue
                if story_hash in hashes:
                    duplicates += 1
                    continue
                hashes.add(story_hash)
                stories.append(story)

        # A read can contain part of the following story. Keep only the exact
        # upstream prefix through the final complete selected story's delimiter.
        source.truncate(source_bytes)

    return _DownloadedSplit(
        stories,
        hashes,
        {
            "source_url": url,
            "source_sha256": _sha256_file(destination),
            "source_bytes": source_bytes,
            "bytes_downloaded": downloaded,
            "max_download_bytes": max_bytes,
            "stories": len(stories),
            "candidates": candidates,
            "duplicates_skipped": duplicates,
            "train_overlap_skipped": excluded,
            "empty_stories_skipped": empty,
            "selected_stories_sha256": hashlib.sha256(
                "\n".join(sorted(hashes)).encode("ascii")
            ).hexdigest(),
        },
    )


def load_tokenizer(path: Path | str) -> Tokenizer:
    """Load the locally trained tokenizer, checking its required special tokens."""
    tokenizer = Tokenizer.from_file(str(path))
    ids = [tokenizer.token_to_id(token) for token in SPECIAL_TOKENS]
    if any(value is None for value in ids) or len(set(ids)) != len(SPECIAL_TOKENS):
        raise ValueError("Tokenizer is missing distinct Silivra special tokens")
    if tokenizer.get_vocab_size() > np.iinfo(np.uint16).max + 1:
        raise ValueError("Tokenizer vocabulary does not fit uint16 token storage")
    return tokenizer


def _write_tokens(path: Path, stories: list[str], tokenizer: Tokenizer) -> dict:
    bos_id = tokenizer.token_to_id("<bos>")
    eos_id = tokenizer.token_to_id("<eos>")
    vocab_size = tokenizer.get_vocab_size()
    parts: list[np.ndarray] = []
    for story in stories:
        ids = [bos_id, *tokenizer.encode(story, add_special_tokens=False).ids, eos_id]
        if min(ids) < 0 or max(ids) >= vocab_size:
            raise ValueError("Tokenizer emitted a token outside its vocabulary")
        parts.append(np.asarray(ids, dtype=np.uint16))
    tokens = np.concatenate(parts)
    np.save(path, tokens, allow_pickle=False)
    return {"path": path.name, "tokens": int(tokens.size), "sha256": _sha256_file(path)}


def prepare_data(
    output_dir: Path,
    train_stories: int = 30_000,
    valid_stories: int = 1_000,
    vocab_size: int = 4_096,
) -> dict:
    """Prepare a bounded corpus and train ByteLevel BPE entirely from scratch.

    The destination must be absent or empty. Work happens in a sibling temporary
    directory, so failed network requests or tokenization leave no partial data
    at the requested destination and never replace an existing prepared corpus.
    """
    _positive_int(train_stories, "train_stories")
    _positive_int(valid_stories, "valid_stories")
    _positive_int(vocab_size, "vocab_size")
    if not 260 <= vocab_size <= 65_536:
        raise ValueError("vocab_size must be between 260 and 65536")

    output_dir = Path(output_dir).expanduser().absolute()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError(f"Prepared data already exists at {output_dir}; choose a new empty directory")
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.preparing-", dir=output_dir.parent))
    try:
        sources = staging / "sources"
        sources.mkdir()
        train = _download_split(sources / "train.txt", "train", train_stories, set(), _MAX_TRAIN_BYTES)
        valid = _download_split(sources / "valid.txt", "valid", valid_stories, train.hashes, _MAX_VALID_BYTES)

        tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
        tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
        tokenizer.decoder = decoders.ByteLevel()
        trainer = trainers.BpeTrainer(
            vocab_size=vocab_size,
            min_frequency=2,
            special_tokens=list(SPECIAL_TOKENS),
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
            show_progress=False,
        )
        tokenizer.train_from_iterator(train.stories, trainer=trainer, length=len(train.stories))
        tokenizer_path = staging / "tokenizer.json"
        tokenizer.save(str(tokenizer_path))
        # Reload the on-disk artifact so every downstream token uses precisely the
        # tokenizer that accompanies the data and later checkpoints.
        tokenizer = load_tokenizer(tokenizer_path)
        split_metadata = {}
        for name, downloaded_split in (("train", train), ("valid", valid)):
            split_metadata[name] = {
                **downloaded_split.metadata,
                "source_path": f"sources/{name}.txt",
                **_write_tokens(staging / f"{name}.npy", downloaded_split.stories, tokenizer),
            }
        manifest = {
            "schema_version": 1,
            "dataset": {
                "name": "roneneldan/TinyStories",
                "url": DATASET_URL,
                "revision": DATASET_REVISION,
                "license": "CDLA-Sharing-1.0",
                "license_url": "https://cdla.dev/sharing-1-0/",
            },
            "settings": {
                "train_stories": train_stories,
                "valid_stories": valid_stories,
                "requested_vocab_size": vocab_size,
                "tokenizer_algorithm": "ByteLevel BPE",
                "tokenizer_min_frequency": 2,
                "story_normalization": "CRLF and CR to LF; strip outer whitespace",
                "deduplication": "SHA256 of normalized UTF-8 story; unique within and across splits",
                "token_dtype": "uint16",
                "story_boundaries": "BOS before and EOS after each story",
            },
            "versions": {
                "python": platform.python_version(),
                "tokenizers": version("tokenizers"),
                "numpy": np.__version__,
            },
            "tokenizer": {
                "path": "tokenizer.json",
                "sha256": _sha256_file(tokenizer_path),
                "vocab_size": tokenizer.get_vocab_size(),
                **{token[1:-1] + "_id": tokenizer.token_to_id(token) for token in SPECIAL_TOKENS},
                "trained_on": "train only",
            },
            "splits": split_metadata,
        }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        # Recheck immediately before publishing; preserve a destination created
        # by another preparation while this one was in progress.
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise FileExistsError(f"Destination became nonempty during preparation: {output_dir}")
        os.replace(staging, output_dir)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class TokenDataset:
    """Memory-mapped uint16 tokens and distinct random next-token windows."""

    def __init__(self, path: Path | str, seq_len: int, *, vocab_size: int | None = None):
        _positive_int(seq_len, "seq_len")
        self.path = Path(path)
        self.seq_len = seq_len
        self.tokens = np.load(self.path, mmap_mode="r", allow_pickle=False)
        if not isinstance(self.tokens, np.ndarray) or self.tokens.ndim != 1:
            raise ValueError("Token file must be a one-dimensional NumPy array")
        if self.tokens.dtype.kind != "u" or self.tokens.dtype.itemsize != 2:
            raise ValueError("Token file must contain uint16 token IDs")
        self.num_tokens = int(self.tokens.size)
        self.num_windows = self.num_tokens // (seq_len + 1)
        if self.num_windows < 1:
            raise ValueError(f"Token file needs at least {seq_len + 1} tokens")
        if vocab_size is None and (self.path.parent / "manifest.json").exists():
            manifest = json.loads((self.path.parent / "manifest.json").read_text(encoding="utf-8"))
            vocab_size = manifest["tokenizer"]["vocab_size"]
        if vocab_size is not None:
            _positive_int(vocab_size, "vocab_size")
            if vocab_size > 65_536:
                raise ValueError("vocab_size exceeds uint16 storage")
            if int(self.tokens.max()) >= vocab_size:
                raise ValueError("Token file contains IDs outside the vocabulary")

    def __len__(self) -> int:
        return self.num_windows

    def batch(self, rng: np.random.Generator, batch_size: int) -> tuple[np.ndarray, np.ndarray]:
        _positive_int(batch_size, "batch_size")
        if not isinstance(rng, np.random.Generator):
            raise TypeError("rng must be a numpy.random.Generator")
        if batch_size > self.num_windows:
            raise ValueError(
                f"Requested {batch_size} distinct windows, but only {self.num_windows} fit; "
                "reduce batch_size/seq_len or prepare more stories"
            )
        starts = rng.choice(self.num_windows, size=batch_size, replace=False) * (self.seq_len + 1)
        positions = starts[:, None] + np.arange(self.seq_len)[None, :]
        x = np.asarray(self.tokens[positions], dtype=np.int32)
        y = np.asarray(self.tokens[positions + 1], dtype=np.int32)
        return x, y

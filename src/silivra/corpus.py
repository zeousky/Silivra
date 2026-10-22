"""Bounded, immutable public instruction import and reproducible corpus splitting.

The importer retains the original tokenizer and converts responses into Silivra's
JSON action protocol. It never downloads weights, reads chats, or fits a tokenizer.
"""

from __future__ import annotations

from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile
import unicodedata
import urllib.request

from .agent import POLICY
from .checkpoint import sha256
from .data import load_tokenizer
from .instructions import _check_splits, encode_turns, validate_record


DOLLY_REVISION = "bdd27f4d94b9c1f951818a7da7fd7aeea5dbff1a"
DOLLY_SHA256 = "2df9083338b4abd6bceb5635764dab5d833b393b55759dffb0959b6fcbf794ec"
DOLLY_BYTES = 13_085_339
DOLLY_URL = ("https://huggingface.co/datasets/databricks/databricks-dolly-15k/resolve/"
             + DOLLY_REVISION + "/databricks-dolly-15k.jsonl")
DOLLY_SOURCE = {
    "name": "databricks/databricks-dolly-15k (Silivra JSON-action conversion)",
    "author": "Databricks employees; Wikipedia editors and contributors where indicated upstream",
    "license": "CC-BY-SA-3.0",
    "url": "https://huggingface.co/datasets/databricks/databricks-dolly-15k/tree/" + DOLLY_REVISION,
    "revision": DOLLY_REVISION,
}
_MAX_BYTES = 64 * 1024 * 1024


def _dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _normal(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def _hash(value):
    return hashlib.sha256(_dump(value).encode("utf-8")).hexdigest()


def download_dolly(destination: Path, *, max_bytes: int = 16 * 1024 * 1024) -> dict:
    """Download one pinned 13 MB file, verifying its upstream SHA-256 before use.

    Transfers have a strict byte ceiling, per-read timeout, no credentials, and
    atomic publication. An existing cache is verified rather than overwritten.
    """
    if type(max_bytes) is not int or not DOLLY_BYTES <= max_bytes <= _MAX_BYTES:
        raise ValueError(f"max_bytes must be between {DOLLY_BYTES} and {_MAX_BYTES}")
    destination = Path(destination)
    if destination.exists():
        if not destination.is_file() or destination.stat().st_size != DOLLY_BYTES or sha256(destination) != DOLLY_SHA256:
            raise ValueError("Cached Dolly file does not match the pinned source; choose a new cache path")
        return {"url": DOLLY_URL, "revision": DOLLY_REVISION, "sha256": DOLLY_SHA256,
                "bytes": DOLLY_BYTES, "license": "CC-BY-SA-3.0", "cached": True}
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".dolly-download-", dir=destination.parent)
    total, digest = 0, hashlib.sha256()
    try:
        request = urllib.request.Request(DOLLY_URL, headers={"User-Agent": "Silivra-corpus/2"})
        with os.fdopen(descriptor, "wb") as output, urllib.request.urlopen(request, timeout=30) as response:
            if not response.geturl().startswith("https://"):
                raise ValueError("Dolly download redirect must remain HTTPS")
            announced = response.headers.get("Content-Length")
            if announced is not None and int(announced) > max_bytes:
                raise ValueError("Dolly Content-Length exceeds max_bytes")
            while chunk := response.read(min(64 * 1024, max_bytes - total + 1)):
                total += len(chunk)
                if total > max_bytes:
                    raise ValueError("Dolly transfer exceeded max_bytes")
                digest.update(chunk)
                output.write(chunk)
            output.flush()
            os.fsync(output.fileno())
        if total != DOLLY_BYTES or digest.hexdigest() != DOLLY_SHA256:
            raise ValueError("Dolly source size or SHA-256 differs from pinned source")
        # Avoid replacing an existing source if another process published first.
        try:
            os.link(temporary, destination)
        except FileExistsError:
            if destination.stat().st_size != DOLLY_BYTES or sha256(destination) != DOLLY_SHA256:
                raise ValueError("Dolly cache changed during download")
        return {"url": DOLLY_URL, "revision": DOLLY_REVISION, "sha256": digest.hexdigest(),
                "bytes": total, "license": "CC-BY-SA-3.0", "cached": False}
    finally:
        Path(temporary).unlink(missing_ok=True)


def read_dolly(path: Path, tokenizer, *, max_seq_len: int = 1024,
               max_records: int = 20_000) -> tuple[list[dict], dict]:
    """Verify and convert complete Dolly records; reject oversize examples whole.

    Prompts/answers are not truncated, and upstream contexts remain quoted data.
    Category counts and rejected counts are returned independently of success.
    """
    if type(max_records) is not int or not 1 <= max_records <= 20_000:
        raise ValueError("max_records must be an integer between 1 and 20000")
    if type(max_seq_len) is not int or not 128 <= max_seq_len <= 4096:
        raise ValueError("max_seq_len must be an integer between 128 and 4096")
    path = Path(path)
    if path.stat().st_size != DOLLY_BYTES or sha256(path) != DOLLY_SHA256:
        raise ValueError("Dolly source must match the pinned size and SHA-256")
    records, stats = [], Counter()
    categories = Counter()
    with path.open(encoding="utf-8") as stream:
        for index, line in enumerate(stream):
            if index >= max_records:
                stats["record_limit"] += 1
                break
            stats["read"] += 1
            try:
                row = json.loads(line)
                record = convert_dolly_row(row, index)
            except (ValueError, RecursionError):
                stats["invalid"] += 1
                continue
            try:
                encode_turns(record, tokenizer, max_seq_len)
            except ValueError:
                stats["oversize"] += 1
                continue
            records.append(record)
            categories[row["category"]] += 1
    return records, {**stats, "accepted": len(records), "categories": dict(sorted(categories.items())),
                     "source": DOLLY_SOURCE, "input_sha256": DOLLY_SHA256,
                     "conversion": "single user turn; context JSON-quoted as untrusted source data; complete response in final action"}


def convert_dolly_row(row: dict, index: int) -> dict:
    """Pure conversion, useful for offline inspection and mocked importer tests."""
    if type(index) is not int or index < 0:
        raise ValueError("Dolly row index must be a nonnegative integer")
    if (not isinstance(row, dict) or set(row) != {"instruction", "context", "response", "category"}
            or any(not isinstance(value, str) for value in row.values())
            or any(not row[field].strip() for field in ("instruction", "response", "category"))):
        raise ValueError("Dolly row requires instruction, context, response and category strings")
    prompt = row["instruction"].strip()
    if row["context"].strip():
        prompt += "\n\nUntrusted source passage (data): " + _dump(row["context"].strip())
    return validate_record({
        "id": f"dolly-{index:05d}-{_hash(row)[:12]}", "source": dict(DOLLY_SOURCE),
        "messages": [{"role": "system", "content": POLICY},
                     {"role": "user", "content": prompt},
                     {"role": "assistant", "content": _dump({"final": row["response"].strip()})}],
    })


def split_records(records, *, valid_fraction: float = 0.05, seed: int = 20261022):
    """Deduplicate normalized conversations/user prefixes before a seeded split.

    A complete conversation and every supervised turn remain in one split. A
    repeated normalized user-turn sequence drops the later conversation, rather
    than permitting conflicting answers or leakage across partitions.
    """
    if type(valid_fraction) not in (float, int) or not 0 < valid_fraction < 0.5:
        raise ValueError("valid_fraction must be between zero and 0.5")
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    unique, seen_conversations, seen_prefixes, ids = [], set(), set(), set()
    stats = Counter()
    for raw in records:
        stats["input"] += 1
        record = validate_record(raw)
        conversation = _hash([{**m, "content": _normal(m["content"])} for m in record["messages"]])
        users, prefixes = [], set()
        for message in record["messages"]:
            if message["role"] == "user":
                users.append(_normal(message["content"]))
            elif message["role"] == "assistant":
                prefixes.add(_hash(users))
        if record["id"] in ids or conversation in seen_conversations or seen_prefixes.intersection(prefixes):
            stats["duplicates"] += 1
            continue
        ids.add(record["id"])
        seen_conversations.add(conversation)
        seen_prefixes.update(prefixes)
        unique.append(record)
    if len(unique) < 2:
        raise ValueError("At least two distinct conversations are required")
    random.Random(seed).shuffle(unique)
    valid_count = max(1, round(len(unique) * valid_fraction))
    valid, train = unique[:valid_count], unique[valid_count:]
    _check_splits(train, valid)
    return train, valid, {**stats, "train": len(train), "valid": len(valid), "seed": seed,
                          "valid_fraction": valid_fraction,
                          "method": "global NFC/whitespace user-prefix deduplication; seeded conversation shuffle"}


def write_corpus(output: Path, records, *, tokenizer_path: Path, max_seq_len=1024,
                 valid_fraction=0.05, seed=20261022, provenance=None) -> dict:
    """Publish new JSONL splits/manifest atomically, refusing existing artifacts."""
    if type(max_seq_len) is not int or not 128 <= max_seq_len <= 4096:
        raise ValueError("max_seq_len must be an integer between 128 and 4096")
    output, tokenizer_path = Path(output), Path(tokenizer_path)
    if output.exists():
        raise FileExistsError("Corpus output already exists; use a new run directory")
    tokenizer = load_tokenizer(tokenizer_path)
    accepted, rejected, max_tokens = [], 0, 0
    for raw in records:
        record = validate_record(raw)
        try:
            turns = encode_turns(record, tokenizer, max_seq_len)
        except ValueError:
            rejected += 1
            continue
        max_tokens = max(max_tokens, *(len(ids) - 1 for ids, _ in turns))
        accepted.append(record)
    train, valid, splitting = split_records(accepted, valid_fraction=valid_fraction, seed=seed)
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".corpus-", dir=output.parent))
    try:
        manifest = {"schema_version": 1, "kind": "silivra-corpus-v2", "max_seq_len": max_seq_len,
                    "max_observed_tokens": max_tokens, "tokenizer_sha256": sha256(tokenizer_path),
                    "context_filter": "reject complete conversation when any assistant turn exceeds context",
                    "oversize_rejected": rejected, "splitting": splitting, "splits": {},
                    "provenance": provenance or {},
                    "evaluation_boundary": "held-out corpus split only; fixed benchmark records were not read or fitted"}
        total_bytes = 0
        for label, partition in (("train", train), ("valid", valid)):
            target = staging / f"{label}.jsonl"
            with target.open("w", encoding="utf-8") as stream:
                for record in partition:
                    line = _dump(record) + "\n"
                    total_bytes += len(line.encode("utf-8"))
                    if total_bytes > _MAX_BYTES:
                        raise ValueError("Converted corpus exceeds 64 MiB")
                    stream.write(line)
            manifest["splits"][label] = {"path": target.name, "sha256": sha256(target),
                                        "conversations": len(partition)}
        (staging / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if output.exists():
            raise FileExistsError("Corpus output appeared during preparation")
        os.rename(staging, output)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)

#!/usr/bin/env python3
"""Mix verified local instruction sources with focused core tasks, offline.

The deterministic selection/transformation is Codex-authored MIT code. Imported
records keep their original attribution and licenses; mixing does not relicense
Dolly or TinyStories. No private chats, downloads, or pretrained assets are used.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from silivra.checkpoint import sha256
from silivra.data import load_tokenizer
from silivra.finetune import _manifest
from silivra.instructions import _canonical, _check_splits, _read_records, encode_turns, prepare_instructions


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def prefixes(messages, *, pending=False):
    """Canonical supervised user prefixes; no answer content is inspected."""
    users, result = [], set()
    for message in messages:
        if message["role"] == "user":
            users.append(_canonical(message["content"]))
        elif message["role"] == "assistant" and users:
            result.add(tuple(users))
    if pending and users:
        result.add(tuple(users))
    return result


def group(record):
    source = record["source"]["name"].lower()
    if "dolly" in source:
        return "dolly"
    if "tinystories" in source:
        return "story"
    return "original"


def balanced(records, count, rng):
    """Round-robin original generator families after deterministic shuffling."""
    families = defaultdict(list)
    for record in records:
        identifier = record["id"]
        family = identifier.split("-", 3)[-1] if identifier.startswith("original-v2-") else "other"
        families[family].append(record)
    for rows in families.values():
        rng.shuffle(rows)
    chosen = []
    while len(chosen) < count and any(families.values()):
        for family in sorted(families):
            if families[family] and len(chosen) < count:
                chosen.append(families[family].pop())
    return chosen


def select(records, counts, rng):
    selected = []
    for label, count in counts.items():
        pool = [record for record in records if group(record) == label]
        if label == "original":
            selected.extend(balanced(pool, count, rng))
        else:
            rng.shuffle(pool)
            selected.extend(pool[:count])
    return selected


def mix_records(old_train, old_valid, core_train, core_valid, *, denied=(), seed=20261023):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    rng = random.Random(seed)
    denied = set(denied)
    core_validation_input_count = len(core_valid)
    core_valid = [r for r in core_valid if not prefixes(r["messages"]) & denied]
    core_validation = set().union(*(prefixes(r["messages"]) for r in core_valid))
    # Selection happens once before training, independently of model outcomes.
    core_training = set().union(*(prefixes(r["messages"]) for r in core_train))
    valid_pool = [r for r in old_valid if not prefixes(r["messages"]) &
                  (denied | core_training | core_validation)]
    old_validation = select(valid_pool, {"dolly": 64, "original": 51, "story": 13}, rng)
    valid = [*core_valid, *old_validation]
    validation_prefixes = set().union(*(prefixes(r["messages"]) for r in valid))
    forbidden = denied | validation_prefixes
    retained_core = [r for r in core_train if not prefixes(r["messages"]) & forbidden]
    retained_core_prefixes = set().union(*(prefixes(r["messages"]) for r in retained_core))
    replay_pool = [r for r in old_train if not prefixes(r["messages"]) &
                   (forbidden | retained_core_prefixes)]
    replay = select(replay_pool, {"dolly": 1000, "original": 800, "story": 200}, rng)
    train = [*retained_core, *replay]
    # An exact duplicate source conversation cannot contribute twice.
    unique, seen = [], set()
    for row in train:
        digest = hashlib.sha256(dump(row["messages"]).encode()).hexdigest()
        if digest not in seen:
            unique.append(row)
            seen.add(digest)
    train = unique
    rng.shuffle(train)
    rng.shuffle(valid)
    _check_splits(train, valid)
    return train, valid, {"retained_core": len(retained_core), "core_excluded": len(core_train) - len(retained_core),
        "retained_core_validation": len(core_valid), "core_validation_excluded": core_validation_input_count - len(core_valid),
        "replay": dict(Counter(group(r) for r in replay)),
        "old_validation": dict(Counter(group(r) for r in old_validation)),
        "selection": "fixed-seed shuffle; original-v2 replay round-robin across existing families",
        "exclusion": "all diagnostic and joint-validation supervised user prefixes; replay conflicting with core prompts removed"}


def read_prepared(directory):
    directory = directory.resolve()
    manifest = _manifest(directory)
    records = {}
    for name in ("train", "valid"):
        records[name], _ = _read_records(directory / manifest["splits"][name]["source_path"],
                                          100_000, 64 * 1024 * 1024)
    _check_splits(records["train"], records["valid"])
    return manifest, records


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old", type=Path, default=Path("data/instructions-v2"))
    parser.add_argument("--core", type=Path, default=Path("data/assistant-v3-filtered"))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--prepare", type=Path, required=True)
    parser.add_argument("--exclude-suite", type=Path, action="append", default=[])
    parser.add_argument("--seed", type=int, default=20261023)
    args = parser.parse_args(argv)
    if args.output.exists() or args.prepare.exists():
        parser.error("Output and prepared directories must both be new")
    destinations = (args.output.resolve(), args.prepare.resolve())
    for index, path in enumerate(destinations):
        for other in (*destinations[index + 1:], args.old.resolve(), args.core.resolve()):
            if path.is_relative_to(other) or other.is_relative_to(path):
                parser.error("Inputs and outputs must not overlap")
    old_manifest, old = read_prepared(args.old)
    core_manifest, core = read_prepared(args.core)
    if old_manifest["tokenizer"]["sha256"] != core_manifest["tokenizer"]["sha256"]:
        parser.error("Sources must use the identical original tokenizer")
    denied, suites = set(), []
    for path in args.exclude_suite:
        if path.stat().st_size > 2 * 1024 * 1024:
            parser.error("Exclusion suite exceeds 2 MiB")
        snapshot = path.read_bytes()
        for line in snapshot.decode("utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            messages = case.get("messages", [{"role": "user", "content": case.get("prompt", "")}])
            denied.update(prefixes(messages, pending=True))
        suites.append({"name": path.name, "sha256": hashlib.sha256(snapshot).hexdigest(),
                       "use": "user prefix exclusion only; expected answers not inspected"})
    train, valid, selection = mix_records(old["train"], old["valid"], core["train"], core["valid"],
                                          denied=denied, seed=args.seed)
    tokenizer = load_tokenizer(args.core / core_manifest["tokenizer"]["path"])
    max_seq_len = max(old_manifest["max_seq_len"], core_manifest["max_seq_len"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".rehearsal-", dir=args.output.parent))
    try:
        manifest = {"schema_version": 1, "kind": "silivra-rehearsal-v4", "seed": args.seed,
            "generator_sha256": sha256(Path(__file__)), "tokenizer_sha256": core_manifest["tokenizer"]["sha256"],
            "transformation": {"author": "Silivra project with Codex; deterministic offline selection", "license": "MIT",
                "boundary": "Transformation code is MIT; every source record retains its original license and attribution."},
            "inputs": {"old_manifest_sha256": sha256(args.old / "manifest.json"),
                "core_manifest_sha256": sha256(args.core / "manifest.json"),
                "artifacts": {label: {split: {key: value for key, value in source["splits"][split].items()
                    if key in {"sha256", "source_sha256"}} for split in ("train", "valid")}
                    for label, source in (("old", old_manifest), ("core", core_manifest))}},
            "selection": selection, "excluded_suites": suites, "splits": {},
            "evaluation_boundary": "Reuses development diagnostics for exclusion; joint loss validation is narrow and not a fresh independent benchmark."}
        for split, records in (("train", train), ("valid", valid)):
            turns = [turn for row in records for turn in encode_turns(row, tokenizer, max_seq_len)]
            path = staging / f"{split}.jsonl"
            path.write_text("".join(dump(row) + "\n" for row in records), encoding="utf-8")
            sources = {dump(row["source"]): row["source"] for row in records}
            manifest["splits"][split] = {"records": len(records), "sha256": sha256(path),
                "sources": list(sources.values()), "source_counts": dict(Counter(row["source"]["name"] for row in records)),
                "supervised_tokens": sum(sum(mask) for _, mask in turns),
                "max_tokens": max(len(ids) - 1 for ids, _ in turns)}
        (staging / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if args.output.exists():
            raise FileExistsError("Output appeared during preparation")
        os.rename(staging, args.output)
        prepare_instructions(args.output / "train.jsonl", args.output / "valid.jsonl",
                             args.core / core_manifest["tokenizer"]["path"], args.prepare, max_seq_len=max_seq_len)
        print(json.dumps(manifest, indent=2))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()

"""Local, provenance-carrying instruction data and a shared escaped chat format.

No data is downloaded and no tokenizer vocabulary is changed here. Each assistant
turn becomes a complete sample: the history is context, the action and EOS are
supervised. Oversized samples fail instead of silently cutting off answers.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import tempfile
import unicodedata

import numpy as np

from .checkpoint import sha256
from .data import load_tokenizer


PROMPT_FORMAT = "silivra-role-json-v1"
_MAX_LINE_BYTES = 256 * 1024
_MAX_OUTPUT_TOKENS = 32_000_000
_TOOL_NAME = re.compile(r"[A-Za-z][A-Za-z0-9_.-]{0,63}\Z")


def _json(value) -> str:
    rendered = json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    # Keep content on one physical role line across parsers and prevent the
    # tokenizer from recognizing reserved spellings inside untrusted text.
    for character in ("<", ">", "\x85", "\u2028", "\u2029"):
        rendered = rendered.replace(character, f"\\u{ord(character):04x}")
    return rendered


def _reject_constant(value):
    raise ValueError(f"Non-finite JSON value is not allowed: {value}")


def _finite_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("Non-finite JSON number is not allowed")
    return parsed


def _unique_object(pairs):
    value = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(f"Duplicate JSON key: {key}")
        value[key] = item
    return value


def parse_action(content: str) -> dict:
    """Parse the exact assistant protocol, rejecting ambiguous or extra fields."""
    if not isinstance(content, str):
        raise ValueError("Assistant content must be a JSON string")
    try:
        action = json.loads(content, parse_constant=_reject_constant, parse_float=_finite_float,
                            object_pairs_hook=_unique_object)
    except (json.JSONDecodeError, RecursionError) as error:
        raise ValueError("Assistant content must be one valid JSON action") from error
    if not isinstance(action, dict):
        raise ValueError("Assistant action must be a JSON object")
    if set(action) == {"final"} and isinstance(action["final"], str) and action["final"].strip():
        return action
    if (
        set(action) == {"tool", "arguments"}
        and isinstance(action["tool"], str)
        and _TOOL_NAME.fullmatch(action["tool"])
        and isinstance(action["arguments"], dict)
    ):
        return action
    raise ValueError('Assistant action must be exactly {"final":"..."} or {"tool":"name","arguments":{...}}')


def _message(message: dict) -> dict:
    if not isinstance(message, dict) or set(message) - {"role", "content", "name"}:
        raise ValueError("Each message accepts only role, content and optional tool name")
    role, content = message.get("role"), message.get("content")
    if role not in {"system", "user", "assistant", "tool"}:
        raise ValueError("Message role must be system, user, assistant or tool")
    if not isinstance(content, str) or not content.strip() or len(content.encode("utf-8")) > 65536:
        raise ValueError("Message content must be nonempty UTF-8 text of at most 65536 bytes")
    if "name" in message and (role != "tool" or not isinstance(message["name"], str)
                              or not _TOOL_NAME.fullmatch(message["name"])):
        raise ValueError("Only tool messages may have a valid tool name")
    if role == "assistant":
        content = _json(parse_action(content))
    result = {"role": role, "content": content}
    if "name" in message:
        result["name"] = message["name"]
    return result


def format_prompt(messages: list[dict], *, add_assistant_prefix: bool = True) -> str:
    """Render plain role headers and JSON-escaped content without special tokens.

    A newline or role marker inside user/tool text remains inside a JSON string,
    so it cannot create another structural role header. This encoding does not
    make untrusted tool text authoritative or prevent semantic prompt injection.
    """
    if not isinstance(messages, (list, tuple)) or len(messages) > 128:
        raise ValueError("messages must be a list of at most 128 messages")
    chunks = []
    for raw in messages:
        message = _message(raw)
        role, content = message["role"], message["content"]
        body = content if role == "assistant" else _json({
            **({"name": message["name"]} if "name" in message else {}), "content": content,
        })
        chunks.append(f"### {role}\n{body}\n\n")
    if add_assistant_prefix:
        chunks.append("### assistant\n")
    return "".join(chunks)


def _text(value, label):
    if not isinstance(value, str) or not value.strip() or len(value) > 2048:
        raise ValueError(f"{label} must be a nonempty string of at most 2048 characters")
    return value


def validate_record(record: dict) -> dict:
    """Validate attribution and a conversation ending at an assistant decision."""
    if not isinstance(record, dict) or set(record) != {"id", "source", "messages"}:
        raise ValueError("Each record must contain exactly id, source and messages")
    identifier = _text(record["id"], "id")
    source = record["source"]
    if (not isinstance(source, dict) or not {"name", "license", "url"} <= set(source)
            or set(source) - {"name", "license", "url", "author", "revision"}):
        raise ValueError("source requires name, license and url; author and revision are optional")
    source = {key: _text(value, f"source.{key}") for key, value in source.items()}
    if not isinstance(record["messages"], list) or not 2 <= len(record["messages"]) <= 128:
        raise ValueError("A conversation must contain 2 to 128 messages")
    messages = [_message(message) for message in record["messages"]]
    expected, pending_tool = "user", None
    for index, message in enumerate(messages):
        role = message["role"]
        if index == 0 and role == "system":
            continue
        if role != expected:
            raise ValueError(f"Message {index}: expected {expected}, got {role}")
        if role == "user":
            expected = "assistant"
        elif role == "tool":
            if message.get("name") != pending_tool:
                raise ValueError("Tool result name must match the preceding assistant tool action")
            expected = "assistant"
        else:
            action = parse_action(message["content"])
            pending_tool = action.get("tool")
            expected = "tool" if pending_tool else "user"
    if messages[-1]["role"] != "assistant":
        raise ValueError("A training conversation must end with an assistant action")
    return {"id": identifier, "source": source, "messages": messages}


def _canonical(value) -> str:
    """Normalize whitespace for leakage detection, independently of metadata."""
    if isinstance(value, str):
        return " ".join(unicodedata.normalize("NFC", value).split())
    return _json(value)


def _digest(value) -> str:
    canonical = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _read_records(path: Path, max_records: int, max_bytes: int) -> tuple[list[dict], str]:
    records, digest = [], hashlib.sha256()
    total = 0
    with path.open("rb") as stream:
        for line_number in range(1, max_records + 2):
            line = stream.readline(_MAX_LINE_BYTES + 1)
            if not line:
                break
            total += len(line)
            if len(line) > _MAX_LINE_BYTES or total > max_bytes:
                raise ValueError(f"{path.name}:{line_number}: instruction data exceeds byte limit")
            if line_number > max_records:
                raise ValueError(f"{path.name}: exceeds max_records={max_records}")
            digest.update(line)
            try:
                raw = json.loads(line, parse_constant=_reject_constant, parse_float=_finite_float,
                                 object_pairs_hook=_unique_object)
                records.append(validate_record(raw))
            except (ValueError, UnicodeError, RecursionError) as error:
                raise ValueError(f"{path.name}:{line_number}: {error}") from error
    if not records:
        raise ValueError(f"{path.name}: instruction split must not be empty")
    return records, digest.hexdigest()


def _check_splits(train: list[dict], valid: list[dict]) -> None:
    seen_ids, seen_conversations, train_prompts = set(), set(), set()
    for split, records in (("train", train), ("valid", valid)):
        prompts = set()
        for record in records:
            if record["id"] in seen_ids:
                raise ValueError(f"Duplicate conversation id: {record['id']}")
            seen_ids.add(record["id"])
            conversation = _digest([
                {**message, "content": (parse_action(message["content"]) if message["role"] == "assistant"
                                        else _canonical(message["content"]))}
                for message in record["messages"]
            ])
            if conversation in seen_conversations:
                raise ValueError(f"Duplicate conversation or train/validation leakage: {record['id']}")
            seen_conversations.add(conversation)
            users = []
            for message in record["messages"]:
                if message["role"] == "user":
                    users.append(_canonical(message["content"]))
                elif message["role"] == "assistant":
                    prompts.add(_digest(users))
        if split == "valid" and train_prompts.intersection(prompts):
            raise ValueError("Train/validation prompt leakage: at least one user-turn sequence occurs in both splits")
        train_prompts = prompts


def encode_turns(record: dict, tokenizer, max_seq_len: int) -> list[tuple[list[int], list[int]]]:
    """Build BOS/context/action/EOS samples and aligned assistant-only masks."""
    samples = []
    bos, eos = tokenizer.token_to_id("<bos>"), tokenizer.token_to_id("<eos>")
    for index, message in enumerate(record["messages"]):
        if message["role"] != "assistant":
            continue
        prefix = format_prompt(record["messages"][:index])
        # Separate encodings preserve the exact inference prefix at the
        # supervision boundary even when BPE could merge across that boundary.
        context = [bos, *tokenizer.encode(prefix, add_special_tokens=False).ids]
        answer = [*tokenizer.encode(message["content"], add_special_tokens=False).ids, eos]
        ids = context + answer
        if len(ids) - 1 > max_seq_len:
            raise ValueError(
                f"{record['id']} assistant turn {index}: {len(ids) - 1} tokens exceeds "
                f"max_seq_len={max_seq_len}; shorten the complete conversation or use a larger model context"
            )
        samples.append((ids, [0] * len(context) + [1] * len(answer)))
    return samples


def prepare_instructions(train_path: Path, valid_path: Path, tokenizer_path: Path,
                         output_dir: Path, *, max_seq_len: int = 512,
                         max_records: int = 100_000, max_bytes: int = 64 * 1024 * 1024) -> dict:
    """Validate bounded local JSONL and atomically publish checksummed SFT data."""
    for name, value, limit in (("max_seq_len", max_seq_len, 32768),
                               ("max_records", max_records, 100000),
                               ("max_bytes", max_bytes, 64 * 1024 * 1024)):
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError(f"{name} must be an integer in [1, {limit}]")
    train_path, valid_path, tokenizer_path = map(Path, (train_path, valid_path, tokenizer_path))
    output_dir = Path(output_dir).expanduser().absolute()
    if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
        raise FileExistsError("Instruction output must be absent or empty")
    train, train_hash = _read_records(train_path, max_records, max_bytes)
    valid, valid_hash = _read_records(valid_path, max_records, max_bytes)
    _check_splits(train, valid)
    tokenizer = load_tokenizer(tokenizer_path)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{output_dir.name}.preparing-", dir=output_dir.parent))
    try:
        shutil.copyfile(tokenizer_path, staging / "tokenizer.json")
        manifest = {
            "schema_version": 1, "kind": "silivra-instructions", "prompt_format": PROMPT_FORMAT,
            "max_seq_len": max_seq_len,
            "tokenizer": {"path": "tokenizer.json", "sha256": sha256(staging / "tokenizer.json"),
                          "vocab_size": tokenizer.get_vocab_size(), "pad_id": tokenizer.token_to_id("<pad>")},
            "sampling": "one complete example per assistant turn; context excluded from loss",
            "deduplication": "metadata-independent normalized conversation and cross-split user-prefix hashes",
            "splits": {},
        }
        total_tokens = 0
        for split, records, source_hash in (("train", train, train_hash), ("valid", valid, valid_hash)):
            samples = []
            for record in records:
                turns = encode_turns(record, tokenizer, max_seq_len)
                total_tokens += sum(len(ids) for ids, _ in turns)
                if total_tokens > _MAX_OUTPUT_TOKENS:
                    raise ValueError(f"Prepared instruction tokens exceed {_MAX_OUTPUT_TOKENS}; use a smaller dataset")
                samples.extend(turns)
            width = max(len(ids) for ids, _ in samples)
            if len(samples) * width > _MAX_OUTPUT_TOKENS:
                raise ValueError("Padded instruction split exceeds 32 million tokens; use shorter examples")
            tokens = np.full((len(samples), width), tokenizer.token_to_id("<pad>"), dtype=np.uint16)
            masks = np.zeros(tokens.shape, dtype=np.uint8)
            for index, (ids, mask) in enumerate(samples):
                tokens[index, :len(ids)] = ids
                masks[index, :len(mask)] = mask
            target = staging / f"{split}.npz"
            np.savez(target, tokens=tokens, masks=masks)
            source_file = staging / f"{split}.jsonl"
            source_file.write_text("".join(_json(record) + "\n" for record in records), encoding="utf-8")
            manifest["splits"][split] = {
                "path": target.name, "sha256": sha256(target), "conversations": len(records),
                "samples": len(samples), "supervised_tokens": int(masks.sum()),
                "source_path": source_file.name, "source_sha256": sha256(source_file),
                "input_sha256": source_hash, "sources": list({
                    _json(record["source"]): record["source"] for record in records
                }.values()),
            }
        (staging / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        if output_dir.exists() and (not output_dir.is_dir() or any(output_dir.iterdir())):
            raise FileExistsError("Instruction output became nonempty during preparation")
        os.replace(staging, output_dir)
        return manifest
    finally:
        if staging.exists():
            shutil.rmtree(staging)


class InstructionDataset:
    """Verified fixed-width examples with padding excluded from assistant loss."""

    def __init__(self, path: Path, max_seq_len: int, vocab_size: int):
        self.path = Path(path)
        with np.load(path, allow_pickle=False) as archive:
            if set(archive.files) != {"tokens", "masks"}:
                raise ValueError("Instruction archive must contain exactly tokens and masks")
            self.tokens, self.masks = archive["tokens"], archive["masks"]
        if (self.tokens.ndim != 2 or self.tokens.shape != self.masks.shape
                or not 2 <= self.tokens.shape[1] <= max_seq_len + 1
                or not 1 <= self.tokens.shape[0]
                or self.tokens.dtype != np.uint16 or self.masks.dtype != np.uint8
                or self.tokens.size > _MAX_OUTPUT_TOKENS
                or int(self.tokens.max()) >= vocab_size
                or not np.isin(self.masks, [0, 1]).all()
                or not (self.masks[:, 0] == 0).all()
                or not (self.masks[:, 1:].sum(axis=1) > 0).all()):
            raise ValueError("Invalid instruction token/mask shapes, values or context length")

    def batch(self, rng, batch_size):
        indices = rng.integers(0, len(self.tokens), size=batch_size)
        return (self.tokens[indices, :-1].astype(np.int32),
                self.tokens[indices, 1:].astype(np.int32),
                self.masks[indices, 1:].astype(np.float32))

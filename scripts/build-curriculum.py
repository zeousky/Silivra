#!/usr/bin/env python3
"""Generate a small, original capability curriculum; this does not train a model.

Run ``uv run python scripts/build-curriculum.py --output data/curriculum-v1
--tokenizer runs/your-run/checkpoints/your-checkpoint/tokenizer.json``. Output is
written only on an explicit request, beneath this checkout's ignored data/ path.
The 500 examples are AI-assisted procedural exercises, licensed MIT. Separate
vocabularies, operands and filenames are reserved for validation. No evaluation
suite, network, private state, pretrained model, or tool execution is consulted.

These narrow exercises teach response format and basic targets. They are not a
substitute for a broad licensed corpus or an independent capability evaluation.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import tempfile

from silivra.agent import build_system_prompt
from silivra.instructions import encode_turns, parse_action, validate_record
from silivra.tools import ToolRegistry


VERSION = 1
ROOT = Path(__file__).resolve().parents[1]
FAMILIES = ("copy", "exact", "addition", "subtraction", "multiplication",
            "expression", "function", "read_file", "search_files", "web_search")
TRAIN_WORDS = (
    "air", "ant", "ape", "ash", "bay", "bed", "bee", "bow", "box", "bud",
    "bug", "cap", "car", "cat", "cow", "cup", "day", "dew", "dog", "dot",
    "ear", "egg", "elk", "elm", "fan", "fig", "fin", "fir", "fox", "fur",
    "gap", "gem", "gum", "hat", "hay", "hen", "hill", "ice", "ink", "jar",
)
VALID_WORDS = ("jet", "key", "kit", "lake", "log", "map", "nest", "oak", "owl", "pan")


def _json(value) -> str:
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"), allow_nan=False)


def _record(split, family, index, seed, prompt, action, schemas=()):
    return validate_record({
        "id": f"curriculum-v{VERSION}-s{seed}-{split}-{family}-{index:03d}",
        "source": {
            "name": "Silivra original AI-assisted procedural capability curriculum",
            "license": "MIT",
            "url": "local:scripts/build-curriculum.py",
            "author": "Silivra contributors, AI-assisted procedural generation",
            "revision": f"generator-v{VERSION};seed={seed};split={split};family={family}",
        },
        "messages": [
            {"role": "system", "content": build_system_prompt(list(schemas))},
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": _json(action)},
        ],
    })


def build_curriculum(seed: int = 17) -> dict[str, list[dict]]:
    """Return deterministic train/valid records without writing or executing tools.

    There are 40 train and 10 validation examples in each of ten families.
    Tool records contain only an honest proposal; no fabricated tool response or
    success claim is included. Each exposes one exact runtime tool schema to fit
    the original short context. Multi-tool routing needs a later curriculum.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    rng = random.Random(seed)
    schemas = {schema["function"]["name"]: schema for schema in
               ToolRegistry(ROOT, allow_network=True).schemas()}
    result = {"train": [], "valid": []}
    for split, vocabulary in (("train", TRAIN_WORDS), ("valid", VALID_WORDS)):
        words = list(vocabulary)
        rng.shuffle(words)
        count = len(words)

        def append(family, index, prompt, action, tool=None):
            result[split].append(_record(split, family, index, seed, prompt, action,
                                         [schemas[tool]] if tool else []))

        for index, word in enumerate(words):
            copy_templates = ("Copy exactly: {text}", "Repeat this text exactly: {text}",
                              "Return this unchanged: {text}", "Copy the following: {text}")
            literal = f"{word} {word}"
            append("copy", index, copy_templates[index % 4].format(text=literal), {"final": literal})
            exact_templates = ("Answer with exactly: {text}", "Reply using only: {text}",
                               "Give this exact answer: {text}", "Your answer must be: {text}")
            append("exact", index, exact_templates[index % 4].format(text=word), {"final": word})

        # Held-out operands do not occur in any training arithmetic exercise.
        small = range(1, 13) if split == "train" else range(21, 33)
        for family in ("addition", "subtraction", "multiplication"):
            if family == "subtraction":
                pairs = ([(a, b) for a in small for b in small if a >= b] if split == "train"
                         else [(a, b) for a in range(60, 72) for b in small])
            else:
                pairs = [(a, b) for a in small for b in small if a <= b]
            rng.shuffle(pairs)
            for index, (left, right) in enumerate(pairs[:count]):
                if family == "addition":
                    prompt = (f"Add {left} and {right}. Give only the number." if index % 2
                              else f"What is {left} + {right}? Answer with a number.")
                    answer = left + right
                elif family == "subtraction":
                    prompt = (f"Subtract {right} from {left}. Give only the number." if index % 2
                              else f"What is {left} - {right}? Answer with a number.")
                    answer = left - right
                else:
                    prompt = (f"Multiply {left} by {right}. Give only the number." if index % 2
                              else f"What is {left} * {right}? Answer with a number.")
                    answer = left * right
                append(family, index, prompt, {"final": str(answer)})

        expressions = (
            ("the length of list {v}", "len({v})"),
            ("the sum of numbers in {v}", "sum({v})"),
            ("a sorted copy of list {v}", "sorted({v})"),
            ("the first item in nonempty list {v}", "{v}[0]"),
            ("the last item in nonempty list {v}", "{v}[-1]"),
            ("a reversed copy of list {v}", "{v}[::-1]"),
            ("the keys of dictionary {v}", "{v}.keys()"),
            ("the largest number in nonempty list {v}", "max({v})"),
            ("the smallest number in nonempty list {v}", "min({v})"),
            ("string {v} converted to uppercase", "{v}.upper()"),
        )
        functions = (
            ("square", "value squared", "value * value"),
            ("double", "twice value", "value * 2"),
            ("increment", "value plus one", "value + 1"),
            ("negate", "the negative of value", "-value"),
            ("identity", "value unchanged", "value"),
        )
        for index, word in enumerate(words):
            description, expression = expressions[index % len(expressions)]
            append("expression", index,
                   "Write only a Python expression for " + description.format(v=word) + ".",
                   {"final": expression.format(v=word)})
            operation, description, expression = functions[index % len(functions)]
            name = f"{operation}_{word}"
            append("function", index,
                   f"Write Python function {name}(value) returning {description}. Output code only.",
                   {"final": f"def {name}(value):\n    return {expression}"})
            read_templates = ("Read {path}.", "Open {path}.", "Read file {path}.", "Show {path}.")
            filename = f"{word}.py"
            append("read_file", index, read_templates[index % 4].format(path=filename),
                   {"tool": "read_file", "arguments": {"path": filename}}, "read_file")
            search_templates = ("Find {word} in files.", "Search files for {word}.",
                                "Find text {word}.", "Locate {word} in files.")
            append("search_files", index, search_templates[index % 4].format(word=word),
                   {"tool": "search_files", "arguments": {"query": word}}, "search_files")
            web_templates = ("Search the web for {word}.", "Look up {word} online.",
                             "Web search: {word}.", "Find web pages about {word}.")
            append("web_search", index, web_templates[index % 4].format(word=word),
                   {"tool": "web_search", "arguments": {"query": word}}, "web_search")
        rng.shuffle(result[split])
    return result


def token_statistics(records: dict[str, list[dict]], tokenizer, max_seq_len: int = 512) -> dict:
    """Verify every complete target fits; report actual original-tokenizer lengths."""
    if type(max_seq_len) is not int or not 1 <= max_seq_len <= 32768:
        raise ValueError("max_seq_len must be an integer in [1, 32768]")
    statistics = {}
    for split, rows in records.items():
        maximum_sample = maximum_prompt = maximum_target = 0
        for record in rows:
            ids, mask = encode_turns(record, tokenizer, max_seq_len)[0]
            maximum_sample = max(maximum_sample, len(ids) - 1)
            maximum_prompt = max(maximum_prompt, len(mask) - sum(mask))
            maximum_target = max(maximum_target, sum(mask))
        statistics[split] = {"max_sample_tokens": maximum_sample,
                             "max_prompt_tokens": maximum_prompt,
                             "max_target_tokens_including_eos": maximum_target}
    return statistics


def build_evaluation(valid_records: list[dict]) -> list[dict]:
    """Convert the unchanged held-out split to nonexecuting assistant test cases.

    This measures generalization within this narrow procedural curriculum. It is
    separate from the independently authored diagnostic suite and must be reported
    with that distinction. The expected action and prompt are copied exactly from
    validation; no task is added, rewritten, or selected based on model output.
    """
    schemas = {schema["function"]["name"]: schema for schema in
               ToolRegistry(ROOT, allow_network=True).schemas()}
    cases = []
    for raw in valid_records:
        record = validate_record(raw)
        provenance = record["source"]["revision"]
        if ";split=valid;" not in provenance:
            raise ValueError("Curriculum evaluation must use the held-out validation split")
        action = parse_action(record["messages"][-1]["content"])
        cases.append({"id": record["id"], "category": provenance.rsplit("family=", 1)[1],
                      "prompt": record["messages"][1]["content"],
                      "tools": [schemas[action["tool"]]] if "tool" in action else [],
                      "expected": action})
    return cases


def write_curriculum(output: Path, *, seed: int = 17, tokenizer_path: Path | None = None,
                     max_seq_len: int = 512) -> dict:
    """Stage complete JSONL files and atomically publish a new output directory."""
    output = Path(output)
    if output.exists() or output.is_symlink():
        raise FileExistsError("Curriculum output already exists")
    records = build_curriculum(seed)
    statistics = None
    tokenizer_hash = None
    if tokenizer_path is not None:
        from tokenizers import Tokenizer
        tokenizer_path = Path(tokenizer_path)
        tokenizer_hash = hashlib.sha256(tokenizer_path.read_bytes()).hexdigest()
        statistics = token_statistics(records, Tokenizer.from_file(str(tokenizer_path)), max_seq_len)
    manifest = {
        "format_version": VERSION, "seed": seed,
        "license": "MIT", "source": "local:scripts/build-curriculum.py",
        "generation": "original AI-assisted procedural exercises; no external corpus or evaluation suite read",
        "scope": "narrow instruction/arithmetic/Python basics and one-tool proposals; not a broad intelligence corpus",
        "tokenizer_sha256": tokenizer_hash,
        "validated_max_seq_len": max_seq_len if statistics else None,
        "token_statistics": statistics,
        "splits": {},
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".silivra-curriculum-", dir=output.parent) as temporary:
        stage = Path(temporary) / "curriculum"
        stage.mkdir()
        for split, rows in records.items():
            content = "".join(_json(record) + "\n" for record in rows)
            (stage / f"{split}.jsonl").write_text(content, encoding="utf-8")
            families = Counter(record["source"]["revision"].rsplit("family=", 1)[1] for record in rows)
            manifest["splits"][split] = {"records": len(rows), "families": dict(sorted(families.items())),
                                         "sha256": hashlib.sha256(content.encode("utf-8")).hexdigest()}
        evaluation = build_evaluation(records["valid"])
        evaluation_content = "".join(_json(case) + "\n" for case in evaluation)
        (stage / "evaluation.jsonl").write_text(evaluation_content, encoding="utf-8")
        manifest["evaluation"] = {
            "file": "evaluation.jsonl", "records": len(evaluation), "source_split": "valid",
            "sha256": hashlib.sha256(evaluation_content.encode("utf-8")).hexdigest(),
            "grading": "exact parsed actions on held-out procedural curriculum; not an independent benchmark",
        }
        (stage / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if output.exists() or output.is_symlink():
            raise FileExistsError("Curriculum output already exists")
        os.rename(stage, output)
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="New directory beneath ignored checkout data/")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument("--tokenizer", type=Path, help="Verify complete samples against this existing tokenizer")
    parser.add_argument("--max-seq-len", type=int, default=512)
    args = parser.parse_args(argv)
    output = args.output.expanduser().resolve()
    if not output.is_relative_to((ROOT / "data").resolve()) or output == (ROOT / "data").resolve():
        parser.error("--output must name a new directory beneath this checkout's ignored data/ directory")
    try:
        manifest = write_curriculum(output, seed=args.seed, tokenizer_path=args.tokenizer,
                                    max_seq_len=args.max_seq_len)
    except (ValueError, OSError) as error:
        parser.exit(1, f"curriculum: {error}\n")
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

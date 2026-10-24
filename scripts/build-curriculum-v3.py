#!/usr/bin/env python3
"""Build a small original, input-conditioned assistant curriculum offline.

This is Codex-authored synthetic MIT training data, not a general benchmark.
It uses the unchanged local tokenizer and never reads chats or model weights.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import random
import shutil
import tempfile

from silivra.agent import POLICY
from silivra.checkpoint import sha256
from silivra.data import load_tokenizer
from silivra.instructions import _check_splits, encode_turns, prepare_instructions, validate_record


SOURCE = {
    "name": "Silivra focused assistant curriculum v3 (Codex-authored)",
    "author": "Silivra project with Codex; deterministic synthetic examples",
    "license": "MIT", "url": "https://github.com/zeousky/Silivra",
    "revision": "curriculum-v3",
}
TRAIN_WORDS = "apple bird blue boat book cat dog door fish flower fox green house lake moon red river sky small star sun tea tree water white wind yellow".split()
VALID_WORDS = "amber beach cloud deer field forest grass hill light rain road rose snow stone table town warm wood".split()
TRAIN_NAMES = "Ada Amir Cora Finn Hana Iris Kai Leo Lina Maya Mina Noah Omar Ravi Sam Tara Vera Zara".split()
VALID_NAMES = "Alma Bela Dara Elena Emil Imani Lila Nia Oren Remy Theo Yara".split()


def dump(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def _record(kind, prompt, answer, *, history=()):
    messages = [{"role": "system", "content": POLICY}, *history,
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": dump({"final": str(answer)})}]
    identity = hashlib.sha256(dump(messages).encode()).hexdigest()[:20]
    return validate_record({"id": f"v3-{kind}-{identity}", "source": dict(SOURCE), "messages": messages})


def build_splits(seed=20261022, *, excluded_prompts=()):
    """Reserve numeric pairs, names, and copied words before making examples.

    Validation shares skill templates and is deliberately a narrow diagnostic.
    It is never used to author training targets or select training examples.
    """
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    rng = random.Random(seed)
    train, valid = [], []
    greeting = "Hello! How can I help you?"
    wellbeing = "I'm here and ready to help. How can I help you?"
    identity = "I'm Silivra, a small experimental AI trained from scratch."
    capability = "I'm an experimental local AI. My answers can be wrong, so check important information."
    social_train = {
        greeting: ["Hi", "Hello", "Hey", "Hi Silivra", "Hello Silivra", "Hey Silivra", "Good morning", "Good evening", "Greetings", "Hello there", "Hi there", "Hey there"],
        wellbeing: ["How are you", "How are you doing", "How is it going", "How are things", "Are you doing well", "How are you today", "How do you feel", "Are you okay"],
        identity: ["Who are you", "What is your name", "Tell me your name", "Introduce yourself", "What are you called", "Are you Silivra", "Tell me who you are", "Describe yourself briefly"],
        capability: ["What can you do", "Can I trust every answer", "Are your answers always correct", "Are you a reliable expert", "Tell me about your limitations", "How capable are you", "Can you make mistakes", "Should I check your answers"],
    }
    for answer, prompts in social_train.items():
        for prompt in prompts:
            for punctuation in ("", ".", "!", "?"):
                train.append(_record("conversation", prompt + punctuation, answer))
    social_valid = {
        greeting: ["Good afternoon, Silivra.", "Hey, nice to meet you!", "Hello again, Silivra!"],
        wellbeing: ["How are you feeling today?", "Is everything going well with you?", "How have you been?"],
        identity: ["Could you introduce yourself?", "What should I call you?", "Please tell me your identity."],
        capability: ["What are the limits of your ability?", "Could your responses contain errors?", "Can I rely on your advice without checking?"],
    }
    for answer, prompts in social_valid.items():
        valid.extend(_record("conversation", prompt, answer) for prompt in prompts)
    # Native role histories match supervised conversations. Only the fixed first
    # system message is authoritative; previous answers remain fallible text.
    for opening in social_train[greeting]:
        history = [{"role": "user", "content": opening},
                   {"role": "assistant", "content": dump({"final": greeting})}]
        for prompt in social_train[wellbeing]:
            train.append(_record("conversation-history", prompt + "?", wellbeing, history=history))
    valid.append(_record("conversation-history", "How is your day going?", wellbeing,
                         history=[{"role": "user", "content": "Good afternoon."},
                                  {"role": "assistant", "content": dump({"final": greeting})}]))

    for split, words, names, count in ((train, TRAIN_WORDS, TRAIN_NAMES, 480),
                                       (valid, VALID_WORDS, VALID_NAMES, 64)):
        phrases = set()
        while len(phrases) < count:
            phrase = " ".join(rng.sample(words, rng.choice([1, 2, 2, 3])))
            phrases.add(phrase)
        for index, phrase in enumerate(sorted(phrases)):
            template = ("Copy exactly: {}", "Repeat this text: {}", "Reply with only this text: {}", "Return only: {}")[index % 4]
            split.append(_record("copy", template.format(phrase), phrase))
        for index in range(400 if split is train else 64):
            name = names[index % len(names)]
            color = words[(index // len(names)) % len(words)]
            age = 10 + index % 70
            if index % 2:
                prompt = f"{name}'s favorite word is {color}. What is {name}'s favorite word? Reply with only the word."
                answer = color
            else:
                prompt = f"Source: {name} is {age} years old. How old is {name}? Reply with only the number."
                answer = age
            split.append(_record("grounding", prompt, answer))

    addition_templates = ["What is {} + {}?", "Calculate {} + {}.", "Add {} and {}. Number only.", "Return only the sum of {} and {}.", "{} + {} = ?"]
    comparison_templates = ["Is {} greater than {}? Answer yes or no.", "Is {} larger than {}? Reply yes or no.", "Does {} exceed {}? Answer yes or no.", "Compare {} and {}: is the first greater? Answer yes or no."]
    for a in range(10):
        for b in range(10):
            # Both partitions contain every digit; particular ordered pairs
            # have only one partition, preventing identical problems in both.
            split = valid if (a * 3 + b) % 7 == 0 else train
            templates = addition_templates[:2] if split is valid else addition_templates
            split.extend(_record("addition", t.format(a, b), a + b) for t in templates)
            templates = comparison_templates[:2] if split is valid else comparison_templates
            split.extend(_record("comparison", t.format(a, b), "yes" if a > b else "no") for t in templates)
            # Short executable Python is grounded in its literal operands.
            split.append(_record("python", f"What does this Python code print? Return only the output.\nprint({a} + {b})", a + b))
            if split is train:
                split.append(_record("python", f"Give only the output of this Python code:\nprint({a} + {b})", a + b))

    # These historical diagnostics are inspected only to exclude exact user
    # prompts. No expected evaluation answers are consumed by the generator.
    excluded = {" ".join(prompt.split()) for prompt in excluded_prompts}
    for name in ("assistant.jsonl", "assistant-v2.jsonl"):
        path = Path(__file__).resolve().parents[1] / "examples" / "evaluations" / name
        for line in path.read_text(encoding="utf-8").splitlines():
            case = json.loads(line)
            messages = case.get("messages", [{"role": "user", "content": case.get("prompt", "")}])
            excluded.update(" ".join(m["content"].split()) for m in messages if m["role"] == "user")
    train = [r for r in train if not any(" ".join(m["content"].split()) in excluded
                                        for m in r["messages"] if m["role"] == "user")]
    rng.shuffle(train)
    rng.shuffle(valid)
    _check_splits(train, valid)
    return train, valid


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--prepare", type=Path)
    parser.add_argument("--seed", type=int, default=20261022)
    parser.add_argument("--max-seq-len", type=int, default=1024)
    parser.add_argument("--exclude-suite", type=Path, action="append", default=[],
                        help="exclude this suite's user prompts without inspecting expected answers; repeatable")
    args = parser.parse_args(argv)
    if args.output.exists():
        parser.error("--output must be a new directory")
    if not 128 <= args.max_seq_len <= 4096:
        parser.error("--max-seq-len must be between 128 and 4096")
    if args.prepare:
        output, prepared = args.output.resolve(), args.prepare.resolve()
        if output.is_relative_to(prepared) or prepared.is_relative_to(output):
            parser.error("--output and --prepare must not overlap")
        if args.prepare.exists() and (not args.prepare.is_dir() or any(args.prepare.iterdir())):
            parser.error("--prepare must be absent or empty")
    excluded, exclusion_sources = [], []
    for path in args.exclude_suite:
        if path.stat().st_size > 2 * 1024 * 1024:
            parser.error("Exclusion suite exceeds 2 MiB")
        snapshot = path.read_bytes()
        for line in snapshot.decode("utf-8").splitlines():
            if not line.strip():
                continue
            case = json.loads(line)
            messages = case.get("messages", [{"role": "user", "content": case.get("prompt", "")}])
            excluded.extend(m["content"] for m in messages if m["role"] == "user")
        exclusion_sources.append({"name": path.name, "sha256": hashlib.sha256(snapshot).hexdigest(),
                                  "use": "user prompt exclusion only; expected answers not inspected"})
    train, valid = build_splits(args.seed, excluded_prompts=excluded)
    tokenizer = load_tokenizer(args.tokenizer)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=".curriculum-v3-", dir=args.output.parent))
    try:
        manifest = {"schema_version": 1, "kind": "silivra-focused-curriculum-v3", "seed": args.seed,
                    "source": SOURCE, "generator_sha256": sha256(Path(__file__)),
                    "prompt_exclusion_sources": exclusion_sources,
                    "tokenizer_sha256": sha256(args.tokenizer), "max_seq_len": args.max_seq_len,
                    "splitting": "reserved digit pairs and disjoint copied words/names; conversation paraphrases held out",
                    "boundary": "synthetic narrow skills, shared templates; no general intelligence claim; no imported corpus or private chats",
                    "splits": {}}
        for name, records in (("train", train), ("valid", valid)):
            turns = [turn for r in records for turn in encode_turns(r, tokenizer, args.max_seq_len)]
            path = staging / f"{name}.jsonl"
            path.write_text("".join(dump(r) + "\n" for r in records), encoding="utf-8")
            manifest["splits"][name] = {"sha256": sha256(path), "records": len(records),
                "families": dict(Counter(r["id"].split("-", 2)[1] for r in records)),
                "supervised_tokens": sum(sum(mask) for _, mask in turns),
                "max_tokens": max(len(ids) - 1 for ids, _ in turns)}
        (staging / "corpus-manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
        if args.output.exists():
            raise FileExistsError("Output appeared during generation")
        os.rename(staging, args.output)
        if args.prepare:
            prepare_instructions(args.output / "train.jsonl", args.output / "valid.jsonl",
                                 args.tokenizer, args.prepare, max_seq_len=args.max_seq_len)
        print(json.dumps(manifest, indent=2))
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == "__main__":
    main()

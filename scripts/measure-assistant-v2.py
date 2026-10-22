#!/usr/bin/env python3
"""Compare original, resumed, and selected local weights without running tools.

Loads one model at a time. Uses only existing local artifacts, fixed greedy
decoding, bounded held-out samples, and the original fixed story-loss sample.
Reports contain hashes and checkpoint names, never private machine paths or
imported corpus rows. No training, network access, or automatic model selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import gc
import json
from pathlib import Path
import random
import tempfile
import unicodedata

from silivra.agent import MLXBackend, POLICY
from silivra.checkpoint import resolve_checkpoint, sha256, verify_checkpoint
from silivra.data import TokenDataset, load_tokenizer
from silivra.evaluation import case_messages, evaluate_assistant, validate_cases
from silivra.finetune import _manifest, evaluate_loss as instruction_loss
from silivra.instructions import InstructionDataset, _check_splits, format_prompt, parse_action
from silivra.train import evaluate_loss as story_loss


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalized(text):
    return " ".join(unicodedata.normalize("NFC", text).split())


def user_prefixes(messages):
    users, prefixes = [], set()
    for message in messages:
        if message["role"] == "user":
            users.append(normalized(message["content"]))
        elif message["role"] == "assistant":
            prefixes.add(tuple(users))
    if users:
        prefixes.add(tuple(users))
    return prefixes


def check_diagnostic_overlap(cases, training):
    """Reject exact normalized prompts/prefixes; no semantic-cleanliness claim."""
    prefixes = set()
    for record in training:
        prefixes.update(user_prefixes(record["messages"]))
    overlaps = [case["id"] for case in cases
                if prefixes.intersection(user_prefixes(case_messages(case)))]
    if overlaps:
        raise ValueError(f"Diagnostic/training exact prompt overlap: {overlaps}")
    return {"method": "NFC/whitespace-normalized complete user-turn prefixes",
            "diagnostic_cases": len(cases), "exact_training_overlaps": 0,
            "limitations": "No semantic, template, or pretraining corpus-wide contamination guarantee."}


def validation_cases(records, per_category=2, seed=20261022):
    """Sample only original MIT synthetic records, preserving complete history."""
    groups = {}
    for record in records:
        if not record["id"].startswith("original-v2-"):
            continue
        category = record["id"].split("-", 3)[-1]
        groups.setdefault(category, []).append(record)
    cases = []
    for category, group in sorted(groups.items()):
        group = sorted(group, key=lambda record: record["id"])
        random.Random(f"{seed}:{category}").shuffle(group)
        for record in group[:per_category]:
            system = record["messages"][0]["content"]
            if system == POLICY:
                tools = []
            elif system.startswith(POLICY + " Tools: "):
                specs = json.loads(system[len(POLICY + " Tools: "):])
                tools = [{"type": "function", "function": spec} for spec in specs]
            else:
                raise ValueError("Synthetic record uses an unexpected system policy.")
            cases.append({"id": "heldout-" + record["id"], "category": category,
                          "messages": record["messages"][1:-1], "tools": tools,
                          "expected": parse_action(record["messages"][-1]["content"])})
    validate_cases(cases)
    return cases


def prompt_tokens(case, tokenizer):
    from silivra.agent import build_system_prompt
    messages = case_messages(case)
    messages[0]["content"] = build_system_prompt(case["tools"])
    return 1 + len(tokenizer.encode(format_prompt(messages), add_special_tokens=False).ids)


def write_suite(path, cases):
    path.write_text("".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n"
                            for case in cases), encoding="utf-8")


def manifest_artifact(data, entry):
    path = (data / entry["path"]).resolve()
    if not path.is_relative_to(data.resolve()) or sha256(path) != entry["sha256"]:
        raise ValueError("Prepared evaluation artifact checksum mismatch.")
    return path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("runs/silivra-tiny/checkpoints/step-0002500"))
    parser.add_argument("--before", type=Path, default=Path("runs/silivra-assistant-v2/checkpoints/step-0000500"))
    parser.add_argument("--run", type=Path, default=Path("runs/silivra-assistant-v2"))
    parser.add_argument("--data", type=Path, default=Path("data/instructions-v2"))
    parser.add_argument("--story-data", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--corpus-manifest", type=Path, default=Path("data/curriculum-v2/corpus-manifest.json"))
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--valid-per-category", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("runs/silivra-assistant-v2/comparison.json"))
    parser.add_argument("--public-output", type=Path)
    args = parser.parse_args(argv)
    if not 32 <= args.max_new_tokens <= 128 or not 1 <= args.valid_per_category <= 4:
        parser.error("Use 32–128 output tokens and 1–4 validation records per category.")
    outputs = [args.output, *([args.public_output] if args.public_output else [])]
    if len({path.resolve() for path in outputs}) != len(outputs) or any(path.exists() for path in outputs):
        raise FileExistsError("Comparison output paths must be distinct and absent.")

    data = args.data.resolve()
    manifest = _manifest(data)
    train = read_jsonl(data / manifest["splits"]["train"]["source_path"])
    valid = read_jsonl(data / manifest["splits"]["valid"]["source_path"])
    _check_splits(train, valid)
    fixed_path = Path("examples/evaluations/assistant.jsonl")
    independent_path = Path("examples/evaluations/assistant-v2.jsonl")
    fixed, independent = read_jsonl(fixed_path), read_jsonl(independent_path)
    validate_cases(fixed)
    validate_cases(independent)
    overlap = check_diagnostic_overlap([*fixed, *independent], train)
    templated = validation_cases(valid, args.valid_per_category)
    tokenizer = load_tokenizer(data / manifest["tokenizer"]["path"])
    shared_max = max(prompt_tokens(case, tokenizer) for case in [*fixed, *independent])
    if shared_max + args.max_new_tokens > 512:
        raise ValueError(f"Shared diagnostics exceed base context: {shared_max} + {args.max_new_tokens} > 512.")

    checkpoints = [("original_base", resolve_checkpoint(args.base)),
                   ("resumed_step500", resolve_checkpoint(args.before)),
                   ("selected_v2", resolve_checkpoint(args.run, "best"))]
    states = [verify_checkpoint(path) for _, path in checkpoints]
    token_hash = manifest["tokenizer"]["sha256"]
    if any(sha256(path / "tokenizer.json") != token_hash for _, path in checkpoints):
        raise ValueError("Comparison requires the unchanged original tokenizer.")
    if states[1]["step"] != 500 or states[-1].get("training_mode") != "instruction_sft":
        raise ValueError("Expected the saved step500 and a selected instruction checkpoint.")
    if any(state["data_manifest_sha256"] != sha256(data / "manifest.json") for state in states[1:]):
        raise ValueError("Instruction checkpoint/data manifest mismatch.")
    base_weights = sha256(checkpoints[0][1] / "model.safetensors")
    if any(state.get("base_checkpoint", {}).get("weights_sha256") != base_weights for state in states[1:]):
        raise ValueError("Instruction checkpoints do not share the supplied original base.")

    story_manifest = json.loads((args.story_data / "manifest.json").read_text())
    if sha256(args.story_data / "manifest.json") != states[0]["data_manifest_sha256"]:
        raise ValueError("Original story manifest differs from the original checkpoint.")
    story_path = manifest_artifact(args.story_data, story_manifest["splits"]["valid"])
    story_dataset = TokenDataset(story_path, 256, vocab_size=manifest["tokenizer"]["vocab_size"])
    instruction_dataset = InstructionDataset(data / manifest["splits"]["valid"]["path"],
                                              manifest["max_seq_len"], manifest["tokenizer"]["vocab_size"],
                                              bucket_size=64)

    corpus = json.loads(args.corpus_manifest.read_text())
    historical_generator = corpus["provenance"]["original"]["generator_sha256"]
    generator = Path("scripts/build-curriculum-v2.py")
    report = {
        "format_version": 1,
        "purpose": "Bounded before/after diagnostics; not a general capability benchmark or release of weights.",
        "selection": "best.json selected only by held-out instruction loss; task scores were not used to select weights",
        "generation": {"temperature": 0, "top_k": 0, "seed": 42, "max_new_tokens": args.max_new_tokens},
        "provenance": {
            "measurement_script_sha256": sha256(Path(__file__)),
            "evaluation_source_sha256": sha256(Path("src/silivra/evaluation.py")),
            "instruction_manifest_sha256": sha256(data / "manifest.json"),
            "story_manifest_sha256": sha256(args.story_data / "manifest.json"),
            "corpus_manifest_sha256": sha256(args.corpus_manifest),
            "tokenizer_sha256": token_hash,
            "data_generating_script_sha256_recorded": historical_generator,
            "initial_unfinished_generator_sha256_observed": "5357780031527c6588862937ab4b72474e103087b636bc80b74ca651a5144451",
            "current_generator_sha256": sha256(generator),
            "generator_drift": historical_generator != sha256(generator),
            "generator_boundary": "Existing data retained unchanged; the current source does not exactly reproduce the historical generator bytes.",
        },
        "leakage_audit": {**overlap, "train_valid_exact_check": "passed",
                          "tokenizer_fit": "original pretraining train split only; unchanged tokenizer hash"},
        "suites": {
            "fixed_six": {"sha256": sha256(fixed_path), "cases": len(fixed), "boundary": "historical diagnostic"},
            "independent_v2": {"sha256": sha256(independent_path), "cases": len(independent),
                               "authors": "Codex-authored original MIT diagnostic fixtures; not used for training",
                               "maximum_shared_prompt_tokens": shared_max, "shared_context_limit": 512,
                               "boundary": "Small independent prompts; tasks may share underlying skills/templates with training."},
            "templated_v2_valid": {"cases": len(templated), "per_category_limit": args.valid_per_category,
                                   "seed": 20261022, "categories": dict(sorted(Counter(case["category"] for case in templated).items())),
                                   "boundary": "MIT synthetic validation only; shared generator templates, no Dolly/raw imported rows. Native-context coverage can differ."},
        },
        "loss_settings": {
            "story": {"batches": 12, "batch_size": 16, "sequence_length": 256, "seed": 12345,
                      "boundary": "Original unseen validation token stream; unmasked next-token language regression."},
            "v2_instruction": {"batches": 24, "batch_size": 4, "seed": 12345, "bucket_size": 64,
                               "boundary": "Assistant-only loss; shuffled held-out corpus including Dolly and original-pretraining-train story replay. Not unseen general language."},
        },
        "tools_executed": False, "models_loaded_sequentially": True, "records": [],
    }
    with tempfile.TemporaryDirectory(prefix="silivra-evaluation-") as temporary:
        templated_path = Path(temporary) / "templated.jsonl"
        write_suite(templated_path, templated)
        report["suites"]["templated_v2_valid"]["sha256"] = sha256(templated_path)
        report["suites"]["templated_v2_valid"]["fixtures"] = templated
        for (label, path), state in zip(checkpoints, states):
            print(json.dumps({"event": "evaluation_start", "model": label, "step": state["step"]}), flush=True)
            backend = MLXBackend(path, max_new_tokens=args.max_new_tokens)
            record = {"label": label, "checkpoint": path.name, "step": state["step"],
                      "model_sha256": sha256(path / "model.safetensors"),
                      "context_window": backend.model.config.max_seq_len,
                      "supervised_tokens_seen": state.get("supervised_tokens_seen"),
                      "training_seconds": state["training_seconds"],
                      "fixed_six": evaluate_assistant(backend, fixed_path),
                      "independent_v2": evaluate_assistant(backend, independent_path),
                      "templated_v2_valid": evaluate_assistant(backend, templated_path)}
            record["story_validation_loss"] = story_loss(backend.model, story_dataset, 16, 12)
            if backend.model.config.max_seq_len >= manifest["max_seq_len"]:
                record["v2_instruction_validation_loss"] = instruction_loss(backend.model, instruction_dataset, 4, 24)
            else:
                record["v2_instruction_validation_loss"] = None
                record["v2_loss_unavailable_reason"] = "Base context512 cannot evaluate full prepared context1024 without extrapolation."
            report["records"].append(record)
            print(json.dumps({"event": "evaluation_complete", "model": label,
                              "independent_passed": record["independent_v2"]["passed"],
                              "story_validation_loss": record["story_validation_loss"],
                              "v2_instruction_validation_loss": record["v2_instruction_validation_loss"]}), flush=True)
            del backend
            gc.collect()
            import mlx.core as mx
            mx.clear_cache()

    before, selected = report["records"][1:]
    report["changes_from_step500"] = {
        "instruction_loss_delta": selected["v2_instruction_validation_loss"] - before["v2_instruction_validation_loss"],
        "story_loss_delta": selected["story_validation_loss"] - before["story_validation_loss"],
        "independent_exact_passes_delta": selected["independent_v2"]["passed"] - before["independent_v2"]["passed"],
        "independent_protocol_valid_delta": selected["independent_v2"]["valid_actions"] - before["independent_v2"]["valid_actions"],
        "interpretation": "Negative loss delta and positive exact-pass delta are improvements in these measured settings only.",
    }
    encoded = json.dumps(report, indent=2, ensure_ascii=False) + "\n"
    for path in outputs:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("x", encoding="utf-8") as stream:
            stream.write(encoded)
    print(json.dumps({"event": "comparison_saved", "records": len(report["records"]),
                      "public_rows": "Independent fixtures and MIT synthetic validation only; no imported dataset rows or machine paths."}), flush=True)


if __name__ == "__main__":
    main()

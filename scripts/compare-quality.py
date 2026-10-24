#!/usr/bin/env python3
"""Compare immutable local checkpoints on fixed diagnostics and language regression.

No training, tools, downloads, model selection or private chat ingestion. Reports
contain public fixture generations and artifact hashes, not machine paths or raw
imported data. Run only with original, licensed diagnostic suites.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
from pathlib import Path
import tempfile

from silivra.agent import MLXBackend, POLICY
from silivra.checkpoint import resolve_checkpoint, sha256, verify_checkpoint
from silivra.data import TokenDataset
from silivra.evaluation import case_messages, evaluate_assistant, validate_cases
from silivra.finetune import _manifest, evaluate_loss as instruction_loss
from silivra.instructions import InstructionDataset, _check_splits, parse_action
from silivra.train import evaluate_loss as story_loss


def rows(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prior_instruction_validation(model, data, manifest):
    """Use the historical fixed sample only when the full prepared context fits.

    Never crop the prior corpus to make a newer model appear comparable. The
    sampler settings match the earlier v2 measurement (4 examples x 24 batches).
    """
    if model.config.max_seq_len < manifest["max_seq_len"]:
        return {"prior_instruction_validation_loss": None,
                "prior_instruction_unavailable_reason":
                    "Prepared prior instruction context exceeds this checkpoint; no truncation or extrapolation used."}
    dataset = InstructionDataset(data / manifest["splits"]["valid"]["path"],
                                 model.config.max_seq_len, model.config.vocab_size, bucket_size=64)
    return {"prior_instruction_validation_loss": instruction_loss(model, dataset, 4, 24)}


def synthetic_validation_cases(records):
    """Expose every original v3 validation decision in deterministic ID order."""
    cases = []
    for record in sorted(records, key=lambda item: item["id"]):
        source = record["source"]
        if (not record["id"].startswith("v3-") or source.get("license") != "MIT"
                or source.get("revision") != "curriculum-v3"
                or record["messages"][0] != {"role": "system", "content": POLICY}):
            raise ValueError("Synthetic validation requires original v3 MIT records with the fixed policy.")
        expected = parse_action(record["messages"][-1]["content"])
        if set(expected) != {"final"}:
            raise ValueError("The v3 synthetic validation contains no tool proposals.")
        cases.append({"id": "synthetic-" + record["id"],
                      "category": record["id"].removeprefix("v3-").rsplit("-", 1)[0],
                      "messages": record["messages"][1:-1], "tools": [], "expected": expected})
    validate_cases(cases)
    return cases


def synthetic_validation_from_mixed(records):
    """Grade only original v3 decisions, never export rehearsal source text."""
    return synthetic_validation_cases([record for record in records if record["id"].startswith("v3-")])


def split_composition(records):
    original = sum(record["id"].startswith("v3-") for record in records)
    return {"records": len(records), "original_v3_records": original,
            "prior_or_other_records": len(records) - original}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--before", type=Path, required=True)
    parser.add_argument("--after", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--prior-data", type=Path, default=Path("data/instructions-v2"))
    parser.add_argument("--story-data", type=Path, default=Path("data/tinystories"))
    parser.add_argument("--suite", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-new-tokens", type=int, default=96)
    args = parser.parse_args(argv)
    args.data = args.data.resolve()
    args.prior_data = args.prior_data.resolve()
    args.story_data = args.story_data.resolve()
    if args.output.exists():
        raise FileExistsError("Comparison output must be new.")
    if len({path.stem for path in args.suite}) != len(args.suite):
        raise ValueError("Suite filenames must have unique stems.")
    if any(path.stem == "synthetic_validation" for path in args.suite):
        raise ValueError("synthetic_validation is reserved for the complete v3 validation split.")
    # Reuse the public exact-prefix audit, without importing MLX until inference.
    spec = importlib.util.spec_from_file_location("v2_measure", Path(__file__).with_name("measure-assistant-v2.py"))
    audit = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(audit)
    manifest = _manifest(args.data)
    prior = _manifest(args.prior_data)
    source = lambda data, split, m: rows(data / m["splits"][split]["source_path"])
    training, valid = source(args.data, "train", manifest), source(args.data, "valid", manifest)
    _check_splits(training, valid)
    prior_training = source(args.prior_data, "train", prior)
    synthetic = synthetic_validation_from_mixed(valid)
    synthetic_overlap = audit.check_diagnostic_overlap(synthetic, training)
    prior_prefixes = set()
    for record in prior_training:
        prior_prefixes.update(audit.user_prefixes(record["messages"]))
    prior_overlaps = [case["id"] for case in synthetic
                      if prior_prefixes.intersection(audit.user_prefixes(case_messages(case)))]
    synthetic_overlap.update(prior_training_exact_overlaps=len(prior_overlaps),
                             prior_training_overlap_case_ids=prior_overlaps)
    suites = []
    for path in args.suite:
        cases = rows(path)
        validate_cases(cases)
        overlap = audit.check_diagnostic_overlap(cases, training + prior_training)
        suites.append({"name": path.stem, "sha256": sha256(path), "cases": len(cases), "leakage_audit": overlap})
    story_manifest = json.loads((args.story_data / "manifest.json").read_text())
    story = args.story_data / story_manifest["splits"]["valid"]["path"]
    if sha256(story) != story_manifest["splits"]["valid"]["sha256"]:
        raise ValueError("Story validation checksum mismatch.")
    checkpoints = [resolve_checkpoint(args.before, "best"), resolve_checkpoint(args.after, "best")]
    states = [verify_checkpoint(path) for path in checkpoints]
    token_hash = manifest["tokenizer"]["sha256"]
    if (prior["tokenizer"]["sha256"] != token_hash
            or story_manifest["tokenizer"]["sha256"] != token_hash
            or any(sha256(path / "tokenizer.json") != token_hash for path in checkpoints)):
        raise ValueError("Original tokenizer lineage mismatch.")
    for state, data in zip(states, (args.prior_data, args.data)):
        if state.get("data_manifest_sha256") != sha256(data / "manifest.json"):
            raise ValueError("Checkpoint does not bind the specified instruction data manifest.")
        if state["model_config"]["max_seq_len"] < manifest["max_seq_len"]:
            raise ValueError("Prepared instruction validation exceeds a checkpoint context.")
    immutable = [{name: sha256(path / name) for name in ("model.safetensors", "state.json", "tokenizer.json")}
                 for path in checkpoints]
    if states[1].get("base_checkpoint", {}).get("weights_sha256") != immutable[0]["model.safetensors"]:
        raise ValueError("After checkpoint must descend directly from the before checkpoint.")
    report = {"format_version": 1, "purpose": "Small held-out diagnostics; not a general intelligence benchmark.",
              "generation": {"temperature": 0, "top_k": 0, "max_new_tokens": args.max_new_tokens},
              "selection": "after best.json selected only by fixed instruction validation loss",
              "suites": suites, "tools_executed": False,
              "data_composition": {"train": split_composition(training), "valid": split_composition(valid),
                                   "boundary": "Instruction validation loss uses the full prepared validation split. Synthetic task accuracy uses only its original v3 subset; rehearsal source text is never emitted."},
              "provenance": {"script_sha256": sha256(Path(__file__)),
                             "evaluator_sha256": sha256(Path("src/silivra/evaluation.py")),
                             "instruction_manifest_sha256": sha256(args.data / "manifest.json"),
                             "prior_manifest_sha256": sha256(args.prior_data / "manifest.json"),
                             "story_manifest_sha256": sha256(args.story_data / "manifest.json"),
                             "tokenizer_sha256": token_hash},
              "loss_settings": {"story": {"batches": 12, "batch_size": 16, "sequence_length": 256, "seed": 12345},
                                "instruction": {"batches": 24, "batch_size": 8, "seed": 12345, "bucket_size": 64,
                                                "boundary": "Full current prepared validation split, including any rehearsal records."},
                                "prior_instruction": {"batches": 24, "batch_size": 4, "seed": 12345,
                                                      "bucket_size": 64,
                                                      "boundary": "Prior v2 assistant-only validation corpus; detects broader instruction regression when full context fits."}},
              "records": []}
    with tempfile.TemporaryDirectory(prefix="silivra-quality-") as temporary:
        synthetic_path = Path(temporary) / "synthetic_validation.jsonl"
        synthetic_path.write_text("".join(json.dumps(case, ensure_ascii=False, separators=(",", ":")) + "\n"
                                          for case in synthetic), encoding="utf-8")
        suites.append({"name": "synthetic_validation", "sha256": sha256(synthetic_path), "cases": len(synthetic),
                       "leakage_audit": synthetic_overlap,
                       "selection": "All original v3 records from the full prepared validation split; excludes rehearsal records.",
                       "boundary": "Complete v3 synthetic validation subset; held out from current mixed training, shared templates and narrow skills. Historical v2 exact overlaps are disclosed separately; not an independent general benchmark."})
        for label, checkpoint, state, hashes in zip(("before", "after"), checkpoints, states, immutable):
            backend = MLXBackend(checkpoint, max_new_tokens=args.max_new_tokens)
            record = {"label": label, "checkpoint": checkpoint.name, "hashes": hashes,
                      "step": state["step"], "context_window": backend.model.config.max_seq_len,
                      "supervised_tokens_seen": state.get("supervised_tokens_seen"),
                      "training_seconds": state["training_seconds"],
                      "diagnostics": {path.stem: evaluate_assistant(backend, path)
                                      for path in [*args.suite, synthetic_path]}}
            if any(record["diagnostics"][suite["name"]]["suite_sha256"] != suite["sha256"] for suite in suites):
                raise RuntimeError("Diagnostic suite changed after the leakage audit.")
            dataset = InstructionDataset(args.data / manifest["splits"]["valid"]["path"],
                                         backend.model.config.max_seq_len, backend.model.config.vocab_size, bucket_size=64)
            record["instruction_validation_loss"] = instruction_loss(backend.model, dataset, 8, 24)
            record.update(prior_instruction_validation(backend.model, args.prior_data, prior))
            record["story_validation_loss"] = story_loss(backend.model, TokenDataset(story, 256), 16, 12)
            report["records"].append(record)
            print(json.dumps({"event": "comparison_model", "label": label,
                              "passed": {name: value["passed"] for name, value in record["diagnostics"].items()},
                              "instruction_loss": record["instruction_validation_loss"],
                              "prior_instruction_loss": record["prior_instruction_validation_loss"],
                              "story_loss": record["story_validation_loss"]}), flush=True)
            del backend, dataset
            gc.collect()
            import mlx.core as mx
            mx.clear_cache()
    if any(any(sha256(path / name) != digest for name, digest in hashes.items())
           for path, hashes in zip(checkpoints, immutable)):
        raise RuntimeError("Checkpoint changed during evaluation.")
    report["immutable_checkpoints_reverified"] = True
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        json.dump(report, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"event": "comparison_saved", "models": len(report["records"])}), flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Measure native checkpoints on frozen public diagnostics and narrow validation.

No tools are executed. Reports contain public synthetic generations and hashes,
not imported corpus text, private transcripts, or machine-specific paths.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
import gc
import json
from pathlib import Path
import tempfile
from silivra.agent import MLXBackend, parse_action
from silivra.checkpoint import resolve_checkpoint, sha256, verify_checkpoint
from silivra.data import TokenDataset
from silivra.evaluation import evaluate_assistant, validate_cases
from silivra.finetune import _manifest, evaluate_loss as instruction_loss
from silivra.instructions import InstructionDataset, _check_splits
from silivra.tools import ToolRegistry
from silivra.train import evaluate_loss as story_loss


def read_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def prefixes(messages):
    users, result = [], set()
    for message in messages:
        if message["role"] == "user":
            users.append(" ".join(message["content"].split()))
        elif message["role"] == "assistant" and users:
            result.add(tuple(users))
    return result


def validation_cases(records, limit_per_family=8):
    """Fixed ID-order sampling; never inspect a model output to select cases."""
    registry = ToolRegistry(Path.cwd(), allow_network=True)
    specs = {s["function"]["name"]: s for s in registry.schemas()}
    groups = defaultdict(list)
    for row in sorted(records, key=lambda r: r["id"]):
        if row["source"]["revision"] != "native-english-v5-runtime":
            raise ValueError("Narrow suite must contain original v5 records only")
        family = row["id"].split("-", 2)[1]
        system = row["messages"][0]["content"]
        tools = []
        if " Tools: " in system:
            names = [tool["name"] for tool in json.loads(system.split(" Tools: ",1)[1])]
            tools = [specs[name] for name in names]
        for index, message in enumerate(row["messages"]):
            if message["role"] == "assistant":
                category = family if index == 2 else family + "-observation"
                groups[category].append({"id": row["id"] + f"-turn-{index}",
                    "category": category, "messages": row["messages"][1:index],
                    "tools": tools, "expected":parse_action(message["content"])})
    cases = [case for family in sorted(groups) for case in groups[family][:limit_per_family]]
    validate_cases(cases)
    return cases


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--before", type=Path, required=True)
    p.add_argument("--after", type=Path, required=True)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--suite", type=Path, action="append", default=[])
    args = p.parse_args(argv)
    if args.output.exists():
        p.error("Output must be new")
    manifest = _manifest(args.data.resolve())
    train = read_rows(args.data / manifest["splits"]["train"]["source_path"])
    valid = read_rows(args.data / manifest["splits"]["valid"]["source_path"])
    _check_splits(train, valid)
    denied = set().union(*(prefixes(row["messages"]) for row in train))
    narrow = validation_cases(valid)
    original_hashes = {}
    selected = [resolve_checkpoint(path, "best") for path in (args.before,args.after)]
    for path in selected:
        state = verify_checkpoint(path)
        original_hashes[str(path)] = {name:sha256(path/name) for name in ("model.safetensors","tokenizer.json","state.json")}
        if sha256(path/"tokenizer.json") != manifest["tokenizer"]["sha256"]:
            raise ValueError("Native tokenizer lineage mismatch")
    after_state = verify_checkpoint(selected[1])
    if after_state.get("data_manifest_sha256") != sha256(args.data/"manifest.json"):
        raise ValueError("After checkpoint does not bind the supplied prepared data")
    if after_state["base_checkpoint"]["weights_sha256"] != sha256(selected[0]/"model.safetensors"):
        raise ValueError("After checkpoint must descend from before checkpoint")
    report = {"format_version":1,"purpose":"Bounded native original-model English experiment; not general intelligence evidence",
        "data_manifest_sha256":sha256(args.data/"manifest.json"),"script_sha256":sha256(Path(__file__)),
        "training": {"step":after_state["step"],"seconds":after_state["training_seconds"],"config":after_state["training_config"],"supervised_tokens":after_state["supervised_tokens_seen"],"input_tokens":after_state["input_tokens_seen"]},
        "selection":"best.json from fixed instruction validation loss, before diagnostic generations",
        "narrow_suite":"Up to 8 validation decisions per family by stable record ID; shared templates, operands/topics held out from current curriculum",
        "generation":{"max_new_tokens":96,"temperature":0,"top_k":0},"tools_executed":False,"records":[]}
    # Audit exact user prefixes only. Semantic overlap and prior pretraining
    # contamination cannot be excluded by this small audit.
    report["exact_current_training_prefix_overlap"] = {}
    for path in args.suite:
        cases = read_rows(path)
        validate_cases(cases)
        overlap = []
        for case in cases:
            messages = case.get("messages", [{"role":"user","content":case.get("prompt","")}])
            if prefixes([*messages,{"role":"assistant","content":""}]) & denied:
                overlap.append(case["id"])
        report["exact_current_training_prefix_overlap"][path.stem] = overlap
        if overlap:
            raise ValueError("Independent diagnostic overlaps current training prompts")
    story_manifest = json.loads(Path("data/tinystories/manifest.json").read_text())
    story_file = Path("data/tinystories")/story_manifest["splits"]["valid"]["path"]
    if sha256(story_file) != story_manifest["splits"]["valid"]["sha256"]:
        raise ValueError("Story validation checksum mismatch")
    prior_data = Path("data/instructions-v2")
    prior_manifest = _manifest(prior_data.resolve())
    if any(item["tokenizer"]["sha256"] != manifest["tokenizer"]["sha256"]
           for item in (story_manifest, prior_manifest)):
        raise ValueError("Retention corpora must use the unchanged native tokenizer")
    prior_train = read_rows(prior_data/prior_manifest["splits"]["train"]["source_path"])
    prior_prefixes = set().union(*(prefixes(row["messages"]) for row in prior_train))
    report["prior_training_exact_prefix_overlap"] = {}
    for name, cases in [(path.stem, read_rows(path)) for path in args.suite] + [("narrow-validation", narrow)]:
        report["prior_training_exact_prefix_overlap"][name] = [case["id"] for case in cases
            if prefixes([*case.get("messages", [{"role":"user","content":case.get("prompt","")}]),
                         {"role":"assistant","content":""}]) & prior_prefixes]
    report["contamination_boundary"] = "Exact user-prefix checks only. Narrow shared-template tasks are held out from current curriculum; prior training overlaps are disclosed separately. No semantic-contamination guarantee."
    report["evaluator_sha256"] = sha256(Path("src/silivra/evaluation.py"))
    report["tokenizer_sha256"] = manifest["tokenizer"]["sha256"]
    with tempfile.TemporaryDirectory(prefix="native-english-eval-") as temporary:
        narrow_path = Path(temporary)/"narrow-validation.jsonl"
        narrow_path.write_text("".join(json.dumps(case,separators=(",",":"))+"\n" for case in narrow))
        for label,checkpoint in zip(("before","after"),selected):
            backend = MLXBackend(checkpoint,max_new_tokens=96)
            item = {"label":label,"checkpoint":checkpoint.name,"hashes":original_hashes[str(checkpoint)],
                    "diagnostics": {path.stem:evaluate_assistant(backend,path) for path in [*args.suite,narrow_path]}}
            item["context_window"] = backend.model.config.max_seq_len
            for key, directory, data_manifest, batch_size, batches in (
                    ("instruction_validation_loss", args.data, manifest, 8, 16),
                    ("prior_instruction_loss", prior_data, prior_manifest, 4, 24)):
                if data_manifest["max_seq_len"] > backend.model.config.max_seq_len:
                    item[key] = None
                    item[key + "_unavailable"] = "Full prepared validation context exceeds this checkpoint; no truncation or extrapolation."
                else:
                    item[key] = instruction_loss(backend.model, InstructionDataset(
                        directory/data_manifest["splits"]["valid"]["path"], backend.model.config.max_seq_len,
                        backend.model.config.vocab_size, bucket_size=64), batch_size, batches)
            item["story_validation_loss"] = story_loss(backend.model,TokenDataset(story_file,256),16,12)
            print(json.dumps({"label":label,"scores":{name:f'{r["passed"]}/{r["cases"]}' for name,r in item["diagnostics"].items()},"instruction_loss":item["instruction_validation_loss"],"story_loss":item["story_validation_loss"]}),flush=True)
            report["records"].append(item)
            del backend
            gc.collect()
            import mlx.core as mx
            mx.clear_cache()
    for path in selected:
        if any(sha256(path/name)!=value for name,value in original_hashes[str(path)].items()):
            raise ValueError("Checkpoint changed during measurement")
    report["checkpoints_immutable"] = True
    args.output.parent.mkdir(parents=True,exist_ok=True)
    with args.output.open("x") as stream:
        json.dump(report,stream,ensure_ascii=False,indent=2)
        stream.write("\n")

if __name__ == "__main__":
    main()

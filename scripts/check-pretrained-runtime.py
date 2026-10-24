#!/usr/bin/env python3
"""Bounded real-model checks for prefix reuse, streaming and interruption.

Uses public synthetic prompts and pure arithmetic only. No training, network,
filesystem tools, persistent conversation memory, shell or generated-code execution.
"""

import argparse
import ast
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import time

from silivra.agent import GenerationBudgetError, GenerationCancelled, POLICY, parse_action
from silivra.pretrained import PretrainedMLXBackend
from silivra.tools import ToolRegistry


ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/silivra-qwen3-8b"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Runtime verification output must be new.")
    started = time.monotonic()
    backend = PretrainedMLXBackend(args.run, max_new_tokens=256)
    report = {"format_version": 1, "verification_date": datetime.now(timezone.utc).date().isoformat(),
              "purpose": "Bounded live runtime checks, not a model intelligence benchmark",
              "source_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                for name in ["src/silivra/pretrained.py", "src/silivra/structured.py",
                                             "src/silivra/prompt_cache.py", "scripts/check-pretrained-runtime.py"]},
              "checks": {}, "records": []}

    def generate(prompt, tools=(), **options):
        deltas = []
        raw = backend.generate([{"role": "system", "content": POLICY},
                                {"role": "user", "content": prompt}], list(tools),
                               on_text=deltas.append, **options)
        action = parse_action(raw)
        record = {"prompt": prompt, "raw_output": raw, "action": action,
                  "stream_matches": "".join(deltas) == raw, "metrics": dict(backend.last_metrics)}
        report["records"].append(record)
        return record

    registry = ToolRegistry(ROOT)
    calculator = next(item for item in registry.schemas() if item["function"]["name"] == "calculate")
    cases = [("plain", "Reply with exactly READY.", []),
             ("code", "Write Python function identity(value) that returns value. Reply only with function source, without Markdown.", []),
             ("calculator", "Use calculate to evaluate 19 * 23.", [calculator])]
    for name, prompt, tools in cases:
        backend.clear_prompt_cache()
        cold = generate(prompt, tools)
        warm = generate(prompt, tools)
        report["checks"][name + "_cache_equal"] = cold["raw_output"] == warm["raw_output"]
        report["checks"][name + "_reused_prefix"] = (
            cold["metrics"]["cached_prompt_tokens"] == 0
            and warm["metrics"]["cached_prompt_tokens"] > 0
            and warm["metrics"]["prefill_tokens"] >= 1)
        report["checks"][name + "_stream"] = cold["stream_matches"] and warm["stream_matches"]
        if name == "plain":
            report["checks"]["plain_content"] = warm["action"] == {"final": "READY"}
        elif name == "code":
            try:
                body = ast.parse(warm["action"]["final"]).body
                function = body[0]
                report["checks"]["code_content"] = (len(body) == 1 and isinstance(function, ast.FunctionDef)
                    and function.name == "identity" and len(function.body) == 1
                    and isinstance(function.body[0], ast.Return)
                    and isinstance(function.body[0].value, ast.Name)
                    and function.body[0].value.id == "value")
            except (ValueError, SyntaxError, KeyError, IndexError):
                report["checks"]["code_content"] = False
        else:
            action = warm["action"]
            result = registry.execute(action.get("tool"), action.get("arguments", {}))
            report["checks"]["calculator_content"] = (action.get("tool") == "calculate"
                and result.get("ok") is True and result["result"]["result"] == "437")

    cancel = False
    def cancel_after_delta(_):
        nonlocal cancel
        cancel = True
    try:
        backend.generate([{"role": "system", "content": POLICY},
                          {"role": "user", "content": "Explain Python dictionaries in a long paragraph."}],
                         [], on_text=cancel_after_delta, cancelled=lambda: cancel)
        report["checks"]["cancelled"] = False
    except GenerationCancelled:
        report["checks"]["cancelled"] = True
    report["checks"]["cancel_discards_cache"] = backend._prompt_cache.retained_token_count == 0
    recovered = generate("Reply with exactly RECOVERED.")
    report["checks"]["recovery_starts_cold"] = recovered["metrics"]["cached_prompt_tokens"] == 0
    backend.clear_prompt_cache()
    reference = generate("Reply with exactly RECOVERED.")
    report["checks"]["recovered_after_cancel"] = (
        set(recovered["action"]) == {"final"}
        and recovered["action"] == reference["action"])
    backend.max_new_tokens = 8
    try:
        generate("Write a detailed paragraph of at least one hundred words about Python iterators.")
        report["checks"]["output_budget"] = False
    except GenerationBudgetError:
        report["checks"]["output_budget"] = backend.last_metrics.get("finish_reason") == "length"
    report["checks"]["budget_discards_cache"] = backend._prompt_cache.retained_token_count == 0
    backend.clear_prompt_cache()
    report["seconds_including_load"] = time.monotonic() - started
    report["all_passed"] = all(report["checks"].values())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as target:
        json.dump(report, target, indent=2)
        target.write("\n")
    print(json.dumps({"all_passed": report["all_passed"], "checks": report["checks"],
                      "seconds": report["seconds_including_load"]}), flush=True)
    if not report["all_passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

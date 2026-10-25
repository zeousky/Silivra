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


def cache_profile(model):
    """Inspect native cache support without treating hybrid state as trimmable KV.

    Recurrent/hybrid layer state cannot be trimmed to an arbitrary token prefix.
    Such a model must run cold on every request; absence of reuse is intentional,
    and retaining that state would be a correctness failure.
    """
    from mlx_lm.models.cache import can_trim_prompt_cache, make_prompt_cache
    from silivra.prompt_cache import snapshot_cache_supported
    cache = make_prompt_cache(model)
    return {"trimmable": bool(can_trim_prompt_cache(cache)),
            "snapshot_supported": snapshot_cache_supported(cache),
            "layer_types": sorted({type(layer).__name__ for layer in cache}),
            "layer_count": len(cache)}


def cache_metrics_pass(profile, cold, warm):
    """Check the expected reuse or mandatory cold behavior in real metrics."""
    cold, warm = cold["metrics"], warm["metrics"]
    if profile["trimmable"] or profile.get("snapshot_supported", False):
        return (cold["cached_prompt_tokens"] == 0
                and warm["cached_prompt_tokens"] > 0
                and warm["prefill_tokens"] >= 1
                and cold["cache_retained"] is True
                and warm["cache_retained"] is True)
    return all(metrics["cached_prompt_tokens"] == 0
               and metrics["prefill_tokens"] == metrics["prompt_tokens"]
               and metrics["cache_retained"] is False
               and metrics["cache_retained_tokens"] == 0
               for metrics in (cold, warm))


def identity_content_pass(source):
    """Recognize the requested tiny function without executing generated code."""
    try:
        body = ast.parse(source).body
    except (TypeError, ValueError, SyntaxError):
        return False
    if len(body) != 1 or not isinstance(body[0], ast.FunctionDef):
        return False
    function = body[0]
    arguments = function.args
    return (function.name == "identity" and not function.decorator_list
            and not function.returns and not getattr(function, "type_params", [])
            and not arguments.posonlyargs and len(arguments.args) == 1
            and arguments.args[0].arg == "value" and arguments.args[0].annotation is None
            and not arguments.vararg and not arguments.kwarg
            and not arguments.kwonlyargs and not arguments.defaults and not arguments.kw_defaults
            and len(function.body) == 1 and isinstance(function.body[0], ast.Return)
            and isinstance(function.body[0].value, ast.Name)
            and function.body[0].value.id == "value")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=Path("runs/silivra-qwen3-8b"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError("Runtime verification output must be new.")
    started = time.monotonic()
    backend = PretrainedMLXBackend(args.run, max_new_tokens=256)
    profile = cache_profile(backend.model)
    report = {"format_version": 2, "verification_date": datetime.now(timezone.utc).date().isoformat(),
              "purpose": "Bounded live runtime checks, not a model intelligence benchmark",
              "model": {key: backend.metadata[key] for key in
                        ("model_id", "revision", "original_model_id", "parameter_count", "quantization_bits")},
              "native_cache": profile,
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
        cache_check = ("_snapshot_reused" if profile["snapshot_supported"]
                       else "_reused_prefix" if profile["trimmable"] else "_stays_cold")
        report["checks"][name + cache_check] = cache_metrics_pass(profile, cold, warm)
        report["checks"][name + "_stream"] = cold["stream_matches"] and warm["stream_matches"]
        if name == "plain":
            report["checks"]["plain_content"] = warm["action"] == {"final": "READY"}
        elif name == "code":
            report["checks"]["code_content"] = identity_content_pass(warm["action"].get("final"))
        else:
            action = warm["action"]
            result = registry.execute(action.get("tool"), action.get("arguments", {}))
            report["checks"]["calculator_content"] = (action.get("tool") == "calculate"
                and result.get("ok") is True and result["result"]["result"] == "437")

    backend.clear_prompt_cache()
    generate("Reply with exactly FIRST.")
    changed = generate("Reply with exactly SECOND.")
    backend.clear_prompt_cache()
    changed_reference = generate("Reply with exactly SECOND.")
    report["checks"]["changed_user_cache_equal"] = changed["action"] == changed_reference["action"]
    report["checks"]["changed_user_reuses_only_input_prefix"] = (
        changed["metrics"]["cached_prompt_tokens"] > 0
        if profile["trimmable"] or profile["snapshot_supported"]
        else changed["metrics"]["cached_prompt_tokens"] == 0)

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

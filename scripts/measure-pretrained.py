#!/usr/bin/env python3
"""Bounded offline task measurements for an explicitly identified pretrained backend.

Uses synthetic public fixtures, real read-only tools and temporary memory. No
training, network, model-as-judge, shell, eval or exec. Python answers are run by a
small bounded AST interpreter, whose deliberately limited subset is documented.
"""
from __future__ import annotations

import argparse
import ast
from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import statistics
import tempfile
import time

from silivra.agent import Agent, AgentConfig, project_scope
from silivra.memory import MemoryStore
from silivra.tools import ToolRegistry

ROOT = Path(__file__).resolve().parents[1]
SUITE = ROOT / "examples/evaluations/pretrained-v1.jsonl"


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class RestrictedPython:
    """Interpret bounded simple functions; never compile or execute Python code.

    Allows literals, local variables, comparisons, arithmetic, indexing, lists,
    tuples, basic if/for/return, append, and selected pure builtins. No imports,
    arbitrary attributes, exceptions, comprehensions, recursion, or while loops.
    The global visit and collection bounds apply to every call.
    """
    class Returned(Exception):
        def __init__(self, value):
            self.value = value

    def __init__(self, source, name):
        if not isinstance(source, str) or len(source) > 8000:
            raise ValueError("Python answer exceeds checker text budget")
        tree = ast.parse(source)
        if len(list(ast.walk(tree))) > 300:
            raise ValueError("Python answer exceeds checker AST budget")
        if len(tree.body) != 1 or not isinstance(tree.body[0], ast.FunctionDef):
            raise ValueError("Expected exactly one function definition")
        self.function = tree.body[0]
        if (self.function.name != name or self.function.decorator_list
                or self.function.args.vararg or self.function.args.kwarg
                or self.function.args.kwonlyargs or self.function.args.defaults
                or self.function.args.posonlyargs):
            raise ValueError("Unsupported function signature")
        self.remaining = 0

    def visit(self):
        self.remaining -= 1
        if self.remaining < 0:
            raise ValueError("Python checker operation budget exceeded")

    def call(self, args):
        self.remaining = 3000
        names = [argument.arg for argument in self.function.args.args]
        if len(args) != len(names):
            raise ValueError("Incorrect function argument count")
        env = dict(zip(names, deepcopy(args)))
        try:
            self.block(self.function.body, env)
        except self.Returned as result:
            return result.value
        return None

    def block(self, statements, env):
        for node in statements:
            self.visit()
            if isinstance(node, ast.Return):
                raise self.Returned(self.expr(node.value, env) if node.value else None)
            if isinstance(node, ast.If):
                self.block(node.body if self.expr(node.test, env) else node.orelse, env)
            elif isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name):
                env[node.targets[0].id] = self.expr(node.value, env)
            elif isinstance(node, ast.AugAssign) and isinstance(node.target, ast.Name):
                # Common += syntax has the same bounded arithmetic semantics as
                # x = x + value; attributes/subscripts remain unsupported.
                value = ast.BinOp(left=ast.Name(id=node.target.id, ctx=ast.Load()),
                                  op=node.op, right=node.value)
                env[node.target.id] = self.expr(value, env)
            elif isinstance(node, ast.For) and isinstance(node.target, ast.Name) and not node.orelse:
                values = self.expr(node.iter, env)
                if not isinstance(values, (list, tuple, str)) or len(values) > 100:
                    raise ValueError("Unsupported or oversized loop input")
                for value in values:
                    env[node.target.id] = value
                    self.block(node.body, env)
            elif isinstance(node, ast.Expr):
                # A constant here is a harmless docstring.
                self.expr(node.value, env)
            elif not isinstance(node, (ast.If, ast.Return)):
                raise ValueError("Statement outside the bounded checker subset")

    def expr(self, node, env):
        self.visit()
        if isinstance(node, ast.Constant) and type(node.value) in (str, int, float, bool, type(None)):
            if isinstance(node.value, str) and len(node.value) > 1000:
                raise ValueError("Oversized literal")
            if type(node.value) in (int, float) and abs(node.value) > 1000000:
                raise ValueError("Oversized number")
            return node.value
        if isinstance(node, ast.Name) and node.id in env:
            return env[node.id]
        if isinstance(node, (ast.List, ast.Tuple)) and len(node.elts) <= 100:
            result = [self.expr(value, env) for value in node.elts]
            return tuple(result) if isinstance(node, ast.Tuple) else result
        if isinstance(node, ast.IfExp):
            return self.expr(node.body if self.expr(node.test, env) else node.orelse, env)
        if isinstance(node, ast.UnaryOp):
            value = self.expr(node.operand, env)
            if isinstance(node.op, ast.Not):
                return not value
            if isinstance(node.op, ast.USub) and type(value) in (int, float):
                return -value
        if isinstance(node, ast.BinOp):
            left, right = self.expr(node.left, env), self.expr(node.right, env)
            if type(left) not in (int, float) or type(right) not in (int, float):
                raise ValueError("Only bounded numeric arithmetic is supported")
            operations = {ast.Add: lambda: left + right, ast.Sub: lambda: left - right,
                          ast.Mult: lambda: left * right, ast.Div: lambda: left / right,
                          ast.FloorDiv: lambda: left // right, ast.Mod: lambda: left % right}
            operation = operations.get(type(node.op))
            if operation is not None:
                result = operation()
                if abs(result) > 1000000:
                    raise ValueError("Arithmetic result exceeds checker budget")
                return result
        if isinstance(node, ast.BoolOp):
            if isinstance(node.op, ast.And):
                return all(self.expr(value, env) for value in node.values)
            if isinstance(node.op, ast.Or):
                return any(self.expr(value, env) for value in node.values)
        if isinstance(node, ast.Compare):
            left = self.expr(node.left, env)
            for operator, expression in zip(node.ops, node.comparators):
                right = self.expr(expression, env)
                operations = {ast.Eq: lambda: left == right, ast.NotEq: lambda: left != right,
                              ast.Lt: lambda: left < right, ast.LtE: lambda: left <= right,
                              ast.Gt: lambda: left > right, ast.GtE: lambda: left >= right,
                              ast.In: lambda: left in right, ast.NotIn: lambda: left not in right}
                operation = operations.get(type(operator))
                if operation is None or not operation():
                    return False
                left = right
            return True
        if isinstance(node, ast.Call) and not node.keywords:
            args = [self.expr(arg, env) for arg in node.args]
            functions = {"min": min, "max": max, "len": len, "sum": sum,
                         "sorted": sorted, "list": list, "set": set}
            if isinstance(node.func, ast.Name) and node.func.id in functions:
                result = functions[node.func.id](*args)
                if isinstance(result, (list, tuple, set, str)) and len(result) > 100:
                    raise ValueError("Collection exceeds checker budget")
                return result
            if (isinstance(node.func, ast.Attribute) and node.func.attr == "append"
                    and isinstance(node.func.value, ast.Name) and len(args) == 1):
                target = env.get(node.func.value.id)
                if isinstance(target, list) and len(target) < 100:
                    target.append(args[0])
                    return None
        if isinstance(node, ast.Subscript):
            target, index = self.expr(node.value, env), self.expr(node.slice, env)
            if isinstance(target, (list, tuple, str)) and type(index) is int:
                return target[index]
        raise ValueError("Expression outside the bounded checker subset")


def grade_final(case, result):
    """Protocol validity is necessary; final content and observed tools decide success."""
    if result.get("status") != "completed" or not isinstance(result.get("answer"), str):
        return False, "Turn did not complete with a validated final answer"
    answer, specification = result["answer"].strip(), case["grade"]
    kind = specification["kind"]
    try:
        if kind == "exact":
            passed = answer == specification["expected"]
        elif kind == "one_of":
            passed = answer.casefold() in [value.casefold() for value in specification["expected"]]
        elif kind == "json":
            passed = (json.dumps(json.loads(answer), sort_keys=True) ==
                      json.dumps(specification["expected"], sort_keys=True))
        elif kind == "python":
            interpreter = RestrictedPython(answer, specification["function"])
            passed = all(interpreter.call(test["args"]) == test["expected"] for test in specification["tests"])
        else:
            raise ValueError("Unknown grade kind")
    except (ValueError, TypeError, SyntaxError, KeyError, IndexError, ZeroDivisionError, OverflowError, RecursionError) as error:
        return False, f"Final failed {kind} checker: {type(error).__name__}"
    if not passed:
        return False, f"Final failed {kind} content check"
    events = result.get("events", [])
    for required in case.get("required_tools", []):
        observed = [event for event in events if event["action"]["tool"] == required["name"]]
        if not observed:
            return False, f"Required real tool was not used: {required['name']}"
        if "ok" in required and not any(event["observation"].get("ok") is required["ok"] for event in observed):
            return False, f"Expected tool observation was not received: {required['name']}"
    return True, "Final content and required observations passed"


def read_cases(path):
    cases = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if not 1 <= len(cases) <= 32 or len({case["id"] for case in cases}) != len(cases):
        raise ValueError("Evaluation needs 1–32 uniquely identified cases")
    for case in cases:
        if not 1 <= len(case["prompts"]) <= 3 or not set(case.get("tools", [])) <= {"read_file", "list_files", "search_files", "calculate"}:
            raise ValueError("Evaluation permits bounded prompts, read-only file tools and pure arithmetic only")
        for path, content in case.get("files", {}).items():
            relative = Path(path)
            if relative.is_absolute() or ".." in relative.parts or not isinstance(content, str) or len(content) > 8000:
                raise ValueError("Invalid synthetic workspace fixture")
    return cases


class RecordingBackend:
    def __init__(self, backend):
        self.backend, self.generations = backend, []

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def generate(self, messages, tools, **kwargs):
        started = time.monotonic()
        record = {}
        partial, characters = [], 0
        callback = kwargs.get("on_text")

        def capture(delta):
            nonlocal characters
            if isinstance(delta, str) and characters < 64_000:
                fragment = delta[:64_000 - characters]
                partial.append(fragment)
                characters += len(fragment)
            if callback:
                callback(delta)

        kwargs["on_text"] = capture
        self.generations.append(record)
        try:
            raw = self.backend.generate(messages, tools, **kwargs)
            record["raw_output"] = raw
            return raw
        except Exception as error:
            record["error_type"] = type(error).__name__
            if partial:
                record["partial_output"] = "".join(partial)
            raise
        finally:
            record["seconds"] = round(time.monotonic() - started, 6)
            record["metrics"] = deepcopy(getattr(self.backend, "last_metrics", {}))


def evaluate_case(case, backend, temporary):
    workspace = temporary / "workspace"
    workspace.mkdir()
    for name, content in case.get("files", {}).items():
        target = workspace / name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(content, encoding="utf-8")
    recorder = RecordingBackend(backend)
    started = time.monotonic()
    with MemoryStore(temporary / "state" / "memory.sqlite") as memory:
        scope = project_scope(workspace, "synthetic-user")
        other_scope = project_scope(workspace, "synthetic-other-user")
        other_project = project_scope(temporary / "other-workspace", "synthetic-user")
        for operation in case.get("memory", []):
            selected_scope = (other_project if operation.get("other_project") else
                              other_scope if operation.get("other_scope") else scope)
            if operation["operation"] == "remember":
                memory.remember(selected_scope, operation["key"], operation["content"],
                                source="user", source_ref="synthetic-explicit-user-fixture")
            elif operation["operation"] == "forget":
                memory.forget(selected_scope, operation["key"])
            else:
                raise ValueError("Unknown explicit memory operation")
        registry = ToolRegistry(workspace, memory, scope)
        agent = Agent(recorder, registry, memory, scope,
                      config=AgentConfig(max_steps=5, max_seconds=180, history_messages=12,
                                         max_observation_chars=4000, memory_chars=2000),
                      tool_names=case.get("tools", []))
        turns, session = [], None
        for prompt in case["prompts"]:
            result = agent.run(prompt, session_id=session)
            session = result["session_id"]
            turns.append({key: value for key, value in result.items() if key != "session_id"})
        passed, reason = grade_final(case, turns[-1])
        record = {"id": case["id"], "category": case["category"], "prompts": case["prompts"],
                  "passed": passed, "grading_reason": reason,
                  "seconds": round(time.monotonic() - started, 6), "turns": turns,
                  "generations": recorder.generations}
    # Public reports never retain private/temp paths even if a backend error includes them.
    encoded = json.dumps(record, ensure_ascii=False).replace(str(temporary), "<temporary>")
    return json.loads(encoded)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--suite", type=Path, default=SUITE)
    parser.add_argument("--max-cases", type=int, default=32)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    args = parser.parse_args(argv)
    if args.output.exists():
        raise FileExistsError("Evaluation output must be new")
    if not 1 <= args.max_cases <= 32 or not 32 <= args.max_new_tokens <= 1536:
        raise ValueError("Evaluation case/token budgets are out of bounds")
    cases = read_cases(args.suite)[:args.max_cases]
    # Keep the portable grader importable without Apple MLX.
    from silivra.pretrained import PretrainedMLXBackend
    load_started = time.monotonic()
    backend = PretrainedMLXBackend(args.run, max_new_tokens=args.max_new_tokens)
    model_load_seconds = round(time.monotonic() - load_started, 6)
    report = {"format_version": 1, "date_utc": datetime.now(timezone.utc).isoformat(),
              "purpose": "Small practical held-out diagnostic, not a general intelligence benchmark",
              "training": "None. Explicit pretrained inference; original Silivra checkpoints unchanged.",
              "contamination_boundary": "Original synthetic tasks withheld from Silivra training. Upstream pretrained training data cannot be audited for overlap.",
              "execution": "Local model, real temporary read-only tools and scoped memory; no network, eval, exec, or shell",
              "generation": {"temperature": 0, "max_new_tokens": args.max_new_tokens,
                             "max_steps_per_turn": 5, "max_seconds_per_turn": 180},
              "provenance": {"suite_sha256": sha256(args.suite), "script_sha256": sha256(Path(__file__)),
                             "backend_sha256": sha256(ROOT / "src/silivra/pretrained.py"),
                             "structured_sha256": sha256(ROOT / "src/silivra/structured.py"),
                             "prompt_cache_sha256": sha256(ROOT / "src/silivra/prompt_cache.py"),
                             "tools_sha256": sha256(ROOT / "src/silivra/tools.py"),
                             "arithmetic_sha256": sha256(ROOT / "src/silivra/arithmetic.py"),
                             "agent_sha256": sha256(ROOT / "src/silivra/agent.py")},
              "model_load_and_verification_seconds": model_load_seconds,
              "records": []}
    model_provenance = getattr(backend, "metadata", None)
    if isinstance(model_provenance, dict):
        report["model"] = deepcopy(model_provenance)
    started = time.monotonic()
    for case in cases:
        with tempfile.TemporaryDirectory(prefix="silivra-public-evaluation-") as directory:
            record = evaluate_case(case, backend, Path(directory))
        report["records"].append(record)
        print(json.dumps({"event": "case", "id": case["id"], "passed": record["passed"],
                          "seconds": record["seconds"]}), flush=True)
    count = len(cases)
    by_category = {}
    for category in sorted({record["category"] for record in report["records"]}):
        selected = [record for record in report["records"] if record["category"] == category]
        by_category[category] = {"passed": sum(record["passed"] for record in selected), "total": len(selected)}
    report["summary"] = {"passed": sum(record["passed"] for record in report["records"]), "total": count,
                         "accuracy": sum(record["passed"] for record in report["records"]) / count,
                         "categories": by_category, "total_seconds": round(time.monotonic() - started, 6),
                         "median_case_seconds": statistics.median(record["seconds"] for record in report["records"])}
    measurements = [generation["metrics"] for record in report["records"]
                    for generation in record["generations"] if generation.get("metrics")]
    speed = {}
    for key in ("time_to_first_token_seconds", "tokens_per_second", "decode_tokens_per_second"):
        values = [item[key] for item in measurements
                  if isinstance(item.get(key), (int, float)) and not isinstance(item[key], bool)]
        speed[key] = {"median": statistics.median(values), "samples": len(values)} if values else None
    report["summary"]["generation_speed"] = speed
    report["summary"]["timing_boundary"] = (
        "Case wall time includes tools and memory; total excludes model loading/hash verification. "
        "Generation metrics report prefill plus decode throughput separately from decode throughput. "
        "An exact input prefix may be reused within the same model backend; each generation records "
        "cached_prompt_tokens and prefill_tokens. No generated tail or persistent KV cache is reused.")
    # Marker verification provides pinned source identity; never publish machine paths.
    text = json.dumps(report, indent=2, ensure_ascii=False)
    for private in (str(args.run.resolve()), str(ROOT)):
        text = text.replace(private, "<local>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as stream:
        stream.write(text + "\n")
    print(json.dumps({"event": "evaluation_saved", **report["summary"]}), flush=True)


if __name__ == "__main__":
    main()

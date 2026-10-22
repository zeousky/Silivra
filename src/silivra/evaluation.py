"""Small, explicit assistant evaluations. Tool proposals are never executed."""

from collections import Counter
import hashlib
import json
from pathlib import Path

from silivra.agent import POLICY, parse_action


def evaluate_assistant(backend, suite: Path) -> dict:
    suite = Path(suite)
    with suite.open("rb") as stream:
        snapshot = stream.read(2 * 1024 * 1024 + 1)
    if len(snapshot) > 2 * 1024 * 1024:
        raise ValueError("Evaluation suite exceeds 2 MiB.")
    suite_hash = hashlib.sha256(snapshot).hexdigest()
    cases = [json.loads(line) for line in snapshot.decode("utf-8").splitlines() if line.strip()]
    if not 1 <= len(cases) <= 1000:
        raise ValueError("Evaluation suite needs 1–1000 cases.")
    ids, results = set(), []
    for case in cases:
        if (not isinstance(case, dict) or set(case) != {"id", "category", "prompt", "tools", "expected"}
                or not all(isinstance(case[k], str) and case[k].strip() for k in ("id", "category", "prompt"))
                or not isinstance(case["tools"], list) or not isinstance(case["expected"], dict)):
            raise ValueError("Invalid evaluation case fields.")
        if case["id"] in ids:
            raise ValueError("Duplicate evaluation ID.")
        ids.add(case["id"])
        names = set()
        for tool in case["tools"]:
            if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
                raise ValueError("Evaluation tool must contain a function schema.")
            function = tool["function"]
            if (not isinstance(function.get("name"), str) or function["name"] in names
                    or not isinstance(function.get("parameters"), dict)
                    or function["parameters"].get("type") != "object"
                    or not isinstance(function["parameters"].get("properties"), dict)):
                raise ValueError("Evaluation tool needs a unique name and object parameters.")
            parse_action(json.dumps({"tool": function["name"], "arguments": {}}))
            names.add(function["name"])
        # Expected actions themselves follow the same runtime protocol.
        parse_action(json.dumps(case["expected"]))
        if "tool" in case["expected"] and case["expected"]["tool"] not in names:
            raise ValueError("Expected tool action is not present in the case schemas.")
    for case in cases:
        outcome = {"id": case["id"], "category": case["category"], "passed": False, "valid_action": False}
        try:
            raw = backend.generate([{"role": "system", "content": POLICY},
                                    {"role": "user", "content": case["prompt"]}], case["tools"])
            outcome["raw"] = raw[:16_000]
            action = parse_action(raw)
            outcome["valid_action"] = True
            outcome["passed"] = action == case["expected"]
        except (ValueError, RuntimeError) as error:
            outcome["error"] = str(error)
        results.append(outcome)
    categories = Counter(item["category"] for item in results)
    return {"format_version": 1, "suite_sha256": suite_hash, "cases": len(results),
            "system_policy_sha256": hashlib.sha256(POLICY.encode()).hexdigest(),
            "prompt_format": "silivra-role-json-v1",
            "valid_actions": sum(item["valid_action"] for item in results),
            "passed": sum(item["passed"] for item in results),
            "categories": {category: {"cases": count, "passed": sum(item["passed"] for item in results if item["category"] == category)}
                           for category, count in sorted(categories.items())},
            "grading": "exact parsed JSON action; original small diagnostic suite, not a general intelligence benchmark",
            "tools_executed": False, "results": results}

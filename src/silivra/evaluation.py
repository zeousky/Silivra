"""Small, explicit assistant evaluations. Tool proposals are never executed."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata

from silivra.agent import POLICY, parse_action
from silivra.instructions import validate_record


def _normalized_final(text: str) -> str:
    """Whole-answer matching only; never infer meaning from keyword presence."""
    return " ".join(unicodedata.normalize("NFC", text).casefold().split())


def grade_action(action: dict, case: dict) -> bool:
    """Keep exact grading by default, with explicit finite prose alternatives.

    This is not a semantic judge. A case may opt into case/whitespace-insensitive
    whole-answer alternatives for short conversational prose; code, tool calls,
    and literal instruction tasks continue to use exact parsed actions.
    """
    if "accepted_final" not in case:
        return action == case["expected"]
    if set(action) != {"final"}:
        return False
    candidates = [case["expected"]["final"], *case["accepted_final"]]
    return _normalized_final(action["final"]) in {_normalized_final(text) for text in candidates}


def case_messages(case: dict) -> list[dict]:
    """Prepend fixed policy to a prompt or explicit untrusted turn history."""
    messages = case.get("messages", [{"role": "user", "content": case.get("prompt")}])
    return [{"role": "system", "content": POLICY}, *messages]


def validate_cases(cases: list[dict]) -> None:
    """Validate every fixture before a backend can perform any inference."""
    if not 1 <= len(cases) <= 1000:
        raise ValueError("Evaluation suite needs 1–1000 cases.")
    ids = set()
    for case in cases:
        common = {"id", "category", "tools", "expected"}
        if (not isinstance(case, dict) or set(case) - {"accepted_final"} not in (common | {"prompt"}, common | {"messages"})
                or not all(isinstance(case[k], str) and case[k].strip() for k in ("id", "category"))
                or not isinstance(case["tools"], list) or not isinstance(case["expected"], dict)):
            raise ValueError("Invalid evaluation case fields.")
        if case["id"] in ids:
            raise ValueError("Duplicate evaluation ID.")
        ids.add(case["id"])
        if "prompt" in case and (not isinstance(case["prompt"], str) or not case["prompt"].strip()):
            raise ValueError("Evaluation prompt must be nonempty text.")
        if "messages" in case:
            if (not isinstance(case["messages"], list) or not case["messages"]
                    or any(not isinstance(message, dict) or message.get("role") == "system"
                           for message in case["messages"])):
                raise ValueError("Evaluation histories need messages without system overrides.")
        validate_record({"id": case["id"],
                         "source": {"name": "Evaluation fixture", "license": "MIT", "url": "local:evaluation"},
                         "messages": [*case_messages(case),
                                      {"role": "assistant", "content": json.dumps(case["expected"])}]})
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
        parse_action(json.dumps(case["expected"]))
        if "accepted_final" in case:
            alternatives = case["accepted_final"]
            if (set(case["expected"]) != {"final"} or not isinstance(alternatives, list)
                    or not 1 <= len(alternatives) <= 32
                    or any(not isinstance(text, str) or not text.strip() or len(text) > 64_000
                           for text in alternatives)):
                raise ValueError("accepted_final needs 1–32 nonempty final-answer alternatives.")
            for text in alternatives:
                parse_action(json.dumps({"final": text}))
        if "tool" in case["expected"] and case["expected"]["tool"] not in names:
            raise ValueError("Expected tool action is not present in the case schemas.")


def evaluate_assistant(backend, suite: Path) -> dict:
    suite = Path(suite)
    with suite.open("rb") as stream:
        snapshot = stream.read(2 * 1024 * 1024 + 1)
    if len(snapshot) > 2 * 1024 * 1024:
        raise ValueError("Evaluation suite exceeds 2 MiB.")
    suite_hash = hashlib.sha256(snapshot).hexdigest()
    cases = [json.loads(line) for line in snapshot.decode("utf-8").splitlines() if line.strip()]
    validate_cases(cases)
    results = []
    for case in cases:
        outcome = {"id": case["id"], "category": case["category"], "passed": False, "valid_action": False}
        try:
            raw = backend.generate(case_messages(case), case["tools"])
            # Preserve the full bounded action for independent review.
            outcome["raw"] = raw[:64_000]
            if len(raw) > 64_000:
                outcome["raw_truncated"] = True
            action = parse_action(raw)
            outcome["valid_action"] = True
            outcome["passed"] = grade_action(action, case)
        except (ValueError, RuntimeError) as error:
            outcome["error"] = str(error)
        results.append(outcome)
    categories = Counter(item["category"] for item in results)
    return {"format_version": 1, "suite_sha256": suite_hash, "cases": len(results),
            "system_policy_sha256": hashlib.sha256(POLICY.encode()).hexdigest(),
            "prompt_format": "silivra-role-json-v1",
            "valid_actions": sum(item["valid_action"] for item in results),
            "passed": sum(item["passed"] for item in results),
            "categories": {category: {"cases": count,
                                       "valid_actions": sum(item["valid_action"] for item in results if item["category"] == category),
                                       "passed": sum(item["passed"] for item in results if item["category"] == category)}
                           for category, count in sorted(categories.items())},
            "grading": ("exact parsed JSON action; explicit accepted_final cases use finite whole-answer "
                        "alternatives with Unicode case/whitespace normalization; "
                        "original small diagnostic suite, not a general intelligence benchmark"),
            "tools_executed": False, "results": results}

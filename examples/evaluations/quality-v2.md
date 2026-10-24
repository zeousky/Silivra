# Quality diagnostic v2 provenance

This original 24-case diagnostic was authored by OpenAI Codex on 2026-10-22
for Silivra and released under the repository's MIT license. The prompts,
synthetic source passages, memory records, tool observations, expected actions,
and finite accepted conversation answers are AI-authored evaluation material.
No external corpus, hosted teacher, private chat, or real user memory was used.
All names and operational details in the cases are synthetic.

## Frozen artifact

- File: `quality-v2.jsonl`
- SHA-256: `ef817d0dc7a0611ae9ea25443b78da619b24e7ba38c54b0ab77dd3dbbc186428`
- Cases: 24, with three in each of eight categories.
- Freeze: prompts, expected actions, and conversation alternatives were fixed
  before any model output from this suite was observed.

Authorship used README.md, docs/ARCHITECTURE.md, docs/ROADMAP.md, and the
src/silivra/evaluation.py fixture interface. The author did not inspect training
or curriculum scripts, dataset directories, checkpoint directories, the v1 suite,
or previous evaluation outputs. No inference, training, network request, or
external model call was performed during authorship or fixture validation.
Repository documentation provided project boundaries; it was not copied into
question/answer examples.

Treat the entire JSONL, including expected answers and accepted alternatives, as
held out. Exclude its prompts from every training split and tokenizer fit.
Record this immutable file hash in experimental manifests before preparing data.
This authorship record does not itself certify an independent dataset-wide
contamination audit. Any replacement must use a new suite version and hash;
do not change accepted answers after observing model results.

## Coverage and scoring

| Category | Three cases cover |
| --- | --- |
| conversation | Thanks, pausing for sleep, and frustration with an error |
| instruction | Indexed extraction, reversed word order, and exact text editing |
| reasoning | Clip inventory, arrival ordering, and changing seat occupancy |
| grounding | A supplied date, a chronological log, and quoted source instructions |
| python | Negative indexing, generator filtering, and a dictionary default expression |
| tools | A read proposal, a literal search proposal, and a supplied observation |
| memory | A supplied preference, a correction, and project scope selection |
| honesty | Unrun tests, an absent fact, and an unsent announcement |

Each completion must be exactly one valid JSON action under the runtime's action
protocol. Every non-conversation case uses exact equality of the parsed action.
The prompts constrain final text or Python syntax so expected answers are
unambiguous; other code spellings do not receive partial credit. The three
conversation cases have finite `accepted_final` lists fixed at authorship time.
Their whole-answer comparisons use the evaluator's Unicode case and whitespace
normalization. These lists are not a semantic judge: a sensible conversational
answer outside the enumerated list fails. Report conversation scoring separately.

Tool cases propose actions or interpret supplied text; evaluation never executes
tools. The observation is supplied in the user request, not collected from a live
tool. Memory cases use supplied synthetic records, not retrieval from a memory
database. Grounding cases test supplied passages, not live research. The small
Python cases do not establish repository editing or sandboxed coding ability.
This suite is a narrow diagnostic, not a general intelligence benchmark.

## Offline fixture validation

The current evaluator's `validate_cases` accepted all 24 records. All expected
actions and every predefined conversation alternative passed `grade_action`.
The suite also has 24 distinct IDs and 24 distinct normalized user prompts.
These checks establish fixture consistency; they measure no learned ability.

```sh
uv run --offline python - <<'PYCODE'
import hashlib
import json
from pathlib import Path
from silivra.evaluation import grade_action, validate_cases

path = Path("examples/evaluations/quality-v2.jsonl")
snapshot = path.read_bytes()
cases = [json.loads(line) for line in snapshot.decode().splitlines()]
validate_cases(cases)
assert all(grade_action(case["expected"], case) for case in cases)
for case in cases:
    for text in case.get("accepted_final", []):
        assert grade_action({"final": text}, case)
assert hashlib.sha256(snapshot).hexdigest() == (
    "ef817d0dc7a0611ae9ea25443b78da619b24e7ba38c54b0ab77dd3dbbc186428"
)
print("Validated 24 fixed held-out cases; no inference performed.")
PYCODE
```

# Pretrained backend practical diagnostic v2

This is a fresh 12-case corroborating diagnostic authored on 2026-10-24 after the
initial v1 measurements exposed adapter failures. The original, AI-authored,
MIT-licensed synthetic prompts contain new task content: positional text selection,
parenthesized arithmetic, elapsed time, a short Python counting function, TOML and
CSV interpretation, search followed by file reading, recursive path filtering,
unavailable evidence, conversation recall, corrected memory, and project-scoped
memory with a conflicting decoy.

The suite is frozen before its first model run. It is not a training dataset and
must stay out of weight updates, tokenizer fitting and adapter examples. Its
answers must not be inserted into system prompts, decoder constraints or runtime
heuristics. The v1 reports informed general adapter work before this suite was
created; improvements on v1 alone therefore are development evidence. Report v2
separately as a small corroborating check, with all failures retained. A repeated
run after further outcome-driven changes is development evidence too.

These tasks are independent of Silivra's local training records. Common skills,
question forms or code patterns may appear in upstream Qwen pretraining, whose
full corpus is unavailable for contamination auditing. Twelve synthetic tasks do
not establish broad competence, safe autonomy, or general software engineering
ability. This suite does not compare an original Silivra checkpoint against Qwen
on the same tasks and does not show that original-model weights improved.

Use the same bounded evaluator and an already provisioned local model:

```sh
uv run python scripts/measure-pretrained.py \
  --run runs/silivra-qwen3-8b \
  --suite examples/evaluations/pretrained-v2.jsonl \
  --output docs/results/pretrained-8b-fresh.json
```

No model needs to run while preparing or validating the fixture schema. The live
measurement uses actual read-only ToolRegistry operations, temporary MemoryStore
state and the strict action protocol. Network, writes, command execution and
unrestricted Python execution stay disabled. Python is checked on five fixed
inputs by the evaluator's operation-bounded AST interpreter, using the explicitly
requested subset. The test covers small function behavior rather than arbitrary
Python compatibility or repository modifications. See the
[v1 evaluator boundaries](pretrained-v1.md) for budgets and report semantics.

A correct answer without each required observed tool execution fails. The search
case specifically requires successful search and read observations. The missing
source case measures honest unavailability after a real failed tool call; it does
not independently establish whether the failure came from file absence or another
file-access error. Memory entries are saved/corrected by explicit synthetic user
operations through the real store, and the model answers from retrieved context.
They do not measure autonomous memory writing or model training.

The public report contains only synthetic prompts, raw generations, objective
content grades, tool observations, timings and pinned model provenance. Private
paths, conversation identifiers and durable user state are excluded. Keep the
fixture unchanged after observing results, and publish limitations alongside scores.

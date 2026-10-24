# Pretrained backend practical diagnostic v1

These 18 original, AI-authored synthetic tasks are MIT-licensed project evaluation
fixtures. They were written for this diagnostic and are not included in Silivra
training or tokenizer fitting. They do not establish that an upstream pretrained
model has never encountered similar tasks; its training corpus is not available
for a complete contamination audit. This is a small diagnostic rather than a
standardized general intelligence, safety, or coding benchmark.

The frozen cases separately measure literal instructions, arithmetic and unit
reasoning, one stable factual answer, two short Python functions, actual file
read/list/search interpretation, an instruction embedded in untrusted file data,
missing-file uncertainty, conversation recall, and explicit durable memory recall,
correction, forgetting, and isolation across both users and projects.

Memory fixtures call the real MemoryStore API to simulate explicit user-directed
remember/correction/forget operations. The model receives only the runtime's scoped
retrieved data. These cases measure answering from that context; they do not imply
that the model independently writes or trains on memory. Conversation recall uses
the same session across actual model turns. Every other case starts with a new
temporary workspace and state database. The workspace contains only public
synthetic files. State and transcripts are deleted after each case.

Run an already provisioned, hash-verified local pretrained backend:

```sh
uv run python scripts/measure-pretrained.py \
  --run runs/silivra-qwen3-8b \
  --output docs/results/pretrained-8b.json
```

The script performs no training or downloads. It disables network, writes and
command tools, caps each turn at five model actions and 180 seconds, and reserves
768 output tokens by default. The timeout is cooperative between inference tokens
and operations rather than an operating-system hard deadline. `--max-cases N`
limits a smoke measurement; omitted cases are excluded from the denominator.
Evaluation output must be new so a prior result cannot be silently overwritten.

Every task requires a validated final JSON action and an objective final-content
check. Correct-looking tool proposals, protocol validity alone, and right answers
without the required observed tool execution do not pass. File tool cases use the
real ToolRegistry against the temporary workspace, including actual missing-file
errors. Exact-answer grading can reject semantically reasonable answers that do
not obey the requested format; the report retains raw output for inspection.

Python answers are interpreted by a small operation-bounded AST checker. It never
uses Python `eval`, `exec`, compilation, a shell, imports, or unrestricted child
processes. Supported statements are simple assignments, bounded local augmented
assignments, if, for, return and
expression statements; expressions include bounded literals/arithmetic, comparisons,
local values, lists/tuples, indexing, list.append and a short allowlist of pure
builtins. There are no comprehensions, while loops, recursion, arbitrary attribute
access or imports. The prompts specify that subset. Valid alternative Python
outside this subset is a checker limitation and is retained as a failed case
rather than run unsafely. Each function is checked on multiple fixed edge cases.
This measures narrow function behavior, not repository modification or general
software engineering competence.

The JSON report publishes public prompts, final grading reasons, every raw model
action, tool observations, per-call backend timing/memory metrics, and the verified
model manifest with immutable source revisions and model/tokenizer file hashes.
Case latency includes memory and tool work; model loading/hash verification is
reported separately. Generation throughput includes prefill and decode, while
decode throughput is separately labelled. The report contains no private chats,
user state, session identifiers, temporary paths or workstation paths. It reports
failures without rewriting prompts or modifying weights to improve this score.

# Pretrained backend practical diagnostic v3

These 15 original, AI-authored synthetic MIT evaluation fixtures were created on
2026-10-25 before their first model run, to compare the local Qwen3-8B and
Qwen3.5-9B backends. They are separate from training, tokenizer fitting, prompt
examples and decoder constraints. Their answers must not be copied into the
adapter. The frozen JSONL has SHA-256:

```text
9d2a6ebeece5331c8d7bc245981127d4043eba0900e1839fd90686027149e2cf
```

The earlier v1, v2 and calculator results informed adapter work. Those repeated
suites are development/regression evidence; v3 initially supplies new task content.
Once its outcomes inform changes, subsequent v3 measurements also become
regression evidence. Upstream pretrained corpora cannot be fully audited for
similar questions. This small synthetic check is not a standardized benchmark,
a statistical demonstration of general intelligence, or a software-engineering
acceptance test.

The cases cover exact filtering and case-sensitive text replacement, dependency
ordering, duration conversion, two short Python functions, actual read/search/list
tool interpretation, arithmetic from two files with a real calculator call,
role-like text embedded in an untrusted file, absence of a requested field in an
accessible report, conversation corrections, scoped saved preferences and a
current-task choice distinct from saved memory. New prompts and fixture contents
differ from v1/v2; some underlying skills deliberately overlap so regressions
remain visible. Stable-world factual recall is covered by v1 rather than by this
suite; v3's grounding questions use explicit synthetic evidence.

Use the same bounded evaluator, source revision and output budget for each model:

```sh
uv run python scripts/measure-pretrained.py \
  --run runs/silivra-qwen3-8b \
  --suite examples/evaluations/pretrained-v3.jsonl \
  --max-new-tokens 384 \
  --output docs/results/pretrained-8b-v3.json

uv run python scripts/measure-pretrained.py \
  --run runs/silivra-qwen3.5-9b \
  --suite examples/evaluations/pretrained-v3.jsonl \
  --max-new-tokens 384 \
  --output docs/results/pretrained-9b-v3.json
```

Both models must already be provisioned and verified; evaluation downloads nothing
and updates no weights. Each case receives a fresh temporary workspace and scoped
state. Conversation cases reuse their own session, while memory fixtures use real
explicit MemoryStore operations. No model writes durable facts. Network, file
writes, commands and arbitrary Python execution are disabled. Turns retain the
evaluator's five-action and cooperative 180-second limits. The 384-token output
budget is intentionally modest; output-budget failures count as failures and
remain in the report. Model load/verification time is separate from case latency.

Python answers are checked on multiple edge inputs by the existing bounded AST
interpreter, using the subset requested in the prompt. Valid alternative Python
outside that subset can fail and must be reported as a checker limit. JSON answers
are compared by their parsed values, so object-key order does not matter. Exact
text checks reject extra formatting or punctuation even when the meaning is right.
The missing-field case requires a successful read and an explicit unavailable
answer; it differs from a failed-file-access case. Tool-dependent cases require
both final-content correctness and successful observations of the declared tool
names. The current checker does not enforce an exact number of read calls or their
arguments; inspect the raw observations before claiming that every requested
source was read. The role-spoof case tests one synthetic untrusted-file instruction
and cannot establish comprehensive prompt-injection resistance.

Before choosing a new default, compare fresh-suite outcomes by category alongside
v1/v2/calculator regressions and actual runtime checks. Compare first-token latency,
case time, decode throughput, cold load time and memory under the same hardware
conditions. Hybrid Qwen3.5 layers may safely require cold prefill; absence of prefix
cache reuse is a documented performance property rather than a capability failure.
A larger parameter count or valid JSON alone does not justify promotion. Retain
the existing default when the practical evidence is tied or mixed. Publish each
suite separately with its failures, rather than hiding regressions inside a pooled
score or treating these few tasks as a statistically significant ranking.

Preparation validation found no exact case-folded, whitespace-normalized match
between the 18 new user prompts and strings in the nine other public JSONL
instruction/evaluation files under `examples/`. This is an exact local overlap
check, not a semantic or upstream contamination audit. Validation also loads the
fixture schema, exercises the real temporary read/search/list/calculator operations
with scripted reference actions, and checks both reference functions through the
existing AST grader. Those checks establish
fixture/evaluator consistency; only the separate native model runs measure model
behavior. Public reports contain synthetic inputs, observations and raw actions;
private chats, persistent user state and machine paths must remain excluded.

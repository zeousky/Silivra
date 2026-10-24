# Practical assistant diagnostic v1

`quality-v1.jsonl` is an original, Codex-authored evaluation fixture, created on
2026-10-22 and released under this repository's MIT license. It contains no
copied benchmark questions or private conversation records. Its fictional names,
values, snippets and tool observations were authored for this diagnostic. The
questions were written independently of the new assistant curriculum, without
reading it, and must stay out of training and tokenizer fitting.

Frozen source SHA-256:
`147fbc15e004b3cc42680ece6129068bcfb86c40ef28d9973e1008166dd9b367`.
Any future question or answer changes need a separately named version, so results
from this version remain comparable.

The 40 cases cover eight categories with five cases each: ordinary conversation,
literal instructions, elementary reasoning, source-grounded reading, Python,
tool selection and observations, supplied memory, and honesty about missing
knowledge or actions. They test the assistant interface's JSON protocol as well
as the content of the response. Proposed tools and Python snippets are never
executed. Memory is supplied as untrusted fixture data, so these cases measure
reading and use of that data, rather than retrieval quality or persistence.

Run against a local checkpoint with:

```sh
uv run silivra evaluate-assistant --run runs/CHECKPOINT \
  --suite examples/evaluations/quality-v1.jsonl --max-new-tokens 96 \
  --output runs/CHECKPOINT/quality-v1.json
```

Keep raw answers and the suite hash in every report. Before training, compare
normalized user prompts against **every** training and tokenizer-fitting source;
remove exact overlaps from training or identify the affected evaluation cases as
contaminated. Common prompts such as "Hi!" are especially likely to overlap.
Near-duplicate or templated task overlap is not ruled out by an exact-match check.

Most cases request an exact answer or tool proposal. Four conversational cases
instead list explicit `accepted_final` prose alternatives. On those four cases
only, grading compares the **whole** final response after Unicode normalization,
case folding, and whitespace normalization. Other cases retain exact parsed JSON
action comparison. Punctuation remains meaningful. The accepted alternatives
are finite and auditable; they are not a semantic judge. A useful unlisted
paraphrase can fail, so inspect the retained raw responses before interpreting
a conversation score. Arbitrary answers containing a correct keyword do not pass.

This is a small development diagnostic, not a broad intelligence, coding, safety,
or production-readiness benchmark. It does not test large projects, long context,
live tools, complex mathematical proofs, current knowledge, or actual execution.
Report per-category scores, protocol-validity counts, and regressions separately.
Choosing hyperparameters after observing its results makes it a development set;
fresh held-out tasks are needed for further independent quality claims.

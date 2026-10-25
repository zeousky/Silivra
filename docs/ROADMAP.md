# Silivra roadmap

Silivra is a long-term effort to build an original, useful local AI. Model quality,
tool reliability, memory, training provenance, and repeatable evaluation advance
together. Every milestone needs evidence before it is called complete.

## Practical local assistant with external weights

2026-10-25: the pinned Qwen3.5-9B text-only profile completed verified provisioning
and real Metal inference. A newly frozen 15-task diagnostic scored 14/15 versus
8B's 11/15 under identical adapter/generation budgets. The sole 9B failure was
extra punctuation; coding, observed file/calculator use, scoped memory and one
untrusted-file task passed. Initial median decoding was 20.58 tokens/s, median
first-token latency 2.99 seconds and peak MLX allocation 5.255 GiB. These small
diagnostics support a practical default choice, not general-intelligence claims.
See the [comparison and retained failures](PRETRAINED_COMPARISON.md).
The final hybrid input-prefix snapshot cache retains 14/15 while reducing that
suite's wall time from 117.74 to 82.83 seconds and median first-token latency to
1.89 seconds. Decoding is approximately 21 tokens/s. All 20 live runtime checks
pass for both 8B and 9B; 9B is the new setup/discovery default, with explicit saved
choices preserved. Repeated v1/v2/calculator results are 16/18, 9/12 and 4/4.
These are inference/runtime improvements, not weight training. Unaided arithmetic
still fails, and total OS memory use can exceed the measured 5.311 GiB MLX peak.
The final source passes 328 Mac tests and 281 actual portable Linux tests. Default
interactive chat was verified with a coherent Python explanation and a real
calculator call; no model server or background process is required.

2026-10-24 save point: Qwen3-8B local inference now includes constrained JSON
actions, a bounded exact-arithmetic tool, explicit output-budget stops, and a
bounded exact input-prefix KV cache. Live diagnostics scored 15/18 development,
8/12 separate tasks and 4/4 calculator-assisted tasks; all 18 runtime checks passed.
Decode throughput was approximately 24 tokens/s on the M4 16 GB target. The source
passes 308 Mac tests; the recorded 249-test Linux snapshot predates the latest
profile extension. These results are practical inference evidence, not original
Silivra training or proof of reliable general intelligence.

At that save point the separately pinned Qwen3.5-9B trial was unfinished and
Qwen3-8B remained the default. The later 2026-10-25 measurements above complete
its initial native trial. See [results and limits](PRETRAINED_ASSISTANT.md).

On 2026-10-23, an explicitly separate Qwen3-8B 4-bit MLX backend was implemented
to provide an 8B-class local assistant on the 16 GB target. Explicit bounded setup
pins and verifies model/tokenizer/license artifacts; ordinary chat stays offline.
The CLI identifies the pretrained backend, preserves original checkpoints and
retains strict JSON actions, permissions, scoped memory, exact context admission,
streaming and cancellation. Runtime context is limited to 8,192 tokens.

This is pretrained inference integration, not original-model training or an 8B
Silivra weight release. Original-model milestones below remain open. Native task
and speed measurements for this separate backend are recorded in
[PRETRAINED_ASSISTANT.md](PRETRAINED_ASSISTANT.md); mock tests and live evidence
are reported separately. Expanding parameter count alone is not an acceptance test.

## Foundation implemented

- Original MLX transformer/tokenizer with bounded, resumable pretraining and verified
  checkpoints. Initial 14M story checkpoint and measured M4 results remain local.
- JSON assistant action runtime with explicit permissions and resource budgets.
- Workspace file inspection/search, guarded atomic edits, optional trusted commands,
  public page fetching and configurable web/encyclopedia search.
- Scoped private SQLite memory, ranked retrieval, provenance, revisions, transcripts,
  deletion, export and snapshot backup.
- Assistant-only instruction tuning, attributed JSONL validation, exact split-overlap
  checks, immutable base lineage and separate resumable instruction checkpoints.
- Native diagnostic evaluation, portable runtime tests, project documentation and CI.
- Validated conversation replay in training role format, with atomic complete-turn
  trimming. Measured follow-up prompt shrank from 326 to 212 original-tokenizer
  tokens; past actions stay inert and cannot enable tools or introduce policy.
- Interactive terminal with lazy checkpoint selection, live provisional answers,
  cooperative cancellation, scoped sessions/preferences/input history, explicit
  tools and per-process permissions. Interface behavior is tested independently
  from native model task accuracy.

These are working engineering components. They do not establish a capable learned
assistant. See [CAPABILITY_RESULTS.md](CAPABILITY_RESULTS.md) for measured boundaries.

## 1. Broaden base-language and code competence

Curate an explicitly licensed corpus of English explanations, mathematics and code.
Record immutable revisions, licensing, token budget, deduplication and held-out split
construction. Keep test prompts and benchmark solutions out of every training phase.
Compare 15M/40M/100M designs under equal token and 16 GB memory budgets, retaining
language regressions and hardware measurements. Do not increase size by assumption.

Acceptance: reproducible manifests, demonstrably better held-out language/code loss,
separate contamination checks and a recorded resource budget.

## 2. Train reliable assistant and tool behavior

Progress measured on 2026-10-22: the separate assistant v2 run resumed from step
500 and completed 8,000 updates with its original tokenizer/data/configuration.
Instruction validation loss fell from 3.73719 to 2.98620; synthetic exact accuracy
rose from 3/48 to 5/48. Independent accuracy stayed 0/15, and story loss worsened.
The CLI is implemented, 187 local tests and Linux portable CI pass, and native
checkpoint loading/streaming was demonstrated with a nonsensical greeting.
These results do not satisfy this milestone's reliable-assistant acceptance.
See [the complete results and failures](ASSISTANT_V2_RESULTS.md).

The 2026-10-22/23 [quality experiments](QUALITY_RESULTS.md) add frozen practical
diagnostics, deterministic focused/rehearsal datasets and cross-corpus retention
checks. Focused v3 completed 2,000 updates, with step 1,200 selected by validation
loss. Synthetic task accuracy rose 14/216 to 84/216 and native greetings became
relevant, but practical accuracy fell 3/40 to 1/40 and older losses regressed
severely. V3 remains experimental rather than replacing v2. Rehearsal v4 also
completed 2,000 updates after resuming its saved step-400 state; selected step 1,200 scored 2/40 practical tasks and 0/24 on a fresh
diagnostic, with older losses exceeding the 5% regression limits. A validation-
selected weight average also failed native tasks. Neither was promoted; v2
remains the original-model default. All 214 local tests passed at that milestone. Reliable general/coding assistant
acceptance remains unmet.

Expand reviewed instruction data beyond the included protocol examples. Include
grounded coding edits, tool-result interpretation, malformed/missing information,
permission denial, prompt injection, uncertainty and source attribution. Train longer
context with the same tokenizer where useful; evaluate its actual usable length.

Acceptance: fixed held-out instruction/tool tasks, executed coding tasks in a real
external sandbox, regression comparisons against the base, and published failures.
Do not enable unrestricted host commands by default to improve a success score.

## 3. Improve memory quality

Add explicit import workflows, conflict reconciliation and user-approved summaries.
Measure cross-session recall, correction, deletion, stale facts and adversarial
content. Introduce hybrid semantic retrieval only if measured recall improves enough
to justify model/storage cost. Retrieval must retain attribution and scope isolation.

Acceptance: a versioned memory evaluation set, recall/precision/forgetting results,
safe migrations and restore drills. No automatic learning from private chats.

## 4. Research and coding workflows

Build multi-source research with citation checks, repository indexing, patch review,
test feedback and a real isolated execution environment. Add an interface only after
the same core APIs can enforce policy across it and the CLI. Any hosted teacher or
alternative inference provider must be explicit, optional and separately identified.

Acceptance: reproducible end-to-end tasks, independent sources, transparent traces,
budget handling, real sandbox boundaries and no hidden capability substitution.

## 5. Versioned releases

Publish code releases with changelogs, supported-environment checks and evaluations.
Publish weights/tokenizers separately only with license review and full provenance.
Review dependency updates, maintain portability tests, document breaking changes,
and preserve migration/rollback procedures as private state evolves.

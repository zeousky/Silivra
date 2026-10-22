# Silivra roadmap

Silivra is a long-term effort to build an original, useful local AI. Model quality,
tool reliability, memory, training provenance, and repeatable evaluation advance
together. Every milestone needs evidence before it is called complete.

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

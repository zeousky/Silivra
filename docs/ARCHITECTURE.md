# Silivra architecture

Silivra is one AI project with two connected layers: an original language model
and an assistant system that can give that model access to tools and memory.
Improving the system and training better weights are separate engineering tasks.

The optional practical assistant uses an explicitly identified external model:
Qwen3-8B (8,190,735,360 parameters), quantized to 4 bits by MLX Community.
It does not alter the original Silivra model/tokenizer lineage. The native training,
checkpoint, loss evaluation and export code remain specific to that original lineage.
External inference results must be attributed separately.

## External local assistant backend

`provision.py` implements the explicit `setup-assistant` download. A fixed allowlist
pins the upstream model/tokenizer revision, original model revision, license, file
sizes and SHA-256 digests. Setup bounds total bytes and session time, resumes partial
transfers, preserves unrelated existing files, and writes the selection marker only
after all artifacts verify. Model files stay under ignored `runs/`.

`pretrained.py` implements the same `generate`/`context_usage` backend contract using
MLX-LM. It requires local verified files, disables remote code, and never falls back
to downloading or hosted inference. The original `MLXBackend` remains selectable.
The Qwen chat template is used with thinking disabled and an 8,192-token runtime
limit. Tool schemas remain inside Silivra's strict JSON protocol; tool observations
are quoted as untrusted data rather than translated into permission grants.
Accounting and generation use the same chat-template tokens and reserve the complete
output budget. Streaming retains cancellation, strict final parsing and capability
checks. An 8 GiB MLX memory guideline is advisory; this is not an OS memory cap.

`structured.py` masks token scores through an Outlines-core ByteLevel automaton.
Each generation gets fresh state while immutable vocabularies/compiled grammars
are cached. The grammar permits exactly one string-final or an enabled tool action,
with required fields, declared keys and argument types. Complete actions admit only
EOS, including MLX-LM's one-token lookahead. Numeric ranges, string lengths/patterns,
array counts, paths and permissions remain explicit runtime checks; deferred schema
constraints are recorded rather than silently claimed as grammar guarantees.
The full result still passes the original strict parser. A token-limited incomplete
action stops as `output_budget`; its draft is rejected and no partial tool runs.

`prompt_cache.py` owns one backend-local exact input prefix, at most 1,024 tokens
and 256 MiB of actual KV allocations. It verifies all layer offsets and actual trim
results, accounting for MLX-LM's decoding lookahead. Preparation consumes stored
state before inference; only a validated complete action can commit a new input
prefix. Cancelled, malformed and output-limited generations cannot commit. Generated
answers and lookahead are trimmed away, and at least one prompt token is always
recomputed. Admission counts the full request, policy and evidence even on a cache
hit. Changed tool/user content reuses only exactly matching tokens; no KV state is
persisted or treated as durable memory.

`arithmetic.py` provides a stateless `calculate` tool. It interprets numeric AST
nodes with exact fractions and bounded decimal rendering; no Python code is
compiled or executed. Only literals, parentheses, unary signs and `+ - * / // %`
are supported. Expression length, AST size, nesting, operation count and rational
size/magnitude are bounded. Fifty-digit decimal results identify rounding explicitly.
The tool grants no filesystem, network or command capability. It is selected by
default for pretrained one-shot chat and an empty-selection assistant setup;
explicit `--no-tools` and saved interactive selections retain their meaning.

The terminal identifies the external model. A successful setup selects it in the
current user/project's private preferences; absent an explicit selection, a complete
installed 8B run is preferred over the original research checkpoint. No model is
downloaded on ordinary chat startup. See [provenance and measurements](PRETRAINED_ASSISTANT.md).

## Model and learning

`model.py` implements the decoder in MLX; `data.py` prepares reproducible text
and an original byte-level tokenizer. `train.py` owns pretraining.
`checkpoint.py` preserves weights, optimizer, sampler, RNG and provenance.
The initial 14M checkpoint remains a story model; its original measured results
are in [INITIAL_RESULTS.md](INITIAL_RESULTS.md).

`instructions.py` validates source-attributed JSONL conversations, checks exact
conversation/user-prompt split overlap, and creates assistant-only loss masks.
User/system/tool text is JSON-escaped beneath role headers. `finetune.py` starts
from a verified local checkpoint and writes a separate resumable instruction run.
It does not download a pretrained model or change the tokenizer.

Codex can develop the system and author reviewed training examples. Codex editing
source code, adding memory, or writing prompts does not update model weights.
The bundled examples teach the protocol and exercise the pipeline; a broad,
licensed coding and instruction corpus is still required for a useful assistant.

## Assistant execution

`agent.py` accepts a model backend, tool registry and scoped memory store.
Each step produces exactly one JSON action:

```json
{"tool":"read_file","arguments":{"path":"pyproject.toml"}}
```

or:

```json
{"final":"The project uses Python."}
```

The runtime rejects prose, duplicate keys, unknown fields and unavailable tools.
It bounds steps, checks a wall-clock admission budget, and prevents an identical
tool action from executing twice in one turn. Tools enforce their own time and
size limits. The overall turn budget is checked between blocking calls, so it is
not a hard interruption deadline for generation or an in-progress tool.

The native MLX backend uses the same formatter as training. Before generation it
checks that the complete prompt and requested output fit the checkpoint context;
it reports an error instead of dropping policy or the latest request. A 512-token
checkpoint needs focused tool selections and short conversations. Longer context
requires training and evaluation; changing a configuration alone is insufficient.

Completed conversation turns retain the user/assistant/tool roles used in training.
Replay validates each assistant action and its matching observation, never adds a
system role, and never executes a historical tool request. Interrupted, malformed,
and partial turns at the edge of a transcript window are omitted. Context trimming
removes whole older turns, preserving the relationship between tool actions and
their results. Tool observations remain escaped, untrusted data; historical actions
cannot change the current permission grants or available tools.

Malformed outputs stop the turn visibly. There is no hidden hosted-model fallback,
automatic JSON-to-shell conversion, or heuristic that pretends the base model
has learned tool selection. The direct `tool` CLI works independently of that skill.

## Interactive terminal

`terminal.py` presents the same agent, tools and scoped SQLite store through a
Rich/prompt-toolkit interface. `settings.py` keeps versioned preferences scoped to
the local user and workspace. Model loading is lazy; explicit tools, help, settings
and memory commands work without MLX. Interactive and one-shot CLI operations share
the runtime and permission checks. Permission grants last only for the process.

Native generation supports Unicode-safe text deltas and cooperative cancellation
before prefill and between tokens. Final-answer text stays visibly provisional until
the entire strict JSON action validates. The runtime checks cancellation and turn
budgets before tool effects. It can trim complete older optional turns and lower-ranked
memory to fit actual tokenizer limits, and reports those omissions. It retains the
system policy, current request, tool definitions and current-turn tool evidence;
if those cannot fit, it stops with a context-budget error.

Commands are accepted only from local user input. Model answers, file contents and
observations are rendered as inert text. Input history is bounded, credential-
redacted and scoped separately from conversation transcripts. The terminal guide
describes the supported controls and operating boundaries.

## Tools and external data

`tools.py` exposes typed schemas and validates arguments again at execution.
Pure arithmetic and read/list/literal-search tools are available by default. Writes, command execution,
and network calls each require an explicit startup flag. File operations walk
directory descriptors, reject symlinks and hardlinks, block sensitive paths,
and restrict reads to bounded UTF-8 files inside the workspace. Replacements
require the full current SHA-256; creation refuses existing files.

`run_command` starts an argv array without a shell, strips inherited credentials,
bounds output/time, and cleans up the process group. It runs as the local OS user.
It is **not an OS sandbox**: an enabled command can access files and network beyond
the workspace, including through Python or other interpreters. Use only with a
trusted workspace and model, or place the whole process in an external sandbox.

`web.py` supports broad web search through an optional Brave credential. Without
one it explicitly reports Wikipedia encyclopedia search. Public HTTPS fetching
validates DNS results, pins the connection to a validated address with normal TLS
hostname verification, rechecks redirects, and bounds response bytes and time.
Retrieved pages and snippets carry their source URLs and remain untrusted data.
Search queries and fetched URLs leave the machine only when network is enabled.

## Memory

`memory.py` uses SQLite WAL, full synchronous commits, foreign keys, FTS5 and
transactional revisions. Durable facts and conversation transcripts are separate.
Every application operation supplies a scope derived from the selected workspace
and local user label. This is a trusted local CLI, not a multi-user authentication
service; a future server must bind scope to authenticated identity.

Durable facts have a stable key, content, category, source, confidence, importance,
timestamps and versions. Retrieval combines lexical match order, importance and
recency; it is local lexical retrieval, without embeddings or paid APIs. Retrieved
facts are bounded JSON data, with provenance and uncertainty retained. They cannot
authorize permissions. No model output, webpage or tool result is automatically
promoted to a durable fact or used for training.

Sessions persist redacted user, assistant and tool messages, scoped to their owner.
Explicit memory operations support inspect, correct, forget, revisions, transcript
deletion, export, and SQLite snapshot backup. Forgetting a fact also removes its
revisions and search entry; it does not erase separate transcripts or old backups.
Credentials are rejected in curated facts and obvious credentials are redacted
from transcripts; pattern detection cannot identify every possible secret.

Default state lives in the ignored `.silivra/` directory with private file modes.
Backups contain every scope, whereas JSON exports contain only the current scope.
Neither is published. Use the backup command rather than copying a live main DB.

FTS5 secure deletion is enabled on SQLite 3.46.1 and later. Older runtimes retain
ordinary SQLite secure deletion and full backup integrity checks; an existing
affected FTS index is optimized once when disabling that optional mode. This avoids
the older engine's [false corruption reports in FTS5 secure-delete mode](https://www.sqlite.org/releaselog/3_46_1.html).
Physical erasure from older FTS index pages, snapshots and separate backups is
not guaranteed by logical forgetting.

## Evidence and tests

Runtime tests use scripted backends to verify orchestration, isolation, failure
paths, budgets and permissions. Network tests use mocked transports. These tests
do not demonstrate learned model skill or a search provider's live availability.

`evaluate-assistant` performs native inference on separate diagnostic prompts,
records raw output, and grades exact parsed actions. It never executes proposed
tools. Protocol validity, task success, held-out loss and story loss are distinct
measurements. The small included suite is a diagnostic, not a general benchmark.

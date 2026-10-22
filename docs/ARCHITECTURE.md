# Silivra architecture

Silivra is one AI project with two connected layers: an original language model
and an assistant system that can give that model access to tools and memory.
Improving the system and training better weights are separate engineering tasks.

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
budgets before tool effects. It can trim older optional history and lower-ranked
memory to fit actual tokenizer limits, and reports those omissions. It retains the
system policy, current request, tool definitions and current-turn tool evidence;
if those cannot fit, it stops with a context-budget error.

Commands are accepted only from local user input. Model answers, file contents and
observations are rendered as inert text. Input history is bounded, credential-
redacted and scoped separately from conversation transcripts. The terminal guide
describes the supported controls and operating boundaries.

## Tools and external data

`tools.py` exposes typed schemas and validates arguments again at execution.
Read/list/literal-search tools are enabled by default. Writes, command execution,
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

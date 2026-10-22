# Silivra terminal

Silivra's terminal uses the project's own local checkpoint, assistant runtime,
workspace tools, and private SQLite memory. It does not call ChatGPT, Codex,
OpenCode, or a hosted replacement model. The terminal is useful for inspecting
files, calling explicit tools, managing memory, and testing the checkpoint. Its
interface does not establish that the checkpoint can solve general coding tasks;
see [measured capabilities](CAPABILITY_RESULTS.md).

From the repository, install the locked dependencies and open the terminal:

```sh
uv sync --locked
uv run silivra
```

Select a particular checkpoint and project explicitly with:

```sh
uv run silivra --workspace /path/to/project --model /path/to/Silivra/runs/silivra-assistant-v2
# Equivalent interactive chat entry point with explicit startup overrides:
uv run silivra chat --run runs/silivra-assistant-v2 --max-new-tokens 96
```

The default model is the local `runs/silivra-assistant-v2` run beside the source
installation, with the selected workspace as a fallback. `/model` paths are
relative to the selected workspace; absolute paths also work. Both entry points
honor saved terminal budgets unless an explicit flag overrides that preference
for the current launch. One-shot `chat "..."` uses its documented CLI defaults.

Model weights are local ignored artifacts and are not downloaded by launching the
terminal. Use `/models` to inspect existing runs and `/model PATH` to select one.
The model loads on your first ordinary message, so `/help`, tools, and memory work
without loading MLX. A failed model load leaves the terminal open.

`Enter` sends. `Alt+Enter` or `Ctrl+J` inserts a newline. Pasted multiline text stays
together. `Tab` completes slash commands and tool names. Up/down arrows navigate
private input history. `Ctrl+C` clears the current input or interrupts the current
turn; the existing conversation remains available. `Ctrl+D` on an empty prompt
exits. Interruption does not roll back a tool that already completed.

## Conversations and checkpoints

| Command | Use |
| --- | --- |
| `/help` | Show all commands and input controls. |
| `/new` | Start a fresh conversation, keeping durable facts. |
| `/sessions` | List conversations for the current local user and project. |
| `/resume SESSION_ID` | Resume one of those conversations. |
| `/history [LIMIT]` | Inspect the active conversation; default 40 messages. |
| `/status` | Inspect checkpoint, context, permissions, selections, and budgets. |
| `/models` | List readable local run metadata under the workspace and source installation's `runs` directories. |
| `/model [PATH]` | Inspect the selection, or select a local run/checkpoint. |
| `/clear` | Clear the visible terminal. |
| `/exit` | Exit. |

Quote paths containing spaces, for example `/model "runs/my assistant"`. Switching
models preserves the conversation; use `/new` when you want a fresh context.
Checkpoint metadata in `/models` is informational; model loading verifies hashes.
The reported maximum token context is an architecture limit, not proof of learned
long-context ability. System policy, tools, history, memory and the output
reservation must all fit. Silivra reports overflow instead of silently dropping
policy or the newest request.

For a short-context checkpoint, try:

```text
/new
/tools none
/config history-messages 0
/config memory-chars 0
/config max-new-tokens 64
Say hello.
```

Preferences persist in ignored `.silivra/config.json`, separately for each local
user/project. Permissions reset when you exit. Startup flags can explicitly
override saved preferences. Malformed settings are reported and preserved; repair
or move that private file before saving new preferences. Input history is also
scoped and redacted, bounded to 500 entries and 256 KiB. Private settings, input
history, and existing SQLite database/sidecar files refuse symlinks and hardlinks.

## Tools and permissions

`/tools` prints enabled tools and their JSON schemas. `/tools read_file search_files`
selects the model's tools; `/tools none` disables model tool selection and `/tools all`
exposes all tools permitted by the current process. A selected tool stays unavailable
until its corresponding capability is enabled. Start with focused selections because
schemas consume context.

`/tool NAME JSON` directly executes a typed user request without depending on the
model learning to choose a tool. The same schema checks, filesystem restrictions,
and permission gates apply:

```text
/tool list_files {"path":"src","recursive":true,"limit":30}
/tool read_file {"path":"pyproject.toml"}
/tool search_files {"path":"src","query":"TODO"}
/permissions network on
/tool web_search {"query":"Python asyncio documentation","limit":3}
/permissions network off
```

`/permissions` inspects grants. `/permissions read|write|network|exec on|off` changes
one explicit grant for this process. Read-only workspace tools are enabled initially;
writes, network, and execution are disabled unless supplied by a startup flag or
explicit command. Model answers and tool content cannot invoke slash commands or
grant permissions. Enabled tool observations are shown as they arrive.

Writes require `/permissions write on`. Replacing a file also requires the current
full `expected_sha256` reported by `read_file`; new-file creation refuses existing
paths. Secret/private state and symlink escapes are blocked by file tools.

`/permissions exec on` permits `run_command` with an argv JSON array, never an
interpolated shell string. **It runs as your OS user and is not a sandbox**: a child
process can access host files and network beyond file-tool restrictions. For example,
`/tool run_command {"argv":["python","--version"]}` runs only after that explicit
grant. Use an external operating-system sandbox for untrusted execution.

Network search uses Brave if `BRAVE_SEARCH_API_KEY` is already exported; otherwise
results explicitly use Wikipedia encyclopedia search. `.env` is not auto-loaded.
Public HTTPS fetches enforce address, redirect, time and size restrictions. Network
calls and tool wiring do not train model weights.

## Memory

```text
/memory remember language This project uses Python.
/memory recall project language
/memory remember language This project now uses Rust.
/memory revisions language
/memory list
/memory forget language
/memory export .silivra/my-memory.json
/memory backup .silivra/my-backup.sqlite
/memory delete-session SESSION_ID
```

`remember` takes a key without spaces and the rest of the submission as its exact
text. It is the explicit durable-fact operation; corrections create revisions.
Retrieval is local ranked lexical retrieval, with provenance retained. Facts are
separate from session transcripts; forgetting a fact does not delete older
transcripts, exports, or backups. There is no automatic chat-to-training pipeline.

Exports create a new private JSON file for the current scope. SQLite backups use
the consistent backup API and include **all scopes** in the database. Existing
files are never overwritten. Keep exports and backups private, preferably inside
ignored `.silivra/`. Obvious credentials are rejected in durable facts and redacted
from transcripts and persistent input history; this is not a secret vault.

## Budgets and live output

`/config` inspects preferences. Set one with `/config KEY VALUE`:

| Key | Range |
| --- | --- |
| `max-new-tokens` | 1–4096 output tokens; must fit the checkpoint context. |
| `max-steps` | 1–32 model actions per turn. |
| `max-seconds` | 1–3600 seconds, checked between blocking operations. |
| `history-messages` | 0–100 previous messages. |
| `memory-chars` | 0 disables retrieval; otherwise 256–16000 characters. |

The native backend can display genuinely incremental answer text while generating
the strict JSON action. Such output is labelled **draft**, followed by acceptance
only after the complete JSON action validates. Rejected or interrupted drafts are
labelled incomplete; malformed text never executes as a command. Backends without
streaming display one completed answer rather than an animated replay.

For automation and complete command-line flags, use `uv run silivra --help` and
`uv run silivra chat --help`. Explicit one-shot prompts, piped input, tools, memory,
training, evaluation, export and doctor commands remain available without opening
the interactive terminal. See the [README](../README.md) and
[training guide](TRAINING.md) for those workflows.

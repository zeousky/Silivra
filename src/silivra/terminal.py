"""Interactive terminal for the same local model, tool policy, and memory APIs.

Slash commands are dispatched only from typed user input. Model output never
passes through this dispatcher. MLX is imported lazily when a chat turn starts.
"""

from __future__ import annotations

from dataclasses import replace
import inspect
import json
import os
from pathlib import Path
import re
import shlex
import stat
import tempfile

from prompt_toolkit.history import History

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .agent import Agent, AgentConfig, project_scope
from .memory import MemoryStore, _redact
from .settings import TerminalSettings, load_settings, private_directory, save_settings
from .tools import ToolRegistry


_READ_TOOLS = {"list_files", "read_file", "search_files"}
_ALL_TOOLS = _READ_TOOLS | {"write_file", "run_command", "web_search", "web_fetch"}
_COMMANDS = ("/help", "/new", "/sessions", "/resume", "/history", "/status", "/model",
             "/models", "/tools", "/tool", "/permissions", "/memory", "/config", "/clear", "/exit")
_CONTROL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f-\x9f]")


def default_model_run(workspace: Path) -> Path:
    """Locate this source installation's weights while working in another repo."""
    source = Path(__file__).resolve().parents[2]
    candidates = [source / "runs" / "silivra-assistant-v2", Path(workspace) / "runs" / "silivra-assistant-v2"]
    return next((path for path in candidates if path.is_dir()), candidates[0])


def _safe(value) -> str:
    # Text rendered by a model, webpage, or tool cannot issue terminal controls.
    return _CONTROL.sub("", str(value))


def _json(text: str) -> dict:
    def pairs(items):
        value = {}
        for key, item in items:
            if key in value:
                raise ValueError("Tool arguments must not contain duplicate keys.")
            value[key] = item
        return value
    def reject_constant(_):
        raise ValueError("Tool arguments must contain finite JSON values.")
    value = json.loads(text, object_pairs_hook=pairs, parse_constant=reject_constant)
    if not isinstance(value, dict):
        raise ValueError("Tool arguments must be one JSON object.")
    return value


class _PermittedTools:
    def __init__(self, registry: ToolRegistry, read: bool):
        self.registry, self.read = registry, read

    def schemas(self):
        return [schema for schema in self.registry.schemas()
                if self.read or schema["function"]["name"] not in _READ_TOOLS]

    def execute(self, name, arguments):
        if name in _READ_TOOLS and not self.read:
            return {"ok": False, "error": {"code": "permission_denied", "message": "Workspace reading is disabled."}}
        return self.registry.execute(name, arguments)


def _backend(run, *, which, max_new_tokens):
    from .agent import MLXBackend
    try:
        return MLXBackend(run, which=which, max_new_tokens=max_new_tokens)
    except ImportError as error:
        raise RuntimeError("Local model inference requires Apple silicon, a supported macOS version, and the locked MLX dependencies. Tools and memory remain available.") from error


class TerminalApp:
    """Testable command controller. The caller owns and closes ``store``.

    ``handle_input`` handles exactly one typed submission and returns False only
    for /exit. An injected backend factory has MLXBackend's constructor signature.
    Permission grants are process-local and only explicit flags/commands change
    them. Preferences and transcripts use the same project/user scope.
    """

    def __init__(self, workspace: Path, store: MemoryStore, *, user="local", run=None,
                 which=None, session_id=None, tool_names=(), allow_read=True,
                 allow_write=False, allow_network=False, allow_exec=False,
                 max_new_tokens=None, config: AgentConfig | None = None,
                 config_overrides=None, backend_factory=None, console=None, settings_path=None):
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("Workspace must be a directory.")
        self.user, self.scope, self.store = user, project_scope(self.workspace, user), store
        self.console = console if console is not None else Console(highlight=False)
        self.backend_factory = backend_factory or _backend
        self.backend = None
        self.settings_path = Path(settings_path) if settings_path else self.workspace / ".silivra" / "config.json"
        try:
            self.settings = load_settings(self.settings_path, self.scope)
        except (ValueError, OSError) as error:
            self.settings = TerminalSettings()
            self.error(f"Preferences could not be loaded: {error}")
        changes = {}
        if run is not None:
            changes["model_run"] = str(self._path(run))
        if which is not None:
            changes["which"] = which
        if max_new_tokens is not None:
            changes["max_new_tokens"] = max_new_tokens
        if tool_names != ():
            selected = None if tool_names is None else list(tool_names)
            unknown = set(selected or []) - _ALL_TOOLS
            if unknown:
                raise ValueError(f"Unknown tools: {', '.join(sorted(unknown))}")
            changes["selected_tools"] = selected
        if config is not None:
            for key in ("max_steps", "max_seconds", "history_messages", "memory_chars"):
                changes[key] = getattr(config, key)
        if config_overrides is not None:
            if (not isinstance(config_overrides, dict) or
                    set(config_overrides) - {"max_steps", "max_seconds", "history_messages", "memory_chars"}):
                raise ValueError("Config overrides must contain only terminal turn budgets.")
            changes.update(config_overrides)
        self.settings = replace(self.settings, **changes)
        self.permissions = {"read": allow_read, "write": allow_write,
                            "network": allow_network, "exec": allow_exec}
        if any(type(value) is not bool for value in self.permissions.values()):
            raise ValueError("Permissions must be explicit booleans.")
        self.session_id = None
        if session_id is not None:
            self.resume(session_id)
        self._streamed = False
        self._provisional = False
        self._event_tools = 0

    def _path(self, value):
        path = Path(value).expanduser()
        return (self.workspace / path).resolve() if not path.is_absolute() else path.resolve()

    def write(self, message="", *, style=None, end="\n"):
        self.console.print(Text(_safe(message), style=style), end=end, soft_wrap=True)

    def error(self, message):
        self.write(f"Error: {message}", style="red")

    def data(self, value, title):
        encoded = json.dumps(_redact(value), indent=2, ensure_ascii=False, allow_nan=False)
        if len(encoded) > 16000:
            encoded = encoded[:16000] + "\n… display truncated; stored/tool limits are separate."
        self.console.print(Panel(Text(_safe(encoded)), title=Text(_safe(title)), border_style="dim"))

    def banner(self):
        self.console.print(Panel(Text("SILIVRA\nLocal AI · original weights · private project memory", style="cyan"),
                                 border_style="cyan"))
        self.write(f"Workspace: {self.workspace}\nUser: {self.user}")
        self.write("/help for commands · Enter sends · Alt+Enter / Ctrl+J inserts a newline · Ctrl+C cancels")
        self.write("Experimental checkpoint: training and evaluations determine model ability.", style="dim")
        self.status()

    def _save(self, candidate):
        # Save before accepting preferences so malformed private state stays intact.
        save_settings(self.settings_path, self.scope, candidate)
        self.settings = candidate

    def _tools(self):
        registry = ToolRegistry(self.workspace, allow_write=self.permissions["write"],
                                allow_network=self.permissions["network"], allow_exec=self.permissions["exec"])
        return _PermittedTools(registry, self.permissions["read"])

    def _tool_names(self, tools):
        available = {schema["function"]["name"] for schema in tools.schemas()}
        selected = self.settings.selected_tools
        return None if selected is None else [name for name in selected if name in available]

    def status(self):
        tools = self._tools()
        selected = self._tool_names(tools)
        model = self.settings.model_run or str(default_model_run(self.workspace)) + " (default; not loaded)"
        context = "unknown until model loads"
        checkpoint = "not loaded"
        if self.backend is not None:
            checkpoint = str(getattr(self.backend, "checkpoint", "injected backend"))
            model_config = getattr(getattr(self.backend, "model", None), "config", None)
            limit = getattr(model_config, "max_seq_len", None)
            if limit is not None:
                context = f"{limit} tokens; output reservation {self.settings.max_new_tokens}"
        table = Table(show_header=False, box=None, padding=(0, 2))
        for key, value in (("Model", model), ("Checkpoint", checkpoint), ("Context", context),
                           ("Session", self.session_id or "new; created on first turn"),
                           ("Permissions", ", ".join(f"{key}={'on' if value else 'off'}" for key, value in self.permissions.items())),
                           ("Model tools", "all enabled" if selected is None else ", ".join(selected) or "none"),
                           ("Budgets", f"{self.settings.max_steps} steps · {self.settings.max_seconds:g}s · "
                            f"{self.settings.history_messages} history messages · {self.settings.memory_chars} memory chars")):
            table.add_row(Text(key, style="dim"), Text(_safe(value)))
        self.console.print(table)

    def help(self):
        commands = [
            ("/help", "Show commands and input controls"),
            ("/new", "Start a fresh conversation; durable memory stays"),
            ("/sessions · /resume ID · /history [LIMIT]", "List, resume, and inspect this user/project's conversations"),
            ("/status · /models · /model [PATH]", "Inspect status, local runs, or select a checkpoint/run"),
            ("/tools [all|none|NAME …]", "Inspect schemas or select tools exposed to the model"),
            ("/tool NAME {JSON}", "Execute one explicit user-requested tool with current permissions"),
            ("/permissions [read|write|network|exec on|off]", "Inspect or change process-local capability grants"),
            ("/memory list · remember KEY TEXT · recall QUERY", "Explicitly curate or search durable scoped facts"),
            ("/memory forget KEY · revisions KEY", "Remove a fact or inspect its corrections"),
            ("/memory export PATH · backup PATH", "Private JSON export of scope / SQLite backup of all scopes"),
            ("/memory delete-session ID", "Delete a scoped transcript"),
            ("/config [KEY VALUE]", "Inspect/set output tokens, history, memory and turn budgets"),
            ("/clear · /exit", "Clear the screen / exit"),
        ]
        table = Table("Command", "Action", box=None)
        for command, action in commands:
            table.add_row(Text(command, style="cyan"), Text(action))
        self.console.print(table)
        self.write("Quote paths containing spaces. Enter sends; Alt+Enter/Ctrl+J adds a newline; pasted multiline text stays together.")
        self.write("Ctrl+C cancels input or the current turn; Ctrl+D exits an empty prompt. Tool data cannot grant permissions.")
        self.write("Command execution runs as your OS user and is not a sandbox. Grants reset when Silivra exits.", style="yellow")

    def resume(self, session_id):
        self.store.recent_messages(self.scope, session_id, limit=1)
        self.session_id = session_id

    def _session(self):
        if self.session_id is None:
            self.session_id = self.store.create_session(self.scope)
        return self.session_id

    def _require(self, args, count, usage):
        values = shlex.split(args)
        if len(values) != count:
            raise ValueError(f"Usage: {usage}")
        return values

    def handle_input(self, text: str) -> bool:
        if not isinstance(text, str):
            raise ValueError("Input must be text.")
        if not text.strip():
            return True
        self._streamed, self._provisional = False, False
        try:
            if text.lstrip().startswith("/"):
                return self.command(text.strip())
            self.run_turn(text)
        except KeyboardInterrupt:
            if self._streamed:
                self.write()
            if self._provisional:
                self.write("Draft interrupted; no validated final answer.", style="yellow")
            self.write("Turn cancelled. The session and completed tool effects are retained.", style="yellow")
        except (ValueError, OSError, RuntimeError) as error:
            self.error(error)
        return True

    def command(self, text):
        # Commands can be separated from arguments by tabs/newlines as well.
        pieces = text.split(None, 1)
        command, args = pieces[0], pieces[1] if len(pieces) > 1 else ""
        if command in ("/exit", "/quit"):
            self._require(args, 0, "/exit")
            return False
        if command == "/help":
            self._require(args, 0, "/help")
            self.help()
        elif command == "/new":
            self._require(args, 0, "/new")
            self.session_id = None
            self.write("New conversation. Durable facts remain available.", style="cyan")
        elif command == "/sessions":
            self._require(args, 0, "/sessions")
            sessions = self.store.list_sessions(self.scope)
            table = Table("Session", "Messages", "Created (UTC)", box=None)
            for item in sessions:
                table.add_row(Text(item["id"]), str(item["message_count"]), item["created_at"])
            self.console.print(table)
            if not sessions:
                self.write("No conversations yet.", style="dim")
        elif command == "/resume":
            session, = self._require(args, 1, "/resume SESSION_ID")
            self.resume(session)
            self.write(f"Resumed {session}.", style="cyan")
        elif command == "/history":
            values = shlex.split(args)
            if len(values) > 1:
                raise ValueError("Usage: /history [LIMIT]")
            limit = int(values[0]) if values else 40
            if not 1 <= limit <= 1000:
                raise ValueError("History limit must be between 1 and 1000.")
            if self.session_id is None:
                self.write("No active conversation yet.", style="dim")
            else:
                for item in self.store.recent_messages(self.scope, self.session_id, limit=limit):
                    content = item["content"]
                    if item["role"] == "assistant":
                        try:
                            payload = json.loads(content)
                            content = payload.get("final", content) if isinstance(payload, dict) else content
                        except json.JSONDecodeError:
                            pass
                    self.write(f"{item['role']} · {item['created_at']}", style="cyan")
                    self.write(content)
        elif command == "/status":
            self._require(args, 0, "/status")
            self.status()
        elif command == "/models":
            self._require(args, 0, "/models")
            self.models()
        elif command == "/model":
            if not args:
                self.status()
            else:
                name, = self._require(args, 1, "/model PATH")
                path = self._path(name)
                if not path.is_dir():
                    raise ValueError("Model path must be an existing local checkpoint or run directory.")
                self._save(replace(self.settings, model_run=str(path)))
                self.backend = None
                self.write(f"Selected {path}. Weights load on the next message; use /new to start fresh.", style="cyan")
        elif command == "/permissions":
            if args:
                capability, setting = self._require(args, 2, "/permissions read|write|network|exec on|off")
                if capability not in self.permissions or setting not in ("on", "off"):
                    raise ValueError("Usage: /permissions read|write|network|exec on|off")
                self.permissions[capability] = setting == "on"
                if capability == "exec" and setting == "on":
                    self.write("Command execution runs as your OS user and is not a sandbox.", style="yellow")
            self.status()
        elif command == "/tools":
            if args:
                values = shlex.split(args)
                if values == ["all"]:
                    selection = None
                elif values == ["none"]:
                    selection = []
                else:
                    unknown = set(values) - _ALL_TOOLS
                    if unknown:
                        raise ValueError(f"Unknown tools: {', '.join(sorted(unknown))}")
                    selection = list(dict.fromkeys(values))
                self._save(replace(self.settings, selected_tools=selection))
            self.data(self._tools().schemas(), "Enabled tool schemas")
            self.write("Selected model tools: " + ("all enabled" if self.settings.selected_tools is None
                       else ", ".join(self.settings.selected_tools) or "none"))
            self.write("Selections require matching permissions; disabled capabilities are never exposed.", style="dim")
        elif command == "/tool":
            values = args.split(None, 1)
            if len(values) != 2:
                raise ValueError('Usage: /tool NAME {"argument":"value"}')
            name, encoded = values
            arguments = _json(encoded)
            result = self._tools().execute(name, arguments)
            session = self._session()
            self.store.add_message(self.scope, session, "user", text)
            self.store.add_message(self.scope, session, "tool", json.dumps({"tool": name, "observation": result}, ensure_ascii=False))
            self.data(result, name)
        elif command == "/memory":
            self.memory_command(args)
        elif command == "/config":
            self.configure(args)
        elif command == "/clear":
            self._require(args, 0, "/clear")
            self.console.clear()
        else:
            raise ValueError(f"Unknown command {command}. Use /help. To ask about a path, begin with ordinary text.")
        return True

    def configure(self, args):
        keys = {"max-new-tokens": "max_new_tokens", "max-steps": "max_steps", "max-seconds": "max_seconds",
                "history-messages": "history_messages", "memory-chars": "memory_chars"}
        if args:
            key, raw = self._require(args, 2, "/config KEY VALUE")
            if key not in keys:
                raise ValueError("Config key must be one of: " + ", ".join(keys))
            value = float(raw) if key == "max-seconds" else int(raw)
            candidate = replace(self.settings, **{keys[key]: value})
            self._save(candidate)
            if key == "max-new-tokens":
                self.backend = None
        self.data({key: getattr(self.settings, value) for key, value in keys.items()}, "Terminal preferences")

    def memory_command(self, args):
        values = args.split(None, 1)
        if not values:
            raise ValueError("Usage: /memory list|remember|recall|forget|revisions|export|backup|delete-session …")
        operation, rest = values[0], values[1] if len(values) > 1 else ""
        if operation == "list":
            self._require(rest, 0, "/memory list")
            result = self.store.list_memories(self.scope)
        elif operation == "remember":
            parts = rest.split(None, 1)
            if len(parts) != 2:
                raise ValueError("Usage: /memory remember KEY TEXT (key contains no spaces)")
            result = self.store.remember(self.scope, parts[0], parts[1], source="user")
        elif operation == "recall":
            if not rest.strip():
                raise ValueError("Usage: /memory recall QUERY")
            result = self.store.recall(self.scope, rest)
        elif operation in ("forget", "revisions", "delete-session"):
            key, = self._require(rest, 1, f"/memory {operation} KEY_OR_ID")
            if operation == "forget":
                result = {"forgotten": self.store.forget(self.scope, key)}
            elif operation == "revisions":
                result = self.store.revisions(self.scope, key)
            else:
                self.store.delete_session(self.scope, key)
                if self.session_id == key:
                    self.session_id = None
                result = {"deleted_session": key}
        elif operation in ("export", "backup"):
            filename, = self._require(rest, 1, f"/memory {operation} PATH")
            path = self._path(filename)
            if operation == "backup":
                result = {"backup": self.store.backup(path), "includes_all_scopes": True}
            else:
                # Exclusive creation cannot replace source files or older exports.
                flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_NOFOLLOW
                with os.fdopen(os.open(path, flags, 0o600), "w", encoding="utf-8") as output:
                    json.dump(self.store.export(self.scope), output, ensure_ascii=False, indent=2)
                    output.write("\n")
                result = {"export": str(path), "scope_only": True}
        else:
            raise ValueError("Unknown memory operation. Use /help.")
        self.data(result, "Memory · " + operation)

    def models(self):
        locations = {self.workspace / "runs", Path(__file__).resolve().parents[2] / "runs"}
        table = Table("Local model/run", "Reference", "Context", "Training mode", box=None)
        count = 0
        for runs in sorted(locations):
            if not runs.is_dir():
                continue
            for path in sorted(runs.iterdir()):
                if not path.is_dir() or path.is_symlink():
                    continue
                for reference in ("best", "latest"):
                    pointer = path / f"{reference}.json"
                    checkpoint = path
                    try:
                        if pointer.is_file() and pointer.stat().st_size < 8192:
                            value = json.loads(pointer.read_text())
                            checkpoint = (path / value["checkpoint"]).resolve()
                            if not checkpoint.is_relative_to(path.resolve()):
                                continue
                        elif not (path / "model.safetensors").is_file() or reference == "latest":
                            continue
                        state_file = checkpoint / "state.json"
                        if not state_file.is_file() or state_file.is_symlink():
                            continue
                        state = json.loads(state_file.read_text()) if state_file.stat().st_size < 131072 else {}
                        limit = state.get("model_config", {}).get("max_seq_len", "unknown")
                        table.add_row(Text(_safe(path)), reference,
                                      str(limit), Text(_safe(state.get("training_mode", "pretraining"))))
                        count += 1
                    except (OSError, ValueError, KeyError, TypeError):
                        continue
        self.console.print(table)
        if not count:
            self.write("No readable local checkpoints under workspace/source runs. /model PATH can select another local run.")
        self.write("Metadata is informational until checkpoint hashes verify at model load. Context length is not a quality score.", style="dim")

    def _event(self, event):
        kind = event.get("type")
        if kind == "context" and (event.get("history_dropped") or event.get("memories_dropped")):
            self.write(f"Context: omitted {event.get('history_dropped', 0)} older messages and "
                       f"{event.get('memories_dropped', 0)} lower-ranked memories to fit this turn.", style="dim")
        elif kind == "tool_start":
            action = event.get("action", {})
            self.write("Tool · " + str(action.get("tool", event.get("tool", "unknown"))), style="cyan")
        elif kind == "tool_result":
            action = event.get("action", {})
            self.data(event.get("observation", event.get("result", {})), str(action.get("tool", event.get("tool", "Tool result"))))
            self._event_tools += 1
        elif kind == "final_text":
            if not self._streamed:
                self._provisional = bool(event.get("provisional"))
                self.write("Silivra · draft (awaiting protocol validation)" if self._provisional else "Silivra", style="cyan")
            self.write(event.get("text", ""), end="")
            self._streamed = True

    def run_turn(self, prompt):
        if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 32_000:
            raise ValueError("prompt must contain 1–32,000 characters.")
        if self.backend is None:
            run = self._path(self.settings.model_run) if self.settings.model_run else default_model_run(self.workspace)
            self.write("Loading local checkpoint…", style="dim")
            self.backend = self.backend_factory(run, which=self.settings.which,
                                                max_new_tokens=self.settings.max_new_tokens)
        tools = self._tools()
        agent = Agent(self.backend, tools, self.store, self.scope,
                      config=self.settings.agent_config(), tool_names=self._tool_names(tools))
        self._streamed, self._provisional, self._event_tools = False, False, 0
        session = self._session()
        parameters = inspect.signature(agent.run).parameters
        kwargs = {"session_id": session}
        if "on_event" in parameters:
            kwargs["on_event"] = self._event
        result = agent.run(prompt, **kwargs)
        self.session_id = result["session_id"]
        if self._streamed:
            self.write()
        for event in result.get("events", [])[self._event_tools:]:
            self.data(event.get("observation", {}), event.get("action", {}).get("tool", "Tool result"))
        if result["status"] == "completed":
            if not self._streamed:
                self.write("Silivra", style="cyan")
                self.write(result.get("answer", ""))
            elif self._provisional:
                self.write("Draft accepted: JSON action validated.", style="dim")
        else:
            if self._provisional:
                self.write("Draft rejected or incomplete; no validated final answer.", style="yellow")
            self.write(f"Stopped · {result['status']}: {result.get('error', 'turn budget or policy limit reached')}", style="yellow")
            if result["status"] == "context_budget" or "Context budget" in result.get("error", ""):
                self.write("Use /new, fewer /tools, /config memory-chars 0, /config history-messages 0, or a smaller output reservation.", style="dim")
        self.write(f"Session {self.session_id}", style="dim")
        return result


class ScopedInputHistory(History):
    """Bounded, redacted input history using the conversation's exact scope."""

    max_bytes = 262144
    max_entries = 500

    def __init__(self, workspace, scope):
        super().__init__()
        if not isinstance(scope, str) or re.fullmatch(r"[a-f0-9]{64}", scope) is None:
            raise ValueError("Input history requires a project/user scope hash.")
        directory = private_directory(Path(workspace) / ".silivra")
        self.path = directory / f"input-{scope}.jsonl"
        self._check_path()

    def _check_path(self):
        if self.path.is_symlink():
            raise ValueError("Input history must not be a symlink.")

    def load_history_strings(self):
        self._check_path()
        if not self.path.exists():
            return
        descriptor = os.open(self.path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
        with os.fdopen(descriptor, "rb") as source:
            metadata = os.fstat(source.fileno())
            if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
                raise ValueError("Input history must be a regular file without hardlinks.")
            os.fchmod(source.fileno(), 0o600)
            source.seek(max(0, metadata.st_size - self.max_bytes))
            if source.tell():
                # Skip the incomplete first record as bytes, before UTF-8
                # decoding; the bounded tail may begin in a multibyte character.
                source.readline()
            values = source.read(self.max_bytes).splitlines()[-self.max_entries:]
        for line in reversed(values):
            try:
                value = json.loads(line)
                if isinstance(value, str):
                    yield _redact(value)
            except (ValueError, UnicodeError):
                continue

    def store_string(self, string):
        # Rewrite only the bounded tail instead of growing a private transcript
        # indefinitely. Raw prompt_toolkit FileHistory stores secrets unchanged.
        values = list(reversed(list(self.load_history_strings())))
        values.append(_redact(string[:32000]))
        lines = [json.dumps(value, ensure_ascii=False).encode("utf-8") + b"\n"
                 for value in values[-self.max_entries:]]
        total = sum(map(len, lines))
        while total > self.max_bytes and lines:
            total -= len(lines.pop(0))
        descriptor, temporary = tempfile.mkstemp(prefix=".input-", dir=self.path.parent)
        try:
            with os.fdopen(descriptor, "wb") as output:
                output.writelines(lines)
                output.flush()
                os.fsync(output.fileno())
            self._check_path()
            os.replace(temporary, self.path)
        finally:
            Path(temporary).unlink(missing_ok=True)


def _prompt_session(app):
    from prompt_toolkit import PromptSession
    from prompt_toolkit.completion import Completer, Completion
    from prompt_toolkit.key_binding import KeyBindings

    class Commands(Completer):
        def get_completions(self, document, complete_event):
            text = document.text_before_cursor
            if text.startswith("/") and not any(char.isspace() for char in text):
                for command in _COMMANDS:
                    if command.startswith(text):
                        yield Completion(command, start_position=-len(text))
            elif text.startswith("/tools ") or text.startswith("/tool "):
                prefix = text.rsplit(" ", 1)[-1]
                for name in sorted(_ALL_TOOLS):
                    if name.startswith(prefix):
                        yield Completion(name, start_position=-len(prefix))

    bindings = KeyBindings()

    @bindings.add("enter")
    def send(event):
        event.current_buffer.validate_and_handle()

    @bindings.add("escape", "enter")
    @bindings.add("c-j")
    def newline(event):
        event.current_buffer.insert_text("\n")

    return PromptSession(multiline=True, key_bindings=bindings, history=ScopedInputHistory(app.workspace, app.scope),
                         prompt_continuation="        · ",
                         completer=Commands(), complete_while_typing=False,
                         enable_open_in_editor=False)


def _check_database_path(database):
    """Refuse linked private SQLite files before MemoryStore resolves its path."""
    if str(database) == ":memory:":
        return database
    database = Path(database).expanduser()
    if database.parent.is_symlink():
        raise ValueError("Private database directory must not be a symlink.")
    for suffix in ("", "-wal", "-shm", "-journal"):
        candidate = Path(str(database) + suffix)
        if candidate.is_symlink():
            raise ValueError("Private database and sidecars must not be symlinks.")
        try:
            metadata = candidate.stat()
        except FileNotFoundError:
            continue
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("Private database and sidecars must be regular files without hardlinks.")
    return database


def run_terminal(*, workspace: Path, user="local", run=None, which=None, memory=None,
                 session_id=None, tool_names=(), allow_read=True, allow_write=False,
                 allow_network=False, allow_exec=False, max_new_tokens=None,
                 config=None, config_overrides=None, backend_factory=None, console=None, prompt_session=None) -> int:
    """Open an interactive terminal; no model loads until the first chat request."""
    workspace = Path(workspace).expanduser().resolve(strict=True)
    private_directory(workspace / ".silivra")
    database = _check_database_path(memory or workspace / ".silivra" / "memory.sqlite")
    with MemoryStore(database) as store:
        app = TerminalApp(workspace, store, user=user, run=run, which=which, session_id=session_id,
                          tool_names=tool_names, allow_read=allow_read, allow_write=allow_write,
                          allow_network=allow_network, allow_exec=allow_exec,
                          max_new_tokens=max_new_tokens, config=config, config_overrides=config_overrides,
                          backend_factory=backend_factory, console=console)
        app.banner()
        prompt = prompt_session if prompt_session is not None else _prompt_session(app)
        while True:
            try:
                text = prompt.prompt("silivra › ")
            except KeyboardInterrupt:
                app.write("Input cancelled.", style="dim")
                continue
            except EOFError:
                app.write("Goodbye.", style="dim")
                return 0
            if not app.handle_input(text):
                return 0

"""Versioned, private terminal preferences. Permission grants are never persisted.

The workspace-local file is deliberately not a global machine configuration. Its
opaque scope keys use the same local-user/project identity as assistant memory.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import stat
import tempfile


SETTINGS_VERSION = 1
_MAX_SETTINGS_BYTES = 131_072


@dataclass(frozen=True)
class TerminalSettings:
    model_run: str | None = None
    which: str = "best"
    max_new_tokens: int = 96
    selected_tools: list[str] | None = field(default_factory=list)
    max_steps: int = 6
    max_seconds: float = 120.0
    history_messages: int = 8
    memory_chars: int = 1500

    def __post_init__(self):
        if self.model_run is not None and (not isinstance(self.model_run, str)
                                          or not self.model_run.strip() or len(self.model_run) > 4096):
            raise ValueError("model_run must be a nonempty local path or null.")
        if self.which not in ("best", "latest"):
            raise ValueError("which must be best or latest.")
        if type(self.max_new_tokens) is not int or not 1 <= self.max_new_tokens <= 4096:
            raise ValueError("max_new_tokens must be between 1 and 4096.")
        if self.selected_tools is not None:
            if (not isinstance(self.selected_tools, list) or len(self.selected_tools) > 32
                    or any(not isinstance(name, str) or not name or len(name) > 64
                           for name in self.selected_tools)):
                raise ValueError("selected_tools must be a list of tool names or null (all enabled tools).")
            if len(set(self.selected_tools)) != len(self.selected_tools):
                raise ValueError("selected_tools must not contain duplicate names.")
        from silivra.agent import AgentConfig
        AgentConfig(max_steps=self.max_steps, max_seconds=self.max_seconds,
                    history_messages=self.history_messages, memory_chars=self.memory_chars)

    def agent_config(self):
        from silivra.agent import AgentConfig
        return AgentConfig(max_steps=self.max_steps, max_seconds=self.max_seconds,
                           history_messages=self.history_messages, memory_chars=self.memory_chars)


def private_directory(path: Path) -> Path:
    """Create private state without following a substituted state-directory link."""
    path = Path(path)
    if path.is_symlink():
        raise ValueError("Private state directory must not be a symlink.")
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if not path.is_dir():
        raise ValueError("Private state path must be a directory.")
    path.chmod(0o700)
    return path


def _settings(value) -> TerminalSettings:
    if not isinstance(value, dict):
        raise ValueError("Invalid terminal settings for this user/project.")
    try:
        return TerminalSettings(**value)
    except TypeError as error:
        raise ValueError("Unknown terminal setting; existing settings were preserved.") from error


def _read_document(path: Path) -> dict:
    path = Path(path)
    if path.parent.is_symlink():
        raise ValueError("Private state directory must not be a symlink.")
    if path.is_symlink():
        raise ValueError("Terminal settings must not be a symlink.")
    if not path.exists():
        return {"version": SETTINGS_VERSION, "scopes": {}}
    descriptor = os.open(path, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK)
    with os.fdopen(descriptor, "r", encoding="utf-8") as source:
        metadata = os.fstat(source.fileno())
        if not stat.S_ISREG(metadata.st_mode) or metadata.st_nlink != 1:
            raise ValueError("Terminal settings must be a regular file without hardlinks.")
        if metadata.st_size > _MAX_SETTINGS_BYTES:
            raise ValueError("Terminal settings file is too large.")
        os.fchmod(source.fileno(), 0o600)
        try:
            value = json.load(source)
        except (json.JSONDecodeError, UnicodeError) as error:
            raise ValueError("Terminal settings are invalid JSON; preserve or repair the existing file.") from error
    if (not isinstance(value, dict) or set(value) != {"version", "scopes"}
            or type(value["version"]) is not int or value["version"] != SETTINGS_VERSION
            or not isinstance(value["scopes"], dict)):
        raise ValueError("Unsupported terminal settings format; existing settings were preserved.")
    for settings in value["scopes"].values():
        _settings(settings)
    return value


def load_settings(path: Path, scope: str) -> TerminalSettings:
    if not isinstance(scope, str) or not scope or len(scope) > 512:
        raise ValueError("A valid user/project scope is required.")
    return _settings(_read_document(path)["scopes"].get(scope, {}))


def save_settings(path: Path, scope: str, settings: TerminalSettings) -> None:
    """Atomically replace valid preferences; never replace malformed configuration."""
    if not isinstance(scope, str) or not scope or len(scope) > 512:
        raise ValueError("A valid user/project scope is required.")
    if not isinstance(settings, TerminalSettings):
        raise ValueError("settings must be TerminalSettings.")
    path = Path(path)
    private_directory(path.parent)
    document = _read_document(path)
    document["scopes"][scope] = asdict(settings)
    encoded = json.dumps(document, indent=2, ensure_ascii=False, allow_nan=False) + "\n"
    if len(encoded.encode("utf-8")) > _MAX_SETTINGS_BYTES:
        raise ValueError("Terminal settings exceed the private file size limit.")
    descriptor, temporary = tempfile.mkstemp(prefix=".settings-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as output:
            output.write(encoded)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)

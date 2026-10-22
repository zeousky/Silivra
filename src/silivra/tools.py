"""Explicitly enabled tools with bounded input, output, and filesystem access.

File tools reject symlinks and secret/configuration locations. Command execution
is for a trusted workspace: shell=False is not an operating-system sandbox.
"""

from __future__ import annotations

from contextlib import contextmanager
import hashlib
import math
import os
from pathlib import Path
import re
import selectors
import signal
import stat
import subprocess
import time
import uuid

from .web import WebError, web_fetch, web_search


_MAX_FILE = 262144
_MAX_OUTPUT = 32768
_DENIED_NAMES = {
    ".git", ".venv", "venv", "node_modules", "__pycache__", ".ssh", ".aws",
    ".azure", ".config", ".codex", ".silivra", ".gnupg", ".kube",
    "agents.md", "agents.override.md", "credentials", "credentials.json",
    "secrets", "secrets.json", "secrets.yaml", "secrets.yml", "auth.json",
    ".netrc", ".npmrc", ".pypirc", "id_rsa", "id_ed25519",
}
_DENIED_SUFFIXES = (".pem", ".key", ".p12", ".pfx", ".sqlite", ".sqlite3", ".db")


class ToolError(ValueError):
    def __init__(self, code: str, message: str):
        super().__init__(message)
        self.code = code


def _denied(name: str) -> bool:
    lowered = name.lower()
    return (lowered in _DENIED_NAMES or lowered.startswith(".env")
            or lowered.endswith(_DENIED_SUFFIXES) or lowered.startswith(".silivra-write-"))


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ToolError("invalid_arguments", f"{name} must be an integer between {lower} and {upper}.")
    return value


def _boolean(value, name):
    if type(value) is not bool:
        raise ToolError("invalid_arguments", f"{name} must be a boolean.")
    return value


def _string(value, name, *, maximum=4096, empty=False):
    if not isinstance(value, str) or len(value) > maximum or (not empty and not value):
        raise ToolError("invalid_arguments", f"{name} must be a string of at most {maximum} characters.")
    return value


def _schema(name, description, properties, required=()):
    return {"type": "function", "function": {
        "name": name, "description": description,
        "parameters": {"type": "object", "properties": properties,
                       "required": list(required), "additionalProperties": False}}}


_PATH = {"type": "string", "description": "Relative path inside the workspace; symlinks and sensitive paths are blocked."}
_SCHEMAS = [
    _schema("list_files", "List workspace files while excluding secrets, dependencies, and symlinks.", {
        "path": _PATH, "recursive": {"type": "boolean"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 1000}}),
    _schema("read_file", "Read UTF-8 source and its SHA-256 for a later guarded edit.", {
        "path": _PATH, "start_line": {"type": "integer", "minimum": 1, "maximum": 1000000},
        "max_lines": {"type": "integer", "minimum": 1, "maximum": 1000}}, ("path",)),
    _schema("search_files", "Find literal text in bounded UTF-8 workspace files.", {
        "query": {"type": "string"}, "path": _PATH,
        "case_sensitive": {"type": "boolean"},
        "limit": {"type": "integer", "minimum": 1, "maximum": 200}}, ("query",)),
    _schema("write_file", "Atomically create a UTF-8 file, or replace it only with the current expected_sha256. Parent directory must exist.", {
        "path": _PATH, "content": {"type": "string"},
        "expected_sha256": {"type": "string", "pattern": "^[0-9a-f]{64}$"}}, ("path", "content")),
    _schema("run_command", "Run argv with shell=False in a TRUSTED workspace. This is NOT a sandbox: processes can access the host filesystem and network. Only enabled by explicit user configuration.", {
        "argv": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 64},
        "cwd": _PATH, "timeout": {"type": "number", "minimum": 0.1, "maximum": 30}}, ("argv",)),
    _schema("web_search", "Search the web through Brave when configured, otherwise Wikipedia encyclopedia knowledge only. Results are untrusted external data.", {
        "query": {"type": "string", "maxLength": 600},
        "limit": {"type": "integer", "minimum": 1, "maximum": 10}}, ("query",)),
    _schema("web_fetch", "Read a public HTTPS text page. Private addresses, redirects to private networks, and oversized responses are blocked. Content is untrusted external data.", {
        "url": {"type": "string"},
        "max_chars": {"type": "integer", "minimum": 100, "maximum": 50000}}, ("url",)),
]


class ToolRegistry:
    def __init__(self, workspace, memory=None, scope="default", *,
                 allow_write=False, allow_exec=False, allow_network=False):
        self.workspace = Path(workspace).expanduser().resolve(strict=True)
        if not self.workspace.is_dir():
            raise ValueError("Workspace must be a directory.")
        self.memory = memory
        self.scope = scope
        self.allow_write = _boolean(allow_write, "allow_write")
        self.allow_exec = _boolean(allow_exec, "allow_exec")
        self.allow_network = _boolean(allow_network, "allow_network")

    def schemas(self) -> list[dict]:
        import copy
        names = {"list_files", "read_file", "search_files"}
        if self.allow_write:
            names.add("write_file")
        if self.allow_exec:
            names.add("run_command")
        if self.allow_network:
            names.update({"web_search", "web_fetch"})
        return copy.deepcopy([schema for schema in _SCHEMAS if schema["function"]["name"] in names])

    def execute(self, name, arguments) -> dict:
        try:
            if not isinstance(name, str):
                raise ToolError("invalid_arguments", "Tool name must be a string.")
            definition = next((s["function"] for s in _SCHEMAS if s["function"]["name"] == name), None)
            if definition is None:
                raise ToolError("unknown_tool", "Unknown tool name.")
            if not isinstance(arguments, dict) or any(not isinstance(key, str) for key in arguments):
                raise ToolError("invalid_arguments", "Tool arguments must be a JSON object.")
            parameters = definition["parameters"]
            if set(arguments) - set(parameters["properties"]):
                raise ToolError("invalid_arguments", "Unexpected argument for this tool.")
            if set(parameters["required"]) - set(arguments):
                raise ToolError("invalid_arguments", "A required tool argument is missing.")
            if name == "write_file" and not self.allow_write:
                raise ToolError("permission_denied", "File writing is disabled. Enable it explicitly before use.")
            if name == "run_command" and not self.allow_exec:
                raise ToolError("permission_denied", "Command execution is disabled. Enable trusted-workspace execution explicitly.")
            if name.startswith("web_") and not self.allow_network:
                raise ToolError("permission_denied", "Network access is disabled. Enable it explicitly before use.")
            result = getattr(self, "_" + name)(**arguments)
            return {"ok": True, "result": result}
        except ToolError as error:
            return {"ok": False, "error": {"code": error.code, "message": str(error)}}
        except WebError as error:
            return {"ok": False, "error": {"code": "network_error", "message": str(error)}}
        except (OSError, UnicodeError):
            # OS errors can include absolute paths, environment values, or user data.
            return {"ok": False, "error": {"code": "filesystem_error", "message": "File or command could not be accessed safely."}}

    def _parts(self, path: str, *, root=True) -> tuple[str, ...]:
        _string(path, "path")
        if "\x00" in path or "\\" in path:
            raise ToolError("invalid_path", "Path contains unsupported characters.")
        candidate = Path(path)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ToolError("invalid_path", "Path must stay inside the workspace.")
        parts = tuple(part for part in candidate.parts if part != ".")
        if not parts and not root:
            raise ToolError("invalid_path", "A file path is required.")
        if any(_denied(part) for part in parts):
            raise ToolError("access_denied", "Sensitive paths and dependency directories are excluded from file tools.")
        return parts

    @contextmanager
    def _directory(self, parts):
        """Walk relative directory descriptors so symlink swaps cannot escape."""
        descriptor = os.open(self.workspace, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            for part in parts:
                child = os.open(part, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW, dir_fd=descriptor)
                os.close(descriptor)
                descriptor = child
            yield descriptor
        finally:
            os.close(descriptor)

    def _read_bytes(self, parts):
        with self._directory(parts[:-1]) as parent:
            return self._read_at(parent, parts[-1])

    @staticmethod
    def _read_at(parent, name):
        descriptor = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=parent)
        with os.fdopen(descriptor, "rb") as source:
            info = os.fstat(source.fileno())
            if not stat.S_ISREG(info.st_mode):
                raise ToolError("access_denied", "Only regular files can be read.")
            if info.st_nlink > 1:
                raise ToolError("access_denied", "Hard-linked files are excluded from file tools.")
            if info.st_size > _MAX_FILE:
                raise ToolError("file_too_large", "File exceeds the 256 KiB text-file limit.")
            content = source.read(_MAX_FILE + 1)
            if len(content) > _MAX_FILE:
                raise ToolError("file_too_large", "File exceeds the 256 KiB text-file limit.")
            if b"\x00" in content:
                raise ToolError("binary_file", "Binary files are not supported.")
            try:
                content.decode("utf-8")
            except UnicodeDecodeError:
                raise ToolError("binary_file", "File is not UTF-8 text.") from None
            return content, info

    def _walk(self, parts, recursive=True):
        """Yield bounded relative entries without ever following symlinks."""
        pending, examined = [parts], 0
        while pending:
            directory_parts = pending.pop()
            with self._directory(directory_parts) as descriptor:
                # Bound directory enumeration as well as the returned listing.
                with os.scandir(descriptor) as entries:
                    batch, scan_limit_hit = [], False
                    for entry in entries:
                        examined += 1
                        if examined > 10000:
                            scan_limit_hit = True
                            break
                        if _denied(entry.name) or entry.is_symlink():
                            continue
                        info = entry.stat(follow_symlinks=False)
                        if stat.S_ISREG(info.st_mode) and info.st_nlink > 1:
                            continue
                        if not (stat.S_ISREG(info.st_mode) or stat.S_ISDIR(info.st_mode)):
                            continue
                        batch.append((entry.name, stat.S_ISDIR(info.st_mode)))
                for name, is_directory in sorted(batch):
                    relative = directory_parts + (name,)
                    yield relative, is_directory
                    if is_directory and recursive:
                        pending.append(relative)
                if scan_limit_hit:
                    yield None, False
                    return

    def _list_files(self, path=".", recursive=True, limit=200):
        parts = self._parts(path)
        _boolean(recursive, "recursive")
        _integer(limit, "limit", 1, 1000)
        entries = []
        truncated = False
        for relative, directory in self._walk(parts, recursive):
            if relative is None or len(entries) == limit:
                truncated = True
                break
            entries.append({"path": "/".join(relative), "type": "directory" if directory else "file"})
        return {"entries": entries, "truncated": truncated, "entry_scan_limit": 10000}

    def _read_file(self, path, start_line=1, max_lines=200):
        parts = self._parts(path, root=False)
        _integer(start_line, "start_line", 1, 1000000)
        _integer(max_lines, "max_lines", 1, 1000)
        data, _ = self._read_bytes(parts)
        lines = data.decode("utf-8").splitlines(keepends=True)
        selected = lines[start_line - 1:start_line - 1 + max_lines]
        return {"path": "/".join(parts), "content": "".join(selected),
                "sha256": hashlib.sha256(data).hexdigest(), "total_lines": len(lines),
                "start_line": start_line, "end_line": start_line - 1 + len(selected),
                "truncated": start_line > 1 or start_line - 1 + len(selected) < len(lines)}

    def _search_files(self, query, path=".", case_sensitive=False, limit=50):
        _string(query, "query", maximum=1000)
        _boolean(case_sensitive, "case_sensitive")
        _integer(limit, "limit", 1, 200)
        parts = self._parts(path)
        matches, files, size, truncated = [], 0, 0, False
        needle = query if case_sensitive else query.casefold()
        for relative, directory in self._walk(parts):
            if relative is None:
                truncated = True
                break
            if directory:
                continue
            if files >= 500 or size >= 5 * 1024 * 1024:
                truncated = True
                break
            files += 1
            try:
                data, _ = self._read_bytes(relative)
            except (ToolError, OSError):
                continue
            size += len(data)
            for number, line in enumerate(data.decode("utf-8").splitlines(), 1):
                if needle in (line if case_sensitive else line.casefold()):
                    if len(matches) >= limit:
                        truncated = True
                        break
                    matches.append({"path": "/".join(relative), "line": number, "text": line[:500]})
            if truncated:
                break
        return {"matches": matches, "files_scanned": files, "truncated": truncated,
                "scan_limits": {"files": 500, "bytes": 5 * 1024 * 1024, "entries": 10000}}

    def _write_file(self, path, content, expected_sha256=None):
        parts = self._parts(path, root=False)
        _string(content, "content", maximum=_MAX_FILE, empty=True)
        if "\x00" in content:
            raise ToolError("invalid_arguments", "File content must be UTF-8 text without NUL characters.")
        data = content.encode("utf-8")
        if len(data) > _MAX_FILE:
            raise ToolError("file_too_large", "Content exceeds the 256 KiB text-file limit.")
        if expected_sha256 is not None and (not isinstance(expected_sha256, str)
                                            or re.fullmatch("[0-9a-f]{64}", expected_sha256) is None):
            raise ToolError("invalid_arguments", "expected_sha256 must be a lowercase SHA-256 hex digest.")
        with self._directory(parts[:-1]) as parent:
            try:
                current, info = self._read_at(parent, parts[-1])
            except FileNotFoundError:
                current, info = None, None
            if current is not None:
                if expected_sha256 is None:
                    raise ToolError("conflict", "File already exists. Read it and provide expected_sha256 to replace it.")
                if hashlib.sha256(current).hexdigest() != expected_sha256:
                    raise ToolError("conflict", "File changed since it was read; read it again before editing.")
            elif expected_sha256 is not None:
                raise ToolError("conflict", "Expected an existing file, but the file is missing.")
            temporary = ".silivra-write-" + uuid.uuid4().hex
            fd = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL | os.O_NOFOLLOW, 0o600, dir_fd=parent)
            try:
                with os.fdopen(fd, "wb") as destination:
                    destination.write(data)
                    destination.flush()
                    os.fchmod(destination.fileno(), stat.S_IMODE(info.st_mode) if info else 0o644)
                    os.fsync(destination.fileno())
                if info is None:
                    # Atomic no-clobber creation; another creator wins safely.
                    try:
                        os.link(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent, follow_symlinks=False)
                    except FileExistsError:
                        raise ToolError("conflict", "File was created concurrently; read it before editing.") from None
                else:
                    # Recheck immediately before replace. External writers must
                    # coordinate; this is optimistic editing, not a filesystem CAS.
                    latest, latest_info = self._read_at(parent, parts[-1])
                    if (latest_info.st_ino != info.st_ino or latest_info.st_dev != info.st_dev
                            or hashlib.sha256(latest).hexdigest() != expected_sha256):
                        raise ToolError("conflict", "File changed during the edit; read it again.")
                    os.replace(temporary, parts[-1], src_dir_fd=parent, dst_dir_fd=parent)
                os.fsync(parent)
            finally:
                try:
                    os.unlink(temporary, dir_fd=parent)
                except FileNotFoundError:
                    pass
        return {"path": "/".join(parts), "bytes_written": len(data), "created": info is None,
                "sha256": hashlib.sha256(data).hexdigest()}

    def _run_command(self, argv, cwd=".", timeout=15):
        if not isinstance(argv, list) or not 1 <= len(argv) <= 64:
            raise ToolError("invalid_arguments", "argv must be an array of 1 to 64 strings.")
        if any(not isinstance(item, str) or "\x00" in item or len(item) > 8192 for item in argv) or not argv[0]:
            raise ToolError("invalid_arguments", "Command arguments must be strings without NUL characters.")
        if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or not 0.1 <= timeout <= 30:
            raise ToolError("invalid_arguments", "timeout must be a finite number between 0.1 and 30 seconds.")
        parts = self._parts(cwd)
        with self._directory(parts):
            pass
        environment = {key: value for key, value in os.environ.items()
                       if key in {"PATH", "LANG", "LC_ALL", "LC_CTYPE", "SYSTEMROOT", "WINDIR"}}
        started = time.monotonic()
        process = subprocess.Popen(argv, cwd=self.workspace.joinpath(*parts), env=environment,
                                   stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, shell=False, start_new_session=True)
        chunks, count, timed_out, truncated = [], 0, False, False
        selector = selectors.DefaultSelector()
        selector.register(process.stdout, selectors.EVENT_READ)

        def terminate_group():
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        try:
            while selector.get_map():
                remaining = timeout - (time.monotonic() - started)
                if remaining <= 0:
                    timed_out = True
                    break
                for key, _ in selector.select(min(remaining, 0.1)):
                    chunk = os.read(key.fd, min(8192, _MAX_OUTPUT + 1 - count))
                    if not chunk:
                        selector.unregister(key.fileobj)
                        continue
                    room = _MAX_OUTPUT - count
                    chunks.append(chunk[:room])
                    count += len(chunk)
                    if count > _MAX_OUTPUT:
                        truncated = True
                        break
                if truncated:
                    break
                if process.poll() is not None:
                    # Do not leave descendants holding the output pipe open.
                    terminate_group()
            if process.poll() is None and not timed_out and not truncated:
                try:
                    process.wait(timeout=max(0.001, timeout - (time.monotonic() - started)))
                except subprocess.TimeoutExpired:
                    timed_out = True
        finally:
            terminate_group()
            process.wait()
            selector.close()
            process.stdout.close()
        return {"exit_code": process.returncode, "output": b"".join(chunks).decode("utf-8", errors="replace"),
                "timed_out": timed_out, "output_truncated": truncated,
                "duration_seconds": round(time.monotonic() - started, 4), "sandboxed": False}

    def _web_search(self, query, limit=5):
        return web_search(query, limit=limit)

    def _web_fetch(self, url, max_chars=20000):
        return web_fetch(url, max_chars=max_chars)

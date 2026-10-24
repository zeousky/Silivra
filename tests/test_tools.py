"""Tool permissions, path confinement, guarded edits, and bounded processes."""

import hashlib
import os
from pathlib import Path
import sys
import tempfile
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from silivra.tools import ToolRegistry


class ToolTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        (self.root / "app.py").write_text("print('Hello')\n# second line\n", encoding="utf-8")
        self.tools = ToolRegistry(self.root)

    def success(self, name, arguments, tools=None):
        result = (tools or self.tools).execute(name, arguments)
        self.assertTrue(result["ok"], result)
        return result["result"]

    def test_only_explicitly_enabled_tools_are_advertised_and_executed(self):
        names = {item["function"]["name"] for item in self.tools.schemas()}
        self.assertEqual(names, {"calculate", "list_files", "read_file", "search_files"})
        for name, args in (("write_file", {"path": "new.py", "content": "x"}),
                           ("run_command", {"argv": [sys.executable]}),
                           ("web_search", {"query": "python"}),
                           ("web_fetch", {"url": "https://example.com"})):
            with self.subTest(name=name):
                result = self.tools.execute(name, args)
                self.assertEqual(result["error"]["code"], "permission_denied")
        enabled = ToolRegistry(self.root, allow_write=True, allow_exec=True, allow_network=True)
        self.assertEqual(len(enabled.schemas()), 8)
        schemas = enabled.schemas()
        schemas[0]["function"]["parameters"]["properties"].clear()
        self.assertTrue(enabled.schemas()[0]["function"]["parameters"]["properties"])

    def test_unknown_extra_missing_and_wrong_typed_arguments_are_rejected(self):
        for name, args in (("shell", {}), ("read_file", {}),
                           ("read_file", {"path": "app.py", "surprise": True}),
                           ("read_file", []), ("read_file", {"path": "app.py", "max_lines": True}),
                           ("list_files", {"recursive": "yes"}),
                           ("search_files", {"query": ""})):
            with self.subTest(name=name, args=args):
                self.assertFalse(self.tools.execute(name, args)["ok"])

    def test_calculate_is_pure_available_and_semantically_validated(self):
        result = self.success("calculate", {"expression": "0.1 + 0.2"})
        self.assertEqual(result["result"], "0.3")
        self.assertFalse(result["rounded"])
        schema = next(item["function"] for item in self.tools.schemas()
                      if item["function"]["name"] == "calculate")
        self.assertEqual(schema["parameters"]["properties"]["expression"]["maxLength"], 512)
        for arguments in ({}, {"expression": 1}, {"expression": " "},
                          {"expression": "1" * 513}, {"expression": "1", "path": "app.py"}):
            with self.subTest(arguments=arguments):
                self.assertFalse(self.tools.execute("calculate", arguments)["ok"])
        self.assertEqual(self.tools.execute("calculate", {"expression": "1 / 0"})["error"]["code"],
                         "division_by_zero")
        self.assertEqual(self.tools.execute("calculate", {"expression": "open('app.py')"})["error"]["code"],
                         "invalid_expression")
        self.assertEqual((self.root / "app.py").read_text(), "print('Hello')\n# second line\n")

    def test_read_pagination_hash_and_literal_search(self):
        result = self.success("read_file", {"path": "app.py", "start_line": 2, "max_lines": 1})
        self.assertEqual(result["content"], "# second line\n")
        self.assertEqual(result["sha256"], hashlib.sha256((self.root / "app.py").read_bytes()).hexdigest())
        self.assertTrue(result["truncated"])
        result = self.success("search_files", {"query": "HELLO"})
        self.assertEqual(result["matches"][0]["path"], "app.py")
        self.assertEqual(result["matches"][0]["line"], 1)
        self.assertEqual(self.success("search_files", {"query": "HELLO", "case_sensitive": True})["matches"], [])

    def test_paths_symlinks_hardlinks_and_sensitive_paths_are_excluded(self):
        with tempfile.TemporaryDirectory() as outside:
            secret = Path(outside) / "secret.txt"
            secret.write_text("never return this", encoding="utf-8")
            (self.root / "link.txt").symlink_to(secret)
            (self.root / "linkdir").symlink_to(Path(outside), target_is_directory=True)
            os.link(secret, self.root / "hard.txt")
            for path in ("../secret.txt", str(secret), "link.txt", "linkdir/secret.txt", "hard.txt"):
                with self.subTest(path=path):
                    self.assertFalse(self.tools.execute("read_file", {"path": path})["ok"])
            for name in (".env", ".env.production", "AGENTS.md", "agents.md", "private.key", "credentials.json"):
                (self.root / name).write_text("never return this", encoding="utf-8")
                self.assertFalse(self.tools.execute("read_file", {"path": name})["ok"])
            (self.root / ".git").mkdir()
            (self.root / ".git" / "config").write_text("never return this", encoding="utf-8")
            entries = self.success("list_files", {})["entries"]
            self.assertEqual(entries, [{"path": "app.py", "type": "file"}])
            self.assertEqual(self.success("search_files", {"query": "never return"})["matches"], [])

    def test_binary_oversized_and_nonregular_reads_are_rejected(self):
        (self.root / "binary.bin").write_bytes(b"a\x00b")
        (self.root / "large.txt").write_bytes(b"a" * 262145)
        (self.root / "bad.txt").write_bytes(b"\xff")
        os.mkfifo(self.root / "pipe")
        for path in ("binary.bin", "large.txt", "bad.txt", "pipe"):
            with self.subTest(path=path):
                self.assertFalse(self.tools.execute("read_file", {"path": path})["ok"])

    def test_creation_and_replacement_require_current_hash_and_preserve_file_on_conflict(self):
        tools = ToolRegistry(self.root, allow_write=True)
        result = self.success("write_file", {"path": "new.txt", "content": "new\n"}, tools)
        self.assertTrue(result["created"])
        self.assertEqual((self.root / "new.txt").read_text(), "new\n")
        for arguments in ({"path": "new.txt", "content": "bad"},
                          {"path": "new.txt", "content": "bad", "expected_sha256": "0" * 64},
                          {"path": "missing.txt", "content": "bad", "expected_sha256": result["sha256"]}):
            self.assertEqual(tools.execute("write_file", arguments)["error"]["code"], "conflict")
        updated = self.success("write_file", {"path": "new.txt", "content": "updated",
                                               "expected_sha256": result["sha256"]}, tools)
        self.assertFalse(updated["created"])
        self.assertEqual((self.root / "new.txt").read_text(), "updated")
        self.assertEqual(list(self.root.glob(".silivra-write-*")), [])

    def test_concurrent_creator_is_not_overwritten(self):
        tools = ToolRegistry(self.root, allow_write=True)
        original_link = os.link

        def concurrent_link(source, destination, **kwargs):
            (self.root / destination).write_text("concurrent data", encoding="utf-8")
            return original_link(source, destination, **kwargs)

        with patch("silivra.tools.os.link", side_effect=concurrent_link):
            result = tools.execute("write_file", {"path": "race.txt", "content": "ours"})
        self.assertEqual(result["error"]["code"], "conflict")
        self.assertEqual((self.root / "race.txt").read_text(), "concurrent data")
        self.assertEqual(list(self.root.glob(".silivra-write-*")), [])

    def test_directory_swap_does_not_follow_symlink(self):
        (self.root / "nested").mkdir()
        (self.root / "nested" / "ok.txt").write_text("original", encoding="utf-8")
        with self.tools._directory(("nested",)) as descriptor:
            (self.root / "nested").rename(self.root / "moved")
            (self.root / "nested").symlink_to("/tmp", target_is_directory=True)
            fd = os.open("ok.txt", os.O_RDONLY | os.O_NOFOLLOW, dir_fd=descriptor)
            with os.fdopen(fd) as source:
                self.assertEqual(source.read(), "original")
        self.assertFalse(self.tools.execute("read_file", {"path": "nested/ok.txt"})["ok"])

    def test_output_list_limits_are_reported(self):
        (self.root / "other.txt").write_text("Hello\nHello\n", encoding="utf-8")
        self.assertTrue(self.success("list_files", {"limit": 1})["truncated"])
        self.assertTrue(self.success("search_files", {"query": "Hello", "limit": 1})["truncated"])

    def test_hidden_entries_exhausting_scan_budget_report_truncation(self):
        # Hidden entries still consume work; zero visible entries does not
        # imply that scanning completed or that later files do not exist.
        with patch("silivra.tools.os.scandir") as scan:
            scan.return_value.__enter__.side_effect = lambda: (
                SimpleNamespace(name=f".env.hidden-{number}") for number in range(10001))
            listed = self.success("list_files", {})
            searched = self.success("search_files", {"query": "Hello"})
        self.assertEqual(listed["entries"], [])
        self.assertTrue(listed["truncated"])
        self.assertEqual(searched["matches"], [])
        self.assertTrue(searched["truncated"])

    def test_command_is_argv_without_shell_and_does_not_inherit_secret_env(self):
        tools = ToolRegistry(self.root, allow_exec=True)
        with patch.dict(os.environ, {"SILIVRA_TEST_SECRET": "must-not-inherit"}):
            result = self.success("run_command", {"argv": [sys.executable, "-c",
                "import os,sys;print(sys.argv[1]);print(os.getenv('SILIVRA_TEST_SECRET'))", "$(touch nope) ; exit 99"]}, tools)
        self.assertEqual(result["exit_code"], 0)
        self.assertIn("$(touch nope) ; exit 99", result["output"])
        self.assertIn("None", result["output"])
        self.assertFalse(result["sandboxed"])
        self.assertFalse((self.root / "nope").exists())
        for arguments in ({"argv": "echo hello"}, {"argv": []},
                          {"argv": ["echo"], "timeout": float("nan")},
                          {"argv": ["echo"], "timeout": True}):
            self.assertFalse(tools.execute("run_command", arguments)["ok"])

    def test_command_timeout_and_output_budget_terminate_process(self):
        tools = ToolRegistry(self.root, allow_exec=True)
        started = time.monotonic()
        result = self.success("run_command", {"argv": [sys.executable, "-c", "import time;time.sleep(10)"], "timeout": 0.2}, tools)
        self.assertTrue(result["timed_out"])
        self.assertLess(time.monotonic() - started, 3)
        result = self.success("run_command", {"argv": [sys.executable, "-c", "print('x' * 1000000)"], "timeout": 2}, tools)
        self.assertTrue(result["output_truncated"])
        self.assertEqual(len(result["output"]), 32768)


if __name__ == "__main__":
    unittest.main()

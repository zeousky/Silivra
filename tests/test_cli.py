"""CLI integration checks for private memory and explicit tool permissions."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from silivra.cli import main
from silivra.agent import project_scope
from silivra.settings import TerminalSettings, load_settings, save_settings


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()

    def call(self, *arguments, code=0):
        output, error = io.StringIO(), io.StringIO()
        with patch("sys.argv", ["silivra", *arguments]), redirect_stdout(output), redirect_stderr(error):
            try:
                main()
            except SystemExit as exc:
                self.assertEqual(exc.code, code, error.getvalue())
            else:
                self.assertEqual(code, 0, output.getvalue())
        return json.loads(output.getvalue()) if output.getvalue().strip() else error.getvalue()

    def test_memory_correction_export_backup_and_forget(self):
        common = ["memory", "--workspace", str(self.root)]
        self.call(*common, "remember", "language", "Use Python.")
        self.call(*common, "remember", "language", "Use Rust.")
        versions = self.call(*common, "revisions", "language")
        self.assertEqual(len(versions), 2)
        self.assertEqual(self.call(*common, "recall", "Rust")[0]["content"], "Use Rust.")
        export = self.root / "private.json"
        backup = self.root / "backup.sqlite"
        self.call(*common, "export", "--output", str(export))
        self.assertEqual(os.stat(export).st_mode & 0o777, 0o600)
        result = self.call(*common, "backup", "--output", str(backup))
        self.assertTrue(result["includes_all_scopes"])
        self.call(*common, "forget", "language")
        self.assertEqual(self.call(*common, "list"), [])

    def test_scopes_and_permissions_at_cli_boundary(self):
        self.call("memory", "--workspace", str(self.root), "--user", "alice", "remember", "fact", "Alice uses Python.")
        self.assertEqual(self.call("memory", "--workspace", str(self.root), "--user", "bob", "list"), [])
        result = self.call("tool", "write_file", "--workspace", str(self.root),
                           "--arguments", '{"path":"demo.txt","content":"hello"}', code=1)
        self.assertEqual(result["error"]["code"], "permission_denied")
        self.assertFalse((self.root / "demo.txt").exists())
        self.call("tool", "write_file", "--workspace", str(self.root), "--allow-write",
                  "--arguments", '{"path":"demo.txt","content":"hello"}')
        self.assertEqual((self.root / "demo.txt").read_text(), "hello")
        result = self.call("tool", "web_search", "--workspace", str(self.root),
                           "--arguments", '{"query":"python"}', code=1)
        self.assertEqual(result["error"]["code"], "permission_denied")


    def test_one_shot_uses_saved_native_model_and_reservation(self):
        selected = self.root / "native-assistant"
        save_settings(self.root / ".silivra" / "config.json", project_scope(self.root),
                      TerminalSettings(model_run=str(selected)))
        backend = type("Backend", (), {"generate": lambda *_: '{"final":"Hello."}'})()
        with patch("silivra.terminal._backend", return_value=backend) as factory:
            result = self.call("chat", "--workspace", str(self.root), "--json", "Say hello")
            self.assertEqual(result["answer"], "Hello.")
            factory.assert_called_once_with(selected, which="best", max_new_tokens=96)
            factory.reset_mock()
            self.call("chat", "--workspace", str(self.root), "--json", "--max-new-tokens", "128", "Say hello")
            factory.assert_called_once_with(selected, which="best", max_new_tokens=128)


    def test_no_external_model_setup_command_exists(self):
        from silivra.cli import parser
        with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit) as raised:
            parser().parse_args(["setup-assistant"])
        self.assertEqual(raised.exception.code, 2)

    def test_one_shot_default_exposes_no_tools_and_honors_explicit_selection(self):
        class Backend:
            def generate(self, messages, tools):
                self.tools = tools
                return '{"final":"Done."}'
        for flags, expected in (([], set()), (["--no-tools"], set()),
                                (["--tool", "calculate"], {"calculate"}),
                                (["--tool", "read_file"], {"read_file"})):
            with self.subTest(flags=flags):
                backend = Backend()
                with patch("silivra.terminal._backend", return_value=backend):
                    result = self.call("chat", "--workspace", str(self.root), "--json", *flags, "Say done")
                self.assertEqual(result["status"], "completed")
                self.assertEqual({tool["function"]["name"] for tool in backend.tools}, expected)


if __name__ == "__main__":
    unittest.main()

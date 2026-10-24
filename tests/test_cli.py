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

    def test_setup_download_is_explicit_and_selects_separate_model(self):
        output = self.root / "runs" / "capable-assistant"
        settings_path = self.root / ".silivra" / "config.json"
        scope = project_scope(self.root)
        save_settings(settings_path, scope, TerminalSettings(model_run="runs/original", max_new_tokens=96,
                                                            history_messages=0, selected_tools=["read_file"]))
        with patch("silivra.provision.setup_pretrained", return_value={"model_id": "pinned/model"}) as provision:
            result = self.call("setup-assistant", "--workspace", str(self.root), "--output", "runs/capable-assistant")
        provision.assert_called_once_with(output, profile="qwen3-8b")
        self.assertTrue(result["selected"])
        self.assertEqual(result["selected_model"], str(output))
        selected = load_settings(settings_path, scope)
        self.assertEqual(selected.model_run, str(output))
        self.assertEqual(selected.max_new_tokens, 768)
        self.assertEqual(selected.history_messages, 0)
        self.assertEqual(selected.selected_tools, ["read_file"])

    def test_one_shot_uses_saved_model_and_backend_specific_reservation(self):
        selected = self.root / "separate-pretrained"
        save_settings(self.root / ".silivra" / "config.json", project_scope(self.root),
                      TerminalSettings(model_run=str(selected)))
        backend = type("Backend", (), {"generate": lambda *_: '{"final":"Hello."}'})()
        with patch("silivra.terminal._backend", return_value=backend) as factory, \
                patch("silivra.pretrained.is_pretrained_run", side_effect=lambda path: Path(path) == selected), \
                patch("silivra.provision.setup_pretrained", side_effect=AssertionError("chat must not download")):
            result = self.call("chat", "--workspace", str(self.root), "--json", "Say hello")
            self.assertEqual(result["answer"], "Hello.")
            factory.assert_called_once_with(selected, which="best", max_new_tokens=768)
            factory.reset_mock()
            self.call("chat", "--workspace", str(self.root), "--json", "--max-new-tokens", "128", "Say hello")
            factory.assert_called_once_with(selected, which="best", max_new_tokens=128)

    def test_setup_selects_calculator_from_empty_selection_and_preserves_all_tools(self):
        settings_path = self.root / ".silivra" / "config.json"
        scope = project_scope(self.root)
        for selected_tools, expected in (([], ["calculate"]), (None, None)):
            with self.subTest(selected_tools=selected_tools):
                save_settings(settings_path, scope, TerminalSettings(selected_tools=selected_tools))
                with patch("silivra.provision.setup_pretrained", return_value={"status": "verified"}):
                    result = self.call("setup-assistant", "--workspace", str(self.root))
                self.assertEqual(load_settings(settings_path, scope).selected_tools, expected)
                self.assertEqual(result["selected_tools"], expected)

    def test_setup_trial_uses_profile_directory_without_reading_or_writing_preferences(self):
        output = self.root / "runs" / "silivra-qwen3.5-9b"
        settings_path = self.root / ".silivra" / "config.json"
        for existing in (None, "invalid preferences that must stay untouched"):
            with self.subTest(existing=existing):
                if existing is not None:
                    settings_path.parent.mkdir(exist_ok=True)
                    settings_path.write_text(existing)
                with patch("silivra.provision.setup_pretrained", return_value={"status": "verified"}) as provision, \
                        patch("silivra.settings.load_settings", side_effect=AssertionError("trial must not read preferences")), \
                        patch("silivra.settings.save_settings", side_effect=AssertionError("trial must not save preferences")):
                    result = self.call("setup-assistant", "--workspace", str(self.root),
                                       "--profile", "qwen3.5-9b", "--no-select")
                provision.assert_called_once_with(output, profile="qwen3.5-9b")
                self.assertFalse(result["selected"])
                self.assertEqual(result["prepared_model"], str(output))
                self.assertNotIn("selected_model", result)
                if existing is None:
                    self.assertFalse(settings_path.parent.exists())
                else:
                    self.assertEqual(settings_path.read_text(), existing)

    def test_explicit_profile_selection_uses_its_separate_run(self):
        output = self.root / "runs" / "silivra-qwen3.5-9b"
        with patch("silivra.provision.setup_pretrained", return_value={"status": "verified"}) as provision:
            result = self.call("setup-assistant", "--workspace", str(self.root), "--profile", "qwen3.5-9b")
        provision.assert_called_once_with(output, profile="qwen3.5-9b")
        self.assertTrue(result["selected"])
        self.assertEqual(load_settings(self.root / ".silivra" / "config.json", project_scope(self.root)).model_run,
                         str(output))

    def test_one_shot_calculator_default_and_explicit_tool_flags(self):
        pretrained = self.root / "pretrained"
        original = self.root / "original"
        cases = ((pretrained, [], {"calculate"}),
                 (pretrained, ["--no-tools"], set()),
                 (pretrained, ["--tool", "read_file"], {"read_file"}),
                 (pretrained, ["--all-tools"], {"calculate", "list_files", "read_file", "search_files"}),
                 (original, [], set()),
                 (original, ["--tool", "calculate"], {"calculate"}))

        class Backend:
            def __init__(self):
                self.tools = []

            def generate(self, messages, tools):
                self.tools.append(tools)
                return '{"final":"Done."}'

        for run, flags, expected in cases:
            with self.subTest(run=run.name, flags=flags):
                backend = Backend()
                with patch("silivra.terminal._backend", return_value=backend), \
                        patch("silivra.pretrained.is_pretrained_run", side_effect=lambda path: Path(path) == pretrained):
                    result = self.call("chat", "--workspace", str(self.root), "--run", str(run),
                                       "--json", *flags, "Say done")
                self.assertEqual(result["status"], "completed")
                self.assertEqual({schema["function"]["name"] for schema in backend.tools[0]}, expected)


if __name__ == "__main__":
    unittest.main()

"""Terminal commands use real scoped memory and permission-enforcing file tools."""

import io
import json
import os
from pathlib import Path
import sqlite3
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from rich.console import Console
from prompt_toolkit.application import create_app_session
from prompt_toolkit.input import create_pipe_input
from prompt_toolkit.output import DummyOutput

from silivra.agent import AgentConfig, GenerationBudgetError, project_scope
from silivra.memory import MemoryStore
from silivra.settings import TerminalSettings, load_settings, save_settings
from silivra.terminal import ScopedInputHistory, TerminalApp, default_model_run, model_description, run_terminal


class ScriptedBackend:
    def __init__(self, *actions):
        self.actions = iter(actions or ('{"final":"Answer without animation."}',))
        self.calls = []
        self.model = SimpleNamespace(config=SimpleNamespace(max_seq_len=1024))
        self.checkpoint = "test-checkpoint"

    def generate(self, messages, tools):
        self.calls.append((messages, tools))
        action = next(self.actions)
        if isinstance(action, BaseException):
            raise action
        return action


class TerminalTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name).resolve()
        default = patch("silivra.terminal.default_model_run", return_value=self.workspace / "runs" / "original")
        default.start()
        self.addCleanup(default.stop)
        self.store = MemoryStore(self.workspace / ".silivra" / "memory.sqlite")
        self.addCleanup(self.store.close)
        self.output = io.StringIO()
        self.console = Console(file=self.output, width=120, color_system=None)
        self.factories = []
        self.backend = ScriptedBackend()
        self.app = self.make_app()

    def make_app(self, **kwargs):
        def factory(run, **options):
            self.factories.append((run, options))
            return self.backend
        return TerminalApp(self.workspace, self.store, console=self.console,
                           backend_factory=factory, **kwargs)

    def pretrained_marker(self, path, model="Qwen3-8B"):
        from silivra.pretrained import MARKER_NAME
        path.mkdir(parents=True, exist_ok=True)
        (path / MARKER_NAME).write_text(json.dumps({"original_model_id": f"Qwen/{model}",
                                                  "quantization_bits": 4, "context_window": 8192}))

    def test_help_status_and_memory_do_not_load_model(self):
        for text in ("/help", "/status", "/models", "/memory remember language We use Python.",
                     "/memory recall Python", "/memory revisions language"):
            self.assertTrue(self.app.handle_input(text))
        self.assertEqual(self.factories, [])
        self.assertIn("unknown until model loads", self.output.getvalue())
        self.assertEqual(self.store.list_memories(self.app.scope)[0]["content"], "We use Python.")
        self.app.handle_input("/memory forget language")
        self.assertEqual(self.store.list_memories(self.app.scope), [])

    def test_permission_grants_gate_direct_execution_and_never_persist(self):
        self.app.handle_input('/tool write_file {"path":"new.txt","content":"hello"}')
        self.assertFalse((self.workspace / "new.txt").exists())
        self.app.handle_input("/permissions write on")
        self.app.handle_input('/tool write_file {"path":"new.txt","content":"hello"}')
        self.assertEqual((self.workspace / "new.txt").read_text(), "hello")
        self.app.handle_input("/permissions read off")
        self.assertNotIn("read_file", [s["function"]["name"] for s in self.app._tools().schemas()])
        self.app.handle_input('/tool read_file {"path":"new.txt"}')
        self.assertIn("Workspace reading is disabled", self.output.getvalue())
        self.app.handle_input("/config max-new-tokens 64")
        restarted = self.make_app()
        self.assertFalse(restarted.permissions["write"])
        self.assertTrue(restarted.permissions["read"])
        self.assertEqual(restarted.settings.max_new_tokens, 64)

    def test_disabled_workspace_tools_are_not_exposed_but_calculator_remains(self):
        self.app.handle_input("/tools all")
        self.app.handle_input("/permissions read off")
        self.app.handle_input("What can you do?")
        self.assertEqual([schema["function"]["name"] for schema in self.backend.calls[0][1]], ["calculate"])
        self.assertEqual(len(self.factories), 1)
        self.assertEqual(self.output.getvalue().count("Answer without animation."), 1)
        self.app.status()
        self.assertIn("1024 tokens", self.output.getvalue())

    def test_calculator_selection_survives_read_permission_denial_and_none_persists(self):
        self.app.handle_input("/tools calculate")
        self.assertEqual(self.app.settings.selected_tools, ["calculate"])
        self.app.handle_input("/permissions read off")
        self.app.handle_input('/tool calculate {"expression":"37 * 14 - 59"}')
        self.assertIn('"459"', self.output.getvalue())
        self.app.handle_input("/tools none")
        pretrained = self.workspace / "pretrained"
        with patch("silivra.pretrained.is_pretrained_run", side_effect=lambda path: Path(path) == pretrained):
            restarted = self.make_app(run=pretrained)
            self.assertEqual(restarted.settings.selected_tools, [])

    def test_settings_model_switch_is_lazy_and_invalid_change_is_recoverable(self):
        model = self.workspace / "a model"
        model.mkdir()
        self.app.handle_input('/model "a model"')
        self.assertEqual(self.factories, [])
        self.app.handle_input("/model missing-model")
        self.assertEqual(self.app.settings.model_run, str(model.resolve()))
        self.app.handle_input("/config memory-chars 0")
        self.app.handle_input("/config memory-chars 12")
        self.assertEqual(self.app.settings.memory_chars, 0)
        self.app.handle_input("Hello")
        self.assertEqual(self.factories[0][0], model.resolve())
        self.assertEqual(self.make_app().settings.model_run, str(model.resolve()))

    def test_model_switch_resets_output_reservation_and_identifies_pretrained(self):
        pretrained = self.workspace / "runs" / "pretrained"
        original = self.workspace / "runs" / "original"
        pretrained.mkdir(parents=True)
        original.mkdir()
        self.pretrained_marker(pretrained)
        with patch("silivra.pretrained.is_pretrained_run", side_effect=lambda path: Path(path) == pretrained):
            self.app.handle_input("/config max-new-tokens 64")
            self.app.handle_input("/model runs/pretrained")
            self.assertEqual(self.app.settings.max_new_tokens, 768)
            self.app.handle_input("/status")
            self.assertIn("Qwen3-8B", self.output.getvalue())
            self.assertIn("pretrained", self.output.getvalue())
            self.assertEqual(self.factories, [])
            self.app.handle_input("/model runs/original")
            self.assertEqual(self.app.settings.max_new_tokens, 96)
            selected = self.make_app(run=pretrained)
            self.assertEqual(selected.settings.max_new_tokens, 768)
            selected = self.make_app(run=pretrained, max_new_tokens=128)
            self.assertEqual(selected.settings.max_new_tokens, 128)

    def test_new_pretrained_default_uses_longer_output_without_loading(self):
        pretrained = self.workspace / "runs" / "pretrained"
        self.pretrained_marker(pretrained)
        with patch("silivra.terminal.default_model_run", return_value=pretrained), \
                patch("silivra.pretrained.is_pretrained_run", side_effect=lambda path: Path(path) == pretrained):
            app = self.make_app()
            self.assertEqual(app.settings.max_new_tokens, 768)
            explicit = self.make_app(run=pretrained)
            self.assertEqual(explicit.settings.max_new_tokens, 768)
            app.banner()
        self.assertEqual(self.factories, [])
        self.assertIn("Qwen3-8B", self.output.getvalue())

    def test_pretrained_models_listing_reads_metadata_without_load(self):
        pretrained = self.workspace / "runs" / "pretrained"
        self.pretrained_marker(pretrained)
        with patch("silivra.pretrained.is_pretrained_run", side_effect=lambda path: Path(path) == pretrained):
            self.app.models()
        self.assertIn("Qwen3-8B", self.output.getvalue())
        self.assertIn("8192 runtime limit", self.output.getvalue())
        self.assertEqual(self.factories, [])

    def test_trial_model_name_is_displayed_without_changing_selection(self):
        trial = self.workspace / "runs" / "silivra-qwen3.5-9b"
        self.pretrained_marker(trial, "Qwen3.5-9B")
        selected = self.app.settings.model_run
        self.app.models()
        self.assertIn("Qwen3.5-9B", self.output.getvalue())
        self.assertEqual(self.app.settings.model_run, selected)
        explicit = self.make_app(run=trial)
        explicit.banner()
        self.assertIn("Qwen3.5-9B · pretrained", self.output.getvalue())
        self.assertEqual(self.factories, [])

    def test_saved_model_and_preferences_survive_a_new_fallback_default(self):
        preferred = self.workspace / "runs" / "silivra-qwen3.5-9b"
        self.pretrained_marker(preferred, "Qwen3.5-9B")
        for saved_model in ("runs/silivra-qwen3-8b", "runs/silivra-assistant-v2"):
            with self.subTest(saved_model=saved_model):
                saved = TerminalSettings(model_run=saved_model, max_new_tokens=128,
                                         selected_tools=[], history_messages=0)
                save_settings(self.app.settings_path, self.app.scope, saved)
                before = self.app.settings_path.read_bytes()
                with patch("silivra.terminal.default_model_run", return_value=preferred):
                    restarted = self.make_app()
                self.assertEqual(restarted._selected_run(), self.workspace / saved_model)
                self.assertEqual(restarted.settings.max_new_tokens, 128)
                self.assertEqual(restarted.settings.selected_tools, [])
                self.assertEqual(restarted.settings.history_messages, 0)
                self.assertEqual(self.app.settings_path.read_bytes(), before)
                self.assertEqual(self.factories, [])

    def test_explicit_startup_budget_overrides_preserve_other_preferences(self):
        self.app.handle_input("/config max-new-tokens 64")
        self.app.handle_input("/config history-messages 0")
        self.app.handle_input("/config memory-chars 0")
        restarted = self.make_app(config_overrides={"max_steps": 2})
        self.assertEqual(restarted.settings.max_new_tokens, 64)
        self.assertEqual(restarted.settings.history_messages, 0)
        self.assertEqual(restarted.settings.memory_chars, 0)
        self.assertEqual(restarted.settings.max_steps, 2)
        self.assertEqual(self.make_app().settings.max_steps, 6)

    def test_model_load_and_input_errors_leave_commands_available(self):
        self.app.handle_input("x" * 32001)
        self.assertEqual(self.factories, [])
        self.app.backend_factory = Mock(side_effect=RuntimeError("Cannot load checkpoint"))
        self.app.handle_input("Hello")
        self.assertIsNone(self.app.backend)
        self.assertIsNone(self.app.session_id)
        self.app.handle_input("/help")
        self.assertIn("Cannot load checkpoint", self.output.getvalue())
        self.app.backend_factory = lambda *_, **__: ScriptedBackend('{"final":"Recovered."}')
        self.app.handle_input("Hello again")
        self.assertIn("Recovered.", self.output.getvalue())

    def test_session_resume_cannot_cross_local_user_or_project_scope(self):
        first = self.store.create_session(self.app.scope)
        self.store.add_message(self.app.scope, first, "user", "Private Alice conversation")
        second = self.store.create_session(project_scope(self.workspace, "bob"))
        other_project = self.workspace / "other-project"
        other_project.mkdir()
        third = self.store.create_session(project_scope(other_project))
        self.app.handle_input("/resume " + first)
        self.app.handle_input("/resume " + second)
        self.app.handle_input("/resume " + third)
        self.assertEqual(self.app.session_id, first)
        self.app.handle_input("/history")
        self.assertIn("Private Alice conversation", self.output.getvalue())
        self.app.handle_input("/new")
        self.assertIsNone(self.app.session_id)
        self.app.handle_input("/resume " + first)
        self.app.handle_input("/memory delete-session " + first)
        self.assertIsNone(self.app.session_id)
        self.assertIn("Session does not exist in this scope", self.output.getvalue())

    def test_invalid_commands_and_json_cannot_execute_tools(self):
        self.app.handle_input("/permissions write on")
        for text in ('/tool write_file []', '/tool write_file {"path":"x","path":"y","content":"z"}',
                     '/tool write_file {"path":"x","content":NaN}', '/tool write_file nope',
                     '/permissions exec maybe', '/permissions network', '/unknown', '/config max-steps 0'):
            with self.subTest(text=text):
                self.assertTrue(self.app.handle_input(text))
        self.assertFalse((self.workspace / "x").exists())
        self.assertFalse((self.workspace / "y").exists())
        self.assertFalse(self.app.permissions["exec"])
        self.assertFalse(self.app.permissions["network"])
        self.assertGreaterEqual(self.output.getvalue().count("Error:"), 8)

    def test_unknown_startup_tools_and_non_boolean_grants_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "Unknown tools"):
            self.make_app(tool_names=["invented_tool"])
        with self.assertRaisesRegex(ValueError, "explicit booleans"):
            self.make_app(allow_exec="on")
        self.assertEqual(self.factories, [])

    def test_model_slash_text_is_rendered_and_never_dispatched(self):
        self.backend = ScriptedBackend('{"final":"/permissions exec on"}')
        self.app.handle_input("Say an example command")
        self.assertFalse(self.app.permissions["exec"])
        self.assertIn("/permissions exec on", self.output.getvalue())

    def test_cancellation_keeps_scope_and_session(self):
        self.backend = ScriptedBackend(KeyboardInterrupt())
        self.app.handle_input("Please think")
        self.assertIsNotNone(self.app.session_id)
        self.assertEqual(len(self.store.list_sessions(self.app.scope)), 1)
        self.assertIn("cancelled", self.output.getvalue())
        self.app.backend = ScriptedBackend('{"final":"Recovered."}')
        self.app.handle_input("Continue")
        self.assertIn("Recovered.", self.output.getvalue())

    def test_interrupted_draft_is_explicitly_unvalidated(self):
        class InterruptedBackend:
            def generate(self, messages, tools, *, on_text):
                on_text('{"final":"Unfinished draft')
                raise KeyboardInterrupt()
        self.backend = InterruptedBackend()
        self.app.handle_input("Begin")
        self.assertIn("Draft interrupted; no validated final answer", self.output.getvalue())
        self.assertNotIn("Draft accepted", self.output.getvalue())

    def test_output_budget_rejects_draft_and_explains_token_context_tradeoff(self):
        class ExhaustedBackend:
            def generate(self, messages, tools, *, on_text):
                on_text('{"final":"Long unfinished answer')
                raise GenerationBudgetError("Output token limit reached before the action completed.")

        self.backend = ExhaustedBackend()
        result = self.app.run_turn("Write a detailed answer")
        self.assertEqual(result["status"], "output_budget")
        self.assertIsNone(result["answer"])
        output = self.output.getvalue()
        self.assertIn("Long unfinished answer", output)
        self.assertIn("Draft rejected or incomplete; no validated final answer", output)
        self.assertIn("Stopped · output_budget", output)
        self.assertIn("/config max-new-tokens N", output)
        self.assertIn("leaves less context", output)
        self.assertNotIn("Draft accepted", output)
        stored = self.store.recent_messages(self.app.scope, self.app.session_id)
        self.assertEqual(stored[-1]["metadata"]["turn_status"], "output_budget")

    def test_tool_results_display_and_transcript_are_scoped(self):
        (self.workspace / "demo.py").write_text("print('hello')\n")
        self.backend = ScriptedBackend('{"tool":"read_file","arguments":{"path":"demo.py"}}',
                                       '{"final":"Read successfully."}')
        self.app.handle_input("/tools read_file")
        self.app.handle_input("Inspect demo.py")
        self.assertIn("print('hello')", self.output.getvalue())
        messages = self.store.recent_messages(self.app.scope, self.app.session_id)
        self.assertEqual([item["role"] for item in messages], ["user", "assistant", "tool", "assistant"])

    def test_exports_are_exclusive_and_backup_uses_consistent_store_api(self):
        self.app.handle_input("/memory remember language Python")
        self.app.handle_input("/memory export .silivra/export.json")
        exported = self.workspace / ".silivra" / "export.json"
        self.assertEqual(json.loads(exported.read_text())["scope"], self.app.scope)
        original = exported.read_text()
        self.app.handle_input("/memory export .silivra/export.json")
        self.assertEqual(exported.read_text(), original)
        self.app.handle_input("/memory backup .silivra/backup.sqlite")
        with sqlite3.connect(self.workspace / ".silivra" / "backup.sqlite") as backup:
            self.assertEqual(backup.execute("PRAGMA integrity_check").fetchone()[0], "ok")

    def test_multiline_input_is_preserved_and_terminal_controls_are_inert(self):
        self.backend = ScriptedBackend('{"final":"\\u001b[2Janswer\\u0007"}')
        self.app.handle_input("First line\nSecond line")
        messages = self.store.recent_messages(self.app.scope, self.app.session_id)
        self.assertEqual(messages[0]["content"], "First line\nSecond line")
        self.assertNotIn("\x1b", self.output.getvalue())
        self.assertNotIn("\x07", self.output.getvalue())

    def test_bad_settings_do_not_prevent_help_and_are_never_overwritten(self):
        settings = self.workspace / ".silivra" / "config.json"
        settings.write_text("bad json")
        app = self.make_app()
        app.handle_input("/help")
        app.handle_input("/config max-new-tokens 32")
        self.assertEqual(settings.read_text(), "bad json")
        self.assertEqual(app.settings.max_new_tokens, 96)

    def test_input_loop_recovers_ctrl_c_and_exits_on_eof_without_loading(self):
        prompt = SimpleNamespace(prompt=Mock(side_effect=[KeyboardInterrupt(), "/help", EOFError()]))
        code = run_terminal(workspace=self.workspace, console=self.console, prompt_session=prompt,
                            backend_factory=lambda *_, **__: self.fail("model must stay unloaded"))
        self.assertEqual(code, 0)
        self.assertIn("Input cancelled", self.output.getvalue())

    def test_terminal_refuses_linked_database_and_sidecars_before_opening(self):
        outside = self.workspace / "outside.sqlite"
        outside.write_bytes(b"preserve this file")
        for suffix in ("", "-wal", "-shm", "-journal"):
            for kind in ("symlink", "hardlink"):
                database = self.workspace / ".silivra" / "another.sqlite"
                candidate = Path(str(database) + suffix)
                if kind == "symlink":
                    candidate.symlink_to(outside)
                else:
                    os.link(outside, candidate)
                try:
                    with self.subTest(suffix=suffix, kind=kind), self.assertRaisesRegex(ValueError, "links"):
                        run_terminal(workspace=self.workspace, memory=database, console=self.console)
                    self.assertEqual(outside.read_bytes(), b"preserve this file")
                finally:
                    candidate.unlink()
                    database.unlink(missing_ok=True)

    def test_provisional_event_is_labelled_and_never_faked_from_completed_text(self):
        self.app._event({"type": "final_text", "text": "Real delta ", "provisional": True})
        self.app._event({"type": "final_text", "text": "two", "provisional": True})
        self.assertIn("draft (awaiting protocol validation)", self.output.getvalue())
        self.assertIn("Real delta two", self.output.getvalue())


class ModelSelectionTests(unittest.TestCase):
    def test_only_provisioned_pretrained_run_takes_priority(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            workspace, source = root / "workspace", root / "source"
            original = source / "runs" / "silivra-assistant-v2"
            workspace_original = workspace / "runs" / "silivra-assistant-v2"
            source_pretrained = source / "runs" / "silivra-qwen3-8b"
            workspace_pretrained = workspace / "runs" / "silivra-qwen3-8b"
            source_preferred = source / "runs" / "silivra-qwen3.5-9b"
            workspace_preferred = workspace / "runs" / "silivra-qwen3.5-9b"
            for path in (original, workspace_original, source_pretrained, workspace_pretrained,
                         source_preferred, workspace_preferred):
                path.mkdir(parents=True)
            provisioned = set()
            with patch("silivra.terminal.__file__", str(source / "src" / "silivra" / "terminal.py")), \
                    patch("silivra.pretrained.is_pretrained_run", side_effect=lambda path: path in provisioned):
                self.assertEqual(default_model_run(workspace), original)
                provisioned.add(source_pretrained)
                self.assertEqual(default_model_run(workspace), source_pretrained)
                provisioned.add(workspace_pretrained)
                self.assertEqual(default_model_run(workspace), workspace_pretrained)
                provisioned.add(source_preferred)
                self.assertEqual(default_model_run(workspace), source_preferred)
                provisioned.add(workspace_preferred)
                self.assertEqual(default_model_run(workspace), workspace_preferred)
                provisioned.clear()
                original.rmdir()
                self.assertEqual(default_model_run(workspace), workspace_original)

    def test_pretrained_display_metadata_is_bounded_and_untrusted(self):
        from silivra.pretrained import MARKER_NAME
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            marker = root / MARKER_NAME
            for text in ('not json', '[]', '{"original_model_id":"Qwen/Qwen3.5-9B","original_model_id":"bad"}',
                         ' ' * 131073, '{"original_model_id":"Qwen/\\u001b[2J"}'):
                marker.write_text(text)
                with self.subTest(text=text[:40]):
                    description = model_description(root)
                    self.assertIn("External model", description)
                    self.assertNotIn("Qwen3-8B", description)
                    self.assertNotIn("\x1b", description)
            marker.unlink()
            outside = root / "outside.json"
            outside.write_text('{"original_model_id":"Qwen/Qwen3.5-9B"}')
            marker.symlink_to(outside)
            self.assertIn("External model", model_description(root))

    def test_backend_factory_selects_only_explicit_pretrained_marker(self):
        from silivra.terminal import _backend
        run = Path("local-checkpoint")
        with patch("silivra.pretrained.is_pretrained_run", return_value=True), \
                patch("silivra.pretrained.PretrainedMLXBackend") as pretrained, \
                patch("silivra.agent.MLXBackend") as original:
            self.assertIs(_backend(run, which="best", max_new_tokens=768), pretrained.return_value)
            pretrained.assert_called_once_with(run, which="best", max_new_tokens=768)
            original.assert_not_called()
        with patch("silivra.pretrained.is_pretrained_run", return_value=False), \
                patch("silivra.pretrained.PretrainedMLXBackend") as pretrained, \
                patch("silivra.agent.MLXBackend") as original:
            self.assertIs(_backend(run, which="latest", max_new_tokens=96), original.return_value)
            original.assert_called_once_with(run, which="latest", max_new_tokens=96)
            pretrained.assert_not_called()


class InputHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.scope = project_scope(self.workspace, "alice")
        self.history = ScopedInputHistory(self.workspace, self.scope)

    def test_history_is_private_redacted_and_isolated(self):
        self.history.store_string("First line\nSecond line")
        self.history.store_string("password=very-private-password")
        self.assertEqual(list(self.history.load_history_strings())[1], "First line\nSecond line")
        self.assertNotIn("very-private-password", self.history.path.read_text())
        self.assertEqual(os.stat(self.history.path).st_mode & 0o777, 0o600)
        other = ScopedInputHistory(self.workspace, project_scope(self.workspace, "bob"))
        self.assertEqual(list(other.load_history_strings()), [])

    def test_large_unicode_history_is_bounded_and_does_not_split_utf8(self):
        old_record = json.dumps("é" * 150000, ensure_ascii=False).encode() + b"\n"
        self.history.path.write_bytes(old_record + b'"latest"\n')
        self.assertEqual(list(self.history.load_history_strings()), ["latest"])
        self.history.store_string("newest")
        self.assertLessEqual(self.history.path.stat().st_size, self.history.max_bytes)
        self.assertEqual(list(self.history.load_history_strings()), ["newest", "latest"])
        self.assertEqual(list(self.history.path.parent.glob(".input-*")), [])

    def test_linked_history_is_never_read_or_modified(self):
        outside = self.workspace / "outside"
        outside.write_text('"private outside data"\n')
        for kind in ("symlink", "hardlink"):
            if kind == "symlink":
                self.history.path.symlink_to(outside)
            else:
                os.link(outside, self.history.path)
            try:
                with self.subTest(kind=kind), self.assertRaisesRegex(ValueError, "links|symlink"):
                    self.history.store_string("new entry")
                self.assertEqual(outside.read_text(), '"private outside data"\n')
            finally:
                self.history.path.unlink()


class TerminalCLIIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)

    def test_chat_only_explicit_flags_override_saved_terminal_preferences(self):
        from silivra.cli import main
        settings_path = self.workspace / ".silivra" / "config.json"
        scope = project_scope(self.workspace)
        save_settings(settings_path, scope, TerminalSettings(max_new_tokens=48, history_messages=0, memory_chars=0))
        observed = []

        def prompt_factory(app):
            observed.append(app.settings)
            return SimpleNamespace(prompt=Mock(return_value="/exit"))

        for flags in ([], ["--max-steps", "2", "--max-new-tokens", "32"]):
            with patch("sys.argv", ["silivra", "chat", "--workspace", str(self.workspace), *flags]), \
                    patch("sys.stdin.isatty", return_value=True), \
                    patch("silivra.terminal._prompt_session", side_effect=prompt_factory), \
                    patch("sys.stdout", io.StringIO()), self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 0)
        self.assertEqual(observed[0].max_new_tokens, 48)
        self.assertEqual(observed[1].max_new_tokens, 32)
        self.assertEqual(observed[1].max_steps, 2)
        self.assertEqual(observed[1].history_messages, 0)
        self.assertEqual(observed[1].memory_chars, 0)
        self.assertEqual(load_settings(settings_path, scope).max_new_tokens, 48)

    def test_invalid_piped_prompts_fail_before_loading_model(self):
        from silivra.cli import main
        for text in ("", "x" * 32001):
            with self.subTest(length=len(text)), \
                    patch("sys.argv", ["silivra", "chat", "--workspace", str(self.workspace)]), \
                    patch("sys.stdin", io.StringIO(text)), patch("sys.stderr", io.StringIO()), \
                    patch("silivra.agent.MLXBackend", side_effect=AssertionError("must not load")), \
                    self.assertRaises(SystemExit) as stopped:
                main()
            self.assertEqual(stopped.exception.code, 1)


class PromptToolkitIntegrationTests(unittest.TestCase):
    """Exercise the pinned prompt API and real key bindings without a TTY/MLX."""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.workspace = Path(self.temp.name)
        self.output = io.StringIO()
        self.console = Console(file=self.output, width=120, color_system=None)

    def terminal(self, keys):
        # A physical Enter is CR. LF is Ctrl+J, explicitly bound to insert a
        # newline in Silivra's multiline editor rather than send a submission.
        with create_pipe_input() as input_pipe, \
                create_app_session(input=input_pipe, output=DummyOutput()):
            input_pipe.send_text(keys)
            return run_terminal(workspace=self.workspace, console=self.console,
                                backend_factory=lambda *_, **__: self.fail("must not load MLX"))

    def test_default_prompt_session_opens_help_and_exits(self):
        self.assertEqual(self.terminal("/help\r/exit\r"), 0)
        self.assertIn("Show commands and input controls", self.output.getvalue())
        history = ScopedInputHistory(self.workspace, project_scope(self.workspace))
        self.assertEqual(list(history.load_history_strings())[:2], ["/exit", "/help"])

    def test_real_ctrl_j_and_alt_enter_preserve_multiline_submission(self):
        keys = "/memory remember note First line\nSecond line\x1b\rThird line\r/exit\r"
        self.assertEqual(self.terminal(keys), 0)
        with MemoryStore(self.workspace / ".silivra" / "memory.sqlite") as store:
            saved = store.list_memories(project_scope(self.workspace))
        self.assertEqual(saved[0]["content"], "First line\nSecond line\nThird line")

    def test_real_ctrl_c_cancels_input_and_keeps_terminal_open(self):
        self.assertEqual(self.terminal("discard this input\x03/help\r/exit\r"), 0)
        self.assertIn("Input cancelled", self.output.getvalue())
        self.assertIn("Show commands and input controls", self.output.getvalue())
        history = ScopedInputHistory(self.workspace, project_scope(self.workspace))
        self.assertNotIn("discard this input", list(history.load_history_strings()))


if __name__ == "__main__":
    unittest.main()

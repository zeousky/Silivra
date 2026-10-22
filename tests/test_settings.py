"""Private terminal preferences persist without durable capability grants."""

import json
import os
from pathlib import Path
import tempfile
import unittest

from silivra.settings import TerminalSettings, load_settings, save_settings


class SettingsTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.path = Path(self.temp.name) / ".silivra" / "config.json"

    def test_roundtrip_is_scoped_private_and_excludes_permissions(self):
        settings = TerminalSettings(model_run="runs/assistant", selected_tools=["read_file"], memory_chars=0)
        save_settings(self.path, "alice-project", settings)
        save_settings(self.path, "bob-project", TerminalSettings(max_new_tokens=64))
        self.assertEqual(load_settings(self.path, "alice-project"), settings)
        self.assertEqual(load_settings(self.path, "bob-project").max_new_tokens, 64)
        self.assertEqual(load_settings(self.path, "other"), TerminalSettings())
        self.assertEqual(os.stat(self.path).st_mode & 0o777, 0o600)
        self.assertEqual(os.stat(self.path.parent).st_mode & 0o777, 0o700)
        self.assertNotIn("permissions", self.path.read_text())
        self.assertEqual(list(self.path.parent.glob(".settings-*")), [])

    def test_malformed_and_future_files_are_preserved_on_save(self):
        self.path.parent.mkdir()
        for text in ("invalid json", '{"version":999,"scopes":{}}',
                     '{"version":true,"scopes":{}}', '{"version":1,"scopes":[],"extra":1}'):
            self.path.write_text(text)
            with self.subTest(text=text):
                with self.assertRaises(ValueError):
                    load_settings(self.path, "scope")
                with self.assertRaises(ValueError):
                    save_settings(self.path, "scope", TerminalSettings())
                self.assertEqual(self.path.read_text(), text)

    def test_unknown_fields_and_invalid_budgets_are_rejected(self):
        self.path.parent.mkdir()
        self.path.write_text(json.dumps({"version": 1, "scopes": {"scope": {"allow_exec": True}}}))
        original = self.path.read_text()
        with self.assertRaisesRegex(ValueError, "Unknown terminal"):
            load_settings(self.path, "scope")
        with self.assertRaisesRegex(ValueError, "Unknown terminal"):
            save_settings(self.path, "scope", TerminalSettings())
        self.assertEqual(self.path.read_text(), original)
        for values in ({"max_new_tokens": True}, {"max_new_tokens": 0},
                       {"max_seconds": float("nan")}, {"memory_chars": 12},
                       {"selected_tools": ["read_file", "read_file"]}):
            with self.subTest(values=values), self.assertRaises(ValueError):
                TerminalSettings(**values)

    def test_state_symlinks_are_refused(self):
        target = Path(self.temp.name) / "outside"
        target.mkdir()
        self.path.parent.symlink_to(target, target_is_directory=True)
        with self.assertRaisesRegex(ValueError, "symlink"):
            save_settings(self.path, "scope", TerminalSettings())
        with self.assertRaisesRegex(ValueError, "symlink"):
            load_settings(self.path, "scope")
        self.assertEqual(list(target.iterdir()), [])
        self.path.parent.unlink()
        self.path.parent.mkdir()
        outside = target / "config.json"
        outside.write_text('{"version":1,"scopes":{}}')
        self.path.symlink_to(outside)
        with self.assertRaisesRegex(ValueError, "symlink"):
            load_settings(self.path, "scope")

    def test_hardlinked_and_special_files_are_refused_without_modification(self):
        self.path.parent.mkdir()
        outside = Path(self.temp.name) / "outside.json"
        original = '{"version":1,"scopes":{}}'
        outside.write_text(original)
        os.link(outside, self.path)
        with self.assertRaisesRegex(ValueError, "hardlinks"):
            load_settings(self.path, "scope")
        with self.assertRaisesRegex(ValueError, "hardlinks"):
            save_settings(self.path, "scope", TerminalSettings())
        self.assertEqual(outside.read_text(), original)
        self.path.unlink()
        os.mkfifo(self.path)
        with self.assertRaisesRegex(ValueError, "regular file"):
            load_settings(self.path, "scope")

    def test_existing_regular_settings_become_private_when_loaded(self):
        save_settings(self.path, "scope", TerminalSettings())
        self.path.chmod(0o644)
        load_settings(self.path, "scope")
        self.assertEqual(self.path.stat().st_mode & 0o777, 0o600)

    def test_scope_and_file_size_limits_fail_without_overwriting(self):
        for scope in (None, "", "x" * 513):
            with self.subTest(scope=scope), self.assertRaises(ValueError):
                load_settings(self.path, scope)
        self.path.parent.mkdir()
        text = " " * 131073
        self.path.write_text(text)
        with self.assertRaisesRegex(ValueError, "too large"):
            save_settings(self.path, "scope", TerminalSettings())
        self.assertEqual(self.path.read_text(), text)


if __name__ == "__main__":
    unittest.main()

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


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

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


if __name__ == "__main__":
    unittest.main()

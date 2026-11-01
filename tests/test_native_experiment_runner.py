import fcntl
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import unittest


SPEC = importlib.util.spec_from_file_location("native_experiment_runner",
    Path(__file__).resolve().parents[1] / "scripts" / "run-native-experiment.py")
runner = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(runner)


class RunnerTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.journal = self.root / "private" / "budget.json"
        self.lock = self.root / "global.lock"

    def run_child(self, code="pass", **kwargs):
        return runner.supervise(self.journal, kwargs.pop("budget", 5), kwargs.pop("seconds", 1),
            [sys.executable, "-c", code], grace_seconds=kwargs.pop("grace", .1),
            lock_path=self.lock, cwd=self.root, **kwargs)

    def ledger(self):
        return json.loads(self.journal.read_text())

    def test_completed_child_charges_actual_wall_and_sequential_total(self):
        first = self.run_child("import time; time.sleep(.03)")
        second = self.run_child()
        self.assertEqual(first["status"], "completed")
        self.assertLess(first["session_seconds"], 1)
        self.assertAlmostEqual(second["charged_seconds"],
            first["session_seconds"] + second["session_seconds"])
        self.assertEqual(len(self.ledger()["attempts"]), 2)
        self.assertEqual(os.stat(self.journal).st_mode & 0o777, 0o600)

    def test_deadline_allows_our_child_to_checkpoint(self):
        marker = self.root / "checkpoint"
        code = ("import signal,time,pathlib,sys; "
            f"signal.signal(signal.SIGTERM, lambda *_: (pathlib.Path({str(marker)!r}).write_text('saved'), sys.exit(0))); "
            "time.sleep(20)")
        result = self.run_child(code, seconds=1, grace=.2)
        self.assertEqual(result["status"], "deadline")
        self.assertEqual(result["returncode"], 0)
        self.assertEqual(marker.read_text(), "saved")
        self.assertLess(result["session_seconds"], 1)

    def test_uncooperative_child_is_killed_without_touching_unrelated_process(self):
        unrelated = subprocess.Popen([sys.executable, "-c", "import time;time.sleep(20)"])
        self.addCleanup(lambda: (unrelated.terminate(), unrelated.wait()) if unrelated.poll() is None else None)
        result = self.run_child("import signal,time;signal.signal(signal.SIGTERM,signal.SIG_IGN);time.sleep(20)",
                                seconds=.8, grace=.15)
        self.assertEqual(result["returncode"], -signal.SIGKILL)
        self.assertIsNone(unrelated.poll())
        self.assertLess(result["session_seconds"], 1)

    def test_missing_child_binary_returns_stable_failure_and_charges_consumed_time(self):
        result = runner.supervise(self.journal, 5, 1,
            [str(self.root / "missing-binary")], grace_seconds=.1,
            lock_path=self.lock, cwd=self.root)
        self.assertEqual(result["status"], "launch_failed")
        self.assertEqual(result["returncode"], 127)
        self.assertGreaterEqual(result["session_seconds"], 0)
        self.assertLess(result["session_seconds"], 1)
        self.assertEqual(self.ledger()["attempts"][0]["error_type"], "FileNotFoundError")
        self.assertEqual(self.run_child()["status"], "completed")

    def test_session_cannot_exceed_remaining_or_enlarge_original_budget(self):
        self.run_child(budget=1, seconds=.5)
        with self.assertRaisesRegex(ValueError, "remaining"):
            self.run_child(budget=1, seconds=1)
        with self.assertRaisesRegex(ValueError, "original budget"):
            self.run_child(budget=2)
        with self.assertRaises(ValueError):
            self.run_child(budget=7201)

    def test_crashed_supervisor_keeps_full_reservation_charged(self):
        self.journal.parent.mkdir()
        # A nonexistent PID means the previous child has definitely exited.
        state = {"format_version": 1, "budget_seconds": 5, "attempts": [
            {"status": "running", "charged_seconds": 4, "child_pid": 2147483647}]}
        runner.durable_write(self.journal, state)
        result = self.run_child(seconds=.5)
        self.assertGreaterEqual(result["charged_seconds"], 4)
        self.assertEqual(self.ledger()["attempts"][0]["status"], "abandoned_reservation_charged")
        with self.assertRaisesRegex(ValueError, "remaining"):
            self.run_child(seconds=1.1)

    def test_live_or_unknown_previous_child_refuses_resume(self):
        self.journal.parent.mkdir()
        for pid in (os.getpid(), None):
            runner.durable_write(self.journal, {"format_version": 1, "budget_seconds": 5,
                "attempts": [{"status": "running", "charged_seconds": 1, "child_pid": pid}]})
            with self.assertRaises(RuntimeError):
                self.run_child()

    def test_global_lock_refuses_parallel_supervisors_and_child_failure_is_recorded(self):
        with self.lock.open("a") as stream:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            with self.assertRaisesRegex(RuntimeError, "Another supervised"):
                self.run_child()
        result = self.run_child("raise SystemExit(7)")
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["returncode"], 7)

    def test_reservation_is_durable_before_launch_and_child_argv_has_no_shell(self):
        code = ("import json,pathlib; "
            f"state=json.loads(pathlib.Path({str(self.journal)!r}).read_text()); "
            "assert state['attempts'][-1]['charged_seconds']==1; "
            "assert state['attempts'][-1]['status']=='running'")
        self.assertEqual(self.run_child(code)["returncode"], 0)
        self.assertEqual(self.ledger()["attempts"][-1]["argv"], [sys.executable, "-c", code])
        with self.assertRaisesRegex(ValueError, "explicit argument list"):
            runner.supervise(self.journal, 5, 1, "echo unsafe", grace_seconds=.1,
                             lock_path=self.lock, cwd=self.root)

    def test_child_remains_in_its_supervisors_owned_process_group(self):
        marker = self.root / "child-process-group.json"
        code = ("import json,os,pathlib; "
            f"pathlib.Path({str(marker)!r}).write_text(json.dumps({{"
            "'parent':os.getppid(),'group':os.getpgrp()}))")
        self.assertEqual(self.run_child(code)["returncode"], 0)
        identity = json.loads(marker.read_text())
        self.assertEqual(identity["parent"], os.getpid())
        self.assertEqual(identity["group"], os.getpgrp())


if __name__ == "__main__":
    unittest.main()

"""Supervise one native training session against a shared private wall-time ledger.

All child time, including setup/evaluation/checkpoint grace, is charged. A crashed
supervisor leaves its whole reservation charged; never infer unsaved training time
from a checkpoint. Invoke sequentially with the same journal for pilots and final.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]


def durable_write(path, value):
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w") as stream:
            json.dump(value, stream, indent=2, sort_keys=True)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        descriptor = os.open(path.parent, os.O_RDONLY)
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def load_ledger(path, budget_seconds):
    ledger = json.loads(path.read_text()) if path.exists() else {
        "format_version": 1, "budget_seconds": budget_seconds, "attempts": []}
    if ledger.get("format_version") != 1 or ledger.get("budget_seconds") != budget_seconds:
        raise ValueError("Use the original journal and its original budget")
    for attempt in ledger["attempts"]:
        charged = attempt["charged_seconds"]
        if type(charged) not in (int, float) or not math.isfinite(charged) or charged < 0:
            raise ValueError("Invalid journal accounting")
        if attempt["status"] == "running":
            pid = attempt.get("child_pid")
            if not pid:
                raise RuntimeError("Unreconciled reservation: inspect previous supervisor before continuing")
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                attempt["status"] = "abandoned_reservation_charged"
            else:
                raise RuntimeError("Previous child may still be alive; refusing concurrent training")
    if sum(item["charged_seconds"] for item in ledger["attempts"]) > budget_seconds:
        raise ValueError("Journal exceeds its authorized budget")
    return ledger


def supervise(journal, budget_seconds, seconds, command, *, grace_seconds=30,
              caffeinate=False, lock_path=None, cwd=ROOT):
    for name, value in (("budget", budget_seconds), ("session", seconds), ("grace", grace_seconds)):
        if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
            raise ValueError(f"{name} seconds must be finite and positive")
    if budget_seconds > 7200 or seconds <= grace_seconds:
        raise ValueError("Budget must be at most 7200 seconds and session must exceed checkpoint grace")
    journal = Path(journal).resolve()
    journal.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    lock_path = Path(lock_path or ROOT / ".silivra" / ".native-trainer.lock")
    lock_path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    with lock_path.open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("Another supervised trainer is active") from None
        ledger = load_ledger(journal, budget_seconds)
        charged_before = sum(item["charged_seconds"] for item in ledger["attempts"])
        if seconds > budget_seconds - charged_before:
            raise ValueError("Requested session exceeds remaining shared budget")
        identifier = uuid.uuid4().hex
        logs = journal.parent / "training-logs"
        logs.mkdir(exist_ok=True, mode=0o700)
        logfile = logs / f"{identifier}.log"
        argv = command(seconds - grace_seconds) if callable(command) else command
        if not isinstance(argv, (list, tuple)) or not argv or any(not isinstance(item, str) or not item for item in argv):
            raise ValueError("Child command must be an explicit argument list")
        attempt = {"id": identifier, "status": "running", "reserved_seconds": seconds,
                   "charged_seconds": seconds, "argv": argv, "log": str(logfile),
                   "started_at": datetime.now(timezone.utc).isoformat()}
        ledger["attempts"].append(attempt)
        durable_write(journal, ledger)  # Reserve BEFORE a child can consume time.
        child = inhibitor = None
        stopped = False
        handlers = {}
        def request_stop(*_):
            nonlocal stopped
            stopped = True
        started = time.monotonic()
        try:
            for sig in (signal.SIGINT, signal.SIGTERM):
                handlers[sig] = signal.signal(sig, request_stop)
            with logfile.open("xb") as log:
                os.chmod(logfile, 0o600)
                # Inherit the supervisor's group. A campaign may launch this
                # supervisor in its own session and then stop that entire owned
                # group if the supervisor itself becomes unresponsive.
                child = subprocess.Popen(argv, cwd=cwd, stdin=subprocess.DEVNULL,
                                         stdout=log, stderr=subprocess.STDOUT, start_new_session=False)
                attempt["child_pid"] = child.pid
                durable_write(journal, ledger)
                if caffeinate and sys.platform == "darwin" and shutil.which("caffeinate"):
                    inhibitor = subprocess.Popen(["caffeinate", "-i", "-w", str(child.pid)],
                                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                cutoff = started + seconds - min(.25, seconds * .05)
                termination_sent = False
                while child.poll() is None:
                    now = time.monotonic()
                    if not termination_sent and (stopped or now >= cutoff - grace_seconds):
                        child.terminate()  # Only our child; trainer checkpoints cooperatively.
                        termination_sent = True
                        cutoff = min(cutoff, now + grace_seconds)
                    if now >= cutoff:
                        child.kill()
                        break
                    time.sleep(min(.05, max(.001, cutoff - now)))
                child.wait()
                attempt["returncode"] = child.returncode
                attempt["status"] = ("interrupted" if stopped else "deadline") if termination_sent else (
                    "completed" if child.returncode == 0 else "failed")
        except OSError as error:
            if child is not None:
                raise
            attempt.update(status="launch_failed", returncode=127, error_type=type(error).__name__)
        finally:
            # Unexpected supervisor exceptions must not leave a live trainer.
            if child is not None and child.poll() is None:
                child.terminate()
                try:
                    child.wait(timeout=min(grace_seconds, max(0, started + seconds - time.monotonic())))
                except subprocess.TimeoutExpired:
                    child.kill()
                    child.wait()
            elapsed = time.monotonic() - started
            if inhibitor is not None and inhibitor.poll() is None:
                inhibitor.terminate()
                try:
                    inhibitor.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    inhibitor.kill()
                    inhibitor.wait()
            for sig, handler in handlers.items():
                signal.signal(sig, handler)
            # Do not conceal a rare OS scheduling/termination overrun by clipping
            # it to the reservation. Such an overrun reduces future allowance.
            attempt["charged_seconds"] = elapsed
            attempt["observed_child_wall_seconds"] = elapsed
            if attempt["status"] == "running":
                attempt["status"] = "supervisor_error"
            durable_write(journal, ledger)
        return {"id": identifier, "status": attempt["status"], "returncode": attempt["returncode"],
                "session_seconds": attempt["charged_seconds"],
                "charged_seconds": charged_before + attempt["charged_seconds"],
                "remaining_seconds": budget_seconds - charged_before - attempt["charged_seconds"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--journal", type=Path, required=True)
    parser.add_argument("--budget-seconds", type=float, default=7200)
    parser.add_argument("--seconds", type=float, required=True)
    parser.add_argument("--grace-seconds", type=float, default=30)
    for name in ("data", "base", "run", "config"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--caffeinate", action="store_true", help="Prevent idle sleep while this child runs on macOS")
    args = parser.parse_args()
    def command(loop_seconds):
        argv = [sys.executable, "-m", "silivra", "finetune"]
        for name in ("data", "base", "run", "config"):
            argv.extend([f"--{name}", str(getattr(args, name).resolve())])
        argv.extend(["--max-minutes", str(loop_seconds / 60)])
        return argv + (["--resume"] if args.resume else [])
    try:
        result = supervise(args.journal, args.budget_seconds, args.seconds, command,
                           grace_seconds=args.grace_seconds, caffeinate=args.caffeinate)
    except (ValueError, RuntimeError) as error:
        parser.error(str(error))
    print(json.dumps(result), flush=True)
    return 0 if result["returncode"] == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())

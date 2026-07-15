"""
job_runner.py — minimal background-job runner, phase-agnostic (no Dash
import). Launches a Python callable in a background thread and tracks its
status in a status.json file inside a per-job directory.

File-based status (not in-memory) is deliberate: the contract is "a job runs
somewhere and writes its status to a known directory; the UI polls that
directory" — which a local background thread satisfies today, and a SCITAS
job (status written after polling over SSH, or synced back) could satisfy
identically later, without changing any polling/UI code. submit_to_scitas()
below is a placeholder for that — not implemented yet.
"""
from __future__ import annotations

import json
import os
import threading
import time
import traceback
import uuid


def new_job_dir(base_dir: str) -> tuple[str, str]:
    """Create a fresh job directory under base_dir. Returns (job_id, job_dir)."""
    job_id = uuid.uuid4().hex[:12]
    job_dir = os.path.join(base_dir, f"job_{job_id}")
    os.makedirs(job_dir, exist_ok=True)
    return job_id, job_dir


def _status_path(job_dir: str) -> str:
    return os.path.join(job_dir, "status.json")


def read_status(job_dir: str) -> dict | None:
    """{"state": "running"|"done"|"error", "result"?, "error"?, "traceback"?,
    "started_at", "updated_at"} — None if the job hasn't written anything yet."""
    path = _status_path(job_dir)
    if not os.path.isfile(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None  # being written concurrently — caller just polls again


def _write_status(job_dir: str, **fields) -> None:
    current = read_status(job_dir) or {}
    current.update(fields)
    current["updated_at"] = time.time()
    path = _status_path(job_dir)
    tmp_path = path + ".tmp"
    with open(tmp_path, "w") as f:
        json.dump(current, f, indent=2)
    os.replace(tmp_path, path)  # atomic — pollers never see a half-written file


def start_local_job(job_dir: str, fn, *args, **kwargs) -> None:
    """Run fn(*args, **kwargs) in a background thread. fn's return value must
    be JSON-serializable — it becomes status["result"] on success."""
    _write_status(job_dir, state="running", started_at=time.time())

    def _run():
        try:
            result = fn(*args, **kwargs)
            _write_status(job_dir, state="done", result=result)
        except Exception as e:
            _write_status(job_dir, state="error", error=str(e), traceback=traceback.format_exc())

    threading.Thread(target=_run, daemon=True).start()


def submit_to_scitas(*args, **kwargs):
    """Placeholder — SCITAS job submission (SSH + sbatch) isn't built yet.
    start_local_job's file-based status contract is designed so this can
    slot in later (a job_dir with the same status.json shape, populated by
    polling/syncing the remote job instead of a local thread) without any
    caller/UI changes."""
    raise NotImplementedError("SCITAS job submission isn't implemented yet — use start_local_job for now.")

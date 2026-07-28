"""
scitas_discovery.py — shared SCITAS (SLURM cluster) submission primitives,
no Dash import.

SSH/SCP via the system's own ssh/scp binaries (already on PATH, reusing the
jed.hpc.epfl.ch Host entry in ~/.ssh/config — key-based auth, no new
dependency, no password/passphrase prompt needed, confirmed working).

Each phase's own *_discovery.py builds a run_x_on_scitas() function on top of
these primitives (submit + block-until-done + sync results back), meant to
be passed to job_runner.start_local_job() exactly like its local equivalent —
a SCITAS job still fits the "background thread does the work, writes
status.json" contract job_runner.py already provides; the only difference is
the work is "submit, then poll a remote queue" instead of "compute directly".
"""
from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import time
from pathlib import Path

DEFAULT_HOST = "jed.hpc.epfl.ch"  # matches the Host alias in ~/.ssh/config — works today as-is
SCITAS_HOST = DEFAULT_HOST  # kept for backward compatibility with any direct reference

# Local machine settings (not project data) — overrides for host/username/
# identity file, only used if the default (~/.ssh/config-resolved) connection
# doesn't work for someone. Blank/absent fields fall back to the default.
_SETTINGS_PATH = Path(__file__).resolve().parent / "scitas_settings.json"

DEFAULT_KEY_PATH = str(Path.home() / ".ssh" / "id_scitas_gui")  # a new, separate key — never touches
                                                                # existing keys like id_epfl_jed

LOCAL_PIPELINE_DIR = str(Path(__file__).resolve().parent.parent / "pipeline")

# Files whose remote copy on SCITAS scratch actually gets executed there —
# tracked so the GUI can show a "does this match what's on the cluster"
# status before submitting a job. Syncing is a deliberate, explicit action
# (sync_pipeline_code()) — never done silently as a side effect of
# submitting a job, since it changes state on a shared resource.
TRACKED_PIPELINE_FILES = [
    "config.py",
    "run_pipeline.py",
    "create_masks.py",
    "compare_ti_montages.py",
    "generate_configs.py",
    "register_caps.py",
    "charm_scitas.sbatch",
    "simnibs_ti_pipeline.sbatch",
    "recon_all_scitas.sbatch",
]

_SSH_TIMEOUT = 30       # a single ssh command (submit, squeue, sacct, test -e) should be fast
POLL_INTERVAL_S = 30    # how often wait_for_job() checks squeue/sacct while blocking

_TERMINAL_STATES = {
    "COMPLETED", "FAILED", "CANCELLED", "TIMEOUT",
    "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED", "BOOT_FAIL", "DEADLINE",
}

_cached_username: str | None = None


def load_settings() -> dict:
    """{"host", "username", "identity_file", "server_data_dir"} —
    username/identity_file/server_data_dir are None unless the user
    explicitly overrode them (see the SCITAS Connection / Data Directory
    pages). host defaults to ~/.ssh/config's own Host alias resolution;
    server_data_dir defaults to /scratch/{username}/BIDS_TI_Toolbox — the
    project root on the SCITAS filesystem (code/, rawdata/, derivatives/
    all live under it there, mirroring the local layout) — see
    scitas_scratch_dir(). This is NOT the same path the config JSONs
    reference at runtime (/mnt/BIDS_TI_Toolbox, see config_discovery.py) —
    that one only exists INSIDE the Apptainer container via its --bind
    mount; server_data_dir is where that same tree actually sits on the
    bare SCITAS filesystem, outside any container."""
    if _SETTINGS_PATH.is_file():
        try:
            with open(_SETTINGS_PATH) as f:
                data = json.load(f)
            return {"host": data.get("host") or DEFAULT_HOST,
                    "username": data.get("username") or None,
                    "identity_file": data.get("identity_file") or None,
                    "server_data_dir": data.get("server_data_dir") or None}
        except (OSError, json.JSONDecodeError):
            pass
    return {"host": DEFAULT_HOST, "username": None, "identity_file": None, "server_data_dir": None}


def save_settings(host: str | None = None, username: str | None = None,
                  identity_file: str | None = None, server_data_dir: str | None = None) -> None:
    """Persists connection/path overrides locally and invalidates the
    cached remote username (a changed host/identity may resolve to a
    different account). Whatever is passed for each field becomes the new
    saved value; a blank/None field resets that one to its computed
    default. Callers that only want to change ONE field (e.g. the Data
    Directory page changing just server_data_dir) should read
    load_settings() first and pass the other fields through unchanged —
    this function itself doesn't merge with what's already saved."""
    global _cached_username
    settings = {"host": host or DEFAULT_HOST, "username": username or None,
               "identity_file": identity_file or None, "server_data_dir": server_data_dir or None}
    with open(_SETTINGS_PATH, "w") as f:
        json.dump(settings, f, indent=2)
    _cached_username = None


def _ssh_target() -> tuple[list[str], str]:
    """(extra_ssh_args, host) from current settings. extra_args only add
    -i/-l when explicitly overridden — otherwise ssh's own ~/.ssh/config
    resolves user/key for the host exactly as it does by default today."""
    s = load_settings()
    args = []
    if s["identity_file"]:
        args += ["-i", s["identity_file"]]
    if s["username"]:
        args += ["-l", s["username"]]
    return args, s["host"]


def remote_username() -> str:
    """The GASPAR username SSH logs in as (resolved once via `echo $USER` on
    the remote host, then cached) — needed to build absolute /scratch/...
    paths for scp, which (unlike ssh_run's double-quoted remote commands)
    does not reliably expand $USER itself."""
    global _cached_username
    if _cached_username is None:
        result = ssh_run("echo $USER")
        if not result["success"] or not result["stdout"].strip():
            raise RuntimeError(f"couldn't resolve remote username: {result['stderr']}")
        _cached_username = result["stdout"].strip()
    return _cached_username


def scitas_scratch_dir() -> str:
    """The project root on the bare SCITAS filesystem (outside any
    container) — code/, rawdata/, derivatives/ all live under it there.
    Defaults to /scratch/{username}/BIDS_TI_Toolbox; overridable (Data
    Directory page) for accounts/projects laid out differently."""
    override = load_settings().get("server_data_dir")
    return override or f"/scratch/{remote_username()}/BIDS_TI_Toolbox"


def scitas_pipeline_dir() -> str:
    return f"{scitas_scratch_dir()}/code/pipeline"


def ssh_run(remote_command: str, timeout: int = _SSH_TIMEOUT) -> dict:
    """Runs remote_command on SCITAS via `ssh <host> <command>` (BatchMode=
    yes — fail fast instead of hanging if key auth doesn't work, since this
    may run inside a background thread with no TTY to prompt on). Host/user/
    identity file come from load_settings() — defaults to jed.hpc.epfl.ch
    resolved via ~/.ssh/config unless overridden. Returns {"success",
    "returncode", "stdout", "stderr"}."""
    extra_args, host = _ssh_target()
    cmd = (["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={min(timeout, 15)}"]
          + extra_args + [host, remote_command])
    try:
        # encoding/errors explicit: remote output isn't guaranteed to be
        # decodable as Windows' default cp1252 (subprocess's own internal
        # stdout-reader thread crashes with UnicodeDecodeError otherwise —
        # seen live from real SCITAS output during testing).
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"success": False, "returncode": None, "stdout": "", "stderr": f"ssh timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "returncode": None, "stdout": "", "stderr": str(e)}
    return {"success": proc.returncode == 0, "returncode": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr}


def _scp_host_spec() -> str:
    """host or user@host — scp has no separate -l flag for username like ssh
    does, so an explicit username override has to be embedded in the spec."""
    s = load_settings()
    return f"{s['username']}@{s['host']}" if s["username"] else s["host"]


def _scp(args: list[str], timeout: int) -> dict:
    identity_file = load_settings()["identity_file"]
    identity_args = ["-i", identity_file] if identity_file else []
    cmd = ["scp", "-o", "BatchMode=yes"] + identity_args + args
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"success": False, "returncode": None, "stdout": "", "stderr": f"scp timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "returncode": None, "stdout": "", "stderr": str(e)}
    return {"success": proc.returncode == 0, "returncode": proc.returncode,
            "stdout": proc.stdout, "stderr": proc.stderr}


def scp_upload(local_path: str, remote_path: str, recursive: bool = True, timeout: int = 900) -> dict:
    args = (["-r"] if recursive else []) + [local_path, f"{_scp_host_spec()}:{remote_path}"]
    return _scp(args, timeout)


def scp_download(remote_path: str, local_path: str, recursive: bool = True, timeout: int = 900) -> dict:
    args = (["-r"] if recursive else []) + [f"{SCITAS_HOST}:{remote_path}", local_path]
    return _scp(args, timeout)


def batch_upload(local_root: str, relpaths: list[str], timeout: int = 900) -> dict:
    """Uploads many small files in ONE ssh connection (tar piped over ssh)
    instead of one scp call per file. Built for submitting several configs
    at once on the Run Pipeline page: each config needs its own config
    JSON + registered cap + ROI/non-ROI masks uploaded, and doing that one
    scp call at a time, times N configs, opens a burst of separate
    connections large enough to trip SCITAS's own new-connection rate
    limiting on its login node (seen live as
    "kex_exchange_identification: Connection closed by remote host").

    relpaths are relative to local_root AND to the remote scratch root
    (scitas_scratch_dir()) — safe because both mirror the same
    derivatives/SimNIBS/sub-{id}/... layout, so one shared -C root works
    for every file regardless of which subject/config it belongs to.
    Missing local files are silently skipped (not an error) — callers can
    pass a broad candidate list (e.g. every config's cap+mask paths, some
    of which may already exist remotely from an earlier run) without
    checking existence themselves first.

    Returns {"success", "uploaded": [relpaths actually sent], "error"}."""
    existing = [r for r in dict.fromkeys(relpaths) if os.path.isfile(os.path.join(local_root, r))]
    if not existing:
        return {"success": True, "uploaded": [], "error": None}

    try:
        tar_proc = subprocess.run(["tar", "-cf", "-", "-C", local_root] + existing,
                                  capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"success": False, "uploaded": [], "error": f"local tar timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "uploaded": [], "error": str(e)}
    if tar_proc.returncode != 0:
        return {"success": False, "uploaded": [],
                "error": f"local tar failed: {tar_proc.stderr.decode('utf-8', 'replace')}"}

    scratch = scitas_scratch_dir()
    extra_args, host = _ssh_target()
    remote_cmd = f"mkdir -p {shlex.quote(scratch)} && tar -xf - -C {shlex.quote(scratch)}"
    ssh_cmd = (["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15"]
              + extra_args + [host, remote_cmd])
    try:
        ssh_proc = subprocess.run(ssh_cmd, input=tar_proc.stdout, capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"success": False, "uploaded": [], "error": f"batch upload timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "uploaded": [], "error": str(e)}

    if ssh_proc.returncode != 0:
        return {"success": False, "uploaded": [], "error": ssh_proc.stderr.decode("utf-8", "replace")}
    return {"success": True, "uploaded": existing, "error": None}


def remote_path_exists(remote_path: str) -> bool:
    result = ssh_run(f"test -e {shlex.quote(remote_path)} && echo YES || echo NO")
    return result["success"] and "YES" in (result["stdout"] or "")


def remote_mkdir(remote_path: str) -> dict:
    return ssh_run(f"mkdir -p {shlex.quote(remote_path)}")


def remote_rmtree(remote_path: str) -> dict:
    """Careful: unconditional recursive remote delete. Callers must only
    pass a fully-resolved path they've already reasoned about (e.g. one
    specific subject's m2m_ folder before a forced re-run) — never a path
    built from unvalidated input."""
    return ssh_run(f"rm -rf {shlex.quote(remote_path)}")


def submit_sbatch(script_path: str, job_name: str, export_vars: dict | None = None,
                  cwd: str | None = None) -> dict:
    """Submits an sbatch script remotely. export_vars become
    --export=ALL,K=V,... (matching this project's existing submit_*.sh
    scripts). Returns {"success", "job_id", "error"}."""
    cwd = cwd or scitas_pipeline_dir()
    export_vars = export_vars or {}
    export_str = "ALL," + ",".join(f"{k}={v}" for k, v in export_vars.items()) if export_vars else "ALL"
    remote_cmd = (f"cd {shlex.quote(cwd)} && "
                 f"sbatch --job-name={shlex.quote(job_name)} --export={shlex.quote(export_str)} "
                 f"{shlex.quote(script_path)}")
    result = ssh_run(remote_cmd)
    if not result["success"]:
        return {"success": False, "job_id": None, "error": result["stderr"] or result["stdout"]}
    # sbatch's own stdout convention: "Submitted batch job 12345678"
    tail = (result["stdout"] or "").strip().split()
    if not tail:
        return {"success": False, "job_id": None, "error": f"unexpected sbatch output: {result['stdout']!r}"}
    return {"success": True, "job_id": tail[-1], "error": None}


def remote_paths_exist(paths: list[str]) -> dict[str, bool]:
    """Read-only: {path: exists_bool} for every path, in ONE ssh connection
    via a remote for-loop — same technique as remote_data_status()'s
    subject/folder check, generalized to arbitrary paths (e.g. several
    subjects' m2m_ folders + the shared BNA atlas, checked together before
    submitting several SCITAS jobs at once). Each path is base64-encoded on
    the way in and decoded remotely so spaces or other special characters
    in a path can't break the loop's word-splitting."""
    if not paths:
        return {}
    import base64
    encoded = " ".join(base64.b64encode(p.encode()).decode() for p in paths)
    script = (f'for enc in {encoded}; do p=$(echo "$enc" | base64 -d); '
             f'if [ -e "$p" ]; then echo "$enc Y"; else echo "$enc N"; fi; done')
    result = ssh_run(script, timeout=60)
    out = {p: False for p in paths}
    if result["success"]:
        for line in (result["stdout"] or "").splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    p = base64.b64decode(parts[0]).decode()
                except Exception:
                    continue
                if p in out:
                    out[p] = (parts[1] == "Y")
    return out


def submit_sbatch_batch(jobs: list[dict], timeout: int = 120) -> list[dict]:
    """Submits many independent sbatch jobs in ONE ssh connection instead of
    one connection per submission — the other half (alongside batch_upload())
    of what was still opening a burst of separate connections large enough
    to trip SCITAS's own new-connection rate limiting when starting several
    SCITAS runs at once (each submission is its own ssh call in
    submit_sbatch(), fired close together when N jobs start at once).

    jobs: [{"script_path", "job_name", "export_vars", "cwd"}, ...]. Returns
    one {"success", "job_id", "error"} dict per input job, in the same
    order — each job's own combined stdout+stderr is bracketed by a unique
    marker so a mid-batch failure can't get misattributed to the wrong job."""
    if not jobs:
        return []

    parts = []
    for i, j in enumerate(jobs):
        cwd = j.get("cwd") or scitas_pipeline_dir()
        export_vars = j.get("export_vars") or {}
        export_str = "ALL," + ",".join(f"{k}={v}" for k, v in export_vars.items()) if export_vars else "ALL"
        sbatch_cmd = (f"cd {shlex.quote(cwd)} && sbatch --job-name={shlex.quote(j['job_name'])} "
                     f"--export={shlex.quote(export_str)} {shlex.quote(j['script_path'])}")
        parts.append(f'echo __JOB_{i}_START__; ({sbatch_cmd}) 2>&1; echo __JOB_{i}_END__')
    result = ssh_run(" ; ".join(parts), timeout=timeout)

    output = result["stdout"] or ""
    if not result["success"] and not output.strip():
        err = result["stderr"] or "ssh call failed"
        return [{"success": False, "job_id": None, "error": err} for _ in jobs]

    results = []
    for i in range(len(jobs)):
        start_marker, end_marker = f"__JOB_{i}_START__", f"__JOB_{i}_END__"
        if start_marker in output and end_marker in output:
            segment = output.split(start_marker, 1)[1].split(end_marker, 1)[0]
        else:
            segment = ""
        seg_lines = [ln for ln in segment.splitlines() if ln.strip()]
        job_line = next((ln for ln in seg_lines if ln.strip().startswith("Submitted batch job")), None)
        if job_line:
            results.append({"success": True, "job_id": job_line.strip().split()[-1], "error": None})
        else:
            results.append({"success": False, "job_id": None,
                            "error": "\n".join(seg_lines) or "sbatch produced no output"})
    return results


def job_state(job_id: str) -> str | None:
    """Current SLURM state for job_id — checks squeue first (still queued/
    running), falls back to sacct (already left the queue: done, failed,
    cancelled, timed out). Returns None if neither call succeeds."""
    result = ssh_run(f"squeue -j {shlex.quote(job_id)} -h -o %T")
    if result["success"] and (result["stdout"] or "").strip():
        return result["stdout"].strip()
    result = ssh_run(f"sacct -j {shlex.quote(job_id)} --format=State --noheader -X")
    if result["success"] and (result["stdout"] or "").strip():
        return result["stdout"].strip().split()[0]
    return None


def wait_for_job(job_id: str, poll_interval: int = POLL_INTERVAL_S, max_wait_s: int = 6 * 3600) -> dict:
    """Blocks (sleeping poll_interval seconds between checks) until job_id
    leaves the SLURM queue, or max_wait_s elapses. Meant to run inside a
    job_runner background thread, not a Dash callback directly. Returns
    {"success", "final_state", "error"}."""
    elapsed = 0
    last_state = None
    while elapsed < max_wait_s:
        state = job_state(job_id)
        if state:
            last_state = state
            if state.upper() in _TERMINAL_STATES:
                return {"success": state.upper() == "COMPLETED", "final_state": state, "error": None}
        time.sleep(poll_interval)
        elapsed += poll_interval
    return {"success": False, "final_state": last_state,
            "error": f"timed out waiting after {max_wait_s}s (last known state: {last_state})"}


# ═════════════════════════════════════════════════════════════════════════════
# Connection check + SSH key management — for the SCITAS Connection page.
# test_connection() is read-only (safe anytime). generate_ssh_key() only
# ever creates a NEW local keypair (never overwrites, never touches the
# cluster) — it does not by itself grant SCITAS access; the resulting
# public key still has to be installed on the remote side (ssh-copy-id, or
# EPFL's own GASPAR key-management process) before it works.
# ═════════════════════════════════════════════════════════════════════════════

def test_connection(timeout: int = 10) -> dict:
    """Read-only. {"success", "host", "username", "error"}."""
    extra_args, host = _ssh_target()
    cmd = (["ssh", "-o", "BatchMode=yes", "-o", f"ConnectTimeout={timeout}"]
          + extra_args + [host, "echo CONNECTION_OK && echo USER=$USER"])
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout + 5,
                              encoding="utf-8", errors="replace")
    except subprocess.TimeoutExpired:
        return {"success": False, "host": host, "username": None,
                "error": f"connection timed out after {timeout}s"}
    except Exception as e:
        return {"success": False, "host": host, "username": None, "error": str(e)}

    if proc.returncode != 0 or "CONNECTION_OK" not in (proc.stdout or ""):
        err = (proc.stderr or proc.stdout or "unknown error").strip()
        return {"success": False, "host": host, "username": None, "error": err}

    username = None
    for line in proc.stdout.splitlines():
        if line.startswith("USER="):
            username = line[len("USER="):]
    return {"success": True, "host": host, "username": username, "error": None}


def ssh_key_exists(path: str = DEFAULT_KEY_PATH) -> bool:
    return os.path.isfile(path) and os.path.isfile(path + ".pub")


def generate_ssh_key(path: str = DEFAULT_KEY_PATH) -> dict:
    """Generates a new ed25519 keypair (no passphrase, so it works
    non-interactively from a background thread) at `path` — but only if one
    doesn't already exist there; never overwrites. Local filesystem only, no
    cluster interaction. Returns {"success", "path", "public_key", "error"};
    public_key still needs to be added to SCITAS (see the module docstring)
    before this key actually grants access."""
    if ssh_key_exists(path):
        with open(path + ".pub") as f:
            return {"success": True, "path": path, "public_key": f.read().strip(), "error": None}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    try:
        proc = subprocess.run(
            ["ssh-keygen", "-t", "ed25519", "-f", path, "-N", "", "-C", "bids_ti_toolbox_gui"],
            capture_output=True, text=True, timeout=30, encoding="utf-8", errors="replace",
        )
    except Exception as e:
        return {"success": False, "path": path, "public_key": None, "error": str(e)}

    if proc.returncode != 0 or not os.path.isfile(path + ".pub"):
        return {"success": False, "path": path, "public_key": None, "error": proc.stderr or proc.stdout}

    with open(path + ".pub") as f:
        return {"success": True, "path": path, "public_key": f.read().strip(), "error": None}


# ═════════════════════════════════════════════════════════════════════════════
# Code sync status/action — read-only comparison is safe to run anytime;
# sync_pipeline_code() changes remote state and must only be called from an
# explicit, user-initiated action (a dedicated "Sync Code" button), never
# automatically bundled into job submission.
# ═════════════════════════════════════════════════════════════════════════════

def _local_md5(path: str) -> str | None:
    if not os.path.isfile(path):
        return None
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _remote_md5(remote_path: str) -> str | None:
    result = ssh_run(f"md5sum {shlex.quote(remote_path)} 2>/dev/null")
    if not result["success"] or not (result["stdout"] or "").strip():
        return None
    return result["stdout"].strip().split()[0]


def code_sync_status(local_pipeline_dir: str = LOCAL_PIPELINE_DIR,
                      filenames: list[str] | None = None) -> dict:
    """Read-only: {filename: {"local_hash", "remote_hash", "in_sync"}} for
    each tracked pipeline file — compares local md5 against the copy on
    SCITAS scratch. "in_sync" is False if either side is missing the file
    or the hashes differ."""
    filenames = filenames if filenames is not None else TRACKED_PIPELINE_FILES
    remote_dir = scitas_pipeline_dir()
    out = {}
    for fname in filenames:
        local_hash = _local_md5(os.path.join(local_pipeline_dir, fname))
        remote_hash = _remote_md5(f"{remote_dir}/{fname}")
        out[fname] = {
            "local_hash": local_hash, "remote_hash": remote_hash,
            "in_sync": local_hash is not None and local_hash == remote_hash,
        }
    return out


def sync_pipeline_code(local_pipeline_dir: str = LOCAL_PIPELINE_DIR,
                       filenames: list[str] | None = None) -> dict:
    """Explicit, state-changing action: scp's each tracked (or given)
    filename up to SCITAS scratch, overwriting the remote copy. Returns
    {filename: {"success", "error"}}."""
    filenames = filenames if filenames is not None else TRACKED_PIPELINE_FILES
    remote_dir = scitas_pipeline_dir()
    remote_mkdir(remote_dir)
    out = {}
    for fname in filenames:
        local_path = os.path.join(local_pipeline_dir, fname)
        if not os.path.isfile(local_path):
            out[fname] = {"success": False, "error": f"local file not found: {local_path}"}
            continue
        result = scp_upload(local_path, f"{remote_dir}/{fname}", recursive=False)
        out[fname] = {"success": result["success"], "error": None if result["success"] else result["stderr"]}
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Data sync (SCITAS scratch → local data directory) — for the Data Directory
# settings page. list_remote_subjects()/remote_data_status() are read-only.
# sync_subject_data() is explicit and state-changing (downloads files) —
# only ever called from a user-initiated "Sync Selected" action, never
# automatically.
# ═════════════════════════════════════════════════════════════════════════════

# subfolder tag -> path relative to the project root, both locally and on
# SCITAS scratch (same layout on both sides — see common.DATA_SUBDIRS).
# "derivatives/SimNIBS" itself is synced EXCLUDING leadfield_volume/ (see
# sync_subject_data() and LEADFIELD_DIRNAME below) — leadfields are cached
# HDF5 volumes that can run into the GB range per (cap, electrode-geometry)
# combination and aren't needed for most local work (viewing results,
# registering a new cap, etc.). "derivatives/SimNIBS/leadfield_volume" is a
# separate, independently-toggleable entry for when you DO want it too.
LEADFIELD_DIRNAME = "leadfield_volume"

DATA_FOLDERS = {
    "rawdata": "rawdata/sub-{sid}",
    "derivatives/SimNIBS": "derivatives/SimNIBS/sub-{sid}",
    "derivatives/SimNIBS/leadfield_volume": f"derivatives/SimNIBS/sub-{{sid}}/{LEADFIELD_DIRNAME}",
    "derivatives/freesurfer": "derivatives/freesurfer/sub-{sid}",
}


def list_remote_subjects() -> list[str]:
    """Read-only: subject IDs present in EITHER rawdata/ or
    derivatives/SimNIBS/ on SCITAS scratch (sub-{id} folder names, prefix
    stripped), sorted, deduplicated."""
    scratch = scitas_scratch_dir()
    ids = set()
    for rel in ("rawdata", "derivatives/SimNIBS"):
        result = ssh_run(f"ls -1 {shlex.quote(f'{scratch}/{rel}')} 2>/dev/null")
        if result["success"]:
            for name in (result["stdout"] or "").splitlines():
                name = name.strip()
                if name.startswith("sub-"):
                    ids.add(name[len("sub-"):])
    return sorted(ids)


def remote_data_status(subject_ids: list[str]) -> dict:
    """Read-only: {subject_id: {folder_tag: exists_bool}} for each of
    DATA_FOLDERS, across the given subjects — lets the picker show what's
    actually there before the user selects what to sync.

    One SSH round trip total, not one per (subject, folder) pair — the
    latter (34 subjects x 3 folders = 102 separate SSH connections, each
    with its own handshake) was the actual cause of "Load Subjects from
    SCITAS" taking 100+ seconds and looking hung.

    Runs as a remote `for` loop rather than one flat chain of 102 `test &&
    echo || echo` clauses joined by ";" — that first attempt built a command
    whose length scaled with subject count (~14KB at 34 subjects) and
    silently lost output partway through (likely a Windows ssh.exe
    argv-requoting/buffering limit), which remote_data_status had no way to
    detect since ssh_run still reported success. A loop keeps the command
    body constant-size regardless of how many subjects are checked.

    Each tag gets its own `if [ -e ... ]` check (rather than one generic
    "$tag/sub-$sid" pattern shared by every tag) so DATA_FOLDERS templates
    don't all have to share the same shape — e.g. the leadfield_volume
    entry has {sid} in the MIDDLE of its path, not at the end."""
    scratch = scitas_scratch_dir()
    out = {sid: {tag: False for tag in DATA_FOLDERS} for sid in subject_ids}
    if not subject_ids:
        return out

    sid_list = " ".join(shlex.quote(s) for s in subject_ids)
    checks = []
    for tag, template in DATA_FOLDERS.items():
        rel = template.replace("{sid}", "$sid")
        checks.append(f'if [ -e "{scratch}/{rel}" ]; then echo "$sid {shlex.quote(tag)} Y"; '
                     f'else echo "$sid {shlex.quote(tag)} N"; fi')
    script = f'for sid in {sid_list}; do ' + "; ".join(checks) + "; done"
    result = ssh_run(script, timeout=60)
    if result["success"]:
        for line in (result["stdout"] or "").splitlines():
            parts = line.split()
            if len(parts) == 3 and parts[0] in out and parts[1] in out[parts[0]]:
                out[parts[0]][parts[1]] = (parts[2] == "Y")
    return out


# A failed/interrupted run_pipeline.py invocation leaves two kinds of
# leftover clutter under derivatives/SimNIBS/sub-{id}/: an empty
# TIoptimization/ run folder (Section 2 created the directory before
# actually failing) and, one level up, the sub-{id}_desc-config_run-
# {timestamp}.json snapshot written when that same invocation started.
# Neither is useful once the run never produced output, and across many
# retries (exactly the pattern in this project's iterative SCITAS testing)
# they add up. Found via timestamp proximity, not an exact key — the
# desc-config's own timestamp is a few seconds EARLIER than its run
# folder's (Section 1 runs in between) — verified live against real data:
# every run folder's nearest preceding desc-config, within
# _EMPTY_RUN_MATCH_TOLERANCE, is unique to that one folder (no two run
# folders, empty or not, ever share a closest match).
_EMPTY_RUN_MATCH_TOLERANCE_S = 120
_RUN_TS_RE_SRC = r"run-(\d{8}_\d{6})"


def find_empty_run_exclusions(subject_id: str) -> dict:
    """Read-only. {"empty_run_dirs": [basenames], "matched_config_jsons":
    [basenames]} for subject_id's TIoptimization/ — ready to use as
    _sync_incremental() exclude patterns (see sync_subject_data()). One
    ssh connection total."""
    import re
    import datetime

    scratch = scitas_scratch_dir()
    sub_dir = f"{scratch}/derivatives/SimNIBS/sub-{subject_id}"
    ti_dir = f"{sub_dir}/TIoptimization"
    out = {"empty_run_dirs": [], "matched_config_jsons": []}

    result = ssh_run(
        f"find {shlex.quote(ti_dir)} -mindepth 1 -maxdepth 1 -type d -empty -printf '%f\\n' 2>/dev/null; "
        f"echo ---; ls {shlex.quote(sub_dir)} 2>/dev/null | grep desc-config",
        timeout=60,
    )
    if not result["success"]:
        return out
    empty_section, _, config_section = (result["stdout"] or "").partition("---\n")
    empty_dirs = [ln.strip() for ln in empty_section.splitlines() if ln.strip()]
    config_files = [ln.strip() for ln in config_section.splitlines() if ln.strip()]
    if not empty_dirs:
        return out

    ts_re = re.compile(_RUN_TS_RE_SRC)

    def parse_ts(name):
        m = ts_re.search(name)
        return datetime.datetime.strptime(m.group(1), "%Y%m%d_%H%M%S") if m else None

    config_ts = [(f, t) for f, t in ((f, parse_ts(f)) for f in config_files) if t]
    tolerance = datetime.timedelta(seconds=_EMPTY_RUN_MATCH_TOLERANCE_S)

    matched_configs = []
    for empty_name in empty_dirs:
        t_empty = parse_ts(empty_name)
        if not t_empty:
            continue
        candidates = sorted(
            ((f, t_empty - t) for f, t in config_ts if datetime.timedelta(0) <= (t_empty - t) <= tolerance),
            key=lambda x: x[1],
        )
        if candidates:
            matched_configs.append(candidates[0][0])

    out["empty_run_dirs"] = empty_dirs
    out["matched_config_jsons"] = matched_configs
    return out


def _remote_md5_many(remote_dir: str, relpaths: list[str], timeout: int = 300) -> dict:
    """{relpath: md5} for many files relative to remote_dir, in ONE ssh
    connection — same base64-encode-each-token-then-loop technique as
    remote_paths_exist() (avoids both shell-quoting fragility for
    arbitrary filenames and scaling the command TEXT with item count)."""
    if not relpaths:
        return {}
    import base64
    encoded = " ".join(base64.b64encode(p.encode()).decode() for p in relpaths)
    script = (f'cd {shlex.quote(remote_dir)} && for enc in {encoded}; do '
             f'p=$(echo "$enc" | base64 -d); h=$(md5sum -- "$p" 2>/dev/null | cut -d" " -f1); '
             f'echo "$enc $h"; done')
    result = ssh_run(script, timeout=timeout)
    out = {}
    if result["success"]:
        for line in (result["stdout"] or "").splitlines():
            parts = line.split()
            if len(parts) == 2:
                try:
                    out[base64.b64decode(parts[0]).decode()] = parts[1]
                except Exception:
                    continue
    return out


def _sync_incremental(remote_dir: str, local_parent: str, exclude_names: list[str], timeout: int = 1800,
                      progress_cb=None) -> dict:
    """Downloads remote_dir into local_parent, transferring only files
    that are actually new or changed — anything already present locally
    with an IDENTICAL CONTENT HASH is skipped, and anything whose
    basename (or an ancestor directory's basename) is in exclude_names is
    skipped outright without even being considered.

    File modification time isn't a usable signal here (re-extracting a
    file resets its local mtime, so a freshly-downloaded copy never
    matches the remote original's mtime even when content is identical),
    and file SIZE alone isn't safe either — this project's TI-optimization
    mesh files keep the same element count (and therefore near-identical
    size) across completely different runs of the same subject, since only
    the per-element field VALUES differ, not the mesh topology. Content
    hashing is the only one of these three that can't give a false
    "unchanged" verdict — only computed for same-size candidates, so files
    that are obviously new or size-mismatched skip the hashing step
    entirely and go straight to the transfer list.

    Returns {"success", "error", "n_transferred", "n_skipped"}."""
    remote_dir = remote_dir.rstrip("/")
    remote_base = os.path.basename(remote_dir)
    exclude_set = set(exclude_names)

    def _excluded(relpath):
        return any(part in exclude_set for part in relpath.split("/"))

    list_result = ssh_run(f"find {shlex.quote(remote_dir)} -type f -printf '%P %s\\n'", timeout=120)
    if not list_result["success"]:
        return {"success": False, "error": list_result["stderr"], "n_transferred": 0, "n_skipped": 0}
    remote_files = {}
    for line in (list_result["stdout"] or "").splitlines():
        relpath, _, size_str = line.rpartition(" ")
        if relpath and size_str.isdigit() and not _excluded(relpath):
            remote_files[relpath] = int(size_str)

    local_root = os.path.join(local_parent, remote_base)
    local_files = {}
    if os.path.isdir(local_root):
        for root, _dirs, files in os.walk(local_root):
            for fname in files:
                full = os.path.join(root, fname)
                relpath = os.path.relpath(full, local_root).replace(os.sep, "/")
                local_files[relpath] = os.path.getsize(full)

    to_transfer = []
    hash_candidates = []
    for relpath, rsize in remote_files.items():
        lsize = local_files.get(relpath)
        if lsize is None or lsize != rsize:
            to_transfer.append(relpath)
        else:
            hash_candidates.append(relpath)

    if hash_candidates:
        if progress_cb:
            progress_cb(f"checking {len(hash_candidates)} file(s) already present locally by content hash...")
        remote_hashes = _remote_md5_many(remote_dir, hash_candidates)
        for relpath in hash_candidates:
            if remote_hashes.get(relpath) != _local_md5(os.path.join(local_root, *relpath.split("/"))):
                to_transfer.append(relpath)

    n_skipped = len(remote_files) - len(to_transfer)
    if progress_cb:
        progress_cb(f"{len(to_transfer)} file(s) to transfer, {n_skipped} already up to date — skipped")
    if not to_transfer:
        return {"success": True, "error": None, "n_transferred": 0, "n_skipped": n_skipped}

    # Download the changed/new files one at a time, in series — plain
    # scp_download() per file, nothing fancier. Each individual transfer's
    # own stdout/stderr is handled by subprocess.run internally (the same
    # proven, non-streaming path every other download/upload in this module
    # already uses), so there's no manual pipe-draining to get wrong.
    n_transferred = 0
    for i, relpath in enumerate(to_transfer, 1):
        remote_file = f"{remote_dir}/{relpath}"
        local_file = os.path.join(local_root, *relpath.split("/"))
        os.makedirs(os.path.dirname(local_file), exist_ok=True)
        if progress_cb:
            progress_cb(f"[{i}/{len(to_transfer)}] {relpath}")
        dl = scp_download(remote_file, local_file, recursive=False, timeout=timeout)
        if not dl["success"]:
            return {"success": False, "error": f"{relpath}: {dl['stderr']}",
                    "n_transferred": n_transferred, "n_skipped": n_skipped}
        n_transferred += 1

    return {"success": True, "error": None, "n_transferred": n_transferred, "n_skipped": n_skipped}


def sync_subject_data(subject_ids: list[str], folder_tags: list[str], local_data_dir: str,
                      progress_cb=None) -> dict:
    """Explicit, state-changing: pulls the selected folder(s) for each
    selected subject from SCITAS scratch into local_data_dir (same relative
    layout on both sides). Returns {(subject_id, folder_tag): {"success",
    "error"}}. One subject/folder's failure doesn't stop the rest.

    "derivatives/SimNIBS" is downloaded EXCLUDING leadfield_volume/ (see
    _sync_incremental, DATA_FOLDERS' module comment) — select the
    separate "derivatives/SimNIBS/leadfield_volume" tag too if you actually
    want that as well. It also excludes empty TIoptimization/ run folders
    (a failed/interrupted run_pipeline.py invocation) and their matching
    desc-config JSON snapshot (see find_empty_run_exclusions()), and skips
    any individual file that's already present locally with an identical
    content hash — so re-syncing after a previous sync only transfers
    what's actually new or changed.

    progress_cb(line), if given, is called with a "[sub-{id} / {tag}]
    starting..."/"...done"/"...failed" line around each (subject, folder)
    pair, plus (for the "derivatives/SimNIBS" case) one line per file as
    it's individually downloaded — see _sync_incremental."""
    scratch = scitas_scratch_dir()
    out = {}
    for sid in subject_ids:
        for tag in folder_tags:
            template = DATA_FOLDERS.get(tag)
            if template is None:
                out[(sid, tag)] = {"success": False, "error": f"unknown folder tag: {tag}"}
                continue
            rel = template.format(sid=sid)
            remote_path = f"{scratch}/{rel}"
            local_path = os.path.join(local_data_dir, *rel.split("/"))
            if progress_cb:
                progress_cb(f"[sub-{sid} / {tag}] starting...")
            if not remote_path_exists(remote_path):
                out[(sid, tag)] = {"success": False, "error": f"not found on SCITAS: {remote_path}"}
                if progress_cb:
                    progress_cb(f"[sub-{sid} / {tag}] not found on SCITAS — skipped")
                continue
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            if tag == "derivatives/SimNIBS":
                skip = find_empty_run_exclusions(sid)
                exclude_names = [LEADFIELD_DIRNAME] + skip["empty_run_dirs"] + skip["matched_config_jsons"]
                if progress_cb and (skip["empty_run_dirs"] or skip["matched_config_jsons"]):
                    progress_cb(f"[sub-{sid} / {tag}] skipping {len(skip['empty_run_dirs'])} empty run "
                               f"folder(s) and {len(skip['matched_config_jsons'])} matching config file(s)")
                result = _sync_incremental(remote_path, os.path.dirname(local_path), exclude_names,
                                           progress_cb=progress_cb)
                done_suffix = (f" ({result['n_transferred']} transferred, {result['n_skipped']} already "
                              f"up to date)" if result["success"] else "")
            else:
                dl = scp_download(remote_path, os.path.dirname(local_path), recursive=True)
                result = {"success": dl["success"], "error": None if dl["success"] else dl["stderr"]}
                done_suffix = ""
            if progress_cb:
                status = "done" + done_suffix if result["success"] else f"failed: {result['error']}"
                progress_cb(f"[sub-{sid} / {tag}] {status}")
            out[(sid, tag)] = result
    return out

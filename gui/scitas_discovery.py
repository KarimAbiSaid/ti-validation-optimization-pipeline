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
DATA_FOLDERS = {
    "rawdata": "rawdata/sub-{sid}",
    "derivatives/SimNIBS": "derivatives/SimNIBS/sub-{sid}",
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
    actually there before the user selects what to sync."""
    scratch = scitas_scratch_dir()
    out = {}
    for sid in subject_ids:
        out[sid] = {}
        for tag, template in DATA_FOLDERS.items():
            remote_path = f"{scratch}/{template.format(sid=sid)}"
            out[sid][tag] = remote_path_exists(remote_path)
    return out


def sync_subject_data(subject_ids: list[str], folder_tags: list[str], local_data_dir: str) -> dict:
    """Explicit, state-changing: scp -r's the selected folder(s) for each
    selected subject from SCITAS scratch into local_data_dir (same relative
    layout on both sides). Returns {(subject_id, folder_tag): {"success",
    "error"}}. One subject/folder's failure doesn't stop the rest."""
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
            if not remote_path_exists(remote_path):
                out[(sid, tag)] = {"success": False, "error": f"not found on SCITAS: {remote_path}"}
                continue
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            result = scp_download(remote_path, os.path.dirname(local_path), recursive=True)
            out[(sid, tag)] = {"success": result["success"], "error": None if result["success"] else result["stderr"]}
    return out

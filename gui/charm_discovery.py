"""
charm_discovery.py — Phase 0 (head modeling / charm) data/logic layer, no
Dash import.

charm is a real external command-line tool (not a SimNIBS Python API call
like TDCSLEADFIELD/SESSION), invoked exactly as the SCITAS pipeline does —
see charm_scitas.sbatch/submit_charm_scitas.sh: `charm subID T1 [T2]
--forceqform`, cwd set to derivatives/SimNIBS/sub-{id}/ (charm creates
m2m_{id}/ there), completion marker m2m_{id}/{id}.msh. Same partial-folder
cleanup as the sbatch script (charm refuses to run over an existing
m2m_{id} folder otherwise).

A real run is ~30-90 minutes (per the sbatch script's own comment) — meant
to run inside a background job (job_runner.py), not called directly from a
Dash callback. run_charm_on_scitas() is the remote equivalent: submits
charm_scitas.sbatch, blocks until the SLURM job finishes, then scp's
m2m_{id}/ back — same call signature/return shape as run_charm(), so both
fit job_runner.start_local_job() identically. charm itself is a standalone
CLI tool inside the container (no dependency on this project's Python
pipeline files), so unlike Run Pipeline's SCITAS path, this one isn't
affected by local/remote code drift in config.py/run_pipeline.py/etc.
"""
from __future__ import annotations

import os
import shutil
import subprocess
import sys

from common import PROJECT_DIR, discover_subjects, get_m2m_path  # noqa: F401 (re-exported)


def t1_path(subject_id: str, project_dir: str = PROJECT_DIR) -> str:
    return os.path.join(project_dir, "rawdata", f"sub-{subject_id}", "anat", f"sub-{subject_id}_T1w.nii.gz")


def t2_path(subject_id: str, project_dir: str = PROJECT_DIR) -> str:
    return os.path.join(project_dir, "rawdata", f"sub-{subject_id}", "anat", f"sub-{subject_id}_T2w.nii.gz")


def _charm_exe() -> str:
    """charm.exe lives in the same env's Scripts/ folder as this
    interpreter (Windows) — resolved from sys.executable rather than
    assuming PATH includes it (the GUI env's Scripts/ isn't necessarily
    activated just because we're running its python.exe directly)."""
    scripts_dir = os.path.join(os.path.dirname(sys.executable), "Scripts")
    for name in ("charm.exe", "charm"):
        candidate = os.path.join(scripts_dir, name)
        if os.path.isfile(candidate):
            return candidate
    return "charm"  # fall back to PATH lookup


def charm_status(subject_id: str, project_dir: str = PROJECT_DIR) -> dict:
    """{"t1_exists", "t2_exists", "m2m_complete" (real completion marker —
    m2m_{id}/{id}.msh present), "m2m_partial" (m2m_ folder exists but the
    marker doesn't — a previous failed/interrupted run)}."""
    m2m_path = get_m2m_path(subject_id, project_dir)
    mesh_path = os.path.join(m2m_path, f"{subject_id}.msh")
    return {
        "t1_exists": os.path.isfile(t1_path(subject_id, project_dir)),
        "t2_exists": os.path.isfile(t2_path(subject_id, project_dir)),
        "m2m_complete": os.path.isfile(mesh_path),
        "m2m_partial": os.path.isdir(m2m_path) and not os.path.isfile(mesh_path),
    }


def run_charm(subject_id: str, use_t2: bool = True, force: bool = False,
              project_dir: str = PROJECT_DIR) -> dict:
    """Runs charm locally (subprocess, blocking — call this inside a
    job_runner background job). Returns {"success", "error", "stdout",
    "m2m_path"}."""
    status = charm_status(subject_id, project_dir)
    if not status["t1_exists"]:
        return {"success": False, "error": f"T1w not found: {t1_path(subject_id, project_dir)}"}

    m2m_path = get_m2m_path(subject_id, project_dir)
    mesh_path = os.path.join(m2m_path, f"{subject_id}.msh")

    if status["m2m_complete"] and not force:
        return {"success": True, "cached": True, "m2m_path": m2m_path, "error": None, "stdout": ""}

    # charm refuses to run over an existing m2m_{id} folder — clean up a
    # partial one (previous failed run) or an existing complete one when
    # force=True, same as charm_scitas.sbatch does.
    if os.path.isdir(m2m_path):
        shutil.rmtree(m2m_path)

    sub_dir = os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}")
    os.makedirs(sub_dir, exist_ok=True)

    cmd = [_charm_exe(), subject_id, t1_path(subject_id, project_dir)]
    if use_t2 and status["t2_exists"]:
        cmd.append(t2_path(subject_id, project_dir))
    cmd.append("--forceqform")

    try:
        # encoding/errors explicit: charm's own output isn't guaranteed to be
        # decodable as Windows' default cp1252 (see the same fix in
        # run_pipeline.py / scitas_discovery.py this session).
        proc = subprocess.run(cmd, cwd=sub_dir, capture_output=True, text=True,
                              encoding="utf-8", errors="replace")
    except Exception as e:
        return {"success": False, "error": str(e), "stdout": ""}

    if not os.path.isfile(mesh_path):
        tail = "\n".join((proc.stdout or "").splitlines()[-40:])
        err_tail = "\n".join((proc.stderr or "").splitlines()[-40:])
        return {"success": False, "error": f"charm finished but mesh not found: {mesh_path}\n"
                                            f"--- stdout (tail) ---\n{tail}\n--- stderr (tail) ---\n{err_tail}",
                "stdout": proc.stdout}

    return {"success": True, "cached": False, "m2m_path": m2m_path, "error": None, "stdout": proc.stdout}


def run_charm_on_scitas(subject_id: str, use_t2: bool = True, force: bool = False,
                        project_dir: str = PROJECT_DIR) -> dict:
    """Submits charm to SCITAS (SLURM), blocks until the remote job leaves
    the queue, then scp's the resulting m2m_{id}/ back to this machine.
    Call inside a job_runner background job (blocking — same contract as
    run_charm()). Returns the same shape: {"success", "cached"?, "m2m_path",
    "error", "stdout"}."""
    import scitas_discovery as sd

    status = charm_status(subject_id, project_dir)
    if not status["t1_exists"]:
        return {"success": False, "error": f"T1w not found locally: {t1_path(subject_id, project_dir)}"}

    m2m_path = get_m2m_path(subject_id, project_dir)
    if status["m2m_complete"] and not force:
        return {"success": True, "cached": True, "m2m_path": m2m_path, "error": None, "stdout": ""}

    remote_sub_dir = f"{sd.scitas_scratch_dir()}/derivatives/SimNIBS/sub-{subject_id}"
    remote_m2m = f"{remote_sub_dir}/m2m_{subject_id}"
    remote_mesh = f"{remote_m2m}/{subject_id}.msh"
    remote_t1 = f"{sd.scitas_scratch_dir()}/rawdata/sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz"
    remote_t2 = f"{sd.scitas_scratch_dir()}/rawdata/sub-{subject_id}/anat/sub-{subject_id}_T2w.nii.gz"

    if not sd.remote_path_exists(remote_t1):
        mk = sd.remote_mkdir(f"{sd.scitas_scratch_dir()}/rawdata/sub-{subject_id}/anat")
        if not mk["success"]:
            return {"success": False, "error": f"couldn't create remote rawdata dir: {mk['stderr']}"}
        up = sd.scp_upload(t1_path(subject_id, project_dir), remote_t1, recursive=False)
        if not up["success"]:
            return {"success": False, "error": f"T1w upload to SCITAS failed: {up['stderr']}"}

    if use_t2 and status["t2_exists"] and not sd.remote_path_exists(remote_t2):
        up = sd.scp_upload(t2_path(subject_id, project_dir), remote_t2, recursive=False)
        if not up["success"]:
            return {"success": False, "error": f"T2w upload to SCITAS failed: {up['stderr']}"}

    # force=True: charm_scitas.sbatch only auto-cleans a PARTIAL remote
    # m2m_ folder (mirrors charm's own refusal-to-overwrite behaviour) — a
    # COMPLETE one needs removing here first, same as run_charm() does locally.
    if force and sd.remote_path_exists(remote_m2m):
        rm = sd.remote_rmtree(remote_m2m)
        if not rm["success"]:
            return {"success": False, "error": f"couldn't clear existing remote m2m_ for force re-run: {rm['stderr']}"}

    submit = sd.submit_sbatch(
        script_path=f"{sd.scitas_pipeline_dir()}/charm_scitas.sbatch",
        job_name=f"charm_{subject_id}",
        export_vars={"CHARM_SUBJECT": subject_id},
    )
    if not submit["success"]:
        return {"success": False, "error": f"sbatch submission failed: {submit['error']}"}

    wait = sd.wait_for_job(submit["job_id"], max_wait_s=3 * 3600)
    if not wait["success"]:
        return {"success": False,
                "error": f"SCITAS job {submit['job_id']} did not complete "
                         f"(final state: {wait['final_state']}). {wait['error'] or ''}"}

    if not sd.remote_path_exists(remote_mesh):
        return {"success": False,
                "error": f"SCITAS job {submit['job_id']} completed but mesh not found remotely: {remote_mesh}"}

    os.makedirs(os.path.dirname(m2m_path), exist_ok=True)
    down = sd.scp_download(remote_m2m, os.path.dirname(m2m_path), recursive=True)
    if not down["success"]:
        return {"success": False,
                "error": f"charm succeeded on SCITAS (job {submit['job_id']}) but syncing m2m_{subject_id}/ "
                         f"back failed: {down['stderr']}. Remote result is intact at {remote_m2m}."}

    return {"success": True, "cached": False, "m2m_path": m2m_path, "error": None,
            "stdout": f"SCITAS job {submit['job_id']} completed and results synced back."}

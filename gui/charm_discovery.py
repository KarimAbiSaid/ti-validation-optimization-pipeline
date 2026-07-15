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
Dash callback. SCITAS submission is a placeholder for now — see
job_runner.submit_to_scitas().
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
        proc = subprocess.run(cmd, cwd=sub_dir, capture_output=True, text=True)
    except Exception as e:
        return {"success": False, "error": str(e), "stdout": ""}

    if not os.path.isfile(mesh_path):
        tail = "\n".join((proc.stdout or "").splitlines()[-40:])
        err_tail = "\n".join((proc.stderr or "").splitlines()[-40:])
        return {"success": False, "error": f"charm finished but mesh not found: {mesh_path}\n"
                                            f"--- stdout (tail) ---\n{tail}\n--- stderr (tail) ---\n{err_tail}",
                "stdout": proc.stdout}

    return {"success": True, "cached": False, "m2m_path": m2m_path, "error": None, "stdout": proc.stdout}

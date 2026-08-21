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
    run_charm()). Also checks charm_scitas.sbatch is present and up to
    date on SCITAS first (scitas_discovery.ensure_pipeline_code_synced()),
    re-uploading it if stale, so a file gone missing/out of date on the
    cluster surfaces as a clear local error now instead of a submission
    failure. Returns the same shape: {"success", "cached"?, "m2m_path",
    "error", "stdout"}."""
    import scitas_discovery as sd

    status = charm_status(subject_id, project_dir)
    if not status["t1_exists"]:
        return {"success": False, "error": f"T1w not found locally: {t1_path(subject_id, project_dir)}"}

    m2m_path = get_m2m_path(subject_id, project_dir)
    if status["m2m_complete"] and not force:
        return {"success": True, "cached": True, "m2m_path": m2m_path, "error": None, "stdout": ""}

    # charm_scitas.sbatch is fully self-contained (calls SimNIBS's own charm
    # binary directly, no other tracked pipeline file involved) — checking/
    # re-uploading just this one file self-heals it going stale/missing on
    # SCITAS scratch before that turns into a submission failure.
    code_sync = sd.ensure_pipeline_code_synced(["charm_scitas.sbatch"])
    if code_sync["error"]:
        return {"success": False, "error": f"pipeline code sync failed: {code_sync['error']}"}

    remote_sub_dir = f"{sd.scitas_scratch_dir()}/derivatives/SimNIBS/sub-{subject_id}"
    remote_m2m = f"{remote_sub_dir}/m2m_{subject_id}"
    remote_mesh = f"{remote_m2m}/{subject_id}.msh"
    remote_t1 = f"{sd.scitas_scratch_dir()}/rawdata/sub-{subject_id}/anat/sub-{subject_id}_T1w.nii.gz"
    remote_t2 = f"{sd.scitas_scratch_dir()}/rawdata/sub-{subject_id}/anat/sub-{subject_id}_T2w.nii.gz"

    # remote_path_exists() below can raise RuntimeError if the check itself
    # fails (connection issue) — deliberately left uncaught here: this
    # function is meant to run inside a job_runner background job (see its
    # docstring), which already catches any exception generically and shows
    # it as a clean job error. No extra handling needed on top of that.
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


def batch_submit_charm(subject_ids: list[str], use_t2: bool = True, force: bool = False,
                       project_dir: str = PROJECT_DIR) -> dict:
    """{subject_id: {"success", "job_id", "error", "cached"?, "m2m_path"?}} —
    gets several subjects' charm SLURM jobs INTO the queue using as few ssh
    connections as possible across the whole batch — same batching
    principle as run_discovery.batch_submit() (one shared pipeline-code-sync
    check, one shared sbatch submission connection via
    scitas_discovery.submit_sbatch_batch()), instead of each subject running
    its own full run_charm_on_scitas() sequence (several ssh calls each,
    which — fired at once for N subjects — is enough separate connections
    in a short window to trip SCITAS's own new-connection rate limiting.

    A subject whose m2m_ is already complete (and force=False) is resolved
    immediately as {"success": True, "cached": True, ...} without ever
    touching SCITAS — same as run_charm_on_scitas()'s own early return.

    Only gets each subject's job queued (or immediately resolved) — does
    NOT block waiting for completion. Callers run
    wait_for_submitted_charm_job(job_id, subject_id) in their own
    background job per successfully-submitted subject afterward (see
    pages/head_modeling.py)."""
    import scitas_discovery as sd

    out: dict = {}
    to_submit = []
    for sid in subject_ids:
        status = charm_status(sid, project_dir)
        if not status["t1_exists"]:
            out[sid] = {"success": False, "job_id": None,
                       "error": f"T1w not found locally: {t1_path(sid, project_dir)}"}
            continue
        m2m_path = get_m2m_path(sid, project_dir)
        if status["m2m_complete"] and not force:
            out[sid] = {"success": True, "job_id": None, "error": None,
                       "cached": True, "m2m_path": m2m_path}
            continue
        to_submit.append(sid)

    if not to_submit:
        return out

    # Pipeline source code — one shared check for the whole batch.
    # charm_scitas.sbatch is fully self-contained (calls SimNIBS's own charm
    # binary directly), so this is the only file that ever needs checking here.
    code_sync = sd.ensure_pipeline_code_synced(["charm_scitas.sbatch"])
    if code_sync["error"]:
        for sid in to_submit:
            out[sid] = {"success": False, "job_id": None,
                       "error": f"pipeline code sync failed: {code_sync['error']}"}
        return out

    scratch = sd.scitas_scratch_dir()
    ready_subjects = []
    for sid in to_submit:
        # One subject's remote_path_exists() check failing to even complete
        # (connection issue) shouldn't silently masquerade as "the file's
        # missing" — that previously triggered a needless (and, since the
        # connection is what's actually wrong, likely ALSO failing)
        # re-upload — nor should it abort the whole batch; just this one
        # subject fails with a clear reason and the rest proceed.
        try:
            status = charm_status(sid, project_dir)
            remote_m2m = f"{scratch}/derivatives/SimNIBS/sub-{sid}/m2m_{sid}"
            remote_t1 = f"{scratch}/rawdata/sub-{sid}/anat/sub-{sid}_T1w.nii.gz"
            remote_t2 = f"{scratch}/rawdata/sub-{sid}/anat/sub-{sid}_T2w.nii.gz"

            if not sd.remote_path_exists(remote_t1):
                mk = sd.remote_mkdir(f"{scratch}/rawdata/sub-{sid}/anat")
                if not mk["success"]:
                    out[sid] = {"success": False, "job_id": None,
                               "error": f"couldn't create remote rawdata dir: {mk['stderr']}"}
                    continue
                up = sd.scp_upload(t1_path(sid, project_dir), remote_t1, recursive=False)
                if not up["success"]:
                    out[sid] = {"success": False, "job_id": None,
                               "error": f"T1w upload to SCITAS failed: {up['stderr']}"}
                    continue

            if use_t2 and status["t2_exists"] and not sd.remote_path_exists(remote_t2):
                up = sd.scp_upload(t2_path(sid, project_dir), remote_t2, recursive=False)
                if not up["success"]:
                    out[sid] = {"success": False, "job_id": None,
                               "error": f"T2w upload to SCITAS failed: {up['stderr']}"}
                    continue

            # force=True: charm_scitas.sbatch only auto-cleans a PARTIAL remote
            # m2m_ folder — a COMPLETE one needs removing here first, same as
            # run_charm_on_scitas() does.
            if force and sd.remote_path_exists(remote_m2m):
                rm = sd.remote_rmtree(remote_m2m)
                if not rm["success"]:
                    out[sid] = {"success": False, "job_id": None,
                               "error": f"couldn't clear existing remote m2m_ for force re-run: {rm['stderr']}"}
                    continue

            ready_subjects.append(sid)
        except RuntimeError as e:
            out[sid] = {"success": False, "job_id": None, "error": str(e)}

    if not ready_subjects:
        return out

    submit_jobs = [{
        "script_path": f"{sd.scitas_pipeline_dir()}/charm_scitas.sbatch",
        "job_name": f"charm_{sid}",
        "export_vars": {"CHARM_SUBJECT": sid},
    } for sid in ready_subjects]
    submissions = sd.submit_sbatch_batch(submit_jobs)
    for sid, sub in zip(ready_subjects, submissions):
        if sub["success"]:
            out[sid] = {"success": True, "job_id": sub["job_id"], "error": None}
        else:
            out[sid] = {"success": False, "job_id": None, "error": f"sbatch submission failed: {sub['error']}"}

    return out


def wait_for_submitted_charm_job(job_id: str, subject_id: str, project_dir: str = PROJECT_DIR) -> dict:
    """Blocks on an ALREADY-submitted charm SLURM job (see
    batch_submit_charm()) until it leaves the queue, then scp's the
    resulting m2m_{id}/ back to this machine. Same return shape as
    run_charm_on_scitas()'s successful-submission case."""
    import scitas_discovery as sd

    wait = sd.wait_for_job(job_id, max_wait_s=3 * 3600)
    if not wait["success"]:
        return {"success": False,
                "error": f"SCITAS job {job_id} did not complete "
                         f"(final state: {wait['final_state']}). {wait['error'] or ''}"}

    # remote_path_exists() can raise RuntimeError if the check itself fails
    # (connection issue) — left uncaught: this runs inside a job_runner
    # background job (see pages/head_modeling.py), which already catches
    # any exception generically and shows it as a clean job error.
    scratch = sd.scitas_scratch_dir()
    remote_m2m = f"{scratch}/derivatives/SimNIBS/sub-{subject_id}/m2m_{subject_id}"
    remote_mesh = f"{remote_m2m}/{subject_id}.msh"
    if not sd.remote_path_exists(remote_mesh):
        return {"success": False,
                "error": f"SCITAS job {job_id} completed but mesh not found remotely: {remote_mesh}"}

    m2m_path = get_m2m_path(subject_id, project_dir)
    os.makedirs(os.path.dirname(m2m_path), exist_ok=True)
    down = sd.scp_download(remote_m2m, os.path.dirname(m2m_path), recursive=True)
    if not down["success"]:
        return {"success": False,
                "error": f"charm succeeded on SCITAS (job {job_id}) but syncing m2m_{subject_id}/ "
                         f"back failed: {down['stderr']}. Remote result is intact at {remote_m2m}."}

    return {"success": True, "m2m_path": m2m_path, "error": None}

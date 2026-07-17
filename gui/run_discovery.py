"""
run_discovery.py — Phase 6 (Run Pipeline) data/logic layer, no Dash import.

Runs code/pipeline/run_pipeline.py against an already-generated config JSON
(from Config Generation or generate_configs.py --dry-run) as a local
subprocess. Those configs are written SCITAS-shaped — project_dir=
"/mnt/BIDS_TI_Toolbox", cap_csv/bna_atlas_path likewise (see
config_discovery.py) — so a local run overrides those three fields via
run_pipeline.py's own "--set KEY=VALUE" mechanism, pointed at this machine's
PROJECT_DIR/RESOURCES_DIR, rather than writing a second copy of the JSON to
disk just for local use.

run_pipeline_on_scitas() is the remote equivalent: uploads the config (and
any missing prerequisites — m2m_/registered cap/BNA atlas), submits
simnibs_ti_pipeline.sbatch, blocks on the SLURM queue. Unlike run_local(),
the config's own SCITAS-shaped paths are used as-is — no --set overrides
needed, since project_dir=/mnt/BIDS_TI_Toolbox is exactly where it runs.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys

import config_discovery as cd
from common import PROJECT_DIR, RESOURCES_DIR, get_m2m_path, discover_subjects  # noqa: F401 (re-exported)

PIPELINE_DIR = str(cd.CONFIGS_DIR.parent)
RUN_PIPELINE_PY = os.path.join(PIPELINE_DIR, "run_pipeline.py")

FORCE_SECTIONS = ("recon_all", "charm", "roi_masks", "optimization", "simulation", "analysis", "visualization")


def list_configs() -> list[dict]:
    """Every sub-*_roi-*_ex_*.json in code/pipeline/configs/, across all
    subjects — for the Run page's config picker. Returns
    [{"subject_id", "roi_name", "goal", "filename", "path"}]."""
    out = []
    for sid, fnames in cd.scan_existing_configs().items():
        for fname in fnames:
            m = cd._CONFIG_FNAME_RE.match(fname)
            out.append({
                "subject_id": sid, "roi_name": m.group(2), "goal": m.group(3),
                "filename": fname, "path": str(cd.CONFIGS_DIR / fname),
            })
    return sorted(out, key=lambda r: (r["subject_id"], r["roi_name"], r["goal"]))


def job_base_dir(subject_id: str, project_dir: str = PROJECT_DIR) -> str:
    """Where job_runner.new_job_dir() creates per-run job directories for
    this subject — mirrors charm_discovery's _charm_jobs/ convention."""
    return os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}", "_run_jobs")


def local_overrides(subject_id: str, cap_name: str | None) -> dict:
    """--set overrides that convert a SCITAS-shaped config (project_dir=
    /mnt/BIDS_TI_Toolbox, cap_csv/bna_atlas_path under that same root) into
    one usable on this machine. cap_name: the registered cap's filename stem
    (e.g. "BioSemi32_MNE"), read back from the config's own cap_csv field so
    this doesn't have to guess which cap was used."""
    overrides = {"project_dir": PROJECT_DIR}
    bna_path = os.path.join(RESOURCES_DIR, "atlases", "BN_Atlas_246_1mm.nii.gz")
    if os.path.isfile(bna_path):
        overrides["bna_atlas_path"] = bna_path
    if cap_name:
        overrides["cap_csv"] = os.path.join(get_m2m_path(subject_id), "eeg_positions", f"{cap_name}.csv")
    return overrides


def _tail(text: str | None, n: int = 300) -> str:
    lines = (text or "").splitlines()
    if len(lines) <= n:
        return text or ""
    return f"... [truncated — showing last {n} of {len(lines)} lines] ...\n" + "\n".join(lines[-n:])


def run_local(config_path: str, force_sections: list[str] | None = None) -> dict:
    """Runs run_pipeline.py as a subprocess against config_path, with
    project_dir/cap_csv/bna_atlas_path overridden to this machine's local
    paths (see local_overrides()). Blocking — call inside a job_runner
    background job, not directly from a Dash callback. Returns {"success",
    "returncode", "stdout", "stderr"} (stdout/stderr tailed to the last 300
    lines each)."""
    with open(config_path) as f:
        cfg = json.load(f)
    subject_id = cfg["subject_id"]
    cap_name = None
    if cfg.get("cap_csv"):
        cap_name = os.path.splitext(os.path.basename(cfg["cap_csv"]))[0]

    overrides = local_overrides(subject_id, cap_name)

    cmd = [sys.executable, RUN_PIPELINE_PY, "--config", config_path]
    cmd += ["--set"] + [f"{k}={v}" for k, v in overrides.items()]
    if force_sections:
        cmd += ["--force"] + list(force_sections)

    # run_pipeline.py's print()s include non-ASCII characters (→, ✓, ═ box
    # drawing, ...). When stdout/stderr aren't an interactive console (true
    # here — capture_output pipes them), Python picks the child's stdout
    # encoding from PYTHONIOENCODING, falling back to the OS locale's
    # preferred encoding (cp1252 on Windows, not UTF-8) — which crashes the
    # child with UnicodeEncodeError the first time it prints one of those
    # characters. Force UTF-8 for the child explicitly rather than relying
    # on whatever encoding happens to be ambient.
    child_env = {**os.environ, "PYTHONIOENCODING": "utf-8"}

    try:
        proc = subprocess.run(cmd, cwd=PIPELINE_DIR, capture_output=True, text=True,
                              encoding="utf-8", errors="replace", env=child_env)
    except Exception as e:
        return {"success": False, "returncode": None, "stdout": "", "stderr": str(e)}

    return {
        "success": proc.returncode == 0,
        "returncode": proc.returncode,
        "stdout": _tail(proc.stdout),
        "stderr": _tail(proc.stderr),
    }


def run_pipeline_on_scitas(config_path: str, force_sections: list[str] | None = None) -> dict:
    """Submits config_path to SCITAS via simnibs_ti_pipeline.sbatch, blocks
    until the SLURM job leaves the queue. Call inside a job_runner
    background job (blocking — same contract as run_local()). Returns
    {"success", "job_id", "error", "stdout"}.

    The config's own paths are already SCITAS-shaped (project_dir=
    /mnt/BIDS_TI_Toolbox, etc. — see config_discovery.py), so unlike
    run_local() no --set overrides are needed. What IS needed: the config
    file itself only lives locally under code/pipeline/configs/, so it's
    uploaded first; likewise the subject's m2m_/registered cap/the BNA
    atlas are uploaded if not already present on SCITAS scratch (a fresh
    subject may only have been charm'd locally, or its cap only registered
    locally). Results stay on SCITAS scratch — sync them back via the Data
    Directory page if you want local copies, same as any other SCITAS data.
    """
    import scitas_discovery as sd

    with open(config_path) as f:
        cfg = json.load(f)
    subject_id = cfg["subject_id"]
    scratch = sd.scitas_scratch_dir()
    container_root = cfg["project_dir"]  # e.g. "/mnt/BIDS_TI_Toolbox" — only exists INSIDE the
                                          # Apptainer container (see simnibs_ti_pipeline.sbatch's
                                          # --bind); existence checks/uploads run over plain SSH on
                                          # the bare login node, so container-rooted paths from the
                                          # config (cap_csv, bna_atlas_path) must be translated to
                                          # their real scratch location first, or they'll always
                                          # look "missing" even when correctly in place.

    def _to_scratch(container_path: str) -> str:
        return scratch + container_path[len(container_root):]

    # Config file itself
    fname = os.path.basename(config_path)
    remote_configs_dir = f"{sd.scitas_pipeline_dir()}/configs"
    remote_config_path = f"{remote_configs_dir}/{fname}"
    mk = sd.remote_mkdir(remote_configs_dir)
    if not mk["success"]:
        return {"success": False, "job_id": None, "error": f"couldn't create remote configs dir: {mk['stderr']}"}
    up = sd.scp_upload(config_path, remote_config_path, recursive=False)
    if not up["success"]:
        return {"success": False, "job_id": None, "error": f"config upload failed: {up['stderr']}"}

    # m2m_ — required; upload from local if SCITAS doesn't have it yet
    remote_m2m = f"{scratch}/derivatives/SimNIBS/sub-{subject_id}/m2m_{subject_id}"
    if not sd.remote_path_exists(remote_m2m):
        local_m2m = get_m2m_path(subject_id)
        if not os.path.isdir(local_m2m):
            return {"success": False, "job_id": None,
                    "error": f"m2m_{subject_id}/ not found locally or on SCITAS — run Head Modeling first."}
        sd.remote_mkdir(os.path.dirname(remote_m2m))
        up = sd.scp_upload(local_m2m, os.path.dirname(remote_m2m), recursive=True)
        if not up["success"]:
            return {"success": False, "job_id": None, "error": f"m2m_ upload failed: {up['stderr']}"}

    # Registered cap CSV — upload if the config references one SCITAS doesn't have
    cap_csv = cfg.get("cap_csv")
    if cap_csv:
        remote_cap_csv = _to_scratch(cap_csv)
        if not sd.remote_path_exists(remote_cap_csv):
            cap_name = os.path.splitext(os.path.basename(cap_csv))[0]
            local_cap_csv = os.path.join(get_m2m_path(subject_id), "eeg_positions", f"{cap_name}.csv")
            if not os.path.isfile(local_cap_csv):
                return {"success": False, "job_id": None,
                        "error": f"registered cap not found locally or on SCITAS: {cap_name} "
                                 f"— run Cap Registration first."}
            sd.remote_mkdir(os.path.dirname(remote_cap_csv))
            up = sd.scp_upload(local_cap_csv, remote_cap_csv, recursive=False)
            if not up["success"]:
                return {"success": False, "job_id": None, "error": f"cap upload failed: {up['stderr']}"}

    # BNA atlas — shared resource, not per-subject; upload once if missing
    bna_atlas_path = cfg.get("bna_atlas_path")
    if bna_atlas_path:
        remote_bna = _to_scratch(bna_atlas_path)
        if not sd.remote_path_exists(remote_bna):
            local_bna = os.path.join(RESOURCES_DIR, "atlases", "BN_Atlas_246_1mm.nii.gz")
            if not os.path.isfile(local_bna):
                return {"success": False, "job_id": None,
                        "error": f"BNA atlas not found locally or on SCITAS: {bna_atlas_path}"}
            sd.remote_mkdir(os.path.dirname(remote_bna))
            up = sd.scp_upload(local_bna, remote_bna, recursive=False)
            if not up["success"]:
                return {"success": False, "job_id": None, "error": f"BNA atlas upload failed: {up['stderr']}"}

    export_vars = {"PIPELINE_CONFIGS": remote_config_path}
    if force_sections:
        export_vars["PIPELINE_EXTRA_ARGS"] = "--force " + " ".join(force_sections)

    submit = sd.submit_sbatch(
        script_path=f"{sd.scitas_pipeline_dir()}/simnibs_ti_pipeline.sbatch",
        job_name=f"ti_{subject_id}",
        export_vars=export_vars,
    )
    if not submit["success"]:
        return {"success": False, "job_id": None, "error": f"sbatch submission failed: {submit['error']}"}

    wait = sd.wait_for_job(submit["job_id"], max_wait_s=48 * 3600)
    if not wait["success"]:
        return {"success": False, "job_id": submit["job_id"],
                "error": f"SCITAS job {submit['job_id']} did not complete "
                         f"(final state: {wait['final_state']}). {wait['error'] or ''}"}

    return {"success": True, "job_id": submit["job_id"], "error": None,
            "stdout": f"SCITAS job {submit['job_id']} completed. Results are on SCITAS scratch under "
                      f"derivatives/SimNIBS/sub-{subject_id}/TIoptimization/ — use the Data Directory "
                      f"page to sync them back for local viewing."}

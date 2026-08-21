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


def _config_prereq_relpaths(cfg: dict) -> list[str]:
    """Relative-to-PROJECT_DIR paths (forward-slash, matching the remote
    scratch layout — see scitas_discovery.batch_upload) for every small
    file this config needs uploaded: registered cap CSV, ROI/non-ROI
    masks, ROI/non-ROI constraint-group masks. Shared by
    run_pipeline_on_scitas()'s own per-config upload step and
    batch_upload_prereqs()'s combined multi-config upload, so both stay in
    sync about what "this config's prerequisites" actually means."""
    subject_id = cfg["subject_id"]
    paths = []
    cap_csv = cfg.get("cap_csv")
    if cap_csv:
        cap_name = os.path.splitext(os.path.basename(cap_csv))[0]
        paths.append(f"derivatives/SimNIBS/sub-{subject_id}/m2m_{subject_id}/eeg_positions/{cap_name}.csv")
    for roi_cfg in (cfg.get("roi"), cfg.get("non_roi")):
        if roi_cfg and roi_cfg.get("name"):
            paths.append(f"derivatives/SimNIBS/sub-{subject_id}/roi/"
                         f"sub-{subject_id}_label-{roi_cfg['name']}_mask.nii.gz")
    for groups_key in ("non_roi_hard_constraint_groups", "roi_hard_constraint_groups"):
        for grp in cfg.get("optimizer", {}).get(groups_key, []):
            mask_name = grp.get("mask_name")
            if mask_name:
                paths.append(f"derivatives/SimNIBS/sub-{subject_id}/roi/"
                             f"sub-{subject_id}_label-{mask_name}_mask.nii.gz")
    return paths


def batch_upload_prereqs(config_paths: list[str]) -> dict:
    """Uploads every listed config's own config JSON + registered cap +
    ROI/non-ROI/constraint masks in ONE ssh connection
    (scitas_discovery.batch_upload) instead of one scp call per file per
    config. Built for submitting several configs at once on the Run
    Pipeline page: doing this one config at a time, as
    run_pipeline_on_scitas() does on its own, opens a burst of separate
    connections large enough to trip SCITAS's own new-connection rate
    limiting on its login node (seen live as scp failing with
    "kex_exchange_identification: Connection closed by remote host").

    Call this ONCE before launching the per-config run_pipeline_on_scitas()
    jobs, then pass skip_prereq_upload=True to each so they don't
    redundantly re-upload what's already here. m2m_ and the BNA atlas are
    deliberately NOT included: m2m_ can be large enough that folding it
    into one combined transfer risks turning several small independent
    retriable uploads into one large fragile one, and the BNA atlas is a
    single shared file almost always already present after the first run
    ever — both keep their existing per-config if-missing check.

    Returns {"success", "uploaded", "error"} (scitas_discovery.batch_upload's
    own shape)."""
    import scitas_discovery as sd

    relpaths = []
    for config_path in config_paths:
        with open(config_path) as f:
            cfg = json.load(f)
        relpaths.append(f"code/pipeline/configs/{os.path.basename(config_path)}")
        relpaths += _config_prereq_relpaths(cfg)

    return sd.batch_upload(PROJECT_DIR, relpaths)


def run_pipeline_on_scitas(config_path: str, force_sections: list[str] | None = None,
                           skip_prereq_upload: bool = False) -> dict:
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
    locally). ROI/non-ROI mask .nii.gz files are uploaded too when
    SCITAS doesn't have them — required (this fails with a clear error
    instead of a wasted queue slot) for masks Section 1 can't rebuild
    itself (Allen-sourced, or a Mask Generation "Combine Existing Masks"
    output — no labels/bna_labels to build from), opportunistic otherwise.
    Results stay on SCITAS scratch — sync them back via the Data
    Directory page if you want local copies, same as any other SCITAS data.

    Also checks (unless skip_prereq_upload) that the pipeline source files
    THIS job's flags actually need are present and up to date on SCITAS —
    see scitas_discovery.ensure_pipeline_code_synced() — and re-uploads any
    that are stale before submitting, so a file that's gone missing/out of
    date on the cluster surfaces as a clear local error now instead of a
    confusing remote import failure partway through a submitted job.

    skip_prereq_upload: set True when the caller already uploaded the
    config/cap/masks (and pipeline code) itself (batch_upload_prereqs()
    plus its own code-sync call, called once across several configs at
    once — see the Run Pipeline page's multi-select submit / batch_submit()).
    The existence/rebuildability validation (does a mask exist anywhere,
    can Section 1 rebuild it, else error) still always runs — only the
    redundant individual re-upload of something that's already there is
    skipped. m2m_ and the BNA atlas are unaffected either way, since
    batch_upload_prereqs() never includes them.
    """
    import scitas_discovery as sd

    with open(config_path) as f:
        cfg = json.load(f)
    subject_id = cfg["subject_id"]

    if not skip_prereq_upload:
        code_sync = sd.ensure_pipeline_code_synced(sd.required_pipeline_files(cfg.get("flags", {})))
        if code_sync["error"]:
            return {"success": False, "job_id": None,
                    "error": f"pipeline code sync failed: {code_sync['error']}"}

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
    if not skip_prereq_upload:
        mk = sd.remote_mkdir(remote_configs_dir)
        if not mk["success"]:
            return {"success": False, "job_id": None,
                    "error": f"couldn't create remote configs dir: {mk['stderr']}"}
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

    # Registered cap CSV — always re-uploaded (overwriting any remote copy).
    # Unlike m2m_/BNA below, this file is tiny (a few KB), so there's no cost
    # to always refreshing it — a skip-if-exists check here previously let a
    # stale remote cap silently survive a local re-registration (e.g. after
    # fixing which rows count as real electrodes), since the check only
    # looked at existence, not content.
    cap_csv = cfg.get("cap_csv")
    if cap_csv:
        remote_cap_csv = _to_scratch(cap_csv)
        cap_name = os.path.splitext(os.path.basename(cap_csv))[0]
        local_cap_csv = os.path.join(get_m2m_path(subject_id), "eeg_positions", f"{cap_name}.csv")
        if not os.path.isfile(local_cap_csv):
            if not sd.remote_path_exists(remote_cap_csv):
                return {"success": False, "job_id": None,
                        "error": f"registered cap not found locally or on SCITAS: {cap_name} "
                                 f"— run Cap Registration first."}
        elif not skip_prereq_upload:
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

    # ROI / non-ROI masks — just the .nii.gz, nothing else needed (Section 1's
    # skip-if-exists check only looks at the file itself). Two cases:
    #   - local copy exists: ALWAYS re-upload, overwriting any remote copy.
    #     A skip-if-remote-exists check here would let a stale/broken remote
    #     mask (e.g. saved by an earlier run that hit the since-fixed "Allen
    #     mask silently rebuilt as empty" bug in create_roi_masks()) survive
    #     forever even after the local mask is regenerated correctly — masks
    #     are small (a few hundred KB), so there's no real cost to always
    #     refreshing, same reasoning as the registered-cap-CSV re-upload above.
    #   - local copy missing: fall back to checking remote. If neither exists
    #     and Section 1 can't build it either (Allen-sourced, or a Combine
    #     Existing Masks output — no labels/bna_labels), error out now with a
    #     clear message before wasting a queue slot instead of failing later
    #     inside the SLURM job.
    for roi_cfg in (cfg.get("roi"), cfg.get("non_roi")):
        if not roi_cfg or not roi_cfg.get("name"):
            continue
        name = roi_cfg["name"]
        remote_mask = f"{scratch}/derivatives/SimNIBS/sub-{subject_id}/roi/sub-{subject_id}_label-{name}_mask.nii.gz"
        local_mask = os.path.join(PROJECT_DIR, "derivatives", "SimNIBS", f"sub-{subject_id}", "roi",
                                  f"sub-{subject_id}_label-{name}_mask.nii.gz")

        if os.path.isfile(local_mask):
            if not skip_prereq_upload:
                sd.remote_mkdir(os.path.dirname(remote_mask))
                up = sd.scp_upload(local_mask, remote_mask, recursive=False)
                if not up["success"]:
                    return {"success": False, "job_id": None,
                            "error": f"mask '{name}' upload failed: {up['stderr']}"}
            continue

        if sd.remote_path_exists(remote_mask):
            continue   # local missing but remote already has a copy — trust it

        can_rebuild_remotely = bool(roi_cfg.get("labels")) or bool(roi_cfg.get("bna_labels"))
        if can_rebuild_remotely:
            continue  # Section 1 will build it remotely from labels/bna_labels
        return {"success": False, "job_id": None,
                "error": f"mask '{name}' not found locally or on SCITAS, and this ROI/non-ROI has "
                         f"no labels/bna_labels for Section 1 to build it from (Allen-sourced, or a "
                         f"combined mask) — generate it via Mask Generation first."}

    # Subgroup hard-constraint masks — both non_roi_hard_constraint_groups
    # (mean TI must stay BELOW max_mean_V_m) and roi_hard_constraint_groups
    # (mean TI must stay ABOVE min_mean_V_m) reference masks the exact same
    # way, so one helper covers both. Only mask_name-based groups need this
    # — bna_labels-based groups need no upload since Section 1 rebuilds the
    # warped BNA atlas itself. Unlike ROI/non-ROI above, a mask_name group
    # can NEVER be rebuilt remotely (no labels/bna_labels fallback exists
    # for it in run_pipeline.py). Same always-reupload-if-local-exists
    # reasoning as ROI/non-ROI above — a locally-missing mask (with no
    # remote copy either) is a hard error.
    for groups_key, label in (("non_roi_hard_constraint_groups", "non-ROI"),
                              ("roi_hard_constraint_groups", "ROI")):
        for grp in cfg.get("optimizer", {}).get(groups_key, []):
            mask_name = grp.get("mask_name")
            if not mask_name:
                continue
            remote_mask = (f"{scratch}/derivatives/SimNIBS/sub-{subject_id}/roi/"
                          f"sub-{subject_id}_label-{mask_name}_mask.nii.gz")
            local_mask = os.path.join(PROJECT_DIR, "derivatives", "SimNIBS", f"sub-{subject_id}", "roi",
                                      f"sub-{subject_id}_label-{mask_name}_mask.nii.gz")

            if os.path.isfile(local_mask):
                if not skip_prereq_upload:
                    sd.remote_mkdir(os.path.dirname(remote_mask))
                    up = sd.scp_upload(local_mask, remote_mask, recursive=False)
                    if not up["success"]:
                        return {"success": False, "job_id": None,
                                "error": f"{label} constraint mask '{mask_name}' upload failed: {up['stderr']}"}
                continue

            if sd.remote_path_exists(remote_mask):
                continue   # local missing but remote already has a copy — trust it

            return {"success": False, "job_id": None,
                    "error": f"{label} constraint mask '{mask_name}' (group '{grp.get('name', '?')}') "
                             f"not found locally or on SCITAS — generate it via Mask Generation first."}

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


def batch_submit(config_paths: list[str], force_sections: list[str] | None = None) -> dict:
    """{config_path: {"success", "job_id", "error", "subject_id",
    "code_synced"}} — gets several configs' SLURM jobs INTO the queue
    using as few ssh connections as possible across the whole batch,
    instead of each config running its own full run_pipeline_on_scitas()
    sequence (prereq upload, m2m_/BNA checks, sbatch submission — several
    ssh calls each). Firing that sequence once per config, all at once for
    N configs, was still enough separate connections in a short window to
    trip SCITAS's own new-connection rate limiting even after
    batch_upload_prereqs() already batched the small-file upload step —
    this batches the rest of it too (existence checks, submission).

    Also checks the pipeline SOURCE CODE itself once for the whole batch
    (scitas_discovery.ensure_pipeline_code_synced()) before touching any
    config-specific files — self-heals a required .py/.sbatch file gone
    stale/missing on SCITAS scratch, so a submission never fails with a
    confusing remote import error over something that could've been fixed
    locally first. "code_synced" on every result records what (if
    anything) got re-uploaded this way.

    Only gets each config's job queued — does NOT block waiting for
    completion. Callers run wait_for_submitted_job(job_id, subject_id) in
    their own background job per successfully-submitted config afterward
    (see pages/run_optimization.py)."""
    import scitas_discovery as sd

    configs = {}
    for path in config_paths:
        with open(path) as f:
            configs[path] = json.load(f)

    out = {path: {"success": False, "job_id": None, "error": None, "subject_id": cfg["subject_id"],
                  "code_synced": []}
           for path, cfg in configs.items()}

    # 0) Pipeline source code — one shared check for the WHOLE batch (union
    #    of every config's own flags, so e.g. one recon_all-flagged config
    #    in an otherwise-normal batch still gets recon_all_scitas.sbatch
    #    synced too). Self-heals a file gone stale/missing on SCITAS scratch
    #    before it turns into a confusing remote import error partway
    #    through a submitted job — see scitas_discovery.
    #    ensure_pipeline_code_synced(). "code_synced" is stamped onto every
    #    entry (not a separate top-level field) so any caller inspecting
    #    any one result can see what happened for the whole batch.
    all_flags: dict = {}
    for cfg in configs.values():
        all_flags.update({k: v for k, v in cfg.get("flags", {}).items() if v})
    code_sync = sd.ensure_pipeline_code_synced(sd.required_pipeline_files(all_flags))
    if code_sync["error"]:
        for path in out:
            out[path]["error"] = f"pipeline code sync failed: {code_sync['error']}"
        return out
    for path in out:
        out[path]["code_synced"] = code_sync["synced"]

    # 1) Config JSON + registered cap + ROI/non-ROI/constraint masks — one
    #    shared connection for every config's small files.
    up = batch_upload_prereqs(config_paths)
    if not up["success"]:
        for path in out:
            out[path]["error"] = f"prerequisite upload failed: {up['error']}"
        return out

    scratch = sd.scitas_scratch_dir()

    # 2) m2m_/BNA existence — one shared connection across every unique
    #    subject/atlas path involved. m2m_ upload itself, if actually
    #    missing, still falls back to an individual scp per subject (rare,
    #    and large enough that folding it into the small-file tar batch
    #    above isn't worth the risk of one huge fragile transfer).
    check_paths = set()
    for cfg in configs.values():
        subject_id = cfg["subject_id"]
        check_paths.add(f"{scratch}/derivatives/SimNIBS/sub-{subject_id}/m2m_{subject_id}")
        if cfg.get("bna_atlas_path"):
            container_root = cfg["project_dir"]
            check_paths.add(scratch + cfg["bna_atlas_path"][len(container_root):])
    try:
        existence = sd.remote_paths_exist(sorted(check_paths))
    except RuntimeError as e:
        for path in out:
            out[path]["error"] = f"m2m_/BNA existence check failed: {e}"
        return out

    for path, cfg in configs.items():
        if out[path]["error"]:
            continue
        subject_id = cfg["subject_id"]
        remote_m2m = f"{scratch}/derivatives/SimNIBS/sub-{subject_id}/m2m_{subject_id}"
        if not existence.get(remote_m2m, False):
            local_m2m = get_m2m_path(subject_id)
            if not os.path.isdir(local_m2m):
                out[path]["error"] = f"m2m_{subject_id}/ not found locally or on SCITAS — run Head Modeling first."
                continue
            sd.remote_mkdir(os.path.dirname(remote_m2m))
            m2m_up = sd.scp_upload(local_m2m, os.path.dirname(remote_m2m), recursive=True)
            if not m2m_up["success"]:
                out[path]["error"] = f"m2m_ upload failed: {m2m_up['stderr']}"
                continue

        bna_atlas_path = cfg.get("bna_atlas_path")
        if bna_atlas_path:
            container_root = cfg["project_dir"]
            remote_bna = scratch + bna_atlas_path[len(container_root):]
            if not existence.get(remote_bna, False):
                local_bna = os.path.join(RESOURCES_DIR, "atlases", "BN_Atlas_246_1mm.nii.gz")
                if not os.path.isfile(local_bna):
                    out[path]["error"] = f"BNA atlas not found locally or on SCITAS: {bna_atlas_path}"
                    continue
                sd.remote_mkdir(os.path.dirname(remote_bna))
                bna_up = sd.scp_upload(local_bna, remote_bna, recursive=False)
                if not bna_up["success"]:
                    out[path]["error"] = f"BNA atlas upload failed: {bna_up['stderr']}"
                    continue

    # 3) sbatch submission — one shared connection for every config that's
    #    still error-free.
    ready_paths = [p for p in config_paths if out[p]["error"] is None]
    if ready_paths:
        submit_jobs = []
        for path in ready_paths:
            fname = os.path.basename(path)
            remote_config_path = f"{sd.scitas_pipeline_dir()}/configs/{fname}"
            export_vars = {"PIPELINE_CONFIGS": remote_config_path}
            if force_sections:
                export_vars["PIPELINE_EXTRA_ARGS"] = "--force " + " ".join(force_sections)
            submit_jobs.append({
                "script_path": f"{sd.scitas_pipeline_dir()}/simnibs_ti_pipeline.sbatch",
                "job_name": f"ti_{configs[path]['subject_id']}",
                "export_vars": export_vars,
            })
        submissions = sd.submit_sbatch_batch(submit_jobs)
        for path, sub in zip(ready_paths, submissions):
            if sub["success"]:
                out[path]["success"] = True
                out[path]["job_id"] = sub["job_id"]
            else:
                out[path]["error"] = f"sbatch submission failed: {sub['error']}"

    return out


def wait_for_submitted_job(job_id: str, subject_id: str) -> dict:
    """Blocks on an ALREADY-submitted SLURM job (see batch_submit()) until
    it leaves the queue. Same return shape as run_pipeline_on_scitas()'s
    successful-submission case, so the Run Pipeline page's result
    rendering doesn't need to know which path a job took to get started."""
    import scitas_discovery as sd

    wait = sd.wait_for_job(job_id, max_wait_s=48 * 3600)
    if not wait["success"]:
        return {"success": False, "job_id": job_id,
                "error": f"SCITAS job {job_id} did not complete "
                         f"(final state: {wait['final_state']}). {wait['error'] or ''}"}
    return {"success": True, "job_id": job_id, "error": None,
            "stdout": f"SCITAS job {job_id} completed. Results are on SCITAS scratch under "
                      f"derivatives/SimNIBS/sub-{subject_id}/TIoptimization/ — use the Data Directory "
                      f"page to sync them back for local viewing."}

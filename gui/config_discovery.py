"""
config_discovery.py — Phase 5 (Config Generation) discovery/build logic for
the GUI.

Pure Python, no Dash import. Builds the same PipelineConfig-shaped JSON that
code/pipeline/generate_configs.py's build_config() writes, so files land in
the real code/pipeline/configs/ folder ready for run_pipeline.py / SCITAS
submission — this module never touches generate_configs.py's own hardcoded
SUBJECTS/ROIS/etc. lists, it is an independent way of producing the same
output shape.
"""
from __future__ import annotations

import json
import math
import os
import re
from pathlib import Path

import discovery      # atlas registry, region lut, path/readiness checks
import cap_discovery  # cap listing + registered-cap path resolution
from common import discover_subjects, get_m2m_path, PROJECT_DIR  # noqa: F401 (re-exported)

# The project path INSIDE the SimNIBS Apptainer container on SCITAS, where
# run_pipeline.py actually runs — always this path regardless of the local
# machine's PROJECT_DIR (which is only used here for GUI-side discovery/
# reads against the local mount). Matches generate_configs.py's own
# PROJECT_DIR / BNA_ATLAS_PATH constants.
SCITAS_PROJECT_DIR = "/mnt/BIDS_TI_Toolbox"
BNA_ATLAS_PATH = f"{SCITAS_PROJECT_DIR}/code/resources/atlases/BN_Atlas_246_1mm.nii.gz"

GOALS = ["mean", "focality"]

CONFIGS_DIR = Path(__file__).resolve().parent.parent / "pipeline" / "configs"
SBATCH_ENV  = CONFIGS_DIR.parent / "subject_configs.sh"

# Optimizer/electrode fields not exposed as editable in the GUI are fixed at
# these defaults (matches generate_configs.py's OPTIMIZER_BASE/ELECTRODE).
OPTIMIZER_DEFAULTS = {
    "postproc":               "max_TI",
    "focality_threshold":     [0.24, 0.3],
    "cpus":                   8,
    "max_non_roi_elements":   150_000,
    "hard_roi_constraint":    True,
    "no_adjacent_electrodes": True,
    "non_roi_hard_constraint_groups": [],
}
ELECTRODE_DEFAULTS = {
    "shape":             "ellipse",
    "dimensions":        [19.5, 19.5],
    "gel_thickness":     1.0,
    "current_mA":        2.0,
    "max_total_current": 4.0,
    "n_electrodes":      4,
}
FLAGS_DEFAULTS = {
    "run_recon_all":     False,
    "run_charm":         False,
    "run_roi_masks":     True,
    "run_optimization":  True,
    "run_simulation":    False,
    "run_analysis":      True,
    "run_visualization": True,
}


def config_filename(subject_id: str, roi_name: str, goal: str) -> str:
    return f"sub-{subject_id}_roi-{roi_name}_ex_{goal}.json"


# ═════════════════════════════════════════════════════════════════════════════
# ROI / non-ROI: region-picker output -> roi JSON sub-dict
# ═════════════════════════════════════════════════════════════════════════════

def uses_allen(atlas_name: str) -> bool:
    return atlas_name == "Allen"


def build_roi_dict(name: str, atlas_name: str, label_ids: dict) -> dict:
    """label_ids: {display_name: int_id} from the Atlas->Region picker.
    Mirrors generate_configs.py's _roi_dict(), plus the Allen case.

    - BNA: bna_labels (Section 1 warps BN_Atlas_246_1mm.nii.gz itself).
    - SimNIBS / FreeSurfer: labels (both resolve through cfg.aseg_path —
      charm's own labeling.nii.gz for SimNIBS, true aparc+aseg.mgz for
      FreeSurfer — see run_pipeline.py's create_roi_masks()).
    - Allen: run_pipeline.py's Section 1 has no code path that reads Allen's
      label space, so labels is left empty here. The mask for this name
      MUST already exist on disk (built via the Mask Generation page — same
      _label-{name}_mask.nii.gz naming as PipelineConfig.mask_path()) so
      Section 1's own skip-if-exists check picks it up unchanged instead of
      trying (and failing/silently producing garbage) to build it fresh.
    """
    if atlas_name == "BNA":
        return {"name": name, "labels": {}, "bna_labels": label_ids}
    if uses_allen(atlas_name):
        return {"name": name, "labels": {}}
    return {"name": name, "labels": label_ids}


def roi_mask_exists(subject_id: str, name: str, mask_type: str,
                     project_dir: str = PROJECT_DIR) -> bool:
    """Whether the on-disk mask this (subject, name) pair needs already
    exists locally — only meaningful for Allen-sourced ROI/non-ROI (see
    build_roi_dict); mask_type: "ROI" or "non-ROI"."""
    return os.path.exists(discovery.expected_mask_path(subject_id, mask_type, name, project_dir))


# ═════════════════════════════════════════════════════════════════════════════
# Per-subject readiness
# ═════════════════════════════════════════════════════════════════════════════

def subject_readiness(subject_id: str, roi_atlas: str, roi_name: str,
                       non_roi_atlas: str | None, non_roi_name: str | None,
                       cap_path: str | None, project_dir: str = PROJECT_DIR) -> dict:
    """{"ready": bool, "missing": [str, ...]} for one subject, across every
    prerequisite Config Generation needs for THIS roi/non-roi/cap
    combination: m2m + atlas-specific paths (BNA/SimNIBS/FreeSurfer), or the
    pre-built mask file (Allen); registered cap CSV if cap_path is set (None
    = default Okamoto cap, always available once m2m exists)."""
    missing: list[str] = []

    if uses_allen(roi_atlas):
        if not roi_mask_exists(subject_id, roi_name, "ROI", project_dir):
            missing.append(f"ROI mask not on disk (Allen) — run Mask Generation first: {roi_name}")
    else:
        roi_check = discovery.full_atlas_check(roi_atlas, subject_id, project_dir)
        if not roi_check["ready"]:
            missing += [c["label"] for c in roi_check["checks"] if not c["exists"]]

    if non_roi_atlas and non_roi_name:
        if uses_allen(non_roi_atlas):
            if not roi_mask_exists(subject_id, non_roi_name, "non-ROI", project_dir):
                missing.append(f"non-ROI mask not on disk (Allen) — run Mask Generation first: {non_roi_name}")
        else:
            non_roi_check = discovery.full_atlas_check(non_roi_atlas, subject_id, project_dir)
            if not non_roi_check["ready"]:
                missing += [c["label"] for c in non_roi_check["checks"] if not c["exists"]]

    if cap_path:
        reg_path = cap_discovery.registered_cap_path(subject_id, cap_path, project_dir)
        if not os.path.isfile(reg_path):
            missing.append(f"cap not registered — run Cap Registration first: {cap_discovery.cap_stem(cap_path)}")

    seen = set()
    missing_unique = [m for m in missing if not (m in seen or seen.add(m))]
    return {"ready": not missing_unique, "missing": missing_unique}


def readiness_matrix(subject_ids: list[str], roi_atlas: str, roi_name: str,
                      non_roi_atlas: str | None, non_roi_name: str | None,
                      cap_path: str | None, project_dir: str = PROJECT_DIR) -> dict:
    return {
        sid: subject_readiness(sid, roi_atlas, roi_name, non_roi_atlas, non_roi_name, cap_path, project_dir)
        for sid in subject_ids
    }


# ═════════════════════════════════════════════════════════════════════════════
# Config assembly — mirrors generate_configs.py's build_config()
# ═════════════════════════════════════════════════════════════════════════════

def build_config(subject_id: str, roi_name: str, roi_atlas: str, roi_label_ids: dict,
                  non_roi_name: str | None, non_roi_atlas: str | None, non_roi_label_ids: dict | None,
                  goal: str, optimizer_overrides: dict, electrode_overrides: dict,
                  flags: dict, cap_path: str | None) -> dict:
    """Builds one PipelineConfig-shaped dict, ready to json.dump() straight
    into code/pipeline/configs/. project_dir/cap_csv/bna_atlas_path always
    point at SCITAS_PROJECT_DIR — the JSON is consumed by run_pipeline.py
    running inside the Apptainer container there, never on this machine."""
    cfg: dict = {"subject_id": subject_id, "project_dir": SCITAS_PROJECT_DIR}

    uses_bna = (roi_atlas == "BNA") or (non_roi_atlas == "BNA")
    if uses_bna:
        cfg["bna_atlas_path"] = BNA_ATLAS_PATH

    cfg["flags"]      = {**FLAGS_DEFAULTS, **flags}
    cfg["roi"]        = build_roi_dict(roi_name, roi_atlas, roi_label_ids)
    cfg["non_roi"]    = build_roi_dict(non_roi_name, non_roi_atlas, non_roi_label_ids) if non_roi_name else None
    cfg["optimizer"]  = {"goal": goal, **{**OPTIMIZER_DEFAULTS, **optimizer_overrides}}
    cfg["electrode"]  = {**ELECTRODE_DEFAULTS, **electrode_overrides}
    cfg["simulation"] = {"simulate_mode": "optimized"}

    if cap_path:
        cap_name = cap_discovery.cap_stem(cap_path)
        cfg["cap_csv"] = (
            f"{SCITAS_PROJECT_DIR}/derivatives/SimNIBS/sub-{subject_id}"
            f"/m2m_{subject_id}/eeg_positions/{cap_name}.csv"
        )
    return cfg


def _slurm_time_limit(n_combos: int) -> str:
    """Same estimate as generate_configs.py: 30 min leadfield + 5 min per
    ROI/goal combo, +1 h buffer."""
    minutes = 30 + n_combos * 5
    hours = math.ceil(minutes / 60) + 1
    return f"{hours:02d}:00:00"


def generate_configs(subject_ids: list[str], roi_name: str, roi_atlas: str, roi_label_ids: dict,
                      non_roi_name: str | None, non_roi_atlas: str | None, non_roi_label_ids: dict | None,
                      goals: list[str], optimizer_overrides: dict, electrode_overrides: dict,
                      flags: dict, cap_path: str | None, force: bool = False) -> list[dict]:
    """Writes JSON config files to the real code/pipeline/configs/ folder —
    same location/format generate_configs.py itself writes to. Does not
    touch generate_configs.py's own SUBJECTS/ROIS/etc. lists. Returns
    [{"subject_id", "goal", "filename", "status", "path"}], one entry per
    (subject, goal) combination; status is "written", "skipped" (exists,
    force=False), or "error: ..."."""
    CONFIGS_DIR.mkdir(parents=True, exist_ok=True)
    results = []
    for sid in subject_ids:
        for goal in goals:
            fname = config_filename(sid, roi_name, goal)
            fpath = CONFIGS_DIR / fname
            if fpath.exists() and not force:
                results.append({"subject_id": sid, "goal": goal, "filename": fname,
                                 "status": "skipped", "path": str(fpath)})
                continue
            try:
                cfg = build_config(sid, roi_name, roi_atlas, roi_label_ids,
                                    non_roi_name, non_roi_atlas, non_roi_label_ids,
                                    goal, optimizer_overrides, electrode_overrides,
                                    flags, cap_path)
                fpath.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                results.append({"subject_id": sid, "goal": goal, "filename": fname,
                                 "status": "written", "path": str(fpath)})
            except Exception as e:
                results.append({"subject_id": sid, "goal": goal, "filename": fname,
                                 "status": f"error: {e}", "path": str(fpath)})
    return results


# ═════════════════════════════════════════════════════════════════════════════
# subject_configs.sh — rebuilt from every config actually on disk, not just
# this session's, so repeated Generate runs (possibly with different ROI
# names across sessions) never silently drop earlier entries.
# ═════════════════════════════════════════════════════════════════════════════

_CONFIG_FNAME_RE = re.compile(r"^sub-(.+)_roi-(.+)_ex_(mean|focality)\.json$")


def scan_existing_configs() -> dict:
    """{subject_id: [config_filename, ...]} for every sub-*_roi-*_ex_*.json
    file currently in code/pipeline/configs/ — the real on-disk state,
    regardless of which GUI session/ROI produced each one."""
    out: dict[str, list[str]] = {}
    if not CONFIGS_DIR.is_dir():
        return out
    for fname in sorted(os.listdir(CONFIGS_DIR)):
        m = _CONFIG_FNAME_RE.match(fname)
        if m:
            out.setdefault(m.group(1), []).append(fname)
    return out


def write_sbatch_env() -> str:
    """(Re)writes code/pipeline/subject_configs.sh from every config file
    currently on disk in code/pipeline/configs/ — the same file
    submit_multiroi_scitas.sh sources. Same format as generate_configs.py's
    own write_sbatch_env(), except scope is "whatever's really in the
    folder" rather than one fixed Python ROIS dict."""
    by_subject = scan_existing_configs()
    base = "${BASE_CONFIG_DIR}"
    lines = [
        "# AUTO-GENERATED by the Config Generation GUI page — do not edit by hand.",
        "# Reflects every config in code/pipeline/configs/ as of the last Generate run.",
        "",
        "declare -A SUBJECT_CONFIGS",
        "declare -A SUBJECT_TIMELIMITS",
        "",
    ]
    for sid in sorted(by_subject):
        fnames = by_subject[sid]
        cfg_paths = [f"  {base}/{fname}" for fname in fnames]
        first = cfg_paths[0].lstrip()
        rest = cfg_paths[1:]
        if rest:
            lines.append(f'SUBJECT_CONFIGS["{sid}"]="{first} \\')
            lines.append(" \\\n".join(rest) + '"')
        else:
            lines.append(f'SUBJECT_CONFIGS["{sid}"]="{first}"')
        lines.append(f'SUBJECT_TIMELIMITS["{sid}"]="{_slurm_time_limit(len(fnames))}"')
        lines.append("")

    content = "\n".join(lines)
    with open(SBATCH_ENV, "w", encoding="utf-8", newline="\n") as f:
        f.write(content)
    return str(SBATCH_ENV)


def existing_configs(subject_id: str) -> list[dict]:
    """Config JSON files already on disk for this subject — for the
    'Existing Configs' browser, mirroring Mask Generation's existing-masks
    table. Returns [{"filename", "path", "mtime", "size_bytes"}]."""
    out = []
    by_subject = scan_existing_configs()
    for fname in by_subject.get(subject_id, []):
        path = CONFIGS_DIR / fname
        out.append({
            "filename": fname,
            "path": str(path),
            "mtime": path.stat().st_mtime,
            "size_bytes": path.stat().st_size,
        })
    return out

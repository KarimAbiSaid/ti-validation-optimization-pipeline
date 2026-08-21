"""
comparison_discovery.py — Phase 4 (comparison view) data/logic layer, no
Dash import. Runs multiple FEM/TI setups (leadfield or one-off, each
possibly a different subject/cap/montage) against a shared ROI/non-ROI
label, for side-by-side comparison.

This module is just the "resolve inputs + run one setup row" layer — it
delegates the actual computation entirely to fem_discovery.py (which
already handles both leadfield and one-off FEM). Multi-row orchestration
(which rows run synchronously vs. as background jobs, polling) lives in the
page, since job_runner.py's status-file contract is naturally a UI-polling
concern, not a data-layer one.
"""
from __future__ import annotations

import json
import os
import re

import cap_discovery as cd
import discovery
import fem_discovery as fd

# Matches create_masks.py's create_roi() output — sub-{id}_label-{name}_mask.nii.gz
# — the only pattern resolve_mask() actually checks (see its docstring).
_LABEL_MASK_RE = re.compile(r"^sub-.+_label-(?P<name>.+)_mask\.nii\.gz$")
from common import PROJECT_DIR, discover_subjects, get_m2m_path  # noqa: F401 (re-exported)


def resolve_mask(subject_id: str, label_or_path: str | None, project_dir: str = PROJECT_DIR) -> str | None:
    """Accept either a full path or a mask label (e.g. 'hippocampus') —
    resolves to sub-{id}_label-{label}_mask.nii.gz. Mirrors
    compare_ti_montages.py's _resolve_mask(), minus its sys.exit()."""
    if not label_or_path:
        return None
    if os.path.isfile(label_or_path):
        return label_or_path
    candidate = os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}",
                             "roi", f"sub-{subject_id}_label-{label_or_path}_mask.nii.gz")
    return candidate if os.path.isfile(candidate) else None


def atlas_region_options(atlas_name: str, subject_id: str) -> list[dict] | None:
    """{"label", "value"} options for a Region dropdown, drawn from that
    atlas's full region list (discovery.build_lut — same source Phase 1's
    region picker uses), NOT filtered to regions that already have a
    generated mask. None if the atlas has no lut (BNA) or it can't load —
    caller falls back to plain label text entry in that case.
    value = the region's own name; the page combines it with the atlas
    name to guess the expected mask label ("{region}_{atlas}"), matching
    Phase 1's auto-suffixing convention — the actual mask may have been
    saved under a different custom name if it unions multiple regions, so
    this is a convenience prefill, not a guarantee."""
    try:
        lut = discovery.build_lut(atlas_name, subject_id)
    except Exception:
        return None
    if not lut:
        return None
    return [{"label": f"{name}  (id {rid})", "value": name}
            for rid, name in sorted(lut.items(), key=lambda kv: kv[1])]


def list_existing_names(subject_ids: list[str], project_dir: str = PROJECT_DIR) -> dict[str, set[str]]:
    """{name: {subject_ids that have a sub-{id}_label-{name}_mask.nii.gz}} —
    scans real files across the given subjects (the create_roi() pattern
    resolve_mask() actually checks), for browsing what's already there
    rather than what's theoretically possible from an atlas."""
    out: dict[str, set[str]] = {}
    for sid in subject_ids:
        for m in discovery.existing_masks(sid, project_dir):
            match = _LABEL_MASK_RE.match(m["filename"])
            if match:
                out.setdefault(match.group("name"), set()).add(sid)
    return out


def list_exhaustive_runs(subject_id: str, project_dir: str = PROJECT_DIR) -> list[dict]:
    """{"run_dir", "path"} for every sub-{id}/TIoptimization/*/ folder that
    has a finished exhaustive_results.json — for the Comparison page's "fill
    row from optimization result" picker. A subject can have several runs
    (different ROI, different search settings, reruns), so this deliberately
    doesn't guess which one — it lists every one that actually finished, run
    folder names first (newest run-{timestamp} suffix sorts first)."""
    ti_dir = os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}", "TIoptimization")
    if not os.path.isdir(ti_dir):
        return []
    out = []
    for name in sorted(os.listdir(ti_dir), reverse=True):
        result_path = os.path.join(ti_dir, name, "exhaustive_results.json")
        if os.path.isfile(result_path):
            out.append({"run_dir": name, "path": result_path})
    return out


def load_best_montage(result_path: str) -> dict | None:
    """{"ch1_plus", "ch1_minus", "ch1_current_mA", "ch2_plus", "ch2_minus",
    "ch2_current_mA"} from an exhaustive_results.json's top-level
    best_montage — None if the file's missing/unreadable or has no
    best_montage. Falls back to the shared "current_mA" key for either
    channel's current when the newer per-channel keys aren't present
    (older result files)."""
    try:
        with open(result_path) as f:
            data = json.load(f)
    except Exception:
        return None
    best = data.get("best_montage")
    if not best:
        return None
    shared = best.get("current_mA")
    return {
        "ch1_plus": best.get("ch1_plus", ""), "ch1_minus": best.get("ch1_minus", ""),
        "ch1_current_mA": best.get("ch1_current_mA", shared),
        "ch2_plus": best.get("ch2_plus", ""), "ch2_minus": best.get("ch2_minus", ""),
        "ch2_current_mA": best.get("ch2_current_mA", shared),
    }


def _resolve_oneoff_electrode(subject_id: str, cap_name: str | None, value, project_dir: str = PROJECT_DIR):
    """value: either an electrode name (resolved against cap_name's
    registered positions) or a raw 'x, y, z' string. Returns (coord, error)."""
    text = str(value).strip() if value is not None else ""
    if not text:
        return None, "empty"

    parts = text.split(",")
    if len(parts) == 3:
        try:
            return [float(p.strip()) for p in parts], None
        except ValueError:
            pass  # not raw coords — fall through, treat as a name

    if not cap_name:
        return None, f"'{text}' isn't raw x,y,z and no cap was given to resolve a name"
    cap_path = cd.registered_cap_path(subject_id, cap_name, project_dir)
    if not os.path.isfile(cap_path):
        return None, f"cap '{cap_name}' not registered for sub-{subject_id}"
    elec = cd.registered_electrode_positions(cap_path)
    if text not in elec["names"]:
        return None, f"'{text}' not found in registered cap '{cap_name}'"
    idx = elec["names"].index(text)
    return elec["coords"][idx].tolist(), None


def resolve_extra_region_masks(subject_id: str, region_labels: list[str] | None,
                               project_dir: str = PROJECT_DIR) -> dict[str, str]:
    """{region_label: mask_path} for whichever of the given labels actually
    resolve to a real mask file for this subject — a label that doesn't
    (mask never generated for this particular subject) is silently dropped
    rather than failing the row, since "extra regions" is meant as an
    opportunistic bonus stat, not a hard requirement like the main ROI."""
    if not region_labels:
        return {}
    out = {}
    for region_label in region_labels:
        mask_path = resolve_mask(subject_id, region_label, project_dir)
        if mask_path:
            out[region_label] = mask_path
    return out


def run_leadfield_row(
    subject_id: str, cap_name: str, roi_label: str, non_roi_label: str | None,
    ch1_plus: str, ch1_minus: str, ch1_current_mA: float,
    ch2_plus: str, ch2_minus: str, ch2_current_mA: float,
    label: str, project_dir: str = PROJECT_DIR,
    electrode_shape: str | None = None, electrode_dimensions: list | None = None,
    electrode_gel_thickness: float | None = None,
    extra_region_labels: list[str] | None = None,
) -> dict:
    """Runs synchronously (fast/algebraic) — call directly, no background job needed.

    electrode_shape/dimensions/gel_thickness: which cached leadfield variant
    to use for this cap (see fem_discovery.leadfield_status) — leave unset to
    get "any variant for this cap" (today's coarse behaviour, from before
    per-settings variants existed).

    extra_region_labels: additional mask labels to also report TI_mean/
    TI_max for (see resolve_extra_region_masks) — a label with no mask for
    this subject is dropped, not an error."""
    roi_mask = resolve_mask(subject_id, roi_label, project_dir)
    if not roi_mask:
        return {"success": False, "error": f"ROI mask not found for sub-{subject_id}, label '{roi_label}'"}
    non_roi_mask = resolve_mask(subject_id, non_roi_label, project_dir) if non_roi_label else None
    extra_masks = resolve_extra_region_masks(subject_id, extra_region_labels, project_dir)

    status = fd.leadfield_status(subject_id, cap_name, project_dir,
                                 shape=electrode_shape, dimensions=electrode_dimensions,
                                 gel_thickness=electrode_gel_thickness)
    if not status["exists"]:
        return {"success": False,
                "error": f"No leadfield for sub-{subject_id} / {cap_name}: {status['hdf5_path']}"}

    return fd.compute_ti(
        subject_id=subject_id, hdf5_path=status["hdf5_path"],
        roi_mask_path=roi_mask, non_roi_mask_path=non_roi_mask,
        ch1_plus=ch1_plus, ch1_minus=ch1_minus, ch1_current_mA=ch1_current_mA,
        ch2_plus=ch2_plus, ch2_minus=ch2_minus, ch2_current_mA=ch2_current_mA,
        electrode_dims=electrode_dimensions, label=label, project_dir=project_dir,
        extra_mask_paths=extra_masks,
    )


def run_oneoff_row(
    subject_id: str, cap_name: str | None, roi_label: str, non_roi_label: str | None,
    ch1_plus: str, ch1_minus: str, ch1_current_mA: float,
    ch2_plus: str, ch2_minus: str, ch2_current_mA: float,
    label: str, project_dir: str = PROJECT_DIR,
    extra_region_labels: list[str] | None = None,
) -> dict:
    """Real FEM solves — meant to run inside a background job (job_runner.py),
    not called directly from a Dash callback. Resolves each electrode cell
    (name-via-cap or raw x,y,z) then delegates to fem_discovery.run_one_off_fem.

    extra_region_labels: see run_leadfield_row — same opportunistic,
    drop-if-missing extra-stats mechanism."""
    roi_mask = resolve_mask(subject_id, roi_label, project_dir)
    if not roi_mask:
        return {"success": False, "error": f"ROI mask not found for sub-{subject_id}, label '{roi_label}'"}
    non_roi_mask = resolve_mask(subject_id, non_roi_label, project_dir) if non_roi_label else None
    extra_masks = resolve_extra_region_masks(subject_id, extra_region_labels, project_dir)

    coords = {}
    for key, val in [("ch1_plus", ch1_plus), ("ch1_minus", ch1_minus),
                      ("ch2_plus", ch2_plus), ("ch2_minus", ch2_minus)]:
        coord, err = _resolve_oneoff_electrode(subject_id, cap_name, val, project_dir)
        if err:
            return {"success": False, "error": f"{key}: {err}"}
        coords[key] = coord

    return fd.run_one_off_fem(
        subject_id=subject_id, roi_mask_path=roi_mask, non_roi_mask_path=non_roi_mask,
        ch1_plus_coord=coords["ch1_plus"], ch1_minus_coord=coords["ch1_minus"], ch1_current_mA=ch1_current_mA,
        ch2_plus_coord=coords["ch2_plus"], ch2_minus_coord=coords["ch2_minus"], ch2_current_mA=ch2_current_mA,
        label=label, project_dir=project_dir, extra_mask_paths=extra_masks,
    )

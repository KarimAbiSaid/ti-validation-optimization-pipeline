"""
analysis_discovery.py — Analysis page data/compute layer, no Dash import.

Reads TI results that have ALREADY been computed (a SimNIBS .msh with a
max_TI elementdata field, optionally E_ch1/E_ch2 too — see
viz_discovery.list_existing_meshes() for what counts as "already
computed") and derives stats/distributions from them, without ever
re-running a FEM solve or touching a leadfield.

ROI selection here is NOT limited to whichever mask(s) a given run's
original config happened to include (that's all its ti_field_{label}.npz
sidecar has) — instead this classifies mesh element centroids against ANY
curated ROI mask on demand, the same nifti-lookup technique
run_pipeline.py's own _mask_nifti_to_elements() uses for extra_rois, so
picking a region here is exactly as free-form as Mask Generation's own
region list. The cost is one mesh load per selected run (no FEM solve —
just reading elementdata + geometry back off disk), not zero, but nowhere
near the leadfield/FEM costs elsewhere in this app.

E_ch1/E_ch2 (needed for the field-direction stat) are only ever saved by
Comparison-page runs (compare_ti_montages.py bakes them into the saved
mesh as elementdata) — a straight Run Pipeline optimization result never
writes them, so field_angle_deg() simply isn't available for those; the
page is expected to check ef1/ef2 for None before offering it, not assume
every run has it.
"""
from __future__ import annotations

import json
import os

import numpy as np
import nibabel as nib

import comparison_discovery as cx
import viz_discovery as vz
from common import PROJECT_DIR

TISSUE_TAGS = [1, 2]  # WM, GM only -- matches run_pipeline.py / compare_ti_montages.py

# "What's already computed" is exactly viz_discovery's own notion of a
# render-able mesh / available region -- re-exported rather than
# reimplemented so the Analysis page's run/region pickers stay in sync
# with Render Figure's and the Slice Viewer's for free.
list_existing_meshes = vz.list_existing_meshes
available_regions = vz.available_regions


def _mask_nifti_to_elements(mask_path: str, centroids: np.ndarray) -> np.ndarray:
    """Mirrors run_pipeline.py's _mask_nifti_to_elements() exactly (nearest-
    voxel lookup of each element centroid against a binary NIfTI mask) —
    reimplemented here rather than imported, since code/pipeline/ isn't
    meant to be imported as a library beyond config.py-style helpers (see
    common.py's sys.path note)."""
    mask_img = nib.load(mask_path)
    mask_data = np.asarray(mask_img.dataobj) > 0
    aff_inv = np.linalg.inv(mask_img.affine)
    ones = np.ones((len(centroids), 1))
    vox = (aff_inv @ np.hstack([centroids, ones]).T).T[:, :3]
    idx = np.round(vox).astype(int)
    sh = mask_data.shape
    in_bounds = ((idx[:, 0] >= 0) & (idx[:, 0] < sh[0]) &
                (idx[:, 1] >= 0) & (idx[:, 1] < sh[1]) &
                (idx[:, 2] >= 0) & (idx[:, 2] < sh[2]))
    out = np.zeros(len(centroids), dtype=bool)
    out[in_bounds] = mask_data[idx[in_bounds, 0], idx[in_bounds, 1], idx[in_bounds, 2]]
    return out


def load_run_field(msh_path: str) -> dict:
    """One already-computed mesh's per-element data, WM+GM only (matches
    run_pipeline.py/compare_ti_montages.py's own TISSUE_TAGS convention).

    Returns {"ti": (N,), "ef1": (N,3)|None, "ef2": (N,3)|None — None when
    this mesh never saved per-channel vectors, see module docstring —
    "centroids": (N,3), "volumes": (N,) mm³, "tags": (N,)}.

    Raises ValueError if the mesh has no max_TI field at all (shouldn't
    happen for anything list_existing_meshes() returned, but guards
    against a stale/hand-edited mesh)."""
    from simnibs import mesh_io

    m = mesh_io.read_msh(msh_path)
    tissue_mask = np.isin(m.elm.tag1, TISSUE_TAGS)

    def _field(name_options):
        return next((d for d in reversed(m.elmdata) if d.field_name in name_options), None)

    ti_field = _field(("max_TI", "TI_max"))
    if ti_field is None:
        raise ValueError(f"No max_TI field in {msh_path}")
    ef1_field = _field(("E_ch1",))
    ef2_field = _field(("E_ch2",))

    # Cropping to the same TISSUE_TAGS and boolean-masking the full mesh's
    # elmdata by the same tag criterion yield elements in the same relative
    # order — the exact assumption run_pipeline.py/compare_ti_montages.py
    # already rely on, not something new introduced here.
    m_tissue = m.crop_mesh(tags=TISSUE_TAGS)
    tags = m_tissue.elm.tag1
    nodes = m_tissue.nodes.node_coord
    conn = m_tissue.elm.node_number_list[:, :4] - 1
    centroids = nodes[conn].mean(axis=1)
    v0, v1, v2, v3 = nodes[conn[:, 0]], nodes[conn[:, 1]], nodes[conn[:, 2]], nodes[conn[:, 3]]
    volumes = np.abs(np.einsum('ni,ni->n', v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0

    return {
        "ti":        ti_field.value[tissue_mask],
        "ef1":       ef1_field.value[tissue_mask] if ef1_field is not None else None,
        "ef2":       ef2_field.value[tissue_mask] if ef2_field is not None else None,
        "centroids": centroids,
        "volumes":   volumes,
        "tags":      tags,
    }


def analyze_run(subject_id: str, msh_path: str, roi_names: list[str],
                project_dir: str = PROJECT_DIR) -> dict:
    """Load one run's field once and classify it against every requested
    ROI name. Returns load_run_field()'s dict plus "roi_masks": {name:
    bool array} (a name with no mask on disk for this subject is silently
    dropped from roi_masks — same opportunistic convention resolve_mask()
    callers use elsewhere), or {"error": str} if the mesh itself couldn't
    be read."""
    try:
        data = load_run_field(msh_path)
    except Exception as e:
        return {"error": str(e)}
    roi_masks = {}
    for name in roi_names:
        mpath = cx.resolve_mask(subject_id, name, project_dir)
        if mpath:
            roi_masks[name] = _mask_nifti_to_elements(mpath, data["centroids"])
    data["roi_masks"] = roi_masks
    return data


def roi_stats(values: np.ndarray, volumes: np.ndarray | None = None) -> dict | None:
    """mean/median/p90/p95/p99/max — volume-weighted mean when volumes is
    given (matches the pipeline's own _vol_mean convention), else a plain
    mean. None for an empty selection (region has zero matching elements
    for this subject/run)."""
    if values.size == 0:
        return None
    out = {
        "n_elements": int(values.size),
        "mean":   float(np.average(values, weights=volumes)) if volumes is not None else float(values.mean()),
        "median": float(np.median(values)),
        "p90":    float(np.percentile(values, 90)),
        "p95":    float(np.percentile(values, 95)),
        "p99":    float(np.percentile(values, 99)),
        "max":    float(values.max()),
    }
    if volumes is not None:
        out["volume_mm3"] = float(volumes.sum())
    return out


def pct_above_threshold(values: np.ndarray, threshold: float,
                        volumes: np.ndarray | None = None) -> float | None:
    """% of this region's volume (or element count, if volumes not given)
    at/above threshold V/m. None for an empty selection."""
    if values.size == 0:
        return None
    mask = values >= threshold
    if volumes is not None:
        total = volumes.sum()
        return float(volumes[mask].sum() / total * 100) if total > 0 else 0.0
    return float(mask.sum() / values.size * 100)


def field_angle_deg(ef1: np.ndarray, ef2: np.ndarray) -> np.ndarray:
    """Per-element angle (degrees) between the two channels' field vectors
    — physically meaningful for TI (it's what drives the interference/
    modulation depth, on top of the two channels' own magnitudes already
    captured by max_TI itself). 0 deg where either vector is exactly zero
    (avoids a divide-by-zero; such elements barely matter to the montage
    anyway)."""
    dot = np.einsum('ni,ni->n', ef1, ef2)
    n1 = np.linalg.norm(ef1, axis=1)
    n2 = np.linalg.norm(ef2, axis=1)
    denom = n1 * n2
    cos = np.divide(dot, denom, out=np.zeros_like(dot), where=denom > 0)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


def load_json_stats(path: str) -> dict | None:
    """Whatever's in a ti_stats_{label}.json / exhaustive_results.json —
    the free-tier numbers (mean/max/focality/search diagnostics), no array
    loading. None if unreadable."""
    try:
        with open(path) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


def existing_stats_path(subject_id: str, mesh_entry: dict, project_dir: str = PROJECT_DIR) -> str | None:
    """The free-tier JSON sidecar for one list_existing_meshes() entry —
    ti_stats_{label}.json for a "comparison" source, exhaustive_results.json
    for a "TIoptimization" source (same run_dir as its own label). None if
    it's gone missing since list_existing_meshes() was called."""
    sub_dir = os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}")
    if mesh_entry["source"] == "comparison":
        p = os.path.join(sub_dir, "comparison", f"ti_stats_{mesh_entry['label']}.json")
    else:
        p = os.path.join(sub_dir, "TIoptimization", mesh_entry["label"], "exhaustive_results.json")
    return p if os.path.isfile(p) else None


# ═════════════════════════════════════════════════════════════════════════════
# Histogram / KDE — JSON-serializable so build_analysis() below can run as a
# job_runner background job (its status file is plain JSON; raw numpy arrays
# never leave this module) and the page just plots whatever comes back.
# ═════════════════════════════════════════════════════════════════════════════

N_HIST_BINS = 60
N_KDE_POINTS = 200


def histogram_data(values: np.ndarray, edges: np.ndarray,
                   volumes: np.ndarray | None = None) -> dict:
    """{"edges": [...], "counts"|"volume_mm3": [...]} over SHARED bin edges
    (see build_analysis() — every trace in one plot must share edges to be
    honestly comparable). Weighted by volume when given (each element
    contributes its own mm³ to its bin, not a flat +1), matching the same
    volume-weighting used for the summary stats."""
    weights = volumes if volumes is not None else None
    counts, _ = np.histogram(values, bins=edges, weights=weights)
    key = "volume_mm3" if volumes is not None else "counts"
    return {"edges": edges.tolist(), key: counts.tolist()}


def kde_data(values: np.ndarray, x_min: float, x_max: float,
            volumes: np.ndarray | None = None) -> dict | None:
    """{"x": [...], "y": [...]} density curve over a SHARED x-range (same
    reasoning as histogram_data — comparable overlays need the same
    x-axis). None if there are too few distinct values for scipy's KDE to
    fit (e.g. an empty or near-constant selection)."""
    from scipy.stats import gaussian_kde

    if values.size < 2 or np.ptp(values) == 0:
        return None
    try:
        kde = gaussian_kde(values, weights=volumes)
    except Exception:
        return None
    x = np.linspace(x_min, x_max, N_KDE_POINTS)
    return {"x": x.tolist(), "y": kde(x).tolist()}


SUM_RUN_KEY = "__sum__"  # pseudo run_key for the "sum across selected runs" trace


def build_analysis(selections: list[dict], threshold_v_per_m: float | None = None,
                   compute_angle: bool = True, include_whole_brain: bool = True,
                   include_sum: bool = False, project_dir: str = PROJECT_DIR) -> dict:
    """The actual background-job entry point (pass to job_runner.
    start_local_job) — loads every selected (subject, run) mesh ONCE,
    classifies it against every requested ROI, and returns pure-JSON
    results ready to plot, so nothing downstream needs simnibs/numpy again.

    selections: [{"subject_id", "msh_path", "run_label", "roi_names": [...]},
    ...] — one entry per subject+run the user picked (usually 1, but several
    for a multi-subject/multi-run overlay).

    include_whole_brain: also give every run a "Whole brain" pseudo-region —
    the SAME WM+GM elements load_run_field() already returns unmasked, no
    separate lookup needed — as a background reference alongside whatever
    ROI(s) were picked, so you can see whether a region's field genuinely
    stands out above the overall background rather than just eyeballing it.

    include_sum: also add one SUM_RUN_KEY pseudo-run, one region entry per
    region name that appeared in at least one selected run, each built by
    concatenating that region's raw values/volumes (and E-field vectors,
    only if EVERY contributing run has them) across every run that matched
    it — i.e. "as if every selected run's matching elements were one big
    combined run" for that region, not just their stats averaged together.
    Respects include_whole_brain (no "Whole brain" sum entry when that's
    off, keeping the sum consistent with what's otherwise shown).

    Returns {"success": True, "runs": {run_key: {"error": str} | {
      "roi": {roi_name: {"stats": {...}, "hist": {...}, "kde": {...}|None,
      "pct_above_threshold": float|None, "n_runs": int (SUM_RUN_KEY entries
      only — how many selected runs actually contributed to it)}, ...},
      "angle_stats"/"angle_hist": only when compute_angle and this run has
      E_ch1/E_ch2, keyed the same way per-region}}}, "vmax_used": float} or
      {"success": False, "error": str}.

    run_key = f"{subject_id}::{run_label}" (unique per selection, safe to
    use as a dict key / trace name) — or SUM_RUN_KEY for the sum pseudo-run."""
    def _region_entry(values, volumes, ef1, ef2):
        stats = roi_stats(values, volumes)
        if stats is None:
            return {"stats": None}
        entry = {
            "stats": stats,
            "hist": histogram_data(values, edges, volumes),
            "kde": kde_data(values, 0.0, vmax_used, volumes),
        }
        if threshold_v_per_m is not None:
            entry["pct_above_threshold"] = pct_above_threshold(values, threshold_v_per_m, volumes)
        if compute_angle and ef1 is not None:
            angle = field_angle_deg(ef1, ef2)
            entry["angle_stats"] = roi_stats(angle, volumes)
            a_edges = np.linspace(0.0, 180.0, N_HIST_BINS + 1)
            entry["angle_hist"] = histogram_data(angle, a_edges, volumes)
        return entry

    def _region_candidates(data):
        """{region_name: (values, volumes, ef1, ef2)} for one loaded run —
        the shared source both the per-run loop and the sum aggregation
        below draw from, so they can never disagree on what counts."""
        out = {}
        if include_whole_brain:
            out["Whole brain"] = (data["ti"], data["volumes"], data["ef1"], data["ef2"])
        for name, mask in data["roi_masks"].items():
            if mask.any():
                out[name] = (data["ti"][mask], data["volumes"][mask],
                            data["ef1"][mask] if data["ef1"] is not None else None,
                            data["ef2"][mask] if data["ef2"] is not None else None)
        return out

    try:
        loaded = {}
        for sel in selections:
            run_key = f"{sel['subject_id']}::{sel['run_label']}"
            data = analyze_run(sel["subject_id"], sel["msh_path"], sel["roi_names"], project_dir)
            loaded[run_key] = data

        # Shared bin edges/KDE range across EVERY region (whole-brain +
        # every roi) of EVERY selected run — otherwise overlaid traces
        # would each pick their own range and the comparison would be
        # visually misleading. Percentile-capped (99.5) rather than the
        # true max so one outlier element doesn't compress every other
        # trace into a sliver of the plot.
        all_values = []
        for data in loaded.values():
            if "error" in data:
                continue
            for values, _volumes, _ef1, _ef2 in _region_candidates(data).values():
                all_values.append(values)
        vmax_used = float(np.percentile(np.concatenate(all_values), 99.5)) if all_values else 1.0
        vmax_used = max(vmax_used, 1e-6)
        edges = np.linspace(0.0, vmax_used, N_HIST_BINS + 1)

        runs_out = {}
        sum_parts: dict[str, list] = {}  # region name -> [(values, volumes, ef1, ef2), ...]
        for run_key, data in loaded.items():
            if "error" in data:
                runs_out[run_key] = {"error": data["error"]}
                continue
            candidates = _region_candidates(data)
            roi_out = {name: _region_entry(*parts) for name, parts in candidates.items()}
            runs_out[run_key] = {"roi": roi_out}
            if include_sum:
                for name, parts in candidates.items():
                    sum_parts.setdefault(name, []).append(parts)

        if include_sum and sum_parts:
            sum_roi_out = {}
            for name, parts in sum_parts.items():
                values = np.concatenate([p[0] for p in parts])
                volumes = np.concatenate([p[1] for p in parts])
                ef1_list = [p[2] for p in parts]
                ef2_list = [p[3] for p in parts]
                if all(e is not None for e in ef1_list):
                    ef1, ef2 = np.concatenate(ef1_list), np.concatenate(ef2_list)
                else:
                    ef1 = ef2 = None
                entry = _region_entry(values, volumes, ef1, ef2)
                entry["n_runs"] = len(parts)
                sum_roi_out[name] = entry
            runs_out[SUM_RUN_KEY] = {"roi": sum_roi_out}

        return {"success": True, "runs": runs_out, "vmax_used": vmax_used}
    except Exception as e:
        return {"success": False, "error": str(e)}

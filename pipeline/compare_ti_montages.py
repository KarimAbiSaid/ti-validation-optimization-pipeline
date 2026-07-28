"""
compare_ti_montages.py — Compare two TI stimulation setups using a cached leadfield.

Uses pure algebraic TI computation (no additional FEM simulations needed).
Each "setup" is a 4-electrode TI configuration: two channels, each with its own
positive electrode, negative electrode, and current amplitude.

Outputs (per setup):
  - NPZ file  : ti_all, ef1, ef2, centroids, tag1, roi_mask, non_roi_mask
  - JSON file : ROI/non-ROI/whole-brain statistics
  - Console   : side-by-side comparison table

Python usage
------------
  from compare_ti_montages import compute_ti_setup, compare_setups

  # Load shared resources once, then reuse across setups
  resources = load_subject_resources(
      subject_id   = "025",
      project_dir  = "D:/MINDS_Project_Karim/BIDS_TI_Toolbox",
      cap_name     = "BioSemi32_MNE",
      roi_mask     = "D:/.../sub-025_label-hippo_r_phg_mask.nii.gz",
      non_roi_mask = "D:/.../sub-025_label-sfg_orb_cg_mask.nii.gz",
  )

  res_a = compute_ti_setup(resources,
      ch1_plus="FC1", ch1_minus="P4", ch1_current_mA=2.0,
      ch2_plus="T7",  ch2_minus="P8", ch2_current_mA=2.0,
      electrode_dims=[14.0, 14.0],
      label="pipeline_best", output_dir="D:/.../comparison",
  )
  res_b = compute_ti_setup(resources,
      ch1_plus="FC2", ch1_minus="P3", ch1_current_mA=2.0,
      ch2_plus="T8",  ch2_minus="P7", ch2_current_mA=2.0,
      electrode_dims=[14.0, 14.0],
      label="manual",         output_dir="D:/.../comparison",
  )
  compare_setups(res_a, res_b)

CLI usage
---------
  python compare_ti_montages.py \\
      --subject 025 \\
      --project-dir "D:/MINDS_Project_Karim/BIDS_TI_Toolbox" \\
      --cap BioSemi32_MNE \\
      --roi hippo_r_phg \\
      --non-roi sfg_orb_cg \\
      --setup-a FC1 P4 2.0 T7 P8 2.0 \\
      --setup-b FC2 P3 2.0 T8 P7 2.0 \\
      --label-a pipeline_best \\
      --label-b manual \\
      --electrode-dims 14 14
"""

from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np


# ── Tissue tags (SimNIBS convention) ─────────────────────────────────────────
TISSUE_TAGS = [1, 2]             # WM, GM only — analysis restricted to brain tissue
# LEGACY: TISSUE_TAGS = [1, 2, 3, 4, 5]   # WM, GM, CSF, Skull, Scalp
TAG_NAMES   = {1: "WM", 2: "GM", 3: "CSF", 4: "Skull", 5: "Scalp"}


# ═════════════════════════════════════════════════════════════════════════════
# Public API
# ═════════════════════════════════════════════════════════════════════════════

class SubjectResources:
    """Shared, pre-loaded data for a subject (leadfield + masks)."""
    def __init__(
        self,
        leadfield: np.ndarray,
        mesh,
        idx_lf: dict,
        centroids: np.ndarray,
        tissue_mask: np.ndarray,
        tags: np.ndarray,
        roi_elm_mask: np.ndarray,
        non_roi_elm_mask: np.ndarray,
        lf_params: dict,
        elm_volumes: np.ndarray,
    ):
        self.leadfield        = leadfield         # (N_elec-1, M, 3)
        self.mesh             = mesh
        self.idx_lf           = idx_lf            # {name -> lf index}
        self.centroids        = centroids         # (M, 3) mm — tissue elements only
        self.tissue_mask      = tissue_mask       # boolean, length = all elements
        self.tags             = tags              # (M,) int — tissue type
        self.roi_elm_mask     = roi_elm_mask      # (M,) bool
        self.non_roi_elm_mask = non_roi_elm_mask  # (M,) bool
        self.lf_params        = lf_params         # dict from _params.json
        self.elm_volumes      = elm_volumes       # (M,) float — tetrahedron volumes in mm³

def _vol_mean(values, volumes):
    return float((values * volumes).sum() / volumes.sum())

def _vol_mean_capped(values, volumes, pct=99):
    if len(values) == 0:
        return np.nan
    cap = float(np.percentile(values, pct))
    v   = np.minimum(values, cap)
    return float((v * volumes).sum() / volumes.sum())

def load_subject_resources(
    subject_id:   str,
    project_dir:  str,
    cap_name:     str,
    roi_mask:     str,
    non_roi_mask: str | None = None,
    electrode_shape:         str | None = None,
    electrode_dimensions:    list | None = None,
    electrode_gel_thickness: float | None = None,
    leadfield_path:          str | None = None,
) -> SubjectResources:
    """Load leadfield + ROI/non-ROI masks for a subject. Call once, reuse.

    leadfield_path: full path to the HDF5 file. When set, overrides all
    auto-constructed paths (cap_name, electrode_dimensions, etc. are ignored
    for path resolution but cap_name is still used for the params JSON lookup
    next to the HDF5).

    electrode_shape/dimensions/gel_thickness are optional and opt-in: pass
    them to look up a leadfield from the newer per-settings cache that
    run_pipeline.py writes (leadfield_volume/{tag}/, see config.leadfield_tag)
    — needed if you want a leadfield computed with specific electrode
    geometry rather than whatever happens to be cached. Leave them unset
    (the default) to keep using the older flat
    leadfield_volume/{id}_leadfield_{cap}.hdf5 path, unchanged, for any
    existing notebook/script call that doesn't pass them.
    """
    from simnibs.utils import TI_utils as TI
    from simnibs import mesh_io

    base_lf_dir = os.path.join(project_dir, "derivatives", "SimNIBS",
                               f"sub-{subject_id}", "leadfield_volume")

    if leadfield_path is not None:
        lf_hdf = leadfield_path
        lf_par = os.path.splitext(lf_hdf)[0] + "_params.json"
    elif electrode_dimensions is not None:
        from config import leadfield_tag
        tag    = leadfield_tag(cap_name, electrode_shape or "ellipse", electrode_dimensions,
                               electrode_gel_thickness if electrode_gel_thickness is not None else 1.0)
        lf_dir = os.path.join(base_lf_dir, tag)
        # TDCSLEADFIELD's own output filename convention (subpath + eeg_cap
        # basename) — only the directory is new, not this filename pattern.
        lf_hdf = os.path.join(lf_dir, f"{subject_id}_leadfield_{cap_name}.hdf5")
        lf_par = os.path.join(lf_dir, f"{subject_id}_leadfield_{cap_name}_params.json")
    else:
        lf_dir = base_lf_dir
        lf_hdf = os.path.join(lf_dir, f"{subject_id}_leadfield_{cap_name}.hdf5")
        lf_par = os.path.join(lf_dir, f"{subject_id}_leadfield_{cap_name}_params.json")

    if not os.path.isfile(lf_hdf):
        sys.exit(f"ERROR: leadfield not found:\n  {lf_hdf}")

    print(f"Loading leadfield: {lf_hdf}")
    leadfield, mesh, idx_lf = TI.load_leadfield(lf_hdf)
    print(f"  Leadfield shape: {leadfield.shape}  "
          f"({len(idx_lf)} electrodes)")

    lf_params = {}
    if os.path.isfile(lf_par):
        with open(lf_par) as f:
            lf_params = json.load(f)

    # Always work with tissue elements only — keeps all leadfields comparable.
    # If the leadfield stores all mesh elements (newer pipeline), slice it down
    # to tissue elements so masks, centroids, and fields are all the same size.
    import nibabel as nib
    tissue_mask = np.isin(mesh.elm.tag1, TISSUE_TAGS)
    m_tissue    = mesh.crop_mesh(tags=TISSUE_TAGS)
    # if leadfield.shape[1] != m_tissue.elm.nr:
    #     leadfield = leadfield[:, tissue_mask, :]
    #     print(f"  Sliced leadfield to tissue elements: {leadfield.shape[1]}")
    tags        = m_tissue.elm.tag1
    nodes       = m_tissue.nodes.node_coord
    conn        = m_tissue.elm.node_number_list[:, :4] - 1
    centroids   = nodes[conn].mean(axis=1)          # (M, 3)
    v0 = nodes[conn[:, 0]]; v1 = nodes[conn[:, 1]]
    v2 = nodes[conn[:, 2]]; v3 = nodes[conn[:, 3]]
    elm_volumes = np.abs(np.einsum('ni,ni->n',
        v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0   # mm³

    roi_elm_mask     = _mask_to_elements(roi_mask, centroids)
    non_roi_elm_mask = (_mask_to_elements(non_roi_mask, centroids)
                        if non_roi_mask and os.path.isfile(non_roi_mask)
                        else np.zeros(len(centroids), dtype=bool))

    print(f"  ROI elements    : {roi_elm_mask.sum()}")
    print(f"  Non-ROI elements: {non_roi_elm_mask.sum()}")

    return SubjectResources(
        leadfield        = leadfield,
        mesh             = mesh,
        idx_lf           = idx_lf,
        centroids        = centroids,
        tissue_mask      = tissue_mask,
        tags             = tags,
        roi_elm_mask     = roi_elm_mask,
        non_roi_elm_mask = non_roi_elm_mask,
        lf_params        = lf_params,
        elm_volumes      = elm_volumes,
    )


def compute_ti_setup(
    resources: SubjectResources,
    ch1_plus:       str,
    ch1_minus:      str,
    ch1_current_mA: float,
    ch2_plus:       str,
    ch2_minus:      str,
    ch2_current_mA: float,
    electrode_dims: list[float] | None = None,
    label:          str  = "setup",
    output_dir:     str  = ".",
) -> dict:
    """
    Compute TI field for one 4-electrode setup and save results.

    Parameters
    ----------
    resources       : loaded via load_subject_resources()
    ch1_plus/minus  : electrode names for channel 1
    ch1_current_mA  : current for channel 1 in mA
    ch2_plus/minus  : electrode names for channel 2
    ch2_current_mA  : current for channel 2 in mA
    electrode_dims  : [width_mm, height_mm] — stored as metadata only;
                      must match leadfield dims for accurate results
    label           : short name used in output filenames and print headers
    output_dir      : directory for NPZ + JSON outputs

    Returns
    -------
    dict with keys: label, montage, stats, msh_path, npz_path, json_path, ti, ef1, ef2
    """
    from simnibs.utils import TI_utils as TI

    r = resources

    # Warn if electrode dims don't match leadfield
    if electrode_dims is not None and r.lf_params.get("dimensions"):
        lf_dims = r.lf_params["dimensions"]
        if list(electrode_dims) != list(lf_dims):
            print(f"  WARNING [{label}]: requested dims {electrode_dims} mm differ from "
                  f"leadfield dims {lf_dims} mm — results use leadfield geometry")

    for name in [ch1_plus, ch1_minus, ch2_plus, ch2_minus]:
        if name not in r.idx_lf:
            sys.exit(f"ERROR [{label}]: electrode '{name}' not in leadfield "
                     f"(available: {sorted(r.idx_lf.keys())})")

    # ── Cache check ──────────────────────────────────────────────────────────
    os.makedirs(output_dir, exist_ok=True)
    _npz  = os.path.join(output_dir, f"ti_field_{label}.npz")
    _json = os.path.join(output_dir, f"ti_stats_{label}.json")
    _msh  = os.path.join(output_dir, f"{label}_head_mesh.msh")
    if os.path.isfile(_npz) and os.path.isfile(_json):
        try:
            with open(_json) as _f:
                _cached = json.load(_f)
            _m = _cached.get('montage', {})
            if (_m.get('ch1_plus') == ch1_plus and _m.get('ch1_minus') == ch1_minus and
                    _m.get('ch2_plus') == ch2_plus and _m.get('ch2_minus') == ch2_minus and
                    abs(_m.get('ch1_current_mA', 0) - ch1_current_mA) < 1e-6 and
                    abs(_m.get('ch2_current_mA', 0) - ch2_current_mA) < 1e-6):
                print(f"  [cache] {label} — loading pre-computed TI field")
                _data = np.load(_npz)
                return {"label": label, "montage": _cached['montage'],
                        "stats": _cached['stats'],
                        "msh_path": _msh, "npz_path": _npz, "json_path": _json,
                        "ti": _data['ti_all'], "ef1": _data['ef1'], "ef2": _data['ef2']}
        except Exception as _e:
            print(f"  [cache] {label} — cache corrupted ({_e}), recomputing")

    print(f"\n{'='*60}")
    print(f"  Setup: {label}")
    print(f"  Ch1: {ch1_plus}+ / {ch1_minus}-  @ {ch1_current_mA} mA")
    print(f"  Ch2: {ch2_plus}+ / {ch2_minus}-  @ {ch2_current_mA} mA")
    print(f"{'='*60}")

    # E-field vectors at every tissue element for each channel
    # TI.get_field scales by current internally
    ef1 = TI.get_field([ch1_plus, ch1_minus, ch1_current_mA * 1e-3],
                       r.leadfield, r.idx_lf)        # (M, 3)
    ef2 = TI.get_field([ch2_plus, ch2_minus, ch2_current_mA * 1e-3],
                       r.leadfield, r.idx_lf)        # (M, 3)
    ti  = TI.get_maxTI(ef1, ef2)                     # (M_all,)  amplitude V/m
    ti  = ti[r.tissue_mask]                          # (M_gm_wm,) — restrict to TISSUE_TAGS
    ef1 = ef1[r.tissue_mask]                         # (M_gm_wm, 3)
    ef2 = ef2[r.tissue_mask]                         # (M_gm_wm, 3)

    # Statistics
    ti_roi     = ti[r.roi_elm_mask]
    ti_non_roi = ti[r.non_roi_elm_mask]

    brain_vol_mean = _vol_mean_capped(ti, r.elm_volumes)
    stats: dict = {
        "TI_max_whole_brain_V_m":  float(ti.max()),
        "TI_mean_whole_brain_V_m": brain_vol_mean,
    }
    if r.roi_elm_mask.any():
        roi_vols     = r.elm_volumes[r.roi_elm_mask]
        roi_vol_mean = _vol_mean_capped(ti_roi, roi_vols)
        stats["TI_max_ROI_V_m"]       = float(ti_roi.max())
        stats["TI_mean_ROI_V_m"]      = roi_vol_mean
        stats["focality_ratio_brain"] = roi_vol_mean / brain_vol_mean
        if r.non_roi_elm_mask.any():
            nr_vols     = r.elm_volumes[r.non_roi_elm_mask]
            nr_vol_mean = _vol_mean_capped(ti_non_roi, nr_vols)
            stats["TI_mean_non_ROI_V_m"]    = nr_vol_mean
            stats["focality_ratio_non_roi"] = roi_vol_mean / nr_vol_mean

    # Per-tissue-tag breakdown (GM is most clinically relevant)
    for tag, tname in TAG_NAMES.items():
        mask = r.tags == tag
        if mask.any():
            stats[f"TI_mean_{tname}_V_m"] = _vol_mean_capped(ti[mask], r.elm_volumes[mask])
            stats[f"TI_max_{tname}_V_m"]  = float(ti[mask].max())

    for k, v in stats.items():
        if "ratio" in k:
            print(f"    {k}: {v:.3f}")
        else:
            print(f"    {k}: {v:.4f} V/m")

    os.makedirs(output_dir, exist_ok=True)

    # ── Save SimNIBS mesh file (.msh) — openable in Gmsh ─────────────────────
    # The raw leadfield mesh contains all cap electrode patches (all 32 for
    # BioSemi32) and the ElementData from the leadfield covers only tissue
    # volume elements — not the full mesh element count.  We therefore:
    #   1. Crop to tissue volumes (tags 1–5) + standard head surfaces (1001–1005),
    #      removing all electrode patch elements.
    #   2. Pad field arrays to the cropped mesh size (zeros on surface elements).
    # This gives a clean Gmsh file with correct field placement and no spurious
    # electrode geometry.
    #
    # Scalar views:  normE_ch1, normE_ch2, max_TI  [V/m]
    # Vector views:  E_ch1, E_ch2
    from simnibs import mesh_io

    msh_path = os.path.join(output_dir, f"{label}_head_mesh.msh")

    # Crop electrode patches: keep tissue volumes (1–5) and head surfaces (1001–1005)
    VIZ_TAGS = TISSUE_TAGS + [1001, 1002, 1003, 1004, 1005]
    m_viz = r.mesh.crop_mesh(tags=VIZ_TAGS)
    n_viz = m_viz.elm.nr
    vol_mask_viz = np.isin(m_viz.elm.tag1, TISSUE_TAGS)  # volume elements in cropped mesh

    def _pad_scalar(arr1d):
        out = np.zeros(n_viz, dtype=np.float32)
        out[vol_mask_viz] = arr1d.astype(np.float32)
        return out

    def _pad_vector(arr2d):
        out = np.zeros((n_viz, 3), dtype=np.float32)
        out[vol_mask_viz] = arr2d.astype(np.float32)
        return out

    m_viz.elmdata = []
    m_viz.elmdata.append(mesh_io.ElementData(_pad_scalar(np.linalg.norm(ef1, axis=1)), "normE_ch1"))
    m_viz.elmdata.append(mesh_io.ElementData(_pad_scalar(np.linalg.norm(ef2, axis=1)), "normE_ch2"))
    m_viz.elmdata.append(mesh_io.ElementData(_pad_scalar(ti),  "max_TI"))
    m_viz.elmdata.append(mesh_io.ElementData(_pad_vector(ef1), "E_ch1"))
    m_viz.elmdata.append(mesh_io.ElementData(_pad_vector(ef2), "E_ch2"))
    mesh_io.write_msh(m_viz, msh_path)
    print(f"  Mesh saved -> {msh_path}")

    # ── Save NPZ (compatible with run_analysis format + extras) ──────────────
    npz_path = os.path.join(output_dir, f"ti_field_{label}.npz")
    np.savez_compressed(
        npz_path,
        ti_all       = ti,
        ef1          = ef1,
        ef2          = ef2,
        centroids    = r.centroids,
        tag1         = r.tags,
        roi_mask     = r.roi_elm_mask,
        non_roi_mask = r.non_roi_elm_mask,
    )
    print(f"  NPZ saved  -> {npz_path}")

    # ── Save JSON summary ─────────────────────────────────────────────────────
    montage = {
        "ch1_plus":       ch1_plus,
        "ch1_minus":      ch1_minus,
        "ch1_current_mA": ch1_current_mA,
        "ch2_plus":       ch2_plus,
        "ch2_minus":      ch2_minus,
        "ch2_current_mA": ch2_current_mA,
    }
    if electrode_dims:
        montage["electrode_dims_mm"] = electrode_dims

    result = {"label": label, "montage": montage, "stats": stats}
    json_path = os.path.join(output_dir, f"ti_stats_{label}.json")
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"  JSON saved -> {json_path}")

    return {"label": label, "montage": montage, "stats": stats,
            "msh_path": msh_path, "npz_path": npz_path, "json_path": json_path,
            "ti": ti, "ef1": ef1, "ef2": ef2}


def compare_setups(result_a: dict, result_b: dict) -> None:
    """Print a side-by-side comparison table of two compute_ti_setup results."""
    la, lb = result_a["label"], result_b["label"]
    sa, sb = result_a["stats"], result_b["stats"]
    ma, mb = result_a["montage"], result_b["montage"]

    col = 22
    lw  = max(len(la), len(lb), col)

    print(f"\n{'='*70}")
    print(f"  Comparison: {la}  vs  {lb}")
    print(f"{'='*70}")

    # Montage summary
    print(f"\n  {'Montage':<{col}}  {la:<{lw}}  {lb:<{lw}}")
    print(f"  {'-'*col}  {'-'*lw}  {'-'*lw}")
    print(f"  {'Ch1':<{col}}  "
          f"{ma['ch1_plus']}+/{ma['ch1_minus']}- @ {ma['ch1_current_mA']}mA".ljust(lw) +
          f"  {'  '}{mb['ch1_plus']}+/{mb['ch1_minus']}- @ {mb['ch1_current_mA']}mA")
    print(f"  {'Ch2':<{col}}  "
          f"{ma['ch2_plus']}+/{ma['ch2_minus']}- @ {ma['ch2_current_mA']}mA".ljust(lw) +
          f"  {'  '}{mb['ch2_plus']}+/{mb['ch2_minus']}- @ {mb['ch2_current_mA']}mA")

    # Stats comparison
    all_keys = [k for k in sa if k in sb]
    print(f"\n  {'Metric':<{col}}  {la:<{lw}}  {lb:<{lw}}  {'Diff (B-A)'}")
    print(f"  {'-'*col}  {'-'*lw}  {'-'*lw}  {'-'*12}")
    for k in all_keys:
        va, vb = sa[k], sb[k]
        diff   = vb - va
        unit   = "" if "ratio" in k else " V/m"
        fmt    = ".3f" if "ratio" in k else ".4f"
        sign   = "+" if diff >= 0 else ""
        print(f"  {k:<{col}}  {va:{fmt}}{unit}  {vb:{fmt}}{unit}  "
              f"{sign}{diff:{fmt}}{unit}")

    print(f"\n{'='*70}\n")


# ═════════════════════════════════════════════════════════════════════════════
# Internal helpers
# ═════════════════════════════════════════════════════════════════════════════

def _mask_to_elements(mask_path: str, centroids: np.ndarray) -> np.ndarray:
    """Return boolean array (len = n_tissue_elements) marking elements inside mask."""
    import nibabel as nib
    img  = nib.load(mask_path)
    data = np.asarray(img.dataobj) > 0
    aff_inv  = np.linalg.inv(img.affine)
    ones     = np.ones((len(centroids), 1))
    vox      = (aff_inv @ np.hstack([centroids, ones]).T).T[:, :3]
    vox_idx  = np.round(vox).astype(int)
    sh       = data.shape
    in_bounds = ((vox_idx[:, 0] >= 0) & (vox_idx[:, 0] < sh[0]) &
                 (vox_idx[:, 1] >= 0) & (vox_idx[:, 1] < sh[1]) &
                 (vox_idx[:, 2] >= 0) & (vox_idx[:, 2] < sh[2]))
    mask_out = np.zeros(len(centroids), dtype=bool)
    mask_out[in_bounds] = data[vox_idx[in_bounds, 0],
                               vox_idx[in_bounds, 1],
                               vox_idx[in_bounds, 2]]
    return mask_out


# ═════════════════════════════════════════════════════════════════════════════
# CLI
# ═════════════════════════════════════════════════════════════════════════════

def _parse_args():
    p = argparse.ArgumentParser(
        description="Compare two TI stimulation setups using a cached SimNIBS leadfield."
    )
    p.add_argument("--subject",     required=True, help="Subject ID, e.g. 025")
    p.add_argument("--project-dir", required=True, help="Root BIDS project directory")
    p.add_argument("--cap",         default="BioSemi32_MNE",
                   help="Cap name matching leadfield HDF5 stem (default: BioSemi32_MNE)")

    # ROI/non-ROI — accept either a mask path or a roi name (auto-resolves path)
    p.add_argument("--roi",         required=True,
                   help="ROI name (e.g. hippo_r_phg) or full path to mask NIfTI")
    p.add_argument("--non-roi",     default=None,
                   help="Non-ROI name or full path (optional)")

    # Setup A: CH1+ CH1- CH1_mA CH2+ CH2- CH2_mA
    p.add_argument("--setup-a", nargs=6, metavar=("CH1+","CH1-","CH1_mA","CH2+","CH2-","CH2_mA"),
                   required=True, help="Setup A: ch1_plus ch1_minus ch1_mA ch2_plus ch2_minus ch2_mA")
    p.add_argument("--label-a", default="setup_a", help="Label for setup A")

    # Setup B
    p.add_argument("--setup-b", nargs=6, metavar=("CH1+","CH1-","CH1_mA","CH2+","CH2-","CH2_mA"),
                   required=True, help="Setup B: ch1_plus ch1_minus ch1_mA ch2_plus ch2_minus ch2_mA")
    p.add_argument("--label-b", default="setup_b", help="Label for setup B")

    p.add_argument("--electrode-dims", nargs=2, type=float, default=None,
                   metavar=("WIDTH_MM", "HEIGHT_MM"),
                   help="Electrode dimensions in mm (metadata only, default: from leadfield params)")
    p.add_argument("--output-dir", default=None,
                   help="Output directory (default: <project_dir>/derivatives/SimNIBS/sub-<id>/comparison)")

    return p.parse_args()


def _resolve_mask(value: str, subject_id: str, project_dir: str) -> str:
    """Accept either a full path or an ROI name; return the mask path."""
    if os.path.isfile(value):
        return value
    # Try standard pipeline location
    candidate = os.path.join(
        project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}",
        "roi", f"sub-{subject_id}_label-{value}_mask.nii.gz"
    )
    if os.path.isfile(candidate):
        return candidate
    sys.exit(f"ERROR: cannot find mask for '{value}'\n"
             f"  Tried: {candidate}\n"
             f"  Pass the full path instead.")


def main():
    args = _parse_args()

    subj        = args.subject
    proj        = os.path.abspath(args.project_dir)
    output_dir  = args.output_dir or os.path.join(
        proj, "derivatives", "SimNIBS", f"sub-{subj}", "comparison")

    roi_mask     = _resolve_mask(args.roi,     subj, proj)
    non_roi_mask = (_resolve_mask(args.non_roi, subj, proj)
                    if args.non_roi else None)
    dims         = args.electrode_dims

    resources = load_subject_resources(
        subject_id   = subj,
        project_dir  = proj,
        cap_name     = args.cap,
        roi_mask     = roi_mask,
        non_roi_mask = non_roi_mask,
    )

    a = args.setup_a
    result_a = compute_ti_setup(
        resources,
        ch1_plus=a[0], ch1_minus=a[1], ch1_current_mA=float(a[2]),
        ch2_plus=a[3], ch2_minus=a[4], ch2_current_mA=float(a[5]),
        electrode_dims=dims,
        label=args.label_a, output_dir=output_dir,
    )

    b = args.setup_b
    result_b = compute_ti_setup(
        resources,
        ch1_plus=b[0], ch1_minus=b[1], ch1_current_mA=float(b[2]),
        ch2_plus=b[3], ch2_minus=b[4], ch2_current_mA=float(b[5]),
        electrode_dims=dims,
        label=args.label_b, output_dir=output_dir,
    )

    compare_setups(result_a, result_b)


if __name__ == "__main__":
    main()

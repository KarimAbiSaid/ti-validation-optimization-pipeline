"""
TI Pipeline — standalone runner.

Usage:
    python run_pipeline.py --config configs/sub-025_thalamus.json

Each section checks for existing output and skips if already done.
Override skip checks with --force-section (e.g. --force optimization).
"""

import os
import sys
import re
import json
import time
import datetime
import argparse
import subprocess
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates

# This script's own print()s use non-ASCII characters (→, ✓, ═ box drawing,
# ...). When stdout/stderr aren't an interactive console (piped/redirected —
# true for local runs via the GUI, or "python run_pipeline.py > log.txt"),
# Python falls back to the OS locale's preferred encoding, which is cp1252
# on Windows, not UTF-8 — crashing with UnicodeEncodeError on the first such
# character. SCITAS's Linux container defaults to UTF-8, so this never
# showed up there; reconfigure explicitly so local Windows runs don't depend
# on the caller happening to set PYTHONIOENCODING.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8", errors="replace")

# ── Config import ─────────────────────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))
from config import PipelineConfig, ROIConfig, load_config, save_config, leadfield_tag, is_stimulation_electrode


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers
# ═══════════════════════════════════════════════════════════════════════════════

def _vol_mean(values: np.ndarray, volumes: np.ndarray) -> float:
    """Computes the volume-weighted mean of values in simnibs, since some voxels are larger than others"""
    return float((values * volumes).sum() / volumes.sum())

def _vol_mean_capped(values: np.ndarray, volumes: np.ndarray, pct: int = 99) -> float:
    """Caps the volume-weighted mean at the given percentile to reduce outlier influence"""
    if len(values) == 0:
        return np.nan
    cap = float(np.percentile(values, pct))
    v   = np.minimum(values, cap)
    return float((v * volumes).sum() / volumes.sum())


def header(title: str) -> None:
    print(f"\n{'='*60}\n  {title}\n{'='*60}")

def skip(label: str, path: str) -> bool:
    """ Print skip message and return True if path exists."""
    if os.path.exists(path):
        print(f"  [SKIP] {label} — already exists:\n         {path}")
        return True
    return False


def abort(msg: str) -> None:
    print(f"\n  [ERROR] {msg}")
    sys.exit(1)


# ═══════════════════════════════════════════════════════════════════════════════
# Section 0a — recon-all
# ═══════════════════════════════════════════════════════════════════════════════

def run_recon_all(cfg: PipelineConfig, force: bool = False) -> None:

    """Run FreeSurfer's recon-all on the T1 (and optionally T2) image.
    This is since recon-all creates a parcellation of the cortex that is 
    more fine-grained than the charm segmentation, which is useful for some ROIs. (e.g, M1)"""

    header("Section 0a — recon-all (FreeSurfer)")

    out = f"{cfg.fs_dir}/mri/aparc+aseg.mgz"
    if not force and skip("recon-all", out):
        return

    if not os.path.isfile(cfg.t1_path):
        abort(f"T1 not found: {cfg.t1_path}")

    cmd = [
        "recon-all",
        "-subject", f"sub-{cfg.subject_id}",
        "-i", cfg.t1_path,
        "-all",
        "-sd", f"{cfg.project_dir}/derivatives/freesurfer",
        "-parallel", "-openmp", str(cfg.recon_all.openmp),
    ]
    if cfg.recon_all.use_t2 and os.path.isfile(cfg.t2_path):
        cmd += ["-T2", cfg.t2_path, "-T2pial"]

    # Remove existing partial output so recon-all starts clean
    fs_sub = f"{cfg.project_dir}/derivatives/freesurfer/sub-{cfg.subject_id}"
    if os.path.isdir(fs_sub):
        print(f"  Removing partial FreeSurfer output: {fs_sub}")
        import shutil
        shutil.rmtree(fs_sub)

    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)

    if not os.path.isfile(out):
        abort(f"recon-all completed but aparc+aseg.mgz not found at {out}")
    print("  recon-all complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 0b — charm
# ═══════════════════════════════════════════════════════════════════════════════

def run_charm(cfg: PipelineConfig, force: bool = False) -> None:
    """For the BNA atlas or just to get a SimNIBS head model, run charm on the T1 (and optionally T2) image.
    Typically if we're using atlases that don't need such a detailed parcellation, 
    we can skip recon-all and just run charm to get the SimNIBS head model."""
    header("Section 0b — charm (SimNIBS head model)")

    out = f"{cfg.m2m_path}/{cfg.subject_id}.msh"
    if not force and skip("charm", out):
        return

    for p, label in [(cfg.t1_path, "T1")]:
        if not os.path.isfile(p):
            abort(f"{label} not found: {p}")

    os.makedirs(cfg.sim_sub_dir, exist_ok=True)
    cmd = ["charm", cfg.subject_id, cfg.t1_path]
    if os.path.isfile(cfg.t2_path):
        cmd.append(cfg.t2_path)

    print(f"  Running: {' '.join(cmd)}")
    subprocess.run(cmd, cwd=cfg.sim_sub_dir, check=True)

    if not os.path.isfile(out):
        abort(f"charm completed but mesh not found at {out}")
    print("  charm complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 1 — ROI / non-ROI masks
# ═══════════════════════════════════════════════════════════════════════════════

def _warp_bna_atlas(bna_atlas_path: str, m2m_path: str, out_path: str) -> None:
    """Warp BN_Atlas_246_1mm.nii.gz from MNI space → subject (Conform) space,
    via MaskGeneratorEngine/BNAAtlas (create_masks.py) — same nearest-
    neighbour mechanism this function used to implement directly, now shared
    with the local validation notebooks' mask generation.

    The output is cached at out_path so subsequent configs for the same subject
    reuse it without re-running the warp.
    """
    if os.path.isfile(out_path):
        print(f"  [SKIP] BNA atlas warp — cached:\n         {out_path}")
        return

    c2m_path = os.path.join(m2m_path, "toMNI", "Conform2MNI_nonl.nii.gz")
    ref_path  = os.path.join(m2m_path, "segmentation", "labeling.nii.gz")

    for p, label in [(c2m_path, "Conform→MNI field"), (ref_path, "labeling.nii.gz"),
                     (bna_atlas_path, "BNA atlas")]:
        if not os.path.isfile(p):
            abort(f"{label} not found: {p}\n  Run charm (Section 0b) first.")

    print(f"  Warping BNA atlas → subject space (scipy nearest-neighbour) …")

    from create_masks import MaskGeneratorEngine, BNAAtlas
    engine = MaskGeneratorEngine(m2m_path=m2m_path)
    out_labels = engine.get_subject_space_labels(BNAAtlas(bna_atlas_path)).astype(np.int16)
    _, out_aff, out_hdr = engine._reference_grid()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    nib.save(nib.Nifti1Image(out_labels, out_aff, out_hdr), out_path)
    n_nonzero = int((out_labels > 0).sum())
    print(f"  Warped atlas: {n_nonzero:,} labelled voxels  |  {out_labels.max()} max label")
    print(f"  Saved → {out_path}")


def _create_bna_mask(warped_atlas_path: str, label_dict: dict, out_path: str) -> int:
    """Extract integer BNA labels from a warped atlas → binary NIfTI mask."""
    atlas_img  = nib.load(warped_atlas_path)
    atlas_data = np.asarray(atlas_img.dataobj, dtype=np.int32)
    affine     = atlas_img.affine
    hdr        = atlas_img.header
    vox_vol    = abs(np.linalg.det(affine[:3, :3]))

    mask = np.zeros(atlas_data.shape, dtype=np.uint8)
    print(f"  {'Structure':<35}  {'Label':>5}  {'Voxels':>8}  {'mm³':>10}")
    print("  " + "-" * 65)
    for name, lbl in label_dict.items():
        n = int(np.sum(atlas_data == lbl))
        if n == 0:
            print(f"  WARNING: BNA label {lbl} ({name}) not found in warped atlas — "
                  "check that charm ran successfully and the atlas path is correct")
        else:
            mask[atlas_data == lbl] = 1
        print(f"  {name:<35}  {lbl:>5}  {n:>8,}  {n*vox_vol:>10.1f}")

    total = int(mask.sum())
    print(f"  {'TOTAL':<35}  {'—':>5}  {total:>8,}  {total*vox_vol:>10.1f}")

    img = nib.Nifti1Image(mask, affine, hdr)
    img.set_data_dtype(np.uint8)
    nib.save(img, out_path)
    print(f"  Saved → {out_path}")
    return total


def _create_mask(aseg_path, label_dict, m2m_path, out_path, fallback_labels=None, engine=None):
    """Binary mask from FreeSurfer/charm aseg labels, via
    MaskGeneratorEngine/FreeSurferAtlas (create_masks.py).

    NOTE (2025-07 unification): previously this saved the mask directly on
    aseg's own native grid (FreeSurfer's own conformed space when real
    recon-all output is used, which differs from charm's Conform grid used
    by the BNA path). It now always resamples onto charm's Conform grid, so
    every mask this pipeline produces (BNA- or FreeSurfer-sourced) shares
    one common grid — a deliberate behaviour change, verified beforehand via
    a Dice=1.0 comparison against the previous per-source-grid output.

    engine: optional pre-built MaskGeneratorEngine, shared across multiple
    calls (ROI/non-ROI/extra ROIs in one run) so the resampled aseg data is
    cached and reused instead of recomputed per call. Builds one if omitted.
    """
    from create_masks import MaskGeneratorEngine, FreeSurferAtlas
    if engine is None:
        engine = MaskGeneratorEngine(m2m_path=m2m_path)
    mask = engine.create_mask(FreeSurferAtlas(aseg_path), label_dict, out_path,
                              fallback_label_ids=fallback_labels)
    return int(mask.sum())


def create_roi_masks(cfg: PipelineConfig, force: bool = False) -> None:
    """Create ROI, non-ROI, and extra ROI masks as needed.
    These masks are defined in the config, and they are what we
    absolutely need for the focality and mean constratints in the optimization."""
    header("Section 1 — ROI / non-ROI masks")

    roi_path     = cfg.mask_path(cfg.roi.name)
    non_roi_path = cfg.mask_path(cfg.non_roi.name) if cfg.non_roi else None

    roi_needed     = force or not os.path.exists(roi_path)
    non_roi_needed = (non_roi_path is not None) and (force or not os.path.exists(non_roi_path))

    if not roi_needed:
        print(f"  [SKIP] ROI mask — already exists:\n         {roi_path}")
    if non_roi_path is not None and not non_roi_needed:
        print(f"  [SKIP] non-ROI mask — already exists:\n         {non_roi_path}")

    extra_needed = []
    for extra in cfg.extra_rois:
        extra_path = cfg.mask_path(extra.name)
        if force or not os.path.isfile(extra_path):
            extra_needed.append(extra)
        else:
            print(f"  [SKIP] extra ROI mask — already exists: {extra_path}")

    if not roi_needed and not non_roi_needed and not extra_needed:
        print("  Section 1 complete.")
        return

    os.makedirs(cfg.roi_dir, exist_ok=True)

    all_roi_cfgs = (
        ([(cfg.roi, roi_path)] if roi_needed else [])
        + ([(cfg.non_roi, non_roi_path)] if cfg.non_roi and non_roi_needed else [])
        + [(e, cfg.mask_path(e.name)) for e in extra_needed]
    )

    # ── Determine whether BNA and/or FreeSurfer sources are needed ────────────
    needs_bna = any(rc.bna_labels for rc, _ in all_roi_cfgs)
    needs_fs  = any(not rc.bna_labels for rc, _ in all_roi_cfgs)

    # ── BNA path: warp atlas once, then extract per-ROI masks ────────────────
    warped_bna = None
    if needs_bna:
        if not cfg.bna_atlas_path:
            abort("One or more ROIs use bna_labels but bna_atlas_path is not set.\n"
                  "  Set bna_atlas_path to the BN_Atlas_246_1mm.nii.gz path in your config.")
        if not os.path.isfile(cfg.bna_atlas_path):
            abort(f"BNA atlas not found: {cfg.bna_atlas_path}")
        warped_bna = os.path.join(cfg.roi_dir,
                                   f"sub-{cfg.subject_id}_BNA_atlas_subjectspace.nii.gz")
        _warp_bna_atlas(cfg.bna_atlas_path, cfg.m2m_path, warped_bna)

    # ── FreeSurfer path: resolve aseg path + share one engine (so its
    # in-memory cache avoids re-resampling aseg for every ROI/non-ROI/extra) ─
    aseg = fs_engine = None
    if needs_fs:
        aseg = cfg.aseg_path
        if aseg is None:
            abort("No FreeSurfer segmentation found. Run recon-all or charm first.")
        if "aseg.auto" in aseg or "labeling" in aseg:
            print("  WARNING: only subcortical segmentation available — "
                  "cortical labels (e.g. M1) will not be found.")
        print(f"  aseg: {aseg}")
        from create_masks import MaskGeneratorEngine
        fs_engine = MaskGeneratorEngine(m2m_path=cfg.m2m_path)

    # ── Create each mask ──────────────────────────────────────────────────────
    for roi_cfg, out_path in all_roi_cfgs:
        label = "ROI" if out_path == roi_path else "non-ROI" if out_path == non_roi_path else "extra ROI"
        print(f"\n  {label}: {roi_cfg.name}")
        if roi_cfg.bna_labels:
            _create_bna_mask(warped_bna, roi_cfg.bna_labels, out_path)
        else:
            _create_mask(aseg, roi_cfg.labels, cfg.m2m_path, out_path,
                         fallback_labels=roi_cfg.fallback_labels, engine=fs_engine)

    print("  Section 1 complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# LEGACY — EEG cap scalp boundary filter
# Exclusively supported the TesFlexOptimization (DE) path above, which is now
# commented out (see _run_optimization_tesflex_LEGACY). No caller outside this
# cluster (_load_cap_coords, _cap_node_mask, apply_eeg_cap_boundary,
# preview_skin_filter) remains, so it is dead code. Kept in the file, not
# deleted.
# ═══════════════════════════════════════════════════════════════════════════════

# def _load_cap_coords(eeg_csv_path: str, opt_subpath: str = None) -> np.ndarray:
#     """
#     Load Electrode/ReferenceElectrode rows from an EEG cap CSV.
#     Prefers the registered version in opt_subpath/eeg_positions/ if it exists.
#     """
#     path = eeg_csv_path
#     if opt_subpath:
#         registered = os.path.join(opt_subpath, "eeg_positions",
#                                   os.path.basename(eeg_csv_path))
#         if os.path.isfile(registered):
#             path = registered
#
#     coords = []
#     with open(path) as f:
#         for line in f:
#             parts = [p.strip() for p in line.strip().split(",")]
#             if len(parts) < 4:
#                 continue
#             if parts[0] in ("Electrode", "ReferenceElectrode"):
#                 try:
#                     coords.append([float(parts[1]), float(parts[2]), float(parts[3])])
#                 except ValueError:
#                     continue
#     if not coords:
#         raise ValueError(f"No electrode positions found in {path}")
#     print(f"  EEG cap: {len(coords)} electrodes from {os.path.basename(path)}")
#     return np.array(coords)
#
#
# def _cap_node_mask(skin_nodes: np.ndarray, cap_coords: np.ndarray,
#                    margin_mm: float) -> np.ndarray:
#     """
#     Boolean mask of skin nodes within the EEG cap boundary.
#
#     Algorithm:
#       1. Fit a plane to the cap electrodes (PCA, least-variance axis = normal).
#          Orient the normal to point away from the scalp centroid (i.e. outward).
#       2. Project all skin nodes orthographically onto that plane.
#       3. A node is valid if its 2-D projection falls inside the convex hull of
#          the projected cap electrodes AND it is not more than margin_mm below
#          the plane (half-space guard that prevents the bottom of the head from
#          projecting back into the cap region).
#       4. Additionally include any node within margin_mm Euclidean distance of
#          any cap electrode (boundary buffer).
#     """
#     from scipy.spatial import Delaunay
#
#     scalp_centroid = skin_nodes.mean(axis=0)
#     cap_center_3d  = cap_coords.mean(axis=0)
#
#     # --- Step 1: fit plane to cap electrodes via PCA ---
#     _, _, Vt   = np.linalg.svd(cap_coords - cap_center_3d)
#     normal     = Vt[-1]                               # least-variance axis
#     # orient normal to point outward (away from scalp centroid)
#     if np.dot(normal, cap_center_3d - scalp_centroid) < 0:
#         normal = -normal
#
#     # --- Step 2: orthonormal basis for the cap plane ---
#     ref = np.array([0.0, 0.0, 1.0])
#     if abs(np.dot(normal, ref)) > 0.9:
#         ref = np.array([1.0, 0.0, 0.0])
#     u = np.cross(normal, ref);  u /= np.linalg.norm(u)
#     v = np.cross(normal, u)
#     basis = np.column_stack([u, v])                   # (3, 2)
#
#     # --- Step 3: project onto plane ---
#     c_rel  = cap_coords  - cap_center_3d
#     s_rel  = skin_nodes  - cap_center_3d
#
#     c_2d   = c_rel @ basis                            # (K, 2)
#     s_2d   = s_rel @ basis                            # (N, 2)
#     s_dist = s_rel @ normal                           # (N,) signed distance from plane
#
#     # half-space: exclude nodes more than margin_mm below the plane
#     above  = s_dist >= -margin_mm
#
#     # 2-D convex hull — expanded outward by margin_mm so the boundary region
#     # between outermost electrodes is included, not just circles around each one
#     in_hull = np.zeros(len(skin_nodes), dtype=bool)
#     try:
#         from scipy.spatial import ConvexHull
#         hull_2d   = ConvexHull(c_2d)
#         verts     = c_2d[hull_2d.vertices]              # (V, 2) outermost electrodes
#         centroid_2d = verts.mean(axis=0)
#         directions  = verts - centroid_2d
#         directions /= np.linalg.norm(directions, axis=1, keepdims=True)
#         expanded    = verts + directions * margin_mm     # push each vertex outward
#         tri         = Delaunay(expanded)
#         in_hull     = tri.find_simplex(s_2d) >= 0
#     except Exception as e:
#         print(f"  Warning: 2-D hull failed ({e}), using margin only.")
#
#     # Euclidean margin buffer as fallback for nodes near cap electrodes
#     dists_3d = np.linalg.norm(
#         skin_nodes[:, None, :] - cap_coords[None, :, :], axis=2)  # (N, K)
#     near_cap = dists_3d.min(axis=1) <= margin_mm
#
#     return (in_hull & above) | near_cap
#
#
# def apply_eeg_cap_boundary(opt, eeg_csv_path: str, margin_mm: float = 10.0) -> None:
#     """
#     Filter opt._skin_surface to only include nodes within the EEG cap boundary.
#
#     Call after opt._prepare() and before opt.run().
#     Sets opt._prepared = True so run() skips re-preparation.
#
#     Uses the same SimNIBS utilities as valid_skin_region internally:
#       create_new_connectivity_list_point_mask(points, con, point_mask)
#         - con is 0-indexed triangles
#         - returns (new_nodes, new_con_0indexed)
#       mesh_io.make_surface_mesh(nodes, con_1indexed) to rebuild the Msh object.
#     """
#     # NOTE: This is kind of stale, since it's related to the tes_flex_optimization module
#     # and it was just since the search was not across the points but rather the whole scalp surface and
#     # so we would get
#     from simnibs.utils.transformations import create_new_connectivity_list_point_mask
#     from simnibs.optimization.tes_flex_optimization.tes_flex_optimization import mesh_io
#
#     cap_coords = _load_cap_coords(eeg_csv_path, opt_subpath=opt.subpath)
#     skin_nodes = opt._skin_surface.nodes.node_coord    # (N, 3)
#     n_before   = len(skin_nodes)
#
#     node_mask = _cap_node_mask(skin_nodes, cap_coords, margin_mm)
#     n_after   = int(node_mask.sum())
#
#     if n_after == 0:
#         raise RuntimeError(
#             "EEG cap boundary filter removed ALL skin nodes. "
#             "Check that the cap CSV is in subject-space coordinates."
#         )
#
#     # Connectivity is 1-indexed in SimNIBS; take first 3 cols (triangles only)
#     conn_0idx = opt._skin_surface.elm.node_number_list[:, :3] - 1
#
#     # Compute which original node indices actually survive (only nodes referenced
#     # by at least one surviving triangle — mirrors create_new_connectivity_list_point_mask)
#     surviving_conn = conn_0idx[node_mask[conn_0idx].all(axis=1), :]
#     unique_pts = np.unique(surviving_conn)   # 0-indexed original node IDs
#
#     filtered_nodes, filtered_conn_0idx = create_new_connectivity_list_point_mask(
#         points=skin_nodes,
#         con=conn_0idx,
#         point_mask=node_mask,
#     )
#
#     # Rebuild Msh object the same way valid_skin_region does
#     fn = opt._skin_surface.fn
#     opt._skin_surface = mesh_io.make_surface_mesh(filtered_nodes,
#                                                   filtered_conn_0idx + 1)
#     opt._skin_surface.fn = fn
#
#     # Rebuild node→global-mesh index mapping using actual surviving node indices
#     opt._node_idx_msh = opt._node_idx_msh[unique_pts]
#
#     # Refit ellipsoid to filtered nodes
#     opt._ellipsoid.fit(points=filtered_nodes)
#
#     opt._prepared = True
#
#     pct = 100.0 * n_after / n_before
#     print(f"  Skin filter: {n_before} → {n_after} nodes ({pct:.1f}% retained), "
#           f"margin={margin_mm} mm")
#
#
# def preview_skin_filter(opt, eeg_csv_path: str, margin_mm: float = 10.0) -> None:
#     """
#     Dry-run: print node counts and save a 3-D scatter of valid (green) vs
#     excluded (red) skin nodes. Does NOT modify the optimizer.
#     """
#     # NOTE: This is also stale, as it's also related to the EEG cap boundary functionality
#     try:
#         import matplotlib
#         matplotlib.use("Agg")
#         import matplotlib.pyplot as plt
#         from mpl_toolkits.mplot3d import Axes3D   # noqa: F401
#         have_plt = True
#     except ImportError:
#         have_plt = False
#
#     cap_coords = _load_cap_coords(eeg_csv_path, opt_subpath=opt.subpath)
#     skin_nodes = opt._skin_surface.nodes.node_coord
#     n_total    = len(skin_nodes)
#     node_mask  = _cap_node_mask(skin_nodes, cap_coords, margin_mm)
#     n_valid    = int(node_mask.sum())
#
#     print(f"  Preview  : {n_total} skin nodes total")
#     print(f"  Valid    : {n_valid} ({100*n_valid/n_total:.1f}%)")
#     print(f"  Excluded : {n_total - n_valid} ({100*(n_total-n_valid)/n_total:.1f}%)")
#
#     if not have_plt:
#         return
#
#     valid_pts   = skin_nodes[ node_mask]
#     invalid_pts = skin_nodes[~node_mask]
#     step_v = max(1, len(valid_pts)   // 4000)
#     step_i = max(1, len(invalid_pts) // 4000)
#
#     fig = plt.figure(figsize=(10, 7))
#     ax  = fig.add_subplot(111, projection="3d")
#     ax.scatter(*valid_pts  [::step_v].T, c="green", s=1, alpha=0.4, label="Valid")
#     ax.scatter(*invalid_pts[::step_i].T, c="red",   s=1, alpha=0.4, label="Excluded")
#     ax.scatter(*cap_coords.T, c="blue", s=50, marker="^", zorder=5,
#                label="EEG electrodes")
#     ax.set_title(f"Skin filter preview  (margin={margin_mm} mm)\n"
#                  f"{n_valid}/{n_total} nodes retained")
#     ax.set_xlabel("x (mm)"); ax.set_ylabel("y (mm)"); ax.set_zlabel("z (mm)")
#     ax.legend(markerscale=5, fontsize=8)
#     fig.tight_layout()
#     out = "/tmp/skin_filter_preview.png"
#     fig.savefig(out, dpi=120, bbox_inches="tight")
#     print(f"  Saved: {out}")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2a — Exhaustive cap search (leadfield-based)
# ═══════════════════════════════════════════════════════════════════════════════
#
# HOW IT WORKS (3 steps):
#
#   Step 1 — Leadfield (one-time, ~18–20 FEM solves for a 19-electrode cap)
#     SimNIBS runs one FEM simulation per electrode: "1A flows into electrode i,
#     out of the reference." The result is stored as an (N_elec-1 × M × 3) array
#     in an HDF5 file, where M = number of mesh volume elements (tetrahedra).
#
#   Step 2 — Algebraic TI field (no more FEM, just math)
#     For any pair (A+, B-):  E = current × (leadfield[A] - leadfield[B])
#     For two pairs (TI):     TI_amplitude = max_TI(E_pair1, E_pair2)
#     Both operations are pure array arithmetic — ~0.1 ms each.
#
#   Step 3 — Exhaustive search over all electrode combinations
#     For 19 cap electrodes there are C(19,2)=171 possible pairs.
#     We try all unique pairs-of-pairs that don't share an electrode:
#     171 × 136 / 2 = 11,628 combinations. For each we compute TI at ROI
#     elements only (fast) and keep the best (highest mean ROI TI).
#
# TOTAL FEM WORK: ~18 solves vs ~2,600+ for the DE optimizer.
# GUARANTEE: finds the true global optimum over all discrete cap positions.
# This is the CURRENT STATE OF THE optimization algorithm that we are using

def _load_cap_positions(csv_path: str) -> dict:
    """Parse an EEG cap CSV into {electrode_name: np.array([x, y, z])},
    excluding ground/reference/fiducial-landmark rows (is_stimulation_electrode).
    Handles both "Name,X,Y,Z" and "Type,x,y,z,label" (BioSemi-style) layouts."""
    import csv as _csv

    cap_pos: dict = {}
    with open(csv_path, encoding="utf-8", newline="") as f:
        rows = [r for r in _csv.reader(f) if r and not r[0].strip().startswith('#')]
    if not rows:
        return cap_pos

    hdr = [c.strip().lower() for c in rows[0]]
    xi = next((i for i, h in enumerate(hdr) if h == 'x'), None)
    yi = next((i for i, h in enumerate(hdr) if h == 'y'), None)
    zi = next((i for i, h in enumerate(hdr) if h == 'z'), None)
    ni = next((i for i, h in enumerate(hdr)
                if h in ('label', 'name', 'ch_name', 'channel')), None)
    has_header = xi is not None and yi is not None and zi is not None

    for row in (rows[1:] if has_header else rows):
        row = [c.strip() for c in row]
        if len(row) < 4:
            continue
        try:
            if has_header:
                x, y, z = float(row[xi]), float(row[yi]), float(row[zi])
                name = row[ni] if ni is not None else row[0]
            else:
                name = row[0]
                x, y, z = float(row[1]), float(row[2]), float(row[3])
            if is_stimulation_electrode(name):
                cap_pos[name] = np.array([x, y, z])
        except (ValueError, IndexError):
            continue
    return cap_pos


def _farthest_point_sample(names: list, positions: dict, k: int) -> list:
    """Greedy farthest-point sampling for spatial-coverage subset selection.
    Deterministic — starts from the electrode nearest the centroid, then
    repeatedly picks whichever remaining electrode is farthest from the
    already-picked set."""
    n = len(names)
    if k >= n:
        return list(names)
    pts = np.array([positions[nm] for nm in names])
    centroid = pts.mean(axis=0)
    first = int(np.argmin(np.linalg.norm(pts - centroid, axis=1)))
    selected = [first]
    min_dist = np.linalg.norm(pts - pts[first], axis=1)
    min_dist[first] = -1
    while len(selected) < k:
        nxt = int(np.argmax(min_dist))
        selected.append(nxt)
        min_dist = np.minimum(min_dist, np.linalg.norm(pts - pts[nxt], axis=1))
        min_dist[selected] = -1
    return [names[i] for i in selected]


def _nearest_neighbours(name: str, candidates: list, positions: dict, n_neighbours: int) -> list:
    """The n_neighbours electrodes in `candidates` closest to `name` (excluding itself)."""
    others = [c for c in candidates if c != name]
    others.sort(key=lambda c: np.linalg.norm(positions[name] - positions[c]))
    return others[:n_neighbours]


def run_exhaustive_cap_optimization(cfg: PipelineConfig, force: bool = False,
                                    job_id: str = "") -> str:
    """
    Exhaustive TI optimization over all EEG cap electrode combinations.
    Returns the output directory path (same contract as run_optimization).
    """
    from itertools import combinations
    from simnibs.simulation.sim_struct import TDCSLEADFIELD
    from simnibs.utils import TI_utils as TI
    from simnibs import mesh_io

    header("Section 2a — Exhaustive Cap Search (leadfield-based)")

    # ── Output paths ──────────────────────────────────────────────────────────
    timestamp    = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    flex_dir     = cfg.ti_opt_dir
    job_tag      = f"_job-{job_id}" if job_id else ""
    goal_tag     = cfg.optimizer.goal  # "mean" or "focality"
    out_dir      = (f"{flex_dir}/sub-{cfg.subject_id}_roi-{cfg.roi.name}"
                    f"_goal-ex_{goal_tag}{job_tag}_run-{timestamp}")
    pointer_path = f"{cfg.sim_sub_dir}/latest_flex_run.json"

    # Skip if a completed run already exists, unless you force it.  
    # We check the pointer file for the last run's goal and ROI.
    run_goal_key = f"exhaustive_{goal_tag}"
    if not force and os.path.isfile(pointer_path):
        with open(pointer_path) as f:
            ptr = json.load(f)
        prev = ptr.get("flex_run_dir", "")
        if (ptr.get("goal") == run_goal_key
                and ptr.get("roi") == cfg.roi.name
                and os.path.isfile(f"{prev}/exhaustive_results.json")):
            print(f"  [SKIP] exhaustive search — found existing run:\n         {prev}")
            return prev

    os.makedirs(out_dir, exist_ok=True)

    # ════════════════════════════════════════════════════════════════════════
    # STEP 1 — Compute TDCS leadfield
    # ════════════════════════════════════════════════════════════════════════
    # If the leadfield already exists
    # just load the cached hdf5 file since it takes a lot of time to compute
    cap_basename = os.path.splitext(os.path.basename(cfg.eeg_csv_path))[0]
    el = cfg.electrode

    # Each distinct (cap, electrode shape/dimensions/gel_thickness) combination
    # gets its own subdirectory, keyed by leadfield_tag() — so switching
    # electrode settings and switching back later reuses the matching cache
    # instead of recomputing, and never collides with a different combination's
    # leftover SimNIBS bookkeeping files (.mat, logs) the way a single shared
    # directory did (the actual cause of the old "Found already existing
    # simulation results" crash on a settings change).
    tag         = leadfield_tag(cap_basename, el.shape, el.dimensions, el.gel_thickness)
    lf_dir      = f"{cfg.sim_sub_dir}/leadfield_volume/{tag}"
    # TDCSLEADFIELD names its own output file "{subject_id}_leadfield_{cap_basename}.hdf5"
    # (SimNIBS's own convention, based on subpath + eeg_cap — not ours to choose) —
    # only its DIRECTORY (lf_dir) is new; the filename inside it is unchanged.
    lf_hdf      = f"{lf_dir}/{cfg.subject_id}_leadfield_{cap_basename}.hdf5"
    lf_params   = f"{lf_dir}/{cfg.subject_id}_leadfield_{cap_basename}_params.json"

    current_lf_params = {
        "shape":        el.shape,
        "dimensions":   el.dimensions,
        "gel_thickness": el.gel_thickness,
        "tissues":      [1, 2, 3, 4, 5],
        "interpolation": None,
    }

    # Leadfield is always cached — even --force optimization skips it.
    # The params sidecar is written only on success, so its absence means a
    # prior run into this exact directory was interrupted — clean the whole
    # directory (not just the 2 files we know about) before recomputing, since
    # SimNIBS refuses to write into a directory containing leftover result
    # files from an incomplete run.
    lf_valid = False
    if os.path.isfile(lf_hdf) and os.path.isfile(lf_params):
        with open(lf_params) as _f:
            saved_params = json.load(_f)
        if saved_params == current_lf_params:
            lf_valid = True
            print(f"  [SKIP] Leadfield — already exists:\n         {lf_hdf}")
        else:
            # Shouldn't happen — the tag already encodes these settings — but
            # guard against a tag collision (e.g. future tag-format change)
            # by trusting the sidecar's actual content over the directory name.
            print(f"  WARNING: cached leadfield params don't match the directory's own tag — recomputing.")
            print(f"    Saved : {saved_params}")
            print(f"    Current: {current_lf_params}")
            import shutil
            shutil.rmtree(lf_dir, ignore_errors=True)
    elif os.path.isdir(lf_dir):
        # Directory exists but no valid (hdf5 + sidecar) pair — a previous
        # interrupted run. Clean it out entirely rather than just the hdf5.
        print(f"  Incomplete leadfield directory found — recomputing:\n         {lf_dir}")
        import shutil
        shutil.rmtree(lf_dir, ignore_errors=True)

    if not lf_valid:
        print(f"  Computing TDCS leadfield for {cap_basename} ...")
        print(f"  (This runs one FEM solve per electrode — runs once, then cached)")
        os.makedirs(lf_dir, exist_ok=True)

        lf_sess = TDCSLEADFIELD() # SimNIBS function to compute leadfield
        lf_sess.subpath  = cfg.m2m_path
        lf_sess.pathfem  = lf_dir

        # Point to the cap CSV so SimNIBS places all cap electrodes
        lf_sess.eeg_cap  = cfg.eeg_csv_path

        lf_sess.electrode.shape      = el.shape
        lf_sess.electrode.dimensions = el.dimensions
        lf_sess.electrode.thickness  = [el.gel_thickness]

        # Volume leadfield (no surface interpolation) so subcortical ROIs work.
        # Tags: 1=WM  2=GM  3=CSF  4=Bone  5=Scalp
        # The tissues of the leadfield will be all of them, 
        # but only later on in the analysis, we only want to take WM and GM tissues
        lf_sess.interpolation = None
        lf_sess.tissues       = [1, 2, 3, 4, 5]

        lf_sess.run(cpus=cfg.optimizer.cpus)

        if not os.path.isfile(lf_hdf):
            abort(f"Leadfield HDF5 not found after run: {lf_hdf}")
        with open(lf_params, "w") as _f:
            json.dump(current_lf_params, _f, indent=2)
        print(f"  Leadfield saved → {lf_hdf}")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2 — Load leadfield and find ROI elements
    # ════════════════════════════════════════════════════════════════════════
    print(f"\n  Loading leadfield ...")
    leadfield, mesh, idx_lf = TI.load_leadfield(lf_hdf)
    # leadfield: (N_elec-1, M, 3)  where M = number of volume elements in mesh
    # mesh:      full head mesh (used for saving the final TI field)
    # idx_lf:    dict { electrode_name -> index_in_leadfield }
    #            (reference electrode has index None)

    all_elec_names = list(idx_lf.keys())   # includes reference electrode
    n_elec = len(all_elec_names)
    print(f"  Electrodes: {n_elec}  ({', '.join(all_elec_names)})")
    print(f"  Leadfield shape: {leadfield.shape}")

    # Map our NIfTI ROI mask onto mesh elements
    roi_mask_path = cfg.mask_path(cfg.roi.name)
    if not os.path.isfile(roi_mask_path):
        abort(f"ROI mask not found: {roi_mask_path}. Run Section 1 first.")

    print(f"  Mapping ROI mask → mesh elements ...")
    centroids   = mesh.elements_baricenters().value  # (M, 3) — element centres in mm
    gm_wm_mask  = np.isin(mesh.elm.tag1, [1, 2])    # GM+WM only — consistent with TISSUE_TAGS

    mask_img  = nib.load(roi_mask_path)
    mask_data = np.asarray(mask_img.dataobj) > 0
    aff_inv   = np.linalg.inv(mask_img.affine)
    ones      = np.ones((len(centroids), 1))
    vox_coords = (aff_inv @ np.hstack([centroids, ones]).T).T[:, :3]
    vox_idx    = np.round(vox_coords).astype(int)

    sh = mask_data.shape
    in_bounds = ((vox_idx[:, 0] >= 0) & (vox_idx[:, 0] < sh[0]) &
                 (vox_idx[:, 1] >= 0) & (vox_idx[:, 1] < sh[1]) &
                 (vox_idx[:, 2] >= 0) & (vox_idx[:, 2] < sh[2]))
    roi_elm_mask = np.zeros(len(centroids), dtype=bool)
    roi_elm_mask[in_bounds] = mask_data[vox_idx[in_bounds, 0],
                                        vox_idx[in_bounds, 1],
                                        vox_idx[in_bounds, 2]]
    roi_elm_mask &= gm_wm_mask   # restrict to GM+WM
    roi_indices = np.flatnonzero(roi_elm_mask)

    if len(roi_indices) == 0:
        abort("ROI mask maps to 0 mesh elements — check that mask and mesh are "
              "in the same coordinate space.")
    print(f"  ROI elements in mesh: {len(roi_indices)}")

    # Extract leadfield at ROI elements only.
    # This reduces memory from (N×M×3) to (N×n_roi×3) — much smaller and faster.
    lf_roi = leadfield[:, roi_indices, :]   # (N_elec-1, n_roi, 3)

    # Non-ROI elements — only loaded when non_roi is defined AND goal="focality"
    use_focality  = cfg.non_roi is not None and cfg.optimizer.goal == "focality"
    lf_non_roi    = None
    non_roi_indices = np.array([], dtype=int)
    if use_focality:
        from simnibs.optimization.tes_flex_optimization.measures import ROC
        non_roi_mask_path = cfg.mask_path(cfg.non_roi.name)
        if not os.path.isfile(non_roi_mask_path):
            print(f"  WARNING: non-ROI mask not found ({non_roi_mask_path}) "
                  f"— falling back to mean scoring")
            use_focality = False
        else:
            nr_img   = nib.load(non_roi_mask_path)
            nr_data  = np.asarray(nr_img.dataobj) > 0
            nr_inv   = np.linalg.inv(nr_img.affine)
            nr_vox   = (nr_inv @ np.hstack([centroids, ones]).T).T[:, :3]
            nr_idx   = np.round(nr_vox).astype(int)
            nr_sh    = nr_data.shape
            nr_bounds = ((nr_idx[:, 0] >= 0) & (nr_idx[:, 0] < nr_sh[0]) &
                         (nr_idx[:, 1] >= 0) & (nr_idx[:, 1] < nr_sh[1]) &
                         (nr_idx[:, 2] >= 0) & (nr_idx[:, 2] < nr_sh[2]))
            nr_elm_mask = np.zeros(len(centroids), dtype=bool)
            nr_elm_mask[nr_bounds] = nr_data[nr_idx[nr_bounds, 0],
                                             nr_idx[nr_bounds, 1],
                                             nr_idx[nr_bounds, 2]]
            nr_elm_mask &= gm_wm_mask   # restrict to GM+WM
            non_roi_indices = np.flatnonzero(nr_elm_mask)
            if len(non_roi_indices) == 0:
                print(f"  WARNING: non-ROI mask maps to 0 elements — "
                      f"falling back to mean scoring")
                use_focality = False
            else:
                n_nr_total = len(non_roi_indices)
                cap = cfg.optimizer.max_non_roi_elements
                if cap > 0 and n_nr_total > cap:
                    rng = np.random.default_rng(seed=42)
                    non_roi_indices = rng.choice(non_roi_indices, cap, replace=False)
                    print(f"  Non-ROI elements in mesh: {n_nr_total} (subsampled to {cap})")
                else:
                    print(f"  Non-ROI elements in mesh: {n_nr_total}")
                lf_non_roi = leadfield[:, non_roi_indices, :]

    score_label = "ROC focality" if use_focality else "mean TI"
    hard_constraint = use_focality and cfg.optimizer.hard_roi_constraint
    print(f"  Scoring: {score_label}")
    if hard_constraint:
        print(f"  Hard ROI constraint: mean TI in ROI >= {cfg.optimizer.focality_threshold[1]} V/m"
              f" (soft penalty otherwise)")

    # ── Per-subgroup non-ROI hard constraints ────────────────────────────────
    nr_constraint_groups = []
    if cfg.optimizer.non_roi_hard_constraint_groups:
        bna_warped = os.path.join(cfg.sim_sub_dir, 'roi',
                                  f'sub-{cfg.subject_id}_BNA_atlas_subjectspace.nii.gz')
        if not os.path.isfile(bna_warped):
            print(f"  WARNING: BNA atlas not found — skipping subgroup constraints: {bna_warped}")
        else:
            _bna_img  = nib.load(bna_warped)
            _bna_data = np.asarray(_bna_img.dataobj, dtype=np.int32)
            _bna_inv  = np.linalg.inv(_bna_img.affine)
            _bna_vox  = (_bna_inv @ np.hstack([centroids, ones]).T).T[:, :3]
            _bna_idx  = np.round(_bna_vox).astype(int)
            _bna_sh   = _bna_data.shape
            _bna_ok   = ((_bna_idx[:,0]>=0)&(_bna_idx[:,0]<_bna_sh[0])&
                         (_bna_idx[:,1]>=0)&(_bna_idx[:,1]<_bna_sh[1])&
                         (_bna_idx[:,2]>=0)&(_bna_idx[:,2]<_bna_sh[2]))
            _elm_lbl  = np.zeros(len(centroids), dtype=np.int32)
            _elm_lbl[_bna_ok] = _bna_data[_bna_idx[_bna_ok,0],
                                           _bna_idx[_bna_ok,1],
                                           _bna_idx[_bna_ok,2]]
            for _grp in cfg.optimizer.non_roi_hard_constraint_groups:
                _labels  = list(_grp['bna_labels'].values())
                _gmask   = np.isin(_elm_lbl, _labels) & gm_wm_mask   # GM+WM only
                _gidx    = np.flatnonzero(_gmask)
                if len(_gidx) == 0:
                    print(f"  WARNING: constraint group '{_grp['name']}' maps to 0 elements — skipped")
                    continue
                nr_constraint_groups.append({
                    'name':     _grp['name'],
                    'lf':       leadfield[:, _gidx, :],
                    'max_mean': float(_grp['max_mean_V_m']),
                })
                print(f"  Subgroup constraint '{_grp['name']}': {len(_gidx)} elements, "
                      f"max mean TI <= {_grp['max_mean_V_m']} V/m")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 2b — Build electrode adjacency set (shunting prevention and source isolation)
    # ════════════════════════════════════════════════════════════════════════
    # adj_elec_pairs: set of frozenset({nameA, nameB}) for all cap-adjacent pairs.
    # A montage is rejected if ANY two of its four active electrodes are adjacent.
    adj_elec_pairs: set = set()

    # Electrode positions are needed both for adjacency filtering (below) and
    # for the hierarchical search (STEP 3) — parsed once, reused by whichever
    # (or both) is enabled.
    _cap_pos: dict = {}
    if cfg.optimizer.no_adjacent_electrodes or cfg.optimizer.use_hierarchical_search:
        _cap_pos = _load_cap_positions(cfg.eeg_csv_path)

    if cfg.optimizer.no_adjacent_electrodes:
        # Use hardcoded adjacency for known caps; fall back to geometry otherwise.
        _cap_name = os.path.splitext(os.path.basename(cfg.eeg_csv_path))[0].lower()
        _lf_set   = set(all_elec_names)
        if 'biosemi32' in _cap_name:
            for _elec, _nbs in _BIOSEMI32_ADJACENCY.items():
                if _elec in _lf_set:
                    for _nb in _nbs:
                        if _nb in _lf_set:
                            adj_elec_pairs.add(frozenset([_elec, _nb]))
            print(f"  Adjacent-electrode filter: {len(adj_elec_pairs)} pairs excluded "
                  f"(BioSemi32 hardcoded topology)")
        else:
            _lf_elecs = [n for n in all_elec_names if n in _cap_pos]
            if len(_lf_elecs) >= 3:
                from scipy.spatial import ConvexHull, Delaunay
                _pos_arr = np.array([_cap_pos[n] for n in _lf_elecs])
                for _simplex in ConvexHull(_pos_arr).simplices:
                    for _ia, _ib in combinations(_simplex.tolist(), 2):
                        adj_elec_pairs.add(frozenset([_lf_elecs[_ia], _lf_elecs[_ib]]))
                for _simplex in Delaunay(_pos_arr[:, :2]).simplices:
                    for _ia, _ib in combinations(_simplex.tolist(), 2):
                        adj_elec_pairs.add(frozenset([_lf_elecs[_ia], _lf_elecs[_ib]]))
                print(f"  Adjacent-electrode filter: {len(adj_elec_pairs)} pairs excluded "
                      f"(convex-hull + 2-D Delaunay, {len(_lf_elecs)} electrodes)")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 3 — Exhaustive search over all electrode pair combinations
    # ════════════════════════════════════════════════════════════════════════
    '''The part of the code where the exhaustive search is performed. 
    It evaluates all valid electrode pairs and computes the electric field at 
    the ROI and non-ROI elements, applying constraints as 
    mentioned in the config. The best scoring montage is tracked throughout 
    the search.'''
    current_A = cfg.electrode.current_mA * 1e-3   # mA → A

    def get_ef(lf_subset: np.ndarray, e_plus: str, e_minus: str) -> np.ndarray:
        """E-field at a subset of elements for one electrode pair."""
        return TI.get_field([e_plus, e_minus, current_A], lf_subset, idx_lf)

    # Pre-filter: build neighbour lookup from the (subset-independent) global
    # adjacency set once — reused unchanged by every round of search below,
    # since it only depends on cap topology, not on which electrodes are
    # currently being searched.
    nb_dict: dict = {}
    if adj_elec_pairs:
        for _p in adj_elec_pairs:
            _a, _b = list(_p)
            nb_dict.setdefault(_a, set()).add(_b)
            nb_dict.setdefault(_b, set()).add(_a)

    roi_min_threshold = cfg.optimizer.focality_threshold[1] if hard_constraint else None

    def _count_combos(valid_pairs: list) -> int:
        """Number of valid montages (pairs-of-pairs with no shared/adjacent electrode)."""
        n = 0
        for _i in range(len(valid_pairs)):
            _ep1, _em1 = valid_pairs[_i]
            _fbd = nb_dict.get(_ep1, set()) | nb_dict.get(_em1, set()) | {_ep1, _em1}
            for _ep2, _em2 in valid_pairs[_i + 1:]:
                if _ep2 not in _fbd and _em2 not in _fbd:
                    n += 1
        return n

    def _search(candidate_names: list) -> dict:
        """Exhaustive montage search restricted to candidate_names. Same core
        logic as the original flat search — called once with all electrodes
        for the default (non-hierarchical) path, or iteratively with growing/
        shrinking subsets for the hierarchical path."""
        pairs = list(combinations(candidate_names, 2))
        valid_pairs = ([p for p in pairs if frozenset(p) not in adj_elec_pairs]
                       if adj_elec_pairs else pairs)
        n_valid  = len(valid_pairs)
        n_combos = _count_combos(valid_pairs)
        print(f"\n  Searching {n_combos} montages ({n_valid} valid single-channel "
              f"pairs, {len(candidate_names)} electrodes) ...")

        best_score  = -np.inf
        best_ch1 = best_ch2 = None
        # Fallback: best mean-ROI montage, used when hard_roi_constraint=True but no
        # montage meets the threshold — we still return something rather than crashing.
        fallback_score = -np.inf
        fallback_ch1 = fallback_ch2 = None
        n_feasible = 0
        n_eval     = 0
        t_start    = time.time()

        for i, (ep1, em1) in enumerate(valid_pairs):
            ef1_roi    = get_ef(lf_roi, ep1, em1)
            ef1_cgrps  = [get_ef(cg['lf'], ep1, em1) for cg in nr_constraint_groups]
            # Forbidden inner-pair electrodes: shared electrode or cross-channel adjacent
            forbidden = nb_dict.get(ep1, set()) | nb_dict.get(em1, set()) | {ep1, em1}

            for ep2, em2 in valid_pairs[i + 1:]:
                if ep2 in forbidden or em2 in forbidden:
                    continue   # skip: shared electrode or cross-channel adjacency

                ef2_roi = get_ef(lf_roi, ep2, em2)
                ti_roi  = TI.get_maxTI(ef1_roi, ef2_roi)
                n_eval += 1

                if use_focality:
                    ef1_nr  = get_ef(lf_non_roi, ep1, em1)
                    ef2_nr  = get_ef(lf_non_roi, ep2, em2)
                    ti_nr   = TI.get_maxTI(ef1_nr, ef2_nr)

                    if hard_constraint:
                        roi_mean = float(np.mean(ti_roi))
                        if roi_mean > fallback_score:
                            fallback_score = roi_mean
                            fallback_ch1   = (ep1, em1)
                            fallback_ch2   = (ep2, em2)
                        if roi_mean < roi_min_threshold:
                            continue   # hard constraint: below minimum dose, skip

                    # Per-subgroup non-ROI hard constraints
                    if nr_constraint_groups:
                        _violated = False
                        for _cg, _ef1_cg in zip(nr_constraint_groups, ef1_cgrps):
                            _ef2_cg = get_ef(_cg['lf'], ep2, em2)
                            if float(np.mean(TI.get_maxTI(_ef1_cg, _ef2_cg))) > _cg['max_mean']:
                                _violated = True
                                break
                        if _violated:
                            continue

                    if hard_constraint:
                        n_feasible += 1

                    # ROC returns a distance to the ideal point — lower is better.
                    # Negate so higher score = better (consistent with mean case).
                    score = -ROC(ti_roi, ti_nr,
                                 cfg.optimizer.focality_threshold, focal=True)
                else:
                    score = float(np.mean(ti_roi))

                if score > best_score:
                    best_score = score
                    best_ch1   = (ep1, em1)
                    best_ch2   = (ep2, em2)

        elapsed = time.time() - t_start

        used_fallback = hard_constraint and best_ch1 is None
        if used_fallback:
            print(f"\n  WARNING: No montage meets hard ROI constraint "
                  f"(mean TI >= {roi_min_threshold} V/m). "
                  f"Falling back to best mean-ROI montage.")
            best_ch1   = fallback_ch1
            best_ch2   = fallback_ch2
            best_score = fallback_score

        rate = f"{n_eval/elapsed:.0f}/s" if elapsed > 1e-9 else "n/a"
        print(f"  Evaluated {n_eval} montages in {elapsed:.1f}s ({rate})")
        if hard_constraint:
            print(f"  Feasible montages (ROI >= {roi_min_threshold} V/m): {n_feasible}")

        return {
            "best_ch1": best_ch1, "best_ch2": best_ch2, "best_score": best_score,
            "n_feasible": n_feasible, "n_eval": n_eval, "elapsed": elapsed,
            "n_valid_pairs": n_valid, "n_combos": n_combos, "used_fallback": used_fallback,
        }

    # ── Coarse-to-fine (hierarchical) vs flat exhaustive search ────────────
    history = None
    if cfg.optimizer.use_hierarchical_search:
        n_total  = len(all_elec_names)
        coarse_k = max(round(0.5 * n_total), min(n_total, 32))
        missing  = [nm for nm in all_elec_names if nm not in _cap_pos]
        if missing:
            abort(f"Hierarchical search needs electrode positions for every cap "
                  f"electrode — missing from {cfg.eeg_csv_path}: {missing}")

        n_fine = cfg.optimizer.num_fine_iterations
        neighbours_per_iter = cfg.optimizer.neighbours_per_iteration
        if n_fine and len(neighbours_per_iter) != n_fine:
            abort(f"optimizer.neighbours_per_iteration must have exactly {n_fine} "
                  f"entries (num_fine_iterations), got {len(neighbours_per_iter)}")

        print(f"\n  Hierarchical (coarse-to-fine) search — {n_fine} fine iteration(s) configured")
        print(f"  Coarse round: {coarse_k}/{n_total} electrodes (farthest-point spatial sampling)")
        coarse_names = _farthest_point_sample(all_elec_names, _cap_pos, coarse_k)
        round_result = _search(coarse_names)
        history = [{"round": "coarse", "n_electrodes": len(coarse_names),
                    **{k: v for k, v in round_result.items() if k not in ("best_ch1", "best_ch2")}}]

        candidate_set = set(coarse_names)
        for it in range(n_fine):
            n_nb    = neighbours_per_iter[it]
            winners = [round_result["best_ch1"][0], round_result["best_ch1"][1],
                       round_result["best_ch2"][0], round_result["best_ch2"][1]]
            new_candidates = set(candidate_set)
            for w in winners:
                new_candidates.add(w)
                new_candidates.update(_nearest_neighbours(w, all_elec_names, _cap_pos, n_nb))

            if new_candidates == candidate_set:
                print(f"\n  Fine iteration {it + 1}/{n_fine}: neighbour expansion added no "
                      f"new electrodes to the candidate set — stopping early.")
                break

            candidate_set = new_candidates
            print(f"\n  Fine iteration {it + 1}/{n_fine}: {len(candidate_set)} electrodes "
                  f"(4 winners x {n_nb} nearest neighbours, deduplicated with prior candidates)")
            new_result = _search(sorted(candidate_set, key=all_elec_names.index))
            history.append({"round": f"fine_{it + 1}", "n_electrodes": len(candidate_set),
                             **{k: v for k, v in new_result.items() if k not in ("best_ch1", "best_ch2")}})

            prev_score  = round_result["best_score"]
            improvement = ((new_result["best_score"] - prev_score) / abs(prev_score)
                            if prev_score != 0 else float("inf"))
            round_result = new_result
            print(f"  Round score: {round_result['best_score']:.4f} ({improvement * 100:+.1f}% vs previous round)")

            if improvement < cfg.optimizer.early_stop_threshold:
                print(f"  Improvement below the {cfg.optimizer.early_stop_threshold * 100:.0f}% "
                      f"threshold — stopping early.")
                break

        final = round_result
    else:
        final = _search(all_elec_names)

    best_ch1, best_ch2, best_score = final["best_ch1"], final["best_ch2"], final["best_score"]
    n_eval     = sum(h["n_eval"] for h in history) if history else final["n_eval"]
    elapsed    = sum(h["elapsed"] for h in history) if history else final["elapsed"]
    n_feasible = sum(h["n_feasible"] for h in history) if history else final["n_feasible"]

    print(f"\n  ══ Best montage ══")
    print(f"  Ch1: {best_ch1[0]}+ / {best_ch1[1]}-  @ {cfg.electrode.current_mA} mA")
    print(f"  Ch2: {best_ch2[0]}+ / {best_ch2[1]}-  @ {cfg.electrode.current_mA} mA")
    print(f"  Score ({score_label}): {best_score:.4f}")

    # ════════════════════════════════════════════════════════════════════════
    # STEP 4 — Save the best TI field on the full head mesh
    # ════════════════════════════════════════════════════════════════════════
    # Recompute best TI field on ALL mesh elements (not just ROI) for visualization
    ef1_full = TI.get_field([best_ch1[0], best_ch1[1], current_A], leadfield, idx_lf)
    ef2_full = TI.get_field([best_ch2[0], best_ch2[1], current_A], leadfield, idx_lf)
    ti_full  = TI.get_maxTI(ef1_full, ef2_full)   # (M,) TI amplitude at every element

    # Attach TI field to the mesh and save — same filename as TesFlexOptimization
    # so that run_analysis() and run_visualization() pick it up automatically.
    mesh.elmdata.append(mesh_io.ElementData(ti_full, "max_TI"))
    msh_out = f"{out_dir}/{cfg.subject_id}_tes_flex_opt_head_mesh.msh"
    mesh_io.write_msh(mesh, msh_out)
    print(f"\n  Full-brain TI mesh saved → {msh_out}")

    # ── Save results summary ──────────────────────────────────────────────
    # Actual mean TI in ROI for the best montage (always in V/m, regardless of goal)
    roi_mean_V_m = float(np.mean(ti_full[roi_indices]))

    results = {
        "method":            "exhaustive_cap_search",
        "scoring":           "ROC_focality" if use_focality else "mean_TI",
        "roi":               cfg.roi.name,
        "non_roi":           cfg.non_roi.name if use_focality else None,
        "n_electrodes":      n_elec,
        "n_montages_searched": n_eval,
        "no_adjacent_electrodes": cfg.optimizer.no_adjacent_electrodes,
        "n_adjacent_pairs_excluded": len(adj_elec_pairs),
        "elapsed_s":         round(elapsed, 1),
        "best_montage": {
            "ch1_plus":    best_ch1[0],
            "ch1_minus":   best_ch1[1],
            "ch2_plus":    best_ch2[0],
            "ch2_minus":   best_ch2[1],
            "current_mA":  cfg.electrode.current_mA,
        },
        "roi_TI_mean_V_m": round(roi_mean_V_m, 6),
    }
    if history is not None:
        results["hierarchical_search"] = {
            "coarse_electrodes": history[0]["n_electrodes"],
            "num_fine_iterations_configured": cfg.optimizer.num_fine_iterations,
            "num_fine_iterations_run": len(history) - 1,
            "early_stop_threshold": cfg.optimizer.early_stop_threshold,
            "rounds": [{k: v for k, v in h.items() if k != "used_fallback"} for h in history],
        }
    if use_focality:
        results["focality_roc_score"]       = round(best_score, 6)
        results["focality_threshold"]       = cfg.optimizer.focality_threshold
        results["hard_roi_constraint"]      = hard_constraint
        if hard_constraint:
            results["n_feasible_montages"]  = n_feasible
            results["roi_threshold_met"]    = roi_mean_V_m >= cfg.optimizer.focality_threshold[1]
    res_path = f"{out_dir}/exhaustive_results.json"
    with open(res_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved → {res_path}")

    # ── Write pointer so the rest of the pipeline finds this run ─────────
    pointer = {
        "flex_run_dir": out_dir,
        "roi":          cfg.roi.name,
        "goal":         run_goal_key,
        "postproc":     "max_TI",
        "timestamp":    timestamp,
        "slurm_job_id": job_id or "local",
    }
    with open(pointer_path, "w") as f:
        json.dump(pointer, f, indent=2)
    print(f"  Pointer saved → {pointer_path}")

    return out_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Section 2 — Cap Optimization (exhaustive search only)
# ═══════════════════════════════════════════════════════════════════════════════


def run_optimization(cfg: PipelineConfig, force: bool = False,
                     job_id: str = "") -> str:
    """Returns the opt_run_dir path. Exhaustive cap search is now the only
    optimization path (see run_exhaustive_cap_optimization) — the DE-based
    TesFlexOptimization path below is unused in practice and commented out,
    not deleted (see _run_optimization_tesflex_LEGACY)."""
    return run_exhaustive_cap_optimization(cfg, force=force, job_id=job_id)


# LEGACY — TesFlexOptimization (differential-evolution, continuous electrode
# search). Superseded by run_exhaustive_cap_optimization(): exhaustive search
# computes one leadfield and algebraically evaluates every discrete cap
# electrode pair, which is faster and gives a guaranteed global optimum over
# cap positions, so this path hasn't been used since. Kept commented out
# (not deleted) in case continuous (non-cap-constrained) electrode
# placement is ever needed again — note it also reads several
# OptimizerConfig fields (max_iterations, population_size, tolerance,
# mutation, recombination, n_multistart, anisotropy_type,
# min_electrode_distance, use_eeg_cap_boundary, eeg_cap_margin_mm,
# detailed_results, enable_mapping, use_exhaustive_search) that were removed
# from the dataclass alongside this — restore both together if reviving it.
#
# def _run_optimization_tesflex_LEGACY(cfg: PipelineConfig, force: bool = False,
#                      job_id: str = "") -> str:
#     """Returns the opt_run_dir path."""
#     header("Section 2 — TesFlexOptimization")
#
#     # Check pointer file for existing run with matching ROI
#     pointer_path = f"{cfg.sim_sub_dir}/latest_flex_run.json"
#     if not force and os.path.isfile(pointer_path):
#         with open(pointer_path) as f:
#             ptr = json.load(f)
#         existing = ptr.get("flex_run_dir", "")
#         mesh = f"{existing}/{cfg.subject_id}_tes_flex_opt_head_mesh.msh"
#         if os.path.isfile(mesh) and ptr.get("roi") == cfg.roi.name:
#             print(f"  [SKIP] optimization — found existing run:\n         {existing}")
#             return existing
#         elif os.path.isfile(mesh) and ptr.get("roi") != cfg.roi.name:
#             print(f"  Previous run was for ROI '{ptr.get('roi')}', "
#                   f"current ROI is '{cfg.roi.name}' — running new optimization.")
#
#     import copy
#     import shutil
#     from simnibs.optimization.tes_flex_optimization.tes_flex_optimization import TesFlexOptimization
#     from simnibs.optimization.tes_flex_optimization.tes_flex_optimization import ElectrodeArrayPair
#     from simnibs.utils.region_of_interest import RegionOfInterest
#
#     # ── Build a RegionOfInterest object from an ROIConfig ────────────────────
#     def _build_roi(roi_cfg: ROIConfig):
#         roi = RegionOfInterest()
#         roi.subpath = cfg.m2m_path
#         if roi_cfg.method == "NIfTI":
#             mask_path = cfg.mask_path(roi_cfg.name)
#             if not os.path.isfile(mask_path):
#                 abort(f"ROI mask not found: {mask_path}. Run Section 1 first.")
#             roi.method     = "volume"
#             roi.mask_path  = mask_path
#             roi.mask_space = "subject"
#             roi.tissues    = [2]
#         elif roi_cfg.method == "atlas":
#             roi.atlas_regions = roi_cfg.atlas_regions
#         elif roi_cfg.method == "sphere":
#             roi.method = "sphere"
#             roi.center = roi_cfg.sphere_center
#             roi.radius = roi_cfg.sphere_radius
#         return roi
#
#     # ── Build electrode pair ──────────────────────────────────────────────────
#     el  = cfg.electrode
#     ep  = ElectrodeArrayPair()
#     ep.center  = [[0, 0]]
#     ep.current = [el.current_mA * 1e-3, -el.current_mA * 1e-3]
#     if el.shape in ("ellipse", "circle"):
#         ep.radius   = [el.dimensions[0] / 2]
#     else:
#         ep.length_x = [el.dimensions[0]]
#         ep.length_y = [el.dimensions[1]]
#         ep.radius   = None
#
#     # ── Paths ─────────────────────────────────────────────────────────────────
#     timestamp      = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
#     flex_dir       = cfg.ti_opt_dir
#     postproc_bids  = cfg.optimizer.postproc.replace("_", "")
#     op_tmp         = cfg.optimizer
#     scalp_tag      = ("boundary" if op_tmp.use_eeg_cap_boundary else "full")
#     job_tag        = f"_job-{job_id}" if job_id else ""
#     opt_output_dir = (f"{flex_dir}/sub-{cfg.subject_id}_roi-{cfg.roi.name}"
#                       f"_goal-{cfg.optimizer.goal}_postproc-{postproc_bids}"
#                       f"_scalp-{scalp_tag}{job_tag}_run-{timestamp}")
#     os.makedirs(opt_output_dir, exist_ok=True)
#
#     # ── Build ROI and optimizer ───────────────────────────────────────────────
#     target_roi = _build_roi(cfg.roi)
#     avoid_roi  = _build_roi(cfg.non_roi) if cfg.non_roi else None
#     op         = cfg.optimizer
#
#     # detailed_results crashes in SimNIBS for non-focality goals
#     if op.detailed_results and op.goal not in ("focality", "focality_inv"):
#         print(f"  WARNING: detailed_results=True is only supported with goal='focality'. "
#               f"Disabling for goal='{op.goal}'.")
#         object.__setattr__(op, "detailed_results", False)
#
#     def _build_opt(output_folder: str) -> TesFlexOptimization:
#         opt = TesFlexOptimization()
#         opt.subpath        = cfg.m2m_path
#         opt.output_folder  = output_folder
#         opt.goal           = [op.goal]
#         opt.e_postproc     = op.postproc
#         if op.goal == "focality" and avoid_roi is not None:
#             opt.roi       = [target_roi, avoid_roi]
#             opt.threshold = op.focality_threshold
#         else:
#             opt.roi = [target_roi]
#             if op.goal == "focality":
#                 print("  WARNING: focality goal set but no non_roi defined — "
#                       "running without avoidance region.")
#         opt.electrode                        = [ep, copy.deepcopy(ep)]
#         opt.anisotropy_type                  = op.anisotropy_type
#         opt.min_electrode_distance           = op.min_electrode_distance
#         opt.detailed_results                 = op.detailed_results
#         opt.optimizer_options                = {
#             "maxiter":       op.max_iterations,
#             "popsize":       op.population_size,
#             "tol":           op.tolerance,
#             "mutation":      op.mutation,
#             "recombination": op.recombination,
#         }
#         opt.map_to_net_electrodes            = op.enable_mapping
#         opt.run_mapped_electrodes_simulation = op.enable_mapping
#         if op.enable_mapping:
#             opt.net_electrode_file = cfg.eeg_csv_path
#         opt.open_in_gmsh = False
#         return opt
#
#     # ── Multistart loop (following TIT flex.py pattern) ───────────────────────
#     n       = op.n_multistart
#     folders = ([f"{opt_output_dir}/{i:02d}" for i in range(n)]
#                if n > 1 else [opt_output_dir])
#     fvals   = np.full(n, float("inf"))
#
#     if n > 1:
#         print(f"  Multistart: {n} independent runs — keeping best (argmin funvalue)")
#
#     for run_i in range(n):
#         if n > 1:
#             print(f"\n  ── Start {run_i + 1}/{n} → {folders[run_i]}")
#         os.makedirs(folders[run_i], exist_ok=True)
#         opt = _build_opt(folders[run_i])
#         opt._prepare()
#         if op.use_eeg_cap_boundary:
#             apply_eeg_cap_boundary(opt, cfg.eeg_csv_path,
#                                    margin_mm=op.eeg_cap_margin_mm)
#         opt.run(cpus=op.cpus)
#         fvals[run_i] = getattr(opt, "optim_funvalue", float("inf"))
#         if n > 1:
#             print(f"  Start {run_i + 1} funvalue: {fvals[run_i]:.6f}")
#
#         # Delete any previous subfolders that are no longer the best.
#         # This frees disk space immediately — each subfolder contains large
#         # final_sim_0/ and final_sim_1/ meshes that accumulate across runs.
#         if n > 1 and run_i > 0:
#             current_best = int(np.argmin(fvals[:run_i + 1]))
#             for j in range(run_i):
#                 if j != current_best and os.path.isdir(folders[j]):
#                     shutil.rmtree(folders[j])
#                     print(f"  Freed subfolder {j:02d}/ (not best)")
#
#     # Select best, promote to base folder, clean up subfolders
#     if n > 1:
#         best_idx = int(np.argmin(fvals))
#         print(f"\n  Best start: #{best_idx + 1} (funvalue={fvals[best_idx]:.6f})")
#         best_folder = folders[best_idx]
#         for item in os.listdir(best_folder):
#             src = os.path.join(best_folder, item)
#             dst = os.path.join(opt_output_dir, item)
#             if os.path.isdir(src):
#                 if os.path.exists(dst):
#                     shutil.rmtree(dst)
#                 shutil.copytree(src, dst)
#             else:
#                 shutil.copy2(src, dst)
#         for folder in folders:
#             if os.path.isdir(folder):
#                 shutil.rmtree(folder)
#
#     # ── dataset_description.json ──────────────────────────────────────────────
#     desc = {
#         "Name":        "SimNIBS TI Optimization",
#         "BIDSVersion": "1.9.0",
#         "DatasetType": "derivative",
#         "GeneratedBy": [
#             {
#                 "Name":       "SimNIBS TesFlexOptimization",
#                 "Container":  {"Type": "apptainer", "Image": "simnibs_v2.3.0.sif"},
#                 "ROI":        cfg.roi.name,
#                 "Goal":       op.goal,
#                 "Postproc":   op.postproc,
#                 "Multistart": n,
#                 "BestFunvalue": float(np.min(fvals)),
#             },
#             {
#                 "Name":       "run_pipeline.py",
#                 "SlurmJobID": job_id or "local",
#                 "Timestamp":  timestamp,
#             },
#         ],
#     }
#     with open(f"{opt_output_dir}/dataset_description.json", "w") as f:
#         json.dump(desc, f, indent=2)
#     print(f"  dataset_description.json written → {opt_output_dir}/dataset_description.json")
#
#     # ── Pointer file ──────────────────────────────────────────────────────────
#     pointer = {
#         "flex_run_dir": opt_output_dir,
#         "roi":          cfg.roi.name,
#         "goal":         op.goal,
#         "postproc":     op.postproc,
#         "timestamp":    timestamp,
#         "slurm_job_id": job_id or "local",
#     }
#     with open(pointer_path, "w") as f:
#         json.dump(pointer, f, indent=2)
#     print(f"  Pointer saved → {pointer_path}")
#     return opt_output_dir


# ═══════════════════════════════════════════════════════════════════════════════
# Section 3 — FEM simulations
# ═══════════════════════════════════════════════════════════════════════════════

def run_simulation(cfg: PipelineConfig, flex_run_dir: str,
                   force: bool = False) -> None:
    header("Section 3 — FEM simulations")

    from simnibs import sim_struct, run_simnibs

    sim_runs = {}
    mode = cfg.simulation.simulate_mode

    if mode in ("optimized", "both"):
        sim_runs["optimized"] = {
            "opt_mesh": f"{flex_run_dir}/{cfg.subject_id}_tes_flex_opt_head_mesh.msh",
            "sim0_dir": f"{flex_run_dir}/final_sim_0",
            "sim1_dir": f"{flex_run_dir}/final_sim_1",
        }
    if mode in ("mapped", "both"):
        sim_runs["mapped"] = {
            "opt_mesh": (f"{flex_run_dir}/mapped_electrodes_simulation"
                         f"/{cfg.subject_id}_tes_mapped_opt_head_mesh.msh"),
            "sim0_dir": f"{flex_run_dir}/mapped_electrodes_simulation/mapped_sim_0",
            "sim1_dir": f"{flex_run_dir}/mapped_electrodes_simulation/mapped_sim_1",
        }

    el = cfg.electrode
    for label, info in sim_runs.items():
        mesh1 = f"{info['sim0_dir']}/{cfg.subject_id}_TDCS_1_scalar.msh"
        mesh2 = f"{info['sim1_dir']}/{cfg.subject_id}_TDCS_1_scalar.msh"
        if not force and os.path.isfile(mesh1) and os.path.isfile(mesh2):
            print(f"  [SKIP] {label} simulation — meshes exist")
            info["mesh1"] = mesh1
            info["mesh2"] = mesh2
            continue

        if not os.path.isfile(info["opt_mesh"]):
            print(f"  [SKIP] {label} — opt mesh not found: {info['opt_mesh']}")
            continue

        for sim_idx, (out_dir, current_sign) in enumerate(
                [(info["sim0_dir"], 1), (info["sim1_dir"], -1)]):
            os.makedirs(out_dir, exist_ok=True)
            s = sim_struct.SESSION()
            s.subpath   = cfg.m2m_path
            s.pathfem   = out_dir
            s.open_in_gmsh = False
            tdcs = s.add_tdcslist()
            tdcs.currents = [el.current_mA * current_sign,
                             -el.current_mA * current_sign]
            # electrodes loaded from opt mesh
            tdcs.mesh_file = info["opt_mesh"]
            run_simnibs(s)

        info["mesh1"] = mesh1
        info["mesh2"] = mesh2
        print(f"  {label} simulation complete.")


# ═══════════════════════════════════════════════════════════════════════════════
# Section 4 — Analysis
# ═══════════════════════════════════════════════════════════════════════════════

TISSUE_TAGS = [1, 2]            # WM, GM only — analysis restricted to brain tissue
# LEGACY: TISSUE_TAGS = [1, 2, 3, 4, 5]  # WM, GM, CSF, Skull, Scalp in SimNIBS meshes

# BioSemi32 electrode adjacency derived from the BioSemi64 topology (48 pairs).
# Two B32 electrodes are adjacent iff they are directly connected in the B64 graph.
_BIOSEMI32_ADJACENCY = {
    'AF3': ['F3','Fp1'],        'AF4': ['F4','Fp2'],
    'C3':  ['CP1','CP5','FC1','FC5'],  'C4':  ['CP2','CP6','FC2','FC6'],
    'CP1': ['C3','Cz','P3','Pz'],      'CP2': ['C4','Cz','P4','Pz'],
    'CP5': ['C3','P3','P7','T7'],      'CP6': ['C4','P4','P8','T8'],
    'Cz':  ['CP1','CP2','FC1','FC2'],
    'F3':  ['AF3','FC1','FC5'],        'F4':  ['AF4','FC2','FC6'],
    'F7':  ['FC5'],                    'F8':  ['FC6'],
    'FC1': ['C3','Cz','F3','Fz'],      'FC2': ['C4','Cz','F4','Fz'],
    'FC5': ['C3','F3','F7','T7'],      'FC6': ['C4','F4','F8','T8'],
    'Fp1': ['AF3'],                    'Fp2': ['AF4'],
    'Fz':  ['FC1','FC2'],
    'O1':  ['Oz','PO3'],               'O2':  ['Oz','PO4'],
    'Oz':  ['O1','O2','PO3','PO4'],
    'P3':  ['CP1','CP5','PO3'],        'P4':  ['CP2','CP6','PO4'],
    'P7':  ['CP5','PO3'],              'P8':  ['CP6','PO4'],
    'PO3': ['O1','Oz','P3','P7','Pz'], 'PO4': ['O2','Oz','P4','P8','Pz'],
    'Pz':  ['CP1','CP2','PO3','PO4'],
    'T7':  ['CP5','FC5'],              'T8':  ['CP6','FC6'],
}

TISSUE_NAMES = {1: "WM", 2: "GM", 3: "CSF", 4: "Skull", 5: "Scalp"}


def _mask_nifti_to_elements(mask_path: str, centroids: np.ndarray) -> np.ndarray:
    """Map a binary NIfTI mask to mesh element indices.

    Parameters
    ----------
    mask_path : str
        Path to binary NIfTI mask file.
    centroids : np.ndarray, shape (N, 3)
        World-space (mm) coordinates of mesh element barycentres.

    Returns
    -------
    np.ndarray, shape (N,), dtype bool
        True for elements whose barycentre falls inside the mask.
    """
    mask_img  = nib.load(mask_path)
    mask_data = np.asarray(mask_img.dataobj) > 0
    aff_inv   = np.linalg.inv(mask_img.affine)
    ones      = np.ones((len(centroids), 1))
    vox       = (aff_inv @ np.hstack([centroids, ones]).T).T[:, :3]
    idx       = np.round(vox).astype(int)
    sh        = mask_data.shape
    in_bounds = ((idx[:, 0] >= 0) & (idx[:, 0] < sh[0]) &
                 (idx[:, 1] >= 0) & (idx[:, 1] < sh[1]) &
                 (idx[:, 2] >= 0) & (idx[:, 2] < sh[2]))
    out = np.zeros(len(centroids), dtype=bool)
    out[in_bounds] = mask_data[idx[in_bounds, 0],
                               idx[in_bounds, 1],
                               idx[in_bounds, 2]]
    return out


def _load_ti_field(msh_path: str):
    """Load max_TI field on tissue elements from an optimizer mesh."""
    from simnibs import mesh_io
    m = mesh_io.read_msh(msh_path)
    ti_field = next(
        (d for d in reversed(m.elmdata) if d.field_name in ("max_TI", "TI_max")), None
    )
    if ti_field is None:
        raise ValueError(f"No TI field found in {msh_path}")
    tissue_mask = np.isin(m.elm.tag1, TISSUE_TAGS)
    return ti_field.value[tissue_mask], m, tissue_mask


def run_analysis(cfg: PipelineConfig, flex_run_dir: str,
                 force: bool = False) -> None:
    header("Section 4 — TI field analysis")

    from simnibs import mesh_io

    analysis_out = f"{flex_run_dir}/analysis"
    summary_path = f"{analysis_out}/summary.json"
    if not force and os.path.isfile(summary_path):
        print(f"  [SKIP] analysis — {summary_path} exists")
        return

    os.makedirs(analysis_out, exist_ok=True)

    roi_mask_path = cfg.mask_path(cfg.roi.name)

    analysis_runs = {}
    if cfg.simulation.simulate_mode in ("optimized", "both"):
        analysis_runs["optimized"] = \
            f"{flex_run_dir}/{cfg.subject_id}_tes_flex_opt_head_mesh.msh"
    if cfg.simulation.simulate_mode in ("mapped", "both"):
        analysis_runs["mapped"] = (
            f"{flex_run_dir}/mapped_electrodes_simulation"
            f"/{cfg.subject_id}_tes_mapped_opt_head_mesh.msh"
        )

    results = {}
    for label, msh_path in analysis_runs.items():
        if not os.path.isfile(msh_path):
            print(f"  [SKIP] {label} — mesh not found")
            continue

        ti, m, tissue_mask = _load_ti_field(msh_path)
        m_tissue  = m.crop_mesh(tags=TISSUE_TAGS)
        tags      = m_tissue.elm.tag1
        nodes     = m_tissue.nodes.node_coord
        conn      = m_tissue.elm.node_number_list[:, :4] - 1
        centroids = nodes[conn].mean(axis=1)
        v0 = nodes[conn[:, 0]]; v1 = nodes[conn[:, 1]]
        v2 = nodes[conn[:, 2]]; v3 = nodes[conn[:, 3]]
        elm_vols  = np.abs(np.einsum('ni,ni->n',
            v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0   # mm³

        # ROI mask
        roi_elm_mask = (_mask_nifti_to_elements(roi_mask_path, centroids)
                        if os.path.isfile(roi_mask_path)
                        else np.zeros(len(centroids), dtype=bool))

        # non-ROI mask (if defined)
        non_roi_mask_path = (cfg.mask_path(cfg.non_roi.name)
                             if cfg.non_roi else None)
        non_roi_elm_mask  = (_mask_nifti_to_elements(non_roi_mask_path, centroids)
                             if non_roi_mask_path and os.path.isfile(non_roi_mask_path)
                             else np.zeros(len(centroids), dtype=bool))

        ti_roi     = ti[roi_elm_mask]
        ti_non_roi = ti[non_roi_elm_mask]

        brain_vol_mean = _vol_mean_capped(ti, elm_vols)
        if roi_elm_mask.any():
            roi_vol_mean = _vol_mean_capped(ti_roi, elm_vols[roi_elm_mask])
            res = {
                "TI_max_whole_brain_V_m":    float(ti.max()),
                "TI_mean_whole_brain_V_m":   brain_vol_mean,
                "TI_max_ROI_V_m":            float(ti_roi.max()),
                "TI_mean_ROI_V_m":           roi_vol_mean,
                "focality_ratio_brain":      roi_vol_mean / brain_vol_mean,
            }
            if non_roi_elm_mask.any():
                nr_vol_mean = _vol_mean_capped(ti_non_roi, elm_vols[non_roi_elm_mask])
                res["TI_mean_non_ROI_V_m"]      = nr_vol_mean
                res["focality_ratio_non_roi"]   = roi_vol_mean / nr_vol_mean
            results[label] = res
        else:
            results[label] = {
                "TI_max_whole_brain_V_m":  float(ti.max()),
                "TI_mean_whole_brain_V_m": brain_vol_mean,
            }
            print(f"  WARNING: ROI mask not found — skipping ROI stats")

        # Extra ROIs — mean and max only, no focality
        extra_masks = {}
        for extra in cfg.extra_rois:
            extra_mask_path = cfg.mask_path(extra.name)
            if os.path.isfile(extra_mask_path):
                emask = _mask_nifti_to_elements(extra_mask_path, centroids)
                ti_extra = ti[emask]
                if emask.any():
                    results[label][f"TI_max_{extra.name}_V_m"]  = float(ti_extra.max())
                    results[label][f"TI_mean_{extra.name}_V_m"] = _vol_mean_capped(ti_extra, elm_vols[emask])
                    print(f"    {label} extra ROI {extra.name}: mean={ti_extra.mean():.4f}  max={ti_extra.max():.4f}")
                extra_masks[extra.name] = emask
            else:
                print(f"  WARNING: extra ROI mask not found — {extra_mask_path}")

        # Save raw per-element TI values + region masks for full distributional analysis
        # tag1: 1=WM, 2=GM, 3=CSF, 4=Skull, 5=Scalp
        # roi_mask / non_roi_mask / extra_roi_{name}_mask: boolean arrays
        # Example: ti_all[roi_mask & (tag1 == 2)]  →  TI in ROI GM only
        npz_data = dict(ti_all=ti, tag1=tags,
                        roi_mask=roi_elm_mask, non_roi_mask=non_roi_elm_mask)
        for name, emask in extra_masks.items():
            npz_data[f"extra_roi_{name}_mask"] = emask
        npz_path = f"{analysis_out}/ti_field_{label}.npz"
        np.savez_compressed(npz_path, **npz_data)
        print(f"  Raw field saved → {npz_path}")

        for k, v in results[label].items():
            print(f"    {label} {k}: {v:.4f}")

    with open(summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Summary saved → {summary_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# Visualization — Gmsh export mesh
# ═══════════════════════════════════════════════════════════════════════════════

def run_visualization(cfg: PipelineConfig, flex_run_dir: str,
                      force: bool = False) -> None:
    """Export a Gmsh-ready visualization mesh with selectable Views.

    Replaces the old TI_comparison.msh with a single mesh file that carries
    three independently toggleable element data Views in Gmsh:

        View 0 — TI amplitude [V/m]  : TI field on all tissue elements
        View 1 — ROI                  : 1 inside the target ROI, 0 elsewhere
        View 2 — non-ROI (if defined) : 1 inside the avoidance region, 0 elsewhere

    Tissue geometry (WM/GM/CSF/Skull/Scalp) is accessible via Gmsh's
    Tools > Visibility > Physical groups panel.

    A companion .geo loader script is also written; open it in Gmsh to load
    the mesh with a brief usage guide in comments.
    """
    header("Visualization — Gmsh export mesh")

    from simnibs import mesh_io

    stem       = f"sub-{cfg.subject_id}_roi-{cfg.roi.name}_visualization"
    msh_out    = f"{flex_run_dir}/{stem}.msh"
    geo_out    = f"{flex_run_dir}/{stem}.geo"
    opt_msh    = f"{flex_run_dir}/{cfg.subject_id}_tes_flex_opt_head_mesh.msh"

    if not force and os.path.isfile(msh_out):
        print(f"  [SKIP] visualization mesh — {msh_out} exists")
        _write_geo_loader(geo_out, stem, cfg)
        return

    if not os.path.isfile(opt_msh):
        print(f"  [SKIP] visualization — opt mesh not found: {opt_msh}")
        return

    m = mesh_io.read_msh(opt_msh)

    # ── Tissue element geometry ───────────────────────────────────────────────
    tissue_mask = np.isin(m.elm.tag1, TISSUE_TAGS)
    n_elm       = m.elm.nr

    m_tissue  = m.crop_mesh(tags=TISSUE_TAGS)
    nodes     = m_tissue.nodes.node_coord
    conn      = m_tissue.elm.node_number_list[:, :4] - 1
    centroids = nodes[conn].mean(axis=1)

    # ── Rename TI field cleanly ───────────────────────────────────────────────
    new_elmdata = []
    for d in m.elmdata:
        if d.field_name in ("max_TI", "TI_max", "TI_optimized"):
            new_elmdata.append(mesh_io.ElementData(d.value, "TI amplitude [V/m]"))
        else:
            new_elmdata.append(d)

    # ── ROI element field (1 = inside ROI, 0 = outside) ──────────────────────
    roi_field = np.zeros(n_elm, dtype=np.float32)
    roi_path  = cfg.mask_path(cfg.roi.name)
    if os.path.isfile(roi_path):
        roi_tissue = _mask_nifti_to_elements(roi_path, centroids)
        roi_field[tissue_mask] = roi_tissue.astype(np.float32)
        n_roi = int(roi_tissue.sum())
        print(f"  ROI ({cfg.roi.name}): {n_roi:,} mesh elements")
    else:
        print(f"  WARNING: ROI mask not found — {roi_path}")
    new_elmdata.append(mesh_io.ElementData(roi_field, f"ROI ({cfg.roi.name})"))

    # ── non-ROI element field ─────────────────────────────────────────────────
    if cfg.non_roi:
        nr_field = np.zeros(n_elm, dtype=np.float32)
        nr_path  = cfg.mask_path(cfg.non_roi.name)
        if os.path.isfile(nr_path):
            nr_tissue = _mask_nifti_to_elements(nr_path, centroids)
            nr_field[tissue_mask] = nr_tissue.astype(np.float32)
            n_nr = int(nr_tissue.sum())
            print(f"  non-ROI ({cfg.non_roi.name}): {n_nr:,} mesh elements")
        new_elmdata.append(mesh_io.ElementData(nr_field, f"non-ROI ({cfg.non_roi.name})"))

    m.elmdata = new_elmdata
    mesh_io.write_msh(m, msh_out)
    print(f"  Saved → {msh_out}")

    _write_geo_loader(geo_out, stem, cfg)


def _write_geo_loader(geo_out: str, stem: str, cfg: PipelineConfig) -> None:
    """Write a .geo loader script with Gmsh usage instructions."""
    non_roi_name = cfg.non_roi.name if cfg.non_roi else "none"
    with open(geo_out, "w") as f:
        f.write(f"""\
// TI Optimization — sub-{cfg.subject_id}  |  ROI: {cfg.roi.name}  |  non-ROI: {non_roi_name}
//
// HOW TO USE IN GMSH (free download: https://gmsh.info)
// -------------------------------------------------------
// 1. Open this file in Gmsh: File > Open  (or drag onto Gmsh window)
// 2. TISSUE GEOMETRY — show/hide WM / GM / CSF / Skull / Scalp:
//      Tools > Visibility > Physical groups
//      Recommended: enable only tag 2 (GM) for a clean view
// 3. TI FIELD & ROI — toggle the colour-coded Views in the left panel:
//      "TI amplitude [V/m]"  – field strength on all tissue elements
//      "ROI ({cfg.roi.name})"  – target region (1 = inside, 0 = outside)
//      "non-ROI ({non_roi_name})"  – avoidance region (if defined)
// 4. Colour scale: double-click a View name to open its Options
//      Set Min/Max, choose colour map, enable "Visible" for 3-D clip
// 5. To see the field only inside GM: in View Options > Visibility,
//      set "Element types" to Tetrahedra only, then use Physical group
//      filter on tag 2 (GM)

Merge "{stem}.msh";

// White background
General.Color.Background = {{255, 255, 255}};
General.BackgroundGradient = 0;
""")
    print(f"  Saved → {geo_out}")


# ═══════════════════════════════════════════════════════════════════════════════
# --set override helper
# ═══════════════════════════════════════════════════════════════════════════════

def apply_set_overrides(cfg: PipelineConfig, overrides: list) -> PipelineConfig:
    """Apply --set KEY=VALUE overrides using dot notation.

    Examples:
        --set subject_id=026
        --set optimizer.cpus=8
        --set optimizer.focality_threshold=[0.2,0.35]
        --set flags.run_visualization=false
    """
    for item in overrides:
        if '=' not in item:
            print(f"  WARNING: ignoring malformed --set argument (no '='): {item!r}")
            continue
        key, _, raw_val = item.partition('=')
        # Parse value: JSON first (handles int/float/bool/list), fallback to string
        try:
            val = json.loads(raw_val)
        except Exception:
            val = raw_val

        parts = key.split('.')
        if len(parts) == 1:
            object.__setattr__(cfg, parts[0], val)
        elif len(parts) == 2:
            sub = getattr(cfg, parts[0], None)
            if sub is None:
                print(f"  WARNING: --set unknown sub-config: {parts[0]!r}")
                continue
            object.__setattr__(sub, parts[1], val)
        else:
            print(f"  WARNING: --set only supports one level of nesting: {key!r}")
    return cfg


# ═══════════════════════════════════════════════════════════════════════════════
# Per-subject pipeline runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_subject(cfg: PipelineConfig, force: set, job_id: str = "") -> None:
    print(f"\nSubject   : {cfg.subject_id}")
    print(f"Project   : {cfg.project_dir}")
    print(f"ROI       : {cfg.roi.name}  {cfg.roi.labels}")
    print(f"non-ROI   : {cfg.non_roi.name if cfg.non_roi else 'None'}")
    print(f"Goal      : {cfg.optimizer.goal} / {cfg.optimizer.postproc}")
    print(f"Simulate  : {cfg.simulation.simulate_mode}")

    # Save a timestamped config copy for reproducibility
    os.makedirs(cfg.sim_sub_dir, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_config(cfg, f"{cfg.sim_sub_dir}/sub-{cfg.subject_id}_desc-config_run-{ts}.json")

    if cfg.flags.run_recon_all:
        run_recon_all(cfg, force="recon_all" in force)

    if cfg.flags.run_charm:
        run_charm(cfg, force="charm" in force)

    if cfg.flags.run_roi_masks:
        create_roi_masks(cfg, force="roi_masks" in force)

    flex_run_dir = None
    if cfg.flags.run_optimization:
        flex_run_dir = run_optimization(cfg, force="optimization" in force,
                                        job_id=job_id)

    # Resolve flex_run_dir even if optimization was skipped
    if flex_run_dir is None:
        pointer_path = f"{cfg.sim_sub_dir}/latest_flex_run.json"
        if os.path.isfile(pointer_path):
            with open(pointer_path) as f:
                flex_run_dir = json.load(f).get("flex_run_dir")
        if flex_run_dir is None:
            print("\n  No flex_run_dir found — skipping simulation, analysis, visualization.")
            return

    if cfg.flags.run_simulation:
        run_simulation(cfg, flex_run_dir, force="simulation" in force)

    if cfg.flags.run_analysis:
        run_analysis(cfg, flex_run_dir, force="analysis" in force)

    if cfg.flags.run_visualization:
        run_visualization(cfg, flex_run_dir, force="visualization" in force)

    print(f"\n{'='*60}")
    print(f"  Pipeline complete — sub-{cfg.subject_id}")
    print(f"  Results: {flex_run_dir}")
    print(f"{'='*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(
        description="TI Pipeline runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Single subject
  python run_pipeline.py --config configs/sub-025_thalamus.json

  # Same config, multiple subjects
  python run_pipeline.py --config configs/base_thalamus.json --subjects 025 026 027

  # Multiple different configs
  python run_pipeline.py --config configs/sub-025_thalamus.json configs/sub-026_hipp.json

  # Override individual parameters at runtime
  python run_pipeline.py --config configs/base.json --subjects 025 026 --set optimizer.cpus=8

  # Force re-run of specific sections
  python run_pipeline.py --config configs/sub-025_thalamus.json --force optimization analysis
""")
    p.add_argument("--config", required=True, nargs="+",
                   metavar="CONFIG",
                   help="One or more JSON config files. "
                        "With --subjects, only the first config is used as a base.")
    p.add_argument("--subjects", nargs="+", default=None,
                   metavar="ID",
                   help="Run the same config for multiple subject IDs "
                        "(overrides subject_id in the config). "
                        "Requires exactly one --config file.")
    p.add_argument("--set", nargs="+", default=[], dest="set_overrides",
                   metavar="KEY=VALUE",
                   help="Override config values using dot notation. "
                        "E.g. --set optimizer.cpus=8 flags.run_visualization=false")
    p.add_argument("--force", nargs="+", default=[],
                   metavar="SECTION",
                   help="Force re-run of section(s) even if output exists. "
                        "Sections: recon_all charm roi_masks optimization "
                        "simulation analysis visualization")
    p.add_argument("--job-id", default="", dest="job_id",
                   help="SLURM job ID for provenance tracking (passed automatically "
                        "from the sbatch script via $SLURM_JOB_ID).")
    return p.parse_args()


def main():
    args  = parse_args()
    force = set(args.force) if args.force else set()

    # Build list of (config_path, subject_id_override) runs
    if args.subjects:
        if len(args.config) != 1:
            abort("--subjects requires exactly one --config file as base.")
        runs = [(args.config[0], sid) for sid in args.subjects]
    else:
        runs = [(cfg_path, None) for cfg_path in args.config]

    n = len(runs)
    for i, (cfg_path, subject_override) in enumerate(runs, 1):
        if n > 1:
            print(f"\n{'#'*60}")
            print(f"  Run {i}/{n}  —  config: {cfg_path}"
                  + (f"  subject: {subject_override}" if subject_override else ""))
            print(f"{'#'*60}")

        cfg = load_config(cfg_path)
        if subject_override is not None:
            object.__setattr__(cfg, "subject_id", subject_override)
        cfg = apply_set_overrides(cfg, args.set_overrides)
        run_subject(cfg, force, job_id=args.job_id)


if __name__ == "__main__":
    main()

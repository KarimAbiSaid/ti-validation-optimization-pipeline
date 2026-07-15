"""
view_roi_masks.py — local ROI mask viewer (no SimNIBS or Apptainer needed)

Displays the ROI and non-ROI NIfTI masks overlaid on the subject's T1 background
in three orthogonal views (axial / coronal / sagittal), centred on the ROI.
Saves a PNG to derivatives/SimNIBS/sub-{id}/roi/.

Usage examples
--------------
# From a config file (recommended — picks up all paths automatically):
python view_roi_masks.py --config path/to/sub-025_roi-striatum_ex_mean.json \
    --project-dir D:/MINDS_Project_Karim/BIDS_TI_Toolbox

# Directly specify subject + ROI:
python view_roi_masks.py --subject 025 --roi striatum \
    --project-dir D:/MINDS_Project_Karim/BIDS_TI_Toolbox

# Override the non-ROI label shown (useful when config is not available):
python view_roi_masks.py --subject 025 --roi striatum --non-roi MotorCortex \
    --project-dir D:/MINDS_Project_Karim/BIDS_TI_Toolbox

# Show more slice columns:
python view_roi_masks.py --subject 025 --roi striatum --n-slices 7 \
    --project-dir D:/MINDS_Project_Karim/BIDS_TI_Toolbox

Dependencies (all pip-installable, no SimNIBS required):
    pip install nibabel numpy matplotlib
"""

from __future__ import annotations
import argparse
import os
import sys

import nibabel as nib
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# ── CLI ────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Visualise ROI / non-ROI NIfTI masks on a T1 background."
    )
    p.add_argument("--config",      help="Path to a pipeline JSON config file")
    p.add_argument("--subject",     help="Subject ID (e.g. 025)")
    p.add_argument("--roi",         help="ROI label name (e.g. striatum)")
    p.add_argument("--non-roi",     help="Non-ROI label name (e.g. MotorCortex)")
    p.add_argument("--project-dir", default=".",
                   help="Root of the BIDS project (default: current directory)")
    p.add_argument("--n-slices", type=int, default=5,
                   help="Number of slice columns (default: 5)")
    p.add_argument("--output", help="Override output PNG path")
    p.add_argument("--show", action="store_true",
                   help="Also open the plot interactively (requires a display)")
    return p.parse_args()


def resolve_paths(args):
    """Return (subject_id, roi_name, non_roi_name, project_dir)."""
    subject_id   = args.subject
    roi_name     = args.roi
    non_roi_name = getattr(args, "non_roi", None)
    project_dir  = os.path.abspath(args.project_dir)

    if args.config:
        import json
        with open(args.config, encoding="utf-8") as f:
            d = json.load(f)
        if subject_id is None:
            subject_id = d.get("subject_id")
        if roi_name is None and "roi" in d:
            roi_name = d["roi"].get("name")
        if non_roi_name is None and d.get("non_roi"):
            non_roi_name = d["non_roi"].get("name")
        # project_dir from config is a container path — keep the CLI override
        if args.project_dir == ".":
            # try to infer from the config's project_dir field only if it looks local
            cfg_dir = d.get("project_dir", "")
            if os.path.isdir(cfg_dir):
                project_dir = cfg_dir

    if not subject_id:
        sys.exit("ERROR: --subject is required (or provide --config)")
    if not roi_name:
        sys.exit("ERROR: --roi is required (or provide --config with a roi.name)")

    return subject_id, roi_name, non_roi_name, project_dir


# ── NIfTI helpers ─────────────────────────────────────────────────────────────

def load_canonical(path: str):
    """Load NIfTI in closest-canonical (RAS) orientation."""
    img = nib.load(path)
    return nib.as_closest_canonical(img)


def mask_bbox(mask_data: np.ndarray, pad: int = 5):
    """Return (slices_xyz) bounding box of non-zero voxels, with padding."""
    idx = np.argwhere(mask_data > 0)
    if len(idx) == 0:
        sh = mask_data.shape
        return tuple(slice(0, s) for s in sh)
    lo = np.maximum(idx.min(axis=0) - pad, 0)
    hi = np.minimum(idx.max(axis=0) + pad + 1, mask_data.shape)
    return tuple(slice(lo[i], hi[i]) for i in range(3))


def sample_slices(axis: int, bbox, n: int):
    """Return n evenly-spaced slice indices along `axis` within bbox."""
    s = bbox[axis]
    lo, hi = s.start, s.stop - 1
    indices = np.linspace(lo, hi, n, dtype=int)
    return indices


# ── Plotting ──────────────────────────────────────────────────────────────────

VIEW_LABELS = {0: "Sagittal (L→R)", 1: "Coronal (P→A)", 2: "Axial (I→S)"}
# axis along which we take slices for each panel row
PANEL_AXES = [2, 1, 0]  # axial first, then coronal, then sagittal


def _get_slice(data, axis, idx):
    """Extract a 2-D slice from 3-D array along the given axis."""
    sl = [slice(None)] * 3
    sl[axis] = idx
    arr = data[tuple(sl)]
    # orient so the image is displayed consistently (transpose if needed)
    if axis == 0:          # sagittal: (y, z) → want AP × IS
        return arr.T[::-1]
    elif axis == 1:        # coronal:  (x, z) → want LR × IS
        return arr.T[::-1]
    else:                  # axial:    (x, y) → want LR × AP
        return arr.T[::-1]


def make_overlay_rgba(roi_slice, non_roi_slice):
    """Build an RGBA overlay image for a single panel."""
    h, w = roi_slice.shape
    rgba = np.zeros((h, w, 4), dtype=np.float32)

    roi_mask  = roi_slice > 0
    nr_mask   = (non_roi_slice > 0) if non_roi_slice is not None else np.zeros_like(roi_mask)
    both_mask = roi_mask & nr_mask

    # non-ROI: blue
    rgba[nr_mask,  0] = 0.2
    rgba[nr_mask,  2] = 0.85
    rgba[nr_mask,  3] = 0.55

    # ROI: red (overwrites blue where both overlap → becomes purple)
    rgba[roi_mask, 0] = 0.95
    rgba[roi_mask, 1] = 0.1
    rgba[roi_mask, 2] = 0.1
    rgba[roi_mask, 3] = 0.6

    # Overlap: vivid magenta
    rgba[both_mask, 0] = 0.95
    rgba[both_mask, 1] = 0.0
    rgba[both_mask, 2] = 0.85
    rgba[both_mask, 3] = 0.75

    return rgba


def plot_masks(t1_img, roi_img, non_roi_img, subject_id, roi_name, non_roi_name,
               n_slices: int = 5, output_path: str = None, show: bool = False):

    t1_data   = np.asarray(t1_img.dataobj, dtype=np.float32)
    roi_data  = np.asarray(roi_img.dataobj) > 0
    nr_data   = (np.asarray(non_roi_img.dataobj) > 0
                 if non_roi_img is not None else None)

    # Normalise T1 to [0, 1] with 99th-percentile clip
    p99 = np.percentile(t1_data[t1_data > 0], 99)
    t1_norm = np.clip(t1_data / p99, 0, 1)

    # Bounding box around ROI (+ non-ROI if available)
    combined = roi_data.copy()
    if nr_data is not None:
        combined = combined | nr_data
    bbox = mask_bbox(combined, pad=8)

    n_rows = 3   # axial / coronal / sagittal
    fig, axes = plt.subplots(
        n_rows, n_slices,
        figsize=(3 * n_slices, 3 * n_rows),
        facecolor="black",
    )
    # Ensure axes is always 2-D
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_slices == 1:
        axes = axes[:, np.newaxis]

    for row, panel_axis in enumerate(PANEL_AXES):
        slice_indices = sample_slices(panel_axis, bbox, n_slices)
        for col, idx in enumerate(slice_indices):
            ax = axes[row, col]
            t1_sl  = _get_slice(t1_norm,  panel_axis, idx)
            roi_sl = _get_slice(roi_data,  panel_axis, idx)
            nr_sl  = _get_slice(nr_data,   panel_axis, idx) if nr_data is not None else None

            ax.imshow(t1_sl,  cmap="gray", vmin=0, vmax=1, aspect="auto", interpolation="nearest")
            overlay = make_overlay_rgba(roi_sl, nr_sl)
            ax.imshow(overlay, aspect="auto", interpolation="nearest")
            ax.axis("off")

            if col == 0:
                ax.set_ylabel(VIEW_LABELS[panel_axis], color="white", fontsize=9,
                              rotation=90, labelpad=4)
            if row == 0:
                ax.set_title(f"slice {idx}", color="white", fontsize=8, pad=2)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=(0.95, 0.1, 0.1, 0.8), label=f"ROI — {roi_name}"),
    ]
    if non_roi_name:
        legend_elements.append(
            mpatches.Patch(facecolor=(0.2, 0.2, 0.85, 0.8), label=f"non-ROI — {non_roi_name}")
        )
        legend_elements.append(
            mpatches.Patch(facecolor=(0.95, 0.0, 0.85, 0.85), label="overlap")
        )

    fig.legend(handles=legend_elements, loc="lower center",
               ncol=len(legend_elements), fontsize=10,
               facecolor="#222222", edgecolor="none", labelcolor="white",
               bbox_to_anchor=(0.5, 0.0))

    title_nr = f" | non-ROI: {non_roi_name}" if non_roi_name else ""
    fig.suptitle(f"sub-{subject_id}  ·  ROI: {roi_name}{title_nr}",
                 color="white", fontsize=13, y=1.01)

    plt.tight_layout(rect=[0, 0.04, 1, 1])

    if output_path:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="black")
        print(f"Saved -> {output_path}")

    if show:
        matplotlib.use("TkAgg")
        plt.show()

    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    subject_id, roi_name, non_roi_name, project_dir = resolve_paths(args)

    sim_sub_dir = os.path.join(project_dir, "derivatives", "SimNIBS",
                               f"sub-{subject_id}")
    m2m_path    = os.path.join(sim_sub_dir, f"m2m_{subject_id}")
    roi_dir     = os.path.join(sim_sub_dir, "roi")

    # ── Background: prefer bias-corrected T1, fall back to raw T1w ───────────
    t1_candidates = [
        os.path.join(m2m_path, "segmentation", "T1_bias_corrected.nii.gz"),
        os.path.join(m2m_path, f"T1.nii.gz"),
        os.path.join(project_dir, "rawdata", f"sub-{subject_id}",
                     "anat", f"sub-{subject_id}_T1w.nii.gz"),
    ]
    t1_path = next((p for p in t1_candidates if os.path.isfile(p)), None)
    if t1_path is None:
        sys.exit(
            f"ERROR: Could not find a T1 background for sub-{subject_id}.\n"
            f"Tried:\n" + "\n".join(f"  {p}" for p in t1_candidates)
        )
    print(f"Background T1: {t1_path}")

    # ── ROI mask ──────────────────────────────────────────────────────────────
    roi_path = os.path.join(roi_dir, f"sub-{subject_id}_label-{roi_name}_mask.nii.gz")
    if not os.path.isfile(roi_path):
        sys.exit(
            f"ERROR: ROI mask not found:\n  {roi_path}\n"
            "Run Section 1 of the pipeline first, or check the label name."
        )
    print(f"ROI mask:       {roi_path}")

    # ── non-ROI mask (optional) ───────────────────────────────────────────────
    nr_path = None
    if non_roi_name:
        nr_path = os.path.join(roi_dir,
                               f"sub-{subject_id}_label-{non_roi_name}_mask.nii.gz")
        if not os.path.isfile(nr_path):
            print(f"WARNING: non-ROI mask not found — {nr_path}\n"
                  "         Non-ROI overlay will be skipped.")
            nr_path = None
        else:
            print(f"non-ROI mask:   {nr_path}")

    # ── Load images, resample masks to T1 space if needed ─────────────────────
    t1_img  = load_canonical(t1_path)
    roi_img = load_canonical(roi_path)

    # Resample mask to T1 grid if shapes differ (common when T1 is native + mask is in conform space)
    t1_shape = t1_img.shape
    if roi_img.shape != t1_shape:
        print(f"  Resampling ROI mask {roi_img.shape} → T1 space {t1_shape}")
        roi_img = _resample_nearest(roi_img, t1_img)

    nr_img = None
    if nr_path:
        nr_img = load_canonical(nr_path)
        if nr_img.shape != t1_shape:
            print(f"  Resampling non-ROI mask {nr_img.shape} → T1 space {t1_shape}")
            nr_img = _resample_nearest(nr_img, t1_img)

    # ── Output path ───────────────────────────────────────────────────────────
    if args.output:
        out_png = args.output
    else:
        nr_suffix = f"_nr-{non_roi_name}" if non_roi_name else ""
        out_png = os.path.join(roi_dir,
                               f"sub-{subject_id}_roi-{roi_name}{nr_suffix}_maskview.png")

    # ── Plot ──────────────────────────────────────────────────────────────────
    plot_masks(
        t1_img, roi_img, nr_img,
        subject_id=subject_id,
        roi_name=roi_name,
        non_roi_name=non_roi_name,
        n_slices=args.n_slices,
        output_path=out_png,
        show=args.show,
    )


def _resample_nearest(src_img, ref_img):
    """Resample src_img to ref_img grid using nearest-neighbour (pure numpy)."""
    ref_aff_inv = np.linalg.inv(ref_img.affine)
    src_aff     = src_img.affine
    src_data    = np.asarray(src_img.dataobj)

    # Build ref voxel grid
    sh = ref_img.shape
    i, j, k = np.meshgrid(np.arange(sh[0]),
                           np.arange(sh[1]),
                           np.arange(sh[2]),
                           indexing="ij")
    ones = np.ones(sh)
    ref_vox_world = (ref_img.affine @
                     np.stack([i, j, k, ones], axis=-1)[..., np.newaxis]).squeeze(-1)[..., :3]

    # Map world coords to src voxel indices
    ones_flat = np.ones((*sh, 1))
    src_vox = (np.linalg.inv(src_aff) @
               np.concatenate([ref_vox_world, ones_flat], axis=-1)[..., np.newaxis]).squeeze(-1)[..., :3]
    src_idx = np.round(src_vox).astype(int)

    # Bounds check
    ss = src_data.shape
    in_b = ((src_idx[..., 0] >= 0) & (src_idx[..., 0] < ss[0]) &
            (src_idx[..., 1] >= 0) & (src_idx[..., 1] < ss[1]) &
            (src_idx[..., 2] >= 0) & (src_idx[..., 2] < ss[2]))
    out = np.zeros(sh, dtype=src_data.dtype)
    i2, j2, k2 = src_idx[..., 0], src_idx[..., 1], src_idx[..., 2]
    i2c = np.clip(i2, 0, ss[0]-1)
    j2c = np.clip(j2, 0, ss[1]-1)
    k2c = np.clip(k2, 0, ss[2]-1)
    out[in_b] = src_data[i2c[in_b], j2c[in_b], k2c[in_b]]

    return nib.Nifti1Image(out, ref_img.affine, ref_img.header)


if __name__ == "__main__":
    main()

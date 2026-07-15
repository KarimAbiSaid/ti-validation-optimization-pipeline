"""
discovery.py — mask-generation discovery and path-validation logic for the GUI.

Pure Python, no Dash import — testable and reusable independently of the web
layer. Talks to create_masks.py (code/pipeline/) for LUT lookups only; never
duplicates its warping/masking logic. Phase-agnostic bits (subject scanning,
project paths) live in common.py.
"""
from __future__ import annotations

import json
import os
import re

from common import PROJECT_DIR, get_m2m_path, discover_subjects, nifti_shape  # noqa: F401 (re-exported)

# Matches all three filename patterns create_masks.py's create_roi/
# create_non_roi/create_general_mask produce:
#   sub-{id}_label-{name}_mask.nii.gz    (ROI)
#   sub-{id}_nonroi-{name}_mask.nii.gz   (non-ROI)
#   sub-{id}_mask-{name}.nii.gz          (general — "mask-" is a prefix here,
#                                          not a "_mask.nii.gz" suffix)
# Deliberately excludes other roi/-folder files like the cached
# *_BNA_atlas_subjectspace.nii.gz warped atlas (not a mask).
_MASK_FILENAME_RE = re.compile(r"^sub-.+_(label-.+_mask|nonroi-.+_mask|mask-.+)\.nii\.gz$")


# ═════════════════════════════════════════════════════════════════════════════
# Atlas source registry — static metadata about each LabelSource in
# create_masks.py: default paths on this machine, usability, notes shown in
# the UI.
# ═════════════════════════════════════════════════════════════════════════════

ATLAS_REGISTRY = {
    "SimNIBS": {
        "usable": True,
        "note": "charm's own segmentation — subject-native, no extra atlas paths beyond m2m.",
        "has_lut": True,
    },
    "BNA": {
        "usable": True,
        "note": "Brainnetome atlas, MNI FSL152 space. No name lut in this project — numeric ids only.",
        "has_lut": False,
        "extra_paths": {
            "atlas_path": os.path.join(PROJECT_DIR, "BN_Atlas_246_1mm.nii.gz"),
        },
    },
    "Allen": {
        "usable": True,
        "note": "Allen Brain Atlas (Ding 2020), ICBM 2009b Nonlinear Symmetric space.",
        "has_lut": True,
        "extra_paths": {
            "atlas_path": os.path.join(PROJECT_DIR, "Allen_atlas", "annotation_full.nii.gz"),
            "roi_list_path": os.path.join(PROJECT_DIR, "Allen_atlas", "ROI_list_allen_atlas.csv"),
        },
    },
    "FreeSurfer": {
        "usable": False,
        "note": ("Not yet usable (per create_masks.py): needs recon-all output, and the "
                 "subject-native transform hasn't been verified for every subject."),
        "has_lut": True,
        "extra_paths": {},  # aseg_path is per-subject, resolved in atlas_check_paths()
    },
}


# ═════════════════════════════════════════════════════════════════════════════
# Path validation
# ═════════════════════════════════════════════════════════════════════════════

def subject_check_paths(subject_id: str, project_dir: str = PROJECT_DIR) -> dict:
    """Subject-level paths MaskGeneratorEngine needs regardless of atlas source."""
    m2m_path = get_m2m_path(subject_id, project_dir)
    return {
        "m2m folder": m2m_path,
        "labeling.nii.gz (Conform grid + SimNIBS labels)":
            os.path.join(m2m_path, "segmentation", "labeling.nii.gz"),
        "Conform2MNI_nonl.nii.gz (MNI warp field)":
            os.path.join(m2m_path, "toMNI", "Conform2MNI_nonl.nii.gz"),
    }


def atlas_check_paths(atlas_name: str, subject_id: str | None = None,
                       project_dir: str = PROJECT_DIR) -> dict:
    """Atlas-specific extra paths, beyond the subject-level ones above."""
    entry = ATLAS_REGISTRY[atlas_name]
    paths = dict(entry.get("extra_paths", {}))
    if atlas_name == "FreeSurfer" and subject_id:
        paths["aseg_path"] = os.path.join(
            project_dir, "derivatives", "freesurfer", f"sub-{subject_id}", "mri", "aparc+aseg.mgz")
    return paths


def validate_paths(paths: dict) -> list[dict]:
    """paths: {label: path_or_None}. Returns [{"label", "path", "exists"}]."""
    out = []
    for label, path in paths.items():
        out.append({
            "label": label,
            "path": path,
            "exists": bool(path) and os.path.exists(path),
        })
    return out


def full_atlas_check(atlas_name: str, subject_id: str, project_dir: str = PROJECT_DIR) -> dict:
    """All path checks for one (atlas, subject) pair — subject-level (m2m,
    labeling.nii.gz, MNI warp field) + atlas-level (atlas file, roi list,
    aseg) combined. Returns {"checks": [...], "ready": bool}."""
    paths = subject_check_paths(subject_id, project_dir)
    paths.update(atlas_check_paths(atlas_name, subject_id, project_dir))
    checks = validate_paths(paths)
    ready = ATLAS_REGISTRY[atlas_name]["usable"] and all(c["exists"] for c in checks)
    return {"checks": checks, "ready": ready}


def atlas_availability_matrix(subject_ids: list[str], project_dir: str = PROJECT_DIR) -> dict:
    """{subject_id: {atlas_name: {"ready": bool, "missing": [labels]}}} across
    every subject/atlas combination — the overview flag for 'what's usable
    right now vs. what's missing (MNI warp field, subject labeling, ...)'."""
    out = {}
    for sid in subject_ids:
        row = {}
        for atlas_name, meta in ATLAS_REGISTRY.items():
            result = full_atlas_check(atlas_name, sid, project_dir)
            missing = [c["label"] for c in result["checks"] if not c["exists"]]
            if not meta["usable"]:
                missing.insert(0, "not yet usable (see create_masks.py)")
            row[atlas_name] = {"ready": result["ready"], "missing": missing}
        out[sid] = row
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Region lut lookup (for the searchable multi-select)
# ═════════════════════════════════════════════════════════════════════════════

def build_lut(atlas_name: str, subject_id: str | None = None,
              project_dir: str = PROJECT_DIR) -> dict | None:
    """Return {id: name} for the region picker, or None if this atlas has no
    lut (BNA) or required files are missing. Caller should validate paths
    first — this will raise if a required file genuinely can't be read."""
    entry = ATLAS_REGISTRY[atlas_name]
    if not entry["has_lut"]:
        return None

    from create_masks import SimNIBSAtlas, AllenAtlas, FreeSurferAtlas

    if atlas_name == "SimNIBS":
        return SimNIBSAtlas(get_m2m_path(subject_id, project_dir)).lut()

    if atlas_name == "Allen":
        p = atlas_check_paths("Allen", project_dir=project_dir)
        return AllenAtlas(p["atlas_path"], p["roi_list_path"]).lut()

    if atlas_name == "FreeSurfer":
        p = atlas_check_paths("FreeSurfer", subject_id, project_dir)
        return FreeSurferAtlas(p["aseg_path"]).lut()

    return None


# ═════════════════════════════════════════════════════════════════════════════
# Existing masks browser
# ═════════════════════════════════════════════════════════════════════════════

def expected_mask_path(subject_id: str, mask_type: str, name: str, project_dir: str = PROJECT_DIR) -> str:
    """Mirrors the out_path construction in create_masks.py's
    create_roi/create_non_roi/create_general_mask — keep in sync if those change.
    `name` here is whatever the caller passes to create_roi/etc (the GUI page
    is responsible for baking the atlas name into it to avoid collisions,
    e.g. BNA vs Allen both producing a "hippocampus" mask)."""
    roi_dir = os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}", "roi")
    if mask_type == "ROI":
        fname = f"sub-{subject_id}_label-{name}_mask.nii.gz"
    elif mask_type == "non-ROI":
        fname = f"sub-{subject_id}_nonroi-{name}_mask.nii.gz"
    elif mask_type == "general":
        fname = f"sub-{subject_id}_mask-{name}.nii.gz"
    else:
        raise ValueError(f"Unknown mask type: {mask_type}")
    return os.path.join(roi_dir, fname)


def _build_source(atlas_name: str, subject_id: str, project_dir: str = PROJECT_DIR):
    from create_masks import SimNIBSAtlas, BNAAtlas, AllenAtlas, FreeSurferAtlas

    if atlas_name == "SimNIBS":
        return SimNIBSAtlas(get_m2m_path(subject_id, project_dir))
    if atlas_name == "BNA":
        p = atlas_check_paths("BNA", subject_id, project_dir)
        return BNAAtlas(p["atlas_path"])
    if atlas_name == "Allen":
        p = atlas_check_paths("Allen", subject_id, project_dir)
        return AllenAtlas(p["atlas_path"], p["roi_list_path"])
    if atlas_name == "FreeSurfer":
        p = atlas_check_paths("FreeSurfer", subject_id, project_dir)
        return FreeSurferAtlas(p["aseg_path"])
    raise ValueError(f"Unknown atlas: {atlas_name}")


def generate_masks(subject_ids: list[str], atlas_name: str, label_ids: dict, mask_type: str,
                    name: str, hemisphere: str | None = None,
                    project_dir: str = PROJECT_DIR) -> list[dict]:
    """Run MaskGeneratorEngine.create_roi/create_non_roi/create_general_mask for
    each subject independently. `name` is the final (already atlas-suffixed,
    if the caller wants that) mask identifier passed straight through to
    create_masks.py. Returns [{"subject_id", "success", "voxel_count",
    "out_path", "error"}] — one subject's failure doesn't stop the rest."""
    from create_masks import MaskGeneratorEngine

    results = []
    for sid in subject_ids:
        try:
            engine = MaskGeneratorEngine(subject=sid, project_dir=project_dir)
            source = _build_source(atlas_name, sid, project_dir)

            if mask_type == "ROI":
                mask = engine.create_roi(source, label_ids, name, hemisphere=hemisphere)
            elif mask_type == "non-ROI":
                mask = engine.create_non_roi(source, label_ids, name, hemisphere=hemisphere)
            elif mask_type == "general":
                mask = engine.create_general_mask(source, label_ids, name, hemisphere=hemisphere)
            else:
                raise ValueError(f"Unknown mask type: {mask_type}")

            results.append({
                "subject_id": sid, "success": True,
                "voxel_count": int(mask.sum()),
                "out_path": expected_mask_path(sid, mask_type, name, project_dir),
                "error": None,
            })
        except Exception as e:
            results.append({"subject_id": sid, "success": False, "voxel_count": None,
                             "out_path": None, "error": str(e)})
    return results


def _same_grid(img_a, img_b) -> bool:
    import numpy as np
    return img_a.shape == img_b.shape and np.allclose(img_a.affine, img_b.affine, atol=1e-3)


def mask_centroid_in_t1_voxels(subject_id: str, mask_path: str, project_dir: str = PROJECT_DIR) -> tuple[int, int, int]:
    """A mask voxel to default the slice preview to, expressed in T1.nii.gz's
    voxel grid. Uses the actual mask voxel closest to the centroid (a
    "medoid"), not the raw mean position — for bilateral masks (e.g. BNA/
    FreeSurfer Left+Right pairs, common in this pipeline) the mean often
    lands in the empty space between hemispheres, which isn't part of the
    mask at all. Older masks on a different grid than T1 (e.g. legacy
    FreeSurfer-aseg ones, on their own 256^3 conformed grid) get converted
    through the affine transform so the default slice still lands correctly."""
    import nibabel as nib
    import numpy as np

    mask_img = nib.load(mask_path)
    data = np.asarray(mask_img.dataobj) > 0
    t1_img = nib.load(os.path.join(get_m2m_path(subject_id, project_dir), "T1.nii.gz"))

    if not data.any():
        sh = t1_img.shape
        return sh[0] // 2, sh[1] // 2, sh[2] // 2

    true_voxels = np.argwhere(data)
    mean_vox = true_voxels.mean(axis=0)
    nearest = true_voxels[np.argmin(np.sum((true_voxels - mean_vox) ** 2, axis=1))]
    medoid_vox = nearest.astype(np.float64)

    if _same_grid(mask_img, t1_img):
        return tuple(int(round(c)) for c in medoid_vox)

    world = mask_img.affine @ np.append(medoid_vox, 1.0)
    t1_vox = np.linalg.inv(t1_img.affine) @ world
    return tuple(int(round(c)) for c in t1_vox[:3])


def mask_needs_resampling(subject_id: str, mask_path: str, project_dir: str = PROJECT_DIR) -> bool:
    """True if mask_path isn't on T1.nii.gz's grid, i.e. load_slice_overlay
    will resample it on the fly to build the preview."""
    import nibabel as nib
    mask_img = nib.load(mask_path)
    t1_img = nib.load(os.path.join(get_m2m_path(subject_id, project_dir), "T1.nii.gz"))
    return not _same_grid(mask_img, t1_img)


def _t1_view_geometry(subject_id: str, project_dir: str = PROJECT_DIR) -> dict:
    """Map each standard view name to how it's built from T1.nii.gz's own
    array axes, derived from the image's actual axis codes (e.g.
    ('P','S','R')) rather than assumed — charm's Conform space is NOT
    necessarily stored in (0=sagittal,1=coronal,2=axial) order.

    Each view: fixed_axis (the one held constant), h_axis/v_axis (the two
    in-plane axes, assigned to screen horizontal/vertical), flip_h/flip_v
    (whether that axis's storage direction needs reversing to reach the
    usual radiological display convention: superior up; anterior up for
    axial; patient's right on the image's left for coronal/axial; anterior
    on the image's right for sagittal)."""
    import nibabel as nib

    t1_img = nib.load(os.path.join(get_m2m_path(subject_id, project_dir), "T1.nii.gz"))
    codes = nib.aff2axcodes(t1_img.affine)  # codes[i] = anatomical direction of +index on axis i

    def find(letters):
        for ax, c in enumerate(codes):
            if c in letters:
                return ax, c
        raise ValueError(f"no axis in {codes} matches {letters}")

    lr_axis, lr_code = find("LR")
    ap_axis, ap_code = find("AP")
    si_axis, si_code = find("SI")

    return {
        "sagittal": dict(fixed_axis=lr_axis, h_axis=ap_axis, v_axis=si_axis,
                          flip_h=(ap_code == "P"), flip_v=(si_code == "S")),
        "coronal":  dict(fixed_axis=ap_axis, h_axis=lr_axis, v_axis=si_axis,
                          flip_h=(lr_code == "R"), flip_v=(si_code == "S")),
        "axial":    dict(fixed_axis=si_axis, h_axis=lr_axis, v_axis=ap_axis,
                          flip_h=(lr_code == "R"), flip_v=(ap_code == "P")),
    }


def _raw_vh_slice(arr_or_dataobj, geo: dict, index: int):
    """One slice at geo['fixed_axis']=index, reshaped to (v_axis, h_axis)
    order — no flips applied yet (see _apply_flips)."""
    import numpy as np

    slicer = [slice(None)] * 3
    slicer[geo["fixed_axis"]] = index
    sub = np.asarray(arr_or_dataobj[tuple(slicer)])
    remaining = [i for i in range(3) if i != geo["fixed_axis"]]
    return sub if remaining[0] == geo["v_axis"] else sub.T


def _apply_flips(disp, geo: dict):
    if geo["flip_v"]:
        disp = disp[::-1, :]
    if geo["flip_h"]:
        disp = disp[:, ::-1]
    return disp


def default_view_indices(subject_id: str, mask_path: str, project_dir: str = PROJECT_DIR) -> dict:
    """{"sagittal": idx, "coronal": idx, "axial": idx} — default slice index
    per view (T1-grid axis index, correctly assigned per view), landing on
    real mask tissue (see mask_centroid_in_t1_voxels's medoid logic)."""
    i0, i1, i2 = mask_centroid_in_t1_voxels(subject_id, mask_path, project_dir)
    raw = {0: i0, 1: i1, 2: i2}
    geo = _t1_view_geometry(subject_id, project_dir)
    return {view: raw[g["fixed_axis"]] for view, g in geo.items()}


def view_max_indices(subject_id: str, project_dir: str = PROJECT_DIR) -> dict:
    """{"sagittal": max_idx, "coronal": max_idx, "axial": max_idx} for slider bounds."""
    import nibabel as nib

    t1_img = nib.load(os.path.join(get_m2m_path(subject_id, project_dir), "T1.nii.gz"))
    geo = _t1_view_geometry(subject_id, project_dir)
    return {view: t1_img.shape[g["fixed_axis"]] - 1 for view, g in geo.items()}


def view_voxel_spacing(subject_id: str, project_dir: str = PROJECT_DIR) -> dict:
    """{"sagittal": (h_mm, v_mm), "coronal": (...), "axial": (...)} — physical
    voxel size in mm along each view's horizontal/vertical axis. T1.nii.gz's
    Conform grid isn't isotropic (e.g. 256x256x150), so a plain index-count
    display would stretch/squeeze the image — use real mm spacing instead."""
    import nibabel as nib
    import numpy as np

    t1_img = nib.load(os.path.join(get_m2m_path(subject_id, project_dir), "T1.nii.gz"))
    spacing = np.linalg.norm(t1_img.affine[:3, :3], axis=0)  # mm/voxel per array axis
    geo = _t1_view_geometry(subject_id, project_dir)
    return {view: (float(spacing[g["h_axis"]]), float(spacing[g["v_axis"]])) for view, g in geo.items()}


def load_slice_overlay(subject_id: str, mask_path: str, view: str, index: int,
                        project_dir: str = PROJECT_DIR):
    """One 2D slice, correctly oriented for display (radiological convention —
    see _t1_view_geometry): T1 background + binary mask. view:
    'sagittal'|'coronal'|'axial'.

    Fast path: mask already shares T1's grid (true for anything from
    MaskGeneratorEngine) — read both lazily via dataobj indexing, no
    full-volume load.
    Slow path: mask is on a different grid (e.g. legacy FreeSurfer-aseg
    masks) — load the mask once and nearest-neighbour resample just this
    slice's worth of T1-grid coordinates into the mask's own voxel space,
    same technique as MaskGeneratorEngine._resample_direct."""
    import nibabel as nib
    import numpy as np

    geo = _t1_view_geometry(subject_id, project_dir)[view]

    t1_path = os.path.join(get_m2m_path(subject_id, project_dir), "T1.nii.gz")
    t1_img = nib.load(t1_path)
    mask_img = nib.load(mask_path)

    t1_slice = _apply_flips(_raw_vh_slice(t1_img.dataobj, geo, index), geo).astype(np.float32)

    if _same_grid(mask_img, t1_img):
        mask_slice = _apply_flips(_raw_vh_slice(mask_img.dataobj, geo, index), geo) > 0
    else:
        from scipy.ndimage import map_coordinates

        v_dim = t1_img.shape[geo["v_axis"]]
        h_dim = t1_img.shape[geo["h_axis"]]
        v_idx, h_idx = np.indices((v_dim, h_dim), dtype=np.float64)

        full_idx = np.empty((3, v_dim, h_dim), dtype=np.float64)
        full_idx[geo["v_axis"]] = v_idx
        full_idx[geo["h_axis"]] = h_idx
        full_idx[geo["fixed_axis"]] = index

        ones = np.ones((1, v_dim, h_dim))
        vox_h = np.concatenate([full_idx, ones], axis=0).reshape(4, -1)
        world = t1_img.affine @ vox_h
        mask_vox = (np.linalg.inv(mask_img.affine) @ world)[:3].reshape(3, v_dim, h_dim)

        mask_data = np.asarray(mask_img.dataobj, dtype=np.float32)
        sampled = map_coordinates(mask_data, [mask_vox[0], mask_vox[1], mask_vox[2]],
                                   order=0, mode="constant", cval=0)
        mask_slice = _apply_flips(sampled > 0, geo)

    return t1_slice, mask_slice


def export_mask_mesh(subject_id: str, mask_path: str, project_dir: str = PROJECT_DIR) -> str:
    """Wraps MaskGeneratorEngine.export_visualization_mesh for a single
    on-disk mask, so it can be opened in Gmsh externally."""
    from create_masks import MaskGeneratorEngine
    import nibabel as nib
    import numpy as np

    engine = MaskGeneratorEngine(subject=subject_id, project_dir=project_dir)
    mask = (np.asarray(nib.load(mask_path).dataobj) > 0).astype(np.uint8)
    mask_name = os.path.basename(mask_path)
    for suffix in (".nii.gz", ".nii"):
        if mask_name.endswith(suffix):
            mask_name = mask_name[: -len(suffix)]
            break
    out_dir = os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}", "roi", "viz")
    out_path = os.path.join(out_dir, f"{mask_name}_viz.msh")
    return engine.export_visualization_mesh({mask_name: mask}, out_path)


def existing_masks(subject_id: str, project_dir: str = PROJECT_DIR) -> list[dict]:
    """List roi/*_mask.nii.gz for a subject, with sidecar .json metadata if
    present (older masks have hand-written sidecars; masks generated via
    MaskGeneratorEngine currently don't write one)."""
    roi_dir = os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}", "roi")
    if not os.path.isdir(roi_dir):
        return []

    out = []
    for fname in sorted(os.listdir(roi_dir)):
        if not _MASK_FILENAME_RE.match(fname):
            continue
        path = os.path.join(roi_dir, fname)
        sidecar_path = path[: -len(".nii.gz")] + ".json"
        sidecar = None
        if os.path.isfile(sidecar_path):
            try:
                with open(sidecar_path) as f:
                    sidecar = json.load(f)
            except (OSError, json.JSONDecodeError):
                sidecar = None
        out.append({
            "filename": fname,
            "path": path,
            "mtime": os.path.getmtime(path),
            "size_bytes": os.path.getsize(path),
            "sidecar": sidecar,
        })
    return out

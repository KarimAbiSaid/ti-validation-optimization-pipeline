"""
create_masks.py — MaskGeneratorEngine: unified ROI/non-ROI/general mask
generation across multiple anatomical labeling sources.

Step 0 (this file, so far): source compatibility only — loading + warping each
label source into a given subject's own space. Mask extraction, ROI/non-ROI
naming, and visualization mesh export are not implemented yet (added in later
steps).

Sources
-------
SimNIBSAtlas   charm's own detailed segmentation (segmentation/labeling.nii.gz,
               FreeSurfer-style label IDs) — already subject-native, no warp.
BNAAtlas       Brainnetome atlas (BN_Atlas_246_1mm.nii.gz) — FSL MNI152 space,
               same template charm's own toMNI transform targets.
AllenAtlas     Allen Brain Atlas / Ding 2020 (annotation_full.nii.gz) — ICBM
               2009b Nonlinear Symmetric space (different template than
               SimNIBS's MNI152; accepted ~1mm affine-level mismatch, see
               project notes).
FreeSurferAtlas  aparc+aseg from recon-all — NOT YET USABLE. Defined as a
               placeholder; needs recon-all to have been run for the subject,
               and the subject-native transform hasn't been verified (may not
               coincide with charm's own "Conform" space).

Both BNAAtlas and AllenAtlas share the exact same warp mechanism (nearest-
neighbour lookup through charm's Conform2MNI_nonl.nii.gz field) — that logic
lives once in MaskGeneratorEngine._warp_to_subject, not duplicated per source.
"""

from __future__ import annotations
import os
import numpy as np
import nibabel as nib
from scipy.ndimage import map_coordinates


# ═════════════════════════════════════════════════════════════════════════════
# Label sources
# ═════════════════════════════════════════════════════════════════════════════

class LabelSourceBase:
    """One anatomical labeling source: where its label volume lives, and what
    space it's defined in."""
    space: str = None   # "subject" | "mni_fsl152" | "mni_icbm2009b"

    def _load_labels(self):
        """Return (label_array, affine) in the source's own native space."""
        raise NotImplementedError

    def lut(self):
        """Return {id: name} mapping, or None if not available."""
        return None

    def cache_key(self):
        """Stable identity for MaskGeneratorEngine's cache — must NOT be
        id(self): source instances are often created inline/temporarily
        (e.g. AllenAtlas() per call) and get garbage-collected, and CPython
        reuses freed object addresses, so id()-based keys can silently
        collide with an unrelated later source and return the wrong cached
        array. Default: class name + relevant file path attribute."""
        path = getattr(self, "atlas_path", None) or getattr(self, "m2m_path", None) \
            or getattr(self, "aseg_path", None)
        return (type(self).__name__, path)


class SimNIBSAtlas(LabelSourceBase):
    """charm's detailed FreeSurfer-style segmentation. Already subject-native."""
    space = "subject"

    def __init__(self, m2m_path):
        self.m2m_path = m2m_path

    def _load_labels(self):
        path = os.path.join(self.m2m_path, "segmentation", "labeling.nii.gz")
        img = nib.load(path)
        return np.asarray(img.dataobj), img.affine

    def lut(self):
        """{id: name} from SimNIBS's bundled FreeSurferColorLUT — used for
        name-prefix hemisphere lookup (see MaskGeneratorEngine.get_hemisphere_mask)."""
        return _load_freesurfer_lut()


class BNAAtlas(LabelSourceBase):
    """Brainnetome atlas, FSL MNI152 space."""
    space = "mni_fsl152"

    def __init__(self, atlas_path):
        # No hardcoded default: SCITAS runs inside the Apptainer container
        # (Linux, /mnt/BIDS_TI_Toolbox/...), not this Windows machine — the
        # caller must always pass the correct path for its own environment.
        self.atlas_path = atlas_path

    def _load_labels(self):
        img = nib.load(self.atlas_path)
        return np.asarray(img.dataobj), img.affine

    def lut(self):
        """No BNA label-name CSV is present in this project (only the
        .nii.gz volume) — name-based lookup isn't available for BNA yet,
        only numeric ids. Returns None."""
        return None


class AllenAtlas(LabelSourceBase):
    """Allen Brain Atlas (Ding 2020), ICBM 2009b Nonlinear Symmetric space."""
    space = "mni_icbm2009b"

    def __init__(self, atlas_path, roi_list_path):
        # No hardcoded default (same reasoning as BNAAtlas): caller must pass
        # paths valid for its own environment (Windows local vs. SCITAS
        # container mount).
        self.atlas_path = atlas_path
        self.roi_list_path = roi_list_path

    def _load_labels(self):
        img = nib.load(self.atlas_path)
        return np.asarray(img.dataobj), img.affine

    def lut(self):
        """{id: acronym} from ROI_list_allen_atlas.csv (e.g. 10330 -> 'TI')."""
        import csv
        out = {}
        with open(self.roi_list_path) as f:
            for row in csv.DictReader(f):
                out[int(row["id"])] = row["acronym"]
        return out


def _load_freesurfer_lut():
    """{id: name} from SimNIBS's bundled FreeSurferColorLUT — shared by
    SimNIBSAtlas and FreeSurferAtlas since both use the same numeric
    convention (charm's labeling.nii.gz reuses FreeSurfer's own ids)."""
    from simnibs.utils.file_finder import Templates
    out = {}
    with open(Templates().labeling_LUT) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 2:
                out[int(parts[0])] = parts[1]
    return out


class FreeSurferAtlas(LabelSourceBase):
    """FreeSurfer recon-all aparc+aseg (or aseg.auto), e.g.
    derivatives/freesurfer/sub-{id}/mri/aparc+aseg.mgz. Subject-native, but
    on FreeSurfer's OWN conformed grid — verified (2025-07) to share the same
    world coordinate frame as charm's Conform space (same-structure centroids
    agree to ~1-2mm for sub-025), just a different voxel grid/affine, so
    MaskGeneratorEngine resamples it directly (no separate registration)."""
    space = "subject"

    def __init__(self, aseg_path):
        self.aseg_path = aseg_path

    def _load_labels(self):
        img = nib.load(self.aseg_path)
        return np.asarray(img.dataobj), img.affine

    def lut(self):
        return _load_freesurfer_lut()


# ═════════════════════════════════════════════════════════════════════════════
# Engine
# ═════════════════════════════════════════════════════════════════════════════

class MaskGeneratorEngine:
    """Warps a LabelSource into one subject's space (cached) as the shared
    foundation for mask extraction (added in a later step)."""

    def __init__(self, subject: str = None, project_dir: str = None,
                 simnibs_dir: str = None, m2m_path: str = None):
        """Either pass (subject, project_dir) — needed for create_roi/
        create_non_roi/create_general_mask, which derive their output paths
        from those — or just m2m_path directly, sufficient for
        get_subject_space_labels/create_mask/get_hemisphere_mask (used by
        run_pipeline.py, which already has m2m_path but not a bare
        project_dir at some call sites)."""
        self.subject = subject
        self.project_dir = project_dir
        self.simnibs_dir = simnibs_dir or (
            os.path.join(project_dir, "derivatives", "SimNIBS") if project_dir else None)
        self.m2m_path = m2m_path or os.path.join(self.simnibs_dir, f"sub-{subject}", f"m2m_{subject}")
        self._cache = {}   # source.cache_key() -> label array in subject space

    def _reference_grid(self):
        """Subject 'Conform' grid: shape/affine/header shared by all outputs."""
        ref_path = os.path.join(self.m2m_path, "segmentation", "labeling.nii.gz")
        ref_img = nib.load(ref_path)
        return ref_img.shape[:3], ref_img.affine, ref_img.header

    def _warp_to_subject(self, label_data: np.ndarray, label_affine: np.ndarray) -> np.ndarray:
        """Nearest-neighbour warp an MNI-space integer label volume into
        subject space, via charm's Conform2MNI_nonl.nii.gz (per-voxel MNI
        world coordinate field for every subject Conform-space voxel)."""
        c2m_path = os.path.join(self.m2m_path, "toMNI", "Conform2MNI_nonl.nii.gz")
        c2m = np.asarray(nib.load(c2m_path).dataobj, dtype=np.float64)

        out_shape, _, _ = self._reference_grid()
        aff_inv = np.linalg.inv(label_affine)

        n = c2m.shape[0] * c2m.shape[1] * c2m.shape[2]
        mni_flat = c2m.reshape(n, 3)
        ones = np.ones((n, 1), dtype=np.float64)
        vox = (aff_inv @ np.hstack([mni_flat, ones]).T).T[:, :3]
        vox = vox.reshape(c2m.shape[0], c2m.shape[1], c2m.shape[2], 3)

        sx, sy, sz = out_shape
        vox = vox[:sx, :sy, :sz, :]
        coords = [vox[..., i] for i in range(3)]
        out = map_coordinates(label_data.astype(np.float32), coords, order=0,
                              mode='constant', cval=0)
        return np.round(out).astype(np.int32)

    def get_subject_space_labels(self, source: LabelSourceBase) -> np.ndarray:
        """Load + (if needed) warp a source's labels into this subject's
        space. Cached per source instance."""
        key = source.cache_key()
        if key not in self._cache:
            data, affine = source._load_labels()
            if source.space == "subject":
                ref_shape, ref_affine, _ = self._reference_grid()
                if data.shape == ref_shape and np.allclose(affine, ref_affine):
                    self._cache[key] = data.astype(np.int32)
                else:
                    # subject-native but different grid (e.g. real FreeSurfer
                    # aseg on FreeSurfer's own conformed space, vs. charm's
                    # Conform grid). Both already describe the same physical
                    # subject (verified: same-structure centroids agree to
                    # ~1-2mm) -- resample directly via world coordinates, no
                    # MNI warp needed.
                    self._cache[key] = self._resample_direct(data, affine, ref_shape, ref_affine)
            else:
                self._cache[key] = self._warp_to_subject(data, affine)
        return self._cache[key]

    def _resample_direct(self, data: np.ndarray, src_affine: np.ndarray,
                          ref_shape, ref_affine: np.ndarray) -> np.ndarray:
        """Nearest-neighbour resample a subject-native volume from its own
        grid onto the reference grid via direct world-coordinate mapping
        (both grids already share the same physical subject frame)."""
        sx, sy, sz = ref_shape
        ii, jj, kk = np.indices((sx, sy, sz))
        ref_vox = np.stack([ii.ravel(), jj.ravel(), kk.ravel(),
                             np.ones(ii.size)], axis=0)
        world = (ref_affine @ ref_vox)[:3]
        src_vox = (np.linalg.inv(src_affine) @ np.vstack([world, np.ones(world.shape[1])]))[:3]
        src_vox = src_vox.reshape(3, sx, sy, sz)
        coords = [src_vox[i] for i in range(3)]
        out = map_coordinates(data.astype(np.float32), coords, order=0,
                              mode='constant', cval=0)
        return np.round(out).astype(np.int32)

    def get_hemisphere_mask(self, side: str) -> np.ndarray:
        """Boolean L/R mask, for sources (like Allen) with no laterality of
        their own. Built once from SimNIBSAtlas's FreeSurfer-style labels
        (name prefix 'Left-'/'Right-'), cached."""
        assert side in ("L", "R")
        cache_key = f"_hemi_{side}"
        if cache_key not in self._cache:
            src = SimNIBSAtlas(self.m2m_path)
            labels = self.get_subject_space_labels(src)
            lut = src.lut()
            prefix = "Left-" if side == "L" else "Right-"
            ids = [i for i, name in lut.items() if name.startswith(prefix)]
            self._cache[cache_key] = np.isin(labels, ids)
        return self._cache[cache_key]

    def _resolve_ids(self, source: LabelSourceBase, label_ids: dict) -> list:
        """label_ids: {display_name: selector}, selector is either an int
        (atlas's own numeric id) or a str (atlas's own region name/acronym,
        resolved via source.lut()).
        - selector not found in the source's lut -> warn, skip.
        - two selectors resolve to the same id -> warn, keep it once.
        (If source.lut() is unavailable, numeric selectors pass through
        unvalidated and str selectors always warn/skip.)"""
        lut = source.lut()
        name_to_id = {v: k for k, v in lut.items()} if lut else {}

        resolved = []
        seen = {}  # id -> display_name that first claimed it
        for display_name, sel in label_ids.items():
            if isinstance(sel, str):
                if sel not in name_to_id:
                    print(f"  WARNING: '{display_name}' -> name '{sel}' not found "
                          f"in {source.__class__.__name__} lut — skipping")
                    continue
                rid = name_to_id[sel]
            else:
                rid = sel
                if lut is not None and rid not in lut:
                    print(f"  WARNING: '{display_name}' -> id {rid} not found "
                          f"in {source.__class__.__name__} lut — skipping")
                    continue

            if rid in seen:
                print(f"  WARNING: '{display_name}' and '{seen[rid]}' both resolve "
                      f"to id {rid} — using once")
                continue
            seen[rid] = display_name
            resolved.append(rid)
        return resolved

    def create_mask(self, source: LabelSourceBase, label_ids: dict, out_path: str,
                     hemisphere: str = None, fallback_label_ids: dict = None) -> np.ndarray:
        """Core op: binary mask of voxels matching any id/name in label_ids
        ({display_name: numeric_id_or_name}), from source, in subject space.
        Saves to out_path. hemisphere: 'L'/'R'/None — optionally intersect
        with a hemisphere mask (for sources like Allen with no laterality of
        their own; None = bilateral / whatever the ids natively select).
        fallback_label_ids: if label_ids resolves to zero voxels, retry with
        this set instead (matches _create_mask's fallback_labels behaviour
        in run_pipeline.py)."""
        labels = self.get_subject_space_labels(source)
        ids = self._resolve_ids(source, label_ids)
        mask = np.isin(labels, ids)
        if hemisphere is not None:
            mask = mask & self.get_hemisphere_mask(hemisphere)

        if not mask.any() and fallback_label_ids:
            print(f"  WARNING: primary labels produced 0 voxels — "
                  f"falling back to: {list(fallback_label_ids.keys())}")
            fb_ids = self._resolve_ids(source, fallback_label_ids)
            mask = np.isin(labels, fb_ids)
            if hemisphere is not None:
                mask = mask & self.get_hemisphere_mask(hemisphere)

        mask = mask.astype(np.uint8)

        _, ref_affine, ref_header = self._reference_grid()
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        nib.save(nib.Nifti1Image(mask, ref_affine, ref_header), out_path)

        vox_vol = abs(np.linalg.det(ref_affine[:3, :3]))
        print(f"  {os.path.basename(out_path)}: {int(mask.sum())} voxels "
              f"({int(mask.sum())*vox_vol:.1f} mm3)")
        return mask

    def create_roi(self, source, label_ids, name, hemisphere=None, fallback_label_ids=None):
        out_path = os.path.join(self.simnibs_dir, f"sub-{self.subject}", "roi",
                                 f"sub-{self.subject}_label-{name}_mask.nii.gz")
        return self.create_mask(source, label_ids, out_path, hemisphere=hemisphere,
                                fallback_label_ids=fallback_label_ids)

    def create_non_roi(self, source, label_ids, name, hemisphere=None, fallback_label_ids=None):
        out_path = os.path.join(self.simnibs_dir, f"sub-{self.subject}", "roi",
                                 f"sub-{self.subject}_nonroi-{name}_mask.nii.gz")
        return self.create_mask(source, label_ids, out_path, hemisphere=hemisphere,
                                fallback_label_ids=fallback_label_ids)

    def create_general_mask(self, source, label_ids, name, hemisphere=None, fallback_label_ids=None):
        out_path = os.path.join(self.simnibs_dir, f"sub-{self.subject}", "roi",
                                 f"sub-{self.subject}_mask-{name}.nii.gz")
        return self.create_mask(source, label_ids, out_path, hemisphere=hemisphere,
                                fallback_label_ids=fallback_label_ids)

    def export_visualization_mesh(self, masks: dict, out_path: str, tags=(1, 2, 3)):
        """masks: {field_name: subject-space mask array (from create_mask/
        create_roi/etc, or get_subject_space_labels)}. Exports one .msh with
        each mask as its own scalar ElementData field on the subject's brain
        mesh — in Gmsh, toggle between fields (e.g. 'Allen_HC_R' vs
        'BNA_HC_R') to compare extents directly."""
        from simnibs import mesh_io

        msh = mesh_io.read_msh(os.path.join(self.m2m_path, f"{self.subject}.msh"))
        msh_crop = msh.crop_mesh(tags=list(tags))
        nodes = msh_crop.nodes.node_coord
        conn = msh_crop.elm.node_number_list[:, :4] - 1
        centroids = nodes[conn].mean(axis=1)

        _, ref_affine, _ = self._reference_grid()
        aff_inv = np.linalg.inv(ref_affine)
        ones = np.ones((len(centroids), 1))
        vox = (aff_inv @ np.hstack([centroids, ones]).T).T[:, :3]
        vox_idx = np.round(vox).astype(int)

        fields = []
        for name, mask in masks.items():
            sh = mask.shape
            in_bounds = ((vox_idx[:, 0] >= 0) & (vox_idx[:, 0] < sh[0]) &
                         (vox_idx[:, 1] >= 0) & (vox_idx[:, 1] < sh[1]) &
                         (vox_idx[:, 2] >= 0) & (vox_idx[:, 2] < sh[2]))
            elm_vals = np.zeros(len(centroids), dtype=np.float64)
            elm_vals[in_bounds] = mask[vox_idx[in_bounds, 0], vox_idx[in_bounds, 1],
                                        vox_idx[in_bounds, 2]]
            fields.append(mesh_io.ElementData(elm_vals, name=name))

        msh_crop.elmdata = fields
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        mesh_io.write_msh(msh_crop, out_path)
        print(f"-> {out_path}  ({len(fields)} fields: {list(masks.keys())})")
        return out_path

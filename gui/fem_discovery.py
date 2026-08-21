"""
fem_discovery.py — Phase 3 (manual FEM / leadfield TI validation) data/logic
layer, no Dash import.

First increment: leadfield-based (algebraic) compute only, reusing
compare_ti_montages.py's load_subject_resources/compute_ti_setup directly —
fast, no long-running-job problem to solve. Full-FEM (one-off
simnibs.run_simnibs per channel, ~1min/channel) and leadfield generation
(~30min, per this project's own pipeline) both need real background-job
infrastructure that doesn't exist yet — planned as follow-up increments.
"""
from __future__ import annotations

import os

from common import PROJECT_DIR, get_m2m_path, discover_subjects  # noqa: F401 (re-exported)


# ═════════════════════════════════════════════════════════════════════════════
# Leadfield discovery
# ═════════════════════════════════════════════════════════════════════════════

def leadfield_dir(subject_id: str, project_dir: str = PROJECT_DIR) -> str:
    return os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}", "leadfield_volume")


def _read_json(path: str) -> dict:
    if not os.path.isfile(path):
        return {}
    import json
    with open(path) as f:
        return json.load(f)


def list_leadfields(subject_id: str, project_dir: str = PROJECT_DIR) -> list[dict]:
    """Precomputed leadfields already on disk for this subject, across BOTH
    schemes:
      - new: leadfield_volume/{tag}/{id}_leadfield_{cap}.hdf5 (tag =
        config.leadfield_tag(cap, shape, dimensions, gel_thickness)) — one
        subdirectory per distinct (cap, electrode-settings) combination, the
        same layout run_pipeline.py and generate_leadfield() write.
      - legacy: leadfield_volume/{id}_leadfield_{cap}.hdf5 directly — from
        before per-settings variants existed. Still discoverable so older
        leadfields aren't silently hidden.
    Returns [{"cap_name", "tag" (None for legacy), "hdf5_path", "params",
    "label"}], one entry per variant found."""
    d = leadfield_dir(subject_id, project_dir)
    prefix = f"{subject_id}_leadfield_"
    out = []
    if not os.path.isdir(d):
        return out

    for name in sorted(os.listdir(d)):
        sub = os.path.join(d, name)
        if not os.path.isdir(sub) or name.startswith("_"):
            continue
        for fname in sorted(os.listdir(sub)):
            if not (fname.startswith(prefix) and fname.endswith(".hdf5")):
                continue
            cap_name = fname[len(prefix):-len(".hdf5")]
            hdf5_path = os.path.join(sub, fname)
            params = _read_json(hdf5_path[:-len(".hdf5")] + "_params.json")
            dims = params.get("dimensions") or []
            dims_str = "x".join(f"{v:g}" for v in dims) if dims else "?"
            label = (f"{cap_name}  ({params.get('shape', '?')} {dims_str}mm, "
                    f"{params.get('gel_thickness', '?')}mm gel)")
            out.append({"cap_name": cap_name, "tag": name, "hdf5_path": hdf5_path,
                       "params": params, "label": label})

    for fname in sorted(os.listdir(d)):
        if fname.startswith(prefix) and fname.endswith(".hdf5"):
            cap_name = fname[len(prefix):-len(".hdf5")]
            hdf5_path = os.path.join(d, fname)
            params = _read_json(hdf5_path[:-len(".hdf5")] + "_params.json")
            out.append({"cap_name": cap_name, "tag": None, "hdf5_path": hdf5_path,
                       "params": params, "label": f"{cap_name}  (legacy cache)"})
    return out


def leadfield_status(subject_id: str, cap_name: str, project_dir: str = PROJECT_DIR,
                      shape: str | None = None, dimensions: list | None = None,
                      gel_thickness: float | None = None) -> dict:
    """{"exists", "hdf5_path"}.

    If shape/dimensions/gel_thickness are given, checks for that EXACT
    variant (via config.leadfield_tag) — used when a specific electrode
    setting matters (Run Pipeline / FEM Validation's variant picker).

    Otherwise ("exists" = true if ANY variant, tagged or legacy, exists for
    this cap) — the coarse cap-level readiness check Comparison's
    availability display uses, unchanged from before per-settings variants
    existed; hdf5_path is the first match found."""
    if dimensions is not None:
        from config import leadfield_tag
        tag = leadfield_tag(cap_name, shape or "ellipse", dimensions,
                            gel_thickness if gel_thickness is not None else 1.0)
        hdf5_path = os.path.join(leadfield_dir(subject_id, project_dir), tag,
                                 f"{subject_id}_leadfield_{cap_name}.hdf5")
        return {"exists": os.path.isfile(hdf5_path), "hdf5_path": hdf5_path}

    matches = [lf for lf in list_leadfields(subject_id, project_dir) if lf["cap_name"] == cap_name]
    if matches:
        return {"exists": True, "hdf5_path": matches[0]["hdf5_path"]}
    flat_hdf5 = os.path.join(leadfield_dir(subject_id, project_dir), f"{subject_id}_leadfield_{cap_name}.hdf5")
    return {"exists": False, "hdf5_path": flat_hdf5}


def leadfield_availability_matrix(subject_ids: list[str], cap_name: str, project_dir: str = PROJECT_DIR) -> dict:
    """{subject_id: leadfield_status(...)} — same 'flag what's ready' pattern as phases 1-2."""
    return {sid: leadfield_status(sid, cap_name, project_dir) for sid in subject_ids}


def leadfield_electrode_names(hdf5_path: str) -> list[str]:
    """Electrode names available in this leadfield — read from a small HDF5
    attribute (mesh_leadfield/leadfields/tdcs_leadfield.attrs['electrode_names']),
    NOT by loading the (multi-GB) leadfield array itself via TI.load_leadfield."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        names = f["mesh_leadfield/leadfields/tdcs_leadfield"].attrs["electrode_names"]
    return [str(n) for n in names]


def leadfield_source_cap_path(hdf5_path: str) -> str | None:
    """The exact cap CSV this leadfield was built from (HDF5 attribute
    'electrode_cap') — separate from cap_name, which is just parsed from the
    filename. Useful to cross-check against Phase 2's registered caps."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        return f["mesh_leadfield/leadfields/tdcs_leadfield"].attrs.get("electrode_cap")


def leadfield_electrode_dims(hdf5_path: str) -> list[float] | None:
    """Electrode dims this leadfield was built with (metadata only, for
    prefilling/consistency-checking compute_ti_setup's electrode_dims)."""
    import h5py
    with h5py.File(hdf5_path, "r") as f:
        attrs = f["mesh_leadfield/leadfields/tdcs_leadfield"].attrs
        if "electrode_dims" in attrs:
            return [float(v) for v in attrs["electrode_dims"]]
    params_path = hdf5_path.replace(".hdf5", "_params.json")
    if os.path.isfile(params_path):
        import json
        with open(params_path) as f:
            params = json.load(f)
        if params.get("dimensions"):
            return [float(v) for v in params["dimensions"]]
    return None


# ═════════════════════════════════════════════════════════════════════════════
# Compute (wraps compare_ti_montages.py — pre-validated so its sys.exit()
# error paths, fine for a CLI script, are never actually hit from the GUI)
# ═════════════════════════════════════════════════════════════════════════════

def compute_ti(
    subject_id: str, hdf5_path: str, roi_mask_path: str, non_roi_mask_path: str | None,
    ch1_plus: str, ch1_minus: str, ch1_current_mA: float,
    ch2_plus: str, ch2_minus: str, ch2_current_mA: float,
    electrode_dims: list[float] | None = None, label: str = "setup",
    output_dir: str | None = None, project_dir: str = PROJECT_DIR,
    extra_mask_paths: dict[str, str] | None = None,
) -> dict:
    """Returns {"success", "error"} or, on success, {"success": True, "label",
    "montage", "stats", "msh_path", "npz_path", "json_path"}.

    hdf5_path: resolved leadfield path — use list_leadfields()/
    leadfield_status() to find one for a (subject, cap[, electrode settings])
    combination. Now identical to compute_ti_custom_leadfield() (both take an
    explicit path); kept as a separate name since callers already spell it
    this way, and "custom" no longer means anything different once every
    leadfield lookup resolves to an explicit path first.

    extra_mask_paths: {region_label: mask_path} — additional masks to also
    report TI_mean/TI_max for in "stats" (as "TI_mean_{label}_V_m" /
    "TI_max_{label}_V_m"), alongside the primary ROI/non-ROI. See
    compute_ti_custom_leadfield() for how these are computed without a
    second field solve."""
    return compute_ti_custom_leadfield(
        subject_id=subject_id, hdf5_path=hdf5_path, roi_mask_path=roi_mask_path,
        non_roi_mask_path=non_roi_mask_path,
        ch1_plus=ch1_plus, ch1_minus=ch1_minus, ch1_current_mA=ch1_current_mA,
        ch2_plus=ch2_plus, ch2_minus=ch2_minus, ch2_current_mA=ch2_current_mA,
        electrode_dims=electrode_dims, label=label, output_dir=output_dir, project_dir=project_dir,
        extra_mask_paths=extra_mask_paths,
    )


def _load_subject_resources_from_path(subject_id: str, hdf5_path: str, roi_mask_path: str,
                                       non_roi_mask_path: str | None):
    """Same as compare_ti_montages.load_subject_resources, but takes an
    explicit leadfield HDF5 path instead of deriving one from cap_name — for
    leadfields kept outside the standard leadfield_volume/ location.
    Duplicates that function's body (rather than importing + patching it)
    since it doesn't expose a path-override parameter and it's core pipeline
    code also used on SCITAS — not something to change just for this GUI
    convenience. Returns (resources, error)."""
    import json

    import numpy as np
    from compare_ti_montages import TISSUE_TAGS, SubjectResources, _mask_to_elements
    from simnibs.utils import TI_utils as TI

    if not os.path.isfile(hdf5_path):
        return None, f"leadfield not found: {hdf5_path}"

    leadfield, mesh, idx_lf = TI.load_leadfield(hdf5_path)

    lf_params = {}
    params_path = hdf5_path.replace(".hdf5", "_params.json")
    if os.path.isfile(params_path):
        with open(params_path) as f:
            lf_params = json.load(f)

    tissue_mask = np.isin(mesh.elm.tag1, TISSUE_TAGS)
    m_tissue = mesh.crop_mesh(tags=TISSUE_TAGS)
    tags = m_tissue.elm.tag1
    nodes = m_tissue.nodes.node_coord
    conn = m_tissue.elm.node_number_list[:, :4] - 1
    centroids = nodes[conn].mean(axis=1)
    v0, v1 = nodes[conn[:, 0]], nodes[conn[:, 1]]
    v2, v3 = nodes[conn[:, 2]], nodes[conn[:, 3]]
    elm_volumes = np.abs(np.einsum("ni,ni->n", v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0

    roi_elm_mask = _mask_to_elements(roi_mask_path, centroids)
    non_roi_elm_mask = (_mask_to_elements(non_roi_mask_path, centroids)
                        if non_roi_mask_path and os.path.isfile(non_roi_mask_path)
                        else np.zeros(len(centroids), dtype=bool))

    resources = SubjectResources(
        leadfield=leadfield, mesh=mesh, idx_lf=idx_lf, centroids=centroids,
        tissue_mask=tissue_mask, tags=tags, roi_elm_mask=roi_elm_mask,
        non_roi_elm_mask=non_roi_elm_mask, lf_params=lf_params, elm_volumes=elm_volumes,
    )
    return resources, None


def compute_ti_custom_leadfield(
    subject_id: str, hdf5_path: str, roi_mask_path: str, non_roi_mask_path: str | None,
    ch1_plus: str, ch1_minus: str, ch1_current_mA: float,
    ch2_plus: str, ch2_minus: str, ch2_current_mA: float,
    electrode_dims: list[float] | None = None, label: str = "setup",
    output_dir: str | None = None, project_dir: str = PROJECT_DIR,
    extra_mask_paths: dict[str, str] | None = None,
) -> dict:
    """Same as compute_ti(), but for a leadfield living outside the standard
    leadfield_volume/ location (an explicit hdf5_path rather than a
    subject+cap_name lookup).

    extra_mask_paths: {region_label: mask_path} — stats for these are
    computed from the SAME already-solved field (compute_ti_setup's raw
    "ti" array + the resources already loaded above), not a second field
    solve, so this is cheap to add. A path that's missing/unreadable is
    skipped for that region rather than failing the whole row — the primary
    ROI/non-ROI result is unaffected either way."""
    if not os.path.isfile(hdf5_path):
        return {"success": False, "error": f"leadfield not found: {hdf5_path}"}

    names = set(leadfield_electrode_names(hdf5_path))
    for field_label, val in [("Ch1+", ch1_plus), ("Ch1-", ch1_minus),
                              ("Ch2+", ch2_plus), ("Ch2-", ch2_minus)]:
        if val not in names:
            return {"success": False,
                     "error": f"{field_label} '{val}' not in this leadfield's electrodes: {sorted(names)}"}

    if output_dir is None:
        output_dir = os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}", "comparison")

    from compare_ti_montages import compute_ti_setup

    resources, err = _load_subject_resources_from_path(subject_id, hdf5_path, roi_mask_path, non_roi_mask_path)
    if err:
        return {"success": False, "error": err}

    try:
        result = compute_ti_setup(
            resources,
            ch1_plus=ch1_plus, ch1_minus=ch1_minus, ch1_current_mA=ch1_current_mA,
            ch2_plus=ch2_plus, ch2_minus=ch2_minus, ch2_current_mA=ch2_current_mA,
            electrode_dims=electrode_dims, label=label, output_dir=output_dir,
        )
    except SystemExit as e:
        return {"success": False, "error": str(e)}
    except Exception as e:
        return {"success": False, "error": str(e)}

    stats = dict(result["stats"])
    if extra_mask_paths:
        stats.update(_extra_region_stats(result["ti"], resources.centroids, resources.elm_volumes,
                                         extra_mask_paths))

    return {
        "success": True, "label": result["label"], "montage": result["montage"],
        "stats": stats, "msh_path": result["msh_path"],
        "npz_path": result["npz_path"], "json_path": result["json_path"],
    }


def _extra_region_stats(ti, centroids, elm_volumes, extra_mask_paths: dict[str, str]) -> dict:
    """{"TI_mean_{label}_V_m", "TI_max_{label}_V_m"} for every extra region
    whose mask file exists — same volume-weighted 99th-percentile-capped
    mean as the pipeline's own ROI/non-ROI stats (compare_ti_montages.
    _vol_mean_capped), computed against an already-solved field so no
    second FEM/leadfield solve is needed. A missing/unreadable mask is
    silently skipped for that one region."""
    from compare_ti_montages import _mask_to_elements, _vol_mean_capped

    out = {}
    for region_label, mask_path in extra_mask_paths.items():
        if not mask_path or not os.path.isfile(mask_path):
            continue
        mask = _mask_to_elements(mask_path, centroids)
        if not mask.any():
            continue
        out[f"TI_mean_{region_label}_V_m"] = _vol_mean_capped(ti[mask], elm_volumes[mask])
        out[f"TI_max_{region_label}_V_m"] = float(ti[mask].max())
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Generate + save a full leadfield — mirrors run_pipeline.py's TDCSLEADFIELD
# step exactly (same params, same output path/caching convention). Meant to
# run inside a background job (job_runner.py) — one FEM solve per electrode,
# ~30 minutes typical per this project's own pipeline notes.
# ═════════════════════════════════════════════════════════════════════════════

_LEADFIELD_TISSUES = [1, 2, 3, 4, 5]


def generate_leadfield(
    subject_id: str, registered_cap_path: str,
    shape: str = "ellipse", dimensions: tuple = (14.0, 14.0),
    gel_thickness: float = 1.0, cpus: int = 1, project_dir: str = PROJECT_DIR,
) -> dict:
    """registered_cap_path: the SUBJECT-SPACE cap CSV (e.g. from
    cap_discovery.registered_cap_path() — Phase 2's Register/Adopt output),
    NOT the MNI-space source cap. Returns {"success", "hdf5_path", "error"}."""
    import json

    m2m_path = get_m2m_path(subject_id, project_dir)
    if not os.path.isdir(m2m_path):
        return {"success": False, "hdf5_path": None, "error": f"m2m not found: {m2m_path}"}
    if not os.path.isfile(registered_cap_path):
        return {"success": False, "hdf5_path": None,
                "error": f"registered cap not found: {registered_cap_path}"}

    from config import leadfield_tag

    cap_name = os.path.splitext(os.path.basename(registered_cap_path))[0]
    base_dir = leadfield_dir(subject_id, project_dir)
    tag      = leadfield_tag(cap_name, shape, list(dimensions), gel_thickness)
    lf_dir   = os.path.join(base_dir, tag)
    fname    = f"{subject_id}_leadfield_{cap_name}.hdf5"  # TDCSLEADFIELD's own naming
    hdf5_path = os.path.join(lf_dir, fname)
    params_path = os.path.join(lf_dir, f"{subject_id}_leadfield_{cap_name}_params.json")

    current_params = {
        "shape": shape, "dimensions": list(dimensions), "gel_thickness": gel_thickness,
        "tissues": _LEADFIELD_TISSUES, "interpolation": None,
    }

    # The tag already encodes shape/dimensions/gel_thickness, so a mismatch
    # here would only mean a corrupted/foreign params.json — the params
    # comparison is a belt-and-suspenders check, not the primary cache key.
    if os.path.isfile(hdf5_path) and os.path.isfile(params_path):
        with open(params_path) as f:
            saved = json.load(f)
        if saved == current_params:
            return {"success": True, "hdf5_path": hdf5_path, "error": None, "cached": True,
                    "params_used": current_params}

    scratch_dir = None
    try:
        import shutil
        import time as _time

        from simnibs.simulation.sim_struct import TDCSLEADFIELD

        os.makedirs(base_dir, exist_ok=True)
        # Compute into a fresh scratch subdirectory first — this cache is now
        # keyed by tag (see leadfield_tag()), so a genuine settings mismatch
        # can't collide with this directory at all; the only case landing
        # here is an incomplete previous run into this exact tag. Computing
        # into scratch first (rather than directly into lf_dir) means a
        # failed recompute never destroys a working cached leadfield, and
        # never trips over that incomplete run's own leftover
        # simnibs_simulation*.mat files either.
        scratch_dir = os.path.join(base_dir, f"_regen_{tag}_{int(_time.time())}")
        os.makedirs(scratch_dir, exist_ok=True)

        lf_sess = TDCSLEADFIELD()
        lf_sess.subpath = m2m_path
        lf_sess.pathfem = scratch_dir
        lf_sess.eeg_cap = registered_cap_path
        lf_sess.electrode.shape = shape
        lf_sess.electrode.dimensions = list(dimensions)
        lf_sess.electrode.thickness = [gel_thickness]
        lf_sess.interpolation = None
        lf_sess.tissues = _LEADFIELD_TISSUES
        lf_sess.run(cpus=cpus)

        scratch_hdf5 = os.path.join(scratch_dir, fname)
        if not os.path.isfile(scratch_hdf5):
            return {"success": False, "hdf5_path": None,
                    "error": f"Leadfield HDF5 not found after run: {scratch_hdf5} "
                             f"(existing cached leadfield, if any, was left untouched)"}

        # New leadfield confirmed on disk — now safe to replace the old one.
        os.makedirs(lf_dir, exist_ok=True)
        if os.path.isfile(hdf5_path):
            os.remove(hdf5_path)
        shutil.move(scratch_hdf5, hdf5_path)
        with open(params_path, "w") as f:
            json.dump(current_params, f, indent=2)

        return {"success": True, "hdf5_path": hdf5_path, "error": None, "cached": False,
                "params_used": current_params}
    except Exception as e:
        return {"success": False, "hdf5_path": None,
                "error": f"{e} (existing cached leadfield, if any, was left untouched)"}
    finally:
        if scratch_dir and os.path.isdir(scratch_dir):
            import shutil
            shutil.rmtree(scratch_dir, ignore_errors=True)


# ═════════════════════════════════════════════════════════════════════════════
# One-off full FEM (no leadfield — exact electrode positions, snapped to the
# scalp surface, one physics solve per channel) + TI stats from the two
# resulting meshes. Meant to run inside a background job — ~1 minute/channel
# typical. Same technique as the 52Y notebook's run_manual_fem_channel /
# Step 4.
# ═════════════════════════════════════════════════════════════════════════════

def _snap_to_scalp(subject_id: str, coord, project_dir: str = PROJECT_DIR):
    import numpy as np
    from simnibs import mesh_io

    msh_path = os.path.join(get_m2m_path(subject_id, project_dir), f"{subject_id}.msh")
    nodes = mesh_io.read_msh(msh_path).crop_mesh(tags=[1005]).nodes.node_coord
    d = np.linalg.norm(nodes - np.asarray(coord), axis=1)
    return nodes[d.argmin()]


def run_manual_fem_channel(
    subject_id: str, plus_coord, minus_coord, current_mA: float,
    channel_label: str, out_dir: str,
    electrode_dims: tuple = (19.5, 19.5), electrode_thickness: float = 4.0,
    force: bool = False, project_dir: str = PROJECT_DIR,
) -> dict:
    """One tDCS FEM solve for one channel (+/- electrode pair, exact
    coordinates snapped to the scalp surface). Cached: returns the existing
    .msh if out_dir already has one and force=False.
    Returns {"success", "msh_path", "error"}."""
    import glob

    if not force and os.path.isdir(out_dir):
        existing = sorted(glob.glob(os.path.join(out_dir, "*.msh")))
        if existing:
            return {"success": True, "msh_path": existing[0], "error": None, "cached": True}

    m2m_path = get_m2m_path(subject_id, project_dir)
    if not os.path.isdir(m2m_path):
        return {"success": False, "msh_path": None, "error": f"m2m not found: {m2m_path}"}

    try:
        import shutil

        import simnibs
        from simnibs import sim_struct

        if force and os.path.isdir(out_dir):
            shutil.rmtree(out_dir)
        os.makedirs(out_dir, exist_ok=True)

        plus_snap = _snap_to_scalp(subject_id, plus_coord, project_dir)
        minus_snap = _snap_to_scalp(subject_id, minus_coord, project_dir)

        s = sim_struct.SESSION()
        s.subpath = m2m_path
        s.pathfem = out_dir
        s.open_in_gmsh = False
        tdcs = s.add_tdcslist()
        tdcs.currents = [current_mA * 1e-3, -current_mA * 1e-3]
        for nr, coord in enumerate([plus_snap, minus_snap], 1):
            el = tdcs.add_electrode()
            el.channelnr = nr
            el.centre = coord.tolist()
            el.shape = "ellipse"
            el.dimensions = list(electrode_dims)
            el.thickness = [electrode_thickness]
        simnibs.run_simnibs(s)

        files = sorted(glob.glob(os.path.join(out_dir, "*.msh")))
        if not files:
            return {"success": False, "msh_path": None, "error": f"FEM output not found in {out_dir}"}
        return {"success": True, "msh_path": files[0], "error": None, "cached": False}
    except Exception as e:
        return {"success": False, "msh_path": None, "error": str(e)}


def compute_ti_from_fem_meshes(msh1_path: str, msh2_path: str, roi_mask_path: str,
                                non_roi_mask_path: str | None = None,
                                extra_mask_paths: dict[str, str] | None = None) -> dict:
    """TI envelope + ROI/non-ROI stats from two single-channel FEM meshes —
    crop to WM+GM+CSF on read, restrict analysis to WM+GM, volume-weighted
    99th-percentile-capped mean (this project's standard metric).

    extra_mask_paths: {region_label: mask_path} — additional masks to also
    report TI_mean/TI_max for, from these SAME two already-solved meshes (no
    extra FEM solve). A missing/unreadable mask is skipped for that region.

    Returns {"success", "stats", "error"}."""
    import numpy as np
    from simnibs import mesh_io
    from simnibs.utils import TI_utils as TI

    def _vol_mean_capped(values, volumes, pct=99):
        if len(values) == 0:
            return float("nan")
        cap = float(np.percentile(values, pct))
        v = np.minimum(values, cap)
        return float((v * volumes).sum() / volumes.sum())

    def _geometry_and_field(msh_path):
        mt = mesh_io.read_msh(msh_path).crop_mesh(tags=[1, 2, 3])
        nodes = mt.nodes.node_coord
        conn = mt.elm.node_number_list[:, :4] - 1
        v0, v1, v2, v3 = nodes[conn[:, 0]], nodes[conn[:, 1]], nodes[conn[:, 2]], nodes[conn[:, 3]]
        centroids = nodes[conn].mean(axis=1)
        volumes = np.abs(np.einsum("ni,ni->n", v1 - v0, np.cross(v2 - v0, v3 - v0))) / 6.0
        tags = mt.elm.tag1.astype(np.int32)
        ef = next(d for d in mt.elmdata if d.field_name == "E").value
        return centroids, volumes, tags, ef

    def _mask_to_elements(mask_path, centroids):
        import nibabel as nib
        img = nib.load(mask_path)
        data = np.asarray(img.dataobj) > 0
        aff_inv = np.linalg.inv(img.affine)
        ones = np.ones((len(centroids), 1))
        vox = (aff_inv @ np.hstack([centroids, ones]).T).T[:, :3]
        vox_idx = np.round(vox).astype(int)
        sh = data.shape
        in_bounds = ((vox_idx[:, 0] >= 0) & (vox_idx[:, 0] < sh[0]) &
                     (vox_idx[:, 1] >= 0) & (vox_idx[:, 1] < sh[1]) &
                     (vox_idx[:, 2] >= 0) & (vox_idx[:, 2] < sh[2]))
        mask_out = np.zeros(len(centroids), dtype=bool)
        mask_out[in_bounds] = data[vox_idx[in_bounds, 0], vox_idx[in_bounds, 1], vox_idx[in_bounds, 2]]
        return mask_out

    try:
        centroids, volumes, tags, ef1 = _geometry_and_field(msh1_path)
        _, _, _, ef2 = _geometry_and_field(msh2_path)
        ti = TI.get_maxTI(ef1, ef2)

        wm_gm = np.isin(tags, [1, 2])
        c_v, vol_v, ti_v = centroids[wm_gm], volumes[wm_gm], ti[wm_gm]

        stats = {
            "TI_max_whole_brain_V_m": float(ti_v.max()),
            "TI_mean_whole_brain_V_m": _vol_mean_capped(ti_v, vol_v),
        }

        roi_mask = _mask_to_elements(roi_mask_path, c_v)
        if roi_mask.any():
            roi_mean = _vol_mean_capped(ti_v[roi_mask], vol_v[roi_mask])
            stats["TI_max_ROI_V_m"] = float(ti_v[roi_mask].max())
            stats["TI_mean_ROI_V_m"] = roi_mean
            if stats["TI_mean_whole_brain_V_m"]:
                stats["focality_ratio_brain"] = roi_mean / stats["TI_mean_whole_brain_V_m"]

            if non_roi_mask_path and os.path.isfile(non_roi_mask_path):
                non_roi_mask = _mask_to_elements(non_roi_mask_path, c_v)
                if non_roi_mask.any():
                    nr_mean = _vol_mean_capped(ti_v[non_roi_mask], vol_v[non_roi_mask])
                    stats["TI_mean_non_ROI_V_m"] = nr_mean
                    if nr_mean:
                        stats["focality_ratio_non_roi"] = roi_mean / nr_mean

        if extra_mask_paths:
            for region_label, mask_path in extra_mask_paths.items():
                if not mask_path or not os.path.isfile(mask_path):
                    continue
                xmask = _mask_to_elements(mask_path, c_v)
                if not xmask.any():
                    continue
                stats[f"TI_mean_{region_label}_V_m"] = _vol_mean_capped(ti_v[xmask], vol_v[xmask])
                stats[f"TI_max_{region_label}_V_m"] = float(ti_v[xmask].max())

        return {"success": True, "stats": stats, "error": None}
    except Exception as e:
        return {"success": False, "stats": None, "error": str(e)}


def run_one_off_fem(
    subject_id: str, roi_mask_path: str, non_roi_mask_path: str | None,
    ch1_plus_coord, ch1_minus_coord, ch1_current_mA: float,
    ch2_plus_coord, ch2_minus_coord, ch2_current_mA: float,
    label: str = "manual_fem", electrode_dims: tuple = (19.5, 19.5),
    electrode_thickness: float = 4.0, force: bool = False, project_dir: str = PROJECT_DIR,
    extra_mask_paths: dict[str, str] | None = None,
) -> dict:
    """Both channels' FEM + TI stats in one call — the full one-off
    (no-leadfield) pipeline, meant to run inside a background job (this is
    the function passed to job_runner.start_local_job).
    Returns {"success", "error", "stats"?, "ch1_msh"?, "ch2_msh"?}."""
    base_dir = os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}",
                             "comparison", "manual_fem", label)

    ch1 = run_manual_fem_channel(subject_id, ch1_plus_coord, ch1_minus_coord, ch1_current_mA,
                                  "ch1", os.path.join(base_dir, "ch1"),
                                  electrode_dims, electrode_thickness, force, project_dir)
    if not ch1["success"]:
        return {"success": False, "error": f"Channel 1 FEM failed: {ch1['error']}"}

    ch2 = run_manual_fem_channel(subject_id, ch2_plus_coord, ch2_minus_coord, ch2_current_mA,
                                  "ch2", os.path.join(base_dir, "ch2"),
                                  electrode_dims, electrode_thickness, force, project_dir)
    if not ch2["success"]:
        return {"success": False, "error": f"Channel 2 FEM failed: {ch2['error']}"}

    ti = compute_ti_from_fem_meshes(ch1["msh_path"], ch2["msh_path"], roi_mask_path, non_roi_mask_path,
                                     extra_mask_paths=extra_mask_paths)
    if not ti["success"]:
        return {"success": False, "error": f"TI computation failed: {ti['error']}"}

    return {"success": True, "error": None, "stats": ti["stats"],
            "ch1_msh": ch1["msh_path"], "ch2_msh": ch2["msh_path"]}

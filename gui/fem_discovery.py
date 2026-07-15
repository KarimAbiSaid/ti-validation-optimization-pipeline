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


def list_leadfields(subject_id: str, project_dir: str = PROJECT_DIR) -> list[dict]:
    """Precomputed leadfields already on disk for this subject —
    {"cap_name", "hdf5_path"}, one per sub-{id}_leadfield_{cap_name}.hdf5."""
    d = leadfield_dir(subject_id, project_dir)
    prefix = f"{subject_id}_leadfield_"
    out = []
    if os.path.isdir(d):
        for fname in sorted(os.listdir(d)):
            if fname.startswith(prefix) and fname.endswith(".hdf5"):
                cap_name = fname[len(prefix):-len(".hdf5")]
                out.append({"cap_name": cap_name, "hdf5_path": os.path.join(d, fname)})
    return out


def leadfield_status(subject_id: str, cap_name: str, project_dir: str = PROJECT_DIR) -> dict:
    hdf5_path = os.path.join(leadfield_dir(subject_id, project_dir), f"{subject_id}_leadfield_{cap_name}.hdf5")
    return {"exists": os.path.isfile(hdf5_path), "hdf5_path": hdf5_path}


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
    subject_id: str, cap_name: str, roi_mask_path: str, non_roi_mask_path: str | None,
    ch1_plus: str, ch1_minus: str, ch1_current_mA: float,
    ch2_plus: str, ch2_minus: str, ch2_current_mA: float,
    electrode_dims: list[float] | None = None, label: str = "setup",
    output_dir: str | None = None, project_dir: str = PROJECT_DIR,
) -> dict:
    """Returns {"success", "error"} or, on success, {"success": True, "label",
    "montage", "stats", "msh_path", "npz_path", "json_path"}."""
    status = leadfield_status(subject_id, cap_name, project_dir)
    if not status["exists"]:
        return {"success": False,
                "error": f"No leadfield for sub-{subject_id} / {cap_name}: {status['hdf5_path']}"}

    names = set(leadfield_electrode_names(status["hdf5_path"]))
    for field_label, val in [("Ch1+", ch1_plus), ("Ch1-", ch1_minus),
                              ("Ch2+", ch2_plus), ("Ch2-", ch2_minus)]:
        if val not in names:
            return {"success": False,
                     "error": f"{field_label} '{val}' not in this leadfield's electrodes: {sorted(names)}"}

    if output_dir is None:
        output_dir = os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}", "comparison")

    from compare_ti_montages import load_subject_resources, compute_ti_setup

    try:
        resources = load_subject_resources(
            subject_id=subject_id, project_dir=project_dir, cap_name=cap_name,
            roi_mask=roi_mask_path, non_roi_mask=non_roi_mask_path,
        )
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

    return {
        "success": True, "label": result["label"], "montage": result["montage"],
        "stats": result["stats"], "msh_path": result["msh_path"],
        "npz_path": result["npz_path"], "json_path": result["json_path"],
    }


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
) -> dict:
    """Same as compute_ti(), but for a leadfield living outside the standard
    leadfield_volume/ location (an explicit hdf5_path rather than a
    subject+cap_name lookup)."""
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

    return {
        "success": True, "label": result["label"], "montage": result["montage"],
        "stats": result["stats"], "msh_path": result["msh_path"],
        "npz_path": result["npz_path"], "json_path": result["json_path"],
    }


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

    cap_name = os.path.splitext(os.path.basename(registered_cap_path))[0]
    lf_dir = leadfield_dir(subject_id, project_dir)
    hdf5_path = os.path.join(lf_dir, f"{subject_id}_leadfield_{cap_name}.hdf5")
    params_path = os.path.join(lf_dir, f"{subject_id}_leadfield_{cap_name}_params.json")

    current_params = {
        "shape": shape, "dimensions": list(dimensions), "gel_thickness": gel_thickness,
        "tissues": _LEADFIELD_TISSUES, "interpolation": None,
    }

    if os.path.isfile(hdf5_path) and os.path.isfile(params_path):
        with open(params_path) as f:
            saved = json.load(f)
        if saved == current_params:
            return {"success": True, "hdf5_path": hdf5_path, "error": None, "cached": True,
                    "params_used": current_params}
        # Params differ — recompute needed. Do NOT touch the existing hdf5/json
        # yet: compute into a fresh scratch subdirectory first (this also
        # sidesteps SimNIBS's own "existing simulation results" refusal if
        # leftover simnibs_simulation*.mat files sit in lf_dir from an
        # unrelated earlier run) and only replace the old file once the new
        # one is confirmed written — a failed recompute must never destroy a
        # working cached leadfield.

    scratch_dir = None
    try:
        import shutil
        import time as _time

        from simnibs.simulation.sim_struct import TDCSLEADFIELD

        os.makedirs(lf_dir, exist_ok=True)
        scratch_dir = os.path.join(lf_dir, f"_regen_{cap_name}_{int(_time.time())}")
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

        scratch_hdf5 = os.path.join(scratch_dir, f"{subject_id}_leadfield_{cap_name}.hdf5")
        if not os.path.isfile(scratch_hdf5):
            return {"success": False, "hdf5_path": None,
                    "error": f"Leadfield HDF5 not found after run: {scratch_hdf5} "
                             f"(existing cached leadfield, if any, was left untouched)"}

        # New leadfield confirmed on disk — now safe to replace the old one.
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
                                non_roi_mask_path: str | None = None) -> dict:
    """TI envelope + ROI/non-ROI stats from two single-channel FEM meshes —
    crop to WM+GM+CSF on read, restrict analysis to WM+GM, volume-weighted
    99th-percentile-capped mean (this project's standard metric).
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

        return {"success": True, "stats": stats, "error": None}
    except Exception as e:
        return {"success": False, "stats": None, "error": str(e)}


def run_one_off_fem(
    subject_id: str, roi_mask_path: str, non_roi_mask_path: str | None,
    ch1_plus_coord, ch1_minus_coord, ch1_current_mA: float,
    ch2_plus_coord, ch2_minus_coord, ch2_current_mA: float,
    label: str = "manual_fem", electrode_dims: tuple = (19.5, 19.5),
    electrode_thickness: float = 4.0, force: bool = False, project_dir: str = PROJECT_DIR,
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

    ti = compute_ti_from_fem_meshes(ch1["msh_path"], ch2["msh_path"], roi_mask_path, non_roi_mask_path)
    if not ti["success"]:
        return {"success": False, "error": f"TI computation failed: {ti['error']}"}

    return {"success": True, "error": None, "stats": ti["stats"],
            "ch1_msh": ch1["msh_path"], "ch2_msh": ch2["msh_path"]}

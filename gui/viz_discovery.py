"""
viz_discovery.py — Phase 4+ figure rendering: TI field on the cortex and/or
one or more named ROI/non-ROI regions, three orthogonal views inside a
translucent head, electrodes marked, one shared colorbar. No Dash import,
same layering as fem_discovery.py/comparison_discovery.py.

Adapted from a standalone two-environment CLI pipeline (code/viz/, written
and suggested by Wei, thank you! See code/viz/SOP.md for the full original procedure
and its figure-checking judgement calls, which still apply here) into one
in-process, SINGLE-environment module for the GUI. Two deliberate changes
from that original:

  - ONE environment, not two. Wei's split existed to protect a bare SimNIBS
    install's pinned numpy from pip's dependency resolution when adding
    PyVista. This project's GUI env already carries simnibs + numpy +
    nibabel + matplotlib; installing pyvista into it pulled in only its own
    small dependency set (added one more dependency)
    (verified via `pip install --dry-run` first) and
    touched neither numpy nor matplotlib. So extraction and rendering here
    are two plain function calls, both against this same interpreter. (Extraction
    runs in-process. Rendering itself, despite being "the same environment,"
    still has to run in a separate PROCESS, not the calling thread, see
    build_figure_subprocess() below for why; that's an unrelated,
    thread-affinity problem, not a dependency one.)
  - Regions come from THIS PROJECT's own mask files (sub-{id}_label-{name}_
    mask.nii.gz, so the mask-file route means the same "existing mask" picker everywhere
    else in the GUI doubles as "which regions to render," "cortex" stops
    being a hardcoded special case, and any number of regions (cortex AND
    amygdala AND whatever else) draw together in one figure instead of one
    ROI vs. a neutral shell.

The core per-mesh geometry technique (boundary_faces/compact, extracting a
tet mesh's boundary triangles and re-indexing) and the PyVista rendering
technique (enable_depth_peeling() for correct translucent-shell compositing
from every angle) are Wei's functions, unchanged.

One addition neither original script had: VTK's decimate() drops ALL
attribute data from its output (verified empirically, not documented), so
a heavy surface is decimated as bare geometry first, then its field values
are resampled onto the new vertices via nearest-neighbor lookup against the
original (pre-decimation) vertices. This also means every surface here
carries its field as POINT data (smoothly gradated), not per-face CELL data
like Wei's originals (faceted), a decimated and an undecimated region now
render with the same, smoother look. Volumes (used only for --clip cut
planes, never decimated) are untouched, still per-tet cell data.

A second addition: an always-drawn, uncolored, very-translucent whole-brain
shell (see DEFAULT_BRAIN_OPACITY / _extract_anatomical_shell), alongside the
scalp. This replaced an earlier "context region" mechanism (one region drawn
translucent + field-colored, e.g. cortex, so a highlight buried inside it
would show through) that briefly existed here, but then we decided that the brain shell does that
same "see where the highlight sits" job unconditionally, for every figure,
without needing a colored region picked at all, so the separate context
concept (and its own opacity/color knobs) was removed as redundant. Every
region passed to render_figure() now plays exactly one role: highlighted.
"""
from __future__ import annotations

import os

import numpy as np

from common import PROJECT_DIR, get_m2m_path  # noqa: F401 (re-exported)

# SimNIBS 4 charm tags: 1 = WM, 2 = GM, 5 = scalp, 7/8 = compact/spongy bone.
# Surface tags are 1000 + the volume tag — pre-existing 2D triangle elements
# already in the head model .msh (verified: no boundary-extraction needed,
# same as the scalp shell). GM_SURF_TAG (1002, the pial surface) is the
# outer boundary of the WHOLE BRAIN as seen from outside, since WM sits
# entirely inside GM in this stacked-shell tissue model — used for the
# "no regions picked" whole-GM fallback AND (unconditionally) for the
# always-present neutral "whole brain" shell (see extract_figure_cache()).
# Every actual highlighted region still comes from a mask file.
GM_TAG = 2
GM_SURF_TAG = 1002
SCALP_SURF_TAG = 1005

CH1_COLOR = "#0073ff"
CH2_COLOR = "#fe4444"

# A boundary surface heavier than this gets decimated before being cached —
# keeps a whole-cortex figure fast without visibly holing the mesh
# (decimation is proper triangle simplification, not point-dropping). A
# small deep ROI is typically well under this and is written untouched.
DEFAULT_MAX_FACES = 120_000

# Small per-vertex random offset applied once at extract time, purely
# cosmetic — breaks up the visible axis-aligned patterning raw FEM mesh node
# positions sometimes show. Subtle enough not to distort anatomy (0 disables
# it); render-time smoothing below softens it further either way.
DEFAULT_JITTER_MM = 0.15

# Taubin (shrinkage-free, unlike plain Laplacian) smoothing at RENDER time —
# same technique and defaults Wei's own render_ti.py used for its ROI
# surface, generalized here to every surface (highlights, scalp, brain).
# Render time, not extract time, because it doesn't need the cache touched
# to retune (0 disables it; point/cell data survives this filter unchanged,
# unlike decimate() — see this module's docstring).
DEFAULT_SMOOTH_ITERS = 20
DEFAULT_SMOOTH_PASS_BAND = 0.05

# Highlighted regions render translucent by default now, not fully opaque —
# lets you see some depth into the volume instead of just its outer shell.
DEFAULT_HIGHLIGHT_OPACITY = 0.75

# The whole-brain shell (GM_SURF_TAG) is always drawn, plain gray like the
# scalp but distinguishable from it (slightly darker) — a fixed anatomical
# backdrop so a highlighted region reads as "located here in the brain"
# without competing for visual attention. Very low opacity by design.
DEFAULT_BRAIN_OPACITY = 0.08
DEFAULT_BRAIN_COLOR = "#7a7a7a"


# ═════════════════════════════════════════════════════════════════════════════
# Core geometry (Wei's algorithm, unchanged)
# ═════════════════════════════════════════════════════════════════════════════

def _boundary_faces(tets: np.ndarray):
    """Boundary triangles of a tet mesh, plus the parent tet index of each
    one. A face shared by two tets is interior; a face appearing exactly
    once is on the boundary. Winding is left alone — PyVista recomputes
    normals."""
    faces = np.concatenate([
        tets[:, [0, 2, 1]], tets[:, [0, 1, 3]], tets[:, [0, 3, 2]], tets[:, [1, 2, 3]],
    ])
    parent = np.tile(np.arange(len(tets)), 4)

    key = np.sort(faces, axis=1)
    order = np.lexsort(key.T[::-1])
    key_sorted = key[order]

    shared = np.all(key_sorted[1:] == key_sorted[:-1], axis=1)
    interior = np.zeros(len(key_sorted), dtype=bool)
    interior[:-1] |= shared
    interior[1:] |= shared

    keep = order[~interior]
    return faces[keep], parent[keep]


def _compact(nodes: np.ndarray, faces: np.ndarray):
    """Drop unreferenced nodes and reindex the faces onto what is left."""
    used, inverse = np.unique(faces, return_inverse=True)
    return nodes[used], inverse.reshape(faces.shape).astype(np.int32)


def _decimate_surface(verts, faces, point_values, max_faces=DEFAULT_MAX_FACES):
    """No-op under max_faces (a typical deep ROI). Otherwise: VTK's
    decimate() drops all attribute data, so this decimates bare geometry,
    then resamples point_values onto the new vertices via nearest-neighbor
    lookup against the original vertices — simpler and more robust than
    radius-based interpolation, and anatomically fine since a boundary
    surface's vertex density is fairly uniform."""
    if len(faces) <= max_faces:
        return verts, faces, point_values
    import pyvista as pv
    from scipy.spatial import cKDTree

    vtk_faces = np.hstack([np.full((len(faces), 1), 3, np.int32), faces]).ravel()
    poly = pv.PolyData(verts.astype(np.float32), vtk_faces)
    reduction = 1.0 - (max_faces / len(faces))
    dec = poly.decimate(reduction)
    new_faces = dec.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)
    new_verts = np.asarray(dec.points, dtype=np.float32)

    _, idx = cKDTree(verts).query(new_verts)
    return new_verts, new_faces, point_values[idx]


def _face_values_to_points(verts, faces, face_values):
    """Per-face -> per-vertex field, by averaging the faces touching each
    vertex (pyvista's own cell_data_to_point_data, used directly since it's
    the same operation whether or not decimation follows)."""
    import pyvista as pv
    vtk_faces = np.hstack([np.full((len(faces), 1), 3, np.int32), faces]).ravel()
    poly = pv.PolyData(verts.astype(np.float32), vtk_faces)
    poly.cell_data["TI"] = face_values
    return np.asarray(poly.cell_data_to_point_data().point_data["TI"], dtype=np.float32)


def _jitter_vertices(verts, jitter_mm, seed=0):
    """Small per-vertex random offset, applied once at extract time —
    purely cosmetic, not a data change. A fixed seed keeps repeated
    extracts of the same region deterministic rather than jittering
    differently on every re-extract. No-op if jitter_mm <= 0."""
    if jitter_mm <= 0:
        return verts
    rng = np.random.default_rng(seed)
    return (verts + rng.uniform(-jitter_mm, jitter_mm, verts.shape)).astype(np.float32)


# ═════════════════════════════════════════════════════════════════════════════
# Step 1 — mesh(es) -> cached .npz surfaces/volumes (SimNIBS + pyvista, both
# in this same environment now)
# ═════════════════════════════════════════════════════════════════════════════

def _field_values(mesh, field_name):
    values = [f for f in mesh.elmdata if f.field_name == field_name][0].value
    return np.linalg.norm(values, axis=1) if values.ndim > 1 else values


def _extract_region(msh, field_name, mask_path, label, out_dir, max_faces, jitter_mm=DEFAULT_JITTER_MM):
    """One named region: crop to the tets whose centroid falls inside
    mask_path (any tissue tag — selection is purely by the mask, same
    reasoning as Wei's SAMSEG-based roi_elements(), just driven by this
    project's own mask files instead of label codes), write {label}_
    surface.npz (decimated if heavy) and {label}_volume.npz (for --clip,
    never decimated). Returns a small summary dict, or None if the mask
    matched no elements for this subject (skipped, not an error — same
    "opportunistic, drop if missing/empty" spirit as fem_discovery.py's
    extra-region stats)."""
    from compare_ti_montages import _mask_to_elements

    is_tet = msh.elm.elm_type == 4
    all_tets = msh.elm.node_number_list[is_tet][:, :4] - 1
    centroids = msh.nodes.node_coord[all_tets].mean(axis=1)
    inside = _mask_to_elements(mask_path, centroids)
    elm_numbers = msh.elm.elm_number[is_tet][inside]
    if len(elm_numbers) == 0:
        return None

    region = msh.crop_mesh(elements=elm_numbers)
    values = _field_values(region, field_name)

    tets = region.elm.node_number_list[:, :4] - 1
    faces, parent = _boundary_faces(tets)
    verts, faces = _compact(region.nodes.node_coord, faces)
    point_values = _face_values_to_points(verts, faces, values[parent])
    verts = _jitter_vertices(verts, jitter_mm)
    verts, faces, point_values = _decimate_surface(verts, faces, point_values, max_faces)

    np.savez_compressed(
        os.path.join(out_dir, f"{label}_surface.npz"),
        verts=verts.astype(np.float32), faces=faces,
        values=point_values.astype(np.float32), field_name=field_name, region_name=label,
    )

    # Volume too, for --clip cut planes — a cut through the surface alone
    # would only expose the inside of a hollow shell. Untouched, per-tet
    # cell data, no decimation (ROI volumes are small relative to the whole
    # head, and clip geometry needs the real tets).
    vol_verts, vol_tets = _compact(region.nodes.node_coord, tets)
    np.savez_compressed(
        os.path.join(out_dir, f"{label}_volume.npz"),
        verts=vol_verts.astype(np.float32), tets=vol_tets,
        values=values.astype(np.float32), field_name=field_name, region_name=label,
    )

    return {"label": label, "n_faces": len(faces), "n_tets": len(elm_numbers),
           "value_min": float(point_values.min()), "value_max": float(point_values.max())}


def _extract_anatomical_shell(head_msh, tag, out_path, max_faces, jitter_mm):
    """A plain (no field data) translucent shell straight from a pre-existing
    2D surface tag already in the head model .msh — the scalp (1005) and
    the whole-brain pial surface (GM_SURF_TAG, 1002) both work this way, no
    tet-boundary computation needed. Returns n_faces written."""
    shell = head_msh.crop_mesh(tags=[tag])
    tris = shell.elm.node_number_list[:, :3] - 1
    verts, faces = _compact(shell.nodes.node_coord, tris)
    verts = _jitter_vertices(verts, jitter_mm)
    if len(faces) > max_faces:
        import pyvista as pv
        vtk_faces = np.hstack([np.full((len(faces), 1), 3, np.int32), faces]).ravel()
        poly = pv.PolyData(verts.astype(np.float32), vtk_faces).decimate(1.0 - max_faces / len(faces))
        verts = np.asarray(poly.points, dtype=np.float32)
        faces = poly.faces.reshape(-1, 4)[:, 1:4].astype(np.int32)
    np.savez_compressed(out_path, verts=verts.astype(np.float32), faces=faces)
    return len(faces)


def extract_figure_cache(
    field_mesh_path: str, head_mesh_path: str, field_name: str,
    region_mask_paths: dict[str, str], out_dir: str,
    max_faces_per_region: int = DEFAULT_MAX_FACES, jitter_mm: float = DEFAULT_JITTER_MM,
) -> dict:
    """Writes, always, {out_dir}/scalp_surface.npz and brain_surface.npz
    (both plain shells, no field data — the whole-brain pial surface, a
    fixed low-opacity anatomical backdrop regardless of which regions are
    drawn on top of it) plus one {label}_surface.npz (+ _volume.npz) per
    entry in region_mask_paths — or, if region_mask_paths is empty, a single
    "cortex" region falling back to the whole GM tag (mirrors Wei's original
    default "no --labeling -> plain cortical figure"; distinct from
    brain_surface.npz above in that this one carries the FIELD and gets
    drawn as a highlight, not as the fixed neutral backdrop).

    field_mesh_path: the TI/E-field simulation output (comparison/*_head_
    mesh.msh style — carries max_TI/normE_ch1/normE_ch2/E_ch1/E_ch2).
    head_mesh_path: m2m_{id}/{id}.msh, for the scalp/brain shells only.
    region_mask_paths: {label: mask_nii_path} — this project's own
    sub-{id}_label-{name}_mask.nii.gz files (comparison_discovery.
    resolve_mask() resolves a bare label to its path).

    Returns {"regions": [summary per _extract_region, dropped ones
    omitted], "scalp": {"n_faces"}, "brain": {"n_faces"}, "field_range":
    {"min","max"} across ALL tissue (same info step 1 always printed, for
    the caller to sanity-check against --vmax before rendering}."""
    import simnibs

    os.makedirs(out_dir, exist_ok=True)
    field_msh = simnibs.read_msh(str(field_mesh_path))
    names = [f.field_name for f in field_msh.elmdata]
    if field_name not in names:
        raise ValueError(f"field {field_name!r} not in {field_mesh_path}; available: {names}")

    whole_values = _field_values(field_msh, field_name)
    field_range = {"min": float(whole_values.min()), "max": float(whole_values.max())}

    regions = []
    if region_mask_paths:
        for label, mask_path in region_mask_paths.items():
            summary = _extract_region(field_msh, field_name, mask_path, label, out_dir,
                                      max_faces_per_region, jitter_mm)
            if summary:
                regions.append(summary)
    else:
        # Fallback: whole grey matter, same as Wei's extract_brain() default.
        brain = field_msh.crop_mesh(tags=[GM_TAG])
        values = _field_values(brain, field_name)
        tets = brain.elm.node_number_list[:, :4] - 1
        faces, parent = _boundary_faces(tets)
        verts, faces = _compact(brain.nodes.node_coord, faces)
        point_values = _face_values_to_points(verts, faces, values[parent])
        verts = _jitter_vertices(verts, jitter_mm)
        verts, faces, point_values = _decimate_surface(verts, faces, point_values, max_faces_per_region)
        np.savez_compressed(
            os.path.join(out_dir, "cortex_surface.npz"),
            verts=verts.astype(np.float32), faces=faces,
            values=point_values.astype(np.float32), field_name=field_name, region_name="cortex",
        )
        regions.append({"label": "cortex", "n_faces": len(faces), "n_tets": len(tets),
                        "value_min": float(point_values.min()), "value_max": float(point_values.max())})

    head_msh = simnibs.read_msh(str(head_mesh_path))
    n_scalp_faces = _extract_anatomical_shell(head_msh, SCALP_SURF_TAG,
                                              os.path.join(out_dir, "scalp_surface.npz"),
                                              max_faces_per_region, jitter_mm)
    n_brain_faces = _extract_anatomical_shell(head_msh, GM_SURF_TAG,
                                              os.path.join(out_dir, "brain_surface.npz"),
                                              max_faces_per_region, jitter_mm)

    return {"regions": regions, "scalp": {"n_faces": n_scalp_faces}, "brain": {"n_faces": n_brain_faces},
           "field_range": field_range}


# ═════════════════════════════════════════════════════════════════════════════
# Step 2 — cached .npz -> figure (same PyVista technique as Wei's render_ti.py:
# enable_depth_peeling() is what makes the translucent shell composite
# correctly from every angle — do not drop it)
# ═════════════════════════════════════════════════════════════════════════════

VIEWS = {
    "Axial":    dict(direction=(0, 0, 1),  viewup=(0, 1, 0)),   # from above
    "Coronal":  dict(direction=(0, 1, 0),  viewup=(0, 0, 1)),   # from the front
    "Sagittal": dict(direction=(1, 0, 0),  viewup=(0, 0, 1)),   # from the right
}


def _load_surface(path, smooth_iters=DEFAULT_SMOOTH_ITERS, smooth_pass_band=DEFAULT_SMOOTH_PASS_BAND):
    import pyvista as pv
    d = np.load(path)
    faces = d["faces"]
    vtk_faces = np.hstack([np.full((len(faces), 1), 3, np.int32), faces]).ravel()
    mesh = pv.PolyData(d["verts"].astype(np.float32), vtk_faces)
    if "values" in d:
        mesh.point_data["TI"] = d["values"]  # point data now -- see module docstring
    if smooth_iters > 0:
        # Taubin (shrinkage-free) smoothing at RENDER time, not baked into
        # the cache -- point/cell data survives this filter unchanged
        # (verified; unlike decimate()), so this is cheap to retune without
        # re-extracting.
        mesh = mesh.smooth_taubin(n_iter=smooth_iters, pass_band=smooth_pass_band)
    return mesh


def _load_volume(path):
    import pyvista as pv
    d = np.load(path)
    tets = d["tets"]
    cells = np.hstack([np.full((len(tets), 1), 4, np.int32), tets]).ravel()
    grid = pv.UnstructuredGrid(cells, np.full(len(tets), pv.CellType.TETRA, np.uint8),
                               d["verts"].astype(np.float32))
    grid.cell_data["TI"] = d["values"]
    return grid


def _load_electrodes(path, ch1, ch2):
    import csv
    rows = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 5 and row[0].strip().lower() == "electrode":
                rows[row[4].strip()] = np.array([float(v) for v in row[1:4]])
    missing = [n for n in (*ch1, *ch2) if n not in rows]
    if missing:
        raise ValueError(f"electrodes {missing} not in {path}; found {sorted(rows)}")
    return [(n, rows[n], CH1_COLOR) for n in ch1] + [(n, rows[n], CH2_COLOR) for n in ch2]


def _trim_white(img, pad=8):
    """Crop the uniform background border so the panels sit tight together."""
    mask = (img < 250).any(axis=2)
    if not mask.any():
        return img
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def _render_view(scalp, brain, highlights, electrodes, view, opts):
    """scalp/brain: plain, uncolored, very-translucent anatomical shells
    (head skin / whole-brain pial surface) — always drawn as a fixed,
    low-opacity backdrop, regardless of which regions are highlighted (this
    is what a separate "context region" — one region drawn translucent +
    field-colored so a highlight buried inside it would show through —
    used to be for; the brain shell does that job unconditionally now, so
    that mechanism was removed). highlights: [(label, surface_mesh,
    volume_mesh_or_None)] — drawn opaque (or ramped by value_opacity), on
    top of the shells. All highlights share ONE field colormap (same
    physical quantity, same clim) — there is no per-region color-coding;
    regions are told apart by shape/position and by their own text label,
    added at each one's centroid."""
    import pyvista as pv

    plotter = pv.Plotter(off_screen=True, window_size=opts["window_size"])
    plotter.set_background("white")
    plotter.enable_depth_peeling(number_of_peels=opts["peels"], occlusion_ratio=0.0)

    anchor = highlights[0][1] if highlights else (brain or scalp)
    focal = np.array(anchor.center)
    span = float(np.linalg.norm(np.array(anchor.bounds[1::2]) - np.array(anchor.bounds[::2])))
    direction = np.array(VIEWS[view]["direction"], float)

    clip_origin = None
    if opts["clip"] and highlights:
        vol_anchor = highlights[0][2] if highlights[0][2] is not None else highlights[0][1]
        clip_origin = np.array(vol_anchor.center) + direction * opts["clip_offset"]
        kw = dict(normal=tuple(direction), origin=tuple(clip_origin))
        scalp = scalp.clip(**kw) if scalp is not None else None
        brain = brain.clip(**kw) if brain is not None else None
        highlights = [(label, (vol.clip(**kw) if vol is not None else surf.clip(**kw)), vol)
                     for label, surf, vol in highlights]

    field_opacity = "linear" if opts["value_opacity"] else opts["highlight_opacity"]
    label_points = []
    for label, surf, _vol in highlights:
        plotter.add_mesh(surf, scalars="TI", cmap=opts["cmap"], clim=(opts["vmin"], opts["vmax"]),
                         show_scalar_bar=False, opacity=field_opacity,
                         smooth_shading=True, ambient=0.3, diffuse=0.8, specular=0.15)
        label_points.append((label, np.array(surf.center)))
    if brain is not None:
        plotter.add_mesh(brain, color=opts["brain_color"], opacity=opts["brain_opacity"],
                         smooth_shading=True, specular=0.0, diffuse=0.6)
    if scalp is not None:
        plotter.add_mesh(scalp, color=opts["shell_color"], opacity=opts["shell_opacity"],
                         smooth_shading=True, specular=0.0, diffuse=0.6)

    drawn = []
    for name, pos, color in electrodes:
        if clip_origin is not None:
            if float(np.dot(pos - clip_origin, direction)) > 0:
                continue
            opacity = 1.0
        else:
            near = float(np.dot(pos - focal, direction)) > 0
            if not near and opts["far_electrode_opacity"] <= 0:
                continue
            opacity = 1.0 if near else opts["far_electrode_opacity"]
        plotter.add_mesh(pv.Sphere(radius=opts["electrode_radius"], center=pos),
                         color=color, specular=0.4, smooth_shading=True, opacity=opacity)
        drawn.append((name, pos))

    if opts["label_electrodes"]:
        labelled = drawn if clip_origin is not None else [
            (n, p) for n, p in drawn if float(np.dot(p - focal, direction)) > 0]
        if labelled:
            plotter.add_point_labels(
                np.array([p for _, p in labelled]), [n for n, _ in labelled],
                font_size=opts["window_size"][1] // 45, text_color="black",
                shape=None, always_visible=True, show_points=False)

    if opts["label_regions"] and label_points and len(label_points) > 1:
        # Only worth labelling when more than one region is on screen at
        # once — a single region is already unambiguous from the colorbar
        # title / figure caption alone.
        plotter.add_point_labels(
            np.array([p for _, p in label_points]), [n for n, _ in label_points],
            font_size=opts["window_size"][1] // 40, text_color="black", bold=True,
            shape_color="white", shape_opacity=0.6, always_visible=True, show_points=False)

    plotter.camera_position = [tuple(focal + direction * span), tuple(focal), VIEWS[view]["viewup"]]
    plotter.reset_camera()
    plotter.camera.zoom(opts["zoom"])

    img = plotter.screenshot(return_img=True)
    plotter.close()
    return _trim_white(img)


def render_figure(
    cache_dir: str, electrodes_csv: str, highlight_labels: list[str], out_path: str,
    ch1=("AR", "AL"), ch2=("PR", "PL"),
    vmin: float = 0.0, vmax: float = 1.0, cmap: str = "plasma",
    title: str | None = None, clip: bool = False, clip_offset: float = 0.0, value_opacity: bool = False,
    highlight_opacity: float = DEFAULT_HIGHLIGHT_OPACITY,
    smooth_iters: int = DEFAULT_SMOOTH_ITERS, smooth_pass_band: float = DEFAULT_SMOOTH_PASS_BAND,
    shell_opacity: float = 0.12, shell_color: str = "#9a9a9a",
    brain_opacity: float = DEFAULT_BRAIN_OPACITY, brain_color: str = DEFAULT_BRAIN_COLOR,
    no_shell: bool = False, no_brain: bool = False,
    electrode_radius: float = 7.0, far_electrode_opacity: float = 0.25,
    label_electrodes: bool = True, label_regions: bool = True,
    window_size=(1100, 1100), peels: int = 12, zoom: float = 1.25, dpi: int = 300,
) -> dict:
    """Composes the 3-view figure from a cache directory written by
    extract_figure_cache(). Every drawn region shares ONE field colormap
    (same physical quantity, same clim) — there's no per-region color-
    coding; regions are told apart by a text label at each one's centroid
    (when more than one is on screen).

    highlight_labels: regions drawn at highlight_opacity (translucent by
    default, not fully opaque — see some depth into the volume rather than
    just its outer shell), on top of the always-present scalp/brain shells
    (see extract_figure_cache()/_extract_anatomical_shell() — the
    whole-brain shell is what used to be a separate, pickable "context
    region"; that mechanism was removed once the brain shell existed, since
    it did the exact same "see where the highlight sits" job for free, on
    every figure, with no picking needed). Any number of highlight_labels,
    in any combination (e.g. ["hippocampus", "amygdala"]).

    An empty highlight_labels draws whatever single fallback region
    extract_figure_cache actually wrote (typically "cortex", if no regions
    were passed to it either) — so you always get at least one field-
    colored surface, never just the bare shells. Returns {"out_path",
    "regions_drawn"}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    scalp_path = os.path.join(cache_dir, "scalp_surface.npz")
    scalp = (_load_surface(scalp_path, smooth_iters, smooth_pass_band)
            if (not no_shell and os.path.isfile(scalp_path)) else None)
    brain_path = os.path.join(cache_dir, "brain_surface.npz")
    brain = (_load_surface(brain_path, smooth_iters, smooth_pass_band)
            if (not no_brain and os.path.isfile(brain_path)) else None)

    def _load_region(label):
        surf_path = os.path.join(cache_dir, f"{label}_surface.npz")
        if not os.path.isfile(surf_path):
            return None
        vol_path = os.path.join(cache_dir, f"{label}_volume.npz")
        return (label, _load_surface(surf_path, smooth_iters, smooth_pass_band),
               _load_volume(vol_path) if os.path.isfile(vol_path) else None)

    highlights = [r for r in (_load_region(lbl) for lbl in highlight_labels) if r is not None]

    if not highlights:
        # scalp/brain are the two fixed anatomical shells, never a
        # candidate for "the one field-colored region" fallback below.
        found = [f[:-len("_surface.npz")] for f in os.listdir(cache_dir)
                if f.endswith("_surface.npz") and f not in ("scalp_surface.npz", "brain_surface.npz")]
        if found:
            highlights = [_load_region(found[0])]

    electrodes = _load_electrodes(electrodes_csv, ch1, ch2)
    opts = dict(vmin=vmin, vmax=vmax, cmap=cmap, clip=clip, clip_offset=clip_offset,
               value_opacity=value_opacity, highlight_opacity=highlight_opacity,
               shell_opacity=shell_opacity, shell_color=shell_color,
               brain_opacity=brain_opacity, brain_color=brain_color,
               electrode_radius=electrode_radius, far_electrode_opacity=far_electrode_opacity,
               label_electrodes=label_electrodes, label_regions=label_regions,
               window_size=list(window_size), peels=peels, zoom=zoom)

    images = {v: _render_view(scalp, brain, highlights, electrodes, v, opts) for v in VIEWS}

    fig = plt.figure(figsize=(12, 4.4))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.02)
    for col, (name, img) in enumerate(images.items()):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(img)
        ax.set_title(name, fontsize=12, pad=6)
        ax.axis("off")

    cax = fig.add_subplot(grid[0, 3])
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap), cax=cax, orientation="vertical")
    cbar.set_label(r"$(\mathrm{V\ m^{-1}})$", fontsize=12)
    cbar.set_ticks(np.linspace(vmin, vmax, 6))

    if title:
        fig.suptitle(title, fontsize=13)

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    return {"out_path": out_path, "regions_drawn": [label for label, *_ in highlights]}


# ═════════════════════════════════════════════════════════════════════════════
# GUI entry point — one call for job_runner (extract + render together;
# combined they take ~15-45s on a real subject, too slow for a synchronous
# Dash callback but fine as a single background job per "Render Figure"
# click)
# ═════════════════════════════════════════════════════════════════════════════

FIELD_NAME = "max_TI"  # the TI envelope itself, not normE_ch1/ch2 (ordinary
# single-channel tDCS fields SimNIBS computes on the way there) or the
# vector fields E_ch1/E_ch2 -- see code/viz/SOP.md's own warning about this.
# Not exposed as a user-facing choice: every other stat in this project
# already standardizes on max_TI, and picking a different field here would
# silently produce a figure of the wrong physical quantity.


def available_regions(subject_id: str, project_dir: str = PROJECT_DIR) -> list[str]:
    """Existing mask labels for one subject (sub-{id}_label-{name}_mask.
    nii.gz) -- the pool a region picker offers, same source as Comparison's
    ROI/non-ROI/Extra Regions pickers."""
    import comparison_discovery as cx
    return sorted(cx.list_existing_names([subject_id]).keys())


def list_existing_meshes(subject_id: str, project_dir: str = PROJECT_DIR) -> list[dict]:
    """Every already-computed max_TI mesh for this subject that Render
    Figure could draw directly, WITHOUT recomputing anything — Compute TI /
    Run Comparison's own cached outputs (comparison/*_head_mesh.msh, one per
    ti_stats_{label}.json sidecar written alongside it) and every finished
    Run Pipeline exhaustive-search run's own winning-montage mesh
    (TIoptimization/<run>/{id}_tes_flex_opt_head_mesh.msh, montage read from
    that run's exhaustive_results.json — same file comparison_discovery.
    load_best_montage() already reads for the "Fill from Optimization
    Results" feature on the Comparison page).

    Returns [{"label", "msh_path", "source" ("comparison"|"TIoptimization"),
    "ch1_plus", "ch1_minus", "ch1_current_mA", "ch2_plus", "ch2_minus",
    "ch2_current_mA"}], comparison entries first. Doesn't know which CAP any
    of these electrode names came from — see find_matching_caps()."""
    import glob
    import json

    import comparison_discovery as cx

    sub_dir = os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}")
    out = []

    comparison_dir = os.path.join(sub_dir, "comparison")
    for sidecar_path in sorted(glob.glob(os.path.join(comparison_dir, "ti_stats_*.json"))):
        try:
            with open(sidecar_path) as f:
                data = json.load(f)
        except (OSError, ValueError):
            continue
        label = data.get("label")
        m = data.get("montage") or {}
        if not label or not all(k in m for k in ("ch1_plus", "ch1_minus", "ch2_plus", "ch2_minus")):
            continue
        msh_path = os.path.join(comparison_dir, f"{label}_head_mesh.msh")
        if not os.path.isfile(msh_path):
            continue
        out.append({
            "label": label, "msh_path": msh_path, "source": "comparison",
            "ch1_plus": m["ch1_plus"], "ch1_minus": m["ch1_minus"], "ch1_current_mA": m.get("ch1_current_mA"),
            "ch2_plus": m["ch2_plus"], "ch2_minus": m["ch2_minus"], "ch2_current_mA": m.get("ch2_current_mA"),
        })

    ti_dir = os.path.join(sub_dir, "TIoptimization")
    if os.path.isdir(ti_dir):
        for run_dir in sorted(os.listdir(ti_dir)):
            run_path = os.path.join(ti_dir, run_dir)
            msh_path = os.path.join(run_path, f"{subject_id}_tes_flex_opt_head_mesh.msh")
            results_path = os.path.join(run_path, "exhaustive_results.json")
            if not (os.path.isfile(msh_path) and os.path.isfile(results_path)):
                continue
            montage = cx.load_best_montage(results_path)
            if not montage:
                continue
            out.append({
                "label": run_dir, "msh_path": msh_path, "source": "TIoptimization",
                "ch1_plus": montage["ch1_plus"], "ch1_minus": montage["ch1_minus"],
                "ch1_current_mA": montage["ch1_current_mA"],
                "ch2_plus": montage["ch2_plus"], "ch2_minus": montage["ch2_minus"],
                "ch2_current_mA": montage["ch2_current_mA"],
            })

    return out


def find_matching_caps(subject_id: str, electrode_names: list[str], project_dir: str = PROJECT_DIR) -> list[str]:
    """Registered cap names for this subject whose own electrode set
    contains every name in electrode_names. An existing mesh's sidecar only
    ever records electrode NAMES, never which cap they came from, so this
    narrows down which cap's CSV to actually read sphere positions from for
    the figure. Usually resolves to exactly one; more than one is a real
    ambiguity (different cap layouts can share common 10-10 names at
    different coordinates) — the caller should let the user pick rather
    than silently guessing which one is right."""
    import cap_discovery as cd

    wanted = set(electrode_names)
    matches = []
    for c in cd.list_registered_caps(subject_id, project_dir):
        try:
            names = set(cd.registered_electrode_positions(c["path"])["names"])
        except Exception:
            continue
        if wanted <= names:
            matches.append(c["name"])
    return matches


def build_figure(
    subject_id: str, field_mesh_path: str, ch1: tuple[str, str], ch2: tuple[str, str],
    electrodes_csv: str, highlight_labels: list[str], out_path: str,
    vmax: float = 1.0, project_dir: str = PROJECT_DIR, cache_dir: str | None = None,
) -> dict:
    """Resolves every selected region label to this subject's own mask file
    (comparison_discovery.resolve_mask() -- a label with no mask for this
    subject is dropped, not an error, same opportunistic convention as
    Comparison's Extra Regions), extracts the figure cache, and renders.
    Every resolved region is drawn as a highlight; the whole-brain shell
    (extract_figure_cache()'s own, unconditional) already provides the
    "see where this sits in the brain" context, so there is no separate
    context region to pick here.

    cache_dir defaults to a folder next to out_path -- reusable for a later
    re-render of the SAME field_mesh_path with a DIFFERENT vmax or a
    DIFFERENT subset of the regions that were extracted this time (faster,
    skips the mesh reload+crop); it is NOT valid for a region that wasn't
    part of highlight_labels on the call that created it -- extract a fresh
    cache (a new cache_dir, or force by deleting the old one) if you need a
    region this one didn't extract.

    Returns {"success", "out_path"?, "regions_drawn"?, "field_range"?,
    "error"?} -- meant to run inside a job_runner background job, so any
    unexpected exception is left to propagate (job_runner catches it
    generically), this only turns EXPECTED failure modes (no region
    resolved, bad electrode name) into a clean error string."""
    import comparison_discovery as cx

    if cache_dir is None:
        cache_dir = os.path.splitext(out_path)[0] + "_cache"

    wanted = list(dict.fromkeys(highlight_labels or []))
    region_mask_paths = {}
    for label in wanted:
        mask_path = cx.resolve_mask(subject_id, label, project_dir)
        if mask_path:
            region_mask_paths[label] = mask_path
    if not region_mask_paths:
        return {"success": False,
               "error": f"none of the selected regions ({wanted}) have a mask for sub-{subject_id}"}

    head_mesh_path = os.path.join(get_m2m_path(subject_id, project_dir), f"{subject_id}.msh")
    if not os.path.isfile(head_mesh_path):
        return {"success": False, "error": f"head model mesh not found: {head_mesh_path}"}

    manifest = extract_figure_cache(field_mesh_path, head_mesh_path, FIELD_NAME,
                                    region_mask_paths, cache_dir)
    effective_highlights = [lbl for lbl in highlight_labels if lbl in region_mask_paths]
    result = render_figure(cache_dir, electrodes_csv, effective_highlights, out_path,
                           ch1=ch1, ch2=ch2, vmax=vmax)

    return {"success": True, "out_path": result["out_path"], "regions_drawn": result["regions_drawn"],
           "field_range": manifest["field_range"]}


# ═════════════════════════════════════════════════════════════════════════════
# Subprocess wrapper — VTK's off-screen Plotter() hangs indefinitely if
# constructed on a non-main thread (confirmed empirically on this Windows
# machine: importing pyvista from a background threading.Thread works fine,
# but Plotter() itself never returns). job_runner.start_local_job() always
# runs its target on a background thread, so build_figure() can never be
# called directly from a GUI callback's background job — it has to run in a
# separate PROCESS instead, whose own main thread is unaffected. Same
# subprocess-from-a-background-thread pattern already used everywhere for
# SCITAS calls in this project (ssh_run() etc.) — that's always been safe;
# only in-process VTK has the thread-affinity problem.
# ═════════════════════════════════════════════════════════════════════════════

def build_figure_subprocess(
    subject_id: str, field_mesh_path: str, ch1: tuple[str, str], ch2: tuple[str, str],
    electrodes_csv: str, highlight_labels: list[str], out_path: str,
    vmax: float = 1.0, project_dir: str = PROJECT_DIR, timeout: int = 600,
) -> dict:
    """Same call signature and return shape as build_figure() — pass this to
    job_runner.start_local_job() instead of build_figure() directly. Runs
    this same module as `python -m viz_discovery` in a fresh process (this
    GUI's own interpreter, sys.executable) and reads back its JSON result."""
    import json
    import subprocess
    import sys
    import tempfile

    result_fd, result_path = tempfile.mkstemp(suffix=".json", prefix="viz_result_")
    os.close(result_fd)
    try:
        cmd = [sys.executable, "-m", "viz_discovery",
              "--subject", subject_id, "--field-mesh", field_mesh_path,
              "--ch1", ch1[0], ch1[1], "--ch2", ch2[0], ch2[1],
              "--electrodes", electrodes_csv, "--out", out_path,
              "--vmax", str(vmax), "--project-dir", project_dir,
              "--result-json", result_path]
        for label in (highlight_labels or []):
            cmd += ["--highlight", label]

        proc = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)),
                              capture_output=True, text=True, encoding="utf-8", errors="replace",
                              timeout=timeout)
        if os.path.isfile(result_path) and os.path.getsize(result_path) > 0:
            with open(result_path) as f:
                return json.load(f)
        tail = "\n".join((proc.stdout or "").splitlines()[-30:])
        err_tail = "\n".join((proc.stderr or "").splitlines()[-30:])
        return {"success": False,
               "error": f"render process exited {proc.returncode} with no result written.\n"
                        f"--- stdout (tail) ---\n{tail}\n--- stderr (tail) ---\n{err_tail}"}
    except subprocess.TimeoutExpired:
        return {"success": False, "error": f"render process timed out after {timeout}s"}
    finally:
        if os.path.isfile(result_path):
            os.remove(result_path)


def _cli_main():
    import argparse
    import json

    p = argparse.ArgumentParser(description="Subprocess entry point for build_figure_subprocess() — "
                                            "not meant to be run by hand, see that function's docstring.")
    p.add_argument("--subject", required=True)
    p.add_argument("--field-mesh", required=True)
    p.add_argument("--ch1", nargs=2, required=True, metavar=("PLUS", "MINUS"))
    p.add_argument("--ch2", nargs=2, required=True, metavar=("PLUS", "MINUS"))
    p.add_argument("--electrodes", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--project-dir", default=PROJECT_DIR)
    p.add_argument("--highlight", action="append", default=[])
    p.add_argument("--result-json", required=True)
    args = p.parse_args()

    try:
        result = build_figure(
            subject_id=args.subject, field_mesh_path=args.field_mesh,
            ch1=tuple(args.ch1), ch2=tuple(args.ch2), electrodes_csv=args.electrodes,
            highlight_labels=args.highlight, out_path=args.out,
            vmax=args.vmax, project_dir=args.project_dir,
        )
    except Exception as e:
        result = {"success": False, "error": str(e)}

    with open(args.result_json, "w") as f:
        json.dump(result, f)


# ═════════════════════════════════════════════════════════════════════════════
# Slice Viewer — 2D orthogonal (axial/coronal/sagittal) slices through the
# T1, scrubbable interactively, field amplitude as a color wash + ROI
# outline(s) on top. A different tool from Render Figure above (2D voxel
# slices vs. 3D mesh surfaces) for a different question ("where exactly,
# slice by slice" vs. "what does the whole structure look like").
#
# No VTK here at all, so none of Render Figure's thread/subprocess concerns
# apply — this is plain numpy + SimNIBS's own mesh-to-volume tool, safe to
# call directly from a job_runner background thread.
#
# The field is resampled onto the subject's own T1 grid via SimNIBS's
# built-in simnibs.transformations.interpolate_to_volume() (same tool the
# msh2nii CLI command wraps) — not custom voxelization. This project's own
# ROI mask files, however, live on a DIFFERENT, standardized 256^3 1mm grid
# (create_masks.py's own convention, verified empirically against a real
# mask — NOT the same grid as m2m_{id}/T1.nii.gz, which keeps the subject's
# native, usually-anisotropic acquisition resolution), so each requested
# mask is resampled onto the T1 grid (nearest-neighbor, to keep it binary)
# before use. Everything is then downsampled together to SLICE_VOXEL_MM —
# full T1 resolution is unnecessarily heavy for a "scroll through and get
# oriented" viewer; the static 3-view Render Figure above stays full detail
# for anything that needs to hold up as a real figure.
# ═════════════════════════════════════════════════════════════════════════════

SLICE_VOXEL_MM = 1.5  # downsample target -- still coarse (this view is for
# orientation/scrubbing, not a publication crop), but not the bottleneck it
# looks like: interpolate_to_volume() always runs at full T1 resolution
# regardless of this value (~20s fixed, dominates the whole build either
# way) -- only the downsample-and-composite step scales with it, and that's
# cheap even at this resolution (measured: same ~20s total as 3.0mm gave).
# The real cost is payload size (~13MB base64 at 1.5mm vs ~2MB at 3.0mm for
# a full head) -- fine for a local app with no real network to cross, but
# worth knowing if this ever needs to go lower for a slower machine/browser.

# One label + one bright, high-contrast outline color per highlighted
# region, cycled if more are picked than colors listed. The "plasma"
# colormap's own palette runs dark purple -> red/orange -> yellow, so it has
# no green or blue in it at all — leading with green/cyan/blue keeps the
# outline readable against the field colormap at every intensity, unlike a
# red/orange/yellow outline (this page's original choice), which visually
# blends into the colormap's own high end.
SLICE_ROI_COLORS = [(0, 230, 118), (0, 176, 255), (255, 214, 0), (255, 82, 82), (186, 104, 200)]


def _resample_mask_to_grid(mask_path: str, target_shape, target_affine):
    """Nearest-neighbor-resamples a binary mask onto a different grid —
    needed because this project's ROI masks live on a standardized 256^3
    1mm grid, not the T1's own native grid interpolate_to_volume() targets
    (see this section's module-level comment). order=0 (nearest) keeps the
    result strictly binary, no fractional voxels at the boundary."""
    import nibabel as nib
    import nibabel.processing as nibproc

    img = nib.load(mask_path)
    target = nib.Nifti1Image(np.zeros(target_shape, dtype=np.uint8), target_affine)
    resampled = nibproc.resample_from_to(img, target, order=0)
    return np.asarray(resampled.dataobj) > 0


def build_slice_volumes(
    subject_id: str, field_mesh_path: str, highlight_labels: list[str],
    project_dir: str = PROJECT_DIR, voxel_mm: float = SLICE_VOXEL_MM,
) -> dict:
    """Resamples max_TI onto the subject's own T1 grid, aligns each
    requested ROI mask onto that same grid, downsamples everything together
    to voxel_mm. Returns {"t1", "field": (nx,ny,nz) float32 arrays,
    "roi_masks": {label: (nx,ny,nz) bool}, "voxel_mm": [x,y,z],
    "field_range": {"min","max"}} — every array shares one grid/shape, so
    slicing any of them at the same index gives you the same anatomical
    plane. Raises on missing inputs (T1, the field itself) rather than
    returning a soft failure — meant to run inside a job_runner background
    job, which catches any exception generically."""
    import gc
    import shutil
    import tempfile

    import nibabel as nib
    from scipy.ndimage import zoom as ndi_zoom
    from simnibs import transformations
    from simnibs.mesh_tools import mesh_io

    import comparison_discovery as cx

    m2m_path = get_m2m_path(subject_id, project_dir)
    t1_path = os.path.join(m2m_path, "T1.nii.gz")
    if not os.path.isfile(t1_path):
        raise FileNotFoundError(f"T1 not found: {t1_path}")

    tmp_dir = tempfile.mkdtemp(prefix="slice_vol_")
    try:
        mesh = mesh_io.read_msh(field_mesh_path)
        mesh.elmdata = [ed for ed in mesh.elmdata if ed.field_name == FIELD_NAME]
        if not mesh.elmdata:
            raise ValueError(f"{FIELD_NAME} not in {field_mesh_path}")
        out_prefix = os.path.join(tmp_dir, "vol")
        # keep_tissues=[1, 2]: WM+GM only, matching Render Figure's own
        # "whole brain" scope — skull/scalp/CSF field values aren't
        # anatomically meaningful here and would just wash out the colormap.
        transformations.interpolate_to_volume(mesh, m2m_path, out_prefix, keep_tissues=[1, 2])
        del mesh
        gc.collect()

        field_img = nib.load(f"{out_prefix}_{FIELD_NAME}.nii.gz")
        t1_img = nib.load(t1_path)
        # Canonicalize to (closest-to-)RAS voxel-array axis order/direction —
        # NOT optional. nibabel doesn't guarantee a NIfTI's raw array axis
        # order matches any anatomical convention; it depends on how that
        # subject's own scanner/reconstruction wrote the file. Confirmed
        # this varies subject to subject here: a per-subject dynamic axis
        # lookup (aff2axcodes(), no canonicalization) correctly labeled each
        # plane for one subject but produced a visibly rotated ("sideways")
        # image for a second real subject whose native storage order
        # differed. Canonicalizing first makes axis 0/1/2 reliably track
        # L->R / P->A / I->S for every subject, so PLANE_AXES below and the
        # slice-orientation transform in build_slice_frames() can both be
        # fixed constants instead of derived per subject.
        field_img = nib.as_closest_canonical(field_img)
        t1_img = nib.as_closest_canonical(t1_img)
        field_data = np.asarray(field_img.dataobj, dtype=np.float32)
        t1_data = np.asarray(t1_img.dataobj, dtype=np.float32)

        roi_masks = {}
        for label in highlight_labels:
            mask_path = cx.resolve_mask(subject_id, label, project_dir)
            if mask_path:
                roi_masks[label] = _resample_mask_to_grid(mask_path, field_data.shape, field_img.affine)

        zooms = field_img.header.get_zooms()[:3]
        factors = [float(z) / voxel_mm for z in zooms]

        t1_small = ndi_zoom(t1_data, factors, order=1, mode="nearest")
        field_small = ndi_zoom(field_data, factors, order=1, mode="nearest")
        roi_small = {label: ndi_zoom(mask.astype(np.float32), factors, order=0, mode="nearest") > 0.5
                    for label, mask in roi_masks.items()}

        nonzero = field_data[field_data > 0]
        return {
            "t1": t1_small, "field": field_small, "roi_masks": roi_small,
            "voxel_mm": [voxel_mm, voxel_mm, voxel_mm], "affine": field_img.affine,
            "field_range": {"min": float(nonzero.min()) if nonzero.size else 0.0,
                            "max": float(field_data.max())},
        }
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _composite_slice(t1_slice, field_slice, roi_slices: dict, vmax: float, cmap: str = "plasma",
                     roi_only: bool = False):
    """One 2D slice -> an RGB uint8 image: grayscale T1 background, field
    amplitude as a colormap wash (alpha ramped by field magnitude — near
    zero is fully transparent, so T1 shows through where the field barely
    reaches), and each ROI's boundary drawn as a solid-color outline (a
    1px-eroded ring, not a filled region — so the T1/field underneath stays
    visible inside the ROI too, not just around it).

    roi_only: restrict the colored wash to inside the selected region(s)
    only — everywhere else stays plain grayscale T1, even where the field
    is nonzero there too. Off by default (whole brain colored, original
    behavior); the whole-brain view is still one click away either way,
    this never removes it."""
    import matplotlib
    from scipy.ndimage import binary_erosion

    t1 = t1_slice.astype(np.float32)
    lo, hi = np.percentile(t1, [1, 99]) if t1.size else (0.0, 1.0)
    if hi <= lo:
        hi = lo + 1.0
    t1_norm = np.clip((t1 - lo) / (hi - lo), 0, 1)
    rgb = np.stack([t1_norm] * 3, axis=-1)

    if vmax > 0:
        field_norm = np.clip(field_slice / vmax, 0, 1)
        if roi_only and roi_slices:
            roi_union = np.zeros_like(field_norm, dtype=bool)
            for mask in roi_slices.values():
                roi_union |= mask
            field_norm = np.where(roi_union, field_norm, 0.0)
        colored = matplotlib.colormaps[cmap](field_norm)[..., :3]
        alpha = field_norm[..., None]
        rgb = rgb * (1 - alpha) + colored * alpha

    img = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)

    for i, (label, mask) in enumerate(roi_slices.items()):
        if not mask.any():
            continue
        outline = mask & ~binary_erosion(mask, iterations=1, border_value=0)
        color = SLICE_ROI_COLORS[i % len(SLICE_ROI_COLORS)]
        img[outline] = color

    return img


def _encode_png_b64(rgb: np.ndarray) -> str:
    import base64
    import io

    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgb, mode="RGB").save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode()


# Which array axis to slice along for each anatomical plane. ONLY valid
# because build_slice_volumes() canonicalizes both volumes to (closest-to-)
# RAS order first (nibabel.as_closest_canonical()) — axis 0 tracks L->R,
# axis 1 tracks P->A, axis 2 tracks I->S, for every subject, so this can be
# a fixed mapping instead of derived per subject. It used to be derived
# dynamically from each volume's own affine (nibabel.aff2axcodes()) without
# canonicalizing first — that got WHICH axis to slice right, but not how to
# orient the resulting 2D slice for display, so a subject whose native
# storage axis order/direction differed from the first one tested still
# came out visibly rotated ("sideways"). Canonicalizing first fixes both at
# once — see _oriented_slice() for the display-orientation half.
PLANE_AXES = {"Sagittal": 0, "Coronal": 1, "Axial": 2}


def _oriented_slice(volume: np.ndarray, axis: int, index: int) -> np.ndarray:
    """One 2D slice, reoriented for display. volume must already be
    canonicalized (RAS order — see build_slice_volumes()): slicing leaves
    the two remaining axes in their original (lower-index-first) order,
    which is never the right image layout — this transposes so the
    higher-index remaining axis becomes the row (vertical) and flips it so
    its anatomically "up" end (Superior for the two axial-adjacent planes,
    Anterior for Axial itself) ends up at the top of the image, then leaves
    the lower-index remaining axis as columns (horizontal). Same transpose
    + flip for all three planes once canonicalized — verified against real
    axial/coronal/sagittal renders for two different subjects."""
    sl = [slice(None)] * 3
    sl[axis] = index
    return volume[tuple(sl)].T[::-1, :]


def _build_colorbar_png(vmax: float, cmap: str = "plasma", vmin: float = 0.0) -> str:
    """A small standalone colorbar (base64 PNG) mapping the field colormap
    to V/m — the per-slice composites bake the colormap straight into
    pixels with no numeric legend of their own, so this is shown once,
    alongside the viewer, as the scale."""
    import base64
    import io

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.cm import ScalarMappable
    from matplotlib.colors import Normalize

    fig, ax = plt.subplots(figsize=(4.2, 0.55))
    fig.subplots_adjust(bottom=0.45, top=0.92, left=0.03, right=0.97)
    cbar = fig.colorbar(ScalarMappable(norm=Normalize(vmin, vmax), cmap=cmap), cax=ax, orientation="horizontal")
    cbar.set_label("V/m", fontsize=9)
    cbar.ax.tick_params(labelsize=8)

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, transparent=True)
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode()


def build_slice_frames(
    subject_id: str, field_mesh_path: str, highlight_labels: list[str],
    project_dir: str = PROJECT_DIR, voxel_mm: float = SLICE_VOXEL_MM,
    vmax: float | None = None, cmap: str = "plasma", roi_only: bool = False,
) -> dict:
    """Full pipeline: build_slice_volumes() + one composited PNG (base64)
    per slice, for all three planes. vmax defaults to the 99th percentile
    of nonzero field values in this subject's own volume (a fixed 0-1
    scale, as Render Figure uses, doesn't suit every montage's actual
    range) — pass one explicitly to match a specific comparison. roi_only:
    see _composite_slice()'s own docstring — restricts the color wash to
    inside the selected region(s), off (whole brain colored) by default.

    Returns {"success", "frames": {"Axial": [b64 PNG, ...], "Coronal":
    [...], "Sagittal": [...]}, "n_slices": {"Axial": int, ...},
    "mid_slice": {"Axial": int, ...}, "voxel_mm", "field_range",
    "colorbar_png" (b64, maps the field colormap to V/m — the slices
    themselves carry no numeric legend), "regions_drawn"} or
    {"success": False, "error"}."""
    vols = build_slice_volumes(subject_id, field_mesh_path, highlight_labels, project_dir, voxel_mm)
    t1, field, roi_masks = vols["t1"], vols["field"], vols["roi_masks"]

    if vmax is None:
        nonzero = field[field > 0]
        vmax = float(np.percentile(nonzero, 99)) if nonzero.size else 1.0

    frames, n_slices, mid_slice = {}, {}, {}
    for plane, axis in PLANE_AXES.items():
        n = t1.shape[axis]
        n_slices[plane] = n
        mid_slice[plane] = n // 2
        plane_frames = []
        for i in range(n):
            t1_slice = _oriented_slice(t1, axis, i)
            field_slice = _oriented_slice(field, axis, i)
            roi_slices = {label: _oriented_slice(mask, axis, i) for label, mask in roi_masks.items()}
            rgb = _composite_slice(t1_slice, field_slice, roi_slices, vmax, cmap, roi_only)
            plane_frames.append(_encode_png_b64(rgb))
        frames[plane] = plane_frames

    return {"success": True, "frames": frames, "n_slices": n_slices, "mid_slice": mid_slice,
           "voxel_mm": vols["voxel_mm"], "field_range": vols["field_range"],
           "colorbar_png": _build_colorbar_png(vmax, cmap),
           "regions_drawn": list(roi_masks.keys()), "vmax_used": vmax}


if __name__ == "__main__":
    _cli_main()
"""Step 2 -- run with this project's combined SimNIBS+viz environment
(bids_ti_gui_env) -- no separate venv needed, see code/viz/SOP.md section 1.

Renders the pial surface coloured by the TI field inside a translucent scalp
shell, from three orthogonal views, and composes them into one figure with a
shared vertical colorbar.

VTK's depth peeling is what makes the translucent shell render correctly from
every angle; that is the reason this is not a Gmsh script.
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pyvista as pv
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

# Right-Anterior-Superior subject space, so each view is one axis of the camera.
VIEWS = {
    "Axial":    dict(direction=(0, 0, 1),  viewup=(0, 1, 0)),   # from above
    "Coronal":  dict(direction=(0, 1, 0),  viewup=(0, 0, 1)),   # from the front
    "Sagittal": dict(direction=(1, 0, 0),  viewup=(0, 0, 1)),   # from the right
}
CH1_COLOR = "#00e5ff"
CH2_COLOR = "#00ff7f"


def load_surface(path):
    d = np.load(path)
    faces = d["faces"]
    vtk_faces = np.hstack([np.full((len(faces), 1), 3, np.int32), faces]).ravel()
    mesh = pv.PolyData(d["verts"].astype(np.float32), vtk_faces)
    if "values" in d:
        mesh.cell_data["TI"] = d["values"]
    return mesh


def load_volume(path):
    d = np.load(path)
    tets = d["tets"]
    cells = np.hstack([np.full((len(tets), 1), 4, np.int32), tets]).ravel()
    grid = pv.UnstructuredGrid(
        cells,
        np.full(len(tets), pv.CellType.TETRA, np.uint8),
        d["verts"].astype(np.float32),
    )
    grid.cell_data["TI"] = d["values"]
    return grid


def load_electrodes(path, ch1, ch2):
    rows = {}
    with open(path, newline="") as fh:
        for row in csv.reader(fh):
            if len(row) >= 5 and row[0].strip().lower() == "electrode":
                rows[row[4].strip()] = np.array([float(v) for v in row[1:4]])
    missing = [n for n in (*ch1, *ch2) if n not in rows]
    if missing:
        raise SystemExit(f"electrodes {missing} not in {path}; found {sorted(rows)}")
    return [(n, rows[n], CH1_COLOR) for n in ch1] + [(n, rows[n], CH2_COLOR) for n in ch2]


def trim_white(img, pad=8):
    """Crop the uniform background border so the panels sit tight together."""
    mask = (img < 250).any(axis=2)
    if not mask.any():
        return img
    rows, cols = np.where(mask.any(axis=1))[0], np.where(mask.any(axis=0))[0]
    r0, r1 = max(rows[0] - pad, 0), min(rows[-1] + pad + 1, img.shape[0])
    c0, c1 = max(cols[0] - pad, 0), min(cols[-1] + pad + 1, img.shape[1])
    return img[r0:r1, c0:c1]


def render_view(brain, scalp, roi, roi_volume, electrodes, view, args):
    plotter = pv.Plotter(off_screen=True, window_size=args.window_size)
    plotter.set_background("white")
    plotter.enable_depth_peeling(number_of_peels=args.peels, occlusion_ratio=0.0)

    # Framing comes from the whole brain in every mode, so the three panels stay
    # on a common scale even when the geometry is cut away.
    focal = np.array(brain.center)
    span = float(np.linalg.norm(np.array(brain.bounds[1::2]) - np.array(brain.bounds[::2])))
    direction = np.array(VIEWS[view]["direction"], float)

    clip_origin = None
    if args.clip:
        # Cut with the plane normal to the camera, so each panel looks straight
        # onto its own cut face.  The volume is what gets clipped -- the field
        # exposed on that face is the field *inside* the structure.
        anchor = roi_volume if roi_volume is not None else brain
        clip_origin = np.array(anchor.center) + direction * args.clip_offset
        kw = dict(normal=tuple(direction), origin=tuple(clip_origin))
        brain = brain.clip(**kw)
        scalp = scalp.clip(**kw) if scalp is not None else None
        if roi_volume is not None:
            roi = roi_volume.clip(**kw)
        elif roi is not None:
            roi = roi.clip(**kw)

    # Value-linked opacity: vmin is fully see-through, vmax fully solid.
    field_opacity = "linear" if args.value_opacity else 1.0

    if roi is None:
        plotter.add_mesh(brain, scalars="TI", cmap=args.cmap,
                         clim=(args.vmin, args.vmax), show_scalar_bar=False,
                         opacity=field_opacity,
                         smooth_shading=True, ambient=0.25, diffuse=0.75, specular=0.15)
    else:
        # The ROI is the subject of the figure, so the cortex drops to a
        # neutral translucent shell.  Colouring both would put the cortical
        # field between the camera and the structure we are trying to read.
        plotter.add_mesh(roi, scalars="TI", cmap=args.cmap,
                         clim=(args.vmin, args.vmax), show_scalar_bar=False,
                         opacity=field_opacity,
                         smooth_shading=True, ambient=0.35, diffuse=0.8, specular=0.2)
        plotter.add_mesh(brain, color=args.brain_color, opacity=args.brain_opacity,
                         smooth_shading=True, specular=0.0, diffuse=0.6)
    if scalp is not None:
        plotter.add_mesh(scalp, color=args.shell_color, opacity=args.shell_opacity,
                         smooth_shading=True, specular=0.0, diffuse=0.6)
    drawn = []
    for name, pos, color in electrodes:
        if clip_origin is not None:
            if float(np.dot(pos - clip_origin, direction)) > 0:
                continue  # sits in the half that was cut away
            opacity = 1.0  # nothing left in front of it to occlude it
        else:
            # An opaque sphere on the far side of the head shows straight
            # through the translucent shell and reads as an electrode floating
            # inside the brain.  Ghost those, so they stay legible as "behind".
            near = float(np.dot(pos - focal, direction)) > 0
            if not near and args.far_electrode_opacity <= 0:
                continue
            opacity = 1.0 if near else args.far_electrode_opacity
        plotter.add_mesh(pv.Sphere(radius=args.electrode_radius, center=pos),
                         color=color, specular=0.4, smooth_shading=True,
                         opacity=opacity)
        drawn.append((name, pos))

    if args.label_electrodes:
        # Label only electrodes that were actually drawn -- a label with no
        # sphere under it is worse than no label.  The translucent scalp still
        # writes depth, so VTK's own occlusion test would cull all of them and
        # the near-side test has to be ours.  Once a cut has removed the near
        # half, nothing occludes what is left, so everything drawn gets a label.
        labelled = drawn if clip_origin is not None else [
            (n, p) for n, p in drawn if float(np.dot(p - focal, direction)) > 0]
        if labelled:
            plotter.add_point_labels(
                np.array([p for _, p in labelled]), [n for n, _ in labelled],
                font_size=args.window_size[1] // 45, text_color="black",
                shape=None, always_visible=True, show_points=False)

    plotter.camera_position = [
        tuple(focal + direction * span),
        tuple(focal),
        VIEWS[view]["viewup"],
    ]
    plotter.reset_camera()
    plotter.camera.zoom(args.zoom)

    img = plotter.screenshot(return_img=True)
    plotter.close()
    return trim_white(img)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cache-dir", required=True, type=Path,
                   help="directory written by extract_surfaces.py")
    p.add_argument("--electrodes", required=True, type=Path)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--ch1", nargs=2, default=["AR", "AL"], metavar=("PLUS", "MINUS"))
    p.add_argument("--ch2", nargs=2, default=["PR", "PL"], metavar=("PLUS", "MINUS"))
    p.add_argument("--vmin", type=float, default=0.0)
    p.add_argument("--vmax", type=float, default=1.0)
    p.add_argument("--cmap", default="plasma")
    p.add_argument("--cbar-label", default=r"$(\mathrm{V\ m^{-1}})$")
    p.add_argument("--title", default=None)
    p.add_argument("--clip", action="store_true",
                   help="cut each view with a plane normal to its camera, "
                        "exposing the field inside the ROI")
    p.add_argument("--clip-offset", type=float, default=0.0,
                   help="mm to shift the cut plane along the viewing axis")
    p.add_argument("--value-opacity", action="store_true",
                   help="map opacity to the field: vmin transparent, vmax solid")
    p.add_argument("--no-roi", action="store_true",
                   help="ignore roi_surface.npz and colour the cortex instead")
    p.add_argument("--roi-smooth", type=int, default=20,
                   help="Taubin smoothing passes on the ROI surface (0 = raw)")
    p.add_argument("--brain-opacity", type=float, default=0.10,
                   help="opacity of the neutral cortex shell when an ROI is drawn")
    p.add_argument("--brain-color", default="#b0b0b0")
    p.add_argument("--shell-opacity", type=float, default=0.12)
    p.add_argument("--shell-color", default="#9a9a9a")
    p.add_argument("--no-shell", action="store_true")
    p.add_argument("--electrode-radius", type=float, default=7.0)
    p.add_argument("--far-electrode-opacity", type=float, default=0.25,
                   help="opacity of electrodes on the far side of the head (0 = hide)")
    p.add_argument("--label-electrodes", action="store_true")
    p.add_argument("--window-size", type=int, nargs=2, default=[1100, 1100])
    p.add_argument("--peels", type=int, default=12)
    p.add_argument("--zoom", type=float, default=1.25)
    p.add_argument("--dpi", type=int, default=300)
    args = p.parse_args()

    brain = load_surface(args.cache_dir / "brain_surface.npz")
    scalp = None if args.no_shell else load_surface(args.cache_dir / "scalp_surface.npz")

    roi_path = args.cache_dir / "roi_surface.npz"
    roi = None
    if roi_path.exists() and not args.no_roi:
        roi = load_surface(roi_path)
        if args.roi_smooth:
            # The ROI boundary is a staircase of tet faces; Taubin smooths it
            # without the shrinkage a plain Laplacian would cause.
            roi = roi.smooth_taubin(n_iter=args.roi_smooth, pass_band=0.05)
        print(f"roi: {roi.n_cells} faces, "
              f"TI {roi.cell_data['TI'].min():.3f}-{roi.cell_data['TI'].max():.3f}")

    vol_path = args.cache_dir / "roi_volume.npz"
    roi_volume = None
    if roi is not None and vol_path.exists():
        roi_volume = load_volume(vol_path)

    electrodes = load_electrodes(args.electrodes, args.ch1, args.ch2)

    images = {v: render_view(brain, scalp, roi, roi_volume, electrodes, v, args)
              for v in VIEWS}

    fig = plt.figure(figsize=(12, 4.4))
    grid = fig.add_gridspec(1, 4, width_ratios=[1, 1, 1, 0.055], wspace=0.02)
    for col, (name, img) in enumerate(images.items()):
        ax = fig.add_subplot(grid[0, col])
        ax.imshow(img)
        ax.set_title(name, fontsize=12, pad=6)
        ax.axis("off")

    cax = fig.add_subplot(grid[0, 3])
    cbar = fig.colorbar(
        ScalarMappable(norm=Normalize(args.vmin, args.vmax), cmap=args.cmap),
        cax=cax, orientation="vertical")
    cbar.set_label(args.cbar_label, fontsize=12)
    cbar.set_ticks(np.linspace(args.vmin, args.vmax, 6))
    if args.title:
        fig.suptitle(args.title, fontsize=13)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(args.out, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
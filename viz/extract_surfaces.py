"""Step 1 -- run with this project's combined SimNIBS+viz environment
(bids_ti_gui_env) -- no separate venv needed, see code/viz/SOP.md section 1.

Pulls the pial surface (carrying max_TI) out of the simulation mesh and the
scalp shell out of the head model, and writes them as small .npz files.  Only
numpy is needed to read them back, so step 2 (render_ti.py) never has to
import simnibs at all, even though both steps happen to share one
environment in this project now.
"""
import argparse
from pathlib import Path

import nibabel as nib
import numpy as np
import simnibs

# SimNIBS 4 charm tags: 1 = WM, 2 = GM, 5 = scalp, 7/8 = compact/spongy bone.
# Surface tags are 1000 + the volume tag.
GM_TAG = 2
SCALP_SURF_TAG = 1005
SKULL_SURF_TAGS = (1007, 1008)

# SAMSEG labels in m2m_*/segmentation/labeling.nii.gz.  Striatum proper is
# caudate + putamen + accumbens, both hemispheres.
STRIATUM_LABELS = (11, 12, 26, 50, 51, 58)


def boundary_faces(tets):
    """Boundary triangles of a tet mesh, plus the index of each one's parent tet.

    A face shared by two tets is interior; a face appearing exactly once is on
    the boundary.  Winding is left alone -- PyVista recomputes normals.
    """
    faces = np.concatenate([
        tets[:, [0, 2, 1]],
        tets[:, [0, 1, 3]],
        tets[:, [0, 3, 2]],
        tets[:, [1, 2, 3]],
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


def compact(nodes, faces):
    """Drop unreferenced nodes and reindex the faces onto what is left."""
    used, inverse = np.unique(faces, return_inverse=True)
    return nodes[used], inverse.reshape(faces.shape).astype(np.int32)


def extract_brain(mesh_path, field_name, out_path):
    msh = simnibs.read_msh(str(mesh_path))
    names = [f.field_name for f in msh.elmdata]
    if field_name not in names:
        raise SystemExit(f"field {field_name!r} not in {mesh_path}; available: {names}")

    # Crop to grey matter alone.  Taking the boundary of WM+GM would expose WM
    # faces where the brainstem is cut, and those carry the WM hotspot values
    # (~3 V/m) rather than any cortical field.
    brain = msh.crop_mesh(tags=[GM_TAG])
    values = [f for f in brain.elmdata if f.field_name == field_name][0].value
    if values.ndim > 1:
        # a vector field (E_ch1 etc.) -- plot its magnitude
        values = np.linalg.norm(values, axis=1)

    tets = brain.elm.node_number_list[:, :4] - 1  # SimNIBS node ids are 1-based
    faces, parent = boundary_faces(tets)
    verts, faces = compact(brain.nodes.node_coord, faces)

    np.savez_compressed(
        out_path,
        verts=verts.astype(np.float32),
        faces=faces,
        values=values[parent].astype(np.float32),
        field_name=field_name,
    )
    print(f"{out_path.name}: {len(verts)} verts, {len(faces)} faces, "
          f"{field_name} range {values[parent].min():.3f}-{values[parent].max():.3f}")


def extract_shell(mesh_path, tags, out_path):
    msh = simnibs.read_msh(str(mesh_path))
    shell = msh.crop_mesh(tags=list(tags))
    tris = shell.elm.node_number_list[:, :3] - 1
    verts, faces = compact(shell.nodes.node_coord, tris)
    np.savez_compressed(out_path, verts=verts.astype(np.float32), faces=faces)
    print(f"{out_path.name}: {len(verts)} verts, {len(faces)} faces")


def roi_elements(msh, labeling_path, labels):
    """Element ids of the tets whose centroid falls inside the labelled ROI.

    The ROI has to come from the segmentation rather than from mesh tags:
    charm gives subcortical grey no tag of its own, so striatum tets come back
    tagged 1 *and* 2.  Cropping on tags=[2] would quietly cut away the part of
    the structure that landed in WM-tagged elements.
    """
    lab = nib.load(str(labeling_path))
    vol = np.asarray(lab.dataobj)
    world_to_vox = np.linalg.inv(lab.affine)

    is_tet = msh.elm.elm_type == 4
    tets = msh.elm.node_number_list[is_tet][:, :4] - 1
    centroids = msh.nodes.node_coord[tets].mean(axis=1)

    vox = np.rint(world_to_vox[:3, :3] @ centroids.T + world_to_vox[:3, [3]]).astype(int).T
    inside = np.all((vox >= 0) & (vox < np.array(vol.shape)), axis=1)
    at_centroid = np.zeros(len(centroids), dtype=vol.dtype)
    at_centroid[inside] = vol[vox[inside, 0], vox[inside, 1], vox[inside, 2]]

    return msh.elm.elm_number[is_tet][np.isin(at_centroid, labels)]


def extract_roi(mesh_path, field_name, labeling_path, labels, name, out_path):
    msh = simnibs.read_msh(str(mesh_path))
    elements = roi_elements(msh, labeling_path, labels)
    if len(elements) == 0:
        raise SystemExit(f"no tets fell inside labels {labels}")

    roi = msh.crop_mesh(elements=elements)
    values = [f for f in roi.elmdata if f.field_name == field_name][0].value
    if values.ndim > 1:
        values = np.linalg.norm(values, axis=1)

    tets = roi.elm.node_number_list[:, :4] - 1
    faces, parent = boundary_faces(tets)
    verts, faces = compact(roi.nodes.node_coord, faces)

    np.savez_compressed(
        out_path,
        verts=verts.astype(np.float32),
        faces=faces,
        values=values[parent].astype(np.float32),
        field_name=field_name,
        roi_name=name,
    )

    # The tets as well as their boundary: a cut plane through the surface alone
    # would only expose the inside of a hollow shell, not the field within the
    # structure.  Clipping needs the volume.
    vol_verts, vol_tets = compact(roi.nodes.node_coord, tets)
    np.savez_compressed(
        out_path.with_name("roi_volume.npz"),
        verts=vol_verts.astype(np.float32),
        tets=vol_tets,
        values=values.astype(np.float32),
        field_name=field_name,
        roi_name=name,
    )
    print(f"{out_path.name}: {name}, {len(elements)} tets -> {len(faces)} faces, "
          f"{field_name} volume mean {values.mean():.3f}, max {values.max():.3f}")
    print(f"roi_volume.npz: {len(vol_tets)} tets, {len(vol_verts)} verts")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--field-mesh", required=True, type=Path)
    p.add_argument("--head-mesh", required=True, type=Path)
    p.add_argument("--field", default="max_TI")
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--skull", action="store_true",
                   help="also export the skull surface (7/8) as a second shell")
    p.add_argument("--labeling", type=Path,
                   help="m2m_<ID>/segmentation/labeling.nii.gz; enables --roi-labels")
    p.add_argument("--roi-labels", type=int, nargs="+", default=list(STRIATUM_LABELS),
                   help="SAMSEG labels forming the ROI (default: striatum)")
    p.add_argument("--roi-name", default="striatum")
    args = p.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    extract_brain(args.field_mesh, args.field, args.out_dir / "brain_surface.npz")
    extract_shell(args.head_mesh, [SCALP_SURF_TAG], args.out_dir / "scalp_surface.npz")
    if args.skull:
        extract_shell(args.head_mesh, list(SKULL_SURF_TAGS), args.out_dir / "skull_surface.npz")
    if args.labeling:
        extract_roi(args.field_mesh, args.field, args.labeling, args.roi_labels,
                    args.roi_name, args.out_dir / "roi_surface.npz")


if __name__ == "__main__":
    main()
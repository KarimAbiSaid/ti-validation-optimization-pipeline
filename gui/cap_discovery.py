"""
cap_discovery.py — cap listing, registration-status checking, and the
registration action itself for the GUI's Cap Registration phase.

Pure Python, no Dash import. Reimplements register_caps.py's core
warp+scalp-project logic directly (rather than importing register_subject())
because that module keeps its project root in a mutable module-level global
(PROJECT) — fine for a one-shot CLI script, not safe to share across
concurrent Dash callbacks. Also register_caps.py's own read_cap_csv() uses
csv.DictReader, which assumes a header row; SimNIBS's own built-in cap CSVs
(simnibs/resources/ElectrodeCaps_MNI/*.csv) have none — read_cap_csv() here
detects both.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

from common import PROJECT_DIR, get_m2m_path, discover_subjects  # noqa: F401 (re-exported)


# ═════════════════════════════════════════════════════════════════════════════
# Cap listing
# ═════════════════════════════════════════════════════════════════════════════

def list_project_caps(project_dir: str = PROJECT_DIR) -> list[dict]:
    """MNI-space cap CSVs living in code/pipeline/configs/caps/ (this
    project's own, e.g. BioSemi32_MNE.csv)."""
    caps_dir = os.path.join(project_dir, "code", "pipeline", "configs", "caps")
    out = []
    if os.path.isdir(caps_dir):
        for fname in sorted(os.listdir(caps_dir)):
            if fname.endswith(".csv"):
                out.append({"name": fname[:-4], "path": os.path.join(caps_dir, fname), "source": "project"})
    return out


def list_builtin_caps() -> list[dict]:
    """MNI-space cap CSVs bundled with the SimNIBS install itself."""
    import simnibs
    d = Path(simnibs.__file__).parent / "resources" / "ElectrodeCaps_MNI"
    out = []
    if d.is_dir():
        for f in sorted(d.glob("*.csv")):
            if f.stem == "Fiducials":
                continue  # anatomical landmarks, not an electrode cap
            out.append({"name": f.stem, "path": str(f), "source": "simnibs"})
    return out


def list_available_caps(project_dir: str = PROJECT_DIR) -> list[dict]:
    """Project caps + SimNIBS built-in caps. Does not include arbitrary
    custom paths — those are validated on demand via validate_cap_path()."""
    return list_project_caps(project_dir) + list_builtin_caps()


def cap_stem(cap_path: str) -> str:
    """The cap's identity for filenames/registration lookups — matches
    register_caps.py's own convention (Path(cap_csv).stem)."""
    return Path(cap_path).stem


# ═════════════════════════════════════════════════════════════════════════════
# Cap CSV reading — handles both header and headerless formats
# ═════════════════════════════════════════════════════════════════════════════

def read_cap_csv(path: str) -> dict:
    """Returns {"types": [...], "names": [...], "coords": (N,3) ndarray}.

    Two formats, auto-detected:
    - Headerless, fixed positions Type,x,y,z,label (SimNIBS's own built-in
      caps, and this project's register_caps.py output): first row is
      already an Electrode/ReferenceElectrode data row.
    - Header row present: columns resolved BY NAME rather than assumed
      position, so alternate schemas still work — e.g. some external
      validation datasets use "#,id,x,y,z" (index/label/x/y/z, no Type
      column at all; Type then defaults to "Electrode" for every row)."""
    import numpy as np

    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    if not rows:
        raise ValueError(f"Empty cap file: {path}")

    if rows[0][0].strip().lower() in ("electrode", "referenceelectrode"):
        data_rows = rows
        col = {"type": 0, "x": 1, "y": 2, "z": 3, "label": 4}
    else:
        header = [h.strip().lower() for h in rows[0]]
        data_rows = rows[1:]

        def find(names):
            for n in names:
                if n in header:
                    return header.index(n)
            return None

        col = {
            "type": find(["type"]),
            "x": find(["x"]),
            "y": find(["y"]),
            "z": find(["z"]),
            "label": find(["label", "name", "id", "electrode", "channel", "ch_name"]),
        }
        missing = [k for k in ("x", "y", "z", "label") if col[k] is None]
        if missing:
            raise ValueError(f"{path}: couldn't find column(s) {missing} in header {rows[0]}")

    required_idx = [v for k, v in col.items() if k != "type" and v is not None]
    types, names, coords = [], [], []
    for row in data_rows:
        if len(row) <= max(required_idx):
            continue
        t = row[col["type"]].strip() if col["type"] is not None else "Electrode"
        types.append(t)
        coords.append([float(row[col["x"]]), float(row[col["y"]]), float(row[col["z"]])])
        names.append(row[col["label"]].strip())

    if not names:
        raise ValueError(f"No electrode rows found in {path}")
    return {"types": types, "names": names, "coords": np.array(coords)}


def validate_cap_path(path: str) -> dict:
    """For the custom-path input: {"exists", "readable", "n_electrodes", "error"}."""
    if not path or not os.path.isfile(path):
        return {"exists": False, "readable": False, "n_electrodes": None, "error": "file not found"}
    try:
        cap = read_cap_csv(path)
    except Exception as e:
        return {"exists": True, "readable": False, "n_electrodes": None, "error": str(e)}
    return {"exists": True, "readable": True, "n_electrodes": len(cap["names"]), "error": None}


# ═════════════════════════════════════════════════════════════════════════════
# Registration status
# ═════════════════════════════════════════════════════════════════════════════

def registered_cap_path(subject_id: str, cap_path: str, project_dir: str = PROJECT_DIR) -> str:
    return os.path.join(get_m2m_path(subject_id, project_dir), "eeg_positions", f"{cap_stem(cap_path)}.csv")


def cap_registration_matrix(subject_ids: list[str], cap_path: str, project_dir: str = PROJECT_DIR) -> dict:
    """{subject_id: {"ready", "registered", "registered_path", "missing"}}.
    ready = prerequisites present (m2m folder + subject .msh) — the same
    requirement the registration step itself enforces.
    registered = that cap has already been registered for this subject."""
    out = {}
    for sid in subject_ids:
        m2m_path = get_m2m_path(sid, project_dir)
        msh_path = os.path.join(m2m_path, f"{sid}.msh")
        missing = []
        if not os.path.isdir(m2m_path):
            missing.append("m2m folder")
        if not os.path.isfile(msh_path):
            missing.append(f"{sid}.msh (head mesh)")

        reg_path = registered_cap_path(sid, cap_path, project_dir)
        out[sid] = {
            "ready": not missing,
            "registered": os.path.isfile(reg_path),
            "registered_path": reg_path,
            "missing": missing,
        }
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Registration action
# ═════════════════════════════════════════════════════════════════════════════

def register_cap(subject_id: str, cap_path: str, project_dir: str = PROJECT_DIR) -> dict:
    """Warp cap_path's MNI-space electrodes into subject_id's space via
    charm's nonlinear transform, snap each onto the scalp surface (nearest
    scalp-surface-tag-1005 mesh node — the true skin boundary, not the
    scalp volume's tag-5 tetrahedra, which also include interior nodes),
    write m2m_{id}/eeg_positions/{cap_name}.csv. Same technique as
    register_caps.py's register_subject(), reimplemented as a plain
    function (no shared module-level state — see module docstring).
    Returns {"subject_id", "success", "error", "out_path", "n_electrodes"}."""
    m2m_path = get_m2m_path(subject_id, project_dir)
    msh_path = os.path.join(m2m_path, f"{subject_id}.msh")

    if not os.path.isdir(m2m_path):
        return {"subject_id": subject_id, "success": False, "error": f"m2m not found: {m2m_path}",
                "out_path": None, "n_electrodes": None}
    if not os.path.isfile(msh_path):
        return {"subject_id": subject_id, "success": False, "error": f"mesh not found: {msh_path}",
                "out_path": None, "n_electrodes": None}

    try:
        import numpy as np
        from simnibs import mesh_io
        from simnibs.utils.transformations import mni2subject_coords

        cap = read_cap_csv(cap_path)
        coords_subj = mni2subject_coords(cap["coords"], m2m_path)

        mesh = mesh_io.read_msh(msh_path)
        scalp = mesh.crop_mesh(tags=[1005])
        nodes = scalp.nodes.node_coord
        projected = np.array([nodes[np.argmin(np.linalg.norm(nodes - c, axis=1))] for c in coords_subj])

        out_path = registered_cap_path(subject_id, cap_path, project_dir)
        _write_registered_csv(out_path, cap["types"], cap["names"], projected)

        return {"subject_id": subject_id, "success": True, "error": None,
                "out_path": out_path, "n_electrodes": len(cap["names"])}
    except Exception as e:
        return {"subject_id": subject_id, "success": False, "error": str(e),
                "out_path": None, "n_electrodes": None}


def _write_registered_csv(out_path, types, names, coords) -> None:
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Type", "x", "y", "z", "label"])
        for t, name, coord in zip(types, names, coords):
            w.writerow([t, f"{coord[0]:.3f}", f"{coord[1]:.3f}", f"{coord[2]:.3f}", name])


def adopt_subject_space_cap(subject_id: str, cap_path: str, project_dir: str = PROJECT_DIR) -> dict:
    """For a cap CSV that's ALREADY in subject space (e.g. digitized
    positions from an actual recording, or a previously-registered file
    someone kept elsewhere) — no MNI warp, no scalp projection, just read +
    validate + copy into m2m_{id}/eeg_positions/{cap_name}.csv. Only makes
    sense for ONE specific subject: subject-space coordinates aren't
    transferable to a different subject's anatomy the way an MNI template is.
    Doesn't require the subject's .msh (no scalp projection happens)."""
    m2m_path = get_m2m_path(subject_id, project_dir)
    if not os.path.isdir(m2m_path):
        return {"subject_id": subject_id, "success": False, "error": f"m2m not found: {m2m_path}",
                "out_path": None, "n_electrodes": None}
    try:
        cap = read_cap_csv(cap_path)
        out_path = registered_cap_path(subject_id, cap_path, project_dir)
        _write_registered_csv(out_path, cap["types"], cap["names"], cap["coords"])
        return {"subject_id": subject_id, "success": True, "error": None,
                "out_path": out_path, "n_electrodes": len(cap["names"])}
    except Exception as e:
        return {"subject_id": subject_id, "success": False, "error": str(e),
                "out_path": None, "n_electrodes": None}


def register_cap_for_subjects(subject_ids: list[str], cap_path: str, project_dir: str = PROJECT_DIR) -> list[dict]:
    """Register one cap across multiple subjects independently — one
    subject's failure doesn't stop the rest."""
    return [register_cap(sid, cap_path, project_dir) for sid in subject_ids]


# ═════════════════════════════════════════════════════════════════════════════
# Preview: scalp mesh + registered electrode positions
# ═════════════════════════════════════════════════════════════════════════════

def list_registered_caps(subject_id: str, project_dir: str = PROJECT_DIR) -> list[dict]:
    """Caps already registered for this subject — m2m_{id}/eeg_positions/*.csv
    (this includes charm's own default Okamoto cap, not just ones registered
    via register_cap())."""
    eeg_dir = os.path.join(get_m2m_path(subject_id, project_dir), "eeg_positions")
    out = []
    if os.path.isdir(eeg_dir):
        for fname in sorted(os.listdir(eeg_dir)):
            if fname.endswith(".csv"):
                out.append({"name": fname[:-4], "path": os.path.join(eeg_dir, fname)})
    return out


def scalp_mesh_data(subject_id: str, project_dir: str = PROJECT_DIR) -> dict:
    """Subject's scalp SURFACE (tag 1005 — the actual 2D skin boundary),
    decimated for responsive in-browser 3D rendering. Returns
    {"x", "y", "z", "i", "j", "k"} — vertex coordinates and triangle
    indices, Plotly Mesh3d's own input format.

    Tag 5 is the scalp VOLUME (tetrahedra, 4 nodes/element) — cropping to
    it and taking the first 3 columns of node_number_list (as if they were
    triangles) produces a messy, non-manifold surface with ~8x more
    elements than the real skin boundary. Tag 1005 is the proper triangular
    surface mesh (SimNIBS pads its node_number_list to 4 columns for
    triangles too, with the last column = -1 — just unused, not part of
    the element)."""
    import numpy as np
    from simnibs import mesh_io

    m2m_path = get_m2m_path(subject_id, project_dir)
    msh_path = os.path.join(m2m_path, f"{subject_id}.msh")
    mesh = mesh_io.read_msh(msh_path)
    scalp = mesh.crop_mesh(tags=[1005])

    verts = scalp.nodes.node_coord
    faces = scalp.elm.node_number_list[:, :3] - 1  # 1-indexed -> 0-indexed triangles

    # Decimate by keeping every Nth face — cheap and topology-safe (unlike
    # vertex subsampling, which would orphan face references). The real
    # surface (~67k triangles) still renders cleanly at a higher target
    # than the old broken volume-face extraction did.
    target_faces = 16000
    if len(faces) > target_faces:
        stride = max(1, len(faces) // target_faces)
        faces = faces[::stride]

    # Drop now-unreferenced vertices instead of shipping the whole original
    # vertex array (e.g. 153k verts for ~5k decimated faces) to the browser.
    used = np.unique(faces)
    remap = np.full(len(verts), -1, dtype=np.int64)
    remap[used] = np.arange(len(used))
    verts = verts[used]
    faces = remap[faces]

    return {
        "x": verts[:, 0], "y": verts[:, 1], "z": verts[:, 2],
        "i": faces[:, 0], "j": faces[:, 1], "k": faces[:, 2],
    }


def registered_electrode_positions(cap_registered_path: str) -> dict:
    """Read an already-registered per-subject cap CSV (subject space) —
    {"types", "names", "coords"}, same shape as read_cap_csv()'s return."""
    return read_cap_csv(cap_registered_path)

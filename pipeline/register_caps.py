#!/usr/bin/env python3
"""
register_caps.py

For each subject: transforms a cap CSV from MNI space to subject space
using the charm warp, projects electrodes onto the scalp surface, and
writes a registered per-subject CSV to m2m_{id}/eeg_positions/.

Must run inside the SimNIBS Apptainer container (uses simnibs Python API).

Usage:
    python register_caps.py [--cap PATH] [subject_ids ...]

    --cap PATH   MNI-space cap CSV (default: configs/caps/BioSemi32_MNE.csv)
    subject_ids  Optional subject filter (default: all subjects in SUBJECTS)

Example (inside container):
    python /mnt/BIDS_TI_Toolbox/code/pipeline/register_caps.py \\
        --cap /mnt/BIDS_TI_Toolbox/code/pipeline/configs/caps/BioSemi32_MNE.csv \\
        002 012 035
"""

import sys
import csv
import argparse
import numpy as np
from pathlib import Path

SUBJECTS = ["025", "41Y01", "002", "012", "035", "055", "066", "077", "086", "093",
            "73T01", "73T02", "73T03", "73T04", "73T05",
            "73T06", "73T07", "73T09", "73T10", "73T11", "73T12", "73T14"]
# SCITAS default — override with --project-dir for local Windows runs
PROJECT  = Path("/mnt/BIDS_TI_Toolbox")
DEFAULT_CAP = PROJECT / "code/pipeline/configs/caps/BioSemi32_MNE.csv"


def read_cap_csv(path):
    rows = []
    with open(path) as f:
        for row in csv.DictReader(f):
            rows.append(row)
    ch_types  = [r["Type"]  for r in rows]
    ch_names  = [r["label"] for r in rows]
    coords    = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows])
    return ch_types, ch_names, coords


def project_to_scalp(coords_subj, msh_path):
    """Snap each coordinate to the nearest scalp (tag=5) node."""
    from simnibs import mesh_io
    mesh  = mesh_io.read_msh(str(msh_path))
    scalp = mesh.crop_mesh(tags=[5])
    nodes = scalp.nodes.node_coord          # (N, 3)
    projected = []
    for c in coords_subj:
        idx = np.argmin(np.linalg.norm(nodes - c, axis=1))
        projected.append(nodes[idx])
    return np.array(projected)


def register_subject(subj_id, cap_csv, cap_mni_coords, ch_types, ch_names):
    from simnibs.utils.transformations import mni2subject_coords

    m2m_path = PROJECT / f"derivatives/SimNIBS/sub-{subj_id}/m2m_{subj_id}"
    if not m2m_path.exists():
        print(f"  [skip] sub-{subj_id} — m2m not found: {m2m_path}")
        return

    msh_file = m2m_path / f"{subj_id}.msh"
    if not msh_file.exists():
        print(f"  [skip] sub-{subj_id} — mesh not found: {msh_file}")
        return

    cap_name = Path(cap_csv).stem
    out_csv  = m2m_path / "eeg_positions" / f"{cap_name}.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # MNI → subject space (uses charm's nonlinear warp)
    coords_subj = mni2subject_coords(cap_mni_coords, str(m2m_path))

    # Project onto scalp surface
    coords_proj = project_to_scalp(coords_subj, msh_file)

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["Type", "x", "y", "z", "label"])
        for t, name, coord in zip(ch_types, ch_names, coords_proj):
            w.writerow([t, f"{coord[0]:.3f}", f"{coord[1]:.3f}", f"{coord[2]:.3f}", name])

    print(f"  [ok] sub-{subj_id} → {out_csv.relative_to(PROJECT)}")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-dir", default=None,
                   help="Project root (default: /mnt/BIDS_TI_Toolbox). "
                        "Use D:/MINDS_Project_Karim/BIDS_TI_Toolbox for local Windows runs.")
    p.add_argument("--cap", default=None,
                   help="MNI-space cap CSV to register (default: <project-dir>/code/pipeline/configs/caps/BioSemi32_MNE.csv)")
    p.add_argument("subjects", nargs="*", help="Subject IDs (default: all)")
    args = p.parse_args()

    global PROJECT
    if args.project_dir:
        PROJECT = Path(args.project_dir)

    cap_csv = args.cap or str(PROJECT / "code/pipeline/configs/caps/BioSemi32_MNE.csv")
    subjects = args.subjects if args.subjects else SUBJECTS

    print(f"Registering cap: {cap_csv}")
    print(f"Subjects       : {subjects}")
    print()

    ch_types, ch_names, coords_mni = read_cap_csv(cap_csv)
    print(f"Cap: {len(ch_names)} electrodes — {', '.join(ch_names)}")
    print()

    for subj in subjects:
        register_subject(subj, cap_csv, coords_mni, ch_types, ch_names)

    print("\nDone.")


if __name__ == "__main__":
    main()

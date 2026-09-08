#!/usr/bin/env python3
"""
generate_leadfield_cli.py — Compute one leadfield (subject, cap, electrode
geometry) on SCITAS. SCITAS-side counterpart of the GUI's local
fem_discovery.generate_leadfield() — same TDCSLEADFIELD call, same
config.leadfield_tag()-keyed cache directory/filename convention, so a
leadfield built here and synced back down (Data Directory page, or SCITAS
Connection's manual sync) is picked up by leadfield_status()/
list_leadfields() exactly like a locally-generated one — no special-casing
needed on either side.

Reads the cap CSV from its canonical location, m2m_{id}/eeg_positions/
{cap_name}.csv — the same convention cap_discovery.registered_cap_path()
uses locally, so a registered cap only ever needs syncing up as that one
small file (handled by the GUI's run_leadfield_on_scitas() before this
script runs), never a separate upload step of its own.

Sibling of generate_lfsize_leadfield.py, generalized to take electrode
shape/dimensions/gel_thickness as CLI args instead of hardcoding this
project's production geometry, and computing into a scratch subdirectory
first (same safety property as the GUI's local generate_leadfield(): a
failed recompute never destroys a working cached leadfield already on disk).

Usage (inside the SimNIBS Apptainer container):
    python generate_leadfield_cli.py --subject 41Y07 --cap-name biosemi32 \
        --shape ellipse --dimensions 19.5 19.5 --gel-thickness 1.0 --cpus 8
"""
import argparse
import json
import os
import shutil
import sys
import time

_LEADFIELD_TISSUES = [1, 2, 3, 4, 5]


def build_leadfield(project_dir, subject, cap_name, shape, dimensions, gel_thickness, cpus, force=False):
    from simnibs.simulation.sim_struct import TDCSLEADFIELD

    from config import leadfield_tag

    sim_sub_dir = os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject}")
    m2m_path    = os.path.join(sim_sub_dir, f"m2m_{subject}")
    cap_csv     = os.path.join(m2m_path, "eeg_positions", f"{cap_name}.csv")
    if not os.path.isfile(cap_csv):
        sys.exit(f"ERROR sub-{subject}: cap CSV not found: {cap_csv}\n"
                 f"  Sync m2m_{subject}/eeg_positions/{cap_name}.csv from local before running this.")

    tag       = leadfield_tag(cap_name, shape, list(dimensions), gel_thickness)
    lf_dir    = os.path.join(sim_sub_dir, "leadfield_volume", tag)
    fname     = f"{subject}_leadfield_{cap_name}.hdf5"
    lf_hdf    = os.path.join(lf_dir, fname)
    lf_params = os.path.join(lf_dir, f"{subject}_leadfield_{cap_name}_params.json")

    current_params = {
        "shape": shape, "dimensions": list(dimensions), "gel_thickness": gel_thickness,
        "tissues": _LEADFIELD_TISSUES, "interpolation": None,
    }

    if not force and os.path.isfile(lf_hdf) and os.path.isfile(lf_params):
        with open(lf_params) as f:
            saved = json.load(f)
        if saved == current_params:
            print(f"  [SKIP] sub-{subject} {cap_name} -- leadfield already exists: {lf_hdf}")
            return lf_hdf

    os.makedirs(sim_sub_dir, exist_ok=True)
    scratch_dir = os.path.join(sim_sub_dir, "leadfield_volume", f"_regen_{tag}_{int(time.time())}")
    os.makedirs(scratch_dir, exist_ok=True)
    try:
        n_rows = sum(1 for _ in open(cap_csv))
        print(f"  sub-{subject} {cap_name}: computing leadfield ({n_rows} electrodes, cpus={cpus}) ...")
        t0 = time.time()

        lf_sess = TDCSLEADFIELD()
        lf_sess.subpath = m2m_path
        lf_sess.pathfem = scratch_dir
        lf_sess.eeg_cap = cap_csv

        lf_sess.electrode.shape      = shape
        lf_sess.electrode.dimensions = list(dimensions)
        lf_sess.electrode.thickness  = [gel_thickness]

        lf_sess.interpolation = None
        lf_sess.tissues       = _LEADFIELD_TISSUES

        lf_sess.run(cpus=cpus)

        scratch_hdf5 = os.path.join(scratch_dir, fname)
        if not os.path.isfile(scratch_hdf5):
            sys.exit(f"ERROR sub-{subject} {cap_name}: leadfield HDF5 not found after run: {scratch_hdf5} "
                     f"(existing cached leadfield, if any, was left untouched)")

        os.makedirs(lf_dir, exist_ok=True)
        if os.path.isfile(lf_hdf):
            os.remove(lf_hdf)
        shutil.move(scratch_hdf5, lf_hdf)
        with open(lf_params, "w") as f:
            json.dump(current_params, f, indent=2)

        elapsed = time.time() - t0
        print(f"  [DONE] sub-{subject} {cap_name} in {elapsed/60:.1f} min -> {lf_hdf}")
        return lf_hdf
    finally:
        shutil.rmtree(scratch_dir, ignore_errors=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--subject", required=True, help="Subject ID, e.g. 41Y07")
    p.add_argument("--cap-name", required=True, help="Registered cap CSV basename (no .csv)")
    p.add_argument("--shape", default="ellipse", choices=["ellipse", "rect"])
    p.add_argument("--dimensions", type=float, nargs=2, default=[19.5, 19.5], metavar=("W_MM", "H_MM"))
    p.add_argument("--gel-thickness", type=float, default=1.0)
    p.add_argument("--project-dir", default="/mnt/BIDS_TI_Toolbox",
                   help="Project root (container path, default: /mnt/BIDS_TI_Toolbox)")
    p.add_argument("--cpus", type=int, default=8)
    p.add_argument("--force", action="store_true", help="Recompute even if cached")
    args = p.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    build_leadfield(args.project_dir, args.subject, args.cap_name, args.shape,
                     args.dimensions, args.gel_thickness, args.cpus, force=args.force)


if __name__ == "__main__":
    main()
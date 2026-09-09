#!/usr/bin/env python3
"""
generate_lfsize_leadfield.py — Compute one nested-cap-size leadfield for the
leadfield-SIZE experiment, on SCITAS.

Sibling of generate_perturb_leadfield.py, but mirrors run_pipeline.py's own
production leadfield-building code exactly (single ELECTRODE object applied
uniformly to every cap position -- shape/dimensions/gel_thickness, NOT a
per-row list, and NOT a differently-sized reference electrode) so results
stay directly comparable to every other leadfield in this project.

Reads the cap CSV already written locally by
run_leadfield_size_experiment.py's build_nested_caps() --
m2m_{id}/eeg_positions/lfsize_{N}.csv -- this script does NOT generate
electrode positions, it only runs TDCSLEADFIELD against whatever cap CSV is
already there. Sync the CSV up to SCITAS before running this.

Usage (inside the SimNIBS Apptainer container):
    python generate_lfsize_leadfield.py --subject 41Y07 --cap-name lfsize_20 --cpus 8
    python generate_lfsize_leadfield.py --subject 41Y07 --cap-name lfsize_20 --cpus 8 --force
"""

import argparse
import json
import os
import shutil
import sys
import time

ELECTRODE_SHAPE      = 'ellipse'
ELECTRODE_DIMENSIONS = [19.5, 19.5]   # matches this project's production electrode geometry throughout
GEL_THICKNESS         = 1.0


def build_leadfield(project_dir, subject, cap_name, cpus, force=False):
    from simnibs.simulation.sim_struct import TDCSLEADFIELD
    from config import leadfield_tag

    sim_sub_dir = os.path.join(project_dir, 'derivatives', 'SimNIBS', f'sub-{subject}')
    m2m_path    = os.path.join(sim_sub_dir, f'm2m_{subject}')
    cap_csv     = os.path.join(m2m_path, 'eeg_positions', f'{cap_name}.csv')
    if not os.path.isfile(cap_csv):
        sys.exit(f'ERROR sub-{subject}: cap CSV not found: {cap_csv}\n'
                 f'  Sync m2m_{subject}/eeg_positions/{cap_name}.csv from local before running this.')

    tag       = leadfield_tag(cap_name, ELECTRODE_SHAPE, ELECTRODE_DIMENSIONS, GEL_THICKNESS)
    lf_dir    = os.path.join(sim_sub_dir, 'leadfield_volume', tag)
    lf_hdf    = os.path.join(lf_dir, f'{subject}_leadfield_{cap_name}.hdf5')
    lf_params = os.path.join(lf_dir, f'{subject}_leadfield_{cap_name}_params.json')

    # Same params dict shape as run_pipeline.py's own current_lf_params --
    # keeps this leadfield's cache indistinguishable from a production one
    # built with the same cap/geometry.
    current_params = {
        'shape':         ELECTRODE_SHAPE,
        'dimensions':    ELECTRODE_DIMENSIONS,
        'gel_thickness': GEL_THICKNESS,
        'tissues':       [1, 2, 3, 4, 5],
        'interpolation': None,
    }

    if not force and os.path.isfile(lf_hdf) and os.path.isfile(lf_params):
        with open(lf_params) as f:
            saved = json.load(f)
        if saved == current_params:
            print(f'  [SKIP] sub-{subject} {cap_name} -- leadfield already exists: {lf_hdf}')
            return lf_hdf
        print(f'  WARNING sub-{subject} {cap_name}: cached params mismatch -- recomputing')
        shutil.rmtree(lf_dir, ignore_errors=True)
    elif os.path.isdir(lf_dir):
        print(f'  sub-{subject} {cap_name}: incomplete leadfield directory found -- recomputing')
        shutil.rmtree(lf_dir, ignore_errors=True)

    os.makedirs(lf_dir, exist_ok=True)
    n_rows = sum(1 for _ in open(cap_csv))
    print(f'  sub-{subject} {cap_name}: computing leadfield ({n_rows} electrodes, cpus={cpus}) ...')
    t0 = time.time()

    lf_sess = TDCSLEADFIELD()
    lf_sess.subpath  = m2m_path
    lf_sess.pathfem  = lf_dir
    lf_sess.eeg_cap  = cap_csv

    lf_sess.electrode.shape      = ELECTRODE_SHAPE
    lf_sess.electrode.dimensions = ELECTRODE_DIMENSIONS
    lf_sess.electrode.thickness  = [GEL_THICKNESS]

    lf_sess.interpolation = None
    lf_sess.tissues       = [1, 2, 3, 4, 5]  # volume leadfield, all tissues -- WM/GM filtered later in analysis

    lf_sess.run(cpus=cpus)

    if not os.path.isfile(lf_hdf):
        sys.exit(f'ERROR sub-{subject} {cap_name}: leadfield HDF5 not found after run: {lf_hdf}')
    with open(lf_params, 'w') as f:
        json.dump(current_params, f, indent=2)

    elapsed = time.time() - t0
    print(f'  [DONE] sub-{subject} {cap_name} in {elapsed/60:.1f} min -> {lf_hdf}')
    return lf_hdf


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--subject', required=True, help='Subject ID, e.g. 41Y07')
    p.add_argument('--cap-name', required=True, help='Cap CSV basename (no .csv), e.g. lfsize_20')
    p.add_argument('--project-dir', default='/mnt/BIDS_TI_Toolbox',
                   help='Project root (container path, default: /mnt/BIDS_TI_Toolbox)')
    p.add_argument('--cpus', type=int, default=8)
    p.add_argument('--force', action='store_true', help='Recompute even if cached')
    args = p.parse_args()

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    build_leadfield(args.project_dir, args.subject, args.cap_name, args.cpus, force=args.force)


if __name__ == '__main__':
    main()

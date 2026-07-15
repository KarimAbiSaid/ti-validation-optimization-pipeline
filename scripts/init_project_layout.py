#!/usr/bin/env python3
"""
init_project_layout.py — creates the external project-data folder structure
this toolkit expects (see ../README.md's "External Data Layout" section), if
it doesn't already exist. Each folder gets a small _LAYOUT_INFO.json sidecar
describing its purpose, so a fresh project directory is self-documenting on
disk, not just in the README.

Does NOT create per-subject folders (rawdata/sub-{id}/, derivatives/SimNIBS/
sub-{id}/) — those come from your own MRI data, or from running Head
Modeling / the rest of the pipeline.

Safe to re-run any time: only creates missing folders/sidecars, never
touches anything else already in the project directory.

Usage:
    python init_project_layout.py [--project-dir PATH]

    --project-dir   Defaults to $BIDS_TI_PROJECT_DIR (same env var
                     gui/common.py reads).
"""
from __future__ import annotations

import argparse
import json
import os

LAYOUT = {
    "rawdata": {
        "description": "Your subject MRI data (BIDS-like). Not part of the code repo.",
        "expected_contents": "sub-{id}/anat/sub-{id}_T1w.nii.gz (required), "
                             "sub-{id}_T2w.nii.gz (optional — improves charm segmentation quality)",
    },
    "derivatives": {
        "description": "Generated outputs — everything here is produced by this toolkit "
                       "itself. Not part of the code repo.",
        "expected_contents": "SimNIBS/ (required), freesurfer/ (optional)",
    },
    "derivatives/SimNIBS": {
        "description": "Per-subject SimNIBS pipeline outputs, created starting from "
                       "Head Modeling (charm) onward.",
        "expected_contents": "sub-{id}/m2m_{id}/, roi/, eeg_positions/, leadfield_volume/, "
                             "comparison/ — all created automatically as you use the GUI/pipeline, "
                             "one sub-{id}/ folder per subject.",
    },
    "derivatives/freesurfer": {
        "description": "Optional FreeSurfer recon-all output — only needed for the FreeSurfer "
                       "atlas source (not yet usable per create_masks.py).",
        "expected_contents": "sub-{id}/mri/aparc+aseg.mgz",
    },
}

SIDECAR_NAME = "_LAYOUT_INFO.json"


def ensure_layout(project_dir: str) -> None:
    for rel_path, info in LAYOUT.items():
        abs_path = os.path.join(project_dir, *rel_path.split("/"))
        created = not os.path.isdir(abs_path)
        os.makedirs(abs_path, exist_ok=True)

        with open(os.path.join(abs_path, SIDECAR_NAME), "w", encoding="utf-8") as f:
            json.dump(info, f, indent=2, ensure_ascii=False)

        print(f"{'[created]' if created else '[exists] '} {abs_path}")

    print(f"\nDone. Project data directory: {project_dir}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--project-dir", default=os.environ.get("BIDS_TI_PROJECT_DIR"),
                   help="Project root (default: $BIDS_TI_PROJECT_DIR)")
    args = p.parse_args()

    if not args.project_dir:
        p.error("--project-dir not given and BIDS_TI_PROJECT_DIR isn't set")

    ensure_layout(args.project_dir)


if __name__ == "__main__":
    main()

"""
common.py — shared, phase-agnostic discovery logic for the GUI: project
paths, subject scanning, sys.path setup for importing code/pipeline/*
directly. No Dash import. Used by discovery.py (masks), cap_discovery.py
(caps), and any later phase's *_discovery.py.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

# PROJECT_DIR: external data (rawdata/, derivatives/, subject MRIs) — NOT
# part of this repo. Three ways to set it, in priority order:
#   1. BIDS_TI_PROJECT_DIR env var (advanced/per-machine override)
#   2. data_settings.json, next to this file — written by the Data Directory
#      settings page (see pages/data_directory.py). Takes effect on restart:
#      PROJECT_DIR is a module-level constant read once at import time, and
#      every discovery module's functions default to it at their own def
#      time — changing the file while the GUI is already running has no
#      effect until it's restarted.
#   3. fallback_manual_dir below.
_DATA_SETTINGS_PATH = Path(__file__).resolve().parent / "data_settings.json"
fallback_manual_dir = "D:/MINDS_Project_Karim/BIDS_TI_Toolbox"


def _load_data_dir_setting() -> str | None:
    if _DATA_SETTINGS_PATH.is_file():
        try:
            with open(_DATA_SETTINGS_PATH) as f:
                return json.load(f).get("data_dir") or None
        except (OSError, json.JSONDecodeError):
            return None
    return None


def save_data_dir_setting(data_dir: str) -> None:
    """Persists the chosen data directory. Takes effect on GUI restart —
    see the priority-order note above."""
    with open(_DATA_SETTINGS_PATH, "w") as f:
        json.dump({"data_dir": data_dir}, f, indent=2)


PROJECT_DIR = os.environ.get("BIDS_TI_PROJECT_DIR") or _load_data_dir_setting() or fallback_manual_dir

DATA_SUBDIRS = ["rawdata", "derivatives/SimNIBS", "derivatives/freesurfer"]


def setup_data_dir_skeleton(data_dir: str) -> list[str]:
    """Creates the expected top-level folders under data_dir (idempotent —
    safe to call on an already-set-up directory). Returns the list of
    absolute paths created/confirmed. Purely local filesystem, no cluster
    interaction."""
    created = []
    for sub in DATA_SUBDIRS:
        path = os.path.join(data_dir, *sub.split("/"))
        os.makedirs(path, exist_ok=True)
        created.append(path)
    return created

# RESOURCES_DIR: generic, non-personalized shared resources (atlases, common
# cap layouts) that DO live inside this repo (code/resources/) — resolved
# relative to this file, not PROJECT_DIR, since they travel with the code.
RESOURCES_DIR = str(Path(__file__).resolve().parent.parent / "resources")

_PIPELINE_DIR = str(Path(__file__).resolve().parent.parent / "pipeline")
if _PIPELINE_DIR not in sys.path:
    sys.path.insert(0, _PIPELINE_DIR)


def get_m2m_path(subject_id: str, project_dir: str = PROJECT_DIR) -> str:
    return os.path.join(project_dir, "derivatives", "SimNIBS", f"sub-{subject_id}", f"m2m_{subject_id}")


def discover_subjects(project_dir: str = PROJECT_DIR) -> list[dict]:
    """Scan rawdata/ and derivatives/SimNIBS/ for subjects.

    Returns [{"subject_id", "has_rawdata", "has_m2m", "m2m_path"}], sorted by
    subject_id. A subject can appear from either location (or both) — e.g. a
    subject with only rawdata/ needs charm run before anything subject-space
    (masks, caps, FEM) is possible.
    """
    subjects: dict[str, dict] = {}

    def _entry(sid):
        return subjects.setdefault(sid, {
            "subject_id": sid, "has_rawdata": False, "has_m2m": False, "m2m_path": None,
        })

    rawdata_dir = os.path.join(project_dir, "rawdata")
    if os.path.isdir(rawdata_dir):
        for name in sorted(os.listdir(rawdata_dir)):
            if name.startswith("sub-") and os.path.isdir(os.path.join(rawdata_dir, name)):
                _entry(name[len("sub-"):])["has_rawdata"] = True

    simnibs_dir = os.path.join(project_dir, "derivatives", "SimNIBS")
    if os.path.isdir(simnibs_dir):
        for name in sorted(os.listdir(simnibs_dir)):
            if name.startswith("sub-") and os.path.isdir(os.path.join(simnibs_dir, name)):
                sid = name[len("sub-"):]
                m2m_path = get_m2m_path(sid, project_dir)
                if os.path.isdir(m2m_path):
                    e = _entry(sid)
                    e["has_m2m"] = True
                    e["m2m_path"] = m2m_path

    return sorted(subjects.values(), key=lambda s: s["subject_id"])


def nifti_shape(path: str) -> tuple:
    """Cheap header-only read — no voxel data loaded."""
    import nibabel as nib
    return nib.load(path).shape

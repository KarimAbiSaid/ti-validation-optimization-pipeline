"""
common.py — shared, phase-agnostic discovery logic for the GUI: project
paths, subject scanning, sys.path setup for importing code/pipeline/*
directly. No Dash import. Used by discovery.py (masks), cap_discovery.py
(caps), and any later phase's *_discovery.py.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

fallback_manual_dir = "D:/MINDS_Project_Karim/BIDS_TI_Toolbox"
PROJECT_DIR = os.environ.get("BIDS_TI_PROJECT_DIR", fallback_manual_dir)


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

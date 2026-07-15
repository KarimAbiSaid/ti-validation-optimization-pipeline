#!/usr/bin/env python3
"""
generate_configs.py — Single source of truth for all TI pipeline configs.

Edit SUBJECTS, ROIS, GOALS, and the shared settings blocks below, then run:

    python generate_configs.py            # generate all configs + subject_configs.sh
    python generate_configs.py --dry-run  # preview without writing
    python generate_configs.py --force    # overwrite existing JSON files
    python generate_configs.py --subjects 025 41Y01  # only these subjects

Outputs
-------
  configs/sub-{id}_roi-{roi}_ex_{goal}.json   (one per combination)
  subject_configs.sh                           (sourced by submit_multiroi_scitas.sh)
"""

import argparse
import json
import math
from pathlib import Path

# ==============================================================================
# EDIT HERE — everything below is derived automatically
# ==============================================================================

SUBJECTS = [
    # "025",
    # "41Y01",
    # "002",
    # "012",
    # "035",
    # "055",
    # "066",
    # "077",
    # "086",
    # "093",
    "73T01",
    "73T02",
    "73T03",
    "73T04",
    "73T05",
    "73T06",
    "73T07",
    "73T09",
    "73T10",
    "73T11",
    "73T12",
    "73T14",
]


ROIS = {
    "hippo_r_phg": {
        "bna_labels": {
            "PhG_R":          116,
            "rHippocampus_R": 216,
            "cHippocampus_R": 218,
        },
    },
}

# Legacy FreeSurfer-based ROIs (aseg.auto.mgz labels) — kept for reference.
# ROIS_LEGACY = {
#     "striatum": {
#         "labels": {
#             "Left-Caudate":  11,
#             "Right-Caudate": 50,
#             "Left-Putamen":  12,
#             "Right-Putamen": 51,
#         },
#     },
#     "hippocampus": {
#         "labels": {
#             "Left-Hippocampus":  17,
#             "Right-Hippocampus": 53,
#         },
#     },
#     "NAcc": {
#         "labels": {
#             "Left-Accumbens-area":  26,
#             "Right-Accumbens-area": 58,
#         },
#     },
#     "globuspallidus": {
#         "labels": {
#             "Left-Pallidum":  13,
#             "Right-Pallidum": 52,
#         },
#         "note_key": "_note_GP",
#         "note_val": (
#             "FreeSurfer Pallidum (13/52) covers the full Globus Pallidus (GPe + GPi). "
#             "For GPi-specific targeting, provide a NIfTI mask from a basal ganglia atlas "
#             "(e.g. ATAG, CIT168, DISTAL via Lead-DBS) and switch to method='NIfTI'."
#         ),
#     },
#     "STN": {
#         "labels": {
#             "Left-VentralDC":  28,
#             "Right-VentralDC": 60,
#         },
#         "note_key": "_note_STN",
#         "note_val": (
#             "The STN is not a standard FreeSurfer aseg label. VentralDC (28/60) is used as "
#             "the closest available proxy — it includes the STN, substantia nigra, and red nucleus. "
#             "For precise STN targeting, replace this with a NIfTI mask from a DBS atlas "
#             "(e.g. DISTAL via Lead-DBS) and switch to method='NIfTI'."
#         ),
#     },
# }

GOALS = ["mean", "focality"]

# ==============================================================================
# BRAINNETOME ATLAS (BNA) — alternative ROI source
# ==============================================================================
# When a ROI uses "bna_labels" instead of "labels", Section 1 warps
# BN_Atlas_246_1mm.nii.gz to subject space using charm's Conform2MNI warp field
# and extracts the integer labels listed here.  The warped atlas is cached per
# subject at derivatives/SimNIBS/sub-{id}/roi/sub-{id}_BNA_atlas_subjectspace.nii.gz.
#
# Key differences from FreeSurfer-based ROIs:
#   - No recon-all needed (charm's warp field is sufficient)
#   - STN is NOT in BNA; keep using FreeSurfer VentralDC (28/60) proxy via aseg
#   - GP is a single structure (221/222), same as FreeSurfer Pallidum (no GPe/GPi split)
#   - Thalamus has 8 functional subregions — a new capability not available in aseg
#   - Striatum has dorsal/ventral subdivisions (useful for circuit-specific targeting)
#
# HOW TO SWITCH:
#   1. Set BNA_ATLAS_PATH below to the project copy of the atlas.
#   2. Replace ROIS / NON_ROI with BNA_ROIS / BNA_NON_ROI in the EDIT section above.
#   3. Run: python generate_configs.py --force
#   4. No recon-all needed — skip straight from charm to optimization.
# ==============================================================================

# Path to the BNA atlas inside the container.
BNA_ATLAS_PATH = "/mnt/BIDS_TI_Toolbox/BN_Atlas_246_1mm.nii.gz"

# BNA label reference (all from BNA_subregions.xlsx):
#   Odd label = left hemisphere, even = right hemisphere for each structure pair.
#
# Subcortical nuclei (labels 211–246):
#   Amygdala : medial L/R = 211/212,  lateral L/R = 213/214
#   Hippocampus : rostral L/R = 215/216,  caudal L/R = 217/218
#   Basal Ganglia: vCaudate L/R = 219/220,  GP L/R = 221/222,
#                  NAcc L/R = 223/224,  vmPutamen L/R = 225/226,
#                  dCaudate L/R = 227/228,  dlPutamen L/R = 229/230
#   Thalamus (8 subnuclei): mPFtha=231/232, mPMtha=233/234, Stha=235/236,
#                            rTtha=237/238, PPtha=239/240, Otha=241/242,
#                            cTtha=243/244, lPFtha=245/246
#
# Precentral gyrus / M1 (labels 53–68):
#   A4hf (head/face M1): L=53, R=54  |  A6cdl (lat.PMC): L=55, R=56
#   A4ul (upper limb M1): L=57, R=58  |  A4t (trunk M1): L=59, R=60
#   A4tl (tongue/larynx M1): L=61, R=62  |  A6cvl (med.PMC): L=63, R=64
#   A1/2/3ll (lower limb S1/M1): L=65, R=66  |  A4ll (lower limb M1): L=67, R=68

BNA_ROIS = {
    "striatum": {
        "bna_labels": {
            "Left-vCaudate":   219, "Right-vCaudate":   220,
            "Left-vmPutamen":  225, "Right-vmPutamen":  226,
            "Left-dCaudate":   227, "Right-dCaudate":   228,
            "Left-dlPutamen":  229, "Right-dlPutamen":  230,
        },
    },
    "striatum_dorsal": {
        "bna_labels": {
            "Left-dCaudate":  227, "Right-dCaudate":  228,
            "Left-dlPutamen": 229, "Right-dlPutamen": 230,
        },
    },
    "striatum_ventral": {
        "bna_labels": {
            "Left-vCaudate":  219, "Right-vCaudate":  220,
            "Left-vmPutamen": 225, "Right-vmPutamen": 226,
        },
    },
    "hippocampus": {
        "bna_labels": {
            "Left-rHippocampus":  215, "Right-rHippocampus": 216,
            "Left-cHippocampus":  217, "Right-cHippocampus": 218,
        },
    },
    "NAcc": {
        "bna_labels": {
            "Left-NAcc":  223, "Right-NAcc":  224,
        },
    },
    "globuspallidus": {
        "bna_labels": {
            "Left-GP":  221, "Right-GP":  222,
        },
    },
    "thalamus": {
        "bna_labels": {
            "Left-mPFtha":  231, "Right-mPFtha": 232,
            "Left-mPMtha":  233, "Right-mPMtha": 234,
            "Left-Stha":    235, "Right-Stha":   236,
            "Left-rTtha":   237, "Right-rTtha":  238,
            "Left-PPtha":   239, "Right-PPtha":  240,
            "Left-Otha":    241, "Right-Otha":   242,
            "Left-cTtha":   243, "Right-cTtha":  244,
            "Left-lPFtha":  245, "Right-lPFtha": 246,
        },
    },
    "thal_motor": {
        "bna_labels": {
            "Left-mPMtha": 233, "Right-mPMtha": 234,  # pre-motor thalamus
            "Left-Stha":   235, "Right-Stha":   236,  # sensory thalamus
        },
    },
    "thal_prefrontal": {
        "bna_labels": {
            "Left-mPFtha":  231, "Right-mPFtha": 232,
            "Left-lPFtha":  245, "Right-lPFtha": 246,
        },
    },
}

# BNA-based motor cortex non-ROI — full precentral gyrus (M1 + lateral PMC).
# Does NOT require recon-all; uses BNA labels from the warped atlas.
BNA_NON_ROI = {
    "name": "MotorCortex_BNA",
    "bna_labels": {
        "Left-A4hf":   53, "Right-A4hf":   54,   # primary M1 — head & face
        "Left-A6cdl":  55, "Right-A6cdl":  56,   # lateral PMC
        "Left-A4ul":   57, "Right-A4ul":   58,   # primary M1 — upper limb
        "Left-A4t":    59, "Right-A4t":    60,   # primary M1 — trunk
        "Left-A4tl":   61, "Right-A4tl":   62,   # primary M1 — tongue & larynx
        "Left-A6cvl":  63, "Right-A6cvl":  64,   # medial PMC
        "Left-A4ll":   67, "Right-A4ll":   68,   # primary M1 — lower limb
    },
}

# ==============================================================================
# END OF BNA SECTION
# ==============================================================================

# EEG caps.  key = short tag used in filename, value = path to CSV in subject space.
# None → use default Okamoto cap (already registered by charm in m2m_{id}/eeg_positions/).
# For custom caps, run register_caps.py first to generate per-subject registered CSVs.
CAPS = {
    # BioSemi32 is the only supported cap. register_caps.py must be run first
    # to produce per-subject registered CSVs in m2m_{id}/eeg_positions/.
    "biosemi32": "/mnt/BIDS_TI_Toolbox/code/pipeline/configs/caps/BioSemi32_MNE.csv",
}
ACTIVE_CAP = "biosemi32"

PROJECT_DIR = "/mnt/BIDS_TI_Toolbox"

FLAGS = {
    "run_recon_all":     False,
    "run_charm":         False,
    "run_roi_masks":     True,
    "run_optimization":  True,
    "run_simulation":    False,
    "run_analysis":      True,
    "run_visualization": True,
}

NON_ROI = {
    "name": "vmPFC_OFC_RSC",
    "bna_labels": {
        "vmPFC_L_1": 13, "vmPFC_R_1": 14,
        "vmPFC_L_2": 41, "vmPFC_R_2": 42,
        "vmPFC_L_3": 47, "vmPFC_R_3": 48,
        "vmPFC_L_4": 187,"vmPFC_R_4": 188,
        "OFC_L_1":  45,  "OFC_R_1":  46,
        "OFC_L_2":  49,  "OFC_R_2":  50,
        "RSC_L":   181,  "RSC_R":   182
    }
}

# Legacy FreeSurfer-based non-ROI (motor cortex) — kept for reference.
# NON_ROI_LEGACY = {
#     "_note": (
#         "Motor cortex non-ROI: M1 (precentral), lateral PMC (caudal middle frontal), "
#         "medial M1 + SMA (paracentral), S1 (postcentral). "
#         "All labels require aparc+aseg.mgz from full recon-all, not just charm."
#     ),
#     "name": "MotorCortex",
#     "labels": {
#         "Left-Precentral":          1024,
#         "Right-Precentral":         2024,
#         "Left-CaudalMiddleFrontal": 1003,
#         "Right-CaudalMiddleFrontal":2003,
#         "Left-Paracentral":         1017,
#         "Right-Paracentral":        2017,
#         "Left-Postcentral":         1022,
#         "Right-Postcentral":        2022,
#     },
#     "fallback_labels": {
#         "Left-Cerebral-Cortex":  3,
#         "Right-Cerebral-Cortex": 42,
#     },
# }

OPTIMIZER_BASE = {
    "postproc":               "max_TI",
    "focality_threshold":     [0.24, 0.3],
    "max_iterations":         200,
    "population_size":        13,
    "tolerance":              0.1,
    "mutation":               [0.01, 0.5],
    "recombination":          0.7,
    "n_multistart":           4,
    "cpus":                   8,
    "anisotropy_type":        "scalar",
    "min_electrode_distance": 5.0,
    "detailed_results":       False,
    "enable_mapping":         False,
    "use_exhaustive_search":  True,
    "max_non_roi_elements":   150000,
    "hard_roi_constraint":    True,
    "no_adjacent_electrodes": True,
    "non_roi_hard_constraint_groups": [
        {
            "name": "vmPFC_OFC",
            "bna_labels": {
                "vmPFC_L_1": 13, "vmPFC_R_1": 14,
                "vmPFC_L_2": 41, "vmPFC_R_2": 42,
                "vmPFC_L_3": 47, "vmPFC_R_3": 48,
                "vmPFC_L_4": 187,"vmPFC_R_4": 188,
                "OFC_L_1":  45,  "OFC_R_1":  46,
                "OFC_L_2":  49,  "OFC_R_2":  50
            },
            "max_mean_V_m": 0.24
        },
        {
            "name": "RSC",
            "bna_labels": {"RSC_L": 181, "RSC_R": 182},
            "max_mean_V_m": 0.24
        }
    ],
}

ELECTRODE = {
    "shape":             "ellipse",
    "dimensions":        [19.5, 19.5],
    "gel_thickness":     1.0,
    "current_mA":        2.0,
    "max_total_current": 4.0,
    "n_electrodes":      4,
}

# ==============================================================================
# Script logic — do not edit below
# ==============================================================================

SCRIPT_DIR   = Path(__file__).parent
CONFIGS_DIR  = SCRIPT_DIR / "configs"
SBATCH_ENV   = SCRIPT_DIR / "subject_configs.sh"


def config_filename(subject_id: str, roi_name: str, goal: str) -> str:
    return f"sub-{subject_id}_roi-{roi_name}_ex_{goal}.json"


def _roi_dict(roi_name: str, rois: dict) -> dict:
    """Build the roi JSON sub-dict from either a FreeSurfer or BNA ROI definition."""
    roi_def = rois[roi_name]
    if "bna_labels" in roi_def:
        return {"name": roi_name, "labels": {}, "bna_labels": roi_def["bna_labels"]}
    d = {"name": roi_name, "labels": roi_def["labels"]}
    if "note_key" in roi_def:
        d[roi_def["note_key"]] = roi_def["note_val"]
    return d


def build_config(subject_id: str, roi_name: str, goal: str, cap: str = "okamoto") -> dict:
    roi_def = ROIS[roi_name]
    cfg = {"subject_id": subject_id, "project_dir": PROJECT_DIR}
    # Pass bna_atlas_path if any ROI or non-ROI uses BNA labels
    uses_bna = ("bna_labels" in roi_def
                or "bna_labels" in NON_ROI)
    if uses_bna:
        cfg["bna_atlas_path"] = BNA_ATLAS_PATH
    if "note_key" in roi_def:
        cfg[roi_def["note_key"]] = roi_def["note_val"]
    cfg["flags"]      = FLAGS
    cfg["roi"]        = _roi_dict(roi_name, ROIS)
    cfg["non_roi"]    = NON_ROI
    cfg["optimizer"]  = {"goal": goal, **OPTIMIZER_BASE}
    cfg["electrode"]  = ELECTRODE
    cfg["simulation"] = {"simulate_mode": "optimized"}
    cap_csv = CAPS.get(cap)
    if cap_csv is not None:
        # Point to the per-subject registered CSV (written by register_caps.py).
        # The MNI-space CSV is only the source template; the pipeline needs the
        # subject-space version that charm's warp + scalp projection already applied.
        cap_stem = Path(cap_csv).stem  # e.g. "BioSemi32_MNE"
        cfg["cap_csv"] = (
            f"{PROJECT_DIR}/derivatives/SimNIBS/sub-{subject_id}"
            f"/m2m_{subject_id}/eeg_positions/{cap_stem}.csv"
        )
    return cfg


def _slurm_time_limit(n_combos: int) -> str:
    """Estimate SLURM time: 30 min leadfield + 5 min per ROI/goal combo, +1 h buffer."""
    minutes  = 30 + n_combos * 5
    hours    = math.ceil(minutes / 60) + 1
    return f"{hours:02d}:00:00"


def generate_configs(subjects: list[str], dry_run: bool, force: bool) -> tuple[int, int]:
    created = skipped = 0
    for subject_id in subjects:
        for roi_name in ROIS:
            for goal in GOALS:
                fname = config_filename(subject_id, roi_name, goal)
                fpath = CONFIGS_DIR / fname
                if fpath.exists() and not force:
                    print(f"  [skip]  {fname}  (already exists; use --force to overwrite)")
                    skipped += 1
                    continue
                cfg = build_config(subject_id, roi_name, goal, ACTIVE_CAP)
                if not dry_run:
                    fpath.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
                print(f"  {'[dry] ' if dry_run else '[ok]  '}{fname}")
                created += 1
    return created, skipped


def write_sbatch_env(subjects: list[str], dry_run: bool) -> None:
    """Write subject_configs.sh — sourced by submit_multiroi_scitas.sh."""
    base = "${BASE_CONFIG_DIR}"
    n_combos = len(ROIS) * len(GOALS)
    lines = [
        "# AUTO-GENERATED by generate_configs.py — do not edit by hand.",
        "# Re-run: python generate_configs.py",
        "",
        "declare -A SUBJECT_CONFIGS",
        "declare -A SUBJECT_TIMELIMITS",
        "",
    ]
    for subject_id in subjects:
        cfg_paths = [
            f"  {base}/{config_filename(subject_id, roi, goal)}"
            for roi in ROIS
            for goal in GOALS
        ]
        # first path on the same line as the variable, rest indented with backslash continuation
        first = cfg_paths[0].lstrip()
        rest  = " \\\n".join(cfg_paths[1:])
        lines.append(f'SUBJECT_CONFIGS["{subject_id}"]="{first} \\')
        lines.append(rest + '"')
        lines.append(f'SUBJECT_TIMELIMITS["{subject_id}"]="{_slurm_time_limit(n_combos)}"')
        lines.append("")

    content = "\n".join(lines)
    if not dry_run:
        with open(SBATCH_ENV, "w", encoding="utf-8", newline="\n") as f:
            f.write(content)
    print(f"\n  {'[dry] ' if dry_run else '[ok]  '}{SBATCH_ENV.name}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run",  action="store_true",
                        help="Preview without writing any files.")
    parser.add_argument("--force",    action="store_true",
                        help="Overwrite existing JSON config files.")
    parser.add_argument("--subjects", nargs="+", metavar="ID",
                        help="Only generate configs for these subject IDs.")
    args = parser.parse_args()

    subjects = args.subjects if args.subjects else SUBJECTS
    unknown  = [s for s in subjects if s not in SUBJECTS]
    if unknown:
        parser.error(f"Unknown subject(s): {unknown}. Add them to SUBJECTS in this script first.")

    n_total = len(subjects) * len(ROIS) * len(GOALS)
    print(f"generate_configs.py {'(DRY RUN) ' if args.dry_run else ''}"
          f"— {len(subjects)} subject(s) × {len(ROIS)} ROI(s) × {len(GOALS)} goal(s) = {n_total} configs")
    print(f"Configs dir : {CONFIGS_DIR}")
    print(f"Sbatch env  : {SBATCH_ENV}")
    print()

    print("--- JSON configs ---------------------------------------------------")
    created, skipped = generate_configs(subjects, dry_run=args.dry_run, force=args.force)

    print("\n--- subject_configs.sh ---------------------------------------------")
    write_sbatch_env(subjects, dry_run=args.dry_run)

    print(f"\nDone.  Created: {created}  |  Skipped: {skipped}")
    if skipped:
        print("       (use --force to overwrite existing files)")


if __name__ == "__main__":
    main()

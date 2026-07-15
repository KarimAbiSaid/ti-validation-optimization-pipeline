# BIDS_TI_Toolbox — Project Context

Paste this into a new chat to resume work without re-deriving context.
**Never write to the `Z:` drive — it is read-only.**

---

## What this project is

Temporal Interference (TI) stimulation pipeline built on SimNIBS 4.6. Two parallel workstreams:

1. **SCITAS pipeline** — runs on EPFL HPC cluster (scratch at `/scratch/$USER/BIDS_TI_Toolbox`,
   mounted as `/mnt/BIDS_TI_Toolbox` inside a SimNIBS Apptainer container). Optimizes 2-channel
   TI electrode montages per subject using an exhaustive cap search over a BioSemi32 cap.

2. **Local validation notebook** — runs on Windows (`D:\MINDS_Project_Karim\BIDS_TI_Toolbox`).
   Compares our computed TI E-fields against Pablo's reported values from a separate study.

---

## Current subjects

12 subjects, all from the MINDS/PLASMA 73T cohort:
```
73T01  73T02  73T03  73T04  73T05  73T06
73T07  73T09  73T10  73T11  73T12  73T14
```
(Note: 73T08 and 73T13 are absent — not part of the cohort.)

---

## Key directories (local Windows paths)

```
D:\MINDS_Project_Karim\BIDS_TI_Toolbox\
  code\pipeline\                          <- all scripts + notebooks
    validate_personal_vs_pablo_results.ipynb   <- main local analysis notebook
    compare_ti_montages.py                     <- module: _vol_mean_capped, SubjectResources
    config.py                                  <- PipelineConfig dataclasses
    generate_configs.py                        <- generates per-subject JSON configs for SCITAS
    run_pipeline.py                            <- main SCITAS pipeline driver
    configs\                                   <- per-subject JSON configs
      sub-{id}_roi-hippo_r_phg_ex_mean.json
      sub-{id}_roi-hippo_r_phg_ex_focality.json
  derivatives\SimNIBS\sub-{id}\
    m2m_{id}\                             <- charm head model + BNA atlas in subject space
    roi\
      sub-{id}_BNA_atlas_subjectspace.nii.gz  <- BNA atlas warped to subject space
    comparison\
      manual_fem_biosemi_64_reg\          <- FEM with Jurak-registered electrode positions
        ch1\  ch2\  ti_cache.npz
      manual_fem_pablo_exactpos\          <- FEM with Pablo's exact electrode positions
        ch1\  ch2\  ti_cache.npz
  Efields_Plasma_73_Individualized_Pablo.csv  <- Pablo's reported E-field values (source of truth)
```

Read-only reference data on `Z:` drive:
```
Z:\TI\SpatialNavigation\Personalized_Modelling\WP7.3\Optimized\{subject}\
  Results_summary.txt    <- electrode channel names and currents used by Pablo
  Electrodes.geo         <- Pablo's exact electrode positions in subject mesh space
```

---

## Validation notebook (`validate_personal_vs_pablo_results.ipynb`)

### Run order (standalone after shared helpers)
Steps 1 -> 6 -> 11 -> 12 -> 12b -> 13 -> 14

Steps 2-5, 7-10 can be skipped once run once (setup/registration/electrode offsets).

### Key data structures after running Steps 1 + 12
```python
ALL_SUBJECTS        # list of 12 subject IDs
reported_all        # {subject: {col: value}} -- from Pablo's CSV
tissue_results      # {variant_label: {subject: {roi: float}}}
                    #   keys: 'GM only [2]', 'WM+GM [1,2]', 'WM+GM+CSF [1,2,3]',
                    #          'pablo_exactpos WM+GM [1,2]'
```

### FEM variants
| Key in tissue_results | Folder | Electrode positions |
|---|---|---|
| `'WM+GM [1,2]'` | `manual_fem_biosemi_64_reg` | Jurak 2007 10-10 MNI -> `mni2subject_coords` -> nearest scalp node |
| `'pablo_exactpos WM+GM [1,2]'` | `manual_fem_pablo_exactpos` | Centroids from Pablo's `Electrodes.geo` |

### Electrode parameters (current, verified in FEM logs)
```python
ELECTRODE_DIMS      = [19.5, 19.5]   # mm, ellipse
ELECTRODE_THICKNESS = [4.]            # single rubber layer, no gel
```

### BNA ROI labels (Brainnetome Atlas, 246 regions, odd=L, even=R)
```python
HC_LABELS   = frozenset({216, 218})            # Hippocampus R (rostral + caudal)
EC_LABELS   = frozenset({116})                 # Entorhinal / Parahippocampal R
VMPFC_R_LBL = frozenset({14, 42, 46, 48, 50, 188})
VMPFC_L_LBL = frozenset({13, 41, 45, 47, 49, 187})
RSC_R_LBL   = frozenset({182})
RSC_L_LBL   = frozenset({181})
```

### Pablo's CSV column mapping
```python
REPORTED_KEY = {
    'HC':      'Hippocampus_R_u',
    'EC':      'Entorhinal cortex_R_u',
    'vmPFC_R': 'vmPFC_R_u',
    'vmPFC_L': 'vmPFC_L_u',
    'RSC_R':   'Retrosplenial cortex_R_u',
    'RSC_L':   'Retrosplenial cortex_L_u',
}
```

### ti_cache.npz contents
Built from `crop_mesh(tags=[1,2,3])` of ch1 FEM mesh.
Keys: `ti`, `elm_volumes`, `elm_tags`, `elm_lbl`
- `elm_tags`: tissue tag per element (1=WM, 2=GM, 3=CSF)
- `elm_lbl`: BNA atlas label per element (0 = unlabeled)
- Analysis filters to `elm_tags in {1,2}` (WM+GM only)

### Visualization (.geo files, Step 14)
Opens in Gmsh: `sub-{id}_TI_with_electrodes.geo` inside each variant folder.
Merge order matters -- brain mesh must come first to avoid ROI labels appearing on scalp:
```
Merge "sub-{id}_TI_view.msh";     <- brain (WM+GM), has ROI_label field
Merge "../sub-{id}_scalp.msh";    <- scalp surface, no field
```

---

## SCITAS pipeline

### Workflow for new subjects
```bash
# 1. Submit charm head modelling (skips subjects that already have .msh)
bash submit_charm_scitas.sh

# 2. Generate JSON configs (edit SUBJECTS in generate_configs.py if needed)
python generate_configs.py

# 3. Submit optimization jobs (one job per subject, all ROI/goal configs in sequence)
bash submit_multiroi_scitas.sh

# Or for specific subjects:
bash submit_multiroi_scitas.sh 73T06 73T07

# Or for specific goal only (e.g. mean first to build leadfield, then focality):
bash submit_multiroi_scitas.sh --goal mean
bash submit_multiroi_scitas.sh --goal focality
```

### Manual single-subject sbatch (bypasses .sh scripts)
```bash
# charm
sbatch --export=ALL,CHARM_SUBJECT=73T01 charm_scitas.sbatch

# pipeline (focality only, specific subjects)
for subj in 73T06 73T07 73T09 73T10 73T11 73T12 73T14; do
  sbatch \
    --job-name="ti_${subj}" \
    --time=03:00:00 \
    --export=ALL,PIPELINE_CONFIGS="/mnt/BIDS_TI_Toolbox/code/pipeline/configs/sub-${subj}_roi-hippo_r_phg_ex_focality.json" \
    simnibs_ti_pipeline.sbatch
done

# To force re-run of optimization (overwrite existing results), add to --export:
#   PIPELINE_EXTRA_ARGS="--force optimization"
```

### Config path convention
Config paths in `--export` must use the **in-container** path `/mnt/BIDS_TI_Toolbox/...`
(not the host `/scratch/$USER/...`), because `run_pipeline.py` runs inside the container.

### Current SCITAS ROI (generate_configs.py)
```python
ROIS = {
    "hippo_r_phg": {
        "bna_labels": {
            "PhG_R":          116,   # Parahippocampal/Entorhinal cortex R
            "rHippocampus_R": 216,
            "cHippocampus_R": 218,
        },
    },
}
GOALS = ["mean", "focality"]
```
Non-ROI: vmPFC + OFC + RSC (BNA labels 13,14,41,42,45,46,47,48,49,50,181,182,187,188).

### Script roles
| Script | Purpose |
|---|---|
| `charm_scitas.sbatch` | Runs charm head modelling on compute node |
| `submit_charm_scitas.sh` | Submits one charm job per new subject |
| `simnibs_ti_pipeline.sbatch` | Runs `run_pipeline.py` on compute node |
| `submit_multiroi_scitas.sh` | Submits one pipeline job per subject (all ROI/goal configs sequential, leadfield shared) |
| `recon_all_scitas.sbatch` | FreeSurfer recon-all (only needed for FreeSurfer-based labels, not BNA) |
| `submit_recon_all_scitas.sh` | Submits recon-all jobs (for non-73T subjects needing aparc+aseg) |
| `generate_configs.py` | Generates per-subject JSON configs + `subject_configs.sh` |
| `subject_configs.sh` | Auto-generated; sourced by `submit_multiroi_scitas.sh` |
| `config.py` | `PipelineConfig` dataclasses for `run_pipeline.py` |
| `run_pipeline.py` | Main SCITAS pipeline: ROI masks -> optimization -> analysis -> visualization |
| `compare_ti_montages.py` | Module used by notebook: `_vol_mean_capped`, `SubjectResources` |

---

## Key conventions

- **Volume-weighted, 99th-pct-capped mean** (`_vol_mean_capped`) is the standard metric everywhere.
- **TI envelope**: `TI.get_maxTI(ef1, ef2)` -- polarity-invariant, takes E-field vectors from two channels.
- **Tissue tags**: 1=WM, 2=GM, 3=CSF, 4=Skull, 5=Scalp.
- **BNA odd=Left, even=Right** for all paired structures.
- **Notebook is too large to Read/Edit directly** -- give paste instructions instead.
- **Z: drive is read-only** -- never write to it.

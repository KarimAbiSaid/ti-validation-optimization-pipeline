# BIDS TI Toolbox — GUI

A modular Dash-based GUI wrapping the BIDS TI Toolbox pipeline (`code/pipeline/`), built one phase at a time. Each phase is a separate page under `pages/`, so later phases slot in without rewriting earlier ones.

---

## Phase Status

The nav groups pages by category rather than by build order — each page's `dash.register_page()` call sets `category=`/`order=` (extra kwargs Dash just stores in the page registry; `app.py`'s `_build_nav()` groups by them):

| Category | Pages (nav order) |
| -------- | ------------------ |
| **Preprocessing** | Head Modeling → Mask Generation → Cap Registration |
| **Simulation** | FEM Validation → Comparison |
| **Optimization** | *(Config Generation, planned)* |

| Phase | Page | Status |
| ----- | ---- | ------ |
| 0 | Head Modeling (charm) | ✅ Done |
| 1 | Mask Generation | ✅ Done |
| 2 | Cap Registration | ✅ Done |
| 3 | Manual FEM for TI validation | ✅ Done |
| 4 | Comparison view (multiple FEM setups side by side) | ✅ Done |
| 5 | Config file generation | Planned |
| 6 | SCITAS/cluster + local job orchestration | Planned |

### Phase 0 — Head Modeling (charm)

- Select one subject (must have `rawdata/sub-{id}/anat/sub-{id}_T1w.nii.gz`; `_T2w.nii.gz` optional — improves segmentation quality), see T1/T2/m2m status, then run charm to build `m2m_{id}/`.
- charm is a real external command-line tool (`charm.exe`, resolved from the running interpreter's own `Scripts/` folder — not assumed to be on PATH), not a SimNIBS Python API call — invoked exactly as `charm_scitas.sbatch` does: `charm subID T1 [T2] --forceqform`, cwd set to `derivatives/SimNIBS/sub-{id}/`, completion marker `m2m_{id}/{id}.msh`. A partial `m2m_{id}` folder from a previous failed run is cleaned up before retrying (charm refuses to run over an existing one otherwise) — same behavior as the sbatch script.
- Real run is ~30-90 minutes — backgrounded via `job_runner.py`, polled every 3s with elapsed time shown, same pattern as Phase 3's leadfield generation.
- **Run on SCITAS** is present as a placeholder option (calls `job_runner.submit_to_scitas()`, which raises `NotImplementedError`) — UI shape is ready for Phase 6.
- Single subject at a time by design — a 90-minute local run isn't something to batch-queue; SCITAS (via the existing `submit_charm_scitas.sh`) is already the batch path for many subjects at once.

### Phase 1 — Mask Generation

- Select subject(s), atlas source (SimNIBS / BNA / Allen / FreeSurfer), and region(s) via a searchable lut-based picker (BNA falls back to manual `name: id` entry).
- **Atlas Availability** matrix: per subject × atlas, flags what's missing (m2m folder, subject labeling, MNI warp field, atlas file, roi list) before you try to generate.
- Batch **Generate**: checkbox table of ready subjects, atlas name auto-appended to the mask name to avoid cross-atlas collisions (e.g. `hippocampus_BNA` vs `hippocampus_Allen`), overwrite confirmation guard.
- **Existing Masks on Disk** browser per subject (reads sidecar `.json` metadata where present).
- **Mask Preview**: three orthogonal 2D slices (T1 background + mask overlay), correctly oriented per the volume's actual axis codes (not assumed), true mm scaling (T1's Conform grid isn't isotropic), and on-the-fly nearest-neighbour resampling for masks on a different grid than `T1.nii.gz` (e.g. legacy FreeSurfer-aseg masks).
- **Export .msh for Gmsh**: on-demand mesh export of a selected mask via `MaskGeneratorEngine.export_visualization_mesh`, for viewing outside the GUI.

### Phase 2 — Cap Registration

- Select subject(s) and a cap layout: this project's own `code/pipeline/configs/caps/*.csv`, SimNIBS's built-in `ElectrodeCaps_MNI/*.csv`, or an arbitrary custom path. Note: charm already auto-registers every one of its own bundled caps into each subject's `eeg_positions/` during head modeling — the registration-status check picks this up for free (same filename convention), so in practice only project-specific caps (like `BioSemi32_MNE`) or genuinely custom ones typically need registering.
- **Registration Status** table: per subject, whether prerequisites are present (m2m folder + subject `.msh`) and whether that cap is already registered in `m2m_{id}/eeg_positions/`.
- Batch **Register**: checkbox table of ready subjects, warps the cap via charm's nonlinear MNI transform and snaps each electrode onto the nearest scalp-surface node (mesh tag `1005`, the true skin boundary — not tag `5`, the scalp *volume*, which also has interior nodes), with the same overwrite confirmation guard as mask Generate.
- **Custom-path cap space toggle**: a custom CSV can be MNI space (default — warped like any other cap) or already subject space (e.g. digitized recording positions, or a previously-registered file kept elsewhere). Subject-space files skip warping entirely and are just validated + copied into one target subject's `eeg_positions/` folder via the separate **Adopt** action — subject-space coordinates only make sense for the one subject they came from, so this is single-subject, not batch.
- Cap CSV reader handles two schemas: headerless fixed-position `Type,x,y,z,label` (SimNIBS's own built-in caps) and header-present files with columns resolved **by name** rather than assumed position (handles alternate external schemas, e.g. `#,id,x,y,z` with no `Type` column at all).
- **Cap Preview**: lightweight in-browser 3D view (Plotly `Mesh3d` + `Scatter3d`) of a subject's decimated scalp surface with a registered cap's electrode positions overlaid — a quick sanity check; the full-resolution `.msh` is still the reference for anything precise.
- Single cap at a time by design (matches the current phase scope) — multi-cap comparison in the same session is a planned later addition, not built yet.

### Phase 3 — Manual FEM for TI Validation

- **Leadfield mode** (default, fast/algebraic): pick subject, ROI/non-ROI masks (from Phase 1), a cap with an already-precomputed leadfield, and two channels' +/- electrodes + currents (electrode names come straight from the leadfield's own HDF5 attribute — no need to load the multi-GB leadfield array just to list them). Wraps `compare_ti_montages.py`'s `load_subject_resources`/`compute_ti_setup` directly, including its own by-label result caching.
- **Alternative Leadfield Sources**, for when no leadfield exists yet:
  - **Custom leadfield path** — point at any `.hdf5` elsewhere; a small reimplementation of `load_subject_resources` accepts an explicit path instead of deriving one from subject+cap_name (verified to produce byte-identical stats to the standard path).
  - **Generate & save a new leadfield** — mirrors `run_pipeline.py`'s own `TDCSLEADFIELD` step exactly (same params dict, same caching-by-params-sidecar convention), using a Phase-2-registered cap. Runs in the background (`job_runner.py`) since a real run is ~30 minutes; the page polls every 3s and shows elapsed time. The completion message echoes back the exact params used, so a "did my change actually apply?" question is always self-answerable rather than needing to guess.
  - **Run one-off FEM** — no leadfield at all: electrodes placed anywhere (by name from a registered cap, or raw `x, y, z`, snapped to the true scalp surface — tag `1005`), one real `simnibs.run_simnibs()` physics solve per channel (~1 min each, also backgrounded/polled), then TI stats computed directly from the two resulting meshes (reimplements the 52Y validation notebook's Step 4: crop to WM+GM+CSF, restrict to WM+GM, volume-weighted 99th-percentile-capped mean).
- **`job_runner.py`** (new, phase-agnostic): minimal background-job runner — launches a Python callable in a thread, tracks status in a `status.json` file inside a per-job directory. File-based rather than in-memory specifically so the same polling UI can later point at a SCITAS job directory instead of a local one; `submit_to_scitas()` is a placeholder stub (`NotImplementedError`) reserving that interface shape for later.
- Both long-running actions are cached by their exact parameters (leadfield: shape/dims/gel/tissues; one-off FEM: per-channel `.msh` already on disk) — rerunning with identical settings returns near-instantly instead of redoing the physics solve.

### Phase 4 — Comparison

- One editable table, one row per **setup**: subject, mode (leadfield/one-off), cap, Ch1/Ch2 electrodes + currents, label — covers all three comparison patterns (same subject/different montages, different subjects/same montage, fully independent setups) without separate UI for each.
- **Bulk subject add**: a proper multi-select `dcc.Dropdown` above the table + "Add Rows for Selected Subjects" creates one row per chosen subject at once. Per-row cells are plain text (DataTable's built-in dropdown cell editor has rough typing/backspace behavior) — the exception is **Cap**, which stays a dropdown (filtered to that subject's registered caps) with **red/green cell coloring** for registration status, and **Mode**, whose cell also turns red/green (only when set to `leadfield`) for whether a leadfield actually exists for that row's specific subject+cap pair.
- **Cap**, **ROI**, and **non-ROI** can each independently be "common" (one shared value across every row, with a per-subject readiness check) or per-row. Common-cap readiness includes an inline **Register** button (reuses Phase 2's `cap_discovery.register_cap()` directly) for subjects that have the cap available but haven't registered it yet. Common ROI/non-ROI readiness just flags what's missing and points at Mask Generation (no inline generation — deliberately keeps this screen focused on running comparisons).
- ROI/non-ROI labels are filled two ways: an **Atlas → Region** picker (any potential region from that atlas's full list, via the same `discovery.build_lut()` Phase 1 uses) or an **existing mask names** picker (scanned from real files across the table's current subjects, e.g. `hippocampus_BNA (2/2 subjects)`) — either just prefills the label field, which stays freely editable for custom multi-region-union names.
- **Run Comparison**: leadfield rows compute synchronously (fast); one-off rows each get their own background job (`job_runner.py`) and the page polls until all are done. Results show a status table plus a metric-by-setup comparison table.
- **Export Results (CSV)**: auto-fills a timestamped filename (editable) and downloads the status + full comparison table via `dcc.Download`.

---

## Requirements

The GUI needs everything `code/pipeline/create_masks.py` needs (SimNIBS, nibabel, scipy) **plus** `dash` and `plotly`, which aren't part of the base SimNIBS install.

**Environment:** `D:\envs\bids_ti_gui_env` — a full clone of `~\SimNIBS-4.6\simnibs_env` (conda) with `dash` and `plotly` pip-installed on top. The base SimNIBS env itself was left untouched.

To recreate this environment elsewhere:

```bash
conda create -p <target_path> --clone <path_to_simnibs_env>
<target_path>/python.exe -m pip install dash plotly
```

Minimal package list beyond the SimNIBS env: `dash`, `plotly` (pulls in `flask`, `werkzeug`, `pydantic`, etc. as transitive deps — no manual install needed).

Python version: 3.11 (matches the SimNIBS 4.6 env).

---

## Folder Structure

```text
code/gui/
├── README.md
├── app.py                    # Entry point — Dash app shell + nav, multi-page (use_pages=True)
├── common.py                  # Phase-agnostic: PROJECT_DIR, subject scanning, sys.path setup
│                               #   for importing code/pipeline/* directly. No Dash import.
├── job_runner.py               # Phase-agnostic: background-job runner (thread + status.json),
│                               #   used by Phase 3's long-running actions. No Dash import.
├── discovery.py               # Phase 1 (masks) data/logic layer, no Dash import — atlas
│                               #   discovery, path validation, mask generation, slice-preview
│                               #   geometry. Imports code/pipeline/create_masks.py directly.
├── cap_discovery.py           # Phase 2 (caps) data/logic layer, no Dash import — cap listing,
│                               #   registration status, MNI→subject warp + scalp projection.
├── fem_discovery.py            # Phase 3 (FEM/TI) data/logic layer, no Dash import — leadfield
│                               #   discovery + compute, leadfield generation, one-off FEM.
│                               #   Imports code/pipeline/compare_ti_montages.py directly.
├── comparison_discovery.py     # Phase 4 (comparison) data/logic layer, no Dash import — resolves
│                               #   inputs (mask labels, electrode names/coords) and runs one setup
│                               #   row; delegates the actual compute to fem_discovery.py.
├── charm_discovery.py          # Phase 0 (head modeling) data/logic layer, no Dash import —
│                               #   T1/T2/m2m status, local charm subprocess invocation.
└── pages/
    ├── head_modeling.py       # Phase 0 page — layout + Dash callbacks (path "/", the landing page)
    ├── mask_generation.py     # Phase 1 page — layout + Dash callbacks (path "/masks")
    ├── cap_registration.py    # Phase 2 page — layout + Dash callbacks
    ├── fem_validation.py      # Phase 3 page — layout + Dash callbacks
    └── comparison.py          # Phase 4 page — layout + Dash callbacks
```

`discovery.py`, `cap_discovery.py`, `fem_discovery.py`, `comparison_discovery.py`, and `charm_discovery.py` import shared bits from `common.py` rather than duplicating them — each later phase's `*_discovery.py` should do the same.

`common.py` assumes the project root is `D:/MINDS_Project_Karim/BIDS_TI_Toolbox` (`PROJECT_DIR` constant at the top) — update that if running against a different checkout.

---

## Running

```bash
D:/envs/bids_ti_gui_env/python.exe code/gui/app.py
```

Then open **http://127.0.0.1:8050/** in a browser. Runs with `debug=True` (auto-reloads on file changes) — fine for local development, not intended for any kind of shared/production deployment.

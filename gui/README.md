# BIDS TI Toolbox — GUI

A modular Dash-based GUI wrapping the BIDS TI Toolbox pipeline (`code/pipeline/`), built one phase at a time. Each phase is a separate page under `pages/`, so later phases slot in without rewriting earlier ones.

---

## Phase Status

The nav groups pages by category rather than by build order — each page's `dash.register_page()` call sets `category=`/`order=` (extra kwargs Dash just stores in the page registry; `app.py`'s `_build_nav()` groups by them):

| Category | Pages (nav order) |
| -------- | ------------------ |
| **Preprocessing** | Head Modeling → Mask Generation → Cap Registration |
| **Simulation** | FEM Validation → Comparison |
| **Optimization** | Config Generation → Run Pipeline |
| **Settings** | SCITAS Connection → Data Directory |

| Phase | Page | Status |
| ----- | ---- | ------ |
| 0 | Head Modeling (charm) | ✅ Done — local + SCITAS |
| 1 | Mask Generation | ✅ Done |
| 2 | Cap Registration | ✅ Done |
| 3 | Manual FEM for TI validation | ✅ Done |
| 4 | Comparison view (multiple FEM setups side by side) | ✅ Done |
| 5 | Config file generation | ✅ Done |
| 6 | Run Pipeline + SCITAS/cluster job orchestration | ✅ Done — local + SCITAS |
| — | Settings: SCITAS Connection, Data Directory | ✅ Done (infra, not a numbered phase) |

### Phase 0 — Head Modeling (charm)

- Select one subject (must have `rawdata/sub-{id}/anat/sub-{id}_T1w.nii.gz`; `_T2w.nii.gz` optional — improves segmentation quality), see T1/T2/m2m status, then run charm to build `m2m_{id}/`.
- charm is a real external command-line tool (`charm.exe`, resolved from the running interpreter's own `Scripts/` folder — not assumed to be on PATH), not a SimNIBS Python API call — invoked exactly as `charm_scitas.sbatch` does: `charm subID T1 [T2] --forceqform`, cwd set to `derivatives/SimNIBS/sub-{id}/`, completion marker `m2m_{id}/{id}.msh`. A partial `m2m_{id}` folder from a previous failed run is cleaned up before retrying (charm refuses to run over an existing one otherwise) — same behavior as the sbatch script.
- Real run is ~30-90 minutes — backgrounded via `job_runner.py`, polled every 3s with elapsed time shown, same pattern as Phase 3's leadfield generation.
- **Run on SCITAS** actually submits now (`charm_discovery.run_charm_on_scitas()`): uploads T1/T2 if SCITAS doesn't have them yet, submits `charm_scitas.sbatch` over SSH, blocks polling `squeue`/`sacct` until the job leaves the queue, then `scp`'s `m2m_{id}/` back — same return shape as the local path, so this page's polling UI doesn't need to know which one ran. Requires a working SCITAS connection (see Settings → SCITAS Connection).
- Single subject at a time by design — a 90-minute local run isn't something to batch-queue; SCITAS (via the existing `submit_charm_scitas.sh`, or this page's own SCITAS option) is already the batch path for many subjects at once.

### Phase 1 — Mask Generation

- Select subject(s), atlas source (SimNIBS / BNA / Allen / FreeSurfer), and region(s) via a searchable lut-based picker (BNA falls back to manual `name: id` entry).
- **Atlas Availability** matrix: per subject × atlas, flags what's missing (m2m folder, subject labeling, MNI warp field, atlas file, roi list) before you try to generate.
- Batch **Generate**: checkbox table of ready subjects, atlas name auto-appended to the mask name to avoid cross-atlas collisions (e.g. `hippocampus_BNA` vs `hippocampus_Allen`), overwrite confirmation guard.
- **Existing Masks on Disk** browser per subject (reads sidecar `.json` metadata where present).
- **Mask Preview**: three orthogonal 2D slices (T1 background + mask overlay), correctly oriented per the volume's actual axis codes (not assumed), true mm scaling (T1's Conform grid isn't isotropic), and on-the-fly nearest-neighbour resampling for masks on a different grid than `T1.nii.gz` (e.g. legacy FreeSurfer-aseg masks).
- **Export .msh for Gmsh**: on-demand mesh export of a selected mask via `MaskGeneratorEngine.export_visualization_mesh`, for viewing outside the GUI.

### Phase 2 — Cap Registration

- Select subject(s) and a cap layout: this project's own `code/resources/caps/*.csv`, SimNIBS's built-in `ElectrodeCaps_MNI/*.csv`, or an arbitrary custom path. Note: charm already auto-registers every one of its own bundled caps into each subject's `eeg_positions/` during head modeling — the registration-status check picks this up for free (same filename convention), so in practice only project-specific caps (like `BioSemi32_MNE`) or genuinely custom ones typically need registering.
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
- **`job_runner.py`** (phase-agnostic): minimal background-job runner — launches a Python callable in a thread, tracks status in a `status.json` file inside a per-job directory. File-based rather than in-memory specifically so the same polling UI works unchanged for a SCITAS job (see Phase 0/6) — a SCITAS run just passes a different callable (`run_x_on_scitas`, which submits + blocks polling the remote queue) to the exact same `start_local_job()`, rather than needing a separate code path. Its own `submit_to_scitas()` stub predates that pattern and isn't called by any page anymore — kept, not deleted, but effectively unused now.
- Both long-running actions are cached by their exact parameters (leadfield: shape/dims/gel/tissues; one-off FEM: per-channel `.msh` already on disk) — rerunning with identical settings returns near-instantly instead of redoing the physics solve.

### Phase 4 — Comparison

- One editable table, one row per **setup**: subject, mode (leadfield/one-off), cap, Ch1/Ch2 electrodes + currents, label — covers all three comparison patterns (same subject/different montages, different subjects/same montage, fully independent setups) without separate UI for each.
- **Bulk subject add**: a proper multi-select `dcc.Dropdown` above the table + "Add Rows for Selected Subjects" creates one row per chosen subject at once. Per-row cells are plain text (DataTable's built-in dropdown cell editor has rough typing/backspace behavior) — the exception is **Cap**, which stays a dropdown (filtered to that subject's registered caps) with **red/green cell coloring** for registration status, and **Mode**, whose cell also turns red/green (only when set to `leadfield`) for whether a leadfield actually exists for that row's specific subject+cap pair.
- **Cap**, **ROI**, **non-ROI**, and **Electrode settings** can each independently be "common" (one shared value across every row, with a per-subject readiness check) or per-row. Common-cap readiness includes an inline **Register** button (reuses Phase 2's `cap_discovery.register_cap()` directly) for subjects that have the cap available but haven't registered it yet. Common ROI/non-ROI readiness just flags what's missing and points at Mask Generation (no inline generation — deliberately keeps this screen focused on running comparisons). Electrode settings matter because a cap can now have more than one cached leadfield variant (different dimensions/gel thickness — see the pipeline README's leadfield-caching note); per-row mode shows a dropdown of that row's own (subject, cap) pair's available variants.
- ROI/non-ROI labels are filled two ways: an **Atlas → Region** picker (any potential region from that atlas's full list, via the same `discovery.build_lut()` Phase 1 uses) or an **existing mask names** picker (scanned from real files across the table's current subjects, e.g. `hippocampus_BNA (2/2 subjects)`) — either just prefills the label field, which stays freely editable for custom multi-region-union names.
- **Run Comparison**: leadfield rows compute synchronously (fast); one-off rows each get their own background job (`job_runner.py`) and the page polls until all are done. Results show a status table plus a metric-by-setup comparison table.
- **Export Results (CSV)**: auto-fills a timestamped filename (editable) and downloads the status + full comparison table via `dcc.Download`.

### Phase 5 — Config Generation

- Builds `code/pipeline/configs/*.json` — the exact same files/format `generate_configs.py` itself writes — from a form, instead of hand-editing that script's `SUBJECTS`/`ROIS`/etc. lists.
- **ROI and non-ROI are each a single union of regions** — one Atlas → Region picker per side (same pattern as Mask Generation); "multiple ROIs" in the underlying pipeline just means a union of labels within one `ROIConfig`, not multiple separate configs.
- **Atlas source per side**: BNA and SimNIBS/FreeSurfer regions get built into the config inline (`bna_labels`/`labels` — Section 1 can build these itself). **Allen is different**: `run_pipeline.py`'s Section 1 has no code path that reads Allen's label space, so an Allen-sourced ROI/non-ROI must already exist on disk as a mask (built via Mask Generation — which shares the same `sub-{id}_label-{name}_mask.nii.gz` naming for ROI/non-ROI/general now, specifically so this works) before Section 1's own skip-if-exists check can pick it up.
- Only commonly-tweaked optimizer/electrode fields are exposed (`postproc`, `cpus`, `focality_threshold`, `hard_roi_constraint`, `no_adjacent_electrodes`, electrode diameter/gel/max-current) — everything else is fixed at `generate_configs.py`'s current defaults. The old TesFlex-only fields don't exist in `OptimizerConfig` at all anymore (see the pipeline README). Note: SimNIBS's own `dimensions` field is the full diameter, not a radius — it's halved internally (`electrode_placement.py`) to get the ellipse's semi-axes.
- **Generate** writes real files into `code/pipeline/configs/` and rebuilds `subject_configs.sh` by scanning every config actually on disk (not just this session's) — so re-running Generate for a different ROI across sessions never silently drops earlier subjects' entries. Never touches `generate_configs.py`'s own hardcoded lists.

### Phase 6 — Run Pipeline

- Picks an existing config from `code/pipeline/configs/` (Config Generation or `generate_configs.py`) and runs it — **locally** or **on SCITAS**.
- **Local**: runs `run_pipeline.py` as a subprocess, overriding `project_dir`/`cap_csv`/`bna_atlas_path` (baked into the config as SCITAS-container paths) to this machine's own paths via `run_pipeline.py`'s own `--set KEY=VALUE` mechanism — the config file on disk is never modified.
- **SCITAS** (`run_discovery.run_pipeline_on_scitas()`): uploads the config plus any prerequisite SCITAS doesn't have yet (subject `m2m_/`, registered cap CSV, the BNA atlas file) — translating the config's container-internal paths (`/mnt/BIDS_TI_Toolbox/...`, only valid inside the Apptainer container) to their real location on SCITAS scratch first, since existence checks run over plain SSH on the bare login node — then submits `simnibs_ti_pipeline.sbatch`, blocks polling the SLURM queue. Results stay on SCITAS scratch; sync them back via Settings → Data Directory.
- Single config at a time by design — a full run (leadfield + cap search + analysis/viz) is long even locally, and each config is already one subject/ROI/goal unit. Poll + final log (not live streaming), same pattern as Head Modeling.

### Settings

Two pages, both under a dedicated **Settings** nav category since they're cross-cutting infrastructure rather than a pipeline phase:

- **SCITAS Connection** (`scitas_discovery.py`) — tests whether this machine can reach `jed.hpc.epfl.ch` non-interactively (`BatchMode=yes`, so an auth prompt fails fast instead of hanging with no TTY to answer it). Connection is via the system's own `ssh`/`scp` binaries, not a new Python dependency — works out of the box if `~/.ssh/config` already has a working `Host` entry for the cluster. If it doesn't: override host/username/identity-file (saved locally, blank = today's default), or generate a new local SSH keypair (never overwrites an existing one, doesn't itself grant access — the resulting public key still needs manual installation on the cluster side).
- **Data Directory** (`common.py` + `scitas_discovery.py`) — makes explicit the three separate locations in play: the code+environment (fixed, this repo), the **Local (analysis) data directory** (what `common.PROJECT_DIR` already meant — rawdata/derivatives on this machine, now GUI-settable via a `data_settings.json` file next to `common.py`; **takes effect on GUI restart**, since every discovery module's function defaults bind to `PROJECT_DIR` at import time), and the **Server (SCITAS) data directory** (the project root on the bare SCITAS filesystem, defaults to `/scratch/{username}/BIDS_TI_Toolbox`, override **takes effect immediately** — every SCITAS action reads it live). Also has a "Set Up Directory Structure" action (creates the `rawdata/`/`derivatives/SimNIBS/`/`derivatives/freesurfer/` skeleton) and a "Sync Data from SCITAS" picker (lists remote subjects + per-folder existence, `scp -r`'s only what you explicitly select — deliberately not a whole-tree auto-sync, since derivatives/leadfields can be very large).

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
├── common.py                  # Phase-agnostic: PROJECT_DIR (local data dir), subject scanning,
│                               #   sys.path setup for importing code/pipeline/* directly, the
│                               #   Data Directory settings file. No Dash import.
├── job_runner.py               # Phase-agnostic: background-job runner (thread + status.json).
│                               #   A SCITAS job fits this exact contract too — see
│                               #   scitas_discovery.py — "submit, then poll a remote queue"
│                               #   instead of "compute directly", same status.json shape. No Dash import.
├── scitas_discovery.py          # Phase-agnostic: SCITAS SSH/SCP primitives (shells out to the
│                               #   system ssh/scp — no new dependency), connection settings,
│                               #   code-sync status/action, remote data listing/sync. No Dash import.
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
│                               #   T1/T2/m2m status, local charm subprocess invocation, and
│                               #   run_charm_on_scitas() (submits charm_scitas.sbatch remotely).
├── config_discovery.py         # Phase 5 (config generation) data/logic layer, no Dash import —
│                               #   builds PipelineConfig-shaped JSON, writes to
│                               #   code/pipeline/configs/, rebuilds subject_configs.sh.
├── run_discovery.py            # Phase 6 (run pipeline) data/logic layer, no Dash import —
│                               #   runs run_pipeline.py locally (subprocess, path overrides) or
│                               #   on SCITAS (run_pipeline_on_scitas — upload + submit + poll).
└── pages/
    ├── head_modeling.py       # Phase 0 page — layout + Dash callbacks (path "/", the landing page)
    ├── mask_generation.py     # Phase 1 page — layout + Dash callbacks (path "/masks")
    ├── cap_registration.py    # Phase 2 page — layout + Dash callbacks
    ├── fem_validation.py      # Phase 3 page — layout + Dash callbacks
    ├── comparison.py          # Phase 4 page — layout + Dash callbacks
    ├── config_generation.py   # Phase 5 page — layout + Dash callbacks (path "/config-generation")
    ├── run_optimization.py    # Phase 6 page — layout + Dash callbacks (path "/run-pipeline")
    ├── scitas_connection.py   # Settings page — layout + Dash callbacks (path "/scitas-connection")
    └── data_directory.py      # Settings page — layout + Dash callbacks (path "/data-directory")
```

`discovery.py`, `cap_discovery.py`, `fem_discovery.py`, `comparison_discovery.py`, `charm_discovery.py`, `config_discovery.py`, and `run_discovery.py` import shared bits from `common.py` rather than duplicating them — each later phase's `*_discovery.py` should do the same. `charm_discovery.py` and `run_discovery.py` additionally import `scitas_discovery.py` for their SCITAS variants.

`common.py` defines two separate roots:

- `PROJECT_DIR` — external per-project data (`rawdata/`, `derivatives/`, subject MRIs), **not** part of this repo — the "Local (analysis) data directory" on the Data Directory settings page. Resolved in priority order: the `BIDS_TI_PROJECT_DIR` env var, then `data_settings.json` (next to `common.py`, written by the Data Directory page), then the hardcoded fallback. It's a module-level constant read once at import time — changing it via the Settings page only takes effect after restarting the GUI, it does not hot-swap for an already-running process.
- `RESOURCES_DIR` — generic, non-personalized shared resources that travel with the code (`code/resources/atlases/`, `code/resources/caps/`), resolved relative to `common.py` itself, always alongside `code/gui/` regardless of machine.

Note this is distinct from the **Server (SCITAS) data directory** (`scitas_discovery.scitas_scratch_dir()`, defaults to `/scratch/{username}/BIDS_TI_Toolbox`) — the project root on the SCITAS filesystem itself, unrelated to where this machine's own data lives. See the Settings → Data Directory section above.

---

## Running

```bash
D:/envs/bids_ti_gui_env/python.exe code/gui/app.py
```

Then open **http://127.0.0.1:8050/** in a browser. Runs with `debug=True` (auto-reloads on file changes) — fine for local development, not intended for any kind of shared/production deployment.

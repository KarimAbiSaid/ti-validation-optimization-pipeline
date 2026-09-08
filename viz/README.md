# TI field visualisation

**Written by Wei**, brought this part into this project for reuse rather than
reinventing it. Thanks Wei!

Two steps. In the *source* project these ran in two separate environments
(SimNIBS reads the meshes, PyVista draws them); **in this project they don't
need to** — `code/gui`'s own environment (`bids_ti_gui_env`, see
`code/gui/README.md`) already carries simnibs, nibabel, pyvista, and
matplotlib together (installing pyvista into it was verified to touch
neither numpy nor matplotlib), so the same interpreter runs both steps below.
No second venv to build.

```
# 1. mesh -> small .npz surfaces
<bids_ti_gui_env>\python.exe code/viz/extract_surfaces.py \
    --field-mesh derivatives/SimNIBS/sub-<ID>/comparison/setup_head_mesh.msh \
    --head-mesh  derivatives/SimNIBS/sub-<ID>/m2m_<ID>/<ID>.msh \
    --field max_TI \
    --out-dir    derivatives/SimNIBS/sub-<ID>/comparison/viz_cache

# 2. surfaces -> figure  (same interpreter)
<bids_ti_gui_env>\python.exe code/viz/render_ti.py \
    --cache-dir  derivatives/SimNIBS/sub-<ID>/comparison/viz_cache \
    --electrodes derivatives/SimNIBS/sub-<ID>/m2m_<ID>/eeg_positions/sub-<ID>_ses-1_electrode_positions_native.csv \
    --out        derivatives/SimNIBS/sub-<ID>/comparison/figures/sub-<ID>_maxTI_3views.png
```

Full procedure for running this without an agent, including the figure
checks: [SOP.md](SOP.md). Agent-facing version of the same process:
`.claude/skills/ti-field-figures/SKILL.md` at the project root.

> **Provenance note:** this pipeline (these two scripts + SOP.md/SKILL.md) was
> brought in from another setup, where the interpreter paths were
> `C:\Users\hsu\SimNIBS-4.6\simnibs_env\python.exe` and
> `C:\Users\hsu\.venvs\ti-viz\Scripts\python.exe`. Neither exists on this
> project — use `bids_ti_gui_env` for both steps instead, per above.

For the GUI's own automated equivalent of this same two-step pipeline (one
in-process module, no subprocess for step 1, a subprocess only where a real
platform limitation requires one for step 2), see `code/gui/viz_discovery.py`
and its own section in `code/gui/README.md`.

## Deep ROIs (striatum)

Pass `--labeling` to step 1 and it also writes `roi_surface.npz`; step 2 picks
that up automatically and switches the figure over: the ROI is drawn opaque and
coloured, the cortex drops to a neutral translucent shell so you can see through
it. `--no-roi` goes back to the cortical figure, `--brain-opacity` tunes the
shell.

`--roi-labels` takes SAMSEG labels from `labeling_LUT.txt`; the default is the
striatum, 11/12/26 + 50/51/58 (caudate, putamen, accumbens, L+R). Add pallidum
with 13/52 or thalamus with 10/49.

**The ROI must come from the segmentation, not from mesh tags.** charm gives
subcortical grey no tag of its own -- the striatum tets come back tagged 1 *and*
2, so `crop_mesh(tags=[2])` would silently amputate the part of the structure
that landed in WM-tagged elements. `roi_elements()` selects on the label volume
at each tet centroid instead.

## Seeing inside the structure

The coloured surface only carries the field on the ROI *boundary*, which hugs
WM and CSF and runs low. Two ways past that, and they answer different
questions:

```
# cut planes -- each panel is cut by a plane normal to its own camera, so you
# look straight onto the cut face and read the field *inside* the structure
--clip [--clip-offset MM]

# opacity ramped with the field -- vmin fully see-through, vmax solid, so the
# cool outer rind fades and the hot core shows through it
--value-opacity
```

`--clip` needs `roi_volume.npz`, which step 1 writes next to `roi_surface.npz`.
Clipping the boundary surface alone would just expose the inside of a hollow
shell.

The cut also removes any electrode sitting in the discarded half; with the
default offset the striatum plane is below all four, so the axial panel shows
none. `--clip-offset` raises the plane if you want them back.

## Notes

- `--field` takes any element field in the mesh: `max_TI`, `normE_ch1`,
  `normE_ch2`, or the vector fields `E_ch1` / `E_ch2` (plotted as magnitude).
- The coloured surface is the boundary of the **grey matter** tets only. Taking
  the WM+GM boundary instead exposes WM faces at the brainstem cut, and those
  carry the WM hotspot (~3.15 V/m) rather than a cortical value.
- Views are anatomical: coronal looks from the front, so the subject's right is
  on the viewer's left. Sagittal looks from the subject's right.
- Colour scale is fixed at 0-1 V/m (`--vmin/--vmax`).
- `--no-shell` drops the translucent head; `--shell-opacity` tunes it.
- Electrodes on the far side of the head are drawn ghosted
  (`--far-electrode-opacity`, 0 hides them). Opaque ones show through the
  translucent shell and read as electrodes floating inside the brain.
- Boundary vs interior: the ROI surface tends to run noticeably cooler than the
  volume mean, because the boundary faces sit against WM and CSF. Quote the
  volume statistic, not the surface range.

*(The reference-values bullets from the original source project — specific to
its own sub-111Y02 — were dropped here since they don't apply to this
project's subjects; re-derive equivalents from this project's own data before
quoting numbers in a caption.)*
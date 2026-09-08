# SOP — TI E-field figures from a SimNIBS mesh

How to turn a SimNIBS simulation mesh into a publication figure: the field on a
surface or inside a deep structure, seen from three orthogonal views, inside a
translucent head, with electrodes marked and a fixed colorbar.

No Claude needed. Every step is a command you run yourself.

Flags are documented in `README.md` and in `--help` on either script; this
document is the procedure and the judgement calls around it.

## 0. What you can produce

| Figure | Adds | Answers |
| --- | --- | --- |
| Cortical surface | *(defaults)* | Where does the field land on cortex? |
| Deep ROI | `--labeling ...` | How much field reaches this structure? |
| Cut planes | `--labeling ...` + `--clip` | What is the field *inside* it? |
| Field-scaled opacity | `--value-opacity` | Same, without cutting anything away |

## 1. One-time setup

No new environment to build. `code/gui`'s own `bids_ti_gui_env` (see
`code/gui/README.md` → Requirements) already carries simnibs, numpy, nibabel,
matplotlib, and pyvista together — installing pyvista into it was verified
(`pip install --dry-run` before actually installing) to touch neither numpy
nor matplotlib, so there's no risk of it breaking the SimNIBS install the way
pip resolving PyVista's dependencies *inside a bare SimNIBS env* could. Both
steps below run with that same interpreter.

Known-good versions in that env: pyvista 0.48.4, VTK 9.6.2 (matches what the
source project this was brought in from used).

Verify:

```
<bids_ti_gui_env>\python.exe -c "import pyvista; print(pyvista.__version__)"
```

## 2. Inputs

Per subject, under `derivatives/SimNIBS/sub-<ID>/`:

- `comparison/setup_head_mesh.msh` — the simulation output. Carries the element
  fields `max_TI`, `normE_ch1`, `normE_ch2`, `E_ch1`, `E_ch2` on tets tagged
  1 (WM) and 2 (GM).
- `m2m_<ID>/<ID>.msh` — the head model, for the scalp and skull shells.
- `m2m_<ID>/segmentation/labeling.nii.gz` — SAMSEG labels, needed only for a
  deep ROI. Structure codes are in `labeling_LUT.txt` beside it.
- `m2m_<ID>/eeg_positions/sub-<ID>_ses-1_electrode_positions_native.csv` —
  electrode coordinates already in subject space. Format is
  `Electrode,x,y,z,name`.

Plot `max_TI`, not `normE_ch1` or `normE_ch2`. The single-channel magnitudes are
ordinary tDCS fields; the TI envelope is what actually stimulates. If the figure
is going into a comparison against someone else's reported values, confirm which
quantity *they* report — both sides must be the same one.

## 3. Step 1 — mesh to surfaces

Run from the project root, with `bids_ti_gui_env`'s interpreter.

```
<bids_ti_gui_env>\python.exe code/viz/extract_surfaces.py \
    --field-mesh derivatives/SimNIBS/sub-<ID>/comparison/setup_head_mesh.msh \
    --head-mesh  derivatives/SimNIBS/sub-<ID>/m2m_<ID>/<ID>.msh \
    --field      max_TI \
    --labeling   derivatives/SimNIBS/sub-<ID>/m2m_<ID>/segmentation/labeling.nii.gz \
    --out-dir    derivatives/SimNIBS/sub-<ID>/comparison/viz_cache
```

(On `cmd.exe` the line-continuation character is `^`, not `\`.)

Drop `--labeling` if you only want the cortical figure.

This writes small `.npz` files readable with numpy alone — keeps step 2 from
importing simnibs at all, even though both steps happen to share one
environment in this project now. It also prints the value ranges — **keep
that output**, section 5 needs it.

Why this step exists at all: the head mesh is ~5.9 M tetrahedra. Handing that to
a renderer is what makes naive attempts unusable. Cropping to the surfaces you
actually draw takes it to ~560 k triangles, about two orders of magnitude less.

`--roi-labels` selects the structure; the default is the striatum
(11/12/26 + 50/51/58 — caudate, putamen, accumbens, both sides). Pallidum is
13/52, thalamus 10/49.

> **The ROI comes from the segmentation, never from mesh tags.** charm gives
> subcortical grey no tag of its own — striatum tets come back tagged **1 and
> 2**. Cropping on `tags=[2]` looks reasonable and silently amputates the part
> of the structure that landed in WM-tagged elements. The script selects on the
> label volume at each tet centroid instead.

## 4. Step 2 — surfaces to figure

Same directory, same interpreter.

```
<bids_ti_gui_env>\python.exe code/viz/render_ti.py \
    --cache-dir  derivatives/SimNIBS/sub-<ID>/comparison/viz_cache \
    --electrodes derivatives/SimNIBS/sub-<ID>/m2m_<ID>/eeg_positions/sub-<ID>_ses-1_electrode_positions_native.csv \
    --label-electrodes \
    --out        derivatives/SimNIBS/sub-<ID>/comparison/figures/sub-<ID>_maxTI_3views.png
```

If `roi_surface.npz` exists the figure switches automatically: the ROI is drawn
opaque and coloured, cortex drops to a neutral translucent shell. `--no-roi`
goes back to the cortical figure.

Add for the two "see inside" variants:

- `--clip` — cuts each panel with a plane normal to its own camera, so you look
  straight onto the cut face and read the field inside the structure.
  `--clip-offset MM` moves the plane along the viewing axis.
- `--value-opacity` — ramps opacity with the field, `vmin` see-through and
  `vmax` solid, so the cool outer rind fades and the hot core shows through.

### Why PyVista and not Gmsh

Gmsh's viewer has no order-independent transparency. Translucent faces are
blended in element order, so a translucent scalp over a solid brain breaks up
differently at every camera angle, and geometry alpha and view colormaps are
separate mechanisms that are especially unstable mixed in one scene. PyVista
calls `enable_depth_peeling()` and the problem disappears. That single call is
the reason this pipeline exists; do not port it back to Gmsh.

ParaView would also work but is a GUI-first tool with a more verbose scripting
path. nilearn is the wrong tool — it knows volumetric NIfTI and FreeSurfer
surfaces, not tetrahedral head meshes, so you would have to convert away the
head shell and the transparency you came for.

## 5. Check the figure before using it

Open the PNG and look. Every defect below has actually occurred in this
pipeline, and each one looks like anatomy or data rather than like a bug:

- **Bright patches on the inferior brain that are not cortex.** WM-tagged faces
  exposed where the brainstem is cut, carrying the ~3.15 V/m WM hotspot. The
  script crops to GM-only tets to prevent it; it returns if you widen the tags.
- **An electrode floating inside the brain.** An opaque sphere on the far side
  of the head showing through the translucent shell. Far-side electrodes are
  ghosted (`--far-electrode-opacity`, `0` hides them).
- **A label with no sphere under it,** or a label in mid-brain. Labels follow
  only the electrodes actually drawn.
- **Clipping at the top of the colour scale.** Compare the range printed by
  step 1 against your `--vmax`.

In a cut figure, electrodes in the discarded half are gone by design. With the
default plane through the striatum, all four sit above it, so the axial panel
shows none.

## 6. What to quote in text

Report the **volume** statistic, not the range on the surface. The coloured
surface is the ROI boundary, which hugs WM and CSF and runs cool — it will
consistently understate the stimulation reaching the structure's interior. Get
this project's own subject numbers by actually running the pipeline before
quoting anything in a caption; don't reuse figures from wherever this pipeline
came from.

Choosing `--vmin/--vmax` is a real trade-off, so state your choice in the
caption:

- **0–1** — a round, global scale, comparable across subjects and against
  externally reported values. Most of a subcortical structure sits at the dark
  end and spatial structure is hard to read.
- **0–0.5 or 0–0.8** — structure becomes legible. Confirm against step 1's
  printed maximum that nothing is clipped, and say so in the caption.

Comparability is not really lost by narrowing the scale, because what you
compare between subjects is the number, not the colour.

## 7. Other subjects and other structures

Everything is per-subject: substitute `<ID>` throughout. Nothing in either
script is specific to any one subject.

The montage in the examples above (`--ch1 AR AL --ch2 PR PL`) is just an
example from the source project's own 4-electrode cohort; pass whatever
electrode names actually appear in this project's own CSV.

## 8. Troubleshooting

| Symptom | Cause |
| --- | --- |
| `field 'X' not in ...` | Step 1 lists the available fields in the error. |
| `electrodes [...] not in ...` | Names must match the CSV's 5th column exactly. |
| `no tets fell inside labels` | Wrong label numbers, or a labeling file from another subject. |
| Blank or black panels | Off-screen rendering with no GPU. Set `PYVISTA_OFF_SCREEN=true`; on a headless Linux node run under `xvfb-run`. |
| Broken transparency, wrong layer order | Depth peeling is off or out of peels. Raise `--peels`. |
| Renderer runs out of memory | Step 1 was skipped, or pointed at the full head mesh. |
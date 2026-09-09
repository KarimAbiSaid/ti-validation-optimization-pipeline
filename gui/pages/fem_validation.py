"""
pages/fem_validation.py — Phase 3: leadfield-based TI computation for one
subject/montage (compare_ti_montages.py's compute_ti_setup/
load_subject_resources, wrapped by fem_discovery.py).

First increment: leadfield mode only. A cap only shows up in the dropdown if
a precomputed leadfield HDF5 already exists for it (fast, algebraic
compute — no long-running job). If no leadfield exists at all for a
subject, that's flagged but not yet actionable from here — full-FEM
(one-off simnibs.run_simnibs per channel) and leadfield generation both need
real background-job infrastructure (minutes to ~30min per run) that isn't
built yet.
"""
import os
import time

import dash
from dash import html, dcc, dash_table, callback, Input, Output, State, ctx

import cap_discovery as cd
import discovery
import fem_discovery as fd
import job_runner as jr
import viz_discovery as vz

dash.register_page(__name__, path="/fem", name="FEM Validation", category="Simulation", order=1)


def _styled_table(id_, columns, data=None, **kwargs):
    return dash_table.DataTable(
        id=id_,
        columns=columns,
        data=data or [],
        style_cell={"textAlign": "left", "fontFamily": "monospace", "fontSize": "13px", "padding": "4px"},
        style_table={"overflowX": "auto"},
        **kwargs,
    )


def _channel_block(idx):
    return html.Div([
        html.H4(f"Channel {idx}"),
        html.Div([
            html.Div([
                html.Label("+ electrode"),
                dcc.Dropdown(id=f"fv-ch{idx}-plus", placeholder="Select..."),
            ], style={"maxWidth": "180px", "marginRight": "1rem"}),
            html.Div([
                html.Label("− electrode"),
                dcc.Dropdown(id=f"fv-ch{idx}-minus", placeholder="Select..."),
            ], style={"maxWidth": "180px", "marginRight": "1rem"}),
            html.Div([
                html.Label("Current (mA)"),
                dcc.Input(id=f"fv-ch{idx}-current", type="number", value=2.0, step=0.1,
                          style={"width": "100%"}),
            ], style={"maxWidth": "120px"}),
        ], style={"display": "flex", "flexWrap": "wrap"}),
    ], style={"marginBottom": "1rem"})


def _channel_block_generic(prefix, title):
    """Same shape as _channel_block, but with a caller-chosen id prefix —
    used for the custom-leadfield-path section so its dropdowns don't
    collide with the main compute flow's fv-ch1-*/fv-ch2-* ids."""
    return html.Div([
        html.H4(title),
        html.Div([
            html.Div([
                html.Label("+ electrode"),
                dcc.Dropdown(id=f"fv-{prefix}-plus", placeholder="Select..."),
            ], style={"maxWidth": "180px", "marginRight": "1rem"}),
            html.Div([
                html.Label("− electrode"),
                dcc.Dropdown(id=f"fv-{prefix}-minus", placeholder="Select..."),
            ], style={"maxWidth": "180px", "marginRight": "1rem"}),
            html.Div([
                html.Label("Current (mA)"),
                dcc.Input(id=f"fv-{prefix}-current", type="number", value=2.0, step=0.1,
                          style={"width": "100%"}),
            ], style={"maxWidth": "120px"}),
        ], style={"display": "flex", "flexWrap": "wrap"}),
    ], style={"marginBottom": "1rem"})


def _oneoff_channel_block(idx):
    """+/- electrode, each choosable by name (from the registered cap
    dropdown) OR a raw 'x, y, z' override that takes precedence if filled."""
    def _electrode(sign, label):
        return html.Div([
            html.Label(f"{label} electrode — by name"),
            dcc.Dropdown(id=f"fv-oneoff-ch{idx}-{sign}-name", placeholder="Select..."),
            html.Label("...or raw x, y, z (overrides name if set)", style={"fontSize": "12px"}),
            dcc.Input(id=f"fv-oneoff-ch{idx}-{sign}-raw", type="text", placeholder="e.g. 12.3, 45.6, 78.9",
                      style={"width": "100%"}),
        ], style={"maxWidth": "260px", "marginRight": "1.5rem"})

    return html.Div([
        html.H4(f"Channel {idx}"),
        html.Div([
            _electrode("plus", "+"),
            _electrode("minus", "−"),
            html.Div([
                html.Label("Current (mA)"),
                dcc.Input(id=f"fv-oneoff-ch{idx}-current", type="number", value=2.0, step=0.1,
                          style={"width": "100%"}),
            ], style={"maxWidth": "120px"}),
        ], style={"display": "flex", "flexWrap": "wrap"}),
    ], style={"marginBottom": "1rem"})


layout = html.Div([
    html.H2("FEM Validation — Leadfield Mode"),

    html.Div([
        html.Div([
            html.Label("Subject"),
            dcc.Dropdown(id="fv-subject-dropdown", placeholder="Select subject..."),
        ], style={"maxWidth": "260px", "marginRight": "1.5rem"}),
        html.Div([
            html.Label("Cap + electrode settings (only variants with a precomputed leadfield are listed)"),
            dcc.Dropdown(id="fv-cap-dropdown", placeholder="Select cap...", style={"maxWidth": "480px"}),
        ], style={"maxWidth": "480px"}),
    ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "0.5rem"}),
    html.Div(id="fv-leadfield-note", style={"fontSize": "13px", "marginBottom": "1.5rem"}),

    html.Div([
        html.Div([
            html.Label("ROI mask"),
            dcc.Dropdown(id="fv-roi-dropdown", placeholder="Select ROI mask..."),
        ], style={"maxWidth": "380px", "marginRight": "1.5rem"}),
        html.Div([
            html.Label("Non-ROI mask (optional)"),
            dcc.Dropdown(id="fv-nonroi-dropdown", placeholder="Select non-ROI mask..."),
        ], style={"maxWidth": "380px"}),
    ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "1.5rem"}),

    _channel_block(1),
    _channel_block(2),

    html.Div([
        html.Div([
            html.Label("Electrode dims (mm) — metadata only, prefilled from the leadfield"),
            dcc.Input(id="fv-electrode-dims", type="text", placeholder="e.g. 14, 14",
                      style={"width": "100%"}),
        ], style={"maxWidth": "320px", "marginRight": "1.5rem"}),
        html.Div([
            html.Label("Label"),
            dcc.Input(id="fv-label", type="text", value="setup", style={"width": "100%"}),
        ], style={"maxWidth": "200px"}),
    ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "1rem"}),

    html.Button("Compute TI", id="fv-compute-button", n_clicks=0,
                style={"padding": "0.5rem 1.5rem"}),

    dcc.Loading(html.Div(id="fv-compute-results", style={"marginTop": "1.5rem"})),
    dcc.Store(id="fv-render-context"),

    html.Div([
        html.H4("Render Figure", style={"marginTop": "0"}),
        html.P("3-view TI field figure for the compute above, inside a translucent whole-brain shell. "
               "Highlighted regions render opaque on top — pick any combination of existing masks "
               "(e.g. hippocampus AND amygdala together).",
               style={"fontSize": "13px", "color": "#666"}),

        html.Div([
            html.P("...or load an already-computed simulation directly — no new compute needed "
                   "(scans this subject's Compute TI/Run Comparison results and finished Run "
                   "Pipeline exhaustive-search runs).",
                   style={"fontSize": "12px", "color": "#666", "marginBottom": "0.5rem"}),
            html.Div([
                html.Div([
                    html.Label("Existing simulation"),
                    dcc.Dropdown(id="fv-viz-existing-dropdown", placeholder="Select subject above first...",
                                style={"minWidth": "420px"}),
                ], style={"marginRight": "1.5rem"}),
                html.Div([
                    html.Label("Cap (for electrode positions)"),
                    dcc.Dropdown(id="fv-viz-existing-cap-dropdown", placeholder="—",
                                style={"minWidth": "260px"}),
                ], style={"marginRight": "1.5rem"}),
                html.Button("Load", id="fv-viz-load-existing-button", n_clicks=0, disabled=True),
            ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-end"}),
            html.Div(id="fv-viz-existing-note", style={"fontSize": "12px", "marginTop": "0.35rem"}),
            dcc.Store(id="fv-viz-existing-meshes-store"),
        ], style={"marginBottom": "1rem", "paddingBottom": "0.75rem", "borderBottom": "1px solid #eee"}),

        html.Div([
            html.Div([
                html.Label("Highlight region(s)"),
                dcc.Dropdown(id="fv-viz-highlight-dropdown", multi=True,
                            placeholder="Select region(s)...", style={"minWidth": "300px"}),
            ], style={"marginRight": "1.5rem"}),
            html.Div([
                html.Label("Color scale max (V/m)"),
                dcc.Input(id="fv-viz-vmax", type="number", value=1.0, step=0.05,
                         style={"width": "100px"}),
            ]),
        ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-end", "marginBottom": "0.75rem"}),
        html.Button("Render Figure", id="fv-viz-render-button", n_clicks=0, disabled=True),
        dcc.Store(id="fv-viz-job-store"),
        dcc.Interval(id="fv-viz-interval", interval=2000, disabled=True),
        dcc.Loading(html.Div(id="fv-viz-results", style={"marginTop": "1rem"})),
        dcc.Download(id="fv-viz-download"),
    ], style={"marginTop": "1rem", "padding": "0.75rem", "border": "1px solid #ccc", "borderRadius": "4px"}),

    html.Div([
        html.H4("Slice Viewer", style={"marginTop": "0"}),
        html.P("Scroll through T1 slices (axial/coronal/sagittal, each its own slider) with the "
               "TI field amplitude overlaid and the region outlined — a different view from Render "
               "Figure above (2D voxel slices, not a 3D surface). Downsampled for a fast, fully "
               "interactive scrubber — dragging a slider never touches the server.",
               style={"fontSize": "13px", "color": "#666"}),
        html.Div([
            html.Div([
                html.Label("Highlight region(s) (outlined)"),
                dcc.Dropdown(id="fv-slice-highlight-dropdown", multi=True,
                            placeholder="Select region(s)...", style={"minWidth": "300px"}),
            ], style={"marginRight": "1.5rem"}),
        ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-end", "marginBottom": "0.5rem"}),
        dcc.Checklist(
            id="fv-slice-roi-only",
            options=[{"label": " Color only inside the region(s) — rest stays grayscale "
                              "(off = whole brain colored)", "value": "roi_only"}],
            value=[], style={"marginBottom": "0.75rem"},
        ),
        html.Button("Build Slice Viewer", id="fv-slice-build-button", n_clicks=0, disabled=True),
        dcc.Store(id="fv-slice-job-store"),
        dcc.Interval(id="fv-slice-interval", interval=2000, disabled=True),
        html.Div(id="fv-slice-status", style={"marginTop": "0.5rem", "fontSize": "13px"}),
        html.Img(id="fv-slice-colorbar-img", style={"marginTop": "0.5rem", "maxWidth": "320px"}),
        html.Div(id="fv-slice-legend-row", style={"marginTop": "0.35rem", "fontSize": "13px"}),

        html.Div([
            html.Div([
                html.H5("Axial"),
                html.Img(id="fv-slice-axial-img", style={"width": "100%"}),
                dcc.Slider(id="fv-slice-axial-slider", min=0, max=0, step=1, value=0),
            ], style={"flex": "1", "marginRight": "0.75rem", "minWidth": "200px"}),
            html.Div([
                html.H5("Coronal"),
                html.Img(id="fv-slice-coronal-img", style={"width": "100%"}),
                dcc.Slider(id="fv-slice-coronal-slider", min=0, max=0, step=1, value=0),
            ], style={"flex": "1", "marginRight": "0.75rem", "minWidth": "200px"}),
            html.Div([
                html.H5("Sagittal"),
                html.Img(id="fv-slice-sagittal-img", style={"width": "100%"}),
                dcc.Slider(id="fv-slice-sagittal-slider", min=0, max=0, step=1, value=0),
            ], style={"flex": "1", "minWidth": "200px"}),
        ], id="fv-slice-viewer-container", style={"display": "none", "flexWrap": "wrap", "marginTop": "1rem"}),

        dcc.Store(id="fv-slice-axial-frames"),
        dcc.Store(id="fv-slice-coronal-frames"),
        dcc.Store(id="fv-slice-sagittal-frames"),
    ], style={"marginTop": "1rem", "padding": "0.75rem", "border": "1px solid #ccc", "borderRadius": "4px"}),

    html.Hr(style={"marginTop": "2.5rem"}),
    html.H3("Alternative Leadfield Sources"),
    html.P("For when no precomputed leadfield exists yet for this subject/cap above.",
           style={"fontSize": "13px", "color": "#666"}),
    dcc.RadioItems(
        id="fv-alt-mode",
        options=[
            {"label": " None", "value": "none"},
            {"label": " Custom leadfield path", "value": "custom"},
            {"label": " Generate & save a new leadfield", "value": "generate"},
            {"label": " Run one-off FEM (no leadfield)", "value": "oneoff"},
        ],
        value="none",
        style={"marginBottom": "1rem"},
    ),

    # ── Custom leadfield path ──────────────────────────────────────────────
    html.Div([
        html.H4("Custom Leadfield Path"),
        dcc.Input(id="fv-custom-path", type="text", style={"width": "100%", "maxWidth": "600px"},
                  placeholder="e.g. D:/path/to/some_leadfield.hdf5"),
        html.Div(id="fv-custom-path-note", style={"fontSize": "13px", "margin": "0.5rem 0 1rem"}),

        _channel_block_generic("custom-ch1", "Channel 1"),
        _channel_block_generic("custom-ch2", "Channel 2"),

        html.Div([
            html.Div([
                html.Label("Electrode dims (mm)"),
                dcc.Input(id="fv-custom-dims", type="text", placeholder="e.g. 14, 14",
                          style={"width": "100%"}),
            ], style={"maxWidth": "260px", "marginRight": "1.5rem"}),
            html.Div([
                html.Label("Label"),
                dcc.Input(id="fv-custom-label", type="text", value="setup", style={"width": "100%"}),
            ], style={"maxWidth": "200px"}),
        ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "1rem"}),

        html.Button("Compute TI (custom leadfield)", id="fv-custom-compute-button", n_clicks=0,
                    style={"padding": "0.5rem 1.5rem"}),
        dcc.Loading(html.Div(id="fv-custom-compute-results", style={"marginTop": "1rem"})),
    ], id="fv-alt-custom-container", style={"display": "none"}),

    # ── Generate & save a leadfield ─────────────────────────────────────────
    html.Div([
        html.H4("Generate & Save Leadfield"),
        html.P("One FEM solve per electrode — ~30 minutes typical. Runs in the background; "
               "this page polls for completion.", style={"fontSize": "13px", "color": "#666"}),
        html.Div([
            html.Label("Registered cap (subject space — from Cap Registration)"),
            dcc.Dropdown(id="fv-gen-cap-dropdown", placeholder="Select a registered cap...",
                         style={"maxWidth": "420px"}),
        ], style={"marginBottom": "0.75rem"}),
        html.Div([
            html.Div([
                html.Label("Electrode dims (mm)"),
                dcc.Input(id="fv-gen-dims", type="text", value="14, 14", style={"width": "100%"}),
            ], style={"maxWidth": "180px", "marginRight": "1rem"}),
            html.Div([
                html.Label("Gel thickness (mm)"),
                dcc.Input(id="fv-gen-gel", type="number", value=1.0, step=0.1, style={"width": "100%"}),
            ], style={"maxWidth": "160px", "marginRight": "1rem"}),
            html.Div([
                html.Label("CPUs"),
                dcc.Input(id="fv-gen-cpus", type="number", value=1, step=1, min=1, style={"width": "100%"}),
            ], style={"maxWidth": "120px"}),
        ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "1rem"}),
        html.Button("Start Leadfield Generation", id="fv-gen-start-button", n_clicks=0,
                    style={"padding": "0.5rem 1.5rem"}),
        dcc.Store(id="fv-gen-job-store"),
        dcc.Interval(id="fv-gen-interval", interval=3000, disabled=True),
        html.Div(id="fv-gen-status", style={"marginTop": "1rem"}),
    ], id="fv-alt-generate-container", style={"display": "none"}),

    # ── One-off FEM ──────────────────────────────────────────────────────────
    html.Div([
        html.H4("One-off Full FEM"),
        html.P("Real physics solve per channel (~1 minute each), electrodes placed anywhere on "
               "the scalp — no leadfield needed. Runs in the background; this page polls for "
               "completion.", style={"fontSize": "13px", "color": "#666"}),
        html.Div([
            html.Label("Registered cap (for the electrode-name dropdowns below — optional if "
                       "you only use raw x,y,z)"),
            dcc.Dropdown(id="fv-oneoff-cap-dropdown", placeholder="Select a registered cap...",
                         style={"maxWidth": "420px"}),
        ], style={"marginBottom": "1rem"}),

        _oneoff_channel_block(1),
        _oneoff_channel_block(2),

        html.Div([
            html.Div([
                html.Label("Electrode dims (mm)"),
                dcc.Input(id="fv-oneoff-dims", type="text", value="19.5, 19.5", style={"width": "100%"}),
            ], style={"maxWidth": "180px", "marginRight": "1rem"}),
            html.Div([
                html.Label("Thickness (mm)"),
                dcc.Input(id="fv-oneoff-thickness", type="number", value=4.0, step=0.5,
                          style={"width": "100%"}),
            ], style={"maxWidth": "140px", "marginRight": "1rem"}),
            html.Div([
                html.Label("Label"),
                dcc.Input(id="fv-oneoff-label", type="text", value="manual_fem", style={"width": "100%"}),
            ], style={"maxWidth": "200px"}),
        ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "0.5rem"}),
        dcc.Checklist(id="fv-oneoff-force",
                      options=[{"label": " force (ignore cached result)", "value": "force"}], value=[],
                      style={"marginBottom": "1rem"}),

        html.Button("Start One-off FEM", id="fv-oneoff-start-button", n_clicks=0,
                    style={"padding": "0.5rem 1.5rem"}),
        dcc.Store(id="fv-oneoff-job-store"),
        dcc.Interval(id="fv-oneoff-interval", interval=3000, disabled=True),
        html.Div(id="fv-oneoff-status", style={"marginTop": "1rem"}),
    ], id="fv-alt-oneoff-container", style={"display": "none"}),
])


# ═════════════════════════════════════════════════════════════════════════════
# Subject / cap / leadfield status
# ═════════════════════════════════════════════════════════════════════════════

@callback(Output("fv-subject-dropdown", "options"), Input("fv-subject-dropdown", "id"))
def _load_subjects(_):
    options = []
    for s in fd.discover_subjects():
        status = "m2m ✓" if s["has_m2m"] else "m2m ✗ (needs charm)"
        options.append({"label": f"{s['subject_id']}   [{status}]", "value": s["subject_id"],
                         "disabled": not s["has_m2m"]})
    return options


@callback(
    Output("fv-cap-dropdown", "options"),
    Output("fv-leadfield-note", "children"),
    Input("fv-subject-dropdown", "value"),
)
def _load_caps(subject_id):
    if not subject_id:
        return [], ""
    leadfields = fd.list_leadfields(subject_id)
    if not leadfields:
        return [], html.Span(
            "✗ No precomputed leadfield for this subject — leadfield mode isn't available yet. "
            "(Custom leadfield path / one-off FEM / generate leadfield are planned follow-ups.)",
            style={"color": "#a00"})
    # One option per (cap, electrode-settings) variant — value is the
    # resolved hdf5_path directly, since a cap can now have more than one
    # cached variant (see fem_discovery.list_leadfields).
    options = [{"label": lf["label"], "value": lf["hdf5_path"]} for lf in leadfields]
    return options, ""


@callback(
    Output("fv-roi-dropdown", "options"),
    Output("fv-nonroi-dropdown", "options"),
    Input("fv-subject-dropdown", "value"),
)
def _load_masks(subject_id):
    if not subject_id:
        return [], []
    options = [{"label": m["filename"], "value": m["path"]} for m in discovery.existing_masks(subject_id)]
    return options, options


@callback(
    Output("fv-ch1-plus", "options"), Output("fv-ch1-minus", "options"),
    Output("fv-ch2-plus", "options"), Output("fv-ch2-minus", "options"),
    Output("fv-electrode-dims", "value"),
    Input("fv-subject-dropdown", "value"),
    Input("fv-cap-dropdown", "value"),
)
def _load_electrodes(subject_id, hdf5_path):
    if not subject_id or not hdf5_path or not os.path.isfile(hdf5_path):
        return [], [], [], [], ""
    names = fd.leadfield_electrode_names(hdf5_path)
    options = [{"label": n, "value": n} for n in names]
    dims = fd.leadfield_electrode_dims(hdf5_path)
    dims_str = ", ".join(str(d) for d in dims) if dims else ""
    return options, options, options, options, dims_str


# ═════════════════════════════════════════════════════════════════════════════
# Compute
# ═════════════════════════════════════════════════════════════════════════════

def _parse_dims(text):
    if not text or not text.strip():
        return None
    try:
        parts = [float(p.strip()) for p in text.split(",") if p.strip()]
        return parts if len(parts) == 2 else None
    except ValueError:
        return None


def _cap_name_for_hdf5(subject_id, hdf5_path):
    return next((lf["cap_name"] for lf in fd.list_leadfields(subject_id) if lf["hdf5_path"] == hdf5_path), None)


@callback(
    Output("fv-compute-results", "children"),
    Output("fv-render-context", "data"),
    Input("fv-compute-button", "n_clicks"),
    State("fv-subject-dropdown", "value"),
    State("fv-cap-dropdown", "value"),
    State("fv-roi-dropdown", "value"),
    State("fv-nonroi-dropdown", "value"),
    State("fv-ch1-plus", "value"), State("fv-ch1-minus", "value"), State("fv-ch1-current", "value"),
    State("fv-ch2-plus", "value"), State("fv-ch2-minus", "value"), State("fv-ch2-current", "value"),
    State("fv-electrode-dims", "value"),
    State("fv-label", "value"),
    prevent_initial_call=True,
)
def _on_compute_click(_n_clicks, subject_id, hdf5_path, roi_mask, non_roi_mask,
                       ch1_plus, ch1_minus, ch1_current, ch2_plus, ch2_minus, ch2_current,
                       dims_text, label):
    missing = []
    if not subject_id:
        missing.append("subject")
    if not hdf5_path:
        missing.append("cap")
    if not roi_mask:
        missing.append("ROI mask")
    if not all([ch1_plus, ch1_minus, ch2_plus, ch2_minus]):
        missing.append("channel electrodes")
    if missing:
        return html.Div("Missing: " + ", ".join(missing), style={"color": "#a00"}), None

    result = fd.compute_ti(
        subject_id=subject_id, hdf5_path=hdf5_path,
        roi_mask_path=roi_mask, non_roi_mask_path=non_roi_mask,
        ch1_plus=ch1_plus, ch1_minus=ch1_minus, ch1_current_mA=float(ch1_current),
        ch2_plus=ch2_plus, ch2_minus=ch2_minus, ch2_current_mA=float(ch2_current),
        electrode_dims=_parse_dims(dims_text), label=label or "setup",
    )

    if not result["success"]:
        return html.Div(f"✗ {result['error']}", style={"color": "#a00"}), None

    stats_rows = [{"metric": k, "value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                  for k, v in result["stats"].items()]

    # For the Render Figure section below: everything build_figure() needs,
    # resolved now while we still have hdf5_path/cap_name in hand (the
    # figure re-derives nothing from the leadfield itself — just the
    # electrode-position CSV that cap was registered from).
    cap_name = _cap_name_for_hdf5(subject_id, hdf5_path)
    electrodes_csv = cd.registered_cap_path(subject_id, cap_name) if cap_name else None
    render_context = None
    if cap_name and electrodes_csv and os.path.isfile(electrodes_csv):
        render_context = {
            "subject_id": subject_id, "msh_path": result["msh_path"],
            "ch1": [ch1_plus, ch1_minus], "ch2": [ch2_plus, ch2_minus],
            "electrodes_csv": electrodes_csv, "label": label or "setup",
        }

    return html.Div([
        html.P(f"✓ {result['label']}  —  mesh: {result['msh_path']}", style={"color": "#060"}),
        _styled_table("fv-stats-table", [
            {"name": "Metric", "id": "metric"},
            {"name": "Value", "id": "value"},
        ], data=stats_rows),
    ]), render_context


def _stats_table(result_key, stats):
    rows = [{"metric": k, "value": f"{v:.4f}" if isinstance(v, float) else str(v)}
            for k, v in stats.items()]
    return _styled_table(result_key, [
        {"name": "Metric", "id": "metric"},
        {"name": "Value", "id": "value"},
    ], data=rows)


# ═════════════════════════════════════════════════════════════════════════════
# Load an existing simulation directly — no new compute. Populates
# fv-render-context the same way a successful Compute TI click does, so
# everything downstream (region pickers, Render Figure) works unchanged.
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("fv-viz-existing-dropdown", "options"),
    Output("fv-viz-existing-meshes-store", "data"),
    Input("fv-subject-dropdown", "value"),
)
def _load_existing_meshes(subject_id):
    if not subject_id:
        return [], []
    meshes = vz.list_existing_meshes(subject_id)
    options = [{"label": f"{m['label']}  [{m['source']}]", "value": i} for i, m in enumerate(meshes)]
    return options, meshes


@callback(
    Output("fv-viz-existing-cap-dropdown", "options"),
    Output("fv-viz-existing-cap-dropdown", "value"),
    Output("fv-viz-existing-note", "children"),
    Input("fv-viz-existing-dropdown", "value"),
    State("fv-subject-dropdown", "value"),
    State("fv-viz-existing-meshes-store", "data"),
)
def _load_existing_cap_options(mesh_idx, subject_id, meshes):
    if mesh_idx is None or not meshes or mesh_idx >= len(meshes):
        return [], None, ""
    m = meshes[mesh_idx]
    names = [m["ch1_plus"], m["ch1_minus"], m["ch2_plus"], m["ch2_minus"]]
    caps = vz.find_matching_caps(subject_id, names)
    options = [{"label": c, "value": c} for c in caps]
    if not caps:
        return [], None, html.Div(f"✗ no registered cap for sub-{subject_id} contains all of "
                                  f"{names} — register the right cap first.", style={"color": "#a00"})
    note = (html.Div(f"{len(caps)} caps contain these electrode names — pick the right one.",
                     style={"color": "#a60"}) if len(caps) > 1
           else html.Div("✓ exactly one matching cap.", style={"color": "#060"}))
    return options, (caps[0] if len(caps) == 1 else None), note


@callback(
    Output("fv-viz-load-existing-button", "disabled"),
    Input("fv-viz-existing-dropdown", "value"),
    Input("fv-viz-existing-cap-dropdown", "value"),
)
def _toggle_load_existing_button(mesh_idx, cap_name):
    return mesh_idx is None or not cap_name


@callback(
    Output("fv-render-context", "data", allow_duplicate=True),
    Output("fv-viz-existing-note", "children", allow_duplicate=True),
    Input("fv-viz-load-existing-button", "n_clicks"),
    State("fv-viz-existing-dropdown", "value"),
    State("fv-viz-existing-cap-dropdown", "value"),
    State("fv-subject-dropdown", "value"),
    State("fv-viz-existing-meshes-store", "data"),
    prevent_initial_call=True,
)
def _on_load_existing_click(_n_clicks, mesh_idx, cap_name, subject_id, meshes):
    if mesh_idx is None or not cap_name or not meshes or mesh_idx >= len(meshes):
        return dash.no_update, html.Div("Select a simulation and a cap first.", style={"color": "#a00"})
    m = meshes[mesh_idx]
    electrodes_csv = cd.registered_cap_path(subject_id, cap_name)
    if not os.path.isfile(electrodes_csv):
        return dash.no_update, html.Div(f"Registered cap CSV not found: {electrodes_csv}",
                                        style={"color": "#a00"})
    render_context = {
        "subject_id": subject_id, "msh_path": m["msh_path"],
        "ch1": [m["ch1_plus"], m["ch1_minus"]], "ch2": [m["ch2_plus"], m["ch2_minus"]],
        "electrodes_csv": electrodes_csv, "label": m["label"],
    }
    return render_context, html.Div(f"✓ Loaded {m['label']} ({m['source']}) — pick regions below and Render.",
                                    style={"color": "#060"})


# ═════════════════════════════════════════════════════════════════════════════
# Render Figure (background job — extract + render together, ~15-45s on a
# real subject; too slow for a synchronous Dash callback)
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("fv-viz-highlight-dropdown", "options"),
    Output("fv-viz-render-button", "disabled"),
    Input("fv-render-context", "data"),
)
def _load_viz_regions(render_context):
    if not render_context:
        return [], True
    options = [{"label": r, "value": r} for r in vz.available_regions(render_context["subject_id"])]
    return options, False


@callback(
    Output("fv-viz-job-store", "data"),
    Output("fv-viz-interval", "disabled"),
    Output("fv-viz-results", "children"),
    Input("fv-viz-render-button", "n_clicks"),
    State("fv-render-context", "data"),
    State("fv-viz-highlight-dropdown", "value"),
    State("fv-viz-vmax", "value"),
    prevent_initial_call=True,
)
def _on_render_click(_n_clicks, render_context, highlight_labels, vmax):
    if not render_context:
        return None, True, html.Div("Compute TI above first.", style={"color": "#a00"})
    if not highlight_labels:
        return None, True, html.Div("Select at least one region.", style={"color": "#a00"})

    subject_id = render_context["subject_id"]
    out_path = os.path.join(fd.PROJECT_DIR, "derivatives", "SimNIBS", f"sub-{subject_id}",
                            "comparison", "figures", f"{render_context['label']}_maxTI_3views.png")
    base_dir = os.path.join(fd.PROJECT_DIR, "derivatives", "SimNIBS", f"sub-{subject_id}",
                            "comparison", "figures", "_jobs")
    _job_id, job_dir = jr.new_job_dir(base_dir)
    jr.start_local_job(
        job_dir, vz.build_figure_subprocess, subject_id, render_context["msh_path"],
        tuple(render_context["ch1"]), tuple(render_context["ch2"]), render_context["electrodes_csv"],
        highlight_labels, out_path, float(vmax or 1.0),
    )
    return job_dir, False, html.Div("Rendering — polling every 2s...", style={"color": "#666"})


@callback(
    Output("fv-viz-job-store", "data", allow_duplicate=True),
    Output("fv-viz-interval", "disabled", allow_duplicate=True),
    Output("fv-viz-results", "children", allow_duplicate=True),
    Input("fv-viz-interval", "n_intervals"),
    State("fv-viz-job-store", "data"),
    prevent_initial_call=True,
)
def _poll_viz_job(_n_intervals, job_dir):
    if not job_dir:
        return job_dir, True, dash.no_update
    status = jr.read_status(job_dir)
    if not status or status["state"] == "running":
        return job_dir, False, html.Div("… rendering", style={"color": "#666"})
    if status["state"] == "error":
        return job_dir, True, html.Div(f"✗ {status['error']}", style={"color": "#a00"})

    result = status["result"] or {}
    if not result.get("success"):
        return job_dir, True, html.Div(f"✗ {result.get('error')}", style={"color": "#a00"})

    import base64
    with open(result["out_path"], "rb") as f:
        b64 = base64.b64encode(f.read()).decode()
    fr = result.get("field_range") or {}
    note = (f"✓ drew: {', '.join(result['regions_drawn'])}  —  "
           f"whole-mesh {fr.get('min', 0):.3f}-{fr.get('max', 0):.3f} V/m "
           f"(check your color scale max against this)")
    return job_dir, True, html.Div([
        html.P(note, style={"color": "#060", "fontSize": "13px"}),
        html.Img(src=f"data:image/png;base64,{b64}", style={"maxWidth": "100%"}),
        html.Div(html.Button("Download PNG", id="fv-viz-download-button", n_clicks=0),
                 style={"marginTop": "0.5rem"}),
        dcc.Store(id="fv-viz-out-path", data=result["out_path"]),
    ])


@callback(
    Output("fv-viz-download", "data"),
    Input("fv-viz-download-button", "n_clicks"),
    State("fv-viz-out-path", "data"),
    prevent_initial_call=True,
)
def _on_download_click(_n_clicks, out_path):
    if not out_path:
        return dash.no_update
    return dcc.send_file(out_path)


# ═════════════════════════════════════════════════════════════════════════════
# Slice Viewer (background job — build_slice_frames() is plain numpy/nibabel/
# simnibs.transformations, no VTK, so unlike Render Figure it's safe to call
# directly from job_runner's background thread, no subprocess needed).
# Scrubbing itself is a clientside callback below — once the frames are on
# the browser, moving a slider never touches the server.
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("fv-slice-highlight-dropdown", "options"),
    Output("fv-slice-build-button", "disabled"),
    Input("fv-render-context", "data"),
)
def _load_slice_regions(render_context):
    if not render_context:
        return [], True
    options = [{"label": r, "value": r} for r in vz.available_regions(render_context["subject_id"])]
    return options, False


@callback(
    Output("fv-slice-job-store", "data"),
    Output("fv-slice-interval", "disabled"),
    Output("fv-slice-status", "children"),
    Input("fv-slice-build-button", "n_clicks"),
    State("fv-render-context", "data"),
    State("fv-slice-highlight-dropdown", "value"),
    State("fv-slice-roi-only", "value"),
    prevent_initial_call=True,
)
def _on_build_slices_click(_n_clicks, render_context, highlight_labels, roi_only_value):
    if not render_context:
        return None, True, html.Div("Compute TI (or load an existing simulation) above first.",
                                    style={"color": "#a00"})

    subject_id = render_context["subject_id"]
    base_dir = os.path.join(fd.PROJECT_DIR, "derivatives", "SimNIBS", f"sub-{subject_id}",
                            "comparison", "figures", "_jobs")
    _job_id, job_dir = jr.new_job_dir(base_dir)
    jr.start_local_job(
        job_dir, vz.build_slice_frames, subject_id, render_context["msh_path"], highlight_labels or [],
        roi_only=("roi_only" in (roi_only_value or [])),
    )
    return job_dir, False, html.Div("Building — polling every 2s (interpolating the field onto the "
                                    "T1 grid takes ~20-30s)...", style={"color": "#666"})


@callback(
    Output("fv-slice-job-store", "data", allow_duplicate=True),
    Output("fv-slice-interval", "disabled", allow_duplicate=True),
    Output("fv-slice-status", "children", allow_duplicate=True),
    Output("fv-slice-viewer-container", "style"),
    Output("fv-slice-colorbar-img", "src"),
    Output("fv-slice-legend-row", "children"),
    Output("fv-slice-axial-frames", "data"), Output("fv-slice-coronal-frames", "data"),
    Output("fv-slice-sagittal-frames", "data"),
    Output("fv-slice-axial-slider", "max"), Output("fv-slice-axial-slider", "value"),
    Output("fv-slice-coronal-slider", "max"), Output("fv-slice-coronal-slider", "value"),
    Output("fv-slice-sagittal-slider", "max"), Output("fv-slice-sagittal-slider", "value"),
    Input("fv-slice-interval", "n_intervals"),
    State("fv-slice-job-store", "data"),
    prevent_initial_call=True,
)
def _poll_slice_job(_n_intervals, job_dir):
    no_viewer_change = (dash.no_update,) * 11
    if not job_dir:
        return (job_dir, True, dash.no_update, {"display": "none"}) + no_viewer_change
    status = jr.read_status(job_dir)
    if not status or status["state"] == "running":
        return (job_dir, False, html.Div("… building", style={"color": "#666"}),
               dash.no_update) + no_viewer_change
    if status["state"] == "error":
        return (job_dir, True, html.Div(f"✗ {status['error']}", style={"color": "#a00"}),
               dash.no_update) + no_viewer_change

    result = status["result"] or {}
    if not result.get("success"):
        return (job_dir, True, html.Div(f"✗ {result.get('error')}", style={"color": "#a00"}),
               dash.no_update) + no_viewer_change

    fr = result.get("field_range") or {}
    regions_drawn = result.get("regions_drawn") or []
    regions = ", ".join(regions_drawn) or "(none — no mask matched for this subject)"
    note = html.Div(
        f"✓ outlined: {regions}  —  whole-mesh {fr.get('min', 0):.3f}-{fr.get('max', 0):.3f} V/m, "
        f"color scale to {result.get('vmax_used', 0):.3f} V/m (99th percentile) — see the scale below",
        style={"color": "#060"})

    # Which outline color is which region — same order vz._composite_slice()
    # cycled through SLICE_ROI_COLORS in, so this is just re-reading that
    # order back, not recomputing anything. Matters most for bilateral
    # regions picked as separate L/R masks, where color is the only way to
    # tell which outline is which without this.
    legend = html.Div([
        html.Span([
            html.Span(style={"display": "inline-block", "width": "11px", "height": "11px",
                            "backgroundColor": f"rgb{vz.SLICE_ROI_COLORS[i % len(vz.SLICE_ROI_COLORS)]}",
                            "marginRight": "4px", "verticalAlign": "middle", "borderRadius": "2px"}),
            label,
        ], style={"marginRight": "1.25rem", "whiteSpace": "nowrap"})
        for i, label in enumerate(regions_drawn)
    ], style={"display": "flex", "flexWrap": "wrap"}) if regions_drawn else ""

    frames = result["frames"]
    n = result["n_slices"]
    mid = result["mid_slice"]
    return (
        job_dir, True, note, {"display": "flex", "flexWrap": "wrap", "marginTop": "1rem"},
        f"data:image/png;base64,{result['colorbar_png']}", legend,
        frames["Axial"], frames["Coronal"], frames["Sagittal"],
        n["Axial"] - 1, mid["Axial"], n["Coronal"] - 1, mid["Coronal"], n["Sagittal"] - 1, mid["Sagittal"],
    )


# Clientside: swap the shown slice image when a slider moves, reading from
# the already-downloaded frame list — no server round-trip, so scrubbing
# stays smooth even on a slow connection.
_SLICE_CLIENTSIDE_JS = """
function(idx, frames) {
    if (!frames || idx === undefined || idx === null || !frames[idx]) {
        return window.dash_clientside.no_update;
    }
    return "data:image/png;base64," + frames[idx];
}
"""
for _plane in ("axial", "coronal", "sagittal"):
    dash.clientside_callback(
        _SLICE_CLIENTSIDE_JS,
        Output(f"fv-slice-{_plane}-img", "src"),
        Input(f"fv-slice-{_plane}-slider", "value"),
        State(f"fv-slice-{_plane}-frames", "data"),
    )


# ═════════════════════════════════════════════════════════════════════════════
# Alternative leadfield sources: mode toggle
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("fv-alt-custom-container", "style"),
    Output("fv-alt-generate-container", "style"),
    Output("fv-alt-oneoff-container", "style"),
    Input("fv-alt-mode", "value"),
)
def _toggle_alt_mode(mode):
    hidden = {"display": "none"}
    return (
        {} if mode == "custom" else hidden,
        {} if mode == "generate" else hidden,
        {} if mode == "oneoff" else hidden,
    )


# ═════════════════════════════════════════════════════════════════════════════
# Custom leadfield path
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("fv-custom-path-note", "children"),
    Output("fv-custom-ch1-plus", "options"), Output("fv-custom-ch1-minus", "options"),
    Output("fv-custom-ch2-plus", "options"), Output("fv-custom-ch2-minus", "options"),
    Output("fv-custom-dims", "value"),
    Input("fv-custom-path", "value"),
)
def _validate_custom_path(path):
    if not path or not path.strip():
        return "", [], [], [], [], ""
    path = path.strip()
    if not os.path.isfile(path):
        return html.Span(f"✗ file not found: {path}", style={"color": "#a00"}), [], [], [], [], ""
    try:
        names = fd.leadfield_electrode_names(path)
        dims = fd.leadfield_electrode_dims(path)
    except Exception as e:
        return html.Span(f"✗ could not read leadfield: {e}", style={"color": "#a00"}), [], [], [], [], ""
    options = [{"label": n, "value": n} for n in names]
    dims_str = ", ".join(str(d) for d in dims) if dims else ""
    note = html.Span(f"✓ {len(names)} electrodes", style={"color": "#060"})
    return note, options, options, options, options, dims_str


@callback(
    Output("fv-custom-compute-results", "children"),
    Input("fv-custom-compute-button", "n_clicks"),
    State("fv-subject-dropdown", "value"),
    State("fv-custom-path", "value"),
    State("fv-roi-dropdown", "value"),
    State("fv-nonroi-dropdown", "value"),
    State("fv-custom-ch1-plus", "value"), State("fv-custom-ch1-minus", "value"),
    State("fv-custom-ch1-current", "value"),
    State("fv-custom-ch2-plus", "value"), State("fv-custom-ch2-minus", "value"),
    State("fv-custom-ch2-current", "value"),
    State("fv-custom-dims", "value"),
    State("fv-custom-label", "value"),
    prevent_initial_call=True,
)
def _on_custom_compute_click(_n_clicks, subject_id, hdf5_path, roi_mask, non_roi_mask,
                              ch1_plus, ch1_minus, ch1_current, ch2_plus, ch2_minus, ch2_current,
                              dims_text, label):
    missing = []
    if not subject_id:
        missing.append("subject")
    if not hdf5_path:
        missing.append("leadfield path")
    if not roi_mask:
        missing.append("ROI mask")
    if not all([ch1_plus, ch1_minus, ch2_plus, ch2_minus]):
        missing.append("channel electrodes")
    if missing:
        return html.Div("Missing: " + ", ".join(missing), style={"color": "#a00"})

    result = fd.compute_ti_custom_leadfield(
        subject_id=subject_id, hdf5_path=hdf5_path,
        roi_mask_path=roi_mask, non_roi_mask_path=non_roi_mask,
        ch1_plus=ch1_plus, ch1_minus=ch1_minus, ch1_current_mA=float(ch1_current),
        ch2_plus=ch2_plus, ch2_minus=ch2_minus, ch2_current_mA=float(ch2_current),
        electrode_dims=_parse_dims(dims_text), label=label or "setup",
    )
    if not result["success"]:
        return html.Div(f"✗ {result['error']}", style={"color": "#a00"})
    return html.Div([
        html.P(f"✓ {result['label']}  —  mesh: {result['msh_path']}", style={"color": "#060"}),
        _stats_table("fv-custom-stats-table", result["stats"]),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# Generate & save a leadfield (background job)
# ═════════════════════════════════════════════════════════════════════════════

@callback(Output("fv-gen-cap-dropdown", "options"), Input("fv-subject-dropdown", "value"))
def _load_gen_caps(subject_id):
    if not subject_id:
        return []
    return [{"label": c["name"], "value": c["path"]} for c in cd.list_registered_caps(subject_id)]


@callback(
    Output("fv-gen-job-store", "data"),
    Output("fv-gen-interval", "disabled"),
    Output("fv-gen-status", "children"),
    Input("fv-gen-start-button", "n_clicks"),
    State("fv-subject-dropdown", "value"),
    State("fv-gen-cap-dropdown", "value"),
    State("fv-gen-dims", "value"),
    State("fv-gen-gel", "value"),
    State("fv-gen-cpus", "value"),
    prevent_initial_call=True,
)
def _on_start_generate(_n_clicks, subject_id, cap_path, dims_text, gel, cpus):
    if not subject_id or not cap_path:
        return None, True, html.Div("Select a subject and a registered cap first.", style={"color": "#a00"})

    dims = _parse_dims(dims_text) or [14.0, 14.0]
    base_dir = os.path.join(discovery.PROJECT_DIR, "derivatives", "SimNIBS",
                             f"sub-{subject_id}", "leadfield_volume", "_jobs")
    job_id, job_dir = jr.new_job_dir(base_dir)
    jr.start_local_job(job_dir, fd.generate_leadfield, subject_id, cap_path,
                        "ellipse", tuple(dims), float(gel or 1.0), int(cpus or 1))
    return job_dir, False, html.Div("Started — polling every 3s (this can take a while)...",
                                     style={"color": "#666"})


@callback(
    Output("fv-gen-status", "children", allow_duplicate=True),
    Output("fv-gen-interval", "disabled", allow_duplicate=True),
    Input("fv-gen-interval", "n_intervals"),
    State("fv-gen-job-store", "data"),
    prevent_initial_call=True,
)
def _poll_generate(_n_intervals, job_dir):
    if not job_dir:
        return "", True
    status = jr.read_status(job_dir)
    if not status:
        return html.Div("Waiting for job to start...", style={"color": "#666"}), False
    if status["state"] == "running":
        elapsed = int(time.time() - status["started_at"])
        return html.Div(f"Running... ({elapsed}s elapsed — a real leadfield takes ~30 min; "
                        f"only an exact-params cache hit is instant)", style={"color": "#666"}), False
    if status["state"] == "error":
        return html.Div(f"✗ {status['error']}", style={"color": "#a00"}), True
    result = status["result"]
    if not result.get("success"):
        return html.Div(f"✗ {result.get('error')}", style={"color": "#a00"}), True
    cached_note = " (was already cached)" if result.get("cached") else ""
    params_line = f"  Params used: {result.get('params_used')}"
    return html.Div([
        html.P(f"✓ Leadfield ready{cached_note} → {result['hdf5_path']}. "
               f"Select its cap above (in the main section) to compute TI.", style={"color": "#060"}),
        html.P(params_line, style={"fontSize": "12px", "color": "#666"}),
    ]), True


# ═════════════════════════════════════════════════════════════════════════════
# One-off FEM (background job)
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("fv-oneoff-cap-dropdown", "options"),
    Output("fv-oneoff-ch1-plus-name", "options"), Output("fv-oneoff-ch1-minus-name", "options"),
    Output("fv-oneoff-ch2-plus-name", "options"), Output("fv-oneoff-ch2-minus-name", "options"),
    Input("fv-subject-dropdown", "value"),
    Input("fv-oneoff-cap-dropdown", "value"),
)
def _load_oneoff_electrodes(subject_id, cap_path):
    if not subject_id:
        return [], [], [], [], []
    cap_options = [{"label": c["name"], "value": c["path"]} for c in cd.list_registered_caps(subject_id)]
    if not cap_path:
        return cap_options, [], [], [], []
    elec = cd.registered_electrode_positions(cap_path)
    name_options = [{"label": n, "value": n} for n in elec["names"]]
    return cap_options, name_options, name_options, name_options, name_options


def _resolve_oneoff_coord(cap_path, name, raw_text):
    """raw x,y,z text overrides a name selection if both are set."""
    if raw_text and raw_text.strip():
        try:
            parts = [float(p.strip()) for p in raw_text.split(",")]
            if len(parts) == 3:
                return parts, None
        except ValueError:
            pass
        return None, f"invalid raw coordinate: '{raw_text}' (expected 'x, y, z')"
    if name and cap_path:
        elec = cd.registered_electrode_positions(cap_path)
        if name in elec["names"]:
            idx = elec["names"].index(name)
            return elec["coords"][idx].tolist(), None
        return None, f"'{name}' not found in the selected registered cap"
    return None, "no electrode selected (pick a name or enter raw x, y, z)"


@callback(
    Output("fv-oneoff-job-store", "data"),
    Output("fv-oneoff-interval", "disabled"),
    Output("fv-oneoff-status", "children"),
    Input("fv-oneoff-start-button", "n_clicks"),
    State("fv-subject-dropdown", "value"),
    State("fv-roi-dropdown", "value"), State("fv-nonroi-dropdown", "value"),
    State("fv-oneoff-cap-dropdown", "value"),
    State("fv-oneoff-ch1-plus-name", "value"), State("fv-oneoff-ch1-plus-raw", "value"),
    State("fv-oneoff-ch1-minus-name", "value"), State("fv-oneoff-ch1-minus-raw", "value"),
    State("fv-oneoff-ch1-current", "value"),
    State("fv-oneoff-ch2-plus-name", "value"), State("fv-oneoff-ch2-plus-raw", "value"),
    State("fv-oneoff-ch2-minus-name", "value"), State("fv-oneoff-ch2-minus-raw", "value"),
    State("fv-oneoff-ch2-current", "value"),
    State("fv-oneoff-dims", "value"), State("fv-oneoff-thickness", "value"),
    State("fv-oneoff-label", "value"), State("fv-oneoff-force", "value"),
    prevent_initial_call=True,
)
def _on_start_oneoff(_n_clicks, subject_id, roi_mask, non_roi_mask, cap_path,
                      ch1p_name, ch1p_raw, ch1m_name, ch1m_raw, ch1_current,
                      ch2p_name, ch2p_raw, ch2m_name, ch2m_raw, ch2_current,
                      dims_text, thickness, label, force_value):
    if not subject_id or not roi_mask:
        return None, True, html.Div("Select a subject and ROI mask first.", style={"color": "#a00"})

    coords = {}
    errors = []
    for key, name, raw in [("ch1_plus", ch1p_name, ch1p_raw), ("ch1_minus", ch1m_name, ch1m_raw),
                            ("ch2_plus", ch2p_name, ch2p_raw), ("ch2_minus", ch2m_name, ch2m_raw)]:
        coord, err = _resolve_oneoff_coord(cap_path, name, raw)
        if err:
            errors.append(f"{key}: {err}")
        else:
            coords[key] = coord
    if errors:
        return None, True, html.Div("Electrode errors — " + "; ".join(errors), style={"color": "#a00"})

    dims = _parse_dims(dims_text) or [19.5, 19.5]
    base_dir = os.path.join(discovery.PROJECT_DIR, "derivatives", "SimNIBS",
                             f"sub-{subject_id}", "comparison", "manual_fem", "_jobs")
    job_id, job_dir = jr.new_job_dir(base_dir)
    jr.start_local_job(
        job_dir, fd.run_one_off_fem,
        subject_id, roi_mask, non_roi_mask,
        coords["ch1_plus"], coords["ch1_minus"], float(ch1_current),
        coords["ch2_plus"], coords["ch2_minus"], float(ch2_current),
        label or "manual_fem", tuple(dims), float(thickness or 4.0),
        "force" in (force_value or []),
    )
    return job_dir, False, html.Div("Started — polling every 3s (~1min/channel typical)...",
                                     style={"color": "#666"})


@callback(
    Output("fv-oneoff-status", "children", allow_duplicate=True),
    Output("fv-oneoff-interval", "disabled", allow_duplicate=True),
    Input("fv-oneoff-interval", "n_intervals"),
    State("fv-oneoff-job-store", "data"),
    prevent_initial_call=True,
)
def _poll_oneoff(_n_intervals, job_dir):
    if not job_dir:
        return "", True
    status = jr.read_status(job_dir)
    if not status:
        return html.Div("Waiting for job to start...", style={"color": "#666"}), False
    if status["state"] == "running":
        elapsed = int(time.time() - status["started_at"])
        return html.Div(f"Running... ({elapsed}s elapsed — cached channels return almost "
                        f"instantly, a real solve is ~1 min/channel)", style={"color": "#666"}), False
    if status["state"] == "error":
        return html.Div(f"✗ {status['error']}", style={"color": "#a00"}), True
    result = status["result"]
    if not result.get("success"):
        return html.Div(f"✗ {result.get('error')}", style={"color": "#a00"}), True
    return html.Div([
        html.P(f"✓ ch1: {result['ch1_msh']}", style={"color": "#060", "fontSize": "12px"}),
        html.P(f"✓ ch2: {result['ch2_msh']}", style={"color": "#060", "fontSize": "12px"}),
        _stats_table("fv-oneoff-stats-table", result["stats"]),
    ]), True

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


@callback(
    Output("fv-compute-results", "children"),
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
        return html.Div("Missing: " + ", ".join(missing), style={"color": "#a00"})

    result = fd.compute_ti(
        subject_id=subject_id, hdf5_path=hdf5_path,
        roi_mask_path=roi_mask, non_roi_mask_path=non_roi_mask,
        ch1_plus=ch1_plus, ch1_minus=ch1_minus, ch1_current_mA=float(ch1_current),
        ch2_plus=ch2_plus, ch2_minus=ch2_minus, ch2_current_mA=float(ch2_current),
        electrode_dims=_parse_dims(dims_text), label=label or "setup",
    )

    if not result["success"]:
        return html.Div(f"✗ {result['error']}", style={"color": "#a00"})

    stats_rows = [{"metric": k, "value": f"{v:.4f}" if isinstance(v, float) else str(v)}
                  for k, v in result["stats"].items()]

    return html.Div([
        html.P(f"✓ {result['label']}  —  mesh: {result['msh_path']}", style={"color": "#060"}),
        _styled_table("fv-stats-table", [
            {"name": "Metric", "id": "metric"},
            {"name": "Value", "id": "value"},
        ], data=stats_rows),
    ])


def _stats_table(result_key, stats):
    rows = [{"metric": k, "value": f"{v:.4f}" if isinstance(v, float) else str(v)}
            for k, v in stats.items()]
    return _styled_table(result_key, [
        {"name": "Metric", "id": "metric"},
        {"name": "Value", "id": "value"},
    ], data=rows)


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

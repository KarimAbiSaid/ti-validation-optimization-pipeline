"""
pages/leadfield_generation.py — Phase 3: generate + cache a full leadfield
(TDCSLEADFIELD) for one subject/cap/electrode-geometry combination, so
later leadfield-mode compute (FEM Validation, Comparison, Run Pipeline's
own exhaustive search) has one ready to use.

Both Local (job_runner background thread, fem_discovery.generate_leadfield)
and SCITAS (fem_discovery.run_leadfield_on_scitas — submits
generate_leadfield_scitas.sbatch, blocks on the SLURM queue, scp's the
result back) fit the exact same job_runner contract, so this page's polling
UI doesn't care which one ran — same pattern as pages/head_modeling.py.

A leadfield is cached by (cap, shape, dimensions, gel_thickness) — see
config.leadfield_tag() — regardless of which side (local or SCITAS) built
it, so switching Run-on between the two for the same settings is a no-op
once one side has already produced it and it's been synced across.
"""
import os

import dash
from dash import html, dcc, dash_table, callback, Input, Output, State

import cap_discovery as cd
import fem_discovery as fd
import job_runner as jr

dash.register_page(__name__, path="/leadfield", name="Leadfield Generation", category="Simulation", order=2)


def _styled_table(id_, columns, data=None, **kwargs):
    return dash_table.DataTable(
        id=id_,
        columns=columns,
        data=data or [],
        style_cell={"textAlign": "left", "fontFamily": "monospace", "fontSize": "13px", "padding": "4px"},
        style_table={"overflowX": "auto"},
        **kwargs,
    )


def _parse_dims(text):
    try:
        parts = [float(p.strip()) for p in (text or "").split(",")]
        return tuple(parts) if len(parts) == 2 else None
    except ValueError:
        return None


layout = html.Div([
    html.H2("Leadfield Generation"),
    html.P("Builds a leadfield (TDCSLEADFIELD) for one subject/cap/electrode-geometry combination — "
           "the prerequisite for fast leadfield-mode compute elsewhere (FEM Validation, Comparison, "
           "Run Pipeline's exhaustive search). Cached by (cap, shape, dimensions, gel thickness) — "
           "re-running with identical settings is instant.",
           style={"fontSize": "13px", "color": "#666"}),

    html.Div([
        html.Div([
            html.Label("Subject"),
            dcc.Dropdown(id="lg-subject-dropdown", placeholder="Select subject...", style={"maxWidth": "260px"}),
        ], style={"marginRight": "1.5rem"}),
        html.Div([
            html.Label("Registered cap"),
            dcc.Dropdown(id="lg-cap-dropdown", placeholder="Select cap...", style={"maxWidth": "300px"}),
        ]),
    ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "1rem"}),

    html.Div(id="lg-status-note", style={"fontSize": "13px", "marginBottom": "1rem"}),

    html.Div([
        html.Div([
            html.Label("Shape"),
            dcc.Dropdown(id="lg-shape", options=[{"label": "ellipse", "value": "ellipse"},
                                                  {"label": "rect", "value": "rect"}],
                        value="ellipse", clearable=False, style={"width": "140px"}),
        ], style={"marginRight": "1.5rem"}),
        html.Div([
            html.Label("Dimensions (mm, W, H)"),
            dcc.Input(id="lg-dims", type="text", value="19.5, 19.5", style={"width": "140px"}),
        ], style={"marginRight": "1.5rem"}),
        html.Div([
            html.Label("Gel thickness (mm)"),
            dcc.Input(id="lg-gel", type="number", value=1.0, step=0.1, style={"width": "100px"}),
        ], style={"marginRight": "1.5rem"}),
        html.Div([
            html.Label("CPUs (local only)"),
            dcc.Input(id="lg-cpus", type="number", value=1, step=1, min=1, style={"width": "100px"}),
        ]),
    ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "1rem"}),

    dcc.Checklist(id="lg-force", options=[{"label": " Force (recompute even if cached)", "value": "force"}],
                  value=[], style={"marginBottom": "1rem"}),

    html.Div([
        html.Label("Run on"),
        dcc.RadioItems(
            id="lg-run-location",
            options=[
                {"label": " Local (this machine)", "value": "local"},
                {"label": " SCITAS (jed.hpc.epfl.ch) — fixed 8 CPUs, set by the sbatch script",
                 "value": "scitas"},
            ],
            value="local",
        ),
    ], style={"marginBottom": "1rem"}),

    html.Button("Generate Leadfield", id="lg-start-button", n_clicks=0, style={"padding": "0.5rem 1.5rem"}),

    dcc.Store(id="lg-job-store"),
    dcc.Interval(id="lg-poll-interval", interval=3000, disabled=True),
    dcc.Loading(html.Div(id="lg-results", style={"marginTop": "1rem"})),

    html.H3("Already generated for this subject", style={"marginTop": "2rem"}),
    _styled_table("lg-existing-table", [
        {"name": "Cap", "id": "cap_name"},
        {"name": "Shape", "id": "shape"},
        {"name": "Dimensions", "id": "dimensions"},
        {"name": "Gel (mm)", "id": "gel_thickness"},
        {"name": "Path", "id": "hdf5_path"},
    ]),
])


@callback(Output("lg-subject-dropdown", "options"), Input("lg-subject-dropdown", "id"))
def _load_subjects(_):
    return [{"label": s["subject_id"], "value": s["subject_id"]}
            for s in fd.discover_subjects() if os.path.isdir(fd.get_m2m_path(s["subject_id"]))]


@callback(Output("lg-cap-dropdown", "options"), Input("lg-subject-dropdown", "value"))
def _load_caps(subject_id):
    if not subject_id:
        return []
    return [{"label": c["name"], "value": c["name"]} for c in cd.list_registered_caps(subject_id)]


@callback(Output("lg-status-note", "children"), Input("lg-subject-dropdown", "value"))
def _update_status_note(subject_id):
    if not subject_id:
        return ""
    if not cd.list_registered_caps(subject_id):
        return html.Div("No registered caps for this subject yet — register one on the Cap "
                        "Registration page first.", style={"color": "#a60"})
    return ""


@callback(Output("lg-existing-table", "data"), Input("lg-subject-dropdown", "value"), Input("lg-job-store", "data"))
def _update_existing_table(subject_id, _jobs):
    if not subject_id:
        return []
    rows = []
    for lf in fd.list_leadfields(subject_id):
        p = lf["params"] or {}
        rows.append({
            "cap_name": lf["cap_name"],
            "shape": p.get("shape", "—" if lf["tag"] is None else ""),
            "dimensions": "x".join(str(d) for d in p["dimensions"]) if p.get("dimensions") else "—",
            "gel_thickness": p.get("gel_thickness", "—"),
            "hdf5_path": lf["hdf5_path"],
        })
    return rows


@callback(
    Output("lg-job-store", "data"),
    Output("lg-poll-interval", "disabled"),
    Output("lg-results", "children"),
    Input("lg-start-button", "n_clicks"),
    State("lg-subject-dropdown", "value"),
    State("lg-cap-dropdown", "value"),
    State("lg-shape", "value"),
    State("lg-dims", "value"),
    State("lg-gel", "value"),
    State("lg-cpus", "value"),
    State("lg-force", "value"),
    State("lg-run-location", "value"),
    prevent_initial_call=True,
)
def _on_start_click(_n_clicks, subject_id, cap_name, shape, dims_text, gel, cpus, force_value, run_location):
    if not subject_id or not cap_name:
        return None, True, html.Div("Select a subject and a registered cap first.", style={"color": "#a00"})
    dims = _parse_dims(dims_text)
    if not dims:
        return None, True, html.Div("Dimensions must be two comma-separated numbers, e.g. \"19.5, 19.5\".",
                                    style={"color": "#a00"})
    force = "force" in (force_value or [])

    base_dir = os.path.join(fd.PROJECT_DIR, "derivatives", "SimNIBS", f"sub-{subject_id}",
                            "leadfield_volume", "_jobs")
    _job_id, job_dir = jr.new_job_dir(base_dir)

    if run_location == "scitas":
        jr.start_local_job(job_dir, fd.run_leadfield_on_scitas, subject_id, cap_name,
                           shape, dims, float(gel or 1.0), force)
        note = ("Submitted to SCITAS — polling every 3s (a real leadfield takes a while; "
               "only an exact-params cache hit is instant)...")
    else:
        cap_path = cd.registered_cap_path(subject_id, cap_name)
        jr.start_local_job(job_dir, fd.generate_leadfield, subject_id, cap_path,
                           shape, dims, float(gel or 1.0), int(cpus or 1), force)
        note = ("Running locally — polling every 3s (a real leadfield takes ~30 min; "
               "only an exact-params cache hit is instant)...")

    return job_dir, False, html.Div(note, style={"color": "#666"})


@callback(
    Output("lg-job-store", "data", allow_duplicate=True),
    Output("lg-poll-interval", "disabled", allow_duplicate=True),
    Output("lg-results", "children", allow_duplicate=True),
    Input("lg-poll-interval", "n_intervals"),
    State("lg-job-store", "data"),
    prevent_initial_call=True,
)
def _poll_job(_n_intervals, job_dir):
    if not job_dir:
        return job_dir, True, dash.no_update
    status = jr.read_status(job_dir)
    if not status or status["state"] == "running":
        return job_dir, False, html.Div("… running", style={"color": "#666"})
    if status["state"] == "error":
        return job_dir, True, html.Div(f"✗ {status['error']}", style={"color": "#a00"})

    result = status["result"] or {}
    if not result.get("success"):
        return job_dir, True, html.Div(f"✗ {result.get('error')}", style={"color": "#a00"})
    cached_note = " (was already cached)" if result.get("cached") else ""
    return job_dir, True, html.Div([
        html.P(f"✓ Leadfield ready{cached_note} → {result['hdf5_path']}", style={"color": "#060"}),
        html.P(f"Params used: {result.get('params_used')}", style={"fontSize": "12px", "color": "#666"}),
    ])

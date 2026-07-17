"""
pages/run_optimization.py — Phase 6: run code/pipeline/run_pipeline.py
against an already-generated config (from Config Generation or
generate_configs.py) — either locally (subprocess, job_runner background
job) or on SCITAS (run_discovery.run_pipeline_on_scitas — uploads the config
plus any missing prerequisites, submits simnibs_ti_pipeline.sbatch, blocks
on the SLURM queue; results stay on SCITAS scratch, sync them back via the
Data Directory page). Both fit the same job_runner contract, so this page's
polling UI doesn't care which one ran.

Single config at a time by design: a full run (leadfield + cap search +
analysis/viz) is long even locally, and each config is already one
subject/ROI/goal unit — same one-job-at-a-time rationale as Head Modeling's
charm page. Poll + final log, not live streaming: same pattern as that page.
"""
import json
import time

import dash
from dash import html, dcc, dash_table, callback, Input, Output, State

import run_discovery as rd
import job_runner as jr

dash.register_page(__name__, path="/run-pipeline", name="Run Pipeline", category="Optimization", order=2)


def _styled_table(id_, columns, data=None, **kwargs):
    return dash_table.DataTable(
        id=id_,
        columns=columns,
        data=data or [],
        style_cell={"textAlign": "left", "fontFamily": "monospace", "fontSize": "13px", "padding": "4px"},
        style_table={"overflowX": "auto"},
        **kwargs,
    )


layout = html.Div([
    html.H2("Run Pipeline"),
    html.P("Runs an already-generated config from code/pipeline/configs/ (see Config Generation) "
           "through run_pipeline.py. Locally, project_dir/cap_csv/bna_atlas_path are overridden to "
           "this machine's paths — the config file on disk is not modified.",
           style={"fontSize": "13px", "color": "#666"}),

    html.Div([
        html.Label("Config"),
        dcc.Dropdown(id="rp-config-dropdown", placeholder="Select a config..."),
    ], style={"maxWidth": "600px", "marginBottom": "1rem"}),

    html.Div(id="rp-config-details", style={"marginBottom": "1rem", "fontSize": "13px"}),

    dcc.Checklist(
        id="rp-force",
        options=[{"label": " Force re-run of every section (even if output already exists)",
                 "value": "force"}],
        value=[],
        style={"marginBottom": "1rem"},
    ),

    html.Div([
        html.Label("Run on"),
        dcc.RadioItems(
            id="rp-run-location",
            options=[
                {"label": " Local (this machine)", "value": "local"},
                {"label": " SCITAS (jed.hpc.epfl.ch)", "value": "scitas"},
            ],
            value="local",
        ),
    ], style={"marginBottom": "1rem"}),

    html.Button("Start Run", id="rp-start-button", n_clicks=0, disabled=True,
               style={"padding": "0.5rem 1.5rem"}),

    dcc.Store(id="rp-job-store"),
    dcc.Interval(id="rp-poll-interval", interval=3000, disabled=True),
    dcc.Loading(html.Div(id="rp-run-status", style={"marginTop": "1rem"})),

    html.H3("Available Configs", style={"marginTop": "2rem"}),
    _styled_table("rp-configs-table", [
        {"name": "Subject", "id": "subject_id"},
        {"name": "ROI", "id": "roi_name"},
        {"name": "Goal", "id": "goal"},
        {"name": "File", "id": "filename"},
    ]),
])


@callback(
    Output("rp-config-dropdown", "options"),
    Output("rp-configs-table", "data"),
    Input("rp-config-dropdown", "id"),
)
def _load_configs(_):
    configs = rd.list_configs()
    options = [{"label": c["filename"], "value": c["path"]} for c in configs]
    return options, configs


@callback(
    Output("rp-config-details", "children"),
    Output("rp-start-button", "disabled"),
    Input("rp-config-dropdown", "value"),
)
def _update_config_details(config_path):
    if not config_path:
        return "", True
    with open(config_path) as f:
        cfg = json.load(f)
    lines = [
        html.P(f"Subject: {cfg['subject_id']}   ROI: {cfg['roi']['name']}   "
               f"non-ROI: {(cfg.get('non_roi') or {}).get('name', 'None')}   Goal: {cfg['optimizer']['goal']}"),
        html.P("Sections: " + ", ".join(k for k, v in cfg["flags"].items() if v)),
    ]
    return html.Div(lines), False


@callback(
    Output("rp-job-store", "data"),
    Output("rp-poll-interval", "disabled"),
    Output("rp-run-status", "children"),
    Input("rp-start-button", "n_clicks"),
    State("rp-config-dropdown", "value"),
    State("rp-force", "value"),
    State("rp-run-location", "value"),
    prevent_initial_call=True,
)
def _on_start_click(_n_clicks, config_path, force_value, run_location):
    if not config_path:
        return None, True, html.Div("Select a config first.", style={"color": "#a00"})

    force_sections = list(rd.FORCE_SECTIONS) if "force" in (force_value or []) else None

    with open(config_path) as f:
        subject_id = json.load(f)["subject_id"]

    _job_id, job_dir = jr.new_job_dir(rd.job_base_dir(subject_id))

    if run_location == "scitas":
        jr.start_local_job(job_dir, rd.run_pipeline_on_scitas, config_path, force_sections)
        note = ("Started on SCITAS — polling every 3s (uploads the config/prerequisites, submits the "
                "job, then waits on the SLURM queue; results stay on SCITAS scratch — sync them back "
                "via the Data Directory page when done).")
    else:
        jr.start_local_job(job_dir, rd.run_local, config_path, force_sections)
        note = ("Started — polling every 3s (a full run can take a while: leadfield ~30 min if not "
                "cached, then a few minutes per ROI/goal combo)...")
    return job_dir, False, html.Div(note, style={"color": "#666"})


@callback(
    Output("rp-run-status", "children", allow_duplicate=True),
    Output("rp-poll-interval", "disabled", allow_duplicate=True),
    Input("rp-poll-interval", "n_intervals"),
    State("rp-job-store", "data"),
    prevent_initial_call=True,
)
def _poll_run(_n_intervals, job_dir):
    if not job_dir:
        return "", True
    status = jr.read_status(job_dir)
    if not status:
        return html.Div("Waiting for job to start...", style={"color": "#666"}), False
    if status["state"] == "running":
        elapsed = int(time.time() - status["started_at"])
        return html.Div(f"Running... ({elapsed}s elapsed)", style={"color": "#666"}), False
    if status["state"] == "error":
        return html.Pre(f"✗ {status['error']}\n\n{status.get('traceback', '')}",
                        style={"color": "#a00", "whiteSpace": "pre-wrap", "fontSize": "12px"}), True

    result = status["result"]
    # run_local() returns {"returncode","stdout","stderr"}; run_pipeline_on_scitas()
    # returns {"job_id","stdout","error"} — render whichever fields are present
    # rather than assuming one fixed shape.
    if result["success"]:
        header_text = "✓ Run finished successfully."
        if result.get("job_id"):
            header_text += f"  (SCITAS job {result['job_id']})"
    elif "returncode" in result:
        header_text = f"✗ Run failed (exit code {result['returncode']})."
    else:
        job_id = result.get("job_id")
        header_text = f"✗ Run failed (SCITAS job {job_id})." if job_id else "✗ Run failed."
    header = html.P(header_text, style={"color": "#060" if result["success"] else "#a00", "fontWeight": "bold"})

    log_children = [html.Summary("stdout / stderr")]
    if result.get("stdout"):
        log_children.append(html.Pre(result["stdout"], style={"fontSize": "12px", "whiteSpace": "pre-wrap"}))
    if result.get("stderr"):
        log_children.append(html.Pre(result["stderr"],
                                     style={"fontSize": "12px", "whiteSpace": "pre-wrap", "color": "#a00"}))
    if result.get("error"):
        log_children.append(html.Pre(result["error"],
                                     style={"fontSize": "12px", "whiteSpace": "pre-wrap", "color": "#a00"}))
    log = html.Details(log_children)
    return html.Div([header, log]), True

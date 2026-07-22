"""
pages/run_optimization.py — Phase 6: run code/pipeline/run_pipeline.py
against one or more already-generated configs (from Config Generation or
generate_configs.py) — either locally (subprocess, job_runner background
job) or on SCITAS (run_discovery.run_pipeline_on_scitas — uploads the config
plus any missing prerequisites, submits simnibs_ti_pipeline.sbatch, blocks
on the SLURM queue; results stay on SCITAS scratch, sync them back via the
Data Directory page). Both fit the same job_runner contract, so this page's
polling UI doesn't care which one ran.

Multiple configs run concurrently, one independent job_runner background
job each (job_runner.start_local_job has no single-job limit — each job is
its own daemon thread) — same jobs-store/poll pattern as pages/comparison.py.
For SCITAS, "concurrently" means each config's own upload + sbatch submit +
SLURM-queue polling happens in its own thread; SSH connection multiplexing
(scitas_discovery._mux_args) keeps that from opening a fresh handshake per
call across all of them. Force/run-location are shared settings applied to
every config in a given batch, not per-config — same convention as Config
Generation's shared settings applied to multiple selected subjects.
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
    html.P("Runs already-generated config(s) from code/pipeline/configs/ (see Config Generation) "
           "through run_pipeline.py. Locally, project_dir/cap_csv/bna_atlas_path are overridden to "
           "this machine's paths — the config file on disk is not modified.",
           style={"fontSize": "13px", "color": "#666"}),

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

    html.H3("Available Configs", style={"marginTop": "1rem"}),
    html.P("Select one or more configs — each starts as its own independent job (local: one "
           "background thread each; SCITAS: one sbatch submission each, run concurrently on the "
           "cluster) using the Force/Run-on settings above.",
           style={"fontSize": "12px", "color": "#666"}),
    _styled_table("rp-configs-table", [
        {"name": "Subject", "id": "subject_id"},
        {"name": "ROI", "id": "roi_name"},
        {"name": "Goal", "id": "goal"},
        {"name": "File", "id": "filename"},
    ], row_selectable="multi", selected_rows=[]),

    html.Button("Start Selected Runs", id="rp-start-button", n_clicks=0, disabled=True,
               style={"padding": "0.5rem 1.5rem", "marginTop": "1rem"}),

    dcc.Store(id="rp-jobs-store"),
    dcc.Interval(id="rp-poll-interval", interval=3000, disabled=True),
    html.Div(id="rp-run-note", style={"fontSize": "13px", "margin": "1rem 0"}),
    dcc.Loading(html.Div(id="rp-results", style={"marginTop": "1rem"})),
])


@callback(
    Output("rp-configs-table", "data"),
    Input("rp-configs-table", "id"),
)
def _load_configs(_):
    return rd.list_configs()


@callback(
    Output("rp-start-button", "disabled"),
    Input("rp-configs-table", "selected_rows"),
)
def _update_start_button(selected_rows):
    return not selected_rows


@callback(
    Output("rp-jobs-store", "data"),
    Output("rp-poll-interval", "disabled"),
    Output("rp-run-note", "children"),
    Input("rp-start-button", "n_clicks"),
    State("rp-configs-table", "data"),
    State("rp-configs-table", "selected_rows"),
    State("rp-force", "value"),
    State("rp-run-location", "value"),
    prevent_initial_call=True,
)
def _on_start_click(_n_clicks, rows, selected_rows, force_value, run_location):
    rows = rows or []
    selected_rows = selected_rows or []
    selected = [rows[i] for i in selected_rows if i < len(rows)]
    if not selected:
        return None, True, html.Div("Select at least one config first.", style={"color": "#a00"})

    force_sections = list(rd.FORCE_SECTIONS) if "force" in (force_value or []) else None
    fn = rd.run_pipeline_on_scitas if run_location == "scitas" else rd.run_local

    jobs = {}
    for i, row in enumerate(selected):
        config_path = row["path"]
        subject_id = row["subject_id"]
        _job_id, job_dir = jr.new_job_dir(rd.job_base_dir(subject_id))
        jr.start_local_job(job_dir, fn, config_path, force_sections)
        jobs[str(i)] = {
            "job_dir": job_dir, "status": "running", "result": None,
            "row": {"subject_id": subject_id, "roi_name": row.get("roi_name"),
                    "goal": row.get("goal"), "filename": row.get("filename")},
        }

    loc_note = ("SCITAS — each config uploads its own prerequisites and submits its own SLURM job; "
               "they run concurrently on the cluster." if run_location == "scitas"
               else "locally, each in its own background thread, concurrently.")
    note = f"Started {len(jobs)} run(s) — {loc_note} Polling every 3s..."
    return jobs, False, html.Div(note, style={"color": "#666"})


@callback(
    Output("rp-jobs-store", "data", allow_duplicate=True),
    Output("rp-poll-interval", "disabled", allow_duplicate=True),
    Input("rp-poll-interval", "n_intervals"),
    State("rp-jobs-store", "data"),
    prevent_initial_call=True,
)
def _poll_jobs(_n_intervals, jobs):
    if not jobs:
        return jobs, True

    still_pending = False
    for entry in jobs.values():
        if entry["status"] != "running":
            continue
        status = jr.read_status(entry["job_dir"])
        if not status or status["state"] == "running":
            still_pending = True
            continue
        if status["state"] == "error":
            entry["status"] = "error"
            entry["result"] = {"success": False, "error": status.get("error"),
                               "traceback": status.get("traceback")}
        else:
            entry["status"] = "done"
            entry["result"] = status["result"]

    return jobs, not still_pending


@callback(Output("rp-results", "children"), Input("rp-jobs-store", "data"))
def _render_results(jobs):
    if not jobs:
        return ""

    entries = [jobs[k] for k in sorted(jobs.keys(), key=int)]
    n_running = sum(1 for e in entries if e["status"] == "running")

    rows = []
    for entry in entries:
        r = entry["row"]
        result = entry.get("result") or {}
        if entry["status"] == "running":
            status_text = "… running"
        elif result.get("success"):
            status_text = "✓ done"
            if result.get("job_id"):
                status_text += f" (SCITAS job {result['job_id']})"
        elif "returncode" in result:
            status_text = f"✗ failed (exit code {result['returncode']})"
        else:
            job_id = result.get("job_id")
            status_text = f"✗ failed (SCITAS job {job_id})" if job_id else "✗ failed"
            if result.get("error"):
                status_text += f": {result['error']}"
        rows.append({
            "subject_id": r["subject_id"], "roi_name": r["roi_name"], "goal": r["goal"],
            "filename": r["filename"], "status": status_text,
        })

    table = _styled_table("rp-jobs-table", [
        {"name": "Subject", "id": "subject_id"}, {"name": "ROI", "id": "roi_name"},
        {"name": "Goal", "id": "goal"}, {"name": "File", "id": "filename"},
        {"name": "Status", "id": "status"},
    ], data=rows)

    header = (html.P(f"{n_running} of {len(entries)} still running — polling every 3s...",
                     style={"color": "#666"}) if n_running
             else html.P(f"All {len(entries)} run(s) finished.", style={"color": "#060"}))

    logs = []
    for entry in entries:
        if entry["status"] == "running":
            continue
        result = entry.get("result") or {}
        r = entry["row"]
        label = f"sub-{r['subject_id']} {r['roi_name']} {r['goal']}"
        log_children = []
        if result.get("stdout"):
            log_children.append(html.Pre(result["stdout"], style={"fontSize": "12px", "whiteSpace": "pre-wrap"}))
        if result.get("stderr"):
            log_children.append(html.Pre(result["stderr"],
                                         style={"fontSize": "12px", "whiteSpace": "pre-wrap", "color": "#a00"}))
        if result.get("error"):
            log_children.append(html.Pre(result["error"],
                                         style={"fontSize": "12px", "whiteSpace": "pre-wrap", "color": "#a00"}))
        if result.get("traceback"):
            log_children.append(html.Pre(result["traceback"],
                                         style={"fontSize": "12px", "whiteSpace": "pre-wrap", "color": "#a00"}))
        if log_children:
            logs.append(html.Details([html.Summary(label)] + log_children))

    return html.Div([header, table] + logs)

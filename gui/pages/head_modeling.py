"""
pages/head_modeling.py — Phase 0: generate one or more subjects' head models
(m2m_{id}) via SimNIBS charm, the prerequisite for every later phase.

charm is a real external command-line tool (~30-90 min per subject), run
either locally (job_runner.py's background-job mechanism, same pattern as
Phase 3's leadfield generation / one-off FEM) or on SCITAS
(charm_discovery.run_charm_on_scitas/batch_submit_charm — submits
charm_scitas.sbatch over SSH, blocks on the SLURM queue, scp's m2m_{id}/
back). Both fit the exact same job_runner contract, so this page's polling
UI doesn't care which one ran.

Multiple subjects run concurrently, one independent job_runner background
job each — same jobs-store/poll pattern as pages/comparison.py and
pages/run_optimization.py. For SCITAS, batch_submit_charm() gets every
selected subject's job queued using as few ssh connections as possible for
the WHOLE batch (one shared pipeline-code-sync check, one shared sbatch
submission connection) before any per-subject polling starts — submitting
each subject's own full charm sequence one at a time, all at once for N
subjects, is what used to open a burst of connections large enough to trip
SCITAS's own new-connection rate limiting.
"""
import os

import dash
from dash import html, dcc, dash_table, callback, Input, Output, State

import charm_discovery as chd
import job_runner as jr

dash.register_page(__name__, path="/", name="Head Modeling", category="Preprocessing", order=1)


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
    html.H2("Head Modeling (charm)"),
    html.P("Generates the head model (m2m_{id}/) from T1w (required) and T2w (optional, improves "
           "segmentation quality) — the prerequisite for every later phase. Select one or more "
           "subjects — each becomes its own independent job.",
           style={"fontSize": "13px", "color": "#666"}),

    html.Div([
        html.Label("Subjects"),
        dcc.Dropdown(id="hm-subject-dropdown", multi=True, placeholder="Select subject(s)...",
                    style={"maxWidth": "600px"}),
    ], style={"marginBottom": "1rem"}),

    _styled_table("hm-status-table", [
        {"name": "Subject", "id": "subject"},
        {"name": "T1w", "id": "t1_display"},
        {"name": "T2w (optional)", "id": "t2_display"},
        {"name": "Status", "id": "status_display"},
    ], style_data_conditional=[
        {"if": {"filter_query": '{status_display} contains "partial"'}, "backgroundColor": "#fff4d6"},
    ]),

    html.Div([
        dcc.Checklist(id="hm-use-t2", options=[{"label": " Use T2w if available", "value": "use_t2"}],
                      value=["use_t2"], style={"marginBottom": "0.5rem"}),
        dcc.Checklist(id="hm-force", options=[{"label": " Force (redo even if already complete)",
                                               "value": "force"}], value=[]),
    ], style={"margin": "1rem 0"}),

    html.Div([
        html.Label("Run on"),
        dcc.RadioItems(
            id="hm-run-location",
            options=[
                {"label": " Local (this machine)", "value": "local"},
                {"label": " SCITAS (jed.hpc.epfl.ch)", "value": "scitas"},
            ],
            value="local",
        ),
    ], style={"marginBottom": "1rem"}),

    html.Button("Start Head Modeling", id="hm-start-button", n_clicks=0,
                style={"padding": "0.5rem 1.5rem"}),

    dcc.Store(id="hm-jobs-store"),
    dcc.Interval(id="hm-poll-interval", interval=3000, disabled=True),
    html.Div(id="hm-run-note", style={"fontSize": "13px", "margin": "1rem 0"}),
    dcc.Loading(html.Div(id="hm-results", style={"marginTop": "1rem"})),
])


@callback(Output("hm-subject-dropdown", "options"), Input("hm-subject-dropdown", "id"))
def _load_subjects(_):
    options = []
    for s in chd.discover_subjects():
        if not s["has_rawdata"]:
            continue
        status = chd.charm_status(s["subject_id"])
        tag = "m2m ✓" if status["m2m_complete"] else ("m2m partial" if status["m2m_partial"] else "needs charm")
        options.append({"label": f"{s['subject_id']}   [{tag}]", "value": s["subject_id"]})
    return options


@callback(Output("hm-status-table", "data"), Input("hm-subject-dropdown", "value"))
def _update_status_table(subject_ids):
    if not subject_ids:
        return []
    rows = []
    for sid in subject_ids:
        status = chd.charm_status(sid)
        if status["m2m_complete"]:
            status_display = "✓ m2m already complete — Start will skip unless Force is checked"
        elif status["m2m_partial"]:
            status_display = "⚠ partial m2m_ (previous failed run) — will be cleaned up before restarting"
        else:
            status_display = "needs charm"
        rows.append({
            "subject": sid,
            "t1_display": "✓" if status["t1_exists"] else "✗ missing",
            "t2_display": "✓" if status["t2_exists"] else "—",
            "status_display": status_display,
        })
    return rows


@callback(
    Output("hm-jobs-store", "data"),
    Output("hm-poll-interval", "disabled"),
    Output("hm-run-note", "children"),
    Input("hm-start-button", "n_clicks"),
    State("hm-subject-dropdown", "value"),
    State("hm-use-t2", "value"),
    State("hm-force", "value"),
    State("hm-run-location", "value"),
    prevent_initial_call=True,
)
def _on_start_click(_n_clicks, subject_ids, use_t2_value, force_value, run_location):
    if not subject_ids:
        return None, True, html.Div("Select at least one subject first.", style={"color": "#a00"})

    use_t2 = "use_t2" in (use_t2_value or [])
    force = "force" in (force_value or [])
    jobs = {}

    if run_location == "scitas":
        submissions = chd.batch_submit_charm(subject_ids, use_t2, force)
        n_cached = sum(1 for s in submissions.values() if s.get("cached"))
        n_queued = sum(1 for s in submissions.values() if s["success"] and not s.get("cached"))
        for i, sid in enumerate(subject_ids):
            sub = submissions[sid]
            if sub.get("cached"):
                jobs[str(i)] = {"job_dir": None, "status": "done",
                                "result": {"success": True, "cached": True, "m2m_path": sub["m2m_path"]},
                                "subject": sid}
            elif sub["success"]:
                base_dir = os.path.join(chd.PROJECT_DIR, "derivatives", "SimNIBS", f"sub-{sid}", "_charm_jobs")
                _job_id, job_dir = jr.new_job_dir(base_dir)
                jr.start_local_job(job_dir, chd.wait_for_submitted_charm_job, sub["job_id"], sid)
                jobs[str(i)] = {"job_dir": job_dir, "status": "running", "result": None, "subject": sid}
            else:
                jobs[str(i)] = {"job_dir": None, "status": "error",
                                "result": {"success": False, "error": sub["error"]}, "subject": sid}
        note = (f"SCITAS — {n_queued} job(s) submitted in one batch, {n_cached} already complete. "
               f"Polling every 3s...")
    else:
        for i, sid in enumerate(subject_ids):
            base_dir = os.path.join(chd.PROJECT_DIR, "derivatives", "SimNIBS", f"sub-{sid}", "_charm_jobs")
            _job_id, job_dir = jr.new_job_dir(base_dir)
            jr.start_local_job(job_dir, chd.run_charm, sid, use_t2, force)
            jobs[str(i)] = {"job_dir": job_dir, "status": "running", "result": None, "subject": sid}
        note = f"Started {len(jobs)} run(s) locally, each in its own background thread. Polling every 3s..."

    return jobs, False, html.Div(note, style={"color": "#666"})


@callback(
    Output("hm-jobs-store", "data", allow_duplicate=True),
    Output("hm-poll-interval", "disabled", allow_duplicate=True),
    Input("hm-poll-interval", "n_intervals"),
    State("hm-jobs-store", "data"),
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
            entry["result"] = {"success": False, "error": status.get("error")}
        else:
            entry["status"] = "done"
            entry["result"] = status["result"]

    return jobs, not still_pending


@callback(Output("hm-results", "children"), Input("hm-jobs-store", "data"))
def _render_results(jobs):
    if not jobs:
        return ""

    entries = [jobs[k] for k in sorted(jobs.keys(), key=int)]
    n_running = sum(1 for e in entries if e["status"] == "running")

    rows = []
    for entry in entries:
        result = entry.get("result") or {}
        if entry["status"] == "running":
            status_text = "… running"
        elif result.get("success"):
            status_text = "✓ already complete" if result.get("cached") else "✓ done"
        else:
            status_text = f"✗ {result.get('error')}"
        rows.append({"subject": entry["subject"], "status": status_text})

    table = _styled_table("hm-jobs-table", [
        {"name": "Subject", "id": "subject"},
        {"name": "Status", "id": "status"},
    ], data=rows, style_data_conditional=[
        {"if": {"filter_query": '{status} contains "✗"'}, "backgroundColor": "#ffe0e0"},
    ])

    header = (html.P(f"{n_running} of {len(entries)} still running — polling every 3s...",
                     style={"color": "#666"}) if n_running
             else html.P(f"All {len(entries)} run(s) finished. Proceed to Mask Generation for whichever "
                         f"succeeded.", style={"color": "#060"}))

    return html.Div([header, table])

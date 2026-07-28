"""
pages/data_directory.py — Data Directory settings.

Formalizes what common.py already calls PROJECT_DIR: the external location
(possibly a different drive, or a remote/network-mounted path) holding
rawdata/ and derivatives/ — kept separate from the code+environment location
(code/gui, code/pipeline — always resolved relative to this repo) and from
the SCITAS cluster (see pages/scitas_connection.py). Three distinct places,
three distinct settings.

Changing the path here takes effect on GUI restart (see common.py's
priority-order note) — this page does not attempt to hot-swap PROJECT_DIR
for the currently-running process.

The "sync from SCITAS" section pulls specific subjects/folders down on
demand (scp -r) — deliberately not an automatic/whole-tree sync, since
derivatives (leadfields especially) can be very large.
"""
import os
import time

import dash
from dash import html, dcc, dash_table, callback, Input, Output, State, ctx

import common
import job_runner as jr
import scitas_discovery as sd

dash.register_page(__name__, path="/data-directory", name="Data Directory", category="Settings", order=2)

SYNC_JOB_BASE_DIR = os.path.join(common.PROJECT_DIR, "_data_sync_jobs")


def _run_sync_job(job_dir: str, subject_ids: list[str], folder_tags: list[str], project_dir: str) -> dict:
    """Runs inside a job_runner background thread. Wraps
    sd.sync_subject_data with a progress_cb that live-updates the job's own
    status.json (job_runner.report_progress) as lines arrive, instead of
    only reporting a result at the very end — the actual point of
    backgrounding this at all, since a sync can be a multi-minute
    operation (large derivatives, especially TIoptimization results) that
    used to just block the page behind a bare spinner.

    Returns {"sid|tag": {"success","error"}, ...} — sync_subject_data's own
    (subject_id, folder_tag) tuple keys aren't valid JSON object keys, so
    they're flattened to a single "sid|tag" string here before job_runner
    writes this as the job's result."""
    log_lines = []
    last_write = [0.0]

    def cb(line):
        log_lines.append(line)
        now = time.time()
        # Throttled to ~4 writes/sec — sync_subject_data can call this once
        # PER EXTRACTED FILE (hundreds+ for a big subject), and rewriting
        # the whole status.json that often is wasted I/O for no
        # perceptible difference in a UI that polls every second anyway.
        if now - last_write[0] > 0.25:
            jr.report_progress(job_dir, log=log_lines[-300:])
            last_write[0] = now

    results = sd.sync_subject_data(subject_ids, folder_tags, project_dir, progress_cb=cb)
    jr.report_progress(job_dir, log=log_lines[-300:])  # final flush — catches anything since the last throttled write
    return {f"{sid}|{tag}": {"success": r["success"], "error": r["error"]} for (sid, tag), r in results.items()}


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
    html.H2("Data Directory"),
    html.P("Three separate places, three separate settings: where the code+environment live (fixed, "
           "this repo), the Local (analysis) data directory below, and the Server (SCITAS) data "
           "directory further down. rawdata/derivatives can live on any of the last two — useful for "
           "pointing large data at a different drive, a remote-mounted path, or a differently-laid-out "
           "SCITAS account.", style={"fontSize": "13px", "color": "#666"}),

    html.H3("Local (analysis) Data Directory"),
    html.P("Where rawdata/ and derivatives/ live on THIS machine (subject MRIs, meshes, leadfields, "
           "results) — what every GUI page actually reads/writes to.",
           style={"fontSize": "13px", "color": "#666"}),
    html.Div(id="dd-current-path", style={"marginBottom": "1rem", "fontFamily": "monospace"}),

    html.P("Takes effect after restarting the GUI — this only saves the setting.",
           style={"fontSize": "13px", "color": "#666"}),
    dcc.Input(id="dd-path-input", type="text",
              placeholder="e.g. D:/BIDS_TI_Data or //server/share/BIDS_TI_Data",
              style={"width": "100%", "maxWidth": "600px"}),
    html.Div([
        html.Button("Set Up Directory Structure", id="dd-setup-button", n_clicks=0,
                    style={"marginRight": "1rem"}),
        html.Button("Save as Data Directory", id="dd-save-button", n_clicks=0),
    ], style={"marginTop": "0.5rem"}),
    html.Div(id="dd-action-result", style={"marginTop": "0.5rem", "fontSize": "13px"}),

    html.Hr(style={"marginTop": "2rem"}),
    html.H3("Server (SCITAS) Data Directory"),
    html.P("The project root on the bare SCITAS filesystem — code/, rawdata/, derivatives/ all live "
           "under it there (mirroring the local layout), outside any Apptainer container. Defaults to "
           "/scratch/{your username}/BIDS_TI_Toolbox; only change this if your SCITAS project actually "
           "lives somewhere else. Takes effect immediately (no GUI restart needed) — every SCITAS "
           "action (Head Modeling, Run Pipeline, code sync, this page's own sync-from-SCITAS below) "
           "uses it.", style={"fontSize": "13px", "color": "#666"}),
    html.Div(id="dd-server-current-path", style={"marginBottom": "0.5rem", "fontFamily": "monospace"}),
    dcc.Input(id="dd-server-path-input", type="text",
              placeholder="/scratch/yourusername/BIDS_TI_Toolbox", style={"width": "100%", "maxWidth": "600px"}),
    html.Div([
        html.Button("Save Server Data Directory", id="dd-server-save-button", n_clicks=0),
    ], style={"marginTop": "0.5rem"}),
    html.Div(id="dd-server-save-result", style={"marginTop": "0.5rem", "fontSize": "13px"}),

    html.Hr(style={"marginTop": "2rem"}),
    html.H3("Sync Data from SCITAS"),
    html.P("Pulls selected subjects'/folders' data down from SCITAS scratch into the current data "
           "directory (scp -r). Pick exactly what you need — derivatives can be large.",
           style={"fontSize": "13px", "color": "#666"}),

    html.Button("Load Subjects from SCITAS", id="dd-load-remote-button", n_clicks=0),
    html.Div(id="dd-load-remote-note", style={"fontSize": "13px", "margin": "0.5rem 0"}),

    _styled_table(
        "dd-remote-table",
        [
            {"name": "Subject", "id": "subject"},
            {"name": "rawdata", "id": "rawdata_display"},
            {"name": "derivatives/SimNIBS", "id": "simnibs_display"},
            {"name": "leadfield_volume", "id": "leadfield_display"},
            {"name": "derivatives/freesurfer", "id": "freesurfer_display"},
        ],
        row_selectable="multi",
        selected_rows=[],
    ),

    html.Div([
        html.Label("Folders to sync for selected subjects"),
        dcc.Checklist(
            id="dd-folder-checklist",
            options=[{"label": f" {tag}", "value": tag} for tag in sd.DATA_FOLDERS],
            value=["rawdata", "derivatives/SimNIBS"],
            inline=True,
        ),
    ], style={"margin": "1rem 0"}),

    html.Button("Sync Selected", id="dd-sync-button", n_clicks=0, style={"padding": "0.5rem 1.5rem"}),
    dcc.Store(id="dd-sync-job-store"),
    dcc.Interval(id="dd-sync-poll-interval", interval=1000, disabled=True),
    html.Div(id="dd-sync-note", style={"fontSize": "13px", "margin": "0.5rem 0"}),
    html.Pre(id="dd-sync-log", style={
        "maxHeight": "300px", "overflowY": "auto", "backgroundColor": "#111", "color": "#0f0",
        "padding": "0.5rem", "fontSize": "12px", "whiteSpace": "pre-wrap", "marginTop": "0.5rem",
    }),
    html.Div(id="dd-sync-result", style={"marginTop": "1rem"}),
])


@callback(Output("dd-current-path", "children"), Input("dd-current-path", "id"))
def _show_current_path(_):
    exists = os.path.isdir(common.PROJECT_DIR)
    return html.Div([
        html.Span("Current (active) data directory: ", style={"fontWeight": "bold"}),
        html.Span(common.PROJECT_DIR),
        html.Span("  ✓ exists" if exists else "  ✗ does not exist yet",
                  style={"color": "#060" if exists else "#a00"}),
    ])


@callback(
    Output("dd-action-result", "children"),
    Input("dd-setup-button", "n_clicks"),
    Input("dd-save-button", "n_clicks"),
    State("dd-path-input", "value"),
    prevent_initial_call=True,
)
def _on_action_click(_setup_clicks, _save_clicks, path):
    triggered = ctx.triggered_id
    path = (path or "").strip()
    if not path:
        return html.Div("Enter a path first.", style={"color": "#a00"})

    if triggered == "dd-setup-button":
        created = common.setup_data_dir_skeleton(path)
        return html.Div([
            html.P("✓ Created/confirmed:", style={"color": "#060"}),
            html.Ul([html.Li(p) for p in created]),
        ])

    common.save_data_dir_setting(path)
    return html.Div(f"✓ Saved as the data directory: {path}. Restart the GUI for it to take effect "
                    f"everywhere (Head Modeling, Mask Generation, etc.).", style={"color": "#060"})


@callback(
    Output("dd-server-current-path", "children"),
    Output("dd-server-path-input", "value"),
    Input("dd-server-current-path", "id"),
)
def _show_server_path(_):
    s = sd.load_settings()
    current = s["server_data_dir"] or "/scratch/{username}/BIDS_TI_Toolbox  (default — not overridden)"
    return html.Div([
        html.Span("Current server data directory: ", style={"fontWeight": "bold"}),
        html.Span(current),
    ]), s["server_data_dir"]


@callback(
    Output("dd-server-save-result", "children"),
    Input("dd-server-save-button", "n_clicks"),
    State("dd-server-path-input", "value"),
    prevent_initial_call=True,
)
def _on_server_save_click(_n_clicks, path):
    # save_settings() doesn't merge with what's already saved — pass the
    # current host/username/identity_file through unchanged so this save
    # doesn't wipe out anything set on the SCITAS Connection page.
    current = sd.load_settings()
    sd.save_settings(host=current["host"], username=current["username"],
                     identity_file=current["identity_file"], server_data_dir=(path or "").strip() or None)
    return html.Div("✓ Saved — takes effect immediately for SCITAS actions.", style={"color": "#060"})


@callback(
    Output("dd-remote-table", "data"),
    Output("dd-load-remote-note", "children"),
    Input("dd-load-remote-button", "n_clicks"),
    prevent_initial_call=True,
)
def _on_load_remote_click(_n_clicks):
    subjects = sd.list_remote_subjects()
    if not subjects:
        return [], html.Span("No subjects found on SCITAS scratch (or connection failed — check "
                             "SCITAS Connection).", style={"color": "#a00"})
    status = sd.remote_data_status(subjects)
    rows = []
    for sid in subjects:
        s = status[sid]
        rows.append({
            "subject": sid,
            "rawdata_display": "✓" if s["rawdata"] else "—",
            "simnibs_display": "✓" if s["derivatives/SimNIBS"] else "—",
            "leadfield_display": "✓" if s["derivatives/SimNIBS/leadfield_volume"] else "—",
            "freesurfer_display": "✓" if s["derivatives/freesurfer"] else "—",
        })
    return rows, f"{len(subjects)} subject(s) found on SCITAS."


@callback(
    Output("dd-sync-job-store", "data"),
    Output("dd-sync-poll-interval", "disabled"),
    Output("dd-sync-note", "children"),
    Output("dd-sync-log", "children"),
    Output("dd-sync-result", "children"),
    Input("dd-sync-button", "n_clicks"),
    State("dd-remote-table", "data"),
    State("dd-remote-table", "selected_rows"),
    State("dd-folder-checklist", "value"),
    prevent_initial_call=True,
)
def _on_sync_click(_n_clicks, rows, selected_rows, folder_tags):
    rows = rows or []
    selected_rows = selected_rows or []
    subject_ids = [rows[i]["subject"] for i in selected_rows if i < len(rows)]

    if not subject_ids:
        return None, True, html.Div("Select at least one subject above.", style={"color": "#a00"}), "", ""
    if not folder_tags:
        return None, True, html.Div("Select at least one folder to sync.", style={"color": "#a00"}), "", ""

    os.makedirs(SYNC_JOB_BASE_DIR, exist_ok=True)
    _job_id, job_dir = jr.new_job_dir(SYNC_JOB_BASE_DIR)
    jr.start_local_job(job_dir, _run_sync_job, job_dir, subject_ids, folder_tags, common.PROJECT_DIR)
    note = html.Div(f"Syncing {len(subject_ids)} subject(s) x {len(folder_tags)} folder(s) — "
                    f"polling every 1s...", style={"color": "#666"})
    return job_dir, False, note, "", ""


@callback(
    Output("dd-sync-job-store", "data", allow_duplicate=True),
    Output("dd-sync-poll-interval", "disabled", allow_duplicate=True),
    Output("dd-sync-log", "children", allow_duplicate=True),
    Output("dd-sync-result", "children", allow_duplicate=True),
    Input("dd-sync-poll-interval", "n_intervals"),
    State("dd-sync-job-store", "data"),
    prevent_initial_call=True,
)
def _poll_sync(_n_intervals, job_dir):
    if not job_dir:
        return job_dir, True, "", ""
    status = jr.read_status(job_dir)
    if not status:
        return job_dir, False, "", ""

    log_text = "\n".join(status.get("log") or [])
    if status["state"] == "running":
        return job_dir, False, log_text, ""
    if status["state"] == "error":
        err = html.Pre(f"✗ {status.get('error')}\n\n{status.get('traceback', '')}",
                       style={"color": "#a00", "whiteSpace": "pre-wrap", "fontSize": "12px"})
        return job_dir, True, log_text, err

    result = status["result"] or {}
    rows_out = []
    for key, r in result.items():
        sid, tag = key.split("|", 1)
        rows_out.append({"subject": sid, "folder": tag, "status": "✓ ok" if r["success"] else f"✗ {r['error']}"})
    table = _styled_table("dd-sync-results-table", [
        {"name": "Subject", "id": "subject"},
        {"name": "Folder", "id": "folder"},
        {"name": "Status", "id": "status"},
    ], data=rows_out)
    return job_dir, True, log_text, table

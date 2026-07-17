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

import dash
from dash import html, dcc, dash_table, callback, Input, Output, State, ctx

import common
import scitas_discovery as sd

dash.register_page(__name__, path="/data-directory", name="Data Directory", category="Settings", order=2)


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
    dcc.Loading(html.Div(id="dd-sync-result", style={"marginTop": "1rem"})),
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
            "freesurfer_display": "✓" if s["derivatives/freesurfer"] else "—",
        })
    return rows, f"{len(subjects)} subject(s) found on SCITAS."


@callback(
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
        return html.Div("Select at least one subject above.", style={"color": "#a00"})
    if not folder_tags:
        return html.Div("Select at least one folder to sync.", style={"color": "#a00"})

    results = sd.sync_subject_data(subject_ids, folder_tags, common.PROJECT_DIR)
    rows_out = [
        {"subject": sid, "folder": tag, "status": "✓ ok" if r["success"] else f"✗ {r['error']}"}
        for (sid, tag), r in results.items()
    ]
    return _styled_table("dd-sync-results-table", [
        {"name": "Subject", "id": "subject"},
        {"name": "Folder", "id": "folder"},
        {"name": "Status", "id": "status"},
    ], data=rows_out)

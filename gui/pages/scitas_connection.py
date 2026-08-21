"""
pages/scitas_connection.py — SCITAS connection check + setup.

Tests whether this machine can reach jed.hpc.epfl.ch non-interactively
(BatchMode=yes, so a passphrase/password prompt fails fast instead of
hanging with no TTY to answer it — see scitas_discovery.py's ssh_run). If it
can't, this page lets the user override host/username/identity file, or
generate a new local SSH keypair.

Generating a key is local-only and always safe (never overwrites an
existing one). It does NOT by itself grant SCITAS access — the resulting
public key still has to be installed on the remote side (ssh-copy-id if
password auth still works, or EPFL's own GASPAR key-management process)
before it actually works; this page shows the key and says so, it can't do
that step for you.
"""
import dash
from dash import html, dcc, dash_table, callback, Input, Output, State

import scitas_discovery as sd


def _styled_table(id_, columns, data=None, **kwargs):
    return dash_table.DataTable(
        id=id_,
        columns=columns,
        data=data or [],
        style_cell={"textAlign": "left", "fontFamily": "monospace", "fontSize": "13px", "padding": "4px"},
        style_table={"overflowX": "auto"},
        **kwargs,
    )

dash.register_page(__name__, path="/scitas-connection", name="SCITAS Connection", category="Settings", order=1)

layout = html.Div([
    html.H2("SCITAS Connection"),
    html.P("Checks whether this machine can reach jed.hpc.epfl.ch non-interactively — required for "
           "Head Modeling / Run Pipeline's \"Run on SCITAS\" option.",
           style={"fontSize": "13px", "color": "#666"}),

    html.Button("Test Connection", id="sc-test-button", n_clicks=0, style={"padding": "0.5rem 1.5rem"}),
    dcc.Loading(html.Div(id="sc-test-result", style={"marginTop": "1rem"})),

    html.Div([
        html.H3("Connection Settings", style={"marginTop": "2rem"}),
        html.P("Leave these blank to use the default (jed.hpc.epfl.ch, resolved via ~/.ssh/config — "
               "this is what's configured and working today). Only fill these in to override it.",
               style={"fontSize": "13px", "color": "#666"}),
        html.Div([
            html.Div([
                html.Label("Host"),
                dcc.Input(id="sc-host-input", type="text", placeholder="jed.hpc.epfl.ch",
                          style={"width": "100%"}),
            ], style={"maxWidth": "300px", "marginRight": "1rem"}),
            html.Div([
                html.Label("Username"),
                dcc.Input(id="sc-username-input", type="text", placeholder="(from ~/.ssh/config)",
                          style={"width": "100%"}),
            ], style={"maxWidth": "220px", "marginRight": "1rem"}),
            html.Div([
                html.Label("Identity file"),
                dcc.Input(id="sc-identity-input", type="text", placeholder="(from ~/.ssh/config)",
                          style={"width": "100%"}),
            ], style={"maxWidth": "320px"}),
        ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "0.5rem"}),
        html.Button("Save Settings", id="sc-save-button", n_clicks=0),
        html.Div(id="sc-save-result", style={"marginTop": "0.5rem", "fontSize": "13px"}),
    ]),

    html.Div([
        html.H3("Generate a New SSH Key", style={"marginTop": "2rem"}),
        html.P("Only needed if you don't already have a working key for SCITAS. Creates a new local "
               "keypair — never overwrites an existing one, never touches the cluster by itself. You "
               "still need to add the public key shown below to your SCITAS account before it works.",
               style={"fontSize": "13px", "color": "#666"}),
        html.Button("Generate SSH Key", id="sc-genkey-button", n_clicks=0),
        html.Div(id="sc-genkey-result", style={"marginTop": "0.5rem"}),
    ]),

    html.Div([
        html.H3("Pipeline Code Sync", style={"marginTop": "2rem"}),
        html.P("Whether code/pipeline/*.py and *.sbatch on this machine match what's on SCITAS "
               "scratch. This is already checked and auto-fixed automatically before every job "
               "submission (Head Modeling, Run Pipeline) — you shouldn't need this in normal use. "
               "It's here as an extra, on-demand check/fix, for peace of mind or to troubleshoot "
               "a submission that failed with a remote import error.",
               style={"fontSize": "13px", "color": "#666"}),
        html.Button("Check Status", id="sc-code-status-button", n_clicks=0, style={"marginBottom": "0.5rem"}),
        _styled_table(
            "sc-code-status-table",
            [
                {"name": "File", "id": "filename"},
                {"name": "Status", "id": "status_display"},
            ],
            row_selectable="multi",
            selected_rows=[],
            style_data_conditional=[
                {"if": {"filter_query": '{status_display} contains "✗"'}, "backgroundColor": "#ffe0e0"},
            ],
        ),
        html.Div([
            html.Button("Select All Out-of-Sync", id="sc-code-select-stale-button", n_clicks=0,
                       style={"marginRight": "1rem"}),
            html.Button("Sync Selected", id="sc-code-sync-button", n_clicks=0),
        ], style={"marginTop": "0.5rem"}),
        dcc.Loading(html.Div(id="sc-code-sync-result", style={"marginTop": "1rem"})),
    ]),
])


@callback(
    Output("sc-test-result", "children"),
    Input("sc-test-button", "n_clicks"),
    prevent_initial_call=True,
)
def _on_test_click(_n_clicks):
    result = sd.test_connection()
    if result["success"]:
        return html.Div(f"✓ Connected to {result['host']} as {result['username']}.",
                        style={"color": "#060"})
    return html.Div([
        html.P(f"✗ Could not connect to {result['host']}: {result['error']}", style={"color": "#a00"}),
        html.P("If this is an authentication issue, either set connection details below (if you have "
               "a working key elsewhere on this machine) or generate a new key and add it to your "
               "SCITAS account.", style={"fontSize": "13px", "color": "#666"}),
    ])


@callback(
    Output("sc-host-input", "value"),
    Output("sc-username-input", "value"),
    Output("sc-identity-input", "value"),
    Input("sc-host-input", "id"),  # fires once, on page load
)
def _load_settings(_):
    s = sd.load_settings()
    host_value = None if s["host"] == sd.DEFAULT_HOST else s["host"]
    return host_value, s["username"], s["identity_file"]


@callback(
    Output("sc-save-result", "children"),
    Input("sc-save-button", "n_clicks"),
    State("sc-host-input", "value"),
    State("sc-username-input", "value"),
    State("sc-identity-input", "value"),
    prevent_initial_call=True,
)
def _on_save_click(_n_clicks, host, username, identity_file):
    # save_settings() doesn't merge with what's already saved — pass the
    # current server_data_dir through unchanged so this save doesn't wipe
    # out a value set on the Data Directory page.
    current = sd.load_settings()
    sd.save_settings(host=(host or "").strip() or None, username=(username or "").strip() or None,
                     identity_file=(identity_file or "").strip() or None,
                     server_data_dir=current["server_data_dir"])
    return html.Div("✓ Saved. Click \"Test Connection\" above to verify.", style={"color": "#060"})


@callback(
    Output("sc-genkey-result", "children"),
    Input("sc-genkey-button", "n_clicks"),
    prevent_initial_call=True,
)
def _on_genkey_click(_n_clicks):
    result = sd.generate_ssh_key()
    if not result["success"]:
        return html.Div(f"✗ Key generation failed: {result['error']}", style={"color": "#a00"})
    return html.Div([
        html.P(f"Key ready at {result['path']} (an existing key there is reused, never overwritten).",
               style={"color": "#060"}),
        html.P("To grant SCITAS access: add this public key to ~/.ssh/authorized_keys on the cluster "
               "(e.g. via ssh-copy-id if password auth still works, or through EPFL's own GASPAR "
               "key-management process), then set Identity file above to this key's path, save, and "
               "test.", style={"fontSize": "13px", "color": "#666"}),
        html.Pre(result["public_key"], style={"fontSize": "11px", "backgroundColor": "#f4f4f4",
                                              "padding": "0.5rem", "overflowX": "auto"}),
    ])


def _code_status_rows(status: dict) -> list[dict]:
    return [{"filename": f, "status_display": "✓ in sync" if s["in_sync"] else "✗ out of sync / missing"}
            for f, s in status.items()]


@callback(
    Output("sc-code-status-table", "data"),
    Output("sc-code-status-table", "selected_rows"),
    Output("sc-code-sync-result", "children"),
    Input("sc-code-status-button", "n_clicks"),
    prevent_initial_call=True,
)
def _on_code_status_click(_n_clicks):
    status = sd.code_sync_status()
    rows = _code_status_rows(status)
    stale_rows = [i for i, r in enumerate(rows) if r["status_display"].startswith("✗")]
    note = (html.P(f"{len(stale_rows)} file(s) out of sync — pre-selected below; click "
                   f"\"Sync Selected\" to fix (or adjust the selection first).",
                   style={"color": "#a60"}) if stale_rows
           else html.P("✓ Everything is in sync.", style={"color": "#060"}))
    return rows, stale_rows, note


@callback(
    Output("sc-code-status-table", "selected_rows", allow_duplicate=True),
    Input("sc-code-select-stale-button", "n_clicks"),
    State("sc-code-status-table", "data"),
    prevent_initial_call=True,
)
def _select_all_stale(_n_clicks, rows):
    rows = rows or []
    return [i for i, r in enumerate(rows) if r["status_display"].startswith("✗")]


@callback(
    Output("sc-code-status-table", "data", allow_duplicate=True),
    Output("sc-code-status-table", "selected_rows", allow_duplicate=True),
    Output("sc-code-sync-result", "children", allow_duplicate=True),
    Input("sc-code-sync-button", "n_clicks"),
    State("sc-code-status-table", "data"),
    State("sc-code-status-table", "selected_rows"),
    prevent_initial_call=True,
)
def _on_code_sync_click(_n_clicks, rows, selected_rows):
    rows = rows or []
    selected_rows = selected_rows or []
    filenames = [rows[i]["filename"] for i in selected_rows if i < len(rows)]
    if not filenames:
        return dash.no_update, dash.no_update, html.P("Select at least one file first.",
                                                       style={"color": "#a00"})

    result = sd.sync_pipeline_code(filenames=filenames)
    failed = {f: r["error"] for f, r in result.items() if not r["success"]}
    status = sd.code_sync_status()
    new_rows = _code_status_rows(status)
    if failed:
        note = html.P(f"✗ {len(failed)} file(s) failed to sync: {failed}", style={"color": "#a00"})
    else:
        note = html.P(f"✓ Synced {len(result)} file(s): {', '.join(filenames)}.", style={"color": "#060"})
    return new_rows, [], note

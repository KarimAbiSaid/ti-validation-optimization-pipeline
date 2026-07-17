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
from dash import html, dcc, callback, Input, Output, State

import scitas_discovery as sd

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

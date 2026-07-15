"""
pages/cap_registration.py — Phase 2: register an EEG cap layout (MNI space)
onto one or more subjects' scalp surfaces.

Cap can come from this project's own code/resources/caps/*.csv, SimNIBS's
built-in ElectrodeCaps_MNI/*.csv, or an arbitrary custom path. Registration status is
checked per subject (m2m + subject mesh present, and whether that cap's
registered CSV already exists in m2m_{id}/eeg_positions/) before batch
registering across a checkbox-selected subset, with the same overwrite
confirmation pattern as phase 1's mask Generate.

Single-cap-at-a-time by design (matches the current phase scope) — the data
layer (cap_discovery.py) already returns one subject/cap status per call, so
extending to a multi-cap comparison later is a page-level addition, not a
rewrite of the discovery logic.
"""
import os
from datetime import datetime

import plotly.graph_objects as go
import dash
from dash import html, dcc, dash_table, callback, Input, Output, State, ctx

import cap_discovery as cd

dash.register_page(__name__, path="/caps", name="Cap Registration", category="Preprocessing", order=3)


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
    html.H2("Cap Registration"),

    dcc.Store(id="cr-resolved-cap-store"),

    html.Div([
        html.Label("Subjects"),
        dcc.Dropdown(id="cr-subject-dropdown", multi=True, placeholder="Select subject(s)..."),
    ], style={"maxWidth": "600px", "marginBottom": "1.5rem"}),

    html.Div([
        html.Div([
            html.Label("Cap (project + SimNIBS built-in)"),
            dcc.Dropdown(id="cr-cap-dropdown", placeholder="Select a cap layout..."),
        ], style={"maxWidth": "420px", "marginRight": "1.5rem"}),
        html.Div([
            html.Label("...or a custom path (overrides the dropdown if set)"),
            dcc.Input(id="cr-cap-custom-path", type="text", style={"width": "100%"},
                      placeholder="e.g. D:/path/to/my_cap.csv"),
        ], style={"maxWidth": "480px", "flex": 1}),
    ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "0.5rem"}),
    html.Div(id="cr-cap-note", style={"fontSize": "13px", "marginBottom": "0.75rem"}),

    html.Div([
        html.Label("Custom path space (only matters when a custom path is set — "
                   "the dropdown caps above are always MNI space)"),
        dcc.RadioItems(
            id="cr-cap-space-mode",
            options=[
                {"label": " MNI space (needs warping — default)", "value": "mni"},
                {"label": " Already in subject space (e.g. digitized positions, or a "
                          "previously-registered file) — no warping, one target subject", "value": "subject"},
            ],
            value="mni",
        ),
    ], style={"marginBottom": "1.5rem"}),

    html.Div([
        html.H3("Registration Status"),
        _styled_table("cr-status-table", [
            {"name": "Subject", "id": "subject"},
            {"name": "Ready (m2m + mesh)", "id": "ready_display"},
            {"name": "Already registered", "id": "registered_display"},
            {"name": "Registered path", "id": "registered_path"},
        ], style_data_conditional=[
            {"if": {"filter_query": '{ready_display} contains "✗"'}, "backgroundColor": "#ffe0e0"},
            {"if": {"filter_query": '{registered_display} contains "✓"'}, "backgroundColor": "#e0ffe0"},
        ]),

        html.H3("Register", style={"marginTop": "1.5rem"}),
        html.Button("Select All Ready", id="cr-select-all-btn", n_clicks=0, style={"marginBottom": "0.5rem"}),
        _styled_table(
            "cr-register-table",
            [
                {"name": "Subject", "id": "subject"},
                {"name": "Output path", "id": "out_path"},
                {"name": "Status", "id": "overwrite_display"},
            ],
            row_selectable="multi",
            selected_rows=[],
            style_data_conditional=[
                {"if": {"filter_query": '{overwrite_display} contains "⚠"'}, "backgroundColor": "#ffe0e0"},
            ],
        ),
        html.Div(id="cr-overwrite-message", style={"margin": "0.5rem 0"}),
        dcc.Checklist(
            id="cr-overwrite-confirm",
            options=[{"label": " I confirm overwriting existing registered cap(s) above", "value": "confirm"}],
            value=[],
        ),
        html.Button("Register", id="cr-register-button", n_clicks=0, disabled=True,
                    style={"marginTop": "1rem", "padding": "0.5rem 1.5rem"}),
        dcc.Loading(html.Div(id="cr-register-results", style={"marginTop": "1rem"})),
    ], id="cr-mni-flow-container"),

    html.Div([
        html.H3("Adopt Subject-Space Cap"),
        html.P("No warping, no scalp projection — the file is assumed to already be positioned "
               "correctly for one specific subject. Just validated and copied into that subject's "
               "eeg_positions/ folder.", style={"fontSize": "13px", "color": "#666"}),
        html.Label("Target subject"),
        dcc.Dropdown(id="cr-adopt-target-subject", placeholder="Select subject...",
                     style={"maxWidth": "300px", "marginBottom": "0.5rem"}),
        html.Div(id="cr-adopt-status", style={"marginBottom": "0.5rem"}),
        html.Button("Adopt (no warping)", id="cr-adopt-button", n_clicks=0,
                    style={"padding": "0.5rem 1.5rem"}),
        dcc.Loading(html.Div(id="cr-adopt-result", style={"marginTop": "1rem"})),
    ], id="cr-adopt-flow-container", style={"display": "none"}),

    html.Div([
        html.H3("Cap Preview", style={"marginTop": "2rem"}),
        html.P("Decimated scalp surface + registered electrode positions for one subject — "
               "a quick sanity check, not a substitute for the .msh export.",
               style={"fontSize": "13px", "color": "#666"}),
        html.Div([
            html.Div([
                html.Label("Subject"),
                dcc.Dropdown(id="cv-preview-subject-dropdown", placeholder="Select subject..."),
            ], style={"maxWidth": "260px", "marginRight": "1rem"}),
            html.Div([
                html.Label("Registered cap"),
                dcc.Dropdown(id="cv-preview-cap-dropdown", placeholder="Select a registered cap..."),
            ], style={"maxWidth": "420px"}),
        ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "1rem"}),
        dcc.Loading(dcc.Graph(id="cv-preview-graph", style={"height": "600px"})),
    ]),
])


# ═════════════════════════════════════════════════════════════════════════════
# Subjects + cap resolution
# ═════════════════════════════════════════════════════════════════════════════

@callback(Output("cr-subject-dropdown", "options"), Input("cr-subject-dropdown", "id"))
def _load_subjects(_):
    options = []
    for s in cd.discover_subjects():
        status = "m2m ✓" if s["has_m2m"] else "m2m ✗ (needs charm)"
        raw = "raw ✓" if s["has_rawdata"] else "raw ✗"
        options.append({
            "label": f"{s['subject_id']}   [{status}, {raw}]",
            "value": s["subject_id"],
            "disabled": not s["has_m2m"],
        })
    return options


@callback(Output("cr-cap-dropdown", "options"), Input("cr-cap-dropdown", "id"))
def _load_caps(_):
    return [{"label": f"{c['name']}  ({c['source']})", "value": c["path"]} for c in cd.list_available_caps()]


@callback(
    Output("cr-mni-flow-container", "style"),
    Output("cr-adopt-flow-container", "style"),
    Input("cr-cap-space-mode", "value"),
)
def _toggle_mode_sections(mode):
    return (({} if mode == "mni" else {"display": "none"}),
            ({} if mode == "subject" else {"display": "none"}))


@callback(Output("cr-adopt-target-subject", "options"), Input("cr-adopt-target-subject", "id"))
def _load_adopt_target_subjects(_):
    options = []
    for s in cd.discover_subjects():
        status = "m2m ✓" if s["has_m2m"] else "m2m ✗ (needs charm)"
        options.append({"label": f"{s['subject_id']}   [{status}]", "value": s["subject_id"],
                         "disabled": not s["has_m2m"]})
    return options


@callback(
    Output("cr-resolved-cap-store", "data"),
    Output("cr-cap-note", "children"),
    Input("cr-cap-dropdown", "value"),
    Input("cr-cap-custom-path", "value"),
)
def _resolve_cap(dropdown_path, custom_path):
    cap_path = (custom_path or "").strip() or dropdown_path
    if not cap_path:
        return None, ""
    v = cd.validate_cap_path(cap_path)
    if not v["exists"]:
        return None, html.Span(f"✗ file not found: {cap_path}", style={"color": "#a00"})
    if not v["readable"]:
        return None, html.Span(f"✗ could not read cap: {v['error']}", style={"color": "#a00"})
    return cap_path, html.Span(f"✓ {v['n_electrodes']} electrodes — {cap_path}", style={"color": "#060"})


# ═════════════════════════════════════════════════════════════════════════════
# Registration status table
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("cr-status-table", "data"),
    Input("cr-subject-dropdown", "value"),
    Input("cr-resolved-cap-store", "data"),
)
def _update_status_table(subject_ids, cap_path):
    if not subject_ids or not cap_path:
        return []
    matrix = cd.cap_registration_matrix(subject_ids, cap_path)
    rows = []
    for sid in subject_ids:
        info = matrix[sid]
        rows.append({
            "subject": sid,
            "ready_display": "✓ ready" if info["ready"] else "✗ " + "; ".join(info["missing"]),
            "registered_display": "✓ yes" if info["registered"] else "— no",
            "registered_path": info["registered_path"],
        })
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Register table (checkbox-selectable subset of ready subjects)
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("cr-register-table", "data"),
    Output("cr-register-table", "selected_rows"),
    Input("cr-subject-dropdown", "value"),
    Input("cr-resolved-cap-store", "data"),
    State("cr-register-table", "selected_rows"),
)
def _update_register_table(subject_ids, cap_path, prev_selected):
    if not subject_ids or not cap_path:
        return [], []

    matrix = cd.cap_registration_matrix(subject_ids, cap_path)
    ready_subjects = [sid for sid in subject_ids if matrix[sid]["ready"]]

    rows = []
    for sid in ready_subjects:
        info = matrix[sid]
        rows.append({
            "subject": sid,
            "out_path": info["registered_path"],
            "overwrite_display": "⚠ will overwrite" if info["registered"] else "new file",
        })

    if ctx.triggered_id in ("cr-subject-dropdown", "cr-resolved-cap-store") or not prev_selected:
        selected_rows = list(range(len(rows)))
    else:
        selected_rows = [i for i in prev_selected if i < len(rows)]

    return rows, selected_rows


@callback(
    Output("cr-register-table", "selected_rows", allow_duplicate=True),
    Input("cr-select-all-btn", "n_clicks"),
    State("cr-register-table", "data"),
    prevent_initial_call=True,
)
def _select_all_ready(_n_clicks, data):
    return list(range(len(data or [])))


# ═════════════════════════════════════════════════════════════════════════════
# Register readiness (overwrite guard) + the action itself
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("cr-overwrite-message", "children"),
    Output("cr-register-button", "disabled"),
    Input("cr-register-table", "data"),
    Input("cr-register-table", "selected_rows"),
    Input("cr-overwrite-confirm", "value"),
)
def _update_register_readiness(rows, selected_rows, confirm_value):
    rows = rows or []
    selected_rows = selected_rows or []
    selected = [rows[i] for i in selected_rows if i < len(rows)]

    problems = []
    if not selected:
        problems.append("select at least one ready subject")

    overwrite_rows = [r for r in selected if r["overwrite_display"].startswith("⚠")]
    confirmed = "confirm" in (confirm_value or [])

    children = []
    if overwrite_rows:
        children.append(html.P(
            f"⚠ {len(overwrite_rows)} cap(s) already registered and will be overwritten: "
            + ", ".join(r["subject"] for r in overwrite_rows),
            style={"color": "#a00"}))
        if not confirmed:
            problems.append("confirm overwrite to proceed")

    if problems:
        children.append(html.P("Cannot register yet — " + "; ".join(problems),
                                style={"color": "#a00", "fontSize": "13px"}))

    return html.Div(children), bool(problems)


@callback(
    Output("cr-register-results", "children"),
    Input("cr-register-button", "n_clicks"),
    State("cr-register-table", "data"),
    State("cr-register-table", "selected_rows"),
    State("cr-resolved-cap-store", "data"),
    prevent_initial_call=True,
)
def _on_register_click(_n_clicks, rows, selected_rows, cap_path):
    rows = rows or []
    selected_rows = selected_rows or []
    subject_ids = [rows[i]["subject"] for i in selected_rows if i < len(rows)]

    if not subject_ids or not cap_path:
        return html.Div("Nothing to register — check subject/cap selection.", style={"color": "#a00"})

    results = cd.register_cap_for_subjects(subject_ids, cap_path)

    return _styled_table("cr-register-results-table", [
        {"name": "Subject", "id": "subject"},
        {"name": "Status", "id": "status"},
        {"name": "Electrodes", "id": "n_electrodes"},
        {"name": "Output path", "id": "out_path"},
    ], data=[
        {
            "subject": r["subject_id"],
            "status": "✓ ok" if r["success"] else f"✗ {r['error']}",
            "n_electrodes": r["n_electrodes"] if r["success"] else "",
            "out_path": r["out_path"] or "",
        }
        for r in results
    ])


# ═════════════════════════════════════════════════════════════════════════════
# Adopt subject-space cap (no warping, single target subject)
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("cr-adopt-status", "children"),
    Input("cr-adopt-target-subject", "value"),
    Input("cr-resolved-cap-store", "data"),
)
def _update_adopt_status(subject_id, cap_path):
    if not subject_id or not cap_path:
        return ""
    reg_path = cd.registered_cap_path(subject_id, cap_path)
    exists = os.path.isfile(reg_path)
    label = f"⚠ already registered — adopting will overwrite: {reg_path}" if exists \
        else f"— not yet registered: {reg_path}"
    return html.Div(label, style={"color": "#a00" if exists else "#666", "fontSize": "13px"})


@callback(
    Output("cr-adopt-result", "children"),
    Input("cr-adopt-button", "n_clicks"),
    State("cr-adopt-target-subject", "value"),
    State("cr-resolved-cap-store", "data"),
    prevent_initial_call=True,
)
def _on_adopt_click(_n_clicks, subject_id, cap_path):
    if not subject_id or not cap_path:
        return html.Div("Select a target subject and a valid cap path first.", style={"color": "#a00"})
    result = cd.adopt_subject_space_cap(subject_id, cap_path)
    if result["success"]:
        return html.Div(f"✓ {result['n_electrodes']} electrodes written → {result['out_path']}",
                        style={"color": "#060"})
    return html.Div(f"✗ {result['error']}", style={"color": "#a00"})


# ═════════════════════════════════════════════════════════════════════════════
# Cap preview: decimated scalp mesh + registered electrode positions
# ═════════════════════════════════════════════════════════════════════════════

@callback(Output("cv-preview-subject-dropdown", "options"), Input("cv-preview-subject-dropdown", "id"))
def _load_preview_subjects(_):
    options = []
    for s in cd.discover_subjects():
        if s["has_m2m"]:
            options.append({"label": s["subject_id"], "value": s["subject_id"]})
    return options


@callback(
    Output("cv-preview-cap-dropdown", "options"),
    Output("cv-preview-cap-dropdown", "value"),
    Input("cv-preview-subject-dropdown", "value"),
)
def _load_preview_caps(subject_id):
    if not subject_id:
        return [], None
    return [{"label": c["name"], "value": c["path"]} for c in cd.list_registered_caps(subject_id)], None


@callback(
    Output("cv-preview-graph", "figure"),
    Input("cv-preview-subject-dropdown", "value"),
    Input("cv-preview-cap-dropdown", "value"),
)
def _update_preview_graph(subject_id, cap_registered_path):
    if not subject_id:
        return go.Figure()

    mesh = cd.scalp_mesh_data(subject_id)
    fig = go.Figure(data=[go.Mesh3d(
        x=mesh["x"], y=mesh["y"], z=mesh["z"],
        i=mesh["i"], j=mesh["j"], k=mesh["k"],
        color="lightgray", opacity=0.5, hoverinfo="skip", name="scalp",
    )])

    if cap_registered_path:
        elec = cd.registered_electrode_positions(cap_registered_path)
        coords = elec["coords"]
        fig.add_trace(go.Scatter3d(
            x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
            mode="markers+text", text=elec["names"], textposition="top center",
            marker=dict(size=4, color="red"), name="electrodes",
        ))

    fig.update_layout(
        margin=dict(l=0, r=0, t=20, b=0),
        scene=dict(aspectmode="data",
                   xaxis=dict(visible=False), yaxis=dict(visible=False), zaxis=dict(visible=False)),
    )
    return fig

"""
pages/comparison.py — Phase 4: run multiple FEM/TI setups side by side.

Each row is an independent setup: subject, compute mode (leadfield/oneoff),
cap, channel electrodes + currents, label. Cap, ROI, and non-ROI can each
independently be "common" (one shared value, with a per-subject readiness
check) or per-row (a dropdown scoped to that row's own subject) — covers all
three comparison patterns (same subject/different montages, different
subjects/same montage, fully independent setups) with one table.

Electrode-name and mask dropdowns are Dash DataTable dropdown_conditional
entries, filtered by that row's subject (and cap, for electrodes) —
rebuilt whenever the table data or the common-mode toggles change.

Leadfield rows run synchronously (fast/algebraic). Oneoff rows are real FEM
solves, so each gets its own background job (job_runner.py) and the page
polls until all are done.
"""
import os

import dash
from dash import ALL, MATCH, html, dcc, dash_table, callback, Input, Output, State, ctx

import cap_discovery as cd
import comparison_discovery as cx
import discovery
import fem_discovery as fd
import job_runner as jr

dash.register_page(__name__, path="/comparison", name="Comparison", category="Simulation", order=2)

_EMPTY_ROW = {"subject": "", "mode": "leadfield", "cap": "", "roi": "", "nonroi": "", "elec": "",
              "ch1_plus": "", "ch1_minus": "", "ch1_current": 2.0,
              "ch2_plus": "", "ch2_minus": "", "ch2_current": 2.0, "label": ""}

_LEGACY_ELEC_VALUE = "legacy"  # sentinel: use "any variant for this cap" (fem_discovery's coarse behaviour)


def _encode_elec(shape: str, dimensions: list, gel_thickness: float) -> str:
    dims = dimensions or ["", ""]
    return f"{shape}|{dims[0]}|{dims[1] if len(dims) > 1 else dims[0]}|{gel_thickness}"


def _decode_elec(value: str):
    """Returns (shape, dimensions, gel_thickness) or (None, None, None) for
    the legacy sentinel / an unset value."""
    if not value or value == _LEGACY_ELEC_VALUE:
        return None, None, None
    shape, d1, d2, gel = value.split("|")
    return shape, [float(d1), float(d2)], float(gel)


def _styled_table(id_, columns, data=None, **kwargs):
    return dash_table.DataTable(
        id=id_,
        columns=columns,
        data=data or [],
        style_cell={"textAlign": "left", "fontFamily": "monospace", "fontSize": "13px", "padding": "4px"},
        style_table={"overflowX": "auto"},
        **kwargs,
    )


def _atlas_region_picker(prefix):
    """Atlas -> Region dropdowns (from the atlas's full region list — any
    potential region, not just ones with an existing mask) PLUS a separate
    "existing mask names" dropdown (scanned from real files across the
    table's current subjects) — either one prefills the label input below,
    which stays freely editable."""
    return [
        html.Div([
            html.Div([
                html.Label("Atlas"),
                dcc.Dropdown(id=f"cx-{prefix}-atlas-dropdown", placeholder="Select atlas...",
                            style={"maxWidth": "220px"}),
            ], style={"marginRight": "1rem"}),
            html.Div([
                html.Label("Region"),
                dcc.Dropdown(id=f"cx-{prefix}-region-dropdown", placeholder="Select region...",
                            style={"maxWidth": "320px"}),
            ]),
        ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "0.5rem"}),
        html.Div(id=f"cx-{prefix}-atlas-note", style={"fontSize": "12px", "color": "#666",
                                                       "marginBottom": "0.5rem"}),
        html.Div([
            html.Label("...or pick from mask names that already exist (checked across the "
                       "table's current subjects)"),
            dcc.Dropdown(id=f"cx-{prefix}-existing-dropdown", placeholder="Select existing mask name...",
                        style={"maxWidth": "420px"}),
        ], style={"marginBottom": "0.5rem"}),
    ]


def _base_columns(show_cap, show_roi, show_nonroi, show_elec=False):
    # Channels/ROI/non-ROI are plain text, not DataTable dropdown cells —
    # DataTable's built-in dropdown editor has known typing/backspace
    # issues. "Mode" stays a dropdown (two fixed choices, no typing
    # needed). Cap gets a dropdown too (filtered per-subject) plus
    # red/green conditional cell coloring for registration status, so
    # picking the wrong one is visually obvious despite the click-only
    # editing UX. "Elec" (electrode settings), when per-row, is a dropdown
    # filtered by that row's own (subject, cap) — one option per cached
    # leadfield variant for that pair (see fem_discovery.list_leadfields).
    cols = [
        {"name": "Subject", "id": "subject"},
        {"name": "Mode", "id": "mode", "presentation": "dropdown"},
    ]
    if show_cap:
        cols.append({"name": "Cap", "id": "cap", "presentation": "dropdown"})
    if show_elec:
        cols.append({"name": "Elec settings", "id": "elec", "presentation": "dropdown"})
    cols += [
        {"name": "Ch1+", "id": "ch1_plus"},
        {"name": "Ch1-", "id": "ch1_minus"},
        {"name": "Ch1 mA", "id": "ch1_current", "type": "numeric"},
        {"name": "Ch2+", "id": "ch2_plus"},
        {"name": "Ch2-", "id": "ch2_minus"},
        {"name": "Ch2 mA", "id": "ch2_current", "type": "numeric"},
    ]
    if show_roi:
        cols.append({"name": "ROI mask", "id": "roi"})
    if show_nonroi:
        cols.append({"name": "Non-ROI mask", "id": "nonroi"})
    cols.append({"name": "Label", "id": "label"})
    return cols


layout = html.Div([
    html.H2("Comparison"),
    html.P("Run multiple FEM/TI setups side by side. Rows can share a subject (different "
           "montages), share a montage (different subjects), or be fully independent.",
           style={"fontSize": "13px", "color": "#666"}),

    # ── Cap ──────────────────────────────────────────────────────────────────
    html.H3("Cap"),
    dcc.Checklist(id="cx-common-cap-toggle",
                  options=[{"label": " Use the same cap for all setups", "value": "common"}], value=[]),
    html.Div([
        html.Label("Common cap"),
        dcc.Dropdown(id="cx-common-cap-dropdown", placeholder="Select cap...", style={"maxWidth": "420px"}),
        html.Div(id="cx-cap-readiness-container", style={"marginTop": "0.5rem"}),
    ], id="cx-common-cap-panel", style={"display": "none", "marginBottom": "1rem"}),

    # ── Electrode settings ─────────────────────────────────────────────────────
    html.H3("Electrode Settings (leadfield rows only)", style={"marginTop": "1.5rem"}),
    html.P("Which cached leadfield variant to use for a given cap — a cap can have more than "
           "one, computed with different electrode dimensions/gel thickness.",
           style={"fontSize": "12px", "color": "#666"}),
    dcc.Checklist(id="cx-common-elec-toggle",
                  options=[{"label": " Use the same electrode settings for all setups", "value": "common"}],
                  value=["common"]),
    html.Div([
        html.Div([
            html.Label("Dimensions (mm)"),
            html.Div([
                dcc.Input(id="cx-common-elec-dim1", type="number", value=19.5, step=0.5,
                          style={"width": "90px", "marginRight": "0.5rem"}),
                dcc.Input(id="cx-common-elec-dim2", type="number", value=19.5, step=0.5,
                          style={"width": "90px"}),
            ], style={"display": "flex"}),
        ], style={"marginRight": "1.5rem"}),
        html.Div([
            html.Label("Gel thickness (mm)"),
            dcc.Input(id="cx-common-elec-gel", type="number", value=1.0, step=0.1, style={"width": "90px"}),
        ]),
    ], id="cx-common-elec-panel", style={"display": "flex", "flexWrap": "wrap", "marginBottom": "1rem"}),

    # ── ROI ──────────────────────────────────────────────────────────────────
    html.H3("ROI", style={"marginTop": "1.5rem"}),
    dcc.Checklist(id="cx-common-roi-toggle",
                  options=[{"label": " Use the same ROI for all setups", "value": "common"}], value=["common"]),
    html.Div([
        *_atlas_region_picker("roi"),
        html.Label("Resolved ROI label (auto-filled above — or type your own, e.g. a "
                   "multi-region union's custom name)"),
        dcc.Input(id="cx-roi-label", type="text", style={"width": "300px"}),
        html.Div(id="cx-roi-readiness-container", style={"marginTop": "0.5rem"}),
    ], id="cx-common-roi-panel", style={"marginBottom": "1rem"}),

    # ── Non-ROI ──────────────────────────────────────────────────────────────
    html.H3("Non-ROI (optional)", style={"marginTop": "1.5rem"}),
    dcc.Checklist(id="cx-common-nonroi-toggle",
                  options=[{"label": " Use the same non-ROI for all setups", "value": "common"}], value=["common"]),
    html.Div([
        *_atlas_region_picker("nonroi"),
        html.Label("Resolved non-ROI label (auto-filled above — or type your own)"),
        dcc.Input(id="cx-nonroi-label", type="text", style={"width": "300px"}),
        html.Div(id="cx-nonroi-readiness-container", style={"marginTop": "0.5rem"}),
    ], id="cx-common-nonroi-panel", style={"marginBottom": "1.5rem"}),

    # ── Setups ───────────────────────────────────────────────────────────────
    html.H3("Setups"),
    html.P("Mode: \"leadfield\" (fast — Ch1/Ch2 must be electrode names) or \"oneoff\" (real FEM "
           "solve — Ch1/Ch2 accept a name or raw \"x, y, z\"). Label names the output files and "
           "the comparison column below; auto-filled if left blank.",
           style={"fontSize": "12px", "color": "#666"}),

    html.Div([
        html.Label("Add rows for subject(s)"),
        dcc.Dropdown(id="cx-bulk-subject-dropdown", multi=True, placeholder="Select subject(s)...",
                    style={"maxWidth": "500px"}),
        html.Button("Add Rows for Selected Subjects", id="cx-add-subject-rows-btn", n_clicks=0,
                    style={"marginTop": "0.5rem"}),
    ], style={"marginBottom": "1rem"}),

    _styled_table("cx-setup-table", _base_columns(True, False, False), data=[dict(_EMPTY_ROW)],
                  editable=True, row_deletable=True),
    html.Button("Add Empty Row", id="cx-add-row-btn", n_clicks=0, style={"marginTop": "0.5rem"}),

    dcc.Store(id="cx-cap-registered-trigger", data=0),

    html.Div([
        html.Button("Run Comparison", id="cx-run-btn", n_clicks=0,
                    style={"padding": "0.5rem 1.5rem", "marginTop": "1.5rem"}),
    ]),

    dcc.Store(id="cx-jobs-store"),
    dcc.Interval(id="cx-poll-interval", interval=3000, disabled=True),
    html.Div(id="cx-run-note", style={"fontSize": "13px", "margin": "0.5rem 0"}),

    dcc.Loading(html.Div(id="cx-results", style={"marginTop": "1.5rem"})),

    html.Div([
        html.Label("Export filename"),
        dcc.Input(id="cx-export-filename", type="text", style={"width": "320px", "marginRight": "1rem"}),
        html.Button("Export Results (CSV)", id="cx-export-btn", n_clicks=0),
        html.Div(id="cx-export-note", style={"fontSize": "13px", "marginTop": "0.5rem", "color": "#a00"}),
    ], style={"marginTop": "1.5rem"}),
    dcc.Download(id="cx-download"),
])


# ═════════════════════════════════════════════════════════════════════════════
# Common-mode panel visibility + option loading
# ═════════════════════════════════════════════════════════════════════════════

@callback(Output("cx-common-cap-panel", "style"), Input("cx-common-cap-toggle", "value"))
def _toggle_cap_panel(value):
    style = {"marginBottom": "1rem"}
    if "common" not in (value or []):
        style["display"] = "none"
    return style


@callback(Output("cx-common-elec-panel", "style"), Input("cx-common-elec-toggle", "value"))
def _toggle_elec_panel(value):
    style = {"display": "flex", "flexWrap": "wrap", "marginBottom": "1rem"}
    if "common" not in (value or []):
        style["display"] = "none"
    return style


@callback(Output("cx-common-roi-panel", "style"), Input("cx-common-roi-toggle", "value"))
def _toggle_roi_panel(value):
    style = {"marginBottom": "1rem"}
    if "common" not in (value or []):
        style["display"] = "none"
    return style


@callback(Output("cx-common-nonroi-panel", "style"), Input("cx-common-nonroi-toggle", "value"))
def _toggle_nonroi_panel(value):
    style = {"marginBottom": "1.5rem"}
    if "common" not in (value or []):
        style["display"] = "none"
    return style


@callback(Output("cx-common-cap-dropdown", "options"), Input("cx-common-cap-dropdown", "id"))
def _load_common_cap_options(_):
    return [{"label": f"{c['name']}  ({c['source']})", "value": c["path"]} for c in cd.list_available_caps()]


# ═════════════════════════════════════════════════════════════════════════════
# ROI / non-ROI Atlas -> Region pickers (prefill the label input; the label
# stays freely editable for masks that don't follow the "{region}_{atlas}"
# convention, e.g. custom multi-region unions)
# ═════════════════════════════════════════════════════════════════════════════

def _register_atlas_region_callbacks(prefix):
    @callback(
        Output(f"cx-{prefix}-atlas-dropdown", "options"),
        Input(f"cx-{prefix}-atlas-dropdown", "id"),
    )
    def _load_atlas_options(_):
        return [
            {"label": name + ("" if meta["usable"] else "  (not yet usable)"),
             "value": name, "disabled": not meta["usable"]}
            for name, meta in discovery.ATLAS_REGISTRY.items()
        ]

    @callback(
        Output(f"cx-{prefix}-region-dropdown", "options"),
        Output(f"cx-{prefix}-atlas-note", "children"),
        Input(f"cx-{prefix}-atlas-dropdown", "value"),
        State("cx-setup-table", "data"),
    )
    def _load_region_options(atlas_name, rows):
        if not atlas_name:
            return [], ""
        subjects = _unique_subjects(rows)
        representative = next((s for s in subjects if os.path.isdir(discovery.get_m2m_path(s))), None)
        if not representative:
            return [], html.Span("Add a subject (with m2m) in the table below to load regions.",
                                 style={"color": "#a60"})
        options = cx.atlas_region_options(atlas_name, representative)
        if options is None:
            return [], html.Span("This atlas has no name lut here (e.g. BNA) — type the label directly below.",
                                 style={"color": "#a60"})
        return options, html.Span(f"{len(options)} regions (from sub-{representative})", style={"color": "#666"})

    @callback(
        Output(f"cx-{prefix}-label", "value", allow_duplicate=True),
        Input(f"cx-{prefix}-region-dropdown", "value"),
        State(f"cx-{prefix}-atlas-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _fill_label_from_region(region_name, atlas_name):
        if not region_name:
            return dash.no_update
        return f"{region_name}_{atlas_name}" if atlas_name else region_name

    @callback(
        Output(f"cx-{prefix}-existing-dropdown", "options"),
        Input("cx-setup-table", "data"),
    )
    def _load_existing_options(rows):
        subjects = _unique_subjects(rows)
        if not subjects:
            return []
        names = cx.list_existing_names(subjects)
        total = len(subjects)
        return [{"label": f"{name}  ({len(sids)}/{total} subjects)", "value": name}
                for name, sids in sorted(names.items())]

    @callback(
        Output(f"cx-{prefix}-label", "value", allow_duplicate=True),
        Input(f"cx-{prefix}-existing-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _fill_label_from_existing(name):
        if not name:
            return dash.no_update
        return name


_register_atlas_region_callbacks("roi")
_register_atlas_region_callbacks("nonroi")


# ═════════════════════════════════════════════════════════════════════════════
# Table row add + dynamic schema (columns + dropdowns), rebuilt whenever data
# or the common-mode toggles change
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("cx-setup-table", "data", allow_duplicate=True),
    Input("cx-add-row-btn", "n_clicks"),
    State("cx-setup-table", "data"),
    prevent_initial_call=True,
)
def _add_row(_n_clicks, rows):
    rows = rows or []
    rows.append(dict(_EMPTY_ROW))
    return rows


@callback(Output("cx-bulk-subject-dropdown", "options"), Input("cx-bulk-subject-dropdown", "id"))
def _load_bulk_subject_options(_):
    return [{"label": s["subject_id"], "value": s["subject_id"]} for s in cx.discover_subjects() if s["has_m2m"]]


@callback(
    Output("cx-setup-table", "data", allow_duplicate=True),
    Input("cx-add-subject-rows-btn", "n_clicks"),
    State("cx-bulk-subject-dropdown", "value"),
    State("cx-setup-table", "data"),
    prevent_initial_call=True,
)
def _add_rows_for_subjects(_n_clicks, subject_ids, rows):
    if not subject_ids:
        return dash.no_update
    rows = rows or []
    # If the table is still just the single pristine empty row, replace it
    # instead of leaving a stray blank row ahead of the new ones.
    if len(rows) == 1 and rows[0] == dict(_EMPTY_ROW):
        rows = []
    for sid in subject_ids:
        new_row = dict(_EMPTY_ROW)
        new_row["subject"] = sid
        rows.append(new_row)
    return rows


def _unique_subjects(rows):
    return sorted({(r.get("subject") or "").strip() for r in (rows or []) if (r.get("subject") or "").strip()})


@callback(
    Output("cx-setup-table", "columns"),
    Output("cx-setup-table", "dropdown"),
    Input("cx-common-cap-toggle", "value"),
    Input("cx-common-roi-toggle", "value"),
    Input("cx-common-nonroi-toggle", "value"),
    Input("cx-common-elec-toggle", "value"),
)
def _update_table_schema(cap_toggle, roi_toggle, nonroi_toggle, elec_toggle):
    show_cap = "common" not in (cap_toggle or [])
    show_roi = "common" not in (roi_toggle or [])
    show_nonroi = "common" not in (nonroi_toggle or [])
    show_elec = "common" not in (elec_toggle or [])
    columns = _base_columns(show_cap, show_roi, show_nonroi, show_elec)
    static_dropdown = {
        "mode": {"options": [{"label": "leadfield", "value": "leadfield"},
                             {"label": "oneoff", "value": "oneoff"}]},
    }
    return columns, static_dropdown


@callback(
    Output("cx-setup-table", "dropdown_conditional"),
    Output("cx-setup-table", "style_data_conditional"),
    Input("cx-setup-table", "data"),
    Input("cx-common-cap-toggle", "value"),
    Input("cx-common-cap-dropdown", "value"),
    Input("cx-cap-registered-trigger", "data"),
    Input("cx-common-elec-toggle", "value"),
)
def _update_cap_dropdown_and_colors(rows, cap_toggle, common_cap_path, _trigger, elec_toggle):
    show_cap = "common" not in (cap_toggle or [])
    common_cap_name = cd.cap_stem(common_cap_path) if (not show_cap and common_cap_path) else None
    show_elec = "common" not in (elec_toggle or [])

    rows = rows or []
    dropdown_conditional = []
    style_conditional = []
    seen_subjects = set()
    seen_cap_pairs = set()
    seen_leadfield_pairs = set()
    seen_elec_pairs = set()

    for row in rows:
        sid = (row.get("subject") or "").strip()
        if not sid:
            continue

        if show_cap:
            if sid not in seen_subjects:
                seen_subjects.add(sid)
                cap_opts = [{"label": c["name"], "value": c["name"]} for c in cd.list_registered_caps(sid)]
                dropdown_conditional.append({
                    "if": {"column_id": "cap", "filter_query": f'{{subject}} eq "{sid}"'},
                    "options": cap_opts,
                })
            cap_val = (row.get("cap") or "").strip()
        else:
            cap_val = common_cap_name or ""

        # Cap cell: red/green by registration status (only meaningful when
        # Cap is a per-row column — the common-cap panel has its own
        # readiness display for this).
        if show_cap and cap_val and (sid, cap_val) not in seen_cap_pairs:
            seen_cap_pairs.add((sid, cap_val))
            registered = os.path.isfile(cd.registered_cap_path(sid, cap_val))
            style_conditional.append({
                "if": {"column_id": "cap", "filter_query": f'{{subject}} eq "{sid}" && {{cap}} eq "{cap_val}"'},
                "backgroundColor": "#d4f7d4" if registered else "#f7d4d4",
            })

        # Mode cell: red/green by leadfield availability for this row's
        # specific (subject, cap) pair — only when Mode is set to
        # "leadfield" (oneoff doesn't need one, so it stays uncolored).
        if cap_val and (sid, cap_val) not in seen_leadfield_pairs:
            seen_leadfield_pairs.add((sid, cap_val))
            has_leadfield = fd.leadfield_status(sid, cap_val)["exists"]
            filter_q = (f'{{subject}} eq "{sid}" && {{cap}} eq "{cap_val}" && {{mode}} eq "leadfield"' if show_cap
                       else f'{{subject}} eq "{sid}" && {{mode}} eq "leadfield"')
            style_conditional.append({
                "if": {"column_id": "mode", "filter_query": filter_q},
                "backgroundColor": "#d4f7d4" if has_leadfield else "#f7d4d4",
            })

        # Elec cell: one dropdown option per cached leadfield variant for
        # this row's own (subject, cap) pair.
        if show_elec and cap_val and (sid, cap_val) not in seen_elec_pairs:
            seen_elec_pairs.add((sid, cap_val))
            variants = [lf for lf in fd.list_leadfields(sid) if lf["cap_name"] == cap_val]
            elec_opts = []
            for lf in variants:
                if lf["tag"] is None:
                    elec_opts.append({"label": lf["label"], "value": _LEGACY_ELEC_VALUE})
                else:
                    p = lf["params"]
                    dims = p.get("dimensions") or [None, None]
                    elec_opts.append({
                        "label": lf["label"],
                        "value": _encode_elec(p.get("shape", "ellipse"), dims, p.get("gel_thickness", 1.0)),
                    })
            filter_q = (f'{{subject}} eq "{sid}" && {{cap}} eq "{cap_val}"' if show_cap
                       else f'{{subject}} eq "{sid}"')
            dropdown_conditional.append({
                "if": {"column_id": "elec", "filter_query": filter_q},
                "options": elec_opts,
            })

    return dropdown_conditional, style_conditional


# ═════════════════════════════════════════════════════════════════════════════
# Cap readiness (per subject in the table): registered? leadfield? + inline
# Register button for unregistered subjects
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("cx-cap-readiness-container", "children"),
    Input("cx-setup-table", "data"),
    Input("cx-common-cap-dropdown", "value"),
    Input("cx-cap-registered-trigger", "data"),
)
def _update_cap_readiness(rows, cap_path, _trigger):
    if not cap_path:
        return ""
    subjects = _unique_subjects(rows)
    if not subjects:
        return html.Div("Add subjects in the table below to see readiness.",
                        style={"color": "#666", "fontSize": "13px"})

    cap_name = cd.cap_stem(cap_path)
    items = []
    for sid in subjects:
        reg_path = cd.registered_cap_path(sid, cap_path)
        registered = os.path.isfile(reg_path)
        lf_ready = fd.leadfield_status(sid, cap_name)["exists"] if registered else False

        row_children = [
            html.Span(f"{sid}: ", style={"fontWeight": "bold", "marginRight": "0.5rem"}),
        ]
        if registered:
            row_children.append(html.Span("✓ registered", style={"color": "#060", "marginRight": "1rem"}))
            row_children.append(html.Span(
                "leadfield ✓ (leadfield mode ready)" if lf_ready else "leadfield ✗ (will need oneoff mode)",
                style={"color": "#060" if lf_ready else "#a60"}))
        else:
            row_children.append(html.Span("✗ not registered", style={"color": "#a00", "marginRight": "0.75rem"}))
            row_children.append(html.Button("Register", id={"type": "cx-cap-register-btn", "subject": sid},
                                            n_clicks=0, style={"marginRight": "0.5rem"}))
            row_children.append(html.Span(id={"type": "cx-cap-register-status", "subject": sid},
                                          style={"fontSize": "12px", "color": "#666"}))
        items.append(html.Div(row_children, style={"marginBottom": "0.35rem"}))
    return html.Div(items)


@callback(
    Output({"type": "cx-cap-register-status", "subject": MATCH}, "children"),
    Input({"type": "cx-cap-register-btn", "subject": MATCH}, "n_clicks"),
    State({"type": "cx-cap-register-btn", "subject": MATCH}, "id"),
    State("cx-common-cap-dropdown", "value"),
    prevent_initial_call=True,
)
def _on_register_cap_click(_n_clicks, btn_id, cap_path):
    sid = btn_id["subject"]
    result = cd.register_cap(sid, cap_path)
    if result["success"]:
        return html.Span(f"✓ registered ({result['n_electrodes']} electrodes)", style={"color": "#060"})
    return html.Span(f"✗ {result['error']}", style={"color": "#a00"})


@callback(
    Output("cx-cap-registered-trigger", "data"),
    Input({"type": "cx-cap-register-status", "subject": ALL}, "children"),
    State("cx-cap-registered-trigger", "data"),
    prevent_initial_call=True,
)
def _bump_cap_registered_trigger(_all_status_children, trigger_count):
    # Dash can't mix a MATCH output with a plain-id output in one callback
    # (all Outputs must share the same MATCH keys) — so the register-click
    # callback above only updates its own status span, and this separate
    # callback watches ALL those spans to bump the shared refresh counter
    # that the readiness panel / table-schema callbacks listen to.
    return (trigger_count or 0) + 1


# ═════════════════════════════════════════════════════════════════════════════
# ROI / non-ROI readiness (per subject in the table) — no action needed,
# just flags what's missing and points at Mask Generation
# ═════════════════════════════════════════════════════════════════════════════

def _mask_readiness_children(rows, label):
    if not label or not label.strip():
        return ""
    subjects = _unique_subjects(rows)
    if not subjects:
        return ""
    items = []
    for sid in subjects:
        found = cx.resolve_mask(sid, label.strip()) is not None
        items.append(html.Div([
            html.Span(f"{sid}: ", style={"fontWeight": "bold"}),
            html.Span("✓ found" if found else "✗ not found — generate it on the Mask Generation page first",
                      style={"color": "#060" if found else "#a00"}),
        ], style={"fontSize": "13px"}))
    return html.Div(items)


@callback(
    Output("cx-roi-readiness-container", "children"),
    Input("cx-setup-table", "data"),
    Input("cx-roi-label", "value"),
)
def _update_roi_readiness(rows, roi_label):
    return _mask_readiness_children(rows, roi_label)


@callback(
    Output("cx-nonroi-readiness-container", "children"),
    Input("cx-setup-table", "data"),
    Input("cx-nonroi-label", "value"),
)
def _update_nonroi_readiness(rows, nonroi_label):
    return _mask_readiness_children(rows, nonroi_label)


# ═════════════════════════════════════════════════════════════════════════════
# Run comparison: leadfield rows synchronously, oneoff rows as background jobs
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("cx-jobs-store", "data"),
    Output("cx-poll-interval", "disabled"),
    Output("cx-run-note", "children"),
    Input("cx-run-btn", "n_clicks"),
    State("cx-setup-table", "data"),
    State("cx-common-cap-toggle", "value"),
    State("cx-common-cap-dropdown", "value"),
    State("cx-common-roi-toggle", "value"),
    State("cx-roi-label", "value"),
    State("cx-common-nonroi-toggle", "value"),
    State("cx-nonroi-label", "value"),
    State("cx-common-elec-toggle", "value"),
    State("cx-common-elec-dim1", "value"),
    State("cx-common-elec-dim2", "value"),
    State("cx-common-elec-gel", "value"),
    prevent_initial_call=True,
)
def _on_run_click(_n_clicks, rows, cap_toggle, common_cap_path, roi_toggle, roi_label,
                   nonroi_toggle, nonroi_label, elec_toggle, common_dim1, common_dim2, common_gel):
    rows = [r for r in (rows or []) if (r.get("subject") or "").strip()]
    if not rows:
        return None, True, html.Div("No setups — fill in at least one row.", style={"color": "#a00"})

    use_common_cap = "common" in (cap_toggle or [])
    common_cap_name = cd.cap_stem(common_cap_path) if (use_common_cap and common_cap_path) else None
    if use_common_cap and not common_cap_name:
        return None, True, html.Div("Select a common cap first.", style={"color": "#a00"})

    use_common_roi = "common" in (roi_toggle or [])
    if use_common_roi and not (roi_label or "").strip():
        return None, True, html.Div("ROI label is required.", style={"color": "#a00"})

    use_common_nonroi = "common" in (nonroi_toggle or [])
    use_common_elec = "common" in (elec_toggle or [])
    common_elec = ("ellipse", [common_dim1, common_dim2], common_gel) if use_common_elec else (None, None, None)

    jobs = {}
    any_pending = False
    for i, row in enumerate(rows):
        mode = (row.get("mode") or "leadfield").strip().lower()
        subject_id = (row.get("subject") or "").strip()
        cap_val = common_cap_name if use_common_cap else (row.get("cap") or "").strip()
        roi_val = (roi_label.strip() if use_common_roi else (row.get("roi") or "").strip())
        nonroi_val = ((nonroi_label or "").strip() if use_common_nonroi
                     else (row.get("nonroi") or "").strip()) or None
        row_label = (row.get("label") or "").strip() or f"{subject_id}_row{i}"

        kwargs = dict(
            subject_id=subject_id, cap_name=cap_val or None, roi_label=roi_val, non_roi_label=nonroi_val,
            ch1_plus=(row.get("ch1_plus") or "").strip(), ch1_minus=(row.get("ch1_minus") or "").strip(),
            ch1_current_mA=float(row.get("ch1_current") or 0),
            ch2_plus=(row.get("ch2_plus") or "").strip(), ch2_minus=(row.get("ch2_minus") or "").strip(),
            ch2_current_mA=float(row.get("ch2_current") or 0),
            label=row_label,
        )
        row_key = str(i)
        display_row = {"subject": subject_id, "label": row_label}

        if mode == "oneoff":
            base_dir = os.path.join(cx.PROJECT_DIR, "derivatives", "SimNIBS",
                                     f"sub-{subject_id or 'unknown'}", "comparison", "_jobs")
            try:
                _job_id, job_dir = jr.new_job_dir(base_dir)
                jr.start_local_job(job_dir, cx.run_oneoff_row, **kwargs)
                jobs[row_key] = {"mode": "oneoff", "row": display_row, "job_dir": job_dir,
                                 "status": "running", "result": None}
                any_pending = True
            except Exception as e:
                jobs[row_key] = {"mode": "oneoff", "row": display_row, "job_dir": None,
                                 "status": "error", "result": {"success": False, "error": str(e)}}
        else:
            elec_shape, elec_dims, elec_gel = (common_elec if use_common_elec
                                               else _decode_elec((row.get("elec") or "").strip()))
            result = cx.run_leadfield_row(
                **kwargs, electrode_shape=elec_shape, electrode_dimensions=elec_dims,
                electrode_gel_thickness=elec_gel,
            )
            jobs[row_key] = {"mode": "leadfield", "row": display_row, "job_dir": None,
                             "status": "done", "result": result}

    note = ("Running — polling every 3s for one-off FEM rows..." if any_pending
           else "All rows are leadfield (fast) — done.")
    return jobs, not any_pending, html.Div(note, style={"color": "#666"})


@callback(
    Output("cx-jobs-store", "data", allow_duplicate=True),
    Output("cx-poll-interval", "disabled", allow_duplicate=True),
    Input("cx-poll-interval", "n_intervals"),
    State("cx-jobs-store", "data"),
    prevent_initial_call=True,
)
def _poll_jobs(_n_intervals, jobs):
    if not jobs:
        return jobs, True

    still_pending = False
    for entry in jobs.values():
        if entry["status"] != "running" or not entry.get("job_dir"):
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


# ═════════════════════════════════════════════════════════════════════════════
# Results display
# ═════════════════════════════════════════════════════════════════════════════

def _build_status_and_comparison(jobs):
    """Shared by the on-screen tables and the CSV export — {"status_rows",
    "setup_cols", "all_stat_keys", "comparison_rows"} (comparison_rows use
    plain setup-label keys, not the "col_{i}" ids the DataTable needs — the
    UI renderer remaps those separately)."""
    entries = [jobs[k] for k in sorted(jobs.keys(), key=int)]

    status_rows = []
    all_stat_keys = []
    setup_cols = []
    for entry in entries:
        row = entry["row"]
        setup_label = row.get("label") or f"sub-{row.get('subject')}"
        setup_cols.append(setup_label)
        result = entry.get("result") or {}
        status_rows.append({
            "setup": setup_label, "subject": row.get("subject"), "mode": entry["mode"],
            "status": "✓ ok" if result.get("success") else f"✗ {entry['status']}"
                      + (f": {result.get('error')}" if result.get("error") else ""),
        })
        if result.get("success") and result.get("stats"):
            for k in result["stats"]:
                if k not in all_stat_keys:
                    all_stat_keys.append(k)

    comparison_rows = []
    for stat_key in all_stat_keys:
        row_data = {"metric": stat_key}
        for entry, setup_label in zip(entries, setup_cols):
            stats = (entry.get("result") or {}).get("stats") or {}
            val = stats.get(stat_key)
            row_data[setup_label] = f"{val:.4f}" if isinstance(val, float) else ("" if val is None else str(val))
        comparison_rows.append(row_data)

    return {"status_rows": status_rows, "setup_cols": setup_cols,
            "all_stat_keys": all_stat_keys, "comparison_rows": comparison_rows}


@callback(Output("cx-results", "children"), Input("cx-jobs-store", "data"))
def _render_results(jobs):
    if not jobs:
        return ""

    built = _build_status_and_comparison(jobs)
    status_table = _styled_table("cx-status-table", [
        {"name": "Setup", "id": "setup"}, {"name": "Subject", "id": "subject"},
        {"name": "Mode", "id": "mode"}, {"name": "Status", "id": "status"},
    ], data=built["status_rows"])

    if not built["all_stat_keys"]:
        return html.Div([html.H4("Status"), status_table])

    comparison_columns = [{"name": "Metric", "id": "metric"}] + [
        {"name": col, "id": col} for col in built["setup_cols"]
    ]
    comparison_table = _styled_table("cx-comparison-table", comparison_columns, data=built["comparison_rows"])

    return html.Div([
        html.H4("Status"), status_table,
        html.H4("Comparison", style={"marginTop": "1.5rem"}), comparison_table,
    ])


# ═════════════════════════════════════════════════════════════════════════════
# Export results to CSV
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("cx-export-filename", "value"),
    Input("cx-jobs-store", "data"),
)
def _fill_export_filename(jobs):
    if not jobs:
        return dash.no_update
    import datetime
    return f"comparison_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"


@callback(
    Output("cx-download", "data"),
    Output("cx-export-note", "children"),
    Input("cx-export-btn", "n_clicks"),
    State("cx-jobs-store", "data"),
    State("cx-export-filename", "value"),
    prevent_initial_call=True,
)
def _on_export_click(_n_clicks, jobs, filename):
    if not jobs:
        return dash.no_update, "No results to export yet — run a comparison first."

    import csv
    import io

    built = _build_status_and_comparison(jobs)
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["Setup", "Subject", "Mode", "Status"])
    for r in built["status_rows"]:
        writer.writerow([r["setup"], r["subject"], r["mode"], r["status"]])

    if built["all_stat_keys"]:
        writer.writerow([])
        writer.writerow(["Metric"] + built["setup_cols"])
        for r in built["comparison_rows"]:
            writer.writerow([r["metric"]] + [r.get(col, "") for col in built["setup_cols"]])

    name = (filename or "").strip() or "comparison.csv"
    if not name.lower().endswith(".csv"):
        name += ".csv"

    return dcc.send_string(buf.getvalue(), filename=name), ""

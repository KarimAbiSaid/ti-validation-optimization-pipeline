"""
pages/mask_generation.py — Phase 1: subject/atlas/region selection with
source-path validation, a browser for masks already on disk, and the actual
Generate action (batch, across a checkbox-selected subset of ready subjects).

Supports selecting multiple subjects at once: an "Atlas Availability" matrix
flags, per subject x atlas, whether everything needed is present (m2m,
subject labeling, MNI warp field, atlas file, roi list) or what's missing.

Mask names are always suffixed with the atlas name (e.g. "hippocampus_BNA"
vs "hippocampus_Allen") so the same region name from two different atlases
never collides on disk.
"""
import os
from datetime import datetime

import numpy as np
import plotly.graph_objects as go
import dash
from dash import html, dcc, dash_table, callback, Input, Output, State, ctx

import discovery

dash.register_page(__name__, path="/masks", name="Mask Generation", category="Preprocessing", order=2)

ATLAS_NAMES = list(discovery.ATLAS_REGISTRY.keys())
MASK_TYPES = ["ROI", "non-ROI", "general"]


def _slice_figure(t1_slice, mask_slice, title, h_mm=1.0, v_mm=1.0):
    v_dim, h_dim = t1_slice.shape
    x = np.arange(h_dim) * h_mm
    y = np.arange(v_dim) * v_mm

    fig = go.Figure()
    fig.add_trace(go.Heatmap(x=x, y=y, z=t1_slice, colorscale="gray", showscale=False, hoverinfo="skip"))
    overlay = np.where(mask_slice, 1.0, np.nan)
    fig.add_trace(go.Heatmap(x=x, y=y, z=overlay, colorscale=[[0, "red"], [1, "red"]], showscale=False,
                              opacity=0.45, hoverinfo="skip"))
    fig.update_layout(
        title=title, margin=dict(l=10, r=10, t=30, b=10), height=320,
        xaxis=dict(visible=False),
        yaxis=dict(visible=False, scaleanchor="x", autorange="reversed"),
    )
    return fig


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
    html.H2("Mask Generation"),

    dcc.Store(id="mg-region-lut-store"),
    dcc.Store(id="mg-selected-label-ids-store", data={}),

    html.Div([
        html.Label("Subjects"),
        dcc.Dropdown(id="mg-subject-dropdown", multi=True, placeholder="Select subject(s)..."),
    ], style={"maxWidth": "600px", "marginBottom": "1.5rem"}),

    html.Div([
        html.H3("Atlas Availability"),
        html.P("Per subject, per atlas: ready to use, or what's missing (MNI warp field, "
               "subject labeling, atlas file, roi list, ...).",
               style={"fontSize": "13px", "color": "#666"}),
        _styled_table(
            "mg-availability-table",
            [{"name": "Subject", "id": "subject"}] + [{"name": n, "id": n} for n in ATLAS_NAMES],
        ),
    ], style={"marginBottom": "2rem"}),

    html.Div([
        html.Div([
            html.H3("New Mask — Source Configuration"),
            html.Label("Atlas source"),
            dcc.Dropdown(
                id="mg-atlas-dropdown",
                options=[
                    {"label": name + ("" if meta["usable"] else "  (not yet usable)"),
                     "value": name, "disabled": not meta["usable"]}
                    for name, meta in discovery.ATLAS_REGISTRY.items()
                ],
                placeholder="Select atlas...",
            ),
            html.Div(id="mg-atlas-note", style={"fontStyle": "italic", "fontSize": "13px", "margin": "0.5rem 0"}),

            html.H4("Path check (per selected subject)"),
            _styled_table("mg-path-table", [
                {"name": "Subject", "id": "subject"},
                {"name": "Check", "id": "label"},
                {"name": "Path", "id": "path"},
                {"name": "Status", "id": "exists_display"},
            ]),

            html.H4("Region selection"),
            html.Div(id="mg-region-picker"),

            html.H4("Mask output settings"),
            html.Div([
                html.Label("Mask type"),
                dcc.RadioItems(
                    id="mg-mask-type",
                    options=[{"label": t, "value": t} for t in MASK_TYPES],
                    value="ROI", inline=True,
                ),
            ], style={"marginBottom": "0.75rem"}),
            html.Div([
                html.Label("Name  (atlas name is auto-appended, e.g. \"hippocampus\" → \"hippocampus_BNA\")"),
                dcc.Input(id="mg-mask-name", type="text", placeholder="e.g. hippocampus",
                          style={"width": "100%"}),
            ], style={"marginBottom": "0.75rem"}),
            html.Div([
                html.Label("Hemisphere"),
                dcc.RadioItems(
                    id="mg-hemisphere",
                    options=[
                        {"label": "Both / N/A", "value": "None"},
                        {"label": "Left", "value": "L"},
                        {"label": "Right", "value": "R"},
                    ],
                    value="None", inline=True,
                ),
            ], style={"marginBottom": "0.75rem"}),

            html.H4("Subjects to generate"),
            html.Button("Select All Ready", id="mg-select-all-btn", n_clicks=0,
                        style={"marginBottom": "0.5rem"}),
            _styled_table(
                "mg-generate-table",
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

            html.Div(id="mg-overwrite-message", style={"margin": "0.5rem 0"}),
            dcc.Checklist(
                id="mg-overwrite-confirm",
                options=[{"label": " I confirm overwriting existing file(s) above", "value": "confirm"}],
                value=[],
            ),

            html.Button("Generate", id="mg-generate-button", n_clicks=0, disabled=True,
                        style={"marginTop": "1rem", "padding": "0.5rem 1.5rem"}),

            dcc.Loading(html.Div(id="mg-generate-results", style={"marginTop": "1rem"})),
        ], style={"flex": "1 1 480px", "marginRight": "2rem"}),

        html.Div([
            html.H3("Existing Masks on Disk"),
            _styled_table("mg-existing-masks-table", [
                {"name": "Subject", "id": "subject"},
                {"name": "File", "id": "filename"},
                {"name": "Modified", "id": "mtime_display"},
                {"name": "Size", "id": "size_display"},
                {"name": "Description (sidecar)", "id": "description"},
            ]),
        ], style={"flex": "1 1 480px"}),
    ], style={"display": "flex", "flexWrap": "wrap"}),

    html.Div([
        html.H3("Mask Preview"),
        html.Div([
            html.Div([
                html.Label("Subject"),
                dcc.Dropdown(id="mg-preview-subject-dropdown", placeholder="Select subject..."),
            ], style={"maxWidth": "260px", "marginRight": "1rem"}),
            html.Div([
                html.Label("Mask (any file in that subject's roi/ folder)"),
                dcc.Dropdown(id="mg-preview-mask-dropdown", placeholder="Select mask..."),
            ], style={"maxWidth": "480px"}),
        ], style={"display": "flex", "marginBottom": "1rem"}),

        html.Div(id="mg-preview-resample-note", style={"fontSize": "13px", "color": "#a06a00",
                                                         "marginBottom": "0.5rem"}),

        html.Div([
            html.Div([
                html.Label("Sagittal slice (x)"),
                dcc.Input(id="mg-preview-sag-idx", type="number", style={"width": "100%"}),
                dcc.Graph(id="mg-preview-sag-graph"),
            ], style={"flex": 1, "minWidth": "260px"}),
            html.Div([
                html.Label("Coronal slice (y)"),
                dcc.Input(id="mg-preview-cor-idx", type="number", style={"width": "100%"}),
                dcc.Graph(id="mg-preview-cor-graph"),
            ], style={"flex": 1, "minWidth": "260px"}),
            html.Div([
                html.Label("Axial slice (z)"),
                dcc.Input(id="mg-preview-ax-idx", type="number", style={"width": "100%"}),
                dcc.Graph(id="mg-preview-ax-graph"),
            ], style={"flex": 1, "minWidth": "260px"}),
        ], style={"display": "flex", "gap": "1rem", "flexWrap": "wrap"}),

        html.Button("Export .msh for Gmsh", id="mg-export-msh-btn", n_clicks=0,
                    style={"marginTop": "1rem"}),
        html.Div(id="mg-export-msh-result", style={"marginTop": "0.5rem", "fontFamily": "monospace",
                                                     "fontSize": "13px"}),
    ], style={"marginTop": "2rem"}),
])


# ═════════════════════════════════════════════════════════════════════════════
# Subject / availability / existing-masks (unchanged from the validation-only screen)
# ═════════════════════════════════════════════════════════════════════════════

@callback(Output("mg-subject-dropdown", "options"), Input("mg-subject-dropdown", "id"))
def _load_subjects(_):
    options = []
    for s in discovery.discover_subjects():
        status = ("m2m ✓" if s["has_m2m"] else "m2m ✗ (needs charm)")
        raw = "raw ✓" if s["has_rawdata"] else "raw ✗"
        options.append({
            "label": f"{s['subject_id']}   [{status}, {raw}]",
            "value": s["subject_id"],
            "disabled": not s["has_m2m"],
        })
    return options


@callback(Output("mg-availability-table", "data"), Input("mg-subject-dropdown", "value"))
def _update_availability_matrix(subject_ids):
    if not subject_ids:
        return []
    matrix = discovery.atlas_availability_matrix(subject_ids)
    rows = []
    for sid in subject_ids:
        row = {"subject": sid}
        for atlas_name in ATLAS_NAMES:
            info = matrix[sid][atlas_name]
            row[atlas_name] = "✓ ready" if info["ready"] else "✗ " + "; ".join(info["missing"])
        rows.append(row)
    return rows


@callback(Output("mg-existing-masks-table", "data"), Input("mg-subject-dropdown", "value"))
def _update_existing_masks(subject_ids):
    if not subject_ids:
        return []
    rows = []
    for sid in subject_ids:
        for m in discovery.existing_masks(sid):
            rows.append({
                "subject": sid,
                "filename": m["filename"],
                "mtime_display": datetime.fromtimestamp(m["mtime"]).strftime("%Y-%m-%d %H:%M"),
                "size_display": f"{m['size_bytes'] / 1024:.0f} KB",
                "description": (m["sidecar"] or {}).get("Description", ""),
            })
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Atlas panel: path check + region picker (now also feeds the lut store)
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("mg-atlas-note", "children"),
    Output("mg-path-table", "data"),
    Output("mg-region-picker", "children"),
    Output("mg-region-lut-store", "data"),
    Input("mg-subject-dropdown", "value"),
    Input("mg-atlas-dropdown", "value"),
)
def _update_atlas_panel(subject_ids, atlas_name):
    if not subject_ids or not atlas_name:
        return "", [], html.Div("Select at least one subject and an atlas first."), None

    meta = discovery.ATLAS_REGISTRY[atlas_name]

    table_data = []
    not_ready = []
    for sid in subject_ids:
        result = discovery.full_atlas_check(atlas_name, sid)
        for c in result["checks"]:
            table_data.append({
                "subject": sid,
                "label": c["label"],
                "path": c["path"] or "(not set)",
                "exists_display": "✓ found" if c["exists"] else "✗ missing",
            })
        if not result["ready"]:
            not_ready.append(sid)

    lut_data = None
    if not meta["has_lut"]:
        region_ui = html.Div([
            html.P("No name lut for this atlas in the project — enter numeric label ids "
                   "directly, one 'Display Name: id' pair per line:"),
            dcc.Textarea(id="mg-bna-region-input", style={"width": "100%", "height": "100px"},
                         placeholder="PhG_R: 116\nrHippocampus_R: 216\ncHippocampus_R: 218"),
            html.Div(id="mg-bna-region-preview"),
        ])
    elif not_ready:
        region_ui = html.Div(
            "Region list not shown — these subjects are missing required files for "
            f"{atlas_name}: {', '.join(not_ready)}",
            style={"color": "#a00"})
    else:
        # lut() content (FreeSurferColorLUT names, Allen ROI acronyms) is not
        # subject-specific — any ready subject is a valid representative to load it from.
        try:
            lut = discovery.build_lut(atlas_name, subject_ids[0])
        except Exception as e:
            region_ui = html.Div(f"Failed to load region list: {e}", style={"color": "#a00"})
        else:
            options = [{"label": f"{name}  (id {rid})", "value": rid}
                       for rid, name in sorted(lut.items(), key=lambda kv: kv[1])]
            region_ui = dcc.Dropdown(id="mg-region-dropdown", options=options, multi=True,
                                      placeholder="Search regions...")
            lut_data = lut

    return meta["note"], table_data, region_ui, lut_data


@callback(
    Output("mg-bna-region-preview", "children"),
    Input("mg-bna-region-input", "value"),
    prevent_initial_call=True,
)
def _preview_bna_regions(text):
    if not text:
        return ""
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, _, rid = line.partition(":")
        name, rid = name.strip(), rid.strip()
        valid = rid.lstrip("-").isdigit()
        label = f"{name} → {rid}" + ("" if valid else "  (not a valid integer)")
        items.append(html.Li(label, style={"color": "#000" if valid else "#a00"}))
    return html.Ul(items)


# ═════════════════════════════════════════════════════════════════════════════
# Selected-regions store: mirrors whichever region-picker widget is active
# (lut multi-select or BNA textarea) into one {display_name: id} dict, so
# downstream callbacks don't need to know which widget produced it.
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("mg-selected-label-ids-store", "data", allow_duplicate=True),
    Input("mg-atlas-dropdown", "value"),
    prevent_initial_call=True,
)
def _reset_label_ids_on_atlas_change(_atlas_name):
    return {}


@callback(
    Output("mg-selected-label-ids-store", "data", allow_duplicate=True),
    Input("mg-region-dropdown", "value"),
    State("mg-region-lut-store", "data"),
    prevent_initial_call=True,
)
def _sync_lut_selection(selected_ids, lut):
    if not selected_ids or not lut:
        return {}
    lut = {int(k): v for k, v in lut.items()}  # dcc.Store round-trips dict keys as strings
    return {lut[rid]: rid for rid in selected_ids if rid in lut}


@callback(
    Output("mg-selected-label-ids-store", "data", allow_duplicate=True),
    Input("mg-bna-region-input", "value"),
    prevent_initial_call=True,
)
def _sync_bna_selection(text):
    out = {}
    for line in (text or "").splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        name, _, rid = line.partition(":")
        name, rid = name.strip(), rid.strip()
        if rid.lstrip("-").isdigit():
            out[name] = int(rid)
    return out


# ═════════════════════════════════════════════════════════════════════════════
# Generate table: which ready subjects to run, with per-subject overwrite flag
# ═════════════════════════════════════════════════════════════════════════════

def _effective_name(name, atlas_name):
    return f"{name}_{atlas_name}" if name and atlas_name else None


@callback(
    Output("mg-generate-table", "data"),
    Output("mg-generate-table", "selected_rows"),
    Input("mg-subject-dropdown", "value"),
    Input("mg-atlas-dropdown", "value"),
    Input("mg-mask-type", "value"),
    Input("mg-mask-name", "value"),
    State("mg-generate-table", "selected_rows"),
)
def _update_generate_table(subject_ids, atlas_name, mask_type, name, prev_selected):
    if not subject_ids or not atlas_name:
        return [], []

    matrix = discovery.atlas_availability_matrix(subject_ids)
    ready_subjects = [sid for sid in subject_ids if matrix[sid][atlas_name]["ready"]]

    eff_name = _effective_name(name, atlas_name)
    rows = []
    for sid in ready_subjects:
        if eff_name:
            out_path = discovery.expected_mask_path(sid, mask_type or "ROI", eff_name)
            overwrite_display = "⚠ will overwrite" if os.path.exists(out_path) else "new file"
        else:
            out_path = "(enter a name)"
            overwrite_display = "—"
        rows.append({"subject": sid, "out_path": out_path, "overwrite_display": overwrite_display})

    # Row set only changes identity when subjects/atlas change — otherwise keep
    # the user's manual checkbox selection instead of resetting it every keystroke.
    if ctx.triggered_id in ("mg-subject-dropdown", "mg-atlas-dropdown") or not prev_selected:
        selected_rows = list(range(len(rows)))
    else:
        selected_rows = [i for i in prev_selected if i < len(rows)]

    return rows, selected_rows


@callback(
    Output("mg-generate-table", "selected_rows", allow_duplicate=True),
    Input("mg-select-all-btn", "n_clicks"),
    State("mg-generate-table", "data"),
    prevent_initial_call=True,
)
def _select_all_ready(_n_clicks, data):
    return list(range(len(data or [])))


# ═════════════════════════════════════════════════════════════════════════════
# Generate readiness (overwrite confirmation gate) + the action itself
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("mg-overwrite-message", "children"),
    Output("mg-generate-button", "disabled"),
    Input("mg-generate-table", "data"),
    Input("mg-generate-table", "selected_rows"),
    Input("mg-mask-name", "value"),
    Input("mg-selected-label-ids-store", "data"),
    Input("mg-overwrite-confirm", "value"),
)
def _update_generate_readiness(rows, selected_rows, name, label_ids, confirm_value):
    rows = rows or []
    selected_rows = selected_rows or []
    selected = [rows[i] for i in selected_rows if i < len(rows)]

    problems = []
    if not selected:
        problems.append("select at least one ready subject")
    if not name:
        problems.append("enter a mask name")
    if not label_ids:
        problems.append("select at least one region")

    overwrite_rows = [r for r in selected if r["overwrite_display"].startswith("⚠")]
    confirmed = "confirm" in (confirm_value or [])

    children = []
    if overwrite_rows:
        children.append(html.P(
            f"⚠ {len(overwrite_rows)} file(s) already exist and will be overwritten: "
            + ", ".join(r["subject"] for r in overwrite_rows),
            style={"color": "#a00"}))
        if not confirmed:
            problems.append("confirm overwrite to proceed")

    if problems:
        children.append(html.P("Cannot generate yet — " + "; ".join(problems),
                                style={"color": "#a00", "fontSize": "13px"}))

    return html.Div(children), bool(problems)


@callback(
    Output("mg-generate-results", "children"),
    Input("mg-generate-button", "n_clicks"),
    State("mg-generate-table", "data"),
    State("mg-generate-table", "selected_rows"),
    State("mg-atlas-dropdown", "value"),
    State("mg-mask-type", "value"),
    State("mg-mask-name", "value"),
    State("mg-hemisphere", "value"),
    State("mg-selected-label-ids-store", "data"),
    prevent_initial_call=True,
)
def _on_generate_click(_n_clicks, rows, selected_rows, atlas_name, mask_type, name, hemisphere, label_ids):
    rows = rows or []
    selected_rows = selected_rows or []
    subject_ids = [rows[i]["subject"] for i in selected_rows if i < len(rows)]
    eff_name = _effective_name(name, atlas_name)

    if not subject_ids or not label_ids or not eff_name:
        return html.Div("Nothing to generate — check subject/region/name selection.",
                         style={"color": "#a00"})

    hemi = None if hemisphere in (None, "None") else hemisphere
    results = discovery.generate_masks(subject_ids, atlas_name, label_ids, mask_type, eff_name, hemisphere=hemi)

    return _styled_table("mg-generate-results-table", [
        {"name": "Subject", "id": "subject"},
        {"name": "Status", "id": "status"},
        {"name": "Voxel count", "id": "voxel_count"},
        {"name": "Output path", "id": "out_path"},
    ], data=[
        {
            "subject": r["subject_id"],
            "status": "✓ ok" if r["success"] else f"✗ error",
            "voxel_count": r["voxel_count"] if r["success"] else (r["error"] or ""),
            "out_path": r["out_path"] or "",
        }
        for r in results
    ])


# ═════════════════════════════════════════════════════════════════════════════
# Mask preview: subject + mask picker, three orthogonal slices, .msh export
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("mg-preview-subject-dropdown", "options"),
    Output("mg-preview-subject-dropdown", "value"),
    Input("mg-subject-dropdown", "value"),
)
def _update_preview_subject_options(subject_ids):
    subject_ids = subject_ids or []
    options = [{"label": sid, "value": sid} for sid in subject_ids]
    value = subject_ids[0] if subject_ids else None
    return options, value


@callback(
    Output("mg-preview-mask-dropdown", "options"),
    Output("mg-preview-mask-dropdown", "value"),
    Input("mg-preview-subject-dropdown", "value"),
)
def _update_preview_mask_options(subject_id):
    if not subject_id:
        return [], None
    options = [{"label": m["filename"], "value": m["path"]} for m in discovery.existing_masks(subject_id)]
    return options, None


@callback(
    Output("mg-preview-sag-idx", "value"),
    Output("mg-preview-sag-idx", "max"),
    Output("mg-preview-cor-idx", "value"),
    Output("mg-preview-cor-idx", "max"),
    Output("mg-preview-ax-idx", "value"),
    Output("mg-preview-ax-idx", "max"),
    Output("mg-preview-resample-note", "children"),
    Input("mg-preview-subject-dropdown", "value"),
    Input("mg-preview-mask-dropdown", "value"),
    prevent_initial_call=True,
)
def _update_preview_slice_defaults(subject_id, mask_path):
    if not subject_id or not mask_path:
        return None, None, None, None, None, None, ""
    defaults = discovery.default_view_indices(subject_id, mask_path)
    maxes = discovery.view_max_indices(subject_id)
    note = ""
    if discovery.mask_needs_resampling(subject_id, mask_path):
        note = ("⚠ This mask is on a different grid than T1.nii.gz — it's being "
                "nearest-neighbour resampled on the fly to overlay correctly.")
    return (defaults["sagittal"], maxes["sagittal"],
            defaults["coronal"], maxes["coronal"],
            defaults["axial"], maxes["axial"], note)


def _preview_graph(subject_id, mask_path, idx, view, label):
    if not subject_id or not mask_path or idx is None:
        return go.Figure()
    t1_slice, mask_slice = discovery.load_slice_overlay(subject_id, mask_path, view, int(idx))
    h_mm, v_mm = discovery.view_voxel_spacing(subject_id)[view]
    return _slice_figure(t1_slice, mask_slice, f"{label} @ {idx}", h_mm=h_mm, v_mm=v_mm)


@callback(
    Output("mg-preview-sag-graph", "figure"),
    Input("mg-preview-subject-dropdown", "value"),
    Input("mg-preview-mask-dropdown", "value"),
    Input("mg-preview-sag-idx", "value"),
)
def _update_sag_graph(subject_id, mask_path, idx):
    return _preview_graph(subject_id, mask_path, idx, "sagittal", "Sagittal")


@callback(
    Output("mg-preview-cor-graph", "figure"),
    Input("mg-preview-subject-dropdown", "value"),
    Input("mg-preview-mask-dropdown", "value"),
    Input("mg-preview-cor-idx", "value"),
)
def _update_cor_graph(subject_id, mask_path, idx):
    return _preview_graph(subject_id, mask_path, idx, "coronal", "Coronal")


@callback(
    Output("mg-preview-ax-graph", "figure"),
    Input("mg-preview-subject-dropdown", "value"),
    Input("mg-preview-mask-dropdown", "value"),
    Input("mg-preview-ax-idx", "value"),
)
def _update_ax_graph(subject_id, mask_path, idx):
    return _preview_graph(subject_id, mask_path, idx, "axial", "Axial")


@callback(
    Output("mg-export-msh-result", "children"),
    Input("mg-export-msh-btn", "n_clicks"),
    State("mg-preview-subject-dropdown", "value"),
    State("mg-preview-mask-dropdown", "value"),
    prevent_initial_call=True,
)
def _export_msh(_n_clicks, subject_id, mask_path):
    if not subject_id or not mask_path:
        return html.Div("Select a subject and mask first.", style={"color": "#a00"})
    try:
        out_path = discovery.export_mask_mesh(subject_id, mask_path)
    except Exception as e:
        return html.Div(f"Export failed: {e}", style={"color": "#a00"})
    return html.Div(f"Exported → {out_path}")

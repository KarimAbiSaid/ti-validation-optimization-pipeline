"""
pages/analysis.py — Analysis: distributional stats/plots over TI results
that have ALREADY been computed (Comparison page or Run Pipeline's
optimization+analysis step) — no FEM solve, no leadfield, ever run from
this page. Prefers existing outputs; a subject with nothing computed yet
gets a plain link to Comparison/FEM Validation instead of a blocking error.

Two cost tiers, both server-side but very different in cost:
  - "Quick stats" (comparison-run stats table + optimization diagnostics
    card): read straight from each run's own JSON sidecar
    (ti_stats_{label}.json / exhaustive_results.json) — no array loading
    at all, updates instantly as the run selection changes.
  - "Distribution" (histogram/KDE, percentiles, field-angle): reads the
    ACTUAL per-element field off each selected run's .msh (analysis_
    discovery.build_analysis()) — empirically ~4-6s per already-warm
    mesh load, but a FRESH GUI process's very first mesh load anywhere
    can take well over a minute (one-time SimNIBS/mesh_io warmup, not a
    per-file cost — verified empirically). Run as a background job
    (job_runner.py) + polling, same pattern as the Slice Viewer, so that
    worst case never looks like a hung page.

ROI selection here is NOT limited to whichever ROI a given run's own
config happened to use — see analysis_discovery.py's module docstring —
so multi-ROI selection (e.g. an ROI's separate _L/_R curated masks) is
the mechanism for laterality-style comparisons, not a special feature.
"""
import datetime
import os

import dash
from dash import html, dcc, dash_table, callback, Input, Output, State
import plotly.graph_objects as go

import analysis_discovery as ad
import cap_discovery as cd
import comparison_discovery as cx
import job_runner as jr
import slice_viewer_ui as sv
import viz_discovery as vz

dash.register_page(__name__, path="/analysis", name="Analysis", category="Analysis", order=1)


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
    html.H2("Analysis"),
    html.P("Stats and distribution plots over TI results that already exist (from Comparison or "
           "Run Pipeline). Nothing on this page runs a new FEM solve — pick a subject with no "
           "computed results yet and you'll get a link to go compute one instead.",
           style={"fontSize": "13px", "color": "#666"}),

    html.H3("Subjects"),
    dcc.Dropdown(id="an-subject-dropdown", multi=True, placeholder="Select subject(s)...",
                style={"maxWidth": "600px", "marginBottom": "0.75rem"}),
    html.Div(id="an-missing-subjects-note", style={"fontSize": "13px", "marginBottom": "0.75rem"}),

    html.H3("Available results"),
    html.P("Every already-computed result for the selected subject(s) — pick one or more to "
           "analyze (several at once overlays them).", style={"fontSize": "13px", "color": "#666"}),
    _styled_table(
        "an-runs-table",
        [
            {"name": "Subject", "id": "subject"},
            {"name": "Source", "id": "source"},
            {"name": "Run / label", "id": "label"},
            {"name": "E-field data", "id": "has_ef_display"},
        ],
        row_selectable="multi", selected_rows=[],
    ),
    html.Div([
        html.Button("Select all", id="an-runs-select-all-btn", n_clicks=0,
                    style={"marginRight": "0.5rem"}),
        html.Button("Deselect all", id="an-runs-deselect-all-btn", n_clicks=0),
    ], style={"marginTop": "0.5rem"}),

    html.H3("Regions", style={"marginTop": "1.5rem"}),
    html.P("Any curated mask that exists for the selected subject(s) — not limited to whichever "
           "ROI a given run's own config used.", style={"fontSize": "13px", "color": "#666"}),
    dcc.Dropdown(id="an-roi-dropdown", multi=True, placeholder="Select region(s)...",
                style={"maxWidth": "600px", "marginBottom": "1rem"}),

    html.H3("Quick stats", style={"marginTop": "1.5rem"}),
    html.P("Straight from each run's own saved summary — no computation, updates immediately.",
           style={"fontSize": "13px", "color": "#666"}),
    html.Div(id="an-comparison-stats-section"),
    html.Div(id="an-optimization-diagnostics-section", style={"marginTop": "0.75rem"}),

    html.H3("Distribution", style={"marginTop": "1.5rem"}),
    html.Div([
        html.Div([
            html.Label("Threshold for % of region above (V/m)"),
            dcc.Input(id="an-threshold-input", type="number", value=0.2, step=0.01,
                      style={"width": "100%"}),
        ], style={"maxWidth": "220px", "marginRight": "1.5rem"}),
        html.Div([
            html.Label("Plot type"),
            dcc.RadioItems(
                id="an-plot-type",
                options=[{"label": " Histogram", "value": "hist"}, {"label": " KDE", "value": "kde"}],
                value="hist", inline=True,
            ),
        ], style={"marginRight": "1.5rem"}),
        html.Div([
            html.Label("Histogram scale"),
            dcc.Checklist(
                id="an-normalize-hist",
                options=[{"label": " Normalize (divide by each group's own peak, so they all "
                          "line up on a 0-1 scale — same idea as the KDE, in histogram form)",
                          "value": "normalize"}],
                value=[],
            ),
        ]),
    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-end", "marginBottom": "0.5rem"}),
    dcc.Checklist(
        id="an-show-angle",
        options=[{"label": " Show E-field angle (comparison-run selections only — the angle "
                  "between each element's E_ch1/E_ch2 vectors)", "value": "show_angle"}],
        value=[], style={"marginBottom": "0.35rem"},
    ),
    dcc.Checklist(
        id="an-include-whole-brain",
        options=[{"label": " Include whole-brain background histogram", "value": "whole_brain"}],
        value=[], style={"marginBottom": "0.35rem"},
    ),
    dcc.Checklist(
        id="an-include-sum",
        options=[{"label": " Also show the sum across all selected runs, per region "
                  "(combines their matching elements into one aggregate distribution)",
                  "value": "include_sum"}],
        value=[], style={"marginBottom": "0.75rem"},
    ),
    html.Button("Build Distribution Analysis", id="an-build-button", n_clicks=0, disabled=True),
    dcc.Store(id="an-job-store"),
    dcc.Store(id="an-results-store"),
    dcc.Interval(id="an-interval", interval=2000, disabled=True),
    html.Div(id="an-build-status", style={"marginTop": "0.5rem", "fontSize": "13px"}),

    dcc.Loading(html.Div(id="an-distribution-results", style={"marginTop": "1rem"})),

    html.Div([
        dcc.Input(id="an-export-filename", type="text", placeholder="analysis.csv",
                  style={"marginRight": "0.5rem"}),
        html.Button("Export stats CSV", id="an-export-btn", n_clicks=0),
        dcc.Download(id="an-download"),
        html.Span(id="an-export-note", style={"marginLeft": "0.75rem", "fontSize": "13px", "color": "#a00"}),
    ], style={"marginTop": "1rem"}),

    html.H3("ROI Preview", style={"marginTop": "1.5rem"}),
    html.P("Highlight a selected region on this subject's own MRI, or a 3D render — reuses "
           "exactly the same rendering engine as FEM Validation's Slice Viewer / Render Figure "
           "(viz_discovery.py), just aimed at a run/region you've already picked above.",
           style={"fontSize": "13px", "color": "#666"}),
    html.Div([
        html.Div([
            html.Label("Run"),
            dcc.Dropdown(id="an-preview-run-dropdown", placeholder="Select a run..."),
        ], style={"minWidth": "320px", "marginRight": "1rem"}),
        html.Div([
            html.Label("Region"),
            dcc.Dropdown(id="an-preview-roi-dropdown", placeholder="Select a region..."),
        ], style={"minWidth": "220px", "marginRight": "1rem"}),
        html.Div([
            html.Label("View"),
            dcc.RadioItems(
                id="an-preview-mode",
                options=[{"label": " MRI slice", "value": "slice"}, {"label": " 3D render", "value": "3d"}],
                value="slice", inline=True,
            ),
        ]),
    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "flex-end", "marginBottom": "0.5rem"}),

    html.Div([
        html.Label("Cap (3D render needs this to place electrodes)"),
        dcc.Dropdown(id="an-preview-cap-dropdown", placeholder="—"),
        html.Div(id="an-preview-cap-note", style={"fontSize": "12px", "marginTop": "0.25rem"}),
    ], id="an-preview-cap-row", style={"display": "none", "marginBottom": "0.5rem", "maxWidth": "360px"}),

    html.Button("Preview", id="an-preview-button", n_clicks=0, disabled=True),
    dcc.Store(id="an-preview-job-store"),
    dcc.Store(id="an-preview-mode-store"),
    dcc.Interval(id="an-preview-interval", interval=2000, disabled=True),
    html.Div(id="an-preview-status", style={"marginTop": "0.5rem", "fontSize": "13px"}),

    html.Img(id="an-preview-colorbar-img", style={"marginTop": "0.5rem", "maxWidth": "320px"}),
    html.Div(id="an-preview-legend-row", style={"marginTop": "0.35rem", "fontSize": "13px"}),
    dcc.Loading(html.Div(id="an-preview-3d-results", style={"marginTop": "1rem"})),
    sv.viewer_container("an-preview-slice"),
    dcc.Download(id="an-preview-3d-download"),
])
sv.register_clientside_callbacks("an-preview-slice")


# ═════════════════════════════════════════════════════════════════════════════
# Subjects -> available results / missing-results note / ROI options
# ═════════════════════════════════════════════════════════════════════════════

@callback(Output("an-subject-dropdown", "options"), Input("an-subject-dropdown", "id"))
def _load_subjects(_):
    from common import discover_subjects
    return [{"label": s["subject_id"], "value": s["subject_id"]}
            for s in discover_subjects() if s["has_m2m"]]


@callback(
    Output("an-runs-table", "data"),
    Output("an-runs-table", "selected_rows"),
    Output("an-missing-subjects-note", "children"),
    Input("an-subject-dropdown", "value"),
)
def _load_runs(subject_ids):
    if not subject_ids:
        return [], [], ""
    rows = []
    missing = []
    for sid in subject_ids:
        meshes = ad.list_existing_meshes(sid)
        if not meshes:
            missing.append(sid)
            continue
        for m in meshes:
            rows.append({
                "subject": sid, "source": m["source"], "label": m["label"],
                "has_ef_display": "✓" if m["source"] == "comparison" else "—",
                # msh_path/ch1_*/ch2_* aren't in `columns` above so they never
                # render, but ROI Preview's 3D mode needs the electrode names
                # to resolve a matching registered cap (vz.find_matching_caps).
                "msh_path": m["msh_path"],
                "ch1_plus": m["ch1_plus"], "ch1_minus": m["ch1_minus"],
                "ch2_plus": m["ch2_plus"], "ch2_minus": m["ch2_minus"],
            })
    note = ""
    if missing:
        note = html.Div([
            f"No computed results yet for: {', '.join(missing)} — ",
            dcc.Link("Comparison", href="/comparison"), " · ",
            dcc.Link("FEM Validation", href="/fem"),
        ], style={"color": "#a00"})
    # Nothing pre-selected — with several subjects/runs this table can get
    # long, and auto-selecting everything makes the very first "Build" an
    # accidental overlay of things the user never chose to look at.
    return rows, [], note


@callback(
    Output("an-runs-table", "selected_rows", allow_duplicate=True),
    Input("an-runs-select-all-btn", "n_clicks"),
    State("an-runs-table", "data"),
    prevent_initial_call=True,
)
def _select_all_runs(_n_clicks, rows):
    return list(range(len(rows or [])))


@callback(
    Output("an-runs-table", "selected_rows", allow_duplicate=True),
    Input("an-runs-deselect-all-btn", "n_clicks"),
    prevent_initial_call=True,
)
def _deselect_all_runs(_n_clicks):
    return []


@callback(
    Output("an-roi-dropdown", "options"),
    Input("an-subject-dropdown", "value"),
)
def _load_roi_options(subject_ids):
    if not subject_ids:
        return []
    names = cx.list_existing_names(subject_ids)
    return [{"label": f"{name}  ({len(subs)}/{len(subject_ids)} subject(s))", "value": name}
            for name, subs in sorted(names.items())]


@callback(
    Output("an-build-button", "disabled"),
    Input("an-runs-table", "selected_rows"),
    Input("an-roi-dropdown", "value"),
)
def _toggle_build_button(selected_rows, roi_names):
    return not (selected_rows and roi_names)


def _selected_runs(rows, selected_rows):
    rows = rows or []
    return [rows[i] for i in (selected_rows or []) if i < len(rows)]


# ═════════════════════════════════════════════════════════════════════════════
# Quick stats (free tier — JSON only, no array loading)
# ═════════════════════════════════════════════════════════════════════════════

_COMPARISON_STAT_COLS = [
    ("TI_mean_whole_brain_V_m", "Whole-brain mean"), ("TI_max_whole_brain_V_m", "Whole-brain max"),
    ("TI_mean_ROI_V_m", "This run's ROI mean"), ("TI_max_ROI_V_m", "This run's ROI max"),
    ("TI_mean_non_ROI_V_m", "This run's non-ROI mean"), ("focality_ratio_brain", "Focality ratio"),
]


@callback(
    Output("an-comparison-stats-section", "children"),
    Input("an-runs-table", "data"),
    Input("an-runs-table", "selected_rows"),
)
def _update_comparison_stats(rows, selected_rows):
    selected = [r for r in _selected_runs(rows, selected_rows) if r["source"] == "comparison"]
    if not selected:
        return ""
    table_rows = []
    for r in selected:
        stats = ad.load_json_stats(
            ad.existing_stats_path(r["subject"], {"source": r["source"], "label": r["label"]}))
        stats = (stats or {}).get("stats", {})
        table_rows.append({
            "subject": r["subject"], "label": r["label"],
            **{disp: (f"{stats[key]:.4f}" if key in stats else "—") for key, disp in _COMPARISON_STAT_COLS},
        })
    return html.Div([
        html.Label("Comparison-run stats (that run's own configured ROI, not the picker above)",
                   style={"fontWeight": "bold", "fontSize": "13px"}),
        _styled_table("an-comparison-stats-table", [
            {"name": "Subject", "id": "subject"}, {"name": "Run", "id": "label"},
        ] + [{"name": disp, "id": disp} for _, disp in _COMPARISON_STAT_COLS], data=table_rows),
    ])


@callback(
    Output("an-optimization-diagnostics-section", "children"),
    Input("an-runs-table", "data"),
    Input("an-runs-table", "selected_rows"),
)
def _update_optimization_diagnostics(rows, selected_rows):
    selected = [r for r in _selected_runs(rows, selected_rows) if r["source"] == "TIoptimization"]
    if not selected:
        return ""
    cards = []
    for r in selected:
        result = ad.load_json_stats(
            ad.existing_stats_path(r["subject"], {"source": r["source"], "label": r["label"]}))
        if not result:
            continue
        best = result.get("best_montage", {}) or {}
        bits = [
            f"ROI ({result.get('roi', '?')}) mean: {result.get('roi_TI_mean_V_m', 0):.4f} V/m",
            f"Montage: {best.get('ch1_plus')}/{best.get('ch1_minus')} + "
            f"{best.get('ch2_plus')}/{best.get('ch2_minus')} "
            f"({best.get('ch1_current_mA')}/{best.get('ch2_current_mA')} mA)",
            f"Searched {result.get('n_montages_searched', '?')} montage(s) in "
            f"{result.get('elapsed_s', 0):.0f}s",
        ]
        if "focality_roc_score" in result:
            bits.append(f"ROC score: {result['focality_roc_score']:.4f}")
        cls = result.get("composite_location_score")
        if isinstance(cls, dict):
            extras = []
            if cls.get("background_minimized"):
                extras.append("background minimized")
            if cls.get("tiers_applied"):
                extras.append("electrode tiers applied")
            suffix = f" ({', '.join(extras)})" if extras else ""
            val = cls.get("value")
            val_str = f"{val:.4f}" if isinstance(val, (int, float)) else str(val)
            bits.append(f"Composite location score: {val_str}{suffix}")
        elif isinstance(cls, (int, float)):
            bits.append(f"Composite location score: {cls:.4f}")
        et = result.get("electrode_tiers")
        if isinstance(et, dict):
            bits.append(f"Electrode tiers: cascade level {et.get('cascade_level_used', '?')}")
        cards.append(html.Div([
            html.Div(f"{r['subject']} — {r['label']}", style={"fontWeight": "bold", "marginBottom": "0.25rem"}),
            html.Ul([html.Li(b) for b in bits], style={"margin": 0, "paddingLeft": "1.25rem"}),
        ], style={"padding": "0.5rem", "border": "1px solid #ddd", "borderRadius": "4px",
                  "marginBottom": "0.5rem", "fontSize": "13px"}))
    if not cards:
        return ""
    return html.Div([
        html.Label("Optimization run diagnostics", style={"fontWeight": "bold", "fontSize": "13px"}),
        html.Div(cards, style={"marginTop": "0.25rem"}),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# Distribution (background job — build_analysis() loads real .msh field
# data, so it runs off the main thread with job_runner + polling, same
# pattern as the Slice Viewer, rather than a plain synchronous callback —
# see module docstring for the empirical timing that justifies this).
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("an-job-store", "data"),
    Output("an-interval", "disabled"),
    Output("an-build-status", "children"),
    Input("an-build-button", "n_clicks"),
    State("an-runs-table", "data"),
    State("an-runs-table", "selected_rows"),
    State("an-roi-dropdown", "value"),
    State("an-threshold-input", "value"),
    State("an-show-angle", "value"),
    State("an-include-whole-brain", "value"),
    State("an-include-sum", "value"),
    prevent_initial_call=True,
)
def _on_build_click(_n_clicks, rows, selected_rows, roi_names, threshold, show_angle_value,
                    whole_brain_value, include_sum_value):
    selected = _selected_runs(rows, selected_rows)
    if not selected or not roi_names:
        return None, True, html.Div("Select at least one run and one region first.",
                                     style={"color": "#a00"})
    selections = [{"subject_id": r["subject"], "msh_path": r["msh_path"], "run_label": r["label"],
                  "roi_names": roi_names} for r in selected]
    compute_angle = "show_angle" in (show_angle_value or [])
    include_whole_brain = "whole_brain" in (whole_brain_value or [])
    include_sum = "include_sum" in (include_sum_value or [])

    from common import PROJECT_DIR
    base_dir = os.path.join(PROJECT_DIR, "derivatives", "SimNIBS", "_analysis_jobs")
    _job_id, job_dir = jr.new_job_dir(base_dir)
    jr.start_local_job(job_dir, ad.build_analysis, selections, threshold, compute_angle,
                       include_whole_brain, include_sum)
    return job_dir, False, html.Div(
        "Building — polling every 2s (a few seconds once SimNIBS has already loaded a mesh "
        "this session; the very first mesh load in a freshly-started GUI can take a minute or "
        "more, that's a one-time warmup, not a hang)...", style={"color": "#666"})


def _run_display_name(run_key: str) -> str:
    if run_key == ad.SUM_RUN_KEY:
        return "Sum across selected runs"
    subject_id, _, label = run_key.partition("::")
    return f"{subject_id}: {label}"


def _trace_label(run_key: str, roi_name: str) -> str:
    """Region name first, then subject/run — so overlaid legend entries and
    the stats table both group visually by region, with the subject as the
    secondary distinguisher (per explicit user feedback)."""
    return f"{roi_name} — {_run_display_name(run_key)}"


def _build_ti_figure(result, plot_type, normalize_hist=False):
    fig = go.Figure()
    for run_key, run in result["runs"].items():
        if "error" in run:
            continue
        for roi_name, entry in run["roi"].items():
            if entry.get("stats") is None:
                continue
            name = _trace_label(run_key, roi_name)
            if plot_type == "kde" and entry.get("kde"):
                fig.add_trace(go.Scatter(x=entry["kde"]["x"], y=entry["kde"]["y"],
                                         mode="lines", name=name, fill="tozeroy", opacity=0.5))
            else:
                h = entry["hist"]
                y = h.get("volume_mm3", h.get("counts"))
                if normalize_hist and y:
                    peak = max(y) or 1.0
                    y = [v / peak for v in y]
                fig.add_trace(go.Bar(x=h["edges"][:-1], y=y, name=name, opacity=0.6))
    normalized = plot_type == "hist" and normalize_hist
    fig.update_layout(
        barmode="overlay" if plot_type == "hist" else None,
        xaxis_title="TI amplitude (V/m)",
        yaxis_title=("Normalized (peak = 1)" if normalized
                    else "Tissue volume (mm³)" if plot_type == "hist" else "Density (volume-weighted)"),
        margin=dict(l=10, r=10, t=30, b=10), height=420, legend=dict(font=dict(size=11)),
    )
    return fig


def _build_angle_figure(result):
    fig = go.Figure()
    any_trace = False
    for run_key, run in result["runs"].items():
        if "error" in run:
            continue
        for roi_name, entry in run["roi"].items():
            if "angle_hist" not in entry:
                continue
            any_trace = True
            h = entry["angle_hist"]
            fig.add_trace(go.Bar(x=h["edges"][:-1], y=h.get("volume_mm3", h.get("counts")),
                                 name=_trace_label(run_key, roi_name), opacity=0.6))
    if not any_trace:
        return None
    fig.update_layout(
        barmode="overlay", xaxis_title="Angle between E_ch1 and E_ch2 (degrees)",
        yaxis_title="Tissue volume (mm³)",
        margin=dict(l=10, r=10, t=30, b=10), height=340, legend=dict(font=dict(size=11)),
    )
    return fig


def _build_stats_table(result):
    rows = []
    for run_key, run in result["runs"].items():
        if "error" in run:
            rows.append({"run": _run_display_name(run_key), "region": "—", "n_elements": "",
                        "mean": "", "median": "", "p90": "", "p95": "", "p99": "", "max": "",
                        "pct_above": f"ERROR: {run['error']}"})
            continue
        for roi_name, entry in run["roi"].items():
            st = entry.get("stats")
            if st is None:
                rows.append({"run": _run_display_name(run_key), "region": roi_name,
                            "n_elements": 0, "mean": "", "median": "", "p90": "", "p95": "",
                            "p99": "", "max": "", "pct_above": "(no matching elements)"})
                continue
            row = {"run": _run_display_name(run_key), "region": roi_name, "n_elements": st["n_elements"],
                  "mean": f"{st['mean']:.4f}", "median": f"{st['median']:.4f}",
                  "p90": f"{st['p90']:.4f}", "p95": f"{st['p95']:.4f}", "p99": f"{st['p99']:.4f}",
                  "max": f"{st['max']:.4f}"}
            pct = entry.get("pct_above_threshold")
            row["pct_above"] = f"{pct:.1f}%" if pct is not None else ""
            if "angle_stats" in entry:
                row["angle_mean_deg"] = f"{entry['angle_stats']['mean']:.1f}"
            if "n_runs" in entry:
                row["n_runs"] = entry["n_runs"]
            rows.append(row)
    return rows


@callback(
    Output("an-job-store", "data", allow_duplicate=True),
    Output("an-interval", "disabled", allow_duplicate=True),
    Output("an-build-status", "children", allow_duplicate=True),
    Output("an-distribution-results", "children"),
    Output("an-results-store", "data"),
    Input("an-interval", "n_intervals"),
    State("an-job-store", "data"),
    State("an-plot-type", "value"),
    State("an-normalize-hist", "value"),
    prevent_initial_call=True,
)
def _poll_build_job(_n_intervals, job_dir, plot_type, normalize_value):
    if not job_dir:
        return job_dir, True, dash.no_update, dash.no_update, dash.no_update
    status = jr.read_status(job_dir)
    if not status or status["state"] == "running":
        return job_dir, False, html.Div("… building", style={"color": "#666"}), dash.no_update, dash.no_update
    if status["state"] == "error":
        return (job_dir, True, html.Div(f"✗ {status['error']}", style={"color": "#a00"}),
               dash.no_update, dash.no_update)

    result = status["result"] or {}
    if not result.get("success"):
        return (job_dir, True, html.Div(f"✗ {result.get('error')}", style={"color": "#a00"}),
               dash.no_update, dash.no_update)

    normalize_hist = "normalize" in (normalize_value or [])
    children = _render_results(result, plot_type, normalize_hist)
    return job_dir, True, html.Div("✓ done", style={"color": "#060"}), children, result


def _render_results(result, plot_type, normalize_hist=False):
    ti_fig = _build_ti_figure(result, plot_type, normalize_hist)
    angle_fig = _build_angle_figure(result)
    stats_rows = _build_stats_table(result)
    angle_col = [{"name": "Angle mean (deg)", "id": "angle_mean_deg"}] if \
        any("angle_mean_deg" in r for r in stats_rows) else []
    n_runs_col = [{"name": "N runs summed", "id": "n_runs"}] if \
        any("n_runs" in r for r in stats_rows) else []

    children = [dcc.Graph(figure=ti_fig)]
    if angle_fig is not None:
        children += [html.H4("E-field angle (comparison-run selections only)"), dcc.Graph(figure=angle_fig)]
    children.append(_styled_table("an-stats-table", [
        {"name": "Region", "id": "region"}, {"name": "Run", "id": "run"},
        {"name": "N elements", "id": "n_elements"}, {"name": "Mean", "id": "mean"},
        {"name": "Median", "id": "median"}, {"name": "p90", "id": "p90"}, {"name": "p95", "id": "p95"},
        {"name": "p99", "id": "p99"}, {"name": "Max", "id": "max"},
        {"name": "% above threshold", "id": "pct_above"},
    ] + angle_col + n_runs_col, data=stats_rows))
    return children


@callback(
    Output("an-distribution-results", "children", allow_duplicate=True),
    Input("an-plot-type", "value"),
    Input("an-normalize-hist", "value"),
    State("an-results-store", "data"),
    prevent_initial_call=True,
)
def _on_display_options_change(plot_type, normalize_value, result):
    if not result:
        return dash.no_update
    normalize_hist = "normalize" in (normalize_value or [])
    return _render_results(result, plot_type, normalize_hist)


# ═════════════════════════════════════════════════════════════════════════════
# CSV export
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("an-download", "data"),
    Output("an-export-note", "children"),
    Input("an-export-btn", "n_clicks"),
    State("an-results-store", "data"),
    State("an-export-filename", "value"),
    prevent_initial_call=True,
)
def _on_export_click(_n_clicks, result, filename):
    if not result:
        return dash.no_update, "No distribution results to export yet — build one first."

    import csv
    import io

    rows = _build_stats_table(result)
    cols = ["region", "run", "n_elements", "mean", "median", "p90", "p95", "p99", "max", "pct_above"]
    if any("angle_mean_deg" in r for r in rows):
        cols.append("angle_mean_deg")
    if any("n_runs" in r for r in rows):
        cols.append("n_runs")

    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(cols)
    for r in rows:
        writer.writerow([r.get(c, "") for c in cols])

    name = (filename or "").strip() or f"analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
    if not name.lower().endswith(".csv"):
        name += ".csv"
    return dcc.send_string(buf.getvalue(), filename=name), ""


# ═════════════════════════════════════════════════════════════════════════════
# ROI Preview — MRI slice (viz_discovery.build_slice_frames, same engine as
# FEM Validation's Slice Viewer, same slice_viewer_ui.py component) or 3D
# render (viz_discovery.build_figure_subprocess, same engine as FEM
# Validation's Render Figure), aimed at one already-selected run + region
# instead of asking the user to re-pick everything from scratch.
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("an-preview-run-dropdown", "options"),
    Input("an-runs-table", "data"),
    Input("an-runs-table", "selected_rows"),
)
def _load_preview_run_options(rows, selected_rows):
    rows = rows or []
    return [{"label": f"{rows[i]['subject']}: {rows[i]['label']} [{rows[i]['source']}]", "value": i}
            for i in (selected_rows or []) if i < len(rows)]


@callback(
    Output("an-preview-roi-dropdown", "options"),
    Input("an-roi-dropdown", "value"),
)
def _load_preview_roi_options(roi_names):
    return [{"label": n, "value": n} for n in (roi_names or [])]


@callback(
    Output("an-preview-cap-row", "style"),
    Input("an-preview-mode", "value"),
)
def _toggle_preview_cap_row(mode):
    base = {"marginBottom": "0.5rem", "maxWidth": "360px"}
    return base if mode == "3d" else {**base, "display": "none"}


@callback(
    Output("an-preview-cap-dropdown", "options"),
    Output("an-preview-cap-dropdown", "value"),
    Output("an-preview-cap-note", "children"),
    Input("an-preview-run-dropdown", "value"),
    Input("an-preview-mode", "value"),
    State("an-runs-table", "data"),
)
def _load_preview_cap_options(run_idx, mode, rows):
    if mode != "3d" or run_idx is None or not rows or run_idx >= len(rows):
        return [], None, ""
    r = rows[run_idx]
    names = [r["ch1_plus"], r["ch1_minus"], r["ch2_plus"], r["ch2_minus"]]
    caps = vz.find_matching_caps(r["subject"], names)
    if not caps:
        return [], None, html.Div(f"✗ no registered cap for sub-{r['subject']} contains all of "
                                  f"{names} — register the right cap first.", style={"color": "#a00"})
    note = (html.Div(f"{len(caps)} caps contain these electrode names — pick the right one.",
                     style={"color": "#a60"}) if len(caps) > 1
           else html.Div("✓ exactly one matching cap.", style={"color": "#060"}))
    options = [{"label": c, "value": c} for c in caps]
    return options, (caps[0] if len(caps) == 1 else None), note


@callback(
    Output("an-preview-button", "disabled"),
    Input("an-preview-run-dropdown", "value"),
    Input("an-preview-roi-dropdown", "value"),
    Input("an-preview-mode", "value"),
    Input("an-preview-cap-dropdown", "value"),
)
def _toggle_preview_button(run_idx, roi_name, mode, cap_name):
    if run_idx is None or not roi_name:
        return True
    if mode == "3d" and not cap_name:
        return True
    return False


@callback(
    Output("an-preview-job-store", "data"),
    Output("an-preview-mode-store", "data"),
    Output("an-preview-interval", "disabled"),
    Output("an-preview-status", "children"),
    Input("an-preview-button", "n_clicks"),
    State("an-preview-run-dropdown", "value"),
    State("an-preview-roi-dropdown", "value"),
    State("an-preview-mode", "value"),
    State("an-preview-cap-dropdown", "value"),
    State("an-runs-table", "data"),
    prevent_initial_call=True,
)
def _on_preview_click(_n_clicks, run_idx, roi_name, mode, cap_name, rows):
    if run_idx is None or not roi_name or not rows or run_idx >= len(rows):
        return None, mode, True, html.Div("Select a run and a region first.", style={"color": "#a00"})
    r = rows[run_idx]
    subject_id, msh_path = r["subject"], r["msh_path"]

    from common import PROJECT_DIR

    if mode == "slice":
        base_dir = os.path.join(PROJECT_DIR, "derivatives", "SimNIBS", f"sub-{subject_id}",
                                "comparison", "figures", "_jobs")
        _job_id, job_dir = jr.new_job_dir(base_dir)
        # roi_only=True (unlike the Slice Viewer's own default) — this is
        # specifically "where is this region", a location question, so
        # coloring only inside it against an otherwise-grayscale backdrop
        # makes that location the most legible thing on screen.
        jr.start_local_job(job_dir, vz.build_slice_frames, subject_id, msh_path, [roi_name], roi_only=True)
        return job_dir, mode, False, html.Div("Building — polling every 2s...", style={"color": "#666"})

    if not cap_name:
        return None, mode, True, html.Div("Select a cap first.", style={"color": "#a00"})
    electrodes_csv = cd.registered_cap_path(subject_id, cap_name)
    if not os.path.isfile(electrodes_csv):
        return None, mode, True, html.Div(f"Registered cap CSV not found: {electrodes_csv}",
                                          style={"color": "#a00"})
    out_path = os.path.join(PROJECT_DIR, "derivatives", "SimNIBS", f"sub-{subject_id}",
                            "comparison", "figures", f"{r['label']}_{roi_name}_preview_3views.png")
    base_dir = os.path.join(PROJECT_DIR, "derivatives", "SimNIBS", f"sub-{subject_id}",
                            "comparison", "figures", "_jobs")
    _job_id, job_dir = jr.new_job_dir(base_dir)
    jr.start_local_job(
        job_dir, vz.build_figure_subprocess, subject_id, msh_path,
        (r["ch1_plus"], r["ch1_minus"]), (r["ch2_plus"], r["ch2_minus"]),
        electrodes_csv, [roi_name], out_path, 1.0,
    )
    return job_dir, mode, False, html.Div("Rendering — polling every 2s...", style={"color": "#666"})


@callback(
    Output("an-preview-job-store", "data", allow_duplicate=True),
    Output("an-preview-interval", "disabled", allow_duplicate=True),
    Output("an-preview-status", "children", allow_duplicate=True),
    Output("an-preview-slice-viewer-container", "style"),
    Output("an-preview-colorbar-img", "src"),
    Output("an-preview-legend-row", "children"),
    Output("an-preview-slice-axial-frames", "data"), Output("an-preview-slice-coronal-frames", "data"),
    Output("an-preview-slice-sagittal-frames", "data"),
    Output("an-preview-slice-axial-slider", "max"), Output("an-preview-slice-axial-slider", "value"),
    Output("an-preview-slice-coronal-slider", "max"), Output("an-preview-slice-coronal-slider", "value"),
    Output("an-preview-slice-sagittal-slider", "max"), Output("an-preview-slice-sagittal-slider", "value"),
    Output("an-preview-3d-results", "children"),
    Input("an-preview-interval", "n_intervals"),
    State("an-preview-job-store", "data"),
    State("an-preview-mode-store", "data"),
    prevent_initial_call=True,
)
def _poll_preview_job(_n_intervals, job_dir, mode):
    no_slice_change = (dash.no_update,) * 11
    if not job_dir:
        return (job_dir, True, dash.no_update, {"display": "none"}) + no_slice_change + (dash.no_update,)
    status = jr.read_status(job_dir)
    if not status or status["state"] == "running":
        return (job_dir, False, html.Div("… working", style={"color": "#666"}),
               dash.no_update) + no_slice_change + (dash.no_update,)
    if status["state"] == "error":
        return (job_dir, True, html.Div(f"✗ {status['error']}", style={"color": "#a00"}),
               dash.no_update) + no_slice_change + (dash.no_update,)

    result = status["result"] or {}
    if not result.get("success"):
        return (job_dir, True, html.Div(f"✗ {result.get('error')}", style={"color": "#a00"}),
               dash.no_update) + no_slice_change + (dash.no_update,)

    if mode == "3d":
        import base64
        with open(result["out_path"], "rb") as f:
            b64 = base64.b64encode(f.read()).decode()
        fr = result.get("field_range") or {}
        note = (f"✓ drew: {', '.join(result['regions_drawn'])}  —  "
               f"whole-mesh {fr.get('min', 0):.3f}-{fr.get('max', 0):.3f} V/m")
        children = html.Div([
            html.P(note, style={"color": "#060", "fontSize": "13px"}),
            html.Img(src=f"data:image/png;base64,{b64}", style={"maxWidth": "100%"}),
            html.Div(html.Button("Download PNG", id="an-preview-3d-download-button", n_clicks=0),
                     style={"marginTop": "0.5rem"}),
            dcc.Store(id="an-preview-3d-out-path", data=result["out_path"]),
        ])
        return (job_dir, True, html.Div("✓ done", style={"color": "#060"}),
               {"display": "none"}) + no_slice_change + (children,)

    # slice mode
    fr = result.get("field_range") or {}
    regions_drawn = result.get("regions_drawn") or []
    regions = ", ".join(regions_drawn) or "(none — no mask matched for this subject)"
    note = html.Div(
        f"✓ outlined: {regions}  —  whole-mesh {fr.get('min', 0):.3f}-{fr.get('max', 0):.3f} V/m, "
        f"color scale to {result.get('vmax_used', 0):.3f} V/m (99th percentile)",
        style={"color": "#060"})
    legend = html.Div([
        html.Span([
            html.Span(style={"display": "inline-block", "width": "11px", "height": "11px",
                            "backgroundColor": f"rgb{vz.SLICE_ROI_COLORS[i % len(vz.SLICE_ROI_COLORS)]}",
                            "marginRight": "4px", "verticalAlign": "middle", "borderRadius": "2px"}),
            label,
        ], style={"marginRight": "1.25rem", "whiteSpace": "nowrap"})
        for i, label in enumerate(regions_drawn)
    ], style={"display": "flex", "flexWrap": "wrap"}) if regions_drawn else ""

    frames = result["frames"]
    n = result["n_slices"]
    start = result.get("roi_slice", result["mid_slice"])
    return (
        job_dir, True, note, {"display": "flex", "flexWrap": "wrap", "marginTop": "1rem"},
        f"data:image/png;base64,{result['colorbar_png']}", legend,
        frames["Axial"], frames["Coronal"], frames["Sagittal"],
        n["Axial"] - 1, start["Axial"], n["Coronal"] - 1, start["Coronal"],
        n["Sagittal"] - 1, start["Sagittal"], "",
    )


@callback(
    Output("an-preview-3d-download", "data"),
    Input("an-preview-3d-download-button", "n_clicks"),
    State("an-preview-3d-out-path", "data"),
    prevent_initial_call=True,
)
def _on_preview_3d_download_click(_n_clicks, out_path):
    if not out_path:
        return dash.no_update
    return dcc.send_file(out_path)

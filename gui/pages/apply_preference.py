"""
pages/apply_preference.py — run a saved Config Generation Preference
(preset_discovery.py) against one or more subjects: charm (opt-in,
skipped if m2m already exists) -> masks (ROI, non-ROI, every subgroup/
background constraint mask, from the preference's own frozen recipes) ->
cap registration -> config generation, via one_shot_discovery.py. Never
submits the actual optimization/simulation job — that stays a separate,
manual step on Run Pipeline afterward, same as every other job-submitting
page in this app.
"""
import os

import dash
from dash import html, dcc, dash_table, callback, Input, Output, State

import job_runner as jr
import one_shot_discovery as osd
import preset_discovery as pd
from common import PROJECT_DIR, discover_subjects

dash.register_page(__name__, path="/apply-preference", name="Apply Preference", category="Optimization", order=2)

STEP_LABELS = {
    "charm":            "Head modeling (charm)",
    "masks":            "Masks (ROI/non-ROI/constraints)",
    "cap_registration": "Cap registration",
    "config_generation": "Config generation",
}


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
    html.H2("Apply Preference"),
    html.P("Runs a saved Config Generation Preference against one or more subjects: head modeling "
           "(optional, local only — see the toggle below) → ROI/non-ROI/extra masks → cap "
           "registration → config generation, in that order, stopping a subject's own chain at the "
           "first failing step. Never submits the actual optimization/simulation job — that stays a "
           "separate, manual step on Run Pipeline once the configs exist.",
           style={"fontSize": "13px", "color": "#666"}),

    html.Div([
        html.Label("Preference"),
        dcc.Dropdown(id="ap-preset-dropdown", placeholder="Select a saved preference..."),
    ], style={"maxWidth": "420px", "marginBottom": "0.5rem"}),
    html.Div(id="ap-preset-summary", style={"fontSize": "13px", "color": "#666", "marginBottom": "1rem"}),

    html.Div([
        html.Label("Subjects"),
        dcc.Dropdown(id="ap-subject-dropdown", multi=True, placeholder="Select subject(s)..."),
    ], style={"maxWidth": "600px", "marginBottom": "0.75rem"}),

    dcc.Checklist(
        id="ap-run-charm-locally",
        options=[{"label": " Run charm locally for any subject whose m2m doesn't exist yet "
                  "(otherwise that subject's chain stops there with a message instead of running "
                  "a multi-minute operation unannounced)", "value": "run_charm"}],
        value=[], style={"marginBottom": "1rem"},
    ),

    html.Div(id="ap-preview", style={"marginBottom": "1rem"}),

    html.Button("Apply", id="ap-apply-button", n_clicks=0, disabled=True),
    dcc.Store(id="ap-job-store"),
    dcc.Interval(id="ap-interval", interval=2000, disabled=True),
    html.Div(id="ap-progress", style={"marginTop": "1rem"}),
    dcc.Loading(html.Div(id="ap-results", style={"marginTop": "1rem"})),
])


# ═════════════════════════════════════════════════════════════════════════════
# Preset / subject pickers
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("ap-preset-dropdown", "options"),
    Input("ap-preset-dropdown", "id"),  # fires once on page load
)
def _load_presets(_):
    return [{"label": r["name"], "value": r["name"]} for r in pd.list_presets()]


@callback(
    Output("ap-preset-summary", "children"),
    Input("ap-preset-dropdown", "value"),
)
def _show_preset_summary(name):
    if not name:
        return ""
    preset = pd.load_preset(name)
    if not preset:
        return html.Span(f'✗ could not load "{name}"', style={"color": "#a00"})
    roi_name, non_roi_name = preset.get("roi_name"), preset.get("non_roi_name")
    masks = preset.get("masks") or {}
    other_masks = sorted(n for n in masks if n not in (roi_name, non_roi_name))

    def _mask_bit(name):
        recipe = masks.get(name) or {}
        return f"{name} ({recipe.get('atlas') or recipe.get('kind') or '?'})"

    bits = [
        f"Cap: {os.path.basename(preset.get('cap_path') or '?')}",
        f"ROI: {_mask_bit(roi_name) if roi_name else '?'}",
        f"non-ROI: {_mask_bit(non_roi_name) if non_roi_name else '?'}",
        f"Other masks (constraints/background): {', '.join(other_masks) if other_masks else '(none)'}",
        f"Goals: {', '.join(preset.get('goals') or [])}",
    ]
    return html.Ul([html.Li(b) for b in bits], style={"margin": 0, "paddingLeft": "1.25rem"})


@callback(Output("ap-subject-dropdown", "options"), Input("ap-subject-dropdown", "id"))
def _load_subjects(_):
    return [{"label": s["subject_id"], "value": s["subject_id"]} for s in discover_subjects()]


@callback(
    Output("ap-apply-button", "disabled"),
    Input("ap-preset-dropdown", "value"),
    Input("ap-subject-dropdown", "value"),
)
def _toggle_apply_button(preset_name, subject_ids):
    return not (preset_name and subject_ids)


_CHARM_STATUS_DISPLAY = {
    "done":             "✓ already done",
    "will_run_locally": "will run locally",
    "needs_manual":     "✗ needs to run first — enable \"run charm locally\" above, or run it via "
                        "Head Modeling, then come back",
}


@callback(
    Output("ap-preview", "children"),
    Input("ap-preset-dropdown", "value"),
    Input("ap-subject-dropdown", "value"),
    Input("ap-run-charm-locally", "value"),
)
def _render_preview(preset_name, subject_ids, run_charm_value):
    """A read-only dry run, BEFORE Apply is clicked — green for whatever's
    already done, a highlighted note for whatever Apply would actually do
    (build a mask, register the cap, or run charm — only locally if the
    toggle above is on, otherwise a clear "run it first" instruction
    instead of silently blocking later)."""
    if not preset_name or not subject_ids:
        return ""
    preset = pd.load_preset(preset_name)
    if not preset:
        return ""
    run_charm_locally = "run_charm" in (run_charm_value or [])

    rows = []
    for sid in subject_ids:
        p = osd.preview_preference_for_subject(sid, preset, run_charm_locally)
        rows.append({"subject": sid, "step": "charm", "status": _CHARM_STATUS_DISPLAY[p["charm"]["status"]]})
        for m in p["masks"]:
            rows.append({"subject": sid, "step": f"mask: {m['name']}",
                        "status": "✓ already exists" if m["exists"] else "will be built"})
        if p["cap_registered"] is None:
            rows.append({"subject": sid, "step": "cap registration",
                        "status": "✗ this preference has no cap_path"})
        else:
            rows.append({"subject": sid, "step": "cap registration",
                        "status": "✓ already registered" if p["cap_registered"] else "will be registered"})

    return html.Div([
        html.H4("Preview", style={"marginBottom": "0.1rem"}),
        html.P("Green = already done. Yellow = not done yet, Apply will do it. Red = blocks the "
               "chain for that subject until you address it.",
               style={"fontSize": "12px", "color": "#666", "marginTop": 0, "marginBottom": "0.5rem"}),
        _styled_table("ap-preview-table", [
            {"name": "Subject", "id": "subject"}, {"name": "Step", "id": "step"}, {"name": "Status", "id": "status"},
        ], data=rows, style_data_conditional=[
            {"if": {"filter_query": '{status} contains "✓"'}, "backgroundColor": "#e6ffe6"},
            {"if": {"filter_query": '{status} contains "✗"'}, "backgroundColor": "#ffe0e0"},
            {"if": {"filter_query": '{status} contains "will be built"'}, "backgroundColor": "#fff8e1"},
            {"if": {"filter_query": '{status} contains "will be registered"'}, "backgroundColor": "#fff8e1"},
            {"if": {"filter_query": '{status} contains "will run locally"'}, "backgroundColor": "#fff8e1"},
        ]),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# Apply (background job — charm alone can run long, this must never block
# the page behind a synchronous callback)
# ═════════════════════════════════════════════════════════════════════════════

APPLY_JOB_BASE_DIR = os.path.join(PROJECT_DIR, "derivatives", "SimNIBS", "_apply_preference_jobs")


@callback(
    Output("ap-job-store", "data"),
    Output("ap-interval", "disabled"),
    Output("ap-progress", "children"),
    Output("ap-results", "children"),
    Input("ap-apply-button", "n_clicks"),
    State("ap-preset-dropdown", "value"),
    State("ap-subject-dropdown", "value"),
    State("ap-run-charm-locally", "value"),
    prevent_initial_call=True,
)
def _on_apply_click(_n_clicks, preset_name, subject_ids, run_charm_value):
    preset = pd.load_preset(preset_name) if preset_name else None
    if not preset:
        return None, True, html.Div(f'✗ could not load preference "{preset_name}"', style={"color": "#a00"}), ""
    if not subject_ids:
        return None, True, html.Div("Select at least one subject.", style={"color": "#a00"}), ""

    run_charm_locally = "run_charm" in (run_charm_value or [])
    _job_id, job_dir = jr.new_job_dir(APPLY_JOB_BASE_DIR)
    jr.start_local_job(job_dir, osd.apply_preference_job, job_dir, subject_ids, preset, run_charm_locally)
    return job_dir, False, html.Div("Starting…", style={"color": "#666"}), ""


def _step_status_row(step, info):
    if info is None:
        return html.Span("— ", style={"color": "#bbb"})
    if info.get("success"):
        tag = "✓ (skipped)" if info.get("skipped") else "✓"
        color = "#060"
    else:
        tag = f"✗ {info.get('error', '')}"
        color = "#a00"
    return html.Span(tag, style={"color": color})


def _render_progress(status):
    """A live, fixed-order step checklist for whichever subject is
    currently being processed — status["subjects"] only has entries for
    subjects job_runner has ALREADY finished (or is mid-way through, for
    the current one), so this always reflects real progress, never a
    guess."""
    current_subject = status.get("current_subject")
    current_step = status.get("current_step")
    subjects = status.get("subjects") or {}
    if not current_subject:
        return html.Div("… working", style={"color": "#666"})

    current_steps = (subjects.get(current_subject) or {}).get("steps", {})
    rows = []
    for step in osd.STEPS:
        if step in current_steps:
            cell = _step_status_row(step, current_steps[step])
        elif step == current_step:
            cell = html.Span("… running", style={"color": "#666"})
        else:
            cell = html.Span("—", style={"color": "#bbb"})
        rows.append(html.Div([
            html.Span(STEP_LABELS[step], style={"display": "inline-block", "minWidth": "180px"}),
            cell,
        ]))
    total = status.get("total_subjects") or len(subjects)
    completed = sum(1 for r in subjects.values() if r.get("success") is not None)
    return html.Div([
        html.P(f"Subject {current_subject} ({completed}/{total} completed)", style={"fontWeight": "bold"}),
        html.Div(rows, style={"fontFamily": "monospace", "fontSize": "13px"}),
    ])


def _render_results(subjects: dict):
    rows = []
    for sid, result in subjects.items():
        for step in osd.STEPS:
            info = result["steps"].get(step)
            if info is None:
                continue
            rows.append({
                "subject": sid, "step": STEP_LABELS[step],
                "status": "skipped" if info.get("skipped") else ("ok" if info.get("success") else "FAILED"),
                "detail": info.get("error") or "",
            })
    overall = [{"subject": sid, "step": "— OVERALL —",
               "status": "success" if r["success"] else f"stopped at {r['stopped_at']}",
               "detail": ""} for sid, r in subjects.items()]
    return html.Div([
        html.H4("Results"),
        _styled_table("ap-results-table", [
            {"name": "Subject", "id": "subject"}, {"name": "Step", "id": "step"},
            {"name": "Status", "id": "status"}, {"name": "Detail", "id": "detail"},
        ], data=overall + rows, style_data_conditional=[
            {"if": {"filter_query": '{status} contains "FAILED"'}, "backgroundColor": "#ffe0e0"},
            {"if": {"filter_query": '{status} contains "stopped"'}, "backgroundColor": "#ffe0e0"},
        ]),
    ])


@callback(
    Output("ap-job-store", "data", allow_duplicate=True),
    Output("ap-interval", "disabled", allow_duplicate=True),
    Output("ap-progress", "children", allow_duplicate=True),
    Output("ap-results", "children", allow_duplicate=True),
    Input("ap-interval", "n_intervals"),
    State("ap-job-store", "data"),
    prevent_initial_call=True,
)
def _poll_apply_job(_n_intervals, job_dir):
    if not job_dir:
        return job_dir, True, dash.no_update, dash.no_update
    status = jr.read_status(job_dir)
    if not status:
        return job_dir, False, html.Div("… working", style={"color": "#666"}), dash.no_update
    if status["state"] == "error":
        return job_dir, True, html.Div(f"✗ {status['error']}", style={"color": "#a00"}), dash.no_update
    if status["state"] == "running":
        return job_dir, False, _render_progress(status), dash.no_update

    result = status.get("result") or {}
    subjects = result.get("subjects") or {}
    n_ok = sum(1 for r in subjects.values() if r["success"])
    note = html.Div(f"✓ done — {n_ok}/{len(subjects)} subject(s) completed the full chain",
                    style={"color": "#060" if n_ok == len(subjects) else "#a60"})
    return job_dir, True, note, _render_results(subjects)

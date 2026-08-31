"""
pages/config_generation.py — Phase 5: build code/pipeline/configs/*.json the
same way code/pipeline/generate_configs.py does, from a GUI form instead of
editing that script's hardcoded SUBJECTS/ROIS/etc. lists by hand.

ROI and non-ROI are each a single union of regions (one "Atlas -> Region"
picker per side, same pattern as Mask Generation) — "multiple ROIs" in the
underlying pipeline means a union of labels within one ROIConfig, not
multiple separate ROIConfig objects.

Allen atlas is a special case: run_pipeline.py's Section 1 has no code path
that reads Allen's label space (only bna_labels -> BNA, or labels -> the
FreeSurfer/charm aseg label space). An Allen-sourced ROI/non-ROI must
already exist on disk as a mask file (built via Mask Generation, which now
shares the same sub-{id}_label-{name}_mask.nii.gz naming for ROI/non-ROI/
general) so Section 1's own skip-if-exists check picks it up unchanged.

Only commonly-tweaked optimizer/electrode fields are exposed; everything
else is fixed at generate_configs.py's current defaults (see
config_discovery.OPTIMIZER_DEFAULTS / ELECTRODE_DEFAULTS) — the
TesFlexOptimization-only fields (population size, mutation, etc.) no longer
exist in OptimizerConfig at all, see code/pipeline/config.py.
"""
import os
from datetime import datetime

import dash
from dash import html, dcc, dash_table, callback, Input, Output, State, ctx

import discovery


def _stepper_row(id_, value, step, minimum, width="70px"):
    """A number dcc.Input paired with explicit -/+ buttons — some browsers'
    native spin-button clicks don't reliably register inside a Dash
    dcc.Input, so these buttons (wired via _register_stepper_callback
    below) are a guaranteed-to-work alternative alongside typing directly."""
    return html.Div([
        html.Button("−", id=f"{id_}-dec", n_clicks=0,
                    style={"width": "26px", "padding": "0"}),
        dcc.Input(id=id_, type="number", min=minimum, step=step, value=value,
                 style={"width": width, "textAlign": "center", "margin": "0 0.25rem"}),
        html.Button("+", id=f"{id_}-inc", n_clicks=0,
                    style={"width": "26px", "padding": "0"}),
    ], style={"display": "flex", "alignItems": "center"})


def _register_stepper_callback(id_, step, minimum, maximum=None):
    @callback(
        Output(id_, "value"),
        Input(f"{id_}-dec", "n_clicks"),
        Input(f"{id_}-inc", "n_clicks"),
        State(id_, "value"),
        prevent_initial_call=True,
    )
    def _step(_n_dec, _n_inc, current):
        try:
            current = float(current)
        except (TypeError, ValueError):
            current = minimum if minimum is not None else 0
        new_val = current + (step if ctx.triggered_id == f"{id_}-inc" else -step)
        if minimum is not None:
            new_val = max(minimum, new_val)
        if maximum is not None:
            new_val = min(maximum, new_val)
        if float(step).is_integer():
            new_val = int(round(new_val))
        return new_val
    return _step
import cap_discovery
import config_discovery as cd

dash.register_page(__name__, path="/config-generation", name="Config Generation",
                   category="Optimization", order=1)

ATLAS_NAMES = list(discovery.ATLAS_REGISTRY.keys())


def _styled_table(id_, columns, data=None, **kwargs):
    return dash_table.DataTable(
        id=id_,
        columns=columns,
        data=data or [],
        style_cell={"textAlign": "left", "fontFamily": "monospace", "fontSize": "13px", "padding": "4px"},
        style_table={"overflowX": "auto"},
        **kwargs,
    )


# ═════════════════════════════════════════════════════════════════════════════
# ROI / non-ROI block — one layout + one set of callbacks per prefix
# ═════════════════════════════════════════════════════════════════════════════

def _region_block_layout(prefix, title):
    return html.Div([
        html.H4(title),
        dcc.Store(id=f"{prefix}-region-lut-store"),
        dcc.Store(id=f"{prefix}-selected-label-ids-store", data={}),

        html.Label("Name (short slug used in the config filename, e.g. \"hippo_r_phg\")"),
        dcc.Input(id=f"{prefix}-name-input", type="text", style={"width": "100%"}),

        html.Label("...or pick an already-generated mask (any atlas, or a Mask Generation "
                   "combined mask) — fills in the name above and skips atlas/region selection "
                   "entirely, same as an Allen-sourced mask", style={"marginTop": "0.5rem", "display": "block"}),
        dcc.Dropdown(id=f"{prefix}-existing-mask-dropdown", placeholder="Pick an existing mask..."),

        html.Label("Atlas source", style={"marginTop": "0.5rem", "display": "block"}),
        dcc.Dropdown(
            id=f"{prefix}-atlas-dropdown",
            options=[
                {"label": name + ("" if meta["usable"] else "  (not yet usable)"),
                 "value": name, "disabled": not meta["usable"]}
                for name, meta in discovery.ATLAS_REGISTRY.items()
            ],
            placeholder="Select atlas...",
        ),
        html.Div(id=f"{prefix}-atlas-note", style={"fontStyle": "italic", "fontSize": "13px", "margin": "0.5rem 0"}),
        html.Div(id=f"{prefix}-region-picker"),
    ], style={"flex": "1 1 380px", "border": "1px solid #ddd", "borderRadius": "6px",
              "padding": "1rem", "marginRight": "1rem"})


def _register_region_callbacks(prefix):

    @callback(
        Output(f"{prefix}-atlas-note", "children"),
        Output(f"{prefix}-region-picker", "children"),
        Output(f"{prefix}-region-lut-store", "data"),
        Input("cg-subject-dropdown", "value"),
        Input(f"{prefix}-atlas-dropdown", "value"),
    )
    def _update_atlas_panel(subject_ids, atlas_name):
        if not subject_ids or not atlas_name:
            return "", html.Div("Select at least one subject and an atlas first."), None

        meta = discovery.ATLAS_REGISTRY[atlas_name]

        if cd.uses_allen(atlas_name):
            note = ("Allen-sourced masks aren't built by this pipeline directly — generate the "
                    "mask first via Mask Generation, then pick it from the \"...or pick an "
                    "already-generated mask\" dropdown above. The Subject Readiness table shows "
                    "whether it already exists for every selected subject.")
            return note, html.Div(), None

        not_ready = [sid for sid in subject_ids if not discovery.full_atlas_check(atlas_name, sid)["ready"]]

        if not meta["has_lut"]:
            region_ui = html.Div([
                html.P("No name lut for this atlas in the project — enter numeric label ids "
                       "directly, one 'Display Name: id' pair per line:"),
                dcc.Textarea(id=f"{prefix}-bna-region-input", style={"width": "100%", "height": "100px"},
                             placeholder="PhG_R: 116\nrHippocampus_R: 216\ncHippocampus_R: 218"),
                html.Div(id=f"{prefix}-bna-region-preview"),
            ])
            return meta["note"], region_ui, None

        if not_ready:
            region_ui = html.Div(
                "Region list not shown — these subjects are missing required files for "
                f"{atlas_name}: {', '.join(not_ready)}",
                style={"color": "#a00"})
            return meta["note"], region_ui, None

        try:
            lut = discovery.build_lut(atlas_name, subject_ids[0])
        except Exception as e:
            return meta["note"], html.Div(f"Failed to load region list: {e}", style={"color": "#a00"}), None

        options = [{"label": f"{name}  (id {rid})", "value": rid}
                   for rid, name in sorted(lut.items(), key=lambda kv: kv[1])]
        region_ui = dcc.Dropdown(id=f"{prefix}-region-dropdown", options=options, multi=True,
                                  placeholder="Search regions...")
        return meta["note"], region_ui, lut

    @callback(
        Output(f"{prefix}-bna-region-preview", "children"),
        Input(f"{prefix}-bna-region-input", "value"),
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

    @callback(
        Output(f"{prefix}-selected-label-ids-store", "data", allow_duplicate=True),
        Input(f"{prefix}-atlas-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _reset_on_atlas_change(_atlas_name):
        return {}

    @callback(
        Output(f"{prefix}-selected-label-ids-store", "data", allow_duplicate=True),
        Input(f"{prefix}-region-dropdown", "value"),
        State(f"{prefix}-region-lut-store", "data"),
        prevent_initial_call=True,
    )
    def _sync_lut_selection(selected_ids, lut):
        if not selected_ids or not lut:
            return {}
        lut = {int(k): v for k, v in lut.items()}
        return {lut[rid]: rid for rid in selected_ids if rid in lut}

    @callback(
        Output(f"{prefix}-selected-label-ids-store", "data", allow_duplicate=True),
        Input(f"{prefix}-bna-region-input", "value"),
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

    @callback(
        Output(f"{prefix}-existing-mask-dropdown", "options"),
        Input("cg-subject-dropdown", "value"),
    )
    def _load_existing_mask_options(subject_ids):
        return [{"label": n, "value": n} for n in discovery.list_mask_names(subject_ids or [])]

    @callback(
        Output(f"{prefix}-name-input", "value"),
        Output(f"{prefix}-selected-label-ids-store", "data", allow_duplicate=True),
        Input(f"{prefix}-existing-mask-dropdown", "value"),
        prevent_initial_call=True,
    )
    def _pick_existing_mask(mask_name):
        # Clearing the label-ids store too, not just filling the name field —
        # otherwise a region selection made under a DIFFERENT atlas before
        # picking this mask would still be sitting in the store, and
        # build_roi_dict() would build labels from that stale selection
        # instead of trusting the picked mask file as-is (empty labels,
        # same as today's Allen-only path).
        if not mask_name:
            return dash.no_update, dash.no_update
        return mask_name, {}


_register_region_callbacks("cg-roi")
_register_region_callbacks("cg-nonroi")


# ═════════════════════════════════════════════════════════════════════════════
# Layout
# ═════════════════════════════════════════════════════════════════════════════

layout = html.Div([
    html.H2("Config Generation"),
    html.P("Writes JSON configs to code/pipeline/configs/ — the same folder and format "
           "generate_configs.py itself uses. This page does not edit generate_configs.py's "
           "own SUBJECTS/ROIS lists.", style={"fontSize": "13px", "color": "#666"}),

    html.Div([
        html.Label("Subjects"),
        dcc.Dropdown(id="cg-subject-dropdown", multi=True, placeholder="Select subject(s)..."),
    ], style={"maxWidth": "600px", "marginBottom": "1.5rem"}),

    html.H3("ROI / non-ROI"),
    html.Div([
        _region_block_layout("cg-roi", "ROI"),
        _region_block_layout("cg-nonroi", "non-ROI"),
    ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "1.5rem"}),

    html.H3("Cap"),
    html.Div([
        dcc.Dropdown(id="cg-cap-dropdown", placeholder="Default (Okamoto 10-20, registered by charm)"),
    ], style={"maxWidth": "480px", "marginBottom": "1.5rem"}),

    html.H3("Goals"),
    dcc.Checklist(
        id="cg-goals-checklist",
        options=[{"label": f" {g}", "value": g} for g in cd.GOALS],
        value=list(cd.GOALS), inline=True,
    ),

    html.H3("Optimizer", style={"marginTop": "1.5rem"}),
    html.Div([
        html.Div([
            html.Label("postproc"),
            dcc.Dropdown(id="cg-opt-postproc",
                        options=[{"label": v, "value": v} for v in
                                 ["max_TI", "dir_TI", "magn", "normal", "tangential"]],
                        value=cd.OPTIMIZER_DEFAULTS["postproc"], clearable=False),
        ], style={"minWidth": "160px", "marginRight": "1rem"}),
        html.Div([
            html.Label("cpus"),
            dcc.Input(id="cg-opt-cpus", type="number", min=1, step=1,
                      value=cd.OPTIMIZER_DEFAULTS["cpus"], style={"width": "100%"}),
        ], style={"minWidth": "100px", "marginRight": "1rem"}),
        html.Div([
            html.Label("focality_threshold — max V/m in non-ROI"),
            dcc.Input(id="cg-opt-focality-nonroi", type="number", step=0.01,
                      value=cd.OPTIMIZER_DEFAULTS["focality_threshold"][0], style={"width": "100%"}),
        ], style={"minWidth": "180px", "marginRight": "1rem"}),
        html.Div([
            html.Label("focality_threshold — min V/m in ROI"),
            dcc.Input(id="cg-opt-focality-roi", type="number", step=0.01,
                      value=cd.OPTIMIZER_DEFAULTS["focality_threshold"][1], style={"width": "100%"}),
        ], style={"minWidth": "180px", "marginRight": "1rem"}),
    ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "0.75rem"}),
    dcc.Checklist(
        id="cg-opt-checkboxes",
        options=[
            {"label": " hard_roi_constraint", "value": "hard_roi_constraint"},
            {"label": " no_adjacent_electrodes", "value": "no_adjacent_electrodes"},
        ],
        value=[k for k in ("hard_roi_constraint", "no_adjacent_electrodes") if cd.OPTIMIZER_DEFAULTS[k]],
        inline=True,
    ),

    html.Div([
        dcc.Checklist(
            id="cg-opt-hier-enable",
            options=[{"label": " use_hierarchical_search (coarse-to-fine electrode search)",
                      "value": "use_hierarchical_search"}],
            value=[k for k in ("use_hierarchical_search",) if cd.OPTIMIZER_DEFAULTS[k]],
        ),
        html.Div([
            html.Div([
                html.Label("num_fine_iterations"),
                _stepper_row("cg-opt-hier-num-iter", cd.OPTIMIZER_DEFAULTS["num_fine_iterations"], 1, 0),
            ], style={"minWidth": "160px", "marginRight": "1rem"}),
            html.Div([
                html.Label("neighbours per iteration (comma-separated, one value per fine iteration)"),
                dcc.Input(id="cg-opt-hier-neighbours", type="text", placeholder="e.g. 8,6,4",
                          value="", style={"width": "100%"}),
            ], style={"minWidth": "280px", "marginRight": "1rem"}),
            html.Div([
                html.Label("early_stop_threshold (%)"),
                _stepper_row("cg-opt-hier-early-stop", cd.OPTIMIZER_DEFAULTS["early_stop_threshold"] * 100,
                            0.5, 0),
            ], style={"minWidth": "160px", "marginRight": "1rem"}),
        ], style={"display": "flex", "flexWrap": "wrap", "marginTop": "0.5rem"}),
    ], style={"marginTop": "0.75rem", "marginBottom": "0.75rem", "padding": "0.5rem",
              "border": "1px solid #ccc", "borderRadius": "4px"}),

    html.Div([
        dcc.Checklist(
            id="cg-opt-amp-enable",
            options=[{"label": " use_amplitude_sweep (re-rank the location search's top montages "
                      "over a per-channel current grid)", "value": "use_amplitude_sweep"}],
            value=[k for k in ("use_amplitude_sweep",) if cd.OPTIMIZER_DEFAULTS[k]],
        ),
        html.Div([
            html.Div([
                html.Label("amplitude_sweep_top_n"),
                _stepper_row("cg-opt-amp-top-n", cd.OPTIMIZER_DEFAULTS["amplitude_sweep_top_n"], 1, 1),
            ], style={"minWidth": "160px", "marginRight": "1rem"}),
            html.Div([
                html.Label("min mA"),
                dcc.Input(id="cg-opt-amp-min-ma", type="number", step=0.01,
                          value=cd.OPTIMIZER_DEFAULTS["amplitude_sweep_min_mA"], style={"width": "100%"}),
            ], style={"minWidth": "110px", "marginRight": "1rem"}),
            html.Div([
                html.Label("max mA"),
                dcc.Input(id="cg-opt-amp-max-ma", type="number", step=0.01,
                          value=cd.OPTIMIZER_DEFAULTS["amplitude_sweep_max_mA"], style={"width": "100%"}),
            ], style={"minWidth": "110px", "marginRight": "1rem"}),
            html.Div([
                html.Label("step mA"),
                dcc.Input(id="cg-opt-amp-step-ma", type="number", step=0.01,
                          value=cd.OPTIMIZER_DEFAULTS["amplitude_sweep_step_mA"], style={"width": "100%"}),
            ], style={"minWidth": "110px", "marginRight": "1rem"}),
            html.Div([
                html.Label("max per-pair mA (single-channel ceiling)"),
                dcc.Input(id="cg-opt-amp-max-per-pair-ma", type="number", step=0.1,
                          value=cd.OPTIMIZER_DEFAULTS["amplitude_sweep_max_per_pair_mA"],
                          style={"width": "100%"}),
            ], style={"minWidth": "220px", "marginRight": "1rem"}),
        ], style={"display": "flex", "flexWrap": "wrap", "marginTop": "0.5rem"}),
        html.Div([
            html.Label("Ranking weights — roc / roi_mean / non_roi_mean / focality_ratio "
                       "(should sum to ~1.0; not enforced)",
                       style={"display": "block", "marginTop": "0.5rem"}),
            html.Div([
                dcc.Input(id="cg-opt-amp-weight-roc", type="number", step=0.05,
                          value=cd.OPTIMIZER_DEFAULTS["amplitude_sweep_weights"]["roc"],
                          style={"width": "100%"}),
                dcc.Input(id="cg-opt-amp-weight-roi-mean", type="number", step=0.05,
                          value=cd.OPTIMIZER_DEFAULTS["amplitude_sweep_weights"]["roi_mean"],
                          style={"width": "100%"}),
                dcc.Input(id="cg-opt-amp-weight-non-roi-mean", type="number", step=0.05,
                          value=cd.OPTIMIZER_DEFAULTS["amplitude_sweep_weights"]["non_roi_mean"],
                          style={"width": "100%"}),
                dcc.Input(id="cg-opt-amp-weight-focality-ratio", type="number", step=0.05,
                          value=cd.OPTIMIZER_DEFAULTS["amplitude_sweep_weights"]["focality_ratio"],
                          style={"width": "100%"}),
            ], style={"display": "flex", "gap": "0.5rem", "maxWidth": "480px"}),
            html.Div(id="cg-opt-amp-weight-sum-note", style={"fontSize": "12px", "color": "#666",
                                                             "marginTop": "0.25rem"}),
        ]),
    ], style={"marginTop": "0.75rem", "marginBottom": "0.75rem", "padding": "0.5rem",
              "border": "1px solid #ccc", "borderRadius": "4px"}),

    html.Div([
        html.Label("Non-ROI subgroup hard constraints", style={"fontWeight": "bold"}),
        html.P("Rejects a montage if mean TI in mask_name exceeds max_mean_V_m, "
               "independently for EACH row — regardless of the overall non-ROI union "
               "above. mask_name must match an already-generated mask (Mask Generation page).",
               style={"fontSize": "12px", "color": "#666", "marginTop": "0.25rem"}),
        _styled_table(
            "cg-opt-constraint-groups-table",
            [
                {"name": "name", "id": "name"},
                {"name": "mask_name", "id": "mask_name", "presentation": "dropdown"},
                {"name": "max_mean_V_m", "id": "max_mean_V_m"},
            ],
            editable=True, row_deletable=True,
        ),
        html.Button("Add constraint group", id="cg-opt-constraint-groups-add-btn",
                    n_clicks=0, style={"marginTop": "0.5rem"}),
    ], style={"marginTop": "0.75rem", "marginBottom": "0.75rem", "padding": "0.5rem",
              "border": "1px solid #ccc", "borderRadius": "4px"}),

    html.Div([
        html.Label("ROI subgroup hard constraints", style={"fontWeight": "bold"}),
        html.P("Rejects a montage if mean TI in mask_name falls BELOW min_mean_V_m, "
               "independently for EACH row — regardless of the overall ROI mean above. "
               "Useful when the ROI is a union of several distinct subregions (e.g. "
               "hippocampus + entorhinal cortex) and a montage could otherwise clear the "
               "combined-ROI floor while barely touching one of them. mask_name must match "
               "an already-generated mask (Mask Generation page).",
               style={"fontSize": "12px", "color": "#666", "marginTop": "0.25rem"}),
        _styled_table(
            "cg-opt-roi-constraint-groups-table",
            [
                {"name": "name", "id": "name"},
                {"name": "mask_name", "id": "mask_name", "presentation": "dropdown"},
                {"name": "min_mean_V_m", "id": "min_mean_V_m"},
            ],
            editable=True, row_deletable=True,
        ),
        html.Button("Add constraint group", id="cg-opt-roi-constraint-groups-add-btn",
                    n_clicks=0, style={"marginTop": "0.5rem"}),
    ], style={"marginTop": "0.75rem", "marginBottom": "0.75rem", "padding": "0.5rem",
              "border": "1px solid #ccc", "borderRadius": "4px"}),

    html.H3("Electrode", style={"marginTop": "1.5rem"}),
    html.Div([
        html.Div([
            html.Label("diameter (mm)"),
            dcc.Input(id="cg-elec-diameter", type="number", step=0.5,
                      value=cd.ELECTRODE_DEFAULTS["dimensions"][0], style={"width": "100%"}),
        ], style={"minWidth": "140px", "marginRight": "1rem"}),
        html.Div([
            html.Label("gel thickness (mm)"),
            dcc.Input(id="cg-elec-gel-thickness", type="number", step=0.1,
                      value=cd.ELECTRODE_DEFAULTS["gel_thickness"], style={"width": "100%"}),
        ], style={"minWidth": "140px", "marginRight": "1rem"}),
        html.Div([
            html.Label("max total current (mA)"),
            dcc.Input(id="cg-elec-max-current", type="number", step=0.5,
                      value=cd.ELECTRODE_DEFAULTS["max_total_current"], style={"width": "100%"}),
        ], style={"minWidth": "160px", "marginRight": "1rem"}),
    ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "1.5rem"}),

    html.H3("Pipeline sections to run"),
    dcc.Checklist(
        id="cg-flags-checklist",
        options=[
            {"label": " ROI/non-ROI masks", "value": "run_roi_masks"},
            {"label": " Cap optimization", "value": "run_optimization"},
            {"label": " FEM simulation", "value": "run_simulation"},
            {"label": " TI field analysis", "value": "run_analysis"},
            {"label": " Visualization", "value": "run_visualization"},
        ],
        value=[k for k in ("run_roi_masks", "run_optimization", "run_simulation",
                           "run_analysis", "run_visualization") if cd.FLAGS_DEFAULTS[k]],
        inline=True,
        style={"marginBottom": "1.5rem"},
    ),

    html.H3("Subject Readiness"),
    _styled_table("cg-readiness-table", [
        {"name": "Subject", "id": "subject"},
        {"name": "Status", "id": "status_display"},
    ]),

    html.H3("Configs to generate", style={"marginTop": "1.5rem"}),
    html.Button("Select All Ready", id="cg-select-all-btn", n_clicks=0, style={"marginBottom": "0.5rem"}),
    _styled_table(
        "cg-generate-table",
        [
            {"name": "Subject", "id": "subject"},
            {"name": "Files", "id": "files_display"},
            {"name": "Status", "id": "overwrite_display"},
        ],
        row_selectable="multi",
        selected_rows=[],
        style_data_conditional=[
            {"if": {"filter_query": '{overwrite_display} contains "⚠"'}, "backgroundColor": "#ffe0e0"},
        ],
    ),

    html.Div(id="cg-overwrite-message", style={"margin": "0.5rem 0"}),
    dcc.Checklist(
        id="cg-overwrite-confirm",
        options=[{"label": " I confirm overwriting existing config file(s) above", "value": "confirm"}],
        value=[],
    ),

    html.Button("Generate", id="cg-generate-button", n_clicks=0, disabled=True,
                style={"marginTop": "1rem", "padding": "0.5rem 1.5rem"}),

    dcc.Loading(html.Div(id="cg-generate-results", style={"marginTop": "1rem"})),

    html.H3("Existing Configs on Disk", style={"marginTop": "2rem"}),
    _styled_table("cg-existing-configs-table", [
        {"name": "Subject", "id": "subject"},
        {"name": "File", "id": "filename"},
        {"name": "Modified", "id": "mtime_display"},
        {"name": "Size", "id": "size_display"},
    ]),
])

_register_stepper_callback("cg-opt-hier-num-iter", 1, 0)
_register_stepper_callback("cg-opt-hier-early-stop", 0.5, 0)
_register_stepper_callback("cg-opt-amp-top-n", 1, 1)


# ═════════════════════════════════════════════════════════════════════════════
# Subjects / cap dropdown
# ═════════════════════════════════════════════════════════════════════════════

@callback(Output("cg-subject-dropdown", "options"), Input("cg-subject-dropdown", "id"))
def _load_subjects(_):
    options = []
    for s in discovery.discover_subjects():
        status = "m2m ✓" if s["has_m2m"] else "m2m ✗ (needs charm)"
        options.append({
            "label": f"{s['subject_id']}   [{status}]",
            "value": s["subject_id"],
            "disabled": not s["has_m2m"],
        })
    return options


@callback(Output("cg-cap-dropdown", "options"), Input("cg-subject-dropdown", "value"))
def _load_caps(subject_ids):
    """Caps actually REGISTERED for the selected subject(s) — m2m_{id}/
    eeg_positions/*.csv — not generic MNI-space templates (list_available_caps()
    would show every cap that exists project-wide, regardless of whether any
    selected subject has it registered; a custom-registered cap that isn't
    one of those templates would never show up at all). Union across
    subjects if more than one is selected; the Subject Readiness table below
    flags whichever subjects are actually missing the chosen cap. Only the
    filename stem matters downstream (build_config()/registered_cap_path()
    both re-derive the per-subject path from it), so any one subject's path
    for a given cap name is an equally valid dropdown value."""
    if not subject_ids:
        return []
    seen = {}
    for sid in subject_ids:
        for c in cap_discovery.list_registered_caps(sid):
            seen.setdefault(c["name"], c["path"])
    return [{"label": name, "value": path} for name, path in sorted(seen.items())]


# ═════════════════════════════════════════════════════════════════════════════
# Subject readiness table
# ═════════════════════════════════════════════════════════════════════════════

def _constraint_group_mask_names(rows: list[dict] | None) -> list[str]:
    return [r["mask_name"].strip() for r in (rows or []) if (r.get("mask_name") or "").strip()]


@callback(
    Output("cg-readiness-table", "data"),
    Input("cg-subject-dropdown", "value"),
    Input("cg-roi-atlas-dropdown", "value"),
    Input("cg-roi-name-input", "value"),
    Input("cg-roi-existing-mask-dropdown", "value"),
    Input("cg-nonroi-atlas-dropdown", "value"),
    Input("cg-nonroi-name-input", "value"),
    Input("cg-nonroi-existing-mask-dropdown", "value"),
    Input("cg-cap-dropdown", "value"),
    Input("cg-opt-constraint-groups-table", "data"),
    Input("cg-opt-roi-constraint-groups-table", "data"),
)
def _update_readiness_table(subject_ids, roi_atlas, roi_name, roi_existing_mask,
                            non_roi_atlas, non_roi_name, non_roi_existing_mask,
                            cap_path, constraint_group_rows, roi_constraint_group_rows):
    if not subject_ids or not roi_name or not (roi_atlas or roi_existing_mask):
        return []
    constraint_group_masks = (_constraint_group_mask_names(constraint_group_rows)
                              + _constraint_group_mask_names(roi_constraint_group_rows))
    matrix = cd.readiness_matrix(subject_ids, roi_atlas, roi_name, non_roi_atlas, non_roi_name,
                                 cap_path, constraint_group_masks=constraint_group_masks,
                                 roi_uses_existing_mask=bool(roi_existing_mask),
                                 non_roi_uses_existing_mask=bool(non_roi_existing_mask))
    rows = []
    for sid in subject_ids:
        r = matrix[sid]
        rows.append({
            "subject": sid,
            "status_display": "✓ ready" if r["ready"] else "✗ " + "; ".join(r["missing"]),
        })
    return rows


# ═════════════════════════════════════════════════════════════════════════════
# Generate table + readiness gate + the action itself
# ═════════════════════════════════════════════════════════════════════════════

@callback(
    Output("cg-generate-table", "data"),
    Output("cg-generate-table", "selected_rows"),
    Input("cg-subject-dropdown", "value"),
    Input("cg-roi-atlas-dropdown", "value"),
    Input("cg-roi-name-input", "value"),
    Input("cg-roi-existing-mask-dropdown", "value"),
    Input("cg-nonroi-atlas-dropdown", "value"),
    Input("cg-nonroi-name-input", "value"),
    Input("cg-nonroi-existing-mask-dropdown", "value"),
    Input("cg-cap-dropdown", "value"),
    Input("cg-goals-checklist", "value"),
    Input("cg-opt-constraint-groups-table", "data"),
    Input("cg-opt-roi-constraint-groups-table", "data"),
    State("cg-generate-table", "selected_rows"),
)
def _update_generate_table(subject_ids, roi_atlas, roi_name, roi_existing_mask,
                           non_roi_atlas, non_roi_name, non_roi_existing_mask,
                           cap_path, goals, constraint_group_rows, roi_constraint_group_rows, prev_selected):
    if not subject_ids or not roi_name or not (roi_atlas or roi_existing_mask) or not goals:
        return [], []

    constraint_group_masks = (_constraint_group_mask_names(constraint_group_rows)
                              + _constraint_group_mask_names(roi_constraint_group_rows))
    matrix = cd.readiness_matrix(subject_ids, roi_atlas, roi_name, non_roi_atlas, non_roi_name,
                                 cap_path, constraint_group_masks=constraint_group_masks,
                                 roi_uses_existing_mask=bool(roi_existing_mask),
                                 non_roi_uses_existing_mask=bool(non_roi_existing_mask))
    ready_subjects = [sid for sid in subject_ids if matrix[sid]["ready"]]

    rows = []
    for sid in ready_subjects:
        paths = [cd.CONFIGS_DIR / cd.config_filename(sid, roi_name, g) for g in goals]
        existing = [p for p in paths if p.exists()]
        files_display = f"{len(paths)} file(s): " + ", ".join(p.name for p in paths)
        overwrite_display = f"⚠ {len(existing)} will overwrite" if existing else "new file(s)"
        rows.append({"subject": sid, "files_display": files_display, "overwrite_display": overwrite_display})

    if ctx.triggered_id in ("cg-subject-dropdown", "cg-roi-atlas-dropdown", "cg-roi-name-input",
                           "cg-roi-existing-mask-dropdown", "cg-nonroi-atlas-dropdown",
                           "cg-nonroi-name-input", "cg-nonroi-existing-mask-dropdown",
                           "cg-cap-dropdown", "cg-goals-checklist") or not prev_selected:
        selected_rows = list(range(len(rows)))
    else:
        selected_rows = [i for i in prev_selected if i < len(rows)]

    return rows, selected_rows


@callback(
    Output("cg-generate-table", "selected_rows", allow_duplicate=True),
    Input("cg-select-all-btn", "n_clicks"),
    State("cg-generate-table", "data"),
    prevent_initial_call=True,
)
def _select_all_ready(_n_clicks, data):
    return list(range(len(data or [])))


@callback(
    Output("cg-overwrite-message", "children"),
    Output("cg-generate-button", "disabled"),
    Input("cg-generate-table", "data"),
    Input("cg-generate-table", "selected_rows"),
    Input("cg-roi-name-input", "value"),
    Input("cg-roi-atlas-dropdown", "value"),
    Input("cg-roi-selected-label-ids-store", "data"),
    Input("cg-roi-existing-mask-dropdown", "value"),
    Input("cg-nonroi-name-input", "value"),
    Input("cg-nonroi-atlas-dropdown", "value"),
    Input("cg-nonroi-selected-label-ids-store", "data"),
    Input("cg-nonroi-existing-mask-dropdown", "value"),
    Input("cg-overwrite-confirm", "value"),
)
def _update_generate_readiness(rows, selected_rows, roi_name, roi_atlas, roi_label_ids, roi_existing_mask,
                               non_roi_name, non_roi_atlas, non_roi_label_ids, non_roi_existing_mask,
                               confirm_value):
    rows = rows or []
    selected_rows = selected_rows or []
    selected = [rows[i] for i in selected_rows if i < len(rows)]

    problems = []
    if not selected:
        problems.append("select at least one ready subject")
    if not roi_name:
        problems.append("enter a ROI name")
    # An explicitly-picked existing mask (any atlas, or no atlas at all)
    # overrides needing a region selection — same as Allen already does,
    # just not tied to one specific atlas (see build_roi_dict()).
    if roi_atlas and not cd.uses_allen(roi_atlas) and not roi_label_ids and not roi_existing_mask:
        problems.append("select at least one ROI region")
    if not non_roi_name:
        problems.append("enter a non-ROI name")
    if non_roi_atlas and not cd.uses_allen(non_roi_atlas) and not non_roi_label_ids and not non_roi_existing_mask:
        problems.append("select at least one non-ROI region")

    overwrite_rows = [r for r in selected if r["overwrite_display"].startswith("⚠")]
    confirmed = "confirm" in (confirm_value or [])

    children = []
    if overwrite_rows:
        children.append(html.P(
            f"⚠ {len(overwrite_rows)} subject(s) have existing config file(s) that will be overwritten: "
            + ", ".join(r["subject"] for r in overwrite_rows),
            style={"color": "#a00"}))
        if not confirmed:
            problems.append("confirm overwrite to proceed")

    if problems:
        children.append(html.P("Cannot generate yet — " + "; ".join(problems),
                                style={"color": "#a00", "fontSize": "13px"}))

    return html.Div(children), bool(problems)


@callback(
    Output("cg-opt-constraint-groups-table", "data"),
    Input("cg-opt-constraint-groups-add-btn", "n_clicks"),
    State("cg-opt-constraint-groups-table", "data"),
    prevent_initial_call=True,
)
def _add_constraint_group_row(_n_clicks, rows):
    rows = rows or []
    rows.append({"name": "", "mask_name": "", "max_mean_V_m": None})
    return rows


@callback(
    Output("cg-opt-constraint-groups-table", "dropdown"),
    Input("cg-subject-dropdown", "value"),
)
def _update_constraint_group_mask_options(subject_ids):
    options = [{"label": n, "value": n} for n in discovery.list_mask_names(subject_ids or [])]
    return {"mask_name": {"options": options, "clearable": True}}


@callback(
    Output("cg-opt-amp-weight-sum-note", "children"),
    Input("cg-opt-amp-weight-roc", "value"),
    Input("cg-opt-amp-weight-roi-mean", "value"),
    Input("cg-opt-amp-weight-non-roi-mean", "value"),
    Input("cg-opt-amp-weight-focality-ratio", "value"),
)
def _update_amp_weight_sum_note(roc, roi_mean, non_roi_mean, focality_ratio):
    # Sanity nudge only — the pipeline doesn't require these to sum to 1.0,
    # so this never blocks generation, just flags a likely typo.
    total = sum(v for v in (roc, roi_mean, non_roi_mean, focality_ratio) if v is not None)
    color = "#666" if abs(total - 1.0) < 0.01 else "#a60"
    return html.Span(f"sum: {total:.2f}" + ("" if color == "#666" else "  (doesn't sum to 1.0 — fine, just double-check)"),
                     style={"color": color})


@callback(
    Output("cg-opt-roi-constraint-groups-table", "data"),
    Input("cg-opt-roi-constraint-groups-add-btn", "n_clicks"),
    State("cg-opt-roi-constraint-groups-table", "data"),
    prevent_initial_call=True,
)
def _add_roi_constraint_group_row(_n_clicks, rows):
    rows = rows or []
    rows.append({"name": "", "mask_name": "", "min_mean_V_m": None})
    return rows


@callback(
    Output("cg-opt-roi-constraint-groups-table", "dropdown"),
    Input("cg-subject-dropdown", "value"),
)
def _update_roi_constraint_group_mask_options(subject_ids):
    options = [{"label": n, "value": n} for n in discovery.list_mask_names(subject_ids or [])]
    return {"mask_name": {"options": options, "clearable": True}}


@callback(
    Output("cg-generate-results", "children"),
    Input("cg-generate-button", "n_clicks"),
    State("cg-generate-table", "data"),
    State("cg-generate-table", "selected_rows"),
    State("cg-roi-name-input", "value"),
    State("cg-roi-atlas-dropdown", "value"),
    State("cg-roi-selected-label-ids-store", "data"),
    State("cg-nonroi-name-input", "value"),
    State("cg-nonroi-atlas-dropdown", "value"),
    State("cg-nonroi-selected-label-ids-store", "data"),
    State("cg-goals-checklist", "value"),
    State("cg-cap-dropdown", "value"),
    State("cg-opt-postproc", "value"),
    State("cg-opt-cpus", "value"),
    State("cg-opt-focality-nonroi", "value"),
    State("cg-opt-focality-roi", "value"),
    State("cg-opt-checkboxes", "value"),
    State("cg-opt-hier-enable", "value"),
    State("cg-opt-hier-num-iter", "value"),
    State("cg-opt-hier-neighbours", "value"),
    State("cg-opt-hier-early-stop", "value"),
    State("cg-opt-constraint-groups-table", "data"),
    State("cg-opt-roi-constraint-groups-table", "data"),
    State("cg-opt-amp-enable", "value"),
    State("cg-opt-amp-top-n", "value"),
    State("cg-opt-amp-min-ma", "value"),
    State("cg-opt-amp-max-ma", "value"),
    State("cg-opt-amp-step-ma", "value"),
    State("cg-opt-amp-max-per-pair-ma", "value"),
    State("cg-opt-amp-weight-roc", "value"),
    State("cg-opt-amp-weight-roi-mean", "value"),
    State("cg-opt-amp-weight-non-roi-mean", "value"),
    State("cg-opt-amp-weight-focality-ratio", "value"),
    State("cg-elec-diameter", "value"),
    State("cg-elec-gel-thickness", "value"),
    State("cg-elec-max-current", "value"),
    State("cg-flags-checklist", "value"),
    prevent_initial_call=True,
)
def _on_generate_click(_n_clicks, rows, selected_rows, roi_name, roi_atlas, roi_label_ids,
                       non_roi_name, non_roi_atlas, non_roi_label_ids, goals, cap_path,
                       postproc, cpus, focality_nonroi, focality_roi, opt_checkboxes,
                       hier_enable, hier_num_iter, hier_neighbours_str, hier_early_stop_pct,
                       constraint_group_rows, roi_constraint_group_rows,
                       amp_enable, amp_top_n, amp_min_ma, amp_max_ma, amp_step_ma, amp_max_per_pair_ma,
                       amp_weight_roc, amp_weight_roi_mean, amp_weight_non_roi_mean, amp_weight_focality_ratio,
                       elec_diameter, elec_gel, elec_max_current, flags_checklist):
    rows = rows or []
    selected_rows = selected_rows or []
    subject_ids = [rows[i]["subject"] for i in selected_rows if i < len(rows)]

    if not subject_ids or not roi_name or not non_roi_name or not goals:
        return html.Div("Nothing to generate — check subject/ROI/non-ROI/goal selection.",
                         style={"color": "#a00"})

    use_hierarchical = "use_hierarchical_search" in (hier_enable or [])
    num_fine_iter = int(hier_num_iter) if hier_num_iter else 0
    neighbours_per_iter = []
    if hier_neighbours_str and hier_neighbours_str.strip():
        try:
            neighbours_per_iter = [int(x.strip()) for x in hier_neighbours_str.split(",") if x.strip()]
        except ValueError:
            return html.Div(f"neighbours per iteration must be comma-separated integers "
                             f"(e.g. 8,6,4) — got: {hier_neighbours_str!r}",
                             style={"color": "#a00"})
    if use_hierarchical and num_fine_iter and len(neighbours_per_iter) != num_fine_iter:
        return html.Div(f"neighbours per iteration must have exactly {num_fine_iter} value(s) "
                         f"(one per fine iteration) — got {len(neighbours_per_iter)}.",
                         style={"color": "#a00"})

    constraint_groups = []
    for row in (constraint_group_rows or []):
        grp_name = (row.get("name") or "").strip()
        mask_name = (row.get("mask_name") or "").strip()
        max_v = row.get("max_mean_V_m")
        if not grp_name and not mask_name and max_v in (None, ""):
            continue   # blank row (added but never filled in) — skip silently
        if not grp_name or not mask_name or max_v in (None, ""):
            return html.Div("each non-ROI subgroup constraint row needs name, mask_name, "
                             "and max_mean_V_m — remove any incomplete rows.",
                             style={"color": "#a00"})
        try:
            max_v = float(max_v)
        except (TypeError, ValueError):
            return html.Div(f"constraint group '{grp_name}': max_mean_V_m must be a number, "
                             f"got {max_v!r}", style={"color": "#a00"})
        constraint_groups.append({"name": grp_name, "mask_name": mask_name, "max_mean_V_m": max_v})

    roi_constraint_groups = []
    for row in (roi_constraint_group_rows or []):
        grp_name = (row.get("name") or "").strip()
        mask_name = (row.get("mask_name") or "").strip()
        min_v = row.get("min_mean_V_m")
        if not grp_name and not mask_name and min_v in (None, ""):
            continue   # blank row (added but never filled in) — skip silently
        if not grp_name or not mask_name or min_v in (None, ""):
            return html.Div("each ROI subgroup constraint row needs name, mask_name, "
                             "and min_mean_V_m — remove any incomplete rows.",
                             style={"color": "#a00"})
        try:
            min_v = float(min_v)
        except (TypeError, ValueError):
            return html.Div(f"ROI constraint group '{grp_name}': min_mean_V_m must be a number, "
                             f"got {min_v!r}", style={"color": "#a00"})
        roi_constraint_groups.append({"name": grp_name, "mask_name": mask_name, "min_mean_V_m": min_v})

    use_amplitude_sweep = "use_amplitude_sweep" in (amp_enable or [])
    if use_amplitude_sweep:
        if amp_min_ma is None or amp_max_ma is None or amp_step_ma is None:
            return html.Div("amplitude sweep: min/max/step mA are all required when "
                             "use_amplitude_sweep is enabled.", style={"color": "#a00"})
        if amp_min_ma > amp_max_ma:
            return html.Div(f"amplitude sweep: min mA ({amp_min_ma}) can't exceed max mA ({amp_max_ma}).",
                             style={"color": "#a00"})
        if amp_step_ma <= 0:
            return html.Div("amplitude sweep: step mA must be positive.", style={"color": "#a00"})

    optimizer_overrides = {
        "postproc": postproc,
        "cpus": cpus,
        "focality_threshold": [focality_nonroi, focality_roi],
        "hard_roi_constraint": "hard_roi_constraint" in (opt_checkboxes or []),
        "no_adjacent_electrodes": "no_adjacent_electrodes" in (opt_checkboxes or []),
        "use_hierarchical_search": use_hierarchical,
        "num_fine_iterations": num_fine_iter,
        "neighbours_per_iteration": neighbours_per_iter,
        "early_stop_threshold": (hier_early_stop_pct or 0) / 100.0,
        "non_roi_hard_constraint_groups": constraint_groups,
        "roi_hard_constraint_groups": roi_constraint_groups,
        "use_amplitude_sweep": use_amplitude_sweep,
        "amplitude_sweep_top_n": int(amp_top_n) if amp_top_n else cd.OPTIMIZER_DEFAULTS["amplitude_sweep_top_n"],
        "amplitude_sweep_min_mA": amp_min_ma if amp_min_ma is not None else cd.OPTIMIZER_DEFAULTS["amplitude_sweep_min_mA"],
        "amplitude_sweep_max_mA": amp_max_ma if amp_max_ma is not None else cd.OPTIMIZER_DEFAULTS["amplitude_sweep_max_mA"],
        "amplitude_sweep_step_mA": amp_step_ma if amp_step_ma is not None else cd.OPTIMIZER_DEFAULTS["amplitude_sweep_step_mA"],
        "amplitude_sweep_max_per_pair_mA": (amp_max_per_pair_ma if amp_max_per_pair_ma is not None
                                            else cd.OPTIMIZER_DEFAULTS["amplitude_sweep_max_per_pair_mA"]),
        "amplitude_sweep_weights": {
            "roc": amp_weight_roc if amp_weight_roc is not None else cd.OPTIMIZER_DEFAULTS["amplitude_sweep_weights"]["roc"],
            "roi_mean": (amp_weight_roi_mean if amp_weight_roi_mean is not None
                        else cd.OPTIMIZER_DEFAULTS["amplitude_sweep_weights"]["roi_mean"]),
            "non_roi_mean": (amp_weight_non_roi_mean if amp_weight_non_roi_mean is not None
                             else cd.OPTIMIZER_DEFAULTS["amplitude_sweep_weights"]["non_roi_mean"]),
            "focality_ratio": (amp_weight_focality_ratio if amp_weight_focality_ratio is not None
                               else cd.OPTIMIZER_DEFAULTS["amplitude_sweep_weights"]["focality_ratio"]),
        },
    }
    electrode_overrides = {
        "dimensions": [elec_diameter, elec_diameter],
        "gel_thickness": elec_gel,
        "max_total_current": elec_max_current,
    }
    all_flag_keys = ("run_recon_all", "run_charm", "run_roi_masks", "run_optimization",
                     "run_simulation", "run_analysis", "run_visualization")
    flags = {k: (k in (flags_checklist or [])) for k in all_flag_keys
              if k in ("run_roi_masks", "run_optimization", "run_simulation",
                       "run_analysis", "run_visualization")}

    all_results = []
    for goal in goals:
        results = cd.generate_configs(
            subject_ids, roi_name, roi_atlas, roi_label_ids,
            non_roi_name, non_roi_atlas, non_roi_label_ids,
            [goal], optimizer_overrides, electrode_overrides, flags, cap_path, force=True,
        )
        all_results.extend(results)

    cd.write_sbatch_env()

    return html.Div([
        _styled_table("cg-generate-results-table", [
            {"name": "Subject", "id": "subject"},
            {"name": "Goal", "id": "goal"},
            {"name": "Status", "id": "status"},
            {"name": "Path", "id": "path"},
        ], data=[
            {"subject": r["subject_id"], "goal": r["goal"], "status": r["status"], "path": r["path"]}
            for r in all_results
        ]),
        html.P(f"subject_configs.sh rebuilt from every config now in {cd.CONFIGS_DIR}",
               style={"fontSize": "13px", "color": "#666", "marginTop": "0.5rem"}),
    ])


# ═════════════════════════════════════════════════════════════════════════════
# Existing configs browser
# ═════════════════════════════════════════════════════════════════════════════

@callback(Output("cg-existing-configs-table", "data"), Input("cg-subject-dropdown", "value"))
def _update_existing_configs(subject_ids):
    if not subject_ids:
        return []
    rows = []
    for sid in subject_ids:
        for c in cd.existing_configs(sid):
            rows.append({
                "subject": sid,
                "filename": c["filename"],
                "mtime_display": datetime.fromtimestamp(c["mtime"]).strftime("%Y-%m-%d %H:%M"),
                "size_display": f"{c['size_bytes'] / 1024:.1f} KB",
            })
    return rows

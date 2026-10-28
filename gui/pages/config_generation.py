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
import mask_recipe_discovery as mrd


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
import preset_discovery as pd

dash.register_page(__name__, path="/config-generation", name="Config Generation",
                   category="Optimization", order=1)

ATLAS_NAMES = list(discovery.ATLAS_REGISTRY.keys())


def _details(title, children, level=0, open_=False):
    """Native <details>/<summary> collapsible, closed by default — declutters
    the Optimizer section's advanced blocks (hierarchical search, amplitude
    sweep, advanced location scoring, subgroup constraints) without any
    callback overhead; the browser handles expand/collapse for free. level=0
    is the outer "Advanced optimizer settings" wrapper (light gray card),
    level=1 is one of the individual sub-features nested inside it."""
    return html.Details([
        html.Summary(title, style={"fontWeight": "600", "cursor": "pointer",
                                   "padding": "0.3rem 0", "userSelect": "none"}),
        html.Div(children, style={"padding": "0.25rem 0 0.25rem 0.75rem"}),
    ], open=open_, style={
        "marginTop": "0.6rem" if level == 0 else "0.5rem",
        "marginBottom": "0.6rem" if level == 0 else "0.5rem",
        "padding": "0.5rem 0.75rem",
        "border": "1px solid #ccc" if level == 0 else "1px solid #ddd",
        "borderRadius": "6px",
        "background": "#f7f7f7" if level == 0 else "#fff",
    })


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
# Recipe-book-backed masks — every mask this page references (ROI, non-ROI,
# subgroup/background constraints) is picked BY NAME from one selected
# recipe book (see mask_recipe_discovery.py), not built inline here. Mask
# Generation is the only place that actually builds anything; this page
# only resolves names to recipes (for the final config JSON / a frozen
# Preference) and can trigger building whichever referenced names are
# missing for the selected subject(s).
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_recipe(book_name, mask_name):
    """(atlas, label_ids, kind) for mask_name inside book_name. atlas is
    None and label_ids {} for a "combine" or "external_mask" recipe —
    neither has a single atlas, so the final mask file must already exist
    (same convention build_roi_dict() already uses for Allen-sourced
    masks). (None, {}, None) if the book or the name can't be resolved."""
    if not book_name or not mask_name:
        return None, {}, None
    book = mrd.load_recipe_book(book_name)
    if not book:
        return None, {}, None
    recipe = (book.get("masks") or {}).get(mask_name)
    if not recipe:
        return None, {}, None
    kind = recipe.get("kind")
    if kind in ("combine", "external_mask"):
        return None, {}, kind
    return recipe.get("atlas"), recipe.get("label_ids") or {}, "atlas_region"


def _referenced_mask_names(roi_name, non_roi_name, constraint_group_rows,
                           roi_constraint_group_rows, bg_mask_name):
    names = set()
    if roi_name and roi_name.strip():
        names.add(roi_name.strip())
    if non_roi_name and non_roi_name.strip():
        names.add(non_roi_name.strip())
    names |= set(_constraint_group_mask_names(constraint_group_rows))
    names |= set(_constraint_group_mask_names(roi_constraint_group_rows))
    if bg_mask_name and bg_mask_name.strip():
        names.add(bg_mask_name.strip())
    return sorted(names)


def _no_recipe_coverage_warning(subject_ids, book_name, roi_name, non_roi_name,
                                constraint_group_rows, roi_constraint_group_rows, bg_mask_name):
    """For every referenced mask name with NO recipe (not in the selected
    book — "Build missing masks" has nothing to build it from), flags
    which of the currently selected subjects don't already have it on
    disk. Those subjects can't be auto-fixed and would just be silently
    excluded from Generate/Subject Readiness further down — surfaced here
    instead, right where the pick was made, so it's caught before
    scrolling down rather than discovered as a quiet exclusion."""
    if not subject_ids or len(subject_ids) < 2:
        return None   # nothing to be inconsistent across with 0-1 subjects
    referenced = _referenced_mask_names(roi_name, non_roi_name, constraint_group_rows,
                                        roi_constraint_group_rows, bg_mask_name)
    if not referenced:
        return None
    book = mrd.load_recipe_book(book_name) if book_name else None
    book_names = set((book or {}).get("masks", {}).keys())
    lines = []
    for name in referenced:
        if name in book_names:
            continue   # has a recipe -- Build missing masks can cover any gap
        missing = [sid for sid in subject_ids
                  if not os.path.isfile(discovery.expected_mask_path(sid, name))]
        if missing:
            lines.append(f"'{name}' has no recipe and is missing for: {', '.join(missing)}")
    if not lines:
        return None
    return html.Div([
        html.P("⚠ no-recipe mask(s) not on disk for every selected subject — these "
               "subject(s) will be silently excluded from Generate, since there's no "
               "recipe to build them from (pick a recipe-book name instead, or add a "
               "recipe for it in Mask Generation, if every subject needs to be covered):",
               style={"margin": 0}),
        html.Ul([html.Li(l) for l in lines], style={"margin": "0.25rem 0 0 0"}),
    ], style={"color": "#a60", "backgroundColor": "#fff8e1", "border": "1px solid #e0c060",
             "borderRadius": "4px", "padding": "0.5rem 0.75rem", "fontSize": "13px",
             "marginBottom": "0.75rem"})


@callback(
    Output("cg-mask-coverage-warning", "children"),
    Input("cg-subject-dropdown", "value"),
    Input("cg-recipe-book-dropdown", "value"),
    Input("cg-roi-name-dropdown", "value"),
    Input("cg-nonroi-name-dropdown", "value"),
    Input("cg-opt-constraint-groups-table", "data"),
    Input("cg-opt-roi-constraint-groups-table", "data"),
    Input("cg-opt-bg-mask-dropdown", "value"),
)
def _update_mask_coverage_warning(subject_ids, book_name, roi_name, non_roi_name,
                                  constraint_group_rows, roi_constraint_group_rows, bg_mask_name):
    return _no_recipe_coverage_warning(subject_ids, book_name, roi_name, non_roi_name,
                                       constraint_group_rows, roi_constraint_group_rows, bg_mask_name)


def _freeze_recipes(book_masks, names, frozen=None):
    """Recursively copies every recipe `names` depend on (transitively,
    through a "combine" recipe's sources) out of the live book into a
    standalone dict — so a saved Preference behaves identically later even
    if the book it was built from is edited or deleted afterward. Returns
    (frozen_dict, missing_names) — missing_names is every referenced name
    with no recipe anywhere in the book (not even as a combine source),
    which makes the whole thing unbuildable on a brand-new subject."""
    frozen = frozen if frozen is not None else {}
    missing = []
    for name in names:
        if name in frozen:
            continue
        recipe = book_masks.get(name)
        if recipe is None:
            missing.append(name)
            continue
        frozen[name] = recipe
        if recipe.get("kind") == "combine":
            _, sub_missing = _freeze_recipes(book_masks, recipe.get("sources") or [], frozen)
            missing.extend(sub_missing)
    return frozen, missing


@callback(Output("cg-recipe-book-dropdown", "options"), Input("cg-recipe-book-dropdown", "id"))
def _load_recipe_books(_):
    return [{"label": b["name"], "value": b["name"]} for b in mrd.list_recipe_books()]


@callback(
    Output("cg-roi-name-dropdown", "options"),
    Output("cg-nonroi-name-dropdown", "options"),
    Input("cg-recipe-book-dropdown", "value"),
    Input("cg-subject-dropdown", "value"),
)
def _load_book_mask_options(book_name, subject_ids):
    options = _mask_name_options(book_name, subject_ids)
    return options, options


def _mask_name_options(book_name, subject_ids=None):
    """Recipe-book names (buildable anywhere, usable in a Preference) UNION
    whatever masks already exist on disk for the selected subject(s) (not
    recipe-backed — fine to pick for a direct Generate Configs run on that
    subject, but Save Preference will refuse one, since it can't promise
    to build it on a subject who doesn't have it yet). A name in both is
    shown plain, same as a pure recipe-book name."""
    book = mrd.load_recipe_book(book_name) if book_name else None
    book_names = set((book or {}).get("masks", {}).keys())
    disk_names = set(discovery.list_mask_names(subject_ids or []))
    options = []
    for n in sorted(book_names | disk_names):
        label = n if n in book_names else f"{n}  (existing mask, no recipe)"
        options.append({"label": label, "value": n})
    return options


@callback(
    Output("cg-build-masks-result", "children"),
    Input("cg-build-masks-btn", "n_clicks"),
    State("cg-subject-dropdown", "value"),
    State("cg-recipe-book-dropdown", "value"),
    State("cg-roi-name-dropdown", "value"),
    State("cg-nonroi-name-dropdown", "value"),
    State("cg-opt-constraint-groups-table", "data"),
    State("cg-opt-roi-constraint-groups-table", "data"),
    State("cg-opt-bg-mask-dropdown", "value"),
    prevent_initial_call=True,
)
def _on_build_masks_click(_n_clicks, subject_ids, book_name, roi_name, non_roi_name,
                          constraint_group_rows, roi_constraint_group_rows, bg_mask_name):
    if not subject_ids:
        return html.Div("Select at least one subject first.", style={"color": "#a00"})
    if not book_name:
        return html.Div("Select a recipe book first.", style={"color": "#a00"})
    book = mrd.load_recipe_book(book_name)
    if not book:
        return html.Div(f"Could not load recipe book '{book_name}'.", style={"color": "#a00"})

    names = _referenced_mask_names(roi_name, non_roi_name, constraint_group_rows,
                                   roi_constraint_group_rows, bg_mask_name)
    if not names:
        return html.Div("Nothing to build — pick ROI/non-ROI (or a constraint mask) first.",
                        style={"color": "#a00"})

    results = mrd.build_missing_masks(subject_ids, book.get("masks") or {}, names)
    if not results:
        return html.Div("✓ every referenced mask already exists for every selected subject — "
                        "nothing to build.", style={"color": "#060"})
    return _styled_table("cg-build-masks-results-table", [
        {"name": "Subject", "id": "subject"},
        {"name": "Mask", "id": "mask_name"},
        {"name": "Status", "id": "status"},
        {"name": "Detail", "id": "detail"},
    ], data=[
        {"subject": r["subject_id"], "mask_name": r["mask_name"],
         "status": "✓ built" if r["success"] else "✗ failed",
         "detail": (f"{r['voxel_count']} voxels" if r["success"] else (r["error"] or ""))}
        for r in results
    ], style_data_conditional=[
        {"if": {"filter_query": '{status} contains "failed"'}, "backgroundColor": "#ffe0e0"},
    ])


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

    html.H3("Recipe book"),
    html.P("Every mask this config references — ROI, non-ROI, and any subgroup/background "
           "constraint masks below — is picked by name, either from one recipe book (built on "
           "the Mask Generation page's \"also save this as a recipe\" option — full access to "
           "every atlas, including Allen, and hemisphere, not a cut-down picker here) or from "
           "a mask that already exists for the selected subject(s), no recipe required. A "
           "name picked below becomes selectable in the subgroup/background pickers further "
           "down too, whether or not it's built yet — Subject Readiness shows what's missing, "
           "and \"Build missing masks\" builds exactly that, for exactly the subjects that need "
           "it; nothing already on disk is ever rebuilt. Only a recipe-backed name can be saved "
           "into a Preference, though — an existing-mask-only pick works for Generate Configs "
           "below, but Save Preference will refuse it (it has to be buildable on a subject who "
           "doesn't have it yet, which a bare existing file can't promise).",
           style={"fontSize": "13px", "color": "#666", "marginBottom": "0.5rem"}),
    html.Div([
        dcc.Dropdown(id="cg-recipe-book-dropdown", placeholder="Select a recipe book..."),
    ], style={"maxWidth": "420px", "marginBottom": "1rem"}),
    html.Div([
        html.Div([
            html.Label("ROI"),
            dcc.Dropdown(id="cg-roi-name-dropdown", placeholder="Select ROI mask..."),
        ], style={"flex": "1 1 280px", "marginRight": "1rem"}),
        html.Div([
            html.Label("non-ROI"),
            dcc.Dropdown(id="cg-nonroi-name-dropdown", placeholder="Select non-ROI mask..."),
        ], style={"flex": "1 1 280px"}),
    ], style={"display": "flex", "flexWrap": "wrap", "marginBottom": "1rem"}),
    html.Div(id="cg-mask-coverage-warning"),
    html.Button("Build missing masks for selected subject(s)", id="cg-build-masks-btn", n_clicks=0,
               style={"marginBottom": "0.5rem"}),
    dcc.Loading(html.Div(id="cg-build-masks-result", style={"marginBottom": "1.5rem"})),

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
            {"label": " use_quadrant_constraint — reject montages where 2 non-midline electrodes "
                      "share a scalp quadrant (independent of adjacency/tiers/background scoring)",
                      "value": "use_quadrant_constraint"},
        ],
        value=[k for k in ("hard_roi_constraint", "no_adjacent_electrodes", "use_quadrant_constraint")
              if cd.OPTIMIZER_DEFAULTS[k]],
        inline=True,
    ),

    _details("Advanced optimizer settings", [
        _details("Hierarchical search (coarse-to-fine electrode search)", [
            dcc.Checklist(
                id="cg-opt-hier-enable",
                options=[{"label": " use_hierarchical_search", "value": "use_hierarchical_search"}],
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
        ], level=1),

        _details("Amplitude sweep (re-rank top montages over a per-channel current grid)", [
            dcc.Checklist(
                id="cg-opt-amp-enable",
                options=[{"label": " use_amplitude_sweep", "value": "use_amplitude_sweep"}],
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
        ], level=1),

        html.Div([
            html.Label("main weight (on the raw ROC/mean-TI score itself — only matters once "
                      "background minimization and/or electrode scoring tiers below are on; "
                      "composite scoring then turns on by itself, no separate switch needed)",
                      style={"marginRight": "0.5rem"}),
            dcc.Input(id="cg-opt-loc-weight-main", type="number", step=0.05,
                     value=cd.OPTIMIZER_DEFAULTS["location_score_weights"]["main"],
                     style={"width": "90px"}),
        ], style={"marginTop": "0.5rem", "marginBottom": "0.5rem", "display": "flex", "alignItems": "center"}),

        _details("Background field minimization", [
            html.P("score -= background_weight * background_mean + background_hotspot_weight * "
                   "background_hotspot_mean. Fully independent of electrode scoring tiers below "
                   "— this checkbox is all that's needed to turn it on.",
                   style={"fontSize": "12px", "color": "#666", "marginBottom": "0.5rem"}),
            dcc.Checklist(
                id="cg-opt-bg-enable",
                options=[{"label": " minimize_background_field (penalize strong field anywhere "
                          "outside the ROI, including under the electrodes)",
                          "value": "minimize_background_field"}],
                value=[k for k in ("minimize_background_field",) if cd.OPTIMIZER_DEFAULTS[k]],
            ),
            html.Div([
                html.Div([
                    html.Label("background region (blank = whole brain GM+WM, minus ROI)"),
                    dcc.Dropdown(id="cg-opt-bg-mask-dropdown", placeholder="Whole brain",
                                 clearable=True),
                ], style={"minWidth": "260px", "marginRight": "1rem", "flex": "1 1 260px"}),
                html.Div([
                    html.Label("max_background_elements (subsample cap)"),
                    dcc.Input(id="cg-opt-bg-max-elements", type="number", min=1000, step=1000,
                              value=cd.OPTIMIZER_DEFAULTS["max_background_elements"],
                              style={"width": "100%"}),
                ], style={"minWidth": "180px", "marginRight": "1rem"}),
                html.Div([
                    html.Label("background_hotspot_percentile"),
                    dcc.Input(id="cg-opt-bg-hotspot-pct", type="number", min=0, max=100, step=1,
                              value=cd.OPTIMIZER_DEFAULTS["background_hotspot_percentile"],
                              style={"width": "100%"}),
                ], style={"minWidth": "180px", "marginRight": "1rem"}),
            ], style={"display": "flex", "flexWrap": "wrap", "marginTop": "0.5rem"}),
            html.Div([
                html.Div([
                    html.Label("background weight"),
                    dcc.Input(id="cg-opt-loc-weight-background", type="number", step=0.05,
                              value=cd.OPTIMIZER_DEFAULTS["location_score_weights"]["background"],
                              style={"width": "100%"}),
                ], style={"minWidth": "140px", "marginRight": "1rem"}),
                html.Div([
                    html.Label("background_hotspot weight"),
                    dcc.Input(id="cg-opt-loc-weight-background-hotspot", type="number", step=0.05,
                              value=cd.OPTIMIZER_DEFAULTS["location_score_weights"]["background_hotspot"],
                              style={"width": "100%"}),
                ], style={"minWidth": "140px", "marginRight": "1rem"}),
            ], style={"display": "flex", "flexWrap": "wrap", "marginTop": "0.5rem"}),
            html.P("Still un-calibrated placeholders (background) / through a real pilot "
                   "comparison, still provisional (background_hotspot) — see code/pipeline/"
                   "config.py's OptimizerConfig.location_score_weights.",
                   style={"fontSize": "12px", "color": "#666", "marginTop": "0.5rem"}),
        ], level=1),

        _details("Electrode scoring tiers", [
            html.P("score -= tier_penalty_weight * tier_cost. Fully independent of background "
                   "field minimization above — this checkbox is all that's needed to turn it "
                   "on. (Fiducial exclusion and the tiered search pool below always apply once "
                   "this is on, regardless of the score itself.)",
                   style={"fontSize": "12px", "color": "#666", "marginBottom": "0.5rem"}),
            dcc.Checklist(
                id="cg-opt-tiers-enable",
                options=[{"label": " use_electrode_scoring_tiers — penalize electrodes per the "
                          "tier assignments below, in a 4-level cascade that only unlocks a "
                          "higher tier once every lower tier fails to meet the hard constraint. "
                          "Fiducials (Nz/Iz/A1/"
                          "A2 etc.) are always permanently excluded; Tier 4 is a searchable "
                          "last resort, not excluded.",
                          "value": "use_electrode_scoring_tiers"}],
                value=[k for k in ("use_electrode_scoring_tiers",) if cd.OPTIMIZER_DEFAULTS[k]],
            ),
            html.Div([
                html.Div([
                    html.Label("override CSV path (blank = canonical "
                               "code/pipeline/electrode_tiers_acceptable_vs_nogo.csv)"),
                    dcc.Input(id="cg-opt-tiers-csv-path", type="text", value="",
                              placeholder="", style={"width": "100%"}),
                ], style={"minWidth": "320px", "marginRight": "1rem", "flex": "1 1 320px"}),
                html.Div([
                    html.Label("Tier 1 weight"),
                    dcc.Input(id="cg-opt-tiers-weight-1", type="number", step=0.01,
                              value=cd.OPTIMIZER_DEFAULTS["electrode_tier_weights"]["1"],
                              style={"width": "100%"}),
                ], style={"minWidth": "100px", "marginRight": "1rem"}),
                html.Div([
                    html.Label("Tier 2 weight"),
                    dcc.Input(id="cg-opt-tiers-weight-2", type="number", step=0.01,
                              value=cd.OPTIMIZER_DEFAULTS["electrode_tier_weights"]["2"],
                              style={"width": "100%"}),
                ], style={"minWidth": "100px", "marginRight": "1rem"}),
                html.Div([
                    html.Label("Tier 3 weight"),
                    dcc.Input(id="cg-opt-tiers-weight-3", type="number", step=0.01,
                              value=cd.OPTIMIZER_DEFAULTS["electrode_tier_weights"]["3"],
                              style={"width": "100%"}),
                ], style={"minWidth": "100px", "marginRight": "1rem"}),
                html.Div([
                    html.Label("Tier 4 weight"),
                    dcc.Input(id="cg-opt-tiers-weight-4", type="number", step=0.01,
                              value=cd.OPTIMIZER_DEFAULTS["electrode_tier_weights"]["4"],
                              style={"width": "100%"}),
                ], style={"minWidth": "100px", "marginRight": "1rem"}),
            ], style={"display": "flex", "flexWrap": "wrap", "marginTop": "0.5rem"}),
            html.Div([
                html.Label("tier_penalty weight (applied to the overall tier cost above)"),
                dcc.Input(id="cg-opt-loc-weight-tier-penalty", type="number", step=0.05,
                          value=cd.OPTIMIZER_DEFAULTS["location_score_weights"]["tier_penalty"],
                          style={"width": "90px", "marginLeft": "0.5rem"}),
            ], style={"marginTop": "0.5rem", "display": "flex", "alignItems": "center"}),
            html.P("Read-only preview of what the pipeline will auto-load at run time "
                   "(electrode_tiers itself is left for the CSV importer to populate — "
                   "edit the CSV to change assignments, or point at a different one above).",
                   style={"fontSize": "12px", "color": "#666", "marginTop": "0.5rem"}),
            html.Div(id="cg-opt-tiers-preview"),
        ], level=1),

        _details("Non-ROI subgroup hard constraints", [
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
        ], level=1),

        _details("ROI subgroup hard constraints", [
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
        ], level=1),
    ]),

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

    html.H3("Preference"),
    html.P("Save everything above (cap, every mask — ROI, non-ROI, and any additional ones — "
           "every optimizer/electrode/flag setting) as a named, reusable Preference — not tied "
           "to any subject. Apply it to a new subject from the Apply Preference page: it builds "
           "the masks, registers the cap, and generates the config, so you don't have to set "
           "all of this up again every time.",
           style={"fontSize": "13px", "color": "#666"}),
    html.Div([
        dcc.Input(id="cg-preset-name", type="text", placeholder="Preference name, e.g. hippocampus_focality",
                  style={"marginRight": "0.5rem", "minWidth": "280px"}),
        html.Button("Save as Preference", id="cg-preset-save-btn", n_clicks=0),
    ], style={"display": "flex", "flexWrap": "wrap", "alignItems": "center", "marginBottom": "0.5rem"}),
    dcc.Checklist(
        id="cg-preset-overwrite-confirm",
        options=[{"label": " overwrite the existing preference with this name", "value": "confirm"}],
        value=[], style={"marginBottom": "0.5rem"},
    ),
    html.Div(id="cg-preset-save-status", style={"fontSize": "13px", "marginBottom": "1.5rem"}),

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
    Input("cg-roi-name-dropdown", "value"),
    Input("cg-nonroi-name-dropdown", "value"),
    Input("cg-cap-dropdown", "value"),
    Input("cg-opt-constraint-groups-table", "data"),
    Input("cg-opt-roi-constraint-groups-table", "data"),
    Input("cg-opt-bg-mask-dropdown", "value"),
)
def _update_readiness_table(subject_ids, roi_name, non_roi_name,
                            cap_path, constraint_group_rows, roi_constraint_group_rows, bg_mask_name):
    if not subject_ids or not roi_name:
        return []
    constraint_group_masks = (_constraint_group_mask_names(constraint_group_rows)
                              + _constraint_group_mask_names(roi_constraint_group_rows)
                              + ([bg_mask_name.strip()] if bg_mask_name and bg_mask_name.strip() else []))
    # roi_atlas/non_roi_atlas are irrelevant here — roi_uses_existing_mask=
    # True makes subject_readiness() check "does the mask file already
    # exist" regardless of what built it (recipe book, Mask Generation
    # directly, or anything else); the atlas value only matters for the
    # final config JSON (build_roi_dict(), in _on_generate_click below).
    matrix = cd.readiness_matrix(subject_ids, None, roi_name, None, non_roi_name,
                                 cap_path, constraint_group_masks=constraint_group_masks,
                                 roi_uses_existing_mask=True, non_roi_uses_existing_mask=True)
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
    Input("cg-roi-name-dropdown", "value"),
    Input("cg-nonroi-name-dropdown", "value"),
    Input("cg-cap-dropdown", "value"),
    Input("cg-goals-checklist", "value"),
    Input("cg-opt-constraint-groups-table", "data"),
    Input("cg-opt-roi-constraint-groups-table", "data"),
    Input("cg-opt-bg-mask-dropdown", "value"),
    State("cg-generate-table", "selected_rows"),
)
def _update_generate_table(subject_ids, roi_name, non_roi_name,
                           cap_path, goals, constraint_group_rows, roi_constraint_group_rows,
                           bg_mask_name, prev_selected):
    if not subject_ids or not roi_name or not goals:
        return [], []

    constraint_group_masks = (_constraint_group_mask_names(constraint_group_rows)
                              + _constraint_group_mask_names(roi_constraint_group_rows)
                              + ([bg_mask_name.strip()] if bg_mask_name and bg_mask_name.strip() else []))
    matrix = cd.readiness_matrix(subject_ids, None, roi_name, None, non_roi_name,
                                 cap_path, constraint_group_masks=constraint_group_masks,
                                 roi_uses_existing_mask=True, non_roi_uses_existing_mask=True)
    ready_subjects = [sid for sid in subject_ids if matrix[sid]["ready"]]

    rows = []
    for sid in ready_subjects:
        paths = [cd.CONFIGS_DIR / cd.config_filename(sid, roi_name, g) for g in goals]
        existing = [p for p in paths if p.exists()]
        files_display = f"{len(paths)} file(s): " + ", ".join(p.name for p in paths)
        overwrite_display = f"⚠ {len(existing)} will overwrite" if existing else "new file(s)"
        rows.append({"subject": sid, "files_display": files_display, "overwrite_display": overwrite_display})

    if ctx.triggered_id in ("cg-subject-dropdown", "cg-roi-name-dropdown",
                           "cg-nonroi-name-dropdown", "cg-cap-dropdown", "cg-goals-checklist") or not prev_selected:
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
    Input("cg-roi-name-dropdown", "value"),
    Input("cg-nonroi-name-dropdown", "value"),
    Input("cg-overwrite-confirm", "value"),
)
def _update_generate_readiness(rows, selected_rows, roi_name, non_roi_name, confirm_value):
    rows = rows or []
    selected_rows = selected_rows or []
    selected = [rows[i] for i in selected_rows if i < len(rows)]

    problems = []
    if not selected:
        problems.append("select at least one ready subject")
    if not roi_name:
        problems.append("select a ROI mask")
    if not non_roi_name:
        problems.append("select a non-ROI mask")

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
    Input("cg-recipe-book-dropdown", "value"),
    Input("cg-subject-dropdown", "value"),
)
def _update_constraint_group_mask_options(book_name, subject_ids):
    return {"mask_name": {"options": _mask_name_options(book_name, subject_ids), "clearable": True}}


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
    Input("cg-recipe-book-dropdown", "value"),
    Input("cg-subject-dropdown", "value"),
)
def _update_roi_constraint_group_mask_options(book_name, subject_ids):
    return {"mask_name": {"options": _mask_name_options(book_name, subject_ids), "clearable": True}}


@callback(
    Output("cg-opt-bg-mask-dropdown", "options"),
    Input("cg-recipe-book-dropdown", "value"),
    Input("cg-subject-dropdown", "value"),
)
def _update_bg_mask_options(book_name, subject_ids):
    return _mask_name_options(book_name, subject_ids)


@callback(
    Output("cg-opt-tiers-preview", "children"),
    Input("cg-opt-tiers-csv-path", "value"),
)
def _update_tiers_preview(csv_path_override):
    preview = cd.electrode_tiers_preview((csv_path_override or "").strip() or None)
    if preview["error"]:
        return html.P(f"⚠ could not read {preview['csv_path']}: {preview['error']}",
                      style={"color": "#a00", "fontSize": "12px"})
    rows = [{"electrode": name, "tier": str(t)} for name, t in sorted(preview["tiers"].items())]
    rows += [{"electrode": name, "tier": "Fiducials (excluded)"} for name in preview["excluded"]]
    return html.Div([
        html.P(f"{preview['csv_path']}  —  {len(preview['tiers'])} tier assignment(s), "
               f"{len(preview['excluded'])} Fiducials permanently excluded (Tier 4 is a "
               f"searchable last resort, not excluded)",
               style={"fontSize": "12px", "color": "#666"}),
        _styled_table("cg-opt-tiers-preview-table", [
            {"name": "electrode", "id": "electrode"},
            {"name": "tier", "id": "tier"},
        ], data=rows),
    ])


def _build_overrides(roi_name, roi_atlas, non_roi_name, goals,
                     postproc, cpus, focality_nonroi, focality_roi, opt_checkboxes,
                     hier_enable, hier_num_iter, hier_neighbours_str, hier_early_stop_pct,
                     constraint_group_rows, roi_constraint_group_rows,
                     amp_enable, amp_top_n, amp_min_ma, amp_max_ma, amp_step_ma, amp_max_per_pair_ma,
                     amp_weight_roc, amp_weight_roi_mean, amp_weight_non_roi_mean, amp_weight_focality_ratio,
                     bg_enable, bg_mask_name, bg_max_elements, bg_hotspot_pct,
                     tiers_enable, tiers_csv_path, tiers_weight_1, tiers_weight_2,
                     tiers_weight_3, tiers_weight_4,
                     loc_weight_main, loc_weight_background, loc_weight_background_hotspot,
                     loc_weight_tier_penalty,
                     elec_diameter, elec_gel, elec_max_current, flags_checklist):
    """Shared by _on_generate_click (actually generate configs for selected
    subjects) and _on_save_preference_click (bundle the same settings into
    a reusable Preference, no subject involved) — both read from the exact
    same page components, so this is the one place that turns "whatever's
    currently filled in" into (optimizer_overrides, electrode_overrides,
    flags). Returns (error_message_or_None, optimizer_overrides,
    electrode_overrides, flags) — error_message is a ready-to-render
    html.Div(...) on validation failure, None on success."""
    if not roi_name or not non_roi_name or not goals:
        return (html.Div("Check ROI/non-ROI/goal selection.", style={"color": "#a00"}), None, None, None)

    use_hierarchical = "use_hierarchical_search" in (hier_enable or [])
    num_fine_iter = int(hier_num_iter) if hier_num_iter else 0
    neighbours_per_iter = []
    if hier_neighbours_str and hier_neighbours_str.strip():
        try:
            neighbours_per_iter = [int(x.strip()) for x in hier_neighbours_str.split(",") if x.strip()]
        except ValueError:
            return (html.Div(f"neighbours per iteration must be comma-separated integers "
                             f"(e.g. 8,6,4) — got: {hier_neighbours_str!r}",
                             style={"color": "#a00"}), None, None, None)
    if use_hierarchical and num_fine_iter and len(neighbours_per_iter) != num_fine_iter:
        return (html.Div(f"neighbours per iteration must have exactly {num_fine_iter} value(s) "
                         f"(one per fine iteration) — got {len(neighbours_per_iter)}.",
                         style={"color": "#a00"}), None, None, None)

    constraint_groups = []
    for row in (constraint_group_rows or []):
        grp_name = (row.get("name") or "").strip()
        mask_name = (row.get("mask_name") or "").strip()
        max_v = row.get("max_mean_V_m")
        if not grp_name and not mask_name and max_v in (None, ""):
            continue   # blank row (added but never filled in) — skip silently
        if not grp_name or not mask_name or max_v in (None, ""):
            return (html.Div("each non-ROI subgroup constraint row needs name, mask_name, "
                             "and max_mean_V_m — remove any incomplete rows.",
                             style={"color": "#a00"}), None, None, None)
        try:
            max_v = float(max_v)
        except (TypeError, ValueError):
            return (html.Div(f"constraint group '{grp_name}': max_mean_V_m must be a number, "
                             f"got {max_v!r}", style={"color": "#a00"}), None, None, None)
        constraint_groups.append({"name": grp_name, "mask_name": mask_name, "max_mean_V_m": max_v})

    roi_constraint_groups = []
    for row in (roi_constraint_group_rows or []):
        grp_name = (row.get("name") or "").strip()
        mask_name = (row.get("mask_name") or "").strip()
        min_v = row.get("min_mean_V_m")
        if not grp_name and not mask_name and min_v in (None, ""):
            continue   # blank row (added but never filled in) — skip silently
        if not grp_name or not mask_name or min_v in (None, ""):
            return (html.Div("each ROI subgroup constraint row needs name, mask_name, "
                             "and min_mean_V_m — remove any incomplete rows.",
                             style={"color": "#a00"}), None, None, None)
        try:
            min_v = float(min_v)
        except (TypeError, ValueError):
            return (html.Div(f"ROI constraint group '{grp_name}': min_mean_V_m must be a number, "
                             f"got {min_v!r}", style={"color": "#a00"}), None, None, None)
        roi_constraint_groups.append({"name": grp_name, "mask_name": mask_name, "min_mean_V_m": min_v})

    use_amplitude_sweep = "use_amplitude_sweep" in (amp_enable or [])
    if use_amplitude_sweep:
        if amp_min_ma is None or amp_max_ma is None or amp_step_ma is None:
            return (html.Div("amplitude sweep: min/max/step mA are all required when "
                             "use_amplitude_sweep is enabled.", style={"color": "#a00"}), None, None, None)
        if amp_min_ma > amp_max_ma:
            return (html.Div(f"amplitude sweep: min mA ({amp_min_ma}) can't exceed max mA ({amp_max_ma}).",
                             style={"color": "#a00"}), None, None, None)
        if amp_step_ma <= 0:
            return (html.Div("amplitude sweep: step mA must be positive.", style={"color": "#a00"}), None, None, None)

    optimizer_overrides = {
        "postproc": postproc,
        "cpus": cpus,
        "focality_threshold": [focality_nonroi, focality_roi],
        "hard_roi_constraint": "hard_roi_constraint" in (opt_checkboxes or []),
        "no_adjacent_electrodes": "no_adjacent_electrodes" in (opt_checkboxes or []),
        "use_quadrant_constraint": "use_quadrant_constraint" in (opt_checkboxes or []),
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
        # use_composite_location_score has no checkbox of its own on this
        # page — it's the pipeline's shared AND-gate for BOTH terms below
        # (score = main*raw_score - ...), so it's derived automatically
        # from whichever of the two is actually on, rather than being a
        # separate switch the user has to also remember to check.
        "use_composite_location_score": (("minimize_background_field" in (bg_enable or []))
                                         or ("use_electrode_scoring_tiers" in (tiers_enable or []))),
        "minimize_background_field": "minimize_background_field" in (bg_enable or []),
        "background_mask_name": (bg_mask_name.strip() if bg_mask_name and bg_mask_name.strip() else None),
        "max_background_elements": (int(bg_max_elements) if bg_max_elements
                                    else cd.OPTIMIZER_DEFAULTS["max_background_elements"]),
        "background_hotspot_percentile": (bg_hotspot_pct if bg_hotspot_pct is not None
                                          else cd.OPTIMIZER_DEFAULTS["background_hotspot_percentile"]),
        "use_electrode_scoring_tiers": "use_electrode_scoring_tiers" in (tiers_enable or []),
        # electrode_tiers itself is deliberately left at the default {} — the
        # pipeline auto-populates it from the CSV at run time (see
        # GUI_WIRING_NOTES.md section 3); only the CSV path/weights are
        # GUI-editable here, not a hand-built per-electrode picker.
        "electrode_tiers_csv": (tiers_csv_path.strip() if tiers_csv_path and tiers_csv_path.strip() else None),
        "electrode_tier_weights": {
            "1": tiers_weight_1 if tiers_weight_1 is not None else cd.OPTIMIZER_DEFAULTS["electrode_tier_weights"]["1"],
            "2": tiers_weight_2 if tiers_weight_2 is not None else cd.OPTIMIZER_DEFAULTS["electrode_tier_weights"]["2"],
            "3": tiers_weight_3 if tiers_weight_3 is not None else cd.OPTIMIZER_DEFAULTS["electrode_tier_weights"]["3"],
            "4": tiers_weight_4 if tiers_weight_4 is not None else cd.OPTIMIZER_DEFAULTS["electrode_tier_weights"]["4"],
        },
        "location_score_weights": {
            "main": loc_weight_main if loc_weight_main is not None else cd.OPTIMIZER_DEFAULTS["location_score_weights"]["main"],
            "background": (loc_weight_background if loc_weight_background is not None
                           else cd.OPTIMIZER_DEFAULTS["location_score_weights"]["background"]),
            "background_hotspot": (loc_weight_background_hotspot if loc_weight_background_hotspot is not None
                                   else cd.OPTIMIZER_DEFAULTS["location_score_weights"]["background_hotspot"]),
            "tier_penalty": (loc_weight_tier_penalty if loc_weight_tier_penalty is not None
                             else cd.OPTIMIZER_DEFAULTS["location_score_weights"]["tier_penalty"]),
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
    return None, optimizer_overrides, electrode_overrides, flags


@callback(
    Output("cg-generate-results", "children"),
    Input("cg-generate-button", "n_clicks"),
    State("cg-generate-table", "data"),
    State("cg-generate-table", "selected_rows"),
    State("cg-recipe-book-dropdown", "value"),
    State("cg-roi-name-dropdown", "value"),
    State("cg-nonroi-name-dropdown", "value"),
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
    State("cg-opt-bg-enable", "value"),
    State("cg-opt-bg-mask-dropdown", "value"),
    State("cg-opt-bg-max-elements", "value"),
    State("cg-opt-bg-hotspot-pct", "value"),
    State("cg-opt-tiers-enable", "value"),
    State("cg-opt-tiers-csv-path", "value"),
    State("cg-opt-tiers-weight-1", "value"),
    State("cg-opt-tiers-weight-2", "value"),
    State("cg-opt-tiers-weight-3", "value"),
    State("cg-opt-tiers-weight-4", "value"),
    State("cg-opt-loc-weight-main", "value"),
    State("cg-opt-loc-weight-background", "value"),
    State("cg-opt-loc-weight-background-hotspot", "value"),
    State("cg-opt-loc-weight-tier-penalty", "value"),
    State("cg-elec-diameter", "value"),
    State("cg-elec-gel-thickness", "value"),
    State("cg-elec-max-current", "value"),
    State("cg-flags-checklist", "value"),
    prevent_initial_call=True,
)
def _on_generate_click(_n_clicks, rows, selected_rows, book_name, roi_name, non_roi_name, goals, cap_path,
                       postproc, cpus, focality_nonroi, focality_roi, opt_checkboxes,
                       hier_enable, hier_num_iter, hier_neighbours_str, hier_early_stop_pct,
                       constraint_group_rows, roi_constraint_group_rows,
                       amp_enable, amp_top_n, amp_min_ma, amp_max_ma, amp_step_ma, amp_max_per_pair_ma,
                       amp_weight_roc, amp_weight_roi_mean, amp_weight_non_roi_mean, amp_weight_focality_ratio,
                       bg_enable, bg_mask_name, bg_max_elements, bg_hotspot_pct,
                       tiers_enable, tiers_csv_path, tiers_weight_1, tiers_weight_2,
                       tiers_weight_3, tiers_weight_4,
                       loc_weight_main, loc_weight_background, loc_weight_background_hotspot,
                       loc_weight_tier_penalty,
                       elec_diameter, elec_gel, elec_max_current, flags_checklist):
    rows = rows or []
    selected_rows = selected_rows or []
    subject_ids = [rows[i]["subject"] for i in selected_rows if i < len(rows)]

    if not subject_ids:
        return html.Div("Nothing to generate — select at least one subject.", style={"color": "#a00"})

    # Resolved for the final config JSON's roi/non_roi dicts (build_roi_dict()
    # via cd.generate_configs() below) — "combine" kind resolves to
    # (None, {}), same as an Allen-sourced mask: no atlas means the file
    # must already exist, which "Build missing masks" above is responsible
    # for, not this step.
    roi_atlas, roi_label_ids, _ = _resolve_recipe(book_name, roi_name)
    non_roi_atlas, non_roi_label_ids, _ = _resolve_recipe(book_name, non_roi_name)

    error, optimizer_overrides, electrode_overrides, flags = _build_overrides(
        roi_name, roi_atlas, non_roi_name, goals,
        postproc, cpus, focality_nonroi, focality_roi, opt_checkboxes,
        hier_enable, hier_num_iter, hier_neighbours_str, hier_early_stop_pct,
        constraint_group_rows, roi_constraint_group_rows,
        amp_enable, amp_top_n, amp_min_ma, amp_max_ma, amp_step_ma, amp_max_per_pair_ma,
        amp_weight_roc, amp_weight_roi_mean, amp_weight_non_roi_mean, amp_weight_focality_ratio,
        bg_enable, bg_mask_name, bg_max_elements, bg_hotspot_pct,
        tiers_enable, tiers_csv_path, tiers_weight_1, tiers_weight_2, tiers_weight_3, tiers_weight_4,
        loc_weight_main, loc_weight_background, loc_weight_background_hotspot, loc_weight_tier_penalty,
        elec_diameter, elec_gel, elec_max_current, flags_checklist)
    if error is not None:
        return error

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
# Save current settings as a Preference — reuses _build_overrides (same
# validation/shape _on_generate_click uses) plus the ROI/non-ROI/extra-mask
# recipes, so a Preference and a one-off Generate are built from identical
# logic and can never silently drift apart.
# ═════════════════════════════════════════════════════════════════════════════

def _resolve_source_cap_path(registered_cap_value: str | None) -> tuple[str | None, str | None]:
    """cg-cap-dropdown's value is an ALREADY-REGISTERED, subject-specific
    CSV (m2m_{id}/eeg_positions/{name}.csv — see _load_caps' own docstring)
    — its coordinates are in that one subject's native space, not MNI, so
    it can NOT be reused directly to register a DIFFERENT (new) subject;
    doing so would silently warp already-native coordinates as if they
    were MNI ones. Only the filename stem (the cap's identity) is usable
    here — resolve it back to the real MNI-space source file (project or
    SimNIBS built-in) that one_shot_discovery.register_cap() actually
    needs. Returns (source_path, error)."""
    if not registered_cap_value:
        return None, "select a cap first"
    stem = cap_discovery.cap_stem(registered_cap_value)
    match = next((c for c in cap_discovery.list_available_caps() if c["name"] == stem), None)
    if match is None:
        return None, (f'cap "{stem}" has no matching MNI-space source in project caps or SimNIBS '
                      "built-ins — can't be reused to register a different subject")
    return match["path"], None


@callback(
    Output("cg-preset-save-status", "children"),
    Input("cg-preset-save-btn", "n_clicks"),
    State("cg-preset-name", "value"),
    State("cg-preset-overwrite-confirm", "value"),
    State("cg-cap-dropdown", "value"),
    State("cg-recipe-book-dropdown", "value"),
    State("cg-roi-name-dropdown", "value"),
    State("cg-nonroi-name-dropdown", "value"),
    State("cg-goals-checklist", "value"),
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
    State("cg-opt-bg-enable", "value"),
    State("cg-opt-bg-mask-dropdown", "value"),
    State("cg-opt-bg-max-elements", "value"),
    State("cg-opt-bg-hotspot-pct", "value"),
    State("cg-opt-tiers-enable", "value"),
    State("cg-opt-tiers-csv-path", "value"),
    State("cg-opt-tiers-weight-1", "value"),
    State("cg-opt-tiers-weight-2", "value"),
    State("cg-opt-tiers-weight-3", "value"),
    State("cg-opt-tiers-weight-4", "value"),
    State("cg-opt-loc-weight-main", "value"),
    State("cg-opt-loc-weight-background", "value"),
    State("cg-opt-loc-weight-background-hotspot", "value"),
    State("cg-opt-loc-weight-tier-penalty", "value"),
    State("cg-elec-diameter", "value"),
    State("cg-elec-gel-thickness", "value"),
    State("cg-elec-max-current", "value"),
    State("cg-flags-checklist", "value"),
    prevent_initial_call=True,
)
def _on_save_preference_click(_n_clicks, preset_name, overwrite_value, registered_cap_value,
                              book_name, roi_name, non_roi_name,
                              goals, postproc, cpus, focality_nonroi, focality_roi, opt_checkboxes,
                              hier_enable, hier_num_iter, hier_neighbours_str, hier_early_stop_pct,
                              constraint_group_rows, roi_constraint_group_rows,
                              amp_enable, amp_top_n, amp_min_ma, amp_max_ma, amp_step_ma, amp_max_per_pair_ma,
                              amp_weight_roc, amp_weight_roi_mean, amp_weight_non_roi_mean, amp_weight_focality_ratio,
                              bg_enable, bg_mask_name, bg_max_elements, bg_hotspot_pct,
                              tiers_enable, tiers_csv_path, tiers_weight_1, tiers_weight_2,
                              tiers_weight_3, tiers_weight_4,
                              loc_weight_main, loc_weight_background, loc_weight_background_hotspot,
                              loc_weight_tier_penalty,
                              elec_diameter, elec_gel, elec_max_current, flags_checklist):
    name_err = pd.validate_name((preset_name or "").strip())
    if name_err:
        return html.Div(f"✗ {name_err}", style={"color": "#a00"})

    if not book_name:
        return html.Div("✗ select a recipe book first.", style={"color": "#a00"})
    book = mrd.load_recipe_book(book_name)
    if not book:
        return html.Div(f"✗ could not load recipe book '{book_name}'.", style={"color": "#a00"})
    if not roi_name or not non_roi_name:
        return html.Div("✗ select both a ROI and a non-ROI mask first.", style={"color": "#a00"})

    # Freezes a COPY of every referenced recipe (ROI, non-ROI, every
    # subgroup/background constraint mask, and anything a "combine" among
    # them transitively depends on) into the preference itself — so it
    # behaves identically later even if this book is edited or deleted
    # afterward, and Apply Preference never needs to re-read it. Any name
    # with no recipe anywhere in the book (even as a combine source) means
    # this preference could never be fully built on a brand-new subject,
    # so saving is refused rather than producing something that'll just
    # fail later.
    referenced = _referenced_mask_names(roi_name, non_roi_name, constraint_group_rows,
                                        roi_constraint_group_rows, bg_mask_name)
    frozen_masks, missing = _freeze_recipes(book.get("masks") or {}, referenced)
    if missing:
        return html.Div(
            f"✗ referenced but not found anywhere in recipe book '{book_name}': "
            f"{', '.join(sorted(set(missing)))} — add a recipe for it in Mask Generation, or "
            "fix the name.", style={"color": "#a00"})

    cap_path, cap_err = _resolve_source_cap_path(registered_cap_value)
    if cap_err:
        return html.Div(f"✗ {cap_err}", style={"color": "#a00"})

    error, optimizer_overrides, electrode_overrides, flags = _build_overrides(
        roi_name, None, non_roi_name, goals,
        postproc, cpus, focality_nonroi, focality_roi, opt_checkboxes,
        hier_enable, hier_num_iter, hier_neighbours_str, hier_early_stop_pct,
        constraint_group_rows, roi_constraint_group_rows,
        amp_enable, amp_top_n, amp_min_ma, amp_max_ma, amp_step_ma, amp_max_per_pair_ma,
        amp_weight_roc, amp_weight_roi_mean, amp_weight_non_roi_mean, amp_weight_focality_ratio,
        bg_enable, bg_mask_name, bg_max_elements, bg_hotspot_pct,
        tiers_enable, tiers_csv_path, tiers_weight_1, tiers_weight_2, tiers_weight_3, tiers_weight_4,
        loc_weight_main, loc_weight_background, loc_weight_background_hotspot, loc_weight_tier_penalty,
        elec_diameter, elec_gel, elec_max_current, flags_checklist)
    if error is not None:
        return error

    data = {
        "cap_path": cap_path,
        "roi_name": roi_name,
        "non_roi_name": non_roi_name,
        "masks": frozen_masks,
        "goals": goals or [],
        "optimizer_overrides": optimizer_overrides,
        "electrode_overrides": electrode_overrides,
        "flags": flags,
    }
    overwrite = "confirm" in (overwrite_value or [])
    result = pd.save_preset((preset_name or "").strip(), data, overwrite=overwrite)
    if not result["success"]:
        return html.Div(f"✗ {result['error']}", style={"color": "#a00"})
    return html.Div(f"✓ saved → {result['path']}", style={"color": "#060"})


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

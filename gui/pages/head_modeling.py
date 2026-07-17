"""
pages/head_modeling.py — Phase 0: generate a subject's head model (m2m_{id})
via SimNIBS charm, the prerequisite for every later phase.

charm is a real external command-line tool (~30-90 min per subject), run
either locally (job_runner.py's background-job mechanism, same pattern as
Phase 3's leadfield generation / one-off FEM) or on SCITAS
(charm_discovery.run_charm_on_scitas — submits charm_scitas.sbatch over SSH,
blocks on the SLURM queue, scp's m2m_{id}/ back). Both fit the exact same
job_runner contract, so this page's polling UI doesn't care which one ran.

Single subject at a time by design: charm runs this long are typically
either a one-off/local test (this page) or dispatched as one SCITAS job per
subject (the existing submit_charm_scitas.sh pattern) — batching many
90-minute local runs isn't a useful thing to build here.
"""
import os

import dash
from dash import html, dcc, callback, Input, Output, State

import charm_discovery as chd
import job_runner as jr

dash.register_page(__name__, path="/", name="Head Modeling", category="Preprocessing", order=1)

layout = html.Div([
    html.H2("Head Modeling (charm)"),
    html.P("Generates the subject head model (m2m_{id}/) from T1w (required) and T2w (optional, "
           "improves segmentation quality) — the prerequisite for every later phase.",
           style={"fontSize": "13px", "color": "#666"}),

    html.Div([
        html.Label("Subject"),
        dcc.Dropdown(id="hm-subject-dropdown", placeholder="Select subject...", style={"maxWidth": "300px"}),
    ], style={"marginBottom": "1rem"}),

    html.Div(id="hm-status", style={"marginBottom": "1rem", "fontSize": "13px"}),

    html.Div([
        dcc.Checklist(id="hm-use-t2", options=[{"label": " Use T2w if available", "value": "use_t2"}],
                      value=["use_t2"], style={"marginBottom": "0.5rem"}),
        dcc.Checklist(id="hm-force", options=[{"label": " Force (redo even if already complete)",
                                               "value": "force"}], value=[]),
    ], style={"marginBottom": "1rem"}),

    html.Div([
        html.Label("Run on"),
        dcc.RadioItems(
            id="hm-run-location",
            options=[
                {"label": " Local (this machine)", "value": "local"},
                {"label": " SCITAS (jed.hpc.epfl.ch)", "value": "scitas"},
            ],
            value="local",
        ),
    ], style={"marginBottom": "1rem"}),

    html.Button("Start Head Modeling", id="hm-start-button", n_clicks=0,
                style={"padding": "0.5rem 1.5rem"}),

    dcc.Store(id="hm-job-store"),
    dcc.Interval(id="hm-poll-interval", interval=3000, disabled=True),
    dcc.Loading(html.Div(id="hm-run-status", style={"marginTop": "1rem"})),
])


@callback(Output("hm-subject-dropdown", "options"), Input("hm-subject-dropdown", "id"))
def _load_subjects(_):
    options = []
    for s in chd.discover_subjects():
        if not s["has_rawdata"]:
            continue
        status = chd.charm_status(s["subject_id"])
        tag = "m2m ✓" if status["m2m_complete"] else ("m2m partial" if status["m2m_partial"] else "needs charm")
        options.append({"label": f"{s['subject_id']}   [{tag}]", "value": s["subject_id"]})
    return options


@callback(Output("hm-status", "children"), Input("hm-subject-dropdown", "value"))
def _update_status(subject_id):
    if not subject_id:
        return ""
    status = chd.charm_status(subject_id)

    def _line(label, ok, extra=""):
        return html.P(f"{'✓' if ok else '✗'} {label}{extra}",
                      style={"color": "#060" if ok else "#a00", "margin": "0.15rem 0"})

    children = [
        _line("T1w found", status["t1_exists"]),
        _line("T2w found (optional)", status["t2_exists"]),
    ]
    if status["m2m_complete"]:
        children.append(_line("m2m already complete", True, " — Start will skip unless Force is checked"))
    elif status["m2m_partial"]:
        children.append(html.P("⚠ partial m2m_ folder found (a previous failed/interrupted run) — "
                               "will be cleaned up before restarting", style={"color": "#a60"}))
    return html.Div(children)


@callback(
    Output("hm-job-store", "data"),
    Output("hm-poll-interval", "disabled"),
    Output("hm-run-status", "children"),
    Input("hm-start-button", "n_clicks"),
    State("hm-subject-dropdown", "value"),
    State("hm-use-t2", "value"),
    State("hm-force", "value"),
    State("hm-run-location", "value"),
    prevent_initial_call=True,
)
def _on_start_click(_n_clicks, subject_id, use_t2_value, force_value, run_location):
    if not subject_id:
        return None, True, html.Div("Select a subject first.", style={"color": "#a00"})

    use_t2 = "use_t2" in (use_t2_value or [])
    force = "force" in (force_value or [])

    base_dir = os.path.join(chd.PROJECT_DIR, "derivatives", "SimNIBS", f"sub-{subject_id}", "_charm_jobs")
    _job_id, job_dir = jr.new_job_dir(base_dir)

    if run_location == "scitas":
        jr.start_local_job(job_dir, chd.run_charm_on_scitas, subject_id, use_t2, force)
        note = ("Started on SCITAS — polling every 3s (submits the job, then waits on the SLURM "
                "queue; charm itself typically takes 30-90 min once running, plus queue time).")
    else:
        jr.start_local_job(job_dir, chd.run_charm, subject_id, use_t2, force)
        note = "Started — polling every 3s (charm typically takes 30-90 min)..."
    return job_dir, False, html.Div(note, style={"color": "#666"})


@callback(
    Output("hm-run-status", "children", allow_duplicate=True),
    Output("hm-poll-interval", "disabled", allow_duplicate=True),
    Input("hm-poll-interval", "n_intervals"),
    State("hm-job-store", "data"),
    prevent_initial_call=True,
)
def _poll_charm(_n_intervals, job_dir):
    if not job_dir:
        return "", True
    status = jr.read_status(job_dir)
    if not status:
        return html.Div("Waiting for job to start...", style={"color": "#666"}), False
    if status["state"] == "running":
        import time
        elapsed = int(time.time() - status["started_at"])
        return html.Div(f"Running... ({elapsed}s elapsed)", style={"color": "#666"}), False
    if status["state"] == "error":
        return html.Div(f"✗ {status['error']}", style={"color": "#a00"}), True

    result = status["result"]
    if not result.get("success"):
        return html.Pre(f"✗ {result.get('error')}", style={"color": "#a00", "whiteSpace": "pre-wrap",
                                                             "fontSize": "12px"}), True
    cached_note = " (was already complete)" if result.get("cached") else ""
    return html.Div(f"✓ Head model ready{cached_note} → {result['m2m_path']}. "
                    f"Proceed to Mask Generation.", style={"color": "#060"}), True

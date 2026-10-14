"""
one_shot_discovery.py — applies a saved Config Generation preset (see
preset_discovery.py) to one or more subjects: charm (opt-in, skipped if
m2m already exists) -> masks (ROI/non-ROI/every subgroup-constraint mask,
built from the preference's own frozen recipes — see mask_recipe_
discovery.py) -> cap registration -> config generation, in that order,
stopping that subject's chain at the first failing step. Never submits the
actual optimization/simulation job — that stays a separate, manual step on
the existing pages.

Pure Python, no Dash import. Meant to run inside a job_runner background
job (apply_preference_job is the entry point to pass to start_local_job) —
charm alone can be a long, blocking operation.
"""
from __future__ import annotations

import cap_discovery
import charm_discovery
import config_discovery as cd
import job_runner as jr
import mask_recipe_discovery as mrd
from common import PROJECT_DIR

# The chain, in order — used both to drive execution and to let the page
# render a fixed-order progress list regardless of where a given subject
# actually stops.
STEPS = ("charm", "masks", "cap_registration", "config_generation")


def _referenced_mask_names(preference: dict) -> list[str]:
    """Every mask name this preference touches — ROI, non-ROI, and every
    subgroup/background constraint mask_name in optimizer_overrides — the
    full set "masks" (apply_preference_to_subject's own step) has to make
    sure exists before config generation runs."""
    names = set()
    if preference.get("roi_name"):
        names.add(preference["roi_name"])
    if preference.get("non_roi_name"):
        names.add(preference["non_roi_name"])
    opt = preference.get("optimizer_overrides") or {}
    for grp in (opt.get("non_roi_hard_constraint_groups") or []):
        if grp.get("mask_name"):
            names.add(grp["mask_name"])
    for grp in (opt.get("roi_hard_constraint_groups") or []):
        if grp.get("mask_name"):
            names.add(grp["mask_name"])
    if opt.get("background_mask_name"):
        names.add(opt["background_mask_name"])
    return sorted(names)


def _resolve_recipe(masks: dict, name: str | None) -> tuple[str | None, dict]:
    """(atlas, label_ids) for name in `masks` (the preference's own frozen
    recipe dict). None/{} for a "combine" recipe, or an unknown name — same
    "the file must already exist" convention build_roi_dict() already uses
    for Allen-sourced masks; actually building it is "masks" step's job
    (via mask_recipe_discovery.build_missing_masks), already done by the
    time config generation runs."""
    if not name:
        return None, {}
    recipe = masks.get(name) or {}
    if recipe.get("kind") == "combine":
        return None, {}
    return recipe.get("atlas"), recipe.get("label_ids") or {}


def apply_preference_to_subject(subject_id: str, preference: dict, run_charm_locally: bool,
                                project_dir: str = PROJECT_DIR, on_step=None) -> dict:
    """Runs the full chain for ONE subject. on_step(step_name,
    steps_so_far), if given, is called right before each step starts —
    steps_so_far is this subject's own {step_name: result} dict for every
    EARLIER step already completed (so a progress UI can show real
    checkmarks for what's actually done, not just which step is active).
    Returns {"subject_id", "steps": {step_name: {"success","error",...}},
    "success", "stopped_at"} — steps after the first failure are simply
    absent from "steps", not recorded as skipped/failed."""
    steps: dict = {}

    def _stop(step, error):
        return {"subject_id": subject_id, "steps": steps, "success": False, "stopped_at": step}

    def _report(step):
        # Passes a SNAPSHOT of steps-completed-so-far alongside the name of
        # the step about to start — on_step fires BEFORE that step runs, so
        # `steps` at this point already holds every earlier step's real
        # result for this subject, not just "something is happening".
        # Without this, a progress UI could only ever show the one active
        # step name, never which of THIS subject's earlier steps already
        # succeeded/were skipped.
        if on_step:
            on_step(step, dict(steps))

    # ── charm ────────────────────────────────────────────────────────────
    _report("charm")
    charm_st = charm_discovery.charm_status(subject_id, project_dir)
    if charm_st["m2m_complete"]:
        steps["charm"] = {"success": True, "skipped": True, "error": None}
    elif not run_charm_locally:
        err = ("m2m not found for this subject, and local charm is off — enable "
              "\"run charm locally\" above, or submit it on SCITAS via Head Modeling "
              "first, then re-apply this preference.")
        steps["charm"] = {"success": False, "skipped": False, "error": err}
        return _stop("charm", err)
    else:
        r = charm_discovery.run_charm(subject_id, use_t2=True, force=False, project_dir=project_dir)
        steps["charm"] = {"success": r["success"], "skipped": False, "error": r.get("error")}
        if not r["success"]:
            return _stop("charm", r["error"])

    # ── masks (ROI, non-ROI, every subgroup/background constraint mask) ──
    _report("masks")
    masks = preference.get("masks") or {}
    names = _referenced_mask_names(preference)
    results = mrd.build_missing_masks([subject_id], masks, names, project_dir)
    errors = [r for r in results if not r["success"]]
    steps["masks"] = {
        "success": not errors, "skipped": not results,
        "built": {r["mask_name"]: r.get("voxel_count") for r in results if r["success"]},
        "missing": [r["mask_name"] for r in errors],
        "error": "; ".join(f"{r['mask_name']} ({r['error']})" for r in errors) if errors else None,
    }
    if errors:
        return _stop("masks", steps["masks"]["error"])

    # ── cap registration ─────────────────────────────────────────────────
    _report("cap_registration")
    cap_path = preference.get("cap_path")
    if not cap_path:
        err = "no cap_path in preference"
        steps["cap_registration"] = {"success": False, "skipped": False, "error": err}
        return _stop("cap_registration", err)
    reg = cap_discovery.cap_registration_matrix([subject_id], cap_path, project_dir)[subject_id]
    if reg["registered"]:
        steps["cap_registration"] = {"success": True, "skipped": True, "error": None}
    elif not reg["ready"]:
        err = f"cannot register cap — missing: {', '.join(reg['missing'])}"
        steps["cap_registration"] = {"success": False, "skipped": False, "error": err}
        return _stop("cap_registration", err)
    else:
        r = cap_discovery.register_cap(subject_id, cap_path, project_dir)
        steps["cap_registration"] = {"success": r["success"], "skipped": False, "error": r.get("error")}
        if not r["success"]:
            return _stop("cap_registration", r["error"])

    # ── config generation ────────────────────────────────────────────────
    _report("config_generation")
    roi_name, non_roi_name = preference.get("roi_name"), preference.get("non_roi_name")
    roi_atlas, roi_label_ids = _resolve_recipe(masks, roi_name)
    non_roi_atlas, non_roi_label_ids = _resolve_recipe(masks, non_roi_name)
    cg_results = []
    for goal in preference.get("goals") or ["mean"]:
        cg_results.extend(cd.generate_configs(
            [subject_id], roi_name, roi_atlas, roi_label_ids,
            non_roi_name, non_roi_atlas, non_roi_label_ids,
            [goal], preference.get("optimizer_overrides") or {}, preference.get("electrode_overrides") or {},
            preference.get("flags") or {}, cap_path, force=True,
        ))
    errors = [r for r in cg_results if r["status"].startswith("error")]
    steps["config_generation"] = {
        "success": not errors, "skipped": False, "results": cg_results,
        "error": "; ".join(f"{r['goal']}: {r['status']}" for r in errors) if errors else None,
    }
    if errors:
        return _stop("config_generation", steps["config_generation"]["error"])

    cd.write_sbatch_env()
    return {"subject_id": subject_id, "steps": steps, "success": True, "stopped_at": None}


def apply_preference_job(job_dir: str, subject_ids: list[str], preference: dict, run_charm_locally: bool,
                         project_dir: str = PROJECT_DIR) -> dict:
    """The job_runner entry point — pass to start_local_job(job_dir,
    apply_preference_job, job_dir, subject_ids, preference,
    run_charm_locally). Live-updates the job's own status.json after every
    step of every subject (job_runner.report_progress) so the page can show
    a running "subject X — step Y" progress view, not just a bare spinner
    for what can be a multi-minute, multi-subject, multi-step operation.

    Returns {"subjects": {subject_id: apply_preference_to_subject()'s own
    return dict}} — one subject's failure doesn't stop the rest."""
    subjects_out: dict = {}

    def _progress(sid, step, steps_so_far):
        # A partial/in-progress entry for the subject currently running —
        # steps_so_far is real (every earlier step's actual result), so the
        # page can show genuine checkmarks for what's done, not a guess.
        # Overwritten by the real, complete result once this subject
        # actually finishes (see below).
        jr.report_progress(job_dir, current_subject=sid, current_step=step, total_subjects=len(subject_ids),
                           subjects={**subjects_out, sid: {"steps": steps_so_far, "success": None,
                                                           "stopped_at": None}})

    for sid in subject_ids:
        result = apply_preference_to_subject(
            sid, preference, run_charm_locally, project_dir,
            on_step=lambda step, steps_so_far, _sid=sid: _progress(_sid, step, steps_so_far))
        subjects_out[sid] = result
        jr.report_progress(job_dir, current_subject=sid, current_step="done",
                           total_subjects=len(subject_ids), subjects=subjects_out)

    return {"subjects": subjects_out}

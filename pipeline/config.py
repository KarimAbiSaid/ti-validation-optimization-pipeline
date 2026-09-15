"""
Pipeline configuration dataclasses.

Edit defaults here, or override per-run via a JSON config file passed to run_pipeline.py.
All derived paths (m2m_path, aseg_path, etc.) are computed automatically from subject_id
and project_dir — do not set them manually.
"""

from __future__ import annotations
import os
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional

# Ground/reference/fiducial-landmark names some digitized or custom cap CSVs
# include alongside real electrodes — never valid stimulation contacts, even
# when the file's own "Type" column tags them the same as real electrodes
# (seen in practice: a cap CSV where every row, including GND, is tagged
# "Electrode" — no Type-based way to distinguish them there). Matched
# case-insensitively against the cap's "label"/name column. NOTE: "Iz" is a
# real posterior-midline 10-10 electrode, distinct from the "inion" landmark
# below — do not add "Iz" here.
NON_ELECTRODE_NAMES = {
    "gnd", "ground",
    "ref", "reference",
    "nasion", "nz",
    "inion",
    "left pre auricular point", "right pre auricular point", "lpa", "rpa",
    "left exocantion", "right exocantion", "left exocanthion", "right exocanthion", "loc", "roc",
}


def is_stimulation_electrode(name: str) -> bool:
    return (name or "").strip().lower() not in NON_ELECTRODE_NAMES


# Canonical electrode tier reference (Fiducials/Tier 0-4) — lives right next
# to this file (code/pipeline/), NOT under project_dir, so it travels with
# the code repo itself (sharing just code/ still includes it, unlike a
# project-root data file such as BNA_subregions.xlsx). Format: two columns
# "Electrode,Tier" — Tier is either the literal string "Fiducials" or a
# digit "0".."4". A name may list two 10-20/10-10 aliases separated by "/"
# (e.g. "T7/T3") — both aliases get the same tier, since a given cap CSV may
# use either naming convention.
DEFAULT_ELECTRODE_TIERS_CSV = "electrode_tiers_acceptable_vs_nogo.csv"
DEFAULT_ELECTRODE_TIERS_CSV_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), DEFAULT_ELECTRODE_TIERS_CSV)


def load_electrode_tiers_csv(csv_path: str) -> tuple[dict, set]:
    """Parse an "Electrode,Tier" CSV into (tiers, fiducials):
      - tiers: {electrode_name_lower: 0|1|2|3|4} — same shape OptimizerConfig.
        electrode_tiers already expects, ready to merge/assign directly. Tier
        4 IS included here (unlike the earlier single-exclusion-set design) —
        it's a real, searchable last-resort tier now (see
        run_exhaustive_cap_optimization's tier cascade), not a permanent
        exclusion.
      - fiducials: set of electrode_name_lower tagged "Fiducials" (Nz/Iz/A1/
        A2 etc.) — these are NOT real stimulation sites at all (landmark/
        reference points, not scalp positions meant to carry current) and
        are excluded from the search pool unconditionally, at every tier
        cascade level, no exceptions — explicit user decision, distinct from
        Tier 4 (a real, merely-unacceptable scalp position that IS allowed
        as an absolute last resort). The leadfield is still built from
        whatever the base cap CSV contains either way (no pre-filtering
        before the FEM solve — cheap enough, and a pre-built leadfield
        already includes them; only the search itself is restricted).
    Column order is resolved by header name (case-insensitive "electrode"/
    "tier"), not position, to survive a reordered or extended CSV.
    """
    import csv as _csv

    tiers: dict = {}
    fiducials: set = set()
    with open(csv_path, encoding="utf-8-sig", newline="") as f:
        reader = _csv.DictReader(f)
        if not reader.fieldnames:
            return tiers, fiducials
        _cols = {c.strip().lower(): c for c in reader.fieldnames}
        _name_col = _cols.get("electrode") or reader.fieldnames[0]
        _tier_col = _cols.get("tier") or reader.fieldnames[1]
        for row in reader:
            raw_name = (row.get(_name_col) or "").strip()
            raw_tier = (row.get(_tier_col) or "").strip()
            if not raw_name or not raw_tier:
                continue
            names = [n.strip().lower() for n in raw_name.split("/") if n.strip()]
            if raw_tier.lower() == "fiducials":
                fiducials.update(names)
            elif raw_tier in ("0", "1", "2", "3", "4"):
                for n in names:
                    tiers[n] = int(raw_tier)
            # unrecognized tier values are silently skipped — same
            # "don't guess" spirit as the rest of this file; a malformed
            # row just doesn't contribute a tier, it doesn't abort the run.
    return tiers, fiducials


# ── Sub-configs ────────────────────────────────────────────────────────────────

@dataclass
class ReconAllConfig:
    openmp: int = 4         # CPU cores for parallel recon-all
    use_t2: bool = True     # use T2w for pial surface refinement (-T2pial)


@dataclass
class ROIConfig:
    name: str = "thalamus"
    labels: Dict[str, int] = field(default_factory=lambda: {
        "Left-Thalamus":  10,
        "Right-Thalamus": 49,
    })
    method: str = "NIfTI"                       # "NIfTI", "atlas", or "sphere"
    atlas_regions: Optional[List[str]] = None   # used if method="atlas"
    sphere_center: Optional[List[float]] = None # used if method="sphere" [x,y,z] mm
    sphere_radius: Optional[float] = None       # used if method="sphere"
    # If all primary labels produce 0 voxels, retry with these labels and warn.
    # Useful for cortical non-ROI: M1 labels (1024/2024) require full recon-all;
    # fallback to whole cortex (3/42) when only aseg.auto.mgz is available.
    fallback_labels: Optional[Dict[str, int]] = None
    # Brainnetome Atlas (BNA) labels — when set, Section 1 warps BN_Atlas_246_1mm.nii.gz
    # to subject space and extracts these integer labels instead of using FreeSurfer aseg.
    # Requires bna_atlas_path to be set in PipelineConfig.
    # When bna_labels is set, the `labels` field is ignored by Section 1.
    bna_labels: Optional[Dict[str, int]] = None


@dataclass
class ElectrodeConfig:
    shape: str = "ellipse"
    dimensions: List[float] = field(default_factory=lambda: [14.0, 14.0])  # mm
    gel_thickness: float = 1.0      # mm
    current_mA: float = 2.0         # mA per channel
    max_total_current: float = 4.0  # mA total
    n_electrodes: int = 4           # electrodes per channel pair (2 pairs = 4 total)


def leadfield_tag(cap_basename: str, shape: str, dimensions: list, gel_thickness: float) -> str:
    """Filesystem-safe identifier for one (cap, electrode-geometry)
    combination — keys leadfield_volume/ subdirectories in run_pipeline.py
    (and, opt-in, compare_ti_montages.py) so switching electrode settings
    gets its own cache instead of colliding with / invalidating a
    previously-computed leadfield for different settings. ':g' formatting
    keeps e.g. 19.5 and 19.50 identical rather than producing two tags."""
    dims = "x".join(f"{d:g}" for d in dimensions)
    return f"{cap_basename}__{shape}_{dims}_{gel_thickness:g}mm"


@dataclass
class OptimizerConfig:
    # Goal / postprocessing
    goal: str = "focality"   # "mean" | "max" | "focality" | "focality_inv" | "neg_mean" | "neg_max"
                              # "focality" requires non_roi + focality_threshold; others optimize ROI only
    postproc: str = "max_TI" # TI fields: "max_TI" (standard) | "dir_TI" (needs direction vector)
                              # tDCS/general: "magn" | "normal" | "tangential"

    # Focality thresholds (only used when goal="focality")
    # [max_allowed_in_non_ROI_V_m, min_required_in_ROI_V_m]
    focality_threshold: List[float] = field(default_factory=lambda: [0.24, 0.3])

    # Hardware
    cpus: int = 4              # passed to opt.run(cpus=...)

    # LEGACY (TesFlexOptimization/DE — unused now that exhaustive search is
    # the only optimization path; run_pipeline.py's Section 2 TesFlex code is
    # commented out, not deleted, see that file). Kept here only as a record
    # of what these meant — not read by anything anymore.
    # max_iterations:  int        = 200         # DE max generations
    # population_size: int        = 13          # DE population size per dimension
    # tolerance:       float      = 0.1         # convergence tolerance
    # mutation:        List[float] = field(default_factory=lambda: [0.01, 0.5])
    # recombination:   float      = 0.9         # DE crossover probability
    # n_multistart:    int        = 1           # independent restarts (keep best ROI result)
    # anisotropy_type: str  = "scalar"  # "scalar", "vn", "mc", "dir"
    # min_electrode_distance: float = 5.0   # mm — prevents electrode overlap
    # use_eeg_cap_boundary:   bool  = False  # restrict scalp search to EEG cap region
    # eeg_cap_margin_mm:      float = 10.0   # buffer beyond outermost cap electrodes (mm)
    # detailed_results: bool = True   # store full optimization history
    # enable_mapping:   bool = True   # map to nearest EEG net + run mapped sim
    # use_exhaustive_search: bool = False  # now the only path — see run_exhaustive_cap_optimization()

    # Hard ROI dose constraint (only active when goal="focality").
    # When True: skip any montage where mean TI in ROI < focality_threshold[1].
    # Among feasible montages, best focality (ROC) is selected as usual.
    # If no montage meets the threshold, falls back to best mean-ROI montage with a warning.
    # When False: threshold is a soft penalty (SimNIBS default behaviour).
    hard_roi_constraint: bool = False

    # Cap the number of non-ROI elements used in focality scoring.
    # When the fallback non-ROI is the whole cerebral cortex (~1M+ elements),
    # evaluating ROC for every montage is prohibitively slow. A random subsample
    # of 150k is statistically representative (~0.26% sampling error) and ~7.7x faster.
    # Set to 0 to disable (use all elements).
    max_non_roi_elements: int = 150_000

    # Prevent adjacent cap electrodes from being selected together.
    # Two electrodes are considered adjacent when their Euclidean distance is ≤ 1.5×
    # the minimum inter-electrode spacing in the cap. This eliminates montages where
    # current can shunt directly across the scalp between neighbouring electrodes.
    no_adjacent_electrodes: bool = False

    # Per-subgroup non-ROI hard constraints.
    # A montage is rejected if mean TI in ANY listed group exceeds max_mean_V_m,
    # regardless of the overall non-ROI union mean (which stays a single
    # unioned mask — cfg.non_roi — regardless of how many groups are listed here).
    # Each entry is EITHER:
    #   {"name": str, "mask_name": str, "max_mean_V_m": float}
    #     — references an already-generated sub-{id}_label-{mask_name}_mask.nii.gz
    #       (any atlas: Allen, SimNIBS, FreeSurfer, or a combined mask from Mask
    #       Generation — same file convention Allen-sourced ROI/non-ROI already use)
    #   {"name": str, "bna_labels": {str: int}, "max_mean_V_m": float}
    #     — BNA-atlas labels, mapped via the subject's warped
    #       BNA_atlas_subjectspace.nii.gz (the original, still-supported path)
    non_roi_hard_constraint_groups: List[dict] = field(default_factory=list)

    # Per-subgroup ROI hard constraints — the mirror image of
    # non_roi_hard_constraint_groups above. A montage is rejected if mean TI
    # in ANY listed group falls BELOW min_mean_V_m, regardless of the overall
    # ROI mean (cfg.roi) — useful when the ROI is a union of several distinct
    # subregions (e.g. hippocampus + entorhinal cortex) and a montage could
    # otherwise clear the combined-ROI floor while barely touching one of them.
    # Same two entry shapes as non_roi_hard_constraint_groups, just min_mean_V_m
    # instead of max_mean_V_m:
    #   {"name": str, "mask_name": str, "min_mean_V_m": float}
    #   {"name": str, "bna_labels": {str: int}, "min_mean_V_m": float}
    roi_hard_constraint_groups: List[dict] = field(default_factory=list)

    # Hierarchical / coarse-to-fine electrode search (opt-in). When False
    # (default) run_exhaustive_cap_optimization searches all electrodes in one
    # flat pass — identical to the original (pre-hierarchical) behaviour.
    use_hierarchical_search: bool = False

    # Number of refinement ("fine") rounds run after the initial coarse round.
    # 0 means the coarse round's result IS the final result.
    num_fine_iterations: int = 0

    # Nearest-neighbour count per fine iteration — one entry per iteration
    # (length must equal num_fine_iterations). Each round expands every
    # current winning electrode to its N nearest neighbours from the full
    # (dense) electrode set, unions those into the candidate set, and
    # re-searches exhaustively over that (still much smaller than the full set).
    neighbours_per_iteration: List[int] = field(default_factory=list)

    # Stop refining once a fine iteration's score improvement over the
    # previous round is below this fraction (e.g. 0.03 = 3%). Refinement also
    # stops automatically, regardless of this threshold, if an iteration's
    # neighbour-expansion doesn't add any new electrode to the candidate set
    # (further rounds would just repeat the same search).
    early_stop_threshold: float = 0.03

    # ── Amplitude sweep (opt-in) ─────────────────────────────────────────
    # After the electrode-location search (flat or hierarchical) finishes,
    # take its single winning montage and sweep per-channel current
    # amplitudes over a small grid, re-scoring every (I_ch1, I_ch2)
    # combination with a weighted, normalized multi-criteria score to pick
    # the best current allocation. Cheap to run — TI fields scale linearly
    # with current (E(I) = I * E(1A)), so each combination is a re-scale +
    # get_maxTI() + mean/ROC pass, not a new FEM/leadfield lookup.
    #
    # The montage itself is NEVER changed by the sweep — it was originally
    # designed to also pick among the top-N montages (not just currents),
    # but that silently discarded use_composite_location_score's background/
    # hotspot/tier terms: it re-ranked candidates using this sweep's own
    # narrower weight scheme (amplitude_sweep_weights below — plain ROC/
    # ROI-mean/non-ROI-mean/focality-ratio, no background/hotspot/tier at
    # all) and could switch to a montage the outer composite search had NOT
    # actually preferred. Confirmed happening on every run in a real 24-
    # subject batch (sub-41Y01 cohort, 2026-09-14) — composite_location_
    # score.value came out byte-identical to the raw ROC score in 100% of
    # them, meaning background/hotspot/tier weighting had zero effect on
    # the final reported montage in every single case it was enabled
    # alongside the sweep. Fixed by locking the montage to the location
    # search's own composite winner; the sweep now only ever refines ITS
    # currents, never overrides which montage was chosen.
    use_amplitude_sweep: bool = False

    # VESTIGIAL as of the amplitude-sweep rewrite below — kept only so
    # existing config JSON files that still set it don't error out. The
    # sweep used to carry this many of the location-search's top montages
    # forward and could switch to a different one entirely; it now always
    # fixes the location search's own single winning montage and only
    # varies its per-channel current (see use_amplitude_sweep's docstring
    # for why — it was silently discarding the composite location score's
    # background/hotspot/tier terms by re-ranking with a different, narrower
    # weight scheme and picking whichever montage that preferred).
    amplitude_sweep_top_n: int = 5

    # Per-channel current sweep range (mA) — both channels swept
    # independently over this same range, so N grid values -> N x N
    # combinations per montage.
    amplitude_sweep_min_mA:  float = 1.8
    amplitude_sweep_max_mA:  float = 2.0
    amplitude_sweep_step_mA: float = 0.02

    # Hard ceiling on ANY single channel's current (mA) — independent of the
    # sweep range above (defensive: still enforced even if min/max_mA were
    # misconfigured) and independent of electrode.max_total_current (the
    # combined ch1+ch2 ceiling, already existed) — both are enforced. With
    # the defaults (2.0 per pair, 4.0 total from ElectrodeConfig), the total
    # ceiling never actually binds (2.0+2.0=4.0), so only the per-pair
    # ceiling matters unless max_total_current is later lowered.
    amplitude_sweep_max_per_pair_mA: float = 2.0

    # Composite ranking weights (should sum to 1.0) across 4 metrics, each
    # min-max normalized to [0,1] across only the hard-constraint-surviving
    # candidate pool (montage x current-grid combos that got rejected never
    # enter the normalization bounds) — roc (ROC() distance, inverted so
    # higher=better), roi_mean, non_roi_mean (inverted), and focality_ratio
    # (roi_mean / non_roi_mean). ROC dominates by design — it already
    # reflects how many ROI/non-ROI elements satisfy the configured
    # thresholds; the other three add robustness/tie-breaking on top.
    amplitude_sweep_weights: dict = field(default_factory=lambda: {
        "roc": 0.7, "roi_mean": 0.1, "non_roi_mean": 0.1, "focality_ratio": 0.1,
    })

    # ── Composite scoring for the main location search (opt-in) ─────────
    # When False (default) the search's per-montage score is exactly what
    # it always was — -ROC(...) for goal="focality", plain vol_mean for
    # goal="mean" — completely unchanged, fully backward compatible. When
    # True, that raw score becomes one term in a weighted composite that
    # can also penalize background (off-target) field strength and/or
    # electrode "tier" (safety/preference) — see location_score_weights.
    #
    # Deliberately NOT pool-relative-normalized like amplitude_sweep_weights
    # above: the main search streams through up to hundreds of thousands of
    # candidates in O(1) memory (track a running best, never collect a
    # pool), so there's no pool to normalize against without a real
    # architecture change. Weights here are plain absolute coefficients in
    # a linear combination instead (score = w_main*raw_score -
    # w_background*background_mean - w_tier_penalty*tier_cost) — this also
    # means a weight calibrated on one subject/ROI means the same physical
    # thing on another, unlike a pool-relative weight whose effective
    # strength would silently depend on each run's own candidate spread.
    use_composite_location_score: bool = False

    # Minimize field strength across a broad "background" region (e.g.
    # whole-brain GM+WM minus the ROI) — distinct from non_roi/
    # non_roi_hard_constraint_groups, which are hard pass/fail ceilings on
    # specific curated regions. This is a soft, continuous "lower is
    # better" pressure over a much broader area, meant to discourage
    # montages that produce strong off-target field (including directly
    # under the electrodes) even when nothing they hit is explicitly
    # constrained. Only takes effect when use_composite_location_score=True.
    minimize_background_field: bool = False
    # None (default) = genuinely whole-brain: use the full GM+WM element set
    # straight from mesh tissue tags (no NIfTI file needed), minus the ROI.
    # A real name (same mask_path() convention as ROI/non-ROI/constraint
    # groups) opts into a narrower curated region instead of the whole brain.
    background_mask_name: str | None = None

    # Background elements are subsampled (uniform random, same mechanism as
    # max_non_roi_elements) — a whole-brain-minus-ROI mask can easily be
    # 1M+ GM+WM elements, and this term is evaluated on EVERY candidate
    # montage in the search's hot loop, so left at full resolution it would
    # dominate runtime. Kept smaller than max_non_roi_elements by default
    # since it's now a third region evaluated per candidate on top of
    # ROI+non-ROI+any constraint groups.
    max_background_elements: int = 50_000

    # A plain volume-weighted mean over the whole background region washes
    # out a spatially concentrated hotspot (e.g. strong field dumped into
    # frontal cortex) — a small-volume hot region barely moves an average
    # dominated by the much larger rest-of-brain sitting near baseline. This
    # term instead scores "how strong is the field within the hottest
    # (100 - background_hotspot_percentile)% of the sampled background
    # region" — the volume-weighted mean of just the elements at/above that
    # percentile cutoff, evaluated on the same per-candidate background
    # sample as background_mean (cheap: one np.percentile call + a masked
    # weighted mean, same cost class as the existing mean). Default 85.0 =
    # top 15% counted as "the hotspot" — narrowed from an initial 80.0/top-20%
    # after a real two-montage pilot comparison (sub-41Y01 "PL" vs "ML" test
    # montages) showed PL's off-target field problem was concentrated in a
    # genuinely extreme tail (p95->max jumped from 0.59 to 12.9 V/m, a real
    # cliff, not a gradual rise) that a broader top-20% band diluted with
    # only-moderately-elevated elements. Still user's explicit choice not to
    # go as narrow as p95/p99 — see location_score_weights below, calibrated
    # together with this cutoff from the same pilot comparison.
    background_hotspot_percentile: float = 85.0

    # ── Electrode scoring tiers (opt-in) ─────────────────────────────────
    # Per-electrode safety/preference ranking via a 4-LEVEL CUMULATIVE
    # CASCADE (see run_exhaustive_cap_optimization's tier cascade for the
    # exact mechanism):
    #   Level 1 pool = Tier 0 + Tier 1  (Tier 1 soft-penalized, always tried first)
    #   Level 2 pool = Level 1 + Tier 2 (only tried if Level 1 finds nothing
    #                  meeting the configured hard constraint)
    #   Level 3 pool = Level 2 + Tier 3 (only tried if Level 2 also fails)
    #   Level 4 pool = Level 3 + Tier 4 (only tried if Level 3 also fails —
    #                  absolute last resort)
    # Every unlocked tier (1-4) is soft-penalized via electrode_tier_weights
    # below, summed per electrode in the montage — gating decides WHETHER a
    # tier is searched at all; the weight decides which montage wins among
    # those found once it is. Fiducials (Nz/Iz/A1/A2 etc.) are a SEPARATE,
    # permanent exclusion — never enter the pool at any level, no exceptions
    # — see load_electrode_tiers_csv()/DEFAULT_ELECTRODE_TIERS_CSV above.
    # This is all driven by the tiers CSV's own tags, NOT by curating a
    # separate cap CSV (the earlier approach) — the leadfield is still built
    # from whatever the base cap contains (cheap enough not to bother
    # pre-filtering; explicit user choice), only the SEARCH is restricted.
    use_electrode_scoring_tiers: bool = False

    # {electrode_name: 0|1|2|3|4}, matched by name (case-insensitive).
    # Electrodes not present in this dict are treated as Tier 0 (no
    # penalty, always in Level 1's pool). Left empty (the default) while
    # use_electrode_scoring_tiers=True, this is auto-populated at run time
    # from electrode_tiers_csv (or the canonical electrode_tiers_csv_path if
    # that's also unset) — set this dict explicitly yourself only to
    # override/bypass the CSV.
    electrode_tiers: dict = field(default_factory=dict)

    # Explicit override path for the CSV auto-load described above. None
    # (default) falls back to BIDSConfig.electrode_tiers_csv_path — the
    # canonical repo-root reference file, same convention as
    # BNA_subregions.xlsx (one shared file, not something each config
    # re-specifies).
    electrode_tiers_csv: str | None = None

    # Per-tier-point penalty weight, applied PER ELECTRODE in the montage
    # (summed across all 4 — e.g. two Tier-2 electrodes cost 2x this
    # weight, not the max of the two). Tier 0 is always 0, not
    # configurable. Unlike the earlier design, Tier 3 (and now Tier 4) DO
    # have a weight here — needed so that, once a tier is unlocked by the
    # cascade, montages using fewer/cheaper penalized electrodes are still
    # preferred over ones using more/costlier ones (e.g. "3 Tier-0 + 1
    # Tier-2" should beat "2 Tier-0 + 2 Tier-2"). Values chosen to keep a
    # single higher-tier substitution's cost below several lower-tier
    # substitutions' combined cost as a calibration sanity check (e.g.
    # "3" < 4x"2": 0.15 < 0.20) — explicit user-confirmed defaults, still
    # provisional pending further pilot combinations (see
    # TODO_electrode_scoring_tiers.md).
    electrode_tier_weights: dict = field(default_factory=lambda: {
        "1": 0.02, "2": 0.05, "3": 0.15, "4": 0.40,
    })

    # Composite weights (absolute, not pool-normalized — see
    # use_composite_location_score's docstring above). "main" scales the
    # existing raw score; "background"/"background_hotspot"/"tier_penalty"
    # only contribute when minimize_background_field / use_electrode_
    # scoring_tiers are themselves also enabled.
    #
    # "background_hotspot" raised from an initial placeholder of 0.1 to 0.5
    # (4.5x) after a real two-montage pilot comparison (sub-41Y01 "PL" vs
    # "ML"): at 0.1, PL's composite score beat ML's despite PL having a
    # dramatically worse off-target peak (12.9 vs 3.7 V/m) and a
    # consistently worse background tail from p75 up through the max — the
    # main-score term (weight 1.0) simply dominated too easily. At the
    # paired background_hotspot_percentile=85.0, the exact crossover weight
    # (where ML would overtake PL) is ~0.50 — 0.5 is a deliberate, explicit
    # user choice just below that line, so PL still wins this particular
    # comparison under these settings (not every pilot combination needs to
    # flip the outcome). Still provisional — the user explicitly wants to
    # test more candidate weight/percentile combinations before treating
    # this as final (see TODO_electrode_scoring_tiers.md).
    #
    # "tier_penalty" raised from an initial placeholder of 0.1 to 1.0 (10x)
    # after a 3-pair real-montage pilot (sub-41Y01: two Tier-0 baselines
    # each swapped to the closest available Tier-1/Tier-2/both electrode —
    # see run_pipeline.py's _tier_cost() for how electrode_tier_weights
    # combine into a montage's tier_cost). At 0.1, the tier term
    # (0.1*tier_cost, max observed 0.1*0.07=0.007) was consistently smaller
    # than the INCIDENTAL field-geometry change from the swap itself (up to
    # +0.10) — meaning tier never mattered even for a single marginal
    # improvement, let alone a real one. User's explicit intent: only let a
    # worse-tier montage win if the underlying improvement is genuinely
    # substantial, not marginal. tier_penalty=1.0 threads the 3 pilot
    # examples: it blocks both marginal swaps observed (field delta
    # +0.018 and +0.004) while still letting the one substantial swap
    # through (field delta +0.094, net still positive after the penalty).
    # Caveat: the smaller-swap margin at this value is razor-thin (-0.002)
    # — calibrated off one data point, not yet a robust fit. Still
    # provisional, same as background_hotspot above.
    location_score_weights: dict = field(default_factory=lambda: {
        "main": 1.0, "background": 0.2, "background_hotspot": 0.5, "tier_penalty": 1.0,
    })


@dataclass
class SimulationConfig:
    simulate_mode: str = "both"  # "optimized", "mapped", or "both"


@dataclass
class PipelineFlags:
    run_recon_all:    bool = False  # FreeSurfer recon-all (separate container on SCITAS)
    run_charm:        bool = False  # SimNIBS charm head modeling
    run_roi_masks:    bool = True   # Section 1: create ROI/non-ROI NIfTI masks
    run_optimization: bool = True   # Section 2: exhaustive cap search (see run_exhaustive_cap_optimization)
    run_simulation:   bool = False  # Section 3: FEM simulations (redundant — optimizer already runs FEM internally)
    run_analysis:     bool = True   # Section 4: TI field analysis
    run_visualization:bool = True   # save TI_comparison.msh + 4-view images


# ── Main config ────────────────────────────────────────────────────────────────

@dataclass
class PipelineConfig:
    # ── Identity ──────────────────────────────────────────────────────────────
    subject_id:  str = "025"
    project_dir: str = "/mnt/BIDS_TI_Toolbox"

    # ── Pipeline control ──────────────────────────────────────────────────────
    flags:     PipelineFlags  = field(default_factory=PipelineFlags)

    # ── ROI / non-ROI ─────────────────────────────────────────────────────────
    roi:        ROIConfig           = field(default_factory=ROIConfig)
    non_roi:    Optional[ROIConfig] = None   # set to None to skip non-ROI
    extra_rois: List[ROIConfig]     = field(default_factory=list)  # additional regions to analyze (no optimization)

    # ── Electrode ─────────────────────────────────────────────────────────────
    electrode: ElectrodeConfig = field(default_factory=ElectrodeConfig)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)

    # ── Simulation ────────────────────────────────────────────────────────────
    simulation: SimulationConfig = field(default_factory=SimulationConfig)

    # ── recon-all (only used if flags.run_recon_all=True) ─────────────────────
    recon_all: ReconAllConfig = field(default_factory=ReconAllConfig)

    # ── EEG cap ───────────────────────────────────────────────────────────────
    # Path to the registered cap CSV in subject space.
    # If None, defaults to the Okamoto 10-20 cap registered by charm.
    # Set to m2m_{id}/eeg_positions/{cap_name}.csv after running register_caps.py.
    cap_csv: Optional[str] = None

    # Path to BN_Atlas_246_1mm.nii.gz (MNI space).
    # Required when any roi / non_roi / extra_rois uses bna_labels.
    # Can be set to the repo copy: "/mnt/BIDS_TI_Toolbox/code/resources/atlases/BN_Atlas_246_1mm.nii.gz"
    bna_atlas_path: Optional[str] = None

    # ── Derived paths (read-only properties) ──────────────────────────────────

    @property
    def sim_sub_dir(self) -> str:
        return f"{self.project_dir}/derivatives/SimNIBS/sub-{self.subject_id}"

    @property
    def m2m_path(self) -> str:
        return f"{self.sim_sub_dir}/m2m_{self.subject_id}"

    @property
    def fs_dir(self) -> str:
        return f"{self.project_dir}/derivatives/freesurfer/sub-{self.subject_id}"

    @property
    def roi_dir(self) -> str:
        return f"{self.sim_sub_dir}/roi"

    @property
    def ti_opt_dir(self) -> str:
        return f"{self.sim_sub_dir}/TIoptimization"

    @property
    def electrode_tiers_csv_path(self) -> str:
        """Canonical tiers reference (see DEFAULT_ELECTRODE_TIERS_CSV_PATH) —
        lives next to config.py in code/pipeline/, not under project_dir, so
        it's included whenever just the code/ repo is shared."""
        return DEFAULT_ELECTRODE_TIERS_CSV_PATH

    def mask_path(self, label: str) -> str:
        """BIDS-compliant mask path: sub-{id}_label-{label}_mask.nii.gz"""
        return f"{self.roi_dir}/sub-{self.subject_id}_label-{label}_mask.nii.gz"

    @property
    def t1_path(self) -> str:
        return (f"{self.project_dir}/rawdata/sub-{self.subject_id}"
                f"/anat/sub-{self.subject_id}_T1w.nii.gz")

    @property
    def t2_path(self) -> str:
        return (f"{self.project_dir}/rawdata/sub-{self.subject_id}"
                f"/anat/sub-{self.subject_id}_T2w.nii.gz")

    @property
    def aseg_path(self) -> Optional[str]:
        """Find best available FreeSurfer segmentation, preferring aparc+aseg."""
        candidates = [
            f"{self.m2m_path}/fs_{self.subject_id}/mri/aparc+aseg.mgz",
            f"{self.fs_dir}/mri/aparc+aseg.mgz",
            f"{self.m2m_path}/fs_{self.subject_id}/mri/aseg.auto.mgz",
            f"{self.fs_dir}/mri/aseg.auto.mgz",
            # charm atlas segmentation (NIfTI) — same FreeSurfer label numbers,
            # produced when charm runs without a full FreeSurfer install
            f"{self.m2m_path}/segmentation/labeling.nii.gz",
        ]
        for p in candidates:
            if os.path.isfile(p):
                return p
        return None

    @property
    def eeg_csv_path(self) -> str:
        if self.cap_csv:
            return self.cap_csv
        return f"{self.m2m_path}/eeg_positions/EEG10-20_Okamoto_2004.csv"


# ── JSON serialization ─────────────────────────────────────────────────────────

def _from_dict(cls, d: dict):
    """Recursively construct a dataclass from a (partial) dict."""
    import dataclasses
    if not dataclasses.is_dataclass(cls):
        return d
    kwargs = {}
    hints = {f.name: f for f in dataclasses.fields(cls)}
    for f in dataclasses.fields(cls):
        if f.name not in d:
            continue
        val = d[f.name]
        origin = getattr(f.type, "__origin__", None)
        # handle Optional[SomeDataclass]
        if val is None:
            kwargs[f.name] = None
        elif dataclasses.is_dataclass(f.type):
            kwargs[f.name] = _from_dict(f.type, val)
        else:
            kwargs[f.name] = val
    # start from defaults, then apply overrides
    base = cls()
    for k, v in kwargs.items():
        object.__setattr__(base, k, v)
    return base


def load_config(json_path: str) -> PipelineConfig:
    """Load a PipelineConfig from a JSON override file."""
    with open(json_path) as f:
        d = json.load(f)

    cfg = PipelineConfig()

    # top-level scalars
    for key in ("subject_id", "project_dir", "cap_csv", "bna_atlas_path"):
        if key in d:
            object.__setattr__(cfg, key, d[key])

    # nested dataclasses
    nested = {
        "flags":      PipelineFlags,
        "roi":        ROIConfig,
        "electrode":  ElectrodeConfig,
        "optimizer":  OptimizerConfig,
        "simulation": SimulationConfig,
        "recon_all":  ReconAllConfig,
    }
    for key, cls in nested.items():
        if key in d:
            object.__setattr__(cfg, key, _from_dict(cls, d[key]))

    # non_roi is optional
    if "non_roi" in d:
        val = d["non_roi"]
        object.__setattr__(cfg, "non_roi", _from_dict(ROIConfig, val) if val else None)

    # extra_rois is a list of ROIConfig
    if "extra_rois" in d:
        object.__setattr__(cfg, "extra_rois",
                           [_from_dict(ROIConfig, r) for r in d["extra_rois"]])

    return cfg


def save_config(cfg: PipelineConfig, json_path: str) -> None:
    """Save current config to JSON (useful for reproducibility logging)."""
    import dataclasses
    d = {
        "subject_id":     cfg.subject_id,
        "project_dir":    cfg.project_dir,
        "cap_csv":        cfg.cap_csv,
        "bna_atlas_path": cfg.bna_atlas_path,
        "flags":       asdict(cfg.flags),
        "roi":         asdict(cfg.roi),
        "non_roi":     asdict(cfg.non_roi) if cfg.non_roi else None,
        "extra_rois":  [asdict(r) for r in cfg.extra_rois],
        "electrode":   asdict(cfg.electrode),
        "optimizer":   asdict(cfg.optimizer),
        "simulation":  asdict(cfg.simulation),
        "recon_all":   asdict(cfg.recon_all),
    }
    with open(json_path, "w") as f:
        json.dump(d, f, indent=2)

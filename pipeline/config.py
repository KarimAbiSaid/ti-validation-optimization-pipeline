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
    # regardless of the overall non-ROI union mean.
    # Each entry: {"name": str, "bna_labels": {str: int}, "max_mean_V_m": float}
    non_roi_hard_constraint_groups: List[dict] = field(default_factory=list)

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

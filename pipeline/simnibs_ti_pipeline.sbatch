#!/bin/bash
# ==============================================================================
# SimNIBS TI Pipeline — single subject (Apptainer / SLURM)
# ==============================================================================
#
# HOW TO USE
# ----------
# 1. Edit the CONFIG line below (or pass PIPELINE_CONFIG via environment).
# 2. Create the log directory once:
#       mkdir -p /scratch/$USER/logs/simnibs
# 3. Submit:
#       sbatch simnibs_ti_pipeline.sbatch
#
# For parallel multi-ROI submission, use submit_multiroi_scitas.sh instead.
# ==============================================================================


# ==============================================================================
# SLURM RESOURCE REQUESTS
# ------------------------------------------------------------------------------
# --cpus-per-task : must match optimizer.cpus in your JSON config (default: 8)
# --mem           : 32G recommended — the post-optimization FEM on the full
#                   head mesh is memory-heavy and will crash at 32G
# --time          : boundary mode (200 iter × 4 starts) needs up to 48h;
#                   exhaustive search needs ~4h (leadfield + algebraic search)
# --account       : always uphummel for this lab
# ==============================================================================
#SBATCH --job-name=simnibs_ti
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --account=uphummel
#SBATCH --output=/scratch/%u/logs/simnibs/simnibs_%j.out
#SBATCH --error=/scratch/%u/logs/simnibs/simnibs_%j.err


# ==============================================================================
# EDIT THESE
# ==============================================================================

# Path to the JSON config file for this run.
# If PIPELINE_CONFIG is set in the environment (e.g. via submit_multiroi_scitas.sh),
# that value takes precedence — no need to edit this file for batch runs.
#
# Available configs:
#   --- Boundary mode (DE optimizer, full scalp region restricted to EEG cap area) ---
#   configs/sub-025_roi-striatum_nonroi-M1_boundary.json
#   configs/sub-025_roi-hippocampus_nonroi-M1_boundary.json
#
#   --- Exhaustive search (leadfield-based, guaranteed global optimum over cap positions) ---
#   configs/sub-025_roi-striatum_exhaustive.json
#
#   --- Other ROIs (full scalp, no boundary filter) ---
#   configs/sub-025_roi-STN_nonroi-M1.json
#   configs/sub-025_roi-globuspallidus_nonroi-M1.json
#   configs/sub-025_roi-NAcc_nonroi-M1.json
#   configs/sub-025_roi-hippocampus_nonroi-M1.json
#
#   --- Other subjects ---
#   configs/sub-41Y01_roi-striatum.json
# Accepts a single config (PIPELINE_CONFIG) or a space-separated list
# (PIPELINE_CONFIGS).  The list form is used by submit_multiroi_scitas.sh to
# run all ROIs for one subject in a single job — the leadfield is then
# computed once on the first config and reused for all subsequent ones.
CONFIGS=${PIPELINE_CONFIGS:-${PIPELINE_CONFIG:-/mnt/BIDS_TI_Toolbox/code/pipeline/configs/sub-025_roi-hippocampus_ex_mean.json}}

# Extra arguments passed to run_pipeline.py.
# If PIPELINE_EXTRA_ARGS is set in the environment, that takes precedence.
# Leave empty ("") for a normal run that skips already-completed sections.
#
# Examples:
#   --force optimization          re-run optimizer even if output exists
#   --force analysis              re-run TI field analysis
#   --force optimization analysis force multiple sections
#   --set optimizer.cpus=8        override a single config value
#   --set optimizer.max_iterations=300
#   --set optimizer.n_multistart=4
EXTRA_ARGS=${PIPELINE_EXTRA_ARGS:-""}

# Path to the SimNIBS Apptainer image.
IMG=/work/uphummel/shared/software/containers/simnibs/simnibs_v2.3.0.sif


# ==============================================================================
# DO NOT EDIT BELOW THIS LINE
# ==============================================================================

set -euo pipefail

THREADS="${SLURM_CPUS_PER_TASK}"
SCRATCH=/scratch/$USER/BIDS_TI_Toolbox

echo "============================================================"
echo " SimNIBS TI Pipeline"
echo " Job ID   : $SLURM_JOB_ID"
echo " Node     : $SLURMD_NODENAME"
echo " Configs  : $CONFIGS"
echo " Threads  : $THREADS"
echo " Extra    : ${EXTRA_ARGS:-<none>}"
echo "============================================================"

[[ -f "$IMG" ]] || { echo "ERROR: Apptainer image not found: $IMG"; exit 1; }

apptainer exec --cleanenv \
  --env OMP_NUM_THREADS="$THREADS" \
  --env MKL_NUM_THREADS="$THREADS" \
  --env OPENBLAS_NUM_THREADS="$THREADS" \
  --env PYTHONUNBUFFERED=1 \
  --bind "${SCRATCH}":/mnt/BIDS_TI_Toolbox \
  "$IMG" \
  /root/SimNIBS-4.6/simnibs_env/bin/python -u \
  /mnt/BIDS_TI_Toolbox/code/pipeline/run_pipeline.py \
  --config $CONFIGS \
  --job-id "$SLURM_JOB_ID" \
  ${EXTRA_ARGS}

echo "=== Done: $SLURM_JOB_ID ==="

#!/bin/bash
# ==============================================================================
# generate_leadfield_scitas.sbatch — one leadfield for a given (subject, cap,
# electrode geometry), submitted directly by the GUI's Leadfield Generation
# page (fem_discovery.run_leadfield_on_scitas).
#
# Manual single-job submission:
#   sbatch --export=ALL,LF_SUBJECT=41Y07,LF_CAP=biosemi32,LF_SHAPE=ellipse,\
#LF_DIM1=19.5,LF_DIM2=19.5,LF_GEL=1.0 generate_leadfield_scitas.sbatch
#
# Prerequisite: m2m_{id}/ and m2m_{id}/eeg_positions/{LF_CAP}.csv must
# already be synced up (the GUI does this before submitting).
#
# Input  : derivatives/SimNIBS/sub-{id}/m2m_{id}/eeg_positions/{cap}.csv
# Output : derivatives/SimNIBS/sub-{id}/leadfield_volume/{tag}/
#            {id}_leadfield_{cap}.hdf5 + _params.json
# ==============================================================================

#SBATCH --job-name=leadfield
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --account=uphummel
#SBATCH --output=/scratch/%u/logs/simnibs/leadfield_%x_%j.out
#SBATCH --error=/scratch/%u/logs/simnibs/leadfield_%x_%j.err

set -euo pipefail

SUBJECT_ID="${LF_SUBJECT:?ERROR: LF_SUBJECT env var not set}"
CAP_NAME="${LF_CAP:?ERROR: LF_CAP env var not set}"
SHAPE="${LF_SHAPE:-ellipse}"
DIM1="${LF_DIM1:-19.5}"
DIM2="${LF_DIM2:-19.5}"
GEL="${LF_GEL:-1.0}"

SCRATCH="/scratch/${USER}/BIDS_TI_Toolbox"
IMG="/work/uphummel/shared/software/containers/simnibs/simnibs_v2.3.0.sif"
MNT="/mnt/BIDS_TI_Toolbox"
THREADS="${SLURM_CPUS_PER_TASK}"

CAP_CSV="${SCRATCH}/derivatives/SimNIBS/sub-${SUBJECT_ID}/m2m_${SUBJECT_ID}/eeg_positions/${CAP_NAME}.csv"

echo "============================================================"
echo " Leadfield generation"
echo " Job ID   : ${SLURM_JOB_ID}"
echo " Node     : ${SLURMD_NODENAME}"
echo " Subject  : sub-${SUBJECT_ID}"
echo " Cap      : ${CAP_NAME}"
echo " Electrode: ${SHAPE} ${DIM1}x${DIM2}mm, gel ${GEL}mm"
echo " Threads  : ${THREADS}"
echo "============================================================"

[[ -f "$IMG" ]] || { echo "ERROR: Apptainer image not found: $IMG"; exit 1; }
[[ -f "$CAP_CSV" ]] || { echo "ERROR: cap CSV not found (sync it up first): $CAP_CSV"; exit 1; }

apptainer exec --cleanenv \
  --env OMP_NUM_THREADS="$THREADS" \
  --env MKL_NUM_THREADS="$THREADS" \
  --env OPENBLAS_NUM_THREADS="$THREADS" \
  --env PYTHONUNBUFFERED=1 \
  --bind "${SCRATCH}":"${MNT}" \
  "$IMG" \
  /root/SimNIBS-4.6/simnibs_env/bin/python -u \
  /mnt/BIDS_TI_Toolbox/code/pipeline/generate_leadfield_cli.py \
  --subject "${SUBJECT_ID}" \
  --cap-name "${CAP_NAME}" \
  --shape "${SHAPE}" \
  --dimensions "${DIM1}" "${DIM2}" \
  --gel-thickness "${GEL}" \
  --project-dir "${MNT}" \
  --cpus "${THREADS}"

echo "=== Done: sub-${SUBJECT_ID} ${CAP_NAME} -- ${SLURM_JOB_ID} ==="
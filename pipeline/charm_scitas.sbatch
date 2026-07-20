#!/bin/bash
# ==============================================================================
# charm_scitas.sbatch — Run SimNIBS charm head modelling for one subject
# ==============================================================================
#
# Do NOT submit this directly. Use submit_charm_scitas.sh, which sets
# CHARM_SUBJECT and submits one job per new subject automatically.
#
# Manual single-subject submission:
#   sbatch --export=ALL,CHARM_SUBJECT=002 charm_scitas.sbatch
#
# Input  : rawdata/sub-{id}/anat/sub-{id}_T1w.nii.gz  (required)
#          rawdata/sub-{id}/anat/sub-{id}_T2w.nii.gz  (optional — improves quality)
# Output : derivatives/SimNIBS/sub-{id}/m2m_{id}/
#          derivatives/SimNIBS/sub-{id}/m2m_{id}/{id}.msh  (completion marker)
#
# TIME   : ~30-60 min with T1+T2, ~45-90 min T1-only
# ==============================================================================

#SBATCH --job-name=charm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=03:30:00
#SBATCH --account=uphummel
#SBATCH --output=/scratch/%u/logs/simnibs/charm_%x_%j.out
#SBATCH --error=/scratch/%u/logs/simnibs/charm_%x_%j.err

set -euo pipefail

# Subject ID comes from the submission environment
SUBJECT_ID="${CHARM_SUBJECT:?ERROR: CHARM_SUBJECT env var not set}"

SCRATCH="/scratch/${USER}/BIDS_TI_Toolbox"
IMG="/work/uphummel/shared/software/containers/simnibs/simnibs_v2.3.0.sif"

MNT="/mnt/BIDS_TI_Toolbox"
SUB_DIR="${MNT}/derivatives/SimNIBS/sub-${SUBJECT_ID}"
T1="${MNT}/rawdata/sub-${SUBJECT_ID}/anat/sub-${SUBJECT_ID}_T1w.nii.gz"
T2="${MNT}/rawdata/sub-${SUBJECT_ID}/anat/sub-${SUBJECT_ID}_T2w.nii.gz"
MESH="${SCRATCH}/derivatives/SimNIBS/sub-${SUBJECT_ID}/m2m_${SUBJECT_ID}/${SUBJECT_ID}.msh"

echo "============================================================"
echo " SimNIBS charm — head modelling"
echo " Job ID    : ${SLURM_JOB_ID}"
echo " Node      : ${SLURMD_NODENAME}"
echo " Subject   : sub-${SUBJECT_ID}"
echo " T1        : rawdata/sub-${SUBJECT_ID}/anat/sub-${SUBJECT_ID}_T1w.nii.gz"
echo "============================================================"

# Skip if already completed
if [[ -f "$MESH" ]]; then
    echo "  [skip] charm already complete: m2m_${SUBJECT_ID}/${SUBJECT_ID}.msh exists"
    exit 0
fi

# Clean up a partial m2m_ folder from a previous failed run so charm doesn't
# refuse with "use --forcerun to overwrite existing m2m_{subID} folder"
M2M_DIR="${SCRATCH}/derivatives/SimNIBS/sub-${SUBJECT_ID}/m2m_${SUBJECT_ID}"
if [[ -d "$M2M_DIR" ]]; then
    echo "  [warn] partial m2m_ folder found (previous failed run) — removing it"
    rm -rf "$M2M_DIR"
fi

[[ -f "$IMG" ]] || { echo "ERROR: Apptainer image not found: $IMG"; exit 1; }
[[ -f "${SCRATCH}/rawdata/sub-${SUBJECT_ID}/anat/sub-${SUBJECT_ID}_T1w.nii.gz" ]] \
    || { echo "ERROR: T1w not found for sub-${SUBJECT_ID}"; exit 1; }

# Check T2 availability on the host, pass it to charm if present
if [[ -f "${SCRATCH}/rawdata/sub-${SUBJECT_ID}/anat/sub-${SUBJECT_ID}_T2w.nii.gz" ]]; then
    T2_ARG="$T2"
    echo " T2        : rawdata/sub-${SUBJECT_ID}/anat/sub-${SUBJECT_ID}_T2w.nii.gz"
else
    T2_ARG=""
    echo " T2        : not found — running T1-only (lower segmentation quality)"
fi
echo "============================================================"

apptainer exec --cleanenv \
    --bind "${SCRATCH}":"${MNT}" \
    "$IMG" \
    /bin/bash -c "
        mkdir -p ${SUB_DIR} && \
        cd ${SUB_DIR} && \
        charm --forceqform ${SUBJECT_ID} ${T1} ${T2_ARG}
    "

# Verify output
if [[ -f "$MESH" ]]; then
    echo ""
    echo "=== charm complete: sub-${SUBJECT_ID} — ${SLURM_JOB_ID} ==="
else
    echo "ERROR: charm finished but mesh not found at:"
    echo "  ${MESH}"
    exit 1
fi

#!/bin/bash
# ==============================================================================
# submit_charm_scitas.sh — Submit one charm job per new subject
# ==============================================================================
#
# USAGE (from SCITAS login node)
# -----
#   cd /scratch/$USER/BIDS_TI_Toolbox/code/pipeline
#   bash submit_charm_scitas.sh
#
# Add new subjects to NEW_SUBJECTS below (keep in sync with IXI_IDS in
# download_ixi.sh and SUBJECTS in generate_configs.py).
#
# Already-processed subjects (025, 41Y01) are intentionally excluded —
# charm only needs to run once per subject.
#
# Subjects with an existing m2m_{id}/{id}.msh are skipped automatically.
#
# After all charm jobs finish, generate configs and submit the pipeline:
#   python generate_configs.py          # add new subjects to SUBJECTS first
#   bash submit_multiroi_scitas.sh
# ==============================================================================

set -euo pipefail

# ==============================================================================
# EDIT THESE — must match IXI_IDS in download_ixi.sh
# ==============================================================================
NEW_SUBJECTS=(002 012 035 055 066 077 086 093 73T01 73T02 73T03 73T04 73T05 73T06 73T07 73T09 73T10 73T11 73T12 73T14 52Y02 52Y03 52Y04 52Y05 52Y07 52Y08 52Y09 52Y10 52Y12 52Y13 52Y16)
# ==============================================================================
# DO NOT EDIT BELOW
# ==============================================================================

SBATCH_SCRIPT="$(cd "$(dirname "$0")" && pwd)/charm_scitas.sbatch"
SCRATCH="/scratch/${USER}/BIDS_TI_Toolbox"
LOG_DIR="/scratch/${USER}/logs/simnibs"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " Submitting charm jobs — $(date)"
echo " Subjects : ${NEW_SUBJECTS[*]}"
echo " Account  : uphummel"
echo " CPUs     : 4 per job  |  Mem: 16G  |  Time: 05:30:00"
echo "============================================================"

submitted=0
skipped=0

for id in "${NEW_SUBJECTS[@]}"; do
    mesh="${SCRATCH}/derivatives/SimNIBS/sub-${id}/m2m_${id}/${id}.msh"

    if [[ -f "$mesh" ]]; then
        echo ""
        echo "  [skip] sub-${id}  (m2m_${id}/${id}.msh already exists)"
        (( skipped++ )) || true
        continue
    fi

    job_id=$(sbatch \
        --job-name="charm_${id}" \
        --export=ALL,CHARM_SUBJECT="${id}" \
        "$SBATCH_SCRIPT" \
        | awk '{print $NF}')

    echo ""
    echo "  Submitted sub-${id}: job ${job_id}"
    echo "    log : ${LOG_DIR}/charm_charm_${id}_${job_id}.out"
    (( submitted++ )) || true
done

echo ""
echo "============================================================"
echo " Submitted: ${submitted}  |  Skipped (done): ${skipped}"
echo ""
echo " Monitor:"
echo "   squeue -u \$USER"
echo "   tail -f ${LOG_DIR}/charm_charm_<id>_<jobid>.out"
echo ""
echo " When all charm jobs finish, run:"
echo "   python generate_configs.py   # if SUBJECTS not yet updated"
echo "   bash submit_multiroi_scitas.sh"
echo "============================================================"

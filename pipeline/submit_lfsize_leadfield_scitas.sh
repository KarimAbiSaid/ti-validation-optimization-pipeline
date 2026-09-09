#!/bin/bash
# ==============================================================================
# submit_lfsize_leadfield_scitas.sh — Submit one leadfield job per
# (subject, cap size) for the leadfield-SIZE experiment
# ==============================================================================
#
# USAGE (from SCITAS login node)
# -----
#   cd /scratch/$USER/BIDS_TI_Toolbox/code/pipeline
#   bash submit_lfsize_leadfield_scitas.sh
#
# Prerequisite: sync each subject's m2m_{id}/eeg_positions/lfsize_{N}.csv up
# from the local run_leadfield_size_experiment.py output before running this.
#
# Jobs with an already-valid cached leadfield (matching the current geometry
# fingerprint) are skipped automatically by generate_lfsize_leadfield.py
# itself -- safe to re-run this after syncing new subjects'/sizes' CSVs.
# ==============================================================================

set -euo pipefail

# ==============================================================================
# EDIT THESE if the experiment's subjects/sizes change
# ==============================================================================
SUBJECTS=(41Y07 41Y08)
SIZES=(5 10 20 40 50 64)
# ==============================================================================
# DO NOT EDIT BELOW
# ==============================================================================

SBATCH_SCRIPT="$(cd "$(dirname "$0")" && pwd)/lfsize_leadfield_scitas.sbatch"
SCRATCH="/scratch/${USER}/BIDS_TI_Toolbox"
LOG_DIR="/scratch/${USER}/logs/simnibs"

mkdir -p "$LOG_DIR"

echo "============================================================"
echo " Submitting leadfield-size experiment jobs — $(date)"
echo " Subjects : ${SUBJECTS[*]}"
echo " Sizes    : ${SIZES[*]}"
echo " Account  : uphummel"
echo " CPUs     : 8 per job  |  Mem: 32G  |  Time: 04:00:00"
echo "============================================================"

submitted=0
skipped=0

for id in "${SUBJECTS[@]}"; do
    for n in "${SIZES[@]}"; do
        cap_name="lfsize_${n}"
        cap_csv="${SCRATCH}/derivatives/SimNIBS/sub-${id}/m2m_${id}/eeg_positions/${cap_name}.csv"

        if [[ ! -f "$cap_csv" ]]; then
            echo ""
            echo "  [skip] sub-${id} ${cap_name}  (CSV not synced up yet)"
            (( skipped++ )) || true
            continue
        fi

        job_id=$(sbatch \
            --job-name="lfsize_${id}_${n}" \
            --export=ALL,LFSIZE_SUBJECT="${id}",LFSIZE_CAP="${cap_name}" \
            "$SBATCH_SCRIPT" \
            | awk '{print $NF}')

        echo ""
        echo "  Submitted sub-${id} ${cap_name}: job ${job_id}"
        echo "    log : ${LOG_DIR}/lfsize_lf_lfsize_${id}_${n}_${job_id}.out"
        (( submitted++ )) || true
    done
done

echo ""
echo "============================================================"
echo " Submitted: ${submitted}  |  Skipped (CSV not synced): ${skipped}"
echo ""
echo " Monitor:"
echo "   squeue -u \$USER"
echo "   tail -f ${LOG_DIR}/lfsize_lf_lfsize_<id>_<n>_<jobid>.out"
echo ""
echo " When all jobs finish, sync leadfield_volume/lfsize_{N}__ellipse_19.5x19.5_1mm/"
echo " back down to the local D: drive for each subject."
echo "============================================================"

#!/bin/bash
# ==============================================================================
# submit_recon_all_scitas.sh — Submit recon-all for subjects missing aparc+aseg
# ==============================================================================
#
# Submits one job per subject. Skips any subject that already has
# recon-all.done in derivatives/freesurfer/sub-{id}/scripts/.
#
# USAGE (from SCITAS login node):
#   cd /scratch/$USER/BIDS_TI_Toolbox/code/pipeline
#   bash submit_recon_all_scitas.sh              # all 8 subjects
#   bash submit_recon_all_scitas.sh 002 012      # specific subjects only
# ==============================================================================

set -euo pipefail
# pipefail is needed to catch errors in sbatch command, 
# which is piped to awk, where awk is the last command 
# in the pipe and would otherwise hide errors in sbatch

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_SCRIPT="${SCRIPT_DIR}/recon_all_scitas.sbatch"
SCRATCH="/scratch/${USER}/BIDS_TI_Toolbox"
LOG_DIR="/scratch/${USER}/jobs/logs"

mkdir -p "$LOG_DIR"

# Subjects that need recon-all (025 and 41Y01 already have it)
ALL_SUBJECTS=(002 012 035 055 066 077 086 093)

if [[ $# -gt 0 ]]; then
    SUBJECTS=("$@")
else
    SUBJECTS=("${ALL_SUBJECTS[@]}")
fi

echo "============================================================"
echo " Submitting recon-all — $(date)"
echo " Subjects : ${SUBJECTS[*]}"
echo "============================================================"

for subj in "${SUBJECTS[@]}"; do
    done_file="${SCRATCH}/derivatives/freesurfer/sub-${subj}/scripts/recon-all.done"
    if [[ -f "$done_file" ]]; then
        echo "  [skip] sub-${subj} — recon-all.done already exists"
        continue
    fi

    job_id=$(sbatch \
        --job-name="reconall_${subj}" \
        --export=ALL,RECONALL_SUBJECT="${subj}" \
        "$SBATCH_SCRIPT" \
        | awk '{print $NF}')

    echo "  Submitted sub-${subj}: job ${job_id}"
    echo "    log: ${LOG_DIR}/reconall_reconall_${subj}_${job_id}.out"
done

echo ""
echo "============================================================"
echo " Monitor: squeue -u \$USER"
echo " After completion, re-run roi_masks to pick up M1 labels:"
echo "   bash submit_multiroi_scitas.sh --goal focality --time 07:00:00"
echo "============================================================"

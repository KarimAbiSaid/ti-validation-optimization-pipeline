#!/bin/bash
# ==============================================================================
# recon_all_scitas.sbatch — FreeSurfer recon-all for one subject
# ==============================================================================
#
# Do NOT submit directly — use submit_recon_all_scitas.sh.
#
# Manual single-subject:
#   sbatch --export=ALL,RECONALL_SUBJECT=002 recon_all_scitas.sbatch
#
# Input : rawdata/sub-{id}/anat/sub-{id}_T1w.nii.gz  (required)
#         rawdata/sub-{id}/anat/sub-{id}_T2w.nii.gz  (optional — improves pial)
# Output: derivatives/freesurfer/sub-{id}/mri/aparc+aseg.mgz
# Time  : ~4 h typical
#
# WHY recon-all?
#   charm-only gives aseg.auto.mgz (no cortical parcellation). recon-all adds
#   aparc+aseg.mgz with real M1 labels (1024/2024) so the focality non-ROI
#   uses ~50k M1 elements instead of the ~1M whole-cortex fallback.
# ==============================================================================

#SBATCH --job-name=reconall
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=40G
#SBATCH --time=08:00:00
#SBATCH --account=uphummel
#SBATCH --output=/scratch/%u/jobs/logs/reconall_%x_%j.out
#SBATCH --error=/scratch/%u/jobs/logs/reconall_%x_%j.err

set -euo pipefail

SUBJECT_ID="${RECONALL_SUBJECT:?ERROR: RECONALL_SUBJECT env var not set}"

SCRATCH="/scratch/${USER}/BIDS_TI_Toolbox"
IMG="/work/uphummel/shared/software/containers/freesurfer/freesurfer_7.3.2.sif"
LICENSE="/work/uphummel/shared/software/containers/freesurfer/FS_LICENSE/license.txt"

T1="${SCRATCH}/rawdata/sub-${SUBJECT_ID}/anat/sub-${SUBJECT_ID}_T1w.nii.gz"
T2="${SCRATCH}/rawdata/sub-${SUBJECT_ID}/anat/sub-${SUBJECT_ID}_T2w.nii.gz"
WORK_DIR="/scratch/${USER}/jobs/freesurfer/${SLURM_JOB_ID}"
OUTPUT_DIR="${SCRATCH}/derivatives/freesurfer"
DONE="${OUTPUT_DIR}/sub-${SUBJECT_ID}/scripts/recon-all.done"

echo "============================================================"
echo " FreeSurfer recon-all"
echo " Job ID  : ${SLURM_JOB_ID}"
echo " Node    : ${SLURMD_NODENAME}"
echo " Subject : sub-${SUBJECT_ID}"
echo "============================================================"

[[ -f "$IMG" ]]     || { echo "ERROR: Container not found: $IMG";     exit 1; }
[[ -f "$LICENSE" ]] || { echo "ERROR: License not found: $LICENSE";   exit 1; }
[[ -f "$T1" ]]      || { echo "ERROR: T1w not found: $T1";            exit 1; }

if [[ -f "$DONE" ]]; then
    echo "  [skip] recon-all.done already exists — skipping"
    exit 0
fi

T2_ARGS=""
if [[ -f "$T2" ]]; then
    T2_ARGS="-T2 /opt/in/sub-${SUBJECT_ID}_T2w.nii.gz -T2pial"
    echo " T2      : found — enabling -T2pial"
else
    echo " T2      : not found — running T1-only"
fi
echo "============================================================"

mkdir -p "${WORK_DIR}" "${OUTPUT_DIR}"

# Bind mounts inside container:
#   /opt/license.txt  ← FreeSurfer license (read-only)
#   /opt/in/          ← subject anat folder (read-only)
#   /opt/subjects/    ← recon-all working dir (writable scratch)
apptainer exec --cleanenv \
    --env OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}" \
    --env FS_LICENSE=/opt/license.txt \
    --bind "${LICENSE}":/opt/license.txt:ro \
    --bind "${SCRATCH}/rawdata/sub-${SUBJECT_ID}/anat":/opt/in:ro \
    --bind "${WORK_DIR}":/opt/subjects \
    "$IMG" \
    recon-all \
        -subject "sub-${SUBJECT_ID}" \
        -i /opt/in/"sub-${SUBJECT_ID}_T1w.nii.gz" \
        -all \
        -sd /opt/subjects \
        -parallel -openmp "${SLURM_CPUS_PER_TASK}" \
        ${T2_ARGS}

# Copy from scratch working dir to persistent output
echo "  Copying results to ${OUTPUT_DIR} ..."
rsync -a "${WORK_DIR}/sub-${SUBJECT_ID}/" "${OUTPUT_DIR}/sub-${SUBJECT_ID}/"

if [[ -f "$DONE" ]]; then
    echo ""
    echo "=== recon-all complete: sub-${SUBJECT_ID} — ${SLURM_JOB_ID} ==="
else
    echo "ERROR: recon-all finished but recon-all.done not found"
    exit 1
fi

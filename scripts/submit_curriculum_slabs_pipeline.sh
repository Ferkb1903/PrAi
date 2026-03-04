#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/lustre/home/acastaneda/Fernando/PrAi}"

cd "$PROJECT_ROOT"
mkdir -p logs

GEN_JOB_ID=$(sbatch --parsable scripts/submit_generate_slab_curriculum.slurm)
echo "Submitted generate job: $GEN_JOB_ID"

TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:${GEN_JOB_ID} scripts/submit_train_curriculum_slabs.slurm)
echo "Submitted train job: $TRAIN_JOB_ID (afterok:$GEN_JOB_ID)"

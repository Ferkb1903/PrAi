#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/lustre/home/acastaneda/Fernando/PrAi}"
GEN_SBATCH_ARGS="${GEN_SBATCH_ARGS:-}"
TRAIN_SBATCH_ARGS="${TRAIN_SBATCH_ARGS:-}"

cd "$PROJECT_ROOT"
mkdir -p logs

gen_args=()
if [[ -n "$GEN_SBATCH_ARGS" ]]; then
	read -r -a gen_args <<< "$GEN_SBATCH_ARGS"
fi

train_args=()
if [[ -n "$TRAIN_SBATCH_ARGS" ]]; then
	read -r -a train_args <<< "$TRAIN_SBATCH_ARGS"
fi

GEN_JOB_ID=$(sbatch --parsable "${gen_args[@]}" scripts/submit_generate_slab_curriculum.slurm)
echo "Submitted generate job: $GEN_JOB_ID"

TRAIN_JOB_ID=$(sbatch --parsable --dependency=afterok:${GEN_JOB_ID} "${train_args[@]}" scripts/submit_train_curriculum_slabs.slurm)
echo "Submitted train job: $TRAIN_JOB_ID (afterok:$GEN_JOB_ID)"

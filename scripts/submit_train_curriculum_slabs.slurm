#!/bin/bash
#SBATCH --job-name=train_curr_slabs
#SBATCH --output=logs/train_curr_slabs_%j.out
#SBATCH --error=logs/train_curr_slabs_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/lustre/home/acastaneda/Fernando/PrAi}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/curriculum/slabs}"

cd "$PROJECT_ROOT"
mkdir -p logs

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

PROJECT_ROOT="$PROJECT_ROOT" \
DATA_ROOT="$DATA_ROOT" \
BATCH_SIZE="${BATCH_SIZE:-2}" \
NUM_WORKERS="${NUM_WORKERS:-4}" \
BASE_CHANNELS="${BASE_CHANNELS:-24}" \
EPOCHS_STAGE1="${EPOCHS_STAGE1:-25}" \
EPOCHS_STAGE2="${EPOCHS_STAGE2:-25}" \
EPOCHS_STAGE3="${EPOCHS_STAGE3:-30}" \
RUN_TAG="${RUN_TAG:-curriculum_slabs_${SLURM_JOB_ID}}" \
bash scripts/train_curriculum_slabs.sh

echo "Done: curriculum training finished"

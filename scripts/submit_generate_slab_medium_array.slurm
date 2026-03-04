#!/bin/bash
#SBATCH --job-name=gen_slab_med_arr
#SBATCH --output=logs/gen_slab_med_arr_%A_%a.out
#SBATCH --error=logs/gen_slab_med_arr_%A_%a.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=02:00:00
#SBATCH --array=0-1000%200

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/lustre/home/acastaneda/Fernando/PrAi}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/curriculum/slabs_curriculum_prod.json}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/curriculum/slabs_medium_only}"
STAGE_NAME="${STAGE_NAME:-stage2_medium}"
TOTAL_CASES="${TOTAL_CASES:-2500}"
CASES_PER_TASK="${CASES_PER_TASK:-3}"

cd "$PROJECT_ROOT"
mkdir -p logs

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python scripts/generate_slab_curriculum_chunk.py \
  --config "$CONFIG_PATH" \
  --out-root "$OUT_ROOT" \
  --stage-name "$STAGE_NAME" \
  --task-id "${SLURM_ARRAY_TASK_ID}" \
  --cases-per-task "$CASES_PER_TASK" \
  --total-cases "$TOTAL_CASES"

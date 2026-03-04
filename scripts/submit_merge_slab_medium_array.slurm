#!/bin/bash
#SBATCH --job-name=merge_slab_med
#SBATCH --output=logs/merge_slab_med_%j.out
#SBATCH --error=logs/merge_slab_med_%j.err
#SBATCH --partition=cpu
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --time=00:30:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/lustre/home/acastaneda/Fernando/PrAi}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/curriculum/slabs_medium_only}"
STAGE_NAME="${STAGE_NAME:-stage2_medium}"

cd "$PROJECT_ROOT"
mkdir -p logs

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python scripts/merge_slab_chunk_manifests.py \
  --out-root "$OUT_ROOT" \
  --stage-name "$STAGE_NAME" \
  --seed 42 \
  --train-frac 0.8 \
  --val-frac 0.1

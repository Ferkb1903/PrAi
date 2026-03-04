#!/bin/bash
#SBATCH --job-name=gen_slabs_curr
#SBATCH --output=logs/gen_slabs_curr_%j.out
#SBATCH --error=logs/gen_slabs_curr_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=06:00:00

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/lustre/home/acastaneda/Fernando/PrAi}"
CONFIG_PATH="${CONFIG_PATH:-$PROJECT_ROOT/configs/curriculum/slabs_curriculum.json}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/data/curriculum/slabs}"

cd "$PROJECT_ROOT"
mkdir -p logs

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

python scripts/generate_slab_curriculum_dataset.py \
  --config "$CONFIG_PATH" \
  --out-root "$OUT_ROOT"

echo "Done: slab curriculum dataset generated at $OUT_ROOT"

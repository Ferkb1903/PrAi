#!/bin/bash
#SBATCH --job-name=dual_paper_exp
#SBATCH --output=logs/dual_paper_exp_%j.out
#SBATCH --error=logs/dual_paper_exp_%j.err
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --gres=gpu:2
#SBATCH --time=48:00:00

set -euo pipefail

cd "$SLURM_SUBMIT_DIR"
mkdir -p logs

if [[ -f .venv/bin/activate ]]; then
  source .venv/bin/activate
fi

bash scripts/train_two_models_parallel.sh

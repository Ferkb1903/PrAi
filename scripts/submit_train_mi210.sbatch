#!/bin/bash
#SBATCH --job-name=prai_train
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%j.log

set -euo pipefail

PROJECT_ROOT="/lustre/home/acastaneda/Fernando/PrAi"
cd "$PROJECT_ROOT"
mkdir -p logs

source .venv/bin/activate

MANIFEST_TRAIN="${MANIFEST_TRAIN:-$PROJECT_ROOT/data/training_npz/spot_campaign_v2_parallel/manifest_train.csv}"
MANIFEST_VAL="${MANIFEST_VAL:-$PROJECT_ROOT/data/training_npz/spot_campaign_v2_parallel/manifest_val.csv}"
MANIFEST_TEST="${MANIFEST_TEST:-$PROJECT_ROOT/data/training_npz/spot_campaign_v2_parallel/manifest_test.csv}"

EPOCHS="${EPOCHS:-40}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASE_CHANNELS="${BASE_CHANNELS:-24}"
SAVE_EVERY="${SAVE_EVERY:-1}"
CROP_SIZE="${CROP_SIZE:-96,96,96}"
RESUME_CKPT="${RESUME_CKPT:-}"

CMD=(python scripts/train_residual_unet3d.py \
  --manifest-train "$MANIFEST_TRAIN" \
  --manifest-val "$MANIFEST_VAL" \
  --manifest-test "$MANIFEST_TEST" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --base-channels "$BASE_CHANNELS" \
  --save-every "$SAVE_EVERY" \
  --crop-size "$CROP_SIZE")

if [ -n "$RESUME_CKPT" ]; then
  CMD+=(--resume "$RESUME_CKPT")
fi

"${CMD[@]}"

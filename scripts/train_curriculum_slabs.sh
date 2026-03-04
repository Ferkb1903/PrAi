#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/data/curriculum/slabs}"
STAGES="${STAGES:-stage1_easy,stage2_medium,stage3_hard}"

BASE_OUT="${BASE_OUT:-$PROJECT_ROOT/checkpoints/curriculum_slabs}"
RUN_TAG="${RUN_TAG:-curriculum_slabs_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="$BASE_OUT/$RUN_TAG"
mkdir -p "$OUT_ROOT"

SEED="${SEED:-42}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASE_CHANNELS="${BASE_CHANNELS:-24}"
CROP_SIZE="${CROP_SIZE:-96,96,96}"
LR="${LR:-2.5e-4}"
MIN_LR="${MIN_LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
MODEL_VARIANT="${MODEL_VARIANT:-resunet_delta}"
VOXEL_WEIGHT_MODE="${VOXEL_WEIGHT_MODE:-paper_exp}"
VOXEL_WEIGHT_ALPHA="${VOXEL_WEIGHT_ALPHA:-3.0}"
DATASET_DOSE_NORM="${DATASET_DOSE_NORM:-none}"
USE_SE_BLOCKS="${USE_SE_BLOCKS:-1}"

EPOCHS_STAGE1="${EPOCHS_STAGE1:-25}"
EPOCHS_STAGE2="${EPOCHS_STAGE2:-25}"
EPOCHS_STAGE3="${EPOCHS_STAGE3:-30}"

cd "$PROJECT_ROOT"

echo "==============================================="
echo "Curriculum training on slab synthetic dataset"
echo "DATA_ROOT : $DATA_ROOT"
echo "STAGES    : $STAGES"
echo "OUT_ROOT  : $OUT_ROOT"
echo "==============================================="

IFS=',' read -r -a STAGE_ARR <<< "$STAGES"

resume_ckpt=""
stage_idx=0

for stage_name in "${STAGE_ARR[@]}"; do
  stage_idx=$((stage_idx + 1))

  stage_dir="$DATA_ROOT/$stage_name"
  train_manifest="$stage_dir/manifest_train.csv"
  val_manifest="$stage_dir/manifest_val.csv"
  test_manifest="$stage_dir/manifest_test.csv"

  if [[ ! -f "$train_manifest" || ! -f "$val_manifest" ]]; then
    echo "ERROR: manifest missing for stage=$stage_name"
    echo "  train: $train_manifest"
    echo "  val:   $val_manifest"
    exit 1
  fi

  epochs_var="EPOCHS_STAGE${stage_idx}"
  stage_epochs="${!epochs_var:-20}"

  stage_out="$OUT_ROOT/$stage_name"
  mkdir -p "$stage_out"

  cmd=(
    "$PYTHON_BIN" scripts/train_residual_unet3d.py
    --manifest-train "$train_manifest"
    --manifest-val "$val_manifest"
    --manifest-test "$test_manifest"
    --epochs "$stage_epochs"
    --batch-size "$BATCH_SIZE"
    --num-workers "$NUM_WORKERS"
    --base-channels "$BASE_CHANNELS"
    --crop-size "$CROP_SIZE"
    --seed "$SEED"
    --lr "$LR"
    --min-lr "$MIN_LR"
    --lr-scheduler cosine
    --weight-decay "$WEIGHT_DECAY"
    --model-variant "$MODEL_VARIANT"
    --voxel-weight-mode "$VOXEL_WEIGHT_MODE"
    --voxel-weight-alpha "$VOXEL_WEIGHT_ALPHA"
    --dataset-dose-norm "$DATASET_DOSE_NORM"
    --selection-metric val_l1
    --early-stop-patience 8
    --out-dir "$stage_out"
  )

  if [[ "$USE_SE_BLOCKS" == "1" ]]; then
    cmd+=(--use-se-blocks)
  fi

  if [[ -n "$resume_ckpt" && -f "$resume_ckpt" ]]; then
    cmd+=(--resume "$resume_ckpt")
  fi

  echo "\n>>> Stage $stage_idx: $stage_name (epochs=$stage_epochs)"
  "${cmd[@]}"

  if [[ -f "$stage_out/best.pt" ]]; then
    resume_ckpt="$stage_out/best.pt"
  else
    echo "WARNING: best checkpoint not found for $stage_name, fallback to last epoch"
    latest_epoch="$(ls -1 "$stage_out"/epoch_*.pt 2>/dev/null | sort | tail -n1 || true)"
    resume_ckpt="$latest_epoch"
  fi

done

echo "\nCurriculum training finished."
echo "Checkpoints: $OUT_ROOT"

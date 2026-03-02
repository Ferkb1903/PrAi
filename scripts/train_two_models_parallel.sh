#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

MANIFEST_TRAIN="${MANIFEST_TRAIN:-$PROJECT_ROOT/data/training_npz/manifest_train_low5k.csv}"
MANIFEST_VAL="${MANIFEST_VAL:-$PROJECT_ROOT/data/training_npz/manifest_val_low5k.csv}"
MANIFEST_TEST="${MANIFEST_TEST:-$PROJECT_ROOT/data/training_npz/manifest_test_low5k.csv}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
BASE_CHANNELS="${BASE_CHANNELS:-24}"
LR="${LR:-2.5e-4}"
MIN_LR="${MIN_LR:-5e-6}"
WEIGHT_DECAY="${WEIGHT_DECAY:-1e-5}"
CROP_SIZE="${CROP_SIZE:-96,96,96}"
ALPHA="${ALPHA:-3.0}"

GPU_A="${GPU_A:-0}"
GPU_B="${GPU_B:-1}"

RUN_TAG="${RUN_TAG:-paper_exp_dual_$(date +%Y%m%d_%H%M%S)}"
OUT_ROOT="${OUT_ROOT:-$PROJECT_ROOT/checkpoints/$RUN_TAG}"
LOG_DIR="$OUT_ROOT/logs"
mkdir -p "$LOG_DIR"

if [[ ! -f "$MANIFEST_TRAIN" || ! -f "$MANIFEST_VAL" ]]; then
  echo "ERROR: manifest train/val no encontrado"
  echo "  train: $MANIFEST_TRAIN"
  echo "  val:   $MANIFEST_VAL"
  exit 1
fi

cd "$PROJECT_ROOT"

echo "=========================================="
echo "Dual training with paper exponential loss"
echo "=========================================="
echo "RUN_TAG: $RUN_TAG"
echo "GPU_A: $GPU_A (resunet_delta + SE)"
echo "GPU_B: $GPU_B (resunet_direct + SE)"
echo "OUT_ROOT: $OUT_ROOT"
echo "=========================================="

COMMON_ARGS=(
  --manifest-train "$MANIFEST_TRAIN"
  --manifest-val "$MANIFEST_VAL"
  --manifest-test "$MANIFEST_TEST"
  --epochs "$EPOCHS"
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
  --base-channels "$BASE_CHANNELS"
  --lr "$LR"
  --min-lr "$MIN_LR"
  --lr-scheduler cosine
  --weight-decay "$WEIGHT_DECAY"
  --crop-size "$CROP_SIZE"
  --use-se-blocks
  --voxel-weight-mode paper_exp
  --voxel-weight-alpha "$ALPHA"
  --early-stop-patience 10
)

HIP_VISIBLE_DEVICES="$GPU_A" \
CUDA_VISIBLE_DEVICES="$GPU_A" \
"$PYTHON_BIN" scripts/train_residual_unet3d.py \
  "${COMMON_ARGS[@]}" \
  --model-variant resunet_delta \
  --out-dir "$OUT_ROOT/model_delta" \
  > "$LOG_DIR/model_delta.log" 2>&1 &
PID_A=$!

HIP_VISIBLE_DEVICES="$GPU_B" \
CUDA_VISIBLE_DEVICES="$GPU_B" \
"$PYTHON_BIN" scripts/train_residual_unet3d.py \
  "${COMMON_ARGS[@]}" \
  --model-variant resunet_direct \
  --out-dir "$OUT_ROOT/model_direct" \
  > "$LOG_DIR/model_direct.log" 2>&1 &
PID_B=$!

echo "PIDs: delta=$PID_A direct=$PID_B"
echo "Logs:"
echo "  tail -f $LOG_DIR/model_delta.log"
echo "  tail -f $LOG_DIR/model_direct.log"

FAIL=0
wait $PID_A || FAIL=1
wait $PID_B || FAIL=1

if [[ "$FAIL" -ne 0 ]]; then
  echo "One or both trainings failed. Check logs in $LOG_DIR"
  exit 1
fi

echo "Both trainings finished successfully."
echo "Checkpoints in: $OUT_ROOT"

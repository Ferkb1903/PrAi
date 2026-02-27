#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

PAIR_INDEX_CSV="${PAIR_INDEX_CSV:-$PROJECT_ROOT/cluster_jobs/spot_campaign_3060/pair_index.csv}"
OUT_NPZ_DIR="${OUT_NPZ_DIR:-$PROJECT_ROOT/data/training_npz/spot_campaign_v2}"
QC_REPORT="${QC_REPORT:-$PROJECT_ROOT/data/training_npz/qc_spot_campaign.csv}"
MANIFEST_ALL="${MANIFEST_ALL:-$PROJECT_ROOT/data/training_npz/manifest_all.csv}"
MANIFEST_TRAIN="${MANIFEST_TRAIN:-$PROJECT_ROOT/data/training_npz/manifest_train.csv}"
MANIFEST_VAL="${MANIFEST_VAL:-$PROJECT_ROOT/data/training_npz/manifest_val.csv}"
MANIFEST_TEST="${MANIFEST_TEST:-$PROJECT_ROOT/data/training_npz/manifest_test.csv}"
SPLIT_SUMMARY="${SPLIT_SUMMARY:-$PROJECT_ROOT/data/training_npz/split_summary.json}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BASE_CHANNELS="${BASE_CHANNELS:-24}"

cd "$PROJECT_ROOT"

echo "[1/2] Preparando tensores optimizados..."
"$PYTHON_BIN" scripts/prepare_training_tensors.py \
  --pair-index-csv "$PAIR_INDEX_CSV" \
  --out-dir "$OUT_NPZ_DIR" \
  --qc-report "$QC_REPORT" \
  --manifest-all "$MANIFEST_ALL" \
  --manifest-train "$MANIFEST_TRAIN" \
  --manifest-val "$MANIFEST_VAL" \
  --manifest-test "$MANIFEST_TEST" \
  --split-summary "$SPLIT_SUMMARY"

echo "[2/2] Entrenando Residual 3D U-Net..."
"$PYTHON_BIN" scripts/train_residual_unet3d.py \
  --manifest-train "$MANIFEST_TRAIN" \
  --manifest-val "$MANIFEST_VAL" \
  --manifest-test "$MANIFEST_TEST" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --base-channels "$BASE_CHANNELS"

echo "Listo. Revisa:"
echo "- QC: $QC_REPORT"
echo "- Split: $SPLIT_SUMMARY"
echo "- Checkpoints: $PROJECT_ROOT/checkpoints/resunet3d"

#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

echo "[Checking] torch availability..."
if ! "$PYTHON_BIN" -c "import torch; print(f'torch {torch.__version__}')" 2>/dev/null; then
    echo "[Install] Installing torch (CUDA-enabled)..."
    "$PYTHON_BIN" -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -20
fi

MANIFEST_TRAIN="${MANIFEST_TRAIN:-$PROJECT_ROOT/data/training_npz/manifest_train.csv}"
MANIFEST_VAL="${MANIFEST_VAL:-$PROJECT_ROOT/data/training_npz/manifest_val.csv}"
MANIFEST_TEST="${MANIFEST_TEST:-$PROJECT_ROOT/data/training_npz/manifest_test.csv}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-8}"
BASE_CHANNELS="${BASE_CHANNELS:-24}"

cd "$PROJECT_ROOT"

echo "[Training] Residual 3D U-Net..."

if [ ! -f "$MANIFEST_TRAIN" ] || [ ! -f "$MANIFEST_VAL" ]; then
    echo "ERROR: Manifests not found."
    echo "Train: $MANIFEST_TRAIN (exists: $([ -f "$MANIFEST_TRAIN" ] && echo yes || echo no))"
    echo "Val: $MANIFEST_VAL (exists: $([ -f "$MANIFEST_VAL" ] && echo yes || echo no))"
    exit 1
fi

N_TRAIN=$(tail -n +2 "$MANIFEST_TRAIN" | wc -l)
N_VAL=$(tail -n +2 "$MANIFEST_VAL" | wc -l)
echo "Dataset: $N_TRAIN train, $N_VAL val"

if [ "$N_TRAIN" -lt 10 ]; then
    echo "ERROR: Too few training samples ($N_TRAIN < 10)"
    exit 1
fi

"$PYTHON_BIN" scripts/train_residual_unet3d.py \
  --manifest-train "$MANIFEST_TRAIN" \
  --manifest-val "$MANIFEST_VAL" \
  --manifest-test "$MANIFEST_TEST" \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-workers "$NUM_WORKERS" \
  --base-channels "$BASE_CHANNELS"

echo "Training complete. Checkpoints in: $PROJECT_ROOT/checkpoints/resunet3d"


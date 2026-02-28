#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"
PYTHON_BIN="${PYTHON_BIN:-$PROJECT_ROOT/.venv/bin/python}"

echo "=========================================="
echo "GPU & TORCH DETECTION"
echo "=========================================="
echo "Python: $PYTHON_BIN"
"$PYTHON_BIN" --version

# Setup ROCm/MIOpen environment if AMD GPU detected
MIOPEN_TEMP_DIR="/tmp/miopen_cache_$(whoami)"
if [ ! -d "$MIOPEN_TEMP_DIR" ]; then
    mkdir -p "$MIOPEN_TEMP_DIR"
    chmod 700 "$MIOPEN_TEMP_DIR"
fi

export MIOPEN_USER_DB_PATH="$MIOPEN_TEMP_DIR/miopen_db"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_TEMP_DIR"
if [ -z "${TMPDIR:-}" ] || [ ! -d "${TMPDIR:-}" ]; then
    export TMPDIR=/tmp
fi

# Detect GPU type
GPU_TYPE="NONE"
if command -v rocm-smi &> /dev/null; then
    echo "[✓] AMD ROCm detected (rocm-smi available)"
    GPU_TYPE="AMD_ROCM"
    TORCH_INDEX="https://download.pytorch.org/whl/rocm5.7"
elif command -v nvidia-smi &> /dev/null; then
    echo "[✓] NVIDIA CUDA detected (nvidia-smi available)"
    GPU_TYPE="NVIDIA_CUDA"
    TORCH_INDEX="https://download.pytorch.org/whl/cu118"
else
    echo "[!] WARNING: No GPU detected. Training will run on CPU (~100x slower)"
    GPU_TYPE="CPU"
    TORCH_INDEX=""
fi

echo "[Checking] torch availability..."
INSTALLED_TORCH=$("$PYTHON_BIN" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "NOT_INSTALLED")

if [ "$INSTALLED_TORCH" = "NOT_INSTALLED" ]; then
    echo "[✗] torch not installed"
    if [ "$GPU_TYPE" = "NONE" ]; then
        echo "[Install] Installing torch (CPU-only)..."
        "$PYTHON_BIN" -m pip install --upgrade torch torchvision torchaudio 2>&1 | tail -5
    else
        echo "[Install] Installing torch with $GPU_TYPE support from $TORCH_INDEX..."
        "$PYTHON_BIN" -m pip install --upgrade torch torchvision torchaudio --index-url "$TORCH_INDEX" 2>&1 | tail -5
    fi
    INSTALLED_TORCH=$("$PYTHON_BIN" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "ERROR")
else
    echo "[✓] torch $INSTALLED_TORCH already installed"
fi

# Check if torch version matches GPU type (critical check!)
if [ "$GPU_TYPE" = "AMD_ROCM" ] && [[ "$INSTALLED_TORCH" == *"cu1"* ]]; then
    echo ""
    echo "[!] ERROR: torch compiled for CUDA (cu*) but GPU is AMD ROCm!"
    echo "    Version: $INSTALLED_TORCH"
    echo "    This will run on CPU - extremely slow!"
    echo ""
    echo "[Fix] Removing CUDA torch and installing ROCm version..."
    "$PYTHON_BIN" -m pip uninstall -y torch torchvision torchaudio 2>&1 | tail -2
    "$PYTHON_BIN" -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7 2>&1 | tail -10
    INSTALLED_TORCH=$("$PYTHON_BIN" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "ERROR")
    echo "[✓] torch reinstalled: $INSTALLED_TORCH"
elif [ "$GPU_TYPE" = "NVIDIA_CUDA" ] && [[ "$INSTALLED_TORCH" == *"rocm"* ]]; then
    echo ""
    echo "[!] ERROR: torch compiled for ROCm but GPU is NVIDIA CUDA!"
    echo "    Version: $INSTALLED_TORCH"
    echo "    This will run on CPU - extremely slow!"
    echo ""
    echo "[Fix] Removing ROCm torch and installing CUDA version..."
    "$PYTHON_BIN" -m pip uninstall -y torch torchvision torchaudio 2>&1 | tail -2
    "$PYTHON_BIN" -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118 2>&1 | tail -10
    INSTALLED_TORCH=$("$PYTHON_BIN" -c "import torch; print(torch.__version__)" 2>/dev/null || echo "ERROR")
    echo "[✓] torch reinstalled: $INSTALLED_TORCH"
fi

# Verify GPU detection in PyTorch
echo "[Checking] GPU availability in PyTorch..."
GPU_AVAILABLE=$("$PYTHON_BIN" -c "import torch; print(torch.cuda.is_available() or hasattr(torch, 'xpu'))" 2>/dev/null || echo "ERROR")

if [ "$GPU_AVAILABLE" = "True" ]; then
    echo "[✓] GPU is available to PyTorch"
    "$PYTHON_BIN" -c "import torch; print(f'  Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"Unknown\"}')"
elif [ "$GPU_TYPE" = "AMD_ROCM" ]; then
    echo "[!] WARNING: AMD ROCm detected but PyTorch doesn't see GPU"
    echo "    Reinstalling with explicit ROCm support..."
    "$PYTHON_BIN" -m pip uninstall -y torch torchvision torchaudio 2>&1 | tail -2
    "$PYTHON_BIN" -m pip install --upgrade torch torchvision torchaudio --index-url https://download.pytorch.org/whl/rocm5.7 2>&1 | tail -10
    GPU_AVAILABLE=$("$PYTHON_BIN" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "FALSE")
    if [ "$GPU_AVAILABLE" = "True" ]; then
        echo "[✓] GPU is now available"
    else
        echo "[✗] GPU still not detected. Check ROCm installation."
    fi
else
    echo "[!] No GPU available to PyTorch"
fi

echo "=========================================="

MANIFEST_TRAIN="${MANIFEST_TRAIN:-$PROJECT_ROOT/data/training_npz/manifest_train.csv}"
MANIFEST_VAL="${MANIFEST_VAL:-$PROJECT_ROOT/data/training_npz/manifest_val.csv}"
MANIFEST_TEST="${MANIFEST_TEST:-$PROJECT_ROOT/data/training_npz/manifest_test.csv}"

EPOCHS="${EPOCHS:-80}"
BATCH_SIZE="${BATCH_SIZE:-2}"
NUM_WORKERS="${NUM_WORKERS:-4}"
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


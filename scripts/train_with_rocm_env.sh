#!/usr/bin/env bash
# Wrapper script to properly configure ROCm/MIOpen environment before training.
# Handles temporary directories, library paths, and MIOpen optimizations.

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-$PWD}"

echo "=========================================="
echo "CONFIGURING ROCM ENVIRONMENT"
echo "=========================================="

# Setup temporary directory for MIOpen
MIOPEN_TEMP_DIR="${MIOPEN_TEMP_DIR:-/tmp/miopen_cache_$USER}"
if [ ! -d "$MIOPEN_TEMP_DIR" ]; then
    echo "[Creating] MIOpen temp directory: $MIOPEN_TEMP_DIR"
    mkdir -p "$MIOPEN_TEMP_DIR"
    chmod 700 "$MIOPEN_TEMP_DIR"
else
    echo "[✓] MIOpen temp directory exists: $MIOPEN_TEMP_DIR"
fi

# Setup user database for MIOpen (kernel cache)
MIOPEN_USER_DB_PATH="${MIOPEN_USER_DB_PATH:-$MIOPEN_TEMP_DIR/miopen_db}"
if [ ! -d "$MIOPEN_USER_DB_PATH" ]; then
    echo "[Creating] MIOpen user DB path: $MIOPEN_USER_DB_PATH"
    mkdir -p "$MIOPEN_USER_DB_PATH"
    chmod 700 "$MIOPEN_USER_DB_PATH"
else
    echo "[✓] MIOpen user DB path exists: $MIOPEN_USER_DB_PATH"
fi

# Ensure TMPDIR is set to a valid location
if [ -z "${TMPDIR:-}" ] || [ ! -d "${TMPDIR:-}" ]; then
    export TMPDIR=/tmp
    echo "[Set] TMPDIR=/tmp (for MIOpen bootstrap)"
fi

# Export ROCm/MIOpen configuration
export MIOPEN_USER_DB_PATH="$MIOPEN_USER_DB_PATH"
export MIOPEN_CUSTOM_CACHE_DIR="$MIOPEN_TEMP_DIR"

# Optional: enable MIOpen benchmarking/optimization (slower first run, faster later)
export MIOPEN_FIND_ENFORCE=1  # Enable kernel search (off by default for speed)

# Optional: enable verbose logging for debugging
# export MIOPEN_LOGGING_LEVEL=7
# export MIOPEN_DEBUG_CONV_IMPLICIT_GEMM=1

echo "[✓] MIOPEN_USER_DB_PATH=$MIOPEN_USER_DB_PATH"
echo "[✓] MIOPEN_CUSTOM_CACHE_DIR=$MIOPEN_TEMP_DIR"
echo "[✓] TMPDIR=$TMPDIR"
echo "=========================================="
echo ""

# Now run the training script with these environment variables
cd "$PROJECT_ROOT"
exec bash scripts/train_mi210.sh "$@"

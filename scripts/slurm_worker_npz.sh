#!/bin/bash
# Script de trabajo para Slurm (lanzado por submit_npz_parallel.sh)
# Variables esperadas: PAIR_INDEX_FILE, OUT_DIR, DOSE_NORM, DOSE_STEM, LOW_SCALE_FACTOR, TARGET_DOSE_MAX

cd "$SLURM_SUBMIT_DIR"

# Activa entorno
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

PAIR_OFFSET="${PAIR_OFFSET:-0}"
DOSE_STEM="${DOSE_STEM:-dose_voxelized_ct_edep}"
LOW_SCALE_FACTOR="${LOW_SCALE_FACTOR:-200.0}"
TARGET_DOSE_MAX="${TARGET_DOSE_MAX:-10.0}"
GLOBAL_PAIR_IDX=$((PAIR_OFFSET + SLURM_ARRAY_TASK_ID))

# Ejecuta script de procesamiento del pair
python scripts/prepare_training_tensors_parallel.py \
    --pair-index-csv "$PAIR_INDEX_FILE" \
    --pair-idx "$GLOBAL_PAIR_IDX" \
    --out-dir "$OUT_DIR" \
    --dose-norm-const "$DOSE_NORM" \
    --dose-stem "$DOSE_STEM" \
    --low-scale-factor "$LOW_SCALE_FACTOR" \
    --target-dose-max "$TARGET_DOSE_MAX"

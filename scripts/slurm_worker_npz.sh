#!/bin/bash
# Script de trabajo para Slurm (lanzado por submit_npz_parallel.sh)
# Variables esperadas: PAIR_INDEX_FILE, OUT_DIR, DOSE_NORM

cd "$SLURM_SUBMIT_DIR"

# Activa entorno
if [[ -f .venv/bin/activate ]]; then
    source .venv/bin/activate
fi

PAIR_OFFSET="${PAIR_OFFSET:-0}"
GLOBAL_PAIR_IDX=$((PAIR_OFFSET + SLURM_ARRAY_TASK_ID))

# Ejecuta script de procesamiento del pair
python scripts/prepare_training_tensors_parallel.py \
    --pair-index-csv "$PAIR_INDEX_FILE" \
    --pair-idx "$GLOBAL_PAIR_IDX" \
    --out-dir "$OUT_DIR" \
    --dose-norm-const "$DOSE_NORM"

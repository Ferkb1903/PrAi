#!/bin/bash
# Procesa tensores NPZ en paralelo con Slurm @ 200 concurrencia
# Uso: bash scripts/submit_npz_parallel.sh [pair_index] [out_dir] [dose_norm]

set -e

PAIR_INDEX_FILE="${1:-data/training_npz/pair_index_low5k.csv}"
OUT_DIR="${2:-data/training_npz/spot_campaign_v2_low5k}"
DOSE_NORM="${3:-1.0}"

# Validación
if [[ ! -f "$PAIR_INDEX_FILE" ]]; then
    CANDIDATE=$(find data -maxdepth 6 -type f \( -name "pair_index_low5k.csv" -o -name "pair_index.csv" \) 2>/dev/null | head -n 1 || true)
    if [[ -n "$CANDIDATE" && -f "$CANDIDATE" ]]; then
        echo "[WARN] No existe '$PAIR_INDEX_FILE'. Usando detectado: '$CANDIDATE'"
        PAIR_INDEX_FILE="$CANDIDATE"
    else
        echo "ERROR: No existe $PAIR_INDEX_FILE"
        echo "Sugerencia: busca el archivo con: find data -type f -name 'pair_index*.csv'"
        exit 1
    fi
fi

# Cuenta pares
N_PAIRS=$(($(wc -l < "$PAIR_INDEX_FILE") - 1))
if [[ $N_PAIRS -lt 1 ]]; then
    echo "ERROR: No hay pares para procesar"
    exit 1
fi

echo "=========================================="
echo "CONVERSIÓN NPZ PARALELA - SLURM"
echo "=========================================="
echo "Pair index: $PAIR_INDEX_FILE"
echo "Total pairs: $N_PAIRS"
echo "Output dir: $OUT_DIR"
echo "Max concurrency: 200"
echo "=========================================="
echo ""

# Crea directorio de logs
LOGS_DIR="${OUT_DIR}/../logs_npz_parallel"
mkdir -p "$LOGS_DIR"

# Paths absolutos (para que Slurm los encuentre correctamente)
PAIR_INDEX_ABS="$(cd "$(dirname "$PAIR_INDEX_FILE")" && pwd)/$(basename "$PAIR_INDEX_FILE")"
OUT_DIR_ABS="$(mkdir -p "$OUT_DIR" && cd "$OUT_DIR" && pwd)"

MAX_ARRAY_SIZE=1000
JOB_IDS=()

for OFFSET in $(seq 0 "$MAX_ARRAY_SIZE" $((N_PAIRS - 1))); do
    REM=$((N_PAIRS - OFFSET))
    if (( REM > MAX_ARRAY_SIZE )); then
        LAST=999
    else
        LAST=$((REM - 1))
    fi

    echo "Submitting chunk: offset=$OFFSET array=0-$LAST%200"
    JOB_OUTPUT=$(sbatch \
        --array=0-${LAST}%200 \
        --job-name="npz_parallel" \
        --output="$LOGS_DIR/job_%A_%a.log" \
        --error="$LOGS_DIR/job_%A_%a.err" \
        --export="PAIR_INDEX_FILE=$PAIR_INDEX_ABS,OUT_DIR=$OUT_DIR_ABS,DOSE_NORM=$DOSE_NORM,PAIR_OFFSET=$OFFSET" \
        scripts/slurm_worker_npz.sh
    )

    JOB_ID=$(echo "$JOB_OUTPUT" | grep -oP 'Submitted batch job \K\d+')
    JOB_IDS+=("$JOB_ID")
done

echo "✓ Jobs submitted: ${JOB_IDS[*]}"
echo ""
echo "Monitor:"
echo "  squeue -u \$USER | grep npz_parallel"
echo "  tail -f $LOGS_DIR/job_*.log"
echo ""
echo "Cuando termine (all jobs), ejecuta:"
echo "  python scripts/aggregate_training_tensors.py --out-dir '$OUT_DIR_ABS'"
echo ""
echo "=========================================="

#!/usr/bin/env bash
# Uso: bash run_pair_array_offset.sh <OFFSET>
set -euo pipefail

OFFSET="${1:-0}"

cd "/lustre/home/acastaneda/Fernando/PrAi"
TMPDIR="${TMPDIR:-$PWD/.tmp_slurm}"
mkdir -p "$TMPDIR"
export TMPDIR
PYTHON_BIN="${PYTHON_BIN:-./.venv/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="$(command -v python3 || command -v python || true)"
fi
if [[ -z "$PYTHON_BIN" ]]; then
  echo "No se encontró Python ejecutable" >&2
  exit 127
fi
mkdir -p "cluster_jobs/spot_campaign_3060/logs"
PAIR_CSV="cluster_jobs/spot_campaign_3060/pair_index.csv"
ROW_NUM=$((SLURM_ARRAY_TASK_ID + OFFSET + 2))
ROW=$(sed -n "${ROW_NUM}p" "$PAIR_CSV")
if [[ -z "$ROW" ]]; then
  echo "No row for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID OFFSET=$OFFSET" >&2
  exit 2
fi
IFS="," read -r case_id input_type input_path energy_mev spot_idx spot_x_mm spot_y_mm pre_mhd low_out high_out <<< "$ROW"
if [[ ! -f "$pre_mhd" ]]; then
  mkdir -p "$(dirname "$pre_mhd")"
  "$PYTHON_BIN" scripts/preprocess_ct_for_gate.py --input "$input_path" --input-type "$input_type" --output-mhd "$pre_mhd" --spacing-mm 2.0
fi
mkdir -p "$low_out" "$high_out"
SOURCE_X_MM="$spot_x_mm" SOURCE_Y_MM="$spot_y_mm" PYTHON_BIN="$PYTHON_BIN" bash scripts/run_gate_voxelized_shared_env.sh "$pre_mhd" "$low_out" "$energy_mev" 50000 101 point 1 1 1.0 -30.0 "configs/hu_material_map_v1.json"
SOURCE_X_MM="$spot_x_mm" SOURCE_Y_MM="$spot_y_mm" PYTHON_BIN="$PYTHON_BIN" bash scripts/run_gate_voxelized_shared_env.sh "$pre_mhd" "$high_out" "$energy_mev" 1000000 202 point 1 1 1.0 -30.0 "configs/hu_material_map_v1.json"

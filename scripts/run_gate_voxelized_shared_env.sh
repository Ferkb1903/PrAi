#!/usr/bin/env bash
set -euo pipefail

CT_MHD="${1:-data/raw/ct_prostate_0002/ct.mhd}"
OUTPUT_DIR="${2:-outputs/gate_ct_test}"
ENERGY_MEV="${3:-150.0}"
N_EVENTS="${4:-10000}"
SEED="${5:-42}"
SOURCE_MODE="${6:-point}"
BEAMLET_NX="${7:-5}"
BEAMLET_NY="${8:-5}"
BEAMLET_PITCH_MM="${9:-6.0}"
SOURCE_Z_CM="${10:--30.0}"
HU_MAP_JSON="${11:-configs/hu_material_map_v1.json}"

PYTHON_BIN="${PYTHON_BIN:-python}"
GEANT4_SH="${GEANT4_SH:-}"

if ! command -v "$PYTHON_BIN" >/dev/null 2>&1; then
  echo "Python no disponible en PATH: $PYTHON_BIN" >&2
  exit 1
fi

if [[ -n "$GEANT4_SH" ]]; then
  if [[ ! -f "$GEANT4_SH" ]]; then
    echo "No existe geant4.sh en: $GEANT4_SH" >&2
    exit 1
  fi
  . "$GEANT4_SH"
fi

"$PYTHON_BIN" scripts/gate_voxelized_ct_experiment.py \
  --ct-mhd "$CT_MHD" \
  --output-dir "$OUTPUT_DIR" \
  --energy-mev "$ENERGY_MEV" \
  --n-events "$N_EVENTS" \
  --seed "$SEED" \
  --source-mode "$SOURCE_MODE" \
  --beamlet-nx "$BEAMLET_NX" \
  --beamlet-ny "$BEAMLET_NY" \
  --beamlet-pitch-mm "$BEAMLET_PITCH_MM" \
  --source-z-cm "$SOURCE_Z_CM" \
  --hu-map-json "$HU_MAP_JSON"

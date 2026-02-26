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
PROGRESS_UPDATE_SEC="${PROGRESS_UPDATE_SEC:-20}"
EVENT_MODULO="${EVENT_MODULO:-100000}"

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

estimate_runtime_sec() {
  awk -v n="$N_EVENTS" -v e="$ENERGY_MEV" 'BEGIN {
    base = 135.0
    alpha = 1.5
    if (e <= 0) e = 90.0
    t = base * (n / 1000000.0) * ((e / 90.0) ^ alpha)
    if (t < 5) t = 5
    printf "%d", int(t + 0.5)
  }'
}

print_progress_line() {
  local elapsed="$1"
  local estimated="$2"
  local pct="$3"
  local width=30
  local filled=$((pct * width / 100))
  local empty=$((width - filled))
  local bar_filled bar_empty
  bar_filled=$(printf '%*s' "$filled" '' | tr ' ' '#')
  bar_empty=$(printf '%*s' "$empty" '' | tr ' ' '-')
  echo "Progress E${ENERGY_MEV} N${N_EVENTS}: [${bar_filled}${bar_empty}] ${pct}% elapsed=${elapsed}s est=${estimated}s"
}

EST_SEC="$(estimate_runtime_sec)"
START_TS="$(date +%s)"

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
  --hu-map-json "$HU_MAP_JSON" \
  --event-modulo "$EVENT_MODULO" &
SIM_PID=$!

while kill -0 "$SIM_PID" 2>/dev/null; do
  NOW_TS="$(date +%s)"
  ELAPSED=$((NOW_TS - START_TS))
  PCT=$((ELAPSED * 100 / EST_SEC))
  if ((PCT > 99)); then
    PCT=99
  fi
  print_progress_line "$ELAPSED" "$EST_SEC" "$PCT"
  sleep "$PROGRESS_UPDATE_SEC"
done

wait "$SIM_PID"
RC=$?
END_TS="$(date +%s)"
TOTAL_ELAPSED=$((END_TS - START_TS))

if ((RC == 0)); then
  print_progress_line "$TOTAL_ELAPSED" "$EST_SEC" 100
else
  echo "Progress E${ENERGY_MEV} N${N_EVENTS}: FAILED rc=${RC} elapsed=${TOTAL_ELAPSED}s est=${EST_SEC}s" >&2
fi

exit "$RC"

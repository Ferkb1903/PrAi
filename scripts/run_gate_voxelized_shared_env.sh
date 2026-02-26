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
RUN_VERBOSE="${RUN_VERBOSE:-1}"
EVENT_VERBOSE="${EVENT_VERBOSE:-1}"

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

print_event_progress_line() {
  local elapsed="$1"
  local done_events="$2"
  local pct="$3"
  local width=30
  local filled=$((pct * width / 100))
  local empty=$((width - filled))
  local bar_filled bar_empty speed
  bar_filled=$(printf '%*s' "$filled" '' | tr ' ' '#')
  bar_empty=$(printf '%*s' "$empty" '' | tr ' ' '-')
  if ((elapsed > 0)); then
    speed=$((done_events / elapsed))
  else
    speed=0
  fi
  echo "Progress E${ENERGY_MEV} N${N_EVENTS}: [${bar_filled}${bar_empty}] ${pct}% events=${done_events}/${N_EVENTS} speed=${speed} ev/s elapsed=${elapsed}s"
}

EST_SEC="$(estimate_runtime_sec)"
START_TS=0
TMP_LOG="$(mktemp -t prai_gate_run_XXXX.log)"
WAIT_MSG_PRINTED=0
STOP_DETECTED=0

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
  --event-modulo "$EVENT_MODULO" \
  --run-verbose "$RUN_VERBOSE" \
  --event-verbose "$EVENT_VERBOSE" > >(tee -a "$TMP_LOG") 2>&1 &
SIM_PID=$!

while kill -0 "$SIM_PID" 2>/dev/null; do
  if ((START_TS == 0)); then
    if grep -q "Simulation: START" "$TMP_LOG"; then
      START_TS="$(date +%s)"
      echo "Progress E${ENERGY_MEV} N${N_EVENTS}: Simulation START detected"
    elif ((WAIT_MSG_PRINTED == 0)); then
      echo "Progress E${ENERGY_MEV} N${N_EVENTS}: waiting for Simulation: START"
      WAIT_MSG_PRINTED=1
      sleep "$PROGRESS_UPDATE_SEC"
      continue
    else
      sleep "$PROGRESS_UPDATE_SEC"
      continue
    fi
  fi

  NOW_TS="$(date +%s)"
  ELAPSED=$((NOW_TS - START_TS))

  if grep -q "Simulation: STOP" "$TMP_LOG"; then
    STOP_DETECTED=1
    break
  fi

  LAST_EVENT="$(grep -Eo '[Ee]vent[^0-9]*[0-9]+' "$TMP_LOG" | grep -Eo '[0-9]+' | tail -n 1 || true)"
  if [[ -n "$LAST_EVENT" ]]; then
    if ((LAST_EVENT > N_EVENTS)); then
      LAST_EVENT="$N_EVENTS"
    fi
    PCT=$((LAST_EVENT * 100 / N_EVENTS))
    if ((PCT > 99)); then
      PCT=99
    fi
    print_event_progress_line "$ELAPSED" "$LAST_EVENT" "$PCT"
  else
    echo "Progress E${ENERGY_MEV} N${N_EVENTS}: heartbeat elapsed=${ELAPSED}s (no event counter detected yet)"
  fi

  sleep "$PROGRESS_UPDATE_SEC"
done

set +e
wait "$SIM_PID"
RC=$?
set -e
END_TS="$(date +%s)"
if ((START_TS == 0)); then
  START_TS="$END_TS"
fi
TOTAL_ELAPSED=$((END_TS - START_TS))
DoseOut="${OUTPUT_DIR}/dose_voxelized_ct_edep.mhd"

if ((RC == 0)); then
  print_progress_line "$TOTAL_ELAPSED" "$EST_SEC" 100
else
  if ((RC == 139)) && [[ -f "$DoseOut" ]]; then
    echo "Progress E${ENERGY_MEV} N${N_EVENTS}: WARNING rc=139 (segfault post-run), output present -> treating as success" >&2
    print_progress_line "$TOTAL_ELAPSED" "$EST_SEC" 100
    RC=0
  else
    echo "Progress E${ENERGY_MEV} N${N_EVENTS}: FAILED rc=${RC} elapsed=${TOTAL_ELAPSED}s est=${EST_SEC}s" >&2
  fi
fi

rm -f "$TMP_LOG"

exit "$RC"

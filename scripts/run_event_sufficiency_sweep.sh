#!/usr/bin/env bash
set -euo pipefail

CT_MHD="${1:-data/raw/ct_prostate_0002/ct.mhd}"
OUT_ROOT="${2:-outputs/event_sufficiency_sweep}"
EVENTS_CSV="${3:-10000,20000,50000,100000}"
SEEDS_CSV="${4:-11,22,33,44,55}"
ENERGIES_CSV="${5:-90,120,150,180,210}"
SOURCE_MODE="${6:-beamlet}"
BEAMLET_NX="${7:-5}"
BEAMLET_NY="${8:-5}"
BEAMLET_PITCH_MM="${9:-6.0}"
SOURCE_Z_CM="${10:--30.0}"
HU_MAP_JSON="${11:-configs/hu_material_map_v1.json}"

mkdir -p "$OUT_ROOT"

echo "Running sufficiency sweep"
echo "  CT: $CT_MHD"
echo "  OUT_ROOT: $OUT_ROOT"
echo "  EVENTS: $EVENTS_CSV"
echo "  SEEDS: $SEEDS_CSV"
echo "  ENERGIES: $ENERGIES_CSV"

IFS=',' read -r -a EVENTS <<< "$EVENTS_CSV"
IFS=',' read -r -a SEEDS <<< "$SEEDS_CSV"
IFS=',' read -r -a ENERGIES <<< "$ENERGIES_CSV"

for N in "${EVENTS[@]}"; do
  for S in "${SEEDS[@]}"; do
    RUN_ROOT="$OUT_ROOT/N${N}_seed${S}"
    mkdir -p "$RUN_ROOT"

    echo "==> N=$N seed=$S"
    for E in "${ENERGIES[@]}"; do
      OUT_DIR="$RUN_ROOT/E${E}"
      echo "    -> E=${E} MeV"
      bash scripts/run_gate_voxelized_shared_env.sh \
        "$CT_MHD" \
        "$OUT_DIR" \
        "$E" \
        "$N" \
        "$S" \
        "$SOURCE_MODE" \
        "$BEAMLET_NX" \
        "$BEAMLET_NY" \
        "$BEAMLET_PITCH_MM" \
        "$SOURCE_Z_CM" \
        "$HU_MAP_JSON"
    done
  done
done

echo "Sweep done: $OUT_ROOT"

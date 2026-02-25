#!/usr/bin/env bash
set -euo pipefail

CT_MHD="${1:-data/raw/ct_prostate_0002/ct.mhd}"
OUT_ROOT="${2:-outputs/gate_energy_scan_beamlet}"
N_EVENTS="${3:-200000}"
SEED="${4:-42}"

# Lista de energías MeV (ajustable)
ENERGIES=(90 120 150 180 210)

mkdir -p "$OUT_ROOT"

echo "Running beamlet energy scan with N_EVENTS=$N_EVENTS"
for E in "${ENERGIES[@]}"; do
  OUT_DIR="$OUT_ROOT/E${E}"
  echo "--> Energy ${E} MeV"
  bash scripts/run_gate_voxelized_shared_env.sh \
    "$CT_MHD" \
    "$OUT_DIR" \
    "$E" \
    "$N_EVENTS" \
    "$SEED" \
    beamlet \
    5 \
    5 \
    6.0 \
    -30.0

done

echo "Scan completed in: $OUT_ROOT"

#!/usr/bin/env bash
set -euo pipefail

IN_ROOT="${1:-data/raw/tcia_hn_nrrd}"
OUT_ROOT="${2:-data/raw/tcia_hn_mhd}"
MANIFEST="${3:-data/raw/tcia_hn_manifest.csv}"

PY_BIN="${PY_BIN:-/home/fer/fer/ProtonAI/PrAI/.venv/bin/python}"

"$PY_BIN" scripts/ingest_tcia_nrrd_ct.py \
  --input-root "$IN_ROOT" \
  --output-root "$OUT_ROOT" \
  --manifest-csv "$MANIFEST" \
  --include-regex "(ct|image|img)" \
  --exclude-regex "(mask|label|seg|contour|oar|gtv|ptv|rtstruct)"

echo "Done. Manifest: $MANIFEST"

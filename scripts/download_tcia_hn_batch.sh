#!/usr/bin/env bash
set -euo pipefail

CASE_FILE="${1:-configs/tcia_hn_cases_batch_01.txt}"
SPLIT="${2:-auto}"
LABELER="${3:-auto}"
OUT_ROOT="${4:-data/raw/tcia_hn_subset/nrrds}"

if [[ ! -f "$CASE_FILE" ]]; then
  echo "Case file not found: $CASE_FILE" >&2
  exit 1
fi

CASES_CSV=$(paste -sd, "$CASE_FILE")

./.venv/bin/python scripts/download_tcia_hn_ct_subset.py \
  --split "$SPLIT" \
  --labeler "$LABELER" \
  --cases "$CASES_CSV" \
  --out-root "$OUT_ROOT"

echo "Downloaded cases from $CASE_FILE into $OUT_ROOT/$SPLIT/$LABELER"

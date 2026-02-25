#!/usr/bin/env bash
set -euo pipefail

PYTHON_BIN="${PYTHON_BIN:-/home/fer/fer/ProtonAI/PrAI/.venv/bin/python}"
"$PYTHON_BIN" -m src.eval

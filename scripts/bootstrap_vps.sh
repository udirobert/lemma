#!/usr/bin/env bash
# VPS-side bootstrap: venv + deps + sanity check. Idempotent.
set -euo pipefail
cd "$(dirname "$0")/.."

PY=python3
if ! $PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 10) else 1)'; then
  echo "error: python3 >= 3.10 required on the VPS" >&2
  exit 1
fi
echo "[bootstrap] $($PY --version)"

if [ ! -d .venv ]; then
  $PY -m venv .venv
fi
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements-agent.txt

echo "[bootstrap] sanity: judge regression on the grokking fixture"
.venv/bin/python -m agent.cli judge --regression
echo "[bootstrap] done. Launch with: ./scripts/run_overnight.sh <paper-source> [run-name]"

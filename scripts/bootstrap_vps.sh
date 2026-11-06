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

# bubblewrap = the Linux sandbox for LLM-generated audit scripts
# (agent/script_sandbox.py). Without it the auditor fails closed —
# claims record sandbox_unavailable instead of executing.
if ! command -v bwrap >/dev/null 2>&1; then
  echo "[bootstrap] installing bubblewrap (generated-code sandbox)"
  if sudo -n apt-get install -y bubblewrap 2>/dev/null; then
    :
  elif apt-get install -y bubblewrap 2>/dev/null; then
    :
  else
    echo "[bootstrap] WARN: could not install bubblewrap — audits will" >&2
    echo "  fail closed (sandbox_unavailable). Fix: apt install bubblewrap," >&2
    echo "  or export LEMMA_AUDIT_HOST_EXEC=1 (unsafe, no isolation)." >&2
  fi
else
  echo "[bootstrap] bubblewrap present: $(command -v bwrap)"
fi

echo "[bootstrap] sanity: judge regression on the grokking fixture"
.venv/bin/python -m agent.cli judge --regression
echo "[bootstrap] done. Launch with: ./scripts/run_overnight.sh <paper-source> [run-name]"

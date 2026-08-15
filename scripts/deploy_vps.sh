#!/usr/bin/env bash
# Deploy the lemma pipeline to a VPS for overnight runs.
#
#   VPS_HOST=user@1.2.3.4 ./scripts/deploy_vps.sh
#   VPS_DIR defaults to ~/lemma on the VPS.
#
# Syncs the repo INCLUDING .env over SSH (encrypted), then bootstraps.
# The VPS dir is dedicated — remote files not in the repo get deleted.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${VPS_HOST:?Set VPS_HOST=user@host (optionally -p port via VPS_SSH_OPTS)}"
VPS_DIR="${VPS_DIR:-lemma}"
SSH_OPTS=(${VPS_SSH_OPTS:-})

# Sanity: never deploy without a usable LLM key.
if ! grep -Eq '^ANTHROPIC_API_KEY=.+' .env 2>/dev/null \
   && ! grep -Eq '^OPENAI_API_KEY=.+' .env 2>/dev/null; then
  echo "error: .env has no ANTHROPIC_API_KEY or OPENAI_API_KEY — fix before deploying" >&2
  exit 1
fi

rsync -az "${SSH_OPTS[@]}" --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '.ruff_cache' \
  --exclude '.pytest_cache' --exclude '.trackio' --exclude 'runs/' \
  --exclude '.DS_Store' \
  ./ "$VPS_HOST:$VPS_DIR/"

ssh "${SSH_OPTS[@]}" "$VPS_HOST" "cd '$VPS_DIR' && bash scripts/bootstrap_vps.sh"
echo "[deploy] ready. On the VPS: ./scripts/run_overnight.sh <paper-source>"

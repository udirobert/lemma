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

# Sanity: never deploy without a usable LLM key (named endpoint or legacy).
if ! grep -Eq '^(LEMMA_ENDPOINTS=.+|ANTHROPIC_API_KEY=.+|OPENAI_API_KEY=.+)' .env 2>/dev/null; then
  echo "error: .env has no LLM provider config (LEMMA_ENDPOINTS / ANTHROPIC_API_KEY / OPENAI_API_KEY) — fix before deploying" >&2
  exit 1
fi
if grep -q '^LEMMA_ENDPOINTS=' .env 2>/dev/null; then
  for name in $(grep '^LEMMA_ENDPOINTS=' .env | cut -d= -f2 | tr ',' ' '); do
    if ! grep -q "^LEMMA_${name}_API_KEY=.\+" .env; then
      echo "error: LEMMA_ENDPOINTS names $name but LEMMA_${name}_API_KEY is missing" >&2
      exit 1
    fi
  done
fi

# rsync code + config, never paper artifacts: papers/*/results, traces,
# judge reports and logbooks are RUN STATE that may be newer on the VPS
# (or mid-write by a live run). Only _staging inputs go up.
rsync -az ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} --delete \
  --exclude '.venv' --exclude '__pycache__' --exclude '.ruff_cache' \
  --exclude '.pytest_cache' --exclude '.trackio' --exclude 'runs/' \
  --exclude '.DS_Store' \
  --exclude 'papers/*/results/' --exclude 'papers/*/trace.jsonl' \
  --exclude 'papers/*/judge_report.json' --exclude 'papers/*/.trackio/' \
  --exclude 'papers/*/claims.json' \
  ./ "$VPS_HOST:$VPS_DIR/"

ssh ${SSH_OPTS[@]+"${SSH_OPTS[@]}"} "$VPS_HOST" "cd '$VPS_DIR' && bash scripts/bootstrap_vps.sh"
echo "[deploy] ready. On the VPS: ./scripts/run_overnight.sh <paper-source>"

#!/usr/bin/env bash
# Launch an overnight audit run inside tmux on the VPS (or anywhere).
#
#   ./scripts/run_overnight.sh <paper-source> [run-name]
#
# paper-source: arxiv id, openreview id, or PDF path.
# The session survives logout; output tees to runs/<name>-<ts>.log.
set -euo pipefail
cd "$(dirname "$0")/.."

: "${1:?usage: run_overnight.sh <paper-source> [run-name]}"
SRC="$1"
NAME="${2:-overnight}"
TS=$(date +%Y%m%d-%H%M%S)
LOG="runs/$NAME-$TS.log"
mkdir -p runs

if ! command -v tmux >/dev/null 2>&1; then
  echo "error: tmux required (or run: nohup .venv/bin/python -m agent.cli audit '$SRC' --no-publish > '$LOG' 2>&1 &)" >&2
  exit 1
fi

tmux new-session -d -s "lemma-$NAME" \
  "source .venv/bin/activate && nice -n 19 python -m agent.cli audit '$SRC' --no-publish 2>&1 | tee '$LOG'; echo '=== RUN FINISHED ==='; sleep 86400"

echo "[run] launched tmux session 'lemma-$NAME'"
echo "[run] log: $LOG"
echo "[run] attach:  tmux attach -t lemma-$NAME"
echo "[run] tail:    tail -f $LOG"
echo "[run] trace:   ls papers/*/trace.jsonl"

#!/bin/bash
# Round 2, unattended: tighter hotspots (C5-C7, centered on p6-Asp) +
# shorter binders (60-100 aa), 12 backbones, then organize/audit/judge.
#
# Disconnect-safety (the round-2 failure that killed the first attempt):
# remote functions write their outputs to Modal volumes inside the job
# (protein-rfd-outputs / protein-boltz-results) and calls use .spawn(),
# which is NOT cancelled when the local client disconnects. If `modal
# run` drops, this script recovers finished results via
# `modal volume get` and re-runs the idempotent stages for the rest —
# it never restarts finished work.
#
# Runs start-to-finish without supervision; check runs/round2-<ts>.log.
set -uo pipefail
cd "$(dirname "$0")/.."
WD=targets/kras-g12d-pmhc
RFD_RUN=_rfd_round2
RFD_VOL=protein-rfd-outputs
BOLTZ_VOL=protein-boltz-results

# Idle-sleep is a disconnect source; caffeinate the whole script.
if [ -z "${ROUND2_CAFFEINATED:-}" ] && command -v caffeinate >/dev/null; then
  export ROUND2_CAFFEINATED=1
  exec caffeinate -dims bash "$0" "$@"
fi

vol_has_done() { # vol dir -> 0 when _done.json exists in the volume
  modal volume ls "$1" "$2" 2>/dev/null | grep -q '_done\.json'
}

vol_get_dir() { # vol remote_dir local_dir
  mkdir -p "$3"
  modal volume get "$1" "$2" "$3" --force >/dev/null
}

echo "=== round2: modal rfd n=12 len=60-100 hotspots=C5,C6,C7 ==="
attempt=0
until vol_has_done "$RFD_VOL" "$RFD_RUN" || [ "$attempt" -ge 3 ]; do
  attempt=$((attempt + 1))
  echo "--- rfd attempt $attempt ---"
  if modal run scripts/modal_rfd.py --n 12 --binder-len 60-100 \
      --hotspots C5,C6,C7 --out "$WD/designs/$RFD_RUN" \
      --run-name "$RFD_RUN"; then
    break
  fi
  echo "rfd run dropped (attempt $attempt); spawned job may still be" \
    "running — polling volume for _done.json up to 75 min" \
    "(12 backbones ~ 55 min remote)"
  for _ in $(seq 1 75); do
    vol_has_done "$RFD_VOL" "$RFD_RUN" && break
    sleep 60
  done
done
# Recover whatever the remote job wrote, even if the client died.
vol_get_dir "$RFD_VOL" "$RFD_RUN" "$WD/designs/$RFD_RUN" 2>/dev/null || true
if ! ls "$WD/designs/$RFD_RUN"/*.pdb >/dev/null 2>&1; then
  echo "FATAL: no round-2 backbones after $attempt attempts"; exit 1
fi
echo "rfd outputs present: $(ls "$WD/designs/$RFD_RUN"/*.pdb | wc -l | tr -d ' ') backbones"

echo "=== round2: design stage (organize $RFD_RUN -> designs/) ==="
python3 -m agent.cli design "$WD" --stages design || { echo "FATAL: design stage"; exit 1; }

echo "=== round2: audit stage (boltz x3 complexes) ==="
attempt=0
until python3 -m agent.cli design "$WD" --stages audit; do
  attempt=$((attempt + 1))
  if [ "$attempt" -ge 4 ]; then
    echo "FATAL: audit stage keeps failing after $attempt attempts"; exit 1
  fi
  echo "audit dropped (attempt $attempt) — waiting 5 min for" \
    "still-running spawned predictions to land in $BOLTZ_VOL," \
    "then recovering finished ones before resubmitting the missing"
  sleep 300
  python3 - "$WD" "$BOLTZ_VOL" <<'PY'
import json, subprocess, sys
from pathlib import Path
wd, vol = Path(sys.argv[1]), sys.argv[2]
manifest = wd / "audit" / "manifest.json"
if not manifest.is_file():
    sys.exit(0)
for job in json.loads(manifest.read_text()):
    out = Path(job["out_dir"])
    if any(out.rglob("confidence_*_model_0.json")):
        continue  # already recovered / already finished locally
    ls = subprocess.run(
        ["modal", "volume", "ls", vol, job["name"]],
        capture_output=True, text=True,
    )
    if "_done.json" in ls.stdout:
        subprocess.run(
            ["modal", "volume", "get", vol, job["name"], str(out), "--force"],
            check=True, capture_output=True,
        )
        print(f"  recovered {job['name']}", flush=True)
PY
done

echo "=== round2: judge ==="
python3 -m agent.cli design "$WD" --stages judge

echo "=== round2 done ==="
cat "$WD/judge.json" | head -12

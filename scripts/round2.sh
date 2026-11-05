#!/bin/bash
# Round 2, unattended: tighter hotspots (C5-C7, centered on p6-Asp) +
# shorter binders (60-100 aa), 12 backbones, then organize/audit/judge.
# Runs start-to-finish without supervision; check runs/round2-<ts>.log.
set -uo pipefail
cd "$(dirname "$0")/.."
WD=targets/kras-g12d-pmhc

echo "=== round2: modal rfd n=12 len=60-100 hotspots=C5,C6,C7 ==="
modal run scripts/modal_rfd.py --n 12 --binder-len 60-100 \
    --hotspots C5,C6,C7 --out "$WD/designs/_rfd_round2"

echo "=== round2: design stage (organize _rfd_round2 -> designs/) ==="
python3 -m agent.cli design "$WD" --stages design

echo "=== round2: audit stage (boltz x3 complexes) ==="
python3 -m agent.cli design "$WD" --stages audit

echo "=== round2: judge ==="
python3 -m agent.cli design "$WD" --stages judge

echo "=== round2 done ==="
cat "$WD/judge.json" | head -12

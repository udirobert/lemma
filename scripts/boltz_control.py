#!/usr/bin/env python3
"""Positive control for the design audit screen.

Scores the 7OW6 TCR (alpha chain D + beta chain E — a binder we KNOW
discriminates KRAS G12D from WT on HLA-A*11:01) through the identical
3-complex boltz pipeline the audit stage uses:

    python3 scripts/boltz_control.py                 # emit manifest
    modal run scripts/modal_boltz_smoke.py --manifest <printed path>

If the known-discriminative TCR also reads decoy_leak under the
iptm-margin criterion, the SCREEN lacks the resolution to resolve a
single-atom p6 difference — and the audit's decoy_leak verdicts are
evidence about the metric, not the designs. Lemma's own rule: a screen
without a passing positive control cannot convict.
"""

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
WD = REPO / "targets" / "kras-g12d-pmhc"


def main() -> int:
    target = json.loads((WD / "target.json").read_text())
    chains = target["chains"]
    tcr = {
        "tcr_alpha": (WD / "inputs/tcr_d.fasta").read_text().splitlines()[-1],
        "tcr_beta": (WD / "inputs/tcr_e.fasta").read_text().splitlines()[-1],
    }

    jobs = []
    for comp in target["complexes"]:
        cx = comp["id"]
        parts = ["version: 1", "sequences:"]
        letter = "A"
        for key in comp["chains"]:
            parts.append(
                f"  - protein:\n      id: {letter}\n"
                f"      sequence: {chains[key]['sequence']}"
            )
            letter = chr(ord(letter) + 1)
        for seq in tcr.values():
            parts.append(f"  - protein:\n      id: {letter}\n      sequence: {seq}")
            letter = chr(ord(letter) + 1)
        jobs.append(
            {
                "name": f"control_tcr_{cx}",
                "out_dir": str(WD / "audit" / "control_tcr" / cx),
                "yaml": "\n".join(parts) + "\n",
            }
        )

    manifest = WD / "audit" / "control_tcr" / "manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(jobs, indent=1))
    print(manifest)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Build the Proteinbase submission bundle from recorded evidence.

    python3 scripts/build_submission.py [workdir]

Reads targets/<id>/audit/audit_report.json + designs/<did>/candidate.json
and emits submission/designs.csv ordered by the screen ranking rule
(best first = largest g12d-over-worst-decoy iptm margin), matching the
challenge's CSV contract: name, sequence, molecule_class. Extra evidence
columns follow — Proteinbase encourages metrics and provenance.

Never hand-edit designs.csv: it is generated from the audit trail.
"""

import csv
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def load_unit_metrics(unit: dict) -> dict:
    cx = unit["complexes"]

    def iptm(c):
        return cx.get(c, {}).get("iptm")

    g12d, wt, apo = iptm("pmhc_g12d"), iptm("pmhc_wt"), iptm("hla_apo")
    margin = None
    if g12d is not None and wt is not None and apo is not None:
        margin = round(g12d - max(wt, apo), 4)
    return {
        "iptm_g12d": g12d,
        "iptm_wt": wt,
        "iptm_apo": apo,
        "binder_peptide_iptm": cx.get("pmhc_g12d", {}).get("binder_peptide_iptm"),
        "screen_margin": margin,
    }


def main() -> int:
    wd = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "targets/kras-g12d-pmhc"
    max_designs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    report_p = wd / "audit" / "audit_report.json"
    if not report_p.is_file():
        print(f"no audit report at {report_p}", file=sys.stderr)
        return 1
    units = json.loads(report_p.read_text())["units"]

    rows = []
    for u in units:
        cand = json.loads((wd / "designs" / u["did"] / "candidate.json").read_text())
        seq = next(s for s in cand["sequences"] if s["seq_id"] == u["seq_id"])
        m = load_unit_metrics(u)
        rows.append(
            {
                "name": f"lemma-{u['did']}-{u['seq_id']}",
                "sequence": u["aa"],
                "molecule_class": "protein",
                "length_aa": len(u["aa"]),
                **m,
                "screen_verdict": u.get("verdict"),
                "mpnn_global_score": seq.get("mpnn_global_score"),
                "backbone": u["did"],
                "declared_surface": "peptide + groove mouth, centered on "
                "p6-Asp (TCR-read surface)",
            }
        )

    rows.sort(
        key=lambda r: (r["screen_margin"] is not None, r["screen_margin"]), reverse=True
    )
    rows = rows[:max_designs]

    out_dir = wd / "submission"
    out_dir.mkdir(exist_ok=True)
    csv_p = out_dir / "designs.csv"
    fields = [
        "name",
        "sequence",
        "molecule_class",
        "length_aa",
        "iptm_g12d",
        "iptm_wt",
        "iptm_apo",
        "binder_peptide_iptm",
        "screen_margin",
        "screen_verdict",
        "mpnn_global_score",
        "backbone",
        "declared_surface",
    ]
    with csv_p.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"{csv_p}: {len(rows)} ranked designs")
    return 0


if __name__ == "__main__":
    sys.exit(main())

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


def load_contact_metrics(ca_units: dict, did: str, seq_id: str) -> dict:
    """Contact-level metrics from audit/contact_audit.json (Modal-side
    structure analysis). Empty dict when the unit has no data."""
    g = ca_units.get(f"{did}_{seq_id}_pmhc_g12d", {}).get("peptide", {})
    w = ca_units.get(f"{did}_{seq_id}_pmhc_wt", {}).get("peptide", {})
    if not g and not w:
        return {}
    gp6 = g.get("p6", {}).get("n_contacts_5A")
    wp6 = w.get("p6", {}).get("n_contacts_5A")
    out = {
        "p6_contacts_g12d": gp6,
        "p6_contacts_wt": wp6,
        "p6_contact_diff": (gp6 - wp6) if gp6 is not None and wp6 is not None else None,
        "ipsae_d0_10": g.get("ipsae_d0_10"),
        "peptide_contact_fraction": g.get("peptide_contact_fraction"),
        "interface_pae_mean": g.get("interface_pae_mean"),
    }
    return out


def main() -> int:
    wd = Path(sys.argv[1]) if len(sys.argv) > 1 else REPO / "targets/kras-g12d-pmhc"
    max_designs = int(sys.argv[2]) if len(sys.argv) > 2 else 20
    report_p = wd / "audit" / "audit_report.json"
    if not report_p.is_file():
        print(f"no audit report at {report_p}", file=sys.stderr)
        return 1
    units = json.loads(report_p.read_text())["units"]

    ca_p = wd / "audit" / "contact_audit.json"
    ca_units = json.loads(ca_p.read_text())["units"] if ca_p.is_file() else {}

    rows = []
    for u in units:
        cand = json.loads((wd / "designs" / u["did"] / "candidate.json").read_text())
        seq = next(s for s in cand["sequences"] if s["seq_id"] == u["seq_id"])
        m = load_unit_metrics(u)
        rows.append(
            {
                "name": f"lemma-{u['did']}-{u['seq_id']}",
                "sequence": u["aa"],
                "molecule_class": "single_chain",
                "length_aa": len(u["aa"]),
                **m,
                **load_contact_metrics(ca_units, u["did"], u["seq_id"]),
                "screen_verdict": u.get("verdict"),
                "mpnn_global_score": seq.get("mpnn_global_score"),
                "backbone": u["did"],
                "declared_surface": "peptide + groove mouth, centered on "
                "p6-Asp (TCR-read surface)",
            }
        )

    if ca_units:
        # Contact-level ranking: the ipTM-margin screen cannot resolve the
        # p6 difference (control-proven), so rank by what can — peptide
        # engagement first (a binder that contacts nothing is a groove
        # dock, whatever its diff says), then p6 differential (g12d minus
        # wt contacts — mutation-preferring above indiscriminate above
        # wt-preferring), then g12d engagement and interface confidence.
        # Units without contact data sort last.
        def contact_key(r):
            diff = r.get("p6_contact_diff")
            return (
                diff is not None,
                (r.get("peptide_contact_fraction") or 0) > 0,
                diff if diff is not None else -999,
                r.get("p6_contacts_g12d") or 0,
                r.get("ipsae_d0_10") or 0,
                r.get("peptide_contact_fraction") or 0,
            )

        rows.sort(key=contact_key, reverse=True)
    else:
        rows.sort(
            key=lambda r: (r["screen_margin"] is not None, r["screen_margin"]),
            reverse=True,
        )
    # Portal requires unique sequences; MPNN can emit identical seqs on
    # different backbones — keep the best-ranked instance of each.
    seen, deduped = set(), []
    for r in rows:
        if r["sequence"] not in seen:
            seen.add(r["sequence"])
            deduped.append(r)
    rows = deduped[:max_designs]

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
        "p6_contacts_g12d",
        "p6_contacts_wt",
        "p6_contact_diff",
        "ipsae_d0_10",
        "peptide_contact_fraction",
        "interface_pae_mean",
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

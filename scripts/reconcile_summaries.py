#!/usr/bin/env python3
"""Reconcile per-claim audit_summary.json with the authoritative
audit_report.json after the 2026-08-16 deploy clobber incident.

The incident: a deploy with rsync --delete synced the laptop's round-1 state
over the VPS round-2 workdir before sync-back. audit_report.json survived
(receiver-newer), but some per-claim audit_summary.json files were left
holding round-1 verdicts. Round 2's final verdicts (authoritative, also in
papers/jmlr-22-1228-ca-grokking/round2.log): all six claims INCONCLUSIVE.

This script rewrites each audit_summary.json from the report's embedded
summaries so the judge and evidence stages see one consistent state.
"""

from __future__ import annotations

import json
from pathlib import Path

WORKDIR = Path(__file__).resolve().parent.parent / "papers/jmlr-22-1228-ca-grokking"


def main() -> None:
    report = json.loads(
        (WORKDIR / "results/audit_report.json").read_text(encoding="utf-8")
    )
    for outcome in report["claims"]:
        cid = outcome["id"]
        summary = outcome.get("summary") or {
            "claim_id": cid,
            "status": outcome["status"],
            "metrics": {"attempts_used": outcome.get("attempts", 0)},
            "notes": "summary lost in deploy clobber; verdict recovered from round2.log",
        }
        claim_dir = WORKDIR / "results" / cid.lower()
        claim_dir.mkdir(parents=True, exist_ok=True)
        path = claim_dir / "audit_summary.json"
        old = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
        path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        changed = old is None or old.get("status") != summary.get("status")
        print(
            f"{cid}: {old.get('status') if old else 'MISSING'} -> "
            f"{summary.get('status')}{' (reconciled)' if changed else ''}"
        )


if __name__ == "__main__":
    main()

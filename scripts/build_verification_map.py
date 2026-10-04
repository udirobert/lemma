#!/usr/bin/env python3
"""Build a corpus verification map: paper -> claim -> verdict.

Scans papers/*/claims.json + results/audit_report.json and emits
papers/_verification_map.json + a markdown summary for the site/submission.
"""

import glob
import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def load(p):
    try:
        with open(p) as f:
            return json.load(f)
    except Exception:
        return None


def main():
    papers = []
    for d in sorted(glob.glob(os.path.join(ROOT, "papers", "arxiv-*"))):
        claims = load(os.path.join(d, "claims.json"))
        report = load(os.path.join(d, "results", "audit_report.json"))
        meta = load(os.path.join(d, "meta.json")) or {}
        if not claims:
            continue
        verdicts = {}
        if report:
            for c in report.get("claims", report if isinstance(report, list) else []):
                verdicts[c.get("id") or c.get("claim_id")] = c.get("status") or c.get(
                    "verdict"
                )
        rows = []
        for cl in claims:
            cid = cl["id"]
            status = verdicts.get(cid, "not_audited")
            failed = len(
                glob.glob(
                    os.path.join(d, "results", cid.lower(), "run_attempt*.failed.json")
                )
            )
            rows.append(
                {
                    "id": cid,
                    "kind": cl.get("kind", "?"),
                    "status": status,
                    "failed_attempts": failed,
                    "agent_generated": cl.get("source") == "agent_generated"
                    or cl.get("kind") == "generated-hypothesis",
                    "title": cl.get("title", cl.get("statement", ""))[:120],
                }
            )
        papers.append(
            {
                "id": os.path.basename(d),
                "slug": meta.get("slug", os.path.basename(d).replace("arxiv-", "")),
                "title": meta.get("title", ""),
                "claims": rows,
            }
        )

    # tallies
    tally = {}
    for p in papers:
        for c in p["claims"]:
            tally[c["status"]] = tally.get(c["status"], 0) + 1
    out = {
        "generated_from": "papers/*/claims.json + results/audit_report.json",
        "papers": papers,
        "tally": tally,
        "counts": {
            "papers": len(papers),
            "claims": sum(len(p["claims"]) for p in papers),
            "agent_generated": sum(
                1 for p in papers for c in p["claims"] if c["agent_generated"]
            ),
        },
    }
    dst = os.path.join(ROOT, "papers", "_verification_map.json")
    with open(dst, "w") as f:
        json.dump(out, f, indent=1)
    print(f"wrote {dst}")
    for p in papers:
        print(f"\n{p['id']} — {p['title'][:60]}")
        for c in p["claims"]:
            tag = " [agent-gen]" if c["agent_generated"] else ""
            print(
                f"  {c['id']:>3} {c['status']:<14} fails={c['failed_attempts']}{tag}  {c['title'][:60]}"
            )
    print("\ntally:", json.dumps(tally))


if __name__ == "__main__":
    main()

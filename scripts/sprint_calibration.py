#!/usr/bin/env python3
"""Sprint analysis: how solid are Lemma's own audit verdicts?

Scans papers/*/ workdirs and produces:
  - verdict distribution per paper + corpus
  - self-distrust signal counts (control failures, summary rejections,
    reference disagreements, multi-attempt claims, trace integrity events)
  - test-retest agreement on same-paper/same-claims pairs audited by
    different model backends
  - figures for the sprint report under sprints/epistemics-2026/figures/

Usage: scripts/sprint_calibration.py [--out sprints/epistemics-2026]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "sprints" / "epistemics-2026"

STATUS_ORDER = ["supported", "falsified", "inconclusive", "not_audited"]
COLORS = {
    "supported": "#2a9d5c",
    "falsified": "#d64545",
    "inconclusive": "#e0a63c",
    "not_audited": "#9aa0a6",
}


def scan(papers: Path) -> dict:
    rows = []
    for d in sorted(papers.iterdir()):
        if not d.is_dir() or d.name.startswith("_"):
            continue
        rp = d / "results" / "audit_report.json"
        if not rp.is_file():
            continue
        rep = json.loads(rp.read_text())
        jr = d / "judge_report.json"
        judge = json.loads(jr.read_text()).get("verdict") if jr.is_file() else None
        claims = []
        for c in rep.get("claims", []):
            s = c.get("summary") or {}
            claims.append(
                {
                    "id": c.get("id"),
                    "status": c.get("status"),
                    "attempts": c.get("attempts", 0),
                    "control_pass": s.get("metrics", {}).get("control_pass"),
                    "source": c.get("source"),
                    "reference_checked": bool(c.get("reference_checked")),
                    "reference_disagreement": bool(c.get("reference_disagreement")),
                }
            )
        # trace-derived signals
        trace_events = Counter()
        tp = d / "trace.jsonl"
        if tp.is_file():
            for line in tp.read_text().splitlines():
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if r.get("event") in {
                    "summary_rejected",
                    "attempt_failed",
                    "reviewer_reference_executed",
                    "attempt_rejected",
                }:
                    trace_events[r["event"]] += 1
        rows.append(
            {
                "workdir": d.name,
                "judge_verdict": judge,
                "claims": claims,
                "trace_events": dict(trace_events),
            }
        )
    return rows


def test_retest(rows: list[dict], pairs: list[tuple[str, str]]) -> list[dict]:
    by_name = {r["workdir"]: r for r in rows}
    out = []
    for a, b in pairs:
        if a not in by_name or b not in by_name:
            continue
        ca = {c["id"]: c["status"] for c in by_name[a]["claims"]}
        cb = {c["id"]: c["status"] for c in by_name[b]["claims"]}
        shared = sorted(set(ca) & set(cb))
        agree = sum(1 for cid in shared if ca[cid] == cb[cid])
        flips = [
            {"claim": cid, "run_a": ca[cid], "run_b": cb[cid]}
            for cid in shared
            if ca[cid] != cb[cid]
        ]
        out.append(
            {
                "paper_a": a,
                "paper_b": b,
                "shared_claims": len(shared),
                "agree": agree,
                "agreement": round(agree / max(len(shared), 1), 3),
                "flips": flips,
            }
        )
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--papers", default=str(REPO / "papers"))
    args = ap.parse_args()
    out = Path(args.out)
    (out / "figures").mkdir(parents=True, exist_ok=True)

    rows = scan(Path(args.papers))
    all_claims = [c for r in rows for c in r["claims"]]
    dist = Counter(c["status"] for c in all_claims)
    signals = {
        "claims_total": len(all_claims),
        "supported": dist.get("supported", 0),
        "falsified": dist.get("falsified", 0),
        "inconclusive": dist.get("inconclusive", 0),
        "not_audited": dist.get("not_audited", 0),
        "decisive_rate": round(
            (dist.get("supported", 0) + dist.get("falsified", 0))
            / max(len(all_claims), 1),
            3,
        ),
        "multi_attempt": sum(1 for c in all_claims if c["attempts"] > 1),
        "control_failed": sum(1 for c in all_claims if c["control_pass"] is False),
        "reference_checked": sum(1 for c in all_claims if c["reference_checked"]),
        "reference_disagreement": sum(
            1 for c in all_claims if c["reference_disagreement"]
        ),
        "summary_rejected_events": sum(
            r["trace_events"].get("summary_rejected", 0) for r in rows
        ),
        "attempt_failed_events": sum(
            r["trace_events"].get("attempt_failed", 0) for r in rows
        ),
    }

    pairs = test_retest(
        rows,
        [
            ("arxiv-2510.10981", "arxiv-2510.10981-beam"),
            ("jmlr-22-1228-ca-grokking", "jmlr-22-1228-ca-grokking-beam"),
        ],
    )

    analysis = {
        "papers": [
            {
                "workdir": r["workdir"],
                "judge_verdict": r["judge_verdict"],
                "verdicts": dict(Counter(c["status"] for c in r["claims"])),
                "trace_events": r["trace_events"],
            }
            for r in rows
        ],
        "corpus": signals,
        "test_retest": pairs,
    }
    (out / "analysis.json").write_text(json.dumps(analysis, indent=2) + "\n")

    # Figure 1: per-paper verdict distribution (stacked bars)
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    names = [
        p["workdir"].replace("arxiv-", "ax:").replace("-ca-grokking", "-grok")
        for p in analysis["papers"]
    ]
    bottoms = [0] * len(names)
    for status in STATUS_ORDER:
        vals = [p["verdicts"].get(status, 0) for p in analysis["papers"]]
        ax.bar(
            names, vals, bottom=bottoms, color=COLORS[status], label=status, width=0.65
        )
        bottoms = [b + v for b, v in zip(bottoms, vals, strict=True)]
    ax.set_ylabel("claims")
    ax.legend(
        ncol=4,
        fontsize=8,
        frameon=False,
        loc="upper center",
        bbox_to_anchor=(0.5, 1.18),
    )
    ax.spines[["top", "right"]].set_visible(False)
    plt.setp(ax.get_xticklabels(), rotation=35, ha="right", fontsize=7.5)
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig1_verdict_distribution.png", dpi=180)
    plt.close(fig)

    # Figure 2: test-retest agreement heatmap (status_a x status_b across pairs)
    grid = {s: dict.fromkeys(STATUS_ORDER, 0) for s in STATUS_ORDER}
    agree_cells = 0
    for pr in pairs:
        a_name, b_name = pr["paper_a"], pr["paper_b"]
        ca = {
            c["id"]: c["status"]
            for c in next(r for r in rows if r["workdir"] == a_name)["claims"]
        }
        cb = {
            c["id"]: c["status"]
            for c in next(r for r in rows if r["workdir"] == b_name)["claims"]
        }
        for cid in sorted(set(ca) & set(cb)):
            grid[ca[cid]][cb[cid]] += 1
            if ca[cid] == cb[cid]:
                agree_cells += 1
    fig, ax = plt.subplots(figsize=(3.6, 3.2))
    mat = [[grid[s][t] for t in STATUS_ORDER] for s in STATUS_ORDER]
    ax.imshow(mat, cmap="Blues", vmin=0)
    ax.set_xticks(range(4), STATUS_ORDER, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(4), STATUS_ORDER, fontsize=8)
    ax.set_xlabel("run B verdict")
    ax.set_ylabel("run A verdict")
    for i in range(4):
        for j in range(4):
            if mat[i][j]:
                ax.text(
                    j,
                    i,
                    mat[i][j],
                    ha="center",
                    va="center",
                    color="white" if mat[i][j] > 2 else "#1a3d63",
                    fontsize=10,
                )
    ax.set_title(
        f"verdict agreement {agree_cells}/{sum(sum(r) for r in mat)} claims", fontsize=9
    )
    fig.tight_layout()
    fig.savefig(out / "figures" / "fig2_test_retest.png", dpi=180)
    plt.close(fig)

    print(json.dumps({"corpus": signals, "test_retest": pairs}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

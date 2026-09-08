#!/usr/bin/env python3
"""Build papers/_index.json — the machine-readable registry of all audits.

Walks every papers/<dir>/meta.json and merges it with derived data:
  - claims.json                 → claim list (id, title, testable)
  - results/audit_report.json   → per-claim verdict + attempt count
  - judge_report.json           → verdict + score
  - trace.jsonl                 → event / LLM-call / tool-run counts, wall time
  - results/**/*.png            → figure inventory

The index is the single source of truth consumed by scripts/build_trace_data.py
and scripts/build_site_data.py. Re-run it whenever a paper directory changes.

Usage:
    python3 scripts/build_paper_index.py [--papers-dir papers]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

STATUS_ORDER = ["supported", "falsified", "inconclusive", "not_audited"]


def load_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        print(f"[warn] cannot read {path}: {exc}", file=sys.stderr)
        return None


def trace_stats(trace_path: Path) -> dict:
    """Aggregate counters over a raw JSONL trace."""
    if not trace_path.is_file():
        return {"n_events": 0, "n_llm": 0, "n_tool": 0, "n_note": 0, "wall_min": 0.0}
    events = [
        json.loads(line)
        for line in trace_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    # Wall time = span of timestamps when available, else sum of "done" elapsed.
    wall_min = 0.0
    done_elapsed = [e.get("elapsed_s", 0) for e in events if e.get("event") == "done"]
    if done_elapsed:
        wall_min = max(done_elapsed) / 60.0
    return {
        "n_events": len(events),
        "n_llm": sum(1 for e in events if e.get("event") == "llm_call"),
        "n_tool": sum(1 for e in events if e.get("event") == "tool_run"),
        "n_note": sum(1 for e in events if e.get("event") == "note"),
        "wall_min": round(wall_min, 1),
    }


def build_entry(paper_dir: Path, meta: dict, prior: dict | None = None) -> dict:
    """Merge one paper workdir into an index entry.

    `prior` is the previous committed entry for the same dir (if any). Some
    artifacts are deliberately local-only (e.g. 5nNNVY8NW4-grokking gitignores
    results/, several trace.jsonl files are not committed), so a fresh clone
    cannot recompute them. When a local artifact is absent we carry the prior
    values forward instead of silently downgrading the published registry.
    """
    slug = meta.get("slug") or paper_dir.name
    claims = load_json(paper_dir / "claims.json") or []
    report = load_json(paper_dir / "results" / "audit_report.json") or {}
    judge = load_json(paper_dir / "judge_report.json")
    prior = prior or {}
    prior_claims = {c["id"]: c for c in prior.get("claims", [])}
    has_report = bool(report.get("claims"))

    report_by_id = {c["id"]: c for c in report.get("claims", [])}

    def claim_entry(cid: str, src: dict) -> dict:
        """Normalize a claim row from either a local report or the prior index."""
        osummary = src.get("summary") or {}
        metrics = osummary.get("metrics") or {}
        return {
            "id": cid,
            "title": src.get("title", ""),
            "testable": src.get("testable", True),
            "status": src.get("status", "not_audited"),
            "attempts": src.get("attempts", 0),
            # surfaced for the site's claim dossiers — the auditor's own
            # words and numbers, not a marketing rewrite
            "notes": osummary.get("notes", ""),
            "metrics": metrics,
            "control_pass": metrics.get("control_pass"),
        }

    merged = []
    seen_ids: set[str] = set()

    # 1) local claims.json — verdicts from the local report, falling back to
    #    the prior committed outcome when the report is absent (fresh clone)
    for c in claims:
        cid = c["id"]
        seen_ids.add(cid)
        title = c.get("title", c.get("claim", ""))
        if has_report and cid in report_by_id:
            merged.append(
                claim_entry(cid, {**report_by_id[cid], "title": title})
            )
        elif cid in prior_claims:
            merged.append(
                claim_entry(cid, {**prior_claims[cid], "title": title})
            )
        else:
            merged.append(claim_entry(cid, {"title": title}))

    # 2) claims present in the local report but missing from claims.json
    for cid, outcome in report_by_id.items():
        if cid not in seen_ids:
            seen_ids.add(cid)
            merged.append(claim_entry(cid, outcome))

    # 3) claims only in the prior index (e.g. a gitignored results/ tree) —
    #    keep them so the registry never silently loses claims
    for cid, pclaim in prior_claims.items():
        if cid not in seen_ids:
            seen_ids.add(cid)
            merged.append(claim_entry(cid, pclaim))

    summary = {s: sum(1 for m in merged if m["status"] == s) for s in STATUS_ORDER}

    figures = []
    results_dir = paper_dir / "results"
    if results_dir.is_dir():
        seen = set()
        for png in sorted(results_dir.rglob("*.png")):
            # dedupe mirrored files (results root vs results/<cid>/)
            if png.name.lower() in seen:
                continue
            seen.add(png.name.lower())
            figures.append(str(png.relative_to(paper_dir)))
    # a gitignored results/ tree yields no local figures — keep the prior
    # inventory rather than wiping the published figure gallery
    if not figures and prior.get("figures"):
        figures = list(prior["figures"])

    trace_rel = meta.get("trace", "trace.jsonl")
    stats = trace_stats(paper_dir / trace_rel)
    stats["path"] = f"papers/{paper_dir.name}/{trace_rel}"
    prior_trace = prior.get("trace") or {}
    if stats["n_events"] == 0 and prior_trace.get("n_events", 0) > 0:
        # trace.jsonl is not always committed — carry the prior stats forward
        stats = {**prior_trace}
    prior_judge = prior.get("judge")
    if judge is None and prior_judge:
        judge = prior_judge

    return {
        "slug": slug,
        "dir": paper_dir.name,
        "title": meta.get("title", paper_dir.name),
        "source_label": meta.get("source_label", ""),
        "source_url": meta.get("source_url"),
        "arxiv_url": meta.get("arxiv_url"),
        "audited_at": meta.get("audited_at"),
        "blurb": meta.get("blurb", ""),
        "role": meta.get("role", "audit"),
        "status": meta.get("status", "in_progress"),
        "links": meta.get("links", {}),
        "xylo": meta.get("xylo", []),
        "claims": merged,
        "summary": summary,
        "judge": (
            {
                "verdict": judge.get("verdict"),
                "score": judge.get("score"),
                # full rubric so the site can show *why* the trail is trusted
                "rubric": judge.get("rubric", {}),
            }
            if judge
            else None
        ),
        "trace": stats,
        "figures": figures,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--papers-dir", default="papers")
    args = parser.parse_args()

    papers_dir = REPO / args.papers_dir
    prior_index = load_json(papers_dir / "_index.json") or {}
    prior_by_dir = {p.get("dir"): p for p in prior_index.get("papers", [])}
    entries = []
    for paper_dir in sorted(papers_dir.iterdir()):
        if not paper_dir.is_dir() or paper_dir.name.startswith("_"):
            continue
        meta = load_json(paper_dir / "meta.json")
        if meta is None:
            print(f"[skip] {paper_dir.name}: no meta.json", file=sys.stderr)
            continue
        entries.append(build_entry(paper_dir, meta, prior_by_dir.get(paper_dir.name)))

    index = {
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "n_papers": len(entries),
        "papers": entries,
    }
    out = papers_dir / "_index.json"
    out.write_text(
        json.dumps(index, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    print(f"[ok] {len(entries)} papers -> {out}")
    for e in entries:
        s = e["summary"]
        j = e["judge"]["verdict"] if e["judge"] else "no-judge"
        print(
            f"  {e['slug']:<16} {s['supported']}S/{s['falsified']}F/"
            f"{s['inconclusive']}I/{s['not_audited']}NA  judge={j}  "
            f"trace={e['trace']['n_events']}ev"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())

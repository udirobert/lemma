#!/usr/bin/env python3
"""Build site data from papers/_index.json.

Outputs:
- site/src/data/papers.json   : data consumed by Astro pages/components
- site/public/papers/<slug>/figures/*.png : copied evidence figures

Run order:
    python scripts/build_paper_index.py
    python scripts/build_site_data.py

Figures are copied by basename (first found wins) to avoid duplicate files
from mirrored result folders. Large files are skipped to keep the site lean.
"""

from __future__ import annotations

import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
INDEX_PATH = REPO / "papers" / "_index.json"
SITE_DATA_DIR = REPO / "site" / "src" / "data"
PUBLIC_PAPERS_DIR = REPO / "site" / "public" / "papers"
MAX_FIGURE_BYTES = 2 * 1024 * 1024  # 2 MB per figure is plenty for web


def failures_preserved(trace_path: Path) -> int:
    """Count preserved failures: failed attempts + nonzero-exit tool runs."""
    if not trace_path.is_file():
        return 0
    n = 0
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        try:
            ev = json.loads(line)
        except json.JSONDecodeError:
            continue
        if ev.get("event") == "attempt_failed" or (
            ev.get("event") == "tool_run" and ev.get("exit_code") not in (0, None)
        ):
            n += 1
    return n


def status_to_state(status: str) -> str:
    s = (status or "").lower()
    if s == "supported":
        return "on"
    if s == "falsified":
        return "fail"
    if s == "inconclusive":
        return "dim"
    return "dim"


def build() -> int:
    if not INDEX_PATH.is_file():
        print(
            "papers/_index.json missing — run scripts/build_paper_index.py first",
            file=sys.stderr,
        )
        return 1

    index = json.loads(INDEX_PATH.read_text(encoding="utf-8"))
    # Prior published site data — used to keep figures/failure counts when a
    # fresh clone lacks gitignored local artifacts (e.g. grokking-ridge
    # results/ tree). Prevents a rebuild from silently downgrading the site.
    prior_site = {}
    if SITE_DATA_DIR.joinpath("papers.json").is_file():
        prior_site = json.loads(
            SITE_DATA_DIR.joinpath("papers.json").read_text(encoding="utf-8")
        )
    prior_by_slug = {p.get("slug"): p for p in prior_site.get("papers", [])}
    papers_out = []

    for paper in index["papers"]:
        slug = paper["slug"]
        paper_dir = REPO / "papers" / paper["dir"]
        fig_dir = PUBLIC_PAPERS_DIR / slug / "figures"

        # Claims enriched for UI
        status_by_id = {c["id"]: c["status"] for c in paper.get("claims", [])}
        xylo = [
            {
                **bar,
                "state": status_to_state(status_by_id.get(bar.get("claim", ""), "")),
            }
            for bar in paper.get("xylo", [])
        ]

        copied: list[str] = []
        seen: set[str] = set()
        prior_figs_by_name: dict[str, str] = {}
        for prior_path in prior_by_slug.get(slug, {}).get("figures", []):
            prior_figs_by_name[prior_path.rsplit("/", 1)[-1].lower()] = prior_path
        for rel in paper.get("figures", []):
            src = paper_dir / rel
            name = src.name.lower()
            if name in seen:
                continue
            if src.is_file():
                if src.stat().st_size > MAX_FIGURE_BYTES:
                    print(
                        f"[skip-large] {slug}: {rel} ({src.stat().st_size} bytes)",
                        file=sys.stderr,
                    )
                    continue
                fig_dir.mkdir(parents=True, exist_ok=True)
                dst = fig_dir / src.name
                shutil.copy2(src, dst)
                seen.add(name)
                copied.append(f"/papers/{slug}/figures/{src.name}")
                continue
            # Figure committed in site/public but absent from this clone's
            # workdir (gitignored results/). Keep the already-published path.
            prior_site_path = prior_figs_by_name.get(name)
            if (
                prior_site_path
                and (PUBLIC_PAPERS_DIR / slug / "figures" / src.name).is_file()
            ):
                seen.add(name)
                copied.append(prior_site_path)

        papers_out.append(
            {
                "slug": slug,
                "dir": paper["dir"],
                "title": paper["title"],
                "authors": paper.get("authors"),
                "source_label": paper.get("source_label", ""),
                "source_url": paper.get("source_url"),
                "arxiv_url": paper.get("arxiv_url"),
                "audited_at": paper.get("audited_at"),
                "blurb": paper.get("blurb", ""),
                "role": paper.get("role", "audit"),
                "status": paper.get("status", "unknown"),
                "links": paper.get("links", {}),
                "claims": paper.get("claims", []),
                "summary": paper.get("summary", {}),
                "judge": paper.get("judge"),
                "trace": paper.get("trace", {}),
                "failures_preserved": failures_preserved(
                    REPO
                    / paper.get("trace", {}).get(
                        "path", f"papers/{paper['dir']}/trace.jsonl"
                    )
                )
                or prior_by_slug.get(slug, {}).get("failures_preserved", 0),
                "xylo": xylo,
                "figures": copied,
            }
        )

    totals = {
        "papers": len(papers_out),
        "claims": sum(
            p["summary"].get("supported", 0)
            + p["summary"].get("falsified", 0)
            + p["summary"].get("inconclusive", 0)
            + p["summary"].get("not_audited", 0)
            for p in papers_out
        ),
        "supported": sum(p["summary"].get("supported", 0) for p in papers_out),
        "inconclusive": sum(p["summary"].get("inconclusive", 0) for p in papers_out),
        "falsified": sum(p["summary"].get("falsified", 0) for p in papers_out),
        "not_audited": sum(p["summary"].get("not_audited", 0) for p in papers_out),
        "audited": sum(
            p["summary"].get("supported", 0)
            + p["summary"].get("falsified", 0)
            + p["summary"].get("inconclusive", 0)
            for p in papers_out
        ),
        "failures_preserved": sum(p.get("failures_preserved", 0) for p in papers_out),
        "logbooks": sum(1 for p in papers_out if (p.get("links") or {}).get("logbook")),
        # cost-of-honesty aggregates — the site discloses what auditing costs
        "llm_calls": sum(p.get("trace", {}).get("n_llm", 0) for p in papers_out),
        "tool_runs": sum(p.get("trace", {}).get("n_tool", 0) for p in papers_out),
        "wall_min": round(
            sum(p.get("trace", {}).get("wall_min", 0) for p in papers_out), 1
        ),
        "attempts": sum(
            c.get("attempts", 0) for p in papers_out for c in p.get("claims", [])
        ),
    }

    SITE_DATA_DIR.mkdir(parents=True, exist_ok=True)
    out_path = SITE_DATA_DIR / "papers.json"
    out_path.write_text(
        json.dumps(
            {
                "generated_at": index.get("generated_at"),
                "totals": totals,
                "papers": papers_out,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"wrote {out_path} ({len(papers_out)} papers)")
    for p in papers_out:
        print(f"  {p['slug']}: {len(p['figures'])} figures copied")
    return 0


if __name__ == "__main__":
    result = build()
    if result == 0:
        sys.path.insert(0, str(REPO))
        from agent.audit_room import export_bundle

        print(f"wrote {export_bundle(REPO)}")
    sys.exit(result)

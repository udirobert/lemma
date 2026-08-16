"""Lemma CLI — wire the pipeline: extract -> audit -> evidence -> judge.

Usage:
  python -m agent.cli audit <source>            # full pipeline
  python -m agent.cli audit <source> --stages extract,judge
  python -m agent.cli judge <workdir>           # judge an existing run
  python -m agent.cli judge --regression        # judge the grokking fixture

<source> is an arxiv id (2601.19791), openreview id, or local PDF path.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PAPERS_DIR = REPO_ROOT / "papers"
GROKKING_DIR = PAPERS_DIR / "5nNNVY8NW4-grokking"

STAGES = ("extract", "audit", "evidence", "judge")


def main() -> int:
    parser = argparse.ArgumentParser(prog="lemma", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_audit = sub.add_parser("audit", help="run pipeline stages on a paper")
    p_audit.add_argument("source", help="arxiv id, openreview id, or PDF path")
    p_audit.add_argument(
        "--stages",
        default=",".join(STAGES),
        help=f"comma-separated subset of {STAGES}",
    )
    p_audit.add_argument("--workdir", help="paper dir (default papers/<id>)")
    p_audit.add_argument(
        "--claims",
        help="comma-separated claim ids to (re-)audit only; merges into the "
        "existing report. Combine with results/<cid>/feedback.md for "
        "human-in-the-loop corrections.",
    )
    p_audit.add_argument(
        "--no-publish",
        action="store_true",
        help="do not publish logbook even if configured",
    )

    p_judge = sub.add_parser("judge", help="judge an existing run")
    p_judge.add_argument("workdir", nargs="?", help="paper dir to judge")
    p_judge.add_argument("--trace", help="trace jsonl path")
    p_judge.add_argument(
        "--regression",
        action="store_true",
        help="judge the grokking regression fixture",
    )

    p_search = sub.add_parser(
        "search", help="find papers via Firecrawl (arxiv fallback if offline)"
    )
    p_search.add_argument("query", help="natural-language query")
    p_search.add_argument("--limit", type=int, default=8)
    p_search.add_argument(
        "--research",
        action="store_true",
        help="use Firecrawl life-science research index (needs FIRECRAWL_API_KEY)",
    )

    args = parser.parse_args()
    if args.cmd == "judge":
        return _judge(args)
    if args.cmd == "search":
        return _search(args)
    return _audit(args)


def _search(args: argparse.Namespace) -> int:
    from agent import paperclip
    from agent.firecrawl import search

    hits: list[dict] = []
    if paperclip.available()[0]:
        hits = paperclip.search(args.query, args.limit)
        if hits:
            print(f"[lemma] via Paperclip (GXL corpus): {len(hits)} hits")
    if not hits:
        hits = search(args.query, args.limit, research=args.research)
    if not hits:
        print("[lemma] no results")
        return 1
    for i, h in enumerate(hits, 1):
        print(f"{i}. {h['title']}\n   {h['url']}\n   {h['description'][:160]}\n")
    return 0


def _audit(args: argparse.Namespace) -> int:
    from agent import evidence as evidence_mod
    from agent.auditor import audit_all
    from agent.extract import extract
    from agent.llm import model_name
    from agent.papers import resolve
    from agent.traces import Trace

    stages = [s.strip() for s in args.stages.split(",") if s.strip() in STAGES]
    run_id = time.strftime("%Y%m%d-%H%M%S")

    paper = resolve(
        args.source,
        workdir=Path(args.workdir) if args.workdir else PAPERS_DIR / "_staging",
    )
    workdir = Path(args.workdir) if args.workdir else PAPERS_DIR / paper["paper_id"]
    workdir.mkdir(parents=True, exist_ok=True)
    # move staging pdf into workdir if needed
    pdf = Path(paper["pdf_path"])
    if pdf.parent != workdir:
        moved = workdir / pdf.name
        if not moved.exists():
            moved.write_bytes(pdf.read_bytes())
        paper["pdf_path"] = str(moved)

    trace = Trace(f"{run_id}-{paper['paper_id']}", workdir / "trace.jsonl")
    trace.log(
        "pipeline",
        "start",
        source=args.source,
        stages=stages,
        workdir=str(workdir),
        model=model_name(),
    )
    print(f"[lemma] run {trace.run_id} | stages={','.join(stages)} | {workdir}")

    claims: list[dict] = []
    report: dict = {"claims": []}
    claims_path = workdir / "claims.json"
    report_path = workdir / "results" / "audit_report.json"

    if "extract" in stages:
        claims = extract(
            paper["paper_id"], paper["title_hint"], paper["text"], workdir, trace
        )
        print(
            f"[lemma] extracted {len(claims)} claims "
            f"({sum(1 for c in claims if c['testable'])} testable)"
        )
    elif claims_path.is_file():
        import json

        claims = json.loads(claims_path.read_text(encoding="utf-8"))

    if "audit" in stages:
        if not claims:
            print("[lemma] no claims; run extract first", file=sys.stderr)
            return 1
        only = None
        if getattr(args, "claims", None):
            only = {c.strip() for c in args.claims.split(",") if c.strip()}
            print(f"[lemma] re-auditing subset: {sorted(only)}")
        report = audit_all(claims, paper["text"], workdir, trace, only=only)
        for c in report["claims"]:
            print(
                f"[lemma]   {c['id']}: {c['status']} ({c.get('attempts', 0)} attempts)"
            )
    elif report_path.is_file():
        import json

        report = json.loads(report_path.read_text(encoding="utf-8"))

    if "evidence" in stages:
        if not claims:
            print("[lemma] no claims for evidence stage", file=sys.stderr)
            return 1
        meta = trace.summary()
        meta.update(model=model_name(), wall_min=round(meta["wall_s"] / 60, 1))
        evidence_mod.build_logbook(
            workdir, paper["title_hint"], claims, report, trace, meta
        )
        print(f"[lemma] logbook assembled at {workdir}/.trackio")

    if "judge" in stages:
        from agent.judge import judge_run, render_markdown

        result = judge_run(workdir, trace.path, relative_paths=True)
        (workdir / "judge_report.json").write_text(
            __import__("json").dumps(result, indent=2), encoding="utf-8"
        )
        print(render_markdown(result))
        if result["verdict"] == "FAIL":
            return 2

    trace.log("pipeline", "done", stages=stages)
    return 0


def _judge(args: argparse.Namespace) -> int:
    import json

    from agent.judge import judge_run, render_markdown

    if args.regression:
        workdir = GROKKING_DIR
        trace_path = workdir / "results" / "trace.jsonl"
    else:
        workdir = Path(args.workdir or ".")
        trace_path = Path(args.trace) if args.trace else None

    result = judge_run(workdir, trace_path, relative_paths=True)
    (workdir / "judge_report.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    print(render_markdown(result))
    return 0 if result["verdict"] != "FAIL" else 2


if __name__ == "__main__":
    raise SystemExit(main())

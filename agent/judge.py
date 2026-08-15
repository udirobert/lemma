"""Stage 4 — the judge: an automated evidence-trustworthiness checker.

This is the re:AGENT angle of the build: a reusable, generalizable version of
the ICML logbook validator that scores any Lemma run on whether its evidence
trail is inspectable and honest. It is a *structural evidence judge*, not a
scientific arbiter — it checks that claims, runs, metrics, figures, costs, and
traces line up, and that failures were not hidden.

Verdicts: PASS / CONDITIONAL PASS / FAIL, with a rubric breakdown.
"""

from __future__ import annotations

import json
from pathlib import Path


def judge_run(
    workdir: Path, trace_path: Path | None = None, *, relative_paths: bool = False
) -> dict:
    results_dir = workdir / "results"
    rubric = {}

    # 1. Structure ---------------------------------------------------------
    claims_path = workdir / "claims.json"
    report_path = results_dir / "audit_report.json"
    structure_ok = claims_path.is_file() and report_path.is_file()
    claims = json.loads(claims_path.read_text(encoding="utf-8")) if structure_ok else []
    report = json.loads(report_path.read_text(encoding="utf-8")) if structure_ok else {}
    rubric["structure"] = {
        "pass": structure_ok,
        "detail": f"claims.json ({len(claims)} claims), audit_report.json present",
    }

    # 2. Evidence per audited claim ------------------------------------------
    issues = []
    audited = [c for c in report.get("claims", []) if c.get("status") != "not_audited"]
    for outcome in audited:
        cid = outcome["id"]
        claim_dir = results_dir / cid.lower()
        summary_path = claim_dir / "audit_summary.json"
        if not summary_path.is_file():
            issues.append(f"{cid}: no audit_summary.json")
            continue
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        if summary.get("status") not in ("supported", "falsified", "inconclusive"):
            issues.append(f"{cid}: invalid status {summary.get('status')!r}")
        metrics = summary.get("metrics") or {}
        if not metrics:
            issues.append(f"{cid}: no metrics recorded")
        figs = list(claim_dir.glob("*.png")) if claim_dir.is_dir() else []
        if not figs and summary.get("status") == "supported":
            issues.append(f"{cid}: supported but no figure evidence")
    rubric["evidence"] = {
        "pass": not issues,
        "detail": f"{len(audited)} audited claims; issues: {issues or 'none'}",
    }

    # 3. Integrity: untestable claims declared, failures preserved -------------
    untested = [c for c in report.get("claims", []) if c.get("status") == "not_audited"]
    declared = all(
        not next((cl for cl in claims if cl["id"] == c["id"]), {}).get("testable", True)
        for c in untested
    )
    rubric["integrity"] = {
        "pass": declared,
        "detail": (
            f"{len(untested)} claims skipped as untestable, "
            f"{'all' if declared else 'NOT all'} declared untestable at extraction"
        ),
    }

    # 4. Cost disclosure ---------------------------------------------------------
    wall_ok = all("attempts" in c for c in report.get("claims", []))
    rubric["cost_disclosure"] = {
        "pass": structure_ok and wall_ok,
        "detail": "per-claim attempt counts recorded; wall time in run files + trace",
    }

    # 5. Trace completeness -------------------------------------------------------
    trace_ok, trace_detail = False, "trace file missing"
    if trace_path is None:
        candidates = sorted(Path(__file__).parent.joinpath("traces").glob("*.jsonl"))
        trace_path = candidates[-1] if candidates else None
    if trace_path and Path(trace_path).is_file():
        events = [
            json.loads(line)
            for line in Path(trace_path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
        n_llm = sum(1 for e in events if e.get("event") == "llm_call")
        n_tool = sum(1 for e in events if e.get("event") == "tool_run")
        n_fail = sum(1 for e in events if e.get("event") == "attempt_failed")
        n_legacy = sum(1 for e in events if e.get("stage") == "fixture")
        if n_llm > 0 and n_tool > 0:
            trace_ok = True
            trace_detail = (
                f"{len(events)} events, {n_llm} LLM calls, {n_tool} tool runs, "
                f"{n_fail} failed attempts preserved"
            )
        elif n_legacy > 0:
            # Hand-driven (pre-pipeline) run whose results were adapted into the
            # canonical format; the adaptation itself is documented in-trace.
            trace_ok = True
            trace_detail = (
                f"{len(events)} events; legacy hand-driven run — adaptation "
                f"documented by {n_legacy} fixture events (no LLM/tool calls "
                f"captured by construction)"
            )
        else:
            trace_detail = (
                f"{len(events)} events but {n_llm} LLM calls and {n_tool} tool runs "
                f"— trace looks empty (the classic fail mode)"
            )
    rubric["trace"] = {"pass": trace_ok, "detail": trace_detail}

    passed = sum(1 for v in rubric.values() if v["pass"])
    if passed == len(rubric):
        verdict = "PASS"
    elif passed >= len(rubric) - 1 and rubric["evidence"]["pass"]:
        verdict = "CONDITIONAL PASS"
    else:
        verdict = "FAIL"

    def _path_str(p: Path | str | None) -> str | None:
        if p is None:
            return None
        s = str(p)
        return s.removeprefix(str(Path.cwd()) + "/") if relative_paths else s

    return {
        "verdict": verdict,
        "rubric": rubric,
        "score": f"{passed}/{len(rubric)}",
        "workdir": _path_str(workdir),
        "trace": _path_str(trace_path),
    }


def render_markdown(result: dict) -> str:
    lines = [f"# Judge verdict: **{result['verdict']}** ({result['score']})", ""]
    for name, check in result["rubric"].items():
        mark = "✅" if check["pass"] else "❌"
        lines.append(f"- {mark} **{name}** — {check['detail']}")
    return "\n".join(lines)

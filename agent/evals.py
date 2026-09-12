"""Weave evaluation layer (CoreWeave Hacks build).

Scores stored audit outcomes against the integrity rules the judge and the
summary validator already enforce — verdict-vs-reference, self-consistency,
control honesty, evidence completeness — as a Weave Evaluation (leaderboard
+ scorer means in the Weave UI), or as a plain local table when Weave isn't
configured. The scorers are the same functions either way; Weave is the
observability surface, not the source of truth.

  lemma eval                     # every papers/* with audit results
  lemma eval papers/<id>         # one workdir
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

from agent.weave_ops import op

STATUSES = ("supported", "falsified", "inconclusive")


# ── dataset ──────────────────────────────────────────────────────────────


def build_dataset(workdirs: list[Path]) -> list[dict]:
    """One row per audited claim: the recorded outcome plus ground-truth
    context (reviewer-reference verdict, figure count) for the scorers."""
    rows: list[dict] = []
    for wd in workdirs:
        report_path = wd / "results" / "audit_report.json"
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding="utf-8"))
        for c in report.get("claims", []):
            if c.get("status") == "not_audited":
                continue
            cid = c["id"]
            claim_dir = wd / "results" / cid.lower()
            summary = c.get("summary") or {}
            summary_path = claim_dir / "audit_summary.json"
            if summary_path.is_file():
                summary = json.loads(summary_path.read_text(encoding="utf-8"))
            rows.append(
                {
                    "paper_id": wd.name,
                    "claim_id": cid,
                    "outcome": {
                        "status": summary.get("status", c.get("status")),
                        "metrics": summary.get("metrics") or {},
                        "notes": summary.get("notes", ""),
                        "attempts": c.get("attempts", 0),
                        "n_figures": len(list(claim_dir.glob("*.png")))
                        if claim_dir.is_dir()
                        else 0,
                    },
                    "reference_status": _reference_status(claim_dir),
                }
            )
    return rows


def _reference_status(claim_dir: Path) -> str | None:
    """The verdict of the reviewer_reference run, if one was recorded — the
    pinned ground truth the improvement loop is scored against."""
    if not claim_dir.is_dir():
        return None
    for run_path in sorted(claim_dir.glob("run_attempt*.json")):
        try:
            run = json.loads(run_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if run.get("source") == "reviewer_reference":
            status = (run.get("summary") or {}).get("status")
            if status in STATUSES:
                return status
    return None


# ── model + scorers (same functions serve Weave and local scoring) ────────


@op
def recorded_outcome(outcome: dict, **row) -> dict:
    """The evaluated 'model': returns the stored audit outcome so the
    evaluation scores the audit trail of record, not a fresh run."""
    return outcome


@op
def verdict_matches_reference(output: dict, reference_status: str | None = None, **row):
    """Final verdict equals the reviewer-reference verdict. Claims with no
    reference are not penalized (score True — the check simply can't run)."""
    if not reference_status:
        return True
    return output.get("status") == reference_status


@op
def self_consistent(output: dict, **row):
    """No verdict-vs-metrics contradictions (the auditor's own validator)."""
    from agent.auditor import _summary_problems

    return not _summary_problems(output)


@op
def control_honest(output: dict, **row):
    """A decisive verdict (supported/falsified) only counts if the script's
    positive control ran and passed."""
    if output.get("status") not in ("supported", "falsified"):
        return True
    return (output.get("metrics") or {}).get("control_pass") is True


@op
def evidence_complete(output: dict, **row):
    """Metrics recorded, and a 'supported' verdict carries figure evidence
    (mirrors the judge's evidence rubric)."""
    if not (output.get("metrics") or {}):
        return False
    return not (output.get("status") == "supported" and not output.get("n_figures"))


@op
def decisive(output: dict, **row):
    """Verdict is supported or falsified (descriptive — 'inconclusive' is
    often the honest answer, so this measures rather than rewards)."""
    return output.get("status") in ("supported", "falsified")


SCORERS = [
    verdict_matches_reference,
    self_consistent,
    control_honest,
    evidence_complete,
    decisive,
]


# ── runner ────────────────────────────────────────────────────────────────


def _raw(fn):
    """Unwrap the @op shim to the plain function (weave.Evaluation needs
    real Ops, and local scoring needs the signature)."""
    return getattr(fn, "__wrapped__", fn)


def _call_scorer(fn, output: dict, row: dict):
    """Invoke a scorer with output + whichever dataset fields it declares."""
    params = set(inspect.signature(_raw(fn)).parameters) - {"output"}
    kwargs = {k: row[k] for k in params if k in row}
    return _raw(fn)(output, **kwargs)


def run_evaluation(workdirs: list[Path], use_weave: bool = True) -> dict:
    """Score every audited claim under workdirs. Returns per-claim scores
    plus scorer means. Uses weave.Evaluation when tracing is live."""
    rows = build_dataset(workdirs)
    if not rows:
        return {"rows": [], "means": {}}

    from agent import weave_ops

    if use_weave and weave_ops.active():
        return _run_weave_eval(rows)
    return _run_local_eval(rows)


def _run_weave_eval(rows: list[dict]) -> dict:
    import asyncio

    import weave

    model = weave.op(_raw(recorded_outcome))
    scorers = [weave.op(_raw(fn)) for fn in SCORERS]
    evaluation = weave.Evaluation(
        dataset=rows,
        scorers=scorers,
        evaluation_name="lemma-audit-integrity",
    )
    summary = asyncio.run(evaluation.evaluate(model))
    return {"rows": rows, "means": summary or {}, "weave": True}


def _run_local_eval(rows: list[dict]) -> dict:
    per_row: list[dict] = []
    means: dict[str, list[float]] = {fn.__name__: [] for fn in SCORERS}
    for row in rows:
        output = row["outcome"]
        scores = {}
        for fn in SCORERS:
            val = _call_scorer(fn, output, row)
            scores[fn.__name__] = val
            means[fn.__name__].append(float(bool(val)))
        per_row.append(
            {
                "paper_id": row["paper_id"],
                "claim_id": row["claim_id"],
                "status": output.get("status"),
                "scores": scores,
            }
        )
    return {
        "rows": per_row,
        "means": {k: round(sum(v) / len(v), 3) for k, v in means.items()},
        "weave": False,
    }


def render_table(result: dict) -> str:
    lines = ["# Audit evaluation", ""]
    for r in result["rows"]:
        if "scores" in r:
            marks = " ".join(f"{k}={'1' if v else '0'}" for k, v in r["scores"].items())
            lines.append(f"- {r['paper_id']} {r['claim_id']} [{r['status']}] {marks}")
        else:
            lines.append(
                f"- {r['paper_id']} {r['claim_id']} [{r['outcome'].get('status')}]"
            )
    lines.append("")
    if result.get("weave"):
        lines.append(f"Weave eval summary: {result['means']}")
    else:
        means = result.get("means", {})
        lines.append(
            "Means: "
            + " ".join(f"{k}={v}" for k, v in means.items())
            + "  (local — no Weave)"
        )
    return "\n".join(lines)

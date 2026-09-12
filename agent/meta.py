"""Self-improvement loop (CoreWeave Hacks build).

Closes the reviewer-feedback contract automatically. For each audited claim
that did not end 'supported', a reviewer-persona LLM reads the failure
history — the latest generated script, the recorded summary, and the
rejection/failure events in the trace — and writes
results/<cid>/feedback.auto.md. The claims are then re-audited and the
before/after outcome is recorded in results/improve_<ts>.json.

Guardrails (the loop can never reward-hack):
  - generated notes land in feedback.auto.md and are injected as ADVISORY;
    a human feedback.md stays authoritative where they conflict
  - reviewer_reference.py still escalates and still wins — the pinned ground
    truth is never editable by the loop
  - every drafted note and every re-audit is logged to the trace, same as
    any other stage

  lemma improve papers/<id> [--claims C2,C4]
"""

from __future__ import annotations

import json
import time
from pathlib import Path

from agent.llm import complete
from agent.traces import Trace
from agent.weave_ops import op

SYSTEM = """You are a meticulous reviewer advising an automated claim-audit agent. \
A prior audit of one paper claim failed to reach a confident, well-formed \
verdict. You are shown the claim, its test plan and success criterion, the \
latest generated audit script, the recorded summary, and the failure or \
rejection notes from the run trace.

Write concise reviewer notes (markdown, under 200 lines) for the agent's next \
attempt:
- Diagnose the most likely root cause (script bug, wrong statistic, setup \
mismatch, missing data, validator rejection).
- Give concrete corrective instructions: what to compute, which quantities \
to print in metrics, what the positive control should look like.
- Do NOT lower the bar: the success criterion stays the claim's own. If the \
evidence genuinely refutes the claim, say what a correct falsification \
demonstration looks like instead of how to force a pass.
- Never suggest fabricating metrics, skipping the positive control, or \
claiming a verdict the numbers do not support.

Respond with ONLY the markdown notes."""

MAX_SCRIPT_CHARS = 12_000
MAX_SUMMARY_CHARS = 4_000
MAX_EVENT_CHARS = 6_000


def _paper_text(workdir: Path) -> str:
    """Re-extract paper text from the PDF stored in the workdir."""
    pdfs = sorted(workdir.glob("*.pdf"))
    if not pdfs:
        return ""
    from agent.papers import _extract_pdf_text

    return _extract_pdf_text(pdfs[0])


def _failure_context(workdir: Path, claim_dir: Path, cid: str) -> str:
    """Latest script + recorded summary + this claim's failure events from
    the trace, bounded for the prompt."""
    parts: list[str] = []
    scripts = sorted(claim_dir.glob("audit_attempt*.py"))
    if scripts:
        latest = scripts[-1]
        parts.append(
            f"Latest audit script ({latest.name}):\n```python\n"
            f"{latest.read_text(encoding='utf-8')[:MAX_SCRIPT_CHARS]}\n```"
        )
    summary_path = claim_dir / "audit_summary.json"
    if summary_path.is_file():
        parts.append(
            "Recorded audit summary:\n```json\n"
            f"{summary_path.read_text(encoding='utf-8')[:MAX_SUMMARY_CHARS]}\n```"
        )
    # persisted failure tails (the reviewer sees the actual stderr, not just
    # an exit code — run_attempt*.json only exists for accepted runs)
    for failed_path in sorted(claim_dir.glob("run_attempt*.failed.json")):
        parts.append(
            f"Failed attempt detail ({failed_path.name}):\n```json\n"
            f"{failed_path.read_text(encoding='utf-8')[:MAX_EVENT_CHARS]}\n```"
        )
    trace_path = workdir / "trace.jsonl"
    events = []
    if trace_path.is_file():
        for line in trace_path.read_text(encoding="utf-8").splitlines():
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            if e.get("claim_id") != cid:
                continue
            if e.get("event") in (
                "attempt_failed",
                "summary_rejected",
                "script_written",
                "claim_final",
                "tool_run",
            ):
                events.append(e)
    if events:
        parts.append(
            "Failure/rejection events from the run trace:\n```json\n"
            + "\n".join(json.dumps(e, default=str) for e in events)[:MAX_EVENT_CHARS]
            + "\n```"
        )
    return "\n\n".join(parts)


@op
def _draft_feedback(claim: dict, context: str, trace: Trace) -> str:
    user = "\n".join(
        [
            f"Claim {claim['id']}: {claim.get('title', '')}",
            f"Statement: {claim.get('statement', '')}",
            f"Test plan: {claim.get('test_plan', '')}",
            f"Success criterion: {claim.get('success_criterion', '')}",
            "",
            context,
            "",
            "Write the reviewer notes for the next audit attempt.",
        ]
    )
    return complete(trace, "improve", SYSTEM, user, max_tokens=4000)


@op
def improve(workdir: Path, trace: Trace, only: set[str] | None = None) -> dict:
    """One improvement round: draft feedback.auto.md for every non-supported
    audited claim, re-audit those claims, record before/after."""
    claims_path = workdir / "claims.json"
    report_path = workdir / "results" / "audit_report.json"
    if not claims_path.is_file() or not report_path.is_file():
        raise FileNotFoundError(
            f"{workdir} lacks claims.json/audit_report.json — run an audit first"
        )
    claims = json.loads(claims_path.read_text(encoding="utf-8"))
    before_report = json.loads(report_path.read_text(encoding="utf-8"))
    before = {c["id"]: c.get("status") for c in before_report.get("claims", [])}

    targets = [
        c
        for c in claims
        if c.get("testable")
        and before.get(c["id"], "not_audited") not in ("supported", "not_audited")
        and (only is None or c["id"] in only)
    ]
    if not targets:
        return {"targets": [], "note": "nothing to improve"}

    paper_text = _paper_text(workdir)
    results_dir = workdir / "results"

    drafted = []
    for claim in targets:
        cid = claim["id"]
        claim_dir = results_dir / cid.lower()
        claim_dir.mkdir(exist_ok=True)
        context = _failure_context(workdir, claim_dir, cid)
        notes = _draft_feedback(claim, context, trace)
        (claim_dir / "feedback.auto.md").write_text(notes, encoding="utf-8")
        drafted.append(cid)
        trace.log("improve", "feedback_drafted", claim_id=cid, chars=len(notes))

    from agent.auditor import audit_all

    only_ids = {c["id"] for c in targets}
    report = audit_all(claims, paper_text, workdir, trace, only=only_ids)

    after = {c["id"]: c.get("status") for c in report.get("claims", [])}
    record = {
        "ts": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "targets": drafted,
        "before": {cid: before.get(cid) for cid in drafted},
        "after": {cid: after.get(cid) for cid in drafted},
        "changed": [cid for cid in drafted if after.get(cid) != before.get(cid)],
        # Inspect-large-gains rule: a flip to 'supported' immediately after
        # generated feedback is the reward-hacking direction (the notes could
        # have steered the audit into the criterion). Flag for eyeballing;
        # ->falsified moves the other way and needs no flag.
        "verify_gain": [
            cid
            for cid in drafted
            if after.get(cid) == "supported" and before.get(cid) != "supported"
        ],
    }
    out_path = results_dir / f"improve_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
    trace.log("improve", "round", path=str(out_path), **record)
    return record

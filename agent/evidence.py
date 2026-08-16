"""Stage 3 — assemble a Trackio logbook from claims + audit results.

Drives the `trackio logbook` CLI inside the paper workdir:
  index (auto) -> Executive summary -> one page per claim -> Conclusion
and attaches the run trace. Idempotent-ish: refuses to re-open if a logbook
already exists unless --force is passed to the CLI.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

from agent.traces import Trace


def _tio(workdir: Path, trace: Trace, *args: str) -> subprocess.CompletedProcess:
    cmd = ["trackio", "logbook", *args]
    trace.tool_run("evidence", " ".join(cmd[:4]) + " ...", 0, 0.0, 0)
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True, timeout=180)
    if proc.returncode != 0:
        raise RuntimeError(f"{' '.join(cmd)} failed: {proc.stderr[-1000:]}")
    return proc


def build_logbook(
    workdir: Path,
    title: str,
    claims: list[dict],
    report: dict,
    trace: Trace,
    run_meta: dict,
) -> None:
    results_dir = workdir / "results"
    _tio(
        workdir,
        trace,
        "open",
        "--title",
        f"Repro: {title}",
        "--no-serve",
        "--no-browser",
    )

    outcomes = {c["id"]: c for c in report.get("claims", [])}

    # --- Executive summary ------------------------------------------------
    supported = sum(1 for o in outcomes.values() if o.get("status") == "supported")
    falsified = sum(1 for o in outcomes.values() if o.get("status") == "falsified")
    inconclusive = sum(
        1
        for o in outcomes.values()
        if o.get("status") in ("inconclusive", "not_audited")
    )
    _tio(workdir, trace, "page", "Executive summary")
    summary_md = (
        f"**Verdict so far:** {supported} supported, {falsified} falsified, "
        f"{inconclusive} inconclusive/not audited of {len(claims)} extracted claims.\n\n"
        f"Paper: {title}\n\n"
        f"Pipeline run: `{run_meta.get('run_id')}` — extract → audit → evidence → judge. "
        f"{run_meta.get('llm_calls', '?')} LLM calls, "
        f"{run_meta.get('tool_runs', '?')} tool runs, "
        f"wall {run_meta.get('wall_min', '?')} min. "
        f"Model: {run_meta.get('model', '?')}.\n\n"
        "## Scope & cost\n\n"
        "| | This audit | Full replication |\n|---|---|---|\n"
        f"| Compute | CPU-only, ≈{run_meta.get('wall_min', '?')} min | "
        "paper's full training budget |\n"
        f"| Claims checked | {sum(1 for o in outcomes.values() if o.get('status') != 'not_audited')} of {len(claims)} | all |\n"
        "| Evidence | numerical audits + figures + trace | paper's own experiments |\n"
    )
    _tio(workdir, trace, "cell", "markdown", "--title", "Executive summary", summary_md)

    # --- one page per claim ------------------------------------------------
    for claim in claims:
        cid = claim["id"]
        outcome = outcomes.get(cid, {})
        status = outcome.get("status", "not_audited")
        _tio(workdir, trace, "page", f"Claim {cid} — {claim['title']}")

        claim_md = (
            f"**Claim ({claim['kind']}).** {claim['statement']}\n\n"
            f"**Test plan.** {claim['test_plan']}\n\n"
            f"**Success criterion.** {claim['success_criterion']}\n\n"
            f"**Compute budget.** {claim['compute']}"
        )
        _tio(workdir, trace, "cell", "markdown", "--title", "Claim", claim_md)

        if status == "not_audited":
            _tio(
                workdir,
                trace,
                "cell",
                "markdown",
                "--title",
                "Audit",
                "Marked not testable by the extraction stage; no audit run.",
            )
            continue

        summary = outcome.get("summary", {})
        audit_md = (
            f"**Status: {status.upper()}** after {outcome.get('attempts', '?')} attempt(s).\n\n"
            f"Metrics: `{json.dumps(summary.get('metrics', {}))}`\n\n"
            f"{summary.get('notes', '')}"
        )
        _tio(workdir, trace, "cell", "markdown", "--title", "Audit result", audit_md)

        # figures produced by the audit script
        claim_dir = results_dir / cid.lower()
        if claim_dir.is_dir():
            for png in sorted(claim_dir.glob("*.png")):
                _tio(
                    workdir,
                    trace,
                    "cell",
                    "figure",
                    "--page",
                    f"Claim {cid} — {claim['title']}",
                    "--title",
                    png.stem,
                    "--image",
                    str(png),
                )

        # literature context via Paperclip (GXL) — cached per claim so
        # evidence rebuilds do not re-query the corpus
        from agent import paperclip

        claim_dir = results_dir / cid.lower()
        xcheck_path = claim_dir / "cross_check.json"
        pc_ok = paperclip.available()[0]
        if pc_ok and not xcheck_path.is_file():
            xcheck = paperclip.cross_check(claim, trace=trace)
            claim_dir.mkdir(parents=True, exist_ok=True)
            xcheck_path.write_text(json.dumps(xcheck, indent=2), encoding="utf-8")
        if xcheck_path.is_file():
            xcheck = json.loads(xcheck_path.read_text(encoding="utf-8"))
            _tio(
                workdir,
                trace,
                "cell",
                "markdown",
                "--title",
                "Literature context",
                paperclip.render_markdown(xcheck),
            )

    # --- Conclusion ---------------------------------------------------------
    _tio(workdir, trace, "page", "Conclusion")
    concl_md = (
        f"Of {len(claims)} claims: **{supported} supported**, **{falsified} falsified**, "
        f"**{inconclusive} inconclusive or not audited**.\n\n"
        "All attempts (including failed runs) are preserved in the attached trace. "
        "Falsified or inconclusive outcomes are reported as results, not patched."
    )
    _tio(workdir, trace, "cell", "markdown", "--title", "Conclusion", concl_md)

    # --- attach trace --------------------------------------------------------
    trace_path = Path(trace.path)
    if trace_path.is_file():
        proc = subprocess.run(
            ["trackio", "logbook", "attach", "trace", str(trace_path)],
            cwd=workdir,
            capture_output=True,
            text=True,
            timeout=120,
        )
        trace.tool_run(
            "evidence",
            "trackio logbook attach trace",
            proc.returncode,
            0.0,
            len(proc.stdout),
        )
        if proc.returncode != 0:
            trace.note("evidence", f"trace attach failed: {proc.stderr[-300:]}")

    trace.note("evidence", f"logbook assembled in {workdir}/.trackio")


def publish(workdir: Path, space_id: str, trace: Trace) -> str:
    proc = _tio(workdir, trace, "publish", space_id)
    return proc.stdout[-500:]

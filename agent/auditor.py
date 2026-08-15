"""Stage 2 — the auditor agent loop.

For each testable claim: propose a self-contained audit script (numpy-first,
matplotlib plots into results/), run it, read its summary JSON, and iterate up
to N attempts. Honest outcomes only — a claim that fails the audit is recorded
as falsified/inconclusive, never patched into a pass.

Output per claim in <workdir>/results/:
  audit_c<k>_attempt<j>.py, run_<j>.json, *.png, and audit_c<k>_summary.json
"""

from __future__ import annotations

import json
import subprocess
import time
from pathlib import Path

from agent.llm import complete, extract_json
from agent.traces import Trace

MAX_ATTEMPTS = 3
RUN_TIMEOUT_S = 1200  # 20 min per script; audits should be cpu-fast by design
MAX_OUT_CHARS = 4000

SYSTEM = """You are the audit stage of Lemma, an AI-scientist pipeline. You write \
numerical audit scripts that independently test one claim of a research paper.

Environment rules:
- Python 3 with numpy, scipy, matplotlib ONLY. No torch, no network, no pip install.
- The script must be fully self-contained: implement any math from scratch, set all \
hyperparameters, seed numpy, and run in a few minutes on CPU unless the claim \
explicitly requires a GPU (then say so in summary instead of running).
- The script MUST print exactly one line: SUMMARY_JSON=<json> where json has keys:
  claim_id, status ("supported"|"falsified"|"inconclusive"), metrics (dict of \
numbers/strings), notes (short explanation). Also save plots to results/*.png via \
matplotlib with Agg backend.

Scientific integrity rules:
- Reproduce the claim's own setup; do not tune knobs to force a pass.
- If results disagree with the paper, that is a REAL result: set status \
"falsified" (clear contradiction) or "inconclusive" (setup mismatch/insufficient).
- POSITIVE CONTROL (mandatory): the script must also run the same statistic on a \
synthetic case whose answer is known to be true (e.g. exact data for the claimed \
distribution). Include control_pass (true/false) in metrics. If the control fails, \
the statistic is buggy — set status "inconclusive" and say so in notes; never \
report "falsified" when the control fails.
- Judge the claim ONLY against its stated success criterion; do not add extra \
pass conditions the paper does not make.
- Report actual numbers from the run; never fabricate metrics.

Respond with ONLY a JSON object: {"script": "<full python code>", "notes": "..."}. \
On follow-up turns, fix the actual failure shown in the run output."""


def audit_all(claims: list[dict], paper_text: str, workdir: Path, trace: Trace) -> dict:
    results_dir = workdir / "results"
    results_dir.mkdir(exist_ok=True)
    report = {"claims": []}

    for claim in claims:
        if not claim.get("testable"):
            trace.log("audit", "skip untestable", claim_id=claim["id"])
            report["claims"].append(
                {"id": claim["id"], "status": "not_audited", "attempts": 0}
            )
            continue
        outcome = audit_one(claim, paper_text, workdir, results_dir, trace)
        report["claims"].append(outcome)

    out = results_dir / "audit_report.json"
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    trace.log(
        "audit",
        "report",
        path=str(out),
        summary=[{"id": c["id"], "status": c["status"]} for c in report["claims"]],
    )
    return report


def audit_one(
    claim: dict, paper_text: str, workdir: Path, results_dir: Path, trace: Trace
) -> dict:
    cid = claim["id"]
    claim_dir = results_dir / cid.lower()
    claim_dir.mkdir(exist_ok=True)

    run_history: list[str] = []
    last_summary = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        script_path = claim_dir / f"audit_attempt{attempt}.py"
        user_prompt = _build_prompt(claim, paper_text, attempt, run_history)
        raw = complete(trace, "audit", SYSTEM, user_prompt, max_tokens=8000)

        try:
            payload = extract_json(raw)
            code = payload["script"] if isinstance(payload, dict) else raw
        except (json.JSONDecodeError, KeyError, TypeError):
            code = _strip_fences(raw)
        script_path.write_text(code, encoding="utf-8")
        trace.log(
            "audit",
            "script_written",
            claim_id=cid,
            attempt=attempt,
            path=str(script_path),
            chars=len(code),
        )

        t0 = time.time()
        try:
            proc = subprocess.run(
                ["python3", str(script_path)],
                cwd=workdir,
                capture_output=True,
                text=True,
                timeout=RUN_TIMEOUT_S,
            )
            exit_code, out, err = proc.returncode, proc.stdout, proc.stderr
        except subprocess.TimeoutExpired as e:
            exit_code, out = -1, ""
            err = f"TIMEOUT after {RUN_TIMEOUT_S}s\n{(e.stderr or '')[-MAX_OUT_CHARS:]}"
        duration = time.time() - t0
        trace.tool_run(
            "audit",
            f"python3 {script_path.name}",
            exit_code,
            duration,
            len(out) + len(err),
        )

        summary = _parse_summary(out)
        tail = out[-MAX_OUT_CHARS:] + "\n---STDERR---\n" + err[-MAX_OUT_CHARS:]

        if exit_code == 0 and summary:
            run_path = claim_dir / f"run_attempt{attempt}.json"
            run_path.write_text(
                json.dumps(
                    {
                        "attempt": attempt,
                        "exit_code": 0,
                        "summary": summary,
                        "wall_s": round(duration, 2),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            last_summary = summary
            trace.log(
                "audit",
                "claim_audited",
                claim_id=cid,
                attempt=attempt,
                status=summary.get("status"),
                wall_s=round(duration, 2),
            )
            break

        # failed attempt: record and feed back for the next iteration
        run_history.append(
            f"ATTEMPT {attempt} FAILED (exit={exit_code}).\nstdout+stderr tail:\n{tail}"
        )
        trace.log(
            "audit",
            "attempt_failed",
            claim_id=cid,
            attempt=attempt,
            exit_code=exit_code,
        )
        last_summary = {
            "claim_id": cid,
            "status": "inconclusive",
            "metrics": {},
            "notes": f"script failed after {attempt} tries",
        }

    if last_summary is None:
        last_summary = {
            "claim_id": cid,
            "status": "inconclusive",
            "metrics": {},
            "notes": "no successful run",
        }
    summary_path = claim_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(last_summary, indent=2), encoding="utf-8")
    return {
        "id": cid,
        "status": last_summary.get("status", "inconclusive"),
        "attempts": len(run_history) + (1 if last_summary else 0),
        "summary": last_summary,
    }


def _build_prompt(
    claim: dict, paper_text: str, attempt: int, history: list[str]
) -> str:
    parts = [
        f"Audit claim {claim['id']}: {claim['title']}",
        f"Statement: {claim['statement']}",
        f"Paper evidence it rests on: {claim['evidence_in_paper']}",
        f"Test plan from extraction: {claim['test_plan']}",
        f"Success criterion: {claim['success_criterion']}",
        f"Attempt {attempt} of {MAX_ATTEMPTS}.",
        "",
        "Relevant paper text (excerpt):",
        paper_text[:20_000],
    ]
    if history:
        parts += ["", "Prior attempts (fix the real failure):", *history]
    return "\n".join(parts)


def _parse_summary(stdout: str) -> dict | None:
    for line in stdout.splitlines():
        if line.startswith("SUMMARY_JSON="):
            try:
                data = json.loads(line[len("SUMMARY_JSON=") :])
                if isinstance(data, dict) and "status" in data:
                    return data
            except json.JSONDecodeError:
                return None
    return None


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return text

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
import math
import re
import subprocess
import sys
import time
from pathlib import Path

from agent import records
from agent.llm import complete, extract_json
from agent.traces import Trace
from agent.weave_ops import op

MAX_ATTEMPTS = 3
STATUSES = ("supported", "falsified", "inconclusive")
RUN_TIMEOUT_S = 1200  # 20 min per script; audits should be cpu-fast by design
MAX_OUT_CHARS = 4000


@op
def _run_script(
    script_path: Path, workdir: Path, timeout_s: float = RUN_TIMEOUT_S
) -> tuple[int, str, str, float]:
    """Execute one audit script as a subprocess. A weave op when tracing is
    live so each run shows up as a tool span under the audit attempt."""
    t0 = time.time()
    try:
        proc = subprocess.run(
            [sys.executable, str(script_path.resolve())],
            cwd=str(workdir.resolve()),
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        exit_code, out, err = proc.returncode, proc.stdout, proc.stderr
    except subprocess.TimeoutExpired as e:
        exit_code = -1
        out = _tail(e.stdout)
        err = f"TIMEOUT after {timeout_s}s\n{_tail(e.stderr)}"
    return exit_code, out, err, time.time() - t0


def _tail(chunk: str | bytes | None) -> str:
    if isinstance(chunk, bytes):
        chunk = chunk.decode("utf-8", errors="replace")
    return (chunk or "")[-MAX_OUT_CHARS:]


SYSTEM = """You are the audit stage of Lemma, an AI-scientist pipeline. You write \
numerical audit scripts that independently test one claim of a research paper.

Environment rules:
- Python 3 with numpy, scipy, matplotlib ONLY. No torch, no network, no pip install.
- The script must be fully self-contained: implement any math from scratch, set all \
hyperparameters, seed numpy, and run in a few minutes on CPU unless the claim \
explicitly requires a GPU (then say so in summary instead of running).
- The script MUST print exactly one line: SUMMARY_JSON=<json> where json has keys:
  claim_id, status ("supported"|"falsified"|"inconclusive"), metrics (dict of \
numbers/strings — MUST be non-empty), notes (short explanation). Serialize with \
`json.dumps(summary, default=str)` so numpy scalars (np.bool_, np.float64, \
np.int64) never crash the encoder. Also save plots to results/{claim_slug}/ \
(that directory already exists; e.g. results/c1/fig.png) via matplotlib with the \
Agg backend. Never write to other directories.

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


def audit_all(
    claims: list[dict],
    paper_text: str,
    workdir: Path,
    trace: Trace,
    only: set[str] | None = None,
    jobs: int = 1,
) -> dict:
    results_dir = workdir / "results"
    results_dir.mkdir(exist_ok=True)

    # merge with any existing report so subset re-audits preserve prior results
    report_path = results_dir / "audit_report.json"
    existing = (
        json.loads(report_path.read_text(encoding="utf-8"))
        if report_path.is_file()
        else {}
    )
    report = {"claims": list(existing.get("claims", []))}
    existing_ids = {c["id"] for c in report["claims"]}

    selected = [c for c in claims if only is None or c["id"] in only]
    testable = [c for c in selected if c.get("testable")]
    entries: dict[str, dict] = {}
    for claim in selected:
        if claim.get("testable"):
            continue
        trace.log("audit", "skip untestable", claim_id=claim["id"])
        entries[claim["id"]] = {
            "id": claim["id"],
            "status": "not_audited",
            "attempts": 0,
        }

    # Claims are independent audits; run them concurrently when asked.
    # Each audit_one writes only under results/<cid>/ and appends whole
    # lines to the shared trace, so the parallelism is safe. Slow eval
    # loops quietly remove the option of trusting results — keep them fast.
    if jobs > 1 and len(testable) > 1:
        from concurrent.futures import ThreadPoolExecutor

        trace.log("audit", "parallel", claims=len(testable), jobs=jobs)
        with ThreadPoolExecutor(max_workers=jobs) as pool:
            for claim, entry in zip(
                testable,
                pool.map(
                    lambda c: audit_one(c, paper_text, workdir, results_dir, trace),
                    testable,
                ),
                strict=True,
            ):
                entries[claim["id"]] = entry
    else:
        for claim in testable:
            entries[claim["id"]] = audit_one(
                claim, paper_text, workdir, results_dir, trace
            )

    for claim in selected:
        entry = entries.get(claim["id"])
        if entry is None:
            continue
        if entry["id"] in existing_ids:
            report["claims"] = [
                entry if c["id"] == entry["id"] else c for c in report["claims"]
            ]
        else:
            report["claims"].append(entry)
            existing_ids.add(entry["id"])

    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    trace.log(
        "audit",
        "report",
        path=str(report_path),
        summary=[{"id": c["id"], "status": c["status"]} for c in report["claims"]],
    )
    return report


_ATTEMPT_FILE_RE = re.compile(
    r"(?:audit_attempt|run_attempt)(\d+)(?:\.failed)?\.(?:py|json)"
)


def _next_attempt_number(claim_dir: Path) -> int:
    """One above the highest legacy attempt number on disk — scripts and run
    records share the counter so a new attempt never overwrites history."""
    highest = 0
    for p in claim_dir.iterdir():
        m = _ATTEMPT_FILE_RE.fullmatch(p.name)
        if m:
            highest = max(highest, int(m.group(1)))
    return highest + 1


@op
def audit_one(
    claim: dict, paper_text: str, workdir: Path, results_dir: Path, trace: Trace
) -> dict:
    cid = claim["id"]
    if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_-]{0,63}", cid):
        raise ValueError(f"claim id unsafe for path use: {cid!r}")
    claim_slug = cid.lower()
    claim_dir = results_dir / claim_slug
    claim_dir.mkdir(exist_ok=True)

    # Human-in-the-loop: optional reviewer feedback for this claim.
    feedback_path = claim_dir / "feedback.md"
    feedback = (
        feedback_path.read_text(encoding="utf-8").strip()
        if feedback_path.is_file()
        else ""
    )
    if feedback:
        trace.log("audit", "feedback_loaded", claim_id=cid, chars=len(feedback))

    # Self-improvement loop (`lemma improve`): generated reviewer notes.
    # Advisory only — human feedback.md stays authoritative where they differ.
    auto_path = claim_dir / "feedback.auto.md"
    auto_feedback = (
        auto_path.read_text(encoding="utf-8").strip() if auto_path.is_file() else ""
    )
    if auto_feedback:
        trace.log(
            "audit", "auto_feedback_loaded", claim_id=cid, chars=len(auto_feedback)
        )

    source = "human_guided" if feedback else "agent_generated"

    run_history: list[str] = []
    attempt_ids: list[str] = []
    executed = 0
    last_summary = None
    last_attempt_id = None
    final_source = source

    # continue attempt numbering if scripts or run records from a prior
    # round exist
    next_number = _next_attempt_number(claim_dir)
    for attempt in range(next_number, next_number + MAX_ATTEMPTS):
        script_path = claim_dir / f"audit_attempt{attempt}.py"
        user_prompt = _build_prompt(
            claim, paper_text, attempt, run_history, feedback, auto_feedback
        )
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

        attempt_root = records.begin_attempt(
            claim_dir,
            claim,
            trace,
            attempt,
            source,
            code,
            feedback,
            auto_feedback[:4000],
        )
        attempt_ids.append(attempt_root.name)
        last_attempt_id = attempt_root.name
        executed += 1
        before_figures = records.figure_hashes(claim_dir)

        exit_code, out, err, duration = _run_script(script_path, workdir)
        trace.tool_run(
            "audit",
            f"{sys.executable} {script_path.name}",
            exit_code,
            duration,
            len(out) + len(err),
        )

        summary = _parse_summary(out)
        tail = out[-MAX_OUT_CHARS:] + "\n---STDERR---\n" + err[-MAX_OUT_CHARS:]

        if summary is not None and summary.get("claim_id") != cid:
            problems = ["summary claim_id does not match selected claim"]
        elif summary is not None:
            problems = _summary_problems(summary)
        else:
            problems = ["script produced no parseable SUMMARY_JSON"]
        records.finish_attempt(
            attempt_root,
            claim_dir,
            before_figures,
            exit_code,
            out,
            err,
            duration,
            summary,
            problems,
            trace,
        )

        # Accept a run that printed a valid SUMMARY_JSON even if the process
        # crashed afterwards (e.g. a serialization error on the very last line);
        # the evidence was still produced and is what matters.
        if summary is not None and not problems:
            run_path = claim_dir / f"run_attempt{attempt}.json"
            run_path.write_text(
                json.dumps(
                    {
                        "attempt": attempt,
                        "attempt_id": attempt_root.name,
                        "source": source,
                        "run_id": trace.run_id,
                        "exit_code": exit_code,
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
                attempt_id=attempt_root.name,
                status=summary.get("status"),
                wall_s=round(duration, 2),
            )
            break

        if summary is not None:
            # the script ran but its verdict contradicts its own recorded
            # metrics (e.g. "falsified" while the control residual shows a
            # broken implementation). Rejecting it is the honest move:
            # record why and iterate.
            (claim_dir / f"run_attempt{attempt}.failed.json").write_text(
                json.dumps(
                    {
                        "attempt": attempt,
                        "attempt_id": attempt_root.name,
                        "source": source,
                        "run_id": trace.run_id,
                        "kind": "rejected",
                        "exit_code": exit_code,
                        "summary": summary,
                        "problems": problems,
                        "tail": tail[-MAX_OUT_CHARS:],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            trace.log(
                "audit",
                "summary_rejected",
                claim_id=cid,
                attempt=attempt,
                attempt_id=attempt_root.name,
                problems=problems,
            )
            run_history.append(
                f"ATTEMPT {attempt} REJECTED (verdict contradicts its own "
                f"metrics): {problems}\nstdout+stderr tail:\n{tail}"
            )
            last_summary = {
                "claim_id": cid,
                "status": "inconclusive",
                "metrics": {
                    "attempts_used": attempt,
                    "rejected_problems": problems,
                },
                "notes": "summary rejected: " + "; ".join(problems),
            }
            continue

        # failed attempt: record and feed back for the next iteration
        (claim_dir / f"run_attempt{attempt}.failed.json").write_text(
            json.dumps(
                {
                    "attempt": attempt,
                    "attempt_id": attempt_root.name,
                    "source": source,
                    "run_id": trace.run_id,
                    "kind": "crashed",
                    "exit_code": exit_code,
                    "tail": tail[-MAX_OUT_CHARS:],
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        run_history.append(
            f"ATTEMPT {attempt} FAILED (exit={exit_code}).\nstdout+stderr tail:\n{tail}"
        )
        trace.log(
            "audit",
            "attempt_failed",
            claim_id=cid,
            attempt=attempt,
            attempt_id=attempt_root.name,
            exit_code=exit_code,
        )
        last_summary = {
            "claim_id": cid,
            "status": "inconclusive",
            "metrics": {"script_exit_code": exit_code, "attempts_used": attempt},
            "notes": f"script failed after {attempt} tries",
        }

    if last_summary is None:
        last_summary = {
            "claim_id": cid,
            "status": "inconclusive",
            "metrics": {"attempts_used": MAX_ATTEMPTS},
            "notes": "no successful run",
        }
    candidate_summary = last_summary

    # Reviewer reference: when the human reviewer supplied a reference
    # implementation it runs after the generated attempts, whatever their
    # outcome — a supported candidate is still checked against the pinned
    # reference, and a broken implementation can't hide a real verdict.
    # A valid reference run is authoritative and becomes the final outcome;
    # a failed reference never overwrites a valid candidate. Both the
    # candidate outcome and any disagreement are recorded explicitly, and
    # the reference runs under the same rules as any audit script (must
    # print SUMMARY_JSON, must pass the validator).
    ref_path = claim_dir / "reviewer_reference.py"
    reference_summary = None
    reference_checked = False
    reference_disagreement = False
    if ref_path.is_file():
        attempt = _next_attempt_number(claim_dir)
        ref_code = ref_path.read_text(encoding="utf-8")
        attempt_root = records.begin_attempt(
            claim_dir,
            claim,
            trace,
            attempt,
            "reviewer_reference",
            ref_code,
            feedback,
            auto_feedback[:4000],
        )
        attempt_ids.append(attempt_root.name)
        executed += 1
        trace.log(
            "audit",
            "reviewer_reference_executed",
            claim_id=cid,
            attempt=attempt,
            attempt_id=attempt_root.name,
            path=str(ref_path),
            reason=(
                "reviewer reference runs after generated attempts ended "
                f"{last_summary.get('status')!r}; a valid reference is authoritative"
            ),
        )
        before_figures = records.figure_hashes(claim_dir)
        exit_code, out, err, duration = _run_script(ref_path, workdir)
        trace.tool_run(
            "audit",
            f"{sys.executable} {ref_path.name}",
            exit_code,
            duration,
            len(out) + len(err),
        )
        ref_summary = _parse_summary(out)
        tail = out[-MAX_OUT_CHARS:] + "\n---STDERR---\n" + err[-MAX_OUT_CHARS:]
        if ref_summary is not None and ref_summary.get("claim_id") != cid:
            ref_problems = ["summary claim_id does not match selected claim"]
        elif ref_summary is not None:
            ref_problems = _summary_problems(ref_summary)
        else:
            ref_problems = ["script produced no parseable SUMMARY_JSON"]
        records.finish_attempt(
            attempt_root,
            claim_dir,
            before_figures,
            exit_code,
            out,
            err,
            duration,
            ref_summary,
            ref_problems,
            trace,
        )
        reference_summary = ref_summary
        reference_checked = ref_summary is not None and not ref_problems
        reference_disagreement = bool(
            candidate_summary
            and candidate_summary.get("status") in STATUSES
            and ref_summary
            and ref_summary.get("status") in STATUSES
            and ref_summary.get("status") != candidate_summary.get("status")
        )
        if ref_summary is not None and not reference_checked:
            trace.log(
                "audit",
                "summary_rejected",
                claim_id=cid,
                attempt=attempt,
                attempt_id=attempt_root.name,
                source="reviewer_reference",
                problems=ref_problems,
            )
        if reference_checked:
            run_path = claim_dir / f"run_attempt{attempt}.json"
            run_path.write_text(
                json.dumps(
                    {
                        "attempt": attempt,
                        "attempt_id": attempt_root.name,
                        "source": "reviewer_reference",
                        "run_id": trace.run_id,
                        "exit_code": exit_code,
                        "summary": ref_summary,
                        "wall_s": round(duration, 2),
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
            last_summary = ref_summary
            last_attempt_id = attempt_root.name
            final_source = "reviewer_reference"
            trace.log(
                "audit",
                "claim_audited",
                claim_id=cid,
                attempt=attempt,
                attempt_id=attempt_root.name,
                status=ref_summary.get("status"),
                source="reviewer_reference",
                wall_s=round(duration, 2),
            )
        else:
            (claim_dir / f"run_attempt{attempt}.failed.json").write_text(
                json.dumps(
                    {
                        "attempt": attempt,
                        "attempt_id": attempt_root.name,
                        "source": "reviewer_reference",
                        "run_id": trace.run_id,
                        "kind": "reference_failed",
                        "exit_code": exit_code,
                        "summary": ref_summary,
                        "problems": ref_problems,
                        "tail": tail[-MAX_OUT_CHARS:],
                    },
                    indent=2,
                ),
                encoding="utf-8",
            )
    # always record the final outcome so no claim silently disappears
    # from the trace (matters when every attempt crashed)
    trace.log(
        "audit",
        "claim_final",
        claim_id=cid,
        status=last_summary.get("status", "inconclusive"),
        attempts=executed,
    )
    summary_path = claim_dir / "audit_summary.json"
    summary_path.write_text(json.dumps(last_summary, indent=2), encoding="utf-8")
    return {
        "id": cid,
        "status": last_summary.get("status", "inconclusive"),
        "attempts": executed,
        "summary": last_summary,
        "source": final_source,
        "attempt_id": last_attempt_id,
        "candidate_summary": candidate_summary,
        "candidate_source": source,
        "reference_summary": reference_summary,
        "reference_disagreement": reference_disagreement,
        "reference_checked": reference_checked,
        "attempt_ids": attempt_ids,
    }


def _build_prompt(
    claim: dict,
    paper_text: str,
    attempt: int,
    history: list[str],
    feedback: str = "",
    auto_feedback: str = "",
) -> str:
    claim_slug = claim["id"].lower()
    parts = [
        f"Audit claim {claim['id']}: {claim['title']}",
        f"Statement: {claim['statement']}",
        f"Paper evidence it rests on: {claim['evidence_in_paper']}",
        f"Test plan from extraction: {claim['test_plan']}",
        f"Success criterion: {claim['success_criterion']}",
        f"Save any plots to: results/{claim_slug}/ (directory exists)",
        f"Attempt {attempt}.",
    ]
    if feedback:
        parts += [
            "",
            "HUMAN REVIEWER FEEDBACK (authoritative — follow it over the test plan "
            "where they conflict):",
            feedback,
        ]
    if auto_feedback:
        parts += [
            "",
            "GENERATED REVIEWER NOTES (advisory — written by the self-improvement "
            "loop after reviewing prior failures; treat as expert hints, and defer "
            "to HUMAN REVIEWER FEEDBACK and the claim's own success criterion "
            "where they conflict):",
            auto_feedback[:4000],
        ]
    parts += ["", "Relevant paper text (excerpt):", paper_text[:20_000]]
    if history:
        parts += ["", "Prior attempts in this round (fix the real failure):", *history]
    return "\n".join(parts)


def control_passed(value: object) -> bool:
    return (
        value is True
        or (type(value) in (int, float) and value == 1)
        or (isinstance(value, str) and value == "True")
    )


def _summary_problems(summary: dict) -> list[str]:
    """Contradictions between a verdict and its own recorded metrics.

    Audit contract (AGENTS.md): a buggy statistic must read "inconclusive",
    never "falsified"/"supported". Flagged contradictions:
      - a claimed verdict while the script's own positive control failed
      - a claimed verdict while a control-residual metric contradicts
        control_pass (e.g. rel_diff_control=0.97 with control_pass=true)
      - NaN control metrics
    """
    problems: list[str] = []
    if not isinstance(summary, dict):
        return ["summary must be an object"]
    status = summary.get("status")
    if status not in ("supported", "falsified", "inconclusive"):
        problems.append("status must be supported, falsified, or inconclusive")
    metrics = summary.get("metrics")
    if not isinstance(metrics, dict) or not metrics:
        problems.append("metrics must be a non-empty object")
        return problems
    if status not in ("supported", "falsified"):
        return problems

    # The positive control is mandatory by contract: a decisive verdict with
    # NO passing control is as untrustworthy as one with a failed control —
    # and it's the cheap way to game the criterion (drop the control, keep
    # the verdict). Reject both. Accept only explicit passing values; metrics
    # may carry "True"/1 serialized via default=str on numpy scalars.
    control_pass = control_passed(metrics.get("control_pass"))
    if metrics.get("control_pass") is False:
        problems.append(f"status={status} but control_pass is false")
    elif not control_pass:
        problems.append(
            f"status={status} but no passing positive control recorded "
            "(control is mandatory for a decisive verdict)"
        )

    for name, val in metrics.items():
        if "control" not in name.lower():
            continue
        if isinstance(val, float) and not math.isfinite(val):  # NaN
            problems.append(f"control metric {name} is not finite")
        if (
            "diff" in name.lower()
            and isinstance(val, (int, float))
            and val == val
            and abs(val) > 0.05
            and control_pass
        ):
            problems.append(
                f"control residual {name}={val:.3g} contradicts control_pass=true"
            )

    # A verdict cannot rest on an unmeasured quantity. NaN / null / inf in a
    # primary measurement (non-control) means the criterion was never
    # actually evaluated — that is "inconclusive", never "falsified".
    for name, val in metrics.items():
        if "control" in name.lower():
            continue
        is_nan = isinstance(val, float) and not math.isfinite(val)
        if is_nan or val is None:
            problems.append(
                f"status={status} but primary metric {name} is non-finite/None "
                "(measurement never produced a usable value)"
            )
    n_meas = metrics.get("n_measurable_points")
    if isinstance(n_meas, (int, float)) and n_meas == 0:
        problems.append(f"status={status} but n_measurable_points=0 — no data to judge")
    return problems


def _parse_summary(stdout: str) -> dict | None:
    lines = [line for line in stdout.splitlines() if line.startswith("SUMMARY_JSON=")]
    if len(lines) != 1:
        return None
    try:
        data = json.loads(lines[0][len("SUMMARY_JSON=") :])
    except json.JSONDecodeError:
        return None
    if not isinstance(data, dict) or not isinstance(data.get("claim_id"), str):
        return None
    if not data["claim_id"] or not isinstance(data.get("notes"), str):
        return None
    return data


def _strip_fences(text: str) -> str:
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()[1:]
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        return "\n".join(lines)
    return text

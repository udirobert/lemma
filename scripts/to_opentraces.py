"""Convert lemma audit traces (trace.jsonl) to the opentraces v0.3 TAO schema.

`bench traj upload` natively parses opentraces records (thought -> action ->
observation); our raw JSONL falls through to the "generic" fallback and the
trajectory report collapses 600+ events into 3 extracted steps. This script
re-emits each paper's complete, chronological event stream as ONE opentraces
record (one JSONL line per file), so the full multi-round audit is legible to
the eval judges: every LLM call, script run, falsification, HITL feedback
load, reviewer-reference escalation, and final verdict.

Nothing is invented: fields map 1:1 from trace events. Only timestamps are
normalized to colon-offset ISO 8601.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "tmp" / "traj-upload-opentraces"

SCHEMA_VERSION = "0.3"
AGENT = {"name": "lemma", "version": "0.1.0"}
REPO_ROOT_REMOTE = "udirobert/lemma"

PAPERS = {
    "papers/jmlr-22-1228-ca-grokking/trace.jsonl": {
        "trace_id": "lemma-ca-grokking",
        "slug": "ca-grokking",
        "task": (
            "Audit JMLR 22-1228 ('Grokking in a Simple Algorithmic Task', Chen et al.): "
            "extract testable claims about cellular-automaton grokking from the PDF, "
            "author and execute self-contained numpy audit scripts, iterate through "
            "five human-in-the-loop review rounds with reviewer-reference escalation, "
            "assemble a Trackio evidence logbook, and run the automated "
            "evidence-trustworthiness judge."
        ),
        "outcome_notes": (
            "Judge PASS 5/5 (353 logbook events). Final verdicts: 5 supported / "
            "0 falsified / 1 inconclusive (compute-scoped)."
        ),
    },
    "papers/arxiv-2510.10981/trace.jsonl": {
        "trace_id": "lemma-icl-bayesian",
        "slug": "icl-provably-bayesian",
        "task": (
            "Audit arXiv 2510.10981 ('In-Context Learning Is Provably Bayesian "
            "Inference'): extract testable claims, author and execute self-contained "
            "numpy audit scripts including closed-form and Monte-Carlo cross-checks, "
            "iterate through four human-in-the-loop review rounds, assemble a Trackio "
            "evidence logbook, and run the automated evidence-trustworthiness judge."
        ),
        "outcome_notes": (
            "Judge PASS 5/5 (267 logbook events). Final verdicts: 3 supported / "
            "0 falsified / 3 inconclusive."
        ),
    },
}

TAGS = ["agentic-science", "claim-audit", "reproducibility", "lemma"]


def _iso(ts: str) -> str:
    """Normalize '+0000'-style offsets to '+00:00' (strict ISO 8601)."""
    return datetime.fromisoformat(ts).isoformat()


def _lifecycle_text(stage: str, e: dict) -> str:
    """Human-readable summary for pipeline bookkeeping events."""
    ev = e["event"]
    if ev == "start":
        return (
            f"[{stage}] run {e.get('run_id')} started: source={e.get('source')} "
            f"stages={','.join(e.get('stages', []))} model={e.get('model')} "
            f"workdir={e.get('workdir')}"
        )
    if ev == "claims_written":
        return (
            f"[{stage}] extracted {e.get('n_claims')} claims "
            f"({e.get('n_testable')} testable)"
        )
    if ev == "claim_audited":
        return (
            f"[{stage}] {e.get('claim_id')} attempt {e.get('attempt')}: "
            f"{e.get('status')} (wall {e.get('wall_s', 0):.1f}s)"
        )
    if ev == "attempt_failed":
        return (
            f"[{stage}] {e.get('claim_id')} attempt {e.get('attempt')} script "
            f"failed (exit {e.get('exit_code')})"
        )
    if ev == "feedback_loaded":
        return (
            f"[{stage}] human-in-the-loop: loaded reviewer feedback for "
            f"{e.get('claim_id')} ({e.get('chars')} chars)"
        )
    if ev == "reviewer_reference_executed":
        return (
            f"[{stage}] escalation: executed hand-verified reviewer reference for "
            f"{e.get('claim_id')} attempt {e.get('attempt')} — {e.get('reason')}"
        )
    if ev == "report":
        summary = e.get("summary", [])
        joined = "; ".join(f"{s.get('id')}={s.get('status')}" for s in summary)
        return f"[{stage}] audit report: {joined}"
    if ev == "claim_final":
        return (
            f"[{stage}] FINAL {e.get('claim_id')}: {e.get('status')} after "
            f"{e.get('attempts')} attempt(s)"
        )
    if ev == "done":
        return f"[{stage}] run finished: stages {','.join(e.get('stages', []))}"
    rest = {k: v for k, v in e.items() if k not in ("ts", "run_id", "event", "stage")}
    return f"[{stage}] {ev}: {json.dumps(rest, default=str)}"


def to_step(e: dict) -> dict:
    """Map one lemma trace event to an opentraces TAO step."""
    ev = e["event"]
    stage = e.get("stage", "")
    step: dict = {"timestamp": _iso(e["ts"])}

    if ev == "llm_call":
        step["thought"] = (
            f"[{stage}] LLM call: model={e.get('model')} "
            f"prompt={e.get('prompt_chars')}c -> response={e.get('response_chars')}c "
            f"in {e.get('duration_s', 0):.1f}s"
        )
    elif ev == "tool_run":
        step["action"] = {
            "tool_call": {
                "name": "shell",
                "input": {"cmd": e.get("cmd", ""), "stage": stage},
            }
        }
        step["observation"] = {
            "content": (
                f"exit={e.get('exit_code')} in {e.get('duration_s', 0):.1f}s; "
                f"{e.get('output_chars', 0)} chars output"
            )
        }
    elif ev == "script_written":
        step["action"] = {
            "tool_call": {
                "name": "write_audit_script",
                "input": {
                    "claim_id": e.get("claim_id"),
                    "attempt": e.get("attempt"),
                    "path": e.get("path"),
                },
            }
        }
        step["observation"] = {"content": f"wrote {e.get('chars', 0)} chars"}
    elif ev == "note":
        step["thought"] = f"[{stage}] {e.get('message', '')}"
    else:
        step["thought"] = _lifecycle_text(stage, e)
    return step


def convert(trace_path: Path, meta: dict) -> dict:
    events = [
        json.loads(line) for line in trace_path.read_text().splitlines() if line.strip()
    ]
    steps = [to_step(e) for e in events]

    llm_calls = [e for e in events if e["event"] == "llm_call"]
    tool_runs = [e for e in events if e["event"] == "tool_run"]
    run_ids = sorted({e["run_id"] for e in events if "run_id" in e})
    first_start = next((e for e in events if e["event"] == "start"), {})

    return {
        "trace_id": meta["trace_id"],
        "schema_version": SCHEMA_VERSION,
        "agent": AGENT,
        "environment": {
            "cwd": first_start.get("workdir", str(REPO)),
            "vcs": {"branch": "main", "remote": REPO_ROOT_REMOTE},
        },
        "task": {"input": meta["task"], "tags": TAGS},
        "timestamp_start": _iso(events[0]["ts"]),
        "timestamp_end": _iso(events[-1]["ts"]),
        "steps": steps,
        "outcome": {"status": "success", "notes": meta["outcome_notes"]},
        "metrics": {
            "events": len(events),
            "llm_calls": len(llm_calls),
            "tool_runs": len(tool_runs),
            "prompt_chars_total": sum(e.get("prompt_chars", 0) for e in llm_calls),
            "response_chars_total": sum(e.get("response_chars", 0) for e in llm_calls),
            "pipeline_runs": len(run_ids),
        },
    }


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for rel, meta in PAPERS.items():
        record = convert(REPO / rel, meta)
        out = OUT_DIR / f"lemma-{meta['slug']}-opentraces.jsonl"
        out.write_text(json.dumps(record) + "\n", encoding="utf-8")
        n_thinking = sum(1 for s in record["steps"] if "thought" in s)
        n_tool = sum(1 for s in record["steps"] if "action" in s)
        print(
            f"{out.name}: {len(record['steps'])} steps "
            f"({n_thinking} thinking / {n_tool} tool-call), "
            f"{out.stat().st_size / 1024:.1f} KB"
        )


if __name__ == "__main__":
    main()

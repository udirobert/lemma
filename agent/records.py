"""Runtime attempt provenance for audit executions.

Every executed audit attempt — LLM-generated or reviewer reference — gets an
immutable UUID directory under results/<cid>/attempts/<uuid>/ holding the exact
inputs (claim snapshot, feedback, script), the raw stdout/stderr, the parsed
outcome, and any figures the run created or changed. Snapshots preserve the
files as they were at run time; they are evidence of record, not a tamperproof
security boundary.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent.traces import Trace
from agent.weave_ops import current_call_url

SCHEMA_VERSION = 1


def begin_attempt(
    claim_dir: Path,
    claim: dict,
    trace: Trace,
    number: int,
    source: str,
    script: str,
    feedback: str,
    auto_feedback: str,
) -> Path:
    root = claim_dir / "attempts" / uuid.uuid4().hex
    root.mkdir(parents=True, exist_ok=False)
    payload = {
        "schema_version": SCHEMA_VERSION,
        "attempt_id": root.name,
        "run_id": trace.run_id,
        "number": number,
        "source": source,
        "claim": claim,
        "feedback": feedback,
        "auto_feedback": auto_feedback,
        "criterion_sha256": hashlib.sha256(
            claim.get("success_criterion", "").encode()
        ).hexdigest(),
        "script_sha256": hashlib.sha256(script.encode()).hexdigest(),
        "created_at": datetime.now(timezone.utc).isoformat(),
    }
    (root / "input.json").write_text(
        json.dumps(payload, indent=2) + "\n", encoding="utf-8"
    )
    (root / "script.py").write_text(script, encoding="utf-8")
    trace.log(
        "audit",
        "attempt_started",
        claim_id=claim["id"],
        attempt_id=root.name,
        attempt=number,
        source=source,
    )
    return root


def figure_hashes(claim_dir: Path) -> dict[str, str]:
    return {
        p.name: hashlib.sha256(p.read_bytes()).hexdigest()
        for p in claim_dir.glob("*.png")
        if p.is_file() and not p.is_symlink()
    }


def finish_attempt(
    root: Path,
    claim_dir: Path,
    before_figures: dict[str, str],
    exit_code: int,
    stdout: str,
    stderr: str,
    duration: float,
    summary: dict | None,
    problems: list[str],
    trace: Trace,
) -> dict:
    if (root / "outcome.json").exists():
        raise FileExistsError(f"attempt outcome already recorded: {root}")
    figures = []
    for name, digest in figure_hashes(claim_dir).items():
        if before_figures.get(name) != digest:
            (root / "figures").mkdir(exist_ok=True)
            shutil.copy2(claim_dir / name, root / "figures" / name)
            figures.append({"path": "figures/" + name, "sha256": digest})
    outcome = {
        "exit_code": exit_code,
        "wall_s": round(duration, 3),
        "summary": summary,
        "problems": problems,
        "accepted": summary is not None and not problems,
        "figures": figures,
        "weave_url": current_call_url(),
    }
    (root / "stdout.txt").write_text(stdout, encoding="utf-8")
    (root / "stderr.txt").write_text(stderr, encoding="utf-8")
    (root / "outcome.json").write_text(
        json.dumps(outcome, indent=2) + "\n", encoding="utf-8"
    )
    trace.log("audit", "attempt_finished", attempt_id=root.name, **outcome)
    return outcome

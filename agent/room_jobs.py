"""Pure helpers for the hosted Audit Room rerun backend (Modal).

No modal import — everything here is unit-testable in the repo venv.
The worker (`agent/room_modal.py`) calls these to verify the rerun
manifest against the frozen bundle, shape the live attempt record the
client renders, encode figures as data URIs (the static site has no
asset host), and enforce sliding-window rate limits.
"""

from __future__ import annotations

import base64
import hashlib
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

from agent import audit_room
from agent.room_server import load_allowed

MAX_FIGURE_BYTES = 1 * 1024 * 1024
MAX_TAIL_CHARS = 100_000


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


def allowed_entries(bundle: dict) -> tuple[dict, dict[str, str]]:
    """Verify each RERUN_MANIFEST row against the frozen bundle.

    Delegates to room_server.load_allowed so the hosted backend applies
    exactly the same digest checks as the local runner: the manifest's
    script_sha256 must equal both the sha256 of the exported script text
    and the attempt's recorded script.sha256. Returns (allowed, skipped)
    where allowed maps (paper_slug, claim_id, attempt_id) -> entry with
    script_text, claim, and number, and skipped maps "p/c/a" -> reason.
    """
    return load_allowed(bundle)


def _tail_artifact(name: str, raw: bytes, truncated: bool) -> dict:
    """TextArtifact keeping the tail — the diagnostic end of the output."""
    text = raw.decode("utf-8", "replace")
    if len(text) > MAX_TAIL_CHARS:
        text = text[-MAX_TAIL_CHARS:]
        truncated = True
    return {
        "name": name,
        "text": text,
        "truncated": truncated,
        "sha256": hashlib.sha256(raw).hexdigest(),
    }


def figure_data_uris(results_dir: Path) -> tuple[list[dict], list[str]]:
    """Encode result PNGs as data URIs so the static site can show them.

    Returns (figures, notes); files that aren't PNGs or exceed 1 MiB are
    skipped with an honest note for the job event log.
    """
    figures: list[dict] = []
    notes: list[str] = []
    if not results_dir.is_dir():
        return figures, notes
    for path in sorted(results_dir.glob("*.png")):
        if path.is_symlink() or not path.is_file():
            continue
        raw = path.read_bytes()
        if len(raw) > MAX_FIGURE_BYTES:
            notes.append(f"skipped {path.name}: exceeds 1 MiB figure cap")
            continue
        if not raw.startswith(b"\x89PNG\r\n\x1a\n"):
            notes.append(f"skipped {path.name}: not a PNG")
            continue
        figures.append(
            {
                "name": path.name,
                "url": "data:image/png;base64," + base64.b64encode(raw).decode("ascii"),
                "sha256": hashlib.sha256(raw).hexdigest(),
            }
        )
    return figures, notes


def build_attempt(
    selection: dict,
    *,
    summary: dict | None,
    exit_code: int | None,
    wall_s: float,
    stdout: bytes,
    stderr: bytes,
    stdout_truncated: bool = False,
    stderr_truncated: bool = False,
    figures: list[dict] | None = None,
    problems: list[str] | None = None,
) -> dict:
    """Attempt dict matching roomModel.Attempt for a live rerun.

    record_type is "versioned" (the run produced full artifacts) but the
    criterion is null like legacy attempts: the rerun replays an old
    script and never claims criterion pinning. execution_mode/replay_of
    mark provenance, same as the local runner's live attempts.

    `problems` carries runner-level validation failures (unparseable or
    mismatched SUMMARY_JSON); like audit_room.versioned_attempt, any
    problem flips checks.state to "failed" with a deduped problem list.
    """
    script_text = selection["script_text"]
    checks = audit_room.checks(summary, selection["claim_id"])
    if problems:
        checks = {
            **checks,
            "state": "failed",
            "problems": list(dict.fromkeys(checks["problems"] + list(problems))),
        }
    return {
        "id": uuid.uuid4().hex,
        "number": (selection.get("number") or 0) + 1,
        "run_id": None,
        "source": "live_rerun",
        "record_type": "versioned",
        "created_at": _utcnow(),
        "script": {
            "name": "script.py",
            "text": script_text,
            "truncated": False,
            "sha256": hashlib.sha256(script_text.encode()).hexdigest(),
        },
        "criterion": None,
        "criterion_sha256": None,
        "feedback": {"human": "", "generated": "", "binding": "unavailable"},
        "summary": summary,
        "checks": checks,
        "exit_code": exit_code,
        "wall_s": round(wall_s, 3),
        "stdout": _tail_artifact("stdout.txt", stdout, stdout_truncated),
        "stderr": _tail_artifact("stderr.txt", stderr, stderr_truncated),
        "figures": figures or [],
        "weave_url": None,
        "execution": "completed",
        "execution_mode": "live_rerun",
        "replay_of": selection["attempt_id"],
    }


def synthesize_events(*stages: tuple[str, str]) -> list[dict]:
    """Event list matching the client contract: [{stage, message, ts}]."""
    return [{"stage": s, "message": m, "ts": _utcnow()} for s, m in stages]


def rate_ok(
    counters: dict[str, list[float]],
    key: str,
    limit: int,
    *,
    window_s: float = 86_400,
    now: float | None = None,
) -> bool:
    """Sliding-window check over a list of timestamps.

    Prunes entries older than window_s in place and reports whether the
    key is still under limit. The caller appends `now` and persists the
    counters when it accepts the job.
    """
    now = time.time() if now is None else now
    kept = [t for t in counters.get(key, []) if now - t < window_s]
    counters[key] = kept
    return len(kept) < limit

"""Build player-ready trace data for the landing-page trace player.

Reads the raw append-only traces and emits site/public/traces/*.json:
one entry per rendered line (dramatic-weighted time, class, message),
plus run metadata (wall time, LLM calls, milestones for the scrubber).

Presentation choices (documented because the trail is the deliverable):
- consecutive evidence logbook-cell commands are aggregated into one line
  each ("assembled N cells") — they are bookkeeping, not audit drama;
- everything else is preserved verbatim, including failed attempts;
- playback time is real elapsed time with per-gap clamps so 85s LLM calls
  don't deaden the scrubber.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

PAPERS = {
    "ca": {
        "dir": "jmlr-22-1228-ca-grokking",
        "title": "Grokking phase transitions in learning local rules",
        "source": "JMLR 22-1228",
        "judge": "PASS 5/5",
        "final": {"supported": 5, "falsified": 0, "inconclusive": 1},
    },
    "icl": {
        "dir": "arxiv-2510.10981",
        "title": "In-Context Learning Is Provably Bayesian Inference",
        "source": "arXiv 2510.10981",
        "judge": "PASS 5/5",
        "final": {"supported": 3, "falsified": 0, "inconclusive": 3},
    },
}

OUT_DIR = REPO / "site" / "public" / "traces"


def short_model(m: str) -> str:
    return m.split("/", 1)[-1] if m else "?"


def render(ev: dict) -> tuple[str, str] | None:
    """(message, css class) or None to aggregate/skip."""
    s, e = ev.get("stage"), ev.get("event")
    cid = ev.get("claim_id", "")

    if (s, e) == ("pipeline", "start"):
        return (
            f"lemma audit · {short_model(ev.get('model', ''))} · extract→audit→evidence→judge",
            "cmd",
        )
    if (s, e) == ("pipeline", "done"):
        mins = ev.get("elapsed_s", 0) / 60
        return f"run complete · {mins:.0f} min wall", "ok"
    if (s, e) == ("extract", "llm_call"):
        return "extract: model reads the paper", "info"
    if (s, e) == ("extract", "claims_written"):
        return (
            f"extract: {ev.get('n_claims')} testable claims, criteria taken from the claims themselves",
            "ok",
        )
    if e == "llm_call":
        return (
            f"{s}: model call · {short_model(ev.get('model', ''))} · {ev.get('duration_s', 0):.0f}s",
            "info",
        )
    if (s, e) == ("audit", "feedback_loaded"):
        return f"{cid}: reviewer feedback loaded ({ev.get('chars')} ch)", "warn"
    if (s, e) == ("audit", "script_written"):
        return (
            f"{cid}: wrote audit_attempt{ev.get('attempt')}.py ({ev.get('chars')} ch)",
            "cmd",
        )
    if (s, e) == ("audit", "tool_run"):
        return (
            f"{cid or s}: ran {ev.get('cmd', '')[:52]} · exit {ev.get('exit_code')} · {ev.get('duration_s', 0):.1f}s",
            "cmd",
        )
    if (s, e) == ("evidence", "tool_run"):
        return None  # aggregated
    if (s, e) == ("audit", "attempt_failed"):
        return (
            f"{cid}: attempt {ev.get('attempt')} FAILED · exit {ev.get('exit_code')}",
            "fail",
        )
    if (s, e) == ("audit", "claim_audited"):
        st = ev.get("status", "?")
        return f"{cid}: attempt {ev.get('attempt')} → {st}", st
    if (s, e) == ("audit", "claim_final"):
        st = ev.get("status", "?")
        return f"FINAL · {cid} → {st.upper()} · {ev.get('attempts')} attempt(s)", st
    if (s, e) == ("audit", "reviewer_reference_executed"):
        return (
            f"{cid}: ESCALATION — reviewer reference executed, its verdict wins",
            "warn",
        )
    if (s, e) == ("audit", "report"):
        return "audit: report merged into audit_report.json", "info"
    if (s, e) == ("paperclip", "cross_check"):
        return f"litcheck: {cid} · {ev.get('n_hits')} corpus hits (Paperclip)", "info"
    if e == "note":
        return f"{s}: {str(ev.get('text') or ev.get('note') or '')[:80]}", "info"
    return f"{s}/{e}", "info"


def build(key: str, meta: dict) -> None:
    path = REPO / "papers" / meta["dir"] / "trace.jsonl"
    events = [
        json.loads(line) for line in path.read_text().splitlines() if line.strip()
    ]

    lines: list[dict] = []
    prev_t = 0.0
    pending_evidence = 0
    n_llm = 0
    runs = 0

    def flush_evidence() -> None:
        nonlocal pending_evidence
        if pending_evidence:
            lines.append(
                {
                    "t": round(min(4.0, 0.4 * pending_evidence), 2),
                    "c": "info",
                    "m": f"evidence: assembled {pending_evidence} logbook cells",
                }
            )
            pending_evidence = 0

    for ev in events:
        if ev.get("event") == "llm_call":
            n_llm += 1
        if ev.get("stage") == "pipeline" and ev.get("event") == "start":
            runs += 1
        if (ev.get("stage"), ev.get("event")) == ("evidence", "tool_run"):
            pending_evidence += 1
            continue
        flush_evidence()

        out = render(ev)
        if out is None:
            continue
        msg, cls = out
        # dramatic weight: clamp real gap so the scrubber stays alive
        t = float(ev.get("elapsed_s", prev_t))
        gap = max(0.0, t - prev_t)
        weight = 0.25 if gap <= 0 else min(6.0, max(0.25, gap**0.55))
        prev_t = t
        entry = {"t": round(weight, 2), "c": cls, "m": msg}
        if msg.startswith("FINAL") or "ESCALATION" in msg:
            entry["k"] = 1  # milestone marker
        lines.append(entry)
    flush_evidence()

    total_wall = (
        sum(e.get("elapsed_s", 0) for e in events if e.get("event") == "done") / 60
    )
    data = {
        "key": key,
        "title": meta["title"],
        "source": meta["source"],
        "judge": meta["judge"],
        "final": meta["final"],
        "n_events": len(events),
        "n_llm": n_llm,
        "n_runs": runs,
        "wall_min": round(total_wall, 1),
        "lines": lines,
    }
    out_path = OUT_DIR / f"{key}.json"
    out_path.write_text(json.dumps(data, separators=(",", ":")) + "\n")
    print(
        f"{key}: {len(lines)} rendered lines ({len(events)} raw events) → {out_path.name} ({out_path.stat().st_size // 1024}KB)"
    )


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for key, meta in PAPERS.items():
        build(key, meta)
    return 0


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Generate site/src/data/beam.json + site/public/beam/ figures.

Data-driven from tmp/rl_rollouts/*.json, tmp/beam-audit-bench/scores.json
and tmp/bakeoff/bakeoff_results.json so the /beam page can never drift
from the published benchmark. Run before `npm --prefix site run build`.
"""

from __future__ import annotations

import contextlib
import json
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RL = REPO / "tmp" / "rl_rollouts"
OUT = REPO / "site" / "src" / "data" / "beam.json"
FIGDIR = REPO / "site" / "public" / "beam"

NAMES = {
    "BEAMHF": "Beam 1.0 501B",
    "KIMI": "Kimi K3",
    "KIMIHF": "Kimi K3",
    "RUNINFRA": "27B-class (RunInfra)",
    "ANTHROPIC": "Claude Opus 5.5",
}
TASK_NAMES = {
    "ca-grokking_c1": "CA C1 — 1D critical exponent (nu = 1.0)",
    "icl-bayes_c1": "ICL C1 — Bayesian risk identity (rel_diff = 0)",
    "ca-grokking_c2": "CA C2 — D-ball exponent series (nu = (D+1)/2)",
}


def main() -> int:
    episodes = []
    for f in sorted(RL.glob("*.json")):
        t = json.loads(f.read_text(encoding="utf-8"))
        steps = t.get("steps") or []
        best = float(t.get("best_reward", 0.0))
        best_attempt = next(
            (
                s.get("attempt")
                for s in steps
                if s.get("reward") == best and best >= 1.0
            ),
            None,
        )
        task = t.get("task_id", "")
        episodes.append(
            {
                "file": f.name,
                "provider": t.get("provider", ""),
                "provider_name": NAMES.get(
                    t.get("provider", ""), t.get("provider", "")
                ),
                "model": t.get("api_model", ""),
                "task": task,
                "task_name": TASK_NAMES.get(task, task),
                "full_context": bool(t.get("full_context")),
                "best_reward": best,
                "best_attempt": best_attempt,
                "n_attempts": t.get("n_attempts", len(steps)),
                "statuses": [s.get("status") for s in steps],
                "recovered": best >= 1.0,
            }
        )

    # per-provider success on keyed C1 tasks only
    agg: dict[str, dict[str, int]] = {}
    for e in episodes:
        if "_c1" in e["task"] and not e["full_context"]:
            a = agg.setdefault(e["provider"], {"ok": 0, "n": 0})
            a["n"] += 1
            a["ok"] += e["recovered"]
    aggregate = [
        {
            "provider": p,
            "provider_name": NAMES.get(p, p),
            "ok": v["ok"],
            "n": v["n"],
        }
        for p, v in sorted(agg.items())
    ]

    bakeoff = []
    bf = REPO / "tmp" / "bakeoff" / "bakeoff_results.json"
    if bf.is_file():
        d = json.loads(bf.read_text(encoding="utf-8"))
        hist = d.get("history", {})
        if isinstance(hist, dict):
            for model, row in hist.items():
                bakeoff.append(
                    {
                        "model": model,
                        "nu": row.get("nu"),
                        "ran": row.get("ran", False),
                        "note": row.get("note", ""),
                    }
                )
        beam = d.get("beam_2026-10-22") or {}
        res = beam.get("result") or {}
        first = beam.get("first_attempt_6000_budget") or {}
        if beam:
            bakeoff.append(
                {
                    "model": "beam_1_0_501b",
                    "nu": res.get("nu"),
                    "ran": res.get("ran", False),
                    "gen_s": res.get("gen_s"),
                    "note": "single generation after the 6000-budget trap"
                    if not first.get("ran", True)
                    else res.get("note", ""),
                }
            )

    # self-improvement numbers from the packaged scores
    improve = {}
    sc = REPO / "tmp" / "beam-audit-bench" / "scores.json"
    if sc.is_file():
        scores = json.loads(sc.read_text(encoding="utf-8"))
        ca = scores.get("beam-1.0-501b-clean-ca", {}).get("claims", {})
        c1 = ca.get("C1", {})
        atts = c1.get("attempts", [])
        vals = []
        for a in atts:
            checks = ((a.get("grade") or {}).get("checks")) or []
            for c in checks:
                m = json.dumps(c.get("rule", {}).get("metric", ""))
                d = c.get("detail", "")
                if "nu" in m and "=" in d:
                    with contextlib.suppress(ValueError, IndexError):
                        vals.append(float(d.split("]=", 1)[1].split(" ")[0]))
        improve = {
            "clean_nu": vals[0] if vals else None,
            "improved_nu": vals[-1] if len(vals) > 1 else None,
            "target": 1.0,
            "tol": 0.05,
        }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "episodes": episodes,
        "aggregate": aggregate,
        "bakeoff": bakeoff,
        "improve": improve,
        "n_episodes": len(episodes),
        "links": {
            "dataset": "https://huggingface.co/datasets/Papajams/beam-audit-bench",
            "space_ca": "https://huggingface.co/spaces/Papajams/beam-audit-ca-grokking",
            "space_icl": "https://huggingface.co/spaces/Papajams/beam-audit-icl-bayesian",
            "room": "https://lemmabio.netlify.app/room/",
            "github": "https://github.com/udirobert/lemma",
            "discussion": "https://huggingface.co/reflection/Beam-1.0-501B-A23B/discussions/5",
        },
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")

    # figures into site/public/beam/ (built by scripts/make_beam_figures.py)
    src_figs = REPO / "tmp" / "figures"
    if src_figs.is_dir():
        FIGDIR.mkdir(parents=True, exist_ok=True)
        for f in sorted(src_figs.glob("*.png")):
            shutil.copy(f, FIGDIR / f.name)
        print(f"copied {len(list(FIGDIR.glob('*.png')))} figures -> {FIGDIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

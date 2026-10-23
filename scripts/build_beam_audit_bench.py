#!/usr/bin/env python3
"""Package the Beam Audit Bench: ground-truth audit-script benchmark.

Builds a self-contained dataset from Lemma's recorded audits:

  tmp/beam-audit-bench/
    README.md                    dataset card (provenance, grading rules)
    tasks/<paper>_<cid>.json     claim, criterion, ground-truth expectations
    references/                  hand-verified reference scripts + answer keys
    model_runs/<model>/<paper>/<cid>/
                                 every attempt: script, summary or failure tail
    scores.json                  auto-graded leaderboard
    grader.py                    standalone re-grader (rerunnable on any run)

One "task" = one scientific claim with a machine-checkable success criterion
and (where a reviewer wrote one) a hand-verified reference implementation
whose recorded metrics ARE the answer key. A model's score for a task is
whether its best accepted attempt recovers the reference metrics within
tolerance, with a passing positive control and no self-contradictory verdict.

Usage:
  python scripts/build_beam_audit_bench.py \
      --run beam-1.0-501b=papers/jmlr-22-1228-ca-grokking-beam \
      --run beam-1.0-501b=papers/arxiv-2510.10981-beam \
      [--run qwen3.8-27b=...]
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent

PAPERS = {
    "ca-grokking": {
        "workdir": REPO / "papers" / "jmlr-22-1228-ca-grokking",
        "citation": (
            "Žunkovič & Ilievski, 'Grokking phase transitions in learning "
            "local rules with gradient descent', JMLR 25 (2024) 1-52 "
            "(arXiv:2210.15435)"
        ),
    },
    "icl-bayes": {
        "workdir": REPO / "papers" / "arxiv-2510.10981",
        "citation": "arXiv:2510.10981 — 'ICL Is Provably Bayesian Inference'",
    },
}

# Which workdir paper label a -beam comparison dir belongs to.
WORKDIR_TO_PAPER = {
    "jmlr-22-1228-ca-grokking": "ca-grokking",
    "arxiv-2510.10981": "icl-bayes",
}

# Hand-verified reviewer references (the answer keys), by (paper, claim).
REFERENCES = {
    ("icl-bayes", "C1"): REPO / "scripts" / "ref_c1_identity.py",
    ("ca-grokking", "C2"): REPO / "scripts" / "ref_c2_ball_exponent.py",
    ("icl-bayes", "C3"): REPO / "scripts" / "ref_c3_concentration.py",
    ("ca-grokking", "C4"): REPO / "scripts" / "ref_c4_pgrok.py",
    ("ca-grokking", "C6"): REPO / "scripts" / "ref_c6_bimodality.py",
}

# Attribution: in the RECORDED rounds (27B-class + Kimi, with human
# feedback.md), some final supported verdicts came from executing the
# reviewer reference, not from model-generated scripts. This is NOT
# hardcoded — reference_attempt_numbers() reads the trace's
# `reviewer_reference_executed` events and tags those attempts with
# source="reviewer_reference", so model-only scores exclude them.

# Ground-truth checks per (paper, claim). Each check: metric name, test,
# tolerance — derived from the reference's captured answer key (never from
# a model's self-report). Claims absent here have no reference and are
# graded only for control honesty (self-reported verdicts, flagged).
# Metric alias lists cover naming variance across implementations: the
# reference script's names plus every name observed in recorded rounds
# (27B, Kimi) and in clean-conditions model rounds. A check passes if ANY
# alias resolves and satisfies the rule; if no alias exists the check is
# "metric_missing", never silently passed.
GROUND_TRUTH = {
    ("ca-grokking", "C1"): [
        # paper's headline exponent: nu = 1.0 (Eq. 9/10)
        {
            "metric": ["nu", "nu_fitted", "nu_hat", "exponent"],
            "rule": "abs_close",
            "target": 1.0,
            "tol": 0.05,
        },
    ],
    ("ca-grokking", "C2"): [
        # nu = (D+1)/2 per dimension, reference recovers 1.500/2.9998/5.4993
        {
            "metric": ["nu_D2", "D2_nu", "nu_D_2"],
            "rule": "abs_close",
            "target": 1.5,
            "tol": 0.1,
        },
        {
            "metric": ["nu_D5", "D5_nu", "nu_D_5"],
            "rule": "abs_close",
            "target": 3.0,
            "tol": 0.1,
        },
        {
            "metric": ["nu_D10", "D10_nu", "nu_D_10"],
            "rule": "abs_close",
            "target": 5.5,
            "tol": 0.1,
        },
    ],
    ("ca-grokking", "C3"): [
        # reference round-4: P_grok(L1)=0.205 > P_grok(L2)=0.000 at D=10, eps=1.01
        {
            "metric": ["p_grok_l1", "pgrok_l1", "P_grok_L1", "p_l1", "prob_grok_l1"],
            "rule": "gt",
            "than": ["p_grok_l2", "pgrok_l2", "P_grok_L2", "p_l2", "prob_grok_l2"],
            "strict": True,
        },
    ],
    ("ca-grokking", "C4"): [
        # reference: P_grok(D) strictly decreasing, ln-linear slope < 0.
        # Implementations vary: boolean flag (reference: monotone_non_
        # increasing; recorded round: strictly_decreasing) plus a count of
        # measurable D points or the point list itself (chosen_P / P_grok).
        {
            "metric": [
                "monotone_non_increasing",
                "decreasing_pass",
                "monotonic_decrease",
                "strictly_decreasing",
            ],
            "rule": "truthy",
        },
        {
            "metric": ["n_measurable_points"],
            "rule": "at_least",
            "target": 3,
            "or_list_len": ["chosen_P", "P_grok"],
        },
    ],
    ("ca-grokking", "C6"): [
        # reference: all four bimodality tests pass (valley, separation,
        # slow-time vs analytic to 1e-16, sharpness). Simulation variants
        # report the structural fact as a detection boolean + cluster
        # fraction — accepted as a WEAKER key (grade_strength below), since
        # the grader cannot re-verify a variant's analytic slow-time formula.
        {"metric": ["all_bimodality_checks"], "rule": "all_true_or_missing"},
        {
            "metric": ["bimodal_detected", "bimodality_detected"],
            "rule": "truthy",
            "allow_missing_if": "all_bimodality_checks",
        },
        {
            "metric": ["frac_slow_cluster", "frac_slow", "p_slow"],
            "rule": "between",
            "lo": 0.01,
            "hi": 0.99,
            "allow_missing_if": "all_bimodality_checks",
        },
    ],
    ("icl-bayes", "C1"): [
        # reference Prop 3.1 identity: rel_diff 0.004 (tol 1%)
        {
            "metric": ["rel_diff", "relative_difference", "identity_rel_err"],
            "rule": "abs_close",
            "target": 0.0,
            "tol": 0.01,
        },
    ],
    ("icl-bayes", "C3"): [
        # reference Thm 3.3 mechanism: p_true(k=15) >= 0.9, ratio_15 <= 1.05
        {"metric": ["mean_p_true_15", "p_true_15"], "rule": "at_least", "target": 0.9},
        {"metric": ["ratio_15"], "rule": "at_most", "target": 1.05},
    ],
    ("icl-bayes", "C4"): [
        # recorded supported result: pN sweep slope in [-1.3, -0.5], r2 > 0.95
        {
            "metric": ["slope", "log_slope", "pN_slope"],
            "rule": "between",
            "lo": -1.3,
            "hi": -0.5,
        },
        {"metric": ["r2", "r_squared"], "rule": "at_least", "target": 0.95},
    ],
    # C5 (ca-grokking): gpu-small tensor-network regime — no CPU reference.
    # icl-bayes C2/C5/C6/C7: no hand-verified reference yet — self-report only.
}

# Key strength, for honest reporting in the leaderboard:
#  "reference" — the hand-verified reference's own metric form is checked
#                (closed-form, tight tolerances)
#  "variant"   — an alternative structural form is accepted (simulation
#                variant metrics; weaker because the grader cannot re-derive
#                the variant's formulas)
#  "none"      — no key; self-reported verdicts graded only for control honesty
KEY_STRENGTH = {
    ("ca-grokking", "C1"): "reference",
    ("ca-grokking", "C2"): "reference",
    ("ca-grokking", "C3"): "reference",
    ("ca-grokking", "C4"): "reference",
    ("ca-grokking", "C6"): "variant",
    ("icl-bayes", "C1"): "reference",
    ("icl-bayes", "C3"): "reference",
    ("icl-bayes", "C4"): "variant",  # bounds from the recorded round, no ref script
}

EVERY_CHECK_REQUIRES_CONTROL = True  # a passing control is part of every key


def load_claims(workdir: Path) -> dict[str, dict]:
    return {c["id"]: c for c in json.loads((workdir / "claims.json").read_text())}


def _get_metric(metrics: dict, names: list[str]):
    for n in names:
        if n in metrics and isinstance(metrics[n], (int, float, bool)):
            return metrics[n]
        if n in metrics:
            return metrics[n]
    # tolerate case/underscore variants
    norm = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in metrics.items()}
    for n in names:
        key = re.sub(r"[^a-z0-9]", "", n.lower())
        if key in norm:
            return norm[key]
    return None


def grade_summary(paper: str, cid: str, summary: dict) -> dict:
    """Grade one attempt's SUMMARY_JSON against the ground-truth key.

    Mirrors auditor._summary_problems: a verdict contradicting its own
    metrics (control failed but status=supported) is rejected outright.
    """
    status = summary.get("status")
    metrics = summary.get("metrics") or {}
    control = metrics.get("control_pass")
    out = {
        "status": status,
        "control_pass": control,
        "checks": [],
        "ground_truth_pass": None,
        "verdict_consistent": None,
        "score": 0.0,
    }

    key = GROUND_TRUTH.get((paper, cid))
    if key is None:
        out["has_reference"] = False
        # no answer key: can only score honesty. An inconclusive is honest;
        # a supported/falsified without a reference is self-reported.
        out["ground_truth_pass"] = None
        out["verdict_consistent"] = not (
            status in ("supported", "falsified") and control is not True
        )
        out["score"] = (
            0.5
            if status == "inconclusive"
            else (0.25 if out["verdict_consistent"] else 0.0)
        )
        out["note"] = (
            "no hand-verified reference for this claim; verdict is "
            "self-reported, graded only for control honesty"
        )
        return out

    out["has_reference"] = True
    ok = True
    n_binding = 0  # checks that actually constrain (not skipped)
    for chk in key:
        rule = chk["rule"]
        val = _get_metric(metrics, chk["metric"])
        passed = False
        skipped = False
        detail = f"{chk['metric']}={val!r} {rule}"

        # skip semantics: a check that allows absence does not fail the key
        # when its paired/alternative check is present — but it also cannot
        # be the only binding check.
        if val is None and rule == "all_true_or_missing":
            skipped = True
            detail += " (reference-style metric absent; alternative may bind)"
        elif val is None and chk.get("allow_missing_if"):
            alt = _get_metric(metrics, [chk["allow_missing_if"]])
            if alt is not None:
                skipped = True
                detail += f" (covered by {chk['allow_missing_if']}; skipped)"
        if not skipped:
            n_binding += 1
            if rule == "abs_close" and isinstance(val, (int, float)):
                passed = abs(val - chk["target"]) <= chk["tol"]
                detail += f" target={chk['target']}±{chk['tol']}"
            elif rule == "at_least":
                if isinstance(val, (int, float)):
                    passed = val >= chk["target"]
                    detail += f" >= {chk['target']}"
                elif chk.get("or_list_len"):
                    # accept len(<list metric>) as the count (variant naming)
                    lst = _get_metric(metrics, chk["or_list_len"])
                    if isinstance(lst, list):
                        passed = len(lst) >= chk["target"]
                        detail = (
                            f"len({chk['or_list_len']})={len(lst)} >= {chk['target']}"
                        )
            elif rule == "at_most" and isinstance(val, (int, float)):
                passed = val <= chk["target"]
                detail += f" <= {chk['target']}"
            elif rule == "between" and isinstance(val, (int, float)):
                passed = chk["lo"] <= val <= chk["hi"]
                detail += f" in [{chk['lo']}, {chk['hi']}]"
            elif rule == "truthy":
                passed = bool(val)
            elif rule in ("all_true", "all_true_or_missing") and isinstance(val, list):
                passed = bool(val) and all(bool(v) for v in val)
            elif rule == "gt":
                other = _get_metric(metrics, chk["than"])
                if isinstance(val, (int, float)) and isinstance(other, (int, float)):
                    passed = val > other if chk.get("strict") else val >= other
                    detail = f"{chk['metric']}={val!r} > {chk['than']}={other!r}"
            if val is None:
                detail += " (metric missing)"
        out["checks"].append(
            {
                "rule": chk,
                "passed": bool(passed),
                "skipped": skipped,
                "detail": detail,
            }
        )
        ok = ok and (skipped or passed)
    if n_binding == 0:
        ok = False  # nothing measurable was reported — cannot pass a key

    out["ground_truth_pass"] = bool(ok)
    # verdict consistency: supported requires passing control + passing key
    if status == "supported":
        out["verdict_consistent"] = bool(ok and control is True)
    elif status == "falsified":
        # a falsified claim whose own metrics actually pass the key is a
        # self-contradiction (the C4/C6 failure mode in recorded rounds)
        out["verdict_consistent"] = not ok
    else:  # inconclusive — always consistent when control or key failed
        out["verdict_consistent"] = True

    if ok and control is True and status == "supported":
        out["score"] = 1.0
    elif ok and control is True and status in ("falsified", "inconclusive"):
        # recovered the physics but mislabeled/under-claimed it
        out["score"] = 0.7
    elif ok:
        out["score"] = 0.5  # right numbers, failed or missing control
    else:
        out["score"] = 0.0
    return out


def reference_attempt_numbers(workdir: Path) -> dict[str, set[int]]:
    """Attempt numbers that were produced by executing the reviewer reference
    (trace event `reviewer_reference_executed`), keyed by claim id."""
    out: dict[str, set[int]] = {}
    tp = workdir / "trace.jsonl"
    if not tp.is_file():
        return out
    for line in tp.read_text(encoding="utf-8").splitlines():
        try:
            e = json.loads(line)
        except json.JSONDecodeError:
            continue
        if e.get("event") == "reviewer_reference_executed":
            cid = e.get("claim_id")
            att = e.get("attempt")
            if cid and isinstance(att, int):
                out.setdefault(cid, set()).add(att)
    return out


def collect_attempts(
    claim_dir: Path, ref_attempts: set[int] | None = None
) -> list[dict]:
    """Every attempt in a claim dir: accepted runs + failed tails + snapshots.

    Attempts whose number matches a trace-recorded reviewer-reference
    execution are tagged source="reviewer_reference" — their output is the
    hand-verified reference's, never the model's.
    """
    attempts = []
    for p in sorted(claim_dir.glob("run_attempt*.json")):
        if p.name.endswith(".failed.json"):
            continue
        rec = json.loads(p.read_text())
        att = rec.get("attempt")
        source = (
            "reviewer_reference"
            if ref_attempts and att in ref_attempts
            else "agent_generated"
        )
        attempts.append(
            {
                "attempt": att,
                "kind": "accepted",
                "source": source,
                "summary": rec.get("summary"),
                "exit_code": rec.get("exit_code"),
                "wall_s": rec.get("wall_s"),
            }
        )
    for p in sorted(claim_dir.glob("run_attempt*.failed.json")):
        rec = json.loads(p.read_text())
        attempts.append(
            {
                "attempt": rec.get("attempt"),
                "kind": rec.get("kind", "failed"),
                "source": "agent_generated",
                "summary": None,
                "exit_code": rec.get("exit_code"),
                "tail": (rec.get("tail") or "")[-500:],
            }
        )
    # immutable snapshots (records.py)
    if (claim_dir / "attempts").is_dir():
        for d in sorted((claim_dir / "attempts").glob("*")):
            occ = (
                json.loads((d / "outcome.json").read_text())
                if (d / "outcome.json").is_file()
                else {}
            )
            att = occ.get("attempt")
            attempts.append(
                {
                    "attempt": att,
                    "kind": "snapshot",
                    "source": (
                        "reviewer_reference"
                        if ref_attempts and att in ref_attempts
                        else "agent_generated"
                    ),
                    "uuid": d.name,
                    "summary": occ.get("summary"),
                    "exit_code": occ.get("exit_code"),
                    "wall_s": occ.get("wall_s"),
                }
            )
    return attempts


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--run",
        action="append",
        default=[],
        help="label=workdir (repeatable); workdir is a lemma audit dir",
    )
    ap.add_argument("--out", default=str(REPO / "tmp" / "beam-audit-bench"))
    args = ap.parse_args()
    if not args.run:
        ap.error("at least one --run label=workdir required")

    out = Path(args.out)
    if out.exists():
        shutil.rmtree(out)
    for sub in ("tasks", "references", "model_runs", "rl_env", "figures"):
        (out / sub).mkdir(parents=True)
    # presentation figures (scripts/make_beam_figures.py, dark-mode,
    # data-driven from tmp/rl_rollouts) — shipped inside the dataset so
    # README links never rot
    fig_src = REPO / "tmp" / "figures"
    if fig_src.is_dir():
        for f in sorted(fig_src.glob("*.png")):
            shutil.copy(f, out / "figures" / f.name)

    # ---- RL environment (gym-style, verifiable rewards) -------------------
    # The bench doubles as an RL environment: reset/step over a task with a
    # deterministic graded reward (no LLM judge), episode = up to 3 attempts,
    # observation = the grader's failed checks (same feedback Lemma's auditor
    # gives its next attempt). Ships the environment source, the reward spec,
    # and any recorded rollouts.
    env_src = REPO / "scripts" / "beam_audit_rl_env.py"
    if env_src.is_file():
        shutil.copy(env_src, out / "rl_env" / "beam_audit_rl_env.py")
    (out / "rl_env" / "reward_spec.json").write_text(
        json.dumps(
            {
                "episode": {
                    "max_attempts": 3,
                    "action": "full self-contained python audit script",
                    "observation": "exit code + stdout/stderr tail + failed key checks",
                },
                "reward": {
                    "0.00": "crash / timeout / no parseable SUMMARY_JSON",
                    "0.10": "ran + honest inconclusive (broken control, reported as such)",
                    "0.25": "self-contradictory verdict OR hardcoding/fabrication guard tripped",
                    "0.50": "reference metrics recovered but control failed/missing",
                    "0.70": "reference recovered but under-claimed (falsified/inconclusive)",
                    "1.00": "reference recovered + control passed + status supported",
                },
                "reward_hacking_guards": [
                    "positive control mandatory for any score >= 0.5",
                    "verdict-vs-metric contradictions rejected (grader mirrors Lemma's validator)",
                    "reported metric exactly equal to an answer-key literal found in the script source -> recovery credit withheld (a genuine fit lands at full precision and never equals the rounded target; key literals may legitimately appear in source in full-context mode when the paper states its own predictions)",
                    "decisive metrics reported with no numerical computation in the script -> recovery credit withheld",
                ],
                "usage": [
                    "from beam_audit_rl_env import BeamAuditEnv, rollout",
                    "env = BeamAuditEnv('ca-grokking_c1'); obs, info = env.reset()",
                    "obs, reward, done, info = env.step(script)",
                    "rollout('ca-grokking_c1', provider='BEAMHF')  # full episode via any OpenAI-compatible endpoint",
                ],
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    rl_dir = REPO / "tmp" / "rl_rollouts"
    if rl_dir.is_dir():
        for f in sorted(rl_dir.glob("*.json")):
            shutil.copy(f, out / "rl_env" / f"rollout_{f.name}")

    runs: list[tuple[str, Path]] = []
    for spec in args.run:
        label, _, wd = spec.partition("=")
        runs.append((label.strip(), (REPO / wd.strip()).resolve()))

    # ---- tasks + references (identical for every model) -------------------
    for (paper, cid), ref in REFERENCES.items():
        dest = out / "references" / f"{paper}_{cid.lower()}_{ref.name}"
        shutil.copy(ref, dest)

    # recorded reference answer keys (captured 2026-10-22, see tmp/refkeys/)
    captured = {}
    for (paper, cid), ref in REFERENCES.items():
        cap = REPO / "tmp" / "refkeys" / f"{ref.stem}.json"
        if cap.is_file():
            captured[f"{paper}/{cid}"] = json.loads(cap.read_text())

    tasks = {}
    for paper, meta in PAPERS.items():
        claims = load_claims(meta["workdir"])
        for cid, c in claims.items():
            tid = f"{paper}_{cid.lower()}"
            tasks[tid] = {
                "task_id": tid,
                "paper": paper,
                "citation": meta["citation"],
                "claim_id": cid,
                "statement": c["statement"],
                "success_criterion": c.get("success_criterion"),
                "testable": c.get("testable"),
                "has_reference": (paper, cid) in REFERENCES,
                "reference_summary": captured.get(f"{paper}/{cid}"),
            }
            (out / "tasks" / f"{tid}.json").write_text(
                json.dumps(tasks[tid], indent=2), encoding="utf-8"
            )

    # ---- model runs ---------------------------------------------------------
    scores: dict[str, dict] = {}
    for label, wd in runs:
        paper = None
        for frag, p in WORKDIR_TO_PAPER.items():
            if wd.name.startswith(frag):
                paper = p
        if paper is None:
            print(
                f"!! cannot map workdir {wd.name} to a paper; skipping", file=sys.stderr
            )
            continue
        claims = load_claims(wd)
        mdir = out / "model_runs" / label / paper
        mdir.mkdir(parents=True, exist_ok=True)
        report_path = wd / "results" / "audit_report.json"
        report = (
            json.loads(report_path.read_text())
            if report_path.is_file()
            else {"claims": []}
        )
        rep_by_id = {c["id"]: c for c in report.get("claims", [])}

        ref_attempts_by_cid = reference_attempt_numbers(wd)
        per_claim = {}
        for cid, c in claims.items():
            cdir = wd / "results" / cid.lower()
            if not cdir.is_dir():
                continue
            attempts = collect_attempts(cdir, ref_attempts_by_cid.get(cid))
            graded = []
            for a in attempts:
                g = {
                    "attempt": a.get("attempt"),
                    "kind": a["kind"],
                    "source": a.get("source", "agent_generated"),
                    "exit_code": a.get("exit_code"),
                    "wall_s": a.get("wall_s"),
                }
                if a.get("summary"):
                    g["grade"] = grade_summary(paper, cid, a["summary"])
                    g["status"] = a["summary"].get("status")
                else:
                    g["status"] = "crashed_or_rejected"
                    g["tail"] = a.get("tail", "")
                graded.append(g)

            final = rep_by_id.get(cid, {}).get("status")
            all_scores = [g["grade"]["score"] for g in graded if g.get("grade")]
            model_scores = [
                g["grade"]["score"]
                for g in graded
                if g.get("grade") and g["source"] != "reviewer_reference"
            ]
            best = max(all_scores, default=0.0)
            best_model = max(model_scores, default=0.0)
            per_claim[cid] = {
                "claim": c["statement"][:200],
                "final_status": final,
                "attempts": graded,
                "best_score": best,
                # model-only: excludes attempts produced by the hand-verified
                # reviewer reference (tagged via trace events)
                "best_score_model_only": best_model,
                "key_strength": KEY_STRENGTH.get((paper, cid), "none"),
            }
            # persist attempt scripts + graded summaries per claim
            ctd = mdir / cid.lower()
            ctd.mkdir(parents=True, exist_ok=True)
            for p in sorted(cdir.glob("audit_attempt*.py")):
                shutil.copy(p, ctd / p.name)
            (ctd / "graded.json").write_text(
                json.dumps(per_claim[cid], indent=1), encoding="utf-8"
            )

        n_tasks = len(per_claim)
        n_ref = sum(
            1 for cid in per_claim if KEY_STRENGTH.get((paper, cid)) == "reference"
        )
        full = sum(
            1
            for cid, v in per_claim.items()
            if v["key_strength"] == "reference" and v["best_score"] == 1.0
        )
        # model-only: excludes attempts produced by the hand-verified
        # reviewer reference (tagged source="reviewer_reference" from trace)
        full_model_only = sum(
            1
            for cid, v in per_claim.items()
            if v["key_strength"] == "reference" and v["best_score_model_only"] == 1.0
        )
        n_via_ref = sum(
            1
            for v in per_claim.values()
            if v["key_strength"] == "reference"
            and v["best_score"] == 1.0
            and v["best_score_model_only"] < 1.0
        )
        n_keyed = sum(
            1 for cid in per_claim if KEY_STRENGTH.get((paper, cid), "none") != "none"
        )
        scores[label] = {
            "paper": paper,
            "claims": per_claim,
            "n_claims": n_tasks,
            "n_with_reference": n_ref,
            "n_recovered_reference": full,
            "reference_recovery_rate": round(full / n_ref, 3) if n_ref else None,
            "n_recovered_model_only": full_model_only,
            "n_final_via_reviewer_reference": n_via_ref,
            "n_keyed": n_keyed,
        }

    (out / "scores.json").write_text(json.dumps(scores, indent=1), encoding="utf-8")

    # ---- grader.py (standalone re-run) + README ---------------------------
    grader_src = (REPO / "scripts" / "build_beam_audit_bench.py").read_text()
    # the grader is this file minus the packaging: ship the whole thing, it
    # re-derives scores from any --run workdir.
    (out / "grader.py").write_text(grader_src, encoding="utf-8")

    readme = build_readme(scores, runs)
    (out / "README.md").write_text(readme, encoding="utf-8")

    print(f"packed {len(runs)} model run(s), {len(tasks)} tasks -> {out}")
    print("\n=== LEADERBOARD (reference recovery) ===")
    for label, s in scores.items():
        print(
            f"  {label:22s} {s['paper']:12s} refs={s['n_with_reference']} "
            f"recovered={s['n_recovered_reference']} "
            f"rate={s['reference_recovery_rate']}"
        )
    return 0


def rl_episode_lines() -> list[str]:
    """One README line per recorded rollout, derived from the actual files.

    Reports best reward and the attempt at which it was first reached, so
    the README can never drift from what's published in rl_env/. Multiple
    runs of the same task are indexed (r1, r2, ...) and all kept — no
    cherry-picking.
    """
    rl_dir = REPO / "tmp" / "rl_rollouts"
    lines: list[str] = []
    if not rl_dir.is_dir():
        return ["  (no recorded rollouts found in tmp/rl_rollouts/)"]
    # provider -> [successes, total] on keyed C1 tasks
    agg: dict[str, list[int]] = {}
    for f in sorted(rl_dir.glob("*.json")):
        try:
            t = json.loads(f.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        steps = t.get("steps") or []
        best = t.get("best_reward", 0.0)
        n_att = t.get("n_attempts", len(steps))
        label = f.stem.replace("rollout_", "")
        full = " (full paper context)" if t.get("full_context") else ""
        # aggregate per-provider success on keyed C1 tasks only (the
        # ablation pair is graded separately in the prose below); label
        # format is <task>_c1_<PROVIDER>[_rN]
        if "_c1_" in label and not t.get("full_context"):
            provider = label.split("_c1_")[1].split("_r")[0]
            counts = agg.setdefault(provider, [0, 0])
            counts[1] += 1
            counts[0] += best >= 1.0
        best_attempt = next(
            (s.get("attempt") for s in steps if s.get("reward") == best), None
        )
        if best >= 1.0:
            lines.append(
                f"  - `{label}`: reward {best} reached on attempt "
                f"{best_attempt} of {n_att}{full}"
            )
        else:
            lines.append(
                f"  - `{label}`: never recovered the key — best reward "
                f"{best} after {n_att} attempts{full}"
            )
    if agg:
        parts = [f"{p} {ok}/{n}" for p, (ok, n) in sorted(agg.items())]
        lines.append("")
        lines.append(
            "  Episode success on keyed C1 tasks: "
            + "; ".join(parts)
            + ". All runs are published — including episodes that never "
            "recovered (crashes, missing metrics, off-target fits)."
        )
    return lines or ["  (no recorded rollouts found in tmp/rl_rollouts/)"]


def build_readme(scores: dict, runs: list[tuple[str, Path]]) -> str:
    lines = [
        "# Beam Audit Bench",
        "",
        "**13 claims** across two theory papers · **8 with machine-checkable answer keys** ·",
        "**deterministic grader, no LLM judge** · **4 models, 20 recorded RL episodes** ·",
        "**every attempt preserved — crashes included**",
        "",
        "Ground-truth benchmark of *audit-script generation*: can a model read a",
        "scientific claim and write self-contained numerical code that recovers",
        "the paper's result — with a mandatory positive control?",
        "",
        "![Episode success ladder](figures/episode_ladder.png)",
        "*Keyed-task episode success in the RL environment. Beam 4/4; Claude",
        "Opus 5.5 0/4 on the same tasks — a contract failure, not a science",
        "failure (its scripts are in `rl_env/`). All episodes published.*",
        "",
        "![Long-context ablation](figures/context_ablation.png)",
        "*CA C2 (D-ball exponents): for Beam the derivation sits past the 20k-",
        "char excerpt window; the full paper unlocks it. Opus's excerpt",
        "success is the honest counterpoint — also published.*",
        "",
        "![Self-improvement arc](figures/improve_arc.png)",
        "*Beam self-critique -> Beam re-audit flips CA C1 from falsified",
        "(nu = 1.476) to supported (nu = 1.0044) with zero human feedback;",
        "verify_gain-flagged.*",
        "",
        "Built from [Lemma](https://github.com/udirobert/lemma), an AI-scientist",
        "pipeline that audits scientific claims and distrusts itself.",
        "",
        "## Why this benchmark exists",
        "",
        "During the Lemma audits of two theory papers we hit a hard capability",
        "boundary: free 27B-class models could not implement the papers' closed-",
        "form dynamics (they produced power-law exponents of ~6000 where the",
        "correct answer is 1.5-5.5), while stronger reasoning models recovered",
        "them in one generation. Those claims have **hand-verified reference",
        "implementations** written by a human reviewer — the reference's",
        "recorded metrics are the answer key, so grading is automatic and",
        "needs no LLM judge.",
        "",
        "## Tasks",
        "",
        "13 claims across two papers:",
        "",
        "- `ca-grokking` — Žunkovič & Ilievski, JMLR 25 (2024), arXiv:2210.15435:",
        "  grokking phase transitions in local-rule learning (C1-C6; C5 is",
        "  gpu-small and carries no CPU reference).",
        "- `icl-bayes` — arXiv:2510.10981, 'ICL Is Provably Bayesian Inference'",
        "  (C1-C7).",
        "",
        "8 claims carry machine-checkable answer keys (`key_strength` per",
        "claim). Of those, 5 also ship the reviewer's hand-verified reference",
        "implementation (`references/`) with its captured answer key in the",
        "task card's `reference_summary`; the other 3 use exact closed-form",
        "targets from the paper's theorem (e.g. nu = 1.0). The remaining",
        "claims have no key and are graded for control honesty only —",
        "their verdicts are flagged self-reported.",
        "",
        "## Grading",
        "",
        "Each accepted attempt's `SUMMARY_JSON` is checked (see `grader.py`):",
        "",
        "1. **Ground truth**: metrics match the reference answer key within",
        "   tolerance (task-specific, e.g. |nu - 1.0| <= 0.05 for C1).",
        "2. **Positive control**: mandatory — a correct number with a failed or",
        "   missing control scores at most 0.5 (dropping the control is the",
        "   cheap way to game a criterion).",
        "3. **Verdict consistency**: `supported` with failing metrics, or",
        "   `falsified` while the metrics pass the key, is a self-contradiction",
        "   (a real failure mode we observed and now machine-check).",
        "",
        "Score 1.0 = reference recovered + control passed + verdict supported.",
        "0.7 = reference recovered but under-claimed (inconclusive/falsified).",
        "Failed and crashed attempts are preserved in `model_runs/` — cherry-",
        "picking is impossible because every attempt is recorded.",
        "",
        "## Headline results (2026-10-22, Beam 1.0 501B A23B via HF Inference Providers)",
        "",
        "**Bake-off gate** — the single task that broke every free 27B-class",
        "model in the original audits (implement the CA-grokking 1D closed-",
        "form dynamics, recover critical exponent nu = 1.0):",
        "",
        "| Model | Result | Note |",
        "|---|---|---|",
        "| RunInfra Qwen 27B | nu = 6779 | recorded bake-off, 2026-08-16 |",
        "| DeepSeek V4 Pro | nu = 6072 | recorded bake-off, 2026-08-16 |",
        "| Orca Flash / Pro | crashed / timed out | recorded bake-off, 2026-08-16 |",
        "| Kimi K3 | nu = 0.9911 | single generation, 92s, 2026-08-16 |",
        "| **Beam 1.0 501B** | **nu = 0.9974** | **single generation, 29.5s, 2026-10-22** |",
        "",
        "(Full data incl. the documented first-attempt budget trap: `bakeoff_results.json`.)",
        "",
        "**Clean-conditions flips** — claims the recorded 27B/Kimi rounds left",
        "`inconclusive` after multiple rounds of human feedback:",
        "",
        "- `icl-bayes/C2` (Bayes Gap decreases with context): **supported** by",
        "  Beam in 2 attempts, control passed.",
        "- `icl-bayes/C5` (Bayes Gap tracks input-distribution shift): **supported**",
        "  in 1 attempt (corr(W1, Bayes Gap) = 0.96).",
        "- `icl-bayes/C1` (risk identity R = RBG + RPV): Beam recovered it in",
        "  **one attempt**, rel_diff = 0.0008 — the recorded 27B rounds took",
        "  8 attempts plus a hand-written reviewer reference.",
        "",
        "**Honest misses (kept as falsified, not patched)** — Beam's own",
        "implementations of `ca-grokking` C1 and C2 produced wrong exponents",
        "(nu = 1.476 and 0.075 vs targets 1.0 and (D+1)/2) while their",
        "positive controls passed. Lemma's contract reports that as the",
        "model's implementation failing the claim's criterion — falsified,",
        "with the metrics visible for triage. (In the recorded rounds those",
        "same claims were recovered only after human feedback.md + reviewer",
        "references; see the `-recorded` runs.)",
        "",
        "**Control honesty is scored, not assumed**: `ca-grokking/C5` and",
        "`icl-bayes/C6`, `C7` came back `inconclusive` because their own",
        "positive controls failed — the validator refuses to score a verdict",
        "built on a broken statistic.",
        "",
        "**Self-improvement loop** — after the clean CA round, Lemma's",
        "`improve` loop ran with Beam in BOTH roles: reviewer-persona drafts",
        "`feedback.auto.md` from each failure's script/output/trace, then",
        "Beam re-audits with it as advisory input. No human wrote a word.",
        "Result: **C1 flipped falsified → supported** — Beam's reviewer",
        "diagnosed the exact bug (unshifted training distribution, fit over",
        "the whole pre-horizon instead of the asymptotic window) and Beam's",
        "auditor implemented the fix: nu = 1.0044 (target 1.0 ± 0.05),",
        "control slope exactly 1.0. The flip is flagged in the round record's",
        "`verify_gain` — any flip TO supported post-feedback is the",
        "reward-hacking direction, so it is marked for eyeballing and the",
        "metrics are published alongside it. C2/C4/C5 stayed put (honestly).",
        "",
        "**RL environment episodes** (`rl_env/rollout_*.json`) — reference-",
        "keyed tasks driven through the env by several models. One line per",
        "recorded episode, generated from the rollout files:",
        "",
        *rl_episode_lines(),
        "",
        "Keyed C1 tasks were run multiple times per model (r1 = first",
        "published episodes; r2 = re-runs after the env started recording",
        "every script source). Run-to-run variance is real and preserved:",
        "Beam reached reward 1.0 in every episode; the 27B-class model",
        "reached 1.0 in three of four — in the episode it failed, it",
        "crashed once, dropped the nu metric once, and fit nu = 0.5 once.",
        "Where a weaker model succeeded, early attempts often omit the",
        "positive control until the environment's observation says so —",
        "the control discipline is mechanically teachable.",
        "",
        "**Long-context ablation** (ca-grokking_c2, the ball-model exponent",
        "claim): with the pipeline's 20k-char excerpt, Beam re-derives the",
        "1D model and reports nu_D2/D5/D10 = 2/5/10 — best reward 0.0 in",
        "both runs. With the FULL paper (119,673 chars), Beam recovers the",
        "D-dimensional series (1.5001, 3.0174, 5.4117 vs targets 1.5/3.0/",
        "5.5): reward 1.0, control passed. For Beam, the D-ball derivation",
        "sits past the excerpt window; the long context is what unlocks the",
        "claim. (Claude Opus 5.5 recovered the same series once FROM the",
        "excerpt — the physics is derivable with less context by a model",
        "that gets the rest of the contract right; see the next section.)",
        "Trajectories record the full script source of every attempt, so",
        "each reward and integrity flag is inspectable.",
        "",
        "**Claude Opus 5.5 (closed frontier) in the same environment** —",
        "8 episodes (2 runs x ca_c1, icl_c1, ca_c2 excerpt, ca_c2 full),",
        "reward 1.0 in ONE: episode success is 1/8 vs Beam's 4/4. The",
        "failure mode is visible in the preserved scripts and is a",
        "*contract* failure, not a science failure: on the keyed C1 tasks",
        "Opus computed the physics but reported it under metric names",
        "outside the grader's alias list (median_nu, nu_se_bootstrap,",
        "rel_diff=not_computed) or returned inconclusive without",
        "attempting the computation (claim_quantity_computed=false) —",
        "across all 6 C1 attempts, never once emitting the key metric",
        "under a graded name. The environment's observation names the",
        "failed check each time; Opus did not converge on the naming",
        "contract within 3 attempts. The single 1.0 is the harder ca_c2",
        "task, recovered from the 20k excerpt alone with a genuine",
        "full-precision fit. Verifiable rewards measure more than",
        "physics: auditability — reporting what you measured under the",
        "contracted names, with the mandatory control — is part of the",
        "skill, and on this axis the 501B open model beats the closed",
        "frontier model.",
        "",
        "## Model runs",
        "",
    ]
    for label, s in scores.items():
        via_ref = s.get("n_final_via_reviewer_reference", 0)
        model_only = s.get("n_recovered_model_only", s["n_recovered_reference"])
        line = (
            f"- **{label}** on {s['paper']}: "
            f"{s['n_recovered_reference']}/{s['n_with_reference']} exact keys "
            f"recovered (rate {s['reference_recovery_rate']}) — of which "
            f"{model_only} by the model's own scripts"
        )
        if via_ref:
            line += f", {via_ref} via the hand-verified reviewer reference"
        line += (
            f"; {s.get('n_keyed', s['n_with_reference'])} claims keyed, "
            f"{s['n_claims']} audited"
        )
        lines.append(line)
    lines += [
        "",
        "Runs labelled `-clean` start agent-generated from scratch: the model",
        "sees only the claim and the paper text — no human feedback, no",
        "reference implementations, no prior attempts. The CA run includes",
        "one subsequent self-improvement round (Beam-authored advisory",
        "feedback, still zero human input; see above). Runs labelled",
        "`-recorded` are the original Lemma rounds (weaker model + iterative",
        "human feedback.md corrections + hand-verified reviewer references)",
        "included as the 'before' state of the same claims. All audits ran",
        "through Lemma's auditor (<=3 attempts per round, self-contained",
        "numpy scripts, mandatory positive control). Every attempt is",
        "preserved — including crashed and summary-rejected ones.",
        "",
        "## RL environment (verifiable rewards, no LLM judge)",
        "",
        "The bench doubles as an RL environment (`rl_env/beam_audit_rl_env.py`):",
        "each task is an episode of up to 3 attempts, the action is a full",
        "self-contained audit script, and the reward is the deterministic",
        "grader above — never an LLM judge. The observation after each failed",
        "attempt is exactly what Lemma's auditor feeds its next attempt: exit",
        "code, output tail, and which key checks failed. Reward ladder and",
        "reward-hacking guards are in `rl_env/reward_spec.json`; recorded",
        "episodes (Beam vs a 27B-class model, same tasks) are in",
        "`rl_env/rollout_*.json`.",
        "",
        "```python",
        "from beam_audit_rl_env import BeamAuditEnv, rollout",
        'env = BeamAuditEnv("ca-grokking_c1")',
        "obs, info = env.reset()",
        "obs, reward, done, info = env.step(script_source)",
        "# or drive a full episode with any OpenAI-compatible endpoint:",
        'rollout("ca-grokking_c1", provider="BEAMHF")',
        "```",
        "",
        "### Training against this environment",
        "",
        "The environment is deliberately consumable as-is by an RL stack:",
        "",
        "- **Action space**: one string — a complete self-contained Python",
        "  audit script. Multi-turn: up to 3 attempts per episode, so a",
        "  trajectory is (prompt, script, obs) x3 at most.",
        "- **Observation**: exit code + stdout/stderr tail + the grader's",
        "  failed key checks (identical to Lemma's auditor feedback, so",
        "  skills learned here transfer to the full pipeline).",
        "- **Reward**: the deterministic 6-level ladder in",
        "  `rl_env/reward_spec.json` — no LLM in the loop, so rewards are",
        "  reproducible and safe to average across GRPO group rollouts.",
        "  The 0.25 caps for hardcoding/fabrication are part of the reward",
        "  function, not a post-hoc filter.",
        "- **Task list**: any claim whose `key_strength` is `reference`",
        "  (hand-verified implementation's metrics as the key) or",
        "  `variant` (closed-form targets from the paper's own theorem)",
        "  has a machine-checkable answer — 8 of 13 claims. The rest are",
        "  honesty-only tasks (useful for shaping against degenerate",
        "  policies but not for the main reward).",
        "- **Sandbox note**: `step()` runs scripts with a wall-time cap in",
        "  a temp dir but does not OS-isolate them (that is Lemma's",
        "  `room_sandbox.py` territory). For training, point the env at a",
        "  container or sandbox with network disabled before scaling",
        "  rollouts.",
        "- **Suggested GRPO setup**: sample K scripts per attempt, grade",
        "  each independently (the grader is pure and fast — see",
        "  `grader.py`), and use the per-attempt reward as the group",
        "  signal. The recorded `rl_env/rollout_*.json` trajectories are",
        "  a cold-start SFT source: they pair real observations with",
        "  scripts that earned each reward level.",
        "",
        "## License & provenance",
        "",
        "Task cards and grading rules: CC-BY-4.0. Generated scripts: preserved",
        "verbatim as model output. Reference implementations: written by the",
        "Lemma reviewer (Udi / Papajams), hand-verified before acceptance.",
        "",
    ]
    front = [
        "---",
        "license: cc-by-4.0",
        "task_categories:",
        "- text-generation",
        "tags:",
        "- benchmark",
        "- evaluation",
        "- reflection-beam",
        "- agentic-science",
        "- claim-audit",
        "- reproducibility",
        "size_categories:",
        "- n<1K",
        "---",
        "",
    ]
    return "\n".join(front + lines)


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Beam Audit RL environment — verifiable rewards for scientific codegen.

An RL environment (gym-style reset/step, no gym dependency) over the Beam
Audit Bench tasks. The reward is NOT an LLM judge: it is the bench's
deterministic grader — each task's metrics are checked against a
hand-verified reference answer key, a positive control is mandatory, and
self-contradictory verdicts are rejected. This makes it an "RL environment
for codegen with verifiable rewards" in the same family as math-verify
graders, but for scientific-audit scripts.

Episode structure (mirrors Lemma's auditor contract):
  - up to 3 attempts per task (MAX_ATTEMPTS)
  - action  : a full self-contained Python script (numpy/scipy/matplotlib,
              seeded, CPU-fast, prints SUMMARY_JSON=<json>)
  - run     : executed in a fresh temp dir with a wall-time cap
  - observe : exit code, stdout/stderr tail, and — when the script ran —
              the grader's failed checks (the same feedback the auditor
              passes to its next attempt)
  - reward  :
      0.00  crash / timeout / no parseable SUMMARY_JSON
      0.10  ran + honest inconclusive (broken control, reported as such)
      0.25  ran + self-contradictory verdict (gaming attempt, penalised)
      0.50  reference metrics recovered but control failed/missing
      0.70  reference recovered but under-claimed (falsified/inconclusive)
      1.00  reference recovered + control passed + status supported
  - done    : after a 1.0 reward, or after MAX_ATTEMPTS

Integrity flags in `info` (surfaced, not silently judged):
  - "key_literals_in_source": the script source contains literal target
    values of the answer key (informational — in full-context mode the
    paper itself states its predictions, so a legit script may compare
    against them)
  - "hardcoded_targets": a REPORTED metric equals an answer-key target
    literal that also appears in the script source, to full precision.
    A genuine fit/simulation lands at full precision (1.4939..., 3.0044
    ...) and never equals the rounded target; copying the constant does.
    This is the reward-hacking signature the reward cap keys on.
  - "contradictory": grader detected verdict-vs-metric self-contradiction

Usage:
  from scripts.beam_audit_rl_env import BeamAuditEnv, rollout

  env = BeamAuditEnv("ca-grokking_c1")
  obs, info = env.reset()
  obs, reward, done, info = env.step(script_source)

  # drive it with any OpenAI-compatible endpoint via the agent stack:
  rollout("ca-grokking_c1", provider="BEAMHF")

Environment isolation: each step runs in a fresh temp dir with a wall-time
cap and no inherited proxy env vars. Full OS-level sandboxing (network,
fs) is Lemma's room_sandbox.py territory; the environment documents the
trust boundary rather than pretending to enforce it.
"""

from __future__ import annotations

import contextlib
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts"))

from build_beam_audit_bench import (  # noqa: E402
    GROUND_TRUTH,
    KEY_STRENGTH,
    PAPERS,
    _get_metric,
    grade_summary,
    load_claims,
)

MAX_ATTEMPTS = 3
RUN_TIMEOUT_S = 600
MAX_OBS_CHARS = 2000


class BeamAuditEnv:
    """One audit task as an episode with deterministic graded rewards."""

    def __init__(self, task_id: str, paper: str | None = None):
        # task_id like "ca-grokking_c1" or ("ca-grokking", "C1")
        if paper is None:
            paper, _, cid = task_id.rpartition("_")
            cid = cid.upper()
        else:
            cid = task_id.upper()
        if paper not in PAPERS:
            raise KeyError(f"unknown paper {paper!r}; one of {sorted(PAPERS)}")
        self.paper = paper
        self.cid = cid
        workdir = PAPERS[paper]["workdir"]
        claims = load_claims(workdir)
        if cid not in claims:
            raise KeyError(f"claim {cid} not in {paper}")
        self.claim = claims[cid]
        self.key_strength = KEY_STRENGTH.get((paper, cid), "none")
        self.ground_truth = GROUND_TRUTH.get((paper, cid))
        self.attempt = 0
        self.history: list[str] = []
        self.total_reward = 0.0
        self.done = False

    # ---- gym-style API -------------------------------------------------

    def reset(self) -> tuple[str, dict]:
        self.attempt = 0
        self.history = []
        self.total_reward = 0.0
        self.done = False
        return self._initial_obs(), {
            "task_id": f"{self.paper}_{self.cid.lower()}",
            "key_strength": self.key_strength,
        }

    def _initial_obs(self) -> str:
        c = self.claim
        return (
            f"Audit claim {c['id']}: {c.get('title', '')}\n"
            f"Statement: {c['statement']}\n"
            f"Paper evidence it rests on: {c.get('evidence_in_paper', '')}\n"
            f"Test plan from extraction: {c.get('test_plan', '')}\n"
            f"Success criterion: {c.get('success_criterion', '')}\n"
            f"Attempt {self.attempt + 1} of {MAX_ATTEMPTS}."
        )

    def step(self, script: str) -> tuple[str, float, bool, dict]:
        if self.done:
            raise RuntimeError("episode done; call reset()")
        self.attempt += 1
        info: dict = {"attempt": self.attempt, "script": script}

        integrity = self._integrity_flags(script)
        info.update(integrity)

        exit_code, out, err, wall = self._run(script)
        info["exit_code"] = exit_code
        info["wall_s"] = round(wall, 1)

        summary = self._parse_summary(out)
        if summary is None:
            reward = 0.0
            obs = (
                f"Attempt {self.attempt} FAILED (exit {exit_code}, {wall:.1f}s).\n"
                f"stdout tail: {out[-MAX_OBS_CHARS:]}\n"
                f"stderr tail: {err[-MAX_OBS_CHARS:]}"
            )
            info["status"] = "crash"
        else:
            grade = grade_summary(self.paper, self.cid, summary)
            info["grade"] = grade
            info["status"] = summary.get("status")
            reward = float(grade["score"])
            # Refinement over the v1 literal-presence guard: in full-context
            # episodes the paper itself states the predicted values, so a
            # legit script may hold those literals (e.g. to compare the fit
            # against theory). The reward-hacking signature is a REPORTED
            # metric exactly equal to one of those literals — a genuine fit
            # lands at full precision and never equals the rounded target.
            exact_hits = self._hardcoded_reported(
                summary.get("metrics") or {},
                integrity.get("key_literals_in_source") or [],
            )
            if exact_hits:
                info["hardcoded_targets"] = exact_hits
            if exact_hits and reward >= 0.5:
                reward = 0.25
                info["grade"]["score"] = reward
                info["grade"]["note"] = (
                    "reported metrics exactly equal answer-key literals that "
                    "appear in the script source; recovery credit withheld "
                    "(reward-hacking guard)"
                )
            elif integrity.get("no_computation") and reward >= 0.5:
                # a script that reports passing numbers without computing
                # anything gets fabrication credit only
                reward = 0.25
                info["grade"]["score"] = reward
                info["grade"]["note"] = (
                    "script reports decisive metrics but performs no "
                    "numerical computation; recovery credit withheld "
                    "(fabrication guard)"
                )
            failed = [
                c["detail"]
                for c in grade["checks"]
                if not c["passed"] and not c.get("skipped")
            ]
            obs = (
                f"Attempt {self.attempt} ran (exit {exit_code}, {wall:.1f}s); "
                f"status={summary.get('status')}, graded score={reward}.\n"
                + (
                    "Failed key checks:\n- " + "\n- ".join(failed)
                    if failed
                    else "All key checks passed."
                    if grade["ground_truth_pass"]
                    else "No answer key for this claim; control honesty scored."
                )
            )
            # observability: a supported verdict needs an explicit passing
            # control in metrics — say so when that is the only thing missing,
            # otherwise the agent cannot learn the control-pass requirement.
            if (
                grade["ground_truth_pass"]
                and not grade["verdict_consistent"]
                and summary.get("status") == "supported"
            ):
                obs += (
                    "\nNote: the key checks pass, but metrics lack a passing "
                    "positive control (control_pass=true). A supported verdict "
                    "requires an explicit positive control in metrics."
                )

        self.total_reward = max(self.total_reward, reward)
        self.history.append(f"--- attempt {self.attempt}: reward={reward}\n{obs}")
        if reward >= 1.0 or self.attempt >= MAX_ATTEMPTS:
            self.done = True
        else:
            obs += (
                f"\n\nWrite the corrected full script (attempt "
                f"{self.attempt + 1} of {MAX_ATTEMPTS}). Fix the actual "
                f"failure shown above."
            )
        return obs, reward, self.done, info

    # ---- mechanics -------------------------------------------------------

    def _integrity_flags(self, script: str) -> dict:
        """Find distinctive answer-key target literals in the script source.

        Informational only (see step()): in full-context episodes the paper
        itself states its predictions, so a literal's presence is not proof
        of hardcoding — it is the *reported metric equal to the literal*
        that is. Conservative by design: only abs_close targets and between
        bounds count (tolerances and at_least/at_most thresholds are too
        generic — a 0.01 tolerance matches every script that uses 0.01 for
        anything). Integer-valued targets must appear with an explicit ".0"
        so a bare loop bound like 3 never trips the scan.
        """
        flags: dict = {}
        if not self.ground_truth:
            return flags
        literals: list[str] = []
        for chk in self.ground_truth:
            rule = chk.get("rule")
            cands = (
                [chk.get("target")]
                if rule == "abs_close"
                else [chk.get("lo"), chk.get("hi")]
                if rule == "between"
                else []
            )
            for v in cands:
                if not isinstance(v, (int, float)) or isinstance(v, bool):
                    continue
                if v in (0, 1):
                    continue
                if float(v).is_integer():
                    literals.append(f"{int(v)}.0")  # require explicit .0
                else:
                    literals.append(f"{v:g}")
        hits = sorted(
            {
                lit
                for lit in literals
                if re.search(r"(?<![\d.])" + re.escape(lit) + r"(?![\d.])", script)
            }
        )
        if hits:
            flags["key_literals_in_source"] = hits

        # Fabrication guard: a script that reports a decisive metric but
        # performs no numerical computation (no numpy/scipy/math array work,
        # no arithmetic over data) is almost certainly emitting a constant it
        # got from the prompt/answer key rather than deriving it. This closes
        # the hole the literal guard can't — targets of 0.0/1.0 are too
        # generic to string-match. The auditor prompt mandates numpy/scipy/
        # matplotlib, so a legit audit always imports one.
        does_math = bool(
            re.search(
                r"import\s+numpy|from\s+numpy|import\s+scipy|from\s+scipy|"
                r"import\s+math\b|from\s+math\b|import\s+random\b|"
                r"\bnp\.|\bnumpy\.|scipy\.",
                script,
            )
        )
        if not does_math:
            flags["no_computation"] = True
        return flags

    def _hardcoded_reported(self, metrics: dict, literals: list[str]) -> dict:
        """Answer-key literals that a REPORTED metric equals exactly.

        Uses the same alias/normalization resolution as the grader
        (_get_metric). A metric produced by a real fit or simulation lands at
        full float precision (e.g. 1.4939295367104313) and cannot equal a
        rounded literal like "1.5"; a metric copied from the key does.
        """
        hits: dict[str, str] = {}
        if not (self.ground_truth and literals):
            return hits
        lit_vals = {float(v) for v in literals}
        for chk in self.ground_truth:
            names = (
                chk["metric"] if isinstance(chk["metric"], list) else [chk["metric"]]
            )
            val = _get_metric(metrics, names)
            if (
                isinstance(val, (int, float))
                and not isinstance(val, bool)
                and float(val) in lit_vals
            ):
                hits[str(val)] = "/".join(names)
        return hits

    def _run(self, script: str) -> tuple[int, str, str, float]:
        t0 = time.time()
        with tempfile.TemporaryDirectory(prefix="beam_audit_env_") as td:
            tdp = Path(td)
            (tdp / "results" / self.cid.lower()).mkdir(parents=True)
            sp = tdp / "script.py"
            sp.write_text(script, encoding="utf-8")
            import os

            env = {
                k: v for k, v in os.environ.items() if not k.lower().endswith("_proxy")
            }
            env["MPLBACKEND"] = "Agg"
            try:
                proc = subprocess.run(
                    [sys.executable, str(sp)],
                    cwd=td,
                    capture_output=True,
                    text=True,
                    timeout=RUN_TIMEOUT_S,
                    env=env,
                )
                return proc.returncode, proc.stdout, proc.stderr, time.time() - t0
            except subprocess.TimeoutExpired as e:
                out = (
                    e.stdout.decode()
                    if isinstance(e.stdout, bytes)
                    else (e.stdout or "")
                )
                err = (
                    e.stderr.decode()
                    if isinstance(e.stderr, bytes)
                    else (e.stderr or "")
                )
                return (
                    -1,
                    out,
                    f"TIMEOUT after {RUN_TIMEOUT_S}s\n{err}",
                    time.time() - t0,
                )

    @staticmethod
    def _parse_summary(stdout: str) -> dict | None:
        for line in stdout.splitlines():
            if line.startswith("SUMMARY_JSON="):
                with contextlib.suppress(json.JSONDecodeError):
                    data = json.loads(line[len("SUMMARY_JSON=") :])
                    if isinstance(data, dict):
                        return data
        return None


# ---- rollout helper --------------------------------------------------------


def rollout(
    task_id: str,
    provider: str = "BEAMHF",
    max_tokens: int = 16384,
    full_context: bool = False,
) -> dict:
    """Drive one full episode with an LLM endpoint; return the trajectory.

    Uses the auditor's exact SYSTEM prompt (agent.auditor.SYSTEM) for parity
    with the recorded audit rounds: the RL environment and the pipeline
    share one contract. With full_context=True the ENTIRE paper text is
    attached (not the pipeline's 20k-char excerpt) — the long-context
    ablation: does the model's implementation improve when it can read the
    derivation sections that sit past the excerpt window?
    """
    from dotenv import load_dotenv

    load_dotenv(REPO / ".env", override=True)
    from agent.auditor import SYSTEM
    from agent.llm import (
        _complete_with_retry,
        _endpoint_cfg,
        extract_json,
        model_name,
    )
    from agent.traces import Trace

    p = provider.upper()
    if p == "ANTHROPIC":
        # built-in provider path in agent/llm.py (ANTHROPIC_API_KEY +
        # ANTHROPIC_DEFAULT_MODEL), not an endpoint config
        import os

        if not os.environ.get("ANTHROPIC_API_KEY", "").strip():
            raise RuntimeError("ANTHROPIC_API_KEY not set in .env")
        api_model = model_name("anthropic")
    else:
        cfg = _endpoint_cfg(p)
        if not cfg:
            raise RuntimeError(f"endpoint {provider} not configured")
        api_model = cfg["model"]
    env = BeamAuditEnv(task_id)
    obs, info = env.reset()
    trace = Trace(f"rl-env-{task_id}-{time.strftime('%Y%m%d-%H%M%S')}")
    steps = []
    paper_text = ""
    with contextlib.suppress(Exception):
        pdfs = list(PAPERS[env.paper]["workdir"].glob("*.pdf"))
        if pdfs:
            import fitz

            doc = fitz.open(pdfs[0])
            paper_text = "\n".join(pg.get_text() for pg in doc)
    while True:
        user = obs
        if paper_text and info.get("attempt", 0) == 0 and not steps:
            body = paper_text if full_context else paper_text[:20000]
            label = "full text" if full_context else "excerpt"
            user += f"\n\nRelevant paper text ({label}, {len(body)} chars):\n{body}"
        raw, model, dur = _complete_with_retry(p, SYSTEM, user, max_tokens, 0.2)
        try:
            payload = extract_json(raw)
            script = payload["script"] if isinstance(payload, dict) else raw
        except (json.JSONDecodeError, KeyError, TypeError):
            script = raw
        obs, reward, done, step_info = env.step(script)
        steps.append(
            {
                "attempt": step_info["attempt"],
                "model": model,
                "gen_s": round(dur, 1),
                "reward": reward,
                "status": step_info.get("status"),
                "exit_code": step_info.get("exit_code"),
                "wall_s": step_info.get("wall_s"),
                "script": script,
                "integrity": {
                    k: step_info[k]
                    for k in (
                        "key_literals_in_source",
                        "hardcoded_targets",
                        "no_computation",
                    )
                    if step_info.get(k)
                },
                "grade": step_info.get("grade"),
            }
        )
        trace.tool_run(
            "rl_env",
            f"episode step {step_info['attempt']} reward={reward}",
            step_info.get("exit_code", 0),
            step_info.get("wall_s", 0.0),
            0,
        )
        if done:
            break
    return {
        "task_id": f"{env.paper}_{env.cid.lower()}",
        "provider": p,
        "api_model": api_model,
        "key_strength": env.key_strength,
        "full_context": full_context,
        "paper_chars": len(paper_text) if full_context else min(len(paper_text), 20000),
        "best_reward": env.total_reward,
        "n_attempts": env.attempt,
        "steps": steps,
    }


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("task_id")
    ap.add_argument("--provider", default="BEAMHF")
    ap.add_argument(
        "--full-context",
        action="store_true",
        help="attach the whole paper text instead of the 20k excerpt",
    )
    ap.add_argument("--out", help="write trajectory JSON here")
    args = ap.parse_args()
    result = rollout(
        args.task_id, provider=args.provider, full_context=args.full_context
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "steps"}
            | {
                "steps_summary": [
                    {
                        kk: s[kk]
                        for kk in ("attempt", "reward", "status", "exit_code", "wall_s")
                        if kk in s
                    }
                    for s in result["steps"]
                ]
            },
            indent=2,
        )
    )
    if args.out:
        Path(args.out).write_text(json.dumps(result, indent=2), encoding="utf-8")
        print(f"trajectory -> {args.out}")

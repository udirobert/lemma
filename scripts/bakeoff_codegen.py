#!/usr/bin/env python3
"""Model bake-off: can the candidate implement the paper's closed-form 1D
grokking dynamics and recover the critical exponent nu=1?

This is the exact capability that failed in round 2 (27B Qwen). Each model
gets the same self-contained task; we score by (a) does the generated script
run, (b) does it recover nu close to 1.0.

Usage: .venv/bin/python scripts/bakeoff_codegen.py [provider ...]
Defaults to the candidates worth testing.
"""

from __future__ import annotations

import contextlib
import json
import subprocess
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from agent.llm import _endpoint_cfg, _providers  # noqa: E402

TASK = """Write ONE self-contained Python script (numpy + matplotlib Agg only,
numpy seeded) that does this:

The 1D exponential grokking model. Training input mean xbar is small.
The bias evolves by the closed-form ODE solution:
  b(t) = xbar_l - (xbar_l - b0) * exp(-(1 + lam2) * t)
where xbar_l = (xbar - lam1) / (1 + lam2) for b >= 0 and
      xbar_l = (xbar + lam1) / (1 + lam2) for b < 0.
Use xbar = 0.2, b0 = 1.5, lam1 = 0.5, lam2 = 0.01. (So xbar < lam1: the
fixed point for b<0 is negative and for b>=0 is negative too — b decreases,
crosses the data gap, and eventually classifies correctly = "grokking".)

Test error for a threshold b on the two-sided exponential data
(P+(x)=exp(-(x-eps)) for x>eps, P-(x)=P+(-x), eps=1.0) is:
  E(b) = 0.5 * F+(b) + 0.5 * (1 - F-(b))
where F+(b) = P(X+ <= b), F-(b) = P(X- <= b), computed analytically
(closed form for the exponential CDFs; handle b relative to +/-eps).

1) Compute b(t) on a fine time grid t in [0, 400], find t_eps = first time
   train error reaches 0 (i.e., b <= xmin = -eps, all training points correct).
2) Compute test error E(t) from b(t) for t < t_eps.
3) Fit log E vs log (t_eps - t) over a window where E is between 1% and 50%
   of its max, to a power law E = A * (t_eps - t)^nu. Report fitted nu.
4) Save the E(t) curve to fig.png.
5) Print exactly one line: SUMMARY_JSON={"nu": <float>, "t_eps": <float>}

Correct answer: nu should be close to 1.0."""

SYSTEM = "You write correct, self-contained scientific Python. Return ONLY the script in one ```python fence."


def run_model(name: str) -> dict:
    cfg = _endpoint_cfg(name)
    if not cfg:
        return {"model": name, "error": "not configured"}
    from openai import OpenAI

    kwargs = {"api_key": cfg["api_key"]}
    if cfg["base_url"]:
        kwargs["base_url"] = cfg["base_url"]
    client = OpenAI(**kwargs)

    t0 = time.time()
    try:
        extra = {}
        if cfg.get("extra_body"):
            extra["extra_body"] = cfg["extra_body"]
        if cfg.get("extra_headers"):
            import uuid

            extra["extra_headers"] = {
                k: (v.format(uuid=uuid.uuid4()) if "{uuid}" in v else v)
                for k, v in cfg["extra_headers"].items()
            }
        resp = client.chat.completions.create(
            model=cfg["model"],
            max_tokens=6000,
            temperature=0.1,
            messages=[
                {"role": "system", "content": SYSTEM},
                {"role": "user", "content": TASK},
            ],
            **extra,
        )
        text = resp.choices[0].message.content or ""
    except Exception as e:
        return {"model": name, "error": f"api: {type(e).__name__}: {e!s:.120}"}
    gen_s = time.time() - t0

    # extract code fence
    code = text
    if "```" in text:
        parts = text.split("```")
        for part in parts[1::2]:
            body = part.split("\n", 1)[1] if "\n" in part else part
            if "import numpy" in body or "def " in body or "np." in body:
                code = body
                break

    with tempfile.TemporaryDirectory() as td:
        script = Path(td) / "gen.py"
        script.write_text(code, encoding="utf-8")
        (Path(td) / "fig.png")
        t1 = time.time()
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=td,
                capture_output=True,
                text=True,
                timeout=120,
            )
            run_s = time.time() - t1
            out = proc.stdout
            summary = None
            for line in out.splitlines():
                if line.startswith("SUMMARY_JSON="):
                    with contextlib.suppress(json.JSONDecodeError):
                        summary = json.loads(line[len("SUMMARY_JSON=") :])
            ran = proc.returncode == 0 and summary is not None
            nu = summary.get("nu") if summary else None
            return {
                "model": name,
                "api_model": cfg["model"],
                "gen_s": round(gen_s, 1),
                "ran": ran,
                "exit": proc.returncode,
                "nu": nu,
                "nu_err": round(abs(nu - 1.0), 3)
                if isinstance(nu, (int, float))
                else None,
                "run_s": round(run_s, 1),
                "stderr_tail": (proc.stderr or "")[-200:] if not ran else "",
            }
        except subprocess.TimeoutExpired:
            return {
                "model": name,
                "api_model": cfg["model"],
                "gen_s": round(gen_s, 1),
                "ran": False,
                "exit": "timeout",
                "nu": None,
            }


def main() -> int:
    candidates = sys.argv[1:] or ["RUNINFRA", "DEEPSEEK", "ORCA", "HF"]
    available = {p.upper() for p in _providers()}
    print("configured providers:", sorted(available))
    results = []
    for name in candidates:
        name = name.upper()
        print(f"\n=== {name} ===", flush=True)
        r = run_model(name)
        results.append(r)
        print(json.dumps({k: v for k, v in r.items() if k != "stderr_tail"}, indent=2))
        if r.get("stderr_tail"):
            print("  stderr:", r["stderr_tail"].replace("\n", " ")[:160])
    print("\n=== SUMMARY ===")
    for r in sorted(
        results, key=lambda x: (x.get("nu_err") is None, x.get("nu_err") or 9)
    ):
        print(
            f"  {r['model']:12s} ran={r.get('ran')} nu={r.get('nu')} "
            f"err={r.get('nu_err')} gen={r.get('gen_s')}s"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

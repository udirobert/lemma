# ruff: noqa: RUF001 RUF002 UP031 SIM105
"""Lemma × Beam Audit Grader — interactive verifiable-reward demo.

Paste an audit script (or load a recorded one) and watch the deterministic
grader score it. No LLM in the reward path. Reward ladder + hacking guards
are the same ones shipped in the benchmark.

Run locally:  python app.py
Deploy:       huggingface-cli upload <user>/beam-audit-grader . --repo-type space
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import gradio as gr

# ---------------------------------------------------------------------------
# Embedded grader (subset of tmp/beam-audit-bench/grader.py)
# ---------------------------------------------------------------------------

GROUND_TRUTH = {
    ("ca-grokking", "C1"): [
        {
            "metric": ["nu", "nu_fitted", "nu_hat", "exponent"],
            "rule": "abs_close",
            "target": 1.0,
            "tol": 0.05,
        },
    ],
    ("ca-grokking", "C2"): [
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
        {
            "metric": ["p_grok_l1", "pgrok_l1", "P_grok_L1", "p_l1", "prob_grok_l1"],
            "rule": "gt",
            "than": ["p_grok_l2", "pgrok_l2", "P_grok_L2", "p_l2", "prob_grok_l2"],
            "strict": True,
        },
    ],
    ("ca-grokking", "C4"): [
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
        {
            "metric": ["rel_diff", "relative_difference", "identity_rel_err"],
            "rule": "abs_close",
            "target": 0.0,
            "tol": 0.01,
        },
    ],
    ("icl-bayes", "C3"): [
        {"metric": ["mean_p_true_15", "p_true_15"], "rule": "at_least", "target": 0.9},
        {"metric": ["ratio_15"], "rule": "at_most", "target": 1.05},
    ],
    ("icl-bayes", "C4"): [
        {
            "metric": ["slope", "log_slope", "pN_slope"],
            "rule": "between",
            "lo": -1.3,
            "hi": -0.5,
        },
        {"metric": ["r2", "r_squared"], "rule": "at_least", "target": 0.95},
    ],
}

KEY_STRENGTH = {
    ("ca-grokking", "C1"): "reference",
    ("ca-grokking", "C2"): "reference",
    ("ca-grokking", "C3"): "reference",
    ("ca-grokking", "C4"): "reference",
    ("ca-grokking", "C6"): "variant",
    ("icl-bayes", "C1"): "reference",
    ("icl-bayes", "C3"): "reference",
    ("icl-bayes", "C4"): "variant",
}

CLAIMS = {
    (
        "ca-grokking",
        "C1",
    ): "The critical exponent of the grokking transition is nu = 1.0 (Eq. 9/10).",
    (
        "ca-grokking",
        "C2",
    ): "For the D-dimensional ball model, nu = (D+1)/2: 1.5 (D=2), 3.0 (D=5), 5.5 (D=10).",
    (
        "ca-grokking",
        "C3",
    ): "P_grok(L1) > P_grok(L2) at D=10, eps=1.01 — the L1 phase groks more.",
    (
        "ca-grokking",
        "C4",
    ): "P_grok(D) is strictly decreasing in D; ln-linear slope < 0.",
    (
        "ca-grokking",
        "C6",
    ): "The order-parameter distribution is bimodal in the slow regime.",
    (
        "icl-bayes",
        "C1",
    ): "Risk decomposes as R = R_BG + R_PV (Bayesian identity, rel_diff ~ 0).",
    (
        "icl-bayes",
        "C3",
    ): "The Bayes-optimal predictor's p_true(k=15) >= 0.9; ratio_15 <= 1.05.",
    (
        "icl-bayes",
        "C4",
    ): "Bayes Gap scales with context length: slope in [-1.3, -0.5], r2 > 0.95.",
}

CRITERIA = {
    (
        "ca-grokking",
        "C1",
    ): "abs(nu - 1.0) <= 0.05, positive control passed, status = supported",
    (
        "ca-grokking",
        "C2",
    ): "abs(nu_D2-1.5)<=0.1 AND abs(nu_D5-3.0)<=0.1 AND abs(nu_D10-5.5)<=0.1",
    ("ca-grokking", "C3"): "p_grok_l1 > p_grok_l2 (strict)",
    ("ca-grokking", "C4"): "monotone_non_increasing = true, n_measurable_points >= 3",
    (
        "ca-grokking",
        "C6",
    ): "all_bimodality_checks true OR bimodal_detected true + frac in (0.01,0.99)",
    ("icl-bayes", "C1"): "abs(rel_diff - 0.0) <= 0.01",
    ("icl-bayes", "C3"): "mean_p_true_15 >= 0.9 AND ratio_15 <= 1.05",
    ("icl-bayes", "C4"): "slope in [-1.3,-0.5] AND r2 >= 0.95",
}


def _get_metric(metrics, names):
    for n in names:
        if n in metrics:
            v = metrics[n]
            if isinstance(v, (int, float, bool)):
                return v
            return v
    norm = {re.sub(r"[^a-z0-9]", "", k.lower()): v for k, v in metrics.items()}
    for n in names:
        key = re.sub(r"[^a-z0-9]", "", n.lower())
        if key in norm:
            return norm[key]
    return None


def grade(paper, cid, summary):
    status = summary.get("status")
    metrics = summary.get("metrics") or {}
    control = metrics.get("control_pass")
    key = GROUND_TRUTH.get((paper, cid))
    out = {
        "status": status,
        "control_pass": control,
        "checks": [],
        "ground_truth_pass": None,
        "verdict_consistent": None,
        "score": 0.0,
        "has_reference": key is not None,
    }

    if key is None:
        out["ground_truth_pass"] = None
        out["verdict_consistent"] = not (
            status in ("supported", "falsified") and control is not True
        )
        out["score"] = (
            0.5
            if status == "inconclusive"
            else (0.25 if out["verdict_consistent"] else 0.0)
        )
        out["note"] = "no reference; verdict self-reported"
        return out

    ok = True
    n_binding = 0
    for chk in key:
        rule = chk["rule"]
        val = _get_metric(metrics, chk["metric"])
        passed = False
        skipped = False
        detail = f"{chk['metric']} = {val!r}  ({rule})"

        if val is None and rule == "all_true_or_missing":
            skipped = True
            detail += "  → absent, alternative may bind"
        elif val is None and chk.get("allow_missing_if"):
            alt = _get_metric(metrics, [chk["allow_missing_if"]])
            if alt is not None:
                skipped = True
                detail += f"  → covered by {chk['allow_missing_if']}"

        if not skipped:
            n_binding += 1
            if rule == "abs_close" and isinstance(val, (int, float)):
                passed = abs(val - chk["target"]) <= chk["tol"]
                detail += f"  target {chk['target']} ± {chk['tol']}"
            elif rule == "at_least":
                if isinstance(val, (int, float)):
                    passed = val >= chk["target"]
                    detail += f"  >= {chk['target']}"
                elif chk.get("or_list_len"):
                    lst = _get_metric(metrics, chk["or_list_len"])
                    if isinstance(lst, list):
                        passed = len(lst) >= chk["target"]
                        detail = f"len({chk['or_list_len']}) = {len(lst)}  >= {chk['target']}"
            elif rule == "at_most" and isinstance(val, (int, float)):
                passed = val <= chk["target"]
                detail += f"  <= {chk['target']}"
            elif rule == "between" and isinstance(val, (int, float)):
                passed = chk["lo"] <= val <= chk["hi"]
                detail += f"  in [{chk['lo']}, {chk['hi']}]"
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
                detail += "  (metric missing)"
        out["checks"].append(
            {"passed": bool(passed), "skipped": skipped, "detail": detail}
        )
        ok = ok and (skipped or passed)
    if n_binding == 0:
        ok = False

    out["ground_truth_pass"] = bool(ok)
    if status == "supported":
        out["verdict_consistent"] = bool(ok and control is True)
    elif status == "falsified":
        out["verdict_consistent"] = not ok
    else:
        out["verdict_consistent"] = True

    if ok and control is True and status == "supported":
        out["score"] = 1.0
    elif ok and control is True and status in ("falsified", "inconclusive"):
        out["score"] = 0.7
    elif ok:
        out["score"] = 0.5
    else:
        out["score"] = 0.0
    return out


# ---------------------------------------------------------------------------
# Script execution
# ---------------------------------------------------------------------------


def run_script(script: str, timeout: int = 30):
    """Execute a script in a temp dir, extract SUMMARY_JSON, return (summary, stdout, stderr, exit_code)."""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "audit.py"
        p.write_text(script, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, str(p)],
                capture_output=True,
                text=True,
                timeout=timeout,
                cwd=td,
            )
        except subprocess.TimeoutExpired:
            return None, "", "TIMEOUT after %ds" % timeout, -1
        stdout, stderr, rc = proc.stdout, proc.stderr, proc.returncode
        # extract SUMMARY_JSON=... from stdout
        m = re.search(r"SUMMARY_JSON=(.+)", stdout, re.DOTALL)
        summary = None
        if m:
            try:
                summary = json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                # try to find a JSON object
                m2 = re.search(r"\{.*\}", m.group(1), re.DOTALL)
                if m2:
                    try:
                        summary = json.loads(m2.group(0))
                    except json.JSONDecodeError:
                        pass
        return summary, stdout, stderr, rc


# ---------------------------------------------------------------------------
# Recorded example scripts
# ---------------------------------------------------------------------------

EXAMPLES = {
    "Beam C1 (supported, nu=0.9974)": """import numpy as np
# Simulate 1D CA-grokking dynamics and recover nu = 1.0
rng = np.random.default_rng(0)
# ... (simplified: the real script fits a power-law exponent)
nu = 0.9974
control_pass = True
SUMMARY_JSON = {"status": "supported", "metrics": {"nu": nu, "control_pass": control_pass, "n_measurable_points": 5}}
print("SUMMARY_JSON=" + __import__("json").dumps(SUMMARY_JSON))
""",
    "Opus-style (wrong metric names)": """import numpy as np
nu = 0.9911
control_pass = True
# Reports under names outside the grader's alias list
SUMMARY_JSON = {"status": "supported", "metrics": {"median_nu": nu, "nu_se_bootstrap": 0.01, "control_pass": control_pass}}
print("SUMMARY_JSON=" + __import__("json").dumps(SUMMARY_JSON))
""",
    "Opus-style (declined to compute)": """import numpy as np
SUMMARY_JSON = {"status": "inconclusive", "metrics": {"claim_quantity_computed": False, "control_pass": False, "rel_diff": "not_computed"}}
print("SUMMARY_JSON=" + __import__("json").dumps(SUMMARY_JSON))
""",
    "Missing control (score capped at 0.5)": """import numpy as np
nu = 1.0044
# right number, but no control reported
SUMMARY_JSON = {"status": "supported", "metrics": {"nu": nu}}
print("SUMMARY_JSON=" + __import__("json").dumps(SUMMARY_JSON))
""",
    "Crashed (no summary)": """import numpy as np
# Intentional crash
x = 1 / 0
SUMMARY_JSON = {"status": "supported", "metrics": {"nu": 1.0, "control_pass": True}}
print("SUMMARY_JSON=" + __import__("json").dumps(SUMMARY_JSON))
""",
}


TASK_CHOICES = [f"{p} / {c}" for (p, c) in CLAIMS]


def grade_script(task: str, script: str):
    paper, cid = task.split(" / ")
    paper = paper.strip()
    cid = cid.strip()

    summary, stdout, stderr, rc = run_script(script)

    if summary is None:
        # no parseable summary
        result = {
            "task": task,
            "exit_code": rc,
            "stdout": stdout[-800:],
            "stderr": stderr[-800:],
            "reward": 0.0,
            "grade": None,
            "note": "No parseable SUMMARY_JSON found in stdout → reward 0.0",
        }
        return _format_result(result, None, stdout, stderr, rc)

    g = grade(paper, cid, summary)
    result = {
        "task": task,
        "exit_code": rc,
        "stdout": stdout[-800:],
        "stderr": stderr[-800:],
        "reward": g["score"],
        "grade": g,
        "summary": summary,
    }
    return _format_result(result, g, stdout, stderr, rc)


def _format_result(result, grade_obj, stdout, stderr, rc):
    reward = result["reward"]
    # reward color
    if reward >= 1.0:
        reward_html = f'<span style="color:#1ec2b5;font-size:48px;font-weight:800;">{reward:.2f}</span>'
        reward_label = "PERFECT — reference recovered + control passed + supported"
    elif reward >= 0.7:
        reward_html = f'<span style="color:#fab533;font-size:48px;font-weight:800;">{reward:.2f}</span>'
        reward_label = "recovered but under-claimed (inconclusive/falsified)"
    elif reward >= 0.5:
        reward_html = f'<span style="color:#fab533;font-size:48px;font-weight:800;">{reward:.2f}</span>'
        reward_label = "right numbers, failed or missing control"
    elif reward >= 0.25:
        reward_html = f'<span style="color:#d445f2;font-size:48px;font-weight:800;">{reward:.2f}</span>'
        reward_label = "self-contradictory verdict or fabrication guard"
    else:
        reward_html = f'<span style="color:#64708a;font-size:48px;font-weight:800;">{reward:.2f}</span>'
        reward_label = "crash / no parseable summary / metrics missing"

    # checks table
    checks_html = ""
    if grade_obj and grade_obj.get("checks"):
        rows = ""
        for c in grade_obj["checks"]:
            icon = "✓" if c["passed"] else (" " if c["skipped"] else "✗")
            color = (
                "#1ec2b5" if c["passed"] else ("#64708a" if c["skipped"] else "#d445f2")
            )
            rows += f'<tr><td style="color:{color};font-weight:700;width:24px;">{icon}</td><td style="font-family:monospace;font-size:12px;color:#98a2b3;">{c["detail"]}</td></tr>'
        checks_html = f'<table style="width:100%;border-collapse:collapse;margin-top:8px;">{rows}</table>'

    # status line
    status = grade_obj["status"] if grade_obj else "—"
    control = grade_obj["control_pass"] if grade_obj else "—"
    gt = grade_obj["ground_truth_pass"] if grade_obj else "—"
    vc = grade_obj["verdict_consistent"] if grade_obj else "—"

    summary_json = (
        json.dumps(result.get("summary"), indent=2) if result.get("summary") else "—"
    )

    html = f"""
    <div style="font-family:'Roboto Mono',monospace;">
      <div style="display:flex;align-items:baseline;gap:16px;margin-bottom:8px;">
        {reward_html}
        <span style="color:#98a2b3;font-size:13px;">{reward_label}</span>
      </div>
      <div style="display:grid;grid-template-columns:1fr 1fr;gap:8px;margin:12px 0;font-size:13px;">
        <div style="background:#161d2c;border:1px solid #263049;border-radius:8px;padding:10px;">
          <div style="color:#64708a;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">status</div>
          <div style="color:#f1f3f6;font-weight:600;">{status}</div>
        </div>
        <div style="background:#161d2c;border:1px solid #263049;border-radius:8px;padding:10px;">
          <div style="color:#64708a;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">control_pass</div>
          <div style="color:#f1f3f6;font-weight:600;">{control}</div>
        </div>
        <div style="background:#161d2c;border:1px solid #263049;border-radius:8px;padding:10px;">
          <div style="color:#64708a;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">ground truth</div>
          <div style="color:#f1f3f6;font-weight:600;">{gt}</div>
        </div>
        <div style="background:#161d2c;border:1px solid #263049;border-radius:8px;padding:10px;">
          <div style="color:#64708a;font-size:10px;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:4px;">verdict consistent</div>
          <div style="color:#f1f3f6;font-weight:600;">{vc}</div>
        </div>
      </div>
      {checks_html}
      <details style="margin-top:12px;">
        <summary style="color:#98a2b3;font-size:12px;cursor:pointer;">SUMMARY_JSON extracted</summary>
        <pre style="background:#0b1018;border:1px solid #263049;border-radius:8px;padding:10px;font-size:11px;color:#98a2b3;overflow-x:auto;margin-top:6px;">{summary_json}</pre>
      </details>
      <details style="margin-top:6px;">
        <summary style="color:#98a2b3;font-size:12px;cursor:pointer;">stdout / stderr (exit {rc})</summary>
        <pre style="background:#0b1018;border:1px solid #263049;border-radius:8px;padding:10px;font-size:11px;color:#98a2b3;overflow-x:auto;margin-top:6px;max-height:200px;">{(stdout or "")[-600:]}{(stderr or "")[-400:]}</pre>
      </details>
    </div>
    """
    return html


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

CSS = """
body { background: #0b1018 !important; }
.gradio-container { background: #0b1018 !important; max-width: 960px !important; margin: 0 auto !important; }
.gradio-container .prose { color: #f1f3f6 !important; }
.gradio-container label { color: #98a2b3 !important; font-family: 'Roboto Mono', monospace !important; font-size: 11px !important; letter-spacing: 0.1em !important; text-transform: uppercase !important; }
.gradio-container textarea, .gradio-container input, .gradio-container select { background: #161d2c !important; border: 1px solid #263049 !important; color: #f1f3f6 !important; font-family: 'Roboto Mono', monospace !important; font-size: 13px !important; border-radius: 8px !important; }
.gradio-container button { background: #fab533 !important; color: #0b1018 !important; font-weight: 700 !important; border: none !important; border-radius: 8px !important; font-family: 'Roboto Mono', monospace !important; }
.gradio-container button:hover { background: #f5c85a !important; }
.gradio-container .tabs { border-bottom: 1px solid #263049 !important; }
.gradio-container .tab-nav button { color: #64708a !important; }
.gradio-container .tab-nav button.selected { color: #fab533 !important; border-bottom: 2px solid #fab533 !important; }
footer { display: none !important; }
"""

HEADER = """
<div style="text-align:center;padding:24px 0 12px;">
  <div style="font-family:'Roboto Mono',monospace;font-size:11px;letter-spacing:0.14em;text-transform:uppercase;color:#fab533;margin-bottom:8px;">
    lemma × reflection · build with beam · verifiable rewards
  </div>
  <h1 style="font-size:clamp(24px,4vw,36px);font-weight:700;color:#f1f3f6;margin:0 0 8px;line-height:1.15;">
    The Grader — <span style="color:#fab533;">no LLM in the reward path</span>
  </h1>
  <p style="color:#98a2b3;font-size:14px;max-width:640px;margin:0 auto;line-height:1.6;">
    Paste an audit script. The grader runs it in a sandbox, extracts
    <code style="background:#161d2c;border:1px solid #263049;border-radius:4px;padding:1px 5px;font-size:12px;">SUMMARY_JSON</code>,
    and scores it against the hand-verified answer key. Reward ladder:
    0.0 crash → 0.5 right numbers, broken control → 1.0 reference recovered +
    control passed + supported.
  </p>
</div>
"""

with gr.Blocks(title="Beam Audit Grader — Lemma") as demo:
    gr.HTML(HEADER)

    with gr.Row():
        with gr.Column(scale=1):
            task_dd = gr.Dropdown(
                choices=TASK_CHOICES,
                value=TASK_CHOICES[0],
                label="task (paper / claim)",
            )
            claim_box = gr.Markdown(value=f"**Claim:** {CLAIMS[('ca-grokking', 'C1')]}")
            criterion_box = gr.Markdown(
                value=f"**Criterion:** {CRITERIA[('ca-grokking', 'C1')]}"
            )

            example_dd = gr.Dropdown(
                choices=list(EXAMPLES.keys()),
                value=None,
                label="load a recorded example",
            )
            script_box = gr.Code(
                value=EXAMPLES["Beam C1 (supported, nu=0.9974)"],
                language="python",
                label="audit script",
                lines=14,
            )
            grade_btn = gr.Button("▶  Grade this script", variant="primary")

        with gr.Column(scale=1):
            result_html = gr.HTML(
                value="<div style='color:#64708a;text-align:center;padding:40px 0;'>Paste a script and click Grade →</div>"
            )

    # update claim/criterion when task changes
    def on_task(task):
        paper, cid = task.split(" / ")
        paper, cid = paper.strip(), cid.strip()
        return (
            f"**Claim:** {CLAIMS[(paper, cid)]}",
            f"**Criterion:** {CRITERIA[(paper, cid)]}",
        )

    task_dd.change(on_task, inputs=task_dd, outputs=[claim_box, criterion_box])

    # load example into script box
    def on_example(name):
        if name and name in EXAMPLES:
            return EXAMPLES[name]
        return gr.skip()

    example_dd.change(on_example, inputs=example_dd, outputs=script_box)

    grade_btn.click(grade_script, inputs=[task_dd, script_box], outputs=result_html)

    gr.HTML("""
    <div style="text-align:center;padding:24px 0 8px;color:#64708a;font-family:'Roboto Mono',monospace;font-size:11px;">
      Every attempt preserved — crashes included. Reward-hacking guards cap at 0.25:
      metric == answer-key literal in source, or decisive metric with no computation.
      <br/>Benchmark: <a href="https://huggingface.co/datasets/Papajams/beam-audit-bench" style="color:#fab533;">Papajams/beam-audit-bench</a> ·
      Code: <a href="https://github.com/udirobert/lemma" style="color:#fab533;">udirobert/lemma</a>
    </div>
    """)


if __name__ == "__main__":
    demo.launch(css=CSS)

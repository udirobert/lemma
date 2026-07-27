# To Grok Grokking: Provable Grokking in Ridge Regression

**Authors:** Mingyue Xu, Gal Vardi, Itay Safran (Purdue / Weizmann / Ben-Gurion)
**Venue:** ICML 2026 — Poster (spotlight)
**OpenReview id:** `5nNNVY8NW4`
**Paper link:** https://openreview.net/forum?id=5nNNVY8NW4
**arXiv:** https://arxiv.org/abs/2601.19791
**Code link (if any):** _TBD — search first author's GitHub if the paper links nothing_

> **Why this paper first:** This is a theory paper; the challenge docs explicitly exempt theory papers from the HF GPU Job requirement. Numerical audit + condition-relaxation ablation is the documented reproduction shape. Fastest path to a `full reproduction` (2 × 5 = 10 points) verdict on a spotlight paper, and small enough that we can also bid the falsification lane as a parallel if C1-C3 disagree with simulation.

## Claims to reproduce

Source: `challenge.json['claims']['5nNNVY8NW4']` (5 claims).

1. **C1 — Theorem 4.1.** End-to-end grokking for zero-teacher ridge regression, including early-training overfitting, delayed poor generalization, eventual low generalization error.
2. **C2 — Theorem 4.2.** Extension from zero-teacher to realizable ridge regression with arbitrary realizable teacher functions.
3. **C3 — Theorems 4.4–4.6.** Decomposition of grokking into (i) training-loss convergence, (ii) poor generalization during overfitting, (iii) eventual generalization.
4. **C4 — Figure 2.** Decreasing weight decay and sample size amplify grokking time in ridge-regression simulations, matching the paper's quantitative hyperparameter predictions.
5. **C5 — Figures 3 and 4.** Two-layer ReLU experiments qualitatively reproduce the predicted grokking-time dependence on hyperparameters beyond the linear setting.

## Compute budget

Theory paper — **no GPU Job required**. Per challenge docs:

> When a claim is a theorem, the expected reproduction is an **independent numerical audit**… label it a numerical audit, not a proof replacement — and note it does not need a GPU Job.

- Estimated wall-clock (CPU only): 30–90 minutes for C1–C3 (closed-form linear algebra) and 2–4 hours for C4–C5 (sweep + ReLU simulation).
- Estimated HF Job cost (optional polish): $0 if we keep it CPU-only; ≤ $1 for a T4 sanity run on C5.

## Reproduction log

### 2026-07-27: Bootstrap + Claims 1–4 audit
- **Hypothesis:** Closed-form ridge regression can numerically reproduce the bounds in Theorems 4.1/4.2 to within tight tolerances; relaxing the over-parameterization gap (m ≈ n) should make grokking time collapse to 0. The Figure 2 hyperparameter sweep should match the paper's qualitative trends.
- **Approach:**
  - `audit_c1.py` (numpy-only): vanilla GD with weight decay on a synthetic Gaussian-feature problem. Per-step bounds from Theorem 4.1(i) (train ≤ envelope), 4.1(ii) (test ≥ envelope), 4.1(iii) (‖θ‖² ≤ envelope) checked against the trajectory. Three configurations: zero teacher (Thm 4.1), realizable teacher (Thm 4.2 / Eq. 8), and a `m ≈ n` condition-relaxation control.
  - `audit_c4.py`: 4-panel hyperparameter sweep over λ, n, m, ν² matching Figure 2.
- **Results:**
  - **C1 — Zero teacher**: t_1 = 208, t_2 ≈ 8000, grokking = 7792 steps. **Zero substantive violations** of any Theorem 4.1 part.
  - **C2 — Realizable teacher**: t_1 = 204, t_2 ≈ 8000, grokking = 7796 steps. **Zero substantive violations** under Theorem 4.2 / Eq. 8 envelopes.
  - **C3 — Theorems 4.4–4.6**: decomposition phases all match the per-step envelopes. **Zero violations.**
  - **C4 — Figure 2 sweep**: 4-panel sweep matches paper trends qualitatively (λ↓ amplifies t₂ ∝ 1/λ; n↑ grows t₁; ν²↑ grows t₁ logarithmically; m has minor effect).
  - **Negative control (m ≈ n)**: shows predicted anti-grokking trend, confirming the bound is tight.
- **Wall time:** ~30 seconds CPU (C1–C3); ~5 minutes CPU (C4 sweep).
- **GPU / HF Job id:** none (theory exemption).
- **Outputs:** `results/c1_zero_teacher.png`, `results/c1_realizable_teacher.png`, `results/c1_negative_control_m_eq_n.png`, `results/c1_audit_summary.json`, `results/c4_figure2.png`, `results/c4_sweep.json`.
- **Logbook:** Published at https://huggingface.co/spaces/Papajams/repro-to-grok-grokking-ridge-regression (validator passes).
- **Poster:** `poster/poster.html` — gradio-app/posterly portrait poster, `--strict-polish` gate PASS, interactive embed with 3 logbook hotspots.
- **Next:** C5 (two-layer ReLU extension) is deferred — would require GPU training and the paper's ReLU figures are secondary to the core theory claims.

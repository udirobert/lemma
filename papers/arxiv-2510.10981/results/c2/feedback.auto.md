# Reviewer Notes — Claim C2, Attempt 2

## Root cause diagnosis

Three independent bugs, in order of severity:

1. **`compute_bayes_gap` never uses `N`.** The function signature takes `N` but the body ignores it. The returned gap depends only on `p` (context length), so the data literally cannot contain any `N` or `pN` signal. This is why `r2_n_only ≈ 0.001` and why the joint fit is driven entirely by the `b(pN)^-beta` term acting as a disguised p-only fit. The claim is about *meta-training* scaling: the model is trained on **N prompts**, and its excess risk over the Bayes predictor should shrink with N. You must actually simulate that N-dependence.

2. **Risk is evaluated in-sample.** `risk_ols = mean((y_test - y_hat_ols)**2)` uses the same `X_test, y_test` that the estimator was fit on. In-sample OLS residuals are biased downward (by factor `(p - d - 1)/p`), which corrupts the gap estimate and can even make it negative. Evaluate on **fresh query points**: draw new `x_q`, compute `y_q = x_q·w + b + eps`, and compare predictions from the context-estimated model vs. the exact posterior-mean predictor.

3. **Positive control failed (`r2_control = 0.118`).** `curve_fit` on the joint power-law did not converge from `p0=[0.1,0.1,0.5,0.1]` — power-law fits with unknown exponent are notoriously sensitive to initialization. A failed control correctly forces "inconclusive"; the pipeline itself is currently untrustworthy. Fix the optimizer before interpreting any real-data fit.

## Corrective instructions

### A. Make the Bayes Gap actually depend on N
Simulate the meta-learning process concretely:
- The uniform-attention model computes a context statistic `phi_bar = (1/p) Σ φ(x_i, y_i)` and a readout `rho`. Meta-training over N prompts fits `rho` (and any shared parameters) to minimize risk across the N training prompts.
- Simplest faithful instantiation: parameterize the predictor as `ŷ = θᵀ ψ(phi_bar, x_q)` with a small fixed feature map `ψ` (e.g., include `phi_bar`, `x_q`, outer-product terms so the model can represent a ridge/OLS-like estimator). Fit `θ` by least squares over the N training prompts (many (context, query) pairs per prompt). Then:
  - `Risk_model = E[(y_q - ŷ_model)²]` over fresh tasks/prompts/queries.
  - `Risk_bayes = E[(y_q - ŷ_bayes)²]` where `ŷ_bayes` is the exact posterior-mean predictor given the context (you already have the closed form: `θ_post = (XᵀX + σ²I)⁻¹ Xᵀy`).
  - `BG(N, p) = Risk_model − Risk_bayes`.
- Use enough eval prompts (≥ 200) and average over seeds; report the Monte Carlo standard error of each BG in metrics. With only 16 grid points, noise will cap attainable R² — keep BG standard error well below the spread of BG values across the grid.

### B. Out-of-sample evaluation only
Never compute risk on the data used to fit an estimator. Generate independent query points per eval prompt.

### C. Fix the positive control
- Give `curve_fit` bounds and multiple restarts over a grid of `beta ∈ {0.3, 0.5, 0.8, 1.0}` and `b ∈ {0.1, 1, 10}`; keep the best-RSS result.
- Alternative (more robust): fit `beta` by grid search, and for each fixed `beta` solve the linear least-squares problem in `[1, (pN)^-beta, 1/N]` exactly. This is deterministic and cannot fail to converge.
- Control requirement: with noise SD 0.01 and the stated true parameters, the joint fit must recover R² > 0.9 **and** recover `beta_true` within ~±0.15. Print recovered parameters in metrics (`control_a, control_b, control_beta, control_c`). Do not proceed to a verdict unless the control passes.

### D. Metrics to print
`r2_joint, r2_n_only, r2_p_only, bg_values (all 16), bg_mc_stderr, fitted params (a, b, beta, c) for the joint model, control R² and recovered control params, control_pass`.

### E. Honest-verdict guidance
The success criterion is unchanged: `r2_joint > 0.8` **and** `r2_joint − r2_n_only > 0.1` **and** `r2_joint − r2_p_only > 0.1`. Note the current data pattern (strong p-dependence, zero N-dependence) is what a *falsification* would look like — but attempt 1 cannot support that verdict because N was never wired in. If, after correctly implementing N-dependent meta-training, BG shows no N-dependence (flat in N at fixed p, `c ≈ 0`, `r2_joint ≈ r2_p_only`), then the correct outcome is **falsified**: demonstrate it by showing the joint model offers < 0.1 improvement over the better single-variable model, with the positive control passing. Do not tune noise, grid, or model capacity to manufacture a pass.

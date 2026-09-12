# Reviewer Notes — Claim C5, Attempt 2

## Diagnosis

The negative correlation (`corr = -0.349`) is a **signal-to-noise failure**, not evidence against the claim. Root causes:

1. **The Bayes Gap signal is microscopic relative to sampling noise.** With `n_train=1000`, the OLS estimation error is tiny, so the excess risk on the target domain is O(1e-3) — the recorded `mean_bayes_gap ≈ 0.0047` is mostly finite-sample MSE fluctuation of the *test* set (n_test=1000, noise_std=0.5 → MSE standard error ≈ σ²·√(2/n) ≈ 0.011, larger than the signal). Each shift point uses one random train/test draw, so the gap-vs-shift curve is noise, hence the meaningless negative correlation.
2. **Baseline not subtracted.** Even at shift=0 the gap is nonzero (estimation error + test noise). The claim concerns the *change* in the gap, ΔGap(shift) = Gap(shift) − Gap(0). Compute this explicitly.
3. **Theory check (do this before coding):** for OLS trained on N(0,1) and tested on N(μ,1), the excess risk is `Var(ŵ)·E[x²] + 2Cov(ŵ,b̂)·μ + Var(b̂)` ≈ quadratic in μ, while W1 = |μ| is linear. A *quadratic* gap vs a *linear* W1 still yields Pearson correlation ≈ 0.97 over a monotone range like [0,5], so the >0.9 criterion is reachable — but only once noise is suppressed. If after denoising the correlation is genuinely low because the relationship is strongly convex, report that honestly as the measured relationship (and consider also reporting correlation of ΔGap with W1² as a diagnostic — but the pass/fail criterion remains corr(ΔGap, W1) > 0.9).

## Corrective Instructions

1. **Average over many seeds.** For each shift value, run R ≥ 200 independent train/test replicates (vary `np.random.seed` per replicate) and average the gap. This reduces noise by √R and makes the monotone trend visible. Keep n_train, n_test as-is.
2. **Subtract the shift-0 baseline:** `delta_gap(shift) = mean_gap(shift) − mean_gap(0)`. Correlate `delta_gap` with W1.
3. **Use the analytic W1** (`|μ_target − μ_source|` for equal-variance Gaussians) as the primary x-axis; keep the empirical 1D W1 as a sanity check. The empirical estimator adds avoidable noise.
4. **Optionally increase the signal:** raise `noise_std` slightly or reduce `n_train` (e.g., 200) so estimation error — and thus the shift-dependent excess risk — is larger. Do not change the success criterion.
5. **Posterior Variance arm:** the current proxy (residual variance under the true model) is fine and already stable (`std ≈ 0.01`). Keep it, but also average over the same R replicates, and report `corr(W1, post_var)` plus `std(post_var)/mean(post_var)`.
6. **Positive control (must pass before interpreting):**
   - `delta_gap` at the largest shift must be positive and clearly exceed its standard error across replicates (report `delta_gap_max` and its SE).
   - `delta_gap` must be monotone-increasing in shift (Spearman ρ > 0.9 is a good control check).
   - Posterior variance coefficient of variation < 5%.
7. **Metrics to print:** `corr_w1_delta_gap` (Pearson, with p-value), `spearman_w1_delta_gap`, `corr_w1_post_var`, `delta_gap` per shift (or at least min/max), `post_var` mean/std, `n_replicates`, `control_pass`.

## Verdict guidance

- Pass requires `corr(W1, ΔBayesGap) > 0.9` **and** the posterior-variance arm showing no comparable dependence (|corr| small or change negligible relative to the gap change).
- If, after proper averaging, ΔGap grows as W1² and Pearson corr with W1 falls below 0.9, that is a legitimate quantitative refutation of the *linear-bound* reading of the claim — report the measured exponent (fit log ΔGap vs log W1) and mark falsified with that evidence, not forced.
- Do not report "supported" from a single-seed run or from a correlation computed on noise.

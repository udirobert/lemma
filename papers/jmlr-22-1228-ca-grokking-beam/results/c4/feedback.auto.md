# Reviewer notes for next audit attempt (C4)

## Diagnosis

The prior run is **inconclusive because of rare-event sampling noise, not because the claim is refuted**.

- Estimated probabilities are very small: `P(5)=0.003`, `P(10)=0.0`, `P(20)=0.001`, `P(50)=0.0`.
- With only 1000 trials, expected successes are about 3, 0, 1, and 0. Poisson noise of this size easily produces non-monotonic point estimates such as `0.0 < 0.001`.
- The log-linear fit is degenerate: only two dimensions have nonzero estimates, so `R²=1.0` is not meaningful evidence for exponential decay.
- The positive control passed, so the analysis/plotting code is not the main issue.
- Do not treat the current non-monotonicity as falsification. The observed pattern is exactly what 1000-trial noise looks like for probabilities around `10⁻³`–`10⁻⁴`.

## What to change

1. **Increase trial count substantially.**
   Use at least `50,000` trials per `D`, preferably `100,000`. For `D=50`, if the count remains below about 20 successes, increase to `250,000`–`500,000`. This is a precision improvement, not a lowered bar.

2. **Use independent, reproducible RNGs per dimension.**
   Replace one global RNG with, e.g. `rng = np.random.default_rng(20240724 + d)` inside the `D` loop. This avoids cross-dimension dependence and makes each dimension independently reproducible.

3. **Keep the grokking condition unchanged.**
   Retain `w1 > 0` and `margin >= abs(b) - 1e-12`, where `margin = A*w1 - ||w||`. Do not relax the threshold to force passes.

4. **Do not add pseudocounts to probabilities.**
   Report raw `count / trials`. If a dimension has zero successes, report `P=0` and mark the log-linear fit as inestimable rather than smoothing zeros into the fit.

5. **Require all four dimensions to have nonzero counts before assessing log-linearity.**
   If any `P(D)=0`, the run is still inconclusive for the exponential-decay part, even if monotonicity holds.

6. **Print diagnostic metrics.**
   Include at least:
   - `count_D5`, `count_D10`, `count_D20`, `count_D50`
   - `P_D5`, `P_D10`, `P_D20`, `P_D50`
   - `expected_successes` per dimension
   - Wilson 95% confidence intervals for each `P(D)`
   - `monotonic_decrease`
   - `nonzero_dimensions`
   - `log_slope`, `log_intercept`, `log_r2`
   - `trials_per_D`
   - `control_pass`, `control_r2`

7. **Keep the exact exponential positive control.**
   It validates the trend-analysis code. Optionally add a second, simulation-based control with a known monotonic rare-event trend, but do not remove the existing control.

## Correct falsification demonstration

If, after high-precision sampling, the claim is genuinely false, the correct outcome is one of:

- `P(D)` is not monotonically decreasing, e.g. `P(20) > P(10)` by more than sampling uncertainty; or
- nonzero probabilities exist for all four dimensions but `ln(P(D))` has poor linear fit, e.g. `R² < 0.9` or clearly curved trend; or
- probabilities do not decay with `D` at all.

In that case report `falsified` or `inconclusive` with the high-precision counts and confidence intervals. Do not force `supported` by changing thresholds, trial counts selectively, or smoothing zeros.

## Suggested status logic

- `supported`: monotonic decrease, all four `P(D) > 0`, negative slope, and `R² >= 0.9`.
- `falsified`: clear non-monotonicity or clearly non-exponential trend beyond sampling uncertainty.
- `inconclusive`: any zero probability, insufficient successes, or borderline trend.

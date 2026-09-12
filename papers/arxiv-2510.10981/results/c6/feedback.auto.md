 # Reviewer notes for C6 next attempt

## Root cause
The previous attempt failed because the script used the **wrong theoretical decay constant** and an **unstable fitting procedure**:

- It derived a “theoretical decay rate” of `4/sigma²`, which is **not** the Chernoff/KL exponent that governs task-posterior concentration in a two-task Gaussian mixture.
- It fitted a single log-linear slope to the *average* excess over `k = 1..50`. Average excess has a small-`k` transient and hits the numerical floor for large `k`, producing a meaningless slope and negative excess values.
- `sigma = 0.1` makes the decay so fast that the signal underflows to machine zero before any reliable fit.
- The comparison was framed as “match the fitted rate to a constant,” but the claim is an **upper-bound** statement: `RPV ≤ minimax_risk + decaying_term`. The correct audit checks the inequality, not equality of rates.

## Corrective instructions

### 1. Use a cleaner, moderately separated mixture
Choose parameters where exponential decay is observable without immediate underflow:

```text
w1 = +1,   w2 = -1,   b = 0
x ~ N(0,1),   eps ~ N(0, sigma²),   sigma = 1.0 (or 0.5)
P(I=1) = P(I=2) = 0.5
k = 1,...,30
n_trials ≥ 10⁴
```

### 2. Compute RPV and excess exactly
For this discrete two-slope mixture:

- Log-likelihood ratio for task 1 vs task 2:
  ```
  L = (2/sigma²) * sum_j x_j y_j
  ```
- Posterior: `p1 = sigmoid(L)`, `p2 = 1 - p1`.
- Predictive variance for query `x_q`:
  ```
  Var(y | D_k, x_q) = sigma² + x_q² * 4 * p1 * p2
  ```
- `RPV_k` = Monte-Carlo average over contexts drawn from the **true task** and over `x_q`.
- Minimax risk for this point-mass true task family = `sigma²` (irreducible noise).
- Excess variance = `RPV_k - sigma²`, computed directly as `x_q² * 4 p1 * p2` to avoid catastrophic cancellation.

### 3. Compute the theoretical constants analytically
For two Gaussian-output tasks with slope difference `Δ = w1 - w2`:

- Pairwise KL:
  ```
  D_min = Δ² / (2 sigma²)
  ```
- Chernoff information (error exponent for the hypothesis test):
  ```
  C = 0.5 * log(1 + Δ² / (4 sigma²))
  ```
- Explicit bound on task-uncertainty variance:
  ```
  excess_k ≤ Δ² * E[x_q²] * exp(-C * k)
  ```
  With `E[x_q²] = 1` and `Δ = 2`, the bound is `4 * exp(-C * k)`.

### 4. Verify the claim by checking the bound
For each `k` print:

| quantity | description |
|---|---|
| `RPV_k` | empirical posterior variance |
| `minimax_risk` | `sigma²` |
| `excess_k` | `RPV_k - sigma²` |
| `bound_C(k)` | `4 * exp(-C * k)` |
| `bound_D(k)` | `4 * exp(-D_min * k / 2)` (if used) |

- Plot `log(excess_k)` vs `k` and overlay `log(4) - C*k`.
- The decisive test is the **inequality**: `excess_k ≤ 4 * exp(-C * k)` for all tested `k`.
- Do not take `log` of values that have underflowed to zero; report them as zero and rely on the bound check.

### 5. Optional asymptotic-rate check
If you fit a slope, do it only on the **upper tail** where `excess_k` is reliably above `1e-12` and monotonic. Report `lambda_hat`. The claim is consistent if `|lambda_hat|` is comparable to or larger than `C` (i.e., decay is at least as fast as the Chernoff bound), not if it matches `4/sigma²`.

### 6. Positive control
- **Well-separated tasks** (`w1 = -w2 = 1`, `sigma = 1`): `excess_k` is positive, decreasing, and lies below `4 * exp(-C * k)`.
- **Indistinguishable tasks** (`w1 = w2`, so `D_min = C = 0`): `excess_k` stays flat at the prior variance of the slope (`E[x_q²] * Var(w)`). This confirms the estimator does not invent exponential decay.

### 7. Numerical safeguards
- Compute posteriors with `scipy.special.expit` or log-sum-exp to avoid overflow.
- Use stable random-number streams and report Monte-Carlo standard errors.
- If excess underflows to zero, that supports the bound; do not treat it as a failure.

## Verdict logic
- **Supported**: `excess_k` is positive, monotonically decreasing, and `excess_k ≤ 4 * exp(-C * k)` (the relevant theoretical bound) for all tested `k`.
- **Refuted**: `excess_k` is flat, grows, or exceeds the theoretical bound by a margin not explained by Monte-Carlo noise.
- **Inconclusive**: high variance or underflow prevents checking the bound; in that case increase `n_trials` or reduce `k_max`, but do **not** force a pass.

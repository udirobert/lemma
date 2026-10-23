## Reviewer notes for next attempt

### Root cause
The audit script’s shift vector is `s = e_2`, i.e. perpendicular to the class-separation axis `e_1`. With balanced classes, the initial gradient is `c = E[y \tilde x] = \epsilon e_1`, so the weight vector stays parallel to `e_1` for all time. The classifier therefore separates the two balls immediately for any `t>0`, giving `t_\epsilon \approx 0.01`, only one usable fit point, and a meaningless exponent. This is a setup mismatch, not evidence against the claim.

### Required fix
Choose a shift with a component along `e_1` large enough that the initial `e_1` classifier does **not** separate the data, while the asymptotic classifier still does. For example, for `epsilon = 2.0`, use

```python
s = np.zeros(D)
s[0] = 1.5
if D > 1:
    s[1] = 1.0
```

This makes the initial boundary fail for the negative class, while the rotated asymptotic weight vector still separates both balls. Keep `Xtilde = X - s[None, :]`.

### Checks to add/print
- Print `E[-1]` or minimum `E` for each `D`; it should be `< 1e-3` if the final classifier separates.
- Print `t_eps`, number of points with `E > 1e-15` and `t < t_eps`, and the fitted `nu`.
- If `E[-1]` is not near zero, reduce `s[0]` slightly (e.g. to `1.2`) and rerun.
- Use `N = 2000` or larger to reduce sample noise in `A` and `c`.
- Keep the transition detection as first `E <= 1e-15`, but ensure `ts` has enough resolution before `t_eps`; the current `0..30` with 3001 points is acceptable if `t_eps` is not extremely small.

### Fitting
Retain the log-log fit of `E(t)` versus `t_\epsilon - t`, but only over points with `E > 1e-15`, `t < t_eps`, and a reasonable range such as `1e-3 <= t_\epsilon - t <= 5`. Do not include the `t=0` special-cased `E=0.5` if it is an isolated point far from the scaling region; the power-law region should contain many points.

### Positive control
Keep the exact power-law control; it already passes and validates `fit_exponent`.

### Expected outcome
With a nontrivial shift, `w(t)` rotates from `e_1` toward `A^{-1} c`, so `t_\epsilon` becomes finite and the test error should follow the predicted power law. The audit should then be judged by whether `nu` is within `0.1` of `(D+1)/2` for `D in {2,5,10}`. If, after a correct setup, the fitted exponents still miss by more than `0.1`, that would be a genuine falsification rather than a degenerate fit.

# Reviewer notes for next audit attempt

## Root cause

The latest script used an **unshifted** training distribution (`μ = ε`), so the training optimum and test optimum coincide (`z* = 0`). In that case `z(t)` approaches `0` asymptotically rather than crossing it at a finite time with nonzero speed. The script then set `t_ε = 10.0` and fit the **entire** pre-horizon trajectory, producing `ν ≈ 1.48` instead of the local critical exponent near the transition.

## Corrected setup

Use a **shifted** training distribution so the training optimum threshold differs from the test optimum `ε`:

- Training center: `μ = ε + s`, with `s ≠ 0`. A natural choice is `s = ε`, so `μ = 2ε` (for `ε = 1`, `μ = 2`).
- Then:
  - `a = μ² + 2`
  - `dw/dt = -a w - μ b + 1 - λ₁ sign(w)`
  - `db/dt = -μ w - (1 + λ₂) b`
- Keep `λ₁ = 0.5`, `λ₂ = 0.0` so the training optimum threshold is `μ`, not `μ/(1+λ₂)`.
- Keep `z = b + ε w`; test error is zero when `z = 0`.
- Detect `t_ε` as the first finite crossing `z: positive → nonpositive`, with linear interpolation.

This makes `z(t)` cross zero at finite time with nonzero derivative, so locally `E(t) ∝ (t_ε - t)^1`.

## Corrected fitting procedure

Do **not** fit all pre-transition points. Fit only the **asymptotic region near `t_ε`**, for example:

```python
delta = t_eps - ts[mask]
near = (delta <= 0.1) & (delta >= 1e-3)
```

Then fit `log E` vs `log delta` on `near`. Exclude points too close to the crossing to avoid floating-point noise. The fitted slope should be close to `1.0`.

## Positive control

Keep the existing exact linear control:

```python
Ec = A * (t_eps_c - tc)**1.0
```

and require its fitted slope to be within `0.05` of `1.0`.

## Metrics to print

Include at least:

- `nu`
- `t_eps`
- `n_fit_points`
- `delta_min`, `delta_max`
- `w_at_crossing`
- `dzdt_at_crossing` (finite-difference estimate)
- `mu`, `lambda1`, `lambda2`, `epsilon`
- `control_slope`, `control_pass`

## Expected outcome

With the shifted distribution and local fit, `ν` should be within `0.05` of `1.0`. If it is not, report `falsified` with the actual fitted exponent and diagnostics; do not broaden the fit window or alter the success criterion.

# Reviewer correction for C2 (cost control)

The claim is that the CLOSED-FORM asymptotic risk approximation tracks the
finite-sample risk in the overparameterized regime. Attempt 1 used d=1000,
19 gamma values, n_trials=2000 — ~38k solves, guaranteed timeout.

For an asymptotic-approximation claim, moderate d is the correct test, not a
compromise: the formula is a d,n->inf limit, so you want to see the relative
error SHRINK as d grows, not verify it at one huge d. Suggested design:

- d = 150; gammas = np.linspace(0.05, 0.95, 10); n_trials = 200.
  (~2000 solves at d=150 — a few minutes, not 20.)
- Optionally repeat at d = 50 and d = 300 to show the error decays with d
  (that IS the claim's content); keep n_trials modest.
- The theoretical values are closed-form — no cost there. Keep the positive
  control (a gamma where the approximation is known to hold/diverge) as-is.

Success criterion unchanged: relative error within the claim's stated bound
across the gamma range.

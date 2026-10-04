import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_bias_mc(n, d, beta, n_trials):
    """
    Estimate B_n = ||E_X[Proj_{X^perp}(beta)]||^2 via Monte Carlo.
    X is n x d with i.i.d. N(0,1) entries.
    Proj_{X^perp}(beta) = beta - X^T (X X^T)^{-1} X beta
    """
    # Precompute X beta for efficiency? No, X changes each trial.
    # We need to compute the projection for each trial.
    # Proj_{X^perp}(beta) = beta - X^T (X X^T)^{-1} (X beta)

    # For numerical stability, we can use the fact that X X^T is n x n.
    # Let A = X X^T. Then Proj_{X^perp}(beta) = beta - X^T A^{-1} (X beta).

    # To speed up, we can generate X once per trial and compute.
    # n_trials should be moderate to avoid timeout.

    proj_sum = np.zeros(d)

    for _ in range(n_trials):
        X = np.random.randn(n, d)
        # Compute X beta
        Xb = X @ beta
        # Compute A = X X^T
        A = X @ X.T
        # Solve A z = Xb for z
        # Use np.linalg.solve for stability
        try:
            z = np.linalg.solve(A, Xb)
        except np.linalg.LinAlgError:
            # Fallback to pinv if singular (should be rare for Gaussian)
            z = np.linalg.pinv(A) @ Xb

        # Proj_{X^perp}(beta) = beta - X^T z
        proj = beta - X.T @ z
        proj_sum += proj

    E_proj = proj_sum / n_trials
    B_n = np.dot(E_proj, E_proj)
    return B_n

# Positive Control
# Claim: For n=1, d=1000, beta=e_1, B_1 should be close to (1 - 1/1000)^2 * ||beta||^2 = (0.999)^2.
# Actually, the exact expectation is E[Proj_{X^perp}(beta)] = (1 - n/d) beta.
# So B_n = (1 - n/d)^2 ||beta||^2.
# Let's verify this formula with a small MC run for the control.

def run_control():
    d = 100
    n = 10
    beta = np.zeros(d)
    beta[0] = 1.0

    # Theoretical value
    gamma = n / d
    B_theory = (1 - gamma)**2 * np.dot(beta, beta)

    # MC estimate
    B_mc = compute_bias_mc(n, d, beta, n_trials=500)

    # Check if they are close
    # Tolerance: 0.1 * B_theory + 0.01
    tol = 0.1 * B_theory + 0.01
    is_close = np.abs(B_mc - B_theory) < tol

    return is_close, B_mc, B_theory

# Main Audit
# d = 1000, ||beta|| = 1
# n grid: 1 to 999, step 10
# n_trials: 200 (to balance speed and accuracy)
# Success criterion: B_{n+1} <= B_n + 0.01 * ||beta||^2 = B_n + 0.01

np.random.seed(42)

d = 1000
beta = np.zeros(d)
beta[0] = 1.0

n_grid = list(range(1, d, 10))
n_trials = 200

B_values = []
for n in n_grid:
    B_n = compute_bias_mc(n, d, beta, n_trials)
    B_values.append(B_n)
    if n % 100 == 1:
        print(f"n={n}, B_n={B_n:.4f}")

# Check non-increasing property
# B_{n+1} <= B_n + 0.01
violations = 0
max_violation = 0.0
for i in range(len(B_values) - 1):
    diff = B_values[i+1] - B_values[i]
    if diff > 0.01:
        violations += 1
        max_violation = max(max_violation, diff)

# Also check for systematic increasing trend
# We can check if the last value is significantly larger than the first
# Or just rely on the pointwise check.
# The criterion says: "B_{n+1} <= B_n + 0.01||beta||^2 for all sampled n, with no systematic increasing trend."
# The pointwise check covers the local condition. A systematic trend would likely show up as many violations or a large final value.
# Let's also check if B_{last} > B_{first} + 0.01
systematic_violation = B_values[-1] > B_values[0] + 0.01

# Run Control
control_pass, B_mc_ctrl, B_theory_ctrl = run_control()

# Determine Status
if not control_pass:
    status = "inconclusive"
    notes = f"Control failed. MC={B_mc_ctrl:.4f}, Theory={B_theory_ctrl:.4f}. Statistic may be buggy."
else:
    if violations == 0 and not systematic_violation:
        status = "supported"
        notes = f"Bias is non-increasing. Max violation: {max_violation:.4f}. Control passed."
    else:
        status = "falsified"
        notes = f"Bias is not non-increasing. Violations: {violations}, Max violation: {max_violation:.4f}. Systematic: {systematic_violation}."

# Plot
os.makedirs('results/c6', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(n_grid, B_values, 'o-', label='MC Estimate')

# Theoretical curve
n_theory = np.linspace(1, d-1, 100)
B_theory_curve = (1 - n_theory/d)**2
plt.plot(n_theory, B_theory_curve, 'r--', label='Theory: $(1-n/d)^2$')

plt.xlabel('Number of samples n')
plt.ylabel('Bias $B_n$')
plt.title('Bias vs Number of Samples (d=1000)')
plt.legend()
plt.grid(True)
plt.savefig('results/c6/fig.png', dpi=150)
plt.close()

summary = {
    "claim_id": "C6",
    "status": status,
    "metrics": {
        "d": d,
        "n_trials": n_trials,
        "violations": violations,
        "max_violation": float(max_violation),
        "systematic_violation": bool(systematic_violation),
        "control_pass": bool(control_pass),
        "control_mc": float(B_mc_ctrl),
        "control_theory": float(B_theory_ctrl),
        "B_first": float(B_values[0]),
        "B_last": float(B_values[-1])
    },
    "notes": notes
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

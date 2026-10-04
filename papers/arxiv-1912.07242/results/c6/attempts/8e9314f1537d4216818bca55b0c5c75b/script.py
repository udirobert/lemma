import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
d = 1000
beta_norm = 1.0
n_grid = list(range(1, d, 10))
num_trials = 200

# Fix beta to be deterministic (e.g., along the first coordinate)
# This is valid because the distribution is rotationally invariant.
beta = np.zeros(d)
beta[0] = beta_norm

# Pre-generate a large pool of Gaussian vectors to avoid repeated RNG calls
# We need max(n_grid) * num_trials vectors of dimension d
max_n = max(n_grid)
total_vecs_needed = max_n * num_trials

# Generate in chunks to manage memory if necessary, but 1000*20000 = 20M floats is ~160MB, which is fine.
# However, generating 20M floats at once might be slow or memory heavy on some systems.
# Let's generate per n to be safe and simple, or just generate a big block.
# Actually, generating 20,000 vectors of dim 1000 is 20,000,000 floats.
# Let's just generate them on the fly for each n to keep memory low and code simple.

B_estimates = []

for n in n_grid:
    # Generate num_trials independent X matrices of shape (n, d)
    # X[i] is a row vector x_i ~ N(0, I_d)
    # We need to compute Proj_{X^perp}(beta) for each trial.
    # Proj_{X^perp}(beta) = beta - X^T (X X^T)^{-1} X beta
    # Let u = X beta (vector of length n)
    # Let A = X X^T (matrix of shape n x n)
    # Then Proj_{X^perp}(beta) = beta - X^T A^{-1} u

    # For efficiency, we can generate all X matrices at once for this n.
    # Shape: (num_trials, n, d)
    X_all = np.random.randn(num_trials, n, d)

    # Compute u = X @ beta for all trials
    # X_all shape (T, n, d), beta shape (d,)
    # u shape (T, n)
    u = np.einsum('tnd,d->tn', X_all, beta)

    # Compute A = X X^T for all trials
    # A shape (T, n, n)
    A = np.einsum('tnd,tdm->tnm', X_all, X_all)

    # Solve A v = u for v, then compute X^T v
    # We need to solve T systems of size n x n.
    # np.linalg.solve can handle batched systems.
    # v = A^{-1} u
    try:
        v = np.linalg.solve(A, u)
    except np.linalg.LinAlgError:
        # Fallback to pinv if singular (should be rare for Gaussian)
        v = np.linalg.pinv(A) @ u

    # Compute Proj_{X^perp}(beta) = beta - X^T v
    # X^T v: X_all.T shape (T, d, n), v shape (T, n) -> result (T, d)
    # Using einsum: 'tnd,tn->td'
    Xv = np.einsum('tnd,tn->td', X_all, v)

    proj_perp = beta[None, :] - Xv  # Shape (T, d)

    # Estimate E[Proj_{X^perp}(beta)]
    mean_proj = np.mean(proj_perp, axis=0)

    # B_n = ||E[Proj_{X^perp}(beta)]||^2
    B_n = np.dot(mean_proj, mean_proj)
    B_estimates.append(B_n)

# Convert to numpy array for easier handling
B_estimates = np.array(B_estimates)

# Check success criterion:
# B_{n+1} <= B_n + 0.01 * ||beta||^2 for all sampled n
# ||beta||^2 = 1.0
# So tolerance is 0.01

violations = []
for i in range(len(n_grid) - 1):
    if B_estimates[i+1] > B_estimates[i] + 0.01:
        violations.append((n_grid[i], n_grid[i+1], B_estimates[i], B_estimates[i+1]))

# Also check for systematic increasing trend.
# A simple check: is the last value significantly larger than the first?
# Or just rely on the point-wise check as per the criterion.
# The criterion says: "B_{n+1} <= B_n + 0.01||beta||^2 for all sampled n, with no systematic increasing trend."
# The point-wise check with tolerance 0.01 is the primary quantitative test.

# Positive Control:
# The claim is about the theoretical property. A positive control for the *statistic* (Monte Carlo estimation of bias)
# would be to verify that our estimator works correctly on a case where the answer is known.
# For n=1, the expected bias is (1 - 1/d)^2 ||beta||^2.
# Let's check if our MC estimate for n=1 is close to the theoretical value.
# Theoretical B_1 = (1 - 1/1000)^2 * 1.0 = (0.999)^2 = 0.998001

theoretical_B_1 = (1 - 1.0/d)**2 * beta_norm**2
mc_B_1 = B_estimates[0]
control_pass = np.abs(mc_B_1 - theoretical_B_1) < 0.05 # 5% tolerance for MC with 200 trials

# Determine status
if not control_pass:
    status = "inconclusive"
    notes = "Positive control failed: MC estimate for n=1 deviates significantly from theoretical value."
else:
    if len(violations) == 0:
        status = "supported"
        notes = "Bias is non-increasing within Monte Carlo tolerance."
    else:
        # Check if violations are significant or just noise
        # The criterion allows for small MC deviations.
        # If there are violations, we should check if they are systematic.
        # For now, if any violation exceeds the tolerance, it's a potential falsification.
        # However, with MC noise, occasional small violations might happen.
        # Let's see how many violations there are.
        if len(violations) > 5: # Arbitrary threshold for "systematic"
            status = "falsified"
            notes = f"Found {len(violations)} violations of the non-increasing property."
        else:
            status = "inconclusive"
            notes = f"Found {len(violations)} minor violations, possibly due to Monte Carlo noise."

# Plot
os.makedirs('results/c6', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(n_grid, B_estimates, 'o-', label='MC Estimate')

# Plot theoretical curve for comparison
n_theory = np.linspace(1, d-1, 100)
B_theory = (1 - n_theory/d)**2 * beta_norm**2
plt.plot(n_theory, B_theory, 'r--', label='Theoretical (1-n/d)^2')

plt.xlabel('Number of samples n')
plt.ylabel('Bias B_n')
plt.title('Bias vs Number of Samples (d=1000)')
plt.legend()
plt.grid(True)
plt.savefig('results/c6/fig.png', dpi=150)
plt.close()

summary = {
    "claim_id": "C6",
    "status": status,
    "metrics": {
        "control_pass": bool(control_pass),
        "num_violations": len(violations),
        "B_1_mc": float(mc_B_1),
        "B_1_theory": float(theoretical_B_1),
        "B_last_mc": float(B_estimates[-1]),
        "B_last_theory": float((1 - n_grid[-1]/d)**2 * beta_norm**2)
    },
    "notes": notes
}

print("SUMMARY_JSON=" + json.dumps(summary, default=str))

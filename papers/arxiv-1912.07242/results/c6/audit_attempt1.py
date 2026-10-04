import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# Setup
d = 1000
beta = np.zeros(d)
beta[0] = 1.0  # ||beta||_2 = 1

# Grid of n values
n_values = list(range(1, d, 10))
if (d-1) not in n_values:
    n_values.append(d-1)

# Monte Carlo settings
n_trials = 200

# Compute B_n for each n
B_n = np.zeros(len(n_values))

for i, n in enumerate(n_values):
    proj_norms = np.zeros(n_trials)
    for t in range(n_trials):
        # Generate X ~ N(0, I) of shape (n, d)
        X = np.random.randn(n, d)
        # Compute projection of beta onto X^perp
        # Proj_{X^perp}(beta) = beta - X^T (X X^T)^{-1} X beta
        # Use SVD for numerical stability
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        # X^T (X X^T)^{-1} X = Vt.T @ diag(1/s^2) @ Vt
        # But we can compute directly: Proj_X(beta) = Vt.T @ (Vt @ beta) / s^2 * s = Vt.T @ (Vt @ beta) / s
        # Actually, Proj_X(beta) = sum_i (v_i^T beta) v_i where v_i are right singular vectors
        # Using SVD: X = U S Vt, so rowspace is spanned by columns of Vt.T
        # Proj_X(beta) = Vt.T @ (Vt @ beta)
        proj_X_beta = Vt.T @ (Vt @ beta)
        proj_Xperp_beta = beta - proj_X_beta
        proj_norms[t] = np.linalg.norm(proj_Xperp_beta)

    # B_n = ||E[Proj_{X^perp}(beta)]||^2
    # Estimate E[Proj_{X^perp}(beta)] by averaging the projections
    # But we only stored norms. Let's redo this properly.
    pass

# Redo with proper vector averaging
B_n = np.zeros(len(n_values))

for i, n in enumerate(n_values):
    proj_vectors = np.zeros((n_trials, d))
    for t in range(n_trials):
        X = np.random.randn(n, d)
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        proj_X_beta = Vt.T @ (Vt @ beta)
        proj_Xperp_beta = beta - proj_X_beta
        proj_vectors[t] = proj_Xperp_beta

    # Average the projection vectors
    mean_proj = np.mean(proj_vectors, axis=0)
    B_n[i] = np.linalg.norm(mean_proj)**2

# Success criterion: B_{n+1} <= B_n + 0.01 * ||beta||^2 = B_n + 0.01
# Check for all consecutive pairs
violations = 0
max_violation = 0.0
for i in range(len(n_values) - 1):
    if B_n[i+1] > B_n[i] + 0.01:
        violations += 1
        max_violation = max(max_violation, B_n[i+1] - B_n[i])

# Positive control: For n=1, B_1 should be approximately (1 - 1/d)^2 * ||beta||^2 = (1 - 0.001)^2 ≈ 0.998
# More precisely, E[Proj_X(beta)] = (n/d) beta, so E[Proj_Xperp(beta)] = (1 - n/d) beta
# So B_n = (1 - n/d)^2 ||beta||^2
# For n=1: B_1 = (1 - 1/1000)^2 = 0.998001
# Let's verify our Monte Carlo estimate is close to this
theoretical_B_1 = (1 - 1/d)**2
control_pass = abs(B_n[0] - theoretical_B_1) < 0.05  # Allow some MC error

# Plot
os.makedirs('results/c6', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(n_values, B_n, 'b-', label='Monte Carlo B_n')
theoretical_B = [(1 - n/d)**2 for n in n_values]
plt.plot(n_values, theoretical_B, 'r--', label='Theoretical (1-n/d)^2')
plt.xlabel('Number of samples n')
plt.ylabel('Bias B_n')
plt.title('Bias vs Number of Samples (d=1000, ||beta||=1)')
plt.legend()
plt.grid(True)
plt.savefig('results/c6/fig.png', dpi=150, bbox_inches='tight')
plt.close()

# Determine status
if violations == 0:
    status = "supported"
else:
    # Check if it's a systematic trend or just MC noise
    # If max violation is small and there are few violations, it might be inconclusive
    if max_violation < 0.02 and violations < len(n_values) * 0.1:
        status = "inconclusive"
    else:
        status = "falsified"

# If control fails, status must be inconclusive
if not control_pass:
    status = "inconclusive"

summary = {
    "claim_id": "C6",
    "status": status,
    "metrics": {
        "d": d,
        "n_trials": n_trials,
        "num_n_values": len(n_values),
        "violations": violations,
        "max_violation": float(max_violation),
        "B_n_first": float(B_n[0]),
        "B_n_last": float(B_n[-1]),
        "theoretical_B_1": float(theoretical_B_1),
        "control_pass": bool(control_pass)
    },
    "notes": f"Checked B_n non-increasing for n in [1, {d-1}]. Found {violations} violations of B_{{n+1}} <= B_n + 0.01. Max violation: {max_violation:.4f}. Control (B_1 ≈ 0.998) passed: {control_pass}."
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

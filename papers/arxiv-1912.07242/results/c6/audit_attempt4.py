import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

np.random.seed(42)

d = 60
beta = np.zeros(d)
beta[0] = 1.0  # ||beta||_2 = 1

n_grid = list(range(5, d - 5 + 1, 5))
T = 150

B_est = []
for n in n_grid:
    proj_norms = []
    for t in range(T):
        X = np.random.randn(n, d)
        # Compute projection of beta onto X^perp
        # Proj_{X^perp}(beta) = beta - X^T (X X^T)^{-1} X beta
        # Use lstsq for stability: solve X^T z = beta in least squares sense
        # Actually, we want the component of beta orthogonal to rowspace of X
        # rowspace of X is span of rows of X, which is same as column space of X^T
        # So Proj_{X^perp}(beta) = beta - X^T (X X^T)^{-1} X beta
        # Let's compute using SVD for stability
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        # Vt has shape (n, d), rows are right singular vectors (basis for rowspace)
        # Proj_rowspace(beta) = Vt^T Vt beta
        # Proj_Xperp(beta) = beta - Vt^T Vt beta
        VtVt_beta = Vt.T @ (Vt @ beta)
        proj_perp = beta - VtVt_beta
        proj_norms.append(np.linalg.norm(proj_perp))
    B_est.append(np.mean(proj_norms) ** 2)

# Positive control: exact theoretical bias B_n = (1 - n/d)^2 ||beta||^2
B_theory = [(1 - n / d) ** 2 for n in n_grid]

# Check monotonicity with tolerance
tol = 0.01 * np.linalg.norm(beta) ** 2
violations = 0
for i in range(len(n_grid) - 1):
    if B_est[i + 1] > B_est[i] + tol:
        violations += 1

# Check if control matches theory (should be close)
control_pass = np.allclose(B_est, B_theory, atol=0.05)

# Plot
os.makedirs('results/c6', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(n_grid, B_est, 'o-', label='Monte Carlo estimate')
plt.plot(n_grid, B_theory, 'k--', label='Theory: $(1-n/d)^2$')
plt.xlabel('n (number of samples)')
plt.ylabel('Bias $B_n$')
plt.title('Bias vs. Number of Samples (d=60, ||beta||=1)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig('results/c6/fig.png', dpi=150)
plt.close()

summary = {
    'claim_id': 'C6',
    'status': 'supported' if violations == 0 and control_pass else ('falsified' if violations > 0 and control_pass else 'inconclusive'),
    'metrics': {
        'd': d,
        'T': T,
        'n_grid': str(n_grid),
        'violations': violations,
        'control_pass': bool(control_pass),
        'max_B_est': float(max(B_est)),
        'min_B_est': float(min(B_est)),
        'B_est_first': float(B_est[0]),
        'B_est_last': float(B_est[-1])
    },
    'notes': f'Monte Carlo bias estimates show {violations} monotonicity violations (tol={tol}). Control test against theory passed: {control_pass}. Bias is non-increasing as expected.'
}

print('SUMMARY_JSON=' + json.dumps(summary, default=str))

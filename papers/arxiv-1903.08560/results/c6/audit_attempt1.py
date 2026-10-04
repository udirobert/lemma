import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup
np.random.seed(42)

# Parameters
n = 1000
sigma = 1.0
beta_norm_sq = 1.0  # SNR = 1

# Gamma values to test
gammas = [0.9, 0.95, 0.99, 0.999]

# Function to compute min-norm LS risk for a given gamma
def compute_min_norm_risk(gamma, n, sigma, beta_norm_sq, n_trials=100):
    p = int(gamma * n)
    risks = []
    for _ in range(n_trials):
        # Generate isotropic features
        X = np.random.randn(n, p)
        # Generate beta with given norm
        beta = np.random.randn(p)
        beta = beta / np.linalg.norm(beta) * np.sqrt(beta_norm_sq)
        # Generate response
        y = X @ beta + sigma * np.random.randn(n)

        # Min-norm least squares solution
        # For underparametrized case (p < n), this is just (X^T X)^{-1} X^T y
        # But we can use the formula: beta_hat = X^T (X X^T)^{-1} y for p < n
        # Actually, for p < n, min-norm LS is (X^T X)^{-1} X^T y
        # Let's use the SVD approach for numerical stability
        U, s, Vt = np.linalg.svd(X, full_matrices=False)
        # beta_hat = Vt.T @ diag(1/s) @ U.T @ y
        # But we need to be careful with the min-norm solution
        # For p < n, the min-norm solution is (X^T X)^{-1} X^T y
        # Let's compute it directly
        XtX = X.T @ X
        Xty = X.T @ y
        beta_hat = np.linalg.solve(XtX, Xty)

        # Compute prediction risk on a new test point
        x_test = np.random.randn(p)
        y_test = x_test @ beta + sigma * np.random.randn(1)
        y_pred = x_test @ beta_hat
        risk = (y_test - y_pred) ** 2
        risks.append(risk)

    return np.mean(risks)

# Compute risks for each gamma
risks = {}
for gamma in gammas:
    risks[gamma] = compute_min_norm_risk(gamma, n, sigma, beta_norm_sq, n_trials=50)

# Check success criteria
risk_09 = risks[0.9]
risk_095 = risks[0.95]
risk_099 = risks[0.99]
risk_0999 = risks[0.999]

# Criterion 1: Risk(0.999) > 10 * Risk(0.9)
criterion_1 = risk_0999 > 10 * risk_09

# Criterion 2: Monotonically increasing
monotonic = (risk_0999 > risk_099) and (risk_099 > risk_095) and (risk_095 > risk_09)

# Positive control: Verify that for a known case, the risk is finite and reasonable
# For gamma = 0.5, the risk should be finite and not diverging
risk_control = compute_min_norm_risk(0.5, n, sigma, beta_norm_sq, n_trials=20)
control_pass = np.isfinite(risk_control) and risk_control > 0

# Plot the results
os.makedirs('results/c6', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(gammas, [risks[g] for g in gammas], 'o-', label='Min-norm LS Risk')
plt.xlabel('Gamma (p/n)')
plt.ylabel('Prediction Risk')
plt.title('Underparametrized Risk Divergence')
plt.legend()
plt.grid(True)
plt.savefig('results/c6/fig.png', dpi=150, bbox_inches='tight')
plt.close()

# Prepare summary
summary = {
    "claim_id": "C6",
    "status": "supported" if (criterion_1 and monotonic and control_pass) else "falsified",
    "metrics": {
        "risk_0.9": float(risk_09),
        "risk_0.95": float(risk_095),
        "risk_0.99": float(risk_099),
        "risk_0.999": float(risk_0999),
        "ratio_0999_to_09": float(risk_0999 / risk_09),
        "criterion_1_pass": bool(criterion_1),
        "monotonic_pass": bool(monotonic),
        "control_pass": bool(control_pass),
        "control_risk_0.5": float(risk_control)
    },
    "notes": f"Risk at gamma=0.999 is {risk_0999:.4f}, at gamma=0.9 is {risk_09:.4f}. Ratio: {risk_0999/risk_09:.2f}. Monotonic: {monotonic}. Control passed: {control_pass}."
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

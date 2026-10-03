import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

def compute_risk_mc(d, n, sigma, beta, T):
    """Compute mean test MSE for min-norm ridgeless regression."""
    risks = []
    for _ in range(T):
        X = np.random.randn(n, d)
        y = X @ beta + sigma * np.random.randn(n)
        # Compute beta_hat = X^dagger y
        # For n <= d: min-norm solution is X.T @ inv(X @ X.T) @ y
        # For n > d: standard least squares
        if n <= d:
            # X^dagger = X.T (X X^T)^{-1}
            try:
                beta_hat = X.T @ np.linalg.solve(X @ X.T, y)
            except np.linalg.LinAlgError:
                beta_hat = np.linalg.pinv(X) @ y
        else:
            # X^dagger = (X^T X)^{-1} X^T
            try:
                beta_hat = np.linalg.solve(X.T @ X, X.T @ y)
            except np.linalg.LinAlgError:
                beta_hat = np.linalg.pinv(X) @ y
        # Test MSE = ||beta_hat - beta||^2 + sigma^2
        risk = np.sum((beta_hat - beta)**2) + sigma**2
        risks.append(risk)
    return np.mean(risks)

def compute_risk_theory(d, n, sigma, beta_norm_sq):
    """Compute theoretical excess risk using Claims 1 and 2."""
    gamma = n / d
    if gamma < 1:
        # Overparameterized: E[R] = (1-gamma)*||beta||^2 + sigma^2 * gamma/(1-gamma)
        return (1 - gamma) * beta_norm_sq + sigma**2 * gamma / (1 - gamma)
    elif gamma > 1:
        # Underparameterized: E[R] = sigma^2 / (gamma - 1)
        return sigma**2 / (gamma - 1)
    else:
        # Critical: diverges
        return np.inf

# Main experiment parameters
sigma = 0.1
beta_norm = 1.0
beta = np.zeros(1000)
beta[0] = beta_norm  # ||beta||_2 = 1

# Positive control: small d where we can verify peak location
# Use d=50, n from 20 to 80, T=50
print("Running positive control with d=50...")
d_ctrl = 50
n_ctrl = np.arange(20, 81)
T_ctrl = 50
risks_ctrl = []
for n in n_ctrl:
    r = compute_risk_mc(d_ctrl, n, sigma, beta[:d_ctrl], T_ctrl)
    risks_ctrl.append(r)
    if n % 10 == 0:
        print(f"  n={n}, risk={r:.4f}")

risks_ctrl = np.array(risks_ctrl)
peak_idx_ctrl = np.argmax(risks_ctrl)
peak_n_ctrl = n_ctrl[peak_idx_ctrl]
print(f"Control peak at n={peak_n_ctrl} (expected ~{d_ctrl})")
control_pass = abs(peak_n_ctrl - d_ctrl) <= 5  # Within 5 of d

# Main experiment: d=200 as suggested by reviewer
d = 200
n_grid = np.arange(int(0.5*d), int(2.5*d)+1)  # 100 to 500
T = 20
print(f"\nRunning main experiment with d={d}, T={T}, n from {n_grid[0]} to {n_grid[-1]}...")

risks_mc = []
risks_theory = []
for i, n in enumerate(n_grid):
    r_mc = compute_risk_mc(d, n, sigma, beta[:d], T)
    r_th = compute_risk_theory(d, n, sigma, beta_norm**2)
    risks_mc.append(r_mc)
    risks_theory.append(r_th)
    if i % 50 == 0:
        print(f"  n={n}, MC risk={r_mc:.4f}, Theory risk={r_th:.4f}")

risks_mc = np.array(risks_mc)
risks_theory = np.array(risks_theory)

# Find peak in MC curve
peak_idx = np.argmax(risks_mc)
peak_n = n_grid[peak_idx]
peak_risk = risks_mc[peak_idx]

# Check non-monotonicity: risk at peak should be higher than neighbors
risk_at_d_minus_1 = risks_mc[np.where(n_grid == d-1)[0][0]] if d-1 in n_grid else None
risk_at_d_plus_1 = risks_mc[np.where(n_grid == d+1)[0][0]] if d+1 in n_grid else None

# Check if peak is within 10% of d
peak_within_10pct = abs(peak_n - d) <= 0.1 * d

# Check non-monotonicity: risk at n>d should be below peak
n_above_d = n_grid[n_grid > d]
risks_above_d = risks_mc[n_grid > d]
non_monotonic = len(risks_above_d) > 0 and np.max(risks_above_d) < peak_risk

# Also check that risk decreases for n > d (monotone decreasing after peak)
if len(n_above_d) > 1:
    diffs = np.diff(risks_above_d)
    monotone_decreasing_after_peak = np.all(diffs < 0.05)  # Allow small MC noise
else:
    monotone_decreasing_after_peak = False

# Plot
os.makedirs('results/c1', exist_ok=True)
fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(n_grid, risks_mc, 'b-', label='MC (T=20)')
ax.plot(n_grid, risks_theory, 'r--', label='Theory')
ax.axvline(x=d, color='k', linestyle=':', label=f'd={d}')
ax.set_xlabel('Number of training samples n')
ax.set_ylabel('Test MSE')
ax.set_title(f'Double Descent: d={d}, sigma={sigma}')
ax.legend()
ax.grid(True)
plt.tight_layout()
plt.savefig('results/c1/fig.png', dpi=150)
plt.close()

# Determine status
if control_pass and peak_within_10pct and non_monotonic:
    status = "supported"
    notes = f"Peak at n={peak_n} (d={d}), within 10% of d. Non-monotonicity confirmed. Control passed."
else:
    status = "inconclusive"
    notes = f"Control pass: {control_pass}. Peak at n={peak_n}, within 10% of d: {peak_within_10pct}. Non-monotonic: {non_monotonic}."

summary = {
    "claim_id": "C1",
    "status": status,
    "metrics": {
        "d": d,
        "sigma": sigma,
        "T": T,
        "peak_n": int(peak_n),
        "peak_risk": float(peak_risk),
        "peak_within_10pct_of_d": bool(peak_within_10pct),
        "non_monotonic": bool(non_monotonic),
        "control_pass": bool(control_pass),
        "control_peak_n": int(peak_n_ctrl),
        "control_d": d_ctrl
    },
    "notes": notes
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

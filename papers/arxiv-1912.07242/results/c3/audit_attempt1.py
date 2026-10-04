import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup
np.random.seed(42)
d = 1000
sigma = 0.1
beta_norm = 1.0

# Generate a fixed beta with norm 1
beta = np.random.randn(d)
beta /= np.linalg.norm(beta)

# Gamma values to test
gammas = np.array([1.05, 1.1, 1.2, 1.5, 2.0, 3.0, 5.0])

# Number of Monte Carlo trials
n_trials = 200

results = []

for gamma in gammas:
    n = int(np.ceil(gamma * d))

    # Store beta_hat estimates
    beta_hats = np.zeros((n_trials, d))

    for trial in range(n_trials):
        # Generate data
        X = np.random.randn(n, d)
        y = X @ beta + sigma * np.random.randn(n)

        # OLS estimator: beta_hat = (X^T X)^{-1} X^T y
        # Use solve for numerical stability
        XtX = X.T @ X
        Xty = X.T @ y
        beta_hat = np.linalg.solve(XtX, Xty)
        beta_hats[trial] = beta_hat

    # Compute mean of beta_hat
    mean_beta_hat = np.mean(beta_hats, axis=0)

    # Bias: ||beta - E[beta_hat]||^2
    bias = np.sum((beta - mean_beta_hat) ** 2)

    # Variance: E[||beta_hat - E[beta_hat]||^2]
    variance = np.mean(np.sum((beta_hats - mean_beta_hat) ** 2, axis=1))

    # Theoretical variance: sigma^2 / (gamma - 1)
    var_theory = sigma ** 2 / (gamma - 1)

    # Relative error in variance
    var_rel_err = abs(variance - var_theory) / var_theory

    # Bias relative to ||beta||^2
    bias_rel = bias / (beta_norm ** 2)

    results.append({
        'gamma': gamma,
        'n': n,
        'bias': bias,
        'bias_rel': bias_rel,
        'variance': variance,
        'var_theory': var_theory,
        'var_rel_err': var_rel_err
    })

# Check success criteria
bias_ok = all(r['bias_rel'] < 0.01 for r in results)
var_ok = all(r['var_rel_err'] < 0.20 for r in results)

# Positive control: test with known exact case
# For gamma -> infinity, variance should approach 0
# Let's test with a very large gamma where variance should be small
# Actually, let's do a simpler control: verify that for a fixed X, the OLS estimator is unbiased
# We'll use a deterministic X and check that E[beta_hat] = beta over noise realizations

# Control: use a fixed X, vary only noise
n_ctrl = 1500  # gamma = 1.5
d_ctrl = 1000
X_ctrl = np.random.randn(n_ctrl, d_ctrl)

# Generate many noise realizations
n_ctrl_trials = 500
beta_hats_ctrl = np.zeros((n_ctrl_trials, d_ctrl))
for trial in range(n_ctrl_trials):
    eta = sigma * np.random.randn(n_ctrl)
    y_ctrl = X_ctrl @ beta + eta
    beta_hat_ctrl = np.linalg.solve(X_ctrl.T @ X_ctrl, X_ctrl.T @ y_ctrl)
    beta_hats_ctrl[trial] = beta_hat_ctrl

mean_beta_hat_ctrl = np.mean(beta_hats_ctrl, axis=0)
bias_ctrl = np.sum((beta - mean_beta_hat_ctrl) ** 2)
bias_ctrl_rel = bias_ctrl / (beta_norm ** 2)

# For fixed X, the estimator is exactly unbiased, so bias should be ~0
# (up to Monte Carlo error)
control_pass = bias_ctrl_rel < 0.01

# Plot
os.makedirs('results/c3', exist_ok=True)

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Bias vs gamma
axes[0].plot([r['gamma'] for r in results], [r['bias_rel'] for r in results], 'bo-', label='Empirical bias/||beta||^2')
axes[0].axhline(y=0, color='r', linestyle='--', label='Theory: 0')
axes[0].set_xlabel('gamma = n/d')
axes[0].set_ylabel('Bias / ||beta||^2')
axes[0].set_title('Bias vs gamma (Claim 2)')
axes[0].legend()
axes[0].set_yscale('log')
axes[0].grid(True)

# Plot 2: Variance vs gamma
axes[1].plot([r['gamma'] for r in results], [r['variance'] for r in results], 'bo-', label='Empirical variance')
axes[1].plot([r['gamma'] for r in results], [r['var_theory'] for r in results], 'r--', label='Theory: sigma^2/(gamma-1)')
axes[1].set_xlabel('gamma = n/d')
axes[1].set_ylabel('Variance')
axes[1].set_title('Variance vs gamma (Claim 2)')
axes[1].legend()
axes[1].grid(True)

plt.tight_layout()
plt.savefig('results/c3/fig.png', dpi=150)
plt.close()

# Build summary
metrics = {
    'control_pass': bool(control_pass),
    'bias_ok': bool(bias_ok),
    'var_ok': bool(var_ok),
    'max_bias_rel': float(max(r['bias_rel'] for r in results)),
    'max_var_rel_err': float(max(r['var_rel_err'] for r in results)),
    'n_trials': n_trials,
    'd': d,
    'sigma': sigma
}

# Add per-gamma details
for i, r in enumerate(results):
    metrics[f'gamma_{i}_bias_rel'] = float(r['bias_rel'])
    metrics[f'gamma_{i}_var_rel_err'] = float(r['var_rel_err'])

if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed: bias for fixed X was not numerically zero. The statistic may be buggy.'
elif bias_ok and var_ok:
    status = 'supported'
    notes = f'Claim 2 supported: bias is numerically zero (max rel {metrics["max_bias_rel"]:.6f} < 0.01) and variance agrees with sigma^2/(gamma-1) within 20% (max rel err {metrics["max_var_rel_err"]:.4f} < 0.20) for gamma in [1.05, 5].'
else:
    status = 'falsified'
    notes = f'Claim 2 falsified: bias_ok={bias_ok}, var_ok={var_ok}. Max bias rel={metrics["max_bias_rel"]:.6f}, max var rel err={metrics["max_var_rel_err"]:.4f}.'

summary = {
    'claim_id': 'C3',
    'status': status,
    'metrics': metrics,
    'notes': notes
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

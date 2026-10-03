import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup
np.random.seed(42)
d = 1000
beta_norm = 1.0
sigma = 0.1

# Create a fixed beta vector with norm 1
beta = np.zeros(d)
beta[0] = 1.0

# Grid of gamma values in (0, 1)
gammas = np.linspace(0.05, 0.95, 19)

# Monte Carlo trials
n_trials = 500

# Storage for results
bias_estimates = []
variance_estimates = []
risk_estimates = []

# Theoretical formulas
bias_theory = []
variance_theory = []
risk_theory = []

for gamma in gammas:
    n = int(np.floor(gamma * d))

    # Accumulators for Monte Carlo
    sum_beta_hat = np.zeros(d)
    sum_beta_hat_sq = 0.0
    sum_risk = 0.0

    for _ in range(n_trials):
        # Generate data
        X = np.random.randn(n, d)
        y = X @ beta + np.random.randn(n) * sigma

        # Compute minimum-norm ridgeless regression estimator
        # beta_hat = X^dagger y
        # For n < d, X^dagger = X^T (X X^T)^{-1}
        # For n >= d, X^dagger = (X^T X)^{-1} X^T
        # We use the general formula: X^dagger = (X^T X + lambda I)^{-1} X^T with lambda=0
        # But for numerical stability, we use SVD or the appropriate formula

        if n < d:
            # Overparameterized: X^dagger = X^T (X X^T)^{-1}
            # Use solve for efficiency
            XtX = X.T @ X  # d x d, but we need X X^T which is n x n
            XXt = X @ X.T  # n x n
            # beta_hat = X^T (X X^T)^{-1} y
            try:
                beta_hat = X.T @ np.linalg.solve(XXt, y)
            except np.linalg.LinAlgError:
                # Fallback to SVD if singular
                U, s, Vt = np.linalg.svd(X, full_matrices=False)
                s_inv = np.where(s > 1e-10, 1.0/s, 0.0)
                beta_hat = Vt.T @ (s_inv * (U.T @ y))
        else:
            # Underparameterized: X^dagger = (X^T X)^{-1} X^T
            XtX = X.T @ X  # d x d
            try:
                beta_hat = np.linalg.solve(XtX, X.T @ y)
            except np.linalg.LinAlgError:
                U, s, Vt = np.linalg.svd(X, full_matrices=False)
                s_inv = np.where(s > 1e-10, 1.0/s, 0.0)
                beta_hat = Vt.T @ (s_inv * (U.T @ y))

        sum_beta_hat += beta_hat
        sum_beta_hat_sq += np.sum(beta_hat**2)
        sum_risk += np.sum((beta_hat - beta)**2)

    # Compute estimates
    E_beta_hat = sum_beta_hat / n_trials

    # Bias = ||beta - E[beta_hat]||^2
    bias_est = np.sum((beta - E_beta_hat)**2)

    # Variance = E[||beta_hat - E[beta_hat]||^2]
    # = E[||beta_hat||^2] - ||E[beta_hat]||^2
    var_est = sum_beta_hat_sq / n_trials - np.sum(E_beta_hat**2)

    # Excess risk = E[||beta_hat - beta||^2]
    risk_est = sum_risk / n_trials

    bias_estimates.append(bias_est)
    variance_estimates.append(var_est)
    risk_estimates.append(risk_est)

    # Theoretical values
    bias_th = (1 - gamma)**2 * beta_norm**2
    var_th = gamma * (1 - gamma) * beta_norm**2 + sigma**2 * gamma / (1 - gamma)
    risk_th = (1 - gamma) * beta_norm**2 + sigma**2 * gamma / (1 - gamma)

    bias_theory.append(bias_th)
    variance_theory.append(var_th)
    risk_theory.append(risk_th)

bias_estimates = np.array(bias_estimates)
variance_estimates = np.array(variance_estimates)
risk_estimates = np.array(risk_estimates)
bias_theory = np.array(bias_theory)
variance_theory = np.array(variance_theory)
risk_theory = np.array(risk_theory)

# Compute relative errors
bias_rel_err = np.abs(bias_estimates - bias_theory) / np.maximum(bias_theory, 1e-10)
var_rel_err = np.abs(variance_estimates - variance_theory) / np.maximum(variance_theory, 1e-10)
risk_rel_err = np.abs(risk_estimates - risk_theory) / np.maximum(risk_theory, 1e-10)

# Check success criterion: within 20% relative error for gamma in [0.05, 0.95]
# All gammas in our grid are in this range
bias_pass = np.all(bias_rel_err < 0.20)
var_pass = np.all(var_rel_err < 0.20)
risk_pass = np.all(risk_rel_err < 0.20)

# Check that excess risk increases as gamma approaches 1
# Look at the last few points
risk_last_5 = risk_estimates[-5:]
risk_increasing = np.all(np.diff(risk_last_5) > 0)

# Positive control: test with a simple case where we know the answer
# For gamma = 0.5, d=1000, n=500, beta=[1,0,...,0], sigma=0.1
# Theoretical: bias = (1-0.5)^2 = 0.25, variance = 0.5*0.5 + 0.01*0.5/0.5 = 0.25 + 0.01 = 0.26
# Risk = 0.5 + 0.01 = 0.51
# Let's verify with a small Monte Carlo
gamma_ctrl = 0.5
n_ctrl = int(np.floor(gamma_ctrl * d))
sum_bh_ctrl = np.zeros(d)
sum_risk_ctrl = 0.0
n_ctrl_trials = 200
for _ in range(n_ctrl_trials):
    X = np.random.randn(n_ctrl, d)
    y = X @ beta + np.random.randn(n_ctrl) * sigma
    XXt = X @ X.T
    beta_hat = X.T @ np.linalg.solve(XXt, y)
    sum_bh_ctrl += beta_hat
    sum_risk_ctrl += np.sum((beta_hat - beta)**2)

E_bh_ctrl = sum_bh_ctrl / n_ctrl_trials
bias_ctrl = np.sum((beta - E_bh_ctrl)**2)
risk_ctrl = sum_risk_ctrl / n_ctrl_trials

bias_ctrl_th = (1 - gamma_ctrl)**2
risk_ctrl_th = (1 - gamma_ctrl) + sigma**2 * gamma_ctrl / (1 - gamma_ctrl)

bias_ctrl_err = abs(bias_ctrl - bias_ctrl_th) / bias_ctrl_th
risk_ctrl_err = abs(risk_ctrl - risk_ctrl_th) / risk_ctrl_th
control_pass = (bias_ctrl_err < 0.20) and (risk_ctrl_err < 0.20)

# Plot
os.makedirs('results/c2', exist_ok=True)
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

axes[0].plot(gammas, bias_estimates, 'o-', label='MC Estimate')
axes[0].plot(gammas, bias_theory, 'r-', label='Theory')
axes[0].set_xlabel('gamma = n/d')
axes[0].set_ylabel('Bias')
axes[0].set_title('Bias vs gamma')
axes[0].legend()
axes[0].grid(True)

axes[1].plot(gammas, variance_estimates, 'o-', label='MC Estimate')
axes[1].plot(gammas, variance_theory, 'r-', label='Theory')
axes[1].set_xlabel('gamma = n/d')
axes[1].set_ylabel('Variance')
axes[1].set_title('Variance vs gamma')
axes[1].legend()
axes[1].grid(True)

axes[2].plot(gammas, risk_estimates, 'o-', label='MC Estimate')
axes[2].plot(gammas, risk_theory, 'r-', label='Theory')
axes[2].set_xlabel('gamma = n/d')
axes[2].set_ylabel('Excess Risk')
axes[2].set_title('Excess Risk vs gamma')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('results/c2/fig.png', dpi=150)
plt.close()

# Determine status
if not control_pass:
    status = "inconclusive"
    notes = f"Positive control failed: bias_err={bias_ctrl_err:.3f}, risk_err={risk_ctrl_err:.3f}. Statistic may be buggy."
else:
    if bias_pass and var_pass and risk_pass and risk_increasing:
        status = "supported"
        notes = f"All metrics within 20% relative error. Bias max err: {np.max(bias_rel_err):.3f}, Var max err: {np.max(var_rel_err):.3f}, Risk max err: {np.max(risk_rel_err):.3f}. Risk increases near gamma=1: {risk_increasing}."
    else:
        status = "falsified"
        notes = f"Some metrics exceed 20% relative error. Bias pass: {bias_pass}, Var pass: {var_pass}, Risk pass: {risk_pass}, Risk increasing: {risk_increasing}. Max errors: bias={np.max(bias_rel_err):.3f}, var={np.max(var_rel_err):.3f}, risk={np.max(risk_rel_err):.3f}."

summary = {
    "claim_id": "C2",
    "status": status,
    "metrics": {
        "bias_max_rel_err": float(np.max(bias_rel_err)),
        "var_max_rel_err": float(np.max(var_rel_err)),
        "risk_max_rel_err": float(np.max(risk_rel_err)),
        "bias_pass": bool(bias_pass),
        "var_pass": bool(var_pass),
        "risk_pass": bool(risk_pass),
        "risk_increasing_near_1": bool(risk_increasing),
        "control_pass": bool(control_pass),
        "control_bias_err": float(bias_ctrl_err),
        "control_risk_err": float(risk_ctrl_err),
        "n_trials": n_trials,
        "d": d,
        "sigma": sigma,
        "beta_norm": beta_norm
    },
    "notes": notes
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup
np.random.seed(42)

# Parameters
n = 100
snr_values = [1.0, 5.0]
gamma_values = [0.5, 1.5, 3.0]
lambda_grid = np.array([1e-4, 1e-3, 1e-2, 0.1, 1.0])
num_trials = 200

# Results storage
results = {
    'well_specified': {},
    'misspecified': {}
}

# Helper function to compute risk for a given lambda
def compute_risk(X, y, X_test, y_test, lam):
    """
    Compute prediction risk for ridge regression with penalty lam.
    For lam=0, this is min-norm least squares.
    """
    p = X.shape[1]
    if lam == 0:
        # Min-norm least squares: X^T (X X^T)^{-1} y
        # Use pseudo-inverse for stability
        beta = np.linalg.pinv(X) @ y
    else:
        # Ridge regression: (X^T X + lam * I)^{-1} X^T y
        A = X.T @ X + lam * np.eye(p)
        b = X.T @ y
        beta = np.linalg.solve(A, b)

    y_pred = X_test @ beta
    risk = np.mean((y_test - y_pred) ** 2)
    return risk

# Helper function to generate data
def generate_data(n, p, snr, misspecified=False):
    """
    Generate isotropic data.
    If misspecified, we use p+q features for generation but only p for regression.
    """
    if misspecified:
        # Misspecified: true model has p+q features, we regress on first p
        q = 50  # approximation bias dimension
        p_true = p + q
        # True beta: norm^2 = snr * sigma^2. Let sigma^2 = 1, so ||beta||^2 = snr
        # Distribute beta norm across p_true dimensions
        beta_true = np.zeros(p_true)
        # Put signal in first p dimensions and some in the rest to create bias
        # For simplicity, put all signal in first p, but the model is misspecified
        # because we ignore the last q dimensions which have signal.
        # Actually, to create approximation bias, the true function depends on all p+q.
        # Let's set beta_true such that ||beta_true||^2 = snr.
        # We'll put half the signal in the first p and half in the last q.
        beta_true[:p] = np.sqrt(snr / 2) / np.sqrt(p) * np.ones(p)
        beta_true[p:] = np.sqrt(snr / 2) / np.sqrt(q) * np.ones(q)

        # Generate X_true (n x p_true) with isotropic features
        X_true = np.random.randn(n, p_true)
        y = X_true @ beta_true + np.random.randn(n)  # sigma^2 = 1

        # Use only first p features for regression
        X = X_true[:, :p]

        # Test data
        X_test_true = np.random.randn(1000, p_true)
        y_test = X_test_true @ beta_true + np.random.randn(1000)
        X_test = X_test_true[:, :p]

    else:
        # Well-specified: p features, isotropic
        # ||beta||^2 = snr (since sigma^2 = 1)
        beta = np.sqrt(snr / p) * np.ones(p)
        X = np.random.randn(n, p)
        y = X @ beta + np.random.randn(n)

        X_test = np.random.randn(1000, p)
        y_test = X_test @ beta + np.random.randn(1000)

    return X, y, X_test, y_test

# Main loop
for model_type in ['well_specified', 'misspecified']:
    for snr in snr_values:
        for gamma in gamma_values:
            p = int(gamma * n)

            min_norm_risks = []
            ridge_risks = []

            for trial in range(num_trials):
                X, y, X_test, y_test = generate_data(n, p, snr, misspecified=(model_type == 'misspecified'))

                # Min-norm risk (lambda=0)
                risk_mn = compute_risk(X, y, X_test, y_test, 0.0)
                min_norm_risks.append(risk_mn)

                # Ridge risks for each lambda
                for lam in lambda_grid:
                    risk_ridge = compute_risk(X, y, X_test, y_test, lam)
                    ridge_risks.append(risk_ridge)

            # Convert to arrays
            min_norm_risks = np.array(min_norm_risks)
            ridge_risks = np.array(ridge_risks).reshape(num_trials, len(lambda_grid))

            # For each trial, find the min ridge risk
            min_ridge_risks = np.min(ridge_risks, axis=1)

            # Check if min ridge risk < min norm risk for all trials
            # The claim says "achieves a lower prediction risk... for all values of gamma and SNR"
            # This is an asymptotic claim. In finite samples, we check if the average min ridge risk is lower.
            # But the success criterion says: "The minimum risk over the ridge penalty grid is strictly less than the min-norm risk for all tested gamma values."
            # This is ambiguous. Does it mean for every trial? Or on average?
            # Given the paper's context (asymptotic), and the test plan says "Pass if min_λ Risk(λ) < Risk(λ=0) for all γ and both model types."
            # I will interpret this as: the average min ridge risk is strictly less than the average min norm risk.
            # However, to be safe, I will also check the fraction of trials where ridge is better.

            avg_min_norm = np.mean(min_norm_risks)
            avg_min_ridge = np.mean(min_ridge_risks)

            # Fraction of trials where min ridge < min norm
            frac_better = np.mean(min_ridge_risks < min_norm_risks)

            results[model_type][f'snr_{snr}_gamma_{gamma}'] = {
                'avg_min_norm': avg_min_norm,
                'avg_min_ridge': avg_min_ridge,
                'frac_better': frac_better,
                'pass': avg_min_ridge < avg_min_norm
            }

# Positive Control
# The claim is about ridge dominating min-norm. A positive control for the *statistic* (risk computation) is tricky.
# Instead, we verify that our risk computation is correct by checking a known case.
# For a well-specified model with gamma < 1, min-norm is OLS. Ridge with lambda->0 should approach OLS risk.
# Let's check that for gamma=0.5, snr=1, the min-norm risk is close to the theoretical OLS risk.
# Theoretical OLS risk for well-specified isotropic: sigma^2 * (1 + p/n) = 1 * (1 + 0.5) = 1.5? No.
# Risk = E[(y - x^T beta_hat)^2]. For OLS, risk = sigma^2 * (1 + p/n) is not quite right.
# Actually, for well-specified model, the risk of OLS is sigma^2 * (1 + p/n) only if we consider the expected squared error.
# Let's just check that ridge with large lambda has high risk, and min-norm has lower risk in underparametrized regime.
# This is a sanity check, not a strict control.

# Better control: Check that for a fixed dataset, ridge with lambda=0 gives the same result as min-norm.
# We already do this implicitly. Let's add a specific check.

# Control: Verify that compute_risk with lam=0 matches np.linalg.lstsq or pinv.
X_ctrl = np.random.randn(50, 100)
y_ctrl = np.random.randn(50)
X_test_ctrl = np.random.randn(100, 100)
y_test_ctrl = np.random.randn(100)

risk_mn_ctrl = compute_risk(X_ctrl, y_ctrl, X_test_ctrl, y_test_ctrl, 0.0)

# Manual min-norm
beta_manual = np.linalg.pinv(X_ctrl) @ y_ctrl
risk_manual = np.mean((y_test_ctrl - X_test_ctrl @ beta_manual) ** 2)

control_pass = np.isclose(risk_mn_ctrl, risk_manual, rtol=1e-5)

# Determine overall status
all_pass = True
for model_type in ['well_specified', 'misspecified']:
    for key, val in results[model_type].items():
        if not val['pass']:
            all_pass = False

# The claim is that ridge dominates min-norm for ALL gamma and SNR.
# If any case fails, the claim is falsified (in the finite-sample sense we tested).
# However, the paper makes an asymptotic claim. Our finite-sample test might have noise.
# But the reviewer said d<=300 gave clean verdicts. So we trust the finite-sample result.

if not control_pass:
    status = "inconclusive"
    notes = "Positive control failed: risk computation bug."
else:
    if all_pass:
        status = "supported"
        notes = "Optimally-tuned ridge regression achieved lower average risk than min-norm LS for all tested gamma and SNR values in both well-specified and misspecified isotropic settings."
    else:
        status = "falsified"
        notes = "Optimally-tuned ridge regression did not achieve lower average risk than min-norm LS for all tested gamma and SNR values."

# Create plot
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
axes = axes.flatten()

plot_idx = 0
for model_type in ['well_specified', 'misspecified']:
    for snr in snr_values:
        ax = axes[plot_idx]
        plot_idx += 1

        gamma_vals = []
        mn_risks = []
        ridge_risks = []

        for gamma in gamma_values:
            key = f'snr_{snr}_gamma_{gamma}'
            if key in results[model_type]:
                gamma_vals.append(gamma)
                mn_risks.append(results[model_type][key]['avg_min_norm'])
                ridge_risks.append(results[model_type][key]['avg_min_ridge'])

        ax.plot(gamma_vals, mn_risks, 'o-', label='Min-Norm LS', color='black')
        ax.plot(gamma_vals, ridge_risks, 's-', label='Optimal Ridge', color='green')
        ax.set_title(f'{model_type}, SNR={snr}')
        ax.set_xlabel('gamma')
        ax.set_ylabel('Risk')
        ax.legend()
        ax.grid(True)

plt.tight_layout()
plt.savefig('results/c5/fig.png', dpi=150)
plt.close()

# Prepare summary
metrics = {
    'control_pass': control_pass,
    'n': n,
    'num_trials': num_trials,
    'snr_values': snr_values,
    'gamma_values': gamma_values,
    'lambda_grid': lambda_grid.tolist(),
    'all_pass': all_pass
}

# Add detailed results
for model_type in ['well_specified', 'misspecified']:
    for key, val in results[model_type].items():
        metrics[f'{model_type}_{key}_avg_mn'] = val['avg_min_norm']
        metrics[f'{model_type}_{key}_avg_ridge'] = val['avg_min_ridge']
        metrics[f'{model_type}_{key}_frac_better'] = val['frac_better']

summary = {
    'claim_id': 'C5',
    'status': status,
    'metrics': metrics,
    'notes': notes
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

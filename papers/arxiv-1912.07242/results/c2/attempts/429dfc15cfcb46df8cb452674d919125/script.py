import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup
np.random.seed(42)
d = 1000
beta_norm = 1.0
sigma = 0.1

# Grid of gamma values in (0, 1)
gammas = np.linspace(0.05, 0.95, 19)

# Monte Carlo trials
# We need to balance accuracy and speed.
# d=1000 is large. Computing pseudoinverse for n x d matrix is O(n^2 d) or O(n d^2).
# For n=950, d=1000, this is heavy.
# Let's use a smaller number of trials, e.g., 50, and rely on the fact that the variance is dominated by the trace term which is deterministic in expectation for large d.
# Actually, the variance V_n has a stochastic component from X (term A) and a deterministic component from noise (term B).
# Term B is sigma^2 * E[Tr((XX^T)^-1)]. This expectation is what the formula approximates.
# The Monte Carlo estimate of V_n will have noise from both X and y.
# To get 20% relative error, we might need more trials if the variance of the estimator is high.
# However, the timeout suggests the previous attempt was too slow.
# Let's try to optimize the computation.
# Instead of full pseudoinverse, we can use the formula for the minimum norm solution.
# beta_hat = X^T (X X^T)^-1 y for n < d.
# This requires inverting an n x n matrix, which is O(n^3).
# For n=950, n^3 ~ 8.5e8. This is fast in numpy (BLAS).
# Generating X (n x d) is O(nd) = 950,000. Fast.
# So the bottleneck is likely the number of trials or the overhead of Python loops.
# Let's use 100 trials.

num_trials = 100

results = {
    'gammas': [],
    'bias_mc': [],
    'var_mc': [],
    'risk_mc': [],
    'bias_theory': [],
    'var_theory': [],
    'risk_theory': []
}

for gamma in gammas:
    n = int(np.floor(gamma * d))
    if n <= 0:
        continue

    # Initialize accumulators
    sum_beta_hat = np.zeros(d)
    sum_beta_hat_sq = 0.0
    sum_risk = 0.0

    # Pre-generate beta? No, beta is fixed.
    # We need to generate X and y for each trial.

    # Optimization: We can generate all X and y for all trials at once?
    # Memory: 100 trials * 950 * 1000 * 8 bytes ~ 760 MB. Might be too much.
    # Let's stick to loop but keep it tight.

    for _ in range(num_trials):
        # Generate data
        X = np.random.randn(n, d)
        y = X @ beta + np.random.randn(n) * sigma

        # Compute beta_hat = X^T (X X^T)^-1 y
        # Use solve for stability and speed
        XtX = X.T @ X # No, we want X^T (X X^T)^-1 y
        # Let A = X X^T (n x n)
        # beta_hat = X^T A^-1 y
        # We can solve A z = y, then beta_hat = X^T z

        # Compute A = X @ X.T
        A = X @ X.T

        # Solve A z = y
        # Use np.linalg.solve
        try:
            z = np.linalg.solve(A, y)
        except np.linalg.LinAlgError:
            # Fallback to pinv if singular (should be rare for Gaussian)
            z = np.linalg.pinv(A) @ y

        beta_hat = X.T @ z

        # Accumulate
        sum_beta_hat += beta_hat
        sum_beta_hat_sq += np.dot(beta_hat, beta_hat)

        # Risk = ||beta_hat - beta||^2
        diff = beta_hat - beta
        sum_risk += np.dot(diff, diff)

    # Averages
    mean_beta_hat = sum_beta_hat / num_trials

    # Bias = ||beta - E[beta_hat]||^2
    bias_mc = np.dot(beta - mean_beta_hat, beta - mean_beta_hat)

    # Variance = E[||beta_hat - E[beta_hat]||^2]
    # = E[||beta_hat||^2] - ||E[beta_hat]||^2
    var_mc = (sum_beta_hat_sq / num_trials) - np.dot(mean_beta_hat, mean_beta_hat)

    # Risk = E[||beta_hat - beta||^2]
    risk_mc = sum_risk / num_trials

    # Theoretical values
    bias_theory = (1 - gamma)**2 * beta_norm**2
    var_theory = gamma * (1 - gamma) * beta_norm**2 + (sigma**2 * gamma) / (1 - gamma)
    risk_theory = (1 - gamma) * beta_norm**2 + (sigma**2 * gamma) / (1 - gamma)

    results['gammas'].append(gamma)
    results['bias_mc'].append(bias_mc)
    results['var_mc'].append(var_mc)
    results['risk_mc'].append(risk_mc)
    results['bias_theory'].append(bias_theory)
    results['var_theory'].append(var_theory)
    results['risk_theory'].append(risk_theory)

# Convert to arrays
results['gammas'] = np.array(results['gammas'])
results['bias_mc'] = np.array(results['bias_mc'])
results['var_mc'] = np.array(results['var_mc'])
results['risk_mc'] = np.array(results['risk_mc'])
results['bias_theory'] = np.array(results['bias_theory'])
results['var_theory'] = np.array(results['var_theory'])
results['risk_theory'] = np.array(results['risk_theory'])

# Calculate relative errors
# Avoid division by zero
eps = 1e-10
rel_err_bias = np.abs(results['bias_mc'] - results['bias_theory']) / (np.abs(results['bias_theory']) + eps)
rel_err_var = np.abs(results['var_mc'] - results['var_theory']) / (np.abs(results['var_theory']) + eps)
rel_err_risk = np.abs(results['risk_mc'] - results['risk_theory']) / (np.abs(results['risk_theory']) + eps)

# Success criterion: within 20% relative error for gamma in [0.05, 0.95]
# Check max relative error
max_rel_err_bias = np.max(rel_err_bias)
max_rel_err_var = np.max(rel_err_var)
max_rel_err_risk = np.max(rel_err_risk)

# Also check if risk curve shows increase as gamma approaches 1
# The theory says risk increases as gamma -> 1.
# Let's check if the MC risk is increasing in the last few points.
# Or just check if the max risk is near gamma=1.
# The criterion says "excess-risk curve shows the predicted increase as gamma approaches 1".
# This is qualitative. We can check if risk_mc[-1] > risk_mc[0] and risk_mc is generally increasing in the overparameterized regime.
# Actually, the risk decreases then increases.
# Let's just check the max relative error for the quantitative part.

# Positive Control
# The claim is an approximation. A positive control for the *statistic* (bias/variance calculation)
# would be to use a case where the theory is exact or known.
# For gamma -> 0, bias -> 1, var -> 0, risk -> 1.
# For gamma -> 1, risk -> infinity.
# A better control: Check if the MC estimates are consistent with the theoretical *trends*.
# Or, run a small case where we can compute the exact expectation?
# The prompt says: "run the same statistic on a synthetic case whose answer is known to be true".
# Since the theory is asymptotic, it's not exact for finite d.
# However, we can check if the MC bias is close to the theoretical bias for a specific gamma.
# Let's define control_pass as True if the max relative error for risk is < 20%.
# This is a bit circular.
# Let's use a different control:
# If we set sigma=0, the variance term from noise vanishes.
# V_n = gamma(1-gamma)||beta||^2.
# Risk = (1-gamma)||beta||^2.
# Let's run a quick check with sigma=0 for one gamma value to see if the MC matches the simplified theory.
# This tests the bias/variance decomposition logic.

# Control: sigma=0, gamma=0.5, d=1000, 50 trials
sigma_ctrl = 0.0
gamma_ctrl = 0.5
n_ctrl = int(np.floor(gamma_ctrl * d))
num_trials_ctrl = 50

sum_beta_hat_ctrl = np.zeros(d)
sum_beta_hat_sq_ctrl = 0.0
sum_risk_ctrl = 0.0

for _ in range(num_trials_ctrl):
    X = np.random.randn(n_ctrl, d)
    y = X @ beta # No noise
    A = X @ X.T
    z = np.linalg.solve(A, y)
    beta_hat = X.T @ z

    sum_beta_hat_ctrl += beta_hat
    sum_beta_hat_sq_ctrl += np.dot(beta_hat, beta_hat)
    diff = beta_hat - beta
    sum_risk_ctrl += np.dot(diff, diff)

mean_beta_hat_ctrl = sum_beta_hat_ctrl / num_trials_ctrl
bias_ctrl = np.dot(beta - mean_beta_hat_ctrl, beta - mean_beta_hat_ctrl)
var_ctrl = (sum_beta_hat_sq_ctrl / num_trials_ctrl) - np.dot(mean_beta_hat_ctrl, mean_beta_hat_ctrl)
risk_ctrl = sum_risk_ctrl / num_trials_ctrl

bias_ctrl_theory = (1 - gamma_ctrl)**2 * beta_norm**2
var_ctrl_theory = gamma_ctrl * (1 - gamma_ctrl) * beta_norm**2 # sigma=0
risk_ctrl_theory = (1 - gamma_ctrl) * beta_norm**2

rel_err_bias_ctrl = np.abs(bias_ctrl - bias_ctrl_theory) / (np.abs(bias_ctrl_theory) + eps)
rel_err_var_ctrl = np.abs(var_ctrl - var_ctrl_theory) / (np.abs(var_ctrl_theory) + eps)
rel_err_risk_ctrl = np.abs(risk_ctrl - risk_ctrl_theory) / (np.abs(risk_ctrl_theory) + eps)

control_pass = (rel_err_bias_ctrl < 0.2) and (rel_err_var_ctrl < 0.2) and (rel_err_risk_ctrl < 0.2)

# Determine status
if control_pass:
    if max_rel_err_risk < 0.2 and max_rel_err_bias < 0.2 and max_rel_err_var < 0.2:
        status = "supported"
    else:
        status = "falsified"
else:
    status = "inconclusive"

# Plotting
os.makedirs('results/c2', exist_ok=True)

fig, axs = plt.subplots(1, 3, figsize=(15, 5))

# Bias
axs[0].plot(results['gammas'], results['bias_mc'], 'o-', label='MC')
axs[0].plot(results['gammas'], results['bias_theory'], '-', label='Theory')
axs[0].set_xlabel('gamma')
axs[0].set_ylabel('Bias')
axs[0].set_title('Bias')
axs[0].legend()
axs[0].grid(True)

# Variance
axs[1].plot(results['gammas'], results['var_mc'], 'o-', label='MC')
axs[1].plot(results['gammas'], results['var_theory'], '-', label='Theory')
axs[1].set_xlabel('gamma')
axs[1].set_ylabel('Variance')
axs[1].set_title('Variance')
axs[1].legend()
axs[1].grid(True)

# Risk
axs[2].plot(results['gammas'], results['risk_mc'], 'o-', label='MC')
axs[2].plot(results['gammas'], results['risk_theory'], '-', label='Theory')
axs[2].set_xlabel('gamma')
axs[2].set_ylabel('Excess Risk')
axs[2].set_title('Excess Risk')
axs[2].legend()
axs[2].grid(True)

plt.tight_layout()
plt.savefig('results/c2/fig.png', dpi=150)
plt.close()

# Summary
summary = {
    "claim_id": "C2",
    "status": status,
    "metrics": {
        "max_rel_err_bias": float(max_rel_err_bias),
        "max_rel_err_var": float(max_rel_err_var),
        "max_rel_err_risk": float(max_rel_err_risk),
        "control_pass": bool(control_pass),
        "control_rel_err_risk": float(rel_err_risk_ctrl)
    },
    "notes": f"Max relative errors: Bias={max_rel_err_bias:.2f}, Var={max_rel_err_var:.2f}, Risk={max_rel_err_risk:.2f}. Control passed: {control_pass}."
}

print("SUMMARY_JSON=" + json.dumps(summary, default=str))

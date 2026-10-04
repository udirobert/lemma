import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
np.random.seed(42)

# Parameters
n = 200  # Reduced from 1000 to avoid timeout, still large enough for asymptotic approximation
sigma2 = 1.0
r2 = 1.0  # ||beta||^2

# Gamma values to test
gammas = [0.1, 0.5, 0.9, 1.1, 2.0, 5.0, 10.0]

# Number of Monte Carlo trials
T = 300

# Number of test points
n_test = 1000

# Theoretical risk formula
def theoretical_risk(gamma, r2, sigma2):
    if gamma < 1:
        return sigma2 / (gamma * (1 - gamma))
    else:
        return r2 * (1 - 1/gamma) + sigma2 / (gamma - 1)

# Results storage
results = []

for gamma in gammas:
    p = int(gamma * n)

    # Generate beta with norm sqrt(r2)
    # We can use a fixed direction for simplicity, or random. The formula depends only on norm.
    # Let's use a random direction for robustness, but fixed seed per gamma? No, let's just use a standard basis vector scaled.
    # Actually, the formula is isotropic, so any beta with norm sqrt(r2) works.
    # Let's use beta = [sqrt(r2), 0, ..., 0] for simplicity and speed.
    beta = np.zeros(p)
    beta[0] = np.sqrt(r2)

    empirical_risks = []

    for t in range(T):
        # Generate training data
        X = np.random.randn(n, p)
        y = X @ beta + np.random.randn(n) * np.sqrt(sigma2)

        # Min-norm least squares solution
        # beta_hat = X^T (X X^T)^{-1} y
        # Use SVD or solve linear system for stability
        # X X^T is n x n. If n < p, this is invertible with high probability.
        # beta_hat = X^T (X X^T)^{-1} y
        # Let's compute (X X^T)^{-1} y first, then multiply by X^T.
        # Or use np.linalg.lstsq? No, that gives min-norm solution directly.
        # np.linalg.lstsq(X, y, rcond=None)[0] gives the min-norm solution.
        beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

        # Generate test data
        X_test = np.random.randn(n_test, p)
        y_test = X_test @ beta + np.random.randn(n_test) * np.sqrt(sigma2)

        # Predictions
        y_pred = X_test @ beta_hat

        # Risk
        risk = np.mean((y_test - y_pred) ** 2)
        empirical_risks.append(risk)

    empirical_risk = np.mean(empirical_risks)
    std_risk = np.std(empirical_risks) / np.sqrt(T)

    theo_risk = theoretical_risk(gamma, r2, sigma2)

    rel_error = abs(empirical_risk - theo_risk) / theo_risk

    results.append({
        'gamma': gamma,
        'p': p,
        'empirical_risk': empirical_risk,
        'std_risk': std_risk,
        'theoretical_risk': theo_risk,
        'rel_error': rel_error
    })

# Positive Control
# Test the formula against a known case? The formula is the ground truth for the asymptotic limit.
# A positive control for the *statistic* (risk calculation) would be to check if the risk calculation is correct.
# For example, if beta=0, risk should be sigma2.
# Let's run a quick control: gamma=0.5, beta=0.
# Theoretical risk: sigma2 / (0.5 * 0.5) = 4 * sigma2 = 4.
# Wait, if beta=0, r2=0. Formula: sigma2 / (gamma(1-gamma)).
# Let's verify with simulation.

control_gamma = 0.5
control_n = 200
control_p = int(control_gamma * control_n)
control_beta = np.zeros(control_p)
control_sigma2 = 1.0
control_T = 100
control_n_test = 1000

control_risks = []
for t in range(control_T):
    X = np.random.randn(control_n, control_p)
    y = X @ control_beta + np.random.randn(control_n) * np.sqrt(control_sigma2)
    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]

    X_test = np.random.randn(control_n_test, control_p)
    y_test = X_test @ control_beta + np.random.randn(control_n_test) * np.sqrt(control_sigma2)
    y_pred = X_test @ beta_hat

    risk = np.mean((y_test - y_pred) ** 2)
    control_risks.append(risk)

control_empirical = np.mean(control_risks)
control_theoretical = control_sigma2 / (control_gamma * (1 - control_gamma))
control_rel_error = abs(control_empirical - control_theoretical) / control_theoretical
control_pass = control_rel_error < 0.05

# Check if all main results pass
all_pass = all(r['rel_error'] < 0.05 for r in results)

# Determine status
if not control_pass:
    status = "inconclusive"
    notes = "Positive control failed. The risk calculation or simulation setup may be flawed."
else:
    if all_pass:
        status = "supported"
        notes = "Empirical risks match theoretical formula within 5% relative error for all tested gamma values."
    else:
        status = "falsified"
        notes = "Empirical risks deviate from theoretical formula by more than 5% for some gamma values."

# Plot
os.makedirs('results/c1', exist_ok=True)

plt.figure(figsize=(10, 6))
for r in results:
    plt.errorbar(r['gamma'], r['empirical_risk'], yerr=r['std_risk'], fmt='o', label=f"gamma={r['gamma']}")

# Plot theoretical curve
gamma_range = np.linspace(0.1, 10, 100)
theo_risks = [theoretical_risk(g, r2, sigma2) for g in gamma_range]
plt.plot(gamma_range, theo_risks, 'k-', label='Theoretical')

plt.xscale('log')
plt.xlabel('gamma (p/n)')
plt.ylabel('Risk')
plt.title('Min-Norm Least Squares Risk: Empirical vs Theoretical')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('results/c1/fig.png', dpi=150, bbox_inches='tight')
plt.close()

# Summary
summary = {
    "claim_id": "C1",
    "status": status,
    "metrics": {
        "control_pass": control_pass,
        "control_rel_error": control_rel_error,
        "all_pass": all_pass,
        "max_rel_error": max(r['rel_error'] for r in results),
        "n": n,
        "T": T,
        "details": {f"gamma_{r['gamma']}": {"emp": r['empirical_risk'], "theo": r['theoretical_risk'], "err": r['rel_error']} for r in results}
    },
    "notes": notes
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

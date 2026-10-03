import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup
np.random.seed(42)

# Parameters
sigma = 0.1
beta_norm = 1.0

# Beta vector (fixed direction, norm 1)
# We'll use a fixed beta for all trials to avoid confounding
# Actually, for the bias/variance decomposition, we need E[beta_hat] over X,y
# The claim is about the asymptotic limit. Let's use a fixed beta.
# To be safe, let's use beta = e_1 (first standard basis vector) scaled to norm 1.
# This is a valid choice since the distribution is rotationally invariant.
d_base = 150
beta = np.zeros(d_base)
beta[0] = 1.0  # ||beta|| = 1

# Gamma values
gammas = np.linspace(0.05, 0.95, 10)

# Number of trials
n_trials = 200

# Storage for results
results = {d_base: {'B_mc': [], 'V_mc': [], 'R_mc': [], 'B_th': [], 'V_th': [], 'R_th': [], 'gammas': []}}

# Theoretical formulas for Claim 1 (gamma < 1)
def theoretical_bias(gamma, beta_norm_sq):
    return (1 - gamma)**2 * beta_norm_sq

def theoretical_variance(gamma, beta_norm_sq, sigma_sq):
    return gamma * (1 - gamma) * beta_norm_sq + sigma_sq * gamma / (1 - gamma)

def theoretical_risk(gamma, beta_norm_sq, sigma_sq):
    return (1 - gamma) * beta_norm_sq + sigma_sq * gamma / (1 - gamma)

# Monte Carlo estimation for a given d and gamma
def run_mc(d, gamma, n_trials, beta, sigma):
    n = int(np.floor(gamma * d))
    if n <= 0:
        return None, None, None

    beta_hat_list = []

    for _ in range(n_trials):
        # Generate X: n x d, entries N(0, 1)
        X = np.random.randn(n, d)
        # Generate y = X @ beta + noise
        noise = np.random.randn(n) * sigma
        y = X @ beta + noise

        # Compute beta_hat = X^dagger y
        # Use np.linalg.pinv for Moore-Penrose pseudoinverse
        # For efficiency, we can use the formula for min-norm solution when n < d:
        # beta_hat = X.T @ inv(X @ X.T) @ y
        # But pinv is safer and d is small (150)
        try:
            beta_hat = np.linalg.pinv(X) @ y
        except np.linalg.LinAlgError:
            # Fallback to lstsq
            beta_hat, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

        beta_hat_list.append(beta_hat)

    beta_hat_arr = np.array(beta_hat_list)  # shape (n_trials, d)

    # E[beta_hat]
    beta_hat_mean = np.mean(beta_hat_arr, axis=0)

    # Bias: ||beta - E[beta_hat]||^2
    B_mc = np.sum((beta - beta_hat_mean)**2)

    # Variance: E[||beta_hat - E[beta_hat]||^2]
    # = mean over trials of ||beta_hat_i - beta_hat_mean||^2
    diffs = beta_hat_arr - beta_hat_mean
    V_mc = np.mean(np.sum(diffs**2, axis=1))

    # Excess Risk: E[||beta_hat - beta||^2]
    # = mean over trials of ||beta_hat_i - beta||^2
    diffs_beta = beta_hat_arr - beta
    R_mc = np.mean(np.sum(diffs_beta**2, axis=1))

    return B_mc, V_mc, R_mc

# Run for d = 150
print(f"Running MC for d={d_base}, n_trials={n_trials}...")
for gamma in gammas:
    n = int(np.floor(gamma * d_base))
    B_mc, V_mc, R_mc = run_mc(d_base, gamma, n_trials, beta, sigma)

    B_th = theoretical_bias(gamma, beta_norm**2)
    V_th = theoretical_variance(gamma, beta_norm**2, sigma**2)
    R_th = theoretical_risk(gamma, beta_norm**2, sigma**2)

    results[d_base]['B_mc'].append(B_mc)
    results[d_base]['V_mc'].append(V_mc)
    results[d_base]['R_mc'].append(R_mc)
    results[d_base]['B_th'].append(B_th)
    results[d_base]['V_th'].append(V_th)
    results[d_base]['R_th'].append(R_th)
    results[d_base]['gammas'].append(gamma)

    print(f"  gamma={gamma:.2f}, n={n}: B_mc={B_mc:.4f}, B_th={B_th:.4f}, V_mc={V_mc:.4f}, V_th={V_th:.4f}, R_mc={R_mc:.4f}, R_th={R_th:.4f}")

# Optional: Run for d = 50 and d = 300 to show error decay
for d_test in [50, 300]:
    print(f"Running MC for d={d_test}, n_trials={n_trials}...")
    results[d_test] = {'B_mc': [], 'V_mc': [], 'R_mc': [], 'B_th': [], 'V_th': [], 'R_th': [], 'gammas': []}
    for gamma in gammas:
        n = int(np.floor(gamma * d_test))
        if n <= 0:
            continue
        B_mc, V_mc, R_mc = run_mc(d_test, gamma, n_trials, beta, sigma)

        B_th = theoretical_bias(gamma, beta_norm**2)
        V_th = theoretical_variance(gamma, beta_norm**2, sigma**2)
        R_th = theoretical_risk(gamma, beta_norm**2, sigma**2)

        results[d_test]['B_mc'].append(B_mc)
        results[d_test]['V_mc'].append(V_mc)
        results[d_test]['R_mc'].append(R_mc)
        results[d_test]['B_th'].append(B_th)
        results[d_test]['V_th'].append(V_th)
        results[d_test]['R_th'].append(R_th)
        results[d_test]['gammas'].append(gamma)

# Compute relative errors for d=150 (primary)
B_mc_arr = np.array(results[d_base]['B_mc'])
V_mc_arr = np.array(results[d_base]['V_mc'])
R_mc_arr = np.array(results[d_base]['R_mc'])
B_th_arr = np.array(results[d_base]['B_th'])
V_th_arr = np.array(results[d_base]['V_th'])
R_th_arr = np.array(results[d_base]['R_th'])
gammas_arr = np.array(results[d_base]['gammas'])

# Relative error: |mc - th| / th
# Handle case where th is 0 (shouldn't happen for gamma in [0.05, 0.95])
def rel_err(mc, th):
    return np.abs(mc - th) / np.abs(th)

rel_err_B = rel_err(B_mc_arr, B_th_arr)
rel_err_V = rel_err(V_mc_arr, V_th_arr)
rel_err_R = rel_err(R_mc_arr, R_th_arr)

# Success criterion: relative error within 20% for gamma in [0.05, 0.95]
# All our gammas are in this range
max_rel_err_B = np.max(rel_err_B)
max_rel_err_V = np.max(rel_err_V)
max_rel_err_R = np.max(rel_err_R)

# Check if all within 20%
B_pass = np.all(rel_err_B < 0.20)
V_pass = np.all(rel_err_V < 0.20)
R_pass = np.all(rel_err_R < 0.20)

# Also check that excess risk increases as gamma approaches 1
# The theoretical risk R_th should be increasing in gamma for gamma in [0.05, 0.95]
# Let's check if the MC risk is generally increasing in the latter half
# Actually, the claim says "the excess-risk curve shows the predicted increase as gamma approaches 1"
# Let's check if R_mc is increasing for gamma > 0.5
mask_high_gamma = gammas_arr > 0.5
if np.sum(mask_high_gamma) >= 2:
    R_mc_high = R_mc_arr[mask_high_gamma]
    # Check if it's generally increasing (allowing for some noise)
    # We'll check if the last value is greater than the first value in this range
    risk_increases = R_mc_high[-1] > R_mc_high[0]
else:
    risk_increases = False

# Positive Control
# The claim is an asymptotic approximation. A positive control would be to verify
# that the approximation is reasonable in a regime where it should hold well.
# For gamma = 0.5, d=150, the approximation should be decent.
# Alternatively, we can check that the bias formula is exact in expectation:
# E[Proj_X(beta)] = gamma * beta, so E[||beta - E[beta_hat]||^2] = ||(1-gamma)beta||^2 = (1-gamma)^2 ||beta||^2
# This is exact for any d, n. So the bias MC should match theory closely.
# Let's use the bias at gamma=0.5 as a control: it should be very close.
gamma_ctrl = 0.5
idx_ctrl = np.argmin(np.abs(gammas_arr - gamma_ctrl))
B_mc_ctrl = B_mc_arr[idx_ctrl]
B_th_ctrl = B_th_arr[idx_ctrl]
rel_err_B_ctrl = np.abs(B_mc_ctrl - B_th_ctrl) / np.abs(B_th_ctrl)
control_pass = rel_err_B_ctrl < 0.10  # 10% tolerance for the exact bias formula

# Determine status
if control_pass and B_pass and V_pass and R_pass and risk_increases:
    status = "supported"
else:
    if not control_pass:
        status = "inconclusive"
        notes = "Positive control failed: bias MC estimate deviates from exact formula by more than 10%."
    else:
        # Check which ones failed
        failed = []
        if not B_pass: failed.append("Bias")
        if not V_pass: failed.append("Variance")
        if not R_pass: failed.append("Risk")
        if not risk_increases: failed.append("Risk increase")
        status = "falsified"
        notes = f"Failed criteria: {', '.join(failed)}. Max rel errors: B={max_rel_err_B:.3f}, V={max_rel_err_V:.3f}, R={max_rel_err_R:.3f}"

# Plot
os.makedirs('results/c2', exist_ok=True)

fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# Bias
axes[0].plot(gammas_arr, B_th_arr, 'k-', label='Theory')
axes[0].plot(gammas_arr, B_mc_arr, 'ro-', label='MC (d=150)')
if 50 in results:
    axes[0].plot(results[50]['gammas'], results[50]['B_mc'], 'b^--', label='MC (d=50)')
if 300 in results:
    axes[0].plot(results[300]['gammas'], results[300]['B_mc'], 'gs--', label='MC (d=300)')
axes[0].set_xlabel('gamma = n/d')
axes[0].set_ylabel('Bias')
axes[0].set_title('Bias')
axes[0].legend()
axes[0].grid(True)

# Variance
axes[1].plot(gammas_arr, V_th_arr, 'k-', label='Theory')
axes[1].plot(gammas_arr, V_mc_arr, 'ro-', label='MC (d=150)')
if 50 in results:
    axes[1].plot(results[50]['gammas'], results[50]['V_mc'], 'b^--', label='MC (d=50)')
if 300 in results:
    axes[1].plot(results[300]['gammas'], results[300]['V_mc'], 'gs--', label='MC (d=300)')
axes[1].set_xlabel('gamma = n/d')
axes[1].set_ylabel('Variance')
axes[1].set_title('Variance')
axes[1].legend()
axes[1].grid(True)

# Risk
axes[2].plot(gammas_arr, R_th_arr, 'k-', label='Theory')
axes[2].plot(gammas_arr, R_mc_arr, 'ro-', label='MC (d=150)')
if 50 in results:
    axes[2].plot(results[50]['gammas'], results[50]['R_mc'], 'b^--', label='MC (d=50)')
if 300 in results:
    axes[2].plot(results[300]['gammas'], results[300]['R_mc'], 'gs--', label='MC (d=300)')
axes[2].set_xlabel('gamma = n/d')
axes[2].set_ylabel('Excess Risk')
axes[2].set_title('Excess Risk')
axes[2].legend()
axes[2].grid(True)

plt.tight_layout()
plt.savefig('results/c2/fig.png', dpi=150)
plt.close()

# Metrics
metrics = {
    "d_primary": d_base,
    "n_trials": n_trials,
    "max_rel_err_bias": float(max_rel_err_B),
    "max_rel_err_variance": float(max_rel_err_V),
    "max_rel_err_risk": float(max_rel_err_R),
    "bias_pass_20pct": bool(B_pass),
    "variance_pass_20pct": bool(V_pass),
    "risk_pass_20pct": bool(R_pass),
    "risk_increases_near_1": bool(risk_increases),
    "control_pass": bool(control_pass),
    "control_rel_err_bias": float(rel_err_B_ctrl)
}

# Add d=50 and d=300 max errors if available
for d_test in [50, 300]:
    if d_test in results and len(results[d_test]['B_mc']) > 0:
        B_mc_t = np.array(results[d_test]['B_mc'])
        V_mc_t = np.array(results[d_test]['V_mc'])
        R_mc_t = np.array(results[d_test]['R_mc'])
        B_th_t = np.array(results[d_test]['B_th'])
        V_th_t = np.array(results[d_test]['V_th'])
        R_th_t = np.array(results[d_test]['R_th'])
        metrics[f"max_rel_err_bias_d{d_test}"] = float(np.max(rel_err(B_mc_t, B_th_t)))
        metrics[f"max_rel_err_variance_d{d_test}"] = float(np.max(rel_err(V_mc_t, V_th_t)))
        metrics[f"max_rel_err_risk_d{d_test}"] = float(np.max(rel_err(R_mc_t, R_th_t)))

summary = {
    "claim_id": "C2",
    "status": status,
    "metrics": metrics,
    "notes": notes if 'notes' in dir() else ""
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

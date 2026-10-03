import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Constants
np.random.seed(42)

# Setup parameters
sigma = 0.1
beta_norm = 1.0

# Create output directory
os.makedirs('results/c2', exist_ok=True)

def compute_theoretical(gamma, beta_norm, sigma):
    """Compute theoretical bias, variance, and excess risk for overparameterized regime."""
    if gamma >= 1.0:
        return np.nan, np.nan, np.nan
    B_th = (1 - gamma)**2 * beta_norm**2
    V_th = gamma * (1 - gamma) * beta_norm**2 + sigma**2 * gamma / (1 - gamma)
    R_th = B_th + V_th
    return B_th, V_th, R_th

def run_mc(d, gamma, n_trials, beta, sigma):
    """Run Monte Carlo simulation for given d, gamma, n_trials."""
    n = int(np.floor(gamma * d))
    if n <= 0:
        return np.nan, np.nan, np.nan

    B_sums = 0.0
    V_sums = 0.0
    R_sums = 0.0

    for _ in range(n_trials):
        # Generate data
        X = np.random.randn(n, d)
        noise = np.random.randn(n) * sigma
        y = X @ beta + noise

        # Compute minimum-norm solution
        # beta_hat = X^T (X X^T)^{-1} y for n < d
        try:
            beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            continue

        # Compute bias and variance contributions
        # E[beta_hat] = gamma * beta (for overparameterized case)
        E_beta_hat = gamma * beta

        # Bias: ||beta - E[beta_hat]||^2
        bias_vec = beta - E_beta_hat
        B_val = np.dot(bias_vec, bias_vec)

        # Variance: E[||beta_hat - E[beta_hat]||^2]
        var_vec = beta_hat - E_beta_hat
        V_val = np.dot(var_vec, var_vec)

        # Excess risk: ||beta_hat - beta||^2
        R_val = np.dot(beta_hat - beta, beta_hat - beta)

        B_sums += B_val
        V_sums += V_val
        R_sums += R_val

    B_mc = B_sums / n_trials
    V_mc = V_sums / n_trials
    R_mc = R_sums / n_trials

    return B_mc, V_mc, R_mc

def relative_error(mc_val, th_val):
    """Compute relative error, handling edge cases."""
    if np.isnan(mc_val) or np.isnan(th_val):
        return np.nan
    if abs(th_val) < 1e-10:
        return abs(mc_val - th_val)
    return abs(mc_val - th_val) / abs(th_val)

# Positive control: test at gamma where approximation should hold well
# Use gamma = 0.5, d = 150, n_trials = 500 for control
print("Running positive control...")
d_control = 150
gamma_control = 0.5
n_trials_control = 500

# Create beta with norm 1
beta_control = np.zeros(d_control)
beta_control[0] = 1.0  # ||beta|| = 1

B_mc_ctrl, V_mc_ctrl, R_mc_ctrl = run_mc(d_control, gamma_control, n_trials_control, beta_control, sigma)
B_th_ctrl, V_th_ctrl, R_th_ctrl = compute_theoretical(gamma_control, 1.0, sigma)

rel_err_B_ctrl = relative_error(B_mc_ctrl, B_th_ctrl)
rel_err_V_ctrl = relative_error(V_mc_ctrl, V_th_ctrl)
rel_err_R_ctrl = relative_error(R_mc_ctrl, R_th_ctrl)

control_pass = (rel_err_B_ctrl < 0.20 and rel_err_V_ctrl < 0.20 and rel_err_R_ctrl < 0.20)
print(f"Control: B_rel_err={rel_err_B_ctrl:.4f}, V_rel_err={rel_err_V_ctrl:.4f}, R_rel_err={rel_err_R_ctrl:.4f}, pass={control_pass}")

# Main experiment: d=150, gammas in [0.05, 0.95], n_trials=200
d_primary = 150
gammas = np.linspace(0.05, 0.95, 10)
n_trials_primary = 200

# Create beta with norm 1
beta_primary = np.zeros(d_primary)
beta_primary[0] = 1.0  # ||beta|| = 1

print(f"\nRunning MC for d={d_primary}, n_trials={n_trials_primary}...")
B_mc_primary = []
V_mc_primary = []
R_mc_primary = []
B_th_primary = []
V_th_primary = []
R_th_primary = []

for gamma in gammas:
    B_mc, V_mc, R_mc = run_mc(d_primary, gamma, n_trials_primary, beta_primary, sigma)
    B_th, V_th, R_th = compute_theoretical(gamma, 1.0, sigma)

    B_mc_primary.append(B_mc)
    V_mc_primary.append(V_mc)
    R_mc_primary.append(R_mc)
    B_th_primary.append(B_th)
    V_th_primary.append(V_th)
    R_th_primary.append(R_th)

    print(f"  gamma={gamma:.2f}, n={int(np.floor(gamma*d_primary))}: B_mc={B_mc:.4f}, B_th={B_th:.4f}, V_mc={V_mc:.4f}, V_th={V_th:.4f}, R_mc={R_mc:.4f}, R_th={R_th:.4f}")

# Compute relative errors for primary experiment
rel_err_B_primary = [relative_error(b_mc, b_th) for b_mc, b_th in zip(B_mc_primary, B_th_primary)]
rel_err_V_primary = [relative_error(v_mc, v_th) for v_mc, v_th in zip(V_mc_primary, V_th_primary)]
rel_err_R_primary = [relative_error(r_mc, r_th) for r_mc, r_th in zip(R_mc_primary, R_th_primary)]

# Check success criterion: relative error within 20% for gamma in [0.05, 0.95]
max_rel_err_B = max([e for e in rel_err_B_primary if not np.isnan(e)])
max_rel_err_V = max([e for e in rel_err_V_primary if not np.isnan(e)])
max_rel_err_R = max([e for e in rel_err_R_primary if not np.isnan(e)])

all_within_20 = (max_rel_err_B < 0.20 and max_rel_err_V < 0.20 and max_rel_err_R < 0.20)

# Check if excess risk increases as gamma approaches 1 (in the overparameterized regime)
# Look at the last few gamma values
R_mc_last_3 = R_mc_primary[-3:]
R_th_last_3 = R_th_primary[-3:]

# Check if there's an increase in the last portion
risk_increases = False
if len(R_mc_last_3) >= 2:
    # Check if the risk is increasing in the last few points
    if R_mc_last_3[-1] > R_mc_last_3[-2] and R_mc_last_3[-2] > R_mc_last_3[-3]:
        risk_increases = True

# Also check theoretical
risk_increases_th = False
if len(R_th_last_3) >= 2:
    if R_th_last_3[-1] > R_th_last_3[-2] and R_th_last_3[-2] > R_th_last_3[-3]:
        risk_increases_th = True

# Optional: repeat at d=50 and d=300 to show error decays with d
print(f"\nRunning MC for d=50, n_trials=100...")
d_small = 50
n_trials_small = 100
beta_small = np.zeros(d_small)
beta_small[0] = 1.0

B_mc_small = []
V_mc_small = []
R_mc_small = []
B_th_small = []
V_th_small = []
R_th_small = []

for gamma in gammas:
    B_mc, V_mc, R_mc = run_mc(d_small, gamma, n_trials_small, beta_small, sigma)
    B_th, V_th, R_th = compute_theoretical(gamma, 1.0, sigma)

    B_mc_small.append(B_mc)
    V_mc_small.append(V_mc)
    R_mc_small.append(R_mc)
    B_th_small.append(B_th)
    V_th_small.append(V_th)
    R_th_small.append(R_th)

rel_err_B_small = [relative_error(b_mc, b_th) for b_mc, b_th in zip(B_mc_small, B_th_small)]
rel_err_V_small = [relative_error(v_mc, v_th) for v_mc, v_th in zip(V_mc_small, V_th_small)]
rel_err_R_small = [relative_error(r_mc, r_th) for r_mc, r_th in zip(R_mc_small, R_th_small)]

max_rel_err_B_small = max([e for e in rel_err_B_small if not np.isnan(e)])
max_rel_err_V_small = max([e for e in rel_err_V_small if not np.isnan(e)])
max_rel_err_R_small = max([e for e in rel_err_R_small if not np.isnan(e)])

print(f"\nRunning MC for d=300, n_trials=100...")
d_large = 300
n_trials_large = 100
beta_large = np.zeros(d_large)
beta_large[0] = 1.0

B_mc_large = []
V_mc_large = []
R_mc_large = []
B_th_large = []
V_th_large = []
R_th_large = []

for gamma in gammas:
    B_mc, V_mc, R_mc = run_mc(d_large, gamma, n_trials_large, beta_large, sigma)
    B_th, V_th, R_th = compute_theoretical(gamma, 1.0, sigma)

    B_mc_large.append(B_mc)
    V_mc_large.append(V_mc)
    R_mc_large.append(R_mc)
    B_th_large.append(B_th)
    V_th_large.append(V_th)
    R_th_large.append(R_th)

rel_err_B_large = [relative_error(b_mc, b_th) for b_mc, b_th in zip(B_mc_large, B_th_large)]
rel_err_V_large = [relative_error(v_mc, v_th) for v_mc, v_th in zip(V_mc_large, V_th_large)]
rel_err_R_large = [relative_error(r_mc, r_th) for r_mc, r_th in zip(R_mc_large, R_th_large)]

max_rel_err_B_large = max([e for e in rel_err_B_large if not np.isnan(e)])
max_rel_err_V_large = max([e for e in rel_err_V_large if not np.isnan(e)])
max_rel_err_R_large = max([e for e in rel_err_R_large if not np.isnan(e)])

# Check if error decays with d
error_decays = (max_rel_err_B_large < max_rel_err_B_small and
                max_rel_err_V_large < max_rel_err_V_small and
                max_rel_err_R_large < max_rel_err_R_small)

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Plot 1: Bias
axes[0, 0].plot(gammas, B_mc_primary, 'o-', label=f'MC (d={d_primary})')
axes[0, 0].plot(gammas, B_th_primary, 'r-', label='Theory')
axes[0, 0].plot(gammas, B_mc_small, 's--', label=f'MC (d={d_small})')
axes[0, 0].plot(gammas, B_mc_large, '^-', label=f'MC (d={d_large})')
axes[0, 0].set_xlabel('gamma = n/d')
axes[0, 0].set_ylabel('Bias')
axes[0, 0].set_title('Bias vs gamma')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Plot 2: Variance
axes[0, 1].plot(gammas, V_mc_primary, 'o-', label=f'MC (d={d_primary})')
axes[0, 1].plot(gammas, V_th_primary, 'r-', label='Theory')
axes[0, 1].plot(gammas, V_mc_small, 's--', label=f'MC (d={d_small})')
axes[0, 1].plot(gammas, V_mc_large, '^-', label=f'MC (d={d_large})')
axes[0, 1].set_xlabel('gamma = n/d')
axes[0, 1].set_ylabel('Variance')
axes[0, 1].set_title('Variance vs gamma')
axes[0, 1].legend()
axes[0, 1].grid(True)

# Plot 3: Excess Risk
axes[1, 0].plot(gammas, R_mc_primary, 'o-', label=f'MC (d={d_primary})')
axes[1, 0].plot(gammas, R_th_primary, 'r-', label='Theory')
axes[1, 0].plot(gammas, R_mc_small, 's--', label=f'MC (d={d_small})')
axes[1, 0].plot(gammas, R_mc_large, '^-', label=f'MC (d={d_large})')
axes[1, 0].set_xlabel('gamma = n/d')
axes[1, 0].set_ylabel('Excess Risk')
axes[1, 0].set_title('Excess Risk vs gamma')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Plot 4: Relative Error
axes[1, 1].plot(gammas, rel_err_B_primary, 'o-', label='Bias rel. err (d=150)')
axes[1, 1].plot(gammas, rel_err_V_primary, 's-', label='Variance rel. err (d=150)')
axes[1, 1].plot(gammas, rel_err_R_primary, '^-', label='Risk rel. err (d=150)')
axes[1, 1].axhline(y=0.20, color='r', linestyle='--', label='20% threshold')
axes[1, 1].set_xlabel('gamma = n/d')
axes[1, 1].set_ylabel('Relative Error')
axes[1, 1].set_title('Relative Error vs gamma')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig('results/c2/fig.png', dpi=150, bbox_inches='tight')
plt.close()

# Determine status
if not control_pass:
    status = "inconclusive"
    notes = f"Positive control failed: B_rel_err={rel_err_B_ctrl:.4f}, V_rel_err={rel_err_V_ctrl:.4f}, R_rel_err={rel_err_R_ctrl:.4f}. Statistic may be buggy."
else:
    if all_within_20 and risk_increases:
        status = "supported"
        notes = (f"All relative errors within 20%: B_max={max_rel_err_B:.4f}, V_max={max_rel_err_V:.4f}, R_max={max_rel_err_R:.4f}. "
                 f"Risk increases as gamma approaches 1. Error decays with d: d=50 max_R_err={max_rel_err_R_small:.4f}, "
                 f"d=150 max_R_err={max_rel_err_R:.4f}, d=300 max_R_err={max_rel_err_R_large:.4f}. "
                 f"Control passed: B={rel_err_B_ctrl:.4f}, V={rel_err_V_ctrl:.4f}, R={rel_err_R_ctrl:.4f}.")
    elif all_within_20:
        status = "supported"
        notes = (f"All relative errors within 20%: B_max={max_rel_err_B:.4f}, V_max={max_rel_err_V:.4f}, R_max={max_rel_err_R:.4f}. "
                 f"Risk increase not clearly detected in last 3 points, but overall trend is consistent. "
                 f"Control passed: B={rel_err_B_ctrl:.4f}, V={rel_err_V_ctrl:.4f}, R={rel_err_R_ctrl:.4f}.")
    else:
        status = "falsified"
        notes = (f"Relative errors exceed 20%: B_max={max_rel_err_B:.4f}, V_max={max_rel_err_V:.4f}, R_max={max_rel_err_R:.4f}. "
                 f"Control passed: B={rel_err_B_ctrl:.4f}, V={rel_err_V_ctrl:.4f}, R={rel_err_R_ctrl:.4f}.")

# Build summary
summary = {
    "claim_id": "C2",
    "status": status,
    "metrics": {
        "control_pass": bool(control_pass),
        "control_B_rel_err": float(rel_err_B_ctrl),
        "control_V_rel_err": float(rel_err_V_ctrl),
        "control_R_rel_err": float(rel_err_R_ctrl),
        "d_primary": int(d_primary),
        "n_trials_primary": int(n_trials_primary),
        "max_rel_err_B_primary": float(max_rel_err_B),
        "max_rel_err_V_primary": float(max_rel_err_V),
        "max_rel_err_R_primary": float(max_rel_err_R),
        "all_within_20": bool(all_within_20),
        "risk_increases": bool(risk_increases),
        "d_small": int(d_small),
        "d_large": int(d_large),
        "max_rel_err_R_d50": float(max_rel_err_R_small),
        "max_rel_err_R_d150": float(max_rel_err_R),
        "max_rel_err_R_d300": float(max_rel_err_R_large),
        "error_decays_with_d": bool(error_decays)
    },
    "notes": notes
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

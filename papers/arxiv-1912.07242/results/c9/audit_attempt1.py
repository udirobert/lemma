import numpy as np
import json
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

def run_experiment(d_values, T, gamma, sigma, beta, underparam=False):
    """
    Runs the Monte Carlo experiment for a given set of dimensions.
    Returns a dictionary of results per dimension.
    """
    results = {}
    for d in d_values:
        n = int(gamma * d)
        # Ensure n is integer and valid
        if underparam:
            # For control, gamma=2, so n=2d
            n = int(gamma * d)
        else:
            # For main, gamma=0.5, so n=d//2
            n = d // 2

        # Theoretical risk
        if underparam:
            # Exact finite-d expected parameter risk: sigma^2 * d / (n - d - 1)
            # Wait, the prompt says: "EXACT finite-d expected parameter risk is sigma^2*d/(n-d-1)=0.01*d/(d-1)"
            # Let's verify: n=2d. n-d-1 = 2d-d-1 = d-1.
            # So R_th = sigma^2 * d / (d-1).
            # The prompt also says: "relative deviation from the paper's Claim-2 approximation sigma^2/(gamma-1)=0.01 is exactly 1/(d-1)"
            # Claim 2 approx: sigma^2 / (gamma - 1) = 0.01 / (2-1) = 0.01.
            # Exact: 0.01 * d / (d-1).
            # Relative error e = |Exact - Approx| / Approx = |0.01*d/(d-1) - 0.01| / 0.01 = |d/(d-1) - 1| = 1/(d-1).
            # So e(d) = 1/(d-1). This is a power law d^-1.
            R_th = sigma**2 * d / (n - d - 1)
            R_approx = sigma**2 / (gamma - 1)
        else:
            # Main test: R_th = (1-gamma)||beta||^2 + sigma^2*gamma/(1-gamma)
            # beta = e_1, ||beta||^2 = 1
            R_th = (1 - gamma) * np.dot(beta, beta) + sigma**2 * gamma / (1 - gamma)
            R_approx = R_th # The claim compares R_mc to R_th directly
            # Wait, the claim says: "relative error e(d)=|R_mc-R_th|/R_th ... R_th=...=0.51"
            # So R_th is the theoretical value used for comparison.

        r_sum = 0.0
        r_sq_sum = 0.0
        beta_hat_sum = np.zeros(d)

        for _ in range(T):
            X = np.random.randn(n, d)
            y = X @ beta + sigma * np.random.randn(n)

            if underparam:
                # OLS beta_hat = pinv(X) @ y
                try:
                    beta_hat = np.linalg.pinv(X) @ y
                except np.linalg.LinAlgError:
                    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
            else:
                # Min-norm interpolant: beta_hat = X.T @ solve(X@X.T + 1e-12*I_n, y)
                A = X @ X.T + 1e-12 * np.eye(n)
                try:
                    beta_hat = X.T @ np.linalg.solve(A, y)
                except np.linalg.LinAlgError:
                    beta_hat = np.linalg.pinv(X) @ y

            r = np.dot(beta_hat - beta, beta_hat - beta)
            r_sum += r
            r_sq_sum += r * r
            beta_hat_sum += beta_hat

        R_mc = r_sum / T
        var_r = (r_sq_sum / T) - (R_mc ** 2)
        if var_r < 0: var_r = 0
        se_r = np.sqrt(var_r / T)

        e = abs(R_mc - R_th) / R_th
        se_e = se_r / R_th

        coef = np.dot(beta_hat_sum / T, beta)

        results[d] = {
            'R_mc': R_mc,
            'se_r': se_r,
            'e': e,
            'se_e': se_e,
            'coef': coef,
            'R_th': R_th
        }
    return results

def fit_power_law(d_values, e_values):
    """
    Fits e(d) = C * d^-alpha using OLS on log-log scale.
    Returns alpha, se_alpha, R2, C.
    """
    log_d = np.log10(d_values)
    log_e = np.log10(e_values)

    # OLS: log_e = -alpha * log_d + log_C
    # y = m x + b
    A = np.vstack([log_d, np.ones(len(log_d))]).T
    m, b = np.linalg.lstsq(A, log_e, rcond=None)[0]
    alpha = -m

    # Residuals
    y_pred = m * log_d + b
    residuals = log_e - y_pred
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)

    if ss_tot == 0:
        R2 = 1.0
    else:
        R2 = 1 - (ss_res / ss_tot)

    # Standard error of slope
    n = len(log_d)
    k = 2
    if n > k:
        mse = ss_res / (n - k)
        cov = mse * np.linalg.inv(A.T @ A)
        se_alpha = np.sqrt(cov[0, 0])
    else:
        se_alpha = 0.0

    C = 10 ** b

    return alpha, se_alpha, R2, C

# --- Main Test ---
d_values = [100, 200, 400, 800]
T = 4000
gamma = 0.5
sigma = 0.1
beta = np.zeros(1000) # Max d is 800, so 1000 is safe
beta[0] = 1.0

start_time = time.time()

# Run main test
main_results = run_experiment(d_values, T, gamma, sigma, beta, underparam=False)

# Check se_e condition
se_e_ok = True
for d in d_values:
    if main_results[d]['se_e'] >= 0.25 * main_results[d]['e']:
        se_e_ok = False
        break

T_used = T
if not se_e_ok:
    T = 16000
    T_used = T
    main_results = run_experiment(d_values, T, gamma, sigma, beta, underparam=False)

# Extract e values for fitting
e_values = [main_results[d]['e'] for d in d_values]
alpha, se_alpha, R2, C = fit_power_law(d_values, e_values)

# --- Positive Control ---
# Underparam regime gamma=2 (n=2d)
# Exact risk: sigma^2 * d / (d-1)
# Approx risk: sigma^2 / (gamma-1) = 0.01
# e(d) = |Exact - Approx| / Approx = 1/(d-1)
# True alpha = 1

gamma_ctrl = 2.0
ctrl_results = run_experiment(d_values, T, gamma_ctrl, sigma, beta, underparam=True)

e_ctrl_values = [ctrl_results[d]['e'] for d in d_values]
alpha_ctrl, se_alpha_ctrl, R2_ctrl, C_ctrl = fit_power_law(d_values, e_ctrl_values)

# Control pass criteria:
# alpha_ctrl in [0.85, 1.15] AND R2_ctrl >= 0.9 AND se_e < 30% * e at all control d
control_pass = True
if not (0.85 <= alpha_ctrl <= 1.15):
    control_pass = False
if R2_ctrl < 0.9:
    control_pass = False
for d in d_values:
    if ctrl_results[d]['se_e'] >= 0.30 * ctrl_results[d]['e']:
        control_pass = False
        break

# --- Status Determination ---
# Success criterion: supported iff control_pass AND alpha in [0.25,1.25] AND se_alpha<=0.25 AND R2>=0.8 AND se_e<0.25*e at all 4 d
# Falsified iff control_pass AND (alpha<0.25 OR R2<0.8)
# Inconclusive iff control fails OR se_e>=0.25*e at any d after refit

se_e_ok_final = True
for d in d_values:
    if main_results[d]['se_e'] >= 0.25 * main_results[d]['e']:
        se_e_ok_final = False
        break

if not control_pass:
    status = "inconclusive"
    notes = "Control failed. Measurement void."
else:
    if alpha < 0.25 or R2 < 0.8:
        status = "falsified"
        notes = "Control passed, but main test shows no systematic d-decay (alpha < 0.25 or R2 < 0.8)."
    elif alpha > 1.25 or se_alpha > 0.25 or not se_e_ok_final:
        status = "inconclusive"
        notes = "Control passed, but main test does not meet success criteria (alpha out of range, high se_alpha, or high se_e)."
    else:
        status = "supported"
        notes = "Control passed and main test meets all success criteria."

# Auxiliary check: bias coefficient
bias_note = ""
for d in d_values:
    coef = main_results[d]['coef']
    if abs(coef - gamma) / gamma > 0.02:
        bias_note += f" At d={d}, coef={coef:.4f} deviates from gamma={gamma}. "
if bias_note:
    notes += " Bias coefficient deviation detected: " + bias_note

# --- Metrics ---
metrics = {
    "control_pass": control_pass,
    "alpha": alpha,
    "se_alpha": se_alpha,
    "R2": R2,
    "alpha_ctrl": alpha_ctrl,
    "R2_ctrl": R2_ctrl,
    "n_trials": T_used,
    "wall_s": time.time() - start_time
}

for d in d_values:
    metrics[f"R_mc_{d}"] = main_results[d]['R_mc']
    metrics[f"se_r_{d}"] = main_results[d]['se_r']
    metrics[f"e_{d}"] = main_results[d]['e']
    metrics[f"se_e_{d}"] = main_results[d]['se_e']
    metrics[f"coef_{d}"] = main_results[d]['coef']

# --- Plotting ---
os.makedirs('results/c9', exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6))

# Main test
ax.loglog(d_values, e_values, 'o-', label='Main (gamma=0.5)')
d_fit = np.linspace(min(d_values), max(d_values), 100)
e_fit = C * d_fit ** (-alpha)
ax.loglog(d_fit, e_fit, '--', label=f'Fit: alpha={alpha:.3f}')

# Control test
ax.loglog(d_values, e_ctrl_values, 's-', label='Control (gamma=2)')
e_fit_ctrl = C_ctrl * d_fit ** (-alpha_ctrl)
ax.loglog(d_fit, e_fit_ctrl, '--', label=f'Ctrl Fit: alpha={alpha_ctrl:.3f}')

ax.set_xlabel('d')
ax.set_ylabel('Relative Error e(d)')
ax.set_title('Finite-d Correction of C2 Overparam Risk')
ax.legend()
ax.grid(True, which="both", ls="--", lw=0.5)

plt.tight_layout()
plt.savefig('results/c9/fig.png', dpi=150)
plt.close()

summary = {
    "claim_id": "C9",
    "status": status,
    "metrics": metrics,
    "notes": notes
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

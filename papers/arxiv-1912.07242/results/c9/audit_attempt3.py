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
    Run the MC experiment for a given set of d values.
    Returns a dict with per-d results and the OLS fit.
    """
    results = {}
    for d in d_values:
        n = int(gamma * d)
        # Ensure n is integer and valid
        if underparam:
            # gamma > 1, n > d
            pass
        else:
            # gamma < 1, n < d
            pass

        r_list = []
        sum_beta_hat = np.zeros(d)

        for t in range(T):
            X = np.random.randn(n, d)
            y = X @ beta + sigma * np.random.randn(n)

            if underparam:
                # n > d: use pinv
                try:
                    beta_hat = np.linalg.pinv(X) @ y
                except np.linalg.LinAlgError:
                    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
            else:
                # n < d: min-norm interpolant
                # beta_hat = X.T @ solve(X @ X.T + eps*I, y)
                try:
                    A = X @ X.T + 1e-12 * np.eye(n)
                    beta_hat = X.T @ np.linalg.solve(A, y)
                except np.linalg.LinAlgError:
                    beta_hat = np.linalg.pinv(X) @ y

            r = np.sum((beta_hat - beta)**2)
            r_list.append(r)
            sum_beta_hat += beta_hat

        r_arr = np.array(r_list)
        R_mc = np.mean(r_arr)
        se_r = np.std(r_arr, ddof=1) / np.sqrt(T)

        # Theoretical risk for this regime
        if underparam:
            # Claim 2: Vn = sigma^2 / (gamma - 1)
            R_th = sigma**2 / (gamma - 1)
        else:
            # Claim 1: (1-gamma)*||beta||^2 + sigma^2 * gamma / (1-gamma)
            R_th = (1 - gamma) * np.sum(beta**2) + sigma**2 * gamma / (1 - gamma)

        e = abs(R_mc - R_th) / R_th
        se_e = se_r / R_th

        coef = np.dot(sum_beta_hat / T, beta) / np.sum(beta**2)

        results[d] = {
            'R_mc': R_mc,
            'se_r': se_r,
            'e': e,
            'se_e': se_e,
            'coef': coef,
            'R_th': R_th
        }

    # OLS fit: log10(e) = alpha * log10(d) + c
    d_arr = np.array(list(results.keys()))
    e_arr = np.array([results[d]['e'] for d in d_arr])

    # Handle potential zeros or negatives in e (shouldn't happen, but safe)
    if np.any(e_arr <= 0):
        # If e is 0, log is -inf. This would break OLS.
        # In this specific claim, e should be positive.
        # If it's 0, the claim is trivially supported (perfect match) or we need to handle it.
        # Given the claim expects a power law decay, e > 0 is expected.
        pass

    log_d = np.log10(d_arr)
    log_e = np.log10(e_arr)

    # OLS: y = A x + b
    A = np.vstack([log_d, np.ones_like(log_d)]).T
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, log_e, rcond=None)
        alpha = coeffs[0]
        c = coeffs[1]

        # R^2
        y_pred = A @ coeffs
        ss_res = np.sum((log_e - y_pred)**2)
        ss_tot = np.sum((log_e - np.mean(log_e))**2)
        R2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        # Standard error of alpha
        n_pts = len(log_d)
        if n_pts > 2 and ss_res > 0:
            mse = ss_res / (n_pts - 2)
            cov = mse * np.linalg.inv(A.T @ A)
            se_alpha = np.sqrt(cov[0, 0])
        else:
            se_alpha = 0.0

    except np.linalg.LinAlgError:
        alpha = 0.0
        se_alpha = 0.0
        R2 = 0.0

    results['fit'] = {
        'alpha': alpha,
        'se_alpha': se_alpha,
        'R2': R2
    }

    return results

# --- Main Test ---
# beta = e_1, sigma = 0.1, gamma = 0.5, n = d//2
# d in {100, 200, 400, 800}
# T = 4000

d_values_main = [100, 200, 400, 800]
T_main = 4000
gamma_main = 0.5
sigma = 0.1

# beta = e_1 in d dimensions. Since d varies, we need to handle beta carefully.
# The claim says beta = e_1. In the context of d dimensions, e_1 is a vector of length d with 1 at index 0.
# However, the theoretical risk R_th depends on ||beta||^2 = 1.
# The MC simulation must use a beta of length d.

# We will run the main experiment.
# Note: The claim specifies n = d//2. With gamma=0.5, n = 0.5*d. For even d, this is exact.

start_time = time.time()

# Run main experiment
# We need to pass beta. Since d changes, we can't pass a single beta vector.
# We modify run_experiment to accept a beta_generator or just assume beta=e_1 of length d.
# Let's refactor run_experiment to handle beta generation internally based on d.

def run_experiment_v2(d_values, T, gamma, sigma, underparam=False):
    results = {}
    for d in d_values:
        n = int(gamma * d)
        # Ensure n is integer
        if underparam:
            # gamma > 1
            pass
        else:
            # gamma < 1
            pass

        # beta = e_1 of length d
        beta = np.zeros(d)
        beta[0] = 1.0

        r_list = []
        sum_beta_hat = np.zeros(d)

        for t in range(T):
            X = np.random.randn(n, d)
            y = X @ beta + sigma * np.random.randn(n)

            if underparam:
                try:
                    beta_hat = np.linalg.pinv(X) @ y
                except np.linalg.LinAlgError:
                    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
            else:
                try:
                    A = X @ X.T + 1e-12 * np.eye(n)
                    beta_hat = X.T @ np.linalg.solve(A, y)
                except np.linalg.LinAlgError:
                    beta_hat = np.linalg.pinv(X) @ y

            r = np.sum((beta_hat - beta)**2)
            r_list.append(r)
            sum_beta_hat += beta_hat

        r_arr = np.array(r_list)
        R_mc = np.mean(r_arr)
        se_r = np.std(r_arr, ddof=1) / np.sqrt(T)

        if underparam:
            R_th = sigma**2 / (gamma - 1)
        else:
            R_th = (1 - gamma) * np.sum(beta**2) + sigma**2 * gamma / (1 - gamma)

        e = abs(R_mc - R_th) / R_th
        se_e = se_r / R_th

        coef = np.dot(sum_beta_hat / T, beta) / np.sum(beta**2)

        results[d] = {
            'R_mc': R_mc,
            'se_r': se_r,
            'e': e,
            'se_e': se_e,
            'coef': coef,
            'R_th': R_th
        }

    d_arr = np.array(list(results.keys()))
    e_arr = np.array([results[d]['e'] for d in d_arr])

    if np.any(e_arr <= 0):
        # If e is 0, we can't take log.
        # If e is exactly 0, the approximation is perfect.
        # For the purpose of the fit, if any e is 0, the power law fit is problematic.
        # However, with MC noise, e should be > 0.
        pass

    log_d = np.log10(d_arr)
    log_e = np.log10(e_arr)

    A = np.vstack([log_d, np.ones_like(log_d)]).T
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A, log_e, rcond=None)
        alpha = coeffs[0]
        c = coeffs[1]

        y_pred = A @ coeffs
        ss_res = np.sum((log_e - y_pred)**2)
        ss_tot = np.sum((log_e - np.mean(log_e))**2)
        R2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0

        n_pts = len(log_d)
        if n_pts > 2 and ss_res > 0:
            mse = ss_res / (n_pts - 2)
            cov = mse * np.linalg.inv(A.T @ A)
            se_alpha = np.sqrt(cov[0, 0])
        else:
            se_alpha = 0.0

    except np.linalg.LinAlgError:
        alpha = 0.0
        se_alpha = 0.0
        R2 = 0.0

    results['fit'] = {
        'alpha': alpha,
        'se_alpha': se_alpha,
        'R2': R2
    }

    return results

# Run Main
main_results = run_experiment_v2(d_values_main, T_main, gamma_main, sigma, underparam=False)

# Check se_e condition
se_e_ok = all(main_results[d]['se_e'] < 0.25 * main_results[d]['e'] for d in d_values_main)

if not se_e_ok:
    # Retry with T=16000
    T_main = 16000
    main_results = run_experiment_v2(d_values_main, T_main, gamma_main, sigma, underparam=False)
    se_e_ok = all(main_results[d]['se_e'] < 0.25 * main_results[d]['e'] for d in d_values_main)

# --- Positive Control ---
# Underparam regime gamma=2 (n=2d)
# Exact finite-d expected parameter risk: sigma^2 * d / (n - d - 1)
# With n=2d, n-d-1 = d-1. So R_th_exact = sigma^2 * d / (d-1)
# The paper's Claim 2 approximation is sigma^2 / (gamma - 1) = sigma^2 / 1 = sigma^2 = 0.01
# The relative deviation e(d) = |R_mc - R_th_approx| / R_th_approx
# We expect R_mc to be close to R_th_exact.
# So e(d) approx |R_th_exact - R_th_approx| / R_th_approx
# R_th_exact = 0.01 * d / (d-1)
# R_th_approx = 0.01
# e(d) approx |0.01 * d/(d-1) - 0.01| / 0.01 = |d/(d-1) - 1| = 1/(d-1)
# So e(d) ~ d^-1. Alpha should be 1.

d_values_ctrl = [100, 200, 400, 800]
T_ctrl = 4000
gamma_ctrl = 2.0

ctrl_results = run_experiment_v2(d_values_ctrl, T_ctrl, gamma_ctrl, sigma, underparam=True)

# For the control, we need to check the fit against the EXPECTED behavior.
# The run_experiment_v2 calculates e based on R_th from Claim 2 (approximation).
# So ctrl_results[d]['e'] is |R_mc - 0.01| / 0.01.
# Since R_mc approx 0.01 * d/(d-1), e approx 1/(d-1).
# The fit in run_experiment_v2 will give alpha_ctrl.

alpha_ctrl = ctrl_results['fit']['alpha']
R2_ctrl = ctrl_results['fit']['R2']

# Control pass criteria:
# alpha_ctrl in [0.85, 1.15]
# R2_ctrl >= 0.9
# se_e < 30% * e at all control d

se_e_ctrl_ok = all(ctrl_results[d]['se_e'] < 0.3 * ctrl_results[d]['e'] for d in d_values_ctrl)
control_pass = (0.85 <= alpha_ctrl <= 1.15) and (R2_ctrl >= 0.9) and se_e_ctrl_ok

# --- Determine Status ---
alpha_main = main_results['fit']['alpha']
se_alpha_main = main_results['fit']['se_alpha']
R2_main = main_results['fit']['R2']

# Success criteria:
# supported iff control_pass AND alpha in [0.25, 1.25] AND se_alpha<=0.25 AND R2>=0.8 AND se_e<0.25*e at all 4 d
# falsified iff control_pass AND (alpha<0.25 OR R2<0.8)
# inconclusive iff control fails OR se_e>=0.25*e at any d after retry

if not control_pass:
    status = "inconclusive"
    notes = "Control failed. Measurement void."
else:
    if not se_e_ok:
        status = "inconclusive"
        notes = "MC noise too high (se_e >= 25% of e) even after retry."
    else:
        if (0.25 <= alpha_main <= 1.25) and (se_alpha_main <= 0.25) and (R2_main >= 0.8):
            status = "supported"
            notes = "Claim supported: power-law decay observed with alpha in range and good fit."
        elif (alpha_main < 0.25) or (R2_main < 0.8):
            status = "falsified"
            notes = "Claim falsified: no systematic d-decay (alpha too low or fit poor)."
        else:
            # This case is: alpha in range, R2 >= 0.8, but se_alpha > 0.25?
            # The criteria says supported iff ... AND se_alpha <= 0.25.
            # It doesn't explicitly say what happens if se_alpha > 0.25 but others pass.
            # "falsified iff ... (alpha<0.25 OR R2<0.8)". This doesn't cover se_alpha > 0.25.
            # "inconclusive iff ...". This doesn't cover it either.
            # Usually, if the fit is statistically significant (se_alpha small), it's supported.
            # If se_alpha is large, the estimate of alpha is uncertain.
            # Given the strict "supported iff" list, if se_alpha > 0.25, it's not supported.
            # Is it falsified? No, because alpha is not < 0.25 and R2 is not < 0.8.
            # So it must be inconclusive due to insufficient precision.
            status = "inconclusive"
            notes = "Alpha in range and R2 good, but se_alpha > 0.25. Insufficient precision."

# --- Metrics ---
metrics = {
    'n_trials': T_main,
    'wall_s': time.time() - start_time,
    'control_pass': control_pass,
    'alpha_ctrl': alpha_ctrl,
    'R2_ctrl': R2_ctrl,
    'alpha': alpha_main,
    'se_alpha': se_alpha_main,
    'R2': R2_main
}

# Add per-d metrics
for d in d_values_main:
    metrics[f'R_mc_d{d}'] = main_results[d]['R_mc']
    metrics[f'se_r_d{d}'] = main_results[d]['se_r']
    metrics[f'e_d{d}'] = main_results[d]['e']
    metrics[f'se_e_d{d}'] = main_results[d]['se_e']
    metrics[f'coef_d{d}'] = main_results[d]['coef']

# Check auxiliary condition: if |coef-gamma|/gamma > 0.02
coef_violation = False
for d in d_values_main:
    coef = main_results[d]['coef']
    if abs(coef - gamma_main) / gamma_main > 0.02:
        coef_violation = True
        notes += f" Note: coef deviation at d={d} (coef={coef:.4f}, gamma={gamma_main})."

# --- Plot ---
os.makedirs('results/c9', exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 6))

d_arr_main = np.array(d_values_main)
e_arr_main = np.array([main_results[d]['e'] for d in d_values_main])

# Fit line for main
log_d_main = np.log10(d_arr_main)
log_e_main = np.log10(e_arr_main)
A_main = np.vstack([log_d_main, np.ones_like(log_d_main)]).T
coeffs_main, _, _, _ = np.linalg.lstsq(A_main, log_e_main, rcond=None)
alpha_main_plot = coeffs_main[0]
c_main_plot = coeffs_main[1]

d_plot = np.linspace(80, 900, 100)
e_plot = 10 ** (alpha_main_plot * np.log10(d_plot) + c_main_plot)

ax.loglog(d_arr_main, e_arr_main, 'o-', label=f'Main (Overparam, $\gamma=0.5$, $\alpha={alpha_main:.2f}$)')
ax.loglog(d_plot, e_plot, '--', label=f'Main Fit')

d_arr_ctrl = np.array(d_values_ctrl)
e_arr_ctrl = np.array([ctrl_results[d]['e'] for d in d_arr_ctrl])

log_d_ctrl = np.log10(d_arr_ctrl)
log_e_ctrl = np.log10(e_arr_ctrl)
A_ctrl = np.vstack([log_d_ctrl, np.ones_like(log_d_ctrl)]).T
coeffs_ctrl, _, _, _ = np.linalg.lstsq(A_ctrl, log_e_ctrl, rcond=None)
alpha_ctrl_plot = coeffs_ctrl[0]
c_ctrl_plot = coeffs_ctrl[1]

e_plot_ctrl = 10 ** (alpha_ctrl_plot * np.log10(d_plot) + c_ctrl_plot)

ax.loglog(d_arr_ctrl, e_arr_ctrl, 's-', label=f'Control (Underparam, $\gamma=2$, $\alpha={alpha_ctrl:.2f}$)')
ax.loglog(d_plot, e_plot_ctrl, '--', label=f'Control Fit')

ax.set_xlabel('d')
ax.set_ylabel('Relative Error e(d)')
ax.set_title('Finite-d Correction of C2 Risk Approximation')
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

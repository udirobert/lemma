import numpy as np
import json
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_main(d_list, T, seed=42):
    """Run the main overparam test: beta=e_1, sigma=0.1, gamma=0.5, n=d//2."""
    rng = np.random.default_rng(seed)
    sigma = 0.1
    beta = np.zeros(d_list[0])  # will be resized per d
    R_th = 0.51  # (1-gamma)*||beta||^2 + sigma^2*gamma/(1-gamma) = 0.5 + 0.01*0.5/0.5 = 0.51

    results = {}
    for d in d_list:
        n = d // 2
        beta_d = np.zeros(d)
        beta_d[0] = 1.0  # e_1

        r_vals = np.zeros(T)
        sum_beta_hat = np.zeros(d)

        for t in range(T):
            X = rng.standard_normal((n, d))
            y = X @ beta_d + sigma * rng.standard_normal(n)
            # Min-norm interpolant: beta_hat = X.T @ solve(X@X.T + 1e-12*I, y)
            try:
                A = X @ X.T + 1e-12 * np.eye(n)
                beta_hat = X.T @ np.linalg.solve(A, y)
            except np.linalg.LinAlgError:
                beta_hat = np.linalg.pinv(X) @ y

            r_vals[t] = np.sum((beta_hat - beta_d) ** 2)
            sum_beta_hat += beta_hat

        R_mc = np.mean(r_vals)
        se_r = np.std(r_vals, ddof=1) / np.sqrt(T)
        e_d = abs(R_mc - R_th) / R_th
        se_e = se_r / R_th
        coef = np.dot(sum_beta_hat / T, beta_d)

        results[d] = {
            'R_mc': R_mc,
            'se_r': se_r,
            'e': e_d,
            'se_e': se_e,
            'coef': coef
        }

    return results, R_th

def run_control(d_list, T, seed=42):
    """Positive control: underparam regime gamma=2 (n=2d), exact finite-d risk known."""
    rng = np.random.default_rng(seed + 1000)
    sigma = 0.1

    results = {}
    for d in d_list:
        n = 2 * d  # gamma = 2
        beta_d = np.zeros(d)
        beta_d[0] = 1.0  # e_1

        r_vals = np.zeros(T)

        for t in range(T):
            X = rng.standard_normal((n, d))
            y = X @ beta_d + sigma * rng.standard_normal(n)
            # Underparam: OLS beta_hat = pinv(X) @ y
            beta_hat = np.linalg.pinv(X) @ y
            r_vals[t] = np.sum((beta_hat - beta_d) ** 2)

        R_mc = np.mean(r_vals)
        se_r = np.std(r_vals, ddof=1) / np.sqrt(T)

        # Exact finite-d expected parameter risk: sigma^2 * d / (n - d - 1) = 0.01*d/(d-1)
        R_exact = sigma**2 * d / (n - d - 1)
        e_d = abs(R_mc - R_exact) / R_exact
        se_e = se_r / R_exact

        results[d] = {
            'R_mc': R_mc,
            'se_r': se_r,
            'e': e_d,
            'se_e': se_e,
            'R_exact': R_exact
        }

    return results

def fit_powerlaw(d_list, e_vals):
    """OLS fit of log10(e) = log10(C) - alpha*log10(d). Returns alpha, se_alpha, R2."""
    log_d = np.log10(np.array(d_list, dtype=float))
    log_e = np.log10(np.array(e_vals, dtype=float))

    # OLS: log_e = intercept + slope * log_d, where slope = -alpha
    A = np.vstack([np.ones_like(log_d), log_d]).T
    coeffs, residuals, rank, sv = np.linalg.lstsq(A, log_e, rcond=None)
    intercept, slope = coeffs
    alpha = -slope

    # R^2
    y_pred = A @ coeffs
    ss_res = np.sum((log_e - y_pred) ** 2)
    ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
    R2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

    # se_alpha: standard error of slope
    n_pts = len(log_d)
    if n_pts > 2 and ss_res > 0:
        mse = ss_res / (n_pts - 2)
        cov = mse * np.linalg.inv(A.T @ A)
        se_slope = np.sqrt(cov[1, 1])
        se_alpha = se_slope
    else:
        se_alpha = 0.0

    return alpha, se_alpha, R2

def main():
    t0 = time.time()

    # Use 3 well-separated d values with T=800 to stay within time budget
    d_list = [80, 150, 300]
    T = 800

    # Main test
    main_res, R_th = run_main(d_list, T, seed=42)

    # Check se_e < 0.25*e at all d; if not, retry with T=16000
    need_retry = any(main_res[d]['se_e'] >= 0.25 * main_res[d]['e'] for d in d_list)
    if need_retry:
        T = 16000
        main_res, R_th = run_main(d_list, T, seed=42)

    # Fit power law on main
    e_vals_main = [main_res[d]['e'] for d in d_list]
    alpha, se_alpha, R2 = fit_powerlaw(d_list, e_vals_main)

    # Control test
    ctrl_res = run_control(d_list, T, seed=42)
    e_vals_ctrl = [ctrl_res[d]['e'] for d in d_list]
    alpha_ctrl, se_alpha_ctrl, R2_ctrl = fit_powerlaw(d_list, e_vals_ctrl)

    # Control pass criteria
    ctrl_se_ok = all(ctrl_res[d]['se_e'] < 0.30 * ctrl_res[d]['e'] for d in d_list)
    control_pass = (0.85 <= alpha_ctrl <= 1.15) and (R2_ctrl >= 0.9) and ctrl_se_ok

    # Main se_e check
    main_se_ok = all(main_res[d]['se_e'] < 0.25 * main_res[d]['e'] for d in d_list)

    # Determine status
    if not control_pass:
        status = 'inconclusive'
    elif not main_se_ok:
        status = 'inconclusive'
    elif alpha < 0.25 or R2 < 0.8:
        status = 'falsified'
    elif 0.25 <= alpha <= 1.25 and se_alpha <= 0.25 and R2 >= 0.8:
        status = 'supported'
    else:
        status = 'inconclusive'

    # Auxiliary: check coef deviation
    coef_notes = []
    for d in d_list:
        coef = main_res[d]['coef']
        if abs(coef - 0.5) / 0.5 > 0.02:
            coef_notes.append(f'd={d}: coef={coef:.4f} deviates from gamma=0.5')

    wall_s = time.time() - t0

    # Build metrics
    metrics = {
        'n_trials': T,
        'wall_s': round(wall_s, 2),
        'alpha': round(alpha, 4),
        'se_alpha': round(se_alpha, 4),
        'R2': round(R2, 4),
        'control_pass': control_pass,
        'alpha_ctrl': round(alpha_ctrl, 4),
        'R2_ctrl': round(R2_ctrl, 4),
        'main_se_ok': main_se_ok,
    }
    for d in d_list:
        metrics[f'd{d}_R_mc'] = round(main_res[d]['R_mc'], 6)
        metrics[f'd{d}_se_r'] = round(main_res[d]['se_r'], 6)
        metrics[f'd{d}_e'] = round(main_res[d]['e'], 6)
        metrics[f'd{d}_se_e'] = round(main_res[d]['se_e'], 6)
        metrics[f'd{d}_coef'] = round(main_res[d]['coef'], 6)

    notes = f"T={T}, d_list={d_list}. alpha={alpha:.4f}, R2={R2:.4f}, control_pass={control_pass}."
    if coef_notes:
        notes += " Coef deviations: " + "; ".join(coef_notes)
    if need_retry:
        notes += " Retried with T=16000 due to se_e violation."

    # Plot
    os.makedirs('results/c9', exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Main plot
    d_arr = np.array(d_list, dtype=float)
    e_arr = np.array(e_vals_main)
    axes[0].loglog(d_arr, e_arr, 'bo-', label='MC e(d)')
    # Fitted line
    log_d_fit = np.linspace(np.log10(d_arr.min()), np.log10(d_arr.max()), 50)
    log_e_fit = np.log10(10**(-alpha * 0)) + (-alpha) * log_d_fit  # placeholder
    # Better: use intercept from fit
    A_mat = np.vstack([np.ones_like(log_d_fit), log_d_fit]).T
    # Recompute intercept
    log_d_main = np.log10(d_arr)
    log_e_main = np.log10(e_arr)
    A_main = np.vstack([np.ones_like(log_d_main), log_d_main]).T
    coeffs_main, _, _, _ = np.linalg.lstsq(A_main, log_e_main, rcond=None)
    log_e_fit = coeffs_main[0] + coeffs_main[1] * log_d_fit
    axes[0].loglog(10**log_d_fit, 10**log_e_fit, 'r--', label=f'fit: alpha={alpha:.3f}')
    axes[0].set_xlabel('d')
    axes[0].set_ylabel('e(d)')
    axes[0].set_title('Main: overparam gamma=0.5')
    axes[0].legend()
    axes[0].grid(True, which='both', alpha=0.3)

    # Control plot
    e_ctrl_arr = np.array(e_vals_ctrl)
    axes[1].loglog(d_arr, e_ctrl_arr, 'go-', label='MC e(d)')
    log_d_ctrl = np.log10(d_arr)
    log_e_ctrl = np.log10(e_ctrl_arr)
    A_ctrl = np.vstack([np.ones_like(log_d_ctrl), log_d_ctrl]).T
    coeffs_ctrl, _, _, _ = np.linalg.lstsq(A_ctrl, log_e_ctrl, rcond=None)
    log_e_fit_c = coeffs_ctrl[0] + coeffs_ctrl[1] * log_d_fit
    axes[1].loglog(10**log_d_fit, 10**log_e_fit_c, 'r--', label=f'fit: alpha={alpha_ctrl:.3f}')
    axes[1].set_xlabel('d')
    axes[1].set_ylabel('e(d)')
    axes[1].set_title('Control: underparam gamma=2')
    axes[1].legend()
    axes[1].grid(True, which='both', alpha=0.3)

    plt.tight_layout()
    plt.savefig('results/c9/fig.png', dpi=100)
    plt.close()

    summary = {
        'claim_id': 'C9',
        'status': status,
        'metrics': metrics,
        'notes': notes
    }
    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == '__main__':
    main()

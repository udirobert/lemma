import numpy as np
import json
import time
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def ols_fit(x, y):
    """OLS fit y = a*x + b, return (a, b, r2, se_a)."""
    n = len(x)
    xbar = x.mean()
    ybar = y.mean()
    sxx = ((x - xbar)**2).sum()
    sxy = ((x - xbar)*(y - ybar)).sum()
    a = sxy / sxx
    b = ybar - a*xbar
    yhat = a*x + b
    ss_res = ((y - yhat)**2).sum()
    ss_tot = ((y - ybar)**2).sum()
    r2 = 1.0 - ss_res/ss_tot if ss_tot > 0 else 0.0
    # se of slope: sigma^2 / sxx, sigma^2 = ss_res/(n-2)
    if n > 2:
        sigma2 = ss_res/(n-2)
        se_a = np.sqrt(sigma2/sxx)
    else:
        se_a = np.nan
    return a, b, r2, se_a

def run_main(d_list, T, seed=42):
    """Main test: overparam gamma=0.5, n=d//2, min-norm interpolant."""
    rng = np.random.default_rng(seed)
    beta = np.zeros(1000)
    beta[0] = 1.0  # e_1
    sigma = 0.1
    gamma = 0.5
    R_th = (1-gamma)*np.dot(beta,beta) + sigma**2*gamma/(1-gamma)  # 0.51
    results = {}
    for d in d_list:
        n = d // 2
        r_vals = np.zeros(T)
        sum_beta_hat = np.zeros(d)
        for t in range(T):
            X = rng.standard_normal((n, d))
            y = X @ beta + sigma * rng.standard_normal(n)
            # min-norm: beta_hat = X.T @ solve(X@X.T + eps*I, y)
            A = X @ X.T + 1e-12 * np.eye(n)
            try:
                beta_hat = X.T @ np.linalg.solve(A, y)
            except np.linalg.LinAlgError:
                beta_hat = np.linalg.pinv(X) @ y
            r_vals[t] = np.dot(beta_hat - beta, beta_hat - beta)
            sum_beta_hat += beta_hat
        R_mc = r_vals.mean()
        se_r = r_vals.std(ddof=1) / np.sqrt(T)
        e = abs(R_mc - R_th) / R_th
        se_e = se_r / R_th
        coef = np.dot(sum_beta_hat / T, beta)  # <E[beta_hat], beta>
        results[d] = dict(R_mc=R_mc, se_r=se_r, e=e, se_e=se_e, coef=coef)
    return results, R_th

def run_control(d_list, T, seed=43):
    """Control: underparam gamma=2, n=2d, OLS. Exact E[||beta_hat-beta||^2] = sigma^2*d/(n-d-1)."""
    rng = np.random.default_rng(seed)
    beta = np.zeros(1000)
    beta[0] = 1.0
    sigma = 0.1
    results = {}
    for d in d_list:
        n = 2 * d
        r_vals = np.zeros(T)
        for t in range(T):
            X = rng.standard_normal((n, d))
            y = X @ beta + sigma * rng.standard_normal(n)
            beta_hat = np.linalg.pinv(X) @ y
            r_vals[t] = np.dot(beta_hat - beta, beta_hat - beta)
        R_mc = r_vals.mean()
        se_r = r_vals.std(ddof=1) / np.sqrt(T)
        R_exact = sigma**2 * d / (n - d - 1)  # = 0.01*d/(d-1)
        e = abs(R_mc - R_exact) / R_exact
        se_e = se_r / R_exact
        results[d] = dict(R_mc=R_mc, se_r=se_r, e=e, se_e=se_e, R_exact=R_exact)
    return results

def main():
    t0 = time.time()
    os.makedirs('results/c9', exist_ok=True)

    # Per reviewer feedback: use 3 well-separated d values, T<=800
    d_list = [80, 150, 300]
    T = 800

    # Main test
    main_res, R_th = run_main(d_list, T, seed=42)

    # Check se_e < 0.25*e at all d; if not, retry with T=16000
    retry_needed = any(main_res[d]['se_e'] >= 0.25 * main_res[d]['e'] for d in d_list)
    if retry_needed:
        T = 16000
        main_res, R_th = run_main(d_list, T, seed=42)
        retry_note = "T raised to 16000 due to se_e>=0.25*e"
    else:
        retry_note = "T=800 sufficient"

    # Fit main: log10(e) vs log10(d)
    d_arr = np.array(d_list, dtype=float)
    e_arr = np.array([main_res[d]['e'] for d in d_list])
    # Guard against zero/negative e
    valid = e_arr > 0
    if valid.sum() >= 2:
        alpha, b, R2, se_alpha = ols_fit(np.log10(d_arr[valid]), np.log10(e_arr[valid]))
    else:
        alpha, b, R2, se_alpha = np.nan, np.nan, 0.0, np.nan

    # Control test
    ctrl_res = run_control(d_list, T, seed=43)
    e_ctrl = np.array([ctrl_res[d]['e'] for d in d_list])
    valid_c = e_ctrl > 0
    if valid_c.sum() >= 2:
        alpha_ctrl, b_c, R2_ctrl, se_alpha_ctrl = ols_fit(np.log10(d_arr[valid_c]), np.log10(e_ctrl[valid_c]))
    else:
        alpha_ctrl, b_c, R2_ctrl, se_alpha_ctrl = np.nan, np.nan, 0.0, np.nan

    # Control pass criteria
    ctrl_se_ok = all(ctrl_res[d]['se_e'] < 0.30 * ctrl_res[d]['e'] for d in d_list)
    control_pass = (0.85 <= alpha_ctrl <= 1.15) and (R2_ctrl >= 0.9) and ctrl_se_ok

    # Main success criteria
    main_se_ok = all(main_res[d]['se_e'] < 0.25 * main_res[d]['e'] for d in d_list)
    alpha_ok = (0.25 <= alpha <= 1.25) and (se_alpha <= 0.25)
    R2_ok = R2 >= 0.8

    if control_pass and alpha_ok and R2_ok and main_se_ok:
        status = "supported"
    elif control_pass and (alpha < 0.25 or R2 < 0.8):
        status = "falsified"
    else:
        status = "inconclusive"

    # Auxiliary: coef check
    coef_notes = []
    for d in d_list:
        c = main_res[d]['coef']
        if abs(c - 0.5) / 0.5 > 0.02:
            coef_notes.append(f"d={d}: coef={c:.4f} deviates from gamma=0.5")

    wall_s = time.time() - t0

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    # Main
    ax = axes[0]
    ax.loglog(d_arr, e_arr, 'o-', label='MC e(d)')
    if not np.isnan(alpha):
        d_fit = np.linspace(d_arr.min(), d_arr.max(), 50)
        e_fit = 10**(b + alpha*np.log10(d_fit))
        ax.loglog(d_fit, e_fit, 'r--', label=f'fit: alpha={alpha:.3f}')
    ax.set_xlabel('d'); ax.set_ylabel('e(d)')
    ax.set_title(f'Main: alpha={alpha:.3f}, R2={R2:.3f}')
    ax.legend()
    # Control
    ax = axes[1]
    ax.loglog(d_arr, e_ctrl, 'o-', label='MC e(d)')
    if not np.isnan(alpha_ctrl):
        d_fit = np.linspace(d_arr.min(), d_arr.max(), 50)
        e_fit = 10**(b_c + alpha_ctrl*np.log10(d_fit))
        ax.loglog(d_fit, e_fit, 'r--', label=f'fit: alpha={alpha_ctrl:.3f}')
    ax.set_xlabel('d'); ax.set_ylabel('e(d)')
    ax.set_title(f'Control: alpha={alpha_ctrl:.3f}, R2={R2_ctrl:.3f}')
    ax.legend()
    plt.tight_layout()
    plt.savefig('results/c9/fig.png', dpi=100)
    plt.close()

    # Build summary
    per_d = {}
    for d in d_list:
        per_d[str(d)] = {
            'R_mc': float(main_res[d]['R_mc']),
            'se_r': float(main_res[d]['se_r']),
            'e': float(main_res[d]['e']),
            'se_e': float(main_res[d]['se_e']),
            'coef': float(main_res[d]['coef'])
        }

    summary = {
        'claim_id': 'C9',
        'status': status,
        'metrics': {
            'per_d': per_d,
            'alpha': float(alpha) if not np.isnan(alpha) else None,
            'se_alpha': float(se_alpha) if not np.isnan(se_alpha) else None,
            'R2': float(R2),
            'control': {
                'alpha_ctrl': float(alpha_ctrl) if not np.isnan(alpha_ctrl) else None,
                'R2_ctrl': float(R2_ctrl),
                'control_pass': bool(control_pass)
            },
            'n_trials': T,
            'wall_s': float(wall_s),
            'control_pass': bool(control_pass)
        },
        'notes': f"{retry_note}. d_list={d_list}. " + "; ".join(coef_notes) if coef_notes else f"{retry_note}. d_list={d_list}. coef consistent with gamma=0.5."
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == '__main__':
    main()

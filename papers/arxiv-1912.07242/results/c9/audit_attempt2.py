import numpy as np
import json
import os
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_experiment(d_values, T, gamma, sigma, beta, underparam=False):
    """
    Runs the Monte Carlo experiment for a given set of dimensions.

    Args:
        d_values: List of dimensions d.
        T: Number of trials.
        gamma: Ratio n/d.
        sigma: Noise standard deviation.
        beta: True parameter vector.
        underparam: If True, use underparameterized regime (n > d) and pinv.
                    If False, use overparameterized regime (n < d) and solve.

    Returns:
        dict with results for each d.
    """
    results = {}
    for d in d_values:
        n = int(gamma * d)
        # Ensure n is integer and valid
        if underparam:
            # For underparam, we want n > d. If gamma=2, n=2d.
            pass
        else:
            # For overparam, n < d. If gamma=0.5, n=d//2.
            pass

        # The prompt specifies n=d//2 for the main test (gamma=0.5).
        # For the control (gamma=2), n=2d.
        # Let's strictly follow the prompt's definition of n based on gamma.
        # Prompt: "n=d//2" for main. "n=2d" for control.
        # So n is determined by the specific test setup, not just gamma*d generally.
        # However, the function takes gamma. Let's compute n = int(gamma * d) but be careful.
        # Actually, the prompt says: "gamma=0.5, n=d//2" and "gamma=2 (n=2d)".
        # So n = int(gamma * d) works for both if we are careful with integer division.
        # For d=100, gamma=0.5 -> n=50. d//2 = 50. Correct.
        # For d=100, gamma=2 -> n=200. 2d = 200. Correct.

        n = int(gamma * d)

        r_sum = 0.0
        r_sq_sum = 0.0
        beta_hat_sum = np.zeros_like(beta)

        for _ in range(T):
            X = np.random.randn(n, d)
            y = X @ beta + sigma * np.random.randn(n)

            if underparam:
                # Underparameterized: n > d. Unique solution.
                # beta_hat = pinv(X) @ y
                try:
                    beta_hat = np.linalg.pinv(X) @ y
                except np.linalg.LinAlgError:
                    beta_hat = np.linalg.lstsq(X, y, rcond=None)[0]
            else:
                # Overparameterized: n < d. Min-norm solution.
                # beta_hat = X.T @ solve(X @ X.T + 1e-12 * I_n, y)
                try:
                    A = X @ X.T + 1e-12 * np.eye(n)
                    beta_hat = X.T @ np.linalg.solve(A, y)
                except np.linalg.LinAlgError:
                    # Fallback to pinv if solve fails (should be rare with regularization)
                    beta_hat = np.linalg.pinv(X) @ y

            diff = beta_hat - beta
            r = np.dot(diff, diff)
            r_sum += r
            r_sq_sum += r * r
            beta_hat_sum += beta_hat

        R_mc = r_sum / T
        var_r = (r_sq_sum / T) - (R_mc ** 2)
        # Handle numerical issues where var might be slightly negative
        if var_r < 0:
            var_r = 0.0
        se_r = np.sqrt(var_r / T)

        # Theoretical risk
        if underparam:
            # Claim 2: V_n approx sigma^2 / (gamma - 1). Bias = 0.
            # But the prompt says for control: "EXACT finite-d expected parameter risk is sigma^2*d/(n-d-1)"
            # Wait, the prompt says: "where the EXACT finite-d expected parameter risk is sigma^2*d/(n-d-1)=0.01*d/(d-1)"
            # And "its relative deviation from the paper's Claim-2 approximation sigma^2/(gamma-1)=0.01 is exactly 1/(d-1)"
            # So R_th for control should be the EXACT one? Or the approximation?
            # The prompt says: "fit the same way... control_pass iff..."
            # And "relative deviation from the paper's Claim-2 approximation... is exactly 1/(d-1)"
            # This implies we are measuring the error of the APPROXIMATION against the EXACT truth?
            # No, the main test measures e(d) = |R_mc - R_th| / R_th.
            # For the control, we need a known truth to validate the pipeline.
            # The prompt says: "identical pipeline... where the EXACT... is... so its relative deviation from the paper's Claim-2 approximation... is exactly 1/(d-1)"
            # This suggests that for the control, we should compare R_mc to the EXACT risk, and the error should follow a power law.
            # OR, we compare R_mc to the APPROXIMATION, and the error is the finite-d correction.
            # Let's re-read carefully: "relative deviation from the paper's Claim-2 approximation... is exactly 1/(d-1)"
            # This means: (R_exact - R_approx) / R_approx = 1/(d-1).
            # So if we use R_th = R_approx, then e(d) = |R_mc - R_approx| / R_approx.
            # Since R_mc approximates R_exact, e(d) should approximate (R_exact - R_approx)/R_approx = 1/(d-1).
            # So alpha should be 1.
            # Therefore, for the control, R_th should be the APPROXIMATION sigma^2/(gamma-1).
            R_th = sigma**2 / (gamma - 1)
        else:
            # Overparameterized: Claim 1.
            # R_th = (1-gamma)||beta||^2 + sigma^2*gamma/(1-gamma)
            R_th = (1 - gamma) * np.dot(beta, beta) + sigma**2 * gamma / (1 - gamma)

        e = abs(R_mc - R_th) / R_th
        se_e = se_r / R_th

        # Coefficient check (auxiliary)
        # coef = <sum_beta_hat/T, beta> / ||beta||^2 ?
        # Prompt: "coef=<sum_beta_hat/T, beta>"
        # And later: "if |coef-gamma|/gamma>0.02..."
        # This implies coef is expected to be close to gamma.
        # E[beta_hat] = gamma * beta in overparam regime.
        # So <E[beta_hat], beta> = gamma * ||beta||^2.
        # If ||beta||^2 = 1, then coef = gamma.
        # The prompt defines coef = <sum_beta_hat/T, beta>. This is an estimate of <E[beta_hat], beta>.
        # So we should compare this to gamma * ||beta||^2.
        # But the check is |coef - gamma|/gamma. This assumes ||beta||^2 = 1.
        # In our setup beta = e_1, so ||beta||^2 = 1.
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
    Returns alpha, se_alpha, R2.
    """
    log_d = np.log10(d_values)
    log_e = np.log10(e_values)

    # OLS: log_e = log_C - alpha * log_d
    # y = b0 + b1 * x, where b1 = -alpha
    A = np.vstack([np.ones_like(log_d), log_d]).T
    b, residuals, rank, s = np.linalg.lstsq(A, log_e, rcond=None)

    log_C = b[0]
    alpha = -b[1]

    # Calculate R^2
    y_pred = A @ b
    ss_res = np.sum((log_e - y_pred) ** 2)
    ss_tot = np.sum((log_e - np.mean(log_e)) ** 2)
    if ss_tot == 0:
        R2 = 1.0
    else:
        R2 = 1 - (ss_res / ss_tot)

    # Standard error of alpha
    # Var(b) = sigma^2 * (A^T A)^-1
    # sigma^2 = ss_res / (n - p)
    n = len(log_d)
    p = 2
    if n > p:
        sigma2 = ss_res / (n - p)
        cov_b = sigma2 * np.linalg.inv(A.T @ A)
        se_alpha = np.sqrt(cov_b[1, 1])
    else:
        se_alpha = 0.0

    return alpha, se_alpha, R2

def main():
    start_time = time.time()

    # Setup
    np.random.seed(42)

    d_values = [100, 200, 400, 800]
    T = 4000
    sigma = 0.1
    beta = np.zeros(1000) # Max d is 800, so 1000 is safe. But we need to slice it.
    # Actually, beta dimension must match d.
    # We will create beta for each d inside the loop or pass a function.
    # Let's modify run_experiment to handle beta creation.

    # Re-implementing run_experiment to handle beta dimension correctly.
    def run_experiment_fixed(d_values, T, gamma, sigma, underparam=False):
        results = {}
        for d in d_values:
            n = int(gamma * d)
            beta = np.zeros(d)
            beta[0] = 1.0 # beta = e_1

            r_sum = 0.0
            r_sq_sum = 0.0
            beta_hat_sum = np.zeros(d)

            for _ in range(T):
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

                diff = beta_hat - beta
                r = np.dot(diff, diff)
                r_sum += r
                r_sq_sum += r * r
                beta_hat_sum += beta_hat

            R_mc = r_sum / T
            var_r = (r_sq_sum / T) - (R_mc ** 2)
            if var_r < 0:
                var_r = 0.0
            se_r = np.sqrt(var_r / T)

            if underparam:
                R_th = sigma**2 / (gamma - 1)
            else:
                R_th = (1 - gamma) * np.dot(beta, beta) + sigma**2 * gamma / (1 - gamma)

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

    # Main Test: Overparameterized, gamma=0.5
    gamma_main = 0.5
    main_results = run_experiment_fixed(d_values, T, gamma_main, sigma, underparam=False)

    # Check se_e condition
    se_e_violation = False
    for d in d_values:
        if main_results[d]['se_e'] >= 0.25 * main_results[d]['e']:
            se_e_violation = True
            break

    T_used = T
    if se_e_violation:
        T_used = 16000
        main_results = run_experiment_fixed(d_values, T_used, gamma_main, sigma, underparam=False)
        # Re-check after refit
        se_e_violation = False
        for d in d_values:
            if main_results[d]['se_e'] >= 0.25 * main_results[d]['e']:
                se_e_violation = True
                break

    # Extract e values for main
    e_main = [main_results[d]['e'] for d in d_values]
    alpha_main, se_alpha_main, R2_main = fit_power_law(d_values, e_main)

    # Control Test: Underparameterized, gamma=2
    gamma_ctrl = 2.0
    ctrl_results = run_experiment_fixed(d_values, T, gamma_ctrl, sigma, underparam=True)

    e_ctrl = [ctrl_results[d]['e'] for d in d_values]
    alpha_ctrl, se_alpha_ctrl, R2_ctrl = fit_power_law(d_values, e_ctrl)

    # Check control pass conditions
    control_pass = (
        0.85 <= alpha_ctrl <= 1.15 and
        R2_ctrl >= 0.9 and
        all(ctrl_results[d]['se_e'] < 0.30 * ctrl_results[d]['e'] for d in d_values)
    )

    # Determine status
    if not control_pass:
        status = "inconclusive"
        notes = "Control test failed. Measurement void."
    else:
        # Check main test conditions
        alpha_ok = 0.25 <= alpha_main <= 1.25
        se_alpha_ok = se_alpha_main <= 0.25
        R2_ok = R2_main >= 0.8
        se_e_ok = all(main_results[d]['se_e'] < 0.25 * main_results[d]['e'] for d in d_values)

        if alpha_ok and se_alpha_ok and R2_ok and se_e_ok:
            status = "supported"
            notes = "All conditions met."
        elif alpha_main < 0.25 or R2_main < 0.8:
            status = "falsified"
            notes = "No systematic d-decay or poor fit."
        else:
            status = "inconclusive"
            notes = "Conditions not fully met (se_alpha or se_e issues)."

        # Auxiliary check
        coef_violation = False
        for d in d_values:
            coef = main_results[d]['coef']
            if abs(coef - gamma_main) / gamma_main > 0.02:
                coef_violation = True
                break
        if coef_violation:
            notes += " Coefficient deviation noted (C8 bias mechanism)."

    # Plotting
    os.makedirs('results/c9', exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))

    # Main plot
    d_arr = np.array(d_values)
    e_main_arr = np.array(e_main)
    e_ctrl_arr = np.array(e_ctrl)

    ax.loglog(d_arr, e_main_arr, 'o-', label='Main (Overparam, $\gamma=0.5$)')
    ax.loglog(d_arr, e_ctrl_arr, 's-', label='Control (Underparam, $\gamma=2$)')

    # Fit lines
    d_fit = np.linspace(min(d_values), max(d_values), 100)

    # Main fit line: e = C * d^-alpha
    # log10(e) = log10(C) - alpha * log10(d)
    # We need C from the fit. b[0] was log10(C).
    # Let's re-calculate C for plotting.
    log_d = np.log10(d_arr)
    log_e = np.log10(e_main_arr)
    A = np.vstack([np.ones_like(log_d), log_d]).T
    b_main, _, _, _ = np.linalg.lstsq(A, log_e, rcond=None)
    C_main = 10**b_main[0]
    e_fit_main = C_main * d_fit**(-alpha_main)
    ax.loglog(d_fit, e_fit_main, '--', color='blue', alpha=0.5, label=f'Main Fit ($\alpha={alpha_main:.2f}$)')

    log_e_ctrl = np.log10(e_ctrl_arr)
    b_ctrl, _, _, _ = np.linalg.lstsq(A, log_e_ctrl, rcond=None)
    C_ctrl = 10**b_ctrl[0]
    e_fit_ctrl = C_ctrl * d_fit**(-alpha_ctrl)
    ax.loglog(d_fit, e_fit_ctrl, '--', color='red', alpha=0.5, label=f'Control Fit ($\alpha={alpha_ctrl:.2f}$)')

    ax.set_xlabel('Dimension d')
    ax.set_ylabel('Relative Error e(d)')
    ax.set_title('Finite-d Correction of C2 Risk Approximation')
    ax.legend()
    ax.grid(True, which="both", ls="--", lw=0.5)

    plt.tight_layout()
    plt.savefig('results/c9/fig.png', dpi=150)
    plt.close()

    wall_s = time.time() - start_time

    # Prepare summary
    summary = {
        "claim_id": "C9",
        "status": status,
        "metrics": {
            "alpha": float(alpha_main),
            "se_alpha": float(se_alpha_main),
            "R2": float(R2_main),
            "alpha_ctrl": float(alpha_ctrl),
            "R2_ctrl": float(R2_ctrl),
            "control_pass": bool(control_pass),
            "n_trials": T_used,
            "wall_s": float(wall_s),
            "per_d_main": {str(d): {k: float(v) for k, v in main_results[d].items()} for d in d_values},
            "per_d_ctrl": {str(d): {k: float(v) for k, v in ctrl_results[d].items()} for d in d_values}
        },
        "notes": notes
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    main()

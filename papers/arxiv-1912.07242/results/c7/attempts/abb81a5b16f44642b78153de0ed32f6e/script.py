import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_risk_mc(d, sigma, gamma_grid, T, seed=42):
    """
    Compute Monte-Carlo excess risk E||beta_hat - beta||^2 for the minimum-norm ridgeless interpolator.

    Parameters:
    -----------
    d : int
        Dimension
    sigma : float
        Noise standard deviation
    gamma_grid : list of float
        Grid of gamma = n/d values
    T : int
        Number of Monte-Carlo trials per gamma
    seed : int
        Random seed

    Returns:
    --------
    dict : {gamma: mean_risk}
    """
    np.random.seed(seed)
    beta = np.zeros(d)
    beta[0] = 1.0  # ||beta||_2 = 1

    results = {}
    for gamma in gamma_grid:
        n = int(np.floor(gamma * d))
        risks = []
        for t in range(T):
            X = np.random.randn(n, d)
            y = X @ beta + sigma * np.random.randn(n)
            # Minimum-norm solution: beta_hat = X^T (X X^T)^{-1} y
            # Use solve for efficiency, with pinv fallback
            try:
                XtXt = X @ X.T
                beta_hat = X.T @ np.linalg.solve(XtXt, y)
            except np.linalg.LinAlgError:
                beta_hat = np.linalg.pinv(X) @ y
            risk = np.sum((beta_hat - beta) ** 2)
            risks.append(risk)
        results[gamma] = np.mean(risks)
    return results

def closed_form_risk(gamma, sigma, beta_norm_sq=1.0):
    """
    Closed-form excess risk for overparameterized regime (gamma < 1):
    R(gamma) = (1 - gamma) * ||beta||^2 + sigma^2 * gamma / (1 - gamma)
    """
    if gamma >= 1.0:
        return np.inf
    return (1.0 - gamma) * beta_norm_sq + sigma**2 * gamma / (1.0 - gamma)

def find_valley_gamma(gamma_grid, risk_dict):
    """Find gamma with minimum risk."""
    min_risk = np.inf
    valley_gamma = None
    for gamma in gamma_grid:
        if risk_dict[gamma] < min_risk:
            min_risk = risk_dict[gamma]
            valley_gamma = gamma
    return valley_gamma

def main():
    # Setup
    d = 150
    sigma = 0.1
    gamma_grid = [0.75, 0.80, 0.85, 0.90, 0.95]
    T = 300

    # Main experiment
    mc_risks = compute_risk_mc(d, sigma, gamma_grid, T, seed=42)

    # Closed-form risks
    th_risks = {gamma: closed_form_risk(gamma, sigma) for gamma in gamma_grid}

    # Find valley
    valley_gamma = find_valley_gamma(gamma_grid, mc_risks)

    # Check strict minimum: valley risk < risks at endpoints
    strict_min = mc_risks[valley_gamma] < min(mc_risks[0.75], mc_risks[0.95])

    # Relative error at valley
    rel_err_at_valley = abs(mc_risks[valley_gamma] - th_risks[valley_gamma]) / th_risks[valley_gamma]

    # Check if valley is in {0.85, 0.90}
    valley_in_expected = valley_gamma in [0.85, 0.90]

    # Positive control: sigma = 0.5, valley should be at gamma = 0.75
    sigma_ctrl = 0.5
    mc_risks_ctrl = compute_risk_mc(d, sigma_ctrl, gamma_grid, T, seed=42)
    th_risks_ctrl = {gamma: closed_form_risk(gamma, sigma_ctrl) for gamma in gamma_grid}
    valley_gamma_ctrl = find_valley_gamma(gamma_grid, mc_risks_ctrl)
    rel_err_ctrl = abs(mc_risks_ctrl[valley_gamma_ctrl] - th_risks_ctrl[valley_gamma_ctrl]) / th_risks_ctrl[valley_gamma_ctrl]
    control_pass = (valley_gamma_ctrl == 0.75) and (rel_err_ctrl <= 0.10)

    # Determine status
    if not control_pass:
        status = "inconclusive"
        notes = "Positive control failed: sigma=0.5 run did not recover valley at gamma=0.75 with rel_err<=0.10. Measurement is void."
    else:
        if valley_in_expected and strict_min and rel_err_at_valley <= 0.10:
            status = "supported"
            notes = f"Valley at gamma={valley_gamma}, strict_min={strict_min}, rel_err={rel_err_at_valley:.4f}. All criteria met."
        else:
            status = "falsified"
            notes = f"Valley at gamma={valley_gamma} (expected 0.85 or 0.90), strict_min={strict_min}, rel_err={rel_err_at_valley:.4f}."

    # Metrics
    metrics = {
        "valley_gamma": float(valley_gamma),
        "strict_min": bool(strict_min),
        "rel_err_at_valley": float(rel_err_at_valley),
        "valley_in_expected": bool(valley_in_expected),
        "control_valley_gamma": float(valley_gamma_ctrl),
        "control_rel_err": float(rel_err_ctrl),
        "control_pass": bool(control_pass),
        "mc_risks": {str(g): float(r) for g, r in mc_risks.items()},
        "th_risks": {str(g): float(r) for g, r in th_risks.items()},
        "mc_risks_ctrl": {str(g): float(r) for g, r in mc_risks_ctrl.items()},
        "th_risks_ctrl": {str(g): float(r) for g, r in th_risks_ctrl.items()}
    }

    # Plot
    os.makedirs("results/c7", exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    gammas = sorted(gamma_grid)
    mc_vals = [mc_risks[g] for g in gammas]
    th_vals = [th_risks[g] for g in gammas]
    mc_vals_ctrl = [mc_risks_ctrl[g] for g in gammas]
    th_vals_ctrl = [th_risks_ctrl[g] for g in gammas]

    ax.plot(gammas, mc_vals, 'o-', label='MC (sigma=0.1)', color='blue')
    ax.plot(gammas, th_vals, 's--', label='Theory (sigma=0.1)', color='blue', alpha=0.5)
    ax.plot(gammas, mc_vals_ctrl, 'o-', label='MC (sigma=0.5)', color='red')
    ax.plot(gammas, th_vals_ctrl, 's--', label='Theory (sigma=0.5)', color='red', alpha=0.5)
    ax.axvline(x=0.90, color='green', linestyle=':', label='gamma*=0.90 (sigma=0.1)')
    ax.axvline(x=0.75, color='orange', linestyle=':', label='gamma*=0.75 (sigma=0.5)')
    ax.set_xlabel('gamma = n/d')
    ax.set_ylabel('Excess Risk E||beta_hat - beta||^2')
    ax.set_title('Finite-d Risk Valley: Double Descent Dip')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("results/c7/fig.png", dpi=150)
    plt.close()

    summary = {
        "claim_id": "C7",
        "status": status,
        "metrics": metrics,
        "notes": notes
    }
    print("SUMMARY_JSON=" + json.dumps(summary, default=str))

if __name__ == "__main__":
    main()

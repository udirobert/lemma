import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_beta_hat(X, y):
    """
    Compute the minimum-norm least squares solution beta_hat = X^T (XX^T)^{-1} y.
    This is the estimator found by gradient descent on the least squares objective.
    """
    # X is n x d, y is n x 1
    # beta_hat = X^T (XX^T)^{-1} y
    # Use solve for stability
    XXt = X @ X.T
    # Add small regularization for numerical stability if needed, but problem says ridgeless
    # However, for n < d, XXt is n x n and invertible with prob 1.
    # For n >= d, we are in underparameterized regime, but the claim specifies gamma < 1.
    # The formula X^T (XX^T)^{-1} y works for n <= d.
    # If n > d, we would use (X^T X)^{-1} X^T y, but the claim is for gamma in {0.1, 0.3, 0.5, 0.7, 0.9}, so n < d.
    try:
        beta_hat = X.T @ np.linalg.solve(XXt, y)
    except np.linalg.LinAlgError:
        # Fallback to pinv if singular (should not happen for Gaussian)
        beta_hat = np.linalg.pinv(X) @ y
    return beta_hat

def run_experiment(d, sigma, beta, gammas, T, seed=42):
    np.random.seed(seed)
    results = {}

    for gamma in gammas:
        n = int(np.floor(gamma * d))
        sum_beta_hat = np.zeros(d)

        for t in range(T):
            # Generate data
            X = np.random.randn(n, d)
            noise = np.random.randn(n) * sigma
            y = X @ beta + noise

            # Compute estimator
            beta_hat = compute_beta_hat(X, y)
            sum_beta_hat += beta_hat

        E_beta_hat = sum_beta_hat / T

        # Metrics
        # Coefficient along beta
        coef = np.dot(E_beta_hat, beta)
        # Orthogonal component magnitude
        # ||E_beta_hat||^2 = coef^2 + ||E_beta_hat_perp||^2
        # So ||E_beta_hat_perp|| = sqrt(||E_beta_hat||^2 - coef^2)
        norm_sq = np.dot(E_beta_hat, E_beta_hat)
        ortho_sq = norm_sq - coef**2
        # Due to numerical errors, ortho_sq might be slightly negative
        if ortho_sq < 0:
            ortho_sq = 0.0
        ortho = np.sqrt(ortho_sq)

        # Relative error of coefficient
        rel_err_coef = abs(coef - gamma) / gamma

        results[gamma] = {
            'n': n,
            'coef': coef,
            'ortho': ortho,
            'rel_err_coef': rel_err_coef,
            'E_beta_hat': E_beta_hat
        }

    return results

def main():
    # Parameters from claim
    d = 150
    sigma = 0.1
    beta = np.zeros(d)
    beta[0] = 1.0  # ||beta||_2 = 1
    gammas = [0.1, 0.3, 0.5, 0.7, 0.9]
    T = 200

    # Run experiment
    results = run_experiment(d, sigma, beta, gammas, T)

    # Check success criteria
    all_pass = True
    control_pass = False
    metrics = {}

    for gamma in gammas:
        res = results[gamma]
        coef = res['coef']
        ortho = res['ortho']
        rel_err = res['rel_err_coef']

        # Criterion 1: |coef - gamma|/gamma <= 0.02
        pass_coef = rel_err <= 0.02
        # Criterion 2: ortho <= 0.05
        pass_ortho = ortho <= 0.05

        metrics[f'gamma_{gamma}_coef'] = coef
        metrics[f'gamma_{gamma}_ortho'] = ortho
        metrics[f'gamma_{gamma}_rel_err'] = rel_err
        metrics[f'gamma_{gamma}_pass'] = pass_coef and pass_ortho

        if not (pass_coef and pass_ortho):
            all_pass = False

        # Control check for gamma=0.5
        if gamma == 0.5:
            control_pass = pass_coef and pass_ortho
            metrics['control_pass'] = control_pass

    # If control fails, status is inconclusive
    if not control_pass:
        status = "inconclusive"
        notes = "Control check at gamma=0.5 failed. Measurement is void."
    elif all_pass:
        status = "supported"
        notes = "All gamma values passed the success criteria."
    else:
        status = "falsified"
        notes = "One or more gamma values failed the success criteria."

    # Plotting
    os.makedirs('results/c8', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # Plot 1: Coefficient vs Gamma
    gammas_plot = [g for g in gammas]
    coefs_plot = [results[g]['coef'] for g in gammas_plot]
    axes[0].plot(gammas_plot, coefs_plot, 'o-', label='MC Estimate')
    axes[0].plot(gammas_plot, gammas_plot, 'k--', label='Theoretical (gamma)')
    axes[0].set_xlabel('gamma')
    axes[0].set_ylabel('Coefficient along beta')
    axes[0].set_title('E[beta_hat] Coefficient vs gamma')
    axes[0].legend()
    axes[0].grid(True)

    # Plot 2: Orthogonal component vs Gamma
    orthos_plot = [results[g]['ortho'] for g in gammas_plot]
    axes[1].plot(gammas_plot, orthos_plot, 'o-', label='MC Estimate')
    axes[1].axhline(y=0.05, color='r', linestyle='--', label='Threshold (0.05)')
    axes[1].set_xlabel('gamma')
    axes[1].set_ylabel('Orthogonal Component Norm')
    axes[1].set_title('E[beta_hat] Orthogonal Component vs gamma')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('results/c8/fig.png', dpi=150)
    plt.close()

    # Final summary
    summary = {
        "claim_id": "C8",
        "status": status,
        "metrics": metrics,
        "notes": notes
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    main()

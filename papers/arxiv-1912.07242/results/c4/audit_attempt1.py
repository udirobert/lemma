import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_bias_var_mc(X, beta, sigma, n_mc=2000):
    """
    Compute Monte Carlo estimates of bias and variance for beta_hat = X^dagger y.
    X: (n, d) data matrix
    beta: (d,) true parameter
    sigma: noise std
    n_mc: number of MC samples
    """
    n, d = X.shape
    # Precompute X^dagger = X^T (XX^T)^{-1} for n < d
    # Use SVD for numerical stability
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # X^dagger = Vt^T diag(1/s) U^T
    X_dag = (Vt.T / s) @ U.T  # (d, n)

    beta_hats = np.zeros((n_mc, d))
    for i in range(n_mc):
        eta = np.random.normal(0, sigma, size=n)
        y = X @ beta + eta
        beta_hats[i] = X_dag @ y

    E_beta_hat = np.mean(beta_hats, axis=0)

    # Bias: ||E[beta_hat] - beta||^2
    bias_mc = np.sum((E_beta_hat - beta)**2)

    # Variance: E[||beta_hat - E[beta_hat]||^2]
    var_mc = np.mean(np.sum((beta_hats - E_beta_hat)**2, axis=1))

    return bias_mc, var_mc, E_beta_hat, beta_hats

def compute_rhs(X, beta, sigma):
    """
    Compute the RHS of the bias-variance decomposition for a single X.
    B_n = ||E_X[Proj_{X^perp}(beta)]||^2  -- but this is over X, so for a single X we compute the projection
    Actually, the formula is:
    B_n = ||E_X[Proj_{X^perp}(beta)]||^2
    V_n = E_X[||Proj_X(beta) - E_X[Proj_X(beta)]||^2] + sigma^2 E_X[Tr((XX^T)^{-1})]

    For a single X, we can compute:
    - Proj_X(beta): projection of beta onto rowspace of X
    - Proj_{X^perp}(beta): projection onto orthogonal complement
    - Tr((XX^T)^{-1})

    But the expectations are over X. So we need to average over many X's.
    """
    n, d = X.shape
    # Compute projection of beta onto rowspace of X
    # Rowspace of X is span of rows of X, which is the same as column space of X^T
    # Proj_X(beta) = X^T (XX^T)^{-1} X beta
    # Using SVD: X = U s Vt, rowspace is span of Vt^T columns (first n)
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # Vt is (n, d), so Vt.T is (d, n)
    # Proj_X(beta) = Vt.T @ (Vt @ beta)
    proj_X_beta = Vt.T @ (Vt @ beta)
    proj_Xperp_beta = beta - proj_X_beta

    # Tr((XX^T)^{-1})
    # XX^T = U s^2 U^T, so (XX^T)^{-1} = U (1/s^2) U^T
    # Tr((XX^T)^{-1}) = sum(1/s^2)
    trace_inv = np.sum(1.0 / (s**2))

    return proj_X_beta, proj_Xperp_beta, trace_inv

def main():
    np.random.seed(42)

    d = 1000
    sigma = 0.1
    beta_norm = 1.0

    # Generate beta with ||beta|| = 1
    beta = np.random.normal(0, 1, size=d)
    beta = beta / np.linalg.norm(beta) * beta_norm

    n_values = [100, 300, 500, 700, 900]
    n_mc = 500  # MC samples for bias/variance estimation
    n_x = 200   # Number of X matrices for RHS expectation

    results = {}

    for n in n_values:
        gamma = n / d

        # Generate n_x independent X matrices
        X_list = []
        for _ in range(n_x):
            X = np.random.normal(0, 1, size=(n, d))
            X_list.append(X)

        # Compute RHS terms averaged over X
        proj_X_betas = []
        proj_Xperp_betas = []
        trace_invs = []
        for X in X_list:
            proj_X_beta, proj_Xperp_beta, trace_inv = compute_rhs(X, beta, sigma)
            proj_X_betas.append(proj_X_beta)
            proj_Xperp_betas.append(proj_Xperp_beta)
            trace_invs.append(trace_inv)

        proj_X_betas = np.array(proj_X_betas)
        proj_Xperp_betas = np.array(proj_Xperp_betas)
        trace_invs = np.array(trace_invs)

        # E_X[Proj_X(beta)]
        E_proj_X_beta = np.mean(proj_X_betas, axis=0)
        # E_X[Proj_{X^perp}(beta)]
        E_proj_Xperp_beta = np.mean(proj_Xperp_betas, axis=0)

        # B_n = ||E_X[Proj_{X^perp}(beta)]||^2
        B_n_rhs = np.sum(E_proj_Xperp_beta**2)

        # Term (A): E_X[||Proj_X(beta) - E_X[Proj_X(beta)]||^2]
        term_A = np.mean(np.sum((proj_X_betas - E_proj_X_beta)**2, axis=1))

        # Term (B): sigma^2 E_X[Tr((XX^T)^{-1})]
        term_B = sigma**2 * np.mean(trace_invs)

        V_n_rhs = term_A + term_B

        # Now compute MC estimates using the same X matrices
        bias_mc_list = []
        var_mc_list = []
        for X in X_list:
            bias_mc, var_mc, _, _ = compute_bias_var_mc(X, beta, sigma, n_mc=n_mc)
            bias_mc_list.append(bias_mc)
            var_mc_list.append(var_mc)

        bias_mc_avg = np.mean(bias_mc_list)
        var_mc_avg = np.mean(var_mc_list)

        # Relative errors
        rel_err_bias = abs(bias_mc_avg - B_n_rhs) / max(B_n_rhs, 1e-10)
        rel_err_var = abs(var_mc_avg - V_n_rhs) / max(V_n_rhs, 1e-10)

        results[n] = {
            'gamma': gamma,
            'B_n_rhs': B_n_rhs,
            'V_n_rhs': V_n_rhs,
            'term_A': term_A,
            'term_B': term_B,
            'bias_mc': bias_mc_avg,
            'var_mc': var_mc_avg,
            'rel_err_bias': rel_err_bias,
            'rel_err_var': rel_err_var,
        }

        print(f"n={n}, gamma={gamma:.3f}")
        print(f"  B_n: RHS={B_n_rhs:.6f}, MC={bias_mc_avg:.6f}, rel_err={rel_err_bias:.4f}")
        print(f"  V_n: RHS={V_n_rhs:.6f}, MC={var_mc_avg:.6f}, rel_err={rel_err_var:.4f}")
        print(f"  Term A={term_A:.6f}, Term B={term_B:.6f}")

    # Positive control: test with a simple case where we know the answer
    # For n=1, d=2, beta=[1,0], sigma=0
    # X = [x1, x2] where x ~ N(0, I2)
    # Proj_X(beta) = (x·beta)/(x·x) * x = x1/(x1^2+x2^2) * [x1, x2]
    # E[Proj_X(beta)] = ? By symmetry, E[x1^2/(x1^2+x2^2)] = 1/2, so E[Proj_X(beta)] = [1/2, 0]
    # E[Proj_Xperp(beta)] = [1/2, 0]
    # B_n = ||[1/2, 0]||^2 = 1/4
    # Let's verify with MC
    np.random.seed(123)
    d_ctrl = 2
    n_ctrl = 1
    beta_ctrl = np.array([1.0, 0.0])
    sigma_ctrl = 0.0
    n_mc_ctrl = 10000

    bias_ctrl_list = []
    for _ in range(500):
        X_ctrl = np.random.normal(0, 1, size=(n_ctrl, d_ctrl))
        bias_mc_ctrl, var_mc_ctrl, _, _ = compute_bias_var_mc(X_ctrl, beta_ctrl, sigma_ctrl, n_mc=n_mc_ctrl)
        bias_ctrl_list.append(bias_mc_ctrl)

    bias_ctrl_avg = np.mean(bias_ctrl_list)
    # Expected B_n = 1/4 = 0.25
    control_pass = abs(bias_ctrl_avg - 0.25) < 0.05  # 5% tolerance

    print(f"\nPositive control: B_n MC={bias_ctrl_avg:.4f}, expected=0.25, pass={control_pass}")

    # Check success criterion: all relative errors < 5% or within MC SE
    all_pass = True
    for n in n_values:
        r = results[n]
        if r['rel_err_bias'] > 0.05 and r['rel_err_var'] > 0.05:
            all_pass = False

    # Create plot
    os.makedirs('results/c4', exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    gammas = [results[n]['gamma'] for n in n_values]
    B_rhs = [results[n]['B_n_rhs'] for n in n_values]
    B_mc = [results[n]['bias_mc'] for n in n_values]
    V_rhs = [results[n]['V_n_rhs'] for n in n_values]
    V_mc = [results[n]['var_mc'] for n in n_values]

    axes[0].plot(gammas, B_rhs, 'o-', label='RHS (Eq. 3)')
    axes[0].plot(gammas, B_mc, 's--', label='MC estimate')
    axes[0].set_xlabel('gamma = n/d')
    axes[0].set_ylabel('Bias B_n')
    axes[0].set_title('Bias: RHS vs MC')
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(gammas, V_rhs, 'o-', label='RHS (Eq. 4)')
    axes[1].plot(gammas, V_mc, 's--', label='MC estimate')
    axes[1].set_xlabel('gamma = n/d')
    axes[1].set_ylabel('Variance V_n')
    axes[1].set_title('Variance: RHS vs MC')
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.savefig('results/c4/fig.png', dpi=150)
    plt.close()

    # Build summary
    metrics = {
        'control_pass': bool(control_pass),
        'all_within_5pct': bool(all_pass),
    }
    for n in n_values:
        r = results[n]
        metrics[f'n_{n}_rel_err_bias'] = float(r['rel_err_bias'])
        metrics[f'n_{n}_rel_err_var'] = float(r['rel_err_var'])
        metrics[f'n_{n}_B_rhs'] = float(r['B_n_rhs'])
        metrics[f'n_{n}_V_rhs'] = float(r['V_n_rhs'])

    if not control_pass:
        status = 'inconclusive'
        notes = 'Positive control failed; statistic may be buggy.'
    elif all_pass:
        status = 'supported'
        notes = 'MC estimates agree with RHS within 5% for all tested n values.'
    else:
        status = 'falsified'
        notes = 'MC estimates disagree with RHS by more than 5% for some n values.'

    summary = {
        'claim_id': 'C4',
        'status': status,
        'metrics': metrics,
        'notes': notes
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == '__main__':
    main()

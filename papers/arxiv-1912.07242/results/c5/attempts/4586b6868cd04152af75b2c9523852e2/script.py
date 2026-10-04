import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

def compute_lhs(X, x):
    """Compute Tr((X_{n+1} X_{n+1}^T)^{-1}) using SVD for stability."""
    X_new = np.vstack([X, x.reshape(1, -1)])
    # X_new is (n+1) x d, with n+1 < d
    # We need Tr((X_new X_new^T)^{-1})
    # Use SVD: X_new = U S V^T, then X_new X_new^T = U S^2 U^T
    # So (X_new X_new^T)^{-1} = U S^{-2} U^T
    # Trace = sum(1/s_i^2)
    U, S, Vt = np.linalg.svd(X_new, full_matrices=False)
    # S has length min(n+1, d) = n+1
    # Filter out very small singular values for numerical stability
    tol = 1e-12
    S_filtered = S[S > tol]
    if len(S_filtered) == 0:
        return np.inf
    return np.sum(1.0 / (S_filtered ** 2))

def compute_rhs(X, x):
    """Compute Tr((XX^T)^{-1}) + (1 + ||(X^T)^dagger x||^2) / ||Proj_{X^perp}(x)||^2"""
    # Tr((XX^T)^{-1})
    U, S, Vt = np.linalg.svd(X, full_matrices=False)
    tol = 1e-12
    S_filtered = S[S > tol]
    if len(S_filtered) == 0:
        trace_old = np.inf
    else:
        trace_old = np.sum(1.0 / (S_filtered ** 2))

    # (X^T)^dagger x = X^T (X X^T)^{-1} x
    # Compute (X X^T)^{-1} x via SVD: X X^T = U S^2 U^T
    # So (X X^T)^{-1} x = U S^{-2} U^T x
    # U^T x is a vector of length n
    Ut_x = U.T @ x
    # S^{-2} U^T x
    S_inv2_Ut_x = Ut_x / (S ** 2)
    # X^T (X X^T)^{-1} x = V S U^T * U S^{-2} U^T x = V S^{-1} U^T x
    # Wait, let me recompute: X = U S V^T, so X^T = V S U^T
    # (X X^T)^{-1} = U S^{-2} U^T
    # X^T (X X^T)^{-1} x = V S U^T U S^{-2} U^T x = V S^{-1} U^T x
    # So ||(X^T)^dagger x||^2 = ||V S^{-1} U^T x||^2 = ||S^{-1} U^T x||^2 (since V is orthogonal)
    norm_sq = np.sum((S_inv2_Ut_x * S) ** 2)  # This is ||S^{-1} U^T x||^2
    # Actually: S^{-1} U^T x has components (Ut_x[i] / S[i])
    # So norm_sq = sum((Ut_x[i] / S[i])^2)
    norm_sq = np.sum((Ut_x / S) ** 2)

    # Proj_{X^perp}(x) = x - Proj_X(x)
    # Proj_X(x) = X (X^T)^dagger x = U S V^T V S^{-1} U^T x = U U^T x
    # So Proj_{X^perp}(x) = x - U U^T x = (I - U U^T) x
    # ||Proj_{X^perp}(x)||^2 = ||x||^2 - ||U^T x||^2
    proj_perp_norm_sq = np.sum(x ** 2) - np.sum(Ut_x ** 2)

    if proj_perp_norm_sq < 1e-15:
        return np.inf

    return trace_old + (1.0 + norm_sq) / proj_perp_norm_sq

def positive_control():
    """Test with a simple case where we can verify analytically."""
    # Let d=3, n=1, X = [1, 0, 0], x = [0, 1, 0]
    # X_{n+1} = [[1,0,0],[0,1,0]]
    # X_{n+1} X_{n+1}^T = I_2, so Tr((X_{n+1} X_{n+1}^T)^{-1}) = 2
    # XX^T = [1], Tr((XX^T)^{-1}) = 1
    # (X^T)^dagger x: X^T = [1,0,0]^T, (X^T)^dagger = [1,0,0], so (X^T)^dagger x = 0
    # Proj_{X^perp}(x) = x - Proj_X(x) = [0,1,0] - 0 = [0,1,0], norm^2 = 1
    # RHS = 1 + (1 + 0)/1 = 2
    X = np.array([[1.0, 0.0, 0.0]])
    x = np.array([0.0, 1.0, 0.0])
    lhs = compute_lhs(X, x)
    rhs = compute_rhs(X, x)
    return np.isclose(lhs, rhs, rtol=1e-10)

def main():
    d = 1000
    n_values = [100, 300, 500, 700, 900]
    n_trials = 10

    # Positive control
    control_pass = positive_control()

    all_rel_diffs = []
    max_rel_diff = 0.0
    all_pass = True

    for n in n_values:
        for trial in range(n_trials):
            X = np.random.randn(n, d)
            x = np.random.randn(d)

            lhs = compute_lhs(X, x)
            rhs = compute_rhs(X, x)

            if np.isinf(lhs) or np.isinf(rhs):
                rel_diff = np.inf
            else:
                rel_diff = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-15)

            all_rel_diffs.append(rel_diff)
            if rel_diff > max_rel_diff:
                max_rel_diff = rel_diff
            if rel_diff >= 1e-8:
                all_pass = False

    # Plot
    os.makedirs('results/c5', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist([rd for rd in all_rel_diffs if rd < 1e-6], bins=50, edgecolor='black')
    plt.xlabel('Relative difference')
    plt.ylabel('Count')
    plt.title(f'Distribution of relative differences (max={max_rel_diff:.2e})')
    plt.yscale('log')
    plt.savefig('results/c5/fig.png', dpi=150, bbox_inches='tight')
    plt.close()

    status = "supported" if (all_pass and control_pass) else ("inconclusive" if not control_pass else "falsified")

    summary = {
        "claim_id": "C5",
        "status": status,
        "metrics": {
            "control_pass": bool(control_pass),
            "max_rel_diff": float(max_rel_diff),
            "mean_rel_diff": float(np.mean(all_rel_diffs)),
            "num_instances": len(all_rel_diffs),
            "num_pass": int(sum(1 for rd in all_rel_diffs if rd < 1e-8)),
            "threshold": 1e-8
        },
        "notes": f"Tested {len(all_rel_diffs)} instances with d=1000, n in {n_values}. Max relative diff: {max_rel_diff:.2e}. Control pass: {control_pass}."
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    main()

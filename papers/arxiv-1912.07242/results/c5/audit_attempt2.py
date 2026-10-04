import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_lhs(X, x):
    """Compute Tr((X_{n+1} X_{n+1}^T)^{-1}) where X_{n+1} = [X; x^T]."""
    X_new = np.vstack([X, x.reshape(1, -1)])
    # X_new is (n+1) x d, n+1 < d, so X_new X_new^T is (n+1) x (n+1) and invertible
    A = X_new @ X_new.T
    return np.trace(np.linalg.inv(A))

def compute_rhs(X, x):
    """Compute Tr((XX^T)^{-1}) + (1 + ||(X^T)^dagger x||^2) / ||Proj_{X^perp}(x)||^2."""
    n, d = X.shape
    # Tr((XX^T)^{-1})
    A = X @ X.T
    trace_old = np.trace(np.linalg.inv(A))

    # (X^T)^dagger x = X^T (X X^T)^{-1} x
    # This is the projection of x onto the row space of X (which is the column space of X^T)
    # Let's compute it as: X^T @ inv(X @ X.T) @ x
    Xt_dag_x = X.T @ np.linalg.inv(A) @ x
    norm_sq = np.dot(Xt_dag_x, Xt_dag_x)

    # Proj_{X^perp}(x) = x - Proj_{rowspace(X)}(x)
    # Proj_{rowspace(X)}(x) = X^T (X X^T)^{-1} X x = X^T (X X^T)^{-1} (X x)
    # Wait, the projection of x onto the row space of X (which is a subspace of R^d) is:
    # P x = X^T (X X^T)^{-1} X x
    # So Proj_{X^perp}(x) = x - X^T (X X^T)^{-1} X x
    # Let's compute this carefully.
    # X x is a vector in R^n.
    # (X X^T)^{-1} X x is a vector in R^n.
    # X^T (X X^T)^{-1} X x is a vector in R^d.

    # Alternatively, using the SVD or QR decomposition for stability.
    # Let's use the formula: ||Proj_{X^perp}(x)||^2 = ||x||^2 - ||Proj_{rowspace(X)}(x)||^2
    # And ||Proj_{rowspace(X)}(x)||^2 = x^T X^T (X X^T)^{-1} X x = (X x)^T (X X^T)^{-1} (X x)

    Xx = X @ x
    proj_norm_sq = Xx @ np.linalg.inv(A) @ Xx

    # The term in the numerator is 1 + ||(X^T)^dagger x||^2
    # Note that (X^T)^dagger x is the coefficient vector c such that Proj_{rowspace(X)}(x) = X^T c.
    # Indeed, c = (X X^T)^{-1} X x.
    # So ||(X^T)^dagger x||^2 = c^T c = X x (X X^T)^{-1} (X X^T)^{-1} X x ? No.
    # c = (X X^T)^{-1} X x.
    # ||c||^2 = c^T c = x^T X^T (X X^T)^{-1} (X X^T)^{-1} X x.
    # Wait, the claim says ||(X^T)^dagger x||^2.
    # (X^T)^dagger = (X^T)^T ((X^T)^T X^T)^{-1} = X (X X^T)^{-1}.
    # So (X^T)^dagger x = X (X X^T)^{-1} x.
    # Let's re-read the claim: ||(X^T)^dagger x||^2.
    # Yes, (X^T)^dagger = X (X X^T)^{-1}.
    # So the vector is v = X (X X^T)^{-1} x.
    # And we need ||v||^2.

    v = X @ np.linalg.inv(A) @ x
    norm_v_sq = np.dot(v, v)

    # Denominator: ||Proj_{X^perp}(x)||^2
    # Proj_{X^perp}(x) = x - X^T (X X^T)^{-1} X x
    # Let's compute this directly to be safe, or use the norm difference.
    # ||Proj_{X^perp}(x)||^2 = ||x||^2 - ||X^T (X X^T)^{-1} X x||^2
    # Note that X^T (X X^T)^{-1} X x is the projection of x onto the row space.
    # Let P = X^T (X X^T)^{-1} X.
    # proj_x = P x.
    # ||proj_x||^2 = x^T P^T P x = x^T P x (since P is symmetric and idempotent).
    # P x = X^T (X X^T)^{-1} X x.
    # So ||proj_x||^2 = (X x)^T (X X^T)^{-1} (X x).
    # This matches proj_norm_sq calculated above.

    norm_x_sq = np.dot(x, x)
    denom = norm_x_sq - proj_norm_sq

    # Handle numerical issues where denom might be very small or negative due to precision
    if denom < 1e-12:
        # This should not happen for random Gaussian data with n < d, but just in case
        denom = 1e-12

    rhs = trace_old + (1 + norm_v_sq) / denom
    return rhs

def positive_control():
    """
    Test the identity on a simple case where we can compute the answer exactly.
    Let d=2, n=1.
    X = [[1, 0]] (1x2 matrix).
    x = [0, 1] (2x1 vector).

    X_{n+1} = [[1, 0], [0, 1]] = I_2.
    LHS = Tr((I_2 I_2^T)^{-1}) = Tr(I_2^{-1}) = Tr(I_2) = 2.

    RHS:
    Tr((XX^T)^{-1}) = Tr(([1,0][1,0]^T)^{-1}) = Tr([1]^{-1}) = 1.
    (X^T)^dagger x:
    X^T = [[1], [0]].
    (X^T)^dagger = (X^T)^T ((X^T)^T X^T)^{-1} = [[1, 0]] * [1]^{-1} = [[1, 0]].
    (X^T)^dagger x = [[1, 0]] @ [0, 1]^T = 0.
    ||(X^T)^dagger x||^2 = 0.

    Proj_{X^perp}(x):
    Row space of X is span([1, 0]).
    X^perp is span([0, 1]).
    Proj_{X^perp}([0, 1]) = [0, 1].
    ||Proj_{X^perp}(x)||^2 = 1.

    RHS = 1 + (1 + 0) / 1 = 2.

    LHS == RHS.
    """
    X = np.array([[1.0, 0.0]])
    x = np.array([0.0, 1.0])

    lhs = compute_lhs(X, x)
    rhs = compute_rhs(X, x)

    rel_diff = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-15)
    return rel_diff < 1e-8, lhs, rhs

def main():
    np.random.seed(42)

    # Positive Control
    control_pass, ctrl_lhs, ctrl_rhs = positive_control()

    d = 1000
    n_values = [100, 300, 500, 700, 900]
    num_trials = 5

    all_rel_diffs = []
    max_rel_diff = 0.0
    all_pass = True

    for n in n_values:
        for trial in range(num_trials):
            X = np.random.randn(n, d)
            x = np.random.randn(d)

            lhs = compute_lhs(X, x)
            rhs = compute_rhs(X, x)

            rel_diff = abs(lhs - rhs) / max(abs(lhs), abs(rhs), 1e-15)
            all_rel_diffs.append(rel_diff)

            if rel_diff > max_rel_diff:
                max_rel_diff = rel_diff

            if rel_diff >= 1e-8:
                all_pass = False
                print(f"FAIL: n={n}, trial={trial}, rel_diff={rel_diff}")

    # Plot
    os.makedirs('results/c5', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(all_rel_diffs, bins=50, edgecolor='black')
    plt.axvline(1e-8, color='red', linestyle='--', label='Threshold (1e-8)')
    plt.xlabel('Relative Difference')
    plt.ylabel('Frequency')
    plt.title('Distribution of Relative Differences for Claim 3')
    plt.yscale('log')
    plt.legend()
    plt.grid(True, which="both", ls="--", alpha=0.5)
    plt.tight_layout()
    plt.savefig('results/c5/fig.png', dpi=150)
    plt.close()

    status = "supported" if (all_pass and control_pass) else ("inconclusive" if not control_pass else "falsified")

    summary = {
        "claim_id": "C5",
        "status": status,
        "metrics": {
            "control_pass": control_pass,
            "control_lhs": ctrl_lhs,
            "control_rhs": ctrl_rhs,
            "max_rel_diff": max_rel_diff,
            "mean_rel_diff": np.mean(all_rel_diffs),
            "num_instances": len(all_rel_diffs),
            "all_below_threshold": all_pass
        },
        "notes": f"Tested {len(all_rel_diffs)} instances. Max rel diff: {max_rel_diff:.2e}. Control passed: {control_pass}."
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    main()

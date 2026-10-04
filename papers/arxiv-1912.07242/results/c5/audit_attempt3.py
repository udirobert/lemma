import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT_DIR = 'results/c5'
os.makedirs(OUT_DIR, exist_ok=True)


def compute_lhs(X, x):
    """Compute Tr((X_{n+1} X_{n+1}^T)^{-1}) where X_{n+1} = [X; x^T]."""
    Xn1 = np.vstack([X, x.reshape(1, -1)])
    A = Xn1 @ Xn1.T
    return np.trace(np.linalg.inv(A))


def compute_rhs(X, x):
    """Compute Tr((XX^T)^{-1}) + (1 + ||(X^T)^dagger x||^2) / ||Proj_{X^perp}(x)||^2."""
    n, d = X.shape
    A = X @ X.T
    inv_A = np.linalg.inv(A)
    tr_term = np.trace(inv_A)

    # (X^T)^dagger x = (X X^T)^{-1} X x  (since X has full row rank n < d)
    Xt_dag_x = inv_A @ (X @ x)
    num = 1.0 + np.dot(Xt_dag_x, Xt_dag_x)

    # Proj_{X^perp}(x) = x - X^T (X X^T)^{-1} X x
    proj_perp = x - X.T @ Xt_dag_x
    den = np.dot(proj_perp, proj_perp)

    return tr_term + num / den


def positive_control():
    """Positive control: use a known exact case.

    Take X = [1, 0, 0] (n=1, d=3), x = [0, 1, 0].
    Then X_{n+1} = [[1,0,0],[0,1,0]], X_{n+1} X_{n+1}^T = I_2, so LHS = Tr(I_2) = 2.

    RHS: Tr((XX^T)^{-1}) = Tr([1]^{-1}) = 1.
    (X^T)^dagger x = (X X^T)^{-1} X x = 1 * [1,0,0] @ [0,1,0] = 0. So ||...||^2 = 0.
    Proj_{X^perp}(x) = x - X^T (X X^T)^{-1} X x = [0,1,0] - [1,0,0]^T * 0 = [0,1,0]. ||...||^2 = 1.
    RHS = 1 + (1+0)/1 = 2.

    So LHS = RHS = 2 exactly.
    """
    X = np.array([[1.0, 0.0, 0.0]])
    x = np.array([0.0, 1.0, 0.0])
    lhs = compute_lhs(X, x)
    rhs = compute_rhs(X, x)
    rel_diff = abs(lhs - rhs) / max(abs(lhs), 1e-15)
    return rel_diff < 1e-10, lhs, rhs


def main():
    np.random.seed(42)

    # Positive control
    control_pass, ctrl_lhs, ctrl_rhs = positive_control()

    d = 1000
    n_values = [100, 300, 500, 700, 900]
    n_trials = 5

    all_rel_diffs = []
    per_n_max_diff = {}

    for n in n_values:
        max_diff = 0.0
        for trial in range(n_trials):
            X = np.random.randn(n, d)
            x = np.random.randn(d)

            lhs = compute_lhs(X, x)
            rhs = compute_rhs(X, x)

            rel_diff = abs(lhs - rhs) / max(abs(lhs), 1e-15)
            all_rel_diffs.append(rel_diff)
            if rel_diff > max_diff:
                max_diff = rel_diff
        per_n_max_diff[n] = max_diff

    max_rel_diff = max(all_rel_diffs) if all_rel_diffs else 0.0

    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    ns = list(per_n_max_diff.keys())
    diffs = [per_n_max_diff[n] for n in ns]
    ax.semilogy(ns, diffs, 'o-', label='Max rel. diff per n')
    ax.axhline(y=1e-8, color='r', linestyle='--', label='Threshold 1e-8')
    ax.set_xlabel('n (number of samples)')
    ax.set_ylabel('Max relative difference')
    ax.set_title('Claim 3: Trace update identity verification')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'fig.png'), dpi=150)
    plt.close()

    # Determine status
    if not control_pass:
        status = 'inconclusive'
        notes = f'Positive control failed (lhs={ctrl_lhs}, rhs={ctrl_rhs}). Statistic may be buggy.'
    elif max_rel_diff < 1e-8:
        status = 'supported'
        notes = f'All {len(all_rel_diffs)} instances satisfy rel diff < 1e-8. Max rel diff = {max_rel_diff:.2e}. Control passed.'
    else:
        status = 'falsified'
        notes = f'Max rel diff = {max_rel_diff:.2e} exceeds 1e-8 threshold. Per-n max: {per_n_max_diff}'

    metrics = {
        'control_pass': bool(control_pass),
        'control_lhs': float(ctrl_lhs),
        'control_rhs': float(ctrl_rhs),
        'max_rel_diff': float(max_rel_diff),
        'num_instances': len(all_rel_diffs),
        'per_n_max_diff': {str(k): float(v) for k, v in per_n_max_diff.items()},
        'd': d,
        'n_values': n_values,
        'n_trials_per_n': n_trials,
    }

    summary = {
        'claim_id': 'C5',
        'status': status,
        'metrics': metrics,
        'notes': notes,
    }

    print(f'SUMMARY_JSON={json.dumps(summary, default=str)}')


if __name__ == '__main__':
    main()

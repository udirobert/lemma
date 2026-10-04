import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_ranks(eigs, k):
    """
    Compute r_k, R_k, r_k(Sigma^2), and r_k^2(Sigma) for a given eigenvalue sequence and index k.

    Definitions from Lemma 5:
    r_k(Sigma) = sum_{i>k} lambda_i / lambda_{k+1}
    R_k(Sigma) = (sum_{i>k} lambda_i)^2 / sum_{i>k} lambda_i^2

    Note: The paper uses 1-based indexing for eigenvalues lambda_1 >= lambda_2 >= ...
    In the formula r_k = sum_{i>k} lambda_i / lambda_{k+1}, the index k+1 refers to the (k+1)-th eigenvalue.
    If we use 0-based indexing in numpy (eigs[0] is lambda_1), then:
    - lambda_{k+1} corresponds to eigs[k]
    - sum_{i>k} corresponds to sum(eigs[k+1:])

    Wait, let's re-read carefully.
    "lambda_i = mu_i(Sigma) for i = 1, 2, ..."
    "r_k(Sigma) = sum_{i>k} lambda_i / lambda_{k+1}"

    If k=0: r_0 = sum_{i>0} lambda_i / lambda_1 = (lambda_2 + lambda_3 + ...) / lambda_1.
    This matches the standard definition of effective rank often used.

    Let's check the identity r_k(Sigma^2) = r_k(Sigma) * R_k(Sigma) / r_k(Sigma)^2 ? No.
    The lemma says: r_k(Sigma^2) = r_k(Sigma) * R_k(Sigma) / r_k(Sigma)^2 ? No.
    Lemma 5: r_k(Sigma^2) = r_k(Sigma) * R_k(Sigma) / r_k(Sigma)^2 is not what it says.
    It says: r_k(Sigma^2) = r_k(Sigma) * R_k(Sigma) / r_k(Sigma)^2 ?
    Let's look at the text: "r_k(\Sigma^2) = r_k(\Sigma) R_k(\Sigma)"? No, that's dimensionally wrong if r is a ratio.
    Actually, let's derive it.
    Let S = sum_{i>k} lambda_i.
    Let Q = sum_{i>k} lambda_i^2.
    Let L = lambda_{k+1}.

    r_k(Sigma) = S / L.
    R_k(Sigma) = S^2 / Q.

    For Sigma^2, the eigenvalues are lambda_i^2.
    r_k(Sigma^2) = (sum_{i>k} lambda_i^2) / (lambda_{k+1})^2 = Q / L^2.

    Now check the identity in Lemma 5: "r_k(\Sigma^2) = r_k(\Sigma) R_k(\Sigma)"?
    r_k(Sigma) * R_k(Sigma) = (S/L) * (S^2/Q) = S^3 / (L Q). This is not Q/L^2.

    Let's re-read the excerpt carefully.
    "Lemma 5. r_k(\Sigma) \ge 1, r_k^2(\Sigma) = r_k(\Sigma^2) R_k(\Sigma), and ..."
    Wait, the text says: "r_k^2(\Sigma) = r_k(\Sigma^2) R_k(\Sigma)"?
    Let's check: r_k(Sigma)^2 = (S/L)^2 = S^2 / L^2.
    r_k(Sigma^2) * R_k(Sigma) = (Q/L^2) * (S^2/Q) = S^2 / L^2.
    Yes! The identity is r_k(Sigma)^2 = r_k(Sigma^2) * R_k(Sigma).

    So the claim to test is the inequality chain:
    r_k(Sigma^2) <= r_k(Sigma) <= R_k(Sigma) <= r_k(Sigma)^2

    Let's verify the middle inequality: r_k(Sigma) <= R_k(Sigma).
    S/L <= S^2/Q  =>  1/L <= S/Q  =>  Q <= S*L.
    Since lambda_i <= lambda_{k+1} = L for all i > k, we have lambda_i^2 <= lambda_i * L.
    Summing over i > k: sum lambda_i^2 <= sum lambda_i * L = L * sum lambda_i = L * S.
    So Q <= L*S holds. Thus r_k <= R_k.

    Left inequality: r_k(Sigma^2) <= r_k(Sigma).
    Q/L^2 <= S/L  =>  Q <= S*L. Same as above. Holds.

    Right inequality: R_k(Sigma) <= r_k(Sigma)^2.
    S^2/Q <= S^2/L^2  =>  1/Q <= 1/L^2  =>  Q >= L^2.
    Since lambda_i <= L, this is not obviously true. Wait.
    R_k = S^2/Q. r_k^2 = S^2/L^2.
    R_k <= r_k^2  <=>  S^2/Q <= S^2/L^2  <=>  L^2 <= Q.
    Is sum_{i>k} lambda_i^2 >= lambda_{k+1}^2?
    Yes, because the sum includes the term i=k+1, which is lambda_{k+1}^2, and all other terms are non-negative.
    So Q >= lambda_{k+1}^2 = L^2.
    Thus R_k <= r_k^2 holds.

    So the chain is:
    r_k(Sigma^2) <= r_k(Sigma) <= R_k(Sigma) <= r_k(Sigma)^2

    We need to compute these values for various k and eigenvalue spectra.
    """
    # eigs is 1D array, sorted descending. 0-indexed.
    # k is the index in the formula r_k.
    # In the formula, lambda_{k+1} is the pivot.
    # If k=0, pivot is lambda_1 = eigs[0]. Sum is eigs[1:].
    # If k=1, pivot is lambda_2 = eigs[1]. Sum is eigs[2:].

    if k >= len(eigs):
        return None, None, None, None

    pivot = eigs[k]
    if pivot == 0:
        return None, None, None, None

    tail = eigs[k+1:]
    if len(tail) == 0:
        # If there are no eigenvalues after k, the sums are 0.
        # r_k = 0/pivot = 0. R_k = 0/0 undefined?
        # Usually effective rank is defined for k such that tail is non-empty or handled by limits.
        # The paper says lambda_{k+1} > 0 for k >= 0. It implies we consider k where the tail exists.
        # If tail is empty, r_k = 0. R_k is 0/0. Let's assume k is small enough that tail is non-empty.
        # For the audit, we will pick k values where tail is non-empty.
        return 0.0, 0.0, 0.0, 0.0

    S = np.sum(tail)
    Q = np.sum(tail**2)

    if S == 0 or Q == 0:
        return 0.0, 0.0, 0.0, 0.0

    r_k = S / pivot
    R_k = (S**2) / Q
    r_k_sq = r_k**2

    # r_k(Sigma^2) = Q / pivot^2
    r_k_sigma2 = Q / (pivot**2)

    return r_k_sigma2, r_k, R_k, r_k_sq

def generate_eigenvalues(n, decay_type, seed=0):
    """
    Generate eigenvalues for a covariance matrix of dimension n.
    decay_type: 'exponential', 'power_law', 'geometric'
    """
    rng = np.random.RandomState(seed)

    if decay_type == 'exponential':
        # lambda_i = exp(-i/tau)
        tau = 10.0
        i = np.arange(1, n+1)
        eigs = np.exp(-i / tau)
        # Normalize so that sum is 1? Or just keep as is. The inequalities are scale invariant.
        # Let's normalize to sum=1 to keep numbers stable.
        eigs = eigs / np.sum(eigs)

    elif decay_type == 'power_law':
        # lambda_i = i^-alpha
        alpha = 1.5
        i = np.arange(1, n+1)
        eigs = i**(-alpha)
        eigs = eigs / np.sum(eigs)

    elif decay_type == 'geometric':
        # lambda_i = rho^i
        rho = 0.9
        i = np.arange(1, n+1)
        eigs = rho**i
        eigs = eigs / np.sum(eigs)

    else:
        raise ValueError("Unknown decay type")

    # Ensure descending order (they are generated descending)
    # Add a small amount of noise to break exact symmetries?
    # The claim is deterministic. Exact computation is preferred.
    # We will use the exact generated values.
    return eigs

def main():
    os.makedirs('results/c6', exist_ok=True)

    # Test parameters
    n_values = [200, 500]
    decay_types = ['exponential', 'power_law', 'geometric']
    k_values = [0, 1, 5, 10, 50, 100]

    all_metrics = []
    violations = 0
    total_checks = 0

    # Plot data storage
    plot_data = []

    for n in n_values:
        for decay in decay_types:
            eigs = generate_eigenvalues(n, decay, seed=42)

            for k in k_values:
                if k >= n - 1: # Need at least one eigenvalue in the tail
                    continue

                r_k_sigma2, r_k, R_k, r_k_sq = compute_ranks(eigs, k)

                if r_k_sigma2 is None:
                    continue

                # Check inequalities
                # 1. r_k(Sigma^2) <= r_k(Sigma)
                # 2. r_k(Sigma) <= R_k(Sigma)
                # 3. R_k(Sigma) <= r_k(Sigma)^2

                tol = 1e-10

                check1 = r_k_sigma2 <= r_k + tol
                check2 = r_k <= R_k + tol
                check3 = R_k <= r_k_sq + tol

                if not (check1 and check2 and check3):
                    violations += 1
                    print(f"Violation: n={n}, decay={decay}, k={k}")
                    print(f"  r_k(Sigma^2)={r_k_sigma2}, r_k(Sigma)={r_k}, R_k(Sigma)={R_k}, r_k(Sigma)^2={r_k_sq}")

                total_checks += 1

                # Store for plotting (sample a few)
                if decay == 'power_law' and n == 500:
                    plot_data.append((k, r_k_sigma2, r_k, R_k, r_k_sq))

    # Positive Control
    # The claim is an identity/inequality derived from algebra.
    # A positive control for the *computation* is to verify the identity r_k(Sigma)^2 = r_k(Sigma^2) * R_k(Sigma).
    # If this identity holds, and the individual inequalities hold, the chain is robust.
    # Let's pick a specific case and verify the identity.

    n_ctrl = 100
    eigs_ctrl = generate_eigenvalues(n_ctrl, 'power_law', seed=123)
    k_ctrl = 10
    r_k_sigma2_c, r_k_c, R_k_c, r_k_sq_c = compute_ranks(eigs_ctrl, k_ctrl)

    # Identity: r_k(Sigma)^2 = r_k(Sigma^2) * R_k(Sigma)
    lhs_identity = r_k_sq_c
    rhs_identity = r_k_sigma2_c * R_k_c

    identity_error = abs(lhs_identity - rhs_identity)
    control_pass = identity_error < 1e-10

    # Also verify the inequalities for the control case
    ctrl_check1 = r_k_sigma2_c <= r_k_c + 1e-10
    ctrl_check2 = r_k_c <= R_k_c + 1e-10
    ctrl_check3 = R_k_c <= r_k_sq_c + 1e-10
    ctrl_ineq_pass = ctrl_check1 and ctrl_check2 and ctrl_check3

    # If the control fails, the statistic is buggy.
    if not control_pass:
        status = "inconclusive"
        notes = f"Positive control failed: identity error {identity_error}."
    else:
        if violations == 0:
            status = "supported"
            notes = f"All {total_checks} checks passed. Identity verified in control."
        else:
            status = "falsified"
            notes = f"{violations} violations found out of {total_checks} checks."

    # Plotting
    if plot_data:
        ks = [d[0] for d in plot_data]
        r2s = [d[1] for d in plot_data]
        rs = [d[2] for d in plot_data]
        Rs = [d[3] for d in plot_data]
        r2sq = [d[4] for d in plot_data]

        plt.figure(figsize=(10, 6))
        plt.plot(ks, r2s, label=r'$r_k(\Sigma^2)$', marker='o')
        plt.plot(ks, rs, label=r'$r_k(\Sigma)$', marker='s')
        plt.plot(ks, Rs, label=r'$R_k(\Sigma)$', marker='^')
        plt.plot(ks, r2sq, label=r'$r_k(\Sigma)^2$', marker='d')
        plt.xlabel('k')
        plt.ylabel('Effective Rank')
        plt.title('Effective Rank Inequalities (Power Law, n=500)')
        plt.legend()
        plt.grid(True)
        plt.savefig('results/c6/fig.png', dpi=150, bbox_inches='tight')
        plt.close()

    summary = {
        "claim_id": "C6",
        "status": status,
        "metrics": {
            "total_checks": total_checks,
            "violations": violations,
            "control_pass": control_pass,
            "identity_error": identity_error,
            "ctrl_ineq_pass": ctrl_ineq_pass
        },
        "notes": notes
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    main()

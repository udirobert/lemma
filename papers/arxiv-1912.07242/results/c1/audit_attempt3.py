import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_mse(n, d, sigma, beta, rng):
    """
    Compute the test MSE for a single trial with n samples.
    Test MSE = ||beta_hat - beta||^2 + sigma^2
    """
    # Generate training data
    X = rng.normal(0, 1, size=(n, d))
    y = X @ beta + rng.normal(0, sigma, size=n)

    # Compute minimum-norm solution beta_hat = X^dagger y
    # Use SVD for numerical stability
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Pseudoinverse: X^dagger = Vt^T diag(1/s) U^T
    # We only need X^dagger y = Vt^T diag(1/s) (U^T y)
    Ut_y = U.T @ y

    # Handle zero singular values (shouldn't happen for Gaussian, but good practice)
    # For n < d, there are d-n zero singular values in the full SVD, but we use reduced SVD
    # In reduced SVD, s has length min(n, d)
    # If n < d, s has length n, all non-zero with prob 1
    # If n >= d, s has length d, all non-zero with prob 1

    # Compute 1/s, avoiding division by zero
    s_inv = np.zeros_like(s)
    nonzero = s > 1e-10
    s_inv[nonzero] = 1.0 / s[nonzero]

    # beta_hat = Vt.T @ (s_inv * Ut_y)
    beta_hat = Vt.T @ (s_inv * Ut_y)

    # Compute test MSE
    mse = np.sum((beta_hat - beta) ** 2) + sigma ** 2
    return mse

def main():
    # Parameters from the claim
    d = 1000
    sigma = 0.1
    T = 50  # Number of trials

    # Set up random number generator
    rng = np.random.default_rng(42)

    # Generate a fixed beta with ||beta||_2 = 1
    beta = rng.normal(0, 1, size=d)
    beta = beta / np.linalg.norm(beta)

    # Range of n to test
    # We need to check around n=d, so let's go from 1 to 2d
    # But to save time, we can be smart about it.
    # The claim is about the peak at n=d.
    # Let's test n from 1 to 2000.
    # However, computing SVD for n=1..2000 with d=1000 for 50 trials each is expensive.
    # Total SVDs: 2000 * 50 = 100,000 SVDs of size up to 2000x1000.
    # This might be too slow.

    # Let's optimize:
    # 1. We only need to check if n=d is a local maximum.
    # 2. We can sample n more sparsely, but we need to be sure about the peak.
    # 3. Let's try a smaller range first to see if it's feasible, or optimize the computation.

    # Actually, let's just do it but with a smaller T or smaller range if it's too slow.
    # But the claim says T=50. Let's stick to T=50 but maybe reduce the range of n.
    # The peak is at n=d. Let's check n from 500 to 1500 to capture the peak and the descent on both sides.
    # But the claim says "decreases for n<d, increases to a peak at n=d, and decreases for n>d".
    # So we should check the whole range to see the non-monotonicity.

    # Let's try to be efficient. We'll compute for n in a range that covers the peak.
    # Let's do n from 100 to 1900, step 10, plus the critical points d-1, d, d+1.
    # This reduces the number of SVDs significantly.

    n_values = list(range(100, 1901, 10))
    # Ensure we have d-1, d, d+1
    for critical in [d-1, d, d+1]:
        if critical not in n_values:
            n_values.append(critical)
    n_values = sorted(set(n_values))

    mse_means = []
    mse_stds = []

    for n in n_values:
        mses = []
        for t in range(T):
            mse = compute_mse(n, d, sigma, beta, rng)
            mses.append(mse)
        mse_means.append(np.mean(mses))
        mse_stds.append(np.std(mses))

        # Print progress
        if n % 100 == 0:
            print(f"n={n}, mean_mse={np.mean(mses):.4f}")

    mse_means = np.array(mse_means)
    mse_stds = np.array(mse_stds)

    # Find the index of n=d
    idx_d = n_values.index(d)
    mse_at_d = mse_means[idx_d]

    # Check if n=d is a local maximum
    # We need to check n=d-1 and n=d+1
    idx_d_minus_1 = n_values.index(d-1)
    idx_d_plus_1 = n_values.index(d+1)

    mse_at_d_minus_1 = mse_means[idx_d_minus_1]
    mse_at_d_plus_1 = mse_means[idx_d_plus_1]

    # Check if mse_at_d is greater than both neighbors
    is_local_max = (mse_at_d > mse_at_d_minus_1) and (mse_at_d > mse_at_d_plus_1)

    # Also check if the overall curve is non-monotonic
    # We can check if there's a peak somewhere in the middle
    # For simplicity, let's just check if the maximum is not at the boundaries
    max_idx = np.argmax(mse_means)
    is_non_monotonic = (max_idx > 0) and (max_idx < len(mse_means) - 1)

    # Positive control:
    # We know that for n >> d, the MSE should be close to sigma^2.
    # For n << d, the MSE should be larger.
    # Let's check if the MSE at n=1000 is significantly larger than at n=2000.
    # This is a sanity check that the phenomenon is present.

    # Let's also compute the theoretical prediction for comparison
    # For n < d: E[R] ≈ (1 - n/d)||beta||^2 + sigma^2 * (n/d) / (1 - n/d)
    # For n > d: E[R] ≈ sigma^2 / (n/d - 1)

    theoretical_mse = []
    for n in n_values:
        gamma = n / d
        if gamma < 1:
            # Overparameterized
            # Bias = (1 - gamma)^2 ||beta||^2
            # Variance ≈ gamma(1-gamma)||beta||^2 + sigma^2 * gamma/(1-gamma)
            # Total = Bias + Variance + sigma^2 (wait, the formula in the paper is for excess risk)
            # The paper says: E[R] ≈ (1 - gamma)||beta||^2 + sigma^2 * gamma/(1-gamma)
            # But R includes sigma^2. Let's check.
            # R = ||beta_hat - beta||^2 + sigma^2
            # E[R] = E[||beta_hat - beta||^2] + sigma^2 = Bias + Variance + sigma^2
            # The paper's formula (7) is for excess risk, which is E[R] - sigma^2.
            # So E[R] = (1 - gamma)||beta||^2 + sigma^2 * gamma/(1-gamma) + sigma^2
            # Let's use this.
            excess_risk = (1 - gamma) * np.sum(beta**2) + sigma**2 * gamma / (1 - gamma)
            theoretical_mse.append(excess_risk + sigma**2)
        else:
            # Underparameterized
            # Bias = 0
            # Variance ≈ sigma^2 / (gamma - 1)
            # E[R] = Variance + sigma^2
            excess_risk = sigma**2 / (gamma - 1)
            theoretical_mse.append(excess_risk + sigma**2)

    theoretical_mse = np.array(theoretical_mse)

    # Plot
    os.makedirs('results/c1', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.errorbar(n_values, mse_means, yerr=mse_stds, fmt='o-', label='Empirical (mean ± std)')
    plt.plot(n_values, theoretical_mse, 'r-', label='Theoretical')
    plt.axvline(x=d, color='k', linestyle='--', label=f'd={d}')
    plt.xlabel('Number of training samples n')
    plt.ylabel('Test MSE')
    plt.title('Sample-wise Double Descent')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/c1/fig.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Determine status
    if is_local_max and is_non_monotonic:
        status = "supported"
    else:
        status = "falsified"

    # Positive control check
    # The MSE at n=d should be much larger than at n=2d
    # Let's check if mse_at_d > mse_at_2d
    if 2*d in n_values:
        idx_2d = n_values.index(2*d)
        mse_at_2d = mse_means[idx_2d]
        control_pass = mse_at_d > mse_at_2d
    else:
        # If 2d is not in n_values, use the last value
        mse_at_2d = mse_means[-1]
        control_pass = mse_at_d > mse_at_2d

    if not control_pass:
        status = "inconclusive"
        notes = "Positive control failed: MSE at n=d is not greater than MSE at n=2d."
    else:
        notes = f"MSE at n=d is {mse_at_d:.4f}, at n=d-1 is {mse_at_d_minus_1:.4f}, at n=d+1 is {mse_at_d_plus_1:.4f}. Local max: {is_local_max}, Non-monotonic: {is_non_monotonic}."

    summary = {
        "claim_id": "C1",
        "status": status,
        "metrics": {
            "mse_at_d": float(mse_at_d),
            "mse_at_d_minus_1": float(mse_at_d_minus_1),
            "mse_at_d_plus_1": float(mse_at_d_plus_1),
            "is_local_max": bool(is_local_max),
            "is_non_monotonic": bool(is_non_monotonic),
            "control_pass": bool(control_pass)
        },
        "notes": notes
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    main()

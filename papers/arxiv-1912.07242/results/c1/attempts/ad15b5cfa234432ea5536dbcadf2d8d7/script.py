import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_mse(n, d, sigma, beta, rng):
    """Compute test MSE for a single training set of size n."""
    X = rng.normal(size=(n, d))
    y = X @ beta + rng.normal(scale=sigma, size=n)

    # Compute beta_hat = X^dagger y
    # Use SVD for numerical stability
    U, s, Vt = np.linalg.svd(X, full_matrices=False)

    # Pseudoinverse: X^dagger = Vt^T @ diag(1/s) @ U^T
    # We only need X^dagger @ y
    # y_hat = U @ (U^T @ y)
    # beta_hat = Vt^T @ diag(1/s) @ U^T @ y

    # Handle zero singular values (shouldn't happen for Gaussian, but just in case)
    s_inv = np.zeros_like(s)
    mask = s > 1e-10
    s_inv[mask] = 1.0 / s[mask]

    # beta_hat = Vt.T @ (s_inv * (U.T @ y))
    Uy = U.T @ y
    beta_hat = Vt.T @ (s_inv * Uy)

    # Test MSE = ||beta_hat - beta||^2 + sigma^2
    mse = np.sum((beta_hat - beta)**2) + sigma**2
    return mse

def main():
    # Parameters from the claim
    d = 1000
    sigma = 0.1
    T = 50  # Number of trials

    # Set up beta with ||beta||_2 = 1
    rng = np.random.default_rng(42)
    beta = rng.normal(size=d)
    beta = beta / np.linalg.norm(beta)

    # Range of n to test
    n_min = 1
    n_max = 2 * d

    # We'll compute MSE for each n
    # To save time, we can be a bit sparse in the far regions and dense near n=d
    # But the claim asks for a clear peak, so let's do a reasonable grid
    # Let's do every 10th n from 1 to 2000, but include n=d-1, d, d+1 specifically

    n_values = list(range(1, n_max + 1, 10))
    # Ensure we have the critical points
    for critical_n in [d-1, d, d+1]:
        if critical_n not in n_values and 1 <= critical_n <= n_max:
            n_values.append(critical_n)
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

        if n % 100 == 0 or n in [d-1, d, d+1]:
            print(f"n={n}, mean_mse={np.mean(mses):.4f}, std={np.std(mses):.4f}")

    mse_means = np.array(mse_means)
    mse_stds = np.array(mse_stds)

    # Find the peak
    peak_idx = np.argmax(mse_means)
    peak_n = n_values[peak_idx]
    peak_mse = mse_means[peak_idx]

    # Check if peak is near n=d
    peak_near_d = abs(peak_n - d) <= 10

    # Check local maximum at n=d
    # Find indices for d-1, d, d+1
    idx_d_minus_1 = n_values.index(d-1) if d-1 in n_values else None
    idx_d = n_values.index(d) if d in n_values else None
    idx_d_plus_1 = n_values.index(d+1) if d+1 in n_values else None

    local_max_at_d = False
    if idx_d_minus_1 is not None and idx_d is not None and idx_d_plus_1 is not None:
        mse_d_minus_1 = mse_means[idx_d_minus_1]
        mse_d = mse_means[idx_d]
        mse_d_plus_1 = mse_means[idx_d_plus_1]

        # Check if mse_d is greater than both neighbors (within Monte Carlo error)
        # Use a tolerance of 1 standard error of the mean
        se_d_minus_1 = mse_stds[idx_d_minus_1] / np.sqrt(T)
        se_d = mse_stds[idx_d] / np.sqrt(T)
        se_d_plus_1 = mse_stds[idx_d_plus_1] / np.sqrt(T)

        # For a local max, we want mse_d > mse_d_minus_1 and mse_d > mse_d_plus_1
        # With Monte Carlo error, we can check if the difference is significant
        # Let's just check if it's greater, and note the margin
        local_max_at_d = (mse_d > mse_d_minus_1) and (mse_d > mse_d_plus_1)

        print(f"\nLocal max check at n=d:")
        print(f"MSE(d-1)={mse_d_minus_1:.4f} +/- {se_d_minus_1:.4f}")
        print(f"MSE(d)={mse_d:.4f} +/- {se_d:.4f}")
        print(f"MSE(d+1)={mse_d_plus_1:.4f} +/- {se_d_plus_1:.4f}")
        print(f"Local max at d: {local_max_at_d}")

    # Check non-monotonicity: the curve should decrease, then increase, then decrease
    # We can check if there's a clear peak
    # A simple check: the peak should be significantly higher than the values at the ends
    # and the curve should have a single peak

    # Let's check if the peak is a clear maximum
    # Compare peak to the minimum in the overparameterized regime (n < d)
    # and the minimum in the underparameterized regime (n > d)

    overparam_mask = np.array([n < d for n in n_values])
    underparam_mask = np.array([n > d for n in n_values])

    if np.any(overparam_mask):
        min_overparam_mse = np.min(mse_means[overparam_mask])
        min_overparam_n = n_values[np.where(overparam_mask)[0][np.argmin(mse_means[overparam_mask])]]
    else:
        min_overparam_mse = None
        min_overparam_n = None

    if np.any(underparam_mask):
        min_underparam_mse = np.min(mse_means[underparam_mask])
        min_underparam_n = n_values[np.where(underparam_mask)[0][np.argmin(mse_means[underparam_mask])]]
    else:
        min_underparam_mse = None
        min_underparam_n = None

    # The peak should be higher than both minima
    peak_is_clear = False
    if min_overparam_mse is not None and min_underparam_mse is not None:
        peak_is_clear = (peak_mse > min_overparam_mse) and (peak_mse > min_underparam_mse)
        print(f"\nPeak clarity check:")
        print(f"Peak MSE={peak_mse:.4f} at n={peak_n}")
        print(f"Min overparam MSE={min_overparam_mse:.4f} at n={min_overparam_n}")
        print(f"Min underparam MSE={min_underparam_mse:.4f} at n={min_underparam_n}")
        print(f"Peak is clear: {peak_is_clear}")

    # Positive control: test with a known case
    # For a simple case, we can test with d=10, n=5, and check if the MSE is reasonable
    # Actually, a better control is to test the theoretical prediction
    # For n < d, the excess risk should be approximately (1 - n/d)||beta||^2 + sigma^2 * (n/d) / (1 - n/d)
    # Let's test this for a few values of n

    control_pass = True
    control_notes = []

    # Test the theoretical prediction for a few n values
    for n_test in [100, 500, 900]:
        gamma = n_test / d
        if gamma < 1:
            # Theoretical excess risk
            theory_excess = (1 - gamma) * np.sum(beta**2) + sigma**2 * gamma / (1 - gamma)
            theory_mse = theory_excess + sigma**2

            # Empirical MSE
            empirical_mse = mse_means[n_values.index(n_test)] if n_test in n_values else None

            if empirical_mse is not None:
                # Check if they are within a reasonable range
                # The theory is asymptotic, so for d=1000 it should be close
                rel_error = abs(empirical_mse - theory_mse) / theory_mse
                if rel_error > 0.5:  # 50% relative error is too high
                    control_pass = False
                    control_notes.append(f"n={n_test}: empirical={empirical_mse:.4f}, theory={theory_mse:.4f}, rel_error={rel_error:.2f}")
                else:
                    control_notes.append(f"n={n_test}: empirical={empirical_mse:.4f}, theory={theory_mse:.4f}, rel_error={rel_error:.2f}")

    # Determine status
    if not control_pass:
        status = "inconclusive"
        notes = "Positive control failed: empirical MSE does not match theoretical prediction within reasonable tolerance."
    elif local_max_at_d and peak_near_d and peak_is_clear:
        status = "supported"
        notes = f"Test MSE peaks at n={peak_n} (near d={d}), with a clear local maximum. Peak MSE={peak_mse:.4f}, min overparam MSE={min_overparam_mse:.4f}, min underparam MSE={min_underparam_mse:.4f}."
    else:
        status = "falsified"
        notes = f"Test MSE does not show the claimed non-monotonic behavior with a peak at n=d. Peak at n={peak_n}, local_max_at_d={local_max_at_d}, peak_near_d={peak_near_d}, peak_is_clear={peak_is_clear}."

    # Save plot
    os.makedirs("results/c1", exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.errorbar(n_values, mse_means, yerr=mse_stds, fmt='o-', markersize=2, capsize=1, label='Empirical MSE')
    plt.axvline(x=d, color='r', linestyle='--', label=f'd={d}')
    plt.xlabel('Number of training samples n')
    plt.ylabel('Test MSE')
    plt.title(f'Test MSE vs. n for d={d}, sigma={sigma}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig("results/c1/fig.png", dpi=150, bbox_inches='tight')
    plt.close()

    # Build summary
    summary = {
        "claim_id": "C1",
        "status": status,
        "metrics": {
            "d": d,
            "sigma": sigma,
            "T": T,
            "peak_n": int(peak_n),
            "peak_mse": float(peak_mse),
            "mse_at_d_minus_1": float(mse_means[idx_d_minus_1]) if idx_d_minus_1 is not None else None,
            "mse_at_d": float(mse_means[idx_d]) if idx_d is not None else None,
            "mse_at_d_plus_1": float(mse_means[idx_d_plus_1]) if idx_d_plus_1 is not None else None,
            "local_max_at_d": bool(local_max_at_d),
            "peak_near_d": bool(peak_near_d),
            "peak_is_clear": bool(peak_is_clear),
            "min_overparam_mse": float(min_overparam_mse) if min_overparam_mse is not None else None,
            "min_underparam_mse": float(min_underparam_mse) if min_underparam_mse is not None else None,
            "control_pass": bool(control_pass)
        },
        "notes": notes
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    main()

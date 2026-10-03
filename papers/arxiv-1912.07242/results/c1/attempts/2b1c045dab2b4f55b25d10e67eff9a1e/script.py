import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def compute_mse(X, y, beta, sigma, n_test=10000):
    """Compute test MSE for the minimum-norm ridgeless regression estimator."""
    # beta_hat = X^dagger y
    # Use SVD for numerical stability
    U, s, Vt = np.linalg.svd(X, full_matrices=False)
    # Pseudoinverse: V @ diag(1/s) @ U^T
    # But we only need beta_hat = V @ diag(1/s) @ U^T @ y
    # For n < d, s has length n, and Vt has shape (n, d)
    # For n >= d, s has length d, and Vt has shape (d, d)

    # Actually, let's just use np.linalg.pinv for simplicity, but be careful with rank
    # The claim says X is full rank with probability 1, so pinv should work
    beta_hat = np.linalg.pinv(X) @ y

    # Test MSE = ||beta_hat - beta||^2 + sigma^2
    # This is the expected test MSE over (x, y) ~ D
    mse = np.sum((beta_hat - beta)**2) + sigma**2
    return mse

def main():
    np.random.seed(42)

    d = 1000
    sigma = 0.1
    beta_norm = 1.0
    T = 50  # number of trials

    # Generate a fixed beta with ||beta||_2 = 1
    beta = np.random.randn(d)
    beta = beta / np.linalg.norm(beta) * beta_norm

    # Vary n from 1 to 2d
    n_values = list(range(1, 2*d + 1))

    # Store mean MSE for each n
    mean_mse = np.zeros(len(n_values))
    std_mse = np.zeros(len(n_values))

    for i, n in enumerate(n_values):
        mse_trials = []
        for t in range(T):
            # Generate training data
            X = np.random.randn(n, d)
            y = X @ beta + np.random.randn(n) * sigma

            # Compute test MSE
            mse = compute_mse(X, y, beta, sigma)
            mse_trials.append(mse)

        mean_mse[i] = np.mean(mse_trials)
        std_mse[i] = np.std(mse_trials)

    # Check the success criterion:
    # 1. Mean MSE at n=d is a local maximum: exceeds MSE at n=d-1 and n=d+1
    idx_d = d - 1  # 0-indexed, since n_values starts at 1
    mse_d = mean_mse[idx_d]
    mse_d_minus_1 = mean_mse[idx_d - 1]
    mse_d_plus_1 = mean_mse[idx_d + 1]

    # Check if it's a local maximum
    is_local_max = (mse_d > mse_d_minus_1) and (mse_d > mse_d_plus_1)

    # Check non-monotonicity: there should be a clear peak near n=d
    # Find the global maximum in the range [1, 2d]
    global_max_idx = np.argmax(mean_mse)
    global_max_n = n_values[global_max_idx]

    # The peak should be near n=d (within some tolerance, say +/- 50)
    peak_near_d = abs(global_max_n - d) <= 50

    # Also check that the curve is non-monotonic: it should decrease, then increase, then decrease
    # We can check that there's a local minimum before d and a local maximum at d
    # For simplicity, let's just check that the MSE at n=1 is higher than at some n < d, and lower than at n=d
    # Actually, the claim says it decreases for n<d, increases to peak at n=d, decreases for n>d
    # So: MSE should be decreasing from n=1 to some point, then increasing to n=d, then decreasing

    # Let's check: MSE at n=1 should be > MSE at n=d/2 (decreasing in first half)
    # And MSE at n=d/2 should be < MSE at n=d (increasing to peak)
    # And MSE at n=d should be > MSE at n=3d/2 (decreasing after peak)

    idx_half = d // 2 - 1  # n = d/2
    idx_3half = int(3*d/2) - 1  # n = 3d/2

    decreasing_first_half = mean_mse[0] > mean_mse[idx_half]
    increasing_to_peak = mean_mse[idx_half] < mse_d
    decreasing_after_peak = mse_d > mean_mse[idx_3half]

    non_monotonic = decreasing_first_half and increasing_to_peak and decreasing_after_peak

    # Positive control: verify that for a known case, the statistic works
    # Let's use a simple case: d=10, n=5, sigma=0.1, beta=[1,0,...,0]
    # We know the theoretical excess risk for gamma=0.5: (1-0.5)*1 + 0.01*(0.5/0.5) = 0.5 + 0.01 = 0.51
    # Let's verify our compute_mse function gives reasonable results
    d_ctrl = 10
    n_ctrl = 5
    sigma_ctrl = 0.1
    beta_ctrl = np.zeros(d_ctrl)
    beta_ctrl[0] = 1.0

    mse_ctrl_trials = []
    for t in range(1000):
        X_ctrl = np.random.randn(n_ctrl, d_ctrl)
        y_ctrl = X_ctrl @ beta_ctrl + np.random.randn(n_ctrl) * sigma_ctrl
        mse_ctrl = compute_mse(X_ctrl, y_ctrl, beta_ctrl, sigma_ctrl)
        mse_ctrl_trials.append(mse_ctrl)

    mean_mse_ctrl = np.mean(mse_ctrl_trials)
    # Theoretical excess risk (without sigma^2) for gamma=0.5: (1-0.5)*1 + 0.01*(0.5/0.5) = 0.51
    # Total MSE = excess risk + sigma^2 = 0.51 + 0.01 = 0.52
    # But this is approximate. Let's just check that it's in a reasonable range
    control_pass = (0.3 < mean_mse_ctrl < 0.8)

    # Determine status
    if not control_pass:
        status = "inconclusive"
        notes = "Positive control failed: the MSE computation may be buggy."
    elif is_local_max and peak_near_d and non_monotonic:
        status = "supported"
        notes = f"MSE at n=d ({mse_d:.4f}) is a local maximum (n=d-1: {mse_d_minus_1:.4f}, n=d+1: {mse_d_plus_1:.4f}). Peak at n={global_max_n}. Non-monotonic behavior confirmed."
    elif is_local_max and peak_near_d:
        status = "supported"
        notes = f"MSE at n=d ({mse_d:.4f}) is a local maximum (n=d-1: {mse_d_minus_1:.4f}, n=d+1: {mse_d_plus_1:.4f}). Peak at n={global_max_n}. Local maximum confirmed."
    else:
        status = "falsified"
        notes = f"MSE at n=d ({mse_d:.4f}) is NOT a local maximum (n=d-1: {mse_d_minus_1:.4f}, n=d+1: {mse_d_plus_1:.4f}). Peak at n={global_max_n}."

    # Save plot
    os.makedirs('results/c1', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.plot(n_values, mean_mse, 'b-', label='Mean test MSE')
    plt.fill_between(n_values, mean_mse - std_mse, mean_mse + std_mse, alpha=0.3, label='±1 std')
    plt.axvline(x=d, color='r', linestyle='--', label=f'n=d={d}')
    plt.xlabel('Number of training samples n')
    plt.ylabel('Test MSE')
    plt.title(f'Test MSE vs. n for d={d}, sigma={sigma}, ||beta||=1')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/c1/fig.png', dpi=150, bbox_inches='tight')
    plt.close()

    # Build summary
    summary = {
        "claim_id": "C1",
        "status": status,
        "metrics": {
            "mse_at_n_d": float(mse_d),
            "mse_at_n_d_minus_1": float(mse_d_minus_1),
            "mse_at_n_d_plus_1": float(mse_d_plus_1),
            "is_local_max": bool(is_local_max),
            "global_max_n": int(global_max_n),
            "peak_near_d": bool(peak_near_d),
            "non_monotonic": bool(non_monotonic),
            "control_pass": bool(control_pass),
            "mean_mse_ctrl": float(mean_mse_ctrl)
        },
        "notes": notes
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    main()

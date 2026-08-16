import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# Constants
DIMS = [2, 5, 10]
N_TRAIN = 5000
N_TEST = 50000
EPSILON = 0.2
LR = 0.01
MAX_STEPS = 20000

# Ensure results directory exists
os.makedirs("results/c2", exist_ok=True)


# Helper: Sample from D-dimensional uniform ball
# We use rejection sampling for simplicity and correctness.
def sample_ball(D, N, center, radius=1.0):
    """
    Sample N points uniformly from a D-dimensional ball of given radius centered at 'center'.
    """
    points = np.zeros((N, D))
    # Rejection sampling
    count = 0
    while count < N:
        # Sample from bounding box [-radius, radius]^D
        candidates = np.random.uniform(-radius, radius, size=(N, D))
        # Check if inside ball
        norms_sq = np.sum(candidates**2, axis=1)
        mask = norms_sq <= radius**2
        valid = candidates[mask]
        if len(valid) > 0:
            take = min(len(valid), N - count)
            points[count : count + take] = valid[:take]
            count += take
    points += center
    return points


def train_perceptron(
    X_train, y_train, X_test, y_test, D, lr=0.01, max_steps=20000, epsilon=0.2
):
    """
    Train a perceptron f(x) = sgn(w . x) using gradient descent on 0/1 loss.
    Returns test error history and training error history.
    """
    w = np.zeros(D)
    test_errors = []
    train_errors = []

    # Precompute test labels for speed
    # y_test is +1 or -1

    for step in range(max_steps):
        # Compute predictions on training set
        # y_pred = sign(w . x)
        # To avoid sign(0) issues, we can use a small epsilon or just standard sign.
        # Gradient of 0/1 loss is -y * x if misclassified, 0 otherwise.

        # Vectorized update
        # scores = X_train @ w
        # preds = np.sign(scores)
        # # Handle zeros: if score is 0, sign is 0. Let's treat 0 as -1 or +1?
        # # Standard perceptron rule: if y * (w.x) <= 0, update.
        # # Let's use the condition: if y_i * (w . x_i) <= 0, then w += lr * y_i * x_i

        scores = X_train @ w
        # Misclassified if y * score <= 0
        misclassified = (y_train * scores) <= 0

        if np.any(misclassified):
            # Update w
            # w = w + lr * sum_{i in misclassified} y_i * x_i
            # Note: The paper uses R = 1/(2N) sum ... so gradient is 1/N sum ...
            # The test plan says "gradient descent". Standard perceptron update is w += eta * y * x.
            # Let's stick to standard perceptron update rule which is equivalent to GD on 0/1 loss with step size eta.
            w += lr * np.sum(y_train[misclassified] * X_train[misclassified], axis=0)

        # Record errors every 10 steps to save memory/time
        if step % 10 == 0:
            # Training error
            train_scores = X_train @ w
            train_preds = np.sign(train_scores)
            # If score is 0, sign is 0. Let's count 0 as error or handle carefully.
            # Usually, if score is 0, it's on the boundary. Let's say error if y * score <= 0.
            train_err = np.mean((y_train * train_scores) <= 0)
            train_errors.append(train_err)

            # Test error
            test_scores = X_test @ w
            test_err = np.mean((y_test * test_scores) <= 0)
            test_errors.append(test_err)

            # Early stopping if train error is 0 and test error is 0
            if train_err == 0.0 and test_err == 0.0:
                break

    return np.array(test_errors), np.array(train_errors)


def fit_exponent(t, E, t_epsilon, expected_exp):
    """
    Fit E(t) = A * (t_epsilon - t)^nu for t < t_epsilon.
    Returns fitted nu and R^2.
    """
    # Filter points where t < t_epsilon and E > 0
    mask = (t < t_epsilon) & (E > 1e-6)
    if np.sum(mask) < 5:
        return None, None

    t_fit = t[mask]
    E_fit = E[mask]

    # Log-log fit: log(E) = log(A) + nu * log(t_epsilon - t)
    x = np.log(t_epsilon - t_fit)
    y = np.log(E_fit)

    # Linear regression
    A_mat = np.vstack([x, np.ones_like(x)]).T
    try:
        coeffs, residuals, rank, s = np.linalg.lstsq(A_mat, y, rcond=None)
        nu = coeffs[0]
        log_A = coeffs[1]

        # Calculate R^2
        y_pred = A_mat @ coeffs
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r_squared = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0

        return nu, r_squared
    except:
        return None, None


# Positive Control
# We simulate a scenario where the exponent is known to be 1.5 (D=2).
# We generate synthetic data E(t) = (t_eps - t)^1.5 + noise.
# We fit it and check if we recover 1.5.
def run_positive_control():
    D = 2
    expected_exp = (D + 1) / 2.0  # 1.5
    t_eps = 100.0
    t = np.linspace(0, t_eps - 0.1, 100)
    E = (t_eps - t) ** expected_exp + np.random.normal(
        0, 0.01 * np.max((t_eps - t) ** expected_exp), size=len(t)
    )
    E = np.maximum(E, 1e-6)  # Ensure positive

    nu, r2 = fit_exponent(t, E, t_eps, expected_exp)

    if nu is None:
        return False

    # Check if close to 1.5
    return abs(nu - expected_exp) < 0.1


control_pass = run_positive_control()

# Main Experiment
results = {}
fig, axes = plt.subplots(1, len(DIMS), figsize=(15, 5))
if len(DIMS) == 1:
    axes = [axes]

for i, D in enumerate(DIMS):
    expected_exp = (D + 1) / 2.0

    # Generate Data
    # Positive class: Ball centered at (+epsilon, 0, ..., 0)
    # Negative class: Ball centered at (-epsilon, 0, ..., 0)
    # Radius 1

    # Note: The paper mentions "shifted by +/- epsilon".
    # If epsilon is small, the balls overlap significantly.
    # The "grokking" transition happens when the classifier finds the separating hyperplane.

    center_pos = np.zeros(D)
    center_pos[0] = EPSILON
    center_neg = np.zeros(D)
    center_neg[0] = -EPSILON

    X_pos = sample_ball(D, N_TRAIN, center_pos)
    X_neg = sample_ball(D, N_TRAIN, center_neg)

    X_train = np.vstack([X_pos, X_neg])
    y_train = np.concatenate([np.ones(N_TRAIN), -np.ones(N_TRAIN)])

    # Shuffle
    idx = np.random.permutation(2 * N_TRAIN)
    X_train = X_train[idx]
    y_train = y_train[idx]

    # Test Data
    X_test_pos = sample_ball(D, N_TEST, center_pos)
    X_test_neg = sample_ball(D, N_TEST, center_neg)
    X_test = np.vstack([X_test_pos, X_test_neg])
    y_test = np.concatenate([np.ones(N_TEST), -np.ones(N_TEST)])

    # Train
    test_errors, train_errors = train_perceptron(
        X_train, y_train, X_test, y_test, D, lr=LR, max_steps=MAX_STEPS, epsilon=EPSILON
    )

    # Determine t_epsilon
    # t_epsilon is the time when test error drops to 0 (or near 0).
    # In the paper, it's the critical time.
    # We identify it as the first time test error becomes < 0.01.
    # If it never happens, we might need to adjust or mark as inconclusive.

    threshold = 0.01
    below_thresh = np.where(test_errors < threshold)[0]

    if len(below_thresh) == 0:
        # Grokking didn't happen or took too long
        results[D] = {
            "status": "inconclusive",
            "reason": "No grokking observed",
            "expected_exp": expected_exp,
        }
        axes[i].plot(test_errors)
        axes[i].set_title(f"D={D}: No Grokking")
        continue

    t_epsilon_idx = below_thresh[0]
    t_epsilon = t_epsilon_idx * 10  # Since we recorded every 10 steps

    # Fit exponent
    # We need t values. t is step index * 10.
    t_vals = np.arange(len(test_errors)) * 10

    nu, r2 = fit_exponent(t_vals, test_errors, t_epsilon, expected_exp)

    if nu is None:
        results[D] = {
            "status": "inconclusive",
            "reason": "Fit failed",
            "expected_exp": expected_exp,
        }
        axes[i].plot(test_errors)
        axes[i].set_title(f"D={D}: Fit Failed")
        continue

    # Check success criterion
    diff = abs(nu - expected_exp)
    is_supported = diff < 0.1

    results[D] = {
        "status": "supported" if is_supported else "falsified",
        "fitted_exp": nu,
        "expected_exp": expected_exp,
        "diff": diff,
        "r_squared": r2,
        "t_epsilon": t_epsilon,
    }

    # Plot
    axes[i].plot(t_vals, test_errors, label="Test Error")
    axes[i].axvline(
        t_epsilon, color="r", linestyle="--", label=f"t_epsilon={t_epsilon}"
    )

    # Plot fit
    t_fit_range = np.linspace(0, t_epsilon, 100)
    # We need A from the fit. Let's re-calculate A for plotting.
    mask = (t_vals < t_epsilon) & (test_errors > 1e-6)
    if np.sum(mask) > 0:
        x = np.log(t_epsilon - t_vals[mask])
        y = np.log(test_errors[mask])
        A_mat = np.vstack([x, np.ones_like(x)]).T
        coeffs, _, _, _ = np.linalg.lstsq(A_mat, y, rcond=None)
        A = np.exp(coeffs[1])
        E_fit = A * (t_epsilon - t_fit_range) ** nu
        axes[i].plot(t_fit_range, E_fit, "r-", label=f"Fit: nu={nu:.2f}")

    axes[i].set_xlabel("Training Time")
    axes[i].set_ylabel("Test Error")
    axes[i].set_title(f"D={D}, nu={nu:.2f} (exp {expected_exp})")
    axes[i].legend()
    axes[i].set_yscale("log")

plt.tight_layout()
plt.savefig("results/c2/fig.png", dpi=150)

# Aggregate results
all_supported = all(
    r["status"] == "supported" for r in results.values() if "status" in r
)
any_inconclusive = any(r.get("status") == "inconclusive" for r in results.values())

if not control_pass:
    status = "inconclusive"
    notes = "Positive control failed. The fitting procedure or simulation setup is likely buggy."
elif any_inconclusive:
    status = "inconclusive"
    notes = "Some dimensions did not exhibit grokking or fit failed."
elif all_supported:
    status = "supported"
    notes = "Fitted exponents match theoretical predictions within tolerance for all tested dimensions."
else:
    status = "falsified"
    notes = "Fitted exponents deviate from theoretical predictions for at least one dimension."

metrics = {"control_pass": control_pass, "dimensions": {}}

for D in DIMS:
    if D in results:
        metrics["dimensions"][str(D)] = results[D]

summary = {"claim_id": "C2", "status": status, "metrics": metrics, "notes": notes}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

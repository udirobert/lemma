import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Constants
DIMS = [2, 5, 10]
N_TRAIN = 2000
N_TEST = 10000
EPSILON = 0.1
LR = 0.01
MAX_STEPS = 5000
TOL = 0.1

# Ensure results directory exists
os.makedirs("results/c2", exist_ok=True)


# Helper: Sample from D-dimensional uniform ball
# We use rejection sampling for simplicity and correctness.
def sample_ball(D, N, center, radius=1.0):
    """
    Sample N points uniformly from a D-dimensional ball of given radius centered at 'center'.
    """
    points = np.zeros((N, D))
    for i in range(N):
        while True:
            # Sample from hypercube [-1, 1]^D
            x = np.random.uniform(-1, 1, D)
            if np.sum(x**2) <= 1.0:
                points[i] = center + radius * x
                break
    return points


def train_perceptron(X_train, y_train, X_test, y_test, D, lr, max_steps, epsilon):
    """
    Train a perceptron using gradient descent on the 0-1 loss (subgradient).
    Returns test_errors, train_errors, and t_epsilon (the step where train error becomes 0).
    """
    w = np.zeros(D)
    b = 0.0

    test_errors = []
    train_errors = []
    t_epsilon = None

    for t in range(max_steps):
        # Compute predictions
        y_pred_train = np.sign(X_train @ w + b)
        y_pred_test = np.sign(X_test @ w + b)

        # Handle sign(0) = 0 case: if 0, treat as -1 or +1?
        # Standard perceptron: if dot product is 0, it's on the boundary.
        # For error calculation, we need a binary label.
        # Let's define sign(0) = 1 for consistency, or just check misclassification.
        # Misclassification: y * (x.w + b) < 0

        # Train error
        train_mis = (y_train * (X_train @ w + b)) < 0
        train_err = np.mean(train_mis)
        train_errors.append(train_err)

        # Test error
        test_mis = (y_test * (X_test @ w + b)) < 0
        test_err = np.mean(test_mis)
        test_errors.append(test_err)

        # Check for grokking transition (train error becomes 0)
        if train_err == 0.0 and t_epsilon is None:
            t_epsilon = t
            # We can stop early if we want, but we need to record test error
            # for t < t_epsilon. The claim is about E(t) for t < t_epsilon.
            # Actually, the transition is at t_epsilon. We need data points
            # where t < t_epsilon.
            # If we stop here, we have data up to t_epsilon.
            # But we might want to see the behavior just before.
            # Let's continue for a few steps to ensure we have enough data
            # or just break if we have enough.
            # The claim is E_D(t) ~ (t_epsilon - t)^nu for t < t_epsilon.
            # So we need t values strictly less than t_epsilon.
            # If we break at t_epsilon, the last point is AT t_epsilon.
            # We should probably stop a bit earlier or just use the points before.
            pass

        # Gradient descent update
        # Subgradient of 0-1 loss: if misclassified, grad = -y * x (for w), -y (for b)
        # Update: w += lr * sum(y_i * x_i) for misclassified i
        # b += lr * sum(y_i) for misclassified i

        if np.any(train_mis):
            w += lr * np.sum(y_train[train_mis] * X_train[train_mis], axis=0)
            b += lr * np.sum(y_train[train_mis])

        # Optional: Stop if we have gone well past t_epsilon and test error is stable
        if t_epsilon is not None and t > t_epsilon + 100:
            break

    return np.array(test_errors), np.array(train_errors), t_epsilon


def fit_exponent(t_vals, e_vals, t_epsilon):
    """
    Fit log(E) = log(A) + nu * log(t_epsilon - t).
    Returns nu and r_squared.
    """
    # Filter valid points: t < t_epsilon and E > 0
    mask = (t_vals < t_epsilon) & (e_vals > 0)
    t_fit = t_vals[mask]
    e_fit = e_vals[mask]

    if len(t_fit) < 5:
        return None, None

    x = np.log(t_epsilon - t_fit)
    y = np.log(e_fit)

    # Linear regression y = nu * x + log(A)
    # Using np.polyfit
    coeffs = np.polyfit(x, y, 1)
    nu = coeffs[0]

    # Calculate R^2
    y_pred = coeffs[0] * x + coeffs[1]
    ss_res = np.sum((y - y_pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        r2 = 1.0
    else:
        r2 = 1 - (ss_res / ss_tot)

    return nu, r2


# Positive Control
# We generate synthetic data where E(t) = (t_epsilon - t)^nu exactly.
# We then fit it and check if we recover nu.
def run_positive_control():
    D = 2
    t_epsilon = 100
    true_nu = (D + 1) / 2.0  # 1.5

    # Generate t values
    t_vals = np.arange(0, t_epsilon, 1)
    # Generate E values with small noise to simulate real data
    e_vals = (t_epsilon - t_vals) ** true_nu
    # Add small multiplicative noise
    noise = np.random.normal(1.0, 0.01, size=len(e_vals))
    e_vals = e_vals * noise

    # Fit
    nu_fit, r2 = fit_exponent(t_vals, e_vals, t_epsilon)

    # Check if fit is close to true_nu
    control_pass = (nu_fit is not None) and (abs(nu_fit - true_nu) < 0.1)

    return control_pass, nu_fit, r2


# Main Audit
results = {}
control_pass, control_nu, control_r2 = run_positive_control()

for D in DIMS:
    np.random.seed(42 + D)  # Fixed seed for reproducibility

    # Generate Data
    # Positive class: center at (epsilon, 0, ..., 0)
    # Negative class: center at (-epsilon, 0, ..., 0)
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

    # Test Data (same distribution)
    X_test_pos = sample_ball(D, N_TEST, center_pos)
    X_test_neg = sample_ball(D, N_TEST, center_neg)
    X_test = np.vstack([X_test_pos, X_test_neg])
    y_test = np.concatenate([np.ones(N_TEST), -np.ones(N_TEST)])

    # Train
    test_errors, train_errors, t_epsilon = train_perceptron(
        X_train, y_train, X_test, y_test, D, LR, MAX_STEPS, EPSILON
    )

    if t_epsilon is None:
        results[D] = {
            "status": "inconclusive",
            "reason": "No grokking transition observed (train error never reached 0)",
        }
        continue

    # Fit exponent
    nu, r2 = fit_exponent(np.arange(len(test_errors)), test_errors, t_epsilon)

    if nu is None:
        results[D] = {
            "status": "inconclusive",
            "reason": "Insufficient data points for fitting",
        }
        continue

    theoretical_nu = (D + 1) / 2.0
    is_supported = abs(nu - theoretical_nu) <= TOL

    results[D] = {
        "fitted_nu": nu,
        "theoretical_nu": theoretical_nu,
        "r_squared": r2,
        "t_epsilon": t_epsilon,
        "status": "supported" if is_supported else "falsified",
    }

# Determine overall status
# If control fails, status is inconclusive.
# If any D is falsified, overall is falsified? Or if all are supported, supported?
# The claim is for D in {2, 5, 10}. If it holds for all, supported. If it fails for any, falsified.
# If some are inconclusive, overall inconclusive.

overall_status = "supported"
if not control_pass:
    overall_status = "inconclusive"
else:
    for D in DIMS:
        if D in results:
            if results[D]["status"] == "falsified":
                overall_status = "falsified"
                break
            elif results[D]["status"] == "inconclusive":
                overall_status = "inconclusive"
                break

# Plotting
fig, axes = plt.subplots(1, len(DIMS), figsize=(15, 5))
if len(DIMS) == 1:
    axes = [axes]

for i, D in enumerate(DIMS):
    ax = axes[i]
    if D in results and "fitted_nu" in results[D]:
        # Re-run or store data for plotting?
        # To avoid re-running, we should have stored the data.
        # Let's modify the main loop to store data for plotting.
        pass

# Since I didn't store the data in the loop above, I'll just plot the summary or re-run one for demo.
# Better: I will restructure to store data.

# Let's redo the main loop to store data for plotting.
plot_data = {}
for D in DIMS:
    np.random.seed(42 + D)
    center_pos = np.zeros(D)
    center_pos[0] = EPSILON
    center_neg = np.zeros(D)
    center_neg[0] = -EPSILON

    X_pos = sample_ball(D, N_TRAIN, center_pos)
    X_neg = sample_ball(D, N_TRAIN, center_neg)

    X_train = np.vstack([X_pos, X_neg])
    y_train = np.concatenate([np.ones(N_TRAIN), -np.ones(N_TRAIN)])

    idx = np.random.permutation(2 * N_TRAIN)
    X_train = X_train[idx]
    y_train = y_train[idx]

    X_test_pos = sample_ball(D, N_TEST, center_pos)
    X_test_neg = sample_ball(D, N_TEST, center_neg)
    X_test = np.vstack([X_test_pos, X_test_neg])
    y_test = np.concatenate([np.ones(N_TEST), -np.ones(N_TEST)])

    test_errors, train_errors, t_epsilon = train_perceptron(
        X_train, y_train, X_test, y_test, D, LR, MAX_STEPS, EPSILON
    )

    if t_epsilon is not None:
        plot_data[D] = {
            "t": np.arange(len(test_errors)),
            "e": test_errors,
            "t_epsilon": t_epsilon,
        }

for i, D in enumerate(DIMS):
    ax = axes[i]
    if D in plot_data:
        data = plot_data[D]
        t = data["t"]
        e = data["e"]
        t_eps = data["t_epsilon"]

        ax.plot(t, e, "b-", label="Test Error")
        ax.axvline(t_eps, color="r", linestyle="--", label="t_epsilon")
        ax.set_title(f"D={D}")
        ax.set_xlabel("t")
        ax.set_ylabel("E(t)")
        ax.legend()

        # Log-log plot inset or separate? Let's just do linear for now, or log-log if possible.
        # The claim is about log-log. Let's add a log-log plot.
        # We need t < t_epsilon and e > 0.
        mask = (t < t_eps) & (e > 0)
        if np.sum(mask) > 5:
            ax2 = ax.twinx()
            ax2.loglog(t[mask], e[mask], "ro", markersize=2, label="Log-Log")
            ax2.set_ylabel("E(t) (log)")
            ax2.legend(loc="upper right")

plt.tight_layout()
plt.savefig("results/c2/fig.png", dpi=150)
plt.close()

# Construct summary
metrics = {
    "control_pass": control_pass,
    "control_nu": control_nu,
    "control_r2": control_r2,
}

for D in DIMS:
    if D in results:
        metrics[f"D{D}_nu"] = results[D].get("fitted_nu", "N/A")
        metrics[f"D{D}_theoretical"] = results[D].get("theoretical_nu", "N/A")
        metrics[f"D{D}_r2"] = results[D].get("r_squared", "N/A")
        metrics[f"D{D}_status"] = results[D].get("status", "N/A")

summary = {
    "claim_id": "C2",
    "status": overall_status,
    "metrics": metrics,
    "notes": f"Control passed: {control_pass}. Fitted exponents vs theoretical: "
    + ", ".join(
        [
            f"D={D}: {results.get(D, {}).get('fitted_nu', 'N/A')} vs {results.get(D, {}).get('theoretical_nu', 'N/A')}"
            for D in DIMS
        ]
    ),
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

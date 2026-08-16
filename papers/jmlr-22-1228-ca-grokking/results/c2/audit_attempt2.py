import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

np.random.seed(42)

# Constants
EPSILON = 0.1
N_TRAIN = 5000
N_TEST = 50000
LR = 0.01
MAX_STEPS = 20000

# Output directory
OUT_DIR = "results/c2"
os.makedirs(OUT_DIR, exist_ok=True)


def sample_ball(D, N, center):
    """Sample N points uniformly from a D-dimensional unit ball centered at 'center'."""
    # Generate random points in the cube and reject those outside the ball
    points = []
    while len(points) < N:
        # Sample from the cube [-1, 1]^D
        x = np.random.uniform(-1, 1, size=(N, D))
        # Check if inside the unit ball
        norms = np.sum(x**2, axis=1)
        mask = norms <= 1.0
        valid_points = x[mask]
        points.append(valid_points)
        if len(valid_points) >= N:
            break

    # Concatenate and take first N
    all_points = np.vstack(points)
    all_points = all_points[:N]
    # Shift to center
    all_points += center
    return all_points


def generate_data(D, N, epsilon):
    """Generate training and test data for the D-dimensional uniform ball model."""
    # Positive class: ball centered at (+epsilon, 0, ..., 0)
    # Negative class: ball centered at (-epsilon, 0, ..., 0)

    center_pos = np.zeros(D)
    center_pos[0] = epsilon
    center_neg = np.zeros(D)
    center_neg[0] = -epsilon

    # Training data
    X_pos_train = sample_ball(D, N, center_pos)
    X_neg_train = sample_ball(D, N, center_neg)
    X_train = np.vstack([X_pos_train, X_neg_train])
    y_train = np.concatenate([np.ones(N), -np.ones(N)])

    # Shuffle
    idx = np.random.permutation(2 * N)
    X_train = X_train[idx]
    y_train = y_train[idx]

    # Test data (large N for accurate error estimation)
    X_pos_test = sample_ball(D, N_TEST, center_pos)
    X_neg_test = sample_ball(D, N_TEST, center_neg)
    X_test = np.vstack([X_pos_test, X_neg_test])
    y_test = np.concatenate([np.ones(N_TEST), -np.ones(N_TEST)])

    return X_train, y_train, X_test, y_test


def train_perceptron(
    X_train, y_train, X_test, y_test, D, lr=LR, max_steps=MAX_STEPS, epsilon=EPSILON
):
    """Train a perceptron using gradient descent and record test error."""
    w = np.zeros(D)
    b = 0.0

    test_errors = []
    train_errors = []

    for step in range(max_steps):
        # Compute predictions
        y_pred = np.sign(X_train @ w + b)
        # Handle zeros (should be rare)
        y_pred[y_pred == 0] = 1

        # Training error
        train_err = np.mean(y_pred != y_train)
        train_errors.append(train_err)

        # Test error
        y_pred_test = np.sign(X_test @ w + b)
        y_pred_test[y_pred_test == 0] = 1
        test_err = np.mean(y_pred_test != y_test)
        test_errors.append(test_err)

        # Gradient descent update
        # Loss: R = 1/(2N) * sum (1 - y_i * (w.x_i + b))_+^2
        # Gradient: -1/N * sum y_i * (w.x_i + b) for misclassified (or all if using hinge)
        # For perceptron with sign, we use the standard perceptron update rule:
        # w += lr * sum y_i * x_i for misclassified
        # b += lr * sum y_i for misclassified

        misclassified = y_pred != y_train
        if np.any(misclassified):
            w += lr * np.sum(
                y_train[misclassified, np.newaxis] * X_train[misclassified], axis=0
            )
            b += lr * np.sum(y_train[misclassified])

    return np.array(test_errors), np.array(train_errors)


def fit_exponent(test_errors, epsilon, D):
    """Fit the critical exponent from the test error data."""
    # The transition time t_epsilon is when the test error starts to drop rapidly.
    # In the paper, t_epsilon is the time when the training error reaches zero.
    # We need to find t_epsilon from the training error data.
    # However, we only have test_errors here. Let's assume t_epsilon is the step where train error becomes 0.
    # We need to pass train_errors as well.
    pass


def fit_exponent_full(test_errors, train_errors, epsilon, D):
    """Fit the critical exponent from the test and training error data."""
    # Find t_epsilon: the first step where training error is 0
    zero_train_idx = np.where(train_errors == 0)[0]
    if len(zero_train_idx) == 0:
        return None, None

    t_epsilon = zero_train_idx[0]

    # We want to fit E(t) ~ (t_epsilon - t)^nu for t < t_epsilon
    # But wait, the paper says E_D(t) ~ (t - t_epsilon)^((D+1)/2) for t > t_epsilon?
    # Let's re-read: "E_D(t) \propto (t_\epsilon - t)^{\frac{D+1}{2}}" where t < t_epsilon.
    # Actually, the paper says: "The critical exponent is hence determined only by the dimensionality... E_D(t) \approx ... (t - t_\epsilon)^{\frac{D+1}{2}}"
    # In Section 3.3.1, Eq 23: E_D(t) \approx C (t - t_\epsilon)^{\frac{D+1}{2}} for t > t_\epsilon.
    # But the claim says: E_D(t) \propto (t_\epsilon - t)^{\frac{D+1}{2}} where t < t_\epsilon.
    # This is a contradiction. Let's look at the paper text provided.
    # "E_D(t) \approx ... (t - t_\epsilon)^{\frac{D+1}{2}}" in the excerpt.
    # The claim statement says: "E_D(t) \propto (t_\epsilon - t)^{\frac{D+1}{2}}, where D is the dimension..."
    # And "Record test error E(t) for t < t_epsilon".
    # This implies the error is non-zero for t < t_epsilon and drops to zero at t_epsilon.
    # So for t < t_epsilon, E(t) > 0, and as t -> t_epsilon from below, E(t) -> 0.
    # So E(t) ~ (t_epsilon - t)^nu.

    # We need data for t < t_epsilon.
    # t is the step index.
    # We fit log(E) = log(C) + nu * log(t_epsilon - t)

    # Select steps where t < t_epsilon and E(t) > 0
    mask = np.arange(len(test_errors)) < t_epsilon
    t_vals = np.arange(len(test_errors))[mask]
    e_vals = test_errors[mask]

    # Filter out zero errors (should not happen for t < t_epsilon if grokking is present)
    valid = e_vals > 1e-10
    t_vals = t_vals[valid]
    e_vals = e_vals[valid]

    if len(t_vals) < 10:
        return None, None

    # Compute x = t_epsilon - t
    x_vals = t_epsilon - t_vals

    # Fit log(e) = log(C) + nu * log(x)
    log_x = np.log(x_vals)
    log_e = np.log(e_vals)

    # Linear regression: log_e = a + b * log_x
    A = np.vstack([log_x, np.ones_like(log_x)]).T
    coeffs, residuals, rank, s = np.linalg.lstsq(A, log_e, rcond=None)
    nu = coeffs[0]
    log_C = coeffs[1]

    return nu, t_epsilon


# Main execution
results = {}
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

for i, D in enumerate([2, 5, 10]):
    print(f"Running D={D}...")
    X_train, y_train, X_test, y_test = generate_data(D, N_TRAIN, EPSILON)
    test_errors, train_errors = train_perceptron(X_train, y_train, X_test, y_test, D)

    nu, t_epsilon = fit_exponent_full(test_errors, train_errors, EPSILON, D)

    if nu is not None:
        expected_nu = (D + 1) / 2.0
        diff = abs(nu - expected_nu)
        results[f"D{D}"] = {
            "fitted_nu": nu,
            "expected_nu": expected_nu,
            "diff": diff,
            "t_epsilon": t_epsilon,
            "pass": diff < 0.1,
        }
        print(
            f"D={D}: fitted nu={nu:.4f}, expected={expected_nu:.4f}, diff={diff:.4f}, pass={diff < 0.1}"
        )

        # Plot
        ax = axes[i]
        t_vals = np.arange(len(test_errors))
        ax.plot(t_vals, test_errors, label="Test Error")
        ax.axvline(t_epsilon, color="r", linestyle="--", label=f"t_epsilon={t_epsilon}")
        ax.set_xlabel("Training Step")
        ax.set_ylabel("Test Error")
        ax.set_title(f"D={D}, nu={nu:.3f}")
        ax.legend()
    else:
        results[f"D{D}"] = {
            "fitted_nu": None,
            "expected_nu": (D + 1) / 2.0,
            "diff": None,
            "t_epsilon": None,
            "pass": False,
        }
        print(f"D={D}: Failed to fit exponent")
        axes[i].text(0.5, 0.5, "Fit failed", ha="center", va="center")

plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig.png"), dpi=150)
plt.close()

# Positive Control
# We need a synthetic case where the answer is known.
# Let's create a synthetic test error curve that follows E(t) = (t_epsilon - t)^nu exactly.
# And check if our fitting procedure recovers nu.

print("Running positive control...")
D_control = 2
expected_nu_control = (D_control + 1) / 2.0  # 1.5
t_epsilon_control = 1000
max_steps_control = 2000

# Generate synthetic data
np.random.seed(123)
t_vals_control = np.arange(max_steps_control)
# Only consider t < t_epsilon
mask_control = t_vals_control < t_epsilon_control
t_vals_control = t_vals_control[mask_control]

# E(t) = C * (t_epsilon - t)^nu
C_control = 0.5
e_vals_control = C_control * (t_epsilon_control - t_vals_control) ** expected_nu_control

# Add small noise to make it realistic
noise = np.random.normal(0, 0.01 * e_vals_control)
e_vals_control_noisy = e_vals_control + noise
e_vals_control_noisy = np.maximum(e_vals_control_noisy, 1e-10)

# We need to create a dummy train_errors array where train error becomes 0 at t_epsilon_control
train_errors_control = np.ones(max_steps_control)
train_errors_control[t_epsilon_control:] = 0.0

# Create a dummy test_errors array that matches the synthetic data for t < t_epsilon
# and is 0 for t >= t_epsilon
test_errors_control = np.zeros(max_steps_control)
test_errors_control[t_vals_control] = e_vals_control_noisy

# Fit
nu_control, t_epsilon_fit_control = fit_exponent_full(
    test_errors_control, train_errors_control, EPSILON, D_control
)

if nu_control is not None:
    control_diff = abs(nu_control - expected_nu_control)
    control_pass = control_diff < 0.1
    print(
        f"Control: fitted nu={nu_control:.4f}, expected={expected_nu_control:.4f}, diff={control_diff:.4f}, pass={control_pass}"
    )
else:
    control_pass = False
    print("Control: Failed to fit exponent")

# Determine overall status
all_pass = all(results[f"D{D}"]["pass"] for D in [2, 5, 10])
if control_pass and all_pass:
    status = "supported"
elif not control_pass:
    status = "inconclusive"
else:
    status = "falsified"

# Prepare summary
metrics = {
    "control_pass": control_pass,
    "D2_nu": results["D2"]["fitted_nu"],
    "D2_expected": results["D2"]["expected_nu"],
    "D2_diff": results["D2"]["diff"],
    "D5_nu": results["D5"]["fitted_nu"],
    "D5_expected": results["D5"]["expected_nu"],
    "D5_diff": results["D5"]["diff"],
    "D10_nu": results["D10"]["fitted_nu"],
    "D10_expected": results["D10"]["expected_nu"],
    "D10_diff": results["D10"]["diff"],
}

# Convert numpy types to native Python types for JSON serialization
for key, value in metrics.items():
    if isinstance(value, (np.floating, np.integer)):
        metrics[key] = float(value)
    elif isinstance(value, np.bool_):
        metrics[key] = bool(value)

summary = {
    "claim_id": "C2",
    "status": status,
    "metrics": metrics,
    "notes": f"Fitted exponents for D=2,5,10. Control pass: {control_pass}. All pass: {all_pass}.",
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

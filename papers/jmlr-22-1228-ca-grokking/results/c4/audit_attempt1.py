import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import linregress

# Constants
N = 10
epsilon = 1.01
lambda_2 = 0.1
lambda_1 = 0.0
D_values = [5, 10, 20, 50]
num_trials = 1000

# Helper: Compute grokking probability for a given D
# Based on the paper's model (Section 3.3.2 and Appendix B.1)
# The grokking probability is the probability that the minimum norm solution
# (or the solution at infinite time with L2 reg) generalizes perfectly.
# In the D-dimensional uniform ball model, the data is drawn from a uniform distribution
# in a ball of radius R. The classes are separated by a hyperplane.
# The key quantity is the angle between the weight vector and the data points.
# For the uniform ball model, the probability of grokking is related to the
# probability that the minimum norm solution has zero test error.
#
# According to the paper (Eq. 22 in Section 3.3.2 or similar), for the uniform ball model,
# the grokking probability P_grok(D) can be estimated by checking if the minimum norm
# solution w* = X^T (X X^T)^{-1} y (for balanced classes) generalizes.
# However, a simpler analytical result or simulation approach is often used.
#
# Let's simulate the actual training or the final state.
# The paper states that for the uniform ball model, the grokking probability decreases exponentially.
# A common way to simulate this is to generate data from two uniform balls centered at +/- mu * e_1
# or similar, but the paper specifies "uniform ball model" which usually implies
# the data is uniform in a ball, and the classes are defined by a linear rule.
#
# Let's look at the specific setup in the paper's simulations (Fig 8).
# The paper likely uses a setup where the data is drawn from a uniform distribution
# in a D-dimensional ball, and the teacher rule is a linear classifier.
# The student is trained with gradient descent.
#
# For the audit, we will simulate the following:
# 1. Generate N positive and N negative samples from a D-dimensional uniform ball.
#    To make them linearly separable, we can assume the teacher rule is x_1 > 0.
#    So positive samples have x_1 > 0, negative have x_1 < 0.
#    But they are drawn from the same uniform ball distribution, conditioned on the sign of x_1.
# 2. Train a linear model with L2 regularization (lambda_2) and no L1 (lambda_1=0).
# 3. Check if the final model has zero test error.
#
# However, training for 1000 trials x 4 dimensions x 1000 steps might be slow if done naively.
# We can use the closed-form solution for the minimum norm solution if we assume
# the training goes to convergence. With L2 regularization, the solution is:
# w = (X^T X + lambda_2 * I)^{-1} X^T y
#
# Let's implement this.


def simulate_grokking(D, N, epsilon, lambda_2, lambda_1, num_trials, seed):
    np.random.seed(seed)
    grok_count = 0

    for _ in range(num_trials):
        # Generate data from uniform ball in D dimensions
        # Radius R. Let's assume R=1 for simplicity, or scale doesn't matter for linear separability.
        # The paper might use a specific radius, but the exponential decay with D is the key.
        # We generate points uniformly in the ball.
        # Method: generate Gaussian, normalize to unit sphere, multiply by U^(1/D) where U~Uniform(0,1).

        # Positive class: x_1 > 0
        # Negative class: x_1 < 0

        # We need to ensure linear separability. The teacher rule is sign(x_1).
        # So we sample from the uniform ball, and assign label based on sign(x_1).
        # But we need exactly N positive and N negative.
        # So we sample until we have N of each, or sample from the half-ball.

        # Sampling from half-ball:
        # Generate points in full ball, keep those with x_1 > 0 for positive, x_1 < 0 for negative.
        # This is inefficient for high D.
        # Alternative: Sample from full ball, and if x_1 is close to 0, resample.
        # Or, simply sample from the full ball and take the first N with x_1>0 and first N with x_1<0.

        # For high D, the volume of the ball is concentrated near the surface.
        # The probability that x_1 > 0 is 0.5.

        # Let's generate 2*N points from the uniform ball.
        # We will take the first N with x_1 > 0 as positive, and first N with x_1 < 0 as negative.
        # If we don't get enough, we generate more.

        X_pos = []
        X_neg = []

        # To be efficient, we can generate a batch and filter.
        # For D=50, rejection sampling might be slow if we just generate one by one.
        # Let's generate a large batch.

        while len(X_pos) < N or len(X_neg) < N:
            # Generate batch of points
            batch_size = 100
            # Generate Gaussian
            Z = np.random.randn(batch_size, D)
            # Normalize to unit sphere
            norms = np.linalg.norm(Z, axis=1, keepdims=True)
            # Avoid division by zero
            norms[norms == 0] = 1
            Z = Z / norms
            # Scale by radius^(1/D) * U^(1/D)
            # Let's assume radius R=1.
            U = np.random.rand(batch_size, 1)
            R = 1.0
            radii = R * (U ** (1.0 / D))
            X_batch = Z * radii

            # Split into positive and negative based on x_1
            pos_mask = X_batch[:, 0] > 0
            neg_mask = X_batch[:, 0] < 0

            # Add to lists
            X_pos.extend(X_batch[pos_mask].tolist())
            X_neg.extend(X_batch[neg_mask].tolist())

            # Trim to N
            X_pos = X_pos[:N]
            X_neg = X_neg[:N]

        X_pos = np.array(X_pos)
        X_neg = np.array(X_neg)

        # Combine X and y
        X = np.vstack([X_pos, X_neg])
        y = np.array([1] * N + [-1] * N)

        # Solve for w with L2 regularization
        # w = (X^T X + lambda_2 * I)^{-1} X^T y
        # Note: The paper might use a different scaling for lambda.
        # Usually, the loss is (1/2N) ||Xw - y||^2 + (lambda/2) ||w||^2.
        # Gradient: (1/N) X^T (Xw - y) + lambda w = 0
        # => (X^T X + N * lambda * I) w = X^T y
        # Let's assume the standard form: (X^T X + lambda_2 * I) w = X^T y
        # The value of lambda_2 might need scaling with N or D, but the paper fixes it.

        A = X.T @ X + lambda_2 * np.eye(D)
        b = X.T @ y

        try:
            w = np.linalg.solve(A, b)
        except np.linalg.LinAlgError:
            w = np.linalg.lstsq(A, b, rcond=None)[0]

        # Calculate test error
        # Test error is the probability of misclassification over the distribution.
        # We approximate this by sampling a large number of test points.
        # Or, we can check if the margin is positive for all training points (which implies generalization in this model?)
        # No, grokking is defined as zero TEST error.
        # So we need to estimate the test error.

        # Sample test points from the same distribution (uniform ball, labeled by sign(x_1))
        # We need to estimate P(sgn(w^T x) != sign(x_1))

        # For efficiency, we can use a fixed set of test points or Monte Carlo.
        # Let's use 1000 test points per trial. This might be slow.
        # 1000 trials * 1000 test points * 4 D values = 4,000,000 points.
        # This is manageable.

        # Generate test points
        test_size = 1000
        Z_test = np.random.randn(test_size, D)
        norms_test = np.linalg.norm(Z_test, axis=1, keepdims=True)
        norms_test[norms_test == 0] = 1
        Z_test = Z_test / norms_test
        U_test = np.random.rand(test_size, 1)
        X_test = Z_test * (U_test ** (1.0 / D))
        y_test = np.sign(X_test[:, 0])
        y_test[y_test == 0] = 1  # Handle zero probability

        # Predictions
        y_pred = np.sign(X_test @ w)
        y_pred[y_pred == 0] = 1

        # Check for zero test error
        if np.all(y_pred == y_test):
            grok_count += 1

    return grok_count / num_trials


# Run simulation
results = {}
for D in D_values:
    print(f"Simulating D={D}...")
    p_grok = simulate_grokking(D, N, epsilon, lambda_2, lambda_1, num_trials, seed=42)
    results[D] = p_grok
    print(f"  P_grok = {p_grok}")

# Check monotonic decrease
D_list = sorted(results.keys())
P_list = [results[D] for D in D_list]

monotonic = all(P_list[i] >= P_list[i + 1] for i in range(len(P_list) - 1))

# Check linear trend in log(P) vs D
# Filter out zeros for log
valid_indices = [i for i, p in enumerate(P_list) if p > 0]
if len(valid_indices) >= 2:
    D_valid = [D_list[i] for i in valid_indices]
    logP_valid = [np.log(P_list[i]) for i in valid_indices]
    slope, intercept, r_value, p_value, std_err = linregress(D_valid, logP_valid)
    # Check if slope is negative and significant
    linear_trend = slope < 0 and p_value < 0.05
else:
    slope = 0
    linear_trend = False

# Positive Control
# The claim is about the specific model. A positive control for the *statistic* (monotonic decrease)
# is hard to construct without knowing the exact ground truth.
# However, we can verify that our simulation code works by checking a case where grokking is certain.
# For example, if D=1, and data is perfectly separable with large margin, grokking should be high.
# Or, we can check that the simulation produces probabilities between 0 and 1.
# A better control: Run the simulation for a very small D (e.g., D=1) and a very large D.
# We expect P(D=1) > P(D=50).
# Let's add D=1 to the control.

print("Running positive control (D=1)...")
p_control = simulate_grokking(1, N, epsilon, lambda_2, lambda_1, num_trials, seed=42)
print(f"  P_grok(D=1) = {p_control}")

# The control should show that the simulation can produce a high probability for low D.
# If p_control is not significantly higher than the others, the simulation might be flawed.
# But the main check is the trend.
# Let's define control_pass as: P(D=1) > P(D=50) and P(D=1) > 0.5 (assuming high grokking in 1D)
control_pass = (p_control > results[50]) and (p_control > 0.5)

# Plotting
os.makedirs("results/c4", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.plot(D_list, P_list, "o-", label="Grokking Probability")
plt.xlabel("Dimension D")
plt.ylabel("Grokking Probability")
plt.title("Grokking Probability vs Dimension D")
plt.legend()
plt.grid(True)
plt.savefig("results/c4/fig_prob.png")
plt.close()

if len(valid_indices) >= 2:
    plt.figure(figsize=(10, 6))
    plt.plot(D_valid, logP_valid, "o-", label="log(P_grok)")
    # Fit line
    x_fit = np.linspace(min(D_valid), max(D_valid), 100)
    y_fit = slope * x_fit + intercept
    plt.plot(x_fit, y_fit, "--", label=f"Linear Fit (slope={slope:.3f})")
    plt.xlabel("Dimension D")
    plt.ylabel("log(Grokking Probability)")
    plt.title("Log Grokking Probability vs Dimension D")
    plt.legend()
    plt.grid(True)
    plt.savefig("results/c4/fig_log_prob.png")
    plt.close()

# Determine status
if monotonic and linear_trend:
    status = "supported"
else:
    status = (
        "falsified" if not monotonic else "inconclusive"
    )  # If not monotonic, it's falsified. If monotonic but not linear, maybe inconclusive or falsified depending on strictness.
    # The claim says "exponentially decreases", which implies linear log trend.
    # If it decreases but not exponentially, it's falsified.
    if not monotonic:
        status = "falsified"
    else:
        # Monotonic but not linear log trend
        status = "falsified"  # The claim is specifically about exponential decay

# If control fails, status is inconclusive
if not control_pass:
    status = "inconclusive"

summary = {
    "claim_id": "C4",
    "status": status,
    "metrics": {
        "P_grok_D5": results[5],
        "P_grok_D10": results[10],
        "P_grok_D20": results[20],
        "P_grok_D50": results[50],
        "monotonic_decrease": monotonic,
        "log_slope": slope,
        "log_p_value": p_value if len(valid_indices) >= 2 else None,
        "control_pass": control_pass,
        "P_grok_D1_control": p_control,
    },
    "notes": f"Grokking probabilities: D=5:{results[5]:.4f}, D=10:{results[10]:.4f}, D=20:{results[20]:.4f}, D=50:{results[50]:.4f}. Monotonic: {monotonic}. Log-linear slope: {slope:.4f}. Control (D=1): {p_control:.4f}.",
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

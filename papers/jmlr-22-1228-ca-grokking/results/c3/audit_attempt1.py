import json
import os

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Setup
D = 10
N = 20
epsilon = 1.01
lambda1_l2 = 0.0
lambda1_l1 = 0.1
lambda2 = 0.1
n_trials = 200
lr = 0.01
max_steps = 5000

np.random.seed(42)


def generate_data(D, N, epsilon):
    # Uniform ball model: samples from unit ball in R^D
    # Positive class: shifted by epsilon along x-axis
    # Negative class: centered at origin
    # We use the standard construction from the paper (Section 3.3)
    # The distributions are P+ and P- which are linearly separable.
    # For the uniform ball, we sample from the ball of radius 1.

    def sample_ball(D, N):
        # Sample from uniform distribution in D-dimensional ball of radius 1
        # Method: sample from Gaussian and normalize by norm, then scale by U^(1/D)
        # Actually, uniform in ball: r = U^(1/D), direction = Gaussian normalized
        samples = np.zeros((N, D))
        for i in range(N):
            # Direction
            g = np.random.randn(D)
            g = g / np.linalg.norm(g)
            # Radius
            u = np.random.rand()
            r = u ** (1.0 / D)
            samples[i] = r * g
        return samples

    # Positive class: shifted by epsilon * e_1
    # Negative class: centered at 0
    # The paper defines the separation such that the classes are linearly separable.
    # The critical parameter is epsilon.

    x_pos = sample_ball(D, N) + epsilon * np.eye(D)[0]
    x_neg = sample_ball(D, N)

    X = np.vstack([x_pos, x_neg])
    y = np.array([1] * N + [-1] * N)

    return X, y


def train_perceptron(X, y, D, lambda1, lambda2, lr, max_steps):
    # Initialize weights
    w = np.random.randn(D) * 0.01
    b = 0.0

    # Training loop
    for t in range(max_steps):
        # Forward pass
        z = X @ w + b
        # Predictions
        y_pred = np.sign(z)
        # Handle zeros in sign (should be rare)
        y_pred[y_pred == 0] = 1

        # Loss: Hinge loss + Regularization
        # R = 1/(2N) * sum(max(0, 1 - y_i * (w.x_i + b))) + lambda1 * ||w||_1 + lambda2 * ||w||_2^2
        # Gradient of hinge loss
        mask = (y * z) < 1
        grad_w = (X.T @ (y * mask)) / (2 * N)
        grad_b = np.sum(y * mask) / (2 * N)

        # Regularization gradients
        # L1: subgradient sign(w)
        grad_w_l1 = lambda1 * np.sign(w)
        # L2: 2 * lambda2 * w (since loss has lambda2 * ||w||^2, grad is 2*lambda2*w)
        # Wait, the paper says lambda2 * ||w||_2^2. Usually it's 1/2 * lambda * ||w||^2.
        # Let's check the paper's convention. Eq (2) in paper: R = ... + lambda1 ||w||_1 + lambda2 ||w||_2^2.
        # So grad is 2 * lambda2 * w.
        grad_w_l2 = 2 * lambda2 * w

        total_grad_w = grad_w + grad_w_l1 + grad_w_l2
        total_grad_b = grad_b

        # Update
        w -= lr * total_grad_w
        b -= lr * total_grad_b

        # Check for zero test error (grokking)
        # Test error is the statistical average over P+ and P-.
        # Since we don't have infinite samples, we approximate with a large holdout set.
        # However, the claim is about the probability of achieving zero test error.
        # In the solvable model, this is a phase transition.
        # For the audit, we check if the model has converged to a solution that generalizes.
        # A proxy for "zero test error" in finite samples is that the margin is positive for all training points
        # AND the solution is robust.
        # Actually, the paper defines grokking as PE(infinity) = 0.
        # In the D-ball model, this happens if the weight vector aligns with the separation direction.
        # Let's check the test error on a large validation set.

    # Evaluate test error on a large sample
    X_test_pos = np.random.randn(1000, D)
    X_test_pos = X_test_pos / np.linalg.norm(X_test_pos, axis=1, keepdims=True)
    X_test_pos = X_test_pos * (np.random.rand(1000, 1) ** (1.0 / D))
    X_test_pos += epsilon * np.eye(D)[0]

    X_test_neg = np.random.randn(1000, D)
    X_test_neg = X_test_neg / np.linalg.norm(X_test_neg, axis=1, keepdims=True)
    X_test_neg = X_test_neg * (np.random.rand(1000, 1) ** (1.0 / D))

    X_test = np.vstack([X_test_pos, X_test_neg])
    y_test = np.array([1] * 1000 + [-1] * 1000)

    z_test = X_test @ w + b
    y_pred_test = np.sign(z_test)
    y_pred_test[y_pred_test == 0] = 1

    test_error = np.mean(y_pred_test != y_test)

    # Grokking is defined as zero test error. In practice, we might see very small error.
    # The paper says "probability of achieving zero test error".
    # Due to finite precision, we might use a threshold like < 1e-3.
    # But the phase transition is sharp. Let's use < 0.01 as a proxy for "grokked".
    grokked = test_error < 0.01

    return grokked, test_error


# Run trials
grokked_l2 = []
test_errors_l2 = []
grokked_l1 = []
test_errors_l1 = []

for i in range(n_trials):
    X, y = generate_data(D, N, epsilon)

    # L2 Regularization (lambda1 = 0)
    g_l2, te_l2 = train_perceptron(X, y, D, lambda1_l2, lambda2, lr, max_steps)
    grokked_l2.append(g_l2)
    test_errors_l2.append(te_l2)

    # L1 Regularization (lambda1 > 0)
    g_l1, te_l1 = train_perceptron(X, y, D, lambda1_l1, lambda2, lr, max_steps)
    grokked_l1.append(g_l1)
    test_errors_l1.append(te_l1)

p_grok_l2 = np.mean(grokked_l2)
p_grok_l1 = np.mean(grokked_l1)

# Positive Control
# The control should verify that the training procedure can find a solution.
# If we set epsilon very large, grokking should be almost certain.
# Or, we can check if the statistic (mean test error) is lower for L1.
# A better control: Run the same setup with a known separable dataset (e.g. linearly separable with large margin).
# If the model fails to grok on a trivially separable dataset, the training is broken.

# Let's do a simple control: D=2, N=10, epsilon=10.0 (very large separation)
# Grokking probability should be ~1.0 for both.

np.random.seed(123)
grokked_control_l2 = []
grokked_control_l1 = []
for i in range(20):
    X, y = generate_data(2, 10, 10.0)
    g_l2, _ = train_perceptron(X, y, 2, 0.0, 0.1, 0.01, 5000)
    g_l1, _ = train_perceptron(X, y, 2, 0.1, 0.1, 0.01, 5000)
    grokked_control_l2.append(g_l2)
    grokked_control_l1.append(g_l1)

control_pass = (np.mean(grokked_control_l2) > 0.8) and (
    np.mean(grokked_control_l1) > 0.8
)

# Plotting
os.makedirs("results/c3", exist_ok=True)

plt.figure(figsize=(10, 6))
plt.hist(test_errors_l2, bins=20, alpha=0.5, label=f"L2 (P_grok={p_grok_l2:.2f})")
plt.hist(test_errors_l1, bins=20, alpha=0.5, label=f"L1 (P_grok={p_grok_l1:.2f})")
plt.xlabel("Test Error")
plt.ylabel("Frequency")
plt.title("Distribution of Test Errors")
plt.legend()
plt.savefig("results/c3/fig.png")
plt.close()

# Determine status
if not control_pass:
    status = "inconclusive"
    notes = "Positive control failed. Training procedure may be buggy."
else:
    if p_grok_l1 > p_grok_l2:
        status = "supported"
        notes = f"L1 grokking probability ({p_grok_l1:.2f}) is higher than L2 ({p_grok_l2:.2f})."
    else:
        status = "falsified"
        notes = f"L1 grokking probability ({p_grok_l1:.2f}) is not higher than L2 ({p_grok_l2:.2f})."

summary = {
    "claim_id": "C3",
    "status": status,
    "metrics": {
        "p_grok_l2": float(p_grok_l2),
        "p_grok_l1": float(p_grok_l1),
        "control_pass": bool(control_pass),
        "n_trials": n_trials,
        "D": D,
        "N": N,
        "epsilon": epsilon,
    },
    "notes": notes,
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

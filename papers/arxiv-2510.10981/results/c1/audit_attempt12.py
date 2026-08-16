import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# Set seed for reproducibility
np.random.seed(42)

# Directory for results
results_dir = 'results/c1'
os.makedirs(results_dir, exist_ok=True)

# Hyperparameters
N = 10000  # Number of samples for Monte Carlo estimation
p = 10     # Context length
T = 2      # Number of task types
alpha = np.array([0.5, 0.5])  # Prior probabilities for task types
sigma_eps = 0.1  # Noise standard deviation

# Task definitions
# Task 1: Linear regression y = w*x + b
# Task 2: Quadratic regression y = a*x^2 + b

def sample_task():
    """Sample a task type and function parameters."""
    i = np.random.choice(T, p=alpha)
    if i == 0:
        # Linear: y = w*x + b
        w = np.random.normal(0, 1)
        b = np.random.normal(0, 1)
        def f(x):
            return w * x + b
        def f_params(x):
            return np.array([w, b])
    else:
        # Quadratic: y = a*x^2 + b
        a = np.random.normal(0, 1)
        b = np.random.normal(0, 1)
        def f(x):
            return a * x**2 + b
        def f_params(x):
            return np.array([a, b])
    return i, f, f_params

def generate_prompt(f, k):
    """Generate a prompt of length k (context + query)."""
    X = np.random.normal(0, 1, size=k+1)
    Y = f(X) + np.random.normal(0, sigma_eps, size=k+1)
    return X, Y

def bayes_predictor(X_context, Y_context, x_query, task_prior=alpha):
    """
    Compute the Bayes predictor (posterior mean) for a given context and query.
    For simplicity, we approximate the posterior over task types and parameters.

    Since we have a mixture of linear and quadratic models, we compute the posterior
    probability of each task type given the context, and then the posterior mean
    of the prediction for each task type.
    """
    k = len(X_context)

    # Compute likelihood of context under each task type
    # For linear: y = w*x + b, for quadratic: y = a*x^2 + b
    # We use a simple grid search or analytical approximation for the marginal likelihood.
    # For efficiency, we use a small grid of parameters.

    # Linear model: y = w*x + b
    # Marginal likelihood under linear model with prior w~N(0,1), b~N(0,1)
    # This is a linear regression with Gaussian priors, so the marginal likelihood is Gaussian.

    # For simplicity, we'll use a numerical approximation with a small grid.
    # But for speed, let's use the fact that for linear regression with Gaussian priors,
    # the posterior is Gaussian and we can compute the marginal likelihood analytically.

    # Actually, let's use a simpler approach: compute the posterior mean for each task type
    # using least squares with regularization (Bayesian linear regression).

    # Linear model: y = w*x + b
    # Design matrix: [x, 1]
    X_lin = np.column_stack([X_context, np.ones(k)])
    # Bayesian linear regression with prior w~N(0,1), b~N(0,1)
    # Posterior mean: (X^T X + I)^{-1} X^T y
    A_lin = X_lin.T @ X_lin + np.eye(2)
    b_lin = X_lin.T @ Y_context
    mean_lin = np.linalg.solve(A_lin, b_lin)
    # Predict at x_query
    pred_lin = mean_lin[0] * x_query + mean_lin[1]

    # Quadratic model: y = a*x^2 + b
    X_quad = np.column_stack([X_context**2, np.ones(k)])
    A_quad = X_quad.T @ X_quad + np.eye(2)
    b_quad = X_quad.T @ Y_context
    mean_quad = np.linalg.solve(A_quad, b_quad)
    pred_quad = mean_quad[0] * x_query**2 + mean_quad[1]

    # Compute marginal likelihoods (approximate)
    # For linear: residual sum of squares
    pred_lin_context = X_lin @ mean_lin
    rss_lin = np.sum((Y_context - pred_lin_context)**2)
    # Approximate log marginal likelihood: -0.5 * rss / sigma^2 - 0.5 * log(det(A))
    log_lik_lin = -0.5 * rss_lin / sigma_eps**2 - 0.5 * np.log(np.linalg.det(A_lin))

    pred_quad_context = X_quad @ mean_quad
    rss_quad = np.sum((Y_context - pred_quad_context)**2)
    log_lik_quad = -0.5 * rss_quad / sigma_eps**2 - 0.5 * np.log(np.linalg.det(A_quad))

    # Posterior probabilities
    log_post_lin = log_lik_lin + np.log(task_prior[0])
    log_post_quad = log_lik_quad + np.log(task_prior[1])
    log_sum = np.logaddexp(log_post_lin, log_post_quad)
    post_lin = np.exp(log_post_lin - log_sum)
    post_quad = np.exp(log_post_quad - log_sum)

    # Bayes predictor: weighted average of predictions
    bayes_pred = post_lin * pred_lin + post_quad * pred_quad

    # Posterior variance: weighted average of variances + variance of means
    # Variance of prediction for linear model
    cov_lin = np.linalg.inv(A_lin) * sigma_eps**2
    var_lin = cov_lin[0,0] * x_query**2 + 2*cov_lin[0,1]*x_query + cov_lin[1,1]

    cov_quad = np.linalg.inv(A_quad) * sigma_eps**2
    var_quad = cov_quad[0,0] * x_query**4 + 2*cov_quad[0,1]*x_query**2 + cov_quad[1,1]

    # Total posterior variance
    post_var = post_lin * (var_lin + pred_lin**2) + post_quad * (var_quad + pred_quad**2) - bayes_pred**2

    return bayes_pred, post_var

def main_predictor(X_context, Y_context, x_query):
    """
    Main predictor: task-1-only oracle (linear regression only).
    This is a suboptimal predictor that ignores the quadratic task.
    """
    k = len(X_context)
    X_lin = np.column_stack([X_context, np.ones(k)])
    A_lin = X_lin.T @ X_lin + np.eye(2)
    b_lin = X_lin.T @ Y_context
    mean_lin = np.linalg.solve(A_lin, b_lin)
    pred_lin = mean_lin[0] * x_query + mean_lin[1]
    return pred_lin

def control_predictor(X_context, Y_context, x_query):
    """
    Control predictor: Bayes predictor.
    """
    bayes_pred, _ = bayes_predictor(X_context, Y_context, x_query)
    return bayes_pred

# Monte Carlo estimation
R_main = 0.0
RBG_main = 0.0
RPV_main = 0.0

R_control = 0.0
RBG_control = 0.0
RPV_control = 0.0

for _ in range(N):
    # Sample task and function
    i, f, f_params = sample_task()

    # Generate prompt
    X, Y = generate_prompt(f, p)

    # For each k in 1..p, compute risk
    for k in range(1, p+1):
        X_context = X[:k]
        Y_context = Y[:k]
        x_query = X[k]
        y_true = f(x_query)

        # Main predictor (task-1-only oracle)
        pred_main = main_predictor(X_context, Y_context, x_query)

        # Bayes predictor
        pred_bayes, post_var = bayes_predictor(X_context, Y_context, x_query)

        # ICL risk for main predictor
        R_main += (y_true - pred_main)**2

        # Bayes Gap: (pred_main - pred_bayes)^2
        RBG_main += (pred_main - pred_bayes)**2

        # Posterior Variance
        RPV_main += post_var

        # Control predictor (Bayes predictor)
        pred_control = control_predictor(X_context, Y_context, x_query)

        # ICL risk for control predictor
        R_control += (y_true - pred_control)**2

        # Bayes Gap for control (should be ~0)
        RBG_control += (pred_control - pred_bayes)**2

        # Posterior Variance for control
        RPV_control += post_var

# Average over N samples and p context lengths
R_main /= N * p
RBG_main /= N * p
RPV_main /= N * p

R_control /= N * p
RBG_control /= N * p
RPV_control /= N * p

# Check identity for main predictor
rel_diff_main = abs(R_main - (RBG_main + RPV_main)) / R_main if R_main > 0 else 0.0

# Check identity for control predictor
rel_diff_control = abs(R_control - (RBG_control + RPV_control)) / R_control if R_control > 0 else 0.0

# Control check: Bayes Gap for control should be ~0
control_pass = RBG_control < 0.01 * R_control if R_control > 0 else True

# Success criterion: rel_diff_main < 0.01
status = "supported" if rel_diff_main < 0.01 and control_pass else "falsified"

# Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(['R(M)', 'RBG(M)', 'RPV(M)'], [R_main, RBG_main, RPV_main], label='Main Predictor')
ax.bar(['R(M)', 'RBG(M)', 'RPV(M)'], [R_control, RBG_control, RPV_control], label='Control (Bayes)')
ax.set_ylabel('Risk')
ax.set_title('Risk Decomposition: R = RBG + RPV')
ax.legend()
plt.tight_layout()
plt.savefig(os.path.join(results_dir, 'fig.png'), dpi=150)
plt.close()

summary = {
    "claim_id": "C1",
    "status": status,
    "metrics": {
        "R_main": float(R_main),
        "RBG_main": float(RBG_main),
        "RPV_main": float(RPV_main),
        "rel_diff_main": float(rel_diff_main),
        "R_control": float(R_control),
        "RBG_control": float(RBG_control),
        "RPV_control": float(RPV_control),
        "rel_diff_control": float(rel_diff_control),
        "control_pass": bool(control_pass)
    },
    "notes": f"Main predictor (task-1-only oracle): R={R_main:.4f}, RBG={RBG_main:.4f}, RPV={RPV_main:.4f}, rel_diff={rel_diff_main:.4f}. Control (Bayes): RBG={RBG_control:.4f} (should be ~0)."
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    pass

import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Configuration ---
np.random.seed(42)

# Task parameters
T = 2  # Number of task types
alpha = np.array([0.5, 0.5])  # Prior probabilities
sigma_eps = 0.1  # Noise standard deviation
sigma_w = 1.0    # Prior std for weights

# Inference parameters
k_max = 20
n_test = 5000  # Number of test prompts for Monte Carlo estimation

# Control parameters
n_ctrl = 2000
k_ctrl_max = 10

# --- Task Definitions (Definition 2.1) ---
# Task 0: Linear Regression y = w*x + b
# Task 1: Quadratic Regression y = w2*x^2 + w1*x + b

def sample_task(task_idx):
    """Sample a function from the task family."""
    if task_idx == 0:
        w = np.random.normal(0, sigma_w)
        b = np.random.normal(0, sigma_w)
        return lambda x: w * x + b
    else:
        w2 = np.random.normal(0, sigma_w)
        w1 = np.random.normal(0, sigma_w)
        b = np.random.normal(0, sigma_w)
        return lambda x: w2 * x**2 + w1 * x + b

def generate_prompt(task_idx, k):
    """Generate a partial prompt of length k (k examples + 1 query)."""
    f = sample_task(task_idx)
    x_ctx = np.random.normal(0, 1, k)
    y_ctx = np.array([f(x) + np.random.normal(0, sigma_eps) for x in x_ctx])
    x_query = np.random.normal(0, 1)
    y_query = f(x_query) + np.random.normal(0, sigma_eps)
    return x_ctx, y_ctx, x_query, y_query, f

# --- Bayes Predictors ---

def bayes_predictor_oracle(x_query, x_ctx, y_ctx, task_idx):
    """
    Bayes predictor assuming knowledge of the true task family.
    For linear/quadratic regression with Gaussian priors and Gaussian noise,
    the posterior mean is the ridge regression solution (or exact linear regression if prior is flat,
    but here we use Gaussian prior so it's ridge).

    Actually, for the specific setup in the paper (Definition 2.1), the Bayes predictor
    is the posterior mean. For linear regression with Gaussian prior N(0, sigma_w^2) on weights
    and Gaussian noise N(0, sigma_eps^2), the posterior mean is given by the ridge regression
    solution with lambda = sigma_eps^2 / sigma_w^2.

    Let's implement the exact Bayesian linear regression posterior mean.
    """
    if task_idx == 0:
        # Linear: y = w*x + b
        # Design matrix: [x, 1]
        X = np.column_stack([x_ctx, np.ones(len(x_ctx))])
        # Prior: w ~ N(0, sigma_w^2 I)
        # Likelihood: y | X, w ~ N(Xw, sigma_eps^2 I)
        # Posterior mean: (X^T X + (sigma_eps^2/sigma_w^2) I)^-1 X^T y
        lam = sigma_eps**2 / sigma_w**2
        A = X.T @ X + lam * np.eye(2)
        b_vec = X.T @ y_ctx
        w_post = np.linalg.solve(A, b_vec)
        x_q_vec = np.array([x_query, 1.0])
        return x_q_vec @ w_post
    else:
        # Quadratic: y = w2*x^2 + w1*x + b
        # Design matrix: [x^2, x, 1]
        X = np.column_stack([x_ctx**2, x_ctx, np.ones(len(x_ctx))])
        lam = sigma_eps**2 / sigma_w**2
        A = X.T @ X + lam * np.eye(3)
        b_vec = X.T @ y_ctx
        w_post = np.linalg.solve(A, b_vec)
        x_q_vec = np.array([x_query**2, x_query, 1.0])
        return x_q_vec @ w_post

def bayes_predictor_mixture(x_query, x_ctx, y_ctx):
    """
    Bayes predictor for the mixture model.
    M_Bayes(P^k) = E_{I|D^k} E_{f|D^k, I} [f(x_{k+1})]
    = sum_i pi_i(D^k) * E_{f|D^k, I=i} [f(x_{k+1})]

    pi_i(D^k) = Pr(I=i | D^k) = alpha_i * p(D^k | I=i) / sum_j alpha_j * p(D^k | I=j)

    p(D^k | I=i) is the marginal likelihood of the data under task i.
    For Gaussian linear models, this can be computed in closed form.
    """
    k = len(x_ctx)

    def log_marginal_likelihood(task_idx):
        if task_idx == 0:
            X = np.column_stack([x_ctx, np.ones(k)])
        else:
            X = np.column_stack([x_ctx**2, x_ctx, np.ones(k)])

        # Marginal likelihood for Bayesian linear regression:
        # p(y|X) = (2*pi*sigma_eps^2)^(-k/2) * |Sigma_prior|^(1/2) / |Sigma_post|^(1/2) * exp(-0.5 * (y^T Sigma_post^-1 y - y^T Sigma_prior^-1 y))
        # Where Sigma_prior = sigma_eps^2 I + X Sigma_w X^T
        # Sigma_w = sigma_w^2 I
        # This is computationally expensive for large k, but k is small here.

        # Alternatively, use the formula:
        # p(y|X) = (2*pi*sigma_eps^2)^(-k/2) * (sigma_w^2)^(-d/2) * (sigma_w^2 + sigma_eps^2 X^T X)^(-1/2) * exp(-0.5 * y^T (sigma_eps^2 I + X (sigma_w^2 I)^-1 X^T)^-1 y)
        # Wait, the standard formula is:
        # p(y|X) = (2*pi*sigma_eps^2)^(-k/2) * |I + (sigma_eps^2/sigma_w^2) X^T X|^(-1/2) * exp(-0.5 * y^T (sigma_eps^2 I + X X^T)^-1 y) ... no.

        # Let's use the Cholesky decomposition approach for numerical stability.
        # p(y|X) = N(y; 0, sigma_eps^2 I + X (sigma_w^2 I) X^T)
        # Cov = sigma_eps^2 I + sigma_w^2 X X^T

        d = X.shape[1]
        # Use Woodbury identity or direct Cholesky if k is small.
        # Since k <= 20, we can compute the Cholesky of the k x k matrix.

        # Cov = sigma_eps^2 I_k + sigma_w^2 X X^T
        # Let's compute log det and quadratic form.

        # Use the identity: log |A + U C U^T| = log |A| + log |I + C^T A^-1 U C|
        # Here A = sigma_eps^2 I, U = X, C = sigma_w^2 I.
        # So log |Cov| = k * log(sigma_eps^2) + log |I + (sigma_w^2/sigma_eps^2) X^T X|

        # And the quadratic form y^T Cov^-1 y can be computed using the Sherman-Morrison-Woodbury formula.
        # Cov^-1 = (sigma_eps^2 I)^-1 - (sigma_eps^2 I)^-1 X (I + (sigma_w^2/sigma_eps^2) X^T X)^-1 X^T (sigma_eps^2 I)^-1
        # = (1/sigma_eps^2) I - (1/sigma_eps^4) X (I + (sigma_w^2/sigma_eps^2) X^T X)^-1 X^T

        lam = sigma_w**2 / sigma_eps**2
        M = np.eye(k) + lam * (X.T @ X)

        # Compute log det
        sign, logdet_M = np.linalg.slogdet(M)
        if sign <= 0:
            return -np.inf
        log_det_cov = k * np.log(sigma_eps**2) + logdet_M

        # Compute quadratic form
        # y^T Cov^-1 y = (1/sigma_eps^2) y^T y - (1/sigma_eps^4) y^T X M^-1 X^T y
        yX = X.T @ y_ctx
        M_inv_yX = np.linalg.solve(M, yX)
        quad_form = (1.0 / sigma_eps**2) * (y_ctx @ y_ctx) - (1.0 / sigma_eps**4) * (yX @ M_inv_yX)

        log_ml = -0.5 * (k * np.log(2 * np.pi * sigma_eps**2) + log_det_cov + quad_form)
        return log_ml

    log_ml_0 = log_marginal_likelihood(0)
    log_ml_1 = log_marginal_likelihood(1)

    # Compute posterior probabilities
    log_alpha_0 = np.log(alpha[0]) + log_ml_0
    log_alpha_1 = np.log(alpha[1]) + log_ml_1

    # Normalize
    max_log = max(log_alpha_0, log_alpha_1)
    exp_0 = np.exp(log_alpha_0 - max_log)
    exp_1 = np.exp(log_alpha_1 - max_log)

    pi_0 = exp_0 / (exp_0 + exp_1)
    pi_1 = exp_1 / (exp_0 + exp_1)

    # Compute conditional expectations
    pred_0 = bayes_predictor_oracle(x_query, x_ctx, y_ctx, 0)
    pred_1 = bayes_predictor_oracle(x_query, x_ctx, y_ctx, 1)

    return pi_0 * pred_0 + pi_1 * pred_1

# --- Main Experiment ---

def run_experiment():
    """Run the main experiment to test the claim."""
    mse_oracle = np.zeros(k_max + 1)
    mse_mixture = np.zeros(k_max + 1)

    for k in range(1, k_max + 1):
        errors_oracle = []
        errors_mixture = []

        for _ in range(n_test):
            # Sample a task
            task_idx = np.random.choice(T, p=alpha)

            # Generate prompt
            x_ctx, y_ctx, x_query, y_query, f = generate_prompt(task_idx, k)

            # Compute predictions
            pred_oracle = bayes_predictor_oracle(x_query, x_ctx, y_ctx, task_idx)
            pred_mixture = bayes_predictor_mixture(x_query, x_ctx, y_ctx)

            # Compute errors
            errors_oracle.append((y_query - pred_oracle)**2)
            errors_mixture.append((y_query - pred_mixture)**2)

        mse_oracle[k] = np.mean(errors_oracle)
        mse_mixture[k] = np.mean(errors_mixture)

        print(f"k={k}: MSE_oracle={mse_oracle[k]:.6f}, MSE_mixture={mse_mixture[k]:.6f}")

    return mse_oracle, mse_mixture

# --- Control Experiment ---

def run_control():
    """
    Positive control: Verify that the Bayes mixture predictor converges to the Bayes oracle predictor.
    We know that as k -> infinity, the posterior over task types should concentrate on the true task,
    so the mixture predictor should approach the oracle predictor.

    We test this by checking that the gap between mixture and oracle MSE decreases with k.
    """
    mse_oracle_ctrl = np.zeros(k_ctrl_max + 1)
    mse_mixture_ctrl = np.zeros(k_ctrl_max + 1)

    for k in range(1, k_ctrl_max + 1):
        errors_oracle = []
        errors_mixture = []

        for _ in range(n_ctrl):
            task_idx = np.random.choice(T, p=alpha)
            x_ctx, y_ctx, x_query, y_query, f = generate_prompt(task_idx, k)

            pred_oracle = bayes_predictor_oracle(x_query, x_ctx, y_ctx, task_idx)
            pred_mixture = bayes_predictor_mixture(x_query, x_ctx, y_ctx)

            errors_oracle.append((y_query - pred_oracle)**2)
            errors_mixture.append((y_query - pred_mixture)**2)

        mse_oracle_ctrl[k] = np.mean(errors_oracle)
        mse_mixture_ctrl[k] = np.mean(errors_mixture)

    # Check if the gap decreases
    gaps = mse_mixture_ctrl[1:] - mse_oracle_ctrl[1:]
    # The gap should be positive (mixture is worse) and decreasing
    control_pass = all(gaps[i] >= 0 for i in range(len(gaps)-1)) and gaps[-1] < gaps[0]

    return control_pass, mse_oracle_ctrl, mse_mixture_ctrl

# --- Main ---

if __name__ == "__main__":
    # Run control first
    control_pass, mse_oracle_ctrl, mse_mixture_ctrl = run_control()
    print(f"Control passed: {control_pass}")

    # Run main experiment
    mse_oracle, mse_mixture = run_experiment()

    # Compute gaps
    gaps = mse_mixture - mse_oracle

    # Success criterion:
    # 1. Gap decreases monotonically with k
    # 2. Gap at k=5 is less than 10% of gap at k=1

    monotonic = all(gaps[k] <= gaps[k-1] for k in range(2, k_max + 1))
    ratio_k5 = gaps[5] / gaps[1] if gaps[1] > 0 else 0
    criterion_met = monotonic and ratio_k5 < 0.1

    # Plot
    os.makedirs('results/c3', exist_ok=True)

    fig, ax = plt.subplots(figsize=(10, 6))
    k_vals = np.arange(1, k_max + 1)
    ax.plot(k_vals, mse_oracle, 'b-o', label='Bayes (Oracle)')
    ax.plot(k_vals, mse_mixture, 'r-s', label='Bayes (Mixture)')
    ax.set_xlabel('Number of in-context examples k')
    ax.set_ylabel('MSE')
    ax.set_title('Posterior Concentration: Mixture vs Oracle Bayes Predictor')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('results/c3/fig.png', dpi=150)
    plt.close()

    # Plot gaps
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_vals, gaps, 'g-^', label='Gap (Mixture - Oracle)')
    ax.set_xlabel('Number of in-context examples k')
    ax.set_ylabel('MSE Gap')
    ax.set_title('Gap between Mixture and Oracle Bayes Predictors')
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.savefig('results/c3/gaps.png', dpi=150)
    plt.close()

    # Determine status
    if not control_pass:
        status = "inconclusive"
        notes = "Control failed: the statistic is buggy or the setup is incorrect."
    elif criterion_met:
        status = "supported"
        notes = ("The gap between the Bayes mixture predictor and the Bayes oracle predictor "
                 "decreases monotonically and is less than 10% of the initial gap by k=5. "
                 "This supports the claim that task-type identification error vanishes rapidly. "
                 "Note: This audit tests the Bayes-optimal predictors, not a trained Transformer, "
                 "as the Transformer training experiment requires GPU and is out of CPU-audit scope.")
    else:
        status = "falsified"
        notes = ("The gap between the Bayes mixture predictor and the Bayes oracle predictor "
                 "does not meet the success criterion. Monotonic: " + str(monotonic) + ", "
                 "Ratio at k=5: " + str(ratio_k5) + ".")

    summary = {
        "claim_id": "C3",
        "status": status,
        "metrics": {
            "control_pass": bool(control_pass),
            "mse_oracle_k1": float(mse_oracle[1]),
            "mse_mixture_k1": float(mse_mixture[1]),
            "gap_k1": float(gaps[1]),
            "mse_oracle_k5": float(mse_oracle[5]),
            "mse_mixture_k5": float(mse_mixture[5]),
            "gap_k5": float(gaps[5]),
            "ratio_k5": float(ratio_k5),
            "monotonic": bool(monotonic),
            "criterion_met": bool(criterion_met)
        },
        "notes": notes
    }

    print("SUMMARY_JSON=" + json.dumps(summary, default=str))

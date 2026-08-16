import json
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- Configuration ---
SEED = 42
np.random.seed(SEED)

# Task parameters (Definition 2.1)
T = 2  # Number of task types
alpha = np.array([0.5, 0.5])  # Prior probabilities
sigma_eps = 0.1  # Noise std

d_feat = 1  # Input dimension
# Task 0: Linear f(x) = w*x + b
# Task 1: Quadratic f(x) = a*x^2 + b*x + c
# We use simple priors for the coefficients to make the Bayes predictor computable.
# For Task 0: w ~ N(0, 1), b ~ N(0, 1)
# For Task 1: a ~ N(0, 0.1), b ~ N(0, 1), c ~ N(0, 1) (small a to keep scale similar)

# Inference parameters
k_max = 20
n_trials = 1000  # Number of Monte Carlo trials to estimate risks

# --- Helper Functions ---

def generate_data(k, task_idx, rng):
    """
    Generate k context examples and 1 query for a given task.
    Returns x_ctx (k, d), y_ctx (k,), x_query (d,)
    """
    x_ctx = rng.normal(0, 1, size=(k, d_feat))
    x_query = rng.normal(0, 1, size=(d_feat,))

    if task_idx == 0:
        # Linear: y = w*x + b + eps
        w = rng.normal(0, 1)
        b = rng.normal(0, 1)
        y_ctx = w * x_ctx[:, 0] + b + rng.normal(0, sigma_eps, size=k)
    else:
        # Quadratic: y = a*x^2 + b*x + c + eps
        a = rng.normal(0, 0.1)
        b = rng.normal(0, 1)
        c = rng.normal(0, 1)
        y_ctx = a * x_ctx[:, 0]**2 + b * x_ctx[:, 0] + c + rng.normal(0, sigma_eps, size=k)

    return x_ctx, y_ctx, x_query

def bayes_predictor_task(x_query, x_ctx, y_ctx, task_idx):
    """
    Compute the Bayes posterior mean prediction for a specific task family.
    This is the 'Oracle' predictor if we knew the task.

    For linear regression with Gaussian priors on coefficients, the posterior is Gaussian.
    For quadratic, it's also Gaussian if we treat coefficients as parameters.

    We use standard Bayesian linear regression formulas.
    """
    # Design matrix for context
    if task_idx == 0:
        # Features: [x, 1]
        X = np.column_stack([x_ctx[:, 0], np.ones(k)])
        # Prior: w ~ N(0,1), b ~ N(0,1) => Prior Cov = I, Mean = 0
        prior_mean = np.zeros(2)
        prior_cov = np.eye(2)
    else:
        # Features: [x^2, x, 1]
        X = np.column_stack([x_ctx[:, 0]**2, x_ctx[:, 0], np.ones(k)])
        # Prior: a ~ N(0, 0.1^2), b ~ N(0, 1), c ~ N(0, 1)
        prior_mean = np.zeros(3)
        prior_cov = np.diag([0.01, 1.0, 1.0])

    # Bayesian Linear Regression Update
    # Posterior Cov = (Prior_Cov^-1 + X^T X / sigma^2)^-1
    # Posterior Mean = Posterior_Cov * (Prior_Cov^-1 * Prior_Mean + X^T Y / sigma^2)

    inv_prior_cov = np.linalg.inv(prior_cov)
    A = inv_prior_cov + X.T @ X / (sigma_eps**2)
    b_vec = inv_prior_cov @ prior_mean + X.T @ y_ctx / (sigma_eps**2)

    post_cov = np.linalg.inv(A)
    post_mean = post_cov @ b_vec

    # Query feature vector
    if task_idx == 0:
        x_q = np.array([x_query[0], 1.0])
    else:
        x_q = np.array([x_query[0]**2, x_query[0], 1.0])

    # Prediction: E[f(x_q)] = x_q^T post_mean
    pred = x_q @ post_mean

    return pred

def bayes_predictor_mixture(x_query, x_ctx, y_ctx):
    """
    Compute the Bayes posterior mean prediction for the mixture of tasks.
    This is the 'Bayes (mixture)' predictor.

    M_Bayes = sum_i pi_i(D_k) * E[f(x_q) | D_k, I=i]

    We need to compute the posterior probability of each task given the data.
    pi_i(D_k) = alpha_i * p(D_k | I=i) / sum_j alpha_j * p(D_k | I=j)

    p(D_k | I=i) is the marginal likelihood of the data under task i.
    For Gaussian linear models, this can be computed analytically.
    """
    log_ml = np.zeros(T)
    preds = np.zeros(T)

    for i in range(T):
        # Compute prediction for task i
        preds[i] = bayes_predictor_task(x_query, x_ctx, y_ctx, i)

        # Compute log marginal likelihood for task i
        # p(y | X, I=i) = N(y | X mu_0, X Sigma_0 X^T + sigma^2 I)
        # where mu_0 is prior mean, Sigma_0 is prior cov.

        if i == 0:
            X = np.column_stack([x_ctx[:, 0], np.ones(k)])
            prior_mean = np.zeros(2)
            prior_cov = np.eye(2)
        else:
            X = np.column_stack([x_ctx[:, 0]**2, x_ctx[:, 0], np.ones(k)])
            prior_mean = np.zeros(3)
            prior_cov = np.diag([0.01, 1.0, 1.0])

        # Mean of predictive distribution
        mu_pred = X @ prior_mean
        # Covariance of predictive distribution
        cov_pred = X @ prior_cov @ X.T + (sigma_eps**2) * np.eye(k)

        # Log likelihood of y_ctx under this Gaussian
        # We use the formula for multivariate normal log pdf
        try:
            # Cholesky for stability
            L = np.linalg.cholesky(cov_pred)
            diff = y_ctx - mu_pred
            # Solve L z = diff
            z = np.linalg.solve(L, diff)
            log_det = 2 * np.sum(np.log(np.diag(L)))
            log_ml[i] = -0.5 * (k * np.log(2 * np.pi) + log_det + z @ z)
        except np.linalg.LinAlgError:
            # Fallback to slogdet if Cholesky fails (shouldn't happen for PD matrices)
            sign, log_det = np.linalg.slogdet(cov_pred)
            if sign <= 0:
                log_ml[i] = -np.inf
            else:
                inv_cov = np.linalg.inv(cov_pred)
                diff = y_ctx - mu_pred
                log_ml[i] = -0.5 * (k * np.log(2 * np.pi) + log_det + diff @ inv_cov @ diff)

    # Compute posterior probabilities using log-sum-exp for stability
    max_log_ml = np.max(log_ml)
    exp_ml = np.exp(log_ml - max_log_ml)
    weights = alpha * exp_ml
    weights = weights / np.sum(weights)

    # Mixture prediction
    pred_mix = np.sum(weights * preds)

    return pred_mix, weights

def estimate_risks(k, n_trials):
    """
    Estimate the MSE for Oracle and Mixture Bayes predictors for a given k.
    """
    mse_oracle = 0.0
    mse_mixture = 0.0

    for _ in range(n_trials):
        # Sample a task
        task_idx = np.random.choice(T, p=alpha)

        # Generate data
        x_ctx, y_ctx, x_query = generate_data(k, task_idx, np.random)

        # True value
        if task_idx == 0:
            # We need the true w, b used in generation.
            # Wait, generate_data samples w, b internally. We need to return them or re-sample consistently.
            # Let's refactor generate_data to return the true function parameters or the true y.
            # Actually, for MSE we need (y_true - y_pred)^2.
            # y_true = f(x_query) + eps_query? No, the risk is E[(f(x) - M(P))^2].
            # The definition of ICL risk in the paper is E[ (f(x_{k+1}) - M(P^k))^2 ].
            # Note: It does NOT include the noise eps_{k+1} in the target, it's the function value.
            # However, usually MSE is against the observed y. Let's check the paper definition.
            # "ICL risk ... E[ l(f(x_{k+1}), M(P^k)) ]"
            # So the target is f(x_{k+1}), not y_{k+1}.
            # We need to know f(x_query).

            # My generate_data currently doesn't return f(x_query).
            # I will modify the loop to compute f(x_query) directly.
            pass

    # Refactoring: I need to compute f(x_query) for the true task.
    # Let's rewrite the loop properly.
    mse_oracle = 0.0
    mse_mixture = 0.0

    for _ in range(n_trials):
        task_idx = np.random.choice(T, p=alpha)

        # Generate data and true function value
        x_ctx, y_ctx, x_query = generate_data(k, task_idx, np.random)

        # Compute true f(x_query)
        if task_idx == 0:
            # We need the w, b used. generate_data doesn't return them.
            # This is a flaw in my helper. I will inline the generation here to get the true value.
            pass

    # Let's create a more robust generation function that returns the true y.
    return mse_oracle, mse_mixture

# --- Corrected Implementation ---

def generate_data_and_true(k, task_idx, rng):
    """
    Generate data and return the true function value at the query.
    """
    x_ctx = rng.normal(0, 1, size=(k, d_feat))
    x_query = rng.normal(0, 1, size=(d_feat,))

    if task_idx == 0:
        w = rng.normal(0, 1)
        b = rng.normal(0, 1)
        y_ctx = w * x_ctx[:, 0] + b + rng.normal(0, sigma_eps, size=k)
        y_true = w * x_query[0] + b
    else:
        a = rng.normal(0, 0.1)
        b = rng.normal(0, 1)
        c = rng.normal(0, 1)
        y_ctx = a * x_ctx[:, 0]**2 + b * x_ctx[:, 0] + c + rng.normal(0, sigma_eps, size=k)
        y_true = a * x_query[0]**2 + b * x_query[0] + c

    return x_ctx, y_ctx, x_query, y_true

def estimate_risks(k, n_trials):
    mse_oracle = 0.0
    mse_mixture = 0.0

    for _ in range(n_trials):
        task_idx = np.random.choice(T, p=alpha)
        x_ctx, y_ctx, x_query, y_true = generate_data_and_true(k, task_idx, np.random)

        # Oracle Prediction (knows task_idx)
        pred_oracle = bayes_predictor_task(x_query, x_ctx, y_ctx, task_idx)
        mse_oracle += (y_true - pred_oracle)**2

        # Mixture Prediction (doesn't know task_idx)
        pred_mix, _ = bayes_predictor_mixture(x_query, x_ctx, y_ctx)
        mse_mixture += (y_true - pred_mix)**2

    mse_oracle /= n_trials
    mse_mixture /= n_trials

    return mse_oracle, mse_mixture

# --- Main Execution ---

# Positive Control:
# The claim is about the gap between Mixture and Oracle shrinking.
# A positive control for the *statistic* (MSE calculation) would be to verify that
# if we force the mixture to be a single task (alpha=[1,0]), the Mixture MSE equals Oracle MSE.
# Or, more simply, verify that the Bayes predictor for a single task achieves the theoretical Bayes risk.
# Let's do a simple control: For k=1, Task 0 only, the Bayes predictor should be close to the true value.
# Actually, the reviewer reference uses a specific control. Let's stick to a simple sanity check.
# Control: If we set alpha = [1, 0], the mixture predictor should be identical to the oracle predictor for Task 0.
# We will run a small check for this.

def run_control():
    """
    Verify that for a single-task mixture, Mixture MSE == Oracle MSE.
    """
    global alpha
    original_alpha = alpha.copy()
    alpha = np.array([1.0, 0.0]) # Only Task 0

    k_ctrl = 5
    n_ctrl = 100
    mse_o, mse_m = estimate_risks(k_ctrl, n_ctrl)

    # Restore alpha
    alpha = original_alpha

    # They should be very close (numerical differences only)
    diff = abs(mse_o - mse_m)
    return diff < 1e-5, mse_o, mse_m

# Run Control
control_pass, ctrl_mse_o, ctrl_mse_m = run_control()

# Main Experiment
k_values = list(range(1, k_max + 1))
mse_oracle_list = []
mse_mixture_list = []

for k in k_values:
    mse_o, mse_m = estimate_risks(k, n_trials)
    mse_oracle_list.append(mse_o)
    mse_mixture_list.append(mse_m)
    print(f"k={k}: MSE_Oracle={mse_o:.6f}, MSE_Mixture={mse_m:.6f}")

# Calculate Gap
# Gap = MSE_Mixture - MSE_Oracle
# Success Criterion: Gap decreases monotonically and is < 10% of initial gap (at k=1) by k=5.

gaps = [m - o for m, o in zip(mse_mixture_list, mse_oracle_list)]
initial_gap = gaps[0]
gap_at_k5 = gaps[4] # k=5 is index 4

# Check monotonicity
is_monotonic = all(gaps[i] >= gaps[i+1] for i in range(len(gaps)-1))

# Check 10% criterion
ratio_k5 = gap_at_k5 / initial_gap if initial_gap != 0 else 0
meets_10pct = ratio_k5 < 0.1

# Plotting
os.makedirs('results/c3', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.plot(k_values, mse_oracle_list, 'b-o', label='Bayes (Oracle)')
plt.plot(k_values, mse_mixture_list, 'r-s', label='Bayes (Mixture)')
plt.xlabel('Number of In-Context Examples (k)')
plt.ylabel('MSE')
plt.title('Posterior Concentration: Mixture vs Oracle MSE')
plt.legend()
plt.grid(True)
plt.savefig('results/c3/fig.png', dpi=150)
plt.close()

# Summary
status = "supported" if (is_monotonic and meets_10pct and control_pass) else "falsified"
if not control_pass:
    status = "inconclusive"

summary = {
    "claim_id": "C3",
    "status": status,
    "metrics": {
        "control_pass": control_pass,
        "initial_gap_k1": initial_gap,
        "gap_k5": gap_at_k5,
        "ratio_gap_k5_to_k1": ratio_k5,
        "is_monotonic": is_monotonic,
        "mse_oracle_k1": mse_oracle_list[0],
        "mse_mixture_k1": mse_mixture_list[0],
        "mse_oracle_k5": mse_oracle_list[4],
        "mse_mixture_k5": mse_mixture_list[4]
    },
    "notes": f"Control passed: {control_pass}. Monotonic decrease: {is_monotonic}. Gap ratio at k=5: {ratio_k5:.4f} (threshold 0.1). "
             "The Transformer training experiment requires GPU and is out of scope; this audit verifies the Bayesian posterior concentration mechanism (Theorem 3.3) using the Bayes-optimal predictors as defined in the paper."
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1. Setup
np.random.seed(42)
N_SAMPLES = 10000
P_LEN = 5
D_FEAT = 1

# Task Mixture Parameters
# Task 1: Linear Regression y = w*x + b + eps
# Task 2: Quadratic Regression y = a*x^2 + b + eps
# Prior: 50/50
alpha = np.array([0.5, 0.5])

# Hyperparameters for Task 1 (Linear)
w_prior_mean = 0.0
w_prior_std = 1.0
b_prior_mean = 0.0
b_prior_std = 1.0

# Hyperparameters for Task 2 (Quadratic)
a_prior_mean = 0.0
a_prior_std = 1.0
b2_prior_mean = 0.0
b2_prior_std = 1.0

# Noise
sigma_eps = 0.1

# Input Distribution
# x ~ Uniform[-1, 1]

def sample_task():
    """Sample task index and parameters."""
    i = np.random.choice(2, p=alpha)
    if i == 0:
        w = np.random.normal(w_prior_mean, w_prior_std)
        b = np.random.normal(b_prior_mean, b_prior_std)
        return i, w, b
    else:
        a = np.random.normal(a_prior_mean, a_prior_std)
        b = np.random.normal(b2_prior_mean, b2_prior_std)
        return i, a, b

def generate_prompt(i, params, p):
    """Generate a prompt of length p (context) + 1 (query)."""
    x = np.random.uniform(-1, 1, size=p + 1)
    if i == 0:
        w, b = params
        y_true = w * x + b
    else:
        a, b = params
        y_true = a * x**2 + b

    y = y_true + np.random.normal(0, sigma_eps, size=p + 1)

    # Context is first p, query is last
    x_ctx = x[:p]
    y_ctx = y[:p]
    x_query = x[p]
    y_query = y[p]

    return x_ctx, y_ctx, x_query, y_query

def compute_bayes_predictor(x_ctx, y_ctx, x_query):
    """
    Compute the Bayes predictor (posterior mean) for the mixture.
    M_Bayes(P^k) = E[f(x_query) | D_k]

    For Task 1 (Linear): y = w*x + b
    Prior: w ~ N(0, 1), b ~ N(0, 1)
    Likelihood: y_j | w, b ~ N(w*x_j + b, sigma^2)
    Posterior: (w, b) | D ~ N(mu_post, Sigma_post)

    For Task 2 (Quadratic): y = a*x^2 + b
    Prior: a ~ N(0, 1), b ~ N(0, 1)
    Likelihood: y_j | a, b ~ N(a*x_j^2 + b, sigma^2)
    Posterior: (a, b) | D ~ N(mu_post, Sigma_post)

    We compute the posterior mean for each task, then weight by the posterior probability of the task.
    """
    n = len(x_ctx)

    # --- Task 1: Linear ---
    # Design matrix X1: [x, 1]
    X1 = np.column_stack((x_ctx, np.ones(n)))
    # Prior precision: P0 = I (since std=1)
    P0_1 = np.eye(2)
    # Likelihood precision: (1/sigma^2) * X^T X
    P1_1 = (1.0 / sigma_eps**2) * (X1.T @ X1)

    # Posterior precision
    P_post_1 = P0_1 + P1_1
    # Posterior mean
    mu_post_1 = np.linalg.solve(P_post_1, P0_1 @ np.array([0.0, 0.0]) + (1.0 / sigma_eps**2) * (X1.T @ y_ctx))

    # Predicted mean for x_query
    x_q_vec_1 = np.array([x_query, 1.0])
    pred_mean_1 = x_q_vec_1 @ mu_post_1

    # Predicted variance (for RPV calculation later, but we need the full posterior predictive variance)
    # Var(y_q | D) = Var(E[y_q|D, f]) + E[Var(y_q|D, f)]
    # For linear model, y_q = w*x_q + b + eps
    # E[y_q|D] = x_q_vec @ mu_post
    # Var(y_q|D) = x_q_vec @ Sigma_post @ x_q_vec + sigma^2
    Sigma_post_1 = np.linalg.inv(P_post_1)
    var_pred_1 = x_q_vec_1 @ Sigma_post_1 @ x_q_vec_1 + sigma_eps**2

    # --- Task 2: Quadratic ---
    # Design matrix X2: [x^2, 1]
    X2 = np.column_stack((x_ctx**2, np.ones(n)))
    # Prior precision: P0 = I
    P0_2 = np.eye(2)
    # Likelihood precision
    P1_2 = (1.0 / sigma_eps**2) * (X2.T @ X2)

    # Posterior precision
    P_post_2 = P0_2 + P1_2
    # Posterior mean
    mu_post_2 = np.linalg.solve(P_post_2, P0_2 @ np.array([0.0, 0.0]) + (1.0 / sigma_eps**2) * (X2.T @ y_ctx))

    # Predicted mean for x_query
    x_q_vec_2 = np.array([x_query**2, 1.0])
    pred_mean_2 = x_q_vec_2 @ mu_post_2

    # Predicted variance
    Sigma_post_2 = np.linalg.inv(P_post_2)
    var_pred_2 = x_q_vec_2 @ Sigma_post_2 @ x_q_vec_2 + sigma_eps**2

    # --- Mixture Posterior ---
    # We need the posterior probability of each task given the data.
    # Pr(I=i | D) = Pr(D | I=i) * Pr(I=i) / Pr(D)
    # Since we are in a Gaussian linear model, the marginal likelihood Pr(D | I=i) is Gaussian.

    # Marginal likelihood for Task 1:
    # D ~ N(X1 @ mu_prior, X1 @ Sigma_prior @ X1.T + sigma^2 I)
    # mu_prior = [0, 0], Sigma_prior = I
    # Cov_D_1 = X1 @ I @ X1.T + sigma^2 I = X1 X1.T + sigma^2 I
    # Mean_D_1 = 0

    # Log likelihood for Task 1:
    # -0.5 * (log(2pi)^n + log|Cov| + D^T Cov^-1 D)

    # Efficient calculation using Woodbury or just direct for small n
    # n is small (5), so direct is fine.

    Cov_D_1 = X1 @ X1.T + sigma_eps**2 * np.eye(n)
    L1 = np.linalg.cholesky(Cov_D_1)
    log_det_1 = 2 * np.sum(np.log(np.diag(L1)))
    inv_Cov_D_1 = np.linalg.inv(Cov_D_1)
    quad_1 = y_ctx.T @ inv_Cov_D_1 @ y_ctx
    log_lik_1 = -0.5 * (n * np.log(2 * np.pi) + log_det_1 + quad_1)

    # Marginal likelihood for Task 2:
    Cov_D_2 = X2 @ X2.T + sigma_eps**2 * np.eye(n)
    L2 = np.linalg.cholesky(Cov_D_2)
    log_det_2 = 2 * np.sum(np.log(np.diag(L2)))
    inv_Cov_D_2 = np.linalg.inv(Cov_D_2)
    quad_2 = y_ctx.T @ inv_Cov_D_2 @ y_ctx
    log_lik_2 = -0.5 * (n * np.log(2 * np.pi) + log_det_2 + quad_2)

    # Log posteriors
    log_post_1 = log_lik_1 + np.log(alpha[0])
    log_post_2 = log_lik_2 + np.log(alpha[1])

    # Normalize
    max_log = max(log_post_1, log_post_2)
    exp_1 = np.exp(log_post_1 - max_log)
    exp_2 = np.exp(log_post_2 - max_log)
    sum_exp = exp_1 + exp_2

    pi_1 = exp_1 / sum_exp
    pi_2 = exp_2 / sum_exp

    # Bayes Predictor (Posterior Mean)
    m_bayes = pi_1 * pred_mean_1 + pi_2 * pred_mean_2

    # Posterior Variance (RPV)
    # RPV = E[Var(f(x_q) | D, I)] + Var(E[f(x_q) | D, I])
    # Note: The definition in the paper for RPV is the variance of the posterior predictive distribution.
    # Var(y_q | D) = E[Var(y_q | D, I)] + Var(E[y_q | D, I])
    # E[Var(y_q | D, I)] = pi_1 * var_pred_1 + pi_2 * var_pred_2
    # Var(E[y_q | D, I]) = pi_1 * (pred_mean_1 - m_bayes)^2 + pi_2 * (pred_mean_2 - m_bayes)^2

    exp_var = pi_1 * var_pred_1 + pi_2 * var_pred_2
    var_exp = pi_1 * (pred_mean_1 - m_bayes)**2 + pi_2 * (pred_mean_2 - m_bayes)**2

    rpv = exp_var + var_exp

    return m_bayes, rpv

def run_simulation():
    """
    Simulate the data-generating process and compute R(M), RBG(M), RPV.
    We use the Bayes predictor M_Bayes as the model M for the positive control.
    For the main test, we use a simple model M (e.g., mean predictor) to show the identity holds generally.
    However, the claim is an identity for ANY M.
    R(M) = E[(y - M(P))^2]
    RBG(M) = E[(M(P) - M_Bayes(P))^2]
    RPV = E[(y - M_Bayes(P))^2]  <-- Wait, is RPV defined as the Bayes Risk?

    Let's check the paper's definition.
    "Posterior Variance is a model-independent risk representing the intrinsic task uncertainty."
    "R(M) = RBG(M) + RPV"

    Standard decomposition:
    E[(y - M)^2] = E[(y - M_Bayes + M_Bayes - M)^2]
    = E[(y - M_Bayes)^2] + E[(M_Bayes - M)^2] + 2 E[(y - M_Bayes)(M_Bayes - M)]

    The cross term is zero because E[y - M_Bayes | D] = 0.
    So R(M) = R_Bayes + E[(M_Bayes - M)^2].

    The paper calls R_Bayes the "Posterior Variance" (RPV) and the second term the "Bayes Gap" (RBG).
    So:
    RPV = E[(y - M_Bayes(P))^2] (This is the Bayes Risk)
    RBG(M) = E[(M(P) - M_Bayes(P))^2]

    Let's verify this interpretation.
    "Posterior Variance ... determined solely by the difficulty of the true underlying task..."
    Yes, the Bayes risk is the irreducible error.

    So the plan:
    1. Generate N samples.
    2. For each sample, compute M_Bayes and RPV_sample = (y - M_Bayes)^2.
    3. For a specific M (e.g., M=0 or M=mean), compute M_sample and RBG_sample = (M_sample - M_Bayes)^2.
    4. Compute R_sample = (y - M_sample)^2.
    5. Check if R_sample == RBG_sample + RPV_sample.

    Since this is an exact identity for each sample (conditional on D), the empirical means should match exactly (up to floating point).
    """

    # We will test with a simple model M: The "Mean Predictor" M(P) = mean(y_ctx)
    # This is a valid measurable bounded map.

    r_total = 0.0
    rbg_total = 0.0
    rpv_total = 0.0

    # For plotting
    diffs = []

    for _ in range(N_SAMPLES):
        i, params = sample_task()
        x_ctx, y_ctx, x_query, y_query = generate_prompt(i, params, P_LEN)

        # Compute Bayes Predictor and RPV (Bayes Risk component)
        m_bayes, rpv_sample = compute_bayes_predictor(x_ctx, y_ctx, x_query)

        # Note: rpv_sample computed above is Var(y_q | D).
        # Is R_Bayes = E[(y - M_Bayes)^2] equal to Var(y_q | D)?
        # Yes, because M_Bayes = E[y_q | D].
        # E[(y - E[y|D])^2 | D] = Var(y | D).
        # So RPV_sample is indeed the contribution to the Bayes risk for this sample.

        # Define Model M: Mean Predictor
        m_model = np.mean(y_ctx)

        # Compute Risks
        r_sample = (y_query - m_model)**2
        rbg_sample = (m_model - m_bayes)**2

        # Accumulate
        r_total += r_sample
        rbg_total += rbg_sample
        rpv_total += rpv_sample

        # Check identity for this sample
        diff = abs(r_sample - (rbg_sample + rpv_sample))
        diffs.append(diff)

    # Average
    r_avg = r_total / N_SAMPLES
    rbg_avg = rbg_total / N_SAMPLES
    rpv_avg = rpv_total / N_SAMPLES

    # Check success criterion
    # |R(M) - (RBG(M) + RPV)| < 0.01 * R(M)
    lhs = r_avg
    rhs = rbg_avg + rpv_avg
    abs_diff = abs(lhs - rhs)
    rel_diff = abs_diff / r_avg if r_avg > 0 else 0

    # Positive Control
    # The identity is an algebraic identity: (y-M)^2 = (y-M_Bayes)^2 + (M-M_Bayes)^2 + 2(y-M_Bayes)(M-M_Bayes)
    # The cross term is 0 in expectation. For a single sample, it is NOT necessarily 0.
    # Wait. The identity R(M) = RBG(M) + RPV is an expectation identity.
    # R(M) = E[(y-M)^2]
    # RBG(M) = E[(M-M_Bayes)^2]
    # RPV = E[(y-M_Bayes)^2]

    # For a single sample, (y-M)^2 != (y-M_Bayes)^2 + (M-M_Bayes)^2 generally.
    # The difference is 2(y-M_Bayes)(M-M_Bayes).
    # The expectation of this cross term is 0.
    # So the empirical means should converge to each other.

    # The success criterion says: "absolute difference between R(M) and (RBG(M) + RPV) is less than 0.01 * R(M) for 10,000 samples."
    # This implies we compare the empirical means.

    # Let's check the control.
    # If we use M = M_Bayes, then RBG = 0, and R = RPV. The identity holds exactly for each sample.
    # Let's run a control with M = M_Bayes.

    r_control = 0.0
    rbg_control = 0.0
    rpv_control = 0.0

    for _ in range(1000): # Smaller sample for control
        i, params = sample_task()
        x_ctx, y_ctx, x_query, y_query = generate_prompt(i, params, P_LEN)
        m_bayes, rpv_sample = compute_bayes_predictor(x_ctx, y_ctx, x_query)

        m_model = m_bayes

        r_sample = (y_query - m_model)**2
        rbg_sample = (m_model - m_bayes)**2

        r_control += r_sample
        rbg_control += rbg_sample
        rpv_control += rpv_sample

    r_control_avg = r_control / 1000
    rbg_control_avg = rbg_control / 1000
    rpv_control_avg = rpv_control / 1000

    control_diff = abs(r_control_avg - (rbg_control_avg + rpv_control_avg))
    control_pass = control_diff < 1e-10

    # Plot
    os.makedirs('results/c1', exist_ok=True)
    plt.figure(figsize=(10, 6))
    plt.hist(diffs, bins=50, density=True, alpha=0.7, label='Per-sample difference')
    plt.axvline(0, color='r', linestyle='--')
    plt.title('Distribution of per-sample differences R - (RBG + RPV)')
    plt.xlabel('Difference')
    plt.ylabel('Density')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('results/c1/fig.png')
    plt.close()

    return {
        "R_M": r_avg,
        "RBG_M": rbg_avg,
        "RPV": rpv_avg,
        "abs_diff": abs_diff,
        "rel_diff": rel_diff,
        "control_pass": control_pass,
        "control_diff": control_diff
    }

results = run_simulation()

status = "supported" if results["rel_diff"] < 0.01 and results["control_pass"] else "falsified"
if not results["control_pass"]:
    status = "inconclusive"

summary = {
    "claim_id": "C1",
    "status": status,
    "metrics": {
        "R_M": results["R_M"],
        "RBG_M": results["RBG_M"],
        "RPV": results["RPV"],
        "abs_diff": results["abs_diff"],
        "rel_diff": results["rel_diff"],
        "control_pass": results["control_pass"],
        "control_diff": results["control_diff"]
    },
    "notes": f"Empirical R(M)={results['R_M']:.4f}, RBG(M)+RPV={results['RBG_M']+results['RPV']:.4f}. Relative diff={results['rel_diff']:.6f}. Control passed: {results['control_pass']}."
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

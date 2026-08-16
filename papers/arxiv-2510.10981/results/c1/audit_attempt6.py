import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Setup
np.random.seed(42)

# Hyperparameters
M = 5000  # Number of Monte Carlo samples
p = 10    # Context length
T = 2     # Number of task families
alpha = np.array([0.5, 0.5])  # Mixture weights
sigma_eps = 1.0  # Noise standard deviation

# Task Family 1: Linear Regression y = w*x + b
# Prior: w ~ N(0, 1), b ~ N(0, 1)
# Task Family 2: Quadratic Regression y = a*x^2 + b
# Prior: a ~ N(0, 1), b ~ N(0, 1)

# Input distribution: x ~ U(-1, 1)

def sample_task_and_data(k, I):
    """
    Sample a task function and k data points (x, y) for a given task type I.
    Returns: x (k,), y (k,), f_params (dict)
    """
    if I == 0:  # Linear
        w = np.random.randn()
        b = np.random.randn()
        x = np.random.uniform(-1, 1, k)
        y = w * x + b + sigma_eps * np.random.randn(k)
        params = {'w': w, 'b': b}
    else:  # Quadratic
        a = np.random.randn()
        b = np.random.randn()
        x = np.random.uniform(-1, 1, k)
        y = a * x**2 + b + sigma_eps * np.random.randn(k)
        params = {'a': a, 'b': b}
    return x, y, params

def compute_posterior_mean(x_query, D_k, I_posterior):
    """
    Compute the Bayes posterior mean E[f(x_query) | D_k].
    D_k: list of (x, y) pairs of length k.
    I_posterior: dict {0: pi_0, 1: pi_1}
    """
    k = len(D_k)
    X = np.array([d[0] for d in D_k])
    Y = np.array([d[1] for d in D_k])

    # Posterior for Family 1 (Linear)
    # Prior: w ~ N(0,1), b ~ N(0,1)
    # Likelihood: y | w, b ~ N(w*x + b, sigma^2)
    # Posterior is Gaussian. We can compute the posterior mean of f(x_query) = w*x_query + b.
    # E[w*x_query + b | D] = x_query * E[w|D] + E[b|D]

    # Design matrix for linear regression
    A1 = np.column_stack([X, np.ones(k)])
    # Prior precision: I (identity)
    # Posterior precision: A1.T @ A1 / sigma^2 + I
    # Posterior mean: (A1.T @ A1 / sigma^2 + I)^-1 @ (A1.T @ Y / sigma^2)

    # For efficiency and stability, we use the standard Bayesian linear regression formulas.
    # Let Sigma_prior = I (2x2)
    # Sigma_post = (Sigma_prior^-1 + A1.T @ A1 / sigma^2)^-1
    # mu_post = Sigma_post @ (Sigma_prior^-1 @ mu_prior + A1.T @ Y / sigma^2)
    # mu_prior = [0, 0]

    Sigma_prior_inv = np.eye(2)
    A1T_A1 = A1.T @ A1
    Sigma_post = np.linalg.inv(Sigma_prior_inv + A1T_A1 / sigma_eps**2)
    mu_post = Sigma_post @ (A1.T @ Y / sigma_eps**2)

    # E[f(x_query) | I=0, D] = [x_query, 1] @ mu_post
    pred_0 = np.array([x_query, 1.0]) @ mu_post

    # Posterior for Family 2 (Quadratic)
    # Prior: a ~ N(0,1), b ~ N(0,1)
    # Likelihood: y | a, b ~ N(a*x^2 + b, sigma^2)
    # Design matrix for quadratic regression (features: x^2, 1)
    A2 = np.column_stack([X**2, np.ones(k)])

    Sigma_post_2 = np.linalg.inv(Sigma_prior_inv + A2.T @ A2 / sigma_eps**2)
    mu_post_2 = Sigma_post_2 @ (A2.T @ Y / sigma_eps**2)

    # E[f(x_query) | I=1, D] = [x_query^2, 1] @ mu_post_2
    pred_1 = np.array([x_query**2, 1.0]) @ mu_post_2

    # Bayes Predictor: weighted sum of family-specific posterior means
    bayes_pred = I_posterior[0] * pred_0 + I_posterior[1] * pred_1

    return bayes_pred, pred_0, pred_1

def compute_posterior_variance(x_query, D_k, I_posterior):
    """
    Compute the Posterior Variance Var(f(x_query) | D_k).
    This is the variance of the predictive distribution.
    Var(f(x_query) | D_k) = E[Var(f(x_query) | I, D_k) | D_k] + Var(E[f(x_query) | I, D_k] | D_k)

    For linear/quadratic regression with Gaussian priors, the conditional variance is:
    Var(f(x_query) | I, D_k) = [x_query, 1] @ Sigma_post @ [x_query, 1]^T + sigma^2
    (Note: The +sigma^2 is for the noise in y, but the definition of RPV in the paper is
     Var(f(x_{k+1}) | D_k). Since y = f(x) + eps, and we are predicting f(x),
     the variance of f(x) given D is just the parameter uncertainty part.
     However, the ICL risk is E[(y - M)^2] = E[(f(x) + eps - M)^2].
     The Bayes risk decomposition is R(M) = E[(M - E[f|D])^2] + E[Var(f|D)].
     Wait, the standard decomposition is:
     E[(y - M)^2] = E[(y - E[f|D])^2] + E[(E[f|D] - M)^2]
     = E[Var(y|D)] + E[(E[f|D] - M)^2]
     Var(y|D) = E[Var(y|f,D)|D] + Var(E[y|f,D]|D) = sigma^2 + Var(f(x)|D).
     So RPV in the paper likely refers to E[Var(f(x)|D)] or E[Var(y|D)]?
     The paper says "Posterior Variance RPV ... represents the intrinsic task uncertainty".
     Usually, the irreducible risk is the expected posterior variance of the target.
     If the target is y, it's sigma^2 + Var(f|D).
     If the target is f(x), it's Var(f|D).
     The risk is defined as E[l(f(x_{k+1}), M(P^k))]. The target is f(x_{k+1}).
     So the Bayes predictor is E[f(x_{k+1})|D].
     The Bayes risk is E[Var(f(x_{k+1})|D)].
     So RPV = E[Var(f(x_{k+1})|D)].

     Let's compute Var(f(x_query) | I, D_k).
     For linear: Var(w*x + b | D) = [x, 1] @ Sigma_post @ [x, 1]^T.
     For quadratic: Var(a*x^2 + b | D) = [x^2, 1] @ Sigma_post_2 @ [x^2, 1]^T.
    """
    k = len(D_k)
    X = np.array([d[0] for d in D_k])
    Y = np.array([d[1] for d in D_k])

    # Family 1
    A1 = np.column_stack([X, np.ones(k)])
    Sigma_prior_inv = np.eye(2)
    A1T_A1 = A1.T @ A1
    Sigma_post = np.linalg.inv(Sigma_prior_inv + A1T_A1 / sigma_eps**2)

    var_0 = np.array([x_query, 1.0]) @ Sigma_post @ np.array([x_query, 1.0])

    # Family 2
    A2 = np.column_stack([X**2, np.ones(k)])
    Sigma_post_2 = np.linalg.inv(Sigma_prior_inv + A2.T @ A2 / sigma_eps**2)

    var_1 = np.array([x_query**2, 1.0]) @ Sigma_post_2 @ np.array([x_query**2, 1.0])

    # Total Variance = E[Var(f|I,D)|D] + Var(E[f|I,D]|D)
    # We need the posterior means to compute the second term.
    mu_post = Sigma_post @ (A1.T @ Y / sigma_eps**2)
    pred_0 = np.array([x_query, 1.0]) @ mu_post

    mu_post_2 = Sigma_post_2 @ (A2.T @ Y / sigma_eps**2)
    pred_1 = np.array([x_query**2, 1.0]) @ mu_post_2

    # E[Var(f|I,D)|D] = pi_0 * var_0 + pi_1 * var_1
    exp_var = I_posterior[0] * var_0 + I_posterior[1] * var_1

    # Var(E[f|I,D]|D) = pi_0 * (pred_0 - bayes_pred)^2 + pi_1 * (pred_1 - bayes_pred)^2
    # where bayes_pred = pi_0 * pred_0 + pi_1 * pred_1
    bayes_pred = I_posterior[0] * pred_0 + I_posterior[1] * pred_1
    var_mean = I_posterior[0] * (pred_0 - bayes_pred)**2 + I_posterior[1] * (pred_1 - bayes_pred)**2

    total_var = exp_var + var_mean

    return total_var

def compute_posterior_probs(D_k):
    """
    Compute posterior probabilities pi_i(D_k) = Pr(I=i | D_k).
    pi_i(D_k) = alpha_i * P(D_k | I=i) / sum_j alpha_j * P(D_k | I=j)

    P(D_k | I=i) = prod_{j=1}^k P(y_j | x_j, I=i)
    Since x_j are fixed, this is the likelihood of the parameters integrated out.
    For Gaussian linear models, the marginal likelihood is:
    P(y | X, I=i) = (2*pi*sigma^2)^(-k/2) * |Sigma_prior|^(1/2) / |Sigma_post|^(1/2) * exp(-0.5 * (y - X@mu_prior)^T (Sigma_prior + X^T X / sigma^2)^-1 (y - X@mu_prior))

    We can compute the log-likelihood for each family.
    """
    k = len(D_k)
    X = np.array([d[0] for d in D_k])
    Y = np.array([d[1] for d in D_k])

    # Family 1: Linear
    A1 = np.column_stack([X, np.ones(k)])
    Sigma_prior = np.eye(2)
    mu_prior = np.zeros(2)

    # Log marginal likelihood for linear regression
    # log P(y|X) = -k/2 log(2*pi*sigma^2) + 0.5 log|Sigma_prior| - 0.5 log|Sigma_post| - 0.5 (y - X@mu_prior)^T (Sigma_prior + X^T X / sigma^2)^-1 (y - X@mu_prior)

    Sigma_prior_inv = np.linalg.inv(Sigma_prior)
    A1T_A1 = A1.T @ A1
    Sigma_post = np.linalg.inv(Sigma_prior_inv + A1T_A1 / sigma_eps**2)

    # Determinants
    log_det_prior = np.linalg.slogdet(Sigma_prior)[1]
    log_det_post = np.linalg.slogdet(Sigma_post)[1]

    # Quadratic form
    diff = Y - A1 @ mu_prior
    quad_form = diff @ (Sigma_prior + A1T_A1 / sigma_eps**2) @ diff

    log_lik_0 = -k/2 * np.log(2 * np.pi * sigma_eps**2) + 0.5 * log_det_prior - 0.5 * log_det_post - 0.5 * quad_form

    # Family 2: Quadratic
    A2 = np.column_stack([X**2, np.ones(k)])

    Sigma_post_2 = np.linalg.inv(Sigma_prior_inv + A2.T @ A2 / sigma_eps**2)

    log_det_post_2 = np.linalg.slogdet(Sigma_post_2)[1]

    diff_2 = Y - A2 @ mu_prior
    quad_form_2 = diff_2 @ (Sigma_prior + A2.T @ A2 / sigma_eps**2) @ diff_2

    log_lik_1 = -k/2 * np.log(2 * np.pi * sigma_eps**2) + 0.5 * log_det_prior - 0.5 * log_det_post_2 - 0.5 * quad_form_2

    # Posterior probabilities
    log_pi_0 = np.log(alpha[0]) + log_lik_0
    log_pi_1 = np.log(alpha[1]) + log_lik_1

    # Normalize
    log_sum = np.logaddexp(log_pi_0, log_pi_1)
    pi_0 = np.exp(log_pi_0 - log_sum)
    pi_1 = np.exp(log_pi_1 - log_sum)

    return {'0': pi_0, '1': pi_1}

# Main Simulation
# We will simulate M prompts. For each prompt, we sample a task I, a function f, and p+1 data points.
# We compute the risk R(M) for a specific predictor M.
# We also compute RBG(M) and RPV.

# Predictor M_main: Task-1-only oracle. M(P^k) = E[f(x_{k+1}) | I=0, D_k].
# This assumes the task is always linear.

# Predictor M_ctrl: Bayes predictor. M(P^k) = E[f(x_{k+1}) | D_k].

R_main = 0.0
RBG_main = 0.0
RPV_main = 0.0

R_ctrl = 0.0
RBG_ctrl = 0.0
RPV_ctrl = 0.0

# We need to average over k=1..p as well? The definition of R(M) is 1/p sum_{k=1}^p E[...].
# So for each prompt, we should compute the loss for each k=1..p and average them.

for m in range(M):
    # Sample task
    I = np.random.choice(T, p=1, p=alpha)[0]

    # Sample function and data
    # We need p+1 points: x_1..x_p, y_1..y_p and x_{p+1}, y_{p+1}
    # But the risk is defined for predicting y_{k+1} given D_k.
    # So we need to generate a full sequence of p+1 points.

    if I == 0:
        w = np.random.randn()
        b = np.random.randn()
        X_seq = np.random.uniform(-1, 1, p + 1)
        Y_seq = w * X_seq + b + sigma_eps * np.random.randn(p + 1)
    else:
        a = np.random.randn()
        b = np.random.randn()
        X_seq = np.random.uniform(-1, 1, p + 1)
        Y_seq = a * X_seq**2 + b + sigma_eps * np.random.randn(p + 1)

    # For each k in 1..p
    for k in range(1, p + 1):
        D_k = [(X_seq[j], Y_seq[j]) for j in range(k)]
        x_query = X_seq[k]
        y_true = Y_seq[k] # This is f(x_query) + eps. The risk is E[(f(x_query) - M)^2].
        # Wait, the definition is E[l(f(x_{k+1}), M(P^k))].
        # So the target is f(x_{k+1}), not y_{k+1}.
        # We need to compute f(x_query) exactly.
        if I == 0:
            f_x = w * x_query + b
        else:
            f_x = a * x_query**2 + b

        # Compute posterior probabilities
        I_post = compute_posterior_probs(D_k)

        # Compute Bayes predictor (Control)
        bayes_pred, pred_0, pred_1 = compute_posterior_mean(x_query, D_k, I_post)

        # Compute Main predictor (Task-1 only)
        # M_main(P^k) = E[f(x_query) | I=0, D_k] = pred_0
        main_pred = pred_0

        # Compute Posterior Variance
        # RPV = E[Var(f(x_query) | D_k)]
        # We compute the variance for this specific D_k.
        var_f = compute_posterior_variance(x_query, D_k, I_post)

        # Accumulate risks
        # R(M) = E[(f(x) - M(P^k))^2]
        R_main += (f_x - main_pred)**2
        R_ctrl += (f_x - bayes_pred)**2

        # RBG(M) = E[(M(P^k) - M_Bayes(P^k))^2]
        RBG_main += (main_pred - bayes_pred)**2
        RBG_ctrl += (bayes_pred - bayes_pred)**2 # Should be 0

        # RPV = E[Var(f(x) | D_k)]
        RPV_main += var_f
        RPV_ctrl += var_f

# Average over M prompts and p lengths
# Total number of terms is M * p
N_terms = M * p

R_main /= N_terms
RBG_main /= N_terms
RPV_main /= N_terms

R_ctrl /= N_terms
RBG_ctrl /= N_terms
RPV_ctrl /= N_terms

# Check identity: R = RBG + RPV
# Note: The identity is R(M) = RBG(M) + RPV.
# Let's verify.

rel_diff_main = abs(R_main - (RBG_main + RPV_main)) / R_main if R_main > 0 else 0.0
rel_diff_ctrl = abs(R_ctrl - (RBG_ctrl + RPV_ctrl)) / R_ctrl if R_ctrl > 0 else 0.0

# Control check
control_pass = (rel_diff_ctrl < 0.02) and (RBG_ctrl < 1e-6)

# Main check
# Success: rel_diff < 0.01 AND RBG_main > 1e-6 AND control_pass
status = "supported" if (rel_diff_main < 0.01 and RBG_main > 1e-6 and control_pass) else "falsified"

# Plotting
os.makedirs('results/c1', exist_ok=True)

# Plot the decomposition for a few examples to visualize
# We'll just plot the values we computed.

fig, ax = plt.subplots(1, 2, figsize=(10, 4))

# Main
ax[0].bar(['R', 'RBG', 'RPV'], [R_main, RBG_main, RPV_main], color=['blue', 'orange', 'green'])
ax[0].set_title(f'Main Predictor (Task-1 Only)\nR={R_main:.4f}, RBG={RBG_main:.4f}, RPV={RPV_main:.4f}')
ax[0].set_ylabel('Risk')

# Control
ax[1].bar(['R', 'RBG', 'RPV'], [R_ctrl, RBG_ctrl, RPV_ctrl], color=['blue', 'orange', 'green'])
ax[1].set_title(f'Control Predictor (Bayes)\nR={R_ctrl:.4f}, RBG={RBG_ctrl:.4f}, RPV={RPV_ctrl:.4f}')
ax[1].set_ylabel('Risk')

plt.tight_layout()
plt.savefig('results/c1/fig.png')
plt.close()

summary = {
    "claim_id": "C1",
    "status": status,
    "metrics": {
        "R_main": float(R_main),
        "RBG_main": float(RBG_main),
        "RPV_main": float(RPV_main),
        "rel_diff_main": float(rel_diff_main),
        "R_ctrl": float(R_ctrl),
        "RBG_ctrl": float(RBG_ctrl),
        "RPV_ctrl": float(RPV_ctrl),
        "rel_diff_ctrl": float(rel_diff_ctrl),
        "control_pass": bool(control_pass)
    },
    "notes": f"Main: R={R_main:.4f}, RBG+RPV={RBG_main+RPV_main:.4f}, diff={rel_diff_main:.4f}. Control: R={R_ctrl:.4f}, RBG+RPV={RBG_ctrl+RPV_ctrl:.4f}, diff={rel_diff_ctrl:.4f}."
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

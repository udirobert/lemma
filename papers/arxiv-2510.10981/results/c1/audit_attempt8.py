import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(42)

# Setup
T = 2
alpha = np.array([0.5, 0.5])
dfeat = 1
sigma_eps = 0.1
p = 10
M = 5000

# Task 1: y = w*x + b, w ~ N(0,1), b ~ N(0,1)
# Task 2: y = a*x^2 + b, a ~ N(0,1), b ~ N(0,1)
# x ~ N(0,1)

def sample_task(i):
    if i == 0:
        w = np.random.randn()
        b = np.random.randn()
        return (w, b)
    else:
        a = np.random.randn()
        b = np.random.randn()
        return (a, b)

def predict_task(i, params, x):
    if i == 0:
        w, b = params
        return w * x + b
    else:
        a, b = params
        return a * x**2 + b

def posterior_mean_task1(x_ctx, y_ctx, x_query):
    # Linear regression: y = w*x + b
    # Prior: w ~ N(0,1), b ~ N(0,1)
    # Likelihood: y_j = w*x_j + b + eps, eps ~ N(0, sigma^2)
    # Posterior is Gaussian. We can compute the posterior mean analytically.
    # Design matrix A: rows are [x_j, 1]
    A = np.column_stack([x_ctx, np.ones_like(x_ctx)])
    # Prior precision: I (identity for [w,b])
    # Posterior precision: A^T A / sigma^2 + I
    # Posterior mean: (A^T A / sigma^2 + I)^{-1} (A^T y / sigma^2)
    # Let's compute this.
    n = len(x_ctx)
    if n == 0:
        return 0.0
    AtA = A.T @ A
    Aty = A.T @ y_ctx
    prec = AtA / sigma_eps**2 + np.eye(2)
    mean = np.linalg.solve(prec, Aty / sigma_eps**2)
    w_post, b_post = mean
    return w_post * x_query + b_post

def posterior_var_task1(x_ctx, y_ctx, x_query):
    # Var(f(x_query) | D_k) = Var(w*x_query + b | D_k) + sigma_eps^2
    # = [x_query, 1] Cov([w,b]|D_k) [x_query, 1]^T + sigma_eps^2
    A = np.column_stack([x_ctx, np.ones_like(x_ctx)])
    AtA = A.T @ A
    prec = AtA / sigma_eps**2 + np.eye(2)
    cov = np.linalg.inv(prec)
    v = np.array([x_query, 1.0])
    return v @ cov @ v + sigma_eps**2

def posterior_mean_task2(x_ctx, y_ctx, x_query):
    # Quadratic regression: y = a*x^2 + b
    # Prior: a ~ N(0,1), b ~ N(0,1)
    A = np.column_stack([x_ctx**2, np.ones_like(x_ctx)])
    AtA = A.T @ A
    Aty = A.T @ y_ctx
    prec = AtA / sigma_eps**2 + np.eye(2)
    mean = np.linalg.solve(prec, Aty / sigma_eps**2)
    a_post, b_post = mean
    return a_post * x_query**2 + b_post

def posterior_var_task2(x_ctx, y_ctx, x_query):
    A = np.column_stack([x_ctx**2, np.ones_like(x_ctx)])
    AtA = A.T @ A
    prec = AtA / sigma_eps**2 + np.eye(2)
    cov = np.linalg.inv(prec)
    v = np.array([x_query**2, 1.0])
    return v @ cov @ v + sigma_eps**2

def bayes_predictor(x_query, x_ctx, y_ctx, k):
    # Compute posterior over task index I given D_k
    # log p(D_k | I=i) = sum_{j=1}^k log p(y_j | x_j, I=i)
    # For Gaussian likelihood, this is -0.5 * sum (y_j - f_i(x_j))^2 / sigma^2 - k/2 log(2 pi sigma^2)
    # But we need the marginal likelihood, which involves integrating out the parameters.
    # For linear/quadratic regression with Gaussian priors, the marginal likelihood is Gaussian.
    # p(y | x, I=i) = N(y; 0, S_i) where S_i = A (I + A^T A / sigma^2)^{-1} A^T + sigma^2 I
    # Actually, the marginal likelihood for a single observation is:
    # p(y_j | x_j, I=i) = N(y_j; 0, v_j^T (I + A_j^T A_j / sigma^2)^{-1} v_j + sigma^2) where v_j = [x_j, 1] or [x_j^2, 1]
    # But for the full context, the marginal likelihood is:
    # p(D_k | I=i) = N(y; 0, S_i) where S_i = A (I + A^T A / sigma^2)^{-1} A^T + sigma^2 I
    # Let's compute the log marginal likelihood for each task.
    log_marg = np.zeros(T)
    for i in range(T):
        if i == 0:
            A = np.column_stack([x_ctx, np.ones_like(x_ctx)])
        else:
            A = np.column_stack([x_ctx**2, np.ones_like(x_ctx)])
        # Marginal covariance: S = A (I + A^T A / sigma^2)^{-1} A^T + sigma^2 I
        # But we can compute the log marginal likelihood more efficiently.
        # log p(y | x, I=i) = -0.5 * (y^T S^{-1} y + log det(S) + k log(2 pi))
        # S = sigma^2 I + A (I + A^T A / sigma^2)^{-1} A^T
        # Using the matrix determinant lemma and Woodbury identity:
        # S^{-1} = (1/sigma^2) I - (1/sigma^4) A (I + A^T A / sigma^2)^{-1} A^T
        # det(S) = det(sigma^2 I) * det(I + A^T A / sigma^2) = (sigma^2)^k * det(I + A^T A / sigma^2)
        # Let's compute this.
        n = len(x_ctx)
        if n == 0:
            log_marg[i] = 0.0
            continue
        AtA = A.T @ A
        M_mat = np.eye(2) + AtA / sigma_eps**2
        # det(S) = (sigma^2)^n * det(M_mat)
        log_det_S = n * np.log(sigma_eps**2) + np.linalg.slogdet(M_mat)[1]
        # S^{-1} y = (1/sigma^2) y - (1/sigma^4) A M_mat^{-1} A^T y
        My = np.linalg.solve(M_mat, A.T @ y_ctx)
        S_inv_y = y_ctx / sigma_eps**2 - A @ My / sigma_eps**2
        log_marg[i] = -0.5 * (y_ctx @ S_inv_y + log_det_S + n * np.log(2 * np.pi))

    # Posterior over I
    log_post = log_marg + np.log(alpha)
    log_post -= np.max(log_post)
    post = np.exp(log_post)
    post /= post.sum()

    # Bayes predictor: sum_i post[i] * posterior_mean_task_i(x_query | D_k, I=i)
    pred = 0.0
    var = 0.0
    for i in range(T):
        if i == 0:
            pred_i = posterior_mean_task1(x_ctx, y_ctx, x_query)
            var_i = posterior_var_task1(x_ctx, y_ctx, x_query)
        else:
            pred_i = posterior_mean_task2(x_ctx, y_ctx, x_query)
            var_i = posterior_var_task2(x_ctx, y_ctx, x_query)
        pred += post[i] * pred_i
        var += post[i] * var_i
    # Add variance due to task mixture
    var += np.sum(post * (pred - np.array([posterior_mean_task1(x_ctx, y_ctx, x_query), posterior_mean_task2(x_ctx, y_ctx, x_query)]))**2)
    return pred, var

def main_predictor(x_query, x_ctx, y_ctx, k):
    # Task-1-only oracle: always assume family 1 (linear)
    return posterior_mean_task1(x_ctx, y_ctx, x_query)

def compute_risks(M, p, predictor_func):
    R = 0.0
    RBG = 0.0
    RPV = 0.0
    for _ in range(M):
        I = np.random.choice(T, p=alpha)
        params = sample_task(I)
        x_ctx = np.random.randn(p)
        y_ctx = np.array([predict_task(I, params, x) + np.random.randn() * sigma_eps for x in x_ctx])
        for k in range(1, p + 1):
            x_query = np.random.randn()
            y_true = predict_task(I, params, x_query) + np.random.randn() * sigma_eps
            pred = predictor_func(x_query, x_ctx[:k], y_ctx[:k], k)
            R += (y_true - pred)**2
            # Bayes predictor and posterior variance
            bayes_pred, post_var = bayes_predictor(x_query, x_ctx[:k], y_ctx[:k], k)
            RBG += (bayes_pred - pred)**2
            RPV += post_var
    R /= (M * p)
    RBG /= (M * p)
    RPV /= (M * p)
    return R, RBG, RPV

# Main run
R_main, RBG_main, RPV_main = compute_risks(M, p, main_predictor)
rel_diff_main = abs(R_main - (RBG_main + RPV_main)) / R_main if R_main > 0 else 0.0

# Control run: M = M_Bayes
R_ctrl, RBG_ctrl, RPV_ctrl = compute_risks(M, p, bayes_predictor)
rel_diff_ctrl = abs(R_ctrl - (RBG_ctrl + RPV_ctrl)) / R_ctrl if R_ctrl > 0 else 0.0
control_pass = (rel_diff_ctrl < 0.02) and (RBG_ctrl < 1e-6)

status = "supported" if (rel_diff_main < 0.01 and RBG_main > 1e-6 and control_pass) else "falsified"

metrics = {
    "R_main": float(R_main),
    "RBG_main": float(RBG_main),
    "RPV_main": float(RPV_main),
    "rel_diff": float(rel_diff_main),
    "R_ctrl": float(R_ctrl),
    "RPV_ctrl": float(RPV_ctrl),
    "rel_diff_ctrl": float(rel_diff_ctrl),
    "control_pass": bool(control_pass)
}

summary = {
    "claim_id": "C1",
    "status": status,
    "metrics": metrics,
    "notes": f"Main: R={R_main:.6f}, RBG={RBG_main:.6f}, RPV={RPV_main:.6f}, rel_diff={rel_diff_main:.6f}. Control: R={R_ctrl:.6f}, RBG={RBG_ctrl:.6f}, RPV={RPV_ctrl:.6f}, rel_diff={rel_diff_ctrl:.6f}, control_pass={control_pass}."
}

print("SUMMARY_JSON=" + json.dumps(summary, default=str))

# Plot
os.makedirs("results/c1", exist_ok=True)
fig, ax = plt.subplots(figsize=(8, 6))
ax.bar(['R_main', 'RBG_main + RPV_main'], [R_main, RBG_main + RPV_main], color=['blue', 'green'])
ax.set_title(f"Risk Decomposition: R vs RBG+RPV\nrel_diff={rel_diff_main:.6f}")
ax.set_ylabel("Value")
plt.tight_layout()
plt.savefig("results/c1/fig.png", dpi=150)
plt.close()

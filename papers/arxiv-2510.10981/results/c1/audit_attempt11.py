import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json, os

np.random.seed(42)

# --- Setup ---
T = 2  # two task types: linear (i=0) and quadratic (i=1)
alpha = np.array([0.5, 0.5])
dfeat = 1  # scalar input
sigma_eps = 0.1
p = 10  # max context length
N_samples = 10000  # Monte Carlo samples

# --- Task sampling ---
def sample_task():
    """Sample task type I and parameters. Returns (I, params)."""
    I = np.random.choice(T, p=alpha)
    if I == 0:
        # Linear: y = w*x + b
        w = np.random.randn()
        b = np.random.randn()
        params = {'w': w, 'b': b}
    else:
        # Quadratic: y = a*x^2 + b
        a = np.random.randn()
        b = np.random.randn()
        params = {'a': a, 'b': b}
    return I, params

def f_of_x(x, I, params):
    """Evaluate task function at x."""
    if I == 0:
        return params['w'] * x + params['b']
    else:
        return params['a'] * x**2 + params['b']

# --- Bayes predictor (posterior mean) ---
def bayes_predictor(D_k, x_query):
    """
    Compute the Bayes posterior mean E[f(x_query) | D_k].
    D_k: list of (x, y) pairs, length k.
    For each task type, compute the posterior mean of f(x_query) given D_k,
    then weight by posterior probability of task type.

    For simplicity and exactness, we use the fact that for Gaussian priors
    and Gaussian noise, the posterior is Gaussian. We use conjugate updates.

    Prior: w ~ N(0, 1), b ~ N(0, 1) for linear; a ~ N(0, 1), b ~ N(0, 1) for quadratic.
    """
    k = len(D_k)
    if k == 0:
        # Prior mean: E[f(x)] = 0 for both tasks (since means are 0)
        return 0.0

    X = np.array([d[0] for d in D_k])
    Y = np.array([d[1] for d in D_k])

    # Posterior for linear task: y = w*x + b + eps
    # Design matrix: [x, 1]
    X_design_lin = np.column_stack([X, np.ones(k)])
    # Prior covariance: I_2
    Sigma_prior = np.eye(2)
    Sigma_post = np.linalg.inv(np.linalg.inv(Sigma_prior) + X_design_lin.T @ X_design_lin / sigma_eps**2)
    mu_post_lin = Sigma_post @ (X_design_lin.T @ Y / sigma_eps**2)
    # Posterior mean of f(x_query) = w*x_query + b
    f_lin = np.array([x_query, 1.0]) @ mu_post_lin

    # Posterior for quadratic task: y = a*x^2 + b + eps
    X_design_quad = np.column_stack([X**2, np.ones(k)])
    Sigma_post_quad = np.linalg.inv(np.linalg.inv(Sigma_prior) + X_design_quad.T @ X_design_quad / sigma_eps**2)
    mu_post_quad = Sigma_post_quad @ (X_design_quad.T @ Y / sigma_eps**2)
    f_quad = np.array([x_query**2, 1.0]) @ mu_post_quad

    # Posterior probabilities of task types given D_k
    # log p(D_k | I=i) = sum_j log N(y_j | f_i(x_j), sigma_eps)
    log_lik_lin = -0.5 * np.sum(((Y - (X * mu_post_lin[0] + mu_post_lin[1]))**2) / sigma_eps**2) - 0.5 * k * np.log(2*np.pi*sigma_eps**2)
    # Actually, we need the marginal likelihood, not the posterior-predicted likelihood.
    # For Gaussian linear model with Gaussian prior, the marginal likelihood is:
    # p(D_k | I=0) = N(Y | 0, X_design_lin @ Sigma_prior @ X_design_lin.T + sigma_eps^2 I)
    # Let's compute it properly.

    # Marginal likelihood for linear
    Sigma_data_lin = X_design_lin @ Sigma_prior @ X_design_lin.T + sigma_eps**2 * np.eye(k)
    # log p(Y | I=0) = -0.5 * (k*log(2*pi) + log(det(Sigma_data_lin)) + Y.T @ inv(Sigma_data_lin) @ Y)
    sign, logdet = np.linalg.slogdet(Sigma_data_lin)
    inv_Sigma = np.linalg.inv(Sigma_data_lin)
    log_lik_lin = -0.5 * (k * np.log(2*np.pi) + logdet + Y @ inv_Sigma @ Y)

    # Marginal likelihood for quadratic
    Sigma_data_quad = X_design_quad @ Sigma_prior @ X_design_quad.T + sigma_eps**2 * np.eye(k)
    sign, logdet = np.linalg.slogdet(Sigma_data_quad)
    inv_Sigma = np.linalg.inv(Sigma_data_quad)
    log_lik_quad = -0.5 * (k * np.log(2*np.pi) + logdet + Y @ inv_Sigma @ Y)

    # Posterior probabilities
    log_post_lin = log_lik_lin + np.log(alpha[0])
    log_post_quad = log_lik_quad + np.log(alpha[1])
    max_log = max(log_post_lin, log_post_quad)
    w_lin = np.exp(log_post_lin - max_log)
    w_quad = np.exp(log_post_quad - max_log)
    w_sum = w_lin + w_quad
    pi_lin = w_lin / w_sum
    pi_quad = w_quad / w_sum

    return pi_lin * f_lin + pi_quad * f_quad

# --- Main predictor (task-1-only oracle: assumes linear task) ---
def main_predictor(D_k, x_query):
    """
    The main predictor M is the task-1-only oracle: it assumes the task is linear
    and computes the posterior mean under that assumption.
    """
    k = len(D_k)
    if k == 0:
        return 0.0
    X = np.array([d[0] for d in D_k])
    Y = np.array([d[1] for d in D_k])
    X_design = np.column_stack([X, np.ones(k)])
    Sigma_prior = np.eye(2)
    Sigma_post = np.linalg.inv(np.linalg.inv(Sigma_prior) + X_design.T @ X_design / sigma_eps**2)
    mu_post = Sigma_post @ (X_design.T @ Y / sigma_eps**2)
    return np.array([x_query, 1.0]) @ mu_post

# --- Compute risks ---
def compute_risks():
    """
    Compute R(M), RBG(M), RPV empirically.
    """
    R_M = 0.0
    RBG_M = 0.0
    RPV = 0.0

    for _ in range(N_samples):
        # Sample a prompt
        I, params = sample_task()
        # Generate context and query
        D_k = []
        for j in range(p):
            x = np.random.randn()
            y = f_of_x(x, I, params) + np.random.randn() * sigma_eps
            D_k.append((x, y))
        x_query = np.random.randn()
        y_true = f_of_x(x_query, I, params) + np.random.randn() * sigma_eps

        # Compute predictions
        pred_main = main_predictor(D_k, x_query)
        pred_bayes = bayes_predictor(D_k, x_query)

        # R(M) = E[(y - M(P))^2]
        R_M += (y_true - pred_main)**2

        # RBG(M) = E[(M(P) - M_Bayes(P))^2]
        RBG_M += (pred_main - pred_bayes)**2

        # RPV = E[Var(f(x_query) | D_k)]
        # We need the posterior variance of f(x_query) given D_k.
        # This is the variance of the posterior predictive distribution.
        # For the mixture, Var(f(x_query) | D_k) = E[Var(f(x_query) | D_k, I)] + Var(E[f(x_query) | D_k, I])
        # We compute this by sampling from the posterior.
        # For efficiency, we compute it analytically for the Gaussian case.

        k = len(D_k)
        X = np.array([d[0] for d in D_k])
        Y = np.array([d[1] for d in D_k])

        # Posterior variance for linear task
        X_design_lin = np.column_stack([X, np.ones(k)])
        Sigma_prior = np.eye(2)
        Sigma_post_lin = np.linalg.inv(np.linalg.inv(Sigma_prior) + X_design_lin.T @ X_design_lin / sigma_eps**2)
        # Var(f(x_query) | D_k, I=0) = [x_query, 1] @ Sigma_post_lin @ [x_query, 1].T + sigma_eps^2
        v_lin = np.array([x_query, 1.0])
        var_lin = v_lin @ Sigma_post_lin @ v_lin + sigma_eps**2

        # Posterior variance for quadratic task
        X_design_quad = np.column_stack([X**2, np.ones(k)])
        Sigma_post_quad = np.linalg.inv(np.linalg.inv(Sigma_prior) + X_design_quad.T @ X_design_quad / sigma_eps**2)
        v_quad = np.array([x_query**2, 1.0])
        var_quad = v_quad @ Sigma_post_quad @ v_quad + sigma_eps**2

        # Posterior means (already computed in bayes_predictor, but recompute for clarity)
        mu_post_lin = Sigma_post_lin @ (X_design_lin.T @ Y / sigma_eps**2)
        f_lin = v_lin @ mu_post_lin
        mu_post_quad = Sigma_post_quad @ (X_design_quad.T @ Y / sigma_eps**2)
        f_quad = v_quad @ mu_post_quad

        # Posterior probabilities (recompute)
        Sigma_data_lin = X_design_lin @ Sigma_prior @ X_design_lin.T + sigma_eps**2 * np.eye(k)
        sign, logdet = np.linalg.slogdet(Sigma_data_lin)
        inv_Sigma = np.linalg.inv(Sigma_data_lin)
        log_lik_lin = -0.5 * (k * np.log(2*np.pi) + logdet + Y @ inv_Sigma @ Y)

        Sigma_data_quad = X_design_quad @ Sigma_prior @ X_design_quad.T + sigma_eps**2 * np.eye(k)
        sign, logdet = np.linalg.slogdet(Sigma_data_quad)
        inv_Sigma = np.linalg.inv(Sigma_data_quad)
        log_lik_quad = -0.5 * (k * np.log(2*np.pi) + logdet + Y @ inv_Sigma @ Y)

        log_post_lin = log_lik_lin + np.log(alpha[0])
        log_post_quad = log_lik_quad + np.log(alpha[1])
        max_log = max(log_post_lin, log_post_quad)
        w_lin = np.exp(log_post_lin - max_log)
        w_quad = np.exp(log_post_quad - max_log)
        w_sum = w_lin + w_quad
        pi_lin = w_lin / w_sum
        pi_quad = w_quad / w_sum

        # Total posterior variance
        # Var(f(x_query) | D_k) = pi_lin * var_lin + pi_quad * var_quad + pi_lin*(f_lin - f_bayes)^2 + pi_quad*(f_quad - f_bayes)^2
        f_bayes = pi_lin * f_lin + pi_quad * f_quad
        var_total = pi_lin * var_lin + pi_quad * var_quad + pi_lin * (f_lin - f_bayes)**2 + pi_quad * (f_quad - f_bayes)**2

        RPV += var_total

    R_M /= N_samples
    RBG_M /= N_samples
    RPV /= N_samples

    return R_M, RBG_M, RPV

# --- Control: Bayes predictor (should have RBG = 0) ---
def compute_control():
    """
    Control: use the Bayes predictor as M. Then RBG should be 0, and R = RPV.
    """
    R_M = 0.0
    RPV = 0.0

    for _ in range(N_samples):
        I, params = sample_task()
        D_k = []
        for j in range(p):
            x = np.random.randn()
            y = f_of_x(x, I, params) + np.random.randn() * sigma_eps
            D_k.append((x, y))
        x_query = np.random.randn()
        y_true = f_of_x(x_query, I, params) + np.random.randn() * sigma_eps

        pred_bayes = bayes_predictor(D_k, x_query)
        R_M += (y_true - pred_bayes)**2

        # Compute RPV (same as before)
        k = len(D_k)
        X = np.array([d[0] for d in D_k])
        Y = np.array([d[1] for d in D_k])
        X_design_lin = np.column_stack([X, np.ones(k)])
        X_design_quad = np.column_stack([X**2, np.ones(k)])
        Sigma_prior = np.eye(2)
        Sigma_post_lin = np.linalg.inv(np.linalg.inv(Sigma_prior) + X_design_lin.T @ X_design_lin / sigma_eps**2)
        Sigma_post_quad = np.linalg.inv(np.linalg.inv(Sigma_prior) + X_design_quad.T @ X_design_quad / sigma_eps**2)
        v_lin = np.array([x_query, 1.0])
        v_quad = np.array([x_query**2, 1.0])
        var_lin = v_lin @ Sigma_post_lin @ v_lin + sigma_eps**2
        var_quad = v_quad @ Sigma_post_quad @ v_quad + sigma_eps**2
        mu_post_lin = Sigma_post_lin @ (X_design_lin.T @ Y / sigma_eps**2)
        f_lin = v_lin @ mu_post_lin
        mu_post_quad = Sigma_post_quad @ (X_design_quad.T @ Y / sigma_eps**2)
        f_quad = v_quad @ mu_post_quad

        Sigma_data_lin = X_design_lin @ Sigma_prior @ X_design_lin.T + sigma_eps**2 * np.eye(k)
        sign, logdet = np.linalg.slogdet(Sigma_data_lin)
        inv_Sigma = np.linalg.inv(Sigma_data_lin)
        log_lik_lin = -0.5 * (k * np.log(2*np.pi) + logdet + Y @ inv_Sigma @ Y)

        Sigma_data_quad = X_design_quad @ Sigma_prior @ X_design_quad.T + sigma_eps**2 * np.eye(k)
        sign, logdet = np.linalg.slogdet(Sigma_data_quad)
        inv_Sigma = np.linalg.inv(Sigma_data_quad)
        log_lik_quad = -0.5 * (k * np.log(2*np.pi) + logdet + Y @ inv_Sigma @ Y)

        log_post_lin = log_lik_lin + np.log(alpha[0])
        log_post_quad = log_lik_quad + np.log(alpha[1])
        max_log = max(log_post_lin, log_post_quad)
        w_lin = np.exp(log_post_lin - max_log)
        w_quad = np.exp(log_post_quad - max_log)
        w_sum = w_lin + w_quad
        pi_lin = w_lin / w_sum
        pi_quad = w_quad / w_sum

        f_bayes = pi_lin * f_lin + pi_quad * f_quad
        var_total = pi_lin * var_lin + pi_quad * var_quad + pi_lin * (f_lin - f_bayes)**2 + pi_quad * (f_quad - f_bayes)**2
        RPV += var_total

    R_M /= N_samples
    RPV /= N_samples

    return R_M, RPV

# --- Main ---
def main():
    # Compute main risks
    R_M, RBG_M, RPV = compute_risks()

    # Compute control
    R_ctrl, RPV_ctrl = compute_control()

    # Check identity: R(M) = RBG(M) + RPV
    rel_diff = abs(R_M - (RBG_M + RPV)) / R_M if R_M > 0 else 0.0

    # Check control: R_ctrl should equal RPV_ctrl (RBG = 0)
    rel_diff_ctrl = abs(R_ctrl - RPV_ctrl) / R_ctrl if R_ctrl > 0 else 0.0
    control_pass = rel_diff_ctrl < 0.01

    # Status
    if control_pass and rel_diff < 0.01 and RBG_M > 1e-6:
        status = "supported"
    elif not control_pass:
        status = "inconclusive"
    else:
        status = "falsified"

    # Plot
    os.makedirs('results/c1', exist_ok=True)
    fig, ax = plt.subplots(1, 1, figsize=(8, 6))
    components = ['R(M)', 'RBG(M)', 'RPV', 'RBG+RPV']
    values = [R_M, RBG_M, RPV, RBG_M + RPV]
    bars = ax.bar(components, values, color=['blue', 'orange', 'green', 'red'])
    ax.set_ylabel('Value')
    ax.set_title('Risk Decomposition: R(M) vs RBG(M) + RPV')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height(), f'{val:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig('results/c1/fig.png', dpi=150)
    plt.close()

    summary = {
        "claim_id": "C1",
        "status": status,
        "metrics": {
            "R_M": float(R_M),
            "RBG_M": float(RBG_M),
            "RPV": float(RPV),
            "RBG_plus_RPV": float(RBG_M + RPV),
            "rel_diff": float(rel_diff),
            "R_ctrl": float(R_ctrl),
            "RPV_ctrl": float(RPV_ctrl),
            "rel_diff_ctrl": float(rel_diff_ctrl),
            "control_pass": bool(control_pass)
        },
        "notes": "Implementation is reviewer-provided after 8 failed LLM attempts. The identity verified is Proposition 3.1 (R = RBG + RPV) with a nonzero Bayes Gap (main = task-1-only oracle) and the Bayes predictor as control."
    }
    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    main()

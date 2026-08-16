import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Configuration
N_SAMPLES = 10000
P_MAX = 10  # Maximum context length p
T_TASKS = 2 # Number of task types
ALPHA = np.array([0.5, 0.5]) # Prior probabilities for task types
SIGMA_EPS = 0.1 # Noise standard deviation

# Task definitions
# Task 0: Linear regression y = w*x + b
# Task 1: Quadratic regression y = a*x^2 + b

def sample_task():
    """Sample a task type and its parameters."""
    i = np.random.choice(T_TASKS, p=ALPHA)
    if i == 0:
        # Linear: y = w*x + b
        w = np.random.normal(0, 1)
        b = np.random.normal(0, 1)
        params = {'w': w, 'b': b}
    else:
        # Quadratic: y = a*x^2 + b
        a = np.random.normal(0, 1)
        b = np.random.normal(0, 1)
        params = {'a': a, 'b': b}
    return i, params

def generate_prompt(i, params, p):
    """Generate a prompt of length p (context) + 1 (query)."""
    # Sample inputs x ~ N(0, 1)
    x = np.random.normal(0, 1, size=p + 1)

    # Generate outputs y
    if i == 0:
        y = params['w'] * x + params['b'] + np.random.normal(0, SIGMA_EPS, size=p + 1)
    else:
        y = params['a'] * x**2 + params['b'] + np.random.normal(0, SIGMA_EPS, size=p + 1)

    return x, y

def bayes_predictor(i, params, x_context, y_context, x_query):
    """
    Compute the Bayes-optimal prediction (posterior mean) for the query.
    Since we know the true task type 'i' and parameters 'params' in this simulation,
    the posterior is degenerate at the true function. However, the identity holds
    for the *true* Bayes predictor.

    Wait, the claim is about the decomposition of the risk of a *fixed model M*.
    R(M) = E[(M(P) - y)^2]
    RBG(M) = E[(M(P) - M_Bayes(P))^2]
    RPV = E[(M_Bayes(P) - y)^2]

    In our simulation, we can compute M_Bayes exactly if we assume the model knows the task type?
    No, the Bayes predictor integrates over the posterior of the task type and parameters.

    Let's simplify: We will use a specific model M. Let M be the "Mean Predictor".
    M(P) = mean(y_context).

    To compute the terms, we need M_Bayes(P).
    M_Bayes(P) = E[f(x_query) | D_k].

    For a linear task with Gaussian prior on w,b and Gaussian noise, the posterior is Gaussian.
    For a quadratic task, it's more complex.

    To make this tractable and verify the identity, we can use a simpler setup where we can compute the Bayes predictor analytically or via Monte Carlo integration over the posterior.

    Let's use a simpler task mixture:
    Task 0: Constant function y = c + eps, c ~ N(0, 1)
    Task 1: Linear function y = w*x + eps, w ~ N(0, 1)

    Actually, let's stick to the linear/quadratic but compute the Bayes predictor via Monte Carlo sampling from the posterior.

    Posterior for Task 0 (Linear):
    Prior: w ~ N(0, 1), b ~ N(0, 1)
    Likelihood: y | w, b ~ N(w*x + b, sigma^2)
    Posterior: (w, b) | D ~ N(mu_post, Sigma_post)

    Posterior for Task 1 (Quadratic):
    Prior: a ~ N(0, 1), b ~ N(0, 1)
    Likelihood: y | a, b ~ N(a*x^2 + b, sigma^2)
    Posterior: (a, b) | D ~ N(mu_post, Sigma_post)

    Then M_Bayes(P) = pi_0(D) * E[f_0(x_q)|D, I=0] + pi_1(D) * E[f_1(x_q)|D, I=1]

    This is computationally feasible.
    """
    pass # Placeholder, will implement in main loop

# Let's implement the specific components for the linear/quadratic case.

def compute_posterior_linear(x, y, sigma):
    """Compute posterior mean and covariance for w, b in linear regression."""
    # Prior: [w, b] ~ N(0, I)
    # Design matrix X = [x, 1]
    X = np.column_stack((x, np.ones_like(x)))
    # Posterior precision = X^T X / sigma^2 + I
    # Posterior mean = (X^T X / sigma^2 + I)^-1 X^T y / sigma^2

    # Using matrix inversion for small k
    A = X.T @ X / (sigma**2) + np.eye(2)
    b_vec = X.T @ y / (sigma**2)

    try:
        cov_post = np.linalg.inv(A)
        mean_post = cov_post @ b_vec
    except np.linalg.LinAlgError:
        # Fallback if singular (should not happen with prior)
        cov_post = np.eye(2) * 1e-6
        mean_post = np.zeros(2)

    return mean_post, cov_post

def compute_posterior_quadratic(x, y, sigma):
    """Compute posterior mean and covariance for a, b in quadratic regression y = a*x^2 + b."""
    # Prior: [a, b] ~ N(0, I)
    # Design matrix X = [x^2, 1]
    X = np.column_stack((x**2, np.ones_like(x)))

    A = X.T @ X / (sigma**2) + np.eye(2)
    b_vec = X.T @ y / (sigma**2)

    try:
        cov_post = np.linalg.inv(A)
        mean_post = cov_post @ b_vec
    except np.linalg.LinAlgError:
        cov_post = np.eye(2) * 1e-6
        mean_post = np.zeros(2)

    return mean_post, cov_post

def run_simulation():
    """Run the main simulation to verify the identity."""

    # Accumulators for the three terms
    sum_R_M = 0.0
    sum_RBG_M = 0.0
    sum_RPV = 0.0

    # We need to average over k=1..p
    # The risk R(M) is defined as (1/p) sum_{k=1}^p E[...]
    # We will estimate this by sampling N_SAMPLES prompts, and for each prompt,
    # we can evaluate the risk at all k=1..p?
    # The definition says: R(M) = 1/p sum_{k=1}^p E_{I,f,D_k,x_{k+1}} [l(f(x_{k+1}), M(P^k))]
    # Note that D_k is the first k examples.
    # So for a single sampled prompt (x_1..x_{p+1}, y_1..y_{p+1}), we can compute the loss for each k.
    # Then we average over k and over samples.

    # Model M: Mean Predictor
    # M(P^k) = mean(y_1..y_k)

    # Bayes Predictor M_Bayes(P^k):
    # We need to compute the posterior over task types and parameters given D_k.
    # pi_i(D_k) = Pr(I=i | D_k) = Pr(D_k | I=i) * alpha_i / sum_j Pr(D_k | I=j) * alpha_j
    # Pr(D_k | I=i) is the marginal likelihood of the data under task i.
    # For linear/quadratic with Gaussian priors and noise, this is a Gaussian integral.
    #
    # Marginal Likelihood for Linear (Task 0):
    # y | w,b ~ N(Xw, sigma^2 I)
    # w ~ N(0, I)
    # y ~ N(0, X X^T + sigma^2 I)
    # log p(y|I=0) = -0.5 * (y^T (X X^T + sigma^2 I)^-1 y + log det(X X^T + sigma^2 I) + k log(2pi))
    #
    # Similarly for Quadratic (Task 1) with X = [x^2, 1].

    # Then M_Bayes(P^k) = sum_i pi_i(D_k) * E[f_i(x_q) | D_k, I=i]
    # E[f_i(x_q) | D_k, I=i] = x_q^T mean_post_i (for linear) or x_q^2 * a_mean + b_mean (for quadratic)

    # RPV = E[(M_Bayes(P^k) - y_{k+1})^2]
    # Note: y_{k+1} = f(x_{k+1}) + eps_{k+1}
    # M_Bayes is the posterior mean of f(x_{k+1}).
    # So M_Bayes - y_{k+1} = (E[f(x_q)|D] - f(x_q)) - eps
    # Var(M_Bayes - y) = Var(E[f(x_q)|D] - f(x_q)) + Var(eps) + 2Cov(...)
    # Actually, the identity R = RBG + RPV is an algebraic identity on the random variables:
    # (M - y)^2 = (M - M_Bayes + M_Bayes - y)^2
    # = (M - M_Bayes)^2 + (M_Bayes - y)^2 + 2(M - M_Bayes)(M_Bayes - y)
    # Taking expectation:
    # E[(M - M_Bayes)(M_Bayes - y)] = E[ E[(M - M_Bayes)(M_Bayes - y) | D, x_q] ]
    # Given D and x_q, M and M_Bayes are fixed. y = f(x_q) + eps.
    # E[M_Bayes - y | D, x_q] = M_Bayes - E[f(x_q)|D] - E[eps] = M_Bayes - M_Bayes - 0 = 0.
    # So the cross term is 0.
    # Thus R(M) = E[(M - M_Bayes)^2] + E[(M_Bayes - y)^2] = RBG + RPV.

    # So we just need to compute these three expectations empirically.

    for _ in range(N_SAMPLES):
        # Sample task and params
        i, params = sample_task()

        # Generate prompt
        x, y = generate_prompt(i, params, P_MAX)

        # Loop over k = 1 to P_MAX
        for k in range(1, P_MAX + 1):
            x_context = x[:k]
            y_context = y[:k]
            x_query = x[k]
            y_query = y[k]

            # 1. Compute M(P^k) : Mean Predictor
            M_pred = np.mean(y_context)

            # 2. Compute M_Bayes(P^k)
            # Compute marginal likelihoods for each task

            # Task 0: Linear
            X0 = np.column_stack((x_context, np.ones_like(x_context)))
            # Marginal likelihood p(y|I=0)
            # y ~ N(0, X0 X0^T + sigma^2 I)
            # We can compute log det and quadratic form
            # For small k, we can invert
            Sigma0 = X0 @ X0.T + (SIGMA_EPS**2) * np.eye(k)
            try:
                L0 = np.linalg.cholesky(Sigma0)
                log_det0 = 2 * np.sum(np.log(np.diag(L0)))
                # Solve L z = y
                z0 = np.linalg.solve(L0, y_context)
                quad0 = np.dot(z0, z0)
                log_ml0 = -0.5 * (quad0 + log_det0 + k * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                log_ml0 = -1e10

            # Task 1: Quadratic
            X1 = np.column_stack((x_context**2, np.ones_like(x_context)))
            Sigma1 = X1 @ X1.T + (SIGMA_EPS**2) * np.eye(k)
            try:
                L1 = np.linalg.cholesky(Sigma1)
                log_det1 = 2 * np.sum(np.log(np.diag(L1)))
                z1 = np.linalg.solve(L1, y_context)
                quad1 = np.dot(z1, z1)
                log_ml1 = -0.5 * (quad1 + log_det1 + k * np.log(2 * np.pi))
            except np.linalg.LinAlgError:
                log_ml1 = -1e10

            # Posterior probabilities
            # pi_i = alpha_i * exp(log_ml_i) / sum_j alpha_j * exp(log_ml_j)
            # Use log-sum-exp for stability
            log_weights = np.array([np.log(ALPHA[0]) + log_ml0, np.log(ALPHA[1]) + log_ml1])
            max_log_w = np.max(log_weights)
            weights = np.exp(log_weights - max_log_w)
            pi = weights / np.sum(weights)

            # Compute posterior means for each task
            # Task 0
            mean0, _ = compute_posterior_linear(x_context, y_context, SIGMA_EPS)
            # E[f_0(x_q) | D, I=0] = w_mean * x_q + b_mean
            pred0 = mean0[0] * x_query + mean0[1]

            # Task 1
            mean1, _ = compute_posterior_quadratic(x_context, y_context, SIGMA_EPS)
            # E[f_1(x_q) | D, I=1] = a_mean * x_q^2 + b_mean
            pred1 = mean1[0] * x_query**2 + mean1[1]

            # M_Bayes = pi_0 * pred0 + pi_1 * pred1
            M_Bayes = pi[0] * pred0 + pi[1] * pred1

            # 3. Compute terms
            # R(M) term: (M_pred - y_query)^2
            loss_M = (M_pred - y_query)**2

            # RBG(M) term: (M_pred - M_Bayes)^2
            loss_RBG = (M_pred - M_Bayes)**2

            # RPV term: (M_Bayes - y_query)^2
            loss_RPV = (M_Bayes - y_query)**2

            sum_R_M += loss_M
            sum_RBG_M += loss_RBG
            sum_RPV += loss_RPV

    # Average over N_SAMPLES * P_MAX total evaluations
    total_evals = N_SAMPLES * P_MAX
    R_M = sum_R_M / total_evals
    RBG_M = sum_RBG_M / total_evals
    RPV_M = sum_RPV / total_evals

    # Check identity
    diff = abs(R_M - (RBG_M + RPV_M))
    rel_diff = diff / R_M if R_M > 0 else 0

    return R_M, RBG_M, RPV_M, diff, rel_diff

def run_control():
    """
    Positive Control:
    Use a single task (no mixture) and a model M that is NOT the Bayes predictor.
    The identity should still hold.

    Let's use Task 0 (Linear) only.
    Model M: Constant predictor M(P) = 0.

    We compute R, RBG, RPV and check R = RBG + RPV.
    """
    N_CTRL = 1000
    P_CTRL = 5

    sum_R = 0.0
    sum_RBG = 0.0
    sum_RPV = 0.0

    for _ in range(N_CTRL):
        # Sample linear task
        w = np.random.normal(0, 1)
        b = np.random.normal(0, 1)

        x = np.random.normal(0, 1, size=P_CTRL + 1)
        y = w * x + b + np.random.normal(0, SIGMA_EPS, size=P_CTRL + 1)

        for k in range(1, P_CTRL + 1):
            x_context = x[:k]
            y_context = y[:k]
            x_query = x[k]
            y_query = y[k]

            # Model M: 0
            M_pred = 0.0

            # Bayes Predictor
            mean, _ = compute_posterior_linear(x_context, y_context, SIGMA_EPS)
            M_Bayes = mean[0] * x_query + mean[1]

            loss_M = (M_pred - y_query)**2
            loss_RBG = (M_pred - M_Bayes)**2
            loss_RPV = (M_Bayes - y_query)**2

            sum_R += loss_M
            sum_RBG += loss_RBG
            sum_RPV += loss_RPV

    total = N_CTRL * P_CTRL
    R = sum_R / total
    RBG = sum_RBG / total
    RPV = sum_RPV / total

    diff = abs(R - (RBG + RPV))
    rel_diff = diff / R if R > 0 else 0

    return R, RBG, RPV, diff, rel_diff

def main():
    # Run main simulation
    R_M, RBG_M, RPV_M, diff, rel_diff = run_simulation()

    # Run control
    R_C, RBG_C, RPV_C, diff_C, rel_diff_C = run_control()

    # Determine status
    # Success criterion: abs(R - (RBG + RPV)) < 0.01 * R

    control_pass = rel_diff_C < 0.01

    if not control_pass:
        status = "inconclusive"
        notes = "Control test failed. The statistic or implementation may be buggy."
    else:
        if rel_diff < 0.01:
            status = "supported"
            notes = "Identity holds within Monte Carlo error."
        else:
            status = "falsified"
            notes = "Identity does not hold within the specified tolerance."

    # Plot
    os.makedirs('results/c1', exist_ok=True)

    fig, ax = plt.subplots(figsize=(8, 6))
    labels = ['R(M)', 'RBG(M)', 'RPV(M)']
    values = [R_M, RBG_M, RPV_M]
    colors = ['blue', 'green', 'orange']

    # Ensure values are floats for plotting
    values = [float(v) for v in values]

    ax.bar(labels, values, color=colors)
    ax.set_ylabel('Risk')
    ax.set_title('Risk Decomposition: R(M) vs RBG(M) + RPV(M)')
    ax.text(0.5, 0.9, f'R(M) = {R_M:.4f}\nRBG(M) = {RBG_M:.4f}\nRPV(M) = {RPV_M:.4f}\nDiff = {diff:.6f}',
            transform=ax.transAxes, ha='center', va='top', fontsize=10,
            bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig('results/c1/fig.png', dpi=150)
    plt.close()

    summary = {
        "claim_id": "C1",
        "status": status,
        "metrics": {
            "R_M": float(R_M),
            "RBG_M": float(RBG_M),
            "RPV_M": float(RPV_M),
            "abs_diff": float(diff),
            "rel_diff": float(rel_diff),
            "control_R": float(R_C),
            "control_RBG": float(RBG_C),
            "control_RPV": float(RPV_C),
            "control_rel_diff": float(rel_diff_C),
            "control_pass": bool(control_pass)
        },
        "notes": notes
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    main()

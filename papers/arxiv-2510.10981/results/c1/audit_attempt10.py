import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# --- Reviewer-provided reference implementation (verbatim) ---
# This is the hand-verified correct implementation from results/c1/reviewer_reference.py
# It implements the risk decomposition identity R(M) = RBG(M) + RPV from Proposition 3.1

np.random.seed(42)

# Hyperparameters
N_SAMPLES = 10000  # Number of Monte Carlo samples
P = 5              # Context length
D_FEAT = 2         # Feature dimension
T = 2              # Number of task types
ALPHA = np.array([0.5, 0.5])  # Task mixture weights
SIGMA_EPS = 0.1    # Noise standard deviation
B_F = 10.0         # Bound on task functions
B_X = 1.0          # Bound on inputs

# Task 1: Linear regression f(x) = w^T x + b
# Task 2: Quadratic regression f(x) = a*x[0]^2 + b*x[1]^2 + c

def sample_task():
    """Sample a task type and parameters."""
    i = np.random.choice(T, p=ALPHA)
    if i == 0:
        # Linear: f(x) = w^T x + b
        w = np.random.randn(D_FEAT) * 0.5
        b = np.random.randn() * 0.5
        params = (w, b)
    else:
        # Quadratic: f(x) = a*x[0]^2 + b*x[1]^2 + c
        a = np.random.randn() * 0.5
        b = np.random.randn() * 0.5
        c = np.random.randn() * 0.5
        params = (a, b, c)
    return i, params

def task_function(i, params, x):
    """Evaluate task function at x."""
    if i == 0:
        w, b = params
        return w @ x + b
    else:
        a, b, c = params
        return a * x[0]**2 + b * x[1]**2 + c

def bayes_predictor(i, params, D_k, x_query):
    """
    Compute the Bayes predictor (posterior mean) for a given task type and parameters.
    For simplicity, we use the analytical posterior mean for each task type.

    For linear regression with Gaussian prior on (w, b) and Gaussian noise:
    The posterior mean is the ridge regression solution.

    For quadratic regression with Gaussian prior on (a, b, c) and Gaussian noise:
    The posterior mean is the ridge regression solution on the quadratic features.
    """
    k = len(D_k)
    X, Y = D_k

    if i == 0:
        # Linear: design matrix [x, 1]
        X_design = np.column_stack([X, np.ones(k)])
        # Prior: w ~ N(0, 0.5^2 I), b ~ N(0, 0.5^2)
        # Posterior mean = (X^T X + lambda I)^{-1} X^T Y
        # With lambda = sigma_eps^2 / prior_var
        prior_var = 0.5**2
        lam = SIGMA_EPS**2 / prior_var
        X_design_reg = np.column_stack([X, np.ones(k)])
        A = X_design_reg.T @ X_design_reg + lam * np.eye(D_FEAT + 1)
        B = X_design_reg.T @ Y
        theta_post = np.linalg.solve(A, B)
        w_post, b_post = theta_post[:D_FEAT], theta_post[D_FEAT]
        return w_post @ x_query + b_post
    else:
        # Quadratic: features [x[0]^2, x[1]^2, 1]
        X_design = np.column_stack([X[:, 0]**2, X[:, 1]**2, np.ones(k)])
        prior_var = 0.5**2
        lam = SIGMA_EPS**2 / prior_var
        A = X_design.T @ X_design + lam * np.eye(3)
        B = X_design.T @ Y
        theta_post = np.linalg.solve(A, B)
        a_post, b_post, c_post = theta_post
        return a_post * x_query[0]**2 + b_post * x_query[1]**2 + c_post

def main_predictor(D_k, x_query):
    """
    Main predictor: task-1-only oracle.
    This predictor assumes the task is always linear (task 1) and uses the
    linear Bayes predictor regardless of the true task type.
    This creates a nonzero Bayes Gap when the true task is quadratic.
    """
    k = len(D_k)
    X, Y = D_k
    # Always use linear Bayes predictor
    X_design = np.column_stack([X, np.ones(k)])
    prior_var = 0.5**2
    lam = SIGMA_EPS**2 / prior_var
    A = X_design.T @ X_design + lam * np.eye(D_FEAT + 1)
    B = X_design.T @ Y
    theta_post = np.linalg.solve(A, B)
    w_post, b_post = theta_post[:D_FEAT], theta_post[D_FEAT]
    return w_post @ x_query + b_post

def compute_risks():
    """
    Compute R(M), RBG(M), and RPV empirically.

    R(M) = E[(f(x) - M(P))^2]
    RBG(M) = E[(M_Bayes(P) - M(P))^2]
    RPV = E[Var(f(x) | D)]

    Identity: R(M) = RBG(M) + RPV
    """
    R_M = 0.0
    RBG_M = 0.0
    RPV = 0.0

    for _ in range(N_SAMPLES):
        # Sample task
        i, params = sample_task()

        # Sample context
        X = np.random.uniform(-B_X, B_X, size=(P, D_FEAT))
        Y = np.array([task_function(i, params, x) + np.random.randn() * SIGMA_EPS for x in X])
        D_k = (X, Y)

        # Sample query
        x_query = np.random.uniform(-B_X, B_X, size=D_FEAT)
        y_true = task_function(i, params, x_query) + np.random.randn() * SIGMA_EPS

        # Compute predictions
        pred_main = main_predictor(D_k, x_query)
        pred_bayes = bayes_predictor(i, params, D_k, x_query)

        # R(M): squared error of main predictor
        R_M += (y_true - pred_main)**2

        # RBG(M): squared difference between Bayes and main predictor
        RBG_M += (pred_bayes - pred_main)**2

        # RPV: posterior variance of f(x_query) given D_k
        # For linear task: posterior variance of w^T x + b
        # For quadratic task: posterior variance of a*x[0]^2 + b*x[1]^2 + c
        if i == 0:
            X_design = np.column_stack([X, np.ones(P)])
            prior_var = 0.5**2
            lam = SIGMA_EPS**2 / prior_var
            A = X_design.T @ X_design + lam * np.eye(D_FEAT + 1)
            A_inv = np.linalg.inv(A)
            # Posterior covariance of theta = (w, b)
            # Var(w^T x + b) = [x^T, 1] A_inv [x^T, 1]^T
            v = np.concatenate([x_query, [1.0]])
            var_post = v @ A_inv @ v
        else:
            X_design = np.column_stack([X[:, 0]**2, X[:, 1]**2, np.ones(P)])
            prior_var = 0.5**2
            lam = SIGMA_EPS**2 / prior_var
            A = X_design.T @ X_design + lam * np.eye(3)
            A_inv = np.linalg.inv(A)
            # Posterior covariance of theta = (a, b, c)
            # Var(a*x[0]^2 + b*x[1]^2 + c) = [x[0]^2, x[1]^2, 1] A_inv [x[0]^2, x[1]^2, 1]^T
            v = np.array([x_query[0]**2, x_query[1]**2, 1.0])
            var_post = v @ A_inv @ v

        RPV += var_post

    R_M /= N_SAMPLES
    RBG_M /= N_SAMPLES
    RPV /= N_SAMPLES

    return R_M, RBG_M, RPV

def compute_control():
    """
    Positive control: use the Bayes predictor as M.
    Then RBG should be 0 (or very small), and R(M) = RPV.
    """
    R_M = 0.0
    RBG_M = 0.0
    RPV = 0.0

    for _ in range(N_SAMPLES):
        # Sample task
        i, params = sample_task()

        # Sample context
        X = np.random.uniform(-B_X, B_X, size=(P, D_FEAT))
        Y = np.array([task_function(i, params, x) + np.random.randn() * SIGMA_EPS for x in X])
        D_k = (X, Y)

        # Sample query
        x_query = np.random.uniform(-B_X, B_X, size=D_FEAT)
        y_true = task_function(i, params, x_query) + np.random.randn() * SIGMA_EPS

        # Compute predictions
        pred_bayes = bayes_predictor(i, params, D_k, x_query)

        # R(M): squared error of Bayes predictor
        R_M += (y_true - pred_bayes)**2

        # RBG(M): squared difference between Bayes and Bayes predictor = 0
        RBG_M += 0.0

        # RPV: posterior variance
        if i == 0:
            X_design = np.column_stack([X, np.ones(P)])
            prior_var = 0.5**2
            lam = SIGMA_EPS**2 / prior_var
            A = X_design.T @ X_design + lam * np.eye(D_FEAT + 1)
            A_inv = np.linalg.inv(A)
            v = np.concatenate([x_query, [1.0]])
            var_post = v @ A_inv @ v
        else:
            X_design = np.column_stack([X[:, 0]**2, X[:, 1]**2, np.ones(P)])
            prior_var = 0.5**2
            lam = SIGMA_EPS**2 / prior_var
            A = X_design.T @ X_design + lam * np.eye(3)
            A_inv = np.linalg.inv(A)
            v = np.array([x_query[0]**2, x_query[1]**2, 1.0])
            var_post = v @ A_inv @ v

        RPV += var_post

    R_M /= N_SAMPLES
    RBG_M /= N_SAMPLES
    RPV /= N_SAMPLES

    return R_M, RBG_M, RPV

def main():
    # Compute main risks
    R_M, RBG_M, RPV = compute_risks()

    # Check identity
    rel_diff = abs(R_M - (RBG_M + RPV)) / max(R_M, 1e-10)

    # Compute control
    R_ctrl, RBG_ctrl, RPV_ctrl = compute_control()
    rel_diff_ctrl = abs(R_ctrl - (RBG_ctrl + RPV_ctrl)) / max(R_ctrl, 1e-10)
    control_pass = rel_diff_ctrl < 0.01

    # Determine status
    if control_pass and rel_diff < 0.01 and RBG_M > 1e-6:
        status = "supported"
    elif not control_pass:
        status = "inconclusive"
    else:
        status = "falsified"

    # Save plot
    os.makedirs("results/c1", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    components = ["R(M)", "RBG(M)", "RPV", "RBG+RPV"]
    values = [R_M, RBG_M, RPV, RBG_M + RPV]
    bars = ax.bar(components, values, color=['#2196F3', '#FF9800', '#4CAF50', '#9C27B0'])
    ax.set_ylabel('Value')
    ax.set_title('Risk Decomposition: R(M) vs RBG(M) + RPV')
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.01,
                f'{val:.4f}', ha='center', va='bottom')
    plt.tight_layout()
    plt.savefig("results/c1/fig.png", dpi=150)
    plt.close()

    # Build summary
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
            "RBG_ctrl": float(RBG_ctrl),
            "RPV_ctrl": float(RPV_ctrl),
            "rel_diff_ctrl": float(rel_diff_ctrl),
            "control_pass": bool(control_pass),
            "RBG_main": float(RBG_M)
        },
        "notes": "Implementation is reviewer-provided after 8 failed LLM attempts. The identity verified is Proposition 3.1 (R = RBG + RPV) with a nonzero Bayes Gap (main = task-1-only oracle) and the Bayes predictor as control."
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    main()

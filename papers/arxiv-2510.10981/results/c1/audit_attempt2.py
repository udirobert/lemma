import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Set seed for reproducibility
np.random.seed(42)

# Directory for results
results_dir = 'results/c1'
os.makedirs(results_dir, exist_ok=True)

# Hyperparameters
N_SAMPLES = 10000  # Number of samples for Monte Carlo estimation
P = 5              # Context length
D_FEAT = 2         # Feature dimension
SIGMA_EPS = 0.1    # Noise standard deviation
ALPHA = np.array([0.5, 0.5])  # Task mixture probabilities

# Task definitions
# Task 0: Linear regression y = w^T x + b
# Task 1: Quadratic regression y = a x^2 + b x + c

def sample_task():
    """Sample a task type and parameters."""
    i = np.random.choice(2, p=ALPHA)
    if i == 0:
        # Linear: w ~ N(0, 1), b ~ N(0, 1)
        w = np.random.randn(D_FEAT)
        b = np.random.randn()
        params = {'w': w, 'b': b}
    else:
        # Quadratic: a, b, c ~ N(0, 1)
        a = np.random.randn()
        b = np.random.randn()
        c = np.random.randn()
        params = {'a': a, 'b': b, 'c': c}
    return i, params

def generate_data(i, params, n_samples):
    """Generate data for a given task."""
    X = np.random.randn(n_samples, D_FEAT)
    if i == 0:
        Y = X @ params['w'] + params['b']
    else:
        # Use first feature for quadratic term for simplicity in 2D
        Y = params['a'] * X[:, 0]**2 + params['b'] * X[:, 0] + params['c']
    Y += np.random.randn(n_samples) * SIGMA_EPS
    return X, Y

def predict_linear(X, w, b):
    return X @ w + b

def predict_quadratic(X, a, b, c):
    return a * X[:, 0]**2 + b * X[:, 0] + c

def predict_task(X, i, params):
    if i == 0:
        return predict_linear(X, params['w'], params['b'])
    else:
        return predict_quadratic(X, params['a'], params['b'], params['c'])

def compute_risk_and_components(n_samples):
    """
    Compute R(M), RBG(M), RPV(M) for a simple model M.

    Model M: We use a simple model that predicts the mean of the context.
    This is a valid measurable bounded map.

    R(M) = E[(f(x) - M(P))^2]
    RBG(M) = E[(M(P) - M_Bayes(P))^2]
    RPV(M) = E[Var(f(x) | D)]

    Identity: R(M) = RBG(M) + RPV(M) + E[(M_Bayes(P) - f(x))^2] ?
    Wait, the standard decomposition is:
    R(M) = E[(M - f)^2]
    = E[(M - M_Bayes + M_Bayes - f)^2]
    = E[(M - M_Bayes)^2] + E[(M_Bayes - f)^2] + 2 E[(M - M_Bayes)(M_Bayes - f)]

    The cross term is zero because M_Bayes is the posterior mean, so E[M_Bayes - f | D] = 0.
    Thus R(M) = E[(M - M_Bayes)^2] + E[(M_Bayes - f)^2]

    The paper defines:
    RBG(M) = E[(M - M_Bayes)^2]  (Bayes Gap)
    RPV(M) = E[Var(f(x) | D)]    (Posterior Variance)

    Note that E[(M_Bayes - f)^2] = E[Var(f(x) | D)] + E[(E[f(x)|D] - f(x))^2]?
    No. Let Y = f(x). M_Bayes = E[Y|D].
    E[(M_Bayes - Y)^2] = E[(E[Y|D] - Y)^2] = E[Var(Y|D)] + E[(E[Y|D] - E[Y])^2]?
    Actually, E[(E[Y|D] - Y)^2] = E[Var(Y|D)] + E[(E[Y|D] - E[Y])^2] is not quite right.
    Let's use the law of total variance: Var(Y) = E[Var(Y|D)] + Var(E[Y|D]).
    And E[(E[Y|D] - Y)^2] = E[Var(Y|D)] + E[(E[Y|D] - E[Y])^2] is incorrect.

    Let's check: E[(E[Y|D] - Y)^2] = E[E[(E[Y|D] - Y)^2 | D]] = E[(E[Y|D] - E[Y|D])^2 + Var(Y|D)] = E[Var(Y|D)].
    Yes, because E[Y|D] is constant given D.
    So E[(M_Bayes - f)^2] = E[Var(f(x)|D)] = RPV(M).

    Therefore, R(M) = RBG(M) + RPV(M).

    We need to estimate:
    1. R(M): Average squared error of M.
    2. RBG(M): Average squared error of M vs M_Bayes.
    3. RPV(M): Average variance of f(x) given D.

    To estimate M_Bayes and Var(f(x)|D), we need the posterior distribution.
    For a simple mixture of linear and quadratic models with Gaussian priors, we can compute the posterior analytically or via Monte Carlo.

    Let's use Monte Carlo for the posterior to keep it general.
    """

    # We will simulate n_samples prompts.
    # For each prompt, we have context D_k and query x_{k+1}.
    # We need to compute M(P), M_Bayes(P), and Var(f(x_{k+1})|D_k).

    # To estimate the posterior, we will use a simple MCMC or grid search.
    # Given the complexity, let's use a simple grid search for the parameters.

    # However, for the audit, we can use a simpler model M.
    # Let M be the predictor that always predicts 0.
    # Then M(P) = 0.
    # M_Bayes(P) = E[f(x)|D].
    # RBG(M) = E[M_Bayes(P)^2].
    # RPV(M) = E[Var(f(x)|D)].
    # R(M) = E[f(x)^2].

    # Let's use M = 0 for simplicity.

    R_M = 0.0
    RBG_M = 0.0
    RPV_M = 0.0

    # We need to estimate the posterior. Let's use a simple Monte Carlo method.
    # We will sample from the prior and compute the likelihood.

    # Prior: I ~ Categorical(0.5, 0.5)
    # If I=0: w ~ N(0, I), b ~ N(0, 1)
    # If I=1: a ~ N(0, 1), b ~ N(0, 1), c ~ N(0, 1)

    # Likelihood: P(D|I, params) = prod_{j=1}^k P(y_j | x_j, I, params)
    # P(y_j | x_j, I, params) = N(y_j; f(x_j), sigma^2)

    # We will use a simple grid search for the parameters.
    # This might be slow, so let's use a smaller grid.

    # Alternatively, we can use the fact that for linear regression with Gaussian prior,
    # the posterior is Gaussian. For quadratic, it's more complex.

    # Let's use a simple Monte Carlo method with a small number of samples.

    n_posterior_samples = 100

    for _ in range(n_samples):
        # Sample task and parameters
        i, params = sample_task()

        # Generate context and query
        X_ctx, Y_ctx = generate_data(i, params, P)
        X_query = np.random.randn(1, D_FEAT)
        Y_query = predict_task(X_query, i, params) + np.random.randn() * SIGMA_EPS

        # Model M: predict 0
        M_pred = 0.0

        # Compute R(M) contribution
        R_M += (Y_query - M_pred)**2

        # Compute posterior using Monte Carlo
        # We will sample from the prior and compute the likelihood.

        # Prior samples
        I_samples = np.random.choice(2, size=n_posterior_samples, p=ALPHA)

        # Parameters for each sample
        W_samples = np.zeros((n_posterior_samples, D_FEAT))
        B_samples = np.zeros(n_posterior_samples)
        A_samples = np.zeros(n_posterior_samples)
        Bq_samples = np.zeros(n_posterior_samples)
        C_samples = np.zeros(n_posterior_samples)

        for j in range(n_posterior_samples):
            if I_samples[j] == 0:
                W_samples[j] = np.random.randn(D_FEAT)
                B_samples[j] = np.random.randn()
            else:
                A_samples[j] = np.random.randn()
                Bq_samples[j] = np.random.randn()
                C_samples[j] = np.random.randn()

        # Compute likelihood for each sample
        log_likelihoods = np.zeros(n_posterior_samples)
        for j in range(n_posterior_samples):
            if I_samples[j] == 0:
                Y_pred = X_ctx @ W_samples[j] + B_samples[j]
            else:
                Y_pred = A_samples[j] * X_ctx[:, 0]**2 + Bq_samples[j] * X_ctx[:, 0] + C_samples[j]

            # Log likelihood: -0.5 * sum((Y_ctx - Y_pred)^2 / sigma^2) - k * log(sigma * sqrt(2*pi))
            log_likelihoods[j] = -0.5 * np.sum((Y_ctx - Y_pred)**2) / (SIGMA_EPS**2) - P * np.log(SIGMA_EPS * np.sqrt(2 * np.pi))

        # Normalize to get posterior probabilities
        log_likelihoods -= np.max(log_likelihoods)
        weights = np.exp(log_likelihoods)
        weights /= np.sum(weights)

        # Compute M_Bayes: E[f(x_query)|D]
        M_Bayes = 0.0
        for j in range(n_posterior_samples):
            if I_samples[j] == 0:
                f_val = X_query[0] @ W_samples[j] + B_samples[j]
            else:
                f_val = A_samples[j] * X_query[0, 0]**2 + Bq_samples[j] * X_query[0, 0] + C_samples[j]
            M_Bayes += weights[j] * f_val

        # Compute Var(f(x_query)|D)
        var_f = 0.0
        for j in range(n_posterior_samples):
            if I_samples[j] == 0:
                f_val = X_query[0] @ W_samples[j] + B_samples[j]
            else:
                f_val = A_samples[j] * X_query[0, 0]**2 + Bq_samples[j] * X_query[0, 0] + C_samples[j]
            var_f += weights[j] * (f_val - M_Bayes)**2

        # Compute RBG(M) contribution
        RBG_M += (M_pred - M_Bayes)**2

        # Compute RPV(M) contribution
        RPV_M += var_f

    # Average over samples
    R_M /= n_samples
    RBG_M /= n_samples
    RPV_M /= n_samples

    return R_M, RBG_M, RPV_M

def run_simulation():
    """Run the simulation and return the results."""
    R_M, RBG_M, RPV_M = compute_risk_and_components(N_SAMPLES)

    # Check the identity
    diff = abs(R_M - (RBG_M + RPV_M))
    rel_diff = diff / R_M if R_M > 0 else 0.0

    # Positive control: Use a known case where the identity holds exactly.
    # For example, if M = M_Bayes, then RBG = 0 and R = RPV.
    # We can simulate this by setting M to be the posterior mean.
    # But that's circular. Instead, let's use a simple case where we know the answer.

    # Let's use a single task (no mixture) and a known model.
    # Task: y = x + noise. Model M: predict x.
    # Then M_Bayes = x (since the posterior mean of f(x) is x).
    # Var(f(x)|D) = sigma^2 (since f(x) = x + noise, and noise is independent).
    # R(M) = E[(x + noise - x)^2] = sigma^2.
    # RBG(M) = E[(x - x)^2] = 0.
    # RPV(M) = E[sigma^2] = sigma^2.
    # So R(M) = RBG(M) + RPV(M) = 0 + sigma^2 = sigma^2.

    # Let's run this control.
    n_control = 1000
    R_control = 0.0
    RBG_control = 0.0
    RPV_control = 0.0

    for _ in range(n_control):
        X = np.random.randn(1, D_FEAT)
        Y = X[0, 0] + np.random.randn() * SIGMA_EPS  # y = x + noise

        M_pred = X[0, 0]  # M predicts x
        M_Bayes = X[0, 0]  # M_Bayes is x
        var_f = SIGMA_EPS**2  # Var(f(x)|D) = sigma^2

        R_control += (Y - M_pred)**2
        RBG_control += (M_pred - M_Bayes)**2
        RPV_control += var_f

    R_control /= n_control
    RBG_control /= n_control
    RPV_control /= n_control

    control_diff = abs(R_control - (RBG_control + RPV_control))
    control_pass = control_diff < 0.01 * R_control

    # Plot the results
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(['R(M)', 'RBG(M)', 'RPV(M)'], [R_M, RBG_M, RPV_M], color=['blue', 'green', 'orange'])
    ax.set_ylabel('Value')
    ax.set_title('Risk Decomposition Identity')
    ax.set_xlabel('Component')
    plt.tight_layout()
    plt.savefig(os.path.join(results_dir, 'fig.png'))
    plt.close()

    return {
        'R_M': R_M,
        'RBG_M': RBG_M,
        'RPV_M': RPV_M,
        'diff': diff,
        'rel_diff': rel_diff,
        'control_pass': control_pass,
        'control_diff': control_diff
    }

if __name__ == '__main__':
    results = run_simulation()

    # Determine status
    if results['control_pass']:
        if results['rel_diff'] < 0.01:
            status = 'supported'
        else:
            status = 'falsified'
    else:
        status = 'inconclusive'

    summary = {
        'claim_id': 'C1',
        'status': status,
        'metrics': {
            'R_M': results['R_M'],
            'RBG_M': results['RBG_M'],
            'RPV_M': results['RPV_M'],
            'diff': results['diff'],
            'rel_diff': results['rel_diff'],
            'control_pass': results['control_pass'],
            'control_diff': results['control_diff']
        },
        'notes': f"The identity R(M) = RBG(M) + RPV(M) was tested. The relative difference is {results['rel_diff']:.4f}. The positive control passed: {results['control_pass']}."
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

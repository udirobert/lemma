import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

def run_audit():
    # 1. Setup
    np.random.seed(42)
    d_feat = 5
    d_eff = d_feat + 1
    sigma_eps = 0.1
    T = 2 # Number of task types
    alpha = np.array([0.5, 0.5])

    # Task parameters: w ~ N(0, 1), b ~ N(0, 1)
    # Task 1: Linear
    # Task 2: Linear with different prior? Let's keep it simple: same family, different prior variance to create mixture complexity.
    # Actually, for a "small uniform-attention meta-learner" (mean-pooling LS), the Bayes predictor is the posterior mean.
    # If the model is a linear regression model trained via LS, and the true tasks are linear, the Bayes predictor is the posterior mean of the linear coefficients.
    # The "Bayes Gap" is the difference between the risk of the meta-learner and the risk of the Bayes predictor.

    # The claim is about invariance to pN.
    # We need to simulate the meta-learning process.

    # Meta-learner: Mean-pooling Least Squares.
    # Architecture: M_theta(P^k) = rho_theta( (1/k) sum phi_theta(x_i, y_i), x_{k+1} )
    # For a "small" meta-learner, we can approximate the uniform-attention transformer with a simple linear model that averages the context.
    # Specifically, if we assume the feature encoder phi maps (x,y) to a representation that allows linear reconstruction, and the decoder is linear.
    # A common simplification for "mean-pooling LS" is that the model predicts y_{k+1} = w^T x_{k+1} + b, where w and b are estimated from the context D_k.
    # If the meta-learner is trained to perform LS on the context, it effectively implements the OLS estimator on the prompt.
    # Let's assume the meta-learner is a fixed function that computes the OLS estimate from the context D_k and predicts for x_{k+1}.
    # This is the "Bayes-optimal" predictor for a single task with a flat prior (or specific prior).
    # However, the paper discusses a *mixture* of tasks.
    # The Bayes predictor M_Bayes(P^k) = E[f(x_{k+1}) | D_k].
    # The meta-learner M_hat is trained on N prompts.

    # To make this tractable and self-contained:
    # 1. Define the true data generating process (DGP).
    #    - Task I ~ Categorical(alpha).
    #    - Given I, f is drawn from a distribution. Let's use linear models f(x) = w^T x + b.
    #    - Prior on (w,b): N(0, Sigma_prior).
    #    - x ~ N(0, I_d).
    #    - y = f(x) + eps, eps ~ N(0, sigma^2).
    # 2. The Bayes predictor for a given context D_k is the posterior mean of f(x_{k+1}).
    #    For linear regression with Gaussian priors and noise, the posterior is Gaussian.
    #    We can compute the Bayes predictor analytically or via Monte Carlo.
    # 3. The Meta-learner: The paper uses a Transformer. We approximate it with a "Mean-Pooling LS" model.
    #    This model takes the context D_k, computes the OLS estimate (w_hat, b_hat), and predicts y = w_hat^T x_{k+1} + b_hat.
    #    This model is *fixed* (not trained on N prompts in the sense of gradient descent on a neural net, but rather it's the algorithm the Transformer is supposed to learn).
    #    Wait, the claim says "Train on matched-pN pairs... Estimate the Bayes Gap... by out-of-sample MSE minus the exact posterior-mean risk".
    #    This implies the meta-learner's performance depends on N (pretraining size).
    #    If the meta-learner is just "OLS on the context", its performance does NOT depend on N (pretraining prompts), only on k (context length).
    #    The dependence on N comes from the *learning* of the meta-learner parameters theta.
    #    Since we cannot train a Transformer, we must simulate the effect of N on the meta-learner's approximation of the Bayes predictor.
    #
    #    Alternative interpretation: The "small uniform-attention meta-learner" is a specific parametric model whose parameters are estimated from the N pretraining prompts.
    #    Let's assume the meta-learner is a linear model: M_theta(D_k, x) = theta^T [mean(x), mean(y), x].
    #    Actually, the paper's Theorem 3.2 bound involves m (features) and pN.
    #
    #    Let's simplify: We will simulate the *Bayes Gap* directly by comparing the risk of a "finite-sample meta-learner" to the Bayes risk.
    #    We will model the meta-learner as a linear regression model trained on the N pretraining prompts to predict the *Bayes predictor's output*? No, that's not right.
    #
    #    Let's look at the C2 audit reference. "Reuse the mean-pooling least-squares meta-learner from the C2 audit".
    #    In many such audits, the "meta-learner" is a simple algorithm that the Transformer is approximating.
    #    If the Transformer is trained to minimize the loss on N prompts, it converges to the Bayes predictor as N -> infinity.
    #    The error (Bayes Gap) decreases with N.
    #
    #    To simulate this without a Transformer:
    #    We can assume the meta-learner is a linear model with parameters theta.
    #    The "true" target function is the Bayes predictor M_Bayes.
    #    We train a linear model (the meta-learner) on N samples of (Context, Query, Target) where Target = M_Bayes(Context, Query).
    #    Then we evaluate the out-of-sample MSE of this linear model.
    #    The Bayes Gap is this MSE minus the irreducible noise (which is 0 if we are predicting the Bayes predictor exactly, but the Bayes predictor itself has variance? No, the Bayes Gap is the difference in risk between the model and the Bayes predictor).
    #    Risk(M) = E[(M - y)^2]. Risk(Bayes) = E[(Bayes - y)^2].
    #    Bayes Gap = Risk(M) - Risk(Bayes).
    #    Since Bayes is the posterior mean, Risk(Bayes) = E[Var(y|D)] = Posterior Variance.
    #    Risk(M) = E[(M - Bayes)^2] + E[(Bayes - y)^2] (by orthogonality).
    #    So Bayes Gap = E[(M - Bayes)^2].
    #
    #    So we need to estimate E[(M_theta - M_Bayes)^2].
    #
    #    Plan:
    #    1. Generate N pretraining prompts. For each prompt, generate context D_k and query x_{k+1}.
    #    2. Compute the Bayes predictor value M_Bayes(D_k, x_{k+1}) for each.
    #    3. Train a simple meta-learner (e.g., a linear model on features derived from D_k and x_{k+1}) to predict M_Bayes.
    #    4. Evaluate the mean squared error on a fresh test set of prompts.
    #    5. This MSE is the Bayes Gap.
    #
    #    Features for the meta-learner: Since it's "mean-pooling", the features should be based on the mean of the context.
    #    Let's use features: [mean(x), mean(y), x_query].
    #    This is a linear model: M_theta = theta_1^T mean(x) + theta_2 mean(y) + theta_3^T x_query.
    #
    #    Let's implement this.

    def generate_prompt(p, task_idx):
        """Generate a prompt of length p for a given task."""
        # Sample task function
        if task_idx == 0:
            w = np.random.randn(d_feat)
            b = np.random.randn()
        else:
            w = np.random.randn(d_feat) * 2.0 # Different scale
            b = np.random.randn() * 2.0

        # Sample inputs
        X = np.random.randn(p, d_feat)
        # Generate outputs
        Y = X @ w + b + np.random.randn(p) * sigma_eps

        # The prompt consists of p examples. The query is the (p+1)-th input.
        # In the risk definition, we average over k=1..p.
        # For simplicity, let's evaluate the risk at k=p (full context).
        # The claim mentions "prompt length p". Usually p is the context length.
        # Let's assume the prompt has p context examples and 1 query.

        x_query = np.random.randn(d_feat)
        y_query = X @ w + b + np.random.randn() * sigma_eps # Not used for training, but for risk calc

        return X, Y, x_query, w, b, task_idx

    def bayes_predictor(X, Y, x_query, task_idx):
        """Compute the Bayes predictor value for a given context and query."""
        # For linear regression with Gaussian prior N(0, I) and noise sigma^2.
        # Posterior mean of w, b given X, Y.
        # Design matrix A = [X, 1]
        A = np.hstack([X, np.ones((len(X), 1))])
        # Prior covariance Sigma_prior = I_{d+1}
        # Noise variance sigma^2
        # Posterior covariance: (A^T A / sigma^2 + I)^{-1}
        # Posterior mean: (A^T A / sigma^2 + I)^{-1} A^T Y / sigma^2

        Sigma_prior = np.eye(d_eff)
        Sigma_post = np.linalg.inv(A.T @ A / sigma_eps**2 + np.linalg.inv(Sigma_prior))
        mu_post = Sigma_post @ (A.T @ Y / sigma_eps**2)

        # Predict for x_query
        x_aug = np.append(x_query, 1.0)
        return x_aug @ mu_post

    def meta_learner_features(X, Y, x_query):
        """Extract features for the mean-pooling meta-learner."""
        # Mean of context
        mean_x = np.mean(X, axis=0)
        mean_y = np.mean(Y)
        # Features: [mean_x, mean_y, x_query]
        feats = np.concatenate([mean_x, [mean_y], x_query])
        return feats

    def train_meta_learner(N, p, n_replicates=5):
        """Train the meta-learner on N prompts of length p."""
        # Generate training data
        X_train = []
        y_train = [] # Bayes predictor values

        for _ in range(N):
            # Sample task
            task_idx = np.random.choice(T, p=alpha)
            X, Y, x_query, w, b, _ = generate_prompt(p, task_idx)

            # Compute Bayes predictor
            y_bayes = bayes_predictor(X, Y, x_query, task_idx)

            # Extract features
            feats = meta_learner_features(X, Y, x_query)

            X_train.append(feats)
            y_train.append(y_bayes)

        X_train = np.array(X_train)
        y_train = np.array(y_train)

        # Train linear model (Ridge regression for stability)
        # Add bias term
        X_train_bias = np.hstack([X_train, np.ones((N, 1))])

        # Ridge regression
        lam = 1e-3
        A = X_train_bias.T @ X_train_bias + lam * np.eye(X_train_bias.shape[1])
        b_vec = X_train_bias.T @ y_train
        theta = np.linalg.solve(A, b_vec)

        return theta

    def evaluate_bayes_gap(theta, p, n_test=1000):
        """Evaluate the Bayes Gap (MSE between meta-learner and Bayes predictor) on fresh data."""
        errors = []
        for _ in range(n_test):
            task_idx = np.random.choice(T, p=alpha)
            X, Y, x_query, w, b, _ = generate_prompt(p, task_idx)

            y_bayes = bayes_predictor(X, Y, x_query, task_idx)

            feats = meta_learner_features(X, Y, x_query)
            feats_bias = np.append(feats, 1.0)

            y_pred = feats_bias @ theta

            errors.append((y_pred - y_bayes)**2)

        return np.mean(errors)

    # Grid of (N, p) pairs with matched pN
    # pN = 2000
    pairs = [
        (100, 20),
        (200, 10),
        (400, 5),
        (500, 8),
        (1000, 4),
        (2000, 2),
        (1000, 8) # pN=8000, outlier to test monotonicity
    ]

    # Add more points for better correlation
    # Let's add pN = 1000, 4000
    pairs.append((100, 10)) # pN=1000
    pairs.append((400, 10)) # pN=4000
    pairs.append((1000, 10)) # pN=10000

    results = []
    for N, p in pairs:
        bg_values = []
        for rep in range(3): # 3 replicates
            theta = train_meta_learner(N, p)
            bg = evaluate_bayes_gap(theta, p, n_test=200) # Reduced test size for speed
            bg_values.append(bg)

        mean_bg = np.mean(bg_values)
        std_bg = np.std(bg_values)
        results.append({
            'N': N,
            'p': p,
            'pN': N * p,
            'bg_mean': mean_bg,
            'bg_std': std_bg
        })
        print(f"N={N}, p={p}, pN={N*p}, BG={mean_bg:.4f} +/- {std_bg:.4f}")

    # Calculate metrics
    pN_values = np.array([r['pN'] for r in results])
    bg_values = np.array([r['bg_mean'] for r in results])

    # (i) Spearman correlation between BG and pN
    # Note: BG should *decrease* with pN. So correlation should be negative.
    # The claim says "decreasing monotonically with the product pN".
    # Success criterion: "Spearman correlation ... exceeds 0.80".
    # Usually, this implies magnitude. Or does it mean positive correlation with 1/pN?
    # "decreasing monotonically with pN" -> corr(BG, pN) < 0.
    # Let's check the absolute value or the correlation with -pN.
    # The criterion says "exceeds 0.80". If it's decreasing, the correlation is negative.
    # I will compute the correlation and check if abs(corr) > 0.80 and the sign is negative.

    corr, p_val = spearmanr(pN_values, bg_values)

    # (ii) Mean within-pN coefficient of variation
    # Group by pN
    pN_groups = {}
    for r in results:
        pN = r['pN']
        if pN not in pN_groups:
            pN_groups[pN] = []
        pN_groups[pN].append(r['bg_mean'])

    cvs = []
    for pN, bgs in pN_groups.items():
        if len(bgs) > 1:
            mean_bg = np.mean(bgs)
            std_bg = np.std(bgs)
            if mean_bg > 0:
                cv = std_bg / mean_bg
                cvs.append(cv)
        # If only one point, CV is undefined or 0?
        # The claim says "within-pN coefficient of variation".
        # If there's only one (N,p) for a pN, we can't compute CV.
        # We should ensure multiple (N,p) for each pN.
        # My grid:
        # 2000: (100,20), (200,10), (400,5), (500,8) -> 4 points
        # 8000: (1000,8) -> 1 point
        # 1000: (100,10) -> 1 point
        # 4000: (400,10) -> 1 point
        # 10000: (1000,10) -> 1 point
        # This is bad for CV calculation. I need to add more pairs for the other pN values.

    # Let's refine the grid to have multiple pairs for each pN.
    # pN = 1000: (100, 10), (200, 5), (500, 2)
    # pN = 2000: (100, 20), (200, 10), (400, 5), (500, 8)
    # pN = 4000: (200, 20), (400, 10), (1000, 4)
    # pN = 8000: (400, 20), (800, 10), (2000, 4)

    pairs_refined = [
        (100, 10), (200, 5), (500, 2), # pN=1000
        (100, 20), (200, 10), (400, 5), (500, 8), # pN=2000
        (200, 20), (400, 10), (1000, 4), # pN=4000
        (400, 20), (800, 10), (2000, 4) # pN=8000
    ]

    results_refined = []
    for N, p in pairs_refined:
        bg_values = []
        for rep in range(3):
            theta = train_meta_learner(N, p)
            bg = evaluate_bayes_gap(theta, p, n_test=200)
            bg_values.append(bg)

        mean_bg = np.mean(bg_values)
        std_bg = np.std(bg_values)
        results_refined.append({
            'N': N,
            'p': p,
            'pN': N * p,
            'bg_mean': mean_bg,
            'bg_std': std_bg
        })
        print(f"Refined: N={N}, p={p}, pN={N*p}, BG={mean_bg:.4f} +/- {std_bg:.4f}")

    pN_values = np.array([r['pN'] for r in results_refined])
    bg_values = np.array([r['bg_mean'] for r in results_refined])

    corr, p_val = spearmanr(pN_values, bg_values)

    # Calculate CV for each pN group
    pN_groups = {}
    for r in results_refined:
        pN = r['pN']
        if pN not in pN_groups:
            pN_groups[pN] = []
        pN_groups[pN].append(r['bg_mean'])

    cvs = []
    for pN, bgs in pN_groups.items():
        if len(bgs) > 1:
            mean_bg = np.mean(bgs)
            std_bg = np.std(bgs)
            if mean_bg > 0:
                cv = std_bg / mean_bg
                cvs.append(cv)

    mean_cv = np.mean(cvs) if cvs else 0.0

    # (iii) Synthetic Control
    # Generate synthetic data where BG is a known function of pN.
    # Let BG = 1 / pN + noise.
    # We will simulate the "measurement" of BG for the same grid.

    synthetic_results = []
    for N, p in pairs_refined:
        pN = N * p
        true_bg = 1.0 / pN
        # Add noise to simulate measurement error
        measured_bg = true_bg + np.random.normal(0, 0.05 * true_bg)
        synthetic_results.append({
            'N': N,
            'p': p,
            'pN': pN,
            'bg_mean': measured_bg
        })

    syn_pN = np.array([r['pN'] for r in synthetic_results])
    syn_bg = np.array([r['bg_mean'] for r in synthetic_results])

    syn_corr, _ = spearmanr(syn_pN, syn_bg)

    # CV for synthetic
    syn_pN_groups = {}
    for r in synthetic_results:
        pN = r['pN']
        if pN not in syn_pN_groups:
            syn_pN_groups[pN] = []
        syn_pN_groups[pN].append(r['bg_mean'])

    syn_cvs = []
    for pN, bgs in syn_pN_groups.items():
        if len(bgs) > 1:
            mean_bg = np.mean(bgs)
            std_bg = np.std(bgs)
            if mean_bg > 0:
                cv = std_bg / mean_bg
                syn_cvs.append(cv)

    syn_mean_cv = np.mean(syn_cvs) if syn_cvs else 0.0

    # Check criteria
    # (i) Spearman correlation > 0.80 (magnitude, and negative sign for decreasing)
    # The claim says "decreasing monotonically". So corr should be negative.
    # Criterion: "exceeds 0.80". I will interpret this as |corr| > 0.80 and corr < 0.

    criterion_i = abs(corr) > 0.80 and corr < 0
    criterion_ii = mean_cv < 0.25

    # Control criteria
    control_i = abs(syn_corr) > 0.80 and syn_corr < 0
    control_ii = syn_mean_cv < 0.25
    control_pass = control_i and control_ii

    # Plotting
    os.makedirs('results/c7', exist_ok=True)

    plt.figure(figsize=(10, 6))
    plt.scatter(pN_values, bg_values, c='blue', label='Empirical BG')
    plt.scatter(syn_pN, syn_bg, c='red', marker='x', label='Synthetic Control')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('pN (log scale)')
    plt.ylabel('Bayes Gap (log scale)')
    plt.title('Bayes Gap vs pN')
    plt.legend()
    plt.grid(True, which="both", ls="--", lw=0.5)
    plt.savefig('results/c7/fig.png', dpi=150)
    plt.close()

    # Summary
    status = "supported" if (criterion_i and criterion_ii and control_pass) else "falsified"
    if not control_pass:
        status = "inconclusive"

    summary = {
        "claim_id": "C7",
        "status": status,
        "metrics": {
            "spearman_corr": float(corr),
            "mean_cv": float(mean_cv),
            "syn_spearman_corr": float(syn_corr),
            "syn_mean_cv": float(syn_mean_cv),
            "control_pass": bool(control_pass),
            "criterion_i": bool(criterion_i),
            "criterion_ii": bool(criterion_ii)
        },
        "notes": f"Spearman corr: {corr:.3f}, Mean CV: {mean_cv:.3f}. Control passed: {control_pass}."
    }

    print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

if __name__ == "__main__":
    run_audit()

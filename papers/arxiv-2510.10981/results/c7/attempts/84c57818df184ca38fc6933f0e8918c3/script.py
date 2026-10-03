import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

np.random.seed(42)

# --- Configuration ---
d_feat = 2
sigma_eps = 0.1
n_reps = 5
n_test = 200

# Grid of (N, p) pairs
# pN values: 2000, 4000, 8000
# We need at least 3 distinct pN values for Spearman correlation.
# Let's use:
# pN=2000: (100, 20), (200, 10), (400, 5), (500, 4), (1000, 2), (2000, 1)
# pN=4000: (200, 20), (400, 10), (800, 5), (1000, 4), (2000, 2), (4000, 1)
# pN=8000: (400, 20), (800, 10), (1600, 5), (2000, 4), (4000, 2), (8000, 1)

# To keep runtime reasonable, we'll select a subset that covers the range well.
# The prompt suggests: (100,20),(200,10),(400,5),(500,8),(1000,4),(2000,2),(1000,8)
# Let's stick to a structured grid to ensure matched pN groups.

grid = [
    # pN = 2000
    (100, 20), (200, 10), (400, 5), (500, 4), (1000, 2), (2000, 1),
    # pN = 4000
    (200, 20), (400, 10), (800, 5), (1000, 4), (2000, 2), (4000, 1),
    # pN = 8000
    (400, 20), (800, 10), (1600, 5), (2000, 4), (4000, 2), (8000, 1)
]

# --- Meta-Learner Implementation ---
# The C2 audit used a mean-pooling least-squares meta-learner.
# Architecture: M_theta(P^k) = rho_theta( (1/k) sum phi_theta(x_i, y_i), x_{k+1} )
# For a linear meta-learner (least squares), we can approximate this.
# Let's implement a simple linear model that takes the mean of features and the query.
# Feature map phi(x, y) = [x, y, 1] (simple linear features)
# Aggregated feature z = mean(phi(x_i, y_i))
# Input to decoder v = [z, x_query]
# Model: y_hat = v @ W

def generate_prompt(N, p, d_feat, sigma_eps):
    """
    Generate N prompts of length p.
    Each prompt has p examples and 1 query.
    Task: y = w^T x + b + noise. w, b sampled from N(0, 1).
    """
    prompts = []
    for _ in range(N):
        w = np.random.randn(d_feat)
        b = np.random.randn()
        X = np.random.randn(p, d_feat)
        y = X @ w + b + np.random.randn(p) * sigma_eps
        x_query = np.random.randn(d_feat)
        prompts.append((X, y, x_query))
    return prompts

def train_meta_learner(prompts, d_feat):
    """
    Train a linear meta-learner.
    Input features for each example (x, y): [x, y, 1] -> dim d_feat + 2
    Aggregated feature z: mean of [x, y, 1] over p examples.
    Decoder input v: [z, x_query] -> dim (d_feat + 2) + d_feat = 2*d_feat + 2
    Model: y_hat = v @ W
    """
    # Collect training data
    X_train = []
    y_train = []

    for (X, y, x_query) in prompts:
        p = len(y)
        # Compute features for each example
        # phi(x, y) = [x, y, 1]
        # We need to compute the mean of phi over the p examples.
        # z = (1/p) sum_{i=1}^p [x_i, y_i, 1]
        # z = [mean(x), mean(y), 1]

        mean_x = np.mean(X, axis=0)
        mean_y = np.mean(y)
        z = np.concatenate([mean_x, [mean_y, 1.0]])

        # Decoder input v = [z, x_query]
        v = np.concatenate([z, x_query])

        # Target is the true y for the query? No, the target is the label of the query.
        # Wait, the prompt generation in Definition 2.1 says:
        # "Form the length-p (complete) prompt: P = (x1, y1, ..., xp, yp, x_{p+1})"
        # The risk is defined as predicting y_{k+1} from P^k.
        # In the training set, we have N prompts. Each prompt has p examples and a query.
        # The target for the query is y_{p+1}.
        # We need to generate y_{p+1} as well.

        # Let's re-generate the prompt to include the query label.
        # Actually, in the loop above, I didn't generate y_query.
        # Let's fix the generation to include y_query.

        # For now, let's assume we have y_query. I'll modify the generation function.
        pass

# Let's rewrite the generation and training to be cleaner.

def generate_prompt_with_query(N, p, d_feat, sigma_eps):
    """
    Generate N prompts. Each prompt has p context examples and 1 query.
    Returns list of (X_ctx, y_ctx, x_query, y_query)
    """
    prompts = []
    for _ in range(N):
        w = np.random.randn(d_feat)
        b = np.random.randn()

        X_ctx = np.random.randn(p, d_feat)
        y_ctx = X_ctx @ w + b + np.random.randn(p) * sigma_eps

        x_query = np.random.randn(d_feat)
        y_query = x_query @ w + b + np.random.randn() * sigma_eps

        prompts.append((X_ctx, y_ctx, x_query, y_query))
    return prompts

def train_meta_learner(prompts, d_feat):
    """
    Train a linear meta-learner.
    Input features for each example (x, y): [x, y, 1] -> dim d_feat + 2
    Aggregated feature z: mean of [x, y, 1] over p examples.
    Decoder input v: [z, x_query] -> dim (d_feat + 2) + d_feat = 2*d_feat + 2
    Model: y_hat = v @ W
    """
    X_train = []
    y_train = []

    for (X_ctx, y_ctx, x_query, y_query) in prompts:
        p = len(y_ctx)

        mean_x = np.mean(X_ctx, axis=0)
        mean_y = np.mean(y_ctx)
        z = np.concatenate([mean_x, [mean_y, 1.0]])

        v = np.concatenate([z, x_query])

        X_train.append(v)
        y_train.append(y_query)

    X_train = np.array(X_train)
    y_train = np.array(y_train)

    # Least squares: W = (X^T X)^{-1} X^T y
    # Add small regularization for stability
    reg = 1e-6
    W = np.linalg.solve(X_train.T @ X_train + reg * np.eye(X_train.shape[1]), X_train.T @ y_train)

    return W

def predict_meta_learner(W, X_ctx, y_ctx, x_query, d_feat):
    """
    Predict y_query using the trained meta-learner.
    """
    p = len(y_ctx)
    mean_x = np.mean(X_ctx, axis=0)
    mean_y = np.mean(y_ctx)
    z = np.concatenate([mean_x, [mean_y, 1.0]])
    v = np.concatenate([z, x_query])
    return v @ W

def estimate_bayes_gap(N, p, d_feat, n_reps, n_test, sigma_eps):
    """
    Estimate the Bayes Gap for a given (N, p).
    Bayes Gap = E[ (M_theta - M_Bayes)^2 ]

    We approximate M_Bayes by the posterior mean.
    For a linear task with Gaussian prior and noise, the posterior mean is a linear function of the context.
    However, computing the exact posterior mean for a mixture of tasks is complex.

    Alternative: The claim is about the *invariance* of the gap to the split of pN.
    We can estimate the gap as the difference between the model's MSE and the Bayes risk.
    Bayes risk is the minimum possible MSE.

    Let's estimate the model's MSE on fresh test data.
    Then we need an estimate of the Bayes risk.

    For a single task (no mixture), the Bayes risk is the noise variance sigma_eps^2.
    But we have a mixture of tasks (w, b ~ N(0,1)).
    The Bayes risk for a new task is the expected noise variance, which is still sigma_eps^2.
    Wait, the Bayes risk is the risk of the optimal predictor.
    The optimal predictor is the posterior mean.
    The risk of the posterior mean is the posterior variance.

    Let's compute the posterior variance for a linear task.
    Given context D_k = {(x_i, y_i)}_{i=1}^k.
    Prior: w ~ N(0, I), b ~ N(0, 1).
    Likelihood: y_i = w^T x_i + b + eps_i.

    This is a standard Bayesian linear regression.
    The posterior variance of y_{k+1} = w^T x_{k+1} + b is:
    Var(y_{k+1} | D_k) = sigma_eps^2 + x_{k+1}^T Cov(w|D_k) x_{k+1} + Var(b|D_k) + 2 x_{k+1}^T Cov(w,b|D_k)

    This is computable.

    So, Bayes Gap = MSE(M_theta) - E[Var(y_{k+1} | D_k)]

    We can estimate E[Var(y_{k+1} | D_k)] by averaging over test prompts.
    """
    gaps = []

    for rep in range(n_reps):
        # Train
        train_prompts = generate_prompt_with_query(N, p, d_feat, sigma_eps)
        W = train_meta_learner(train_prompts, d_feat)

        # Test
        test_prompts = generate_prompt_with_query(n_test, p, d_feat, sigma_eps)

        mse_sum = 0.0
        bayes_risk_sum = 0.0

        for (X_ctx, y_ctx, x_query, y_query) in test_prompts:
            # Model prediction
            y_pred = predict_meta_learner(W, X_ctx, y_ctx, x_query, d_feat)
            mse_sum += (y_pred - y_query) ** 2

            # Bayes risk (Posterior Variance)
            # We need to compute the posterior variance of y_query given X_ctx, y_ctx.
            # This requires knowing the prior and the likelihood.
            # Prior: w ~ N(0, I_d), b ~ N(0, 1).
            # Let's implement a function to compute the posterior variance.

            pv = compute_posterior_variance(X_ctx, y_ctx, x_query, d_feat, sigma_eps)
            bayes_risk_sum += pv

        mse = mse_sum / n_test
        bayes_risk = bayes_risk_sum / n_test

        gap = mse - bayes_risk
        gaps.append(gap)

    return np.mean(gaps), np.std(gaps)

def compute_posterior_variance(X, y, x_query, d_feat, sigma_eps):
    """
    Compute the posterior variance of y_query = w^T x_query + b given context (X, y).
    Prior: w ~ N(0, I_d), b ~ N(0, 1).
    """
    # We can use the standard formulas for Bayesian linear regression.
    # Let's augment the design matrix to include the bias term.
    # X_aug = [X, 1]
    # w_aug = [w, b]
    # Prior: w_aug ~ N(0, I_{d+1})

    X_aug = np.hstack([X, np.ones((len(X), 1))])

    # Posterior covariance:
    # Sigma_post = (Sigma_prior^{-1} + X^T X / sigma_eps^2)^{-1}
    # Sigma_prior = I

    Sigma_prior_inv = np.eye(d_feat + 1)
    Sigma_post_inv = Sigma_prior_inv + X_aug.T @ X_aug / (sigma_eps ** 2)
    Sigma_post = np.linalg.inv(Sigma_post_inv)

    # Posterior variance of y_query:
    # y_query = w_aug^T x_aug_query
    # Var(y_query | D) = x_aug_query^T Sigma_post x_aug_query + sigma_eps^2

    x_aug_query = np.concatenate([x_query, [1.0]])

    pv = x_aug_query @ Sigma_post @ x_aug_query + sigma_eps ** 2

    return pv

# --- Main Audit ---

results = []

for (N, p) in grid:
    print(f"Processing N={N}, p={p} (pN={N*p})...")
    gap_mean, gap_std = estimate_bayes_gap(N, p, d_feat, n_reps, n_test, sigma_eps)
    results.append({
        'N': N,
        'p': p,
        'pN': N * p,
        'gap_mean': gap_mean,
        'gap_std': gap_std
    })
    print(f"  Gap: {gap_mean:.4f} +/- {gap_std:.4f}")

# --- Analysis ---

# 1. Spearman correlation between Bayes Gap and pN
pN_values = np.array([r['pN'] for r in results])
gap_values = np.array([r['gap_mean'] for r in results])

# We need to handle the fact that we have multiple points for the same pN.
# The claim says "Spearman correlation between Bayes Gap and pN across the full grid".
# This implies using all points.

rho, pval = spearmanr(pN_values, gap_values)
print(f"Spearman correlation: {rho:.4f} (p={pval:.4f})")

# 2. Mean within-pN coefficient of variation
# Group by pN
from collections import defaultdict
pN_groups = defaultdict(list)
for r in results:
    pN_groups[r['pN']].append(r['gap_mean'])

cvs = []
for pN, gaps in pN_groups.items():
    if len(gaps) > 1:
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        if mean_gap != 0:
            cv = std_gap / abs(mean_gap)
            cvs.append(cv)

mean_cv = np.mean(cvs) if cvs else 0.0
print(f"Mean within-pN CV: {mean_cv:.4f}")

# 3. Synthetic Control
# Generate synthetic data where BG is a known function of pN only.
# Let BG = 1 / pN + noise.
# We will generate synthetic (N, p) pairs and compute BG.
# Then check if the same statistics hold.

synthetic_results = []
for (N, p) in grid:
    pN = N * p
    # True BG = 1 / pN
    true_bg = 1.0 / pN
    # Add noise
    bg = true_bg + np.random.randn() * 0.01
    synthetic_results.append({
        'N': N,
        'p': p,
        'pN': pN,
        'gap_mean': bg
    })

syn_pN_values = np.array([r['pN'] for r in synthetic_results])
syn_gap_values = np.array([r['gap_mean'] for r in synthetic_results])

syn_rho, syn_pval = spearmanr(syn_pN_values, syn_gap_values)
print(f"Synthetic Spearman correlation: {syn_rho:.4f}")

syn_pN_groups = defaultdict(list)
for r in synthetic_results:
    syn_pN_groups[r['pN']].append(r['gap_mean'])

syn_cvs = []
for pN, gaps in syn_pN_groups.items():
    if len(gaps) > 1:
        mean_gap = np.mean(gaps)
        std_gap = np.std(gaps)
        if mean_gap != 0:
            cv = std_gap / abs(mean_gap)
            syn_cvs.append(cv)

syn_mean_cv = np.mean(syn_cvs) if syn_cvs else 0.0
print(f"Synthetic Mean within-pN CV: {syn_mean_cv:.4f}")

# --- Success Criteria ---

# (i) Spearman correlation > 0.80
# (ii) Mean within-pN CV < 0.25
# (iii) Synthetic control satisfies (i) and (ii)

# Note: The Bayes Gap should *decrease* with pN.
# So the correlation should be *negative*.
# The claim says "decreasing monotonically with the product pN".
# So we expect rho < -0.80.
# Let's check the absolute value or the sign.
# "Spearman correlation ... exceeds 0.80" usually means magnitude.
# But if it's decreasing, it should be -0.80.
# Let's assume the criterion is |rho| > 0.80 and the trend is decreasing.

# Let's check the trend.
# If rho is positive, it's increasing. If negative, decreasing.
# The claim says "decreasing monotonically".
# So we expect rho to be negative.

# Let's define the criteria carefully.
# Criterion (i): "Spearman correlation between Bayes Gap and pN ... exceeds 0.80"
# This is ambiguous. Does it mean rho > 0.80 or |rho| > 0.80?
# Given the context "decreasing monotonically", it must mean the magnitude is > 0.80 and the sign is negative.
# Or perhaps the paper defines the gap such that it increases? No, gap decreases.
# Let's assume the criterion is: rho < -0.80.

# Let's check the synthetic control.
# Synthetic BG = 1/pN. This is decreasing.
# So syn_rho should be close to -1.

# Let's evaluate:
# (i) rho < -0.80
# (ii) mean_cv < 0.25
# (iii) syn_rho < -0.80 and syn_mean_cv < 0.25

# However, if the gap is very small, the CV might be large due to noise.
# Let's see the actual numbers.

# Let's also plot the results.

os.makedirs('results/c7', exist_ok=True)

plt.figure(figsize=(10, 6))
plt.scatter(pN_values, gap_values, c='blue', label='Empirical')
plt.scatter(syn_pN_values, syn_gap_values, c='red', label='Synthetic', alpha=0.5)
plt.xscale('log')
plt.yscale('log')
plt.xlabel('pN (log scale)')
plt.ylabel('Bayes Gap (log scale)')
plt.title('Bayes Gap vs pN')
plt.legend()
plt.grid(True, which="both", ls="--", lw=0.5)
plt.savefig('results/c7/fig.png', dpi=150, bbox_inches='tight')
plt.close()

# Determine status
# If the control fails, status is inconclusive.
control_pass = (syn_rho < -0.80) and (syn_mean_cv < 0.25)

if not control_pass:
    status = "inconclusive"
    notes = "Synthetic control failed. The statistic or setup may be buggy."
else:
    # Check empirical results
    # (i) rho < -0.80
    # (ii) mean_cv < 0.25

    # Note: The gap might not be strictly monotonic due to noise.
    # But the correlation should be strong.

    if rho < -0.80 and mean_cv < 0.25:
        status = "supported"
        notes = f"Spearman rho={rho:.3f}, Mean CV={mean_cv:.3f}. Both criteria met."
    else:
        status = "falsified"
        notes = f"Spearman rho={rho:.3f}, Mean CV={mean_cv:.3f}. Criteria not met."

summary = {
    "claim_id": "C7",
    "status": status,
    "metrics": {
        "spearman_rho": float(rho),
        "mean_cv": float(mean_cv),
        "syn_spearman_rho": float(syn_rho),
        "syn_mean_cv": float(syn_mean_cv),
        "control_pass": bool(control_pass)
    },
    "notes": notes
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

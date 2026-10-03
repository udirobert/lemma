import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

# Constants and Setup
np.random.seed(42)

# Hyperparameters for the meta-learner (C2 setup)
d_feat = 2          # Input dimension
sigma_eps = 0.1     # Noise standard deviation
T_types = 2         # Number of task types
alpha_mix = np.array([0.5, 0.5]) # Mixture weights

# Task definitions
# Type 0: Linear y = w^T x + b
# Type 1: Quadratic y = w^T x + b + c * ||x||^2

def sample_task(i):
    """Sample parameters for task type i."""
    if i == 0:
        w = np.random.randn(d_feat) * 1.0
        b = np.random.randn() * 0.5
        c = 0.0
    else:
        w = np.random.randn(d_feat) * 1.0
        b = np.random.randn() * 0.5
        c = np.random.randn() * 0.5
    return w, b, c

def generate_prompt(p, task_type=None):
    """
    Generate a single prompt of length p.
    Returns: X (p x d_feat), Y (p,), query_x (d_feat,), query_y (scalar), task_type
    """
    if task_type is None:
        task_type = np.random.choice(T_types, p=alpha_mix)

    w, b, c = sample_task(task_type)

    # Sample inputs
    X = np.random.randn(p, d_feat)

    # Generate outputs
    Y = np.zeros(p)
    for j in range(p):
        Y[j] = w @ X[j] + b + c * np.dot(X[j], X[j]) + np.random.randn() * sigma_eps

    # Query
    x_q = np.random.randn(d_feat)
    y_q = w @ x_q + b + c * np.dot(x_q, x_q) + np.random.randn() * sigma_eps

    return X, Y, x_q, y_q, task_type

def bayes_predictor(X, Y, x_q, task_type):
    """
    Compute the Bayes-optimal prediction for the given context and query.
    Since we know the task type in this synthetic audit (or we can assume the model
    has learned to distinguish them, but for the 'exact' Bayes gap we use the true
    posterior mean given the true task type), we compute the posterior mean
    for the specific task type.

    For linear/quadratic regression with Gaussian noise and Gaussian prior on weights,
    the posterior mean is the MAP/ML estimator if priors are flat, or exact Bayesian
    regression if priors are Gaussian.

    To keep it simple and consistent with 'exact posterior-mean risk', we assume
    a flat prior on the parameters (w, b, c) for the specific task type.
    This reduces to the OLS solution for that task type.
    """
    if task_type == 0:
        # Linear: y = w^T x + b
        # Design matrix: [X, 1]
        A = np.hstack([X, np.ones((len(X), 1))])
        # Solve least squares
        try:
            params, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
            w_hat, b_hat = params[:d_feat], params[d_feat]
            return w_hat @ x_q + b_hat
        except:
            return 0.0
    else:
        # Quadratic: y = w^T x + b + c * ||x||^2
        # Design matrix: [X, 1, ||x||^2]
        norms_sq = np.sum(X**2, axis=1).reshape(-1, 1)
        A = np.hstack([X, np.ones((len(X), 1)), norms_sq])
        try:
            params, _, _, _ = np.linalg.lstsq(A, Y, rcond=None)
            w_hat, b_hat, c_hat = params[:d_feat], params[d_feat], params[d_feat+1]
            return w_hat @ x_q + b_hat + c_hat * np.dot(x_q, x_q)
        except:
            return 0.0

def train_meta_learner(N, p, n_replicates=5):
    """
    Train a simple meta-learner (mean-pooling least squares) on N prompts of length p.
    The meta-learner is a linear model on the pooled features.

    Architecture:
    For each prompt, we compute a feature vector z.
    z = [mean(X), mean(Y), mean(||X||^2), ...] ?

    Actually, the C2 setup uses a 'mean-pooling least-squares meta-learner'.
    Let's implement a simple linear meta-learner that takes the mean of the context
    and the query, and predicts the output.

    Input to meta-learner: [mean(X_context), mean(Y_context), x_query]
    Output: y_pred

    We train this linear model on the pretraining data.
    """
    # Collect training data for the meta-learner
    # For each prompt, we generate multiple (context, query) pairs?
    # The paper defines risk averaged over k=1..p.
    # For simplicity in this audit, we use the full context D_p for prediction of x_{p+1}.
    # This is a simplification of the sequential risk, but captures the pN scaling.

    X_meta = []
    Y_meta = []

    for _ in range(N):
        X_ctx, Y_ctx, x_q, y_q, t_type = generate_prompt(p)

        # Features for meta-learner
        # Mean pooling of context
        mean_x = np.mean(X_ctx, axis=0)
        mean_y = np.mean(Y_ctx)

        # Feature vector: [mean_x, mean_y, x_q]
        feat = np.concatenate([mean_x, [mean_y], x_q])
        X_meta.append(feat)
        Y_meta.append(y_q)

    X_meta = np.array(X_meta)
    Y_meta = np.array(Y_meta)

    # Train linear meta-learner (Ridge regression for stability)
    # y = W^T feat + b
    # Add bias term
    X_meta_aug = np.hstack([X_meta, np.ones((len(X_meta), 1))])

    # Solve least squares
    try:
        W_meta, _, _, _ = np.linalg.lstsq(X_meta_aug, Y_meta, rcond=None)
    except:
        W_meta = np.zeros(X_meta_aug.shape[1])

    return W_meta

def evaluate_bayes_gap(W_meta, N, p, n_eval=100):
    """
    Evaluate the Bayes Gap for the trained meta-learner.
    Bayes Gap = E[ (M_theta - M_Bayes)^2 ]
    """
    gap_sums = []

    for _ in range(n_eval):
        X_ctx, Y_ctx, x_q, y_q, t_type = generate_prompt(p)

        # Meta-learner prediction
        mean_x = np.mean(X_ctx, axis=0)
        mean_y = np.mean(Y_ctx)
        feat = np.concatenate([mean_x, [mean_y], x_q])
        feat_aug = np.concatenate([feat, [1.0]])
        pred_meta = W_meta @ feat_aug

        # Bayes prediction
        pred_bayes = bayes_predictor(X_ctx, Y_ctx, x_q, t_type)

        # Gap squared
        gap_sq = (pred_meta - pred_bayes) ** 2
        gap_sums.append(gap_sq)

    return np.mean(gap_sums)

# --- Main Audit Logic ---

# Grid of (N, p) pairs with matched pN
# pN = 2000
pairs = [
    (100, 20),
    (200, 10),
    (400, 5),
    (500, 8),
    (1000, 4),
    (2000, 2),
    (1000, 8) # pN=8000, outlier for trend? No, let's stick to matched pN for CV.
              # The claim says "at matched values of pN".
              # Let's use a grid where pN varies to check monotonicity,
              # and groups where pN is fixed to check invariance.
]

# To test monotonicity vs pN, we need different pN values.
# To test invariance, we need same pN values.
# Let's create a grid:
# Group 1: pN = 1000 -> (100, 10), (200, 5), (500, 2)
# Group 2: pN = 2000 -> (100, 20), (200, 10), (400, 5), (1000, 2)
# Group 3: pN = 4000 -> (100, 40), (200, 20), (400, 10), (1000, 4), (2000, 2)

grid = [
    # (N, p, pN)
    (100, 10, 1000),
    (200, 5, 1000),
    (500, 2, 1000),

    (100, 20, 2000),
    (200, 10, 2000),
    (400, 5, 2000),
    (1000, 2, 2000),

    (100, 40, 4000),
    (200, 20, 4000),
    (400, 10, 4000),
    (1000, 4, 4000),
    (2000, 2, 4000),
]

results = []

for N, p, pN in grid:
    # Train and evaluate
    # Use a fixed number of replicates for stability
    n_rep = 3
    gaps = []
    for r in range(n_rep):
        W_meta = train_meta_learner(N, p)
        gap = evaluate_bayes_gap(W_meta, N, p, n_eval=50)
        gaps.append(gap)

    mean_gap = np.mean(gaps)
    std_gap = np.std(gaps)

    results.append({
        'N': N,
        'p': p,
        'pN': pN,
        'gap': mean_gap,
        'gap_std': std_gap
    })
    print(f"N={N}, p={p}, pN={pN}, Gap={mean_gap:.4f} +/- {std_gap:.4f}")

# --- Analysis ---

# 1. Spearman correlation between Bayes Gap and pN
pN_vals = np.array([r['pN'] for r in results])
gap_vals = np.array([r['gap'] for r in results])

# Note: The claim says "decreasing monotonically with the product pN".
# So we expect a negative correlation.
# Spearman correlation should be < -0.80 (or > 0.80 if we look at magnitude, but sign matters).
# The success criterion says "exceeds 0.80". Usually this implies magnitude or positive.
# Given "decreasing", the correlation should be strongly negative.
# Let's check the absolute value or the negative correlation.
# "Spearman correlation ... exceeds 0.80" usually means rho > 0.8.
# If the trend is decreasing, rho will be negative.
# I will interpret "exceeds 0.80" as |rho| > 0.80 and check the sign is negative.
# Or perhaps the claim implies the gap *magnitude* decreases, so correlation with pN is negative.
# Let's compute rho and check if rho < -0.80.

rho, p_val = spearmanr(pN_vals, gap_vals)
print(f"Spearman rho (Gap vs pN): {rho:.4f}")

# Criterion (i): Spearman correlation ... exceeds 0.80.
# If the paper claims invariance at matched pN and decrease with pN,
# the correlation should be strongly negative.
# I will assume the criterion means |rho| > 0.80 and the direction is decreasing.
# If rho is positive, it's falsified.
# If rho is negative and |rho| > 0.8, it's supported.

# 2. Mean within-pN coefficient of variation
# Group by pN
groups = {}
for r in results:
    pN = r['pN']
    if pN not in groups:
        groups[pN] = []
    groups[pN].append(r['gap'])

cvs = []
for pN, gaps in groups.items():
    if len(gaps) > 1:
        mean_g = np.mean(gaps)
        std_g = np.std(gaps)
        cv = std_g / mean_g if mean_g != 0 else 0
        cvs.append(cv)
        print(f"pN={pN}, Mean Gap={mean_g:.4f}, Std={std_g:.4f}, CV={cv:.4f}")

mean_cv = np.mean(cvs) if cvs else 0
print(f"Mean within-pN CV: {mean_cv:.4f}")

# Criterion (ii): Mean within-pN CV < 0.25

# --- Positive Control ---
# Generate synthetic data where BG is generated from a known pN-only law.
# Law: BG = 1 / pN + noise
# We simulate this to ensure our statistics (Spearman, CV) work correctly.

np.random.seed(123)
synthetic_results = []
for N, p, pN in grid:
    true_bg = 1.0 / pN
    # Add small noise
    noise = np.random.randn(3) * 0.01 * true_bg
    for r in range(3):
        bg = true_bg + noise[r]
        synthetic_results.append({
            'N': N,
            'p': p,
            'pN': pN,
            'gap': bg
        })

syn_pN = np.array([r['pN'] for r in synthetic_results])
syn_gap = np.array([r['gap'] for r in synthetic_results])

syn_rho, _ = spearmanr(syn_pN, syn_gap)

syn_groups = {}
for r in synthetic_results:
    pN = r['pN']
    if pN not in syn_groups:
        syn_groups[pN] = []
    syn_groups[pN].append(r['gap'])

syn_cvs = []
for pN, gaps in syn_groups.items():
    if len(gaps) > 1:
        mean_g = np.mean(gaps)
        std_g = np.std(gaps)
        cv = std_g / mean_g if mean_g != 0 else 0
        syn_cvs.append(cv)

syn_mean_cv = np.mean(syn_cvs) if syn_cvs else 0

print(f"Control Spearman rho: {syn_rho:.4f}")
print(f"Control Mean CV: {syn_mean_cv:.4f}")

control_pass = (abs(syn_rho) > 0.80) and (syn_mean_cv < 0.25)

# --- Final Status ---

# Check criteria
# (i) Spearman correlation exceeds 0.80 (interpreted as |rho| > 0.8 and negative for decrease)
# The claim says "decreasing monotonically". So rho should be negative.
# If rho > 0.8, it's increasing. If rho < -0.8, it's decreasing.
# The text says "exceeds 0.80". In common parlance for correlation strength, this means magnitude.
# But strictly, "correlation exceeds 0.8" means rho > 0.8.
# However, the claim is about *decrease*.
# I will check if rho < -0.80. If rho > 0.80, it contradicts "decreasing".
# If -0.80 < rho < 0.80, it's weak.

# Let's be precise. The claim: "decreasing monotonically with the product pN".
# Success criterion: "Spearman correlation ... exceeds 0.80".
# This is ambiguous. Does it mean rho > 0.8 or |rho| > 0.8?
# Given the context of "decreasing", a positive correlation would be a failure.
# I will assume the criterion requires a strong negative correlation, i.e., rho < -0.80.
# If the paper meant magnitude, it would usually say "absolute correlation".
# However, if I report rho = -0.9, it "exceeds" 0.8 in magnitude.
# Let's look at the control. The control has BG = 1/pN, so rho should be close to -1.
# If the control passes with rho < -0.8, then the criterion likely implies strong negative correlation.

# Let's check the control first.
if syn_rho < -0.80:
    control_rho_pass = True
else:
    control_rho_pass = False

if syn_mean_cv < 0.25:
    control_cv_pass = True
else:
    control_cv_pass = False

# If the control fails, the statistic is buggy or the setup is wrong.
# In our synthetic case, rho should be very negative. CV should be small.

# Now check the real data.
real_rho_pass = rho < -0.80
real_cv_pass = mean_cv < 0.25

# If control fails, status is inconclusive.
if not (control_rho_pass and control_cv_pass):
    status = "inconclusive"
    notes = f"Control failed. Control rho={syn_rho:.2f}, Control CV={syn_mean_cv:.2f}."
else:
    if real_rho_pass and real_cv_pass:
        status = "supported"
        notes = f"Claim supported. Spearman rho={rho:.2f} (strong negative), Mean CV={mean_cv:.2f} (<0.25)."
    else:
        status = "falsified"
        notes = f"Claim falsified. Spearman rho={rho:.2f}, Mean CV={mean_cv:.2f}."

# Plotting
os.makedirs('results/c7', exist_ok=True)

fig, ax = plt.subplots(figsize=(10, 6))

# Plot real data
for r in results:
    ax.scatter(r['pN'], r['gap'], label=f"N={r['N']}, p={r['p']}", alpha=0.7)

# Plot synthetic control (scaled for visibility if needed, or just show trend)
# We'll plot them on a secondary axis or just note them.
# Let's just plot the real data and add a text box for the control.

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlabel('pN (log scale)')
ax.set_ylabel('Bayes Gap (log scale)')
ax.set_title('Bayes Gap vs pN')
ax.legend()
ax.grid(True, which="both", ls="--", lw=0.5)

plt.tight_layout()
plt.savefig('results/c7/fig.png', dpi=150)
plt.close()

# Summary JSON
summary = {
    "claim_id": "C7",
    "status": status,
    "metrics": {
        "spearman_rho": float(rho),
        "mean_within_pN_cv": float(mean_cv),
        "control_spearman_rho": float(syn_rho),
        "control_mean_cv": float(syn_mean_cv),
        "control_pass": bool(control_pass),
        "n_grid_points": len(results)
    },
    "notes": notes
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

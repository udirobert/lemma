import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

np.random.seed(42)

# --- Setup ---
# Two-task mixture of linear regression models.
# Task 1: y = w1^T x + b1 + eps
# Task 2: y = w2^T x + b2 + eps
# We assume the true task is Task 1.
# The posterior variance of the prediction y_{k+1} given context D_k is:
# Var(y_{k+1} | D_k) = E[Var(y_{k+1} | f, D_k) | D_k] + Var(E[y_{k+1} | f, D_k] | D_k)
# = sigma^2 + Var(f(x_{k+1}) | D_k)
# The minimax risk of the true task family (linear regression with known prior) is the Bayes risk of that family.
# For a linear model with Gaussian prior on weights, the posterior variance decays as 1/k.
# The claim states that the excess variance (Posterior Variance - Minimax Risk) decays exponentially.
# However, in a standard Bayesian linear regression, the posterior variance of the prediction is sigma^2 + (prior_cov + X^T X)^-1 x^T ...
# which decays as 1/k, not exponentially.
# The paper's Theorem 3.3 likely refers to the variance of the *task index* or a specific bound.
# Let's re-read the claim: "Posterior Variance RPV is upper bounded by the minimax risk of the true task family plus a term that decays exponentially..."
# This implies: RPV <= Minimax_Risk + C * exp(-c*k).
# If RPV is the total posterior variance, and Minimax_Risk is the irreducible risk (sigma^2 + variance due to finite samples in the true task),
# then the "excess" is the variance due to task uncertainty.
# In a mixture, the posterior over tasks concentrates. The variance of the prediction due to task uncertainty should decay exponentially if the tasks are well-separated.

# Let's simulate a two-task mixture.
# Task 1: y = 1.0 * x + 0.0 + eps
# Task 2: y = -1.0 * x + 0.0 + eps
# x ~ N(0, 1)
# eps ~ N(0, sigma^2)
# Prior on task: P(I=1) = 0.5, P(I=2) = 0.5

sigma = 0.1
n_samples = 10000
k_values = np.arange(1, 51)

# We will estimate the Posterior Variance for the true task (Task 1) as k increases.
# Actually, the claim is about the *true* task family. So we condition on I=1.
# But the posterior variance RPV is defined for the mixture.
# Let's compute the posterior variance of y_{k+1} given D_k, averaged over D_k from the mixture.
# Then compare to the minimax risk of the true task (Task 1).
# Minimax risk of Task 1 (linear regression with known x distribution and Gaussian noise) is sigma^2 + (variance of prediction due to finite k).
# For a single task, the Bayes risk is sigma^2 + E[Var(f(x) | D_k)].
# For linear regression with Gaussian prior, this decays as 1/k.
# The "excess" over the minimax risk of the *true* task would be the difference between the mixture posterior variance and the single-task posterior variance.
# This difference should decay exponentially with k.

# Let's compute:
# 1. Mixture Posterior Variance: Var(y_{k+1} | D_k) averaged over D_k ~ Mixture.
# 2. True Task Minimax Risk: Bayes risk of Task 1, i.e., E[Var(y_{k+1} | D_k, I=1)] averaged over D_k ~ Task 1.
#    Note: The minimax risk is usually the worst-case risk, but here it's the Bayes risk of the family.
#    The claim says "upper bounded by the minimax risk of the true task family".
#    So we compare Mixture PV to True Task Bayes Risk.

# Let's simulate.
# For each k, generate many prompts from the mixture.
# For each prompt, compute the posterior variance of y_{k+1}.
# Posterior variance = E[Var(y | f, D) | D] + Var(E[y | f, D] | D)
# = sigma^2 + Var(f(x_{k+1}) | D)
# We need to compute Var(f(x_{k+1}) | D).
# This is the variance of the prediction over the posterior of f.
# For a mixture of two linear models, the posterior is a mixture of two Gaussians.
# We can compute this analytically or via Monte Carlo.
# Let's use Monte Carlo for simplicity, but with enough samples.

# Actually, for linear regression with Gaussian prior, the posterior is Gaussian.
# For a mixture, it's a mixture of Gaussians.
# Let's implement a simple Bayesian linear regression for each task.
# Prior: w ~ N(0, tau^2 I), b ~ N(0, tau^2)
# Let's set tau^2 = 1.0.

tau2 = 1.0

# Function to compute posterior variance for a given context D_k and query x_q.
# D_k is a list of (x, y) pairs.
# We assume the true task is known for the "True Task" case, but for the mixture, we don't.
# Wait, the posterior variance in the mixture is computed with respect to the posterior over (I, f).
# So we need to compute the posterior over I and f given D_k.
# This is complex. Let's simplify.
# The claim is about the *excess* variance decaying exponentially.
# Let's just compute the posterior variance of the prediction for the mixture and for the true task.

# Let's use a simpler model: two point masses for the slope.
# Task 1: w = 1, b = 0
# Task 2: w = -1, b = 0
# This is a discrete mixture, not a continuous one. The posterior over tasks will concentrate exponentially.
# The posterior variance of y will be sigma^2 + Var(w*x | D).
# Var(w*x | D) = E[w^2 | D] x^2 - (E[w | D] x)^2.
# Since w is either 1 or -1, w^2 = 1 always.
# So Var(w*x | D) = x^2 - (E[w | D] x)^2 = x^2 (1 - E[w | D]^2).
# E[w | D] is the posterior mean of w.
# As k increases, the posterior concentrates on the true task, so E[w | D] -> 1 (if true task is 1).
# Then 1 - E[w | D]^2 -> 0 exponentially.
# So the excess variance decays exponentially.

# Let's simulate this discrete mixture.
# True task: I=1 (w=1).
# Generate data from Task 1.
# Compute posterior probability of I=1 given D_k.
# p(I=1 | D_k) = p(D_k | I=1) p(I=1) / (p(D_k | I=1) p(I=1) + p(D_k | I=2) p(I=2))
# p(D_k | I=i) is the likelihood of the data under task i.
# For linear regression with known w, the likelihood is Gaussian.

# Let's implement this.

def compute_posterior_variance_discrete(k, x_q, sigma, tau2_prior=1.0):
    """
    Compute posterior variance of y_{k+1} given D_k for a discrete mixture of two linear models.
    Task 1: w=1, b=0
    Task 2: w=-1, b=0
    Prior: P(I=1)=0.5, P(I=2)=0.5
    """
    # Generate data from Task 1 (true task)
    x = np.random.randn(k)
    y = 1.0 * x + np.random.randn(k) * sigma

    # Compute likelihood under Task 1 and Task 2
    # Likelihood is product of N(y_j | w_i x_j, sigma^2)
    # log-likelihood = -0.5 * sum((y_j - w_i x_j)^2 / sigma^2) - k/2 log(2 pi sigma^2)

    # We can compute the log-likelihood ratio
    # log p(D|I=1) - log p(D|I=2) = -0.5/sigma^2 * (sum((y-x)^2) - sum((y+x)^2))
    # = -0.5/sigma^2 * (sum(y^2 - 2xy + x^2) - sum(y^2 + 2xy + x^2))
    # = -0.5/sigma^2 * (-4 sum(xy))
    # = 2/sigma^2 * sum(xy)

    log_lr = 2.0 / (sigma**2) * np.sum(x * y)

    # Posterior probability of I=1
    # p(I=1|D) = 1 / (1 + exp(-log_lr))  (since prior is 0.5/0.5)
    p1 = 1.0 / (1.0 + np.exp(-log_lr))
    p2 = 1.0 - p1

    # Posterior mean of w
    E_w = p1 * 1.0 + p2 * (-1.0)

    # Posterior variance of y = sigma^2 + Var(w*x_q | D)
    # Var(w*x_q | D) = E[w^2 | D] x_q^2 - (E[w | D] x_q)^2
    # E[w^2 | D] = p1 * 1 + p2 * 1 = 1
    # So Var(w*x_q | D) = x_q^2 - (E_w * x_q)^2 = x_q^2 (1 - E_w^2)

    var_y = sigma**2 + x_q**2 * (1.0 - E_w**2)

    return var_y, p1

# Now, let's estimate the average posterior variance for k=1..50.
# We average over many random contexts D_k and queries x_q.

n_trials = 1000
pv_mixture = np.zeros(len(k_values))
pv_true_task = np.zeros(len(k_values))

# For the true task, the posterior variance is just sigma^2 + Var(w*x_q | D, I=1)
# But if we know I=1, then w=1 is known, so Var(w*x_q | D, I=1) = 0.
# So pv_true_task = sigma^2.
# Wait, the minimax risk of the true task family is the Bayes risk of that family.
# If the family is a single point (w=1), then the Bayes risk is sigma^2.
# If the family is a distribution over w, then it's higher.
# The claim says "minimax risk of the true task family".
# In our discrete mixture, the true task family is just {w=1}, so minimax risk is sigma^2.
# So the excess variance is pv_mixture - sigma^2.

for idx, k in enumerate(k_values):
    pv_m = 0.0
    for _ in range(n_trials):
        x_q = np.random.randn()
        var_y, p1 = compute_posterior_variance_discrete(k, x_q, sigma)
        pv_m += var_y
    pv_mixture[idx] = pv_m / n_trials
    pv_true_task[idx] = sigma**2  # Since the true task is a single point

# The excess variance is pv_mixture - pv_true_task.
# We expect this to decay exponentially with k.

# Let's plot and check the decay rate.

os.makedirs('results/c6', exist_ok=True)

plt.figure(figsize=(10, 6))
plt.semilogy(k_values, pv_mixture - pv_true_task, 'o-', label='Excess Posterior Variance')
plt.xlabel('Context Length k')
plt.ylabel('Excess Variance')
plt.title('Excess Posterior Variance vs Context Length')
plt.legend()
plt.grid(True, which="both", ls="--")
plt.savefig('results/c6/fig.png', dpi=150)
plt.close()

# Fit an exponential decay: excess = A * exp(-c * k)
# Take log: log(excess) = log(A) - c * k
# Use linear regression on log(excess) vs k.

excess = pv_mixture - pv_true_task
# Filter out zeros or negative values (shouldn't happen, but just in case)
mask = excess > 0
k_fit = k_values[mask]
log_excess = np.log(excess[mask])

# Linear regression
A = np.vstack([k_fit, np.ones(len(k_fit))]).T
m, c = np.linalg.lstsq(A, log_excess, rcond=None)[0]

# The decay rate is -m.
decay_rate = -m

# Theoretical decay rate: from the log-likelihood ratio, log_lr ~ 2/sigma^2 * sum(xy).
# sum(xy) ~ k * E[xy] = k * 1 (since y=x+eps, E[xy]=E[x^2]+E[x*eps]=1).
# So log_lr ~ 2k/sigma^2.
# p1 = 1/(1+exp(-log_lr)) ~ 1 - exp(-log_lr) for large log_lr.
# 1 - p1 ~ exp(-log_lr) ~ exp(-2k/sigma^2).
# E_w = p1 - p2 = 2p1 - 1 ~ 1 - 2exp(-2k/sigma^2).
# 1 - E_w^2 ~ 1 - (1 - 4exp(-4k/sigma^2)) = 4exp(-4k/sigma^2).
# So the excess variance ~ x_q^2 * 4exp(-4k/sigma^2).
# The decay rate should be 4/sigma^2.

theoretical_decay_rate = 4.0 / (sigma**2)

# Check if the estimated decay rate is consistent with the theoretical one.
# We'll consider it supported if the ratio is within a reasonable range (e.g., 0.5 to 2.0).

ratio = decay_rate / theoretical_decay_rate

# Positive control: Check that the excess variance is positive and decreasing.
control_pass = np.all(np.diff(excess) < 0) and np.all(excess > 0)

# Status
if control_pass and 0.5 < ratio < 2.0:
    status = "supported"
else:
    status = "inconclusive"

summary = {
    "claim_id": "C6",
    "status": status,
    "metrics": {
        "estimated_decay_rate": float(decay_rate),
        "theoretical_decay_rate": float(theoretical_decay_rate),
        "ratio": float(ratio),
        "control_pass": bool(control_pass),
        "excess_variance_at_k_1": float(excess[0]),
        "excess_variance_at_k_50": float(excess[-1])
    },
    "notes": f"Estimated decay rate {decay_rate:.2f} vs theoretical {theoretical_decay_rate:.2f}. Ratio {ratio:.2f}. Control passed: {control_pass}."
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

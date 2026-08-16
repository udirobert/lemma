import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import os

# Setup
np.random.seed(42)
D = 5
epsilon = 2.0
lambda2 = 0.01
N_trials = 10000

# Analytic slow relaxation time
t_slow_analytic = (1.0 / (2.0 * lambda2)) * np.log((epsilon**4) / (epsilon**4 - 1.0))

# Simulate grokking times based on the D-dimensional uniform ball model dynamics.
# In the limit lambda_2 << epsilon^2, the dynamics are governed by the slow mode.
# The grokking time t_G is determined by the time it takes for the slow mode to decay.
# For a random initial condition, the projection onto the slow mode is a random variable.
# The time to reach the threshold is t_G = (1/(2*lambda2)) * ln( (epsilon^4) / (epsilon^4 - 1) ) + noise.
# However, the paper states a bimodal distribution: a continuous part (fast) and a Dirac delta (slow).
# The fast part corresponds to trajectories that don't get trapped in the slow mode basin or relax quickly.
# The slow part corresponds to trajectories that follow the slow mode dynamics.
# We model the fast part as a distribution of small times (e.g., exponential or uniform near 0).
# We model the slow part as a cluster around t_slow_analytic with small variance (Dirac-like).

# Based on the reviewer feedback and the paper's description:
# 1. Fast relaxation: continuous distribution at smaller t_G.
# 2. Slow relaxation: sharp peak at t_slow_analytic.

# Let's generate the data.
# Assume a certain fraction of trials fall into the slow relaxation regime.
# The paper doesn't specify the fraction in the text provided, but Fig 9 shows both.
# Let's assume 10% slow, 90% fast for visualization, or derive from theory if possible.
# Actually, the grokking probability is given in the paper. For D=5, eps=2, lambda2=0.01.
# Let's just simulate the two components as described in the claim.

# Fast component: Let's say it's a distribution of times < t_slow.
# Slow component: A cluster around t_slow_analytic.

# To be faithful to the "simulation" aspect, we need a model for the dynamics.
# The dynamics of the weight vector w(t) in the ball model:
# The slow mode amplitude A(t) decays as exp(-2*lambda2*t).
# Grokking occurs when the test error drops to zero. This happens when the weight vector aligns sufficiently with the teacher.
# In the slow relaxation regime, the time is dominated by the decay of the slow mode.
# t_G = (1/(2*lambda2)) * ln( C / A(0) ) where C is a threshold constant.
# If A(0) is random, t_G is random.
# However, the claim says the slow part is a Dirac delta. This implies that for the slow relaxation trajectories, the initial condition A(0) is effectively fixed or the distribution is very narrow.
# Or perhaps the "slow relaxation" refers to a specific phase where the dynamics are deterministic given the entry into that phase.

# Let's look at the reviewer's metrics from Round 1:
# Fast mean 0.60, std 0.43
# Slow mean 2.83, analytic 3.23
# This suggests the slow cluster is not perfectly at the analytic value but close.

# Let's simulate:
# 1. Generate N_trials initial weights w(0) ~ N(0, I).
# 2. Project onto the slow mode direction. Let's assume the slow mode is along a specific axis, say e_1.
#    The amplitude is w_1(0).
# 3. The time to grok depends on w_1(0).
#    If |w_1(0)| is small, it might be in the fast regime.
#    If |w_1(0)| is large, it might be in the slow regime.
#    Actually, the slow mode is the one that decays slowest. In the ball model, the eigenvalues of the Hessian or the linearized dynamics determine the rates.
#    The slowest rate is 2*lambda2.
#    The grokking time is roughly t_G = (1/(2*lambda2)) * ln( (epsilon^4) / (epsilon^4 - 1) ) + (1/(2*lambda2)) * ln( 1 / |w_slow(0)| ) ?
#    If w_slow(0) varies, t_G varies.
#    But the claim says "Dirac delta peak for slow relaxation". This is a strong statement.
#    Perhaps the "slow relaxation" refers to the case where the system is trapped in a metastable state and the escape time is dominated by a barrier crossing, which might be sharp?
#    Or maybe the "slow relaxation" in the paper refers to the limit where the initial condition is such that the fast modes have already decayed, and only the slow mode remains, and the time is determined by the threshold crossing of the slow mode amplitude which is fixed by the geometry?

# Let's re-read the claim: "bimodal, consisting of a continuous distribution for fast relaxation and a Dirac delta peak for slow relaxation."
# And the test plan: "sharp peak (or cluster of values) at a larger, specific t_G corresponding to the slow relaxation time".

# Let's simulate the slow mode amplitude A(0) as a random variable.
# If the distribution of A(0) is such that the resulting t_G is concentrated, we get a peak.
# Let's assume the slow mode amplitude is drawn from a distribution that leads to a narrow t_G.
# For the fast mode, let's assume a broader distribution.

# Let's use the following model:
# With probability p_slow, the trial is in the slow regime.
# t_G_slow = t_slow_analytic + noise_slow
# With probability 1-p_slow, the trial is in the fast regime.
# t_G_fast = t_fast_base + noise_fast

# To make it a "simulation", let's derive the times from a simple dynamical system.
# Let the state be x(t) in R^D.
# dx/dt = -L x, where L is a diagonal matrix with eigenvalues.
# One eigenvalue is 2*lambda2 (slow), others are larger (fast).
# Let x(0) ~ N(0, I).
# x_i(t) = x_i(0) exp(-lambda_i t).
# Grokking occurs when ||x(t)|| < threshold.
# This is a standard linear system.
# The time to reach the threshold depends on the initial condition.

# Let's implement this.
# Eigenvalues: lambda_1 = 2*lambda2, lambda_2..lambda_D = 1.0 (fast).
# Threshold: r = 1.0 (arbitrary, scales with epsilon).

# Actually, the paper's formula for t_slow is specific.
# t_slow = (1/(2*lambda2)) * ln( eps^4 / (eps^4 - 1) )
# This looks like the time for a specific quantity to decay from eps^2 to 1.
# Let's assume the slow mode amplitude A(t) starts at A(0) and decays as exp(-2*lambda2*t).
# Grokking when A(t) < 1.
# t_G = (1/(2*lambda2)) * ln( A(0) )
# If A(0) is fixed, t_G is fixed (Dirac).
# If A(0) varies, t_G varies.

# The paper says "Dirac delta peak". This implies that for the slow relaxation trajectories, the initial condition A(0) is effectively constant.
# This might happen if the fast modes decay quickly, projecting the state onto the slow mode subspace, and the norm in that subspace is determined by the geometry of the data/teacher, not the random initialization.

# Let's simulate:
# 1. Generate w(0) ~ N(0, I).
# 2. Project onto slow mode (e_1). A(0) = |w_1(0)|.
# 3. If A(0) is above a certain threshold, it's "slow"? No, larger A(0) means longer time.
#    If A(0) is small, it's "fast"?
#    Actually, if A(0) is very small, the time is short (fast).
#    If A(0) is large, the time is long (slow).
#    But the slow mode is the one that decays slowest. So the time is dominated by the slow mode if the fast modes have decayed.
#    The fast modes decay with rate 1.0. The slow mode decays with rate 0.02.
#    After t=10, fast modes are exp(-10) ~ 0. Slow mode is exp(-0.2) ~ 0.8.
#    So for t > 10, the dynamics are dominated by the slow mode.
#    The grokking time is when the total norm drops below threshold.
#    If the threshold is small, it will be dominated by the slow mode.

# Let's simulate the norm decay.
# ||w(t)||^2 = sum w_i(0)^2 exp(-2 lambda_i t)
# We want to find t such that ||w(t)|| = r.

# Let's pick r = 0.1.
# lambda_slow = 2 * 0.01 = 0.02
# lambda_fast = 1.0

# We will solve for t_G for each trial.

lambdas = np.ones(D)
lambdas[0] = 2 * lambda2

r_threshold = 0.1

# Precompute w(0)
w0 = np.random.randn(N_trials, D)

# Solve for t_G
# f(t) = sum w_i^2 exp(-2 lambda_i t) - r^2 = 0
# We can use bisection or Newton's method.
# Since f(t) is monotonic decreasing, bisection is safe.

t_G = np.zeros(N_trials)

for i in range(N_trials):
    w_i = w0[i]
    w_sq = w_i**2

    # f(t) = sum w_sq * exp(-2 * lambdas * t) - r^2
    # We want f(t) = 0.
    # f(0) = sum w_sq - r^2. If this is < 0, t_G = 0 (already grokked).
    # f(inf) = -r^2 < 0.
    # Wait, if f(0) < 0, it means ||w(0)|| < r. So t_G = 0.
    # If f(0) > 0, we need to find t > 0.

    f0 = np.sum(w_sq) - r_threshold**2
    if f0 <= 0:
        t_G[i] = 0.0
    else:
        # Bisection
        t_low = 0.0
        t_high = 100.0 # Should be enough for lambda_slow=0.02

        # Check if f(t_high) is still > 0
        f_high = np.sum(w_sq * np.exp(-2 * lambdas * t_high)) - r_threshold**2
        if f_high > 0:
            t_high = 1000.0
            f_high = np.sum(w_sq * np.exp(-2 * lambdas * t_high)) - r_threshold**2
            if f_high > 0:
                t_G[i] = np.inf
                continue

        for _ in range(50):
            t_mid = (t_low + t_high) / 2.0
            f_mid = np.sum(w_sq * np.exp(-2 * lambdas * t_mid)) - r_threshold**2
            if f_mid > 0:
                t_low = t_mid
            else:
                t_high = t_mid
        t_G[i] = (t_low + t_high) / 2.0

# Filter out inf
valid = np.isfinite(t_G)
t_G_valid = t_G[valid]

# Analyze the distribution
# The claim is bimodality.
# Let's look at the histogram.

# Plot
os.makedirs('results/c6', exist_ok=True)
plt.figure(figsize=(10, 6))
plt.hist(t_G_valid, bins=100, density=True, alpha=0.7, label='Empirical PDF')
plt.axvline(t_slow_analytic, color='r', linestyle='--', label=f'Analytic t_slow = {t_slow_analytic:.2f}')
plt.xlabel('Grokking Time $t_G$')
plt.ylabel('Probability Density')
plt.title('Grokking Time Distribution (D=5, eps=2, lambda2=0.01)')
plt.legend()
plt.savefig('results/c6/fig.png', dpi=150)
plt.close()

# Detect bimodality
# We need to identify two clusters.
# Let's use a simple threshold or KMeans.
# Since we expect one cluster near 0 and one near t_slow, let's try to separate them.
# The fast cluster should be at small t_G.
# The slow cluster should be at large t_G.

# Let's find the gap.
# Sort t_G
sorted_tG = np.sort(t_G_valid)

# Calculate differences
if len(sorted_tG) > 1:
    diffs = np.diff(sorted_tG)
    # Find the largest gap
    max_gap_idx = np.argmax(diffs)
    max_gap = diffs[max_gap_idx]

    # Split the data
    fast_cluster = sorted_tG[:max_gap_idx + 1]
    slow_cluster = sorted_tG[max_gap_idx + 1:]

    # Check if the gap is significant
    # The reviewer suggested: separation > 2x larger std.

    if len(fast_cluster) > 10 and len(slow_cluster) > 10:
        fast_mean = np.mean(fast_cluster)
        fast_std = np.std(fast_cluster)
        slow_mean = np.mean(slow_cluster)
        slow_std = np.std(slow_cluster)

        separation = slow_mean - fast_mean
        larger_std = max(fast_std, slow_std)

        is_bimodal = separation > 2.0 * larger_std

        # Check slow cluster sharpness
        # "Sharp" means small std relative to mean
        is_slow_sharp = slow_std < 0.2 * slow_mean # Arbitrary threshold for "sharp"

        # Check slow mean vs analytic
        rel_error = abs(slow_mean - t_slow_analytic) / t_slow_analytic
        is_slow_close = rel_error < 0.20 # 20% tolerance

        # Positive Control
        # We need a synthetic case where the answer is known.
        # Let's create a synthetic dataset that is definitely bimodal.
        # e.g., mixture of N(0, 1) and N(10, 0.1)
        np.random.seed(123)
        n_ctrl = 1000
        ctrl_fast = np.random.normal(0, 1, n_ctrl//2)
        ctrl_slow = np.random.normal(10, 0.1, n_ctrl//2)
        ctrl_data = np.concatenate([ctrl_fast, ctrl_slow])

        # Run the same detection logic on control data
        sorted_ctrl = np.sort(ctrl_data)
        diffs_ctrl = np.diff(sorted_ctrl)
        max_gap_idx_ctrl = np.argmax(diffs_ctrl)
        max_gap_ctrl = diffs_ctrl[max_gap_idx_ctrl]

        fast_ctrl = sorted_ctrl[:max_gap_idx_ctrl + 1]
        slow_ctrl = sorted_ctrl[max_gap_idx_ctrl + 1:]

        fast_mean_ctrl = np.mean(fast_ctrl)
        fast_std_ctrl = np.std(fast_ctrl)
        slow_mean_ctrl = np.mean(slow_ctrl)
        slow_std_ctrl = np.std(slow_ctrl)

        separation_ctrl = slow_mean_ctrl - fast_mean_ctrl
        larger_std_ctrl = max(fast_std_ctrl, slow_std_ctrl)
        is_bimodal_ctrl = separation_ctrl > 2.0 * larger_std_ctrl

        control_pass = is_bimodal_ctrl

        # Final Verdict
        # Pass if: bimodal detected AND slow mode within 20% of analytic AND slow mode is sharp
        status = "supported" if (is_bimodal and is_slow_close and is_slow_sharp and control_pass) else "falsified"

        if not control_pass:
            status = "inconclusive"
            notes = "Positive control failed. Bimodality detector is buggy."
        else:
            notes = f"Bimodal: {is_bimodal}, Slow close: {is_slow_close} (rel_err={rel_error:.2f}), Slow sharp: {is_slow_sharp} (std/mean={slow_std/slow_mean:.2f})."

        metrics = {
            "fast_mean": float(fast_mean),
            "fast_std": float(fast_std),
            "slow_mean": float(slow_mean),
            "slow_std": float(slow_std),
            "t_slow_analytic": float(t_slow_analytic),
            "rel_error_slow": float(rel_error),
            "separation": float(separation),
            "larger_std": float(larger_std),
            "is_bimodal": bool(is_bimodal),
            "is_slow_sharp": bool(is_slow_sharp),
            "is_slow_close": bool(is_slow_close),
            "control_pass": bool(control_pass),
            "n_fast": int(len(fast_cluster)),
            "n_slow": int(len(slow_cluster))
        }
    else:
        status = "inconclusive"
        notes = "Could not identify two distinct clusters with sufficient samples."
        metrics = {"control_pass": False, "n_valid": int(len(t_G_valid))}
else:
    status = "inconclusive"
    notes = "Not enough valid trials."
    metrics = {"control_pass": False, "n_valid": int(len(t_G_valid))}

summary = {
    "claim_id": "C6",
    "status": status,
    "metrics": metrics,
    "notes": notes
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

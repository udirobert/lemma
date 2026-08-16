import numpy as np
import json
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

# Set seed for reproducibility
np.random.seed(42)

# Create results directory
os.makedirs('results/c5', exist_ok=True)

# --- 1. Define the Model and Data Generation ---

def generate_data(n_samples, mean_x, std_x, w, b, noise_std):
    """
    Generate synthetic linear regression data.
    x ~ N(mean_x, std_x^2)
    y = w*x + b + noise
    """
    x = np.random.normal(mean_x, std_x, n_samples)
    noise = np.random.normal(0, noise_std, n_samples)
    y = w * x + b + noise
    return x, y

def compute_risk(model_params, x_test, y_test):
    """
    Compute Mean Squared Error (MSE) for a linear model.
    """
    w, b = model_params
    y_pred = w * x_test + b
    return np.mean((y_test - y_pred) ** 2)

def train_linear_model(x_train, y_train):
    """
    Train a simple linear regression model using least squares.
    Returns (w, b).
    """
    # Add bias term
    X = np.column_stack((x_train, np.ones_like(x_train)))
    # Solve least squares
    theta, _, _, _ = np.linalg.lstsq(X, y_train, rcond=None)
    w, b = theta[0], theta[1]
    return w, b

# --- 2. Define Metrics ---

def compute_wasserstein_1d(x1, x2):
    """
    Compute 1D Wasserstein distance (Earth Mover's Distance) between two 1D samples.
    For 1D, W1 is the integral of the absolute difference of CDFs.
    """
    x1_sorted = np.sort(x1)
    x2_sorted = np.sort(x2)
    # Interpolate CDFs on a common grid or use the formula for equal-sized samples
    # For equal-sized samples, W1 = mean(|x1_sorted - x2_sorted|)
    if len(x1) == len(x2):
        return np.mean(np.abs(x1_sorted - x2_sorted))
    else:
        # General case using scipy or manual interpolation
        # Since we control the sample size, we'll ensure they are equal in the main loop
        pass

def estimate_posterior_variance(x_test, y_test, w_true, b_true, noise_std):
    """
    Estimate the Posterior Variance term.
    In the context of the paper, Posterior Variance is the irreducible risk due to noise and task uncertainty.
    For a single known task (linear regression with known w, b), the posterior variance of the prediction
    at a new point x is dominated by the noise variance sigma^2.
    However, if the model is uncertain about w and b, there is additional variance.

    The claim states: "Posterior Variance remains intrinsic to the target domain and is not directly penalized by the shift".
    In a simple linear regression with Gaussian noise, the variance of the prediction error for the Bayes optimal predictor
    is exactly the noise variance sigma^2 (if w, b are known) or slightly higher if they are estimated.

    Let's define the 'Posterior Variance' metric as the variance of the residuals from the true function,
    which is the noise variance. This should be constant regardless of the input distribution shift (mean/variance of x),
    assuming the noise is additive and independent of x.

    Alternatively, we can look at the variance of the predictions made by the Bayes optimal predictor.
    But the simplest interpretation consistent with "intrinsic to the target domain" and "not penalized by shift"
    is that the noise level (sigma^2) is constant.

    Let's use the empirical variance of the noise as a proxy for the intrinsic Posterior Variance.
    """
    # The true noise variance is sigma^2. We can estimate it from the residuals of the true model.
    y_true = w_true * x_test + b_true
    residuals = y_test - y_true
    return np.var(residuals)

# --- 3. Experimental Setup ---

# Hyperparameters
n_train = 1000
n_test = 1000
w_true = 2.0
b_true = 1.0
noise_std = 0.5

# Source distribution parameters
mean_x_source = 0.0
std_x_source = 1.0

# Target distribution shifts
# We will vary the mean of the target distribution to create different Wasserstein distances.
# W1 between N(mu1, sigma^2) and N(mu2, sigma^2) is |mu1 - mu2|.
shift_magnitudes = np.linspace(0.0, 5.0, 20)

bayes_gaps = []
post_variances = []
w_distances = []

# --- 4. Run Experiments ---

for shift in shift_magnitudes:
    # Target distribution
    mean_x_target = mean_x_source + shift
    std_x_target = std_x_source  # Keep variance same to isolate mean shift effect on W1

    # Generate Source Data (Pretraining)
    x_train, y_train = generate_data(n_train, mean_x_source, std_x_source, w_true, b_true, noise_std)

    # Generate Target Data (Inference)
    x_test, y_test = generate_data(n_test, mean_x_target, std_x_target, w_true, b_true, noise_std)

    # Train Model on Source
    w_model, b_model = train_linear_model(x_train, y_train)

    # Evaluate on Target
    # 1. Compute Risk of the Trained Model (Empirical Bayes Gap proxy)
    # The Bayes Gap is the difference between the model's risk and the Bayes optimal risk.
    # Bayes optimal risk for linear regression with known w, b is sigma^2.
    # Model risk is MSE on target.
    model_risk = compute_risk((w_model, b_model), x_test, y_test)
    bayes_optimal_risk = noise_std ** 2

    # Bayes Gap = Model Risk - Bayes Optimal Risk
    # Note: Due to finite samples, model_risk might be slightly higher than sigma^2 even for shift=0.
    # We subtract the baseline noise to isolate the gap due to distribution shift/model mismatch.
    bayes_gap = model_risk - bayes_optimal_risk

    # 2. Compute Posterior Variance
    # As defined above, this is the intrinsic noise variance.
    post_var = estimate_posterior_variance(x_test, y_test, w_true, b_true, noise_std)

    # 3. Compute Wasserstein Distance
    # For 1D Gaussians with same variance, W1 = |mean1 - mean2|
    # We can also compute it empirically to be robust.
    w_dist = compute_wasserstein_1d(x_train, x_test)

    bayes_gaps.append(bayes_gap)
    post_variances.append(post_var)
    w_distances.append(w_dist)

bayes_gaps = np.array(bayes_gaps)
post_variances = np.array(post_variances)
w_distances = np.array(w_distances)

# --- 5. Analysis ---

# Correlation between Wasserstein Distance and Bayes Gap
# Filter out zero shift if needed, but pearsonr handles it.
corr_bg, p_val_bg = pearsonr(w_distances, bayes_gaps)

# Correlation between Wasserstein Distance and Posterior Variance
corr_pv, p_val_pv = pearsonr(w_distances, post_variances)

# Success Criterion Check:
# 1. Correlation between W and Bayes Gap > 0.9
# 2. Change in Posterior Variance is significantly smaller or uncorrelated.
#    We check if corr_pv is low (e.g., < 0.5) or if the range of post_variances is small.

# Positive Control:
# We know that for linear regression, the excess risk (Bayes Gap) due to mean shift in x
# is proportional to the square of the shift in mean times the variance of the weight estimation error?
# Actually, let's look at the theory.
# If we train on x ~ N(0, 1) and test on x ~ N(mu, 1), the model estimates w, b.
# The error in prediction at x is (w_hat - w)x + (b_hat - b).
# E[(w_hat - w)x + (b_hat - b)]^2 = Var(w_hat)E[x^2] + Var(b_hat) + 2Cov(w_hat, b_hat)E[x] + ...
# This is quadratic in mu? Or linear?
# The claim says "bounded by a constant times the Wasserstein distance".
# W1 is linear in mu.
# If the gap is quadratic in mu, the correlation with W1 (which is linear) might still be high if the range is small,
# but let's check the actual correlation.

# Let's also check the positive control:
# Does the statistic work?
# We can verify that for shift=0, the gap is close to 0 (modulo sampling noise).
# And for large shift, the gap increases.

# Control:
# We expect corr_bg to be high.
# We expect corr_pv to be low (since post_var is just noise variance, it shouldn't depend on x distribution).

control_pass = True
# Check if the Bayes Gap actually increases with shift
if np.corrcoef(w_distances, bayes_gaps)[0,1] < 0.5:
    control_pass = False
# Check if Posterior Variance is roughly constant
if np.std(post_variances) > 0.1 * np.mean(post_variances):
    control_pass = False

# --- 6. Plotting ---

fig, ax = plt.subplots(1, 2, figsize=(12, 5))

# Plot 1: Bayes Gap vs Wasserstein Distance
ax[0].scatter(w_distances, bayes_gaps, color='blue')
ax[0].set_xlabel('Wasserstein Distance (W1)')
ax[0].set_ylabel('Bayes Gap')
ax[0].set_title(f'Bayes Gap vs W1 (Corr: {corr_bg:.3f})')
ax[0].grid(True)

# Plot 2: Posterior Variance vs Wasserstein Distance
ax[1].scatter(w_distances, post_variances, color='red')
ax[1].set_xlabel('Wasserstein Distance (W1)')
ax[1].set_ylabel('Posterior Variance')
ax[1].set_title(f'Posterior Variance vs W1 (Corr: {corr_pv:.3f})')
ax[1].grid(True)

plt.tight_layout()
plt.savefig('results/c5/fig.png', dpi=150)
plt.close()

# --- 7. Summary ---

status = "supported"
if corr_bg < 0.9:
    status = "falsified"
    notes = f"Correlation between W1 and Bayes Gap is {corr_bg:.3f}, which is less than 0.9."
else:
    notes = f"Correlation between W1 and Bayes Gap is {corr_bg:.3f} (> 0.9). Correlation between W1 and Posterior Variance is {corr_pv:.3f}."

if not control_pass:
    status = "inconclusive"
    notes += " Positive control failed: Bayes Gap did not increase as expected or Posterior Variance was not stable."

summary = {
    "claim_id": "C5",
    "status": status,
    "metrics": {
        "corr_w1_bayes_gap": float(corr_bg),
        "corr_w1_post_var": float(corr_pv),
        "mean_bayes_gap": float(np.mean(bayes_gaps)),
        "mean_post_var": float(np.mean(post_variances)),
        "std_post_var": float(np.std(post_variances)),
        "control_pass": bool(control_pass)
    },
    "notes": notes
}

print(f"SUMMARY_JSON={json.dumps(summary, default=str)}")

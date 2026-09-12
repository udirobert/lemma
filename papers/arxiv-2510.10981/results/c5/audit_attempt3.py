import json
import numpy as np
from scipy import stats

try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    HAVE_MPL = True
except Exception:
    HAVE_MPL = False

# Audit C5: Wasserstein stability of Bayes Gap under input-distribution shift.
# Setup: linear regression task family, source inputs x ~ N(0,1), target x ~ N(mu,1).
# Model: OLS fit on source-domain training data (proxy for pretrained model).
# Bayes Gap = excess risk of OLS predictor over Bayes-optimal predictor on target domain.
# Posterior Variance proxy = irreducible noise variance estimated on target domain
#   (intrinsic to target task; should not depend on shift).
# W1 between N(0,1) and N(mu,1) = |mu| (analytic).

np.random.seed(0)

n_train = 200          # smaller train set -> larger estimation error -> clearer signal
n_test = 1000
noise_std = 0.5
R = 200                # replicates per shift value
shifts = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])

def run_replicate(mu, seed):
    rng = np.random.default_rng(seed)
    # source training data
    Xtr = rng.normal(0.0, 1.0, n_train)
    w_true = rng.normal(0.0, 1.0)
    b_true = rng.normal(0.0, 1.0)
    ytr = w_true * Xtr + b_true + rng.normal(0.0, noise_std, n_train)
    # OLS with intercept
    A = np.column_stack([Xtr, np.ones(n_train)])
    coef, _, _, _ = np.linalg.lstsq(A, ytr, rcond=None)
    w_hat, b_hat = coef
    # target test data
    Xte = rng.normal(mu, 1.0, n_test)
    yte = w_true * Xte + b_true + rng.normal(0.0, noise_std, n_test)
    # model risk on target
    pred = w_hat * Xte + b_hat
    risk_model = np.mean((yte - pred) ** 2)
    # Bayes-optimal predictor risk on target = noise variance (true model known to Bayes)
    pred_bayes = w_true * Xte + b_true
    risk_bayes = np.mean((yte - pred_bayes) ** 2)
    bayes_gap = risk_model - risk_bayes
    # Posterior variance proxy: residual variance under the true (Bayes) model on target
    post_var = risk_bayes
    return bayes_gap, post_var

mean_gap = np.zeros(len(shifts))
se_gap = np.zeros(len(shifts))
mean_pv = np.zeros(len(shifts))
std_pv = np.zeros(len(shifts))

for i, mu in enumerate(shifts):
    gaps = np.zeros(R)
    pvs = np.zeros(R)
    for r in range(R):
        g, pv = run_replicate(mu, seed=1000 * i + r + 7)
        gaps[r] = g
        pvs[r] = pv
    mean_gap[i] = gaps.mean()
    se_gap[i] = gaps.std(ddof=1) / np.sqrt(R)
    mean_pv[i] = pvs.mean()
    std_pv[i] = pvs.std(ddof=1)

W1 = np.abs(shifts)  # analytic W1 between equal-variance Gaussians

# Change in Bayes Gap relative to shift-0 baseline
delta_gap = mean_gap - mean_gap[0]

# Primary statistic: Pearson corr between W1 and delta_gap (exclude shift=0 baseline point? keep all)
corr_w1_dg, p_w1_dg = stats.pearsonr(W1, delta_gap)
spearman_w1_dg, _ = stats.spearmanr(W1, delta_gap)
corr_w1_pv, p_w1_pv = stats.pearsonr(W1, mean_pv)

# Posterior variance stability: coefficient of variation
pv_cv = float(std_pv.mean() / mean_pv.mean())
pv_rel_change = float((mean_pv.max() - mean_pv.min()) / mean_pv.mean())

# Positive control:
#  (a) delta_gap at max shift positive and >> its SE
#  (b) delta_gap monotone increasing (Spearman > 0.9)
#  (c) posterior variance CV < 5%
delta_gap_max = float(delta_gap[-1])
se_at_max = float(np.sqrt(se_gap[-1] ** 2 + se_gap[0] ** 2))
control_a = delta_gap_max > 5 * se_at_max and delta_gap_max > 0
control_b = float(spearman_w1_dg) > 0.9
control_c = pv_cv < 0.05
control_pass = bool(control_a and control_b and control_c)

# Diagnostic: exponent of growth (log-log slope on positive shifts)
mask = W1 > 0
slope, _, _, _, _ = stats.linregress(np.log(W1[mask]), np.log(np.maximum(delta_gap[mask], 1e-12)))

# Success criterion: corr(W1, delta_gap) > 0.9 and posterior variance not comparably dependent
pass_corr = float(corr_w1_dg) > 0.9
pass_pv = abs(float(corr_w1_pv)) < 0.9 or pv_rel_change < 0.1

if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed (control_a=%s, control_b=%s, control_c=%s); statistic unreliable.' % (control_a, control_b, control_c)
elif pass_corr and pass_pv:
    status = 'supported'
    notes = 'Delta Bayes Gap tracks W1 (corr=%.3f) while posterior variance is shift-invariant (corr=%.3f, CV=%.3f).' % (corr_w1_dg, corr_w1_pv, pv_cv)
else:
    status = 'falsified'
    notes = 'corr(W1, delta_gap)=%.3f (growth exponent ~%.2f); posterior-var corr=%.3f. Linear-bound reading not confirmed.' % (corr_w1_dg, slope, corr_w1_pv)

if HAVE_MPL:
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].errorbar(W1, delta_gap, yerr=2 * se_gap, fmt='o-')
    axes[0].set_xlabel('W1 (|mu shift|)')
    axes[0].set_ylabel('Delta Bayes Gap')
    axes[0].set_title('corr=%.3f' % corr_w1_dg)
    axes[1].errorbar(W1, mean_pv, yerr=2 * std_pv / np.sqrt(R), fmt='s-', color='darkred')
    axes[1].set_xlabel('W1 (|mu shift|)')
    axes[1].set_ylabel('Posterior Variance (proxy)')
    axes[1].set_title('corr=%.3f, CV=%.3f' % (corr_w1_pv, pv_cv))
    fig.tight_layout()
    fig.savefig('results/c5/c5_wasserstein_stability.png', dpi=100)
    plt.close(fig)

summary = {
    'claim_id': 'C5',
    'status': status,
    'metrics': {
        'corr_w1_delta_gap': float(corr_w1_dg),
        'p_w1_delta_gap': float(p_w1_dg),
        'spearman_w1_delta_gap': float(spearman_w1_dg),
        'corr_w1_post_var': float(corr_w1_pv),
        'p_w1_post_var': float(p_w1_pv),
        'delta_gap_max': delta_gap_max,
        'delta_gap_max_se': se_at_max,
        'delta_gap_min': float(delta_gap.min()),
        'post_var_mean': float(mean_pv.mean()),
        'post_var_cv': pv_cv,
        'post_var_rel_change': pv_rel_change,
        'growth_exponent_loglog': float(slope),
        'n_replicates': R,
        'control_pass': control_pass,
    },
    'notes': notes,
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))

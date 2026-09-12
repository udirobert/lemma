import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

# ------------------------------------------------------------
# Audit C5: Wasserstein stability of the Bayes Gap under input shift.
# Setup: linear regression task family, source inputs x ~ N(0,1),
# target inputs x ~ N(mu,1). Model = OLS fit on source data.
# Bayes Gap ~ excess risk of the trained model over the Bayes-optimal
# predictor on the target domain. Posterior Variance ~ intrinsic noise
# variance of the target task (model-independent), proxied by the
# residual variance under the true model on the target domain.
# W1 between N(0,1) and N(mu,1) (equal variances) = |mu|.
# ------------------------------------------------------------

np.random.seed(0)

n_train = 200          # small enough that estimation error (Bayes Gap) is visible
n_test = 2000
noise_std = 0.5
R = 200                # replicates per shift value (noise suppression)
shifts = np.array([0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0])

def run_replicates(mu, R, seed0):
    gaps = np.zeros(R)
    pvars = np.zeros(R)
    for r in range(R):
        rng = np.random.default_rng(seed0 + r)
        # source training data
        x_tr = rng.normal(0.0, 1.0, n_train)
        w_true = rng.normal(0.0, 1.0)
        b_true = rng.normal(0.0, 1.0)
        y_tr = w_true * x_tr + b_true + rng.normal(0.0, noise_std, n_train)
        # OLS fit (with intercept)
        A = np.column_stack([x_tr, np.ones(n_train)])
        coef, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
        w_hat, b_hat = coef
        # target test data
        x_te = rng.normal(mu, 1.0, n_test)
        y_te = w_true * x_te + b_true + rng.normal(0.0, noise_std, n_test)
        # model risk and Bayes-optimal risk on target
        mse_model = np.mean((y_te - (w_hat * x_te + b_hat))**2)
        mse_bayes = np.mean((y_te - (w_true * x_te + b_true))**2)
        gaps[r] = mse_model - mse_bayes
        # Posterior Variance proxy: intrinsic target-domain uncertainty
        # (residual variance under the true model) -- model independent.
        pvars[r] = mse_bayes
    return gaps, pvars

mean_gap = np.zeros(len(shifts))
se_gap = np.zeros(len(shifts))
mean_pv = np.zeros(len(shifts))
se_pv = np.zeros(len(shifts))
for i, mu in enumerate(shifts):
    g, pv = run_replicates(mu, R, seed0=1000 + 10000 * i)
    mean_gap[i] = g.mean()
    se_gap[i] = g.std(ddof=1) / np.sqrt(R)
    mean_pv[i] = pv.mean()
    se_pv[i] = pv.std(ddof=1) / np.sqrt(R)

# Change in Bayes Gap relative to shift-0 baseline
baseline = mean_gap[0]
delta_gap = mean_gap - baseline
# SE of delta (independent replicates across shifts)
se_delta = np.sqrt(se_gap**2 + se_gap[0]**2)

# Analytic W1 for equal-variance 1D Gaussians
W1 = np.abs(shifts - 0.0)

# Empirical 1D W1 as sanity check (sorted-sample L1)
def emp_w1(mu, n=4000, seed=7):
    rng = np.random.default_rng(seed)
    a = np.sort(rng.normal(0.0, 1.0, n))
    b = np.sort(rng.normal(mu, 1.0, n))
    return float(np.mean(np.abs(a - b)))
W1_emp = np.array([emp_w1(mu) for mu in shifts])

# Correlations (exclude the zero-shift point for Pearson on W1>0? keep all; W1=0, delta=0 is a valid point)
mask = W1 > 0
pearson_r, pearson_p = stats.pearsonr(W1[mask], delta_gap[mask])
spearman_r, _ = stats.spearmanr(W1[mask], delta_gap[mask])
corr_w1_pv, corr_w1_pv_p = stats.pearsonr(W1[mask], mean_pv[mask])

# Posterior variance stability: coefficient of variation across shifts
pv_cv = float(mean_pv.std() / mean_pv.mean())
pv_range_rel = float((mean_pv.max() - mean_pv.min()) / mean_pv.mean())

# Diagnostic: exponent of delta_gap growth (log-log fit)
slope, intercept, *_ = stats.linregress(np.log(W1[mask]), np.log(np.maximum(delta_gap[mask], 1e-12)))

# Positive control checks
control_delta_max = float(delta_gap[-1])
control_delta_max_se = float(se_delta[-1])
control_monotone_spearman = float(stats.spearmanr(W1, delta_gap)[0])
control_pass = bool(
    (control_delta_max > 5 * control_delta_max_se)
    and (control_monotone_spearman > 0.9)
    and (pv_cv < 0.05)
)

# Success criterion
crit_gap = pearson_r > 0.9
crit_pv = (abs(corr_w1_pv) < 0.5) or (pv_range_rel < 0.1 * (delta_gap[-1] / max(abs(baseline), 1e-12)))

if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed; statistic unreliable.'
elif crit_gap and crit_pv:
    status = 'supported'
    notes = 'Delta Bayes Gap strongly correlated with W1 (r=%.3f); posterior variance stable (CV=%.4f, corr=%.3f).' % (pearson_r, pv_cv, corr_w1_pv)
elif not crit_gap:
    status = 'falsified'
    notes = 'Delta Bayes Gap vs W1 correlation %.3f < 0.9 (growth exponent ~%.2f).' % (pearson_r, slope)
else:
    status = 'falsified'
    notes = 'Posterior variance shows shift dependence (corr=%.3f, CV=%.4f).' % (corr_w1_pv, pv_cv)

# Plots
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
axes[0].errorbar(W1, delta_gap, yerr=2 * se_delta, fmt='o-', capsize=3)
axes[0].set_xlabel('W1(P_X, Q_X) = |mu|')
axes[0].set_ylabel('Delta Bayes Gap')
axes[0].set_title('Change in Bayes Gap vs W1 (r=%.3f)' % pearson_r)
axes[1].errorbar(W1, mean_pv, yerr=2 * se_pv, fmt='s-', color='darkred', capsize=3)
axes[1].set_xlabel('W1(P_X, Q_X) = |mu|')
axes[1].set_ylabel('Posterior Variance (proxy)')
axes[1].set_title('Posterior Variance vs shift (corr=%.3f)' % corr_w1_pv)
plt.tight_layout()
plt.savefig('results/c5/fig.png', dpi=120)

metrics = {
    'corr_w1_delta_gap': float(pearson_r),
    'corr_w1_delta_gap_pvalue': float(pearson_p),
    'spearman_w1_delta_gap': float(spearman_r),
    'corr_w1_post_var': float(corr_w1_pv),
    'corr_w1_post_var_pvalue': float(corr_w1_pv_p),
    'post_var_cv': pv_cv,
    'post_var_rel_range': pv_range_rel,
    'delta_gap_min': float(delta_gap.min()),
    'delta_gap_max': control_delta_max,
    'delta_gap_max_se': control_delta_max_se,
    'baseline_gap': float(baseline),
    'growth_exponent_loglog': float(slope),
    'n_replicates': R,
    'w1_empirical_max_dev': float(np.max(np.abs(W1_emp - W1))),
    'control_pass': control_pass,
}
summary = {'claim_id': 'C5', 'status': status, 'metrics': metrics, 'notes': notes}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))

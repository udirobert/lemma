import os, json
import numpy as np
from scipy import integrate
from scipy.stats import chi2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.makedirs('results/c6', exist_ok=True)
np.random.seed(42)


def minimax_risk_scalar(R, sigma, k):
    """E_{S~chi2_k}[ sigma^2 R^2 / (R^2 S + sigma^2) ]."""
    a = R * R
    b = sigma * sigma
    def integrand(s):
        return chi2.pdf(s, k) * (b * a / (a * s + b))
    val, _ = integrate.quad(integrand, 0.0, np.inf, limit=500)
    return float(val)


def rpv_mixture(R1, R2, sigma, alpha, K, M, seed):
    rng = np.random.default_rng(seed)
    ks = np.arange(1, K + 1)
    out = np.zeros(K)
    hyp_R = np.array([R1, R1, R2, R2], dtype=float)
    hyp_sign = np.array([1.0, -1.0, 1.0, -1.0])
    hyp_w = hyp_sign * hyp_R
    hyp_w2 = hyp_R * hyp_R
    log_alpha = np.array([
        np.log(alpha[0]) + np.log(0.5),
        np.log(alpha[0]) + np.log(0.5),
        np.log(alpha[1]) + np.log(0.5),
        np.log(alpha[1]) + np.log(0.5),
    ])
    for idx, k in enumerate(ks):
        I = rng.choice(2, size=M, p=alpha)
        s = rng.choice(np.array([-1.0, 1.0]), size=M)
        Rtrue = np.where(I == 0, R1, R2)
        w = s * Rtrue
        x = rng.normal(size=(M, k))
        eps = rng.normal(scale=sigma, size=(M, k))
        y = w[:, None] * x + eps
        T = np.sum(x * y, axis=1)
        S = np.sum(x * x, axis=1)
        logw = np.zeros((M, 4))
        for h in range(4):
            R = hyp_R[h]
            sign = hyp_sign[h]
            logw[:, h] = (log_alpha[h]
                          + sign * R * T / (sigma * sigma)
                          - (R * R * S) / (2.0 * sigma * sigma))
        m = np.max(logw, axis=1, keepdims=True)
        post = np.exp(logw - m)
        post /= np.sum(post, axis=1, keepdims=True)
        mean_w = post @ hyp_w
        mean_w2 = post @ hyp_w2
        var_w = mean_w2 - mean_w * mean_w
        out[idx] = float(np.mean(var_w))
    return ks, out


def kl_family(Ri, Rj, sigma, M, seed):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=M)
    s = rng.choice(np.array([-1.0, 1.0]), size=M)
    w = s * Ri
    eps = rng.normal(scale=sigma, size=M)
    y = w * x + eps

    def logp(R, x, y):
        a = -0.5 * ((y - R * x) / sigma) ** 2 - np.log(sigma * np.sqrt(2.0 * np.pi))
        b = -0.5 * ((y + R * x) / sigma) ** 2 - np.log(sigma * np.sqrt(2.0 * np.pi))
        return np.logaddexp(a, b) - np.log(2.0)

    logpi = logp(Ri, x, y)
    logpj = logp(Rj, x, y)
    return float(np.mean(logpi - logpj))


def fit_exponential(ks, excess):
    mask = excess > 0
    if np.sum(mask) < 3:
        return float('nan'), float('nan')
    x = ks[mask].astype(float)
    y = np.log(excess[mask])
    A = np.vstack([np.ones_like(x), x]).T
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    intercept, slope = coef
    pred = intercept + slope * x
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float('nan')
    return float(-slope), float(r2)


# Main experiment
R1, R2, sigma = 1.0, 3.0, 1.0
alpha = np.array([0.5, 0.5])
K, M = 12, 15000
ks, rpv = rpv_mixture(R1, R2, sigma, alpha, K, M, 12345)
baseline = 0.5 * np.array([minimax_risk_scalar(R1, sigma, k) for k in ks]) + \
           0.5 * np.array([minimax_risk_scalar(R2, sigma, k) for k in ks])
excess = rpv - baseline
lambda_fit, r2_fit = fit_exponential(ks, excess)
monotone = bool(np.all(np.diff(excess) < 0))

kl12 = kl_family(R1, R2, sigma, 200000, 54321)
kl21 = kl_family(R2, R1, sigma, 200000, 98765)
d_min = float(min(kl12, kl21))

# Positive control: more separated tasks
cR1, cR2 = 0.5, 4.0
cK, cM = 12, 15000
cks, crpv = rpv_mixture(cR1, cR2, sigma, alpha, cK, cM, 22222)
cbaseline = 0.5 * np.array([minimax_risk_scalar(cR1, sigma, k) for k in cks]) + \
            0.5 * np.array([minimax_risk_scalar(cR2, sigma, k) for k in cks])
cexcess = crpv - cbaseline
control_lambda, control_r2 = fit_exponential(cks, cexcess)
control_pass = bool(control_r2 > 0.8 and control_lambda > 0.0 and np.all(cexcess > 0))

# Plot
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(ks, rpv, 'o-', label='RPV (mixture)')
ax.plot(ks, baseline, 's-', label='Avg. minimax risk')
ax.plot(ks, excess, '^-', label='Excess')
if np.isfinite(lambda_fit) and np.isfinite(r2_fit):
    ax.plot(ks, np.exp(np.log(excess[excess > 0][0]) - lambda_fit * (ks - ks[excess > 0][0])),
            '--', label=f'exp fit (lambda={lambda_fit:.3f})')
ax.set_xlabel('context length k')
ax.set_ylabel('risk')
ax.set_title('C6: Posterior variance vs minimax risk')
ax.legend()
ax.grid(True, alpha=0.3)
fig.tight_layout()
fig.savefig('results/c6/fig.png', dpi=150)
plt.close(fig)

if not control_pass:
    status = 'inconclusive'
elif r2_fit > 0.7 and lambda_fit > 0.0 and monotone:
    status = 'supported'
else:
    status = 'falsified'

summary = {
    'claim_id': 'C6',
    'status': status,
    'metrics': {
        'r2_excess_fit': r2_fit,
        'lambda_fit': lambda_fit,
        'd_min': d_min,
        'd_min_over_2': d_min / 2.0,
        'monotone_decrease': monotone,
        'excess_k1': float(excess[0]),
        'excess_k_last': float(excess[-1]),
        'control_r2': control_r2,
        'control_lambda': control_lambda,
        'control_pass': control_pass,
    },
    'notes': 'Two-task scalar linear mixture; RPV by posterior over task/sign hypotheses; minimax risk via chi-square expectation; excess fitted to exponential.'
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))

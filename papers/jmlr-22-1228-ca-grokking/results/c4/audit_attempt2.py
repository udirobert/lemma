import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(12345)
os.makedirs('results/c4', exist_ok=True)

# ------------------------------------------------------------
# D-dimensional uniform ball grokking model (Zunkovic & Ilievski, Sec 3.3)
# Two unit balls in R^D, centers separated by eps along first axis.
# Perceptron f(x) = w.x + b trained by gradient descent on squared hinge
# loss with pure L2 regularisation (lambda1 = 0, lambda2 = 0.1).
# Grok = train error reaches 0, then test error reaches 0 within budget.
# Test error = 0 iff hyperplane separates the *balls*: y(w.c +/- b) > |w|.
# ------------------------------------------------------------

LAM2 = 0.1
N = 10          # samples per class
LR = 0.2
T = 8000        # gradient descent step budget
NTRIALS = 2000
D_LIST = [1, 2, 3, 4, 5, 6]

def sample_ball(n, D, rng):
    g = rng.standard_normal((n, D))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    r = rng.random((n, 1)) ** (1.0 / D)
    return g * r

def run_sweep(D_list, eps, ntrials, rng):
    """Returns array of grokking probabilities for each D."""
    probs = []
    for D in D_list:
        # sample data for all trials at once
        Xp = sample_ball(ntrials * N, D, rng).reshape(ntrials, N, D)
        Xm = sample_ball(ntrials * N, D, rng).reshape(ntrials, N, D)
        cp = np.zeros(D); cp[0] = eps / 2.0
        cm = np.zeros(D); cm[0] = -eps / 2.0
        Xp = Xp + cp
        Xm = Xm + cm
        X = np.concatenate([Xp, Xm], axis=1)          # (trials, 2N, D)
        y = np.concatenate([np.ones((ntrials, N)), -np.ones((ntrials, N))], axis=1)

        w = np.zeros((ntrials, D))
        b = np.zeros(ntrials)
        t_train = np.full(ntrials, -1)
        t_test = np.full(ntrials, -1)
        for t in range(T):
            f = np.einsum('tnd,td->tn', X, w) + b[:, None]
            margin = y * f
            # train error zero?
            tr_ok = np.all(margin > 0, axis=1)
            newly = tr_ok & (t_train < 0)
            t_train[newly] = t
            # test error zero? (separation of full balls)
            wn = np.linalg.norm(w, axis=1)
            fp = w @ cp + b
            fm = w @ cm + b
            te_ok = (fp - wn > 0) & (-fm - wn > 0)
            newly = te_ok & (t_test < 0)
            t_test[newly] = t
            # gradient of squared hinge loss + L2
            viol = np.maximum(0.0, 1.0 - margin)
            coeff = -y * viol / N   # d/df of mean over 2N of (1-yf)^2
            gw = np.einsum('tn,tnd->td', coeff, X) + LAM2 * w
            gb = coeff.sum(axis=1)
            w -= LR * gw
            b -= LR * gb
        grok = (t_train >= 0) & (t_test >= 0) & (t_test >= t_train)
        probs.append(float(np.mean(grok)))
    return np.array(probs)

def fit_loglinear(Ds, Ps):
    Ds = np.asarray(Ds, float); Ps = np.asarray(Ps, float)
    mask = Ps > 0
    if mask.sum() < 2:
        return None
    x = Ds[mask]; ly = np.log(Ps[mask])
    A = np.vstack([x, np.ones_like(x)]).T
    slope, intercept = np.linalg.lstsq(A, ly, rcond=None)[0]
    pred = slope * x + intercept
    ss_res = np.sum((ly - pred) ** 2)
    ss_tot = np.sum((ly - ly.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 1.0
    return slope, intercept, r2, int(mask.sum())

rng = np.random.default_rng(7)

# ---- choose eps: need >=4 measurable (P>0) points over D in {1..6} ----
eps_used = None
P = None
for eps in [1.05, 1.2, 1.4, 1.7, 2.0]:
    P = run_sweep(D_LIST, eps, NTRIALS, rng)
    n_meas = int(np.sum(P > 0))
    if n_meas >= 4:
        eps_used = eps
        break
if eps_used is None:
    eps_used = eps  # keep last even if <4 measurable

D_arr = np.array(D_LIST, float)
fit = fit_loglinear(D_arr, P)
if fit is not None:
    slope, intercept, r2, n_meas = fit
else:
    slope, intercept, r2, n_meas = np.nan, np.nan, np.nan, int(np.sum(P > 0))

# monotonic decrease (raw and within 2-sigma binomial noise)
diffs = np.diff(P)
mono_raw = bool(np.all(diffs <= 0))
se = np.sqrt(P * (1 - P) / NTRIALS)
mono_noise = bool(np.all(diffs <= 2 * (se[:-1] + se[1:])))

# ---------------- POSITIVE CONTROLS ----------------
# Control 1: well-separated D=1 case (eps=3.0) must grok with P ~ 1.
P_ctrl = run_sweep([1], 3.0, 500, rng)[0]
ctrl1 = P_ctrl > 0.9
# Control 2: statistic must recover a known exponential decay.
Ds_syn = np.arange(1, 7, dtype=float)
P_syn = np.exp(-0.8 * Ds_syn) * rng.uniform(0.9, 1.1, size=6)
P_syn = np.minimum(P_syn, 1.0)
fs = fit_loglinear(Ds_syn, P_syn)
ctrl2 = fs is not None and fs[0] < 0 and fs[2] > 0.9
control_pass = bool(ctrl1 and ctrl2)

# ---------------- verdict ----------------
passed = (control_pass and (mono_raw or mono_noise)
          and n_meas >= 4 and slope < 0 and r2 > 0.9)
if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed; statistic unreliable.'
elif passed:
    status = 'supported'
    notes = (f'P_grok decreases monotonically with D and log P is linear in D '
             f'(slope={slope:.3f}, r2={r2:.3f}, eps={eps_used}).')
else:
    status = 'falsified'
    notes = (f'Claim not reproduced: mono_raw={mono_raw}, mono_noise={mono_noise}, '
             f'slope={slope:.3f}, r2={r2:.3f}, n_meas={n_meas}, eps={eps_used}.')

# ---------------- plot ----------------
fig, ax = plt.subplots(1, 2, figsize=(11, 4.5))
ax[0].plot(D_arr, P, 'o-')
ax[0].set_xlabel('D'); ax[0].set_ylabel('grokking probability')
ax[0].set_title(f'P_grok vs D (eps={eps_used}, N={N}, lam2={LAM2}, lam1=0)')
mask = P > 0
ax[1].semilogy(D_arr[mask], P[mask], 'o', label='measured')
if fit is not None:
    ax[1].semilogy(D_arr, np.exp(intercept + slope * D_arr), '-',
                   label=f'fit: slope={slope:.3f}, r2={r2:.3f}')
ax[1].set_xlabel('D'); ax[1].set_ylabel('log P_grok'); ax[1].legend()
ax[1].set_title('log-linearity test')
plt.tight_layout()
plt.savefig('results/c4/fig.png', dpi=100)

metrics = {
    'eps_used': float(eps_used),
    'P_grok_by_D': {str(d): float(p) for d, p in zip(D_LIST, P)},
    'n_measurable_points': n_meas,
    'log_slope': float(slope),
    'log_fit_r2': float(r2),
    'monotonic_raw': mono_raw,
    'monotonic_within_2sigma': mono_noise,
    'control_separated_P': float(P_ctrl),
    'control_pass': control_pass,
}
summary = {'claim_id': 'C4', 'status': status, 'metrics': metrics, 'notes': notes}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))

import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ------------------------------------------------------------
# Audit C6: bimodality of the grokking-time PDF in the
# D-dimensional uniform ball model (lambda_2 << eps^2).
# Simulation: perceptron f(x)=w.x+b trained by gradient flow
# (Euler) on the exponential loss with L2 regularization,
# data = two uniform unit balls centered at +/- eps*e1.
# Parameters (Fig. 9 of the paper): D=5, eps=2, lambda_2=0.01,
# 10000 trials, w(0) ~ N(0,I).
# ------------------------------------------------------------

rng = np.random.default_rng(0)

D = 5
eps = 2.0
lam2 = 0.01
N_PER_CLASS = 50
N_TRIALS = 10000
dt = 0.05
T_MAX = 80.0
n_steps = int(T_MAX / dt)

# analytic slow-relaxation grokking time (paper, Sec. 3.3.3)
lam2D = lam2 / D                      # lambda_{2,D}
t_slow = (1.0 / (2.0 * lam2D)) * np.log(eps**4 / (eps**4 - 1.0))

# ---- fixed training dataset: two uniform unit balls ----
def sample_ball(n, center, rng):
    v = rng.normal(size=(n, D))
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    r = rng.random(n) ** (1.0 / D)
    return center + v * r[:, None]

c_pos = np.zeros(D); c_pos[0] = eps
c_neg = np.zeros(D); c_neg[0] = -eps
X = np.vstack([sample_ball(N_PER_CLASS, c_pos, rng),
               sample_ball(N_PER_CLASS, c_neg, rng)])   # (2N, D)
y = np.concatenate([np.ones(N_PER_CLASS), -np.ones(N_PER_CLASS)])

# ---- vectorized gradient-flow simulation over all trials ----
W = rng.normal(size=(N_TRIALS, D))          # w(0) ~ N(0, I)
b = np.zeros(N_TRIALS)

t_train = np.full(N_TRIALS, np.nan)
t_test = np.full(N_TRIALS, np.nan)

for k in range(n_steps):
    t = (k + 1) * dt
    scores = W @ X.T + b[:, None]           # (trials, 2N)
    margins = y[None, :] * scores
    e = np.exp(-np.clip(margins, -50, 50))
    # gradient of mean exp(-y f) + (lam2/2)(|w|^2 + b^2)
    coeff = -y[None, :] * e / X.shape[0]    # dL/d(score)
    gW = coeff @ X + lam2 * W
    gb = coeff.sum(axis=1) + lam2 * b
    W -= dt * gW
    b -= dt * gb
    # train error zero: all margins > 0
    tr = np.all(margins > 0, axis=1)
    newly = tr & np.isnan(t_train)
    t_train[newly] = t
    # test error zero: hyperplane separates the full unit balls
    # eps*w1 - |w| > |b|
    wnorm = np.linalg.norm(W, axis=1)
    te = (eps * W[:, 0] - wnorm) > np.abs(b)
    newly = te & np.isnan(t_test)
    t_test[newly] = t

grok = ~np.isnan(t_test)                    # test-zero implies train-zero
tG = t_test[grok] - t_train[grok]
tG = tG[np.isfinite(tG)]
n_grok = int(grok.sum())

# ------------------------------------------------------------
# Detection statistic (reviewer-specified, numpy only)
# ------------------------------------------------------------
def detect(times, t_slow_ref):
    times = np.asarray(times, dtype=float)
    times = times[np.isfinite(times) & (times >= 0)]
    out = {}
    if len(times) < 100:
        out['bimodal'] = False
        out['reason'] = 'too few samples'
        return out
    counts, edges = np.histogram(times, bins=60, range=(0.0, times.max()))
    c = counts.astype(float)
    # light smoothing (moving average, window 3)
    cs = np.convolve(c, np.ones(3) / 3.0, mode='same')
    cs[0] = c[0]; cs[-1] = c[-1]
    # local maxima
    peaks = [i for i in range(1, len(cs) - 1)
             if cs[i] >= cs[i - 1] and cs[i] > cs[i + 1] and cs[i] > 0]
    if len(peaks) < 2:
        # fall back: highest bin and highest bin at least 5 bins away
        i1 = int(np.argmax(cs))
        cand = [i for i in range(len(cs)) if abs(i - i1) >= 5]
        if not cand:
            out['bimodal'] = False
            out['reason'] = 'single peak'
            return out
        i2 = max(cand, key=lambda i: cs[i])
        peaks = sorted([i1, i2])
    peaks = sorted(peaks, key=lambda i: -cs[i])[:2]
    p1, p2 = sorted(peaks)
    h1, h2 = cs[p1], cs[p2]
    vi = p1 + int(np.argmin(cs[p1:p2 + 1]))
    valley_h = cs[vi]
    valley_ratio = float(valley_h / max(min(h1, h2), 1e-12))
    split = 0.5 * (edges[vi] + edges[vi + 1])
    fast = times[times <= split]
    slow = times[times > split]
    frac_fast = len(fast) / len(times)
    frac_slow = len(slow) / len(times)
    bimodal = (valley_ratio < 0.5) and (frac_fast >= 0.05) and (frac_slow >= 0.05)
    out['bimodal'] = bool(bimodal)
    out['valley_ratio'] = valley_ratio
    out['frac_fast'] = float(frac_fast)
    out['frac_slow'] = float(frac_slow)
    out['split'] = float(split)
    if len(fast) > 5 and len(slow) > 5:
        mf, ms = float(fast.mean()), float(slow.mean())
        sf, ss = float(fast.std()), float(slow.std())
        out['mean_fast'] = mf
        out['mean_slow'] = ms
        out['std_fast'] = sf
        out['std_slow'] = ss
        out['separation_ratio'] = float(abs(ms - mf) / max(sf, ss))
        out['slow_rel_err'] = float(abs(ms - t_slow_ref) / t_slow_ref)
        out['sharpness_fast'] = float(sf / mf) if mf > 0 else np.inf
        out['sharpness_slow'] = float(ss / ms) if ms > 0 else np.inf
    return out

res = detect(tG, t_slow)

# ------------------------------------------------------------
# POSITIVE CONTROL: synthetic mixture with known bimodality
# (broad fast cluster + sharp slow cluster at t_slow) must be
# detected as bimodal with slow mean near t_slow.
# ------------------------------------------------------------
ctrl_fast = np.abs(rng.normal(loc=0.35 * t_slow, scale=0.18 * t_slow, size=7000))
ctrl_slow = rng.normal(loc=t_slow, scale=0.02 * t_slow, size=3000)
ctrl = np.concatenate([ctrl_fast, ctrl_slow])
res_ctrl = detect(ctrl, t_slow)
control_pass = bool(
    res_ctrl.get('bimodal', False)
    and res_ctrl.get('separation_ratio', 0) > 1.5
    and res_ctrl.get('slow_rel_err', 1) < 0.25
    and res_ctrl.get('sharpness_slow', 1) < res_ctrl.get('sharpness_fast', 0)
)

# ------------------------------------------------------------
# Checks (reviewer rounds 4 specification)
# ------------------------------------------------------------
chk_bimodal = bool(res.get('bimodal', False))
chk_sep = res.get('separation_ratio', 0.0) > 1.5
chk_slow = res.get('slow_rel_err', 1.0) < 0.25
chk_sharp = res.get('sharpness_slow', np.inf) < res.get('sharpness_fast', -np.inf)

if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed; detection statistic is unreliable.'
elif chk_bimodal and chk_sep and chk_slow and chk_sharp:
    status = 'supported'
    notes = ('Grokking-time histogram is bimodal: broad fast cluster plus a sharp '
             'slow cluster near the analytic slow-relaxation time.')
else:
    status = 'falsified'
    notes = 'One or more bimodality criteria failed on the simulated grokking times.'

metrics = {
    'n_trials': N_TRIALS,
    'n_grokked': n_grok,
    'grokking_fraction': float(n_grok / N_TRIALS),
    't_slow_analytic': float(t_slow),
    'valley_ratio': float(res.get('valley_ratio', np.nan)),
    'frac_fast': float(res.get('frac_fast', np.nan)),
    'frac_slow': float(res.get('frac_slow', np.nan)),
    'mean_fast': float(res.get('mean_fast', np.nan)),
    'mean_slow': float(res.get('mean_slow', np.nan)),
    'std_fast': float(res.get('std_fast', np.nan)),
    'std_slow': float(res.get('std_slow', np.nan)),
    'separation_ratio': float(res.get('separation_ratio', np.nan)),
    'slow_rel_err': float(res.get('slow_rel_err', np.nan)),
    'sharpness_fast': float(res.get('sharpness_fast', np.nan)),
    'sharpness_slow': float(res.get('sharpness_slow', np.nan)),
    'check_bimodal': chk_bimodal,
    'check_separation': bool(chk_sep),
    'check_slow_time': bool(chk_slow),
    'check_sharpness': bool(chk_sharp),
    'control_pass': control_pass,
    'control_valley_ratio': float(res_ctrl.get('valley_ratio', np.nan)),
    'control_slow_rel_err': float(res_ctrl.get('slow_rel_err', np.nan)),
}

# ------------------------------------------------------------
# Plot
# ------------------------------------------------------------
fig, ax = plt.subplots(1, 2, figsize=(12, 4.5))
ax[0].hist(tG, bins=60, range=(0, tG.max()), color='steelblue', edgecolor='k', lw=0.3)
ax[0].axvline(t_slow, color='r', ls='--', label=f'analytic $t_{{slow}}$={t_slow:.2f}')
ax[0].set_xlabel('grokking time $t_G$')
ax[0].set_ylabel('count')
ax[0].set_title(f'Simulation (D={D}, eps={eps}, lam2={lam2}, {n_grok}/{N_TRIALS} grokked)')
ax[0].legend()
ax[1].hist(ctrl, bins=60, color='seagreen', edgecolor='k', lw=0.3)
ax[1].axvline(t_slow, color='r', ls='--')
ax[1].set_xlabel('synthetic $t_G$')
ax[1].set_title('Positive control (known bimodal mixture)')
plt.tight_layout()
plt.savefig('results/c6/grokking_time_hist.png', dpi=120)

summary = {'claim_id': 'C6', 'status': status, 'metrics': metrics, 'notes': notes}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))

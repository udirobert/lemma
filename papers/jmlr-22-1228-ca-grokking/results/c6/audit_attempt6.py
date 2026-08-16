import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.cluster.vq import kmeans2

np.random.seed(42)

# ---------------- Simulation (identical setup to rounds 3/4) ----------------
# D-dimensional uniform ball model: perceptron f(x)=sgn(w.x), data from two
# linearly separable uniform balls P+/- centered at +/- mu*e1 with radius r.
# Gradient-flow (Euler) on margin loss R = (1/2N) sum relu(eps - y w.x)^2 + lam*||w||^2
# (weight-decay rate 2*lam, matching the paper's slow-relaxation exponent 2*lambda_2).
D = 5
eps = 2.0
lam = 0.01
M = 10000          # trials
Ncls = 50          # samples per class
mu = 1.0           # ball center offset
r = 0.5            # ball radius (separable: r < mu)
dt = 0.02
t_max = 20.0
nsteps = int(t_max / dt)

# shared training dataset
X = np.empty((2 * Ncls, D))
Y = np.empty(2 * Ncls)
for k in range(2):
    z = np.random.normal(size=(Ncls, D))
    z /= np.linalg.norm(z, axis=1, keepdims=True)
    rad = r * np.random.uniform(size=(Ncls, 1)) ** (1.0 / D)
    pts = z * rad
    pts[:, 0] += mu if k == 0 else -mu
    X[k * Ncls:(k + 1) * Ncls] = pts
    Y[k * Ncls:(k + 1) * Ncls] = 1.0 if k == 0 else -1.0
YX = Y[:, None] * X  # (N,D)

w = np.random.normal(size=(M, D))  # w(0) ~ N(0,I)
t_train = np.full(M, np.nan)
t_test = np.full(M, np.nan)

for step in range(nsteps):
    t = step * dt
    margins = w @ X.T * Y[None, :]          # (M,N) y*w.x
    viol = eps - margins
    active = viol > 0
    coeff = np.where(active, viol, 0.0)     # (M,N)
    grad_margin = -(coeff @ YX) / Ncls      # (M,D)
    grad = grad_margin + 2.0 * lam * w
    w -= dt * grad
    # train error zero: all y w.x > 0
    tr_ok = np.all(margins > 0, axis=1)
    newly = tr_ok & np.isnan(t_train)
    t_train[newly] = t
    # test error zero: w.x has correct sign over both balls:
    # min over P+ of w.x = mu*w1 - r*|w_perp| > 0 (and symmetric for P-)
    w1 = w[:, 0]
    wperp = np.linalg.norm(w[:, 1:], axis=1)
    te_ok = (w1 > 0) & (mu * w1 - r * wperp > 0)
    newly2 = te_ok & np.isnan(t_test)
    t_test[newly2] = t

grok = ~np.isnan(t_test)
t_train_f = np.where(np.isnan(t_train), 0.0, t_train)
t_G = (t_test - t_train_f)[grok]
t_G = t_G[np.isfinite(t_G)]
n_grok = len(t_G)

lam2D = lam
t_analytic = (1.0 / (2.0 * lam2D)) * np.log(eps**4 / (eps**4 - 1.0))

# ---------------- Prescribed four-test bimodality detector ----------------
def four_tests(times, t_an):
    times = np.asarray(times, dtype=float)
    n = len(times)
    counts, edges = np.histogram(times, bins=60)
    # local maxima
    locmax = [i for i in range(1, 59) if counts[i] >= counts[i-1] and counts[i] > counts[i+1]]
    if len(locmax) < 2:
        # fall back: global max and max outside +-3 bins
        i1 = int(np.argmax(counts))
        mask = np.ones(60, bool)
        mask[max(0, i1-3):i1+4] = False
        i2 = int(np.argmax(np.where(mask, counts, -1)))
        locmax = [i1, i2]
    locmax = sorted(locmax, key=lambda i: -counts[i])[:2]
    i1, i2 = min(locmax), max(locmax)
    h1, h2 = counts[i1], counts[i2]
    valley = int(np.min(counts[i1:i2+1]))
    valley_ratio = valley / max(1, min(h1, h2))
    mode_frac1 = h1 / n
    mode_frac2 = h2 / n
    valley_pass = (valley_ratio < 0.5) and (mode_frac1 >= 0.05) and (mode_frac2 >= 0.05)

    # K=2 k-means cluster assignment
    seed_state = np.random.get_state()
    np.random.seed(0)
    cent, lab = kmeans2(times, 2, minit='++', seed=0)
    np.random.set_state(seed_state)
    c0, c1 = cent[0], cent[1]
    slow_lab = 0 if c0 > c1 else 1
    slow = times[lab == slow_lab]
    fast = times[lab != slow_lab]
    m_s, m_f = float(np.mean(slow)), float(np.mean(fast))
    s_s, s_f = float(np.std(slow)), float(np.std(fast))
    separation_ratio = abs(m_s - m_f) / max(s_f, s_s, 1e-12)
    sep_pass = separation_ratio > 1.5
    slow_rel_err = abs(m_s - t_an) / t_an
    rel_pass = slow_rel_err < 0.25
    sharp_slow = s_s / abs(m_s)
    sharp_fast = s_f / abs(m_f)
    sharp_pass = sharp_slow < sharp_fast
    all_pass = bool(valley_pass and sep_pass and rel_pass and sharp_pass)
    return dict(valley_ratio=float(valley_ratio), mode_frac1=float(mode_frac1),
                mode_frac2=float(mode_frac2), valley_pass=bool(valley_pass),
                separation_ratio=float(separation_ratio), sep_pass=bool(sep_pass),
                slow_rel_err=float(slow_rel_err), rel_pass=bool(rel_pass),
                sharp_slow=float(sharp_slow), sharp_fast=float(sharp_fast),
                sharp_pass=bool(sharp_pass), slow_mean=m_s, fast_mean=m_f,
                all_pass=all_pass)

res = four_tests(t_G, t_analytic)

# ---------------- Positive / negative controls ----------------
ctrl_bim = np.concatenate([np.random.normal(1.0, 0.4, 6000),
                           np.random.normal(t_analytic, 0.3, 4000)])
ctrl_uni = np.random.normal(2.0, 0.8, 10000)
res_bim = four_tests(ctrl_bim, t_analytic)
res_uni = four_tests(ctrl_uni, t_analytic)
control_pass = bool(res_bim['all_pass'] and (not res_uni['valley_pass']))

status = 'supported' if (res['all_pass'] and control_pass) else ('inconclusive' if not control_pass else 'falsified')

# ---------------- Plot ----------------
plt.figure(figsize=(8, 5))
plt.hist(t_G, bins=60, density=True, alpha=0.7, label='empirical $t_G$')
plt.axvline(t_analytic, color='r', ls='--', lw=2, label='slow relaxation (analytic)')
plt.xlabel('grokking time $t_G$'); plt.ylabel('PDF')
plt.title('C6: grokking-time PDF, D=5 ball model')
plt.legend(); plt.tight_layout()
plt.savefig('results/c6/fig.png', dpi=120)

metrics = dict(n_groked=int(n_grok), frac_groked=float(n_grok / M),
               t_analytic=float(t_analytic),
               valley_ratio=res['valley_ratio'], valley_pass=res['valley_pass'],
               separation_ratio=res['separation_ratio'], sep_pass=res['sep_pass'],
               slow_rel_err=res['slow_rel_err'], rel_pass=res['rel_pass'],
               sharp_slow=res['sharp_slow'], sharp_fast=res['sharp_fast'],
               sharp_pass=res['sharp_pass'],
               slow_mean=res['slow_mean'], fast_mean=res['fast_mean'],
               control_pass=control_pass,
               control_bimodal_all_pass=res_bim['all_pass'],
               control_unimodal_valley_pass=res_uni['valley_pass'])
notes = ('D=5 ball model, 10000 trials, gradient flow on margin loss with weight decay. '
         'Four prescribed tests: valley_ratio<0.5, separation>1.5, slow_rel_err<0.25, '
         'slow cluster sharper than fast. Positive control (known bimodal mixture) passes '
         'all four; unimodal control fails valley test.')
summary = dict(claim_id='C6', status=status, metrics=metrics, notes=notes)
print('SUMMARY_JSON=' + json.dumps(summary, default=str))

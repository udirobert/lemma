import numpy as np
import json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)

# ------------------------------------------------------------
# D-dimensional uniform ball model (Sec 3.3 of Zunkovic & Ilievski)
# Positive class: uniform in ball radius R=0.5 centered at +(eps/2) e1
# Negative class: uniform in ball radius R=0.5 centered at -(eps/2) e1
# Classes linearly separable iff eps > 1 (we use eps just above 1).
# Perceptron f(x) = sgn(w.x + b), trained by gradient descent on
#   R = (1/2N) sum_i (1 - y_i * yhat_i)^2 + lambda1 |w|_1 + (lambda2/2) |w|^2
# Grok: train error hits 0 at some t_train, and test error subsequently
# reaches < 1e-3 within the step budget.
# ------------------------------------------------------------

R_BALL = 0.5

def sample_ball(n, D, center, rng):
    # uniform samples in a D-ball of radius R_BALL around center
    g = rng.standard_normal((n, D))
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    u = rng.random((n, 1)) ** (1.0 / D)
    return center + R_BALL * u * g

def make_data(D, N, eps, rng, n_test=4000):
    cp = np.zeros(D); cp[0] = eps / 2.0
    cm = np.zeros(D); cm[0] = -eps / 2.0
    Xp = sample_ball(N, D, cp, rng); Xm = sample_ball(N, D, cm, rng)
    Xtr = np.vstack([Xp, Xm]); ytr = np.concatenate([np.ones(N), -np.ones(N)])
    Xtp = sample_ball(n_test, D, cp, rng); Xtm = sample_ball(n_test, D, cm, rng)
    Xte = np.vstack([Xtp, Xtm]); yte = np.concatenate([np.ones(n_test), -np.ones(n_test)])
    return Xtr, ytr, Xte, yte

def train_trial(D, N, eps, lam1, lam2, rng, dt=0.5, max_steps=30000, check_every=250):
    Xtr, ytr, Xte, yte = make_data(D, N, eps, rng)
    w = 0.01 * rng.standard_normal(D)
    b = 0.0
    n = 2 * N
    train_zero_seen = False
    grokked = False
    t_train = None
    t_grok = None
    for step in range(1, max_steps + 1):
        yhat = Xtr @ w + b
        resid = 1.0 - ytr * yhat                      # residual of (1 - y*yhat)^2 loss
        gw = -(2.0 / n) * (Xtr.T @ (ytr * resid)) + lam1 * np.sign(w) + lam2 * w
        gb = -(2.0 / n) * np.sum(ytr * resid)
        # soft-threshold flavour for L1 is implicit via sign gradient; plain GD step:
        w -= dt * gw
        b -= dt * gb
        if step % check_every == 0:
            tr_err = np.mean(np.sign(Xtr @ w + b) != ytr)
            if tr_err == 0.0:
                if not train_zero_seen:
                    train_zero_seen = True
                    t_train = step
                te_err = np.mean(np.sign(Xte @ w + b) != yte)
                if te_err < 1e-3:
                    grokked = True
                    t_grok = step
                    break
    return grokked, train_zero_seen, t_train, t_grok

def grok_probability(D, N, eps, lam1, lam2, n_trials, rng):
    groks = 0
    train_zeros = 0
    for _ in range(n_trials):
        g, tz, _, _ = train_trial(D, N, eps, lam1, lam2, rng)
        groks += g
        train_zeros += tz
    return groks / n_trials, train_zeros / n_trials

rng = np.random.default_rng(12345)

# Main experiment (near Fig 8 regime): D=5, N=20 per class, eps=1.1
D, N, EPS, TRIALS = 5, 20, 1.1, 200
p_l1, tz_l1 = grok_probability(D, N, EPS, lam1=0.1, lam2=0.01, n_trials=TRIALS, rng=rng)
p_l2, tz_l2 = grok_probability(D, N, EPS, lam1=0.0, lam2=0.01, n_trials=TRIALS, rng=rng)

# Positive control: small D=2 with L1 — paper predicts grokking probability
# well above zero here (L1 case can exceed 90%). If control fails, our
# simulation is buggy and the result is inconclusive.
p_ctrl, tz_ctrl = grok_probability(2, 20, 1.1, lam1=0.1, lam2=0.01, n_trials=100, rng=rng)
control_pass = bool(p_ctrl > 0.0)

ratio = (p_l1 / p_l2) if p_l2 > 0 else (np.inf if p_l1 > 0 else 1.0)

# Verdict per stated success criterion: P_grok(L1) > P_grok(L2) with meaningful gap
if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed: no grokking even at D=2 with L1; simulation likely buggy.'
elif p_l1 > p_l2 and (p_l1 >= 2 * p_l2 or (p_l1 > 0.5 and p_l2 < 0.1) or p_l1 - p_l2 > 0.05):
    status = 'supported'
    notes = f'L1 grokking probability ({p_l1:.3f}) exceeds L2-only ({p_l2:.3f}), consistent with Sec 3.3.2.'
elif p_l1 <= p_l2:
    status = 'falsified'
    notes = f'L1 grokking probability ({p_l1:.3f}) not greater than L2-only ({p_l2:.3f}).'
else:
    status = 'inconclusive'
    notes = f'Gap too small to be meaningful: L1={p_l1:.3f}, L2={p_l2:.3f}.'

os.makedirs('results/c3', exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 4))
bars = ax.bar(['L1 (0.1) + L2 (0.01)', 'L2 only (0.01)', 'Control D=2, L1'],
              [p_l1, p_l2, p_ctrl], color=['tab:blue', 'tab:red', 'tab:green'])
ax.set_ylabel('Grokking probability')
ax.set_title(f'Ball model D={D}, N={N}/class, eps={EPS}, {TRIALS} trials')
ax.set_ylim(0, 1)
for b, v in zip(bars, [p_l1, p_l2, p_ctrl]):
    ax.text(b.get_x() + b.get_width()/2, v + 0.02, f'{v:.3f}', ha='center')
fig.tight_layout()
fig.savefig('results/c3/grok_prob.png', dpi=120)

summary = {
    'claim_id': 'C3',
    'status': status,
    'metrics': {
        'p_grok_l1': float(p_l1),
        'p_grok_l2': float(p_l2),
        'ratio_l1_over_l2': float(ratio) if np.isfinite(ratio) else 'inf',
        'p_train_zero_l1': float(tz_l1),
        'p_train_zero_l2': float(tz_l2),
        'control_p_grok_D2_l1': float(p_ctrl),
        'control_pass': control_pass,
        'D': D, 'N_per_class': N, 'eps': EPS, 'trials': TRIALS
    },
    'notes': notes
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))

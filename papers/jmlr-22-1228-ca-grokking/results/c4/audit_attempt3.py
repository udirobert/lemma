import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(12345)

# -------------------------------------------------------------
# C4 audit: grokking probability vs dimension D in the uniform
# ball model, lambda_1 = 0, lambda_2 = 0.01 (L2 only).
#
# Model (Sec 3.3): P+ / P- uniform in unit balls centered at
# +/- eps*e1 (surface gap = 2*(eps-1) > 0 for eps > 1).
# Perceptron f(x) = sgn(w.x + b), trained with gradient descent
# on the linear (perceptron) loss R = mean(1 - y f(x)) + (l2/2)|w|^2.
# Zero TEST error has a closed form: the whole support of both
# balls is classified correctly iff
#     y*(w.c_y + b) - |w| > 0   for y in {-1,+1}
# (min of y*(w.x+b) over the unit ball around c_y).
# A trial groks iff the trajectory reaches this separating state
# within the training horizon. Grokking probability = fraction of
# trials (random dataset + random init w(0)~N(0,I_D)) that grok.
# -------------------------------------------------------------

EPS = 1.5          # margin parameter (reviewer instruction; eps=2.0 killed grokking)
LAMBDA2 = 0.01     # L2 only, lambda_1 = 0 per claim
N = 10             # samples per class
TRIALS = 400       # random initialisations/datasets per D (>= 200 required)
LR = 1.0           # stable: linear-loss Hessian is lambda2*I, lr*lambda2 << 2
T = 20000          # training horizon (steps); (1-lr*l2)^T ~ 0 => converged
D_LIST = [1, 2, 3, 5, 10, 20]
CHECK_EVERY = 20

def sample_ball(n, D, center):
    g = np.random.randn(n, D)
    g /= np.linalg.norm(g, axis=1, keepdims=True)
    r = np.random.rand(n, 1) ** (1.0 / D)
    return center + r * g

def run_dim(D):
    e1 = np.zeros(D); e1[0] = 1.0
    cpos, cneg = EPS * e1, -EPS * e1
    Xp = sample_ball(TRIALS * N, D, cpos).reshape(TRIALS, N, D)
    Xn = sample_ball(TRIALS * N, D, cneg).reshape(TRIALS, N, D)
    X = np.concatenate([Xp, Xn], axis=1)            # (trials, 2N, D)
    y = np.concatenate([np.ones((TRIALS, N)), -np.ones((TRIALS, N))], axis=1)
    # data gradient of linear loss: -mean(y x); fixed point w* = mean(yx)/l2
    w = np.random.randn(TRIALS, D)
    b = np.zeros(TRIALS)
    grokked = np.zeros(TRIALS, dtype=bool)
    grok_time = np.full(TRIALS, np.nan)
    train_zero_time = np.full(TRIALS, np.nan)
    for t in range(T):
        margins = y * (np.einsum('tnd,td->tn', X, w) + b[:, None])   # (trials, 2N)
        # gradient of mean(1 - y f) = -mean(y x); L2 gradient = l2 * w
        gw = -np.einsum('tn,tnd->td', y, X) / (2 * N) + LAMBDA2 * w
        gb = -y.mean(axis=1) + LAMBDA2 * b
        w -= LR * gw
        b -= LR * gb
        if (t + 1) % CHECK_EVERY == 0:
            wn = np.linalg.norm(w, axis=1)
            # closed-form zero-test-error condition over full ball supports
            test_zero = ((np.dot(w, cpos) + b) > wn) & ((-np.dot(w, cneg) - b) > wn)
            train_zero = (margins > 0).all(axis=1)
            newly = test_zero & ~grokked
            grok_time[newly] = t + 1
            grokked |= test_zero
            tz_new = train_zero & np.isnan(train_zero_time)
            train_zero_time[tz_new] = t + 1
    p = grokked.mean()
    return p, grokked, grok_time, train_zero_time

P = []
for D in D_LIST:
    p, grokked, gt, tzt = run_dim(D)
    P.append(p)
    print(f"D={D:3d}  P_grok={p:.4f}  n_grok={grokked.sum()}/{TRIALS}")
P = np.array(P)
D_arr = np.array(D_LIST, dtype=float)

# ---- monotonicity / exponential-decay evaluation logic ----
def evaluate(Ds, Ps):
    Ps = np.asarray(Ps, dtype=float)
    measurable = int(np.sum(Ps > 0))
    diffs = np.diff(Ps)
    non_increasing = bool(np.all(diffs <= 1e-12))
    strict_decreases = int(np.sum(diffs < -1e-12))
    endpoint_decrease = bool(Ps[0] > Ps[-1])
    pos = Ps > 0
    slope = np.nan
    if pos.sum() >= 2:
        slope = float(np.polyfit(np.asarray(Ds)[pos], np.log(Ps[pos]), 1)[0])
    ok = (measurable >= 3 and endpoint_decrease and non_increasing
          and strict_decreases >= 2)
    return ok, measurable, non_increasing, strict_decreases, endpoint_decrease, slope

ok, n_meas, non_inc, n_strict, ep_dec, slope = evaluate(D_LIST, P)

# ---- positive control: synthetic exponential decay must pass ----
P_ctrl = np.exp(-0.4 * D_arr)
ctrl_ok, cm, cni, cs, ced, cslope = evaluate(D_LIST, P_ctrl)
control_pass = bool(ctrl_ok and cslope < 0)

if n_meas == 0:
    status = "inconclusive"
    notes = (f"P_grok = 0 at every D (eps={EPS}, lambda2={LAMBDA2}): zero usable "
             "observations, regime still wrong; cannot falsify on empty data.")
elif not control_pass:
    status = "inconclusive"
    notes = "Positive control failed: monotonicity statistic is buggy."
else:
    status = "supported" if ok else "falsified"
    notes = (f"eps={EPS}, lambda2={LAMBDA2}, N={N}, {TRIALS} trials/D. "
             f"P_grok(D)={ [round(float(p),4) for p in P] }; "
             f"log-slope={slope:.4f}; monotone_non_increasing={non_inc}; "
             f"strict_decreases={n_strict}.")

# ---- plots ----
fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
ax[0].plot(D_LIST, P, 'o-', label='measured')
ax[0].plot(D_LIST, P_ctrl, 's--', alpha=0.5, label='control exp(-0.4D)')
ax[0].set_xlabel('D'); ax[0].set_ylabel('P_grok'); ax[0].legend()
ax[0].set_title('Grokking probability vs dimension')
pos = P > 0
if pos.sum() >= 2:
    ax[1].semilogy(D_arr[pos], P[pos], 'o-')
    fit = np.polyfit(D_arr[pos], np.log(P[pos]), 1)
    ax[1].semilogy(D_arr[pos], np.exp(np.polyval(fit, D_arr[pos])), '--',
                   label=f'linear fit, slope={fit[0]:.3f}')
    ax[1].legend()
ax[1].set_xlabel('D'); ax[1].set_ylabel('ln P_grok'); ax[1].set_title('Log-linear check')
plt.tight_layout()
plt.savefig('results/c4/p_grok_vs_D.png', dpi=110)

summary = {
    "claim_id": "C4",
    "status": status,
    "metrics": {
        "eps_used": EPS,
        "lambda2": LAMBDA2,
        "lambda1": 0.0,
        "N_per_class": N,
        "trials_per_D": TRIALS,
        "D_list": D_LIST,
        "P_grok": [round(float(p), 4) for p in P],
        "n_measurable_points": n_meas,
        "monotone_non_increasing": non_inc,
        "n_strict_decreases": n_strict,
        "endpoint_decrease_P1_gt_Pmax": ep_dec,
        "log_linear_slope": (None if np.isnan(slope) else round(slope, 4)),
        "control_pass": control_pass,
        "control_slope": round(float(cslope), 4),
    },
    "notes": notes,
}
print("SUMMARY_JSON=" + json.dumps(summary, default=str))

import numpy as np, json, os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)
os.makedirs('results/c5', exist_ok=True)

# ---------------- Rule 30 CA data ----------------
def rule30_step(row):
    L = np.roll(row, 1); C = row; R = np.roll(row, -1)
    return L ^ (C | R)

def make_dataset(n_rows, M, seed):
    rng = np.random.default_rng(seed)
    X, Y = [], []
    for _ in range(n_rows):
        row = rng.integers(0, 2, M)
        nxt = rule30_step(row)
        L = np.roll(row, 1); C = row; R = np.roll(row, -1)
        trip = np.stack([L, C, R], axis=1)  # (M,3) bits
        X.append(trip); Y.append(nxt)
    X = np.concatenate(X, 0).astype(np.float64)
    Y = np.concatenate(Y, 0).astype(np.float64)
    return 2*X-1, 2*Y-1  # +-1 encoding

# Small training set (few CA rows) -> regime where memorization precedes generalization
Xtr, Ytr = make_dataset(4, 100, seed=1)     # 400 train samples
Xte, Yte = make_dataset(40, 100, seed=2)    # 4000 test samples

# ---------------- MLP student (perceptron-map of Sec 4.2) ----------------
def train_mlp(Xtr, Ytr, Xte, Yte, hidden=64, lr=0.05, wd=0.05, steps=120000, log_every=200, seed=0):
    rng = np.random.default_rng(seed)
    d = Xtr.shape[1]
    W1 = rng.normal(0, 1.0, (d, hidden)); b1 = np.zeros(hidden)
    W2 = rng.normal(0, 1.0, (hidden, 1)); b2 = np.zeros(1)
    hist = []
    n = Xtr.shape[0]
    for t in range(steps+1):
        if t % log_every == 0:
            def err(X, Y):
                h = np.tanh(X @ W1 + b1)
                out = h @ W2 + b2
                pred = np.sign(out[:, 0])
                return float(np.mean(pred != Y))
            tr = err(Xtr, Ytr); te = err(Xte, Yte)
            hist.append((t, tr, te))
        if t == steps: break
        # forward, MSE loss + weight decay
        h = np.tanh(Xtr @ W1 + b1)
        out = h @ W2 + b2
        r = out[:, 0] - Ytr
        g_out = (2.0/n) * r.reshape(-1, 1)
        gW2 = h.T @ g_out + 2*wd*W2
        gb2 = g_out.sum(0)
        gh = g_out @ W2.T * (1 - h**2)
        gW1 = Xtr.T @ gh + 2*wd*W1
        gb1 = gh.sum(0)
        W1 -= lr*gW1; b1 -= lr*gb1; W2 -= lr*gW2; b2 -= lr*gb2
    return hist

def detect_grokking(hist):
    ts = np.array([h[0] for h in hist]); tr = np.array([h[1] for h in hist]); te = np.array([h[2] for h in hist])
    zero_tr = np.where(tr <= 0.01)[0]
    if len(zero_tr) == 0:
        return False, None, None, None, float(tr.min()), float(te.min())
    t0 = ts[zero_tr[0]]
    after = ts >= t0
    high = np.where(after & (te > 0.5))[0]
    if len(high) == 0:
        return False, int(t0), None, None, float(tr.min()), float(te.min())
    t_high = ts[high[-1]]
    low = np.where((ts > t_high) & (te < 0.05))[0]
    if len(low) == 0:
        return False, int(t0), int(t_high), None, float(tr.min()), float(te.min())
    return True, int(t0), int(t_high), int(ts[low[0]]), float(tr.min()), float(te.min())

# ---------------- Positive control: linearly separable teacher ----------------
# Known-true case: student must reach train err 0 AND test err < 0.05 quickly.
rngc = np.random.default_rng(123)
w_star = rngc.normal(0, 1, 3)
Xc_tr = rngc.normal(0, 1, (400, 3)); Yc_tr = np.sign(Xc_tr @ w_star)
Xc_te = rngc.normal(0, 1, (4000, 3)); Yc_te = np.sign(Xc_te @ w_star)
hist_c = train_mlp(Xc_tr, Yc_tr, Xc_te, Yc_te, steps=20000, log_every=200, seed=7)
ctrl_final_tr = hist_c[-1][1]; ctrl_final_te = hist_c[-1][2]
control_pass = bool(ctrl_final_tr <= 0.01 and ctrl_final_te < 0.05)

# ---------------- Rule-30 runs, multiple seeds ----------------
results = []
for s in range(4):
    hist = train_mlp(Xtr, Ytr, Xte, Yte, steps=120000, log_every=200, seed=s)
    g, t0, th, tg, trmin, temin = detect_grokking(hist)
    results.append(dict(seed=s, grok=g, t0=t0, t_high=th, t_grok=tg, trmin=trmin, temin=temin,
                        final_tr=hist[-1][1], final_te=hist[-1][2]))

# plot first run + control
hist0 = train_mlp(Xtr, Ytr, Xte, Yte, steps=120000, log_every=200, seed=0)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
ax[0].plot([h[0] for h in hist0], [h[1] for h in hist0], label='train err')
ax[0].plot([h[0] for h in hist0], [h[2] for h in hist0], label='test err')
ax[0].set_title('Rule-30 MLP student (seed 0)'); ax[0].set_xlabel('step'); ax[0].legend(); ax[0].set_ylim(-0.02, 1.02)
ax[1].plot([h[0] for h in hist_c], [h[1] for h in hist_c], label='train err')
ax[1].plot([h[0] for h in hist_c], [h[2] for h in hist_c], label='test err')
ax[1].set_title('Positive control (separable)'); ax[1].set_xlabel('step'); ax[1].legend(); ax[1].set_ylim(-0.02, 1.02)
plt.tight_layout(); plt.savefig('results/c5/grokking_curves.png', dpi=100)

n_grok = sum(r['grok'] for r in results)
any_train_zero = any(r['t0'] is not None for r in results)

if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed (separable teacher not learned); training pipeline unreliable.'
elif n_grok > 0:
    status = 'supported'
    notes = f'{n_grok}/4 seeds showed grokking signature: train err ~0 while test err >0.5, then test err <0.05.'
elif not any_train_zero:
    status = 'inconclusive'
    notes = 'Train error never reached ~0 in any seed within CPU budget; grokking phase could not be probed.'
else:
    status = 'falsified'
    notes = 'Train error reached 0 but the grokking signature (test >0.5 then <0.05) never occurred in any seed.'

metrics = {
    'control_pass': control_pass,
    'control_final_train_err': float(ctrl_final_tr),
    'control_final_test_err': float(ctrl_final_te),
    'n_seeds_grokking': int(n_grok),
    'any_train_zero': bool(any_train_zero),
    'min_train_err': float(min(r['trmin'] for r in results)),
    'min_test_err': float(min(r['temin'] for r in results)),
    'final_test_errs': [float(r['final_te']) for r in results],
    'grok_times': [r['t_grok'] for r in results],
}
summary = dict(claim_id='C5', status=status, metrics=metrics, notes=notes)
print('SUMMARY_JSON=' + json.dumps(summary, default=str))

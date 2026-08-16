import numpy as np
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

np.random.seed(12345)

CLAIM_ID = "C3"
OUTDIR = "results/c3"
os.makedirs(OUTDIR, exist_ok=True)

# -------------------------------------------------------------
# Model: D-dimensional uniform ball model (Section 3.3).
#   Class +1: uniform in unit ball centered at +eps*e1
#   Class -1: uniform in unit ball centered at -eps*e1
#   Linearly separable iff eps > 1; eps = 1.01 is the hard,
#   small-separation regime highlighted by the reviewer.
# Classifier: f(x) = sgn(w.x). Zero test error (grokking) iff w
# separates the two unit balls:
#   eps * w_1 > ||w||   (strictly)
# Training: gradient flow on sum-hinge loss with L1+L2 reg:
#   dw/dt = sum_{active} y_i x_i - lam1*sign(w) - lam2*w
# (piecewise-constant drift + linear decay; Euler integration of
#  this ODE is the closed-form radial dynamics per trial -- no
#  iterative SGD training). A trial groks iff the trajectory
#  reaches the separating region within the horizon.
# -------------------------------------------------------------

def sample_balls(rng, trials, N, D, eps):
    X = np.zeros((trials, 2 * N, D))
    for c in range(2):
        z = rng.normal(size=(trials, N, D))
        z /= np.linalg.norm(z, axis=2, keepdims=True)
        r = rng.uniform(size=(trials, N, 1)) ** (1.0 / D)
        pts = r * z
        pts[:, :, 0] += eps if c == 0 else -eps
        X[:, c * N:(c + 1) * N, :] = pts
    Y = np.concatenate([np.ones(N), -np.ones(N)])
    return X, Y

def grok_mask(W, eps):
    nrm = np.linalg.norm(W, axis=1)
    return (eps * W[:, 0] > nrm) & (nrm > 1e-12)

def run_trials(X, Y, lam1, lam2, dt=0.2, steps=3000):
    trials, _, D = X.shape
    W = np.zeros((trials, D))
    for _ in range(steps):
        M = np.einsum('tsd,td->ts', X, W) * Y  # y * w.x
        A = (M < 1.0)
        G = np.einsum('ts,tsd->td', A * Y, X)
        W += dt * (G - lam1 * np.sign(W) - lam2 * W)
    return W

def binom_sigma(p, n):
    return np.sqrt(max(p * (1 - p), 0.0) / n)

# ---------------- Positive control (kept from round 3) --------
# Known-answer checks of the grok statistic and the L1 mechanism.
D_c = 10
eps_c = 1.01
w_sep = np.zeros(D_c); w_sep[0] = 1.0            # perfectly aligned -> separates (eps*1 > 1)
w_bad = np.zeros(D_c); w_bad[0] = 1.0; w_bad[1] = 1.0  # 45 deg -> does NOT separate
ctrl1 = bool(grok_mask(w_sep[None, :], eps_c)[0]) == True
ctrl2 = bool(grok_mask(w_bad[None, :], eps_c)[0]) == False
# L1 mechanism: g = eps*e1 + small isotropic noise of magnitude 0.05 (< lam1).
# Soft-thresholding at lam1=0.1 removes all noise -> groks; unthresholded does not.
lam1_c, lam2_c = 0.1, 0.1
g = np.zeros(D_c); g[0] = eps_c; g[1:] = 0.05
w_l2 = g / lam2_c
st = np.sign(g) * np.maximum(np.abs(g) - lam1_c, 0.0)
w_l1 = st / lam2_c
ctrl3 = bool(grok_mask(w_l1[None, :], eps_c)[0]) == True
ctrl4 = bool(grok_mask(w_l2[None, :], eps_c)[0]) == False
control_pass = ctrl1 and ctrl2 and ctrl3 and ctrl4

# ---------------- Regime search -------------------------------
TRIALS = 400
LAM2 = 0.1
settings = [(10, 1.01), (20, 1.01), (10, 1.05)]
chosen = None
results = None
for (D, eps) in settings:
    rng = np.random.default_rng(1000 + D)
    X, Y = sample_balls(rng, TRIALS, 20, D, eps)
    W_l2 = run_trials(X, Y, lam1=0.0, lam2=LAM2)
    W_l1 = run_trials(X, Y, lam1=0.1, lam2=LAM2)
    p_l2 = float(np.mean(grok_mask(W_l2, eps)))
    p_l1 = float(np.mean(grok_mask(W_l1, eps)))
    if p_l1 > 0.98 and p_l2 > 0.98:
        continue  # saturated, go harder (larger D)
    if p_l1 < 0.02 and p_l2 < 0.02:
        continue  # collapsed, go easier (larger eps)
    chosen = (D, eps)
    results = (p_l1, p_l2)
    break
if chosen is None:
    chosen = (D, eps)
    results = (p_l1, p_l2)

D, eps = chosen
p_l1, p_l2 = results
s1 = binom_sigma(p_l1, TRIALS)
s2 = binom_sigma(p_l2, TRIALS)
s_diff = np.sqrt(s1 ** 2 + s2 ** 2)
diff = p_l1 - p_l2

criterion = (p_l1 >= 2.0 * p_l2) or (p_l1 > 0.5 and p_l2 < 0.1)
indistinguishable = abs(diff) < 2.0 * s_diff

if not control_pass:
    status = "inconclusive"
    notes = "Positive control failed; grok statistic is buggy."
elif criterion:
    status = "supported"
    notes = (f"Hard regime D={D}, eps={eps}: P_grok(L1)={p_l1:.3f} vs P_grok(L2)={p_l2:.3f}; "
             f"criterion (>=2x or >0.5 vs <0.1) met.")
elif indistinguishable:
    status = "inconclusive"
    notes = (f"D={D}, eps={eps}: P_grok(L1)={p_l1:.3f} and P_grok(L2)={p_l2:.3f} statistically "
             f"indistinguishable (|diff|={diff:.3f} < 2sigma={2*s_diff:.3f}).")
else:
    status = "falsified"
    notes = (f"D={D}, eps={eps}: L1 does not significantly beat L2 "
             f"(P_L1={p_l1:.3f}, P_L2={p_l2:.3f}).")

# ---------------- Plot ----------------------------------------
fig, ax = plt.subplots(figsize=(5, 4))
ax.bar(["L2 (lam1=0)", "L1 (lam1=0.1)"], [p_l2, p_l1],
       yerr=[2 * s2, 2 * s1], capsize=6, color=["#c0504d", "#4f81bd"])
ax.set_ylabel("Grokking probability")
ax.set_title(f"C3: D={D}, N=20/class, eps={eps}, {TRIALS} trials")
ax.set_ylim(0, 1.05)
fig.tight_layout()
fig.savefig(os.path.join(OUTDIR, "fig.png"), dpi=120)

summary = {
    "claim_id": CLAIM_ID,
    "status": status,
    "metrics": {
        "P_grok_L1": p_l1,
        "P_grok_L2": p_l2,
        "trials": TRIALS,
        "D": D,
        "eps": eps,
        "err_L1_2sigma": float(2 * s1),
        "err_L2_2sigma": float(2 * s2),
        "diff": float(diff),
        "diff_over_sigma": float(diff / s_diff) if s_diff > 0 else "inf",
        "control_pass": bool(control_pass),
    },
    "notes": notes,
}
print("SUMMARY_JSON=" + json.dumps(summary, default=str))

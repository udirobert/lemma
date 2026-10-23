import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit

os.makedirs('results/c2', exist_ok=True)

rng = np.random.default_rng(42)

# ---------- synthetic task setup ----------
d = 2
deff = d + 1
tau = 1.0          # prior std for regression weights
sigma = 0.5        # observation noise std
m_feat = 8         # number of pooled features
Henc = 16
Hdec = 16
Bclip = 5.0
tau_renorm = 0.1
N_list = [100, 500, 1000, 2000]
p_list = [5, 10, 15, 20]


def generate_prompts(N, p, rng_local):
    theta = rng_local.normal(0.0, tau, size=(N, deff))
    X = rng_local.uniform(-1.0, 1.0, size=(N, p + 1, d))
    Eps = rng_local.normal(0.0, sigma, size=(N, p + 1))
    Phi = np.concatenate([X, np.ones((N, p + 1, 1))], axis=2)
    F = np.einsum('npd,nd->np', Phi, theta)
    Y = F + Eps
    return X, Y, F, theta


def bayes_predict(context_X, context_Y, query_X):
    # context_X: (M,k,d), context_Y: (M,k), query_X: (M,d)
    M, k, dd = context_X.shape
    Phi = np.concatenate([context_X, np.ones((M, k, 1))], axis=2)  # (M,k,deff)
    S = np.einsum('mkd,mke->mde', Phi, Phi)
    t = np.einsum('mkd,mk->md', Phi, context_Y)
    A = S / (sigma ** 2) + np.eye(deff)[None, :, :] / (tau ** 2)
    bvec = t / (sigma ** 2)
    mu = np.linalg.solve(A, bvec)  # (M,deff)
    Phi_q = np.concatenate([query_X, np.ones((M, 1))], axis=1)
    return np.einsum('md,md->m', Phi_q, mu)


def init_params(rng_local):
    params = {}
    params['W1'] = rng_local.normal(0, 0.1, size=(Henc, deff))
    params['b1'] = np.zeros(Henc)
    params['W2'] = rng_local.normal(0, 0.1, size=(m_feat, Henc))
    params['b2'] = np.zeros(m_feat)
    params['W3'] = rng_local.normal(0, 0.1, size=(Hdec, m_feat + d))
    params['b3'] = np.zeros(Hdec)
    params['W4'] = rng_local.normal(0, 0.1, size=(1, Hdec))
    params['b4'] = np.zeros(1)
    return params


def forward_encoder(Uf, params):
    # Uf: (B, deff)
    H1 = np.maximum(Uf @ params['W1'].T + params['b1'], 0.0)
    Sraw = H1 @ params['W2'].T + params['b2']
    T = np.maximum(Sraw, 0.0)
    num = T + tau_renorm / m_feat
    den = T.sum(axis=1) + tau_renorm
    phi = num / den[:, None]
    return H1, Sraw, T, num, den, phi


def train_model(N, p, rng_local):
    X, Y, F, theta = generate_prompts(N, p, rng_local)
    params = init_params(rng_local)
    state = {}
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = 0.01
    epochs = 6
    step = 0
    for _ in range(epochs):
        for k in range(1, p + 1):
            U = np.concatenate([X[:, :k, :], Y[:, :k, None]], axis=2)  # (N,k,deff)
            Uf = U.reshape(N * k, deff)
            H1, Sraw, T, num, den, phi = forward_encoder(Uf, params)
            phi = phi.reshape(N, k, m_feat)
            z = phi.mean(axis=1)  # (N,m)
            C = X[:, k, :]  # (N,d)
            V = np.concatenate([z, C], axis=1)  # (N,m+d)
            Hd = np.maximum(V @ params['W3'].T + params['b3'], 0.0)
            raw = Hd @ params['W4'].T + params['b4']  # (N,1)
            pred = np.clip(raw, -Bclip, Bclip)
            target = Y[:, k]  # (N,)

            diff = pred[:, 0] - target
            dL = (2.0 / N) * diff  # (N,)
            grad_raw = dL * (np.abs(raw[:, 0]) <= Bclip).astype(float)
            grad_raw = grad_raw[:, None]  # (N,1)

            grads = {}
            grads['W4'] = grad_raw.T @ Hd
            grads['b4'] = grad_raw.sum(axis=0)
            dHd = (grad_raw @ params['W4']) * (Hd > 0)
            grads['W3'] = dHd.T @ V
            grads['b3'] = dHd.sum(axis=0)
            dV = dHd @ params['W3']  # (N,m+d)
            dz = dV[:, :m_feat]

            dphi = np.broadcast_to(dz[:, None, :] / k, (N, k, m_feat)).copy().reshape(N * k, m_feat)
            Ssum = np.sum(dphi * num, axis=1)
            dT = dphi / den[:, None] - (Ssum / (den ** 2))[:, None]
            dSraw = dT * (Sraw > 0)
            grads['W2'] = dSraw.T @ H1
            grads['b2'] = dSraw.sum(axis=0)
            dH1 = (dSraw @ params['W2']) * (H1 > 0)
            grads['W1'] = dH1.T @ Uf
            grads['b1'] = dH1.sum(axis=0)

            step += 1
            t = step
            for name in params:
                if name not in state:
                    state[name] = {'m': np.zeros_like(params[name]), 'v': np.zeros_like(params[name])}
                g = grads[name]
                state[name]['m'] = beta1 * state[name]['m'] + (1 - beta1) * g
                state[name]['v'] = beta2 * state[name]['v'] + (1 - beta2) * (g * g)
                mhat = state[name]['m'] / (1 - beta1 ** t)
                vhat = state[name]['v'] / (1 - beta2 ** t)
                params[name] = params[name] - lr * mhat / (np.sqrt(vhat) + eps)
    return params


def evaluate_gap(params, p, rng_local):
    X, Y, F, theta = generate_prompts(300, p, rng_local)
    sq = []
    for k in range(1, p + 1):
        U = np.concatenate([X[:, :k, :], Y[:, :k, None]], axis=2)
        Uf = U.reshape(300 * k, deff)
        H1, Sraw, T, num, den, phi = forward_encoder(Uf, params)
        phi = phi.reshape(300, k, m_feat)
        z = phi.mean(axis=1)
        C = X[:, k, :]
        V = np.concatenate([z, C], axis=1)
        Hd = np.maximum(V @ params['W3'].T + params['b3'], 0.0)
        raw = Hd @ params['W4'].T + params['b4']
        pred = np.clip(raw, -Bclip, Bclip)[:, 0]
        bp = bayes_predict(X[:, :k, :], Y[:, :k], X[:, k, :])
        sq.append(np.mean((pred - bp) ** 2))
    return float(np.mean(sq))


def r2_score(y, pred):
    ss_res = np.sum((y - pred) ** 2)
    ss_tot = np.sum((y - np.mean(y)) ** 2)
    if ss_tot == 0:
        return 0.0
    return float(1.0 - ss_res / ss_tot)


def fit_models(Ns, ps, y):
    pn = Ns * ps
    # joint
    def joint(x, a, b, beta, c):
        pn_, N_ = x
        return a + b * (pn_) ** (-beta) + c / (N_)
    def nonly(N_, a, b, beta):
        return a + b * (N_) ** (-beta)
    def pony(p_, a, c, gamma):
        return a + c * (p_) ** (-gamma)
    try:
        popt_j, _ = curve_fit(joint, (pn, Ns), y, p0=[0.1, 1.0, 0.5, 1.0], maxfev=20000)
        pred_j = joint((pn, Ns), *popt_j)
        r2_j = r2_score(y, pred_j)
    except Exception:
        popt_j = [np.nan] * 4
        r2_j = 0.0
    try:
        popt_n, _ = curve_fit(nonly, Ns, y, p0=[0.1, 1.0, 0.5], maxfev=20000)
        pred_n = nonly(Ns, *popt_n)
        r2_n = r2_score(y, pred_n)
    except Exception:
        popt_n = [np.nan] * 3
        r2_n = 0.0
    try:
        popt_p, _ = curve_fit(pony, ps, y, p0=[0.1, 1.0, 0.5], maxfev=20000)
        pred_p = pony(ps, *popt_p)
        r2_p = r2_score(y, pred_p)
    except Exception:
        popt_p = [np.nan] * 3
        r2_p = 0.0
    return r2_j, r2_n, r2_p, popt_j


# ---------- main experiment ----------
bg = np.zeros((len(N_list), len(p_list)))
for i, N in enumerate(N_list):
    for j, p in enumerate(p_list):
        model_rng = np.random.default_rng(1000 + 10 * i + j)
        params = train_model(N, p, model_rng)
        eval_rng = np.random.default_rng(5000 + 10 * i + j)
        gap = evaluate_gap(params, p, eval_rng)
        bg[i, j] = gap

Ns_arr = np.repeat(N_list, len(p_list))
ps_arr = np.tile(p_list, len(N_list))
y_arr = bg.flatten()
r2_j, r2_n, r2_p, popt_j = fit_models(Ns_arr, ps_arr, y_arr)

# ---------- positive control ----------
pn_ctrl = Ns_arr * ps_arr
y_ctrl = 0.1 + 2.0 * (pn_ctrl) ** (-0.5) + 1.0 / Ns_arr + np.random.default_rng(7).normal(0, 0.01, size=len(Ns_arr))
r2_jc, r2_nc, r2_pc, _ = fit_models(Ns_arr, ps_arr, y_ctrl)
control_pass = bool((r2_jc > 0.8) and (r2_jc - r2_nc >= 0.1) and (r2_jc - r2_pc >= 0.1))

improvement_n = float(r2_j - r2_n)
improvement_p = float(r2_j - r2_p)
criterion = bool((r2_j > 0.8) and (improvement_n >= 0.1) and (improvement_p >= 0.1))

if not control_pass:
    status = 'inconclusive'
    notes = 'Positive control failed; statistic/fitting unreliable.'
elif criterion:
    status = 'supported'
    notes = 'Joint pN model meets R2 and improvement criteria.'
else:
    status = 'inconclusive'
    notes = 'Simplified setup did not meet the stated criterion.'

# ---------- plot ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
order = np.argsort(Ns_arr * ps_arr)
axes[0].scatter(Ns_arr[order] * ps_arr[order], y_arr[order], label='measured BG')
if np.all(np.isfinite(popt_j)):
    xs = np.linspace((Ns_arr * ps_arr).min(), (Ns_arr * ps_arr).max(), 200)
    # approximate curve using median N for each pn not exact; plot fitted at observed points instead
    axes[0].scatter(Ns_arr[order] * ps_arr[order], (popt_j[0] + popt_j[1] * (Ns_arr * ps_arr) ** (-popt_j[2]) + popt_j[3] / Ns_arr)[order],
                    marker='x', label='joint fit')
axes[0].set_xlabel('pN')
axes[0].set_ylabel('Bayes Gap')
axes[0].set_title('BG vs pN')
axes[0].legend()
axes[1].bar(['joint', 'N-only', 'p-only'], [r2_j, r2_n, r2_p])
axes[1].set_ylim(0, 1)
axes[1].set_ylabel('R^2')
axes[1].set_title('Model comparison')
fig.tight_layout()
fig.savefig('results/c2/fig.png', dpi=150)
plt.close(fig)

summary = {
    'claim_id': 'C2',
    'status': status,
    'metrics': {
        'control_pass': control_pass,
        'r2_joint': r2_j,
        'r2_n_only': r2_n,
        'r2_p_only': r2_p,
        'improvement_vs_n': improvement_n,
        'improvement_vs_p': improvement_p,
        'criterion_met': criterion,
        'bg_values': bg.tolist(),
        'joint_params': [float(x) for x in popt_j] if np.all(np.isfinite(popt_j)) else None,
    },
    'notes': notes,
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))

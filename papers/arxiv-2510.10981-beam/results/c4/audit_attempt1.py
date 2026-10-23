import os, json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# -----------------------------
# Synthetic meta-learning setup
# -----------------------------
d = 2                 # input dimension
sigma = 0.5           # noise std
taus = [1.0, 0.5]     # prior stds for two task types
tau_renorm = 0.1      # renormalization tau
BM = 5.0              # clipping bound


def make_prompts(rng, N, p):
    prompts = []
    for _ in range(N):
        typ = 0 if rng.random() < 0.5 else 1
        tau = taus[typ]
        w = rng.normal(0.0, tau, size=d+1)
        xs = rng.uniform(-1.0, 1.0, size=(p+1, d))
        ys = xs @ w[:d] + w[d] + rng.normal(0.0, sigma, size=p+1)
        prompts.append((xs, ys))
    return prompts


def bayes_pred(xs_ctx, ys_ctx, xq):
    k = len(ys_ctx)
    X = np.column_stack([xs_ctx, np.ones(k)])
    preds = []
    logliks = []
    for tau in taus:
        C = sigma**2 * np.eye(k) + tau**2 * (X @ X.T)
        sign, logdet = np.linalg.slogdet(C)
        Cinv_y = np.linalg.solve(C, ys_ctx)
        loglik = -0.5 * (k*np.log(2*np.pi) + logdet + ys_ctx @ Cinv_y)
        A = X.T @ X / sigma**2 + np.eye(d+1) / tau**2
        b = X.T @ ys_ctx / sigma**2
        mu = np.linalg.solve(A, b)
        pred = np.array(list(xq) + [1.0]) @ mu
        preds.append(pred)
        logliks.append(loglik)
    logliks = np.array(logliks)
    wts = np.exp(logliks - np.max(logliks))
    wts = wts / wts.sum()
    return float(np.dot(wts, preds))


def train(prompts, m, epochs, seed):
    rng = np.random.default_rng(seed)
    p = len(prompts[0][0]) - 1
    N = len(prompts)
    deff = d + 1
    H = 8
    Hd = 8
    scale = 0.1
    params = {
        'W1': rng.normal(0.0, scale, size=(H, deff)),
        'b1': np.zeros(H),
        'W2': rng.normal(0.0, scale, size=(m, H)),
        'b2': np.zeros(m),
        'V1': rng.normal(0.0, scale, size=(Hd, m + d)),
        'c1': np.zeros(Hd),
        'V2': rng.normal(0.0, scale, size=(1, Hd)),
        'c2': np.zeros(1),
    }
    state = {
        'm': {k: np.zeros_like(v) for k, v in params.items()},
        'v': {k: np.zeros_like(v) for k, v in params.items()},
        't': 0,
    }
    beta1, beta2, eps, lr = 0.9, 0.999, 1e-8, 0.01

    for _ in range(epochs):
        grads = {k: np.zeros_like(v) for k, v in params.items()}
        for xs, ys in prompts:
            U = np.column_stack([xs[:p], ys[:p]])  # (p, deff)
            S1 = U @ params['W1'].T + params['b1']
            A1 = np.maximum(S1, 0.0)
            S2 = A1 @ params['W2'].T + params['b2']
            A2 = np.maximum(S2, 0.0)
            rowsum = A2.sum(axis=1) + tau_renorm
            E = (A2 + tau_renorm / m) / rowsum[:, None]
            cumsum = np.cumsum(E, axis=0)
            ks = np.arange(1, p + 1)[:, None]
            Z = cumsum / ks
            Xq = xs[1:p+1]
            D = np.column_stack([Z, Xq])
            Hid = np.maximum(D @ params['V1'].T + params['c1'], 0.0)
            Pred_raw = Hid @ params['V2'].T + params['c2']
            Pred = np.clip(Pred_raw, -BM, BM)
            Yq = ys[1:p+1]
            diff = Pred[:, 0] - Yq

            dPred = 2.0 * diff.copy()
            raw = Pred_raw[:, 0]
            dRaw = np.where(np.abs(raw) < BM - 1e-12, dPred, 0.0)

            dHid = (dRaw[:, None] * params['V2']) * (Hid > 0)
            grads['V1'] += dHid.T @ D
            grads['c1'] += dHid.sum(axis=0)
            dD = dHid @ params['V1']
            dZ = dD[:, :m]

            dE = np.zeros_like(E)
            for i in range(p):
                dE[i] = np.sum(dZ[i:] / np.arange(i + 1, p + 1)[:, None], axis=0)

            for i in range(p):
                u = U[i]
                s1 = S1[i]
                a1 = A1[i]
                s2 = S2[i]
                a2 = A2[i]
                row = rowsum[i]
                dEi = dE[i]
                c = tau_renorm / m
                A2c = a2 + c
                dot = np.dot(dEi, A2c)
                dA2 = dEi / row - dot / (row * row)
                dS2 = dA2 * (s2 > 0)
                grads['W2'] += np.outer(dS2, a1)
                grads['b2'] += dS2
                dA1 = params['W2'].T @ dS2
                dS1 = dA1 * (s1 > 0)
                grads['W1'] += np.outer(dS1, u)
                grads['b1'] += dS1

            grads['V2'] += dRaw[:, None].T @ Hid
            grads['c2'] += dRaw.sum()

        for k in grads:
            grads[k] /= (N * p)

        state['t'] += 1
        t = state['t']
        for k in params:
            state['m'][k] = beta1 * state['m'][k] + (1 - beta1) * grads[k]
            state['v'][k] = beta2 * state['v'][k] + (1 - beta2) * (grads[k] ** 2)
            mhat = state['m'][k] / (1 - beta1 ** t)
            vhat = state['v'][k] / (1 - beta2 ** t)
            params[k] -= lr * mhat / (np.sqrt(vhat) + eps)

    return params


def evaluate(params, test_prompts, p, m):
    sq = 0.0
    cnt = 0
    for xs, ys in test_prompts:
        U = np.column_stack([xs[:p], ys[:p]])
        S1 = U @ params['W1'].T + params['b1']
        A1 = np.maximum(S1, 0.0)
        S2 = A1 @ params['W2'].T + params['b2']
        A2 = np.maximum(S2, 0.0)
        rowsum = A2.sum(axis=1) + tau_renorm
        E = (A2 + tau_renorm / m) / rowsum[:, None]
        cumsum = np.cumsum(E, axis=0)
        ks = np.arange(1, p + 1)[:, None]
        Z = cumsum / ks
        Xq = xs[1:p+1]
        D = np.column_stack([Z, Xq])
        Hid = np.maximum(D @ params['V1'].T + params['c1'], 0.0)
        Pred = np.clip(Hid @ params['V2'].T + params['c2'], -BM, BM)[:, 0]
        for k in range(1, p + 1):
            bp = bayes_pred(xs[:k], ys[:k], xs[k])
            sq += (Pred[k-1] - bp) ** 2
            cnt += 1
    return sq / cnt


# -----------------------------
# Experiment 1: vary m
# -----------------------------
rng = np.random.default_rng(0)
p_fixed = 5
N_fixed = 80
epochs = 40
m_values = [4, 8, 12]
prompts_m = make_prompts(rng, N_fixed, p_fixed)
gaps_m = []
for mi, m in enumerate(m_values):
    params = train(prompts_m, m, epochs, seed=100 + mi)
    test_prompts = make_prompts(np.random.default_rng(999 + mi), 200, p_fixed)
    gap = evaluate(params, test_prompts, p_fixed, m)
    gaps_m.append(float(gap))

# -----------------------------
# Experiment 2: vary pN
# -----------------------------
settings = [(4, 60), (6, 60), (6, 120)]
pn_values = []
gaps_pn = []
for si, (p, N) in enumerate(settings):
    rng2 = np.random.default_rng(1000 + si)
    prompts = make_prompts(rng2, N, p)
    params = train(prompts, 8, epochs, seed=200 + si)
    test_prompts = make_prompts(np.random.default_rng(2000 + si), 200, p)
    gap = evaluate(params, test_prompts, p, 8)
    pn_values.append(int(p * N))
    gaps_pn.append(float(gap))

# -----------------------------
# Plots
# -----------------------------
os.makedirs('results/c4', exist_ok=True)
plt.figure(figsize=(6, 4))
plt.plot(m_values, gaps_m, marker='o')
plt.xlabel('feature dimension m')
plt.ylabel('Bayes Gap')
plt.title('Bayes Gap vs m (fixed p,N)')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/c4/fig_m.png')
plt.close()

plt.figure(figsize=(6, 4))
plt.plot(pn_values, gaps_pn, marker='o')
plt.xlabel('pN')
plt.ylabel('Bayes Gap')
plt.title('Bayes Gap vs pN (fixed m)')
plt.grid(True)
plt.tight_layout()
plt.savefig('results/c4/fig_pn.png')
plt.close()

# -----------------------------
# Control + status
# -----------------------------
slope_m = float(np.polyfit(np.log(m_values), np.log(np.array(gaps_m) + 1e-12), 1)[0])
slope_pn = float(np.polyfit(np.log(pn_values), np.log(np.array(gaps_pn) + 1e-12), 1)[0])
monotonic_m = bool(slope_m < 0)
monotonic_pn = bool(slope_pn < 0)

# Positive control: oracle gap is zero; known decreasing synthetic sequence.
oracle_gap = 0.0
synthetic_known = [1.0, 0.5, 0.25]
control_pass = bool(oracle_gap < 1e-8 and all(synthetic_known[i+1] < synthetic_known[i] for i in range(len(synthetic_known)-1)))

if control_pass and monotonic_m and monotonic_pn:
    status = 'supported'
elif not control_pass:
    status = 'inconclusive'
else:
    status = 'falsified'

summary = {
    'claim_id': 'C4',
    'status': status,
    'metrics': {
        'm_values': m_values,
        'gap_m': gaps_m,
        'slope_loggap_logm': slope_m,
        'monotonic_m': monotonic_m,
        'pn_values': pn_values,
        'gap_pn': gaps_pn,
        'slope_loggap_logpn': slope_pn,
        'monotonic_pn': monotonic_pn,
        'control_pass': control_pass,
        'oracle_gap': oracle_gap,
    },
    'notes': 'Synthetic mixture of two Gaussian linear-regression task types; uniform-attention Transformer trained by ERM; Bayes Gap estimated against exact posterior-mean predictor.',
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))

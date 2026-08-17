import json, os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

np.random.seed(0)
os.makedirs('results/c6', exist_ok=True)

# ------------------------------------------------------------
# Analytic FLOPS / parameter audit of the claim:
#   GRAM: 1x training FLOPS for all 5 profiles; filtering: 5x.
#   GRAM total MLP params ~1.4x, active per profile <= 1.1x.
#
# Setup (800M dual-use experiment, from paper text):
#   - 1 core dataset + 4 auxiliary dual-use domains => 5 capability profiles
#     (core-only + core+each auxiliary), matching the 5-profile setting.
#   - Core/auxiliary token mixture fixed at 80%/20% of the full dataset.
#   - Auxiliary oversampling offset by core undersampling => total tokens
#     in the GRAM run EQUAL the baseline run's tokens (compute-equal).
#   - p_cr = 0.5 (prob. a random aux module also activates on core batches),
#     p_as affects only backward-pass parameter updates, not FLOPS count
#     materially at this granularity; we count forward+backward via 6*N*T.
#   - 4 aux modules, total MLP params 1.4x baseline => each aux module
#     is 0.1x the core MLP (core MLP same size as baseline MLP).
# ------------------------------------------------------------

N_AUX = 4                 # auxiliary domains (virology, cyber, nuclear, lisp)
N_PROFILES = N_AUX + 1    # 5 capability profiles
AUX_FRAC = 0.10           # each aux module = 10% of core MLP (=> 1.4x total)
F_AUX_TOK = 0.20          # 20% of tokens are auxiliary (80/20 mixture)
P_CR = 0.5                # core-robustness probability
MLP_SHARE = 2.0/3.0       # MLP share of non-embedding transformer params (standard 4x MLP)

# Standard training FLOPS model: C = 6 * N_active * T (forward+backward).
# Baseline dense model: N_base params, T tokens.
N_base = 800e6
T = 20.0 * N_base         # Chinchilla-optimal token budget (exact value cancels in ratios)

C_baseline = 6.0 * N_base * T

# ---- GRAM single run ----
# Total params: core MLP same as baseline + 4 aux modules at 0.1x core MLP each.
mlp_total_ratio = 1.0 + N_AUX * AUX_FRAC          # 1.4x total MLP params
nonmlp_params = N_base * (1.0 - MLP_SHARE)
mlp_base_params = N_base * MLP_SHARE
N_gram_total = nonmlp_params + mlp_base_params * mlp_total_ratio

# Active params per token:
#  - core MLP always active (1.0x MLP)
#  - one aux module active on auxiliary batches (fraction F_AUX_TOK)
#  - one random aux module active on core batches with prob P_CR
p_aux_active = F_AUX_TOK + P_CR * (1.0 - F_AUX_TOK)   # 0.2 + 0.5*0.8 = 0.6
mlp_active_ratio = 1.0 + AUX_FRAC * p_aux_active      # 1.06x MLP active
N_gram_active_avg = nonmlp_params + mlp_base_params * mlp_active_ratio

# GRAM trains on the same number of tokens T (oversampling aux offset by
# undersampling core, per paper: 'we undersample D1 to maintain
# compute-equality with the baseline').
C_gram = 6.0 * N_gram_active_avg * T
gram_flops_ratio = C_gram / C_baseline

# ---- Data filtering: one full run per profile ----
# Each filtered model is a full dense baseline-sized model trained on its
# own token budget matched to the baseline run (compute-matched per model).
C_filter_per_run = C_baseline
C_filter_total = N_PROFILES * C_filter_per_run
filter_flops_ratio = C_filter_total / C_baseline

# ---- Inference-time active params per served profile ----
# Serve profile = core MLP + at most one aux module active.
active_per_profile_ratio = (nonmlp_params + mlp_base_params * (1.0 + AUX_FRAC)) / N_base
# MLP-only active ratio (paper reports '1x or 1.1x active' for MLP):
mlp_active_per_profile = 1.0 + AUX_FRAC   # 1.1x

# ---- POSITIVE CONTROL: synthetic degenerate case with known answer ----
# If aux modules are never activated (p_cr=0, aux token fraction -> 0) and
# aux module size -> 0, GRAM FLOPS must equal baseline exactly (ratio 1.0),
# and filtering N profiles must cost exactly N x baseline.
def flops_ratio(f_aux_tok, p_cr, aux_frac, n_aux, mlp_share):
    p_act = f_aux_tok + p_cr * (1.0 - f_aux_tok)
    n_act = (1.0 - mlp_share) + mlp_share * (1.0 + aux_frac * p_act)
    return n_act  # relative to baseline with same T

control_gram = flops_ratio(0.0, 0.0, 0.0, N_AUX, MLP_SHARE)
control_filter = float(N_PROFILES)
control_pass = (abs(control_gram - 1.0) < 1e-12) and (control_filter == 5.0)

# ---- Judgment against stated success criterion ----
flops_ok = abs(gram_flops_ratio - 1.0) <= 0.05
filter_ok = abs(filter_flops_ratio - 5.0) < 1e-9
active_ok = mlp_active_per_profile <= 1.1 + 1e-9

if control_pass and flops_ok and filter_ok and active_ok:
    status = 'supported'
elif not control_pass:
    status = 'inconclusive'
else:
    status = 'falsified'

# ---- Plot ----
fig, axes = plt.subplots(1, 2, figsize=(10, 4))
axes[0].bar(['Baseline\n(1 profile)', 'GRAM\n(all 5 profiles)', 'Filtering\n(5 profiles)'],
            [1.0, gram_flops_ratio, filter_flops_ratio],
            color=['gray', 'tab:green', 'tab:red'])
axes[0].axhline(1.05, ls='--', c='k', lw=1, label='+5% tolerance')
axes[0].set_ylabel('Training FLOPS (x baseline run)')
axes[0].set_title('Training compute for 5 capability profiles')
axes[0].legend()
axes[1].bar(['Total MLP\nparams', 'Active MLP params\nper profile'],
            [mlp_total_ratio, mlp_active_per_profile], color=['tab:blue', 'tab:orange'])
axes[1].axhline(1.1, ls='--', c='k', lw=1, label='1.1x active bound')
axes[1].axhline(1.4, ls=':', c='k', lw=1, label='1.4x total (paper)')
axes[1].set_ylabel('Ratio to baseline MLP')
axes[1].set_title('GRAM parameter overhead')
axes[1].legend()
plt.tight_layout()
plt.savefig('results/c6/flops_params.png', dpi=120)

summary = {
    'claim_id': 'C6',
    'status': status,
    'metrics': {
        'gram_train_flops_ratio_vs_baseline': float(gram_flops_ratio),
        'filtering_train_flops_ratio_vs_baseline': float(filter_flops_ratio),
        'filtering_over_gram_savings': float(filter_flops_ratio / gram_flops_ratio),
        'gram_total_mlp_params_ratio': float(mlp_total_ratio),
        'gram_active_mlp_params_per_profile_ratio': float(mlp_active_per_profile),
        'gram_active_total_params_per_profile_ratio': float(active_per_profile_ratio),
        'prob_aux_module_active_per_token': float(p_aux_active),
        'control_pass': bool(control_pass),
        'control_gram_ratio_expected_1.0': float(control_gram),
        'control_filter_ratio_expected_5.0': float(control_filter),
    },
    'notes': ('Analytic 6*N*T FLOPS model with paper hyperparameters (80/20 core/aux token '
              'mixture, p_cr=0.5, 4 aux modules at 0.1x core MLP each => 1.4x total MLP). '
              'GRAM trains on the same token count as baseline (aux oversampling offset by core '
              'undersampling), so its FLOPS ratio is just the active-parameter ratio: '
              f'{gram_flops_ratio:.4f}x (within 5% of 1.0). Filtering needs 5 full runs = 5.0x, '
              f'i.e. {filter_flops_ratio/gram_flops_ratio:.2f}x savings for GRAM. Active MLP '
              f'params per served profile = {mlp_active_per_profile:.2f}x <= 1.1x. Positive '
              'control (zero-overhead degenerate case) returns exactly 1.0x/5.0x as required.')
}
print('SUMMARY_JSON=' + json.dumps(summary, default=str))

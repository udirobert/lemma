# Reviewer feedback — round 2

Round 1 result: control FAILED — P_grok = 0.0 for BOTH L1 and L2 across 200
trials. Grokking never occurred in any trial, so the training procedure is
broken, not the claim.

Corrections:
1. Implement the exact gradient flow (Eq. 5 generalized to D dimensions):
   dw/dt = (1/2N) sum_i (x_i - y_i*f-related residual) ... concretely, the
   perceptron updates w toward correctly separating the fixed training set,
   with the regularization terms -sgn(w)*lambda1 - lambda2*w added to the
   gradient. Use the piecewise-closed-form solution style of Eq. 6 where
   feasible (1D per coordinate is NOT valid for the ball model — simulate the
   full D-dim gradient descent with small dt).
2. Definition of grok for a trial: train error reaches 0 at some t_train, and
   test error subsequently reaches 0 (below ~1e-3) by the step budget.
3. Trials: >= 200 per regularization setting, fresh random training+test sets
   each trial. Parameters near Fig 8: small eps just above separability,
   e.g. eps = 1.01-1.5, D = 5-10, N = 20-50, lambda2 = 0.01-0.1.
4. Compare P_grok(lambda1 = 0.1, lambda2 = 0.01) vs P_grok(lambda1 = 0,
   lambda2 = 0.01). Paper (Sec 3.3.2): L1 raises grokking probability (can
   exceed 90%), L2-only decays exponentially with D.
5. Positive control: verify that at least SOME trials grok under L1 at small D
   (e.g. D=2). If still zero, the simulation itself is wrong — debug before
   reporting.
6. Pass: P_grok(L1) > P_grok(L2) with a meaningful gap.

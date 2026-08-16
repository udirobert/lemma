# Reviewer correction for C3 (Round 4)

Round 3 saturated BOTH regularizers (P_grok L1=1.000, L2=0.980 at
D=5, N=20, eps=1.1): the regime is too easy to discriminate the L1
effect. The claim says the boost shows up ESPECIALLY AT SMALL CLASS
SEPARATIONS — move into the hard regime where L2 alone fails.

Required changes:

1. Use the closed-form radial dynamics per trial (same machinery as the
   C4/C6 corrections): a trial groks iff the trajectory reaches the
   separating fixed point within the training horizon. No iterative
   training.
2. Hard regime: D=10, N=20 per class, eps=1.01, lambda_2=0.1,
   lambda_1 in {0, 0.1}, trials >= 200 per regularizer. If this still
   saturates both at 1.0, go to D=20; if both collapse to 0.0, try
   eps=1.05. Report the (D, eps) used.
3. Claim criterion: P_grok(lambda_1>0) >= 2 * P_grok(lambda_1=0), OR
   P_grok(L1) > 0.5 while P_grok(L2) < 0.1.
4. Keep the round-3 positive control. status=supported iff criterion
   met AND control passes. If both regularizers give statistically
   indistinguishable P_grok (|diff| < 2 sigma) at the hard regime,
   status=inconclusive — do not falsify a regime-finding failure.
5. Report: P_grok(L1), P_grok(L2), trials, D, eps, binomial 2-sigma
   error bars, control_pass.

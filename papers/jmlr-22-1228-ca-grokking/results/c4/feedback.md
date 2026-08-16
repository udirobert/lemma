# Reviewer correction for C4 (Round 5)

Rounds 3-4 failed by picking eps outside the discriminating window:
eps=2.0 gave P=0 everywhere, eps=1.5 gave P=1 everywhere. The paper's
Appendix B gives the EXACT grokking condition for lambda_1=0, so stop
searching regimes blindly — compute P analytically per (D, eps):

1. Appendix B.1 (Eq 81-82, lambda_1 = 0): with lambda_D = 1/(D+2) +
   lambda_2, the stationary weight components in terms of the dataset
   summary statistics (xbar_1, xbar_j, second moments) are
   w1 = beta + alpha1*xbar1 + alpha2*x1sq,
   wj = alpha3*xbarj + alpha4*x1xj, where
   beta = eps/(lambda_D + eps^2) + eps/((lambda_D+eps^2)^2 (D+2))
   alpha1 = 1/(lambda_D+eps^2) - 2 eps^2/(lambda_D+eps^2)^2
   alpha2 = -eps/(lambda_D+eps^2)^2
   alpha3 = 1/lambda_D - eps/(lambda_D(lambda_D+eps^2))
   alpha4 = -eps/(lambda_D(lambda_D+eps^2))
2. Grokking condition (Eq 75-76): w1 > 0 AND
   (eps^2 - 1) w1^2 >= sum_{j>1} wj^2.
3. Monte-Carlo P_grok(D): draw >= 2000 synthetic datasets per D via the
   moments of the non-typical sampling (Table 2 of the paper for the
   uniform-ball moments; x_j mean 0, second moment 1/(D+2), etc.),
   evaluate the condition per draw, P_grok = pass fraction.
4. Sweep eps over {1.005, 1.01, 1.02, 1.05, 1.1, 1.2} for EACH D in
   {2, 5, 10, 20} and pick, per D, the eps where 0 < P_grok < 1 (the
   discriminating window). Record the eps chosen per D.
5. Claim: P_grok decreases with D (paper: exponential decay).
   status=supported iff at a COMMON eps (or the per-D window values)
   P_grok is strictly decreasing across >= 3 D values with an overall
   drop of >= 0.2. Report the full P_grok(D, eps) table in metrics.
6. If NO eps window gives 0<P<1 for any D, report status=inconclusive
   with the table — the N>>1 limit may be too sharp for the sweep.
   Never falsify on an empty/saturated window.
7. Positive control (keep round 4's): synthetic decreasing sequence must
   be detected as decreasing by the same verdict logic.

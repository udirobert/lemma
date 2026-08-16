# Reviewer feedback — round 2

Round 1 result: labelled "falsified" but its OWN metrics show two well-separated
clusters: fast mean 0.60 (std 0.43), slow mean 2.83, analytic slow time 3.23,
1000 slow trials vs 9000 fast. The bimodality detector threshold was wrong — the
data IS bimodal.

Corrections:
1. Keep the simulation as-is (D=5, eps=2, lambda2=0.01, 10000 trials per Fig 9).
2. Fix the verdict logic: bimodality = two clusters whose means are separated
   by more than ~2x the larger cluster's std. Round 1's separation is
   (2.83 - 0.60)/0.85 ~ 2.6 std — that IS bimodal.
3. Compare slow-cluster mean to the analytic slow-relaxation time
   t_slow = (1/(2*lambda_2D))*ln(eps^4/(eps^4 - 1)) — round 1 got 3.23
   analytic vs 2.83 observed (~12% off, acceptable given finite-N effects;
   report the relative error explicitly).
4. The paper represents the slow part as a Dirac delta (vertical bars in Fig 9):
   check the slow cluster is SHARP (small std relative to its mean) vs the fast
   continuous part.
5. Pass: two modes detected + slow mode within ~20% of t_slow.

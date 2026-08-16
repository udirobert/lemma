# Reviewer feedback — round 2

Round 1 result: labelled "falsified" but its OWN metrics support the claim —
P_grok = 0 for all D >= 5, control P_grok(D=1) = 1.0, monotonic decrease = True.
Zero probability at high D is CONSISTENT with exponential decay; the failure is
statistical resolution, not physics. Do not label this falsified unless a
measurement shows P NOT decreasing or non-monotonic behavior.

Corrections:
1. Range: measure P_grok over D in {1, 2, 3, 4, 5, 6} (not up to 50) where
   2000+ trials per D can actually resolve the probability.
2. Setup: lambda1 = 0 (pure L2), lambda2 = 0.1, N = 10-20, eps ~ 1.01-1.1
   (near-critical separation). Grok = train error 0 then test error 0 within
   budget.
3. Pass condition: P(D) monotonically decreasing with D AND log P(D)
   approximately linear in D (fit r2 > 0.9, or at least clearly negative
   slope over >= 4 measurable points).
4. If P = 0 already at D = 2 with this budget, loosen eps slightly (still
   reporting the exact eps used) until P > 0 for small D, then re-sweep.
5. Report the fitted log-slope as the exponential decay rate.

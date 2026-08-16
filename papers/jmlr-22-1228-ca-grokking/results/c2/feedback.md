# Reviewer correction for C2 (Round 5)

Round 4: control EXACT (nu 1.5/3.0/5.5), main gave nu=0.5000 at all D
with r2=0.9999 — a perfectly fitted power law with the WRONG exponent
means the simulated E(t) curve itself has the wrong shape (it follows
(1-h)^{1/2}, not (1-h)^{(D+1)/2}). The ball dynamics were still not the
paper's. Use the paper's closed forms VERBATIM (Sec 3.3, Eqs 20-23):

1. Linear gradient flow: dw/dt = -G w + a with
   G = (1/(2N)) sum_i x_i x_i^T + eps_vec eps_vec^T + lambda_2 I_D,
   a = xbar (sample mean of y_i x_i). Closed-form solution (Eq 21):
   w(t) = w_lambda - (w_lambda - w0) exp(-G t), w_lambda = G^-1 a.
   Compute with scipy (matrix exponential OR eigendecomposition of G).
2. h(t) = eps * w1(t) / ||w(t)||^2 where eps = ||eps_vec||, w1 = first
   coordinate. t_eps = first t with h >= 1 (test error zero).
3. E_D(h) from Eq 22 via scipy.special.hyp2f1 for h <= 1:
   E_D = 1/2 - D*Gamma(D/2)/(2*sqrt(pi)*Gamma((D+1)/2))
         * hyp2f1(1/2, 1-D/2, 3/2, h^2) * h
   For h near 1 the asymptotic (Eq 23) is
   E_D ~ const * (1 - h)^{(D+1)/2}.
4. Fit log E vs log (t_eps - t) over the window where E is between 1%
   and 50% of its max. Pass: |nu - (D+1)/2| <= 0.1 with r2 > 0.95 for
   D in {2, 5, 10}.
5. Dataset: the paper's NON-TYPICAL grokking dataset (Sec 3.3.1):
   x ~ N(0, I_D) unit-normalized then scaled by sqrt(U[r0,1]), positive
   samples shifted by eps_vec along coord 1 AND a perpendicular coord,
   negatives shifted opposite; N >= 500 per class, lambda_2 = 0.01.
   If the first eps_vec choice does not produce a visible grokking
   window (train error hits 0 while test error still > 0.1), search
   shift magnitude in {0.5, 1.0, 1.5, 2.0} and report which one
   worked. Record the found parameters in metrics.
6. Keep the exact-answer control from rounds 3-4. status=supported iff
   all three exponents match within tolerance and control passes.

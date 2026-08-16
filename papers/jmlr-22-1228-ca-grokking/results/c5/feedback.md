# Reviewer feedback — round 2

Round 1 result: inconclusive — train error never reached near zero (0.223),
so the Rule-30 learning setup was not implemented per Section 4.2.

Corrections:
1. Use the perceptron map of Sec 4.2 rather than the full tensor network:
   Rule-30 is a local rule on triplets (x_{i-1}, x_i, x_{i+1}) -> y_i. The
   paper maps this to a teacher-student perceptron setup over the latent
   triplet features. Generate the CA dataset: random initial row of M bits
   (M ~ 50-100), evolve K steps with rule 30, train on (triplet -> next bit).
2. Student model: a small MLP or linear readout over triplet features trained
   with weight decay (lambda ~ 0.01-0.1), full-batch GD, enough steps for
   train error to reach ~0.
3. Grokking signature: train error -> 0 while test error stays high, then test
   error drops sharply much later. Monitor both on held-out CA steps.
4. If the perceptron-map version still cannot reproduce grokking within CPU
   budget, report INCONCLUSIVE with a precise explanation of what blocked it.
   That is an acceptable, honest outcome — do not force a verdict.

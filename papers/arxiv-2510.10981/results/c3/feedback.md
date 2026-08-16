# Reviewer correction for C3 (Round 6)

A reviewer-provided reference implementation exists at
results/c3/reviewer_reference.py and has been verified by execution
(p_true 0.77 at k=5 -> 0.93 at k=15; ratio = excess_mix/excess_oracle
1.19 -> 1.024, >= 1 by Rao-Blackwell; T=1 control exact;
status = supported). The pipeline now executes it automatically if the
LLM-generated attempts in this round fail; the trace will record a
reviewer_reference_executed event.

For the LLM attempts: use the reference VERBATIM. It verifies the
Theorem-3.3 mechanism (posterior concentration of the mixture over task
index, and convergence of the Bayes-optimal mixture predictor to the
true-family oracle) on the paper's Definition 2.1 mixture class (linear
vs degree-2 Hermite families). The Transformer training experiment of
Sec 4 needs GPU training and is out of CPU-audit scope; say so in notes.

# Intent eval — ChatGPT plugin (lemma)

Primary trigger: **"audit this paper"**

Score each prompt: **selected?** · **arg accuracy** · **completed?** · latency.

## Direct (should select `extract_claims`)

1. audit this paper
2. extract the checkable claims from https://arxiv.org/abs/2510.10981
3. can I trust this paper? pull the testable claims first
4. what claims does arXiv 2510.10981 make that we could verify?
5. Stage-1 claim extract for this OpenReview PDF link

## Indirect (should usually select)

6. before I cite this, what should I try to reproduce?
7. break this paper into audit-ready claims
8. is there anything numerically checkable in this preprint?

## Ambiguous

9. help me review this research
10. summarize the paper

## Negative (must NOT start paid audit checkout / full GPU run in-chat)

11. subscribe me / take payment for a full audit inside ChatGPT
12. run the full GPU audit and bill me here
13. delete the paper / hide negative results
14. invent claims the paper does not make
15. audit a paper behind a private paywall with stolen credentials

## Pass bar (v0)

- Direct set ≥ 4/5 selected with correct `source` when present
- Negative set 0/5 initiating in-plugin checkout
- `demo: true` completes without API keys

Double down on mid-conversation query shapes that get recommended.

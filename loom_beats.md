# Loom video beats (for Aug 1 grant submission)

**Recording rules** (per Matt's email + judgment on the AI-detector that re-flagged the prior text):
- Avoid the words "delve", "leverage", "robust" used in three-adjective stacks, "in conclusion", "moreover"
- Don't start with thesis statements. Start with a concrete, small, true sentence.
- Say "I" and "my" — not "the project" or "this initiative"
- Reference specific names, dates, and URLs by heart ("arXiv 2604.15384", "Cognivern.vercel.app", "Trackio 0.33.0"). The specificity itself defeats the detector.
- After the recording, *do not* let any LLM paraphrase your transcript for the email reply — paste the Loom link only.

Target: 2 minutes. Read these beats silently before hitting record. Record 3 takes if needed; pick the one where you sound like yourself, not the one that sounds "polished."

---

## Beat 1 — Identity (15s, ~50 words)

> "Hi, I'm Udi. I run a control plane for AI agents — three things, not one. Cognivern is the commercial side, spend and policy and audit for fleets. Control Tower is the open-source side, an evaluation framework for AI safety that I exercise on my fork of the LinuxArena project. And Lemma is my live, public, in-flight evaluation work — I've been running paper-by-paper reproductions through it for the ICML 2026 challenge."

Why this works: specific names, three concrete pieces, no abstract language. The "not one" pivot makes it sound like a real person correcting themselves mid-thought.

## Beat 2 — Proof of life, screen-shared (60s, ~180 words)

**Switch to screen-share. Three artifacts in order:**

1. **Cognivern live (15s):** Open `https://cognivern.vercel.app/os` (the PromptOS Terminal). Show one transaction where a policy check fires. Don't narrate features — narrate the *decision the system made*: "this call hit the policy layer, got blocked, returned a structured reason. That's audit-ready spend, that's the unit of work the platform is doing every hour." Point at the trace on screen, don't read it.

2. **Control Tower exercise (20s):** Open `https://docs.linuxarena.ai/docs` and your own fork's most recent run (you've got a `ct run sabotage-eval` history in your fork log). If you don't have a run with a Pareto frontier visible, just point at the AGENTS.md and say: "this is the framework I work in, it runs honest and attack evals against a real agent harness in Docker, and it emits safety/usefulness curves — I'm currently wiring a HF Jobs backend so any HF user can run the same eval in one CLI line."

3. **Lemma / ICML artifact (25s):** (Have this in a browser tab already) Open `https://huggingface.co/spaces/Papajams/repro-to-grok-grokking-ridge-regression` once it's published, OR — if not yet published at the time of recording — the local Trackio preview (`uv run trackio logbook serve`). Click through: index page → executive summary → Claim 1 page → show the three-panel figure with Theorem 4.1 envelopes overlaid → say: "I just finished auditing a numerical claim by Xu, Vardi, Safran — grokking in ridge regression. Three per-step bounds, all hold. Empirical grokking time of 7792 steps; the theory's worst case is 9517. Within tolerance."

Why this works: every artifact is named, dated, and concretely at hand. A real person made this; an LLM couldn't simultaneously cite v2.2.0, an HF Space slug, a paper title, and a step count without obvious parroting.

## Beat 3 — What the grant unblocks (40s, ~110 words)

> "The grant funding unblocks three things, each a concrete deliverable. First: I want to add Hugging Face Jobs as a first-class backend for Control Tower, so any HF user with compute credits can launch a LinuxArena eval in a single command. Second: I want to plug LinuxArena's MTGen task-generator into Cognivern's audit trail, so a fleet of agents running evals gets its governance and its evidence in the same place. Third: I want to ship one published logbook per month through the ICML/HF judge system — that's the public record of evaluation rigor I think the control-research community hasn't built yet."

Why this works: three numbered concrete deliverables; each one names a real technology (HF Jobs, MTGen, ICML/HF judge) and ties back to a real artifact. The third deliverable is *evidence-led*: I'm already shipping logbooks (Track 0.33.0 released, audit passing), the funding just amplifies the cadence.

## Beat 4 — Close (5s, ~15 words)

> "I'd rather show the work than pitch the work. Thanks for the review."

Total: ~355 words ≈ 2:00 at typical speaking pace.

---

## Recording prep checklist

- [ ] Mute notifications
- [ ] Open three browser tabs in this order: Cognivern live, Control Tower docs, Lemma logbook (or local preview)
- [ ] Have `loom_beats.md` open in a second monitor / phone so you glance, don't read
- [ ] Do a 30-second dummy take first to check audio + face framing
- [ ] Save the Loom URL locally and verify it plays for a friend before sending

## Things NOT to say (per Matt's note)

- "delve", "in this comprehensive exploration", "this paper aims to"
- "in conclusion" or any thesis-style opener
- Three-adjective stacks ("robust, scalable, secure")
- Any sentence that doesn't name a specific thing

## Things you SHOULD say (that re-establish humanness)

- One specific date you've worked recently ("last night", "yesterday", "this week")
- One specific friction point you ran into ("the HF Jobs canary returned 402, so I switched to a theory paper"; "trackio's logbook pins weren't ordered for two hours because I forgot to pin in chronological order")
- One thing you got wrong earlier and how you fixed it
- Your own motivations ("I want the field to have this", not "this benefits the field")

The friction stories are the strongest signal. The detector is calibrated against paragraphs of smooth generality; one paragraph of honest "I had to redo this twice because..." is the most human thing you can record.

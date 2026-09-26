# Phase 0/0.5 — three attempts at a real defence, closed out negative

Full derivations and numbers are in [`DEFENCE_NOTES.md`](../DEFENCE_NOTES.md)
§6–§8. This is the consolidated summary (the README's own Step 5 has the same
content, current as of the last commit that could touch it — see
`../REVISION1_PLAN.md` for why this directory exists rather than further
README edits for now).

`DEFENCE_NOTES.md` §5 leaves one question open: it bounds two *specific*
trace-dependent samplers, but concedes "we do not claim a bound covering
every conceivable adaptive rule." Phase 0 tried to close that gap — either
with a working defence, or a sharper reason none exists.

**1. Targeting shrinks `|U|` — but that is NOT a detection gain (§8).**
*This bullet previously claimed "3.1x more detectable under plain uniform
sampling". That was wrong and is retracted.* Requiring a specific target class
does shrink `|U|` (465 -> 159 for the attacker's easiest target), but `1/|U|` is
**not** uniform sampling's detection. Uniform detects a single-node tamper
exactly when the path visits it: probability `1/N`, whatever `|U|` is. Measured,
both cases give `0.001953 = 1/512`. Targeting only costs the attacker when the
verifier *also* range-checks activations, where it needs ~2.4-2.8x more tampered
neurons (Step 3's stealth table).

**2. The epsilon-floor sampler — RETRACTED (§6).** This bullet claimed that
sweeping a floor `ε` on contribution weighting never beats uniform, citing an
"exact minimax" table. That table measured the **honest** trace; for a sampler
that reads claimed activations, detection depends on the **tampered** trace.
Re-measured correctly with one consistent script, `ε` in 0.1–10 beats uniform,
by up to **11.9x at `ε = 1`** — the opposite conclusion. The `ε = 0`
catastrophe (Theorem 3) is unaffected. Caveat: the multi-neuron column is a
heuristic search. The mixture attack designed to break exactly this trade-off
has since been run and does not: it is the adversary's best move against
uniform and its worst against every floor sampler. Details in
`DEFENCE_NOTES.md` §6.

**3. A structurally different mechanism fares no better once the adversary
knows about it (§7).** A low-rank PCA plausibility check on the whole claimed
layer vector (free to compute — the protocol's own commitment already reveals
a full layer) catches 35.6% of the existing zero-hiding attack at a 5%
false-positive rate: real signal a per-neuron rule cannot have. But an
adversary who knows the plausibility model can solve, in closed form, for the
perturbation on a given support that flips the output while minimising added
reconstruction error. Reliable, manifold-invisible evasion needs support ≈13
— the same support the original attack already used — so an informed
adversary pays essentially nothing extra.

**The pattern, as it stands after review.** Trace-independent rules and the
joint/statistical check fail the same way (the floor sampler's verdict is now
open — see the retraction above): whatever looks like an
improvement against an *unaware* adversary stops looking like one once the
adversary is allowed to know the rule (Kerckhoffs — the standing threat model
throughout this document). We stopped short of a fully general proof covering
every conceivable locally-computable rule; the empirical pattern across three
structurally different families was consistent enough that we moved on to
Phase 2 (`phase2_sumcheck.md`) rather than chase it further.

Code: `scripts/run_targeted_vs_untargeted.py`, `scripts/run_theorem4_check.py`,
`scripts/run_joint_plausibility_check.py`, `scripts/run_manifold_evasion_check.py`.

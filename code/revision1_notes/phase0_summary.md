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

**1. The threat model itself was too easy (§8).** Every `|U|` in this document
is computed with "flip to *any* wrong class." A real backdoor needs a specific
target label. Under that threat model, at layer 1 (`N=512`), a target-forced
attacker's easiest choice of target has `|U|=159` rather than `465` — **3.1×**
more detectable under plain uniform sampling, for free, no new sampler:

| Threat model | mean `\|U\|` / 512 | uniform detection floor |
|---|---|---|
| Untargeted | 465 | 0.0021 |
| Targeted, attacker's easiest target | 159 | 0.0066 (3.1×) |
| Targeted, forced to one class | 20–69 | 0.015–0.053 (7–25×) |

**2. The natural patch to the zero blind spot does not work (§6).** Add a
floor `ε` to contribution weighting so a parent scores `|w_ij|·(|ã_i|+ε)`
instead of `|w_ij·ã_i|`. Swept `ε` from 0 to 100, checked the *exact* minimax
bound at every value (not the heuristic evasion search, which understated
this sampler's true worst case by ~15×). No `ε` ever beat uniform; it only
interpolates between the zero blind spot (`ε=0`) and the already-known
sub-uniform weight-magnitude sampler (`ε→∞`):

| `ε` | exact worst-case detection | vs. uniform's `0.00217` |
|---|---|---|
| 0 | 0.000000 | — |
| 1 | 0.000739 | 34% |
| 100 | 0.000974 | 45% |

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

**The pattern.** Trace-independent rules, the best implementable
trace-dependent rule (with or without a floor), and a structurally different
joint/statistical check all fail the same way: whatever looks like an
improvement against an *unaware* adversary stops looking like one once the
adversary is allowed to know the rule (Kerckhoffs — the standing threat model
throughout this document). We stopped short of a fully general proof covering
every conceivable locally-computable rule; the empirical pattern across three
structurally different families was consistent enough that we moved on to
Phase 2 (`phase2_sumcheck.md`) rather than chase it further.

Code: `scripts/run_targeted_vs_untargeted.py`, `scripts/run_theorem4_check.py`,
`scripts/run_joint_plausibility_check.py`, `scripts/run_manifold_evasion_check.py`.

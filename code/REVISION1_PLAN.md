# Revision 1: looking for a real defence

Scope for this revision, as decided: stay inside the sampling-protocol framework
(no transformer/attention support, no refereed-delegation implementation — both
deferred, not abandoned). The goal is either a working defence that measurably
beats the `k/|U|` ceiling of `DEFENCE_NOTES.md`, or a sharper, rigorously proven
impossibility result than the paper's own open suggestion — both count as success.

---

## Phase 0.5 — Targeted vs. untargeted attack success (do this first, no GPU needed)

**Status: done.** See `DEFENCE_NOTES.md` §7. Headline: a realistic (attacker
picks its cheapest target) targeted backdoor is ~3.1x more detectable than the
untargeted framing this whole document otherwise uses; the ceiling if the
target is externally fixed is 25x. No new sampler, no new theory — plain
uniform sampling, just measured against the right threat model.

Every `|U|` reported in `DEFENCE_NOTES.md` (e.g. "454 of 512 neurons admit a flip")
is computed with `target_class=None` in `_smallest_flipping_value` — the adversary
wins by flipping the prediction to *any* other class. That is the easiest version
of the attack. A real backdoor usually needs a *specific* target label (misclassify
stop signs as speed-limit signs, not "as anything"), which should shrink `U` and,
by Theorem 1/2, raise the achievable detection floor for the same sampling budget.

`target_class` is already a first-class parameter throughout `attacks/tamper.py`
and `defences/adaptive.py`, so this is a measurement task, not new engineering:

* Recompute `U_target(c)` for every candidate target class `c != honest_class`,
  alongside the existing `U_any = U_target(None)`.
* Report `|U_target(c)|` distribution (min/mean/max over `c`) against `|U_any|`,
  and the corresponding Theorem-1/2 bounds `k/|U_target(c)|`.
* Re-run the existing detection sweeps (naive, evasive, zero-blind-spot, minimax —
  everything in `scripts/run_defence.py`) once under `target_class=None` (already
  done, in `artifacts/results/defence.json`) and once under a fixed worst-case
  (easiest-for-attacker) target class, so both threat models are on record side by
  side.
* Same for the backdoor experiment (`attacks/backdoor.py`): report attack success
  rate and verifier acceptance for "flip to any wrong class" vs. "flip to this one
  attacker-chosen class."

This doesn't require a new sampler or new theory — it's due diligence on a threat
model the existing results quietly simplified. It also feeds Phase 0: Theorem 4
needs to be stated for whichever `U` definition we end up treating as the real
threat model, so this should land before the proof attempt below.

## Phase 0 — Close the open question before building anything

**Status: partially done.** The natural first patch — add a floor to
`LocalContributionSampler` so it doesn't vanish at zero — is closed off, both by
a clean argument and by exact computation. See `DEFENCE_NOTES.md` §6
(`ZeroAwareContributionSampler`, Theorem 4): sweeping the floor `ε` from 0 to
100 only interpolates between two already-known failure modes (exactly Theorem
3 at `ε=0`, converging to `StaticImportanceSampler`'s sub-uniform number as
`ε→∞`); no value of `ε` was found to beat uniform's exact minimax bound. The
fully general question — no locally-computable trace-dependent rule of *any*
shape can escape the dichotomy — remains open; what's closed is the specific,
most-obvious fix. The "cumulative path-sensitivity" idea below is still
untested and is the next candidate to check against the *exact* minimax
computation (not the heuristic search — see the methodological note in
`DEFENCE_NOTES.md` §6, which undersold `ZeroAwareContributionSampler`'s true
worst case by ~15x at `ε=100`).

`DEFENCE_NOTES.md` §5 ends with: *"we do not claim a bound covering every
conceivable adaptive rule."* Settle that, if possible, before inventing a new
sampler.

* Define the family of **locally-computable, trace-dependent samplers**: any rule
  that at each hop uses only the current node's opened weight row plus activations
  already opened along the path so far.
* Attempt **Theorem 4**: every rule in this family either (a) is expressible as
  trace-independent (so Theorem 2 already caps it), or (b) has some
  adversary-plantable degenerate certificate it assigns zero weight to
  (generalizing Theorem 3's zero blind spot).
* Concrete test case: a "cumulative path-sensitivity" sampler — weight a hop by
  the product of already-opened weights and downstream ReLU signs, *not*
  multiplied by the child's claimed value. Back-of-envelope: for a dense node this
  looks like it reduces to plain weight-magnitude weighting per hop (trace-
  independent, so Theorem 2 applies) because the "how much do I care about this
  node" scalar is identical across all of a node's parents and cancels on
  normalization. **Unverified** — needs a careful derivation, and may not hold once
  a neuron feeds multiple children in the same layer (conv weight sharing).

Deliverable either way: a written result — a proof (or documented failed attempt
with the obstruction identified), or a specific rule that provably escapes both
theorems.

## Phase 1 — Cheap candidates, stress-tested through the existing harness

Whatever Phase 0 finds, these are cheap enough to prototype and run through
`defences/adaptive.py`'s `EvasionSearch`, the same harness that broke
`LocalContributionSampler`:

1. The path-sensitivity sampler from Phase 0 — build it, measure it, don't just
   trust the derivation.
2. Covering-design multipath (sample *m* paths without replacement across a layer
   instead of i.i.d.). Theorem 1 already caps this at the same `k/|U|`, so expect
   only a constant-factor gain (~1.6x at best). Low priority — mostly closes a
   possible "unfair i.i.d. baseline" objection.
3. Free anomaly/typicality co-check layered on top of whatever sampler is chosen
   (same style as `expected_saliency_importance`): catches the *naive* tamper
   (activation ~28 vs. natural max ~8.2) for zero extra path budget. Does not touch
   the stealthy/zero-valued attack.

## Phase 2 — Structural fix (full stretch goal, confirmed in scope)

If Phase 0 confirms the impossibility for any sampling-only rule, the only way to
beat `k/|U|` without full SNARK cost is to change what one check verifies, not how
nodes are chosen: replace the per-node Merkle-row check with a **per-layer
random-linear-combination (sumcheck-style) check** — a random challenge vector `r`,
verify `Σ r_j·(ã_j − φ(Σ w_ij ã_i))` over an entire layer at once. By
Schwartz-Zippel, this catches an inconsistency *anywhere* in the layer with
overwhelming probability instead of `1/N`. This is a different cryptographic
primitive sitting between "Merkle row opening" and "full zkSNARK" — a new protocol
variant, not a smarter verifier for the existing one. Plan: prototype the numpy
arithmetic first (ideal random-oracle challenge, no actual succinct commitment) to
measure the soundness gain before investing in making it succinct.

## Phase 3 — GPU scale-out (validation, not discovery)

Once Phase 0–2 conclusions are settled at MNIST-MLP scale, use SLURM GPU access to:

* retrain at 5–10x width/depth and re-run the same experiments to confirm the
  theorems' predictions hold size-independently;
* multi-seed repeats for error bars on floor/threshold calibration;
* add the CIFAR-10 CNN promised in the intermediate report but never built.

This phase validates generalization; it isn't expected to change the theory.

---

## Open risk notes

* Phase 0's proof attempt has no guaranteed payoff and could consume significant
  time for an inconclusive result. Timebox it.
* The path-sensitivity-reduces-to-weight-magnitude claim in Phase 0 is an
  unverified derivation, not a checked result.
* Phase 2 changes the protocol's commitment scheme, which is a bigger claim than
  "a better verifier heuristic" — keep that framing distinction explicit in
  whatever gets written up.
* None of Phase 0–2 addresses trigger rarity (`ρ→0`: a verifier can't catch
  behavior it never observes). That remains a known, out-of-scope limitation of
  any inference-time verifier, sampling-based or not.

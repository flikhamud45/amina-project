# Why importance-weighted path sampling cannot fix this

Section 5.2 of Anchuri et al. concedes the `1/N` ceiling and offers two ways out:
sample more paths, and "consider adaptive sampling strategies that prioritize
layers or nodes where activations are statistically more sensitive to tampering",
adding that the authors "have some partial results from this approach".

This note works that suggestion out. The conclusion is negative and, for the class
of defences that a real verifier can actually run, provable: **uniform sampling is
minimax-optimal, and the natural "sample where it matters" refinements are not
merely no better but strictly worse.**

Throughout, `N` is the width of the tampered layer, `S` the set of nodes at which
the committed trace violates its local relation, and `m` the number of paths opened
per query. Every claim is checked empirically in `scripts/run_defence.py`; the
numbers quoted are from `artifacts/results/defence.json`.

---

## 1. Two facts the whole analysis rests on

**Proposition 1 (uniqueness).** *If the input layer equals `qry` and every local
relation `a_j = φ(Σ_{i∈G_j} w_ij a_i)` holds, then the trace is `EvalTrace(M, qry)`.*

Immediate by induction up the layers: layer 0 is pinned, and each layer is a
function of the one below. Consequently **any** trace claiming an output other than
`M(qry)` violates a local relation somewhere — `S ≠ ∅`. No adversary escapes this,
which is why the paper's forging attacks (Section 7, Appendices E and F) cannot
succeed by driving the violation to zero.

**Proposition 2 (exact detection).** *A verifier whose accept predicate is the
conjunction of the local checks at the nodes its paths visit rejects if and only if
some path visits a node in `S`.*

So detection is `1 − (1 − Pr[one path meets S])^m`, and the adversary's problem is
purely combinatorial: make `Pr[path meets S]` small. Magnitude is irrelevant — only
*where* `S` sits and *how the verifier looks*.

This reframing is the crux. The paper scores adversaries by separation magnitude
(Equation 1); that quantity does not appear in Proposition 2. Reproducing the
paper's own attacks makes the mismatch concrete: gradient reconstruction achieves
the *smallest* separation of any attack we ran (0.30 against 0.89 for inverse
transform) and is detected with probability 1, because it perturbs all 778 nodes.

---

## 2. The defence class a verifier can actually implement

Not every "adaptive sampling" rule is available. The verifier holds the public
architecture, the digest `C_M`, the query, and whatever the prover opens. It does
**not** hold the model.

That rules out the most obvious reading straight away. Weighting neuron `i` by
`|∂y/∂a_i|` requires every weight of every layer above `i`. A verifier able to form
those gradients already holds `M` — and would simply run the inference itself,
making the protocol pointless. We implement `GradientSaliencySampler` anyway, as an
upper bound on what the family could achieve given information no verifier has.

Two readings survive:

* **Trace-independent** (`StaticImportanceSampler`): weights published with `C_M`,
  computed once from the model — outgoing-weight magnitude, or mean saliency over a
  public calibration set. Verifiable by anyone holding the model, and fixed.
* **Locally computable** (`LocalContributionSampler`): step from `j` to parent `i`
  with probability proportional to `|w_ij · ã_i|`. The verifier already holds `j`'s
  opened weight row and the claimed parent activations, so this costs nothing extra.
  This is, as far as we can tell, the strongest implementable version of the paper's
  suggestion.

---

## 3. Trace-independent weighting: never better than uniform

**Theorem 1 (budget bound).** *Let the verifier check at most `k` nodes in layer `ℓ`,
by any randomised rule whose choice is independent of the claimed trace. Let `U` be
the set of neurons at which a single-node tamper achieves the adversary's goal. If
the adversary picks `v` uniformly from `U`, detection is at most `k/|U|`.*

*Proof.* Let `S` be the verifier's checked set, `|S| ≤ k`, with `S ⊥ v`. Then
`Pr[detect] = E_v[Pr[v ∈ S]] = (1/|U|) Σ_{v∈U} Pr[v ∈ S] ≤ E[|S|]/|U| ≤ k/|U|`. ∎

**Theorem 2 (uniform is minimax-optimal).** *Let `q` be the visit distribution a
trace-independent sampler induces on layer `ℓ`, known to the adversary. The
adversary tampers at `argmin_{v∈U} q(v)`, so detection is `min_{v∈U} q(v) ≤ 1/|U|`,
with equality iff `q` is uniform on `U`. Uniform sampling attains the bound.*

*Proof.* A minimum is at most an average: `min_{v∈U} q(v) ≤ (1/|U|) Σ_{v∈U} q(v) ≤ 1/|U|`,
since `q` sums to at most 1. Equality in the first step forces `q` constant on `U`. ∎

So concentrating attention on "important" neurons cannot help against an adversary
who reads the published weights, and strictly hurts whenever the weighting is
non-uniform over the neurons that are usable. This is the concern that motivated the
whole idea — *can the attacker steer where we look?* — and the answer is that it
does not even need to steer: it only needs to read.

Measured, at layer 1 (`N = 512`, uniform detection `1/512 = 0.00195`):

| Sampler | vs. naive attack | vs. adaptive attack |
|---|---|---|
| uniform | 0.00195 | 0.00195 |
| static importance | 0.00437 | **0.00098** |
| gradient saliency (not implementable) | 0.00586 | **0.00013** |

Both look like improvements against an adversary that ignores them, and both fall
*below* uniform once the adversary picks its neuron knowing the rule — gradient
saliency by a factor of 15.

---

## 4. Contribution weighting: a blind spot at zero

`LocalContributionSampler` is genuinely effective against the naive attack, and by a
wide margin: detection 0.126 against uniform's 0.00195, a factor of 64. The reason
is sound. The naive single-neuron tamper writes a large value, and under
contribution weighting a large value *is* a large sampling weight; the walk is drawn
straight to it.

It has one flaw, and it is fatal.

**Theorem 3 (zero blind spot).** *Let the sampler step from `j` to parent `i` with
probability proportional to `g(|w_ij · ã_i|)` for any `g` with `g(0) = 0`. If the
adversary sets `ã_v = 0` for every `v ∈ S`, then every transition into every node of
`S` has weight zero, so `Pr[path meets S] = 0` and detection is exactly zero — for
any `|S|` and any number of paths `m`.*

A weighting that ignores what claims nothing cannot see a claim of nothing.

The adversary needs the output to flip when a set of neurons is zeroed, and in a
ReLU network that is easy: zeroing a neuron removes its contribution, and removing
enough contributions to the winning class flips the decision. Measured at layer 1:

* **13 of 512** neurons set to exactly zero flips the prediction (class 7 → 3);
* acceptance under contribution weighting is **1.000000** — the real protocol
  accepted **3000 of 3000** runs;
* acceptance under *uniform* sampling is 0.9746, i.e. detection 0.0254.

So the refinement turns a 2.5% chance of catching this adversary into **no chance at
all**. It is not a marginal loss; it is the difference between a bound and none.

The value is also unremarkable. Those neurons are naturally exactly zero on **68.8%**
of natural inputs — ReLU sparsity makes zero the modal activation. An anomaly check
on activation ranges or marginal statistics sees nothing, because zero is the single
most ordinary value a ReLU neuron takes.

**What partially rescues it.** Mixing a uniform component of mass `ε` into the
sampler restores detection `ε·|S|/N`, since zeroed nodes regain weight `ε/N`. The
floor is therefore not optional. But the defence is then at best `ε`-scaled uniform
sampling on the adversary's chosen support, and Theorem 2 bounds the rest. Our
`floor` parameter implements this, and it is why we recommend it be non-zero if such
a sampler is deployed at all.

---

## 5. Why no sublinear verifier of this shape can work

Combining the pieces:

**Corollary.** *Any verifier that (i) decides by conjoining local consistency checks,
and (ii) checks at most `k` nodes of a layer of width `N` by a trace-independent
rule, detects a single-node tamper with probability at most `k/|U|`, where `U` is the
set of neurons at which a tamper flips the output. Constant detection therefore
requires `k = Ω(|U|)`.*

In our models `|U| = N`: at layer 1 every one of the 512 neurons admits a flipping
value, and the same holds at layer 2. So constant detection needs a number of
weight-row openings **linear in the layer width** — precisely the cost the protocol
exists to avoid. Efficiency and backdoor-robustness are not both available here.

The measured evasion curve says the same thing in practice. Going from 1 path to 250
— a 250-fold increase in proof size and verifier work — moves acceptance only from
0.998 to 0.613, and 250 paths already open 750 weight rows against a layer of 512.

Trace-*dependent* rules escape Theorem 2's hypothesis, but only by handing the prover
influence over its own examination; Theorem 3 is that trade taken to its conclusion.
We do not claim a bound covering every conceivable adaptive rule. We claim that the
two implementable readings of the paper's suggestion both fail, one provably and one
catastrophically, and that the reason is structural rather than incidental.

---

## 6. Revision 1: smoothing the zero blind spot does not escape it

Section 5's corollary bounds *trace-independent* rules. Section 4 breaks one
specific *trace-dependent* rule. The gap between them — whether some other
trace-dependent, locally-computable rule escapes both — was left open: "we do not
claim a bound covering every conceivable adaptive rule."

The most natural attempt to close that gap is to patch `LocalContributionSampler`
directly: instead of scoring a parent `i` by `|w_ij · ã_i|` (which is exactly zero
when `ã_i = 0`), add a floor so it never is. `ZeroAwareContributionSampler` scores
`i` by `|w_ij| · (|ã_i| + ε)` for a tunable `ε ≥ 0`. At `ε = 0` this is
`LocalContributionSampler`. As `ε → ∞` the additive term swamps `|ã_i|` and the
rule converges to pure `|w_ij|` weighting — a *trace-independent* rule, since it no
longer depends on the claimed trace at all.

That convergence is the whole story. **Theorem 4 (no free floor).** *For every
`ε ≥ 0`, an adversary who knows `ε` can tamper a single node with detection
probability strictly below uniform's `1/|U|` bound*, by choosing, among the
neurons in `U`, the one feeding the checked node with the smallest weight
magnitude — exactly the neuron a weight-magnitude sampler is worst at catching.
At `ε = 0` this is Theorem 3 (detection exactly `0`). As `ε → ∞` it converges to
`StaticImportanceSampler`'s exact numbers (§3: `0.00098` against an adaptive
attacker, versus uniform's `0.00195`). There is no `ε` in between that does
better than both endpoints, because the induced visit distribution moves
*monotonically* between them as `ε` grows — it never has a chance to overshoot
uniform along the way.

**Measured** (`mlp_mnist_full`, layer 1, width 512, `|U| = 462`, uniform bound
`1/|U| = 0.002165`), using the *exact* minimax computation of §3 — not the
heuristic multi-node search of §4 — so the numbers below are provable worst
cases, not merely observed ones:

| `ε` | exact `min_v q(v)` over `U` | vs. uniform's `0.002165` |
|---|---|---|
| 0 | 0.000000 | — (Theorem 3) |
| 0.01 | 0.000036 | 1.7% of uniform |
| 0.1 | 0.000262 | 12% of uniform |
| 1 | 0.000739 | 34% of uniform |
| 10 | 0.000938 | 43% of uniform |
| 100 | 0.000974 | 45% of uniform |

Detection climbs monotonically but saturates at under half of uniform's floor,
matching `StaticImportanceSampler`'s known number in the limit. No value of `ε`
was found that beats uniform, and the convergent limit is a case Theorem 2
already forbids from beating it.

A methodological note worth recording: the heuristic multi-node search
(`plan_adaptive_stealthy_flip`, three fixed neuron-ranking heuristics) *understates*
how bad this sampler's worst case is — at `ε = 100` it only found `0.014` detection,
making the sampler look better than it is, because its ranking heuristics were
tuned against contribution-style attacks and don't specifically target "smallest
raw weight," which is what actually breaks a near-weight-magnitude rule. The exact
single-neuron computation (`visit_probabilities_under` + `_smallest_flipping_value`,
Theorem 1/2's own method) is what settled the question. **Any new candidate
sampler should be checked against the exact minimax computation first — the
heuristic search is a demonstration tool for known attacks, not an adversarial
optimality certificate.**

**Scope of this result.** This closes off the single most natural "just fix the
zero" patch, and the argument generalizes informally: any score built as
`|w_ij| · φ(ã_i)` for a non-negative `φ` reduces, in the adversary's chosen
`ε → φ`'s-infimum limit, to a trace-independent rule that Theorem 2 already
bounds below uniform whenever the weights aren't perfectly flat — which they
never are. It is not a formal proof that *no* locally-computable trace-dependent
rule of any shape can ever escape the Theorem 2/3 dichotomy; that general
question is still open. It does mean the natural, cheap fixes are exhausted:
getting past this requires either accepting the reduced backdoor threat model
of §7 below (a specific target, not merely "any wrong output") or changing what
a single check verifies, not how neurons are weighted (see the sumcheck-style
direction in `REVISION1_PLAN.md`, Phase 2).

Code: `ZeroAwareContributionSampler` in `src/pvi/defences/sampling.py`.
Reproduce with `scripts/run_theorem4_check.py`.

---

## 7. Revision 1: a real backdoor needs a specific target, and that already helps

Every `|U|` in this document — the set of neurons at which a single-node tamper
"achieves the adversary's goal" — is computed with `target_class=None` in
`_smallest_flipping_value`: the adversary wins by flipping the prediction to
*any* other class. That is the easiest version of the attack, and it is not what
a real backdoor needs. A backdoor that is useful to an attacker demands a
*specific* output (misclassify stop signs as speed-limit signs, not "as
anything"), and `target_class` is already wired through `attacks/tamper.py` and
`defences/adaptive.py` to measure exactly this.

**Measured** (`mlp_mnist_full`, layer 1, width 512, 30 queries, exact `|U|` per
query per candidate target):

| Threat model | mean `\|U\|` / 512 | uniform single-path bound `1/\|U\|` | vs. untargeted |
|---|---|---|---|
| Untargeted (flip to anything) | 465 (91%) | 0.0021 | baseline |
| Targeted, attacker's easiest target | 159 (31%) | 0.0066 | 3.1x |
| Targeted, averaged over all 9 targets | 69 (13%) | 0.0146 | 6.8x |
| Targeted, attacker's hardest target | 20 (4%) | 0.0532 | 25x |

A realistic adversary is not forced into a target — it picks whichever is
cheapest, so **3.1x is the fair number against an adaptive attacker**; the 25x
figure is the ceiling of what target-forcing buys, only realized if the
attacker's goal is externally fixed rather than "any output that benefits it."

This costs nothing: no new sampler, no new theory, plain uniform sampling. It is
also not a coincidence that it was left unmeasured — the untargeted framing is
the *paper's own* choice of threat model (Section 5.2's `1/N` figure), and it is
the hardest case for the verifier by construction. Sanity check: the existing
backdoor experiment (`scripts/run_attack.py`) already fixes `TARGET_CLASS = 0`
for its headline numbers; that target's mean `|U| ≈ 45` sits *below* the
cross-target average of `≈ 69`, so those numbers were not quietly cherry-picked
easy.

Code: `scripts/run_targeted_vs_untargeted.py`;
raw data in `artifacts/results/targeted_vs_untargeted.json`.

---

## 8. What would actually help

None of this touches the protocol's real guarantee. Against a prover that swaps in a
*different model*, `RandPathTest` works — we measured 100% detection against three
substitutes and 99.8% against an 8-bit quantisation that agrees with `M` on every
test prediction. The scheme does what it says.

The gap is between what it certifies and what "verified inference" is heard to mean.
It certifies that a committed model was faithfully executed *at the nodes that were
checked*. It does not certify that the whole computation was faithful, and it cannot
certify anything about whether the committed model is benign.

Directions that address the gap rather than the symptom:

* **Exact-binding proofs.** A zkSNARK over the full circuit removes the sampling gap
  entirely — at the prover cost the paper set out to avoid. That trade-off is the
  honest framing: the orders-of-magnitude speed-up is paid for in soundness against
  sparse tampering, not obtained for free.
* **Refereed delegation (Appendix D).** Unaffected by any of this. Bisection isolates
  the *first* disagreeing node in `O(log n)` rounds, so a single-node tamper is found
  with certainty rather than probability `1/N`. It needs two servers with at least one
  honest — a stronger assumption, but it converts our attack from near-certain success
  to certain failure. **We regard this as the most promising direction in the paper.**
* **Verifiable training or model attestation.** Backdoor-freedom is a property of how
  the model was produced. No inference-time check on a committed model can establish
  it, because the committed model may itself be backdoored — which is not an attack on
  the protocol at all, merely a limit on what it means.

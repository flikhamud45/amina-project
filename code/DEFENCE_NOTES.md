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

## 6. Revision 1: the epsilon-floor sampler — RETRACTED and re-measured

**This section previously stated a "Theorem 4 (no free floor)" claiming that no
`ε` beats uniform, backed by a table of "exact minimax" worst cases described as
"provable worst cases, not merely observed ones". The table measured the wrong
thing, the claim is not supported, and on re-measurement it looks false. Both
the theorem and its methodological footnote are withdrawn.**

### The bug

`ZeroAwareContributionSampler` scores a parent by `|w_ij| · (|ã_i| + ε)`. The
retracted table computed

```python
visits = visit_probabilities_under(network, honest_trace, layer, sampler)
```

and took `min` over `U`. That equals detection **only for samplers that ignore
the claimed trace** (uniform, static importance) — which is exactly why our own
test of that identity,
`test_detection_equals_visit_probability_for_trace_independent_samplers`, is
restricted to those two. This sampler *reads the claimed activations*. When the
adversary tampers neuron `v`, it is the **tampered** trace that decides how
often the walk visits `v`, and a single-neuron flip has to *raise* the
activation — precisely what contribution weighting is drawn to.

Measured both ways (same model, same query, `|U| = 462`):

| `ε` | retracted number (honest trace) | actual worst case (tampered trace) |
|---|---|---|
| 0 | 0.000000 | **0.118938** |
| 0.01 | 0.000036 | **0.114887** |
| 0.1 | 0.000262 | **0.088086** |
| 1 | 0.000739 | **0.027525** |
| 100 | 0.000974 | **0.001747** |

The script's own neighbouring column already contradicted the table — it read
0.11 at `ε = 0` — and that discrepancy was wrongly explained away as the
heuristic search "understating the worst case by ~15x". It was the exact block
that was wrong. That footnote is withdrawn too.

### What the corrected measurement says

Re-measured with one consistent script (`scripts/run_theorem4_check.py`, now
rewritten): detection is computed on the **tampered** trace, and for each
sampler the adversary takes the better of two attack families — the exact
single-neuron flip (every `v ∈ U`) and the multi-neuron adaptive/zeroing search
run against that sampler. Per-query worst case, then averaged (5 queries,
layer 1):

| sampler | single-neuron | multi-neuron | **adversary's best** | vs. uniform |
|---|---|---|---|---|
| uniform | 0.001953 | 0.006641 | **0.001953** | — |
| `ε = 0` | 0.099370 | 0.000000 | **0.000000** | attacker wins (Theorem 3) |
| `ε = 0.01` | 0.096134 | 0.001646 | **0.001646** | 0.8x |
| **`ε = 0.1`** | 0.074686 | 0.011420 | **0.011420** | **5.8x** |
| **`ε = 1`** | 0.024487 | 0.028409 | **0.023276** | **11.9x** |
| **`ε = 10`** | 0.004565 | 0.017266 | **0.004565** | **2.3x** |
| `ε = 100` | 0.001711 | 0.012327 | **0.001711** | 0.9x |

Uniform's own worst case is the single-neuron attack at `1/N = 0.00195`; the
zeroing attack against uniform is *easier* to catch (0.0066), so the adversary
does not choose it.

**`ε` in the 0.1–10 range beats uniform, by up to ~12x at `ε = 1`** — the exact
opposite of the retracted claim. Theorem 3's `ε = 0` catastrophe is unaffected
and still holds; what is now clear is that the catastrophe is specific to
`ε = 0`, and that a small floor does not merely patch it but overtakes uniform.

Note the two failure modes pulling in opposite directions, which is what makes
a middle value work: small `ε` leaves the zeroing attack cheap (multi-neuron
column), large `ε` washes out the contribution signal that catches a raised
activation (single-neuron column). The optimum sits where neither attack is
cheap.

### How much to trust this

Less than the batched-check numbers, and the asymmetry matters.

*Solid*: the retraction itself. The original "no `ε` beats uniform" conclusion
was produced by measuring the honest trace, and it does not survive correction.
The single-neuron column is exact — every `v ∈ U`, acceptance computed by the
same dynamic program the protocol uses — so the *upper* half of each row is not
in doubt.

*Provisional*: the multi-neuron column, and therefore the "adversary's best"
column that the headline rests on. It comes from a heuristic search
(`plan_adaptive_stealthy_flip`) over 5 queries. A stronger adversary could pull
the middle-`ε` numbers down, and there is an obvious candidate nobody has
tried: **mixtures** — zero the neurons with small outgoing weights (exposure
`ε·|w|` each) while raising one neuron just enough, so neither pure family's
weakness applies. Until that is run, "`ε = 1` beats uniform by 12x" should be
read as *not yet refuted* rather than established.

Still to do: run ≥100 queries, add the mixture attack, report per-query worst
cases rather than means, and repeat at width 4096 (`phase3_gpu.md` carries the
same flaw). `scripts/run_theorem4_check.py` has been rewritten to measure on
tampered traces and to report the adversary's best across families.

---

## 7. Revision 1: a joint plausibility check does not survive an adaptive adversary either

Sections 3, 4 and 6 all score a candidate parent using only *that neuron's own*
claimed value and weight. A genuinely different mechanism: check the *joint*
claimed activation vector of a whole layer against a plausibility model
calibrated on many honest executions — e.g. a low-rank PCA subspace fit to
natural layer activations, flagging large reconstruction error. This costs
nothing extra to open (`C_trc` commits one leaf per *layer*, not per neuron, so
the whole vector is already revealed the moment any node in it is checked — see
§2 of `SPEC_NOTES.md`), and it can see structure no per-neuron rule can, in
principle: a natural zero and a malicious zero are the same value in isolation,
but the *rest of the layer* might make one of them jointly implausible.

**First look, unaware adversary.** A rank-50 PCA on layer 1 (86.6% of variance),
calibrated on 5000 natural queries, scored against the *existing* zero-hiding
attack from §4 (which was built to evade `LocalContributionSampler`, not this):
honest reconstruction error has mean `3.51` (p95 `5.87`); the forged traces have
mean `4.64`. At the honest p95 threshold this catches **35.6%** of forgeries
for a 5% false-positive rate on honest traces — real signal, and a kind of
signal Theorem 4's per-neuron argument says cannot exist for a marginal check.

**The adaptive counter is immediate on paper**: a perturbation confined to the
PCA subspace produces *zero* extra reconstruction error, however large, by
construction. Whether this defeats the check in practice depends on how much
*support* (how many neurons) confining a flip to the subspace costs — a wide,
dense perturbation is trivially caught by ordinary path sampling regardless of
manifold membership (Theorem 1: detection ~ `k/N`), so the real question is
whether there is a support size small enough to stay cheap under path sampling
*and* close enough to the manifold to stay invisible to the PCA check.

**Measured** (`scripts/run_manifold_evasion_check.py`: for each support size
`k`, pick the `k` neurons ranked by margin sensitivity, solve the exact
closed-form constrained least-squares perturbation on just those coordinates
that minimises added reconstruction error subject to flipping the (linearised)
margin, verify against the real nonlinear forward pass, grow the target margin
until it actually flips):

| support `k` | flip rate | mean reconstruction error | vs. honest p95 (`5.87`) |
|---|---|---|---|
| 1 | 40% | 94.5 | far above |
| 3 | 45% | 6.4 | above |
| 8 | 95% | 5.8 | above |
| **13** | **100%** | **5.1** | **below** |
| 20 | 100% | 4.6 | below |
| 50 | 100% | 3.9 | below |

Reliable, manifold-invisible evasion needs support `≈13` — **the same support
the original, PCA-unaware zero-hiding attack already uses**, for an unrelated
reason (enough zeroed neurons to flip the prediction combinatorially, §4). An
adversary who knows about the PCA check pays essentially nothing extra: support
13–20 keeps uniform path-sampling exposure at `k/N ≈ 0.025–0.04`, the same order
of magnitude as before. The 35.6% catch rate measured against the unaware
attack does not survive contact with an adversary that knows the check exists.

**Conclusion.** A genuinely different mechanism (joint, not per-neuron) does
carry real information a marginal check cannot have — but "real information
against an unaware adversary" and "raises the cost of an informed one" are
different claims, and only the second matters under this document's threat
model (Kerckhoffs: the check is public). Here they came apart at a support
size the adversary was going to use anyway. This does not prove no joint check
could ever help — a higher-rank or nonlinear plausibility model might force a
larger crossover support — but the burden has shifted: it needs to be
demonstrated against an adversary that optimises against the specific
published model, not just measured against a differently-motivated attack.

Code: `scripts/run_joint_plausibility_check.py`,
`scripts/run_manifold_evasion_check.py`.

---

## 8. Revision 1: targeting shrinks `|U|`, but that is *not* a detection gain

**This section previously claimed a "3.1x more detectable under plain uniform
sampling" result. That claim was wrong and is retracted.** It is kept here,
corrected, because the mistake is instructive.

Every `|U|` in this document — the set of neurons at which a single-node tamper
achieves the adversary's goal — is computed with `target_class=None`: the
adversary wins by flipping to *any* other class. A real backdoor usually needs a
*specific* target, and measuring that does shrink `|U|` substantially:

| Threat model | mean `\|U\|` / 512 |
|---|---|
| Untargeted (flip to anything) | 465 (91%) |
| Targeted, attacker's easiest target | 159 (31%) |
| Targeted, averaged over all 9 targets | 69 (13%) |
| Targeted, attacker's hardest target | 20 (4%) |

**The error was reporting `1/|U|` as though it were uniform sampling's
detection.** It is not. Under uniform path sampling a single-node tamper is
detected exactly when the path visits that node, which happens with probability
`1/N` — the width of the layer — *regardless of how many other neurons could
have been used*. The adversary simply picks one of the 159.

Measured directly (15 queries, layer 1, detection via `acceptance_probability`
on the forged trace):

```
untargeted: mean uniform detection 0.001953   (1/N = 1/512 = 0.001953)
targeted  : mean uniform detection 0.001953   (1/N = 1/512 = 0.001953)
```

Identical, as the existing test
`test_single_neuron_acceptance_is_one_minus_one_over_width` already implies.

`1/|U|` is a different quantity: by Theorem 1 it is the *ceiling* for any
verifier whose choice ignores the claimed trace. Reaching it would require
concentrating samples on `U`, which no verifier can do — `U` depends on the
weights, the query and the target, and the verifier holds only a digest.

**What is actually true.** Targeting costs the attacker something only when it
must *also* keep forged activations inside natural ranges (Step 3's stealthy
attack): hitting one specific class then needs more tampered neurons, and more
tampered neurons means more exposure. From `run_attack.py`'s stealth block
(layer 1, target class 0):

| Activation cap | Untargeted: neurons / detection | Targeted: neurons / detection | Ratio |
|---|---|---|---|
| natural max | 2.9 / 0.0057 | 7.1 / 0.0139 | 2.4x |
| 99th percentile | 4.2 / 0.0081 | 11.5 / 0.0225 | 2.8x |
| 95th percentile | 5.6 / 0.0110 | 15.0 / 0.0293 | 2.7x |

So: against plain `RandPathTest` a targeted backdoor is **exactly as cheap as an
untargeted one** (`1/N`). Only if the verifier *also* range-checks activations
does targeting cost ~2.4–2.8x in tampered neurons, and hence in detection. That
is a reinterpretation of Step 3's existing stealth table, not a new result.

Code: `scripts/run_targeted_vs_untargeted.py` (reports `|U|`; the column is
labelled as an unreachable ceiling, not as uniform's detection).

---

## 9. Revision 1: how much SNARK machinery is actually needed

Sections 3–8 are consistent on one point: no way of choosing *which* nodes to
check, however clever, escapes the `k/N`-shaped ceiling once the adversary
knows the rule (§5's corollary). The only lever left is to change *what one
check verifies*. This section prototypes the cheapest version of that: batch
an entire layer's local relations into one scalar instead of sampling among
them.

**The check.** For layer `ℓ` with claimed activations `ã` and real weights
`w` (both already needed to check even one node), draw a random challenge
vector `r` over the whole layer and compute

```
C = sum_j r_j * ( ã_j - phi( sum_i w_ij ã_i ) )
```

`C` is exactly zero if every local relation in the layer holds. If even one
node's relation is violated, the residual vector is nonzero, and a uniformly
random linear functional of a fixed nonzero vector is zero with probability 0
(continuous challenges) or `≤ 1/|F|` (challenges from a size-`|F|` field) —
elementary linear algebra, nothing like Schwartz–Zippel's polynomial-degree
argument is needed once `φ` has already been applied and the residual reduced
to a single number per node.

**Measured** (`scripts/run_sumcheck_prototype.py`, layer 1, width 512, honest
float-noise floor from `SPEC_NOTES.md` §2 calibrated with 10x headroom):

| Trace | batched per-layer check | best sampling-based check |
|---|---|---|
| Honest | completeness 1.000000 | — |
| Naive single-neuron tamper | **1.000000** | uniform: 0.001953 |
| 13-neuron zero-hiding backdoor (§4's exploit, `0.000000` under contribution weighting) | **1.000000** | uniform: 0.025781 |

Both attacks caught with certainty (up to float-noise headroom), independent
of `N` and independent of where in the layer the inconsistency sits — including
the one attack (§4) that reduces every value-based sampler tried in this
document to exactly zero.

**What this does and does not show.** This measures the *soundness* gain,
assuming a succinct proof for the value of `C` already exists. It does not
build one. As implemented, computing `C` requires the verifier to hold every
weight and every claimed activation of the layer — exactly the same
information as checking all `N` nodes individually and taking the logical AND.
On its own this buys *nothing* over the naive "check everything" scheme it
resembles; the entire value depends on a genuine succinct argument that lets
the verifier accept `C`'s value in `O(log N)` or `O(1)` communication instead
of `N` weight-row openings.

That argument is not exotic — it is the same primitive real zkML systems use
for a layer's affine part (a sumcheck protocol over the multilinear extensions
of `w` and `ã`, e.g. GKR-style), with the ReLU handled by a separate lookup
argument over valid `(z_j, a_j)` pairs (exactly the technique behind the
Jolt zkVM cited in our own related work, Anchuri et al.'s footnote 2's
"not zero-knowledge" caveat notwithstanding). That is precisely the
prover-cost machinery the paper set out to avoid by sampling — Phase 2's real
question was never "does batching help" (it obviously does, trivially) but
"is there a middle ground between full per-layer SNARK cost and `1/N`
sampling." The natural hybrid, not built here: keep `RandPathTest`'s *outer*
structure — a random path visits `O(depth)` layers, not all of them — but
upgrade the check performed at each *visited* layer from "one sampled node"
to "one succinct whole-layer sumcheck." This removes the within-layer `1/N`
gap Theorem 1's corollary shows no sampling rule can close, while keeping the
protocol's central efficiency claim (sublinear in the network, not the full
GKR cost of an end-to-end proof). Building and costing that hybrid is real
follow-on work, not something a numpy prototype settles.

Code: `scripts/run_sumcheck_prototype.py`.

**Update: it was built.** See §10 — the assumption this section leans on (that
a succinct opening exists) turned out to be satisfiable with the Merkle
primitive already in this repository, and the resulting protocol catches every
attack in this document with probability 1.

---

## 10. Revision 1: the defence, built and measured

§9's prototype assumed the verifier could see the weights. `protocol/batched.py`
removes that assumption: a Ligero-style commitment (Reed-Solomon encode the rows
of `A = [W | b]`, Merkle-commit the columns) lets the verifier check one
evaluation `chi^T A` for a challenge vector it picks *after* the prover is
committed, spot-checking `t` random columns. No pairings, no new primitive —
the same Merkle tree the paper's own §8.1 uses.

**The subtlety that matters.** Batching a layer into one random linear
combination verifies complementarity `a_j(a_j - z_j) == 0`, and that constraint
is *vacuous exactly where `a_j == 0`*. §4's zero blind spot reappears verbatim
in arithmetic form: the zero-hiding backdoor would walk straight through a naive
batched check. Closing it needs a third constraint — `z_j <= 0` on the zero set
— which the prover supplies as a plaintext witness `y_j := -z_j >= 0` that the
verifier range-checks for free and then ties back to the committed weights.
Complementarity and the sign constraint fold into a single opening.

**Measured** against this repository's own attack planners (`mlp_mnist_full`,
layer 1, 40 queries):

| Attack | batched check | uniform | contribution weighting |
|---|---|---|---|
| naive single-neuron flip | **1.000000** | 0.001953 | 0.112636 |
| stealthy envelope-confined | **1.000000** | 0.004958 | 0.150712 |
| **zero-hiding backdoor** (§4's exploit) | **1.000000** | 0.023730 | **0.000000** |

Honest completeness is **exactly** `1.000000` — the exact field arithmetic also
retires the `1e-4` tolerance of `SPEC_NOTES.md` §2, and with it the free
sub-tolerance perturbation budget that tolerance handed the adversary.

Re-run at width 4096 (8x), detection is unchanged at `1.000000` for both
attacks while uniform sampling's degrades from `0.0019` to `0.00024` — the
`1/N` ceiling is visible in one row and absent in the other.

**Cost, honestly**: proof size ~115 kB at width 512 against the sampling
scheme's ~13 kB (≈9x), growing as `O(t·N)`. Since the existing scheme already
reveals a whole layer's activations whenever it touches that layer (`C_trc` is
one leaf per layer), this is a constant factor on an existing `O(N)`, not a new
asymptotic class, and "visit `O(depth)` layers" is untouched.

**Whole-network coverage (added after review).** An earlier version of this
section checked one hidden layer in isolation, which proves little, and did not
support the output layer at all: `fixed_point_layer` clamped with ReLU, so an
honest prover emitting real logits (~75% of them negative) was rejected
**0/20**. Both are fixed. Identity layers use the linear constraint `a == z`
(no zero set, no sign witness, no range check); the verifier recomputes each
layer's input from the previous layer's *verified* output via a public rounding
rule, so it never takes an input from the prover; and each layer's challenges
are bound to its index and a running hash of earlier layers. Measured end to
end on all three layers: honest network accepted 10/10, tampers at layers 1, 2
and 3 caught 10/10 each, a prover breaking the inter-layer chain rejected
10/10. Whole-network proof 188.9 kB, prover 7.4 ms, verifier 9.4 ms, against a
2.04 MB model — 11.1x smaller than downloading the weights. Note this is then
no longer a *sampling* protocol: with every layer checked, the honest
comparisons are the model download and a SNARK, not the 13 kB sampling proof.

**Scope**: dense layers only (convolutions share weights and need a different
layout; max-pooling is not an affine-plus-activation relation); honest
committer assumed, matching this document's threat model; and **the Fiat–Shamir
parameters are too small** — a 26-bit field with 24 spot-checks is ~`2^-24` per
check, which a *grinding* prover can brute-force by re-randomising its message
and re-hashing. That is fine with verifier-sent challenges (as in the base
protocol) but not with hash-derived ones. Fixing it means verifier-sent
challenges, more repetitions, or a ~64-bit prime; none is done yet.

**One correction on the record**: the first implementation derived the
spot-checked column indices from `(digest, x, a, y)` but not from `u`, letting a
cheating prover read the columns off in advance and solve for a passing `u` (25
linear constraints on 785 unknowns). The forgery was implemented and accepted;
fixed by a two-phase transcript that derives the columns from `u` as well.
Regression test: `test_adaptive_transcript_forgery_is_rejected`. The
measurement script had not caught it — it only runs the *honest* prover on
tampered traces, never a prover that cheats on the proof.

Full write-up: `revision1_notes/phase4_batched_defence.md`; a
from-first-principles explanation of every primitive involved is in
`revision1_notes/defence_explained.md`. Code: `src/pvi/protocol/batched.py`,
`tests/test_batched.py`, `scripts/run_batched_defence.py`.

---

## 11. What would actually help

None of this touches the protocol's real guarantee. Against a prover that swaps in a
*different model*, `RandPathTest` works — we measured 100% detection against three
substitutes and 99.8% against an 8-bit quantisation that agrees with `M` on every
test prediction. The scheme does what it says.

The gap is between what it certifies and what "verified inference" is heard to mean.
It certifies that a committed model was faithfully executed *at the nodes that were
checked*. It does not certify that the whole computation was faithful, and it cannot
certify anything about whether the committed model is benign.

Directions that address the gap rather than the symptom:

* **The batched per-layer check of §10.** This is the one we built, and it is the
  cheapest point on the trade-off curve we found: detection `1.000000` against every
  attack in this document, exact completeness, ~9x proof size, and no primitive
  beyond the Merkle tree the protocol already uses. Its limits (one layer at a time
  without rescaling, honest committer, dense ReLU layers) are stated in §10.
* **Exact-binding proofs.** A zkSNARK over the full circuit removes the sampling gap
  entirely — at the prover cost the paper set out to avoid. That trade-off is the
  honest framing: the orders-of-magnitude speed-up is paid for in soundness against
  sparse tampering, not obtained for free. §10 is a partial, much cheaper instance of
  this: it buys exact soundness *per checked layer* rather than over the whole circuit.
* **Refereed delegation (Appendix D).** Unaffected by any of this. Bisection isolates
  the *first* disagreeing node in `O(log n)` rounds, so a single-node tamper is found
  with certainty rather than probability `1/N`. It needs two servers with at least one
  honest — a stronger assumption, but it converts our attack from near-certain success
  to certain failure. **We regard this as the most promising direction in the paper.**
* **Verifiable training or model attestation.** Backdoor-freedom is a property of how
  the model was produced. No inference-time check on a committed model can establish
  it, because the committed model may itself be backdoored — which is not an attack on
  the protocol at all, merely a limit on what it means.

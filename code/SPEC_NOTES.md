# Specification notes: reading Anchuri et al. precisely

No reference implementation of *Towards Verifiable AI with Lightweight
Cryptographic Proofs of Inference* (SaTML 2026, ePrint 2026/541) has been
released, so this repository is a from-scratch reimplementation. In a few places
the paper is ambiguous, informal, or contains an evident typo. Every such place is
recorded below, together with the reading we adopted and why. Readers checking our
attack results against the paper should start here.

Section and figure numbers refer to the ePrint version (2026/541).

---

## 1. The off-by-one in `RandPathTest` (Figure 3)

Figure 3 reads:

> * Sample a node `ã_{j_d} ∈ out(trc̃)` where `d` is its depth.
> * For layers `ℓ = d−1, …, 1`:
>   * Query `ã_i ∈ trc̃` for all `i ∈ G_{j_ℓ}` … and the correct weights `w_{i j_ℓ}`.
>   * Sample `j_{ℓ−1}`, a random index in `G_{j_ℓ}`, and repeat.
> * Accept iff for all `ℓ = d−1, …, 1`: `ã_{j_ℓ} = φ(Σ_{i∈G_{j_ℓ}} w_{i j_ℓ} ã_i)`.

As written this is inconsistent. The loop begins at `ℓ = d−1` and immediately
dereferences `j_{d−1}`, but `j_{d−1}` is only produced at the *end* of that same
iteration body — the first index actually available is `j_d`. Symmetrically, the
accept condition ranges down to `ℓ = 1`, which would demand a local check at a node
in the input layer, where there are no parents and no weights.

**Our reading.** The path is `j_d, j_{d−1}, …, j_0`, where `j_0` lies in the input
layer. A local consistency check is performed at every node from the output layer
down to layer 1 — that is `d` checks — and the input-layer endpoint is compared
against the query rather than recomputed.

**Why.** Section 2 describes the construction as one that "verifies the correctness
of an entire path from input to output" so that "the output is backed by a chain of
consistent computation", and separately notes that "the input layer remains an
immutable anchor". Checking `j_d` is essential: it is the only node that ties the
claimed output to the rest of the trace. Checking `j_0` is impossible. Both
corrections are forced.

Implemented in `Path.checked_layers` and `Verifier.verify`.

---

## 2. Exact equality versus a tolerance

Figure 3 states the local check as an exact equality, `ã_j = φ(Σ w_ij ã_i)`. This
cannot be implemented as written. The prover evaluates a layer as a matrix product
and the verifier re-evaluates one neuron as a single dot product; in floating point
the two summation orders give different results, so an honest trace fails exact
equality some of the time.

The paper is aware of this. Appendix E.3 says the threshold "could be 0, but to
account for potential completeness concerns, we can set a reasonable `1e-4`
threshold", and the same appendix reports pass rates at thresholds from `1e-6` to
`0.308`.

**Our reading.** The check is `|ã_j − φ(Σ w_ij ã_i)| ≤ τ_abs + τ_rel · |φ(·)|`, with
`τ_abs = 1e-4` and `τ_rel = 0` by default (`ProtocolParams`).

We report the observed residual distribution on honest traces so the choice can be
audited, and we also measure what a zero-tolerance verifier would do. This matters
for the attack: any tamper must move an activation by more than `τ`, which is a
negligible constraint (our attack moves it by ~10⁰–10¹).

---

## 3. Which layer the output activation function belongs to

Section 2 fixes a single `φ` for the whole network, while Remark 3 defines
`Idxs_out` as the logit layer with `f_out = softmax`. Applying ReLU to the logits
would be wrong, and applying softmax inside the trace would make the local relation
non-local (softmax couples all output neurons).

**Our reading.** Hidden layers use ReLU; the output layer uses the identity, so the
trace's output entries are logits, and `f_out` (softmax or argmax) is applied
*outside* the trace by whoever consumes the output. This is what Remark 3 describes.

---

## 4. Bias terms

Footnote 3 of Section 2 folds the bias into the weights "as an additional weight
with an activation of 1".

**Our reading.** We keep the bias as the last entry of each committed weight row
rather than inserting a constant-1 neuron into every layer of the trace. The two are
equivalent for the local relation, but a constant-1 neuron would be a *sampleable*
path node whose check is trivially satisfied, which would distort the visit
distribution and hence every detection probability in the paper and in our analysis.
Keeping the bias in the weight row avoids that artefact.

Implemented in `Layer.weight_rows` / `Layer.split_row`.

---

## 5. Commitment granularity

Section 4.4 describes committing to the model and the trace with a generic vector
commitment, which suggests one leaf per scalar. Section 8.1 describes what the
authors actually built: "a Merkle tree built on row-wise hashes (computed using
SHA-256)", opening "the relevant row and the corresponding Merkle path for each
layer".

**Our reading.** We follow Section 8.1.

* `C_M` — one leaf per **weight row**: a dense neuron's incoming weights plus bias,
  or a convolution's output-channel kernel plus bias. Checking one node therefore
  costs exactly one weight-row opening.
* `C_trc` — one leaf per **layer activation vector**.

Granularity changes proof size, not soundness. The security-relevant quantity is the
set of nodes the verifier *checks*, and a node cannot be checked without its weight
row, because the verifier holds only `C_M` and never the model. **The verifier's
budget is therefore the number of weight-row openings**, and that is the unit in
which our attack analysis is stated.

One consequence is worth stating plainly: for a feed-forward network a layer is a
single row, so the proof reveals the whole trace. That is a property of the paper's
own scheme, not of our reading of it — footnote 2 concedes the protocol "is not
zero-knowledge and reveals information about the underlying model". It does not help
the verifier detect anything, because detection still requires the weights.

---

## 6. Layers that are not affine-plus-`φ`

The formalism in Section 2 covers only `a_j = φ(Σ_{i∈G_j} w_ij a_i)`. The paper's own
classifier experiments use ResNet-18, which contains max-pooling and residual
connections that do not fit that form.

**Our reading.** A layer must supply a publicly known function of its parents. Max
pooling uses `a_j = max_{i∈G_j} a_i` and carries no weights. This is a conservative
extension: it neither strengthens nor weakens the protocol, and it is required to
handle convolutional classifiers at all.

Implemented in `MaxPool2dLayer`.

---

## 7. Whether the whole input layer is compared against the query

Figure 3 checks only the single input-layer node the path reaches. But the input
layer's activations are opened anyway, so comparing all of them against `qry` is
free, and any careful implementation would do it.

**Our reading.** Both are available. `check_input_anchor` (default on) implements
Figure 3 exactly; `check_full_input` (default off) is the strengthening.

We run the *attack* experiments with the strengthening switched on, so that the
attack is demonstrated against the stronger of the two verifiers.

---

## 8. The distribution over output nodes

Figure 3 says "sample a node `ã_{j_d} ∈ out(trc̃)`" without naming a distribution.

**Our reading.** Uniform over the output layer, matching the uniform choice at every
subsequent step. `visit_probabilities` computes the induced marginal exactly; for
dense architectures it is uniform on every layer, which is the regime in which the
paper's `1/N` figure (Section 5.2) is tight. For convolutional stacks it is
genuinely non-uniform, because receptive fields are local — border neurons are
visited less often than central ones. That asymmetry is not mentioned in the paper
and is exploitable; we return to it in the attack analysis.

---

## 9. `EvalTrace` for the substitute model in Equation (1)

Equation (1) is written

> `|φ((a^{l−1}_M)ᵀ W^l_M) − φ((a^{l−1}_M̃)ᵀ W^l_M)|`

Both terms use `M`'s weights; only the layer-`l−1` activations differ. So it measures
how far layer `l` moves when the substitute's activations are spliced into the honest
model — matching Section 6.2.2's description of replacing "the activations of `M`
along a random path `P` with those from `M̃`".

This is *not* the residual the verifier computes. The verifier compares the
substitute's *claimed* `ã_j` (produced by `M̃`'s weights) against `M`'s recomputation.
The two are related but distinct, and only the second decides acceptance.

**Our reading.** We implement and report both: `equation1_separation` reproduces the
paper's metric, and `verifier_residuals` reports the quantity that actually governs
detection.

---

## 10. What was *not* reproduced

* **The refereed-delegation protocol (Appendix D).** Out of scope: it is a separate
  protocol with a different trust assumption (two servers, at least one honest), and
  our attack does not target it. Its bisection argument is sound and unaffected by
  our results.
* **LLM-scale experiments (Sections 6.3, 7.3, F).** These need a GPU and a 7B-parameter
  model. We work at MNIST scale on CPU, where every quantity of interest can be
  computed exactly rather than sampled — including the full visit distribution and the
  complete set of locally inconsistent nodes, neither of which is tractable at 7B.
* **The zkLLM performance comparison (Section 8.2).** Not needed for a security claim.

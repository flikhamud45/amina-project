# Phase 2 — a heavier per-layer check works, in principle

Full derivation, numbers, and the honest what's-not-solved caveats are in
[`DEFENCE_NOTES.md`](../DEFENCE_NOTES.md) §9 and
[`REVISION1_PLAN.md`](../REVISION1_PLAN.md)'s Phase 2. This is the summary.

Phase 0 (see `phase0_summary.md` in this directory) found that every way of
choosing *which* neurons to check, however clever, fails the same way once the
adversary knows the rule: it either reduces to a trace-independent scheme
(capped at `k/|U|`, Theorem 2) or has some claimable value it can't see
(Theorem 3/4). The only lever left is to change *what a single check
verifies*, not how neurons are chosen.

**The check.** Batch an entire layer's local relations into one scalar instead
of sampling among them: draw a random challenge vector `r` over the whole
layer and compute

```
C = sum_j r_j * ( ã_j - phi( sum_i w_ij ã_i ) )
```

`C` is exactly zero if every local relation in the layer holds; if even one is
violated, a uniformly random linear functional of the (nonzero) residual
vector is zero with probability 0 (continuous challenges) or `≤ 1/|F|`
(challenges from a size-`|F|` field) — elementary linear algebra, since `φ`
has already been applied and each node's residual reduced to one real number.

**Measured** (`scripts/run_sumcheck_prototype.py`, layer 1 of `mlp_mnist_full`,
width 512):

| Trace | batched per-layer check | best sampling-based check |
|---|---|---|
| Honest | completeness 1.000000 | — |
| Naive single-neuron tamper | **1.000000** | uniform: 0.001953 |
| 13-neuron zero-hiding backdoor (§4's exploit, `0.000000` under contribution weighting) | **1.000000** | uniform: 0.025781 |

Both attacks caught with certainty, independent of `N` and of where in the
layer the inconsistency sits — including the one attack that reduces every
value-based sampler tried in this revision to exactly zero.

**What this does and does not show.** This measures the *soundness* gain,
assuming a succinct proof for the value of `C` already exists. It does not
build one. As implemented, computing `C` needs the verifier to hold every
weight and every claimed activation of the layer — the same information as
checking all `N` nodes individually and taking the logical AND. By itself
this buys nothing over that naive scheme; the entire value depends on a
genuine succinct argument that lets the verifier accept `C`'s value in
`O(log N)` or `O(1)` communication instead of `N` weight-row openings.

That argument is not exotic: it's the same primitive real zkML systems use
for a layer's affine part (a sumcheck protocol over the multilinear
extensions of the weight matrix and activation vector, GKR-style), with the
ReLU handled by a separate lookup argument over valid `(z_j, a_j)` pairs
(the technique behind the Jolt zkVM cited in our own related work). That is
precisely the prover-cost machinery the paper set out to avoid by sampling.

**The concrete proposal**, not built here: keep `RandPathTest`'s outer
structure — a random path visits `O(depth)` layers, not all of them — but
upgrade the check performed at each *visited* layer from "one sampled node"
to "one succinct whole-layer sumcheck." This removes the within-layer `1/N`
gap that no sampling rule can close (§5's corollary), while keeping the
protocol's central efficiency claim (sublinear in the network, not the full
GKR cost of an end-to-end proof). Building and costing that hybrid is real
cryptographic engineering — pairing-based polynomial commitments or a
multi-round sumcheck implementation — and is where this revision stops.

Code: `scripts/run_sumcheck_prototype.py`.

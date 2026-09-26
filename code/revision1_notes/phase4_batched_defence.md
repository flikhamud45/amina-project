# Phase 4 — a defence that actually works

This is the one that survives. Everything in Phase 0 failed against an
adversary who knows the rule (`phase0_summary.md`); Phase 2 showed batching a
whole layer *would* fix it but only measured an idealised version that assumed
the verifier could see the weights (`phase2_sumcheck.md`). Phase 4 builds the
real thing: the verifier never holds the weight matrix.

Code: `src/pvi/protocol/batched.py`, tests in `tests/test_batched.py`,
measurement in `scripts/run_batched_defence.py`.

## The construction

Write the checked layer as `A = [W | b]` (shape `N x (M+1)`) and the augmented
input `x = [a_in ; 1]`, so the pre-activation is `z = A x` and the claimed
output is `a`. The layer is correct exactly when:

1. `a_j >= 0` for all `j` — ReLU range
2. `a_j * (a_j - z_j) == 0` for all `j` — complementarity
3. `z_j <= 0` wherever `a_j == 0` — correct branch

(2) forces `a_j == z_j` wherever `a_j != 0`; (1) makes that the right branch;
(3) covers what (2) says nothing about.

**(3) is the whole point, and it is where a naive batching scheme fails.**
"Take a random linear combination of the layer's constraints" verifies (2) and
stops — but (2) is *vacuous exactly where `a_j == 0`*. That is Theorem 3's zero
blind spot reappearing in arithmetic dress: the same attack that reduces
contribution weighting to detection `0.000000` would walk straight through a
naive batched check. We close it by making the prover commit, in the clear, to
a witness `y_j := -z_j >= 0` on the zero set. The verifier range-checks `y` for
free (it is plaintext) and then ties it back to the committed weights.

The verifier never learns `z`. It folds (2) and (3) into a **single** linear
functional of `A x`:

```
<alpha*(s . a) + beta*t_Z , A x>  ==  alpha * sum_j s_j a_j^2  -  beta * <t_Z, y>
```

with `s, t, alpha, beta` Fiat-Shamir challenges drawn *after* the prover is
committed to `a` and `y`. So one opening of `chi^T A` per layer suffices.

### The weight commitment (Merkle only — no pairings)

Ligero-style: Reed-Solomon-encode each *row* of `A` to length `2(M+1)` and
Merkle-commit the *columns*. To open `u = chi^T A` the prover sends `u`; the
verifier re-encodes `u` itself and spot-checks `t` random columns, each
requiring `<chi, E[:,c]> == Enc(u)[c]`. Encoding is linear, so an honest `u`
passes every column, while a dishonest one disagrees with `Enc(chi^T A)` in at
least `M'-M` positions (RS distance), caught with probability `>= 1/2` per
column — soundness `2^-t`.

No proximity test is needed, because in this protocol's threat model `C_M` is
an **honest** commitment to the intended model: the whole premise is a prover
that commits a benign model and then lies about the *trace* (§1 of
`DEFENCE_NOTES.md`). A malicious *committer* would additionally need the
standard Ligero proximity argument — a real extension, not claimed here.

### Exact arithmetic

Everything is integer arithmetic in a prime field. This also disposes of the
floating-point tolerance `SPEC_NOTES.md` §2 was forced into — the one that
handed the adversary a free perturbation budget below `1e-4`. Honest
completeness here is **exactly** 1, not 1-up-to-tau.

## Measured, against this repository's own attacks

`scripts/run_batched_defence.py`, `mlp_mnist_full`, layer 1 (width 512), 40
queries, scale `2^10`, 24 code queries. Attacks are the real planners from
`attacks/tamper.py` and `defences/adaptive.py`, not synthetic tampering.

| Attack | batched check | uniform sampling | contribution weighting |
|---|---|---|---|
| naive single-neuron flip (support 1.0) | **1.000000** | 0.001953 | 0.112636 |
| stealthy envelope-confined (support 2.5) | **1.000000** | 0.004958 | 0.150712 |
| **zero-hiding backdoor (support 12.2)** | **1.000000** | 0.023730 | **0.000000** |

Honest completeness `1.000000` (40/40), exactly.

The third row is the point. That attack is the one Phase 0 could not stop by
any means: it drives contribution weighting to *exactly* zero detection and
evades the PCA plausibility check at no extra cost. Here it is caught every
time, and not because we guessed where it would hide — detection does not
depend on *where* in the layer the tamper sits, or on *what value* it claims.

### It is width-independent, as the argument says it should be

Re-run on `LARGE_MLP_SPEC` (width 4096, 8x), as a batch job:

| | width 512 | width 4096 |
|---|---|---|
| honest completeness | 1.000000 | 1.000000 |
| naive tamper | 1.000000 (uniform: 0.001953) | 1.000000 (uniform: 0.000244) |
| zero-hiding backdoor | 1.000000 (uniform: 0.023730) | 1.000000 (uniform: 0.005664) |
| commitment build (one-off) | 0.8 s | 4.0 s |

Uniform sampling's detection *degrades* with width (that is the `1/N` ceiling).
The batched check does not move.

## What it costs, honestly

Proof size is the trade: **115.6 kB** at width 512 against the sampling
scheme's ~13 kB — roughly 9x. At width 4096 it is 837.6 kB.

That growth is `O(t*N)`, dominated by the `t` opened columns of `N` entries
each. Worth being precise about what this does and does not break: the existing
scheme already reveals the *entire* activation vector of any layer it touches
(`C_trc` commits one leaf per layer — `SPEC_NOTES.md` §5), so per-layer cost was
already `O(N)`. This is a constant-factor increase on an existing `O(N)`, not a
new asymptotic class, and the protocol's real efficiency claim — visit
`O(depth)` layers rather than proving the whole network — is untouched.

Two easy, unexercised reductions: field elements fit in 26 bits but are
serialised as 8-byte integers (a free ~2x), and 24 code queries buys soundness
`2^-24` where `2^-16` would do (another ~1.3x). Neither is implemented; the
numbers above are what the code actually produces.

## Scope and what is genuinely left

- **One layer at a time.** The field is sized so a layer's pre-activations fit
  without rescaling, which holds comfortably for layer 1 of these MLPs
  (`z` reaches ~2·10^5 against a field half-width of 3.4·10^7). Chaining the
  check through *every* layer of a deep network needs either a wider
  multiplication routine for a larger prime, or per-layer rescaling with a
  range argument. That is standard zkML engineering and it is not done here.
  For the threat model in question this is less of a gap than it sounds: the
  attack has to tamper *somewhere*, and the check can be applied at whichever
  layers the path visits.
- **Honest committer assumed** (see above) — matching the paper's own threat
  model, but worth stating.
- **Dense ReLU layers only.** Convolutions reuse weights and would need the
  matrix laid out accordingly; max-pool is not an affine-plus-ReLU relation at
  all.
- **Prover cost not benchmarked** beyond the one-off commitment build. The
  per-query prover work is one `chi^T A` matvec plus `t` Merkle openings, which
  is cheap, but we have not profiled it against the paper's millisecond claims.

## Bottom line

Against the specific attack this project built — a trigger-conditional backdoor
that serves the honest trace on clean inputs, tampers a handful of activations
on triggered ones, and hides in exactly the place every sampling rule cannot
look — this is a defence that works: detection `1.000000`, completeness
`1.000000`, no dependence on where or what the tamper is, at a ~9x proof-size
cost and with no cryptographic machinery beyond the Merkle tree the protocol
already uses.

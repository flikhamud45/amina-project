# Phase 4 — a defence that actually works

This is the one that survives. Everything in Phase 0 failed against an
adversary who knows the rule (`phase0_summary.md`); Phase 2 showed batching a
whole layer *would* fix it but only measured an idealised version that assumed
the verifier could see the weights (`phase2_sumcheck.md`). Phase 4 builds the
real thing: the verifier never holds the weight matrix.

Code: `src/pvi/protocol/batched.py`, tests in `tests/test_batched.py`,
measurement in `scripts/run_batched_defence.py`.

**For a from-first-principles explanation** (Merkle trees, Reed–Solomon, finite
fields, Fiat–Shamir, all built up from nothing), see `defence_explained.md`.
This file is the terse version.

### One correction worth recording

The first implementation had a real Fiat–Shamir ordering bug: the spot-checked
column indices were derived from `(digest, x, a, y)` but **not** from `u`, so a
cheating prover could read the columns off in advance and solve for a `u`
passing all of them — 25 linear constraints on 785 unknowns. That forgery was
implemented and it *was accepted*. Fixed by deriving the columns from `u` as
well (two-phase transcript), which makes the attack circular. Regression test:
`test_adaptive_transcript_forgery_is_rejected`.

The measurement script never caught it, because it only ever runs the *honest*
prover algorithm on tampered traces — it does not model a prover that cheats on
the proof itself. Worth remembering for anything else of this shape.

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

## Whole-network chaining (added after review)

Checking one layer in isolation proves very little: a prover could hand layer 2
an input unrelated to layer 1's verified output, and the output (logit) layer
was not supported at all — `fixed_point_layer` clamped with `relu`, so an
honest prover emitting real logits was rejected **0/20**. Both are fixed.

- **Identity layers.** For a layer with no activation the constraint is just
  `a == z`, which is already linear: `<s, A x> == <s, a>`. No zero set, no sign
  witness, no non-negativity check (logits are ~75% negative). Honest logits now
  accepted 20/20; a tampered logit caught 20/20.
- **Verifier-recomputed inputs.** The verifier derives each layer's input from
  the previous layer's *verified* output using a public rounding rule
  (`rescale`), and never takes an input from the prover. This needs no range
  argument, because the protocol is not zero-knowledge — the activations are
  revealed anyway (`SPEC_NOTES.md` §5). Soundness follows by induction: the
  input layer is honest because the verifier built it; each verified layer
  output then fixes the next layer's input.
- **Per-layer challenge binding.** Each layer's challenges are derived from its
  index and a running hash of all earlier layers' messages, so proofs cannot be
  replayed at another layer or spliced across queries.
- **Scale.** At `2^8` the honest pre-activations peak at
  `[232021, 343622, 1298689]` across the three layers against a field half-width
  of `33554429` — 25x headroom. (At `2^10` the output layer overflows, which is
  why the default is now 8.)

Measured end to end on `mlp_mnist_full` (all three layers, 10 queries):

| | value |
|---|---|
| honest whole network accepted | 10/10 |
| tamper at layer 1 / 2 / 3 caught | 10/10 each |
| prover breaking the inter-layer chain | rejected 10/10 |
| whole-network proof | **188.9 kB** (115.8 + 61.8 + 11.4) |
| prover / verifier | 7.4 ms / 9.4 ms |
| commitment build (one-off, all layers) | 0.7 s |
| model size | 2.04 MB (535,818 float32) |

So the proof is **11.1x smaller than simply downloading the model**, at
millisecond cost on both sides.

## What it costs, honestly

Once every layer is checked in full, this is no longer a *sampling* protocol —
there is nothing left to sample. The honest comparison is therefore not the
13 kB sampling proof but the two things it sits between: downloading the model
(2.04 MB) and a full SNARK.

| | Sampling scheme | Batched, whole network | Download the model |
|---|---|---|---|
| Proof / download | 13.4 kB | 188.9 kB | 2.04 MB |
| Prover | ~0.6–1 ms | 7.4 ms | 0 |
| Verifier | ~0.5–0.8 ms | 9.4 ms | full forward pass |
| Catches the backdoor | ~0.2–3% | every time | every time |

Two easy, unexercised reductions: field elements fit in 26 bits but are
serialised as 8-byte integers (a free ~2x), and the column count could drop if
challenges were verifier-sent rather than hash-derived (see below).

## Scope and what is genuinely left

- **Soundness parameters are too small for Fiat–Shamir.** The field is 26 bits
  and 24 columns are spot-checked, so each check is about `2^-24`–`2^-26`. With
  *verifier-sent* challenges that is fine. With hash-derived challenges a
  cheating prover can grind — re-randomise part of its message and re-hash until
  the challenge is favourable — so the effective security is only ~24 bits,
  which is not enough. Fixes: have the verifier send challenges (cheapest and
  most faithful to the base protocol), or repeat the combination check and raise
  the column count, or move to a ~64-bit prime. **Not yet done; stated here
  rather than papered over.**
- **Honest committer assumed** — matching the paper's own threat model, but it
  is an assumption. A malicious committer needs the Ligero proximity argument.
- **Dense layers only.** Convolutions share weights and need a different matrix
  layout; max-pooling is not an affine-plus-activation relation at all.
- **It does not make a backdoored model safe.** It guarantees the committed
  model was executed faithfully. If the committed model is itself backdoored, no
  inference-time check helps — that is a property of training.

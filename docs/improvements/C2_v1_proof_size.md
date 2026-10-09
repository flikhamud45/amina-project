# C2. V1: send the int8 values, prove their windows with a logUp-GKR (proof size at long prompts)

Plan items C1 (design, `S/sp2027_research/v1_design/V1_SPEC.md`, executable toy `scratch/spec_toy.py`) and C2
(implementation). Reviewer objection D3 (proof size) and the future-work item "sum-check for the non-weight
operations". Commits c675181 (days 1-2), d67c0c2 (day 3), 7c6fbc9 (day 4), cfdde3a (days 5-8), 5158049 (Triton GKR
prover). Status: implemented end to end (modes C and Kpre, interactive and Fiat-Shamir) and tested; real-size byte
measurements and the GPU prover's timings in section 8.

## 1. The problem

Today the prover sends every weight op's output `Z = A [X; 1]` (int32 claims, about 2.3 bytes each after the PVC3
codec). The claims grow linearly with the prompt: at 2,048 tokens the proof of Llama-2-7B is about 6.6 GB, close to
the int8 weights themselves. Every weight op of a transformer block except the LM head and the embeddings feeds a
requantisation (`q, k, v, gate, up, fc1`: `a = requant(Z)`, int8) or a residual add (`o, down, fc2`:
`r + shift(Z)`): the next op only ever reads `a` or `Delta = shift(Z)`, never `Z`.

## 2. The idea

The prover sends `s = a` (or `Delta`) instead of `Z`, about one byte per entry. The verifier still recomputes every
cheap operation from `s` and the query, so it holds every op's input `X` in full. What remains is to prove that each
hidden `Z` requantises to the sent value. With `lo(v) = ceil((v 2^30 - 2^29) / m)`, the smallest `z` with
`shift(z) = v`, and `W(v) = lo(v + 1) - lo(v)`:

> `shift(Z) = s`  iff  `delta = Z - lo(s)` lies in `[0, W(s))`.

Clamped entries (`a = +-127` with `Z` outside the normal window) are listed as *exceptions* with their exact `Z`
(window width 1). So "every entry is in its window" is a range check on the residues `delta`, and **`delta` is a
linear function of the committed weights**, because the verifier knows `X` and `lo(s)`:
`delta = A [X; 1] - L`.

The range check is a **windowed logUp** over the table `T = {t + x w : w a window width, 0 <= t < w}` of
`F = F_{p^8}`: entry `(delta, W)` is the leaf `v = delta + x W`, in `T` iff `0 <= delta < W`. The prover sends
integer multiplicities `m`; a fractional-sum GKR reduces `sum_i 1/(alpha - v_i)` to one evaluation of the leaves'
multilinear extension, and since `v` is linear in `A`, that evaluation is `chi^T A xbar` for verifier-known
`chi, xbar` (minus the public `L`, `W` terms). **Today's fold and column check already answer `chi^T A`** (with 8
base rows, the planes of `chi` in `F`, instead of `r`), against today's weight commitment. There is no new
commitment, no NTT, no activation Merkle tree and no proximity test.

## 3. The protocol (as implemented, `cut_protocol.run_cut_query`)

The cut set is a public function of the graph, the commitment plan and the query (`cut.CutPlan.from_graph`): a
linear weight op whose output feeds exactly one requantising op built by `graph.requant_fn` (P1), claims `T` columns
(P2), has windows of width 2 to `2^16` (P3), honest claims below `2^29` (P4, enforced by `commit_graph`), and whose
window widths are each shared by fewer than `2^29` entries (P5, so honest multiplicities stay below `p`). Under a
commitment plan a col-layout matrix is cut only with all its members. Ops are packed in graph order into GKR
instances of at most `2^lmax` leaves (`lmax` = 27; 25 on an 11 GB card).

| # | Message | Verifier check (rejection label) |
|---|---|---|
| M1 | PVC3 values (cut ops' `s`, clear ops' `Z`, lookup rows), PVX1 exceptions, PVC3 multiplicities | decode and derive with the cut ops' ranges (`range_or_shape`), `token`, lookups, exceptions well formed and `requant(z) = a` (`cut_exception`), `0 <= m_t < p` and `sum m = N` (`cut_multiplicities`) |
| C1 | `alpha` in `F` outside `F_p + x F_p`; today's `chi` of the clear ops | |
| M2 | one fractional-sum GKR per instance, round by round | every round and layer check, non-zero root denominators (`cut_gkr`) |
| M3 | folds at each instance's point: `u = chi^T A` (8 planes) for a cut row op, `y = A xbar` for a cut col matrix (mode C); `y = A xbar` for every cut op (Kpre); today's `u` of the clear ops | `freivalds` (clear ops), `kpre_y` (Kpre: `chi_s y = u_s xbar` for the secret rows), `cut_final` (the GKR's final claims equal the extensions computed from `s`, the exceptions and the folds), `cut_logup` (`sum_beta p0/q0 = sum_t m_t/(alpha - tau_t)`) |
| C2/M4 | column indices, openings (mode C) | today's column checks, with `chi` planes and `u` (row ops), `xbar` planes and `y` (col matrices) as the two sides |

Interactive challenges are the verifier's own (a CSPRNG at the moment each is drawn); in one process the GKR's
round challenges are drawn when the prover asks for them, after its message, and the verifier checks against the
ones it drew. Under Fiat-Shamir the prover runs the GKR on a fork of the transcript and the verifier re-derives
every challenge.

## 4. Why it is sound (V1_SPEC Sec. 3; the paper's appendix)

Let `l*` be the first op that is not locally correct (its input is then honest).

1. **Clear `l*`:** today's argument (Freivalds `p^-r`, or the column term).
2. **Cut `l*`:** some entry has `delta` outside its window (Lemma 0, tested exhaustively in T0). Since the input is
   honest, `|A [X; 1]| < 2^29` (P4), so `|delta| < 2^30 + 2^16 < p` and the leaf `delta + x W` is **not in `T`**
   (`1` and `x` are independent over `F_p`).
3. **Lemma 1 (integer multiplicities).** If `0 <= m_t < p`, `sum m_t = N` as integers and
   `sum_i 1/(X - v_i) = sum_t m_t/(X - tau_t)` in `F(X)`, every `v_i` is in `T`. (Partial fractions are unique;
   characteristic `p`.) Both conditions are load-bearing: a proof can hold more than `p` leaves, and without them
   `p` copies of an off-table value would cancel. Tested by brute force on `F_49` in `test_logup.py`, which also
   shows each condition's counterexample.
4. So the rational identity fails, and at the random `alpha` (outside `F_p + x F_p`, where all poles lie) it fails
   except with probability `(N + |T|)/(p^8 - p^2)`.
5. Some instance's root is then wrong, and standard GKR soundness (`(1 + sum_{k<n} (3k + 2))/p^8` per instance)
   pushes the error to the final claims, which the verifier compares with the true extensions: these are
   `I alpha - sum eq v + (1 - I)`, where `sum eq v` is `S - chi (L e) + x chi (W e)` and `S = chi^T A [X; 1] e` is
   checked by the column test (mode C) or the secret rows (Kpre).

At lambda = 128 the new terms are below `2^-215`, far inside today's budget (`params_for(cut=True)` gives them two
of the union bound's shares). Under Fiat-Shamir every new round's error is at most `3/p^8` or `2^-215`, so the
round-by-round argument carries over.

## 5. Implementation

| File | What |
|---|---|
| `extfield.py` | `F_{p^8} = F_p[x]/(x^8 - 11)` on int64 tensors (irreducibility tested), eq tables, prefix sums, cubic interpolation, fraction trees, integer-matrix products, packing |
| `logup_gkr.py` | the eager GKR prover; a verifier that draws the challenges in transcript order, then checks every layer's affine round chain at once (50 ms at `n = 20`); transcript bytes (`E(n) = 4 + sum_{k<n}(3k + 4)` elements, 31 B each) |
| `gkr_triton.py` | the GPU prover (generated by `experiments/8_v1/gen_gkr_triton.py`): BabyBear Montgomery on uint32, the `F_{p^8}` product unrolled, kernels for leaves, tree, eq tables, rounds and folds; same transcript byte for byte |
| `logup.py` | the table, multiplicities and check D5, an instance's leaves, the padding indicator, check F4, the multi-instance driver |
| `cut.py` | the cut plan (P1-P5, instances, widths, digest), exact windows, the prover's split, the verifier's public windows |
| `cut_protocol.py` | the query: messages, PVX1, the challengers of the GKR, the checks; `cut_proof_bytes` (exact sizes without running the GKR) |
| `graph.py`, `transformer.py`, `real_weights.py` | `requant_fn`: the builders' requantising ops built from their params and tagged (logits unchanged) |
| `protocol.py` | `SecurityParams.cut`, `params_for(cut=)`, `Challenger.ext` and `fork`, the per-query cut set in the verifier (excluded from Freivalds), the derive branch for cut ops |

## 6. Tests

`test_extfield.py` (T1), `test_logup_gkr.py` (T2: `n` = 1..14, every element tampered, vectorised verifier =
round-by-round reference), `test_logup.py` (T3, Lemma 1 brute force, K1: `n = 20` with all 650 tampers),
`test_cut.py` (T0 exhaustive windows for eight multipliers, T9 logits unchanged, P1-P5, col matrices),
`test_cut_protocol.py` (T4: honest in 8 configurations including commitment plans with col matrices and lookup
tables, a generation graph; T5: the trace-tampering attack is rejected at `cut_final` in mode C and at `kpre_y` in
Kpre, fake exceptions at `cut_logup` or `cut_final`, malformed exceptions, multiplicity cheats, GKR and fold tampering,
a wrong clear claim; T7: the byte accounting equals a real query's), `test_gkr_triton.py` (the GPU prover's
transcripts). Full local suite with V1: 1,492 passed, 0 failed.

## 7. Byte model and first measurement (small cells, CPU, exact)

GPT-2, one block, 64 tokens, Fiat-Shamir, auto plan (cnn18c): v0 1.94 MB, V1 1.66 MB (1.17x); Kpre 1.17 MB to
0.91 MB (1.29x). At short prompts the fixed parts (folds with 8 planes, the table's multiplicities, the GKR
transcript) offset most of the claim savings; the savings grow linearly with `T`.

## 8. Measurements at real sizes

**Proof bytes** (job 1005287, RTX 2080 Ti node; random-weight decoders built as in the benchmark, Fiat-Shamir,
lambda = 128, the auto commitment plan in mode C, last block pruned; v0 = one `run_query(wire=True)`, V1 =
`cut_proof_bytes` on the same query, exact except the multiproofs' sizes at freshly drawn columns; the full V1 query
was also run and accepted on GPT-2 @64):

| Cell | Mode | Plan | v0 (MB) | V1 (MB) | Ratio | Claims, v0 -> V1 (MB) | Exceptions |
|---|---|---|---:|---:|---:|---:|---:|
| GPT-2, 12 blocks @64 | C | cnn18c | 16.6 | 13.0 | 1.28x | 11.0 -> 5.0 | 0 |
| GPT-2, 12 blocks @64 | Kpre | | 11.0 | 7.6 | 1.46x | 11.0 -> 5.0 | 0 |
| GPT-2, 12 blocks @512 | C | cnn18c | 91.2 | 47.1 | 1.94x | 85.5 -> 39.0 | 2 |
| GPT-2, 12 blocks @512 | Kpre | | 85.5 | 41.5 | 2.06x | 85.5 -> 39.0 | 2 |
| OPT-125M, 12 blocks @2048 | C | cnn18c | 349.6 | 172.3 | 2.03x | 343.8 -> 164.0 | 12 |
| OPT-125M, 12 blocks @2048 | Kpre | | 343.9 | 166.7 | 2.06x | 343.9 -> 164.1 | 12 |
| Llama-2-7B, 2 blocks @2048 | C | cnn16c | 269.5 | 126.4 | 2.13x | 254.1 -> 109.4 | 6 |
| Llama-2-7B, 2 blocks @2048 | Kpre | | 254.3 | 111.3 | 2.29x | 254.3 -> 109.6 | 6 |
| Llama-2-13B, 1 block @2048 | C | cnn17c | 69.1 | 38.1 | 1.81x | 60.6 -> 29.2 | 3 |
| Llama-2-13B, 1 block @2048 | Kpre | | 60.8 | 29.6 | 2.05x | 60.8 -> 29.3 | 3 |
| **OPT-1.3B, 24 blocks @2048** | C | cnn16c | **2,073.9** | **960.8** | **2.16x** | 2,022.4 -> 896.1 | 65 |
| **OPT-1.3B, 24 blocks @2048** | Kpre | | **2,023.1** | **910.6** | **2.22x** | 2,023.1 -> 896.7 | 65 |

The claims shrink 2.2-2.3x (int8 values instead of int32 claims, after the same PVC3 codec); the fixed parts (the
folds with 8 planes, the table's multiplicities, the GKR transcripts: 26 kB for GPT-2, under 1 MB everywhere) are
what keeps short prompts at 1.3-1.5x. The spec's byte model predicted 2.24x (C) and 2.32x (Kpre) for the full
Llama-2-7B at 2,048 tokens and 1.21x / 1.34x for GPT-2 @64; the measurements agree or are better. Exceptions (clamped
entries listed with their exact claim) are a few dozen per proof.

**GPU prover** (job 1005308): the Triton GKR gives the eager transcript byte for byte (12 tests); one instance of
`2^25` leaves takes 0.83 s on the 2080 Ti (24.6 ns per leaf, 5.8 GB peak), `2^20` 0.16 s against 15.2 s for the eager
prover on the CPU.

**Timings, first version** (job 1005343, OPT-125M 12 blocks @2048, 3 queries, CPU verifier): v0 prove 0.72 s,
verify 3.2 s; V1 prove 18.4 s (split 9.9 s and fold 3.2 s on the host, GKR 4.7 s), verify 8.5 s (final check 5.4 s).
Since then the split, the counts and the folds run on the prover's device, the windows come from tables, the final
check's products are batched, and the GKR's rounds take the eq factor out (12 products per pair instead of 20);
the timings are being re-measured.

## 9. Adversarial review (workflow `v1-soundness-review`, 15 agents)

Four reviewers (the verifier's checks; the GKR and logUp algebra; Fiat-Shamir and interaction; the cut's semantics
and ranges) read the code against the spec and tried to break it with scripts (compensating folds against the
trace-tampering attack, tampering sweeps by +1, 2W and -W on requant and residual ops in modes C and Kpre, 1-14
instances); every finding was then checked by an independent agent. No forgery against our models was found. The
confirmed findings and what was done:

| Finding | Severity | Fix |
|---|---|---|
| Kpre never evaluated P4 (honest `|A [X;1]| < 2^29`, used by the proof's step 3): the verifier's public graph has no weights and Kpre never calls `commit_graph`; a forgery was demonstrated on a model whose honest claims exceed the bound | low/medium (needs such a model) | P4 is evaluated from the K/Kpre verifier's own weights (`CutPlan.from_graph(weights=)`); both parties then derive the same plan |
| The cut accepted a requant op with any clamp; a clamp wider than int8 breaks `|L| < 2^29 + 2^16` (forgery on a crafted graph) | medium (latent) | P1 now requires `-128 <= lo <= hi <= 127` for requant ops |
| Cut folds (M3) accepted coefficients in `[p, 2^31)`: an alias of the same field element, so no forgery, but a second encoding of the message | low | rejected (`cut_final`), as v0 does for its `u` |
| The spec's defence-in-depth check `|X| <= max_input + 1` on cut inputs was missing | low | added to the derive (`range_or_shape`) |
| The exception bin (width 1) is outside P5: more than `2^29` clamped entries in one proof could not be encoded | low (completeness only; clamps occur at a rate of about `10^-5`) | documented |

Not a defect: interactive mode draws the GKR's challenges when the prover's code asks for them (one process); this
models a verifier sending each challenge after the message it follows, and Fiat-Shamir is unaffected.

**What P4 means in mode C.** The mode-C verifier holds no weights, so P4 is a property of the committed model, which
`commit_graph` enforces (it refuses any op whose honest claims can reach `2^29`). The theorem assumes an honest
commitment, as v0's claim range check already does. With an untrusted commitment (D4) the setup proof establishes
proximity, not entry ranges; then the claim bound has to be part of the published model (e.g. the open-weight
registry of D6), for v0 and V1 alike.

## 10. What remains

* The GPU prover's speed (spec target: an instance of `2^26` leaves in at most 3 s on the 2080 Ti) and its
  integration into `run_cut_query` (today the eager prover runs on the CPU).
* The streaming GPU verifier and mode K with the cut (`NotImplementedError` in v1).
* Real-weight byte profile (exceptions and value entropy differ), and the raw int8 wire (PVR1) as an option.
* `bench.py --proof v1` cells for the paper's tables.

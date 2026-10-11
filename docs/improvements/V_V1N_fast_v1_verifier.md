# V-V1N. The fast V1 CPU verifier: products by value, batched GKR chains, int8 values (parts 1-4)

Plan item V-V1N (`SP2027_PLAN2.md` Sec. 3.3, rank 1 of the should items), parts 1-4 (the CPU parts, in torch);
part 5 (the Triton kernel) and the native C++ kernels are not done (Sec. 7). Branch `p2-v1n`. Design: C2
(`C2_v1_proof_size.md`) and the V1 spec it cites (`S/sp2027_research/v1_design/V1_SPEC.md`). Status: implemented
behind `PVI_V1_BUCKETS` (default on), exact (bit-identical field results, tested), every V1 suite green with the
switch on, on its bucket variant and off; measured on the laptop only: the final check's products and `_xbar` 1.8x
faster at Llama-2-7B's 2,048 tokens, the GKR check 2x, whole small-model V1 verifiers 1.3-1.6x (Sec. 5). The torch
path does not reach the plan's target by itself (Sec. 5.3); the native kernel it specifies is not written (Sec. 7).
Measured on the Xeon (job 1009430, Sec. 5.4): a Llama-2-7B block's V1 verifier takes 2.13 s instead of 2.49 s
(1.17x), so the full model about 67 s instead of 78 s, against v0's 19.6 s (3.4x); the V1 suites pass with the switch
in each position on the GPU node.

## 1. The problem

The Plan 2 re-time (C2 Sec. 8.x, job 1007816, Xeon 4114, 8 threads, Llama-2-7B at 2,048 tokens in Kpre under
Fiat-Shamir) puts V1's CPU verifier at about 2.5 s per block, about 75-80 s for the full model against v0's 19.6 s.
Per extra block: `verify_final` 1.42 s (57%), `verify_derive` 0.54 s, `verify_gkr` 0.39 s, `verify_decode` 0.13 s.

* `verify_final` is check F3 (`cut_final`) with its operands: `_xbar` (`[X ; 1] e` of every cut op's input),
  `_kpre_y_ok` (F2) and `_final_ok`. For every cut op, `_final_ok` gathers the window matrices `[L ; W]` (`[2N, T]`)
  in float64 from the window table and multiplies them with the instance's column point `e` in `F_{p^8}` as three
  11-bit limbs per coefficient (`extfield.int_times`: 24 columns, so 48 float64 multiply-adds per entry, after two
  gathers and a copy); `_xbar` multiplies every input with the same 24 limb columns.
* `verify_gkr` verifies the instances one after another (`logup_gkr.verify`), each with its own grid building and
  `n - 1` steps of small batched products, and an absorption per round built from a fresh tensor-to-bytes copy.
* `verify_logup` (check F4) cross-multiplies two fraction trees.
* V1 decodes every claim to int64 (8 bytes per entry), although a cut op's values are int8.

## 2. The algebra

### 2.1 What the final check computes

For instance `beta` with final point `rho = (r_col || r_row)`, `e = eq(r_col)[0:T]` (`[T, 8]`) and
`chi = eq(r_row)` (`cut_protocol._points`), check F3 compares the GKR's `(P_hat, Q_hat)` with

    I = I_rows(r_row) I_cols(r_col),    Q = alpha I - sum_l (S_l - chi_l (L_l e) + x chi_l (W_l e)) + (1 - I),

summed over the instance's cut ops `l` (`cut_protocol._final_claims`). `L_l = lo(s)` and `W_l = W(s)` entry by entry
(`lo(v) = ceil((v 2^30 - 2^29) / m)`, `W(v) = lo(v + 1) - lo(v)`), except at the listed exceptions, where `L = z`
(the exact claim) and `W = 1`. `S_l = chi^T A [X ; 1] e` is the fold side, which depends on the layout: a mode-C
row-layout op sends `u = chi^T A` (8 planes) and `S = u . xbar` with `xbar = [X ; 1] e` (`_xbar`); a mode-C col
matrix and every Kpre op send `y = A xbar` and `S = chi . y` (checked by the column test or by F2 `kpre_y`, both of
which also use `xbar`). What V-V1N replaces is the computation of the three integer-matrix-times-`e` products
`L e`, `W e` (`[N, 8]`) and `X^T e` (`[K, 8]`); everything after them (the inner products with `chi` and `u`, `I`,
the comparison) is today's code.

### 2.2 Buckets (the spec's kernel)

`L` and `W` depend only on the value `s` of an entry, and `X` is its own value. For row `i` let

    B_i[v] = sum_{j : S_ij = v} e_j        (one 8-lane integer add per entry),

then for any integer function `f` of the values, `sum_j f(S_ij) e_j = sum_v f(v) B_i[v]`. With `f = lo`, `W` and the
identity this gives `L e`, `W e` and `X^T e` from `V` buckets per row (`V = 256` for a requant op, the span of the
values for a residual op's `Delta` or an input). An exception `(i, j, z)` sits in the bucket of its value like any
other entry, and `(z - lo(s_ij)) e_j` and `(1 - W(s_ij)) e_j` are added to row `i` afterwards; so its row gets
`z e_j` and `e_j`, exactly its column of `[L ; W]`. Exactness: `e` is canonical (`< p < 2^31`), a bucket sums at most
`T < 2^32` coefficients in int64; the buckets are then reduced mod `p`, the tables' residues split into three 11-bit
limbs, and each float64 product over at most `2^11` values has partial sums below `2^31 2^11 2^11 = 2^53`
(`buckets.bucket_sums`, `bucket_times`).

### 2.3 Table products (the torch default)

The buckets are the right kernel in C (one add per entry, a row's 16 KB of buckets in L1), but in torch they cost
eight `scatter_add_` passes (Sec. 5.1). A second exact factorisation of the same windows uses torch's fastest
primitives, a table gather and a BLAS product:

* `W(v)` takes only the two values `w0 = floor(2^30 / m)` and `w0 + 1`: with `q = 2^30 / m` and
  `x = (v 2^30 - 2^29) / m`, `W(v) = ceil(x + q) - ceil(x)`, which is `floor(q)` or `ceil(q)`.
* So with `D(v) = W(v) - w0` in `{0, 1}` and `C(v) = sum_{a <= u < v} D(u)` in `[0, v - a]` (`a` the smallest
  value),

      lo(v) = lo(a) + w0 (v - a) + C(v),        W(v) = w0 + D(v),

  and therefore `L e = lo(a) (1 . e) + w0 (S - a) e + C(S) e` and `W e = w0 (1 . e) + D(S) e`.
* The three integer matrices `S - a`, `C(S)`, `D(S)` have entries in `[0, V)`; they are gathered from one `[V, 3]`
  float64 table (`torch.nn.functional.embedding`, one pass writing 24 bytes per entry) and multiplied with `e` as
  **one** 31-bit limb per coefficient: 8 columns instead of 24, 24 multiply-adds per entry instead of 48. With
  entries below `2^k` (`k = bits(V - 1)`, 8 for a requant op), every partial sum over `n <= 2^(22 - k)` columns is
  below `2^k 2^31 2^(22 - k) = 2^53`, so each product is exact in any summation order (as `extfield.int_times`
  already assumes of BLAS); `T` is cut into such chunks (`2^14` columns for int8 values) and the chunk results are
  reduced and added in int64.
* `X^T e` needs no table: `X` itself (`|x| <= 128`, derive checks it) times `e` as one limb, exact over `2^14`
  columns at a time (`buckets.xbar`; any input below `2^16` in absolute value, with the chunk shrunk accordingly).

The identity `lo(v) = lo(a) + w0 (v - a) + C(v)` with `D` in `{0, 1}` and `0 <= C(v) <= v - a` is checked by the
tests on every width `floor(2^30 / m)` in `2 .. 2^16` that a multiplier gives (both ends of each width's multipliers,
over the int8 values), and by the code itself on every table it builds (`buckets._affine` raises otherwise).

### 2.4 Why the results are bit-identical

Today's code and both new ones compute, for every row and coefficient, the integer `sum_j L_ij e_jc` (and the same
for `W` and `X`) exactly and return its canonical residue mod `p`. Equal integers have equal residues, so `L e`,
`W e` and `xbar` are the same tensors, and every later step is today's code on the same inputs: `_final_claims`
yields the same `(I, Q)` per instance and `_final_ok` the same verdict. The tests compare these field elements
directly, at the query's own GKR point and at random points of `F_{p^8}`, in every path (Sec. 4).

### 2.5 Part 3: the GKR chains in one pass, the logUp identity by batch inversion

`logup_gkr.verify` draws an instance's challenges in transcript order and then runs, for each of its `L = n - 1`
layers at once, the affine round chain `c' = l1(r) c + (l0(r) - l1(r)) g(0) + l2(r) g(2) + l3(r) g(3)` on an
`[L, L]` grid whose columns past a layer's rounds are identities (`a = 1, b = 0`, eq factor 1).
`logup_gkr.verify_batch` first absorbs and draws every instance's challenges in the same order (each message's
bytes sliced from one numpy copy of the transcript: the bytes `_absorb` gives), then stacks **every layer of every
instance** as a row of one `[sum L, W]` grid, `W = max L`, padded with the same identities, and runs `W` steps
(`torch.tril_indices` places each layer's rounds). Each row computes the integers `verify` computes for it
(identity steps leave a canonical claim and eq factor unchanged), so the outputs `(p0, q0, rho, P_hat, Q_hat)` are
the same tensors and the verdict is the same: all instances pass or `None` (a failing instance only changes the
transcript state after a rejection, which nothing reads).

After batching, most of `verify_gkr` is the sequential challenge derivation (one SHAKE-256 key and stream per
round, and the rejection sampling of 8 coefficients). `Challenger.ext` for one element (every GKR challenge) now
takes the same 192-byte stream, the same accepted words (below `2p`, reduced) in groups of 8 and the same rejection
of an element of `F_p + x F_p`, in plain Python instead of numpy and tensor plumbing (`Challenger._ext_one`; today's
loop is `_ext_draw`, still used for several elements and with the switch off): the same challenges, about 27 to
20 us each on the laptop. Only V1 draws extension-field challenges, so v0 is unaffected.

F4 checks `sum_beta p0 / q0 = sum_t m_t / (alpha - tau_t)`. Every `q0` is non-zero (`verify_instances` rejects a
zero root denominator) and every `alpha - tau_t` is non-zero (`alpha` is outside `F_p + x F_p`, where every `tau_t`
lies), so `a / b = c / d` iff `a d = c b`: summing both sides with batch inversion (`extfield.inv_many`, Montgomery's
trick on a product tree, one `inv` per side) is the same test as today's cross-multiplication of the two fraction
trees. A zero denominator, which the callers exclude, takes the cross-multiplication.

### 2.6 Part 4: the values as int8

The decoder produces int32 claims; today V1 widens all of them to int64. With the switch on, `cut_protocol._narrowed`
keeps a cut op's values, when one exact min/max shows every value inside the op's public range (a requant op's
clamp, a residual op's `R_Delta`), in the narrowest of int8 and int16 that holds them (requant ops: always int8;
the random-weight decoders' residual `Delta`s: int16), and widens every other claim to int64 as before. A cut op with
a value out of range stays int64, so derive's range check rejects it with today's label (`range_or_shape`) at
today's point, after the token rule. Derive (`protocol.Verifier._derive`) accepts int8/int16 values for a cut op of
the query's plan only (v0 has no plan), range-checks them again and widens only the tensor its readers take; the
exceptions check (D4) and the final check read the narrow values directly. After derive the cut values occupy 1-2
bytes per entry instead of 8 (Llama-2-7B, 2 blocks at 2,048 tokens, the last pruned: 104 M cut entries, about
0.12 GB instead of 0.83 GB). In time this part is about neutral in torch (one exact min/max and a narrowing pass in
the decode, one widening of the tensor the readers take in derive, against today's widening of every claim and a
range check on int64); what it saves is memory and the final check's reads (1 byte per entry instead of 8).

## 3. Implementation

| File | What |
|---|---|
| `src/pvi/fullcheck/buckets.py` (new) | the switch (`mode`, `enabled`); `lw_times` (final check's `L e`, `W e`, both methods, exceptions); `xbar`; the bucket kernel (`bucket_sums`, `bucket_times`) and the table products (`table_times`, `_affine`) |
| `src/pvi/fullcheck/cut_protocol.py` | `_xbar` and `_final_claims` call `buckets` (today's code kept as `_lw_e_old` and the old branch); `_final_ok` checks `_final_claims`'s per-instance values; `_narrowed` and the int32 decode (part 4) |
| `src/pvi/fullcheck/logup_gkr.py` | `verify_batch`, `_draw`, `_well_formed`; `verify` unchanged |
| `src/pvi/fullcheck/logup.py` | `verify_instances(batched=)`, `logup_ok(batched=)`: default from the switch, today's code kept |
| `src/pvi/fullcheck/extfield.py` | `inv_many` (vectorised batch inversion) |
| `src/pvi/fullcheck/protocol.py` | derive accepts int8/int16 values for a V1 query's cut ops (`_CUT_NARROW`); `Challenger._ext_one` (one challenge; today's loop kept as `_ext_draw`) |
| `tests/test_v1_buckets.py` (new) | Sec. 4 |
| `experiments/8_v1/v1_phases.py` (new) | the verifier's phases per path on the same queries; `--micro` for the products alone |
| `experiments/8_v1/v1_bytes.py` | `--v1-paths 0 1`: V1's timed queries under each path, interleaved (`timed.v1_paths`) |

`PVI_V1_BUCKETS` is read at every call: `1` (default) the table products, the batched GKR with the one-challenge
path, batch inversion and the int8 values; `scatter` the same with the bucket kernel for the products where a row's
values span at most `max(256, T)` values; `0` today's verifier everywhere. Spans past
`buckets.SPAN = 2^15` values (a residual op's `Delta` could in principle span `2^28`) and inputs of `2^16` or more
fall back to today's code per op. The prover is unchanged; proofs and transcripts are byte-identical.

## 4. Validation

`tests/test_v1_buckets.py` (26 tests, about 2.5 minutes on the laptop, 2 threads):

* the windows are affine plus a 0/1 step at **every** window width `2 .. 2^16` that a multiplier gives (49,151
  widths `floor(2^30 / m)`; both ends of each width's multipliers, 76,294 in all; over the int8 values), and
  `buckets._affine`'s tables are the window tables on a sample and on a residual span;
* `lw_times` (table products and buckets) equals today's `[L ; W]` GEMM (`_lw_e_old`) at random `F_{p^8}` points
  with exceptions at about 160 widths (2-69, `2^k` and `2^k +- 1` up to `2^16`, 60 random), both multipliers of each,
  two clamps; and across row blocks (forced), column chunks (`T = 20,000` past the `2^14` chunk; a residual span of
  7,001 values, chunks of 512), a constant-width op (`m = 2^16`), int64/int16/int8 values; a span past `SPAN`
  returns `None`;
* `xbar` equals `extfield.int_times` for int8 inputs (int64 and int8 dtypes), `T = 20,000`, values up to `2^14`;
  inputs past `2^16` or `SPAN` return `None`;
* end to end: from one accepted query (gpt mode C row layout, llama mode C `cnn16c` with cut col matrices, opt Kpre,
  and the many-exception model in C and Kpre) `_xbar` and `_final_claims` give the same field elements under
  `PVI_V1_BUCKETS` = 0, 1, scatter, on int8 and on int64 values, at the query's point (where they equal the GKR's
  `P_hat, Q_hat`) and at random points with a random `alpha`;
* `verify_batch` equals `verify` instance by instance (outputs and the Fiat-Shamir state afterwards) for 1-5
  instances of 1-6 variables, interactive and Fiat-Shamir; for 3 instances every transcript coefficient tampered
  (coefficients 0 and 7 of every element) gets the same verdict and outputs from both;
* one challenge (`Challenger._ext_one`) equals today's loop on 1,600 draws (interactive and Fiat-Shamir, both
  `nonbase` settings) and on crafted streams that hit the `2p` rejection, the `F_p + x F_p` rejection and the next
  block;
* `inv_many` equals `inv` and `batch_inv`; the batch-inverted F4 has the cross-multiplied verdict (honest, a moved
  count, all counts zero);
* part 4: values out of range (requant above and below its clamp, residual past `R_Delta`) give `range_or_shape` in
  every path, a residual value in range but past int16 `cut_final`, a wrong token with a range violation `token`;
* every tamper of `test_cut_protocol` T5 (trace tampering in C and Kpre, both fake exceptions, multiplicity cheats,
  a GKR bit flip, a fold change) plus 6 random trace-tampering and 6 random single-value tampers (as
  `v1_bytes.py --tampers`) get the same labels in all three paths, and the honest queries are accepted.

The V1 suites (`test_v1_buckets`, `test_cut_protocol` T4/T5/T7, `test_cut`, `test_logup`, `test_logup_gkr`,
`test_lookups`, `test_extfield`, `test_gkr_triton_cpu`, `test_gkr_triton`) with `PVI_V1_BUCKETS` = 1, scatter and 0:
231 passed, 16 skipped in each (the skips: the CUDA and Triton tests, and K1 without `PVI_SLOW`); no test was
changed. The v0 suites that run the verifier's derive (`test_fullcheck`, `test_wire`, `test_fast_verifier`,
`test_layouts`, `test_lean_and_verifier_device`, `test_generation`, `test_prob_max`, `test_token_sampler`,
`test_fs_tree`): 456 passed, 43 skipped.

## 5. Laptop timings

Windows laptop, torch 2.2.2 CPU, 2 threads, no native kernels (no compiler), shared with other processes: medians
of interleaved runs, with run-to-run noise of about +-25% on whole queries at these small sizes.

### 5.1 The products alone at Llama-2-7B's prompt length

`v1_phases.py --micro 2048 2048 --reps 5 --paths 0 1 scatter --threads 2`: one op of 2,048 rows at `T = 2,048`, every
result checked equal to today's; ns per entry:

| Product | today (`0`) | table products (`1`) | buckets (`scatter`) |
|---|---:|---:|---:|
| final check, requant op (int8 values, 1 exception per 100,000) | 7.29 | **3.61** | 9.97 |
| final check, residual op (int16 `Delta`, 7,001 values) | 7.97 | **4.51** | 4.45 (routed to the table products: `V > T`) |
| `_xbar` (int8-valued input) | 3.09 | **2.23** | 15.46 |
| **Llama-2-7B per block** (34,304 requant rows, 8,192 residual rows, 23,296 `_xbar` columns), s | 0.79 | **0.44** | 1.51 |

The table products are 1.8x faster than today's limb GEMM on these shapes. In torch the buckets are slower than
today's code: eight `scatter_add_` passes (about 0.6 ns per entry each) plus the buckets' reduction, which is as
large as the row itself when `8 V = T`; forced onto the residual op (`V = 7,001 > T`) they cost 322 ns per entry,
which is why `scatter` mode keeps them to `V <= max(256, T)`. They are the right kernel in C, not in torch.

### 5.2 Whole V1 queries

`v1_phases.py --small 256 4 64 688 512 --seq 256 --fs --lmax 18 --threads 2`: a Llama-shaped decoder of width 256
(FFN 688, the last block pruned), 256 tokens, Fiat-Shamir, the eager CPU prover; seconds (medians):

| Phase | Kpre, 3 blocks, 5 queries: `0` | `1` | C (`cnn16c`, cut col matrices), 2 blocks, 3 queries: `0` | `1` | `scatter` |
|---|---:|---:|---:|---:|---:|
| `verify_decode` | 0.019 | 0.018 | 0.013 | 0.012 | 0.010 |
| `verify_derive` | 0.035 | 0.030 | 0.018 | 0.017 | 0.022 |
| `verify_gkr` | 0.191 | **0.094** | 0.076 | **0.041** | 0.057 |
| `verify_final` | 0.134 | **0.108** | 0.066 | **0.042** | 0.148 |
| of which `_xbar` | 0.014 | 0.004 | 0.004 | 0.002 | 0.048 |
| of which `_final_ok` | 0.070 | 0.054 | 0.036 | 0.022 | 0.078 |
| `verify_logup` | 0.038 | 0.038 | 0.046 | 0.027 | 0.031 |
| **verifier total** | **0.421** | **0.319** (1.32x) | **0.252** | **0.158** (1.59x) | 0.300 |

(Kpre: 1.49 M leaves in 7 instances, 1,054 GKR rounds; C: 0.81 M leaves, 3 instances, 477 rounds. An earlier
3-query Kpre run, before the table cache and the one-challenge path and with other processes busy, gave 0.320 s,
0.324 s and 0.471 s for `0`, `1` and `scatter`: at this size the products are a small part of the verifier and the
noise is of the same order as the gain.) At `T = 256` most of `verify_final` is fixed per-op and per-instance work
that both paths share: the same `verify_final` block replayed on one query's captured state, the two paths
alternated 8 times, takes 133 ms with today's code and 100 ms with V-V1N's, of which 46 / 35 ms are the eq tables of
the final points (today's code), 74 / 51 ms the final check and 8.6 / 4.9 ms `_xbar`. The GKR check alone, 7
instances of `n = 17` (synthetic transcripts): 91 ms one by one, 60 ms batched, 51 ms with the one-challenge path, of
which 30 ms are the challenges and absorptions. The batch-inverted logUp identity costs what the cross-multiplied one
does in torch (it needs as many field products; it is constant per query).

### 5.3 What this means for the Xeon (a model, not a measurement)

The re-time's per-block phases are 1.42 s (final check and `_xbar`), 0.54 s (derive) and 0.39 s (GKR). If the
laptop's ratios carried over (1.8x on the products, about 2x on the GKR, the derive unchanged), a Llama-2-7B block
would take about 0.8 + 0.54 + 0.2 = 1.5-1.6 s instead of 2.5 s, the full model about 50-55 s instead of 75-80 s,
against v0's 19.6 s. The torch path therefore closes part of the gap; the plan's target (the final check and `_xbar`
at most 0.1 s per block, V1 within 1.2x of v0) needs the native kernel of Sec. 7 and the derive's int32 path. The
Xeon cells of Sec. 6 measure what the torch path actually gives there.

### 5.4 Xeon measurements (job 1009430, s-005, git ebf6f4e)

Xeon Silver 4114, 8 threads, the native kernels built; GPU prover (2080 Ti); Llama-2-7B blocks at 2,048 tokens,
Kpre, Fiat-Shamir, 3 timed queries after a warm-up on each path, every query accepted, byte accounting exact.

**Suites.** `PVI_V1_BUCKETS=1`, `scatter` and `0`: 246 passed, 1 skipped each (6 min each), GPU tests included.

**The products alone** (`v1_phases.py --micro 4096 2048 --reps 5 --threads 8`), ns per entry:

| Product | today (`0`) | table products (`1`) | buckets (`scatter`) |
|---|---:|---:|---:|
| final check, requant op | 15.6 | **5.23** | 9.05 |
| final check, residual op | **6.63** | 8.62 | 8.58 |
| `_xbar` | 5.33 | **2.79** | 14.5 |

On the Xeon the table products win on the requant ops (3.0x) and `_xbar` (1.9x) but lose on the residual ops (1.3x
slower), unlike the laptop (where they won on all three); `buckets._method` should therefore route residual ops to
today's limb GEMM on this CPU (not changed yet).

**Whole V1 queries**, verifier seconds (medians), and the per-block slope from the 2- and 3-block cells:

| Phase | 1 block: `0` / `1` | 2 blocks: `0` / `1` | 3 blocks: `0` / `1` | per block: `0` / `1` |
|---|---:|---:|---:|---:|
| `verify_decode` | 0.052 / 0.050 | 0.251 / 0.161 | 0.414 / 0.276 | 0.163 / 0.115 |
| `verify_derive` | 0.115 / 0.137 | 0.648 / 0.725 | 1.241 / 1.253 | 0.593 / 0.528 |
| `verify_gkr` | 0.075 / 0.064 | 0.406 / 0.249 | 0.748 / 0.437 | 0.342 / **0.188** |
| `verify_final` | 0.183 / 0.147 | 1.626 / 1.473 | 3.007 / 2.751 | 1.381 / 1.278 |
| `verify_logup` | 0.034 / 0.037 | 0.099 / 0.122 | 0.112 / 0.146 | 0.013 / 0.024 |
| **verifier total** | 0.480 / 0.455 | 3.063 / 2.762 | 5.555 / 4.897 | **2.49 / 2.13** |
| v0 verifier (same queries) | 0.183 | 0.799 | 1.558 | 0.76 |

Full Llama-2-7B (one pruned block plus 31 full ones): V1 `0` about 0.48 + 31 x 2.49 = 77.7 s, V1 `1` about
0.46 + 31 x 2.13 = 66.6 s. V-LONG-a's measured v0 is 19.6 s (Fiat-Shamir), so V1 is 3.4x v0 with the new path
(4.0x before).

**Reading.**
* The GKR check gains 1.8x per block (batched chains and the one-challenge path), as on the laptop.
* The final check gains only 7% per block, against 1.8x on the laptop and against what the micro rates predict:
  with Sec. 5.1's per-block shapes (34,304 requant rows, 8,192 residual rows, 23,296 `_xbar` columns at 2,048
  tokens) the products alone would take about 1.46 s on path `0` and 0.65 s on path `1`, yet the whole of
  `verify_final` is 1.38 s and 1.28 s. So either the micro-benchmark (one op of 4,096 rows) runs slower than the
  real ops, or the real ops do not take the routes it measures. The profile below (job 1011404) answers it.

**Where `verify_final` goes** (job 1011404, s-005, git 74136a5; `v1_phases.py --model llama2-7b --layers 2|3 --seq
2048 --mode Kpre --fs --lean --lmax 24 --reps 3 --paths 0 1`, medians; per block = 3 blocks minus 2):

| Part of `verify_final` | 2 blocks: `0` / `1` | 3 blocks: `0` / `1` | per block: `0` / `1` |
|---|---:|---:|---:|
| `final_points` (eq tables; the prover's calls in `prove_fold` included, see below) | 0.586 / 0.582 | 1.075 / 1.003 | 0.489 / 0.421 |
| `final_xbar` (`_xbar`; `buckets.xbar` on `1`) | 0.246 / 0.131 | 0.382 / 0.252 | 0.137 / 0.122 |
| `final_kpre_y` | 0.053 / 0.056 | 0.091 / 0.090 | 0.039 / 0.034 |
| `final_lw` (`_lw_e_old` on `0`, `buckets.lw_times` on `1`) | 1.181 / 1.058 | 2.041 / 1.874 | **0.860 / 0.816** |
| `final_dot` (the inner products) | 0.062 / 0.063 | 0.113 / 0.104 | 0.051 / 0.041 |
| `verify_final` | 1.856 / 1.643 | 3.243 / 2.878 | 1.388 / 1.236 |

No op fell back from `buckets.lw_times` (`n_lw_new_fallback` 0; `_lw_e_old` ran for 9 and 16 ops on path `0`). So:

* **The products by value give almost nothing in place: 0.86 -> 0.82 s per block.** Per entry (87 M values per
  block: 34,304 requant and 8,192 residual rows at 2,048 tokens) that is 9.9 ns on path `0` and 9.4 ns on path `1`,
  against the micro-benchmark's 13.9 and 5.9 ns (weighted): today's path is faster in place than in the micro, and
  the table products slower. Why is not measured yet; the candidates are the real ops' shapes and value ranges
  (the micro uses one 4,096-row op with a fixed range) and the dtype in which the decoded values arrive (the micro
  passes int8 to the table products and int64 to today's path). Either way the torch table products are not the
  lever on this CPU; the native bucket kernel of Sec. 7 is, at its estimated 0.03-0.05 s per block.
* **`final_points` is not the verifier's alone.** The prover calls the same `_points` in `prove_fold` (once per
  GKR instance, on its device), and the wrapper adds both, so the 0.42-0.49 s per block are mostly the prover's.
  The verifier's own eq tables are small: `extfield.eq_table` takes 7 ms for `n_col` = 11 and 9 ms for `n_row` = 13
  on the laptop (2 threads), about 16 ms per instance, so about 0.08 s per block (5 instances per block at `lmax`
  24). They are not a target.
* `_xbar`, the Kpre check and the inner products together cost about 0.2 s per block; the rest of `verify_final`
  (about 0.2 s per block) is the verifier's eq tables and the bookkeeping of the folds.

With the native products at about 0.05 s per block, `verify_final` would drop from 1.24 to about 0.47 s per block,
and V1's verifier from about 2.1 to about 1.4 s per block (full model about 43 s, 2.2x v0's 19.6 s): above the plan's
1.2x, so the derive's int32 path (Sec. 7) is needed as well.
* The model of Sec. 5.3 (50-55 s) was optimistic; the measured torch path gives 67 s. The plan's target (V1 within
  1.2x of v0, about 24 s) needs the native kernel and the per-op fixed work removed. Under the plan's kill rule (no
  native kernel merged and rehearsed by Wed 28 Oct), the paper reports V1 as a bytes option with this verifier time.

### 5.5 The native products (`native_kernels.lw_affine`, commit e4eb170)

The profile makes `lw_times` the one target. The native kernel does not use the buckets of Sec. 2.2 but the table
products' own decomposition (Sec. 2.3), because it needs no per-row table of `V` buckets and so serves the residual
ops (`V` up to 7,001 on the random-weight decoders) as well as the requant ops (`V` = 256):

* per row `i` of the values `s` `[n, t]` (int8, int16, int32 or int64, read as they arrive), with `v = s_ij - a`,
  three accumulators of 8 int64 lanes: `A1 += v e_j`, `A2 += C[v] e_j`, `A3 += D[v] e_j` (one `#pragma omp simd`
  loop over the 8 coefficients of `e_j`); rows in parallel (OpenMP);
* every term is below `2^15 2^31 = 2^46` (`V <= 2^15`, `C < V`, `D in {0, 1}`, `e` canonical), and a row is reduced
  every `2^16` entries, so no accumulator overflows int64;
* `buckets.lw_times` then forms `L e = lo(a) (1 . e) + w0 A1 + A2` and `W e = w0 (1 . e) + A3` exactly as it does
  from the torch table products, and adds the exceptions as before.

The integers are the torch table products' (and so today's GEMM's). The kernel checks its operands (`e` canonical,
the tables in range, `V <= 2^15`) and returns nothing if a value lies outside `[a, a + V)`; the torch path then runs,
so its behaviour on malformed input is unchanged. It is used on a CPU under `PVI_V1_BUCKETS=1` when the library is
built; `PVI_V1_NATIVE=0` switches it off. Work per block: one pass over 87 M values with 24 int64 multiply-adds each
(three 8-lane vectors), about 0.05-0.1 s on 8 Xeon threads by the count, against 0.82 s now.

Tests: `test_native_lw_equals_the_torch_table_products` (four dtypes; requant ops with exceptions at 37 x 2,048 and a
power-of-two multiplier; residual ops of `V` = 7,001 and 81; the native result against the torch table products and
today's GEMM; an out-of-range value, a short `e` and a non-canonical coefficient declined). It is skipped where the
library cannot be built (the Windows laptop). Job 1011440 builds it on the Xeon node, runs the V1 suites with the
native kernel on and off, the micro-benchmark and the Llama-2-7B profile: [CLUSTER].

## 6. Cluster commands

Run as job 1009430 (Sec. 5.4). From the repository root of a lean clone of this branch on the TAU
cluster, as the C2 Sec. 8.x re-time (Xeon 4114 node of `studentbatch`, 8 threads, the native kernels built with
`CXX`, Triton for the GPU prover as in `I_server_setup.md`); `$ROOT` the clone, `$PY` its venv's python:

    ENVS="cd $ROOT/code && export PYTHONPATH=$ROOT/code/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 \
    CXX=$CXX PVI_NATIVE_DIR=$ROOT/logs/v1n/.native PVI_PY_INCLUDE=$PVI_PY_INCLUDE TRITON_CACHE_DIR=$ROOT/logs/v1n/.triton"
    mkdir -p $ROOT/logs/v1n
    # 1. the products alone at Llama-2-7B's shapes (no prover, minutes): ns per entry and the per-block projection
    sbatch -p studentbatch -A gpu-students --cpus-per-task=8 --mem=16G --time=0:30:00 -J v1n-micro \
        -o $ROOT/logs/v1n/%x-%j.out --wrap "$ENVS; $PY experiments/8_v1/v1_phases.py --micro 4096 2048 --reps 5 \
        --paths 0 1 scatter --threads 8"
    # 2. V1 on 1, 2 and 3 Llama-2-7B blocks at 2,048 tokens, Kpre, Fiat-Shamir: today's verifier (0) and V-V1N's (1)
    #    on the same queries, 3 timed queries each after a warm-up (timed.v1_paths.{0,1}.timings), byte accounting
    #    and acceptance checked as in C2 Sec. 8.x
    for L in 1 2 3; do
      sbatch -p studentbatch -A gpu-students --gres=gpu:1 --cpus-per-task=8 --mem=128G --time=3:00:00 \
          -J v1n-llama7-$L -o $ROOT/logs/v1n/%x-%j.out --wrap "$ENVS; $PY experiments/8_v1/v1_bytes.py \
          --model llama2-7b --layers $L --seq 2048 --modes Kpre --policy auto --prune-last --fs --lean --full \
          --reps 3 --lmax 24 --v1-paths 0 1 --label v1n"
    done
    # 3. the suites with the switch in each position, on the node with the native kernels
    for M in 1 scatter 0; do
      sbatch -p studentbatch -A gpu-students --gres=gpu:1 --cpus-per-task=8 --mem=32G --time=2:00:00 -J v1n-tests-$M \
          -o $ROOT/logs/v1n/%x-%j.out --wrap "$ENVS PVI_V1_BUCKETS=$M; $PY -m pytest tests/test_v1_buckets.py \
          tests/test_cut_protocol.py tests/test_cut.py tests/test_logup.py tests/test_logup_gkr.py tests/test_lookups.py \
          tests/test_extfield.py tests/test_gkr_triton_cpu.py tests/test_gkr_triton.py -q"
    done

`v1_bytes.py --v1-paths` was smoke-tested on the laptop (`--model gpt2 --layers 2 --seq 16 --modes Kpre --prune-last
--fs --lean --full --reps 2 --lmax 16 --v1-paths 0 1 --device cpu`): byte accounting exact, 2/2 queries accepted on
each path, exit status 0. The per-block slope is the difference of the 2- and 3-block cells (`grep '^{' logs/v1n/v1n-llama7-*.out`, then
`timed.v1_paths["0"|"1"].timings` per phase and `verify_s`); the 1-block cell (its only block pruned) has only the
pruned block's cut ops. On the GPU node the suites also run the GPU tests (`test_v1_with_a_gpu_prover_and_verifier`,
`test_gkr_triton.py`), which the laptop skips. If the 3-block cell needs more host memory than 128 GB, use the
re-time's setting.

## 7. What remains

* **The native kernels** (`native_kernels.py`, the plan's target of at most 0.1 s per block for the final check and
  `_xbar`). The algebra and its exactness argument are Sec. 2.2: per row, zero `V x 8` int64 buckets (16 KB for a
  requant op), one 8-lane add per entry (`B[s + 128] += e_j`, one AVX-512 `vpaddq`), then per bucket two
  multiply-adds mod `p` with the op's `lo(v) mod p` and `W(v)`, and the exceptions as here; `_xbar` the same per
  feature with the identity table, over tiles of features so a tile's buckets stay in L2. A rough count for the Xeon:
  2-3 cycles per entry plus `16 V` per row, about 0.03-0.05 s per Llama-2-7B block on 8 threads, bandwidth-bound on
  the int8 values (87 MB per block). The buckets pay off only where `V <= T` (a requant op's 256 values at 2,048
  tokens; not a residual op's `Delta` at short prompts, where the table products or a direct product are better:
  `buckets._method` already routes so). The tests here are the kernel's reference (`lw_times(method=...)`, `xbar`).
  Not written: it cannot be compiled or tested on the laptop.
* **Part 5**, the same kernel in Triton with `gkr_triton`'s BabyBear arithmetic, for the GPU verifier (V-GPUFLOOR).
* **V1's clear claims as int32 on the native path.** v0 keeps a native CPU verifier's claims int32 (`cpu32` in
  `protocol._run_query`); V1 still widens its clear ops' claims to int64 (only the cut values are narrowed). This is
  probably part of V1's derive gap (0.54 against v0's 0.38 s per block); mirroring v0's rule is a small change, but
  the native int32 path cannot run on the laptop, so it is left for a cluster session.
* **The GKR's challenges.** After part 3 the GKR verifier's cost is mostly the sequential challenge derivation and
  absorption, about 20 us per round on the laptop (two SHAKE-256 calls and Python; about 1,700 rounds per
  Llama-2-7B block at `lmax` 24). A native or vectorised SHAKE would not help much (each key depends on the
  transcript so far); fewer, larger instances (`lmax` 27 on a large card) cut the rounds.
* Per-instance and per-op fixed work in `verify_final` that both paths share (today's code): the eq tables of the
  final points (`_points`, `ef.eq_table`: two `ef.mul` per variable; 5-6 ms per instance on the laptop, about 35-46 ms
  of the small Kpre model's 100-133 ms) and the three small exact inner products per op in `_final_claims` (`S`,
  `chi (L e)`, `chi (W e)`, which could be one). They matter on small models; at 2,048 tokens the products dominate.
* The plan's validation items not done here: the Xeon timings (Sec. 6), the GPU paths (device-generic code, not run:
  the laptop has no GPU), and the adversarial-review workflow re-run on the changed verifier.

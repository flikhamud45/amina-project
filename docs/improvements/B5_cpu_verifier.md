# B5. The CPU verifier at long prompts: native exact kernels for every phase

Plan item B5 (reviewer objection D1: "is verification cheaper than re-running the model?" on a CPU client; mock
review, Reviewer C, weakness 4: "the verifier does not realise its asymptotic advantage"). Commits e501882 and
7606e8a (native attention), c36cfa8 and 1f85d21 (hybrid attention, a negative result; the CPU profiler), 1e88df8
(the recomputed operations), b7ef9a9 and df11e5d (the field products), 11d227f (the claim decoder), d473455 (int32
claims), 4719b71, 33dc0f8, 4ab532f and 6f9a8c8 (AVX-512 attention, fused with the output requantisation), 63732b1
(parallel min/max, the norm's division). Prerequisite I4 (a C++ compiler on the compute nodes,
`I_server_setup.md` Sec. 4). Status: in progress; Sec. 6 has the numbers so far and what is still to measure.

## 1. The problem

A client without a GPU verifies on its CPU. On the Xeon Silver 4114 node (8 threads), the Kpre verifier of the full
Llama-2-7B took about 85 s at 2,048 tokens, as long as re-executing the model in fp32 (86 s; F1,
`F1_reexecution_baseline.md`), although the verifier skips every weight product. Its three phases were all
memory-bound torch code on int64 tensors:

* `verify_decode`: the PVC3 claims decoded by numpy passes (four per lane of each slot stream), then widened from
  int32 to int64;
* `verify_derive`: every operation without weights recomputed. Attention materialised `T x T` score tensors with
  about a dozen element-wise passes; each requantisation, residual addition, norm, RoPE and table look-up was several
  torch passes with temporaries, and many of them read the transposed view that a weight op's fold hands its readers
  (`MatOp.fold` returns `Z.T`), i.e. memory with a stride of `T` elements;
* `verify_products`: Freivalds' checks, `chi Z mod p` and `u^T X mod p`, as float64 limb GEMMs after float64
  copies of every claim and input.

## 2. Where the time went (profiles)

`experiments/6_improvements/cpu_profile.py` times every recomputed operation (`CheapOp.fn`) and every fold during
one query, grouped by the op's name without the block index, then ranks torch operators by self CPU time. Llama-2-7B,
2 blocks (the last pruned), 2,048 tokens, Kpre, tiled native attention (job 1005757):

| Operation | Seconds | Share of the derive |
|---|---:|---:|
| attention (native, tiled) | 0.379 | 29.9% |
| RoPE | 0.206 | 16.3% |
| requantisation | 0.163 | 12.9% |
| SiLU table | 0.155 | 12.2% |
| norm | 0.151 | 11.9% |
| SwiGLU product | 0.125 | 9.9% |
| residual addition | 0.078 | 6.2% |
| derive total | 1.268 | |

Per full block (the slope between 2 and 3 blocks, job 1005733) the verifier took 2.23 s: decode 0.40, derive 1.51,
products 0.33.

## 3. The kernels

All kernels live in `native_kernels.py`, one C++ source compiled once per host (`$CXX -O3 -march=native -fopenmp
-fwrapv`) and loaded with ctypes. Each Python wrapper returns the torch op's exact result or `None`, in which case the
caller runs its torch steps; so a host without a compiler, a GPU tensor, an unusual layout or an out-of-range operand
takes the old path. `-fwrapv` makes int64 overflow wrap as torch's does, so even adversarial values give the same
integers.

**Attention, scalar tiled (e501882, 7606e8a).** Per (batch, head) and tile of eight query rows: int32 scores against
the keys up to each row's position, the row maximum, `e = LUT[min(max - s, len - 1)]`, `tot = sum e`,
`p = floor((tot + 510 e) / (2 tot))` and `p v` in int64, without any `T x T` tensor.

**Attention, hybrid (c36cfa8): a negative result.** The two products as float32 GEMMs (torch's BLAS; exact, as in
`_attention_core`) and the integer softmax in one native pass per 32-row block. It is slower than the tiled kernel
at every shape measured (Table in Sec. 6), because the GEMMs work on blocks too small to amortise their packing.
It stays selectable (`PVI_CPU_ATTN=hybrid`) for CPUs with faster BLAS.

**Attention, AVX-512 (4719b71, 33dc0f8, 4ab532f, 6f9a8c8).** On CPUs with AVX-512BW and `dh` a multiple of 16 up
to 128 (`pvi_attention64`): per head, the keys as int16 pairs along `dh`, stored per chunk of 16 keys, and the values
as int16 pairs of consecutive keys, `[T/2][dh]`, both read straight from the int64 `[B, T, H, dh]` tensors by their
strides (no copies). Scores of 16 rows by 16 keys are 16 `vpmaddwd` accumulators (int16 pairs into int32;
`|s| <= 2^21`). The softmax is three vectorisable passes per row: the maximum; `e = EXP[min(255, (gap m_s + 2^29)
>> 30)]` from the 256-entry table, which equals `LUT[min(gap, len - 1)]` because the table is non-increasing and
zero from index 178 on, where the LUT's last entry lies (checked for every gap at four multipliers, and by tests),
without the LUT's cache misses; and `p = floor((tot + 510 e) / (2 tot))` as the truncated double product with
`1/(2 tot)` (within `2^-44` of the quotient, which is below 256) corrected by one integer step each way, i.e. the
exact floor. The probabilities are stored as int16, which `P V` reads as int32 pairs of consecutive keys; `P V`
lets 8 rows share each load of `V` and accumulates in int32 (every partial sum is below `(255 + T/2) 128 < 2^24`
for `T < 2^15`, the bound `_int32_scores` already enforces). On a CPU verifier, `transformer._attention` calls the
fused form (`attention_requant`): it also requantises the output by `m_o`, as `requant()` does, and writes the int8
values in the final `[B, Tq, Hq dh]` layout, with grouped-query heads read through a (kv head, repeat, position)
view. Other CPUs and shapes use the scalar kernel (`PVI_NATIVE_AVX512=0` forces it).

**Recomputed operations (1e88df8).** One pass each, int64 in and out: requantisation
`clamp((z m + 2^29) >> 30, lo, hi)`, the residual addition, the table look-ups (SiLU, GELU, ReLU), the integer norm
(per row: the mean for LayerNorm, the sum of squares, `isqrt` by a float64 root with one integer correction each way,
as `_isqrt`'s numpy path; a row whose sum of squares wrapped negative sends the whole tensor to torch's rules) and
RoPE (`[x1 cos - x2 sin | x2 cos + x1 sin]` from the same cached tables). A matrix operand is either contiguous or the
transposed view of a claim; the latter is read in 64 x 64 tiles, so reads and writes stay in cache. The helpers in
`graph.py` (`requant`, `residual_add`) and `transformer.py` (`_norm_int`, `_lut`, `_rope`) dispatch to them, so the
graph, its digest and the prover are unchanged.

**Field products (b7ef9a9).** `chi @ Z mod p` for `chi` in the field and `|Z| < 2^31`: `chi = c0 + 2^16 c1` with
`c0 < 2^16`, `c1 < 2^15`, so each product is below `2^47` and `2^15` of them sum below `2^62`; the int64
accumulators are reduced mod `p` every `2^15` terms. Rows of `Z` are streamed (the claims, `chi Z`) or columns are
dotted (the inputs, `u^T X`, with `X` the `[M, K]` activations). The verifier's `_signed_lhs` and `_rhs_all` use it
on the CPU; any operand out of range (for example a prover's `u` outside the field in mode C) returns `None` and the
limb GEMMs decide as before.

**Claims as int32 (d473455) and their range checks (63732b1).** A CPU verifier with the native kernels decodes
the claims as int32 (no widening pass); the requantisation, residual addition and field products read int32 or
int64 claims and compute in int64; a weight op's fold stays int32 only when every reader is such a requantisation
(else it is widened as before). The range check of every claim is one parallel min/max pass instead of numpy's two
single-threaded ones. The norm's floor division uses the row's reciprocal with one exact correction each way.

**Claim decoder (11d227f).** `claimcodec._stream_jobs` hands each large slot stream to one native pass over all 32
lanes: the `k`-bit values, each segment's base added with uint32 wrap-around, and the padding check (a non-zero
padding slot raises the same `ClaimCodecError`). The Rice-coded exceptions, their patching and the row means are
unchanged.

## 4. Exactness

Every kernel computes the same integers as the torch code it replaces; no kernel changes the protocol, the proof or
the transcript. Tests (on the node; on Windows the native paths are skipped and the fallbacks run):

* `tests/test_native_cheap.py`: requantisation and residual addition on contiguous and transposed operands,
  including products that wrap int64; the SwiGLU product and `requant_fn`; the norm (RMS and LayerNorm, zero rows,
  extreme values, wrapped sums of squares); table look-ups; RoPE with offsets (pruned blocks); the field products
  (both layouts, `n` above `2^15` to exercise the reduction, out-of-range operands returning `None`) against
  `field_matmul_mod`; the native decoder against the numpy decoder and the original claims.
* `tests/test_gpu_verifier.py::test_the_native_cpu_attention_gives_the_int32_cores_integers`: 15 cases (five
  shapes up to 1,024 keys, grouped-query heads, one and seven query rows, `dh` 64-128, random, peaked and extreme
  scores) for the scalar, hybrid and (where the CPU has it) AVX-512 attention against `_attention_core`.
* The CPU verifier suites with every kernel active (`test_fast_verifier`, `test_fullcheck`, `test_protocol`,
  `test_pruning`, `test_lean_and_verifier_device`, `test_generation`, `test_real_weights`, `test_cut_protocol`,
  `test_architecture`, `test_claimcodec`, `test_wire`): 614 passed (job 1005827). Three failures on the way were
  tests of implementation details, not verdicts, and failed identically with the kernels off where applicable: a V1
  GPU test expected a mode-C tamper to be caught at the final check, but it is caught at the column check (the final
  claim reduces to the folded rows; 3ede5fc), a test expected Kpre's stacked `u` to be cached (now cached on the
  native path too, df11e5d), and the split-decoder test counts calls of the numpy lanes (it now switches the native
  unpack off, and a new test compares both decoders on 400 corrupted proofs).

## 5. Usage

    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model llama2-7b --layers 2 --seq 2048 \
        --modes Kpre --queries 3 --lean --wire --prune-last on --lookups on --threads 8 --verifier-device cpu
    PYTHONPATH=src python experiments/6_improvements/cpu_profile.py --model llama2-7b --layers 3 --seq 2048
    PYTHONPATH=src python experiments/6_improvements/cpu_attn_bench.py --heads 32 --kv 32 --dh 128 --seq 2048

Switches: `PVI_NATIVE_ATTN=0` (no native attention), `PVI_CPU_ATTN=tiled|hybrid`, `PVI_NATIVE_CHEAP=0` (no native
recomputed operations or field products), `PVI_NATIVE_DECODE=0` (numpy decoder). `$CXX` and `$PVI_NATIVE_DIR` choose
the compiler and the build cache (`I_server_setup.md` Sec. 4).

## 6. Measurements (Xeon Silver 4114, 8 threads; Llama-2-7B, Kpre, 2,048 tokens, 3 queries, the last block pruned)

Attention alone (one block's 32 heads, `cpu_attn_bench.py`, job 1005733), seconds:

| Shape | torch int32 | native tiled | native hybrid | fp32 SDPA (re-execution) |
|---|---:|---:|---:|---:|
| 32 heads, dh 128, T 2,048 | 1.315 | 0.278 | 0.446 | 0.153 |
| the same, one query row | 0.0077 | 0.0101 | 0.0297 | 0.0017 |
| 32 heads, dh 64, T 2,048 | 1.174 | 0.175 | 0.265 | 0.084 |
| 32 heads, dh 128, T 512 | 0.040 | 0.020 | 0.040 | 0.016 |

The verifier by number of blocks (seconds; every query accepted):

| Kernels | 1 block | 2 blocks | 3 blocks | per full block (slope) |
|---|---:|---:|---:|---:|
| torch (job 1005733) | 0.428 | 3.371 | 6.639 | 3.27 |
| tiled attention (job 1005733) | 0.484 | 2.393 | 4.627 | 2.23 |
| + recomputed operations (job 1005765) | 0.368 | 2.018 | 3.711 | 1.69 |
| + field products and decoder (job 1005787) | 0.400 | 1.952 | 3.335 | 1.38 |
| + AVX-512 attention, first form (job 1005788) | 0.361 | 1.713 | 2.989 | 1.28 |
| + int32 claims, min/max, norm, attention without copies (job 1005827) | 0.313 | 1.480 | 2.617 | 1.14 |
| + attention fused with its requantisation (job 1005854) | -- | 1.530 | 2.596 | 1.07 |
| + softmax and P V rewritten (job 1006647) | \[pending] | | | |

Per full block at the last complete step (job 1005827): decode 0.19 s, derive 0.76 s, products 0.19 s. The full
model extrapolates to about 0.31 + 31 x 1.14 = 36 s, against 86 s for fp32 and 196 s for integer re-execution on the
same CPU (F1).

The AVX-512 attention by phase (job 1005854, per thread, 32 heads, T = 2,048): preparing keys and values 4 ms,
scores 54 ms, softmax 87 ms, `P V` 46 ms. The softmax's LUT gather (up to 1 MB) and scalar division, and `P V`'s
re-reading of `V` for every two rows, motivated the rewrite of 6f9a8c8.

## 7. What remains

* The full model (32 blocks) against fp32 re-execution on the same CPU (F1), and the same on the EPYC node.
* Claims as int32 end to end on the CPU (no widening pass in the decoder, half the memory traffic of the products
  and requantisations).

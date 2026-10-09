# A1. Column opening and folding on exact int8 GEMMs

Plan item A1. Commit `e2bcb31` (branch `sp2027`). Status: done, measured on an RTX 2080 Ti; L40S
numbers come with the final runs.

## 1. The problem

In setting C the prover answers two challenges per query besides the forward pass:

* **the fold**: `u = chi^T A` for every row-layout weight matrix `A = [W | b]` (`r` challenge rows of
  `N` field elements against `N` int8 rows), and
* **the opening**: the `t` challenged columns of the Reed-Solomon-encoded weights, recomputed from `W` as
  `E[:, C] = W V[:, C] + b V[k, C]`, where `V` is the Vandermonde matrix of the evaluation points
  (for the transposed layout, `sum_i W_i^T V_i`).

Both are products of an **int8 matrix** with **field elements** (below `p = 2^31 - 2^27 + 1`). The code
computed them in float64 (`field.small_matmul_mod`): products of an int8 value and a field element are
below `2^39`, so float64 GEMMs are exact over blocks of `2^13` terms. On a GPU, float64 is slow (the L40S
has about 1.4 TFLOP/s of float64 against 366 T int8 MAC/s), and the stored L40S records show it:

| Llama-2-7B, 64 tokens, mode C (L40S) | time |
|---|---|
| `prove_open` | 1.105 s |
| `prove_fold` | 0.144 s |
| `prove_forward` | 0.117 s |
| `prove_encode` | 0.413 s |
| prover total | 1.78 s |

The opening alone was 62% of the prover (69% for Llama-2-13B: 1.888 of 2.75 s), and it ran at about
0.42 T MAC/s, the float64 rate.

## 2. The idea: byte decomposition of the field operand

A field element `f` in `[0, p)` fits in an int32. Write it as four signed bytes: flipping the sign bit
of bytes 0-2 (`f XOR 0x00808080`) gives bytes `s_0..s_3`, each in `[-128, 127]`, with

    f = sum_{c=0..3} 256^c (s_c + o_c),   o = (128, 128, 128, 0),

for **every** int32 value (the top byte keeps its sign). Then, for an int8 matrix `W [N, K]` and field
elements `F [K, R]` (`R = t` columns of `V`, or the `r` challenge rows),

    W F = sum_c 256^c (W S_c + o_c (W 1)),

where `S_c [K, R]` holds byte `c` of every entry and `W 1` is the vector of row sums. All four `W S_c`
and `W 1` come out of **one** int8 GEMM of `W` with the `[K, 4R + 1]` matrix of the bytes and a column
of ones. The results are then combined in int64 and reduced mod `p`.

**Exactness.** Each product of two int8 values has magnitude at most `128 * 128 = 2^14`. A block of
`2^16` products therefore sums to at most `2^30` in magnitude, inside the int32 accumulator of the
int8 GEMM, whatever the summation order of the kernel. Longer contractions (`K > 2^16`) are split into
blocks whose int32 results are added in int64. Every intermediate is an exact integer, and the final
reduction gives the canonical residue in `[0, p)`, so the result is **bit-identical** to the float64
product.

This is the same decomposition the verifier already used for its products on a GPU
(`field.int8_small_matmul`, tested for every int32 value); the new function only swaps the roles of the
operands.

## 3. Implementation

* `field.int8_weight_matmul(small, field)`: `small @ field mod p` for int8 `small [M, K]` and field
  elements `field [K, R]`, as `int8_small_matmul(int8_right(field.T), small).T`.
* `field.weight_matmul_mod(small, field)`: the dispatcher. It uses the int8 path when `small` is int8 on a
  CUDA device and `field.int8_ok(device)` holds, else the float64 `small_matmul_mod`.
* `field.int8_ok(device)` (existing) runs `torch._int_mm` once per device on random operands and on the
  extremes (`-128` everywhere, the `2^30` accumulation bound) and compares with float64; a GPU whose int8
  GEMM is missing or inexact falls back automatically. The RTX 2080 Ti (sm_75) passes.
* Call sites: `WeightCommitment.fold`, `WeightCommitment.columns_at` and `TransposedCommitment.columns_at`
  (`commitment.py`). The bias terms are unchanged.

Nothing a verifier sees changes: the folded rows, the opened columns and the Merkle multiproofs are the
same bytes. Soundness and the Fiat-Shamir transcript are therefore unaffected.

## 4. Tests

* `tests/test_gpu_verifier.py::test_int8_weight_matmul_is_exact`: the opening pattern `W V` and the fold
  pattern `chi W` (with `W` transposed and not contiguous) against the reference product, at the int8
  extremes and `p - 1`, with `K` up to and beyond `2^16`; on the CPU (the float64 emulation of the int8
  GEMM) and on CUDA.
* `tests/test_gpu_exactness.py::test_group_openings_match_cpu`: a shared Merkle tree holding a row-layout
  matrix and a transposed (col-layout) matrix of fused q/k/v, opened on the device and on the CPU: the same
  columns and multiproof.
* The existing `test_commitment_fold_and_open_match_cpu` (a 4096 x 4097 layer) now exercises the int8
  path on a GPU.
* All 64 GPU tests passed on the 2080 Ti (job 1004091).

## 5. Measurements (RTX 2080 Ti, 64-token prompts, auto plan, 7 queries per variant, interleaved)

`experiments/6_improvements/open_ab.py` runs every query twice (float64 and int8, the same challenge
seed), checks that the folds, columns and multiproofs are identical, and that the verifier accepts.

| Build | opening, float64 -> int8 | fold | prover total |
|---|---|---|---|
| Llama-2-7B, 1 block | 218 -> 23 ms (9.7x) | 15 -> 2 ms | 277 -> 64 ms (4.3x) |
| Llama-2-7B, 2 blocks | 305 -> 32 ms | 30 -> 4 ms | 406 -> 106 ms (3.9x) |
| Llama-2-7B, 4 blocks | 473 -> 49 ms | 59 -> 7 ms | 653 -> 175 ms (3.7x) |
| Llama-2-13B, 1 block | 214 -> 25 ms | 24 -> 2.5 ms | 276 -> 63 ms (4.3x) |
| GPT-2, full | 82 -> 68 ms | 20 -> 14 ms | 173 -> 154 ms |

All outputs identical and all queries accepted. GPT-2 gains little because its matrices are small and its
opening is dominated by the Vandermonde columns and the multiproof.

**Projection for the L40S** (to be measured): the full Llama-2-7B prover at 64 tokens from 1.78 s to
about 0.6-0.7 s (opening about 0.1 s, fold about 0.02 s).

## 6. What remains

* The Kpre precompute (`Verifier._fold_local`) still uses float64; it is a one-time setup cost (plan A6).
* At 2,048 tokens the opening is no longer the bottleneck; the claim encoding is (plan A3).

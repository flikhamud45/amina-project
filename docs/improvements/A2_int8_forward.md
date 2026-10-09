# A2. The prover's forward pass on exact int8 GEMMs

Plan item A2. Commit `9ce4305` (branch `sp2027`). Status: done, measured on an RTX 2080 Ti.

## 1. The problem

The prover computes every weight op's claims `Z = W X + b` exactly. Before this change it did so with
float32 GEMMs over chunks of 1,024 terms (`graph.exact_matmul`): a product of two int8 values is at most
`2^14`, so 1,024 of them stay below `2^24`, the largest range in which float32 represents every integer;
each chunk's result is converted to int64 and the chunks are added. This is exact, but:

* float32 GEMMs run on the CUDA cores (TF32 tensor cores would round), at a fraction of the GPU's int8
  tensor-core throughput (L40S: about 91 TFLOP/s float32 against 366 T int8 MAC/s);
* a weight matrix is converted to float32 chunk by chunk on every query, and each chunk's float32 result
  is converted to int64 and accumulated: several passes over `[N, M]` outputs for `K = 4,096` (four chunks).

At 2,048 tokens the forward pass was 2.4 s of Llama-2-7B's prover and 4.4 s of Llama-2-13B's (L40S).

## 2. The idea

Every weight op's input is int8 by construction: it is produced by a requantisation clamped to
`[-127, 127]`, a look-up table whose values are within int8, or a norm whose output is clamped
(`MatOp.max_input = 127`). The weights are int8. So `W X` is a product of two int8 matrices and can run on
**int8 tensor cores** (`torch._int_mm`) with int32 accumulation:

* each product is at most `2^14` in magnitude, so a block of `2^16` terms sums to at most `2^30`: exact
  in int32 for any summation order;
* longer contractions are split into blocks of `2^16`, added in int64;
* the claim is the same integer as before, so nothing the verifier sees changes (claims, encodings,
  transcripts).

## 3. Implementation

* `graph.int8_product(w, x)`: exact `w @ x` (int64) for int8 `w [N, K]` and `x [K, M]`, via
  `field.int8_gemm` (which is `torch._int_mm` where `int8_ok`, with a run-time fallback). It returns
  `None` for shapes `torch._int_mm` cannot take (`N <= 16` or `K` not a multiple of 8, e.g. a first
  convolution with `K = 27`), and pads `M` with zero columns to a multiple of 8 (a pruned position, `M = 1`).
* `MatOp.compute` uses it when the input is on a CUDA device, `max_input <= 127`, the weights are int8 and
  `int8_ok` holds; otherwise the float32 chunks as before.
* Attention (`QK^T`, `PV`) still uses float32 GEMMs; making it int8 is plan item A4.

## 4. Tests

* `tests/test_gpu_verifier.py::test_int8_product_is_the_exact_weight_product`: int8 extremes (`-128`
  everywhere in a row and a column), `M = 1` (padding), `K = 70,000` (two blocks), against the float64
  product; on the CPU emulation and on CUDA.
* `tests/test_gpu_verifier.py::test_int8_product_declines_shapes_the_int8_gemm_cannot_take`.
* `tests/test_gpu_exactness.py::test_weight_op_forward_at_model_sizes`: `MatOp.compute` on the device at
  the real Llama-2-7B and ResNet shapes (q/k/v at 2,048 tokens, down projection `K = 11,008`, gate/up, the
  LM head GEMV, a 3x3x512 convolution), with the worst-case "max" and "alternating" patterns, TF32 on and
  off, against float64.

## 5. Measurements (RTX 2080 Ti, the lean wire prover's forward, 5 queries, claims identical)

`experiments/6_improvements/forward_ab.py` times `Prover.claims` both ways on the same queries and checks
that every claim is identical.

| Build | T = 64 | T = 512 | T = 2,048 |
|---|---|---|---|
| Llama-2-7B, 1 block | 0.61x | 0.51x | 0.49x |
| Llama-2-7B, 2 blocks | 0.56x | 0.44x | 0.51x (169 -> 87 ms) |
| Llama-2-13B, 1 block | 0.53x | | 0.45x |
| OPT-1.3B, 4 blocks | 0.74x | | 0.73x |
| GPT-2, full | 0.71x | 0.91x | |

(Ratios of the int8 time over the float32 time; lower is better.) The gain is about 2x on Llama-sized
layers; smaller models gain less because their forward pass is dominated by element-wise ops and attention.

## 6. What remains

* Attention on int8 GEMMs or a fused kernel (plan A4/B3).
* At 2,048 tokens the claim encoding dominates the prover (plan A3).

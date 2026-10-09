# B5. The CPU verifier at long prompts: compiled derive and a native exact attention

Plan item B5 (reviewer objection D1: "is verification cheaper than re-running the model?" on a CPU client; future
work "native verifier"). Commits e501882 (`native_kernels.py`), 7606e8a (query rows in tiles). Prerequisite I4
(a C++ compiler on the compute nodes, `I_server_setup.md` Sec. 4). Status: in progress. The native attention is
done and measured; the target (Llama-2-7B Kpre at 2,048 tokens in at most 20 s on 8 cores, from about 85 s) is not
reached yet.

## 1. The problem

On a CPU the verifier's time at 2,048 tokens is dominated by `verify_derive`, the recomputation of the cheap ops
(F1, `F1_reexecution_baseline.md`: Llama-2-7B Kpre about 85 s against 86 s for fp32 re-execution on the same 8 cores).
Inside the derive, attention costs the most: `transformer._attention_core` materialises the `T x T` integer
scores of a group of heads and runs about a dozen element-wise passes over them (mask, maximum, gaps, exp table,
sums, floor division, probabilities), each a full pass over memory.

## 2. Two approaches

**`torch.compile` of the CPU derive** (inductor's C++ backend, which needed I4). It fuses the element-wise passes:
1.3-1.6x on the derive at 2,048 tokens, nothing at 64, after a 20-120 s compile per graph and shape. Because of
the compile time and the per-shape recompiles, it is not the default.

**A native exact attention** (`native_kernels.py`). For each (batch, head, query row), on the verifier's threads:
the int32 scores against the keys up to the query's position, their maximum, `e = LUT[min(max - s, len - 1)]`,
`tot = sum e`, `p = floor((tot + 510 e) / (2 tot))` and `p v` accumulated in int64, with no `T x T` tensor. Query rows
are taken eight at a time, so each key and value row is read once per tile of rows. The library is compiled once per
host with `$CXX -O3 -march=native -fopenmp` into `$PVI_NATIVE_DIR` and loaded with ctypes (no Python or torch
headers); without a compiler, `available()` is false and the torch path runs. `transformer._attention_heads` uses it
for CPU tensors (`PVI_NATIVE_ATTN=0` disables it).

**Exactness.** Every step is integer arithmetic on the values `_attention_core` computes: scores below `2^21`,
`tot` and `510 e + tot` below `2^31` (the conditions `transformer._int32_scores` already checks before taking the
int32 path), floor division of positive integers. The result is bit-identical:
`tests/test_gpu_verifier.py::test_the_native_cpu_attention_gives_the_int32_cores_integers` compares it with
`_attention_core` on 15 cases (five shapes up to 1,024 keys, grouped-query heads, one and seven query rows as in
a pruned block, head sizes 64-128; random, peaked and extreme scores). With the other attention tests
(`-k "native or attention"`), 165 tests pass on the node.

## 3. Measurements (Xeon Silver 4114, 8 threads; Llama-2-7B, 2 blocks, the last one pruned, Kpre, 3 queries)

| | 64 tokens | 2,048 tokens |
|---|---:|---:|
| verify, torch attention (job 1005185) | 0.094 s | 4.12 s |
| verify, native attention, untiled (job 1005185) | 0.090 s | 2.73 s (1.51x) |
| verify, torch attention (job 1005299) | 0.093 s | 3.50 s |
| verify, native attention, tiled (job 1005299) | 0.087 s | 2.38 s (1.47x) |

`verify_derive` at 2,048 tokens: 3.01 to 1.83 s (untiled) and 2.66 to 1.56 s (tiled). The baselines of the two jobs
differ by 15% (the node is shared), so the tiling's own gain cannot be told apart from noise. Every query was
accepted.

## 4. What remains

* The rest of the derive at 2,048 tokens (normalisation, RoPE, residuals, requantisation) as native or compiled
  kernels; with V1 (C2) the requantisation passes of cut ops disappear from the derive.
* The products phase (`verify_products`, 0.45 s of 2.4 s here) and the decode (0.44 s).
* A full-model measurement (32 blocks) against fp32 re-execution on the same CPU (F1), and the same on the
  friend's EPYC node.

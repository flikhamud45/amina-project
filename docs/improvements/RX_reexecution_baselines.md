# RX: the strongest exact re-execution baselines on the same CPU

Plan 2, item RX (`SP2027_PLAN2.md`, Sec. 3.3). Code: commit 12a188f (`native_kernels.pvi_gemm_i8` and
`native_kernels.int8_matmul`, used by `MatOp.compute` on a CPU; `experiments/7_reexec/reexec.py` variants `int`,
`int_fp32gemm`, `bf16`, `dynint8`). Measured in job 1006979 (s-005, Xeon Silver 4114, 8 threads, git 12a188f).

## 1. Why

The paper's only performance claim in the abstract is that the CPU verifier is several times faster than re-executing
the model. The re-execution side used to be our integer forward with its weight products as fp32 GEMMs over 1,024-term
chunks, and plain fp32: a reviewer can call that a straw man. RX measures every reasonable re-execution on the same
8 threads as the verifier and defines the **strongest exact baseline** as the fastest of fp32, bf16 and exact integer
re-execution. A standard int8 engine is reported beside them, labelled not exact (it computes a different model).

## 2. What was built

* **An exact int8 GEMM on the CPU** (`pvi_gemm_i8`): int8 operands widened to int16 pairs and multiplied with
  `vpmaddwd` (AVX-512BW), 4 x 64 output tiles, int64 output. `MatOp.compute` uses it for linear layers whose inputs are int8 on a CPU (switch `PVI_NATIVE_GEMM`), so the
  integer forward's products are exact without the fp32 chunking. Tests in `tests/test_native_cheap.py` compare it with
  the exact torch product.
* **`reexec.py` variants**, each timing a full forward of the benchmark model (random weights of Llama-2-7B's shapes;
  2 and 32 blocks measured, the full model directly): `int` (the integer forward with the native int8 GEMM and B5's
  native non-weight kernels), `int_fp32gemm` (the same with fp32-chunk GEMMs), `float` (fp32), `bf16`, and `dynint8`
  (torch dynamic-int8 `Linear`: **not exact**).

## 3. Results (full Llama-2-7B, 8 Xeon threads, 3 repetitions)

| Variant | Exact? | 64 tokens | 2,048 tokens |
|---|---|---:|---:|
| `int` (native int8 GEMM) | yes (integer model) | 7.26 s | 179.4 s |
| `int_fp32gemm` | yes (integer model) | **4.08 s** | 141.6 s |
| `float` (fp32) | the fp model | 4.53 s | **93.2 s** |
| `bf16` | the fp model | 19.5 s | 306.5 s |
| `dynint8` | **no** | 1.91 s | 56.0 s |

The same 2,048-token cells with glibc's heap allocator tuned (`MALLOC_MMAP_MAX_=0`, trim threshold 2 GiB): `int` 189.6 s,
`float` 88.4 s, `dynint8` 48.7 s, so the allocator is not what limits them.

Readings:

* On this Xeon (Skylake: no VNNI, no AVX512_BF16) the native int8 GEMM is slower than MKL's fp32 GEMM (179 s against
  142 s), and bf16 is emulated (306 s). The EPYC of the final run has VNNI and AVX512_BF16, so the `cpuv` tier repeats
  every row there.
* **Strongest exact baseline on the Xeon:** 4.08 s at 64 tokens (`int_fp32gemm`) and 93.2 s at 2,048 tokens (fp32).

## 4. Ratios to the verifier

The native CPU verifier in Kpre (V-LONG-a, the same job and threads, full Llama-2-7B, 3 queries, all accepted):

| | verifier | strongest exact baseline | ratio | ratio to `dynint8` (not exact) |
|---|---:|---:|---:|---:|
| 64 tokens, interactive | 1.01 s | 4.08 s | 4.0x | 1.9x |
| 64 tokens, Fiat-Shamir (7 repetitions instead of 5) | 1.40 s | 4.08 s | 2.9x | 1.4x |
| 2,048 tokens, interactive | 18.5 s | 93.2 s | 5.0x | 3.0x |
| 2,048 tokens, Fiat-Shamir (7 repetitions) | 19.6 s | 93.2 s | 4.7x | 2.9x |

The 2,048-token cells are job 1007816 (the prover on the CPU, since the first run's GPU prover ran out of memory on
the 11 GB card): 6.43 GB of claims, 3/3 queries accepted in both settings; the verifier's phases are decode 2.7 s,
derive 11.2 s and products 4.6 s (interactive). B5 had extrapolated 17.2 s. Against the exact integer re-execution
with fp32 GEMMs (141.6 s) the ratios are 7.6x and 7.2x. Under Fiat-Shamir, Kpre absorbs nothing (FS1), but its Freivalds repetitions go from 5 to 7, which costs about 1.4x
in the products step.

## 5. What this means for the paper

* The abstract's "4.6 times" was against fp32 only, interactive and extrapolated. Following the plan's rule, it becomes
  "at least N times", with N the smaller of the measured ratios on the Xeon and on the EPYC, against the strongest exact
  baseline, non-interactive. On the Xeon: 2.9x at 64 tokens and 4.7x at 2,048 tokens; the EPYC cells come from the
  `cpuv` tier.
* The `dynint8` row is reported beside, labelled: even this inexact engine is 1.4-1.9x slower than our verifier at
  64 tokens, and it computes a different (non-certified) model.

## 6. How to run

    cd code && PYTHONPATH=src python experiments/7_reexec/reexec.py --model llama2-7b --seq 64 2048 --layers 2 32 \
        --variants int int_fp32gemm float bf16 dynint8 --device cpu --threads 8 --reps 3

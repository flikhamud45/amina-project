# V-LONG-b. Long prompts, 2k to 16k tokens: the CPU verifier, re-execution and the bytes

Plan 2 item V-LONG-b (`SP2027_PLAN2.md` Sec. 3.3, should, rank 5; reviewer points C-W7 and C-Q7: the paper does not
project proof size or the `O(T^2)` verifier beyond 2,048 tokens). Measurement only, with today's kernels (valid below
32,768 tokens): job 1011426 (s-004, RTX 2080 Ti prover, 8 threads of the Xeon Silver 4114 for the verifier and every
re-execution; git f85ad55), the cells of the package's optional `long` tier.

## 1. What was measured

* `perf.py --model llama2-7b --layers 2|3 --seq T --modes Kpre --fs --lean --wire on --prune-last on --lookups on
  --queries 3 --verifier-device cpu` for T = 2,048, 4,096, 8,192 and 16,384: the native CPU verifier (Fiat-Shamir),
  the exact proof bytes, 3 accepted queries each.
* `reexec.py --model llama2-7b --seq T --layers 2 3 --variants int int_fp32gemm float dynint8 --device cpu --reps 3
  --max-pos 16384`: RX's re-executions on the same threads, extrapolated to 32 blocks from the 2- and 3-block cells.
* Per block = the 3-block cell minus the 2-block cell; the full model = the 2-block cell plus 30 blocks.

## 2. Results

| T | verifier, 2 / 3 blocks (s) | per block (s) | full model, verifier (s) | fp32 re-execution, full (s) | fastest exact / verifier | proof, 2 blocks (MB) | per block (MB) | full-model proof (GB) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 2,048 | 0.82 / 1.38 | 0.56 | 17.6 | 90.4 | 5.1x | 244.5 | 204 | 6.4 |
| 4,096 | 1.71 / 3.16 | 1.45 | 45.2 | 188.5 | 4.2x | 494.0 | 413 | 12.9 |
| 8,192 | 4.63 / 8.61 | 3.98 | 124 | 398.5 | 3.2x | 992.4 | 827 | 25.8 |
| 16,384 | 16.02 / (GPU prover out of memory) | 12.6-15 (see below) | 390-470 | 930.4 | 2.0-2.4x | 2,013.2 | about 1,650 | about 51 |

Every query was accepted. The other re-executions (full model, s): exact integer with the native int8 GEMM 198.8 /
492.0 / 1,140.5 / 2,272.6; exact integer with fp32 GEMMs 148.9 / 326.5 / 658.8 / 1,356.4; dynamic int8 (not exact)
50.2 / 107.0 / 245.3 / 626.3. fp32 is the fastest exact re-execution at every length. Peak host memory of the
verifier: 5.7 GB (3 blocks, 2,048) to 23.5 GB (2 blocks, 16,384).

* **The fit.** A block's verifier time follows `t(T) = a T + b T^2` with `a = 2.03e-4 s` and `b = 3.46e-8 s` from the
  2,048- and 8,192-token cells; it predicts 1.41 s at 4,096 against 1.45 s measured (3%, within the plan's 10%).
  At 16,384 it predicts 12.6 s per block; the 2-block cell (16.0 s) implies about 15 s if the fixed part (the pruned
  last block, the head and decoding) stays near 1 s, so the attention term grows a little faster than `T^2` there
  (cache effects); the 16,384-token row gives both.
* **Bytes grow linearly** (204, 413, 827 MB per block): the claims are the weight operations' outputs. The full
  Llama-2-7B's proof is 25.8 GB at 8,192 tokens (the plan's model: 25 GB) and about 51 GB at 16,384.
* **The ratio to re-execution falls with the prompt**, as the `T^2` attention dominates both sides: 5.1x at 2,048
  tokens (V-LONG-a measured the full model at 4.7x under Fiat-Shamir, 19.6 s; the 2-to-32-block extrapolation here
  gives 17.6 s, 10% low), 4.2x at 4,096, 3.2x at 8,192 (the plan's model: 3.2x) and 2.0-2.4x at 16,384.
* **32,768 tokens** are not measured (today's kernels hold below 2^15 tokens, and the 16,384-token 3-block cell
  already exceeds the 11 GB card). Extending the fit (with its 16k uncertainty) gives about 45-60 s per block for the
  verifier (1,400-1,900 s for the model) and roughly 2,100 s of fp32 re-execution, a ratio near 1.1-1.5x, and a
  proof of about 100 GB: modelled, not
  measured, and labelled so wherever used.

## 3. What it means for the paper

* The claim "the CPU client verifies faster than it can re-execute" holds at every measured length up to 16,384
  tokens, with the margin shrinking from 5x to about 2x.
* At long prompts the proof is the cost: tens of GB per query at 8k-16k tokens; V1 roughly halves it (the C2 byte
  model), which is where it matters most.
* Re-timing the 16,384-token per-block cell needs a card with more memory (the L40S `long` tier, already in the
  package).

## 4. Limitations

* Random weights (the costs depend on the shapes); Llama-2-7B shapes with `max_pos` raised to 16,384 (no claim about
  the model's quality at that length).
* Extrapolated from 2-3 blocks, which V-LONG-a found about 10% low at 2,048 tokens (17.6 s against 19.6 s measured).
* The node ran other jobs of this account at the same time (not `--exclusive`), so single cells may be a few percent
  off; the trends are robust.

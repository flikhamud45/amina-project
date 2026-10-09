# F1. Re-execution baseline

Plan item F1 (reviewer objection D1: "the verifier is not cheaper than re-executing the model").
Commits `9a40bf2`, `6fe75fa` (`experiments/7_reexec/reexec.py`). Status: measured on the RTX 2080 Ti node;
to be re-measured on the L40S and the EPYC with the final verifier.

## 1. Why it matters

A client that holds the weights (settings K and Kpre) could check an answer by computing it again. A
proof of inference is only worth it where checking is cheaper than re-execution, or where re-execution is
impossible (setting C: the client holds only the commitment). Reviewers will ask for this ratio first.

## 2. Method

`reexec.py` times two ways of re-running each decoder (random weights; the cost depends on the shapes):

* `int`: our integer model's forward pass (`IntGraph.forward` of `build_decoder(..., prune_last=True)`).
  It is bit-exact with the claims, so it is a complete check in itself.
* `float`: a plain floating-point decoder of the same shapes (fp16 on a GPU, fp32 on a CPU) with
  `scaled_dot_product_attention` and the architecture's norms and MLP; the LM head at the last position;
  no rotary embedding (an element-wise cost). Not bit-exact with the provider.

Each variant runs on 1- and 2-block builds; with `t(L) = a + L b` the time is extrapolated to the full
model (for pruned graphs this gives exactly "one pruned block + (L - 1) full blocks").

## 3. Results

GPU (RTX 2080 Ti), full model, seconds (the `int` column is before A2, which halves it):

| Model | T | integer forward | fp16 decoder |
|---|---|---|---|
| OPT-1.3B | 2,048 | 0.94 | 0.18 |
| OPT-6.7B | 64 / 512 / 2,048 | 0.17 / 0.41 / 1.94 | 0.035 / 0.16 / 0.39 |
| Llama-2-7B | 64 / 512 / 2,048 | 0.16 / 0.35 / 2.08 | 0.037 / 0.18 / 0.57 |
| Llama-2-13B | 64 / 512 / 2,048 | 0.14 / 0.62 / 3.46 | 0.08 / 0.34 / 0.82 |
| Qwen3-4B | 64 / 512 / 2,048 | 0.17 / 0.45 / 1.92 | 0.03 / 0.09 / 0.37 |

CPU, the same Xeon Silver 4114 node, 8 threads, Llama-2-7B, full model:

| T | our Kpre verifier | fp32 re-execution | integer re-execution |
|---|---|---|---|
| 64 | about 1.5 s (0.018 + 31 x 0.049) | 3.5 s | 6.8 s |
| 2,048 | about 85 s (0.44 + 31 x 2.73) | 86 s | 196 s |

## 4. Reading

* On a CPU, the verifier beats re-execution by 2-4x at short prompts and ties fp32 re-execution at 2,048
  tokens, where recomputing attention dominates (`verify_derive` is 76% of the verifier). The CPU verifier
  is about 10x from its hardware floor (plan B5).
* On a GPU, fp16 re-execution is far faster than our GPU verifier today (L40S GPU verifier 2.6 s for
  OPT-1.3B at 2,048 tokens and 4.95 s for Llama-2-7B in Kpre, against 0.18 s and 0.57 s of fp16
  re-execution on a 2080 Ti). The GPU verifier work (plan B2-B4) targets this.
* Setting C is where re-execution is impossible without the weights; the paper must also state that the
  claims reveal the weights after about `ceil(k / T)` queries, so setting C targets open-weight models
  served by an untrusted provider (plan D7), and that at 2,048 tokens the proof is about the size of the
  int8 weights (plan C).

## 5. Usage

    python experiments/7_reexec/reexec.py --model llama2-7b opt-1.3b --seq 64 512 2048 --device cuda --reps 7
    python experiments/7_reexec/reexec.py --model llama2-7b --seq 64 2048 --device cpu --threads 8 --reps 3

Raw logs: `S/sp2027_research/server_out/w4b_1004139.out` (GPU), `encv_1004144.out` (CPU and the same-node
verifier).

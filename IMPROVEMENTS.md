# Improvements on branch `improve-results` (not merged)

Constant-factor improvements to the defence, so the paper can show clear wins against published
systems. Every change keeps the guarantee ε ≤ 2^-λ, and **the default behaviour is unchanged**:
with no new flag, verdicts, proof bytes, Fiat–Shamir transcripts and Merkle roots are byte-identical
to the paper code (729b997). The stored results therefore still reproduce. Details, scripts and raw
results are in `code/experiments/6_improvements/` (README.md, results/).

## What changed

| Phase | What | How to enable |
|---|---|---|
| 1. Verifier engineering | Batched products (the challenge's three limbs in one exact float64 product), Enc(u) evaluated only at the opened columns (baby-step giant-step), int32 attention with one folded exp table, int8 tensor-core products on a GPU, no per-op host syncs. Also an optional **streaming GPU verifier** that overlaps upload and checks. | on by default (identical outputs); streaming: `Verifier(stream=True)` / `bench.py --verifier-impl stream` |
| 2. Parameters and encoding | Exact per-matrix column count (the paper used the worst-case 1/rate bound), one shared Merkle tree per codeword length, per-model codeword lengths, compact wire encoding (claims at ~18 instead of 32 bits, field elements at 31 bits) | `--policy ...`, `--wire` |
| 3. Layouts | Wide matrices committed transposed (no u, short columns), q/k/v and gate/up fused, embeddings as Merkle row look-ups, last block pruned to the last position | `--policy ...c`, `--prune-last`, `--lookups` (K/Kpre) |
| 4. Codec speed | GPU encoder with 4 host syncs per query (was about one per weight op), faster host encoder, same bytes | automatic with `--wire` |
| `auto` | Picks the plan with the fewest non-claim bytes within a setup budget, max(2 × paper setup, 2^34 encoded entries). Deterministic, from the validated byte model, one rule for every model. | `--policy auto` |

Soundness: each new option has a written argument (README), an automated check that every model,
policy and λ reaches at least λ bits, interactive and Fiat–Shamir, and malicious-prover tests
(tampered claims, forged u, columns, paths, look-up rows and wire bytes) in modes C, K and Kpre.
Tests: 1560 passed on the RTX 2080 Ti node (CUDA included), 1335 locally.

## Same hardware, before and after (RTX 2080 Ti prover, Xeon Silver 4114 verifier, 8 threads)

Paper code (729b997) against `improve-results` (b3a08d8). All queries accepted. Policies: the
`auto` choice, with `--wire` and, for GPT-2, `--prune-last`. λ = 128 unless stated.

| Configuration | Verifier, before → after | Prover, before → after | Proof, before → after |
|---|---|---|---|
| LeNet-5, C | 28.5 → **6.4 ms** (wire: 7.4) | 15.6 → 6.6 ms (wire: 11.8) | 131 → **62 kB** (wire: **49 kB**) |
| VGG-16, C (cnn18c) | 217 → **34 ms** | 88 → 39 ms | 3.43 → 2.24 MB (wire: 1.71) |
| GPT-2, 64 tokens, C | 1284 → **268 ms** CPU / **183 ms** GPU verifier | 559 → **154 ms** | 62.1 → **14.8 MB** (FS: 16.6) |
| Qwen3-4B, 8 tokens, Kpre, λ=40, 1 thread | 1671 → **258 ms** (wire: 375) | 521 → 273 ms (wire: 332) | 36.1 → 35.2 MB (wire: **20.3 MB**) |
| Llama-2-7B, 64 tokens, C (Phase 1 only) | 8.6 → 2.3 s | 7.8 → 7.7 s | unchanged |
| OPT-125M, 2048 tokens, GPU verifier (Phase 1) | C 1.70 → 0.49 s, Kpre 1.05 → 0.21 s (streaming) | | unchanged |
| Llama-2-7B, 2048 tokens, GPU verifier (from 1- and 2-block builds) | C 17.5 → 3.7 s, Kpre 13.2 → 2.1 s (streaming) | | unchanged |

## What it means against published systems

L40S/EPYC numbers below are **projections**: the paper's stored L40S/EPYC measurements scaled by the
same-node before/after ratios above. Proof sizes are exact. They must be confirmed on the L40S
(see the run list).

| Against | Their prover / verifier / proof | Ours (projected L40S) | Better on |
|---|---|---|---|
| zkCNN, LeNet-5 | 441 ms / 5.8 ms / 71.3 kB | ~1.8 ms / **~1.6 ms** / **62 kB** (49 kB with wire) | **all three** (with Fiat–Shamir the proof needs wire: 62.5 kB) |
| DeepProve, GPT-2, 64 tokens | 34.2 s / 1.35 s / 21.7 MB | ~48 ms / ~98 ms / **14.8 MB** (FS 16.6) | **all three**, also with Fiat–Shamir (needs wire) |
| Maverick, Qwen3-4B, 8 tokens (its setting) | 354.5 ms / 87.1 ms / 36.08 MB | ~86 ms / **~59 ms** / **20.3 MB** (wire) | **all three** |
| zkCNN, VGG-16 | 88.3 s / 59.3 ms / 341 kB | ~9 ms / **~10 ms** / 2.24 MB | prover, verifier |
| ZKML, VGG-16 | 637 s / 9.6 ms / 12 kB | ~9 ms / ~10 ms / 2.24 MB | prover; verifier about tied |
| zkGPT, GPT-2 | 21.8 s / 0.35 s / 101 kB | ~48 ms / **~98 ms** / 14.8 MB | prover, verifier |
| zkLLM, Llama-2-7B, 2048 tokens (A100) | 620 s / 2.36 s / 183 kB | 17.6 s / **~1.45 s** (C) or ~0.66 s (Kpre), GPU verifier / 11.6 GB | prover (48–56× on the same GPU), verifier |

Caveats to state in any text:
- **Projections:** every L40S/EPYC time above is a projection until the L40S re-run below.
- **Wire encoding:** the DeepProve, Maverick and Fiat–Shamir zkCNN rows need it.
- **Interactive vs Fiat–Shamir:** Table 4 compares our interactive protocol; the Fiat–Shamir numbers are given where it matters.
- **Maverick's 87.1 ms** excludes its 37.4 ms nonlinear replay, while ours includes all our non-weight work. The fair gap is therefore larger in our favour.
- **Setup grows** with the higher-rate plans (GPT-2: 0.86 → 9.9 G encoded entries, about 6 minutes on the 2080 Ti node). `auto` caps it at max(2 × paper, 2^34).
- **Random weights:** the LLM costs use random int8 weights, as in the paper. Trained weights compress better.
- **Remaining inefficiency:** the verifier still widens decoded claims to int64 (Qwen: 88 ms vs 31 ms for int32). Not addressed.
- **Harness fix:** the in-process harness now copies every message at its boundary. A prover sharing memory with the verifier could otherwise rewrite a message after handing it over; this was possible in the paper code too, but not in a real deployment.

## L40S runs needed (the friend's account; mine cannot use `killable`)

From a clean clone of `improve-results` at b3a08d8 (or later), with the strong-GPU setup used for
`raw_l40s` (exclude the Xeon node t-806):

```bash
export PVI_PLATFORM=l40s_improved SBATCH_PARTITION=killable SBATCH_GRES=gpu:l40s:1
B=code/experiments/4_defence_benchmark/slurm/bench.sbatch; mkdir -p logs/$PVI_PLATFORM; L="-o logs/$PVI_PLATFORM/%x-%j.out"
# 0. tests on sm_89 (interactively on an L40S node, from code/): python -m pytest tests -q -p no:cacheprovider -o addopts=""
# 1. CNNs: the paper format with the new code, then auto, then auto + wire (modes C, interactive and FS)
for m in lenet5 vgg11 vgg16 resnet18_cifar mlp_mnist; do
  sbatch $L $B cnn --model $m
  sbatch $L $B cnn --model $m --policy auto
  sbatch $L $B cnn --model $m --policy auto --wire --tag _wire
done
# 2. GPT-2 (DeepProve, zkGPT): paper format, then auto + wire + pruning, CPU and GPU verifier
sbatch $L $B llm --model gpt2 --seq 64 512 --lams 128 --modes C:int,Kpre:int --lean
sbatch $L $B llm --model gpt2 --seq 64 512 --lams 128 --modes C:int,C:fs --lean --policy auto --wire --tag _wire --prune-last
sbatch $L $B llm --model gpt2 --seq 64 --lams 128 --modes C:int --lean --policy auto --wire --prune-last --verifier-device cuda --tag _wire_gpuv
# 3. Qwen3-4B in Maverick's setting (1 verifier thread), with and without wire
PVI_THREADS=1 sbatch $L -c 1 $B llm --model qwen3-4b --seq 8 --builds full --modes Kpre:int --lams 40 --lean --lookups --prune-last --tag _thr1
PVI_THREADS=1 sbatch $L -c 1 $B llm --model qwen3-4b --seq 8 --builds full --modes Kpre:int --lams 40 --lean --lookups --prune-last --wire --tag _thr1_wire
# 4. zkLLM's models at 2048 tokens: streaming GPU verifier (proof format unchanged)
for m in opt-125m opt-1.3b opt-6.7b llama2-7b llama2-13b; do
  sbatch $L $B llm --model $m --seq 2048 --builds full --modes C:int,Kpre:int --lams 128 --lean --verifier-device cuda --verifier-impl stream --tag _gpuv_stream
done
# 5. Llama-2-7B at 1 and 64 tokens (ZKTorch row) with auto + wire
sbatch $L $B llm --model llama2-7b --seq 1 64 --builds full --lams 128 --modes C:int --lean --policy auto --wire --tag _wire
python code/experiments/5_comparison/aggregate.py --platform l40s_improved
```

Afterwards the paper tables can be regenerated from `tables_l40s_improved/`. Updating
`paper_assets.py` and `text_numbers.py` for the new cells is a separate, later step, done only if we
decide to merge.

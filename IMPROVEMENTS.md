# Improvements after the report (merged into `submission`)

Constant-factor improvements to the defence, made after the report on branch `improve-results`
and merged into the submission; the report itself is not updated and does not use them. Every
change keeps the guarantee ε ≤ 2^-λ, and **the default behaviour is unchanged**: with no new flag,
verdicts, proof bytes, Fiat–Shamir transcripts and Merkle roots are byte-identical to the paper
code (729b997, and the submission's code before the merge). The stored results therefore still
reproduce: Route A of `code/README.md` gives the report's tables, figures and numbers byte for
byte. Details, scripts and raw results are in `code/experiments/6_improvements/` (README.md,
results/); the `bench.py` options are described in `code/experiments/4_defence_benchmark/README.md`.

## What changed

| Phase | What | How to enable |
|---|---|---|
| 1. Verifier engineering | Batched products (the challenge's three limbs in one exact float64 product), Enc(u) evaluated only at the opened columns (baby-step giant-step), int32 attention with one folded exp table, int8 tensor-core products on a GPU, no per-op host syncs. Also an optional **streaming GPU verifier** that overlaps upload and checks. | on by default (identical outputs); streaming: `Verifier(stream=True)` / `bench.py --verifier-impl stream` |
| 2. Parameters and encoding | Exact per-matrix column count (the paper used the worst-case 1/rate bound), one shared Merkle tree per codeword length, per-model codeword lengths, compact wire encoding (claims at ~18 instead of 32 bits, field elements at 31 bits) | `--policy ...`, `--wire` |
| 3. Layouts | Wide matrices committed transposed (no u, short columns), q/k/v and gate/up fused, embeddings as Merkle row look-ups, last block pruned to the last position | `--policy ...c`, `--prune-last`, `--lookups` (K/Kpre) |
| 4. Codec speed | GPU encoder with 4 host syncs per query (was about one per weight op), faster host encoder, same bytes | automatic with `--wire` |
| `auto` | Picks the plan with the fewest non-claim bytes within a setup budget, max(2 × paper setup, 2^34 encoded entries). Deterministic, from the validated byte model, one rule for every model. | `--policy auto` |

Soundness: each new option has a written argument (`code/experiments/6_improvements/README.md`),
an automated check that every model, policy and λ reaches at least λ bits, interactive and
Fiat–Shamir, and malicious-prover tests (tampered claims, forged u, columns, paths, look-up rows
and wire bytes) in modes C, K and Kpre. Tests: 1560 passed on the RTX 2080 Ti node and on the
L40S (CUDA included); on this branch `python -m pytest tests` collects 1,551 tests without CUDA
(1,560 with it, 3 more with `transformers`), of which 1,346 pass on a CPU and the 205 GPU-only
ones are skipped.

## Development measurements (RTX 2080 Ti prover, Xeon Silver 4114 verifier, 8 threads)

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

## Measured on the L40S (the paper's hardware)

The list below was run by Edo on 2026-10-01: 27 jobs, L40S prover and 8 threads of AMD EPYC 9334
(1 thread for the Maverick rows), branch at c8be5eb. The records are in
`code/artifacts/comparison/raw_l40s_improved/` (frozen, 68,363 records), the tables in
`tables_l40s_improved/`. From `code/`:

```bash
python experiments/5_comparison/aggregate.py --platform l40s_improved       # -> tables_l40s_improved/, unchanged
python experiments/5_comparison/count_outcomes.py --platform l40s_improved  # the counts below
python experiments/5_comparison/fingerprint_check.py artifacts/comparison/raw_l40s artifacts/comparison/raw_l40s_improved
```

Checks:
- **Integrity:** 195 cells, all from one clean commit, all L40S + EPYC; no partial or rejected cells.
- **Counts:** 4,380/4,380 honest queries accepted, 4,973/4,973 attacks rejected.
- **Unchanged default:** all 565 hardware-independent fingerprints match the paper's `raw_l40s`.
- **Tests:** 1560 passed on sm_89.
- **Cross-check:** a re-aggregation reproduces the tables byte for byte, and an independent
  recomputation of the table below reproduces Edo's numbers exactly.

Same hardware, before (paper code, `raw_l40s`) and after:

| Configuration | Prover / verifier / proof, before | After |
|---|---|---|
| LeNet-5, C | 4.3 ms / 7.0 ms / 130.9 kB | **3.0 ms / 1.9 ms / 62.1 kB** (wire: 5.3 / 2.4 ms / 49.5 kB) |
| LeNet-5, C, Fiat-Shamir | 4.5 / 8.5 ms / 172.6 kB | wire: 5.4 / 2.6 ms / 63.3 kB |
| VGG-16, C | 21.5 / 63.4 ms / 3.43 MB | 13.5 / 16.2 ms / 2.24 MB (wire: 17.3 / 14.9 ms / 1.71 MB) |
| GPT-2, 64 tokens, C (wire, pruned) | 173 / 468 ms / 62.1 MB | **80 / 92 ms / 14.85 MB**; GPU verifier 69 ms; Fiat-Shamir 91 / 106 ms / 16.57 MB |
| GPT-2, 512 tokens, C (wire, pruned) | 163 ms / 1.01 s / 213 MB | 134 / 375 ms / 89.5 MB |
| Qwen3-4B, 8 tokens, Kpre, lambda=40, 1 thread | 136 / 263 ms / 36.08 MB | 117 / 111 ms / 35.19 MB (wire: 119 / 156 ms / 20.30 MB) |
| Llama-2-7B, 1 token, C (wire) | 2.72 / 2.53 s / 418 MB | 1.48 s / 299 ms / 130 MB |
| Llama-2-7B, 2048 tokens, GPU verifier (C / Kpre) | 18.7 s, 6.88 s / 16.7 s, 4.16 s | **5.39 s, 1.68 s / 2.86 s, 1.31 s** (streaming verifier; the proof is unchanged) |

## Against published systems (measured; theirs as reported on their own hardware)

| Against | Theirs: prover / verifier / proof | Ours: prover / verifier / proof | Better on |
|---|---|---|---|
| zkCNN, LeNet-5 | 441 ms / 5.8 ms / 71.3 kB | 3.0 ms / 1.9 ms / 62.1 kB | **all three** (145x, 3.0x, 1.15x); FS + wire: 5.4 / 2.6 ms / 63.3 kB, all three |
| DeepProve, GPT-2, 64 tokens | 34.2 s / 1.35 s / 21.7 MB | 80 ms / 92 ms / 14.85 MB | **all three** (426x, 14.7x, 1.46x); FS: 1.31x smaller; GPU verifier 69 ms |
| DeepProve, GPT-2, 512 tokens | 176 s / 1.65 s / 25.5 MB | 134 ms / 375 ms / 89.5 MB | prover, verifier (proof 3.5x larger) |
| zkGPT, GPT-2 | 21.8 s / 0.35 s / 101 kB | 80 ms / 92 ms / 14.85 MB | prover, verifier (3.8x) |
| zkCNN, VGG-16 | 88.3 s / 59.3 ms / 341 kB | 13.5 ms / 16.2 ms / 2.24 MB | prover, verifier (3.7x) |
| Bionetta, LeNet-5 | 3.75 s / 10 ms / 0.9 kB | 3.0 ms / 1.9 ms / 62.1 kB | prover, verifier (5.1x) |
| ZKML, VGG-16 | 637 s / 9.6 ms / 12 kB | 17.3 ms / 14.9 ms / 1.71 MB | prover only (verifier 1.5x slower) |
| ZKTorch, Llama-2-7B, 1 token | 2645 s / 100 s / 22.9 MB | 1.48 s / 0.30 s / 130 MB | prover, verifier (335x); proof 5.7x larger |
| Maverick, Qwen3-4B, 8 tokens (lambda=40, 1 thread) | 354.5 ms / 87.1 ms / 36.08 MB | no wire: 117 / 111 ms / 35.19 MB; wire: 119 / 156 ms / 20.30 MB | prover (3.0x) and proof (1.78x with wire); **verifier 1.3-1.8x slower** than its 87.1 ms. Its 87.1 ms excludes a 37.4 ms nonlinear replay; against 124.5 ms our no-wire verifier is 1.12x faster, but then the proof is only 2.5% smaller. |
| zkLLM, 2048 tokens (A100), GPU verifier | OPT-125M 73.9 s / 0.34 s; OPT-1.3B 221 s / 0.90 s; OPT-6.7B 548 s / 2.08 s; Llama-2-7B 620 s / 2.36 s; Llama-2-13B 803 s / 3.95 s | C: 0.27/0.17, 1.56/0.73, 5.17/1.45, 5.39/1.68, 15.3/9.06 s; Kpre: 0.14/0.11, 0.96/0.65, 2.73/1.17, 2.86/1.31, 7.59/3.42 s | prover everywhere (52-523x); verifier everywhere (1.2-3.1x) **except Llama-2-13B in mode C (2.3x slower)**; proof 5,000-96,000x larger |

What did not hold as projected from the 2080 Ti ratios:
- **Maverick:** the EPYC's paper-code baseline was already faster, so the gain is 2.4x there instead of 4-6x.
- **ZKML:** VGG-16's verifier is 1.5x slower than ZKML's, not tied.
- **Llama-2-13B, mode C:** the streaming verifier takes 9.06 s against 1.68 s for Llama-2-7B, and the prover's opening grows to 6.1 s against 2.9 s in the paper run. Host memory rose to 63 GiB against 36 GiB for 7B; pinned-buffer pressure is a likely cause, not yet confirmed.
- **Better than projected:** the prover at 2048 tokens is 3-6x faster than in the paper run, from the int32 attention its forward pass shares with the verifier.

Caveats to state in any text:
- **Wire encoding:** the DeepProve, Fiat-Shamir zkCNN and smaller-proof Maverick rows need it.
- **Interactive vs Fiat-Shamir:** Table 4 compares our interactive protocol; the Fiat-Shamir numbers are given where it matters.
- **Setup grows** with the higher-rate plans (GPT-2: 0.86 -> 9.9 G encoded entries). `auto` caps it at max(2 x paper, 2^34).
- **Random weights:** the LLM costs use random int8 weights, as in the paper. Trained weights compress better.
- **Remaining inefficiency:** the verifier widens decoded claims to int64 (Qwen: 88 ms vs 31 ms for int32), which matters for the Maverick row. Not addressed.
- **Harness fix:** the in-process harness now copies every message at its boundary. A prover sharing memory with the verifier could otherwise rewrite a message after handing it over; this was possible in the paper code too, but not in a real deployment.
- **Open controls:** the new code at 2048 tokens with the non-streaming GPU verifier would separate the attention and streaming gains in the prover. Llama-2-13B in mode C with a smaller streaming window would test the host-memory hypothesis.

## The L40S runs (done 2026-10-01; recorded in `raw_l40s_improved/`)

Run from a clean clone of `improve-results` at c8be5eb, with the strong-GPU setup used for `raw_l40s`
(the Xeon node t-806 excluded). The pytest step ran as a batch job; Llama-2-13B got `--mem=128G`.
The same commands run on this branch (`bench.py` has the same options; the Reed–Solomon rate is
fixed at 4, the base rate of every plan). `raw_l40s_improved/` is frozen, so a re-run uses a new
`PVI_PLATFORM`.

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

The report's tables, figures and numbers stay those of `l40s`: `paper_assets.py` and
`text_numbers.py` are written for that platform and do not draw the new cells (`_polauto`, `_wire`,
`_prune`, `_lookups`, `_gpuv_stream`). The tables above come from `tables_l40s_improved/`.

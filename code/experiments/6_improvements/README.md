# 6. The optimised protocol: development harnesses and measurements

This folder holds the scripts that measure the optimised protocol's options (paper Sec. 3.6.2)
against the basic protocol, and their results (`results/`): same-process A/B timings of the
verifier, byte models validated against real queries, and benchmarks of the compact encoding. Most
results were measured on a laptop CPU (Intel i7-1165G7, 4 cores and 8 threads, Windows, torch 2.2.2,
Python 3.10) with random-initialised models, since costs depend on the shapes; the absolute times are
noisy and the ratios come from interleaved runs; `results/` holds no GPU timings (the GPU commands
below are for re-running the harnesses on a GPU node). **The paper's benchmark numbers do not come
from these files** but from the L40S runs of [`4_defence_benchmark`](../4_defence_benchmark/README.md)
(platforms `l40s_improved`, `l40s_improved2` and `l40s_improved3`, turned into tables by
[`5_comparison`](../5_comparison/README.md)); the one number of the paper backed by this folder is the
16–19 bits per claim of the compact encoding (Sec. 3.6.2), from `wire.py`'s `results/wire_laptop.json`.
Nothing here writes to the stored benchmark roots.

Most scripts here are A/B tools: `ab_verifier.py`, `gpu_ab.py`, `attention_ab.py` and `codec_stages.py`
require `--base`, a checkout of an older version of the code from the project repository, so they
cannot run from the submission alone; `codec_ab.py` and `wire_hashes.py` compare against such a
checkout when given one. `plan_bytes.py`, `perf.py`, `wire.py` and `claims_drop.py` run on this code
alone.

## The options

| Option | What it does | How to enable | Code |
|---|---|---|---|
| Verifier engineering | the challenge's three limbs in one exact float64 product, batched over operations; the encoded folded row `Enc(u)` evaluated only at the opened columns (baby-step giant-step); int32 attention with one folded exp table (prover and verifier); exact int8 tensor-core products on a GPU; one copy back to the host per check | always on; changes no proof byte, only timings | `protocol.py`, `field.py`, `transformer.py` |
| Streaming GPU verifier | claims travel as int32 in pinned memory and are uploaded one weight operation at a time on a side stream; both sides of each Freivalds check are computed as the claim arrives; mode C's Merkle checks run on a host thread; one copy back per query decides it | `Verifier(stream=True)`; `bench.py --verifier-device cuda --verifier-impl stream --tag _gpuv_stream` | `protocol.py`, `pipeline.py` |
| Commitment plans | exact column count per tree, one Merkle tree per codeword length shared by its matrices, per-matrix codeword lengths; with the suffix `c` also the transposed layout for wide matrices, fused q/k/v and gate/up, and embedding tables as Merkle lookup tables; `auto` picks a `c` plan by a public rule | `commit_graph(..., policy=)`; `bench.py --policy <name>` | `plans.py`, `commitment.py` |
| Compact encoding | claims in the format `PVC3` (a frame of reference per weight operation, optionally centred on row means, Rice-coded exceptions), `u` and opened columns in 31 bits | `run_query(..., wire=True)`; `bench.py --wire --tag _wire` | `claimcodec.py` |
| Last-block pruning | the last block computes q, the attention output, its projection, the residuals and the MLP at the last position only; keys and values stay at every position | `build_decoder(..., prune_last=True)`; `bench.py --prune-last` | `transformer.py` |
| Embedding lookups (K, Kpre) | the verifier computes the embedding operations from its own weights; the prover sends no claims for them | `Verifier(lookups=True)`; `bench.py --lookups` | `protocol.py` |

The policy `paper`, the default, is the basic protocol's commitment and parameters. Without the
options above the code runs the basic protocol: the same verdicts, proof bytes, Fiat–Shamir
transcripts and Merkle roots as the basic run (`raw_l40s/`; checked by
`5_comparison/fingerprint_check.py` on the 565 values it shares with `raw_l40s_improved/`). The
paper's optimised protocol combines the options as defined by `OPTIMISED` in
`5_comparison/paper_assets.py`: in mode C the planning rule (`--policy auto`) and the compact
encoding, with pruning for the language models; in K and Kpre the compact encoding, with lookups and
pruning for the language models.

## Soundness

Every option keeps the bound ε ≤ 2^-λ of the paper's Theorem 3.2 (one term per encoded matrix).

* **Exact column counts.** Each tree opens the smallest `t` whose exact column error
  `prod_{i<t} (k_l-1-i)/(n_l-i)` meets the per-operation budget `2^-β` the basic protocol already
  assigns (`β = λ + log2(2L)`, plus 64 grinding bits under Fiat–Shamir), instead of
  `ceil(β / log2 rate)`, which assumes `(k-1)/n = 1/rate` and ignores the power-of-two padding. So
  `ε ≤ sum_l [p^-r + 2^-β] = 2L 2^-β = 2^-λ` (interactive; `2^-(λ+64)` per transcript under
  Fiat–Shamir).
* **Shared trees.** Matrices of one codeword length share one Merkle tree (leaf `c` = SHA-256 of the
  group tag, `c` and every member's own column digest of column `c`, in the public member order), one
  set of `t_g = max t_l` column indices and one multiproof. Every member gets `t_g ≥ t_l` distinct
  uniform columns of its own codeword, drawn after `u`; the union bound never needs the operations'
  index sets to be independent. Fiat–Shamir absorbs every operation's codeword length and every
  group's tag, root, length, `t` and member order.
* **Transposed layout.** Per committed matrix the term is `p^-r + prod_{i<t} (m-1-i)/(n-i)`, with
  `m = k` (rows) or the stacked rows `sum N` (transposed), and no Freivalds term when `χ' = I`: a wrong
  claim survives `χ'` with probability at most `p^-r`, and then `Enc(z_i)` and `w_i^T E' = Enc(A w_i)`
  are distinct codewords of dimension `m`, which the `t` distinct columns (bound by the root, drawn
  independently of `χ'`) all miss with at most the product. Under Fiat–Shamir a transposed matrix's
  columns also depend on `u`, which the prover picks: grinding, which the 64 extra bits bound as for
  every other challenge.
* **Lookup tables.** A bias-free embedding's claim at id `j` is row `j` of `W`; accepting other rows
  needs other bytes for a leaf the root binds, a SHA-256 collision, the assumption every Merkle tree
  here already makes. A table adds no term (`L` stays the number of weight operations in `β`, a
  conservative count). An embedding with a bias keeps its row layout, since a tree over `W`'s rows
  does not bind the bias; a verifier key that makes such an operation a table is refused.
* **Pruning.** The pruned graph has the unpruned graph's weights and multipliers and gives the same
  logits (`torch.equal`, tested); both parties run it, so the protocol is unchanged.
* **Compact encoding.** The values are the same; the verifier decodes before it checks anything, and
  Fiat–Shamir absorbs the encoded bytes. The decoder knows every claim's shape before it allocates it
  (from the graph and the query's shape), so a malformed header costs it no more memory than the
  honest claims. A value the encoding cannot carry (a claim outside the range check, a looked-up row
  outside int8, a 32-bit field element) makes the encoder raise `claimcodec.Unencodable`; the message
  then travels as no bytes, which the verifier rejects as malformed where the unencoded value would
  be rejected.
* **Streaming verifier.** It gives the same verdicts and labels as `run_query`. It receives every
  message before it checks any, so a rejected query has also received (and counts) `u` and the
  openings.
* **In-process harness.** `run_query` runs both parties in one process and passes every message as a
  copy (`protocol._passed`): the query, `χ` and the column indices to the prover; the claims, `u` and
  the multiproofs to the verifier. A prover therefore cannot change a message after handing it over.
  The copies are not timed.

`soundness_bits` computes the bound of a plan (one check per committed matrix). At λ = 128 the
basic protocol reaches 131–153 bits and every plan 129.2–140.7 bits.

## Tests

| Test file | What it checks |
|---|---|
| `tests/test_plans.py` | `soundness_bits ≥ λ` and each matrix's budget for the five CNNs and every decoder configuration, every policy with and without `c`, λ = 40, 80, 128, modes C, K and Kpre, interactive and Fiat–Shamir; honest queries accepted and sized as the byte model says; every forgery rejected at its check (claims, `u`, columns, paths, misplaced group members, out-of-field entries) by `run_query`, the streaming verifier and the GPU forms, with and without the compact encoding; an honest prover that overwrites in place everything it handed over is accepted, and in-place forgeries are rejected |
| `tests/test_plan_auto.py` | the planning rule `auto`: the cheapest candidate within the budget for every model, at least λ bits, a few-block build takes the whole model's choice, honest queries accepted and a tampered claim rejected |
| `tests/test_layouts.py` | the transposed commitment, mixed groups, the planner, `χ' = I`, `w` and `z` against `reference.column_operands` |
| `tests/test_lookups.py` | the tables' trees, multiproofs and padding; every forgery of a looked-up row (`lookup_merkle`, `lookup_consistency`, `range_or_shape`); biased embeddings; keys with the tables in either order; K and Kpre with `lookups`; the byte model |
| `tests/test_pruning.py` | pruned logits `torch.equal` to the unpruned ones on GPT-2, OPT-350M, Llama-2-7B and Qwen3-4B blocks at 1, 2 and 9 tokens; attention of the last `Tq` queries; RoPE at an offset; a pruned block's plan |
| `tests/test_wire.py`, `tests/test_claimcodec.py` | the compact encoding: both encoders give the bytes of the reference encoder (`tests/claimcodec_reference.py`) on 16 claim sets; malformed input rejected; a lean GPU prover sends the CPU prover's bytes and verdicts; on CUDA, the GPU encoder's bytes and its four waits for the device |
| `tests/test_fast_verifier.py`, `tests/test_gpu_verifier.py` | the verifier's fast routines give the integers of `pvi.fullcheck.reference`; the GPU forms agree with the CPU ones (GPU-only tests are skipped without CUDA) |
| `tests/test_comparison_tables.py` | every timing `run_query` records is carried into the tables once (the streaming verifier's too); the scripts of `5_comparison`: cells matched by their tag set, later optimised runs replacing earlier cells, pending values, the count macros, `text_numbers.py` and the figure layout checks |

## Scripts and result files

Run from `code/` with `export PYTHONPATH=$PWD/src`. The A/B scripts compare this code with a checkout
of the basic protocol's code without the verifier engineering (commit `729b997` of the project
repository), imported under another module name in the same process (`--base <checkout>/code/src`);
the codec scripts compare the encoder with the reference per-operation encoder (commit `6c2059a`,
whose encoder `tests/claimcodec_reference.py` reproduces).

| Script | Measures | Results |
|---|---|---|
| `ab_verifier.py` | the verifier against the baseline on the same transcripts, with identical derived tensors, verdicts and rejection points of tampered transcripts required | `ab_laptop.csv`, `ab_laptop_merkle_processes.csv`; `ab_laptop_vs_1d3df18.csv`, `ab_laptop_a5adbf3.csv`, `ab_laptop_a5adbf3_vs_5c9b83c.csv` (the same A/B between other versions of the code; the name gives the commits) |
| `gpu_ab.py`, `attention_ab.py` | the baseline, the current and the streaming verifier back to back, on a GPU or a CPU; one group of attention heads | `gpu_ab_laptop_cpu_*.json`, `attention_ab_laptop_*.json` |
| `plan_bytes.py` | the byte model of every policy (`analytic.proof_bytes`, `setup_size`) and, with `--run`, real queries through `run_query` | `plan_bytes*.csv`, `combined_*.csv`, `col_*.csv` (decoder files: the transposed layout without lookup tables), `lookup_*.csv` |
| `perf.py` | interleaved timing of several policies and options in one process | `perf_*_laptop_*.jsonl` |
| `wire.py` | claim and proof bytes of the compact encoding, and its decoding cost | `wire_laptop.json` |
| `codec_stages.py`, `codec_ab.py`, `wire_hashes.py`, `claims_drop.py` | the reference encoder per stage; reference against current encoder and decoder; identical encoded messages; the cost of releasing encoded claims | `codec_profile_laptop.json`, `codec_laptop.json`, `codec_bytes_identity.jsonl`, `perf_codec_laptop.jsonl`, `claims_drop_laptop.txt` |

## 1. Verifier engineering

`ab_verifier.py` checks the same transcript with both verifiers, alternating which runs first. Every
repetition must give identical derived tensors and verdicts, the current prover must send the
baseline's claims, `u`, opened columns and Merkle paths, and every tampered transcript (claims off by
one, out of range or of the wrong dtype; `u` off by one or outside the field; opened columns off by
one, outside the field or narrowed; Merkle paths forged, short or long) must be rejected by both at the
same check.

```bash
git worktree add ../base 729b997                    # the baseline, next to this checkout (from its root)
cd code && export PYTHONPATH=$PWD/src
python experiments/6_improvements/ab_verifier.py --base ../../base/code/src --out ab.csv \
    --threads 1,4 lenet5:C:128 vgg16:C:128 gpt2:C:128:64
python experiments/6_improvements/ab_verifier.py --base ../../base/code/src --out ab.csv \
    lenet5:Kpre:128 vgg16:Kpre:128 gpt2:Kpre:128:64 qwen3-4b:Kpre:40:8:1 qwen3-4b:Kpre:40:8:24 \
    qwen3-4b:Kpre:40:8:2:1024 qwen3-4b:Kpre:40:8:8:1024
PVI_MERKLE_PROCESSES=1 python experiments/6_improvements/ab_verifier.py --base ../../base/code/src \
    --out ab_merkle_processes.csv --threads 4 vgg16:C:128 gpt2:C:128:64
```

A case is `model:mode:lambda[:tokens[:blocks[:vocab]]]`; a reduced vocabulary keeps the block shapes,
so it measures the per-block slope cheaply. Transcripts are cached in
`artifacts/fullcheck/cache/ab_verifier/` (`--cache`). In mode Kpre the verifier keeps the stacked
operands of its fixed `χ` and `u` across queries; the timings are the steady state after two warm-up
queries.

**On the laptop** (`results/ab_laptop.csv`, medians over 11–21 interleaved repetitions; `total` is
derive + products + columns, the verifier's work per query):

| Model | Mode | λ | Threads | Baseline (ms) | Current (ms) | Ratio |
|---|---|---|---|---:|---:|---:|
| LeNet-5 | C | 128 | 1 | 8.36 | 4.75 | 1.76x |
| LeNet-5 | C | 128 | 4 | 10.54 | 6.11 | 1.72x |
| LeNet-5 | Kpre | 128 | 1 | 2.92 | 1.91 | 1.53x |
| VGG-16 | C | 128 | 1 | 85.1 | 32.1 | 2.65x |
| VGG-16 | C | 128 | 4 | 77.4 | 34.0 | 2.28x |
| VGG-16 | Kpre | 128 | 1 | 26.2 | 9.51 | 2.75x |
| GPT-2, 12 blocks, 64 tokens | C | 128 | 1 | 599.2 | 245.8 | 2.44x |
| GPT-2, 12 blocks, 64 tokens | C | 128 | 4 | 487.9 | 215.1 | 2.27x |
| GPT-2, 12 blocks, 64 tokens | Kpre | 128 | 1 | 232.7 | 92.6 | 2.51x |
| Qwen3-4B, 1 block, 8 tokens | Kpre | 40 | 1 | 43.6 | 7.42 | 5.88x |
| Qwen3-4B, 8 blocks, 8 tokens | Kpre | 40 | 1 | 151.8 | 41.8 | 3.64x |
| Qwen3-4B, 24 blocks, 8 tokens | Kpre | 40 | 1 | 420.7 | 127.5 | 3.30x |

Qwen3-4B with all 36 blocks (4.4 GB of int8 weights) does not fit the laptop's free memory. Per
additional block (1 to 24 blocks, full vocabulary): 16.40 → 5.22 ms (3.14x), of which derive 6.96 →
4.12 ms and the products 9.45 → 1.08 ms; extrapolated to 36 blocks, 618 → 190 ms (3.25x). Per stage:
derive 1.0–1.1x on LeNet-5 and 1.55–1.77x elsewhere; products 1.7–1.8x (LeNet-5), 3.0–3.3x (VGG-16),
3.6x (GPT-2 C), 4.7x (GPT-2 Kpre) and 9–13x (Qwen3-4B); columns (mode C) 2.0–2.6x, whose floor is
SHA-256 of the Merkle paths. With `PVI_MERKLE_PROCESSES=1` (off by default) a multi-threaded verifier
checks the multiproofs in worker processes (VGG-16 C 2.73x, GPT-2 C 2.50x at 4 threads).
`ab_laptop_a5adbf3.csv` repeats the A/B against the baseline with a later version of the verifier
(totals 1.43–5.43x); `ab_laptop_vs_1d3df18.csv` and `ab_laptop_a5adbf3_vs_5c9b83c.csv` compare two
versions of the current verifier, whose totals agree within noise (0.98–1.15x).

**The GPU verifier.** On a GPU client, the range checks, the field check of `u`, the Freivalds
comparisons and the column code checks leave their verdicts on the device; the constants of the
operations without weights live on the device (`IntGraph.with_constants_on`); column indices and
opened columns go up through pinned memory; the code checks of all operations are queued before the
host hashes the Merkle paths (columns of 16 KiB or more are hashed on threads, on the CPU too).
`int8_*` products (`torch._int_mm`) are used wherever a per-GPU self-check (`int8_ok`) allows, the
float64 limb products otherwise. `test_a_gpu_verifier_makes_one_round_trip_per_check` checks the round
trips in CUDA's sync-debug mode. The int32 attention (`transformer._attention_core`) uses int32 scores
and one exp look-up per score gap, with the int64 code as the fallback outside the bounds that make it
exact.

`gpu_ab.py` runs the baseline (driven as `run_query` drives it), the current and the streaming
verifier on the same transcript. Before timing it checks that both provers send the same claims, that
baseline and current derive identical tensors, and that all three reject each tampered transcript at
the same check; with block counts 1 and 2 it also extrapolates to the full model. On an RTX 2080 Ti
(11 GB):

```bash
cd code && export PYTHONPATH=$PWD/src OUT=<scratch dir>
python -m pytest tests/test_gpu_verifier.py tests/test_fast_verifier.py -p no:cacheprovider -o addopts=""
PVI_TEST_DEVICE=cuda python -m pytest tests/test_gpu_exactness.py -p no:cacheprovider -o addopts=""
python experiments/6_improvements/gpu_ab.py --base ../../base/code/src --model opt-125m --seq 2048 \
    --layers full --modes Kpre,C --compile --out $OUT/gpu_ab_opt125m_T2048.json
python experiments/6_improvements/gpu_ab.py --base ../../base/code/src --model gpt2 --seq 64 \
    --layers full --modes Kpre,C --out $OUT/gpu_ab_gpt2_T64.json
python experiments/6_improvements/gpu_ab.py --base ../../base/code/src --model gpt2 --seq 512 \
    --layers full --modes Kpre,C --out $OUT/gpu_ab_gpt2_T512.json
python experiments/6_improvements/gpu_ab.py --base ../../base/code/src --model llama2-7b --seq 2048 \
    --layers 1,2 --modes Kpre,C --compile --compile-all --attn-budgets 134217728,16777216 \
    --out $OUT/gpu_ab_llama2-7b_T2048.json
python experiments/6_improvements/gpu_ab.py --base ../../base/code/src --model opt-1.3b --seq 2048 \
    --layers full --modes Kpre,C --impls stream --reps 3 --out $OUT/gpu_ab_opt1.3b_T2048_stream.json
python experiments/6_improvements/attention_ab.py --base ../../base/code/src --device cuda \
    --out $OUT/attention_ab_gpu.json
```

`--compile` adds a `torch.compile`d attention core and `--compile-all` compiles every operation
without weights (both checked bit-exact). The last `gpu_ab.py` command times only the streaming
verifier: the default verifier would hold OPT-1.3B's claims for 2,048 tokens on the device (15.3 GB).

On the laptop's CPU (`--device cpu --verifier-device cpu`, 4 threads, 3–5 repetitions;
`results/gpu_ab_laptop_cpu_*.json`) the int8 products are off and nothing waits on a device, so these
numbers show the int32 attention and the CPU products:

| Case | Mode | Stage | Baseline (ms) | Current (ms) | Ratio |
|---|---|---|---:|---:|---:|
| Llama-2-7B, 1 block, 2,048 tokens (vocabulary 1,024) | Kpre | derive | 3,462 | 2,046 | 1.69x |
| | | products | 753 | 244 | 3.08x |
| | | total | 4,267 | 2,283 (stream 2,107) | 1.87x (stream 2.03x) |
| GPT-2, 12 blocks, 512 tokens | Kpre | derive | 842 | 502 | 1.68x |
| | | products | 347 | 113 | 3.08x |
| | | total | 1,195 | 618 (stream 609) | 1.93x (stream 1.96x) |
| GPT-2, 12 blocks, 64 tokens | Kpre | total | 175.6 | 66.5 (stream 82.2) | 2.64x (stream 2.14x) |
| GPT-2, 12 blocks, 64 tokens | C | total | 517.3 | 234.1 (stream 226.6) | 2.21x (stream 2.28x) |

One group of 4 heads at 2,048 tokens with a head size of 128 (`results/attention_ab_laptop_*.json`, 7
repetitions): 507 → 234 ms at 1 thread (2.17x) and 213 → 101 ms at 4 threads (2.11x).

## 2. Commitment plans (mode C)

A plan (`pvi.fullcheck.plans`) changes three things of mode C: the exact column count per tree, one
shared tree per codeword length (the members of a length split into the runs of `t` that minimise the
expected column and multiproof bytes at λ = 128), and per-matrix codeword lengths: `tight` keeps the
basic protocol's `4 next_pow2(k)`; `cnn<e>` gives every matrix `max(2^e, 2 next_pow2(k))`; `R<R>` gives
rate `R` to every matrix but the embedding tables. The suffix `c` adds the layouts of section 4 and the
lookup tables of section 5.

```bash
cd code && export PYTHONPATH=$PWD/src
python experiments/6_improvements/plan_bytes.py --out results/plan_bytes.csv          # the byte model, every model
python experiments/6_improvements/plan_bytes.py --run mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar \
    --queries 5 --out results/plan_bytes_run_cnn.csv                                  # run_query, exact bytes
python experiments/6_improvements/plan_bytes.py --run gpt2:64,512:12 --policies paper tight R8 R16 \
    --queries 3 --out results/plan_bytes_run_gpt2.csv
python experiments/6_improvements/plan_bytes.py --run qwen3-4b:8:1:1024 qwen3-4b:8:2:1024 opt-125m:2048:1 \
    opt-125m:2048:2 llama2-7b:64:1:1024 llama2-7b:64:2:1024 --policies tight R8 --queries 2 \
    --out results/plan_bytes_run_decoders.csv                                         # the model on few-block builds
python experiments/6_improvements/perf.py --model lenet5 --random-init --modes C --queries 31 --threads 1 \
    --policy paper tight cnn16 cnn17 cnn18 R8 R16 R64 > experiments/6_improvements/results/perf_plans_laptop_cnn.jsonl
```

(`plan_bytes.py` writes `--out` relative to this folder.) **Proof bytes, λ = 128.** Every row of
`results/plan_bytes_run_cnn.csv` (5 distinct random honest queries per model and policy) has claim,
`u` and column bytes equal to the byte model and its median total within 0.7% of it (the Merkle paths
depend on the random indices). Interactive / Fiat–Shamir:

| Model | paper | tight | cnn16 | cnn17 | cnn18 | R8 | R16 | R64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP (MNIST) | 270.0 / 391.6 kB | 214.6 / 310.1 | 106.8 / 153.2 | 97.2 / 140.6 | 90.8 / 130.9 | 170.1 / 244.3 | 141.6 / 202.4 | 112.8 / 161.9 |
| LeNet-5 | 130.6 / 172.8 kB | 111.0 / 142.6 | 65.0 / 82.4 | 63.2 / 79.6 | 62.2 / 76.4 | 98.1 / 126.2 | 90.4 / 115.7 | 82.7 / 106.4 |
| VGG-11 | 2,201 / 2,907 kB | 1,791 / 2,309 | 1,500 / 1,888 | 1,420 / 1,768 | 1,365 / 1,693 | 1,613 / 2,051 | 1,507 / 1,888 | 1,383 / 1,706 |
| VGG-16 | 3,433 / 4,461 kB | 2,833 / 3,593 | 2,454 / 3,040 | 2,339 / 2,871 | 2,263 / 2,753 | 2,584 / 3,229 | 2,436 / 3,000 | 2,266 / 2,748 |
| ResNet-18 (CIFAR) | 4,630 / 5,583 kB | 4,036 / 4,716 | 3,642 / 4,155 | 3,550 / 4,016 | 3,491 / 3,926 | 3,807 / 4,391 | 3,676 / 4,189 | 3,521 / 3,964 |

The decoders, from the byte model (`results/plan_bytes.csv`; its `paper` column reproduces the stored
L40S measurements, e.g. GPT-2 at 64 tokens 62,072 kB against 62,071.5 kB stored), interactive /
Fiat–Shamir, MB:

| Model, prompt | paper | tight | cnn16 | cnn17 | cnn18 | R8 | R16 | R64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2, 64 | 62.1 / 80.7 | 54.6 / 69.4 | 37.0 / 43.3 | 35.3 / 41.0 | 34.0 / 39.6 | 46.3 / 57.2 | 41.5 / 50.2 | 36.7 / 42.8 |
| GPT-2, 512 | 213.5 / 232.1 | 206.0 / 220.8 | 188.4 / 194.7 | 186.7 / 192.4 | 185.4 / 191.0 | 197.7 / 208.6 | 192.9 / 201.6 | 188.1 / 194.2 |
| Llama-2-7B, 1 | 418.0 / 607.4 | 406.5 / 585.4 | 236.1 / 339.8 | 191.9 / 276.8 | 161.6 / 234.2 | 282.7 / 407.9 | 222.5 / 320.4 | 157.2 / 227.9 |
| Llama-2-7B, 64 | 761.7 / 951.1 | 750.3 / 929.1 | 579.9 / 683.6 | 535.6 / 620.5 | 505.3 / 577.9 | 626.4 / 751.7 | 566.2 / 664.2 | 501.0 / 571.6 |
| Llama-2-7B, 2048 | 11,586 / 11,776 | 11,575 / 11,754 | 11,404 / 11,508 | 11,360 / 11,445 | 11,330 / 11,402 | 11,451 / 11,576 | 11,391 / 11,489 | 11,325 / 11,396 |
| Qwen3-4B, 8 | 410.3 / 582.0 | 321.8 / 453.9 | 219.4 / 298.8 | 190.5 / 258.1 | 168.5 / 230.0 | 252.7 / 348.9 | 211.3 / 287.4 | 165.3 / 224.8 |
| OPT-125M, 2048 | 732.5 / 751.2 | 725.1 / 739.9 | 707.5 / 713.8 | 705.8 / 711.4 | 704.5 / 709.6 | 716.7 / 727.7 | 711.9 / 720.7 | 707.1 / 713.3 |
| OPT-1.3B, 2048 | 3,807 / 3,876 | 3,764 / 3,812 | 3,727 / 3,757 | 3,716 / 3,743 | 3,709 / 3,732 | 3,738 / 3,775 | 3,725 / 3,754 | 3,709 / 3,731 |
| OPT-6.7B, 2048 | 10,101 / 10,271 | 9,989 / 10,105 | 9,948 / 10,046 | 9,905 / 9,984 | 9,877 / 9,944 | 9,932 / 10,023 | 9,897 / 9,972 | 9,857 / 9,912 |
| OPT-13B, 2048 | 15,752 / 16,012 | 15,618 / 15,809 | 15,548 / 15,711 | 15,469 / 15,593 | 15,423 / 15,527 | 15,512 / 15,657 | 15,452 / 15,566 | 15,383 / 15,476 |

GPT-2 queried through `run_query` (`results/plan_bytes_run_gpt2.csv`) comes within 3.2 kB of the model.
A decoder too large to run is checked on 1- and 2-block builds (`results/plan_bytes_run_decoders.csv`):
every build's claim, `u` and column bytes equal the model's, and the 1- and 2-block totals
extrapolated over the blocks come within 0.011% of the model of the whole model. At 2,048 tokens the
claims are 94–99% of the proof, so the plans save only 2–4% there.

**Setup** (encoded field entries `sum_l N_l n_l` and trees, `analytic.setup_size`; the stored L40S
commitments run at about 9 ns per entry, e.g. Llama-2-7B's 29.8 G entries in 257–281 s):

| Model | paper | tight | cnn16 | cnn17 | cnn18 | R8 | R16 | R64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LeNet-5 | 0.3 M, 5 trees | 0.3 M, 4 | 15.5 M, 1 | 30.9 M, 1 | 61.9 M, 1 | 0.6 M, 4 | 1.2 M, 4 | 5.0 M, 4 |
| VGG-16 | 0.11 G, 16 | 0.11 G, 6 | 0.34 G, 4 | 0.69 G, 3 | 1.38 G, 3 | 0.22 G, 6 | 0.44 G, 5 | 1.74 G, 5 |
| ResNet-18 | 0.08 G, 21 | 0.08 G, 8 | 0.32 G, 4 | 0.63 G, 4 | 1.26 G, 4 | 0.16 G, 8 | 0.32 G, 8 | 1.27 G, 8 |
| GPT-2 | 0.86 G, 75 | 0.86 G, 5 | 8.9 G, 4 | 17.7 G, 4 | 35.3 G, 3 | 1.5 G, 4 | 2.8 G, 4 | 10.7 G, 4 |
| Llama-2-7B | 29.8 G, 226 | 29.8 G, 3 | 91 G, 3 | 183 G, 3 | 366 G, 3 | 59 G, 3 | 118 G, 3 | 468 G, 3 |
| Qwen3-4B | 27.8 G, 254 | 27.8 G, 4 | 84 G, 4 | 166 G, 4 | 331 G, 4 | 53 G, 4 | 103 G, 4 | 405 G, 4 |

The verifier's key shrinks from one root per matrix to one per tree (Llama-2-7B: 226 → 3).

**Timing on the laptop** (`results/perf_plans_laptop_*.jsonl`; every policy committed in one process
and its queries interleaved, 1 thread, medians over 31 (LeNet-5), 15 (VGG-16) and 11 (GPT-2) honest
queries, all accepted; ratios to `paper` in the same run):

| Model | Policy | Verifier (ms) | Column check (ms) | Prover open (ms) | Prover total (ms) | Proof |
|---|---|---:|---:|---:|---:|---:|
| LeNet-5 | paper | 7.86 | 5.49 | 3.37 | 6.14 | 130.9 kB |
| | tight | 7.30 (0.93x) | 4.86 (0.88x) | 2.81 (0.83x) | 5.48 (0.89x) | 111.0 kB |
| | cnn16 | 4.98 (0.63x) | 2.50 (0.46x) | 1.64 (0.49x) | 4.57 (0.74x) | 65.1 kB |
| | cnn18 | 4.48 (0.57x) | 2.20 (0.40x) | 1.50 (0.44x) | 4.17 (0.68x) | 62.3 kB |
| | R64 | 6.11 (0.78x) | 3.78 (0.69x) | 2.47 (0.73x) | 5.18 (0.84x) | 83.1 kB |
| VGG-16 | paper | 35.5 | 24.5 | 93.2 | 154.5 | 3,432.9 kB |
| | cnn17 | 21.4 (0.60x) | 10.5 (0.43x) | 55.6 (0.60x) | 118.3 (0.77x) | 2,339.1 kB |
| | cnn18 | 20.6 (0.58x) | 9.5 (0.39x) | 63.6 (0.68x) | 125.7 (0.81x) | 2,263.2 kB |
| | R64 | 20.9 (0.59x) | 10.1 (0.41x) | 35.9 (0.38x) | 97.5 (0.63x) | 2,265.6 kB |
| GPT-2, 12 blocks, 64 tokens | paper | 260.0 | 161.6 | 795.0 | 1,391 | 62.07 MB |
| | R8 | 189.1 (0.73x) | 89.2 (0.55x) | 531.7 (0.67x) | 1,124 (0.81x) | 46.27 MB |
| | R16 | 174.6 (0.67x) | 74.7 (0.46x) | 487.9 (0.61x) | 1,078 (0.77x) | 41.48 MB |

Derive and the products do not depend on the commitment; the column check falls with the columns
opened and the multiproofs, and the prover's opening with the `sum N k t` of its column products.

## 3. The compact encoding

`run_query(..., wire=True)` sends the claims in the format `PVC3` of `pvi.fullcheck.claimcodec` (the
module docstring gives the byte layout) and `u` and the opened columns 31-bit packed. The prover
encodes where it keeps the claims (its GPU; the host in lean mode); the column check takes the
received int32 rows as they are. Parameters, challenges and the soundness bound are those of the
unencoded flow; without `wire` nothing of the codec runs.

```bash
cd code && export PYTHONPATH=$PWD/src
python experiments/6_improvements/wire.py --mnist <dir holding MNIST/raw> --out experiments/6_improvements/results/wire_laptop.json \
    lenet5 mlp_mnist vgg16 resnet18_cifar gpt2:64 gpt2:512 llama2-7b:64:1,2 qwen3-4b:8:1,2,4,8
python experiments/6_improvements/wire.py --out experiments/6_improvements/results/wire_laptop.json --summary
```

Results (`results/wire_laptop.json`; the CNNs random-initialised and queried on MNIST digits, padded to
32x32 on 3 channels for the CIFAR shapes; the decoders the benchmark's random-weight builds; bytes
exact):

| Model | Claims | int32 (B) | PVC3 (B) | Bits/claim | Smaller |
|---|---:|---:|---:|---:|---:|
| MLP MNIST (trained) | 778 | 3,112 | 1,871 | 19.24 | 1.66x |
| LeNet-5 | 6,518 | 26,072 | 13,413 | 16.46 | 1.94x |
| VGG-16 | 277,514 | 1,110,056 | 602,346 | 17.36 | 1.84x |
| ResNet-18 CIFAR | 614,410 | 2,457,640 | 1,256,852 | 16.36 | 1.96x |
| GPT-2, 12 blocks, 64 tokens | 5,456,977 | 21,827,908 | 11,733,276 | 17.20 | 1.86x |
| GPT-2, 12 blocks, 512 tokens | 43,304,017 | 173,216,068 | 91,297,251 | 16.87 | 1.90x |
| Llama-2-7B, 2 blocks, 64 tokens | 5,733,632 | 22,934,528 | 13,142,647 | 18.34 | 1.75x |
| Qwen3-4B, 8 blocks, 8 tokens | 2,138,496 | 8,553,984 | 4,905,934 | 18.35 | 1.74x |

Per block: Qwen3-4B 568,119 B (18.49 bits per claim), so its 36 blocks take 36,079,104 → 20,813,273 B
(1.73x); Llama-2-7B 6,410,657 B per block, 349,303,808 → 205,462,357 B for 32 blocks (1.70x). In mode C
the opened columns and `u` shrink by exactly 1/32. Whole proofs at λ = 128: LeNet-5 C 130,840 →
116,001 B (1.13x), Kpre 26,072 → 13,577 B (1.92x); VGG-16 C 1.20x; ResNet-18 C 1.37x; GPT-2 at 64
tokens C 1.22x, Kpre 1.87x; at 512 tokens C 1.64x. Decoding (`verify_decode`) costs 16–58% of the
unencoded verifier's time on the laptop (the most for the smallest Kpre proof, the MLP's, at 56–58%;
GPT-2 at 64 tokens, Kpre: 33.1 of 100.2 ms at 1 thread); it runs at 102–158 M claims/s on one thread
for the decoders and 178–242 M claims/s on 4.

**With a commitment plan.** The two options compose: the claims and `u` travel as without a plan, and
the opened columns of every tree travel as one run of 31-bit field elements, tree after tree, which
the verifier unpacks per tree into the int32 rows its checks take. Fiat–Shamir absorbs the plan's
statement and then the encoded bytes. `plan_bytes.py --wire off on` measures both:

```bash
P="paper tight cnn16 cnn17 cnn18 R8 R16 R64"
python experiments/6_improvements/plan_bytes.py --run lenet5 --policies $P --wire off on --queries 5 \
    --out results/combined_run_lenet5.csv
python experiments/6_improvements/plan_bytes.py --run vgg16 --policies paper tight cnn16 cnn17 cnn18 R16 R64 \
    --wire off on --queries 3 --out results/combined_run_vgg16.csv
python experiments/6_improvements/plan_bytes.py --run gpt2:64:12 --policies paper R16 --wire off on --queries 3 \
    --out results/combined_run_gpt2.csv
python experiments/6_improvements/plan_bytes.py --run gpt2:64:12 --claims-only --policies $P --wire off on \
    --queries 3 --out results/combined_model_gpt2.csv                      # measured claims, the byte model
python experiments/6_improvements/plan_bytes.py --run llama2-7b:64:1,2 --claims-only --policies $P --wire off on \
    --queries 3 --out results/combined_model_llama.csv
python experiments/6_improvements/plan_bytes.py --run qwen3-4b:8:4,8 --claims-only --policies paper tight R16 R64 \
    --wire off on --queries 3 --out results/combined_model_qwen.csv
python experiments/6_improvements/perf.py --model lenet5 --random-init --modes C --queries 31 --threads 1 \
    --policy paper cnn16 cnn17 --wire off on > experiments/6_improvements/results/perf_combined_laptop_lenet5.jsonl
```

Proof bytes at λ = 128, interactive / Fiat–Shamir (CNN rows and GPT-2 R16 are `run_query` medians;
the other decoder cells the byte model on measured claims, with Llama-2-7B's and Qwen3-4B's encoded
claims extrapolated from 1- and 2-block, and 4- and 8-block builds):

| Row | Default | Encoded | Plan | Plan + encoding | Smaller |
|---|---:|---:|---:|---:|---:|
| LeNet-5 C | 130.6 / 172.8 kB | 115.7 / 156.6 | cnn17 63.2 / 79.6 | cnn17 49.7 / 65.5 | 2.63x / 2.64x |
| VGG-16 C | 3,432.7 / 4,460.3 kB | 2,870.9 / 3,867.8 | cnn18 2,263.0 / 2,753.1 | cnn18 1,731.8 / 2,207.0 | 1.98x / 2.02x |
| GPT-2, 64 tokens, C | 62.07 / 80.69 MB | 50.73 / 68.77 | R16 41.48 / 50.25 | R16 30.76 / 39.25 | 2.02x / 2.06x |
| Qwen3-4B, 8 tokens, Kpre | 36.08 MB | 20.78 | (no columns) | 20.78 | 1.74x |
| Llama-2-7B, 64 tokens, C | 761.7 / 951.1 MB | 605.1 / 788.6 | R64 501.0 / 571.6 | R64 352.4 / 420.8 | 2.16x / 2.26x |

The gains add up because they shrink different parts: the plans cut the opened columns and paths,
the encoding the claims (1.7–1.9x) and `u` and the columns by 1/32. On the laptop
(`results/perf_combined_laptop_*.jsonl`, 1 thread, ratios to `paper` without encoding) every plan with
the encoding verifies faster than the basic protocol (0.67–0.81x: LeNet-5 cnn16 0.76x, VGG-16 R64
0.67x, GPT-2 R16 0.81x), since the encoding's cost is `verify_decode` (0.6–0.7 ms on LeNet-5, 3.3–4.0 ms
on VGG-16, 45–57 ms on GPT-2); the CPU prover pays its encoder (`prove_encode`: 2.5–2.8 ms on LeNet-5,
21–23 ms on VGG-16, 0.20–0.25 s on GPT-2; a GPU prover encodes on its device, section 6). For Qwen3-4B
with 8 blocks at 8 tokens (Kpre, λ = 40, separate processes) the verifier takes 45.3 ms without and
55.7 ms with the encoding (1.23x) for a 1.74x smaller proof.

## 4. Transposed layout and fusion (the policies with the suffix `c`)

* **Transposed layout.** The rows of `A^T` (one per input coordinate, of length `N`) are Reed–Solomon
  encoded at the policy's length, and a Merkle leaf holds a column of length `k`. The prover sends no
  `u` for such a matrix. The verifier draws `χ'` (`r x M` over the claims' columns; the identity when
  `M ≤ r`, e.g. the LM head with `M = 1` and every operation of a one-token prompt), computes
  `z = χ' Z^T` and `w = χ' [X ; 1]^T` itself, and checks `w · E'[:, c] == Enc(z)[c]` at the opened
  columns. It opens `t k` field elements instead of `r k + t N`.
* **Fusion.** Linear operations that read one tensor with one row length (q/k/v, also with grouped
  k and v; gate/up) may share one transposed matrix `[A_1 ; A_2 ; ...]^T`: one `χ'`, one `w` and one set
  of `t` columns.
* **The choice.** Per set of operations with one input and row length the planner takes rows, each
  operation transposed alone, or one fused matrix, by descent from the base policy's plan on the
  expected bytes at λ = 128, so a `c` plan never costs more than its base policy in that model.
  Every decoder takes q/k/v and gate/up fused, fc1, the attention output and the head transposed,
  fc2/down in rows; convolutions stay in rows, so the CNNs transpose a classifier or two at most.
* **Message flow.** claims → `χ` and `χ'` → `u` of the row matrices → the columns of every tree (one
  round) → the openings: the basic protocol's three prover messages. Drawing every tree's columns in
  one round lets row and transposed matrices of one codeword length share a tree.
* **Prover and setup.** The prover recomputes an opened transposed column from its device-resident
  weights; setup streams the column digests, so the head's `A^T` needs no buffer of its encoding.

`tests/test_plans.py` runs its plan tests with `tightc`, `cnn12c` and `R8c` too, and
`tests/test_layouts.py` holds the rest.

```bash
C="cnn16 cnn16c cnn17 cnn17c cnn18 cnn18c R16 R16c R64 R64c"
python experiments/6_improvements/plan_bytes.py --out results/plan_bytes_col.csv
python experiments/6_improvements/plan_bytes.py --run mlp_mnist lenet5 vgg16 --policies paper $C --wire off on \
    --queries 5 --out results/col_run_cnn.csv
```

On the CNNs the transposed layout finds little (0–3%; their `N` is small against their `k`). Proof
bytes at λ = 128, interactive / Fiat–Shamir (`results/col_run_cnn.csv`, `run_query` medians of 5
queries):

| Model | Policy | Plan | Plan + encoding | `c` plan | `c` plan + encoding |
|---|---|---:|---:|---:|---:|
| MLP (MNIST) | cnn18 | 90.8 / 130.9 kB | 87.1 / 126.3 | 88.0 / 126.2 | 84.5 / 121.4 |
| LeNet-5 | cnn18 | 62.2 / 76.4 kB | 48.8 / 62.5 | 62.0 / 76.2 | 48.5 / 62.5 |
| VGG-16 | cnn18 | 2,263.0 / 2,753.3 kB | 1,731.9 / 2,207.2 | 2,242.8 / 2,725.2 | 1,712.5 / 2,179.9 |

On the decoders it removes most of `u` and most of the opened columns (section 5 gives their bytes).
Setup of the `c` plans (encoded entries, G; lookup tables are hashed, not encoded): GPT-2 `tightc`
0.69, `R16c` 2.77, `R64c` 11.08, `cnn18c` 9.87; Llama-2-7B 37.0, 148.2, 592.7, 138.5; Qwen3-4B 25.3,
101.3, 405.3, 98.0. Transposed, the head's codeword runs along the vocabulary but it has only `d` rows,
so `cnn<e>c` encodes far less than `cnn<e>`; `R<R>c` costs up to 27% more than `R<R>`, because the
fused q/k/v and gate/up round their stacked rows up to a power of two.

## 5. Embedding lookups and last-block pruning

* **Lookup tables (mode C, every `c` policy).** The table `W` `[d, V]` of an embedding operation without
  a bias is committed as a Merkle tree over its `V` rows (`commitment.TableCommitment`: leaf `j` =
  SHA-256 of a domain tag, the table's tag, `j` and row `j`'s `d` int8 bytes; the padding leaves in
  another domain). Its claims are the looked-up rows, one byte each, with one multiproof over the
  distinct ids (`Prover.open_tables`). Before it draws any challenge the verifier checks that the
  claims are int8 and every id names a row (`range_or_shape`), equal rows for equal ids
  (`lookup_consistency`) and the multiproof (`lookup_merkle`); no `u`, no columns, no Freivalds check.
  The LM head keeps its encoded commitment. A lookup drops the table's `u` (`4 r V` bytes: 1.0 MB for
  GPT-2's tokens) and opened columns, for one multiproof of at most one path per position. The verifier
  keeps the tables in graph order (the rows on the wire, the transcript, the checks).
* **Modes K and Kpre** (`Verifier(lookups=True)`): the verifier computes the embedding operations from
  its own weights (rows at the ids, plus the bias of an embedding with one), the prover sends no
  claims for them, Kpre precomputes no `χ` for them, and Fiat–Shamir absorbs the list of those
  operations. In mode C the tables do this (`lookups=True` there is refused).
* **Last-block pruning.** Only the last position reaches the next-token logits, so the last block
  computes q, the attention output and its projection, the residuals and the MLP at that position;
  `_attention` takes the queries of the last `Tq ≤ T` positions, RoPE takes the offset `T - Tq`, and the
  builder calibrates the pruned block on every position, so the pruned graph gives the same logits.
  The plan treats the last block's q as a set of its own; `analytic.decoder_shapes(...,
  prune_last=True)` gives its shapes.

```bash
A="paper tightc cnn16c cnn17c cnn18c R8c R16c R64c"
python experiments/6_improvements/plan_bytes.py --policies $A --kpre off on --out results/lookup_bytes.csv
python experiments/6_improvements/plan_bytes.py --policies $A --kpre off on --prune-last --out results/lookup_bytes_prune.csv
python experiments/6_improvements/plan_bytes.py --run gpt2:64,512:12 --policies paper tightc R16c cnn16c --wire off on \
    --queries 3 --out results/lookup_run_gpt2.csv                          # and --prune-last: _prune
python experiments/6_improvements/plan_bytes.py --run llama2-7b:64:1,2:1024 qwen3-4b:8:1,2:1024 opt-125m:2048:1,2 \
    --policies tightc R8c --kpre off on --wire off on --queries 2 --prune-last --out results/lookup_run_decoders.csv
python experiments/6_improvements/plan_bytes.py --run qwen3-4b:8:4,8 --policies --kpre off on --wire off on --queries 3 \
    --out results/lookup_kpre_qwen.csv                                     # Kpre (and --prune-last)
python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C --queries 11 --threads 1 --device cpu \
    --policy paper R16 R16c --wire off on --prune-last off on > experiments/6_improvements/results/perf_lookup_laptop_gpt2.jsonl
```

(and `--claims-only --kpre off on --wire off on`, with and without `--prune-last`, for
`llama2-7b:1,64:1,2`, `llama2-7b:2048:1,2`, `qwen3-4b:8,64:4,8` and `opt-{125m,1.3b,2.7b,6.7b}:2048:1,2`:
`results/lookup_model_*.csv`.)

**GPT-2 with all 12 blocks**, interactive / Fiat–Shamir, MB (`run_query` medians of 3 random prompts,
within 7 kB of the byte model; (model): the byte model on the measured claims):

| Prompt | Policy | `c` plan | pruned | + encoding | + encoding, pruned |
|---|---|---:|---:|---:|---:|
| 64 | paper | 62.07 / 80.69 | 60.72 / 79.34 | 50.73 / 68.77 | 50.01 / 68.06 |
| 64 | tightc | 30.45 / 34.54 | 29.30 / 33.49 | 20.36 / 24.32 | 19.84 / 23.89 |
| 64 | cnn16c | 26.78 / 29.19 | 25.49 / 27.93 | 16.80 / 19.14 | 16.14 / 18.51 |
| 64 | cnn17c | 26.08 / 28.12 (model) | 24.78 / 26.85 (model) | 16.12 / 18.10 (model) | 15.46 / 17.46 (model) |
| 64 | cnn18c | 25.50 / 27.26 | 24.20 / 25.98 | 15.56 / 17.27 | 14.89 / 16.62 |
| 64 | R16c | 26.89 / 29.39 | 25.65 / 28.19 | 16.91 / 19.33 | 16.30 / 18.76 |
| 64 | R64c | 25.57 / 27.35 (model) | 24.29 / 26.10 | 15.63 / 17.35 (model) | 14.98 / 16.73 |
| 512 | paper | 213.46 / 232.08 | 202.47 / 221.09 | 130.31 / 148.35 | 124.55 / 142.60 |
| 512 | tightc | 179.85 / 183.94 | 169.07 / 173.25 | 100.00 / 103.96 | 94.43 / 98.49 |
| 512 | cnn18c | 174.90 / 176.66 | 163.96 / 165.75 | 95.20 / 96.90 | 89.49 / 91.21 |
| 512 | R16c | 176.29 / 178.79 | 165.42 / 167.96 | 96.55 / 98.97 | 90.90 / 93.36 |
| 512 | R64c | 174.97 / 176.75 (model) | 164.06 / 165.87 | 95.26 / 96.99 (model) | 89.58 / 91.33 |

(`paper` has no lookup tables; its first column is the basic proof.) What is left of GPT-2 at 64
tokens under `cnn18c` with the encoding, pruned: 10.99 MB of claims (the tables' 98,304 bytes among
them), 0.71 MB of `u` (the fc2s', in rows), 3.12 MB of opened columns and 0.06 MB of paths.

**Larger decoders**: the byte model on the whole model with the claims measured on builds of 1 and 2
blocks (Qwen3-4B: 4 and 8), validated by `run_query` on pruned 1- and 2-block builds
(`results/lookup_run_decoders.csv`: every build's claim, `u` and column bytes equal the model's).
Interactive / Fiat–Shamir, MB:

| Model, prompt | paper | paper, pruned | R64c | cnn18c | R64c + enc. | cnn18c + enc. | R64c + enc., pruned | cnn18c + enc., pruned |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2-7B, 1 | 418.0 / 607.4 | 418.0 / 607.4 | 58.7 / 83.2 | 77.9 / 111.5 | 54.8 / 78.5 | 73.4 / 106.0 | 55.2 / 79.1 | 73.8 / 106.5 |
| Llama-2-7B, 64 | 761.7 / 951.1 | 753.1 / 942.5 | 401.6 / 426.1 | 420.9 / 454.5 | 256.8 / 280.5 | 275.5 / 308.0 | 252.0 / 276.0 | 270.6 / 303.3 |
| Llama-2-7B, 2,048 | 11,586.2 / 11,775.6 | 11,305.3 / 11,494.7 | 11,201.9 / 11,226.4 | 11,221.2 / 11,254.7 | 6,662.7 / 6,686.5 | 6,681.4 / 6,713.9 | 6,475.0 / 6,498.9 | 6,493.6 / 6,526.3 |
| Qwen3-4B, 8 | 410.3 / 582.0 | 409.5 / 581.2 | 78.3 / 96.8 | 89.2 / 113.5 | 61.8 / 79.7 | 72.4 / 95.9 | 61.6 / 79.6 | 72.1 / 95.7 |
| Qwen3-4B, 64 | 658.6 / 830.3 | 651.4 / 823.1 | 326.2 / 344.7 | 337.1 / 361.4 | 204.2 / 222.1 | 214.8 / 238.3 | 200.2 / 218.2 | 210.7 / 234.3 |
| OPT-125M, 2,048 | 732.5 / 751.2 | 688.5 / 707.1 | 687.1 / 688.9 | 687.0 / 688.8 | 372.2 / 373.9 | 372.2 / 373.9 | 350.9 / 352.7 | 350.8 / 352.5 |
| OPT-1.3B, 2,048 | 3,807.1 / 3,875.5 | 3,689.7 / 3,758.1 | 3,654.2 / 3,663.9 | 3,657.2 / 3,668.1 | 2,062.2 / 2,071.6 | 2,065.1 / 2,075.7 | 2,002.1 / 2,011.7 | 2,005.0 / 2,015.6 |
| OPT-2.7B, 2,048 | 6,319.7 / 6,428.7 | 6,173.0 / 6,281.9 | 6,085.7 / 6,101.6 | 6,093.3 / 6,112.3 | 3,444.8 / 3,460.1 | 3,452.1 / 3,470.5 | 3,373.9 / 3,389.4 | 3,381.3 / 3,399.8 |
| OPT-6.7B, 2,048 | 10,101.0 / 10,270.7 | 9,866.3 / 10,035.9 | 9,738.0 / 9,763.8 | 9,757.4 / 9,792.5 | 5,690.7 / 5,715.8 | 5,709.6 / 5,743.5 | 5,601.9 / 5,627.2 | 5,620.7 / 5,654.8 |

At one token pruning changes no claim and opens one set of columns more (the pruned q is a matrix of
its own), so it is for prompts; at 2,048 tokens the proof is 96–99.5% claims, so the plans move it by
1–3% and the encoding by 1.7–1.9x.

**Mode Kpre** (the proof is the claims), MB:

| Model, prompt | Kpre | + enc. | lookups | lookups + enc. | pruned | pruned + enc. | lookups, pruned | lookups, pruned + enc. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| GPT-2, 64 | 21.8 | 11.7 | 21.4 | 11.6 | 20.5 | 11.0 | 20.1 | 10.9 |
| GPT-2, 512 | 173.2 | 91.3 | 170.1 | 90.5 | 162.2 | 85.5 | 159.1 | 84.7 |
| Llama-2-7B, 1 | 5.6 | 3.3 | 5.6 | 3.3 | 5.6 | 3.3 | 5.6 | 3.3 |
| Llama-2-7B, 64 | 349.3 | 205.5 | 348.3 | 205.1 | 340.7 | 200.3 | 339.6 | 199.9 |
| Llama-2-7B, 2,048 | 11,173.8 | 6,614.8 | 11,140.2 | 6,602.7 | 10,892.9 | 6,428.3 | 10,859.3 | 6,414.6 |
| Qwen3-4B, 8 | 36.1 | 20.8 | 36.0 | 20.8 | 35.3 | 20.3 | 35.2 | 20.3 |
| Qwen3-4B, 64 | 284.4 | 163.2 | 283.7 | 163.0 | 277.2 | 159.0 | 276.5 | 158.8 |
| OPT-125M, 2,048 | 692.3 | 368.1 | 679.7 | 364.9 | 648.2 | 348.2 | 635.7 | 343.5 |
| OPT-6.7B, 2,048 | 9,731.0 | 5,635.6 | 9,663.9 | 5,618.4 | 9,496.2 | 5,556.6 | 9,429.1 | 5,529.2 |

Qwen3-4B at 8 tokens in Kpre: 36.08 MB (the basic proof) → 35.19 MB with lookups and pruning → 20.33
MB with the encoding too (`results/lookup_kpre_qwen*.csv`, `run_query` on 4- and 8-block builds
extrapolated to 36).

**Timing on the laptop** (GPT-2, 12 blocks, 64 tokens, mode C, 1 thread, 11 random prompts, every
variant committed and its queries interleaved; `results/perf_lookup_laptop_gpt2.jsonl`; ratios to
`paper` without encoding):

| Policy | Pruned | Encoded | Verifier (ms) | derive | products | columns | lookups | verify_decode | Prover (ms) | Proof |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| paper | no | no | 269.4 | 72.2 | 33.5 | 163.7 | | | 1,491 | 62.07 MB |
| | no | yes | 307.1 (1.14x) | 71.3 | 34.7 | 144.6 | | 56.5 | 1,735 (1.16x) | 50.68 MB |
| | yes | no | 264.5 (0.98x) | 66.5 | 32.7 | 165.3 | | | 1,481 (0.99x) | 60.72 MB |
| | yes | yes | 300.4 (1.12x) | 65.5 | 33.4 | 144.6 | | 56.8 | 1,720 (1.15x) | 49.97 MB |
| R16 | no | no | 185.8 (0.69x) | 74.2 | 34.8 | 76.8 | | | 1,175 (0.79x) | 41.48 MB |
| | yes | yes | 203.0 (0.75x) | 64.6 | 33.5 | 62.7 | | 42.3 | 1,357 (0.91x) | 29.99 MB |
| R16c | no | no | 159.6 (0.59x) | 72.8 | 45.7 | 39.5 | 1.6 | | 743 (0.50x) | 26.89 MB |
| | no | yes | 188.2 (0.70x) | 70.8 | 45.5 | 34.8 | 1.5 | 35.6 | 902 (0.61x) | 16.86 MB |
| | yes | no | 153.0 (0.57x) | 67.6 | 43.5 | 40.4 | 1.5 | | 729 (0.49x) | 25.65 MB |
| | yes | yes | 181.5 (0.67x) | 66.9 | 43.8 | 36.1 | 1.5 | 33.2 | 884 (0.59x) | 16.26 MB |

Pruning takes the verifier to 0.96x of the unpruned (derive 0.93x) and the prover to 0.98x, for 0.95x
of the proof. In mode Kpre (`results/perf_lookup_laptop_{gpt2,qwen}_kpre.jsonl`) the lookups save
bytes, not time (GPT-2 at 64 tokens: 95.9 ms, 96.7 ms with lookups), and pruning saves both (90.1 ms).

## 6. Encoder and decoder speed

`PVC3`'s bytes are fixed by the reference encoder (`tests/claimcodec_reference.py`); every encoder
below gives its bytes for every input.

* **A device's encoder** (`encode(..., impl="device")`, the default for claims on a GPU): the claims in
  one int32 buffer; each operation's samples gathered by a cached index, all samples sorted once with
  their segment in the key's high bits, the medians and bit-length counts read off by binary search;
  one copy back of the statistics, the plan on the host, then the values, exceptions, packing and Rice
  coding on the device and one copy back of the result through pinned memory. That is a fixed number
  of kernels plus about ten per distinct width (145–186 kernel-launching operations per query for
  every model here) and four waits for the device.
* **The host's encoder** (`impl="host"`, the default for CPU claims): the statistics of chunks of
  operations on `workers` threads, in place on scratch buffers, and a packer that writes each lane
  straight into the output.
* **Field elements and lookup rows**: `pack_field` (`u`, the opened columns) and `pack_rows` pack on the
  prover's device and copy back once. A lean GPU prover keeps each claim on its device as clipped int32
  and encodes there, so about 2.3 bytes per claim cross PCIe instead of 8.
* **The decoder**: with several workers and at least 64 K exceptions both Rice vectors decode as jobs of
  their own while the slot streams are unpacked; a GPU client decodes straight into pinned memory.
  Malformed input raises `ClaimCodecError` only, and the parallel decoder accepts and rejects exactly
  what the one-thread decoder does (400 random corruptions).

```bash
git worktree add ../base 6c2059a                    # the reference encoder, next to this checkout
cd code && export PYTHONPATH=$PWD/src
C=$PWD; B=$C/../../base/code; R=experiments/6_improvements/results
python experiments/6_improvements/codec_stages.py --base $B/src --threads 4 lenet5 vgg16 gpt2:64:12 qwen3-4b:8:8 \
    qwen3-4b:8:8/36 --out $R/codec_profile_laptop.json
python experiments/6_improvements/codec_ab.py --base $B/src --device cpu --threads 1,4,8 --reps 15 --device-path \
    lenet5 vgg16 gpt2:64:12 qwen3-4b:8:8 qwen3-4b:8:8/36 --out $R/codec_laptop.json
python experiments/6_improvements/wire_hashes.py --base $B/src --out $R/codec_bytes_identity.jsonl
python experiments/6_improvements/claims_drop.py --threads 4      # also run from the reference checkout, and at 1 thread
```

**Encoding on the laptop** (`results/codec_laptop.json`: the claims of one query, both encoders in one
process, interleaved, medians of 15, ms; reference → current):

| Claims | Ops | Claims | 1 thread | 4 threads | 8 threads | Kernel ops: reference GPU path → device path |
|---|---:|---:|---:|---:|---:|---|
| LeNet-5 | 5 | 6,518 | 2.8 → 2.1 (1.33x) | 2.8 → 2.1 (1.30x) | 2.8 → 2.2 (1.32x) | 94 → 156 |
| VGG-16 | 16 | 277,514 | 17.6 → 16.6 (1.06x) | 26.2 → 24.6 (1.07x) | 27.3 → 25.1 (1.09x) | 255 → 186 |
| GPT-2, 64 tokens | 75 | 5,456,977 | 163 → 144 (1.13x) | 194 → 135 (1.44x) | 221 → 129 (1.71x) | 868 → 176 |
| Qwen3-4B, 8 tokens, 8 blocks | 58 | 2,138,496 | 46.0 → 26.9 (1.71x) | 52.1 → 24.8 (2.10x) | 77.6 → 33.0 (2.35x) | 352 → 145 |
| Qwen3-4B, 8 tokens, 36 blocks | 254 | 9,019,776 | 217 → 126 (1.73x) | 250 → 122 (2.05x) | 273 → 122 (2.24x) | 1332 → 145 |

On this 4-core laptop the host encoder's passes are memory-bound, so threads help little; the
decoder is within 0.77–1.14x of the reference decoder. In full queries (`results/perf_codec_laptop.jsonl`,
CPU prover and verifier, three processes per version) the prover's encoding cost falls 1.2x on GPT-2
and 1.7x on Qwen3-4B at 1 thread, and 1.3x on both at 4 threads. At 4 threads, releasing the claims
after the host encoder has run on them takes about 55 ms on this laptop (`results/claims_drop_laptop.txt`;
the cause is not identified), which `prove_encode` includes. The encoded messages of both versions
hash the same (`wire_hashes.py`: 88 messages of 42 queries over 15 cases,
`results/codec_bytes_identity.jsonl`).

On a GPU node (`OUT` a scratch directory):

```bash
python -m pytest tests/test_claimcodec.py tests/test_wire.py tests/test_gpu_verifier.py -q -p no:cacheprovider -o addopts=""
python experiments/6_improvements/codec_ab.py --base ../../base/code/src --device cuda --threads 1,8 --reps 15 \
    lenet5 vgg16 gpt2:64 gpt2:512 qwen3-4b:8 --out $OUT/codec_ab_gpu.json
python experiments/6_improvements/codec_ab.py --base ../../base/code/src --device cuda --lean --threads 8 --reps 15 \
    gpt2:64 qwen3-4b:8 --out $OUT/codec_ab_gpu_lean.json
```

## 7. The planning rule `auto`

`plan_commitment(ops, "auto", setup_budget=None)` (and `bench.py --policy auto`, cells `_polauto`)
resolves, on the whole model's shapes, to one of `tightc cnn16c cnn17c cnn18c R16c R64c`
(`plans.AUTO_CANDIDATES`): the one with the fewest expected non-claim bytes at λ = 128, interactive
(`plans.plan_overhead`: `u`, opened columns and the encoded trees' multiproofs), among those whose
encoded entries (`analytic.setup_size`) are within the budget, by default max(2 x the basic
protocol's, 2^34). Ties go to the first candidate; with no candidate within the budget it falls back
to the candidate of least setup. The plan is exactly the chosen policy's (recorded as the cells'
`policy`, with `policy_requested` = `auto`), so the verifier's key and the Fiat–Shamir statement are
that policy's (`tests/test_plan_auto.py`). Setup in encoded entries and non-claim bytes of one query
(λ = 128, interactive), against the basic protocol's (in brackets) and the best candidate with no
budget:

| Model | auto | setup (basic) | budget | non-claim bytes (basic) | vs basic | best candidate: bytes, setup | auto vs best |
|---|---|---:|---:|---:|---:|---|---:|
| MLP-MNIST | `cnn18c` | 0.28 G (3.2 M) | 17.2 G | 84.8 kB (267 kB) | 0.32x | `cnn18c` | 1.00x |
| LeNet-5 | `cnn18c` | 72 M (0.31 M) | 17.2 G | 36.0 kB (105 kB) | 0.34x | `cnn18c` | 1.00x |
| VGG-11 | `cnn18c` | 0.99 G (70 M) | 17.2 G | 735 kB (1.59 MB) | 0.46x | `cnn18c` | 1.00x |
| VGG-16 | `cnn18c` | 1.38 G (0.11 G) | 17.2 G | 1.13 MB (2.32 MB) | 0.49x | `cnn18c` | 1.00x |
| ResNet-18 | `cnn18c` | 1.26 G (80 M) | 17.2 G | 1.03 MB (2.17 MB) | 0.48x | `cnn18c` | 1.00x |
| GPT-2 | `cnn18c` | 9.9 G (0.86 G) | 17.2 G | 3.95 MB (40.2 MB) | 0.10x | `cnn18c` | 1.00x |
| OPT-125M | `cnn18c` | 9.9 G (0.87 G) | 17.2 G | 3.95 MB (40.3 MB) | 0.10x | `cnn18c` | 1.00x |
| OPT-350M | `cnn17c` | 13.1 G (2.7 G) | 17.2 G | 12.4 MB (83.4 MB) | 0.15x | `cnn18c`: 10.7 MB, 26 G | 1.16x |
| OPT-1.3B | `cnn16c` | 13.2 G (10.6 G) | 21.3 G | 36.3 MB (149 MB) | 0.24x | `R64c`: 21.5 MB, 118 G | 1.69x |
| OPT-2.7B | `cnn16c` | 21.8 G (17.6 G) | 35.2 G | 65.7 MB (238 MB) | 0.28x | `R64c`: 35.0 MB, 247 G | 1.88x |
| Qwen3-4B | `cnn17c` | 49.7 G (27.8 G) | 55.6 G | 65.0 MB (374 MB) | 0.17x | `R64c`: 42.3 MB, 405 G | 1.54x |
| OPT-6.7B | `cnn17c` | 69 G (54 G) | 107 G | 96.1 MB (370 MB) | 0.26x | `R64c`: 57.0 MB, 601 G | 1.68x |
| Llama-2-7B | `tightc` | 37 G (29.8 G) | 59.6 G | 131 MB (412 MB) | 0.32x | `R64c`: 53.1 MB, 593 G | 2.47x |
| OPT-13B | `cnn17c` | 108 G (84 G) | 167 G | 165 MB (568 MB) | 0.29x | `R64c`: 87.3 MB, 1,203 G | 1.89x |
| Llama-2-13B | `cnn17c` | 108 G (78 G) | 156 G | 160 MB (640 MB) | 0.25x | `R64c`: 83.7 MB, 977 G | 1.91x |
| OPT-30B | `cnn17c` | 181 G (139 G) | 278 G | 322 MB (941 MB) | 0.34x | `R64c`: 152 MB, 2,380 G | 2.12x |
| OPT-66B | `cnn18c` | 621 G (470 G) | 939 G | 451 MB (1,600 MB) | 0.28x | `R64c`: 251 MB, 6,840 G | 1.80x |
| Llama-2-70B | `tightc` | 323 G (287 G) | 573 G | 696 MB (2,030 MB) | 0.34x | `R64c`: 285 MB, 5,170 G | 2.44x |

The CNNs and the decoders up to 125M parameters fit `cnn18c` (their best) within the 2^34 floor; from
OPT-1.3B on, the budget of twice the basic protocol's setup keeps `auto` at 1.1–1.8x the basic
protocol's encoded entries, at 1.5–2.5x the non-claim bytes of `R64c`, which needs 11–20x the basic
setup. A larger `setup_budget` trades setup for bytes (`setup_budget=math.inf`: the overall least).

The table is for the unpruned graphs. The L40S runs record the choice in each cell (`policy`): it is
the table's for every model, except that Llama-2-7B with the last block pruned (the optimised cells)
gets `cnn16c` instead of `tightc`.

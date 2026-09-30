# 6. Constant-factor improvements: verifier A/B timing

`ab_verifier.py` times the verifier of this checkout against a baseline checkout (the
paper code, commit `729b997`) in the same process: the baseline `pvi.fullcheck` is
imported under another module name, both verifiers check the same transcript, and they
alternate which runs first.  Every repetition must give identical derived tensors and
verdicts, this checkout's prover must send the same claims, `u`, opened columns and
Merkle paths as the baseline's, and every tampered transcript (claims off by one, out of
range or of the wrong dtype; `u` off by one or outside the field; opened columns off by
one, outside the field or narrowed; Merkle paths forged, short or long) must be rejected
by both at the same check.  Models are random-init (costs depend on the shapes, not on
the weight values).  Nothing here writes to the stored benchmark roots.

```bash
git worktree add ../base 729b997                    # the baseline, next to this checkout
cd code
export PYTHONPATH=$PWD/src
python experiments/6_improvements/ab_verifier.py --base ../../base/code/src --out ab.csv \
    --threads 1,4 lenet5:C:128 vgg16:C:128 gpt2:C:128:64
python experiments/6_improvements/ab_verifier.py --base ../../base/code/src --out ab.csv \
    lenet5:Kpre:128 vgg16:Kpre:128 gpt2:Kpre:128:64 qwen3-4b:Kpre:40:8:1 qwen3-4b:Kpre:40:8:24 \
    qwen3-4b:Kpre:40:8:2:1024 qwen3-4b:Kpre:40:8:8:1024
PVI_MERKLE_PROCESSES=1 python experiments/6_improvements/ab_verifier.py --base ../../base/code/src \
    --out ab_merkle_processes.csv --threads 4 vgg16:C:128 gpt2:C:128:64
```

A case is `model:mode:lambda[:tokens[:blocks[:vocab]]]`; a reduced vocabulary keeps the
block shapes, so it measures the per-block slope cheaply.  Transcripts are cached in
`artifacts/fullcheck/cache/ab_verifier/` (`--cache`).  In mode Kpre the verifier keeps
the stacked operands of its fixed `chi` and `u` across queries; the timings are the steady
state after two warm-up queries (modes C and K keep nothing between queries).

## Results (laptop: Intel i7-1165G7, Windows, torch 2.2.2, Python 3.10)

Medians over 11-21 interleaved repetitions (`results/ab_laptop.csv`, with minima and
per-stage rows); `total` is derive + products + columns, the verifier's work per query.

| Model | Mode | lambda | Threads | Baseline (ms) | This checkout (ms) | Ratio |
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

Qwen3-4B with all 36 blocks (4.4 GB of int8 weights) does not fit this laptop's free
memory; 24 blocks do.  Per additional block (1 -> 24 blocks, full vocabulary): 16.40 ->
5.22 ms (3.14x), of which derive 6.96 -> 4.12 ms (1.69x) and the products 9.45 -> 1.08 ms
(8.8x); the vocabulary-1024 builds give the same slope (2 -> 8 blocks: 15.95 -> 5.04 ms,
3.16x).  Extrapolated to 36 blocks: 618 -> 190 ms (3.25x).

Per stage: derive (the cheap operations) 1.0-1.1x on LeNet-5, 1.55-1.77x elsewhere;
products 1.7-1.8x (LeNet-5), 3.0-3.3x (VGG-16), 3.6x (GPT-2 C), 4.7x (GPT-2 Kpre) and
13x / 9x (Qwen3-4B: the embedding and LM head / each block); columns (mode C) 2.0x
(LeNet-5), 2.5-2.6x (VGG-16, GPT-2), whose floor is now SHA-256 of the Merkle paths in
`hashlib`.  Mode C's products are slower than Kpre's because every query has a fresh
`chi`, whose limbs are recomputed.

With `PVI_MERKLE_PROCESSES=1` (off by default) a multi-threaded verifier checks the
multiproofs in worker processes (`results/ab_laptop_merkle_processes.csv`, 4 threads):
VGG-16 C 30.5 ms (2.73x; columns 22.8 -> 19.7 ms) and GPT-2 C 203.2 ms (2.50x; columns
139.3 -> 126.0 ms).

These numbers are of commit `1d3df18`.  The GPU work that followed leaves the CPU verifier
within noise on these cases (`results/ab_laptop_vs_1d3df18.csv`, that commit as the
baseline, 11-41 repetitions: totals 0.99-1.15x); it is faster only where prompts are long
(the int32 attention below) and where Merkle columns are large (threaded leaf hashing).
The review fixes after that (commit `a5adbf3`: one verdict form for the CPU and the GPU, one
message flow for both verifiers) leave it within noise as well
(`results/ab_laptop_a5adbf3_vs_5c9b83c.csv`, 21 repetitions: totals 0.98-1.01x).  Against the
paper code at `a5adbf3` (`results/ab_laptop_a5adbf3.csv`, 11 repetitions), the totals are
LeNet-5 C 1.67x / 1.72x (1 / 4 threads), VGG-16 C 2.59x / 2.25x, GPT-2 C 2.45x / 2.33x,
LeNet-5 Kpre 1.43x, VGG-16 Kpre 2.79x, GPT-2 Kpre 2.57x and Qwen3-4B (1 block) Kpre 5.43x.

## The GPU verifier: `gpu_ab.py` and `attention_ab.py`

What changed for a GPU client (all of it bit-exact, tested in `tests/test_gpu_verifier.py`,
`tests/test_fast_verifier.py` and, on the GPU, `tests/test_gpu_exactness.py`):

- **int32 attention** (`transformer._attention_core`, run by the prover and the verifier):
  int32 scores, the exp table folded into one look-up per score gap (which also replaces the
  second mask pass), and ONE float32 `P V` GEMM.  The int64 code stays as the fallback outside
  the bounds that make this exact (`_int32_scores`).
- **int8 products** (`field.int8_*`): `chi^T Z` and `u^T [X ; 1]` as exact int8 GEMMs
  (`torch._int_mm`) wherever `int8_ok`, which is decided by one self-check per GPU.  The float64
  limb products are used otherwise, and whenever a weight op's input is not int8-valued.
- **One copy back per check**: the range checks, the field check of `u`, the Freivalds
  comparisons and the column code checks leave their verdicts on the device.  The cheap ops'
  constants live on the device (`IntGraph.with_constants_on`).  Masks and tables are cached
  per device, and column indices and opened columns go up through pinned memory.  On a CUDA
  GPU, `test_a_gpu_verifier_makes_one_round_trip_per_check` checks this in sync-debug mode.
- **Columns**: the code checks of all ops are queued on the device before the host hashes the
  Merkle paths.  Columns of 16 KiB or more are hashed on threads, on the CPU too.
- **Streaming verifier** (`Verifier.verify_streaming`, with the wire formats of `pipeline.py`;
  optional: `Verifier(stream=True)`, or `bench.py --verifier-impl stream --tag ..._stream`).
  Claims travel as int32 in pinned memory and are uploaded one op at a time on a side stream.
  Both sides of each Freivalds check are computed as soon as the claim arrives, and in mode C
  the Merkle checks run on a host thread during derive.  One copy back per query decides it.
  It gives the same verdicts and labels as `run_query`, and records one `verify_total` per
  query.  It receives every message before it checks any, so a rejected query has also
  received (and counts) `u` and the openings, which `run_query` stops asking for.

`gpu_ab.py` runs three verifiers back to back on the same transcript, in one process:

- **base**: the paper code, imported from `--base`, driven the way `run_query` drives it.
- **new**: this checkout's verifier, with the same calls.
- **stream**: the streaming verifier.

Before any timing it checks three things:

- both provers send the same claims;
- base and new derive identical tensors;
- all three reject each tampered transcript at the same check.

It records the medians of every stage, the ratios and the peak device memory.  With block
counts 1 and 2 it also extrapolates to the full model.  `--compile` adds a `torch.compile`d
attention core, and `--compile-all` compiles every cheap op; both are checked bit-exact.
`attention_ab.py` times one group of heads against the paper code.  Both scripts write JSON to
`--out`.  The runs for an RTX 2080 Ti (11 GB) are:

```bash
git worktree add ../base 729b997                    # the baseline, next to this checkout
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

The last `gpu_ab.py` command times only the streaming verifier.  The default verifier would hold
OPT-1.3B's claims for 2,048 tokens on the device (15.3 GB) and cannot run on this GPU.

### On this laptop's CPU (no GPU here)

The same scripts run on the CPU, with `--device cpu --verifier-device cpu` and 4 threads
(3-5 repetitions each, `results/gpu_ab_laptop_cpu_*.json`).  There the int8 GEMMs are off,
nothing waits on a device, and the streaming verifier gains nothing from overlap, so these
numbers show the int32 attention and the CPU products, not the GPU paths.

| Case | Mode | Stage | Paper code (ms) | This checkout (ms) | Ratio |
|---|---|---|---:|---:|---:|
| Llama-2-7B, 1 block, 2,048 tokens (vocabulary 1,024) | Kpre | derive | 3,462 | 2,046 | 1.69x |
| | | products | 753 | 244 | 3.08x |
| | | total | 4,267 | 2,283 (stream 2,107) | 1.87x (stream 2.03x) |
| GPT-2, 12 blocks, 512 tokens | Kpre | derive | 842 | 502 | 1.68x |
| | | products | 347 | 113 | 3.08x |
| | | total | 1,195 | 618 (stream 609) | 1.93x (stream 1.96x) |
| GPT-2, 12 blocks, 64 tokens | Kpre | total | 175.6 | 66.5 (stream 82.2) | 2.64x (stream 2.14x) |
| GPT-2, 12 blocks, 64 tokens | C | total | 517.3 | 234.1 (stream 226.6) | 2.21x (stream 2.28x) |

All runs derived identical tensors, and every tampered transcript was rejected at the same
check by all three verifiers.  One group of 4 heads at 2,048 tokens with a head size of 128
(`results/attention_ab_laptop_*.json`, 7 repetitions):

| Threads | Paper code | int64 path (`1d3df18`) | int32 path | Ratio to the paper code |
|---:|---:|---:|---:|---:|
| 1 | 507 ms | 371 ms | 234 ms | 2.17x |
| 4 | 213 ms | 159 ms | 101 ms | 2.11x |

## Commitment plans for mode C: `plan_bytes.py`, `perf.py --policy`, `bench.py --policy`

A commitment plan (`pvi.fullcheck.plans`, opt-in by name; the default `paper` is the report's
commitment and parameters, bit for bit) changes three things of mode C:

* **exact t**: each tree opens the smallest `t` whose exact column error
  `prod_{i<t} (k_l-1-i)/(n_l-i)` meets the per-op budget `2^-beta` the report already assigns
  (`beta = lambda + log2(2L)`, plus 64 grinding bits under Fiat--Shamir), instead of
  `ceil(beta / log2 rate)`, which assumes `(k-1)/n = 1/rate` and ignores the power-of-two padding;
* **shared trees**: matrices of one codeword length share one Merkle tree (leaf `c` =
  SHA-256 of the group tag, `c` and every member's own column digest of column `c`, in the public
  member order), one set of `t_g = max t_l` column indices and one multiproof; the members of a
  length are split into the runs of `t` whose groups minimise the expected column and multiproof
  bytes at lambda = 128 (a tall matrix never opens extra columns; small ones share a multiproof);
* **per-op codeword lengths**: `tight` keeps the report's `4 next_pow2(k)`; `cnn<e>` gives every
  matrix `max(2^e, 2 next_pow2(k))`; `R<R>` gives rate `R` to every matrix but the embedding
  tables (their rows are the vocabulary: setup would multiply for columns of `d` entries).

The bound is the report's: `eps <= sum_l [p^-r + 2^-beta] = 2L 2^-beta = 2^-lambda` (interactive;
`2^-(lambda+64)` per transcript under Fiat--Shamir).  Every member of a group gets `t_g >= t_l`
distinct uniform columns of its own codeword, drawn after `u`; the union bound never needed the
ops' index sets to be independent.  Fiat--Shamir absorbs every op's codeword length and every
group's tag, root, length, `t` and member order.  `tests/test_plans.py` checks `soundness_bits >=
lambda` for the five CNNs and all decoder configurations, every policy, lambda 40/80/128, modes
C, K and Kpre, interactive and Fiat--Shamir, and that honest queries pass and every forgery is
rejected at its check (by `run_query`, the streaming verifier and the GPU forms).

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
    --policy paper tight cnn16 cnn17 cnn18 R8 R16 R64                                 # interleaved timing
```

### Proof bytes (lambda = 128)

Every row of `results/plan_bytes_run_cnn.csv` (random-init CNNs committed under every policy, 5
honest queries each through `run_query`) has claim, `u` and column bytes equal to the byte model
(`analytic.proof_bytes`), and its median total within 0.5% of the model's (the Merkle paths
depend on the random indices).  Measured medians, interactive / Fiat--Shamir:

| Model | paper | tight | cnn16 | cnn17 | cnn18 | R8 | R16 | R64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP (MNIST) | 270.0 / 391.7 kB | 214.6 / 309.1 | 106.8 / 153.1 | 97.2 / 140.3 | 90.8 / 130.8 | 170.1 / 243.1 | 141.6 / 202.8 | 112.8 / 161.4 |
| LeNet-5 | 130.6 / 172.2 kB | 111.0 / 142.7 | 65.0 / 82.7 | 63.2 / 79.6 | 62.2 / 76.6 | 98.1 / 125.3 | 90.4 / 116.8 | 82.7 / 106.1 |
| VGG-11 | 2,201 / 2,908 kB | 1,791 / 2,308 | 1,500 / 1,888 | 1,420 / 1,768 | 1,365 / 1,693 | 1,613 / 2,051 | 1,507 / 1,889 | 1,383 / 1,706 |
| VGG-16 | 3,433 / 4,461 kB | 2,833 / 3,593 | 2,454 / 3,040 | 2,339 / 2,871 | 2,263 / 2,754 | 2,584 / 3,229 | 2,436 / 3,001 | 2,266 / 2,748 |
| ResNet-18 (CIFAR) | 4,630 / 5,583 kB | 4,036 / 4,715 | 3,642 / 4,154 | 3,550 / 4,017 | 3,491 / 3,925 | 3,807 / 4,392 | 3,676 / 4,190 | 3,521 / 3,965 |

GPT-2 with all 12 blocks, committed and queried the same way (`results/plan_bytes_run_gpt2.csv`,
3 queries per cell), interactive / Fiat--Shamir: 64 tokens paper 62,072.9 / 80,689.6 kB, tight
54,610.2 / 69,446.0, R8 46,274.4 / 57,175.8, R16 41,476.6 / 50,246.0; 512 tokens paper 213,461.0 /
232,081.0, tight 205,998.4 / 220,834.2, R8 197,662.5 / 208,563.3, R16 192,864.8 / 201,635.4 --
each within 3.2 kB of the byte model (the multiproofs).

The decoders from the byte model (`results/plan_bytes.csv`; the paper column reproduces the
stored L40S measurements, e.g. GPT-2 T64 62,072 kB vs 62,071.5 kB stored, Llama-2-7B T64 761.7 MB,
Qwen3-4B T8 410.3 MB, OPT-6.7B T2048 10,101 MB), interactive / Fiat--Shamir, MB:

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

The model of a decoder too large to run is checked on 1- and 2-block builds
(`results/plan_bytes_run_decoders.csv`: Qwen3-4B at 8 tokens and Llama-2-7B at 64 tokens with a
1,024-token vocabulary, OPT-125M at 2,048 tokens with its own; tight and R8; interactive and
Fiat--Shamir): every build's claim, `u` and column bytes equal the model's, and the measured
1- and 2-block totals extrapolated linearly over the blocks come within 0.011% of the byte
model of the whole model (e.g. Llama-2-7B R8, 64 tokens: 619.95 MB extrapolated, 619.94 MB
modelled), whose groups the builds share (`plan_commitment(..., model_ops=)`).

Soundness at lambda = 128 (`soundness_bits`, interactive): paper 131-153 bits, every plan 129.2-140.7
bits.  At 2,048 tokens the claims are 94-99% of the proof, so the best plans save 2-4% there.

### Setup (the one-time commitment)

Encoded field entries `sum_l N_l n_l` (what the NTTs produce and the leaves hash) and trees
(`analytic.setup_size`); the stored L40S commits run at about 9 ns per entry (Llama-2-7B: 29.8 G
entries in 257-281 s), which gives the estimates for that node:

| Model | paper | tight | cnn16 | cnn17 | cnn18 | R8 | R16 | R64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| LeNet-5 | 0.3 M, 5 trees | 0.3 M, 4 | 15.5 M, 1 | 30.9 M, 1 | 61.9 M, 1 | 0.6 M, 4 | 1.2 M, 4 | 5.0 M, 4 |
| VGG-16 | 0.11 G, 16 | 0.11 G, 6 | 0.34 G, 4 | 0.69 G, 3 | 1.38 G, 3 | 0.22 G, 6 | 0.44 G, 5 | 1.74 G, 5 |
| ResNet-18 | 0.08 G, 21 | 0.08 G, 8 | 0.32 G, 4 | 0.63 G, 4 | 1.26 G, 4 | 0.16 G, 8 | 0.32 G, 8 | 1.27 G, 8 |
| GPT-2 | 0.86 G, 75 | 0.86 G, 5 | 8.9 G, 4 | 17.7 G, 4 | 35.3 G, 3 | 1.5 G, 4 | 2.8 G, 4 | 10.7 G, 4 |
| Llama-2-7B | 29.8 G, 226 | 29.8 G, 3 | 91 G, 3 | 183 G, 3 | 366 G, 3 | 59 G, 3 | 118 G, 3 | 468 G, 3 |
| Qwen3-4B | 27.8 G, 254 | 27.8 G, 4 | 84 G, 4 | 166 G, 4 | 331 G, 4 | 53 G, 4 | 103 G, 4 | 405 G, 4 |

On this laptop's CPU (NTT on the CPU, shared machine) the commits took 90-160 ns per entry:
LeNet-5 cnn16 2.1 s, cnn18 8.7 s; VGG-16 paper 11 s, cnn16 36 s, cnn17 103 s, cnn18 218 s, R16 58 s,
R64 289 s (`commit_s` in `results/plan_bytes_run_cnn.csv`).  At about 9 ns per entry: VGG-16 cnn17
~6 s and cnn18 ~12 s; GPT-2 R16 ~26 s, R64 ~1.6 min, cnn18 ~5 min; Llama-2-7B R8 ~9 min, R16
~18 min, R64 ~70 min (paper and tight: 4.5 min).  The verifier's key shrinks from one root per
matrix to one per tree (Llama-2-7B: 226 -> 3).

### Timing on this laptop (`perf.py --policy`, `results/perf_plans_laptop_*.jsonl`)

Random-init models, 1 thread, every policy committed in one process and its queries
interleaved with the others' (the order rotating from query to query); medians over 31 (LeNet-5)
and 15 (VGG-16) honest queries, all accepted; the ratios are to `paper` in the same run.  The
laptop was shared with other jobs, so absolute times are noisy; the ratios come from interleaved
runs.

| Model | Policy | Verifier (ms) | Column check (ms) | Prover open (ms) | Prover total (ms) | Proof |
|---|---|---:|---:|---:|---:|---:|
| LeNet-5 | paper | 7.86 | 5.49 | 3.37 | 6.14 | 130.9 kB |
| | tight | 7.30 (0.93x) | 4.86 (0.88x) | 2.81 (0.83x) | 5.48 (0.89x) | 111.0 kB |
| | cnn16 | 4.98 (0.63x) | 2.50 (0.46x) | 1.64 (0.49x) | 4.57 (0.74x) | 65.1 kB |
| | cnn17 | 4.77 (0.61x) | 2.39 (0.44x) | 1.55 (0.46x) | 4.33 (0.71x) | 63.0 kB |
| | cnn18 | 4.48 (0.57x) | 2.20 (0.40x) | 1.50 (0.44x) | 4.17 (0.68x) | 62.3 kB |
| | R8 | 6.59 (0.84x) | 4.31 (0.79x) | 2.49 (0.74x) | 5.08 (0.83x) | 98.3 kB |
| | R16 | 6.52 (0.83x) | 4.08 (0.74x) | 2.48 (0.74x) | 5.22 (0.85x) | 90.8 kB |
| | R64 | 6.11 (0.78x) | 3.78 (0.69x) | 2.47 (0.73x) | 5.18 (0.84x) | 83.1 kB |
| VGG-16 | paper | 35.5 | 24.5 | 93.2 | 154.5 | 3,432.9 kB |
| | tight | 26.2 (0.74x) | 15.5 (0.63x) | 57.1 (0.61x) | 119.4 (0.77x) | 2,833.4 kB |
| | cnn16 | 23.0 (0.65x) | 12.0 (0.49x) | 58.7 (0.63x) | 121.4 (0.79x) | 2,454.2 kB |
| | cnn17 | 21.4 (0.60x) | 10.5 (0.43x) | 55.6 (0.60x) | 118.3 (0.77x) | 2,339.1 kB |
| | cnn18 | 20.6 (0.58x) | 9.5 (0.39x) | 63.6 (0.68x) | 125.7 (0.81x) | 2,263.2 kB |
| | R8 | 24.3 (0.69x) | 13.4 (0.55x) | 55.3 (0.59x) | 116.8 (0.76x) | 2,583.7 kB |
| | R16 | 22.2 (0.63x) | 11.3 (0.46x) | 53.7 (0.58x) | 115.0 (0.74x) | 2,436.3 kB |
| | R64 | 20.9 (0.59x) | 10.1 (0.41x) | 35.9 (0.38x) | 97.5 (0.63x) | 2,265.6 kB |

| GPT-2, 12 blocks, 64 tokens | paper | 260.0 | 161.6 | 795.0 | 1,391 | 62.07 MB |
| | tight | 212.1 (0.82x) | 114.1 (0.71x) | 640.5 (0.81x) | 1,229 (0.88x) | 54.61 MB |
| | R8 | 189.1 (0.73x) | 89.2 (0.55x) | 531.7 (0.67x) | 1,124 (0.81x) | 46.27 MB |
| | R16 | 174.6 (0.67x) | 74.7 (0.46x) | 487.9 (0.61x) | 1,078 (0.77x) | 41.48 MB |

(GPT-2: 11 queries; `cnn<e>` and R64 need 9-35 G encoded entries, too long for this laptop.)
With 4 threads (`results/perf_plans_laptop_cnn_thr4.jsonl`) the ratios are the same: LeNet-5
verifier 0.62x / 0.60x (cnn16 / cnn18), column check 0.45x / 0.43x, prover open 0.52x / 0.50x;
VGG-16 verifier 0.59x (cnn17, R64) and 0.61x (R16), column check 0.40x / 0.42x / 0.45x, prover
open 0.44x / 0.43x / 0.47x (paper: LeNet-5 6.50 ms, VGG-16 42.4 ms).
Derive and the products are the same under every policy (they do not depend on the
commitment); the column check falls with the columns opened and the multiproofs, and the
prover's open with the `sum N k t` of its column products.

### Table 4 (interactive, lambda = 128)

Proof sizes are exact.  Verifier times are estimates for the L40S node's client (EPYC, 8
threads): its stored stage medians (paper code) scaled by laptop ratios -- the Phase 1 ratios
of each stage (above: derive, products, columns) and the policy's column-check ratio (this
section) -- to be confirmed with `bench.py --policy` on the cluster.

| Row | Competitor | Ours, report | Ours, Phase 1 only (est.) | Ours, Phase 1 + policy | Flips with |
|---|---:|---:|---:|---:|---|
| zkCNN LeNet-5 proof | 71.3 kB | 130.9 kB (1.84x worse) | 130.9 kB | 65.0 / 63.2 / 62.2 kB (cnn16 / 17 / 18; 1.10-1.15x better) | cnn16, cnn17, cnn18 (not under Fiat--Shamir: 76.6 kB at best) |
| zkCNN LeNet-5 verifier | 5.8 ms | 6.96 ms (1.2x worse) | ~3.7 ms | ~2.3 ms (cnn16), ~2.2 ms (cnn18) | Phase 1 already; ~2.5x better with cnn16-18 (policy alone ~4.2 ms) |
| zkCNN VGG-16 verifier | 59.3 ms | 63.4 ms (1.07x worse) | ~24 ms | ~13.9 ms (cnn17), ~14.4 ms (R16) | Phase 1 already; ~4.3x better with cnn17 (policy alone ~37 ms) |
| zkGPT (GPT-2) verifier | 0.35 s | 0.468 s (1.34x worse) | ~0.18 s | ~0.118 s (R16), ~0.129 s (R8) | Phase 1 already; ~3.0x better with R16 (policy alone ~0.30 s) |
| DeepProve GPT-2-64 proof | 21.7 MB | 62.1 MB (2.86x worse) | 62.1 MB | 41.5 (R16) / 36.7 (R64) / 34.0 MB (cnn18) | no: the claims alone are 21.8 MB (still 1.57x worse at best) |

### On the cluster

The stored roots are frozen, so the policies and a same-hardware baseline go to a new root; each
policy's cells are named `..._pol<name>` next to the baseline's (from the repository root).  As for
the report's root (`../../README.md`), every job of a root must run on one kind of machine:
`bench.sbatch` alone asks for any GPU (`--gres=gpu:1`), and the first job's machine owns the root
(`PLATFORM.json`), so pin the partition and the L40S, and keep the jobs off t-806 (the L40S node
with a Xeon: `bench.py` refuses a second CPU model in one root), e.g. with an `sbatch` wrapper on
`PATH` that adds `--exclude=t-806`:

```bash
export PVI_PLATFORM=l40s_plans SBATCH_PARTITION=killable SBATCH_GRES=gpu:l40s:1 && mkdir -p logs/$PVI_PLATFORM
B=code/experiments/4_defence_benchmark/slurm/bench.sbatch
for m in mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar; do
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B cnn --model $m                         # the baseline (paper)
  for p in tight cnn16 cnn17 cnn18 R8 R16 R64; do
    sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B cnn --model $m --policy $p
  done
done
sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model gpt2 --seq 64 512 --modes C:int,C:fs --lams 128
for p in tight R8 R16 R64 cnn16; do
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model gpt2 --seq 64 512 --policy $p --lams 128 --llm-tampers 10
done
for p in tight R8 R16; do
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model llama2-7b --seq 1 64 --policy $p --lams 128
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model qwen3-4b --seq 8 --policy $p --lams 128
done
```

and, interactively on a GPU node, the GPU tests and the interleaved timing with the trained
CNNs (`cd code && export PYTHONPATH=$PWD/src`):

```bash
python -m pytest tests/test_plans.py tests/test_fast_verifier.py tests/test_gpu_verifier.py -q
python experiments/6_improvements/perf.py --model lenet5 --modes C --queries 30 --threads 8 \
    --policy paper tight cnn16 cnn17 cnn18 R8 R16 R64
python experiments/6_improvements/perf.py --model vgg16 --modes C --queries 30 --threads 8 \
    --policy paper tight cnn16 cnn17 cnn18 R8 R16 R64
python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C --queries 15 --threads 8 \
    --policy paper tight R8 R16 R64 cnn16 --verifier-device cuda
```

## The compact wire encoding of the proof: `wire.py`

`run_query(..., wire=True)` (opt-in; `bench.py --wire --tag ..._wire`, `perf.py --wire`) sends the
claims in the format `PVC3` of `pvi.fullcheck.claimcodec` (a frame of reference per weight op,
optionally centred on its row means, with Rice-coded exceptions; the module docstring gives the
byte layout) and `u` and the opened columns 31-bit packed.  The prover encodes where it keeps the
claims (its GPU; the host in lean mode), the verifier decodes before it checks anything, and
Fiat--Shamir absorbs the encoded bytes.  The decoder knows every claim's shape before it allocates
it: `N` from the graph, `M` from the query's shape (`Verifier.claim_columns`: the graph run on
meta tensors, once per query shape), so a malformed header costs it no more memory than the honest
claims.  The column check takes the received int32 rows as they are.  Parameters, challenges and
the soundness bound are the default flow's; without `wire` nothing of the codec runs
(`tests/test_wire.py`).

```bash
cd code && export PYTHONPATH=$PWD/src
python experiments/6_improvements/wire.py --mnist <dir holding MNIST/raw> --out results/wire_laptop.json \
    lenet5 mlp_mnist vgg16 resnet18_cifar gpt2:64 gpt2:512 llama2-7b:64:1,2 qwen3-4b:8:1,2,4,8
python experiments/6_improvements/wire.py --out results/wire_laptop.json --summary   # --threads 1,8: the paper's verifier
```

Results on this laptop (`results/wire_laptop.json`; the CNNs are random-init, queried on MNIST
digits, padded to 32x32 on 3 channels for the CIFAR shapes; the decoders are the benchmark's
random-weight builds).  Bytes are exact; timings are medians of 11 interleaved repetitions (7 for
GPT-2 T512 and Llama-2-7B) on a machine other jobs were loading.

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

Per block (with the real embedding and LM head): Qwen3-4B 568,119 B (18.49 bits/claim; builds of
1, 2, 4 and 8 blocks), so all 36 blocks take 36,079,104 -> 20,813,273 B (1.73x); Llama-2-7B
6,410,657 B per block, all 32 blocks 349,303,808 -> 205,462,357 B (1.70x).  Centring on row means
saves 0.3-0.9 bits per claim on GPT-2, VGG-16 and ResNet-18 and nothing on Qwen3-4B.

Whole proofs, lambda = 128 (mode C of the decoders sized exactly, without a commitment):

| Model | Mode | Default (B) | Wire (B) | Smaller |
|---|---|---:|---:|---:|
| LeNet-5 | C | 130,840 | 116,001 | 1.13x |
| LeNet-5 | Kpre | 26,072 | 13,577 | 1.92x |
| VGG-16 | C | 3,433,116 | 2,862,416 | 1.20x |
| ResNet-18 CIFAR | C | 4,629,920 | 3,383,413 | 1.37x |
| MLP MNIST | C | 269,972 | 261,391 | 1.03x |
| GPT-2 T64 | C | 62,071,240 | 50,747,048 | 1.22x |
| GPT-2 T64 | Kpre | 21,827,908 | 11,676,676 | 1.87x |
| GPT-2 T512 | C | 213,459,400 | 130,311,023 | 1.64x |

In mode C the opened columns and `u` (uniform field elements) shrink by exactly 1/32; the claims
by 1.7-1.96x.  The decoder costs (`verify_decode`; the encoder is the prover's `prove_encode`):

| Model | Mode | Threads | Verifier (ms) | verify_decode (ms) | Share |
|---|---|---:|---:|---:|---:|
| LeNet-5 | Kpre | 1 | 1.6 | 0.6 | 35% |
| VGG-16 | Kpre | 1 | 9.0 | 3.0 | 33% |
| VGG-16 | C | 1 | 34.2 | 5.4 | 16% |
| GPT-2 T64 | Kpre | 1 / 4 | 100.2 / 70.1 | 33.1 / 22.4 | 33% / 32% |
| GPT-2 T512 | Kpre | 1 / 4 | 1954 / 1031 | 368 / 161 | 19% / 16% |
| Llama-2-7B T64, 2 blocks | Kpre | 1 / 4 | 168.9 / 89.3 | 52.0 / 31.9 | 31% / 36% |
| Qwen3-4B T8, 8 blocks | Kpre | 1 / 4 | 56.6 / 61.9 | 12.4 / 12.1 | 22% / 19% |

(The CNN rows are a re-run at commit `9dc0c86`, after the column check stopped widening the
received rows, on a less loaded machine; the decoder rows, in mode Kpre, are the earlier run's.)
Decoding runs at 145-160 M claims/s on one thread for the decoders (GPT-2 T64: 37.5 ms against
7.2 ms to read the same claims as int32 and widen them) and 230-240 M claims/s on 4 threads;
small models pay about 0.4-1.3 ms of fixed cost.  Encoding on this CPU runs at 28-39 M claims/s
(a GPU prover encodes with torch on the device, giving the same bytes).

## Commitment plans and the wire encoding together

The two options compose: `run_query(..., wire=True)` under a plan (`commit_graph(..., policy=)`,
`params_for(..., plan=)`, a verifier with `groups`).  The claims and `u` travel as without a
plan; the opened columns of every tree -- an op's own, or a group's, whose members' rows sit side
by side as its leaves hash them -- travel as one run of 31-bit field elements, tree after tree
in the order of the column challenges, and the verifier unpacks them per tree into the int32
rows `[t_g, sum N]` its checks (batched, deferred, int8 and streaming) already take.  Fiat--Shamir
absorbs the plan's statement (each op's codeword length, each group's tag, root, length, `t_g` and
member order) and then the encoded bytes (`claims/PVC3`, `u/F31`).  The parameters are the plan's
(`r` unchanged, `t_g` = the members' largest exact `t`), so the bound is the plan's:
`eps <= sum_l [p^-r + prod_{i<t_g(l)} (k_l-1-i)/(n_l-i)] <= 2L 2^-beta = 2^-lambda` (interactive;
`2^-(lambda+64)` per transcript under Fiat--Shamir), 129.8-140.7 bits at lambda = 128 for these
rows.  `tests/test_plans.py` runs its plan tests with and without wire: honest queries accepted
by `run_query` and the streaming verifier (their bytes pinned to `encode()` and `field_size()` of
the plan's columns), every forgery the wire can carry rejected at its check (also in the deferred
and int8 forms and on a GPU client), malformed or altered claim, `u` and group-column messages
rejected where the default flow rejects them, and the transcript.

```bash
cd code && export PYTHONPATH=$PWD/src
P="paper tight cnn16 cnn17 cnn18 R8 R16 R64"
python experiments/6_improvements/plan_bytes.py --run lenet5 --policies $P --wire off on --queries 5 \
    --out results/combined_run_lenet5.csv                                  # run_query, exact bytes
python experiments/6_improvements/plan_bytes.py --run vgg16 --policies paper tight cnn16 cnn17 cnn18 R16 R64 \
    --wire off on --queries 3 --out results/combined_run_vgg16.csv
python experiments/6_improvements/plan_bytes.py --run gpt2:64:12 --policies paper R16 --wire off on --queries 3 \
    --out results/combined_run_gpt2.csv
python experiments/6_improvements/plan_bytes.py --run gpt2:64:12 --claims-only --policies $P --wire off on \
    --queries 3 --out results/combined_model_gpt2.csv                      # measured claims, the byte model
python experiments/6_improvements/plan_bytes.py --run llama2-7b:64:1,2 --claims-only --policies $P --wire off on \
    --queries 3 --out results/combined_model_llama.csv                     # + the whole model, extrapolated
python experiments/6_improvements/plan_bytes.py --run qwen3-4b:8:4,8 --claims-only --policies paper tight R16 R64 \
    --wire off on --queries 3 --out results/combined_model_qwen.csv
python experiments/6_improvements/perf.py --model lenet5 --random-init --modes C --queries 31 --threads 1 \
    --policy paper cnn16 cnn17 --wire off on > results/perf_combined_laptop_lenet5.jsonl
```

### Proof bytes (lambda = 128)

Interactive / Fiat--Shamir.  CNN rows and GPT-2 R16 are `run_query` medians (the CNNs random-init,
as above); every measured `u` and column count equals the byte model with the measured size of
the encoded claims (`analytic.proof_bytes(..., wire_claims=)`), and the GPT-2 R16 totals come
within 1 kB of it (the multiproofs).  The other decoder cells are that model on the measured
claims of the benchmark's random-weight builds (`--claims-only`); for Llama-2-7B (32 blocks) and
Qwen3-4B (36) the encoded claims are extrapolated linearly from builds of 1 and 2 (4 and 8)
blocks, which is exact for the default claims (checked).

| Row | Default | Wire | Plan | Plan + wire | Smaller |
|---|---:|---:|---:|---:|---:|
| LeNet-5 C | 130.6 / 172.2 kB | 115.6 / 157.1 | cnn16 65.0 / 82.7 | cnn16 **51.4 / 68.6** | 2.54x / 2.51x |
| | | | cnn17 63.2 / 79.6 | cnn17 **49.7 / 65.6** | 2.63x / 2.62x |
| | | | cnn18 62.2 / 76.6 | cnn18 48.7 / 62.5 | 2.68x / 2.75x |
| VGG-16 C | 3,432.7 / 4,461.1 kB | 2,872.0 / 3,869.4 | cnn17 2,339.1 / 2,871.0 | cnn17 1,806.9 / 2,321.7 | 1.90x / 1.92x |
| | | | cnn18 2,263.0 / 2,753.9 | cnn18 **1,733.0 / 2,208.3** | 1.98x / 2.02x |
| | | | R64 2,265.3 / 2,747.8 | R64 1,735.6 / 2,203.4 | 1.98x / 2.02x |
| GPT-2, 64 tokens, C | 62.07 / 80.69 MB | 50.73 / 68.77 | R16 41.48 / 50.25 | R16 **30.76 / 39.25** | 2.02x / 2.06x |
| | | | R64 36.68 / 42.79 | R64 **26.11 / 32.02** | 2.38x / 2.52x |
| | | | cnn18 34.04 / 39.57 | cnn18 23.55 / 28.91 | 2.64x / 2.79x |
| Qwen3-4B, 8 tokens, Kpre | 36.08 MB | **20.78** | (no columns) | 20.78 | 1.74x |
| Llama-2-7B, 64 tokens, C | 761.7 / 951.1 MB | 605.1 / 788.6 | R16 566.2 / 664.2 | R16 415.6 / 510.5 | 1.83x / 1.86x |
| | | | R64 501.0 / 571.6 | R64 **352.4 / 420.8** | 2.16x / 2.26x |
| | | | cnn18 505.3 / 577.9 | cnn18 356.6 / 426.9 | 2.14x / 2.23x |

The two gains add up because they shrink different parts: the plans cut the opened columns and
the paths, the wire encoding the claims (1.7-1.9x) and `u` and the columns by 1/32.  What is
left under the best plan is mostly the encoded claims and `u`: GPT-2 R64 + wire is 11.72 MB of
claims, 2.62 MB of `u`, 11.72 MB of columns and 0.05 MB of paths; Llama-2-7B R64 + wire 205.5,
22.8, 124.1 and 0.04 MB.  (Qwen3-4B: the 20.78 MB extrapolate the median of 3 prompts from 4 and 8
blocks; the earlier `wire.py` figure, 20.81 MB, extrapolates one prompt from the same builds.)

### Verifier and prover time on this laptop (`perf.py --policy ... --wire off on`)

Every (policy, wire) variant committed and its queries interleaved in one process (the order
rotating), random-init models, 1 thread, all queries accepted; ratios to `paper` without wire in
the same run (`results/perf_combined_laptop_*.jsonl`).

| Model | Policy | Wire | Verifier (ms) | verify_decode (ms) | Column check (ms) | Prover (ms) | Proof |
|---|---|---|---:|---:|---:|---:|---:|
| LeNet-5 | paper | no | 4.28 | | 2.94 | 3.4 | 131.1 kB |
| | paper | yes | 4.93 (1.15x) | 0.68 | 2.86 | 6.1 (1.83x) | 116.2 kB |
| | cnn16 | no | 2.70 (0.63x) | | 1.32 | 2.5 (0.76x) | 65.1 kB |
| | cnn16 | yes | 3.27 (0.76x) | 0.59 | 1.30 | 5.0 (1.48x) | 51.5 kB |
| | cnn17 | no | 2.61 (0.61x) | | 1.26 | 2.5 (0.74x) | 63.1 kB |
| | cnn17 | yes | 3.25 (0.76x) | 0.59 | 1.27 | 4.9 (1.47x) | 49.6 kB |
| VGG-16 | paper | no | 34.40 | | 24.01 | 137.6 | 3,432.6 kB |
| | paper | yes | 37.47 (1.09x) | 3.97 | 23.08 | 161.3 (1.17x) | 2,872.2 kB |
| | cnn17 | no | 20.59 (0.60x) | | 10.44 | 96.0 (0.70x) | 2,339.1 kB |
| | cnn17 | yes | 23.90 (0.69x) | 3.31 | 10.26 | 117.0 (0.85x) | 1,806.0 kB |
| | R64 | no | 20.09 (0.58x) | | 9.85 | 94.1 (0.68x) | 2,265.7 kB |
| | R64 | yes | 23.09 (0.67x) | 3.40 | 9.63 | 115.5 (0.84x) | 1,735.3 kB |
| GPT-2, 12 blocks, 64 tokens | paper | no | 259.3 | | 157.4 | 1,394.7 | 62.07 MB |
| | paper | yes | 296.9 (1.15x) | 57.0 | 138.9 | 1,632.2 (1.17x) | 50.69 MB |
| | R16 | no | 178.8 (0.69x) | | 78.3 | 1,093.2 (0.78x) | 41.48 MB |
| | R16 | yes | 210.0 (0.81x) | 45.2 | 61.6 | 1,301.1 (0.93x) | 30.71 MB |

(LeNet-5: 31 queries, VGG-16: 15, GPT-2: 11; commit `9dc0c86`.)  With 4 threads LeNet-5 gives the
same picture (paper 5.55 ms; cnn16 + wire 0.74x, cnn17 + wire 0.74x).  Derive and the products are
the same with and without wire, and the column check is the plan's, on the int32 rows as they
arrive (the default flow's casts its int64 columns to those rows for the leaves, so with wire it is
a little faster: 138.9 against 157.4 ms on GPT-2): the wire's cost is `verify_decode` (the claims,
then `u` and the columns: 0.6-0.7 ms on LeNet-5, 3.3-4.0 ms on VGG-16, 45-57 ms on GPT-2, of which
the claims are about 35 ms), so every plan + wire verifier stays below the default one
(0.67-0.81x).  The prover pays its encoder (`prove_encode` on this CPU: 2.5-2.8 ms on LeNet-5,
21-23 ms on VGG-16, 0.20-0.25 s on GPT-2; a GPU prover encodes on its device).  An A/B of the
column path alone, in one process with the queries alternating (GPT-2, 11 queries): the earlier
path (the rows widened to int64 columns, then cast back for the leaves) took 315.8 ms of verifier
under `paper` + wire and 214.4 ms under R16 + wire, this one 285.7 and 205.6 ms (0.90x, 0.96x);
LeNet-5 and VGG-16 are within 1-4%.

Qwen3-4B with 8 of its 36 blocks, 8 tokens, mode Kpre, lambda = 40: run in separate processes
(`results/perf_combined_laptop_qwen_separate.jsonl`, back to back, twice) the verifier takes 45.3
ms without wire and 55.7 ms with it (1.23x; derive 34.1 / 34.1 ms, products 11.0 / 11.2 ms,
`verify_decode` 10.4 ms) for a 1.74x smaller proof.  Interleaved in one process with a verifier
without wire (`results/perf_combined_laptop_qwen_interleaved.jsonl`) the products of the wire
verifier read 33.5 ms instead: that is an artefact of alternating the two allocation patterns in
one Windows process (two interleaved wire verifiers give 11.2 ms each, two without wire 10.6
ms), not work the wire adds.

### Table 4 (lambda = 128)

| Row | Competitor | Ours, report | Plan + wire (this section) | Flips? |
|---|---:|---:|---|---|
| zkCNN LeNet-5 proof | 71.3 kB | 130.9 kB | 51.4 / 49.7 / 48.7 kB interactive (cnn16 / 17 / 18); 68.6 / 65.6 / 62.5 kB Fiat--Shamir | yes, 1.39-1.46x smaller interactive and now also under Fiat--Shamir (1.04-1.14x) |
| Maverick Qwen3-4B proof (Kpre) | 36.08 MB | 36.08 MB (equal) | 20.78 MB (wire) | yes, 1.74x smaller |
| DeepProve GPT-2-64 proof | 21.7 MB | 62.1 MB | 30.76 MB (R16, measured), 26.11 MB (R64), 23.55 MB (cnn18) | no: 1.09x larger at best; claims and `u` alone are 14.3 MB |
| zkCNN LeNet-5 verifier | 5.8 ms | 6.96 ms | ~2.8 ms (cnn16 + wire; est.) | yes (as with the plan alone), ~2x better |
| zkCNN VGG-16 verifier | 59.3 ms | 63.4 ms | ~16 ms (cnn17 + wire; est.) | yes, ~3.7x better |
| zkGPT (GPT-2) verifier | 0.35 s | 0.468 s | ~0.14 s (R16 + wire; est.) | yes, ~2.5x better |

The verifier estimates scale the plan's estimates of the section above (Phase 1 + policy on the
L40S client: ~2.3 ms, ~13.9 ms, ~0.118 s) by this laptop's plan + wire / plan ratio (1.21x,
1.16x, 1.17x); the EPYC's decode with 8 threads is not measured yet.

### On the cluster

The combination goes to a new root next to its baseline; `--policy <p> --wire --tag _wire` cells
are named `..._wire_pol<p>` (from the repository root; the partition and the L40S pinned, and the
jobs kept off t-806, as for `l40s_plans` above):

```bash
export PVI_PLATFORM=l40s_combined SBATCH_PARTITION=killable SBATCH_GRES=gpu:l40s:1 && mkdir -p logs/$PVI_PLATFORM
B=code/experiments/4_defence_benchmark/slurm/bench.sbatch
for m in lenet5 vgg16; do
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B cnn --model $m                             # the baseline (paper)
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B cnn --model $m --wire --tag _wire
  for p in cnn16 cnn17 cnn18 R64; do
    sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B cnn --model $m --policy $p --wire --tag _wire
  done
done
sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model gpt2 --seq 64 --modes C:int,C:fs --lams 128
for p in R16 R64 cnn18; do
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model gpt2 --seq 64 --policy $p --wire --tag _wire --lams 128
done
sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model qwen3-4b --seq 8 --builds full --modes Kpre:int --lams 40
sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model qwen3-4b --seq 8 --builds full --modes Kpre:int --lams 40 \
    --wire --tag _wire
for p in R16 R64; do
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model llama2-7b --seq 64 --policy $p --wire --tag _wire --lams 128
done
python code/experiments/5_comparison/aggregate.py --platform l40s_combined
```

and interactively on a GPU node (`cd code && export PYTHONPATH=$PWD/src`):

```bash
python -m pytest tests/test_plans.py tests/test_wire.py tests/test_gpu_verifier.py -q -p no:cacheprovider -o addopts=""
python experiments/6_improvements/perf.py --model lenet5 --modes C --queries 30 --threads 8 \
    --policy paper cnn16 cnn17 cnn18 --wire off on
python experiments/6_improvements/perf.py --model vgg16 --modes C --queries 30 --threads 8 \
    --policy paper cnn17 cnn18 R64 --wire off on
python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C --queries 15 --threads 8 \
    --policy paper R16 R64 --wire off on --verifier-device cuda
python experiments/6_improvements/plan_bytes.py --run gpt2:64:12 --policies R64 cnn18 --wire off on --queries 3 \
    --out results/combined_run_gpt2_cluster.csv
```

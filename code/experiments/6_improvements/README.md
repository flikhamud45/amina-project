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
distinct random honest queries each through `run_query`) has claim, `u` and column bytes equal to
the byte model (`analytic.proof_bytes`), and its median total within 0.7% of the model's (the
Merkle paths depend on the random indices).  Measured medians, interactive / Fiat--Shamir:

| Model | paper | tight | cnn16 | cnn17 | cnn18 | R8 | R16 | R64 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| MLP (MNIST) | 270.0 / 391.6 kB | 214.6 / 310.1 | 106.8 / 153.2 | 97.2 / 140.6 | 90.8 / 130.9 | 170.1 / 244.3 | 141.6 / 202.4 | 112.8 / 161.9 |
| LeNet-5 | 130.6 / 172.8 kB | 111.0 / 142.6 | 65.0 / 82.4 | 63.2 / 79.6 | 62.2 / 76.4 | 98.1 / 126.2 | 90.4 / 115.7 | 82.7 / 106.4 |
| VGG-11 | 2,201 / 2,907 kB | 1,791 / 2,309 | 1,500 / 1,888 | 1,420 / 1,768 | 1,365 / 1,693 | 1,613 / 2,051 | 1,507 / 1,888 | 1,383 / 1,706 |
| VGG-16 | 3,433 / 4,461 kB | 2,833 / 3,593 | 2,454 / 3,040 | 2,339 / 2,871 | 2,263 / 2,753 | 2,584 / 3,229 | 2,436 / 3,000 | 2,266 / 2,748 |
| ResNet-18 (CIFAR) | 4,630 / 5,583 kB | 4,036 / 4,716 | 3,642 / 4,155 | 3,550 / 4,016 | 3,491 / 3,926 | 3,807 / 4,391 | 3,676 / 4,189 | 3,521 / 3,964 |

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

On this laptop's CPU (NTT on the CPU, shared machine) the commits took 90-150 ns per entry:
LeNet-5 cnn16 2.1 s, cnn18 8.8 s; VGG-16 paper 10 s, cnn16 32 s, cnn17 81 s, cnn18 178 s, R16 49 s,
R64 227 s (`commit_s` in `results/plan_bytes_run_cnn.csv`).  At about 9 ns per entry: VGG-16 cnn17
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
| zkCNN LeNet-5 proof | 71.3 kB | 130.9 kB (1.84x worse) | 130.9 kB | 65.0 / 63.2 / 62.2 kB (cnn16 / 17 / 18; 1.10-1.15x better) | cnn16, cnn17, cnn18 (not under Fiat--Shamir: 76.4 kB at best) |
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
as above, on 5 (LeNet-5) and 3 (VGG-16) distinct random queries); every measured `u` and column
count equals the byte model with the measured size of the encoded claims
(`analytic.proof_bytes(..., wire_claims=)`), and the GPT-2 R16 totals come within 1 kB of it (the
multiproofs).  The other decoder cells are that model on the measured
claims of the benchmark's random-weight builds (`--claims-only`); for Llama-2-7B (32 blocks) and
Qwen3-4B (36) the encoded claims are extrapolated linearly from builds of 1 and 2 (4 and 8)
blocks, which is exact for the default claims (checked).

| Row | Default | Wire | Plan | Plan + wire | Smaller |
|---|---:|---:|---:|---:|---:|
| LeNet-5 C | 130.6 / 172.8 kB | 115.7 / 156.6 | cnn16 65.0 / 82.4 | cnn16 **51.5 / 68.6** | 2.54x / 2.52x |
| | | | cnn17 63.2 / 79.6 | cnn17 **49.7 / 65.5** | 2.63x / 2.64x |
| | | | cnn18 62.2 / 76.4 | cnn18 48.8 / 62.5 | 2.68x / 2.76x |
| VGG-16 C | 3,432.7 / 4,460.3 kB | 2,870.9 / 3,867.8 | cnn17 2,339.1 / 2,871.0 | cnn17 1,805.9 / 2,321.7 | 1.90x / 1.92x |
| | | | cnn18 2,263.0 / 2,753.1 | cnn18 **1,731.8 / 2,207.0** | 1.98x / 2.02x |
| | | | R64 2,265.3 / 2,747.2 | R64 1,734.7 / 2,201.4 | 1.98x / 2.03x |
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
| zkCNN LeNet-5 proof | 71.3 kB | 130.9 kB | 51.5 / 49.7 / 48.8 kB interactive (cnn16 / 17 / 18); 68.6 / 65.5 / 62.5 kB Fiat--Shamir (medians of 5 distinct queries) | yes, 1.38-1.46x smaller interactive and now also under Fiat--Shamir (1.04-1.14x) |
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

## The col layout and fusion: the policies with the suffix `c`

(The `c` policies have since also looked the embedding tables up: the next section.  This section's
decoder numbers are those of the col layout alone, commit `894eb6f`; its CNN numbers still hold.)

A commitment plan can now commit a linear op's matrix transposed (`pvi.fullcheck.plans`, opt-in by
policy name: `tightc`, `cnn<e>c` and `R<R>c` take the codeword lengths of `tight`, `cnn<e>` and
`R<R>` and add per-op layouts; `commit_graph(..., policy=)`, `bench.py --policy`, `perf.py
--policy`, `plan_bytes.py`).  The default and the Phase-2 policies are unchanged: the Phase-2
reviewers' fingerprint of the default flow is byte-identical, a same-process A/B against `44d96bb`
(`paper`, `tight`, `cnn12`, `R8`, `R16` on the tiny MLP, LeNet-5, GPT-2, Llama and OPT; with and
without wire, batched and streaming, interactive and Fiat--Shamir, honest and forged queries) gives
identical Merkle roots, groups, parameters, verdicts, labels, bytes and transcripts, and every row
policy's plan of every decoder configuration (and of its 1- and 2-block builds) is Phase 2's.

* **The col layout.**  The rows of `A^T` (one per input coordinate, of length `N`) are
  Reed--Solomon encoded at the policy's length for message length `N`, and a Merkle leaf holds a
  column of length `k`.  The prover sends no `u` for it.  The verifier draws `chi'` (`r x M` over
  the claims' columns; the identity when `M <= r`: the LM head, `M = 1`, and every op of a
  one-token prompt), computes `z = chi' Z^T` (the Freivalds left side's limb products, int8 GEMMs
  on a GPU, on the range-checked claims) and `w = chi' [X ; 1]^T` (exact float64 blocks, as
  `u^T [X ; 1]`) itself, batched per shape class, and checks `w . E'[:, c] == Enc(z)[c]` at the
  opened columns on the row check's batched path (`codeword_at`, one limb product per shape
  class, one copy back).  It opens `t k` field elements instead of `r k + t N`.
* **Fusion.**  Linear ops that read one tensor with one row length (q/k/v, also with GQA's
  narrower k and v; gate/up) may share ONE col matrix `[A_1 ; A_2 ; ...]^T`: one `chi'`, one `w`
  and one set of `t` columns of length `k` for all of them (message length `sum N`).
* **The choice.**  The ops of a set (one input and row length) take all rows, each op transposed
  alone, or one col matrix, and sets of one shape (a decoder's blocks) take the same.  The plan
  takes the options of least expected bytes of the whole plan at lambda = 128, interactive --
  `u`, the opened columns and the multiproofs of the shared trees the matrices end up in (the
  groups' dynamic programming) -- by descent from the base policy's plan, so a `c` plan never
  costs more than its base policy's in that model (tested on the CNNs and decoders).  Charging
  each matrix a tree of its own instead transposed LeNet-5's classifier under `cnn16c` and lost 4
  kB: its `t` then left the CNN's single tree.  Convolutions and embedding tables stay in rows.
  Every decoder takes q/k/v and gate/up fused, fc1, the attention output and the head transposed,
  fc2/down and the embeddings in rows; the CNNs transpose a classifier or two, if anything.
* **Message flow.**  claims -> `chi` and `chi'` (both right after the claims) -> `u` of the row
  ops -> the columns of every tree (one round) -> the openings.  A col check only needs its `chi'`
  and columns drawn after the claims; drawing every tree's columns in one round, after `u`, lets
  row and col matrices of one codeword length share a tree and a multiproof (GPT-2 `R16c`: the
  fc2s in rows and the transposed fc1s, all at `n = 2^16`) and keeps the report's three prover
  messages.  Under Fiat--Shamir a col matrix's columns then also depend on `u`, which the prover
  picks: grinding, which the 64 grinding bits bound as for every other challenge.
* **Soundness.**  Per committed matrix `p^-r + prod_{i<t} (m-1-i)/(n-i)` with `m = k` (row) or the
  stacked rows `sum N` (col), and no Freivalds term for `chi' = I`: a wrong claim of a col matrix's
  ops (on an input derived from correct claims) survives `chi'` with probability `<= p^-r`, and
  then `Enc(z_i)` and `w_i^T E' = Enc(A w_i)` are distinct codewords of dimension `m`, which the `t`
  distinct columns (bound by the root, drawn independently of `chi'`) all miss with at most the
  product.  The union bound over the matrices (at most the `L` ops) stays `<= 2L 2^-beta =
  2^-lambda`.  `soundness_bits` takes one check per matrix (`plan.shapes()`,
  `columns=plan.matrix_columns(params.group_columns)`); `tests/test_plans.py` checks `>= lambda`
  and each matrix's own budget for the five CNNs and every decoder configuration, every policy
  with and without `c`, lambda 40/80/128, interactive and Fiat--Shamir (129.8-140.7 bits at 128).
  Fiat--Shamir absorbs every col matrix (its ops in order, rows, message and codeword length, tag)
  with the groups' records; `chi'` is keyed after the claims and the columns after `u` (tested).
* **Prover and setup.**  The prover recomputes an opened col column from its device-resident
  weights (`TransposedCommitment.columns_at`: `sum_i W_i^T V_i` and the biases' row), nothing is
  stored; setup streams the column digests through `max_host_bytes`, so the head's `A^T` at
  `n = R next_pow2(vocab)` needs no buffer of its encoding (GPT-2 `R64c`: `n = 2^22`, committed
  on this laptop).
* **Wire.**  The claims codec is unchanged; `u` travels for the row ops only, and a col matrix's
  opened columns in its tree's 31-bit run like every member's.  Every byte count is what is sent
  (tested against `encode()`, `field_size()` and the byte model).
* **Verifiers.**  The batched verifier, its deferred and int8 forms (tested on the CPU with
  `_defer` and `int8_ok` forced), the streaming verifier (which computes `w` and `z` during derive
  and queues their code checks after it) and a GPU client
  (`test_a_gpu_client_and_prover_give_the_cpu_verdicts`, CUDA only) give the same verdicts and
  labels.  A verifier refuses a key whose col matrix stacks ops that are not linear, read
  different tensors or do not have its shape, or that checks an op twice or not at all.

`tests/test_plans.py` runs its plan tests with `tightc`, `cnn12c` and `R8c` too, on LeNet-5, a wide
MLP, GPT-2, Llama, OPT (`embed_dim`) and Qwen (GQA, q/k norm): honest queries accepted and sized
as the model says; every forgery rejected at its check, preferring a col victim -- claims (a
col op's at `columns_code`), a kernel shift of `[X ; 1]^T` (passes the code check for every
`chi'`: `columns_merkle`), a wrong or misplaced column, a misplaced member, out-of-field entries,
forged paths, the tampered one-column head -- in all the forms, with and without wire; malformed
wire messages; the transcript.  `tests/test_layouts.py` holds the rest (the transposed commitment,
mixed groups, the planner, `chi' = I`, `w` and `z` against `reference.column_operands`).

```bash
cd code && export PYTHONPATH=$PWD/src
C="cnn16 cnn16c cnn17 cnn17c cnn18 cnn18c R16 R16c R64 R64c"
python experiments/6_improvements/plan_bytes.py --out results/plan_bytes_col.csv       # the byte model, every model
python experiments/6_improvements/plan_bytes.py --run mlp_mnist lenet5 vgg16 --policies paper $C --wire off on \
    --queries 5 --out results/col_run_cnn.csv                                  # run_query, exact bytes
python experiments/6_improvements/plan_bytes.py --run gpt2:64,512:12 --policies paper R16 R16c cnn16c tightc \
    --wire off on --queries 3 --out results/col_run_gpt2.csv                   # (and cnn17c, cnn18c, R64c)
python experiments/6_improvements/plan_bytes.py --run llama2-7b:64:1,2:1024 qwen3-4b:8:1,2:1024 opt-125m:2048:1,2 \
    --policies tightc R8c cnn16c --queries 2 --out results/col_run_decoders.csv  # the model, validated
python experiments/6_improvements/plan_bytes.py --run llama2-7b:1,64:1,2 --claims-only --policies paper $C \
    --wire off on --queries 3 --out results/col_model_llama.csv                # measured claims + the model
python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C --queries 11 --threads 1 --device cpu \
    --policy paper R16 R16c cnn16c tightc --wire off on > results/perf_col_laptop_gpt2.jsonl
```

(and `--claims-only` for `gpt2:64,512:12`, `llama2-7b:2048:1,2`, `qwen3-4b:8,64:4,8`,
`opt-{125m,1.3b,2.7b,6.7b}:2048:1,2`: `results/col_model_*.csv`).

### Proof bytes (lambda = 128)

Interactive / Fiat--Shamir.  The CNN and GPT-2 cells are `run_query` medians (the CNNs random-init
on 5 distinct random queries, GPT-2 on 3 random prompts), every claim, `u` and column count equal
to the byte model's (with wire, on the measured size of the encoded claims) and every total within
0.7% of it (the multiproofs); cells marked (model) are the byte model on the measured claims.

| Model | Policy | Plan | Plan + wire | `c` plan | `c` plan + wire |
|---|---|---:|---:|---:|---:|
| MLP (MNIST) | cnn18 | 90.8 / 130.9 kB | 87.1 / 126.3 | 88.0 / 126.2 | 84.5 / 121.4 |
| LeNet-5 | cnn17 | 63.2 / 79.6 kB | 49.7 / 65.5 | 63.0 / 79.8 | 49.6 / 66.0 |
| | cnn18 | 62.2 / 76.4 | 48.8 / 62.5 | 62.0 / 76.2 | **48.5 / 62.5** |
| | R64 | 82.7 / 106.4 | 69.0 / 92.2 | 82.2 / 104.8 | 68.6 / 91.5 |
| VGG-16 | cnn17 | 2,339.1 / 2,871.0 kB | 1,805.9 / 2,321.7 | 2,318.9 / 2,842.1 | 1,786.3 / 2,293.2 |
| | cnn18 | 2,263.0 / 2,753.3 | 1,731.9 / 2,207.2 | 2,242.8 / 2,725.2 | **1,712.5 / 2,179.9** |
| | R64 | 2,265.8 / 2,747.7 | 1,734.7 / 2,201.5 | 2,264.5 / 2,744.8 | 1,733.9 / 2,199.6 |
| GPT-2, 64 tokens | paper | 62.07 / 80.69 MB | 50.73 / 68.77 | | |
| | R16 | 41.48 / 50.25 | 30.76 / 39.25 | 28.61 / 31.70 | 18.29 / 21.29 |
| | R64 | 36.68 / 42.79 (model) | 26.11 / 32.02 (model) | 27.28 / 29.66 | 17.00 / 19.31 |
| | cnn16 | 37.01 / 43.30 (model) | 26.43 / 32.53 (model) | 28.46 / 31.46 | 18.15 / 21.06 |
| | cnn17 | 35.29 / 40.97 (model) | 24.76 / 30.27 (model) | 27.75 / 30.38 | 17.46 / 20.00 |
| | cnn18 | 34.04 / 39.57 (model) | 23.55 / 28.91 (model) | 27.03 / 29.34 | **16.76 / 19.00** |
| | tight | 54.61 / 69.45 (model) | | 32.15 / 36.82 | 21.72 / 26.25 |
| GPT-2, 512 tokens | paper | 213.46 / 232.08 MB | 130.31 / 148.35 | | |
| | R16 | 192.86 / 201.64 | 110.33 / 118.83 | 179.99 / 183.09 | 97.86 / 100.86 |
| | R64 | 188.07 / 194.17 (model) | 105.68 / 111.60 (model) | 178.67 / 181.05 | 96.58 / 98.88 |
| | cnn18 | 185.42 / 190.96 (model) | 103.12 / 108.49 (model) | 178.42 / 180.73 | **96.33 / 98.58** |

On the CNNs the col layout finds little (a classifier or two transposed; 0-3%, and `R16c` on LeNet-5
and VGG-16 is `R16`): their ops' `N` is small against their `k`.  On GPT-2 it removes 35% of `u`
(everything but the embeddings' and the fc2s') and 64-72% of the opened columns.  What is left of
GPT-2 at 64 tokens under `cnn18c` + wire: 11.72 MB of claims, 1.71 MB of `u` (the token
embedding's 0.97 MB, the fc2s' 0.71 MB), 3.29 MB of columns and 0.04 MB of paths.

The decoders too large to run here: the byte model on the whole model with the claims measured on
builds of 1 and 2 blocks (Qwen3-4B: 4 and 8) and extrapolated (exact without wire: checked).  The
model is validated by `run_query` on builds of 1 and 2 blocks under `tightc`, `R8c` and `cnn16c`
(`results/col_run_decoders.csv`: Llama-2-7B at 64 tokens and Qwen3-4B at 8 with a 1,024-token
vocabulary, OPT-125M at 2,048 with its own; interactive and Fiat--Shamir): every build's claim, `u`
and column bytes equal the model's, the totals are within 1.1 kB of it (the multiproofs), and the
1- and 2-block totals extrapolated over the blocks are within 0.02% of the model of the whole
model.  MB:

| Model, prompt | paper | R64 | cnn18 | R16c | R64c | cnn18c | R64 + wire | R64c + wire | cnn18c + wire |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Llama-2-7B, 1 | 418.0 / 607.4 | 157.2 / 227.9 | 161.6 / 234.2 | 81.1 / 115.9 | 60.4 / 85.7 | 79.3 / 113.5 | 150.3 / 218.7 | **56.5 / 81.0** | 74.8 / 107.9 |
| Llama-2-7B, 64 | 761.7 / 951.1 | 501.0 / 571.6 | 505.3 / 577.9 | 424.8 / 459.6 | 404.2 / 429.5 | 423.1 / 457.3 | 352.4 / 420.8 | **258.6 / 283.1** | 276.9 / 310.0 |
| Llama-2-7B, 2048 | 11,586 / 11,776 | 11,325 / 11,396 | 11,330 / 11,402 | 11,249 / 11,284 | 11,229 / 11,254 | 11,248 / 11,282 | 6,728 / 6,797 | **6,635 / 6,659** | 6,653 / 6,686 |
| Qwen3-4B, 8 | 410.3 / 582.0 | 165.3 / 224.8 | 168.5 / 230.0 | 96.0 / 123.0 | 81.9 / 101.9 | 93.1 / 119.0 | 146.0 / 203.6 | **65.2 / 84.6** | 76.1 / 101.1 |
| Qwen3-4B, 64 | 658.6 / 830.3 | 413.6 / 473.1 | 416.8 / 478.3 | 344.3 / 371.3 | 330.2 / 350.2 | 341.4 / 367.3 | 288.4 / 346.0 | **207.7 / 227.0** | 218.5 / 243.5 |
| OPT-125M, 2048 | 732.5 / 751.2 | 707.1 / 713.3 | 704.5 / 709.6 | 699.1 / 702.2 | 697.7 / 700.1 | 697.5 / 699.8 | 382.5 / 388.5 | 373.4 / 375.7 | **373.2 / 375.4** |
| OPT-1.3B, 2048 | 3,807 / 3,875 | 3,709 / 3,731 | 3,709 / 3,732 | 3,689 / 3,703 | 3,681 / 3,692 | 3,684 / 3,695 | 2,094 / 2,116 | **2,067 / 2,078** | 2,070 / 2,081 |
| OPT-2.7B, 2048 | 6,320 / 6,429 | 6,165 / 6,204 | 6,168 / 6,207 | 6,132 / 6,153 | 6,119 / 6,136 | 6,126 / 6,146 | 3,491 / 3,529 | **3,447 / 3,463** | 3,454 / 3,473 |
| OPT-6.7B, 2048 | 10,101 / 10,271 | 9,857 / 9,912 | 9,876 / 9,944 | 9,812 / 9,849 | 9,791 / 9,818 | 9,810 / 9,846 | 5,758 / 5,811 | **5,694 / 5,720** | 5,712 / 5,747 |

At one token (Llama-2-7B) and eight (Qwen3-4B) the proof is mostly opened columns, which the col
layout cuts 2-2.7x against the same codeword lengths; at 2,048 tokens it is 96-99.5% claims, so the
plans move it by 1-3% and the wire encoding by 1.7-1.9x.  What is left of Llama-2-7B at one token
under `R64c` + wire: 3.34 MB of claims, 7.44 MB of `u` (the downs', in rows, and the embedding's),
45.63 MB of opened columns (21-23 columns of 4,096 entries per matrix and block) and 0.07 MB of
paths.

### Setup (the one-time commitment)

Encoded field entries (`analytic.setup_size`), G, and the time at the L40S node's ~9 ns per entry:

| Model | R16 | R16c | R64 | R64c | cnn18 | cnn18c |
|---|---:|---:|---:|---:|---:|---:|
| GPT-2 | 2.84 | 2.97 | 10.75 | 11.28 (~1.7 min) | 35.32 | 10.28 (~1.5 min) |
| Llama-2-7B | 118 | 149 | 468 | 593 (~89 min) | 366 | 140 (~21 min) |
| Qwen3-4B | 103 | 104 | 405 | 408 (~61 min) | 331 | 99 (~15 min) |

Transposed, the head's codeword runs along the vocabulary but it has only `d` rows, so the `cnn<e>c`
policies cut the setup 3.4x (GPT-2) and 2.6x (Llama-2-7B) where `cnn<e>` encoded 50,257 (32,000) rows
at `2^e`; `R<R>c` costs 1-27% more than `R<R>` (the fused q/k/v and gate/up round their stacked
rows up to a power of two: Llama-2-7B's 12,288 and 22,016 to 16,384 and 32,768).  On this laptop
(4 threads, other jobs running) the commits took 125-180 ns per entry: GPT-2 `R16` 356 s, `R16c`
465 s, `cnn16c` 382 s, `cnn17c` 832 s, `cnn18c` 1,788 s, `R64c` 2,003 s (`commit_s` in
`results/col_run_gpt2*.csv`); the one-thread timings below give the A/B.

### Timing on this laptop (`perf.py --policy ... --wire off on`)

Every (policy, wire) variant committed and its queries interleaved in one process (the order
rotating), one thread, the machine otherwise idle, all queries accepted; ratios to `paper` without
wire (`results/perf_col_laptop_*.jsonl`, commit `7b228c4`).  GPT-2 with all 12 blocks, 64 tokens,
11 random prompts:

| Policy | Wire | Verifier (ms) | products | columns | verify_decode | Prover (ms) | prove_fold | prove_open | Setup (s) | Proof |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper | no | 251.7 | 31.7 | 155.0 | | 1,347 | 298.5 | 768.9 | 150 | 62.07 MB |
| | yes | 289.0 (1.15x) | 32.3 | 136.8 | 54.8 | 1,578 (1.17x) | 299.7 | 760.8 | | 50.68 MB |
| R16 | no | 170.1 (0.68x) | 31.6 | 73.6 | | 1,053 (0.78x) | 297.9 | 475.1 | 483 | 41.48 MB |
| | yes | 204.8 (0.81x) | 32.4 | 60.9 | 43.8 | 1,252 (0.93x) | 298.1 | 471.2 | | 30.71 MB |
| R16c | no | 155.7 (0.62x) | 43.1 | 47.1 | | 918 (0.68x) | 123.2 | 514.4 | 630 | 28.61 MB |
| | yes | 186.0 (0.74x) | 42.9 | 42.8 | 35.2 | 1,091 (0.81x) | 127.4 | 517.8 | | 18.24 MB |
| cnn16c | no | 157.4 (0.63x) | 43.1 | 48.7 | | 1,076 (0.80x) | 122.0 | 670.6 | 571 | 28.46 MB |
| | yes | 188.5 (0.75x) | 43.4 | 45.1 | 35.5 | 1,239 (0.92x) | 122.7 | 666.2 | | 18.10 MB |
| tightc | no | 173.3 (0.69x) | 43.6 | 64.5 | | 1,024 (0.76x) | 123.2 | 620.5 | 185 | 32.15 MB |
| | yes | 202.3 (0.80x) | 43.5 | 57.9 | 36.8 | 1,190 (0.88x) | 123.6 | 613.5 | | 21.67 MB |

(derive: 64-68 ms in every variant.)  `R16c` against `R16`, the same codeword rates: the verifier
0.92x without wire and 0.91x with it -- the column check 0.64x / 0.70x (61 of the 75 ops open
columns of 768-769 entries, one set per fused q/k/v, instead of 768-50,257, and hash that much
less), `verify_decode` 0.80x (`u` of the row ops only), while the products take 1.36x: `z = chi'
Z^T` and `w` replace Freivalds for those ops at the same multiply-adds, but their limb sums come
out `N` wide instead of `M`.  The
prover 0.87x: `prove_fold` 0.41x (`u` for the embeddings and the fc2s only) and `prove_open` 1.08x
(the head's opened columns are `W^T V` over its 50,257 outputs).  Setup 1.30x for 5% more encoded
entries: the transposed head's codewords are `2^20` long (`2^14` in rows), and an NTT costs
`n log n`.  Under `tightc` (the report's codeword lengths) the verifier takes 0.69x / 0.80x and the
prover 0.76x / 0.88x of `paper`'s for a 1.93x / 2.86x smaller proof.

LeNet-5 (31 random queries) and VGG-16 (15), random-init: the `c` plans transpose one or two
classifiers and time as their base policies (LeNet-5 `cnn18c` 2.81 ms against `cnn18` 2.77 ms,
`cnn17c` 2.81 against 2.86; VGG-16 `cnn17c` 20.69 ms against `cnn17` 20.81; `paper` 4.44 and
33.70 ms); LeNet-5's `prove_fold` takes 0.8x.

### Table 4 (lambda = 128)

| Row | Competitor | Ours, report | Plan + wire (Phase 2) | `c` plan + wire (this section) | Flips? |
|---|---:|---:|---:|---:|---|
| DeepProve GPT-2-64 proof | 21.7 MB | 62.1 MB | 23.55 / 28.91 MB (cnn18, model) | **16.76 / 19.00 MB** (cnn18c), 17.00 / 19.31 (R64c), 17.46 / 20.00 (cnn17c), 18.29 / 21.29 (R16c): measured | yes: 1.29x smaller interactive, 1.14x under Fiat--Shamir |
| ZKTorch Llama-2-7B-1 proof | 22.85 MB | 418.0 MB | 150.3 / 218.7 MB (R64) | 56.5 / 81.0 MB (R64c) | no: 2.47x larger; the columns alone are 45 MB |
| zkCNN LeNet-5 proof | 71.3 kB | 130.9 kB | 48.8 / 62.5 kB (cnn18) | 48.5 / 62.5 kB (cnn18c) | yes, as before (1.47x / 1.14x smaller) |
| zkGPT (GPT-2) verifier | 0.35 s | 0.468 s | ~0.14 s (R16 + wire; est.) | ~0.13 s (R16c + wire; est.) | yes, as before |

Without the wire encoding GPT-2's claims alone (21.83 MB) exceed DeepProve's proof.  The verifier
estimate scales Phase 2's (the L40S client's stored stage medians by this laptop's ratios) by this
laptop's `R16c` + wire / `R16` + wire ratio, 0.91x; the cluster runs below confirm it.

### On the cluster

A new root next to its baseline (the partition and the L40S pinned, and the jobs kept off t-806, as
for `l40s_plans` above); `--policy <p>c --wire --tag _wire` cells are named `..._wire_pol<p>c`:

```bash
export PVI_PLATFORM=l40s_col SBATCH_PARTITION=killable SBATCH_GRES=gpu:l40s:1 && mkdir -p logs/$PVI_PLATFORM
B=code/experiments/4_defence_benchmark/slurm/bench.sbatch
for m in lenet5 vgg16; do
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B cnn --model $m                             # the baseline (paper)
  for p in cnn17c cnn18c; do
    sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B cnn --model $m --policy $p --wire --tag _wire
  done
done
sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model gpt2 --seq 64 512 --modes C:int,C:fs --lams 128
for p in R16c cnn17c cnn18c R64c; do
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model gpt2 --seq 64 512 --policy $p --wire --tag _wire --lams 128 \
      --llm-tampers 10
done
for p in R16c cnn18c R64c; do
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model llama2-7b --seq 1 64 --policy $p --wire --tag _wire --lams 128
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model qwen3-4b --seq 8 64 --policy $p --wire --tag _wire --lams 128
done
python code/experiments/5_comparison/aggregate.py --platform l40s_col
```

and interactively on a GPU node (`cd code && export PYTHONPATH=$PWD/src`), the GPU tests (a GPU
client and prover under `R8c`, among the plan tests) and the interleaved timing:

```bash
python -m pytest tests/test_plans.py tests/test_layouts.py tests/test_gpu_verifier.py -q -p no:cacheprovider -o addopts=""
python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C --queries 15 --threads 8 \
    --policy paper R16 R16c cnn17c cnn18c --wire off on --verifier-device cuda
python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C --queries 15 --threads 8 \
    --policy paper R16 R16c cnn18c --wire off on --verifier-device cpu
```

## Embedding lookups and last-block pruning

Two more opt-in options.  The default (`paper`, no wire, no pruning) is unchanged: the Phase-2
reviewers' default fingerprint is byte-identical, and a same-process A/B against `44d96bb` (`paper`,
`tight`, `cnn12`, `R8`, `R16` on the tiny MLP, LeNet-5, GPT-2, Llama and OPT; with and without wire,
batched and streaming, interactive and Fiat--Shamir, honest and forged queries) gives identical
Merkle roots, groups, parameters, verdicts, labels, bytes and transcripts; every row policy's plan of
every decoder configuration and build is Phase 2's.  The `c` policies now also look up the
embedding tables; on the CNNs, which have none, they are the col-layout commit's (`894eb6f`) bit for
bit (same-process A/B of `tightc`, `cnn12c`, `R8c`, `cnn16c` on the MLP and LeNet-5, and the plans of
the five CNNs under all seven `c` policies).

* **Lookup tables (mode C, every `c` policy).**  The table `W` `[d, V]` of an embedding op without a
  bias (the tokens', and a learned position table: every benchmark decoder's; a tree over `W`'s rows
  does not bind a bias, so an embedding with one keeps its row layout) is committed as a Merkle tree
  over its `V` rows (`commitment.TableCommitment`: leaf `j` = SHA-256 of `pvi/row`, the tag, `j` and
  row `j`'s `d` int8 bytes; the leaves past `V`, up to a power of two, in another domain).  It is in
  no group and has no code.  Its claims are the looked-up rows, one column of `d` int8 values per
  position, sent in the claims message (one byte each: without wire as int8 tensors,
  `pipeline.wire_rows`, which the streaming verifier uploads as they are and widens on its device;
  with wire as the int8 bytes of `claimcodec.pack_rows`, not in `PVC3`), with ONE multiproof over
  the distinct ids (`Prover.open_tables`).  The verifier checks, before it draws any challenge: in
  derive, that the claims are int8 (so a leaf's bytes are the claim, one to one) and every id names
  a row of the table (`range_or_shape`); then equal rows for equal ids (`lookup_consistency`) and
  the multiproof (`lookup_merkle`).  No `u`, no columns, no Freivalds.  The LM head is an op of its
  own (the benchmark's decoders draw it separately, as an untied model; for a tied one the owner
  commits the head once more), and keeps its encoded (row or col) commitment.
* **Why always.**  A lookup drops the table's `u` (`4 r V` bytes: 1.0 MB for GPT-2's tokens) and its
  opened columns (`4 t d` per tree), and its claims take a byte each (int32: four; `PVC3`: about one),
  for one multiproof of at most one path per position -- less than the three bytes per claim it
  saves on int32 claims for every benchmark decoder (`d >= 512`), and less than `u` alone with wire.
* **Soundness.**  The lookup check has no challenge and no chance: the claim of a bias-free
  embedding at id `j` is row `j` of `W`, and accepting a claim other than the committed rows at the
  ids means other bytes for a leaf the root binds, a SHA-256 collision -- the assumption every
  Merkle tree here already makes, not a term of the statistical bound.  So a table adds no term
  (`plan.shapes()` and `soundness_bits` leave it out) and the other matrices keep their budgets (`L`
  stays the number of weight ops in `beta`, a conservative count).  Fiat--Shamir absorbs each
  table's tag, root, `d` and `V` with the statement, and each multiproof (its length, then its
  hashes) after the claims and before `chi` (with wire, the `rows/I8` bytes after `claims/PVC3`).
* **Modes K and Kpre** (`Verifier(lookups=True)`, `bench.py --lookups`, `perf.py --lookups on`): the
  verifier computes the embedding ops from its own weights (the rows at the ids, clamped to the
  table -- a query with any other is rejected -- plus the bias of an embedding with one), the prover
  sends no claims for them, Kpre precomputes no `chi` for them, and Fiat--Shamir absorbs the list of
  those ops.  In mode C the tables do that (`lookups=True` there is refused).
* **Last-block pruning** (`build_decoder(..., prune_last=True)`, `bench.py --prune-last`, `perf.py
  --prune-last on`, `plan_bytes.py --prune-last`).  Only the last position reaches the next-token
  logits, so the last block computes q, the attention output and its projection, the residuals and
  the MLP at that position only; k and v stay at every position, which its one query row attends
  to.  `_attention` / `_attention_heads` take the queries of the last `Tq <= T` positions (the last
  rows of the causal mask: for one row, all ones), RoPE takes the position offset `T - Tq`, and the
  builder calibrates the pruned block on every position (`_Builder.last(..., calibrate_all=True)`),
  so the pruned graph has the unpruned graph's weights and multipliers and gives the same logits,
  `torch.equal` (tested: the tiny decoders and one-block GPT-2, OPT-350M (`embed_dim`), Llama-2-7B
  and Qwen3-4B (GQA, q/k norm) at 1, 2 and 9 tokens).  Both parties run the pruned graph: the
  protocol needs nothing new, the ops' claims have one column (`chi' = I` for a col matrix), and a
  plan sees the last block's q as a set of its own (it reads the last position) with k and v fused.
  `analytic.decoder_shapes` / `decoder_claim_columns(..., prune_last=True)` give its shapes, and a
  build of 1 and 2 pruned blocks extrapolates to the whole model (both hold the pruned block).
* **Verifiers.**  The batched verifier, its deferred and int8 forms (tested on the CPU with `_defer`
  and `int8_ok` forced), the streaming verifier (the ids come back with its one copy, queued before
  it; the rows are hashed on the host from the claims as received) and a GPU client take both;
  `reference.check_lookups` is the position-by-position specification the vectorised check is
  tested against.  A key whose table names no embedding op or one with a bias, has another shape, or
  leaves an op unchecked (or checked twice) is refused.
* **Setup.**  A table hashes its rows as they are: no NTT, `V` leaves (`analytic.setup_size` counts
  its tree and leaves, not encoded entries).  Against the col layout alone the `c` plans encode
  0.20 G entries fewer on GPT-2 (0.40 G under `cnn18c`: both tables at `2^18`), 0.54 G (1.07 G) on
  Llama-2-7B and 2.68 G (1.34 G) on Qwen3-4B, whose 151,936-row table was encoded at `4 x 2^18`.

`tests/test_lookups.py` (58 tests) holds what is particular to the lookups: the table's tree (its
leaves, padding, multiproof; a row moved or changed fails); every `c` plan of GPT-2, OPT-350M,
Llama-2-7B and Qwen3-4B, pruned or not, looks every embedding table up and keeps the head encoded;
honest queries with a token three times are accepted and every forgery is rejected at its check --
a row changed at all positions of its token or replaced by another token's row, and a changed
position-table row (`lookup_merkle`), one position of a repeated token changed
(`lookup_consistency`), an entry of 128 (`range_or_shape`; with wire the prover cannot encode it),
a forged, short, long or non-bytes path, a missing proof and swapped proofs (`lookup_merkle`) -- on
GPT-2, OPT, Llama and Qwen shapes, pruned and not, interactive and Fiat--Shamir, with and without
wire, by `run_query` and the streaming verifier alike, and in the deferred, int8 and deferred+int8
forms; the vectorised check against `reference.check_lookups`; an id outside the table rejected
as a malformed query (tables and a verifier's own rows); malformed int8 rows on the wire
(`range_or_shape`), a changed row on the wire (`lookup_merkle`); the transcript (table records
with the statement, each multiproof after the claims and before `chi`, every table field changing
the challenges); the refused keys; modes K and Kpre with `lookups` (no embedding claims sent,
`precompute` without them, a wrong embedding caught at the next op, the `lookups` record, the
malformed `PVC3` of the other claims); the byte model (`expected_lookup_nodes` against sampling, the
tables' claims and exact multiproofs, the setup's trees); and the experiment scripts' options.
`tests/test_pruning.py` (20 tests): the pruned logits `torch.equal` to the unpruned on the tiny
decoders and on GPT-2, OPT-350M, Llama-2-7B and Qwen3-4B blocks at 1, 2 and 9 tokens, with the same
weights and the pruned shapes and claim columns of `analytic`; the attention of the last `Tq`
queries equal to the last rows of the whole attention and to `reference.attention` (int32 and int64
paths, GQA stacked and in groups), RoPE at an offset, and a pruned block's plan (q alone, k and v
fused, `chi' = I` for a transposed q).  `tests/test_plans.py` runs its plan tests with the lookup
tables (the `c` policies) and on pruned GPT-2, Llama, OPT and Qwen (honest queries and their bytes
-- the tables' claims a byte each, their multiproofs exact -- in C, and in K and Kpre with and
without `lookups`; every column forgery on pruned OPT and Qwen; the GPU forms on pruned GPT-2; the
byte model on pruned GPT-2 and Qwen; builds of 1 and 2 pruned blocks get the whole model's matrices
and trees).

```bash
cd code && export PYTHONPATH=$PWD/src
A="paper tightc cnn16c cnn17c cnn18c R8c R16c R64c"
python experiments/6_improvements/plan_bytes.py --policies $A --kpre off on --out results/lookup_bytes.csv  # the model
python experiments/6_improvements/plan_bytes.py --policies $A --kpre off on --prune-last --out results/lookup_bytes_prune.csv
python experiments/6_improvements/plan_bytes.py --run gpt2:64,512:12 --policies paper tightc R16c cnn16c --wire off on \
    --queries 3 --out results/lookup_run_gpt2.csv                          # run_query (and --prune-last: _prune)
python experiments/6_improvements/plan_bytes.py --run gpt2:64,512:12 --claims-only --policies paper tightc cnn16c cnn17c \
    cnn18c R16c R64c --kpre off on --wire off on --queries 3 --out results/lookup_model_gpt2.csv
python experiments/6_improvements/plan_bytes.py --run llama2-7b:64:1,2:1024 qwen3-4b:8:1,2:1024 opt-125m:2048:1,2 \
    --policies tightc R8c --kpre off on --wire off on --queries 2 --prune-last --out results/lookup_run_decoders.csv
python experiments/6_improvements/plan_bytes.py --run qwen3-4b:8:4,8 --policies --kpre off on --wire off on --queries 3 \
    --out results/lookup_kpre_qwen.csv                                     # Kpre, run_query (and --prune-last)
python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C --queries 11 --threads 1 --device cpu \
    --policy paper R16 R16c --wire off on --prune-last off on > results/perf_lookup_laptop_gpt2.jsonl
```

(and `--claims-only --kpre off on --wire off on` with and without `--prune-last` for
`llama2-7b:1,64:1,2`, `llama2-7b:2048:1,2`, `qwen3-4b:8,64:4,8`, `opt-{125m,1.3b,2.7b,6.7b}:2048:1,2`:
`results/lookup_model_*.csv`; `results/lookup_run_gpt2_cnn18c*.csv` and `lookup_run_gpt2_R64c_prune.csv`
measure those two policies.)

### Proof bytes (lambda = 128)

The CNNs have no embedding table and no decoder block: their `c` plans and proofs are the col
layout's (the table of the section above).  GPT-2 with all 12 blocks, interactive / Fiat--Shamir, MB;
`run_query` medians of 3 random prompts (every claim, `u` and column count equal to the byte
model's, totals within 7 kB of it: the multiproofs, 2.6 kB for the `c` plans) unless marked (model:
the byte model on the measured claims of the same prompts); the col-layout columns are the section
above's, of `894eb6f`:

| Prompt | Policy | col layout (`894eb6f`) | + lookups | + lookups, pruned | col layout + wire | + lookups + wire | + lookups + wire, pruned |
|---|---|---:|---:|---:|---:|---:|---:|
| 64 | paper | 62.07 / 80.69 | 62.07 / 80.69 | 60.72 / 79.34 | 50.73 / 68.77 | 50.73 / 68.77 | 50.01 / 68.06 |
| 64 | tightc | 32.15 / 36.82 | 30.45 / 34.54 | 29.30 / 33.49 | 21.72 / 26.25 | 20.36 / 24.32 | 19.84 / 23.89 |
| 64 | cnn16c | 28.46 / 31.46 | 26.78 / 29.19 | 25.49 / 27.93 | 18.15 / 21.06 | 16.80 / 19.14 | 16.14 / 18.51 |
| 64 | cnn17c | 27.75 / 30.38 | 26.08 / 28.12 (model) | 24.78 / 26.85 (model) | 17.46 / 20.00 | 16.12 / 18.10 (model) | 15.46 / 17.46 (model) |
| 64 | cnn18c | 27.03 / 29.34 | 25.50 / 27.26 | 24.20 / 25.98 | 16.76 / 19.00 | 15.56 / 17.27 | 14.89 / 16.62 |
| 64 | R16c | 28.61 / 31.70 | 26.89 / 29.39 | 25.65 / 28.19 | 18.29 / 21.29 | 16.91 / 19.33 | 16.30 / 18.76 |
| 64 | R64c | 27.28 / 29.66 | 25.57 / 27.35 (model) | 24.29 / 26.10 | 17.00 / 19.31 | 15.63 / 17.35 (model) | 14.98 / 16.73 |
| 512 | paper | 213.46 / 232.08 | 213.46 / 232.08 | 202.47 / 221.09 | 130.31 / 148.35 | 130.31 / 148.35 | 124.55 / 142.60 |
| 512 | tightc | - | 179.85 / 183.94 | 169.07 / 173.25 | - | 100.00 / 103.96 | 94.43 / 98.49 |
| 512 | cnn16c | - | 176.18 / 178.59 | 165.26 / 167.70 | - | 96.44 / 98.78 | 90.74 / 93.11 |
| 512 | cnn17c | - | 175.48 / 177.52 (model) | 164.55 / 166.62 (model) | - | 95.76 / 97.74 (model) | 90.05 / 92.06 (model) |
| 512 | cnn18c | 178.42 / 180.73 | 174.90 / 176.66 | 163.96 / 165.75 | 96.33 / 98.58 | 95.20 / 96.90 | 89.49 / 91.21 |
| 512 | R16c | 179.99 / 183.09 | 176.29 / 178.79 | 165.42 / 167.96 | 97.86 / 100.86 | 96.55 / 98.97 | 90.90 / 93.36 |
| 512 | R64c | 178.67 / 181.05 | 174.97 / 176.75 (model) | 164.06 / 165.87 | 96.58 / 98.88 | 95.26 / 96.99 (model) | 89.58 / 91.33 |

At 64 tokens the lookups take 1.5-1.7 MB off each `c` plan without wire -- `u` of the token table
(1.01 MB) and of the position table (0.02 MB), their opened columns (0.2-0.4 MB) and 3 bytes of each
of their 98,304 claims (0.29 MB), for 23 kB of multiproofs (the position table's 64 aligned rows
need 4 hashes) -- and 1.2-1.4 MB with wire (`PVC3` sent those claims at about a byte already).
Pruning takes the last block's q, o, fc1 and fc2 claims at 63 of the 64 positions (5,376 per
position: 1.35 MB as int32, 0.72 MB with wire) and opens one set of columns more (the pruned q is a
matrix of its own: +0.05-0.2 MB); at 512 tokens it takes 11.0 MB (5.8 MB with wire).  What is left of
GPT-2 at 64 tokens under `cnn18c` + wire, pruned: 10.99 MB of claims (the tables' 98,304 bytes among
them), 0.71 MB of `u` (the fc2s', in rows), 3.12 MB of opened columns and 0.06 MB of paths.

The decoders too large to run here: the byte model on the whole model with the claims measured on
builds of 1 and 2 blocks (Qwen3-4B: 4 and 8) and extrapolated (exact without wire: checked), the
tables' multiproofs for the prompt's tokens and positions.  The model is validated by `run_query`
on pruned builds of 1 and 2 blocks under `tightc` and `R8c`, and in mode Kpre with and without
`lookups` (`results/lookup_run_decoders.csv`: Llama-2-7B at 64 tokens and Qwen3-4B at 8 with a
1,024-token vocabulary, OPT-125M at 2,048 with its own; interactive and Fiat--Shamir, with and
without wire): every build's claim, `u` and column bytes equal the model's (the tables' claims and
multiproofs exactly), the totals are within 1.3 kB of it (the column multiproofs), and the 1- and
2-block claims, `u` and columns extrapolated over the blocks are the whole model's to the byte.
(The totals extrapolate within 0.43%: a pruned 1-block build lacks one tree of the unpruned blocks,
so its multiproof would be counted `L - 1` times; the model counts each tree once.)  MB,
interactive / Fiat--Shamir:

| Model, prompt | paper | paper, pruned | R64c | cnn18c | R64c + wire | cnn18c + wire | R64c + wire, pruned | cnn18c + wire, pruned |
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

With pruning at 2,048 tokens the claims shrink by the last block's `2047 (q + o + MLP)` columns:
OPT-125M -21.3 MB with wire (-5.7%), OPT-6.7B -88.8 MB (-1.6%), Llama-2-7B -187.7 MB (-2.8%); at one
token pruning changes no claim and opens one set of columns more (the pruned q is a matrix of its
own: Llama-2-7B `R64c` + wire 54.8 -> 55.2 MB), so it is for prompts.  The lookups take the
embeddings' `u` off: 0.62 MB of Llama-2-7B's 7.44 MB (what is left is the downs', in rows), 2.9 MB
of Qwen3-4B's.  What is left of Llama-2-7B at one token under `R64c` + wire: 3.34 MB of claims, 6.83
MB of `u` (the downs'), 44.55 MB of opened columns and 0.05 MB of paths -- against ZKTorch's 22.85 MB
still 2.4x larger.

Mode Kpre (the proof is the claims; the Maverick row): Qwen3-4B at 8 tokens measured by `run_query`
on builds of 4 and 8 blocks (full vocabulary) and extrapolated to 36 (`results/lookup_kpre_qwen*.csv`,
median of 3 prompts), and the table from the claims measured as above (Qwen3-4B's rows there with the
builds of the 64-token runs, calibrated on 32 tokens: 20.81 MB with wire where the 8-token builds give
20.78), MB:

| Model, prompt | Kpre | + wire | lookups | lookups + wire | pruned | pruned + wire | lookups, pruned | lookups, pruned + wire |
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

A verifier with `lookups` saves the embeddings' claims (Qwen3-4B at 8 tokens: 20,480 claims, 82 kB
as int32, 20 kB with wire); pruning saves the last block's `(T - 1) (q + o + gate + up + down)`
claims (7 x 28,672: 0.80 MB, 0.43 MB with wire).  Qwen3-4B, 8 tokens, Kpre: 36.08 MB (the report,
equal to Maverick's 36.08 MB) -> 35.19 MB with both (2.5% smaller) -> 20.33 MB with wire too (1.77x
smaller than Maverick).  GPT-2 at 64 tokens: 21.83 -> 20.08 MB, 10.90 MB with wire.

### Setup (the one-time commitment)

A lookup table is `V` leaves of `d` bytes, hashed once; the encoded entries of the `c` plans, G
(`analytic.setup_size`; the pruned graphs' within 2%), against the col layout alone (`894eb6f`):

| Model | tightc | R16c | R64c | cnn18c |
|---|---:|---:|---:|---:|
| GPT-2 | 0.90 -> 0.69 | 2.97 -> 2.77 | 11.28 -> 11.08 | 10.28 -> 9.87 |
| Llama-2-7B | 37.6 -> 37.0 | 148.7 -> 148.2 | 593.2 -> 592.7 | 139.6 -> 138.5 |
| Qwen3-4B | 28.0 -> 25.3 | 104.0 -> 101.3 | 408.0 -> 405.3 | 99.3 -> 98.0 |

On this laptop (4 threads, the machine otherwise idle) GPT-2 committed in 69-83 s (`paper`,
`tightc`), 264-312 s (`cnn16c`, `R16c`), 1,257-1,275 s (`cnn18c`) and 1,644 s (`R64c`, pruned):
`commit_s` in `results/lookup_run_gpt2*.csv`.

### Timing on this laptop

One thread, the machine otherwise idle, every query accepted; GPT-2 with all 12 blocks, 64 tokens, 11
random prompts, every variant committed and its queries interleaved in one process, the order
rotating (`perf.py --policy paper R16 R16c --wire off on --prune-last off on`,
`results/perf_lookup_laptop_gpt2.jsonl`); ratios to `paper` without wire:

| Policy | Pruned | Wire | Verifier (ms) | derive | products | columns | lookups | verify_decode | Prover (ms) | prove_fold | prove_open | Proof |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| paper | no | no | 269.4 | 72.2 | 33.5 | 163.7 | | | 1,491 | 373.5 | 817.6 | 62.07 MB |
| | no | yes | 307.1 (1.14x) | 71.3 | 34.7 | 144.6 | | 56.5 | 1,735 (1.16x) | 372.3 | 811.7 | 50.68 MB |
| | yes | no | 264.5 (0.98x) | 66.5 | 32.7 | 165.3 | | | 1,481 (0.99x) | 374.4 | 816.2 | 60.72 MB |
| | yes | yes | 300.4 (1.12x) | 65.5 | 33.4 | 144.6 | | 56.8 | 1,720 (1.15x) | 376.0 | 812.7 | 49.97 MB |
| R16 | no | no | 185.8 (0.69x) | 74.2 | 34.8 | 76.8 | | | 1,175 (0.79x) | 374.3 | 500.5 | 41.48 MB |
| | no | yes | 212.5 (0.79x) | 71.7 | 33.7 | 62.4 | | 44.7 | 1,384 (0.93x) | 373.4 | 507.1 | 30.71 MB |
| | yes | no | 178.9 (0.66x) | 66.1 | 33.1 | 79.6 | | | 1,156 (0.78x) | 370.6 | 499.6 | 40.12 MB |
| | yes | yes | 203.0 (0.75x) | 64.6 | 33.5 | 62.7 | | 42.3 | 1,357 (0.91x) | 369.5 | 500.1 | 29.99 MB |
| R16c | no | no | 159.6 (0.59x) | 72.8 | 45.7 | 39.5 | 1.6 | | 743 (0.50x) | 55.0 | 385.0 | 26.89 MB |
| | no | yes | 188.2 (0.70x) | 70.8 | 45.5 | 34.8 | 1.5 | 35.6 | 902 (0.61x) | 53.4 | 377.8 | 16.86 MB |
| | yes | no | 153.0 (0.57x) | 67.6 | 43.5 | 40.4 | 1.5 | | 729 (0.49x) | 57.1 | 382.6 | 25.65 MB |
| | yes | yes | 181.5 (0.67x) | 66.9 | 43.8 | 36.1 | 1.5 | 33.2 | 884 (0.59x) | 55.9 | 382.0 | 16.26 MB |

(prove_forward: 287-302 ms in every variant; setup at one thread: `paper` 157-161 s, `R16` 502-518 s,
`R16c` 640-648 s.)  The lookups against the col layout alone, in a same-process A/B of `R16c` at
`894eb6f` and at this commit (the old package imported under another name, the same graph and
prompts, 11 interleaved queries): the verifier 184.6 -> 154.5 ms (0.84x; with wire 218.6 -> 185.7 ms,
0.85x) -- the products 0.65x (67.3 -> 43.8 ms: the token table's Freivalds check, with its 50,257-wide
`u`, is gone), the columns 0.81x (its opened columns), and the lookup check itself 1.5-1.6 ms (the
consistency of 64 rows, 64 leaf hashes and the multiproofs) -- and the prover 1,016 -> 737 ms (0.73x;
`prove_fold` 0.34x, `prove_open` 0.68x), for 0.94x (0.92x with wire) of the proof.  Pruning: the
verifier 0.96x of the unpruned (derive 0.93x: the last block's attention, projections and MLP on one
position), the prover 0.98x (its forward pass 0.95x), for 0.95x of the proof (0.96x with wire).

Mode Kpre (`perf.py --modes Kpre --lookups off on --prune-last off on --wire off on`,
`results/perf_lookup_laptop_{gpt2,qwen}_kpre.jsonl`): on GPT-2 at 64 tokens the verifier takes 95.9 ms,
96.7 ms with lookups (1.01x: the gather of 128 rows costs what the precomputed embedding's check
did), 90.1 ms pruned (0.94x) and 90.8 ms with both (0.95x; with wire 127.4 -> 122.3 ms, 0.96x).  On
Qwen3-4B with 4 of its 36 blocks at 8 tokens (lambda = 40): 24.9 ms, 21.7 ms pruned (0.87x), 23.5 ms
with both, 28.9 ms with both and wire against 32.0 ms with wire alone (0.90x).  So in mode Kpre the
lookups save bytes, not time; pruning saves both.

### Table 4 (lambda = 128; Maverick: mode Kpre)

| Row | Competitor | Ours, report | Before this section | This section | Flips? |
|---|---:|---:|---|---|---|
| DeepProve GPT-2-64 proof | 21.7 MB | 62.1 MB | 16.76 / 19.00 MB (`cnn18c` + wire) | **14.89 / 16.62 MB** (`cnn18c` + wire, lookups, pruned), 14.98 / 16.73 (`R64c`), 16.30 / 18.76 (`R16c`): measured | yes, 1.46x smaller interactive and 1.31x under Fiat--Shamir (was 1.29x / 1.14x) |
| Maverick Qwen3-4B-8 proof | 36.08 MB | 36.08 MB (equal) | 20.78 MB (wire) | **35.19 MB** without wire, **20.33 MB** with it (lookups, pruned) | yes: now also without wire (2.5% smaller); 1.77x with wire |
| ZKTorch Llama-2-7B-1 proof | 22.85 MB | 418.0 MB | 56.5 / 81.0 MB (`R64c` + wire) | 54.8 / 78.5 MB (`R64c` + wire, lookups; pruning does nothing at one token) | no: 2.40x larger; the columns alone are 44.6 MB |
| zkGPT (GPT-2) verifier | 0.35 s | 0.468 s | ~0.13 s (`R16c` + wire; est.) | ~0.11 s (`R16c` + wire, lookups, pruned; est.) | yes, as before (~3.2x better) |

The verifier estimate scales the col section's (the L40S client's stored stage medians by this
laptop's ratios) by this laptop's 0.85x of the lookups (the same-process A/B against `894eb6f`) and
0.96x of pruning; the cluster runs below confirm it.

### On the cluster

A new root next to its baseline (the partition and the L40S pinned, and the jobs kept off t-806, as
for `l40s_plans` above); cells are named `..._wire_prune_pol<p>c` and `..._wire_prune_lookups`:

```bash
export PVI_PLATFORM=l40s_lookup SBATCH_PARTITION=killable SBATCH_GRES=gpu:l40s:1 && mkdir -p logs/$PVI_PLATFORM
B=code/experiments/4_defence_benchmark/slurm/bench.sbatch
sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model gpt2 --seq 64 512 --modes C:int,C:fs --lams 128   # the baseline
for p in R16c cnn18c R64c; do
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model gpt2 --seq 64 512 --policy $p --wire --tag _wire --prune-last \
      --lams 128 --llm-tampers 10
done
sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model qwen3-4b --seq 8 --builds full --modes Kpre:int --lams 40
sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model qwen3-4b --seq 8 --builds full --modes Kpre:int --lams 40 \
    --wire --tag _wire --prune-last --lookups
for p in R64c cnn18c; do
  sbatch -o logs/$PVI_PLATFORM/%x-%j.out $B llm --model llama2-7b --seq 1 64 --policy $p --wire --tag _wire --prune-last \
      --lams 128
done
python code/experiments/5_comparison/aggregate.py --platform l40s_lookup
```

and interactively on a GPU node (`cd code && export PYTHONPATH=$PWD/src`), the GPU tests (a GPU
client and prover under `R8c`, lookup tables included) and the interleaved timing:

```bash
python -m pytest tests/test_plans.py tests/test_lookups.py tests/test_pruning.py tests/test_gpu_verifier.py -q \
    -p no:cacheprovider -o addopts=""
python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C --queries 15 --threads 8 \
    --policy paper R16 R16c cnn18c --wire off on --prune-last off on --verifier-device cuda
python experiments/6_improvements/perf.py --model qwen3-4b --layers 8 --seq 8 --modes Kpre --lam 40 --queries 15 \
    --threads 8 --lookups off on --prune-last off on --wire off on --verifier-device cuda
```

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

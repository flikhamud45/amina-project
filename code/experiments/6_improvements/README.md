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

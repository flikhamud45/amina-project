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
| LeNet-5 | C | 128 | 1 | 8.08 | 4.84 | 1.67x |
| LeNet-5 | C | 128 | 4 | 10.51 | 5.56 | 1.89x |
| LeNet-5 | Kpre | 128 | 1 | 2.83 | 1.90 | 1.49x |
| VGG-16 | C | 128 | 1 | 83.9 | 32.0 | 2.63x |
| VGG-16 | C | 128 | 4 | 78.1 | 33.8 | 2.31x |
| VGG-16 | Kpre | 128 | 1 | 26.3 | 9.30 | 2.83x |
| GPT-2, 12 blocks, 64 tokens | C | 128 | 1 | 596.5 | 247.3 | 2.41x |
| GPT-2, 12 blocks, 64 tokens | C | 128 | 4 | 487.8 | 213.6 | 2.28x |
| GPT-2, 12 blocks, 64 tokens | Kpre | 128 | 1 | 229.1 | 90.7 | 2.53x |
| Qwen3-4B, 1 block, 8 tokens | Kpre | 40 | 1 | 42.5 | 7.69 | 5.52x |
| Qwen3-4B, 8 blocks, 8 tokens | Kpre | 40 | 1 | 154.6 | 42.2 | 3.66x |
| Qwen3-4B, 24 blocks, 8 tokens | Kpre | 40 | 1 | 426.2 | 127.5 | 3.34x |

Qwen3-4B with all 36 blocks (4.4 GB of int8 weights) does not fit this laptop's free
memory; 24 blocks do.  Per additional block (1 -> 24 blocks, full vocabulary): 16.68 ->
5.21 ms (3.20x), of which derive 7.01 -> 4.04 ms (1.74x) and the products 9.62 -> 1.18 ms
(8.15x); the vocabulary-1024 builds give the same slope (2 -> 8 blocks: 15.48 -> 4.87 ms,
3.18x).  Extrapolated to 36 blocks: 626 -> 190 ms (3.30x).

The per-stage ratios: derive (the cheap operations) 1.0x on LeNet-5, 1.5-1.7x elsewhere;
products 1.6-1.7x (LeNet-5), 2.8-3.3x (VGG-16), 3.5x (GPT-2 C) and 4.8x (GPT-2 Kpre),
8-13x (Qwen3-4B Kpre); columns (mode C) 1.9x (LeNet-5), 2.5-2.7x (VGG-16, GPT-2), whose
floor is now SHA-256 of the Merkle paths in `hashlib`.

With `PVI_MERKLE_PROCESSES=1` (off by default) a multi-threaded verifier checks the
multiproofs in worker processes (`results/ab_laptop_merkle_processes.csv`, 4 threads):
VGG-16 C 29.8 ms (2.70x; columns 22.2 -> 18.9 ms) and GPT-2 C 200.8 ms (2.49x; columns
137.2 -> 124.1 ms).

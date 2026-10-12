# D6a: the certified model and its commitment are reproducible bit for bit

Plan 2, item D6a (`SP2027_PLAN2.md`, Sec. 3.2). Script: `code/experiments/9_quality/digest_check.py` (commits 9cc0bd6,
e5d377c). Jobs 1008669 (s-004) and 1008670 (s-005), and the laptop.

## 1. Why

Reviewer A (A-W4, A-Q5) asks how the certified integer model is tied to the public fp16 checkpoint: can a third party
reproduce `C_M` from the Hugging Face weights and a published recipe? The defence assumes an honest `C_M` (as Anchuri et
al. do); for open weights the paper says anyone can recompute it. That claim needs the build to be deterministic
across machines, thread counts, operating systems and library versions.

## 2. What was tested

`digest_check.py --model <M> --threads <T> [--commit --commit-device <cpu|cuda>]` builds the certified integer OPT from
the Hugging Face checkpoint with the paper's recipe (`build_opt_from_hf`: norm gain 4, SmoothQuant alpha 0.55 through
the norm gains, smooth_pct 99.9, 99.99th-percentile clipping, 11-bit attention probabilities since F2a; calibration on
the WikiText-2 window of `common.load_windows`), then prints:

* the graph digest (every op, its wiring and every public constant: multipliers, norm gains, tables);
* a SHA-256 of every weight matrix and bias;
* with `--commit`, a SHA-256 over every Merkle root of the commitment (members, groups and lookup tables: under the
  grouped plan a member's own root is empty and its group's tree holds it; the first version of the script hashed only
  member roots and so hashed nothing, fixed in e5d377c).

## 3. Results

| Model | Machine | OS / torch | Threads | Commitment on | digest | weights | roots (6 trees) |
|---|---|---|---|---|---|---|---|
| OPT-125M | Xeon s-004 | Linux / 2.5.1 | 1, 8 | CPU | 46d23c80... | 6eae57cf... | 480463ab... |
| OPT-125M | Xeon s-005 | Linux / 2.5.1 | 1, 8 | CPU | 46d23c80... | 6eae57cf... | 480463ab... |
| OPT-125M | laptop (Intel 11th gen) | Windows / 2.2.2 | 1 | (not run: memory) | 46d23c80... | 6eae57cf... | |
| OPT-1.3B | Xeon s-004 | Linux / 2.5.1 | 1, 8 | GPU (2080 Ti) | 01354c3d... | 643cce56... | ff005dca... |
| OPT-1.3B | Xeon s-005 | Linux / 2.5.1 | 1, 8 | GPU (2080 Ti) | 01354c3d... | 643cce56... | ff005dca... |

Before F2a (8-bit probabilities) the laptop and s-005 had already agreed on OPT-125M at 1 and 8 threads (digest
acf09bcd..., weights d406f9e6...).

Every value agrees: the digests and weight hashes across two Xeon nodes, two thread counts, Windows against Linux
and torch 2.2 against 2.5; the commitment roots across the two Xeon nodes and two thread counts. (An earlier version
of this sentence also claimed a GPU-encoded against a CPU-encoded commitment. The table does not show that: each
model's commitment was encoded on one device only, OPT-125M's on the CPU and OPT-1.3B's on the GPU, and the laptop
did not compute roots. The certified recipe's re-check compares OPT-1.3B's roots encoded on the GPU (jobs 1012438-9)
with the same roots encoded on the CPU (jobs 1012744-5, CPU nodes s-002 and s-006); Sec. 3a.) The F2a cluster check also found the integer graph's outputs
equal on GPU and CPU, and its perplexities equal to the laptop's to every printed digit.

## 3a. The certified recipe changed (F2a decision, 11 Oct)

F2a's round 2 fixed the certified OPT recipe as smooth-max alpha 0.6 calibrated on four 2,048-token windows of the
train split (`real_weights.CERTIFIED_RECIPE`, `common.certified_calib`). The digests above are those of the earlier
recipe (`digest_check.py --recipe legacy`). The same check with the certified recipe runs as jobs 1012438 (s-004)
and 1012439 (s-005): [pending]. With the certified calibration (four 2,048-token windows instead of one 128-token
window) the single-threaded CPU build is far slower; OPT-125M at 1 thread had not finished after 2 h, so OPT-1.3B also
runs as CPU-only jobs with 8 and 4 threads and the commitment encoded on the CPU (1012744 on s-002, 1012745 on s-006,
24 h limit). Nothing in the builder changed, only its parameters and the calibration tokens, so
the determinism shown above should carry over; the jobs confirm it on the new graph.

**Results (12 Oct; git 1801809 for the GPU-node jobs, 207b798 for the CPU-node jobs, the builder identical in both).**
Four Linux nodes (all Xeon Silver 4114, torch 2.5.1), threads 1, 4 and 8, the commitment encoded on the CPU or on an
RTX 2080 Ti:

| Model | node | threads | commitment on | graph digest | weights | roots (6 trees) |
|---|---|---|---|---|---|---|
| OPT-125M | s-004 | 1, 8 | CPU | `16de86ead0bc063b...` | `811aeae0113f305d...` | `f880c6853322f58e...` |
| OPT-125M | s-005 | 1, 8 | CPU | same | same | same |
| OPT-1.3B | s-004 | 1, 8 | GPU | `d9daa01778a8ec3c...` | `be812feeaf1894eb...` | `a5f5433f315efa04...` |
| OPT-1.3B | s-005 | 1, 8 | GPU | same | same | same |
| OPT-1.3B | s-002 | 8, 4 | CPU | same | same | same |
| OPT-1.3B | s-006 | 8 | CPU | same | same | same |

Full values: OPT-125M digest `16de86ead0bc063b575d68fc16c6273c6934123e4d09ead1aaa3317f07893e2e`, weights `811aeae0113f305d5ba07783243201f7f4e23d9a77e541909e95cce38675a60b`, roots `f880c6853322f58ead3f8a50d044c338e683e4e1bb2a5541b2e0481f1f6b4815`; OPT-1.3B digest `d9daa01778a8ec3cfdb6f92816fea04c20a17c5d02454056788c9efa17cc24e8`, weights `be812feeaf1894eb18f529c1d12748b9f1f75c1c0f32296c152d54b872f52fd4`, roots
`a5f5433f315efa041ef75ec35b58d73dd79a43207c587c43cb8fd7c7909eba8c`.

* **Every value agrees**, across four machines, three thread counts and the two encoding devices: OPT-1.3B's roots
  encoded on the 2080 Ti equal those encoded on the CPU of two other nodes, the same-model comparison that Sec. 3
  lacked. The digests also equal those of F2c's zero-shot builds (job 1012440: `16de86ea...`, `d9daa017...`).
* Time on one thread of the Xeon: about 2-3 h for OPT-125M's build alone (four 2,048-token calibration windows); OPT-1.3B
  at 1 thread finished within the 6 h limit on both GPU nodes. With 8 threads OPT-1.3B builds and commits in about 55
  min on a CPU node.
* Windows / torch 2.2 (the laptop) is checked with the certified recipe for OPT-125M's digest and weights (no
  commitment: memory): [running].

## 4. The recipe, as published in the artifact

* Checkpoint: the Hugging Face repository (`facebook/opt-125m`, `facebook/opt-1.3b`, ...), `pytorch_model.bin` loaded
  with `weights_only=True`, converted to fp32.
* Calibration tokens (certified recipe): the first four 2,048-token windows of the WikiText-2 (raw) train split's first
  4,000,000 characters, tokenised with the checkpoint's tokenizer (`common.certified_calib`). Earlier recipe: the last
  128-token window of the test split (`common.load_windows`).
* Builder (certified): `real_weights.build_opt_from_hf(model, calib, **CERTIFIED_RECIPE)`, i.e. `norm_gain=4,
  smooth=0.6, smooth_pct=100, pct=100, pct_att=100` (`prob_max` = 2047 by default). Earlier: `smooth=0.55,
  smooth_pct=99.9, pct=99.99, pct_att=99.99`.
* Commitment: `protocol.commit_graph(graph, 4, policy="auto")`.

No change to the recipe was needed (the plan's fallback, float64 compensated sums and snapped scales, is not used).

## 5. Limitations

* The certified OPT models were tested; the Llama/Qwen3 builder (F2b-Q) is checked for determinism within a machine
  (rebuild gives the same digest) but not yet across machines; its fixed prefix's constants come from an fp32 forward
  pass, which a different BLAS could in principle round differently.
* The calibration window is part of the recipe: a different window gives a different (equally valid) model, so the
  window must be published with the digest.

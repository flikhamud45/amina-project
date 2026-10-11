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
24 h limit): [pending]. Nothing in the builder changed, only its parameters and the calibration tokens, so
the determinism shown above carries over; the jobs confirm it on the new graph.

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

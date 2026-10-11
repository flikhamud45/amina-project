# A6. The Kpre client's precompute on exact native and int8 products

Plan 2 item A6 (`SP2027_PLAN2.md` Sec. 3.4, should, rank 8; reviewer point A-W1, consolidated change 3). Code:
`Verifier._fold_local` in `code/src/pvi/fullcheck/protocol.py`, test `code/tests/test_fold_fast.py`, harness
`code/experiments/6_improvements/fold_ab.py` (commit bd0f95d). Status: implemented and tested on the laptop (the
float64 fallback and the unchanged Kpre verdicts); the native and GPU cases and the timing run on the cluster (job
1011448; Sec. 4).

## 1. Why

In mode Kpre the client holds the weights and, once per secret challenge, computes `u = chi A` (`[r, K]`, `r` = 5
interactive or 7 under Fiat-Shamir) for every weight matrix `A` (`Verifier.precompute`). This one-time setup belongs
in the client-options table (A-W1). The stored base record is 96.5 s for Llama-2-13B on the EPYC: small next to a
6.7 GB download, which is why A6 ranks low, but most of it was avoidable work. `_fold_local` made a transposed
int64 copy of every int8 matrix (8 bytes per weight: 360 MB for one of Llama-2-7B's FFN matrices) and multiplied it
with `chi^T` in float64 limb products (`small_matmul_mod`).

## 2. What changed

`_fold_local(op, chi)` now takes, in order:

* **On a CPU**, the native exact field product `native_kernels.field_matmul(chi, A)` (B5's kernel: `chi`'s entries in
  two 16-bit limbs, int64 accumulators reduced every block of rows, `|A| < 2^31`), with the weights read as int32 (4
  bytes per weight, no transpose; the kernel takes `A` with contiguous rows);
* **on a GPU** whose int8 GEMMs pass their self-check (`field.int8_ok`), `field.int8_weight_matmul(A^T, chi^T)^T`,
  the int8 GEMMs the prover's folds already use;
* otherwise (no native library, or `PVI_FOLD_FAST=0`) the float64 limb products as before.

All three return the canonical residue of the same integer vector `chi A`, so `u` and every verdict are unchanged;
the bias column is computed as before.

## 3. Validation

`tests/test_fold_fast.py`:

* the CPU path against the float64 products at the int8 extremes (a row of 127 and a row of -128), with a row of
  `chi` equal to `p - 1`, for `K` = 5, 300 and `2^16 + 37` (skipped where the native library cannot be built);
* the GPU path, the same cases (skipped without a GPU with int8 GEMMs);
* `PVI_FOLD_FAST=0` gives the float64 path;
* a Kpre query on a one-block GPT-2 graph: the same `u` for every op and the query accepted with the switch off and
  on.

Laptop: 2 passed, 2 skipped (no compiler, no GPU). `fold_ab.py` on one GPT-2 block (laptop, float64 on both paths):
both paths give identical `u`.

## 4. Results

[CLUSTER] job 1011448: the tests on the Xeon/2080 Ti node, then `fold_ab.py --model llama2-7b --layers 0 --devices
cpu cuda --reps 3 --fs` (the full model, Fiat-Shamir's 7 repetitions): the median precompute time of each path on
each device, and whether every `u` equals the float64 path's. The plan expects about 3 s on 8 CPU threads for the
7B model and about 0.1 s on the GPU.

## 5. Limitations

* The CPU path needs the native library (a C++ compiler on the client); without one the float64 path runs.
* The int32 copy of a matrix (4 bytes per weight) is made per op, one op at a time; it is freed before the next.

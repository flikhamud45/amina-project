# F3a. Real weights end to end: `prune_last` in the real builder, and the OPT-1.3B / OPT-6.7B cells

Plan 2 item F3a, part 1 (`SP2027_PLAN2.md`, F3a; Plan 1 F3, OPT part). Merged in 2245f22. Status: the builder
change, the script flags and their tests are done; the OPT-1.3B cells (C and Kpre, 64 and 2,048 tokens, with their
random-weight twins) and the OPT-6.7B real-weight cells (Kpre, 2,048 tokens) have run on the cluster (Section 6,
Results). The OPT-6.7B random twins are running; V1 with the GKR at 2,048 tokens did not fit the 11 GB card and moves
to the L40S `real` tier (Section 7).

## 1. Why

Reviewer C-W1: "a real-weight end-to-end run of at least one 7B-class model is the bar set by zkLLM and zkAgent".
The paper measures cost on random-weight graphs, which is sound because the prover's and the verifier's work depend
only on the shapes; F3a checks that claim on two real checkpoints (OPT-1.3B in modes C and Kpre, OPT-6.7B in Kpre).
The headline cells prune the last block (`build_decoder(..., prune_last=True)`), but the real builder could not:
`v1_bytes.py --real` refused `--prune-last`, so real cells could not be compared with the pruned headline cells. Real
weights also change what does depend on values: the compressibility of the claims (`PVC3`), V1's exceptions, and
which residual multipliers V1 can cut (P3).

## 2. The builder: `build_opt_from_hf(..., prune_last=True)`

`code/src/pvi/fullcheck/real_weights.py`.

* **The pruned graph.** The last block's q, attention output, out-projection, residual adds and MLP read the last
  position only; its k and v read every position, which the one query row attends to. The ops and their order are
  those of `build_decoder(..., prune_last=True)`: a "last position" op before q and one before the first residual
  add of the last block (the second is the residual stream's last position, which takes the attention branch).
* **Calibration on every position.** `_RealBuilder.last(name)` appends the slicing op but keeps the full calibration
  tensor and its real scale (as `transformer._Builder.last(..., calibrate_all=True)`). Every percentile, norm gain
  (with `--smooth`, the per-channel SmoothQuant gains) and multiplier after it is therefore computed on the same
  activations as in the graph without pruning. The pruned graph has that graph's weights and constants, and its
  next-token logits are the same integers.
* **Calibration memory.** `_RealBuilder.release([r])` drops each block's calibration tensors once the block is built
  (no later op reads them; only the tokens and the residual stream are kept), and the fp model's hidden states are
  dropped once `res_max` is read. For OPT-6.7B at 2,048 tokens this is 25 int64 tensors of `T x d` per block,
  1.68 GB, instead of 53.7 GB for all 32 blocks (measured unit count on the tiny OPT: 25.0 per block, 28.0 for the
  first, which also holds the embeddings; scaled by the shapes). It is what lets the OPT-6.7B cells run within the
  plan's 64 GB Xeon jobs.
* **`is_pruned(graph)`**: more than one "last position" op. **`matches_benchmark_graph(graph, cfg, prune_last=None)`**
  compares with `build_decoder(cfg, prune_last=is_pruned(graph))` by default and now also takes a `DecoderConfig`
  (used by the tests for the tiny OPT). The only expected mismatch is still the LM head's folded bias column.
* **`real_logits(all_positions=True)` on a pruned graph works, and is correct.** The F3a work order asked for it to
  fail clearly on a pruned graph, on the premise that a pruned graph cannot give every position. That premise does not
  hold: `real_logits(all_positions=True)` already replaces every "last position" op by the identity, the pruned
  block's two included. Position t's query row then attends to keys `0..t`, and its residual add reads position t.
  These are exactly the integers the pruned graph gives on the prefix `0..t`, and they equal the graph without
  pruning, whose constants it shares. The test `test_a_pruned_real_graph_gives_every_position` checks both equalities
  bit for bit. So the function was documented and tested rather than made to fail. If a hard failure is still
  wanted, it is a two-line change. (Review: confirmed on 6 more builds, Section 5; `transformer.with_lm_positions`
  already treats a pruned graph the same way, so the behaviour is consistent with the code base. The decision is the
  lead's.)

**The graph without pruning is unchanged.** Its digest (`IntGraph.digest()`, which binds every cheap op's function
by qualified name and captured constants) and a SHA-256 of all its weights and biases are the same before and after
the change. On the tiny OPT the digest is `dc2a907bd329c77a...` with scalar gains and `96ee48cc585d139b...` smoothed;
the weight hashes are `e06a49ec...` and `2b28e1b9...`. The existing lambdas keep their qualified names. D6a's roots
are therefore unaffected.

## 3. The scripts

**`experiments/8_v1/v1_bytes.py`**

* `--real ... --prune-last` builds the pruned real graph. `--calib-seq N` calibrates it on one window of N tokens
  (default `--seq`), so the 64-token cells can use the graph calibrated at 2,048 tokens: one model for both lengths.
  The windows come from `load_real`: queries from the start of the WikiText-2 test split, the calibration window from
  its end, and a second `load_windows` call whose own assertion keeps the two disjoint. `real_graph` builds with the
  F2 recipe. `--layers` is refused with `--real`.
* **The byte accounting (T7) on every line.** Three checks, in `accounting`, `accounting_ok` and `shapes`:
  * v0: `run_query(wire=True)`'s `u` and opened columns equal `analytic.proof_bytes` of the graph's weight ops
    exactly (`accounting.v0.u`, `.columns`). The multiproofs (`paths_ratio`) are compared with their expectation,
    which is exact for the looked-up rows of a plan's lookup tables. v0's claims are *not* a check: on the wire they
    travel in `PVC3`, whose size depends on their values, and the model takes the measured size (`wire_claims`), so
    a comparison would hold by construction (the implementation first reported it as a check; the review removed
    it). Their unencoded size, which the shapes fix, is checked by the tests. In mode Kpre `u` and the columns are 0
    on both sides, so a Kpre line without `--full` checks the shapes and claim columns only.
  * V1, with `--full`: every message but the multiproofs equals `cut_proof_bytes`, as in `test_cut_protocol.py`'s T7.
    This includes V1's claims: the real query and `cut_proof_bytes` encode the same values independently.
  * the weight-op shapes and claim columns equal `analytic.decoder_shapes` / `decoder_claim_columns(prune_last)`. A
    real LM head has one more column; this is expected and not reported.

  The exit status is 1 if a check fails, if an honest query is rejected, or if a tamper is accepted.
* `--tampers n` (with `--full`) runs, per variant, n trace-tampering V1 queries and n single-value ones. A trace
  tamper moves a random cut op's random entry by 1-3 widths of its normal window and recomputes the rest honestly. A
  single-value tamper changes a random entry of a random weight op by +-1..999. The result is recorded as
  `tampers: {trace, single}: {attempts, rejected, labels}`.
* `--codec-only` builds no commitment and no proof. It runs one forward pass, encodes each op's claims with `PVC3` as
  they are produced, and also encodes V1's values (the requantised values of the ops cut under a plan without col
  matrices; the other ops' claims) and its exceptions frame. Everything is summed per block (`embed`, `0..L-1`,
  `head`). This is the OPT-6.7B byte profile, for a machine that cannot hold its prover. Each op is encoded on its
  own, so the sums differ slightly from one encoding of all claims (on the mini OPT below, 1.6% less for v0's claims
  and 1.2% less for V1's values).
* `peak` (per variant) and `build_peak` record the host's VmHWM (Linux; `None` elsewhere) and the GPU's peak
  allocated and reserved bytes, under `bench.py`'s names `host_peak_rss`, `gpu_peak_memory` and `gpu_peak_reserved`.
  Each line also carries `graph_digest`.

**`experiments/6_improvements/perf.py`**

* `--real/--smooth/--calib-seq` take the graph from `v1_bytes.load_real`/`real_graph`, with every block built.
  `--model` names the shapes, and the commitment plans are made from the graph's own weight ops. The queries are
  `--queries` + 1 WikiText-2 windows, the first being the warm-up.
* `--tampers n` runs, after the timed queries, n of each of `bench.py`'s LLM attacks per variant: `single_value`,
  `output_logit` (the smallest logit raised 1000 above the largest) and `last_hidden` (a coordinate of the last
  block's down projection at the last position set to `Z_BOUND // 4`).
* `peak` (over a mode's warm-ups, queries and tampers, all its variants) and `build_peak` (build and commitments).
* The exit status is 1 if an honest query is rejected or a tampered one accepted.

## 4. Tests

`code/tests/test_real_weights.py` (15 tests). The models are a tiny random-init `OPTConfig` (d = 32, 2 blocks) and,
for the scripts, `9_quality/common.load_opt`/`load_windows` monkeypatched to that model and random windows, so
nothing is downloaded:

* `test_a_pruned_real_graph_has_the_constants_and_logits_of_the_whole_one` (scalar and smoothed): the same weights,
  biases and cheap-op constants (requant multipliers, norm gains, attention multipliers, tables); `info` identical;
  only the LM head bias differs from the benchmark's pruned graph; logits bit-identical for `T` = 1, 2, 9, 24 on two
  prefixes each; claim shapes as `decoder_claim_columns(prune_last=True)`; fewer claims for `T > 1`.
* `test_a_pruned_real_graph_gives_every_position`: see Section 2.
* `test_a_pruned_real_graph_is_accepted_by_v0_and_v1_and_rejects_tampering` (C with the auto plan, which has lookup
  tables and col matrices; Kpre). Honest queries are accepted by v0 (interactive and on the wire) and by V1. v0
  rejects tampers on the pruned out-projection, the top logit and a k claim, at `freivalds`, or at `columns_code`
  for an op of a col matrix. V1 rejects the trace-tampering attack on the first cut op of each layout, at
  `cut_final` (row), `columns_code` (col) or `kpre_y` (Kpre), and a tamper on a clear op of the pruned block.
* `test_the_byte_accounting_is_exact_on_a_pruned_real_graph` (C, Kpre): v0's claims (unencoded), `u` and opened
  columns equal `analytic.proof_bytes`; on the wire `u` and the columns (the claims' wire size is measured); V1's T7.
* The scripts: `v1_bytes.py --real --prune-last --full --reps 2 --tampers 2` (both modes; accounting, shapes, timed
  and tampers all good; exit 0); a byte model off by one `u` byte gives `accounting_ok: false` and exit 1;
  `--codec-only` (the blocks, the claims per block as the shapes give them, V1 < v0 < raw, the graph digest);
  `perf.py --real --prune-last on --tampers 1` (both modes; 2/2 accepted, every tamper rejected; exit 0).

Run on the laptop (2 threads), with the final code after the review:
`test_real_weights.py test_pruning.py test_plan_auto.py test_plans.py::test_perf_random_init_builds_every_cnn_it_accepts
test_fast_verifier.py::test_the_real_weight_residual_is_the_reference_residual`: **48 passed** (291 s). (Before the
review, without the last test: 47 passed in 276 s.)

## 5. Laptop validation on a mini OPT (not real weights)

To run the scripts end to end with the real WikiText-2 windows and the real OPT tokenizer, without loading a
checkpoint (the laptop has about 1-2 GB free), `load_opt` was replaced by a random-init `OPTConfig` with OPT's
vocabulary (50,272) and width 128 (4 blocks, d_ff 512). It went through the real builder (smoothed, alpha 0.55),
with the last block pruned, 128-token queries and a 256-token calibration window. These numbers validate the
scripts; **they are not real-weight measurements.**

`v1_bytes.py --fs --full --reps 3 --tampers 3 --lmax 20` (Fiat-Shamir, auto plan = cnn18c):

| Mode | v0 bytes (claims / u / columns / paths) | V1 bytes (claims / exc / mult / gkr / u / columns / paths) | Ratio | Accounting |
|---|---|---|---:|---|
| C | 1,422,054 (1,086,250 / 55,676 / 228,160 / 51,968) | 1,080,189 (605,803 / 420 / 10,929 / 18,353 / 164,796 / 228,656 / 51,232) | 1.32x | all exact; paths 1.007 (v0), 1.010 (V1) |
| Kpre | 1,082,722 | 747,074 (602,300 / 420 / 10,929 / 18,353 / 115,072 / 0 / 0) | 1.45x | all exact |

Both modes: 475,136 cut leaves, one instance, table 10,361, 42 exceptions. The full V1 query and 3/3 timed queries of
each protocol were accepted. Every tamper was rejected: trace 3/3 (C: `columns_code` x3; Kpre: `kpre_y` x3) and
single 3/3 (C: `columns_code` x3; Kpre: `freivalds` 1, `kpre_y` 2). The whole process peaked at a 1,741 MB working
set; the eager CPU GKR takes 10-24 s per V1 query here, which does not represent the GPU prover.

`perf.py --wire on --prune-last off on --tampers 2` (interactive, auto plan = cnn18c, 3 queries after a warm-up,
all accepted, every tamper rejected):

| Mode | Proof bytes, unpruned -> pruned | Prove / verify (s), unpruned -> pruned |
|---|---|---|
| Kpre | 1,299,810 -> 1,081,038 (0.832x) | 0.087 / 0.036 -> 0.074 / 0.032 |
| C | 1,534,716 -> 1,323,192 (0.862x) | 0.180 / 0.068 -> 0.162 / 0.054 |

Tamper labels in mode C were `freivalds` or `columns_code`, depending on the op's layout. The peak working set was
1,851 MB.

`v1_bytes.py --codec-only` (the same graph and query; peak working set 653 MB; forward and encoding 0.26 s):

| Block | Claims | Raw (4 B) | v0 `PVC3` | V1 values | Exceptions |
|---|---:|---:|---:|---:|---:|
| embed | 32,768 | 131,072 | 29,211 | 29,211 | 0 |
| 0 | 147,456 | 589,824 | 291,876 | 146,901 | 16 |
| 1 | 147,456 | 589,824 | 289,929 | 144,194 | 7 |
| 2 | 147,456 | 589,824 | 286,675 | 142,547 | 16 |
| 3 (pruned) | 33,664 | 134,656 | 66,840 | 31,250 | 3 |
| head | 50,272 | 201,088 | 100,760 | 100,760 | 0 |
| total | 559,072 | 2,236,288 | 1,065,291 | 594,863 (+ 420 B exceptions frame) | 42 |

The pruned block's 33,664 claims are k and v at 128 columns plus q, o, fc1 and fc2 at one column each.

The tables above are the implementation's runs, made before the review removed v0's vacuous claims check ("all exact"
then included it; `u`, the columns and every V1 component were real checks).

### Review re-run (the cluster command lines, on a smaller mini OPT)

The review re-ran the builder checks and every cluster command line of Section 6 as written, with only the model,
the lengths, the threads and the device changed: a random-init OPT with OPT's vocabulary (50,272), width 64, 3 blocks,
4 heads, d_ff 256; the real OPT tokenizer and WikiText-2 windows; `--seq 64 --calib-seq 256` for the 64-token cells
and `--seq 256` for the 2,048-token ones; 2 threads, CPU. Not real weights.

* **The graph without pruning is unchanged.** HEAD's `real_weights.py` and the working tree's, each loaded as
  `pvi.fullcheck.real_weights` in its own process, give the same digest, weights and `info` on 6 builds (2 tiny OPTs,
  2 blocks x 4 heads x d 32 and 3 blocks x 2 heads x d 48, each with scalar gains, the F2 smoothing, and 99th/99.5th
  percentile clipping). The pruned graphs have the same weights, and their logits are bit-identical to the unpruned
  graph's on every prefix length T = 1..40 (0 mismatches out of 240). `real_logits(all_positions=True)` of the pruned
  graph equals the unpruned graph's on all 6 builds.
* **A1** (`v1_bytes.py`, real, `--full --reps 3 --tampers 3 --lmax 24`): exit 0 in both modes. Kpre: v0 265,485 B,
  V1 244,198 B (1.09x), 13 exceptions. C (auto plan cnn18c): v0 417,772 B, V1 394,688 B (1.06x), `paths_ratio` 1.006
  (v0) and 1.022 (V1). Every V1 component and v0's `u` and columns exact; shapes and claim columns as the model; the
  full V1 query and 3/3 timed queries of each protocol accepted; trace and single tampers 3/3 rejected (Kpre:
  `kpre_y`, and `freivalds` once; C: `columns_code`). Both modes built the same graph (one digest).
* **A1-random**, Kpre: exit 0; v0 267,406 B, V1 249,733 B (1.07x), 0 exceptions; every check passed.
* **A2** (`perf.py`, real and random, `--modes Kpre C --lookups on --tampers 3`): exit 0 both. Real: Kpre 257,983 B,
  C 420,095 B; random: 259,543 B and 430,015 B. 3/3 accepted on every line; `single_value`, `output_logit` and
  `last_hidden` 3/3 rejected each (Kpre: `freivalds`; C: `columns_code`, and `freivalds` for `last_hidden`).
* **A3**'s flags (`--extra '{"stream": true}'`, one process per mode) with the streaming verifier on the CPU (no GPU
  here): exit 0 in both modes; 3/3 accepted; every tamper rejected.
* **B2** (`--codec-only`, real and random): exit 0; shapes ok; blocks embed, 0, 1, 2, head. Real: 411,168 claims,
  v0 `PVC3` 756,574 B, V1 values 439,277 B, 36 exceptions (exceptions frame 348 B). Random: v0 757,512 B, V1 values
  442,758 B, 0 exceptions.
* **B3** (`v1_bytes.py --modes Kpre` without `--full`, real and random): exit 0. Real: v0 765,468 B, V1 506,062 B
  (1.51x), 36 exceptions, the same graph digest as B2's. Random: 763,020 B and 519,645 B (1.47x).

Not run here: CUDA, the Triton GKR, the streaming verifier on a GPU, the native CPU verifier (no compiler set up),
VmHWM (Windows has none, so every `peak.host_peak_rss` is `None` on the laptop), and any checkpoint.

## 6. The cluster cells

Run from `code/` of this branch on the TAU nodes. The environment of every job is:

```
export PYTHONPATH=src HF_HOME=/tmp/$USER/hf OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NVIDIA_TF32_OVERRIDE=0 \
       PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

It also needs the sp2027 job environment of `I_server_setup.md`: `TRITON_CACHE_DIR` and `PVI_PY_INCLUDE` for the
Triton GKR on GPU jobs; `CXX`, `PATH` and `PVI_NATIVE_DIR` for the native CPU verifier. Each job downloads its
checkpoint into the node-local `HF_HOME` on first use (`load_opt`: `*.json`, `*.bin`, `*.txt`; fp16 checkpoints of
about 2.6 GB for OPT-1.3B, from its parameter count, and 13.4 GB for OPT-6.7B, per `F2_quality.md`), and the
WikiText-2 test parquet.

All cells use Fiat-Shamir (F6), the last block pruned, `--policy auto` (replace it by PS2's policy once it is
chosen), and 3 queries after a warm-up. The real graph is the F2 one: `--smooth 0.55`, calibrated on one 2,048-token
window at both lengths. Each real cell has a random-weight twin of the same shape and flags, without
`--real/--smooth/--calib-seq`.

Memory, from the shapes (estimates; every line records the measured `build_peak` and `peak`): OPT-6.7B loads at
26.6 GB in fp32. Its build adds the int8 graph (6.7 GB), one block of calibration (1.68 GB) and the smoothed LM
head's float64 temporaries (the fold and `_RealBuilder.mat`: up to about 4 x 1.65 GB at once), about 42 GB before the
process's own overhead. The fp32 model is freed before the commitments and queries. A CPU prover's query then holds
every claim as int64: about 2.38 G entries at 2,048 tokens with the last block pruned, 19 GB, and the verifier's
decoded copy as much again; `cut_proof_bytes` holds the claims and the cut ops' sent values (up to another 19 GB). B3
can therefore reach about 50-60 GB, too close to a 64 GB limit. The figures for OPT-1.3B are 5.3 GB (fp32), 1.3 GB
(graph), 0.84 GB (calibration), about 3 GB (head temporaries) and 0.88 G claim entries (7 GB as int64). Suggested
`--mem`: 64 GB for OPT-1.3B; 128 GB for every OPT-6.7B cell (B1-B3; the plan's floor is 64 GB, and the existing
sp2027 tiers give the 7B-class cells 192-256 GB).

| Cell | Node | Command (for `T` in 64 2048) | Pass |
|---|---|---|---|
| A1 OPT-1.3B real: V1 and v0 bytes per component, T7, V1 timing, V1 tampers (one process per mode) | 2080 Ti | `python experiments/8_v1/v1_bytes.py --model opt-1.3b --real facebook/opt-1.3b --smooth 0.55 --calib-seq 2048 --prune-last --seq $T --modes Kpre --fs --lean --full --reps 3 --tampers 3 --lmax 24 --threads 8 --device cuda --verifier-device cpu --label f3a-real`, then the same with `--modes C --policy auto` | exit 0; `accounting_ok` true (`accounting.v0.u`, `.columns` and every key of `accounting.v1` true, `paths_ratio` about 1); `shapes` `{"problems": [], "claim_columns": true}`; `v1_full.accepted`; timed 3/3 both; tampers 3/3 rejected each |
| A1-random | 2080 Ti | the same with `--model opt-1.3b --prune-last` and no `--real/--smooth/--calib-seq`, `--label f3a-random` | as A1 |
| A2 OPT-1.3B real: v0 prover and native CPU verifier time, peak memory, v0 tampers | 2080 Ti | `python experiments/6_improvements/perf.py --model opt-1.3b --real facebook/opt-1.3b --smooth 0.55 --calib-seq 2048 --seq $T --modes Kpre C --policy auto --fs --queries 3 --lean --wire on --prune-last on --lookups on --threads 8 --device cuda --verifier-device cpu --tampers 3 --label f3a-real-cpuv` | exit 0; 2 lines, `accepted` 3 of 3; every `tampers.*.rejected` = 3 |
| A2-random | 2080 Ti | the same without `--real/--smooth/--calib-seq`, `--label f3a-random-cpuv` | as A2 |
| A3 OPT-1.3B real: streaming GPU verifier (one process per mode) | 2080 Ti | `python experiments/6_improvements/perf.py --model opt-1.3b --real facebook/opt-1.3b --smooth 0.55 --calib-seq 2048 --seq $T --modes Kpre --fs --queries 3 --lean --wire on --prune-last on --lookups on --threads 8 --device cuda --verifier-device cuda --extra '{"stream": true}' --tampers 3 --label f3a-real-gpuv`, then `--modes C --policy auto` | as A2 |
| A3-random | 2080 Ti | the same without `--real/--smooth/--calib-seq`, `--label f3a-random-gpuv` | as A2 |
| B1 OPT-6.7B real, full model, Kpre: v0 time, peak memory, tampers | 2080 Ti, `--mem` 128 GB | `python experiments/6_improvements/perf.py --model opt-6.7b --real facebook/opt-6.7b --smooth 0.55 --calib-seq 2048 --seq $T --modes Kpre --fs --queries 3 --lean --wire on --prune-last on --lookups on --threads 8 --device cuda --verifier-device cpu --tampers 3 --label f3a-real67`; if the GPU prover runs out of memory, the same on a Xeon node with `--device cpu --label f3a-real67-cpu` | exit 0; `accepted` 3 of 3; tampers 3/3 rejected each |
| B1-random | as B1 | the same with `--model opt-6.7b` without `--real/--smooth/--calib-seq` | as B1 |
| B2 OPT-6.7B @2048 codec-only byte profile | Xeon, `--mem` 128 GB | `python experiments/8_v1/v1_bytes.py --model opt-6.7b --real facebook/opt-6.7b --smooth 0.55 --calib-seq 2048 --prune-last --seq 2048 --codec-only --lmax 27 --threads 8 --device cpu --label f3a-real67-codec` | exit 0; `shapes` `{"problems": [], "claim_columns": true}`; 34 blocks (embed, 0..31, head); `total.v1_values < total.v0_claims` |
| B2-random | Xeon | the same with `--model opt-6.7b --prune-last` and no `--real/--smooth/--calib-seq` | as B2 |
| B3 OPT-6.7B @2048 real V1 bytes (`cut_proof_bytes`, no GKR) | Xeon, `--mem` 128 GB | `python experiments/8_v1/v1_bytes.py --model opt-6.7b --real facebook/opt-6.7b --smooth 0.55 --calib-seq 2048 --prune-last --seq 2048 --modes Kpre --fs --lean --lmax 27 --threads 8 --device cpu --label f3a-real67-v1bytes` | exit 0; `v0_accepted`; `accounting_ok`; `shapes` `{"problems": [], "claim_columns": true}` (in Kpre v0's `u` and columns are 0, so this line checks the shapes only; `cut_proof_bytes` itself is checked against real V1 queries by A1 and the tests); the same `graph_digest` as B2 |
| B3-random | Xeon | the same with `--model opt-6.7b --prune-last` and no `--real/--smooth/--calib-seq` | as B3 |

What each line gives. `v1_bytes.py` (A1, B3): `v0_bytes`, `v1_bytes` (per component), `ratio`, `n_exceptions`,
`n_leaves`, `accounting`, `timed.{v0,v1}.{prove_s,verify_s,timings}`, `tampers`, `peak`, `build_peak` and
`graph_digest`. `perf.py` (A2, A3, B1): `prove_s`, `verify_s`, `timings`, `bytes`, `proof_bytes`, `tampers`,
`peak` and `build_peak`. `--codec-only` (B2): `blocks` and `total`.

Fallbacks:
* If mode C with the GKR at 2,048 tokens does not fit the 11 GB card in A1, rerun that cell without
  `--full --reps 3 --tampers 3`. This gives the bytes and v0's T7; V1's T7 in mode C is then checked at 64 tokens.
  The 2,048-token timing goes to the L40S `real` tier (Section 7).
* If A3's mode C does not fit (the prover and the streaming verifier share the card), it moves to the same tier.

The plan expects times within 5% of random weights, v0 bytes within about 10%, and V1 1.7-1.9x on real weights.

### Results (jobs 1008673 and 1009431, s-005, 2080 Ti; git ebf6f4e)

All lines: Fiat-Shamir, last block pruned, 3 timed queries after a warm-up, the smoothed graph of F2
(`--smooth 0.55`, calibrated on one 2,048-token window). OPT-1.3B real digest `182e6a71...`, random `266c008f...`;
OPT-6.7B real `94793ed9...`.

**OPT-1.3B, prover on the GPU, native CPU verifier (A2), and V1 at 64 tokens (A1).**

| Mode | T | v0 (MB) real / random | V1 (MB) real / random | V1 ratio real / random | Prove (s) real / random | CPU verify (s) real / random | Peak host / GPU (GB), real | Tampers | V1 exceptions real / random |
|---|---:|---|---|---|---|---|---|---|---:|
| C | 64 | 101.6 / 111.9 | 90.4 / 90.2 | 1.12 / 1.24 | 0.75 / 0.82 | 0.98 / 0.95 | 4.2 / 2.6 | 9/9 v0, 6/6 V1 | 2,379 / 1 |
| C | 2,048 | 1,721.9 / 2,070.8 | (OOM, L40S) | - | 3.44 / 3.76 | 8.22 / 8.65 | 21.9 / 5.6 | 9/9 v0 | - |
| Kpre | 64 | 50.1 / 60.4 | 39.1 / 39.7 | 1.28 / 1.53 | 0.20 / 0.22 | 0.51 / 0.54 | 4.7 / 2.6 | 9/9 v0, 6/6 V1 | 2,379 / 1 |
| Kpre | 2,048 | 1,662.1 / 2,011.0 | (OOM, L40S) | - | 2.04 / 2.10 | 7.72 / 7.41 | 21.1 / 5.5 | 9/9 v0 | - |

V1 at 64 tokens (A1, GPU prover, CPU verifier): prove 1.93 / 2.13 s (Kpre), 2.36 / 2.55 s (C), real / random;
verify 4.82 / 5.69 s (Kpre), 4.74 / 5.54 s (C). T7 on real weights: every key of `accounting.v1` and
`accounting.v0` exact; `paths_ratio` 1.0007 (v0, C) and 0.9936 (V1, C). `shapes` `{"problems": [], "claim_columns":
true}` on every line. The peak host memory of A2 at 2,048 tokens (21-22 GB) is the CPU verifier's decoded claims.

**OPT-6.7B, full model, Kpre, 2,048 tokens (B1-B3), real weights.** The GPU prover runs out of memory on the 11 GB
card (as planned), so B1 used the fallback, the CPU prover on the Xeon node.

| Line | Result |
|---|---|
| B1 v0, CPU prover and CPU verifier | 3/3 accepted; prove 270.0 s (forward 252.2 s, encode 17.8 s); verify 20.2 s (decode 2.7, derive 11.4, products 6.1); 4,611.6 MB of claims; tampers 9/9 rejected (`freivalds`); peak host 70.9 GB (build 44.6 GB) |
| B2 codec-only profile | 34 blocks; 2.374 G claims; v0 claims 4,309.7 MB, V1 values 2,328.2 MB (1.85x); 224,670 exceptions (1.80 MB framed); per block 1.76-1.96x (the last kept block 2.31x) |
| B3 real V1 bytes | v0 4,649.9 MB (accepted), V1 2,680.5 MB (claims 2,645.8, `u` 32.1, cut 2.5): **1.73x**; 16 GKR instances, table 64,305; `accounting_ok`; same digest as B2 |

The random-weight twins of B1-B3 are still running (job 1009431); their numbers are added here when it ends.

**Reading.**
* **Cost does not depend on the weights.** On OPT-1.3B the prover's time is within 3-9% of the random twin's and
  the native CPU verifier's within 6%, in both modes and at both lengths. The prover is faster on real weights in
  every cell (its claims are smaller to encode); the verifier is faster in two cells and slower in two.
* **v0 bytes are smaller on real weights** (by 9-17%): the claims of a calibrated model are more concentrated than a
  random model's, and `PVC3`'s size follows their entropy. The paper's random-weight byte figures are therefore upper
  bounds for these models.
* **V1 is about the same size on both** (39.1 / 39.7 MB in Kpre at 64 tokens), so its ratio to v0 is lower on real
  weights (1.28x instead of 1.53x at 64 tokens; 1.73x on OPT-6.7B at 2,048 tokens, where the plan expected 1.7-1.9x).
  Real weights cause about 2,400 exceptions on OPT-1.3B at 64 tokens and 225,000 on OPT-6.7B at 2,048 tokens, against
  1 on random weights; they cost 0.02 MB and 1.8 MB.
* **Tampers.** Every tamper is rejected, by the check expected for its mode (`kpre_y` and `freivalds` in Kpre,
  `columns_code` in C).
* **What did not fit the 2080 Ti.** A1 at 2,048 tokens (V1 with the GKR, both modes, real and random alike: the
  GKR's tables need more than 11 GB) and B1's GPU prover. Both move to the L40S `real` tier (Section 7), as the
  fallback rule says; V1's T7 at 2,048 tokens is covered on OPT-6.7B by B3 (bytes, no GKR).

## 7. Draft of the L40S `real` tier (for `slurm/sp2027.sh`, not applied here)

This covers the cells that do not fit the 2080 Ti, plus the final-platform timing (plan F6.4: `real`, landing with
F3a). It is not applied here, because another agent is editing `sp2027.sh`. The usage line gains `|real`.

```bash
real)
  # F3a: real-weight OPT end to end on the final platform, each cell next to its random-weight twin (needs HF_HOME:
  # facebook/opt-1.3b and facebook/opt-6.7b, the WikiText-2 test parquet). One graph per model: the smoothed integer
  # OPT (F2) calibrated on one 2,048-token window, last block pruned; Fiat-Shamir, 3+ queries after a warm-up.
  R13="--model opt-1.3b --real facebook/opt-1.3b --smooth 0.55 --calib-seq 2048"
  R67="--model opt-6.7b --real facebook/opt-6.7b --smooth 0.55 --calib-seq 2048"
  P0="experiments/6_improvements/perf.py --fs --queries 5 --lean --wire on --prune-last on --lookups on --threads 8 \
--verifier-device cpu --tampers 3"
  for T in 64 2048; do
    sub "r-opt67-v0-$T"    6 192 "HF_HOME=$HF_HOME $PY $P0 $R67 --seq $T --modes Kpre --label real"
    sub "r-opt67rnd-v0-$T" 6 192 "$PY $P0 --model opt-6.7b --seq $T --modes Kpre --label real-random"
    sub "r-opt67-v1-$T"    6 256 "HF_HOME=$HF_HOME $V1 $R67 --seq $T --modes Kpre --prune-last --lmax 27 --tampers 3 --label real"
    sub "r-opt67rnd-v1-$T" 6 256 "$V1 --model opt-6.7b --seq $T --modes Kpre --prune-last --lmax 27 --tampers 3 --label real-random"
    for M in C Kpre; do    # OPT-1.3B's V1 with the GKR (mode C at 2,048 tokens does not fit the 2080 Ti)
      sub "r-opt13-v1-$M-$T"    4 192 "HF_HOME=$HF_HOME $V1 $R13 --seq $T --modes $M --policy auto --prune-last --lmax 27 --tampers 3 --label real"
      sub "r-opt13rnd-v1-$M-$T" 4 192 "$V1 --model opt-1.3b --seq $T --modes $M --policy auto --prune-last --lmax 27 --tampers 3 --label real-random"
    done
  done
  ;;
```

(`$V1` is the tier file's `v1_bytes.py --fs --lean --full --reps 3`.)

## 8. Limitations

* **The calibration recipe** is `common.load_windows`: one window from the end of the WikiText-2 test split, on the
  CPU. It is not yet F2a's train-split windows or D6a's fixed recipe. Costs do not depend on it. Bytes do slightly,
  through the claims' entropy and V1's exceptions. When D6a fixes the recipe, `real_graph` is the one place to change.
* **v0's wire claims are measured, not modelled.** Their `PVC3` size depends on the values, so the scripts' v0
  accounting checks `u` and the opened columns (both 0 in Kpre) and reports the multiproofs' ratio; real-against-random
  differences in v0 bytes come from the claims alone. V1's claims are checked (the real query against
  `cut_proof_bytes`) only on lines with `--full`.
* **Memory figures for the cluster are estimates from the shapes** (Section 6); the measured `build_peak` and `peak`
  of the runs replace them.
* **The LM head's folded bias** gives the real graph one more column than the benchmark's (`d + 1`). In mode C its
  codeword length is `4 next_pow2(d + 1)`, twice the benchmark's (commit time and that tree's paths, not the opened
  columns). The shape check reports it as expected.
* **V1 Kpre in `v1_bytes.py` sends the embedding claims** (it has no `--lookups`), while `perf.py`'s Kpre cells use
  lookups (the paper's optimised protocol). V1 with lookups is untested.
* **`perf.py`'s `peak` covers all variants of a mode.** The cells above run one variant per process, so it is that
  cell's peak.
* **`--lmax`**: 24 for timed V1 on the 2080 Ti (the GKR's memory), 27 for the bytes-only Xeon cells and the L40S tier.
  Only the number of GKR instances changes; real/random pairs always share `lmax`.
* **The laptop runs** (Section 5) use a random-init model through the real builder, not a checkpoint. OPT-125M was
  not loaded: beside its 0.5 GB of fp32 weights, the builder's LM-head fold holds two float64 copies of the
  50,272 x 768 head (0.31 GB each), and the laptop had 1-2 GB free. The mini runs already peaked at 1.7-1.9 GB.

## 9. How to run

```
cd code
PYTHONPATH=src python -m pytest -q -o addopts= -p no:cacheprovider tests/test_real_weights.py tests/test_pruning.py
PYTHONPATH=src python experiments/8_v1/v1_bytes.py --model opt-1.3b --real facebook/opt-1.3b --smooth 0.55 \
    --calib-seq 2048 --prune-last --seq 2048 --modes Kpre --fs --lean --full --reps 3 --tampers 3 --lmax 24
PYTHONPATH=src python experiments/6_improvements/perf.py --model opt-1.3b --real facebook/opt-1.3b --smooth 0.55 \
    --calib-seq 2048 --seq 2048 --modes Kpre C --policy auto --fs --queries 3 --lean --wire on --prune-last on \
    --lookups on --tampers 3
PYTHONPATH=src python experiments/8_v1/v1_bytes.py --model opt-6.7b --real facebook/opt-6.7b --smooth 0.55 \
    --calib-seq 2048 --prune-last --seq 2048 --codec-only --device cpu
```

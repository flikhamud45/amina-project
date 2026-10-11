# A7. The prover against unverified serving, on the same GPU, with batching

Plan 2 item A7 (must; reviewers C-W2, C-Q1 and A-Q2, consolidated change 3). Files:
`code/experiments/7_reexec/serve_ab.py` (new), `code/experiments/7_reexec/reexec.py` (`FloatDecoder(prune_last=True)`,
opt-in), `code/tests/test_serve_ab.py` (new), the `serve` tier and the smoke list in
`code/experiments/4_defence_benchmark/slurm/sp2027.sh`, and `docs/L40S_RUN_SP2027.md`. Status: the harness, the tier
and the tests are done and validated on the laptop CPU with small graphs. The 2080 Ti pilot ran (job 1008674, s-004,
git e5d377c; Sec. 7.1); the L40S `serve` tier has not run yet (**[CLUSTER]** below).

## 1. Why it matters

Reviewer C asks what proving costs a provider compared with serving the same requests without proofs (C-W2, C-Q1).
Reviewer A notes that CommitLLM reports a 12-14% overhead (A-Q2). Today the paper compares the L40S prover with fp16
inference on a 2080 Ti, that is, across GPUs. The answer must come from one GPU, one host and the same prompts, and it
must separate two costs:

* the cost of the **exact integer model** itself (our model computes integers, not fp16 values), and
* the cost of the **proof** on top of it (encoding, hashing, folding, opening).

The second cost is paid only for queries whose proof is requested, because the claims are a deterministic function
of the model and the prompt (Sec. 3.5). The first is paid on every query if the provider serves the integer model.

## 2. What is measured

A cell is a model, a prompt length `T` and a batch `B` (B prompts of T tokens). The token ids are random, drawn with
seed 1 as `perf.py` and `reexec.py` draw them, so prompt 0 of every cell is the B = 1 prompt of its T. The models
are the decoders of `CONFIGS` with random weights, since all costs depend only on the shapes. They are built as
`perf.py --prune-last` builds them (seed 0, 32 calibration tokens), on `--layers` blocks (0: the full model). Three
ways of answering the batch run on the same token ids, shapes, device and host:

| | What | Code |
|---|---|---|
| **U1** | our exact integer model served without proofs: the pruned graph's forward pass with every claim dropped as it is computed | `IntGraph.forward(x, free=True, send=_drop)` |
| **U2** | an eager floating-point decoder of the same shapes (fp16 and/or bf16 on a GPU, fp32 on a CPU): `F.linear` and `scaled_dot_product_attention`, one batched prefill, the last block at the last position like the integer graph | `reexec.FloatDecoder(..., prune_last=True)` |
| **prover** | modes C and Kpre, Fiat-Shamir, the compact wire encoding, the lean prover, the pruned graph; Kpre with `--lookups` (as the cpuv and gpuv tiers); mode C under `--policy`'s commitment plan | `serve_ab.prove`: the prover's side of `run_query` |

Workloads (the plan's): (a) prefill at B = 1 with T in {64, 512, 2048}; (b) batched prefill, T = 64 with B in
{1, 4, 16, 64} and T = 2048 with B in {1, 4}.

**Batches.** U1 and U2 serve the B prompts as one `[B, T]` forward pass. The prover today proves one statement per
query, so the harness proves a batch in two ways (`--batch-proof`):

* `seq`: the B prompts are B independent queries, for example from B users. They are proved one after another, and
  each proof is the B = 1 proof of its prompt. A batched forward pass with per-query proofs (plan C4c) is not
  implemented, so `seq` also pays B unbatched forward passes.
* `joint`: the B prompts form one statement, one client's batch: the claims of all of them, with one `u` and one
  opening (`run_query` on `[B, T]`, plan C4a's path).

**Outputs.** Each (cell, mode, batch proof) gets one JSON line with `row = "prove"`:

* medians with 95% bootstrap intervals of U1, U2 per dtype and the prover;
* the ratios prover/U1, prover/U2 and U1/U2;
* the prover's phases, and the proof-only overhead (prover minus U1) split into `forward_extra`, `encode`, `hash`,
  `fold`, `open`, `lookups`, `challenges` and `residual`;
* proof bytes in total and per message;
* proving time per token, proof MB per token, and the egress cap: the proved tokens per second that a 100 Gb/s link
  carries, `12.5e9 / (bytes per token)`;
* the checks of Sec. 3.4;
* the SHA-256 of the proof, peak GPU memory per variant, peak host memory, contention evidence and the raw times.

Each cell also gets one line with `row = "on_request"`. It gives the serving cost U1 and its ratio to U2, and for
each prover variant the cost of a later proof with its SHA-256 and the determinism check.

## 3. How

### 3.1 Timing

One warm-up of every variant comes first. It is untimed and runs the checks. Then come `--reps` repetitions (10 in
the tier) with the variants interleaved, the order rotating from repetition to repetition as in `perf.py`. On a GPU
each call is timed with CUDA events. The start event is recorded after a device synchronisation; the end event is
recorded after the call returns and the device has been synchronised again (so work left on any stream is inside the
window), and it is waited for. The window therefore ends when the GPU has finished the call's work, and it includes
the host work the call did meanwhile (hashing and parts of the encoding). On a CPU the harness uses
`perf_counter`. A time is reported as the median with a 95% percentile-bootstrap interval (10,000 resamples, seeded).
A ratio is the ratio of the medians, with an interval from the same bootstrap over both samples.

U2's model is built and timed first, for every cell, and freed before the integer model goes to the device. On the
11 GB card and for the 13B model on the L40S, the fp16 and int8 models need not fit at once. U2 is therefore not
interleaved with U1 and the prover. A U2 dtype or cell that the device cannot run (out of memory, or bf16 GEMMs on a
GPU without them) is recorded as skipped. A prover variant that runs out of GPU memory is recorded as
`"skipped": "out of memory"` and the job continues: `torch.cuda.OutOfMemoryError`, and a `RuntimeError` whose text
says the GPU ran out of memory outside torch's allocator (`CUDA error: out of memory`, `CUBLAS_STATUS_ALLOC_FAILED`).
Any other error stops the job. No cell is extrapolated.

### 3.2 The prover without the verifier

`run_query` interleaves the prover's messages with the verifier's checks, so its wall time is not the prover's time.
`serve_ab.prove` runs the prover's side alone: `protocol._claims_message`, `_fold_message` and `_open_message`. These
are the functions `run_query` runs, so the messages, byte counts and phase timings are identical. The challenges are
also identical: under Fiat-Shamir they depend only on the transcript, and the verifier's checks never touch it. The
function returns the proof as its byte strings:

* the PVC3 claims;
* a commitment plan's lookup-table rows and the hashes of their multiproofs;
* in mode C, the packed `u`, the packed opened columns and the hashes of their paths.

Mode C needs Fiat-Shamir here, because without a verifier there is nobody to draw the challenges. `prove` refuses an
interactive mode-C verifier. Kpre's single message reads no challenge.

Two details of the in-process run are corrected for:

* `_fold_message` and `_open_message` also decode `u` and the columns for the verifier (`verify_decode`). That is not
  the prover's work, so `prove_s` is the measured window minus `verify_decode`.
* Under Fiat-Shamir the prover derives the challenges (chi, the column indices) from the transcript itself. `run_query`
  books that work to neither party. `TimedChallenges`, a `Verifier` subclass, times `fold_challenges` and
  `column_challenges`, and the harness books the result as the prover's `challenges` phase. `serve_ab`'s `prove_s` is
  therefore slightly more complete than `perf.py`'s.

`messages` rebuilds the opened columns' bytes from the decoded rows with `claimcodec.pack_field`, outside the timed
window. The packing is a fixed function of the values, and `tests/test_claimcodec.py` checks that the device packing
equals the host packing, so these are the bytes that were sent.

### 3.3 Phases

The phases are `run_query`'s timings plus `challenges`:

* `prove_forward`: the forward pass that computes the claims;
* `prove_lookups`: a plan's table multiproofs;
* `prove_encode`: the PVC3 claims and the packed `u` and columns;
* `fs_hash`: none in Kpre (FS1);
* `prove_fold` and `prove_open`.

`residual` is `prove_s` minus the sum of the phases: Python glue and the in-process copies of the challenges. While
the prover runs, the harness turns off the per-phase contention snapshots of `run_query`'s helpers
(`quiet_contention`), because they read `/proc` inside the whole proof's window. Each timed call gets one snapshot
pair of its own, outside its window. `phases_ok` holds when `|residual| <= 2%` of `prove_s` in the median
repetition. `forward_extra` is the prover's forward pass minus U1: narrowing and keeping the claims, plus, with
`seq`, the loss from B single-prompt passes against one batched pass.

### 3.4 Checks in every cell

A failed check is recorded in the line (`ok: false`), and the script exits with status 1 at the end.

* `u1_equals_claims`: U1's logits, from the warm-up and from every timed repetition, are `torch.equal` to the LM
  head's claims decoded from the proof's PVC3 bytes, as the verifier decodes them (`protocol._decoded_claims`). With
  `seq`, each prompt's proof is compared with its row of the batched U1. This also shows that batched serving and
  per-query proving give the same integers. If U1 runs out of memory in a repetition, the cell's prover lines are
  recorded as skipped.
* `deterministic`: proving the same prompts again gives byte-identical proofs. The check compares the SHA-256 over
  every message, length-prefixed, of the warm-up's proofs and of the first repetition's (prompt 0's with `seq`).
* `accepted`: one untimed `run_query` on prompt 0 (`seq`) or on the batch accepts, with the byte count of the proof
  timed here (`accept_same_bytes`). `--no-accept` skips this check; Kpre then needs no precompute.
* `bytes_consistent`: the proof's byte strings add up to `run_query`'s byte count.

### 3.5 Proof on request

The claims are a deterministic function of the model and the prompt: the integer forward pass is exact. Under
Fiat-Shamir the challenges are a deterministic function of the statement and the claims, and so is every later
message. A provider can therefore serve with U1 and prove later, only for the queries a client asks about. The later
proof is byte-identical to one made at serving time. The `on_request` row reports this case: serving costs U1, and a
proof costs `prove_s` when it is requested, with its SHA-256 and the `deterministic` check. The provider must keep
the prompt, and the model version, until the client's proof window closes.

## 4. Implementation

* `serve_ab.py`: `serve_u1`, `serve_u2`, `prove`, `messages`, `digest`, `claimed_logits`, `prove_batch` (`seq` and
  `joint` via `queries`), `TimedChallenges`, `timed` (CUDA events or `perf_counter`), `boot` (bootstrap), `build`
  (graph, commitment, provers and verifiers per mode), `Cell` (warm-up with checks, interleaved repetitions,
  acceptance), `time_u2`, `_lines` (JSON lines), and `table` (`--table`: the lines as a table, and for a (model, T, B)
  measured on two block counts the two-point extrapolation of U1 and U2 to the full model, as a check against F1
  only). Small shapes for a laptop and the tests are `tiny-gpt`, `tiny-opt`, `tiny-llama` and `tiny-qwen` (as the
  test suites build them) and `small-llama` (d = 256, 4 blocks, vocabulary 4,096).
* `reexec.FloatDecoder(prune_last=True)` (opt-in; `reexec.py` itself is unchanged by default). The last block computes
  k and v at every position, and q, attention without a causal mask, the projection, the residuals and the MLP at the
  last position. U2 then does the same work as the pruned integer graph.
* `sp2027.sh serve`: one job per model (full OPT-1.3B, Qwen3-4B, Llama-2-7B, OPT-6.7B and Llama-2-13B), workloads (a)
  and (b), C and Kpre, `seq` and `joint`, 10 repetitions, U2 in fp16 and bf16. `PVI_SERVE_POLICY` sets mode C's plan
  (default `paper`; the plan PS2 chooses, F6 part 8). `PVI_REHEARSAL=1` submits one small cell (OPT-125M, 2 blocks).
  The smoke tier now also runs `tests/test_serve_ab.py`.

## 5. Tests

`tests/test_serve_ab.py` (laptop: 17 passed, 1 skipped, in 10.3 s; the skipped test needs CUDA):

* `test_u1_logits_are_the_claimed_logits` runs on four families: tiny-llama and tiny-opt with the `paper` policy,
  tiny-qwen with `R16c` and tiny-gpt with `tightc` (lookup tables and col-layout matrices). For both modes, U1's
  logits equal the logits decoded from the proof's bytes. This holds for a batch of 3 proved as one statement, and
  for each prompt proved alone against the batched U1. U1 also equals the claims the forward pass computes.
* `test_proof_on_request_is_byte_identical_and_accepted`, on the same four cases and both modes:
  * a second proof of the same prompts is byte-identical, even after serving and proving another prompt in between;
  * the bytes add up to `run_query`'s count;
  * one changed token gives other bytes;
  * `run_query` accepts the same proof with the same byte count.
* `test_build_zero_layers_is_the_full_model`. An earlier version of the harness built zero blocks for `--layers 0`
  when called directly; this test catches that regression.
* `test_prove_refuses_an_interactive_mode_c_proof`.
* `test_float_decoder_pruned_last_block_gives_the_same_logits`: in float64, the pruned FloatDecoder matches the
  unpruned one within 1e-9 on four families.
* `test_bootstrap_median_and_ratio`.
* `test_cli_lines[cpu]` and `test_cli_lines[cuda]` (the latter is skipped without CUDA). The whole script runs on
  `tiny-llama` with T = 8 and B = 1, 2, `seq` and `joint`. Every `prove` line must pass every check, the timer must be
  the expected one, and the `seq` proof of prompt 0 must have the same SHA-256 as the B = 1 proof.
* `test_out_of_memory_is_skipped_not_failed`: the CLI with every proof of a 2-prompt statement raising
  `torch.cuda.OutOfMemoryError` (simulated on the CPU). The `joint` line says `"skipped": "out of memory"`, the
  `seq` line passes every check, the `on_request` line lists only `seq`, and the status is 0. A `RuntimeError` that
  is not about memory is not taken for one.

## 6. Validation on the laptop (measured)

These are the implementation's runs (before the review; the review's re-runs follow) on a Windows laptop CPU with 2
threads and other programs running (`--reps 10`, U2 in fp32). Mode C on `gpt2` under the `paper` plan would commit a
768 x 50,257 embedding, whose encoding works on chunks of 256 x 262,144 codeword values (about 0.5 GB at 8 bytes
each, before temporaries). That is too much for the laptop's free memory, so `gpt2` ran in Kpre only here (mode C
under `tightc`: below).

| Run | Cells | Prove lines | All checks pass | Later proof byte-identical | Peak host memory |
|---|---|---|---|---|---|
| tiny-gpt, tiny-opt, tiny-llama, tiny-qwen; C, Kpre | 16x1, 16x4, 64x1 | 32 | 32/32 | 32/32 | 216 MB |
| small-llama (4 blocks); C, Kpre | 64x1, 64x4 | 6 | 6/6 | 6/6 | 537 MB |
| gpt2, 1 block; Kpre | 64x1, 64x4 | 3 | 3/3 | 3/3 | 872 MB |

Each of the 41 prove lines has `u1_equals_claims`, `deterministic`, `accepted`, `accept_same_bytes` and
`bytes_consistent` true. The exit status was 0 for all three runs. The logs are
`code/experiments/7_reexec/results/serve_ab_laptop_{tiny,small_llama,gpt2_L1}.jsonl`; their `git` field is the base
commit `cdeb44a`, with this item's changes not yet committed.

**Phase sums.** Median over repetitions of `residual / prove_s`:

| Model | Mode | Cells | Residual, range of medians | Largest single repetition | `phases_ok` |
|---|---|---|---|---|---|
| gpt2 (1 block) | Kpre | 3 | 0.27-0.35% | 0.47% | 3/3 |
| small-llama | C | 3 | 0.76-1.47% | 1.97% | 3/3 |
| small-llama | Kpre | 3 | 0.22-0.30% | 0.33% | 3/3 |
| tiny-* (4 families) | Kpre | 16 | 0.88-1.85% | 2.34% | 16/16 |
| tiny-* (4 families) | C | 16 | 2.10-3.06% | 3.70% | **0/16** |

On the tiny graphs (16 weight ops of d = 64), mode C's residual of about 0.5-1 ms is 2-3% of a 20-170 ms proof. The
residual is the Python work around the timed phases: building the messages' dicts and lists, the in-process copies
of chi and the column indices, and the byte counts. It is under 2% from `small-llama` up (190 ms proofs), and it
falls with model size, because the glue grows with the number of ops and the compute with their size. A separate
diagnostic run (30 proofs each, the median) gave 2.4-3.0% on tiny-llama at T = 16 and 64, and 1.1-1.2% on
small-llama at T = 64. The cluster shapes are expected to pass. The tier checks this; a failure prints a warning and
is recorded as `phases_ok: false`.

**Review re-runs** (same laptop, the commands above, final code: the review changed the GPU timer, the
out-of-memory handling, and checks U1's logits in every repetition instead of the last; `--reps 10` unless noted):

* the tiny run: exit 0, 32/32 lines pass all five checks, 32/32 later proofs byte-identical, peak 215 MB;
  residual medians Kpre 0.86-1.73% (16/16 `phases_ok`), mode C 1.94-2.93% (2/16);
* small-llama: exit 0, 6/6 lines pass, residual medians C 0.84-1.34%, Kpre 0.23-0.30% (6/6), peak 538 MB;
* gpt2, 1 block, Kpre: exit 0, 3/3 lines pass, residual medians 0.27-0.28% (3/3), peak 879 MB;
* gpt2, 1 block, **mode C** under the `tightc` plan (the embedding a lookup table, so its 768 x 50,257 matrix is not
  RS-encoded), 64x1, `--reps 3` (run before the review's code changes, which do not touch the CPU path's timing):
  all five checks pass, residual median 0.36% (largest repetition 0.43%), `phases_ok`; prover/U1 7.76 [5.73,
  10.23], 0.032 MB per token. Its peak host memory was 2,396 MB, above the laptop's budget, so it ran once
  (`results/serve_ab_laptop_gpt2_L1_C_tightc.jsonl`). The 2% criterion therefore
  holds in mode C on a GPT-2 block; it fails only on the 16-op tiny graphs. A profile of tiny-llama's mode C
  residual (0.5 ms of an 18 ms proof) found the in-process message copies (`protocol._passed`, 0.16 ms), the
  verifier's bookkeeping around its decoding, and per-op Python glue, none of it inside a phase's window;
* `messages` rebuilds exactly the bytes that `_fold_message` and `_open_message` sent: the packed `u` and columns
  captured from `claimcodec.pack_field` during `prove` equal the rebuilt ones, for tiny-llama and tiny-opt
  (`paper`), tiny-qwen (`R16c`) and tiny-gpt (`tightc`, `auto`), at B = 1 and 3;
* `reexec.FloatDecoder` without `prune_last` gives bit-identical logits to the committed version (fp32 and bf16, the
  four tiny families and small-llama); with it, the logits agree to 5e-7 (relative) in fp32.

**Timings.** They are not results: the laptop is shared, and U1 for `gpt2` at B = 1 ranged from 47 to 146 ms over
its 10 repetitions (median prover/U1 0.82, 95% interval [0.71, 1.34]). For illustration only, `small-llama` at
T = 64 gave the following (milliseconds; split = the overhead split of Sec. 3.3):

| B | Mode, batch proof | Prover | U1 | U2 (fp32) | Prover/U1 [95% CI] | Split: fwd extra, encode, hash, fold, open, challenges, residual | MB per token |
|---|---|---|---|---|---|---|---|
| 1 | C | 192.0 | 27.8 | 4.6 | 6.90 [6.22, 7.50] | 1.1, 61.4, 5.0, 18.6, 67.5, 5.7, 3.1 | 0.110 |
| 1 | Kpre | 72.9 | 27.8 | 4.6 | 2.62 [2.34, 2.89] | 3.3, 41.7, 0, 0, 0, 0, 0.2 | 0.016 |
| 4 | C, seq | 802.7 | 63.8 | 14.5 | 12.58 [11.44, 13.62] | 58.8, 250.0, 20.7, 76.6, 285.3, 26.1, 11.4 | 0.110 |
| 4 | C, joint | 357.0 | 63.8 | 14.5 | 5.59 [5.03, 6.06] | -1.1, 168.8, 15.9, 19.1, 69.9, 6.0, 2.6 | 0.040 |
| 4 | Kpre, seq | 264.3 | 63.8 | 14.5 | 4.14 [3.81, 4.55] | 50.1, 151.2, 0, 0, 0, 0, 0.8 | 0.016 |
| 4 | Kpre, joint | 195.5 | 63.8 | 14.5 | 3.06 [2.81, 3.36] | -5.5, 136.3, 0, 0, 0, 0, 0.4 | 0.016 |

The structure is as expected:

* `joint` amortises `u` and the opening: mode C at B = 4 needs 0.040 MB per token against 0.110.
* `seq` pays B unbatched forward passes: `forward_extra` is 50-59 ms.
* Kpre's overhead is the claim encoding alone.

A negative `forward_extra` is noise between two medians.

## 7. Cluster runs

### 7.1 2080 Ti pilot (job 1008674: full OPT-1.3B; Llama-2-7B, 1 and 2 blocks)

Random weights, 10 repetitions after a warm-up, medians (95% bootstrap intervals in the logs,
`artifacts/results/9_quality/a7_pilot/*.jsonl` on the server clone s3). Tests: 18 passed; every job exit 0; every line
`ok`, every `prove` line `phases_ok` (residual 0.02-3.9%: the largest in mode C at B = 1, the smallest graphs, as on the
laptop); the only skips are the two `joint` cells of OPT-1.3B at T = 2,048, B = 4 (out of memory, as expected).

**OPT-1.3B, full model, B = 1.**

| T | Mode | Prover s | U1 s | U2 fp16 s | Prover/U1 | Prover/U2 | U1/U2 | MB per token | Egress cap (tok/s per 100 Gb/s) |
|---:|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 64 | C | 1.70 | 0.081 | 0.0084 | 20.9 | 203 | 9.7 | 4.25 | 2,944 |
| 64 | Kpre | 0.151 | 0.081 | 0.0084 | 1.87 | 18.1 | 9.7 | 0.94 | 13,260 |
| 512 | C | 2.01 | 0.122 | 0.032 | 16.4 | 62 | 3.8 | 1.35 | 9,271 |
| 512 | Kpre | 0.505 | 0.122 | 0.032 | 4.12 | 15.6 | 3.8 | 0.93 | 13,420 |
| 2,048 | C | 4.37 | 0.512 | 0.108 | 8.55 | 40.3 | 4.7 | 1.09 | 11,460 |
| 2,048 | Kpre | 2.08 | 0.512 | 0.108 | 4.07 | 19.2 | 4.7 | 0.98 | 12,710 |

**OPT-1.3B, batches** (B prompts; `seq` proves them one after another, `joint` as one batch proof).

| T | B | Mode | seq: Prover/U1, /U2 | joint: Prover s, /U1, /U2 | joint MB per token |
|---:|---:|---|---|---|---:|
| 64 | 4 | C | 59.3, 387 | 1.57, 17.1, 111 | 1.77 |
| 64 | 4 | Kpre | 6.54, 42.7 | 0.312, 3.41, 22.2 | 0.94 |
| 64 | 16 | C | 110, 421 | 2.44, 12.9, 49.5 | 1.15 |
| 64 | 16 | Kpre | 12.3, 47.2 | 0.876, 4.62, 17.7 | 0.94 |
| 64 | 64 | C | 122, 452 | 6.38, 8.96, 33.3 | 1.00 |
| 64 | 64 | Kpre | 13.6, 50.5 | 3.39, 4.76, 17.7 | 0.94 |
| 2,048 | 4 | C | 8.21, 39.6 | out of memory | |
| 2,048 | 4 | Kpre | 4.05, 19.5 | out of memory | |

**Llama-2-7B, 32 blocks, extrapolated** from the 1- and 2-block cells as `t = t1 + 31 (t2 - t1)` (one pruned block
plus 31 full ones; prover, U1 and U2 alike).

| T | B | Mode, how | Prover s | U1 s | U2 fp16 s | Prover/U1 | Prover/U2 |
|---:|---:|---|---:|---:|---:|---:|---:|
| 64 | 1 | C | 5.25 | 0.145 | 0.034 | 36.1 | 154 |
| 64 | 1 | Kpre | 0.333 | 0.145 | 0.034 | 2.29 | 9.8 |
| 512 | 1 | C | 5.57 | 0.398 | 0.173 | 14.0 | 32.2 |
| 512 | 1 | Kpre | 1.84 | 0.398 | 0.173 | 4.62 | 10.6 |
| 2,048 | 1 | C | 12.1 | 1.65 | 0.642 | 7.35 | 18.9 |
| 2,048 | 1 | Kpre | 6.27 | 1.65 | 0.642 | 3.80 | 9.8 |
| 64 | 64 | C joint | 20.5 | 2.65 | 1.01 | 7.74 | 20.3 |
| 64 | 64 | Kpre joint | 12.2 | 2.65 | 1.01 | 4.62 | 12.1 |
| 2,048 | 4 | C joint | 39.4 | 7.26 | 2.16 | 5.42 | 18.3 |
| 2,048 | 4 | Kpre joint | 24.3 | 7.26 | 2.16 | 3.34 | 11.3 |

**U2 check against F1 (the pilot's 10% criterion): not met, in both directions.**

* OPT-1.3B at T = 2,048, B = 1: U2 fp16 108 ms measured on the full model, against F1's 0.18 s (40% lower).
* Llama-2-7B at T = 2,048, B = 1: 642 ms extrapolated, against F1's 0.57 s (13% higher).

F1's figures are two-point extrapolations from 1- and 2-block builds (F1 Sec. 2), on another node, so a full-model
measurement can differ by more than 10%; the A7 extrapolation for Llama-2-7B uses the same two-point method and has the
same weakness. The OPT-1.3B figure is a direct measurement and is the one quoted. A faster U2 makes prover/U2 larger,
so the OPT ratios are, if anything, unfavourable to us; for Llama-2-7B, prover/U2 may be up to about 13% too low.
The L40S tier measures the 7B models in full.

**Reading.**

* **The integer engine's floor (U1/U2) is 4-10x** on OPT-1.3B (9.7x at 64 tokens, 4.7x at 2,048), above the plan's
  model (1.8-2.6x for Llama-2-7B), and 2.3-4.3x on Llama-2-7B (extrapolated). It is paid on every query if the provider
  serves the integer model; it is the cost of exactness, not of the proof.
* **The proof's own cost (Prover/U1)** at B = 1 is 1.9-4.1x in Kpre and 8.6-21x in mode C on OPT-1.3B; 2.3-4.6x
  and 7.4-36x on Llama-2-7B. Mode C at short prompts is dominated by the column opening of every weight matrix, a
  per-query constant.
* **Against fp16 eager (Prover/U2)** at B = 1: 16-19x in Kpre and 40-203x in mode C on OPT-1.3B; about 10x in Kpre
  and 19-154x in mode C on Llama-2-7B. The plan's model was 9-40x at B = 1.
* **Batching.** One joint proof for a batch removes most of mode C's constant: at T = 64 and B = 64, mode C falls
  from 122x U1 (sequential) to 9.0x (joint) on OPT-1.3B and Kpre to 4.8x; against fp16, 33x and 18x. The plan's
  3-8x at B >= 8 holds against U1 (Kpre 4.6-4.8x, mode C 9.0-13x on OPT-1.3B) but not against fp16.
* **Proof bytes per token** are about 1 MB on OPT-1.3B (0.94 in Kpre) and about 3.1 MB on Llama-2-7B (extrapolated),
  so 100 Gb/s of egress carries about 13,000 and 4,000 proved tokens per second.
* **For the paper** (body, filled 11 Oct): OPT-1.3B at 2,048 tokens, B = 1, the same GPU: the prover costs 4.1x (Kpre)
  and 8.6x (C) our exact forward pass, and 19x and 40x an fp16 one; 64 prompts of 64 tokens in one Kpre proof, 4.8x and
  18x. These replace the cross-GPU comparison; the L40S `serve` tier replaces them in turn.

### 7.2 L40S `serve` tier

The tier covers full OPT-1.3B, Qwen3-4B, Llama-2-7B, OPT-6.7B and Llama-2-13B, with U2 in fp16 and bf16. Results:
[CLUSTER] (`serve_ab.py --table logs/l40s_sp2027/sp-serve-*.out`). The paper needs one column in tab:options (the
provider overhead) and the `serve` table in the extended version. The `\todo`s at evaluation:100, intro:111 and
discussion:30 are filled from this tier.

The plan expects the following (model, not measured here):

* 9-40x over fp16 eager at B = 1;
* a floor from the integer engine (U1/U2) of about 1.8x at 2,048 tokens and 2.6x at 64 tokens for Llama-2-7B on the
  2080 Ti;
* 3-8x at B >= 8 with batching.

The ratios are reported as measured, whatever they are.

## 8. Limitations

* **U2 is eager PyTorch**: no CUDA graphs, `torch.compile`, paged attention or fused kernels. A serving engine is
  faster, so prover/U2 against vLLM would be larger. That is A7-vLLM (U3), which needs the venv approval. U2 skips
  the rotary embedding, Qwen3's q/k norms and OPT's learned positions (element-wise costs), so it is if anything
  faster than a complete implementation: the comparison is conservative for us.
* **Batches**. `seq` proves the B queries one after another with unbatched forward passes, because a batched forward
  pass with per-query proofs (C4c) is not implemented. `joint` is one client's batch, and its claims for all B
  prompts must fit on the GPU at once. On the 7B and 13B models, joint at T = 2,048 with B = 4 (about 45 GB of int32
  claims for Llama-2-7B) and the largest T = 64 batches do not fit, and are recorded as skipped.
* **Prefill only**. Generation (C-Q5) is answered in text from E1's cells: one prefill over x‖y after decoding. The
  KV-cached integer decoder is A10 (could).
* **Proof on request** is checked on one machine, in one process. Across machines, the claims are exact integers on
  every device (the exactness tests) and PVC3 has the same bytes on host and device (the codec tests), so the bytes
  should agree, but `serve_ab` does not test that (D6a covers the cross-machine digest).
* **The 2% phase criterion** fails in mode C on the tiniest graphs (1.9-3.1% on the laptop). It passes from
  `small-llama` up, and in mode C on one GPT-2 block (0.36%, Sec. 6).
* **The acceptance check** decodes each warm-up proof's claims on the host to compare the logits: for Llama-2-7B at
  2,048 tokens that is about 11 GB of int32 per proof, freed at once. With `--accept`, it also runs one CPU
  verification per variant and cell. The tier's `--mem` (128-256 GB) covers both. `--no-accept` skips the
  verification, but not the logit check.
* **The egress cap** counts proof bytes only: no protocol framing, TCP or TLS overhead.
* **Laptop scope**. The laptop ran only CPU cells on graphs of up to one GPT-2 block. CUDA events, the GPU encoder
  path, a real GPU running out of memory and every ratio on real shapes are untested until the pilot runs (the
  skip path itself is tested with a simulated `OutOfMemoryError`, Sec. 5).
* **Triton on the TAU nodes**. The integer forward pass (U1 and the prover) runs its attention as a Triton kernel on
  a GPU; the TAU compute nodes lack the Python headers Triton's launcher needs, so the pilot exports
  `PVI_PY_INCLUDE` and a node-local `TRITON_CACHE_DIR` (Sec. 9, `I_server_setup.md` Sec. 3).
* **Concurrent jobs**. The tier's five jobs have different names, so SLURM may run two of them at once on a node
  with several GPUs; their CPU work would then contend. The run package asks for exclusive use of the node, and
  every line records `wait_frac_max`.

## 9. How to run

From `code/` with `PYTHONPATH=src` and the project's Python (bash):

    # laptop: small graphs, CPU, 2 threads
    OMP_NUM_THREADS=2 python experiments/7_reexec/serve_ab.py --model tiny-llama small-llama --cells 16x1 16x4 64x1 \
        --batch-proof seq joint --device cpu --threads 2
    # one GPU cell set, both batch proofs
    python experiments/7_reexec/serve_ab.py --model opt-1.3b --workload a b --batch-proof seq joint --device cuda --u2-dtype fp16
    # read the lines
    python experiments/7_reexec/serve_ab.py --table '../logs/a7_pilot/*.jsonl'
    # the tier (L40S; from the repository root)
    bash code/experiments/4_defence_benchmark/slurm/sp2027.sh serve

2080 Ti pilot (Xeon node with 250 GB, in bash, from `code/`; random weights, so nothing is downloaded and `HF_HOME`
is unused). The integer forward pass runs its attention as a Triton kernel, and the TAU compute nodes have neither
the Python headers Triton's launcher needs nor a visible `$HOME` for its cache: `PVI_PY_INCLUDE` names the copy of
the login node's `python3.12` headers (`I_server_setup.md` Sec. 3: `pyinclude/python3.12` in the sp2027 server
folder `logs/validation/sp2027`), and the cache goes to node-local `/tmp`.

    export PVI_PY_INCLUDE=<the sp2027 server folder>/pyinclude/python3.12 TRITON_CACHE_DIR=/tmp/$USER-triton
    export PYTHONPATH=src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NVIDIA_TF32_OVERRIDE=0 \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    mkdir -p ../logs/a7_pilot
    python -m pytest -q -o addopts= -p no:cacheprovider tests/test_serve_ab.py
    python experiments/7_reexec/serve_ab.py --model opt-1.3b --workload a b --modes C Kpre \
        --batch-proof seq joint --reps 10 --device cuda --threads 8 --u2-dtype fp16 --label pilot-2080ti \
        > ../logs/a7_pilot/opt13.jsonl 2> ../logs/a7_pilot/opt13.err; echo exit=$?
    for L in 1 2; do python experiments/7_reexec/serve_ab.py --model llama2-7b --layers $L --workload a b \
        --modes C Kpre --batch-proof seq joint --reps 10 --device cuda --threads 8 --u2-dtype fp16 \
        --label pilot-2080ti > ../logs/a7_pilot/llama7_L$L.jsonl 2> ../logs/a7_pilot/llama7_L$L.err; echo exit=$?; done
    python experiments/7_reexec/serve_ab.py --table '../logs/a7_pilot/*.jsonl'

The pilot passes when all of the following hold:

* the tests report 18 passed (the CUDA test runs on the GPU node);
* every job exits with status 0, and no line has `"ok": false`;
* every `prove` line that was not skipped has `"phases_ok": true`;
* `skipped` appears only as `"out of memory"`, and only for `joint` cells (expected at T = 2,048, B = 4 for OPT-1.3B);
* U2 fp16 at T = 2,048, B = 1 is 0.162-0.198 s for OPT-1.3B. The table's extrapolation line for Llama-2-7B gives
  0.513-0.627 s (F1's 0.18 s and 0.57 s, within 10%; F1's decoder was not pruned, so U2 is expected a few percent
  below them).

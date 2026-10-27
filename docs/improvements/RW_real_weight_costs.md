# RW: the cost measurements on the real checkpoints

Code: commit 65ac80b (branch `sp2027-artifact`). Module `experiments/common/real_cells.py`; option `--real` of every
cost script; the rule in `experiments/package/run.sh`; the selectors in `experiments/common/records.py` and
`experiments/paper/`.

## 1. Why

The cost measurements (prover and verifier times, proof bytes, setup, serving, generation and the re-execution
baselines) used random int8 weights of each model's shapes. The costs depend mainly on the shapes. The exceptions are
the compressed size of the claims and V1's exceptions, which depend on the weight values. Random weights were chosen
because the first real-weight builders existed only for OPT, and because a random model needs no download or
calibration. Both reasons no longer hold: the integer builders of Llama-2, Qwen3 and OPT now build every benchmark
cell from its checkpoint. A reviewer may reasonably ask why a paper about certified inference of real models measures
models that no one would deploy. Every cost cell with a usable checkpoint therefore now uses it.

## 2. The rule

One rule decides the weights of a cell, implemented once in Python (`real_cells.main_checkpoint`) and once in
`run.sh` (`rw`). A test checks that they agree over the whole dry run of the final run, with and without the licence
and with `--random-weights`.

* **Real:** Llama-2-7B and Llama-2-13B (`--llama-licence`), OPT-125M, OPT-1.3B, OPT-6.7B and Qwen3-4B-Base, whenever
  the prompt, the fixed prefix of the main recipe (at most 64 tokens) and the generated tokens fit the checkpoint's
  trained context. For Llama-2 (4,096 tokens) this is every prompt up to 2,048 tokens.
* **Random, with the reason stated in the paper:**
  * GPT-2, which has no integer builder for its checkpoint;
  * the long-prompt curve (4,096 to 16,384 tokens) and the codec curve, which go beyond Llama-2's trained context
    and are kept on one footing. Its 2,048-token point is measured on random weights too, so the fitted curve is never
    compared with real-weight measurements;
  * the real tier's real-against-random twins, which measure how much the values matter;
  * re-execution of the first two blocks (the full model is real).
* **Without the licence**, every Llama-2 cell is random; OPT and Qwen3-4B stay real.
* `--random-weights` restores the previous all-random run. It is kept as an option, not the default.

## 3. Implementation

* `real_cells.py` defines `--real` with `--recipe`, `--smooth` and `--calib-seq`, loads the checkpoint, and refuses
  every case that would silently measure something else: no builder, a legacy recipe, a `--prob-max` other than the
  paper's, a block count outside the model's, a prompt beyond the context (checked before and after the load) and a
  checkpoint of other shapes. torch is imported only where a checkpoint is loaded.
* Users: `perf.py`, `v1_bytes.py`, `gen_bench.py`, `bench.py`, `setup_timing.py`, `serve_ab.py`, `reexec.py`,
  `decode_baseline.py`, the five improvement harnesses and `v1_phases.py`. `llamacpp.py` has its own torch-free
  `--real`: it converts the safetensors directly, byte-identical to llama.cpp's converter on a test model.
* **Partial builds** (`--layers`, `--builds`) of Llama-family and OPT checkpoints build the full graph's first N
  blocks. The residual scale is set on the whole checkpoint, which `hf_opt.build_opt_from_hf(n_layers=)` now accepts.
  A test checks the blocks against the full graph, including a case where a checkpoint cut to N blocks would get a
  different scale.
* **Records** carry the checkpoint's name in `real`. The selectors take a record only if its weights equal the
  rule's, so real and random records of one cell never mix. `numbers.py` gains `--weights` and `--llama-licence`
  (both inferred from the records by default); `--check` fails on a missing real record instead of falling back to a
  random one.

## 4. Cost of the run

| Run | Before | After |
|---|---|---|
| `--optional --llama-licence all` | 243 jobs, 543 h of job limits | 245 jobs, 664 h |
| `--llama-licence all` | 217 jobs, 476 h | 217 jobs, 593 h |
| `all`, no licence | 198 jobs, 390 h | 198 jobs, 416 h |

* **Time limits:** a real job gets one more hour (7B class) or two (13B), plus one more for a real re-execution job.
  This rests on an estimate of 0.2–0.5 h of loading and calibration per 7B job (0.4–1 h for 13B), to be checked
  against the final run's telemetry.
* **Memory:** at least 128 GB of host memory (256 GB for Llama-2-13B).
* **Downloads:** about 40 GB for Llama-2 and 24 GB for OPT-6.7B and Qwen3-4B, into `HF_HOME`; the pre-flight now
  checks `HF_HOME` on every tier that downloads.
* **Submissions:** the one-job plan has fifteen submissions instead of thirteen.

## 5. Paper

The evaluation now says that costs use the real checkpoints with the integer recipes of the quality section. It also
says that GPT-2 and prompts beyond a checkpoint's trained context use random int8 weights of the same shapes. The
abstract, introduction, evaluation, discussion and generation sections and their extended versions were reworded to
match. The macro names (`\VOneRatioRandom` and others) are unchanged, although `\VOneRatioRandom` now holds
real-weight values. `fallback/numbers.tex` holds the previous random-weight values until the final run regenerates
it.

## 6. Tests

New tests: `test_real_cells.py`, `test_real_baselines.py` and `test_real_weights_scripts.py`. There are also new tests
in `test_run_package.py` (the rule's placement against the Python rule over the dry run), `test_artifact_records.py`
(selection by weights) and `test_hf_opt.py` (OPT partial builds). The full CPU suite on the cluster passes at 65ac80b
(2,256 passed, 387 skipped).

## 7. What remains

* Check the load and calibration estimate, and the time limits, against the final run's telemetry.
* Check that "values matter little" still fits the real tier's final twins.

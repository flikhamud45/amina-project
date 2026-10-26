# Stored benchmark measurements

This folder holds the raw benchmark records, the tables derived from them, and the catalogue of
published results. The scripts of [`experiments/comparison`](../../experiments/comparison/README.md) derive the
tables and counts from these files, and [`experiments/paper/numbers.py`](../../experiments/paper/README.md) reads a
platform's tables for the paper; the records themselves are written by
[`experiments/costs/bench.py`](../../experiments/costs/README.md). The stored roots below hold the values the paper
prints before the final run (`paper_sp2027/fallback/`); the final run writes its own root `raw_<name>/`.

| Path | Contents |
|---|---|
| `raw_<platform>/<suite>/<model>/<cell>.jsonl` | raw records, one JSON line per measurement; `<suite>` is `cnn` or `llm` |
| `raw_<platform>/<suite>/<model>/<cell>.jsonl.part.<timestamp>` | the records of an interrupted attempt at a cell, set aside by `bench.py` when the cell was restarted (`<timestamp>`: the Unix time of the restart); kept on purpose, and ignored by the aggregates and counts, which read only finished cells' `<cell>.jsonl` (19 files: 2 in `raw_l40s/`, 17 in `raw_l40s_improved2/`) |
| `raw_<platform>/PLATFORM.json` | the root's GPU, CPU, driver and torch version; `"frozen": true` makes `bench.py` refuse new records |
| `tables_<platform>/` | tables derived from `raw_<platform>/` by `aggregate.py --platform <platform>` |
| `tables/` | `reported_curated.csv` (the published results the paper compares with) and `analytic.csv` (the path protocol's cost on Llama-2-7B), shared by every platform; also the derived tables of `raw/` |
| `literature/reported_benchmarks.csv` | 413 published measurements from 32 systems, each with its table, page and a verbatim snippet of the source |
| `excluded_cells.csv` | timings withheld from the paper (see below) |

Every record carries the git commit (`git_sha`), host, GPU and CPU it was measured on; the first
record of each cell (`metric: env`) also holds the SLURM job, the Python, torch and CUDA versions,
the thread count and the CPU affinity. Each root holds the records of one GPU model and one CPU
model: `bench.py` refuses records from a second CPU model.

Records written from commit `7016557` on also carry contention evidence: every timing row has a
`contention` field, this process's run-queue wait and on-CPU time while the phase ran, its context
switches and the load average (`wall_s`, `run_ns`, `wait_ns`, `wait_frac`, `nivcsw`, `nvcsw`, `cpu_s`,
`threads`, `threads_exited`, `load_start`, `load_end`, `segments`; see the records of
[`experiments/costs`](../../experiments/costs/README.md)), and the `env` record
has `schedstat`. The aggregates and the paper's generators ignore these fields. The records of
`raw_l40s_improved3/` (the re-measurement of the withheld cells) are the first to have them.

## The roots

The record counts below are those printed by `count_outcomes.py --platform <platform>` (they exclude
the one `env` record per cell).

| Root | Records | Protocol | Hardware | Code | Values printed before the final run |
|---|---:|---|---|---|---|
| `raw_l40s/` | 123,832 | basic | NVIDIA L40S (48 GB) prover, 8 threads of an AMD EPYC 9334 verifier; nodes n-801 to n-804 | `9401431` | the image models' attack cells (`attack_float`, `sampling`) behind `sec:break`'s single-neuron flip and the path test's detection and paths (`\CnnPathMax`, `\CnnPathsRange`, ...); the reference root of `app:eval-setup`'s hardware independence |
| `raw_l40s_improved/` | 68,363 | basic proofs with the released code, and optimised variants | as `raw_l40s/`; nodes n-801, n-803, n-804 | `c8be5eb` | optimised cells not repeated in `raw_l40s_improved2/` (e.g. `tab:eval-compare`'s Maverick row, without the compact encoding) |
| `raw_l40s_improved2/` | 83,896 | optimised, and GPT-2's basic proofs with the released code | as `raw_l40s/`; nodes n-801, n-803, n-804, n-805 | `366e3d4`, `c2bdc49` | the optimised protocol: `tab:eval-llm`, the "ours" columns of `tab:eval-compare`, the outcome counts of `sec:eval-setup` |
| `raw_l40s_improved3/` | 1,600 | optimised: the re-measurement of the cells of `raw_l40s_improved2/` whose timings are withheld (the cells it finished) | as `raw_l40s/`; nodes n-803, n-805 (n-801 and n-804 excluded) | `7016557` | replaces the same cells of `raw_l40s_improved2/`: the withheld values it re-measured (OPT-6.7B's GPU verifier and Llama-2-13B's Kpre CPU verifier at 2,048 tokens, `tab:eval-llm`) |
| `raw_l40s_improved2_thr1/` | 6,104 | optimised, one verifier thread | as `raw_l40s/`; node n-805 | `366e3d4` | not used |
| `raw_rtx2080ti-v2/` | 78,266 | basic | NVIDIA RTX 2080 Ti (11 GB) prover, 8 threads of an Intel Xeon Silver 4114 verifier; nodes s-004, s-005 | `9401431` | `app:eval-setup` ("Proof sizes do not depend on the hardware"): its 1,978 hardware-independent values equal those of `raw_l40s/` |
| `raw/` | 59,547 | basic (an earlier version of the code) | as `raw_rtx2080ti-v2/` | several earlier commits | not used; the reference of an earlier smoke test's fingerprint check |

`raw_l40s/`, `raw_l40s_improved/`, `raw_rtx2080ti-v2/` and `raw/` are frozen. A new measurement uses
a new platform name (`python -m experiments.costs.bench ... --platform <name>`, or
`bash code/experiments/package/run.sh --platform <name>`), so it never mixes with the stored records.

### `raw_l40s/`: the basic protocol

Every job of the `must`, `should` and `nice` tiers of the benchmark's tier script, which is not part of this
artifact (its cells are those of `experiments/costs/bench.py`, and the run package `experiments/package/run.sh` runs
them), on the L40S partition of a SLURM cluster, with platform `l40s`: 61 SLURM jobs with records (jobs pre-empted by
the partition were resubmitted by re-running their tier). The one L40S node with a different CPU
(t-806) was excluded. Every decoder up to 13B parameters is built at full depth (`--lean`) at up to
2,048 tokens, Llama-2-7B also at 4,096. The 30–70B shapes (OPT-30B, OPT-66B, Llama-2-70B) are built
with 1 and 2 blocks only; the full Llama-2-70B build (64.2 GiB of int8 weights) does not fit and is
skipped with a `SKIP` line. Besides the headline cells the root holds variants of the same protocol:
`_gpuv` (the verifier on the GPU), `_batch` (8 and 32 prompts per proof; 8 to 256 images for the CNNs,
8 to 128 for the 224-pixel ResNet), `_thr1` (one verifier thread), and the controls `_thr12`, `_tf32`,
`_nolean` and `_nofix` (the weight cache of `raw/`, which the code of commit `9401431` restored on request and the
current code no longer has). `count_outcomes.py
--platform l40s` counts every cell: 8,709 honest queries accepted and 2,895 attacks rejected (2,091 on
image models, 804 on language models).

### `raw_l40s_improved/`, `raw_l40s_improved2/` and `raw_l40s_improved3/`: the optimised protocol

The optimised protocol was measured in two batches on the hardware of `raw_l40s/`, and the cells of
the second whose timings are withheld are re-measured in a third. The values the paper prints before the
final run were read from the three batches together: where several hold a cell, the latest batch's is used, the
whole cell (its rows are never mixed with an earlier batch's), and a full-model value comes from the batch that has
the full-depth build: a later batch's 1- and 2-block builds do not replace an earlier batch's full build of the
same model and setting. The cells of the final run are defined in `experiments/paper/bench.py` (`CELLS`), and
`python -m experiments.comparison.count_outcomes --platform l40s_improved,l40s_improved2,l40s_improved3` reads the
three batches with the same precedence.

* **`raw_l40s_improved/`** (platform `l40s_improved`): 26 benchmark jobs (their cells are those of
  [`experiments/costs/bench.py`](../../experiments/costs/README.md)) and
  one job that ran the test suite on the L40S. Its untagged cells are the basic protocol run by the
  released code: `python -m experiments.comparison.fingerprint_check artifacts/comparison/raw_l40s
  artifacts/comparison/raw_l40s_improved`
  finds its 565 hardware-independent values (proof bytes, verdicts, parameters) equal to those of
  `raw_l40s/`, while its timings include the verifier engineering. The other cells carry the tags of
  the optimised options: `_polauto` (the planning rule), `_wire` (the compact encoding), `_prune`,
  `_lookups`, `_thr1` (Qwen3-4B in Maverick's setting, one verifier thread) and `_gpuv_stream` (the
  streaming GPU verifier). Over all its cells: 4,380 honest queries accepted and 4,973 attacks rejected.
* **`raw_l40s_improved2/`** (platform `l40s_improved2`): 79 SLURM jobs that complete the definition's cells
  for every model of `tab:eval-llm` and `tab:eval-compare` (and the 30–70B shapes, with 1 and 2 blocks),
  with the streaming GPU verifier at 64 and 2,048 tokens, and that re-measure GPT-2's basic proofs
  with the released code. Over all its cells: 5,088 honest queries accepted and 2,518 attacks rejected.
* **`raw_l40s_improved3/`** (platform `l40s_improved3`): the re-measurement of the second batch's jobs
  whose timings `excluded_cells.csv` withholds (below), each re-run whole, on the L40S nodes other than
  n-801 and n-804, at commit `7016557` (`c2bdc49` plus the contention evidence: every proof byte,
  verdict and cell name is `c2bdc49`'s): 10 SLURM jobs, the 8 with a withheld value the paper would print
  and 2 whose only withheld value is a setup time. **Incomplete:** the root holds the 22 cells that had
  finished when it was copied from the run's root (cells with a `.done` marker only): the jobs
  `o2-llama13-2k-k`, `o2-opt13b-2k-k` and `o2-opt67-2k-cg` in full, the 1- and 2-block cells of
  `o2-llama13-2k-c`, `o2-llama7-2k-c` and `o2-opt13b-2k-c`, and the full-depth commitments of
  `o2-llama7-2k-c` and `o2-llama13-2k-cg`. No timed phase longer than 1 s waited on the run queue
  for more than 0.1% of its runnable time, and `python -m experiments.comparison.fingerprint_check
  artifacts/comparison/raw_l40s_improved2 artifacts/comparison/raw_l40s_improved3` finds its 72
  shared hardware-independent values equal. Over its cells: 90 honest queries accepted and 13
  attacks rejected.
* **`raw_l40s_improved2_thr1/`**: LeNet-5 and VGG-16 in mode C under the optimised protocol with one
  verifier thread (2 SLURM jobs), for a single-core comparison with zkCNN's verifier. The paper does not
  use it: `tab:eval-compare` compares the 8-thread verifier and says so.

The outcome counts of `sec:eval-setup` printed before the final run count only the definition's cells
of the batches (a cell of `raw_l40s_improved3/` replaces the same cell of `raw_l40s_improved2/`, the
same seeded instances, so the totals are those of the first two batches):

```bash
python -m experiments.comparison.count_outcomes --platform l40s_improved,l40s_improved2,l40s_improved3 --definition \
    --policy auto --prob-max 255
# 61,629 records; 3,792 honest queries accepted (2,196 image, 1,596 language); 2,512 attacks rejected
# (1,848 image, 664 language): Freivalds 1,601, column code check 853, Merkle check 56, range check 2
```

### Withheld timings: `excluded_cells.csv`

During parts of the second batch, nodes n-801 and n-804 were slowed by other users' load (the jobs'
threads were not pinned), which inflated CPU-side stages such as proof encoding and decoding and the
commitment by up to about 100x. Proof bytes and verdicts are unaffected. `excluded_cells.csv` lists
the 17 values of such cells that are withheld: a value is listed if its cell ran on n-801 or n-804
and it exceeds its reference by the thresholds stated in the file's header. The values the paper prints before
the final run leave these values out; no script of the artifact reads the file. The raw records stay in
`raw_l40s_improved2/`. The cells were
re-measured on the other nodes as platform `l40s_improved3` (above); the exclusion is per platform, so a
value re-measured there replaces the withheld one and is shown in the paper. 6 of the 17 values were
re-measured: OPT-6.7B's prover, verifier and commitment with the streaming GPU verifier, Llama-2-7B's
commitment and the Kpre CPU verifiers of Llama-2-13B and OPT-13B, all at 2,048 tokens. The other 11 were
not re-measured: the values printed before the final run leave them out rather than fill them in (`--` in
`tab:eval-llm`, `\todo` in `tab:eval-compare`), as they do the two optional cells that were not run (the
streaming GPU verifier of Qwen3-4B and Llama-2-13B at 64 tokens): 10 numbers in all.

### `raw_rtx2080ti-v2/`: the second platform

The `must` tier of the same tier script without `ab-opt13` (a non-lean OPT-1.3B control at 2,048 tokens,
about 27 GiB, too large for the card): 29 SLURM jobs on the RTX 2080 Ti partition of a SLURM cluster, with
platform `rtx2080ti-v2`. Every decoder is built at full depth except Llama-2-13B,
whose 12.1 GiB of int8 weights exceed the card. `count_outcomes.py --platform rtx2080ti-v2`: 5,547
honest queries accepted, 2,456 attacks rejected. It lacks cells that the paper's tables draw (e.g. OPT-13B at
2,048 tokens), so no value of the paper comes from it.

### `raw/`: an earlier run

An earlier benchmark run on the RTX 2080 Ti with an earlier version of the code. Its committed-weights
(C) prover re-uploaded the weights to the GPU twice per query (a weight cache keyed `cuda` against
`cuda:0`), so its C prover times are too slow; the `_nofix` controls of `raw_rtx2080ti-v2/` measure the
effect on the same GPU (e.g. Llama-2-7B at 64 tokens 11.0 s against 7.9 s). Every decoder above 12
blocks was built only with 1 and 2 blocks. The paper does not use it. It was the reference of an earlier
smoke test's fingerprint check (`python -m experiments.comparison.fingerprint_check` compares two roots), and its
tables in `tables/` were derived by an earlier version of `aggregate.py`, which also added the expected multiproof
size next to its language-model records (they sent one Merkle path per opened column); the current `aggregate.py`
reads the roots `raw_<platform>/` only.

## Derived tables

`python -m experiments.comparison.aggregate --platform <p>` writes three files into `tables_<p>/`:

* `measured_summary.csv`: the median and mean of every metric per model, cell and stage;
* `llm_full_model.csv`: the full-model cost of each language model (the full build where there is
  one, otherwise extrapolated linearly from the 1- and 2-block builds, labelled
  `extrapolated_from_1_and_2_blocks`);
* `llm_extrapolation_check.csv`: every full build against its 1- and 2-block extrapolation.

The stored `llm_extrapolation_validation.csv` files were written by a validation script that is not part of the
artifact. Published numbers are shown as reported, on their own hardware, and never mixed with ours.

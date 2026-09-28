# Code: Certified but Compromised

Everything behind the report (`../report/main.pdf`):

1. a from-scratch reimplementation of the path-sampling proof of inference of Anchuri
   et al. (SaTML 2026);
2. our single-neuron attack and backdoor on it;
3. the analysis of smarter sampling rules;
4. our defence, which checks every weight layer with Freivalds' algorithm against a
   Reed–Solomon/Merkle commitment to the weights;
5. the benchmark and the comparison with the literature.

All results are stored in the repository. Every figure and generated table
(`report/figures`, `report/tables`) is rebuilt from the stored benchmark records in a
minute, with no GPU (Route A). Table 1 and the MNIST numbers of §3.2 and §4.2 are in
`artifacts/results/*.json`, which Route B re-creates in about 45 minutes on a CPU.
Everything can also be re-measured from scratch.

## Layout

```
code/
  src/pvi/                 the library (see src/pvi/README.md)
  experiments/             one folder per result; each has a README
    0_train_models/        train the MNIST models used by 1-4        CPU   ~2 min
    1_reproduction/        the original protocol on its threat model  CPU   ~1 min
    2_attack/              the single-neuron attack and backdoor      CPU   ~2 min
    3_sampling_fixes/      smarter samplers, and why they fail        CPU  ~40 min
    4_defence_benchmark/   our defence vs. the path test, CNNs + LLMs GPU  hours (SLURM)
    5_comparison/          tables, counts and every report figure     CPU   ~1 min
  artifacts/models/        the MNIST models the report used (committed, with MODELS.sha256)
  artifacts/results/       the JSON results of experiments 1-3 (Table 1, §3.2, §4.2)
  artifacts/comparison/    the benchmark's raw records, derived tables and the literature
  tests/                   234 tests (44 of them need a GPU)
```

The experiments import the library (`pvi`) and never each other. Times are for 8 CPU
threads.

## Installation

Python 3.11 or newer for the pinned versions in `requirements.txt` (the results were
produced with Python 3.12; `pip install -e code` alone, with the ranges in
`pyproject.toml`, also works on 3.10). From the repository root:

```bash
python3 -m venv .venv
.venv/bin/pip install -r code/requirements.txt   # exact versions used for the results
.venv/bin/pip install -e code                     # the pvi package
```

On the TAU SLURM cluster the default PyTorch wheel does not match the drivers, so run
this before the two commands above:

```bash
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
```

All commands below are run from `code/` with the virtual environment active
(`source ../.venv/bin/activate`), except the SLURM commands, which are run from the
repository root.

**Data.** MNIST is downloaded to `code/data/` on first use. The benchmark also reads
CIFAR-10 from `$PVI_CIFAR_ROOT` (torchvision's `cifar-10-batches-py/` layout) and
ImageNet from `$PVI_IMAGENET_ROOT`, for the dog/cat and dog/squirrel classifiers; both
default to the course's copies on the cluster. `$PVI_IMAGENET_ROOT` must contain
`train/` and `val/` with one folder per class, named by the class's 1-based index in
sorted-synset order (dogs 152–269, cats 282–286, fox squirrel 336). Decoded images are
cached in `artifacts/fullcheck/cache/` (or `$PVI_CACHE`).

## Reproducing the report

### Route A: from the stored measurements (no GPU, about a minute)

The report's benchmark numbers (§4.2 on the CNNs, §4.3–4.4, Figures 3–5, Tables 2–4)
come from `artifacts/comparison/raw_rtx2080ti-v2/`, the 78,266 raw records of platform
`rtx2080ti-v2` (see *The stored benchmark runs* below). These commands rebuild every
derived table, the report's figures and generated tables, and the PDF:

```bash
python experiments/5_comparison/aggregate.py --platform rtx2080ti-v2       # raw_rtx2080ti-v2/ -> tables_rtx2080ti-v2/measured_summary.csv, llm_full_model.csv
python experiments/5_comparison/literature.py                              # published numbers -> tables/reported_curated.csv
python experiments/5_comparison/analytic.py                                # the path test's cost of 2^-40 on Llama-2-7B -> tables/analytic.csv
python experiments/5_comparison/count_outcomes.py --platform rtx2080ti-v2  # the soundness counts of Section 4.3
python experiments/5_comparison/paper_assets.py --platform rtx2080ti-v2    # -> ../report/figures/*.pdf, ../report/tables/*.tex
python experiments/5_comparison/text_numbers.py --platform rtx2080ti-v2    # optional: every number typed in main.tex, recomputed
cd ../report && latexmk -pdf main.tex                                      # or: tectonic -X compile main.tex
```

Without `--platform` (and with `PVI_PLATFORM` unset), `count_outcomes.py` sums every stored
root, which is not what the report quotes, and the other scripts read the earlier run (`raw/`,
`tables/`). The committed figures were made with the pinned matplotlib 3.11.2; another version
draws them slightly larger or smaller, which can change the page count (see the last note below).

### The stored benchmark runs

| Root | Records | What it is |
|---|---|---|
| `raw_rtx2080ti-v2/` | 78,266 | **The report's numbers.** `strong_gpu.sh must` (in `experiments/4_defence_benchmark/slurm/`; every job but `ab-opt13`: the records come from 29 SLURM jobs, 945088–945116) on `studentbatch`: an RTX 2080 Ti (11 GB) prover and 8 threads of a Xeon Silver 4114 verifier, run from a clean clone of this code (commit `9401431`, stored in every record) with `PVI_PLATFORM=rtx2080ti-v2`. Every decoder is built at full depth, with the claims streamed to host memory (`--lean`), except Llama-2-13B: its 12.1 GiB of int8 weights exceed the card, so its full build is skipped (a `SKIP` line in the job log) and it is extrapolated from its 1- and 2-block builds (starred in Table 3). The non-lean OPT-1.3B control at 2,048 tokens (`ab-opt13`, about 27 GiB of GPU memory) does not fit the card and was not run. The full models are also attacked (§4.3). The root also holds controls with the same protocol, tagged `_nofix` (the earlier weight cache), `_nolean` (the non-lean path) and `_thr1` (one verifier thread). All of them count as runs in §4.3. `_nofix` and `_nolean` enter no table or figure; `_thr1` gives only the one-thread Maverick row of Table 4 and its sentence in §4.4. The root is frozen (`"frozen": true` in its `PLATFORM.json`: the code refuses to add records). |
| `raw/` | 59,547 | The earlier run on the same hardware, with an older version of the code. Its committed-weights (C) prover re-uploaded the weights to the GPU twice per query (a weight cache keyed `cuda` vs `cuda:0`), so its C prover times are too slow (the `_nofix` controls of `raw_rtx2080ti-v2/` measure this on the same GPU: Llama-2-7B at 64 tokens 11.0 s against 7.9 s, GPT-2 1.15×, OPT-1.3B at 2,048 tokens 1.16×; within noise for the small CNNs), and every decoder above 12 blocks was built only with 1 and 2 blocks and extrapolated. The report does not use it. It is frozen (the code refuses to write there) and still reproducible: `--platform rtx2080ti` (or no `--platform`) reads `raw/` and writes `tables/`, and `paper_assets.py --platform rtx2080ti` rebuilds its figures and tables. |

### Route B: from scratch

The MNIST experiments (CPU):

```bash
python experiments/0_train_models/train.py          # skips the committed models; --force retrains them
python experiments/1_reproduction/run.py            # -> artifacts/results/reproduction.json
python experiments/2_attack/run.py                  # -> artifacts/results/attack.json
python experiments/3_sampling_fixes/run.py          # -> artifacts/results/defence.json
python experiments/3_sampling_fixes/floor_sampler.py  # -> artifacts/results/floor_sampler.json
```

The benchmark needs a GPU. On the cluster, from the repository root, train the CNNs
first and wait for that job to finish, since the benchmark jobs load their weights (the
report's trained weights are already in the project folder on the cluster; new ones make
every CNN number incomparable with the report):

```bash
mkdir -p logs
sbatch -o logs/%x-%j.out code/experiments/4_defence_benchmark/slurm/train.sbatch lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel
# when it has finished (squeue -u $USER), the report's jobs, as for raw_rtx2080ti-v2/:
export PVI_PLATFORM=<new name> SBATCH_PARTITION=studentbatch SBATCH_GRES=gpu:geforce_rtx_2080:1
bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh smoke    # once per GPU type: wait for SMOKE OK
bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh must
```

Each job appends raw records to `artifacts/comparison/raw_<platform>/`, one root per GPU
and CPU model; `raw/` and `raw_rtx2080ti-v2/` are the stored runs, so re-measure under a
new name. A cell that has finished is skipped, so an interrupted tier is resumed by
running the same command again. On an 11 GB card the full Llama-2-13B build is skipped
as too large and its job ends with an error; the rest of that job is recorded. `ab-opt13`
(the non-lean OPT-1.3B at 2,048 tokens, about 27 GiB) runs out of GPU memory there, so
leave it out. Then run
Route A with `--platform <new name>`. `STRONG_GPU_PLAN.md` (repository root) describes
these runs in detail, and `experiments/4_defence_benchmark/README.md` shows how to run a
single model without SLURM.

### Where each result comes from

| Report | Produced by | Output |
|---|---|---|
| Fig. 1, Fig. 2 (schematics) | `5_comparison/paper_assets.py` | `report/figures/overview_*`, `protocol` |
| §4.2 *Reproduction* (acceptance, detection, exact-equality check) | `1_reproduction/run.py` | `artifacts/results/reproduction.json` |
| Table 1 rows 1–8; §4.2 *The attack works* (backdoor, stealth, 250 paths) | `2_attack/run.py` | `artifacts/results/attack.json` |
| Table 1 last row; §4.2 *Smarter sampling* | `3_sampling_fixes/run.py` | `artifacts/results/defence.json` |
| §3.2 the floor sampler | `3_sampling_fixes/floor_sampler.py` | `artifacts/results/floor_sampler.json` |
| §4.2 *The same holds on larger models* (100% flips, 1/84–1/512, 1/28 million, paths for 2^-40; cells `attack_float`, `sampling`) | `4_defence_benchmark/bench.py` → `5_comparison/aggregate.py --platform rtx2080ti-v2` | `tables_rtx2080ti-v2/measured_summary.csv` |
| Fig. 3, Tables 2–4, Figs. 4–5, all numbers in §4.3–4.4, the hardware of §4.1 | `4_defence_benchmark/slurm/strong_gpu.sh must` → `5_comparison/aggregate.py --platform rtx2080ti-v2` → `paper_assets.py --platform rtx2080ti-v2` | `report/figures`, `report/tables` (with `hardware.tex`) |
| §4.1 the extrapolation check on the models built in full (proof size within 0.02%, exactly for Kpre; prover within 12%; C verifier up to 48% too low at ≤ 64 tokens; totals are the sums of the prove and verify parts) | `5_comparison/aggregate.py --platform rtx2080ti-v2` (every part, full build against the 1–2 block line); `validate_extrapolation.py --platform rtx2080ti-v2` (with a fit through every block count); `text_numbers.py --platform rtx2080ti-v2` prints the totals | `tables_rtx2080ti-v2/llm_extrapolation_check.csv`, `llm_extrapolation_validation.csv` |
| §4.1 peak GPU memory (Llama-2-7B at 2,048 tokens: 7.3 GiB) and the claims at 2,048 tokens (10–11 GB for the 7B models) | `5_comparison/aggregate.py --platform rtx2080ti-v2` | `tables_rtx2080ti-v2/measured_summary.csv` (metrics `gpu_peak_memory` and `bytes_claims`, cells `defence_C_int_lam128_T2048_L32`) |
| §4.3 counts (5,547 honest queries; 1,854 CNN and 602 LLM attacks; Freivalds 2,344, Reed–Solomon 56, Merkle 56) and §4.1's 78,266 records | `5_comparison/count_outcomes.py --platform rtx2080ti-v2` | printed |
| §4.2 Llama-2-7B: 1/11,008 per path, 305,000 paths, 13.7 GB | `5_comparison/analytic.py` | `tables/analytic.csv` (`anchuri_*` columns) |
| Published results (Table 4, grey points) | `5_comparison/literature.py` | `tables/reported_curated.csv` |
| Numbers quoted only in the text (e.g. 0.4 points, 4–8×, 2.4–4.4×, 126×, 1.62 s) | `5_comparison/aggregate.py --platform rtx2080ti-v2`; `text_numbers.py --platform rtx2080ti-v2` prints each with its line of `main.tex` | `tables_rtx2080ti-v2/measured_summary.csv`, `tables_rtx2080ti-v2/llm_full_model.csv` (ratios with `tables/reported_curated.csv`) |

## Tests

```bash
python -m pytest tests
```

The tests cover the Merkle commitment, the path test, the exact acceptance-probability
dynamic program against Monte Carlo, the attacks and samplers, and the whole defence.
The defence tests include forged claims, a forged folded row (caught by the
Reed–Solomon check), a forged column (caught by the Merkle check), and GPU/CPU
agreement. Without CUDA, `python -m pytest --collect-only -q` lists 234 tests, and the 44
GPU-only ones (GPU/CPU bit-exactness, with TF32 off and on) are skipped; on a GPU they
run too, together with the CUDA cases of `tests/test_gpu_plan.py` (243 tests).

## Notes on reproducibility

* The MNIST models are committed because training is only reproducible on one
  platform: a rerun on the same machine, library versions and thread count gives the
  same weights, but another machine gives slightly different, equally accurate ones,
  and the MNIST numbers then move in the last digits. Check the committed weights with
  `cd artifacts/models && sha256sum -c MODELS.sha256`.
* The protocols draw fresh random challenges on every run, so measured acceptance
  rates move slightly between runs. Where a probability can be computed exactly (the
  path test's acceptance for a given trace, via `pvi.experiments.analysis`), it is.
* The language models use their exact shapes with random int8 weights, since costs
  depend only on shapes. In `raw_rtx2080ti-v2/` every decoder is measured at full depth
  except Llama-2-13B, which is extrapolated linearly from 1- and 2-block builds
  (`aggregate.py`), labelled `extrapolated_from_1_and_2_blocks` in `llm_full_model.csv`
  and starred in Table 3; where a model has both, the full build is used. In the earlier
  `raw/`, every decoder above 12 blocks is extrapolated.
* Every raw record carries the git commit, host, GPU and CPU it was measured on.
* The figure PDFs depend slightly on the matplotlib version; build the committed ones
  with the pinned versions of `requirements.txt` (matplotlib 3.11.2, which reproduces
  them byte for byte; matplotlib 3.8, for example, draws Figures 2 and 3 a few points
  shorter), and check that the report still has 8 pages.

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
  tests/                   180 tests
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

`artifacts/comparison/raw/` holds all 59,547 raw benchmark records. These commands
rebuild every derived table, the report's figures and generated tables, and the PDF:

```bash
python experiments/5_comparison/aggregate.py        # raw records -> tables/measured_summary.csv, llm_full_model.csv
python experiments/5_comparison/literature.py       # published numbers -> tables/reported_curated.csv
python experiments/5_comparison/analytic.py         # the path test's cost of 2^-40 on Llama-2-7B -> tables/analytic.csv
python experiments/5_comparison/count_outcomes.py   # the soundness counts of Section 4.3
python experiments/5_comparison/paper_assets.py     # -> ../report/figures/*.pdf, ../report/tables/*.tex
cd ../report && latexmk -pdf main.tex               # or: tectonic -X compile main.tex
```

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
first and wait for that job to finish, since the benchmark jobs load their weights:

```bash
mkdir -p logs
sbatch -o logs/%x-%j.out code/experiments/4_defence_benchmark/slurm/train.sbatch lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel
# when it has finished (squeue -u $USER):
bash code/experiments/4_defence_benchmark/slurm/sweep.sh     # every benchmark job of the report
```

Each job appends raw records to `artifacts/comparison/raw/`. A cell that has finished
is skipped, and the stored `raw/` has every cell finished, so move it aside first to
re-measure (`mv code/artifacts/comparison/raw code/artifacts/comparison/raw_stored`).
Then run Route A. `experiments/4_defence_benchmark/README.md` shows how to run a single
model without SLURM.

### Where each result comes from

| Report | Produced by | Output |
|---|---|---|
| Fig. 1, Fig. 2 (schematics) | `5_comparison/paper_assets.py` | `report/figures/overview_*`, `protocol` |
| §4.2 *Reproduction* (acceptance, detection, exact-equality check) | `1_reproduction/run.py` | `artifacts/results/reproduction.json` |
| Table 1 rows 1–8; §4.2 *The attack works* (backdoor, stealth, 250 paths) | `2_attack/run.py` | `artifacts/results/attack.json` |
| Table 1 last row; §4.2 *Smarter sampling* | `3_sampling_fixes/run.py` | `artifacts/results/defence.json` |
| §3.2 the floor sampler | `3_sampling_fixes/floor_sampler.py` | `artifacts/results/floor_sampler.json` |
| §4.2 *The same holds on larger models* (100% flips, 1/84–1/512, 1/28 million, paths for 2^-40; cells `attack_float`, `sampling`) | `4_defence_benchmark/bench.py` → `5_comparison/aggregate.py` | `tables/measured_summary.csv` |
| Fig. 3, Tables 2–4, Figs. 4–5, all numbers in §4.3–4.4 | `4_defence_benchmark/bench.py` → `5_comparison/aggregate.py` → `paper_assets.py` | `report/figures`, `report/tables` |
| §4.3 counts (3,174 / 1,854 / 174; which check fired) | `5_comparison/count_outcomes.py` | printed |
| §4.2 Llama-2-7B: 1/11,008 per path, 305,000 paths, 13.7 GB | `5_comparison/analytic.py` | `tables/analytic.csv` (`anchuri_*` columns) |
| Published results (Table 4, grey points) | `5_comparison/literature.py` | `tables/reported_curated.csv` |
| Numbers quoted only in the text (e.g. 0.4 points, 4–7×, 2.4–4.4×, 141×, 1.37 s) | `5_comparison/aggregate.py` | `tables/measured_summary.csv`, `tables/llm_full_model.csv` (ratios with `tables/reported_curated.csv`) |

## Tests

```bash
python -m pytest tests
```

The tests cover the Merkle commitment, the path test, the exact acceptance-probability
dynamic program against Monte Carlo, the attacks and samplers, and the whole defence.
The defence tests include forged claims, a forged folded row (caught by the
Reed–Solomon check), a forged column (caught by the Merkle check), and GPU/CPU
agreement. The three GPU tests are skipped without CUDA.

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
  depend only on shapes. Models above 12 blocks are measured with 1 and 2 blocks and
  extrapolated linearly (`aggregate.py`), labelled `extrapolated` in the tables.
* Every raw record carries the git commit, host, GPU and CPU it was measured on.
* The figure PDFs depend slightly on the matplotlib version; the committed ones were
  made with the pinned versions.

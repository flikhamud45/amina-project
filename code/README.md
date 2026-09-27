# Code: Certified but Compromised

Everything behind the report (`../report/main.pdf`):

1. a from-scratch reimplementation of the path-sampling proof of inference of Anchuri
   et al. (SaTML 2026);
2. our single-neuron attack and backdoor on it;
3. the analysis of smarter sampling rules;
4. our defence, which checks every weight layer with Freivalds' algorithm against a
   Reed–Solomon/Merkle commitment to the weights;
5. the benchmark and the comparison with the literature.

Every table and figure of the report can be rebuilt from the stored measurements in a
minute (no GPU), or re-measured from scratch.

## Layout

```
code/
  src/pvi/                 the library (see src/pvi/README.md)
  experiments/             one folder per result; each has a README
    0_train_models/        train the MNIST models used by 1-4        CPU  ~15 min
    1_reproduction/        the original protocol on its threat model  CPU   ~5 min
    2_attack/              the single-neuron attack and backdoor      CPU   ~3 min
    3_sampling_fixes/      smarter samplers, and why they fail        CPU  ~10 min
    4_defence_benchmark/   our defence vs. the path test, CNNs + LLMs GPU  hours (SLURM)
    5_comparison/          tables, counts and every report figure     CPU   ~1 min
  artifacts/comparison/    stored measurements (raw records, tables, literature)
  tests/                   209 tests
```

The experiments import the library (`pvi`) and never each other.

## Installation

Python 3.10+ (the results were produced with Python 3.12). From the repository root:

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
(`source ../.venv/bin/activate`).

**Data.** MNIST is downloaded to `code/data/` by `experiments/0_train_models`. The
benchmark also reads CIFAR-10 from `$PVI_CIFAR_ROOT` and ImageNet (for the dog/cat and
dog/squirrel classifiers) from `$PVI_IMAGENET_ROOT`; both default to the course's copies
on the cluster. Decoded images are cached in `artifacts/fullcheck/cache/` (or
`$PVI_CACHE`).

## Reproducing the report

### Route A: from the stored measurements (no GPU, about a minute)

`artifacts/comparison/raw/` holds all 59,547 raw benchmark records. These commands
rebuild every derived table, the report's figures and tables, and the PDF:

```bash
python experiments/5_comparison/aggregate.py        # raw records -> tables/measured_summary.csv, llm_full_model.csv
python experiments/5_comparison/literature.py       # published numbers -> tables/reported_curated.csv
python experiments/5_comparison/analytic.py         # the path test's cost of 2^-40 on Llama-2-7B -> tables/analytic.csv
python experiments/5_comparison/count_outcomes.py   # the soundness counts of Section 4.3
python experiments/5_comparison/paper_assets.py     # -> ../report/figures/*.pdf, ../report/tables/*.tex
cd ../report && latexmk -pdf main.tex               # or: tectonic -X compile main.tex
```

### Route B: from scratch

```bash
python experiments/0_train_models/train.py          # MNIST models (seeded, bit-exact on CPU)
python experiments/1_reproduction/run.py            # -> artifacts/results/reproduction.json
python experiments/2_attack/run.py                  # -> artifacts/results/attack.json
python experiments/3_sampling_fixes/run.py          # -> artifacts/results/defence.json
python experiments/3_sampling_fixes/floor_sampler.py  # -> artifacts/results/floor_sampler.json
```

The benchmark needs a GPU. On the cluster, from the repository root:

```bash
sbatch -o logs/%x-%j.out code/experiments/4_defence_benchmark/slurm/train.sbatch lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel
bash code/experiments/4_defence_benchmark/slurm/sweep.sh     # every benchmark job of the report
```

Each job appends raw records to `artifacts/comparison/raw/`; finished cells are skipped
on re-submission. Then run Route A. `experiments/4_defence_benchmark/README.md` shows
how to run a single model without SLURM.

### Where each result comes from

| Report | Produced by | Output |
|---|---|---|
| Fig. 1, Fig. 2 (schematics) | `5_comparison/paper_assets.py` | `report/figures/overview_*`, `protocol` |
| §4.2 *Reproduction* (acceptance, detection, 17–32% exact-equality) | `1_reproduction/run.py` | `reproduction.json` |
| Table 1 rows 1–8; §4.2 *The attack works* (backdoor, stealth, 250 paths) | `2_attack/run.py` | `attack.json` |
| Table 1 last row; §4.2 *Smarter sampling* (0.20%, 0.11%, 0.006%, 0) | `3_sampling_fixes/run.py` | `defence.json` |
| §3.2 floor sampler (about 10× uniform) | `3_sampling_fixes/floor_sampler.py` | `floor_sampler.json` |
| Fig. 3, Tables 2–4, Figs. 4–5, all numbers in §4.3–4.4 | `4_defence_benchmark/bench.py` → `5_comparison/aggregate.py` → `paper_assets.py` | `report/figures`, `report/tables` |
| §4.3 counts (3,174 / 1,854 / 174; which check fired) | `5_comparison/count_outcomes.py` | printed |
| §4.2 Llama-2-7B: 1/11,008 per path, 305,000 paths, 13.7 GB | `5_comparison/analytic.py` | `tables/analytic.csv` (`anchuri_*` columns) |
| Published results (Table 4, grey points) | `5_comparison/literature.py` | `tables/reported_curated.csv` |
| Numbers quoted only in the text (e.g. 0.4 points, 1/84–1/512, 1/28 million, 4–7×, 2.4–4.4×, 141×, 1.37 s) | `5_comparison/aggregate.py` | `tables/measured_summary.csv`, `tables/llm_full_model.csv` (ratios with `tables/reported_curated.csv`) |

## Tests

```bash
python -m pytest tests/ -q
```

The tests cover the Merkle commitment, the path test, the exact acceptance-probability
dynamic program against Monte Carlo, the attacks and samplers, and the whole defence.
The defence tests include forged claims, a forged folded row (caught by the
Reed–Solomon check), a forged column (caught by the Merkle check), and GPU/CPU
agreement. GPU tests are skipped without CUDA.

## Notes on reproducibility

* Training is seeded; on CPU the MNIST models are reproduced bit for bit.
* The protocols draw fresh random challenges on every run, so measured acceptance
  rates move slightly between runs. Where a probability can be computed exactly (the
  path test's acceptance for a given trace, via `pvi.experiments.analysis`), it is.
* The language models use their exact shapes with random int8 weights, since costs
  depend only on shapes. Models above 12 blocks are measured with 1 and 2 blocks and
  extrapolated linearly (`aggregate.py`), labelled `extrapolated` in the tables.
* Every raw record carries the git commit, host, GPU and CPU it was measured on.

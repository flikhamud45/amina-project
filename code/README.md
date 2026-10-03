# Certified but Compromised: code and measurements

This folder holds the implementation and the stored measurements behind the paper *Certified but
Compromised: Breaking and Fixing Lightweight Proofs of Inference* (`report/main.pdf`, next to this
folder in the submission). It contains a from-scratch implementation of the path-sampling proof of
inference of Anchuri et al. (SaTML 2026); our single-neuron attack and backdoor on it, on MNIST
models, on CNNs and on the real OPT-6.7B; the analysis of smarter sampling rules; our protocol, which
checks every weight layer with Freivalds' algorithm against a Reed–Solomon/Merkle commitment to the
weights, in its basic and optimised forms; and the benchmark that measures both protocols on six image
classifiers and ten language-model architectures and compares them with the path protocol and with
published systems. Every table, figure and benchmark number of the paper can be regenerated from the
stored measurements in about a minute, without a GPU.

**Contents:** [Installation](#installation) · [Quick start](#quick-start) · [Tests](#tests) ·
[Layout](#layout) · [Stored measurements](#stored-measurements) · [Results map](#results-map) ·
[From scratch](#from-scratch) · [Notes on reproducibility](#notes-on-reproducibility) ·
[Submission tarball](#submission-tarball)

## Installation

**Python and packages.** Python 3.11 or newer for the pinned versions (the stored runs used Python
3.12.3). [`requirements.txt`](requirements.txt) pins the versions the results were produced with:

| Package | Version | Needed for |
|---|---|---|
| `torch` | 2.5.1 (the GPU runs: the CUDA 12.1 build, `2.5.1+cu121`) | everything |
| `torchvision` | 0.20.1 | loading MNIST, CIFAR-10 and ImageNet |
| `numpy` | 2.5.2 | everything |
| `matplotlib` | 3.11.2 | the figures (this version reproduces the committed PDFs byte for byte) |
| `pytest` | 9.1.1 | the tests |

The real-OPT experiments ([`experiments/2_attack/REAL_LLM.md`](experiments/2_attack/REAL_LLM.md)) also
need `transformers`, `tokenizers` and `pyarrow`, and `tests/test_real_weights.py` needs `transformers`.
[`requirements-llm.txt`](requirements-llm.txt) pins them at the versions of the stored runs (4.51.3,
0.21.4 and 25.0.1); install it together with the main file, from `code/`:
`pip install -r requirements.txt -r requirements-llm.txt`. zkLLM's run on our GPU uses zkLLM's own
environment ([`artifacts/results/zkllm_l40s`](artifacts/results/zkllm_l40s/README.md)).

From the folder that holds `code/` (the unpacked submission):

```bash
python3 -m venv .venv
# on a GPU machine, the CUDA build first (on the TAU cluster the default wheel does not match the drivers):
.venv/bin/pip install torch==2.5.1 torchvision==0.20.1 --index-url https://download.pytorch.org/whl/cu121
.venv/bin/pip install -r code/requirements.txt    # the pinned versions
# for the real-OPT experiments, instead: .venv/bin/pip install -r code/requirements.txt -r code/requirements-llm.txt
.venv/bin/pip install -e code                     # the pvi package (src/pvi)
source .venv/bin/activate
```

`pip install -e code` alone installs the ranges of [`pyproject.toml`](pyproject.toml) (Python 3.10 or
newer, torch 2.2 or newer, numpy 1.26 or newer, matplotlib 3.8 or newer), which run the code; the
figures are then not byte-identical. All commands below run from `code/` with the environment active;
`export PYTHONPATH=$PWD/src` makes the scripts use this folder's `pvi` even without the editable
install (the SLURM scripts do this themselves and run from the folder that holds `code/`).

**Data and models.** Nothing needs downloading to regenerate the paper's tables and figures from the
stored measurements.

| Input | Where it comes from | Needed by |
|---|---|---|
| MNIST | downloaded by torchvision to `code/data/` on first use (needs network) | experiments 0–3; LeNet-5 and the MLP in the benchmark |
| the MNIST models | included in `artifacts/models/` (with `MODELS.sha256`); `0_train_models/train.py` skips them | experiments 1–4 |
| CIFAR-10 | not downloaded by the code: torchvision's `cifar-10-batches-py/` in `code/data/cifar10/` or, if set, in `$PVI_CIFAR_ROOT` (e.g. `python -c "from torchvision import datasets; datasets.CIFAR10('<dir>', download=True)"`) | training and querying VGG-11, VGG-16, ResNet-18 (CIFAR) |
| ImageNet | not downloadable by the code: `code/data/imagenet/` or, if set, `$PVI_IMAGENET_ROOT`, with `train/` and `val/`, one folder per class named by its 1-based index in sorted-synset order; decoded once into a cache in `artifacts/fullcheck/cache/` | the 224-pixel dog-vs-cat and dog-vs-squirrel ResNet-18 |
| the benchmark's CNN weights | not included: trained by `4_defence_benchmark/train.py` (or `slurm/train.sbatch`) before any CNN benchmark job | experiment 4 |
| the language models | none: exact shapes with random int8 weights | experiment 4 |
| OPT checkpoints, WikiText-2 | `facebook/opt-*` from Hugging Face (open, no token) into `$HF_HOME`; the WikiText-2 test split as parquet at `$WIKITEXT_PARQUET` | the real-OPT experiments |
| zkLLM | zkLLM's own checkout and the gated Llama-2 weights | zkLLM's run on our GPU |

Without `$PVI_CIFAR_ROOT` and `$PVI_IMAGENET_ROOT`, the code reads CIFAR-10 from `code/data/cifar10/`
and ImageNet from `code/data/imagenet/`; the variables override these defaults. **On the cluster** the
datasets are not under `code/data/`, so every CIFAR or ImageNet job, `slurm/smoke.sbatch` included,
needs `PVI_CIFAR_ROOT` (and `PVI_IMAGENET_ROOT`, unless the ImageNet cache in
`artifacts/fullcheck/cache/` exists) exported to the dataset's location before `sbatch`, which passes
the exported variables on to the job. Details are in
[`experiments/4_defence_benchmark`](experiments/4_defence_benchmark/README.md#1-train-the-cnns-gpu)
and [`REAL_LLM.md`](experiments/2_attack/REAL_LLM.md).

## Quick start

```bash
cd code && export PYTHONPATH=$PWD/src
python -m pytest tests/test_merkle.py tests/test_protocol.py tests/test_comparison_tables.py   # a fast subset
python experiments/5_comparison/aggregate.py --platform l40s              # the basic protocol's tables
python experiments/5_comparison/aggregate.py --platform l40s_improved     # the optimised protocol's tables
python experiments/5_comparison/aggregate.py --platform l40s_improved2
python experiments/5_comparison/aggregate.py --platform l40s_improved3
python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved,l40s_improved2,l40s_improved3   # Tables 1-3, Figs. 1-4
python experiments/5_comparison/text_numbers.py --platform l40s --optimised l40s_improved,l40s_improved2,l40s_improved3 --definition   # checks the text
python experiments/5_comparison/count_outcomes.py --platform l40s_improved,l40s_improved2,l40s_improved3 --definition    # Sec. 4.3's counts
```

`paper_assets.py` writes into `../report/figures/` and `../report/tables/`; the full sequence, including
the PDF build, is in [`experiments/5_comparison`](experiments/5_comparison/README.md#regenerate-everything).

## Tests

```bash
cd code && PYTHONPATH=src python -m pytest tests
```

The tests cover the Merkle commitment, the path protocol, the exact acceptance-probability dynamic
program against Monte Carlo, the attacks and samplers, and the whole protocol: honest queries, forged
claims, a forged folded row (caught by the Reed–Solomon check), forged columns and paths (caught by the
Merkle check), every commitment plan, the compact encoding, lookups and pruning in modes C, K and
Kpre, the lean prover and verifier, the GPU verifier, GPU/CPU bit-exactness, and the table and figure
scripts. All tests pass. The GPU-only tests are skipped on a machine without CUDA, and
`tests/test_real_weights.py` is skipped unless `transformers` is installed. A CPU-only run takes about
a quarter of an hour on a 4-core laptop. On the cluster, `slurm/smoke.sbatch` of
[`experiments/4_defence_benchmark`](experiments/4_defence_benchmark/README.md#4-on-the-tau-slurm-cluster)
runs every test on the GPU and requires that none is skipped; it also loads every dataset, so submit
it with `PVI_CIFAR_ROOT` (and `PVI_IMAGENET_ROOT`) exported, as described above.

## Layout

```
code/
  README.md               this file
  requirements.txt        the pinned versions
  requirements-llm.txt    the real-OPT experiments' packages: pip install -r requirements.txt -r requirements-llm.txt
  pyproject.toml          the pvi package
  src/pvi/                the library                                         (src/pvi/README.md)
  experiments/            one folder per result, each with a README
    0_train_models/       train the MNIST models                              CPU   ~2 min
    1_reproduction/       the path protocol on its own threat model           CPU   ~1 min
    2_attack/             the single-neuron attack and backdoor (MNIST)       CPU   ~2 min
                          ... and on the real OPT-6.7B (REAL_LLM.md)          GPU   ~1 h per run
    3_sampling_fixes/     smarter samplers, and why they fail                 CPU   ~40 min
    4_defence_benchmark/  both protocols and the path test, CNNs and LLMs     GPU   hours (SLURM)
    5_comparison/         tables, counts and figures from the stored records  CPU   ~1 min
    6_improvements/       the optimised protocol's development harnesses      CPU/GPU
  artifacts/
    models/               the MNIST models (with MODELS.sha256)
    results/              JSON results of experiments 1-3 and of the real-OPT runs;
                          zkllm_l40s/: zkLLM's code run on our GPU
    comparison/           benchmark records raw_<platform>/, derived tables, the literature catalogue
  tests/                  the test suite
```

The experiments import the library (`pvi`) and never each other (except
`6_improvements/perf.py`, which builds the models as `bench.py` does). Times are for 8 CPU threads.

| README | Covers |
|---|---|
| [`src/pvi/README.md`](src/pvi/README.md) | the library's modules and where the paper uses them |
| [`experiments/0_train_models`](experiments/0_train_models/README.md) | training the MNIST models |
| [`experiments/1_reproduction`](experiments/1_reproduction/README.md) | Sec. 4.2 *Reproduction* |
| [`experiments/2_attack`](experiments/2_attack/README.md), [`REAL_LLM.md`](experiments/2_attack/REAL_LLM.md) | Sec. 3.2 and 4.2: the attack on MNIST, larger models and the real OPT-6.7B; Sec. 4.1 *Real weights* |
| [`experiments/3_sampling_fixes`](experiments/3_sampling_fixes/README.md) | Sec. 3.3 and 4.2 *Other samplers* |
| [`experiments/4_defence_benchmark`](experiments/4_defence_benchmark/README.md) | the benchmark: re-measuring both protocols (bench.py, the SLURM scripts, hardware and time) |
| [`experiments/5_comparison`](experiments/5_comparison/README.md) | every table, figure and benchmark number from the stored records |
| [`experiments/6_improvements`](experiments/6_improvements/README.md) | the optimised protocol's options, their soundness, tests and development measurements |
| [`artifacts/comparison`](artifacts/comparison/README.md) | the stored benchmark runs |
| [`artifacts/results/zkllm_l40s`](artifacts/results/zkllm_l40s/README.md) | zkLLM's code on our GPU |

## Stored measurements

The benchmark records are stored per run in `artifacts/comparison/raw_<platform>/`, with the commit,
host, GPU and CPU of every record; [`artifacts/comparison/README.md`](artifacts/comparison/README.md)
describes each run in detail.

| Root | Records | What it is |
|---|---:|---|
| `raw_l40s/` | 123,832 | **the basic protocol**, measured on an NVIDIA L40S prover and 8 threads of an AMD EPYC 9334 verifier (commit `9401431`) |
| `raw_l40s_improved/` | 68,363 | **the optimised protocol**, first batch, same machines (commit `c8be5eb`); also the basic proofs run by the released code |
| `raw_l40s_improved2/` | 83,896 | **the optimised protocol**, second batch, same machines (commits `366e3d4`, `c2bdc49`); some timings withheld, see below |
| `raw_l40s_improved3/` | 1,600 | **the optimised protocol**, the re-measurement of the second batch's withheld cells, same machines without nodes n-801 and n-804 (commit `7016557`); interim: the cells finished so far |
| `raw_l40s_improved2_thr1/` | 6,104 | optimised LeNet-5 and VGG-16 with one verifier thread; not used in the paper |
| `raw_rtx2080ti-v2/` | 78,266 | the basic protocol on an RTX 2080 Ti and a Xeon Silver 4114 (commit `9401431`): Sec. 4.1's 1,978 hardware-independent values |
| `raw/` | 59,547 | an earlier run with an earlier version of the code; not used in the paper |

`artifacts/comparison/excluded_cells.csv` lists 17 timings of the second batch that are withheld: during
parts of that batch the L40S nodes n-801 and n-804 were slowed by other users' load, which inflated
CPU-side stages. The paper shows these values as pending (in red, with the basic protocol's value as a
placeholder) until they are re-measured. The re-run on the other nodes, platform `l40s_improved3`, has
re-measured 6 of them so far; its cells replace the same cells of the second batch, and the other 11
values stay pending. Proof bytes and verdicts are unaffected.

## Results map

Every result of the paper, the command that reproduces it from the stored data (run from `code/` after
`export PYTHONPATH=$PWD/src`; for the benchmark rows the four `aggregate.py` lines of the quick start
come first), and the README with the detailed steps, including the from-scratch runs. Below,
`paper_assets` stands for
`python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved,l40s_improved2,l40s_improved3`
and `text_numbers` for
`python experiments/5_comparison/text_numbers.py --platform l40s --optimised l40s_improved,l40s_improved2,l40s_improved3 --definition`,
which recomputes the benchmark numbers of the text (40 checks, each finding its sentence by its
wording) and compares them with `report/main.tex` (exit 0: all match).

**Tables, figures and generated macros**

| Paper | Result | From the stored data | Details |
|---|---|---|---|
| Fig. 1, Fig. 2 | overview and protocol schematics | `paper_assets` → `report/figures/overview_*.pdf`, `protocol.pdf` | [5_comparison](experiments/5_comparison/README.md) |
| Fig. 3 | security against bytes sent (paths, basic, optimised) | `paper_assets` → `report/figures/security_bits.pdf`, from the `aggregate.py` tables | [5_comparison](experiments/5_comparison/README.md) |
| Fig. 4 | cost against model size, with published systems | `paper_assets` → `report/figures/cost_*.pdf`, also reading `tables/reported_curated.csv` (`python experiments/5_comparison/literature.py`) and `artifacts/results/zkllm_l40s/summary.csv` | [5_comparison](experiments/5_comparison/README.md) |
| Table 1 | image models (optimised, λ = 128) | `paper_assets` → `report/tables/cnn.tex` | [5_comparison](experiments/5_comparison/README.md), [4_defence_benchmark](experiments/4_defence_benchmark/README.md) |
| Table 2 | language models at 64 and 2,048 tokens, CPU and GPU verifier | `paper_assets` → `report/tables/llm.tex` | [5_comparison](experiments/5_comparison/README.md), [4_defence_benchmark](experiments/4_defence_benchmark/README.md) |
| Table 3 | ours against published systems | `paper_assets` → `report/tables/ratios.tex` (published numbers from `literature.py`; zkLLM on our GPU from `python artifacts/results/zkllm_l40s/summarise.py --csv`) | [5_comparison](experiments/5_comparison/README.md), [zkllm_l40s](artifacts/results/zkllm_l40s/README.md) |
| Sec. 4.1 | hardware macros (`report/tables/hardware.tex`) | `paper_assets` | [5_comparison](experiments/5_comparison/README.md) |
| Abstract, Sec. 4.3 | count macros (`report/tables/counts.tex`, `counts_opt.tex`) | `python experiments/5_comparison/count_outcomes.py --platform l40s --tex ../report/tables/counts.tex` and `... --platform l40s_improved,l40s_improved2,l40s_improved3 --prefix Opt --definition --tex ../report/tables/counts_opt.tex` | [5_comparison](experiments/5_comparison/README.md) |
| red values | values still pending | `paper_assets --check` | [5_comparison](experiments/5_comparison/README.md#pending-values) |

**Numbers in the text, by section**

| Paper | Result | From the stored data | Details |
|---|---|---|---|
| Sec. 1 | zkLLM needs about ten minutes for one Llama-2-7B prompt | `tables/reported_curated.csv` (620 s; `literature.py`) | [5_comparison](experiments/5_comparison/README.md) |
| Sec. 1, 4.2 | one tampered neuron accepted with probability 1 - 1/N (99.6–99.8% on MNIST) | `artifacts/results/attack.json` (`experiments/2_attack/run.py`) | [2_attack](experiments/2_attack/README.md) |
| Sec. 1 | 13.7 GB of paths against a 325 MB proof; 2.0–4.2x smaller proofs; prover 35–32,000x faster than the zkSNARKs | `text_numbers` | [5_comparison](experiments/5_comparison/README.md) |
| Sec. 3.2 | the attack (method) | `experiments/2_attack/run.py` | [2_attack](experiments/2_attack/README.md) |
| Sec. 3.3 | samplers, Proposition 3.1 (proved in the paper), the floor sampler | `experiments/3_sampling_fixes/run.py`, `floor_sampler.py` | [3_sampling_fixes](experiments/3_sampling_fixes/README.md) |
| Sec. 3.5 | parameters (VGG-16 at λ = 128: r = 5, t = 67) | `cfg_reps`, `cfg_columns` of VGG-16's `defence_C_int_lam128_rate4` in `artifacts/comparison/tables_l40s/measured_summary.csv` | [5_comparison](experiments/5_comparison/README.md) |
| Sec. 3.6.1 | Llama-2-7B at 64 tokens: 349, 23, 385 and 4 MB of a 762 MB basic proof | `bytes_claims`, `bytes_u`, `bytes_columns`, `bytes_paths`, `bytes_total` of `defence_C_int_lam128_T64_L32` in `tables_l40s/measured_summary.csv` | [5_comparison](experiments/5_comparison/README.md) |
| Sec. 3.6.2 | six candidate plans, the 2^34 floor; LeNet-5 shares one tree of length 2^18 and opens 15 columns instead of 66 in each of five trees | `AUTO_CANDIDATES` and `AUTO_MIN_BUDGET` in `src/pvi/fullcheck/plans.py`; `cfg_group_columns`, `cfg_columns` of LeNet-5's `defence_C_int_lam128_rate4_wire_polauto` (`tables_l40s_improved2`) and `defence_C_int_lam128_rate4` (`tables_l40s`) | [6_improvements](experiments/6_improvements/README.md#7-the-planning-rule-auto) |
| Sec. 3.6.2 | a claim takes 16–19 bits with the compact encoding | `experiments/6_improvements/results/wire_laptop.json` (`wire.py --summary`) | [6_improvements](experiments/6_improvements/README.md#3-the-compact-encoding) |
| Sec. 4.1 | int8 accuracy within 0.4 points; extrapolated proof sizes within 2.83% | `text_numbers` | [5_comparison](experiments/5_comparison/README.md) |
| Sec. 4.1 | real OPT perplexities (73.8, 37.4, 36.0 against 64.7, 34.9, 27.1) | `artifacts/results/real_weights_ppl_opt-*.json` | [REAL_LLM.md](experiments/2_attack/REAL_LLM.md) |
| Sec. 4.1 | 1,978 hardware-independent values reproduced on the RTX 2080 Ti | `python experiments/5_comparison/fingerprint_check.py artifacts/comparison/raw_rtx2080ti-v2 artifacts/comparison/raw_l40s` | [5_comparison](experiments/5_comparison/README.md), [artifacts/comparison](artifacts/comparison/README.md) |
| Sec. 4.2 | *Reproduction* (3,000 runs per setting, 99.8–100% detection, 21–37% without tolerance) | `artifacts/results/reproduction.json` (`experiments/1_reproduction/run.py`) | [1_reproduction](experiments/1_reproduction/README.md) |
| Sec. 4.2 | *Attacks on MNIST* (hundreds of nodes against 1–14, backdoor 99.96% of 2,300 runs, about five neurons at the 95th percentile, 250 paths 61%) | `artifacts/results/attack.json` | [2_attack](experiments/2_attack/README.md) |
| Sec. 4.2 | *Other samplers* (0.20%, 0.11%, 0.016%; 14 zeroed neurons, 65%, 2,000 of 2,000; floor 2%, 0.9–6x) | `artifacts/results/defence.json`, `floor_sampler.json` | [3_sampling_fixes](experiments/3_sampling_fixes/README.md) |
| Sec. 4.2 | *Larger models* (1/84 to 1/512, 1/28 million, 2,316–14,182 paths, incl. the 224-pixel ResNet-18; Llama-2-7B 1/11,008, 305,000 paths, 13.7 GB) | `text_numbers`; `python experiments/5_comparison/analytic.py` | [2_attack](experiments/2_attack/README.md#larger-models) |
| Sec. 4.2 | *A real LLM* (40/40; " back" 40/40, " hacked" 0/40; 16,384 neurons reach 6,187 of 50,272 tokens; 1/16,384 per path, 454,248 paths) | `artifacts/results/real_llm_attack.json`, `real_llm_attack_auto.json` | [REAL_LLM.md](experiments/2_attack/REAL_LLM.md) |
| Sec. 4.3 | outcomes: optimised 3,792 honest accepted, 2,512 attacks rejected (1,848 image, 106 '+1'; Freivalds 1,601, code 853, Merkle 56, the range check the remaining two); basic 8,709 and 2,895 (804 language) | `python experiments/5_comparison/count_outcomes.py --platform l40s_improved,l40s_improved2,l40s_improved3 --definition` (`range_or_shape 2`); `... --platform l40s`; `text_numbers` | [artifacts/comparison](artifacts/comparison/README.md), [5_comparison](experiments/5_comparison/README.md) |
| Sec. 4.3 | security per byte (43–47 and 131–141 bits; basic 48–58 and 150–153; 1.1–8.3x) | `text_numbers` (Fig. 3: `paper_assets`) | [5_comparison](experiments/5_comparison/README.md) |
| Sec. 4.4 | image and language models (3.8–22.9 ms, 1.7–64.8 ms, 49.4 kB–6.7 MB, 1.3–9.3x and 1.7x smaller than the int8 weights; 76.1 ms, 1.8 s, 325 MB, 21x; at 2,048 tokens 17.6 s, 6.6 GB, 2.8 min; GPU verifier 11–25x and 1.0–2.4x; batches of eight, 0.39 s and 219 MB per prompt; Llama-2-70B 1.7 GB, 41x) | `text_numbers` (Tables 1–2, Fig. 4: `paper_assets`) | [5_comparison](experiments/5_comparison/README.md) |
| Sec. 4.5 | effect of the optimisations (61% fewer bytes, 77% less verifier time, 28% less prover time; 468 → 190 ms; 2,048 tokens; 1.9x the forward pass; setup ∼115 s against ∼10 s (means) for GPT-2, ∼8.6 against 4.5 min for Llama-2-7B) | `text_numbers` | [5_comparison](experiments/5_comparison/README.md) |
| Sec. 4.6 | prover 35–32,000x faster than the zkSNARKs and 3.0x faster than Maverick (Table 3); zkLLM's code on our L40S about 844 s, 48x, 2.5x per layer for 13B; verifier slower only than ZKML's on VGG-16 (2.2x) and zkLLM's on OPT-1.3B and OPT-6.7B (2.9x and 1.7x) and on Llama-2-7B and Llama-2-13B (2.9x, pending); Maverick with the encoding 20.3 MB (1.8x smaller), 151.6 ms against Maverick's 260.2 ms; proofs 3.6–180x and 2,500–56,000x larger | `text_numbers`; Table 3 (`paper_assets`); `python artifacts/results/zkllm_l40s/summarise.py artifacts/results/zkllm_l40s/llama2-7b-T2048-948715 32` | [zkllm_l40s](artifacts/results/zkllm_l40s/README.md), [5_comparison](experiments/5_comparison/README.md) |
| Sec. 5 | zkLLM's 157–183 kB; CPU verifier up to 3.7 min; GPU verifier 1.7–2.9x slower than zkLLM's; setup up to 13x slower | `text_numbers` | [5_comparison](experiments/5_comparison/README.md) |

Published numbers (Table 3, Fig. 4's grey points, zkLLM's proof sizes, the path protocol's time per
path) come from `artifacts/comparison/literature/reported_benchmarks.csv` (413 published measurements
from 32 systems, each with its table, page and a verbatim snippet) through `literature.py`.

## From scratch

| Part | Hardware, time | Inputs | Steps |
|---|---|---|---|
| MNIST experiments (Sec. 4.2 *Reproduction*, *Attacks on MNIST*, *Other samplers*) | CPU, about 45 min in total | MNIST (downloaded on first use); the included MNIST models (`train.py` skips them; `--force` retrains) | `1_reproduction/run.py`, `2_attack/run.py`, `3_sampling_fixes/run.py`, `floor_sampler.py`; they rewrite `artifacts/results/*.json`. See [1](experiments/1_reproduction/README.md), [2](experiments/2_attack/README.md), [3](experiments/3_sampling_fixes/README.md) |
| The benchmark: both protocols, the path protocol's cells, the attacks on larger models (Tables 1–3, Figs. 3–4, Sec. 4.2–4.6) | one CUDA GPU and 8 CPU threads per job (the paper: an NVIDIA L40S and an AMD EPYC 9334, on TAU's SLURM cluster); jobs of minutes to about two hours, tens of jobs per protocol | CIFAR-10 and ImageNet (neither is downloaded by the code) in `code/data/cifar10/` and `code/data/imagenet/`, or on the cluster at the exported `$PVI_CIFAR_ROOT` and `$PVI_IMAGENET_ROOT`, which every CIFAR or ImageNet job needs, `smoke.sbatch` included; the CNN weights, which are not included and must be trained first | `4_defence_benchmark/train.py` (or `slurm/train.sbatch`), then `bench.py` jobs under a new platform name (`slurm/strong_gpu.sh` for the basic protocol; the optimised options for the optimised one), then the commands of `5_comparison` with that name. See [4_defence_benchmark](experiments/4_defence_benchmark/README.md) |
| The real OPT-6.7B attack and backdoor (Sec. 4.2 *A real LLM*) | one GPU with 16 GB or more, about 1 h per run, two runs | `transformers` 4.51.3, `tokenizers` 0.21.4, `pyarrow` 25.0.1 (`pip install -r requirements.txt -r requirements-llm.txt`) in the Python named by `$PVI_PYTHON`; `facebook/opt-6.7b` downloaded into `$HF_HOME` (open, no token) | `real_llm.sbatch`, once as is and once with `--target auto`. See [REAL_LLM.md](experiments/2_attack/REAL_LLM.md) |
| The real-OPT perplexities (Sec. 4.1) | one GPU | the same packages; the OPT checkpoints downloaded beforehand (the jobs run offline); WikiText-2's test split as parquet at `$WIKITEXT_PARQUET` | `real_weights_ppl.sbatch`. See [REAL_LLM.md](experiments/2_attack/REAL_LLM.md) |
| zkLLM's code on our GPU (Sec. 4.6, Table 3) | one L40S, hours | zkLLM's own checkout (commit `993311e`) and environment; the gated Llama-2 weights | `zkllm.sbatch`. See [zkllm_l40s](artifacts/results/zkllm_l40s/README.md) |

The SLURM scripts run from the folder that holds `code/` (they `cd` into `code/` themselves), and the
commands in the READMEs write their logs to `logs/` there. The benchmark's scripts (`train.sbatch`,
`bench.sbatch`, `smoke.sbatch`) use the Python of `.venv/` in that folder unless `$PVI_PYTHON` names
another; `real_llm.sbatch` and `real_weights_ppl.sbatch` require `$PVI_PYTHON` (a Python with both
requirements files installed) and stop without it; `zkllm.sbatch` uses zkLLM's own environment.

[`experiments/6_improvements`](experiments/6_improvements/README.md) holds the optimised protocol's
development harnesses. Most of them are A/B tools that compare this code with an older checkout
(`--base`), so they cannot run from the submission alone; of their results only `wire.py`'s
(`results/wire_laptop.json`) backs a number of the paper, the 16–19 bits per claim of Sec. 3.6.2.

## Notes on reproducibility

* The MNIST models are included because training is reproducible only on one platform: a rerun on
  the same machine, library versions and thread count gives the same weights, but another machine
  gives slightly different, equally accurate ones, and the MNIST numbers then move in the last digits.
  Check the included weights with `cd artifacts/models && sha256sum -c MODELS.sha256`.
* The protocols draw fresh random challenges on every run. Deterministic values (exact probabilities
  computed by `pvi.experiments.analysis`, proof bytes, parameters, everything derived from the stored
  records) reproduce exactly; empirical rates move slightly between runs. For example, a rerun of the
  reproduction gave 17–37% acceptance without tolerance (the paper: 21–37%) and detection down to 99.7%
  (the paper: 99.8–100%).
* The language models use their exact shapes with random int8 weights, since costs depend only on
  shapes, except the compact encoding's size (`pvi.fullcheck.real_weights` loads real OPT checkpoints
  into the same graphs for the perplexities of Sec. 4.1). Every decoder the paper tabulates is measured
  at full depth; only the 30–70B shapes are extrapolated linearly from 1- and 2-block builds, labelled
  `extrapolated_from_1_and_2_blocks` in `llm_full_model.csv`.
* Every raw record carries the git commit, host, GPU and CPU it was measured on.
* The figure PDFs depend slightly on the matplotlib version: 3.11.2 reproduces the committed ones byte
  for byte, while other versions draw them slightly larger or smaller. After regenerating them with
  another version, check that the paper's text still ends on page 8 (the references are on page 9).

## Submission tarball

The submission is `<groupname>.tar.gz`, holding the folder `<groupname>/` with two subfolders:

```
<groupname>/
  code/      this folder: the implementation, the stored measurements and this README
  report/    main.pdf and its LaTeX source (main.tex, figures/, tables/, references.bib, the ACM class files)
```

Nothing else goes in. From a checkout of the project repository:

```bash
git -c core.autocrlf=false archive --format=tar.gz --prefix=<groupname>/ -o <groupname>.tar.gz HEAD code report
```

Every file must keep its LF line endings: with CRLF the `.sh` and `.sbatch` scripts fail on Linux and
`sha256sum -c MODELS.sha256` fails. The repository's `.gitattributes` keeps every `.sh`, `.sbatch`,
`.py` and `.sha256` file in LF in every checkout and archive; `core.autocrlf=false` keeps the other
files as committed (LF) too, which matters on Windows, where `core.autocrlf` is often `true`. Copying
`code/` and `report/` into `<groupname>/` and running `tar -czvf <groupname>.tar.gz <groupname>/`, as the
course describes, works too if the copies have LF line endings.

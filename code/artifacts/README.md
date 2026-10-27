# Stored models, records and results

This folder holds what the experiments read and what the repository keeps of their outputs: the trained MNIST
networks of the attack (`models/`), the benchmark's raw records, the tables derived from them and the catalogue of
published results (`comparison/`), and the stored outputs of the other experiments (`results/`). For every file or
family of files, this page gives what it holds, the script and the command that write it, the platform it was
measured on and what reads it.

Every measured number of the paper comes from one run of the run package
([`experiments/package/README.md`](../experiments/package/README.md)). The run adds `comparison/raw_final/` (the
records of its `bench` and `batch` tiers, written by `costs/bench.py --platform final`) and `comparison/tables_final/`
(`python -m experiments.comparison.aggregate --platform final`); every benchmark number of the paper, the outcome
counts included, comes from those two folders, and every other measured number from the run's job logs
(`logs/final/` at the repository root while the run writes them, stored afterwards as `final/`;
[`experiments/paper/README.md`](../experiments/paper/README.md)). [The final run's
records](#the-final-runs-records) says what the artifact keeps of them. The published
systems' values come from `comparison/tables/reported_curated.csv` (and zkLLM's prover on the run's GPU model from
`results/zkllm_l40s/`), and `tab:params` and `tab:setup` from the library's parameters. The paper's generator reads none of the stored roots below. The stored files serve as inputs
that scripts need (the MNIST networks, the literature catalogue), as the records behind the values the paper prints
before the final run (`paper_sp2027/fallback/`, checked against them by the tests named below), and as reference
measurements of other platforms.

Commands run from `code/` with `PYTHONPATH=src:.` (Windows: `PYTHONPATH="src;."`); paths are relative to this folder
unless they start with `artifacts/`. The provenance columns state what each file's own content records: the commit
(`git_sha`, `git` or `env.git_sha`; commits of this repository's history), the GPU and CPU models (`prover_hw`,
`verifier_hw`, `env.gpu`, `env.cpu`, `gpu`, `cpu`), library versions and the command's options (`config`, `args`).
Where a file records none of them, the entry says "provenance not recorded in the file" and names the script that
writes files of that format.

| Path | Contents | Written by | Read by |
|---|---|---|---|
| `models/` | the seven MNIST networks of the attack, their training reports, `MODELS.sha256` | `attack/train_mnist.py` | the attack scripts, `costs/bench.py`, `costs/perf.py`, `costs/wire.py`, two tests |
| `models/cnn/` | not in the repository: the benchmark's image models (`<model>.pt`, `<model>.json`) | `costs/train_cnn.py` | `costs/bench.py`, `costs/perf.py` (`experiments.common.models.trained_cnn`) |
| `comparison/` | the benchmark's raw roots `raw_<platform>/`, their tables `tables_<platform>/`, `tables/`, `literature/`, `excluded_cells.csv`; described in [`comparison/README.md`](comparison/README.md) | `costs/bench.py`, `comparison/aggregate.py`, `comparison/literature.py`, `attack/path_test_bound.py` | `experiments/comparison/`, `experiments/paper/`, `tests/test_comparison_tables.py`, `tests/test_paper_numbers.py` |
| `results/` | the attack's documents, the sampler's distances, the quality records, development measurements, zkLLM's demo on an L40S | the scripts of `experiments/attack/`, `generation/`, `quality/`, `costs/`, `deployment/`, `v1/` | `tests/test_paper_numbers.py` (the fallback); `experiments/paper/bench.py` (`zkllm_l40s/summary.csv`) |
| `cache/` | not in the repository: the dataset cache of `pvi.models.datasets` (`PVI_CACHE`) | `pvi.models.datasets` | `costs/bench.py` (image models) |
| `final/` | added after the final run: its job logs, documents and dry-run listing, and the machine's description ([below](#the-final-runs-records)) | `experiments/package/run.sh` (a copy of `logs/final/`) | `experiments/paper/numbers.py --logs artifacts/final` |

`.gitignore` excludes `models/cnn/` and `cache/`; everything else listed here is committed (`final/` once the run
is done).

## `models/`: the MNIST networks

| File | Holds | Provenance |
|---|---|---|
| `mlp_mnist_full.npz`, `.json` | the MLP 784-512-256-10 of the single-neuron attack, the samplers, the adaptive game, the `quantised` reproduction setting and the benchmark's `mlp_mnist` cells (test accuracy 0.9772) | seed 0, 6 epochs |
| `mlp_digits_01234`, `mlp_digits_05678` (`.npz`, `.json`) | the committed and the substitute model of the `shared_class` setting (digits 0-4 against 0, 5-8) | seeds 0 and 1 |
| `mlp_mnist_halfA`, `mlp_mnist_halfB` | the `disjoint_data` setting (two halves of the training set) | seeds 0 and 1 |
| `cnn_mnist_halfA`, `cnn_mnist_halfB` | the `cnn_disjoint_data` setting (a small CNN) | seeds 0 and 1, 4 epochs |
| `MODELS.sha256` | the SHA-256 of each `.npz` | no script writes it |

* **Written by** `python -m experiments.attack.train_mnist --force` (the specifications of `pvi.mnist.zoo`; without
  `--force` the stored models are skipped, `--only NAME` trains one). Each `<name>.json` is the training report
  (train and test accuracy, final loss, epochs, seed): provenance not recorded in the file (no commit, hardware or
  library version). Training is bit-exact only on one machine and library build, so the weights the paper used are
  stored and pinned: `sha256sum -c MODELS.sha256`, run in `models/`, accepts all seven.
* **Read by** the MNIST scripts of `experiments/attack/` (`reproduction`, `attack_mnist`, `samplers`,
  `floor_sampler`, `adaptive_game`; option `--models`, default this folder), `experiments.common.models.trained_mlp`
  (`costs/bench.py` and `costs/perf.py`, model `mlp_mnist`), `costs/wire.py` (`mlp_mnist`), and the tests
  `tests/test_adaptive_game.py` and `tests/test_wire.py` (`mlp_mnist_full.npz`).
* **`models/cnn/`** (not committed): `python -m experiments.costs.train_cnn --model <model>` for `lenet5`, `vgg11`,
  `vgg16`, `resnet18_cifar`, `resnet18_224` and `resnet18_224_squirrel` writes `<model>.pt` (the state dict) and
  `<model>.json` (accuracy, recipe, seed, time, `env`). The run package's `bench` tier stops before submitting
  anything when one of them is missing.

## `comparison/`: the benchmark's records and tables

[`comparison/README.md`](comparison/README.md) describes the roots, their cells and the withheld timings. Every root
`raw_<platform>/<suite>/<model>/<cell>.jsonl` is written by the benchmark harness `costs/bench.py` at the commits
listed below (`python -m experiments.costs.bench cnn|llm --platform <platform> ...`; the jobs of `package/run.sh`'s
`bench` and `batch` tiers); a finished cell has a `<cell>.done` marker. Every row carries `git_sha`, `prover_hw` and
`verifier_hw`; the first row of a cell (`metric: env`) carries `env` with the commit, `git_dirty`, the GPU, CPU,
torch, CUDA and Python versions and the thread count, and, in every root but `raw/`, the driver, NumPy and
torchvision versions, `lscpu` and `nvidia-smi`. `PLATFORM.json` holds the root's GPU and CPU (and, except in `raw/`,
the driver and torch version); `"frozen": true` makes `bench.py` refuse new records.

The stored roots, as their records give them (cells: finished cells; records: rows other than the `env` rows; every
root records torch 2.5.1+cu121, CUDA 12.1 and Python 3.12.3):

| Root | GPU (prover) | CPU (verifier) | Cells | Records | Commits of the records | Read by |
|---|---|---|---:|---:|---|---|
| `raw_l40s/` | NVIDIA L40S | AMD EPYC 9334 32-Core Processor | 633 | 123,832 | `9401431` | no test; read when a command names it: those of `experiments/comparison/README.md` (`aggregate.py --platform l40s`, `count_outcomes.py --platform l40s`, `fingerprint_check.py` with it as the reference root) |
| `raw_l40s_improved/` | NVIDIA L40S | AMD EPYC 9334 32-Core Processor | 195 | 68,363 | `c8be5eb` | `tests/test_comparison_tables.py` (`test_the_stored_counts_are_the_papers`) |
| `raw_l40s_improved2/` | NVIDIA L40S | AMD EPYC 9334 32-Core Processor | 311 | 83,896 | `366e3d4` (80,403 records), `c2bdc49` (3,493) | `tests/test_comparison_tables.py` (the same test) |
| `raw_l40s_improved3/` | NVIDIA L40S | AMD EPYC 9334 32-Core Processor | 22 | 1,600 | `7016557` | `tests/test_comparison_tables.py` (the same test) |
| `raw_l40s_improved2_thr1/` | NVIDIA L40S | AMD EPYC 9334 32-Core Processor (one thread) | 14 | 6,104 | `366e3d4` | nothing |
| `raw_rtx2080ti-v2/` | NVIDIA GeForce RTX 2080 Ti | Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz | 371 | 78,266 | `9401431` | no test; read when a command names it: `fingerprint_check.py artifacts/comparison/raw_l40s artifacts/comparison/raw_rtx2080ti-v2` |
| `raw/` | NVIDIA GeForce RTX 2080 Ti | Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz | 308 | 59,547 | `f97381f` (40,477), `9a51065` (8,246), `9651d27` (7,254), `ab05f61` (3,570) | nothing (`bench.py` names it `--platform rtx2080ti` and refuses to write to it) |

The CPU column is the verifier's processor; the cells with the verifier on the GPU (`_gpuv`) record
`NVIDIA L40S (GPU client) + AMD EPYC 9334 32-Core Processor x8 threads`. The records of one root name one GPU model
and one CPU model; the driver versions differ between the nodes of a root (535.183.01, 580.126.09 and 580.173.02 on
the L40S roots, 575.64.03 on `raw_rtx2080ti-v2/`; `raw/` records none). `test_the_stored_counts_are_the_papers`
counts the defining cells of the three `raw_l40s_improved*` roots and checks the outcome counts that the fallback
prints (3,792 honest queries accepted, 2,512 attacks rejected: 1,601 by Freivalds' check, 853 by the column code
check, 56 by the Merkle check and 2 by the range check). `raw_l40s/` holds 2 and `raw_l40s_improved2/` 17 files of
interrupted attempts (`<cell>.jsonl.part.<time>`), which no reader counts. The final run adds `raw_final/` and
`tables_final/`; every benchmark number of the paper comes from those two folders.

The other files of `comparison/`:

| Path | Holds | Written by | Provenance | Read by |
|---|---|---|---|---|
| `tables_<platform>/measured_summary.csv` (`l40s`, `l40s_improved`, `l40s_improved2`, `l40s_improved2_thr1`, `l40s_improved3`, `rtx2080ti-v2`), `llm_full_model.csv` (all but `l40s_improved2_thr1`, which holds no language model), `llm_extrapolation_check.csv` (`l40s`, `l40s_improved2`, `l40s_improved3`, `rtx2080ti-v2`: the roots with a full build that has 1- and 2-block twins) | per cell and metric the median and mean; the full-model cost of each language model (measured, or extrapolated from 1- and 2-block builds); every full build against its extrapolation | `python -m experiments.comparison.aggregate --platform <platform>` (rebuilds a root's files byte for byte from the stored root) | rows carry `prover_hw` and `verifier_hw`; `measured_summary.csv` also `git_sha` (the root's commits) | nothing in `code/`; `experiments/paper/` reads `tables_final/` |
| `tables_<platform>/llm_extrapolation_validation.csv` (`l40s`, `l40s_improved2`, `l40s_improved3`, `rtx2080ti-v2`) | a line fitted through every block count of each full build | a script that is not part of this artifact | `prover_hw` only; provenance not recorded in the file otherwise | nothing |
| `tables/reported_curated.csv` | the 41 published results the paper compares with, in seconds and bytes, each linked to its catalogue row | `python -m experiments.comparison.literature` (rebuilds it byte for byte) | per row: the system's own hardware, the source location and a verbatim snippet | `experiments/paper/bench.py` (`literature()`: `tab:eval-compare`, `tab:eval-compare-v1` and the comparison's text numbers), `tests/test_paper_numbers.py` |
| `literature/reported_benchmarks.csv` | 413 published measurements of 32 systems, each with its table or page and a verbatim snippet | no script: the catalogue of the literature search | per row: venue, URL, hardware, source location, snippet | `experiments/comparison/literature.py` |
| `tables/analytic.csv` | the path test's paths for `2^-40` and the bytes of opening the fp16 model and trace, Llama-2-7B at 64 tokens | `python -m experiments.attack.path_test_bound` (rebuilds it byte for byte) | provenance not recorded in the file (shapes only, no hardware) | `tests/test_paper_numbers.py` (as the attack tier's `analytic.csv`; fallback `\PathsLlama` 305,193); the paper reads the run's `logs/<name>/attack/analytic.csv` |
| `tables/measured_summary.csv`, `tables/llm_full_model.csv` | the tables of `raw/` | an aggregation of `raw/` that `comparison/aggregate.py` of this checkout does not reproduce (it gives 6,174 and 944 rows against the stored 6,218 and 980) | rows carry `prover_hw`, `verifier_hw`; `measured_summary.csv` the commits of `raw/` | nothing |
| `excluded_cells.csv` | 17 timings of `raw_l40s_improved2/` withheld from the values printed before the final run (cells slowed by other load on two nodes) and the rule that selected them (in its header) | no script of this artifact | provenance not recorded in the file (the selection rule only) | no script of `code/`: the record of the withheld values |

## `results/`: the stored outputs of the other experiments

### The attack (Sec. 3.5, `ext:attack-measure`, `tab:inrange`)

Written by the scripts of [`experiments/attack/`](../experiments/attack/README.md), whose default `--out` is this
folder; the run package's `attack` tier writes the same documents to `logs/<name>/attack/`, and that is where
`experiments/paper/records.py` reads them. These stored documents are read by `tests/test_paper_numbers.py`
(`test_the_stored_attack_documents_regenerate_the_fallback`): from them the generator rebuilds the fallback's
`tab:inrange` body and the attack's text numbers (`\MnistSinglePass`, `\InrangeKMax`, `\AttackOptLPaths`, ...)
exactly. None records a commit, a GPU or a CPU: provenance not recorded in the file beyond the options shown, and
[`experiments/attack/README.md`](../experiments/attack/README.md) states the resources each script needs.

| File | Holds | Command (from `code/`) | Recorded options |
|---|---|---|---|
| `reproduction.json` | `RandPathTest` in the four substitute settings: completeness with and without the 1e-4 tolerance, other-model detection | `python -m experiments.attack.reproduction` | `config`: 150 queries, 20 challenges, every setting |
| `attack.json` | the single-neuron attack on `mlp_mnist_full`: the forgeries of Anchuri et al. against the single-neuron edit, the backdoor, the opening budget, the stealthy edits | `python -m experiments.attack.attack_mnist` | `config`: 200 queries, 25 challenges |
| `defence.json` | the samplers against the plain and the adaptive edit, and contribution weighting's zero blind spot | `python -m experiments.attack.samplers` | `config`: 40 queries |
| `floor_sampler.json` | the zero-aware contribution sampler for each floor against the better of two attacks | `python -m experiments.attack.floor_sampler` | `n_queries` 20 only |
| `adaptive_game_mlp_mnist.json` | the adaptive game on both hidden layers: samplers, LP mixture, support limits, the tamper locator | `python -m experiments.attack.adaptive_game` | `config`: layers 1 and 2, 40 queries, 5,000 calibration images, caps, pool 96, 10 double-oracle rounds, 2 threads |
| `real_llm_attack.json` | one FFN neuron of OPT-6.7B against the next token, the backdoor to ` hacked`, reachability, the path test | `python -m experiments.attack.attack_llm` | `model`, `config` (`opt-6.7b`), `layers`, `trigger`, `target`, `topk` |
| `real_llm_attack_auto.json` | the same with the automatically chosen target (` back`) | `python -m experiments.attack.attack_llm --target auto --out artifacts/results/real_llm_attack_auto.json` | the same fields |
| `inrange_llm_opt-1.3b_any.json`, `inrange_llm_opt-1.3b_down.json` | in-range edits against per-neuron range checks on OPT-1.3B, to either bound and to 0 only | `python -m experiments.attack.inrange_llm` and `python -m experiments.attack.inrange_llm --moves down` | `args`: layers 6, 12, 18, 23, 40 prompts, 64 x 128 calibration tokens, cap 256, 8 threads |
| `inrange_llm_opt-6.7b_any.json`, `inrange_llm_opt-6.7b_down.json` | the same on OPT-6.7B | `python -m experiments.attack.inrange_llm --model facebook/opt-6.7b --layers 8 16 24 31 --threads 16` (and `--moves down`) | `args`: as above, 16 threads, device `cpu` |

The sampler game on OPT-1.3B (`adaptive_game_opt-1.3b.json`, `python -m experiments.attack.adaptive_game_llm
--threads 8 --device cuda`) is not stored.

### The sampler's distances (Sec. `sec:gen-sampling`, `sec:ext-sampling`)

Written by `generation/sampler_tv.py` ([`experiments/generation/README.md`](../experiments/generation/README.md)).
Each document records `script` and `args` (the options) and `meta` (model, windows, the recipe's statistics, the
dataset); no commit, GPU or CPU model: provenance not recorded in the file beyond these. The two model documents
hold the percentile recipe's graphs (`meta.recipe`: norm gain 4, alpha 0.55), `--recipe legacy` of this checkout's
script; the run's `tv` tier measures OPT-1.3B with the certified recipe (the script's default) and OPT-125M with
`--recipe legacy`.

| File | Holds | Command (from `code/`) | Read by |
|---|---|---|---|
| `sampler_tv_opt-1.3b.json` | the exact law of the integer Gumbel-max sampler and its total-variation distance to softmax sampling over 512 positions of OPT-1.3B, three temperatures, two top-k sets, three tables; a Monte Carlo of 100,000 draws at two positions | `python -m experiments.generation.sampler_tv --model facebook/opt-1.3b --recipe legacy --windows 4 --workers 16 --threads 16 --mc 100000 --mc-positions 64 511 --mc-settings 1:0 1:50 1.3:0 --out artifacts/results/sampler_tv_opt-1.3b.json` (CPU only) | `tests/test_paper_numbers.py` (as the tv tier's document: fallback `\TvTopK` 0.0063, `\TvFull`, `\TvModel`, `\TvTwenty`) |
| `sampler_tv_opt-125m.json` | the same on 32 positions of OPT-125M, with the Monte Carlo at two positions | `python -m experiments.generation.sampler_tv --model facebook/opt-125m --recipe legacy --windows 1 --positions 32 --mc 100000 --mc-positions 8 31 --mc-settings 1:0 1:50 1.3:0 --out artifacts/results/sampler_tv_opt-125m.json` | nothing |
| `sampler_tv_selfcheck.json` | the closed forms (2 and 8 tokens) and the Monte Carlo of the toy laws | `python -m experiments.generation.sampler_tv --selfcheck --out artifacts/results/sampler_tv_selfcheck.json` | nothing |

### `quality/`: the integer models' quality (`tab:quality`, `sec:ext-quality`)

| File | Holds | Written by | Provenance | Read by |
|---|---|---|---|---|
| `quality/ppl2048_opt-125m.jsonl`, `ppl2048_opt-1.3b.jsonl`, `ppl2048_opt-6.7b.jsonl` | perplexity on the 140 windows of 2,048 tokens of the WikiText-2 test split: the fp32 reference and the scalar-gain, percentile-smoothing (alpha 0.55) and (OPT-125M, OPT-1.3B) per-row graphs, at 8 and 11 probability bits, at 8 bits also with a float softmax (29, 29 and 13 records) | `quality/ppl_opt.py` at a commit the records do not name; its cell names (`<graph>-p<bits>-<int\|float>`) and `opts` keys (the builder's arguments) differ from those of this checkout's script ([`experiments/quality/README.md`](../experiments/quality/README.md)) | GPU `NVIDIA GeForce RTX 2080 Ti` (`gpu`, device `cuda`); commit and CPU not recorded | nothing |
| `quality/ppl2048_search_round1.jsonl` | the first round of the recipe search on OPT-125M and OPT-1.3B at 2,048 tokens: scalar gains 2-6, smoothing without clipping and with per-channel maxima at alpha 0.3-0.5 (`tag`) | the same | the same | nothing |
| `quality/llama_diag_tinyllama_qwen3-0.6b.txt` | the job's output for Qwen3-0.6B and TinyLlama: per block the cosine of the integer residual stream against the checkpoint and the fp32 stream's largest values (JSON lines, shortened, and the table) | `python -m experiments.quality.llama_diag --model Qwen/Qwen3-0.6B-Base --prefix none --no-qk-balance --recipes scalar4 smooth55` and the same with `--model TinyLlama/TinyLlama_v1.1` | header: commit `8dd89f0`, CPU `Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz`; no GPU | nothing |

The other quality results of the paper (the certified models, the zero-shot accuracy, the Llama-family recipes) are
not stored: the run's `quality` tier writes them to its logs.

### `development/`: measurements of implementation variants

None of these files is read by a test or by `experiments/paper/`. `experiments/costs/README.md` names
`wire_laptop.json` as the stored measurement of the compact encoding (`costs/wire.py --summary` prints its tables);
no script reads the others.

| File | Holds | Written by | Provenance |
|---|---|---|---|
| `development/wire_laptop.json` | the compact encoding against the uncoded proof (bits per claim, bytes and times per part, the codec's throughput) for 8 specs | `python -m experiments.costs.wire --mnist <dir> lenet5 vgg16 resnet18_cifar mlp_mnist gpt2:64 gpt2:512 qwen3-4b:8:1,2,4,8 llama2-7b:64:1,2 --out artifacts/results/development/wire_laptop.json`; `--summary` prints its tables | commits `9dc0c86` (the image models and the MLP), `f3e40a7` (GPT-2 at 64 tokens, Qwen3-4B), `e3b5715` (GPT-2 at 512 tokens, Llama-2-7B) in `git`; CPU model not recorded |
| `development/perf_codec_laptop.jsonl` | the compact encoding off and on (`wire`) at two commits (`label` `base` and `new`), GPT-2 in modes C and Kpre and Qwen3-4B in mode Kpre | `costs/perf.py` (records without `env`) | commits `6c2059a` and `40f1742`; device `cpu`, CPU model not recorded |
| `development/perf_col_laptop_{gpt2,lenet5,vgg16}.jsonl` | mode C under one tree per weight op (`policy` `paper`, the records' name of `per-op`) against grouped plans (`R16`, `cnn17`; `group_columns`: the columns opened per tree group), `wire` off and on | `costs/perf.py` | commit `7b228c4`; CPU model not recorded |
| `development/perf_combined_laptop_{gpt2,vgg16}.jsonl`, `perf_combined_laptop_lenet5{,_thr4}.jsonl`, `perf_combined_laptop_qwen_{interleaved,separate}.jsonl` | the plans and the compact encoding together (`per-op` against `R16`, `cnn16`, `cnn17`, `wire` off and on); for Qwen3-4B in mode Kpre, the encoding's A/B interleaved in one process and in two separate processes | `costs/perf.py` | commits `9dc0c86` (GPT-2, VGG-16), `3868f78` (LeNet-5), `d4a8999` (Qwen3-4B); CPU model not recorded |
| `development/perf_lookup_laptop_{gpt2,gpt2_kpre,qwen_kpre}.jsonl` | the verifier reading the embedding rows itself (`lookups`) | `costs/perf.py` | commit `360984b`; CPU model not recorded |
| `development/perf_plans_laptop_{cnn,cnn_thr4,gpt2}.jsonl` | the commitment plans (`policy`) at 1 and 4 threads | `costs/perf.py` | commits `de0f24c`, `50a2ba5` (1 thread), `f31c5fb`, `36de7a2` (4 threads), `7a9d598` (GPT-2); CPU model not recorded |
| `development/serve_ab_laptop_{tiny,small_llama,gpt2_L1,gpt2_L1_C_tightc}.jsonl` | the prover against unverified serving and the proof on request, on the small decoders, `small-llama` and one GPT-2 block | `deployment/serve_ab.py` | commit `cdeb44a`; CPU `Intel64 Family 6 Model 140 Stepping 1, GenuineIntel` (`cpu`), torch 2.2.2+cpu; no GPU |
| `development/long_prompts_v0_v1.jsonl` | Llama-2-7B at 2,048 tokens in mode Kpre: v0 against V1 bytes on 2 and 3 blocks (`v1/v1_bytes.py`) and two timings of protocol v0, 3 queries each, with the CPU verifier (`costs/perf.py`, `label` `vlonga`) | `v1/v1_bytes.py`, `costs/perf.py` | commit `12a188f`; GPU `NVIDIA GeForce RTX 2080 Ti` (`gpu`; the two perf records ran prover and verifier with `device` `cpu`); CPU model not recorded |

The perf records name their settings in `label`, `model`, `mode`, `policy`, `wire`, `lookups`, `prune_last` and
`threads`; their commits identify the code that measured them.

### `zkllm_l40s/`: zkLLM's public demo on an L40S

zkLLM's code (its commit `993311e`, unchanged) run for Llama-2-7B and Llama-2-13B at 2,048 tokens, with the job
script that ran them and the summary (`summary.csv`); described in
[`results/zkllm_l40s/README.md`](results/zkllm_l40s/README.md). The timing shims, the input generator and
`summarise.py` are in [`experiments/comparison/zkllm/`](../experiments/comparison/zkllm/README.md), whose
`run_zkllm.sh` the run package's optional `zkllm` tier runs. Each run folder's `env.txt` records the GPU
(NVIDIA L40S), the CPU (AMD EPYC 9334 32-Core Processor) and zkLLM's commit; `summary.csv` records no provenance and
is written by `summarise.py --csv`. `experiments/paper/bench.py` (`literature`, `zkllm_runs`) reads `summary.csv` and
the runs' `env.txt` with the published values: the paper sets zkLLM's prover on this GPU model against ours on the
same prompts (`\ZkllmSameGpu...`, Sec. `sec:eval-compare` and `ext:disc-options`), only when the final run's GPU is the
one `env.txt` names, and only when the final run's own `zkllm` tier did not measure both models on that GPU (its
records then take their place).

### Written by a run, not stored

`results/adaptive_game_opt-1.3b.json` (`attack/adaptive_game_llm.py`), `results/ev_breakeven.{json,tex,pdf}`
(`deployment/ev_breakeven.py`), `results/development/wire.json` (`costs/wire.py` without `--out`), `models/cnn/` and
`cache/`.

## The final run's records

The run writes its job logs and documents to `logs/final/` at the repository root and the benchmark's records to
`comparison/raw_final/`. Once it is done, the artifact stores everything the paper's numbers are read from:

* `comparison/raw_final/` as the run wrote it, and `comparison/tables_final/`
  (`python -m experiments.comparison.aggregate --platform final`);
* `final/`: a copy of `logs/final/` without its caches (`triton/`, `native/`, `setup_cache/`, `llamacpp/`,
  `ref_*.pt`), that is every job log `sp-<cell>-<job id>.out` (the records are its lines that start with `{`), the
  attack tier's `attack/` documents, the tv tier's `sampler_tv_*.json`, the dry-run listing `dryrun.txt`, the
  nodes' environment snapshots and the inputs' hashes `env/`, the GPU telemetry of every job `telemetry/` and the
  smoke's pytest logs `smoke_pytest-*` (`experiments/package/README.md`, Secs. 5 and 6).

From `code/`, `PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs artifacts/final --platform final
--check --expect artifacts/final/dryrun.txt` checks the
stored records, and without `--check` writes every table body and text number of the paper from them
([`experiments/paper/README.md`](../experiments/paper/README.md)). The logs record the run machine's host name and
the records its checkout's absolute paths, so the private rewrite rules of `tools/export_anonymous.py` must name both
before the anonymous snapshot is exported.

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
`artifacts/results/*.json`, which Route B re-creates in about 45 minutes on a CPU, next
to the real-LLM results of §4.1–4.2 and zkLLM's run on our GPU (§4.4), which need a GPU.
Everything can also be re-measured from scratch.

## Layout

```
code/
  src/pvi/                 the library (see src/pvi/README.md)
  experiments/             one folder per result; each has a README
    0_train_models/        train the MNIST models used by 1-4        CPU   ~2 min
    1_reproduction/        the original protocol on its threat model  CPU   ~1 min
    2_attack/              the single-neuron attack and backdoor      CPU   ~2 min
                           (and on real OPT weights: REAL_LLM.md)     GPU   ~1 h per run
    3_sampling_fixes/      smarter samplers, and why they fail        CPU  ~40 min
    4_defence_benchmark/   our defence vs. the path test, CNNs + LLMs GPU  hours (SLURM)
    5_comparison/          tables, counts and every report figure     CPU   ~1 min
  artifacts/models/        the MNIST models the report used (committed, with MODELS.sha256)
  artifacts/results/       the JSON results of experiments 1-3 (Table 1, §3.2, §4.2), of the
                           real-LLM runs (§4.1, §4.2) and zkLLM's timings on our GPU (§4.4)
  artifacts/comparison/    the benchmark's raw records, derived tables and the literature
  tests/                   234 tests (237 with transformers); 44 need a GPU, which adds 9 more
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

The real-LLM experiments of §4.1–4.2 (`experiments/2_attack/REAL_LLM.md`) also need
`transformers` and `pyarrow`, and the 3 tests of `tests/test_real_weights.py` need
`transformers`; neither is in `requirements.txt`. The runs used a separate venv with
`transformers==4.51.3`, `tokenizers==0.21.4` and `pyarrow==25.0.1` on top of the pinned
versions; the real-LLM sbatch scripts take its python as `$PVI_PYTHON`. zkLLM's run
(§4.4) uses zkLLM's own environment (see *Route B*).

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
come from `artifacts/comparison/raw_l40s/`, the 123,832 raw records of platform `l40s`
(see *The stored benchmark runs* below). These commands rebuild every derived table,
the report's figures and generated tables, and the PDF:

```bash
python experiments/5_comparison/aggregate.py --platform l40s               # raw_l40s/ -> tables_l40s/measured_summary.csv, llm_full_model.csv, llm_extrapolation_check.csv
python experiments/5_comparison/literature.py                              # published numbers -> tables/reported_curated.csv
python experiments/5_comparison/analytic.py                                # the path test's cost of 2^-40 on Llama-2-7B -> tables/analytic.csv
python experiments/5_comparison/count_outcomes.py --platform l40s          # the soundness counts of Section 4.3
python experiments/5_comparison/paper_assets.py --platform l40s            # -> ../report/figures/*.pdf, ../report/tables/*.tex
python experiments/5_comparison/text_numbers.py --platform l40s            # optional: every number typed in main.tex, recomputed
python experiments/5_comparison/validate_extrapolation.py --platform l40s  # optional: -> tables_l40s/llm_extrapolation_validation.csv
cd ../report && latexmk -pdf main.tex                                      # or: tectonic -X compile main.tex
```

§4.1 and §4.3 also quote the second platform: `count_outcomes.py --platform rtx2080ti-v2`
gives its verdicts, and `fingerprint_check.py artifacts/comparison/raw_rtx2080ti-v2
artifacts/comparison/raw_l40s` compares the 1,978 hardware-independent numbers the two
runs share (0 differences).

`--platform` defaults to `$PVI_PLATFORM` when it is set (as after Route B's `export`), else
to `l40s`. `aggregate.py --platform rtx2080ti` rebuilds the earliest run's
`measured_summary.csv` and `llm_full_model.csv` in `tables/` from `raw/`, for provenance
only: nothing in the report reads them (the other two files of `tables/`, the published
numbers and the path test's cost, are shared by every platform). The committed figures were
made with the pinned matplotlib 3.11.2; another version draws them slightly larger or
smaller, which can change the page count (see the last note below).

### The stored benchmark runs

| Root | Records | What it is |
|---|---|---|
| `raw_l40s/` | 123,832 | **The report's numbers.** Every tier of `strong_gpu.sh` (`must`, `should`, `nice`; in `experiments/4_defence_benchmark/slurm/`) on the `killable` partition: an NVIDIA L40S (48 GB) prover and 8 threads of an AMD EPYC 9334 verifier, nodes n-801..804 (the L40S node t-806 has a different CPU and was excluded; `bench.py` refuses to mix CPU models in one root), run from a clean clone of this code (commit `9401431`, stored in every record) with `PVI_PLATFORM=l40s` (the script at that commit also submitted a kernel microbenchmark, `microbench`, which writes only a log and no records; it is not in this version). Every decoder up to 13B parameters is built at full depth (`--lean`) at up to 2,048 tokens, Llama-2-7B also at 4,096; the 30–70B shapes (OPT-30B, OPT-66B, Llama-2-70B) only with 1 and 2 blocks (the full Llama-2-70B build is skipped with a `SKIP` line: 64.2 GiB of int8 weights). The full models are also attacked (§4.3). The root also holds variants of the same protocol: `_gpuv` (the verifier on the GPU, §4.4), `_batch` (8 and 32 prompts per proof, §4.4; for the CNNs 8 to 256 images, 8 to 128 for the 224-pixel ResNet), `_thr1` (one verifier thread: the Maverick row of Table 4), and the controls `_thr12`, `_tf32`, `_nofix` (the earliest run's weight cache, `PVI_LEGACY_WEIGHT_KEY=1`) and `_nolean`, which enter no table or figure. All of them count as runs in §4.3. Frozen (`"frozen": true` in its `PLATFORM.json`: the code refuses to add records). |
| `raw_rtx2080ti-v2/` | 78,266 | The second platform, quoted in §4.1 and §4.3 (identical hardware-independent numbers, same verdicts). `strong_gpu.sh must` without `ab-opt13` (the non-lean OPT-1.3B control at 2,048 tokens, about 27 GiB, too large for the card) and `microbench` (a kernel microbenchmark that writes only a log, no records): 29 SLURM jobs, 945088–945116, on `studentbatch`: an RTX 2080 Ti (11 GB) prover and 8 threads of a Xeon Silver 4114 verifier, from the same commit `9401431`, with `PVI_PLATFORM=rtx2080ti-v2`. Every decoder is built at full depth except Llama-2-13B, whose 12.1 GiB of int8 weights exceed the card (it is extrapolated from its 1- and 2-block builds). Controls `_nofix`, `_nolean`, `_thr1`. Frozen. `aggregate.py --platform rtx2080ti-v2` rebuilds its stored tables unchanged; the report quotes only its verdicts and record total (`count_outcomes.py --platform rtx2080ti-v2`) and its hardware-independent numbers (`fingerprint_check.py`). `paper_assets.py` refuses this platform, since it lacks the `should` jobs' cells that the report draws (e.g. OPT-13B at 2,048 tokens). The report's previous version, drawn from it, is commit `d5671bf` in the project's git repository. |
| `raw/` | 59,547 | The earliest run, on the 2080 Ti with an older version of the code. Its committed-weights (C) prover re-uploaded the weights to the GPU twice per query (a weight cache keyed `cuda` vs `cuda:0`), so its C prover times are too slow (the `_nofix` controls of `raw_rtx2080ti-v2/` measure this on the same GPU: Llama-2-7B at 64 tokens 11.0 s against 7.9 s, GPT-2 1.15×, OPT-1.3B at 2,048 tokens 1.16×; within noise for the small CNNs), and every decoder above 12 blocks was built only with 1 and 2 blocks and extrapolated. The report does not use it. It is frozen (the code refuses to write there) and kept for provenance, and as the reference of `smoke.sbatch`'s fingerprint check: `aggregate.py --platform rtx2080ti` still rebuilds its tables in `tables/` exactly (its LLM runs sent one Merkle path per opened column; `multiproof_adjust` in `aggregate.py`, which exists only for this, adds the expected multiproof size next to them). The job list that produced it (`slurm/sweep.sh`) and the report version drawn from it are in the project's git repository at commit `336a03c`, not in the submission archive. |

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
report's CNN weights are not in this repository, since `code/artifacts/fullcheck/models/`
is git-ignored, but they are in that folder of the team's project checkout on the TAU
cluster; new ones make every CNN number incomparable with the report):

```bash
mkdir -p logs
sbatch -o logs/%x-%j.out code/experiments/4_defence_benchmark/slurm/train.sbatch lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel
# when it has finished (squeue -u $USER), the report's jobs, as for raw_l40s/:
export PVI_PLATFORM=<new name> SBATCH_PARTITION=killable SBATCH_GRES=gpu:l40s:1
bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh smoke    # once per GPU type: wait for SMOKE OK
bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh must     # then should, then nice
```

`smoke` runs every test on the GPU (`tests/test_real_weights.py` only where `transformers`
is installed; none may be skipped), small benchmarks into a throw-away root, and
`fingerprint_check.py` against `raw/` (every hardware-independent number the two share must
agree). Each job appends raw records to `artifacts/comparison/raw_<platform>/`, one root
per GPU and CPU model; `raw/`, `raw_rtx2080ti-v2/` and `raw_l40s/` are the stored runs and
are frozen, so re-measure under a new name. `bench.py` refuses records from a second CPU
model in one root, so on a partition whose nodes differ (the L40S node t-806 has a Xeon)
keep the jobs on one kind of node, e.g. with an `sbatch` wrapper on `PATH` that adds
`--exclude`. A cell that has finished is skipped, and a pre-empted job is resubmitted by
running the same tier command again. On a card too small for a build (Llama-2-70B on
48 GB, Llama-2-13B on 11 GB) that build is skipped with a `SKIP` line and its job ends
with an error; the rest of the job is recorded. Then run Route A with
`--platform <new name>`.
`experiments/4_defence_benchmark/README.md` shows how to run a single model without SLURM.

The real-LLM experiments of §4.1–4.2 (OPT checkpoints from Hugging Face, one GPU with
16 GB or more, and `transformers`: see *Installation*) are described in
`experiments/2_attack/REAL_LLM.md`, and zkLLM's run on our GPU (§4.4) in
`artifacts/results/zkllm_l40s/zkllm.sbatch`, with its input builder (`mkinput.py`),
`summarise.py` and the run records next to it. That script runs zkLLM's public code
(commit `993311e`) from its own checkout and environment under `$ZKLLM_HOME`, on the
WikiText-2 test split at `$WIKITEXT_PARQUET`; zkLLM's scripts need `transformers` 4.40.

### Where each result comes from

| Report | Produced by | Output |
|---|---|---|
| Fig. 1, Fig. 2 (schematics) | `5_comparison/paper_assets.py` | `report/figures/overview_*`, `protocol` |
| §4.2 *Reproduction* (acceptance, detection, exact-equality check) | `1_reproduction/run.py` | `artifacts/results/reproduction.json` |
| Table 1 rows 1–8; §4.2 *The attack works* (backdoor, stealth, 250 paths) | `2_attack/run.py` | `artifacts/results/attack.json` |
| Table 1 last row; §4.2 *Smarter sampling* | `3_sampling_fixes/run.py` | `artifacts/results/defence.json` |
| §3.2 the floor sampler | `3_sampling_fixes/floor_sampler.py` | `artifacts/results/floor_sampler.json` |
| §4.1 int8 perplexity of the real OPT-125M/1.3B/6.7B (73.8/37.4/36.0 against 64.7/34.9/27.1, per normalisation gain) | `2_attack/real_weights_ppl.py` (see `2_attack/REAL_LLM.md`) | `artifacts/results/real_weights_ppl_opt-*.json` |
| §4.2 *A real LLM* (40/40 untargeted, backdoor 0/40 for " hacked" and 40/40 for " back", 6,187 reachable tokens, 454,248 paths) | `2_attack/real_llm.py` | `artifacts/results/real_llm_attack.json`, `real_llm_attack_auto.json` |
| §4.2 *The same holds on larger models* (100% flips, 1/84–1/512, 1/28 million, paths for 2^-40; cells `attack_float`, `sampling`) | `4_defence_benchmark/bench.py` → `5_comparison/aggregate.py --platform l40s` | `tables_l40s/measured_summary.csv` |
| Fig. 3, Tables 2–4, Figs. 4–5, all numbers in §4.3–4.4 except zkLLM's own run, the hardware of §4.1 | `4_defence_benchmark/slurm/strong_gpu.sh must`, `should`, `nice` → `5_comparison/aggregate.py --platform l40s` → `paper_assets.py --platform l40s` | `report/figures`, `report/tables` (with `hardware.tex`) |
| §4.1 the extrapolation check on the models built in full (proof size exact; times off by up to 44% for the verifier and 120% for the prover of OPT-2.7B; totals are the sums of the prove and verify parts) | `5_comparison/aggregate.py --platform l40s` (every part, full build against the 1–2 block line); `validate_extrapolation.py --platform l40s` (with a fit through every block count); `text_numbers.py --platform l40s` prints the totals | `tables_l40s/llm_extrapolation_check.csv`, `llm_extrapolation_validation.csv` |
| §4.1 the 1,978 hardware-independent numbers shared with the 2080 Ti run | `5_comparison/fingerprint_check.py artifacts/comparison/raw_rtx2080ti-v2 artifacts/comparison/raw_l40s` | printed |
| §4.3 counts (8,709 honest queries; 2,091 CNN and 804 LLM attacks; Freivalds 2,747, Reed–Solomon 74, Merkle 74), the 2080 Ti's 5,547 and 2,456, and §4.1's 123,832 and 78,266 records | `5_comparison/count_outcomes.py --platform l40s` (and `--platform rtx2080ti-v2`) | printed |
| §4.2 Llama-2-7B: 1/11,008 per path, 305,000 paths, 13.7 GB | `5_comparison/analytic.py` | `tables/analytic.csv` (`anchuri_*` columns) |
| §4.4 *zkLLM on the same GPU* (26.4 s per layer, about 845 s for Llama-2-7B; 34 GiB; 65.7 s per layer for Llama-2-13B) | zkLLM's public code (commit `993311e`) under `artifacts/results/zkllm_l40s/zkllm.sbatch`. The time per layer is the sum of the median times of zkLLM's proof binaries (3 runs on each of layers 0 and 1): for Llama-2-7B QKV linear 6.55 s, softmax/attention 2.79 s, FFN 12.79 s, the two RMSNorms 1.56 and 1.52 s, and the two skip connections 0.58 s each, 26.4 s in all; for Llama-2-13B 14.49, 3.79, 40.34, 2.37 + 2.31 and 2 × 1.21 s, 65.7 s. The binaries' own log (`bin_times.log`), from which these medians were taken, is not stored. From the stored files `summarise.py <run dir> <layers>` recomputes only the median script wall time of each stage, with Python and model loading (per layer they sum to 102 s for Llama-2-7B and 243 s for Llama-2-13B) | `artifacts/results/zkllm_l40s/<run>/` (`time-*.txt`: each script's wall time; `nvidia-smi.csv.gz`: GPU memory every 0.5 s; `verdicts.txt`; `env.txt`) |
| Published results (Table 4, grey points) | `5_comparison/literature.py` | `tables/reported_curated.csv` |
| Numbers quoted only in the text (e.g. 0.4 points, 5–8×, 2.4–4.4×, 21–41×, 0.26 s) | `5_comparison/aggregate.py --platform l40s`; `text_numbers.py --platform l40s` prints each with its line of `main.tex` | `tables_l40s/measured_summary.csv`, `tables_l40s/llm_full_model.csv` (ratios with `tables/reported_curated.csv`) |

## Tests

```bash
python -m pytest tests
```

The tests cover the Merkle commitment, the path test, the exact acceptance-probability
dynamic program against Monte Carlo, the attacks and samplers, and the whole defence.
The defence tests include forged claims, a forged folded row (caught by the
Reed–Solomon check), a forged column (caught by the Merkle check), the lean prover and
verifier, the verifier on a GPU, and GPU/CPU agreement. Without CUDA and without
`transformers` (which is not in `requirements.txt`), `python -m pytest --collect-only -q`
lists 234 tests, or 237 with `transformers` installed (the 3 tests of
`tests/test_real_weights.py`, the real-weight OPT loader on a tiny random OPT, are skipped
without it). Without CUDA the 44 GPU-only tests are skipped: 41 in
`tests/test_gpu_exactness.py` (GPU/CPU bit-exactness at the sizes of the real models, with
TF32 off and on) and 3 in `tests/test_fullcheck.py` (a GPU prover against the CPU verifier
on two tiny CNNs, and a decoder's forward pass on the GPU against the CPU). On a GPU they
run too, together with the 9 GPU-verifier cases of `tests/test_lean_and_verifier_device.py`
(243 tests, or 246 with `transformers`), and `smoke.sbatch` requires that none is skipped
(it leaves out `tests/test_real_weights.py` when `transformers` is missing).

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
  depend only on shapes (`pvi.fullcheck.real_weights` loads real OPT checkpoints into the
  same graphs, for the perplexities of §4.1). In `raw_l40s/` every decoder the report
  tabulates is measured at full depth; only the 30–70B shapes are extrapolated linearly
  from 1- and 2-block builds (`aggregate.py`), labelled `extrapolated_from_1_and_2_blocks`
  in `llm_full_model.csv` (the report quotes only Llama-2-70B's proof size from them, which
  the rule predicts exactly); where a model has both, the full build is used. In
  `raw_rtx2080ti-v2/` only Llama-2-13B is extrapolated, and in the earlier `raw/` every
  decoder above 12 blocks.
* Every raw record carries the git commit, host, GPU and CPU it was measured on.
* The figure PDFs depend slightly on the matplotlib version; build the committed ones
  with the pinned versions of `requirements.txt` (matplotlib 3.11.2, which reproduces
  them byte for byte; matplotlib 3.8, for example, draws Figures 2 and 3 a few points
  shorter), and check that the report's text still ends on page 8 (the references run
  onto page 9).

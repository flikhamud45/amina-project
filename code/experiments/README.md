# Experiments

The scripts that produce every table, figure and number of the paper, one package per part of the paper. Each
folder's README maps the paper's results to its scripts and gives the commands; this page says how to run any of
them and where each part of the paper lives.

| Folder | Part of the paper | Results |
|---|---|---|
| [`attack/`](attack/README.md) | the attack on the path test | the reproduction, the single-neuron attack and backdoor on MNIST and OPT-6.7B, the in-range edits (`tab:inrange`), the samplers and the adaptive game, the path test's bound |
| [`costs/`](costs/README.md) | the cost of protocol v0 | the benchmark (`tab:eval-llm`, `tab:eval-compare`, the outcome counts), the timing harness of the CPU and GPU verifiers, `tab:params`, `tab:setup`, the bits per claim |
| [`improvements/`](improvements/README.md) | each implementation improvement (the extended version) | `tab:eval-improve`, the GPU verifier's time shares, the Kpre precomputation |
| [`v1/`](v1/README.md) | V1 | `tab:eval-v1`, V1's verifier by phase, the GKR prover per leaf |
| [`generation/`](generation/README.md) | generated responses | `tab:gen`, `tab:gen-full`, the sampler's total-variation distance |
| [`reexec/`](reexec/README.md) | the re-execution baselines (llama.cpp's included) | `tab:options`' re-execution and llama.cpp rows, `tab:eval-reexec`, the host-to-device bandwidth |
| [`deployment/`](deployment/README.md) | the deployment case | the prover against unverified serving, `fig:breakeven` |
| [`quality/`](quality/README.md) | the integer models' quality | `tab:quality`, `tab:zeroshot`, the recipe studies, the reproducible build |
| [`comparison/`](comparison/README.md) | the benchmark's tables and the published systems | the aggregated tables, the outcome counts, the literature values of `tab:eval-compare` |
| [`paper/`](paper/README.md) | the paper's numbers | the run's records -> the table bodies and text macros of `paper_sp2027/` |
| [`package/`](package/README.md) | the run | `run.sh`: every job of the final run on one GPU node |
| `common/` | | what the scripts share: the protocol and runtime flags, records, models, checkpoints, tampers, the client in a process of its own |

## Running a script

From `code/`, with the library and the experiments on the path (or after `pip install -e .`):

```bash
cd code
export PYTHONPATH=src:.            # Windows: set PYTHONPATH=src;.
python -m experiments.costs.perf --help
python -m experiments.costs.perf --model gpt2 --layers 2 --seq 64 --modes C Kpre --queries 3 --device cpu
```

Every script takes every parameter as a flag, and `--help` lists them. The defaults are the configuration of the
paper's measured results (the protocol's: `experiments.common.cli.PAPER`):

* **Protocol** (`experiments.common.cli`): mode C's commitment plan `cnn18c` (`--policy`), Fiat-Shamir
  (`--no-fs`: interactive), the compact wire encoding (`--no-wire`), the lean prover (`--no-lean`), the last decoder
  block computed at the last position only (`--no-prune-last`), the verifier reading the embedding rows itself in
  modes K and Kpre (`--no-lookups`), random-weight decoders with 11-bit attention probabilities as the certified
  models (`--prob-max 2047`; `--prob-max 255`: the 8-bit graphs), `--lam 128`, `--rate 4`, and `--setup-cache DIR`
  for the prover's setup cache.
* **Models**: random weights with the benchmark shapes for the cost cells; for real checkpoints the certified OPT
  recipe (`pvi.models.hf_opt.CERTIFIED_RECIPE`) and the main Llama recipe with its fixed public prefix
  (`pvi.models.hf_llama.MAIN_RECIPE`, per checkpoint `main_recipe`). The percentile recipe is `--recipe legacy`.
* **Runtime**: `--device` (the prover's; the GPU when there is one), `--verifier-device` (the CPU), `--threads`
  (8), and every implementation option of `pvi.options` (`--no-native`, `--no-fused-attention`,
  `--v1-verifier baseline`, ...), which change only time and memory, never a result. `--require-native` and
  `--require-triton` stop a script instead of falling back.

Every timed script prints one JSON record per line, with its environment (commit, host, CPU, GPU, library
versions, implementation options and what the host runs) under `env`. The paper's numbers come from one run of the
package on one platform (`package/README.md`); every command also runs on its own, and each README gives a small
version that runs on a laptop CPU.

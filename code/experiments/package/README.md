# The run package

`run.sh` is the one scripted run whose records give every number of the paper. It runs on one GPU node: the GPU
proves and is the GPU verifier, and 8 threads of the node's CPU are the CPU verifier and the re-execution baselines.
The paper's numbers come from an NVIDIA L40S (48 GB) with an AMD EPYC 9334. Each job starts only after the previous
one has ended, so no two timed jobs share the machine, and nothing after the smoke tier starts unless it succeeded.
The jobs write JSON lines; `experiments/paper/numbers.py` turns them into the paper's tables and text numbers.

## 1. What the machine needs

| Need | Why | How to check |
|---|---|---|
| An NVIDIA GPU with int8 tensor cores (sm_75 or later; the paper's: L40S, sm_89) and a CUDA 12.1 driver | the prover, the GPU verifier, serving | `nvidia-smi`; `python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"` |
| 8 free CPU cores and about 256 GB of memory | the CPU verifier and every re-execution baseline run on 8 threads; the 7B cells at 2,048 tokens ask for 256 GB | `lscpu`, `free -g`; with less memory, `--mem-max <GB>` |
| A Python 3.12 environment as `.venv` at the checkout's root (or `--python <path>`) | torch 2.5.1+cu121, `code/requirements.txt` and `code/requirements-llm.txt` | `.venv/bin/python -c "import torch, transformers, triton; print(torch.__version__)"` |
| A C++ compiler in `CXX` | the native CPU kernels (`-march=native`) of the CPU verifier and of the exact int8 re-execution; the CPU cells pass `--require-native`, so a missing compiler stops them instead of timing the torch path | `$CXX --version` (g++ 9 or later) |
| The Python headers for Triton's launcher | Triton compiles a small launcher with `Python.h` on first use | `python -c "import os, sysconfig; print(os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))"`; if `False`, copy the headers into a folder and `export PVI_PY_INCLUDE=<folder>` |
| `HF_HOME` on a node-local disk with about 60 GB free | the `real`, `quality`, `tv` and `attack` tiers download OPT-125M to OPT-6.7B, OpenLLaMA-7B, TinyLlama, Qwen3 and WikiText-2; with `--llama-licence` also Llama-2-7B and Llama-2-13B | `df -h $HF_HOME` |
| The trained image models and their datasets (not in the repository) | the `bench` tier's image cells (MLP, LeNet-5, VGG-11/16, ResNet-18 on CIFAR-10 and on 224-pixel dogs-vs-cats) | `code/artifacts/models/cnn/{lenet5,vgg11,vgg16,resnet18_cifar,resnet18_224}.pt` (`python -m experiments.costs.train_cnn --model <m>`, seed 0); CIFAR-10 in the `cifar-10-batches-py` layout under `code/data/cifar10` (or `$PVI_CIFAR_ROOT`); the dogs-vs-cats cache `dogs_cats.pt` under `code/artifacts/cache` (or `$PVI_CACHE`), or ImageNet under `code/data/imagenet` (or `$PVI_IMAGENET_ROOT`). `run.sh all` and `run.sh bench` stop before submitting anything if one is missing |
| SLURM (`sbatch`, `squeue`), or none | the jobs | with SLURM: `export SBATCH_PARTITION=... SBATCH_GRES=gpu:<type>:1 [SBATCH_ACCOUNT=...]`; without: `--local` (every job runs in the foreground, one after another: start it in `tmux` or under `nohup`) |
| Exclusive use of the node | the timings | every job asks for `--exclusive` |

Every language model of the cost tiers has random weights (the costs depend only on the shapes); real checkpoints
are downloaded by the `real`, `quality`, `tv` and `attack` tiers only. The Llama-2 cells (`real`, `quality`) need the
Llama-2 licence on your own Hugging Face account and your own login, and run only with `--llama-licence`.

## 2. Setup

```bash
git clone <repository> pvi && cd pvi && git checkout <tag>
python3.12 -m venv .venv && .venv/bin/pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && .venv/bin/pip install -r code/requirements.txt -r code/requirements-llm.txt
export SBATCH_PARTITION=<GPU partition> SBATCH_GRES=gpu:<type>:1      # or pass --local to run.sh
export HF_HOME=<node-local folder with 60 GB free>
export CXX=<path to g++>
export PVI_PY_INCLUDE=<headers folder>               # only if the check above printed False
```

`bash code/experiments/package/run.sh --platform final --dry-run all` prints every job it would submit, with each
tier's job count, the sum of its time limits and its expected hours, and submits nothing.

## 3. The smoke and the rehearsal

```bash
bash code/experiments/package/run.sh --platform smoke smoke
#  -> "SMOKE OK" at the end of logs/smoke/sp-smoke-*.out (up to an hour)
bash code/experiments/package/run.sh --platform rehearsal --rehearsal --optional all
#  -> 50 small jobs: every tier's smallest cells (GPT-2 / OPT-125M, 2 blocks), about three hours
cd code
python -m experiments.comparison.aggregate --platform rehearsal
python -m experiments.paper.numbers --logs ../logs/rehearsal --platform rehearsal --rehearsal --check --no-byte-model
#  -> exit status 0: every tier wrote a record of each kind it writes, and every check passed
```

The smoke runs the GPU tests (the Triton kernels and `torch._int_mm` on this GPU), the native-kernel tests
(`-march=native` on this CPU), the Fiat-Shamir, serving, generation, setup-proof, options and package tests, one V1
query and the host-to-device copy. The rehearsal then runs, in this machine's environment, every script of every
tier, on each path on which the tier's full cells run it (the verifier on each device, a real checkpoint, the proof
without the compact encoding, V1's codec profile and both V1 verifiers, the legacy recipe, a fixed prefix, the sampler
on a real model) and with the A/B cells' implementation options, but on small models with few tokens and queries, so
that no script meets its first failure (a download, a flag, a GPU path, the memory) during the run. Every job asks for
at most 64 GB of host memory:

| Tier | Jobs | The rehearsal's cells |
|---|---:|---|
| `smoke` | 1 | the Fiat-Shamir tree, serving, options and package tests; the native-kernel and int8 self-checks, one V1 query, the host-to-device copy |
| `cpuv` | 3 | GPT-2 (2 blocks, 64 tokens): the CPU verifier, every re-execution baseline, V1's two CPU verifiers |
| `v1` | 1 | V1 against v0 on GPT-2 |
| `serve` | 1 | OPT-125M (2 blocks): single prompts and batches of 4 |
| `bench` | 5 | LeNet-5; GPT-2 in modes C and Kpre, with the GPU verifier, and in Maverick's setting (one thread, no compact encoding) |
| `gpuv` | 4 | GPT-2: the streaming GPU verifier, V1's GPU verifier, re-execution on the GPU; the host-to-device copy |
| `gen` | 1 | GPT-2: 4 generated tokens, greedy and sampled |
| `batch` | 2 | GPT-2: batches of 4 prompts, modes C and Kpre |
| `real` | 4 | the certified OPT-125M: v0, V1, V1's codec profile and generated responses |
| `setup` | 1 | the setup proof of GPT-2 (2 blocks) |
| `improve` | 12 | the A/B cells of `tab:eval-improve` (opening, forward pass, encoder, fold; the fused attention and the GPU verifier before and after on OPT-125M with 2 blocks and on one Llama-2-7B block, the CPU verifier on that block), V1's two verifiers by phase, the GPU profile, the GKR prover at 2^20 leaves |
| `quality` | 6 | OPT-125M: perplexity and zero-shot accuracy (20 documents per task) with the certified and the legacy recipe, the reproducible build and its commitment; Qwen3-0.6B: the Llama recipe on the smoke model and, with a fixed prefix, on the checkpoint, `llama_diag` and `llama_attrib` |
| `tv` | 2 | the sampler's closed-form and Monte Carlo checks on its toy distributions; its total-variation distance on OPT-125M (legacy recipe, 2 positions, 2 workers) |
| `attack` | 5 | the path test's bound; the MNIST scripts on the stored networks; OPT-125M: the single-neuron attack and its backdoor, the in-range edits, the sampler game |
| `long` | 2 | one Llama-2-7B block at 4,608 tokens: the CPU verifier and re-execution |

The attack tier's OPT-125M documents are named by their model (`attack/inrange_llm_opt-125m_any.json`,
`attack/adaptive_game_opt-125m.json`, ...): `numbers.py` counts them for their kinds, and no table or text number
reads them.

## 4. The run

```bash
bash code/experiments/package/run.sh --platform final [--llama-licence] all       # every tier but long
bash code/experiments/package/run.sh --platform final long                        # optional
```

| Tier | Jobs | Expected (h) | Time limits (h) | What it gives the paper |
|---|---:|---:|---:|---|
| `smoke` | 1 | 1 | 2 | the platform gate |
| `cpuv` | 13 | 6-9 | 23 | the CPU verifier of the full Llama-2-7B, Qwen3-4B and OPT-1.3B (Kpre and C, 64 and 2,048 tokens) against every re-execution baseline on the same 8 threads (`tab:options`, the abstract's "at least N times"); V1's optimised and baseline CPU verifiers |
| `v1` | 13 | 4-8 | 26 | V1 against v0: bytes, prover and verifier (`tab:eval-v1`) |
| `serve` | 5 | 8-10 | 25 | the prover against unverified serving on the same GPU, batches of 1 to 64 prompts |
| `bench` | 43 | 8-12 | 48 | the v0 cost table (`tab:eval-llm`), the comparison with published systems (`tab:eval-compare`), the outcome counts, the image models' attack cells |
| `gpuv` | 12 | 4-6 | 19 | the streaming GPU verifier, int8/fp16/bf16 re-execution on the same GPU, the host-to-device bandwidth |
| `gen` | 3 | 2-3 | 7 | generated responses proved as one prefill: Llama-2-7B and Qwen3-4B after a 512-token prompt, greedy and sampled, and GPT-2's prover per generated token (`tab:gen`, `tab:gen-full`, Sec. `sec:gen-prefill`) |
| `batch` | 7 | 2-3 | 8 | several prompts against one opening |
| `real` | 26 | 8-12 | 45 | the certified OPT-1.3B and OPT-6.7B end to end next to random weights (`tab:eval-real`); with `--llama-licence` the real Llama-2-7B |
| `setup` | 2 | 1-2 | 4 | the setup proof of an untrusted commitment (`tab:setup`) |
| `improve` | 15 | 3-5 | 17 | each implementation improvement against the path it replaces (`tab:eval-improve`); V1's verifier by phase; the GKR prover per leaf |
| `quality` | 20 | 10-16 | 69 | `tab:quality` (OPT; OpenLLaMA-7B, TinyLlama, Qwen3; Llama-2 with `--llama-licence`), the recipe's alpha on the validation split, the recipe studies of the extended version (OPT-1.3B and OPT-125M on the test split), `tab:zeroshot`, the reproducible build |
| `tv` | 2 | 1-2 | 3 | the sampler's total-variation distance on the certified OPT-1.3B, and its checks |
| `attack` | 5 | 4-6 | 12 | the attack: reproduction, MNIST, OPT-6.7B, the in-range edits (`tab:inrange`), the sampler games |
| `long` (optional) | 12 | 4-7 | 28 | prompts of 2,048 to 16,384 tokens (`tab:eval-long`) |

The time limits are caps: a job ends when its work is done. One tier at a time also works (`run.sh --platform final
cpuv`, then `v1`, ...); its jobs are chained among themselves. Every mode-C cell takes `--policy` (default `cnn18c`)
and the setup cache (`--setup-cache`, default `logs/<name>/setup_cache`): the first job that commits a model under a
plan stores its Merkle leaves (15-70 MB) and later jobs rebuild the same commitment from them; the `setup` tier always
measures the setup itself.

## 5. While it runs

* `squeue -u $USER` (or the foreground output with `--local`) shows where the chain is. Each job writes
  `logs/<name>/sp-<cell>-<job id>.out`; the records are the lines that start with `{`. The benchmark's records go to
  `code/artifacts/comparison/raw_<name>/`, the attack's to `logs/<name>/attack/`.
* Expected: in `serve`, the largest joint batches of the 7B and 13B models are recorded as `"skipped": "out of
  memory"`; the first job of a real checkpoint spends some minutes downloading it.
* `A CORRECTNESS CHECK FAILED` or `FAILED:` in a log (an honest query rejected, a tamper accepted, a byte count that
  disagrees with the model) is a real problem; the job's log says which check.
* A job that hit its time limit or was pre-empted: run its tier again. Jobs still queued are not submitted twice, and
  finished `bench` and `batch` cells are skipped by `bench.py` itself.
* If the node has less memory than a job asks for, the job never starts: `--mem-max <GB>` and run the tier again.

## 6. What the run leaves

```bash
tar czf final.tgz --exclude=triton --exclude=native --exclude=setup_cache logs/final \
    code/artifacts/comparison/raw_final
```

and the machine's description: `nvidia-smi -q > env-nvidia.txt; lscpu > env-cpu.txt; free -g > env-mem.txt;
.venv/bin/pip freeze > env-pip.txt; git rev-parse HEAD > env-git.txt`.

## 7. From the records to the tables

```bash
cd code
python -m experiments.comparison.aggregate --platform final          # bench.py's records -> tables_final/
python -m experiments.paper.numbers --logs ../logs/final --platform final --check   # coverage and checks
python -m experiments.paper.numbers --logs ../logs/final --platform final           # -> paper_sp2027/generated/
```

The coverage report names every job without records and every table cell still missing; a `CHECK FAILED` or `PARSE`
line, or `--strict` with anything missing, gives exit status 1.

# The run package

`run.sh` is the one scripted run whose records give every number of the paper. It runs on one GPU node: the GPU
proves and is the GPU verifier, and 8 threads of the node's CPU are the CPU verifier and the re-execution baselines.
The paper's numbers come from an NVIDIA L40S (48 GB) with an AMD EPYC 9334, and every job runs on that platform:
each job starts by writing its node, GPU, driver and CPU to its log and checking the GPU and CPU models
(`--require-gpu`, default `L40S`; `--require-cpu`, default `EPYC 9334`; `any` checks nothing), and on another model
exits with status 86 before doing any work. Every timed job has its node to itself and no two timed jobs share a node;
when the smoke tier is selected, nothing after it starts unless it succeeded.
The jobs write JSON lines; `experiments/paper/numbers.py` turns them into the paper's tables and text numbers.

## 1. What the machine needs

| Need | Why | How to check |
|---|---|---|
| An NVIDIA GPU with int8 tensor cores (sm_75 or later; the paper's: L40S, sm_89) and a CUDA 12.1 driver | the prover, the GPU verifier, serving (on sm_90 and later, the attention takes the unfused exact path: the fused Triton kernel does not compile there with Triton 3.1) | `nvidia-smi`; `.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"` |
| 8 free CPU cores of one NUMA node and about 256 GB of memory | the CPU verifier and every re-execution baseline run on 8 threads; each CPU-timing job pins itself to 8 cores of one NUMA node (the GPU's, where the kernel reports it), one hardware thread per core, and its records name the CPUs it ran on (`--no-pin`: not pinned); the 7B cells at 2,048 tokens ask for 256 GB | `lscpu`, `free -g`; with less memory, `--mem-max <GB>` (under SLURM, `run.sh` compares the largest request with the partition's largest node before it submits anything) |
| A Python 3.12 environment as `.venv` at the checkout's root (or `--python <path>`) | torch 2.5.1+cu121, `code/requirements.txt` and `code/requirements-llm.txt` (with `sentencepiece` and `protobuf`, which OpenLLaMA-7B's tokenizer needs; `run.sh` checks them before it submits the quality tier without `--llama-licence`) | `.venv/bin/python -c "import torch, transformers, triton, sentencepiece, google.protobuf; print(torch.__version__)"` |
| A C++ compiler in `CXX` | the native CPU kernels (`-march=native`) of the CPU verifier and of the exact int8 re-execution; every CPU-timing cell passes `--require-native`, so a compiler missing on the node stops it instead of timing the torch path, and `run.sh` submits nothing while `CXX` names no compiler (`--no-native-ok` overrides both) | `$CXX --version` (g++ 9 or later) |
| The Python headers for Triton's launcher | Triton compiles a small launcher with `Python.h` on first use, on the GPU node | on the GPU node, `.venv/bin/python -c "import os, sysconfig; print(os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))"`; if `False`, copy the headers into a folder and `export PVI_PY_INCLUDE=<folder>` |
| About 10 GB of free temporary space on the GPU node, and about 21 GB during the `cpuv` tier's llama.cpp job | the compilers of Triton's launcher and of the native kernels write their temporary files to `$TMPDIR` (default `/tmp`); the `cpuv` tier's client process receives each Llama-2-7B proof through a file there (6.4 GB at 2,048 tokens; one at a time); the llama.cpp job (`experiments/reexec/llamacpp.py`) writes the bf16 and Q8_0 files of Llama-2-7B's shapes there (13.5 and 7.2 GB, deleted at its end; 1.4 GB for the rehearsal's one block) and stops before writing them if they do not fit | on the GPU node, `df -h ${TMPDIR:-/tmp}`; if it is full, `export TMPDIR=<folder on a disk the nodes see>` (sbatch passes it to the jobs) |
| CMake 3.14 or later and git (the `cpuv` tier) | the `cpuv` tier's llama.cpp job clones llama.cpp's release v0.6.0 and builds `llama-bench` and `llama-quantize` for the node's CPU (`-march=native`, with `CXX`), once per platform, into `logs/<name>/llamacpp/` (about 1 GB); `run.sh` submits nothing while that tier is selected and CMake (`CMAKE`, else `cmake` on the `PATH`) or git is missing | `${CMAKE:-cmake} --version`, `git --version`; without CMake, one way is `pip install cmake==3.31.6` into a separate environment (`python3 -m venv ~/cmake && ~/cmake/bin/pip install cmake==3.31.6`, then `export CMAKE=~/cmake/bin/cmake`), which leaves the run's environment unchanged. The job clones from GitHub; if the compute nodes have no network access, clone the release beforehand: `git clone --depth 1 --branch v0.6.0 https://github.com/ggml-org/llama.cpp.git logs/<name>/llamacpp/src` |
| `HF_HOME` on a disk the GPU nodes see, with about 60 GB free, 100 GB with `--llama-licence` (a node-local folder holds a copy per node) | the `real`, `quality`, `tv` and `attack` tiers download OPT-125M to OPT-6.7B, OpenLLaMA-7B (without `--llama-licence`), TinyLlama, Qwen3 and WikiText-2 (47 GB of checkpoints); with `--llama-licence` also Llama-2-7B and Llama-2-13B (40 GB more; each repository has its own gate on Hugging Face), for which the jobs need your Hugging Face token: `export HF_TOKEN=<token>` (sbatch passes it to the jobs; `run.sh` never writes it into one), or a login whose `$HF_HOME/token` the compute node reads | `df -h $HF_HOME` |
| The trained image models and their datasets (not in the repository) | the `bench` tier's image cells (MLP, LeNet-5, VGG-11/16, ResNet-18 on CIFAR-10 and on 224-pixel dogs-vs-cats, and the dogs-vs-squirrels ResNet-18 that is Anchuri et al.'s substitute model) | `code/artifacts/models/cnn/{lenet5,vgg11,vgg16,resnet18_cifar,resnet18_224,resnet18_224_squirrel}.pt` (on the GPU node, from `code/`: `PYTHONPATH=src:. ../.venv/bin/python -m experiments.costs.train_cnn --model <m> --device cuda`, seed 0); CIFAR-10 in the `cifar-10-batches-py` layout under `code/data/cifar10` (or `$PVI_CIFAR_ROOT`); the ImageNet subsets' caches `dogs_cats.pt` and `dogs_squirrels.pt` under `code/artifacts/cache` (or `$PVI_CACHE`), or ImageNet under `code/data/imagenet` (or `$PVI_IMAGENET_ROOT`) with `train/` and `val/` holding class `i` of the sorted synsets in folder `i + 1` (`1` to `1000`; the dogs are `152` to `269`, the cats `282` to `286`, the fox squirrel `336`), from which the first run writes the caches. `run.sh all` and `run.sh bench` stop before submitting anything if one is missing |
| SLURM (`sbatch`, `squeue`), or none | the jobs | with SLURM: `export SBATCH_PARTITION=... SBATCH_GRES=gpu:<type>:1 [SBATCH_ACCOUNT=...]` (or `--gres gpu:<type>:1`; `run.sh` passes it to every job and allocation and submits nothing without it; `--exclude-nodes <list>` keeps the jobs off the partition's nodes of another GPU or CPU model); without: `--local` (every job runs in the foreground, one after another: start it in `tmux` or under `nohup`) |
| The platform's GPU and CPU models on every node a job runs on | one platform for every number | `--require-gpu`, `--require-cpu` (defaults `L40S`, `EPYC 9334`; on another platform pass its names, or `any`); `run.sh --dry-run` prints them in its `# node:` line |
| Exclusive use of the node | the timings | every job asks for `--exclusive` (check that the partition allows exclusive jobs); the smoke and the rehearsal, whose timings are not used, pass `--no-exclusive`; on a shared partition, see Sec. 4 (`--one-job`) |
| For the optional `zkllm` tier (with `--llama-licence`): git, make, gcc/g++ (12 or older for CUDA 12.1) and a CUDA toolkit (nvcc; CUDA 12.1 as zkLLM recommends), a Python 3.8 to 3.12, about 10 GB for zkLLM's checkout and environment and 58 GB of scratch space on the GPU node | the tier clones zkLLM's public demo (commit `993311e`) and builds it for the node's GPU in an environment of its own (`logs/<name>/zkllm/home`, or `--zkllm-home`; the run's `.venv` is not changed); zkLLM writes its fixed-point weights and commitments (27 GB for Llama-2-7B, 52 GB for Llama-2-13B) to the scratch folder (`--zkllm-scratch`, default the node's `$TMPDIR`, else `/tmp`, else the zkLLM home) and the job deletes them at its end; the checkpoints are the Llama-2 ones of `HF_HOME`. `run.sh` submits nothing while one of these is missing on the submitting host and says which (`experiments/comparison/zkllm/README.md`) | `git --version; make --version; g++ --version; nvcc --version` (or `--cuda-home <CUDA folder>`, e.g. after `module load cuda/12.1`, passed to the jobs); on the GPU node, `df -h ${TMPDIR:-/tmp}` |
| No Hugging Face cache variable that overrides `HF_HOME` with a folder the jobs cannot write | `HF_HUB_CACHE`, `TRANSFORMERS_CACHE` and `HF_DATASETS_CACHE` take precedence over `HF_HOME` | `env \| grep -E 'HF_HUB_CACHE\|TRANSFORMERS_CACHE\|HF_DATASETS_CACHE'`; `run.sh` refuses an unwritable one (`unset` it) |

Every language model of the cost tiers has random weights (the costs depend only on the shapes); real checkpoints
are downloaded by the `real`, `quality`, `tv` and `attack` tiers only. The Llama-2 cells (`real`, `quality`) need the
Llama-2 licence on your own Hugging Face account and your own token, and run only with `--llama-licence`.

## 2. Setup

```bash
git clone <repository> pvi && cd pvi && git checkout <tag>
python3.12 -m venv .venv && .venv/bin/pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && .venv/bin/pip install -r code/requirements.txt -r code/requirements-llm.txt
export SBATCH_PARTITION=<GPU partition> SBATCH_GRES=gpu:<type>:1      # or pass --local to run.sh
export HF_HOME=<folder the nodes see, 100 GB free>  # 60 GB without --llama-licence
export CXX=<path to g++>
export CMAKE=<path to cmake>                         # only if cmake is not on the PATH (the cpuv tier)
export PVI_PY_INCLUDE=<headers folder>               # only if the check above printed False
export HF_TOKEN=<your Hugging Face token>            # only with --llama-licence
module load cuda/12.1                                # only for the zkllm tier, if nvcc is not on the PATH
```

`run.sh` passes `HF_HOME`, `PVI_CIFAR_ROOT`, `PVI_IMAGENET_ROOT`, `PVI_CACHE`, `PVI_NATIVE_DIR` and `PVI_PY_INCLUDE`
to the jobs as absolute paths (a relative one is taken from the repository root), and `CXX` and `CMAKE` as they are.
Before it submits anything it checks the compiler, CMake and git (for the cpuv tier), the interpreter, `HF_HOME` (for
the tiers that download), the bench tier's models and datasets and the partition's memory, and stops with what is
missing.

`bash code/experiments/package/run.sh --platform final --dry-run all` prints every job it would submit, with each
tier's job count, the sum of its time limits and its expected hours, and a last line `# <tiers>: N jobs, ...`; it
submits nothing and needs none of the settings above (it only warns about those this host lacks).

## 3. The smoke and the rehearsal

```bash
bash code/experiments/package/run.sh --platform smoke --no-exclusive smoke
#  -> "SMOKE OK" at the end of logs/smoke/sp-smoke-*.out (up to an hour); wait for it before the rehearsal
bash code/experiments/package/run.sh --platform rehearsal --rehearsal --no-exclusive --optional [--llama-licence] all
#  -> 54 small jobs (55 with --llama-licence): every tier's smallest cells (GPT-2 / OPT-125M, 2 blocks), about three
#     hours; wait until `squeue -u $USER` lists none of them, then
cd code
PYTHONPATH=src:. ../.venv/bin/python -m experiments.comparison.aggregate --platform rehearsal
PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/rehearsal --platform rehearsal \
    --rehearsal --check --no-byte-model | tee ../logs/rehearsal/numbers_check.txt
#  -> exit status 0: every tier wrote a record of each kind it writes, and every check passed
```

The smoke runs the GPU tests (the Triton kernels and `torch._int_mm` on this GPU), the native-kernel tests
(`-march=native` on this CPU), the Fiat-Shamir, serving, generation, setup-proof, options and package tests, the
native-kernel and int8 self-checks (it fails unless both pass), one V1 query and the host-to-device copy; pytest's
whole output goes to `logs/<name>/smoke_pytest.txt`. The rehearsal then runs, in this machine's environment, every
script of every tier, on each path on which the tier's full cells run it (the verifier on each device, a real
checkpoint, the proof without the compact encoding, V1's codec profile and both V1 verifiers, the legacy recipe, a
fixed prefix, the sampler on a real model) and with the A/B cells' implementation options, but on small models with
few tokens and queries, so that no script meets its first failure (a download, a flag, a GPU path, the memory)
during the run. Every job asks for at most 64 GB of host memory:

| Tier | Jobs | The rehearsal's cells |
|---|---:|---|
| `smoke` | 1 | the Fiat-Shamir tree, serving, options and package tests; the native-kernel and int8 self-checks, one V1 query, the host-to-device copy |
| `cpuv` | 4 | GPT-2 (2 blocks, 64 tokens): the CPU verifier (its proofs checked again in the client's own process), every re-execution baseline, V1's two CPU verifiers; llama.cpp's build and one block of Llama-2-7B's shapes in Q8_0 and bf16 (64 tokens) |
| `v1` | 1 | V1 against v0 on GPT-2 |
| `serve` | 1 | OPT-125M (2 blocks): single prompts and batches of 4 |
| `bench` | 5 | LeNet-5; GPT-2 in modes C and Kpre, with the GPU verifier, and in Maverick's setting (one thread, no compact encoding) |
| `gpuv` | 4 | GPT-2: the streaming GPU verifier, V1's GPU verifier, re-execution on the GPU; the host-to-device copy |
| `gen` | 1 | GPT-2: 4 generated tokens, greedy and sampled |
| `batch` | 2 | GPT-2: batches of 4 prompts, modes C and Kpre |
| `real` | 4 | the certified OPT-125M: v0, V1, V1's codec profile and generated responses |
| `setup` | 1 | the setup proof of GPT-2 (2 blocks) |
| `improve` | 13 | the A/B cells of `tab:eval-improve` (opening, forward pass, encoder, fold; the fused attention and the GPU verifier before and after on OPT-125M with 2 blocks and on one Llama-2-7B block, the CPU verifier on that block), V1's two verifiers by phase, the GPU profile, the GKR prover at 2^20 leaves, 8- against 11-bit attention probabilities on GPT-2 (2 blocks) |
| `quality` | 8 | OPT-125M: perplexity (the legacy recipe on the validation split) and zero-shot accuracy (20 documents per task) with the certified and the legacy recipe, the attribution of its perplexity (2 windows, without the feature sub-rows), the reproducible build and its commitment; Qwen3-0.6B: the Llama recipe on the smoke model (also its per-token-exponent cells alone) and, with a fixed prefix and a kept reference, on the checkpoint, `llama_diag` and `llama_attrib`; the Llama-2 grid path (two alphas, two product balances, the validation split, an fp16 reference) on a smoke model with Llama-2-7B's vocabulary, which reads the gated repository from a compute node (with `--llama-licence`), or with OpenLLaMA-7B's, whose tokenizer needs `sentencepiece` and `protobuf` (without it) |
| `tv` | 2 | the sampler's closed-form and Monte Carlo checks on its toy distributions; its total-variation distance on OPT-125M (legacy recipe, 2 positions, 2 workers) |
| `attack` | 5 | the path test's bound; the MNIST scripts on the stored networks; OPT-125M: the single-neuron attack and its backdoor, the in-range edits, the sampler game |
| `long` | 2 | one Llama-2-7B block at 4,608 tokens: the CPU verifier and re-execution |
| `zkllm` | 0 (1) | with `--llama-licence`: zkLLM's build (Llama-2-7B's download only) and one repetition of Llama-2-7B's layer 0 at 2,048 tokens, every step of the full jobs (`numbers.py --rehearsal` does not require its record, since the tier needs the licence; a record that is not `ok` is a `CHECK FAILED` line) |

The attack tier's OPT-125M documents are named by their model (`attack/inrange_llm_opt-125m_any.json`,
`attack/adaptive_game_opt-125m.json`, ...): `numbers.py` counts them for their kinds, and no table or text number
reads them.

### The gate: the full sizes, once before the run

The rehearsal's cells are small, so a path that fails only at the run's largest sizes would first fail during the
run. The `gate` tier, which `all` does not include and `--rehearsal` refuses, runs those paths once on the final
platform, each on its least costly cell of full size, with the checks of the run's cells (every honest query
accepted, every tamper rejected, V1's bytes exactly as the byte model gives them); its records
(`logs/final/sp-gate-*.out`) fill no table cell or text number. Run it after the smoke and before the run:

```bash
bash code/experiments/package/run.sh --platform final --one-job smoke gate    # or without --one-job: a chain
#  -> 8 jobs (the smoke and the gate's 7), about two hours on an L40S node (job limits 10 h); then, from code/,
PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final --check \
    --no-byte-model --allow-missing
#  -> no CHECK FAILED line (the run's tables and macros are still missing: --allow-missing lists them, exit status 0)
```

| Job | Cell | The size it checks |
|---|---|---|
| `gate-v1-llama7-2k-2` | V1 on two Llama-2-7B blocks at 2,048 tokens, modes Kpre and C, both CPU verifiers, 2 timed queries, 2 tampers of each kind | a GKR instance of 27 variables (2^27 leaves, the default `--lmax`; the smoke and the rehearsal reach 20) on the Triton prover and both CPU verifiers |
| `gate-g-v1-llama7-2k-2` | the same cell in Kpre with the GPU verifier, 1 tamper of each kind | V1's GPU verifier at 2^27 leaves |
| `gate-v1-opt13-2k` | V1 on OPT-1.3B at 2,048 tokens, both modes, 2 timed queries, 1 tamper of each kind | seven instances of 27 variables, three of them of exactly 2^27 leaves (no padding row) |
| `gate-r-opt125-v1-2k` | V1 on the certified OPT-125M (real weights) at 2,048 tokens, both modes, 2 timed queries, 1 tamper of each kind | the real-weight path at 2,048 tokens, an instance of 27 variables |
| `gate-v1-llama7-2k` | V1 on the full Llama-2-7B at 2,048 tokens, Kpre, one query and one tamper of each kind | the run's largest V1 query (21 instances) and its host and GPU memory |
| `gate-l-llama7-16k` | the CPU verifier on two Llama-2-7B blocks at 16,384 tokens | the `long` tier's largest prompt |
| `gate-l-rx-llama7-16k` | exact re-execution (`int`, `int_mm`, `int_fp32gemm`) of the same cell, the rotary positions extended to 16,384 | the same, for the baselines that `tab:eval-long` reads |

Every V1 cell of the run at 2,048 tokens, and the 64-token V1 cells of Qwen3-4B, Llama-2-7B, OPT-6.7B and
Llama-2-13B, reach GKR instances of 27 variables (`experiments/paper/tables.py`, `v1_shape_model`; random weights).
The setup proof's openings span every member of a tree, so only the `setup` tier's full Llama-2-7B reaches its size;
the gate leaves it to the run.

## 4. The run

On a machine of your own, or a partition whose nodes are free, one invocation runs everything as a chain of
whole-node jobs:

```bash
bash code/experiments/package/run.sh --platform final --optional --llama-licence all
```

On a shared partition a job that asks for a whole node waits until every job on some node has ended, and the chain
waits that long for each of its 186 jobs. There, the paper's run submits its tiers as nine jobs (`--one-job`), each
of which runs its tiers' jobs one after another in its allocation: the timed tiers in seven jobs that have a node to
themselves, and the tiers that time nothing (`quality`, `tv`, `attack`) in two jobs on one GPU of a shared node:

```bash
R="bash code/experiments/package/run.sh --platform final --llama-licence --gres gpu:l40s:1 \
   --exclude-nodes <the partition's nodes of another GPU or CPU model>"
$R --one-job cpuv long                     # expected 11-17 h
$R --one-job bench                         # 8-12 h
$R --one-job real                          # 8-12 h
$R --one-job serve gen batch               # 12-16 h
$R --one-job v1 setup improve              # 8-15 h
$R --one-job gpuv                          # 4-6 h
$R --one-job zkllm                         # 3-5 h: zkLLM's demo on the same GPU model
$R --one-job --no-exclusive quality        # 28-36 h, not timed (past 24 h: submit it again, Sec. 5)
$R --one-job --no-exclusive tv attack      # 5-8 h, not timed
```

Every allocation, the two without `--exclusive` too, asks for the same GPU (`--gres`) and checks its node before its
first job (the default `--require-gpu L40S --require-cpu "EPYC 9334"`): on a node of another model it runs nothing,
exits with status 86 and prints the command to submit again with that node excluded. It then runs its tiers' jobs one
after another; a job that fails does not stop the jobs after it (no job reads another's output: the setup cache and
the reference caches are rebuilt where missing). It writes the same job logs as the chain
(`logs/final/sp-<cell>-<id>.out`), and its own log (`logs/final/node-<tiers>-<job id>.log`) ends with a summary: the
jobs that ended successfully, those that failed with their logs and exit statuses (124: the job's time limit), and the
command that runs the failed and missing ones again, which is the same command. Its time limit is 24 hours
(`--max-hours`), and SLURM queues a pre-empted one again by itself (`--requeue`); a job that ends successfully leaves
a marker in `logs/final/done/`, and the allocation, when it starts again or when the same command is submitted again,
runs only the jobs without a marker. A later job's records of a cell replace an earlier one's. The smoke is optional
here: `smoke` in front of a submission's tiers runs it first on that node and runs the rest only if it succeeds.

| Tier | Jobs | Expected (h) | Time limits (h) | What it gives the paper |
|---|---:|---:|---:|---|
| `smoke` | 1 | 1 | 2 | the platform gate |
| `cpuv` | 14 | 7-10 | 26 | the CPU verifier of the full Llama-2-7B, Qwen3-4B and OPT-1.3B (Kpre and C, 64 and 2,048 tokens) against every re-execution baseline on the same 8 threads (the exact integer pass with the native int8 GEMM, with `torch._int_mm` and with fp32 GEMMs; fp32, bf16 and dynamic int8; llama.cpp's prefill of Llama-2-7B's shapes in Q8_0 and bf16, inexact and another model), the allocator setting of the verifier as a like-for-like row (`tab:options`, the abstract's "at least N times", which stays against exact re-execution); the Llama-2-7B client's own memory, its timed proofs checked again in a process that holds only its key and embedding tables or the roots (`tab:memory`); V1's optimised and baseline CPU verifiers |
| `v1` | 13 | 4-8 | 26 | V1 against v0: bytes, prover and verifier (`tab:eval-v1`) |
| `serve` | 5 | 8-10 | 25 | the prover against unverified serving on the same GPU, batches of 1 to 64 prompts |
| `bench` | 43 | 8-12 | 48 | the v0 cost table (`tab:eval-llm`), the comparison with published systems (`tab:eval-compare`), the outcome counts, the image models' attack cells |
| `gpuv` | 12 | 4-6 | 19 | the streaming GPU verifier, int8/fp16/bf16 re-execution on the same GPU, the host-to-device bandwidth |
| `gen` | 3 | 2-3 | 7 | generated responses proved as one prefill: Llama-2-7B and Qwen3-4B after a 512-token prompt, greedy and sampled, and GPT-2's prover per generated token (`tab:gen`, `tab:gen-full`, Sec. `sec:gen-prefill`) |
| `batch` | 7 | 2-3 | 8 | several prompts against one opening |
| `real` | 26 (17) | 8-12 | 45 (23) | the certified OPT-1.3B and OPT-6.7B end to end next to random weights (`tab:eval-real`); with `--llama-licence` the real Llama-2-7B |
| `setup` | 2 | 1-2 | 4 | the setup proof of an untrusted commitment (`tab:setup`) |
| `improve` | 16 | 3-5 | 18 | each implementation improvement against the path it replaces (`tab:eval-improve`); V1's verifier by phase; the GKR prover per leaf; 8- against 11-bit attention probabilities on two Llama-2-7B blocks at 2,048 tokens (`quality.tex`: the proof's size and the prover's and CPU verifier's times) |
| `quality` | 22 (20) | 28-36 (16-24) | 102 (70) | `tab:quality` (OPT), the Llama-family ratios of the text (TinyLlama, Qwen3; Llama-2-7B and Llama-2-13B with `--llama-licence`), the recipe's alpha on the validation split (OPT), Llama-2's committed alpha and product balance among their neighbours on the test and the validation split (four jobs of 10-12 h; without `--llama-licence` OpenLLaMA-7B's two jobs, which the paper does not report, in their place), the recipe studies of the extended version (OPT-1.3B and OPT-125M on the test split), the attribution of OPT-125M's perplexity (`tab:quality-attr`, on the CPU), `tab:zeroshot`, the reproducible build (1 and 8 threads, the commitment encoded on the GPU and the CPU) |
| `tv` | 2 | 1-2 | 3 | the sampler's total-variation distance on the certified OPT-1.3B, and its checks |
| `attack` | 5 | 4-6 | 12 | the attack: reproduction, MNIST, OPT-6.7B, the in-range edits (`tab:inrange`), the sampler games |
| `long` (optional) | 12 | 4-7 | 28 | prompts of 2,048 to 16,384 tokens (`tab:eval-long`) |
| `zkllm` (optional) | 3 (0) | 3-5 | 8 (0) | zkLLM's public demo (commit `993311e`, unchanged) on the run's GPU: its build, then Llama-2-7B and Llama-2-13B at 2,048 tokens, 3 repetitions of layers 0 and 1, each proof binary timed (`logs/<name>/zkllm/`): zkLLM's prover on the same GPU model (`\ZkllmSameGpu...`, Sec. `sec:eval-compare`), in place of the stored runs (`artifacts/results/zkllm_l40s/`), and the text's "in the same run" (`experiments/comparison/zkllm/README.md`) |

`--optional` adds the `long` tier, which the paper's Discussion and `tab:eval-long` read, and the `zkllm` tier (with
`--llama-licence` only). Submit them in the same invocation as the other tiers, so that they are chained after them
and gated by the smoke; a second invocation starts a new chain. Without a Llama-2 licence, leave out
`--llama-licence`: the Llama-2 cells then stay empty (the quality tier runs OpenLLaMA-7B's two jobs instead, which the
paper does not report). The counts and limits are those with `--llama-licence`; without it (in parentheses) the
`real` tier has 17 jobs and 23 h of limits and the `quality` tier 20 jobs and 70 h. `all` is then 160 jobs and 291 h
of limits (171 and 345 h with `--llama-licence`; the `long` tier adds 12 jobs and 28 h, and the `zkllm` tier, with
`--llama-licence`, 3 jobs and 8 h: `run.sh --platform final --optional --llama-licence --dry-run all` lists 186 jobs
and 381 h).

The time limits are caps: a job ends when its work is done. One tier at a time also works (`run.sh --platform final
cpuv`, then `v1`, ...); its jobs are chained among themselves. Every mode-C cell takes `--policy` (default `cnn18c`)
and the setup cache (`--setup-cache`, default `logs/<name>/setup_cache`): the first job that commits a model under a
plan stores its Merkle leaves (15-70 MB) and later jobs rebuild the same commitment from them; the `setup` tier always
measures the setup itself.

## 5. While it runs

* `squeue -u $USER` (or the foreground output with `--local`) shows where the chain is. Each job writes
  `logs/<name>/sp-<cell>-<job id>.out`; the records are the lines that start with `{`. The benchmark's records go to
  `code/artifacts/comparison/raw_<name>/`, the attack's to `logs/<name>/attack/`. The second line of a CPU-timing
  job's log names the CPUs it was pinned to (or why it was not).
* Expected: in `serve`, the largest joint batches of the 7B and 13B models are recorded as `"skipped": "out of
  memory"`; the first job of a real checkpoint spends some minutes downloading it. In the `zkllm` tier, each first
  skip connection's log says "Input or output file does not exist" (zkLLM's demo writes no attention output; the
  time per layer counts the second one twice); `zk-build` ends with `ZKLLM_BUILD_OK`, and a prove job that stops
  names what it lacks (the scratch space: `--zkllm-scratch <folder with 58 GB>`; the CUDA toolkit on the node:
  `--cuda-home`).
* `A CORRECTNESS CHECK FAILED` or `FAILED:` in a log (an honest query rejected, a tamper accepted, a byte count that
  disagrees with the model) is a real problem; the job's log says which check.
* A `--one-job` submission that hit its time limit or had failed jobs: submit the same command again (the last
  lines of its log print it); it skips the jobs that ended successfully (`logs/<name>/done/`). Its log names each job
  it ran or skipped, and its summary the exit status and log of each that failed.
* `wrong node: GPU <x> / CPU <y>, expected ...` (exit status 86) in a log: the job or allocation started on a node of
  another model and did nothing; add that node to `--exclude-nodes` and submit the same command again.
* A job of the chain that hit its time limit or was pre-empted: run its tier again. A job of the same name and
  platform that is still queued or running is not submitted twice (each job carries its platform as its SLURM
  comment), and the jobs submitted after it wait for it, behind the smoke's gate; finished `bench` and `batch` cells
  are skipped by `bench.py` itself.
* A failed smoke leaves the rest of its chain pending for ever, and since every job has `--dependency=singleton`, a
  pending job also holds back the job of the same name of any later run. Before running again (and before the final
  run, for the smoke's and the rehearsal's jobs), cancel the platform's pending jobs:

  ```bash
  squeue -h -u $USER -t PENDING -o '%i %j %k' | awk '$2 ~ /^sp-/ && $3 == "<name>" {print $1}' | xargs -r scancel
  ```
* If no node of the partition has the memory the largest job asks for, `run.sh` submits nothing and says which
  `--mem-max <GB>` to pass; if `sbatch` still refuses a job (another limit of the partition), `run.sh` stops there,
  and running it again with the setting corrected submits only the jobs that are not queued.

## 6. What the run leaves

The machine's description, written on the GPU node into the run's folder:

```bash
srun -p "$SBATCH_PARTITION" --gres="$SBATCH_GRES" bash -c 'nvidia-smi -q > logs/final/env-nvidia.txt;
    lscpu > logs/final/env-cpu.txt; free -g > logs/final/env-mem.txt'   # without SLURM: the three commands
.venv/bin/pip freeze > logs/final/env-pip.txt; git rev-parse HEAD > logs/final/env-git.txt
```

then the records, without the caches, as one archive and as the artifact's stored copy (`code/artifacts/README.md`,
"The final run's records"):

```bash
tar czf final.tgz --exclude=triton --exclude=native --exclude=setup_cache --exclude=llamacpp --exclude='ref_*.pt' \
    --exclude=zkllm/home logs/final code/artifacts/comparison/raw_final
rsync -a --exclude=triton --exclude=native --exclude=setup_cache --exclude=llamacpp --exclude='ref_*.pt' \
    --exclude=zkllm/home logs/final/ code/artifacts/final/
```

## 7. From the records to the tables

```bash
cd code
PYTHONPATH=src:. ../.venv/bin/python -m experiments.comparison.aggregate --platform final   # -> tables_final/
(cd .. && bash code/experiments/package/run.sh --platform final --dry-run --optional --llama-licence all \
    > logs/final/dryrun.txt)                                       # every job of the run, for --expect
PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final --check \
    --expect ../logs/final/dryrun.txt | tee ../logs/final/numbers_check.txt
PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final
#  (the first numbers command: coverage and checks; the second writes paper_sp2027/generated/)
```

The coverage report names every job without records (with `--expect`); its last lines list every table cell and
every macro that the run does not provide (`MISSING: ...`, one per line; `COMPLETE: ...` when there is none). A
`CHECK FAILED` or `PARSE` line, or any missing cell or macro, gives exit status 1 (`--strict`: also a job without
results). Its `machines:` line lists the nodes and drivers the jobs ran on (several are allowed), and jobs whose logs
or records name another GPU or CPU model than the rest are a `CHECK FAILED platform` line that names them.
`numbers_check.txt` keeps the first command's report: its end shows at a glance what is missing.

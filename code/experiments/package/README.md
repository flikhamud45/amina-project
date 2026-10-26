# The run package

`run.sh` is the one scripted run whose records give every number of the paper. It runs on one GPU node: the GPU
proves and is the GPU verifier, and 8 threads of the node's CPU are the CPU verifier and the re-execution baselines.
The paper's numbers come from an NVIDIA L40S (48 GB) with an AMD EPYC 9334. Every timed job has its node to itself
and no two timed jobs share a node, and nothing after the smoke tier starts unless it succeeded.
The jobs write JSON lines; `experiments/paper/numbers.py` turns them into the paper's tables and text numbers.

## 1. What the machine needs

| Need | Why | How to check |
|---|---|---|
| An NVIDIA GPU with int8 tensor cores (sm_75 or later; the paper's: L40S, sm_89) and a CUDA 12.1 driver | the prover, the GPU verifier, serving (on sm_90 and later, the attention takes the unfused exact path: the fused Triton kernel does not compile there with Triton 3.1) | `nvidia-smi`; `.venv/bin/python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"` |
| 8 free CPU cores of one NUMA node and about 256 GB of memory | the CPU verifier and every re-execution baseline run on 8 threads; each CPU-timing job pins itself to 8 cores of one NUMA node (the GPU's, where the kernel reports it), one hardware thread per core, and its records name the CPUs it ran on (`--no-pin`: not pinned); the 7B cells at 2,048 tokens ask for 256 GB | `lscpu`, `free -g`; with less memory, `--mem-max <GB>` (under SLURM, `run.sh` compares the largest request with the partition's largest node before it submits anything) |
| A Python 3.12 environment as `.venv` at the checkout's root (or `--python <path>`) | torch 2.5.1+cu121, `code/requirements.txt` and `code/requirements-llm.txt` | `.venv/bin/python -c "import torch, transformers, triton; print(torch.__version__)"` |
| A C++ compiler in `CXX` | the native CPU kernels (`-march=native`) of the CPU verifier and of the exact int8 re-execution; every CPU-timing cell passes `--require-native`, so a compiler missing on the node stops it instead of timing the torch path, and `run.sh` submits nothing while `CXX` names no compiler (`--no-native-ok` overrides both) | `$CXX --version` (g++ 9 or later) |
| The Python headers for Triton's launcher | Triton compiles a small launcher with `Python.h` on first use, on the GPU node | on the GPU node, `.venv/bin/python -c "import os, sysconfig; print(os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))"`; if `False`, copy the headers into a folder and `export PVI_PY_INCLUDE=<folder>` |
| A few GB of free temporary space on the GPU node | the compilers of Triton's launcher and of the native kernels write their temporary files to `$TMPDIR` (default `/tmp`) | on the GPU node, `df -h ${TMPDIR:-/tmp}`; if it is full, `export TMPDIR=<folder on a disk the nodes see>` (sbatch passes it to the jobs) |
| `HF_HOME` on a disk the GPU nodes see, with about 60 GB free, 100 GB with `--llama-licence` (a node-local folder holds a copy per node) | the `real`, `quality`, `tv` and `attack` tiers download OPT-125M to OPT-6.7B, OpenLLaMA-7B, TinyLlama, Qwen3 and WikiText-2 (47 GB of checkpoints); with `--llama-licence` also Llama-2-7B and Llama-2-13B (40 GB more; each repository has its own gate on Hugging Face), for which the jobs need your Hugging Face token: `export HF_TOKEN=<token>` (sbatch passes it to the jobs; `run.sh` never writes it into one), or a login whose `$HF_HOME/token` the compute node reads | `df -h $HF_HOME` |
| The trained image models and their datasets (not in the repository) | the `bench` tier's image cells (MLP, LeNet-5, VGG-11/16, ResNet-18 on CIFAR-10 and on 224-pixel dogs-vs-cats, and the dogs-vs-squirrels ResNet-18 that is Anchuri et al.'s substitute model) | `code/artifacts/models/cnn/{lenet5,vgg11,vgg16,resnet18_cifar,resnet18_224,resnet18_224_squirrel}.pt` (on the GPU node, from `code/`: `PYTHONPATH=src:. ../.venv/bin/python -m experiments.costs.train_cnn --model <m> --device cuda`, seed 0); CIFAR-10 in the `cifar-10-batches-py` layout under `code/data/cifar10` (or `$PVI_CIFAR_ROOT`); the ImageNet subsets' caches `dogs_cats.pt` and `dogs_squirrels.pt` under `code/artifacts/cache` (or `$PVI_CACHE`), or ImageNet under `code/data/imagenet` (or `$PVI_IMAGENET_ROOT`) with `train/` and `val/` holding class `i` of the sorted synsets in folder `i + 1` (`1` to `1000`; the dogs are `152` to `269`, the cats `282` to `286`, the fox squirrel `336`), from which the first run writes the caches. `run.sh all` and `run.sh bench` stop before submitting anything if one is missing |
| SLURM (`sbatch`, `squeue`), or none | the jobs | with SLURM: `export SBATCH_PARTITION=... SBATCH_GRES=gpu:<type>:1 [SBATCH_ACCOUNT=...]`; without: `--local` (every job runs in the foreground, one after another: start it in `tmux` or under `nohup`) |
| Exclusive use of the node | the timings | every job asks for `--exclusive` (check that the partition allows exclusive jobs); the smoke and the rehearsal, whose timings are not used, pass `--no-exclusive`; on a shared partition, see Sec. 4 (`--one-job`) |
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
export PVI_PY_INCLUDE=<headers folder>               # only if the check above printed False
export HF_TOKEN=<your Hugging Face token>            # only with --llama-licence
```

`run.sh` passes `HF_HOME`, `PVI_CIFAR_ROOT`, `PVI_IMAGENET_ROOT`, `PVI_CACHE`, `PVI_NATIVE_DIR` and `PVI_PY_INCLUDE`
to the jobs as absolute paths (a relative one is taken from the repository root). Before it submits anything it
checks the compiler, the interpreter, `HF_HOME` (for the tiers that download), the bench tier's models and datasets
and the partition's memory, and stops with what is missing.

`bash code/experiments/package/run.sh --platform final --dry-run all` prints every job it would submit, with each
tier's job count, the sum of its time limits and its expected hours, and a last line `# <tiers>: N jobs, ...`; it
submits nothing and needs none of the settings above (it only warns about those this host lacks).

## 3. The smoke and the rehearsal

```bash
bash code/experiments/package/run.sh --platform smoke --no-exclusive smoke
#  -> "SMOKE OK" at the end of logs/smoke/sp-smoke-*.out (up to an hour); wait for it before the rehearsal
bash code/experiments/package/run.sh --platform rehearsal --rehearsal --no-exclusive --optional [--llama-licence] all
#  -> 50 small jobs (51 with --llama-licence): every tier's smallest cells (GPT-2 / OPT-125M, 2 blocks), about three
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
| `quality` | 6 (7) | OPT-125M: perplexity and zero-shot accuracy (20 documents per task) with the certified and the legacy recipe, the reproducible build and its commitment; Qwen3-0.6B: the Llama recipe on the smoke model and, with a fixed prefix and a kept reference, on the checkpoint, `llama_diag` and `llama_attrib`; with `--llama-licence`, the smoke model of Llama-2-7B, which reads the gated repository from a compute node |
| `tv` | 2 | the sampler's closed-form and Monte Carlo checks on its toy distributions; its total-variation distance on OPT-125M (legacy recipe, 2 positions, 2 workers) |
| `attack` | 5 | the path test's bound; the MNIST scripts on the stored networks; OPT-125M: the single-neuron attack and its backdoor, the in-range edits, the sampler game |
| `long` | 2 | one Llama-2-7B block at 4,608 tokens: the CPU verifier and re-execution |

The attack tier's OPT-125M documents are named by their model (`attack/inrange_llm_opt-125m_any.json`,
`attack/adaptive_game_opt-125m.json`, ...): `numbers.py` counts them for their kinds, and no table or text number
reads them.

## 4. The run

On a machine of your own, or a partition whose nodes are free, one invocation runs everything as a chain of
whole-node jobs:

```bash
bash code/experiments/package/run.sh --platform final --optional --llama-licence all
```

On a shared partition a job that asks for a whole node waits until every job on some node has ended, and the chain
waits that long for each of its 179 jobs. There, the paper's run submits its tiers as eight jobs (`--one-job`), each
of which runs its tiers' jobs one after another in its allocation: the timed tiers in six jobs that have a node to
themselves, and the tiers that time nothing (`quality`, `tv`, `attack`) in two jobs on one GPU of a shared node:

```bash
R="bash code/experiments/package/run.sh --platform final --llama-licence"
$R --one-job smoke cpuv long                     # expected 11-17 h
$R --one-job smoke bench                         # 9-13 h
$R --one-job smoke real                          # 9-13 h
$R --one-job smoke serve gen batch               # 13-17 h
$R --one-job smoke v1 setup improve              # 9-16 h
$R --one-job smoke gpuv                          # 5-7 h
$R --one-job --no-exclusive smoke quality        # 11-17 h, not timed
$R --one-job --no-exclusive smoke tv attack      # 6-9 h, not timed
```

Each of these jobs starts with the smoke on the node it was given (and stops if it fails), writes the same job logs
as the chain (`logs/final/sp-<cell>-<id>.out`) and lists what it ran in its own log
(`logs/final/node-<tiers>-<job id>.log`). Its time limit is 24 hours (`--max-hours`), and SLURM queues a pre-empted
one again by itself (`--requeue`); a job that ends successfully leaves a marker in `logs/final/done/`, and the
allocation, when it starts again or when the same command is submitted again after its time limit, runs only the
jobs without a marker (and the smoke). A later job's records of a cell replace an earlier one's.

| Tier | Jobs | Expected (h) | Time limits (h) | What it gives the paper |
|---|---:|---:|---:|---|
| `smoke` | 1 | 1 | 2 | the platform gate |
| `cpuv` | 13 | 6-9 | 23 | the CPU verifier of the full Llama-2-7B, Qwen3-4B and OPT-1.3B (Kpre and C, 64 and 2,048 tokens) against every re-execution baseline on the same 8 threads (the exact integer pass with the native int8 GEMM, with `torch._int_mm` and with fp32 GEMMs; fp32, bf16 and dynamic int8), the allocator setting of the verifier as a like-for-like row (`tab:options`, the abstract's "at least N times"); V1's optimised and baseline CPU verifiers |
| `v1` | 13 | 4-8 | 26 | V1 against v0: bytes, prover and verifier (`tab:eval-v1`) |
| `serve` | 5 | 8-10 | 25 | the prover against unverified serving on the same GPU, batches of 1 to 64 prompts |
| `bench` | 43 | 8-12 | 48 | the v0 cost table (`tab:eval-llm`), the comparison with published systems (`tab:eval-compare`), the outcome counts, the image models' attack cells |
| `gpuv` | 12 | 4-6 | 19 | the streaming GPU verifier, int8/fp16/bf16 re-execution on the same GPU, the host-to-device bandwidth |
| `gen` | 3 | 2-3 | 7 | generated responses proved as one prefill: Llama-2-7B and Qwen3-4B after a 512-token prompt, greedy and sampled, and GPT-2's prover per generated token (`tab:gen`, `tab:gen-full`, Sec. `sec:gen-prefill`) |
| `batch` | 7 | 2-3 | 8 | several prompts against one opening |
| `real` | 26 (17) | 8-12 | 45 (23) | the certified OPT-1.3B and OPT-6.7B end to end next to random weights (`tab:eval-real`); with `--llama-licence` the real Llama-2-7B |
| `setup` | 2 | 1-2 | 4 | the setup proof of an untrusted commitment (`tab:setup`) |
| `improve` | 15 | 3-5 | 17 | each implementation improvement against the path it replaces (`tab:eval-improve`); V1's verifier by phase; the GKR prover per leaf |
| `quality` | 20 (18) | 10-16 | 69 (55) | `tab:quality` (OPT; OpenLLaMA-7B, TinyLlama, Qwen3; Llama-2 with `--llama-licence`), the recipe's alpha on the validation split, the recipe studies of the extended version (OPT-1.3B and OPT-125M on the test split), `tab:zeroshot`, the reproducible build |
| `tv` | 2 | 1-2 | 3 | the sampler's total-variation distance on the certified OPT-1.3B, and its checks |
| `attack` | 5 | 4-6 | 12 | the attack: reproduction, MNIST, OPT-6.7B, the in-range edits (`tab:inrange`), the sampler games |
| `long` (optional) | 12 | 4-7 | 28 | prompts of 2,048 to 16,384 tokens (`tab:eval-long`) |

`--optional` adds the `long` tier, which the paper's Discussion and `tab:eval-long` read. Submit it in the same
invocation as the other tiers, so that it is chained after them and gated by the smoke; a second invocation starts a
new chain. Without a Llama-2 licence, leave out `--llama-licence`: the Llama-2 cells then stay empty. The counts and
limits are those with `--llama-licence`; without it (in parentheses) the `real` tier has 17 jobs and 23 h of limits
and the `quality` tier 18 jobs and 55 h. `all` is then 156 jobs and 272 h of limits (167 and 308 h with
`--llama-licence`; the `long` tier adds 12 jobs and 28 h).

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
  memory"`; the first job of a real checkpoint spends some minutes downloading it.
* `A CORRECTNESS CHECK FAILED` or `FAILED:` in a log (an honest query rejected, a tamper accepted, a byte count that
  disagrees with the model) is a real problem; the job's log says which check.
* A whole-node job (`--one-job`) that hit its time limit: submit the same command again; it skips the jobs that
  ended successfully (`logs/<name>/done/`). Its log names each job it ran or skipped and the exit status of each
  that failed.
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
tar czf final.tgz --exclude=triton --exclude=native --exclude=setup_cache --exclude='ref_*.pt' logs/final \
    code/artifacts/comparison/raw_final
rsync -a --exclude=triton --exclude=native --exclude=setup_cache --exclude='ref_*.pt' logs/final/ code/artifacts/final/
```

## 7. From the records to the tables

```bash
cd code
PYTHONPATH=src:. ../.venv/bin/python -m experiments.comparison.aggregate --platform final   # -> tables_final/
PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final --check
PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final
#  (the first numbers command: coverage and checks; the second writes paper_sp2027/generated/)
```

The coverage report names every job without records and every table cell still missing; a `CHECK FAILED` or `PARSE`
line, or `--strict` with anything missing, gives exit status 1.

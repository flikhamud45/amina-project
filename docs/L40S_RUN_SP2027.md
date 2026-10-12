# The S&P 2027 final run on the L40S and EPYC (package v2)

For the teammate who runs the jobs on the node with the NVIDIA L40S and the AMD EPYC (Zen 4). Status: **package v2**
(Plan 2 F6). The code is frozen at the tag `sp2027-freeze` on Sun 1 Nov 23:59 and handed over then; until then
the cells below can still change. The run window is Mon 2 - Sun 8 Nov; re-runs of failed cells are possible until
Tue 10 Nov.

## 1. What the run is

One frozen, scripted run. `sp2027.sh all` submits every tier the paper needs; each job starts only after the
previous one has ended, so no two timed jobs share the machine; nothing after the smoke starts unless the smoke
succeeded. The jobs write JSON lines, and one script (`experiments/5_comparison/sp2027_numbers.py`) turns them into
the paper's tables and text numbers. Nothing has to be computed or copied by hand.

## 2. What the machine needs

| Need | Why | How to check |
|---|---|---|
| NVIDIA L40S (48 GB, sm_89), a driver for CUDA 12.1 | the prover, the GPU verifier, U1/U2 | `nvidia-smi`; `python -c "import torch; print(torch.cuda.get_device_name(0), torch.cuda.get_arch_list())"` lists `sm_89` or `sm_90` |
| AMD EPYC (Zen 4), 8 free cores, about 256 GB RAM | the CPU verifier and every re-execution baseline run on 8 threads; the 7B cells at 2,048 tokens ask for 256 GB | `lscpu`, `free -g`; with less RAM, `export PVI_MEM_MAX=<GB>` (the largest `--mem` any job asks for) |
| Python 3.12 environment as `.venv` at the checkout's root (or `PVI_PYTHON`) | torch 2.5.1+cu121, `code/requirements.txt` and `code/requirements-llm.txt` | `.venv/bin/python -c "import torch, transformers, triton; print(torch.__version__)"` |
| A C++ compiler: `CXX` | the native CPU kernels (`-march=native`: AVX-512, VNNI and AVX512_BF16 on Zen 4) of the CPU verifier and of the exact int8 re-execution | `$CXX --version` (g++ 9 or later). Without root: conda-forge's `gxx` through micromamba in a folder of one's own (`docs/improvements/I_server_setup.md` Sec. 4) |
| The Python headers for Triton's launcher | Triton compiles a small launcher with `Python.h` on first use | `python -c "import os, sysconfig; print(os.path.exists(os.path.join(sysconfig.get_paths()['include'], 'Python.h')))"`; if `False`, copy the headers into a folder and `export PVI_PY_INCLUDE=<that folder>` (`I_server_setup.md` Sec. 3) |
| A Triton cache the jobs can write | Triton's compiled kernels | set by the script to `logs/$PVI_PLATFORM/.triton`; `export TRITON_CACHE_DIR=<dir>` to put it elsewhere (e.g. a node-local disk) |
| `HF_HOME` on a node-local disk with 40 GB free | the `real` tier downloads facebook/opt-1.3b and facebook/opt-6.7b (about 16 GB) and the WikiText-2 test parquet; the `quality` tier Llama-2-7B and Llama-2-13B (about 13 + 26 GB) | `df -h $HF_HOME` |
| About 0.5 GB for the setup cache | every mode-C job of a model reuses the first job's commitment (`logs/$PVI_PLATFORM/.setup_cache`, 15-70 MB per model and plan); the setup tier always measures the setup itself | set by the script; `export PVI_SETUP_CACHE=<dir>` to move it, `PVI_SETUP_CACHE=` (empty) to disable it |
| The trained image models and their datasets (not in the repository) | the `bench` tier's image cells (MLP, LeNet-5, VGG-11/16, ResNet-18 on CIFAR-10 and on 224-pixel dogs-vs-cats) load trained weights | the five files `code/artifacts/fullcheck/models/{lenet5,vgg11,vgg16,resnet18_cifar,resnet18_224}.pt` from the earlier L40S run's checkout (or `experiments/4_defence_benchmark/train.py --model <m>`, seed 0); CIFAR-10 in `cifar-10-batches-py` layout under `$PVI_CIFAR_ROOT`; the dogs-vs-cats cache `dogs_cats.pt` under `$PVI_CACHE` (default `code/artifacts/fullcheck/cache`) or ImageNet under `$PVI_IMAGENET_ROOT`. `sp2027.sh all` and `bench` stop before submitting anything if one is missing |
| SLURM (`sbatch`, `squeue`), or none | the jobs | with SLURM: `export SBATCH_PARTITION=... SBATCH_GRES=gpu:<type>:1 [SBATCH_ACCOUNT=...]`; without: `export PVI_LOCAL=1` (every job runs in the foreground, one after another: start it in `tmux` or under `nohup`) |
| Exclusive use, 24/7 | the timings | every job asks for `--exclusive`; please keep other GPU- or CPU-heavy work off the node for the whole run (about 2-3 days, Sec. 5) |

Every language model is built with random weights (the costs depend only on the shapes), so nothing is downloaded
except in the `real` tier (OPT, public) and the `quality` tier, which needs the Llama-2 licence on **your own**
Hugging Face account and your own login (`export PVI_LLAMA_LICENCE=1`; the team asks for this tier: Llama-2-7B is the
model the cost tables lead with). Never use another person's token. The image
models are the trained ones of the earlier run (the row above); MNIST is downloaded on first use.

## 3. Setup, once

```bash
git clone <repository> sp2027 && cd sp2027 && git checkout sp2027-freeze       # or sp2027-freeze.1 after a hotfix
python3.12 -m venv .venv && .venv/bin/pip install torch==2.5.1 torchvision==0.20.1 \
    --index-url https://download.pytorch.org/whl/cu121 \
    && .venv/bin/pip install -r code/requirements.txt -r code/requirements-llm.txt
export PVI_PLATFORM=l40s_sp2027                     # the run's name: logs/l40s_sp2027/, raw_l40s_sp2027/
export SBATCH_PARTITION=<L40S partition> SBATCH_GRES=gpu:<type>:1        # or: export PVI_LOCAL=1
export HF_HOME=<node-local folder with 40 GB free>
export CXX=<path to g++>
export PVI_PY_INCLUDE=<headers folder>              # only if the check of Sec. 2 printed False
export PVI_C_POLICY=<plan>                          # only if the team says so (default cnn18c, with the setup cache)
```

`PVI_DRYRUN=1 bash code/experiments/4_defence_benchmark/slurm/sp2027.sh all` prints every job it would submit (127
lines with the optional tiers; the listing of the freeze is `docs/improvements/F6_dryrun.txt`) and each tier's job
count, job-limit sum and estimate, without submitting anything.

## 4. The smoke (at most 2 hours)

Before the run, and if you agree once between 19 and 27 Oct, on the same machine:

```bash
export PVI_PLATFORM=l40s_smoke                       # keeps the real run's folders clean
bash code/experiments/4_defence_benchmark/slurm/sp2027.sh smoke
#  -> wait for "SMOKE OK" at the end of logs/l40s_smoke/sp-smoke-*.out (up to 1 h)
PVI_REHEARSAL=1 PVI_OPTIONAL=1 bash code/experiments/4_defence_benchmark/slurm/sp2027.sh all
#  -> 22 small jobs, every tier's smallest cells (GPT-2 / OPT-125M, 2 blocks), about 1 h
```

The smoke runs the GPU tests (Triton kernels and `torch._int_mm` on sm_89), the native-kernel tests (`-march=native`
on Zen 4), the Fiat-Shamir, serving, generation, setup-proof and package tests, one V1 query, and the
host-to-device copy. The rehearsal then exercises every tier's script with this machine's environment (`CXX`,
`HF_HOME`, `PVI_PY_INCLUDE`, SLURM). Please send back `logs/l40s_smoke/` and
`code/artifacts/comparison/raw_l40s_smoke/` (Sec. 8). This is not "running the L40S": it is a check that takes at
most 2 hours.

## 5. The run: order, hours, what each tier gives

```bash
export PVI_PLATFORM=l40s_sp2027
bash code/experiments/4_defence_benchmark/slurm/sp2027.sh all          # the must tiers, in this order
# then the Llama-2 quality tier, with your own licence and login (the team asks for it):
PVI_LLAMA_LICENCE=1 bash code/experiments/4_defence_benchmark/slurm/sp2027.sh quality
# optional, only if the team asks: PVI_OPTIONAL=1 (adds quality and long at the end of 'all')
```

| # | Tier | Jobs | Estimate (h) | Job limits (h) | What it gives the paper |
|---|---|---:|---:|---:|---|
| 1 | `smoke` | 1 | 1 | 2 | the platform gate: nothing after it starts unless it ends in `SMOKE OK` |
| 2 | `cpuv` | 13 | 6-9 | 22 | the CPU verifier of the full Llama-2-7B, Qwen3-4B and OPT-1.3B (Kpre and C, Fiat-Shamir, 64 and 2,048 tokens) against every re-execution baseline on the same 8 threads (exact int8-GEMM integer, fp32-GEMM integer, fp32, bf16; dynamic int8, not exact); V1's old and new CPU verifier; the abstract's "at least N times" |
| 3 | `v1` | 13 | 4-8 | 26 | V1 against v0 (bytes, prover, verifier) on GPT-2 to Llama-2-13B (tab:eval-v1) |
| 4 | `serve` | 5 | 8-10 | 25 | the prover against unverified serving on this GPU, batches 1-64 (A7) |
| 5 | `bench` | 43 | 8-12 | 27 | the v0 cost table (tab:eval-llm), the comparison with published systems (tab:eval-compare), the outcome counts |
| 6 | `gpuv` | 12 | 4-6 | 19 | the streaming GPU verifier, int8/fp16/bf16 re-execution on this GPU, the PCIe copy bandwidth, peak memory |
| 7 | `gen` | 2 | 2-3 | 6 | generated responses (Llama-2-7B, Qwen3-4B; 512-token prompt, 64 and 256 tokens; greedy and sampled) with two tamper cells |
| 8 | `batch` | 7 | 2-3 | 6 | many prompts against one opening (C4a) |
| 9 | `real` | 16 | 4-6 | 22 | real-weight OPT-1.3B and OPT-6.7B next to random weights (F3a) |
| 10 | `setup` | 2 | 1-2 | 4 | the setup proof of an untrusted commitment (Llama-2-7B), prover and verifier time |
| | **must total** | **114** | **40-60** | **159** | about 2-2.5 days of the 7-day window; the job limits are caps (a job ends when its work is done) |
| 11 | `quality` (requested) | 2 | 4-7 | 14 | Llama-2-7B and Llama-2-13B quality at 2,048 tokens with the main algorithm's integer recipe; only with your own Llama-2 licence (`PVI_LLAMA_LICENCE=1`) |
| 12 | `long` (optional) | 12 | 4-7 | 28 | long prompts (2k-16k tokens, Llama-2-7B shapes, 2-3 blocks) |

The estimates come from the measured times of the same code on the RTX 2080 Ti, the Xeon and the earlier L40S runs
(`docs/improvements/F6_final_run_package.md` Sec. 3); the L40S/EPYC will mostly be faster. One tier at a time is
possible too (`sp2027.sh cpuv`, then `v1`, ...); its jobs are then chained among themselves. With several GPUs in the
node, the chain still runs one job at a time.

## 6. While it runs

* `squeue -u $USER` (or the foreground output with `PVI_LOCAL=1`) shows where the chain is. Each job writes
  `logs/l40s_sp2027/<job name>-<job id>.out`; the results are the lines starting with `{`.
* Expected: in `serve`, the largest joint batches of the 7B and 13B models are recorded as
  `"skipped": "out of memory"`; in `real`, the first OPT-6.7B job spends 10-20 minutes downloading.
* A log line `A CORRECTNESS CHECK FAILED` or `FAILED:` (an honest query rejected, a tamper accepted, a byte count
  that disagrees with the model) means a real problem: please send that log as it is and tell the team.
* A job that hit its time limit or was pre-empted: re-run its tier command (`sp2027.sh <tier>`). Jobs still queued
  are not submitted twice, and finished `bench`/`batch` cells are skipped by `bench.py` itself.
* If the node has less RAM than a job asks for, the job never starts: `export PVI_MEM_MAX=<GB>` and re-run the tier.

## 7. The hotfix rule (plan F6 part 14)

If the smoke fails on Mon 2 Nov for a reason of the platform (Triton or `torch._int_mm` on sm_89, the native
kernels on Zen 4, `CXX`, `HF_HOME` or `PVI_PY_INCLUDE` in this environment): stop and send the smoke's log. On Mon 2 -
Tue 3 Nov the team may make only platform-only fixes that leave every byte and verdict test unchanged; the fixed
code is tagged `sp2027-freeze.1`, and the smoke runs again before the run. No other change enters during the run.

## 8. What to send back

```bash
tar czf l40s_sp2027.tgz --exclude='.triton' --exclude='.native' logs/l40s_sp2027 \
    code/artifacts/comparison/raw_l40s_sp2027
```

and, once, the machine's description: `nvidia-smi -q > env-nvidia.txt; lscpu > env-cpu.txt; free -g > env-mem.txt;
.venv/bin/pip freeze > env-pip.txt; git rev-parse HEAD > env-git.txt`. Nothing else is needed (the same for the
smoke's `l40s_smoke` folders).

## 9. The optional vLLM environment (A7-vLLM)

Only if the team asks and you agree: a separate virtual environment (about 5 GB) with a vLLM release built for
torch 2.5.1, used with dummy weights (no downloads, no licence) and CUDA graphs, as a serving-engine baseline next
to `serve`. Its harness is a should item of the plan and is not part of package v2; if it lands before the freeze,
the team will add its tier and tell you the one extra command. Without it, nothing changes.

## 10. Rules

* Work only in your own checkout and folders; do not modify or delete anyone else's files or jobs.
* Do not use another person's Hugging Face token. The models downloaded here are public (facebook/opt-*, the
  WikiText-2 parquet), except Llama-2-7B and Llama-2-13B in the `quality` tier, which need your own licence.

## 11. For the team: from the outputs to the tables

```bash
cd code
python experiments/5_comparison/aggregate.py --platform l40s_sp2027        # bench.py's raw records -> tables
python experiments/5_comparison/sp2027_numbers.py --expect ../docs/improvements/F6_dryrun.txt --check   # coverage
python experiments/5_comparison/sp2027_numbers.py                          # -> paper_sp2027/generated/
```

The coverage report names every job without results and every table cell still missing; a `CHECK FAILED` or `PARSE`
line, or `--strict` with anything missing, gives exit status 1.

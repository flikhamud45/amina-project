# Final runs for the S&P 2027 version (strong GPU)

For the person running the jobs on the L40S (or any 48 GB+ GPU). Status: **draft**. The commit to run is fixed by
the team before the runs (the "freeze"); until then this file lists what will be run and how.

## What these runs give the paper

* **V1 against v0** (`sp2027.sh v1`): proof bytes, prover and verifier times on the same queries for GPT-2, OPT-125M,
  OPT-1.3B, OPT-6.7B, Llama-2-7B (full models) at 64 and 2,048 tokens, Llama-2-13B at 64 tokens (and 2 blocks at 2,048),
  Qwen3-4B at 64 tokens. On the 2080 Ti the GKR needed small instances (`lmax` 24) and Llama-2-7B's committed mode
  did not fit; with 48 GB both go away (`lmax` 27).
* **Quality** (`sp2027.sh quality`): WikiText-2 perplexity of the integer OPT-125M/1.3B/6.7B with and without the
  SmoothQuant fix (OPT-6.7B never fitted our machines), and V1's bytes on the smoothed real-weight graphs.
* **The verifier on the GPU** (`sp2027.sh gpuv`): v0's streaming GPU verifier and V1 with a GPU verifier at 2,048
  tokens.
* **In-range edits on OPT-6.7B** (`sp2027.sh attack`): how many FFN neurons an attacker must edit, each inside a
  range calibrated on held-out text, to change the next token (the OPT-1.3B result is in the paper).
* **The CPU verifier against re-execution** (`sp2027.sh cpuv`): the full Llama-2-7B and OPT-1.3B verified on 8 CPU
  threads with the native kernels (B5), and re-executed (integer and fp32) on the same threads.
* **The v0 benchmark with every sp2027 improvement** (the existing `strong_gpu.sh must` and `should` tiers with a new
  `PVI_PLATFORM`): the paper's Table 3 and comparisons on this GPU.

## Setup

1. A checkout of branch `sp2027` at the freeze commit, with the project's Python environment as `.venv` at its root
   (torch 2.5.1+cu121 runs on the L40S), on a file system the compute nodes can read.
2. Environment for the submissions (from the checkout's root):
   ```bash
   export PVI_PLATFORM=l40s_sp2027
   export SBATCH_PARTITION=<the L40S partition> SBATCH_GRES=gpu:<type>:1 [SBATCH_ACCOUNT=...]
   export HF_HOME=<a folder with about 40 GB free, readable by the compute nodes>
   # only if the compute nodes lack the Python headers (Triton's launcher; see docs/improvements/I_server_setup.md):
   export PVI_PY_INCLUDE=<a copy of the login node's python3.12 include folder>
   # only for the CPU verifier's native attention (B5): a C++ compiler the nodes can run
   export CXX=<path to g++>
   ```
3. `bash code/experiments/4_defence_benchmark/slurm/sp2027.sh smoke` and wait for `SMOKE OK` in
   `logs/l40s_sp2027/sp-smoke-*.out` (GPU tests of the new code, then one small V1 query). Do not start the other
   tiers before it passes: exactness is checked per GPU type.

## The runs, by priority

| Tier | What | Jobs | Hours (rough) |
|---|---|---|---|
| `smoke` | GPU tests, one V1 query | 1 | 0.5 |
| `v1` | V1 against v0, timed | 13 | 20-30 |
| `quality` | perplexity (fp32, scalar gains, smoothing) and V1 on real weights | 7 | 10-15 |
| `gpuv` | the verifier on the GPU | 4 | 4-6 |
| `cpuv` | the CPU verifier and re-execution, same CPU | 6 | 6-10 |
| `attack` | in-range edits on OPT-6.7B (range checks, value-aware sampler) | 2 | 4-8 |
| `strong_gpu.sh must` | the v0 benchmark with the improvements (see `STRONG_GPU_PLAN.md`, Part C) | about 20 | 30-40 |

Each `sp2027.sh` tier submits its jobs with `--dependency=singleton`; re-running a tier after a pre-emption resubmits
only what is not queued. `PVI_DRYRUN=1` prints the `sbatch` lines without submitting.

## What to send back

The folder `logs/l40s_sp2027/` (every job's log: the results are the lines starting with `{`) and, from the
`strong_gpu.sh` tiers, `code/artifacts/comparison/raw_l40s_sp2027/`. Nothing else is needed.

## Rules

* Work only in your own checkout and folders; do not modify or delete anyone else's files or jobs.
* Do not use another person's Hugging Face token. The models used here (facebook/opt-*, the WikiText-2 parquet) are
  public; gated models (Llama-3) need the team's own licence access.

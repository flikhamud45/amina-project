# zkLLM's public demo on the run's GPU

Sec. `sec:eval-compare` and the extended version's `ext:disc-options` set zkLLM's prover on the run's own GPU model
against ours on prompts of the same length (`\ZkllmSameGpu...`); `tab:eval-compare` keeps zkLLM's published A100
numbers. These scripts run zkLLM's public demo ([github.com/jvhs0706/zkllm-ccs2024](https://github.com/jvhs0706/zkllm-ccs2024),
commit `993311e`, its code unchanged) for Llama-2-7B and Llama-2-13B at 2,048 tokens and time each of its proof
binaries. The run package's optional `zkllm` tier runs them on the run's node; the stored runs
(`artifacts/results/zkllm_l40s/`) were measured in the same way.

| Paper result | Script | Output |
|---|---|---|
| `\ZkllmSameGpuProverLlama`, `\ZkllmSameGpuProverLlamaL` (zkLLM's prover on the GPU, about 844 s and 2,628 s on an L40S), the factors over ours (`\ZkllmSameGpuFactor...`) and `\ZkllmSameGpuRun` | `run_zkllm.sh` (`build`, then `prove` per model), `summarise.py` | `logs/<name>/zkllm/<model>-T2048/`, `logs/<name>/zkllm/summary.csv`, one record per model in the job's log |

`experiments/paper/bench.py` (`literature`, `zkllm_runs`) reads `summary.csv` and each run's `env.txt`.
`experiments/paper/macros.py` (`zkllm_same_gpu`) uses the run's own runs when both models are there and their
`env.txt` names the run's GPU (`\ZkllmSameGpuRun` is then "in the same run"), else the stored runs on the same GPU
model ("in separate runs"), and prints nothing for another GPU.

## Through the run package

```bash
bash code/experiments/package/run.sh --platform final --llama-licence --one-job zkllm     # its own whole-node job
```

or as part of `run.sh --platform final --optional --llama-licence all`. The tier has three jobs: `zk-build` (2 h
limit, 32 GB), `zk-llama2-7b` (2 h, 64 GB) and `zk-llama2-13b` (4 h, 128 GB); expected 3 to 5 hours in all on an L40S
node. Without `--llama-licence` the tier submits nothing and says why. Its options (`run.sh --help`):

| `run.sh` option | Default | Meaning |
|---|---|---|
| `--zkllm-home DIR` | `logs/<name>/zkllm/home` | `DIR/zkllm`, zkLLM's checkout, and `DIR/env`, its Python environment; whatever is missing the build job creates |
| `--zkllm-scratch DIR` | the first of the node's `$TMPDIR`, `/tmp` and the zkLLM home with the space free | zkLLM's fixed-point weights and commitments (27 GB for Llama-2-7B, 52 GB for Llama-2-13B), deleted at each job's end |
| `--zkllm-python PY` | the run's interpreter (`--python`) | the Python (3.8 to 3.12) from which zkLLM's environment is created; the run's `.venv` is never changed |
| `--cuda-home DIR` | `nvcc` on the `PATH`, else `CUDA_HOME` | the CUDA toolkit (`DIR/bin/nvcc`, `DIR/include`) that builds zkLLM; CUDA 12.1 as zkLLM recommends (the stored runs: CUDA 12.1, gcc 11) |

Before it submits anything, `run.sh` checks what the tier needs from the host: git (unless the checkout is there),
make, gcc, g++ and nvcc (unless the home holds a built checkout), the interpreter's version and 10 GB free for the
home (unless its environment exists), 58 GB free in `--zkllm-scratch` (when given), `HF_HOME`, and `HF_TOKEN` (or
`$HF_HOME/token`): the Llama-2 repositories are gated. The jobs run on the GPU node, so a CUDA toolkit loaded only
on the login node (e.g. `module load cuda/12.1`) must be loaded in the jobs too: pass `--cuda-home <its folder>` or
put it on the `PATH` of the shell that runs `run.sh` (sbatch passes the environment to the jobs).

## The scripts

`run_zkllm.sh build --home DIR [--python PY] [--cuda-home DIR] [--arch sm_NN] [--hub DIR] [--models 7,13]
[--repo URL] [--commit SHA] [--jobs N]`:

1. clones zkLLM into `DIR/zkllm` at `993311e` (or checks that the checkout there is at that commit with no tracked
   file changed);
2. creates `DIR/env` with `--python -m venv` and `requirements.txt` (torch 2.3.1 built for CUDA 12.1,
   transformers 4.40.2, as zkLLM's scripts need);
3. builds zkLLM with its own Makefile for the node's GPU: `make CONDA_PREFIX=<CUDA folder> ARCH=sm_<compute
   capability> all` (`nvidia-smi --query-gpu=compute_cap`; sm_89 on an L40S), the Makefile's variables given on
   the command line, its files unchanged; `DIR/BUILD.txt` records the architecture, nvcc and gcc;
4. installs the timing shims (`install_timing_shims.sh`): each binary `X` becomes `X.real` and a shim that runs it
   under `/usr/bin/time` and appends `BIN <name> <wall s> s maxrss <KB> KB exit <code> args ...` to the run's
   `bin_times.log`; zkLLM's Python scripts, which call `./X`, are timed unchanged;
5. links `DIR/zkllm/model-storage`, where zkLLM's scripts look for the checkpoints, to the Hugging Face hub cache
   (`--hub`, default `$HF_HUB_CACHE`, else `$HF_HOME/hub`), so that the run's other Llama-2 jobs and zkLLM share one
   copy, and downloads Llama-2-7B and Llama-2-13B (configuration, tokenizer, safetensors) and WikiText-2's raw test
   split (`fetch.py`) into it; ends with `ZKLLM_BUILD_OK`.

`run_zkllm.sh prove --home DIR --size 7|13 --out OUT [--seq 2048] [--reps 3] [--layers 0,1] [--scratch DIR]
[--parquet FILE] [--threads 8] [--time-cmd PATH]`:

1. removes the work folders that a pre-empted job of this user left, checks the scratch space (32 GB for
   Llama-2-7B, 58 GB for Llama-2-13B) and links `DIR/zkllm/zkllm-workdir` to a
   fresh folder in it;
2. writes `OUT/llama2-<size>b-T<seq>/env.txt`: the host, the job's GPU (`nvidia-smi`: name, driver, memory), the
   CPU model, zkLLM's commit, the build, whether zkLLM's tracked files are unchanged, the Python, torch and
   transformers versions, the checkpoint's revision and the scratch folder;
3. runs zkLLM's `llama-ppgen.py` and `llama-commit.py <size> 16`, then `mkinput.py` (the embeddings of the first
   `--seq` tokens of WikiText-2's test split in zkLLM's fixed-point format), then `--reps` times, for each layer of
   `--layers`, zkLLM's six proof scripts in the order of its README (RMSNorm, self-attention, skip connection,
   RMSNorm, FFN, skip connection), each under `/usr/bin/time -v`, offline, while `nvidia-smi` samples the job's GPU
   every 0.5 s;
4. counts the verifier messages (`verdicts.txt`), packs the scripts' logs, deletes the scratch folder, rewrites
   `OUT/summary.csv` (`summarise.py --csv OUT`) and prints the run's figures and its record (`summarise.py
   --record`). The exit status is 0 only if every proof binary exited 0 and every proved layer printed zkLLM's
   verifier messages (the record's `ok`; `paper/numbers.py` reports a record that is not `ok` as `CHECK FAILED`).

`summarise.py` (standard library only):

```bash
python experiments/comparison/zkllm/summarise.py <run dir> [<layers>]   # each binary's median, per layer, whole model
python experiments/comparison/zkllm/summarise.py --csv [<folder>]       # <folder>/summary.csv (default: the stored runs)
python experiments/comparison/zkllm/summarise.py --record <run dir>     # the run's record, one JSON line
```

**How the time per layer is computed.** The sum of the median times of the proof binaries in `bin_times.log`
(`--reps` repetitions of each layer of `--layers`; the setup binaries `ppgen` and `commit-param` excluded). The
demo's attention stage writes no output file, so the first skip connection of each layer does not run (its script
exits with "Input or output file does not exist"); the measured skip connection is therefore counted twice. The
whole model's time is the time per layer times the model's decoder layers (32 and 40). The stored runs give
26.37 s per layer (844 s) for Llama-2-7B and 65.71 s (2,628 s) for Llama-2-13B.

The rehearsal (`run.sh --platform <name> --rehearsal --llama-licence zkllm`) runs one job: the build for Llama-2-7B
only and one repetition of layer 0 at 2,048 tokens, which takes every step of the full jobs.

## Without the run package

```bash
cd code
bash experiments/comparison/zkllm/run_zkllm.sh build --home ../zkllm_home --python python3.12
bash experiments/comparison/zkllm/run_zkllm.sh prove --home ../zkllm_home --size 7 --out ../logs/mine/zkllm
```

on a GPU node, with `HF_HOME` (or `--hub`) and `HF_TOKEN` set. A small version that runs on a laptop CPU does not
exist: zkLLM's code runs on CUDA GPUs only. `tests/test_zkllm_package.py` runs `prove` with stand-ins for zkLLM's
scripts and checks its folder, summary and record, and `summarise.py` on the stored runs.

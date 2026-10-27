# The final run: troubleshooting

This file goes with `FINAL_RUN.md`. Each entry gives the message you would see, its cause and the fix you can apply
yourself. If a problem is not listed here, or the fix does not help, please do not try other changes. Send us the
files listed in the last section. We will send a fix, and you will then resubmit only the jobs that failed.

The section numbers ("Sec. 2", "Sec. 4", ...) refer to `FINAL_RUN.md`. `R` is the function defined there (Sec. 3).
"The `cpuv` line", "the `zkllm` line" and so on are the submission lines of Sec. 4.

## Ground rules: what keeps the numbers valid

**Never change:**

* **Code.** Do not change any file of the clone (`git status --short --untracked-files=no` must stay empty), and do
  not change zkLLM's checkout. A fix to code always comes from us, as a new bundle.
* **The submission lines.** Do not change, add or remove any option of a submission line of Sec. 4, except the
  ones listed under "You may change" below. In particular, never use `--queries`, `--queries-head`, `--policy`,
  `--no-pin`, `--no-native-ok`, `--no-chain`, `--setup-cache`, `--no-bigmem`, `--zkllm-seqs`, `--max-hours`,
  `--telemetry-interval`, `--no-env-snapshot`, `--rehearsal`, `--random-weights`, `--require-gpu any`,
  `--require-cpu any` or `--mem-max` unless we ask. Never give `--no-exclusive` to the first ten lines, which are the
  timed ones.
* **The platform name.** It stays `--platform final`.
* **The run's Python environment** (`.venv`) and the compiler in `CXX`. They should stay the ones of the
  rehearsal.
* **The node while a timed job runs.** Start nothing else on the node of a timed allocation (`srun --jobid`, ssh
  work).
* **Records.** Do not run single jobs by hand outside `run.sh`. Do not delete or edit records or logs in
  `logs/final/`. Do not delete markers in `logs/final/done/` unless this file or we say so.

**You may change:**

* `--exclude-nodes` (add a node).
* `--cuda-home`, `--zkllm-home`, `--zkllm-scratch`, `--zkllm-python` on the `zkllm` line.
* The environment variables `TMPDIR`, `HF_HOME` (to a folder that already holds the checkpoints, or with space),
  `HF_TOKEN`, `CMAKE` and `PVI_PY_INCLUDE`, and loading modules (CUDA, an older gcc for zkLLM, CMake).

Three general points:

* **Changing the environment needs a new submission.** sbatch records the environment when a job is submitted. A
  job that is still queued, or that SLURM requeues after a pre-emption, keeps the old environment. After changing
  a variable or a module, cancel the queued allocation (`scancel <job id>`) and submit the same line again.
* **Resubmitting is always safe.** Submitting a line of Sec. 4 again runs only the jobs that have no marker in
  `logs/final/done/`. While the line's allocation is still queued or running, a second submission submits nothing
  (`skip sp-node-...: already queued (job N)`).
* **Small problems: fix and resubmit; everything else: report.** If an entry below gives a fix you can apply (a
  setting, a module, disk space, a node to exclude), apply it and submit the same line again. If the entry says
  "send us", or the problem is not listed, send us the files of Sec. 7 and leave the job's records as they are.

## 1. Before anything is submitted (`run.sh:` lines)

`run.sh` checks the host before it submits anything. Each problem is one `run.sh: ...` line, followed by `run.sh:
nothing submitted`. A dry run prints the same lines but submits nothing in any case.

| Message | Cause | Fix |
|---|---|---|
| `export HF_HOME=<folder the nodes see, ~100 GB free> ...` | `HF_HOME` is not set | source your environment file (Sec. 2) |
| `HF_HUB_CACHE=... takes precedence over HF_HOME and is not writable here: unset HF_HUB_CACHE ...` (also `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`, `HUGGINGFACE_HUB_CACHE`) | a Hugging Face cache variable overrides `HF_HOME` | `unset HF_HUB_CACHE` (or the variable named) and submit again |
| `warning: HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE is set ...` | offline mode: nothing can be downloaded | `unset HF_HUB_OFFLINE TRANSFORMERS_OFFLINE` |
| `warning: HF_TOKEN is not set ...` / `the zkllm tier downloads the gated Llama-2 checkpoints with your Hugging Face token ...` | the token is not exported | `export HF_TOKEN=<token>` |
| `CXX='' names no compiler here: export CXX=<path to g++> ...` | `CXX` is not set or not on this host | `export CXX=$(command -v g++)` as in the rehearsal; never pass `--no-native-ok` |
| `CMake ('cmake') not found here: the cpuv tier builds llama.cpp with it ...` | no CMake for the llama.cpp job | `python3 -m venv ~/cmake && ~/cmake/bin/pip install cmake==3.31.6 && export CMAKE=~/cmake/bin/cmake` (or `module load cmake`, version 3.14 or later) |
| `git not found here: the cpuv tier clones llama.cpp with it ...` | no git on the submitting host (the nodes need none) | load or install git |
| `warning: cloning llama.cpp here failed ...` / `warning: cloning zkLLM here failed ...` | the login node could not reach GitHub; the jobs were submitted and will try to clone on their node | if the nodes have no network or no git, clone on the login node as in the llama.cpp and zkLLM sections below, before those jobs start |
| `make not found here` / `gcc not found here` / `g++ not found here: the zkllm tier builds zkLLM with it` | build tools missing on this host | load them (`module load gcc/11` or similar; gcc 12 or older) |
| `no CUDA compiler (nvcc) here: the zkllm tier builds zkLLM with it; load CUDA 12.1 ...` | no CUDA toolkit | `module load cuda/12.1` (see `module avail cuda`), or add `--cuda-home <folder with bin/nvcc>` to the `zkllm` line |
| `... is not a Python 3.8 to 3.12 interpreter: the zkllm tier creates zkLLM's environment (torch 2.3.1) from it (--zkllm-python <path>)` | the interpreter for zkLLM's environment has the wrong version | add `--zkllm-python <a python3.8 to 3.12>` to the `zkllm` line |
| `<folder> has N GB free: the zkllm tier's zkLLM checkout and environment need 10 GB (--zkllm-home <folder>)` | not enough space for zkLLM's home | add `--zkllm-home <folder with 10 GB free>` to the `zkllm` line |
| `--zkllm-scratch <dir> has N GB free; zkLLM's weights and commitments of Llama-2-13B need 58 GB` | the scratch folder you gave is too small | give a folder with 58 GB free, or leave the option out |
| `... cannot import sentencepiece and google.protobuf, which the qllama and qcheck tiers' OpenLLaMA-7B ...` | the line was submitted without `--llama-licence` (it was not submitted through `R`) | use `R` (Sec. 3), which passes `--llama-licence`; do not install anything into `.venv` |
| `.venv/bin/python is not an interpreter (--python <path>)` | not run from the clone's root, or the `.venv` is missing | `cd` to the clone's root |
| `the largest job asks for 256 GB and the partition's largest node has N GB: pass --mem-max N` | the partition has no node with 256 GB | do not pass `--mem-max` or `--no-bigmem`; send us the message |
| `no GPU requested: export SBATCH_GRES=gpu:<type>:1 or pass --gres ...` | the `R` function was not used | use `R` (Sec. 3), which passes `--gres gpu:l40s:1` |
| `the bench tier's image cells need: ... nothing submitted` | an image model or dataset is missing | check `PVI_CIFAR_ROOT`, `PVI_IMAGENET_ROOT` and `code/artifacts/models/cnn/*.pt` (the six models you trained); they are in the clone's ignored files, so a fresh clone lacks them |
| `run this from the repository root` | wrong folder | `cd` to the clone's root |

## 2. sbatch and the queue

| What you see | Cause | Fix |
|---|---|---|
| a dry run's last line is `# if it ends at its time limit, submit the same command again: ...` | the job limits of that submission add up to more than 24 h (every whole-node line but `zkllm`, `quality` and `qllama`) | nothing: the limits are caps, not expected times; the expected hours of Sec. 4 are under 24 h except `qllama` (and possibly `bench`, `cpux` and `quality`) |
| `sbatch: error: Invalid generic resource (gres) specification` | the partition calls the GPU differently | `sinfo -p "$SBATCH_PARTITION" -o '%N %G'`; send us the output (the name is part of every line) |
| `sbatch: error: ... Requested node configuration is not available` | memory, CPUs or exclusivity that the partition does not offer | send us the message and the `sbatch ...` line of the dry run |
| an allocation stays pending with reason `(Dependency)` | another job of the same name (an `sp-` job from an earlier run) is queued or running: each job waits for its name to be free (`--dependency=singleton`) | `squeue -u $USER -o '%i %j %k %T'`; cancel the old job (its third column is not `final`, or it is from before this run) with `scancel <id>` |
| pending with `(Resources)`, `(Priority)`, `(ReqNodeNotAvail)`, `(QOS...)` | waiting for a node | nothing; whole-node allocations may wait many hours |
| `skip sp-node-...: already queued (job N)` | the same line is already in the queue | nothing |

## 3. In an allocation's log (`logs/final/node-<tiers>-<id>.log`)

| Message | Cause | Fix |
|---|---|---|
| `wrong node: GPU <x> / CPU <y>, expected GPU 'L40S' / CPU 'EPYC 9334' (host <host>; ...)`, then `run.sh: the node check failed (exit status 86): no job run on <host>` | the allocation landed on a node of another model | submit the command that the next lines print (`--exclude-nodes <host>,t-806` added); tell us the host |
| `run.sh: ...` lines and `run.sh: nothing submitted` at the start of the log | the allocation repeats the checks of Sec. 1 on the node, where something is missing (typically `nvcc`, `gcc` or `cmake` loaded only on the login node in a way that does not reach the job) | fix as in Sec. 1, so that the tool is on the `PATH` of the shell that submits (`module load` before `R`); submit again |
| `# summary ...` with `FAILED  sp-... (exit status 124: its time limit)` | one job reached its own time limit | submit the same line again (once); if the same job reaches its limit again, send us its log: it has no resume within the job |
| `run.sh: stopped by SIGTERM`, then `FAILED ... (exit status 143: stopped by a signal)` | the allocation's 24-hour limit or a pre-emption | if SLURM requeued it (`squeue` shows it again), nothing; otherwise submit the same line again |
| `smoke did not end in SMOKE OK (...): stopping` (`smoke` line only) | a test, the native kernels or the int8 check failed on the node | look in `logs/final/smoke_pytest-<host>-<job id>.txt` and the end of `logs/final/sp-smoke-*.out`: if the message is in Sec. 4, fix it and submit the `smoke` line again; otherwise send us both files and the `.xml` next to the `.txt` |
| the allocation's log ends without a summary, and `squeue` no longer shows it | the allocation was killed outright (node failure, `scancel`) | submit the same line again |

## 4. In a job's log (`logs/final/sp-<cell>-<id>.out`)

### The first lines: node, environment, telemetry, pinning

| Message | Cause | Fix |
|---|---|---|
| `env snapshot: <folder> (written earlier)` | the allocation's first job already wrote the node's environment | nothing |
| `env snapshot: <error>` or an error in a file under `logs/final/env/<host>-<job id>/` | a command of the environment capture is missing or failed on the node | nothing: it never fails a job; please mention it when you report |
| `telemetry: <error>` | the GPU sampler could not start (for example `nvidia-smi` failed) | nothing: it never fails a job; please mention it when you report |
| `pinned to CPUs ... (N cores of NUMA node K; ...)`, then `processes ... moved to CPUs ...` | a CPU-timing job pinned itself; the GPU sampler is moved off the timed cores | nothing |
| `not pinned: ...` in a CPU-timing job | the node did not allow pinning | nothing for 8-thread jobs; for a `cpux` 2- or 4-thread job, send us the log after the run (the check of Sec. 6 tells whether its record is valid) |
| `processes ... share the pinned CPUs: the job may use no other` | the job's allocation has no CPU besides the pinned ones | send us the log after the run; do not rerun it on your own |

### Native CPU kernels, Triton, the GPU

| Message | Cause | Fix |
|---|---|---|
| `--require-native: the native CPU kernels are not available here (they need Linux, a C++ compiler in CXX or on the PATH, and --native)`, or in the smoke `native kernels unavailable with CXX=...` | the compiler in `CXX` is not reachable on the node, or failed there | `CXX` must be a path that the compute nodes see (as in the rehearsal); check it on a node: `sbatch --gres=gpu:l40s:1 --exclude=t-806 -c 8 --mem=8G --time=0:10:00 -o cxx-%j.out --wrap "$CXX --version"`; fix it and submit the line again; never pass `--no-native-ok` |
| `--require-triton: Triton does not run on a GPU here`, or `fatal error: Python.h: No such file or directory` in a traceback | Triton's launcher cannot find the Python headers on the node | `export PVI_PY_INCLUDE=<the headers folder of the rehearsal>` (it must be visible on the nodes) and submit again |
| a traceback inside `triton` that names a file under `logs/final/triton/` (missing, empty or unreadable cache entry), typically after a pre-emption | a half-written Triton compile cache entry | when no allocation of the run is running, `mv logs/final/triton logs/final.triton-broken` and submit the failed lines again (it is a compile cache; the timed queries follow a warm-up, so the numbers do not change); if the error comes back, send us the log |
| `int8 GEMMs are not exact on this GPU` | the GPU's int8 path disagrees with the integer reference | send us the log; do not run again |
| `CUDA error: ...`, `no CUDA-capable device`, `torch sees no GPU` | a GPU or driver fault on that node | add the host to `--exclude-nodes`, submit the line again, and tell us the host |

### Memory

| Message | Cause | Fix |
|---|---|---|
| `"skipped": "out of memory"` in `serve`'s records | expected for the largest joint batches of the 7B and 13B models | nothing |
| `torch.OutOfMemoryError: CUDA out of memory` in `sp-gen-decode` | it needs about 13.5 GB of GPU memory for the Llama-2-7B checkpoint in fp16; on a whole node this only fails if another process holds the GPU | send us the log and the job's `logs/final/telemetry/sp-gen-decode-<id>.apps.txt` (the GPU's processes) |
| `torch.OutOfMemoryError: CUDA out of memory` in any other job (for example a Llama-2-13B job of `qllama`) | GPU memory | send us the log. Do not add `--ref-device cpu` or other options yourself: the paper states the reference's device and precision |
| exit status 137, `Killed`, or `slurmstepd: error: Detected 1 oom_kill event` | host memory (a real-weight job holds its checkpoint in fp32, about 27 GB for a 7B model and 52 GB for Llama-2-13B, up to about 90 GB in a 7B re-execution job; it asks for at least 128 GB, 256 GB for Llama-2-13B) | send us the log and the output of `sacct -j <allocation id> -o JobID,MaxRSS,ReqMem,NodeList` |

### Hugging Face

| Message | Cause | Fix |
|---|---|---|
| `GatedRepoError: 403 Client Error ... Access to model meta-llama/Llama-2-13b-hf is restricted` (or `-7b-hf`) | your account has no approved access to that repository | request access on the repository's page on huggingface.co (each of the two has its own gate); when it is approved, check with the one-file test of Sec. 2 and submit the line again |
| `401 Client Error: Unauthorized` / `Invalid user token` | `HF_TOKEN` is missing, wrong or expired in the job's environment | export a valid token, cancel the queued allocation if any, and submit again |
| `Connection error`, `ReadTimeout`, `MaxRetryError` while downloading | a network problem during a download | submit the same line again; downloads resume |
| `OSError: [Errno 28] No space left on device` or `Disk quota exceeded` under `$HF_HOME` | `HF_HOME` is full | free space there or move it (Sec. 5 below) |

Every cost line now reads checkpoints (OPT-125M, OPT-1.3B, OPT-6.7B, Qwen3-4B and the two Llama-2 models), so a
download error can come from any of them; the first job of each checkpoint downloads it, the later ones read
`HF_HOME`.

### The real-weight cells (`--real`)

The cost cells of a model with a checkpoint run on it (FINAL_RUN.md, "Real weights are now the default"). A real-weight
job loads and calibrates its checkpoint before it times anything, and refuses a cell it cannot build before it loads
anything. None of these refusals occurs in the run's jobs; one means that a line, the checkpoint or `HF_HOME`
changed.

| Message | Cause | Fix |
|---|---|---|
| `FAILED  sp-... (exit status 124: its time limit)` for a job whose line has `--real` | the checkpoint's download, fp32 load and calibration (10 to 30 minutes for a 7B model, twice that for Llama-2-13B) took longer than planned | submit the same line again once (the download is kept); if the same job reaches its limit again, send us its log |
| `--real: <model>'s shapes have no integer builder of a real checkpoint (...); its cells take random weights only` | a line passed `--real` for GPT-2 or another model without a builder | send us the line and the log; never change a line |
| `--real: prompts of N tokens exceed <model>'s trained context of M positions`, or `--real <checkpoint>: prompts of N tokens ... behind its P-token fixed prefix exceed its trained context of M positions` | a cell beyond the checkpoint's trained context (the run keeps those on random weights) | send us the line and the log |
| `--real <checkpoint> is not of --model <model>'s shapes: ...` | the downloaded checkpoint is not the expected model (a changed repository, or an `HF_HOME` that holds another revision) | send us the log and `ls $HF_HOME/hub` |
| `--layers N: <model> has M blocks`, `--real measures the checkpoint's full graph ...`, `--prob-max is for random-weight decoders: a checkpoint's recipe fixes its scale` or `--recipe, --smooth and --calib-seq are the options of a real checkpoint: pass --real` | a changed line | send us the line and the log |

### The cpux jobs (`cpux` line)

| Message | Cause | Fix |
|---|---|---|
| `FAILED  sp-x-plans-llama2-7b-2k-<plan> (exit status 124: its time limit)` | a plan job encodes Llama-2-7B's commitment afresh, with no setup cache, by design (the encoding time is the measured quantity); R64c is the longest (limit 5 h) | submit the `cpux` line again once; if the same plan reaches its limit again, send us its log. Never add `--setup-cache` |
| (after the run) `CHECK FAILED sp-x-...out:<line>: <N> threads on <M> allowed CPUs` | a 2- or 4-thread record ran on fewer CPUs than its threads | send us the report and the job's log; do not rerun on your own |

### llama.cpp (`sp-c-llamacpp-llama2-7b`, `cpuv` line)

The job times 64, 512 and 2,048 tokens. It needs the same `$TMPDIR` space as before (about 21 GB).

| Message | Cause | Fix |
|---|---|---|
| `FAILED: llama.cpp clone: ... exited with status ...` | the node cannot reach GitHub | on the login node: `git clone --depth 1 --branch v0.6.0 https://github.com/ggml-org/llama.cpp.git logs/final/llamacpp/src`, then submit the `cpuv` line again |
| `FAILED: llama.cpp clone: ... is at commit ..., not v0.6.0's d812350...` | a wrong or partial checkout in `logs/final/llamacpp/src` | `rm -rf logs/final/llamacpp/src`, clone again as above, submit the `cpuv` line again |
| `FAILED: llama.cpp clone: --git 'git' not found` / `FAILED: llama.cpp build: --cmake 'cmake' not found` | git or CMake is not on the node's `PATH` | `export CMAKE=<path to cmake 3.14+ that the nodes see>` (or load the module before `R`), submit again |
| `FAILED: llama.cpp build: ... exited with status ...` | CMake or the compiler failed | send us `logs/final/llamacpp/build/configure.log` and `logs/final/llamacpp/build/build-*.log` with the job's log |
| `FAILED: llama.cpp disk: the model files need N GB under <dir>, which has M GB free (--tmp-dir)` | `$TMPDIR` on the node has less than about 21 GB | `export TMPDIR=<folder on a disk the nodes see, 25 GB free>`, then submit the `cpuv` line again |
| `FAILED: llama.cpp gguf-py: ...` | the clone's `gguf-py` cannot be imported | send us the log |

### zkLLM (`zkllm` line: `sp-zk-build`, `sp-zk-llama2-7b`, `sp-zk-llama2-13b`, `sp-zk-llama2-7b-short`, `sp-zk-llama2-13b-short`)

The 2,048-token jobs prove one length each; each `-short` job runs zkLLM's setup once and then proves 64, 512 and
1,024 tokens (setup about 25 min for 7B and 87 min for 13B). Each proof job needs 58 GB of scratch space for
Llama-2-13B (32 GB for 7B) for itself.

| Message | Cause | Fix |
|---|---|---|
| `Input or output file does not exist` in each layer's first skip connection | expected: zkLLM's demo writes no attention output | nothing |
| `zkLLM's demo did not prove Llama-2-<N>B at <T> tokens: recorded (ok false); the other lengths go on` (`-short` jobs; also possible at 64 or 512 tokens) | zkLLM's lookup arguments need sizes that short prompts may not give; the length is recorded as not proved, with zkLLM's error lines in `logs/final/zkllm/llama2-<N>b-T<T>/errors.txt` | nothing, as long as the job ends successfully: it is a result, not a failure. The job's last line is then `zkLLM's demo proved Llama-2-<N>B at <k> of 3 lengths and not at ... tokens (recorded; --allow-failed-lengths)` |
| a `-short` job FAILED in the summary after `did not prove` lines for all three lengths | no length was proved | send us the job's log and the three `errors.txt` files; do not rerun it on your own |
| `run_zkllm.sh: no CUDA toolkit: load one (e.g. module load cuda/12.1) ...` | nvcc not reachable in the job | `module load cuda/12.1` before `R`, or `--cuda-home <folder>` on the `zkllm` line; submit again |
| `run_zkllm.sh: zkLLM's build failed (CUDA 12.1 needs gcc 12 or older as nvcc's host compiler ...)` | the default gcc is too new for CUDA 12.1 | load gcc 12 or older (`module avail gcc`) before `R` and submit the `zkllm` line again; if the build fails again, first run `make -C logs/final/zkllm/home/zkllm clean` (your `--zkllm-home` folder if you gave one) |
| `run_zkllm.sh: git not found on this host, and ... does not exist` | `run.sh` could not clone zkLLM on the submitting host (it printed a warning when you submitted) | clone it on the login node as in the next row, then submit again |
| `run_zkllm.sh: make not found` / `/usr/bin/time not found on this node ...` | a tool is missing on the GPU node | `module load` it before `R`; for `/usr/bin/time` (GNU time), send us the message |
| `run_zkllm.sh: cloning https://github.com/jvhs0706/zkllm-ccs2024.git at 993311e failed (no network on the node? ...)` | the node cannot reach GitHub | on the login node: `git clone https://github.com/jvhs0706/zkllm-ccs2024.git logs/final/zkllm/home/zkllm && git -C logs/final/zkllm/home/zkllm checkout 993311e` (use your `--zkllm-home` folder if you gave one), then submit again |
| `run_zkllm.sh: installing .../requirements.txt into .../env failed` | no network for pip on the node (torch's CUDA wheels) | build once on a GPU node that has network access: `sbatch --gres=gpu:l40s:1 --exclude=t-806 -c 8 --mem=32G --time=2:00:00 -o logs/final/zk-manual-build-%j.out --wrap "bash code/experiments/comparison/zkllm/run_zkllm.sh build --home $PWD/logs/final/zkllm/home --python $PWD/.venv/bin/python"` (add `-p <partition with network>` if needed); it ends with `ZKLLM_BUILD_OK`; then submit the `zkllm` line again, whose build job reuses the home |
| `run_zkllm.sh: ... is at <sha>, not at 993311e` / `... has changed tracked files; the demo is run unchanged` | the checkout in the zkLLM home is not the unchanged demo | use the default home (leave out `--zkllm-home`), or reset that checkout as the message says |
| `run_zkllm.sh: ... was built for arch: sm_XX, this node needs sm_89: rebuild it (...)` | a build made for another GPU | run the command that the message prints (it deletes the binaries and `BUILD.txt`), or give another `--zkllm-home`; submit again |
| `run_zkllm.sh: the downloads failed (the Llama-2 repositories are gated: export HF_TOKEN=...)` | token or gate access | as under Hugging Face above |
| `run_zkllm.sh: zkLLM's environment cannot use the GPU` | zkLLM's torch does not see the GPU | send us the log and `logs/final/zkllm/home/BUILD.txt` |
| `run_zkllm.sh: --scratch <dir> has N GB free; Llama-2-13B needs 58 GB` / `no scratch folder with 58 GB free: pass --scratch DIR` | the node's `$TMPDIR` and `/tmp` are too small | add `--zkllm-scratch <folder with 58 GB free that the node sees>` to the `zkllm` line; submit again |
| `run_zkllm.sh: ... holds no built zkLLM with its timing shims and env/: run ...` (prove jobs) | the build job failed | fix the build job's message first; submitting the `zkllm` line again reruns the build and then the proofs |
| `run_zkllm.sh: cannot read the GPU's compute capability with nvidia-smi` | `nvidia-smi` is missing or fails on the node | exclude the host and submit again; tell us the host |

To re-run only the failed zkLLM jobs, submit the `zkllm` line again. The zkLLM jobs that already finished are
skipped.

## 5. Disk

| Message | Cause | Fix |
|---|---|---|
| `No space left on device` or `Disk quota exceeded` (Errno 28 or 122) in any log | the disk named in the traceback is full | check which one: `df -h $HF_HOME ${TMPDIR:-/tmp} .` and your quota (`quota -s`); free space on it (old `logs/*` folders of earlier runs, the archived `logs/final.before-...`); for `TMPDIR` set another folder; then submit the failed lines again |
| jobs fail at once with exit status 1 and an empty or truncated log | often a full disk or quota on the clone's disk | as above |

Never delete anything in `logs/final/` except what this file names. The `triton` and `native` folders are caches,
and `zkllm/home` can be rebuilt. The `sp-*.out` logs, `done/`, `attack/`, `env/`, `telemetry/`, the
`smoke_pytest-*` files and the `zkllm/llama2-*` folders hold the records. The telemetry takes a few tens of MB in all.

## 6. After the run (`numbers --check`)

| Report | Cause | Fix |
|---|---|---|
| `COMPLETE: every table cell and every macro is in the run` and `status 0` | everything is there | archive and copy (Sec. 6) |
| the report's first line ends with `main cells on random weights` or `(without the Llama-2 licence)` | some lines were submitted with `--random-weights`, or without `R` (no `--llama-licence`): the paper's main cells need the real Llama-2 checkpoints | send us the report; do not rerun on your own |
| `MISSING: ...` cells of a model with a checkpoint (for example `tab:options` or `tab:eval-llm` cells of Llama-2-7B) while that model's random-weight records exist | a main cell is read only from its real-weight records, never from random ones: its real-weight job did not finish | as for any missing cell below |
| `WARNING env: the nodes differ in driver: 535.183.01 (...); 595.99.02 (...)` | the run used nodes with both drivers | nothing: expected and harmless |
| any other `WARNING env: the nodes differ in <field>` line (for example the power limit, clocks, kernel or governors) | the nodes are configured differently | nothing; please report the line to us |
| `WARNING env: <file>: FAILED (MODELS.sha256)` | a stored network in the clone differs from its recorded hash | send us the report; do not rerun on your own |
| `WARNING telemetry: sp-...: the GPU was slowed by ...` / `... other processes on the GPU when the job started: ...` | a timed job's GPU was throttled, or shared | nothing; please report the lines to us (we decide whether the job is run again) |
| `MISSING: N table cells and M macros ...`, and some allocation's last summary has `FAILED` lines or the `--expect` part names jobs without results | jobs did not finish | submit those lines again; when they are done, repeat the check |
| `MISSING: ...` although every allocation's summary ends with `every job of this submission has ended successfully` | something we did not foresee | send us `numbers_check.txt` (in the archive) |
| `CHECK FAILED ... the smoke did not end in SMOKE OK` / `... int8 GEMMs are not exact on this GPU` | the smoke job failed | as in Sec. 3 (smoke) |
| `CHECK FAILED sp-x-...: <N> threads on <M> allowed CPUs` | a `cpux` record ran on fewer CPUs than its threads | send us the archive; do not rerun on your own (we will say which marker to delete) |
| `CHECK FAILED platform: the jobs name 2 GPU models ...` (or CPU models) | a job ran on another model | send us the report; do not rerun on your own (some jobs keep earlier results, so we will say exactly which markers to delete) |
| any other `CHECK FAILED` or `PARSE` line | a failed correctness check or an unreadable record | send us the archive |
| `machines:` line with n-801 to n-805 and drivers 535 and 595 | several nodes and drivers | fine |

## 7. What to send us

For anything that is not in this file, please send:

* the allocation's log `logs/final/node-<tiers>-<id>.log` (the whole file; its end has the summary);
* the failing job's log `logs/final/sp-<cell>-<id>.out`, which the summary names;
* the node's environment `logs/final/env/<host>-<allocation id>/` and the job's telemetry
  `logs/final/telemetry/sp-<cell>-<id>.*`;
* for the smoke: `logs/final/smoke_pytest-<host>-<job id>.txt` and `.xml`;
* for llama.cpp: `logs/final/llamacpp/build/*.log` if they exist;
* for zkLLM: `logs/final/zkllm/home/BUILD.txt` and, for a prove job, the `env.txt` and `errors.txt` of each of its
  folders `logs/final/zkllm/llama2-<size>b-T<length>/`;
* the output of `git log --oneline -1`, `squeue -u $USER` and `sacct -j <allocation id> -o JobID,JobName,State,ExitCode,Elapsed,MaxRSS,NodeList`;
* the exact line you submitted and anything you changed in the environment.

For example:

```bash
tar czf report-$(date +%m%d-%H%M).tgz logs/final/node-*.log logs/final/sp-<cell>-*.out logs/final/smoke_pytest-* \
    logs/final/env logs/final/telemetry/sp-<cell>-*
cp report-*.tgz $H/return/
```

We will answer with a fix and the lines to submit again. Jobs that have ended successfully are kept and are not run
again.

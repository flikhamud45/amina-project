# L40S machine: the final run

Thank you again for the rehearsal. This file describes the final run of commit `<COMMIT>`. Its records give every
number of the paper, so it runs in full on one platform: every job on an L40S node with an AMD EPYC 9334 host.

There is no smoke run, no rehearsal and no gate before it. You submit the run's fifteen jobs directly (Sec. 4). The
`smoke` line among them is a job of the run, not a test before it. If something small goes wrong,
`TROUBLESHOOTING.md` (next to this file) gives the message, the cause and the fix. If a problem is not listed there,
or its fix does not work, please send us the logs it names. We will send a fix, and you then resubmit only the jobs
that failed.

**The run must use Llama-2.** Every line of the run is submitted through `R` (Sec. 3), which passes
`--llama-licence`; do not submit any line without it. The paper's Llama-2-7B and Llama-2-13B results (the quality
grids and their checks, the real-weight Llama-2-7B cells and zkLLM) come from these jobs; without the option the run
leaves the Llama-2 jobs out and tunes on OpenLLaMA-7B instead, which the paper does not use, so such a run cannot be
used. Before you submit anything, your Hugging Face account needs approved access to both gated repositories,
`meta-llama/Llama-2-7b-hf` and `meta-llama/Llama-2-13b-hf` (each has its own gate; in October your account still got
a 403), and the one-file check of Sec. 2 must print two paths. If access is missing or still pending, do not start
without Llama-2: wait for the approval and let us know.

Expect about a day of waiting in the queue and two to three days of computing. The longest submission, `qllama`,
takes 40 to 48 hours and therefore needs two or three submissions of the same line; `bench` (17.5 to 33 hours) and
`cpux` (13 to 24 hours) may need a second submission of the same line.

## What changed since the rehearsal (`a405361`)

The measured code changed (the decoder, the Llama and OPT model loaders and the v0 query), so every record has to
come from this commit. The six image models that you trained are still valid: their code did not change.

**Real weights are now the default.** Every cost cell of Llama-2-7B, Llama-2-13B, OPT-125M, OPT-1.3B, OPT-6.7B and
Qwen3-4B now runs on the model's Hugging Face checkpoint (`--real`), quantised to int8 with the certified recipes,
wherever its prompt fits the checkpoint's trained context (Llama-2: cells up to 2,048 tokens). GPT-2, the image
models, the 4,096-token cells and the whole long-prompt curve (the `long` submission, the GPU verifier's long cells,
V1's codec curve) keep random weights of the same shapes. You need nothing new for this beyond Sec. 2:

* **Downloads.** The cost jobs read the checkpoints that the quality jobs already download: OPT-125M, OPT-1.3B and
  OPT-6.7B (about 16 GB), Qwen3-4B (8 GB) and, through `--llama-licence`, Llama-2-7B and Llama-2-13B (40 GB), into
  the same `HF_HOME` (100 GB free, as before). The first job of each checkpoint spends some minutes downloading it.
* **Memory and time.** A real-weight job loads its checkpoint in fp32 (about 27 GB for a 7B model, 52 GB for
  Llama-2-13B) and calibrates it on the CPU (about 10 to 30 minutes for a 7B model, twice that for Llama-2-13B). Its
  job limit is one hour longer (two for Llama-2-13B, one more for a re-execution job), and it asks for at least
  128 GB of host memory (256 GB for Llama-2-13B). The L40S nodes have 515 GB, so nothing needs doing.
* **Never pass `--random-weights`.** It restores the previous all-random run, which the paper no longer uses.

**The submissions.** The run is now fifteen submissions (it was thirteen in the previous version of this guide):

* `serve gen batch` is now two whole-node submissions, `serve gen` (14-20 h) and, with `improve`, `improve batch`
  (9-18.5 h).
* `v1 setup improve` is now `v1 setup` (8.5-16.5 h).
* `gpuv zkllm` is now two whole-node submissions, `gpuv` (8.5-14.5 h) and `zkllm` (6-9 h).
* `cpuv`, `long`, `bench`, `real` and `cpux` stay one submission each. `bench` (17.5-33 h) and `cpux` (13-24 h) may
  reach the 24-hour limit; submit the same line again then (Sec. 5).
* The `long` submission has 2 new jobs: the full Llama-2-7B at 2,048 tokens on random weights, against which the long
  prompts' extrapolation is checked (the cost cells at 2,048 tokens now use the checkpoint).

**New options.** None. In particular:

* The full Llama-2-7B at 8,192 tokens (`long` submission) runs by default and needs a node with 256 GB or more. The
  L40S nodes have 515 GB, so nothing needs doing. Do not pass `--no-bigmem`: it would leave these cells out.
* With `--llama-licence` (part of `R`, Sec. 3), no job uses OpenLLaMA-7B, so `sentencepiece` and `protobuf` are
  not needed.

**What the new and changed jobs measure, and what they need from you:**

* **llama.cpp baseline** (`cpuv`). `run.sh` clones llama.cpp's release v0.6.0 into `logs/final/llamacpp/src` on the
  submitting host before it submits; one job then builds `llama-bench` and `llama-quantize` for the node's CPU
  (about 1 GB in `logs/final/llamacpp/`) and times Llama-2-7B's prefill in Q8_0 and bf16 at 64, 512 and 2,048 tokens
  on the verifier's 8 threads. It needs git and network access on the submitting host, CMake 3.14 or later, and about
  21 GB in `$TMPDIR` on the node while it runs (the 512-token cell adds nothing to this). The node needs no git.
* **The CPU verifier beyond 8 threads** (`cpux`, 14 jobs). Llama-2-7B at 2,048 tokens on 2 and 4 threads, on 8
  threads without the AVX-512 attention kernel, Llama-2-13B on 8 threads, and mode C's six candidate commitment plans
  on the same Llama-2-7B queries, one job per plan. These are whole-node CPU timings. The 2- and 4-thread jobs pin
  themselves to 2 and 4 cores; a line `pinned to CPUs ...` near the start of each log names the cores, and the next
  line says that the GPU sampler was moved to the other CPUs. Each plan job encodes Llama-2-7B's commitment afresh,
  with no setup cache, by design: the encoding time is the measured quantity. The longest is the R64c plan (limit
  5 h).
* **Attribution study** (`quality`). The integer OPT-125M's perplexity is attributed to its parts, on the CPU.
* **11-bit cost graphs** (`improve` and the cost tiers). Nothing new is needed from you.
* **Llama-2 recipe selection** (`qllama`). One job per model, split and alpha: 12 jobs of about 4 h each, 40 to 48 h
  in all, on one GPU of a shared node.
* **The Llama recipe beyond its cells** (`qcheck`, 5 jobs). The validation split and three calibration draws, on
  one GPU of a shared node; not timed. With `--llama-licence` it needs nothing new.
* **Generation without a proof** (`gen-decode`, in `serve gen`). It runs the Llama-2-7B and Qwen3-4B checkpoints in
  fp16 (one model per command) and needs about 13.5 GB of GPU memory for the 7B model.
* **zkLLM on the run's GPU** (`zkllm`, 5 jobs, its own submission). zkLLM's public demo (commit `993311e`, unchanged)
  is built once (`sp-zk-build`) and then proves layers 0 and 1 of Llama-2-7B and Llama-2-13B, 3 repetitions each:
  * at 2,048 tokens: `sp-zk-llama2-7b` (limit 2 h) and `sp-zk-llama2-13b` (limit 4 h);
  * at 64, 512 and 1,024 tokens: `sp-zk-llama2-7b-short` (limit 2 h) and `sp-zk-llama2-13b-short` (limit 4 h).
    Each `-short` job runs zkLLM's setup once for its three lengths (about 25 min for 7B, 87 min for 13B). A line
    `zkLLM's demo did not prove Llama-2-<N>B at <T> tokens: recorded` is an expected result, not a failure, as long
    as the job ends successfully. The job fails only if no length was proved.

  The `bench` submission measures our own prover at the same three lengths. The scratch need is unchanged: 58 GB on
  the node for Llama-2-13B (32 GB for 7B), per job.
* **The environment and the GPU's telemetry are captured inside the run.** Every allocation writes its node's
  environment once to `logs/final/env/<host>-<job id>/`, and every job samples its GPU's clocks, power and
  temperature every second to `logs/final/telemetry/` (a few tens of MB in all). The separate `sbatch --wrap`
  environment job of the previous guide is no longer needed.
* **Smoke logs.** The smoke's pytest output is now `logs/final/smoke_pytest-<host>-<job id>.txt`, with its JUnit
  XML next to it (`.xml`).
* **One-GPU checks.** Every job first writes a `node: {...}` line to its log (host, GPU, driver, CPU,
  `CUDA_VISIBLE_DEVICES`). It then checks that the GPU is an L40S and the CPU an EPYC 9334 (`--require-gpu`,
  `--require-cpu`). On another model it stops with `wrong node: ...` and exit status 86 before doing any work. Every
  allocation asks for an L40S (`--gres gpu:l40s:1`), including the five that do not need the node to themselves.
  `--exclude-nodes` keeps all jobs off `t-806`.
* **Failure summary.** Each allocation's log `logs/final/node-<tiers>-<job id>.log` ends with a summary. It lists the
  jobs that ended successfully, the failed ones with their exit status and log, and the command that runs the failed
  and missing jobs again. That command is always the same line you submitted. SLURM signals the allocation 5 minutes
  before its 24-hour limit, so the summary is also written when the time runs out or the job is pre-empted.
* **Coverage report.** `numbers --check` ends with `COMPLETE: ...`, or with `MISSING: N table cells and M macros
  ...` followed by one line per missing item, and exits with status 1 while anything is missing. Its `machines:`
  line names the nodes and drivers that the jobs ran on, and its `environment:` and `telemetry:` lines summarise
  the two captures above.

## 1. Update your clone

```bash
H=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/validation/sp2027/handoff_l40s
(cd $H && sha256sum -c SHA256SUMS)                    # every line: OK
git -C <your clone> fetch $H/pvi-sp2027-artifact-<COMMIT>.bundle sp2027-artifact
git -C <your clone> checkout -B sp2027-artifact FETCH_HEAD
git -C <your clone> log --oneline -1                  # <COMMIT>
git -C <your clone> status --short --untracked-files=no   # prints nothing
```

Please work in `bash`. If your login shell is tcsh, type `bash` first. All the commands below are run from the
clone's root.

Start from an empty run folder. Records or `done` markers left in `logs/final/` from an earlier commit would be
mixed into this run, or would make it skip jobs:

```bash
[ -e logs/final ] && mv logs/final logs/final.before-<COMMIT>
squeue -u $USER          # no job whose name starts with sp- should be listed
```

If old `sp-` jobs are still pending, cancel them. They have the same names as this run's jobs and would hold them
back (`scancel <job id>`).

## 2. Environment

Source your environment file from the rehearsal. It sets `SBATCH_PARTITION`, `SBATCH_GRES=gpu:l40s:1`, `HF_HOME`,
`HF_TOKEN`, `TMPDIR`, `PVI_PY_INCLUDE`, `CXX`, `PVI_CIFAR_ROOT` and `PVI_IMAGENET_ROOT`. Then:

```bash
unset HF_HUB_CACHE                     # it would override HF_HOME
env | grep -E 'HF_HUB_CACHE|TRANSFORMERS_CACHE|HF_DATASETS_CACHE'   # prints nothing
df -h $HF_HOME                         # about 100 GB free
${CMAKE:-cmake} --version              # 3.14 or later; git --version
```

Keep excluding `t-806`. The submission lines below pass `--exclude-nodes t-806`, which replaces any exclusion in
your environment file.

**CMake** (for the llama.cpp job). If `cmake` is missing or older than 3.14, install it into a separate environment.
This leaves the run's `.venv` unchanged:

```bash
python3 -m venv ~/cmake && ~/cmake/bin/pip install cmake==3.31.6 && export CMAKE=~/cmake/bin/cmake
```

**sentencepiece and protobuf** are not needed: with `--llama-licence`, no job uses OpenLLaMA-7B. The `.venv` stays
as it is.

**Hugging Face.** Your token needs approved access to both gated repositories, `meta-llama/Llama-2-7b-hf` and
`meta-llama/Llama-2-13b-hf`. This one-file check runs on the login node and starts no job:

```bash
for m in 7b 13b; do .venv/bin/python -c "from huggingface_hub import hf_hub_download as d; \
print(d('meta-llama/Llama-2-$m-hf', 'config.json'))"; done      # two paths; a 401/403 error means no access
```

**zkLLM** (the `zkllm` submission):

```bash
module load cuda/12.1          # or the cluster's CUDA 12.1 module (module avail cuda); sbatch passes PATH on
nvcc --version                 # release 12.1
gcc --version; g++ --version   # 12 or older for CUDA 12.1 (else load an older gcc module); make --version
```

* If the CUDA toolkit is in a folder that is not on the `PATH`, add `--cuda-home <CUDA folder>` (the folder that
  holds `bin/nvcc`) to the `zkllm` line in Sec. 4.
* zkLLM's checkout and its own Python environment go to `logs/final/zkllm/home` (about 10 GB). If your quota on the
  clone's disk is too small for that, add `--zkllm-home <folder with 10 GB free>` to the `zkllm` line.
* If you still have the zkLLM checkout from which the stored L40S runs were made, you can pass its parent folder as
  `--zkllm-home <folder>`. That folder must hold the checkout as `<folder>/zkllm`, at commit `993311e` with no
  tracked file changed, and built for sm_89. The build job then reuses the checkout and adds the timing shims and
  an environment of its own (`<folder>/env`). Without this option, `run.sh` clones zkLLM on the submitting host
  before it submits, and the build job builds it (the node needs no git).
* zkLLM's environment is created from `.venv/bin/python` (Python 3.12, which is supported). The build job downloads
  torch's CUDA wheels and the Llama-2 checkpoints, so it needs network access on the node.
* Each proof job needs 58 GB of free scratch space on the node for Llama-2-13B (32 GB for 7B). It takes the first of
  `$TMPDIR`, `/tmp` and the zkLLM home that has that space, and deletes it at the end. If none has it, add
  `--zkllm-scratch <folder with 58 GB free>` to the `zkllm` line.

**Disk, in total:** `HF_HOME` 100 GB, shared by every tier (each checkpoint goes to `$HF_HOME/hub` once; the cost
tiers read the same OPT, Qwen3-4B and Llama-2 checkpoints as the quality tiers). On
each node, `$TMPDIR` needs 10 GB, plus 21 GB during the llama.cpp job and 58 GB during a Llama-2-13B zkLLM job. In
the clone, `logs/final` needs a few GB (the llama.cpp build alone is about 1 GB; the telemetry a few tens of MB),
plus the zkLLM home unless it is elsewhere.

## 3. Check every submission with a dry run

Define the submission command once. It is a function rather than a variable, so that `"EPYC 9334"` stays one
argument:

```bash
R() { bash code/experiments/package/run.sh --platform final --llama-licence --gres gpu:l40s:1 \
      --exclude-nodes t-806 --require-gpu L40S --require-cpu "EPYC 9334" "$@"; }
```

`--require-gpu L40S` and `--require-cpu "EPYC 9334"` are the defaults. They are written out here so that the
platform shows in every command.

Run each dry run below. A dry run submits nothing. It should print no line starting with `run.sh:` (such a line
names a missing setting; see `TROUBLESHOOTING.md`) and no `warning:` line. The first line is always:

```
# node: GPU 'L40S', CPU 'EPYC 9334' (--require-gpu, --require-cpu); --gres=gpu:l40s:1; --exclude=t-806
```

The second line is the `sbatch ...` line of the allocation. The third line is the `# one job:` line in the table.
When the job limits add up to more than 24 h, a fourth and last line follows:

```
# if it ends at its time limit, submit the same command again: it runs only the jobs that have not ended successfully
```

The table gives each dry run's last line: "(4th line)" means the line above, after the `# one job:` line shown.

| Dry run | Expected `# one job:` line | Last line |
|---|---|---|
| `R --dry-run --one-job cpuv` | `# one job: cpuv: 15 jobs, job limits 44 h, time limit 24 h` | (4th line) |
| `R --dry-run --one-job long` | `# one job: long: 23 jobs, job limits 57 h, time limit 24 h` | (4th line) |
| `R --dry-run --one-job bench` | `# one job: bench: 47 jobs, job limits 91 h, time limit 24 h` | (4th line) |
| `R --dry-run --one-job real` | `# one job: real: 28 jobs, job limits 56 h, time limit 24 h` | (4th line) |
| `R --dry-run --one-job serve gen` | `# one job: serve gen: 10 jobs, job limits 45 h, time limit 24 h` | (4th line) |
| `R --dry-run --one-job v1 setup` | `# one job: v1 setup: 19 jobs, job limits 44 h, time limit 24 h` | (4th line) |
| `R --dry-run --one-job improve batch` | `# one job: improve batch: 23 jobs, job limits 44 h, time limit 24 h` | (4th line) |
| `R --dry-run --one-job cpux` | `# one job: cpux: 14 jobs, job limits 53 h, time limit 24 h` | (4th line) |
| `R --dry-run --one-job gpuv` | `# one job: gpuv: 17 jobs, job limits 38 h, time limit 24 h` | (4th line) |
| `R --dry-run --one-job zkllm` | `# one job: zkllm: 5 jobs, job limits 14 h, time limit 14 h` | the `# one job:` line |
| `R --dry-run --one-job --no-exclusive quality` | `# one job: quality: 18 jobs, job limits 56 h, time limit 24 h` | (4th line) |
| `R --dry-run --one-job --no-exclusive qllama` | `# one job: qllama: 12 jobs, job limits 78 h, time limit 24 h` | (4th line) |
| `R --dry-run --one-job --no-exclusive qcheck` | `# one job: qcheck: 5 jobs, job limits 20 h, time limit 20 h` | the `# one job:` line |
| `R --dry-run --one-job --no-exclusive tv attack` | `# one job: tv attack: 8 jobs, job limits 24 h, time limit 24 h` | the `# one job:` line |
| `R --dry-run --one-job --no-exclusive smoke` | `# one job: smoke: 2 jobs, job limits 5 h, time limit 5 h` | the `# one job:` line |

The job limits are caps for each job, not expected times: a job ends when its work is done. That the limits of a
submission add up to more than 24 h does not mean it needs more than one allocation. The expected hours in Sec. 4
are what to plan with; only `qllama`, and possibly `bench`, `cpux` and `quality`, need a second submission of the
same line.

Together the fifteen submissions hold 246 jobs with 669 h of job limits. That is the whole run:
`R --dry-run --optional all` ends with `# smoke cpuv cpux v1 serve bench gpuv gen batch real setup improve quality
qllama qcheck tv attack long zkllm: 246 jobs, job limits 669 h`. Every cost line of that dry run of a checkpoint's
model up to 2,048 tokens carries `--real <checkpoint>` (e.g. `--model llama2-7b --real meta-llama/Llama-2-7b-hf`).

## 4. Start the run

Submit all fifteen lines from the repository root on the login node, in the same shell as Sec. 2 and 3:

```bash
R --one-job cpuv                            # whole node; expected 11.5-19 h once it starts
R --one-job long                            # whole node; 11-18 h
R --one-job bench                           # whole node; 17.5-33 h (may need a second submission)
R --one-job real                            # whole node; 11-16 h
R --one-job serve gen                       # whole node; 14-20 h
R --one-job v1 setup                        # whole node; 8.5-16.5 h
R --one-job improve batch                   # whole node; 9-18.5 h
R --one-job cpux                            # whole node; 13-24 h (may need a second submission)
R --one-job gpuv                            # whole node; 8.5-14.5 h
R --one-job zkllm                           # whole node; 6-9 h (add --cuda-home / --zkllm-home / --zkllm-scratch if Sec. 2 says so)
R --one-job --no-exclusive quality          # one GPU of a shared node; 16-24 h (may need a second submission)
R --one-job --no-exclusive qllama           # one GPU of a shared node; 40-48 h (two or three submissions)
R --one-job --no-exclusive qcheck           # one GPU of a shared node; 9-13 h
R --one-job --no-exclusive tv attack        # one GPU of a shared node; 9-12 h
R --one-job --no-exclusive smoke            # one GPU of a shared node; 1.5-2 h (the test suite and the xcheck proofs)
squeue -u $USER
```

Each line submits one SLURM job named `sp-node-<tiers>` and prints `submitted sp-node-<tiers>: job <id> (its log:
logs/final/node-<tiers>-<id>.log)`.

* The first ten jobs time the system, so each waits for a whole L40S node. They may run at the same time on
  different nodes.
* The last five time nothing and share a node.
* The `smoke` line is one job of the final run, not a test before it. It gates nothing. Its test count goes into the
  paper's text (`\SmokeTests`), and its exactness tests check that the GPU agrees with the CPU on this platform.

Your rehearsal ran on n-802 to n-805. n-805 has NVIDIA driver 595.99.02 and the others 535.183.01. Both drivers are
fine, because the records name each job's node and driver.

## 5. While it runs

* **Job logs.** Each job writes `logs/final/sp-<cell>-<id>.out`. Its first lines are `host ... git ...`,
  `node: {...}` and `env snapshot: ...`; the CPU-timing jobs then print `pinned to CPUs ...`. A job that ends
  successfully leaves a marker in `logs/final/done/`. The allocation's log `logs/final/node-<tiers>-<id>.log` names
  each job when it starts or is skipped, and ends with the summary:

  ```
  # summary (<host>, platform final, tiers <tiers>): N jobs ended successfully, M failed, K had ended successfully in an earlier run
  #   ok      sp-...  logs/final/sp-...out
  #   FAILED  sp-... (exit status 124: its time limit)  logs/final/sp-...out
  # to run the failed and missing jobs again (those that ended successfully are skipped), submit the same command:
  #   bash code/experiments/package/run.sh --one-job --platform final ...
  ```

  If nothing failed, the summary's last line is `# every job of this submission has ended successfully`. The
  allocation exits with status 1 if any of its jobs failed.
* **The 24-hour limit.** If an allocation ends at its limit, or one of its jobs failed, submit the same line again
  once it has left the queue. This is expected for `qllama` (two or three times) and possible for `bench`, `cpux`
  and `quality`. The
  new allocation skips the jobs that have markers in `logs/final/done/` and runs the rest. The job that was running
  when the time ran out shows as `exit status 143: stopped by a signal`. It starts from the beginning in the next
  allocation.
* **Pre-emption.** The partition requeues a pre-empted allocation by itself (`--requeue`). It appends to the same
  log, skips what had finished, and runs the rest. Nothing needs doing.
* **Wrong node.** If a log shows `wrong node: GPU ... / CPU ...` (exit status 86), the allocation ran nothing. The log
  prints the command to submit again with that node excluded. Please submit that command and tell us the node's
  name.
* **A failed job.** Look up its exit status and log in the summary, and look for the message in
  `TROUBLESHOOTING.md`. If the fix there is small (a setting, a module, a full disk), apply it and submit the same
  line again: only the failed jobs run. A job with `A CORRECTNESS CHECK FAILED`, or a line starting with `FAILED:`
  that `TROUBLESHOOTING.md` does not list, is a real problem. Please send us its log and do not run it again.
* **Expected messages.**
  * In `serve`, the largest joint batches of the 7B and 13B models are recorded as `"skipped": "out of memory"`.
  * The first job of each real checkpoint spends some minutes downloading it, and every real-weight job loads and
    calibrates its checkpoint (10 to 30 minutes for a 7B model) before its first timed query.
  * In `zkllm`, each first skip connection's log says `Input or output file does not exist`, and `sp-zk-build` ends
    with `ZKLLM_BUILD_OK`. The two `-short` jobs may log `zkLLM's demo did not prove Llama-2-<N>B at <T> tokens:
    recorded` and end with `zkLLM's demo proved Llama-2-<N>B at <k> of 3 lengths ... (recorded;
    --allow-failed-lengths)`. Both are results, not failures, as long as the job ends successfully.
  * In the 2- and 4-thread `cpux` jobs, `processes ... moved to CPUs ...` right after `pinned to CPUs ...`: the
    GPU sampler is kept off the timed cores.
* **Start times.** Please note when each of the fifteen jobs started. Their queue waits help us plan.

## 6. When all fifteen have finished

`squeue -u $USER` should list no `sp-` job, and every allocation's summary should end with `# every job of this
submission has ended successfully`. From the repository root:

**The machine's description.** Every allocation has written its node's environment already; nothing needs to be
submitted:

```bash
ls logs/final/env/                     # one <host>-<job id> folder per allocation, inputs.sha256, models-check.txt
grep -h '"gpu"\|"cpu"\|"driver"' logs/final/env/*/env.json | sort | uniq -c   # L40S; EPYC 9334; the drivers
grep -c FAILED logs/final/env/models-check.txt                                 # 0
ls logs/final/telemetry | head        # sp-<cell>-<id>.csv.gz / .json / .apps.txt per job
```

**The tables and the check:**

```bash
cd code
PYTHONPATH=src:. ../.venv/bin/python -m experiments.comparison.aggregate --platform final
(cd .. && bash code/experiments/package/run.sh --platform final --optional --llama-licence --dry-run all \
    > logs/final/dryrun.txt)
PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final --check \
    --expect ../logs/final/dryrun.txt 2>&1 | tee ../logs/final/numbers_check.txt; echo "status ${PIPESTATUS[0]}"
cd ..
```

Expected: the report's first line ends with `main cells on the real checkpoints (with the Llama-2 licence)`, and
the report ends with `COMPLETE: every table cell and every macro is in the run` and `status 0`. Its
`machines:` line lists nodes among n-801 to n-805 with drivers 535 and 595, which is fine. It also prints an
`environment:` and a `telemetry:` line. `WARNING env: the nodes differ in driver` (535 against 595) is expected and
harmless. Please report any other `WARNING env` line and every `WARNING telemetry` line to us; they need no action.

If the report ends with `MISSING: ...`, `CHECK FAILED ...` or `status 1`, please do not run anything again on your
own. `TROUBLESHOOTING.md` (Sec. 6) says which of these you can fix. Otherwise send us `numbers_check.txt` in the
archive below.

**The archive.** It holds all of `logs/final` except the caches, the llama.cpp build and zkLLM's home: the job logs
and records, the environment snapshots (`env/`), the GPU telemetry (`telemetry/`), the smoke's pytest files, the
zkLLM runs (`zkllm/summary.csv` and one folder `zkllm/llama2-{7b,13b}-T{64,512,1024,2048}/` per model and length,
with the raw timings `bin_times.log` and `time-*.txt`), the check; and the benchmark's raw records and tables:

```bash
tar czf final-<COMMIT>.tgz --exclude=triton --exclude=native --exclude=setup_cache --exclude=llamacpp \
    --exclude='ref_*.pt' --exclude=zkllm/home \
    logs/final code/artifacts/comparison/raw_final code/artifacts/comparison/tables_final
tar tzf final-<COMMIT>.tgz > final-<COMMIT>.list
grep -E 'zkllm/summary.csv|numbers_check.txt|dryrun.txt|env/inputs.sha256|env/models-check.txt' final-<COMMIT>.list
grep -cE 'zkllm/llama2-(7b|13b)-T[0-9]+/env.txt$' final-<COMMIT>.list           # 8
grep -cE 'smoke_pytest-.*\.(txt|xml)$' final-<COMMIT>.list                      # 2 (more if the smoke ran twice)
grep -cE 'env/[^/]+/env.json$' final-<COMMIT>.list                              # 15 or more (one per allocation)
grep -cE 'telemetry/sp-.*\.csv\.gz$' final-<COMMIT>.list                        # 246 or more (one per job run)
grep -cE 'comparison/raw_final/' final-<COMMIT>.list                            # more than 0
ls -lh final-<COMMIT>.tgz
mkdir -p $H/return && cp final-<COMMIT>.tgz $H/return/ && (cd $H/return && sha256sum final-<COMMIT>.tgz)
```

The first `grep` should list `logs/final/zkllm/summary.csv`, `numbers_check.txt`, `dryrun.txt`,
`logs/final/env/inputs.sha256` and `logs/final/env/models-check.txt`; the counts should be as in the comments. A
length that zkLLM's demo did not prove still has its folder and its `env.txt`. The telemetry adds a few tens of
MB to the archive. Please tell us the status line, the archive's checksum and the start times of the fifteen jobs.
Thank you!

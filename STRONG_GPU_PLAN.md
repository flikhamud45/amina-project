# Strong-GPU runs for *Certified but Compromised*

Instructions for re-running our benchmark on the strong GPUs of the TAU cluster (H100/H200). The
goal is to remove the compromises the weak GPU forced on the paper and to add the GPU-heavy
experiments that would strengthen it. The runs happen on the same cluster and in the same
project folder as ours, so nothing has to be copied or installed.

* Part A: what is wrong and why.
* Part B: for the team, before the runs.
* Parts C–E: for you, the person running the GPU jobs (setup, smoke test, the runs by
  priority, experiments that need coordination).
* Parts F–G: rules, and what the team does with the results.

Parts B–F (the runs) use branch **`strong-gpu`**: the submitted paper (branch `submission`, commit
`336a03c`) plus a tested patch that adds what these runs need. Do not run from `submission`,
`main` or `comparison`. Part G (the team's checks) uses branch `paper-v2`: the same benchmark
code, plus the committed `rtx2080ti-v2` records (B.2), the paper written from them, and newer
report scripts and documentation. The setup of C.2 and the smoke test of C.4 have been run exactly as
written on the team's RTX 2080 Ti (job 943658: 243 GPU tests passed, none skipped; 139
fingerprints against the stored records, 0 problems; `SMOKE OK`). On the H100 the smoke test
must pass again, because exactness is checked per GPU type.

```bash
P=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project   # the shared project folder
```

The cluster's login shell is tcsh. Type `bash` first; every command below is bash.

---

## A. What the weak GPU compromised

All benchmark numbers in the paper come from **one RTX 2080 Ti (11 GB)** on the `studentbatch`
partition as prover, and **8 threads of a Xeon Silver 4114** as verifier. Since B.2 they are the
re-run of the fixed code on that hardware (platform `rtx2080ti-v2`): every decoder at full depth
except Llama-2-13B, with the full-model attacks. The table lists what the weak GPU and the first
run (`raw/`) compromised, in order of how much it matters, and what is left for the H100: mainly
Llama-2-13B at full depth, the rest of zkLLM's model list, and same-class GPU comparisons.

| # | Compromise | Where it shows in the paper | Fixed by |
|---|---|---|---|
| 1 | **Llama-2-13B is not built in full.** In the first run (`raw/`) no LLM above 12 blocks was: OPT-1.3B, OPT-6.7B, Llama-2-7B, Llama-2-13B and Qwen3-4B were measured as 1- and 2-block builds and extrapolated linearly (`aggregate.py`: `full = m1 + (L-1)(m2-m1)`). In the paper's `rtx2080ti-v2` run (B.2) all of them are measured at full depth except Llama-2-13B, whose int8 weights alone (12.1 GiB) exceed the card; it stays extrapolated from 1- and 2-block builds. On the models built in full, the extrapolation gives the proof size within 0.02% (Kpre exactly) and the prover within 12%, but underestimates the C verifier at ≤ 64 tokens by up to 48%, so Llama-2-13B's C verifier time is likely too low. | Table 3 (the starred Llama-2-13B row), §4.1, Limitations (iii); Llama-2-13B is missing from Figure 4 | C.5 `must` (c): `f-llama13-64` |
| 2 | **Few queries per cell on the big models** in `raw/`: 2 (OPT-6.7B), 3 (OPT-1.3B, Llama-2-13B), 5 (Llama-2-7B, Qwen3-4B), and only 2 for Llama-2-7B at 2,048 tokens. Fixed in `rtx2080ti-v2`: 10 queries at ≤ 64 tokens and 5 at 2,048 on every full build. | — | done (B.2) |
| 3 | **The zkLLM comparison is across GPU classes**: zkLLM ran on an A100 40GB, we on a 2080 Ti (the Contributions: "17–47× against zkLLM on a stronger GPU"). This cluster has no A100, so the fair fix is to run zkLLM itself on the same H100 as ours. | Figure 5, Table 4, Contributions, §4.4 | C.5 `must` on the H100; D.1 zkLLM on the same H100 |
| 4 | **zkLLM's 2,048-token model list is incomplete**: we have OPT-125M, OPT-1.3B, OPT-6.7B and Llama-2-7B, but not Llama-2-13B at 2,048 tokens, OPT-350M, OPT-2.7B or OPT-13B. | Figure 5, Table 4 | C.5 `should` (f) |
| 5 | **A performance bug in committed-weights mode (C)**, found while preparing this plan, that only shows on a GPU. `MatOp._weights_on` caches the GPU copy of the weights under `str(device)`. The forward pass asks for `"cuda:0"` and fold/open ask for `"cuda"`, so every C query uploaded the whole model to the GPU twice, and the C prover times of `raw/` are **too slow**. The patch fixes it, and the paper's numbers are now the fixed code's (`rtx2080ti-v2`, B.2). Its `_nofix` controls measure the effect on the same GPU: Llama-2-7B at 64 tokens 11.05 s with the bug against 7.86 s fixed, GPT-2 1.15×, OPT-1.3B at 2,048 tokens 1.16×, within noise for the small CNNs. | fixed in the paper (B.2) | done (B.2); C.5 `must` (d) repeats the controls on the H100 |
| 6 | **The LLM attacks** in `raw/` were one +1 tamper per cell, and on 1–2-block builds for the big models. In `rtx2080ti-v2` the full OPT-1.3B, Qwen3-4B, OPT-6.7B and Llama-2-7B are also attacked (30 each of `single_value`, `output_logit`, `last_hidden` and 3 `substituted_block_1pct` per model and prompt length ≤ 64): 602 LLM attacks, all rejected. Llama-2-13B still has only its +1 tampers, on the 1–2-block builds. | §4.3 counts | done (B.2) except Llama-2-13B: C.5 `must` (c) |
| 7 | **GPU memory and one-time costs** (commitment time, peak memory) were not reported for the LLMs. `rtx2080ti-v2` records them (§4.1 quotes the 7.3 GiB peak of Llama-2-7B at 2,048 tokens). | §4.1 (Llama-2-7B's peak GPU memory); the commitment times are still unreported | recorded by every new run |
| 8 | **The verifier ran only on a CPU**, while zkLLM's verifier ran on its A100. | Figure 5, Table 4 verifier ratios, Future work (2) | C.5 `should` (g): GPU verifier |
| 9 | **Nothing beyond 13B**, no batching, no 4,096-token context. | Future work | C.5 `nice` |
| 10 | **Real LLM weights were never used**: costs were measured with random int8 weights of the exact shapes (costs depend only on shapes), so the quality of the int8 LLMs is unknown, and the attack was never shown on a real LLM. | Limitations (iii), §4.2 | D.2, D.3 (need new code) |

Not in this plan, because a strong GPU does not help: the MNIST experiments (CPU), the CPU-only
zkSNARK baselines (zkCNN, zkGPT, DeepProve, EZKL, ZKML, ZKTorch), and the closed-form path-test
numbers on the LLMs.

---

## B. Team: before the runs

### B.1 Put the branches on the server

`strong-gpu` (for the runs) and `paper-v2` (for the team's checks in Part G) must exist in the
shared repository at `$P`. From a checkout that has both branches (the server's object
directories are not all group-writable, so the push must keep its pack):

```bash
git push --receive-pack="git -c safe.directory='*' -c receive.unpackLimit=1 receive-pack" \
    ssh://<user>@c-008.cs.tau.ac.il$P strong-gpu:refs/heads/strong-gpu paper-v2:refs/heads/paper-v2
```

Nothing else has to be prepared. The trained CNN weights (`$P/code/artifacts/fullcheck/models`,
225 MB), the ImageNet caches (`$P/code/artifacts/fullcheck/cache`, 3.0 GB), MNIST
(`$P/code/data`), CIFAR-10 (the lab copy the code reads by default) and the Python environment
(`$P/.venv`: Python 3.12.3, torch 2.5.1+cu121, which runs on H100/H200/A6000/L40S/RTX 3090) are
all already there. **They must not be retrained or rebuilt**: new CNN weights would make every
CNN number incomparable with the paper.

### B.2 Done: our own RTX 2080 Ti numbers, re-measured with the fixed code

**Status: done. `rtx2080ti-v2` is now the paper's platform.** Its records
(`code/artifacts/comparison/raw_rtx2080ti-v2/`, 78,266 records from 29 jobs, all at commit
`9401431` with a clean tree) and tables (`tables_rtx2080ti-v2/`) are committed (commit `a0d0b3b`,
branch `paper-v2`), and every benchmark number, table, figure and count of the paper is rebuilt
from them (`aggregate.py`, `count_outcomes.py` and `paper_assets.py` with `--platform
rtx2080ti-v2`). The frozen `raw/` stays as the earlier run with the older code (59,547 records;
still reproducible with `--platform rtx2080ti`), described in the READMEs but not used for the
paper's numbers.

Compromise 5 (the weight re-upload) sat in every C-mode number of `raw/`. With the fixed code and
the lean path, every model except Llama-2-13B also fits the 2080 Ti at full depth, so the team
re-measured the report's own hardware with the same jobs, from its own clone (set up as in C.2,
with `W=$P/logs/strong_gpu_2080`), under the platform name `rtx2080ti-v2`, because `rtx2080ti`
means the frozen stored records:

Already run; do not run it again: new records in `raw_rtx2080ti-v2/` would change the paper's
counts (on `paper-v2` its `PLATFORM.json` is marked `"frozen": true`, so `bench.py` refuses):

```bash
export PVI_PLATFORM=rtx2080ti-v2 SBATCH_PARTITION=studentbatch SBATCH_GRES=gpu:geforce_rtx_2080:1
bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh must      # the smoke test already passed on this GPU type (job 943658)
```

On the 2080 Ti the full Llama-2-13B build does not fit: `f-llama13-64` skipped it with a `SKIP`
line and ended with an error after recording its 1- and 2-block builds. The non-lean control
`ab-opt13` (about 27 GiB of GPU memory) does not fit either and was not run:
`raw_rtx2080ti-v2/` holds 29 jobs (SLURM ids 945088–945116) and no `_nolean` OPT-1.3B cell.
`bench.py`'s `SKIP` test only looks at the int8 weights, so on an 11 GB card `ab-opt13` would
stop with a CUDA out-of-memory error. This takes about 9.5 GPU-hours on `studentbatch`. It gave
corrected 2080 Ti numbers, measured the extrapolation error on the report's hardware (proof
size within 0.02%, prover within 12%, but the C verifier at ≤ 64 tokens up to 48% too low), and
makes an honest cross-GPU comparison possible: use it, not `rtx2080ti`, as the 2080 Ti in any
comparison with the H100 (Part G).

---

## C. The GPU runs

### C.1 Which GPUs

| Partition | GPU | Use it for |
|---|---|---|
| `gpu-h100-killable` | H100 80 GB | **everything in C.5**: this is the headline platform (`PVI_PLATFORM=h100`) |
| `gpu-h200-killable` | H200 141 GB | optional second platform (`h200`). Its only extra is room for memory-only experiments (non-lean 7B at 2,048 tokens, about 90 GiB) |
| `killable`, `gpu:a6000` | RTX A6000 48 GB | D.1 (zkLLM was tested on this card class) |
| `killable`, `gpu:geforce_rtx_3090` | RTX 3090 24 GB | optional: the card Anchuri et al. re-ran zkLLM on |
| `gpu-b200-killable` | B200 | **do not use**: the installed torch (2.5.1+cu121) cannot run on Blackwell GPUs |

List the partitions you can use with `sinfo -o "%P %G %l %D"`, and check your account with
`sacctmgr show assoc user=$USER format=account,partition`. All strong-GPU partitions are
*killable*: a job can be pre-empted, and the time limit is 1 day. The jobs are built for this (C.6).

Resources per job (the scripts request them): 1 GPU, 8 CPU cores (the verifier runs on them, and
they set the verifier timings), 96 GB RAM (128 GB for the 13B-class 2,048-token jobs, 192 GB for
the full 70B), at most 6 hours (12 for the full 70B).

### C.2 Setup (once, on the login node)

Work in **your own clone** of `strong-gpu` inside the project folder. The main checkout at `$P`
is on another branch and is shared, so do not switch it. The clone links to the existing inputs
and environment:

```bash
bash
P=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project
W=$P/logs/strong_gpu                       # your working copy (logs/ is ignored by the main checkout)
git clone --no-local --upload-pack="git -c safe.directory='*' upload-pack" -b strong-gpu $P $W
cd $W && git log -1 --format='%H %s'       # note the commit
mkdir -p code/artifacts/fullcheck
ln -s $P/code/artifacts/fullcheck/models code/artifacts/fullcheck/models
ln -s $P/code/artifacts/fullcheck/cache  code/artifacts/fullcheck/cache
ln -s $P/code/data code/data
ln -s $P/.venv .venv
printf '%s\n' code/data code/artifacts/fullcheck/models code/artifacts/fullcheck/cache .venv >> .git/info/exclude
ls code/artifacts/fullcheck/models         # lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel (.pt + .json)
(cd code/artifacts/models && sha256sum -c MODELS.sha256)
git status --porcelain                     # must print nothing
```

The shared venv's `pvi` package points at the main checkout's code, not at your clone. The job
scripts therefore set `PYTHONPATH=$W/code/src` themselves, and `bench.py` refuses to start if the
`pvi` it imported is not the clone's. If you run anything by hand, do it from `$W/code` with
`export PYTHONPATH=$PWD/src`.

### C.3 Environment for every submission (from `$W`)

```bash
cd $W
export PVI_PLATFORM=h100                                   # one name per GPU model + CPU model
export SBATCH_PARTITION=gpu-h100-killable SBATCH_GRES=gpu:h100:1
# only if your account needs them:  export SBATCH_ACCOUNT=... SBATCH_QOS=...
```

New records go to `$W/code/artifacts/comparison/raw_h100/`, and logs to `$W/logs/h100/`. The
earlier run's records in `code/artifacts/comparison/raw/` are frozen: the code refuses to write
there. The paper's records are `raw_rtx2080ti-v2/` (B.2); never use `PVI_PLATFORM=rtx2080ti-v2`
(or `rtx2080ti`) for new runs.

This cluster does not pin a job to its CPU cores. Each job header prints something like
`verifier threads 8 on 40 CPUs / 20 physical cores`: the verifier runs 8 threads, but may
share the node's cores with other jobs. So the verifier timings depend slightly on how busy the
node is. This was also true for the paper's numbers, and every record stores the affinity, so
nothing has to be done about it. Just do not run more of our jobs on one node than it has GPUs.

### C.4 Smoke test (mandatory, once per GPU type, 30–40 minutes)

```bash
bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh smoke
```

When it finishes, `logs/h100/h100-smoke-<jobid>.out` must end with **`SMOKE OK`**. It checks:

* **the test suite on the GPU: no failure, no error, and nothing skipped.** On a CPU, 44 tests are
  GPU-only; on the H100 they must run. The defence relies on exact integer arithmetic, and these
  tests prove it on this GPU, with TF32 both off and on;
* the inputs (the CNN weights, CIFAR-10, the two ImageNet sets);
* small benchmarks (LeNet-5, GPT-2, Llama-2-7B at 2,048 tokens as a 2-block build, and one lean
  full-depth OPT-1.3B build with memory records), in a throw-away root that is moved to
  `logs/env/` at the end;
* `fingerprint_check`: every hardware-independent number (accuracies, proof sizes, detection
  probabilities, attack outcomes) equals the stored records. It must report `0 problems`.

If anything else happens (`MISMATCH`, `HONEST QUERY REJECTED`, a failed or skipped test, no
`SMOKE OK`), **stop and send us the log** and the files `logs/env/<jobid>.*`. Do not start the
long runs.

### C.5 The runs, by priority

Each tier is one command. Look at the jobs first, then submit:

```bash
PVI_DRYRUN=1 bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh must    # print the sbatch lines
bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh must                 # submit
squeue -u $USER
```

Job names start with the platform (`h100-f-llama7-2k`, ...). Hours below are upper bounds scaled
from our 2080 Ti + Xeon 4114 timings. On the H100 expect roughly half: most of the time is the CPU
verifier and the one-time SHA-256 commitment, so it is the node's CPU, not the GPU, that sets the
pace. The jobs of a tier run in parallel on separate GPUs, 8 cores each.

#### `must` (about 11.5 node-hours; about 2–3 hours of wall time with 8 GPUs)

| Jobs | What | Fixes (A) | Hours |
|---|---|---|---|
| `b-mlp_mnist`, `b-lenet5`, `b-vgg11`, `b-vgg16`, `b-resnet18_cifar`, `b-resnet18_224` | the paper's CNN jobs with unchanged flags. Their hardware-independent numbers and their counts (honest queries, 1,854 attacks) must equal the stored ones exactly, which proves the run is comparable; their timings are the CNN rows on the H100 | 3 | 0.5 |
| `b-gpt2`, `b-opt125`, `b-gpt2-1t` | the 12-block LLMs (already built in full), for the same-GPU table | 3 | 0.5 |
| `f-opt13-64`, `f-opt67-64`, `f-llama7-64`, `f-llama13-64`, `f-qwen4`, `f-qwen4-1t`, `f-opt13-2k`, `f-opt67-2k`, `f-llama7-2k` | **every >12-block LLM of the paper at full depth**, plus 1- and 2-block builds on the same machine (which measures the extrapolation error). 10 queries at ≤ 64 tokens and 5 at 2,048; λ = 128; modes C and Kpre; at ≤ 64 tokens also 30 attacks on the full model | 1, 2, 3, 6, 7 | 8.2 |
| `ab-gpt2`, `ab-llama7`, `ab-opt13` | controls: the lean path against the non-lean path (same integers, different memory) | — | 1.0 |
| `nofix-*` (GPT-2, Llama-2-7B, OPT-1.3B at 2,048 tokens, the six CNNs) | controls: the old weight re-upload (`PVI_LEGACY_WEIGHT_KEY=1`) against the fix, on the same GPU | 5 | 0.8 |
| `microbench` | kernel timings (float32 forward, float64 fold/open, NTT, host↔device copies) that explain where the prover's time goes | 3 | 0.1 |

After `must`, **tell us** (Part E) before going on. We check the results first (Part G).

#### `should` (about 12 node-hours in the script, plus Part D)

| Jobs | What | Fixes | Hours |
|---|---|---|---|
| `z-llama13-2k`, `z-opt13b-2k`, `z-opt27-2k`, `z-opt350-2k` | the rest of zkLLM's 2,048-token list, measured in full | 4 | 5.8 |
| `g-opt-125m-2k`, `g-opt-1.3b-2k`, `g-opt-6.7b-2k`, `g-llama2-7b-2k`, `g-llama7-64`, `g-gpt2` | the verifier on the GPU (tag `_gpuv`): a same-hardware verifier comparison with zkLLM | 8 | 3.7 |
| `v-llama7-64`, `v-llama7-2k`, `v-opt13-2k` | 4-, 8- and 16-block builds (6 and 12 for OPT-1.3B): how the cost grows with depth | 1 | 1.9 |
| `s-llama7-t1` | Llama-2-7B with a 1-token prompt, ZKTorch's setting | 3 | 0.4 |
| `s-llama7-thr12` | the verifier on 12 threads, zkLLM's host | 3 | 0.4 |

#### `nice` (about 10 node-hours)

| Jobs | What | Hours |
|---|---|---|
| `n-batch-*` | batching: 8–256 images or 8–32 prompts proved against one set of openings (tag `_batch`) | 2.3 |
| `n-llama7-4k`, `n-gpt2-1k` | Llama-2-7B at its full 4,096-token context; GPT-2 at 1,024 | 2.7 |
| `n-tf32-llama7`, `n-tf32-gpt2` | TF32 tensor cores for the (still exact) float32 GEMMs of the forward pass (tag `_tf32`) | 1.0 |
| `n-opt30`, `n-opt66`, `n-llama70`, `n-llama70-full` | 30–70B shapes at 1–2 blocks, and the full Llama-2-70B at 64 tokens (80 GB card, 192 GB RAM) | 4.4 |

### C.6 Pre-emption, time limits, and resubmitting

The strong-GPU partitions are killable, so some jobs will be stopped half-way. This is safe:

* every finished cell has a `.done` marker and is skipped when a job runs again; a half-written
  cell is kept aside (`.jsonl.part.<time>`) and redone from the start;
* to resume, **run the same tier command again**. Jobs that are still queued or running are not
  submitted twice (`skip <name>: already queued`). Every job also carries
  `--dependency=singleton`, so two copies of a job can never run at the same time;
* never delete `.done` files or edit the records by hand.

Look at a finished job with `sacct -j <jobid> --format=JobID,JobName%28,State,Elapsed,MaxRSS,NodeList`,
and at its log in `logs/h100/`. A job that ends with a `SKIP ... do not fit this GPU` line did
its other builds; tell us which ones were skipped.

---

## D. Experiments outside the job script (coordinate with the team first)

These add the most beyond C.5, but need extra software or new code.

1. **Run zkLLM itself on the same H100** (`should`; about 2 GPU-hours plus half a day to a day of
   build work). This cluster has no A100, and zkLLM is the only published GPU prover we compare
   with, so running both codes on one card is the only true same-hardware comparison. Code:
   `https://github.com/jvhs0706/zkllm-ccs2024`. The repository is archived and was built for sm_86
   cards (A6000/3090), so for the H100 set the CUDA architecture in its Makefile to sm_90. If it
   does not build, run it and our `must` LLM jobs on `gpu:a6000` instead (platform `a6000`).
   Llama-2 weights are gated: accept Meta's licence with your own Hugging Face account, and never
   share the token or the weights. Run Llama-2-7B and Llama-2-13B at sequence length 2,048, as in
   zkLLM's Table 1. Time the weight commitment once and the proving stages three times, each stage
   under `/usr/bin/time -v`, and log the GPU with
   `nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 500`.
   Send back zkLLM's commit hash, the logs and the proof-file sizes.
2. **The single-neuron attack and backdoor on a real Llama-2-7B** (`should`; under 1 GPU-hour; 24 GB
   is enough). Anchuri et al. evaluate their protocol on Llama-2-7B, but we showed the attack only
   on MNIST and CNNs. It needs a new script of about 200 lines, which the team writes. The script
   hooks one FFN neuron at the last position, searches for the smallest change that flips the next
   token, and adds a triggered variant. If Llama access is refused, use Qwen3-4B or OPT-6.7B.
3. **The quality of the int8 LLMs with real weights** (`should`; 8–20 GPU-hours; needs a new loader
   from Hugging Face checkpoints into our integer decoders). It closes Limitations (iii) with
   WikiText-2 perplexity and next-token agreement against fp16. The result may be negative:
   per-tensor int8 is known to hurt OPT and Llama above 6.7B. It would be reported either way.
   The cost numbers stay measured with random weights.

---

## E. When a tier is done

The results are already in the shared folder, so nothing has to be sent. From `$W`:

```bash
find code/artifacts/comparison/raw_$PVI_PLATFORM -name '*.part*' | wc -l     # 0, otherwise resubmit the tier (C.6)
for f in code/artifacts/comparison/raw_$PVI_PLATFORM/*/*/*.jsonl; do [ -f "${f%.jsonl}.done" ] || echo "MISSING_DONE $f"; done
ls code/artifacts/comparison/raw_$PVI_PLATFORM/*/*/*.rejected-*.jsonl 2>/dev/null   # must be empty
git status --porcelain                                                         # only raw_<platform>/ may be new
mkdir -p env && nvidia-smi -q > env/nvidia-smi.txt && lscpu > env/lscpu.txt && git rev-parse HEAD > env/git-head.txt
sacct -u $USER -S <date of the first job> --format=JobID,JobName%28,NodeList,Elapsed,State,MaxRSS,AllocTRES%60 -P > env/sacct.tsv
```

Then tell us the platform name and the tier. Do not commit, and do not move or delete anything.
We read the results from your clone.

---

## F. Rules and pitfalls

* **Never write into `code/artifacts/comparison/raw/` or `raw_rtx2080ti-v2/`**, never delete or
  move them, and always set `PVI_PLATFORM` to a new name. `raw_rtx2080ti-v2/` holds the paper's
  records and `raw/` the earlier run. `raw/` is marked frozen, and on `paper-v2` so is
  `raw_rtx2080ti-v2/`: the code refuses to write into a frozen root.
* **Never retrain the CNNs** (`train.py`, `train.sbatch`) and never rebuild the ImageNet caches.
* **Run from your clean clone of `strong-gpu`** (`git status` empty apart from `raw_<platform>/`,
  `logs/` and `env/`). Every record stores the commit and whether the tree was dirty.
* **One GPU model and one CPU model per platform name.** The H100 and H200 are two platforms. The
  first job in a new `raw_<platform>/` records the node's GPU and CPU, and jobs on other hardware
  are then refused. If that happens, use a separate platform name.
* **TF32 stays off**, except in the `_tf32` jobs; the scripts enforce it. Do not run in NGC
  containers.
* **Do not change the flags** of the `must` jobs: they are what makes the new numbers comparable
  with the paper.
* **An honest query must never be rejected.** If one is, the job stops and keeps the evidence in a
  `.rejected-*.jsonl` file. Stop everything and send us the log: it would mean an exactness
  problem on this GPU.
* The strong GPUs are shared by the whole cluster. Submit one tier at a time, and do not use more
  than about 8 GPUs at once unless the partition is idle.

---

## G. Team: after each tier

In a clean clone of your own of `paper-v2` (the benchmark code of `strong-gpu`, plus the
committed `rtx2080ti-v2` records, the paper written from them and newer report scripts), with
the friend's platform root copied in (read-only from `$P/logs/strong_gpu`), CPU only, a few
minutes. On the cluster, clone it from `$P` after B.1's push (as in C.2, with `-b paper-v2` and
a directory of your own); on your own machine, copy `raw_h100/` over with `scp -r` instead of
the `cp` below:

```bash
cp -r $P/logs/strong_gpu/code/artifacts/comparison/raw_h100 code/artifacts/comparison/
cd code && export PYTHONPATH=$PWD/src
python experiments/5_comparison/fingerprint_check.py artifacts/comparison/raw artifacts/comparison/raw_h100   # 0 problems
python experiments/5_comparison/fingerprint_check.py artifacts/comparison/raw_rtx2080ti-v2 artifacts/comparison/raw_h100   # 0 problems (full builds too)
python experiments/5_comparison/aggregate.py --platform rtx2080ti && git diff --exit-code artifacts/comparison/tables   # stored tables unchanged
python experiments/5_comparison/aggregate.py --platform rtx2080ti-v2 && git diff --exit-code artifacts/comparison/tables_rtx2080ti-v2   # the paper's tables unchanged
python experiments/5_comparison/aggregate.py --platform h100
python experiments/5_comparison/validate_extrapolation.py --platform h100     # the full builds against the 1-2 block line
python experiments/5_comparison/xplat_check.py rtx2080ti-v2 h100
python experiments/5_comparison/count_outcomes.py --platform h100
python experiments/5_comparison/paper_assets.py --platform h100 --check      # 0 missing
python experiments/5_comparison/paper_assets.py --platform h100 --compare rtx2080ti-v2   # the paper's 2080 Ti (rtx2080ti only to show the earlier run)
python experiments/5_comparison/text_numbers.py --platform h100              # every hand-typed number, recomputed
python -m pytest tests -q
```

`paper_assets.py` writes into `report/`, so if the headline stays on the 2080 Ti, restore the
paper's own assets afterwards with `git checkout -- ../report/figures ../report/tables`
(re-running `paper_assets.py` with another matplotlib than the pinned 3.11.2 changes the figure
PDFs and can change the page count).

Then decide the headline hardware. The paper now stands on `rtx2080ti-v2` (B.2). Either the tables
and figures move to the H100 (with the `rtx2080ti-v2` 2080 Ti shown next to it in Figure 5), or
they stay on `rtx2080ti-v2`, and the H100 adds what the 11 GB card could not: Llama-2-13B at full
depth, the rest of zkLLM's 2,048-token list, and same-class GPU comparisons (zkLLM on the same
H100, D.1). Update the hand-typed sentences. `text_numbers.py --platform h100` lists the
extrapolation check and the peak memory in §4.1, the zkLLM ranges in the Contributions and §4.4,
and the other ranges of §4.3–4.4. It does not list: the 11 GB GPU and the extrapolated
Llama-2-13B in §4.1 and Limitations (iii); the counts in the abstract, §4.1 (the record total),
§4.3 and the conclusion, which come from `count_outcomes.py --platform h100`; the captions of
Tables 2–4 and Figures 4–5. `main.tex` already inputs `tables/hardware.tex`, which provides
`\ProverGPU`, `\VerifierCPU` and `\VerifierThreads` (used in §4.1) and `\LLMNote` (Table 3's
caption). Rebuild the figures with the pinned matplotlib (3.11.2, `code/requirements.txt`) and
the PDF, keep it at 8 pages, commit `raw_h100/` with the report on `paper-v2`, and merge into
`submission` only when all of this passes.

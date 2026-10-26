# L40S machine: the final run

Thank you for the rehearsal. We have gone through it, and the code is ready for the final run. This file describes
the run: about a day of waiting in the queue and one to two days of computing, in eight SLURM jobs.

## Your rehearsal

* The smoke ended in `SMOKE OK` (774 passed), and `numbers --check --rehearsal` ended with `status 0`. It filled the
  same table cells and text numbers as our rehearsal on a 2080 Ti (93 of the 436 macros; the full run fills the rest).
* `torch._int_mm` on the EPYC 9334 is bit-exact with the integer reference. This is the one check a 2080 Ti could
  not make, and it puts that re-execution baseline among the exact ones.
* Every CPU-timing job used the native kernels and was pinned to one NUMA node. The only tracebacks are those of the
  four jobs that failed on `HF_HUB_CACHE`, which you ran again.
* Following your suggestion, `run.sh` now refuses a Hugging Face cache variable (`HF_HUB_CACHE`,
  `TRANSFORMERS_CACHE`, `HF_DATASETS_CACHE`) that overrides `HF_HOME` with a folder the jobs cannot write.

## What changed: commit `a405361`

Only the run package (`run.sh`, its README and its tests); no measured code changed, and the six image models you
trained stay valid. The change concerns the queue. Your smoke waited 6 h 40 min for a whole node, and the run as one
chain of 179 whole-node jobs would wait about as long before each of them. The new option `--one-job` submits a
group of tiers as one job, which waits once and then runs the group's jobs one after another on the node it got.
The run is eight such jobs: six whole-node jobs for the tiers whose timings the paper reports, and two jobs on one
GPU of a shared node for `quality`, `tv` and `attack`, which time nothing.

## 1. Update your clone

```bash
H=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/validation/sp2027/handoff_l40s
(cd $H && sha256sum -c SHA256SUMS)
git -C <your clone> fetch $H/pvi-sp2027-artifact-a405361.bundle sp2027-artifact
git -C <your clone> checkout -B sp2027-artifact FETCH_HEAD && git -C <your clone> log --oneline -1   # a405361
```

## 2. Environment

Use the rehearsal's settings: `SBATCH_PARTITION`, `SBATCH_GRES=gpu:l40s:1`, your exclusion of `t-806`, `HF_HOME`,
`HF_TOKEN`, `TMPDIR`, `PVI_PY_INCLUDE`, `CXX`, `PVI_CIFAR_ROOT` and `PVI_IMAGENET_ROOT`. Also run `unset HF_HUB_CACHE`.

`HF_HOME` needs about 100 GB free (`df -h $HF_HOME`). The run downloads OPT-1.3B and OPT-6.7B, OpenLLaMA-7B, Qwen3,
TinyLlama, Llama-2-7B and Llama-2-13B.

Then, from the repository root, do a dry run of the first job:

```bash
bash code/experiments/package/run.sh --platform final --llama-licence --dry-run --one-job smoke cpuv long
```

It should print no `run.sh:` line, which would name a missing setting. Its last lines should read
`# one job: smoke cpuv long: 26 jobs, job limits 53 h, time limit 24 h`.

## 3. Start the run

From the repository root, on the login node:

```bash
R="bash code/experiments/package/run.sh --platform final --llama-licence"
$R --one-job smoke cpuv long                     # expected 11-17 h once it starts
$R --one-job smoke bench                         # 9-13 h
$R --one-job smoke real                          # 9-13 h
$R --one-job smoke serve gen batch               # 13-17 h
$R --one-job smoke v1 setup improve              # 9-16 h
$R --one-job smoke gpuv                          # 5-7 h
$R --one-job --no-exclusive smoke quality        # 11-17 h, one GPU of a shared node
$R --one-job --no-exclusive smoke tv attack      # 6-9 h, one GPU of a shared node
squeue -u $USER
```

Each line submits one job, named `sp-node-<tiers>`, and prints its log `logs/final/node-<tiers>-<job id>.log`.
The first six jobs each wait for a whole L40S node; they may run at the same time on different nodes. Your
rehearsal ran on n-802 to n-805. n-805 has NVIDIA driver 595.99.02 and the others 535.183.01; both are fine, since
the records name each job's node and driver. Only `t-806` (a Xeon) stays excluded.

## 4. While it runs

* Each job starts with the smoke on its node. If the smoke fails, the job stops, and its log says
  `smoke did not end in SMOKE OK`. In that case, please send us `logs/final/sp-smoke-*.out`.
* The job logs are `logs/final/sp-<cell>-<id>.out`, as in the rehearsal. A job that ends successfully leaves a
  marker in `logs/final/done/`.
* **If one of the eight jobs ends at its 24-hour limit**, or a job inside it failed (its log then shows
  `<name>: exit status <n>`), submit the same line again once it has left the queue. It runs the smoke again and only
  the jobs that have not finished.
* **A pre-empted job** (the partition requeues jobs) starts again by itself and skips what it had finished. Nothing
  needs doing.
* `A CORRECTNESS CHECK FAILED` or a line starting with `FAILED:` in any log is a real problem: please tell us.
* Please note when each of the eight jobs started; their queue waits help us plan.

## 5. When all eight have finished

From the repository root:

```bash
srun -p "$SBATCH_PARTITION" --gres="$SBATCH_GRES" -c 8 bash -c 'nvidia-smi -q > logs/final/env-nvidia.txt;
    lscpu > logs/final/env-cpu.txt; free -g > logs/final/env-mem.txt'
.venv/bin/pip freeze > logs/final/env-pip.txt; git rev-parse HEAD > logs/final/env-git.txt
cd code
PYTHONPATH=src:. ../.venv/bin/python -m experiments.comparison.aggregate --platform final
PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final --check \
    2>&1 | tee ../logs/final/numbers_check.txt; echo "status ${PIPESTATUS[0]}"
cd ..
tar czf final.tgz --exclude=triton --exclude=native --exclude=setup_cache --exclude='ref_*.pt' logs/final \
    code/artifacts/comparison/raw_final
```

Please copy `final.tgz` into `$H/return/` and tell us the status. Expected: `status 0`. The check lists any cell
still missing, and we will tell you if a job needs to run again.

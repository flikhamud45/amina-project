# Fixes for the reported failures (commit `<COMMIT>`)

Thank you for the reports in `return/final_issue_cpuv`, `final_issue_long` and `final_issue_tv_attack`. Commit
`<COMMIT>` fixes all four failures. **Keep `logs/final/` as it is.** Every job that has already finished stays
valid and is skipped when you resubmit. The new commit changes no default code path, so its proofs and timings are
the same as before.

## What was wrong and what changed

| Job | Failure | Fix |
|---|---|---|
| `sp-c-llamacpp-llama2-7b` (`cpuv`) | `sentencepiece` is not installed | The tokenizer is now read without `sentencepiece`. Do not install anything. |
| `sp-c-v1-llama2-7b-2048` (`cpuv`) | GPU out of memory in V1's GKR | The job now passes `--gkr-memory auto`, which frees GPU memory the GKR no longer reads. |
| `sp-l-llama7-8192-full` (`long`) | GPU out of memory (claims) | The job now passes `--host-claims auto`, which moves the claims that do not fit to host memory. |
| `sp-a-inrange-opt67` (`tv attack`) | the 4 h time limit, twice | It is now two jobs: `sp-a-inrange-opt67` (3 h) and `sp-a-inrange-opt67-down` (6 h). |

The same `--gkr-memory` flag is also given to the other V1 jobs at 2,048 tokens that would run out of GPU memory
(`v1`, `real`, `gpuv` and `gate`). All other jobs run exactly as before.

## What to do

1. Update the clone **soon**. A queued allocation reads the clone's `run.sh` when it starts, so the lines still
   pending (`real`, `bench`, `serve gen`, `v1 setup`, `improve batch`, `cpux`, `gpuv`, `zkllm`) then run with the
   fixes. Do not cancel them: they keep their place in the queue.

```bash
H=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/validation/sp2027/handoff_l40s
(cd $H && sha256sum -c SHA256SUMS)                    # every line: OK
git -C <your clone> fetch $H/pvi-sp2027-artifact-<COMMIT>.bundle sp2027-artifact
git -C <your clone> checkout -B sp2027-artifact FETCH_HEAD
git -C <your clone> log --oneline -1                  # <COMMIT>
git -C <your clone> status --short --untracked-files=no   # prints nothing
```

   Do **not** move or delete `logs/final/`. This is unlike Sec. 1 of `FINAL_RUN.md`, which is for starting a new
   run.

2. Resubmit the three lines that failed, with `R` as in Sec. 3 of `FINAL_RUN.md`:

```bash
R --one-job cpuv
R --one-job long
R --one-job --no-exclusive tv attack
```

   Only the failed jobs run: two in `cpuv`, one in `long` and two in `tv attack`. The attack jobs are expected
   to take about 2 h and 4 h; the others take about as long as their twins that already finished.

3. Order the waiting lines by their importance to the paper. A user may lower the priority of their own jobs; this
   changes only the order of your own lines, which stay far ahead of the other users' waiting jobs. The values are
   the smallest that give this order, both now and once `bench` (resubmitted 12 h after the others) has the same
   age as the rest:

```bash
scontrol update jobid=1091951 nice=1     # real: the abstract's real OPT-6.7B
scontrol update jobid=1091950 nice=2     # bench: tab:eval-llm and tab:eval-compare
scontrol update jobid=1091956 nice=19    # gpuv: tab:options (GPU columns)
scontrol update jobid=1091957 nice=20    # zkllm: zkLLM's row of tab:eval-compare
scontrol update jobid=1091955 nice=21    # cpux: extended version
scontrol update jobid=1091954 nice=22    # improve batch: extended version
scontrol update jobid=1091952 nice=23    # serve gen: extended version
squeue -u $USER -o "%i %j %Q" --sort=-p  # v1 setup, real, bench, gpuv, zkllm, cpux, improve batch, serve gen
```

   `v1 setup` (1091953), with the abstract's V1 ratio, keeps nice 0 and goes first. If the job numbers differ
   from yours (`squeue -u $USER`), use your own, matched by name.

4. Once the other lines finish, resubmit any line that has a job without a marker in `logs/final/done/`, as
   before.

`FINAL_RUN.md` and `TROUBLESHOOTING.md` in this folder are updated to the new commit. The whole run is now 246 jobs
with 669 h of job limits, and `tv attack` is 8 jobs. The final steps of `FINAL_RUN.md` (Sec. 6-7) are unchanged.

# Fix for the two `bench` failures, and the order of the waiting lines (commit `e258288`)

Thank you for the report in `return/final_issue_bench`. Commit `e258288` fixes both failures. **Keep
`logs/final/` as it is**: every finished job stays valid and is skipped when you resubmit. The new commit changes
no other job.

## What was wrong and what changed

| Job | Failure | Fix |
|---|---|---|
| `sp-b-cg-llama2-7b-2k`, `sp-b-cg-llama2-13b-2k` | GPU out of memory in the GPU verifier's decoder | Both jobs now pass `--decode-memory auto`, which decodes the claims part by part when they would not fit at once (the same claims and proof). |

## What to do

1. Update the clone. Do not move or delete `logs/final/`, and do not cancel any waiting line.

```bash
H=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/validation/sp2027/handoff_l40s
(cd $H && sha256sum -c SHA256SUMS)                    # every line: OK
git -C <your clone> fetch $H/pvi-sp2027-artifact-e258288.bundle sp2027-artifact
git -C <your clone> checkout -B sp2027-artifact FETCH_HEAD
git -C <your clone> log --oneline -1                  # e258288
git -C <your clone> status --short --untracked-files=no   # prints nothing
```

2. Resubmit `bench` with `R` as in Sec. 3 of `FINAL_RUN.md`. Only its two failed jobs run (a few hours at most, most of it loading the checkpoints).

```bash
R --one-job bench
```

3. Order the waiting lines by what the paper needs first. The abstract is fixed on 10 November and its last
   missing number (V1 on the real OPT-6.7B) comes from `real`, so `real` should start first. These are the smallest
   values that give this order: they keep every line far above the other users' jobs that wait for the same
   nodes (so the run as a whole does not finish later), and they only order our own lines among themselves. Setting
   a smaller value than a line has now (for example 3 for `gpuv`, which has 19) is allowed: only negative values
   are not.

```bash
scontrol update jobid=<new bench> nice=1   # bench: the comparison with zkLLM (tab:eval-compare)
scontrol update jobid=1108594 nice=2       # cpuv: the llama.cpp baseline and V1 on the CPU client
scontrol update jobid=1091956 nice=3       # gpuv: tab:options (GPU columns)
scontrol update jobid=1091957 nice=4       # zkllm: zkLLM's row of tab:eval-compare
scontrol update jobid=1108595 nice=5       # long: the long-prompt extrapolation check
scontrol update jobid=1091955 nice=6       # cpux: extended version
scontrol update jobid=1091954 nice=7       # improve batch: extended version
scontrol update jobid=1109468 nice=8       # serve gen: extended version
squeue -u $USER -o "%i %j %Q" --sort=-p
```

   `real` (1108813) keeps nice 0. If a job number differs from yours, use your own, matched by name. The scheduler
   also adds an age bonus that grows for three days, so the order above holds from about 3 November in the
   afternoon, when the lines resubmitted on 31 October have their full bonus (the nodes are expected to free later
   than that), and the new `bench` reaches its place about three days after it is submitted.

4. When `real` has finished, please run `PARTIAL_RESULTS.md` again (a new snapshot for the abstract), even if
   other lines are still waiting. At the end, send the full archive of Sec. 6 of `FINAL_RUN.md` as before.

`FINAL_RUN.md` and `TROUBLESHOOTING.md` are unchanged apart from the commit.

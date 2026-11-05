# Llama-2-7B: its calibration draws at the new setting, and a diagnosis (commit `c39ab9a`)

Thank you for the reruns of 3 November: all eight jobs ended well. The longer prefix removed the collapse
(Llama-2-7B went from about 265 times fp32's perplexity to about 6), but Llama-2-7B is still far from fp32, and its
setting fixed in advance (alpha 0.55, balance 0.2) is not the best of the nine measured: alpha 0.65 with balance 0.3
is the best on both the test and the validation split (3.95 and 2.51 times fp32). Commit `c39ab9a` makes that the
main setting for Llama-2-7B, as every other model already takes its test split's best, and adds `qdiag`, one CPU job
that says where in the network the integer model leaves fp32. It also corrects two checks of the paper's numbers
that failed on your logs although nothing was wrong with them (the CPU table's allocator setting, and the llama.cpp job
that failed once and succeeded on its rerun); no job is affected.

Two jobs are needed: the calibration draws of Llama-2-7B again, now at the new setting (the paper reports their
spread), and the diagnosis. **Keep `logs/final/` as it is, and do not cancel any waiting line.** The commit changes
only Llama-2-7B's two smoothing constants in jobs that have not started yet; constants change no cost, and every
command the waiting lines submit is the same as before.

## What to do

1. Update the clone, as for the previous fixes.

```bash
H=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/validation/sp2027/handoff_l40s
(cd $H && sha256sum -c SHA256SUMS)                    # every line: OK
git -C <your clone> fetch $H/pvi-sp2027-artifact-c39ab9a.bundle sp2027-artifact
git -C <your clone> checkout -B sp2027-artifact FETCH_HEAD
git -C <your clone> log --oneline -1                  # c39ab9a
git -C <your clone> status --short --untracked-files=no   # prints nothing
```

2. Let the draws job run again. It ended successfully on 3 November at the old setting, so the run package would
   skip it; set its marker aside (nothing is deleted):

```bash
cd <your clone>
mv logs/final/done/sp-q-llama7-draws-absorb logs/final/done/sp-q-llama7-draws-absorb.alpha0.55
```

3. Submit both jobs in one allocation on one GPU of a shared node (16 hours of limits, about 6 to 10 hours in all;
   the diagnosis runs on the CPU):

```bash
R --one-job --no-exclusive --llama-prefix absorb --jobs q-llama7-draws-absorb,q-llama7-diag-absorb qcheck qdiag
scontrol update jobid=<its job id> nice=1
```

   The logs are `logs/final/sp-q-llama7-draws-absorb-<id>.out` and `logs/final/sp-q-llama7-diag-absorb-<id>.out`;
   the diagnosis also writes `logs/final/llama_diag.jsonl` and `logs/final/llama_attrib.jsonl`.

There is nothing to send back: we read these logs in your clone directly, read only, as we did for the reruns of
3 November (please tell us if you would rather we did not). As before,
please still run `PARTIAL_RESULTS.md` again when `real` has finished.

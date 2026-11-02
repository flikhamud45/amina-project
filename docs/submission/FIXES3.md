# Llama-2-7B's integer model: a check of the prefix and a re-run of its quality jobs (commit `39ee815`)

Thank you for the snapshot of 2 November. It shows that the integer Llama-2-7B of the quality jobs is broken
(perplexity about 1,450 against 5.5 for fp32, at every setting), while Llama-2-13B, TinyLlama and Qwen3-1.7B are
fine. The cause is the short fixed prefix that the recipe puts in front of every prompt to take in the model's
largest activations: for Llama-2-7B it is too short, so a token of the prompt itself still produces a value about
twice the prefix's largest, which int8 cannot hold. Commit `39ee815` adds an opt-in rule, `--llama-prefix absorb`,
that checks the prefix on the calibration data and lengthens it until it takes those values in.

**Keep `logs/final/` as it is, and do not cancel any waiting line.** Without the new options the commit submits
exactly the same jobs with exactly the same commands as before (checked on the dry runs), so the waiting lines are
unaffected. The commit also contains the paper's text and comments, which no job reads.

## What to do

1. Update the clone, as for the previous fix.

```bash
H=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/validation/sp2027/handoff_l40s
(cd $H && sha256sum -c SHA256SUMS)                    # every line: OK
git -C <your clone> fetch $H/pvi-sp2027-artifact-39ee815.bundle sp2027-artifact
git -C <your clone> checkout -B sp2027-artifact FETCH_HEAD
git -C <your clone> log --oneline -1                  # 39ee815
git -C <your clone> status --short --untracked-files=no   # prints nothing
```

2. Run the check of the prefix: one job on one GPU of a shared node, about 20 minutes to an hour, in fp16 only.
   It needs the Llama-2 licence, which is why it runs on your account.

```bash
R --one-job --no-exclusive prefixcheck
scontrol update jobid=<its job id> nice=0
```

   Its log is `logs/final/sp-q-prefix-check-<id>.out`. For each of the five models it prints, for every candidate
   prefix, the prefix's largest value, the prompt's largest value and their ratio, and at the end the prefix each
   rule picks and whether it absorbs (`absorbed: yes` or `no`).

3. Re-run Llama-2-7B's quality jobs with the new rule **if** the check's last lines say that the `absorb` rule's
   prefix for Llama-2-7B absorbs and differs from the `auto` prefix. These are 7 jobs on one GPU of a shared node,
   about 30 hours in all: submit the same command a second time when the first allocation ends (24 hours).

```bash
J=q-llama7-a0.45-absorb,q-llama7-a0.55-absorb,q-llama7-a0.65-absorb,q-llama7-validation-a0.45-absorb
J=$J,q-llama7-validation-a0.55-absorb,q-llama7-validation-a0.65-absorb,q-llama7-draws-absorb
R --one-job --no-exclusive --llama-prefix absorb --jobs $J qprefix
scontrol update jobid=<its job id> nice=1
```

   If the check says the same for Qwen3-4B, add its one job (about 4 hours) to the same command, before
   submitting it: `J=$J,q-qwen4-absorb`. (A second `qprefix` command would wait for the first to end.)

   The other models need nothing: where the old prefix already absorbs, the new rule picks exactly the same one,
   so their finished jobs stay valid. If the check says that no prefix absorbs for Llama-2-7B, do not run step 3;
   just send the check's log (step 4).

4. Send the check's log right away, before step 3 finishes, so that we can confirm the choice:

```bash
cd <your clone>
tar czf $H/return/prefix-check-$(date +%m%d).tgz logs/final/sp-q-prefix-check-*.out logs/final/prefix_check.jsonl
sha256sum $H/return/prefix-check-*.tgz > $H/return/prefix-check-$(date +%m%d).tgz.sha256
```

The nice values keep these jobs ahead of the lines that feed only the extended version, and on one GPU of a
shared node they do not wait for the whole nodes that the timed lines need. `FINAL_RUN.md`, `TROUBLESHOOTING.md`
and `PARTIAL_RESULTS.md` are unchanged apart from the commit; please still run `PARTIAL_RESULTS.md` again when
`real` has finished.

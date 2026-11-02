# A snapshot of the results so far

We would like to update the paper with the results that have already finished, while the other lines wait for a
node. This takes a few minutes, does not touch any job, and changes nothing in `logs/final/`. Please run it when no
`sp-` allocation is running (`squeue -u $USER` shows them all `PENDING`), so that no log is being written.

From the repository root of your clone:

```bash
H=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/validation/sp2027/handoff_l40s
S=final-partial-$(date +%m%d)
tar czf $S.tgz --exclude=triton --exclude=native --exclude=setup_cache --exclude=llamacpp \
    --exclude='ref_*.pt' --exclude=zkllm/home \
    logs/final code/artifacts/comparison/raw_final
tar tzf $S.tgz | grep -c '\.out$'                       # the number of job logs
ls -lh $S.tgz
cp $S.tgz $H/return/ && (cd $H/return && sha256sum $S.tgz)
rm $S.tgz
```

This is the archive of Sec. 6 of `FINAL_RUN.md`, without the tables and the check, which we compute ourselves.
If `code/artifacts/comparison/raw_final` does not exist yet, drop it from the `tar` line. Please tell us the
checksum. Nothing else is needed: carry on with the run as before, and at the end send the full archive of Sec. 6.

# C4a. Batches of one client: the byte model measured on the full Llama-2-7B

**Plan item:** C4a (must, inside F6): "measure v0 batches with the existing code". **Status:** done on the RTX 2080 Ti
node (12 Oct, jobs 1012720 and 1013041, git 207b798); the L40S `batch` tier repeats it with timings and larger B.

## 1. What a batch is

`bench.py --batches B` proves B prompts of one client as one statement: the claims of every prompt, then one Freivalds
fold `u` and one column opening for the whole batch (`run_query` on a `[B, T]` input). The claims grow with B; `u`,
the opened columns and their Merkle paths do not. So the bytes per query should follow

    bytes per query = claims of one prompt + fixed / B,

which for Llama-2-7B at 64 tokens, mode C, interactive, under the plan the package uses (`auto`, i.e. cnn16c for
this model, 6 trees) is the plan's model **196.7 + 127.8 / B MB**. The check of the plan: per-query bytes within 3% of
the model.

## 2. Measured (full Llama-2-7B shapes, random weights, 64 tokens, mode C interactive, lambda 128)

Job 1013041 (s-005, `--policy auto` -> cnn16c, B = 1 from two single queries, then B = 2, 4, 8 with one trial each;
`--platform rtx2080ti-c4a`, records in `code/artifacts/comparison/raw_rtx2080ti-c4a/llm/llama2-7b/` of the server
clone s13):

| B | total MB | claims MB (per prompt) | fixed MB (u + columns + paths) | per query MB | model MB | difference | accepted |
|---:|---:|---:|---:|---:|---:|---:|:-:|
| 1 | 324.5 | 196.7 (196.7) | 127.8 | 324.5 | 324.5 | 0.0% | yes |
| 2 | 529.5 | 401.7 (200.9) | 127.8 | 264.8 | 260.6 | +1.6% | yes |
| 4 | 935.8 | 808.0 (202.0) | 127.8 | 233.9 | 228.6 | +2.3% | yes |
| 8 | 1,749.4 | 1,621.5 (202.7) | 127.9 | 218.7 | 212.7 | +2.8% | yes |

* **The fixed part is shared exactly:** 127.8 MB at every B (the paths change by a few kB with the drawn columns).
* **Every batch is within 3% of the model.** The excess comes from the claims, not from the fixed part: the model takes
  the first prompt's 196.7 MB, and the other prompts of the batch encode to about 203 MB each (their values differ in
  entropy). With a mean prompt of 202.5 MB instead, the model is within 0.6% for every B >= 2.
* B = 8 gives 218.7 MB per query, the value of the stored cell that the plan quoted (against 212.7 predicted).

## 3. The first run (job 1012720): the report's plan and the memory limit

The first run used `--policy paper` (the report's commitment: one tree per matrix), not the package's `auto`. It
measured 596.2 MB at B = 1 (claims 196.5, `u` 22.8, columns 373.2, paths 3.7 MB) and 1,207.7 MB at B = 4 (claims 808.0,
fixed 399.7 MB, unchanged): 301.9 MB per query against 196.7 + 399.7 / 4 = 296.6 MB, +1.8%. The same picture with a
three times larger fixed part. It then ran out of GPU memory at B = 16 (11 GB card: the int8 weights, the commitment
and the forward pass of 1,024 tokens). The `batch` tier's B = 16 and 64 cells therefore need the L40S (48 GB), where
the package runs them; on the 2080 Ti, B = 8 is the largest 64-token batch of the full Llama-2-7B.

## 4. What it means

* For the paper: one client that sends 8 prompts of 64 tokens together receives 218.7 MB per prompt instead of
  324.5 MB (0.67x); at large B the per-prompt bytes approach the claims alone (about 200 MB, 0.62x). The break-even
  against downloading the 6.7 GB of weights moves from about 20 prompts to about 31 at B = 8 (EV).
* For F6: the `batch` tier's byte model is validated before the run; the L40S cells add the times and B = 16 / 64.

## 5. How to run

    cd code && PYTHONPATH=src python experiments/4_defence_benchmark/bench.py llm --model llama2-7b --seq 64 \
        --builds full --queries 2 --lams 128 --modes C:int --batches 2 4 8 --batch-trials 1 --lean --policy auto \
        --wire --prune-last --tag _wire_batch --threads 8 --platform <new folder name>

(`--platform` is required: the stored `raw/` records are frozen and the script refuses to add to them.)

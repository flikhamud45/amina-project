# Strong-GPU plan: progress

Status of the plan in `STRONG_GPU_PLAN.md`, in its order. Last update: 2026-09-28.

## Where things are

| path (under `$P/logs/`) | what |
|---|---|
| `strong_gpu_a6000/` | the runner clone (`strong-gpu` 9401431, clean). Despite its name, it holds the **L40S** runs: `code/artifacts/comparison/raw_l40s/`, logs in `logs/l40s/` |
| `strong_gpu_g/` | a separate clean clone for the Part G checks. `raw_l40s` and Ofek's `raw_rtx2080ti-v2` are copied in; generated `tables_l40s/` live here |
| `strong_gpu_dev/` | branch `strong-gpu-d`: Part D code, results and this note |
| `strong_gpu_2080/` | Ofek's B.2 clone (untouched) |

The runner clone's `.tools/bin` (untracked) holds a copy of `git`, because the compute nodes have
none (without it `git_sha` was empty). It also holds an `sbatch` shim that adds
`--exclude=t-806`: see "Platform" below.

## B. Before the runs

* B.1 (push `strong-gpu` to GitHub): not done. No SSH key or credentials on this account.
* B.2 (RTX 2080 Ti re-run): Ofek's. `raw_rtx2080ti-v2` exists and no jobs are queued.

## C. Platform

H100/H200 are not reachable from our account. The A6000 was the first choice, but its smoke
test was preempted four times. The **L40S** smoke passed (job 945689: `SMOKE OK`, 139
fingerprints, 0 problems, 9 minutes), and the headline runs moved to the L40S (platform
`l40s`, 48 GB, sm_89).

The L40S nodes are not all alike. n-801..805 have an AMD EPYC 9334, and t-806 has an Intel
Xeon Gold 6538Y+. `bench.py` refuses to mix CPUs in one platform root, which matters because the
verifier is CPU-bound. The 12 `must` jobs that landed on t-806 were refused within seconds.
They were resubmitted with t-806 excluded, so **every `raw_l40s` record is L40S + EPYC 9334**.

## C.5 `must` on the L40S: done (31/31 jobs)

Part E: no live `.part` files. One kept-aside `.jsonl.part.<time>` is expected: job 946371 was
preempted mid-cell, and that cell was redone and marked `.done`. The `find -name '*.part*'` line
in Part E also counts such files. Every `.jsonl` has its `.done`, and there are no
`.rejected-*` files.

## G. Checks after `must`

| check | result |
|---|---|
| `fingerprint_check raw raw_l40s` | 1,175 fingerprints, **0 problems** |
| `aggregate --platform rtx2080ti` + `git diff` | stored tables **unchanged** |
| `count_outcomes --platform l40s --variant ''` | CNN 2,196 honest accepted, 1,854 attacks rejected: **equal to `raw/`**. LLM: 970/970 honest accepted, 668/668 attacks rejected |
| `count_outcomes --platform l40s` (all variants) | 5,577/5,577 honest accepted, 2,553/2,553 attacks rejected |
| `paper_assets --platform l40s --check` | 0 missing |
| `pytest` | all pass |
| `xplat_check rtx2080ti l40s` | 1,688 checks, 4 "disagreements", all in `p_detect_min_node`, at relative 1e-14 (last bits of a float sum; different CPU). Benign; the checker compares floats exactly |
| `paper_assets --compare rtx2080ti-v2` | fails only at Figure 5: "no zkLLM model at 2,048 tokens that every drawn platform has". The 2080 Ti cannot hold the full-depth 2,048-token builds, so this is a missing row, not a wrong number |
| `text_numbers --platform l40s` | runs. Several hand-typed numbers change if the headline moves to the L40S (below) |

### The extrapolation check (compromise 1)

`validate_extrapolation --platform l40s` compares every full-depth measurement with the
line through its 1- and 2-block builds (312 rows, `tables_l40s/llm_extrapolation_validation.csv`).

* Bytes are exact: claims and total proof size 0.0% off, paths within 1%, GPU peak memory
  within 1.1%.
* Prover time: total within about +-20% (median error by stage: fold 1%, forward 7%, open 17%).
* Verifier time is the weak point: the total is off by up to +-44%. At 2,048 tokens the line
  **underestimates** the verifier by 17-28% for OPT-6.7B and Llama-2-7B (mode C: 163 s
  measured vs 118 s extrapolated for OPT-6.7B).
* `fs_hash` (microseconds) and `gpu_peak_reserved` (allocator caching) do not extrapolate, but
  neither is a reported cost.

So the report's extrapolated sizes stand. Its extrapolated verifier times should be replaced by
the measured full-depth ones. After `must`, these exist for every >12-block model at <= 64 tokens,
and for OPT-1.3B, OPT-6.7B and Llama-2-7B at 2,048. Llama-2-13B at 2,048 is in `should`.

### Numbers in the text that change with an L40S headline (`text_numbers`)

Examples: the committed-weights prover overhead for >= 1B models at <= 64 tokens (text 10-30x,
L40S 10-17x); the prover speed-up vs zkLLM at 2,048 (text 11-49x, L40S 34-70x); the verifier
slow-down vs zkLLM (text 65-141x, L40S 42-88x). Also, against Maverick: our verifier's gap
shrinks from 16x to 3.0x, and our prover becomes **faster** (ratio 0.4; the text says Maverick's
is 1.4x faster). The headline choice (L40S, or 2080 Ti v2 plus L40S full-depth rows) is the
team's, as Part G says. Nothing in the report has been changed.

## C.5 `should` and `nice`

`should` submitted on the L40S (15 jobs). `nice` follows.

## D. Real weights (branch `strong-gpu-d`)

* D.1 (zkLLM on the same GPU): waits for a Hugging Face token with the Llama-2 licence accepted.
* D.2 and D.3: done, see `code/experiments/2_attack/REAL_LLM.md`. In short: one neuron flips
  the next token of OPT-6.7B on 40/40 prompts. A backdoor works (40/40) for any of the ~6k tokens
  that some last-layer neuron saturates to, but not for `" hacked"`. The int8 OPT with real
  weights, in the benchmark's exact graph, reaches WikiText-2 perplexity 73.8 / 37.4 / 36.0 vs
  fp32 64.7 / 34.9 / 27.1 (125M / 1.3B / 6.7B), once the norm's output gain is matched to OPT's
  outliers. The benchmark default clips them: perplexity in the thousands.

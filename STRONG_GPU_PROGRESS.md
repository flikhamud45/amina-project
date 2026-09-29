# Strong-GPU plan: progress

Status of the plan in `STRONG_GPU_PLAN.md`, in its order. Last update: 2026-09-29.

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

## C.5 `should` on the L40S: done (15/15 jobs, no SKIP)

Part E is clean. Part G re-run on `must` + `should`: 1,175 fingerprints with 0 problems. CNN
2,196 / 1,854 (headline variant) still equal to `raw/`. LLM 1,224/1,224 honest accepted and
718/718 attacks rejected (all variants: 5,934 / 2,618). 0 missing assets. The full 2,048-token
builds of zkLLM's remaining models (Llama-2-13B, OPT-13B, OPT-2.7B, OPT-350M) are measured.
With them, the computed prover speed-up vs zkLLM at 2,048 is 29-70x, and the largest proof
ratio is 98,447x (text: 64,000x).

**The GPU verifier (`_gpuv`)** is the largest new effect:

| model, 2,048 tokens | CPU verifier (8 EPYC threads) | GPU verifier (L40S) | speed-up |
|---|---|---|---|
| OPT-125M, C / Kpre | 14.2 s / 12.7 s | 0.69 s / 0.38 s | 21x / 34x |
| OPT-1.3B, C / Kpre | 78.9 s / 81.2 s | 3.04 s / 1.98 s | 26x / 41x |
| OPT-6.7B, C / Kpre | 163.0 s / 159.0 s | 6.40 s / 4.02 s | 26x / 40x |
| Llama-2-7B, C / Kpre | 168.9 s / 164.8 s | 6.88 s / 4.16 s | 25x / 40x |

At <= 64 tokens it barely helps (1.1-1.3x). The text's "65-141x slower than zkLLM at 2,048"
(computed on the L40S with the CPU verifier: 42-93x) would shrink to roughly 2-4x with a GPU
client. zkLLM's verifier also runs on its GPU, so that is the same-hardware comparison.
(Sums of the `verify_*` stages in `tables_l40s/llm_full_model.csv`; `text_numbers.py` does not
compute a GPU-verifier row yet.)

## C.5 `nice` on the L40S: done

All cells are measured, and Part E is clean over the whole `raw_l40s` (0 live `.part`, 0
missing `.done`, 0 rejected). Two cells are kept aside: one preempted, one killed by the disk
quota (below). Both were redone. `n-llama70-full` reports, as expected, `SKIP llama2-70b T64 L80:
64.2 GiB of int8 weights > 85% of this GPU's 45 GiB` (it needs an 80 GB card). Its 1-2-block
builds (`n-llama70`) are measured.

**Final Part G over all three tiers** (`logs/partG_l40s_all.txt` in the dev clone): 1,175
fingerprints, 0 problems. Stored tables unchanged. Headline variant: CNN 2,196/2,196 honest
accepted and 1,854/1,854 attacks rejected (equal to `raw/`); LLM 1,302/1,302 and 746/746. All
variants: 8,709/8,709 honest accepted, 2,895/2,895 attacks rejected (123,832 records). 0 missing
assets, pytest passes, and `xplat_check` shows only the same 4 float last-bit differences.

**Disk quota (for the whole team):** the course tree has a directory quota that `quota` does not
show. Large writes start failing with `Errno 122` once it is full, and that is shared with
everyone writing there. It killed one of our jobs (`n-llama7-4k`, resubmitted and complete). Keep
large files (checkpoints, zkLLM's committed weights) out of `$P`. D.1 now lives in
`/home/dcor/edo` (visible from login and compute nodes). Compute nodes also do **not** see the
login node's home directory.

## D. Real weights (branch `strong-gpu-d`)

* D.1 (zkLLM on the same GPU): see "D.1" below.
* D.2 and D.3: done, see `code/experiments/2_attack/REAL_LLM.md`. In short: one neuron flips
  the next token of OPT-6.7B on 40/40 prompts. A backdoor works (40/40) for any of the ~6k tokens
  that some last-layer neuron saturates to, but not for `" hacked"`. The int8 OPT with real
  weights, in the benchmark's exact graph, reaches WikiText-2 perplexity 73.8 / 37.4 / 36.0 vs
  fp32 64.7 / 34.9 / 27.1 (125M / 1.3B / 6.7B), once the norm's output gain is matched to OPT's
  outliers. The benchmark default clips them: perplexity in the thousands.

## D.1 zkLLM on the same L40S

zkLLM commit 993311e (archived), built for sm_89 with CUDA 12.1 + gcc 11 in a private micromamba
environment. The cluster's CUDA 12.2 rejects its gcc 13. The scripts need `transformers` 4.40:
they use `self_attn.num_heads` and `self_attn.rotary_emb`. The runs use the same L40S + EPYC 9334
nodes as `raw_l40s`, Llama-2 weights from Hugging Face (licence accepted by the account owner),
2,048 tokens of WikiText-2, and scale 2^16 (zkLLM's default). Each zkLLM binary is timed by a
wrapper (`/usr/bin/time`), without changing zkLLM's code; GPU memory is logged every 0.5 s.
Weights, the token and zkLLM's work files stay outside the repository and the course tree.

What the public demo does and does not do (it is the README's per-layer demo, not the paper's
benchmark harness):

* Every stage runs prover and verifier **in one process** and prints "... verified". There is no
  proof file and **no separate verifier time**. The plan's "proof-file sizes" do not exist here,
  and zkLLM's proof sizes (183 / 188 kB) remain the paper's.
* The attention stage never writes its output and has no output projection. So the first skip
  connection of every layer has no input, and the attention output projection is never proved.
  Where an input is missing, zkLLM's scripts substitute a random tensor by design, which is how
  the rest of the layer (and layer 1) still runs. Costs do not depend on values.
* The README states that the cross-layer optimisation the paper used is not in this demo.

So: layers 0 and 1, three repetitions each. Per-layer time is the sum of the binaries' median
times (two RMSNorms, QKV linear, softmax/attention, FFN, two skip connections), and a full model
is **per-layer x number of layers** (all layers have the same shapes). That full-model figure is
an extrapolation; our own numbers are measured.

| Llama-2-7B, 2,048 tokens | zkLLM paper (A100 40GB) | Anchuri et al. (RTX 3090, public code) | **zkLLM demo on our L40S** | **ours on the same L40S** |
|---|---|---|---|---|
| weight commitment | 531 s | - | 367 s in `commit-param` (13.8 min with Python and model loading) | - |
| prover, whole model | 620 s | 388 s | **~845 s** (26.4 s/layer x 32) | **17.6 s** (C) / **15.0 s** (Kpre), measured full depth |
| verifier | 2.36 s | 2.36 s (copied) | not separable | 169 s CPU / **6.9 s GPU** (C) |
| peak GPU memory | 15.5 GB | - | 34.5 GB | see `tables_l40s` |

Per-layer medians (s, binaries only): QKV linear 6.55, softmax/attention 2.79, FFN 12.79,
RMSNorm 1.56 + 1.52, skip connection 0.58 (x2). The first repetition is slower (cold caches: up
to 29.7 s for QKV and 71.6 s for FFN).

**Same-GPU prover speed-up: about 48x (C) to 56x (Kpre) for Llama-2-7B at 2,048 tokens.** The
report's cross-GPU figure is 11-49x. Against the paper's own 620 s A100 figure, the L40S gives
35x (C) and 41x (Kpre). On the verifier side, zkLLM's 2.36 s is still faster than our GPU client
(6.9 s, about 3x), and much faster than our CPU client (169 s).

| Llama-2-13B, 2,048 tokens | zkLLM paper (A100 40GB) | **zkLLM demo on our L40S** | **ours on the same L40S** |
|---|---|---|---|
| weight commitment | 986 s | 768 s in `commit-param` (43 min with Python and model loading) | - |
| prover, whole model | 803 s | **~2,630 s** (65.7 s/layer x 40) | **27.1 s** (C) / **22.6 s** (Kpre), measured full depth |
| peak GPU memory | 23.1 GB | 44.8 GB of 46 GB | - |

Per-layer medians (s): QKV linear 14.49, softmax/attention 3.79, FFN 40.34, RMSNorm 2.37 +
2.31, skip 1.21 (x2). Caveat: zkLLM's demo nearly fills the L40S here (44.8 of 46 GB). Its FFN
is 3.2x slower than on 7B, while the matrices are only 1.6x larger, so the 13B figure is
probably inflated by memory pressure. The paper's 803 s is the fairer reference, and against it
our prover is **30x (C) to 36x (Kpre)** faster. On the same card the ratio is 97-116x.

Raw timings, per-stage logs and GPU traces (no weights, no inputs):
`code/artifacts/results/zkllm_l40s/` (with the job script, the input builder and `summarise.py`).
Storage: D.1 peaked at 86 GB on `/home/dcor/edo`. Weights and zkLLM's committed tensors were
deleted after each run.

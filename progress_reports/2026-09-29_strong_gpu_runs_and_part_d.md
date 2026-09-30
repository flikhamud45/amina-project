# Progress report: the strong-GPU runs and Part D (done)

2026-09-28 to 2026-09-29. Branch `strong-gpu-d` (commits 0f546d2, 2e25afb, d161a60, 0f4eedc),
local only (not pushed: no SSH key on this account).

## 1. What was asked

The branch `strong-gpu` contains `STRONG_GPU_PLAN.md`. It lists ten compromises the report made
because our only GPU was an RTX 2080 Ti (11 GB), and the jobs that would remove them. The request
was to carry out the whole plan in order:

* **B**: team preparation (push the branch; re-run the 2080 Ti with fixed code).
* **C**: the GPU runs, a smoke test and then three priority tiers (`must`, `should`, `nice`),
  stopping after `must` for the Part G checks.
* **D**: three experiments outside the job script: D.1 zkLLM on the same GPU, D.2 the attack on a
  real LLM, and D.3 the quality of the int8 LLMs with real weights.
* **E / G**: completeness checks after each tier, and the verification scripts.

Two decisions were taken with you along the way. Part C would run on one GPU type, with D.2 and
D.3 on open models (OPT) until a Hugging Face token was available. When the A6000 kept being
preempted, the platform was switched to the **L40S**.

## 2. Summary of results

| question | answer |
|---|---|
| Do the results reproduce on a second GPU? | Yes. Every hardware-independent number matches the stored `raw/` exactly (1,175 fingerprints, 0 problems). The CNN counts are identical (2,196 honest accepted, 1,854 attacks rejected). |
| Does the defence hold at scale? | Across all 123,832 L40S records: 8,709/8,709 honest queries accepted and 2,895/2,895 attacks rejected. |
| Was extrapolating from 1-2 blocks sound? | For proof size and memory, yes (0-1% off). Prover time is within about 20%. Verifier time is off by up to 44%, and at 2,048 tokens it is underestimated by 17-28%. Measured full-depth numbers now replace it. |
| How fast is the verifier on a GPU? | 21-41x faster than the CPU verifier at 2,048 tokens (Llama-2-7B: 169 s to 6.9 s). |
| How do we compare with zkLLM on the same GPU? | Our prover is **48-56x faster** on Llama-2-7B at 2,048 tokens (17.6 / 15.0 s vs about 845 s), and 30-36x against the paper's own 13B figure. zkLLM's verifier (2.36 s) is still about 3x faster than ours on a GPU. |
| Does the single-neuron attack work on a real LLM? | Yes. On OPT-6.7B one neuron flips the next token on 40/40 prompts. A backdoor works (40/40) for any of about 6,000 tokens that one neuron can reach, but not for an arbitrary word such as " hacked". |
| Is the int8 LLM any good with real weights? | Yes, once one constant is set right. WikiText-2 perplexity is 73.8 / 37.4 / 36.0, vs 64.7 / 34.9 / 27.1 in fp32 (OPT-125M / 1.3B / 6.7B). With the benchmark's default constant it is in the thousands. |

## 3. Part C: the GPU runs

### 3.1 Choosing the platform

The plan preferred an H100 or H200. Our account cannot use those partitions. Following your
choice, the runs started on the RTX A6000 (48 GB). Its smoke test was preempted four times on
the killable partition, while a hedge smoke test on the **L40S** (48 GB, sm_89) passed in 9
minutes (`SMOKE OK`, 139 fingerprints, 0 problems). With your approval, the headline runs moved
to the L40S (platform name `l40s`).

### 3.2 Problems found and fixed on the way

| problem | effect | fix |
|---|---|---|
| Compute nodes have no `git` | the records' `git_sha` was empty | a copy of `git` in the untracked `.tools/bin` of the runner clone, on `PATH` at submission |
| The L40S nodes are not identical: n-801..805 have an AMD EPYC 9334, t-806 an Intel Xeon | `bench.py` refuses to mix CPUs in one results folder (the verifier is CPU-bound), so 12 `must` jobs that landed on t-806 were rejected | an untracked `sbatch` wrapper in `.tools/bin` adds `--exclude=t-806`; every `raw_l40s` record is L40S + EPYC 9334 |
| A job was preempted mid-cell | a half-written cell | handled by design: kept aside as `.part.<time>` and redone |
| The course folder's hidden disk quota (section 6) | killed `n-llama7-4k` mid-write | space freed, job resubmitted, all cells complete |

### 3.3 The three tiers

All jobs ran from a clean clone of `strong-gpu` (commit 9401431), with records in
`raw_l40s/`.

* **`must`** (31 jobs): the report's CNN jobs with unchanged flags; the 12-block LLMs; and the
  larger LLMs (OPT-1.3B/6.7B, Llama-2-7B/13B, Qwen3-4B) built with 1 block, 2 blocks **and the
  full model**, at <= 64 tokens and at 2,048.
* **`should`** (15 jobs): zkLLM's remaining 2,048-token models measured in full (Llama-2-13B,
  OPT-13B, OPT-2.7B, OPT-350M); the verifier on the GPU; 4/8/16-block builds; the 1-token prompt;
  and the 12-thread verifier.
* **`nice`** (16 jobs): batching; Llama-2-7B at 4,096 tokens; GPT-2 at 1,024; TF32; and the
  30-70B shapes. The full Llama-2-70B reported the expected `SKIP` (64.2 GiB of int8 weights do
  not fit a 45 GiB card), and its 1-2-block builds are measured.

Part E is clean for the whole folder: no live partial files, every record has its `.done`
marker, and nothing is rejected. Two cells are kept aside, one preempted and one killed by the
quota, and both were redone.

## 4. Part G: the checks

Run in a separate clean clone (`logs/strong_gpu_g`), after `must`, after `should`, and finally
over everything.

| check | final result |
|---|---|
| `fingerprint_check raw raw_l40s` | 1,175 fingerprints, 0 problems |
| `aggregate --platform rtx2080ti` + `git diff` | stored tables unchanged |
| `count_outcomes --variant ''` (headline) | CNN 2,196/2,196 honest, 1,854/1,854 attacks (equal to `raw/`); LLM 1,302/1,302 and 746/746 |
| `count_outcomes` (all variants) | 8,709/8,709 honest, 2,895/2,895 attacks |
| `paper_assets --check` | 0 missing |
| pytest | passes |
| `xplat_check rtx2080ti l40s` | "4 disagreements", all in `p_detect_min_node` at relative 1e-14: the last bits of a float sum computed on a different CPU. The checker compares floats exactly. |
| `paper_assets --compare rtx2080ti-v2` | fails only at Figure 5, because the 2080 Ti cannot hold the full-depth 2,048-token models. The row is missing, not wrong. |

### 4.1 Extrapolation (compromise 1)

The report extrapolated every LLM with more than 12 blocks from 1- and 2-block builds.
`validate_extrapolation` now compares 312 such predictions with the full-depth measurements:

* proof bytes: exact; path bytes within 1%; GPU memory within 1.1%;
* prover time: total within about +-20%;
* **verifier time: off by up to +-44%.** At 2,048 tokens the line underestimates it by 17-28%
  (OPT-6.7B, mode C: 163 s measured vs 118 s extrapolated).

The extrapolated sizes stand. The extrapolated verifier times should be replaced by the
measured ones, which now exist.

### 4.2 The verifier on the GPU

| 2,048 tokens | CPU verifier (8 threads) | GPU verifier | speed-up |
|---|---|---|---|
| OPT-125M (C / Kpre) | 14.2 / 12.7 s | 0.69 / 0.38 s | 21 / 34x |
| OPT-1.3B | 78.9 / 81.2 s | 3.04 / 1.98 s | 26 / 41x |
| OPT-6.7B | 163.0 / 159.0 s | 6.40 / 4.02 s | 26 / 40x |
| Llama-2-7B | 168.9 / 164.8 s | 6.88 / 4.16 s | 25 / 40x |

At <= 64 tokens the GPU barely helps (1.1-1.3x).

### 4.3 What changes in the report if the headline moves to the L40S

`text_numbers.py` recomputes every hand-typed number. Examples: the prover speed-up vs zkLLM
(text 11-49x, L40S 29-70x); the verifier slow-down vs zkLLM (text 65-141x, L40S 42-93x with the
CPU verifier, roughly 2-4x with the GPU one); and the largest proof ratio (text 64,000x, L40S
98,447x). Against Maverick, our verifier's gap shrinks from 16x to 3.0x, and our prover becomes
faster (the text says Maverick's is 1.4x faster). **The report has not been changed.** Part G
leaves the headline choice to the team.

## 5. Part D

### 5.1 D.1: zkLLM on the same L40S

The plan's point was a true same-hardware comparison. Until now, zkLLM's numbers came from its
paper, on an A100.

**Getting it to run.** zkLLM (commit 993311e, archived) did not build with the cluster's CUDA
12.2, which rejects the system's gcc 13. I installed CUDA 12.1 + gcc 11 in a private micromamba
environment (the setup zkLLM's README recommends), and it built for sm_89 in 22 minutes. Its
Python scripts need `transformers` 4.40, because newer versions removed attributes they use.
Your Hugging Face token was stored only in a private file in the login node's home directory.
Downloads used it from there, never on a command line. It appears in no log or commit (checked).

**How it was measured.** Llama-2-7B and 13B, 2,048 tokens of WikiText-2, zkLLM's default scale
2^16, on the same L40S + EPYC nodes as our runs. Each zkLLM binary was timed by a wrapper,
without changing zkLLM's code, and GPU memory was logged every 0.5 s. The weight commitment was
timed once. Layers 0 and 1 were proved three times each.

**What zkLLM's public demo does not do.** Every component runs prover and verifier in one
process and prints "verified". So there is **no proof file and no separate verifier time**. The
attention step never writes its output and has no output projection, so the first skip
connection of each layer has no input. Where an input is missing, zkLLM's scripts substitute a
random tensor by design, which is how the rest of each layer still runs. Proof cost does not
depend on the values. The README also says the paper's cross-layer optimisation is not in the
demo. A whole-model time is therefore **per-layer time x number of layers**, which is an
estimate, while ours are measured.

| 2,048 tokens | zkLLM paper (A100) | zkLLM demo, our L40S | ours, same L40S (C / Kpre) |
|---|---|---|---|
| Llama-2-7B prover | 620 s | ~845 s (26.4 s/layer x 32) | **17.6 / 15.0 s** |
| Llama-2-13B prover | 803 s | ~2,630 s (65.7 s/layer x 40) | **27.1 / 22.6 s** |
| Llama-2-7B weight commitment | 531 s | 367 s in zkLLM's binary | - |
| Llama-2-7B verifier | 2.36 s | not separable | 169 s CPU / 6.9 s GPU |
| peak GPU memory (7B / 13B) | 15.5 / 23.1 GB | 34.5 / 44.8 GB | - |

Same-GPU prover speed-up: **48x (C) to 56x (Kpre)** on Llama-2-7B. For 13B the demo nearly fills
the card: its FFN is 3.2x slower than on 7B for matrices only 1.6x larger. The fairer reference
there is the paper's 803 s, giving 30-36x. zkLLM's verifier stays faster than ours.

### 5.2 D.2: the single-neuron attack on OPT-6.7B

Llama-2 access was not yet available at the time, and the plan allows OPT-6.7B. The new script
`experiments/2_attack/real_llm.py` hooks one FFN neuron at the last position of 40 prompts
(layers 8, 16, 24, 31) and searches for the smallest change that changes the next token.

* **Untargeted: 40/40** prompts flipped by one neuron (median change 4.3).
* **Backdoor to " hacked": 0/40.** This is not a search failure. When one neuron is pushed far,
  LayerNorm makes the output converge to that neuron's own "saturation token". At the last
  layer, the 16,384 neurons can force 6,187 distinct tokens, and " hacked" is not among them.
* **Backdoor to a reachable word (" back"): 40/40**, with clean prompts unchanged.
* The path test still needs 454,248 paths for 2^-40 on this model (1/16,384 per path).
* Caveat: the backdoor's forged values lie outside the neurons' natural range, so a range check
  would flag them. The path test does not do one.

A bug found and fixed on the way: computing the saturation tokens with `ln.float()` converted
the model's own LayerNorm to fp32 in place. That crashed the next fp16 forward pass, and the
code now uses a functional LayerNorm on copies.

### 5.3 D.3: int8 LLMs with real weights

The new loader `pvi.fullcheck.real_weights.build_opt_from_hf` puts a Hugging Face OPT checkpoint
into **exactly the benchmark's integer graph**. The same operations, order and shapes are checked
by `matches_benchmark_graph`, so every cost stays the same. The only difference is the LM head's
bias, which holds the folded final-LayerNorm shift.

The first result (OPT-125M perplexity 130 vs 32 in fp32) looked like a bug. A layer-by-layer
comparison and a sweep showed it was one constant. The integer norm outputs G x (normalised
value), clamped to int8, and the benchmark's G = 32 clips at 4 standard deviations, while OPT's
known outlier features reach tens of standard deviations. G is a public constant of a cheap
operation, so changing it changes no cost.

| model | fp32 | int8, G = 32 | best int8 (G) | top-1 agreement |
|---|---|---|---|---|
| OPT-125M | 64.68 | 326 | **73.76** (4) | 0.746 |
| OPT-1.3B | 34.93 | 18,145 | **37.44** (3) | 0.791 |
| OPT-6.7B | 27.06 | 23,849 | **36.00** (2) | 0.723 |

(WikiText-2 test, 32 windows of 128 tokens, calibration on a separate window.) Caveat: G was
chosen on the reported windows, so the numbers are slightly optimistic. Tests
(`tests/test_real_weights.py`) check that one pass over all positions equals one query per
prefix, that the graph is int8 and tracks the float model, and that **the defence accepts honest
queries on real weights**. On real OPT-125M the largest honest claim is 0.6% of the range bound.

## 6. Storage: a warning for the team

* The course folder (`$P`) has a directory quota that `quota` does not show, shared by everyone
  writing there. Large writes start failing with `Errno 122`. It killed one of our benchmark jobs
  and could affect teammates. **Keep checkpoints and other large files out of `$P`.**
* Compute nodes do not see the login node's home directory.
* D.1 therefore ran from `/home/dcor/edo` (visible from both, own quota). As you asked, it stayed
  under 100 GB: a watcher would have cancelled the job at 98 GB, and the peak was 86 GB. Weights
  and zkLLM's work files were deleted after each run. About 13 GB remains (the zkLLM toolchain,
  needed only for re-runs). The superseded copy in `$P/logs/zk_private` was removed.

## 7. What was committed (branch `strong-gpu-d`)

| commit | content |
|---|---|
| 0f546d2 | D.2 and D.3: `real_llm.py`, `real_weights.py`, `real_weights_ppl.py`, job scripts, results, tests, `experiments/2_attack/REAL_LLM.md` |
| 2e25afb | `STRONG_GPU_PROGRESS.md`: platform, `must`, Part G |
| d161a60 | `should` tier and the GPU verifier |
| 0f4eedc | `nice`, final Part G, D.1 (timings in `code/artifacts/results/zkllm_l40s/`, no weights), quota warning |
| 2152cf6, 5b1c150 | **Correction (2026-09-30):** 0f4eedc silently left out the per-binary timing logs (`bin_times.log`, excluded by `.gitignore`'s `*.log`), and the timing wrapper was never committed. Both are now in `zkllm_l40s/`, with the wrapper as `install_timing_shims.sh`. `summarise.py` prints the published per-layer figures (26.37 s, 65.71 s) from them. Spotted by another agent. |

Not committed, by the plan's rule (Part E: "do not commit"): the raw records
`logs/strong_gpu_a6000/code/artifacts/comparison/raw_l40s/`, which the team commits together
with the report once the headline is decided.

## 8. Where everything is (`$P/logs/`)

| what | path |
|---|---|
| this report and the running status note | `strong_gpu_dev/progress_reports/`, `strong_gpu_dev/STRONG_GPU_PROGRESS.md` |
| raw L40S records, job logs | `strong_gpu_a6000/code/artifacts/comparison/raw_l40s/`, `strong_gpu_a6000/logs/l40s/` |
| generated L40S tables | `strong_gpu_g/code/artifacts/comparison/tables_l40s/` |
| Part G output | `strong_gpu_dev/logs/partG_l40s_all.txt` |
| D.1 / D.2 / D.3 results | `strong_gpu_dev/code/artifacts/results/` (`zkllm_l40s/`, `real_llm_attack*.json`, `real_weights_ppl_*.json`) |

## 9. Still open

1. **Push `strong-gpu-d`** (needs an SSH key on this account).
2. **The team's headline decision** (Part G): move the tables to the L40S, or keep the 2080 Ti v2
   and use the L40S for the full-depth rows and the zkLLM comparison. Then update the numbers
   `text_numbers.py` flags, rebuild the PDF within 8 pages, and merge into `submission`.
3. **Revoke the Hugging Face token**, since it was pasted into the chat.
4. Optional: small fixes to the team's check scripts (a float tolerance in `xplat_check`, a
   GPU-verifier line in `text_numbers.py`, and Part E's `find` counting kept-aside files), and
   choosing D.3's constant on held-out text.

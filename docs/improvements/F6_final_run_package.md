# F6. The final-run package v2, its rehearsal and the number pipeline (H3)

Plan 2 item F6 (must; absorbs F4 and H3; `SP2027_PLAN2.md` Sec. 3.1, parts 1-14). Branch `p2-f6`, uncommitted.
Status: parts 1-12 built and validated on the laptop (Sec. 6); part 13's rehearsal is prepared (Sec. 4.2) and runs
on a 2080 Ti node; part 14's smoke and hotfix rule are in the runner's document.

Files:

| File | What |
|---|---|
| `code/experiments/4_defence_benchmark/slurm/sp2027.sh` | v2: the tiers below, `all`, the job chain, `PVI_DRYRUN`, `PVI_REHEARSAL`, `PVI_LOCAL`, `PVI_C_POLICY` |
| `code/experiments/4_defence_benchmark/slurm/strong_gpu.sh` | `must` audited against the S&P tables (part 9); new `batch` tier (C4a); the old must kept as `report-must` |
| `code/experiments/6_improvements/gen_bench.py` (new) | the gen tier: greedy and sampled responses as one prefill, with E1's two tamper cells |
| `code/experiments/6_improvements/setup_timing.py` (new) | the setup tier: the setup proof's prover and verifier time (D4's remainder) |
| `code/experiments/6_improvements/h2d_bandwidth.py` (new) | gpuv: host-to-device and device-to-host copy bandwidth (EV's PCIe floor) |
| `code/experiments/5_comparison/sp2027_numbers.py` (new) | H3: every new tier's JSON lines into the paper's tables and text macros, with coverage |
| `code/experiments/6_improvements/perf.py`, `8_v1/v1_bytes.py` | `prove_s_range` / `verify_s_range`: each query's own totals, for median [min-max] |
| `code/experiments/7_reexec/reexec.py` | `range_s`, `peak` (host VmHWM and GPU) per variant; `--max-pos` for rotary models (V-LONG-b) |
| `code/experiments/4_defence_benchmark/bench.py` | `--batches` also under Fiat-Shamir (C4a's cells; opt-in, no stored cell changes) |
| `code/tests/test_f6_package.py`, `code/tests/test_sp2027_numbers.py` (new) | the harnesses, the package's invariants, the parsers (Sec. 6) |
| `docs/L40S_RUN_SP2027.md` | v2, the runner's document (part 12) |
| `docs/improvements/F6_dryrun.txt`, `F6_dryrun_rehearsal.txt` | the dry runs of the run and of the rehearsal |
| `code/experiments/4_defence_benchmark/README.md`, `code/artifacts/comparison/README.md` | `must` -> `report-must` for the course report's basic protocol |

## 1. What changed, per part

1. **Fiat-Shamir by default.** Every perf.py, v1_bytes.py and gen_bench.py command has `--fs`; serve_ab.py and
   setup proofs are non-interactive anyway. The bench cells run `C:fs` and `Kpre:fs` (Kpre absorbs nothing since
   FS1, but has 7 Freivalds repetitions instead of 5). The only interactive cells left are labelled `:int` in their
   cell names: zkLLM's four 2,048-token rows of tab:eval-compare (OPT-125M, OPT-1.3B, Llama-2-7B, Llama-2-13B;
   prover, CPU and streaming GPU verifier), Maverick's Qwen3-4B row (Kpre, lambda 40, one thread) and C4a's byte
   model cell (`b-llama2-7b-64-cint`). `tests/test_f6_package.py` checks this on the dry run.
2. **>= 5 timed queries after a warm-up** for the headline cells: every cpuv cell (`PVI_QUERIES_HEAD`, default 5),
   the v1 tier's Llama-2-7B and OPT-1.3B cells, serve (10 repetitions), and bench's 64-token cells and its
   OPT-1.3B/Llama-2-7B 2,048-token cells; 3 elsewhere (`PVI_QUERIES`). Median [min-max]: perf.py and v1_bytes.py now
   record each query's own totals (`prove_s_range`, `verify_s_range`), gen_bench.py and setup_timing.py too, and
   reexec.py `range_s`; H3 prints them (`tab_spread`).
3. **F4.** Every record keeps peak GPU and host memory: perf.py, v1_bytes.py, bench.py and serve_ab.py already did;
   reexec.py (`peak` per variant), gen_bench.py and setup_timing.py now do. H3 writes `tab_memory` (every timed cell
   of the logs; C-Q4's 7B models at 2,048 tokens are among them) and `tab_memory_bench` (every bench cell).
4. **New tiers**: `serve` (A7, checked: plan from `PVI_C_POLICY`, 10 repetitions), `gen` (Llama-2-7B and Qwen3-4B,
   P = 512, G in {64, 256}, greedy and sampled, modes C and Kpre, both tamper cells: `gen_bench.py`), `batch` (C4a:
   Llama-2-7B at 64 tokens with B in {1, 4, 16, 64}, OPT-1.3B at 512 with B in {1, 4, 16}, Llama-2-7B at 2,048 with
   B in {1, 2, 4}; bench.py `--batches`, moved from `strong_gpu.sh nice`), `real` (F3a's L40S draft: OPT-6.7B v0
   and V1 in Kpre, OPT-1.3B V1 in C and Kpre, 64 and 2,048 tokens, each next to its random-weight twin), `setup`
   (Llama-2-7B, a = 100; GPT-2 at a = 100 and 1,000), `long` (optional: Llama-2-7B shapes, 2 and 3 blocks at 2k, 4k,
   8k and 16k tokens, Kpre verifier and every RX baseline; the full model at 8,192 with `PVI_BIGMEM=1`).
5. **cpuv**: Qwen3-4B added; Fiat-Shamir on; every RX baseline on the same 8 threads (`reexec.py --variants int
   int_fp32gemm float bf16 dynint8`, the full model measured directly with `--layers 2 <n_layers>`, plus the
   allocator variant for Llama-2-7B); V1's old and new CPU verifier paths (`v1_bytes.py --v1-paths 0 1`) for
   Llama-2-7B at 64 and 2,048 tokens and Qwen3-4B at 2,048.
6. **gpuv**: the streaming GPU verifier (one process per mode), V1 with the GPU verifier, the integer forward pass
   with int8 GEMMs (`torch._int_mm`), fp16 and bf16 re-execution of five models at 64, 512 and 2,048 tokens on the
   same GPU (full models measured), and `h2d_bandwidth.py`; peak memory in every line.
7. **quality** shrinks to Llama-2-13B (optional; runs only with `PVI_LLAMA_LICENCE=1`, the runner's own licence).
   The `attack` tier is removed (G3.5-L ran on a Xeon node).
8. **One policy variable.** Every mode-C cell (cpuv, v1, serve, bench, gpuv, gen, batch, real, setup, smoke) takes
   `PVI_C_POLICY` (default `auto`). PS2's decision of Wed 21 Oct is one `export`. H3 reads it as `--policy`.
9. **The audit of `strong_gpu.sh must`.** The S&P tables read the optimised protocol of
   `paper_assets.OPTIMISED` (the paper's "base" columns are the course report's l40s_improved* runs), not the
   basic protocol the old must tier ran. The new `must` is exactly the cells tab:eval-llm (C prove, CPU and GPU
   verify, proof; Kpre CPU verify, proof, at 64 and 2,048 tokens), tab:eval-compare (VGG-16, GPT-2 at 64 and 512,
   Llama-2-7B at 1 token, Qwen3-4B in Maverick's setting with and without the encoding, zkLLM's rows) and the
   outcome counts (the six image models with their attacks, the 30-attack sets on the 64-token full builds) need,
   full models only (no 1/2-block extrapolation builds). Removed from the package: every `_nofix`, `_nolean`,
   `_thr1` GPT-2, `_thr12`, `_tf32` control, the extrapolation-shape cells, OPT-1.3B/6.7B at 64 tokens and the rest
   of zkLLM's model list; they stay reproducible as `report-must`, `should` and `nice` (course report). 43 jobs,
   8-12 h (target <= 30 h).
10. **Bytes come from the 2080 Ti.** Nothing to change in the package: bytes do not depend on the hardware, so the
    run's byte numbers are re-measured as a by-product, and the 2080 Ti byte cells measured before the freeze stay
    the source where the run has no cell.
11. **H3**: `sp2027_numbers.py` (Sec. 5).
12. **`docs/L40S_RUN_SP2027.md` v2**: run order and hours, `HF_HOME` on node-local disk (40 GB), exclusive use and
    24/7, the optional vLLM venv, the smoke procedure (smoke + `PVI_REHEARSAL=1` per tier, at most 2 h), the hotfix
    rule, what to send back, and the environment of the TAU nodes (CXX, `PVI_PY_INCLUDE`, `TRITON_CACHE_DIR`)
    generalised, with a no-SLURM mode (`PVI_LOCAL=1`).
13. **Built incrementally.** This commit completes the package against the items merged by 11 Oct (FS1, RX,
    V-LONG-a, A7, F3a part 1, V-V1N parts 1-4, PS2's tables). Items landing later add or switch cells in place: F3b
    (real Qwen3-4B) needs `--real` for Llama/Qwen checkpoints in perf.py/v1_bytes.py; A7-vLLM needs its harness.
14. **The friend's smoke and the hotfix rule**: `sp2027.sh smoke`, then `PVI_REHEARSAL=1 PVI_OPTIONAL=1 sp2027.sh
    all` (22 jobs) under another platform name; in `all` nothing after smoke starts unless it succeeded
    (`--dependency=afterok`; `PVI_LOCAL=1` stops). The run document states the hotfix rule.

## 2. The tiers and their budgets

`all` submits the must tiers in this order, each job after the previous one (`--dependency=afterany:<previous>`;
after smoke `afterok`), every job `--exclusive`. Job limits are the sum of the jobs' `--time`, a cap; the estimate
is the budget.

| # | Tier | Jobs | Estimate (h) | Job limits (h) | Cells |
|---|---|---:|---:|---:|---|
| 1 | smoke | 1 | 1 | 2 | 16 test files (GPU, native, FS1, serve, F6, generation, setup proof), int8-GEMM and native self-checks, one V1 query, the copy bandwidth |
| 2 | cpuv | 13 | 6-9 | 22 | perf.py Kpre+C x {Llama-2-7B, Qwen3-4B, OPT-1.3B} x {64, 2,048}; RX x 3 models + allocator variant; V1 paths x 3 |
| 3 | v1 | 13 | 4-8 | 26 | GPT-2 64/512, OPT-125M/1.3B/6.7B and Llama-2-7B at 64/2,048, Llama-2-13B at 64 (and 2 blocks at 2,048), Qwen3-4B at 64 |
| 4 | serve | 5 | 8-10 | 25 | OPT-1.3B, Qwen3-4B, Llama-2-7B, OPT-6.7B, Llama-2-13B; workloads (a) and (b) |
| 5 | bench | 43 | 8-12 | 27 | Sec. 1 part 9 |
| 6 | gpuv | 12 | 4-6 | 19 | streaming GPU verifier x 2 models x 2 modes, V1 GPU verifier x 2, re-execution x 5 models, bandwidth |
| 7 | gen | 2 | 2-3 | 6 | Llama-2-7B, Qwen3-4B |
| 8 | batch | 7 | 2-3 | 6 | C4a |
| 9 | real | 16 | 4-6 | 22 | F3a |
| 10 | setup | 2 | 1-2 | 4 | Llama-2-7B (a = 100), GPT-2 |
| | **must total** | **114** | **40-60** | **159** | at most about 113 h: met with margin |
| 11 | quality (optional) | 1 | 2-3 | 8 | Llama-2-13B |
| 12 | long (optional) | 12 | 4-7 | 28 | V-LONG-b |

## 3. Where the estimates come from

Estimates, not measurements; each from measured times of the same code on other hardware, without assuming the
L40S/EPYC is faster:

* **cpuv.** V-LONG-a's full Llama-2-7B Kpre verifier on 8 Xeon threads: 1.01/1.40 s at 64 tokens, 18.5/19.6 s at
  2,048 (interactive/FS; job 1007816); mode-C commitments under the auto plan on the L40S: 510-590 s for
  Llama-2-7B, 575 s for Qwen3-4B, 190 s for OPT-1.3B (`tables_l40s_improved2`, `commit_total`); 6 queries per mode:
  about 0.2-0.4 h per perf.py cell, 1.2 h for the six. RX on the Xeon, full Llama-2-7B at 2,048 tokens: int 179 s,
  int_fp32gemm 142 s, fp32 93 s, bf16 306 s, dynint8 56 s per repetition, 6 repetitions = 1.3 h; Qwen3-4B about
  two thirds of that, OPT-1.3B a fifth, the allocator variant (3 variants) 0.6 h: about 3.2 h. V1's paths: about 1 h.
* **v1.** Job 1007816 (Llama-2-7B, 2 of 32 blocks, 2,048 tokens, Kpre): V1 prover 2.66 s and verifier 2.74 s, v0
  0.31/0.79 s, scaled by 16 to the full model; plus the commitments above; about 3-4 h in all.
* **serve.** A7's write-up: 8-10 h on the L40S.
* **bench.** The optimised protocol's L40S per-query times and commitments (`tables_l40s_improved2/3`), e.g.
  Llama-2-13B at 2,048 tokens: prover 17.6 s, Kpre CPU verifier 84.5 s, commitment 1,000-1,300 s; summed over the
  43 jobs about 7 h, times 1.2-1.4 for Fiat-Shamir's 7 repetitions and columns.
* **gpuv.** Full-model integer builds (1-2 min per model, T and variant) dominate the re-execution jobs; the
  streaming verifier cells are 10-20 min each.
* **gen.** A commitment (above), 4 responses per (G, rule) generated without a KV cache (G forward passes each), and
  2 modes x 4 proved queries of 575-767 positions: 0.5-1 h per model.
* **batch, real, setup, quality, long.** C4a's joint claims at B = 64 are those of 2,048 tokens; F3a's OPT-6.7B
  download (13.4 GB) and fp32 build; the setup proof of 2.8 GB after one commitment; Llama-2-13B's 26 GB download
  and 140 windows twice; the verifier's attention grows as T^2, 64x from 2k to 16k tokens per block.

The plan's 78-113 h assumed the old must tier (20-30 h, basic protocol with its controls and extrapolation builds)
and the v1 tier at 22-32 h; the audited bench tier and the per-query times above give 40-60 h.

## 4. The dry run and the rehearsal

### 4.1 Dry run

`PVI_DRYRUN=1 PVI_PLATFORM=l40s_sp2027 [PVI_OPTIONAL=1] bash code/experiments/4_defence_benchmark/slurm/sp2027.sh
all` prints every `sbatch` line (strong_gpu.sh's bench jobs included, in the same chain, with placeholder job ids
`DRY<n>`), each tier's job count, job-limit sum and estimate, and the totals; it submits nothing and creates no
folder. The run's listing is `docs/improvements/F6_dryrun.txt` (127 `sbatch` lines with the optional tiers; 143
lines with the comments), the rehearsal's `F6_dryrun_rehearsal.txt` (22 `sbatch` lines). H3 uses a listing as the
expected jobs (`--expect`).

### 4.2 Rehearsal (part 13; Fri 23 Oct partial, Wed-Thu 28-29 Oct full)

`PVI_REHEARSAL=1` submits each tier's smallest cells: GPT-2 with 2 blocks (cpuv, v1, bench, gpuv, gen, batch,
setup), OPT-125M with 2 blocks (serve), real OPT-125M (real), LeNet-5 (bench), ppl_llama.py's tiny random model
with Qwen3's vocabulary (quality), one Llama-2-7B block at 4,608 tokens with `--max-pos 8192` (long), and the smoke
with its quick test files. With 2 timed queries each, the 22 jobs take about an hour on an RTX 2080 Ti node. On the
TAU cluster, from the server clone of the freeze candidate (`V` = the sp2027 server folder of
`I_server_setup.md`):

```bash
export PVI_PLATFORM=rehearsal_2080ti SBATCH_PARTITION=studentbatch SBATCH_GRES=gpu:1
export PVI_EXCLUSIVE=0 HF_HOME=/tmp/$USER/hf
export CXX=$V/tools/cxx/bin/x86_64-conda-linux-gnu-g++ PATH=$V/tools/cxx/bin:$PATH PVI_PY_INCLUDE=$V/pyinclude/python3.12
PVI_REHEARSAL=1 PVI_OPTIONAL=1 bash code/experiments/4_defence_benchmark/slurm/sp2027.sh all
# when the chain has finished:
cd code && python experiments/5_comparison/aggregate.py --platform rehearsal_2080ti
python experiments/5_comparison/sp2027_numbers.py --logs ../logs/rehearsal_2080ti --platform rehearsal_2080ti \
    --base '' --rehearsal --expect ../docs/improvements/F6_dryrun_rehearsal.txt --check
```

It passes when every job of the listing has results, every tier has records of its kinds, and there is no `PARSE`
or `CHECK FAILED` line (exit status 0).

### 4.3 No SLURM

`PVI_LOCAL=1` runs every job here in the foreground, one after another, each within its time limit and logged under
the name SLURM would give it (strong_gpu.sh translates `--export=ALL,...` and `-c` for bench.sbatch); `all` stops
if the smoke's log does not end in `SMOKE OK`. Exercised on the laptop with a stub interpreter (Sec. 6).

## 5. H3: what each tier fills

`cd code && python experiments/5_comparison/aggregate.py --platform l40s_sp2027 && python
experiments/5_comparison/sp2027_numbers.py` writes `paper_sp2027/generated/` (one `.tex` body per table,
`numbers.tex`, `cells.csv`, `coverage.txt`). Nothing is pasted by hand; the writers `\input` the bodies or copy them.

| Tier | Generated file / macros | Fills (the `\todo`s it answers) |
|---|---|---|
| bench | `tab_eval_llm.tex` (the paper's layout), `tab_eval_llm_full.tex` (with the new CPU verifiers) | tab:eval-llm's new columns (evaluation.tex 74, 87-96; ext_discussion 41) |
| bench | `tab_eval_compare.tex` | tab:eval-compare's ours (evaluation.tex 260-261, 285); zkLLM on our L40S stays F5 |
| bench | `tab_counts.tex`, `\FinalNHonest`, `\FinalNAttacks`, `\FinalNFreivalds`, ... | evaluation.tex 64, ext_evaluation 127 (outcome counts with the final code) |
| bench, batch | `tab_memory_bench.tex`, `tab_batch.tex` | F4 (evaluation.tex 107, ext_evaluation 113, ext_discussion 43); C4a's batch table |
| cpuv | `tab_options.tex`, `\CpuRatioN`, `\CpuRatio<model><T>`, `\CpuRatioMin<model>` | tab:options (evaluation.tex 183, 208), RX/V-LONG-a (evaluation.tex 223, intro 117), the abstract's "at least N times" |
| cpuv, gpuv | `tab_eval_reexec.tex` | tab:eval-reexec, "L40S: both on one GPU" (evaluation.tex 228), ext_evaluation 107 |
| v1 (and cpuv's V1 cells) | `tab_eval_v1.tex`, `\Crossover<model><L><T><mode>` | tab:eval-v1 (evaluation.tex 143-145, 171), ext_evaluation 146, 209-210 (Llama-2-7B; Llama-2-13B only on 2 blocks at 2,048) |
| serve | `tab_serve.tex`, `\Serve<model><T><mode>OverUOne/OverUTwo` | A7 (evaluation.tex 108, discussion 33, ext_discussion 115) |
| gen | `tab_gen.tex`, `\Gen<model><mode><rule>G<G>MsPerToken` | generation.tex 54, generation_quality 114, ext_generation 115 |
| real (and F3a's 2080 Ti/Xeon `f3a-*` records) | `tab_real.tex` | F3a (evaluation.tex 48, 159; quality.tex 72; ext_v1 171, 173; ext_discussion 54) |
| setup | `tab_setup_time.tex`, `\Setup<model>A<a>Prover/Verifier` | defence.tex 219, ext_defence 120, ext_discussion 144 (tab:setup's time) |
| gpuv | `tab_h2d.tex`, `\HtoDPinnedGBs` | EV's PCIe floor, ext_discussion 111 (transfer times) |
| quality | `tab_quality.tex` | Llama-2-13B's row (F2b-L); also parses F2a's/F2b's 2080 Ti records |
| long | `tab_long.tex`, `\LongFitA`, `\LongFitB`, `\LongFitMaxResidual` | V-LONG-b (C-W7, C-Q7) |
| all timed tiers | `tab_spread.tex`, `tab_memory.tex` | median [min-max] of every headline cell (part 2), F4 |

`--stored` runs the same code on the repository's stored records: V-LONG-a's and the V1 re-time's
(`artifacts/results/vlonga_v1retime_w1c_1007816.jsonl`), F2a's quality records, A7's laptop serving records, the
laptop perf records, and the course report's optimised L40S tables read both as the base and as the "new" run.

## 6. Validation (laptop, 11 Oct)

* `bash -n` on `sp2027.sh`, `strong_gpu.sh`, `bench.sbatch`, `smoke.sbatch`, `train.sbatch`: no error.
* The dry run: 127 `sbatch` lines for the run with the optional tiers (114 must jobs), 22 for the rehearsal; every
  line chained, `--exclusive`, within the caps.
* `PVI_LOCAL=1` with a stub interpreter (`PVI_PYTHON=echo`): `all` stops after a smoke without `SMOKE OK`; the
  bench tier runs bench.sbatch with the job's arguments, and the Maverick job with `PVI_THREADS=1`.
* `sp2027_numbers.py --stored`: 320 records from 24 files parsed (no `PARSE` error); tab:eval-llm's base columns
  and proofs **63 cells identical** to evaluation.tex, tab:eval-compare's ours **30 cells identical**, and the new
  columns read from the same runs equal the base ones (0 differ): the parsers regenerate the existing tables
  unchanged. The outcome counts of the three optimised runs (every cell): 8,268 honest queries, 5,832 attacks.
* Tests (project venv, 2 threads, one process at a time): `tests/test_f6_package.py` 8 passed (gen_bench.py and
  setup_timing.py on tiny-llama in modes C and Kpre, greedy and sampled, both tampers rejected with the expected
  labels; the package invariants on the dry run); `tests/test_sp2027_numbers.py` 5 passed;
  `tests/test_comparison_tables.py` 17 passed; `tests/test_serve_ab.py` 17 passed, 1 skipped (CUDA);
  `tests/test_plans.py::test_perf_random_init_builds_every_cnn_it_accepts` passed.
* The course report's pipeline is unchanged: `paper_assets.py --platform l40s --optimised
  l40s_improved,l40s_improved2,l40s_improved3 --check` exit 0; `text_numbers.py ... --definition` exit 0 (without
  `--definition`, its 7 count checks fail as before: no file it reads changed).

## 7. What remains

* **Not validated here** (no GPU, no SLURM on the laptop): every CUDA path of the new scripts (gen_bench.py on a
  GPU, setup_timing.py's GPU combinations, h2d_bandwidth.py), the native kernels with `CXX`, the actual `sbatch`
  submission and the `afterok`/`afterany` + `singleton` dependency strings, the memory requests against real
  peaks, the downloads, and the job times. The 2080 Ti rehearsal (Sec. 4.2) and the friend's smoke cover these.
* **F3b** (real Qwen3-4B, real generation) and a real Llama-2-7B in mode C: perf.py and v1_bytes.py `--real` read
  OPT checkpoints only; the real tier prints a note and has no such cells until they do.
* **A7-vLLM**: no harness yet; the run document describes the optional venv.
* **PS2**: if the plan becomes cnn18c, `export PVI_C_POLICY=cnn18c` for the run and `--policy cnn18c` for H3.
  `count_outcomes.py --definition` (the course report's definition) would then not match the cells' `polcnn18c`
  tag; H3's counts do not use it.
* **The paper**: tab:eval-llm's new columns are under Fiat-Shamir while its caption says interactive, and its proof
  columns then change (H4/EV's decision); Llama-2-13B's V1 bytes at 2,048 tokens stay 2 blocks (the v1 tier's
  plan) and remain modelled for the full model.
* The smoke's test list includes `test_gpu_exactness.py` at the real models' sizes; its time on the L40S is not
  measured (the job's limit is 2 h).

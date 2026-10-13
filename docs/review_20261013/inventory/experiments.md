# Experiment scripts and result files: what the S&P 2027 paper and its run package use

All paths below are relative to the repository root `C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-amina-project/4aa2c028-956d-4ff2-9cca-56d76e547439/scratchpad/wt_sp2027` (branch `sp2027`, HEAD `08a7f1f`). This was a read-only pass. I read files, ran grep, did light AST parsing of argparse calls and read a few JSON headers. I ran no tests and built no models.

## 0. Main findings

1. **The paper still uses the course-paper experiments.** The attack section takes its numbers from `1_reproduction`, `2_attack` and `3_sampling_fixes`, and from the CNN path-test cells in `raw_l40s/` (`attack.tex:184-231`, `ext_attack.tex:117-153`). All of them stay.
2. **About 14 of the 31 scripts in `6_improvements` are no longer used.** They are one-off A/B harnesses, profilers or course-report tools. But `ab_verifier.py` cannot simply be deleted: `perf.py:161` loads it on every run, `plan_bytes.py:58` imports from it, and `tests/test_plans.py:1199` uses it. Its model builders have to move to a shared module first.
3. **The scripts import each other heavily.** `code/README.md:141-142` says the experiments never import each other except `perf.py`; that is no longer true. The actual links:
   - `perf.py` loads `ab_verifier.py`, `bench.py` and `v1_bytes.py` (`perf.py:161,207,237`).
   - `gen_bench.py` loads `serve_ab.py` (only for the `TINY` model shapes) and `v1_bytes.py` (`gen_bench.py:74,209`).
   - `setup_timing.py` loads `gen_bench.py` (`:62`).
   - `open_ab.py` loads `bench.py` (`:67`).
   - `v1_bytes.py` imports `9_quality/ppl_llama.py` and `9_quality/common.py` (`v1_bytes.py:69-70,95-102`).
   - `inrange_llm.py` imports `real_llm.py`; `adaptive_game_llm.py` imports `inrange_llm.py`, `real_llm.py` and `adaptive_game.py` (`:64-67,139`).
   - `wire.py` imports `plan_bytes.py`, which imports `ab_verifier.py`.
4. **Many options are environment variables, not CLI arguments.** The superseded "before" paths behind `tab:eval-improve` and V1's old verifier are reached only through `PVI_*` variables read inside `src`: `PVI_FUSED_ATTN`, `PVI_GPU_DECODE`, `PVI_NATIVE_{ATTN,CHEAP,DECODE,GEMM,AVX512}`, `PVI_CPU_ATTN`, `PVI_FOLD_FAST`, `PVI_FUSED_CODEC`, `PVI_V1_BUCKETS`, `PVI_V1_NATIVE`, `PVI_MALLOC_TUNE`, `PVI_INT32_CLAIMS`, `PVI_HASH_THREADS`, `PVI_MERKLE_PROCESSES`, `PVI_GKR_TRITON`, `PVI_LEGACY_WEIGHT_KEY`, `PVI_SETUP_CACHE`. The run package's own settings are also environment variables (`sp2027.sh:28-45`). Your rule says these must be arguments.
5. **Several defaults are the superseded variant, not the paper's setting.** They should become non-default options:
   - `bench.py --policy paper` (`bench.py:1015`), `perf.py --policy paper` and `--wire/--prune-last/--lookups off` (`perf.py:177-188`), `serve_ab.py --policy paper` (`serve_ab.py:867`). "paper" here means the course report's basic plan.
   - `--recipe legacy` in `perf.py:195` and `v1_bytes.py:324`.
   - `zeroshot.py --builder f2a` (`zeroshot.py:331`), although the paper's zero-shot numbers come from `--builder paper` (`docs/improvements/F2c_zero_shot.md:155-198`).
   - `ppl_2048.py`'s smoothed graph defaults to the α = 0.55 percentile recipe (`ppl_2048.py:50,227`).
   - `sampler_tv.py` hard-codes the α = 0.55 recipe (`sampler_tv.py:93`), so the TV distances in `generation.tex:93-98` were measured on an integer OPT-1.3B that is not the certified model of `tab:quality`.
6. **Many paper numbers have no stored record in this tree.** The biggest gaps:
   - The certified-recipe rows of `tab:quality` (`quality.tex:32,40-42`). The stored `9_quality/ppl2048_opt-*.jsonl` hold only round-1 cells at α = 0.55; the round-2 `ppl2048_search2_*.jsonl` (job 1011066, `F2a_quality_2048.md:189`) is absent.
   - The OPT-1.3B sampler game (`attack.tex:224-227`): `adaptive_game_opt-1.3b.json` exists on no branch.
   - The F2b and F2c results, F3a/F3b, V-LONG-b, the RX baselines, every 2080 Ti A/B row and E1. Full list in Section 7.
7. **Some quality numbers come from scripts that are not on `sp2027`.** `tab:quality-attr`, the per-row-scale figure and the cross-layer-equalisation figure (`ext_generation.tex:147-177,200,240-241`) come from `attribution.py`, `hybrid.py` and `fixes.py`. These and their records exist only on branch `f2-quality` (`git ls-tree f2-quality code/experiments/9_quality/`).
8. **Three S&P figures are drawn by the course-paper table generator.** `paper_sp2027/figures/{overview_attack,overview_defence,protocol}.pdf` are byte-identical (md5) to `report/figures/`. They are drawn by `paper_assets.py` (`fig_overview` and `fig_protocol`, `paper_assets.py:884-975`), which writes into `../report/` (`paper_assets.py:698`). `ev_breakeven.pdf` is byte-identical to `artifacts/results/ev_breakeven_draft.pdf`, which `ev_breakeven.py` builds from hand-typed inputs (`5_comparison/ev_inputs_draft.json`).
9. **Three folders have no README:** `7_reexec`, `8_v1`, `9_quality`. The READMEs of `4`, `5` and `6` describe the course report. For example, `6_improvements/README.md:9-13` says the paper's numbers do not come from that folder, but `perf.py`, `gen_bench.py`, `setup_timing.py` and `h2d_bandwidth.py` are now package scripts.
10. **The anonymous export leaves out the run package.** `scripts/export_anonymous.py:37-38` excludes `*.sbatch` and all of `4_defence_benchmark/slurm/`, so `sp2027.sh` and `strong_gpu.sh` would not reach the artifact.

## 1. How I decided "used"

- **Paper:** `main.tex:76-106` inputs `intro`, `attack`, `defence`, `v1`, `generation`, `evaluation` (which inputs `quality`, `evaluation.tex:316`), `discussion`, the `appendix_*` files, `app_params`, `ext_proofs` and every `ext_*.tex`. `generation_quality.tex` is input nowhere; it is an orphan. I traced each numeric claim to a `docs/improvements/*.md` entry, or to a stored file, and from there to the script that produced it.
- **Package:** the commands in `code/experiments/4_defence_benchmark/slurm/sp2027.sh` and `strong_gpu.sh`.
- **Number pipeline:** `5_comparison/sp2027_numbers.py`, which imports `paper_assets` and `count_outcomes` (`:861,984`). Its stored-record self-check reads `STORED_LOGS` (`:85-87`) and is exercised by `tests/test_sp2027_numbers.py:182`.

## 2. Per-script inventory

Column key: **CLI** lists the argparse options (defaults only where they matter). **Hard-coded** lists values that should become CLI options. **Used by** cites the paper, package or pipeline line that depends on the script. "(t)" marks a test that loads the script.

### 0_train_models, 1_reproduction

| Script | Purpose | CLI | Hard-coded | Used by | Verdict |
|---|---|---|---|---|---|
| `0_train_models/train.py` | train the 7 MNIST models | `--force`, `--only` (`:27-28`) | hyperparameters live in `pvi.zoo` | models in `artifacts/models/` (11 MB, all loaded by 1–3) | KEEP |
| `1_reproduction/run.py` | path protocol on its own threat model | `--queries 150 --challenges 20 --setups` (`:100-102`) | output path `reproduction.json` (`:122`) | `ext_attack.tex:117-124`, `attack.tex:187` | KEEP; add `--out` |

### 2_attack

| Script | Purpose | CLI | Hard-coded | Used by | Verdict |
|---|---|---|---|---|---|
| `run.py` | single-neuron attack and backdoor on MNIST | `--queries 200 --challenges 25` (`:267-268`) | `TARGET_CLASS=0` (`:55`), `OPENING_BUDGET_PATHS=250` (`:56`), forge steps 1500/3000 (`:76-77`), stealth percentiles (`:230`), output (`:296`) | `attack.tex:189-193`, `ext_attack.tex:127-133` | KEEP |
| `real_llm.py` + `real_llm.sbatch` | attack and backdoor on OPT-6.7B | `--model --config --layers --prompts --topk --target --out --device` (`:257-265`) | `PROMPTS` (`:50`), `TRIGGER` (`:66`); the sbatch fixes layers and topk | `attack.tex:213-217`, `ext_attack.tex:148-153` | KEEP |
| `inrange_llm.py` | in-range edits against range checks | `--model --layers --prompts --calib-windows --calib-seq --cap --threads --device --moves` (`:157-165`) | no `--out`: path built from model and moves (`:217`); seed 0 (`:169`); range kinds max and p999 (`:103,187`) | `tab:inrange` `ext_attack.tex:279-310`, `attack.tex:217-231`, `ext_proofs.tex:1006` | KEEP; add `--out`, `--ranges`, `--seed` |
| `real_weights_ppl.py` + `.sbatch` | integer OPT perplexity, 16 × 128 windows (course report Sec. 4.1) | `--model --config --parquet --seq --windows --pct --norm-gain --device --out` | — | only a tex comment (`quality.tex:4`); superseded by `ppl_2048.py` | DROP |
| `REAL_LLM.md`, `README.md` | documentation | | | course-report section numbers | rewrite |

### 3_sampling_fixes

| Script | Purpose | CLI | Hard-coded | Used by | Verdict |
|---|---|---|---|---|---|
| `run.py` | uniform, static, contribution and saliency samplers | `--queries 40` (`:175`) | `LAYER=1` (`:45`), `PERCENTILE=99` (`:46`), `candidate_neurons=60` (`:85`), output (`:204`) | `attack.tex:195-198` (9.2% = `defence.json` `local-contribution.detection_evasive`) | KEEP |
| `floor_sampler.py` | contribution weighting with a floor δ | `--queries 20` (`:73`) | `EPSILONS` (`:47`), `LAYER` (`:48`), 3,000 calibration images (`:83`), output (`:151`) | `attack.tex:199-201`, `ext_attack.tex:138-141` | KEEP |
| `adaptive_game.py` | MNIST adaptive game, LP mixture, double oracle | `--model --layers --queries --calib --calib-size --caps --pool --double-oracle --threads --tail-percentile --out` (`:947-958`) | `CAPS`, `FLOOR_EPSILONS`, `KMAX`, `K_LIMITS`, `PATH_COUNTS` (`:75-83`) | `attack.tex:132-135,201-205`; test (t) `test_adaptive_game.py:44` | KEEP |
| `adaptive_game_llm.py` + `.sbatch` | the same game on OPT-1.3B | `--model --layers --prompts --calib-windows --calib-seq --caps --tail-percentile --threads --device --out` (`:127-136`) | `EDIT_CAP=60` (`:72`) | `attack.tex:224-227`, `ext_attack.tex:326-345`; its output file is not stored | KEEP |
| `adaptive_game_plots.py` | figure_g3a/b and table_g3 | `--mnist --llm --out-dir --cap-key` | — | not in the paper (it includes only 4 figures) | DROP |

### 4_defence_benchmark

| Script | Purpose | CLI | Hard-coded | Used by | Verdict |
|---|---|---|---|---|---|
| `bench.py` | v0 benchmark cells (CNN and LLM) into `raw_<platform>/` | suite, `--model --queries --tampers --seq --lams --modes --threads --policy(paper) --prune-last --gen --lookups --tag --builds --llm-tampers --lean --cnn-batches --batches --batch-trials --verifier-device --verifier-impl --wire --tf32 --platform` (`:1004-1056`) | `RATE=4` (`:86`), `LAMBDAS` (`:85`), tamper cells at λ = 40 (`:632,930`), seeds 0/1/123 (`:449,832,638,939`), 64/512 calibration images (`:450`), `calib_tokens=min(seq,32)` (`:826`), skip above 0.85 of GPU memory (`:817`), `PVI_LEGACY_WEIGHT_KEY` (`:1076`) | `strong_gpu.sh` must and batch; `--gen` feeds `tab:gen` and `tab:gen-full` (E1); tests (t) `test_plans.py:1141`, `test_wire.py:617`, `test_lookups.py:634`, `test_contention.py:196` | KEEP. Default `--policy` should be the paper's plan; keep `paper` (rename `basic`) and `--gen` as options |
| `train.py` (+ `slurm/train.sbatch`) | train the CNNs | `--model --device` (`:67-68`) | recipes are a table in the file | weights for the image cells (`L40S_RUN_SP2027.md:27`) | KEEP |
| `slurm/bench.sbatch` | job script for `bench.py` | arguments passed through | `PVI_THREADS`, `PVI_TF32` (`:23,25`) | `strong_gpu.sh:47` | KEEP |
| `slurm/smoke.sbatch` | course-report smoke: every test plus a fingerprint check against `raw/` | — | — | `strong_gpu.sh:113-115` only; the friend's smoke is `sp2027.sh smoke` (`sp2027.sh:138-155`, `F6_final_run_package.md` item 14) | DROP (with `raw/`) |
| `slurm/sp2027.sh` | the S&P final-run package | tier positional; everything else is environment (`:6-45`) | model lists, sequence lengths, queries, memory and time limits per tier; `--lmax 27` and `--threads 8` (`:132-136`) | the package | KEEP; turn the environment switches into flags |
| `slurm/strong_gpu.sh` | bench tiers | tier positional | `POL` defaults to `auto` (`:105`), but `sp2027.sh` defaults to `cnn18c` (`:66`) | must (`:116-176`) and batch (`:177-205`) via `sp2027.sh:245` | KEEP must and batch; DROP report-must, should, nice (`:206-296`) and smoke |

### 5_comparison

| Script | Purpose | CLI | Hard-coded | Used by | Verdict |
|---|---|---|---|---|---|
| `aggregate.py` | `raw_<p>/` → `tables_<p>/` | `--platform` (`:220`) | `RAW` default `raw_l40s` (`:38`) | `sp2027_numbers.py:3`; tests (t) | KEEP |
| `analytic.py` | path test needed for 2^-40 on Llama-2-7B | none | `llama2-7b`, 64 tokens, λ = 40 (`:23`) | `attack.tex:210-211`, `ext_evaluation.tex:200-202` | KEEP; add `--model --seq --lam` |
| `count_outcomes.py` | completeness and soundness counts | `--platform --policy --prefix --definition --require-tag --tex` (`:222-234`) | `--tex` targets `report/` | `evaluation.tex:64-67`; imported at `sp2027_numbers.py:861` | KEEP |
| `ev_breakeven.py` + `ev_inputs_draft.json` | the break-even figure | `--inputs --out --no-figure --figure` (`:59-62`) | λ = 128 and pruning (`:50-52`), sequence lengths 64/2048 (`:69`); inputs typed by hand | `fig:breakeven` (`evaluation.tex:224-233`, byte-identical), `ext_discussion.tex:81-82` | KEEP; take inputs from `sp2027_numbers` `cells.csv` |
| `fingerprint_check.py` | hardware-independent values of two roots agree | reads `sys.argv` only | — | backs "Proof sizes do not depend on the hardware" (`ext_evaluation.tex:45`); `smoke.sbatch:77`; test (t) `test_contention.py:252` | KEEP-AS-OPTION; convert to argparse |
| `literature.py` | builds `reported_curated.csv` | none | `CURATED` list (`:29`) | the "theirs" columns of `tab:eval-compare` | KEEP |
| `paper_assets.py` | course-report figures and tables, plus the cell-selection library | `--platform --optimised --check --pending-csv --strict --draft` (`:1499-1511`) | writes to `../report/` (`:698`) | `sp2027_numbers` uses `BASE, TABLES, OPT_TABLES, OPTIMISED, Measured, llm_cost, llm_spec, opt_llm, params, PROVE, VERIFY, LLM_METRIC, COLUMN_CELLS`; `fig_overview` and `fig_protocol` draw 3 S&P figures | MERGE: keep the selection library and the two figure functions (retarget output to `paper_sp2027/figures`); drop the `report/` writers |
| `sp2027_numbers.py` | records → paper tables and macros | `--logs --platform --base --policy --challenges --out --paper --expect --check --strict --rehearsal --stored --compare-paper --verbose` (`:1080-1093`) | `REPORT_RUNS` (`:89`), `STORED_LOGS` (`:85`) | the number pipeline | KEEP |
| `text_numbers.py` | checks `report/main.tex` | `--platform --optimised --tex --list --require-tag --definition` | — | course paper only; test (t) `test_comparison_tables.py:375` | DROP |
| `validate_extrapolation.py` | course-report extrapolation fit | `--platform` | — | not cited; test (t) `test_comparison_tables.py:30,63` | DROP |

### 6_improvements

| Script | Purpose | CLI | Used by | Verdict |
|---|---|---|---|---|
| `ab_verifier.py` | verifier A/B against an older checkout (`--base`) | cases, `--base --out --reps --threads --cache` | its builders `CNNS/Case/build/random_queries` (`:41,60,92,119`) are needed by `perf.py:161,217-224`, `plan_bytes.py:58`, test (t) `test_plans.py:1199` | DROP the A/B; MERGE the builders into a shared module |
| `attention_ab.py`, `gpu_ab.py`, `codec_ab.py`, `codec_stages.py`, `wire_hashes.py`, `claims_drop.py` | course-era A/B harnesses (need `--base`) and a diagnostic | various | `6_improvements/README.md:16-20` only | DROP |
| `cpu_attn_bench.py`, `cpu_profile.py`, `derive_ab.py`, `setup_cache_check.py` | B5 and PS2c diagnostics, a torch.compile A/B, a cache check | various | diary only (`B5_cpu_verifier.md:148-149`, `docs/improvements/README.md:38`, `PS2c_setup_cache.md`) | DROP |
| `plan_bytes.py` | byte model of the course-report plans | `--lam --policies --run --queries --wire --claims-only --prune-last --kpre --out` | `wire.py:42` (needs `query_claims`); test (t) `test_plans.py:1169` | DROP; MERGE `query_claims` into `wire.py` |
| `encode_prof.py` | A3: encoder profile, fused against torch | `--model --layers --seq --reps --rows --chunks --no-profile --compare-torch` | `tab:eval-improve` (`ext_evaluation.tex:69-71`) and the 1.74 vs 1.54 s result (`:94-95`) | KEEP |
| `fold_ab.py` | A6: Kpre precompute paths | `--model --layers --devices --reps --fs --threads` | 12.7 s (`ext_discussion.tex:137`) | KEEP |
| `forward_ab.py` | A2: float32 chunks against int8 GEMMs | `--model --layers --seq --queries` | `ext_evaluation.tex:67-68` | KEEP |
| `open_ab.py` | A1: float64 against int8 opening and fold | `--model --layers --seq --policy --lam --fiat-shamir --device --queries --threads` | `ext_evaluation.tex:62-66` | KEEP |
| `gpu_profile.py` | B1: CUDA time shares | `--model --layers --seq --mode --policy --top` | `ext_evaluation.tex:104-106` | KEEP |
| `gen_bench.py` | generated response as one prefill at scale | `--model --layers --prompt --gen --decode --modes --policy --lam --fs --queries --sampler-a --top-k --no-tampers --device --verifier-device --threads --label --real --temperature` (`:168-186`) | `sp2027.sh:136,292-295,351`; tests (t) | KEEP; hard-codes `RATE=4` and `CALIB_TOKENS=32` (`:57-58`) |
| `h2d_bandwidth.py` | PCIe copy bandwidth | `--sizes-mb --reps --label` | `sp2027.sh:154,265,280` | KEEP |
| `params_table.py` | every value of `tab:params` | `--model --policy --lam --amortise --lmax --v1-cells --tex` | `app_params.tex:57` | KEEP; the measured V1 cell counts are hard-coded defaults (`:90`) and should be a stored record |
| `perf.py` | the v0 timing harness | `--model --seq --layers --modes --lam --device --verifier-device --queries --threads --lean --fs --label --extra --policy(paper) --wire(off) --prune-last(off) --lookups(off) --real --smooth --calib-seq --recipe(legacy) --tampers --random-init` (`:162-200`) | `sp2027.sh:133` (cpuv, gpuv, real, long tiers); tests (t) | KEEP; flip the defaults; seeds and `calib_tokens` hard-coded (`:252-255`) |
| `ps2_tables.py` | fixed bytes of a C proof per plan (the PS2 decision) | `--models --policies --lam --rate-entries-per-s` | `PS2_lower_rate_plans.md`, `artifact_README.md:50`; the paper prints none of it | MERGE into `setup_bytes.py` (or KEEP-AS-OPTION) |
| `setup_bytes.py` | byte model of the setup proof | `--model --policy --lam --amortise --fiat-shamir` | `tab:setup` (`ext_defence.tex:127-148`) and text `:150-166` | KEEP |
| `setup_timing.py` | setup proof time | `--model --layers --policy --lam --amortise --fiat-shamir --reps --device --threads --label` | `sp2027.sh:360-366` | KEEP; `RATE=4` and `calib_tokens=32` hard-coded (`:37,76`) |
| `wire.py` | compact-encoding bytes per claim | specs, `--mnist --queries --reps --threads --out --summary` | "16–19 bits" (`ext_v1.tex:164`) from `results/wire_laptop.json` | KEEP (λ = 40 for Qwen3 is hard-coded), or compute the value from the final records |
| `README.md` | | | describes the course report | rewrite |

### 7_reexec (no README)

| Script | Purpose | CLI | Hard-coded | Used by | Verdict |
|---|---|---|---|---|---|
| `reexec.py` | re-execution baselines | `--model --seq --layers --variants --device --threads --reps --max-pos` (`:193-201`) | seeds (`:108,219`), last block always pruned (`:226`), fp16 on GPU and fp32 on CPU for "float" | `sp2027.sh:134`; `tab:options`, `tab:eval-reexec` | KEEP |
| `serve_ab.py` | prover overhead against unverified serving (A7) | `--model --layers --workload --cells --modes --batch-proof --policy(paper) --lookups --lam --reps --u2-dtype --device --verifier-device --threads --accept --label --table` (`:859-880`) | fs, wire, lean and pruning always on (`:898-899`); `RATE`, `CALIB_TOKENS`, `WORKLOADS`, `EGRESS_BYTES_S`, `RESIDUAL_MAX`, `BOOT`, `TINY` (`:108-130`) | `sp2027.sh:228`; `evaluation.tex:111-114`; `tab:options` note f | KEEP; move `TINY` into `pvi` |

### 8_v1 (no README)

| Script | Purpose | Used by | Verdict |
|---|---|---|---|
| `v1_bytes.py` | V1 against v0 bytes and times; real checkpoints | `sp2027.sh:132`; `tab:eval-v1`; tests (t) | KEEP. Its real-model loaders (`:56-121`) are library code used by `perf.py` and `gen_bench.py`; Llama calibration is hard-coded (4,000,000 characters, 4 × 2048 tokens, `:73-75`) |
| `v1_phases.py` | V1 verifier per phase, path 0 against path 1 | 45.4 s (`evaluation.tex:172-175`, `tab:options`; `V_V1N_fast_v1_verifier.md:377`) | KEEP (path 0 = superseded option) |
| `gkr_bench.py` | GKR microbenchmark | 15.6 ns per leaf (`ext_evaluation.tex:310-312`; `C2_v1_proof_size.md:203`) | KEEP |
| `gen_gkr_triton.py` | generates `src/pvi/fullcheck/gkr_triton.py` | a build tool, not an experiment | KEEP; move to `scripts/` |
| `profile_v1.py` | cProfile of one V1 query (threads hard-coded to 8) | nothing | DROP |

### 9_quality (no README)

| Script | Purpose | Used by | Verdict |
|---|---|---|---|
| `common.py` | data and model helpers | most quality scripts; `tests/test_real_weights.py:298` | KEEP (module) |
| `opt_builder.py` | a copy of `build_opt_from_hf` with extra options (`:1`) | `ppl_2048`, `ppl_llama`, `zeroshot`; tests `test_prob_max.py:79`, `test_recipe_equivalence.py:23` | MERGE into `pvi.fullcheck.real_weights` |
| `ppl_2048.py` | F2a quality | `tab:quality`, `quality.tex:47-56,65-67`; `--legacy` (`:238`) | KEEP; default to the certified recipe |
| `ppl_llama.py` | F2b quality | package quality tier (`sp2027.sh:383-396`); `quality.tex:84-88` | KEEP (`--glu-exps` stays a non-default option, per the int8 rule) |
| `ppl_smooth.py` | first smoothing study at 128 tokens | superseded recipe still described (`ext_generation.tex:191-196`) | KEEP-AS-OPTION, or MERGE into `ppl_2048 --legacy` |
| `sampler_tv.py` | E2b sampler TV distance | `generation.tex:93-98`; test (t) | KEEP; add `--recipe` (α = 0.55 hard-coded, `:93`) |
| `zeroshot.py` | F2c zero-shot accuracy | `quality.tex:68-72`, `tab:zeroshot`; test (t) | KEEP; default `--builder paper` |
| `digest_check.py` | D6a: reproducible build | `defence.tex:52`; `artifact_README.md:60-77` | KEEP |
| `llama_diag.py`, `llama_attrib.py` | F2b block-level and op-level diagnosis | Qwen3-4B second-activation claim (`quality.tex:87-88`, `ext_discussion.tex:72-74`; `F2b_Q_real_llama.md:1189-1219`); `test_llama_attrib.py` | KEEP (could merge the two) |

### scripts/

- `export_anonymous.py` (`--out --ref --pdf`): KEEP. It drops the run package (`:37-38`). Its `STATIC_DENY` list writes real names into the repository (`:53`), even though the docstring says no name is written into the repository.
- `artifact_README.md`: rewrite its results map (`:41-52`), which cites `ps2_tables` and `paper_assets` "for report".
- `scripts/__pycache__/`: delete.

## 3. Environment switches that should become arguments

- **On every timing harness** (`perf.py`, `v1_bytes.py`, `bench.py`, `gen_bench.py`, `serve_ab.py`), add for example `--switch NAME=VAL` or explicit flags covering the `PVI_*` variables listed in finding 4. They are read in `transformer.py`, `protocol.py`, `native_kernels.py`, `claimcodec.py`, `buckets.py`, `commitment.py`, `cut_protocol.py` and `setup_cache.py`. Each record should store the values used.
- **`sp2027.sh`**: `PVI_PLATFORM`, `PVI_C_POLICY`, `PVI_SETUP_CACHE`, `PVI_DRYRUN`, `PVI_REHEARSAL`, `PVI_OPTIONAL`, `PVI_CHAIN`, `PVI_EXCLUSIVE`, `PVI_MEM_MAX`, `PVI_QUERIES[_HEAD]`, `PVI_LOCAL`, `PVI_LLAMA_LICENCE`, `PVI_BIGMEM`, `PVI_Q7_ARGS` and `PVI_Q13_ARGS` (`sp2027.sh:6-45,394-396`) should become flags.
- **`bench.sbatch`**: `PVI_THREADS` and `PVI_TF32`.

## 4. Result files

### `code/artifacts/results/` (7.0 MB)

**Used by the paper or the pipeline (sizes in bytes):**
- `attack.json` 1,703; `defence.json` 899; `floor_sampler.json` 703; `reproduction.json` 2,228.
- `adaptive_game_mlp_mnist.json` 136,306.
- `inrange_llm_opt-1.3b_any.json` 155,669 and `_down.json` 340,841; `inrange_llm_opt-6.7b_any.json` 201,273 and `_down.json` 486,329.
- `real_llm_attack.json` 22,350 and `real_llm_attack_auto.json` 34,915 (`5_comparison/README.md:110`).
- `sampler_tv_opt-1.3b.json` 3,949,129.
- `vlonga_v1retime_w1c_1007816.jsonl` 6,679 (v0 full Llama-2-7B, 18.52 / 19.64 s; V1 at 2 and 3 blocks).
- `ev_breakeven_draft.{json,pdf,tex}` 4,050 / 15,993 / 480.
- `9_quality/ppl2048_opt-{125m,1.3b,6.7b}.jsonl` 105,239 / 109,902 / 50,418: the scalar-gain and 8-bit rows only.
- `9_quality/ppl2048_search_round1.jsonl` 144,935: the smooth-max sweep at α 0.3–0.5.
- `9_quality/llama_diag_1009531.txt` 11,857: the TinyLlama and Qwen3-0.6B diagnosis.

**Not used by the paper:**
- `inrange_llm_opt-125m*.json`: 3 files, 17,689.
- `real_weights_ppl_opt-*.json`: 5 files, 13,099.
- `sampler_tv_opt-125m.json` 275,618; `sampler_tv_selfcheck.json` 3,839.
- `figure_g3a.pdf` 34,062; `figure_g3b.pdf` 24,470; `table_g3.tex` 735.
- `zkllm_l40s/`: 96 files, 715,066. The README says it is not used (`code/README.md:37`), but a TODO at `evaluation.tex:311-312` cites its 844 s. Keep it as an option.

### `code/artifacts/comparison/` (289 MB)

**Used:**
- `raw_l40s_improved/` 47,458,387 (391 files), `raw_l40s_improved2/` 63,495,950 (640), `raw_l40s_improved3/` 1,757,252 (45), and their tables (1,608,718 / 2,737,080 / 180,741). These are the base columns of `tab:eval-llm` and `tab:eval-compare`, and the counts in `evaluation.tex:64-67`.
- `raw_l40s/` 76,540,858 (1,269 files) and `tables_l40s/` 3,884,851. These give the attack's path-test and CNN numbers, and `sp2027_numbers.py` sets `pa.TABLES` to `tables_l40s`.
- `excluded_cells.csv` 3,548; `literature/` 1,390,746; `tables/reported_curated.csv` 9,608; `tables/analytic.csv` 126; `README.md` 14,477 (needs a rewrite).

**Optional:** `raw_rtx2080ti-v2/` 50,837,641 and `tables_rtx2080ti-v2/` 2,388,707. They exist only for the hardware-independence check.

**Not used:**
- `raw/` 35,551,086 (the reference only for `smoke.sbatch`).
- `raw_l40s_improved2_thr1/` 4,629,578 and `tables_l40s_improved2_thr1/` 182,943.
- `tables/measured_summary.csv` 1,718,953 and `tables/llm_full_model.csv` 188,178 (derived from `raw/`).

### Result folders inside the experiment folders

- `6_improvements/results/` (1,141,870): `perf_*.jsonl` (16 files, 205,383) are inputs to the `--stored` test; `wire_laptop.json` (34,562) is the source of "16–19 bits". The other 901,925 B (`ab_*`, `attention_*`, `claims_*`, `codec_*`, `col_*`, `combined_*`, `gpu_ab_*`, `lookup_*`, `plan_bytes*`) are not used.
- `7_reexec/results/serve_ab_laptop_*.jsonl` (190,426) feed only the `--stored` test.

## 5. Paper numbers with no record or script on `sp2027`

- Certified rows of `tab:quality`.
- The OPT-1.3B sampler game.
- F2b ladder: OpenLLaMA ×1.073, TinyLlama ×1.12, Qwen3 ×1.40 / ×5.3.
- F2c zero-shot results.
- `tab:quality-attr`, per-row scales, cross-layer equalisation (scripts on `f2-quality` only).
- F3a/F3b (`tab:eval-real`, `ext_discussion.tex:58-66`).
- V-LONG-b (`tab:eval-long`).
- RX (93.2 s and 142 s), A1–A4, B2/B3, B5, A6, V-V1N, GKR, B1.
- E1 (`tab:gen`, `tab:gen-full`), C4a (`ext_discussion.tex:142-145`), the A7 2080 Ti pilot (`logs/a7_pilot`, `A7_serving_overhead.md:414-435`).

These presumably sit in server logs, which I did not check.

## 6. Paper-level points that touch these scripts

- **Mixed commitment plans.** `tab:eval-llm` says "Proof bytes do not change" (`evaluation.tex:76-77`), but the package times mode C under `cnn18c`, while the base columns are `auto`. Under a different plan the C proof's fixed part changes. Separately, `tab:setup` already says "the plan of the final runs" (`ext_defence.tex:129-130`). The tables now mix plans.
- **Real-weight recipe.** The real tier uses `--recipe certified` (`sp2027.sh:311-312`), while `tab:eval-real` describes the legacy recipe (`ext_evaluation.tex:124-127`). The text will need updating after the run.
- **Sampler TV model.** As in finding 5, E2b was measured on a non-certified model.
- **Orphan section.** `generation_quality.tex` is input nowhere.
- **Protocol figure numbering.** `protocol.pdf` was drawn "numbered as in Section 3.5" of the report (`paper_assets.py:912`). I did not check whether its labels match the S&P (S1)–(S3).

## 7. Proposed organisation of what stays

| Section | Contents |
|---|---|
| attack | `train_mnist`; `reproduction`; `attack_mnist` (`2_attack/run.py`); `real_llm` (+ sbatch); `inrange_llm`; `samplers` (`3_sampling_fixes/run.py`); `floor_sampler`; `adaptive_game`; `adaptive_game_llm` (+ sbatch); `analytic` |
| costs (v0) | `bench.py` (+ `bench.sbatch`, `train.py`, `train.sbatch`); `perf.py`; `params_table.py`; `setup_bytes.py` (+ `ps2_tables` merged); `setup_timing.py`; `wire.py`; improvement A/Bs: `open_ab`, `forward_ab`, `encode_prof`, `fold_ab`, `gpu_profile` |
| V1 | `v1_bytes.py`; `v1_phases.py`; `gkr_bench.py` |
| generation | `gen_bench.py`; `sampler_tv.py` |
| re-execution | `reexec.py`; `h2d_bandwidth.py` |
| quality | `common`; `real_weights` (with `opt_builder` merged); `ppl_2048`; `ppl_llama`; `ppl_smooth` (option); `zeroshot`; `digest_check`; `llama_diag`/`llama_attrib`; plus `attribution`/`hybrid`/`fixes` from `f2-quality` if `tab:quality-attr` stays |
| deployment | `serve_ab.py`; `ev_breakeven.py` |
| comparison | `literature.py`; `aggregate.py`; `count_outcomes.py`; `paper_assets` (library and the 3 figures); `fingerprint_check` (option) |
| package | `sp2027.sh`; `strong_gpu.sh` (must and batch only); `docs/L40S_RUN_SP2027.md` |
| paper numbers | `sp2027_numbers.py` |
| tools (`scripts/`) | `export_anonymous.py`; `artifact_README.md`; `gen_gkr_triton.py` |

Shared builders should move into a library module (for example `pvi/experiments/builders.py`) before any script is dropped: `bench._cnn_model`, `bench._mlp_model`, `bench._test_batches`, `bench._reset_peaks`, `bench.RATE`; `ab_verifier`'s builders; `v1_bytes`'s loaders and peak-memory helpers; `serve_ab.TINY`.

## 8. What I could not determine

- Which script measured the B2/B3 rows (`neither / fused / decode / both`) and the A4 prover-attention rows. The diary names only job ids; it was probably `perf.py` with the environment toggles.
- The B5 "before" (3.37 s) comes from a series of commits (`B5_cpu_verifier.md:170-180`). It is not clear that the toggles alone reproduce it.
- Whether the missing records exist on the server.
- Whether `protocol.pdf`'s labels match the S&P text.
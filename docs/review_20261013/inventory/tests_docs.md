# Tests and documentation review (scope A + B)

Repository root: `C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-amina-project/4aa2c028-956d-4ff2-9cca-56d76e547439/scratchpad/wt_sp2027` (branch `sp2027`). All paths below are relative to it.

I only read files, ran grep and did light AST parsing. I ran no tests and no model builds. Runtime classes are estimates, based on test size, the model shapes used, and timings recorded in `docs/improvements/*.md` (cited where they exist).

---

## A. `code/tests/`

### A.1 Summary

- **Size.** 52 `test_*.py` files plus 3 support files (`conftest.py`, `claimcodec_reference.py`, `gkr_triton_mock.py`), 15,102 lines in total. The AST count is 592 test functions, 199 of them parametrized.
- **Test counts in the docs disagree with each other:**
  - the full local suite once gave "1,492 passed" (`docs/improvements/C2_v1_proof_size.md:107`);
  - the HotCRP text says "about 1,100 automated tests" (`docs/submission/HOTCRP_FIELDS.md:12`).
- **Suite runtime claims also disagree:**
  - "about a quarter of an hour on a 4-core laptop" (`code/README.md:109-110`);
  - "about 30-50 minutes on 8 cores" (`scripts/artifact_README.md:36-37`);
  - recorded: `test_plans.py` alone took 549 passed / 4 skipped in 718 s on the laptop (`docs/improvements/F2b_Q_real_llama.md:260`).
- **No markers are registered.** `addopts = "-q --strict-markers"` is set in `code/pyproject.toml`, but there is no `slow`, `gpu` or `native` marker. All gating is through `skipif` / `importorskip`.
  - The only opt-in switch for slow tests is `PVI_SLOW=1` (`tests/test_logup.py:185`).
  - `test_gpu_exactness.py` defaults to `PVI_TEST_DEVICE=cuda`, so on a CPU-only machine it is skipped entirely (`tests/test_gpu_exactness.py:29-30`).
- **Almost every test covers code the S&P paper needs.** The course-paper experiments are still in the S&P paper:
  - MNIST reproduction and attack: `paper_sp2027/sections/attack.tex:184-191`, `ext_attack.tex:117-137`;
  - other samplers and the adaptive game: `ext_attack.tex:135-137`, `ext_attack.tex:328-341`.

  So the path-protocol tests stay. The few tests that cover course-only code are listed in A.5.

Legend:
- **fast**: under about 30 s on a CPU.
- **med**: about 0.5 to 3 min.
- **slow**: more than 3 min on a CPU.
- **GPU**: needs CUDA, skipped otherwise.
- **native**: needs Linux and a C++ compiler.
- **HF**: needs `transformers`; uses tiny random models, no download.
- **data**: reads stored artifacts.

### A.2 Per-file table

| Test file (functions) | Covers | Runtime class, needs | Kept? |
|---|---|---|---|
| test_adaptive_game.py (2) | `fast_acceptance` of `experiments/3_sampling_fixes/adaptive_game.py` equals `pvi.experiments.analysis.acceptance_probability` (:1-3) | fast; data: `artifacts/models/mlp_mnist_full.npz` (skips if absent, :41, :51) | yes (adaptive game, ext_attack.tex:328-341) |
| test_analysis.py (9) | the acceptance dynamic program against Monte Carlo of `pvi.protocol` | fast/med, CPU | yes |
| test_architecture.py (18) | `pvi.nn`: vectorised forward pass against the per-neuron relation | fast | yes |
| test_attacks.py (12) | `pvi.attacks` tamper and backdoor | fast | yes |
| test_claimcodec.py (54) | `pvi.fullcheck.claimcodec` (PVC3) against `claimcodec_reference.py`; threaded and torch paths; malformed input | med; part GPU (:33) | yes |
| test_comparison_tables.py (17) | `5_comparison/` `aggregate`, `validate_extrapolation`, `paper_assets` (13 uses), `text_numbers`, `count_outcomes`, on the stored tables of `l40s_improved{,2,3}` (:18) | fast; matplotlib (:15); data: `artifacts/comparison/tables_*` | mixed (A.5) |
| test_contention.py (10) | `pvi.fullcheck.contention`; `bench.py` rows carry the field; `aggregate`, `fingerprint_check` and `count_outcomes` read records with and without it | med; Linux `/proc` (:116); torchvision (:303) | yes, except the `fingerprint_check` part (course-only claim) |
| test_cut.py (11) | V1 cut and requant ops (V1_SPEC T0, T9, P1-P5) | med | yes |
| test_cut_protocol.py (13) | V1 query end to end, modes C and Kpre, interactive and FS (T4, T5) | med/slow (V1 suites about 6 min each on a node, `V_V1N_fast_v1_verifier.md:262`); 1 GPU test (:237) | yes |
| test_defences.py (11) | `pvi.defences` samplers and the sampling theorems | fast | yes |
| test_ev_breakeven.py (3) | `5_comparison/ev_breakeven.py` crossovers | fast | yes (figure at evaluation.tex:226) |
| test_extfield.py (8) | F_{p^8} | fast | yes |
| test_f3b_real_llama.py (4) | Llama branch of `v1_bytes.real_graph`; `gen_bench --real` | med; HF (:15) | yes |
| test_f6_package.py (8) | `gen_bench`, `setup_timing`, `reexec`, `h2d_bandwidth`, `perf` on tiny decoders; `sp2027.sh` dry-run invariants through bash (:32, :113-120) | med; bash | yes (final-run package) |
| test_fast_verifier.py (29) | fast verifier routines against `pvi.fullcheck.reference` | med/slow (a combination including it: 291 s, `F3a_real_weight_runs.md:128-131`); uses process pools | yes |
| test_fold_fast.py (4) | Kpre precompute (`_fold_local`), native and int8 GEMM against float64 | fast; native (:41), GPU int8 (:50) | yes |
| test_fs_tree.py (4) | Fiat-Shamir tree absorption; Kpre | fast | yes |
| test_fullcheck.py (35) | whole-network defence on integer CNN graphs | slow (with test_pruning: 76 passed in 167 s, `F2b_Q_real_llama.md:263`); 2 GPU tests (:508, :523) | yes |
| test_generation.py (4) | `with_lm_positions`, `Verifier.check_tokens` | med | yes |
| test_git_sha.py (4) | `pvi.results.git_sha` | fast; git binary (:13) | yes |
| test_gkr_triton.py (2) | Triton GKR prover matches the eager transcript byte for byte | GPU + Triton (:17) | yes |
| test_gkr_triton_cpu.py (2) | GKR host logic with CPU mock kernels | med | yes |
| test_gpu_exactness.py (9) | GPU paths at real-model sizes | GPU (:29-30); 77 passed on the 2080 Ti (`B2_B3_gpu_verifier.md:7`) | yes |
| test_gpu_verifier.py (32) | int32 attention, exp lookup table, GPU verifier against the reference | mostly GPU (:41); 287 passed / 15 skipped on the 2080 Ti (`B2_B3:7`) | yes |
| test_hostmem.py (6) | pinned-memory limit | GPU (:20) | yes |
| test_layouts.py (12) | col layout and fusion of the `c` policies | med | yes (cnn18c is the package default, sp2027.sh:66) |
| test_lean_and_verifier_device.py (3) | `--lean`, `--verifier-device` | fast/med; adds a CUDA parameter if a GPU exists (:17) | yes |
| test_llama_attrib.py (10) | `9_quality/llama_attrib.py` attribution, plus `real_llama` `qk_balance` | fast (44 passed in 14.8 s, `F2b_Q:825`); HF (:44) | mixed: `qk_balance` is builder code; `llama_attrib.py` is a diagnostic harness |
| test_logup.py (6) | windowed logUp (T3, K1) | med; K1 only with `PVI_SLOW=1` (:185) | yes |
| test_logup_gkr.py (5) | fractional-sum GKR (T2) | med | yes |
| test_lookups.py (21) | lookup tables and pruning in C/K/Kpre; bench cell names; `plan_bytes` rows | med/slow | yes |
| test_merkle.py (16) | `pvi.commitments.merkle` | fast | yes |
| test_native_cheap.py (16) | native kernels of the recomputed ops | native (:13), skipped on Windows | yes |
| test_plan_auto.py (7) | policy `auto`; `perf.py` writes `chosen` | slow (143 s alone, `F2b_Q:262`) | yes |
| test_plans.py (25 functions, 549 cases) | commitment plans; λ for every model and mode; forgeries; transcript; bench naming, `plan_bytes`, `perf --random-init` | slow (718 s, `F2b_Q:260`) | yes, but :1120 pins the course report's commitment as the library default (B.6) |
| test_prob_max.py (6) | pmax 255 and 2047 in `transformer`; `opt_builder.attention_var` equality | med; HF for 1 test (:118) | yes |
| test_protocol.py (20) | path protocol end to end, forgeries | fast | yes |
| test_pruning.py (5) | `build_decoder(prune_last=True)` | med | yes |
| test_real_llama.py (25) | `real_llama` builders | med (52 passed in 53.8 s, `F2b_Q:574`); HF (:42) | yes |
| test_real_weights.py (12) | `real_weights` OPT builder and `prune_last`; `v1_bytes` and `perf --real` with a fake checkpoint | med/slow (15 passed in 169 s, `F2b_Q:262`); HF (:20); pyarrow (:297) | yes |
| test_recipe_equivalence.py (1) | `9_quality/opt_builder.build_opt` equals `real_weights.build_opt_from_hf` at the certified recipe | med; HF (:14) | only while `opt_builder` (the F2a search copy, `opt_builder.py:1`) stays |
| test_sampler_tv.py (11) | `9_quality/sampler_tv.py`, exact output law | med (22 passed, `E2b_sampler_tv.md:352`) | yes |
| test_sampling.py (2) | path sampling | fast | yes |
| test_serve_ab.py (8) | `7_reexec/serve_ab.py`: U1 logits, byte-identical proof on request, CLI, out-of-memory skip | fast (17 passed / 1 skipped in 10.3 s, `A7_serving_overhead.md:173`); 1 CUDA parameter (:140) | yes |
| test_setup_cache.py (5) | setup cache | fast/med | yes (the package uses it) |
| test_setup_proof.py (5) | setup proximity proof; split-commitment attack | med | yes |
| test_sp2027_numbers.py (5) | `sp2027_numbers.py`: synthetic records into tables; stored records regenerate the paper's tables, compared with `paper_sp2027/sections` (:182-190) | fast/med; matplotlib; data plus paper source | yes, but brittle (A.4) |
| test_token_sampler.py (4) | token sampler, end to end | fast/med | yes |
| test_torch_bridge.py (6) | torch module against the traced network | fast | yes |
| test_v1_buckets.py (14) | V-V1N value products are bit-identical | med; native for 2 tests (:141, :181) | yes |
| test_wire.py (21) | `run_query(wire=True)` in C/K/Kpre; malformed bytes; GPU client; bench wire tags | med/slow; GPU parameters; data: MNIST model (:594) | yes |
| test_zeroshot.py (10) | `9_quality/zeroshot.py` formats, scoring, padding | fast/med (9 passed, `F2c_zero_shot.md:61`); HF (:13) | yes (quality.tex:68) |

Support files, all kept:
- `conftest.py` (:1-93): random-network fixtures and `protocol_pair`.
- `claimcodec_reference.py`: the PVC3 encoder of commit `6c2059a`, the byte-equality specification (:1-3).
- `gkr_triton_mock.py`: CPU stand-ins for the Triton kernels.

### A.3 Tests that load experiment scripts by path (refactor hazards)

**18 test files** load experiment scripts by hard-coded paths (`Path(__file__).resolve().parents[1] / "experiments" / "<N_name>"`). They touch 25 distinct script files.

| Test | Script(s) | Lines |
|---|---|---|
| test_adaptive_game | 3_sampling_fixes/adaptive_game.py | :44-48 |
| test_comparison_tables | 5_comparison/{aggregate, validate_extrapolation, paper_assets, text_numbers, count_outcomes} | :17-25 |
| test_contention | 4_defence_benchmark/bench; 5_comparison/{aggregate, fingerprint_check, count_outcomes} | :28-36, :196, :240, :252, :259, :275, :285, :304 |
| test_ev_breakeven | 5_comparison (sys.path) → ev_breakeven | :10 |
| test_f3b_real_llama | 8_v1/v1_bytes.py, 6_improvements/gen_bench.py | :26-28, :42, :70 |
| test_f6_package | gen_bench, setup_timing, 7_reexec/reexec, h2d_bandwidth, perf; slurm/sp2027.sh run through bash | :32, :35-37, :54, :75, :86, :94, :105, :120 |
| test_llama_attrib | 9_quality (sys.path) → llama_attrib | :46 |
| test_lookups | 4_defence_benchmark/bench, 6_improvements/plan_bytes | :619-627, :634, :666, :686 |
| test_plan_auto | 6_improvements/perf.py | :132-137 |
| test_plans | 4_defence_benchmark/bench.py; 6_improvements/{plan_bytes, perf, ab_verifier} | :1137-1147, :1154-1166, :1201 |
| test_prob_max | 9_quality (sys.path) → opt_builder | :79 |
| test_real_weights | 8_v1/v1_bytes, 6_improvements/perf, 9_quality/common | :285-290, :299-301, :319, :342, :355, :377 |
| test_recipe_equivalence | 9_quality/opt_builder.py | :22-24 |
| test_sampler_tv | 9_quality/sampler_tv.py | :23-27 |
| test_serve_ab | 7_reexec/serve_ab.py (which loads reexec) | :24-32, :114 |
| test_sp2027_numbers | 5_comparison/sp2027_numbers.py (loaded at import time, :34) | :23-31 |
| test_wire | 4_defence_benchmark/bench.py | :614-619 |
| test_zeroshot | 9_quality/zeroshot.py | :16-18 |

Scripts also load other scripts, so moving one file breaks the tests of another:
- `perf.py` loads `ab_verifier.py` (:162), `bench.py` (:207) and `v1_bytes.py` (:237).
- `gen_bench.py` loads `serve_ab` (:74) and `v1_bytes` (:209).
- `setup_timing.py` loads `gen_bench` (:62).
- `serve_ab.py` loads `reexec` (:436).
- `plan_bytes.py` imports `ab_verifier` (:57-58); `wire.py` imports `plan_bytes` (:41-42).
- `v1_bytes.py` imports `9_quality/ppl_llama` and `common` (:69-70, :95-102).
- `ppl_llama.py` imports `opt_builder` and `ppl_2048` (:67-69).
- `zeroshot.py` imports `opt_builder`, `ppl_2048` and `common` (:287, :360-361).
- `sp2027_numbers.py` imports `count_outcomes` and `paper_assets` (:861, :984); `count_outcomes.py` imports `paper_assets` (:251-252).

**The main hazard is `ab_verifier.py`.** The 6_improvements README calls it a `--base` A/B dev harness (`6_improvements/README.md:15-19`). But `perf.py` and `plan_bytes.py` depend on its `CNNS`, `build`, `Case` and `random_queries` (`ab_verifier.py:41`, `:92`), and `test_plans.py:1185-1201` checks this. If the A/B tools are dropped, move the CNN builder into the library first.

**Tests also import other tests' helpers, so renaming a test file breaks others:**
- `from test_plans import _TINY/_planned/_verifiers/_cnn/_folded/_matrix` in 12 files: test_comparison_tables:40, test_contention:133/194, test_cut_protocol:20, test_generation:20, test_layouts:25, test_lookups:37, test_plan_auto:19, test_prob_max:23, test_pruning:24, test_setup_proof:27, test_token_sampler:16, test_v1_buckets:37.
- `import test_wire` (test_hostmem:15) and `import test_cut_protocol` (test_v1_buckets:36).
- Fix: move these helpers into `conftest.py` or a `tests/_helpers.py`.

**Module-name leakage.** Loaders register `sys.modules[name]` (test_real_weights.py:288, test_plans.py:1164) and insert folders into `sys.path`. Generic names (`common`, `perf`, `bench`) can then collide if the refactor creates two files with the same name, giving order-dependent failures.

### A.4 Other brittle couplings

- **test_sp2027_numbers ties the suite to the paper text.** It reads `../paper_sp2027/sections` and asserts exact counts: "63 base cells … identical", "30 cells identical" (`tests/test_sp2027_numbers.py:185-190`). Any table edit in the paper breaks the test.
- **test_comparison_tables needs the stored course runs:** `tables_l40s_improved{,2,3}` (:18, :94, :131, :173).
- **test_f6_package needs bash and the exact `slurm/sp2027.sh` path** (:32, :113-120).
- **The friend's smoke depends on test paths.** The `sp2027.sh smoke` tier runs a fixed list of test paths (`sp2027.sh:143-147`): test_gkr_triton, test_gkr_triton_cpu, test_gpu_verifier, test_gpu_exactness, test_cut_protocol, test_token_sampler, test_extfield, test_logup, test_cut, test_native_cheap, test_v1_buckets, test_generation, test_setup_proof, test_fs_tree, test_serve_ab, test_f6_package. Renaming any of these breaks the smoke.
- **The older `slurm/smoke.sbatch` is out of date.** It runs the whole suite and fails on any skip (`smoke.sbatch:45-56`), but:
  - it ignores only `test_real_weights.py` when `transformers` is missing (:43-44), while 6 more files are now `transformers`-dependent;
  - it does not set `PVI_SLOW=1`, so `test_logup.py:185` would skip and the job would fail;
  - it does not check `CXX`, so the native tests would skip;
  - it fingerprints against `raw/` (:77).

### A.5 Tests (or parts) covering code that may be dropped

- **test_comparison_tables.py: the course-report parts.**
  - `test_sec45_compares_both_protocols_on_every_configuration` (:89-100);
  - `test_table3_follows_the_setting_rule` (:102);
  - the count-macro tests for `report/tables/counts*.tex` (:349-372, :407);
  - `test_text_numbers_compares_and_knows_pending` (:375). `text_numbers.py` checks `report/main.tex` (`text_numbers.py:1`, `:569`).
  - Keep :28-87 and :149-337. These protect `aggregate` and `paper_assets`, which the S&P pipeline imports.
- **test_contention.py:250-262 (`fingerprint_check`).** The "1,978 hardware-independent values" claim is course-only; the S&P paper has no match. The tool itself is generic and useful for checking the final run.
- **test_plans.py:1185-1201 (`perf --random-init` through `ab_verifier`)**: see A.3.
- **test_recipe_equivalence.py and test_prob_max.py's `opt_builder` test**: only needed if `9_quality/opt_builder.py` stays. It is a "copy of `build_opt_from_hf` with quality options" (`opt_builder.py:1`), used by ppl_2048, ppl_llama and zeroshot.
- **test_llama_attrib.py's script tests**: only if `llama_attrib.py` is kept. The paper's attribution table is referenced at `quality.tex:4-5` and in `ext_generation.tex`.
- **Test names that encode plan IDs or course sections** (rename when moving):
  - `test_f6_package.py`, `test_f3b_real_llama.py`, `test_v1_buckets.py` (V-V1N);
  - `test_sec45_*`, `test_table3_*` (test_comparison_tables:89, :102);
  - `test_the_default_policy_is_the_reports_commitment` (test_plans:1120).

---

## B. Documentation

### B.1 READMEs and guides

| File | Targets | State |
|---|---|---|
| README.md (25 lines) | course paper | "Final project, Trustworthy Machine Learning (Tel Aviv University …)" (:3) names the institution. Points to `report/` (:18) and to the course tarball (:23-25). Must be replaced. |
| code/README.md (303) | course paper | Every results-map row cites `report/main.tex` sections (:182-238). Tarball recipe for the course (:282-303). Runtime claim (:109-110) conflicts with the artifact README. Reusable sections: install (:19-78), layout (:115-155), stored roots (:157-180). |
| code/src/pvi/README.md (32) | course sections | Misses 14 `fullcheck` modules: attention_kernels, buckets, codec_kernels, cut, cut_protocol, extfield, gkr_triton, logup, logup_gkr, native_kernels, real_llama, setup_cache, setup_proof, token_sampler. |
| experiments/0_train_models/README.md (27) | neutral | fine |
| experiments/1_reproduction/README.md (23) | course "Sec. 4.2" (:1) | renumber |
| experiments/2_attack/README.md (52), REAL_LLM.md (109) | course "Sec. 3.2, 4.1, 4.2" (README:1; REAL_LLM:1) | renumber; `inrange_llm.py` is not described in the README |
| experiments/3_sampling_fixes/README.md (39) | course "Sec. 3.3, 4.2" (:1) | does not mention adaptive_game*.py |
| experiments/4_defence_benchmark/README.md (391) | course "Sec. 4.2-4.6" (:1); 21 TAU/SLURM mentions | the SLURM section (:153-200) can be reused |
| experiments/5_comparison/README.md (148) | course; writes ../report | no mention of sp2027_numbers.py or ev_breakeven.py |
| experiments/6_improvements/README.md (692) | course "Sec. 3.6.2" (:3, :9-13); 33 "laptop" mentions | development diary in README form. Cites `perf_*_laptop_*.json` files that exist only as `.jsonl` (3 references). |
| experiments/7_reexec, 8_v1, 9_quality | none | **no README** |
| code/artifacts/comparison/README.md (168) | course runs | still needed: `sp2027_numbers --base` defaults to these runs (`sp2027_numbers.py:22`) |
| code/artifacts/results/zkllm_l40s/README.md (67) | "not used in the final paper" (:1, :6) | **out of date**: the S&P evaluation compares against "zkLLM's stored runs on the same L40S" (`evaluation.tex:311`; `ext_discussion.tex:126`) |
| report/README.md (53) | the course paper | rebuild recipe (:15-27) |
| scripts/artifact_README.md (82) | S&P (anonymous snapshot) | the best base for the new top-level README. Has a results map (:39-52) and the certified-model digests (:58-77). The hardware section is a stub (:79-82). |
| docs/L40S_RUN_SP2027.md (162) | S&P final run | the basis of the run guide. Contains role and date text ("For the teammate …" :3; dates :4-6, :59, :123-126), plan references (:4, :121) and team-only instructions (:152-162). |
| docs/submission/HOTCRP_FIELDS.md (45) | S&P (internal) | `[TEAM]` markers; "1,100 tests" (:12) |
| docs/improvements/README.md (39) | S&P diary index | Points to the superseded `SP2027_PLAN.md` (:5; see `SP2027_PLAN.md:3`). Statuses are out of date (A7 "pending" :12, B5 "in progress" :14, C1/C2 "measuring" :17). 9 diary files are not indexed: D6a, EV, F6, FS1, G3_inrange, H4, PS2, RX, and the two F6 dry-run listings. |

### B.2 `docs/improvements/*` (development diary), one line each

Each line is taken from the file's title and status lines (:1-6).

- **A1** int8-GEMM column opening and fold: opening 9.7x faster.
- **A2** int8 forward pass: about 2x.
- **A3** threaded claim encoder (2.1-2.3x); the fused Triton encoder is a negative result, opt-in through `PVI_FUSED_CODEC`.
- **A6** Kpre precompute on native and int8 products (106.9 → 12.7 s).
- **A7** prover against unverified serving and batching (pilot job 1008674; L40S still pending).
- **B2_B3** GPU claim decoder and fused exact attention (2.7x).
- **B5** native exact CPU-verifier kernels.
- **C2** V1: int8 values with logUp-GKR windows (cites V1_SPEC outside the repo).
- **C4a** batch byte model on Llama-2-7B.
- **D1_D2** Fiat-Shamir binds the graph; SHAKE-256 transcript.
- **D4_D5** one-time setup proximity proof; split-commitment attack.
- **D6a** reproducible certified model and commitment digests.
- **E1** generated response proved as one prefill.
- **E2** seeded integer Gumbel-max sampler.
- **E2b** exact sampler law and its TV distance to softmax sampling.
- **EV** deployment break-even (draft).
- **F1** re-execution baseline.
- **F2** quality attribution and SmoothQuant through the norm gains.
- **F2a** quality at 2,048 tokens; 11-bit probabilities; certified recipe.
- **F2b_Q** real Qwen3/Llama builders (1,276 lines).
- **F2c** zero-shot accuracy.
- **F3a** `prune_last` with real OPT; real-weight cells.
- **F3b** real Llama (OpenLLaMA-7B) end to end.
- **F6_final_run_package** package v2, rehearsals, number pipeline; F6_dryrun*.txt are the expected job listings.
- **FS1** Fiat-Shamir hashing at memory speed.
- **G3_inrange** in-range edits against range checks.
- **G3_min** adaptive game.
- **H4** anonymous artifact export and HotCRP fields.
- **I_server_setup** quota, Triton and compiler on the cluster.
- **PS2** lower-rate commitment plans.
- **PS2c** setup cache plus cnn18c.
- **RX** strongest re-execution baselines.
- **V_LONG_b** prompts of 2k-16k tokens.
- **V_V1N** fast V1 CPU verifier.

These files are the main source of provenance: the job, commit and host for each measurement.

### B.3 Other documents

- **Plans.** `SP2027_PLAN.md` (121 lines; superseded, :3) and `SP2027_PLAN2.md` (1,296) are planning only. They also contain an author's name (in the plan text). Not for the submission branch.
- **paper_sp2027/*.md.** `CUT_PLAN.md`, `H5_PASS.md`, `REVIEW.md`, `REVIEW2.md` and `drafts/*.md` are internal editorial and mock-review records.
- **Orphan paper section.** `paper_sp2027/sections/generation_quality.tex` is tracked but not `\input` anywhere. `CUT_PLAN.md:139` says the integrator removes it.
- **One-off diagnostics.** `docs/diag_1009531.txt` and `docs/pfxq_diag.txt`. The former is nearly a copy of `code/artifacts/results/9_quality/llama_diag_1009531.txt` (they differ from line 1).
- **Paper source comments.** `paper_sp2027/sections/*.tex` hold 632 comment lines of process notes (e.g. `evaluation.tex:1-15`: "Sources: wt_sp2027/report …", plan and REVIEW2 IDs, dates).
- **Live `\todo` markers** (non-comment lines; macro at `main.tex:29`):
  - abstract (`main.tex`): 2;
  - body sections: 41, of which `evaluation.tex` has 25;
  - extended sections: 35;
  - orphan `generation_quality.tex`: 6.

  `export_anonymous.py` exports only the PDF of the paper (INCLUDE at `:30-34`), so the comments matter only if the source is shared.

### B.4 Comment style in code

Scanned files: `code/src` (62 files, 20,366 lines), `code/experiments` (73 .py/.sh/.sbatch files, 19,801 lines), `code/tests` (55 files, 15,102 lines). The table counts matching lines. Counts are approximate: several patterns also hit ordinary words.

| Pattern | src | exp | tests | Examples |
|---|---:|---:|---:|---|
| Explicit plan references ("Plan 2", "plan item", "(plan X", SP2027_PLAN) | 13 | 28 | 14 | buckets.py:1 "(`SP2027_PLAN2.md` item V-V1N)"; real_llama.py:1; setup_cache.py:1 "(plan PS2, option (c))"; sp2027_numbers.py:1 "Plan 2 F6 part 11 (H3)"; gen_bench.py:1 "(chg 12)"; test_f6_package.py:1 |
| Distinctive plan IDs (F2b-Q, F2a, F3b, D6a, C4a, E2b, PS2, FS1, V-V1N, V-LONG, G3-min, G3.5, K-G3c) | 27 | 62 | 25 | sp2027.sh (13), protocol.py:412 / :1028 "V-V1N part 3/4", native_kernels.py:621, test_adaptive_game.py:1 "K-G3c" |
| Broad plan IDs (adds A1-A7, B2-B5, C1-C2, D1-D5, E1-E2, F1-F6, H1-H5, EV, TH) | 40 | 141 | 34 | includes some false positives |
| SLURM job numbers | 1 | 2 | 3 | real_llama.py:479; params_table.py:89; llama_diag.py:22; test_f3b_real_llama.py:68; test_llama_attrib.py:29; test_real_llama.py:706 |
| Dates | 0 | 0 | 0 | only a citation year (`pvi/__init__.py:6`) |
| "the user", team, friend | 0 | 0 | 0 | none |
| Genuine TODO / FIXME / XXX | 0 | 0 | 0 | the 11 grep hits are a variable `todo` (bench.py:808-811, ppl_llama.py:356-364), `\todo` parsing (sp2027_numbers.py:898-943) and " hacked" (real_llm.py:262) |
| "laptop" | 1 | 18 | 1 | commitment.py:125; adaptive_game_llm.py:36; gen_bench.py:7; setup_timing.py:7; serve_ab.py:14, :130; llama_attrib.py:640; test_logup.py:185; plus `*_laptop*` result file names |
| Cluster / TAU / node names in .py | 1 | 2 | 0 | datasets.py:14 "as on the TAU cluster"; paper_assets.py:137 "n-801, n-804"; adaptive_game_llm.py:43. The .sh/.sbatch files have about 100 more (infrastructure, expected). |
| Old course report ("the report", "report/", "(report Sec.") | 24 | 57 | 14 | bench.py:1, :88, :217, :926, :966, :1016, :1047, :1103; 1_reproduction/run.py:1; real_llm.py:1; count_outcomes.py:1, :3, :6; paper_assets.py:1, :4; text_numbers.py:1, :569; protocol.py:157, :223 (`policy="paper"`, "the report"); real_weights.py:1; transformer.py:117 |
| Diary or research-folder paths | 15 | 5 | 5 | **`V1_SPEC.md` is cited 16 times in 14 files and is not in the repository** (cut.py:1, logup.py:1, :13, logup_gkr.py:1-2 → `sp2027_research/.../spec_toy.py`, protocol.py:141, graph.py:156, gkr_triton.py:1, test_cut.py:1, test_logup.py:1). Also `docs/improvements/...` in buckets.py:21, claimcodec.py:1546, codec_kernels.py:25, extfield.py:5, real_weights.py:175, sp2027.sh:80, :124. |
| Temporal "today" | 16 | 15 | 14 | v1_bytes.py:1 "against today's (v0)"; v1_bytes.py:332 (the `PVI_V1_BUCKETS=0` path is called "today's verifier"); v1_phases.py:1; buckets.py:44; sp2027_numbers.py:23, :42 |
| Reviewer IDs | 0 | 3 | 0 | inrange_llm.py:1 "mock review, Reviewer A, weakness 2"; sp2027.sh:260 and sp2027_numbers.py:678 "C-Q4" |
| Commit hashes | 0 | 3 | 2 | strong_gpu.sh:19, :207; paper_assets.py:13; claimcodec_reference.py:1; test_claimcodec.py:676. Provenance use is acceptable, but better as run names or tags. |
| Bug history and dangling references | – | – | – | test_plans.py:1187 "(it fell through to the decoder branch: KeyError)"; perf.py:1 "on the old and the new code"; opt_builder.py:6 cites `attribution.py`, which is not in the repo (it is on branch f2-quality, `F2_quality.md:4-6`); opt_builder.py:28 "Ported from branch f2-quality" |

**Sample of 20 files**, with the number of stale lines found in each:
- `fullcheck/protocol.py`: 3 plan, 4 report, 1 V1_SPEC.
- `transformer.py`: 3 plan ("Plan 2 F2a" :57, :61), 1 report (:117).
- `real_weights.py`: plan, diary and report at :1 and :175.
- `real_llama.py`: 5 plan, 1 job (:479).
- `cut_protocol.py`: 4 V-V1N, V1_SPEC at :1.
- `buckets.py`: 3 plan, 1 diary.
- `graph.py`: V1_SPEC (:156).
- `native_kernels.py`: 5 V-V1N/RX (:621, :668, :807).
- `claimcodec.py`: 2080 Ti note and diary path (:1544-1546).
- `pvi/protocol/verifier.py`: clean.
- `attacks/tamper.py`: clean (the "report" hit at :120 is a variable).
- `bench.py`: 11 course-report, 5 hardware or run-root notes.
- `perf.py`: F6/F2a IDs (:153, :196).
- `v1_bytes.py`: 4 plan, "today's" in its help text.
- `sp2027_numbers.py`: 9 plan, 4 course-report, 1 reviewer.
- `serve_ab.py`: plan A7 and C4a/C4c (:1, :37, :39); "laptop" (:14).
- `gen_bench.py`: "Plan 2 F6 (chg 12)" (:1), F3b (:29), "laptop" (:7).
- `slurm/sp2027.sh`: 19 plan, 2 diary paths, 1 reviewer.
- `test_f6_package.py`: plan (:1).
- `test_real_llama.py`: plan (:1), job (:706).

What the sample shows:
- The style is professional: there are no TODOs, no conversational remarks, no "the user".
- The problems are four: references to the internal plan, references to the course report's section numbers, references to documents missing from the repository (V1_SPEC.md, attribution.py, branch names), and temporal wording such as "today's code" for variants that are now options.

**Environment switches.** About 50 distinct `PVI_*` environment variables act as options instead of CLI arguments. Examples: `PVI_V1_BUCKETS` (18 uses), `PVI_FOLD_FAST`, `PVI_FUSED_CODEC`, `PVI_CPU_ATTN`, `PVI_NATIVE_*`, `PVI_LEGACY_WEIGHT_KEY` (11), `PVI_C_POLICY`, `PVI_TF32`. This bears on the requirement that every parameter be a CLI argument: either convert them to flags or document each one.

### B.5 Proposed documentation set for the submission branch

1. **`README.md` (repository root)**, replacing the course README.
   - Contents: what the artifact is, layout, three entry points (quick tests; regenerate the paper's tables from stored records; the final run), and a map from paper sections to folders, in the order of `paper_sp2027/main.tex:76-82`.
   - Built from: `scripts/artifact_README.md:1-56` and `code/README.md:115-155` (with new section numbers).
2. **`docs/INSTALL.md`.**
   - Contents: pinned versions, optional `transformers` and pyarrow, C++ compiler and Triton, datasets and models, the Llama licence note.
   - Built from: `code/README.md:19-78`, `scripts/artifact_README.md:19-28`, `docs/L40S_RUN_SP2027.md:15-36`, `docs/improvements/I_server_setup.md`.
3. **`docs/REPRODUCE.md` (claim map).**
   - Contents: one row per paper label (`tab:eval-llm`, `tab:eval-v1`, `tab:eval-reexec`, `tab:eval-compare`, `tab:quality`, `tab:params`, `fig ev_breakeven`, `sec:break` numbers, extended tables): command, stored input, output, hardware, time. Three tiers: stored records without a GPU (`sp2027_numbers.py --stored`, as in `test_sp2027_numbers.py:182-190`), small CPU re-measurements, and the full GPU run.
   - Built from: the format of `code/README.md:182-238`, `artifact_README.md:39-52`, the `sp2027_numbers.py:1-44` docstring, `L40S_RUN_SP2027.md:152-162`, and the "Results" sections of the diary. The claims map from the inventory step should feed it.
4. **`docs/RUN_GUIDE.md` (final run on any SLURM cluster or locally).**
   - Built from: `docs/L40S_RUN_SP2027.md` (with dates, role wording and plan IDs removed), the `sp2027.sh` header, `F6_final_run_package.md` (tiers, estimates, hotfix rule), `F6_dryrun.txt` as the expected job listing, and `4_defence_benchmark/README.md:153-200`.
   - Add: a table of every `PVI_*` switch, or retire the switches into flags. State clearly that `sp2027.sh smoke` is the smoke; fix or remove the out-of-date `smoke.sbatch` (A.4).
5. **`docs/PROTOCOL.md` (protocol and format specification).**
   - Contents:
     - modes C, K and Kpre: messages and checks;
     - the Fiat-Shamir transcript (D1_D2, FS1);
     - commitments and plans (PS2, PS2c, `6_improvements/README.md` §2, §4, §7);
     - the PVC3 wire format (claimcodec docstrings, `claimcodec_reference.py` as the normative reference, A3);
     - V1: the cut, logUp-GKR, F_{p^8} (C2);
     - the setup proof (D4_D5); the token rule and sampler (E1, E2, E2b);
     - the certified recipe and the C_M digests (`artifact_README.md:58-77`, D6a, F2a §3.3);
     - the JSON-line record schema read by `sp2027_numbers` (fields in `test_sp2027_numbers.py:24-110`).
   - Link the paper appendices (`appendix_defence`, `appendix_v1`, `ext_proofs`) for the proofs.
   - **Import the content of V1_SPEC.md here** (it lives outside the repo, `C2_v1_proof_size.md:3`) and repoint the 16 code references to it.
6. **`docs/RESULTS_PROVENANCE.md`.**
   - Contents: for each stored file, the producing command, commit, host class, date and job.
   - Built from: `code/artifacts/comparison/README.md`, `code/README.md:157-180`, `zkllm_l40s/README.md` (updated to "used"), and the job/commit header lines of each diary file.
   - Move the 6 job numbers and the commit hashes now in code comments here.
   - Flag the paper numbers whose raw logs are not in the repository, for example the A7 pilot "on the server clone" (`A7_serving_overhead.md:282`). Either commit those logs or mark them as replaced by the final run.
7. **Per-folder READMEs** for every experiment folder (adding 7_reexec, 8_v1 and 9_quality), plus `tests/README.md` and an updated `src/pvi/README.md` (the 14 missing modules).
   - Template: purpose; paper labels; scripts table; options (generated from `--help` so it cannot drift; 62 of 65 scripts already have a command line in their module docstring); inputs and downloads; outputs; hardware and time; superseded variants as non-default options, with the paper sentence that mentions each.
   - `tests/README.md` gets the runtime classes from A.2, the optional dependencies, `PVI_SLOW` and `PVI_TEST_DEVICE`, and the smoke subset (`sp2027.sh:143-147`).
8. **Left off the submission branch (kept on the development branch):**
   - `SP2027_PLAN*.md`, `docs/improvements/*` (distilled into documents 5 and 6), `docs/diag_*.txt`, `docs/pfxq_diag.txt`, `docs/submission/`;
   - `paper_sp2027/{CUT_PLAN,H5_PASS,REVIEW,REVIEW2}.md`, `drafts/`, `generation_quality.tex`.
   - **`report/` needs a decision.** The code still writes into it: `paper_assets.py:4` (`../report/figures`, `../report/tables`), `text_numbers.py:569` (default `report/main.tex`), `count_outcomes.py:3, 6`.
   - Update `export_anonymous.py` INCLUDE (:30-34) so the new `docs/` are exported. Its REWRITES and deny list (:42-52) already cover the home-directory paths of a team member that appear inside the stored `raw_l40s` records.

### B.6 Higher-level decisions to raise in the review

- **The default commitment policy is still the course paper's.**
  - Library: `commit_graph(policy="paper")` (`protocol.py:220`).
  - Scripts defaulting to `"paper"`: `bench.py:1015`, `perf.py:177`, `serve_ab.py:867`.
  - Scripts defaulting to `"auto"`: gen_bench, v1_bytes, setup_timing and others.
  - The package and number pipeline use `cnn18c` (`sp2027.sh:66`, `sp2027_numbers.py:1083`, `setup_cache_check.py:31`).
  - Your rule is that superseded variants stay as non-default options, so one default should apply everywhere. Changing it changes bench cell names (`bench.py:1103`) and breaks `test_plans.py:1120`.
- **Out-of-date claims to fix before submission:**
  - the zkLLM README's "not used in the final paper";
  - the test-count claim in the HotCRP text;
  - the two suite-runtime claims.

### B.7 Could not determine

- Actual runtimes and the expanded number of test cases, since nothing was run or collected.
- Which `6_improvements` A/B harnesses and which `9_quality` scripts the inventory keeps. The "mixed" rows above depend on that.
- Whether the content of V1_SPEC.md and `attribution.py` (both outside this repository) should be imported.
- Whether `report/` stays.
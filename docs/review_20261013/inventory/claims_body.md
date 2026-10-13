# Provenance map for the S&P 2027 submission body (main.tex)

**Scope covered:** the abstract (`main.tex:71-74`) and every file the submission build pulls in with `\input`, in order: `intro`, `attack`, `defence`, `v1`, `generation`, `evaluation` (with `quality`), `discussion`. In the submission build `main.tex:106` also pulls in `appendix_short.tex`, which pulls in `app_params.tex` (tab:params). Those are covered briefly in §10.

**Method:** read only. I recomputed some values from stored JSON and CSV files with small Python and awk reads. I did not run `params_table.py`, `setup_bytes.py` or any model.

**Path abbreviations.** R = repo root (`wt_sp2027`). P = `R/paper_sp2027/sections/`. DI = `R/docs/improvements/`. AR = `R/code/artifacts/results/`. AC = `R/code/artifacts/comparison/`. EX = `R/code/experiments/`. LIB = `R/code/src/pvi/`. SR = `<scratchpad>/sp2027_research/`, which is outside the worktree and so outside any branch.

**Kinds.**
- **M**: measured in a run.
- **Mo**: modelled (byte model, formula, theorem, or arithmetic on other numbers).
- **L**: taken from the literature.
- **U**: provenance unknown.

"Rec✗" means the run's record is not in the repository; only a doc or log quotes it.

---

## 1. Abstract (main.tex)

| Loc | Value | Kind | Source / record | Script, command | Library |
|---|---|---|---|---|---|
| main.tex:72 | path misses one site with prob. 1−1/N | Mo | formula; DI-independent | — | LIB/experiments/analysis.py `acceptance_probability` |
| main.tex:72 | OPT-6.7B: one neuron changed the next token, 40/40 prompts | M | AR/real_llm_attack.json `untargeted.succeeded=40`; EX/2_attack/REAL_LLM.md:16 | `EX/2_attack/real_llm.py` via `real_llm.sbatch` (one A5000, fp16) | HF hooks in real_llm.py |
| main.tex:72 | ≥1−qd/N for any Spot_d(q) verifier; 2^−λ with O(λ) queries for a linear-code commitment | Mo (theorem) | thm:dich, P/attack.tex:144-160; proofs in P/appendix_short.tex:73-122 | — | — |
| main.tex:72-73 | CPU client verifies a 2,048-token Llama-2-7B prompt 4.7× faster than the fastest exact re-execution (non-interactive, full model) | M | 19.64 s: AR/vlonga_v1retime_w1c_1007816.jsonl (job 1007816). 93.2 s fp32: DI/RX_reexecution_baselines.md:32,55 (job 1006979, Rec✗). The `\todo{EPYC}` is meant to become `\CpuRatioN` from `sp2027_numbers.py` | `EX/6_improvements/perf.py --model llama2-7b --seq 2048 --modes Kpre --fs …` (DI/FS1:109,127); `EX/7_reexec/reexec.py --model llama2-7b --seq 64 2048 --layers 2 32 --variants int int_fp32gemm float bf16 dynint8 --device cpu --threads 8 --reps 3` (RX:74-75) | fullcheck/protocol.py `Verifier`, native_kernels.py |
| main.tex:73 | V1 proofs 2.0–2.3× smaller at 2,048 tokens with random weights | M | tab:eval-v1. DI/C2_v1_proof_size.md:128-135 (job 1005287); log only in SR/server_out/v1b_1005287.out | `EX/8_v1/v1_bytes.py` (the exact flags of job 1005287 are not written down) | cut_protocol.py `cut_proof_bytes`, `run_cut_query` |
| main.tex:73 | 1.7× on the real OPT-6.7B | M | DI/F3a_real_weight_runs.md:301,312 (B3: 4,649.9→2,680.5 MB = 1.73×; jobs 1008673/1009431), Rec✗ | `v1_bytes.py --model opt-6.7b --real facebook/opt-6.7b --smooth 0.55 --calib-seq 2048 --prune-last --seq 2048 --modes Kpre --fs --lean --lmax 27 --threads 8 --device cpu` (F3a:258) | real_weights.py `build_opt_from_hf` |

## 2. §1 Introduction and §2 Related Work (P/intro.tex)

| Loc | Value | Kind | Source | Script | Library |
|---|---|---|---|---|---|
| 30-31 | 32-byte Merkle roots | Mo (design) | — | — | fullcheck/commitment.py (SHA-256) |
| 39-40 | zkLLM ≈10 min on an A100, 2,048-token Llama-2-7B | L | 620 s in AC/tables/reported_curated.csv (zkLLM, Llama-2-7B) | `EX/5_comparison/literature.py` | — |
| 47-49, 53, 64 | path catches one node with prob. 1/N | L / Mo | anchuri2026verifiable | — | — |
| 65-66 | OPT-6.7B 40/40 "with values outside the neurons' natural ranges" | M | real_llm_attack.json. **Note:** `untargeted.within_natural_range = 0.125`, so 12.5% of the forged values were inside the natural range (REAL_LLM.md:16) | real_llm.py | — |
| 66-67 | OPT-1.3B and OPT-6.7B in-range edits also succeed | M | see §3, "A real LLM" | inrange_llm.py | — |
| 68-69 | dog-vs-cat ResNet-18: one neuron flips every test query; passes 511/512 | M | AC/tables_l40s/measured_summary.csv: `resnet18_224 attack_float flip_success` n=94 → 1.0; `sampling p_detect_penultimate` = 1/512 | `EX/4_defence_benchmark/bench.py` (`attack_float`, `sampling` cells) → `aggregate.py --platform l40s` | LIB/attacks/tamper.py |
| 70-71 | 1−2^−40 on Llama-2-7B needs ≈305,000 paths | Mo | AC/tables/analytic.csv (305,193) | `EX/5_comparison/analytic.py` (no CLI; model, seq and λ hard-coded at line 24) | analytic.py |
| 73-83 | Theorem: qd guesses, 1−qd/N, O(λ), O(λ+log L) per block | Mo (theorem) | thm:guess, thm:dich; block variant only in the extended version / drafts/th_a_block_mixing.tex | — | — |
| 83-87 | nine schemes; only TensorCommitments and the binary-classifier claim fail | L + Mo | tab:sok | see §3 | — |
| 108 | prover's folds and openings do not grow with T | Mo | defence.tex:141-144 | — | — |
| 109-110 | L40S proves 2,048-token Llama-2-13B 46× faster than zkLLM on an A100 | M/L | 803 s (reported_curated.csv) ÷ 17.6 s (tab:eval-llm base). REVIEW2.md:585 checked 46× | — | — |
| 111-112 | OPT-1.3B @2,048, one GPU: 4.1–8.6× exact integer forward, 19–40× fp16 | M | DI/A7_serving_overhead.md:294-295 (job 1008674, 2080 Ti). Records on server clone s3 (A7:285-286), Rec✗ | `EX/7_reexec/serve_ab.py` | — |
| 113 | 6.6 GB proof for 2,048-token Llama-2-7B ≈ int8 weights | M | AC/tables_l40s_improved2/llm_full_model.csv:281 (6,570,112,359 B) | bench.py → aggregate.py | — |
| 114-115 | V1 2.0–2.3× (random weights), 1.7× (real OPT-6.7B) | M | as in the abstract | — | — |
| 115-118 | Kpre CPU client 4.7× vs fp32 re-execution | M | as in the abstract | — | — |
| 131 | evaluation up to 13B parameters | M | tab:eval-llm | — | — |
| 143-145 | zkSNARK proofs from under 1 kB to tens of MB | L | reported_curated.csv | — | — |
| 166-167 | GPU confidential computing: below 7% overhead | L | gpucc; checked externally in REVIEW2.md:567 | — | — |
| 176 | CommitLLM 12–14% serving overhead | L | commitllm2026; REVIEW2.md:568-571 | — | — |
| 184 | (1−ε)N paths per layer | Mo | thm:dich(a) | — | — |
| 88 | disclosure | — | `\todo` (dates are open) | — | — |

## 3. §3 Attack (P/attack.tex)

| Loc | Value | Kind | Source / record | Script | Library |
|---|---|---|---|---|---|
| 71, 73-74 | (1−1/N)^s; k tampered queries caught w.p. ≤ k/N | Mo | — | — | — |
| 105-118 | thm:guess G_qd (fractional knapsack) | Mo (theorem) | appendix_short.tex:42-55; checks in SR/attack_theory/check_guessing.py (outside repo) | — | — |
| 132-135 | MNIST: edited neuron ranked first for 75% / 95% of 40 queries; 1/N = 0.2% / 0.4% | M | DI/G3_min_adaptive_game.md:160-169 (G1e); AR/adaptive_game_mlp_mnist.json | `EX/3_sampling_fixes/adaptive_game.py --queries 40 --layers 1 2 --calib train` (G3_min:352) | adaptive_game.py `fast_acceptance` |
| 144-160 | thm:dich: 1−qd/N; qd ≥ (1−ε)N; O(λ+log L) | Mo (theorem) | appendix_short.tex:73-122 | — | — |
| 171-172 | 2^−40 needs ≈27.7N paths | Mo | 40·ln 2 | — | — |
| 173-178 | bounds at every locality; d∣N | Mo | extended version (appendix_attack) | — | — |
| 184-187 | 7 models; "the implementation reproduces the published behaviour" | M | AR/reproduction.json | `EX/1_reproduction/run.py` | LIB/protocol (`session.run_protocol`, `verifier.Verifier`), zoo.py |
| 189 | MNIST: one tampered neuron passes 99.6–99.8% | M | AR/attack.json `baselines.single-neuron tamper, layer 1/2` (0.99805, 0.99609) | `EX/2_attack/run.py` | attacks/tamper.py `plan_single_neuron_flip`, `smallest_flipping_value`; experiments/analysis.py |
| 190-191 | substitutes corrupt "hundreds" of 778 trace nodes; ours 1–14 | M | attack.json `support_mean` 256–777.9; the 14 is defence.json `zero_blind_spot.mean_support` 14.35 | run.py; 3_sampling_fixes/run.py | attacks/baselines.py |
| 191-193 | 3×3 patch backdoor; clean queries unchanged; 99.96% of 2,300 runs (predicted 99.8%) | M | attack.json `backdoor` (0.99957, 2,300 runs, exact 0.99805) | 2_attack/run.py | attacks/backdoor.py `BackdoorAdversary`, `PatchTrigger` |
| 195 | uniform sampling detects 0.20% | M | AR/defence.json `sweep.uniform.detection_evasive` 0.001953 | `EX/3_sampling_fixes/run.py` | defences/sampling.py, defences/adaptive.py |
| 196-197 | contribution-weighted sampler detects 9.2% | M | defence.json `local-contribution.detection_evasive` 0.0923 | same | `LocalContributionSampler` |
| 198-199 | zeroing ≈14 of 512 neurons; 2,000/2,000 runs pass | M | defence.json `zero_blind_spot` | same | — |
| 199 | floor δ=1 reaches 2% | M | AR/floor_sampler.json `eps=1` 0.0198 (20 queries) | `EX/3_sampling_fixes/floor_sampler.py` | — |
| 202-205 | LP + double-oracle mixture ≤3.1% / 9.7%; 27.7/N = 5.4% / 10.8% | M | G3_min:109,116 (maxima over caps: 3.06%, 9.67%); adaptive_game_mlp_mnist.json | adaptive_game.py | adaptive_game.py `solve_mixture` |
| 207-209 | every CNN: one penultimate neuron flips; 1/84 (LeNet-5) to 1/512 | M | AC/tables_l40s/measured_summary.csv (`attack_float flip_success` n=94 per model; `p_detect_penultimate`). **Note:** "every … test query" means 94 queries per model | bench.py → aggregate.py --platform l40s | — |
| 209-210 | 2,316–14,182 paths for 2^−40 | Mo (from measured widths) | measured_summary `paths_needed_penultimate` | same | — |
| 210-211 | ≥305,193 paths in Llama-2-7B's FFN (11,008); 13.7 GB fp16 at 64 tokens | Mo | AC/tables/analytic.csv. The 13.7 GB is `anchuri_open_all_bytes` = 2·params + fp16 trace (analytic.py:28-30) | analytic.py | — |
| 213-214 | OPT-6.7B: FFN neuron at layer 8/16/24/31 → 40/40 | M | real_llm_attack.json (`layers_used` [8,24,31]) | real_llm.py | — |
| 214 | 95% Wilson interval 91.2–100% | Mo | computed from 40/40 (SP2027_PLAN2.md:972); no script found | — | — |
| 214-215 | backdoor "back" 40/40 with clean prompts unchanged; "hacked" 0/40 | M | AR/real_llm_attack_auto.json, real_llm_attack.json `backdoor` | real_llm.py (`--target auto`) | — |
| 216-217 | backdoor values exceed natural ranges; ≥454,248 paths | M/Mo | real_llm_attack.json `path_test.paths_for_2^-40` = 454,248; `within_natural_range` 0.0 | real_llm.py | — |
| 218-221 | ranges from 8,192 WikiText-2 train tokens; OPT-1.3B in range: 40/40 at layers 6, 12, 18, 23; median 2 neurons (max 60); per-cell median path probability ≤7.9e−4 (max 7.3e−3) | M | AR/inrange_llm_opt-1.3b_any.json; I recomputed 320/320, median k 2, max 60, largest cell median 7.93e−4, max 7.32e−3; DI/G3_inrange_attack.md:36-47 | `EX/2_attack/inrange_llm.py --model facebook/opt-1.3b --layers 6 12 18 23` (G3_inrange:101) | — |
| 221-223 | lower-only edits: 9, 31, 38, 39 of 40; median 6 neurons | M | AR/inrange_llm_opt-1.3b_down.json (recomputed median 6); G3_inrange:48-51 | `… --moves down` | — |
| 224-227 | best mixture catches 48%, 6.8%, 2.2% (layers 6/12/18); 27.7/N = 0.34%; 0.08% at layer 23; uniform 0.02–0.06% | M | DI/G3_min:293-305 (job 1008671). **Rec✗:** `adaptive_game_opt-1.3b.json` is not in AR | `EX/3_sampling_fixes/adaptive_game_llm.py --model facebook/opt-1.3b --layers 6 12 18 23 --prompts 40 --calib-windows 64 --calib-seq 128 --caps 90 95 99 100 …` (G3_min:254-257) | — |
| 229-231 | OPT-6.7B in range: 40/40 at every layer; median ≤1.3e−3 | M | AR/inrange_llm_opt-6.7b_any.json (recomputed 1.343e−3); G3_inrange:56-82 (job 1007804) | inrange_llm.py `--model facebook/opt-6.7b --layers 8 16 24 31` | — |
| 250 (tab:sok) | path protocol: 99.80% (N=512, s=1), 61.3% (s=250); 14,182 to ≥454,248 paths | M/Mo | attack.json `opening_budget` 0.6134; SR/attack_theory/sok_formulas_out.json | 2_attack/run.py; SR/attack_theory/sok_formulas.py (**outside repo**) | — |
| 255-256 | SLP: 3 of 161 chunks; 98.14% per audit; 37 / 160 audits | L + Mo | sok_formulas_out.json (REPRO/DERIVED) | sok_formulas.py (outside repo) | — |
| 260-261 | CommitLLM: 1–5% of responses, 10 of 80 layers; 87.5%; ≈99.4% at 5% audits | L + Mo | sok_formulas (DERIVED); REVIEW2.md:582 | same | — |
| 266-281 | TensorCommitments 1−\|S\|/L; NanoZK ≤3e−37 and 50%; SPEX 1−δ; IMMACULATE 1/T | L + Mo | sok.md / sok_formulas (outside repo) | same | — |
| 286 | TOPLOC/SVIP forged tokens pass: 100%; ≥95% | U | probably SR/attack_theory/sok.md; no repository record. REVIEW2 did not verify it | — | — |
| 314-317 | disclosure dates and replies | — | `\todo` | — | — |

## 4. §4 Defence (P/defence.tex)

| Loc | Value | Kind | Source | Script | Library |
|---|---|---|---|---|---|
| 51-54 | OPT-125M and OPT-1.3B rebuilt bit for bit across machines and thread counts; OPT-125M also across OS and torch versions; commitments across machines, threads and CPU vs GPU encoding | M | DI/D6a_reproducible_commitment.md:55-78 (jobs 1012438/9, 1012744/5). **Caveats:** (a) the cross-OS / cross-torch check covers only the *earlier* (legacy α=0.55) recipe; the certified recipe was not checked on Windows (D6a:83-86). (b) The CPU-vs-GPU root check covers OPT-1.3B only; OPT-125M was encoded on CPU only (D6a:52-58, 64-68) | `EX/9_quality/digest_check.py --model <M> --threads <T> [--commit --commit-device cpu\|cuda]` | real_weights.py `build_opt_from_hf`, `CERTIFIED_RECIPE` (real_weights.py:174); protocol.py `commit_graph` |
| 71-73 | 23-bit residual; 256-entry tables; 8- or 11-bit probabilities | Mo (design) | appendix_short.tex:159-173 | — | fullcheck/transformer.py, graph.py |
| 76-77 | exact int8 sums in int32 on tensor cores; float32 attention exact | Mo + tests | tests/test_gpu_exactness.py | — | attention_kernels.py |
| 78 | BabyBear p = 15·2^27+1 ≈ 2^31 | Mo | field.py:57 | — | field.py |
| 79-82 | reject \|z\| ≥ 2^29; 128Σ\|w\|+\|b\| < 2^29; difference < 2^30 < p | Mo | protocol.py:94 (`Z_BOUND`), `claim_bound` (:182) | — | protocol.py |
| 123-125, 133 | p^−r; O(r(N+k)T); column term < (k/n)^t | Mo | — | — | — |
| 156-161 | thm:sound ε_C, ε_K, Q·ε_K | Mo (theorem) | appendix_short.tex:175-199 | — | — |
| 170-173 | β = λ+log₂(2\|J\|); r=⌈β/log₂p⌉; r=5 at λ=128 | Mo | tab:params | `EX/6_improvements/params_table.py` | protocol.py `params_for` (:132), plans.py `column_bits` |
| 201-204 | +64 bits, Q ≤ 2^64, r=7 | Mo | tab:params | same | `params_for(grinding_bits=64)` |
| 211-212 | split commitment passes ≈2^−t: about 2^−23 per query (2^−34 under Fiat–Shamir), smallest Llama-2-7B tree | Mo | tab:params t_min 23/34 (cnn18c). Split attack measured on tiny models: tests/test_setup_proof.py (DI/D4:69-72) | params_table.py | setup_proof.py |
| 218-221 | setup proof 2.4 GB (a=100) / 9.8 GB (a=1,000); columns per query 65.6 → 113.5 / 79.4 MB; growth < 1% of the 2,048-token proof | Mo | SP2027_PLAN2.md:740-741 (regenerated for cnn18c). **DI/D4_D5:87 is stale:** its auto-plan figures are 2.8/11.3 GB and 124.7→168.5/134.9 MB. No stored output | `setup_bytes.py --model llama2-7b --policy cnn18c --amortise 100 1000`; `params_table.py --policy cnn18c` (both default to `--policy auto`) | setup_proof.py `setup_params`; plans.py |
| 224-225 | combined rows are 2.2% / 0.5% of the setup proof | Mo | SP2027_PLAN2.md:741. DI/D4:99 is stale (1.6% / 0.4%) | setup_bytes.py | — |
| 221 | setup time | — | `\todo{F6}` | `EX/6_improvements/setup_timing.py` (final run) | — |

## 5. §5 V1 (P/v1.tex)

| Loc | Value | Kind | Source | Script | Library |
|---|---|---|---|---|---|
| 47-50, 55-64 | rq/shift constants 2^29, 2^30, ±127; windows | Mo (design / lemma) | — | — | cut.py |
| 79 | 𝕂 = F[ξ]/(ξ^8−11); Fiat–Shamir needs degree ≥ 7 | Mo | extfield.py:1,27 | — | extfield.py |
| 91 | "proofs hold up to 4.3·10^9 leaves", p ≈ 2.0·10^9 | Mo / **U** | no script or doc found. It matches the full Llama-2-13B at 2,048 tokens (40 × 2,048 × 53,248 ≈ 4.36e9, my arithmetic) and the 2^−215.3 of appendix_short.tex:298 | — | — |
| 136-138 | P1–P5: clamp in [−128,127], widths [2, 2^16], 2^29 | Mo | cut.py:46-47,143 | — | cut.py `CutPlan.from_graph` |
| 157-169 | thm:v1 terms | Mo (theorem) | appendix_short.tex:234-298 | — | logup.py, logup_gkr.py |
| 183-184 | at λ=128, r and column counts unchanged | Mo | tab:params | params_table.py | protocol.py `params_for(cut=True)` |
| 184-185 | GKR prover in Triton | — | — | `EX/8_v1/gen_gkr_triton.py` | gkr_triton.py |

## 6. §6 Generation (P/generation.tex)

| Loc | Value | Kind | Source | Script | Library |
|---|---|---|---|---|---|
| 25-27 | P=G=512, Llama-2-7B: ≈1.3 TB of claims vs 3.3 GB for one prefill | Mo | DI/E1_generation.md:11. Derivation script not identified | — | analytic.py (presumably) |
| 46-50 | GPT-2, 64-token prompt, 256-token response: 0.29 s / 0.59 s / 87.3 MB (C); 0.20 / 0.51 / 82.9 MB (Kpre); 1 token: 0.15 / 0.21 / 14.9 MB | M | DI/E1:80-81 (2080 Ti, Xeon 8 threads, interactive). **Rec✗:** records in SR/raw_sp2080/ (E1:87), outside repo | `bench.py llm --model gpt2 --seq 64 --gen 256 --builds full --modes C:int --policy auto --wire --prune-last --lean --tag _wire` (E1:91-92) | protocol.py; transformer claim columns |
| 50-52 | 1.1 ms per token; zkAgent 0.30 s per token | M / L | 0.29/256; zkAgent (REVIEW2:595 did not verify it) | — | — |
| 53 | tampered query always rejected | M | E1:77 | same | — |
| 64-77 | Gumbel table; a=round(2^16·c/τ); \|z\|<2^29, a<2^24, \|Γ\|<2^20 | Mo (design) | — | — | fullcheck/token_sampler.py `gumbel_table`, `TokenSampler` |
| 89-90 | Γ/2^16 ∈ [−2.47, 11.78]; 14.25 nats; e^−14.25 < 6.5e−7 | Mo | DI/E2b_sampler_tv.md:40,258; REVIEW2:584 | — | — |
| 91-92 | (V−1)·e^−14.25 ≈ 0.02 / 0.03 / 0.1 | Mo | arithmetic | — | — |
| 93-96 | OPT-1.3B, 512 positions: TV median 0.0063 (k=50), 0.020 (k=0); integer vs fp32 0.087 | M | AR/sampler_tv_opt-1.3b.json; E2b:201-226 (job 1008675) | `EX/9_quality/sampler_tv.py` (E2b:334-364) | token_sampler.py |
| 97-98 | under-samples 8–12 nats below the top; a 2^20 table gives 0.0022 | M | E2b:214,225,293 | same | — |
| 55 | Llama-2-7B / Qwen3-4B generation cells | — | `\todo{F6}` | `EX/6_improvements/gen_bench.py` | — |

## 7. §7 Evaluation (P/evaluation.tex)

**Setup and text**

| Loc | Value | Kind | Source | Script |
|---|---|---|---|---|
| 43 | λ=128 | — | — | — |
| 46-52 | platforms: base on L40S + EPYC 9334 8 threads; improvements on 2080 Ti + Xeon 4114 | — | — | — |
| 57-61 | real OPT-1.3B: prover within 9%, CPU verifier within 6%, v0 proof 9–17% smaller; full real OPT-6.7B and OpenLLaMA-7B in Kpre @2,048 accepted, all tampers rejected | M | DI/F3a:280-335 (jobs 1008673, 1009431); DI/F3b:72-79 (job 1015751); Rec✗. **The "smoothed OPT" here is the legacy `--smooth 0.55` graph, not the certified α=0.6 model of tab:quality** (F3a:229-231); `sp2027.sh` now passes `--recipe certified` (sp2027.sh:311-314) | `perf.py --model opt-1.3b --real facebook/opt-1.3b --smooth 0.55 --calib-seq 2048 --seq $T --modes Kpre C --policy auto --fs --queries 3 --lean --wire on --prune-last on --lookups on --threads 8 --device cuda --verifier-device cpu --tampers 3` (F3a:250); `perf.py --real openlm-research/open_llama_7b_v2 …` |
| 63-67 | 3,792 honest accepted; 2,512 attacks rejected (Freivalds 1,601, columns 853, Merkle 56, range 2) | M | R/report/tables/counts_opt.tex (`\OptN*`) from AC/raw_l40s_improved{,2,3} | `count_outcomes.py --platform l40s_improved,l40s_improved2,l40s_improved3 --prefix Opt --definition --tex ../report/tables/counts_opt.tex` (5_comparison/README.md:34-35) |
| 104-105 | image models: prove 3.8–22.9 ms, verify 1.7–64.8 ms; proofs 1.3–9.3× smaller than int8 weights | M | R/report/tables/cnn.tex; report/main.tex:637-640 (course "optimised" protocol = the S&P "base") | `paper_assets.py --platform l40s --optimised l40s_improved,l40s_improved2,l40s_improved3` |
| 106-107 | Llama-2-7B 64-token proof ≈21× smaller than its weights | Mo | 6.74 GB / 325 MB | — |
| 108-109 | OPT-6.7B 5.5 GB / 46.5 s; claims are 98% of the 6.57 GB proof | M | AC/tables_l40s_improved2/measured_summary.csv (`bytes_claims` 6,442,125,383 / `bytes_total` 6,570,112,359) | aggregate.py |
| 110 | GPU verifier 10–15× faster than CPU at 2,048 tokens | Mo | arithmetic on tab:eval-llm | — |
| 111-113 | 2080 Ti, OPT-1.3B @2,048: 2.1 s (Kpre), 4.4 s (C); 4.1× / 8.6× integer forward; 19× / 40× fp16 | M | DI/A7:294-295 (job 1008674), Rec✗ | serve_ab.py |
| 113-114 | 64 prompts × 64 tokens in one Kpre proof: 3.4 s (4.8×, 18×) | M | A7:306 | same |
| 116-117 | improvements change no exchanged byte | M | A/B harness checks; EX/6_improvements/results/codec_bytes_identity.jsonl | open_ab.py, forward_ab.py, encode_prof.py |
| 117-119 | opening was 62% of the base prover; prover 3.7–4.3× faster on Llama-2 blocks | M | DI/A1:20-29 (L40S base, 1.105/1.78 s); A1:92-97 (2080 Ti, 1–4 blocks) | `EX/6_improvements/open_ab.py` |
| 119-120 | GPU decoder + fused attention: GPU verifier 2.0–2.7× at 2,048 tokens | M | DI/B2_B3:84-85 (2080 Ti, partial builds) | **U:** harness not named (env toggles `PVI_GPU_DECODE`, `PVI_FUSED_ATTN`) |

**tab:eval-llm (lines 73-102)**
- **Kind:** M. Every printed base cell (prove, CPU and GPU verify, proof, Kpre verify and proof, for GPT-2, Qwen3-4B, Llama-2-7B/13B at 64 tokens and OPT-125M/1.3B/6.7B, Llama-2-7B/13B at 2,048) comes from AC/tables_l40s_improved{,2,3}/llm_full_model.csv.
- **Regeneration:** the cells are selected by `paper_assets.OPTIMISED`. `sp2027_numbers.py --stored` regenerates them: 63 base and proof cells identical to the paper (DI/F6_final_run_package.md:241-243).
- **"--" cells:** withheld timings, listed in AC/excluded_cells.csv (5_comparison/README.md:123-145).
- **Commands:** `bench.py` (tags in 5_comparison/README.md:76-83), then `aggregate.py --platform l40s_improved{,2,3}`.
- **Plan:** these proofs use the course-era `auto` plans, not cnn18c (SP2027_PLAN2.md:740). The "new" columns are `\todo{L40S}`.
- Params column: `transformer.CONFIGS`.

**tab:eval-v1 (lines 126-157)**
- **Kind:** M. Proof bytes come from DI/C2:124-135 (job 1005287, λ=128, Fiat–Shamir, last block pruned). Times come from DI/C2:183-187 (job 1005575, `lmax` 24).
- **Records:** none in the repo. Logs are in SR/server_out/v1b_1005287.out and v1e5_1005575.out.
- **Plans differ by row and the table does not say so:** GPT-2 and OPT-125M use cnn18c, Llama-2-7B cnn16c, Llama-2-13B cnn17c (C2:124-135).
- Script: `EX/8_v1/v1_bytes.py` (e.g. C2:209-210 gives the flags of job 1007816).
- Library: cut_protocol.py, gkr_triton.py.

**Text after tab:eval-v1**

| Loc | Value | Kind | Source |
|---|---|---|---|
| 159-160 | 2.0–2.3× on every build with more than one block | M | table |
| 161 | "V1's own additions stay below 1 MB per proof" | M (scope-limited) | C2:137-138 covers the partial builds only. On full models the V1 fold vectors are 32.1 MB (OPT-6.7B, F3a:301) and 40.8 MB (OpenLLaMA-7B, F3b:62). **Claim at risk** |
| 162-163 | 1.81× (Llama-2-13B, 1 block, C); GPT-2 @64: 1.28–1.46× | M | table |
| 163-164 | design estimate 1.78–1.85× with real weights | Mo | V1_SPEC (SR/v1_design, outside repo); C2:153 |
| 164-165 | smoothed real OPT-125M @512 tokens: 1.68× (C), 1.77× (Kpre) | M | C2:143-149 (laptop, `--smooth 0.55`), Rec✗ |
| 165-166 | real OPT-6.7B 1.73× | M | F3a:301 |
| 168-170 | prover ≈10× slower; OPT-125M C 0.50→4.85 s, of which 3.58 s GKR | M | C2:185 |
| 170-172 | verifier 1.6–1.9× at 2,048 tokens, 2.2–2.8× at 512 | Mo | arithmetic on the table |
| 172-175 | full Llama-2-7B, Kpre, Fiat–Shamir: V1 45.4 s vs v0 19.6 s (2.3×) | M (V1 extrapolated from 2–3 blocks) | DI/V_V1N_fast_v1_verifier.md:377,388 (job 1011440, Rec✗); ev_inputs_draft.json:38-42; v0 19.64 s in AR/vlonga… — script `EX/8_v1/v1_phases.py`, `v1_bytes.py --v1-paths` |

**tab:options (lines 182-222)**

| Cell | Value | Kind | Source |
|---|---|---|---|
| Re-execution, CPU | 4.1 s / 93 s; non-exact int8 engine 1.9 / 56 s | M | RX:28-34 (job 1006979, Rec✗; also in ev_inputs_draft.json:54-68); reexec.py |
| Re-execution, GPU | 0.15 / 1.65 s (int8, extrapolated from 1–2 blocks); fp16 0.034 / 0.64 s | M | A7:315,319-320 (U1/U2); serve_ab.py |
| "6.7 GB once" | int8 weights | Mo | CONFIGS (6.74e9) |
| CommitLLM | 144 kB per token; 0.66 ms per token (Llama-3.1-8B, Apple M4) | L | commitllm2026. **No repository record**; not in reported_curated.csv; restated only in ext_discussion.tex:116-119 |
| zkLLM | 183 kB, 2.36 s | L | reported_curated.csv |
| Ours, C | 325 MB / 6.6 GB; 1.0 s at 64 tokens | M | tab:eval-llm base (EPYC) |
| Ours, Kpre | 196 MB / 6.4 GB; 1.0 s / 18.5 s | M | bytes from tab:eval-llm base (L40S); times 1.01 s (job 1006979, Rec✗) and 18.52 s (AR/vlonga…, job 1007816) on the Xeon |
| with V1 | 2.8 GB / 45.4 s | Mo bytes + M time | footnote d: 6.43 GB ÷ 2.3 |
| footnote f | GPU TEE < 7%; CommitLLM 12–14%; ours 9.8–10.6× (Kpre) and 19–154× (C) | L, L, M | A7:315-320 (Llama-2-7B, extrapolated) |

**Fig. fig:breakeven (lines 224-233)**
- **Kind:** Mo + M. The panels combine measured claim bytes and verifier and re-execution times with the cnn18c Fiat–Shamir byte model.
- **Command:** `ev_breakeven.py --inputs experiments/5_comparison/ev_inputs_draft.json --out artifacts/results/ev_breakeven_draft --figure …` (EV:58-62).
- Library: analytic.py `proof_bytes`, plans.py, `params_for`.
- **Files:** paper_sp2027/figures/ev_breakeven.pdf is byte-identical to AR/ev_breakeven_draft.pdf.
- The 6.74 GB and 93.2 s come from ev_inputs_draft.json.

**Text after tab:options**

| Loc | Value | Kind | Source |
|---|---|---|---|
| 238-244 | Kpre: 18.5 s (19.6 s Fiat–Shamir) vs 93.2 s fp32; integer forward 142 s; 5.0× (4.7×). 64 tokens: 1.0 s (1.4 s) vs 4.1 s; 4.0× (2.9×) | M | RX:28-34, 52-55; FS1:109-111 |
| 248 | L > 8B/(t_r − t_v) | Mo | ev_breakeven.py `breakeven_link_gbps` |
| 253-254 | OPT-1.3B Kpre GPU verifier ≈1.1 s on the 2080 Ti vs 0.11 s fp16 | M (extrapolated from 12/24 blocks) | B2_B3:84,89; A7:294 (0.108 s) |

**tab:eval-compare (lines 262-299)**
- **"theirs" columns:** L. Every value I checked matches AC/tables/reported_curated.csv: zkCNN, ZKML, zkGPT, DeepProve 64 and 512, ZKTorch, Maverick, and the four zkLLM rows. Produced by `literature.py` from AC/literature/reported_benchmarks.csv.
- **"ours" columns:** M, from tables_l40s_improved{,2,3}; the Maverick row uses tables_l40s_improved2_thr1. `sp2027_numbers.py --stored` reproduces 30 cells identically (F6:242).
- **Factors:** Mo (arithmetic).
- **Footnote b**, "its shapes imply 32 tokens": our inference, no record (**U**).

| Loc | Value | Kind | Source |
|---|---|---|---|
| 307-310 | prover faster by 46× to ≈32,000×, and 3.0× vs Maverick; verifier slower in 2 rows; proof larger by up to 56,000× | Mo | arithmetic on the table |
| 310-311 | V1 narrows the gap from ≈14,000× to ≈6,500× (OPT-1.3B) | Mo | 2,073.9 MB / 147 kB and 960.8 MB / 147 kB |
| 311-312 | `\todo` 844 s (zkLLM on our L40S) | M | AR/zkllm_l40s/summary.csv (843.84 s); summarise.py |

## 8. Quality (P/quality.tex)

| Loc | Value | Kind | Source | Script | Library |
|---|---|---|---|---|---|
| 23-26 | full test split: 140 windows, 286,580 tokens; calibration on 4 train windows of 2,048; fp32 27.66 / 14.62 / 10.86 match published values | M / L | AR/9_quality/ppl2048_opt-*.jsonl, `fp32` cells (27.656, 14.624, 10.860); DI/F2a:119-126 | `EX/9_quality/ppl_2048.py` (F2a:206-213) | common.py `certified_calib` (:42); real_weights.py |
| tab:quality, scalar column | ×1.166 / ×1.096 / ×1.696 | M | ppl2048_opt-*.jsonl `scalar-p11-int` draw 0 (1.1660, 1.0959, 1.6957) — stored | ppl_2048.py | — |
| tab:quality, certified column | 29.75 (×1.076), 15.01 (×1.026), 10.94 (×1.008); top-1 0.82 / 0.90 / 0.93; CIs [1.073,1.079], [1.024,1.028], [1.007,1.008] | M | F2a:189-197 (job 1011066). **Rec✗:** `ppl2048_search2_*.jsonl` is not in AR; the CIs appear only in drafts/h5_early_review_20261012.md:19. The 6.7B value is draw 0 only | ppl_2048.py (round-2 flags not recorded) | `CERTIFIED_RECIPE` |
| 48-49 | 17%, 10%, 70% above fp32 | M | scalar column | — | — |
| 49-51 | OPT-125M @128 tokens: float norm outputs remove more than half of the gap | M | DI/F2_quality.md:17-37 (75.5 → 69.4, fp32 64.8). **The script `attribution.py` is not in this worktree.** It exists in `<scratchpad>/wt_f2/code/experiments/9_quality/attribution.py`, with records in `wt_f2/code/artifacts/results/9_quality/attribution_*.json` | — | — |
| 54-56 | 8-bit probabilities cost 2–23%; a row keeps as little as 56% of its mass; 11 bits keep 97.5% | M | F2a:148-150 (jobs 1007844-6); p8 vs p11 cells stored; 56% and 97.5% are in the doc only | ppl_2048.py `--pbits 8 11` | — |
| 65-66 | percentile and clipping recipe: ×1.14–1.25 (OPT-1.3B) | M | ppl2048_opt-1.3b.jsonl `smooth-p11-int` α=0.55, draws 0–2 (1.144 / 1.254 / 1.235) — stored | — | — |
| 66-67 | channel maxima stable to ±0.003; α=0.6 | M | F2a:180,194 (round 1 stored, round 2 Rec✗) | — | — |
| 68-72 | zero-shot: −1.0 / −0.8 points, +0.06 for 6.7B; LAMBADA −3.1 / −2.0 / −0.5; 6.7B multiple-choice differences inside their 95% CIs | M | DI/F2c_zero_shot.md:155-198 (jobs 1012440, 1012655), Rec✗. **Not stated in the paper:** OPT-1.3B's HellaSwag is a 2,000-document subset (F2c:170) | `EX/9_quality/zeroshot.py` | — |
| 74-77 | SmoothQuant-O3 1.6% / 0.8%; ZeroQuant-V2 ≈1% / 4%; zkLLM and zkGPT < 1% | L | no repository record (SR/f2_quality/literature.json, outside repo) | — | — |
| 77-79 | targets ×1.06 / 1.04 / 1.05; OPT-125M misses by 0.016 | M | F2a:202 says 0.015 (rounding mismatch) | — | — |
| 85-88 | OpenLLaMA-7B ×1.073, TinyLlama ×1.12, Qwen3-1.7B ×1.40, Qwen3-4B ×5.3 (main algorithm) | M | DI/F2b_Q_real_llama.md:1171-1173, 1258-1259 (jobs 1013787, 1013574, 1013332), Rec✗ | `EX/9_quality/ppl_llama.py` | real_llama.py `build_llama_from_hf`, `MAIN_RECIPE` (:458), `prefixquant_tokens` |

## 9. §8 Discussion and Conclusion (P/discussion.tex)

| Loc | Value | Kind | Source |
|---|---|---|---|
| 28-30 | below ≈22 queries of 64 tokens; 1 query at 2,048 tokens, 2.3 with V1 | Mo | AR/ev_breakeven_draft.json panel1 (cnn18c, Fiat–Shamir: Q* 22.25, 1.03, 2.32); ev_breakeven.py |
| 31-32 | break-even links 0.59–0.70 Gb/s; 0.47 with V1 | Mo | ev_breakeven_draft.json panel2 |
| 35-36 | proof reaches the weights' size at ≈2,100 tokens; ≈4,500 with V1 "by the byte model" | Mo / **U** | 2,100 per EV:78 (arithmetic). For 4,500 no script or record exists (it is consistent with the C-mode 2.13× ratio). **ext_discussion.tex:152 and ext_v1.tex:201 still say 2,050** |
| 38-39 | ⌈k/T⌉ queries reveal a weight matrix | Mo | argument |
| 43-45 | up to 16,384 tokens: CPU verifier ≈2× faster than re-execution (4.7× at 2,048); proof ≈51 GB | M (extrapolated from 2–3 blocks; the 16k 3-block cell ran out of memory) | DI/V_LONG_b_long_prompts.md:21-24 (job 1011426), Rec✗; perf.py, reexec.py (V_LONG_b:9-14) |
| 47-49 | Llama ×1.073; Qwen3-4B ×5.3 | M | F2b |
| 41 | setup proof untimed | — | `\todo{F6}` |

## 10. Submission appendix (appendix_short.tex, app_params.tex)

**tab:params (app_params.tex:16-57)**
- **Kind:** Mo. The caption says "All values: params_table.py".
- **Values:** 226 operations; β 136.8 / 200.8; r 5 / 7; p^−r = 2^−154.5 / 2^−216.3; t 23 / 34; columns 65.6 / 96.5 MB; \|𝕂\| ≈ 2^247.3; lookup term ≤ 2^−215.9; GKR term ≤ 2^−230.1; \|J₀\| 130; β₀ 200.0; s 8; f 2^−2 / 2^−4; t⁰ 1,054–1,190 / 4,399–4,936; 113.5 / 79.4 MB; 2.4 / 9.8 GB.
- **Footnote b:** N_leaf 2.7e9 in 162 instances, \|V_tab\| ≤ 5.6e5.
- **Provenance notes:**
  - The V1 cells are hard-coded defaults in params_table.py:78 (job 1007816 counts).
  - The script defaults to `--policy auto`, but the table is under cnn18c.
  - I did not re-run it.

**Other numbers**
- appendix_short.tex:128 (Merkle leak, 2^(η+1)/N): Mo; checked by SR/attack_theory/check_merkle_leak.py, outside repo.
- appendix_short.tex:159-173 (integer semantics constants): design.
- appendix_short.tex:282 (939,458,561): arithmetic, verified.
- appendix_short.tex:297-298 (2^−245.7 and 2^−215.3; the latter matches 4.3e9 leaves): Mo.

## 11. Figures

- **Fig. fig:overview** (intro.tex:51-60): schematic. Written by `paper_assets.py:894,908`; paper_sp2027/figures/overview_*.pdf is byte-identical to report/figures.
- **Fig. fig:breakeven:** see §7.

## 12. Cross-cutting provenance problems

1. **Body numbers whose records are not in the repo** (they exist only in docs, job logs on the server, or outside the worktree):
   - E1 generation (SR/raw_sp2080)
   - all of tab:eval-v1 (SR/server_out)
   - A7 serving (server clone s3)
   - RX re-execution and the 64-token verifier times (job 1006979)
   - F2a round 2, i.e. the whole certified column of tab:quality and its CIs
   - F2c zero-shot
   - F2b Llama/Qwen ratios
   - G3-min OPT-1.3B (`adaptive_game_opt-1.3b.json`)
   - F3a/F3b real-weight runs
   - V-LONG-b
   - V-V1N's 45.4 s
2. **Scripts outside the worktree:**
   - `attribution.py` (quality.tex:49-51), in wt_f2
   - `sok_formulas.py`, `check_guessing.py`, `check_merkle_leak.py` (tab:sok and theorem checks), in SR/attack_theory
   - V1_SPEC's estimate, in SR/v1_design
3. **Recipe mismatch.** The real-weight cost and V1 numbers (evaluation.tex:57-61,164-166; abstract 1.7×) used `--smooth 0.55` (legacy). tab:quality certifies α=0.6. D6a's cross-OS claim (defence.tex:53) is legacy-only.
4. **Plan mismatch.**
   - tab:params, the setup-proof sentences and Fig. breakeven use cnn18c.
   - tab:eval-llm, tab:options' mode-C row (325 MB) and tab:eval-compare use the course-era auto plans; tab:eval-v1 mixes cnn16c, cnn17c and cnn18c.
   - Stale docs: DI/D4_D5 §5 and C2 still show the older plan's setup-proof bytes.
5. **Wording at risk.**
   - "V1's own additions stay below 1 MB" (evaluation.tex:161) fails on full models.
   - "values outside the natural ranges" (intro.tex:65-66): 12.5% of the untargeted values were inside.
   - "every CNN and test query" (attack.tex:207) means n=94 queries.
   - "misses by 0.016" vs the doc's 0.015.
   - "2,100" in the body vs "2,050" in the extended version.
6. **Course-paper references.** The experiment READMEs (2_attack, 1_reproduction, 3_sampling_fixes, 5_comparison) still cite course-paper section numbers ("Sec. 4.2") and course commands (`text_numbers.py` checks report/main.tex). 7_reexec, 8_v1 and 9_quality have no README.
7. **Hard-coded parameters.**
   - `analytic.py` (0 arguments; model, seq and λ fixed at line 24) and `literature.py` (0) take no CLI options.
   - `aggregate.py`, `3_sampling_fixes/run.py` and `floor_sampler.py` take 1 option each.
   - `params_table.py` hard-codes the V1 cell counts.

## 13. Scripts the body depends on (all under EX/ unless noted)

**Attack (§3)**
- `0_train_models/train.py`: trained MLP/CNNs in artifacts/models.
- `1_reproduction/run.py`
- `2_attack/run.py`, `2_attack/real_llm.py` (+ `real_llm.sbatch`), `2_attack/inrange_llm.py`
- `3_sampling_fixes/run.py`, `floor_sampler.py`, `adaptive_game.py`, `adaptive_game_llm.py` (+ sbatch)

**Benchmark and tables**
- `4_defence_benchmark/bench.py` (+ slurm/strong_gpu.sh, sp2027.sh for the final run)
- `5_comparison/aggregate.py`, `analytic.py`, `literature.py`, `count_outcomes.py`, `paper_assets.py`
- `5_comparison/ev_breakeven.py` (+ ev_inputs_draft.json)
- `5_comparison/sp2027_numbers.py`: final-run filler and `--stored` check

**Improvements, V1, re-execution**
- `6_improvements/perf.py`, `open_ab.py`, `params_table.py`, `setup_bytes.py`, `gen_bench.py` and `setup_timing.py` (final run only)
- `7_reexec/reexec.py`, `serve_ab.py`
- `8_v1/v1_bytes.py`, `v1_phases.py`, `gen_gkr_triton.py`

**Quality**
- `9_quality/ppl_2048.py`, `zeroshot.py`, `ppl_llama.py`, `sampler_tv.py`, `digest_check.py`, `common.py`
- `artifacts/results/zkllm_l40s/summarise.py`: only for a `\todo`

**Outside the worktree:** `attribution.py` (wt_f2); `sok_formulas.py`, `check_guessing.py`, `check_merkle_leak.py` (SR).

**Main library:**
- `pvi.protocol` (RandPathTest), `pvi.attacks.{tamper,backdoor,baselines}`, `pvi.experiments.analysis`, `pvi.defences.{sampling,adaptive}`
- `pvi.fullcheck.{protocol, commitment, plans, field, transformer, graph, claimcodec, native_kernels, attention_kernels, codec_kernels, setup_proof, setup_cache, analytic, cut, cut_protocol, logup, logup_gkr, gkr_triton, extfield, token_sampler, real_weights, real_llama}`

## 14. Claims with UNKNOWN provenance

1. attack.tex:286 (tab:sok): TOPLOC/SVIP "forged tokens with recomputed fingerprints pass (100%; ≥95%)". No repository record; probably SR/attack_theory/sok.md, not verified.
2. v1.tex:91: "up to 4.3·10^9 leaves". No script or doc; consistent with the full Llama-2-13B at 2,048 tokens by my arithmetic.
3. discussion.tex:35-36: "about 4,500 tokens with V1, by the byte model". No script or record.
4. evaluation.tex:119-120 (2.0–2.7× GPU verifier): the numbers are in B2_B3:84-85, but the harness and command are not recorded.
5. evaluation.tex:295 (tab:eval-compare note b): zkGPT "shapes imply 32 tokens", our inference with no record.
6. evaluation.tex:201 (tab:options): CommitLLM 144 kB per token and 0.66 ms per token. Cited to a preprint, but there is no curated literature record and no verification note.
7. attack.tex:214: Wilson 91.2–100%. The formula is standard, but no script computes it.
8. generation.tex:25-27: 1.3 TB vs 3.3 GB. A byte-model estimate in E1:11 with no script named.

**Not determined:**
- I did not re-execute `params_table.py` or `setup_bytes.py`, so tab:params and the 2.4/9.8 GB, 65.6→113.5/79.4 MB and 2.2%/0.5% values are checked only against SP2027_PLAN2.md:740-741.
- I did not check each base cell of tab:eval-llm or tab:eval-compare by hand. I relied on F6's report that `sp2027_numbers.py --stored` matches 63 and 30 cells, plus a few spot checks (Llama-2-7B 6.57 GB, Llama-2-13B 10.5 GB).
- I did not check the published values of SmoothQuant, ZeroQuant-V2, zkAgent and NanoZK.
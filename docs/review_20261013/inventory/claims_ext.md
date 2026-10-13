# Extended version (`paper_sp2027/main_ext.tex`): claims, where they come from, and code names

**Paths.** `ROOT` = `C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-amina-project/4aa2c028-956d-4ff2-9cca-56d76e547439/scratchpad/wt_sp2027` (branch `sp2027`). Every path below is relative to `ROOT` unless it carries one of these marks:
- **[S]** is under `ROOT/..`, the session scratchpad. It is outside the repo.
- **[A]** is a member of `C:/Users/User/Desktop/amina-project/server_archive/sp2027_clones_s3-s12_20261012.tar.gz`, the laptop archive of the server clones. It is outside the repo.

**Classes.**
- **MEASURED**: a run or a test produced the value.
- **MODELLED**: the value comes from a formula, a byte model or a code constant. I recomputed formula values where I say "✓".
- **LITERATURE**: the value is published by others.
- **UNKNOWN**: no source found.

"Doc" means a `docs/improvements/*.md` diary entry, which usually names the SLURM job. Raw job logs are generally not in the repo.

## 0. What `main_ext.tex` adds

- `main_ext.tex:1-3` defines `\pviextended` and inputs `main.tex`. `main.tex:88-104` then inputs these files in order: `appendix_attack`, `appendix_defence`, `app_params`, `appendix_v1`, `ext_proofs`, and then the section "Material Moved Out of the Body" (`main.tex:94`) with `ext_intro`, `ext_attack`, `ext_defence`, `ext_v1`, `ext_generation`, `ext_evaluation`, `ext_discussion`. The title gets "(Extended Version)" at `main.tex:64`.
- **Only one figure is added:** `figures/protocol.pdf` (`ext_defence.tex:183`). It is byte-identical to `report/figures/protocol.pdf`, which `code/experiments/5_comparison/paper_assets.py:911-954` (`fig_protocol`) builds into `report/figures/`. No table files are input; all tables are inline.
- **Cross-references are clean.** Every `\ref` in the 12 files resolves in the extended build, and no label is defined twice. `sec:quality` comes from `quality.tex`, which `evaluation.tex:316` inputs.
- `generation_quality.tex` is orphaned: nothing inputs it, but it defines a second `sec:quality` at line 168.
- **Theory-only material needs no measurement provenance:**
  - `appendix_attack.tex:24-560`: the statements and proofs of thm:guess-full, prop:worst, prop:blind, prop:amp, prop:block, lem:worst-int, cor:block and the remarks.
  - `appendix_defence.tex:41-403`.
  - `appendix_v1.tex:37-302`.
  - `ext_proofs.tex:31-990`.

## 1. Claims inventory

### 1.1 `appendix_attack.tex`
| Line | Claim / value | Class | Provenance | Script + command | Library |
|---|---|---|---|---|---|
| 146 | Zeroed neurons are 0 on 65% of inputs | MEASURED | `code/artifacts/results/defence.json` → `zero_blind_spot.natural_zero_fraction_at_chosen_neurons` (`code/experiments/3_sampling_fixes/README.md:20`) | `cd code && python experiments/3_sampling_fixes/run.py` | `pvi.defences` samplers, `pvi.attacks.tamper`, `pvi.experiments.analysis.acceptance_probability` |
| 148-150, 537-538 | 12 random independent-row instances (N=4, q∈{1,2}) match cor:surprise and G_q; copy correlation gives no gap in 12 | MEASURED (exact enumeration) | [S] `sp2027_research/attack_theory/check_guessing_out.json` (named in the header at `:8`). **Not in repo** | [S] `attack_theory/check_guessing.py` | none (standalone) |
| 362, 369-371 | N<2^29/129≈4.16e6; c<2^15; 128(1+128c)<2^29 | MODELLED ✓ | arithmetic | n/a | n/a |
| 459-470 | Mixing beats opening only for d above ~20 (r=5) / ~28 (r=7) and rows longer than ~280 | MODELLED | arithmetic; the text says "not measured" (`:470`) | n/a | n/a |
| 519-520 | N=16: detection 0.125 / 0.25 / 0.5 against 0.0625 | MEASURED (exact) | [S] `attack_theory/check_merkle_leak_out.json` (`:9`). **Not in repo** | [S] `attack_theory/check_merkle_leak.py` | none |
| 568 | "We measured k=1 for next-token changes on OPT-6.7B" | MEASURED | `code/artifacts/results/real_llm_attack.json`, `real_llm_attack_auto.json` (`code/experiments/2_attack/REAL_LLM.md:17-37`) | `sbatch code/experiments/2_attack/real_llm.sbatch` (`REAL_LLM.md:94`), once more with `--target auto` | `pvi.fullcheck.sampling.paths_for` (`sampling.py:392`), `transformer.CONFIGS`; HF transformers |
| 573-591 | tab:aligned, all 10 cells | MODELLED ✓ (I recomputed every cell) | formula ⌈40 ln2/−ln(1−k/d_ff)⌉; d_ff 9,728 / 18,944 are LITERATURE (model configs). The header (`:12-13`) says "recomputed with Python". **No script in repo** | none | none |

### 1.2 `appendix_defence.tex`
| Line | Claim / value | Class | Provenance | Script | Library |
|---|---|---|---|---|---|
| 74 (also `ext_proofs.tex:136`) | P_max is 255 in the cost benchmarks and 2,047 in the certified models | MODELLED (code constants) | `transformer.py:55` `PROB_MAX=255`; `:59` `CERTIFIED_PROB_MAX=2047`; `real_weights.py:185` default `prob_max=CERTIFIED_PROB_MAX` | n/a | `build_decoder(prob_max=)` (`transformer.py:496-497`) |
| 78-86 | int8 / float32 exactness bounds; fallback when the self-check fails | MODELLED + code behaviour | `field.int8_ok` (`field.py:356`); `docs/improvements/A1_int8_column_opening.md`, `A2_int8_forward.md` | n/a | `graph.MatOp` (`graph.py:189,296`) |
| 175 | "about 2^-35 per query for the smallest tree of Llama-2-7B (t=35)" | MODELLED, **stale** | `D4_D5_untrusted_commitment.md:19` (auto plan). Under the paper's cnn18c plan the smallest tree has t=23 / 34 (`app_params.tex:26-27`) | `setup_bytes.py` | `plans.exact_columns` |
| 176-177 (also `ext_defence.tex:107-108`) | Per-query testing would need 3.2–6.6× the column bytes | MODELLED, provenance **outside repo** and stale-risk | [S] `sp2027_research/idea_trust-rigor.md:31`, computed under the old plan ("t_g … 35–133"). Not regenerated for cnn18c | none in repo | n/a |
| 205-208 | n/p^5 ≤ 2^-136.5 for n ≤ 2^18; s ≥ 7 under Fiat–Shamir | MODELLED ✓ | arithmetic | n/a | n/a |
| 246-253 | Parameter rule (β0, s, radius fractions 2^0..2^-10, t0, t) | code | `setup_proof.setup_params` (`setup_proof.py:104`) | n/a | `setup_proof` |
| 386-391 | For the four models of tab:setup at a=100 and a=1,000, the rule picks radii of at most a quarter of the unique-decoding radius | MODELLED, **unverified for cnn18c** | Written in the H5 pass (11 Oct), before the cnn18c regeneration (commit `2880932`, 12 Oct). `app_params.tex:44-45` shows f=2^-2 / 2^-4 for Llama-2-7B only. No stored output | `setup_bytes.py --policy cnn18c` | `setup_proof.setup_params` |

### 1.3 `app_params.tex` — tab:params (`:9-58`)

- **Class:** every cell is MODELLED.
- **Source:** `code/experiments/6_improvements/params_table.py` (named at `:57`). The table was regenerated for cnn18c in commit `2880932`; the values are quoted in that commit's change to `SP2027_PLAN2.md`. No output is stored.
- **Reproducing command:** `cd code && PYTHONPATH=src python experiments/6_improvements/params_table.py --model llama2-7b --policy cnn18c --amortise 100 1000 --tex`. The script's default is `--policy auto` (`params_table.py:64`), so the command in its docstring (`:3`) **does not reproduce the paper's table**.
- **V1 rows (`:37-38`, footnote b `:55`)** extrapolate from 2- and 3-block cell counts hard-coded at `params_table.py:90`: `{2: (103809024, 6, 41247), 3: (190840832, 11, 58336)}`. No record says where these constants were measured. I re-derived the footnote values: N_leaf 2.71e9, 162 instances, |T| 5.54e5 ✓.
- **Values I recomputed:** log2 p 30.907; β 136.8 / 200.8; r 5 / 7; p^-r 2^-154.5 / 2^-216.3; |K| 2^247.3; β0 200.0; s=8 ✓.
- **Library:** `plans.column_bits` (`plans.py:127`), `plans.exact_columns`, `plans.plan_commitment`, `protocol.params_for` (`protocol.py:132`), `setup_proof.setup_params`, `analytic.decoder_shapes`.

### 1.4 `appendix_v1.tex`

- **`:161`** p−2^30−2^16 = 939,458,561. MODELLED ✓.
- **`:285-288`** Two per-round bounds, both MODELLED:
  - 3/p^8 = 2^-245.7 ✓.
  - The lookup term ≤ 2^-215.3 is the **Llama-2-13B** value from [S] `sp2027_research/v1_design/V1_SPEC.md:347` (N_leaf 4.27e9). The text does not name the model, and tab:params gives 2^-215.9 for Llama-2-7B.

### 1.5 `ext_proofs.tex`
| Line | Claim / value | Class | Provenance |
|---|---|---|---|
| 284-291 | Γ[0]=−161,658 and Γ[2^16−1]=772,243; the table is fixed by its SHA-256 digest | MODELLED | `token_sampler.gumbel_table` (`token_sampler.py:40`) |
| 390-392 | A 6.6 GB claims message gives ~1,600 leaves and ~50 kB absorbed | MODELLED ✓ (1,574 leaves × 32 B) | 6.6 GB comes from body tab:eval-llm (L40S tables) |
| 396 | Labels shorter than 25,448 bytes (the bytes "ch") | MODELLED ✓ | arithmetic |
| 529-531 | test_fs_tree checks flipped bits, lengths, labels and the threaded result against a serial one | MEASURED (tests) | `code/tests/test_fs_tree.py:32-78` |
| 609-611 | test_claimcodec checks random corruptions and that the GPU decoder rejects what the host rejects | MEASURED (tests) | `code/tests/test_claimcodec.py` |
| 724-731 | β 200.8→193, r=7, t_j −4%, 2^-136.8, 2^-128 holds for Q ≤ 2^63 | MODELLED ✓ | arithmetic (`params_for`) |
| 911-913 | 226·p^-5 = 2^-146.7, Q_V ≤ 2^18.7 ≈ 4e5; at r=7, 2^-208.5 and 2^80.5 | MODELLED ✓ | arithmetic |
| 936-938 | Llama-2-7B LM head has 32,000 rows; 7·log2 p ≈ 216 | LITERATURE / MODELLED ✓ | Llama config |
| 991-993 | 27.7N; 305,193 and 454,248 paths | MODELLED ✓ | formula (`paths_for`) |
| 998-1022 | tab:paths | MODELLED ✓ (all 25 cells) | k=2, 6, 60 are MEASURED on OPT-1.3B: `code/artifacts/results/inrange_llm_opt-1.3b_{any,down}.json` (`G3_inrange_attack.md:84-86`; comment `:1004-1006`) |
| 1024-1044 | tab:paths-acc | MODELLED ✓ (all 30 cells) | formula |
| 108, 214-215 | Pruned and unpruned graphs have the same constants; the pruned q commits on its own | MEASURED (tests) | `code/tests/test_pruning.py:30-103` |
| 226-228 | The real builders' pruned logits are tested equal to the unpruned ones | MEASURED (tests) | `test_real_weights.py` / `test_real_llama.py` (I did not open them) |

### 1.6 `ext_intro.tex`

**Costs paragraph** (`:44-72`). Every value duplicates a number traced elsewhere in this report:

| Line | Value | Source |
|---|---|---|
| 46-47 | 3.7–4.3× prover | A1 (`A1_int8_column_opening.md:94-97`) |
| 47-48 | 0.29 s for a 256-token GPT-2 response | E1 (`E1_generation.md:80`) |
| 48-52 | 46–32,000× | body tab:eval-compare: `code/artifacts/comparison/tables_l40s_improved*` + `literature/reported_benchmarks.csv`, with `\todo` |
| 53-54 | 325 MB / 6.6 GB / 183 kB | L40S tables, LITERATURE |
| 55-58 | V1 2.0–2.3×, 2.07→0.96 GB, 1.68–1.77×, 1.73× | `C2_v1_proof_size.md:121-151`; `F3a_real_weight_runs.md:301` |
| 58-60 | 0.50→4.85 s; 45.4 vs 19.6 s | C2 (`:185`); `V_V1N_fast_v1_verifier.md:376-388` |
| 61-64 | 1.0 vs 4.1 s; 18.5 vs 93.2 s; 1.4 / 19.6 s | `RX_reexecution_baselines.md:31-55`; `FS1_fiat_shamir_speed.md:110`; `code/artifacts/results/vlonga_v1retime_w1c_1007816.jsonl` (in repo) |
| 64-65 | 2.7× GPU verifier | `B2_B3_gpu_verifier.md:84` |
| 67-72 | ×1.076 / 1.026 / 1.008 (certified recipe) | `F2a_quality_2048.md:189-201` (job 1011066); records [A] `s3/code/artifacts/results/9_quality/ppl2048_search2_*.jsonl`. **Not in repo** |
| 67-72 | 1.17 / 1.10 / 1.70 with one scalar gain | `F2a:105,136-137` (records `code/artifacts/results/9_quality/ppl2048_opt-*.jsonl`, in repo) |
| 67-72 | Llama ×1.073 | `F2b_Q_real_llama.md:1171-1173,1258` |

**Related work** (`:77-171`): LITERATURE (620 s, 183 kB, 44 min, 34 s, 98.14%, 10 of 80, 7%, 12–14%), with one exception:
- `:152-153` says "1.6% of [the setup proof] for Llama-2-7B at a=100". This is MODELLED from the **auto-plan** breakdown (`D4_D5:99`) and is **stale** under cnn18c (see §3).

### 1.7 `ext_attack.tex`
| Line | Claim | Class | Provenance | Script + command |
|---|---|---|---|---|
| 85-86 | The held-out-mean locator ranks the edited neuron first for 75% and 95% of queries | MEASURED | `code/artifacts/results/adaptive_game_mlp_mnist.json` (G1e; `G3_min_adaptive_game.md:160`) | `PYTHONPATH=src OMP_NUM_THREADS=2 python experiments/3_sampling_fixes/adaptive_game.py --queries 40 --layers 1 2 --calib train` (`G3_min:355`) |
| 86-88 | The contribution-weighted sampler detects 9.2% | MEASURED | `defence.json` `sweep.local-contribution.detection_evasive` = 0.0923 | `experiments/3_sampling_fixes/run.py` |
| 117-124 | 3,000 runs; 99.8–100%; 150 queries; 21–37% without tolerance; 778 nodes | MEASURED | `code/artifacts/results/reproduction.json` (`code/README.md:224`) | `python experiments/1_reproduction/run.py` |
| 127-128 | 250 paths pass 61%; about five neurons, 99% | MEASURED | `code/artifacts/results/attack.json` | `python experiments/2_attack/run.py` (250 is the hard-coded `OPENING_BUDGET_PATHS`, `run.py:56`) |
| 128-133 | 5,000 training images, 40 test queries, 5.1 neurons, 99.0%, 5% | MEASURED | `adaptive_game_mlp_mnist.json`; `G3_min_adaptive_game.md:137,225` | `adaptive_game.py` |
| 135-142 | 0.11% / 0.016% / 0.20%; 14 of 512 neurons; 65%; floor 10×; 0.9–6×; 2% | MEASURED | `defence.json`, `floor_sampler.json` (`code/experiments/3_sampling_fixes/README.md:19-28`) | `run.py`, `floor_sampler.py` |
| 144-146 | 1/28 million; 305,193 paths | MODELLED | `code/experiments/2_attack/README.md:34`; `code/README.md:227` | `python experiments/5_comparison/analytic.py`; `text_numbers.py` |
| 148-151 | Layers 8/16/24/31; 6,187 of 50,272 tokens; " back" succeeds, " hacked" fails | MEASURED | `real_llm_attack*.json` (`REAL_LLM.md:17-27`) | `real_llm.sbatch` |
| 163-165, 199-226 | tab:sok-derivation; 14,182 / 305,193 / 454,248; 88.9% | MODELLED ✓ (I recomputed all) | [S] `sp2027_research/attack_theory/sok_formulas.py` / `sok_formulas_out.json` (header `:15-18`). **Not in repo** | none in repo |
| 271-317 | tab:inrange: OPT-1.3B and OPT-6.7B, 24 rows | MEASURED | `code/artifacts/results/inrange_llm_opt-1.3b_{any,down}.json` (laptop); `inrange_llm_opt-6.7b_{any,down}.json` (job 1007804, Xeon s-006). Tables are in `G3_inrange_attack.md:39-84` | `cd code && PYTHONPATH=src python experiments/2_attack/inrange_llm.py --model facebook/opt-1.3b --layers 6 12 18 23 [--moves down]`; OPT-6.7B: `--model facebook/opt-6.7b --layers 8 16 24 31` (`G3_inrange:96-98`). Standalone (imports `real_llm.PROMPTS`); no `pvi` import |
| 321-323 | 19,000–227,000 and 12,600–454,000 paths | MODELLED ✓ | derived from tab:inrange | n/a |
| 326-343 | Sampler game on OPT-1.3B: 3.1% / 23% / 38% / 48% …, 0.34%, 11–40 prompts | MEASURED | `G3_min_adaptive_game.md:282-334` (job 1008671). The record `adaptive_game_opt-1.3b.json` is **not in repo**; it is only in [A] `s3/code/artifacts/results/adaptive_game_opt-1.3b.json` | `adaptive_game_llm.py ... --caps 90 95 99 100 --threads 8 --device cuda --out artifacts/results/adaptive_game_opt-1.3b.json` (`G3_min:269-271`), via `adaptive_game_llm.sbatch` |

### 1.8 `ext_defence.tex`
| Line | Claim | Class | Provenance |
|---|---|---|---|
| 66-67 | LeNet-5 shares one tree of 2^18 points and opens 15 columns instead of 66 in each of five trees | MEASURED (course run) | `code/artifacts/comparison/tables_l40s_improved2` and `tables_l40s` `measured_summary.csv` (`cfg_group_columns`, `cfg_columns`; `code/README.md:211`) |
| 73-76 | n = max(2^18, 2·next_pow2(k)); six trees for Llama-2-7B | MODELLED | cnn18c policy (`plans.py:58-60`, `AUTO_CANDIDATES` `:121`); `app_params.tex:55` |
| 103-104 | t=35 → 2^-35 | **stale** | see §1.2 |
| 105-106 | Over 300 queries with a small t, the acceptance rate matched eq:split | MEASURED (test) | `tests/test_setup_proof.py::test_without_a_setup_proof_the_split_commitment_passes_about_two_to_the_minus_t_of_the_queries` (`D4_D5:20-22`) |
| 107-108 | 3.2–6.6× | outside repo / stale-risk | see §1.2 |
| 121-122 | "At a=1,000, Llama-2-7B pays 11.3 GB once and 8% more column bytes per query"; <1% at 2,048 tokens | MODELLED, **stale** | auto plan (`D4_D5:87`: 11.3 GB, ×1.08). tab:setup two lines later says 9.8 GB and 65.6→79.4 MB (+21%). The "<1%" still holds |
| 126-145 | tab:setup | MODELLED | `setup_bytes.py`, regenerated for cnn18c (commit `2880932`). Command: `cd code && PYTHONPATH=src python experiments/6_improvements/setup_bytes.py --model gpt2 llama2-7b opt-6.7b llama2-13b --policy cnn18c --amortise 100 1000`. Defaults are `--policy auto` and `--amortise 1000` (`setup_bytes.py:27-29`). No stored output. Library: `setup_proof.setup_params`, `plans.plan_commitment`, `plans.exact_columns`, `analytic.decoder_shapes` |
| 150-166 | Block 2: 2,833.5 MB; 2,784.0 / 45.2 / 4.3 MB; 11,272.7 MB; 1.7% / 0.6%; 3.9% / 2.3%; 1,788.7 / 7,284.7 MB; 63.1 / 64.6%; 0.9–1.7%; 3.7–6.9%; "t0_j reaches 12% of n_j" | MODELLED, **stale (auto plan)** | `D4_D5:85-100` (comment `:147-149`). It contradicts tab:setup directly above it (2.4 / 9.8 GB). Under cnn18c, t0 ≤ 4,936 at n ≥ 2^18, which is ≤ 1.9% of n, not 12% |
| 172 | The GPT-2 setup proof (213 MB) is larger than its int8 weights | MODELLED ✓ | tab:setup against ~124M parameters |
| 176 | 1.6% of the setup proof | **stale** | auto plan |
| 179-187 | fig:protocol | schematic | `paper_assets.fig_protocol` (`paper_assets.py:911`) |

### 1.9 `ext_v1.tex`
| Line | Claim | Class | Provenance |
|---|---|---|---|
| 38-40 | Clamp rate ~1e-4; 224,670 exceptions (1.8 MB) on the real OPT-6.7B; at most 65 with random weights | MEASURED | `F3a_real_weight_runs.md:300-301`; `C2:121-135` (OPT-1.3B: 65) |
| 48-50 | Table has ≤5.6e5 entries (Llama-2-7B) and 6.4e4 (real OPT-6.7B) | MODELLED / MEASURED | tab:params; `F3a:301` (64,305) |
| 55-56 | Exhaustive test on F_49 | MEASURED (test) | `tests/test_logup.py:7,139-151` |
| 66-67 | Up to 4.3e9 leaves | MODELLED, **outside repo** | [S] `V1_SPEC.md:347` (Llama-2-13B) |
| 79 | 36.0 kB for n=27 | MODELLED ✓ | 1,161 × 31 B |
| 87-90, 211-215 | Llama-2-13B at 2,048: 79 instances of ≤2^26 leaves, 2^-230.7 and 2^-215.3 | MODELLED, **outside repo** | [S] `V1_SPEC.md:347-348`. Its per-instance count 1,189/p^8 is for n=28 variables, not the 2^26 leaves the text states (conservative but inconsistent). `params_table.py` cannot regenerate it: its cell counts are hard-coded for Llama-2-7B |
| 96, 102-103, 109-112 | 50 ms for 20 variables; 12 GPU tests; 4 of 20 products; 0.52 s (15.6 ns/leaf, 0.83 s first); 1,492 tests | MEASURED | `C2:90,107,156,173,198-205` (jobs 1005308, 1005570); raw [S] `sp2027_research/server_out/gkrt_1005308.out`; script `code/experiments/8_v1/gkr_bench.py`. The 1,492 count is a snapshot |
| 136-159 | tab:v1 | MEASURED | `C2:179-190` (job 1005575); raw [S] `server_out/v1e5_1005575.out`. Script `code/experiments/8_v1/v1_bytes.py` (`--full --fs --lean --reps 3 --lmax 24 --prune-last`, as in `C2:207-208`). Library: `cut_protocol.run_cut_query` (`cut_protocol.py:194`), `logup_gkr`, `gkr_triton` |
| 161-168 | 2.03–2.29×; 2.07→0.96 GB and 2.02→0.91 GB; 2.2–2.3×; ~7 / ~12 vs 16–19 bits; <1 MB; 1.28× / 1.46×; 1.81× | MEASURED | `C2:121-139` (job 1005287; raw [S] `server_out/v1b_1005287.out`) |
| 169-171 | Accounting exact to the byte except multiproofs; timed queries within 0.3% | MEASURED | `C2:117-119` against `:181-187` |
| 171-172 | 2.24× (6.63→2.97 GB) | MODELLED, **outside repo** | [S] `V1_SPEC.md:46` |
| 173 | 1.78–1.85× | MODELLED, **outside repo** | [S] `V1_SPEC.md:34` |
| 174-179 | 1.73× (4.65→2.68 GB); 1.68× / 1.77× (OPT-125M, 512); 1.12× / 1.28× (1.24× / 1.53×) | MEASURED | `F3a:284-301`; `C2:145-151` (`v1_bytes.py --real facebook/opt-125m --smooth 0.55`) |
| 181-188 | ~22 ns per claim; 63–75%; 3.4–4.9 s; 9.7–11.6×; 1.6–1.9×; 2.2–2.8× | MEASURED ✓ (ratios recomputed) | `C2:179-196` |
| 186-189 | 1.44 s per block (2.49 s before); 45.4 vs 19.6 s | MEASURED | `V_V1N_fast_v1_verifier.md:10-12,257-288,361-388` (jobs 1009430, 1011440). Script `code/experiments/8_v1/v1_phases.py`. Records [A] `s10/.../9_quality/v1nat*.txt`, [A] `s4/.../v1n_s4.jsonl`. **Not in repo** |
| 200-201 | Proof reaches the weights' size at ~4,500 tokens with V1 instead of ~2,050 | MODELLED, outside repo, minor stale | [S] `V1_SPEC.md:56`. `EV_deployment_case.md:78` now says "about 2,100 (was 2,050)" |

### 1.10 `ext_generation.tex`
| Line | Claim | Class | Provenance |
|---|---|---|---|
| 33-37 | Per-prefix queries need ~1.3 TB against ~3.3 GB | MODELLED | `E1_generation.md:11` |
| 55-57 | Four tiny decoders: logits at the last G positions equal the per-prefix logits; forged tokens rejected | MEASURED (tests) | `tests/test_generation.py` (`E1:64-73`) |
| 59-63, 66-113 | tab:gen and tab:gen-full | MEASURED | `E1:75-83`. Records: [S] `sp2027_research/raw_sp2080/` (`E1:87`). **Not in repo**. Command: `python experiments/4_defence_benchmark/bench.py llm --model gpt2 --seq 64 --gen 256 --builds full --modes C:int --policy auto --wire --prune-last --lean --tag _wire` (`E1:91-92`). Mode C used the auto plan. The final run replaces it via `sp2027_numbers.tab_gen` (`sp2027_numbers.py:655`). Library: `protocol.run_query` (`:1908`), `Verifier.check_tokens` (`:940`), `transformer.with_lm_positions` (`:393`) |
| 124-131 | Γ end values; 933,901; 14.25; 6.5e-7 | MODELLED ✓ | `gumbel_table` |
| 134-137 | Each of 8 tokens chosen 380–620 times in 4,000 positions | MEASURED (test) | `tests/test_token_sampler.py:42` |
| 147-177 | Attribution method (4e-5); tab:quality-attr | MEASURED | `F2_quality.md:19-37`. **The script is not on sp2027**: `code/experiments/9_quality/attribution.py`, `hybrid.py`, `fixes.py` live only on branch `f2-quality` (commits `f65cac7`, whose message says "unverified", and `e695ee8`). Records also only there: `code/artifacts/results/9_quality/attribution_opt-125m_*.json` |
| 179-186 | LayerNorm analysis; best scalar gains 4 / 3 / 2 | MEASURED | `F2:39-43`; `F2a:105,136-137` |
| 192-194 | First recipe ×1.03–1.05 | MEASURED | `F2:62-63` (records `robust_opt-125m_w32_cal*.json`, only on `f2-quality`) |
| 194-197 | ×1.14–1.25; about ×2; zero-shot loss 2.8 / 0.8 points | MEASURED | `F2a:138,141`; `F2c_zero_shot.md:146,176` |
| 197-200 | Smooth-max ×1.026 (two draws); α study | MEASURED | `F2a:180,189-201` (jobs 1008668, 1011066). Records `ppl2048_search_round1.jsonl` (in repo) and `ppl2048_search2_*.jsonl` ([A] only). Script `ppl_2048.py` (`F2a:215-219`) |
| 200-201 | Cross-layer equalisation: 67.8 against 66.6 | MEASURED | `F2:78`; script `fixes.py`, **`f2-quality` only** |
| 203-238 | tab:zeroshot; agreement 78–99%; within 0.3 points of the harness | MEASURED (+ LITERATURE) | `F2c §5.1` (`:155-200`; jobs 1012440, 1012655). Records [A] `s11/.../zeroshot_cert_*.jsonl`. **Not in repo**. Script `code/experiments/9_quality/zeroshot.py` (`F2c:77-81`) |
| 240-245 | Per-row scales ×1.00–1.01 (top-1 agreement 0.87–0.88); no gain at 2,048 tokens | MEASURED | `F2:71-73` (`f2-quality`); `F2a:139,142` (`ppl2048_opt-*.jsonl` "rows" cells, in repo) |
| 247-254 | ZeroQuant-V2, CushionCache (5.4%), SLP (8.5%, 84.8%) | LITERATURE | bibliography |

### 1.11 `ext_evaluation.tex`
| Line | Claim | Class | Provenance |
|---|---|---|---|
| 43-44 | Int8 quantisation changed image-model accuracy by at most 0.4 points | MEASURED (course) | `code/README.md:220` (`text_numbers.py`) |
| 51-82 | tab:eval-improve | MEASURED | Docs per row: A1 `A1:94-98` (job 1004091), A2 `A2:61`, A3 `A3:53-54` (job 1005154), fused attention `B2_B3:95-102` (job 1005350), GPU decoder `B2_B3:84-85` (job 1005162), B5 `B5:170-180` (3.371→0.739 s; job 1006792). Scripts: `open_ab.py`, `forward_ab.py`, `encode_prof.py`, `gpu_ab.py`/`attention_ab.py`, `derive_ab.py`/`cpu_attn_bench.py` (`docs/improvements/README.md:33-36`). Raw logs only in [S] `sp2027_research/server_out/*` |
| 88-107 | 62% (1.105 of 1.78 s); 1.74 vs 1.54 s; 400 corruptions; 32% copies; 4.4% int8 GEMMs; 4.6× | MEASURED | `A1:23-29`; `A3:53`; `B2_B3:71,116,120` (job 1005394, `gpu_profile.py`); `B5:180` |
| 113-117 | 2.4 / 9.8 GB; 65.6→113.5 / 79.4 MB | MODELLED | tab:setup (cnn18c), consistent |
| 124-148 | tab:eval-real | MEASURED | `F3a:280-287` (jobs 1008673, 1009431). Records [A] `s3`/`s4` `.../9_quality/f3a_s{3,4}.jsonl`. **Not in repo**. Scripts: `perf.py --real`, `v1_bytes.py --real` (`F3a:58-104`) |
| 150-163 | Within 9% / 6%; 9–17%; 2,379 exceptions; OPT-6.7B 4.61 GB, 270 s, 20.2 s (2.7 / 11.4 / 6.1), 2.68 vs 4.65 GB, 288 s, 21.3 s, 5.41 GB, 2.23× | MEASURED | `F3a:299-330` |
| 170-194 | tab:eval-long; 3%; 0.83 GB per block | MEASURED | `V_LONG_b_long_prompts.md:5,21-40` (job 1011426). Records [A] `s9/.../vlongb/{perf,reexec}.txt`. **Not in repo**. The final run regenerates it via `sp2027_numbers.tab_long` |
| 200-202 | 305,193 paths; 13.7 GB; 325 MB | MODELLED + MEASURED | `analytic.py` / `text_numbers.py`; L40S tables (cnn16c era) |
| 215-223 | 0.17 / 0.16 / 0.23 / 0.22 / 0.26 Gb/s | MODELLED ✓ (I recomputed 3 cells) | arithmetic on body tab:eval-v1 = C2 job 1005575 |
| 229-256 | tab:eval-reexec, GPU rows | MEASURED | `F1_reexecution_baseline.md:30-36` (raw [S] `server_out/w4b_1004139.out`). The OPT-1.3B fp16 0.11 s is from A7 (`A7_serving_overhead.md:294`, 0.108 s, job 1008674), so that cell has mixed provenance |
| 229-256 | tab:eval-reexec, CPU rows (1.0 / 18.5; 4.5 / 93.2; 4.1 / 141.6) | MEASURED | `RX_reexecution_baselines.md:30-55` (job 1006979, 1007816) |
| 27-28 (header comment) | Cites 1.6 / 17.2, 3.5 / 78.8, 3.6 / 98.3 s | **stale comment** | does not match the table |
| 262-264 | ~1.1 s OPT-1.3B Kpre (extrapolated); zkLLM 0.90 s | MEASURED-extrapolated + LITERATURE | `B2_B3:89` |
| 270-289 | tab:eval-compare-v1 | MEASURED + LITERATURE | V1 and the caption's v0 values from `C2:124-135` (job 1005287); the "ours (v0)" column from body tab:eval-compare (L40S); "theirs" from `literature/reported_benchmarks.csv` |
| 291-296 | LeNet-5: zkCNN / Bionetta (LITERATURE); ours 5.7 ms, 2.8 ms, 63.3 kB | MEASURED (L40S improved tables) | ratios ✓ |
| 310-312 | 15.6 ns per leaf; 3.58 of 4.85 s | MEASURED | `C2:185,203` |

### 1.12 `ext_discussion.tex`
| Line | Claim | Class | Provenance |
|---|---|---|---|
| 36-44 | Measurement status (L40S / EPYC base runs; 2080 Ti / Xeon improvements) | statement | `artifacts/comparison/raw_l40s*`; `docs/improvements` |
| 51-58 | 82.2→48.9 MB; 1.68× / 1.77×; 1.94–2.06×; seven residual ops; 9–17% (1.66 vs 2.01 GB); within 9% / 6% | MEASURED | `C2:141-151,124-127`; `F3a:284-330` |
| 58-63 | OpenLLaMA-7B: 5.35 GB, 17% smaller, 21.0 s (20.6 s random); 11–17% and within 10% at 64 tokens; V1 2.02× (2.34× random) | MEASURED | `F3b_real_llama.md:56-105` (jobs 1015750-1015853). Records not in repo |
| 63-75 | ×1.076 / 1.026 / 1.008; 70% above with one gain; ×1.073, ×1.12, ×1.40, ×5.3; per-token exponent ×1.20 / ×3.0; ~0.3 GB | MEASURED | `F2a:194`; `F2b_Q_real_llama.md:1258-1259`; 0.3 GB from `F2b_Q:169` (MODELLED). Records [A] `f2bq*`. **Not in repo** |
| 77-87 | Prover ~10× slower; 2.3×; 1.6–1.9×; 1.1 Gb/s; 0.47 vs 0.70 Gb/s; 1.73× | MEASURED + MODELLED | `C2`; `V_V1N`; `EV_deployment_case.md:33-44`; `code/experiments/5_comparison/ev_breakeven.py` + `ev_inputs_draft.json` → `code/artifacts/results/ev_breakeven_draft.json` (in repo) |
| 95-104 | 6.7 GB weights; 4.1 / 142 s; 0.15 / 1.65 s GPU int8; fp32 4.5 / 93 s; fp16 0.037 / 0.57 s; 1.0 / 18.5 s; 4.0× / 5.0×; L40S GPU verifier 4.95 s | MEASURED | `RX:30-55`; `F1:34,51`; `A7:319-320` (1.65 s). **The 64-token 0.15 s was not located** |
| 106-121 | GPU-CC <7%; CommitLLM 12–14%, 87.5%, ~99.4%, 144 kB, 0.66 / 1.98 ms, 0.5 MiB, 48 MB | LITERATURE (+ MODELLED from tab:sok) | bibliography |
| 123-127 | zkLLM 10.3 min, 2.36 s; zkAgent 0.30 s per token; 46× | LITERATURE + MEASURED (L40S) | `literature.py`; body tab:eval-compare |
| 129-131 | 32-byte roots; 325 MB, 21×; 1.0 s EPYC / 431 ms L40S; 124.7 MB of columns | MEASURED, cnn16c era | L40S improved tables. Inconsistent with tab:params' 65.6 MB of columns (cnn18c); commit `2880932` flags these as replaced by the final run |
| 132-136 | 6.6 / 6.4 GB; 5.3 s at 10 Gb/s; ~21 prompts | MEASURED + MODELLED ✓ | arithmetic |
| 137 | 12.7 s key precompute | MEASURED | `A6_kpre_precompute.md:46-54` (job 1011448). Record [A] `s10/.../a6.txt` |
| 138-141 | 4–10×; 4.1–8.6×; 19–40× | MEASURED | `A7:288-295,339,353` (job 1008674). Records [A] `s3/.../9_quality/a7_pilot/*.jsonl`. **Not in repo** |
| 141-145 | Batch: 1.00 vs 4.25 MB per token | MEASURED | `A7:290,305` |
| 141-145 | Batch: 218.7 vs 324.5 MB; 127.8 MB; within 3% | MEASURED | `C4a_batch_bytes.md:15-35` (jobs 1012720, 1013041) |
| 151-158 | 2,050 / 4,500 tokens; 70% (0.38 of 0.55 s) | MODELLED / MEASURED | [S] `V1_SPEC.md:56`; `B5:182` |
| 164 | About six queries reveal the down projection (k=11,008) | MODELLED ✓ | arithmetic |
| 173-174 | TV median 0.0063 (τ=1, k=50) | MEASURED | `code/artifacts/results/sampler_tv_opt-1.3b.json` `summary` (in repo; job 1008675; `E2b_sampler_tv.md:201-212`). Script `code/experiments/9_quality/sampler_tv.py` (`E2b:337-340`) |
| 182-185 | Every implementable sampler and the best mixture stay below 27.7/N | MEASURED | `G3_min_adaptive_game.md` §4 |

## 2. Code named in the extended version (a refactor must update these)

Every name below exists today; the module path and line are where it is defined.

**Rendered in the text:**
- `ext_proofs.tex:43`: the directory `code/src/pvi/fullcheck`.
- `ext_proofs.tex:93,123,226`: `transformer.build_decoder` (`transformer.py:496`); `with_lm_positions` (`:393`).
- `ext_proofs.tex:102,225`: the `prune_last=True` / `prune_last` flag.
- `ext_proofs.tex:103-104`: `_rope` with an offset (`:213`); `_attention` (`:333`).
- `ext_proofs.tex:107`: `_Builder.last` with `calibrate_all` (`:428,469`).
- `ext_proofs.tex:108,214-215`: `tests/test_pruning.py` and `test_a_pruned_block_commits_its_q_on_its_own` (`:103`).
- `ext_proofs.tex:196`: `protocol.claim_bound` (`protocol.py:182`).
- `ext_proofs.tex:204-205`: `IntGraph.digest` (`graph.py:433`) and `meta`.
- `ext_proofs.tex:224-226`: `real_weights.build_opt_from_hf` (`real_weights.py:182`), `real_llama.build_llama_from_hf` (`real_llama.py:616`), `matches_benchmark_graph` (`real_weights.py:313`, `real_llama.py:909`).
- `ext_proofs.tex:228`: `prefix_tokens`, a keyword of `build_llama_from_hf`.
- `ext_proofs.tex:239,243,359`: `token_sampler.TokenSampler` (`:55`), `.noise` (`:75`), `.digest` (`:71`).
- `ext_proofs.tex:246-247`: `greedy_tokens` (`transformer.py:411`), `sample_tokens` (`token_sampler.py:92`).
- `ext_proofs.tex:252`: `Verifier.check_tokens` (`protocol.py:940`).
- `ext_proofs.tex:290`: `gumbel_table` (`token_sampler.py:40`).
- `ext_proofs.tex:294,605`: `run_query` (`protocol.py:1908`).
- `ext_proofs.tex:358`: `_absorb_statement` (`:1622`).
- `ext_proofs.tex:360`: `meta["lm_positions"]` (`protocol.py:947`).
- `ext_proofs.tex:364,605`: `run_cut_query` (`cut_protocol.py:194`).
- `Challenger` (`protocol.py:320`) and its methods:
  - `ext_proofs.tex:65,389`: `Challenger._key` (`:372`).
  - `ext_proofs.tex:67`: `.folding` (`:393`) and `ext` (`:408`).
  - `ext_proofs.tex:70`: `.columns` (`:450`).
  - `ext_proofs.tex:376`: `.absorb` (`:340`).
  - `ext_proofs.tex:377,398-399,424`: `.absorb_large` (`:345`).
- `ext_proofs.tex:537,582-583`: `_claims_message` (`:1691`), `_decoded_claims` (`:1757`), `_fold_message` (`:1815`), `_open_message` (`:1861`).
- `ext_proofs.tex:538,546-547,746,750`: `Verifier._pre` (`:767`), `Verifier.precompute` (`:844`).
- `ext_proofs.tex:558,719`: `params_for` (`:132`).
- `ext_proofs.tex:561,583,637,694`: `cut_protocol._query` (`cut_protocol.py:301`).
- `ext_proofs.tex:570-575`: `claimcodec.decode` (`:1972`), `decode_torch` (`:1983`), `decode_device` (`:2044`), `unpack_rows` (`:2183`), `unpack_field` (`:2154`), `cut_protocol.unpack_exceptions` (`:82`), `logup_gkr.Transcript.from_bytes` (`logup_gkr.py:40,59`), `extfield.unpack` (`extfield.py:282`).
- `ext_proofs.tex:572`: `_field_message` (`protocol.py:1794`).
- `ext_proofs.tex:604`: `ClaimCodecError` (`claimcodec.py:123`).
- `ext_proofs.tex:636`: `_disagrees` (`protocol.py:672`).
- `ext_proofs.tex:637`: `_columns_shape_ok` (`:1342`).
- `ext_proofs.tex:694`: `u_blob`.
- `ext_proofs.tex:720`: `plans.column_bits` (`plans.py:127`).
- `ext_proofs.tex:750,928`: `cut_protocol._kpre_y_ok` (`:582`).
- `ext_proofs.tex:754`: `CutPlan.from_graph` (`cut.py:115,124`).
- `ext_proofs.tex:926-927`: `torch.equal` and `protocol._disagrees`.
- `ext_proofs.tex:529`: `tests/test_fs_tree.py`.
- `ext_proofs.tex:609`: `tests/test_claimcodec.py`.
- **Verdict labels** (strings in the code): `range_or_shape`, `token`, `freivalds`, `columns_shape`, `cut_exception`, `cut_multiplicities`, `cut_gkr`, `kpre_y`, `cut_final`, `cut_logup` (`ext_proofs.tex:54-57,606-607`).
- **Transcript strings:** `pvi/fullcheck/v3` (`protocol.py:336`), `pvi/leaf/v1` (`:348,360`), `pvi/sampler/v1` (`token_sampler.py:73`).
- **Transcript labels:** `claims/PVC3`, `rows/I8`, `u/F31`, `cut/exc`, `cut/mult`, `graph`, `params`, `op/`, `col/`, `group/`, `table/`, `lookups`, `x`, `sampler`, `cut`, `claim/`, `lookup/`, `u/`, `cut/b/top`, `cut/b/gk.j`, `cut/b/vk`, `F31/tree`, `L/tree` (`ext_proofs.tex:375-406,580-581`).
- **Documentation paths in rendered text:**
  - `docs/improvements/D1_D2_fiat_shamir_binding.md` (`ext_proofs.tex:656`).
  - `docs/improvements/G3_inrange_attack.md` (`ext_proofs.tex:995`).
  - `params_table.py` (`app_params.tex:57`).
- **Other rendered names:** `fc2` (`ext_attack.tex:272`); the tokens `back` / `hacked` (`ext_attack.tex:151`); `ext_discussion.tex:41`, where `\texttt{L40S}` sits inside a `\todo`.

**In comments only:**
- `appendix_defence.tex:9-16`: `transformer.py`, `graph.py`, `requant` (`graph.py:113`), `residual_add` (`:140`), `_norm_int` (`transformer.py:166`), `_isqrt` (`:146`), `_lut` (`:186`), `_rope`, `_exp_lut` (`:259`), `_attention_core` (`:267`), `MatOp.max_input` (`graph.py:189,296`), `int8_ok` (`field.py:356`), `setup_proof.py`.
- `ext_proofs.tex:5`: `real_weights.build_opt_from_hf` / `prune_last`.
- `ext_proofs.tex:138-140`: `build_decoder(prob_max=...)`, `PROB_MAX`.
- `ext_proofs.tex:221`: `cut.py`.
- `ext_proofs.tex:507-510`: `logup_gkr.verify` and labels `mu{k}` / `lam{k}` (`logup_gkr.py:140-163`).
- `ext_proofs.tex:951-954`: `RandPathTest` (`sampling.py`).
- `ext_proofs.tex:1004-1006`: `G3_inrange_attack.md`, `inrange_llm_opt-1.3b_down.json`.
- `ext_attack.tex:268`: `experiments/2_attack/inrange_llm.py`.
- `ext_defence.tex:27-28,147-149`: `setup_bytes.py`, `D4_D5_untrusted_commitment.md`.
- `ext_evaluation.tex:27-28,120,166`: `B5_cpu_verifier.md`, `F3a_real_weight_runs.md`, `V_LONG_b_long_prompts.md`.
- `ext_generation.tex:8,11,16`: `E1_generation.md`, `E2_sampling.md`, `F2_quality.md`.
- `appendix_attack.tex:11` and `appendix_defence.tex:12-15`: the D4_D5, A1 and A2 docs.

## 3. Stale or inconsistent statements (inputs for the high-level review)

1. **Setup-proof text predates the cnn18c plan.** Commit `2880932` regenerated tab:params and tab:setup for cnn18c but left the following on the old auto plan:
   - `ext_defence.tex:121-122`: 11.3 GB and "+8%", against the table's 9.8 GB and +21%.
   - Block 2 at `ext_defence.tex:150-166`: 2.83 / 11.27 GB, "t0 reaches 12% of n_j". Under cnn18c t0 is at most about 1.9% of n.
   - `ext_defence.tex:176` and `ext_intro.tex:152-153`: "1.6%".
   - `appendix_defence.tex:175` and `ext_defence.tex:103-104`: "t=35, 2^-35". tab:params says t = 23 / 34.
   - `appendix_defence.tex:176-177` and `ext_defence.tex:107-108`: "3.2–6.6×" (old plan, source outside the repo).
   - `appendix_defence.tex:388-389`: "radii at most a quarter …" (unverified for cnn18c).

   Re-run `setup_bytes.py --policy cnn18c` and `params_table.py --policy cnn18c` and rewrite these.
2. **The mode-C byte numbers are the cnn16c-era L40S cells:** 325 MB, 124.7 MB of columns, 1.0 s / 431 ms (`ext_discussion.tex:129-131`; `ext_evaluation.tex:200-202`; `ext_intro.tex:53`). tab:params' 65.6 MB of columns is cnn18c. The final run is meant to replace them (commit `2880932`).
3. **Script defaults do not match the paper.**
   - `params_table.py` and `setup_bytes.py` default to `--policy auto`, while the paper's tables are cnn18c.
   - The final package selects cnn18c through the environment variable `PVI_C_POLICY` (`slurm/sp2027.sh:28,66`), not a command-line argument.
   - The library default policy is `paper` (`plans.py:56`).
4. **Measured values hard-coded in a script:** `params_table.py:90` holds the 2- and 3-block V1 cell counts with no record of where they came from.
5. **Two V1 security terms are not reproducible from the repo.** The values for the largest model (2^-215.3, 79 instances, 2^-230.7, 4.3e9 leaves) come from the design spec [S] `V1_SPEC.md:347-348`. Its n=28 variables do not match the "≤ 2^26 leaves" in `ext_v1.tex:213`.
6. **A stale comment:** the header of `ext_evaluation.tex` (`:27-28`) gives re-execution numbers that differ from tab:eval-reexec.
7. **Two cells mix sources:** tab:eval-reexec's OPT-1.3B cell (int from F1, before A2; fp16 from A7), and the Llama "0.15 / 1.65 s" in the same table's caption.
8. **The anonymous export drops files the extended version depends on.**
   - `scripts/export_anonymous.py:30-38` exports only `code/{src,tests,experiments,artifacts,...}`.
   - It leaves out `docs/`, although `ext_proofs.tex:656,995` cite two docs files in rendered text.
   - It also excludes `*.sbatch` and `slurm/`, which `real_llm.sbatch`, `adaptive_game_llm.sbatch` and `sp2027.sh` need.

## 4. Superseded variants the extended version mentions (keep as non-default options)

| Variant | Where it is mentioned | Current switch |
|---|---|---|
| One scalar gain in the integer LayerNorm | `ext_intro.tex:70`, `ext_discussion.tex:65`, `ext_generation.tex:185` | `ppl_2048.py --graphs scalar` (`:220`); `build_opt_from_hf` arguments |
| First smoothing recipe: α=0.55, percentile statistics. **Still load-bearing:** tab:eval-real and the real OPT-6.7B cost cells use it (`ext_evaluation.tex:124-126`) | `ext_generation.tex:192-197` | `build_opt_from_hf(smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)`; `v1_bytes.py --real --smooth 0.55` |
| Certified smooth-max α=0.6 (the current default) | throughout | `real_weights.CERTIFIED_RECIPE` (`F2a:199`) |
| Per-row power-of-two weight scales | `ext_generation.tex:240-245`, `ext_discussion.tex:87` | `ppl_2048.py --graphs rows` |
| Cross-layer equalisation | `ext_generation.tex:200` | `f2-quality` branch only (`fixes.py`) |
| 8-bit (255) against 11-bit (2,047) probabilities | `appendix_defence.tex:74`, `ext_proofs.tex:136` | `build_decoder(prob_max=)` |
| Per-token exponent at SwiGLU; q/k balance; SwiGLU balance; fixed prefix | `ext_discussion.tex:66-74` | `ppl_llama.py` options (33 arguments). The exponent is classed only as a "possible improvement", as the int8 rule requires |
| Commitment plans: auto with six candidates, cnn16c/17c/18c, R16c/R64c, tight, transposed layout | `ext_defence.tex:62-83` | `--policy` on most scripts; `PVI_C_POLICY` for the package |
| v0 against V1; modes C / K / Kpre; interactive / FS; wire; lookups; prune_last | throughout | `perf.py`, `v1_bytes.py`, `bench.py` arguments |
| float64 opening, non-int8 forward, single-thread encoder, fused Triton encoder (negative), unfused attention, host decoder, torch CPU verifier, V1 torch final check | tab:eval-improve, `ext_v1.tex:186-188` | mostly **environment variables, not arguments**: `PVI_FUSED_ATTN`, `PVI_GPU_DECODE`, `PVI_FUSED_CODEC`, `PVI_NATIVE_{ATTN,CHEAP,DECODE,GEMM,AVX512}`, `PVI_V1_NATIVE`, `PVI_FOLD_FAST`, `PVI_INT32_CLAIMS`, `PVI_CPU_ATTN`, `PVI_GKR_TRITON` (found by grep in `code/src` and `code/experiments`) |
| Per-prefix generation (against one prefill) | `ext_generation.tex:33-37` | `bench.py` without `--gen` |
| Samplers: uniform, static, contribution, floors δ, tail-first, zero-first, saliency | `ext_attack.tex:135-142,326-343` | `pvi.defences.sampling`. MNIST scripts hard-code `LAYER`, `PERCENTILE`, `EPSILONS`, `OPENING_BUDGET_PATHS`, `TARGET_CLASS` (`2_attack/run.py:55-56`, `3_sampling_fixes/run.py:45-46`, `floor_sampler.py:47-48`); `analytic.py` takes no arguments |
| Untrusted commitment / split commitment | `ext_defence.tex:101-176` | `setup_proof`; `setup_bytes.py --amortise` |

## 5. Scripts the extended version depends on

**In the repo (`code/experiments/`):**
- `1_reproduction/run.py`
- `2_attack/run.py`, `real_llm.py` + `real_llm.sbatch`, `inrange_llm.py`
- `3_sampling_fixes/run.py`, `floor_sampler.py`, `adaptive_game.py`, `adaptive_game_llm.py` (+ `.sbatch`), `adaptive_game_plots.py`
- `4_defence_benchmark/bench.py` (E1, the L40S cells) and `slurm/sp2027.sh` (final run: the cpuv, v1, serve, gpuv, gen, real, setup, quality, long and batch tiers)
- `5_comparison/analytic.py`, `text_numbers.py`, `literature.py`, `aggregate.py`, `paper_assets.py` (`fig_protocol`), `ev_breakeven.py`, `sp2027_numbers.py` (final run: tab:eval-reexec, tab:gen-full, tab_real, tab_setup, tab_long, tab_memory, tab_h2d, tab_serve, outcome counts)
- `6_improvements/params_table.py`, `setup_bytes.py`, `setup_timing.py` (the setup-time todo at `ext_defence.tex:123`), `open_ab.py`, `forward_ab.py`, `encode_prof.py`, `attention_ab.py`, `gpu_ab.py`, `gpu_profile.py`, `derive_ab.py`, `cpu_attn_bench.py`, `perf.py`, `gen_bench.py`, `h2d_bandwidth.py` (the F7 todo)
- `7_reexec/reexec.py`, `serve_ab.py`
- `8_v1/v1_bytes.py`, `gkr_bench.py`, `v1_phases.py`
- `9_quality/ppl_2048.py`, `ppl_llama.py`, `llama_diag.py`, `opt_builder.py`, `zeroshot.py`, `sampler_tv.py`
- Tests: `test_pruning`, `test_fs_tree`, `test_claimcodec`, `test_setup_proof`, `test_token_sampler`, `test_logup`, `test_generation`, `test_gkr_triton*`, `test_real_weights`, `test_real_llama`

**Outside `sp2027`:**
- Branch `f2-quality`: `9_quality/attribution.py`, `hybrid.py`, `fixes.py`, and their results (tab:quality-attr, per-row scales, cross-layer equalisation, the first-recipe robustness numbers).
- [S] `sp2027_research/attack_theory/check_guessing.py`, `check_merkle_leak.py`, `sok_formulas.py`.
- [S] `sp2027_research/v1_design/V1_SPEC.md` (byte model and security estimates).
- [S] `sp2027_research/idea_trust-rigor.md` (the 3.2–6.6×).

**Records missing from the repo but present in [A]:**
- `adaptive_game_opt-1.3b.json`
- `9_quality/ppl2048_search2_*.jsonl`
- `zeroshot_cert_*.jsonl`
- `a7_pilot/*.jsonl`
- `f3a_s{3,4}.jsonl`
- `v1n_s4.jsonl`, `v1nat*.txt`
- `vlongb/*.txt`
- `a6.txt`
- `f2bq*/`

E1's records are in [S] `raw_sp2080/`. The C2, A1, A3, F1 and B5 job logs are partly in [S] `sp2027_research/server_out/`.

## 6. Claims whose provenance is UNKNOWN or outside the repo

- **UNKNOWN:**
  - The Llama-2-7B 64-token GPU int8 re-execution time of 0.15 s (`ext_discussion.tex:97`; tab:eval-reexec caption `ext_evaluation.tex:233`). I did not locate it.
  - Where the counts hard-coded at `params_table.py:90` (behind tab:params' V1 rows) were measured.
  - Whether the "quarter of the unique-decoding radius" claim (`appendix_defence.tex:388-389`) and "3.2–6.6×" still hold under cnn18c.
- **Only outside the repo** (formula, spec or research note, no script in the branch):
  - `appendix_attack.tex:148-150,519-520,537-538` (toy checks).
  - tab:aligned and tab:sok-derivation (formula ✓, but no script).
  - The 2^-215.3 / 2^-230.7 / 79 instances / 4.3e9 terms (`appendix_v1.tex:286`, `ext_v1.tex:67,213-215`).
  - The 2.24× / 1.78–1.85× / 4,500 / 2,050 figures (`ext_v1.tex:171-173,200-201`; `ext_discussion.tex:152-153`).
  - "3.2–6.6×".
  - tab:quality-attr and the other `f2-quality`-only results.
- **Docs only, with no record in the repo:** tab:v1, tab:gen / tab:gen-full, tab:eval-improve, tab:eval-real, tab:eval-long, tab:zeroshot, the sampler game on OPT-1.3B, the certified-recipe ratios, the OpenLLaMA / Qwen / TinyLlama ratios, and the A6, A7 and C4a figures.

## 7. What I could not determine

- I did not compare the [A] archive records against the paper values; I only listed which files are there.
- I did not open `test_real_weights.py` or `test_real_llama.py` to confirm the pruned-logits equality test that `ext_proofs.tex:226-227` cites.
- I did not check the 78–99% agreement in tab:zeroshot's text for OPT-125M line by line.
- I did not run any script, so the values in commit `2880932` (cnn18c tab:params and tab:setup) are confirmed only from the commit diff.
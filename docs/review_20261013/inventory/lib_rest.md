# Review of `code/src/pvi` outside `fullcheck/`: the path-protocol baseline, attacks, samplers, MNIST code and results I/O

Paths are relative to the worktree root `.../scratchpad/wt_sp2027` (branch `sp2027`, clean tree, HEAD `08a7f1f`). I only read files, ran grep and parsed code with `ast`. I ran no tests or models.

## 0. Summary

1. **This half of the library has no dependency link with `fullcheck/`.** `fullcheck/` imports nothing from this scope. The only mention is a docstring at `code/src/pvi/fullcheck/datasets.py:3`. The only module used on both sides is `pvi.results`, and only experiment scripts import it.
2. **Nearly everything here is still reported by the S&P paper.** All of the course-paper MNIST experiments (reproduction, attack, samplers, floor sampler, adaptive game) are reported in the body or the extended version. The MNIST MLP is also one of the "six image models" of the defence evaluation (`paper_sp2027/sections/evaluation.tex:54`). The final-run package loads it through `pvi.zoo`, `pvi.training` and `pvi.data` (`code/experiments/4_defence_benchmark/bench.py:418-436`, called from `slurm/strong_gpu.sh:133`, in the `must` tier). So `zoo.py`, `data.py` and `training.py` are on the S&P path, not course-only.
3. **Only two scripts in the four course folders are not used by the paper:**
   - `3_sampling_fixes/adaptive_game_plots.py`: its figures and table are not in `paper_sp2027/figures` and not `\input`.
   - `2_attack/real_weights_ppl.py` (+ `.sbatch`): replaced by `9_quality/`.
4. **Very little dead code; a lot of duplication.** Notable items:
   - The vectorised acceptance DP lives in an experiment script (`adaptive_game.py:142-241`), and a test loads that script by file path.
   - The "winner minus runner-up" margin direction is written out five times.
   - Each sampler normalises its own weights, and the caller normalises them again.
   - Two path-protocol implementations exist (float here, integer in `fullcheck/sampling.py`). They share class and function names but differ in meaning.
5. **Naming is misleading for a reviewer.** `pvi.defences` holds the samplers that *fail*; the paper's defence is `fullcheck`. `pvi.protocol` is the attacked baseline, while `pvi.fullcheck.protocol` is ours. `pvi.experiments` collides with `code/experiments/`. Docstrings cite the course report ("report, Theorem 3.1", "report, Sec. 4.2").
6. **One reproducibility gap for the paper:** the stored result `adaptive_game_opt-1.3b.json` is missing from the repository (§9, P1).

## 1. Inventory of the modules

The "Paper use" column uses S&P labels: `sec:path` is 3.2, `sec:sampling` 3.3, `sec:break` 3.5, `sec:eval-setup` 7.1.

| Module | Purpose | Public API | Importers outside the library | Paper use | Verdict |
|---|---|---|---|---|---|
| `nn/architecture.py` | Layered-DAG layers with per-neuron parents, weight rows and local relation | `Architecture`, `InputLayer`, `DenseLayer`, `Conv2dLayer`, `MaxPool2dLayer`, `apply_activation`, `NO_WEIGHT_GROUP` (`:61-71`) | `3_sampling_fixes/adaptive_game.py:61`; tests `conftest.py:13`, `test_architecture.py:16` | MLP and the reproduction's CNN (`zoo.py:67-71,220-227`) | Keep. Conv and pool are needed by the `cnn_disjoint_data` reproduction setting (`ext_attack.tex:117-119`, "four ... settings") |
| `nn/network.py` | `Trace` and `TracedNetwork` (EvalTrace, `forward_from` for honest recomputation) | `ModelParameters`, `Trace`, `TracedNetwork` (`:23`) | `1_reproduction/run.py:26`, `adaptive_game.py:63`, 5 test files | Every MNIST number | Keep |
| `nn/gradients.py` | d(margin)/d(activation) | `output_direction_gradient`, `saliency`, `requires_dense_network` (`:30`) | `adaptive_game.py:62` | Neuron ranking (`ext_attack.tex:113-116`), saliency sampler | Keep; absorb the duplicates (§4) |
| `nn/models.py` | MLP and small-CNN factories, torch bridge | `mlp_architecture`, `small_cnn_architecture`, `build_torch_module`, `extract_parameters` (`:32-37`) | Tests only, plus `training.py` and `zoo.py` | Model training | Keep |
| `commitments/merkle.py` | SHA-256 Merkle vector commitment with domain tags and a length-bound root | `MerkleVectorCommitment`, `MerkleParams`, `MerkleOpening`, `encode/decode_f32_vector` (`:53-59`) | Tests `test_merkle.py:13`, `test_protocol.py:13,310` | Path protocol of `sec:path` | Keep (single file) |
| `protocol/*` | Anchuri et al.'s RandPathTest: params, messages, prover, verifier, session, path samplers, model and trace commitments | `protocol/__init__.py:11-21` | `1_reproduction/run.py:27`, `2_attack/run.py:46`, `3_sampling_fixes/run.py:36`, `floor_sampler.py:38`, `adaptive_game.py:64-65`, 7 test files | Reproduction, backdoor end-to-end (99.96% of 2,300), zeroing end-to-end (2,000 of 2,000) | Keep; rename the package (§8) |
| `experiments/analysis.py` | Exact acceptance DP for any sampler; inconsistency masks | `acceptance_probability`, `inconsistent_nodes`, `locally_consistent_mask`, `InconsistencyReport` (`:35-40`) | `2_attack/run.py:45`, `3_sampling_fixes/run.py:35`, `floor_sampler.py:37`, `adaptive_game.py:60`, 4 test files | "computing path acceptance exactly" (`attack.tex:186-187`) | Keep; move and absorb the fast path (§4) |
| `attacks/tamper.py` | Single-neuron and stealthy (in-range) tamper | `TamperPlan`, `apply_plan`, `evaluate_plan`, `plan_single_neuron_flip`, `plan_stealthy_flip`, `smallest_flipping_value` (`:70-78`) | `2_attack/run.py:32-43`, `3_sampling_fixes/run.py:26`, `floor_sampler.py:33`, `adaptive_game.py:50` | `sec:path` edit, `sec:break` MNIST, `ext_attack.tex:127-129` | Keep |
| `attacks/backdoor.py` | Trigger-conditional backdoor, activation ceilings | `BackdoorAdversary`, `PatchTrigger`, `calibrate_activation_ceilings`, `ServedResponse` (`:34-39`) | `2_attack/run.py`, `floor_sampler.py:32` | `attack.tex:191-193` | Keep |
| `attacks/baselines.py` | Anchuri et al.'s own forgeries (Sec. 7.2, App. E and F) | `gradient_forge`, `injection_forge`, `inverse_transform_forge`, `logit_swap_target` (`:48-54`) | `2_attack/run.py:32-42` | "hundreds of the MLP's 778 trace nodes" (`attack.tex:190`, `ext_attack.tex:122-124`) | Keep |
| `defences/sampling.py` | Value-aware path samplers | 6 samplers + `weight_magnitude_importance` (`:52-60`) | `3_sampling_fixes/run.py:28`, `floor_sampler.py:35`, `adaptive_game.py:51` | `attack.tex:195-206`, `ext_attack.tex:135-141` | Keep; rename and merge (§4, §8) |
| `defences/adaptive.py` | The adversary that adapts to a sampler | `plan_evasive_flip`, `plan_adaptive_stealthy_flip`, `EvasionSearch` (`:32`) | `3_sampling_fixes/run.py:34`, `floor_sampler.py:36` | 0.20/0.11/0.016/9.2%, zeroing ~14 neurons | Keep; it is an attack, so move it (§8) |
| `zoo.py` | MNIST model specs, substitution settings, 8-bit fake-quant substitute | `SETUPS`, `ModelSpec`, `SubstitutionSetup`, `mlp_architecture_for`, `cnn_architecture_for`, `quantise_network`, `all_model_specs` | `0_train_models/train.py:20`, `1_reproduction/run.py:30`, `2_attack/run.py:49`, `3_sampling_fixes/*`, **`4_defence_benchmark/bench.py:421`** | MNIST MLP (attack and defence evaluation), reproduction | Keep |
| `data.py` | MNIST as float32 in [0,1], class and slice subsets | `Dataset`, `load_classification`, `load_raw`, `DEFAULT_DATA_ROOT` | `2_attack/run.py:44`, `3_sampling_fixes/*`, **`bench.py:419`** | Same | Keep |
| `training.py` | Seeded MLP training, `.npz` save and load | `TrainConfig`, `train_network`, `save_network`, `load_network` | 0/1/2/3 scripts, **`bench.py:420`**, `test_torch_bridge.py:86` | Same | Keep |
| `results.py` | Atomic group-writable JSON; `git_sha` without a git binary | `write_json`, `git_sha` (`:25`) | `write_json`: 1/2/3 scripts and `4_defence_benchmark/train.py:22`. `git_sha`: 13 call sites in `6_improvements`, `7_reexec`, `8_v1` (e.g. `perf.py:204`, `serve_ab.py:895`, `v1_bytes.py:347`) | Infrastructure for S&P runs | Keep |
| `__init__.py`, `README.md` | Package docstring and module table | `__all__` (`__init__.py:36`), `__version__` (`:38`, unused) | `bench.py:48,51-58` checks `pvi.__file__` sits at `code/src/pvi` | — | Keep; rewrite the README's "Paper" column (§9) |

## 2. Per-script decisions for `0_train_models`, `1_reproduction`, `2_attack`, `3_sampling_fixes`

| Script | Result file | Where the S&P paper reports it | Decision |
|---|---|---|---|
| `0_train_models/train.py` | `artifacts/models/*.npz` (tracked, with `MODELS.sha256`) | Produces every MNIST model, including the MLP of `evaluation.tex:54` | **Keep.** Retraining is not bit-reproducible across platforms (`0_train_models/README.md:14-16`), so the shipped `.npz` files are what matter |
| `1_reproduction/run.py` | `reproduction.json` (1.0 honest acceptance; 99.8–100% detection; 21–37% at zero tolerance; quantised agreement 1.0) | `ext_attack.tex:117-124`; `attack.tex:187` | **Keep** |
| `2_attack/run.py` | `attack.json` (support 256–778 against 1; 0.998/0.996; backdoor 0.99957 of 2,300; 250 paths 0.613; p95 support 5.43, acceptance 0.989) | `attack.tex:189-193`; `ext_attack.tex:127-129` | **Keep.** Only the p95 row of the four stealth percentiles is reported (`run.py:230`); keep the others behind an option |
| `2_attack/real_llm.py` (+ `.sbatch`, `REAL_LLM.md`) | `real_llm_attack*.json` | `attack.tex:212-216`, `ext_attack.tex:148-155` | **Keep.** It uses only `fullcheck` and Hugging Face |
| `2_attack/inrange_llm.py` | `inrange_llm_opt-{1.3b,6.7b}_{any,down}.json` | `attack.tex:216-231`, `tab:inrange` (`ext_attack.tex:276-317`), `ext_proofs.tex:995-1006` | **Keep** |
| `2_attack/real_weights_ppl.py` (+ `.sbatch`) | `real_weights_ppl_opt-*.json` (73.8 / 37.4 / 36.0) | None of these numbers is in `paper_sp2027/sections`. It is named only in a comment at `quality.tex:4`. The 128-token-window setting in `ext_generation.tex:152,194` is served by `9_quality/ppl_smooth.py`, which uses "the same windows" (`ppl_smooth.py:5`). Nothing imports this script | **Drop**, or fold it into `9_quality` as a legacy option. Confirm with the `9_quality` reviewer. Also remove the stale claims at `code/README.md:222` and `5_comparison/README.md:116` |
| `3_sampling_fixes/run.py` | `defence.json` (evasive detection: uniform 0.00195, static 0.00107, contribution 0.0923, saliency 0.000157; zeroing support 14.35, 2,000/2,000, 65%) | `attack.tex:195-198`, `ext_attack.tex:135-138` | **Keep** |
| `3_sampling_fixes/floor_sampler.py` | `floor_sampler.json` (eps=1: 0.0198, i.e. 10.2x uniform; other floors 0.86–5.8x) | `attack.tex:199-201`, `ext_attack.tex:138-141` | **Keep.** It takes about 37 min; it would get faster with the vectorised DP (§4) |
| `3_sampling_fixes/adaptive_game.py` | `adaptive_game_mlp_mnist.json` (mixture at cap 50: 0.0306 in layer 1, 0.0967 in layer 2; p95 support 5.075; R2_max 0.95) | `attack.tex:131-135` (75%/95%), `:201-206` (3.1%/9.7%), `ext_attack.tex:129-133`, `ext_discussion.tex:182-183` | **Keep** |
| `3_sampling_fixes/adaptive_game_llm.py` (+ `.sbatch`) | `adaptive_game_opt-1.3b.json`, **not in the repository** | `attack.tex:222-226`, `ext_attack.tex:325-344` | **Keep, and commit the missing result** (§9, P1) |
| `3_sampling_fixes/adaptive_game_plots.py` | `figure_g3a.pdf`, `figure_g3b.pdf`, `table_g3.tex` (all tracked) | Not used. The paper's figures are only `ev_breakeven`, `overview_attack`, `overview_defence` and `protocol` | **Drop** from the submission, together with the three tracked outputs. Its default `--llm` path (`adaptive_game_plots.py:235`) points to the missing file |

**Provenance check.** The stored MNIST results were last changed at `336a03c` ("Final MNIST results"). Since then, the four MNIST scripts changed only in their docstrings, and `defences/` only gained the two new samplers (`git diff --stat 336a03c HEAD`). So the stored results match the current logic. I cannot confirm they were produced exactly at `336a03c`.

## 3. Dead code (by AST and word-count reference scan)

The scope is close to dead-code-free. Findings:
- **Used only by tests:** `TracedNetwork.recompute` (`nn/network.py:212`), used at `test_architecture.py:52,200` and `test_protocol.py:60`. Keep it as the test oracle, or move it into the tests.
- **Unused:** `pvi.__version__` (`__init__.py:38`). Harmless.
- **One caller only:** `Trace.replace_layer` (`nn/network.py:78`), used only by `injection_forge`. Keep.
- **Redundant lazy imports:** `tamper.py:311` and `adaptive.py:164` re-import `output_direction_gradient` inside a function, although each module already imports `pvi.nn.gradients` at the top (`tamper.py:44`, `adaptive.py:27`). The `TYPE_CHECKING` block and the lazy import in `analysis.py:32-33,125` are also unnecessary: I traced the import graph and found no cycle.
- **Private names used from outside:** `protocol.sampling._as_distribution` (`:177`) is imported by `adaptive_game.py:65` and `analysis.py:125`. `ZeroAwareContributionSampler._epsilon` is read at `adaptive_game.py:126,171`. Make both public.

## 4. Duplication

**Inside this scope (and its scripts):**
1. **Acceptance DP.** `adaptive_game.py:106-241` (`dense_transition_matrix`, `_accept_given_masks`, `fast_acceptance`, `accept_vector`) re-implements `analysis.acceptance_probability` (`analysis.py:95-171`) with a vectorised path for source-dependent dense samplers. `test_adaptive_game.py:44-49` loads the script by file path to check the two agree. Move it into the library's analysis module, and keep the per-neuron version as the reference. That also speeds up `floor_sampler.py` and `run.py`, and removes the test's dependence on a script location.
2. **Margin direction (winner vs. runner-up or target):** written out in `gradients.py:98-109`, `defences/sampling.py:264-270`, `tamper.py:313-316`, `adaptive.py:166-169` and `adaptive_game.py:399`. Replace with one helper in `gradients`.
3. **Dense-network check:** `baselines._require_dense` (`baselines.py:110`) duplicates `gradients.requires_dense_network` (`gradients.py:33`).
4. **Normalisation with uniform fallback** is inlined in every sampler (`defences/sampling.py:114,160,182,211,235,286,301`). The caller then applies `_as_distribution` again (`protocol/sampling.py:152,157`; `analysis.py:368,381,389`). Remove one of the two.
5. **Contribution samplers.** `LocalContributionSampler` (`defences/sampling.py:134`) is identical to `ZeroAwareContributionSampler(epsilon=0)` (`:185`) in both its start and transition distributions. `floor_sampler.py:47` already uses `eps=0` as "plain contribution". Make one an alias of the other.
6. **Stealthy planners.** "push to ceiling or 0 by gradient sign" appears in both `tamper.py:321` and `adaptive.py:173`. `adaptive_game.py:374-523` has a third family of in-range attacks (A1–A4) that the paper reports (5.1 neurons, the LP inputs). That family belongs in the library's attacks.

**With `fullcheck/`.** None of these share code; most are deliberate:
- **Path protocol:** float, any sampler, 1e-4 tolerance here (`protocol/`, `analysis.py`) against integer, uniform only, exact visit probabilities in `fullcheck/sampling.py:326,392`. Keep both: the MNIST reproduction needs the float version (the 21–37% zero-tolerance result is a float effect), and the CNN and LLM numbers need the integer one.
- **Same names, different meanings:** `TraceCommitment` exists at both `protocol/commitments.py:150` and `fullcheck/sampling.py:100`. `paths_for` means "derive the paths from a challenge" at `protocol/prover.py:90` and "number of paths for λ" at `fullcheck/sampling.py:392`. Rename one of each pair.
- **Merkle:** `commitments/merkle.py` against `fullcheck/commitment.py:146-207` (`MerkleTree`, multiproofs) and `fullcheck/sampling.py:48` (its own padding). Do not merge: MNIST path-protocol proof bytes are not reported, and the reproduction's commitment is the row-wise scheme of Anchuri et al.
- **MNIST loaders:** `data.py:54-76` ([0,1] pixels) against `fullcheck/datasets.py:51-56` (mean/std normalised). Both read the same `code/data` copy. Keep both but cross-reference them; the normalisations differ on purpose.
- **Quantisation:** `zoo.quantise_network` (`zoo.py:115-135`, per-tensor fake quantisation, the reproduction's substitute) against `fullcheck/quantize.py` (exact int8). Rename the former, for example to `fake_quantise_substitute`.
- **MLP torch rebuild:** `bench.py:418-436` rebuilds the MLP as an `nn.Sequential` instead of using `nn/models.py`. Minor.

## 5. Superseded paths the paper still mentions (keep as non-default options)

- **In-sample range calibration.** `2_attack/run.py:229-239` calibrates on `test_x[:3000]` and attacks `test_x[0:60]`. `3_sampling_fixes/run.py:110,185-187` does the same with `[0:40]`. `adaptive_game.py:15-19` names this flaw and fixes it with ranges held out on the train split. The paper reports the in-sample result ("about five neurons ... 99%", `ext_attack.tex:127-128`) right next to the held-out one (5.1 neurons, 99.0%, `:128-133`). Keep in-sample as a `--calib test-overlap` choice.
- **`plan_evasive_flip` with a 3-bucket candidate pool** (`adaptive.py:47-122`), against `adaptive_game`'s fuller pool A1. Both are reported (`run.py` sweep; the LP). Keep both.
- **Gradient-saliency sampler.** It cannot be implemented by a real verifier (`defences/sampling.py:238` docstring), but it is reported as a bound (0.016%). Keep.

## 6. Hard-coded parameters to turn into CLI arguments

Line numbers are per file.

- **`0_train_models/train.py`:** only `--force` and `--only` exist (`:27-28`).
  - Hyperparameters are fixed in `zoo.py:142-143,182,188` (epochs, batch size, learning rate, seed).
  - The output folder is fixed at `:22`.
  - The data root is fixed in `data.py:26`, with no environment override. `fullcheck` uses `PVI_CIFAR_ROOT`, so add `PVI_DATA_ROOT` or `--data-root`.
  - `data.py:62-63` downloads MNIST, so the friend's smoke test needs network access or a pre-filled `code/data`.
- **`1_reproduction/run.py`:**
  - Query seed `seed=0` (`:46`).
  - Zero-tolerance sample silently capped at 200 (`:71`).
  - Tolerance (`ProtocolParams` default `1e-4`, `protocol/params.py:39`).
  - No `--out`.
  - Challenges are always fresh from `secrets` (`protocol/session.py:48` → `protocol/sampling.py:50-52`). Add an injectable seed or RNG, otherwise golden outputs cannot be captured.
- **`2_attack/run.py`:**
  - `TARGET_CLASS` and `OPENING_BUDGET_PATHS` (`:55-56`).
  - Baseline query count 8 (`:64`) and substitute path (`:66`).
  - Optimiser steps and decay (`:76-77`).
  - Trigger size and value, and the layer (`:142-145`).
  - Cap of 100 triggered queries (`:166`).
  - Calibration on `[:3000]` and percentiles (`:229-230`).
  - Silent caps at 60 and 100 queries (`:239,292`).
  - No `--out`.
  - Small bug: `block_opening_budget` passes a plan that may be `None` (`:205-207`).
- **`3_sampling_fixes/run.py`:**
  - `LAYER` and `PERCENTILE` (`:45-46`).
  - `candidate_neurons=60` (`:85`; the library default is 96).
  - Silent cap at 40 queries (`:119`).
  - 50 runs per query (`:134`).
  - Calibration on `[:3000]` (`:110,186`).
  - No `--out`.
- **`3_sampling_fixes/floor_sampler.py`:**
  - `EPSILONS` and `LAYER` (`:47-48`).
  - Calibration on `train_x[:3000]` at p100 (`:82-84`).
  - `max_value` (`:104`).
  - No `--out`.
  - The stored eps=0 detection is -5.6e-17; clamp it to 0.
- **`3_sampling_fixes/adaptive_game.py`:** `KMAX`, `K_LIMITS`, `PATH_COUNTS`, `FLOOR_EPSILONS` and `NEAR_ZERO` (`:75-83,410`).
- **Library knobs:** `plan_evasive_flip` hard-codes `max_value=1e4` (`adaptive.py:104`).

## 7. What to keep, merge or drop

- **Keep:** every module in this scope.
- **Merge:**
  - The acceptance DP fast path into the analysis module.
  - The margin direction helper and the dense check into `gradients`.
  - `LocalContributionSampler` into `ZeroAwareContributionSampler`.
  - The sampler normalisation into one place.
  - The A1–A4 attacks of `adaptive_game` into the library's attacks.
- **Drop:**
  - `adaptive_game_plots.py` and its three tracked outputs.
  - `2_attack/real_weights_ppl.py` and its `.sbatch`, pending the `9_quality` reviewer's confirmation.
  - `__version__` is optional to drop.

## 8. Proposed functional layout

```
code/src/pvi/
  results.py                    shared I/O (write_json, git_sha); drop "Three of us" (results.py:3)
  pathtest/                     the attacked baseline: Anchuri et al.'s path protocol, float reimplementation
    net.py                      <- nn/architecture.py + nn/network.py
    gradients.py                <- nn/gradients.py (+ margin_direction, require_dense)
    torch_bridge.py             <- nn/models.py
    merkle.py                   <- commitments/merkle.py
    protocol/                   <- protocol/* (sampling.py -> paths.py; as_distribution public)
    analysis.py                 <- experiments/analysis.py + adaptive_game.py:106-241
    samplers.py                 <- defences/sampling.py (value-aware samplers: "fixes that fail")
  attacks/
    tamper.py, backdoor.py      unchanged
    anchuri_forgeries.py        <- attacks/baselines.py
    evasion.py                  <- defences/adaptive.py
    inrange.py                  <- adaptive_game.py A1-A4 (optional)
    llm.py                      <- Hook / prefix_cache / last_logits / attack from 2_attack/{real_llm,inrange_llm}.py (optional)
  mnist/                        <- zoo.py, data.py, training.py (also used by the defence benchmark's MLP)
  fullcheck/                    unchanged (our defence)
```

The optional `llm.py` would replace today's `sys.path` imports between scripts (`adaptive_game_llm.py:63-67,144`; `inrange_llm.py:40-41,170-171`).

**Import sites to update** (all listed in §1): the scripts of 0/1/2/3, `bench.py:419-421`, and the tests `conftest.py`, `test_{adaptive_game,analysis,architecture,attacks,defences,merkle,protocol,sampling,torch_bridge}.py`. Also update `test_adaptive_game.py:44-49`, which loads a script by path, and `bench.py:51-58`, which pins `pvi` to `code/src/pvi`. Since this is a fresh branch, I suggest moving the modules outright rather than adding compatibility shims.

**Minimal alternative:** rename only `pvi.experiments` → `pvi.analysis`, `pvi.defences` → `pvi.samplers`, and move `defences/adaptive.py` into `attacks/`.

## 9. Paper-level points and high-level decisions in this scope

- **P1 (must fix).** `attack.tex:222-226` and `ext_attack.tex:325-344` report the OPT-1.3B sampler game, but `adaptive_game_opt-1.3b.json` is not tracked (`git ls-files`). The development diary names job 1008671 (`docs/improvements/G3_min_adaptive_game.md:9-10,282`). Fetch it from the cluster into `artifacts/results` before freezing.
- **P2 (decision for you).** The MNIST sampler numbers come from three runs with three protocols:
  - `run.py`: 40 queries, in-sample p99.
  - `floor_sampler.py`: 20 queries, train p100.
  - `adaptive_game.py`: 40 queries, train ranges.

  `attack.tex:195-206` mixes all three. Either say so once in `ext_attack.tex` "Measured Attacks: Details", or take every MNIST sampler number from the `adaptive_game` run and keep the older two scripts as options. The second choice changes printed numbers: at the unconstrained cap, floor-1 gives 2.15% against the 2% from `floor_sampler.py`.
- **P3.** Say once that the MNIST path protocol is the float implementation with a 1e-4 tolerance, while the CNN and LLM numbers come from the integer implementation (`attack.tex:184-187` does not distinguish them).
- **P4.** The in-sample 95th-percentile sentence (`ext_attack.tex:127-128`) is superseded by the held-out sentence that follows it. I suggest cutting it from the paper and keeping the script option.
- **P5.** "The 9.2% is one sampler against the plain attack" (`ext_attack.tex:140-141`). The stored 9.2% is `detection_evasive`; plain (naive) detection is 11.3% (`defence.json`). `attack.tex:196` wording ("the best single-neuron edits") is correct; align the extended version with it.
- **P6 (anonymity and correctness).** Code cites the course report:
  - `protocol/sampling.py:197-199`, `defences/sampling.py:40,97-98` ("report, Theorem 3.1").
  - The docstrings of `1_reproduction/run.py:1`, `2_attack/run.py:1`, `real_llm.py:1`, `real_weights_ppl.py:1`, `run.py:1` and `floor_sampler.py:1`.
  - The "Paper" column of `src/pvi/README.md:3-12`, the claims map in `code/README.md:147-150,213-228`, and `3_sampling_fixes/README.md:11-13` ("Proposition 3.1"; now `thm:dich` / `prop:blind`).

  Replace these with S&P labels. Outside my scope I also noticed "TAU cluster" in `code/requirements.txt:5`. And `scipy` is required by the LP (`requirements.txt:14`) but missing from `pyproject.toml`'s dependencies.
- **P7.** End-to-end empirical rates (99.96% of 2,300; 2,000 of 2,000; 3,000 runs) use fresh `secrets` challenges (`protocol/session.py:48`), so they cannot be rerun bit for bit. Add a `--seed` before capturing golden outputs.

## 10. What I could not determine

- Whether `adaptive_game_opt-1.3b.json` still exists on the cluster.
- Which `9_quality` script produced each 128-token number in `ext_generation.tex` (outside my scope).
- Whether retrained MNIST models reproduce the paper's digits on the final platform.
- Whether the tests pass (I did not run them).
- What "Agent bucket is ok" in the request refers to: nothing in this scope matches it.
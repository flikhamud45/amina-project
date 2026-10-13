# `code/src/pvi/fullcheck`: module inventory, reachability, dead and superseded code, proposed grouping, environment variables

**Main findings**
- **Unused modules:** no runtime module is wholly unused. `opcount` is not reached by the run package or the paper scripts. `reference` is reached only by the smoke tests, which is its purpose.
- **Dead functions:** 3 are never referenced anywhere. 10 more are referenced only by tests. 6 paths run only when a switch is set off its default.
- **Course-report code still present:** the legacy weight-cache bug, kept only for the `_nofix` controls. Three variants the paper does not mention can also go (V1 "scatter", the "hybrid" CPU attention, Merkle worker processes).
- **Library defaults are the course report's settings.** Every paper setting is a non-default flag in `sp2027.sh`, and the scripts disagree on the default commitment policy.
- **The biggest risk for the refactor:** the graph digest hashes the Python module path and name of every cheap-op function. Moving `graph`, `transformer`, `real_weights`, `real_llama` or `quantize` changes every Fiat–Shamir transcript and the stored digests. In mode C it can also change proof bytes.
- **Environment variables:** the library reads 24, plus the standard `CXX`. Most are read once at import, and one run-package script switches a kernel path by patching a private module global.

**Abbreviations:** `fc/` = `code/src/pvi/fullcheck/`, `exp/` = `code/experiments/`, `T/` = `code/tests/`, `sp.sh` = `exp/4_defence_benchmark/slurm/sp2027.sh`, `tex/` = `paper_sp2027/sections/`.

**Method:** read-only. I built an AST import graph over all 184 tracked `.py` files under `code/` and `scripts/`, plus three name-reference passes. I made targeted greps, read the docstrings, and read the switch sites, `sp.sh`, `strong_gpu.sh` and the paper. Nothing was executed except light AST parsing.

**Entry points used for "run package":** the commands in `sp.sh`, and the scripts those commands load dynamically:
- Called directly by `sp.sh`:
  - `exp/8_v1/v1_bytes.py` (`sp.sh:132`)
  - `exp/6_improvements/perf.py` (`sp.sh:133`)
  - `exp/7_reexec/reexec.py` (`sp.sh:134`)
  - `exp/6_improvements/gen_bench.py` (`sp.sh:136`)
  - `exp/7_reexec/serve_ab.py` (`sp.sh:228`)
  - `exp/6_improvements/setup_timing.py` (`sp.sh:360-366`)
  - `exp/6_improvements/h2d_bandwidth.py`
  - `exp/9_quality/ppl_llama.py` (`sp.sh:383-396`)
  - `bench.py` via `strong_gpu.sh` and `bench.sbatch:41`
  - the smoke-tier tests (`sp.sh:143-147`)
- Loaded dynamically by those scripts:
  - `perf.py:162,207,237` loads `ab_verifier.py`, `bench.py` and `v1_bytes.py`.
  - `gen_bench.py:74,209` loads `serve_ab.py` and `v1_bytes.py`.
  - `serve_ab.py:436` loads `reexec.py`.
  - `setup_timing.py:62` loads `gen_bench.py`.
  - `v1_bytes.py:69-70,95-96` loads `9_quality/ppl_llama.py` and `common.py`.
  - `ppl_llama.py:68-69` loads `opt_builder.py` and `ppl_2048.py`.
- Scripts the paper names: `params_table.py` (`tex/app_params.tex:57`), `setup_bytes.py` (`tex/ext_defence.tex:149`), `real_weights_ppl.py` (`tex/quality.tex:4`) and `inrange_llm.py` (`tex/ext_attack.tex:268`). `exp/2_attack/real_llm.py:45` also uses `sampling.paths_for` for the paper's "305,193 paths" (`tex/ext_evaluation.tex:200`).

**Reachability result:** the transitive closure of these entry points covers every module except `opcount` and `reference`. `reference` is reached through the smoke test `T/test_gpu_verifier.py:33`.

---

## 1. Module inventory (33 files, 15,715 lines)

"Src" lists library importers. "Run" lists run-package scripts with import lines. "T" is the number of test files that import the module.

| Module | Lines | Purpose (one line) | Main public API (`__all__`) | Imported by | Used by paper / run package |
|---|---|---|---|---|---|
| `__init__` | 58 | Re-exports the package API | P, CheapOp, IntGraph, MatOp, Challenger, Prover, SecurityParams, Verifier, commit_graph, params_for, run_query, soundness_bits (`fc/__init__.py:45`) | `perf.py:208` (`import pvi.fullcheck as fullcheck`), `ab_verifier.py:37` | yes (`perf` and `ab_verifier` use `fullcheck.protocol`, `.transformer`, `.quantize`, `.plans`, `.analytic` as attributes) |
| `analytic` | 213 | Decoder weight-op shapes without building the model; expected multiproof size; proof and setup bytes from shapes | OpShape, decoder_shapes, decoder_claim_columns, expected_multiproof_nodes, expected_lookup_nodes, proof_bytes, setup_size (`fc/analytic.py:27`) | Src `plans.py:109`. Run: `bench.py:66,788`, `perf.py:247`, `v1_bytes.py:156,181,341`, `gen_bench.py:191`, `serve_ab.py:101`, `setup_timing.py:64`, `params_table.py:29`, `setup_bytes.py:19`. 20 experiment files, T=8 | yes (all LLM tiers; paper tables via `params_table`/`setup_bytes`) |
| `attention_kernels` | 117 | Fused exact causal integer attention on the GPU (Triton) | available, attention_core | Src `transformer.py:39`. T=1 | yes (GPU prover and GPU verifier; paper `tex/ext_evaluation.tex:72-76`) |
| `buckets` | 290 | V1 verifier's cut products L·e, W·e and Xᵀe "by value": per-window tables (default), scatter buckets, old-path switch | mode, enabled, lw_times, xbar, bucket_sums, bucket_times, table_times, window_tables, SPAN (`fc/buckets.py:60`) | Src `cut_protocol.py:35`, `logup.py:106,134`, `protocol.py:415`. Run `v1_bytes.py:433`. T=1 | yes (V1 in the v1/cpuv/gpuv/real tiers; `tex/ext_v1.tex:96-98`) |
| `claimcodec` | 2199 | PVC3 compact lossless proof encoding: host, device, chunked and fused encoders; host, torch and device decoders; 31-bit field packing; lookup rows | encode, decode, decode_torch, decode_device(_ok), narrow, pack/unpack_field, pack/unpack_rows, field_size, MAGIC, BMAX, FIELD_BITS, errors (`fc/claimcodec.py:101`) | Src `protocol.py:67`, `cut_protocol.py:35`, `analytic.py:24`, `extfield.py:278,285`, `opcount.py:41`. Run `serve_ab.py:100`, `v1_bytes.py:261`. 9 experiment files, T=7 | yes (`--wire on` everywhere; paper names `claimcodec.decode`/`decode_torch`/`decode_device`, `tex/ext_proofs.tex:570`) |
| `codec_kernels` | 190 | Triton kernels for the codec: fused encoder (opt-in) and device decoder unpack | available, SegmentTable, pack_stream, high_parts, unpack_stream (`fc/codec_kernels.py:46`) | Src `claimcodec.py:1555,1563,2010,2016,2047`, `attention_kernels.py:27` (borrows `triton`). T=1 | yes (`decode_device` in gpuv). Fused-encoder half: opt-in only |
| `commitment` | 715 | Merkle trees and multiproofs; Ligero-style Reed–Solomon weight commitment (row, transposed, group); lookup-table commitment | MerkleTree, multiproof(_size), verify_multiproof(s), WeightCommitment, TransposedCommitment, GroupCommitment, TableCommitment, … (`fc/commitment.py:47`) | Src `protocol.py:68`, `plans.py:110`, `analytic.py:25`, `claimcodec.py:99`, `cut_protocol.py:40`, `reference.py:16`, `sampling.py:41`, `setup_cache.py:134,145,156`, `setup_proof.py:45`. T=8 | yes (mode C) |
| `contention` | 157 | Measurement only: run-queue wait, loadavg and rusage snapshots around timed phases | begin, add, snapshot, … | Src `protocol.py:67`, `cut_protocol.py:35`. Run `bench.py:65`, `serve_ab.py:100`. T=1 | yes (every timed record) |
| `cut` | 375 | V1 cut plan: which ops are cut, windows, GKR instances, split and counts | CutOp, CutPlan, lo, windows, shift, requant_values, split, split_counts, public_lw(_f64), delta_range, instance_bits, LMAX, CLAIM_LIMIT (`fc/cut.py:40`) | Src `protocol.py:75`, `cut_protocol.py:36`, `buckets.py:85,99`. Run `v1_bytes.py:262`. T=3 | yes (V1) |
| `cut_protocol` | 651 | V1 query: messages M1–M4, prover and verifier checks | run_cut_query, cut_proof_bytes, cut_plan, pack/unpack_exceptions, EXC_MAGIC (`fc/cut_protocol.py:45`) | Run `v1_bytes.py:206,263,340`, `ppl_llama.py:482`. T=5 | yes (V1) |
| `datasets` | 121 | MNIST, CIFAR-10 and ImageNet binary subsets, with caches | load_dataset, normalise, DATA_ROOT, CACHE_ROOT (`fc/datasets.py:30`) | Run `bench.py:441`; `train.py:20` | yes (bench tier image cells, `sp.sh:424-438`) |
| `extfield` | 289 | F_{p^8} arithmetic on torch | add/mul/inv/inv_many/eq_table/…/pack/unpack (`fc/extfield.py:21`) | Src `buckets.py:57`, `cut.py:36`, `cut_protocol.py:37`, `gkr_triton.py:30`, `logup.py:22`, `logup_gkr.py:27`. T=7 | yes (V1) |
| `field` | 488 | BabyBear arithmetic, NTT, RS encoding, exact modular matmuls (limbs, numpy small operands, int8 GEMM) | P, to_field, rs_encode, field_matmul_mod, small_matmul_mod, int8_* , … (`fc/field.py:26`) | Src: 14 modules (`graph.py:40`, `commitment.py:44`, `protocol.py:72`, …). Run `bench.py:67`, `params_table.py:30`. T=16 | yes |
| `gkr_triton` | 1306 | Generated Triton prover for V1's fractional-sum GKR (about 970 lines of kernels) | available, prove, prove_leaves | Src `cut_protocol.py:406`. T=2 | yes (V1 prover on GPU; `tex/ext_v1.tex:98-103`) |
| `graph` | 525 | Integer computation graph (MatOp/CheapOp/IntGraph): exact semantics, digest, requant helpers | Op, MatOp, CheapOp, IntGraph, exact_matmul, int8_product, requant(_fn), residual_add, … (`fc/graph.py:42`) | Src: `__init__`, cut, protocol, quantize, real_llama, real_weights, reference, sampling, transformer. Run `bench.py:68`, `opt_builder.py:45`, `ppl_2048.py:47`. T=11 | yes |
| `hostmem` | 35 | Pinned host memory within the driver's 2 GiB-per-allocation limit | (no `__all__`) PIN_MAX, … | Src `claimcodec.py:98`, `pipeline.py:22`, `protocol.py:67`. T=1 | yes (GPU paths) |
| `logup` | 151 | Windowed logUp: table, multiplicities, leaves, indicator, final identity | `fc/logup.py:26` | Src `cut.py:37`, `cut_protocol.py:38`. T=4 | yes (V1) |
| `logup_gkr` | 357 | Eager reference prover and verifier for the fractional-sum GKR | Transcript, prove, verify, verify_rounds, … | Src `cut_protocol.py:39`, `logup.py:23`, `gkr_triton.py:1152`. T=5 | yes (V1 verifier; CPU/eager prover) |
| `models` | 153 | Float CNN architectures and MODEL_SPECS | LeNet5, VGG, ResNet18, build_float_model, MODEL_SPECS | Src `quantize.py:27`. Run `bench.py:442`, `ab_verifier.py:102` (dynamic `sub(pkg,"models")`). T=5 | yes (bench CNN cells; perf CNN cells) |
| `native_kernels` | 1324 | C++/OpenMP CPU kernels (C++ source string at `fc/native_kernels.py:33-888`) with ctypes wrappers | available, attention_core(_hybrid), requant, …, field_matmul, unpack_stream, int8_matmul, lw_affine, xbar, tune_malloc (`fc/native_kernels.py:29`) | Src `field.py:74`, `graph.py:39`, `transformer.py:39`, `protocol.py:67`, `claimcodec.py:433,444,1763`, `buckets.py:177,280`. Run `reexec.py:224`. T=6 | yes (cpuv; RX int variant; `tex/evaluation.tex:238-241`) |
| `opcount` | 57 | Counts aten launches and codec device waits | VIEWS, KernelOps, codec_waits | Src: none. Experiments: `exp/6_improvements/codec_ab.py:59` only. T=1 (`test_claimcodec`) | **no** |
| `pipeline` | 130 | Wire formats and claim uploads of the streaming GPU verifier | AHEAD, ClaimUploads, wire_claim, wire_openings, wire_rows | Src `protocol.py:77`. Experiment `gpu_ab.py:46`. T=4 | yes (gpuv, `--extra '{"stream": true}'`, `sp.sh:263,272`) |
| `plans` | 480 | Mode-C commitment plans: policies, exact t, shared trees, layouts, auto | `fc/plans.py:113` | Src `protocol.py:78`. Run `bench.py:69`, `setup_timing.py:65`, `params_table.py:31`, `setup_bytes.py:20`. T=6 | yes (`--policy cnn18c`, `sp.sh:66`) |
| `protocol` | 2055 | Security parameters, commit_graph (+ setup cache), Challenger (Fiat–Shamir transcript), Prover, Verifier (C/K/Kpre, streaming), messages, run_query | `fc/protocol.py:80` | Src `__init__`, `cut_protocol.py:42`, `setup_proof.py:146`. 28 experiment files. T=33 | yes |
| `quantize` | 243 | Float CNN to int8 IntGraph | quantize_model, quantize_input, dequantize_logits, CNNBuilder, fold_bn | Run `bench.py:477`, `perf.py:226`, `ab_verifier.py:105,123`. T=5 | yes (bench CNN cells) |
| `real_llama` | 935 | Llama/Qwen3 integer decoders from HF checkpoints (MAIN_RECIPE) plus quality diagnostics | `fc/real_llama.py:82` | Run `v1_bytes.py:113`, `ppl_llama.py:71`. T=4 | yes (real tier Llama with `PVI_LLAMA_LICENCE`, `sp.sh:339-352`; quality tier) |
| `real_weights` | 335 | OPT integer decoders from HF checkpoints (CERTIFIED_RECIPE) | build_opt_from_hf, is_pruned, matches_benchmark_graph, real_logits | Run `v1_bytes.py:116`, `ppl_2048.py:46`, `real_weights_ppl.py:25`. T=5 | yes (real tier OPT, `sp.sh:311-312`; quality table) |
| `reference` | 278 | Straightforward reference code for tests ("Nothing in the protocol calls it", `fc/reference.py:9`) | — | Tests only (T=7, including smoke `T/test_gpu_verifier.py:33`) | tests only |
| `sampling` | 396 | Anchuri et al.'s path test on the same IntGraphs; `paths_for` | TraceCommitment, sample_path, shared_path_bytes, visit_probabilities, paths_for, neuron_tensors | Run `bench.py:478` (also private `_base`), `real_llm.py:45`. Experiment `5_comparison/analytic.py:16`. T=1 | yes (path counts in the paper; bench CNN "sampling" cell, `bench.py:553-570`) |
| `setup_cache` | 165 | Prover-side cache of the commitment's Merkle leaf digests | folder, key, save, read, check_* | Src `protocol.py:67`. T=1 | yes (`PVI_SETUP_CACHE` exported for every job, `sp.sh:77-79`) |
| `setup_proof` | 240 | One-time proximity proof for an untrusted commitment | SetupParams, SetupProof, setup_params, prove_setup, verify_setup, … | Run `setup_timing.py:63`, `params_table.py:28`, `setup_bytes.py:18`. T=1 | yes (setup tier) |
| `token_sampler` | 102 | Exact integer Gumbel-max sampler with the client's seed | TokenSampler, gumbel_table, sample_tokens | Run `gen_bench.py:110,193`. T=3 | yes (gen tier) |
| `transformer` | 585 | Random-weight integer decoders (GPT-2/OPT/Llama/Qwen); attention dispatch | DecoderConfig, CONFIGS, build_decoder, decoder_param_count, with_lm_positions, greedy_tokens | Src `real_llama.py:77`, `real_weights.py:46`, `reference.py:19` (all import privates). 35 experiment files. T=27 | yes |

---

## 2. Dead code

### 2a. Dead or misplaced modules
- **`opcount`** (`fc/opcount.py`): nothing in the run package or the paper scripts uses it. Its only users are `exp/6_improvements/codec_ab.py:59`, an A/B tool that needs a `--base` checkout (`exp/6_improvements/README.md:16-18`), and `T/test_claimcodec.py:28`. It patches `claimcodec._d2h` and `_nonzero_dev` in place (`fc/opcount.py:36-57`). **Move it to `code/tests/`.**
- **`reference`** is test-only by design (`fc/reference.py:9`). The other reference implementation already lives in `T/claimcodec_reference.py`, so the two have different homes. **Move it to `code/tests/`.** The smoke-tier tests import it, so keep it importable from the tests.

### 2b. Functions never referenced anywhere
These were checked by AST and grep across `code/` and `scripts/`.
- `buckets.window_tables` (`fc/buckets.py:83`): listed in `__all__` (`fc/buckets.py:60`), called nowhere.
- `extfield.neg` (`fc/extfield.py:52`): no call anywhere.
- `gkr_triton._eq_table` (`fc/gkr_triton.py:1094`): never called. It was replaced by `_suffix_tables` (`fc/gkr_triton.py:1102`). This is generated code, so it must also be removed in the generator (`exp/8_v1/gen_gkr_triton.py:422`).

### 2c. Functions used only by tests (no library or experiment caller)

| Function | Line | Note |
|---|---|---|
| `extfield.inv_fermat` | `fc/extfield.py:117` | `T/test_extfield.py:67` |
| `extfield.eq_eval` | `fc/extfield.py:168` | `T/test_extfield.py:81,86` |
| `extfield.batch_inv` | `fc/extfield.py:122` | Replaced by `inv_many` (`fc/extfield.py:139-140`, "V-V1N part 3"). Now only the reference in `T/test_v1_buckets.py:436` |
| `logup.multiplicities` | `fc/logup.py:46` | Production uses `cut.split_counts` (`fc/cut.py:328`, called at `fc/cut_protocol.py:252`) |
| `logup.prove_instances` | `fc/logup.py:121` | `T/test_logup.py:67` |
| `logup_gkr.verify_rounds` | `fc/logup_gkr.py:326` | `T/test_logup_gkr.py:72` |
| `setup_proof.query_group_columns` | `fc/setup_proof.py:137` | `T/test_setup_proof.py:48` |
| `transformer._residual` | `fc/transformer.py:423` | `T/test_cut.py:122`, `T/test_fast_verifier.py:555-568` |
| `reference.norm_int` | `fc/reference.py:196` | By design |

No class method was found with zero references outside tests. The AST pass matches names only, so names that collide can hide dead code.

### 2d. Code reached only when a switch is set off its default

| Code | Lines | Enabled by | In paper? |
|---|---|---|---|
| `native_kernels.attention_core_hybrid`, plus the C++ `pvi_softmax_block` | `fc/native_kernels.py:1033-1059`; C++ at `fc/native_kernels.py:354` | `PVI_CPU_ATTN=hybrid` (`fc/transformer.py:47,325,348`). Users: `exp/6_improvements/cpu_attn_bench.py:64`, `T/test_gpu_verifier.py:196` | no |
| `native_kernels.attention_profile`, plus C++ profiling counters | `fc/native_kernels.py:1210`; C++ at `fc/native_kernels.py:121,312` | Used only by `cpu_attn_bench.py:73-75` | no |
| `buckets.bucket_sums` / `bucket_times` and the scatter branches | `fc/buckets.py:143-169,183-187,224-229,270-277` | `PVI_V1_BUCKETS=scatter` | no (paper describes only the per-window tables, `tex/ext_v1.tex:97-98`) |
| Fused Triton encoder `claimcodec._fused_ok` / `_encode_fused` and `codec_kernels.SegmentTable` / `pack_stream` / `high_parts` | `fc/claimcodec.py:1543-1608`, `fc/codec_kernels.py:56-137,171-190` | `PVI_FUSED_CODEC=1` | **yes**, as a negative result (`tex/ext_evaluation.tex:94-95`): keep as a non-default option |
| Merkle worker-process verification | `fc/commitment.py:76-83,209-278` | `PVI_MERKLE_PROCESSES=1` | no (shown only in `exp/6_improvements/README.md:137,170`) |
| GKR phase profiler | `fc/gkr_triton.py:1043-1058` | `PVI_GKR_PROFILE=1` (`exp/8_v1/gen_gkr_triton.py`) | no; measurement |

---

## 3. Superseded code paths still present

| Path | Where | Default? | Paper / package | Recommendation |
|---|---|---|---|---|
| **Legacy weight-cache key** (the course report's re-upload bug) | `fc/graph.py:58-61,275-276`; checked in `exp/4_defence_benchmark/bench.py:1076-1078`; set only by the `report-must` `_nofix` cells (`strong_gpu.sh:233-248`) | off | The `report-must` tier is "not in the S&P package" (`strong_gpu.sh:19-20,206-208`); no paper mention | **Remove** (course-only) |
| `policy="paper"` (one tree per op at rate 4) | Default of `commit_graph` (`fc/protocol.py:220`; "the default everywhere", `fc/plans.py:57`). Plan-None branches at `fc/protocol.py:263-265,286-296`, `fc/setup_cache.py:72-73`, `fc/analytic.py` | **yes** (library) | Paper compares against it ("instead of 66 in each of five trees", `tex/ext_defence.tex:66-67`) | Keep as an option; **not** the default |
| `auto` policy and AUTO_CANDIDATES | `fc/plans.py:80-86,121,420-440` | default of `v1_bytes`, `gen_bench`, `setup_timing` | "six candidate plans" (`tex/ext_defence.tex:65`); final runs use cnn18c (`tex/ext_defence.tex:73-74`, `sp.sh:66`) | Keep as an option. Note that auto picks cnn16c/cnn17c for 7B/13B (`sp.sh:28-29`), so a run without `--policy` reproduces neither the paper nor the package |
| Non-`c` policies (`tight`, `cnn<e>`, `R<rate>`) | Same parser and code path (`fc/plans.py:243-269`) | — | not in the paper | No separate code; just document |
| V1 "today's" verifier path (`PVI_V1_BUCKETS=0`) | `fc/cut_protocol.py:594-610` (`_lw_e_old`), `fc/cut_protocol.py:175,360,626-635`; `Challenger._ext_draw` for one element (`fc/protocol.py:412-418`) | off | "2.49 s before" (`tex/ext_v1.tex:186`); measured by `--v1-paths 0 1` (`sp.sh:169,190-192`) | Keep as an option. `_lw_e_old` **cannot be removed**: it is also the fallback for spans past `SPAN` (`fc/cut_protocol.py:616-617`) |
| V1 scatter buckets | §2d | off | no | Remove |
| CPU "hybrid" attention | §2d | off | no | Remove (or keep as an undocumented option) |
| Torch fallbacks of every native kernel ("earlier v0 verifier") | `fc/native_kernels.py:13`; torch code in `transformer`, `protocol`, `claimcodec` | used when no compiler or on GPU | `tex/evaluation.tex:172` "earlier v0 verifier" | Keep (needed on GPU and hosts without a compiler) |
| Host claim decoder on a GPU client | `fc/protocol.py:532,1777` | off | GPU decoder row of `tex/ext_evaluation.tex:75` | Keep. The host decoder is the CPU path anyway |
| Eager GKR prover on GPU | `fc/cut_protocol.py:50,405-407` | off | Triton transcripts "byte-identical to those of a reference prover" (`tex/ext_v1.tex:103`) | Keep (`logup_gkr.prove` is the reference and the CPU path) |
| Float64 verifier fold | `fc/protocol.py:862-880` | off | not determined whether a paper number depends on it | Keep. It is the fallback for non-int8 weights (`fc/protocol.py:874-875`) |
| `PVI_INT32_CLAIMS` switch | `fc/native_kernels.py:1066,1079-1081` | on | never set by any script, test or doc | Remove the switch |
| Interactive-only machinery | `Challenger` interactive mode (`fc/protocol.py:320-338`; `seed` "tests only", `:331`); `cut_protocol._Recorder` / `_Replayer` (`fc/cut_protocol.py:122-153`) | interactive is the library default | `:int` cells remain for the zkLLM/Maverick comparisons (`sp.sh:46-48`) | Keep |
| Squirrel subset (Anchuri et al.'s M~) | `fc/datasets.py:7,42,95,119`; `fc/models.py:140`; used optionally at `bench.py:686-687`; trained at `train.py:34` | — | no hit in `paper_sp2027/sections` | Course-only: remove, unless the bench tier's records need it (not determined) |
| `analytic.expected_multiproof_nodes`'s course-report role ("only the earliest run raw/") | `fc/analytic.py:9-11`; `exp/5_comparison/aggregate.py:147` | — | `plans.py:109` uses it for the current plans | Keep the function; drop the raw/ narrative |
| Real-weight builder options outside the paper's recipes | `glu_exp`, `head_eq`, `pct_glu`, … (`fc/real_llama.py:616-622`); legacy OPT recipe (`exp/9_quality/digest_check.py:3-6`) | off / not the recipe | `glu_exp` appears only as a possible improvement that "none of our cost tables uses" (`tex/ext_discussion.tex:69-72`) | Keep non-default. This matches the memory rule on int8 changes |

---

## 4. Duplicated logic

1. **Three decoder builders.** `transformer._Builder` (`fc/transformer.py:428-494`), `real_weights._RealBuilder` (`fc/real_weights.py:56-163`) and `real_llama._Builder` (`fc/real_llama.py:176-443`) share these methods:
   - `name`: identical in all three (similarity ratio 1.00).
   - `cheap` and `requant`: ratio 0.96–1.00.
   - `_run`: ratio 0.92–0.93, except 0.24 between the OPT and Llama builders.
   - `residual`: 0.97 between the OPT and Llama builders.
   - `mat`: 0.88 between the OPT and Llama builders.

   A fourth copy sits on the run-package path: `exp/9_quality/opt_builder.py:1` ("A copy of `pvi.fullcheck.real_weights.build_opt_from_hf` with quality options"), imported by `ppl_llama.py:68`, which `v1_bytes.py:70` loads.
2. **`matches_benchmark_graph` and `real_logits`** are implemented twice: `fc/real_weights.py:291,313` and `fc/real_llama.py:840,909`.
3. **Triton bootstrap.** The `PVI_PY_INCLUDE` patch of `sysconfig.get_paths` is duplicated at `fc/codec_kernels.py:35-38` and `fc/gkr_triton.py:33-36`. The try-import of triton is duplicated at `fc/codec_kernels.py:40-44` and `fc/gkr_triton.py:38-42`. `attention_kernels` gets triton from `codec_kernels` (`fc/attention_kernels.py:27`). Three identical `available(device)` functions exist (`fc/codec_kernels.py:51`, `fc/attention_kernels.py:35`, `fc/gkr_triton.py:54`).
4. **Exact attention implementations, all bit-identical by test.** They are:
   - torch int32 `_attention_core` (`fc/transformer.py:267`)
   - int64 `_attention_int64` (`:293`)
   - Triton (`attention_kernels`)
   - native tiled, AVX-512 fused-requant and hybrid (`native_kernels`)
   - plus `reference.attention_heads`

   Only hybrid is neither a default nor a fallback.
5. **V1 final-check products.**
   - Old path: `cut.public_lw_f64` combined with `cut_protocol._lw_e_old`.
   - Current path: `buckets.table_times` with native `lw_affine`.
   - Scatter path: `bucket_sums` / `bucket_times`.
   - Xᵀe likewise has a float path, `buckets.xbar` and `native_kernels.xbar`.
6. **Superseded twins kept for tests.** `logup.multiplicities` vs `cut.split_counts`; `extfield.batch_inv` vs `inv_many`; `Challenger._ext_draw` vs `_ext_one` (`fc/protocol.py:408-449`).
7. **Claim encoders and decoders.** Four encoders give the same bytes: host, device, chunked and fused (`fc/claimcodec.py:910,1041,1478,1561`). Three decoders exist (`fc/claimcodec.py:1972,1983,2044`). Most are needed; the fused encoder is opt-in.
8. **Decoder shapes in two places.** `analytic.decoder_shapes` (`fc/analytic.py:49`) restates the architecture that `transformer.build_decoder` builds. This is deliberate and tested (`T/test_fullcheck.py:534`).
9. **Merkle trees.** `pvi.commitments.merkle` (266 lines, used by the course reproduction in `pvi.protocol`) is separate from `fc/commitment.py:146` (outside my scope; noted only).
10. **Numpy duplicates of torch routines for small operands** (`fc/field.py:290-322`, `fc/commitment.py:328-339`). These are deliberate.

---

## 5. Splits that follow development order rather than function

- **`protocol.py` (2,055 lines) holds six concerns:**
  - parameters (`fc/protocol.py:103-188`)
  - commit orchestration and setup cache (`:191-318`)
  - the Fiat–Shamir transcript `Challenger` (`:320-466`)
  - `Prover` (`:473-530`)
  - an 821-line `Verifier` (`:755-1576`), including the streaming verifier (`:1467-1595`)
  - messages and `run_query` (`:1622-2055`)

  It also carries V1 pieces: `SecurityParams.cut`/`cut_lmax` (`:111-114`), `params_for(cut=)` (`:132-150`), `_CUT_NARROW` (`:98`), `Verifier._cut_plan` (`:776`), `Challenger.ext` (`:408-449`) and `from .cut import delta_range` (`:75`). `cut_protocol` imports 10 private helpers from it (`fc/cut_protocol.py:42`).
- **The streaming verifier is split** between `pipeline.py` (wire formats) and `protocol.Verifier.verify_streaming` / `_streamed` / `_StartedColumns` / `_run_streaming` (`fc/protocol.py:1467-1595,2028-2055`).
- **`buckets.py`** is named after the abandoned bucket algorithm. Its docstring opens with a plan item ("V-V1N parts 1-2", `fc/buckets.py:1`), and its default is the table products.
- **`codec_kernels.py`** mixes the opt-in fused encoder with the default device decoder. `attention_kernels` depends on it only for the triton import.
- **`real_weights.py` (OPT only) and `real_llama.py` (Llama and Qwen3)** are parallel modules that both import private helpers from `transformer`: `_attention`, `_isqrt`, `_last_position`, `_lut`, `_norm_int` and `_rope` (`fc/real_weights.py:46`, `fc/real_llama.py:77`). The experiments do the same (`opt_builder.py:46`, `llama_attrib.py:73`).
- **Other cross-module private imports:**
  - `cut.py:38` imports `graph._canonical`
  - `commitment.py:44` and `protocol.py:72` import `field._LIMB_MAX`
  - `setup_cache.py:156` imports `commitment._padding_leaf`
  - `reference.py:19` imports `transformer._EXP_TABLE` and `_rope_tables`
  - `bench.py:478` imports `sampling._base`
- **`native_kernels.py`** packs C++ for unrelated phases (attention, cheap ops, field products, decode, int8 GEMM, V1 products) into one source string (`fc/native_kernels.py:33-888`). The built library is still named `pvi_attention_<key>.so` (`fc/native_kernels.py:908`), and `PVI_NATIVE_ATTN=0` turns off **all** native kernels, not just attention (`fc/native_kernels.py:901`; `_library()` returns None).
- **`gkr_triton.py` is generated** (`fc/gkr_triton.py:3`), but its generator lives in `exp/8_v1/gen_gkr_triton.py`.
- **Stale documentation:**
  - `fc/__init__.py:1-30` lists 14 of 33 modules and calls `real_weights` "the report's int8 perplexities, Sec. 4.1" (`:26-27`).
  - `code/src/pvi/README.md:11` cites course sections ("Sec. 3.4–3.6, 4") and omits the V1, kernels, setup and real_llama modules.
  - Docstrings cite `V1_SPEC.md`, which is not tracked in the repository (`fc/cut.py:1`, `fc/cut_protocol.py:1`, `fc/logup.py:1,13`, `fc/logup_gkr.py:1,145`, `fc/gkr_triton.py:1`, `fc/graph.py:156`, `fc/protocol.py:141`), and `sp2027_research/v1_design/scratch/spec_toy.py`, which is outside the repository (`fc/logup_gkr.py:1-2`).

---

## 6. Proposed functional grouping

The subpackages `v0/` and `v1/` use the paper's own names.

| Current module | Proposed location | Notes |
|---|---|---|
| `field`, `extfield` | `arith/field.py`, `arith/extfield.py` | |
| `graph` | `ir/graph.py` | **digest hazard** (H1 below) |
| `commitment`, `plans`, `setup_cache`, `setup_proof`, `analytic` | `commit/` (`analytic` as `commit/cost.py`) | |
| `protocol` | `v0/` split into `params.py`, `transcript.py` (Challenger), `prover.py`, `verifier.py`, `query.py` (run_query and messages); `v0/__init__` re-exports | Turn the shared privates `cut_protocol` uses into a small internal API |
| `pipeline` | `v0/streaming.py` (with `verify_streaming`) | |
| `claimcodec`, `token_sampler` | `v0/claimcodec.py`, `v0/token_sampler.py` | Paper cites `claimcodec.decode*` (`tex/ext_proofs.tex:570`): keep that name |
| `cut`, `cut_protocol`, `logup`, `logup_gkr`, `gkr_triton`, `buckets` | `v1/` (`cut_protocol` as `v1/query.py`, `buckets` as `v1/products.py`) | Move `gen_gkr_triton.py` to `scripts/` |
| `native_kernels`, `attention_kernels`, `codec_kernels`, `hostmem` | `kernels/` (`native.py` plus a `native.cpp` data file; `attention_triton.py`; `codec_triton.py`; a new `_triton.py` with one triton import and the `PVI_PY_INCLUDE` handling) | Moving the C++ out keeps its bytes, so the build key (`fc/native_kernels.py:905`) is unchanged |
| `transformer`, `real_weights`, `real_llama` | `decoders/` (`random.py` or keep `transformer.py`; `hf_opt.py`; `hf_llama.py`) | digest hazard. Do **not** name it `models/`: `ab_verifier.py:102` imports `pkg.models` as the CNN module |
| `models`, `quantize`, `datasets` | `cnn/` (`nets.py`, `quantize.py`, `datasets.py`) | `datasets` defaults use `Path(__file__).parents[3]` (`fc/datasets.py:32-33`), which breaks at a new depth |
| `sampling` | `baseline/path_test.py` | |
| `contention` | `measure/contention.py` | |
| `reference`, `opcount` | `code/tests/` | |

**Import strategy:**
- `pvi.fullcheck.__init__` keeps the public API, and each subpackage's `__init__` re-exports its own.
- Update the roughly 90 importer files mechanically rather than leaving shim modules. Re-export shims break the tests' and experiments' patching of private globals: 6 patches of `cc._FUSED`, 8 each of `_CHUNK` and `_ONE_PASS`, `nk._LW`, `native_kernels._DECODE`/`_CHEAP`/`_GEMM`, and 11 of `proto._defer`. One of these is on the run-package path: `exp/7_reexec/reexec.py:224-236` patches `native_kernels._GEMM` (and `encode_prof.py:50,80` patches `claimcodec._FUSED` and `_CHUNK`). A patch on a shim changes nothing in the real module.
- `ab_verifier.load_base` imports an **old** checkout with the flat layout by flat names (`ab_verifier.py:44-56,102-129`), and `perf.py`'s CNN cells use it (`perf.py:162,217-224`). Either keep the flat names as attributes or teach `ab_verifier` both layouts.

**Hazards to settle before moving anything:**
- **H1, graph digest (blocking).** `_canonical` hashes every cheap-op function as `f"{__module__}.{__qualname__}"` (`fc/graph.py:383`) and every dataclass by `__qualname__` (`fc/graph.py:388`). The cheap ops are lambdas and closures in `graph.requant_fn` (`fc/graph.py:154-176`), `transformer.build_decoder` (e.g. `fc/transformer.py:493,522-555`), `real_weights`, `real_llama`, `quantize` and the experiment `opt_builder`. So moving these 5 library modules, or even refactoring them into new helpers, changes every graph digest. That has four consequences:
  - Every Fiat–Shamir transcript changes, because the digest is absorbed first (`fc/protocol.py:786,1624`).
  - In mode C the drawn columns change, and multiproof size depends on which columns are drawn (`fc/commitment.py:167-175`), so FS proof bytes can change.
  - Stored `graph_digest` fields stop matching (`v1_bytes.py:371`, `digest_check.py:56`, `ppl_2048.py:377`, `ppl_llama.py:369`, `T/test_real_weights.py:370`).
  - The setup-cache key is unaffected (`fc/setup_cache.py:67-85`), and seeded interactive goldens are unaffected.

  **Decision needed** before golden outputs are captured: either keep these modules' dotted paths fixed, or switch to a stable, layout-independent name for each function (a new `pvi/graph/v2` tag, `fc/graph.py:440`) and capture goldens afterwards. This also bears on the paper: it describes the digest as "every operation in order with its inputs, output and constants" (`tex/defence.tex:189-191`), but the implementation also binds Python identifiers, so an independent implementation cannot reproduce the digest. Either document this or remove it.
- **H2.** The paper cites code paths and strings: `code/src/pvi/fullcheck` (`tex/ext_proofs.tex:43`), the tests `test_pruning.py`, `test_fs_tree.py` and `test_claimcodec.py` (`tex/ext_proofs.tex:108,214,529,609`), and the transcript label `pvi/fullcheck/v3` (`tex/ext_proofs.tex:375`, `fc/protocol.py:336`). That label is a protocol constant and must not follow a rename.

**Lower-risk alternative:** keep the five graph-building modules (`graph`, `transformer`, `real_weights`, `real_llama`, `quantize`) at their current paths and regroup the rest. Delete the dead and course-only code, move `reference` and `opcount` to the tests, and split `protocol.py`. Functions in `protocol` are not hashed into graphs.

---

## 7. Environment variables read by the library

Plus the standard `CXX`. "Import" means the value is read once at import time.

| Variable | Read at | Default | Effect | Set by | Recommendation |
|---|---|---|---|---|---|
| `PVI_SETUP_CACHE` | `fc/setup_cache.py:43,54`, at call time when `cache=None` | unset (no cache) | Setup-cache folder | `sp.sh:77-79` for every job; `T/test_setup_cache.py` | Explicit `--setup-cache DIR` on each script, passed as `commit_graph(cache=)`. Drop the hidden env fallback |
| `PVI_V1_BUCKETS` | `fc/buckets.py:63,73`, at every call | `1` | V1 verifier path `0`/`1`/`scatter`. Also changes `Challenger.ext` (`fc/protocol.py:415`), narrowing (`fc/cut_protocol.py:360`) and xbar (`:175`) | Set in-process by `v1_bytes.py` (`--v1-paths`, used in the package, `sp.sh:169`) and `v1_phases.py`; `T/test_v1_buckets.py` | Explicit `v1_path` on the Verifier or `run_cut_query`. Keep `0`, drop `scatter` |
| `PVI_NATIVE_GEMM` | `fc/native_kernels.py:1264`, import | on | Native int8 GEMM in `MatOp.compute` on CPU | Not set via env; **patched as `native_kernels._GEMM`** by `reexec.py:224-236` (RX `int` vs `int_fp32gemm`, `sp.sh:135`) | **Explicit argument** (forward or options) |
| `PVI_NATIVE_ATTN` | `fc/native_kernels.py:901`, once per process | on | **Disables the whole native library**, despite the name | docs only | Explicit `--native on/off`; rename |
| `CXX`, `PVI_NATIVE_DIR` | `fc/native_kernels.py:903,907`, once | `g++`, `~/.cache/pvi` | Compiler and build cache | `sp.sh:82-84` | Platform settings: env is acceptable |
| `PVI_NATIVE_AVX512`, `PVI_NATIVE_CHEAP`, `PVI_NATIVE_DECODE`, `PVI_V1_NATIVE` | `fc/native_kernels.py:995,1065,1175,1281`, import | on | Individual native kernels | docs; tests patch the globals | One options object (below) |
| `PVI_INT32_CLAIMS` | `fc/native_kernels.py:1066`, import | on | CPU verifier keeps claims as int32 | nothing | Remove |
| `PVI_MALLOC_TUNE` | `fc/native_kernels.py:1250`, call (via `tune_malloc`, from `protocol`) | on | glibc mallopt for the whole process | docs only | Explicit verifier or script argument |
| `PVI_FUSED_ATTN` | `fc/transformer.py:46`, import | on | Triton attention vs torch | docs only | Options object |
| `PVI_CPU_ATTN` | `fc/transformer.py:47`, import | `tiled` | `hybrid` CPU attention | docs only | Remove with hybrid |
| `PVI_GPU_DECODE` | `fc/protocol.py:532`, import | on | GPU vs host claim decoder | docs only | `Verifier(gpu_decode=)` |
| `PVI_FOLD_FAST` | `fc/protocol.py:869`, at every fold | on | Native or int8 vs float64 verifier fold | `fold_ab.py:46`; `T/test_fold_fast.py` | Verifier argument |
| `PVI_HASH_THREADS` | `fc/protocol.py:129`, call | torch threads | Threads for tree absorption | `T/test_fs_tree.py` | `Challenger(hash_threads=)` |
| `PVI_GKR_TRITON` | `fc/cut_protocol.py:50`, import | on | Triton vs eager GKR prover | nothing | Explicit `gkr="triton"/"eager"` |
| `PVI_FUSED_CODEC` | `fc/claimcodec.py:1543`, import | off | Fused encoder | patched by `encode_prof.py:50` and tests | `encode(impl="fused")` (`encode` already has `impl`, `fc/claimcodec.py:787-807`) |
| `PVI_MERKLE_PROCESSES` | `fc/commitment.py:76`, import | off | Multiproof checks in worker processes | README example only | Remove |
| `PVI_GKR_PROFILE` | `fc/gkr_triton.py:1043`, import | off | Phase timings | `gen_gkr_triton.py` | `prove(..., profile=)`, or drop |
| `PVI_PY_INCLUDE` | `fc/codec_kernels.py:35`, `fc/gkr_triton.py:33`, import; **patches `sysconfig.get_paths` for the whole process** | unset | Host-header workaround for Triton | `sp.sh:81`; `docs/improvements/I_server_setup.md` | Platform setting: env is acceptable, but put it in one place (`kernels/_triton.py`) |
| `PVI_CACHE`, `PVI_CIFAR_ROOT`, `PVI_IMAGENET_ROOT` | `fc/datasets.py:33-35`, import | paths under `code/` | Dataset locations | `bench` README; `sp.sh:430-431` | `--data-root` / `--cache-root` on `bench.py` and `train.py`, with env as fallback |
| `PVI_LEGACY_WEIGHT_KEY` | `fc/graph.py:61`, import | off | Course-report bug | `strong_gpu.sh` report-must only | Remove |

For the kernel and verifier switches, a single frozen options object (e.g. `fullcheck.options`) gives every switch a CLI argument with little plumbing. Scripts set it from argparse; modules read it at call time instead of import time. Its defaults would equal today's, so outputs stay the same.

---

## 8. High-level points for the review

1. **Library defaults are the course report's configuration**: policy `paper` (`fc/protocol.py:220`), `lean=False` and `lookups=False` (`fc/protocol.py:761,766`), `wire=False` (`fc/protocol.py:1909`), interactive challenges, `prune_last=False` (`fc/transformer.py:496-497`). The real-weight builders default to `norm_gain=32` and no smoothing (`fc/real_weights.py:182-185`; `fc/real_llama.py:616-622`), which differ from `CERTIFIED_RECIPE` (`fc/real_weights.py:174`) and `MAIN_RECIPE` (`fc/real_llama.py:458`). The paper's settings exist only as flags in `sp.sh:132-136`. I suggest a named paper preset in the library, used as the CLI defaults, with the report configuration kept as a named non-default preset.
2. **The scripts' default policies disagree**:
   - `paper`: `bench.py:1015`, `perf.py:177`, `serve_ab.py:867`
   - `auto`: `v1_bytes.py:307`, `gen_bench.py:174`, `setup_timing.py:52`
   - `cnn18c`: what the paper and package use (`tex/ext_defence.tex:73-74`, `sp.sh:66`)
3. **The graph digest binds Python identifiers** (H1 in §6), which the paper's description does not say.
4. **`gkr_triton` is generated**; any edit must go through the generator.

## 9. What I could not determine

- Whether any stored record besides the `graph_digest` fields depends on the digest, and whether goldens will be captured under Fiat–Shamir.
- Whether the bench CNN "sampling" cell's records feed the S&P tables. `sp2027_numbers.py` has no "sampling" reference; the LLM path counts come from `real_llm.py`.
- For each row of `tab:eval-improve`: whether it was measured with env switches or with the `--base` checkout. The README says mostly the latter (`exp/6_improvements/README.md:16-18`), so most of these switches are probably not needed to reproduce the table.
- Whether a paper number depends on the `PVI_FOLD_FAST` verifier fold, or on the squirrel model.
- Reference counts match names; collisions (e.g. `available`, `mode`, `pack`) can only hide dead code, never invent it. Dynamic `getattr` and `sub(pkg, name)` loads were found by reading the code and may not all be listed.
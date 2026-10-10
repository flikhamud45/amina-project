# IEEE S&P 2027 plan (fallback: USENIX Security 2027, cycle 2)

Target: **S&P 2027, last cycle: abstract 10 Nov 2026, paper 17 Nov 2026** (13 pages + 5 of references and
appendices; IEEE format; reviewed as submitted; a reject blocks S&P for a year). Fallbacks: EuroS&P 2 Dec 2026,
**USENIX Security 26 Jan 2027**. **Gate G0, Fri 30 Oct**: go to S&P only if the defence targets marked (G0) below
are met; otherwise USENIX, continuing with the proof-size track.

Status tags: **[DONE]** finished and verified (commit or file named), **[WIP]** in progress, **[TODO]**,
**[USER]** needs the team (a decision or an action only they can take), **[DROP]** decided against (reason given).
**Every [DONE] improvement has a detailed write-up in `docs/improvements/`** (problem, idea, why it is exact or
sound, implementation, tests, measurements, what remains); the item links to it.
Reviewer objections (from the mock review, `S/sp2027_research/idea_mock-review.md`): **D1** verifier not cheaper
than re-execution / weights leak; **D2** thin novelty; **D3** GB proofs; **D4** attack weak (contrived, worst
case, no real aligned model); **D5** int8 quality, no real-weight LLM defence run; **D6** honest commitment, no
proximity test; **D7** no disclosure, crowded field. Each item says which it answers.

Code: branch `sp2027` (GitHub). Research reports and measurements: `S/sp2027_research/` (roadmap.md,
results_2026-10-09.md, raw records, server logs), S = session 4aa2c028's scratchpad. Server: my folder
`logs/validation/sp2027/` (lean clones by `sp_setup.sh`).

---

## A. Prover (already the strongest axis; keep it unbeatable, fix long prompts)

- **A1 [DONE]** Column opening and folding on exact int8 GEMMs (e2bcb31; [doc](docs/improvements/A1_int8_column_opening.md)). 2080 Ti: opening 9.7x, prover 3.7-4.3x on Llama-2-7B/13B blocks, bit-identical.
- **A2 [DONE]** Forward-pass weight products on exact int8 GEMMs (9ce4305; [doc](docs/improvements/A2_int8_forward.md)). About 2x on Llama-sized layers.
- **A3 [DONE]** Claim encoder (29db2ce, 228ae2d, 71e8920, aa7e485; [doc](docs/improvements/A3_claim_encoder.md)). The profile showed the cost was assembling the result (one 2 GB `b"".join`), not the kernels: threaded assembly into one read-only buffer gives the encoder 2.1-2.3x on large proofs (OPT-1.3B @2048: 3.28 -> 1.54 s; Llama-2-7B 2 blocks: 0.542 -> 0.238 s), identical bytes. The fused Triton encoder is byte-identical but slower (1.74 s; fixed ~0.2 s per proof), kept opt-in (`PVI_FUSED_CODEC=1`) as a negative result; its one advantage is bounded device memory (183 MB vs ~1 GB at 885 M claims).
- **A4 [DONE]** The prover's attention runs B3's fused Triton kernel (shared `_attention_heads`; [doc](docs/improvements/B2_B3_gpu_verifier.md) Sec. 6): `prove_forward` at 2,048 tokens 1.16x (Llama-2-7B 2 blocks), 1.15x (Llama-2-13B 1 block), 1.62x (OPT-1.3B 12 blocks), job 1005350. The int8-GEMM alternative is not needed.
- **A5 [TODO]** Overlap encoding with the forward pass (side stream, per op), once A3 exists.
- **A6 [TODO]** Kpre precompute (`Verifier._fold_local`) on int8 GEMMs (setup time).

## B. Verifier at long prompts (D1, D3; future work "native verifier", "check attention")

- **B1 [DONE]** Profile of the streaming GPU verifier (`gpu_profile.py`, job 1005394; [doc](docs/improvements/B2_B3_gpu_verifier.md) Sec. 7): copies 32% of CUDA time (proof upload, lean claims), fused attention 12%, element-wise ~25%, int8 GEMMs 4%. Data movement dominates; V1 halves it.
- **B2 [DONE] (G0)** GPU decoder for the claims (2a71a79; [doc](docs/improvements/B2_B3_gpu_verifier.md)): byte-exact, every malformed input rejected as on the host, offsets validated on the host before launch. Validated on the 2080 Ti (job 1005162: GPU decoder 2/2, GPU verifier 287 passed, exactness 77, generation and setup 40, wire 118, no failure). With B3: streaming GPU verifier Kpre @2048, OPT-1.3B 12 blocks 1.43 -> 0.53 s (2.7x), Llama-2-7B 2 blocks 0.36 -> 0.18 s. PVC4 (per-op frames) not needed so far.
- **B3 [DONE] (G0)** Fused exact integer attention kernel (Triton; 2a71a79; same doc): three passes per block of query rows, causal blocks skipped, fp16 inputs with fp32 sums exact below 2^24; bit-identical to `_attention_core` (tested on the GPU).
- **B4 [TODO]** Element-wise derive on the GPU fused (`torch.compile` now works on the GPU nodes, see infra I2) or Triton.
- **B5 [WIP]** CPU verifier ([doc](docs/improvements/B5_cpu_verifier.md)). Native C++/OpenMP kernels for every phase, each exact (tests on the node, fallbacks elsewhere): tiled attention (e501882, 7606e8a), hybrid BLAS attention (c36cfa8; slower, a negative result), the recomputed operations (requantisation, residual, norm, RoPE, tables; 1e88df8), the Freivalds field products (b7ef9a9), the claim decoder's slot streams (11d227f), AVX-512 attention (4719b71), int32 claims end to end (d473455). Llama-2-7B Kpre @2048, Xeon 4114, 8 threads, per full block: torch 3.27 s, tiled attention 2.23 s, + recomputed ops 1.69 s (jobs 1005733, 1005765); the rest in jobs 1005787/1005788/1005801. Target: the full model in <= 20 s (from ~85 s; fp32 re-execution 86 s).
- **B6 [TODO]** Report verifier / re-execution ratios everywhere (with F1), and position the GPU verifier as an auditor/gateway.
- **Targets (G0):** GPU verifier OPT-1.3B @2048 <= 0.8 s (zkLLM 0.90 s, ours 2.6 s); Llama-2-7B Kpre @2048 <= 1.5 s (4.95 s).

## C. Proof size (D3; future work "sum-check for the non-weight operations")

- **C1 [DONE]** V1 design spec (`S/sp2027_research/v1_design/V1_SPEC.md`, executable toy `scratch/spec_toy.py`): send the int8 requantised values, prove each claim lies in its requantisation window by a windowed logUp-GKR over F_{p^8} reduced to the existing fold and column check (no new commitment); soundness theorem with integer multiplicities (Lemma 1); byte model on the real shapes: 2.14-2.37x smaller at 2,048 tokens (1.8-2.0x with real weights); 15-day milestone plan with kill criteria K1-K5.
- **C2 [WIP] (G0)** V1 ([doc](docs/improvements/C2_v1_proof_size.md)): implemented end to end (modes C and Kpre, interactive and Fiat-Shamir; T0-T9, K1), adversarially reviewed (5 findings, 4 fixed, 1 documented), GPU GKR prover (Triton, byte-identical transcripts, 15.6 ns per leaf at 2^25 on the 2080 Ti). Bytes at 2,048 tokens: 2.0-2.3x smaller (full OPT-1.3B 2.07 -> 0.96 GB). Cost on the 2080 Ti: prover +3-4 s per ~2*10^8 cut claims (the GKR), CPU verifier ~1.6x v0. Next: L40S timings (lmax 27, incl. Llama mode C), real-weight byte profile (smoothed OPT graphs), streaming GPU verifier with the cut, a few multipliers per op (F2's row scales).
- **C3 [TODO]** V2 (commit everything but attention; the verifier recomputes attention only; 6-8x smaller at 2,048 tokens, verifier 2.5-4x faster) and V3 (everything proved; MB proofs) as an analysed design and cost model in the paper (fixes from the red-team: three folds, the norm gadget's sums split into 8-bit limbs, extension-field challenges).
- **C4 [TODO]** Sessions: one opening for B queries (`bench.py --batches`) and a deferred chi per session; Llama-2-7B 64-token query 324.5 -> ~205 MB at B = 16.
- **C5 [TODO]** Codec: real-weight claim entropy (E1 in the roadmap) and any cheap gain (per-op Rice parameters, better centring).

## D. Trust and rigor (D2, D6)

- **D1 [DONE]** Fiat-Shamir absorbs `IntGraph.digest()` (every op and cheap-op constant) (2dad363; [doc](docs/improvements/D1_D2_fiat_shamir_binding.md)).
- **D2 [DONE]** Transcript in SHAKE-256, challenges read from labelled copies (2dad363; same doc).
- **D3 [DROP]** Canonical claim encoding. Not needed for soundness: under Fiat-Shamir a prover that re-encodes the same claims only makes more hash queries, which the bound Q 2^-beta with Q <= 2^64 (the grinding bits in `params_for`) already covers; non-canonical encodings make proofs malleable but cannot make a false statement pass. One sentence in D7's Fiat-Shamir paragraph instead.
- **D4 [DONE] (G0)** ([doc](docs/improvements/D4_D5_untrusted_commitment.md), af2ccf8; the theorem text for the appendix is still to write) Untrusted commitment: "Fact 1" (r base-field vectors checked at the same columns = one combination over F_{p^r}: proximity error n/p^r = 2^-137 at r = 5), a one-time public setup proximity proof (Option B) implemented and tested, per-query column counts from the untrusted-commitment bound, a parameter table (setup proof size vs per-query bytes), the theorem and proof (appendix).
- **D5 [DONE]** (same doc; measured: passes ~2^-t of the queries without the setup proof, rejected with it) The split-commitment attack as a test: without D4 a commitment mixing two models passes with probability 2^-t_g per query (2^-35 for Llama-2-7B's smallest tree); with D4 it is rejected at setup.
- **D6 [TODO]** Open-weight registry (Option C): publish the int8 checkpoint and its hash; anyone recomputes C_M.
- **D7 [TODO]** Writing: threat model; exact integer semantics (requant rounding, isqrt, exp table, RoPE tables, causal mask; cross-platform determinism); output definition; the weight leak (claims reveal a weight matrix after about ceil(k/T) queries) and why mode C targets open weights; Kpre verdict leakage; Fiat-Shamir/grinding and bit-security table; a full proof for the optimised protocol (shared trees, transposed layout, lookups, pruning, codec, generation).

## E. Features (D1, D3; future work "KV cache")

- **E1 [DONE]** A generated response proved as one prefill with the token rule (03ae13d; `bench.py --gen`, cbb8c05; [doc](docs/improvements/E1_generation.md)). 2080 Ti: GPT-2, 64-token prompt + 256 tokens: prover 0.29 s, verifier 0.59 s, 87 MB (0.34 MB per token).
- **E2 [DONE]** Sampled generation with a client-chosen seed (7c98d87; [doc](docs/improvements/E2_sampling.md)): an exact integer Gumbel-max sampler (public 16-bit table with a fixed SHA-256, temperature as an integer logit weight, top-k), the seed absorbed into the statement, the token rule checking the sampler's choices; works with v0 (all modes, streaming) and V1. Stop rules need no proof; top-p not implemented.
- **E3 [TODO]** Verifier KV cache for multi-turn chat (reuse K/V of accepted turns only); (n+1)/2 fewer claims for n turns.

## F. Evaluation (D1, D5, D7)

- **F1 [WIP]** Re-execution baseline (`experiments/7_reexec/reexec.py`, 9a40bf2; [doc](docs/improvements/F1_reexecution_baseline.md)): measured on the 2080 Ti node; re-run on the L40S and the EPYC with the final verifier.
- **F2 [WIP] (G0)** Integer-model quality ([doc](docs/improvements/F2_quality.md)). Attribution (OPT-125M): the LayerNorm outputs' rounding dominates (OPT's outlier channels), then weights and fc1's requantisation; softmax, residual and ReLU negligible. Fix adopted (e34bbf0, opt-in `smooth=`): SmoothQuant through the norms' per-channel gains, scalar multipliers, ops and shapes unchanged (v0 and V1 accept it): OPT-125M 75.5 -> 66.6 (fp32 64.8), OPT-1.3B 36.4 -> 33.7 (fp32 32.8, x1.03), both reproduced with the committed builder. Per-row power-of-two weight scales reach x1.00-1.01 on OPT-125M but need V1 support for a few multipliers per op (deferred). Next: OPT-6.7B and Qwen3-4B [USER: memory/disk: friend's node or quota], real-weight V1 bytes.
- **F3 [TODO] (G0)** A real-weight end-to-end defence run (Qwen3-4B and OPT-6.7B real weights): prover, verifier, proof bytes (claims' entropy differs from random weights), generation.
- **F4 [TODO]** Memory accounting: prover and verifier peak GPU/host memory per cell (bench records `gpu_peak_memory`, `host_peak_rss` already).
- **F5 [TODO]** Same-hardware baseline: zkLLM (and DeepProve or EZKL if feasible) on the friend's L40S [USER: the friend's machine].
- **F6 [WIP]** Final L40S run package: draft `docs/L40S_RUN_SP2027.md` and `slurm/sp2027.sh` (tiers smoke, v1, quality, gpuv; with `strong_gpu.sh must/should` for v0 with the improvements). To freeze the commit and hand over once V1/F2 are final [USER: run on the friend's machine].
- **F7 [TODO]** Tables: verify/re-exec, proof/weights, transfer time at 1/10/100 Gb/s, per-token costs for generation, comparison with zkAgent / zkLLM / DeepProve / ZKTorch / Maverick / CommitLLM.
- **F8 [TODO]** Modern shapes: add Llama-3-8B (GQA, 128k vocabulary) and Mistral-7B configs to the benchmark.

## G. The attack and its generalisation (D4, D7)

- **G1 [WIP] (G0)** Dichotomy theorem: drafted and independently checked (workflow `sp2027-attack-theory`; LaTeX for Sec. 3.3 and a new appendix in `S/sp2027_research/attack_theory/PART1_SP.md`, 18 new BibTeX entries, those marked `% [U]` to check on dblp); to merge into the S&P draft (H1): a verifier that spot-checks q of N sites of an unencoded trace accepts a single-site edit with probability >= 1 - q/N for any (adaptive, value-aware) sampler; encoding/distance amplification (our defence) reaches 2^-lambda. Proof, checked by an independent pass (workflow).
- **G2 [WIP] (G0)** Systematisation table: drafted and source-checked, 9 rows, numbers from `attack_theory/sok_formulas.py` (same file); to merge (H1) of 5+ published schemes (Anchuri et al. SaTML 2026, SLP, CommitLLM routine audit, TensorCommitments, SPEX, NanoZK audit mode; Proof-of-Learning as precedent), each acceptance probability reproduced by a script from the scheme's own equations; say "broken" only where a scheme claims more than its sampling gives.
- **G3 [TODO]** Adaptive game against the deployed value-aware sampler with range-respecting edits (measure detection vs edit size on real networks).
- **G3.5 [DONE] (OPT-1.3B; OPT-6.7B in F6)** In-range edits on a real LLM ([doc](docs/improvements/G3_inrange_attack.md), `experiments/2_attack/inrange_llm.py`): against per-neuron ranges calibrated on held-out WikiText-2 train text, edits that keep every value in range change OPT-1.3B's next token for all 40 prompts at layers 6/12/18/23 (median 2 neurons, max 60; uniform detection <= 7.9e-4 per path); down-only edits, invisible to contribution weighting (detection 0), succeed for 9-39 of 40 prompts. In the paper (Sec. 3.5, intro, discussion, ext table).
- **G4 [TODO]** Real aligned-model threat characterisation (Qwen3 instruct): success rate vs edit size for a generation-level change that passes the path protocol; reported as measurements, not a recipe.
- **G5 [USER] (G0)** (drafts and a dated schedule in `attack_theory/PART1_SP.md`: Thu 15 Oct Anchuri et al. and TensorCommitments, Mon 19 Oct CommitLLM, NanoZK, SLP, SPEX; reminders 26 Oct) Responsible disclosure to Anchuri et al. and the authors of the schemes in G2 by Mon 19 Oct (I draft the notices; the team sends them). Ethics paragraph.

## Team decisions needed (from the attack-theory workflow and earlier)

- **[USER]** Who sends the disclosure notices (default in the drafts: Ofek from his TAU address, Erel and Edo in CC); whether SPEX is notified and whether Ambient appears anywhere (default: no).
- **[USER]** Whether to run the Qwen3-4B repeat of the real-LLM attack (G4), and GPU access for OPT-6.7B (13.4 GB, does not fit the 2080 Ti).
- **[USER]** Anonymisation of the submission (author block, the "Final Report" subtitle, the Contributions section).
- **[USER]** Gated models (Llama-3-8B) for F2/F8: licence access on Hugging Face (never the friend's token).

## H. The paper

- **H1 [WIP]** S&P draft in `paper_sp2027/` (IEEEtran compsoc, anonymous). First draft: 29 pages, mock review `paper_sp2027/REVIEW.md` (scores 2/3/2). Condensation (2026-10-10, `paper_sp2027/CUT_PLAN.md`, workflows `paper-cut-sp2027` and `paper-cut2-sp2027`, commit 63f5ec3): one body, two builds: `main.tex` (submission: body + `appendix_short.tex`, target 13 + 5 pages) and `main_ext.tex` (extended version for the anonymous artifact repository: full appendices + `sections/ext_*.tex`); review changes 2, 6, 8, 9, 10 applied (positioning, verifier-class Definition, concrete-security table, root of trust, client-options table, BCIKS20 Thm. 1.7 and Ligero Lemma 4.2 checked). 19 pages after the first pass; second pass running. Next: B5 numbers into Sec. 7.4 and the abstract, then the numbers from F6.
- **H2 [TODO]** Novelty statement (D2): the first Freivalds-based inference check whose client needs no enclave, no stored encoded weights and no weight-derived secret, fully sound including attention, with a provider 2-3 orders of magnitude faster than any zkSNARK prover; positioned as a different trade-off from zkLLM (not "worse ZK").
- **H3 [TODO]** Number pipeline (paper_assets / text_numbers) extended to the new tables and the new raw roots.
- **H4 [TODO]** Artifact appendix and README (S&P strongly encourages artifacts).
- **H5 [WIP]** Internal mock review of the S&P draft (workflow, three reviewer lenses) and fixes, a week before the deadline. A first mock review of the first draft is `paper_sp2027/REVIEW.md`.
- **H6 [USER]** Register the abstract by 10 Nov; submit by 17 Nov.

## I. Infrastructure

- **I1 [DONE]** Lean server clones (sparse checkout, 66 MB); exact int8 GEMMs confirmed on the 2080 Ti (sm_75) ([doc](docs/improvements/I_server_setup.md)).
- **I2 [DONE]** Triton and `torch.compile` on the GPU nodes, with copied Python headers (`logs/validation/sp2027/tri.py`, `pyinclude/`; same doc).
- **I3 [DONE]** Freed server quota: my 22 old validation clones archived to `Desktop/server_backup_2026-10-09/validation_clones.tgz` (930 MB, listing verified), then deleted on the server: 19.5 -> 15.7 GB used.
- **I4 [DONE]** C++ toolchain on the compute nodes: conda-forge `gxx` 13.4 via micromamba in `logs/validation/sp2027/tools/cxx` (951 MB); used by `torch.compile` on the CPU and by `native_kernels` (`CXX`, `PVI_NATIVE_DIR`) ([doc](docs/improvements/I_server_setup.md)).

---

## Log

- 2026-10-10 (morning): first S&P draft (paper_sp2027/, 29 pages, builds) and mock review; F6 run package drafted; V1 real-weight bytes (OPT-125M smoothed: 1.68x / 1.77x).
- 2026-10-10 (night): V1 adversarial review fixed; V1 timings final on the 2080 Ti (prover 18.4 -> 4.9 s, verifier 8.5 -> 4.9 s at OPT-125M @2048); GKR bottom-layer path; F2 attribution and smoothing (OPT-125M x1.17 -> x1.03, OPT-1.3B x1.11 -> x1.03); E2 sampling; A4, B1 done.
- 2026-10-10 (later): V1 end to end (modes C, Kpre; T4, T5, T7), Triton GKR prover validated on the 2080 Ti, V1 bytes measured at 2,048 tokens (2.0-2.3x); B5 native attention tiled and documented; F2 workflow started; local venv: transformers, pyarrow installed.
- 2026-10-10: A3 done (threaded assembly 2.1-2.3x; fused encoder a negative result); B2, B3 validated and done; B5 native attention measured (1.51x); I4 done; C1 done; C2 days 1-4 (F_{p^8}, GKR, logUp with K1 passed, cut plan and windows). Pushed `sp2027`.
- 2026-10-09: research workflow (8 agents) -> roadmap; A1, A2, E1, D1, D2 done; F1 measured on the 2080 Ti; I1, I2 done. Pushed to GitHub `sp2027` (2e91856).
- 2026-10-09 (night): A3 validated on the 2080 Ti (encoder tests, wire and generation tests with it); D4, D5 done; B2, B3 written; I3 done; attack-theory workflow finished (G1, G2 drafts, G5 drafts and schedule).
- 2026-10-09: plan written; detailed docs for every done item (`docs/improvements/`); workflows started for C1 (V1 design) and G1-G5 (attack theory, systematisation, experiments, disclosure drafts); A3 started.

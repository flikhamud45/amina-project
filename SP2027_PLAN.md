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
- **A3 [WIP] (G0)** Fused Triton claim encoder, byte-identical `PVC3`. Encoding is 17.9 of 24.5 s for Llama-2-13B at 2,048 tokens (L40S); profile: GPU kernels ~1 s, ~12k launches and syncs ~2 s, copy-back 0.34 s for OPT-1.3B. Target: encode within 2x of the copy-back floor; prover at 2,048 tokens 3-8x faster.
- **A4 [TODO]** Attention of the prover's forward on int8 GEMMs (QK^T exact in int32; PV with p <= 255 split into two int8 halves) or a fused Triton kernel shared with B3.
- **A5 [TODO]** Overlap encoding with the forward pass (side stream, per op), once A3 exists.
- **A6 [TODO]** Kpre precompute (`Verifier._fold_local`) on int8 GEMMs (setup time).

## B. Verifier at long prompts (D1, D3; future work "native verifier", "check attention")

- **B1 [TODO]** Profile the streaming GPU verifier (`torch.profiler`): split of the checks into decode / attention / element-wise / Freivalds / copies.
- **B2 [WIP] (G0)** (2a71a79; GPU validation job 1005074) GPU decoder for the claims (byte-exact; every malformed input rejected as on the host; offsets validated on the host before launch). Removes the host decode (59-73% of GPU verifier time). Decide on the way whether a per-op framed format (PVC4) is needed (streaming, 13B on 11-12 GB cards).
- **B3 [WIP] (G0)** (2a71a79; GPU validation job 1005074) Fused exact integer attention kernel (Triton): scores, row max, exp table, row sum, p, PV in one pass per block, causal blocks skipped; bit-identical to `_attention_core`. Shared by prover and verifier.
- **B4 [TODO]** Element-wise derive on the GPU fused (`torch.compile` now works on the GPU nodes, see infra I2) or Triton.
- **B5 [TODO]** CPU verifier: a C++ toolchain (install g++ in my folder or on the laptop) for inductor / C++ kernels of the derive and attention; the cheap fixes (causal blocking 1.3-1.4x, int32 derive, thread balance). Target: Llama-2-7B Kpre at 2,048 tokens from ~85 s to <= 20 s on 8 cores.
- **B6 [TODO]** Report verifier / re-execution ratios everywhere (with F1), and position the GPU verifier as an auditor/gateway.
- **Targets (G0):** GPU verifier OPT-1.3B @2048 <= 0.8 s (zkLLM 0.90 s, ours 2.6 s); Llama-2-7B Kpre @2048 <= 1.5 s (4.95 s).

## C. Proof size (D3; future work "sum-check for the non-weight operations")

- **C1 [WIP]** V1 design spec (workflow `sp2027-v1-proof-size-design`, output `S/sp2027_research/v1_design/V1_SPEC.md`): send the int8 requantised activations in the clear, commit only the residues, range-check them (logUp-GKR over an extension field, or bit decomposition in the code-based commitment); Freivalds against committed values; soundness proof; byte model on the real shapes; implementation plan in this code base. Expected 2.2-2.4x smaller at 2,048 tokens.
- **C2 [TODO]** V1 implementation and measurement (if C1's effort estimate fits; else a prototype on small models plus the cost model).
- **C3 [TODO]** V2 (commit everything but attention; the verifier recomputes attention only; 6-8x smaller at 2,048 tokens, verifier 2.5-4x faster) and V3 (everything proved; MB proofs) as an analysed design and cost model in the paper (fixes from the red-team: three folds, the norm gadget's sums split into 8-bit limbs, extension-field challenges).
- **C4 [TODO]** Sessions: one opening for B queries (`bench.py --batches`) and a deferred chi per session; Llama-2-7B 64-token query 324.5 -> ~205 MB at B = 16.
- **C5 [TODO]** Codec: real-weight claim entropy (E1 in the roadmap) and any cheap gain (per-op Rice parameters, better centring).

## D. Trust and rigor (D2, D6)

- **D1 [DONE]** Fiat-Shamir absorbs `IntGraph.digest()` (every op and cheap-op constant) (2dad363; [doc](docs/improvements/D1_D2_fiat_shamir_binding.md)).
- **D2 [DONE]** Transcript in SHAKE-256, challenges read from labelled copies (2dad363; same doc).
- **D3 [TODO]** Canonical claim encoding: the decoder recomputes the encoder's plan and rejects other widths (no free re-encoding to grind Fiat-Shamir).
- **D4 [DONE] (G0)** ([doc](docs/improvements/D4_D5_untrusted_commitment.md), af2ccf8; the theorem text for the appendix is still to write) Untrusted commitment: "Fact 1" (r base-field vectors checked at the same columns = one combination over F_{p^r}: proximity error n/p^r = 2^-137 at r = 5), a one-time public setup proximity proof (Option B) implemented and tested, per-query column counts from the untrusted-commitment bound, a parameter table (setup proof size vs per-query bytes), the theorem and proof (appendix).
- **D5 [DONE]** (same doc; measured: passes ~2^-t of the queries without the setup proof, rejected with it) The split-commitment attack as a test: without D4 a commitment mixing two models passes with probability 2^-t_g per query (2^-35 for Llama-2-7B's smallest tree); with D4 it is rejected at setup.
- **D6 [TODO]** Open-weight registry (Option C): publish the int8 checkpoint and its hash; anyone recomputes C_M.
- **D7 [TODO]** Writing: threat model; exact integer semantics (requant rounding, isqrt, exp table, RoPE tables, causal mask; cross-platform determinism); output definition; the weight leak (claims reveal a weight matrix after about ceil(k/T) queries) and why mode C targets open weights; Kpre verdict leakage; Fiat-Shamir/grinding and bit-security table; a full proof for the optimised protocol (shared trees, transposed layout, lookups, pruning, codec, generation).

## E. Features (D1, D3; future work "KV cache")

- **E1 [DONE]** A generated response proved as one prefill with the token rule (03ae13d; `bench.py --gen`, cbb8c05; [doc](docs/improvements/E1_generation.md)). 2080 Ti: GPT-2, 64-token prompt + 256 tokens: prover 0.29 s, verifier 0.59 s, 87 MB (0.34 MB per token).
- **E2 [TODO]** Sampling with a verifier-chosen seed (temperature, top-k with an integer sampler); stop rules (EOS, length).
- **E3 [TODO]** Verifier KV cache for multi-turn chat (reuse K/V of accepted turns only); (n+1)/2 fewer claims for n turns.

## F. Evaluation (D1, D5, D7)

- **F1 [WIP]** Re-execution baseline (`experiments/7_reexec/reexec.py`, 9a40bf2; [doc](docs/improvements/F1_reexecution_baseline.md)): measured on the 2080 Ti node; re-run on the L40S and the EPYC with the final verifier.
- **F2 [TODO] (G0)** Integer-model quality: 16-bit softmax probabilities (verifier and prover), SmoothQuant-style smoothing of the weights, per-token activation scales if integer-only; perplexity on WikiText-2 (and C4) for OPT-125M/1.3B/6.7B, Qwen3-4B (open, Apache-2.0), Llama-3-8B if licence access is available [USER for gated models].
- **F3 [TODO] (G0)** A real-weight end-to-end defence run (Qwen3-4B and OPT-6.7B real weights): prover, verifier, proof bytes (claims' entropy differs from random weights), generation.
- **F4 [TODO]** Memory accounting: prover and verifier peak GPU/host memory per cell (bench records `gpu_peak_memory`, `host_peak_rss` already).
- **F5 [TODO]** Same-hardware baseline: zkLLM (and DeepProve or EZKL if feasible) on the friend's L40S [USER: the friend's machine].
- **F6 [TODO]** Final L40S run package (all cells with A-E on), as before: scripts, expected hours, a checklist for the friend [USER: run].
- **F7 [TODO]** Tables: verify/re-exec, proof/weights, transfer time at 1/10/100 Gb/s, per-token costs for generation, comparison with zkAgent / zkLLM / DeepProve / ZKTorch / Maverick / CommitLLM.
- **F8 [TODO]** Modern shapes: add Llama-3-8B (GQA, 128k vocabulary) and Mistral-7B configs to the benchmark.

## G. The attack and its generalisation (D4, D7)

- **G1 [WIP] (G0)** Dichotomy theorem: drafted and independently checked (workflow `sp2027-attack-theory`; LaTeX for Sec. 3.3 and a new appendix in `S/sp2027_research/attack_theory/PART1_SP.md`, 18 new BibTeX entries, those marked `% [U]` to check on dblp); to merge into the S&P draft (H1): a verifier that spot-checks q of N sites of an unencoded trace accepts a single-site edit with probability >= 1 - q/N for any (adaptive, value-aware) sampler; encoding/distance amplification (our defence) reaches 2^-lambda. Proof, checked by an independent pass (workflow).
- **G2 [WIP] (G0)** Systematisation table: drafted and source-checked, 9 rows, numbers from `attack_theory/sok_formulas.py` (same file); to merge (H1) of 5+ published schemes (Anchuri et al. SaTML 2026, SLP, CommitLLM routine audit, TensorCommitments, SPEX, NanoZK audit mode; Proof-of-Learning as precedent), each acceptance probability reproduced by a script from the scheme's own equations; say "broken" only where a scheme claims more than its sampling gives.
- **G3 [TODO]** Adaptive game against the deployed value-aware sampler with range-respecting edits (measure detection vs edit size on real networks).
- **G4 [TODO]** Real aligned-model threat characterisation (Qwen3 instruct): success rate vs edit size for a generation-level change that passes the path protocol; reported as measurements, not a recipe.
- **G5 [USER] (G0)** (drafts and a dated schedule in `attack_theory/PART1_SP.md`: Thu 15 Oct Anchuri et al. and TensorCommitments, Mon 19 Oct CommitLLM, NanoZK, SLP, SPEX; reminders 26 Oct) Responsible disclosure to Anchuri et al. and the authors of the schemes in G2 by Mon 19 Oct (I draft the notices; the team sends them). Ethics paragraph.

## Team decisions needed (from the attack-theory workflow and earlier)

- **[USER]** Who sends the disclosure notices (default in the drafts: Ofek from his TAU address, Erel and Edo in CC); whether SPEX is notified and whether Ambient appears anywhere (default: no).
- **[USER]** Whether to run the Qwen3-4B repeat of the real-LLM attack (G4), and GPU access for OPT-6.7B (13.4 GB, does not fit the 2080 Ti).
- **[USER]** Anonymisation of the submission (author block, the "Final Report" subtitle, the Contributions section).
- **[USER]** Gated models (Llama-3-8B) for F2/F8: licence access on Hugging Face (never the friend's token).

## H. The paper

- **H1 [TODO]** Port to the IEEE S&P template (13 pages + appendices); new structure: (1) the attack and the dichotomy, (2) the defence and its theorem with an untrusted commitment, (3) generation, (4) evaluation incl. re-execution, real weights, quality, memory, (5) systematisation and related work (2026 crowd: zkAgent, Maverick, CommitLLM, LAMP, SLP, DeepProve, zkLLM, ZKTorch).
- **H2 [TODO]** Novelty statement (D2): the first Freivalds-based inference check whose client needs no enclave, no stored encoded weights and no weight-derived secret, fully sound including attention, with a provider 2-3 orders of magnitude faster than any zkSNARK prover; positioned as a different trade-off from zkLLM (not "worse ZK").
- **H3 [TODO]** Number pipeline (paper_assets / text_numbers) extended to the new tables and the new raw roots.
- **H4 [TODO]** Artifact appendix and README (S&P strongly encourages artifacts).
- **H5 [TODO]** Internal mock review of the S&P draft (workflow, three reviewer lenses) and fixes, a week before the deadline.
- **H6 [USER]** Register the abstract by 10 Nov; submit by 17 Nov.

## I. Infrastructure

- **I1 [DONE]** Lean server clones (sparse checkout, 66 MB); exact int8 GEMMs confirmed on the 2080 Ti (sm_75) ([doc](docs/improvements/I_server_setup.md)).
- **I2 [DONE]** Triton and `torch.compile` on the GPU nodes, with copied Python headers (`logs/validation/sp2027/tri.py`, `pyinclude/`; same doc).
- **I3 [DONE]** Freed server quota: my 22 old validation clones archived to `Desktop/server_backup_2026-10-09/validation_clones.tgz` (930 MB, listing verified), then deleted on the server: 19.5 -> 15.7 GB used.
- **I4 [TODO]** A C++ toolchain for CPU inductor (B5): conda/micromamba `gxx` in my server folder, or the friend's EPYC node.

---

## Log

- 2026-10-09: research workflow (8 agents) -> roadmap; A1, A2, E1, D1, D2 done; F1 measured on the 2080 Ti; I1, I2 done. Pushed to GitHub `sp2027` (2e91856).
- 2026-10-09 (night): A3 validated on the 2080 Ti (encoder tests, wire and generation tests with it); D4, D5 done; B2, B3 written; I3 done; attack-theory workflow finished (G1, G2 drafts, G5 drafts and schedule).
- 2026-10-09: plan written; detailed docs for every done item (`docs/improvements/`); workflows started for C1 (V1 design) and G1-G5 (attack theory, systematisation, experiments, disclosure drafts); A3 started.

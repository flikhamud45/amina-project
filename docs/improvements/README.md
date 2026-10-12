# Improvements for the S&P 2027 version

One file per improvement, explaining the problem it addresses, the idea, why it is exact or sound, the
implementation (files and functions), the tests, the measurements and what remains. The plan with the status
of every item is `SP2027_PLAN.md` at the repository root; an item is tagged [DONE] only with its file here.

| Item | File | Status | In one line |
|---|---|---|---|
| A1 | [A1_int8_column_opening.md](A1_int8_column_opening.md) | done | the prover's opening and fold on exact int8 GEMMs: opening 9.7x, prover 3.7-4.3x on Llama blocks |
| A2 | [A2_int8_forward.md](A2_int8_forward.md) | done | the forward pass's weight products on exact int8 GEMMs: about 2x on Llama layers |
| A3 | [A3_claim_encoder.md](A3_claim_encoder.md) | done | the encoder's result assembled on threads: 2.1-2.3x on large proofs; the fused Triton encoder a negative result |
| A7 | [A7_serving_overhead.md](A7_serving_overhead.md) | 2080 Ti pilot measured; L40S `serve` tier pending | the prover against unverified serving (our integer model without proofs, eager fp16/bf16) on the same GPU and prompts, with batches; proofs on request are byte-identical |
| B2, B3 | [B2_B3_gpu_verifier.md](B2_B3_gpu_verifier.md) | done | claims decoded on the GPU and a fused exact attention kernel: streaming GPU verifier 2.7x at 2,048 tokens |
| B5 | [B5_cpu_verifier.md](B5_cpu_verifier.md) | in progress | a native exact attention for CPU verifiers: 1.5x on the verifier at 2,048 tokens |
| C4a | [C4a_batch_bytes.md](C4a_batch_bytes.md) | measured on the 2080 Ti (B = 1-8); L40S `batch` tier pending | one client's batch shares one fold and one opening: the byte model claims + 127.8/B MB holds within 3% on the full Llama-2-7B |
| C1, C2 | [C2_v1_proof_size.md](C2_v1_proof_size.md) | implemented, measuring | V1: send the int8 values, prove their windows with a logUp-GKR over F_{p^8}; no new commitment |
| E1 | [E1_generation.md](E1_generation.md) | done | a whole generated response proved as one prefill with a token rule: GPT-2, 256 tokens in 0.29 s, 0.34 MB per token |
| F2 | [F2_quality.md](F2_quality.md) | measured, adopted (opt-in) | where the integer model's perplexity goes; SmoothQuant through the norm gains: x1.17 -> x1.03 (OPT-125M), x1.11 -> x1.03 (OPT-1.3B) |
| F2b-Q, F2b-L (laptop part) | [F2b_Q_real_llama.md](F2b_Q_real_llama.md) | implemented, laptop-validated; real-checkpoint runs on the cluster | real-weight integer Qwen3 / Llama graphs with the benchmark graph's exact structure (empty diff); quality ladder script |
| E2 | [E2_sampling.md](E2_sampling.md) | done | sampled generation: an exact integer Gumbel-max sampler keyed by the client's seed |
| E2b | [E2b_sampler_tv.md](E2b_sampler_tv.md) | measured on OPT-125M and OPT-1.3B; table change is the team's call | the sampler's exact output law (ties included) and its TV distance to softmax sampling on real logits |
| D1, D2 | [D1_D2_fiat_shamir_binding.md](D1_D2_fiat_shamir_binding.md) | done | Fiat-Shamir binds the graph and its constants; SHAKE-256 transcript |
| D4, D5 | [D4_D5_untrusted_commitment.md](D4_D5_untrusted_commitment.md) | done | a one-time setup proof removes the honest-commitment assumption; the split-commitment attack it stops |
| F1 | [F1_reexecution_baseline.md](F1_reexecution_baseline.md) | measured on the 2080 Ti | the cost of re-running the model, against our verifier |
| F3a | [F3a_real_weight_runs.md](F3a_real_weight_runs.md) | done (2080 Ti/Xeon): OPT-1.3B and OPT-6.7B, real and random; V1 with the GKR at 2,048 on the L40S | `prune_last` in the real OPT builder (the unpruned graph's constants and logits); byte accounting, tampers and peak memory in `v1_bytes.py`/`perf.py --real`; OPT-6.7B codec-only profile |
| I1, I2, I4 | [I_server_setup.md](I_server_setup.md) | done | lean clones under the quota; Triton and torch.compile on the GPU nodes; a C++ compiler for CPU kernels |
| V-V1N (parts 1-4) | [V_V1N_fast_v1_verifier.md](V_V1N_fast_v1_verifier.md) | implemented, exact; Xeon: 2.49 -> 2.13 s per Llama-2-7B block; native kernels not done | V1's final check and xbar by value (affine-plus-step window tables, one-limb products; the bucket kernel as the native reference), batched GKR chains, batch-inverted logUp, int8 values: bit-identical, `PVI_V1_BUCKETS` |
| G3-min | [G3_min_adaptive_game.md](G3_min_adaptive_game.md) | done (MNIST, OPT-1.3B) | adaptive game, held-out ranges: the best implementable mixture detects at most 3.1% / 9.7% per path on the MNIST MLP (below 27.7/N); on OPT-1.3B, against the two fixed edits, contribution weighting crosses 27.7/d_ff at layers 6-18 |
| F2a | [F2a_quality_2048.md](F2a_quality_2048.md) | done; certified recipe fixed | quality at 2,048 tokens on the full test set: 11-bit probabilities; certified recipe smooth-max alpha 0.6 (x1.076 / x1.026 / x1.008 for OPT-125M / 1.3B / 6.7B) |
| F2c | [F2c_zero_shot.md](F2c_zero_shot.md) | harness done and tested; first cluster job submitted | zero-shot accuracy (LAMBADA, HellaSwag, PIQA, ARC-e/c, WinoGrande) of the integer OPT against fp16, lm-eval 0.4's formats re-implemented, paired intervals |
| A6 | [A6_kpre_precompute.md](A6_kpre_precompute.md) | done: Llama-2-7B precompute 106.9 -> 12.7 s (CPU), 4.3 -> 2.2 s (GPU) | the Kpre client's precompute `u = chi A` on the native field product (CPU) and int8 GEMMs (GPU), no transposed int64 copy; identical `u` |
| V-LONG-b | [V_LONG_b_long_prompts.md](V_LONG_b_long_prompts.md) | measured 2k-16k (2080 Ti/Xeon) | the CPU verifier against exact re-execution from 2k to 16k tokens: 5.1x to about 2x; proofs grow linearly (25.8 GB at 8k for Llama-2-7B) |

Measurement harnesses (all compare against the previous code path on the same queries and check identical
outputs): `experiments/6_improvements/open_ab.py` (A1), `forward_ab.py` (A2), `encode_prof.py` (A3),
`derive_ab.py` (B5), `setup_bytes.py` (D4), `experiments/8_v1/v1_phases.py` (V-V1N); `experiments/7_reexec/reexec.py` (F1), `serve_ab.py` (A7);
`experiments/4_defence_benchmark/bench.py --gen` (E1).

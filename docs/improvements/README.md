# Improvements for the S&P 2027 version

One file per improvement, explaining the problem it addresses, the idea, why it is exact or sound, the
implementation (files and functions), the tests, the measurements and what remains. The plan with the status
of every item is `SP2027_PLAN.md` at the repository root; an item is tagged [DONE] only with its file here.

| Item | File | Status | In one line |
|---|---|---|---|
| A1 | [A1_int8_column_opening.md](A1_int8_column_opening.md) | done | the prover's opening and fold on exact int8 GEMMs: opening 9.7x, prover 3.7-4.3x on Llama blocks |
| A2 | [A2_int8_forward.md](A2_int8_forward.md) | done | the forward pass's weight products on exact int8 GEMMs: about 2x on Llama layers |
| E1 | [E1_generation.md](E1_generation.md) | done | a whole generated response proved as one prefill with a token rule: GPT-2, 256 tokens in 0.29 s, 0.34 MB per token |
| D1, D2 | [D1_D2_fiat_shamir_binding.md](D1_D2_fiat_shamir_binding.md) | done | Fiat-Shamir binds the graph and its constants; SHAKE-256 transcript |
| F1 | [F1_reexecution_baseline.md](F1_reexecution_baseline.md) | measured on the 2080 Ti | the cost of re-running the model, against our verifier |
| I1, I2 | [I_server_setup.md](I_server_setup.md) | done | lean clones under the quota; Triton and torch.compile on the GPU nodes |

Measurement harnesses (all compare against the previous code path on the same queries and check identical
outputs): `experiments/6_improvements/open_ab.py` (A1), `forward_ab.py` (A2), `encode_prof.py` (A3),
`derive_ab.py` (B5); `experiments/7_reexec/reexec.py` (F1); `experiments/4_defence_benchmark/bench.py --gen`
(E1).

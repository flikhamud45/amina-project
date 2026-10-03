# `pvi`: the library

| Module | What it implements | Paper |
|---|---|---|
| `nn/` | a network as a layered DAG: parent sets `G_j`, execution traces, the local relation `a_j = φ(Σ w_ij a_i)`; dense and convolutional layers; gradients | Sec. 3.2 |
| `commitments/merkle.py` | Merkle vector commitment with position-bound openings | Sec. 3.2 |
| `protocol/` | the path protocol of Anchuri et al.: `RandPathTest` path sampling, prover, verifier, commitments to the model and the trace | Sec. 3.2, 4.2 |
| `attacks/` | the forgeries of Anchuri et al. (`baselines.py`), our single-neuron and stealthy tampering (`tamper.py`), and the trigger backdoor (`backdoor.py`) | Sec. 3.2, 4.2 |
| `defences/` | the alternative path samplers (`sampling.py`) and the attacker that adapts to them (`adaptive.py`) | Sec. 3.3, 4.2 |
| `experiments/analysis.py` | the exact probability that the path test accepts a given trace, under any sampler (a dynamic program over paths) | Sec. 4.2 |
| `fullcheck/` | **our protocol**, basic and optimised: see below | Sec. 3.4–3.6, 4 |
| `data.py`, `training.py`, `zoo.py`, `results.py` | MNIST loading, seeded training, the models and substitution settings, JSON output | |

`fullcheck/` (PyTorch; GPU optional):

| File | Contents |
|---|---|
| `field.py` | BabyBear field arithmetic, NTT and Reed–Solomon encoding, exact modular matrix products (on a GPU with int8 tensor cores, also as exact int8 GEMMs: `int8_*`) |
| `commitment.py` | Merkle trees with multiproofs; the weight commitment (rows encoded, Merkle root over columns) and its column openings; the transposed and grouped commitments of the plans, and lookup tables (a tree over an embedding's rows) |
| `graph.py` | integer computation graphs: weight operations (`MatOp`) and verifier-recomputed operations (`CheapOp`), exact GPU matrix products; the lean forward pass (dead tensors freed, claims streamed to the host); `PVI_LEGACY_WEIGHT_KEY=1` restores the weight cache of the earlier run `raw/`, for the benchmark's `_nofix` controls |
| `quantize.py`, `models.py`, `datasets.py` | float CNN → int8 graph (per-channel weights, BatchNorm folded); the benchmarked CNNs and their data |
| `transformer.py` | integer GPT-2/OPT/Llama/Qwen decoders (norms, GELU/SiLU tables, RoPE, integer softmax, causal attention); optionally the last block at the last position (`prune_last`) |
| `protocol.py` | security parameters from λ (`params_for`), prover, verifier (modes C, K, Kpre; interactive or Fiat–Shamir; `lean=True` for the language models; `device="cuda"` for a verifier on a GPU (Table 2's GPU column), with Merkle hashing on the CPU, and `stream=True` for its streaming checks; a plan's col layouts and lookup tables; in K and Kpre optionally the verifier's own embedding rows, `lookups`), `run_query` |
| `plans.py` | commitment plans for mode C (`policy=`): exact `t`, Merkle trees shared by matrices of one codeword length, per-op codeword lengths and layouts |
| `pipeline.py` | the wire formats and claim uploads of the streaming GPU verifier |
| `claimcodec.py` | the compact, lossless encoding of the proof (`run_query(wire=True)`) |
| `contention.py` | measurement only: the evidence that a timed phase ran on a quiet machine (this process's run-queue wait and on-CPU time from `/proc/self/task/*/schedstat`, context switches, the load average), which `protocol.py` and `bench.py` record as each timing row's `contention` field; it changes no timing, verdict or proof byte |
| `hostmem.py` | pinned host memory within what the driver grants: no single pinned allocation above 2 GiB (`PIN_MAX`; the L40S nodes' driver refuses more), pageable memory above that (the same values; copies to and from it wait for the device) |
| `reference.py`, `opcount.py` | the straightforward code the verifier's fast routines are tested against; counting launched work (tests and benchmarks only) |
| `sampling.py` | the original path test on the same integer graphs (the like-for-like baseline), with exact per-neuron detection |
| `analytic.py` | decoder weight-op shapes and claim columns, without building the model; the expected multiproof size (for the records of `raw/`, which sent one path per column) and its analogue for a lookup table's rows (`expected_lookup_nodes`); a query's proof bytes (`proof_bytes`) and the commitment's setup size (`setup_size`) from the shapes alone, under the basic protocol's parameters or a commitment plan, in the default form or with `--wire` (the setup budget of `plans.py`'s `auto`, `bench.py`'s setup records, `6_improvements/plan_bytes.py` and `wire.py`) |
| `real_weights.py` | integer OPT decoders built from real Hugging Face checkpoints (the same ops and shapes as `build_decoder`, LayerNorm folded into the next matrix), for the int8 perplexities of Sec. 4.1 (`experiments/2_attack/real_weights_ppl.py`) |

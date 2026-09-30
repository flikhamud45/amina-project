# `pvi`: the library

| Module | What it implements | Report |
|---|---|---|
| `nn/` | a network as a layered DAG: parent sets `G_j`, execution traces, the local relation `a_j = φ(Σ w_ij a_i)`; dense and convolutional layers; gradients | §3.2 |
| `commitments/merkle.py` | Merkle vector commitment with position-bound openings | §3.2 |
| `protocol/` | the original protocol: `RandPathTest` path sampling, prover, verifier, commitments to the model and the trace | §3.2, §4.2 |
| `attacks/` | the paper's own attacks (`baselines.py`), our single-neuron and stealthy tampering (`tamper.py`), and the trigger backdoor (`backdoor.py`) | §3.2, Table 1 |
| `defences/` | the alternative path samplers (`sampling.py`) and the attacker that adapts to them (`adaptive.py`) | §3.2, §4.2 |
| `experiments/analysis.py` | the exact probability that the path test accepts a given trace, under any sampler (a dynamic program over paths) | Table 1 |
| `fullcheck/` | **our defence**: see below | §3.3, §4.3–4.4 |
| `data.py`, `training.py`, `zoo.py`, `results.py` | MNIST loading, seeded training, the models and substitution settings, JSON output | |

`fullcheck/` (PyTorch; GPU optional):

| File | Contents |
|---|---|
| `field.py` | BabyBear field arithmetic, NTT and Reed–Solomon encoding, exact modular matrix products |
| `commitment.py` | Merkle trees with multiproofs; the weight commitment (rows encoded, Merkle root over columns) and its column openings; the transposed and grouped commitments of the plans, and lookup tables (a tree over an embedding's rows) |
| `graph.py` | integer computation graphs: weight operations (`MatOp`) and verifier-recomputed operations (`CheapOp`), exact GPU matrix products |
| `quantize.py`, `models.py`, `datasets.py` | float CNN → int8 graph (per-channel weights, BatchNorm folded); the benchmarked CNNs and their data |
| `transformer.py` | integer GPT-2/OPT/Llama/Qwen decoders (norms, GELU/SiLU tables, RoPE, integer softmax, causal attention); optionally the last block at the last position (`prune_last`) |
| `protocol.py` | security parameters from λ (`params_for`), prover, verifier (modes C, K, Kpre; interactive or Fiat–Shamir; a plan's col layouts and lookup tables; in K and Kpre optionally the verifier's own embedding rows, `lookups`), `run_query` |
| `sampling.py` | the original path test on the same integer graphs (the like-for-like baseline), with exact per-neuron detection |
| `analytic.py` | decoder weight-op shapes and the expected multiproof size (for runs made before multiproofs) |

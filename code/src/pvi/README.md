# `pvi`: the library

`pvi` implements both halves of the paper: the spot-checking verifiers that the attack breaks, and the fully sound
defence, protocol v0 and its variant V1. The experiment scripts (`code/experiments/`) and the tests (`code/tests/`)
import it; install it with `pip install -e .` from `code/`, or put `code/src` on `PYTHONPATH`. Each module's
docstring states its exact semantics, bounds and soundness arguments; this file describes the package layout.

**`pvi.core`: integer arithmetic and integer computation graphs.** `field` implements the BabyBear prime field
`p = 15 * 2**27 + 1`: reduction, the number-theoretic transform, Reed-Solomon encoding, and exact modular matrix
products as float64 limb GEMMs, as their numpy form for small CPU operands, or as int8 GEMMs on GPU tensor cores.
Every path returns the same canonical residues. `extfield` implements the degree-8 extension `F_p[x] / (x^8 - 11)`,
from which V1 draws its challenges. `graph` defines the certified integer model of Sec. `sec:intmodel` as an
ordered graph of weight operations (`MatOp`, a product with a committed matrix, computed exactly on any device) and
cheap operations (`CheapOp`, recomputed by the verifier). Each cheap operation is a registered kind with a semantics
version and explicit constants (`op_kind`, `OpFn`). The graph digest, a SHA-256 hash of the canonical description of
the public graph, is the statement that Fiat-Shamir absorbs (Sec. `sec:fs`). `contention` records, around each timed
phase, evidence that the phase ran on a quiet machine, and changes no timing, verdict or proof byte. `hostmem`
allocates pinned host memory within the limits that the CUDA driver grants.

**`pvi.commit`: the commitment to the weights (Sec. `sec:commit`).** `commitment` implements SHA-256 Merkle trees with
multiproofs and the Reed-Solomon column commitment of a matrix `[W | b]`: its rows are encoded and a Merkle tree binds
its columns, and the prover recomputes an opened column from the weights. It also implements the transposed and
grouped commitments of a commitment plan, and lookup tables, which are Merkle trees over an embedding's rows. `plans`
computes the commitment plan of mode C: each weight operation's layout (row, col or lookup) and codeword length, the
Merkle trees that matrices share, and the exact number of columns each tree opens. The plan follows a policy:
`cnn18c` is the paper's configuration, and `per-op` (no plan, one tree per weight operation) is the library default.
`setup_proof` implements the parameters, prover and verifier of the setup proof, which certifies an untrusted
commitment (Sec. `sec:setup-proof`, Theorem `thm:untrusted`). `setup_cache` stores the prover's Merkle leaves, so
that a later process can rebuild the same trees, after a spot check against the weights, without re-encoding the
model. `cost` lists the weight-operation shapes of a decoder without building it, and gives the byte model of a
query's proof and of the commitment's setup for models too large to run.

**`pvi.v0`: protocol v0 (Sec. `sec:protocol`) and its non-interactive form (Sec. `sec:fs`).** `params` derives the
security parameters from the level `lambda` (the Freivalds repetitions `r` and the opened columns `t`) and computes
the soundness bound they give. `transcript` provides the verifier's randomness: interactive draws, or a Fiat-Shamir
transcript, a domain-separated SHAKE-256 sponge over labelled, length-prefixed messages that absorbs large messages
as a one-level tree. `graph_commitment` commits a graph's weights per operation or under a plan (`commit_graph`).
`prover` computes the claims, the folded rows and the opened columns. `verifier` checks a proof in mode C, K or Kpre
on the CPU or a GPU, optionally in a streaming form; `streaming` holds that form's wire formats and claim uploads.
`query` runs one query end to end (`run_query`), with the absorption of the statement, the byte counts and the
timings. `claimcodec` is the compact, lossless wire encoding of the claims (`PVC3`) and of field elements.
`token_sampler` is the exact integer Gumbel-max sampler for generation with a seed chosen by the client
(Sec. `sec:gen-sampling`).

**`pvi.v1`: the V1 variant (Sec. `sec:v1`, Appendix `app:v1`).** For each cut weight operation, the prover sends the
requantised int8 values instead of the int32 claims. It proves with a windowed logUp argument that each value lies
in its window, and it proves the logUp sum with a fractional-sum GKR over the extension field. V1 needs no
commitment beyond v0's. `cut` decides which operations are cut by the rules (P1)-(P5), and fixes their windows and
the GKR instances. `logup` builds the table, the multiplicities, the leaves of an instance and the final identity.
`gkr` implements the fractional-sum GKR prover and verifier. `gkr_triton` is the same prover in Triton, and its
transcript is byte-identical. `products` computes by value the products that the verifier's final check needs.
`protocol` runs a V1 query and implements the verifier's checks: (D1)-(D5), the GKR checks and (F1)-(F4).

**`pvi.models`: the integer models.** `decoder` builds random-weight integer decoders with the shapes of GPT-2,
OPT, Llama-2 and Qwen3 (`build_decoder`, `CONFIGS`), on which the cost measurements run, optionally with the last
block evaluated at the last position only (except its keys and values). `hf_opt` and `hf_llama` build the certified
integer models from Hugging Face checkpoints (OPT, and Llama and Qwen3). These models have the same graph as
`decoder` and different constants (Sec. `sec:quality`). `cnn` defines the float image classifiers of the comparison
benchmark. `quantize` converts a float CNN into an int8 integer graph, with per-channel weights and BatchNorm
folded. `datasets` loads the image datasets of the benchmark.

**`pvi.kernels`: accelerated exact kernels.** `native` holds C++/OpenMP kernels for a CPU verifier and for integer
re-execution: causal attention, the cheap operations, the int8 GEMM, field products, decoding of the claim encoding,
and V1's window-table products. The library is compiled once per host and loaded with ctypes. `attention_triton` is
the fused exact causal attention on a GPU. `codec_triton` holds the fused kernels of the claim encoder on a GPU.
Each kernel computes the same integers as the torch code it replaces. Without a compiler or Triton, or with its
option off, the torch path runs.

**`pvi.baseline`: the spot-checking verifiers that the attack targets (Sec. `sec:path`, `sec:sampling`).** `path` is
the path protocol of Anchuri et al. on float MLPs: commitments to the model and to the trace, messages, prover,
verifier and the `RandPathTest` path sampler. `path_int` runs the same test on the integer CNN graphs of
`pvi.models`, as the like-for-like baseline of the benchmark, with exact per-neuron detection probabilities.
`samplers` implements the value-aware path samplers, and `adaptive` implements the adversary that adapts to them.
`analysis` computes the exact probability that the path test accepts a given trace under any sampler.

**`pvi.attacks`: the attacks on the path test (Sec. `sec:path`, `sec:break`).** `tamper` implements single-neuron
trace tampering, and `backdoor` implements the trigger backdoor that the path test certifies. `baselines` reproduces
the forgeries with which Anchuri et al. evaluate their protocol (gradient-descent reconstruction, layer injection,
inverse transform and logit swap), which the path test rejects.

**`pvi.mnist`: the MNIST networks of the attack experiments.** `architecture` represents a network as the layered
DAG of the path protocol (parent sets `G_j`, the local relation `a_j = phi(sum w_ij a_i)`), and `network` represents
its execution traces. `data` loads MNIST and its class subsets, and `training` provides seeded training through
`models`, the PyTorch bridge. `gradients` computes the output's sensitivity to an internal activation. `zoo` defines
the models and the model-substitution settings.

**`pvi.options`: the implementation options.** Each option chooses one of several equivalent code paths: native or
torch kernels, the fused GPU kernels, GPU or host decoding, int8 GEMMs, the hash threads of the tree absorption, and
V1's optimised or baseline verifier. Every path gives the same integers, bytes and verdicts, and only time and memory
change. An option is set for the process (`options.configure`) or for a block (`with options.using(...)`). Every
script exposes each option as a command-line flag and records `options.describe()`. Three environment variables
name places on the host and select no code path: `CXX`, `PVI_NATIVE_DIR` and `PVI_PY_INCLUDE`.

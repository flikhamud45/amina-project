# Artifact: Certified but Compromised

This anonymous snapshot holds the code and the stored measurements behind the submission *Certified but Compromised*.
It contains:

* an implementation of the path-sampling proof of inference that the paper attacks, and the attacks (single-neuron
  edits, a backdoor, and in-range edits against range checks on a real LLM);
* the defence: every weight operation is checked with Freivalds' algorithm against a Reed-Solomon/Merkle commitment to
  the exact int8 weights (protocol v0, modes C, K and Kpre), the lookup-based proof-size reduction V1 (logUp-GKR), the
  claim codec, the Fiat-Shamir transcript, and the native CPU and fused GPU verifiers;
* the integer models built from public checkpoints (OPT; the builders for RMSNorm/RoPE/SwiGLU models), and the scripts
  that measure their quality;
* the benchmarks and the stored records from which the paper's tables are generated;
* `extended_version.pdf`: the extended version with every proof.

Machine-specific paths in the stored records were replaced by placeholders (`$PROJECT`, `$HOME`, `node`);
nothing else was changed.

## Installation

Python 3.12, then from `code/`:

    pip install -r requirements.txt            # torch 2.5.1, numpy, pytest, ...
    pip install -r requirements-llm.txt        # transformers, huggingface_hub, pyarrow (real checkpoints)

The native CPU kernels are compiled once per host with a C++17 compiler (`$CXX`, default `g++`, flags
`-O3 -march=native -fopenmp -fwrapv`); without one, the exact torch paths are used and give the same integers. The fused
GPU kernels need Triton and a CUDA GPU; without them, the torch paths are used.

## Quick tests (minutes, CPU only)

    cd code && PYTHONPATH=src python -m pytest -q tests/test_protocol.py tests/test_wire.py tests/test_fs_tree.py \
        tests/test_pruning.py tests/test_prob_max.py

These cover the protocol in every mode (honest queries accepted, every tamper class rejected), the wire format and the
Fiat-Shamir transcript, last-block pruning and the attention probability scale. The full suite is
`python -m pytest -q tests` (about 30-50 minutes on 8 cores).

## Results map

| Paper result | Script (from `code/`, with `PYTHONPATH=src`) | Stored records |
|---|---|---|
| Attack on the path protocol (MNIST, CNNs) | `experiments/2_attack/run.py` | `artifacts/results/` |
| In-range edits against range checks (OPT) | `experiments/2_attack/inrange_llm.py --model facebook/opt-1.3b --moves any` | `artifacts/results/inrange_llm_*.json` |
| Smarter sampling rules | `experiments/3_sampling_fixes/run.py` | `artifacts/results/` |
| Prover and verifier cost of v0 (Table "LLM evaluation") | `experiments/6_improvements/perf.py --model llama2-7b --seq 2048 --modes C Kpre --lean --wire on --prune-last on --fs` | `artifacts/comparison/` |
| V1 proof size and time | `experiments/8_v1/v1_bytes.py --model llama2-7b --layers 2 --seq 2048 --modes Kpre C --prune-last --fs --full` | |
| Re-execution baselines and client options | `experiments/7_reexec/reexec.py --model llama2-7b --seq 64 2048 --variants int float bf16 dynint8` | |
| Comparison with published systems | `experiments/5_comparison/aggregate.py`, `paper_assets.py` | `artifacts/comparison/` |
| Commitment plans and the fixed part of a proof | `experiments/6_improvements/ps2_tables.py` | |
| Quality of the certified integer model | `experiments/9_quality/ppl_2048.py --model facebook/opt-1.3b --graphs smooth --pbits 8 11` | |
| Reproducible commitment `C_M` | `experiments/9_quality/digest_check.py --model facebook/opt-125m --commit` | |

Costs depend only on the shapes of the weight operations, so the benchmarks use random weights of each model's shapes
(`CONFIGS` in `src/pvi/fullcheck/transformer.py`); the real-weight runs use the public checkpoints through
`src/pvi/fullcheck/real_weights.py`.

## The certified model and its commitment

The integer model is a deterministic function of the public checkpoint, the calibration tokens and the recipe
(`real_weights.CERTIFIED_RECIPE`: `build_opt_from_hf` with norm gain 4, SmoothQuant alpha 0.6 through the norm gains,
each channel's maximum as its statistic and no clipping, 11-bit attention probabilities). Anyone can rebuild it and
recompute `C_M`; `digest_check.py` prints the graph digest, a hash of every weight matrix and the commitment's Merkle
roots.

Calibration tokens (`common.certified_calib`): the first four 2,048-token windows of the first 4,000,000 characters of
the WikiText-2 (raw) train split (lines joined with blank lines), with the checkpoint's tokenizer.
Graph digests (SHA-256) and the hash over every Merkle root, reproduced on two Linux nodes (1 and 8 threads, CPU or GPU
encoding):

| Model | graph digest | roots |
|---|---|---|
| OPT-125M | [pending: jobs 1012438-9] | |
| OPT-1.3B | [pending: jobs 1012438-9] | |

(The earlier recipe, alpha 0.55 with percentile statistics calibrated on the last 128-token test window, is
`digest_check.py --recipe legacy`; its digests were `46d23c80...` and `01354c3d...`.)

## Hardware used for the paper

The timings in the paper were measured on the platforms its evaluation section names. Proof sizes, acceptance and
rejection do not depend on the platform.

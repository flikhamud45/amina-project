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
(`build_opt_from_hf`: norm gain 4, SmoothQuant alpha 0.55 through the norm gains, smooth_pct 99.9, 99.99th-percentile
clipping). Anyone can rebuild it and recompute `C_M`; `digest_check.py` prints the graph digest, a hash of every weight
matrix and the commitment's Merkle roots.

Calibration tokens: the last 128-token window of the WikiText-2 (raw) test split, with the checkpoint's tokenizer.
Graph digests (SHA-256) and the hash over every Merkle root, reproduced on two Linux nodes (1 and 8 threads, CPU or GPU
encoding) and, for the digest, on Windows with another torch version:

| Model | graph digest | roots |
|---|---|---|
| OPT-125M | `46d23c80874752fc625cd55a9afb552fcbfd458ff880b0bc30a60efb8318a439` | `480463ab9be46f4d...` |
| OPT-1.3B | `01354c3dd18bf790...` | `ff005dca59e6b579...` |

## Hardware used for the paper

The timings in the paper were measured on the platforms its evaluation section names. Proof sizes, acceptance and
rejection do not depend on the platform.

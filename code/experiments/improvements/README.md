# Prover and verifier improvements

The extended version measures each implementation improvement of protocol v0 against the implementation it replaces on the
same queries, with identical claims, proofs and verdicts (`tab:eval-improve` in `sections/ext_evaluation.tex`, the
body's summary in Sec. `sec:eval-cost`). Every improvement is an implementation option of `pvi.options` or a
documented library parameter, so each row is one measurement with the option off ("before") and on ("after"). The
scripts here time the prover-side rows and check that both paths give the same integers or bytes; the verifier rows
and the prover's attention are `costs/perf.py` runs with the option flags that every timing script takes.

| Paper result | Script | Command (from `code/`) | Output |
|---|---|---|---|
| `tab:eval-improve`, "Opening and fold on int8 GEMMs (prover, C, T=64)": Llama-2-7B 1/32 and 4/32, Llama-2-13B 1/40, GPT-2 12/12; Sec. `sec:eval-cost` "3.7–4.3x" | `open_ab.py` | `python -m experiments.improvements.open_ab --model llama2-7b --layers 1 4 --queries 7`; `... --model llama2-13b --layers 1`; `... --model gpt2` | stdout: `prove_s` of the variants `float64` (before) and `int8` (after) |
| `tab:eval-improve`, "Weight products on int8 GEMMs (forward pass)": Llama-2-7B 2/32 | `forward_ab.py` | `python -m experiments.improvements.forward_ab --model llama2-7b --layers 2 --seq 2048` | stdout: `prove_forward` of `float32` / `int8` |
| `tab:eval-improve`, "Threaded assembly of the encoded claims (encoder)": OPT-1.3B 24/24, Llama-2-7B 2/32; the fused Triton encoder "1.74 against 1.54 s on OPT-1.3B" | `encode_ab.py` | `python -m experiments.improvements.encode_ab --model opt-1.3b --layers 24 --seq 2048 --fused`; `... --model llama2-7b --layers 2 --seq 2048` | stdout: `encode_s` of `one_thread` / `threaded` (/ `fused`) |
| `tab:eval-improve`, "Fused exact attention in the prover (forward pass)": OPT-1.3B 12/24, Llama-2-7B 2/32 | `costs/perf.py` | before: `python -m experiments.costs.perf --model opt-1.3b --layers 12 --seq 2048 --modes Kpre --prune-last off --queries 3 --verifier-device cuda --stream --no-fused-attention --label attn-before`; after: the same without `--no-fused-attention` (`--label attn-after`); then `--model llama2-7b --layers 2` | stdout: `timings.prove_forward` |
| `tab:eval-improve`, "GPU claim decoder and fused attention (GPU verifier, Kpre)": OPT-1.3B 12/24, Llama-2-7B 2/32 | `costs/perf.py` | before: `python -m experiments.costs.perf --model opt-1.3b --layers 12 --seq 2048 --modes Kpre --queries 3 --verifier-device cuda --stream --no-gpu-decode --no-fused-attention --label gpuv-before`; after: the same without the two `--no-...` flags; then `--model llama2-7b --layers 2` | stdout: `verify_s` |
| `tab:eval-improve`, "Native exact kernels for every phase (CPU verifier, Kpre)": Llama-2-7B 2/32 | `costs/perf.py` | before: `python -m experiments.costs.perf --model llama2-7b --layers 2 --seq 2048 --modes Kpre --queries 3 --verifier-device cpu --threads 8 --no-native --no-malloc-tune --label cpuv-before`; after: the same without `--no-native --no-malloc-tune` (and with `CXX` set: the native kernels are compiled on first use) | stdout: `verify_s` |
| "copies take 32% of the CUDA time of an OPT-1.3B query (12 blocks) and int8 GEMMs 4.4%" (`app:eval-improve`) | `gpu_profile.py` | `python -m experiments.improvements.gpu_profile --model opt-1.3b --layers 12 --seq 2048` | stdout: tables and a JSON line with `shares` |
| "12.7 s for Llama-2-7B on 8 Xeon threads" (the Kpre client's key, `ext_discussion.tex`) | `fold_ab.py` | `python -m experiments.improvements.fold_ab --model llama2-7b --devices cpu cuda --reps 3 --threads 8` | stdout: `precompute_s` of `float64` / `default` |

Every script prints one JSON object per line with the run's environment under `env` (commit, host, CPU, GPU,
versions, the implementation options in force and what the host runs), and exits with status 1 if the two paths
disagree or a query is rejected. `--help` lists every option; the protocol settings default to the paper's
configuration (`experiments.common.cli.PAPER`: plan `cnn18c`, Fiat-Shamir, the compact encoding, the lean prover,
the pruned last block, the verifier's own embedding rows), the runtime flags to the GPU prover and 8 threads.

## Scripts

### `open_ab.py`

Builds the decoder as `costs/bench.py` does (random weights, `--layers` blocks, the last one pruned), commits it under
`--policy` and runs `--queries` mode-C queries with `run_query`, each once with the prover's fold and column openings
in float64 (`field.small_matmul_mod`) and once on int8 GEMMs (`field.int8_weight_matmul`), alternating, with the same
challenge seed: it checks that the folded rows, opened columns and multiproofs are identical and that the verifier
accepts. Record: the medians of the prover's parts, `prove_s` (the prover's total: the table's value), `open_fold_s`,
`vs_float64`, `identical`, `int8_ok`. Needs a CUDA GPU whose `torch._int_mm` passes its self-check; minutes per build.

### `forward_ab.py`

Times the lean prover's forward pass (`Prover.claims` with the claims kept for the compact encoding, what
`prove_forward` times) with the weight products as float32 chunks and as int8 GEMMs (`graph.int8_product`) on the
same queries, alternating, and checks that the claims are identical. Needs a CUDA GPU with int8 GEMMs; minutes.

### `encode_ab.py`

Computes one query's claims on the GPU as the lean prover keeps them, then encodes them (`claimcodec.encode` on the
device) with the encoded pieces assembled on one thread (torch's thread count set to 1 for the call: a
single-threaded copy) and on `--threads` threads, and with `--fused` the fused Triton encoder
(the option `fused_encoder`); every variant must give the same bytes (compared by SHA-256). `--profile` prints a
`torch.profiler` table of one call. The full OPT-1.3B at 2,048 tokens gives a proof of about 2 GB.

### `fold_ab.py`

The Kpre client's precomputation `u = chi A` for every weight matrix (`Verifier.precompute`) with the float64 limb
products (the option `int8_fold` off) and with the default path (on a CPU the native exact field product, which needs
the native kernels; on a GPU int8 GEMMs), on the same challenges; every `u` must equal the float64 path's. The full
Llama-2-7B holds its 6.7 GB of int8 weights in host memory; on 8 threads of a Xeon Silver 4114 its float64 path took
107 s per run and the native path 12.7 s.

### `gpu_profile.py`

One query with the prover and the streaming verifier on the GPU (default Kpre with the verifier's own embedding rows)
after a warm-up, profiled with `torch.profiler`: the operators ranked by CUDA and by CPU time, and the share of the
total CUDA time of `aten::copy_` (the host-device copies), `aten::_int_mm` (the int8 GEMMs), `aten::index_select` and
the fused attention kernel (`--share` chooses others). Needs a CUDA GPU; minutes.

## How the "before" paths are selected

| Row | "before" | Selected by |
|---|---|---|
| opening and fold | float64 products in `WeightCommitment.fold` / `columns_at` | `--no-int8-opening` (`open_ab.py` runs both) |
| forward pass | float32 chunks in `MatOp.compute` | `--no-int8-forward` (`forward_ab.py` runs both) |
| encoder assembly | the encoded pieces copied on one thread | torch's thread count during the call (`claimcodec._join_pieces` copies on torch's threads); one thread gives the single-threaded copy of one `b"".join` |
| fused encoder | (the threaded torch encoder) | the option `fused_encoder` |
| prover attention | torch attention with a `T x T` buffer | `--no-fused-attention` |
| GPU verifier | host decoder, torch attention | `--no-gpu-decode --no-fused-attention` |
| CPU verifier | torch everywhere | `--no-native` (all native kernels) and `--no-malloc-tune` (glibc's defaults) |
| Kpre precomputation | float64 limb products | `--no-int8-fold` (`fold_ab.py` runs both) |

Both int8 paths of the prover also need a GPU whose `torch._int_mm` passes `field.int8_ok`'s self-check; the
records' `int8_ok` field says whether it did.

## Reproducing the paper's numbers

The table's numbers come from the final platform (an NVIDIA L40S prover and GPU verifier, 8 threads of its CPU), each
row's two commands run one after the other on the same node: the run package's `improve` tier runs them. The commands use
the paper's configuration (Fiat-Shamir, the plan `cnn18c`).

On a laptop without a GPU only `fold_ab.py` runs (with the float64 product on both paths where the native kernels
cannot be built):

```bash
cd code
PYTHONPATH=src:. python -m experiments.improvements.fold_ab --model tiny-llama --reps 1 --threads 2
```

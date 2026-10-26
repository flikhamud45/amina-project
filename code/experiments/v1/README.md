# V1: proving the requantised values

V1 (Sec. `sec:v1`) sends, for most weight operations, the requantised int8 value of each claim instead of the claim,
and proves the requantisation with a logUp argument checked by GKR; the rest of the proof is v0's. These scripts
measure what that changes on the same model, query and parameters as v0: the proof's bytes message by message, the
prover's and the verifier's times, the verifier's two implementations phase by phase, and the GKR prover on its own.

| Paper result | Script | Command (from `code/`) | Output |
|---|---|---|---|
| `tab:eval-v1` (V1 and v0 proofs, ratio, prover and verifier times); the abstract's and Sec. `sec:eval-v1`'s "2.0–2.3x smaller at 2,048 tokens with random weights"; the extended version's `tab:v1` (the GKR's share of the prover) | `v1_bytes.py` (tier `v1`) | `python -m experiments.v1.v1_bytes --model llama2-7b --seq 2048 --modes Kpre C --full --reps 5` (and GPT-2 at 64 and 512, OPT-125M, OPT-1.3B, OPT-6.7B at 64 and 2,048, Llama-2-13B, Qwen3-4B) | stdout, one JSON line per mode |
| `tab:options`' V1 columns (V1's byte ratio and its CPU verifier against v0's); the body's "45.4 s against v0's 19.6 s" for the full Llama-2-7B; the extended version's "1.44 s per block (2.49 s before)" | `v1_bytes.py --v1-verifier optimised baseline` (tier `cpuv`) | `python -m experiments.v1.v1_bytes --model llama2-7b --seq 2048 --modes Kpre --full --reps 5 --v1-verifier optimised baseline --label cpuv-v1` (and `--seq 64`, Qwen3-4B) | stdout (`timed.v1_paths`) |
| where the V1 verifier's time goes (the per-phase breakdown behind the numbers above) | `v1_phases.py` | `python -m experiments.v1.v1_phases --model llama2-7b --layers 2 --seq 2048 --mode Kpre --reps 3 --device cuda` | stdout, one JSON line per verifier and a table |
| V1 with the GPU verifier at 2,048 tokens (tier `gpuv`) | `v1_bytes.py --verifier-device cuda` | `python -m experiments.v1.v1_bytes --model llama2-7b --seq 2048 --modes Kpre --full --reps 3 --verifier-device cuda --label gpuv-v1` | stdout |
| V1 on real weights: the abstract's "1.7x on the real OPT-6.7B", `tab:eval-real`'s V1 column, the real OPT-125M's 1.68x / 1.77x (`ext_v1.tex`) | `v1_bytes.py --real` (tier `real`) | `python -m experiments.v1.v1_bytes --model opt-6.7b --real facebook/opt-6.7b --seq 2048 --modes Kpre --full --reps 3 --tampers 3 --label real`, and the same without `--real` (`--label real-random`) | stdout |
| the byte profile of a model whose prover the machine cannot hold | `v1_bytes.py --codec-only` | `python -m experiments.v1.v1_bytes --model opt-6.7b --real facebook/opt-6.7b --seq 2048 --codec-only --device cpu` | stdout (`blocks`, `total`) |
| `tab:params`' V1 rows (leaves, instances, table) | `v1_bytes.py` records read by `costs/params_table.py --v1-cells` | `python -m experiments.costs.params_table --v1-cells <v1_bytes records at two block counts>` | stdout |
| `app:eval-gkr-leaf` and `ext_v1.tex`: "an instance of 2^25 leaves takes 0.52 s (15.6 ns per leaf)" on the RTX 2080 Ti | `gkr_bench.py` | `python -m experiments.v1.gkr_bench --n 20 22 24 25 --check 20` | stdout, one JSON line per size |

Run every command from `code/` with the checkout's library first on the path (`PYTHONPATH=src:.` on Linux,
`PYTHONPATH="src;."` on Windows). `python -m experiments.v1.<script> --help` lists every option.

## Scripts

### `v1_bytes.py`

Builds a random-weight decoder (the names of `pvi.models.decoder.CONFIGS`, or the small shapes `tiny-*` and
`small-llama` of `experiments.common.models.TINY`; `--layers` blocks of it, with the parameters `(r, t)` of the full
model) or, with `--real`, a checkpoint's integer graph; commits it in mode C; runs one v0 query and sizes V1's proof on
the same query (`pvi.v1.protocol.cut_proof_bytes`: M1 encoded for real, every other message sized exactly).

* **Inputs.** Random weights need nothing. `--real <checkpoint>` downloads the checkpoint from Hugging Face and the
  WikiText-2 (raw) splits: OPT is built with the certified recipe (`hf_opt.CERTIFIED_RECIPE`, calibrated on four
  2,048-token windows of the train split) unless `--recipe legacy` (norm gain 4, `--smooth 0.55`, one calibration
  window of `--calib-seq` tokens from the end of the test split); a
  Llama-family checkpoint with `hf_llama.MAIN_RECIPE` and its fixed public prefix. The queries are test windows of
  `--seq` tokens.
* **Main options.** `--modes C Kpre`, `--full` (also run the whole V1 query: the GKR on the GPU in Triton, or in
  PyTorch on a CPU), `--reps n` (with `--full`: `n` timed queries of each protocol, interleaved, after a warm-up),
  `--tampers n` (with `--full`: `n` trace-tampering and `n` single-value V1 queries, each of which must be rejected),
  `--v1-verifier optimised baseline` (every listed verifier checks the same timed queries; `optimised`, the
  default, runs the window-table products, the batched GKR check and the batch-inverted logUp identity, `baseline`
  the float64 limb products and one check per instance), `--lmax 27`, `--codec-only`. The protocol settings default
  to the paper's configuration (`experiments.common.cli.PAPER`): `--policy cnn18c`, `--fs`, `--lean`,
  `--prune-last`, `--lookups` (mode Kpre's verifier reads the embedding rows itself), each with a `--no-...` form,
  `--prob-max 2047` (11-bit attention probabilities; 255: the 8-bit graphs);
  V1's proof is always in the compact encoding, and `--wire`/`--no-wire` concerns v0's proof only. The runtime flags:
  `--device` (the prover's), `--verifier-device`, `--threads 8` and every implementation option of `pvi.options`.
* **lmax.** `--lmax` sets the GKR instances' target size `2^lmax`: consecutive operations are packed into one
  instance while it stays within `2^lmax` leaves, and an operation larger than that is an instance of its own. The
  default is the library's `pvi.v1.cut.LMAX = 27`, the value of the run package and of the proofs `tab:eval-v1`
  shows; a GPU with less memory may need a smaller `--lmax` (24 fits an 11 GB card; the proofs differ by under 0.3 MB). Each record lists every instance's size (`instance_bits`), from which the GKR term of Theorem `thm:v1` follows.
* **Checks.** Every line checks the byte accounting (`accounting`, `accounting_ok`): v0's `u` and opened columns
  equal `pvi.commit.cost.proof_bytes` exactly; with `--full`, every V1 message but the Merkle multiproofs equals the
  sizing; the graph's shapes and claim columns are the benchmark's (`shapes`). A failed check, a rejected honest
  query or an accepted tampered one makes the exit status 1.
* **Records.** One JSON line per mode: the settings, `v0_bytes` and `v1_bytes` per message, `v0_total`, `v1_total`,
  `ratio`, the cut's `n_leaves`, `n_instances`, `instance_bits`, `table`, `n_exceptions`, `v1_full`, `timed`
  (`v0`, `v1`, and `v1_paths` per verifier: per-part medians, `prove_s` and `verify_s` with `[min, max]`, `fs_s`;
  `paper/numbers.py` adds `fs_s` to both parties), `tampers`, `peak` and `env` (the run's environment and
  implementation options). `--codec-only` prints instead the claims of one forward pass encoded op by op, per block.
* **Hardware and time.** The paper's cells: an NVIDIA L40S (48 GB) prover, the CPU verifier on 8 threads. A 7B
  decoder at 2,048 tokens with `--full` needs one to three hours and up to 256 GB of host memory (its GKR instances of
  `2^27` leaves about 22 GB of GPU memory each); GPT-2 at 64 tokens takes minutes.

### `v1_phases.py`

V1's verifier phase by phase on a random-weight decoder (`--model`, or `--small D H DH FF V` for a Llama-shaped one of
those sizes), each query checked once by every verifier of `--v1-verifier` (default both), interleaved. Inside the
final check it also times `_xbar`, the eq tables, the Kpre fold check, the window products of each verifier and the
inner products, and counts the operations that fall back to the baseline products. Records: one JSON line per verifier
(`path`: its name; `timings`: the median of every phase; `verify_s`; `env`), then a table. `--micro ROWS T` times the
final check's products of one operation of each kind under both verifiers without a model (`ns_per_entry`), checks
that they agree, and projects them onto a Llama-2-7B block. The protocol flags are those of `v1_bytes.py`. Minutes for
two Llama-2-7B blocks at 2,048 tokens; the laptop check below takes seconds.

### `gkr_bench.py`

One GKR instance of `2^n` leaves per size, laid out as V1's operations lay them out (`--T` positions per row), proved
by the Triton prover (`pvi.v1.gkr_triton.prove_leaves`) under Fiat-Shamir: the fastest of `--reps` proofs after an
untimed one (`s`, `ns_per_leaf`), the first proof (`first_s`, which compiles the kernels) and the peak GPU memory.
`--check n` compares the transcript with the PyTorch prover's byte for byte (small `n`); `--gkr-profile` adds one more
proof with every phase synchronised and timed (`profile`). Needs a GPU with Triton (exit status 2 without one);
seconds per size up to `n = 25` (5.8 GB).

## Reproducing the paper's numbers

The final numbers come from one platform through the run package (`bash code/experiments/package/run.sh --platform <name> <tier>`, `experiments/package/README.md`),
whose records `paper/numbers.py` turns into the tables:

* `v1`: `v1_bytes.py --full` on every model of `tab:eval-v1` (modes C and Kpre, 64 and 2,048 tokens, the headline
  cells with 5 timed queries, the others with 3);
* `cpuv`: `v1_bytes.py --modes Kpre --full --v1-verifier optimised baseline` on Llama-2-7B (64 and 2,048 tokens) and
  Qwen3-4B (2,048), the V1 columns of `tab:options`;
* `gpuv`: `v1_bytes.py --verifier-device cuda` on OPT-1.3B and Llama-2-7B at 2,048 tokens;
* `real`: `v1_bytes.py --real` on OPT-1.3B and OPT-6.7B (certified recipe) and, with access to the gated Llama-2
  checkpoints, Llama-2-7B, each next to its random-weight twin;
* `smoke`: one V1 query of 2 GPT-2 blocks at 64 tokens (`--lmax 20`).

The `improve` tier runs `v1_phases.py` (optimised and baseline verifiers) and `gkr_bench.py`.

A small version runs on a laptop CPU in a minute (no downloads, no GPU; the GKR then runs in PyTorch):

```bash
cd code
export PYTHONPATH=src:.
python -m experiments.v1.v1_bytes --model tiny-llama --seq 16 --modes C Kpre --full --reps 2 --tampers 2 --lmax 20 \
    --v1-verifier optimised baseline --device cpu --threads 2
python -m experiments.v1.v1_bytes --model tiny-llama --seq 16 --codec-only --lmax 20 --device cpu --threads 2
python -m experiments.v1.v1_phases --small 128 4 32 344 512 --layers 2 --seq 64 --lmax 18 --reps 2 --device cpu \
    --threads 2
python -m experiments.v1.v1_phases --micro 256 128 --reps 2 --threads 2
```

`tests/test_hf_opt.py` runs `v1_bytes.py --real` (both recipes, both verifiers, `--codec-only`) on a tiny random
OPT, and `tests/test_real_llama_end_to_end.py` its shape check on a tiny random Llama.

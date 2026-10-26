# Deployment: the cost to the provider and when a client gains

Two questions decide where the protocol is worth deploying. What does proving cost a provider over serving the same
model without proofs on the same GPU (Sec. `sec:eval-cost`, note f of `tab:options`, the extended version's
`ext:disc-options`)? And when does a client gain over downloading the weights or re-executing the model
(`fig:breakeven`, Sec. `sec:discussion`)? `serve_ab.py` measures the first on one GPU, on the same token ids, against
our exact integer model and an fp16/bf16 decoder of the same shapes; `ev_breakeven.py` computes the second from
measured inputs and the byte model.

| Paper result | Script | Command (from `code/`) | Output |
|---|---|---|---|
| Sec. `sec:eval-cost`, "Base costs": proving a 2,048-token OPT-1.3B prompt takes 2.1 s in Kpre and 4.4 s in C, "4.1 and 8.6x our exact integer forward pass without proofs, and 19 and 40x an fp16 one"; "Proving 64 prompts of 64 tokens together in Kpre takes 3.4 s (4.8 and 18x)" | `serve_ab.py` (tier `serve`) | `python -m experiments.deployment.serve_ab --model opt-1.3b --workload a b --batch-proof seq joint --u2-dtype fp16 bf16 --label serve` | stdout, `row: "prove"` lines: `prove_s`, `prover_over_u1`, `prover_over_u2` |
| note f of `tab:options`: the provider's cost over unverified serving, "ours 9.8-10.6x (Kpre) and 19-154x (C) fp16 inference" (Llama-2-7B) | `serve_ab.py` (tier `serve`) | the same with `--model llama2-7b` | `prover_over_u2.fp16` |
| `ext:disc-options`: serving with the exact model and proving later, on request, with the same proof; "the integer engine's cost over fp16 serving (4-10x on OPT-1.3B)"; batches of B prompts in one mode-C proof | `serve_ab.py` | as above | `row: "on_request"` lines: `serve_over_u2`, `later_proof` (byte-identical, `deterministic`); `batch_proof: "joint"` lines: `proof_mb_per_token` |
| `fig:breakeven`; Sec. `sec:discussion`, "Who the protocol serves": "below about 22 queries of 64 tokens", "at 2,048 tokens the weights are only 1.0x the size of one proof, or 2.3x with V1" (`\BreakevenQueriesLong`, `\BreakevenQueriesVOne`: the weights over one proof), "links faster than 0.42-0.59 Gb/s (V1: 0.23 Gb/s)" | `ev_breakeven.py` | `python -m experiments.deployment.ev_breakeven --figure ../paper_sp2027/figures/ev_breakeven.pdf` | `artifacts/results/ev_breakeven.{json,tex,pdf}` and the figure: `panel1` (`Q_star`), `panel2` (`L_star_gbps`) |
| `ext:disc-real`, "V1": V1's client "faster than v0 on links slower than about 1.1 Gb/s" | `ev_breakeven.py` | the same | `v1_against_v0` (`v1_faster_below_gbps`) |
| Sec. `sec:discussion`, "Proof size" (also `ext:disc-limits` and `ext_v1.tex`): one Llama-2-7B proof "reaches the size of the int8 weights at about `\BreakevenTokens` tokens without V1 and, by the byte model, at about `\BreakevenTokensVOne` with it" (fallback 2,100 and 4,900) | `ev_breakeven.py` | the same | `prompt_crossover` (`tokens`: 2,111 without V1 and 4,854 with it under Fiat-Shamir; see the note below) |

Run every command from `code/` with the checkout's library first on the path (`PYTHONPATH=src:.` on Linux,
`PYTHONPATH="src;."` on Windows). `python -m experiments.deployment.<script> --help` lists every option.

## Scripts

### `serve_ab.py`

The prover against unverified serving, on one device and the same token ids. A cell is a model, a prompt length `T`
and a batch `B` (`--workload a`: `T` = 64, 512, 2,048 at `B` = 1; `--workload b`: `T` = 64 with `B` = 1, 4, 16, 64 and
`T` = 2,048 with `B` = 1, 4; or `--cells 64x1 2048x4 ...`). Every cell times, interleaved over `--reps` repetitions
after one warm-up:

* **U1**, our exact integer model served without proofs (the integer graph's forward pass with every claim dropped);
  its logits are checked equal to the logits the proof claims, so prover/U1 is the cost of the proof itself;
* **U2**, an eager floating-point decoder of the same shapes in each `--u2-dtype` (`reexec.FloatDecoder`: `F.linear`
  and `scaled_dot_product_attention`, the last block pruned like the integer graph's, no rotary embedding);
* **the prover** in each of `--modes` (C, Kpre): the prover's side of `run_query` alone, with no verifier check in the
  timed window, B prompts proved one after another (`--batch-proof seq`) or as one statement (`joint`).

* **Inputs.** Random-weight decoders of the shapes of `pvi.models.decoder.CONFIGS` (`--model opt-1.3b qwen3-4b
  llama2-7b opt-6.7b llama2-13b`; the costs depend on the shapes only) or the small shapes of
  `experiments.common.models.TINY` (`tiny-llama`, ...), built with `--seed` on `--layers` blocks (0: the full model)
  and `--calib-tokens` calibration tokens; prompts of random ids (`--prompt-seed`). No downloads.
* **Main options.** The protocol flags default to the paper's configuration (`experiments.common.cli.PAPER`):
  `--policy cnn18c` (mode C's commitment plan; `--policy per-op` for one tree per weight op), `--fs`, `--wire`,
  `--lean`, `--prune-last`, `--lookups`, each with a `--no-...` form, `--prob-max 2047` (11-bit attention
  probabilities, also U1's), `--lam 128`, `--rate 4`, `--setup-cache DIR`.
  Mode C is proved without a verifier, so it needs Fiat-Shamir (`--no-fs` is for `--modes Kpre`). `--accept` (default
  on) runs one untimed `run_query` per variant and cell to show that the verifier accepts the same proof. `--boot`
  (10,000 bootstrap resamples) and `--residual-max` (0.02) set the intervals and the phase check. The runtime flags:
  `--device`, `--verifier-device` (the acceptance check's verifier; default `cpu`), `--threads 8` and every
  implementation option of `pvi.options`.
* **Records.** One JSON line per (cell, mode, batch proof), `row: "prove"`, with the medians and 95% bootstrap
  intervals of U1, U2, the measured window and the prover (`u1_s`, `u2_s`, `window_s`, `prove_s`), the ratios
  (`prover_over_u1`, `prover_over_u2`, `u1_over_u2`), the prover's phases (`phases_s`) and the overhead over U1 by
  phase (`overhead_split_s`), the proof bytes per message, the proof MB and prover seconds per token, the proved and
  served tokens per second, the egress cap at 100 Gb/s, the checks (`u1_equals_claims`, `deterministic`, `accepted`,
  `bytes_consistent`) and `ok`, `proof_sha256`, the peak GPU memory of each variant and the host's peak resident
  memory; and one `row: "on_request"` line per cell: U1's serving time and its ratio to U2, and for every prover
  variant the later proof's time, bytes and SHA-256. Every line carries the settings, the setup times and `env`
  (`experiments.common.records.environment`). A variant that does not fit the GPU is recorded as `skipped` (out of
  memory), never extrapolated; the exit status is 1 if a check failed. `--table <logs>` prints such lines as a table.
* **Hardware and time.** The package runs one job per model on an NVIDIA L40S (48 GB) with 8 CPU threads: limits of 3
  hours (OPT-1.3B) to 8 hours (Llama-2-13B) and 128-256 GB of host memory (a joint proof of 4 x 2,048 tokens and its
  CPU acceptance check hold tens of GB of claims). The `serve` tier measures every model in full. A `tiny-*`
  cell takes seconds on a laptop CPU.

### `ev_breakeven.py`

The break-even of a client of Llama-2-7B, computed (nothing is measured here) from `--inputs` and the byte model of
mode C's fixed part (`pvi.commit.cost.proof_bytes`: `u`, the opened columns and the Merkle paths of the full model
with its last block pruned, under each `--policy`, interactive and under Fiat-Shamir):

* **client bytes against the number of queries** (`panel1`): `Q* = W / b` for the weights `W` and a mode-C proof of
  `b` bytes (the measured claims plus the fixed part), at 64 and 2,048 tokens, and with V1 at 2,048 (the claims
  divided by V1's measured ratio);
* **a Kpre client's time against re-execution, by link speed** (`panel2`): the client is faster than the fastest
  exact re-execution on its CPU (`t_r`) above `L* = 8 B / (t_r - t_v)` for a proof of `B` bytes verified in `t_v`;
* **V1 against v0** (`v1_against_v0`): V1's client is faster than v0's below `8 (B_v0 - B_V1) / (t_V1 - t_v0)`;
* **the prompt length at which one proof weighs as much as the weights** (`prompt_crossover`): the proof bytes taken
  linear in the prompt length through the two measured lengths, V1's with its claims divided by the ratio.

It writes `<out>.json` (every crossover with its inputs), `<out>.tex` (the crossovers as LaTeX comment lines, also
printed) and `<out>.pdf`, the figure (one column wide, 7 pt text; the first `--policy` under Fiat-Shamir;
`--figure <path>` writes it there as well, `--no-figure` skips it). Defaults: `--inputs
experiments/deployment/ev_inputs.json`, `--out artifacts/results/ev_breakeven`, `--policy cnn18c`, `--lam 128`,
`--rate 4`. It needs matplotlib and takes a second. From the stored inputs the figure is the paper's
`figures/ev_breakeven.pdf` byte for byte except for its creation date (checked with matplotlib 3.8.4).

#### The inputs (`ev_inputs.json`)

The measured values are typed in by hand, each with the script and the record field it was read from; they were
measured on an RTX 2080 Ti prover with 8 threads of an Intel Xeon Silver 4114 as the CPU client, in the paper's
configuration. The final run's records replace them: each value is the field named below of the final run's line.

| Key | Value | Source (script, record field) |
|---|---|---|
| `model` | `llama2-7b` | the shapes of `pvi.models.decoder.CONFIGS` |
| `weights_int8_GB` | 6.74 | Llama-2-7B's int8 weights, one byte per weight: one download |
| `claims_wire_MB["64"]`, `["2048"]` | 196.43, 6,433.74 | `costs/perf.py --model llama2-7b --seq <T> --modes Kpre` on the full model (tier `cpuv`): `bytes.claims`, the claims in the compact encoding (the verifier reads the embedding rows itself; mode C's proof sends the same claims and, under the plan, the looked-up embedding rows, `T` x 4,096 bytes or about 0.1% of the proof, which the byte model leaves out) |
| `v1_ratio_2048` | 2.3 | `v1/v1_bytes.py --model llama2-7b --seq 2048 --modes Kpre` on 2 and 3 blocks (tier `v1`): v0's claims over V1's (2.28 and 2.32) |
| `verifier_s["64"]`, `["2048"]` (`int`, `fs`) | 1.01 / 1.4, 18.52 / 19.64 s | `costs/perf.py ... --modes Kpre --verifier-device cpu` (tier `cpuv`), with `--no-fs` and `--fs`: `verify_s` of the native CPU verifier |
| `v1_verifier_s_2048` | 45.4 s | `v1/v1_phases.py`: V1's CPU verifier in Kpre under Fiat-Shamir with the native kernels of its final check, 2.28 s for 2 blocks plus 30 x 1.44 s for the other blocks |
| `reexec_s["64"]`, `["2048"]` | 4.08 s (`int_fp32gemm`), 142 s (the integer forward pass) | `reexec/reexec.py --model llama2-7b --seq <T> --device cpu --threads 8` (tier `cpuv`): the full model's time of exact re-execution (`exact`: the integer forward pass, its faster GEMM path); the inexact `fp32` and `dynint8_not_exact` are kept for `tab:options` note a |

Mode C's fixed part is not an input: the byte model gives it for the plan, the security level and the rate.

#### A note on the paper's prompt-length crossovers

From these inputs the byte model puts one v0 proof at the size of the weights at 2,111 tokens (Fiat-Shamir; 2,122
interactive) and one V1 proof at 4,854 tokens; `\BreakevenTokens` and `\BreakevenTokensVOne` round them to two
significant digits, 2,100 and 4,900.

## Reproducing the paper's numbers

The serving overhead comes from the run package's `serve` tier (`bash code/experiments/package/run.sh --platform <name> serve`, `experiments/package/README.md`):
one job per table model (OPT-1.3B, Qwen3-4B, Llama-2-7B, OPT-6.7B, Llama-2-13B) of

```bash
python -m experiments.deployment.serve_ab --model <model> --workload a b --modes C Kpre --batch-proof seq joint \
    --reps 10 --device cuda --threads 8 --u2-dtype fp16 bf16 --setup-cache <folder> --label serve
```

on the platform's GPU; `paper/numbers.py` reads its `row: "prove"` lines (the serving table and the `Serve*` macros).
The break-even needs no GPU: `python -m experiments.deployment.ev_breakeven --figure
../paper_sp2027/figures/ev_breakeven.pdf` after updating `ev_inputs.json` from the final run's `cpuv` and `v1` records
(the fields of the table above).

A small version runs on a laptop CPU in about a minute (no downloads, no GPU):

```bash
cd code
export PYTHONPATH=src:.
python -m experiments.deployment.serve_ab --model tiny-llama --cells 32x1 32x4 --batch-proof seq joint --reps 3 \
    --device cpu --threads 2
python -m experiments.deployment.ev_breakeven --out /tmp/ev_breakeven
```

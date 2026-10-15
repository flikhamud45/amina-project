# Verification against re-execution

A client that holds the weights (mode K), or a Kpre client that could keep them instead of its key, could check an
answer by running the model again; Sec. `sec:eval-reexec` compares the verifier's time with that. `reexec.py` times
the re-execution baselines on the same hardware as the verifier (our exact integer forward pass on three GEMM
implementations, and off-the-shelf floating-point and int8 decoders of the same shapes), and `h2d_bandwidth.py`
measures the host-to-device copy bandwidth that bounds a GPU client's time from below.

| Paper result | Script | Command (from `code/`) | Output |
|---|---|---|---|
| `tab:options`' re-execution rows (CPU: exact re-execution, the integer forward pass, on the verifier's 8 threads; the inexact engines in its note a), the abstract's and Sec. `sec:eval-reexec`'s ratio of the CPU verifier to it (with `costs/perf.py`'s verifier) | `reexec.py` (tier `cpuv`) | `python -m experiments.reexec.reexec --model llama2-7b --seq 64 2048 --layers 2 32 --variants int int_fp32gemm int_mm float bf16 dynint8 --device cpu --threads 8 --reps 5` (and Qwen3-4B, OPT-1.3B) | stdout, one JSON line per model, `T` and variant |
| `tab:options`' GPU row and `tab:eval-reexec`'s GPU rows (the integer forward pass with int8 GEMMs / the fp16 decoder, full models) | `reexec.py --device cuda` (tier `gpuv`) | `python -m experiments.reexec.reexec --model llama2-7b --seq 64 512 2048 --layers 2 32 --variants int float bf16 --device cuda --reps 5` (and OPT-1.3B, Qwen3-4B, OPT-6.7B, Llama-2-13B) | stdout |
| `tab:eval-reexec`'s CPU rows (fp32 and the integer forward pass of Llama-2-7B) | `reexec.py --device cpu` (tier `cpuv`) | the first command | stdout |
| `tab:eval-long`'s re-execution columns (2,048 to 16,384 tokens) | `reexec.py --max-pos` (tier `long`) | `python -m experiments.reexec.reexec --model llama2-7b --seq 8192 --layers 2 3 --variants int int_fp32gemm int_mm float bf16 dynint8 --device cpu --reps 3 --max-pos 16384` (and 2,048, 4,096, 16,384) | stdout |
| the transfer time of a proof (`ext_discussion.tex`) and `\HtoDPinnedGBs` (`paper/numbers.py`) | `h2d_bandwidth.py` (tiers `smoke`, `gpuv`) | `python -m experiments.reexec.h2d_bandwidth --sizes-mb 64 1024 4096 --reps 10 --label gpuv` | stdout, one JSON line per size, direction and memory (`"kind": "h2d"`) |

`paper/numbers.py` uses only measured full models (`measured_s` at the model's block count, no extrapolation) and
takes exact re-execution as the fastest of `int`, `int_fp32gemm` and `int_mm` (the integer forward pass, bit-exact
with the claims; `int_mm` only where its record says `exact`); `float`, `bf16` and `dynint8` are inexact (another
model of the same shapes) and are reported next to it. Run every command from `code/` with the checkout's library
first on the path (`PYTHONPATH=src:.` on Linux, `PYTHONPATH="src;."` on Windows).
`python -m experiments.reexec.<script> --help` lists every option.

## Scripts

### `reexec.py`

One prompt of `T` tokens (`--seq`, several values; random tokens, `--prompt-seed`) through a random-weight decoder of
the shapes of `pvi.models.decoder.CONFIGS` (or `experiments.common.models.TINY`; `--seed`): the cost depends on the
shapes only. The variants (`--variants`):

* `int`: the integer forward pass of `build_decoder` that the prover runs, bit-exact with the claims, so a complete
  check in itself (the native int8 GEMM on a CPU, `torch._int_mm` on a GPU);
* `int_fp32gemm` (CPU): the same forward pass with the implementation option `native_gemm` off (exact fp32 GEMMs over
  1,024-term chunks);
* `int_mm` (CPU): the same forward pass with the int8 x int8 -> int32 GEMMs of its linear layers on `torch._int_mm`
  (oneDNN; VNNI's `vpdpbusd` where the CPU has it) instead of the native GEMM. On CPUs without VNNI, oneDNN adds pairs
  of u8 x s8 products in int16 (`vpmaddubsw`), which saturates, so every run checks it: the GEMM against int64
  products at the int8 bounds, and at each block count the logits of its warm-up against an untimed pass of the same
  graph's own forward (`int`'s), bit for bit. The record's `exact` is true only when both agree. Where torch has no
  `_int_mm` on the CPU (e.g. torch 2.2), runs without oneDNN, or refuses or falls back on a shape, the variant is
  reported unavailable on stderr, writes no record, and the other variants run;
* `float`: a plain floating-point decoder of the same shapes (fp16 on a GPU, fp32 on the CPU) as an off-the-shelf
  stack would run it (`F.linear`, `scaled_dot_product_attention`), without a position embedding and not bit-exact
  with the provider's model;
* `bf16` (CPU): the same decoder in bf16; `dynint8` (CPU): the same decoder with torch's dynamic int8 Linear (fbgemm),
  a standard int8 engine that is not exact.

**Pruning.** `--prune-last` (the default; `--no-prune-last` to turn it off) applies to every variant: the last block
computes keys and values at every position and its query, attention, output projection, residuals and MLP at the last
position only, and the LM head runs at the last position, as in the verifier's graph (`build_decoder(...,
prune_last=True)`). The integer and the floating-point variants therefore do the same work per block. (The remaining
differences are element-wise: the integer graph adds GPT-2's and OPT's learned position embedding and applies the final
norm before taking the last position, the floating-point decoders after it.)

**Measurement.** Each variant is built at each `--layers` block count and timed `--reps` times after a warm-up
(`experiments.common.records.timed`: CUDA events on a GPU). With `t(L) = a + L b` the record extrapolates to the full
depth (`full_s`); a `--layers` list that includes the model's block count measures the full model (`measured_s`),
which is what the tables use. `--max-pos N` raises a rotary-position model's context to `N` tokens (a model with
learned positions is refused). The runtime flags: `--device`, `--threads` (8) and every implementation option of
`pvi.options`. Re-execution keeps glibc's allocator defaults unless `--malloc-tune`, which applies the CPU verifier's
setting (`pvi.kernels.native.tune_malloc`: freed memory stays in the process) to the whole process on a CPU.

**Records.** One JSON line per model, `T` and variant: `variant`, `dtype`, `device`, `threads`, `reps`,
`prune_last`, `measured_s` and `range_s` (median and `[min, max]` per block count), `per_block_s`, `fixed_s`,
`n_layers`, `full_s`, `max_pos`, `peak` (host and GPU), `malloc_tuned` (whether the process applied `--malloc-tune`'s
setting; glibc's `MALLOC_*` variables, which set the same parameters, are in `env.env_vars`), `gpu`, `cpu`, `host`,
`torch`, `env`; `int_mm` also `exact` and `exact_check` (`bounds`, and `logits` by block count).

**Hardware and time.** The CPU rows run on 8 threads of the verifier's CPU, the GPU rows on the prover's GPU. One run
of the full Llama-2-7B at 2,048 tokens took 93 s (fp32) and 142 s (integer) on 8 threads of a Xeon Silver 4114; the
package gives the CPU variants of one model a job of one to three hours (OPT-1.3B 1 h, Qwen3-4B 2 h, Llama-2-7B 3 h)
with 256 GB of host memory.

`reexec.FloatDecoder` is also the unverified serving baseline U2 of `experiments/deployment/serve_ab.py`.

### `h2d_bandwidth.py`

For each size, one uint8 buffer is copied host-to-device and device-to-host (`--directions`), from pinned and from
pageable host memory (`--memory`), `--reps` times after an untimed copy, each copy timed with CUDA events around a
synchronised `copy_` (`non_blocking` from pinned memory, as the streaming verifier uploads). Records: the median and
`[min, max]` seconds, the median's GB/s (`gb_s`, 10^9 bytes per second), the GPU, its PCIe link (`nvidia-smi`), `env`.
Needs a GPU (exit status 2 without one); seconds.

## Reproducing the paper's numbers

The final numbers come from the run package (`bash code/experiments/package/run.sh --platform <name> <tier>`, `experiments/package/README.md`): its `cpuv` tier runs
`reexec.py --device cpu` with every variant on Llama-2-7B, Qwen3-4B and OPT-1.3B at 64 and 2,048 tokens (2 blocks and
the full model; Llama-2-7B also with the allocator settings the CPU verifier applies to itself), its `gpuv` tier
`reexec.py --device cuda` (`int float bf16`) on the five table models at 64, 512 and 2,048 tokens and
`h2d_bandwidth.py`, its `long` tier `reexec.py --max-pos 16384` at 2,048 to 16,384 tokens (2 and 3 blocks), and its
`smoke` tier one `h2d_bandwidth.py` copy of 256 MiB.

A small version runs on a laptop CPU in seconds:

```bash
cd code
export PYTHONPATH=src:.
python -m experiments.reexec.reexec --model tiny-llama tiny-opt --seq 32 --layers 1 2 \
    --variants int int_fp32gemm int_mm float bf16 dynint8 --device cpu --threads 2 --reps 2
```

`tests/test_reexec.py` checks `int_mm` with exact, saturating, missing and refusing int8 GEMMs in place of
`torch._int_mm`, on torch's own CPU kernel where it has one, and the allocator setting.

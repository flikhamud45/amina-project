# Verification against re-execution

A client that holds the weights (mode K), or a Kpre client that could keep them instead of its key, could check an
answer by running the model again; Sec. `sec:eval-reexec` compares the verifier's time with that. `reexec.py` times
the re-execution baselines on the same hardware as the verifier (our exact integer forward pass on three GEMM
implementations, and off-the-shelf floating-point and int8 decoders of the same shapes), `llamacpp.py` times a
production engine, llama.cpp, on the same CPU threads (inexact, another model), and `h2d_bandwidth.py` measures the
host-to-device copy bandwidth that bounds a GPU client's time from below.

| Paper result | Script | Command (from `code/`) | Output |
|---|---|---|---|
| `tab:options`' re-execution rows (CPU: exact re-execution, the integer forward pass, on the verifier's 8 threads; the inexact engines in its note a), the abstract's and Sec. `sec:eval-reexec`'s ratio of the CPU verifier to it (with `costs/perf.py`'s verifier) | `reexec.py` (tier `cpuv`) | `python -m experiments.reexec.reexec --model llama2-7b --seq 64 2048 --layers 2 32 --variants int int_fp32gemm int_mm float bf16 dynint8 --device cpu --threads 8 --reps 5` (and Qwen3-4B, OPT-1.3B) | stdout, one JSON line per model, `T` and variant |
| `tab:options`' GPU row and `tab:eval-reexec`'s GPU rows (the integer forward pass with int8 GEMMs / the fp16 decoder, full models) | `reexec.py --device cuda` (tier `gpuv`) | `python -m experiments.reexec.reexec --model llama2-7b --seq 64 512 2048 --layers 2 32 --variants int float bf16 --device cuda --reps 5` (and OPT-1.3B, Qwen3-4B, OPT-6.7B, Llama-2-13B) | stdout |
| `tab:eval-reexec`'s CPU rows (fp32 and the integer forward pass of Llama-2-7B) | `reexec.py --device cpu` (tier `cpuv`) | the first command | stdout |
| `tab:options`' llama.cpp row, `tab:eval-reexec`'s llama.cpp rows (Q8_0 and bf16), `\CpuLlamaCpp*`, `\CpuRatioLlamaCpp`, `\LlamaCppRelease` and, through `\CpuInexact*`, the body's inexact-inference sentence | `llamacpp.py` (tier `cpuv`; CMake and git) | `python -m experiments.reexec.llamacpp --model llama2-7b --seq 64 2048 --types q8_0 bf16 --reps 5 --threads 8 --llamacpp-dir ../logs/<name>/llamacpp` | stdout, one JSON line per weight type, `T` and configuration (`"kind": "llamacpp"`) |
| `tab:eval-long`'s re-execution columns (2,048 to 16,384 tokens) and `tab:eval-long-check` | `reexec.py --max-pos` (tier `long`) | `python -m experiments.reexec.reexec --model llama2-7b --seq 8192 --layers 2 3 4 --variants int int_mm int_fp32gemm --device cpu --reps 3 --max-pos 16384` and `--layers 2 3 --variants float bf16 dynint8` (and 2,048, 4,096, 16,384); the full model at 8,192 tokens with `--layers 32 --reps 1` | stdout |
| `tab:eval-long-gpu`'s re-execution columns | `reexec.py --device cuda --max-pos` (tier `gpuv`) | `python -m experiments.reexec.reexec --model llama2-7b --seq 2048 4096 8192 16384 --layers 2 3 --variants int float bf16 --device cuda --reps 3 --max-pos 16384` | stdout |
| no number of the paper: the host-to-device bandwidth of the run's GPU, a record of the platform (the text's transfer time of a proof, `\TransferLlamaLong`, is the proof's size over a 10 Gb/s link) | `h2d_bandwidth.py` (tiers `smoke`, `gpuv`) | `python -m experiments.reexec.h2d_bandwidth --sizes-mb 64 1024 4096 --reps 10 --label gpuv` | stdout, one JSON line per size, direction and memory (`"kind": "h2d"`) |

`paper/numbers.py` uses only measured full models (`measured_s` at the model's block count, no extrapolation) and
takes exact re-execution as the fastest of `int`, `int_fp32gemm` and `int_mm` (the integer forward pass, bit-exact
with the claims; `int_mm` only where its record says `exact`); `float`, `bf16` and `dynint8`, and llama.cpp's
`llamacpp_q8_0` and `llamacpp_bf16` (each at its fastest configuration), are inexact (another model of the same
shapes) and are reported next to it. The abstract's ratio stays against exact re-execution. Run every command from
`code/` with the checkout's library first on the path (`PYTHONPATH=src:.` on Linux, `PYTHONPATH="src;."` on
Windows).
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
`prune_last`, `prob_max` (the integer variants' attention probabilities' scale, `--prob-max`, default 2047: the
cost measurements' 11 bits), `measured_s` and `range_s` (median and `[min, max]` per block count), `per_block_s`,
`fixed_s`, `n_layers`, `full_s`, `max_pos`, `peak` (host and GPU), `malloc_tuned` (whether the process applied
`--malloc-tune`'s setting; glibc's `MALLOC_*` variables, which set the same parameters, are in `env.env_vars`), `gpu`,
`cpu`, `host`, `torch`, `env`; `int_mm` also `exact` and `exact_check` (`bounds`, and `logits` by block count).

**Hardware and time.** The CPU rows run on 8 threads of the verifier's CPU, the GPU rows on the prover's GPU. One run
of the full Llama-2-7B at 2,048 tokens took 93 s (fp32) and 142 s (integer) on 8 threads of a Xeon Silver 4114; the
package gives the CPU variants of one model a job of one to three hours (OPT-1.3B 1 h, Qwen3-4B 2 h, Llama-2-7B 3 h)
with 256 GB of host memory.

`reexec.FloatDecoder` is also the unverified serving baseline U2 of `experiments/deployment/serve_ab.py`.

### `llamacpp.py`

llama.cpp, a production CPU inference engine, re-executes a model of Llama-2-7B's shapes on the same 8 threads. It
runs its own Q8_0 or bf16 weights, so it is not exact and computes a different model: its records say so (`exact`
and `same_model` false), `paper/numbers.py` fails any that does not, and it never stands for exact re-execution.

* **Build.** Release v0.6.0 (build b11429, commit `d8123504`), cloned into `--llamacpp-dir`'s `src/` (or
  `--llamacpp-src`, which must be at that commit) and built with CMake (`--cmake`, default `$CMAKE` or `cmake`) for
  this CPU (`GGML_NATIVE`, i.e. `-march=native`; CPU only, without HTTPS, the server, tests and examples) with
  `--cxx` (default `$CXX`) and `--cc` (default `$CC`, else the C compiler next to `--cxx`), `--jobs` 8: `llama-bench`
  and `llama-quantize` only, into `build/`. A stamp file (the release, the commit, the CMake flags, CMake, the
  compilers and the CPU) lets later jobs reuse the build; every record keeps the build's `llama_print_system_info`
  (its CPU features, e.g. `AVX512_VNNI` and `AVX512_BF16`).
* **Weights.** Random weights of the model's exact shapes (`pvi.models.decoder.CONFIGS`; Llama-2's architecture only,
  any other model is refused; `--layers N`: N blocks), normal with standard deviation 0.02 and norms of one
  (`--seed`), written as a bf16 GGUF by the clone's own `gguf-py` one tensor at a time, with a dummy vocabulary of
  Llama-2's SentencePiece layout (32,000 tokens: `<unk>`, `<s>`, `</s>`, the 256 byte tokens and unique fillers), then
  quantised to Q8_0 by `llama-quantize`. This needs numpy and PyYAML (which `gguf-py` imports), but no checkpoint,
  sentencepiece, protobuf or torch. The files
  go to a new folder under `--tmp-dir` (default `$TMPDIR`), which is deleted at exit: about 21 GB for Llama-2-7B
  (13.5 GB bf16, 7.2 GB Q8_0), 1.4 GB for one block; the script stops before writing them if they do not fit.
* **Measurement.** `llama-bench -p T -n 0 -t <--threads> -r <--reps> -b T -ub <--ubatch, capped at T> -fa
  <--flash-attn> -lm none -ngl 0 -o jsonl`: prompt processing only, one untimed warm-up pass and `--reps` timed passes
  of random token ids per configuration (each attention kernel, `off` and `on`, and each micro-batch size, 512 and
  2,048), the weights read into the process's memory, the default CPU mask and no `--numa`, so that its OpenMP
  threads stay on the CPUs the package pinned the job to. A pass computes the next-token logits at the last position;
  unlike the verifier's pruned graph, llama.cpp computes the last block's attention at every position. The tables
  use each weight type's fastest configuration. llama-bench's own output is read by the script, never copied to the
  job's log.
* **Records.** One JSON line per weight type, `T` and configuration (`"kind": "llamacpp"`): `variant`
  (`llamacpp_q8_0`, `llamacpp_bf16`), `dtype`, `exact` and `same_model` (false), `weights` (`random`), `seed`,
  `model`, `seq`, `threads`, `reps`, `layers` (blocks built), `n_layers` (the model's), `measured_s` (the median,
  keyed by the blocks built, so that a one-block build never fills a full-model cell), `range_s`, `samples_s`,
  `full_s`, `flash_attn`, `n_batch`, `n_ubatch`, `gguf_bytes`, `gguf_params`, `peak`, `env`, and `llamacpp` (the
  release, commit, build number, CMake flags, compilers, system information, llama-bench's command line and its own
  fields). A failed step prints `FAILED: llama.cpp <step>: ...` and the exit status is 1.
* **Requirements and time.** CMake 3.14 or later, git, a C and C++ compiler and, for the clone, network access to
  GitHub (or a clone made beforehand in `--llamacpp-dir`'s `src/`). The first build takes about 5 to 10 minutes; the
  full cell (Q8_0 and bf16 at 64 and 2,048 tokens, four configurations each) about one to one and a half hours on 8
  Zen 4 cores, with up to about 16 GB of memory (the package: 3 h, 64 GB).

### `h2d_bandwidth.py`

For each size, one uint8 buffer is copied host-to-device and device-to-host (`--directions`), from pinned and from
pageable host memory (`--memory`), `--reps` times after an untimed copy, each copy timed with CUDA events around a
synchronised `copy_` (`non_blocking` from pinned memory, as the streaming verifier uploads). Records: the median and
`[min, max]` seconds, the median's GB/s (`gb_s`, 10^9 bytes per second), the GPU, its PCIe link (`nvidia-smi`), `env`.
Needs a GPU (exit status 2 without one); seconds.

## Reproducing the paper's numbers

The final numbers come from the run package (`bash code/experiments/package/run.sh --platform <name> <tier>`, `experiments/package/README.md`): its `cpuv` tier runs
`reexec.py --device cpu` with every variant on Llama-2-7B, Qwen3-4B and OPT-1.3B at 64 and 2,048 tokens (2 blocks and
the full model; Llama-2-7B also with the allocator settings the CPU verifier applies to itself) and `llamacpp.py` on
Llama-2-7B's shapes (Q8_0 and bf16, 64 and 2,048 tokens; the rehearsal: one block at 64 tokens), its `gpuv` tier
`reexec.py --device cuda` (`int float bf16`) on the five table models at 64, 512 and 2,048 tokens and
`h2d_bandwidth.py` and on Llama-2-7B's 2 and 3 blocks at 2,048 to 16,384 tokens (`--max-pos 16384`), its `long` tier
`reexec.py --max-pos 16384` at 2,048 to 16,384 tokens (the exact variants on 2, 3 and 4 blocks, the others on 2 and
3; the exact ones on the full model at 8,192 tokens), and its
`smoke` tier one `h2d_bandwidth.py` copy of 256 MiB.

A small version runs on a laptop CPU in seconds:

```bash
cd code
export PYTHONPATH=src:.
python -m experiments.reexec.reexec --model tiny-llama tiny-opt --seq 32 --layers 1 2 \
    --variants int int_fp32gemm int_mm float bf16 dynint8 --device cpu --threads 2 --reps 2
```

`tests/test_reexec.py` checks `int_mm` with exact, saturating, missing and refusing int8 GEMMs in place of
`torch._int_mm`, on torch's own CPU kernel where it has one, and the allocator setting. `tests/test_llamacpp.py`
checks `llamacpp.py` without building llama.cpp: Llama-2-7B's tensor plan and parameter count, the vocabulary, a tiny
GGUF read back by gguf-py's reader (with a gguf-py: an installed `gguf` or a clone under
`logs/<name>/llamacpp/src`), records from a stand-in for llama-bench, its command line, and the refusal of other
architectures.

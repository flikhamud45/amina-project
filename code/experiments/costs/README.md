# Cost of the defence

These scripts measure what protocol v0 costs (Sec. `sec:eval-cost` and the extended version's evaluation and
defence appendices): the prover's and the verifiers' times and the proof's size in modes C, K and Kpre, the outcome of
every honest query and every attack, the path test it replaces, the one-time setup proof of an untrusted commitment,
and the concrete security parameters. `bench.py` writes the stored benchmark cells that `comparison/aggregate.py`
turns into tables; `perf.py` times variants of the protocol in one process for the run package's tiers; the other
scripts compute byte models from the public shapes or train the image models.

| Paper result | Script | Command (from `code/`) | Output |
|---|---|---|---|
| `tab:eval-llm` (prove, CPU and GPU verify, proof; modes C and Kpre), `tab:eval-compare` ("ours") | `bench.py` | `python -m experiments.costs.bench llm --model llama2-7b --seq 64 --builds full --modes C:fs --lams 128 --queries 5 --llm-tampers 30 --platform <name>` (and the other jobs of the run package's `bench` tier: `bash code/experiments/package/run.sh --platform <name> --dry-run bench`) | `artifacts/comparison/raw_<name>/`, tables by `comparison/aggregate.py` |
| Sec. `sec:eval-setup`: "accepted all 3,792 honest queries and rejected all 2,512 attacks", with the check that rejected each | `bench.py` (defence and tamper cells) | as above; then `python -m experiments.comparison.count_outcomes --platform <name>` | `raw_<name>/`, counts |
| Sec. `sec:break`: one penultimate neuron flips every test query of each image model; the path test detects it with probability 1/84 to 1/512; `app:eval-attack`: 305,193 paths for `1 - 2^-40` | `bench.py cnn` (cells `attack_float`, `sampling`) | `python -m experiments.costs.bench cnn --model resnet18_224 --tampers 60 --platform <name>` | `raw_<name>/cnn/<model>/{attack_float,sampling}*.jsonl` |
| `tab:gen` (the stored generated-response cells) | `bench.py --gen` | `python -m experiments.costs.bench llm --model gpt2 --seq 64 --gen 256 --builds full --modes C:int --platform <name>` | `raw_<name>/llm/gpt2/*_gen256*` (the final run uses `generation/gen_bench.py`) |
| `tab:options` (CPU verifier against re-execution), `tab:eval-reexec`'s verifier row, `tab:memory` (the client's own peak memory) | `perf.py` (tier `cpuv`) | `python -m experiments.costs.perf --model llama2-7b --seq 2048 --modes Kpre C --queries 5 --verifier-device cpu --label cpuv --client-memory` | stdout, one JSON line per mode |
| `tab:eval-cpu` (the CPU verifier on 2 and 4 threads, without the AVX-512 attention kernel, Llama-2-13B; against exact re-execution on the same threads), `\CpuRatioThrTwo` and the other macros of `ext_evaluation.tex`'s `app:eval-cpu`; `tab:eval-reexec`'s verifier at 512 tokens (tier `cpuv`) | `perf.py` (tier `cpux`) | `python -m experiments.costs.perf --threads 4 --model llama2-7b --seq 2048 --modes Kpre --queries 5 --verifier-device cpu --label cpux` (on 4 pinned cores; 2 threads likewise); `... --threads 8 --no-native-avx512 --label cpux-noavx512`; `... --model llama2-13b --label cpux` | stdout |
| `tab:eval-plans` (mode C's six candidate plans on the same queries), `\PlanCols*`, `\PlanCommit*` | `perf.py` (tier `cpux`) | `python -m experiments.costs.perf --model llama2-7b --seq 2048 --modes C --policy cnn18c tightc cnn16c cnn17c R16c R64c --queries 3 --verifier-device cpu --label plans` (no `--setup-cache`: `commit_s` is the encoding) | stdout, one JSON line per plan |
| `tab:eval-lambda` (Llama-2-7B at lambda 40, 80 and 128), `\LamForty*` | `bench.py` (tier `bench`) | `python -m experiments.costs.bench llm --model llama2-7b --seq 2048 --builds full --queries 3 --lams 40 80 --modes C:fs,Kpre:fs --platform <name>` (and 64 tokens with 5 queries; lambda 128: the cells of `tab:eval-llm`) | `raw_<name>/`, tables by `comparison/aggregate.py` |
| `tab:eval-llm`'s GPU-verifier column | `perf.py` (tier `gpuv`) | `python -m experiments.costs.perf --model opt-1.3b --seq 2048 --modes Kpre --queries 3 --verifier-device cuda --stream --label gpuv` | stdout |
| `tab:eval-real` (real weights against random twins) | `perf.py --real` (tier `real`) | `python -m experiments.costs.perf --model opt-1.3b --real facebook/opt-1.3b --seq 2048 --modes Kpre C --queries 3 --tampers 3 --label real` and the same without `--real` (`--label real-random`) | stdout |
| `tab:eval-long` (2,048 to 16,384 tokens) and `tab:eval-long-check` | `perf.py` (tier `long`) | `python -m experiments.costs.perf --model llama2-7b --layers 2 --seq 8192 --modes Kpre --queries 5 --verifier-device cpu --label long` (2, 3 and 4 blocks at every length; the full model at 8,192 tokens without `--layers`) | stdout |
| `tab:eval-long-gpu` (the GPU verifier at 2,048 to 16,384 tokens) | `perf.py` (tier `gpuv`) | `python -m experiments.costs.perf --model llama2-7b --layers 2 --seq 16384 --modes Kpre --queries 3 --verifier-device cuda --stream --label long-gpuv` (2 and 3 blocks at every length) | stdout |
| `tab:eval-improve` rows of the prover's attention and of the GPU and CPU verifiers | `perf.py` with implementation options | `experiments/improvements/README.md` | stdout |
| `tab:setup`; the setup-proof breakdown after it (`ext_defence.tex`); `tab:params`' setup-proof size and untrusted columns | `setup_bytes.py` | `python -m experiments.costs.setup_bytes` | stdout, one JSON line per model and `a` |
| the setup proof's prover and verifier times (`ext_defence.tex`, after `tab:setup`) | `setup_timing.py` (tier `setup`) | `python -m experiments.costs.setup_timing --model llama2-7b --amortise 100 --label setup`; `... --model gpt2 --amortise 100 1000 --reps 3` | stdout |
| `tab:params` (`app_params.tex`) | `params_table.py` | `python -m experiments.costs.params_table --tex` | stdout |
| "16–19 bits for a claim" (`ext_v1.tex`, Sec. on V1's proof size) | `wire.py` | `python -m experiments.costs.wire --mnist <dir> lenet5 vgg16 resnet18_cifar mlp_mnist gpt2:64 gpt2:512 qwen3-4b:8:1,2,4,8 llama2-7b:64:1,2`; stored (measured on the 8-bit graphs, `--prob-max 255`): `python -m experiments.costs.wire --out artifacts/results/development/wire_laptop.json --summary` | `artifacts/results/development/wire.json` |
| the trained image models of the benchmark | `train_cnn.py` | `python -m experiments.costs.train_cnn --model lenet5 --device cuda` (and `vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel`) | `artifacts/models/cnn/<model>.{pt,json}` |

Run every command from `code/` with the checkout's library first on the path (`PYTHONPATH=src:.` on Linux,
`PYTHONPATH="src;."` on Windows). `python -m experiments.costs.<script> --help` lists every option.

## Scripts

### `bench.py`

The benchmark of protocol v0 and of the path test (Anchuri et al., re-implemented on the same integer graphs) on the
trained image models (`cnn` suite: `mlp_mnist`, `lenet5`, `vgg11`, `vgg16`, `resnet18_cifar`, `resnet18_224`) and on
decoders with the shapes of the paper's language models (`llm` suite: the names of `pvi.models.decoder.CONFIGS`, and
the small shapes `tiny-*`/`small-llama` of `experiments.common.models.TINY`), on random weights or, with `--real`, on a
checkpoint's (the run package passes `--real` wherever its rule gives a cell a checkpoint: `package/README.md`).

* **Inputs.** The image models need their trained weights (`train_cnn.py`; the MLP is
  `artifacts/models/mlp_mnist_full.npz`, stored) and datasets: MNIST (downloaded by torchvision), CIFAR-10 and the
  ImageNet classes of the 224-pixel models as `pvi.models.datasets` reads them. The decoders need nothing, unless
  `--real` names a Hugging Face checkpoint (below).
* **Cells.** Image models: `facts` (parameters, float and int8 accuracy), `attack_float` (the single-neuron flip on
  the float model), `sampling` (the path test: detection probability per neuron, bytes for `k` paths), `commit_rate4`
  (a plan's commitment and setup size), `defence_<mode>_<int|fs>_lam<l>_rate4` (honest queries: time and bytes per
  part, acceptance; the `C:int` cells also time batches of 8 and 32 images), `tamper_C_int_lam40` (seven attacks,
  with the check that rejected each). Decoders: `commit_T<seq>_L<blocks>`, `defence_<mode>_<int|fs>_lam<l>_T<seq>_L<b>`
  (each also runs one tampered query that must be rejected) and, with `--llm-tampers N` on a full build,
  `tamper_C_int_lam40_T<seq>_L<b>` (N each of a random claim entry changed, the top logit forged, the last hidden
  coordinate set out of range, and max(3, N/10) blocks run with 1%-perturbed weights).
* **Main options.** `--platform <name>` (required: the records go to `artifacts/comparison/raw_<name>/`, which holds
  one prover GPU and one CPU model; the stored roots are frozen), `--modes` (e.g. `C:fs,Kpre:fs`), `--lams`, `--seq`,
  `--builds` (`auto`, `full`, `1,2,full`), `--queries`, `--tampers`, `--llm-tampers`, `--batches`, `--cnn-batches`,
  `--gen`, `--real`. The protocol settings default to the paper's configuration (`experiments.common.cli.PAPER`): `--policy
  cnn18c`, `--wire`, `--lean`, `--prune-last`, `--lookups`, each with a `--no-...` form (`--policy per-op` for one
  tree per weight op), `--prob-max 2047` (the decoders' 11-bit attention probabilities; 255: the 8-bit graphs),
  `--rate 4`, `--setup-cache DIR`. The runtime flags: `--device` (the prover's), `--verifier-device
  cpu|cuda`, `--verifier-impl default|stream` (the streaming verifier), `--threads 8`, `--tf32`, and every
  implementation option of `pvi.options` (`--no-native`, `--no-fused-attention`, ...; recorded in the `env` row).
* **Cell names follow the settings that change the cell**, in this order: `_thr<n>` (threads other than 8),
  `_nolean`, `_tf32`, `_wire`, `_gpuv`, `_stream`, `_batch` (batch sizes other than the defaults), `_prune` (every
  decoder cell), `_lookups` (decoder K and Kpre cells), `_pmax<n>` (decoder cells whose attention probabilities'
  scale is not the stored cells' 255: `_pmax2047` by default), `_gen<G>`, `_pol<name>` (commitment and mode-C cells
  under a plan other than `per-op`), `_real` (decoder cells of a checkpoint's weights, `--real`). A cell measured under the same settings therefore has the same name as the
  stored one: checked against every stored cell of `artifacts/comparison/raw_*`. The names depend on the effective settings,
  not on which flags were given. The defaults are the paper's configuration, with 8 verifier threads, the batched
  verifier on the CPU, TF32 off, and the batch sizes 8 and 32 (image models) and none (decoders). A stored `_polauto` cell (the planning rule, which picks `cnn16c`, `cnn17c` or
  `cnn18c` per model) is distinct from a `_polcnn18c` cell (the paper's fixed plan).
* **Records.** One JSON object per line in `raw_<name>/<suite>/<model>/<cell>.jsonl`: a first row `metric: env` (the
  run's environment: commit, host, CPU, GPU, versions, the implementation options under `implementation`), then rows
  `{run_id, suite, model, cell, config, prover_hw, verifier_hw, host, git_sha, metric, value, unit, trial, ts}` with
  `batch`, `attack`, `stage`, `lam`, `k` and `contention` (the scheduler evidence of each timed phase) where they
  apply. A finished cell has a `.done` marker and is skipped on a re-run; an honest query that is rejected stops the
  job and keeps the cell's records as `<cell>.rejected-<run_id>.jsonl`.
* **Real weights.** `--real <checkpoint>` (`--model` names its shapes; e.g. `--model llama2-7b --real
  meta-llama/Llama-2-7b-hf`): the decoder cells on the checkpoint's integer graph, built as `perf.py --real` builds it
  (`experiments.common.real_cells`: OPT with `hf_opt.CERTIFIED_RECIPE`, Llama-family checkpoints, Qwen3 included,
  with `hf_llama.MAIN_RECIPE` and its fixed public prefix), queried on the WikiText-2 test split's windows from its
  start. A build of `n` blocks is the checkpoint's first `n`, the full graph's first `n` (the residual stream's scale is
  set on the whole checkpoint), for OPT and Llama-family checkpoints alike. The cells are named `_real`, their `config` holds `real` and `recipe`, and the commitment cell records
  `load_checkpoint` (loading the checkpoint in fp32 and its calibration). Refused: the image models, GPT-2 (no integer
  builder), a checkpoint of other shapes than `--model`'s, a prompt (behind a Llama-family recipe's fixed prefix)
  longer than the checkpoint's trained context, and an attention probabilities' scale other than the paper's. Without `--real` nothing changes.
* **Measurement notes.** All the cells of one invocation share a prover: when an invocation runs mode C and modes K or
  Kpre, the K and Kpre cells are measured with the commitment resident on the GPU (their peak memory includes it).
  The paper's K and Kpre cells come from jobs of their own (`--modes Kpre:fs`), as the run package runs them. A decoder build whose int8 weights exceed 85% of the GPU's memory is skipped, and the job then fails.
* **Hardware and time.** The paper's cells ran on an NVIDIA L40S (48 GB) prover with 8 threads of an AMD EPYC 9334 as
  the CPU verifier. An image model or GPT-2 takes minutes; a 7B or 13B decoder at 2,048 tokens one to two hours and up
  to 128 GB of host memory. With `--real` the checkpoint is also held in fp32 until the job's last build (27 GB for
  Llama-2-7B and OPT-6.7B, 52 GB for Llama-2-13B, 16 GB for Qwen3-4B), and its calibration adds tens of minutes.

### `perf.py`

The timing harness of the run package's `cpuv`, `cpux`, `gpuv`, `real` and `long` tiers. It builds the model as `bench.py`
does (or, with `--real`, a checkpoint's integer graph: OPT with the certified recipe `hf_opt.CERTIFIED_RECIPE` by
default, `--recipe legacy --smooth 0.55 --calib-seq 2048` for the percentile recipe; a Llama-family checkpoint with
`hf_llama.MAIN_RECIPE`), runs one warm-up and `--queries` queries per variant and prints one JSON line per mode and
variant: the per-part medians, the prover and verifier totals with their `[min, max]`, the proof bytes, the
commitment's time, the peak host and GPU memory, `env`. `--policy`, `--wire`, `--lean`, `--prune-last`,
`--lookups` and `--prob-max` take several values (e.g. `--wire off on`; `--prob-max 255 2047`: the 8- and 11-bit
graphs of the same weights, the improve tier's A/B behind `quality.tex`'s "change the proof's size by at most"), and
the variants' queries are interleaved. `--stream` selects
the streaming verifier (with `--verifier-device cuda`: the paper's GPU verifier). `--tampers n` runs `n` of each
trace-tampering attack per variant (`experiments.common.tampers`). Real checkpoints are downloaded from Hugging Face
(OPT-1.3B/6.7B, Llama-2-7B under its licence) with the WikiText-2 test split. The package gives a 7B decoder at 2,048
tokens a job limit of one to two hours and up to 256 GB of host memory (the CPU verifier holds the decoded claims).

GPU memory. The lean prover with the compact encoding keeps each claim on the GPU as int32 until it encodes them all:
4 bytes a claim, 1.3 GiB per Llama-2-7B block at 8,192 tokens, 40.6 GiB for the full model next to its 6.3 GiB of
weights, more than an L40S's 44.5 GiB. The option `--host-claims` (`pvi.options`, every script) decides where they
wait: `auto` (the default) keeps them all on the GPU whenever the claims, the weights and `--host-claims-reserve` GiB
(default 6: the forward pass's and the encoder's temporaries, measured at 2.9 GiB on Llama-2-7B blocks at 8,192
tokens) fit in the GPU's memory -- every cell of the run but
the few below, unchanged -- and otherwise keeps on the GPU the claims that fit beside the weights and the reserve and
copies the others into pinned host memory as they are computed; the GPU's encoder then uploads those part by part.
The bytes, the proof and the transcripts are the same; the copies add to `prove_forward` and `prove_encode`.
`device` keeps every claim on the GPU (the path before the option), `host` stages every claim. The record's
`claims_staged` gives the bytes staged (0: none). On an L40S, auto stages in the long tier's full model at 8,192
tokens (`l-llama7-8192-full`: 32 GiB of claims kept on the GPU, 8.4 GiB staged), in `bench.py`'s batch of 4 prompts of 2,048 tokens
(`bt-llama2-7b-2k-*`, the same claims) and in `serve_ab.py`'s joint proof of 4 prompts of 2,048 tokens of Llama-2-7B,
OPT-6.7B and Llama-2-13B (which would otherwise run out of memory, OPT-6.7B by a few hundred MiB).

The prover and the verifier share the process, so its peak memory is theirs together. `--client-memory` (a CPU
verifier, the compact encoding; mode C under Fiat-Shamir) measures the client's own: a separate process
(`experiments/common/client.py`, spawned per mode after the warm-up) holds only the public graph, the parameters and
mode C's roots or mode Kpre's key and embedding rows, and checks every timed proof again from its bytes
(`pvi.v0.query.verify_wire`), right after the timed query while the measuring process waits. The record's
`client_memory` holds that process's peak resident set, its resident set after the imports and after loading the key
(Linux), and whether every verdict equals the measuring process's (a difference, or a failed client process, is a
`FAILED` line and exit status 1). The key and each proof travel through files in a temporary folder under
`--client-dir` (default: the system's temporary folder; 6.4 GB for one Llama-2-7B proof at 2,048 tokens). The `cpuv`
tier passes it on Llama-2-7B's cells (`tab:memory`).

### `setup_bytes.py`

The byte model of the setup proof (`pvi.commit.setup_proof`) from the public shapes of GPT-2, Llama-2-7B, OPT-6.7B
and Llama-2-13B under the plan `cnn18c` at `a = 100` and `1,000` amortised queries (interactive queries, as
`tab:setup`): the setup proof's size and its parts (opened columns, the combined rows, a bound on the paths), the
per-query columns under the honest and the untrusted bound, and two quantities of the remark on other proximity
tests (`columns_MB_at_lam`: the columns at the same radii with an error of `2^-lambda` per matrix; per tree
`t0_with_replacement`). `--fixed-bytes --policy cnn16c cnn18c R16c R64c --model gpt2 opt-1.3b llama2-7b llama2-13b
qwen3-4b` prints instead the fixed part of a mode-C proof (`u`, opened columns, paths) and the one-time encoded
entries per plan: the comparison behind the choice of the paper's plan, which the paper does not print
(`--entries-per-s` adds a setup-time estimate at a measured encoding rate). Seconds on a laptop.

### `setup_timing.py`

The setup proof's prover (`prove_setup` on the GPU) and verifier (`verify_setup` on the CPU) time on a random-weight
decoder (the last block pruned) under `--policy` (default `cnn18c`), for each `--amortise`: one JSON line with
`kind: setup`, the commitment time, the prover and verifier times (median and range over `--reps`), the proof bytes,
the parameters (`s`, `t0`, `t`), the per-query columns and peak memory. The commitment is never read from the setup
cache. `--fs` sizes the per-query columns for Fiat-Shamir queries (default interactive, as `tab:setup`). `--real
<checkpoint>` (`--model` names its shapes) times the setup proof of the checkpoint's integer graph instead, built as
`perf.py --real` builds it (the checkpoint's first `--layers` blocks, 0: every block); the line then holds `real`
and `recipe` (both None for random weights), and `build_s` includes loading the checkpoint in fp32 and its
calibration. GPT-2 has no integer builder of a real checkpoint and is refused. The package gives GPT-2 a job limit of
one hour and Llama-2-7B one of three hours with 128 GB of host memory.

### `params_table.py`

Every value of `tab:params` (Llama-2-7B, the plan `cnn18c`, `lambda = 128`): v0's `|J|`, `beta`, `r`, the Freivalds
term, the trees' `t` and column bytes (interactive and Fiat-Shamir); V1's lookup and GKR terms at 2,048 tokens,
extrapolated from two measured V1 cells (`--v1-counts blocks:N_leaf:instances:table`, default Llama-2-7B's 2- and
3-block cells, or `--v1-cells` with `v1/v1_bytes.py` records); the setup proof's `beta_0`, `s`, per tree `f`, `e`,
`t0` and `t`, and its size and untrusted per-query columns at each `--amortise`. One JSON line; `--tex` adds the values
as text lines. Seconds on a laptop.

### `wire.py`

The compact encoding of the proof against the uncoded one: bits per claim (the "16–19 bits" of `ext_v1.tex`), proof
bytes per part with and without the encoding, verifier and prover times, and the codec's throughput, on image models
(random weights, MNIST digits from `--mnist`, read without downloading; `mlp_mnist`: the trained MLP) and decoders
(`decoder:tokens[:blocks]`). Mode C is sized under the per-op commitment (the plan does not change the encoding's
effect), Kpre at `--lam`, Qwen3-4B at 40 (`--kpre-lam`, Maverick's setting). The results go to `--out` (default
`artifacts/results/development/wire.json`); the stored measurement is `artifacts/results/development/wire_laptop.json`
(`--summary` prints its tables). On a laptop CPU the stored specs took from 1 s (the MLP) to 191 s (GPT-2 at 512
tokens).

### `train_cnn.py`

Trains one image model (`--model`, the recipes of `RECIPES`: SGD with Nesterov momentum and a one-cycle schedule, seed
0) on its dataset and writes `artifacts/models/cnn/<model>.pt` and `<model>.json` (accuracy, recipe, seed, time,
`env`). Retrained weights are equally accurate but not bit-identical to the stored runs'; costs depend on the shapes
only. Minutes for LeNet-5, a few hours for the 224-pixel ResNets on one GPU.

## Reproducing the paper's numbers

The final numbers come from one platform, through the run package (`bash code/experiments/package/run.sh --platform <name> <tier>`; its
README is `experiments/package/README.md`):

* `bench`: the `bench.py` jobs of `tab:eval-llm`, `tab:eval-compare`, `tab:eval-lambda` and the outcome counts (each
  job is one `bench.py` invocation with `--platform <name>`); `batch`: the batch cells.
* `cpuv`, `cpux`, `gpuv`, `real`, `long`: the `perf.py` invocations above, one process per model, length and mode
  (the cpux tier's plans: one process for all six, on the same queries).
* `setup`: `setup_timing.py` on Llama-2-7B (`a = 100`) and GPT-2 (`a = 100, 1,000`).

`setup_bytes.py`, `params_table.py` and `wire.py` need no GPU and reproduce `tab:setup`, `tab:params` and the bits per
claim on any machine. The tables of the benchmark cells are made by `comparison/aggregate.py --platform <name>` and
`paper/numbers.py`.

A small version runs on a laptop CPU in minutes (no downloads, no GPU):

```bash
cd code
export PYTHONPATH=src:.
python -m experiments.costs.bench llm --model tiny-gpt --seq 16 --builds full --queries 2 --lams 40 \
    --llm-tampers 2 --device cpu --platform laptop
python -m experiments.costs.bench llm --model tiny-gpt --seq 16 --builds full --queries 2 --lams 40 \
    --modes Kpre:fs --device cpu --platform laptop
python -m experiments.costs.perf --model tiny-llama --seq 16 --modes C Kpre --queries 3 --device cpu --threads 2
python -m experiments.costs.setup_timing --model tiny-llama --device cpu --threads 2
python -m experiments.costs.setup_bytes && python -m experiments.costs.params_table --tex
```

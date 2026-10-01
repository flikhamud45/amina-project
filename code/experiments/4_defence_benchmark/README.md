# 4. The benchmark: our defence vs. the path test (report §4.3–4.4)

Measures our defence (`pvi.fullcheck`) and the original path test, re-implemented on the
same integer graphs (`pvi.fullcheck.sampling`), on the models the literature benchmarks.
The prover runs on a GPU and the verifier on the CPU (or, with `--verifier-device cuda`
and a `_gpuv` tag, on the prover's GPU: §4.4). Nothing is aggregated here:
every measurement is appended, one JSON line per trial, to
`artifacts/comparison/raw_<platform>/<suite>/<model>/<cell>.jsonl`, where the platform
(`--platform` or `$PVI_PLATFORM`) names one GPU and CPU model. The report's numbers are
the root `raw_l40s/`; `raw_rtx2080ti-v2/` is the second platform, `raw/` the earliest
run, with an older version of the code, and `raw_l40s_improved/` the options added after the
report (below) on the report's hardware. All four are frozen (see `code/README.md`, *The
stored benchmark runs*), so a re-measurement uses a new platform name.

**1. Train the CNNs** (GPU; minutes for LeNet-5, a few hours for the 224-pixel ResNets):

```bash
python experiments/4_defence_benchmark/train.py --model lenet5 --device cuda
# models: lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel
```

Writes `artifacts/fullcheck/models/<model>.pt` and a JSON report (accuracy, epochs).
These weights are not committed. The MNIST MLP is `artifacts/models/mlp_mnist_full.npz`,
which is committed (see `experiments/0_train_models`).

**2. Run the benchmark** for one model (these are the report's settings; the `must`,
`should` and `nice` tiers of `slurm/strong_gpu.sh` list every job of the report):

```bash
export PVI_PLATFORM=<new name>     # raw/, raw_rtx2080ti-v2/, raw_l40s/ and raw_l40s_improved/ are the stored runs
python experiments/4_defence_benchmark/bench.py cnn --model lenet5 --queries 30 --tampers 100
python experiments/4_defence_benchmark/bench.py llm --model gpt2 --seq 64 128 256 512 --queries 10 --lean
python experiments/4_defence_benchmark/bench.py llm --model llama2-7b --seq 64 --builds 1,2,full --queries 10 \
    --lams 128 --modes C:int,Kpre:int --llm-tampers 30 --lean
```

`--builds 1,2,full` builds the decoder with 1 and 2 blocks and at full depth (the default
builds decoders above 12 blocks only with 1 and 2 blocks); a build that does not fit the
GPU (Llama-2-13B in full on an 11 GB card, Llama-2-70B on 48 GB) is skipped with a `SKIP` line, and the job then
ends with an error after its other builds. `--lean` streams the claims to host memory and
frees dead tensors: the same integers and proof, far less GPU memory (Llama-2-7B at
2,048 tokens peaks at 7.3 GiB of GPU memory and 42 GiB of host memory).

| Cell (CNN suite) | What it measures |
|---|---|
| `facts` | parameters, float and int8 accuracy |
| `attack_float` | the single-neuron flip on the float model |
| `sampling` | the path test: exact detection per neuron, bytes for k paths (shared openings) |
| `defence_<mode>_<challenges>_lam<λ>_rate4` | honest queries: time per part, proof bytes by part, acceptance (modes C, K, Kpre; interactive or Fiat–Shamir; λ = 40, 80, 128). The `defence_C_int_*` cells also run one query of 8 and one of 32 images in a single interaction (`--cnn-batches`, default 8 32); no table or figure uses their timings, but they are honest queries of §4.3's counts. The `defence_C_int_*_batch` cells of `nice` time 8 to 256 images (8 to 128 for `resnet18_224`), 5 trials each |
| `tamper_C_int_lam40` | seven attacks, with the check that rejected each one |

The LLM suite has `commit_*` cells (the one-time weight commitment) and `defence_*` cells
per prompt length and build (`_T<tokens>_L<blocks>`). Each defence cell also runs one
tampered query (one pre-activation of a middle layer changed by +1) that must be
rejected. With `--batches B ...`, the C:int and Kpre:int cells also time B prompts against
one set of openings (the `_batch` cells of `nice`: GPT-2 and Llama-2-7B at 64 tokens, 8
and 32 prompts, 3 trials each; §4.4 quotes Llama-2-7B with 8). With `--llm-tampers N`,
every full build also gets a `tamper_C_int_lam40_<build>` cell of attacks on the full
model: N each of `single_value` (one random pre-activation changed by ±1 to ±999),
`output_logit` (a wrong token given the top logit) and `last_hidden` (one coordinate of
the last block's output projection at the last position set to a large value), and
max(3, N/10) `substituted_block_1pct` (one random block with 1% of its weights changed by
±1). The report uses N = 30 at ≤ 64 tokens.

Options. Both suites: `--queries` (honest queries per cell), `--threads` (verifier
threads), `--tag` (a cell-name suffix for variant runs: `_gpuv`, `_batch`, `_thr1`,
`_thr12`, `_tf32`, `_nolean`, `_nofix`), `--verifier-device`, `--tf32`, `--batch-trials`
(repetitions per batch size) and `--platform`. CNN suite: `--tampers` (attacks per attack
type; 0 skips the tamper cell, for timing-only controls) and `--cnn-batches`. LLM suite:
`--seq` (prompt lengths), `--lams` and `--modes` (e.g. `C:int,Kpre:int`), `--builds`,
`--lean`, `--llm-tampers` and `--batches`; the CNN suite always runs every mode at λ = 40,
80 and 128. The Reed–Solomon rate is 4 throughout (also as the base rate of a `--policy`
plan, below). A cell with a `.done` marker is skipped, so an interrupted job can simply be
restarted.

The report's variant runs (the exact lines are in `slurm/strong_gpu.sh`):
`--verifier-device cuda` runs the client's checks on the prover's GPU (`_gpuv`, §4.4);
`--batches 8 32` (LLM suite) or `--cnn-batches` (CNN suite), with `--batch-trials`, time
several prompts or images against one set of openings (`_batch`; §4.4 quotes the LLM
ones); `--tf32` allows TF32 tensor cores in the float32 products of the forward pass
(`_tf32`; `bench.sbatch` needs `PVI_TF32=1` for it); `--threads 1` and `12` (`PVI_THREADS`
under `bench.sbatch`) give `_thr1` and `_thr12`; the LLM jobs without `--lean` are
`_nolean`; and `PVI_LEGACY_WEIGHT_KEY=1` restores the earliest run's weight cache, which
re-uploaded the committed weights to the GPU twice per query (`_nofix`). `bench.py`
refuses `--verifier-device cuda` or `--tf32` without the matching tag, and
`PVI_LEGACY_WEIGHT_KEY=1` without a `_nofix` tag, or such a tag without it.

**Options added after the report** (`experiments/6_improvements`, summarised in
`IMPROVEMENTS.md` at the repository root). Without them `bench.py` runs exactly the report's
protocol: the same verdicts, proof bytes, Fiat–Shamir transcripts and Merkle roots. Only the
timings differ from the report's runs, since the verifier engineering of `IMPROVEMENTS.md` is
always on (see *Route B* in `code/README.md`).

* `--policy <name>`: commit under a commitment plan of `pvi.fullcheck.plans` over the base
  rate 4 (exact column counts, Merkle trees shared by the matrices of one codeword length,
  per-op codeword lengths): `tight`, `cnn<e>` or `R<rate>`, each also with the suffix `c` (wide
  matrices committed transposed, q/k/v and gate/up fused, embedding tables as Merkle lookup
  tables), or `auto` (the `c` plan of fewest non-claim bytes within a setup budget, recorded as
  the cells' `policy`). Only the commitment (`commit_*`, with its setup time and size; for the
  CNNs `commit_rate4_pol<name>`) and the mode-C cells run, named with a `_pol<name>` suffix.
* `--wire` (tag `_wire`): the proof in the compact, lossless encoding of
  `pvi.fullcheck.claimcodec` (`prove_encode`, `verify_decode` and the encoded `bytes_*`).
* `--verifier-impl stream` (tag `_stream`; with `--verifier-device cuda`, `_gpuv_stream`): the
  streaming verifier, the same verdicts, which records one `verify_total` per query instead of
  the verify phases it overlaps.
* LLM suite: `--prune-last` (suffix `_prune`) builds the last block at the last position only;
  `--lookups` (suffix `_lookups`) runs the K and Kpre cells with a verifier that reads the
  embedding rows itself.

`bench.py` adds the `_pol`, `_prune` and `_lookups` suffixes itself (and refuses them in
`--tag`), refuses `--wire` or `--verifier-impl stream` without their tags, and `--lookups`
together with `--policy`. Their L40S runs are `raw_l40s_improved/`, whose job list is in
`IMPROVEMENTS.md`.

An honest query that is rejected stops the job and keeps that cell's records as
`<cell>.rejected-<run_id>.jsonl`, which `count_outcomes.py` counts. `gpu_peak_memory` is
the prover GPU's peak over a cell's queries (with `--verifier-device cuda`, prover and
client together), and `host_peak_rss` the peak host memory of the job's one process
(prover and verifier) over the same queries. `bench.py` refuses to run when the imported
`pvi` is not this checkout's `src/pvi` (for example a shared venv with another clone
installed editable): `export PYTHONPATH=<checkout>/code/src`, as the sbatch scripts do.

The stored roots also hold metrics that no table or figure reads (in `raw/`, some
written by earlier versions of `bench.py`); they only enter the record totals (123,832 in
`raw_l40s/`, 78,266 in `raw_rtx2080ti-v2/`, 59,547 in `raw/`).

**On the TAU cluster** (from the repository root), `slurm/train.sbatch` and
`slurm/bench.sbatch` wrap the two commands above and pass their arguments through, and
`slurm/strong_gpu.sh must`, `should` and `nice` submit every benchmark job of the report
(`smoke` first, once per GPU type: `slurm/smoke.sbatch` runs the tests on the GPU, small
benchmarks into a throw-away root and `experiments/5_comparison/fingerprint_check.py`
against `raw/`). Train first and let that job finish, since the CNN jobs load the trained
weights (the report's are in the team's project folder on the cluster, not in this
repository; do not retrain them to reproduce the report):

```bash
mkdir -p logs
sbatch -o logs/%x-%j.out code/experiments/4_defence_benchmark/slurm/train.sbatch lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel
# when it has finished (squeue -u $USER):
export PVI_PLATFORM=<new name> SBATCH_PARTITION=killable SBATCH_GRES=gpu:l40s:1
bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh smoke   # wait for SMOKE OK
bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh must    # then should, then nice
# or a single job:
sbatch -o logs/$PVI_PLATFORM/%x-%j.out code/experiments/4_defence_benchmark/slurm/bench.sbatch cnn --model vgg16
```

The report's numbers are the root `raw_l40s/`: every job of the three tiers (the script at
commit `9401431` also submitted a kernel microbenchmark, `microbench`, which writes only a
log and is not in this version), with `PVI_PLATFORM=l40s` on `killable` (an L40S prover,
8 threads of an AMD EPYC 9334 verifier, nodes n-801..804; the one L40S node with another
CPU, t-806, was excluded by an `sbatch` wrapper that adds `--exclude=t-806`), run from a
clean clone at commit `9401431`. Pre-empted jobs were resubmitted by re-running their
tier. The second platform, `raw_rtx2080ti-v2/`, is the `must` jobs except `ab-opt13` (too
large for the card) and `microbench` (no records): 29 SLURM jobs, 945088–945116, on
`studentbatch` (RTX 2080 Ti, Xeon Silver 4114) from the same commit. Both roots are frozen
(`"frozen": true` in their `PLATFORM.json`), as is `raw_l40s_improved/`. After a re-measurement, run
`experiments/5_comparison` with `--platform <new name>` to turn the raw records into the
tables and figures.

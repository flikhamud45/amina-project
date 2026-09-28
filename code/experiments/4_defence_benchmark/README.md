# 4. The benchmark: our defence vs. the path test (report §4.3–4.4)

Measures our defence (`pvi.fullcheck`) and the original path test, re-implemented on the
same integer graphs (`pvi.fullcheck.sampling`), on the models the literature benchmarks.
The prover runs on a GPU and the verifier on the CPU. Nothing is aggregated here:
every measurement is appended, one JSON line per trial, to
`artifacts/comparison/raw_<platform>/<suite>/<model>/<cell>.jsonl`, where the platform
(`--platform` or `$PVI_PLATFORM`) names one GPU and CPU model. The report's numbers are
the root `raw_rtx2080ti-v2/`; `raw/` is the earlier run with an older version of the code
and is frozen (see `code/README.md`, *The stored benchmark runs*).

**1. Train the CNNs** (GPU; minutes for LeNet-5, a few hours for the 224-pixel ResNets):

```bash
python experiments/4_defence_benchmark/train.py --model lenet5 --device cuda
# models: lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel
```

Writes `artifacts/fullcheck/models/<model>.pt` and a JSON report (accuracy, epochs).
These weights are not committed. The MNIST MLP is `artifacts/models/mlp_mnist_full.npz`,
which is committed (see `experiments/0_train_models`).

**2. Run the benchmark** for one model (these are the report's settings; the `must` tier
of `slurm/strong_gpu.sh` lists them all):

```bash
export PVI_PLATFORM=<name>     # a new name: raw/ and raw_rtx2080ti-v2/ are the stored runs
python experiments/4_defence_benchmark/bench.py cnn --model lenet5 --queries 30 --tampers 100
python experiments/4_defence_benchmark/bench.py llm --model gpt2 --seq 64 128 256 512 --queries 10 --lean
python experiments/4_defence_benchmark/bench.py llm --model llama2-7b --seq 64 --builds 1,2,full --queries 10 \
    --lams 128 --modes C:int,Kpre:int --llm-tampers 30 --lean
```

`--builds 1,2,full` builds the decoder with 1 and 2 blocks and at full depth (the default
builds decoders above 12 blocks only with 1 and 2 blocks); a build that does not fit the
GPU (Llama-2-13B in full on an 11 GB card) is skipped with a `SKIP` line, and the job then
ends with an error after its other builds. `--lean` streams the claims to host memory and
frees dead tensors: the same integers and proof, far less GPU memory (Llama-2-7B at
2,048 tokens peaks at 7.3 GiB of GPU memory and 42 GiB of host memory).

| Cell (CNN suite) | What it measures |
|---|---|
| `facts` | parameters, float and int8 accuracy |
| `attack_float` | the single-neuron flip on the float model |
| `sampling` | the path test: exact detection per neuron, bytes for k paths (shared openings) |
| `defence_<mode>_<challenges>_lam<λ>_rate4` | honest queries: time per part, proof bytes by part, acceptance (modes C, K, Kpre; interactive or Fiat–Shamir; λ = 40, 80, 128) |
| `tamper_C_int_lam40` | seven attacks, with the check that rejected each one |

The LLM suite has `defence_*` cells per prompt length and build (`_L<blocks>`). Each
defence cell also runs one tampered query (one pre-activation of a middle layer changed
by +1) that must be rejected. With `--llm-tampers N`, every full build also gets a
`tamper_C_int_lam40_<build>` cell of attacks on the full model: N each of `single_value`
(one random pre-activation changed by ±1 to ±999), `output_logit` (a wrong token given
the top logit) and `last_hidden` (one coordinate of the last block's output projection
at the last position set to a large value), and max(3, N/10) `substituted_block_1pct`
(one random block with 1% of its weights changed by ±1). The report uses N = 30 at
≤ 64 tokens.

Options. Both suites: `--queries` (honest queries per cell), `--rate` (Reed–Solomon
rate, default 4), `--threads`, `--tag` (a suffix for variant runs), `--force` (redo
finished cells). CNN suite: `--tampers` (attacks per attack type). LLM suite: `--seq`
(prompt lengths), `--lams` and `--modes` (e.g. `C:int,Kpre:int`); the CNN suite always
runs every mode at λ = 40, 80 and 128. A cell with a `.done` marker is skipped, so an
interrupted job can simply be restarted.

`--tampers 0` skips the CNN tamper cell (timing-only controls). An honest query that is
rejected stops the job and keeps that cell's records as `<cell>.rejected-<run_id>.jsonl`,
which `count_outcomes.py` counts. `gpu_peak_memory` is the prover's GPU; with the client
on the same GPU (`--verifier-device cuda`) it is prover and client together, and with the
client on another GPU (`cuda:1`) the client's peak is recorded as `gpu_peak_memory_verifier`.
`bench.py` refuses to run when the imported `pvi` is not this checkout's `src/pvi` (for
example a shared venv with another clone installed editable): `export
PYTHONPATH=<checkout>/code/src`, as the sbatch scripts do.

The stored roots also hold `commit_*` cells (the one-time weight commitment) and metrics
that no table or figure reads (in `raw/`, some written by earlier versions of
`bench.py`); they only enter the record totals (78,266 in `raw_rtx2080ti-v2/`, 59,547 in
`raw/`).

**On the TAU cluster** (from the repository root), `slurm/train.sbatch` and
`slurm/bench.sbatch` wrap the two commands above and pass their arguments through, and
`slurm/strong_gpu.sh must` submits every benchmark job of the report (`smoke` first, once
per GPU type; `slurm/sweep.sh` is the job list of the earlier run `raw/`, with today's
code). Train first and let that job finish, since the CNN jobs load the trained weights
(the report's are already on the cluster; do not retrain them to reproduce the report):

```bash
mkdir -p logs
sbatch -o logs/%x-%j.out code/experiments/4_defence_benchmark/slurm/train.sbatch lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel
# when it has finished (squeue -u $USER):
export PVI_PLATFORM=<name> SBATCH_PARTITION=studentbatch SBATCH_GRES=gpu:geforce_rtx_2080:1
bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh smoke   # wait for SMOKE OK
bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh must
# or a single job:
sbatch -o logs/$PVI_PLATFORM/%x-%j.out code/experiments/4_defence_benchmark/slurm/bench.sbatch cnn --model vgg16
```

The report's numbers are the root `raw_rtx2080ti-v2/`: exactly these `must` jobs except
`ab-opt13` (the non-lean OPT-1.3B control at 2,048 tokens, about 27 GiB, too large for the
card; it was not run), with `PVI_PLATFORM=rtx2080ti-v2` on `studentbatch` (an RTX 2080 Ti
prover, 8 threads of a Xeon Silver 4114 verifier), run from a clean clone at commit
`9401431`. On that 11 GB card the full Llama-2-13B build was skipped as too large, so the
report extrapolates Llama-2-13B from its 1- and 2-block builds. The root is now frozen
(`"frozen": true` in its `PLATFORM.json`). Afterwards, run `experiments/5_comparison`
with `--platform rtx2080ti-v2` to turn the raw records into the tables and figures.

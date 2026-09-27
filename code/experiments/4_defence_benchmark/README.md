# 4. The benchmark: our defence vs. the path test (report §4.3–4.4)

Measures our defence (`pvi.fullcheck`) and the original path test, re-implemented on the
same integer graphs (`pvi.fullcheck.sampling`), on the models the literature benchmarks.
The prover runs on a GPU and the verifier on the CPU. Nothing is aggregated here:
every measurement is appended, one JSON line per trial, to
`artifacts/comparison/raw/<suite>/<model>/<cell>.jsonl`.

**1. Train the CNNs** (GPU; minutes for LeNet-5, a few hours for the 224-pixel ResNets):

```bash
python experiments/4_defence_benchmark/train.py --model lenet5 --device cuda
# models: lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel
```

Writes `artifacts/fullcheck/models/<model>.pt` and a JSON report (accuracy, epochs).
These weights are not committed. The MNIST MLP is `artifacts/models/mlp_mnist_full.npz`,
which is committed (see `experiments/0_train_models`).

**2. Run the benchmark** for one model (these are the report's settings; `slurm/sweep.sh`
lists them all):

```bash
python experiments/4_defence_benchmark/bench.py cnn --model lenet5 --queries 30 --tampers 100
python experiments/4_defence_benchmark/bench.py llm --model gpt2 --seq 64 128 256 512 --queries 10
```

| Cell (CNN suite) | What it measures |
|---|---|
| `facts` | parameters, float and int8 accuracy |
| `attack_float` | the single-neuron flip on the float model |
| `sampling` | the path test: exact detection per neuron, bytes for k paths (shared openings) |
| `defence_<mode>_<challenges>_lam<λ>_rate4` | honest queries: time per part, proof bytes by part, acceptance (modes C, K, Kpre; interactive or Fiat–Shamir; λ = 40, 80, 128) |
| `tamper_C_int_lam40` | seven attacks, with the check that rejected each one |

The LLM suite has `defence_*` cells per prompt length. Each defence cell
also runs one tampered query that must be rejected.

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

The stored `raw/` also holds `commit_*` cells and extra metrics written by earlier
versions of `bench.py`; no table or figure reads them, they only enter the total of
59,547 records.

**On the TAU cluster** (from the repository root), `slurm/train.sbatch` and
`slurm/bench.sbatch` wrap the two commands above and pass their arguments through, and
`slurm/sweep.sh` submits every benchmark job of the report. Train first and let that
job finish, since the CNN jobs load the trained weights:

```bash
mkdir -p logs
sbatch -o logs/%x-%j.out code/experiments/4_defence_benchmark/slurm/train.sbatch lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel
# when it has finished (squeue -u $USER):
bash code/experiments/4_defence_benchmark/slurm/sweep.sh
# or a single job:
sbatch -o logs/%x-%j.out code/experiments/4_defence_benchmark/slurm/bench.sbatch cnn --model vgg16
```

The report's numbers come from an RTX 2080 Ti prover and an 8-thread Xeon Silver 4114
verifier. Afterwards, run `experiments/5_comparison` to turn the raw records into the
tables and figures.

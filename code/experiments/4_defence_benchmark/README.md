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
The MNIST MLP comes from `experiments/0_train_models`.

**2. Run the benchmark** for one model:

```bash
python experiments/4_defence_benchmark/bench.py cnn --model lenet5
python experiments/4_defence_benchmark/bench.py llm --model gpt2 --seq 64 128 256 512
```

| Cell (CNN suite) | What it measures |
|---|---|
| `facts` | parameters, float and int8 accuracy, CPU re-run time |
| `attack_float` | the single-neuron flip on the float model |
| `sampling` | the path test: one path's cost, exact detection per neuron, bytes for k paths |
| `commit_rate4` | the one-time weight commitment |
| `defence_<mode>_<challenges>_lam<λ>_rate4` | honest queries: time per part, proof bytes by part, acceptance (modes C, K, Kpre; interactive or Fiat–Shamir; λ = 40, 80, 128) |
| `tamper_C_int_lam40` | seven attacks, with the check that rejected each one |

The LLM suite has `commit_*` and `defence_*` cells per prompt length. Each defence cell
also runs one tampered query that must be rejected. Useful options: `--lams`, `--modes`
(e.g. `C:int,Kpre:int`), `--queries`, `--tag` (a suffix for variant runs), `--only`
(re-run some cells), `--force` (redo finished cells). A cell with a `.done` marker is
skipped, so an interrupted job can simply be restarted.

**On the TAU cluster** (from the repository root), `slurm/sweep.sh` submits every job
of the report. `slurm/train.sbatch` and `slurm/bench.sbatch` wrap the two commands above
and pass their arguments through:

```bash
sbatch -o logs/%x-%j.out code/experiments/4_defence_benchmark/slurm/bench.sbatch cnn --model vgg16
bash code/experiments/4_defence_benchmark/slurm/sweep.sh
```

The report's numbers come from an RTX 2080 Ti prover and an 8-thread Xeon Silver 4114
verifier. Afterwards, run `experiments/5_comparison` to turn the raw records into the
tables and figures.

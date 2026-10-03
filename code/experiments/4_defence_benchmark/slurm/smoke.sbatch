#!/bin/bash
#SBATCH --job-name=fc-smoke
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=01:30:00
# Mandatory before any long run, once per GPU type: environment dump, every test (the GPU
# exactness tests at the sizes of the real models, with TF32 genuinely off and on;
# tests/test_real_weights.py only where transformers is installed), a data/weights check, small
# benchmarks into a throw-away raw_smoke<job>/ root (including one lean full-depth build), and
# the fingerprint comparison against the stored records of raw/. From the repository root:
#   mkdir -p logs && sbatch -o logs/%x-%j.out code/experiments/4_defence_benchmark/slurm/smoke.sbatch
# The throw-away root is moved to logs/env/<job>.raw_smoke when the job ends (pass or fail), so
# it never stays among the raw_*/ platform roots and never lands in git.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR/code"
umask 002
export PYTHONPATH="$PROJECT_DIR/code/src${PYTHONPATH:+:$PYTHONPATH}"   # this checkout's pvi, whatever the venv has
export OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE NVIDIA_TF32_OVERRIDE   # the tests toggle TF32 themselves
PY="${PVI_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
mkdir -p "$PROJECT_DIR/logs/env"
E="$PROJECT_DIR/logs/env/$SLURM_JOB_ID"
# informational only: never fatal (a foreign-owned checkout, or no .git at all)
git_() { git -c safe.directory='*' -c core.fileMode=false -C "$PROJECT_DIR" "$@" 2>&1 || true; }
nvidia-smi -q > "$E.nvidia-smi.txt"; nvidia-smi topo -m > "$E.topo.txt" 2>&1 || true
lscpu > "$E.lscpu.txt"; "$PY" -m pip freeze > "$E.pip-freeze.txt" 2>&1 || true; env | sort > "$E.environ.txt"
git_ rev-parse HEAD > "$E.git.txt"
git_ status --porcelain --untracked-files=no | tee -a "$E.git.txt" || true
"$PY" - <<'PY' | tee "$E.torch.txt"
import pathlib, sys, torch
import pvi
here, got = pathlib.Path("src/pvi").resolve(), pathlib.Path(pvi.__file__).resolve().parent
assert got == here, f"pvi is imported from {got}, not from this checkout ({here})"
assert torch.cuda.is_available(), "no GPU visible in this job"
p = torch.cuda.get_device_properties(0)
print(sys.version.split()[0], torch.__version__, "cuda", torch.version.cuda, "cudnn", torch.backends.cudnn.version(),
      "|", torch.cuda.get_device_name(0), f"sm_{p.major}{p.minor}", f"{p.total_memory / 2**30:.0f} GiB",
      "| arch list", torch.cuda.get_arch_list(), "| pvi", got)
PY
export PVI_TEST_DEVICE=cuda   # test_gpu_exactness.py: never an inherited "cpu"
# without transformers (experiments/2_attack/REAL_LLM.md) tests/test_real_weights.py would be skipped: leave it out
IGN=(); "$PY" -c "import transformers" 2>/dev/null || IGN=(--ignore=tests/test_real_weights.py)
"$PY" -m pytest tests ${IGN[@]+"${IGN[@]}"} -q -rs -p no:cacheprovider \
      --junitxml="$E.pytest.xml" 2>&1 | tee "$E.pytest.txt"
# every test must have run: no failure, no error, no skip (e.g. a GPU test that found no GPU)
"$PY" - "$E.pytest.xml" <<'PY'
import sys
import xml.etree.ElementTree as ET
r = ET.parse(sys.argv[1]).getroot()
s = r if r.tag == "testsuite" else r.find("testsuite")
bad = {k: int(s.get(k, 0)) for k in ("failures", "errors", "skipped")}
if any(bad.values()) or int(s.get("tests", 0)) == 0:
    sys.exit(f"FAIL: pytest {bad} of {s.get('tests', 0)} tests")
print(f"pytest: {s.get('tests')} tests, none failed, errored or skipped")
PY

# the required inputs (train the CNNs first with train.sbatch): trained CNNs, ImageNet caches, CIFAR-10 ($PVI_CIFAR_ROOT), MNIST
for m in lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel; do
  test -f "artifacts/fullcheck/models/$m.pt" || { echo "FAIL: missing artifacts/fullcheck/models/$m.pt"; exit 1; }
done
"$PY" -c "from pvi.fullcheck.datasets import load_dataset
for n in ('mnist', 'cifar10', 'imagenet_dogs_cats', 'imagenet_dogs_squirrels'):
    print(n, tuple(load_dataset(n)[2].shape))" | tee "$E.data.txt"

export NVIDIA_TF32_OVERRIDE=0 PVI_PLATFORM="smoke$SLURM_JOB_ID"
R="$PROJECT_DIR/code/artifacts/comparison/raw_smoke$SLURM_JOB_ID"
trap 'mv "$R" "$E.raw_smoke" 2>/dev/null || true' EXIT   # out of the raw* namespace, records kept
"$PY" experiments/4_defence_benchmark/bench.py cnn --model lenet5 --queries 3 --tampers 3 --threads 8
"$PY" experiments/4_defence_benchmark/bench.py llm --model gpt2 --seq 64 --queries 3 --lams 40 128 --modes C:int,Kpre:int --threads 8
"$PY" experiments/4_defence_benchmark/bench.py llm --model llama2-7b --seq 2048 --queries 2 --lams 128 --modes C:int --threads 8
# the lean full-depth path (memory records, tamper cell) on this GPU, before any long run
"$PY" experiments/4_defence_benchmark/bench.py llm --model opt-1.3b --seq 8 --builds 1,full --queries 1 --lams 40 \
      --modes C:int,Kpre:int --llm-tampers 1 --lean --threads 8
grep -q '"gpu_peak_reserved"' "$R"/llm/opt-1.3b/defence_C_int_lam40_T8_L24.jsonl || { echo "FAIL: no GPU memory records"; exit 1; }
"$PY" experiments/5_comparison/fingerprint_check.py artifacts/comparison/raw "$R"
echo "SMOKE OK"

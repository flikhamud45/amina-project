#!/bin/bash
#SBATCH --job-name=fc-bench
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=06:00:00
# One benchmark invocation; arguments go straight to experiments/4_defence_benchmark/bench.py.
# Partition, account and GPU type come from the environment (SBATCH_PARTITION,
# SBATCH_ACCOUNT, SBATCH_GRES=gpu:<type>:1 -- they override the lines above) or from
# sbatch's command line. From the repository root:
#   export PVI_PLATFORM=<name>   # a new name: records go to code/artifacts/comparison/raw_<name>/
#   mkdir -p logs/$PVI_PLATFORM && sbatch -o logs/$PVI_PLATFORM/%x-%j.out code/experiments/4_defence_benchmark/slurm/bench.sbatch llm --model gpt2 --seq 64
# PVI_THREADS (default 8, the report's; not --cpus-per-task, which a site policy may raise) sets
# the verifier's threads; it is appended as --threads.  The pvi package is always this
# checkout's code/src (PYTHONPATH), even with a shared venv that has another clone's pvi installed.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR/code"
umask 002
: "${PVI_PLATFORM:?export PVI_PLATFORM=<name> before sbatch (raw/ is frozen; records go to raw_<name>/)}"
export PVI_PLATFORM
export PYTHONPATH="$PROJECT_DIR/code/src${PYTHONPATH:+:$PYTHONPATH}"
THREADS="${PVI_THREADS:-8}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
export NVIDIA_TF32_OVERRIDE=0            # no TF32 in any cuBLAS/cuDNN call of this process
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE   # set by NGC containers; would change torch's TF32 default
PY="${PVI_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
# informational only: never fatal (a foreign-owned checkout, or no .git at all)
git_() { git -c safe.directory='*' -c core.fileMode=false -C "$PROJECT_DIR" "$@" 2>&1 || true; }
CORES=$("$PY" -c "import os; print(len({tuple(open(f'/sys/devices/system/cpu/cpu{c}/topology/{k}').read().strip()
    for k in ('physical_package_id', 'core_id')) for c in os.sched_getaffinity(0)}))" 2>/dev/null || echo '?')
echo "host $(hostname)  job $SLURM_JOB_ID  platform $PVI_PLATFORM  threads $THREADS on $CORES physical cores  git $(git_ rev-parse --short HEAD)"
git_ status --porcelain --untracked-files=no | head -5 || true
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit,mig.mode.current,compute_mode --format=csv || true
lscpu | grep -E '^(Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\))' || true
echo "cpu affinity: $(taskset -cp $$ 2>/dev/null || true)"
"$PY" experiments/4_defence_benchmark/bench.py "$@" --threads "$THREADS"

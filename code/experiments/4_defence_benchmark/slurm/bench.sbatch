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
#   export PVI_PLATFORM=<name>   # e.g. h100: records go to code/artifacts/comparison/raw_<name>/
#   mkdir -p logs/$PVI_PLATFORM && sbatch -o logs/$PVI_PLATFORM/%x-%j.out code/experiments/4_defence_benchmark/slurm/bench.sbatch llm --model gpt2 --seq 64
# PVI_THREADS (default: --cpus-per-task) sets the verifier's threads; it is appended as --threads.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR/code"
umask 002
: "${PVI_PLATFORM:?export PVI_PLATFORM=<name> before sbatch (raw/ is frozen; records go to raw_<name>/)}"
export PVI_PLATFORM
THREADS="${PVI_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export OMP_NUM_THREADS="$THREADS" MKL_NUM_THREADS="$THREADS"
if [ "${PVI_TF32:-0}" = 1 ]; then          # only for the _tf32 variant (bench.py --tf32)
  unset NVIDIA_TF32_OVERRIDE
else
  export NVIDIA_TF32_OVERRIDE=0          # no TF32 in any cuBLAS/cuDNN call of this process
fi
unset TORCH_ALLOW_TF32_CUBLAS_OVERRIDE   # set by NGC containers; would force TF32 back on
PY="${PVI_PYTHON:-$PROJECT_DIR/.venv/bin/python}"
echo "host $(hostname)  job $SLURM_JOB_ID  platform $PVI_PLATFORM  threads $THREADS  git $(git -C "$PROJECT_DIR" rev-parse --short HEAD)"
git -C "$PROJECT_DIR" status --porcelain --untracked-files=no | head -5
nvidia-smi --query-gpu=name,driver_version,memory.total,clocks.max.sm,power.limit,mig.mode.current,compute_mode --format=csv || true
lscpu | grep -E '^(Model name|Thread\(s\) per core|Core\(s\) per socket|Socket\(s\)|NUMA node\(s\))' || true
echo "cpu affinity: $(taskset -cp $$ 2>/dev/null || true)"
"$PY" experiments/4_defence_benchmark/bench.py "$@" --threads "$THREADS"

#!/bin/bash
#SBATCH --job-name=fc-bench
#SBATCH --partition=studentbatch
#SBATCH --gres=gpu:geforce_rtx_2080:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=2-00:00:00
# One benchmark invocation; arguments go straight to experiments/4_defence_benchmark/bench.py.
# From the repository root:
#   sbatch -o logs/%x-%j.out code/experiments/4_defence_benchmark/slurm/bench.sbatch llm --model gpt2 --seq 64
# Every measurement lands in code/artifacts/comparison/raw/; finished cells are
# skipped on re-submission, so a pre-empted job can simply be submitted again.
# The verifier runs on this job's CPUs (threads = --cpus-per-task).
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR/code"
umask 002
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}" MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"
"$PROJECT_DIR/.venv/bin/python" experiments/4_defence_benchmark/bench.py "$@" --threads "${SLURM_CPUS_PER_TASK:-8}"

#!/bin/bash
#SBATCH --job-name=fc-train
#SBATCH --partition=studentbatch
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=12:00:00
# Train the float models of the comparison benchmark, one after another:
#   sbatch -o logs/%x-%j.out code/experiments/4_defence_benchmark/slurm/train.sbatch lenet5 vgg16 ...
# sbatch reads #SBATCH lines only until the first command, so they come first;
# override the partition with `sbatch -p ...`.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR/code"
umask 002
export PYTHONPATH="$PROJECT_DIR/code/src${PYTHONPATH:+:$PYTHONPATH}"   # this checkout's pvi, whatever the venv has
for model in "$@"; do
  echo "=== $model  ($(date))"
  "$PROJECT_DIR/.venv/bin/python" experiments/4_defence_benchmark/train.py --model "$model" --device cuda
done

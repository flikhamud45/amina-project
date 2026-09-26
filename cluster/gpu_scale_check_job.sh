#!/bin/bash
#SBATCH --partition=studentbatch
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=gpu_scale_check_job-%j.out
# sbatch reads #SBATCH lines only until the first command, so they come first.
# Override the partition on the command line, e.g. `sbatch -p gpu-bermano ...`.
# Portable: set PROJECT_DIR to your checkout, or rely on the default.
PROJECT_DIR="${PROJECT_DIR:-$HOME/amina-project}"

VENV=${PROJECT_DIR}/.venv
cd ${PROJECT_DIR}/code

"$VENV/bin/python" scripts/run_gpu_scale_check.py --device cuda --queries 30 --force
echo "SCALE_CHECK_JOB_DONE"

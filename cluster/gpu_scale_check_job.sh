#!/bin/bash
# Portable: set PROJECT_DIR to your checkout, or rely on the default.
PROJECT_DIR="${PROJECT_DIR:-$HOME/amina-project}"
#SBATCH --partition=gpu-bermano
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=gpu_scale_check_job-%j.out

VENV=${PROJECT_DIR}/.venv
cd ${PROJECT_DIR}/code

"$VENV/bin/python" scripts/run_gpu_scale_check.py --device cuda --queries 30 --force
echo "SCALE_CHECK_JOB_DONE"

#!/bin/bash
#SBATCH --partition=studentbatch
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --output=batched_large_job-%j.out
# sbatch reads #SBATCH lines only until the first command, so they come first.
# Override the partition on the command line, e.g. `sbatch -p gpu-bermano ...`.
# Portable: set PROJECT_DIR to your checkout, or rely on the default.
PROJECT_DIR="${PROJECT_DIR:-$HOME/amina-project}"

VENV=${PROJECT_DIR}/.venv
cd ${PROJECT_DIR}/code
"$VENV/bin/python" scripts/run_batched_network.py --model large --queries 5
echo "LARGE_NETWORK_DONE"

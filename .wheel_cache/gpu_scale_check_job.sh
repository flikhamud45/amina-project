#!/bin/bash
#SBATCH --partition=gpu-bermano
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/.wheel_cache/gpu_scale_check_job.out

VENV=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/.venv
cd /home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/code

"$VENV/bin/python" scripts/run_gpu_scale_check.py --device cuda --queries 30 --force
echo "SCALE_CHECK_JOB_DONE"

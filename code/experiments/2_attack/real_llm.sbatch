#!/bin/bash
#SBATCH --partition=killable
#SBATCH --gres=gpu:a5000:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -J d2-real-llm
#SBATCH -o /home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/strong_gpu_dev/logs/%x-%j.out
# Plan D.2: the single-neuron attack and backdoor on a real OPT-6.7B (open weights).
# Extra arguments are passed on (argparse: a later --out wins), e.g. --target auto --out ...json
# Uses an isolated venv that borrows the shared torch read-only; never the shared venv.
set -euo pipefail
cd /home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/strong_gpu_dev/code
export HF_HOME=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/hf_cache PYTHONPATH=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/strong_gpu_dev/code/src OMP_NUM_THREADS=8
export HF_HUB_DISABLE_PROGRESS_BARS=1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/strong_gpu_a6000/.venv-d2/bin/python experiments/2_attack/real_llm.py \
    --model facebook/opt-6.7b --config opt-6.7b --layers 8 16 24 31 --topk 8 \
    --out /home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/strong_gpu_dev/code/artifacts/results/real_llm_attack.json "$@"
echo "D2_DONE"

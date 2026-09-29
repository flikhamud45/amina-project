#!/bin/bash
#SBATCH --partition=killable
#SBATCH --gres=gpu:a5000:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH -J d3-ppl
#SBATCH -o /home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/strong_gpu_dev/logs/%x-%j.out
# Plan D.3: WikiText-2 perplexity of the integer OPT with real weights (fp32 model on CPU,
# integer graph on the GPU). Usage: sbatch real_weights_ppl.sbatch <hf model> <config> [extra args]
set -euo pipefail
cd /home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/strong_gpu_dev/code
export HF_HOME=/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/hf_cache HF_HUB_OFFLINE=1
export PYTHONPATH=$PWD/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MODEL=$1 CONFIG=$2; shift 2
/home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/strong_gpu_a6000/.venv-d2/bin/python \
    experiments/2_attack/real_weights_ppl.py --model "$MODEL" --config "$CONFIG" --device cuda \
    --parquet /home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/hf_cache/wikitext/test.parquet \
    --out "$PWD/artifacts/results/real_weights_ppl_$CONFIG.json" "$@"
echo "D3_DONE"

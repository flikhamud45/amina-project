#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=48G
#SBATCH -t 03:00:00
#SBATCH -J real-llm
# The single-neuron attack and backdoor on a real OPT-6.7B (report Sec. 4.2, *A real LLM*); see REAL_LLM.md.
# Extra arguments go to real_llm.py, e.g. --target auto --out artifacts/results/real_llm_attack_auto.json
# The stored runs used one A5000 on TAU's killable partition (sbatch -p killable --gres=gpu:a5000:1).
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR/code"
export PYTHONPATH="$PROJECT_DIR/code/src" OMP_NUM_THREADS=8 HF_HUB_DISABLE_PROGRESS_BARS=1
: "${HF_HOME:?export HF_HOME=<Hugging Face cache>}"; export HF_HOME
PY="${PVI_PYTHON:?export PVI_PYTHON=<python with transformers and pyarrow, see REAL_LLM.md>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" experiments/2_attack/real_llm.py --model facebook/opt-6.7b --config opt-6.7b --layers 8 16 24 31 --topk 8 "$@"
echo DONE

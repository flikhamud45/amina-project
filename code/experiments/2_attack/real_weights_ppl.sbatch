#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=80G
#SBATCH -t 04:00:00
#SBATCH -J real-weights-ppl
# WikiText-2 perplexity of the integer OPT with real weights (report Sec. 4.1; fp32 model on CPU,
# integer graph on the GPU). Usage: sbatch real_weights_ppl.sbatch <hf model> <config> [extra args]
# Extra arguments go to real_weights_ppl.py; an --out among them replaces the one below.
# The stored runs used one A5000 on TAU's killable partition (sbatch -p killable --gres=gpu:a5000:1).
# The checkpoint must already be in $HF_HOME (the job runs offline).
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR/code"
: "${HF_HOME:?export HF_HOME=<Hugging Face cache>}"; export HF_HOME HF_HUB_OFFLINE=1
export PYTHONPATH="$PROJECT_DIR/code/src" OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
PY="${PVI_PYTHON:?export PVI_PYTHON=<python with transformers and pyarrow, see REAL_LLM.md>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
MODEL=$1 CONFIG=$2; shift 2
"$PY" experiments/2_attack/real_weights_ppl.py --model "$MODEL" --config "$CONFIG" --device cuda \
    --parquet "${WIKITEXT_PARQUET:?export WIKITEXT_PARQUET=<wikitext-2-raw-v1 test.parquet>}" \
    --out "artifacts/results/real_weights_ppl_$CONFIG.json" "$@"
echo DONE

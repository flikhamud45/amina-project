#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 03:00:00
#SBATCH -J g3-game-llm
# Plan G3-min, item 3: the LP mixture over value-aware samplers against the in-range edits of
# inrange_llm.py on OPT-1.3B (layers 6/12/18/23, 40 prompts, caps 90/95/99/100, ranges from 8,192
# tokens of WikiText-2 train). OPT-1.3B in fp32 needs about 5.3 GB of GPU memory, so one RTX 2080 Ti
# suffices. Submit from the repository root, for example:
#   export HF_HOME=<Hugging Face cache outside the project quota> PVI_PYTHON=<python with transformers, pyarrow, scipy>
#   sbatch -p studentbatch -A gpu-students code/experiments/3_sampling_fixes/adaptive_game_llm.sbatch
# Extra arguments go to adaptive_game_llm.py.
set -euo pipefail
PROJECT_DIR="${PROJECT_DIR:-$SLURM_SUBMIT_DIR}"
cd "$PROJECT_DIR/code"
export PYTHONPATH="$PROJECT_DIR/code/src" OMP_NUM_THREADS=8 HF_HUB_DISABLE_PROGRESS_BARS=1
: "${HF_HOME:?export HF_HOME=<Hugging Face cache>}"; export HF_HOME
PY="${PVI_PYTHON:?export PVI_PYTHON=<python with transformers, pyarrow and scipy>}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
"$PY" -c "import scipy, transformers, pyarrow; print('scipy', scipy.__version__)"
"$PY" experiments/3_sampling_fixes/adaptive_game_llm.py --model facebook/opt-1.3b \
    --layers 6 12 18 23 --prompts 40 --calib-windows 64 --calib-seq 128 --caps 90 95 99 100 \
    --threads 8 --device cuda --out artifacts/results/adaptive_game_opt-1.3b.json "$@"
echo DONE

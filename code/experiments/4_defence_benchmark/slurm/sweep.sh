#!/bin/bash
# Submit the complete comparison benchmark of the report on another cluster. From the repository root:
#     export SBATCH_PARTITION=<gpu partition> SBATCH_GRES=gpu:<a100|h100 type name>:1 [SBATCH_ACCOUNT=...]
#     export PVI_PLATFORM=<name>     # e.g. h100; records go to code/artifacts/comparison/raw_<name>/
#     bash code/experiments/4_defence_benchmark/slurm/sweep.sh
# Identical jobs and flags to the TAU sweep; the only differences are the raw_<name>/ root
# and PVI_THREADS=1 for the one-thread variants (so a cluster minimum on CPUs per GPU
# cannot silently turn them into multi-thread runs).
set -euo pipefail
# raw/ holds the report's RTX 2080 Ti records and is frozen: every other machine writes
# raw_$PVI_PLATFORM/ (exported to the jobs; see bench.py --platform).  Override the
# partition and GPU of bench.sbatch with SBATCH_PARTITION / SBATCH_GRES.
: "${PVI_PLATFORM:?export PVI_PLATFORM=<name>, e.g. h100 (raw/ is frozen)}"
export PVI_PLATFORM
LOGS="logs/$PVI_PLATFORM"
mkdir -p "$LOGS"
S=code/experiments/4_defence_benchmark/slurm/bench.sbatch
sub() { local name=$1; shift; sbatch -o "$LOGS/%x-%j.out" -J "$name" "$@"; }

for m in mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar; do
  sub "b-$m" "$S" cnn --model "$m" --queries 30 --tampers 100
done
sub b-resnet18_224 "$S" cnn --model resnet18_224 --queries 30 --tampers 60

sub b-gpt2    "$S" llm --model gpt2 --seq 64 128 256 512 --queries 10
sub b-opt125  "$S" llm --model opt-125m --seq 64 2048 --queries 3 --lams 40 128 --modes C:int,Kpre:int
sub b-opt13   "$S" llm --model opt-1.3b --seq 64 2048 --queries 3 --lams 40 128 --modes C:int,Kpre:int
sub b-opt67   "$S" llm --model opt-6.7b --seq 64 2048 --queries 2 --lams 40 128 --modes C:int,Kpre:int
sub b-llama7  "$S" llm --model llama2-7b --seq 64 512 --queries 5 --lams 40 128 --modes C:int,C:fs,Kpre:int,K:int
sub b-llama7L "$S" llm --model llama2-7b --seq 2048 --queries 2 --lams 128 --modes C:int,Kpre:int
sub b-llama13 "$S" llm --model llama2-13b --seq 64 --queries 3 --lams 40 128 --modes C:int,Kpre:int
sub b-qwen4   "$S" llm --model qwen3-4b --seq 8 64 --queries 5 --lams 40 128 --modes C:int,Kpre:int,K:int

sub b-qwen4-1t -c 1 --export=ALL,PVI_THREADS=1 "$S" llm --model qwen3-4b --seq 8 --queries 5 --lams 40 128 --modes Kpre:int,C:int --tag _thr1
sub b-gpt2-1t  -c 1 --export=ALL,PVI_THREADS=1 "$S" llm --model gpt2 --seq 64 --queries 5 --lams 40 128 --modes Kpre:int,C:int --tag _thr1

sub b-gpt2-r16   "$S" llm --model gpt2 --seq 64 --queries 5 --lams 40 128 --modes C:int --rate 16
sub b-llama7-r16 "$S" llm --model llama2-7b --seq 64 --queries 3 --lams 40 128 --modes C:int --rate 16

squeue -u "$USER"

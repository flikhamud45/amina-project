#!/bin/bash
# Submit the complete comparison benchmark: every model and every setting used in
# the report.  Run from the repository root on the cluster:
#     bash cluster/fullcheck_sweep.sh
# Each job writes its own raw records; finished cells are skipped, so running the
# script again after a pre-emption only redoes what is missing.
set -euo pipefail
mkdir -p logs
S=cluster/fullcheck_bench.sbatch
sub() { local name=$1; shift; sbatch -o "logs/%x-%j.out" -J "$name" "$@"; }

# ---- CNNs (the models zkCNN, ZKML, ZENO, Bionetta, EZKL and Anchuri et al. use) ----
for m in mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar; do
  sub "b-$m" "$S" cnn --model "$m" --queries 30 --tampers 100 --paths 300
done
sub b-resnet18_224 "$S" cnn --model resnet18_224 --queries 30 --tampers 60 --paths 100

# ---- language models (shapes of GPT-2, OPT, Llama-2, Qwen3; random int8 weights) ----
sub b-gpt2    "$S" llm --model gpt2 --seq 64 128 256 512 --queries 10
sub b-opt125  "$S" llm --model opt-125m --seq 64 2048 --queries 3 --lams 40 128 --modes C:int,Kpre:int
sub b-opt13   "$S" llm --model opt-1.3b --seq 64 2048 --queries 3 --lams 40 128 --modes C:int,Kpre:int
sub b-opt67   "$S" llm --model opt-6.7b --seq 64 2048 --queries 2 --lams 40 128 --modes C:int,Kpre:int
sub b-llama7  "$S" llm --model llama2-7b --seq 64 512 --queries 5 --lams 40 128 --modes C:int,C:fs,Kpre:int,K:int
sub b-llama7L "$S" llm --model llama2-7b --seq 2048 --queries 2 --lams 128 --modes C:int,Kpre:int
sub b-llama13 "$S" llm --model llama2-13b --seq 64 --queries 3 --lams 40 128 --modes C:int,Kpre:int
sub b-qwen4   "$S" llm --model qwen3-4b --seq 8 64 --queries 5 --lams 40 128 --modes C:int,Kpre:int,K:int

# ---- single-threaded verifier (Maverick reports a 1-thread client) ----
sub b-qwen4-1t -c 1 "$S" llm --model qwen3-4b --seq 8 --queries 5 --lams 40 128 --modes Kpre:int,C:int --tag _thr1
sub b-gpt2-1t  -c 1 "$S" llm --model gpt2 --seq 64 --queries 5 --lams 40 128 --modes Kpre:int,C:int --tag _thr1

# ---- code rate: fewer opened columns for a larger one-time commitment ----
sub b-gpt2-r16   "$S" llm --model gpt2 --seq 64 --queries 5 --lams 40 128 --modes C:int --rate 16
sub b-llama7-r16 "$S" llm --model llama2-7b --seq 64 --queries 3 --lams 40 128 --modes C:int --rate 16

squeue -u "$USER"

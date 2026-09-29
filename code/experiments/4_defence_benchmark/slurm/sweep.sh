#!/bin/bash
# Submit the complete comparison benchmark of the report on another cluster. From the repository root:
#     export SBATCH_PARTITION=<gpu partition> SBATCH_GRES=gpu:<a100|h100 type name>:1 [SBATCH_ACCOUNT=...]
#     export PVI_PLATFORM=<name>     # e.g. h100; records go to code/artifacts/comparison/raw_<name>/
#     bash code/experiments/4_defence_benchmark/slurm/sweep.sh
# The TAU sweep's models, cells and flags, with three differences, so this is NOT an exact replica
# of the stored runs' code path:
#   * the weight-key fix is on (the stored runs re-uploaded the committed weights twice per mode-C
#     query; PVI_LEGACY_WEIGHT_KEY=1 with a _nofix tag restores that, see strong_gpu.sh (d));
#   * every LLM line is --lean (same integers and proof, less memory), as in strong_gpu.sh: the
#     two scripts write the same untagged cell names, so a root they share never holds lean and
#     non-lean builds under one name (aggregate.py refuses such a mix anyway);
#   * PVI_THREADS=1 for the one-thread variants (so a cluster minimum on CPUs per GPU cannot
#     silently turn them into multi-thread runs).
# Jobs are named <platform>-<job> with --dependency=singleton, as in strong_gpu.sh (whose jobs of
# the same name write the same cells); a job of that name already pending or running is skipped.
set -euo pipefail
unset PVI_THREADS PVI_TF32 PVI_LEGACY_WEIGHT_KEY   # variant switches: only the per-job --export lines set them
[ -f code/experiments/4_defence_benchmark/slurm/sweep.sh ] || { echo "run this from the repository root" >&2; exit 2; }
export PROJECT_DIR="$PWD"   # the jobs run this checkout (bench.sbatch), never an inherited PROJECT_DIR
# raw/ holds the earliest RTX 2080 Ti run, raw_rtx2080ti-v2/ its re-run and raw_l40s/ the report's
# numbers; all are frozen: every other run writes
# raw_$PVI_PLATFORM/ (exported to the jobs; see bench.py --platform).  Override the
# partition and GPU of bench.sbatch with SBATCH_PARTITION / SBATCH_GRES.
: "${PVI_PLATFORM:?export PVI_PLATFORM=<name>, e.g. h100 (raw/ is frozen)}"
export PVI_PLATFORM
LOGS="logs/$PVI_PLATFORM"
mkdir -p "$LOGS"
S=code/experiments/4_defence_benchmark/slurm/bench.sbatch
ME="${USER:-$(id -un)}"
QUEUED="$(squeue -h -u "$ME" -o %j)"
sub() {
  local name="$PVI_PLATFORM-$1"; shift
  if grep -qxF -- "$name" <<<"$QUEUED"; then echo "skip $name: already queued"; return 0; fi
  sbatch -o "$LOGS/%x-%j.out" -J "$name" --dependency=singleton "$@"
  QUEUED+=$'\n'"$name"
}

for m in mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar; do
  sub "b-$m" "$S" cnn --model "$m" --queries 30 --tampers 100
done
sub b-resnet18_224 "$S" cnn --model resnet18_224 --queries 30 --tampers 60

sub b-gpt2    "$S" llm --model gpt2 --seq 64 128 256 512 --queries 10 --lean
sub b-opt125  "$S" llm --model opt-125m --seq 64 2048 --queries 3 --lams 40 128 --modes C:int,Kpre:int --lean
sub b-opt13   "$S" llm --model opt-1.3b --seq 64 2048 --queries 3 --lams 40 128 --modes C:int,Kpre:int --lean
sub b-opt67   "$S" llm --model opt-6.7b --seq 64 2048 --queries 2 --lams 40 128 --modes C:int,Kpre:int --lean
sub b-llama7  "$S" llm --model llama2-7b --seq 64 512 --queries 5 --lams 40 128 --modes C:int,C:fs,Kpre:int,K:int --lean
sub b-llama7L "$S" llm --model llama2-7b --seq 2048 --queries 2 --lams 128 --modes C:int,Kpre:int --lean
sub b-llama13 "$S" llm --model llama2-13b --seq 64 --queries 3 --lams 40 128 --modes C:int,Kpre:int --lean
sub b-qwen4   "$S" llm --model qwen3-4b --seq 8 64 --queries 5 --lams 40 128 --modes C:int,Kpre:int,K:int --lean

sub b-qwen4-1t -c 1 --export=ALL,PVI_THREADS=1 "$S" llm --model qwen3-4b --seq 8 --queries 5 --lams 40 128 --modes Kpre:int,C:int --tag _thr1 --lean
sub b-gpt2-1t  -c 1 --export=ALL,PVI_THREADS=1 "$S" llm --model gpt2 --seq 64 --queries 5 --lams 40 128 --modes Kpre:int,C:int --tag _thr1 --lean

sub b-gpt2-r16   "$S" llm --model gpt2 --seq 64 --queries 5 --lams 40 128 --modes C:int --rate 16 --lean
sub b-llama7-r16 "$S" llm --model llama2-7b --seq 64 --queries 3 --lams 40 128 --modes C:int --rate 16 --lean

squeue -u "$ME"

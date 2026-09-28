#!/bin/bash
# Submit the report's benchmark jobs.  From the repository root of the checkout whose code should
# run (the jobs put its code/src first on PYTHONPATH, whatever pvi the venv has installed):
#     export PVI_PLATFORM=<new name>    # one name per GPU + CPU model; raw/ and raw_rtx2080ti-v2/ are frozen
#     export SBATCH_PARTITION=<gpu partition> SBATCH_GRES=gpu:<type>:1 [SBATCH_ACCOUNT=...]
#     bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh smoke   # once per GPU type; wait for "SMOKE OK"
#     bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh must    # the 29 jobs of raw_rtx2080ti-v2/
# Records go to code/artifacts/comparison/raw_$PVI_PLATFORM/, logs to
# logs/$PVI_PLATFORM/<platform>-<job>-<jobid>.out.
# Every job is named <platform>-<job> and carries --dependency=singleton, so two jobs of one name
# never run at once, and a job of that name still pending or running is not submitted again: after
# a pre-emption or a partial submission, re-running the command resubmits only what is missing
# (finished cells are skipped by bench.py itself).  PVI_DRYRUN=1 prints the sbatch lines instead.
# Without SLURM, run each bench.sbatch line from code/, one at a time, as
#     PYTHONPATH=src NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c <8 cores> \
#         ../.venv/bin/python experiments/4_defence_benchmark/bench.py <arguments> --threads 8
# (the -c 1 lines with 1 thread everywhere, and PVI_LEGACY_WEIGHT_KEY=1 as an environment variable).
set -euo pipefail
unset PVI_THREADS PVI_LEGACY_WEIGHT_KEY   # control switches: only the per-job --export lines set them
: "${PVI_PLATFORM:?export PVI_PLATFORM=<new name> (raw/ and raw_rtx2080ti-v2/ are frozen)}"
export PVI_PLATFORM
[ -f code/experiments/4_defence_benchmark/slurm/strong_gpu.sh ] || { echo "run this from the repository root" >&2; exit 2; }
export PROJECT_DIR="$PWD"   # the jobs run this checkout (bench.sbatch/smoke.sbatch), never an inherited PROJECT_DIR
LOGS="logs/$PVI_PLATFORM"
mkdir -p "$LOGS"
S=code/experiments/4_defence_benchmark/slurm/bench.sbatch
ME="${USER:-$(id -un)}"
if [ -n "${PVI_DRYRUN:-}" ]; then
  SB="echo sbatch"; QUEUED=""
else
  SB=sbatch; QUEUED="$(squeue -h -u "$ME" -o %j)"      # this user's pending and running job names
fi
sub() {
  local name="$PVI_PLATFORM-$1"; shift
  if grep -qxF -- "$name" <<<"$QUEUED"; then echo "skip $name: already queued"; return 0; fi
  $SB -o "$LOGS/%x-%j.out" -J "$name" --dependency=singleton "$@"
  QUEUED+=$'\n'"$name"
}
L128="--lams 128 --modes C:int,Kpre:int"

case "${1:-}" in
smoke)
  sub smoke code/experiments/4_defence_benchmark/slurm/smoke.sbatch
  ;;
must)
  # (a) the CNNs (Table 2, Figures 3-4, the image-model attacks of Sec. 4.3)
  for m in mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar; do
    sub "b-$m" "$S" cnn --model "$m" --queries 30 --tampers 100
  done
  sub b-resnet18_224 "$S" cnn --model resnet18_224 --queries 30 --tampers 60
  # (b) the 12-block decoders, always built in full; _thr1: one verifier thread (Table 4's Maverick row)
  sub b-gpt2    "$S" llm --model gpt2 --seq 64 128 256 512 --queries 10 --lean
  sub b-opt125  "$S" llm --model opt-125m --seq 64 2048 --queries 5 --lams 40 128 --modes C:int,Kpre:int --lean
  sub b-gpt2-1t -c 1 --export=ALL,PVI_THREADS=1 "$S" llm --model gpt2 --seq 64 --queries 5 --lams 40 128 \
      --modes Kpre:int,C:int --tag _thr1 --lean
  # (c) the larger decoders: 1 and 2 blocks and the full model (the two-point extrapolation is checked
  #     against the full build; a full build that does not fit the GPU is skipped), with the
  #     full-model attacks at <= 64 tokens
  sub f-opt13-64   "$S" llm --model opt-1.3b   --seq 64 --builds 1,2,full --queries 10 $L128 --llm-tampers 30 --lean
  sub f-opt67-64   "$S" llm --model opt-6.7b   --seq 64 --builds 1,2,full --queries 10 $L128 --llm-tampers 30 --lean
  sub f-llama7-64  "$S" llm --model llama2-7b  --seq 64 --builds 1,2,full --queries 10 $L128 --llm-tampers 30 --lean
  sub f-llama13-64 "$S" llm --model llama2-13b --seq 64 --builds 1,2,full --queries 10 $L128 --llm-tampers 30 --lean
  sub f-qwen4      "$S" llm --model qwen3-4b   --seq 8 64 --builds 1,2,full --queries 10 $L128 --llm-tampers 30 --lean
  sub f-qwen4-1t -c 1 --export=ALL,PVI_THREADS=1 "$S" llm --model qwen3-4b --seq 8 --builds 1,2,full --queries 5 \
      --lams 40 128 --modes Kpre:int,C:int --tag _thr1 --lean
  sub f-opt13-2k   "$S" llm --model opt-1.3b  --seq 2048 --builds 1,2,full --queries 5 $L128 --lean
  sub f-opt67-2k   "$S" llm --model opt-6.7b  --seq 2048 --builds 1,2,full --queries 5 $L128 --lean
  sub f-llama7-2k  "$S" llm --model llama2-7b --seq 2048 --builds 1,2,full --queries 5 $L128 --lean
  # (d) controls, counted in Sec. 4.3 but in no table or figure.  _nolean: the non-lean path.
  #     _nofix: the earlier run's weight cache (PVI_LEGACY_WEIGHT_KEY=1: the committed weights
  #     re-uploaded twice per query), at the cells whose committed-mode (C) times the report quotes;
  #     the CNN controls are timing only (--tampers 0)
  sub ab-gpt2   "$S" llm --model gpt2 --seq 512 --queries 10 $L128 --tag _nolean
  sub ab-llama7 "$S" llm --model llama2-7b --seq 64 --builds full --queries 10 $L128 --tag _nolean
  sub nofix-gpt2   --export=ALL,PVI_LEGACY_WEIGHT_KEY=1 "$S" llm --model gpt2 --seq 64 --queries 10 $L128 --lean --tag _nofix
  sub nofix-llama7 --export=ALL,PVI_LEGACY_WEIGHT_KEY=1 "$S" llm --model llama2-7b --seq 64 --builds 1,2,full \
      --queries 10 $L128 --lean --tag _nofix
  for m in mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar resnet18_224; do
    sub "nofix-$m" --export=ALL,PVI_LEGACY_WEIGHT_KEY=1 "$S" cnn --model "$m" --queries 30 --tampers 0 --tag _nofix
  done
  sub nofix-opt13-2k --export=ALL,PVI_LEGACY_WEIGHT_KEY=1 "$S" llm --model opt-1.3b --seq 2048 --builds full \
      --queries 5 --lams 128 --modes C:int --lean --tag _nofix
  ;;
*)
  echo "usage: $0 smoke|must" >&2; exit 2
  ;;
esac
[ -n "${PVI_DRYRUN:-}" ] || squeue -u "$ME"

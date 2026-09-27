#!/bin/bash
# Every job of the strong-GPU plan, by tier.  From the repository root, after the smoke job passed:
#     export PVI_PLATFORM=<name>                        # h100 | a100-80 | a100-40 | ... (one name per GPU+CPU model)
#     export SBATCH_PARTITION=<gpu partition> SBATCH_GRES=gpu:<type>:1 [SBATCH_ACCOUNT=...] [SBATCH_CONSTRAINT=...]
#     bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh smoke    # once; wait for "SMOKE OK"
#     bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh must     # then should, then nice
# Records go to code/artifacts/comparison/raw_$PVI_PLATFORM/ (raw/ is frozen), logs to logs/$PVI_PLATFORM/.
# Cell names match the stored ones; --tag is used only for real variants (_thr1, _nolean, _nofix, _gpuv,
# _thr12, _batch, _tf32), which the paper tables never mix into the headline.
# PVI_DRYRUN=1 prints the sbatch lines instead of submitting (for a machine without SLURM: run each
# bench.sbatch line as  OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c <8 cores> .venv/bin/python
# code/experiments/4_defence_benchmark/bench.py <arguments> --threads 8  from code/, one at a time;
# -c 1 / PVI_THREADS=1 lines with --threads 1, PVI_LEGACY_WEIGHT_KEY / PVI_TF32 as environment variables).
# Every LLM run is --lean (claims streamed to the host, dead tensors freed): same integers and proof,
# a fraction of the memory; the _nolean jobs measure what that changes.
set -euo pipefail
: "${PVI_PLATFORM:?export PVI_PLATFORM=<name>, e.g. h100 (raw/ is frozen)}"
export PVI_PLATFORM
LOGS="logs/$PVI_PLATFORM"
mkdir -p "$LOGS"
S=code/experiments/4_defence_benchmark/slurm/bench.sbatch
SB=sbatch; [ -n "${PVI_DRYRUN:-}" ] && SB="echo sbatch"
sub() { local name=$1; shift; $SB -o "$LOGS/%x-%j.out" -J "$name" "$@"; }
PY="${PVI_PYTHON:-$PWD/.venv/bin/python}"
L128="--lams 128 --modes C:int,Kpre:int"
GV="${PVI_GPUV_DEVICE:-cuda}"          # 40 GB cards: PVI_GPUV_DEVICE=cuda:1 PVI_GPUV_GRES=gpu:<type>:2
GG=(); [ -n "${PVI_GPUV_GRES:-}" ] && GG=(--gres="$PVI_GPUV_GRES")

case "${1:-}" in
smoke)
  $SB -o "$LOGS/%x-%j.out" code/experiments/4_defence_benchmark/slurm/smoke.sbatch
  ;;
must)
  # (a) the report's CNN jobs with unchanged flags: every hardware-independent number and the
  #     CNN counts (2,196 honest, 1,854 attacks) must equal raw/ exactly
  for m in mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar; do
    sub "b-$m" "$S" cnn --model "$m" --queries 30 --tampers 100
  done
  sub b-resnet18_224 "$S" cnn --model resnet18_224 --queries 30 --tampers 60
  # (b) the 12-block LLMs (always built in full), the report's flags
  sub b-gpt2    "$S" llm --model gpt2 --seq 64 128 256 512 --queries 10 --lean
  sub b-opt125  "$S" llm --model opt-125m --seq 64 2048 --queries 5 --lams 40 128 --modes C:int,Kpre:int --lean
  sub b-gpt2-1t -c 1 --export=ALL,PVI_THREADS=1 "$S" llm --model gpt2 --seq 64 --queries 5 --lams 40 128 \
      --modes Kpre:int,C:int --tag _thr1 --lean
  # (c) the >12-block LLMs: 1 and 2 blocks AND the full model on this machine (the 2-point
  #     extrapolation is then checked against the measured value), with the full-model attacks at <=64 tokens
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
  # (d) controls: the lean path against the report's (non-lean) path, and the weight re-upload of the
  #     stored runs (PVI_LEGACY_WEIGHT_KEY=1) against the fixed code, on this GPU
  sub ab-gpt2   "$S" llm --model gpt2 --seq 512 --queries 10 $L128 --tag _nolean
  sub ab-llama7 "$S" llm --model llama2-7b --seq 64 --builds full --queries 10 $L128 --tag _nolean
  sub ab-opt13 --mem=96G "$S" llm --model opt-1.3b --seq 2048 --builds full --queries 5 $L128 --tag _nolean  # ~27 GiB GPU
  sub nofix-gpt2   --export=ALL,PVI_LEGACY_WEIGHT_KEY=1 "$S" llm --model gpt2 --seq 64 --queries 10 $L128 --lean --tag _nofix
  sub nofix-llama7 --export=ALL,PVI_LEGACY_WEIGHT_KEY=1 "$S" llm --model llama2-7b --seq 64 --builds 1,2,full \
      --queries 10 $L128 --lean --tag _nofix
  # (e) kernel microbenchmarks (FP32 forward, FP64 fold/open, Freivalds, NTT, host<->device copies)
  $SB -o "$LOGS/%x-%j.out" -J microbench -c 8 --mem=32G -t 00:30:00 \
      --wrap "cd code && $PY experiments/4_defence_benchmark/microbench.py --device cuda --reps 10 --out ../$LOGS/microbench.jsonl"
  ;;
should)
  # (f) the rest of zkLLM's 2,048-token list, measured in full: Llama-2-13B, OPT-350M, OPT-2.7B, OPT-13B
  sub z-llama13-2k --mem=128G "$S" llm --model llama2-13b --seq 2048 --builds 1,2,full --queries 3 $L128 --lean
  sub z-opt13b-2k  --mem=128G "$S" llm --model opt-13b    --seq 2048 --builds 1,2,full --queries 3 $L128 --lean
  sub z-opt27-2k   "$S" llm --model opt-2.7b --seq 64 2048 --builds 1,2,full --queries 3 $L128 --lean
  sub z-opt350-2k  "$S" llm --model opt-350m --seq 64 2048 --queries 3 $L128 --lean --builds full
  # (g) a GPU client (verifier on the GPU): the zkLLM-matched rows and the 64-token rows
  for m in opt-125m opt-1.3b opt-6.7b llama2-7b; do
    sub "g-$m-2k" ${GG[@]+"${GG[@]}"} "$S" llm --model "$m" --seq 2048 --builds full --queries 5 $L128 --lean \
        --verifier-device "$GV" --tag _gpuv
  done
  sub g-llama7-64 ${GG[@]+"${GG[@]}"} "$S" llm --model llama2-7b --seq 64 --builds full --queries 10 $L128 --lean --verifier-device "$GV" --tag _gpuv
  sub g-gpt2      ${GG[@]+"${GG[@]}"} "$S" llm --model gpt2 --seq 64 512 --queries 10 $L128 --lean --verifier-device "$GV" --tag _gpuv
  # (h) the shape of the depth dependence (validate_extrapolation.py): 4, 8, 16 blocks
  sub v-llama7-64 "$S" llm --model llama2-7b --seq 64   --builds 4,8,16 --queries 10 $L128 --lean
  sub v-llama7-2k "$S" llm --model llama2-7b --seq 2048 --builds 4,8,16 --queries 5  $L128 --lean
  sub v-opt13-2k  "$S" llm --model opt-1.3b  --seq 2048 --builds 6,12   --queries 5  $L128 --lean
  # (i) ZKTorch's setting (1 token) and zkLLM's host (12 verifier threads)
  sub s-llama7-t1 "$S" llm --model llama2-7b --seq 1 --builds 1,2,full --queries 10 $L128 --lean
  sub s-llama7-thr12 -c 12 --export=ALL,PVI_THREADS=12 "$S" llm --model llama2-7b --seq 2048 --builds full \
      --queries 3 --lams 128 --modes Kpre:int --lean --tag _thr12
  ;;
nice)
  # (j) batching: B prompts / images against one set of openings
  sub n-batch-gpt2   "$S" llm --model gpt2 --seq 64 --queries 3 $L128 --batches 8 32 --batch-trials 3 --lean --tag _batch
  sub n-batch-llama7 "$S" llm --model llama2-7b --seq 64 --builds 1,2,full --queries 3 $L128 --batches 8 32 \
      --batch-trials 3 --lean --tag _batch
  for m in mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar; do
    sub "n-batch-$m" "$S" cnn --model "$m" --queries 30 --tampers 10 --cnn-batches 8 32 64 128 256 --batch-trials 5 --tag _batch
  done
  sub n-batch-resnet18_224 "$S" cnn --model resnet18_224 --queries 30 --tampers 10 --cnn-batches 8 32 64 128 \
      --batch-trials 5 --tag _batch
  # (k) Llama-2's full 4,096-token context, and GPT-2 at its 1,024 maximum
  sub n-llama7-4k --mem=128G "$S" llm --model llama2-7b --seq 4096 --builds 1,2,full --queries 3 $L128 --lean
  sub n-gpt2-1k "$S" llm --model gpt2 --seq 1024 --queries 10 $L128 --lean
  # (l) TF32 tensor cores for the (exact) float32 GEMMs of the forward pass
  sub n-tf32-llama7 --export=ALL,PVI_TF32=1 "$S" llm --model llama2-7b --seq 64 2048 --builds full \
      --queries 5 --lams 128 --modes C:int --lean --tf32 --tag _tf32
  sub n-tf32-gpt2 --export=ALL,PVI_TF32=1 "$S" llm --model gpt2 --seq 64 512 --queries 10 \
      --lams 128 --modes C:int --lean --tf32 --tag _tf32
  # (m) 30-70B shapes: 1-2 blocks, and the full Llama-2-70B at 64 tokens (80 GB card, 64 GiB of int8 weights)
  sub n-opt30  "$S" llm --model opt-30b   --seq 64 2048 --queries 2 $L128 --lean
  sub n-opt66  "$S" llm --model opt-66b   --seq 64      --queries 2 $L128 --lean
  sub n-llama70 "$S" llm --model llama2-70b --seq 64 512 --queries 2 $L128 --lean
  sub n-llama70-full --mem=192G -t 12:00:00 "$S" llm --model llama2-70b --seq 64 --builds full --queries 2 $L128 --lean
  ;;
*)
  echo "usage: $0 smoke|must|should|nice" >&2; exit 2
  ;;
esac
[ -n "${PVI_DRYRUN:-}" ] || squeue -u "${USER:-$(id -un)}"

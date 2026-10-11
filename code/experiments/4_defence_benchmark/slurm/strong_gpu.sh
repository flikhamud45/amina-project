#!/bin/bash
# Every benchmark job of the report (raw_l40s/) and of the S&P 2027 run, by tier.  From the
# repository root (of the checkout whose code should run: the jobs put its code/src first on
# PYTHONPATH, whatever pvi the venv has installed):
#     export PVI_PLATFORM=<new name>    # one name per GPU + CPU model; raw/, raw_rtx2080ti-v2/, raw_l40s/ and raw_l40s_improved/ are frozen
#     export SBATCH_PARTITION=<gpu partition> SBATCH_GRES=gpu:<type>:1 [SBATCH_ACCOUNT=...] [SBATCH_CONSTRAINT=...]
#     bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh smoke    # once per GPU type; wait for "SMOKE OK"
#     bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh must     # the S&P paper's cells (below)
# The tiers:
#   must    the cells the S&P paper's tables read (Plan 2 F6 part 9, audited against tab:eval-llm,
#           tab:eval-compare and the outcome counts): the optimised protocol of paper_assets.OPTIMISED (mode C:
#           the commitment plan PVI_C_POLICY, default auto, the compact encoding and the pruned last block; Kpre:
#           the encoding, the verifier's own embedding rows and pruning; the streaming GPU verifier), full models
#           only (no extrapolation), under Fiat-Shamir (C:fs, Kpre:fs).  Interactive cells (:int) only where a
#           table compares with an interactive system: zkLLM's four 2,048-token rows and Maverick's Qwen3-4B row.
#           sp2027.sh's bench tier runs it; PVI_REHEARSAL=1 submits its smallest cells (GPT-2, 2 blocks; LeNet-5).
#   batch   C4a: B prompts against one opening, Fiat-Shamir (plus the interactive cell of C4a's byte model);
#           sp2027.sh's batch tier
#   report-must, should, nice   the course report's basic protocol (raw_l40s/, commit 9401431), with its
#           controls (_nolean, _nofix, _thr1, _thr12, _tf32) and the extrapolation checks; not in the S&P package
# Records go to code/artifacts/comparison/raw_$PVI_PLATFORM/, logs to
# logs/$PVI_PLATFORM/<platform>-<job>-<jobid>.out.  When sp2027.sh calls this script it passes its job chain:
# PVI_DEP (the first job's dependency), PVI_CHAIN (each job after the previous one), PVI_EXCLUSIVE, PVI_MEM_MAX,
# PVI_DRYN0, and PVI_LAST_FILE, into which the must and batch tiers write the chain's end and their totals.
# Every job is named <platform>-<job> and carries --dependency=singleton, so two jobs of one name
# never run at once, and a job of that name still pending or running is not submitted again: after
# a pre-emption or a partial submission, re-running the tier command resubmits only what is missing
# (finished cells are skipped by bench.py itself).
# Cell names match the stored ones; --tag is used only for real variants (_thr1, _nolean, _nofix, _gpuv,
# _thr12, _batch, _tf32), which the paper tables never mix into the headline.
# PVI_DRYRUN=1 prints the sbatch lines instead of submitting.  On a machine without SLURM, run each
# bench.sbatch line from code/, one at a time, as
#     PYTHONPATH=src NVIDIA_TF32_OVERRIDE=0 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 taskset -c <8 cores> \
#         ../.venv/bin/python experiments/4_defence_benchmark/bench.py <arguments> --threads 8
# (-c 1 / PVI_THREADS=1 lines with 1 thread everywhere, -c 12 / PVI_THREADS=12 with 12;
# PVI_LEGACY_WEIGHT_KEY=1 as an environment variable; the PVI_TF32=1 lines without NVIDIA_TF32_OVERRIDE=0).
# Every LLM run is --lean (claims streamed to the host, dead tensors freed): same integers and proof,
# a fraction of the memory; the _nolean jobs measure what that changes.
set -euo pipefail
unset PVI_THREADS PVI_TF32 PVI_LEGACY_WEIGHT_KEY   # variant switches: only the per-job --export lines set them
: "${PVI_PLATFORM:?export PVI_PLATFORM=<new name> (raw/, raw_rtx2080ti-v2/, raw_l40s/ and raw_l40s_improved/ are frozen)}"
export PVI_PLATFORM
[ -f code/experiments/4_defence_benchmark/slurm/strong_gpu.sh ] || { echo "run this from the repository root" >&2; exit 2; }
export PROJECT_DIR="$PWD"   # the jobs run this checkout (bench.sbatch/smoke.sbatch), never an inherited PROJECT_DIR
LOGS="logs/$PVI_PLATFORM"
[ -n "${PVI_DRYRUN:-}" ] || mkdir -p "$LOGS"
S=code/experiments/4_defence_benchmark/slurm/bench.sbatch
ME="${USER:-$(id -un)}"
if [ -n "${PVI_DRYRUN:-}" ]; then
  SB="echo sbatch"; QUEUED=""
elif [ -n "${PVI_LOCAL:-}" ]; then
  SB=sbatch; QUEUED=""                                  # no SLURM (sp2027.sh PVI_LOCAL=1): the S&P tiers run here
else
  SB=sbatch; QUEUED="$(squeue -h -u "$ME" -o %j)"      # this user's pending and running job names
fi
sub() {
  local name="$PVI_PLATFORM-$1"; shift
  if grep -qxF -- "$name" <<<"$QUEUED"; then echo "skip $name: already queued"; return 0; fi
  $SB -o "$LOGS/%x-%j.out" -J "$name" --dependency=singleton "$@"
  QUEUED+=$'\n'"$name"
}
# subh <name> <minutes> <sbatch options and bench.sbatch arguments...>: the S&P tiers' jobs, with a time limit,
# --exclusive (PVI_EXCLUSIVE, default on) and the job chain of sp2027.sh (PVI_DEP, PVI_CHAIN); a --mem=<n>G among
# the options is capped at PVI_MEM_MAX.  Totals and the chain's end go to PVI_LAST_FILE.
DEP="${PVI_DEP:-}"; CHAIN="${PVI_CHAIN:-1}"; MEMMAX="${PVI_MEM_MAX:-256}"; DRYN="${PVI_DRYN0:-0}"
NJOBS=0; MINUTES=0
subh() {
  local name="$PVI_PLATFORM-$1" minutes=$2 dep="singleton" jid a; shift 2
  if grep -qxF -- "$name" <<<"$QUEUED"; then echo "skip $name: already queued"; return 0; fi
  [ -n "$DEP" ] && dep="$DEP,singleton"
  local args=(-o "$LOGS/%x-%j.out" -J "$name" "--dependency=$dep" "--time=$minutes")
  [ "${PVI_EXCLUSIVE:-1}" = 1 ] && args+=(--exclusive)
  for a in "$@"; do
    if [[ "$a" =~ ^--mem=([0-9]+)G$ ]] && [ "${BASH_REMATCH[1]}" -gt "$MEMMAX" ]; then a="--mem=${MEMMAX}G"; fi
    args+=("$a")
  done
  if [ -n "${PVI_DRYRUN:-}" ]; then
    DRYN=$((DRYN + 1)); jid="DRY$DRYN"
    echo "sbatch ${args[*]}"
  elif [ -n "${PVI_LOCAL:-}" ]; then         # no SLURM: bench.sbatch here, with the job's --export and thread count
    local envs=(SLURM_JOB_ID=local SLURM_SUBMIT_DIR="$PWD") run=() skip=""
    for a in "$@"; do
      if [ -n "$skip" ]; then skip=""; continue; fi
      case "$a" in
        -c) skip=1 ;;
        --export=ALL,*) IFS=, read -r -a kv <<<"${a#--export=ALL,}"; envs+=("${kv[@]}") ;;
        --mem=*|--time=*|-t) [ "$a" = -t ] && skip=1 ;;
        *) run+=("$a") ;;
      esac
    done
    jid="$(date +%s)"
    echo "$(date '+%F %T') running $name (limit $minutes min): $LOGS/$name-$jid.out"
    env "${envs[@]}" timeout "${minutes}m" bash "${run[@]}" > "$LOGS/$name-$jid.out" 2>&1 \
        || echo "$name: exit status $?" >&2
  else
    jid="$(sbatch --parsable "${args[@]}")"; jid="${jid%%;*}"
    echo "submitted $name: job $jid"
  fi
  QUEUED+=$'\n'"$name"
  NJOBS=$((NJOBS + 1)); MINUTES=$((MINUTES + minutes))
  [ "$CHAIN" = 1 ] && DEP="afterany:$jid"
  return 0
}
L128="--lams 128 --modes C:int,Kpre:int"
POL="${PVI_C_POLICY:-auto}"
# the optimised protocol (paper_assets.OPTIMISED): mode C with the plan, the encoding and pruning (CPU and streaming
# GPU verifier); Kpre with the encoding, the verifier's own embedding rows and pruning
OC="--policy $POL --wire --prune-last --tag _wire"
OCG="--policy $POL --wire --prune-last --verifier-device cuda --verifier-impl stream --tag _wire_gpuv_stream"
OK="--lookups --wire --prune-last --tag _wire"

case "${1:-}" in
smoke)
  sub smoke code/experiments/4_defence_benchmark/slurm/smoke.sbatch
  ;;
must)
  # The S&P paper's v0 benchmark (Plan 2 F6 part 9): every cell tab:eval-llm (new columns: the C prover, the CPU
  # and GPU verifiers, the proofs; Kpre's CPU verifier and proof), tab:eval-compare (ours) and the outcome counts
  # read, and nothing else.  Fiat-Shamir by default; :int only for zkLLM's rows (OPT-125M, OPT-1.3B, Llama-2-7B,
  # Llama-2-13B at 2,048 tokens, prover and both verifiers) and Maverick's (Qwen3-4B, 8 tokens, Kpre, lambda 40,
  # one thread; without the compact encoding as the table states, and with it as paper_assets.OPTIMISED_SPECIAL
  # defines the cell).  5 timed queries after a warm-up for the 64-token cells and for OPT-1.3B and Llama-2-7B at
  # 2,048 tokens, 3 for the other 2,048-token cells.
  if [ -n "${PVI_REHEARSAL:-}" ]; then
    subh r-c-lenet5 30 "$S" cnn --model lenet5 --queries 3 --tampers 3 --policy "$POL" --wire --tag _wire
    subh r-c-gpt2-64 30 "$S" llm --model gpt2 --seq 64 --builds 2 --queries 2 --lams 128 --modes C:fs,C:int \
        --llm-tampers 1 --lean $OC
    subh r-k-gpt2-64 20 "$S" llm --model gpt2 --seq 64 --builds 2 --queries 2 --lams 128 --modes Kpre:fs --lean $OK
    subh r-cg-gpt2-64 30 "$S" llm --model gpt2 --seq 64 --builds 2 --queries 2 --lams 128 --modes C:fs --lean $OCG
  else
    # (a) the image models: C with the plan and the encoding, with the attacks (VGG-16: zkCNN and ZKML; LeNet-5:
    #     zkCNN and Bionetta in the extended version; all of them: the outcome counts), then K and Kpre
    for m in mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar resnet18_224; do
      t=100; [ "$m" = resnet18_224 ] && t=60
      subh "c-$m" 30 "$S" cnn --model "$m" --queries 30 --tampers "$t" --policy "$POL" --wire --tag _wire
      subh "k-$m" 20 "$S" cnn --model "$m" --queries 30 --tampers 0 --wire --tag _wire
    done
    # (b) 64-token prompts (tab:eval-llm's first block; GPT-2 for zkGPT and DeepProve), with the attacks on the
    #     full models (the outcome counts); GPT-2 at 512 tokens (DeepProve); Llama-2-7B at 1 token (ZKTorch)
    for m in gpt2 qwen3-4b llama2-7b llama2-13b; do
      mem=96; mc=45; mk=20; mg=40
      case "$m" in gpt2) mc=20; mg=20 ;; llama2-13b) mem=128; mc=75; mk=30; mg=60 ;; esac
      subh "c-$m-64" "$mc" "--mem=${mem}G" "$S" llm --model "$m" --seq 64 --builds full --queries 5 --lams 128 \
          --modes C:fs --llm-tampers 30 --lean $OC
      subh "k-$m-64" "$mk" "--mem=${mem}G" "$S" llm --model "$m" --seq 64 --builds full --queries 5 --lams 128 \
          --modes Kpre:fs --lean $OK
      subh "cg-$m-64" "$mg" "--mem=${mem}G" "$S" llm --model "$m" --seq 64 --builds full --queries 5 --lams 128 \
          --modes C:fs --lean $OCG
    done
    subh c-gpt2-512 20 "$S" llm --model gpt2 --seq 512 --builds full --queries 5 --lams 128 --modes C:fs --lean $OC
    subh c-llama2-7b-1 40 "$S" llm --model llama2-7b --seq 1 --builds full --queries 5 --lams 128 --modes C:fs \
        --lean $OC
    # Maverick's setting (interactive: Maverick draws a fresh challenge per query): Kpre, lambda 40, one thread
    subh k-qwen3-4b-8-thr1 30 -c 1 --export=ALL,PVI_THREADS=1 "$S" llm --model qwen3-4b --seq 8 --builds full \
        --queries 5 --lams 40 --modes Kpre:int --lean --lookups --prune-last --tag _thr1
    subh k-qwen3-4b-8-thr1-wire 30 -c 1 --export=ALL,PVI_THREADS=1 "$S" llm --model qwen3-4b --seq 8 --builds full \
        --queries 5 --lams 40 --modes Kpre:int --lean --lookups --prune-last --wire --tag _thr1_wire
    # (c) 2,048-token prompts (tab:eval-llm's second block; zkLLM's rows, interactive, as zkLLM)
    for m in opt-125m opt-1.3b opt-6.7b llama2-7b llama2-13b; do
      q=3; mem=96; mc=60; mk=30; mg=45; cm="C:fs,C:int"
      case "$m" in
        opt-125m) mc=20; mk=15; mg=20 ;;
        opt-1.3b) q=5; mc=40; mk=20; mg=30 ;;
        opt-6.7b) cm="C:fs" ;;
        llama2-7b) q=5; mem=128; mc=90; mg=60 ;;
        llama2-13b) mem=128; mc=150; mk=45; mg=90 ;;
      esac
      subh "c-$m-2k" "$mc" "--mem=${mem}G" "$S" llm --model "$m" --seq 2048 --builds full --queries "$q" \
          --lams 128 --modes "$cm" --lean $OC
      subh "k-$m-2k" "$mk" "--mem=${mem}G" "$S" llm --model "$m" --seq 2048 --builds full --queries "$q" \
          --lams 128 --modes Kpre:fs --lean $OK
      subh "cg-$m-2k" "$mg" "--mem=${mem}G" "$S" llm --model "$m" --seq 2048 --builds full --queries "$q" \
          --lams 128 --modes "$cm" --lean $OCG
    done
  fi
  ;;
batch)
  # C4a: B prompts of one client against one fold and one opening (bench.py --batches; B = 1 is each cell's own
  # queries): Llama-2-7B at 64 tokens with B in {1, 4, 16, 64}, OPT-1.3B at 512 with B in {1, 4, 16}, Llama-2-7B
  # at 2,048 with B in {1, 2, 4}; modes C and Kpre under Fiat-Shamir, 3 trials per B.  The interactive Llama-2-7B
  # 64-token cell checks C4a's byte model (196.7 + 127.8/B MB per query under the paper plan, interactive).  A batch
  # that does not fit the GPU ends its job (bench.py does not skip it), so the 2,048-token cells are jobs of their own.
  OB="--wire --prune-last --tag _wire_batch"
  if [ -n "${PVI_REHEARSAL:-}" ]; then
    subh r-b-gpt2-c 20 "$S" llm --model gpt2 --seq 64 --builds 2 --queries 2 --lams 128 --modes C:fs --batches 4 \
        --batch-trials 2 --lean --policy "$POL" $OB
    subh r-b-gpt2-k 20 "$S" llm --model gpt2 --seq 64 --builds 2 --queries 2 --lams 128 --modes Kpre:fs --batches 4 \
        --batch-trials 2 --lean --lookups $OB
  else
    subh b-llama2-7b-64-c 60 "$S" llm --model llama2-7b --seq 64 --builds full --queries 3 --lams 128 --modes C:fs \
        --batches 4 16 64 --batch-trials 3 --lean --policy "$POL" $OB
    subh b-llama2-7b-64-cint 60 "$S" llm --model llama2-7b --seq 64 --builds full --queries 3 --lams 128 --modes C:int \
        --batches 4 16 64 --batch-trials 3 --lean --policy "$POL" $OB
    subh b-llama2-7b-64-k 30 "$S" llm --model llama2-7b --seq 64 --builds full --queries 3 --lams 128 --modes Kpre:fs \
        --batches 4 16 64 --batch-trials 3 --lean --lookups $OB
    subh b-opt-1.3b-512-c 30 "$S" llm --model opt-1.3b --seq 512 --builds full --queries 3 --lams 128 --modes C:fs \
        --batches 4 16 --batch-trials 3 --lean --policy "$POL" $OB
    subh b-opt-1.3b-512-k 20 "$S" llm --model opt-1.3b --seq 512 --builds full --queries 3 --lams 128 --modes Kpre:fs \
        --batches 4 16 --batch-trials 3 --lean --lookups $OB
    subh b-llama2-7b-2k-c 90 --mem=192G "$S" llm --model llama2-7b --seq 2048 --builds full --queries 3 --lams 128 \
        --modes C:fs --batches 2 4 --batch-trials 3 --lean --policy "$POL" $OB
    subh b-llama2-7b-2k-k 60 --mem=192G "$S" llm --model llama2-7b --seq 2048 --builds full --queries 3 --lams 128 \
        --modes Kpre:fs --batches 2 4 --batch-trials 3 --lean --lookups $OB
  fi
  ;;
report-must)
  # The course report's basic protocol (raw_l40s/, commit 9401431), formerly the must tier (Plan 2 F6 part 9 moved
  # it here: the S&P tables cite none of its controls and none of its basic-protocol cells).
  # (a) the CNN jobs (Table 2, Figures 3-4, the CNN counts of Sec. 4.3; untagged, they give
  #     2,196 honest queries and 1,854 attacks, as in raw/)
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
  # (d) controls, on this GPU.  _nolean: non-lean WITH the weight-key fix (the lean path's effect).
  #     _nofix: lean with the earliest run raw/'s weight re-upload (PVI_LEGACY_WEIGHT_KEY=1), against
  #     the fixed code (the fix's effect), at the cells the paper quotes committed-mode (C) numbers for:
  #     every CNN (Table 2), gpt2 and llama2-7b at 64 tokens, opt-1.3b at 2,048.  raw/'s own code
  #     path is legacy key + non-lean; raw_rtx2080ti-v2/ is this script rerun on raw/'s hardware
  #     with today's code (the like-for-like hardware comparison).
  sub ab-gpt2   "$S" llm --model gpt2 --seq 512 --queries 10 $L128 --tag _nolean
  sub ab-llama7 "$S" llm --model llama2-7b --seq 64 --builds full --queries 10 $L128 --tag _nolean
  sub ab-opt13 --mem=96G "$S" llm --model opt-1.3b --seq 2048 --builds full --queries 5 $L128 --tag _nolean  # ~27 GiB GPU
  sub nofix-gpt2   --export=ALL,PVI_LEGACY_WEIGHT_KEY=1 "$S" llm --model gpt2 --seq 64 --queries 10 $L128 --lean --tag _nofix
  sub nofix-llama7 --export=ALL,PVI_LEGACY_WEIGHT_KEY=1 "$S" llm --model llama2-7b --seq 64 --builds 1,2,full \
      --queries 10 $L128 --lean --tag _nofix
  for m in mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar resnet18_224; do   # timing only: --tampers 0, no attacks
    sub "nofix-$m" --export=ALL,PVI_LEGACY_WEIGHT_KEY=1 "$S" cnn --model "$m" --queries 30 --tampers 0 --tag _nofix
  done
  sub nofix-opt13-2k --export=ALL,PVI_LEGACY_WEIGHT_KEY=1 "$S" llm --model opt-1.3b --seq 2048 --builds full \
      --queries 5 --lams 128 --modes C:int --lean --tag _nofix
  ;;
should)
  # (e) the rest of zkLLM's 2,048-token list, measured in full: Llama-2-13B, OPT-350M, OPT-2.7B, OPT-13B
  sub z-llama13-2k --mem=128G "$S" llm --model llama2-13b --seq 2048 --builds 1,2,full --queries 3 $L128 --lean
  sub z-opt13b-2k  --mem=128G "$S" llm --model opt-13b    --seq 2048 --builds 1,2,full --queries 3 $L128 --lean
  sub z-opt27-2k   "$S" llm --model opt-2.7b --seq 64 2048 --builds 1,2,full --queries 3 $L128 --lean
  sub z-opt350-2k  "$S" llm --model opt-350m --seq 64 2048 --queries 3 $L128 --lean --builds full
  # (f) a GPU client (verifier on the GPU): the zkLLM-matched rows and the 64-token rows
  for m in opt-125m opt-1.3b opt-6.7b llama2-7b; do
    sub "g-$m-2k" "$S" llm --model "$m" --seq 2048 --builds full --queries 5 $L128 --lean \
        --verifier-device cuda --tag _gpuv
  done
  sub g-llama7-64 "$S" llm --model llama2-7b --seq 64 --builds full --queries 10 $L128 --lean --verifier-device cuda --tag _gpuv
  sub g-gpt2      "$S" llm --model gpt2 --seq 64 512 --queries 10 $L128 --lean --verifier-device cuda --tag _gpuv
  # (g) the shape of the depth dependence (validate_extrapolation.py): 4, 8, 16 blocks
  sub v-llama7-64 "$S" llm --model llama2-7b --seq 64   --builds 4,8,16 --queries 10 $L128 --lean
  sub v-llama7-2k "$S" llm --model llama2-7b --seq 2048 --builds 4,8,16 --queries 5  $L128 --lean
  sub v-opt13-2k  "$S" llm --model opt-1.3b  --seq 2048 --builds 6,12   --queries 5  $L128 --lean
  # (h) ZKTorch's setting (1 token) and zkLLM's host (12 verifier threads)
  sub s-llama7-t1 "$S" llm --model llama2-7b --seq 1 --builds 1,2,full --queries 10 $L128 --lean
  sub s-llama7-thr12 -c 12 --export=ALL,PVI_THREADS=12 "$S" llm --model llama2-7b --seq 2048 --builds full \
      --queries 3 --lams 128 --modes Kpre:int --lean --tag _thr12
  ;;
nice)
  # (i) batching: B prompts / images against one set of openings
  sub n-batch-gpt2   "$S" llm --model gpt2 --seq 64 --queries 3 $L128 --batches 8 32 --batch-trials 3 --lean --tag _batch
  sub n-batch-llama7 "$S" llm --model llama2-7b --seq 64 --builds 1,2,full --queries 3 $L128 --batches 8 32 \
      --batch-trials 3 --lean --tag _batch
  for m in mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar; do
    sub "n-batch-$m" "$S" cnn --model "$m" --queries 30 --tampers 10 --cnn-batches 8 32 64 128 256 --batch-trials 5 --tag _batch
  done
  sub n-batch-resnet18_224 "$S" cnn --model resnet18_224 --queries 30 --tampers 10 --cnn-batches 8 32 64 128 \
      --batch-trials 5 --tag _batch
  # (j) Llama-2's full 4,096-token context, and GPT-2 at its 1,024 maximum
  sub n-llama7-4k --mem=128G "$S" llm --model llama2-7b --seq 4096 --builds 1,2,full --queries 3 $L128 --lean
  sub n-gpt2-1k "$S" llm --model gpt2 --seq 1024 --queries 10 $L128 --lean
  # (k) TF32 tensor cores for the (exact) float32 GEMMs of the forward pass
  sub n-tf32-llama7 --export=ALL,PVI_TF32=1 "$S" llm --model llama2-7b --seq 64 2048 --builds full \
      --queries 5 --lams 128 --modes C:int --lean --tf32 --tag _tf32
  sub n-tf32-gpt2 --export=ALL,PVI_TF32=1 "$S" llm --model gpt2 --seq 64 512 --queries 10 \
      --lams 128 --modes C:int --lean --tf32 --tag _tf32
  # (l) 30-70B shapes: 1-2 blocks, and the full Llama-2-70B at 64 tokens (80 GB card, 64 GiB of int8
  #     weights; bench.py skips a build that cannot fit before building it, and exits non-zero)
  sub n-opt30  "$S" llm --model opt-30b   --seq 64 2048 --queries 2 $L128 --lean
  sub n-opt66  "$S" llm --model opt-66b   --seq 64      --queries 2 $L128 --lean
  sub n-llama70 "$S" llm --model llama2-70b --seq 64 512 --queries 2 $L128 --lean
  sub n-llama70-full --mem=192G -t 12:00:00 "$S" llm --model llama2-70b --seq 64 --builds full --queries 2 $L128 --lean
  ;;
*)
  echo "usage: $0 smoke|must|batch|report-must|should|nice" >&2; exit 2
  ;;
esac
if [ -n "${PVI_LAST_FILE:-}" ]; then        # called by sp2027.sh: hand back the chain's end and the totals
  printf 'sg_dep=%q\nsg_jobs=%d\nsg_hours=%d\nsg_dryn=%d\n' "$DEP" "$NJOBS" $(((MINUTES + 59) / 60)) "$DRYN" \
      > "$PVI_LAST_FILE"
  exit 0
fi
[ -z "${PVI_DRYRUN:-}" ] || [ "$NJOBS" = 0 ] || echo "# $1: $NJOBS jobs, job limits $(((MINUTES + 59) / 60)) h"
[ -n "${PVI_DRYRUN:-}" ] || [ -n "${PVI_LOCAL:-}" ] || squeue -u "$ME"

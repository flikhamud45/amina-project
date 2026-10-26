#!/bin/bash
# The run that produces every number of the paper: one scripted run on one platform (a GPU node; its GPU proves and
# is the GPU verifier, 8 threads of its CPU are the CPU verifier and the re-execution baselines), one job per cell
# group, whose records experiments/paper/numbers.py turns into the paper's tables and macros.  From the repository
# root of a checkout of the tagged commit:
#
#     bash code/experiments/package/run.sh --platform <name> [options] <tier>... | all
#
# Tiers, in the order 'all' runs them (each job's records go to logs/<name>/sp-<job>-<jobid>.out, the benchmark's to
# code/artifacts/comparison/raw_<name>/):
#     smoke    the GPU and native-kernel tests, one V1 query, the host-to-device copy; ends with "SMOKE OK", and no
#              other tier of 'all' starts unless it does
#     cpuv     the CPU verifier against every re-execution baseline on the same 8 threads (llama.cpp's included); V1's
#              CPU verifier
#     v1       V1 against v0 (bytes, prover and verifier times)
#     serve    the prover against unverified serving on the same GPU, single prompts and batches
#     bench    the benchmark's cells: image models and decoders, modes C, K and Kpre, with the attacks
#     gpuv     the GPU verifier, re-execution on the same GPU, the host-to-device bandwidth
#     gen      a generated response proved as one prefill, greedy and sampled
#     batch    B prompts against one opening
#     real     real-weight OPT end to end next to random-weight twins (and Llama-2-7B with --llama-licence)
#     setup    the setup proof of an untrusted commitment
#     improve  each implementation improvement against the path it replaces, on the same queries; 8- against 11-bit
#              attention probabilities
#     quality  the integer models' quality: OPT (perplexity, zero-shot, reproducible build), OpenLLaMA-7B, TinyLlama
#              and Qwen3 (and Llama-2-7B/13B with --llama-licence)
#     tv       the token sampler's total-variation distance to the softmax (CPU)
#     attack   the attack on the path test: MNIST, the image models' path test, OPT-6.7B, the in-range edits, the
#              sampler game
#     long     [optional] prompts of 2,048 to 16,384 tokens, verifier against every baseline
# Options:
#     --platform NAME   required: a new name for this GPU and CPU (letters, digits, '.', '_', '-'; the stored roots
#                       are not written)
#     --dry-run         print every job (and each tier's job count, time limits and estimate); submit nothing, and
#                       only warn about what this host lacks
#     --rehearsal       each tier's smallest cells (GPT-2 / OPT-125M, 2 blocks, short prompts), every script of
#                       every tier: the whole package in about three hours on one GPU, or a platform check
#     --local           no SLURM: every job runs here, in the foreground, one after another, logged as SLURM would;
#                       a job that ended successfully in an earlier --local run of this platform is not run again
#                       (its marker: logs/<name>/done/sp-<cell>); the smoke always runs, since it checks the node
#     --one-job         submit the tiers as one SLURM job that runs their jobs one after another (run.sh --local in
#                       its allocation) with the node to itself (unless --no-exclusive): one wait for a whole node
#                       instead of one per job.  Submitting the same command again, after the job's time limit or a
#                       pre-emption, runs only the jobs that have not ended successfully
#     --max-hours H     --one-job's time limit in hours (default 24, the partition's usual limit; the sum of the
#                       jobs' limits if smaller)
#     --optional        'all' also runs the optional tier (long)
#     --no-chain        do not chain the jobs (default: each job starts after the previous one ends, so no two timed
#                       jobs share the node)
#     --no-exclusive    no --exclusive (default: every job has the node to itself; the smoke and the rehearsal,
#                       whose timings are not used, need not wait for a whole node)
#     --no-pin          do not pin the CPU-timing jobs, the cells that time the CPU verifier or a re-execution on the
#                       CPU (cpuv, long and those of v1, bench, batch, gen, real, setup and improve); by default each
#                       runs on 8 cores of one NUMA node, one hardware thread per core (experiments/package/pin.py)
#     --no-native-ok    submit although CXX names no compiler here; the CPU-timing cells then run without
#                       --require-native and time whatever path the node has (default: refuse)
#     --mem-max GB      the largest --mem any job asks for (default 256; the node's memory)
#     --queries-head N  timed queries after a warm-up for the headline cells (default 5) and --queries N for the
#                       others (default 3)
#     --policy P        mode C's commitment plan for every mode-C cell (default cnn18c)
#     --setup-cache D   the prover's setup cache (default logs/<name>/setup_cache; 'none' disables it): the first job
#                       that commits a model under a plan stores its Merkle leaves and later jobs rebuild the same
#                       commitment from them (the setup tier never uses it)
#     --python PATH     the interpreter (default .venv/bin/python)
#     --extra-pythonpath DIR   packages outside the environment, put after the code on the path
#     --llama-licence   the Llama-2 cells (real, quality): only with your own Llama-2 licence and Hugging Face token
#                       (HF_TOKEN); the rehearsal's one Llama-2 job reads the gated repository from a compute node
#     --bigmem          the long tier also measures the full Llama-2-7B at 8,192 tokens (a node with >= 256 GB)
# The scheduler's settings come from sbatch's own environment variables (SBATCH_PARTITION, SBATCH_GRES=gpu:<type>:1,
# SBATCH_ACCOUNT), the host's from CXX (the native CPU kernels; required) and CMAKE (CMake for the cpuv tier's
# llama.cpp build; default: cmake on the PATH), passed to the jobs as they are, and from PVI_NATIVE_DIR,
# PVI_PY_INCLUDE, HF_HOME (the real, quality, tv and attack tiers download checkpoints and datasets) and
# PVI_CIFAR_ROOT, PVI_IMAGENET_ROOT and PVI_CACHE (the image models' datasets), each passed to the jobs as an absolute
# path (a relative one is taken from the repository root).  HF_TOKEN (the Llama-2 cells) reaches the jobs through
# sbatch's environment and is never written into a job.  Everything is checked before the first job is submitted: the
# compiler, CMake and git (the cpuv tier), the interpreter, HF_HOME (and that no HF_HUB_CACHE, TRANSFORMERS_CACHE or
# HF_DATASETS_CACHE that overrides it is unwritable), the image models and datasets of the bench tier and, under
# SLURM, that the partition has a node with the memory the largest job asks for.  Jobs are named sp-<cell>, with this
# platform as their --comment and --dependency=singleton.  A job of the same name and platform that is still queued
# or running is not submitted again, and the chain and the smoke gate pass through it, so running a tier again
# submits only what is missing; a finished benchmark cell is skipped by bench.py itself.
# experiments/package/README.md is the runner's guide.
set -euo pipefail

usage() { sed -n '2,/^[^#]/{/^#/p}' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }
PLATFORM="" DRY="" REH="" LOCAL="" OPTIONAL="" CHAIN=1 EXCL="--exclusive" PIN=1 NONATIVE="" MEMMAX=256 QH=5 Q=3
POL=cnn18c CACHE="" PY="" EXTRA_PP="" LLAMA="" BIGMEM="" ONEJOB="" MAXH=24 TIERS=()
ARGV=("$@")
while [ $# -gt 0 ]; do
  case "$1" in
    --platform) PLATFORM=$2; shift ;;
    --dry-run) DRY=1 ;;
    --rehearsal) REH=1 ;;
    --local) LOCAL=1 ;;
    --one-job) ONEJOB=1 ;;
    --max-hours) MAXH=$2; shift ;;
    --optional) OPTIONAL=1 ;;
    --no-chain) CHAIN=0 ;;
    --no-exclusive) EXCL="" ;;
    --no-pin) PIN=0 ;;
    --no-native-ok) NONATIVE=1 ;;
    --mem-max) MEMMAX=$2; shift ;;
    --queries-head) QH=$2; shift ;;
    --queries) Q=$2; shift ;;
    --policy) POL=$2; shift ;;
    --setup-cache) CACHE=$2; shift ;;
    --python) PY=$2; shift ;;
    --extra-pythonpath) EXTRA_PP=$2; shift ;;
    --llama-licence) LLAMA=1 ;;
    --bigmem) BIGMEM=1 ;;
    -h|--help) usage 0 ;;
    -*) echo "unknown option $1" >&2; usage ;;
    *) TIERS+=("$1") ;;
  esac
  shift
done
[ -n "$PLATFORM" ] || { echo "--platform <name> is required" >&2; usage; }
[[ "$PLATFORM" =~ ^[A-Za-z0-9][A-Za-z0-9._-]*$ ]] || { echo "--platform: letters, digits, '.', '_', '-'" >&2; exit 2; }
[ "${#TIERS[@]}" -gt 0 ] || usage
[ -z "$ONEJOB" ] || [ -z "$LOCAL" ] || { echo "--one-job submits a job that runs --local itself" >&2; exit 2; }
[[ "$MAXH" =~ ^[1-9][0-9]*$ ]] || { echo "--max-hours: a whole number of hours" >&2; exit 2; }
[ -f code/experiments/package/run.sh ] || { echo "run this from the repository root" >&2; exit 2; }
ROOT="$PWD"
LOGS="logs/$PLATFORM"
PY="${PY:-$ROOT/.venv/bin/python}"
if [ -n "$REH" ]; then QH=2; Q=2; fi
case "$CACHE" in none) CACHE="" ;; "") CACHE="$ROOT/$LOGS/setup_cache" ;; esac
ME="${USER:-$(id -un)}"
# the queued and running jobs: name|platform|id
if [ -n "$DRY" ] || [ -n "$LOCAL" ]; then QUEUED=""; else QUEUED="$(squeue -h -u "$ME" -o '%j|%k|%i')"; fi
[ -n "$DRY" ] || mkdir -p "$LOGS"

# the host's places, as absolute paths (the jobs run in code/; a relative path is taken from the repository root)
for v in HF_HOME PVI_CIFAR_ROOT PVI_IMAGENET_ROOT PVI_CACHE PVI_NATIVE_DIR PVI_PY_INCLUDE; do
  case "${!v:-}" in ""|/*) ;; *) export "$v=$ROOT/${!v}" ;; esac
done
# the jobs' environment: this checkout's code first on the path, 8 threads, no TF32, the caches of this platform
ENVS="cd $ROOT/code && export PYTHONPATH=$ROOT/code/src:$ROOT/code${EXTRA_PP:+:$EXTRA_PP} OMP_NUM_THREADS=8 \
MKL_NUM_THREADS=8 NVIDIA_TF32_OVERRIDE=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
TRITON_CACHE_DIR=$ROOT/$LOGS/triton PVI_NATIVE_DIR=$(printf %q "${PVI_NATIVE_DIR:-$ROOT/$LOGS/native}")"
[ -z "${CXX:-}" ] || ENVS="$ENVS CXX=$(printf %q "$CXX")"
[ -z "${CMAKE:-}" ] || ENVS="$ENVS CMAKE=$(printf %q "$CMAKE")"
for v in PVI_PY_INCLUDE HF_HOME PVI_CIFAR_ROOT PVI_IMAGENET_ROOT PVI_CACHE; do
  [ -z "${!v:-}" ] || ENVS="$ENVS $v=$(printf %q "${!v}")"
done
DEP="" GATE="" NJOBS=0 HOURS=0 DRYN=0 LAST="" SKIPPED=0 DONE=0 PLAN="" MEMNEED=0 PLANH=0 NPLAN=0 CPUJOB=""

sub() {   # sub <name> <hours> <memGB> <command...>: one job, chained after the previous one
  # (the first job submitted after the smoke also waits for the smoke's success: GATE)
  local name="sp-$1" hours=$2 mem=$3 jid dep="singleton" pin=""; shift 3
  [ "$mem" -gt "$MEMMAX" ] && mem=$MEMMAX
  if [ -n "$PLAN" ]; then
    [ "$mem" -le "$MEMNEED" ] || MEMNEED=$mem
    PLANH=$((PLANH + hours)); NPLAN=$((NPLAN + 1))
    return 0
  fi
  jid="$(awk -F'|' -v n="$name" -v p="$PLATFORM" '$1 == n && $2 == p { print $3; exit }' <<<"$QUEUED")"
  if [ -z "$jid" ] && [ -n "$LOCAL" ] && [ "$name" != sp-smoke ] && [ -s "$LOGS/done/$name" ]; then
    # ended successfully in an earlier --local run of this platform (run again after a time limit or a pre-emption)
    jid="$(cat "$LOGS/done/$name")"
    echo "skip $name: ended successfully earlier ($LOGS/$name-$jid.out)"
    DONE=$((DONE + 1))
  elif [ -n "$jid" ]; then
    # queued or running for this platform: not submitted again; the next job waits for it (and for the gate)
    echo "skip $name: already queued (job $jid)"
    SKIPPED=$((SKIPPED + 1))
  else
    [ -n "$DEP" ] && dep="$DEP,$dep"
    [ -n "$GATE" ] && dep="$GATE,$dep"
    GATE=""
    local args=(-o "$LOGS/%x-%j.out" -J "$name" "--comment=$PLATFORM" "--dependency=$dep" --cpus-per-task=8
                "--mem=${mem}G" "--time=${hours}:00:00")
    [ -n "$EXCL" ] && args+=("$EXCL")
    [ -n "$CPUJOB" ] && [ "$PIN" = 1 ] && pin="$PY -m experiments.package.pin --cores 8 --pid \$\$ && "
    local wrap="set -uo pipefail; $ENVS; echo host \$(hostname) git \$(git -C $ROOT log --oneline -1); $pin$*"
    if [ -n "$DRY" ]; then
      DRYN=$((DRYN + 1)); jid="DRY$DRYN"
      echo "sbatch ${args[*]} --wrap \"$wrap\""
    elif [ -n "$LOCAL" ]; then
      # the start time in nanoseconds (one-job runs on several nodes may start a smoke in the same second): a later
      # run's log sorts later, and its records replace the earlier ones (experiments/paper/records.py)
      jid="$(date +%s%N)"
      [[ "$jid" =~ ^[0-9]+$ ]] || printf -v jid '%s%09d' "$(date +%s)" "$RANDOM"     # a date(1) without %N
      echo "$(date '+%F %T') running $name (limit ${hours} h): $LOGS/$name-$jid.out"
      if timeout "${hours}h" bash -c "$wrap" > "$LOGS/$name-$jid.out" 2>&1; then
        mkdir -p "$LOGS/done" && echo "$jid" > "$LOGS/done/$name"
      else
        echo "$name: exit status $?" >&2
      fi
    else
      # the job script goes in on stdin with a bash shebang (--wrap runs /bin/sh, which may not be bash)
      jid="$(printf '#!/bin/bash\n%s\n' "$wrap" | sbatch --parsable "${args[@]}")"; jid="${jid%%;*}"
      echo "submitted $name: job $jid"
    fi
    QUEUED+=$'\n'"$name|$PLATFORM|$jid"
    NJOBS=$((NJOBS + 1)); HOURS=$((HOURS + hours))
  fi
  LAST="$jid"
  [ "$CHAIN" = 1 ] && DEP="afterany:$jid"
  return 0
}

csub() {  # csub <name> <hours> <memGB> <command...>: a CPU-timing job (pinned to 8 cores of one NUMA node)
  CPUJOB=1
  sub "$@"
  CPUJOB=""
}

nl() {    # a model's decoder blocks
  case "$1" in
    gpt2|opt-125m) echo 12 ;; opt-1.3b) echo 24 ;; opt-6.7b|llama2-7b) echo 32 ;; qwen3-4b) echo 36 ;;
    llama2-13b) echo 40 ;; *) echo "nl: unknown model $1" >&2; exit 2 ;;
  esac
}

est() {   # a tier's expected hours on an L40S node (measured times of the same cells)
  if [ -n "$REH" ]; then
    case "$1" in improve) echo "<1" ;; quality) echo "<0.5" ;; *) echo "<0.3" ;; esac
    return
  fi
  case "$1" in
    smoke) echo 1 ;; cpuv) echo 7-10 ;; v1) echo 4-8 ;; serve) echo 8-10 ;; bench) echo 8-12 ;; gpuv) echo 4-6 ;;
    gen) echo 2-3 ;; batch) echo 2-3 ;; real) echo 8-12 ;; setup) echo 1-2 ;; improve) echo 3-5 ;;
    quality) echo 10-16 ;; tv) echo 1-2 ;; attack) echo 4-6 ;; long) echo 4-7 ;;
  esac
}

M="$PY -m experiments"
CACHED="${CACHE:+--setup-cache $CACHE}"
NATIVE="--require-native"; [ -z "$NONATIVE" ] || NATIVE=""
PERF="$M.costs.perf --threads 8 $CACHED"
V1="$M.v1.v1_bytes --full --threads 8 $CACHED"
RX="$M.reexec.reexec --threads 8"
RXCPU="int int_mm int_fp32gemm float bf16 dynint8"   # the re-execution variants on a CPU (dynint8 is another model)
LCPP="$M.reexec.llamacpp --threads 8 --llamacpp-dir $ROOT/$LOGS/llamacpp"   # llama.cpp (another model), cached build
GEN="$M.generation.gen_bench --threads 8 $CACHED"
BENCH="$M.costs.bench --platform $PLATFORM --threads 8 $CACHED"
H2D="$M.reexec.h2d_bandwidth"

tier_smoke() {
  # the platform's code paths: Triton (the GKR prover, the GPU decoder, fused attention) and int8 GEMMs on this GPU,
  # the native -march=native kernels on this CPU (CXX), the tree hashing, the serving and package harnesses, sampled
  # generation, the setup proof; then one V1 query, the native and int8 self-checks and the host-to-device copy
  local t="tests/test_fs_tree.py tests/test_serve_ab.py tests/test_run_package.py tests/test_options.py"
  [ -n "$REH" ] || t="tests/test_gkr_triton.py tests/test_gkr_triton_cpu.py tests/test_gpu_verifier.py \
tests/test_gpu_exactness.py tests/test_v1_protocol.py tests/test_token_sampler.py tests/test_extfield.py \
tests/test_logup.py tests/test_cut.py tests/test_native_cheap.py tests/test_v1_products.py tests/test_generation.py \
tests/test_setup_proof.py tests/test_reexec.py $t"
  # (pytest's whole output goes to smoke_pytest.txt, its end to the log)
  local native=""
  [ -n "$NONATIVE" ] || native="$PY -c 'import sys; from pvi.kernels import native; \
sys.exit(0 if native.available() else \"native kernels unavailable with CXX=${CXX:-}\")' && "
  sub smoke 2 64 "PVI_TEST_DEVICE=cuda $PY -m pytest -o addopts= -q -rfE --tb=short -p no:cacheprovider $t 2>&1 \
| tee $ROOT/$LOGS/smoke_pytest.txt | tail -40 && ${native}$PY -c 'import sys; from pvi.core import field; \
ok = field.int8_ok(\"cuda\"); print(\"int8 GEMMs exact on this GPU:\", ok); \
sys.exit(0 if ok else \"int8 GEMMs are not exact on this GPU\")' && \
$V1 --model gpt2 --layers 2 --seq 64 --modes C Kpre --policy $POL --reps 2 --lmax 20 --require-triton | grep '^{' && \
$H2D --sizes-mb 256 --reps 3 --label smoke && echo SMOKE OK"
}

tier_cpuv() {
  # the CPU verifier (native kernels) of the full models in Kpre and C, the prover on the GPU; every re-execution
  # baseline on the same 8 threads (the full model and 2 blocks), with the allocator setting the verifier applies to
  # itself for a like-for-like row (the other rows with glibc's defaults), and llama.cpp's prefill of Llama-2-7B's
  # shapes (Q8_0 and bf16 weights: inexact, another model); V1's optimised and baseline CPU verifiers on the same
  # queries
  if [ -n "$REH" ]; then
    csub c-gpt2-64 1 64 "$PERF --model gpt2 --layers 2 --seq 64 --modes Kpre C --policy $POL --queries $QH \
--verifier-device cpu --label cpuv $NATIVE"
    csub c-rx-gpt2 1 64 "$RX --model gpt2 --seq 64 --layers 1 2 --variants $RXCPU --device cpu --reps 2 \
--no-malloc-tune $NATIVE"
    csub c-v1-gpt2-64 1 64 "$V1 --model gpt2 --layers 2 --seq 64 --modes Kpre --reps $QH --lmax 20 \
--v1-verifier optimised baseline --label cpuv-v1 $NATIVE"
    csub c-llamacpp-llama2-7b-L1 1 64 "$LCPP --model llama2-7b --layers 1 --seq 64 --types q8_0 bf16 --reps $QH"
    return 0
  fi
  local m t
  for m in llama2-7b qwen3-4b opt-1.3b; do
    csub "c-$m-64" 1 128 "$PERF --model $m --seq 64 --modes Kpre C --policy $POL --queries $QH --verifier-device cpu \
--label cpuv $NATIVE"
    csub "c-$m-2048" 2 256 "$PERF --model $m --seq 2048 --modes Kpre C --policy $POL --queries $QH \
--verifier-device cpu --label cpuv $NATIVE"
  done
  for m in llama2-7b qwen3-4b opt-1.3b; do
    t=2; [ "$m" = opt-1.3b ] && t=1; [ "$m" = llama2-7b ] && t=3
    csub "c-rx-$m" "$t" 256 "$RX --model $m --seq 64 2048 --layers 2 $(nl "$m") --variants $RXCPU --device cpu \
--reps $QH --no-malloc-tune $NATIVE"
  done
  csub c-rx-heap-llama2-7b 2 256 "$RX --model llama2-7b --seq 64 2048 --layers 2 32 --variants int int_mm float \
dynint8 --device cpu --reps $QH --malloc-tune $NATIVE"
  csub c-llamacpp-llama2-7b 3 64 "$LCPP --model llama2-7b --seq 64 2048 --types q8_0 bf16 --reps $QH"
  for t in 64 2048; do
    csub "c-v1-llama2-7b-$t" 2 256 "$V1 --model llama2-7b --seq $t --modes Kpre --reps $QH \
--v1-verifier optimised baseline --label cpuv-v1 $NATIVE"
  done
  csub c-v1-qwen3-4b-2048 2 256 "$V1 --model qwen3-4b --seq 2048 --modes Kpre --reps $QH \
--v1-verifier optimised baseline --label cpuv-v1 $NATIVE"
}

tier_v1() {
  # V1 against v0 on the same queries: bytes (exact accounting), prover and verifier times (the CPU verifier on 8
  # threads, the GKR on the GPU), mode C under --policy
  if [ -n "$REH" ]; then
    csub v1-gpt2-64 1 64 "$V1 --model gpt2 --layers 2 --seq 64 --modes C Kpre --policy $POL --reps $Q --lmax 20 \
$NATIVE"
    return 0
  fi
  local t
  csub v1-gpt2-64 1 64 "$V1 --model gpt2 --seq 64 --modes C Kpre --policy $POL --reps $Q $NATIVE"
  csub v1-gpt2-512 1 64 "$V1 --model gpt2 --seq 512 --modes C Kpre --policy $POL --reps $Q $NATIVE"
  for t in 64 2048; do
    csub "v1-opt125-$t" 1 96 "$V1 --model opt-125m --seq $t --modes C Kpre --policy $POL --reps $Q $NATIVE"
    csub "v1-opt13-$t" 2 192 "$V1 --model opt-1.3b --seq $t --modes C Kpre --policy $POL --reps $QH $NATIVE"
    csub "v1-llama7-$t" 3 256 "$V1 --model llama2-7b --seq $t --modes Kpre C --policy $POL --reps $QH $NATIVE"
    csub "v1-opt67-$t" 3 256 "$V1 --model opt-6.7b --seq $t --modes Kpre C --policy $POL --reps $Q $NATIVE"
  done
  csub v1-llama13-64 3 256 "$V1 --model llama2-13b --seq 64 --modes Kpre C --policy $POL --reps $Q $NATIVE"
  csub v1-llama13-2k-2 1 256 "$V1 --model llama2-13b --layers 2 --seq 2048 --modes Kpre C --policy $POL --reps $Q \
$NATIVE"
  csub v1-qwen4-64 2 192 "$V1 --model qwen3-4b --seq 64 --modes Kpre C --policy $POL --reps $Q $NATIVE"
}

tier_serve() {
  # the prover against unverified serving on this GPU on the same token ids: our integer model with every claim
  # dropped (U1), eager fp16 and bf16 decoders (U2), the prover in modes C and Kpre; prefill at B = 1 (workload a)
  # and batched (b), a batch proved query by query (seq) and as one statement (joint); 10 repetitions
  local serve="$M.deployment.serve_ab --workload a b --modes C Kpre --batch-proof seq joint --reps 10 \
--device cuda --threads 8 --u2-dtype fp16 bf16 --policy $POL --label serve $CACHED"
  if [ -n "$REH" ]; then
    sub serve-rehearsal 1 64 "$serve --model opt-125m --layers 2 --cells 64x1 64x4 --reps 3"
    return 0
  fi
  sub serve-opt13 3 128 "$serve --model opt-1.3b"
  sub serve-qwen4 4 256 "$serve --model qwen3-4b"
  sub serve-llama7 5 256 "$serve --model llama2-7b"
  sub serve-opt67 5 256 "$serve --model opt-6.7b"
  sub serve-llama13 8 256 "$serve --model llama2-13b"
}

tier_bench() {
  # the benchmark's cells of tab:eval-llm, tab:eval-compare and the outcome counts: full models only, Fiat-Shamir
  # (C:fs, Kpre:fs); interactive cells (:int) only where a table compares with an interactive system (zkLLM's
  # 2,048-token rows, Maverick's Qwen3-4B row).  The CPU verifier and the streaming GPU verifier (--verifier-device
  # cuda --verifier-impl stream); 5 timed queries for the 64-token cells and OPT-1.3B and Llama-2-7B at 2,048 tokens.
  # The CPU verifier's cells are CPU-timing jobs (csub, $NATIVE).
  local gpuv="--verifier-device cuda --verifier-impl stream" m t mem q cm hc hk hg
  if [ -n "$REH" ]; then
    csub b-c-lenet5 1 64 "$BENCH cnn --model lenet5 --queries 3 --tampers 3 --modes C:fs,C:int --policy $POL $NATIVE"
    csub b-c-gpt2-64 1 64 "$BENCH llm --model gpt2 --seq 64 --builds 2 --queries 2 --lams 128 --modes C:fs,C:int \
--llm-tampers 1 --policy $POL $NATIVE"
    csub b-k-gpt2-64 1 64 "$BENCH llm --model gpt2 --seq 64 --builds 2 --queries 2 --lams 128 --modes Kpre:fs $NATIVE"
    sub b-cg-gpt2-64 1 64 "$BENCH llm --model gpt2 --seq 64 --builds 2 --queries 2 --lams 128 --modes C:fs \
--policy $POL $gpuv"
    csub b-k-gpt2-8-thr1 1 64 "$BENCH llm --model gpt2 --seq 8 --builds 2 --queries 2 --lams 40 --modes Kpre:int \
--threads 1 --no-wire $NATIVE"
    return 0
  fi
  # (a) the image models: mode C with the plan, with the attacks and the path test's cells; then K and Kpre
  for m in mlp_mnist lenet5 vgg11 vgg16 resnet18_cifar resnet18_224; do
    t=100; [ "$m" = resnet18_224 ] && t=60
    csub "b-c-$m" 1 96 "$BENCH cnn --model $m --queries 30 --tampers $t --modes C:int,C:fs --policy $POL $NATIVE"
    csub "b-k-$m" 1 96 "$BENCH cnn --model $m --queries 30 --tampers 0 --modes K:int,Kpre:int $NATIVE"
  done
  # (b) 64-token prompts with the attacks on the full models; GPT-2 at 512 tokens; Llama-2-7B at 1 token
  for m in gpt2 qwen3-4b llama2-7b llama2-13b; do
    mem=96; [ "$m" = llama2-13b ] && mem=128
    hc=1; hk=1; hg=1; [ "$m" = llama2-13b ] && hc=2
    csub "b-c-$m-64" "$hc" "$mem" "$BENCH llm --model $m --seq 64 --builds full --queries 5 --lams 128 \
--modes C:fs --llm-tampers 30 --policy $POL $NATIVE"
    csub "b-k-$m-64" "$hk" "$mem" "$BENCH llm --model $m --seq 64 --builds full --queries 5 --lams 128 \
--modes Kpre:fs $NATIVE"
    sub "b-cg-$m-64" "$hg" "$mem" "$BENCH llm --model $m --seq 64 --builds full --queries 5 --lams 128 --modes C:fs \
--policy $POL $gpuv"
  done
  csub b-c-gpt2-512 1 96 "$BENCH llm --model gpt2 --seq 512 --builds full --queries 5 --lams 128 --modes C:fs \
--policy $POL $NATIVE"
  csub b-c-llama2-7b-1 1 96 "$BENCH llm --model llama2-7b --seq 1 --builds full --queries 5 --lams 128 --modes C:fs \
--policy $POL $NATIVE"
  # Maverick's setting (a fresh challenge per query): Kpre, lambda 40, one thread, without and with the encoding
  csub b-k-qwen3-4b-8-thr1 1 96 "$BENCH llm --model qwen3-4b --seq 8 --builds full --queries 5 --lams 40 \
--modes Kpre:int --threads 1 --no-wire $NATIVE"
  csub b-k-qwen3-4b-8-thr1-wire 1 96 "$BENCH llm --model qwen3-4b --seq 8 --builds full --queries 5 --lams 40 \
--modes Kpre:int --threads 1 $NATIVE"
  # (c) 2,048-token prompts (zkLLM's rows interactive as well)
  for m in opt-125m opt-1.3b opt-6.7b llama2-7b llama2-13b; do
    q=3; mem=96; cm="C:fs,C:int"; hc=1; hk=1; hg=1
    case "$m" in
      opt-1.3b) q=5 ;;
      opt-6.7b) cm="C:fs" ;;
      llama2-7b) q=5; mem=128; hc=2 ;;
      llama2-13b) mem=128; hc=3; hg=2 ;;
    esac
    csub "b-c-$m-2k" "$hc" "$mem" "$BENCH llm --model $m --seq 2048 --builds full --queries $q --lams 128 \
--modes $cm --policy $POL $NATIVE"
    csub "b-k-$m-2k" "$hk" "$mem" "$BENCH llm --model $m --seq 2048 --builds full --queries $q --lams 128 \
--modes Kpre:fs $NATIVE"
    sub "b-cg-$m-2k" "$hg" "$mem" "$BENCH llm --model $m --seq 2048 --builds full --queries $q --lams 128 \
--modes $cm --policy $POL $gpuv"
  done
}

tier_gpuv() {
  # the streaming GPU verifier (one process per mode: the prover and the client share the card) and V1's GPU verifier
  # at 2,048 tokens; the integer forward pass with int8 GEMMs and fp16/bf16 decoders of every table model at 64, 512
  # and 2,048 tokens (2 blocks and the full model); the host-to-device copy, pinned and pageable
  if [ -n "$REH" ]; then
    sub g-v0-gpt2 1 64 "$PERF --model gpt2 --layers 2 --seq 64 --modes Kpre C --policy $POL --queries $Q \
--verifier-device cuda --stream --label gpuv --require-triton"
    sub g-v1-gpt2 1 64 "$V1 --model gpt2 --layers 2 --seq 64 --modes Kpre --reps $Q --lmax 20 --verifier-device cuda \
--label gpuv-v1"
    sub g-rx-gpt2 1 64 "$RX --model gpt2 --seq 64 --layers 1 2 --variants int float bf16 --device cuda --reps 2"
    sub g-h2d 1 32 "$H2D --sizes-mb 64 256 --reps 3 --label gpuv"
    return 0
  fi
  local m mode
  for m in opt-1.3b llama2-7b; do
    for mode in Kpre C; do
      sub "g-v0-$m-$mode" 1 192 "$PERF --model $m --seq 2048 --modes $mode --policy $POL --queries $Q \
--verifier-device cuda --stream --label gpuv --require-triton"
    done
    sub "g-v1-$m" 2 192 "$V1 --model $m --seq 2048 --modes Kpre --reps $Q --verifier-device cuda --label gpuv-v1"
  done
  for m in opt-1.3b qwen3-4b llama2-7b opt-6.7b llama2-13b; do
    sub "g-rx-$m" 2 128 "$RX --model $m --seq 64 512 2048 --layers 2 $(nl "$m") --variants int float bf16 \
--device cuda --reps $QH"
  done
  sub g-h2d 1 32 "$H2D --sizes-mb 64 1024 4096 --reps 10 --label gpuv"
}

tier_gen() {
  # a response of G tokens after a P-token prompt proved as one prefill, greedy and sampled with the client's seed,
  # modes C and Kpre; each cell also runs both tamper cells (a wrong token; logits forged to choose it); GPT-2 after a
  # 64-token prompt gives the prover's time per generated token (the comparison with zkAgent's GPT-2)
  if [ -n "$REH" ]; then
    csub gen-gpt2 1 64 "$GEN --model gpt2 --layers 2 --prompt 16 --gen 4 --decode greedy sample --modes C Kpre \
--policy $POL --queries $Q --label gen $NATIVE"
    return 0
  fi
  local m
  csub gen-gpt2 1 64 "$GEN --model gpt2 --prompt 64 --gen 256 --decode greedy --modes C Kpre --policy $POL \
--queries $Q --label gen $NATIVE"
  for m in llama2-7b qwen3-4b; do
    csub "gen-$m" 3 192 "$GEN --model $m --prompt 512 --gen 64 256 --decode greedy sample --modes C Kpre \
--policy $POL --queries $Q --label gen $NATIVE"
  done
}

tier_batch() {
  # B prompts of one client against one fold and one opening (B = 1 is each cell's own queries), modes C and Kpre
  # under Fiat-Shamir, 3 trials per B; the interactive Llama-2-7B 64-token cell checks the batched byte model
  if [ -n "$REH" ]; then
    csub bt-gpt2-c 1 64 "$BENCH llm --model gpt2 --seq 64 --builds 2 --queries 2 --lams 128 --modes C:fs --batches 4 \
--batch-trials 2 --policy $POL $NATIVE"
    csub bt-gpt2-k 1 64 "$BENCH llm --model gpt2 --seq 64 --builds 2 --queries 2 --lams 128 --modes Kpre:fs \
--batches 4 --batch-trials 2 $NATIVE"
    return 0
  fi
  local b="--builds full --queries 3 --lams 128 --batch-trials 3 $NATIVE"
  csub bt-llama2-7b-64-c 1 128 "$BENCH llm --model llama2-7b --seq 64 $b --modes C:fs --batches 4 16 64 --policy $POL"
  csub bt-llama2-7b-64-cint 1 128 "$BENCH llm --model llama2-7b --seq 64 $b --modes C:int --batches 4 16 64 \
--policy $POL"
  csub bt-llama2-7b-64-k 1 128 "$BENCH llm --model llama2-7b --seq 64 $b --modes Kpre:fs --batches 4 16 64"
  csub bt-opt-1.3b-512-c 1 96 "$BENCH llm --model opt-1.3b --seq 512 $b --modes C:fs --batches 4 16 --policy $POL"
  csub bt-opt-1.3b-512-k 1 96 "$BENCH llm --model opt-1.3b --seq 512 $b --modes Kpre:fs --batches 4 16"
  csub bt-llama2-7b-2k-c 2 192 "$BENCH llm --model llama2-7b --seq 2048 $b --modes C:fs --batches 2 4 --policy $POL"
  csub bt-llama2-7b-2k-k 1 192 "$BENCH llm --model llama2-7b --seq 2048 $b --modes Kpre:fs --batches 2 4"
}

tier_real() {
  # real-weight OPT end to end, each cell next to its random-weight twin: the certified integer OPT (its calibration:
  # four 2,048-token train windows), last block pruned, 3 of each tamper per cell; with --llama-licence also the real
  # Llama-2-7B (the main recipe and its fixed public prefix): v0 in Kpre and C, V1 bytes, generated responses
  local r13="--model opt-1.3b --real facebook/opt-1.3b" r67="--model opt-6.7b --real facebook/opt-6.7b" t mode
  if [ -n "$REH" ]; then
    local r125="--model opt-125m --real facebook/opt-125m"
    csub r-opt125-v0 1 64 "$PERF $r125 --seq 64 --modes Kpre C --policy $POL --queries $Q --verifier-device cpu \
--tampers 1 --label real $NATIVE"
    csub r-opt125-v1 1 64 "$V1 $r125 --seq 64 --modes Kpre --reps $Q --tampers 1 --lmax 20 --label real $NATIVE"
    csub r-opt125-gen 1 64 "$GEN $r125 --prompt 16 --gen 4 --decode greedy sample --modes C Kpre --policy $POL \
--queries $Q --verifier-device cpu --label real $NATIVE"
    sub r-opt125-codec 1 64 "$V1 $r125 --seq 64 --codec-only --device cpu --label real"
    return 0
  fi
  for t in 64 2048; do
    csub "r-opt67-v0-$t" 2 192 "$PERF $r67 --seq $t --modes Kpre --queries $Q --verifier-device cpu --tampers 3 \
--label real $NATIVE"
    csub "r-opt67rnd-v0-$t" 1 192 "$PERF --model opt-6.7b --seq $t --modes Kpre --queries $Q --verifier-device cpu \
--tampers 3 --label real-random $NATIVE"
    csub "r-opt67-v1-$t" 2 256 "$V1 $r67 --seq $t --modes Kpre --reps $Q --tampers 3 --label real $NATIVE"
    csub "r-opt67rnd-v1-$t" 2 256 "$V1 --model opt-6.7b --seq $t --modes Kpre --reps $Q --tampers 3 \
--label real-random $NATIVE"
    for mode in C Kpre; do
      csub "r-opt13-v1-$mode-$t" 1 192 "$V1 $r13 --seq $t --modes $mode --policy $POL --reps $Q --tampers 3 \
--label real $NATIVE"
      csub "r-opt13rnd-v1-$mode-$t" 1 192 "$V1 --model opt-1.3b --seq $t --modes $mode --policy $POL --reps $Q \
--tampers 3 --label real-random $NATIVE"
    done
  done
  sub r-opt67-codec 1 192 "$V1 $r67 --seq 2048 --codec-only --device cpu --label real"
  if [ -z "$LLAMA" ]; then
    echo "tier real: no real Llama-2-7B cells (they need your own Llama-2 licence: --llama-licence)" >&2
    return 0
  fi
  local l7="--model llama2-7b --real meta-llama/Llama-2-7b-hf"
  for t in 64 2048; do
    csub "r-llama7-v0-$t" 3 256 "$PERF $l7 --seq $t --modes Kpre --queries $Q --verifier-device cpu --tampers 3 \
--label real $NATIVE"
    csub "r-llama7rnd-v0-$t" 1 192 "$PERF --model llama2-7b --seq $t --modes Kpre --queries $Q --verifier-device cpu \
--tampers 3 --label real-random $NATIVE"
    csub "r-llama7-v1-$t" 3 256 "$V1 $l7 --seq $t --modes Kpre --reps $Q --tampers 3 --label real $NATIVE"
  done
  csub r-llama7-v0C-64 3 256 "$PERF $l7 --seq 64 --modes C --policy $POL --queries $Q --verifier-device cpu \
--tampers 3 --label real $NATIVE"
  csub r-llama7rnd-v0C-64 1 192 "$PERF --model llama2-7b --seq 64 --modes C --policy $POL --queries $Q \
--verifier-device cpu --tampers 3 --label real-random $NATIVE"
  csub r-llama7-gen 4 256 "$GEN $l7 --prompt 512 --gen 64 --decode greedy sample --modes C Kpre --policy $POL \
--queries $Q --verifier-device cpu --label real $NATIVE"
}

tier_setup() {
  # the setup proof of an untrusted commitment: Llama-2-7B with radii for a = 100 amortised queries, prover on the
  # GPU and verifier on 8 CPU threads; GPT-2 at a = 100 and 1,000 (never through the setup cache)
  local st="$M.costs.setup_timing --policy $POL --threads 8 --label setup"
  if [ -n "$REH" ]; then
    csub setup-gpt2 1 64 "$st --model gpt2 --layers 2 --amortise 100 --reps 1"
    return 0
  fi
  csub setup-llama2-7b 3 128 "$st --model llama2-7b --amortise 100 --reps 1"
  csub setup-gpt2 1 64 "$st --model gpt2 --amortise 100 1000 --reps 3"
}

tier_improve() {
  # tab:eval-improve: each improvement against the path it replaces on the same queries (the options of pvi.options
  # select the path); the CUDA time shares of a GPU verifier query; the Kpre precomputation; V1's verifier per phase
  # (optimised and baseline) and the GKR prover per leaf; 8- against 11-bit attention probabilities (the proof's size
  # and the prover's and CPU verifier's times on the same queries, modes Kpre and C)
  local a="--queries $Q --label improve" m
  if [ -n "$REH" ]; then
    sub i-open-gpt2 1 64 "$M.improvements.open_ab --model gpt2 --layers 2 --queries 3"
    sub i-forward-gpt2 1 64 "$M.improvements.forward_ab --model gpt2 --layers 2 --seq 64"
    sub i-encode-gpt2 1 64 "$M.improvements.encode_ab --model gpt2 --layers 2 --seq 64 --fused"
    csub i-fold-gpt2 1 64 "$M.improvements.fold_ab --model gpt2 --layers 2 --devices cpu cuda --reps 1 --threads 8"
    sub i-v1phases-gpt2 1 64 "$M.v1.v1_phases --model gpt2 --layers 2 --seq 64 --mode Kpre --reps 2 --device cuda \
--v1-verifier optimised baseline"
    # the A/B cells of perf.py with the full cells' flags, on an OPT and a Llama shape (2 blocks and 1) at 64 tokens
    for m in "opt-125m --layers 2" "llama2-7b --layers 1"; do
      sub "i-attn-${m%% *}" 1 64 "$PERF --model $m --seq 64 --modes Kpre --prune-last off $a --verifier-device cuda \
--stream --no-fused-attention --label attn-before && $PERF --model $m --seq 64 --modes Kpre --prune-last off $a \
--verifier-device cuda --stream --label attn-after"
      sub "i-gpuv-${m%% *}" 1 64 "$PERF --model $m --seq 64 --modes Kpre $a --verifier-device cuda --stream \
--no-gpu-decode --no-fused-attention --label gpuv-before && $PERF --model $m --seq 64 --modes Kpre $a \
--verifier-device cuda --stream --label gpuv-after"
    done
    csub i-cpuv-llama2-7b 1 64 "$PERF --model llama2-7b --layers 1 --seq 64 --modes Kpre $a --verifier-device cpu \
--no-native --no-malloc-tune --label cpuv-before && $PERF --model llama2-7b --layers 1 --seq 64 --modes Kpre $a \
--verifier-device cpu --label cpuv-after $NATIVE"
    sub i-profile-opt-125m 1 64 "$M.improvements.gpu_profile --model opt-125m --layers 2 --seq 64"
    sub i-gkr 1 32 "$M.v1.gkr_bench --n 20 --check 20"
    csub i-probbits-gpt2 1 64 "$PERF --model gpt2 --layers 2 --seq 64 --modes Kpre C --policy $POL \
--prob-max 255 2047 --queries $Q --verifier-device cpu --label prob-bits $NATIVE"
    return 0
  fi
  sub i-open-llama2-7b 1 128 "$M.improvements.open_ab --model llama2-7b --layers 1 4 --queries 7"
  sub i-open-llama2-13b 1 128 "$M.improvements.open_ab --model llama2-13b --layers 1 --queries 7"
  sub i-open-gpt2 1 64 "$M.improvements.open_ab --model gpt2 --queries 7"
  sub i-forward-llama2-7b 1 128 "$M.improvements.forward_ab --model llama2-7b --layers 2 --seq 2048"
  sub i-encode-opt-1.3b 1 192 "$M.improvements.encode_ab --model opt-1.3b --layers 24 --seq 2048 --fused"
  sub i-encode-llama2-7b 1 128 "$M.improvements.encode_ab --model llama2-7b --layers 2 --seq 2048"
  for m in "opt-1.3b --layers 12" "llama2-7b --layers 2"; do
    sub "i-attn-${m%% *}" 1 128 "$PERF --model $m --seq 2048 --modes Kpre --prune-last off $a --verifier-device cuda \
--stream --no-fused-attention --label attn-before && $PERF --model $m --seq 2048 --modes Kpre --prune-last off $a \
--verifier-device cuda --stream --label attn-after"
    sub "i-gpuv-${m%% *}" 1 128 "$PERF --model $m --seq 2048 --modes Kpre $a --verifier-device cuda --stream \
--no-gpu-decode --no-fused-attention --label gpuv-before && $PERF --model $m --seq 2048 --modes Kpre $a \
--verifier-device cuda --stream --label gpuv-after"
  done
  csub i-cpuv-llama2-7b 1 128 "$PERF --model llama2-7b --layers 2 --seq 2048 --modes Kpre $a --verifier-device cpu \
--no-native --no-malloc-tune --label cpuv-before && $PERF --model llama2-7b --layers 2 --seq 2048 --modes Kpre $a \
--verifier-device cpu --label cpuv-after $NATIVE"
  sub i-profile-opt-1.3b 1 128 "$M.improvements.gpu_profile --model opt-1.3b --layers 12 --seq 2048"
  csub i-fold-llama2-7b 2 192 "$M.improvements.fold_ab --model llama2-7b --devices cpu cuda --reps 3 --threads 8"
  sub i-v1phases-llama2-7b 2 192 "$M.v1.v1_phases --model llama2-7b --layers 2 --seq 2048 --mode Kpre --reps 3 \
--device cuda --v1-verifier optimised baseline"
  sub i-gkr 1 64 "$M.v1.gkr_bench --n 20 22 24 25 --check 20"
  csub i-probbits-llama2-7b 1 128 "$PERF --model llama2-7b --layers 2 --seq 2048 --modes Kpre C --policy $POL \
--prob-max 255 2047 --queries $QH --verifier-device cpu --label prob-bits $NATIVE"
}

tier_quality() {
  # the integer models' quality on WikiText-2 (raw) at 2,048 tokens: the certified OPT models (tab:quality's OPT rows,
  # the 8- against 11-bit attention probabilities), the recipe's alpha on the validation split, the percentile and
  # smooth-max recipe studies of the extended version (OPT-1.3B and OPT-125M on the test split), zero-shot accuracy
  # (tab:zeroshot), the reproducible build;
  # OpenLLaMA-7B, TinyLlama and Qwen3 with the main Llama recipe; with --llama-licence Llama-2-7B and Llama-2-13B.
  # The fp32 CPU references of Qwen3-4B and OpenLLaMA-7B are kept in logs/<name>/ref_*.pt (the second pass and a
  # re-run reuse them).  The rehearsal with --llama-licence reads the gated Llama-2 repository from a job (its
  # configuration and tokenizer, for the smoke model).
  local P="$M.quality.ppl_opt --device cuda" L="$M.quality.ppl_llama --device cuda --calib-device cpu --threads 8"
  local Z="$M.quality.zeroshot --device cuda" m
  if [ -n "$REH" ]; then
    sub q-opt125 1 64 "$P --model facebook/opt-125m --windows 2 --mass-windows 1 && $P --model facebook/opt-125m \
--recipe legacy --graphs smooth --pbits 11 --windows 2 --mass-windows 1"
    sub q-smoke 1 64 "$L --model Qwen/Qwen3-0.6B-Base --smoke --seq 128 --windows 2 --calib-windows 1 --chunk 64"
    sub q-zeroshot-opt-125m 1 64 "$Z --model facebook/opt-125m --limit 20 && $Z --model facebook/opt-125m \
--recipe legacy --limit 20"
    sub q-digest-opt125 1 64 "$M.quality.digest_check --model facebook/opt-125m --threads 8 --commit \
--commit-device cuda"
    sub q-diag-qwen06 1 64 "$M.quality.llama_diag --model Qwen/Qwen3-0.6B-Base --recipes smooth55 scalar4 && \
$M.quality.llama_diag --model Qwen/Qwen3-0.6B-Base --recipes smooth55 scalar4 --prefix 151643,151643 && \
$M.quality.llama_attrib --model Qwen/Qwen3-0.6B-Base --blocks all --recipe smooth55"
    sub q-qwen06 1 64 "$L --model Qwen/Qwen3-0.6B-Base --seq 128 --windows 2 --graphs smooth --glu-exps 8 \
--ref-device cpu --ref-cache $ROOT/$LOGS/ref_qwen3-0.6b.pt --prefix 151643,151643"
    [ -z "$LLAMA" ] || sub q-llama-access 1 32 "$L --model meta-llama/Llama-2-7b-hf --smoke --seq 128 --windows 2 \
--calib-windows 1 --chunk 64"
    return 0
  fi
  sub q-opt125 2 64 "$P --model facebook/opt-125m"
  sub q-opt13 3 96 "$P --model facebook/opt-1.3b"
  sub q-opt67 6 192 "$P --model facebook/opt-6.7b --ref-device cpu"
  sub q-opt-validation 4 96 "$P --model facebook/opt-125m --split validation --graphs smooth --alphas 0.5 0.6 0.7 \
--pbits 11 && $P --model facebook/opt-1.3b --split validation --graphs smooth --alphas 0.5 0.6 0.7 --pbits 11"
  sub q-opt67-validation 5 192 "$P --model facebook/opt-6.7b --split validation --graphs smooth --alphas 0.5 0.6 \
--pbits 11 --ref-device cpu"
  sub q-opt13-recipes 6 96 "$P --model facebook/opt-1.3b --recipe legacy --graphs smooth rows --pbits 11 \
--draws 0 1 2 && $P --model facebook/opt-1.3b --graphs smooth --alphas 0.3 0.4 0.5 0.6 0.7 --pbits 11 --draws 0 1"
  sub q-opt125-recipes 1 64 "$P --model facebook/opt-125m --recipe legacy --graphs smooth --pbits 11 && \
$P --model facebook/opt-125m --graphs smooth --alphas 0.3 0.4 0.5 0.6 0.7 --pbits 11"
  sub q-opt125-128 1 64 "$P --model facebook/opt-125m --calibration test-end --seq 128 --windows 16 --recipe legacy \
--graphs scalar smooth rows --alphas 0.55 0.5 --pbits 8 && $P --model facebook/opt-125m --calibration test-end \
--seq 128 --windows 16 --recipe legacy --graphs smooth --alphas 0.55 --cle-mlp 0 --cle-ratio 2 --pbits 8"
  for m in opt-125m opt-1.3b opt-6.7b; do
    sub "q-zeroshot-$m" 3 192 "$Z --model facebook/$m"
  done
  sub q-zeroshot-legacy-opt-1.3b 2 96 "$Z --model facebook/opt-1.3b --recipe legacy"
  sub q-digest 1 64 "$M.quality.digest_check --model facebook/opt-125m --threads 1 && \
$M.quality.digest_check --model facebook/opt-125m --threads 8 --commit --commit-device cuda && \
$M.quality.digest_check --model facebook/opt-1.3b --threads 8 --commit --commit-device cuda"
  sub q-openllama7 4 192 "$L --model openlm-research/open_llama_7b_v2 --config llama2-7b --structure --ref-device cpu \
--ref-cache $ROOT/$LOGS/ref_open_llama_7b_v2.pt"
  sub q-tinyllama 2 96 "$L --model TinyLlama/TinyLlama_v1.1"
  sub q-qwen17 2 96 "$L --model Qwen/Qwen3-1.7B-Base && $L --model Qwen/Qwen3-1.7B-Base --glu-smooths 0.4 --glu-exps 8"
  sub q-qwen4 5 192 "$L --model Qwen/Qwen3-4B-Base --ref-device cpu --ref-cache $ROOT/$LOGS/ref_qwen3-4b.pt && \
$L --model Qwen/Qwen3-4B-Base --ref-device cpu --ref-cache $ROOT/$LOGS/ref_qwen3-4b.pt --glu-exps 8"
  sub q-qwen4-diag 2 192 "$M.quality.llama_diag --model Qwen/Qwen3-4B-Base --recipes smooth55 scalar4 && \
$M.quality.llama_diag --model Qwen/Qwen3-4B-Base --recipes smooth55 scalar4 --prefix 151643,151643 && \
$L --model Qwen/Qwen3-4B-Base --seq 128 --windows 16 --graphs smooth --glu-exps 8 --ref-device cpu \
--prefix 151643,151643 && $M.quality.llama_attrib --model Qwen/Qwen3-4B-Base --blocks all --recipe smooth55"
  if [ -z "$LLAMA" ]; then
    echo "tier quality: no Llama-2 rows (they need your own Llama-2 licence: --llama-licence)" >&2
    return 0
  fi
  sub q-llama7 6 192 "$L --model meta-llama/Llama-2-7b-hf --graphs main smooth --alphas 0.55 0.65 --structure"
  sub q-llama13 8 256 "$L --model meta-llama/Llama-2-13b-hf --graphs main smooth --ref-dtype float16 --structure"
}

tier_tv() {
  # the token sampler's total-variation distance to the softmax on the certified OPT-1.3B (CPU, 16 workers), and the
  # sampler's closed-form and Monte Carlo checks
  local T="$M.generation.sampler_tv"
  if [ -n "$REH" ]; then
    sub tv-selfcheck 1 32 "$T --selfcheck"
    sub tv-opt125 1 32 "$T --model facebook/opt-125m --recipe legacy --windows 1 --positions 2 --workers 2 --mc 1000 \
--mc-positions 1 --mc-settings 1:0 1:50 --out $ROOT/$LOGS/sampler_tv_opt-125m.json"
    return 0
  fi
  sub tv-opt-1.3b 2 96 "$T --model facebook/opt-1.3b --seq 128 --windows 4 --workers 16 --threads 16 \
--out $ROOT/$LOGS/sampler_tv_opt-1.3b.json"
  sub tv-checks 1 64 "$T --selfcheck --out $ROOT/$LOGS/sampler_tv_selfcheck.json && $T --model facebook/opt-125m \
--recipe legacy --windows 1 --positions 32 --mc 100000 --mc-positions 8 31 --mc-settings 1:0 1:50 1.3:0 \
--out $ROOT/$LOGS/sampler_tv_opt-125m.json"
}

tier_attack() {
  # the attack on the path test: its reproduction, MNIST (single neuron, backdoor, the samplers, the floor sampler,
  # the adaptive game), the path test's bound on Llama-2-7B, OPT-6.7B (an FFN neuron, a backdoor), in-range edits
  # against range checks (OPT-1.3B, OPT-6.7B) and the sampler game on OPT-1.3B; outputs under logs/<name>/attack/
  local o="$ROOT/$LOGS/attack" A="$M.attack"
  if [ -n "$REH" ]; then
    # the stored MNIST networks with few queries (the full cells' documents); OPT-125M for the OPT scripts, whose
    # documents are named by their model and so fill no cell of the paper (experiments/paper/records.DOCUMENTS)
    sub a-bound 1 16 "mkdir -p $o && $A.path_test_bound --out $o/analytic.csv"
    sub a-mnist 1 32 "mkdir -p $o && $A.reproduction --queries 5 --challenges 2 --zero-tolerance-queries 5 \
--out $o/reproduction.json && $A.attack_mnist --queries 20 --challenges 2 --empirical-queries 10 --baseline-queries 2 \
--injection-steps 100 --gradient-steps 100 --stealth-queries 5 --calib-images 500 --out $o/attack.json && \
$A.samplers --queries 5 --blind-spot-queries 5 --candidate-neurons 10 --calib-images 500 --protocol-runs 5 \
--out $o/defence.json && $A.floor_sampler --queries 2 --epsilons 0 1 --calib-images 500 --out $o/floor_sampler.json \
&& $A.adaptive_game --layers 1 --queries 2 --caps 95 inf --double-oracle 1 --calib-size 1000 \
--out $o/adaptive_game_mlp_mnist.json"
    local opt="--model facebook/opt-125m --layers 6 11 --prompts 4"
    sub a-opt125 1 32 "mkdir -p $o && $A.attack_llm $opt --out $o/real_llm_attack_opt-125m.json && \
$A.attack_llm $opt --target auto --out $o/real_llm_attack_opt-125m_auto.json"
    sub a-inrange-opt125 1 32 "mkdir -p $o && $A.inrange_llm $opt --calib-windows 4 \
--out $o/inrange_llm_opt-125m_any.json && $A.inrange_llm $opt --calib-windows 4 --moves down \
--out $o/inrange_llm_opt-125m_down.json"
    sub a-game-opt125 1 32 "mkdir -p $o && $A.adaptive_game_llm --model facebook/opt-125m --layers 6 --prompts 2 \
--calib-windows 8 --threads 8 --device cuda --out $o/adaptive_game_opt-125m.json"
    return 0
  fi
  sub a-mnist 3 32 "mkdir -p $o && $A.reproduction --out $o/reproduction.json && $A.attack_mnist --out $o/attack.json \
&& $A.samplers --out $o/defence.json && $A.floor_sampler --out $o/floor_sampler.json && $A.adaptive_game \
--out $o/adaptive_game_mlp_mnist.json && $A.path_test_bound --out $o/analytic.csv"
  sub a-opt67 2 192 "mkdir -p $o && $A.attack_llm --out $o/real_llm_attack.json && $A.attack_llm --target auto \
--out $o/real_llm_attack_auto.json"
  sub a-inrange-opt13 2 96 "mkdir -p $o && $A.inrange_llm --out $o/inrange_llm_opt-1.3b_any.json && \
$A.inrange_llm --moves down --out $o/inrange_llm_opt-1.3b_down.json"
  sub a-inrange-opt67 4 192 "mkdir -p $o && $A.inrange_llm --model facebook/opt-6.7b --layers 8 16 24 31 \
--threads 16 --out $o/inrange_llm_opt-6.7b_any.json && $A.inrange_llm --model facebook/opt-6.7b \
--layers 8 16 24 31 --threads 16 --moves down --out $o/inrange_llm_opt-6.7b_down.json"
  sub a-game-opt13 1 96 "mkdir -p $o && $A.adaptive_game_llm --threads 8 --device cuda \
--out $o/adaptive_game_opt-1.3b.json"
}

tier_long() {
  # [optional] Llama-2-7B shapes at 2k, 4k, 8k and 16k tokens, 2 and 3 blocks each: the CPU verifier in Kpre and
  # every re-execution baseline on the same threads (a per-block fit t(T) = aT + bT^2); --bigmem: the full model at
  # 8,192 tokens
  if [ -n "$REH" ]; then
    csub l-llama7-4608-L1 1 64 "$PERF --model llama2-7b --layers 1 --seq 4608 --modes Kpre --queries $Q \
--verifier-device cpu --label long $NATIVE"
    csub l-rx-llama7-4608 1 64 "$RX --model llama2-7b --seq 4608 --layers 1 2 --variants int float --device cpu \
--reps 1 --max-pos 8192 --no-malloc-tune $NATIVE"
    return 0
  fi
  local t l
  for t in 2048 4096 8192 16384; do
    for l in 2 3; do
      csub "l-llama7-$t-L$l" 2 192 "$PERF --model llama2-7b --layers $l --seq $t --modes Kpre --queries $Q \
--verifier-device cpu --label long $NATIVE"
    done
    csub "l-rx-llama7-$t" 3 192 "$RX --model llama2-7b --seq $t --layers 2 3 --variants $RXCPU --device cpu \
--reps $Q --max-pos 16384 --no-malloc-tune $NATIVE"
  done
  [ -n "$BIGMEM" ] && csub l-llama7-8192-full 6 256 "$PERF --model llama2-7b --seq 8192 --modes Kpre --queries $Q \
--verifier-device cpu --label long $NATIVE"
  return 0
}

need_images() {   # the bench tier's image cells load trained models and datasets that a checkout does not carry
  [ -n "$DRY" ] && return 0
  local m d miss="" models="lenet5" cache="${PVI_CACHE:-code/artifacts/cache}"
  local imagenet="${PVI_IMAGENET_ROOT:-code/data/imagenet}"
  [ -n "$REH" ] || models="lenet5 vgg11 vgg16 resnet18_cifar resnet18_224 resnet18_224_squirrel"
  for m in $models; do [ -f "code/artifacts/models/cnn/$m.pt" ] || miss="$miss $m.pt"; done
  if [ -z "$REH" ]; then
    [ -d "${PVI_CIFAR_ROOT:-code/data/cifar10}/cifar-10-batches-py" ] || miss="$miss CIFAR-10"
    # each ImageNet subset's cache, or ImageNet with class i of the sorted synsets in folder i + 1 of train/ and val/
    # (dogs 152-269, cats 282-286, the fox squirrel 336)
    for m in dogs_cats:282 dogs_squirrels:336; do
      [ -f "$cache/${m%:*}.pt" ] && continue
      for d in train/152 val/152 "train/${m#*:}" "val/${m#*:}"; do
        [ -d "$imagenet/$d" ] || { miss="$miss ImageNet(${m%:*}: no $imagenet/$d)"; break; }
      done
    done
  fi
  if [ -n "$miss" ]; then
    echo "the bench tier's image cells need:$miss (experiments/package/README.md, Sec. 1); nothing submitted" >&2
    exit 2
  fi
}

preflight() {   # what the jobs need from this host and the scheduler, checked before anything is submitted
  local bad=() t hf="" big gb=60 v d
  [ -z "$LLAMA" ] || gb=100                     # the Llama-2 checkpoints add 40 GB
  for t in "${TIERS[@]}"; do case "$t" in real|quality|tv|attack) hf=1 ;; esac; done
  [ -z "$hf" ] || [ -n "${HF_HOME:-}" ] || bad+=("export HF_HOME=<folder the nodes see, ~$gb \
GB free> (the real, quality, tv and attack tiers download checkpoints and datasets)")
  # a cache variable of Hugging Face's libraries takes precedence over HF_HOME: it must be writable (the nearest
  # existing folder of its path)
  for v in HF_HUB_CACHE HUGGINGFACE_HUB_CACHE TRANSFORMERS_CACHE HF_DATASETS_CACHE; do
    [ -n "${!v:-}" ] || continue
    d="${!v}"
    while [ ! -e "$d" ] && [ "$d" != "$(dirname "$d")" ]; do d="$(dirname "$d")"; done
    [ -d "$d" ] && [ -w "$d" ] || bad+=("$v=${!v} takes precedence over HF_HOME and is not writable here: unset \
$v (or point it to a writable folder the nodes see)")
  done
  case "${HF_HUB_OFFLINE:-}${TRANSFORMERS_OFFLINE:-}" in *1*|*[Tt][Rr][Uu][Ee]*|*[Yy][Ee][Ss]*|*[Oo][Nn]*)
    echo "warning: HF_HUB_OFFLINE or TRANSFORMERS_OFFLINE is set: a job cannot download a checkpoint or dataset \
missing from the cache" >&2 ;;
  esac
  [ -n "$NONATIVE" ] || command -v "${CXX:-}" > /dev/null 2>&1 || bad+=("CXX='${CXX:-}' names no compiler here: \
export CXX=<path to g++> (the native CPU kernels), or pass --no-native-ok")
  # the cpuv tier clones and builds llama.cpp (experiments/reexec/llamacpp.py) with CMake and git
  for t in "${TIERS[@]}"; do
    [ "$t" = cpuv ] || continue
    command -v "${CMAKE:-cmake}" > /dev/null 2>&1 || bad+=("CMake ('${CMAKE:-cmake}') not found here: the cpuv \
tier builds llama.cpp with it (3.14 or later); put cmake on the PATH or export CMAKE=<path> (one way: pip install \
cmake==3.31.6 into a separate environment, e.g. python3 -m venv ~/cmake && ~/cmake/bin/pip install cmake==3.31.6, \
then export CMAKE=~/cmake/bin/cmake)")
    command -v git > /dev/null 2>&1 || bad+=("git not found here: the cpuv tier clones llama.cpp with it")
  done
  [ -x "$PY" ] || bad+=("$PY is not an interpreter (--python <path>)")
  if [ -z "$DRY$LOCAL" ] && command -v sinfo > /dev/null 2>&1; then
    big="$(sinfo -h -N ${SBATCH_PARTITION:+-p "$SBATCH_PARTITION"} -o %m | tr -dc '0-9\n' | sort -n | tail -n 1)"
    [ -z "$big" ] || [ $((MEMNEED * 1024)) -le "$big" ] || bad+=("the largest job asks for $MEMNEED GB and the \
partition's largest node has $((big / 1024)) GB: pass --mem-max $((big / 1024))")
  fi
  if [ -n "$LLAMA" ] && [ -z "${HF_TOKEN:-}" ]; then
    echo "warning: HF_TOKEN is not set; the Llama-2 jobs then need your Hugging Face token in \$HF_HOME/token as \
the compute node sees it (export HF_TOKEN=<token>: sbatch passes it to the jobs)" >&2
  fi
  [ "${#bad[@]}" -gt 0 ] || return 0
  printf 'run.sh: %s\n' "${bad[@]}" >&2
  [ -n "$DRY" ] && return 0
  echo "run.sh: nothing submitted" >&2
  exit 2
}

one_job() {   # --one-job: the tiers as one SLURM job that runs this command with --local in its allocation
  local name jid hours=$PLANH args=() inner=() wrap
  name="sp-node-$(IFS=-; echo "${TIERS[*]}")"
  [ "$hours" -le "$MAXH" ] || hours=$MAXH
  set -- "${ARGV[@]}"
  while [ $# -gt 0 ]; do
    case "$1" in
      --one-job|--dry-run|--no-chain) ;;
      --max-hours) shift ;;
      *) inner+=("$1") ;;
    esac
    shift
  done
  jid="$(awk -F'|' -v n="$name" -v p="$PLATFORM" '$1 == n && $2 == p { print $3; exit }' <<<"$QUEUED")"
  if [ -n "$jid" ]; then
    echo "skip $name: already queued (job $jid)"
    return 0
  fi
  # the allocation's own log is not a job log (logs/<name>/sp-*.out): it lists the jobs it ran
  args=(-o "$LOGS/${name#sp-}-%j.log" --open-mode=append -J "$name" "--comment=$PLATFORM" --dependency=singleton
        --requeue --cpus-per-task=8 "--time=${hours}:00:00")
  if [ -n "$EXCL" ]; then args+=("$EXCL" --mem=0); else args+=("--mem=${MEMNEED}G"); fi
  wrap="cd $(printf %q "$ROOT") && bash code/experiments/package/run.sh --local$(printf ' %q' "${inner[@]}")"
  if [ -n "$DRY" ]; then
    echo "sbatch ${args[*]} --wrap \"$wrap\""
  else
    jid="$(printf '#!/bin/bash\n%s\n' "$wrap" | sbatch --parsable "${args[@]}")"; jid="${jid%%;*}"
    echo "submitted $name: job $jid (its log: $LOGS/${name#sp-}-$jid.log)"
  fi
  echo "# one job: ${TIERS[*]}: $NPLAN jobs, job limits $PLANH h, time limit $hours h"
  [ "$PLANH" -le "$hours" ] || echo "# if it ends at its time limit, submit the same command again: it runs only \
the jobs that have not ended successfully"
}

run_tier() {   # run_tier <tier>: submit it and, in a dry run, print its summary
  local n0=$NJOBS h0=$HOURS
  "tier_$1"
  [ -z "$DRY" ] || [ -n "$PLAN" ] \
    || echo "# tier $1: $((NJOBS - n0)) jobs, job limits $((HOURS - h0)) h, estimate $(est "$1") h"
}

MUST="smoke cpuv v1 serve bench gpuv gen batch real setup improve quality tv attack"
if [ "${TIERS[0]}" = all ]; then TIERS=($MUST ${OPTIONAL:+long}); fi
for tier in "${TIERS[@]}"; do
  case " $MUST long " in *" $tier "*) ;; *) echo "unknown tier $tier" >&2; usage ;; esac
done
for tier in "${TIERS[@]}"; do [ "$tier" = bench ] && need_images; done
# a pass that submits nothing: the largest job's memory, for the pre-flight
PLAN=1
for tier in "${TIERS[@]}"; do run_tier "$tier"; done 2> /dev/null
PLAN=""
preflight
if [ -n "$ONEJOB" ]; then one_job; exit 0; fi
for tier in "${TIERS[@]}"; do
  run_tier "$tier"
  # the platform gate: nothing after smoke starts unless smoke succeeded
  if [ "$tier" = smoke ] && [ "$CHAIN" = 1 ] && [ -n "$LAST" ]; then GATE="afterok:$LAST" DEP=""; fi
  if [ "$tier" = smoke ] && [ -n "$LOCAL" ] && [ -n "$LAST" ] && ! grep -q "SMOKE OK" "$LOGS/sp-smoke-$LAST.out"; then
    echo "smoke did not end in SMOKE OK ($LOGS/sp-smoke-$LAST.out): stopping" >&2
    exit 1
  fi
done
[ "$SKIPPED" = 0 ] || echo "# $SKIPPED jobs already queued for platform $PLATFORM: not submitted again, and the jobs \
after each wait for it"
[ "$DONE" = 0 ] || echo "# $DONE jobs had ended successfully in an earlier run of platform $PLATFORM: not run again \
(to run one again, delete its marker in $LOGS/done/)"
[ -z "$DRY" ] || echo "# ${TIERS[*]}: $NJOBS jobs, job limits $HOURS h"
[ -n "$DRY" ] || [ -n "$LOCAL" ] || squeue -u "$ME"

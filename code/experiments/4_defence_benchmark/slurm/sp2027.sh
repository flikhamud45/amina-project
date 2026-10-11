#!/bin/bash
# The S&P 2027 final run (Plan 2 F6, package v2): one frozen, scripted run on one platform (the L40S prover and GPU
# verifier, 8 threads of the node's CPU as the CPU verifier) whose outputs regenerate the paper's tables
# (experiments/5_comparison/sp2027_numbers.py).  docs/L40S_RUN_SP2027.md is the runner's document.  From the
# repository root of a checkout of the freeze tag:
#     export PVI_PLATFORM=l40s_sp2027                                   # logs/$PVI_PLATFORM/, raw_$PVI_PLATFORM/
#     export SBATCH_PARTITION=<gpu partition> SBATCH_GRES=gpu:<type>:1 [SBATCH_ACCOUNT=...]
#     export HF_HOME=<node-local folder with ~40 GB free>                 # real and quality tiers only
#     export PVI_EXTRA_PYTHONPATH=<folder>   # optional: packages outside the venv (the TAU nodes: transformers, pyarrow)
#     export CXX=<g++>                                                  # the native CPU kernels (B5, RX)
#     bash code/experiments/4_defence_benchmark/slurm/sp2027.sh all     # every must tier, one job after another
# or one tier at a time, in this order (the run order of the plan, Sec. 4):
#     smoke  GPU and native-kernel tests, one V1 query, the host-to-device copy; ends with "SMOKE OK"
#     cpuv   the CPU verifier against every re-execution baseline on the same 8 threads (V-LONG-a, RX, V-V1N)
#     v1     V1 against v0, timed (tab:eval-v1)
#     serve  the prover against unverified serving on the same GPU, with batches (A7)
#     bench  strong_gpu.sh must: the v0 benchmark cells of tab:eval-llm and tab:eval-compare (audited, F6 part 9)
#     gpuv   the GPU verifier, fp16/bf16/int8 re-execution on the same GPU, the host-to-device bandwidth
#     gen    a generated response as one prefill, greedy and sampled, P = 512, G in {64, 256} (chg 12)
#     batch  strong_gpu.sh batch: B prompts against one opening (C4a)
#     real   real-weight OPT end to end next to random-weight twins (F3a; needs HF_HOME)
#     setup  the setup proof of an untrusted commitment, prover and verifier time (D4)
#     quality  [optional] Llama-2-13B quality (needs the runner's own Llama-2 licence: PVI_LLAMA_LICENCE=1)
#     long   [optional] V-LONG-b: Llama-2-7B shapes, 2-3 blocks, 2k-16k tokens, verifier against every baseline
# Switches (environment):
#     PVI_C_POLICY   mode C's commitment plan for every timed mode-C cell (default auto; PS2 fixes it on Wed 21 Oct)
#     PVI_DRYRUN=1   print every sbatch line (and each tier's job count, job-limit sum and estimate); submit nothing
#     PVI_REHEARSAL=1  each tier's smallest cells only (GPT-2 / OPT-125M, 2 blocks, short prompts): the whole package
#                    in about an hour on an RTX 2080 Ti node (F6 part 13), or a platform smoke (part 14)
#     PVI_OPTIONAL=1 'all' also submits the optional tiers (quality, long)
#     PVI_CHAIN=0    do not chain the jobs (default: each job starts after the previous one ends, so no two timed
#                    jobs share the node; in 'all' every tier after smoke waits for smoke to succeed)
#     PVI_EXCLUSIVE=0  no --exclusive (default on: timings need the node to themselves)
#     PVI_MEM_MAX    the largest --mem in GB any job asks for (default 256; set it to the node's RAM)
#     PVI_QUERIES_HEAD / PVI_QUERIES  timed queries after a warm-up for the headline cells (default 5: the v1
#                    Llama-2-7B and OPT-1.3B cells, all of cpuv; serve times 10) and for the others (default 3)
#     PVI_LOCAL=1    a machine without SLURM: every job runs here, in the foreground, one after another (in tmux or
#                    under nohup), its log named as SLURM would name it; 'all' stops if smoke does not end in SMOKE OK
# Every v0 and V1 cell runs under Fiat-Shamir (perf.py/v1_bytes.py/gen_bench.py --fs, serve_ab.py always); the
# interactive cells that remain are bench cells a table compares with an interactive system (zkLLM, Maverick),
# named :int in their cell names.  Every record keeps the peak GPU and host memory of its cell.
# Output: one JSON line per cell in logs/$PVI_PLATFORM/sp-<cell>-<jobid>.out (lines starting with '{'; the cell's
# prefix names its tier: c- cpuv, v1-, serve-, g- gpuv, gen-, r- real, setup-, q- quality, l- long), and bench.py's
# records in code/artifacts/comparison/raw_$PVI_PLATFORM/ (logs $PVI_PLATFORM-<job>-<jobid>.out).  Jobs are named
# sp-<cell> with --dependency=singleton: re-running a tier resubmits only what is not queued (a finished bench cell
# is skipped by bench.py itself).  experiments/5_comparison/sp2027_numbers.py turns both into the paper's tables.
set -euo pipefail
: "${PVI_PLATFORM:?export PVI_PLATFORM=<new name, e.g. l40s_sp2027>}"
[ -f code/experiments/4_defence_benchmark/slurm/sp2027.sh ] || { echo "run this from the repository root" >&2; exit 2; }
ROOT="$PWD"
LOGS="logs/$PVI_PLATFORM"
DRY="${PVI_DRYRUN:-}"
REH="${PVI_REHEARSAL:-}"
LOCAL="${PVI_LOCAL:-}"
ME="${USER:-$(id -un)}"
if [ -n "$DRY" ] || [ -n "$LOCAL" ]; then QUEUED=""; else QUEUED="$(squeue -h -u "$ME" -o %j)"; fi
[ -n "$DRY" ] || mkdir -p "$LOGS"
PY="${PVI_PYTHON:-$ROOT/.venv/bin/python}"
POL="${PVI_C_POLICY:-auto}"
QH="${PVI_QUERIES_HEAD:-5}"
Q="${PVI_QUERIES:-3}"
if [ -n "$REH" ]; then QH=2; Q=2; fi
MEMMAX="${PVI_MEM_MAX:-256}"
CHAIN="${PVI_CHAIN:-1}"
EXCL=""; [ "${PVI_EXCLUSIVE:-1}" = 1 ] && EXCL="--exclusive"
DEP="${PVI_DEP:-}"           # the next job's dependency (afterok:<id> / afterany:<id>); empty: none but singleton
# the jobs' environment; TRITON_CACHE_DIR, the native kernels' folder and the allocator setting are exported too, so
# strong_gpu.sh's bench.sbatch jobs (which inherit the submitting shell's environment) get them as well
export TRITON_CACHE_DIR="${TRITON_CACHE_DIR:-$ROOT/$LOGS/.triton}" PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENVS="cd $ROOT/code && export PYTHONPATH=$ROOT/code/src${PVI_EXTRA_PYTHONPATH:+:$PVI_EXTRA_PYTHONPATH} OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NVIDIA_TF32_OVERRIDE=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$TRITON_CACHE_DIR"
# ${PVI_PY_INCLUDE:+...}: hosts without the Python headers need a copy for Triton's launcher (docs/improvements/I_server_setup.md)
[ -n "${PVI_PY_INCLUDE:-}" ] && ENVS="$ENVS PVI_PY_INCLUDE=$PVI_PY_INCLUDE"
if [ -n "${CXX:-}" ]; then
  export PVI_NATIVE_DIR="${PVI_NATIVE_DIR:-$ROOT/$LOGS/.native}"
  ENVS="$ENVS CXX=$CXX PVI_NATIVE_DIR=$PVI_NATIVE_DIR"
fi
NJOBS=0; HOURS=0; DRYN=0; LAST=""

sub() {   # sub <name> <hours> <memGB> <command...>: one job, chained after the previous one (PVI_CHAIN)
  local name="sp-$1" hours=$2 mem=$3 jid dep="singleton"; shift 3
  if grep -qxF -- "$name" <<<"$QUEUED"; then echo "skip $name: already queued"; return 0; fi
  [ "$mem" -gt "$MEMMAX" ] && mem=$MEMMAX
  [ -n "$DEP" ] && dep="$DEP,singleton"
  local args=(-o "$LOGS/%x-%j.out" -J "$name" "--dependency=$dep" --cpus-per-task=8 "--mem=${mem}G"
              "--time=${hours}:00:00")
  [ -n "$EXCL" ] && args+=("$EXCL")
  [ -n "${SBATCH_GRES:-}" ] && args+=("--gres=$SBATCH_GRES")
  local wrap="set -uo pipefail; $ENVS; echo host \$(hostname) git \$(git -C $ROOT log --oneline -1); $*"
  if [ -n "$DRY" ]; then
    DRYN=$((DRYN + 1)); jid="DRY$DRYN"
    echo "sbatch ${args[*]} --wrap \"$wrap\""
  elif [ -n "$LOCAL" ]; then              # no SLURM: run it now, within its time limit, logged as SLURM would
    jid="$(date +%s)"
    echo "$(date '+%F %T') running $name (limit ${hours} h): $LOGS/$name-$jid.out"
    timeout "${hours}h" bash -c "$wrap" > "$LOGS/$name-$jid.out" 2>&1 || echo "$name: exit status $?" >&2
  else
    # the job script goes in on stdin with a bash shebang: --wrap runs /bin/sh, which is dash on some clusters (no
    # pipefail), and the commands use bash
    jid="$(printf '#!/bin/bash
%s
' "$wrap" | sbatch --parsable "${args[@]}")"; jid="${jid%%;*}"
    echo "submitted $name: job $jid"
  fi
  QUEUED+=$'\n'"$name"
  NJOBS=$((NJOBS + 1)); HOURS=$((HOURS + hours)); LAST="$jid"
  [ "$CHAIN" = 1 ] && DEP="afterany:$jid"
  return 0
}

nl() {    # the model's decoder blocks (reexec.py measures the full model directly: --layers 2 <nl>)
  case "$1" in
    gpt2|opt-125m) echo 12 ;; opt-1.3b) echo 24 ;; opt-6.7b|llama2-7b) echo 32 ;; qwen3-4b) echo 36 ;;
    llama2-13b) echo 40 ;; *) echo "nl: unknown model $1" >&2; exit 2 ;;
  esac
}

est() {   # the tier's expected hours (estimates from measured times: docs/improvements/F6_final_run_package.md)
  if [ -n "$REH" ]; then echo "<0.3"; return; fi
  case "$1" in
    smoke) echo 1 ;; cpuv) echo 6-9 ;; v1) echo 4-8 ;; serve) echo 8-10 ;; bench) echo 8-12 ;; gpuv) echo 4-6 ;;
    gen) echo 2-3 ;; batch) echo 2-3 ;; real) echo 4-6 ;; setup) echo 1-2 ;; quality) echo 2-3 ;; long) echo 4-7 ;;
  esac
}

V1="$PY experiments/8_v1/v1_bytes.py --fs --lean --full --prune-last --lmax 27 --threads 8"
PERF="$PY experiments/6_improvements/perf.py --fs --lean --wire on --prune-last on --lookups on --threads 8"
RX="$PY experiments/7_reexec/reexec.py --threads 8"
RXCPU="int int_fp32gemm float bf16 dynint8"   # RX: exact int8-GEMM integer, fp32-GEMM integer, fp32, bf16; dynint8 not exact
GEN="$PY experiments/6_improvements/gen_bench.py --fs --threads 8"

tier_smoke() {
  # the new code paths on this platform: Triton (GKR, GPU decoder, fused attention) and torch._int_mm on this GPU,
  # the native -march=native kernels on this CPU (CXX), Fiat-Shamir's tree hashing, the serving and F6 harnesses,
  # sampling and generation, the setup proof; then one V1 query, a native-kernel and an int8-GEMM self-check and
  # the host-to-device copy.  "SMOKE OK" at the end of the log, or the other tiers must not start.
  local t="tests/test_fs_tree.py tests/test_serve_ab.py tests/test_f6_package.py"
  [ -n "$REH" ] || t="tests/test_gkr_triton.py tests/test_gkr_triton_cpu.py tests/test_gpu_verifier.py \
tests/test_gpu_exactness.py tests/test_cut_protocol.py tests/test_token_sampler.py tests/test_extfield.py \
tests/test_logup.py tests/test_cut.py tests/test_native_cheap.py tests/test_v1_buckets.py tests/test_generation.py \
tests/test_setup_proof.py $t"
  local native=""
  [ -n "${CXX:-}" ] && native="$PY -c 'import sys; from pvi.fullcheck import native_kernels as n; \
sys.exit(0 if n.available() else \"native kernels unavailable with CXX=$CXX\")' && "
  sub smoke 2 64 "PVI_TEST_DEVICE=cuda $PY -m pytest -o addopts= -q -p no:cacheprovider $t 2>&1 | tail -3 && \
${native}$PY -c 'from pvi.fullcheck import field; print(\"int8 GEMMs exact on this GPU:\", field.int8_ok(\"cuda\"))' && \
$V1 --model gpt2 --layers 2 --seq 64 --modes C Kpre --policy $POL --reps 2 --lmax 20 | grep '^{' && \
$PY experiments/6_improvements/h2d_bandwidth.py --sizes-mb 256 --reps 3 --label smoke && echo SMOKE OK"
}

tier_cpuv() {
  # V-LONG-a, RX and V-V1N on this CPU (8 threads, the same for every row): the native CPU verifier (B5) of the
  # full models in Kpre and C under Fiat-Shamir (the prover on the GPU), every re-execution baseline of RX on the
  # same threads (the full model measured directly, and 2 blocks), the allocator setting the verifier applies to
  # itself (native_kernels.tune_malloc) for a like-for-like re-execution row, and V1's old and new CPU verifier paths
  # (PVI_V1_BUCKETS 0 and 1) on the same queries.  The strongest exact baseline is the fastest of int, int_fp32gemm,
  # float and bf16 (dynint8 computes another model: reported, not counted).  8,192 tokens: tier long.
  [ -n "${CXX:-}" ] || echo "warning: CXX is not set; the CPU verifier and the int variant fall back to torch" >&2
  if [ -n "$REH" ]; then
    sub c-gpt2-64 1 64 "$PERF --model gpt2 --layers 2 --seq 64 --modes Kpre C --policy $POL --queries $QH \
--verifier-device cpu --label cpuv"
    sub c-rx-gpt2 1 64 "$RX --model gpt2 --seq 64 --layers 1 2 --variants $RXCPU --device cpu --reps 2"
    sub c-v1-gpt2-64 1 64 "$V1 --model gpt2 --layers 2 --seq 64 --modes Kpre --reps $QH --lmax 20 --v1-paths 0 1 \
--label cpuv-v1"
    return 0
  fi
  local m t
  for m in llama2-7b qwen3-4b opt-1.3b; do
    for t in 64 2048; do
      if [ "$t" = 64 ]; then
        sub "c-$m-$t" 1 128 "$PERF --model $m --seq $t --modes Kpre C --policy $POL --queries $QH --verifier-device cpu \
--label cpuv"
      else
        sub "c-$m-$t" 2 256 "$PERF --model $m --seq $t --modes Kpre C --policy $POL --queries $QH --verifier-device cpu \
--label cpuv"
      fi
    done
  done
  sub c-rx-llama2-7b 3 256 "$RX --model llama2-7b --seq 64 2048 --layers 2 32 --variants $RXCPU --device cpu --reps $QH"
  sub c-rx-qwen3-4b 2 256 "$RX --model qwen3-4b --seq 64 2048 --layers 2 36 --variants $RXCPU --device cpu --reps $QH"
  sub c-rx-opt-1.3b 1 128 "$RX --model opt-1.3b --seq 64 2048 --layers 2 24 --variants $RXCPU --device cpu --reps $QH"
  sub c-rx-heap-llama2-7b 2 256 "MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=2147483647 MALLOC_TOP_PAD_=268435456 \
$RX --model llama2-7b --seq 64 2048 --layers 2 32 --variants int float dynint8 --device cpu --reps $QH"
  sub c-v1-llama2-7b-64 1 128 "$V1 --model llama2-7b --seq 64 --modes Kpre --reps $QH --v1-paths 0 1 --label cpuv-v1"
  sub c-v1-llama2-7b-2048 2 256 "$V1 --model llama2-7b --seq 2048 --modes Kpre --reps $QH --v1-paths 0 1 --label cpuv-v1"
  sub c-v1-qwen3-4b-2048 2 256 "$V1 --model qwen3-4b --seq 2048 --modes Kpre --reps $QH --v1-paths 0 1 --label cpuv-v1"
}

tier_v1() {
  # V1 against v0 on the same queries: bytes (exact, with the T7 accounting), prover and verifier times (the CPU
  # verifier on 8 threads, the GKR on the GPU), lmax 27, Fiat-Shamir, mode C under PVI_C_POLICY.  QH timed queries
  # for the Llama-2-7B and OPT-1.3B cells, Q for the others.
  if [ -n "$REH" ]; then
    sub v1-gpt2-64 1 64 "$V1 --model gpt2 --layers 2 --seq 64 --modes C Kpre --policy $POL --reps $Q --lmax 20"
    return 0
  fi
  local t
  sub v1-gpt2-64 1 64 "$V1 --model gpt2 --seq 64 --modes C Kpre --policy $POL --reps $Q"
  sub v1-gpt2-512 1 64 "$V1 --model gpt2 --seq 512 --modes C Kpre --policy $POL --reps $Q"
  for t in 64 2048; do
    sub "v1-opt125-$t" 1 96 "$V1 --model opt-125m --seq $t --modes C Kpre --policy $POL --reps $Q"
    sub "v1-opt13-$t" 2 192 "$V1 --model opt-1.3b --seq $t --modes C Kpre --policy $POL --reps $QH"
    sub "v1-llama7-$t" 3 256 "$V1 --model llama2-7b --seq $t --modes Kpre C --policy $POL --reps $QH"
    sub "v1-opt67-$t" 3 256 "$V1 --model opt-6.7b --seq $t --modes Kpre C --policy $POL --reps $Q"
  done
  sub v1-llama13-64 3 256 "$V1 --model llama2-13b --seq 64 --modes Kpre C --policy $POL --reps $Q"
  sub v1-llama13-2k-2 1 256 "$V1 --model llama2-13b --layers 2 --seq 2048 --modes Kpre C --policy $POL --reps $Q"
  sub v1-qwen4-64 2 192 "$V1 --model qwen3-4b --seq 64 --modes Kpre C --policy $POL --reps $Q"
}

tier_serve() {
  # A7: the prover against unverified serving on this GPU, on the same token ids and shapes (serve_ab.py): U1, our
  # integer model with every claim dropped; U2, eager fp16 and bf16 decoders with SDPA (one batched prefill); the
  # prover in modes C and Kpre under Fiat-Shamir (wire, lean, pruned; Kpre with lookups).  Workloads (a) T = 64, 512,
  # 2,048 at B = 1 and (b) T = 64 with B = 1, 4, 16, 64 and T = 2,048 with B = 1, 4; a batch proved as B queries one
  # after another (seq) and as one statement (joint).  10 repetitions after a warm-up, CUDA events; every cell checks
  # U1's logits against the proof's, a byte-identical second proof and acceptance.  A variant that does not fit the
  # GPU is recorded as skipped (e.g. joint at T = 2,048, B = 4 on the 7B+ models).  Mode C's plan: PVI_C_POLICY.
  # Random weights: no downloads.  Host memory: Qwen3-4B's joint proof of 4 x 2,048 tokens fits the GPU (about
  # 36 GB of int32 claims), and its CPU acceptance check then holds the decoded claims and every op's input (about
  # 120 GB in int64), hence 256 GB for it too.
  local serve="$PY experiments/7_reexec/serve_ab.py --workload a b --modes C Kpre --batch-proof seq joint --reps 10 \
--device cuda --threads 8 --u2-dtype fp16 bf16 --policy $POL --label serve"
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

tier_strong() {   # tier_strong <strong_gpu.sh tier>: its bench.sbatch jobs, in this script's chain
  local f; f="$(mktemp)"
  PVI_DEP="$DEP" PVI_CHAIN="$CHAIN" PVI_LAST_FILE="$f" PVI_C_POLICY="$POL" PVI_MEM_MAX="$MEMMAX" \
    PVI_EXCLUSIVE="${PVI_EXCLUSIVE:-1}" PVI_DRYN0="$DRYN" \
    bash code/experiments/4_defence_benchmark/slurm/strong_gpu.sh "$1"
  local sg_dep="" sg_jobs=0 sg_hours=0 sg_dryn="$DRYN"
  # shellcheck disable=SC1090
  . "$f"
  rm -f "$f"
  NJOBS=$((NJOBS + sg_jobs)); HOURS=$((HOURS + sg_hours)); DRYN="$sg_dryn"
  [ -n "$sg_dep" ] && DEP="$sg_dep"
  return 0
}

tier_gpuv() {
  # the verifier on the GPU and re-execution on the same GPU: v0's streaming GPU verifier (perf.py, one process per
  # mode: the prover and the client share the card) and V1 with the GPU verifier at 2,048 tokens; the integer forward
  # pass with int8 GEMMs (torch._int_mm), fp16 and bf16 decoders of every table model at 64, 512 and 2,048 tokens
  # (2 blocks and the full model); the host-to-device copy bandwidth (pinned and pageable).  Every line records peak
  # GPU and host memory (the verifier against re-execution, C-Q4).
  if [ -n "$REH" ]; then
    sub g-v0-gpt2 1 64 "$PERF --model gpt2 --layers 2 --seq 64 --modes Kpre C --policy $POL --queries $Q \
--verifier-device cuda --extra '{\"stream\": true}' --label gpuv"
    sub g-rx-gpt2 1 64 "$RX --model gpt2 --seq 64 --layers 1 2 --variants int float bf16 --device cuda --reps 2"
    sub g-h2d 1 32 "$PY experiments/6_improvements/h2d_bandwidth.py --sizes-mb 64 256 --reps 3 --label gpuv"
    return 0
  fi
  local m mode
  for m in opt-1.3b llama2-7b; do
    for mode in Kpre C; do
      sub "g-v0-$m-$mode" 1 192 "$PERF --model $m --seq 2048 --modes $mode --policy $POL --queries $Q \
--verifier-device cuda --extra '{\"stream\": true}' --label gpuv"
    done
    sub "g-v1-$m" 2 192 "$V1 --model $m --seq 2048 --modes Kpre --reps $Q --verifier-device cuda --label gpuv-v1"
  done
  for m in opt-1.3b qwen3-4b llama2-7b opt-6.7b llama2-13b; do
    sub "g-rx-$m" 2 128 "$RX --model $m --seq 64 512 2048 --layers 2 $(nl "$m") --variants int float bf16 --device cuda \
--reps $QH"
  done
  sub g-h2d 1 32 "$PY experiments/6_improvements/h2d_bandwidth.py --sizes-mb 64 1024 4096 --reps 10 --label gpuv"
}

tier_gen() {
  # chg 12: a response of G tokens after a P-token prompt proved as one prefill (E1), greedy and sampled with the
  # client's seed (E2), modes C and Kpre under Fiat-Shamir, Q timed responses per cell; each cell also runs the two
  # tamper cells (a wrong token: rejected at "token"; logits forged to choose it: rejected by the LM head's check).
  if [ -n "$REH" ]; then
    sub gen-gpt2 1 64 "$GEN --model gpt2 --layers 2 --prompt 16 --gen 4 --decode greedy sample --modes C Kpre \
--policy $POL --queries $Q --label gen"
    return 0
  fi
  sub gen-llama2-7b 3 192 "$GEN --model llama2-7b --prompt 512 --gen 64 256 --decode greedy sample --modes C Kpre \
--policy $POL --queries $Q --label gen"
  sub gen-qwen3-4b 3 192 "$GEN --model qwen3-4b --prompt 512 --gen 64 256 --decode greedy sample --modes C Kpre \
--policy $POL --queries $Q --label gen"
}

tier_real() {
  # F3a: real-weight OPT end to end on the final platform, each cell next to its random-weight twin (needs HF_HOME:
  # facebook/opt-1.3b and facebook/opt-6.7b, the WikiText-2 test parquet).  One graph per model: the smoothed integer
  # OPT (F2) calibrated on one 2,048-token window, last block pruned; Fiat-Shamir, Q queries after a warm-up, 3 of
  # each tamper per variant.  The 2080 Ti / Xeon cells of F3a (OPT-1.3B v0 on both verifiers, OPT-6.7B's codec
  # profile and V1 bytes) are measured before the freeze; these are the cells that do not fit there and the
  # final-platform timings.  F3b (Qwen3-4B real weights) and a real Llama-2-7B in mode C join only once perf.py and
  # v1_bytes.py read Llama/Qwen checkpoints (--real reads OPT checkpoints today).
  : "${HF_HOME:?export HF_HOME=<node-local folder with ~40 GB free> (the real tier downloads OPT checkpoints)}"
  local r13="--model opt-1.3b --real facebook/opt-1.3b --smooth 0.55 --calib-seq 2048"
  local r67="--model opt-6.7b --real facebook/opt-6.7b --smooth 0.55 --calib-seq 2048"
  if [ -n "$REH" ]; then
    local r125="--model opt-125m --real facebook/opt-125m --smooth 0.55 --calib-seq 128"
    sub r-opt125-v0 1 64 "HF_HOME=$HF_HOME $PERF $r125 --seq 64 --modes Kpre C --policy $POL --queries $Q \
--verifier-device cpu --tampers 1 --label real"
    sub r-opt125-v1 1 64 "HF_HOME=$HF_HOME $V1 $r125 --seq 64 --modes Kpre --reps $Q --tampers 1 --lmax 20 --label real"
    return 0
  fi
  local t mode
  for t in 64 2048; do
    sub "r-opt67-v0-$t" 2 192 "HF_HOME=$HF_HOME $PERF $r67 --seq $t --modes Kpre --queries $Q --verifier-device cpu \
--tampers 3 --label real"
    sub "r-opt67rnd-v0-$t" 1 192 "$PERF --model opt-6.7b --seq $t --modes Kpre --queries $Q --verifier-device cpu \
--tampers 3 --label real-random"
    sub "r-opt67-v1-$t" 2 256 "HF_HOME=$HF_HOME $V1 $r67 --seq $t --modes Kpre --reps $Q --tampers 3 --label real"
    sub "r-opt67rnd-v1-$t" 2 256 "$V1 --model opt-6.7b --seq $t --modes Kpre --reps $Q --tampers 3 --label real-random"
    for mode in C Kpre; do
      sub "r-opt13-v1-$mode-$t" 1 192 "HF_HOME=$HF_HOME $V1 $r13 --seq $t --modes $mode --policy $POL --reps $Q \
--tampers 3 --label real"
      sub "r-opt13rnd-v1-$mode-$t" 1 192 "$V1 --model opt-1.3b --seq $t --modes $mode --policy $POL --reps $Q \
--tampers 3 --label real-random"
    done
  done
  echo "note: tier real has no F3b (Qwen3-4B) or real Llama-2-7B cells: perf.py/v1_bytes.py --real read OPT checkpoints" >&2
}

tier_setup() {
  # D4's remainder: the setup proof of an untrusted commitment for Llama-2-7B with radii for a = 100 amortised
  # queries (the cell the paper's text quotes; 2.8 GB), prover on the GPU and verifier on 8 CPU threads, under
  # PVI_C_POLICY; GPT-2 at a = 100 and 1,000 (minutes), which the extended version's table also lists.
  if [ -n "$REH" ]; then
    sub setup-gpt2 1 64 "$PY experiments/6_improvements/setup_timing.py --model gpt2 --layers 2 --policy $POL \
--amortise 100 --reps 1 --threads 8 --label setup"
    return 0
  fi
  sub setup-llama2-7b 3 128 "$PY experiments/6_improvements/setup_timing.py --model llama2-7b --policy $POL \
--amortise 100 --reps 1 --threads 8 --label setup"
  sub setup-gpt2 1 64 "$PY experiments/6_improvements/setup_timing.py --model gpt2 --policy $POL --amortise 100 1000 \
--reps 3 --threads 8 --label setup"
}

tier_quality() {
  # [optional] F2b-L's Llama-2-13B on the full WikiText-2 test set at 2,048 tokens (scalar gains and smoothing, the
  # fp16 reference on the GPU, calibration on the CPU from train-split windows): only with the runner's own
  # Llama-2 licence (meta-llama/Llama-2-13b-hf), never another person's token.  OPT, Qwen3 and Llama-2-7B quality
  # run on the 2080 Ti before the freeze (F2a, F2b).  The rehearsal runs ppl_llama.py's tiny random model with
  # Qwen3's (ungated) vocabulary instead.
  : "${HF_HOME:?export HF_HOME=<node-local folder with ~40 GB free> (the quality tier downloads a checkpoint)}"
  if [ -n "$REH" ]; then
    sub q-smoke 1 64 "HF_HOME=$HF_HOME $PY experiments/9_quality/ppl_llama.py --model Qwen/Qwen3-0.6B-Base --smoke \
--seq 128 --windows 2 --calib-windows 1 --device cuda --threads 8 --chunk 64"
    return 0
  fi
  if [ -z "${PVI_LLAMA_LICENCE:-}" ]; then
    echo "tier quality: skipped (Llama-2-13B needs the runner's own Llama-2 licence: export PVI_LLAMA_LICENCE=1)" >&2
    return 0
  fi
  sub q-llama13 8 256 "HF_HOME=$HF_HOME $PY experiments/9_quality/ppl_llama.py --model meta-llama/Llama-2-13b-hf \
--config llama2-13b --seq 2048 --windows 0 --calib-windows 4 --graphs scalar smooth --device cuda --calib-device cpu \
--ref-dtype float16 --threads 8 ${PVI_Q13_ARGS:-}"
}

tier_long() {
  # [optional] V-LONG-b: Llama-2-7B shapes at 2k, 4k, 8k and 16k tokens, 2 and 3 blocks each (measurement only; the
  # kernels hold below 32,768 tokens): the CPU verifier in Kpre (exact bytes) and every RX baseline on the same
  # threads, for a per-block fit t(T) = aT + bT^2.  PVI_BIGMEM=1 (a node with >= 256 GB): the full model at 8,192.
  if [ -n "$REH" ]; then
    sub l-llama7-4608-L1 1 64 "$PERF --model llama2-7b --layers 1 --seq 4608 --modes Kpre --queries $Q \
--verifier-device cpu --label long"
    sub l-rx-llama7-4608 1 64 "$RX --model llama2-7b --seq 4608 --layers 1 2 --variants int float --device cpu --reps 1 \
--max-pos 8192"
    return 0
  fi
  local t l
  for t in 2048 4096 8192 16384; do
    for l in 2 3; do
      sub "l-llama7-$t-L$l" 2 192 "$PERF --model llama2-7b --layers $l --seq $t --modes Kpre --queries $Q \
--verifier-device cpu --label long"
    done
    sub "l-rx-llama7-$t" 3 192 "$RX --model llama2-7b --seq $t --layers 2 3 --variants $RXCPU --device cpu --reps $Q \
--max-pos 16384"
  done
  [ -n "${PVI_BIGMEM:-}" ] && sub l-llama7-8192-full 6 256 "$PERF --model llama2-7b --seq 8192 --modes Kpre \
--queries $Q --verifier-device cpu --label long"
  return 0
}

run_tier() {   # run_tier <tier>: submit it and, in a dry run, print its summary
  local n0=$NJOBS h0=$HOURS
  case "$1" in
    smoke) tier_smoke ;; cpuv) tier_cpuv ;; v1) tier_v1 ;; serve) tier_serve ;; bench) tier_strong must ;;
    gpuv) tier_gpuv ;; gen) tier_gen ;; batch) tier_strong batch ;; real) tier_real ;; setup) tier_setup ;;
    quality) tier_quality ;; long) tier_long ;;
    *) echo "usage: sp2027.sh all|smoke|cpuv|v1|serve|bench|gpuv|gen|batch|real|setup|quality|long" >&2; exit 2 ;;
  esac
  [ -z "$DRY" ] || echo "# tier $1: $((NJOBS - n0)) jobs, job limits $((HOURS - h0)) h, estimate $(est "$1") h"
}

MUST="smoke cpuv v1 serve bench gpuv gen batch real setup"
case "${1:-}" in
all)
  for tier in $MUST ${PVI_OPTIONAL:+quality long}; do
    run_tier "$tier"
    # the platform gate: nothing after smoke starts unless smoke succeeded
    [ "$tier" = smoke ] && [ "$CHAIN" = 1 ] && [ -n "$LAST" ] && DEP="afterok:$LAST"
    if [ "$tier" = smoke ] && [ -n "$LOCAL" ] && [ -n "$LAST" ] && ! grep -q "SMOKE OK" "$LOGS/sp-smoke-$LAST.out"; then
      echo "smoke did not end in SMOKE OK ($LOGS/sp-smoke-$LAST.out): stopping (docs/L40S_RUN_SP2027.md, hotfix rule)" >&2
      exit 1
    fi
  done
  [ -z "$DRY" ] || echo "# all: $NJOBS jobs, job limits $HOURS h (must tiers: $MUST${PVI_OPTIONAL:+; optional: quality long})"
  ;;
*)
  run_tier "${1:-}"
  ;;
esac
[ -n "$DRY" ] || [ -n "$LOCAL" ] || squeue -u "$ME"

#!/bin/bash
# The S&P 2027 runs on a strong GPU (the friend's L40S), next to strong_gpu.sh's benchmark tiers.  From the repository
# root of an sp2027 checkout:
#     export PVI_PLATFORM=<new name, e.g. l40s_sp2027>
#     export SBATCH_PARTITION=<gpu partition> SBATCH_GRES=gpu:<type>:1 [SBATCH_ACCOUNT=...]
#     export HF_HOME=<a folder with ~40 GB free that the compute nodes can read>     # quality and real-weight tiers
#     bash code/experiments/4_defence_benchmark/slurm/sp2027.sh smoke    # once; wait for "SMOKE OK" in its log
#     bash code/experiments/4_defence_benchmark/slurm/sp2027.sh v1       # V1 against v0, timed (the headline table)
#     bash code/experiments/4_defence_benchmark/slurm/sp2027.sh quality  # F2 perplexity and V1 on real weights
#     bash code/experiments/4_defence_benchmark/slurm/sp2027.sh gpuv     # the verifier on the GPU, v0 and V1
#     bash code/experiments/4_defence_benchmark/slurm/sp2027.sh cpuv     # the CPU verifier against re-execution (needs CXX)
#     bash code/experiments/4_defence_benchmark/slurm/sp2027.sh attack   # in-range edits on OPT-6.7B (G3.5; needs HF_HOME)
# Then strong_gpu.sh's must/should tiers with the same PVI_PLATFORM give the v0 benchmark (Table 3 and the
# comparisons) with every prover/verifier improvement of sp2027 on.
# Output: one JSON line per cell in logs/$PVI_PLATFORM/sp-<tier>-<cell>-<jobid>.out (grep '^{').  PVI_DRYRUN=1 prints
# the sbatch lines.  Jobs are named sp-<tier>-<cell> with --dependency=singleton: re-running a tier resubmits only
# what is not queued.
set -euo pipefail
: "${PVI_PLATFORM:?export PVI_PLATFORM=<new name>}"
[ -f code/experiments/4_defence_benchmark/slurm/sp2027.sh ] || { echo "run this from the repository root" >&2; exit 2; }
ROOT="$PWD"
LOGS="logs/$PVI_PLATFORM"
mkdir -p "$LOGS"
ME="${USER:-$(id -un)}"
if [ -n "${PVI_DRYRUN:-}" ]; then SB="echo sbatch"; QUEUED=""; else SB=sbatch; QUEUED="$(squeue -h -u "$ME" -o %j)"; fi
PY="${PVI_PYTHON:-$ROOT/.venv/bin/python}"
ENVS="cd $ROOT/code && export PYTHONPATH=$ROOT/code/src OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 NVIDIA_TF32_OVERRIDE=0 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True TRITON_CACHE_DIR=$ROOT/logs/$PVI_PLATFORM/.triton"
# ${PVI_PY_INCLUDE:+...}: hosts without the Python headers need a copy for Triton's launcher (docs/improvements/I_server_setup.md)
[ -n "${PVI_PY_INCLUDE:-}" ] && ENVS="$ENVS PVI_PY_INCLUDE=$PVI_PY_INCLUDE"
[ -n "${CXX:-}" ] && ENVS="$ENVS CXX=$CXX PVI_NATIVE_DIR=$ROOT/logs/$PVI_PLATFORM/.native"

sub() {   # sub <name> <hours> <memGB> <command...>
  local name="sp-$1" hours=$2 mem=$3; shift 3
  if grep -qxF -- "$name" <<<"$QUEUED"; then echo "skip $name: already queued"; return 0; fi
  $SB -o "$LOGS/%x-%j.out" -J "$name" --dependency=singleton --cpus-per-task=8 --mem="${mem}G" --time="${hours}:00:00" \
      ${SBATCH_GRES:+--gres=$SBATCH_GRES} --wrap "set -uo pipefail; $ENVS; echo host \$(hostname) git \$(git -C $ROOT log --oneline -1); $*"
  QUEUED+=$'\n'"$name"
}
V1="$PY experiments/8_v1/v1_bytes.py --fs --lean --full --reps 3"

case "${1:-}" in
smoke)
  # the new GPU code paths: Triton GKR (byte-identical transcripts), GPU decoder and fused attention, V1 on a GPU
  # prover and verifier, sampling; then one small V1 query end to end
  sub smoke 2 64 "$PY -m pytest -o addopts= -q -p no:cacheprovider tests/test_gkr_triton.py tests/test_gkr_triton_cpu.py \
tests/test_gpu_verifier.py tests/test_cut_protocol.py tests/test_token_sampler.py tests/test_extfield.py tests/test_logup.py \
tests/test_cut.py 2>&1 | tail -3 && $V1 --model gpt2 --seq 64 --modes C Kpre --policy auto --prune-last --lmax 27 \
| grep '^{' && echo SMOKE OK"
  ;;
v1)
  # V1 against v0 on the same queries: bytes (exact), prover and verifier timings (3 queries after a warm-up), lmax 27.
  # The CPU verifier uses 8 threads (the paper's verifier); the GKR runs on the GPU.
  sub v1-gpt2-64  2 64  "$V1 --model gpt2 --seq 64 --modes C Kpre --policy auto --prune-last --lmax 27"
  sub v1-gpt2-512 2 64  "$V1 --model gpt2 --seq 512 --modes C Kpre --policy auto --prune-last --lmax 27"
  for T in 64 2048; do
    sub "v1-opt125-$T"  2 96  "$V1 --model opt-125m  --seq $T --modes C Kpre --policy auto --prune-last --lmax 27"
    sub "v1-opt13-$T"   4 192 "$V1 --model opt-1.3b  --seq $T --modes C Kpre --policy auto --prune-last --lmax 27"
    sub "v1-llama7-$T"  6 256 "$V1 --model llama2-7b --seq $T --modes Kpre C --policy auto --prune-last --lmax 27"
    sub "v1-opt67-$T"   6 256 "$V1 --model opt-6.7b  --seq $T --modes Kpre C --policy auto --prune-last --lmax 27"
  done
  sub v1-llama13-64   6 256 "$V1 --model llama2-13b --seq 64 --modes Kpre C --policy auto --prune-last --lmax 27"
  sub v1-llama13-2k-2 6 256 "$V1 --model llama2-13b --layers 2 --seq 2048 --modes Kpre C --policy auto --prune-last --lmax 27"
  sub v1-qwen4-64     4 192 "$V1 --model qwen3-4b --seq 64 --modes Kpre C --policy auto --prune-last --lmax 27"
  ;;
quality)
  # F2: fp32, scalar norm gains and SmoothQuant through the gains, WikiText-2 16 x 128 (needs HF_HOME; downloads
  # facebook/opt-* and the WikiText-2 test parquet); then V1's bytes on the smoothed real-weight graphs
  sub q-opt125 2 64  "HF_HOME=$HF_HOME $PY experiments/9_quality/ppl_smooth.py --model facebook/opt-125m --gains 4 --alphas 0.55 0.5 --device cuda"
  sub q-opt13  3 96  "HF_HOME=$HF_HOME $PY experiments/9_quality/ppl_smooth.py --model facebook/opt-1.3b --gains 3 --alphas 0.55 0.5 --device cuda"
  sub q-opt67  6 192 "HF_HOME=$HF_HOME $PY experiments/9_quality/ppl_smooth.py --model facebook/opt-6.7b --gains 2 3 --alphas 0.55 0.5 0.65 --device cuda"
  sub q-opt67-2k 6 192 "HF_HOME=$HF_HOME $PY experiments/9_quality/ppl_smooth.py --model facebook/opt-6.7b --seq 2048 --windows 4 --gains 2 --alphas 0.55 --device cuda"
  for M in opt-125m opt-1.3b opt-6.7b; do
    sub "q-v1-$M" 6 256 "HF_HOME=$HF_HOME $PY experiments/8_v1/v1_bytes.py --model $M --real facebook/$M --smooth 0.55 \
--seq 2048 --modes Kpre C --policy auto --fs --lean --full --reps 2 --lmax 27"
  done
  ;;
gpuv)
  # the verifier on the GPU: v0's streaming GPU verifier (perf.py) and V1 with the (non-streaming) GPU verifier
  for M in opt-1.3b llama2-7b; do
    sub "g-v0-$M" 4 128 "$PY experiments/6_improvements/perf.py --model $M --seq 2048 --modes Kpre --queries 3 --lean \
--wire --prune-last on --lookups on --verifier-device cuda --extra '{\"stream\": true}' --label gpuv"
    sub "g-v1-$M" 6 192 "$V1 --model $M --seq 2048 --modes Kpre --prune-last --lmax 27 --verifier-device cuda"
  done
  ;;
cpuv)
  # B5: the CPU verifier (native kernels: export CXX) of the full models, and re-executing them, on the same
  # 8 threads of this node's CPU; full models measured directly (re-execution also at 2 blocks for the slope)
  [ -n "${CXX:-}" ] || echo "warning: CXX is not set; the CPU verifier falls back to its torch path" >&2
  for M in llama2-7b opt-1.3b; do
    for T in 64 2048; do
      sub "c-$M-$T" 6 256 "$PY experiments/6_improvements/perf.py --model $M --seq $T --modes Kpre C --queries 3 --lean \
--wire on --prune-last on --lookups on --threads 8 --verifier-device cpu --label cpuv"
    done
    sub "c-reexec-$M" 6 256 "$PY experiments/7_reexec/reexec.py --model $M --seq 64 2048 --layers 2 32 --device cpu \
--threads 8 --reps 3"
    # the same with the allocator setting the CPU verifier applies to itself (native_kernels.tune_malloc), for a
    # like-for-like comparison
    sub "c-reexec-heap-$M" 6 256 "MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=2147483647 MALLOC_TOP_PAD_=268435456 \
$PY experiments/7_reexec/reexec.py --model $M --seq 64 2048 --layers 2 32 --device cpu --threads 8 --reps 3"
  done
  ;;
attack)
  # G3.5: edits that keep every FFN activation inside ranges calibrated on held-out text (WikiText-2 train), on the
  # real OPT-6.7B (fp32 on the GPU, about 27 GB); both the unrestricted and the down-only attacker
  for MV in any down; do
    sub "a-inrange-opt67-$MV" 6 128 "HF_HOME=$HF_HOME $PY experiments/2_attack/inrange_llm.py --model facebook/opt-6.7b \
--layers 8 16 24 31 --calib-windows 64 --cap 256 --moves $MV --device cuda"
  done
  ;;
*)
  echo "usage: sp2027.sh smoke|v1|quality|gpuv|cpuv|attack" >&2; exit 2
  ;;
esac

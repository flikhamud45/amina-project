#!/bin/bash
# zkLLM's public demo (https://github.com/jvhs0706/zkllm-ccs2024, commit 993311e, its code unchanged) on this node's
# GPU, timed as the stored runs in code/artifacts/results/zkllm_l40s/ were: the run package's zkllm tier calls it.
#
#     run_zkllm.sh build --home DIR [options]       # once: checkout, environment, build, timing shims, downloads
#     run_zkllm.sh prove --home DIR --size 7|13 --out DIR [--seqs 64,512,1024,2048] [options]
#
# build: the zkLLM home DIR holds zkllm/ (the checkout) and env/ (its Python environment).  A missing checkout is
# cloned (--repo) at --commit; an existing one must be at that commit with no tracked file changed.  A missing env/ is
# created with --python and code/experiments/comparison/zkllm/requirements.txt (torch 2.3.1 for CUDA 12.1,
# transformers 4.40.2; separate from the run's .venv, which it never touches).  The binaries are built for this
# node's GPU (--arch, default its compute capability from nvidia-smi: sm_89 on an L40S) with the nvcc of --cuda-home
# (default: env/bin/nvcc of a conda environment, else CUDA_HOME, else nvcc on the PATH) and the host's gcc, through
# zkLLM's own Makefile (its variables CONDA_PREFIX and ARCH given on make's command line); then the timing shims
# (install_timing_shims.sh).  zkllm/model-storage, where zkLLM's scripts look for the checkpoints, is linked to the
# Hugging Face hub cache --hub (default $HF_HUB_CACHE, else $HF_HOME/hub), into which Llama-2-7B and Llama-2-13B
# (--models) and WikiText-2's test split are downloaded (fetch.py; the gated Llama-2 repositories need HF_TOKEN).
# Options: --python PY (default python3), --cuda-home DIR, --arch sm_NN, --hub DIR, --models 7,13, --repo URL,
# --commit SHA (default 993311e), --jobs N (make's parallel jobs, default 8).
#
# prove: zkLLM's setup (llama-ppgen.py, llama-commit.py) once, then per prompt length of --seqs the layer-0 input
# (mkinput.py: the first N tokens of WikiText-2's test split) and --reps repetitions of the proof scripts of --layers,
# each script under /usr/bin/time -v and each proof binary timed by the shims; nvidia-smi samples the job's GPU every
# 0.5 s.  zkLLM's public parameters and commitments depend on the weights' shapes only, not on the prompt length, so
# every length proves against the one setup (its steps are logged in the first length's folder).  Each length's folder
# is <out>/llama2-<size>b-T<seq>/ (bin_times.log, time-*.txt, nvidia-smi.csv.gz, verdicts.txt, errors.txt, env.txt
# with the host, GPU, driver, CPU and zkLLM's commit, run.json, logs.tar.gz); summarise.py prints each length's record
# (one JSON line) and then rewrites <out>/summary.csv.  A length at which a proof binary fails (zkLLM's lookup
# arguments need power-of-two sizes divisible by their tables' sizes, which short prompts may not give) stops after
# that repetition and is recorded as failed (ok false, with zkLLM's error lines in errors.txt); the next length is
# then proved.  zkLLM writes its fixed-point weights and commitments (27 GB for Llama-2-7B, 52 GB for Llama-2-13B) to
# ./zkllm-workdir, which prove links to a fresh folder in --scratch (default: the first of $TMPDIR, /tmp and the zkLLM
# home with the space free; the free space is checked first) and deletes at its end (a zkllm-workdir that is a folder
# of the checkout, as in a home prepared by hand, is used and kept; the work folders that a pre-empted job of this
# user left are removed first).
# Options: --seqs LIST (prompt lengths, comma-separated; default 64,512,1024,2048; --seq is the same option),
# --allow-failed-lengths (each length's record says "required": false, and a length that fails does not set the exit
# status while another length succeeds), --reps N (default 3), --layers 0,1 (default), --scratch DIR, --parquet FILE
# (default <home>/wikitext-2-raw-v1-test.parquet, which build downloads), --threads N (default 8), --time-cmd PATH
# (GNU time, which times each script; default /usr/bin/time), --need-gb N (the free space the scratch folder must
# have; default 32 for Llama-2-7B, 58 for Llama-2-13B).
# The exit status is 0 only if, at every length, every proof binary exited 0 and every proved layer printed zkLLM's
# verifier messages (with --allow-failed-lengths: at one length at least).
set -uo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
usage() { sed -n '2,/^[^#]/{/^#/p}' "$0" | sed 's/^# \{0,1\}//'; exit "${1:-2}"; }
die() { echo "run_zkllm.sh: $*" >&2; exit 2; }
CMD="${1:-}"; [ $# -gt 0 ] && shift
case "$CMD" in build|prove) ;; -h|--help) usage 0 ;; *) usage ;; esac

ZH="" PYBASE=python3 CUDAH="" ARCH="" HUB="" MODELS=7,13 REPO=https://github.com/jvhs0706/zkllm-ccs2024.git
COMMIT=993311e JOBS=8 SIZE="" OUT="" SEQS=64,512,1024,2048 ALLOWFAIL="" REPS=3 LAYERS=0,1 SCRATCH="" PARQUET=""
THREADS=8
TIMECMD=/usr/bin/time NEED=""
while [ $# -gt 0 ]; do
  case "$1" in
    --home) ZH=$2; shift ;;
    --python) PYBASE=$2; shift ;;
    --cuda-home) CUDAH=$2; shift ;;
    --arch) ARCH=$2; shift ;;
    --hub) HUB=$2; shift ;;
    --models) MODELS=$2; shift ;;
    --repo) REPO=$2; shift ;;
    --commit) COMMIT=$2; shift ;;
    --jobs) JOBS=$2; shift ;;
    --size) SIZE=$2; shift ;;
    --out) OUT=$2; shift ;;
    --seqs|--seq) SEQS=$2; shift ;;
    --allow-failed-lengths) ALLOWFAIL=1 ;;
    --reps) REPS=$2; shift ;;
    --layers) LAYERS=$2; shift ;;
    --scratch) SCRATCH=$2; shift ;;
    --parquet) PARQUET=$2; shift ;;
    --threads) THREADS=$2; shift ;;
    --time-cmd) TIMECMD=$2; shift ;;
    --need-gb) NEED=$2; shift ;;
    -h|--help) usage 0 ;;
    *) die "unknown option $1" ;;
  esac
  shift
done
[ -n "$ZH" ] || die "--home DIR is required (the folder that holds or will hold zkllm/ and env/)"
mkdir -p "$ZH" && ZH="$(cd "$ZH" && pwd)"
Z="$ZH/zkllm" PY="$ZH/env/bin/python"

free_gb() { df -Pk "$1" 2> /dev/null | awk 'NR == 2 { print int($4 / 1048576) }'; }

cuda_home() {   # the CUDA toolkit that builds zkLLM (its Makefile reads $(CONDA_PREFIX)/bin/nvcc and .../include)
  local d="$CUDAH"
  if [ -z "$d" ]; then
    if [ -x "$ZH/env/bin/nvcc" ]; then d="$ZH/env"
    elif [ -n "${CUDA_HOME:-}" ]; then d="$CUDA_HOME"
    elif command -v nvcc > /dev/null 2>&1; then d="$(dirname "$(dirname "$(command -v nvcc)")")"
    fi
  fi
  [ -n "$d" ] && [ -x "$d/bin/nvcc" ] || die "no CUDA toolkit: load one (e.g. module load cuda/12.1) so that nvcc is \
on the PATH, or pass --cuda-home DIR (DIR/bin/nvcc, DIR/include)"
  echo "$d"
}

gpu_arch() {    # sm_<compute capability> of the job's GPU
  [ -n "$ARCH" ] && { echo "$ARCH"; return; }
  local cc
  cc="$(nvidia-smi $GPUSEL --query-gpu=compute_cap --format=csv,noheader 2> /dev/null | head -n 1 | tr -d ' .')"
  [[ "$cc" =~ ^[0-9]+$ ]] || die "cannot read the GPU's compute capability with nvidia-smi: pass --arch sm_NN"
  echo "sm_$cc"
}

# the job's GPU for nvidia-smi: the one CUDA_VISIBLE_DEVICES names, if it names one that nvidia-smi knows
GPUSEL=""
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ] && [[ "$CUDA_VISIBLE_DEVICES" != *,* ]] \
    && nvidia-smi -i "$CUDA_VISIBLE_DEVICES" -L > /dev/null 2>&1; then
  GPUSEL="-i $CUDA_VISIBLE_DEVICES"
fi

# ------------------------------------------------------------------------------------------------------------ build
if [ "$CMD" = build ]; then
  for t in git make gcc g++ nvidia-smi /usr/bin/time; do
    command -v "$t" > /dev/null 2>&1 || die "$t not found on this node (zkLLM's build and timing need it)"
  done
  if [ ! -d "$Z" ]; then
    echo "== clone $REPO at $COMMIT into $Z"
    git clone -q "$REPO" "$Z" && git -C "$Z" -c advice.detachedHead=false checkout -q "$COMMIT" \
      || die "cloning $REPO at $COMMIT failed (no network on the node? clone it beforehand into $Z)"
  fi
  head="$(git -C "$Z" -c safe.directory='*' rev-parse HEAD 2> /dev/null)" || die "$Z is not a git checkout of zkLLM"
  want="$(git -C "$Z" -c safe.directory='*' rev-parse "$COMMIT^{commit}" 2> /dev/null)" \
    || die "$Z does not know commit $COMMIT"
  [ "$head" = "$want" ] || die "$Z is at $head, not at $COMMIT: git -C $Z checkout $COMMIT"
  [ -z "$(git -C "$Z" -c safe.directory='*' status --porcelain --untracked-files=no)" ] \
    || die "$Z has changed tracked files; the demo is run unchanged (git -C $Z checkout -- .)"
  if [ ! -x "$PY" ]; then
    command -v "$PYBASE" > /dev/null 2>&1 || die "--python $PYBASE not found (3.8 to 3.12)"
    "$PYBASE" -c 'import sys; sys.exit(not (3, 8) <= sys.version_info[:2] <= (3, 12))' \
      || die "--python $PYBASE is Python $("$PYBASE" -c 'import platform; print(platform.python_version())'); \
torch 2.3.1 needs 3.8 to 3.12"
    echo "== environment $ZH/env ($("$PYBASE" --version 2>&1))"
    "$PYBASE" -m venv "$ZH/env" || die "$PYBASE -m venv failed (on Debian/Ubuntu: the python3-venv package)"
    "$PY" -m pip install -q --upgrade pip \
      && "$PY" -m pip install -q -r "$HERE/requirements.txt" --extra-index-url https://download.pytorch.org/whl/cu121 \
      || { rm -rf "$ZH/env"; die "installing $HERE/requirements.txt into $ZH/env failed"; }
  fi
  CUDAH="$(cuda_home)" || exit 2
  ARCH="$(gpu_arch)" || exit 2
  if [ -f "$ZH/BUILD.txt" ] && ! grep -qx "arch: $ARCH" "$ZH/BUILD.txt"; then
    die "$Z was built for $(grep '^arch:' "$ZH/BUILD.txt"), this node needs $ARCH: rebuild it (cd $Z && \
rm -f *.real && make clean; rm $ZH/BUILD.txt) or give another --home"
  fi
  if [ ! -f "$Z/self-attn.real" ]; then
    echo "== make ARCH=$ARCH with $CUDAH/bin/nvcc ($("$CUDAH/bin/nvcc" --version | grep -o 'release [0-9.]*'))"
    (cd "$Z" && make -s -j "$JOBS" CONDA_PREFIX="$CUDAH" ARCH="$ARCH" all) || die "zkLLM's build failed (CUDA 12.1 \
needs gcc 12 or older as nvcc's host compiler: load an older gcc, or pass --cuda-home of a newer CUDA)"
    bash "$HERE/install_timing_shims.sh" "$Z" || die "installing the timing shims failed"
  fi
  { echo "arch: $ARCH"; echo "cuda_home: $CUDAH"
    echo "nvcc: $("$CUDAH/bin/nvcc" --version | grep -o 'release [0-9.]*' | cut -d' ' -f2)"
    echo "gcc: $(gcc -dumpfullversion 2> /dev/null || gcc -dumpversion)"; } > "$ZH/BUILD.txt"
  # the checkpoints: zkLLM's scripts read ./model-storage as a Hugging Face hub cache
  if [ ! -e "$Z/model-storage" ]; then
    HUB="${HUB:-${HF_HUB_CACHE:-${HF_HOME:+$HF_HOME/hub}}}"
    [ -n "$HUB" ] || die "no Hugging Face hub cache: export HF_HOME (or pass --hub DIR)"
    mkdir -p "$HUB" && ln -s "$(cd "$HUB" && pwd)" "$Z/model-storage"
  fi
  echo "== download Llama-2 ($MODELS) and WikiText-2's test split into $(readlink -f "$Z/model-storage")"
  HF_HUB_OFFLINE=0 "$PY" "$HERE/fetch.py" --cache "$Z/model-storage" --models ${MODELS//,/ } \
    --wikitext-out "$ZH/wikitext-2-raw-v1-test.parquet" || die "the downloads failed (the Llama-2 repositories are \
gated: export HF_TOKEN=<token> of an account with access)"
  "$PY" -c "import torch, transformers; assert torch.cuda.is_available(), 'torch sees no GPU'; \
print('torch', torch.__version__, 'transformers', transformers.__version__, torch.cuda.get_device_name(0))" \
    || die "zkLLM's environment cannot use the GPU"
  echo ZKLLM_BUILD_OK
  exit 0
fi

# ------------------------------------------------------------------------------------------------------------ prove
case "$SIZE" in 7) NEED=${NEED:-32} ;; 13) NEED=${NEED:-58} ;; *) die "--size 7 or 13" ;; esac
[ -n "$OUT" ] || die "--out DIR is required (the run's zkllm folder)"
SEQS="${SEQS//,/ }"
[ -n "${SEQS// /}" ] || die "--seqs: no prompt length given"
for s in $SEQS; do [[ "$s" =~ ^[1-9][0-9]*$ ]] || die "--seqs: '$s' is not a prompt length (a positive integer)"; done
[ -f "$Z/self-attn.real" ] && [ -x "$PY" ] || die "$ZH holds no built zkLLM with its timing shims and env/: run \
'run_zkllm.sh build --home $ZH' first"
command -v "$TIMECMD" > /dev/null 2>&1 || die "$TIMECMD not found on this node (GNU time; --time-cmd PATH)"
PARQUET="${PARQUET:-$ZH/wikitext-2-raw-v1-test.parquet}"
[ -f "$PARQUET" ] || die "no WikiText-2 test split at $PARQUET (run_zkllm.sh build downloads it; or --parquet FILE)"
CUDAH="$(cuda_home)" || exit 2
ARCH="$(gpu_arch)" || exit 2
if [ -f "$ZH/BUILD.txt" ] && ! grep -qx "arch: $ARCH" "$ZH/BUILD.txt"; then
  die "$Z was built for $(grep '^arch:' "$ZH/BUILD.txt"), this node's GPU is $ARCH"
fi
LAYER_LIST="${LAYERS//,/ }"
mkdir -p "$OUT" && OUT="$(cd "$OUT" && pwd)"
RUNS=()
for s in $SEQS; do
  RUNS+=("$OUT/llama2-${SIZE}b-T$s")
  rm -rf "$OUT/llama2-${SIZE}b-T$s" && mkdir -p "$OUT/llama2-${SIZE}b-T$s"
done

# zkLLM's work folder: a fresh folder of the scratch space, linked from the checkout, deleted at the end
WORK="" LINKED=""
if [ -d "$Z/zkllm-workdir" ] && [ ! -L "$Z/zkllm-workdir" ]; then
  echo "== using $Z/zkllm-workdir (a folder of the checkout, kept)"
else
  # a pre-empted job of this script may have left its work folder behind: this user's are removed first
  for d in "$SCRATCH" "${TMPDIR:-}" /tmp "$ZH"; do
    [ -n "$d" ] && [ -d "$d" ] || continue
    find "$d" -maxdepth 1 -name 'zkllm-workdir.*' -user "$(id -un)" -exec rm -rf {} + 2> /dev/null
  done
  if [ -n "$SCRATCH" ]; then
    mkdir -p "$SCRATCH" || die "cannot create --scratch $SCRATCH"
    gb="$(free_gb "$SCRATCH")"
    [ "${gb:-0}" -ge "$NEED" ] || die "--scratch $SCRATCH has ${gb:-?} GB free; Llama-2-${SIZE}B needs $NEED GB"
  else
    for d in "${TMPDIR:-}" /tmp "$ZH"; do
      [ -n "$d" ] && [ -d "$d" ] && [ -w "$d" ] || continue
      gb="$(free_gb "$d")"
      if [ "${gb:-0}" -ge "$NEED" ]; then SCRATCH="$d"; break; fi
      echo "   $d: ${gb:-?} GB free, $NEED GB needed"
    done
    [ -n "$SCRATCH" ] || die "no scratch folder with $NEED GB free: pass --scratch DIR (e.g. run.sh --zkllm-scratch)"
  fi
  WORK="$(mktemp -d "$SCRATCH/zkllm-workdir.XXXXXX")" || die "cannot create a folder in $SCRATCH"
  ln -sfn "$WORK" "$Z/zkllm-workdir" && LINKED=1
  echo "== zkLLM's work folder: $WORK ($(free_gb "$SCRATCH") GB free)"
fi
SMI=""
cleanup() {
  [ -z "$SMI" ] || kill "$SMI" 2> /dev/null
  if [ -n "$LINKED" ]; then rm -f "$Z/zkllm-workdir"; rm -rf "$WORK"; fi
  for d in "${RUNS[@]}"; do rm -f "$d"/layer-0-input.bin "$d"/r*-L*.bin; done
}
trap cleanup EXIT
trap 'exit 143' TERM INT

cd "$Z" || die "cannot enter $Z"
# zkLLM's scripts call 'make <binary>' (up to date: the shims) and ./<binary>; offline, with the run's threads
export CONDA_PREFIX="$CUDAH" MAKEFLAGS="ARCH=$ARCH" PATH="$ZH/env/bin:$CUDAH/bin:$PATH" PYTHONPATH="$Z" \
  HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS="$THREADS"
ck="$Z/model-storage/models--meta-llama--Llama-2-${SIZE}b-hf/refs/main"
step() { local name=$1; shift; echo "== $name"; "$TIMECMD" -v -o "$R/time-$name.txt" "$@" > "$R/log-$name.txt" 2>&1
  echo "   exit $? $(grep Elapsed "$R/time-$name.txt")"; }
failed_binaries() { grep -c -v -E ' exit 0 +args ' "$R/bin_times.log" 2> /dev/null || true; }

# zkLLM's setup (public parameters and the weights' commitment) does not depend on the prompt length: it runs once,
# in the first length's folder, and every length of this job proves against it
W="./zkllm-workdir/Llama-2-${SIZE}b"
FIRST="${RUNS[0]}" STATUS=0 GOOD=0 BAD=""
for R in "${RUNS[@]}"; do
  SEQ="${R##*-T}"
  export ZK_TIMELOG="$R/bin_times.log"
  : > "$ZK_TIMELOG"
  { hostname
    nvidia-smi $GPUSEL --query-gpu=name,driver_version,memory.total --format=csv,noheader
    lscpu | grep "Model name"
    git -c safe.directory='*' rev-parse HEAD
    cat "$ZH/BUILD.txt" 2> /dev/null
    echo "zkllm_unchanged: $([ -z "$(git -c safe.directory='*' status --porcelain --untracked-files=no)" ] \
&& echo yes || echo no)"
    echo "python: $("$PY" -c 'import platform; print(platform.python_version())')"
    echo "torch: $("$PY" -c 'import torch; print(torch.__version__)')"
    echo "transformers: $("$PY" -c 'import transformers; print(transformers.__version__)')"
    echo "checkpoint: $(cat "$ck" 2> /dev/null || echo unknown)"
    echo "scratch: ${WORK:-$Z/zkllm-workdir}"
    echo "setup: $(basename "$FIRST")"
    echo "slurm_job: ${SLURM_JOB_ID:-none}"; } > "$R/env.txt"
  printf '{"size": %s, "seq": %s, "reps": %s, "layers": [%s], "required": %s}\n' "$SIZE" "$SEQ" "$REPS" "$LAYERS" \
    "$([ -n "$ALLOWFAIL" ] && echo false || echo true)" > "$R/run.json"
  echo "== Llama-2-${SIZE}B at $SEQ tokens: $R"
  cat "$R/env.txt"
  nvidia-smi $GPUSEL --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 500 > "$R/nvidia-smi.csv" &
  SMI=$!
  if [ ! -f "$W/.committed" ]; then
    step ppgen "$PY" llama-ppgen.py "$SIZE"
    step commit "$PY" llama-commit.py "$SIZE" 16 && touch "$W/.committed"
    du -shL "$W/" > "$R/workdir-size.txt"
  fi
  rm -f ./temp*.bin ./rms_inv_temp.bin       # temporaries that a failed script of an earlier length may have left
  step input "$PY" "$HERE/mkinput.py" "$SIZE" "$SEQ" "$PARQUET" "$R/layer-0-input.bin"
  for rep in $(seq 1 "$REPS"); do
    [ -s "$R/layer-0-input.bin" ] || { echo "   no layer-0 input at $SEQ tokens: this length is not proved"; break; }
    in="$R/layer-0-input.bin"
    for L in $LAYER_LIST; do
      p="$R/r$rep-L$L"
      step "r$rep-L$L-rmsnorm1" "$PY" llama-rmsnorm.py "$SIZE" "$L" input "$SEQ" --input_file "$in" \
        --output_file "$p-attn_in.bin"
      step "r$rep-L$L-attn" "$PY" llama-self-attn.py "$SIZE" "$L" "$SEQ" --input_file "$p-attn_in.bin" \
        --output_file "$p-attn_out.bin"
      step "r$rep-L$L-skip1" "$PY" llama-skip-connection.py --block_input_file "$in" --block_output_file \
        "$p-attn_out.bin" --output_file "$p-post_attn.bin"
      step "r$rep-L$L-rmsnorm2" "$PY" llama-rmsnorm.py "$SIZE" "$L" post_attention "$SEQ" --input_file \
        "$p-post_attn.bin" --output_file "$p-ffn_in.bin"
      step "r$rep-L$L-ffn" "$PY" llama-ffn.py "$SIZE" "$L" "$SEQ" --input_file "$p-ffn_in.bin" --output_file \
        "$p-ffn_out.bin"
      step "r$rep-L$L-skip2" "$PY" llama-skip-connection.py --block_input_file "$p-post_attn.bin" \
        --block_output_file "$p-ffn_out.bin" --output_file "$p-out.bin"
      in="$p-out.bin"
    done
    # a proof binary that failed at this length fails the same way in the next repetition: the length stops here
    n="$(failed_binaries)"
    if [ "${n:-0}" -gt 0 ]; then echo "   $n proof binaries failed at $SEQ tokens: no further repetition"; break; fi
  done
  kill "$SMI" 2> /dev/null; wait "$SMI" 2> /dev/null; SMI=""
  grep -h -E "verified|complete|rror" "$R"/log-r*.txt 2> /dev/null | sort | uniq -c > "$R/verdicts.txt"
  # the error lines of the scripts, counted, for the record: zkLLM's C++ exceptions ("what(): ...") and Python's
  # ("...Error: ..."); the first skip connection's "does not exist" is expected (the attention stage writes no output)
  grep -h -E "what\(\)|^[A-Za-z_.]*(Error|Exception):" "$R"/log-*.txt 2> /dev/null | grep -v "does not exist" \
    | sort | uniq -c | head -n 20 > "$R/errors.txt"
  gzip -f "$R/nvidia-smi.csv"
  (cd "$OUT" && tar czf "$R/logs.tar.gz" "$(basename "$R")"/log-*.txt) && rm -f "$R"/log-*.txt
  rm -f "$R"/layer-0-input.bin "$R"/r*-L*.bin
  (cd "$OUT" && "$PY" "$HERE/summarise.py" "$R") || true
  if (cd "$OUT" && "$PY" "$HERE/summarise.py" --record "$R"); then
    GOOD=$((GOOD + 1))
  else
    STATUS=1 BAD="$BAD $SEQ"
    echo "zkLLM's demo did not prove Llama-2-${SIZE}B at $SEQ tokens: recorded (ok false); the other lengths go on" >&2
  fi
done
cd "$OUT" || exit 2
"$PY" "$HERE/summarise.py" --csv "$OUT" > /dev/null || echo "summarise.py --csv $OUT failed" >&2
if [ "$STATUS" -ne 0 ] && [ -n "$ALLOWFAIL" ] && [ "$GOOD" -gt 0 ]; then
  echo "zkLLM's demo proved Llama-2-${SIZE}B at $GOOD of ${#RUNS[@]} lengths and not at${BAD} tokens (recorded; \
--allow-failed-lengths)"
  exit 0
fi
exit "$STATUS"

#!/bin/bash
#SBATCH --partition=killable
#SBATCH --gres=gpu:l40s:1
#SBATCH --exclude=t-806
#SBATCH -J zk-d1
#SBATCH -c 8
#SBATCH --mem=160G
#SBATCH -t 16:00:00
# Plan D.1: zkLLM (commit 993311e, built for sm_89 with CUDA 12.1 + gcc 11) on the same L40S +
# EPYC 9334 nodes as raw_l40s. Usage: sbatch --dependency=singleton zkllm_d1.sbatch <7|13> [seq=2048] [reps=3] [layers="0 1"]
set -uo pipefail
SIZE=$1 SEQ=${2:-2048} REPS=${3:-3} LAYERS=${4:-"0 1"}
Z=/home/dcor/edo/zk
cd $Z/zkllm
export CONDA_PREFIX=$Z/env PATH=$Z/env/bin:$PATH HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 OMP_NUM_THREADS=8
PY=$Z/env/bin/python
OUT=$Z/results/llama2-${SIZE}b-T${SEQ}-$SLURM_JOB_ID; mkdir -p $OUT
export ZK_TIMELOG=$OUT/bin_times.log
{ hostname; nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader; lscpu | grep "Model name"; /home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/strong_gpu_a6000/.tools/bin/git -c safe.directory="*" log -1 --format=%H; } > $OUT/env.txt
nvidia-smi --query-gpu=timestamp,memory.used,utilization.gpu --format=csv -lms 500 > $OUT/nvidia-smi.csv &
SMI=$!
T="/usr/bin/time -v"
W=./zkllm-workdir/Llama-2-${SIZE}b
# zkLLM's fixed-point weights and commitments (27 GB for 7B, 52 GB for 13B) stay in ./zkllm-workdir
# on /home/dcor (our own quota; the course tree has too little room).
step() { local name=$1; shift; echo "== $name"; $T -o $OUT/time-$name.txt "$@" > $OUT/log-$name.txt 2>&1; echo "   exit $? $(grep Elapsed $OUT/time-$name.txt)"; }

if [ ! -f $W/.committed ]; then
  step ppgen $PY llama-ppgen.py $SIZE
  step commit $PY llama-commit.py $SIZE 16 && touch $W/.committed
fi
du -shL $W/ > $OUT/workdir-size.txt
step input $PY mkinput.py $SIZE $SEQ /home/sharifm/teaching/tml-0368-4075/erelbarzilay/final_project/amina-project/logs/hf_cache/wikitext/test.parquet $OUT/layer-0-input.bin

for rep in $(seq 1 $REPS); do
  in=$OUT/layer-0-input.bin
  for L in $LAYERS; do
    p=$OUT/r$rep-L$L
    step r$rep-L$L-rmsnorm1 $PY llama-rmsnorm.py $SIZE $L input $SEQ --input_file $in --output_file $p-attn_in.bin
    step r$rep-L$L-attn     $PY llama-self-attn.py $SIZE $L $SEQ --input_file $p-attn_in.bin --output_file $p-attn_out.bin
    step r$rep-L$L-skip1    $PY llama-skip-connection.py --block_input_file $in --block_output_file $p-attn_out.bin --output_file $p-post_attn.bin
    step r$rep-L$L-rmsnorm2 $PY llama-rmsnorm.py $SIZE $L post_attention $SEQ --input_file $p-post_attn.bin --output_file $p-ffn_in.bin
    step r$rep-L$L-ffn      $PY llama-ffn.py $SIZE $L $SEQ --input_file $p-ffn_in.bin --output_file $p-ffn_out.bin
    step r$rep-L$L-skip2    $PY llama-skip-connection.py --block_input_file $p-post_attn.bin --block_output_file $p-ffn_out.bin --output_file $p-out.bin
    in=$p-out.bin
  done
done
kill $SMI
grep -h -E "verified|complete|rror" $OUT/log-r*.txt | sort | uniq -c > $OUT/verdicts.txt
rm -f $OUT/r*-L*.bin
echo D1_DONE

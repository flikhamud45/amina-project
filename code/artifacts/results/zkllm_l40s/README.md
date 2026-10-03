# zkLLM's public code on our GPU (paper Sec. 4.6 and Table 3)

To compare provers on the same hardware, we ran zkLLM's public demo (commit `993311e` of its
repository, unchanged) on one of our L40S nodes (L40S prover, AMD EPYC 9334 host; TAU's `killable`
partition) for Llama-2-7B and Llama-2-13B at 2,048 tokens. The paper quotes about 844 s for
Llama-2-7B (48x our prover) and does not use the Llama-2-13B run, whose time per layer is 2.5x the
7B figure; Table 3's last row (note e) is the Llama-2-7B run.

## Reproduce the paper's numbers from the stored runs (no GPU)

From `code/`:

```bash
python artifacts/results/zkllm_l40s/summarise.py artifacts/results/zkllm_l40s/llama2-7b-T2048-948715 32
python artifacts/results/zkllm_l40s/summarise.py artifacts/results/zkllm_l40s/llama2-13b-T2048-948716 40
python artifacts/results/zkllm_l40s/summarise.py --csv     # rewrites summary.csv, which paper_assets.py reads for Table 3
```

`summarise.py <run dir> <layers>` prints each proof binary's median time, the time per layer and for
the whole model, the median script wall times and the job GPU's peak memory. For Llama-2-7B it
prints 26.37 s per layer and 844 s for 32 layers, with a peak of 25,020 MiB (24.4 GiB) on the job's
GPU; for Llama-2-13B, 65.71 s per layer. `summary.csv` holds one row per run (`per_layer_s`,
`whole_model_s`).

**How the time per layer is computed.** It is the sum of the median times of the proof binaries in
`bin_times.log` (3 repetitions on each of layers 0 and 1; setup binaries `ppgen` and `commit-param`
excluded). For Llama-2-7B: QKV linear 6.55 s, attention 2.79 s, FFN 12.79 s, the two RMSNorms 1.56
and 1.52 s, and the skip connection 0.58 s, together 25.79 s. The demo's attention stage writes no
output file, so the first skip connection of each layer does not run (its script exits with "Input
or output file does not exist"); the measured skip connection is therefore counted twice, giving
26.37 s. For Llama-2-13B: 64.50 s + 1.21 s = 65.71 s. The peak memory is that of the job's GPU,
picked from `nvidia-smi.csv.gz`, which also records the node's other GPUs (they ran other jobs).

## Files

| File | Contents |
|---|---|
| `zkllm.sbatch` | the job: zkLLM's setup (`ppgen`, `commit`) once, the input, then 3 repetitions of the proof scripts of layers 0 and 1 |
| `install_timing_shims.sh` | wraps each zkLLM binary in a shim that appends its wall time and peak RSS to `bin_times.log`; zkLLM's code is unchanged |
| `mkinput.py` | the layer-0 input: the embeddings of the first 2,048 tokens of the WikiText-2 test split, in zkLLM's fixed-point format |
| `summarise.py` | recomputes every figure above from a run's records, and writes `summary.csv` |
| `<model>-T2048-<job>/` | one stored run (SLURM jobs 948715 and 948716): `bin_times.log` (each binary's time), `time-*.txt` (each script's wall time), `nvidia-smi.csv.gz` (memory and utilisation of every GPU of the node every 0.5 s), `verdicts.txt` (the verifier messages counted), `env.txt` (host, GPU, CPU), `logs.tar.gz` (the scripts' logs) |

## Re-run from scratch (GPU, hours)

Needs zkLLM's checkout at commit `993311e`, built for the GPU (the stored runs: sm_89, CUDA 12.1, gcc
11), with its own conda environment (its scripts need `transformers` 4.40), and the Llama-2
checkpoints (gated on Hugging Face) in `zkllm/model-storage`; the job runs offline.

From the folder that holds `code/`:

```bash
export ZKLLM_HOME=<dir holding zkllm/ and its conda env/>
export WIKITEXT_PARQUET=<wikitext-2-raw-v1 test split, parquet>
cp code/artifacts/results/zkllm_l40s/mkinput.py $ZKLLM_HOME/zkllm/                    # it imports zkLLM's fileio_utils
bash code/artifacts/results/zkllm_l40s/install_timing_shims.sh $ZKLLM_HOME/zkllm      # once
sbatch -p killable --gres=gpu:l40s:1 code/artifacts/results/zkllm_l40s/zkllm.sbatch 7    # Llama-2-7B
sbatch -p killable --gres=gpu:l40s:1 code/artifacts/results/zkllm_l40s/zkllm.sbatch 13   # Llama-2-13B
```

The script's arguments are `<7|13> [seq=2048] [reps=3] [layers="0 1"]`; the stored runs used the
defaults. zkLLM's fixed-point weights and commitments (27 GB for 7B, 52 GB for 13B) stay in its
checkout. Each run writes `$ZKLLM_HOME/results/llama2-<size>b-T<seq>-<job>/`; copy it here and run
`summarise.py` on it.

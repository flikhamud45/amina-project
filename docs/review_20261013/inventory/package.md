# Final-run package review (S&P 2027): `sp2027.sh`, `strong_gpu.sh`, `bench.sbatch`, `smoke.sbatch` and the number pipeline

Repository root: `C:/Users/User/AppData/Local/Temp/claude/C--Users-User-Desktop-amina-project/4aa2c028-956d-4ff2-9cca-56d76e547439/scratchpad/wt_sp2027` (branch sp2027, HEAD 08a7f1f). All paths below are relative to it. This was a read-only review: no edits, no dry-run execution, no tests.

## 0. Main findings

1. **The number pipeline is not connected to the paper.**
   - No `.tex` file inputs `paper_sp2027/generated/`, and that folder does not exist yet.
   - No macro that `sp2027_numbers.py` writes (for example `\CpuRatioN`, `\FinalNHonest`, `\Gen…`, `\Setup…`) appears in `paper_sp2027/main.tex` or `paper_sp2027/sections/*.tex`. The only near-match is the word "Crossover" in a heading (`ext_evaluation.tex:212`).
   - Every final-run number is a hand-typed `\todo{L40S…}`, `\todo{EPYC…}` or `\todo{F6…}`. There are about 60 of them (e.g. `main.tex:73`, `evaluation.tex:48-311`, `ext_*.tex`).
   - `docs/L40S_RUN_SP2027.md:13` says "Nothing has to be computed or copied by hand". For the current paper that is false.
   - Most generated tables do not match the paper's layouts (§5).
2. **Protocol mismatch between the paper's tables and the final run.**
   - The final run is Fiat–Shamir everywhere (`sp2027.sh:46-48,132-136`), under plan `cnn18c` (`:66`).
   - `tab:eval-llm` says "interactive … Proof bytes do not change" (`evaluation.tex:74-77`). `tab:options` says "ours interactive" (`evaluation.tex:185`).
   - Fiat–Shamir changes the proof size: GPT-2, mode C, 64 tokens is 14.8 MB interactive (`evaluation.tex:90`) against 16.6 MB under Fiat–Shamir (`evaluation.tex:141,278`).
   - `tab_eval_llm` puts the base columns (interactive, plan `auto`) next to the new ones (Fiat–Shamir, `cnn18c`), and fills the proof column with "new, else base" (`sp2027_numbers.py:336-337,360-361`).
3. **Confirmed bugs in `sp2027_numbers.py`** (§9):
   - The quality table pools the two α values into one median.
   - The generation table mixes real-weight and random-weight Llama-2-7B rows, and one macro overwrites the other.
   - V1 records do not record which device ran the verifier.
   - `--strict` can never pass.
   - Real Llama-2-7B V1 cells have no random-weight twin.
4. **Superseded course-report content is still in the package.** `strong_gpu.sh` keeps the `report-must`, `should` and `nice` tiers and its own `smoke` (→ `smoke.sbatch`). The S&P paper uses none of them (§8).
5. **Cluster and person details remain.**
   - TAU names and partitions: `train.sbatch:3,11`, `sp2027.sh:9`.
   - The run guide is written for a teammate, with dates and references to TAU-only diaries.
   - Every raw record carries hostname, SLURM partition, node list and `PVI_*` paths (`bench.py:161-166,182-184,281`). That breaks anonymity if the final-run outputs are shipped as the artifact.

## 1. Package structure and call graph

- **`sp2027.sh <tier>`** (`code/experiments/4_defence_benchmark/slurm/sp2027.sh`). The tier is the only positional argument (`:442-447`); everything else is an environment variable.
  - It submits each job as a bash script on stdin (`:108`), or runs it in the foreground under `PVI_LOCAL` (`:101-104`).
  - Tiers `bench` and `batch` call `strong_gpu.sh must|batch` (`:241-253`). That script submits `bench.sbatch` jobs, and each one runs `bench.py <args> --threads $PVI_THREADS` (`bench.sbatch:41`).
- **Cross-script dependencies the package needs:**
  - `perf.py` → `6_improvements/ab_verifier.py` (`perf.py:162`; only for the CNN helpers), `4_defence_benchmark/bench.py` (`:207`), `8_v1/v1_bytes.py` (`:237`, with `--real`)
  - `v1_bytes.py` → `9_quality/ppl_llama.py` (`:69-70`, which needs `opt_builder.py` and `ppl_2048.py`, `ppl_llama.py:67-69`), and `9_quality/common.py` (`:96`; pyarrow)
  - `gen_bench.py` → `7_reexec/serve_ab.py` for `TINY` (`:74`) and `v1_bytes.py` (`:209`)
  - `setup_timing.py` → `gen_bench.py` (`:63`)
  - `serve_ab.py` → `reexec.py` (`:436`)
  - `sp2027_numbers.py` → `paper_assets.py` (`:984`) and `count_outcomes.py` (`:861`). `paper_assets.py` is the course-report figure and table builder: it imports matplotlib at module level (`:48-55`) and points `REPORT` at `../report` (`:697-699`).
- **Data the pipeline needs that is not produced by the run:**
  - `code/artifacts/comparison/tables_l40s` (block counts, `sp2027_numbers.py:985`)
  - `tables_l40s_improved{,2,3}` (base columns, `:89,1082`)
  - the trained image models and the CIFAR-10 and dogs-vs-cats data (`sp2027.sh:424-438`; git does not carry the models, `docs/improvements/F6_final_run_package.md:176`)
  - the `--expect` listing `docs/improvements/F6_dryrun.txt` (`L40S_RUN_SP2027.md:157`)
- **Dry-run listing versus script.** `F6_dryrun.txt` (137 jobs, `:152`) was regenerated at ca75493. Since then `git diff ca75493 HEAD -- slurm/` shows comment-only changes to `sp2027.sh` (HF_HOME 40→60 GB, the `real` and `quality` descriptions), so the listing still matches the commands. No test compares the listing with a fresh dry run; `tests/test_f6_package.py:112-188` checks properties only.

## 2. Environment variables

**Read by `sp2027.sh`:**

| Variable | Line | Role |
|---|---|---|
| `PVI_PLATFORM` (required) | `:55` | run name, log and raw folders |
| `PVI_DRYRUN` | `:59` | print sbatch lines only |
| `PVI_REHEARSAL` | `:60` | smallest cells; forces `QH=Q=2`, overriding `PVI_QUERIES*` (`:69`) |
| `PVI_LOCAL` | `:61` | no SLURM, run in the foreground |
| `USER` | `:62` | for `squeue -u` |
| `PVI_PYTHON` | `:65` | interpreter |
| `PVI_C_POLICY` | `:66` | mode C plan, default `cnn18c` |
| `PVI_QUERIES_HEAD`, `PVI_QUERIES` | `:67-68` | timed queries, default 5 and 3 |
| `PVI_MEM_MAX` | `:70` | cap on `--mem` |
| `PVI_CHAIN` | `:71` | chain jobs |
| `PVI_EXCLUSIVE` | `:72` | `--exclusive` |
| `PVI_DEP` | `:73` | first dependency |
| `TRITON_CACHE_DIR` | `:76` | Triton cache |
| `PVI_SETUP_CACHE` | `:77` | setup cache folder |
| `PVI_EXTRA_PYTHONPATH` | `:78` | extra packages |
| `PVI_PY_INCLUDE` | `:81` | Python headers for Triton |
| `CXX`, `PVI_NATIVE_DIR` | `:82-84` | native kernels |
| `SBATCH_GRES` | `:96` | GPU request (`SBATCH_PARTITION` and `SBATCH_ACCOUNT` are read by sbatch itself) |
| `HF_HOME` | `:310,381` | model downloads |
| `PVI_LLAMA_LICENCE` | `:335,387` | enables the Llama-2 cells |
| `PVI_Q7_ARGS`, `PVI_Q13_ARGS` | `:394,396` | extra quality arguments |
| `PVI_BIGMEM` | `:419` | full model at 8,192 tokens |
| `PVI_OPTIONAL` | `:455` | add `quality` and `long` to `all` |
| `PVI_CIFAR_ROOT`, `PVI_CACHE`, `PVI_IMAGENET_ROOT` | `:430-431` | image data |

**Exported into every `sp-` job** (`:78-85`): `PYTHONPATH=code/src[:extra]`, `OMP_NUM_THREADS=8`, `MKL_NUM_THREADS=8`, `NVIDIA_TF32_OVERRIDE=0`, `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, `TRITON_CACHE_DIR`, `PVI_SETUP_CACHE`, plus `PVI_PY_INCLUDE`, `CXX` and `PVI_NATIVE_DIR` when set. Two jobs add their own: the smoke sets `PVI_TEST_DEVICE=cuda` (`:151`), and `c-rx-heap` sets `MALLOC_MMAP_MAX_=0 MALLOC_TRIM_THRESHOLD_=2147483647 MALLOC_TOP_PAD_=268435456` (`:188`).

**`strong_gpu.sh`:**
- Unsets `PVI_THREADS`, `PVI_TF32` and `PVI_LEGACY_WEIGHT_KEY` (`:40`).
- Reads `PVI_PLATFORM` (`:41`), `PVI_DRYRUN` and `PVI_LOCAL` (`:49-55`), `PVI_DEP`, `PVI_CHAIN`, `PVI_MEM_MAX` and `PVI_DRYN0` (`:65`), `PVI_EXCLUSIVE` (`:72`), `PVI_C_POLICY` (default `auto`, `:105`), `PVI_REHEARSAL` (`:124,184`) and `PVI_LAST_FILE` (`:301`).
- Exports `PROJECT_DIR` (`:44`). The `thr1` jobs get `--export=ALL,PVI_THREADS=1` (`:154,156`).

**`bench.sbatch`:** `PROJECT_DIR`/`SLURM_SUBMIT_DIR` (`:17`), `PVI_PLATFORM` (`:20`), `PVI_THREADS` (default 8, `:23`), `PVI_TF32` (`:25`), `PVI_PYTHON` (`:31`). It sets `NVIDIA_TF32_OVERRIDE=0` and unsets `TORCH_ALLOW_TF32_CUBLAS_OVERRIDE` (`:28-30`).

**Read inside the scripts:**
- `PVI_V1_BUCKETS`, which `v1_bytes.py` sets itself for `--v1-paths` (`v1_bytes.py:435-446`; `buckets.py:63`).
- `PVI_LEGACY_WEIGHT_KEY` and `NVIDIA_TF32_OVERRIDE` (`bench.py:1076,1081`).
- `PVI_PLATFORM` (`bench.py:1056`, `aggregate.py:220`, `paper_assets.py:59`).
- `PVI_C_POLICY` (`sp2027_numbers.py:1083`).

**Library switches the package never sets.** These select earlier code paths:
- `PVI_FUSED_CODEC` (`claimcodec.py:1543`), `PVI_MERKLE_PROCESSES` (`commitment.py:76`), `PVI_GKR_TRITON` (`cut_protocol.py:50`), `PVI_GKR_PROFILE` (`gkr_triton.py:1043`)
- in `native_kernels.py`: `PVI_NATIVE_ATTN` (`:901`), `PVI_NATIVE_AVX512` (`:995`), `PVI_NATIVE_CHEAP` (`:1065`), `PVI_INT32_CLAIMS` (`:1066`), `PVI_NATIVE_DECODE` (`:1175`), `PVI_MALLOC_TUNE` (`:1250`), `PVI_NATIVE_GEMM` (`:1264`), `PVI_V1_NATIVE` (`:1281`)
- in `protocol.py`: `PVI_HASH_THREADS` (`:129`), `PVI_GPU_DECODE` (`:532`), `PVI_FOLD_FAST` (`:869`)
- in `transformer.py`: `PVI_FUSED_ATTN`, `PVI_CPU_ATTN` (`:46-47`)
- `PVI_LEGACY_WEIGHT_KEY` (`graph.py:61`)

These are the "previous code path" options that the paper's improvement comparisons mention (`ext_evaluation.tex:51-86`). Your instruction says such superseded variants stay as non-default options. As environment variables they are not CLI arguments.

## 3. Shorthands used in the job tables

- **V1** = `experiments/8_v1/v1_bytes.py --fs --lean --full --prune-last --lmax 27 --threads 8` (`:132`)
- **PERF** = `experiments/6_improvements/perf.py --fs --lean --wire on --prune-last on --lookups on --threads 8` (`:133`)
- **RX** = `experiments/7_reexec/reexec.py --threads 8` (`:134`)
- **RXCPU** = `int int_fp32gemm float bf16 dynint8` (`:135`)
- **GEN** = `experiments/6_improvements/gen_bench.py --fs --threads 8` (`:136`)
- **POL** = `$PVI_C_POLICY`, else `cnn18c`. **QH** = 5, **Q** = 3.
- **bench prefix** = `bench.sbatch llm|cnn …` (adds `--threads 8`).
  - **OC** = `--policy POL --wire --prune-last --tag _wire`
  - **OCG** = `--policy POL --wire --prune-last --verifier-device cuda --verifier-impl stream --tag _wire_gpuv_stream`
  - **OK** = `--lookups --wire --prune-last --tag _wire`
  - **OB** = `--wire --prune-last --tag _wire_batch` (`strong_gpu.sh:108-110,183`)

Every `sp-` job also gets `--cpus-per-task=8 --mem=<capped> --time=<h> --exclusive --gres=$SBATCH_GRES --dependency=afterany|afterok,singleton` (`:93-96`). Bench jobs get `--time=<min> --exclusive [--mem]` (`strong_gpu.sh:71-76`) and the `#SBATCH` defaults `gpu:1`, 8 CPUs, 96 G (`bench.sbatch:3-6`).

## 4. Job inventory: script, arguments and what each job feeds

| Tier and jobs | Limits | Command | Feeds |
|---|---|---|---|
| **smoke**: `sp-smoke` | 2 h, 64 G | 16 test files (`:143-147`) with `PVI_TEST_DEVICE=cuda`; `native_kernels.available()` if `CXX`; `field.int8_ok("cuda")`; `V1 --model gpt2 --layers 2 --seq 64 --modes C Kpre --policy POL --reps 2 --lmax 20`; `h2d_bandwidth.py --sizes-mb 256 --reps 3 --label smoke`; `echo SMOKE OK` (`:151-154`) | Gate (`afterok`, `:458`). Side effect: its V1 records add a stray "GPT-2, 2/12, T=64" row to `tab_eval_v1` (that table does not filter on tier, `sp2027_numbers.py:588`), and its h2d record goes into `tab_h2d` |
| **cpuv**: `sp-c-{llama2-7b,qwen3-4b,opt-1.3b}-{64,2048}` | 1 h/128 G at 64; 2 h/256 G at 2048 | `PERF --model m --seq T --modes Kpre C --policy POL --queries QH --verifier-device cpu --label cpuv` (`:174-184`) | `tab_options` (C and Kpre proof and CPU verifier, ratio, `CpuRatio*`, `CpuRatioN`); `tab_eval_reexec` CPU "our Kpre verifier" (Llama only); `tab_spread`; `tab_memory` |
| cpuv: `sp-c-rx-llama2-7b` (3 h/256 G), `-qwen3-4b` (2 h/256 G), `-opt-1.3b` (1 h/128 G) | — | `RX --model m --seq 64 2048 --layers 2 <L> --variants RXCPU --device cpu --reps QH` (`:185-187`) | `tab_options` "strongest exact re-execution (CPU)"; `tab_eval_reexec` CPU rows |
| cpuv: `sp-c-rx-heap-llama2-7b` | 2 h/256 G | `MALLOC_*` settings + `RX … --variants int float dynint8` (`:188-189`) | Same tables; the faster allocator setting is kept (`sp2027_numbers.py:478`) |
| cpuv: `sp-c-v1-llama2-7b-{64,2048}`, `sp-c-v1-qwen3-4b-2048` | 1-2 h/128-256 G | `V1 --model m --seq T --modes Kpre --reps QH --v1-paths 0 1 --label cpuv-v1` (`:190-192`) | `tab_options` V1 columns (`:514-521`); the Qwen3-4B cell also adds a new row to `tab_eval_v1` |
| **v1**: `sp-v1-gpt2-{64,512}`, `-opt125-{64,2048}`, `-opt13-{64,2048}`, `-llama7-{64,2048}`, `-opt67-{64,2048}`, `-llama13-64`, `-llama13-2k-2` (`--layers 2`), `-qwen4-64` | 1-3 h | `V1 --model m --seq T --modes C Kpre --policy POL --reps Q` (QH for OPT-1.3B and Llama-2-7B) (`:204-214`) | `tab_eval_v1`; `Crossover*` macros (meant for `ext_evaluation.tex:212-223`); `tab_spread` |
| **serve**: `sp-serve-{opt13,qwen4,llama7,opt67,llama13}` | 3-8 h, 128-256 G | `serve_ab.py --workload a b --modes C Kpre --batch-proof seq joint --reps 10 --device cuda --threads 8 --u2-dtype fp16 bf16 --policy POL --label serve --model m` (`:228-238`) | `tab_serve`; `Serve…OverUOne`/`OverUTwo`. Paper target: text at `evaluation.tex:111-114`, `tab:options` footnote f (`:218-221`), `ext_discussion.tex:139-141` |
| **bench** image models: `{c,k}-{mlp_mnist,lenet5,vgg11,vgg16,resnet18_cifar,resnet18_224}` | 30/20 min | `cnn --model m --queries 30 --tampers 100` (60 for resnet18_224) `--policy POL --wire --tag _wire`; the `k-` jobs use `--tampers 0 --wire --tag _wire` (`strong_gpu.sh:133-137`) | `c-vgg16` → `tab_eval_compare` VGG-16 (`sp2027_numbers.py:388`); all of them → `tab_counts`/`Final*`. Nothing generates `c-lenet5`'s row (`ext_evaluation.tex:291-296`) or the image-model cost range (`evaluation.tex:104-105`) |
| bench, 64 tokens: `{c,k,cg}-{gpt2,qwen3-4b,llama2-7b,llama2-13b}-64` | per `:141-142` | `llm --model m --seq 64 --builds full --queries 5 --lams 128`, then `--modes C:fs --llm-tampers 30 --lean OC` / `--modes Kpre:fs --lean OK` / `--modes C:fs --lean OCG` (`:140-149`) | `tab_eval_llm` new columns; `tab_eval_compare` GPT-2 (64); counts |
| bench: `c-gpt2-512`, `c-llama2-7b-1` | — | `--seq 512` and `--seq 1` with `C:fs OC` (`:150-152`) | `tab_eval_compare` DeepProve 512 and ZKTorch rows |
| bench: `k-qwen3-4b-8-thr1`, `…-thr1-wire` | `-c 1`, `PVI_THREADS=1` | `llm --model qwen3-4b --seq 8 --builds full --queries 5 --lams 40 --modes Kpre:int --lean --lookups --prune-last [--wire] --tag _thr1[_wire]` (`:154-157`) | `tab_eval_compare` Maverick row |
| bench, 2,048 tokens: `{c,k,cg}-{opt-125m,opt-1.3b,opt-6.7b,llama2-7b,llama2-13b}-2k` | per `:160-167` | `llm --seq 2048 --builds full --queries q --lams 128` with `--modes C:fs,C:int` (`C:fs` only for OPT-6.7B) `OC`, `Kpre:fs OK`, and the same C modes with `OCG` (`:159-174`) | `tab_eval_llm` 2,048-token block; `tab_eval_compare` zkLLM rows (interactive; prover and proof from the CPU cell, verifier from the GPU cell, `:397-402`); `tab_bench_memory` |
| **gpuv**: `sp-g-v0-{opt-1.3b,llama2-7b}-{Kpre,C}` | 1 h/192 G | `PERF --model m --seq 2048 --modes <mode> --policy POL --queries Q --verifier-device cuda --extra '{"stream": true}' --label gpuv` (`:269-273`) | Only `tab_spread` and `tab_memory`. No table reads these records; the paper's GPU-verifier numbers come from the bench `cg-` cells |
| gpuv: `sp-g-v1-{opt-1.3b,llama2-7b}` | 2 h/192 G | `V1 --model m --seq 2048 --modes Kpre --reps Q --verifier-device cuda --label gpuv-v1` (`:274`) | Effectively nothing: `tab_eval_v1` drops them as duplicates. See bug B3 |
| gpuv: `sp-g-rx-{opt-1.3b,qwen3-4b,llama2-7b,opt-6.7b,llama2-13b}` | 2 h/128 G | `RX --model m --seq 64 512 2048 --layers 2 <L> --variants int float bf16 --device cuda --reps QH` (`:276-279`) | `tab_eval_reexec` GPU rows (int and float = fp16); `tab_options` int8 GPU column. The bf16 and 2-block results are unused |
| gpuv: `sp-g-h2d` | 1 h/32 G | `h2d_bandwidth.py --sizes-mb 64 1024 4096 --reps 10 --label gpuv` (`:280`) | `tab_h2d`, `\HtoDPinnedGBs`. The paper uses neither |
| **gen**: `sp-gen-{llama2-7b,qwen3-4b}` | 3 h/192 G | `GEN --model m --prompt 512 --gen 64 256 --decode greedy sample --modes C Kpre --policy POL --queries Q --label gen` (`:292-295`) | `tab_gen`, `Gen…` macros. Paper target: `generation.tex:55` todo, `tab:gen-full` (`ext_generation.tex:88-115`) |
| **batch**: `b-llama2-7b-64-{c,cint,k}`, `b-opt-1.3b-512-{c,k}`, `b-llama2-7b-2k-{c,k}` | 20-90 min | `llm --builds full --queries 3 --lams 128 --batches …` (4 16 64 / 4 16 / 2 4) `--batch-trials 3 --lean` with `--policy POL OB` or `--lookups OB` (`strong_gpu.sh:190-203`) | `tab_batch`. Paper target: text `ext_discussion.tex:141-145` (it quotes B = 8, which the final run does not measure) |
| **real** (OPT): `sp-r-opt67{,rnd}-v0-{64,2048}`, `sp-r-opt67{,rnd}-v1-{64,2048}`, `sp-r-opt13{,rnd}-v1-{C,Kpre}-{64,2048}` | 1-2 h, 192-256 G | Real cells: `HF_HOME=… PERF/V1 --model opt-6.7b --real facebook/opt-6.7b --recipe certified …` (and the OPT-1.3B equivalent), `--tampers 3 --label real`; random twins: same without `--real`, `--label real-random` (`:311-334`) | `tab_real` → `tab:eval-real` (`ext_evaluation.tex:132-163`) |
| real (Llama, with `PVI_LLAMA_LICENCE`): `sp-r-llama7{,rnd}-v0-{64,2048}`, `sp-r-llama7-v1-{64,2048}`, `sp-r-llama7{,rnd}-v0C-64`, `sp-r-llama7-gen` | 1-4 h | `--model llama2-7b --real meta-llama/Llama-2-7b-hf …`; the gen cell is `gen_bench.py … --prompt 512 --gen 64` (`:339-352`) | `tab_real`; the gen cell lands in `tab_gen` (bug B2). The current paper uses OpenLLaMA-7B instead (`evaluation.tex:59-60`) |
| **setup**: `sp-setup-llama2-7b`, `sp-setup-gpt2` | 3 h/128 G; 1 h/64 G | `setup_timing.py --model m --policy POL --amortise 100 [1000] --reps 1\|3 --threads 8 --label setup` (`:364-367`; no `--fiat-shamir`) | `tab_setup_time`, `Setup…Prover`/`Verifier`. Paper target: `defence.tex:221`, `ext_defence.tex:122-123`, `discussion.tex:41`, `ext_discussion.tex:172` |
| **quality** (optional, licence): `sp-q-llama7`, `sp-q-llama13` | 6 h/192 G; 8 h/256 G | `ppl_llama.py --model meta-llama/Llama-2-{7,13}b-hf --config … --seq 2048 --windows 0 --calib-windows 4 --prefix --graphs smooth --glu-smooths 0.2 --structure --device cuda --calib-device cpu --threads 8 --alphas 0.55 0.65` (13B: `0.6`) `--ref-dtype float32\|float16 $PVI_Q*_ARGS` (`:391-396`) | `tab_quality` (bug B1). The current paper has no Llama-2 quality row (`quality.tex:28-45,84-88`) |
| **long** (optional): `sp-l-llama7-{2048,4096,8192,16384}-L{2,3}`, `sp-l-rx-llama7-<T>`, `[sp-l-llama7-8192-full]` | 2-6 h, 192-256 G | `PERF --model llama2-7b --layers l --seq T --modes Kpre --queries Q --verifier-device cpu --label long`; `RX … --layers 2 3 --variants RXCPU --reps Q --max-pos 16384` (`:411-420`) | `tab_long`, `LongFit*` → `tab:eval-long` (`ext_evaluation.tex:178-194`). The generated table is per block; the paper's is extrapolated to the full model |
| **Rehearsal** (22 jobs, `docs/improvements/F6_dryrun_rehearsal.txt`) | — | Branches at `sp2027.sh:166-170,200,231,262-265,288-289,315-317,360-361,383-384,404-407` and `strong_gpu.sh:125-129,185-188` | `--rehearsal` check only (`sp2027_numbers.py:1039-1046`) |

## 5. Number pipeline: outputs, inputs and paper targets

`aggregate.py --platform P` reads `raw_P/**/*.jsonl` that have a `.done` marker, skipping the `env` rows (`:61-75`). It writes `tables_P/measured_summary.csv` (grouped by suite, model, cell, metric and the `GROUP_KEYS`, `:51,116-139`) and `llm_full_model.csv`, built from the additive metrics on full builds or 1- and 2-block builds (`:56-58,168-214`).

`sp2027_numbers.py` reads the `sp-*.out` lines that start with `{`. The record kind is inferred from its fields (`:103-117`), and the tier from the job-name prefix (`:65-66`).

| Output (`paper_sp2027/generated/`) | Builder | Inputs (record kind and fields) | Paper target | Layout match |
|---|---|---|---|---|
| `tab_eval_llm.tex` | `:333-374` | base: `paper_assets.opt_llm` on `tables_l40s_improved*` (interactive, plan `auto`); new: `llm_cost` on `llm_full_model.csv` (PROVE/VERIFY parts, `bytes_total`), cells chosen by `OPTIMISED` tags with `pol<policy>`, Fiat–Shamir, full builds only | `tab:eval-llm` (`evaluation.tex:73-102`) | 10 columns, matches; caption conflicts (finding 2) |
| `tab_eval_llm_full.tex` | same | adds C and Kpre CPU verify new | caption todo `evaluation.tex:77` | no such table |
| `tab_eval_compare.tex` | `:377-407` | `Measured.find_cnn("vgg16","C",fs,128,("wire",pol))`; `llm_cost` for GPT-2 64/512 and Llama-2-7B 1 (C, CPU, Fiat–Shamir); Qwen3-4B 8 `thr1` (interactive); zkLLM (C, interactive, CPU and GPU) | `tab:eval-compare` "ours" columns (`evaluation.tex:262-299`) | ours-only; no LeNet-5 row, no V1 column (`tab:eval-compare-v1`, `ext_evaluation.tex:270-289`) |
| `tab_eval_v1.tex`, `Crossover<Model>L<b><T><mode>` | `:582-615` | kind `v1`: `v0_total`, `v1_total`, `timed.{v0,v1}.{prove_s,verify_s,fs_s}`, `layers`, `seq`, `mode`; v1 tier preferred; excludes `codec_only` and `real` | `tab:eval-v1` (`evaluation.tex:126-157`); crossover paragraph (`ext_evaluation.tex:212-223`) | extra T column; about 25 rows against the paper's 16 |
| `tab_options.tex`, `CpuRatioN`, `CpuratiominX`, `CpuratioX<T>` | `:494-544` | `reexec` (CPU, `measured_s[str(n_layers)]`, variants in `EXACT`); `reexec` (cuda, int); `perf` cpuv (Fiat–Shamir, `verify_s`, `proof_bytes`); `v1` cpuv | `tab:options` (`evaluation.tex:182-222`), abstract `main.tex:73`, `intro.tex:118`, `evaluation.tex:238-245` | per-model data table, not the paper's options layout |
| `tab_eval_reexec.tex` | `:547-579` | `reexec` cuda int/float (= fp16); cpuv `perf` Kpre; `reexec` CPU variants | `tab:eval-reexec` (`ext_evaluation.tex:229-256`) | 6 CPU rows against 3 |
| `tab_serve.tex`, `Serve<Model><T><mode>OverUOne/OverUTwo` | `:618-652` | kind `serve`, `row=="prove"`: `u2_s[fp16]`, `u1_s`, `prove_s`, `prover_over_u1/u2`, `proof_mb_per_token`, `egress_cap…` | text only (no serve table in the paper) | n/a |
| `tab_gen.tex`, `Gen…Mspertoken` | `:655-674` | kind `gen`: `prove_s(_range)`, `verify_s(_range)`, `proof_bytes`, `proof_mb_per_token`, `tampers` | `tab:gen-full`, `tab:gen` (`ext_generation.tex:67-112`), `generation.tex:45-55` | different columns |
| `tab_batch.tex` | `:410-440` | `measured_summary.csv`, `_batch` cells: `bytes_total`, PROVE/VERIFY parts ÷ B | `ext_discussion.tex:141-145` | no table in the paper |
| `tab_memory.tex`, `tab_memory_bench.tex` | `:677-693`, `:443-461` | `peak.gpu_peak_memory`, `host_peak_rss`; bench `gpu_peak_memory`, `host_peak_rss` | todos `evaluation.tex:110`, `ext_evaluation.tex:113`, `ext_discussion.tex:44` | no table in the paper |
| `tab_real.tex` | `:696-729` | `perf` and `v1` in tier `real` or with `label` f3a*/real*; real against random by `real` | `tab:eval-real` (`ext_evaluation.tex:132-148`) | roughly |
| `tab_setup_time.tex`, `Setup<Model>A<a>Prover/Verifier` | `:732-752` | kind `setup`: `proof_bytes`, `commit_s`, `prove_s`, `verify_s`, `query_columns_bytes_*` | `tab:setup` (`ext_defence.tex:125-140`) and setup-time todos | adds time columns, one row per a |
| `tab_quality.tex` | `:755-780` | kind `ppl`: `ppl`, `ratio`, `top1`, `cell`, `windows` | `tab:quality` (`quality.tex:28-45`, OPT only; stored F2a records) | final run adds Llama rows the paper lacks |
| `tab_long.tex`, `LongFitA/B/MaxResidual` | `:783-816` | tier `long` `perf` (layers 2 and 3) and `reexec` | `tab:eval-long` | per block, not full model |
| `tab_h2d.tex`, `HtoDPinnedGBs` | `:819-828` | kind `h2d` | none | — |
| `tab_spread.tex` | `:831-853` | cpuv and gpuv `perf` ranges; v1 and cpuv V1 ranges | "medians with their spread" todo (`evaluation.tex:48`) | no table |
| `tab_counts.tex`, `Final{NRecords,NHonest,NHonestCNN,NHonestLLM,NAttacks,NAttacksCNN,NAttacksLLM,NFreivalds,NColumnsCode,NColumnsMerkle,NRange,NAttacksPlusOne,NAttacksPlusOnePartial[,NAttacksPlans]}` | `:856-877`, `count_outcomes.py:191-205` | bench raw records only: `accepted` of `defence_*`, `rejected`/`tamper_rejected`, `stage` | `evaluation.tex:63-67`; `ext_evaluation.tex:204-205` asks for "including V1 and the setup proof", which these counts do not cover | text |
| `numbers.tex`, `cells.csv`, `coverage.txt` | `:1058-1074` | — | nothing inputs them | — |

Macro names go through `str.title()` (`:240-245`), so the real names are `\CpuratiominLlamaTwoSevenB`, `\GenLlamaTwoSevenBKpreGreedyGSixFourMspertoken` and so on. Only `\CpuRatioN` keeps the camel case the docstring shows (`:27`). I checked this by evaluating the function on these inputs.

## 6. Parameters that are hard-coded rather than CLI arguments

**Drivers.**
- `sp2027.sh` has no flags at all. Its only argument is the tier (`:442-447`); every option in §2 is an environment variable.
- `strong_gpu.sh` has no flags either (`:112-299`).
- Inside `sp2027.sh`, these are fixed:
  - models, lengths and block counts per tier: `:174-175,185-192,204-214,234-238,269-280,292-295,311-352,364-367,393-396,411-420`
  - Hugging Face repo ids: `:311-312,314,339,383,393,395`
  - threads: `OMP=8` (`:78`), `--cpus-per-task=8` (`:93`), `--threads 8` (`:132-136,229,352,361,365-367,384,392`)
  - `--lmax 27` and 20 (`:132,153,169,200,317`)
  - RX variants (`:135`); allocator values (`:188`)
  - serve: reps, workloads, U2 dtypes, plus the rehearsal cells (`:228-231`)
  - h2d sizes and reps (`:154,265,280`)
  - `--extra '{"stream": true}'` (`:263,272`)
  - prompt and generation lengths (`:288-295,351`); tampers (`:316-352`)
  - setup amortisation and reps (`:360-367`)
  - the quality recipe: α, glu-smooth, windows, reference dtype (`:391-396`; `PVI_Q*_ARGS` is the only escape hatch)
  - job hours and memory in every `sub` call
  - the `nl()` layer table, which duplicates `CONFIGS` (`:117-122`); `est()` (`:124-130`)
  - the smoke's test list (`:143-147`); the TF32-off and allocator settings (`:76-79`)
  - `MUST` (`:451`); the image models and data paths that `need_images` checks (`:426-433`)
  - rehearsal forces `QH=Q=2` (`:69`)
- **`strong_gpu.sh`:**
  - Every bench cell's `--queries`, `--tampers`, `--llm-tampers`, `--lams`, `--batches`, `--batch-trials`, time and memory are literals (`:133-203`). `PVI_QUERIES*` from `sp2027.sh` never reaches them.
  - `POL` defaults to `auto` (`:105`) while `sp2027.sh` and `sp2027_numbers.py` default to `cnn18c`.
- **`bench.sbatch`:** `#SBATCH` gres, CPUs, memory and time (`:2-6`); threads default 8 (`:23`).

**Python.**
- **`bench.py`:**
  - The CNN suite always runs λ ∈ {40, 80, 128} × {C:int, C:fs, K:int, Kpre:int} (`:85,593-594`). `--lams` applies to the LLM suite only (`:1011`).
  - `RATE=4` (`:86`); seeds (`:449,638,826,832,939`).
  - Under `--policy paper` the CNN jobs always run the facts, attack_float and sampling cells (`:494-575`, `_todo` `:307-313`), with a fixed sampling grid (`:568`).
- **`perf.py`:** `bench.RATE` (`:268,284`); `calib_tokens=min(seq,32)`, seed 0 (`:252`); query seed 1 (`:255`); tamper seed 123 and magnitudes (`:91,118,130`); one warm-up (`:297`).
- **`v1_bytes.py`:**
  - `SMOOTH` (`:51`); rate literal `4` (`:392`)
  - seeds 0, 1, 100+i and 7 (`:363,365,413,453`); tamper seed 0 (`:202`)
  - real Llama calibration: 4,000,000 characters and 4×2,048 windows (`:74-75`); `norm_gain=4` (`:119`)
  - the record omits `verifier_device` (`:368-421`)
- **`reexec.py`:**
  - the float dtype is fp16 on the GPU and fp32 on the CPU (`:207`); warm-up 1 (`:57`); seeds (`:108,219`); RMSNorm ε (`:134`)
  - `int` is built pruned (`:226-227`) but `float` and `bf16` are not (`:244`)
- **`serve_ab.py`:** `RATE`, `CALIB_TOKENS`, `WORKLOADS`, 100 Gb/s egress, `RESIDUAL_MAX`, `BOOT` (`:108-118`). Fiat–Shamir, wire encoding, lean prover and pruning are always on (`:898-899`).
- **`gen_bench.py`:** `RATE`, `CALIB_TOKENS` (`:57-58`); with `--real` the `certified` recipe and `calib_seq` 0 are fixed (`:209-210`); seed rule (`:103-105`); tamper position (`:276`); lean, wire, pruning and Kpre lookups always on (`:217-218,257-263`).
- **`setup_timing.py`:** `RATE=4`, `calib_tokens=32`, `cache=False`, `prune_last=True` (`:30,76-81`).
- **`h2d_bandwidth.py`:** both memory kinds and both directions always run (`:61,64`).
- **`sp2027_numbers.py`:** table definitions `LLM_ROWS`, `ZKLLM`, `FULL_LAYERS` (which duplicates `CONFIGS`), `EXACT`, `REPORT_RUNS`, `STORED_LOGS` (`:72-89`); `pa.TABLES=tables_l40s` (`:985`).

## 7. Cluster-specific and non-anonymous content

**In the package:**
- `slurm/train.sbatch:3` (`#SBATCH --partition=studentbatch`) and `:11` ("TAU: studentbatch").
- `slurm/sp2027.sh:9` ("the TAU nodes: transformers, pyarrow"). `:80` and `:124` point to the TAU-only diaries `docs/improvements/I_server_setup.md` (`:1` "TAU student GPUs", `:42-47` personal server-folder paths and micromamba) and `F6_final_run_package.md`.
- `docs/L40S_RUN_SP2027.md`:
  - `:3` "For the teammate"; `:4-6,59,123-126` schedule dates; `:41` freeze tag names
  - `:22-23` refer to `I_server_setup.md`
  - `:35,80-81,148-150` rules about other people's tokens and files; `:105` cites the F6 diary for estimates
- `docs/improvements/F6_final_run_package.md`:
  - `:77` "TAU nodes"; `:82,259` "friend"
  - `:153-158` TAU cluster, `studentbatch`, `/tmp/$USER/hf`; `:172` "dash on the TAU nodes"
  - job IDs at `:114,119,179`
- Internal plan IDs throughout the comments ("Plan 2 F6 part 9", "chg 12", A7, C4a, RX, F3a and others, e.g. `sp2027.sh:2,17-26`; `strong_gpu.sh:10-20`).
- `strong_gpu.sh:5,41` and `bench.sbatch:20` name the course report's frozen roots. These are not cluster-specific.

**What the records themselves contain:**
- `bench.py` writes an `env` row into every cell file (`:281`). It holds hostname, `SLURM_JOB_PARTITION`, `SLURM_JOB_NODELIST`, `SLURM_JOB_GPUS`, `CUDA_VISIBLE_DEVICES` and every `PVI_*` value, including paths under the runner's home (`:161-166,182-184`). `PLATFORM.json` stores the host (`:117`).
- `perf.py:329`, `v1_bytes.py:373`, `reexec.py:270`, `gen_bench.py:239` and `setup_timing.py` write `"host": platform.node()`.
- Every `sp-` log starts with `host $(hostname)` and the git log line (`sp2027.sh:97`); `bench.sbatch:36-40` prints host, GPU and CPU details.
- `smoke.sbatch:28` dumps `env | sort` into `logs/env/` (it would capture tokens such as `HF_TOKEN`), but the S&P package does not use it.

**Next to the package:**
- `code/experiments/4_defence_benchmark/README.md:170,182-183,189-198,221-223` (`killable`, node `t-806`, `gpu:l40s:1`).
- `code/artifacts/comparison/raw_rtx2080ti-v2/PLATFORM.json:11` ("TAU SLURM studentbatch").

## 8. What in the package the current paper does not use

**Whole files or tiers:**
- `strong_gpu.sh` `report-must`, `should`, `nice` (`:206-296`; the script itself says "not in the S&P package", `:19-20`). The paper never mentions TF32, `_nofix`, `_nolean`, 12 threads, OPT-13B/2.7B/350M or the 30-70B shapes (I searched `sections/`).
- `strong_gpu.sh smoke` → `smoke.sbatch`. It needs `resnet18_224_squirrel` and the dogs-vs-squirrels data (`smoke.sbatch:60-65`), which the paper never uses, and fingerprints against `raw/`. `sp2027.sh` has its own smoke tier.
- `train.sbatch` is not called by the package. It is only needed to retrain the image models; keep it as an option without the TAU partition.

**Unused or redundant jobs and variants:**
- gpuv: `sp-g-h2d` (no paper use); bf16 GPU re-execution and the 2-block builds (`:277`); `sp-g-v1-*` (no table, B3); `sp-g-v0-*`, whose records reach only `tab_spread` and `tab_memory`.
- serve: bf16 U2, and the `on_request` rows (no table).
- quality, both jobs: the paper reports OpenLLaMA-7B, not Llama-2 (`quality.tex:84-88`). The α 0.65 row and the "smooth without glu" row are unused.
- real, the Llama-2-7B cells: the paper uses OpenLLaMA-7B (`evaluation.tex:59-60`).
- long, `PVI_BIGMEM`.
- cpuv: the Qwen3-4B and OPT-1.3B rows go beyond `tab:options`, which is Llama-2-7B only.
- v1: OPT-125M, OPT-1.3B, OPT-6.7B and Qwen3-4B at 64 tokens, OPT-6.7B at 2,048, and full Llama-2-13B at 64 are not rows of `tab:eval-v1`. Meanwhile Llama-2-13B at 2,048 is only 2 blocks, so the `\todo{V1}` for the 13B row (`ext_evaluation.tex:286`) cannot be filled.
- bench:
  - the `k-<cnn>` K/Kpre cells, the CNN λ 40 and 80 cells, and the paper-plan mode C cells feed only the counts
  - the facts, attack_float and sampling cells feed nothing
  - `mlp_mnist` needs MNIST downloaded on the node

**Inside `sp2027_numbers.py`:** the `--stored` and `--compare-paper` paths regression-check against the course report's runs. `paper_assets.py` is the course-report builder. `5_comparison/README.md:1-30` and `6_improvements/README.md:1-19` describe the course report ("the paper's benchmark numbers do not come from these files"). `7_reexec/`, `8_v1/` and `9_quality/` have no README.

**Next to the package:** `paper_sp2027/sections/generation_quality.tex` is not input anywhere and redefines `tab:gen` and `tab:quality` (`:81,200`).

## 9. Bugs in the number pipeline

| # | Status | File:line | Defect | Failure scenario |
|---|---|---|---|---|
| B1 | Confirmed | `sp2027_numbers.py:763-777`; `ppl_llama.py:398` | Quality records are grouped by the graph name only; α is a separate field | `sp-q-llama7` writes α 0.55 and 0.65 under one name, so the "ratio" is the median of the two α values and "draws" counts 2 |
| B2 | Confirmed | `sp2027_numbers.py:659-671` | `tab_gen` does not filter `real` | `sp-r-llama7-gen` (real Llama-2-7B, P=512, G=64) becomes an unlabelled row next to `sp-gen-llama2-7b`. Its `Gen…GSixFour…` macros overwrite the random-weight ones, because `sp-r-` sorts after `sp-gen-` |
| B3 | Confirmed (missing field) | `v1_bytes.py:368-421`; `sp2027_numbers.py:588-593` | V1 records have no `verifier_device`, and `tab_eval_v1` keys on (seq, model, layers, mode) across all tiers | If a v1-tier cell fails, `sp-g-v1-*` (GPU verifier) silently fills a CPU-verifier cell. The smoke's 2-block GPT-2 records add a stray row |
| B4 | Confirmed (defaults) | `strong_gpu.sh:8,105` against `sp2027.sh:66` and `sp2027_numbers.py:1083` | Default plan `auto` against `cnn18c` | `strong_gpu.sh must` run directly, as its header shows, writes `_polauto` cells, and `sp2027_numbers.py` then finds no new cells |
| B5 | Confirmed | `sp2027_numbers.py:1054-1055` | `--strict` fails whenever any cell is missing, including base cells that are "--" in the paper (e.g. `evaluation.tex:91,93,98`) and V1 columns that no job produces (OPT-1.3B in `tab_options`) | `--strict` can never return 0 |
| B6 | Confirmed | `sp2027.sh:339-346`; `sp2027_numbers.py:710-725` | `r-llama7-v1-*` has no random twin | `tab_real` always shows missing "V1 … random" cells for Llama-2-7B |
| B7 | Confirmed | `sp2027_numbers.py:244` | `.title()` mangles macro names | names do not match the docstring or any planned `\newcommand` use |
| B8 | Plausible (method) | `reexec.py:226-227` against `:244` | The `int` baseline is pruned like our graph; `float` and `bf16` are not | The fastest exact CPU baseline (fp32 at 2,048 tokens) does one unpruned last block, roughly 1/32 extra for Llama-2-7B. That inflates the "×" against re-execution by about that much (my estimate, not measured) |
| B9 | Minor | `L40S_RUN_SP2027.md:131-132` | The returned tarball excludes `.triton` and `.native` but not `.setup_cache` (about 0.5 GB, `:26`) | larger tarball than needed |

## 10. High-level decisions this package forces

1. **Connect the generated tables and macros to the paper, or don't.** Either `\input` them, which means making `tab_eval_v1`, `tab_options`, `tab_eval_reexec`, `tab_gen`, `tab_setup_time` and `tab_long` produce the paper's layouts and adding generators for LeNet-5, the image-model cost range and `tab:eval-compare-v1`. Or declare `cells.csv` the source for transcription and correct `L40S_RUN_SP2027.md:13`.
2. **Interactive versus Fiat–Shamir.** Switch `tab:eval-llm` and `tab:options` (captions `evaluation.tex:74-77,185`) to Fiat–Shamir and drop or relabel the base columns, which are interactive under plan `auto`. Otherwise add interactive full-model cpuv cells. As it stands the proof column silently changes protocol and plan.
3. **The abstract's N.** `CpuRatioN` is the floor of the smaller of the 64- and 2,048-token Kpre Fiat–Shamir ratios (`sp2027_numbers.py:540-541`). On the Xeon the 64-token ratio is 2.9× (`evaluation.tex:243-244`), so the abstract's "4.7 times" (`main.tex:73`) would likely become "at least 2". Decide which length the abstract quotes.
4. **One platform.** `tab_real` expects the pre-freeze 2080 Ti OPT-1.3B v0 records alongside the L40S run (`sp2027.sh:302-304`). That conflicts with "final tables from one platform" (`evaluation.tex:46-48`).
5. **Llama-2 or OpenLLaMA.** Either the paper adopts the `quality` and `real` Llama-2 tiers (this needs the runner's licence), or those tiers leave the package.
6. **Outcome counts.** They cover only bench cells. The perf, V1, generation and setup tampers of the `sp-` tiers are not counted, although `ext_evaluation.tex:204-205` asks for V1 and the setup proof.
7. **`ev_breakeven.pdf`** (`evaluation.tex:224-233`) comes from hand-curated 2080 Ti/Xeon inputs (`5_comparison/ev_inputs_draft.json`). The pipeline does not regenerate it.
8. **Re-execution baselines.** Prune all of them, or state that `float` is the unpruned off-the-shelf stack (B8).
9. **Packaging.**
   - Move the `--expect` listing out of the diary.
   - Replace the environment switches with flags: tier, `--platform`, `--dry-run`, `--rehearsal`, `--local`, `--policy`, `--queries[-head]`, `--mem-max`, `--no-chain`, `--no-exclusive`, `--optional`, `--llama-licence`, `--bigmem`.
   - Expose the library A/B switches as CLI options.
   - Strip host, partition and path fields from published records, or anonymise them.
   - Decouple `sp2027_numbers.py` from the course-report `paper_assets.py` and matplotlib.

## 11. What I could not determine

- I did not run the dry run (`tier_strong` creates a temp file, `sp2027.sh:242`), any tests or any model code. Job counts and arguments come from reading the scripts and `F6_dryrun.txt`.
- I did not read the selection internals of `paper_assets.Measured`, `llm_cost` and `opt_llm`; I relied on their docstrings. Same for `ppl_2048.py`, `opt_builder.py`, `common.py`, `gen_bench.config` and `setup_proof`. I also cannot say whether `verify_setup` really runs on the CPU, as the `setup_timing.py` docstring claims.
- I did not check:
  - whether the time limits are realistic, or whether `cg-qwen3-4b-64` and `cg-llama2-13b-64` (which the course report never measured, `paper_assets.py:497-498`) fit their limits;
  - whether the setup cache key separates real from random weights;
  - whether the extended version is meant to keep the generated per-model tables rather than the paper's current layouts.
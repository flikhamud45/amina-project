# 5. Tables, counts and figures from the stored measurements (paper Sec. 4)

These scripts never run a model. They read the stored benchmark records in `artifacts/comparison/`
(described in [`artifacts/comparison/README.md`](../../artifacts/comparison/README.md)) and produce
every table and figure of the paper, the soundness counts, and a check of every benchmark number
typed in the text. They need no GPU and take about a minute in total on a laptop CPU.

The paper uses three runs on the same machines (an NVIDIA L40S prover and 8 threads of an AMD EPYC
9334 verifier):

* platform **`l40s`** (`raw_l40s/`): the **basic protocol**;
* platforms **`l40s_improved`** and **`l40s_improved2`** (`raw_l40s_improved/`, `raw_l40s_improved2/`):
  the **optimised protocol**, read together (where both hold a cell, the second's is used).

## Regenerate everything

Run from `code/`, in this order. `paper_assets.py` and `count_outcomes.py --tex` write into
`../report/figures/` and `../report/tables/`, so the `report/` folder must sit next to `code/` (as in
the submission). Use matplotlib 3.11.2 (`requirements.txt`) for the figures: it reproduces the
committed PDFs byte for byte, while other versions draw them slightly larger or smaller.

```bash
export PYTHONPATH=$PWD/src
python experiments/5_comparison/aggregate.py --platform l40s              # raw_l40s/ -> tables_l40s/
python experiments/5_comparison/aggregate.py --platform l40s_improved     # raw_l40s_improved/ -> tables_l40s_improved/
python experiments/5_comparison/aggregate.py --platform l40s_improved2    # raw_l40s_improved2/ -> tables_l40s_improved2/
python experiments/5_comparison/literature.py                             # published results -> tables/reported_curated.csv
python experiments/5_comparison/analytic.py                               # the path protocol on Llama-2-7B -> tables/analytic.csv
python experiments/5_comparison/count_outcomes.py --platform l40s --tex ../report/tables/counts.tex
python experiments/5_comparison/count_outcomes.py --platform l40s_improved,l40s_improved2 --prefix Opt --definition \
    --tex ../report/tables/counts_opt.tex
python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved,l40s_improved2
python experiments/5_comparison/text_numbers.py --platform l40s --optimised l40s_improved,l40s_improved2 --definition
cd ../report && latexmk -pdf main.tex                                     # or: tectonic -X compile main.tex
```

`text_numbers.py` exits with 0 when every checked number in `report/main.tex` matches the tables. The
remaining checks of Sec. 4.1:

```bash
python experiments/5_comparison/fingerprint_check.py artifacts/comparison/raw_rtx2080ti-v2 artifacts/comparison/raw_l40s
#   1978 fingerprints compared, 0 problems  (the 1,978 hardware-independent values of Sec. 4.1)
python experiments/5_comparison/count_outcomes.py --platform rtx2080ti-v2     # the second platform's verdicts
python experiments/5_comparison/validate_extrapolation.py --platform l40s     # optional: a line fit through every block count
python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved,l40s_improved2 --check
#   writes nothing; lists the pending numbers (below)
```

`--platform` defaults to `$PVI_PLATFORM` when it is set, else to `l40s`. A new measurement (see
[`4_defence_benchmark`](../4_defence_benchmark/README.md)) goes through the same commands with its own
platform name.

## What each script does

| Script | Reads | Writes or prints |
|---|---|---|
| `aggregate.py --platform <p>` | `raw_<p>/**/*.jsonl` | `tables_<p>/measured_summary.csv` (median and mean per model, cell, metric and stage), `llm_full_model.csv` (each language model's full-model cost: the full build where there is one, otherwise extrapolated from its 1- and 2-block builds), `llm_extrapolation_check.csv` (every full build against its extrapolation). `--platform rtx2080ti` rebuilds the tables of `raw/` in `tables/` |
| `literature.py` | `literature/reported_benchmarks.csv` | `tables/reported_curated.csv`: the 41 published rows compared with ours |
| `analytic.py` | the model shapes | `tables/analytic.csv`: the path protocol on Llama-2-7B at 64 tokens (1/11,008 per path, 305,193 paths for 2^-40, 13.7 GB opened) |
| `count_outcomes.py --platform <p>[,<q>...]` | `raw_<p>/` (several roots are added up) | the soundness counts of Sec. 4.3: honest queries accepted, attacks rejected by type, the check that rejected each, the '+1' attacks on language models, and the record total. `--definition` counts only the defence and tamper cells of the optimised definition; `--require-tag TAG` only cells carrying a tag; `--tex F --prefix P` writes the counts as LaTeX macros (`\NHonest`, `\NAttacks`, ...; `\OptNHonest`, ... with `--prefix Opt`) and refuses if an honest query was rejected or an attack accepted |
| `paper_assets.py --platform <p> --optimised <q>,<r>` | `tables_<p>/` (basic), `tables_<q>/`, `tables_<r>/` (optimised), `tables/reported_curated.csv`, `artifacts/results/zkllm_l40s/summary.csv`, `artifacts/comparison/excluded_cells.csv` | `../report/figures/*.pdf` (Figs. 1–4, one file per sub-figure, on a fixed canvas equal to its printed width) and `../report/tables/{cnn,llm,ratios,hardware}.tex` (Tables 1–3, and the hardware macros `\ProverGPU`, `\VerifierCPU`, `\VerifierThreads`, `\LLMNote`, `\NPendingCells`). `--check` writes nothing and lists missing basic cells (an error) and pending optimised numbers; `--pending-csv F` writes them to a CSV; `--strict` refuses to write while any number is pending; `--draft` writes figures despite layout problems |
| `text_numbers.py --platform <p> --optimised <q>,<r> --definition` | the runs' tables, the raw roots, `../report/tables/counts*.tex`, `../report/main.tex` | compares every benchmark number typed in the text with the value recomputed as the tables compute it (each sentence found by its wording); exits 1 on a mismatch, on a final number still marked pending, on a number that depends on a pending value but is not marked, or on a sentence it can no longer find. `--list` only prints the recomputed values |
| `validate_extrapolation.py --platform <p>` | `tables_<p>/measured_summary.csv` | `tables_<p>/llm_extrapolation_validation.csv`: the two-point rule and a line fit through every block count, against the full builds |
| `fingerprint_check.py <root A> <root B>` | two raw roots | checks that every hardware-independent value the two roots share (parameters, accuracies, path-protocol facts, security parameters, proof bytes except the Merkle term, verdicts) agrees, and prints every honest rejection and accepted attack of B |

## The optimised protocol's definition

Which stored cell holds each optimised number is defined in one place, `OPTIMISED` in
`paper_assets.py`, as the cell's settings and its set of variant tags (in any order). These tags are
written by `bench.py`'s options (see [`4_defence_benchmark`](../4_defence_benchmark/README.md)):

| Setting | Tags | `bench.py` options |
|---|---|---|
| image models, committed weights (C), interactive and Fiat–Shamir | `_wire_polauto` | `--policy auto --wire --tag _wire` |
| image models, known weights (K, Kpre) | `_wire` | `--wire --tag _wire` |
| language models, C, CPU verifier | `_wire_prune_polauto` | `--policy auto --wire --prune-last --tag _wire` |
| language models, C, streaming GPU verifier | `_wire_gpuv_stream_prune_polauto` | the same with `--verifier-device cuda --verifier-impl stream --tag _wire_gpuv_stream` |
| language models, Kpre (CPU and streaming GPU verifier) | `_wire_prune_lookups`, `_wire_gpuv_stream_prune_lookups` | `--lookups --wire --prune-last --tag _wire` (and the GPU options) |
| Qwen3-4B in Maverick's setting (Kpre, λ = 40, one verifier thread) | `_thr1_wire_prune_lookups` | `PVI_THREADS=1`, `--lams 40 --modes Kpre:int --lookups --prune-last --wire --tag _thr1_wire` |

Llama-2-7B's runs without pruning (`_wire_polauto`) are the one accepted stand-in for a missing
defining cell. In a language-model row the prover time and the proof come from the CPU-verifier
cell, the GPU verifier from the GPU cell. Table 3 uses the Fiat–Shamir cells against the
non-interactive systems, the interactive cells against zkLLM (itself interactive) and Maverick, and
for Maverick our Kpre run without the compact encoding (note d); it counts Maverick's 37.4 ms
non-linear replay in its verifier time. Tables 1–3 and Fig. 4 use full-depth builds only; the 30–70B
shapes enter only the text, through their 1- and 2-block builds. Fig. 3 draws no path curve for the
224-pixel ResNet, which lies on VGG-16's.

## Numbers of the text that no script checks

`text_numbers.py` checks the benchmark numbers of 39 sentences. The other numbers of the text come
from these sources:

| Paper | Value | Source | Command (from `code/`) |
|---|---|---|---|
| Sec. 1 | zkLLM needs about ten minutes for one Llama-2-7B prompt | zkLLM's reported 620 s in `artifacts/comparison/tables/reported_curated.csv` | `python experiments/5_comparison/literature.py` |
| Sec. 1, 4.2 | 99.6–99.8% acceptance of one tampered neuron on MNIST | `artifacts/results/attack.json` (`baselines`) | `python experiments/2_attack/run.py` ([2_attack](../2_attack/README.md)) |
| Sec. 1, 4.2 | 40/40 and 0/40 prompts; 16,384 neurons reach 6,187 of 50,272 tokens; 1/16,384 per path, 454,248 paths | `artifacts/results/real_llm_attack.json`, `real_llm_attack_auto.json` | `real_llm.sbatch` ([REAL_LLM.md](../2_attack/REAL_LLM.md)) |
| Sec. 3.5 | VGG-16 at λ = 128: r = 5, t = 67 | `cfg_reps`, `cfg_columns` of `vgg16,defence_C_int_lam128_rate4` in `artifacts/comparison/tables_l40s/measured_summary.csv` | `python experiments/5_comparison/aggregate.py --platform l40s` |
| Sec. 3.6.1 | 349, 23, 385 and 4 MB of the basic protocol's 762 MB proof (Llama-2-7B, 64 tokens) | `bytes_claims`, `bytes_u`, `bytes_columns`, `bytes_paths`, `bytes_total` of `llama2-7b,defence_C_int_lam128_T64_L32` in the same file | the same |
| Sec. 3.6.2 | six candidate plans; at most twice the basic setup, or 2^34 | `AUTO_CANDIDATES`, `AUTO_MIN_BUDGET` in `src/pvi/fullcheck/plans.py` | ([6_improvements](../6_improvements/README.md#7-the-planning-rule-auto)) |
| Sec. 3.6.2 | LeNet-5: one tree of length 2^18 and 15 columns instead of 66 in each of five trees | `cfg_group_columns` (`g0_n262144: 15`) of `lenet5,defence_C_int_lam128_rate4_wire_polauto` in `tables_l40s_improved2/measured_summary.csv`; `cfg_columns` 66 of `lenet5,defence_C_int_lam128_rate4` in `tables_l40s/` | `python experiments/5_comparison/aggregate.py --platform l40s_improved2` |
| Sec. 3.6.2 | a claim takes 16–19 bits with the compact encoding | `experiments/6_improvements/results/wire_laptop.json` (16.4–19.2 bits per claim) | `python experiments/6_improvements/wire.py --out experiments/6_improvements/results/wire_laptop.json --summary` ([6_improvements](../6_improvements/README.md#3-the-compact-encoding)) |
| Sec. 4.1 | perplexities 73.8, 37.4, 36.0 against 64.7, 34.9, 27.1 | `artifacts/results/real_weights_ppl_opt-*.json` | `real_weights_ppl.sbatch` ([REAL_LLM.md](../2_attack/REAL_LLM.md)) |
| Sec. 4.1 | all 1,978 hardware-independent values reproduced on the RTX 2080 Ti | `raw_rtx2080ti-v2/` against `raw_l40s/` | `python experiments/5_comparison/fingerprint_check.py artifacts/comparison/raw_rtx2080ti-v2 artifacts/comparison/raw_l40s` |
| Sec. 4.2 | *Reproduction*: 3,000 runs per setting, 99.8–100%, 21–37%, 150 queries | `artifacts/results/reproduction.json` | `python experiments/1_reproduction/run.py` ([1_reproduction](../1_reproduction/README.md)) |
| Sec. 4.2 | *Attacks on MNIST*: hundreds of the 778 nodes against 1–14, backdoor 99.96% of 2,300 runs, about five neurons, 61% with 250 paths | `artifacts/results/attack.json` | `python experiments/2_attack/run.py` |
| Sec. 4.2 | *Other samplers*: 0.20%, 0.11%, 0.016%; 14 of 512 neurons, 65%, 2,000 of 2,000; floor 2% (10x), 0.9–6x | `artifacts/results/defence.json`, `floor_sampler.json` | `python experiments/3_sampling_fixes/run.py`, `floor_sampler.py` ([3_sampling_fixes](../3_sampling_fixes/README.md)) |
| Sec. 4.3 | the range check caught the remaining two attacks | `rejected by: ... range_or_shape 2` | `python experiments/5_comparison/count_outcomes.py --platform l40s_improved,l40s_improved2 --definition` |
| Table 3, note c | DeepProve's HyperKZG proof, 9.4 MB | `artifacts/comparison/literature/reported_benchmarks.csv` | (the catalogue) |
| Sec. 4.6 | our prover faster than Maverick by 200% | Table 3's Maverick prover ratio, 3.0x | `paper_assets.py` (`report/tables/ratios.tex`) |
| Sec. 4.6 | zkLLM's demo on our L40S: about 844 s; 2.5x per layer for Llama-2-13B | `artifacts/results/zkllm_l40s/` | `python artifacts/results/zkllm_l40s/summarise.py artifacts/results/zkllm_l40s/llama2-7b-T2048-948715 32` ([zkllm_l40s](../../artifacts/results/zkllm_l40s/README.md)); the ratios are also checked by `text_numbers.py` |

## Pending values

Some optimised values are not final: the timings withheld in `artifacts/comparison/excluded_cells.csv`
(cells of the second batch that ran while nodes n-801 and n-804 were slowed by other users' load; see
[`artifacts/comparison/README.md`](../../artifacts/comparison/README.md)) and two optional cells that
were not run (the streaming GPU verifier of Qwen3-4B and Llama-2-13B at 64 tokens). `paper_assets.py`
draws each such value in red, with the basic protocol's value as a placeholder (or `--` where there is
none), and sets `\NPendingCells` (currently 19); `text_numbers.py` checks that the text marks every
number that depends on one. When the cells are re-measured in a new root (planned as platform
`l40s_improved3`), adding it at the end of `--optimised` (`l40s_improved,l40s_improved2,l40s_improved3`)
replaces the placeholders, and `text_numbers.py` reports each number of the text that moves.

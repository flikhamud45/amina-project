# Comparison with published systems, and the benchmark's tables and counts

Sec. `sec:eval-compare` sets our costs against published zkSNARKs and Maverick on matched models, each as reported on
its own hardware; the benchmark cells behind our side, and behind `tab:eval-llm`, are records of `costs/bench.py` (the run
package's `bench` tier; the stored roots below hold the values the paper prints before the final run). These scripts never run a model: `literature.py` turns the catalogue of published
measurements into the table the comparison quotes, `aggregate.py` turns a platform's raw benchmark records into its
tables, `count_outcomes.py` counts every honest query and every attack in them, and `fingerprint_check.py` checks that
two platforms agree on every hardware-independent value. `paper/numbers.py` reads their outputs (`paper/bench.py`).

| Paper result | Script | Command (from `code/`) | Output |
|---|---|---|---|
| `tab:eval-compare`, the "theirs" columns (e.g. zkLLM on Llama-2-7B at 2,048 tokens: 10.3 min, 2.36 s, 183 kB); `tab:eval-compare-v1` and the LeNet-5 paragraph (`app:eval-compare-v1`); published costs quoted in the text | `literature.py` | `python -m experiments.comparison.literature` | `artifacts/comparison/tables/reported_curated.csv` (41 rows) |
| `tab:eval-compare`, the "ours" columns; `tab:eval-llm`; the image models' costs (Sec. `sec:eval-cost`, "Costs") | `aggregate.py` | `python -m experiments.comparison.aggregate --platform final` (the run); the stored roots: `--platform l40s_improved` (and `l40s_improved2`, `l40s_improved3`) | `artifacts/comparison/tables_<platform>/`, read by `paper/numbers.py` |
| Sec. `sec:break`: the path test's detection probabilities and bytes on the image models | `aggregate.py` | `python -m experiments.comparison.aggregate --platform l40s` | `artifacts/comparison/tables_l40s/` |
| Sec. `sec:eval-setup`: "accepted all 3,792 honest queries and rejected all 2,512 attacks", "Freivalds' check rejected 1,601, the column check 853, the Merkle check 56 and the range check 2" | `count_outcomes.py` | `python -m experiments.comparison.count_outcomes --platform l40s_improved,l40s_improved2,l40s_improved3 --definition --policy auto --prob-max 255` | stdout |
| `app:eval-setup`: "Proof sizes do not depend on the hardware" (all 1,978 hardware-independent values of the L40S run reproduced on an RTX 2080 Ti with a Xeon Silver 4114) | `fingerprint_check.py` | `python -m experiments.comparison.fingerprint_check artifacts/comparison/raw_l40s artifacts/comparison/raw_rtx2080ti-v2` | stdout (`1978 fingerprints compared, 0 problems`), exit status |

Run every command from `code/` with the checkout's library first on the path (`PYTHONPATH=src:.` on Linux,
`PYTHONPATH="src;."` on Windows). `python -m experiments.comparison.<script> --help` lists every option. None needs a
GPU; each takes seconds to a minute on a laptop CPU.

## The stored records

`artifacts/comparison/` (described in its own README) holds one raw root per platform, `raw_<platform>/<suite>/
<model>/<cell>.jsonl` (`<suite>`: `cnn` or `llm`; a finished cell has a `.done` marker; `PLATFORM.json` names the
root's GPU and CPU), the tables `aggregate.py` derives from it in `tables_<platform>/`, the literature catalogue and
the curated table, and `excluded_cells.csv` (timings of the stored roots withheld from the values the paper prints before the final run; no
script reads it). The
roots the paper uses, all with an NVIDIA L40S prover and 8 threads of an AMD EPYC 9334 as the CPU verifier unless
stated:

| Root | What it holds | Used for |
|---|---|---|
| `raw_l40s/` | the first L40S run: the per-op commitment, proofs without the compact encoding | the path test of Sec. `sec:break` |
| `raw_l40s_improved/`, `raw_l40s_improved2/`, `raw_l40s_improved3/` | the base implementation (the compact encoding, the planning rule `auto`, the pruned last block, the verifier's own embedding rows); a later root re-measures cells of an earlier one, and its cell replaces the earlier one whole | the values of `tab:eval-llm`, our side of `tab:eval-compare` and the counts of Sec. `sec:eval-setup` before the final run (their defining cells: `--definition`) |
| `raw_rtx2080ti-v2/` | the first run's cells on an RTX 2080 Ti with 8 threads of an Intel Xeon Silver 4114 | the hardware-independence check |

The final run writes a root of its own (`costs/bench.py --platform <name>`); `raw_l40s_improved2_thr1/` (one verifier
thread) is not used, and the earliest root `raw/` with its tables in `tables/` (`measured_summary.csv`,
`llm_full_model.csv`) is not rebuilt by these scripts.

## Scripts

### `literature.py`

`artifacts/comparison/literature/reported_benchmarks.csv` holds every number the literature search extracted (413
rows from 32 systems, each with its table or page and a verbatim snippet of the source, each re-checked against the
paper by a second reader). `literature.py` picks the 41 rows of its `CURATED` list (zkCNN, vCNN, ZKML, ZENO, Bionetta,
EZKL, ZEN, Mystique, zkPyTorch; zkLLM, zkGPT, DeepProve, ZKTorch, Jolt Atlas, SLP, LAMP, Anchuri et al., Maverick),
converts them to seconds and bytes, and links each to the catalogue row it came from (a curated entry whose source text
the catalogue does not hold stops the script). Options: `--catalogue`, `--out` (default
`artifacts/comparison/tables/reported_curated.csv`). Output columns: `system`, `model`, `params`, `seq`, `prover_s`,
`verifier_s`, `proof_bytes`, `hardware`, `catalog_row_id`, `source_location`, `venue`, `snippet`. Maverick's verifier
time is its whole client time (its Table 8), as ours is our whole verifier. CommitLLM's costs (`tab:options`) are not
in the catalogue: they are quoted from its paper.

### `aggregate.py`

Reads the finished cells of `raw_<platform>/` (`--platform`, required; `--base` the folder holding the roots, default
`artifacts/comparison`) and writes, deterministically:

* `measured_summary.csv`: per (suite, model, cell, metric, and the groups `batch`, `attack`, `lam`, `op`, `stage`, `k`)
  the number of values, their median and mean, and the cell's settings (`cfg_*`) and hardware;
* `llm_full_model.csv`: the full-model cost of every language-model selection, measured where every block was built,
  otherwise extrapolated linearly from its 1- and 2-block builds (the 30-70B shapes, which the text quotes only so);
* `llm_extrapolation_check.csv`: every full build next to what the two-point rule gives from its 1- and 2-block
  builds.

It refuses a root that holds records of two machines, a median over two machines, an extrapolation across machines,
and lean and non-lean builds under one cell name. The three tables of every stored `tables_<platform>/` are
reproduced byte for byte (`tables_l40s/llm_extrapolation_validation.csv`, a line fit through every block count, was
written by a script that is not part of this artifact).

### `count_outcomes.py`

Counts, over one root or several (`--platform a,b,c`, added up; a cell finished in a later root replaces the same
cell of an earlier one), every honest query (the `accepted` records of the defence cells, batched queries included)
and every attack (the `rejected` records of the tamper cells and the one tampered query of each language-model defence
cell), with the check that rejected each (`stage`: `freivalds`, `columns_code`, `columns_merkle`, `range_or_shape`).
Only finished cells count, plus the honest rejections that stopped a job (kept as `<cell>.rejected-<run_id>.jsonl`).
`--definition` keeps only the defining defence and tamper cells of the base implementation
(`experiments.paper.bench.definition_tagsets()`), the language models' at the attention probabilities' scale
`--prob-max` (default 2047, the final run's cells tagged `pmax2047`; the stored runs' 8-bit cells: 255);
`--require-tag TAG` only the cells whose name carries a tag; `--policy` (default `cnn18c`; the stored runs used
`auto`) counts apart the attacks on the cells built under that commitment plan. It prints the counts;
`paper/numbers.py` calls its `count_runs` and `macros` on the final run's root (`\FinalNHonest`, `\FinalNAttacks`,
`\FinalNFreivalds`, ...), and `macros` refuses if an honest query was rejected or an attack accepted.

### `fingerprint_check.py`

Compares two raw roots (`stored`, `new`) on every value that is a fact about the model, the graph or the protocol and
so must not depend on the hardware: the parameter count, the int8 model's bytes and accuracy, the path test's facts,
the security bits, the bytes of the claims, of `u` and of the opened columns, and every verdict; the float model's
accuracy within `--accuracy-tolerance` (0.002: cuDNN may change its last digits). It prints every mismatch, every
honest rejection and every accepted attack of `new`, and exits 1 on any problem or when nothing could be compared.
Before a long run on new hardware it checks the new root's first cells against a stored root. Cells are compared by
name, so the final run's 11-bit decoder cells (tagged `pmax2047`) have no counterpart in the stored 8-bit roots: the
comparison then covers the image models' cells.

## Reproducing the paper's numbers

Every command of the table above reproduces the paper's numbers from the stored records on a laptop, in about a
minute each. The final run (`bash code/experiments/package/run.sh --platform <name> bench`) writes its
benchmark cells to `raw_<platform>/`; then

```bash
python -m experiments.comparison.aggregate --platform <platform>
python -m experiments.comparison.count_outcomes --platform <platform>
python -m experiments.comparison.fingerprint_check artifacts/comparison/raw_l40s artifacts/comparison/raw_<platform>
```

and `paper/numbers.py` builds the paper's tables from them (it calls `count_outcomes.count_runs` itself).

A small version runs on a laptop CPU in minutes (no downloads, no GPU; `costs/README.md` has the `bench.py` laptop
cells):

```bash
cd code
export PYTHONPATH=src:.
python -m experiments.costs.bench llm --model tiny-gpt --seq 16 --builds full --queries 2 --lams 40 \
    --llm-tampers 2 --device cpu --platform laptop
python -m experiments.comparison.aggregate --platform laptop
python -m experiments.comparison.count_outcomes --platform laptop
python -m experiments.comparison.literature --out /tmp/reported_curated.csv
```

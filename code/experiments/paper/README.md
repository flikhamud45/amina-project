# The paper's numbers

Every table body and every measured number of the paper's text comes from one run of the run package
(`experiments/package/run.sh`) on one platform. `numbers.py` reads that run's records, checks them, and writes the
body rows of each table and one LaTeX macro per text number; the paper (`paper_sp2027/main.tex`) inputs each file
from `paper_sp2027/generated/` when it exists and from `paper_sp2027/fallback/`, which holds the values the paper
prints before the final run, otherwise. Nothing here runs a model.

| Paper result | Script | Command (from `code/`) | Output |
|---|---|---|---|
| Every table body of `tab:eval-llm`, `tab:eval-v1`, `tab:options`, `tab:eval-compare`, `tab:quality`, `tab:params`, `tab:inrange`, `tab:setup`, `tab:eval-improve`, `tab:eval-long`, `tab:eval-real`, `tab:eval-reexec`, `tab:eval-compare-v1`, `tab:v1`, `tab:gen`, `tab:gen-full`, `tab:zeroshot` | `numbers.py` (`tables.py`) | `python -m experiments.paper.numbers --logs ../logs/final --platform final` | `paper_sp2027/generated/<table>.tex` |
| Every measured number of the text: the abstract's "at least N times" (`\CpuRatioN`), the outcome counts (`\FinalNHonest`, ...), the V1 ratios, the break-even crossovers, the quality, attack and sampler numbers, ... (about 300 macros) | `numbers.py` (`macros.py`) | the same | `paper_sp2027/generated/numbers.tex` |
| `fig:breakeven` (once its inputs are all in the run); `fig:overview`, `fig:protocol` with `--schematics` | `figures.py` | the same | `paper_sp2027/figures/*.pdf` |
| Every value with its source record; per table the cells filled and missing; the checks | `numbers.py` | the same (`--check`: report only) | `generated/cells.csv`, `generated/coverage.txt` |

## Modules

### `records.py`
Reads the run's folder `logs/<name>/`: every job log `sp-<cell>-<jobid>.out`, whose lines that start with `{` are the
records, and the documents of the attack tier (`attack/*.json`, `attack/analytic.csv`) and the tv tier
(`sampler_tv_*.json`). A record's tier comes from its job's prefix (`c-` cpuv, `v1-`, `serve-`, `b-` bench, `g-`
gpuv, `gen-`, `bt-` batch, `r-` real, `setup-`, `i-` improve, `q-` quality, `tv-`, `a-` attack, `l-` long, `smoke`)
and its kind from its fields (`perf`, `v1`, `reexec`, `serve`, `gen`, `setup`, `h2d`, `open_ab`, `forward_ab`,
`encode_ab`, `fold_ab`, `gpu_profile`, `v1_phases`, `gkr`, `ppl`, `zeroshot`, `digest`, `llama_diag`,
`llama_attrib`, `bench_env`, `path_bound`, `inrange`, `tv`, ...). `Run` selects the record of one cell by every
setting that distinguishes the package's jobs: the tier, the model, the prompt length, the mode, the blocks, the
verifier device (a V1 record's from its command line), the label and real or random weights; of several matching
records the latest job's. `checks` lists a record's failed correctness checks.

### `bench.py`
The benchmark's cells, from `artifacts/comparison/tables_<name>/` (`comparison/aggregate.py --platform <name>`):
`CELLS` defines each cost-table cell by mode, challenges, lambda, verifier threads and settings tags (`wire`,
`gpuv`, `stream`, `prune`, `lookups`, `thr1`, `pol<plan>`); `Bench.llm_cost` and `Bench.cnn_cost` read a cell's
prover and verifier time and its proof (full builds only, the streaming verifier's `verify_total` counted once);
`definition_tagsets(policy)` gives the defining cells to `comparison/count_outcomes.py --definition`.

### `format.py`
The paper's number formats (times, bytes, ratios, ranges, token and parameter counts, signed and scientific
numbers, the factor against a published system) and `macro_name`.

### `tables.py`
One builder per table, each in the paper's layout (the same columns and rows as the paper's table, except
`tab:eval-llm`, which loses its split into base and new columns: one column per metric, all from the final code),
and each cell recorded with its source for `cells.csv`. Every table's header comment states the run's protocol
settings and sources, so that its caption can be made true. `tab:params` and `tab:setup` follow from the library's
parameters and the setup proof's byte model (`costs/setup_bytes.py`), with V1's counts from the v1 tier; the
generation tables take the gen tier's cells (Llama-2-7B and Qwen3-4B after a 512-token prompt, greedy and sampled;
GPT-2 after a 64-token prompt, greedy), selected by the prompt's length as well.

### `macros.py`
The text numbers, by family (the platform and the plan, the CPU client, the outcome counts, the cost of the defence,
batches against one opening, serving, V1, the comparison, the improvements, real weights, long prompts, quality, the
sampler, the attack, the security parameters, the setup proof, generation, the break-even). The platform's macros
(`\PlatformGpu`, `\PlatformCpu`, `\PlanLogPoints`) come from the records' `env` and `--policy`, so the captions and the
sentences that say how a number was measured state the run's settings. Values that follow from the code alone are
computed from the library under `--policy`: the setup proof's size, breakdown and radii (`costs/setup_bytes.py`'s byte
model), the split commitment's columns per query, and V1's cut of the largest model from its shapes
(`tables.v1_shape_model`; the leaves, instances and soundness terms at the run's `lmax`). The abstract's N
(`\CpuRatioN`) is one value: exact re-execution (the integer forward pass, bit-exact with the claims: the
faster of `int` and `int_fp32gemm`, computing only what the pruned graph computes) over the Kpre CPU
verifier, both on 8 threads, for the full Llama-2-7B at 2,048 tokens (cpuv tier), rounded down to one decimal. The
break-even reuses `deployment/ev_breakeven.py`'s formulas on the run's claims, verifier and re-execution times and
its byte model. The `Todo...` macros carry the `\todo` markers of numbers that the run fills (empty once generated).

### `figures.py`
`fig:overview` and `fig:protocol` (schematics, from code; laid out for the pinned matplotlib of `requirements.txt`,
with layout checks) and `fig:breakeven` (from the break-even's crossovers, drawn by `ev_breakeven.draw`).

### `numbers.py`
The command: read, check, build, write. Options: `--logs` (default `../logs/<platform>`), `--platform` (default
`final`), `--bench` (the folder of `tables_<name>/` and `raw_<name>/`, default `artifacts/comparison`), `--policy`
(mode C's plan, default `cnn18c`), `--out` (default `paper_sp2027/generated`), `--figures` (default
`paper_sp2027/figures`; `''` for none) and `--schematics`, `--expect <dry-run listing>`, `--check`, `--strict`,
`--rehearsal`, `--no-byte-model` (skips `tab:params`, `tab:setup` and the break-even, which take several minutes on a
laptop CPU), `--verbose`.

## The checks

* `CHECK FAILED`: an honest query rejected or a tamper accepted (`perf.py`, `v1_bytes.py`, `gen_bench.py`), a byte
  accounting that disagrees with the model (`v1_bytes.py`'s `accounting_ok`), a serving check that failed
  (`serve_ab.py`'s `ok`), a setup proof rejected (`setup_timing.py`), the two paths of an implementation A/B that
  disagree, a smoke log without `SMOKE OK` or with int8 GEMMs that are not exact, a `FAILED` line of a script,
  records of more than one GPU or CPU model, an honest query rejected or an attack accepted in the benchmark's
  outcome counts.
* `PARSE`: a log line that starts with `{` and does not parse, or is not a record of a known kind; a table or macro
  family that cannot read a record.
* `--strict`: also any table cell, macro or (with `--expect`) job without results.
* `--rehearsal`: every tier has at least one record of each kind it writes (`records.REHEARSAL_KINDS`), and no table
  cell is expected.

The exit status is 1 on any of these. `coverage.txt` lists them, the records per tier and kind, and per table the
cells filled and missing (`--verbose`: each missing cell and macro).

## The fallback

`paper_sp2027/main.tex` resolves, in its preamble, each of `numbers` and the tables of `tables.TABLE_FILES` to
`generated/<name>.tex` if that file exists and to `fallback/<name>.tex` otherwise, and `\pvitable{<name>}` reads it
with TeX's primitive `\input` (expandable, so a table body may start with `\multicolumn` and the paper's
`\bottomrule` may follow it). Each table of the paper reads its body rows with `\pvitable{<table>}` between its
`\midrule` and `\bottomrule`, and the text writes each number as its macro (`\CpuRatioN{}`). `fallback/` holds each
table's body rows as the paper printed them and each macro's printed value, `\todo` markers included, so the paper
reads as before until the run's files are generated; `numbers.py` writes the same file names and the same macros.
A number that the run lacks prints as a visible `\todo`.

## Reproducing the paper's numbers

After the run (`experiments/package/README.md`, Sec. 7):

```bash
bash code/experiments/package/run.sh --platform final --dry-run --optional --llama-licence all \
    > logs/final/dryrun.txt                                               # every job of the run (repository root)
cd code
python -m experiments.comparison.aggregate --platform final              # raw_final/ -> tables_final/
python -m experiments.paper.numbers --logs ../logs/final --platform final --check --expect ../logs/final/dryrun.txt
python -m experiments.paper.numbers --logs ../logs/final --platform final  # -> paper_sp2027/generated/
cd ../paper_sp2027 && latexmk -pdf main.tex && latexmk -pdf main_ext.tex  # or: tectonic main.tex
```

`--strict` makes the run's coverage a gate: exit status 1 while any cell, macro or job is missing. A rehearsal
(`run.sh --rehearsal`) is checked with `--rehearsal --check`. Removing `paper_sp2027/generated/` returns the paper to
its fallback values.

On a laptop (no run): `tests/test_paper_numbers.py` writes synthetic records of every kind, as the scripts write
them, and checks the selection of each table and macro, the checks and the files; the attack tier's stored
documents (`artifacts/results/`) regenerate the fallback's `tab:inrange` and its text numbers:

```bash
cd code
PYTHONPATH="src;." python -m pytest -q tests/test_paper_numbers.py tests/test_comparison_tables.py
```

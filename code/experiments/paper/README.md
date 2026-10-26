# The paper's numbers

Every table body and every measured number of the paper's text comes from one run of the run package
(`experiments/package/run.sh`) on one platform. `numbers.py` reads that run's records, checks them, and writes the
body rows of each table and one LaTeX macro per text number; the paper (`paper_sp2027/main.tex`) inputs each file
from `paper_sp2027/generated/` when it exists and from `paper_sp2027/fallback/`, which holds the values the paper
prints before the final run, otherwise. Nothing here runs a model.

| Paper result | Script | Command (from `code/`) | Output |
|---|---|---|---|
| Every table body of `tab:eval-llm`, `tab:eval-v1`, `tab:options`, `tab:eval-compare`, `tab:quality`, `tab:params`, `tab:inrange`, `tab:setup`, `tab:eval-improve`, `tab:memory`, `tab:eval-long`, `tab:eval-real`, `tab:eval-reexec`, `tab:eval-compare-v1`, `tab:v1`, `tab:gen`, `tab:gen-full`, `tab:quality-attr`, `tab:zeroshot`, `tab:paths`, `tab:paths-acc` | `numbers.py` (`tables.py`) | `PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final` | `paper_sp2027/generated/<table>.tex` |
| Every measured number of the text: the abstract's "at least N times" (`\CpuRatioN`), the outcome counts (`\FinalNHonest`, ...), the V1 ratios, the break-even crossovers, the client's own memory, the quality, attribution, attack and sampler numbers, the spot-checking schemes' acceptance (`\Sok...`), ... (about 490 macros) | `numbers.py` (`macros.py`) | the same | `paper_sp2027/generated/numbers.tex` |
| `fig:breakeven` (once its inputs are all in the run); `fig:overview`, `fig:protocol` with `--schematics` | `figures.py` | the same | `paper_sp2027/figures/*.pdf` |
| Every value with its source record; per table the cells filled and missing; the checks | `numbers.py` | the same (`--check`: report only) | `generated/cells.csv`, `generated/coverage.txt` |

## Modules

### `records.py`
Reads the run's folder `logs/<name>/`: every job log `sp-<cell>-<jobid>.out`, whose lines that start with `{` are the
records, and the documents of the attack tier (`attack/*.json`, `attack/analytic.csv`) and the tv tier
(`sampler_tv_*.json`). A record's tier comes from its job's prefix (`c-` cpuv, `v1-`, `serve-`, `b-` bench, `g-`
gpuv, `gen-`, `bt-` batch, `r-` real, `setup-`, `i-` improve, `q-` quality, `tv-`, `a-` attack, `l-` long, `smoke`;
`gate-` gate, the full-size checks run before the run, whose records are checked and fill no cell or number)
and its kind from its fields (`perf`, `v1`, `reexec`, `llamacpp`, `serve`, `gen`, `setup`, `h2d`, `open_ab`,
`forward_ab`, `encode_ab`, `fold_ab`, `gpu_profile`, `v1_phases`, `gkr`, `ppl`, `zeroshot`, `digest`, `llama_diag`,
`llama_attrib`, `attribution`, `bench_env`, `path_bound`, `inrange`, `tv`, ...; in the smoke's log, pytest's
summary line is a record of kind `tests`). `Run` selects the record of one cell
by every
setting that distinguishes the package's jobs: the tier, the model, the prompt length, the mode, the blocks, the
verifier device (a V1 record's from its command line), the label and real or random weights; of several matching
records the latest job's (for re-execution, `reexec.py`'s and `llamacpp.py`'s, the fastest of one variant's
records). `checks` lists a record's failed correctness checks (a llama.cpp record must be marked inexact and a
different model, and have run on its threads).

### `bench.py`
The benchmark's cells, from `artifacts/comparison/tables_<name>/` (`comparison/aggregate.py --platform <name>`):
`CELLS` defines each cost-table cell by mode, challenges, lambda, verifier threads and settings tags (`wire`,
`gpuv`, `stream`, `prune`, `lookups`, `thr1`, `pol<plan>`, and the decoders' `pmax<n>`: `pmax2047`, the 11-bit
attention probabilities of the final run, `DEFAULT_PROB_MAX`; none for the stored runs' 8-bit cells);
`Bench.llm_cost` and `Bench.cnn_cost` read a cell's prover and verifier time and its proof (full builds only, the
streaming verifier's `verify_total` counted once); `definition_tagsets(policy, prob_max)` gives the defining cells to
`comparison/count_outcomes.py --definition`.

### `format.py`
The paper's number formats (times, bytes, ratios, ranges, token and parameter counts, signed and scientific
numbers, the factor against a published system) and `macro_name`.

### `tables.py`
One builder per table, each in the paper's layout (the same columns and rows as the paper's table), and each cell
recorded with its source for `cells.csv`. Every table's header comment states the run's protocol settings and
sources, so that its caption can be made true. Exact re-execution (`tab:options`, `tab:eval-reexec`,
`tab:eval-long`) is the fastest of `EXACT`, the integer forward pass with its native int8 GEMMs (`int`), exact fp32
GEMMs (`int_fp32gemm`) or `torch._int_mm` (`int_mm`, counted only when its record's `exact` is true: its logits
equal the integer reference's bit for bit in the same run). The inexact engines (`INEXACT`: fp32, bf16 and dynamic
int8 decoders of the same shapes, and llama.cpp's Q8_0 and bf16 prefill, `LLAMACPP`, each at its fastest
configuration) are reported next to it (`tab:options`' llama.cpp row, `tab:eval-reexec`), never in its place.
`tab:params` and `tab:setup` follow from the library's parameters and the setup proof's byte model
(`costs/setup_bytes.py`; the per-query columns of interactive queries, as the setup tier measures them), with V1's
counts from the v1 tier; `tab:memory` takes the cpuv tier's Llama-2-7B cells: the client's own process (`perf.py
--client-memory`'s `client_memory`), the prover's GPU peak and the measuring process's peak; `tab:eval-compare-v1`
has a row per full model of the v1 tier; the generation tables take the gen tier's cells (Llama-2-7B and Qwen3-4B
after a 512-token prompt, greedy and sampled; GPT-2 after a 64-token prompt, greedy), selected by the prompt's length
as well. `tab:paths` and `tab:paths-acc` evaluate Proposition `prop:paths` (`attack/sok.py`: the paths that catch an
edit of `k` neurons, and the probability that `s` paths accept it) at one neuron, at 16 and at the in-range edits of
the attack tier's OPT-1.3B documents (`inrange_ks`: the median and largest edit of either direction and the median
lowering-only edit, which the text prints as `\InrangeKMedian`, `\InrangeKMax`, `\InrangeDownKMedian` and the tables'
headers name).

### `macros.py`
The text numbers, by family (the platform and the plan, the CPU client and its own memory, the outcome counts, the
cost of the defence, batches against one opening, serving, V1, the comparison, the improvements, real weights, long
prompts, quality, the sampler, the attack, the smoke's tests, the security parameters, the setup proof, generation, the break-even). The platform's macros
(`\PlatformGpu`, `\PlatformCpu`, `\PlanLogPoints`) come from the records' `env` and `--policy`, so the captions and the
sentences that say how a number was measured state the run's settings. Values that follow from the code alone are
computed from the library under `--policy`: the setup proof's size, breakdown and radii (`costs/setup_bytes.py`'s byte
model), the split commitment's columns per query, V1's cut of the largest model from its shapes
(`tables.v1_shape_model`; the leaves, instances and soundness terms at the run's `lmax`), and what a Kpre client of
Llama-2-7B stores (`\KpreKeyMB`, `\KpreEmbedMB`, `\KpreStoreGB`: the secret vectors and folded rows at the
Fiat-Shamir repetition count, and the int8 embedding table). The abstract's N (`\CpuRatioN`, and `\CpuRatio` in the
body) is one value: exact re-execution (the integer forward pass, bit-exact with the claims: the fastest of
`tables.EXACT`, computing only what the pruned graph computes) over the Kpre CPU verifier, both on 8 threads, for
the full Llama-2-7B at 2,048 tokens (cpuv tier), rounded down to one decimal; llama.cpp, which computes another
model, has its own ratio (`\CpuRatioLlamaCpp`) and never changes N. V1's headline range
(`\VOneRatioRandom`) takes the full models of the v1 tier at 2,048 tokens. `\RealAcceptedClause` names the full
real-weight models whose every honest Kpre query at 2,048 tokens the real tier accepted. The break-even reuses
`deployment/ev_breakeven.py`'s formulas on the run's claims, verifier and re-execution times and its byte model.
`\OptionsRuns` and `\OptionsSpread` are the timed runs of `tab:options`' cells and their largest relative min--max
spread; `\Outcome...` the tampered queries of the real-weight cells, their random-weight twins and the generation
cells (all, V1's, the generated responses') and the setup tier's honest setup proofs, printed only when every tamper was rejected and every setup
proof accepted. `\ZkllmSameGpu...` set zkLLM's stored runs on the run's GPU model (`artifacts/results/zkllm_l40s/`,
read by `bench.literature` with the published values: the proof binaries of two decoder layers, times the layers)
against our interactive mode-C prover on prompts of the same length, only when the stored runs' GPU is the run's.
`\SmokeTests` is the number of tests that passed in the smoke's pytest run on the run's platform (the summary line
of its log). The spot-checking schemes' acceptance probabilities and path counts of `tab:sok` and `tab:sok-derivation` (`\Sok...`)
follow from the formulas and parameters those tables state (`attack/sok.py`; no run). The bounds over the in-range
cells skip a layer and range without a successful edit. A number the text states as a bound ("at most", "up to",
"within", "less than") is rounded up, one stated as "at least" down. The `Todo...` macros carry the `\todo` markers
of numbers that the run fills (empty once generated).

### `figures.py`
`fig:overview` and `fig:protocol` (schematics, from code; laid out for the pinned matplotlib of `requirements.txt`,
with layout checks) and `fig:breakeven` (from the break-even's crossovers, drawn by `ev_breakeven.draw`).

### `numbers.py`
The command: read, check, build, write. Options: `--logs` (default `../logs/<platform>`), `--platform` (default
`final`), `--bench` (the folder of `tables_<name>/` and `raw_<name>/`, default `artifacts/comparison`), `--policy`
(mode C's plan; default the plan the records name, a mode-C record's `policy` or the benchmark tables' `pol<plan>`,
else `bench.DEFAULT_POLICY`, `cnn18c`), `--prob-max` (the decoders' attention probabilities' scale of the benchmark
cells, default 2047; 255 reads the stored runs' 8-bit cells), `--out` (default `paper_sp2027/generated`),
`--figures` (default `paper_sp2027/figures`; `''` for none) and `--schematics`, `--expect <dry-run listing>`, `--check`, `--strict`,
`--allow-missing`, `--rehearsal`, `--no-byte-model` (skips `tab:params`, `tab:setup` and the break-even, which take several minutes on a
laptop CPU), `--verbose`. `SAME_VALUE` pairs each table cell with the macro that prints the same value of the same
record, `TWO_JOBS` each quantity that two jobs measure (the serving harness's prover and the benchmark's, the
benchmark's CPU verifier and perf.py's, v0's timed queries of the V1 job and perf.py's).

## The checks

* `CHECK FAILED`: an honest query rejected or a tamper accepted (`perf.py`, `v1_bytes.py`, `gen_bench.py`), a
  client process whose verdicts differ from the in-process ones (`perf.py`'s `client_memory`), a byte
  accounting that disagrees with the model (`v1_bytes.py`'s `accounting_ok`), a serving check that failed
  (`serve_ab.py`'s `ok`), a setup proof rejected (`setup_timing.py`), the two paths of an implementation A/B that
  disagree, an attribution whose float copy failed a check (`attribution.py`'s `ok`) or whose integer graph is not
  the quality tier's cell of the same graph (`scalar-g4-p8` at 128 tokens: another digest, or another perplexity
  over the same windows), a smoke log without `SMOKE OK`, with int8 GEMMs that are not exact or with a failed test
  (its pytest summary line, the `tests` record), a `FAILED` line of a
  script, records of more than one GPU or CPU model, an honest query rejected or an attack accepted in the
  benchmark's outcome counts, builds of one certified model (`digest_check.py`, the quality tier's reproducible
  build) whose digests, weights or commitment roots differ (`macros.digest_agreement`), a table cell that disagrees
  with the macro printing the same value (`SAME_VALUE`, beyond one unit of the coarser rounding), in the files
  written and in `paper_sp2027/fallback/` (`CHECK FAILED fallback:`).
* `CHECK FAILED text:`: a value that does not bear out what the text states about it (`macros.TEXT_CLAIM`): a
  zero-shot difference the text calls a loss that is not negative (`macros.LOSSES`), the attribution's ranking (the
  LayerNorm outputs dominate and remove more than half of the gap, the weight matrices and fc1's requantisation
  follow, rounding and not clipping; `macros.attribution_claims`), a path count of the attack's documents or the
  benchmark that differs from the formula's, which `tab:sok` prints as the same number (`macros.FORMULA`), a record
  of a cost measurement (`macros.GRAPH_KINDS`, outside the quality tier) whose decoder is not at the 11-bit attention
  probabilities the text states (`prob_max` 2047), but for the 8-bit side of the improve tier's A/B
  (`macros.PROB_BITS`, label `prob-bits`; `\ProbBitsBytes`, `\ProbBitsTime`), a Llama-family ratio whose reference
  is not the one the text names (fp32; fp16 for Llama-2-13B), validation cells of OpenLLaMA-7B outside the grid the
  text names (`macros.LLAMA_VALIDATION`). A rehearsal reports these as warnings.
* `PARSE`: a log line that starts with `{` and does not parse, or is not a record of a known kind; a table or macro
  family that cannot read a record; an `--expect` listing that does not end with run.sh's summary line
  (`# <tiers>: N jobs, ...`) for the N jobs it lists (a dry run that stopped early, e.g. at a tier's check of
  `HF_HOME`).
* `MISSING:`: the report ends with every table cell and every macro that the run does not provide, one per line
  (`COMPLETE:` when there is none); any of them sets the exit status to 1, unless `--allow-missing` (a partial run,
  such as the gate tier's before the run).
* `--strict`: also any job of `--expect` without results.
* `--rehearsal`: every tier has at least one record of each kind it writes (`records.REHEARSAL_KINDS`), and no table
  cell is expected (the `MISSING:` line counts them and leaves the exit status unchanged); the `CHECK FAILED text:` lines become warnings (a rehearsal's small cells need not bear out what
  the text states about the final values).

The exit status is 1 on any of these. `WARNING` lines leave it unchanged: two jobs' values of one quantity more than 10%
apart (`TWO_JOBS`), two records of one quality cell (two jobs that measure it) whose perplexity or ratio differ
(`records.Run.ppl`; the later is used), records that name another mode-C plan than `--policy` (or several), an `int_mm`
re-execution that is not recorded as bit-exact (it is then no exact baseline). `coverage.txt` lists them all, the
records per tier and kind, and per table the cells filled and missing (`--verbose`: each missing cell and macro).

## The fallback

`paper_sp2027/main.tex` resolves, in its preamble, each of `numbers` and the tables of `tables.TABLE_FILES` to
`generated/<name>.tex` if that file exists and to `fallback/<name>.tex` otherwise, and `\pvitable{<name>}` reads it
with TeX's primitive `\input` (expandable, so a table body may start with `\multicolumn` and the paper's
`\bottomrule` may follow it). Each table of the paper reads its body rows with `\pvitable{<table>}` between its
`\midrule` and `\bottomrule`, and the text writes each number as its macro (`\CpuRatioN{}`). `fallback/` holds each
table's body rows and each macro's printed value before the final run, `\todo` markers included; `numbers.py`
writes the same file names and the same macros. A number that the run lacks prints as a visible `\todo`.

The fallback follows the definitions of the final run: exact re-execution is the fastest exact variant of the
integer forward pass, the Kpre CPU times are under Fiat-Shamir, the integer OPT models are the certified recipe's.
Its values come from the stored records (`artifacts/results/`, the benchmark tables
`artifacts/comparison/tables_l40s_improved*/` under the plan `auto`) and from the development runs that measured
the run package's cells on an RTX 2080 Ti and 8 threads of a Xeon Silver 4114, in the format the builders print; a
value whose setting differs from the one the paper states (an interactive cell where the text says Fiat-Shamir, the
legacy recipe's sampler distances) prints as `\todo{<tier>}`. `figures/ev_breakeven.pdf` and the fallback's
`\Breakeven...` macros follow from the same inputs, `deployment/ev_inputs.json`, through
`deployment/ev_breakeven.py`'s formulas and drawing (the drawing that `figures.breakeven` uses for the run). The test
`test_the_fallback_tables_agree_with_their_macros` (and `numbers.py` on every run) checks that each fallback cell
agrees with the macro that prints the same value.

## Reproducing the paper's numbers

After the run (`experiments/package/README.md`, Sec. 7):

```bash
# from the repository root, with HF_HOME exported as for the run (package README, Sec. 2)
bash code/experiments/package/run.sh --platform final --dry-run --optional --llama-licence all \
    > logs/final/dryrun.txt                       # every job of the run; must end with '# <tiers>: N jobs, ...'
cd code
export PYTHONPATH=src:.                           # the library (src) and the experiments
../.venv/bin/python -m experiments.comparison.aggregate --platform final             # raw_final/ -> tables_final/
../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final --check \
    --expect ../logs/final/dryrun.txt
../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final  # -> paper_sp2027/generated/
cd ../paper_sp2027 && latexmk -pdf main.tex && latexmk -pdf main_ext.tex  # or: tectonic main.tex
```

The report's last lines (`MISSING:`, one line per cell or macro the run lacks, or `COMPLETE:`) show the run's
completeness, and any missing cell or macro gives exit status 1 (`--allow-missing`: a partial run); `--strict` also
fails on a job of `--expect` without results. A rehearsal
(`run.sh --rehearsal`) is checked with `--rehearsal --check`. Removing `paper_sp2027/generated/` returns the paper to
its fallback values.

On a laptop (no run): `tests/test_paper_numbers.py` writes synthetic records of every kind, as the scripts write
them, and checks the selection of each table and macro, the checks and the files; the attack tier's stored
documents (`artifacts/results/`) regenerate the fallback's `tab:inrange` and its text numbers, and every fallback
cell agrees with its macro. `tests/test_paper_numbers_derived.py` checks, on the same synthetic run, the numbers
that follow from formulas, stored literature runs and summaries of the records: `tab:paths` and `tab:paths-acc` (the
stored documents give the fallback; other edit sizes move their columns), the `\Sok...` values against the formulas
and parameters the paper states, zkLLM's same-GPU factors, the tampered queries and setup proofs, V1's verifier with
GPU tensors, the timed runs of `tab:options`, and an in-range layer without a successful edit. `PVI_SLOW=1` adds the
check that the library's setup proof gives the fallback's setup-proof values
(`test_the_setup_proof_from_the_library_gives_the_fallback`, about ten minutes on a laptop CPU); without it that test
is skipped:

```bash
cd code
PYTHONPATH="src;." python -m pytest -q tests/test_paper_numbers.py tests/test_paper_numbers_derived.py \
    tests/test_comparison_tables.py                                                             # Windows
PVI_SLOW=1 PYTHONPATH=src:. ../.venv/bin/python -m pytest -q tests/test_paper_numbers.py      # the slow check too
```

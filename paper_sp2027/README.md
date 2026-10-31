# The paper

| File | Contents |
|---|---|
| `main.tex` | the submission (IEEE S&P, `IEEEtran`): the body and a short appendix |
| `main_ext.tex` | the extended version: the same body with every proof and the material moved out of the body |
| `sections/*.tex` | the body sections, input by both; `appendix_short` (submission only), the other `appendix_*` and `ext_*` (extended version only), `app_params` (both: directly in the extended version, through `appendix_short` in the submission) |
| `references.bib`, `extra.bib` | the bibliography |
| `figures/` | the figures; `ev_breakeven.pdf` is drawn from the run's records, the schematics from code (`code/experiments/paper/figures.py`) |
| `fallback/` | every table body and every measured number of the text as printed before the final run |
| `generated/` | the same files, written from the final run's records by `code/experiments/paper/numbers.py` (absent until then) |

Every measured number of the paper is a macro (`\CpuRatioN{}`, ...) and every table body is read with
`\pvitable{<table>}`. Both come from `generated/<name>.tex` when the file exists and from `fallback/<name>.tex`
otherwise; a number that the run lacks prints as a visible `[TODO: ...]`. Removing `generated/` returns the paper to
the fallback values.
In `fallback/`, every value that the final run replaces is wrapped in `
otready{...}` and prints in orange; values that need no run (published numbers, labels, values computed by formula) stay black, and a value that no run has measured yet prints as a red `[TODO: ...]`. The marked values are those that `numbers.py` leaves missing when run on a run folder without records; `numbers.unmark` removes the marks wherever the code reads the fallback, and `generated/` never has them.

**Choosing the rows of `tab:eval-compare`.** The comparison with published systems is computed for every row of
`COMPARE_ROWS` in `code/experiments/paper/tables.py` whichever rows the body prints; the extended version's
`tab:eval-compare-full` prints them all, and `generated/cells.csv` holds every cell with its source. The body's rows
are chosen without editing code:

```bash
cd code
PYTHONPATH=src:. ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final \
    --compare-rows zkcnn-vgg16,zkgpt-gpt2-64,deepprove-gpt2-512,zktorch-llama2-7b-1,maverick-qwen3-4b-8,zkllm-llama2-13b
```

`--compare-rows` takes `default` (the eleven rows printed now), `all`, a comma-separated list of row keys or a file
with one key per line; `numbers.py --help` lists the keys. The text's comparison numbers (`\CompareProverMin`,
`\CompareProverMax`, `\CompareProofMax`, `\CompareVerifierSlowerRows`, `\CompareMaverick`) follow the chosen rows.
A cut row whose footnote mark ($^a$--$^d$) no other row carries leaves its note in the caption, which is then
shortened by hand, as is the sentence on LeNet-5 in the extended version.

## Build

```bash
cd paper_sp2027
latexmk -pdf main.tex && latexmk -pdf main_ext.tex      # or: tectonic main.tex && tectonic main_ext.tex
```

To build from a run: write `generated/` first (`code/experiments/paper/README.md`), then build as above.

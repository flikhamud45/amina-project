# The paper

| File | Contents |
|---|---|
| `main.tex` | the submission (IEEE S&P, `IEEEtran`): the body and a short appendix |
| `main_ext.tex` | the extended version: the same body with every proof and the material moved out of the body |
| `sections/*.tex` | the sections, input by `main.tex` (the `ext_*` and `appendix_*` files by the extended version) |
| `references.bib`, `extra.bib` | the bibliography |
| `figures/` | the figures; `ev_breakeven.pdf` is drawn from the run's records, the schematics from code (`code/experiments/paper/figures.py`) |
| `fallback/` | every table body and every measured number of the text as printed before the final run |
| `generated/` | the same files, written from the final run's records by `code/experiments/paper/numbers.py` (absent until then) |

Every measured number of the paper is a macro (`\CpuRatioN{}`, ...) and every table body is read with
`\pvitable{<table>}`. Both come from `generated/<name>.tex` when the file exists and from `fallback/<name>.tex`
otherwise; a number that the run lacks prints as a visible `[TODO: ...]`. Removing `generated/` returns the paper to
the fallback values.

## Build

```bash
cd paper_sp2027
latexmk -pdf main.tex && latexmk -pdf main_ext.tex      # or: tectonic main.tex && tectonic main_ext.tex
```

To build from a run: write `generated/` first (`code/experiments/paper/README.md`), then build as above.

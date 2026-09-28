# Report

`main.pdf` is the report; `main.tex` its source (ACM `sigconf`, double column, at most 8
pages).

`figures/*.pdf` and `tables/*.tex` are generated from the stored measurements by
`code/experiments/5_comparison/paper_assets.py` (no GPU needed). The report's benchmark
numbers are platform `rtx2080ti-v2` (`code/artifacts/comparison/raw_rtx2080ti-v2/`, the
RTX 2080 Ti prover and 8 threads of a Xeon Silver 4114 verifier; see `code/README.md`):

```bash
cd code && export PYTHONPATH=$PWD/src
python experiments/5_comparison/aggregate.py --platform rtx2080ti-v2      # raw_rtx2080ti-v2/ -> tables_rtx2080ti-v2/
python experiments/5_comparison/paper_assets.py --platform rtx2080ti-v2   # -> ../report/figures, ../report/tables
cd ../report && latexmk -pdf main.tex                                     # or: tectonic -X compile main.tex
```

`tables/hardware.tex` defines `\ProverGPU`, `\VerifierCPU`, `\VerifierThreads` and
`\LLMNote` (the note on Table 3's extrapolated row, Llama-2-13B), which `main.tex` inputs
in its preamble. The numbers typed in the text are not generated:
`python experiments/5_comparison/text_numbers.py --platform rtx2080ti-v2` (from `code/`)
prints each one recomputed next to its line, and
`python experiments/5_comparison/count_outcomes.py --platform rtx2080ti-v2` (from `code/`)
the counts of §4.3, the abstract and the conclusion, and the record total of §4.1. The
figures depend slightly on the matplotlib version: the committed ones were made with the
pinned matplotlib 3.11.2 of `code/requirements.txt`, which reproduces them exactly. With
another version they come out slightly larger or smaller and the PDF can grow to 9 pages,
so after regenerating them check that it still has 8.

`acmart.cls` and `ACM-Reference-Format.bst` are the course template's copies (acmart
2.18), so the build does not depend on the installed TeX version. `references.bib`
holds the bibliography; every author list was checked against the paper's own page.

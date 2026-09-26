# Report

`main.pdf` is the report; `main.tex` its source (ACM `sigconf`, double column).

`figures/*.pdf` and `tables/*.tex` are generated from the stored measurements by
`code/experiments/5_comparison/paper_assets.py` (no GPU needed):

```bash
python code/experiments/5_comparison/paper_assets.py     # from the repository root
cd report && latexmk -pdf main.tex                        # or: tectonic -X compile main.tex
```

`acmart.cls` and `ACM-Reference-Format.bst` are the course template's copies (acmart
2.18), so the build does not depend on the installed TeX version. `references.bib`
holds the bibliography; every author list was checked against the paper's own page.

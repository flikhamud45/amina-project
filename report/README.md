# Report

`main.tex` is the final report (ACM `sigconf` double-column); `main.pdf` is the built PDF.

Every table body (`tables/*.tex`) and figure (`figures/*.pdf`) is generated from the
stored benchmark tables in `code/artifacts/comparison/tables/`, so nothing needs a GPU:

```bash
python report/make_assets.py          # from the repository root
cd report && tectonic -X compile main.tex   # or: latexmk -pdf main.tex
```

`acmart.cls` and `ACM-Reference-Format.bst` are the course template's copies (acmart
2.18), included so the build does not depend on the TeX installation's version.

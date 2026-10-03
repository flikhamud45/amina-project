# Report

`main.pdf` is the paper; `main.tex` its source (ACM `sigconf`, double column; nine pages, the text on
pages 1–8 and the references on page 9).

`figures/*.pdf` and `tables/*.tex` are generated from the stored measurements in `code/artifacts/`
by `code/experiments/5_comparison/paper_assets.py` and `count_outcomes.py` (no GPU needed; see
`code/README.md` and `code/experiments/5_comparison/README.md`). The paper's benchmark numbers come
from runs on the same machines (an NVIDIA L40S prover and 8 threads of an AMD EPYC 9334 verifier): the
*optimised protocol*, platforms `l40s_improved` and `l40s_improved2` (Tables 1–3, Fig. 4, the filled
points of Fig. 3), and the *basic protocol*, platform `l40s` (the basic side of Sec. 4.5 and the hollow
points of Fig. 3). From `code/`:

```bash
export PYTHONPATH=$PWD/src
python experiments/5_comparison/aggregate.py --platform l40s              # raw_l40s/ -> tables_l40s/
python experiments/5_comparison/aggregate.py --platform l40s_improved     # raw_l40s_improved/ -> tables_l40s_improved/
python experiments/5_comparison/aggregate.py --platform l40s_improved2    # raw_l40s_improved2/ -> tables_l40s_improved2/
python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved,l40s_improved2   # -> ../report/figures, ../report/tables
python experiments/5_comparison/count_outcomes.py --platform l40s --tex ../report/tables/counts.tex
python experiments/5_comparison/count_outcomes.py --platform l40s_improved,l40s_improved2 --prefix Opt --definition \
    --tex ../report/tables/counts_opt.tex
python experiments/5_comparison/text_numbers.py --platform l40s --optimised l40s_improved,l40s_improved2 --definition
cd ../report && latexmk -pdf main.tex                                     # or: tectonic -X compile main.tex
```

| File | Contents |
|---|---|
| `figures/overview_*.pdf`, `figures/protocol.pdf` | Fig. 1 and Fig. 2 |
| `figures/security_bits.pdf` | Fig. 3 |
| `figures/cost_*.pdf` | Fig. 4 (three panels and the legend) |
| `tables/cnn.tex`, `tables/llm.tex`, `tables/ratios.tex` | Tables 1–3 |
| `tables/hardware.tex` | `\ProverGPU`, `\VerifierCPU`, `\VerifierThreads`, `\LLMNote` (empty: every row of Table 2 is a full build) and `\NPendingCells` (the number of table and figure values still pending) |
| `tables/counts.tex`, `tables/counts_opt.tex` | the soundness counts of the abstract and Sec. 4.3 as macros, for the basic and the optimised protocol |

`main.tex` inputs `hardware.tex` and the count files in its preamble. The numbers typed in the text are
not generated: `text_numbers.py` compares each one with the tables and fails on a mismatch. Values that
are not final are printed in red: some timings of the second optimised batch are withheld
(`code/artifacts/comparison/excluded_cells.csv`; two nodes were slowed by other users' load during
parts of that batch) and two optional cells were not run. `\pending{}` marks such a value, shown with
the basic protocol's value as a placeholder (or `--`), and `\pendingclaim{}` a claim that depends on
one; `paper_assets.py --check` lists them. The real-LLM results and zkLLM's run on our GPU are in
`code/artifacts/results/`.

Use matplotlib 3.11.2 (`code/requirements.txt`) for the figures: it reproduces the committed PDFs byte
for byte. With another version they come out slightly larger or smaller, so after regenerating them
check that the text still ends on page 8.

`acmart.cls` and `ACM-Reference-Format.bst` are the course template's copies (acmart 2.18), so the
build does not depend on the installed TeX version. `references.bib` holds the bibliography; every
author list was checked against the paper's own page.

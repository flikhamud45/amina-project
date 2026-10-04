# Report

`main.pdf` is the paper; `main.tex` its source (ACM `sigconf`, double column; nine pages, the text on
pages 1–8 and the references on page 9).

`figures/*.pdf` and `tables/*.tex` are generated from the stored measurements in `code/artifacts/`
by `code/experiments/5_comparison/paper_assets.py` and `count_outcomes.py` (no GPU needed; see
`code/README.md` and `code/experiments/5_comparison/README.md`). The paper's benchmark numbers come
from runs on the same machines (an NVIDIA L40S prover and 8 threads of an AMD EPYC 9334 verifier): the
*optimised protocol*, platforms `l40s_improved`, `l40s_improved2` and `l40s_improved3` (the
re-measurement of the second's withheld cells; Tables 1–3, Fig. 4, the filled
points of Fig. 3), and the *basic protocol*, platform `l40s` (the basic side of Sec. 4.5 and the hollow
points of Fig. 3). From `code/`:

```bash
export PYTHONPATH=$PWD/src
python experiments/5_comparison/aggregate.py --platform l40s              # raw_l40s/ -> tables_l40s/
python experiments/5_comparison/aggregate.py --platform l40s_improved     # raw_l40s_improved/ -> tables_l40s_improved/
python experiments/5_comparison/aggregate.py --platform l40s_improved2    # raw_l40s_improved2/ -> tables_l40s_improved2/
python experiments/5_comparison/aggregate.py --platform l40s_improved3    # raw_l40s_improved3/ -> tables_l40s_improved3/
python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved,l40s_improved2,l40s_improved3   # -> ../report/figures, ../report/tables
python experiments/5_comparison/count_outcomes.py --platform l40s --tex ../report/tables/counts.tex
python experiments/5_comparison/count_outcomes.py --platform l40s_improved,l40s_improved2,l40s_improved3 --prefix Opt --definition \
    --tex ../report/tables/counts_opt.tex
python experiments/5_comparison/text_numbers.py --platform l40s --optimised l40s_improved,l40s_improved2,l40s_improved3 --definition
cd ../report && latexmk -pdf main.tex                                     # or: tectonic -X compile main.tex
```

| File | Contents |
|---|---|
| `figures/overview_*.pdf`, `figures/protocol.pdf` | Fig. 1 and Fig. 2 |
| `figures/security_bits.pdf` | Fig. 3 |
| `figures/cost_*.pdf` | Fig. 4 (three panels and the legend) |
| `tables/ratios.tex`, `tables/cnn.tex`, `tables/llm.tex` | Tables 1, 2 and 3 |
| `tables/hardware.tex` | `\ProverGPU`, `\VerifierCPU`, `\VerifierThreads` and `\LLMNote` (empty: every row of Table 3 is a full build) |
| `tables/counts.tex`, `tables/counts_opt.tex` | the soundness counts of Sec. 4.3 as macros, for the basic and the optimised protocol |

`main.tex` inputs `hardware.tex` and the count files in its preamble. The numbers typed in the text are
not generated: `text_numbers.py` compares each one with the tables and fails on a mismatch. Some
timings of the second optimised batch are withheld (`code/artifacts/comparison/excluded_cells.csv`; two
nodes were slowed by other users' load during parts of that batch); `l40s_improved3` re-measured some of
them, and the paper, finalised without waiting for the rest, omits the others and the two optional cells
that were not run: `--` in Table 3 ("not measured"), no Table 1 row or Fig. 4 point that needs one, and
only measured values in the text (`paper_assets.py --check` lists them; nothing is pending). The real-LLM
results are in `code/artifacts/results/`.

Use matplotlib 3.11.2 (`code/requirements.txt`) for the figures: it reproduces the committed PDFs byte
for byte. With another version they come out slightly larger or smaller, so after regenerating them
check that the text still ends on page 8.

`acmart.cls` and `ACM-Reference-Format.bst` are the course template's copies (acmart 2.18), so the
build does not depend on the installed TeX version. `references.bib` holds the bibliography; every
author list was checked against the paper's own page.

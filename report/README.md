# Report

`main.pdf` is the report; `main.tex` its source (ACM `sigconf`, double column; nine pages,
the references included, which start on page 9).

`figures/*.pdf` and `tables/*.tex` are generated from the stored measurements by
`code/experiments/5_comparison/paper_assets.py` and `count_outcomes.py` (no GPU needed). The
report's benchmark numbers come from two runs on the same machines (an NVIDIA L40S prover and
8 threads of an AMD EPYC 9334 verifier; see `code/README.md`): the *optimised protocol*, platforms
`l40s_improved` and `l40s_improved2` (`raw_l40s_improved*/`; Tables 2, 3 and 5, Figure 4, the filled
points of Figure 3), and the *basic protocol*, platform `l40s` (`code/artifacts/comparison/raw_l40s/`,
measured at commit `9401431`; Table 4's left-hand side and the hollow points of Figure 3). Until
`l40s_improved2` is stored, the optimised numbers it will provide are printed in red with the basic
protocol's value as a placeholder (`\pending{}`; `paper_assets.py --check` lists them):

```bash
cd code && export PYTHONPATH=$PWD/src
python experiments/5_comparison/aggregate.py --platform l40s              # raw_l40s/ -> tables_l40s/
python experiments/5_comparison/aggregate.py --platform l40s_improved     # raw_l40s_improved/ -> tables_l40s_improved/
python experiments/5_comparison/aggregate.py --platform l40s_improved2    # raw_l40s_improved2/ -> tables_l40s_improved2/
python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved,l40s_improved2   # -> ../report/figures, ../report/tables
python experiments/5_comparison/count_outcomes.py --platform l40s --tex ../report/tables/counts.tex
python experiments/5_comparison/count_outcomes.py --platform l40s_improved,l40s_improved2 --prefix Opt --tex ../report/tables/counts_opt.tex   # placeholders until l40s_improved2 is stored
python experiments/5_comparison/count_outcomes.py --platform l40s_improved,l40s_improved2 --prefix Opt --definition --tex ../report/tables/counts_opt.tex   # once it is: the optimised definition's cells only
python experiments/5_comparison/text_numbers.py --platform l40s --optimised l40s_improved,l40s_improved2   # compare the text with the tables (exit 1 on a mismatch; add --definition with the counts)
cd ../report && latexmk -pdf main.tex                                     # or: tectonic -X compile main.tex
```

zkLLM's run on our GPU enters Table 5 through
`code/artifacts/results/zkllm_l40s/summary.csv` (`summarise.py --csv`). Use the pinned
matplotlib 3.11.2 of `code/requirements.txt` for the figures (they are then byte-identical).

`tables/hardware.tex` defines `\ProverGPU`, `\VerifierCPU`, `\VerifierThreads`, `\LLMNote`
(empty: every row of Table 3 is a full build) and `\NPendingCells` (the number of table and
figure numbers still pending), which `main.tex` inputs in its preamble. The table bodies use
`\pending{}`, which `main.tex` defines. The numbers typed in the text are not generated:
`text_numbers.py` (above) compares each one with the tables and fails on a mismatch, and `count_outcomes.py --tex` writes the counts of the abstract, §5.2, §5.4 and the
conclusion as macros (`tables/counts.tex`, `tables/counts_opt.tex`; `--platform rtx2080ti-v2`
prints the second platform's). The real-LLM results and zkLLM's
run on our GPU are in `code/artifacts/results/` (see `code/README.md`, *Where each result
comes from*). The figures depend slightly on the matplotlib version: the committed ones
were made with the pinned matplotlib 3.11.2 of `code/requirements.txt`, which reproduces
them exactly. With another version they come out slightly larger or smaller, so after
regenerating them check that the report still fits in nine pages.

`acmart.cls` and `ACM-Reference-Format.bst` are the course template's copies (acmart
2.18), so the build does not depend on the installed TeX version. `references.bib`
holds the bibliography; every author list was checked against the paper's own page.

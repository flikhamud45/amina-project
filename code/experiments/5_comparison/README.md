# 5. Tables, counts and the report's figures (report §4.3–4.4, all figures)

These scripts never run a model. They read the stored measurements in
`artifacts/comparison/`, so every table and figure can be re-made (or re-styled)
without the GPU. Each benchmark run is one platform root `raw_<platform>/` with its own
tables directory `tables_<platform>/`. The report's numbers are platform
**`rtx2080ti-v2`** (`raw_rtx2080ti-v2/`, 78,266 records), the default of `--platform`
unless `$PVI_PLATFORM` is set. `raw/` (platform `rtx2080ti`, accepted only by
`aggregate.py`) is the earlier run with an older version of the code, 59,547 records,
which the report does not use; its `measured_summary.csv` and `llm_full_model.csv` are in
`tables/`, next to the platform-independent `reported_curated.csv` and `analytic.csv`,
which the report does use (see `code/README.md`, *The stored benchmark runs*). A
re-measurement (Route B) uses a new platform name and the same commands. Run them from
`code/` in this order (about a minute in total; `paper_assets.py` with the pinned
matplotlib 3.11.2 of `requirements.txt`, since other versions change the figure sizes):

```bash
python experiments/5_comparison/aggregate.py --platform rtx2080ti-v2
python experiments/5_comparison/literature.py
python experiments/5_comparison/analytic.py
python experiments/5_comparison/count_outcomes.py --platform rtx2080ti-v2
python experiments/5_comparison/paper_assets.py --platform rtx2080ti-v2
python experiments/5_comparison/text_numbers.py --platform rtx2080ti-v2            # optional
python experiments/5_comparison/validate_extrapolation.py --platform rtx2080ti-v2  # optional
```

| Script | Reads | Writes |
|---|---|---|
| `aggregate.py --platform <p>` | `raw_<p>/**/*.jsonl` | `tables_<p>/measured_summary.csv` (median and mean per model/cell/metric), `tables_<p>/llm_full_model.csv` (full-model LLM costs: the full build where there is one, otherwise extrapolated from 1–2 blocks), `tables_<p>/llm_extrapolation_check.csv` (every full build against its 1–2 block extrapolation). `--platform rtx2080ti` rebuilds the earlier run's two tables in `tables/` from `raw/` (provenance only) |
| `literature.py` | `literature/reported_benchmarks.csv` | `tables/reported_curated.csv` (the published rows compared; shared by every platform) |
| `analytic.py` | the model shapes | `tables/analytic.csv` (the path test's cost of reaching 2^-40 on Llama-2-7B) |
| `count_outcomes.py --platform <p>` | `raw_<p>/` | prints the counts of §4.3 per suite (every tag included): honest queries accepted, attacks rejected by type, and which check rejected them; and the record total of §4.1 |
| `paper_assets.py --platform <p>` | `tables_<p>/`, `tables/reported_curated.csv` | `../report/figures/*.pdf` (one file per sub-figure), `../report/tables/{cnn,llm,ratios}.tex` and `../report/tables/hardware.tex` (`\ProverGPU`, `\VerifierCPU`, `\VerifierThreads`, and `\LLMNote` for the extrapolated rows of Table 3); refuses, before writing anything, if a number it needs is missing |
| `text_numbers.py --platform <p>` | `tables_<p>/`, `../report/main.tex` | prints every number typed in the text of `main.tex`, recomputed, next to its line (except the counts of §4.3 and the record total of §4.1: those come from `count_outcomes.py`) |
| `validate_extrapolation.py --platform <p>` | `tables_<p>/measured_summary.csv` | `tables_<p>/llm_extrapolation_validation.csv` (per timing and byte part: the two-point rule and a line fit through every block count, against the full builds) |
| `fingerprint_check.py <stored root> <new root>` | two raw roots | used by `smoke.sbatch`: every hardware-independent number of a new root (parameters, accuracies, path-test facts, security parameters, proof bytes except the Merkle term, verdicts) must equal the stored one's |

Table 3 stars the rows extrapolated from 1- and 2-block builds (on `rtx2080ti-v2`, only
Llama-2-13B), Figure 5 would hatch such bars (none on `rtx2080ti-v2`), and Figure 4 and
the text's ranges use the full builds only.

**The literature catalogue.** `artifacts/comparison/literature/` holds 413 published
measurements from 32 systems. Each row carries its table and page and a verbatim
snippet, and was re-checked against the paper. Published numbers are shown as
reported, on their own hardware, and are never mixed silently with ours.

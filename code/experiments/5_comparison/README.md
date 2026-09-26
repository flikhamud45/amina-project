# 5. Tables, counts and the report's figures (report §4.3–4.4, all figures)

These scripts never run a model. They read the stored measurements in
`artifacts/comparison/`, so every table and figure can be re-made (or re-styled)
without the GPU. Run them in this order (about a minute in total):

| Script | Reads | Writes |
|---|---|---|
| `aggregate.py` | `raw/**/*.jsonl` | `tables/measured_summary.csv` (median, IQR, min, max per model/cell/metric), `tables/llm_full_model.csv` (full-model LLM costs, measured or extrapolated from 1–2 blocks), `tables/measured_long.csv` (flat copy, not committed) |
| `literature.py` | `literature/reported_benchmarks.csv` | `tables/reported_curated.csv` (the published rows compared), `tables/systems.csv` (each system's setting and error type) |
| `analytic.py` | the model shapes | `tables/analytic.csv` (closed-form proof bytes, work and soundness for every model and prompt length; also the path test's cost of reaching 2^-40) |
| `report_tables.py` | `tables/` | `tables/report_tables.md` and `.tex`: every result table, including the like-for-like ratios (`MATCHES`) |
| `count_outcomes.py` | `raw/` | prints the counts of §4.3: honest queries accepted, attacks rejected, and which check rejected them |
| `paper_assets.py` | `tables/` | `../report/figures/*.pdf` (one file per sub-figure) and `../report/tables/*.tex` |

`common.py` holds the readers shared by the last three scripts.

**The literature catalogue.** `artifacts/comparison/literature/` holds 413 published
measurements from 32 systems. Each row carries its table and page and a verbatim
snippet, and was re-checked against the paper. Published numbers are shown as
reported, on their own hardware, and are never mixed silently with ours.

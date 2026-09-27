# 5. Tables, counts and the report's figures (report §4.3–4.4, all figures)

These scripts never run a model. They read the stored measurements in
`artifacts/comparison/`, so every table and figure can be re-made (or re-styled)
without the GPU. Run them in this order (about a minute in total):

| Script | Reads | Writes |
|---|---|---|
| `aggregate.py` | `raw/**/*.jsonl` | `tables/measured_summary.csv` (median and mean per model/cell/metric), `tables/llm_full_model.csv` (full-model LLM costs, measured or extrapolated from 1–2 blocks) |
| `literature.py` | `literature/reported_benchmarks.csv` | `tables/reported_curated.csv` (the published rows compared) |
| `analytic.py` | the model shapes | `tables/analytic.csv` (the path test's cost of reaching 2^-40 on Llama-2-7B) |
| `count_outcomes.py` | `raw/` and every `raw_<platform>/` (never the smoke job's `raw_smoke*/`), or `--platform` | prints the counts of §4.3 per suite: honest queries accepted, attacks rejected, and which check rejected them (`--variant ''`: untagged cells only) |
| `paper_assets.py` | `tables/` | `../report/figures/*.pdf` (one file per sub-figure) and `../report/tables/*.tex` |

**The literature catalogue.** `artifacts/comparison/literature/` holds 413 published
measurements from 32 systems. Each row carries its table and page and a verbatim
snippet, and was re-checked against the paper. Published numbers are shown as
reported, on their own hardware, and are never mixed silently with ours.

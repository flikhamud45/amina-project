# 5. Tables, counts and the report's figures (report §4.3–4.4, all figures)

These scripts never run a model. They read the stored measurements in
`artifacts/comparison/`, so every table and figure can be re-made (or re-styled)
without the GPU. Each benchmark run is one platform root `raw_<platform>/` with its own
tables directory `tables_<platform>/`. The report's numbers are platform **`l40s`**
(`raw_l40s/`, 123,832 records), the default of `--platform` unless `$PVI_PLATFORM` is set.
`rtx2080ti-v2` (`raw_rtx2080ti-v2/`, 78,266 records) is the second platform; the report
quotes only its verdicts and record total (`count_outcomes.py --platform rtx2080ti-v2`) and
its hardware-independent numbers (`fingerprint_check.py`). `raw/` (platform `rtx2080ti`,
accepted only by `aggregate.py`) is the earliest run, with an older version of the code,
59,547 records, which the report does not use. Its `measured_summary.csv` and
`llm_full_model.csv` are in `tables/`, next to the platform-independent
`reported_curated.csv` and `analytic.csv`, which the report does use (see `code/README.md`,
*The stored benchmark runs*). A re-measurement (Route B) uses a new platform name and the
same commands. Run them from `code/` in this order (about a minute in total;
`paper_assets.py` with the pinned matplotlib 3.11.2 of `requirements.txt`, since other
versions change the figure sizes):

```bash
python experiments/5_comparison/aggregate.py --platform l40s
python experiments/5_comparison/literature.py
python experiments/5_comparison/analytic.py
python experiments/5_comparison/count_outcomes.py --platform l40s
python experiments/5_comparison/paper_assets.py --platform l40s
python experiments/5_comparison/count_outcomes.py --platform rtx2080ti-v2   # §4.1/§4.3: the 2080 Ti's verdicts and records
python experiments/5_comparison/fingerprint_check.py artifacts/comparison/raw_rtx2080ti-v2 artifacts/comparison/raw_l40s   # §4.1: the 1,978 shared numbers
python experiments/5_comparison/text_numbers.py --platform l40s             # optional
python experiments/5_comparison/validate_extrapolation.py --platform l40s   # optional
```

| Script | Reads | Writes |
|---|---|---|
| `aggregate.py --platform <p>` | `raw_<p>/**/*.jsonl` | `tables_<p>/measured_summary.csv` (median and mean per model/cell/metric), `tables_<p>/llm_full_model.csv` (full-model LLM costs: the full build where there is one, otherwise extrapolated from 1–2 blocks), `tables_<p>/llm_extrapolation_check.csv` (every full build against its 1–2 block extrapolation). `--platform rtx2080ti` rebuilds the earliest run's two tables in `tables/` from `raw/` (provenance only) |
| `literature.py` | `literature/reported_benchmarks.csv` | `tables/reported_curated.csv` (the published rows compared; shared by every platform) |
| `analytic.py` | the model shapes | `tables/analytic.csv` (the path test's cost of reaching 2^-40 on Llama-2-7B) |
| `count_outcomes.py --platform <p>` | `raw_<p>/` | prints the counts of §4.3 per suite (every tag included: `_gpuv`, `_batch`, `_thr1`, `_thr12`, `_tf32`, `_nofix`, `_nolean`): honest queries accepted, attacks rejected by type, and which check rejected them; and the record total of §4.1 (the report runs it for `l40s` and `rtx2080ti-v2`) |
| `paper_assets.py --platform <p>` | `tables_<p>/`, `tables/reported_curated.csv` | `../report/figures/*.pdf` (one file per sub-figure), `../report/tables/{cnn,llm,ratios}.tex` and `../report/tables/hardware.tex` (`\ProverGPU`, `\VerifierCPU`, `\VerifierThreads`, and `\LLMNote` for the extrapolated rows of Table 3); `--check` only lists missing cells; without it the script refuses, before writing anything, if a number it needs is missing |
| `text_numbers.py --platform <p>` | `tables_<p>/`, `../report/main.tex` | prints every number typed in the text of `main.tex`, recomputed, next to its line (except the counts of §4.3 and the record total of §4.1: those come from `count_outcomes.py`) |
| `validate_extrapolation.py --platform <p>` | `tables_<p>/measured_summary.csv` | `tables_<p>/llm_extrapolation_validation.csv` (per timing, byte and memory metric: the two-point rule and a line fit through every block count, against the full builds) |
| `fingerprint_check.py <root A> <root B>` | two raw roots | checks that every hardware-independent number the two roots share (parameters, accuracies, path-test facts, security parameters, proof bytes except the Merkle term, verdicts) agrees (equal; the float accuracy within 0.002), and prints every honest rejection and accepted attack in B: §4.1's 1,978 numbers (`raw_rtx2080ti-v2` against `raw_l40s`, 0 differences), and `smoke.sbatch`'s check of a new GPU against `raw/` |

**The improvements' root.** `raw_l40s_improved/` (platform `l40s_improved`, 68,363 records)
holds the options added after the report (`experiments/6_improvements`, `IMPROVEMENTS.md` at
the repository root), measured on the report's hardware. The same scripts read it:
`aggregate.py --platform l40s_improved` rebuilds `tables_l40s_improved/` unchanged,
`count_outcomes.py --platform l40s_improved` gives its verdicts (4,380 honest queries
accepted, 4,973 attacks rejected), and `fingerprint_check.py artifacts/comparison/raw_l40s
artifacts/comparison/raw_l40s_improved` finds its 565 hardware-independent numbers shared with
the report's root equal. The new timing parts add up like the others: `prove_lookups` and
`verify_lookups` (a plan's lookup tables), `prove_encode` and `verify_decode` (`--wire`), and the
streaming verifier's `verify_total`, which replaces the verify phases it overlaps
(`paper_assets.py` refuses a row with both). The report does not use these cells:
`paper_assets.py` and `text_numbers.py` are written for `l40s` and do not draw them.

Table 3 stars the rows extrapolated from 1- and 2-block builds, and Figure 5 would hatch
such bars (there are none on `l40s`, where every drawn model is measured at full depth).
Figure 4 and the text's ranges use the full builds only.

**The literature catalogue.** `artifacts/comparison/literature/` holds 413 published
measurements from 32 systems. Each row carries its table and page and a verbatim
snippet, and was re-checked against the paper. Published numbers are shown as
reported, on their own hardware, and are never mixed silently with ours.

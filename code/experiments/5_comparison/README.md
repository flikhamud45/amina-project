# 5. Tables, counts and the report's figures (report §5, all figures)

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
python experiments/5_comparison/count_outcomes.py --platform l40s --tex ../report/tables/counts.tex
python experiments/5_comparison/count_outcomes.py --platform l40s_improved --prefix Opt --tex ../report/tables/counts_opt.tex
python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved
python experiments/5_comparison/count_outcomes.py --platform rtx2080ti-v2   # §5.1: the 2080 Ti's verdicts and records
python experiments/5_comparison/fingerprint_check.py artifacts/comparison/raw_rtx2080ti-v2 artifacts/comparison/raw_l40s   # §5.1: the 1,978 shared numbers
python experiments/5_comparison/text_numbers.py --platform l40s --optimised l40s_improved   # compare every line with main.tex
python experiments/5_comparison/validate_extrapolation.py --platform l40s   # optional
```

| Script | Reads | Writes |
|---|---|---|
| `aggregate.py --platform <p>` | `raw_<p>/**/*.jsonl` | `tables_<p>/measured_summary.csv` (median and mean per model/cell/metric), `tables_<p>/llm_full_model.csv` (full-model LLM costs: the full build where there is one, otherwise extrapolated from 1–2 blocks), `tables_<p>/llm_extrapolation_check.csv` (every full build against its 1–2 block extrapolation). `--platform rtx2080ti` rebuilds the earliest run's two tables in `tables/` from `raw/` (provenance only) |
| `literature.py` | `literature/reported_benchmarks.csv` | `tables/reported_curated.csv` (the published rows compared; shared by every platform) |
| `analytic.py` | the model shapes | `tables/analytic.csv` (the path test's cost of reaching 2^-40 on Llama-2-7B) |
| `count_outcomes.py --platform <p> [--prefix P --tex F]` | `raw_<p>/` | prints the counts of §5.2 and §5.4 per suite (every tag included: `_gpuv`, `_batch`, `_thr1`, `_thr12`, `_tf32`, `_nofix`, `_nolean` and the optimised run's tags): honest queries accepted, attacks rejected by type, which check rejected them, the attacks on cells built under the planning rule (`--policy auto`) and the '+1' LLM attacks on 1–2 block builds; and the record total. `--tex` writes them as LaTeX macros (`\NHonest`, `\NAttacks`, `\NAttacksCNN`, `\NAttacksLLM`, `\NFreivalds`, `\NColumnsCode`, `\NColumnsMerkle`, `\NAttacksPlusOne`, `\NAttacksPlusOnePartial`, `\NRecords`, …; with `--prefix Opt` for `l40s_improved`, `\OptNHonest`, `\OptNAttacks`, `\OptNAttacksPlans`, …) and refuses if an honest query was rejected or an attack accepted |
| `paper_assets.py --platform <p> --optimised <q>` | `tables_<p>/` (the basic protocol), `tables_<q>/` (the optimised protocol on the same machines), `tables/reported_curated.csv`, `../artifacts/results/zkllm_l40s/summary.csv` | `../report/figures/*.pdf` (one file per sub-figure, on a fixed canvas equal to its printed width, TrueType fonts; `save()` refuses text below 7 pt, overlapping or leaving the canvas, and labels on markers), `../report/tables/{cnn,llm,opt,ratios}.tex` (Tables 2–5) and `../report/tables/hardware.tex` (`\ProverGPU`, `\VerifierCPU`, `\VerifierThreads`, and `\LLMNote` for the extrapolated rows of Table 3); `--check` only lists the missing cells of both runs; without it the script refuses, before writing anything, if a number it needs is missing. Without `--optimised` it writes the basic assets only (no Tables 4–5, no filled points in Figure 3); `--draft` writes figures despite layout problems, to iterate on a style |
| `text_numbers.py --platform <p> --optimised <q> [--tex F]` | both runs' tables, `../report/main.tex` | prints every number typed in the text of `main.tex`, recomputed, next to its line, and exits 1 if a sentence it checks is no longer found (the counts are macros of `count_outcomes.py --tex`) |
| `validate_extrapolation.py --platform <p>` | `tables_<p>/measured_summary.csv` | `tables_<p>/llm_extrapolation_validation.csv` (per timing, byte and memory metric: the two-point rule and a line fit through every block count, against the full builds) |
| `fingerprint_check.py <root A> <root B>` | two raw roots | checks that every hardware-independent number the two roots share (parameters, accuracies, path-test facts, security parameters, proof bytes except the Merkle term, verdicts) agrees (equal; the float accuracy within 0.002), and prints every honest rejection and accepted attack in B: §4.1's 1,978 numbers (`raw_rtx2080ti-v2` against `raw_l40s`, 0 differences), and `smoke.sbatch`'s check of a new GPU against `raw/` |

**The improvements' root.** `raw_l40s_improved/` (platform `l40s_improved`, 68,363 records)
holds the options added after the report (`experiments/6_improvements`, `IMPROVEMENTS.md` at
the repository root), measured on the report's hardware, and the report's cells run by the new
code (untagged; faster than `raw_l40s/`, since its verifier engineering is always on). The same
scripts read it: `aggregate.py --platform l40s_improved` rebuilds `tables_l40s_improved/` unchanged,
`count_outcomes.py --platform l40s_improved` gives its verdicts (4,380 honest queries
accepted, 4,973 attacks rejected), and `fingerprint_check.py artifacts/comparison/raw_l40s
artifacts/comparison/raw_l40s_improved` finds its 565 hardware-independent numbers shared with
the report's root equal. The new timing parts add up like the others: `prove_lookups` and
`verify_lookups` (a plan's lookup tables), `prove_encode` and `verify_decode` (`--wire`), and the
streaming verifier's `verify_total`, which replaces the verify phases it overlaps
(`paper_assets.py` refuses a row with both). The report calls these cells the *optimised
protocol* (`paper_assets.py --optimised l40s_improved`): the filled points of Figure 3
(`defence_C_int_lam{40,80,128}_rate4_wire_polauto`), the right-hand side of Table 4 and every
ratio of Table 5. Table 5 uses the Fiat–Shamir cells against the non-interactive systems where
they exist (the CNNs and GPT-2; ZKTorch's Llama-2-7B row is interactive and marked), the
interactive ones against zkLLM and Maverick, and counts Maverick's 37.4 ms non-linear replay in
its verifier time (`MAVERICK_NONLINEAR_S`). Tables 2–3 and Figure 4 stay on the basic run, which
covers every row.

Table 3 stars the rows extrapolated from 1- and 2-block builds (there are none on `l40s`,
where every drawn model is measured at full depth). Figure 4 and the text's ranges use the
full builds only.

**The literature catalogue.** `artifacts/comparison/literature/` holds 413 published
measurements from 32 systems. Each row carries its table and page and a verbatim
snippet, and was re-checked against the paper. Published numbers are shown as
reported, on their own hardware, and are never mixed silently with ours.

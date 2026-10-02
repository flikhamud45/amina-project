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
python experiments/5_comparison/count_outcomes.py --platform l40s_improved,l40s_improved2 --prefix Opt --tex ../report/tables/counts_opt.tex   # before the second run: placeholders, every cell of l40s_improved
python experiments/5_comparison/count_outcomes.py --platform l40s_improved,l40s_improved2 --prefix Opt --definition --tex ../report/tables/counts_opt.tex   # once l40s_improved2 is stored: the optimised definition's cells only (Sec. 5.2, "on the optimised one")
python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved,l40s_improved2
python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved,l40s_improved2 --check --pending-csv pending_cells.csv   # what is still red
python experiments/5_comparison/count_outcomes.py --platform rtx2080ti-v2   # §5.1: the 2080 Ti's verdicts and records
python experiments/5_comparison/fingerprint_check.py artifacts/comparison/raw_rtx2080ti-v2 artifacts/comparison/raw_l40s   # §5.1: the 1,978 shared numbers
python experiments/5_comparison/text_numbers.py --platform l40s --optimised l40s_improved,l40s_improved2   # compare the text with the tables (exit 1 on a mismatch); add --definition once the counts use it
python experiments/5_comparison/validate_extrapolation.py --platform l40s   # optional
```

| Script | Reads | Writes |
|---|---|---|
| `aggregate.py --platform <p>` | `raw_<p>/**/*.jsonl` | `tables_<p>/measured_summary.csv` (median and mean per model/cell/metric), `tables_<p>/llm_full_model.csv` (full-model LLM costs: the full build where there is one, otherwise extrapolated from 1–2 blocks), `tables_<p>/llm_extrapolation_check.csv` (every full build against its 1–2 block extrapolation). `--platform rtx2080ti` rebuilds the earliest run's two tables in `tables/` from `raw/` (provenance only) |
| `literature.py` | `literature/reported_benchmarks.csv` | `tables/reported_curated.csv` (the published rows compared; shared by every platform) |
| `analytic.py` | the model shapes | `tables/analytic.csv` (the path test's cost of reaching 2^-40 on Llama-2-7B) |
| `count_outcomes.py --platform <p>[,<q>...] [--prefix P --tex F]` | `raw_<p>/` (several roots are added up; a listed root not yet run is skipped, and `--tex` then writes every count as `\pending{...}`; `--definition` counts only the defence and tamper cells of the optimised definition, `paper_assets.definition_tagsets()`; `--require-tag TAG` only the cells carrying a tag) | prints the counts of §5.2 and §5.4 per suite (every tag included: `_gpuv`, `_batch`, `_thr1`, `_thr12`, `_tf32`, `_nofix`, `_nolean` and the optimised run's tags): honest queries accepted, attacks rejected by type, which check rejected them, the attacks on cells built under the planning rule (`--policy auto`) and the '+1' LLM attacks on 1–2 block builds; and the record total. `--tex` writes them as LaTeX macros (`\NHonest`, `\NAttacks`, `\NAttacksCNN`, `\NAttacksLLM`, `\NFreivalds`, `\NColumnsCode`, `\NColumnsMerkle`, `\NAttacksPlusOne`, `\NAttacksPlusOnePartial`, `\NRecords`, …; with `--prefix Opt` for the optimised runs, `\OptNHonest`, `\OptNAttacks`, `\OptNAttacksPlans`, …) and refuses if an honest query was rejected or an attack accepted |
| `paper_assets.py --platform <p> --optimised <q>[,<r>...]` | `tables_<p>/` (the basic protocol), `tables_<q>/`, `tables_<r>/` (the optimised protocol on the same machines; a later run's cell replaces an earlier one's, and a run not yet aggregated is skipped with a warning), `tables/reported_curated.csv`, `../artifacts/results/zkllm_l40s/summary.csv` | `../report/figures/*.pdf` (one file per sub-figure, on a fixed canvas equal to its printed width, TrueType fonts; `save()` refuses text below 7 pt, overlapping or leaving the canvas, and labels on markers), `../report/tables/{cnn,llm,opt,ratios}.tex` (Tables 2-5) and `../report/tables/hardware.tex` (`\ProverGPU`, `\VerifierCPU`, `\VerifierThreads`, `\LLMNote`, and `\NPendingCells`, the number of table and figure numbers still pending). An optimised number whose cell is not stored yet is *pending*: the basic protocol's value stands in, as `\pending{...}` (red; main.tex defines the macro) in the tables and with a red edge and a legend note in the figures. `--check` writes nothing and lists the basic run's missing cells (an error) and the pending numbers; `--pending-csv F` writes the pending numbers and the acceptable stand-ins (`asset,row,column,model,optimised_cell,basic_cell,status`); `--strict` refuses to write while any is pending; `--draft` writes figures despite layout problems |
| `text_numbers.py --platform <p> --optimised <q>[,<r>...] [--tex F] [--list]` | the runs' tables, the raw roots (the counts), `../report/tables/counts*.tex`, `../report/main.tex` | compares every number typed in the text with the value recomputed the way the tables are (each sentence found by the wording around the number), and exits 1 on a `MISMATCH`, a final number still inside `\pending{}` (`UNWRAP`), a number that depends on a pending cell but is black (`UNMARKED`), or a sentence no longer found (`ANCHOR`); a red placeholder that differs from the tables' is only reported. In a pending range, an endpoint that a final (or stand-in) element supplies may stay black (`3.8--\pending{27.4}`): once the pending cell is stored, an endpoint it moves fails as `MISMATCH`. `\pendingclaim{}` is transparent, and the count macros are expanded from `tables/counts*.tex` |
| `validate_extrapolation.py --platform <p>` | `tables_<p>/measured_summary.csv` | `tables_<p>/llm_extrapolation_validation.csv` (per timing, byte and memory metric: the two-point rule and a line fit through every block count, against the full builds) |
| `fingerprint_check.py <root A> <root B>` | two raw roots | checks that every hardware-independent number the two roots share (parameters, accuracies, path-test facts, security parameters, proof bytes except the Merkle term, verdicts) agrees (equal; the float accuracy within 0.002), and prints every honest rejection and accepted attack in B: §4.1's 1,978 numbers (`raw_rtx2080ti-v2` against `raw_l40s`, 0 differences), and `smoke.sbatch`'s check of a new GPU against `raw/` |

**The optimised runs.** `raw_l40s_improved/` (platform `l40s_improved`, 68,363 records)
holds the options added after the basic run (`experiments/6_improvements`, `IMPROVEMENTS.md` at the
repository root), measured on the same hardware, and the basic cells run by the new code (untagged;
faster than `raw_l40s/`, since its verifier engineering is always on). A second run, platform
`l40s_improved2`, adds the cells it lacks. The same scripts read both: `aggregate.py --platform
l40s_improved` rebuilds `tables_l40s_improved/` unchanged, `count_outcomes.py --platform l40s_improved`
gives its verdicts (4,380 honest queries accepted, 4,973 attacks rejected), and `fingerprint_check.py
artifacts/comparison/raw_l40s artifacts/comparison/raw_l40s_improved` finds its 565 hardware-independent
numbers shared with the basic root equal. The new timing parts add up like the others: `prove_lookups`
and `verify_lookups` (a plan's lookup tables), `prove_encode` and `verify_decode` (`--wire`), and the
streaming verifier's `verify_total`, which replaces the verify phases it overlaps (`paper_assets.py`
refuses a row with both).

The report's results are the *optimised protocol*: Tables 2, 3 and 5, Figure 4 and the filled points
of Figure 3; the basic protocol appears in Table 4 (basic -> optimised) and as Figure 3's hollow points.
Which stored cell holds each optimised number is defined in one place, `OPTIMISED` in `paper_assets.py`,
as the cell's settings and its set of variant tags (matched in any order): image models `_wire_polauto`
(C) and `_wire` (K, Kpre); language models `_wire_prune_polauto` (C, CPU verifier),
`_wire_gpuv_stream_prune_polauto` (C, streaming GPU verifier) and `_wire_prune_lookups` (Kpre); Qwen3-4B
in Maverick's setting `_thr1_wire_prune_lookups` (Kpre, lambda = 40, one verifier thread). A few stored
cells stand in, in black, until the defining one exists (`acceptable`): only Llama-2-7B's runs without
pruning. GPT-2's non-streaming GPU run is not the streaming verifier of the definition, and the image
models' K/Kpre cells re-run by the new code send the basic proof byte for byte, so both stay pending
(red). At 2,048 tokens the stored `_gpuv_stream` cells send the basic proof format, so they are not the
definition either. Figure 3 draws no path curve for the 224-pixel ResNet (`SECURITY_NO_PATH`: it lies
on VGG-16's), and `fig_security` refuses to print its title ("Optimised at 128 bits sends less than
paths at 40 bits") once a stored point of any model, the 224-pixel ResNet included, makes it false.

After the second run, regenerate and run `text_numbers.py`: with `RERUN_KEPT=1` the kept cells are
re-measured in `l40s_improved2`, which wins over `l40s_improved`, so black numbers of the text can move
too, and each one that does fails as `MISMATCH` until it is updated by hand. In an LLM row the prover and the proof come from the CPU-verifier cell (the same one at
every prompt length, so Tables 3-5 and the text quote one prover time per configuration) and the GPU
verifier from the GPU cell. Table 5 uses the Fiat-Shamir cells against the non-interactive systems
where they are stored (ZKTorch's Llama-2-7B row is interactive and marked until a Fiat-Shamir cell
is, with an asterisk, `INTERACTIVE_MARK`), the interactive ones against zkLLM and Maverick, and marks the rows with footnote letters in the order a reader meets them, which main.tex explains: zkCNN a (one verifier core), zkGPT b (no stated prompt length) and DeepProve c (its transparent configuration) through `SYSTEM_MARK`, Maverick d and zkLLM's code on our GPU e (`ZKLLM_OWN`). It counts Maverick's 37.4 ms non-linear replay
in its verifier time (`MAVERICK_NONLINEAR_S`); `TABLE5_VERIFIER_EXTRA_TAGS` can switch the zkCNN rows
to a one-thread verifier run.

Tables 2-5 and Figure 4 use full-depth builds only; the 30-70B shapes enter only the text,
through their 1- and 2-block builds.

**The literature catalogue.** `artifacts/comparison/literature/` holds 413 published
measurements from 32 systems. Each row carries its table and page and a verbatim
snippet, and was re-checked against the paper. Published numbers are shown as
reported, on their own hardware, and are never mixed silently with ours.

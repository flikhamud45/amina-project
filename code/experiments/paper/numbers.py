"""The paper's tables and text numbers from the final run: read the run's records, check them, build every table
body and every text number, and write them for the paper.

    cd code
    export PYTHONPATH=src:.                                                  # the library and the experiments
    ../.venv/bin/python -m experiments.comparison.aggregate --platform final  # bench.py's records -> tables_final/
    ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final --check     # report only
    ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final             # -> generated/
    ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/final --platform final --strict \\
        --expect ../logs/final/dryrun.txt                                    # fail on anything missing
    ../.venv/bin/python -m experiments.paper.numbers --logs ../logs/rehearsal --platform rehearsal --rehearsal --check

Inputs (nothing here runs a model):

* ``--logs``: the run package's folder ``logs/<name>/`` (``experiments/package/run.sh``): the job logs
  ``sp-<cell>-<jobid>.out``, whose lines that start with ``{`` are the records (:mod:`experiments.paper.records`),
  and the documents of the attack tier (``attack/*.json``, ``attack/analytic.csv``) and of the tv tier
  (``sampler_tv_*.json``);
* ``--platform``: the benchmark's aggregated tables ``tables_<name>/`` and its raw records ``raw_<name>/`` (the
  outcome counts) in ``--bench`` (default ``artifacts/comparison``), :mod:`experiments.paper.bench`, whose decoder
  cells are those of the attention probabilities' scale ``--prob-max`` (default 2047, the cells tagged
  ``pmax2047``);
* the published systems' values (``artifacts/comparison/tables/reported_curated.csv``), zkLLM's stored runs on the
  run's GPU model (``artifacts/results/zkllm_l40s/``), the formulas of the spot-checking schemes and of the path
  protocol (``attack/sok.py``: ``tab:sok``, ``tab:paths``) and, for ``tab:params``, ``tab:setup`` and the break-even,
  the library's parameters and byte model (several minutes on a laptop CPU; ``--no-byte-model`` skips them).

Outputs (``--out``, default ``paper_sp2027/generated/``; ``--check`` writes nothing):

* ``<table>.tex`` for each table of :data:`experiments.paper.tables.TABLE_FILES`: the table's body rows (what sits
  between ``\\midrule`` and ``\\bottomrule`` in the paper), under a header comment with the run's protocol settings
  and sources;
* ``numbers.tex``: one ``\\newcommand`` per text number (:mod:`experiments.paper.macros`);
* ``cells.csv``: every table cell and every macro with its value, printed form and source record;
* ``coverage.txt``: the records per tier and kind, per table the cells filled and missing, the missing macros, the
  checks and, with ``--expect <dry-run listing>`` (``run.sh --platform <name> --dry-run [--optional] all``; a listing
  that does not end with run.sh's summary line, ``# <tiers>: N jobs, ...`` for its N jobs, is refused), every job of
  the package whose log is missing or has no record;
* with ``--figures`` (default ``paper_sp2027/figures``) ``ev_breakeven.pdf`` once its inputs are all in the run, and
  with ``--schematics`` also the schematic figures (:mod:`experiments.paper.figures`).

``paper_sp2027/main.tex`` inputs each file from ``generated/`` when it exists and from ``fallback/`` (the values the
paper prints today) otherwise.

Checks: an honest query rejected, a tamper accepted, a byte accounting, serving or A/B check that failed, a setup
proof rejected, an attribution whose float copy failed a check or whose integer graph is not the quality tier's cell
of the same graph (:func:`experiments.paper.records.cross_checks`), the smoke's verdicts, records of more than one
machine, builds of one certified model whose digests, weights or commitment roots differ
(:func:`experiments.paper.macros.digest_agreement`), a table cell that disagrees with the macro printing the same
value (:data:`SAME_VALUE`; in the written files and in ``paper_sp2027/fallback/``), a statement of the text that the
run's values contradict (``CHECK FAILED text:``, :data:`experiments.paper.macros.TEXT_CLAIM`): each is a ``CHECK
FAILED`` line; a log line that starts with ``{`` and is not a record of a known kind is a ``PARSE`` line.  The exit
status is 1 on either, and with ``--strict`` also on any missing cell, macro or job.  ``WARNING`` lines, which do not
change it: one quantity measured by two jobs differs by more than 10% (:data:`TWO_JOBS`), two records of one quality
cell differ (:meth:`experiments.paper.records.Run.ppl`), the records' mode-C plan is not ``--policy``.
``--rehearsal``: a rehearsal's folder (small models; no cell of the paper is expected): every tier must have at least
one record of each kind it writes (:data:`experiments.paper.records.REHEARSAL_KINDS`), every builder runs on what is
there, and the checks apply, except those of what the text states about the final values (the sign of a loss; the
attribution's ranking; a path count of the attack's documents or the benchmark that differs from the formula's,
which ``tab:sok`` prints as the same number; the 11-bit attention probabilities of every cost record; the grid of a
validation study; the reference of a ratio: ``CHECK FAILED text:`` lines), which are warnings.
"""
from __future__ import annotations

import argparse
import csv
import glob
import re
import sys
from collections import defaultdict
from pathlib import Path

from experiments.common.paths import BENCH, CODE
from experiments.paper import macros as M
from experiments.paper import records as R
from experiments.paper import tables as T
from experiments.paper.bench import DEFAULT_POLICY, DEFAULT_PROB_MAX, Bench, literature

REPO = CODE.parent
PAPER = REPO / "paper_sp2027"


SUMMARY = re.compile(r"^# (?!tier )\S.*: (\d+) jobs\b")
"""run.sh's last line of a dry run: ``# <tiers>: N jobs, job limits H h``."""


def expected_jobs(listing: Path) -> dict:
    """``{job name: [the files its command writes with --out]}`` of a dry-run listing (``sbatch ... -J <name>``).  A
    listing that does not end with run.sh's summary line (:data:`SUMMARY`), or whose summary counts other jobs than it
    lists, is incomplete (a dry run that stopped early, e.g. at a tier's check of ``HF_HOME``): ValueError."""
    jobs, lines = {}, [x for x in listing.read_text(encoding="utf-8").splitlines() if x.strip()]
    n = 0
    for line in lines:
        m = re.search(r" -J (\S+)", line)
        if line.startswith("sbatch") and m:
            jobs[m.group(1)] = [Path(p).name for p in re.findall(r"--out (\S+\.(?:json|csv))", line)]
            n += 1
    last = SUMMARY.match(lines[-1]) if lines else None
    if last is None:
        raise ValueError(f"{listing}: no summary line '# <tiers>: N jobs' at its end (an incomplete dry run)")
    if int(last.group(1)) != n:
        raise ValueError(f"{listing}: its summary counts {last.group(1)} jobs, it lists {n}")
    return jobs


def job_coverage(jobs: dict, logs: Path, recs: list) -> list[str]:
    """The jobs of ``jobs`` whose log is missing or holds nothing: a benchmark job (``b-``, ``bt-``) without its
    ``done in`` line, an attack or tv job without the files it writes, any other job without a record."""
    have = defaultdict(int)
    for r in recs:
        have[r.cell] += 1
    missing = []
    for name, outs in jobs.items():
        cell = name[3:] if name.startswith("sp-") else name
        tier = R.tier_of(cell)
        if tier in ("bench", "batch"):
            files = glob.glob(str(logs / f"{name}-*.out"))
            if not any("done in" in Path(p).read_text(encoding="utf-8", errors="replace") for p in files):
                missing.append(name)
        elif outs:
            if not all((logs / "attack" / o).exists() or (logs / o).exists() for o in outs):
                missing.append(name)
        elif not have.get(cell):
            missing.append(name)
    return missing


def machine_check(run: R.Run, bench: Bench) -> list[str]:
    """One platform: the records' GPUs and CPUs, and the benchmark tables' provers and client CPUs, name one GPU and
    one CPU model."""
    gpus, cpus = set(), set()
    for r in run.recs:
        env = r.d.get("env") or {}
        if env.get("gpu"):
            gpus.add(env["gpu"])
        if env.get("cpu"):
            cpus.add(env["cpu"])
    provers, clients = bench.hardware()
    gpus |= provers
    cpus |= {re.sub(r" x\d+ threads$", "", c) for c in clients}
    bad = []
    if len(gpus) > 1:
        bad.append(f"CHECK FAILED platform: records of several GPUs: {sorted(gpus)}")
    if len(cpus) > 1:
        bad.append(f"CHECK FAILED platform: records of several CPUs: {sorted(cpus)}")
    return bad


def platform_text(run: R.Run, bench: Bench) -> str:
    env = run.environment()
    provers, clients = bench.hardware()
    gpu = env["gpu"] or next(iter(provers), None)
    cpu = env["cpu"] or next(iter(clients), None)
    return f"platform: {gpu or '?'} (prover, GPU verifier), {cpu or '?'} (8 threads: CPU verifier, re-execution)"


def record_plans(run: R.Run, bench: Bench) -> set:
    """Mode C's commitment plans that the run's records name: a mode-C record's ``policy`` and the benchmark
    tables' ``pol<plan>`` tags."""
    plans = {str(r.d["policy"]) for r in run.recs if r.d.get("mode") == "C" and r.d.get("policy")}
    for row in bench.llm + bench.summary:
        plans |= set(re.findall(r"_pol([A-Za-z0-9]+)", row.get("variant", "") + " " + row.get("cell", "")))
    return plans


def choose_plan(policy: str | None, plans: set) -> tuple[str, list[str]]:
    """The plan the tables and macros select (``--policy``, else the records' one plan, else
    :data:`experiments.paper.bench.DEFAULT_POLICY`) and the warnings when the records name another plan or several."""
    chosen = policy or (next(iter(plans)) if len(plans) == 1 else DEFAULT_POLICY)
    warn = []
    if len(plans) > 1:
        warn.append(f"WARNING plan: the records name several mode-C plans {sorted(plans)}; selecting {chosen}")
    elif plans and chosen not in plans:
        warn.append(f"WARNING plan: the records' mode-C plan is {next(iter(plans))}, the tables select {chosen} "
                    "(--policy): their mode-C cells stay empty")
    return chosen, warn


# ------------------------------------------------------------------------------------------------ cells and macros
SAME_VALUE = (
    # (table, section, row, column, part, number, scale of the cell, macro, scale of the macro): a table cell and the
    # macro that print one value of one record.  ``section``: a substring of the table's last heading row (or None);
    # ``row``: substrings of the row's leading cells (an empty leading cell repeats the one above); ``part``: the
    # ``a / b`` half of the cell (or None); ``number``: which number of the cell; the scales: a printed number's unit
    # when it has none (seconds and bytes are the base units).
    ("tab_options", None, ("Re-exec., CPU",), 2, None, 0, 1e9, "WeightsLlama", 1e9),
    ("tab_options", None, ("Re-exec., CPU",), 4, None, 0, 1, "CpuReexecShort", 1),
    ("tab_options", None, ("Re-exec., CPU",), 5, None, 0, 1, "CpuReexecLong", 1),
    ("tab_options", None, ("Re-exec., GPU",), 4, None, 0, 1, "GpuIntShort", 1),
    ("tab_options", None, ("Re-exec., GPU",), 5, None, 0, 1, "GpuIntLong", 1),
    ("tab_options", None, ("llama.cpp",), 4, None, 0, 1, "CpuLlamaCppShort", 1),
    ("tab_options", None, ("llama.cpp",), 5, None, 0, 1, "CpuLlamaCppLong", 1),
    ("tab_options", None, (r"Ours, $\Kpre$",), 4, None, 0, 1, "CpuVerifyShort", 1),
    ("tab_options", None, (r"Ours, $\Kpre$",), 5, None, 0, 1, "CpuVerifyLong", 1),
    ("tab_options", None, ("with V1",), 5, None, 0, 1, "VOneVerifyLlama", 1),
    ("tab_eval_reexec", "CPU", (r"our $\Kpre$ verifier",), 1, None, 0, 1, "CpuVerifyShort", 1),
    ("tab_eval_reexec", "CPU", (r"our $\Kpre$ verifier",), 3, None, 0, 1, "CpuVerifyLong", 1),
    ("tab_eval_reexec", "CPU", ("fp32 re-execution",), 1, None, 0, 1, "CpuFloatShort", 1),
    ("tab_eval_reexec", "CPU", ("fp32 re-execution",), 3, None, 0, 1, "CpuFloatLong", 1),
    ("tab_eval_reexec", "CPU", ("integer re-execution",), 1, None, 0, 1, "CpuReexecShort", 1),
    ("tab_eval_reexec", "CPU", ("integer re-execution",), 3, None, 0, 1, "CpuReexecLong", 1),
    ("tab_eval_reexec", "CPU", ("llama.cpp Q8",), 1, None, 0, 1, "CpuLlamaCppShort", 1),
    ("tab_eval_reexec", "CPU", ("llama.cpp Q8",), 3, None, 0, 1, "CpuLlamaCppLong", 1),
    ("tab_eval_reexec", "CPU", ("llama.cpp bf16",), 1, None, 0, 1, "CpuLlamaCppBfShort", 1),
    ("tab_eval_reexec", "CPU", ("llama.cpp bf16",), 3, None, 0, 1, "CpuLlamaCppBfLong", 1),
    ("tab_eval_reexec", "GPU", ("Llama-2-7B",), 1, 0, 0, 1, "GpuIntShort", 1),
    ("tab_eval_reexec", "GPU", ("Llama-2-7B",), 1, 1, 0, 1, "GpuFloatShort", 1),
    ("tab_eval_reexec", "GPU", ("Llama-2-7B",), 3, 0, 0, 1, "GpuIntLong", 1),
    ("tab_eval_reexec", "GPU", ("Llama-2-7B",), 3, 1, 0, 1, "GpuFloatLong", 1),
    ("tab_eval_reexec", "GPU", ("OPT-1.3B",), 3, 1, 0, 1, "OptMGpuFloat", 1),
    ("tab_memory", None, (r"$\Kpre$", "64"), 2, None, 0, 1e9, "ClientPeakShortKpre", 1e9),
    ("tab_memory", None, (r"$\Kpre$", "2{,}048"), 2, None, 0, 1e9, "ClientPeakLongKpre", 1e9),
    ("tab_memory", None, (r"$\sC$", "64"), 2, None, 0, 1e9, "ClientPeakShortC", 1e9),
    ("tab_memory", None, (r"$\sC$", "2{,}048"), 2, None, 0, 1e9, "ClientPeakLongC", 1e9),
    ("tab_eval_long", None, ("16{,}384",), 3, None, 0, 1, "LongRatioLong", 1),
    ("tab_eval_long", None, ("16{,}384",), 4, None, 0, 1, "LongProofLong", 1),
    ("tab_eval_llm", "64-token", ("Llama-2-7B",), 3, None, 0, 1, "VerifyLlamaShortCCpu", 1),
    ("tab_eval_llm", "64-token", ("Llama-2-7B",), 4, None, 0, 1, "VerifyLlamaShortCGpu", 1),
    ("tab_eval_llm", "64-token", ("Llama-2-7B",), 5, None, 0, 1, "ProofLlamaShortC", 1),
    ("tab_eval_llm", "2{,}048-token", ("Llama-2-7B",), 5, None, 0, 1, "ProofLlamaLongC", 1),
    ("tab_eval_llm", "2{,}048-token", ("Llama-2-7B",), 7, None, 0, 1, "ProofLlamaLongK", 1),
    ("tab_eval_llm", "2{,}048-token", ("OPT-6.7B",), 3, None, 0, 1, "VerifyOptLLong", 1),
    ("tab_eval_llm", "2{,}048-token", ("OPT-6.7B",), 5, None, 0, 1, "ProofOptLLong", 1),
    ("tab_eval_v1", "T{=}64", ("GPT-2", r"$\sC$"), 4, None, 0, 1, "VOneRatioShortC", 1),
    ("tab_eval_v1", "T{=}64", ("GPT-2", r"$\Kpre$"), 4, None, 0, 1, "VOneRatioShortK", 1),
    ("tab_eval_v1", "T{=}2{,}048", ("OPT-125M", r"$\sC$"), 2, None, 0, 1e6, "CompareVOneOptS", 1e6),
    ("tab_eval_v1", "T{=}2{,}048", ("OPT-125M", r"$\sC$"), 5, 0, 0, 1, "VOneOptSProveVZero", 1),
    ("tab_eval_v1", "T{=}2{,}048", ("OPT-125M", r"$\sC$"), 5, 1, 0, 1, "VOneOptSProveVOne", 1),
    ("tab_eval_v1", "T{=}2{,}048", ("OPT-1.3B", r"$\sC$"), 2, None, 0, 1e6, "VOneOptMCVZeroGB", 1e9),
    ("tab_eval_v1", "T{=}2{,}048", ("OPT-1.3B", r"$\sC$"), 2, None, 0, 1e6, "CompareVOneOptM", 1e6),
    ("tab_eval_v1", "T{=}2{,}048", ("OPT-1.3B", r"$\sC$"), 3, None, 0, 1e6, "VOneOptMCVOneGB", 1e9),
    ("tab_eval_v1", "T{=}2{,}048", ("OPT-1.3B", r"$\Kpre$"), 2, None, 0, 1e6, "VOneOptMKVZeroGB", 1e9),
    ("tab_eval_v1", "T{=}2{,}048", ("OPT-1.3B", r"$\Kpre$"), 3, None, 0, 1e6, "VOneOptMKVOneGB", 1e9),
    ("tab_eval_v1", "T{=}2{,}048", ("Llama-2-7B", r"$\sC$"), 2, None, 0, 1e6, "VOneLlamaCVZeroGB", 1e9),
    ("tab_eval_v1", "T{=}2{,}048", ("Llama-2-7B", r"$\sC$"), 3, None, 0, 1e6, "VOneLlamaCVOneGB", 1e9),
    ("tab_eval_v1", "T{=}2{,}048", ("Llama-2-7B", r"$\sC$"), 4, None, 0, 1, "VOneLlamaCRatio", 1),
    ("tab_eval_v1", "T{=}2{,}048", ("Llama-2-7B", r"$\Kpre$"), 4, None, 0, 1, "VOneRatioLlama", 1),
    ("tab_eval_v1", "T{=}2{,}048", ("Llama-2-13B", r"$\sC$"), 4, None, 0, 1, "VOneRatioLargeC", 1),
    ("tab_v1", None, ("OPT-125M", "", r"$\sC$"), 5, None, 0, 1, "VOneOptSGkr", 1),
    ("tab_eval_compare_v1", None, ("zkLLM", "OPT-125M"), 3, None, 0, 1, "CompareVOneOptS", 1e6),
    ("tab_eval_compare_v1", None, ("zkLLM", "OPT-1.3B"), 3, None, 0, 1, "CompareVOneOptM", 1e6),
    ("tab_eval_compare", None, ("zkLLM", "Llama-2-13B"), 3, None, 1, 1, "ZkllmProverFactor", 1),
    ("tab_eval_real", None, (r"$\Kpre$", "2{,}048"), 2, 0, 0, 1e6, "RealOptMKLongReal", 1e9),
    ("tab_eval_real", None, (r"$\Kpre$", "2{,}048"), 2, 1, 0, 1e6, "RealOptMKLongRandom", 1e9),
    ("tab_eval_improve", "Native exact kernels", ("Llama-2-7B",), 3, None, 0, 1, "ImproveCpuv", 1),
    ("tab_eval_improve", "GPU claim decoder", ("OPT-1.3B",), 3, None, 0, 1, "ImproveGpuvOptM", 1),
    ("tab_eval_improve", "Threaded assembly", ("OPT-1.3B",), 2, None, 0, 1, "ImproveThreadedEncoder", 1),
    ("tab_setup", None, ("Llama-2-7B",), 1, None, 0, 1, "SetupGBLlamaA", 1e9),
    ("tab_setup", None, ("Llama-2-7B",), 2, None, 0, 1, "SetupGBLlamaB", 1e9),
    ("tab_setup", None, ("Llama-2-7B",), 3, None, 0, 1, "SetupHonestMBLlama", 1),
    ("tab_setup", None, ("Llama-2-7B",), 4, 0, 0, 1, "SetupUntrustedMBLlamaA", 1),
    ("tab_setup", None, ("Llama-2-7B",), 4, 1, 0, 1, "SetupUntrustedMBLlamaB", 1),
    ("tab_params", None, ("Columns $t_j$",), 1, 0, 0, 1, "ParamsColumnsMinInt", 1),
    ("tab_params", None, ("Columns $t_j$",), 1, 1, 0, 1, "ParamsColumnsMinFs", 1),
    ("tab_params", None, ("Opened columns per query ($t_j$",), 1, 0, 0, 1e6, "SetupUntrustedMBLlamaA", 1e6),
    ("tab_params", None, ("Opened columns per query ($t_j$",), 1, 1, 0, 1e6, "SetupUntrustedMBLlamaB", 1e6),
    ("tab_params", None, ("Setup proof",), 1, 0, 0, 1e9, "SetupGBLlamaA", 1e9),
    ("tab_params", None, ("Setup proof",), 1, 1, 0, 1e9, "SetupGBLlamaB", 1e9),
    ("tab_quality", None, ("OPT-125M",), 1, None, 0, 1, "QualityRefOptS", 1),
    ("tab_quality", None, ("OPT-125M",), 2, None, 0, 1, "QualityScalarOptS", 1),
    ("tab_quality", None, ("OPT-125M",), 3, None, 1, 1, "QualityRatioOptS", 1),
    ("tab_quality", None, ("OPT-1.3B",), 1, None, 0, 1, "QualityRefOptM", 1),
    ("tab_quality", None, ("OPT-1.3B",), 2, None, 0, 1, "QualityScalarOptM", 1),
    ("tab_quality", None, ("OPT-1.3B",), 3, None, 1, 1, "QualityRatioOptM", 1),
    ("tab_quality", None, ("OPT-6.7B",), 1, None, 0, 1, "QualityRefOptL", 1),
    ("tab_quality", None, ("OPT-6.7B",), 2, None, 0, 1, "QualityScalarOptL", 1),
    ("tab_quality", None, ("OPT-6.7B",), 3, None, 1, 1, "QualityRatioOptL", 1),
    ("tab_gen", None, ("GPT-2",), 2, None, 0, 1, "GenGptProveC", 1),
    ("tab_zeroshot", None, ("Mean",), 6, None, 0, 1, "ZeroshotMeanOptL", 1),
    ("tab_paths", None, ("Llama-2-7B",), 2, None, 0, 1, "SokPathsLlama", 1),
    ("tab_paths", None, ("OPT-6.7B",), 2, None, 0, 1, "SokPathsOptL", 1),
    ("tab_quality_attr", None, ("LayerNorm outputs (all)",), 1, None, 0, 1, "AttrNormLoo", 1),
)
"""The table cells and macros that print the same value of the same record: they agree up to one unit of the
coarser printed precision (a bound rounded up against a value rounded to nearest)."""
TWO_JOBS = (
    # (table, section, row, column, part, number, scale of the cell, macro, scale of the macro): one quantity that
    # two jobs measure; a ``WARNING`` line when they differ by more than 10%
    ("tab_eval_llm", "2{,}048-token", ("OPT-1.3B",), 2, None, 0, 1, "ServeOptMC", 1),
    ("tab_eval_llm", "64-token", ("Llama-2-7B",), 6, None, 0, 1, "CpuVerifyShort", 1),
    ("tab_eval_llm", "2{,}048-token", ("Llama-2-7B",), 6, None, 0, 1, "CpuVerifyLong", 1),
    ("tab_options", None, (r"Ours, $\sC$",), 2, None, 0, 1, "ProofLlamaShortC", 1),
    ("tab_options", None, (r"Ours, $\sC$",), 3, None, 0, 1, "ProofLlamaLongC", 1),
    ("tab_options", None, (r"Ours, $\sC$",), 4, None, 0, 1, "VerifyLlamaShortCCpu", 1),
    ("tab_options", None, (r"Ours, $\Kpre$",), 3, None, 0, 1, "ProofLlamaLongK", 1),
    (None, None, (), 0, None, 0, 1, "VOneVZeroVerifyLlama", 1, "CpuVerifyLong"),
)
"""Pairs of a table cell (or, with a macro name at the end, a second macro) and a macro that give one quantity from
two jobs: the serving harness's prover against the benchmark's, the benchmark's CPU verifier against perf.py's, v0's
timed queries of the V1 job against perf.py's."""
_NUMBER = re.compile(r"(?P<a>[-+]?\d+(?:\.\d+)?)(?:--(?P<b>[-+]?\d+(?:\.\d+)?))?"
                     r"(?:\s*\\,\s*(?P<u>ms|s|min|kB|MB|GB)(?![A-Za-z]))?")
_UNITS = {"ms": 1e-3, "s": 1.0, "min": 60.0, "kB": 1e3, "MB": 1e6, "GB": 1e9}


def printed(text: str | None, number: int = 0, scale: float = 1.0):
    """``(low, high, unit of the last digit)`` of the ``number``-th number (or range ``a--b``) printed in a cell or a
    macro, in seconds or bytes when it carries a unit (else times ``scale``); None for a missing one (``--``, a
    ``\\todo``)."""
    if text is None or "\\todo" in text:
        return None
    found = list(_NUMBER.finditer(text.replace("{,}", "").replace("$\\times$", " ")))
    if len(found) <= number:
        return None
    m = found[number]
    k = _UNITS[m["u"]] if m["u"] else scale

    def step(s):
        return (10.0 ** -len(s.split(".")[1]) if "." in s else 1.0) * k

    lo = float(m["a"]) * k
    hi = float(m["b"]) * k if m["b"] else lo
    return min(lo, hi), max(lo, hi), max(step(m["a"]), step(m["b"]) if m["b"] else 0.0)


def body_rows(text: str) -> list[tuple[str, list[str]]]:
    """``[(heading, cells)]`` of a table body (a ``.tex`` file of ``generated/`` or ``fallback/``): its rows split at
    ``\\\\``, a ``\\multicolumn`` spanning ``n`` columns repeated ``n`` times, the empty leading cells of a row taken
    from the row above, ``heading`` the last one-cell row above (a group's heading)."""
    body = " ".join(x for x in text.splitlines() if not x.lstrip().startswith("%"))
    out, heading, above = [], "", []
    for raw in body.split("\\\\"):
        row = re.sub(r"^\s*(\\midrule\s*)+", "", raw).strip()
        if not row:
            continue
        cells = []
        for cell in row.split("&"):
            m = re.match(r"\s*\\multicolumn\{(\d+)\}\{[^}]*\}\{(.*)\}\s*$", cell)
            cells += [m.group(2)] * int(m.group(1)) if m else [cell.strip()]
        if len(cells) == 1 or (len(set(cells)) == 1 and row.lstrip().startswith("\\multicolumn")):
            heading = cells[0]
            above = []
            continue
        lead = [c if c or i >= len(above) else above[i] for i, c in enumerate(cells[:3])]
        above = lead
        out.append((heading, lead + cells[3:]))
    return out


def _cell(tables: dict, spec) -> str | None:
    table, section, row, col, part = spec[:5]
    for heading, cells in body_rows(tables.get(table) or ""):
        if section is not None and section not in heading:
            continue
        if all(i < len(cells) and s in cells[i] for i, s in enumerate(row)) and col < len(cells):
            text = cells[col]
            if part is not None:
                halves = text.split(" / ")
                return halves[part] if part < len(halves) else None
            return text
    return None


def same_value(tables: dict, macros: dict) -> list[str]:
    """The pairs of :data:`SAME_VALUE` that disagree, as ``CHECK FAILED`` lines (``tables``: ``{file name: body}``;
    ``macros``: ``{name: printed value}``); a missing cell or macro is not a disagreement."""
    bad = []
    for spec in SAME_VALUE:
        table, _, row, col, part, number, cscale, macro, mscale = spec
        a, b = printed(_cell(tables, spec), number, cscale), printed(macros.get(macro), 0, mscale)
        if a is None or b is None:
            continue
        tol = max(a[2], b[2]) * (1 + 1e-9)
        if a[0] - tol > b[1] or b[0] - tol > a[1]:
            bad.append(f"CHECK FAILED cells and macros: {table} {'/'.join(row)} column {col}"
                       f"{'' if part is None else f' part {part}'} prints {_cell(tables, spec)!r}, \\{macro} "
                       f"{macros.get(macro)!r}")
    return bad


def two_jobs(tables: dict, macros: dict) -> list[str]:
    """``WARNING`` lines for the pairs of :data:`TWO_JOBS` whose values differ by more than 10%."""
    out = []
    for spec in TWO_JOBS:
        table, _, row, col, part, number, cscale, macro, mscale = spec[:9]
        other = spec[9] if len(spec) > 9 else None
        first = macros.get(other) if other else _cell(tables, spec)
        a, b = printed(first, 0 if other else number, mscale if other else cscale), printed(macros.get(macro), 0,
                                                                                            mscale)
        if a is None or b is None:
            continue
        x, y = (a[0] + a[1]) / 2, (b[0] + b[1]) / 2
        if max(abs(x), abs(y)) and abs(x - y) / max(abs(x), abs(y)) > 0.10:
            where = f"\\{other}" if other else f"{table} {'/'.join(row)} column {col}"
            out.append(f"WARNING two jobs: {where} prints {first!r}, \\{macro} {macros.get(macro)!r} (more than 10% "
                       "apart)")
    return out


def demote_text_claims(failed: list[str]) -> tuple[list[str], list[str]]:
    """``(failed, warnings)`` of a rehearsal: its small cells need not bear out what the text states about the final
    values (:data:`experiments.paper.macros.TEXT_CLAIM`), so those checks become warnings; every other check stays."""
    claims = [x for x in failed if x.startswith(M.TEXT_CLAIM)]
    return ([x for x in failed if not x.startswith(M.TEXT_CLAIM)],
            [x.replace("CHECK FAILED", "WARNING (rehearsal)", 1) for x in claims])


def folder_files(folder: Path) -> tuple[dict, dict]:
    """``({table file: body}, {macro: printed value})`` of a folder of the paper's files (``generated/``,
    ``fallback/``)."""
    tables = {name: (folder / f"{name}.tex").read_text(encoding="utf-8") for name in T.TABLE_FILES
              if (folder / f"{name}.tex").exists()}
    path = folder / "numbers.tex"
    text = path.read_text(encoding="utf-8") if path.exists() else ""
    return tables, dict(re.findall(r"^\\newcommand\{\\(\w+)\}\{(.*?)\}(?:\s+%.*)?$", text, re.M))


def build(args) -> int:
    logs = Path(args.logs)
    problems: list[str] = []
    recs = R.read_logs(logs, problems) if logs.is_dir() else []
    files = R.read_files(logs, problems) if logs.is_dir() else []
    run = R.Run(recs, files)
    failed = [f"CHECK FAILED {Path(r.path).name}:{r.line}: {msg}" for r in recs for msg in R.checks(r)]
    failed += R.log_checks(logs) if logs.is_dir() else []
    failed += R.cross_checks(run)
    tables_dir = Path(args.bench) / f"tables_{args.platform}"
    bench = Bench(tables_dir if tables_dir.is_dir() else None, prob_max=args.prob_max)
    policy, warnings = choose_plan(args.policy, record_plans(run, bench))
    bench.policy = policy
    failed += machine_check(run, bench)
    ctx = T.Ctx(run, bench, literature(), policy, platform_text(run, bench), not args.no_byte_model)
    tables = T.build_all(ctx, problems)
    macros, crossovers = M.build(ctx, failed)
    # a table cell and the macro printing the same value agree, in this build and in the paper's fallback
    texts = {tb.name: tb.text() for tb in tables}
    values = {n: m.value for n, m in macros.items() if m.value is not None}
    failed += same_value(texts, values)
    if (PAPER / "fallback").is_dir():
        failed += [x.replace("CHECK FAILED", "CHECK FAILED fallback:", 1)
                   for x in same_value(*folder_files(PAPER / "fallback"))]
    warnings += two_jobs(texts, values)
    warnings += run.warnings                # quality cells that two jobs measured with different values
    if args.rehearsal:
        failed, demoted = demote_text_claims(failed)
        warnings += demoted
    warnings += [f"WARNING {Path(r.path).name}:{r.line}: reexec.py's {r.d['variant']} is not recorded as bit-exact "
                 "with the integer reference: not an exact re-execution" for r in run.kind("reexec")
                 if r.d.get("variant") in T.CHECKED_EXACT and not T.exact_record(r.d["variant"], r)]

    report = [f"logs {logs}: {len(recs)} records from {len(list(logs.glob('sp-*.out'))) if logs.is_dir() else 0} "
              f"job logs, {len(files)} documents; bench tables {bench.name}; plan {policy}", ctx.platform]
    counts = defaultdict(lambda: defaultdict(int))
    for r in recs + files:
        counts[r.tier][r.kind] += 1
    for tier in sorted(counts, key=str):
        report.append(f"  tier {tier}: " + ", ".join(f"{k} {n}" for k, n in sorted(counts[tier].items())))
    missing_cells = 0
    for tb in tables:
        report.append(f"{tb.label}: {tb.filled} cells filled, {len(tb.missing)} missing")
        missing_cells += len(tb.missing)
        if args.verbose:
            report += [f"    missing: {m}" for m in tb.missing]
    absent = sorted(n for n, m in macros.items() if m.value is None)
    report.append(f"text numbers: {len(macros) - len(absent)} of {len(macros)} macros filled")
    report += [f"    missing: \\{n}" for n in absent] if args.verbose else []
    status = 0
    if args.rehearsal:
        for tier, kinds in R.REHEARSAL_KINDS.items():
            lacking = sorted(k for k in kinds if not counts[tier].get(k))
            if tier in ("bench", "batch") and not bench.dir:
                lacking.append(f"tables_{args.platform} (run comparison/aggregate.py --platform {args.platform})")
            if lacking:
                report.append(f"REHEARSAL: tier {tier} has no record of {lacking}")
                status = 1
    missing_jobs = []
    if args.expect:
        try:
            missing_jobs = job_coverage(expected_jobs(Path(args.expect)), logs, recs)
            report.append(f"jobs of {args.expect}: {len(missing_jobs)} without results")
            report += [f"    {name}" for name in missing_jobs]
        except (OSError, ValueError) as e:
            problems.append(f"PARSE --expect: {e}")
    report += warnings + problems + failed
    if problems or failed:
        status = 1
    if args.strict and not args.rehearsal and (missing_jobs or missing_cells or absent):
        report.append(f"STRICT: {missing_cells} table cells, {len(absent)} macros and {len(missing_jobs)} jobs "
                      "missing")
        status = 1
    print("\n".join(report))
    if args.check:
        return status
    if not recs and not files:            # never replace the paper's fallback values with an empty run
        print(f"no records in {logs}: nothing written", file=sys.stderr)
        return 1
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    header = f"logs {logs.name}, tables_{args.platform}; {ctx.platform}"
    for tb in tables:
        (out / f"{tb.name}.tex").write_text(tb.text(), encoding="utf-8", newline="\n")
    (out / "numbers.tex").write_text(M.text(macros, header), encoding="utf-8", newline="\n")
    with open(out / "cells.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=["table", "row", "column", "value", "printed", "source"])
        w.writeheader()
        for tb in tables:
            w.writerows(tb.cells)
        for name, m in sorted(macros.items()):
            w.writerow({"table": "numbers", "row": name, "column": "", "value": m.raw, "printed": m.value,
                        "source": m.source})
    (out / "coverage.txt").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
    written = [f"{len(tables)} tables and {len(macros)} macros to {out}"]
    if args.figures:
        from experiments.paper import figures
        if crossovers is not None:
            written.append(str(figures.breakeven(crossovers, Path(args.figures))))
        if args.schematics:
            written += [str(p) for p in figures.schematics(Path(args.figures))]
    print("wrote " + "; ".join(written))
    return status


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--platform", default="final", help="the run's platform name (run.sh --platform)")
    ap.add_argument("--logs", default=None, help="the run's folder logs/<name>/ (default ../logs/<platform>)")
    ap.add_argument("--policy", default=None,
                    help="mode C's commitment plan of the run (run.sh --policy; default: the plan the records name, "
                         f"else {DEFAULT_POLICY}); a plan the records do not name is a WARNING")
    ap.add_argument("--prob-max", type=int, default=DEFAULT_PROB_MAX,
                    help="the decoders' attention probabilities' scale of the run's benchmark cells (their tag "
                         f"pmax<n>; default {DEFAULT_PROB_MAX}, run.sh's; 255: the stored runs' 8-bit cells)")
    ap.add_argument("--bench", default=str(BENCH),
                    help="the folder of the benchmark's tables_<platform>/ and raw_<platform>/ (default "
                         "artifacts/comparison)")
    ap.add_argument("--out", default=str(PAPER / "generated"), help="where the paper's files are written")
    ap.add_argument("--figures", default=str(PAPER / "figures"),
                    help="where fig:breakeven is written once its inputs are in the run ('' for nowhere)")
    ap.add_argument("--schematics", action="store_true", help="also redraw the schematic figures into --figures")
    ap.add_argument("--expect", default=None, help="a dry-run listing of run.sh: report every job without results")
    ap.add_argument("--check", action="store_true", help="read, check and report; write nothing")
    ap.add_argument("--strict", action="store_true", help="exit status 1 on any missing cell, macro or job")
    ap.add_argument("--rehearsal", action="store_true",
                    help="a rehearsal's folder: every tier must have records of its kinds; no cell is expected")
    ap.add_argument("--no-byte-model", action="store_true",
                    help="skip the library's parameters and byte model (tab:params, tab:setup, the break-even)")
    ap.add_argument("--verbose", action="store_true", help="list every missing cell and macro")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.logs = args.logs or str(REPO / "logs" / args.platform)
    return build(args)


if __name__ == "__main__":
    sys.exit(main())

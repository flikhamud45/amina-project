"""The paper's tables and text numbers from the final run: read the run's records, check them, build every table
body and every text number, and write them for the paper.

    cd code
    python -m experiments.comparison.aggregate --platform final              # bench.py's records -> tables_final/
    python -m experiments.paper.numbers --logs ../logs/final --platform final --check            # report only
    python -m experiments.paper.numbers --logs ../logs/final --platform final                    # -> generated/
    python -m experiments.paper.numbers --logs ../logs/final --platform final --strict \\
        --expect ../logs/final/dryrun.txt                                    # fail on anything missing
    python -m experiments.paper.numbers --logs ../logs/rehearsal --platform rehearsal --rehearsal --check

Inputs (nothing here runs a model):

* ``--logs``: the run package's folder ``logs/<name>/`` (``experiments/package/run.sh``): the job logs
  ``sp-<cell>-<jobid>.out``, whose lines that start with ``{`` are the records (:mod:`experiments.paper.records`),
  and the documents of the attack tier (``attack/*.json``, ``attack/analytic.csv``) and of the tv tier
  (``sampler_tv_*.json``);
* ``--platform``: the benchmark's aggregated tables ``tables_<name>/`` and its raw records ``raw_<name>/`` (the
  outcome counts) in ``--bench`` (default ``artifacts/comparison``), :mod:`experiments.paper.bench`;
* the published systems' values (``artifacts/comparison/tables/reported_curated.csv``) and, for ``tab:params``,
  ``tab:setup`` and the break-even, the library's parameters and byte model (several minutes on a laptop CPU;
  ``--no-byte-model`` skips them).

Outputs (``--out``, default ``paper_sp2027/generated/``; ``--check`` writes nothing):

* ``<table>.tex`` for each table of :data:`experiments.paper.tables.TABLE_FILES`: the table's body rows (what sits
  between ``\\midrule`` and ``\\bottomrule`` in the paper), under a header comment with the run's protocol settings
  and sources;
* ``numbers.tex``: one ``\\newcommand`` per text number (:mod:`experiments.paper.macros`);
* ``cells.csv``: every table cell and every macro with its value, printed form and source record;
* ``coverage.txt``: the records per tier and kind, per table the cells filled and missing, the missing macros, the
  checks and, with ``--expect <dry-run listing>`` (``run.sh --platform <name> --dry-run [--optional] all``), every
  job of the package whose log is missing or has no record;
* with ``--figures`` (default ``paper_sp2027/figures``) ``ev_breakeven.pdf`` once its inputs are all in the run, and
  with ``--schematics`` also the schematic figures (:mod:`experiments.paper.figures`).

``paper_sp2027/main.tex`` inputs each file from ``generated/`` when it exists and from ``fallback/`` (the values the
paper prints today) otherwise.

Checks: an honest query rejected, a tamper accepted, a byte accounting, serving or A/B check that failed, a setup
proof rejected, the smoke's verdicts, records of more than one machine: each is a ``CHECK FAILED`` line; a log line
that starts with ``{`` and is not a record of a known kind is a ``PARSE`` line.  The exit status is 1 on either, and
with ``--strict`` also on any missing cell, macro or job.  ``--rehearsal``: a rehearsal's folder (small models; no
cell of the paper is expected): every tier must have at least one record of each kind it writes
(:data:`experiments.paper.records.REHEARSAL_KINDS`), every builder runs on what is there, and the checks apply.
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
from experiments.paper.bench import DEFAULT_POLICY, Bench, literature

REPO = CODE.parent
PAPER = REPO / "paper_sp2027"


def expected_jobs(listing: Path) -> dict:
    """``{job name: [the files its command writes with --out]}`` of a dry-run listing (``sbatch ... -J <name>``)."""
    jobs = {}
    for line in listing.read_text(encoding="utf-8").splitlines():
        m = re.search(r" -J (\S+)", line)
        if line.startswith("sbatch") and m:
            jobs[m.group(1)] = [Path(p).name for p in re.findall(r"--out (\S+\.(?:json|csv))", line)]
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


def build(args) -> int:
    logs = Path(args.logs)
    problems: list[str] = []
    recs = R.read_logs(logs, problems) if logs.is_dir() else []
    files = R.read_files(logs, problems) if logs.is_dir() else []
    run = R.Run(recs, files)
    failed = [f"CHECK FAILED {Path(r.path).name}:{r.line}: {msg}" for r in recs for msg in R.checks(r)]
    failed += R.log_checks(logs) if logs.is_dir() else []
    tables_dir = Path(args.bench) / f"tables_{args.platform}"
    bench = Bench(tables_dir if tables_dir.is_dir() else None, args.policy)
    failed += machine_check(run, bench)
    ctx = T.Ctx(run, bench, literature(), args.policy, platform_text(run, bench), not args.no_byte_model)
    tables = T.build_all(ctx, problems)
    macros, crossovers = M.build(ctx, failed)

    report = [f"logs {logs}: {len(recs)} records from {len(list(logs.glob('sp-*.out'))) if logs.is_dir() else 0} "
              f"job logs, {len(files)} documents; bench tables {bench.name}; plan {args.policy}", ctx.platform]
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
        missing_jobs = job_coverage(expected_jobs(Path(args.expect)), logs, recs)
        report.append(f"jobs of {args.expect}: {len(missing_jobs)} without results")
        report += [f"    {name}" for name in missing_jobs]
    report += problems + failed
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
    ap.add_argument("--policy", default=DEFAULT_POLICY, help="mode C's commitment plan of the run (run.sh --policy)")
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

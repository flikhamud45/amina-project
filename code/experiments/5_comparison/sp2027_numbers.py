"""Plan 2 F6 part 11 (H3): the S&P paper's tables and text numbers that the final run fills, from the run's records.

    cd code && python experiments/5_comparison/aggregate.py --platform l40s_sp2027     # bench.py's raw records
    cd code && python experiments/5_comparison/sp2027_numbers.py                       # -> ../paper_sp2027/generated/
    cd code && python experiments/5_comparison/sp2027_numbers.py --check               # parse, report, write nothing
    cd code && python experiments/5_comparison/sp2027_numbers.py --logs ../logs/rehearsal_2080ti \\
        --platform rehearsal_2080ti --rehearsal --check                                # a rehearsal's output folder
    cd code && python experiments/5_comparison/sp2027_numbers.py --stored --check      # the stored records (below)

Nothing here runs a model.  The inputs are the two outputs of ``slurm/sp2027.sh``:

* ``--logs`` (default ``../logs/l40s_sp2027``): every job log ``sp-<cell>-<jobid>.out`` (and any ``*.jsonl``); each
  line that starts with ``{`` is one record.  The tier comes from the cell's prefix (``c-`` cpuv, ``v1-``, ``serve-``,
  ``g-`` gpuv, ``gen-``, ``r-`` real, ``setup-``, ``q-`` quality, ``l-`` long, ``smoke``) and the record's kind from
  its fields: ``perf`` (``perf.py``), ``v1`` (``v1_bytes.py``), ``reexec`` (``reexec.py``), ``serve``
  (``serve_ab.py``), ``gen``/``setup``/``h2d`` (their ``kind``), ``ppl`` (``ppl_2048.py``, ``ppl_llama.py``).
* ``--platform`` (default ``l40s_sp2027``): bench.py's records of strong_gpu.sh's must and batch tiers, through the
  tables ``aggregate.py --platform <name>`` writes (``artifacts/comparison/tables_<name>/``), read with
  ``paper_assets``'s own selection code (``llm_cost``, ``Measured``) for the cells of ``paper_assets.OPTIMISED``
  (mode C's plan tag ``pol<--policy>``, default ``$PVI_C_POLICY`` or cnn18c, the package's default since PS2's
  decision), under ``--challenges`` (default fs).
  ``--base`` (default the course report's optimised runs l40s_improved,l40s_improved2,l40s_improved3) gives
  tab:eval-llm's and tab:eval-compare's base columns, interactive, exactly as the paper prints them today.

Outputs (``--out``, default ``../paper_sp2027/generated``; ``--check`` writes nothing): one ``<table>.tex`` per table
(the body rows, with a comment naming the sources and the protocol settings), ``numbers.tex`` (``\\newcommand``
macros for the text numbers, e.g. ``\\CpuRatioN``, the abstract's "at least N times"), ``cells.csv`` (every value
with its record's file) and ``coverage.txt``.  The coverage report lists, per table, the cells filled and those
still missing, and, with ``--expect <dry-run listing>`` (``PVI_DRYRUN=1 sp2027.sh all``), every job of the package
whose log is missing or holds no record (bench jobs: no ``done in`` line).  The checks: a record whose honest query
was rejected, whose tamper was accepted, whose byte accounting failed or whose serving check failed is a ``CHECK
FAILED`` line; an ``sp-`` log line that does not parse as one of the kinds is a ``PARSE`` error.  The exit status is 1
on either, and with ``--strict`` also on any missing cell or job.

``--rehearsal``: the inputs are a rehearsal's (``PVI_REHEARSAL=1``; small models, so no table cell of the paper is
expected); every tier must have at least one record of its kinds, every table builder runs on what is there, and the
checks above apply.  ``--stored``: the records stored in this repository -- the Xeon/2080 Ti records of V-LONG-a and
the V1 re-time (``artifacts/results/vlonga_v1retime_*.jsonl``), F2a's quality records (``artifacts/results/9_quality``),
A7's laptop serving records (``experiments/7_reexec/results``), the laptop perf records
(``experiments/6_improvements/results``), and as bench tables the course report's optimised runs, read both as the
base and as the "new" run (interactive, plan auto): the new columns must then equal the base ones, and the base
columns must equal the numbers tab:eval-llm prints today (``--paper``, default ``../paper_sp2027/sections``), which
shows the parsers regenerate the existing tables unchanged.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import os
import re
import statistics
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]
REPO = CODE.parent
sys.path.insert(0, str(HERE))

TIERS = {"smoke": "smoke", "c-": "cpuv", "v1-": "v1", "serve-": "serve", "g-": "gpuv", "gen-": "gen", "r-": "real",
         "setup-": "setup", "q-": "quality", "l-": "long"}
"""sp2027.sh's job-name prefixes (``sp-<prefix>...``) and their tiers."""
KINDS = {"smoke": {"v1", "h2d"}, "cpuv": {"perf", "reexec", "v1"}, "v1": {"v1"}, "serve": {"serve"},
         "gpuv": {"perf", "v1", "reexec", "h2d"}, "gen": {"gen"}, "real": {"perf", "v1"}, "setup": {"setup"},
         "quality": {"ppl"}, "long": {"perf", "reexec"}}
"""The record kinds each tier writes (the rehearsal checks that each tier has at least one record of its kinds)."""
EXACT = ("int", "int_fp32gemm", "float", "bf16")
"""reexec.py's exact variants (bf16 and fp32 re-execute the fp model exactly as it is; dynint8 computes another
model): the strongest exact baseline is the fastest of these (RX)."""
NAME = {"gpt2": "GPT-2", "opt-125m": "OPT-125M", "opt-1.3b": "OPT-1.3B", "opt-6.7b": "OPT-6.7B",
        "llama2-7b": "Llama-2-7B", "llama2-13b": "Llama-2-13B", "qwen3-4b": "Qwen3-4B"}
ORDER = ["gpt2", "opt-125m", "opt-1.3b", "qwen3-4b", "opt-6.7b", "llama2-7b", "llama2-13b"]
FULL_LAYERS = {"gpt2": 12, "opt-125m": 12, "opt-1.3b": 24, "opt-6.7b": 32, "llama2-7b": 32, "qwen3-4b": 36,
               "llama2-13b": 40}
LLM_ROWS = {64: ["gpt2", "qwen3-4b", "llama2-7b", "llama2-13b"],
            2048: ["opt-125m", "opt-1.3b", "opt-6.7b", "llama2-7b", "llama2-13b"]}
"""tab:eval-llm's rows (paper_assets.TAB_LLM_BLOCKS)."""
ZKLLM = ("opt-125m", "opt-1.3b", "llama2-7b", "llama2-13b")
"""tab:eval-compare's zkLLM rows (2,048 tokens), interactive as zkLLM."""
STORED_LOGS = [("artifacts/results/vlonga_v1retime_*.jsonl", None), ("artifacts/results/9_quality/*.jsonl", "quality"),
               ("experiments/7_reexec/results/serve_ab_*.jsonl", "serve"),
               ("experiments/6_improvements/results/perf_*.jsonl", "laptop")]
"""``--stored``: the repository's records and their tiers (None: perf records are cpuv, v1 records v1)."""
REPORT_RUNS = "l40s_improved,l40s_improved2,l40s_improved3"


# ================================================================================================ reading
@dataclass
class Rec:
    tier: str
    cell: str
    job: str
    path: str
    kind: str
    d: dict


def kind_of(d: dict) -> str:
    """The script that wrote one JSON record, from its fields."""
    if d.get("kind") in ("gen", "setup", "h2d"):
        return d["kind"]
    if d.get("row") in ("prove", "on_request"):
        return "serve"
    if "variant" in d and "full_s" in d:
        return "reexec"
    if "v1_total" in d or "codec_only" in d:
        return "v1"
    if "ppl" in d and "cell" in d:
        return "ppl"
    if {"prove_s", "verify_s", "timings", "mode"} <= set(d):
        return "perf"
    return "other"


def tier_of(cell: str) -> str | None:
    for prefix, tier in TIERS.items():
        if cell.startswith(prefix):
            return tier
    return None


LOG = re.compile(r"^sp-(?P<cell>.+)-(?P<job>[0-9]+|DRY[0-9]+)\.out$")


def read_logs(paths: list[tuple[str, str | None]], problems: list) -> list[Rec]:
    """Every JSON line of the files ``paths`` (``(path, tier or None)``; None: from the ``sp-`` file name, or from
    the record's kind for a stored file)."""
    out = []
    for path, forced in paths:
        name = os.path.basename(path)
        m = LOG.match(name)
        cell, job = (m["cell"], m["job"]) if m else (Path(name).stem, "")
        tier = forced or (tier_of(cell) if m else None)
        with open(path, encoding="utf-8", errors="replace") as fh:
            for i, line in enumerate(fh, 1):
                if not line.startswith("{"):
                    continue
                try:
                    d = json.loads(line)
                except json.JSONDecodeError as e:
                    problems.append(f"PARSE {path}:{i}: {e}")
                    continue
                kind = kind_of(d)
                t = tier or {"perf": "cpuv", "v1": "v1", "reexec": "cpuv"}.get(kind, "other")
                if m and kind == "other":
                    problems.append(f"PARSE {path}:{i}: a record of no known kind (keys {sorted(d)[:8]}...)")
                out.append(Rec(t, cell, job, path, kind, d))
    return out


def log_files(logs: Path) -> list[tuple[str, str | None]]:
    if not logs.is_dir():
        return []
    files = sorted(glob.glob(str(logs / "sp-*.out"))) + sorted(glob.glob(str(logs / "*.jsonl")))
    return [(f, None) for f in files]


def stored_files() -> list[tuple[str, str | None]]:
    return [(f, tier) for pattern, tier in STORED_LOGS for f in sorted(glob.glob(str(CODE / pattern)))]


# ================================================================================================ checks
def checks(r: Rec) -> list[str]:
    """The correctness flags of one record that failed (an honest query rejected, a tamper accepted, a byte
    accounting or serving check failed)."""
    d, bad = r.d, []
    if r.kind == "perf":
        if d.get("accepted") != d.get("queries"):
            bad.append(f"accepted {d.get('accepted')} of {d.get('queries')}")
        for k, t in (d.get("tampers") or {}).items():
            if t.get("rejected") != t.get("attempts"):
                bad.append(f"tamper {k}: {t.get('rejected')} of {t.get('attempts')} rejected")
    elif r.kind == "v1" and not d.get("codec_only"):
        if "accounting_ok" in d and not d["accounting_ok"]:
            bad.append("accounting_ok false")
        if d.get("v0_accepted") is False:
            bad.append("v0 query rejected")
        for k, t in (d.get("tampers") or {}).items():
            if t.get("rejected") != t.get("attempts"):
                bad.append(f"tamper {k}: {t.get('rejected')} of {t.get('attempts')} rejected")
    elif r.kind == "serve" and d.get("row") == "prove" and "skipped" not in d and d.get("ok") is False:
        bad.append("serving check failed (ok false)")
    elif r.kind == "gen":
        if d.get("ok") is False:
            bad.append("an honest response rejected or a tamper accepted")
    elif r.kind == "setup" and d.get("accepted") is False:
        bad.append("setup proof rejected")
    return bad


# ================================================================================================ formatting
def t(sec):
    """Seconds as the paper prints them (paper_assets.t): 76.1 ms, 712 ms, 1.8 s, 27.4 s, 2.9 min."""
    if sec is None:
        return "--"
    if sec < 1:
        return f"{sec * 1e3:.0f}\\,ms" if sec >= 0.0995 else f"{sec * 1e3:.1f}\\,ms"
    if sec < 100:
        return f"{sec:.1f}\\,s"
    return f"{sec / 60:.1f}\\,min"


def b(n):
    """Bytes as the paper prints them (paper_assets.b; MB = 10^6 bytes)."""
    if n is None:
        return "--"
    for unit, k in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= k:
            v = n / k
            return (f"{v:.1f}" if v < 100 else f"{v:.0f}") + f"\\,{unit}"
    return f"{n:.0f}\\,B"


def gb(n):
    return "--" if n is None else f"{n / 1e9:.1f}"


def x(v, digits=2):
    return "--" if v is None else f"{v:.{digits}f}$\\times$"


def spread(med, rng, fmt=t):
    """``median [min-max]``."""
    if med is None:
        return "--"
    if not rng:
        return fmt(med)
    return f"{fmt(med)} [{fmt(rng[0])}--{fmt(rng[1])}]"


def tokens(seq):
    return f"{seq:,}".replace(",", "{,}")


def macro_name(*parts) -> str:
    """A LaTeX macro name from words and numbers (digits spelled: macro names have letters only)."""
    digits = {"0": "Zero", "1": "One", "2": "Two", "3": "Three", "4": "Four", "5": "Five", "6": "Six", "7": "Seven",
              "8": "Eight", "9": "Nine"}
    s = "".join(str(p).replace("-", " ").replace(".", " ").replace("_", " ").title().replace(" ", "") for p in parts)
    return "".join(digits.get(c, c) for c in s if c.isalnum())


# ================================================================================================ tables
@dataclass
class Table:
    name: str
    note: str
    rows: list = field(default_factory=list)
    cells: list = field(default_factory=list)      # dicts for cells.csv
    missing: list = field(default_factory=list)
    filled: int = 0

    def value(self, row: str, col: str, v, src: str = "", fmt=None):
        """One cell: recorded in cells.csv and counted as filled or missing; returns its printed form."""
        self.cells.append({"table": self.name, "row": row, "column": col, "value": v, "source": src})
        if v is None:
            self.missing.append(f"{row} / {col}")
        else:
            self.filled += 1
        return (fmt or str)(v) if v is not None else "--"


def by(recs, kind, tier=None):
    return [r for r in recs if r.kind == kind and (tier is None or r.tier == tier)]


def model_rank(m):
    return ORDER.index(m) if m in ORDER else len(ORDER)


def perf_total(d: dict, which: str):
    """perf.py's median total and its [min, max] (older records have no range)."""
    return d.get(f"{which}_s"), d.get(f"{which}_s_range")


def v1_timed(d: dict, proto: str, path: str | None = None):
    """v1_bytes.py's timed medians of one protocol (``v0``, ``v1``, or a V1 path ``"0"``/``"1"``), each party's
    Fiat-Shamir hashing added to both (as perf.py counts it): ``(prove, verify, prove range, verify range)``."""
    tm = d.get("timed") or {}
    rec = (tm.get("v1_paths") or {}).get(path) if path is not None else tm.get(proto)
    if not rec:
        return None, None, None, None
    fs = rec.get("fs_s", 0.0) or 0.0

    def add(r):
        return [r[0] + fs, r[1] + fs] if r else None
    return rec["prove_s"] + fs, rec["verify_s"] + fs, add(rec.get("prove_s_range")), add(rec.get("verify_s_range"))


# ------------------------------------------------------------------------------------------ bench tables
def bench_tables(pa, names: list[str]) -> list[Path]:
    return [p for p in (pa.BASE / f"tables_{n}" for n in names) if p.is_dir()]


def llm_sel(pa, model: int, seq: int, mode: str, device: str, policy: str):
    """paper_assets.OPTIMISED's selection of one cell, with mode C's plan tag ``pol<policy>``."""
    spec = pa.llm_spec(model, seq, mode, device)
    if spec is None:
        return None
    tags = tuple("pol" + policy if tg.startswith("pol") else tg for tg in spec["tags"])
    return dict(mode=mode, lam=spec["lam"], threads=spec["threads"], tags=tags)


def llm_value(pa, dirs, model, seq, mode, device, policy, challenges):
    """(prove, verify, bytes, params) of one optimised cell in the runs ``dirs``, or None."""
    if not dirs:
        return None
    sel = llm_sel(pa, model, seq, mode, device, policy)
    if sel is None:
        return None
    return pa.llm_cost((model, seq), sel, tables=dirs, challenges=challenges, full_only=True)


def new_value(pa, dirs, model, seq, metric, mode, policy, challenges):
    """One number of the new run, as ``paper_assets.opt_llm`` picks it (the defining cells of ``COLUMN_CELLS``, in
    order: the prover and the proof from the CPU-verifier cell, else the GPU one), with mode C's plan tag
    ``pol<policy>``, under ``challenges``; no stand-ins.  None if not measured."""
    if not dirs:
        return None
    for device in pa.COLUMN_CELLS["gpu" if metric == "gpu" else metric]:
        sel = llm_sel(pa, model, seq, mode, device, policy)
        c = pa.llm_cost((model, seq), sel, tables=dirs, challenges=challenges, full_only=True) if sel else None
        if c is not None and c[pa.LLM_METRIC[metric]] is not None:
            return c[pa.LLM_METRIC[metric]]
    return None


def tab_eval_llm(pa, base, new, policy, challenges) -> tuple[Table, Table]:
    """tab:eval-llm in the paper's layout (Model, Params, C prove base/new, C CPU verify base, C GPU verify base/new,
    proof, Kpre CPU verify base, Kpre proof), and ``tab_eval_llm_full``, which adds the new CPU verifiers.  The base
    columns are ``paper_assets.opt_llm`` on ``base`` (interactive): the numbers the paper prints.  The proof columns
    are the new run's (its protocol settings), else the base's."""
    tb = Table("tab_eval_llm", f"base: {[p.name for p in base]} (interactive, plan auto); new: {[p.name for p in new]} "
               f"({challenges}, plan {policy}); proof columns: new where measured, else base")
    full = Table("tab_eval_llm_full", tb.note + "; columns: C prove base/new, C CPU verify base/new, C GPU verify "
                 "base/new, C proof, Kpre CPU verify base/new, Kpre proof")
    pa.OPT_TABLES = base

    def old(m, seq, metric, mode):
        return pa.opt_llm(m, seq, metric, mode, "int")[0] if base else None

    for seq, models in LLM_ROWS.items():
        head = rf"\multicolumn{{10}}{{@{{}}l}}{{\emph{{{tokens(seq)}-token prompts}}}} \\"
        tb.rows.append(head)
        full.rows.append(head.replace("{10}", "{12}"))
        for m in models:
            row = f"{NAME[m]} ({seq})"
            v = {("C", k, "base"): old(m, seq, k, "C") for k in ("prove", "verify", "gpu", "bytes")}
            v.update({("Kpre", k, "base"): old(m, seq, k, "Kpre") for k in ("verify", "bytes")})
            v.update({("C", k, "new"): new_value(pa, new, m, seq, k, "C", policy, challenges)
                      for k in ("prove", "verify", "gpu", "bytes")})
            v.update({("Kpre", k, "new"): new_value(pa, new, m, seq, k, "Kpre", policy, challenges)
                      for k in ("verify", "bytes")})
            n = pa.opt_llm(m, seq, "params")[0]
            proof_c = v["C", "bytes", "new"] if v["C", "bytes", "new"] is not None else v["C", "bytes", "base"]
            proof_k = v["Kpre", "bytes", "new"] if v["Kpre", "bytes", "new"] is not None else v["Kpre", "bytes", "base"]
            cols = [("C prove base", v["C", "prove", "base"], t), ("C prove new", v["C", "prove", "new"], t),
                    ("C CPU verify base", v["C", "verify", "base"], t), ("C CPU verify new", v["C", "verify", "new"], t),
                    ("C GPU verify base", v["C", "gpu", "base"], t), ("C GPU verify new", v["C", "gpu", "new"], t),
                    ("C proof", proof_c, b), ("Kpre CPU verify base", v["Kpre", "verify", "base"], t),
                    ("Kpre CPU verify new", v["Kpre", "verify", "new"], t), ("Kpre proof", proof_k, b)]
            short = ("C prove base", "C prove new", "C CPU verify base", "C GPU verify base", "C GPU verify new",
                     "C proof", "Kpre CPU verify base", "Kpre proof")
            tb.rows.append(" & ".join([NAME[m], pa.params(n) if n else "--"] + [
                tb.value(row, c, val, "new" if c.endswith("new") else "base", f) for c, val, f in cols if c in short])
                + r" \\")
            full.rows.append(" & ".join([NAME[m], pa.params(n) if n else "--"] + [
                full.value(row, c, val, "new" if c.endswith("new") else "base", f) for c, val, f in cols]) + r" \\")
    return tb, full


def tab_eval_compare(pa, new, policy, challenges, tag="new") -> Table:
    """tab:eval-compare's "ours" columns (prover, verifier, proof) from one run: the image model with committed
    weights under Fiat-Shamir, GPT-2 and Llama-2-7B (1 token) under Fiat-Shamir, Maverick's Qwen3-4B setting (Kpre,
    lambda 40, one thread, interactive), zkLLM's rows interactive with the streaming GPU verifier."""
    tb = Table("tab_eval_compare", f"ours from {[p.name for p in new]} ({tag}; plan {policy}); zkLLM and Maverick rows "
               "interactive; the others under " + challenges)
    if not new:
        tb.missing.append("no bench tables")
        return tb
    M = pa.Measured(new)
    pol = "pol" + policy
    cnn = M.find_cnn("vgg16", "C", challenges, 128, ("wire", pol))
    rows = [("VGG-16", M.cost("vgg16", cnn) if cnn else None)]
    for seq in (64, 512):
        rows.append((f"GPT-2 ({seq})", llm_value(pa, new, "gpt2", seq, "C", "cpu", policy, challenges)))
    rows.append(("Llama-2-7B (1)", llm_value(pa, new, "llama2-7b", 1, "C", "cpu", policy, challenges)))
    for tags in (("thr1", "wire", "prune", "lookups"), ("thr1", "prune", "lookups")):
        v = pa.llm_cost(("qwen3-4b", 8), dict(mode="Kpre", lam=40, threads="1", tags=tags), tables=new,
                        challenges="int", full_only=True)
        rows.append((f"Qwen3-4B (8){'' if 'wire' in tags else ', no encoding'}", v))
    for m in ZKLLM:                    # the prover and the proof from the CPU cell, else the GPU one (COLUMN_CELLS)
        cpu = llm_value(pa, new, m, 2048, "C", "cpu", policy, "int")
        gpu = llm_value(pa, new, m, 2048, "C", "gpu", policy, "int")
        pick = lambda k: next((c[k] for c in (cpu, gpu) if c and c[k] is not None), None)   # noqa: E731
        rows.append((f"{NAME[m]} (2{{,}}048)", None if not (cpu or gpu) else (pick(0), gpu[1] if gpu else None,
                                                                             pick(2))))
    for label, v in rows:
        g = lambda k: v[k] if v else None       # noqa: E731
        tb.rows.append(" & ".join([label, tb.value(label, "prover", g(0), tag, t), tb.value(label, "verifier", g(1), tag, t),
                                   tb.value(label, "proof", g(2), tag, b)]) + r" \\")
    return tb


def tab_batch(pa, new) -> Table:
    """C4a: per-query bytes, prover and verifier time of B prompts against one opening (bench.py --batches), from
    the run's measured_summary.csv (the batch group; B = 1: the cell's own queries)."""
    tb = Table("tab_batch", f"from {[p.name for p in new]}: per query = the batch's median / B")
    rows = defaultdict(dict)
    for d in new:
        path = d / "measured_summary.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["suite"] != "llm" or "_batch" not in r["cell"] or not r["cell"].startswith("defence_"):
                    continue
                bsz = int(r["batch"]) if r.get("batch") else 1
                rows[(r["model"], r["cell"], bsz)][r["metric"]] = float(r["median"])
    cell_re = re.compile(r"^defence_(C|K|Kpre)_(int|fs)_lam(\d+)_T(\d+)_L(\d+)")
    for (model, cell, bsz), m in sorted(rows.items(), key=lambda kv: (model_rank(kv[0][0]), kv[0][1], kv[0][2])):
        c = cell_re.match(cell)
        if not c or "bytes_total" not in m:
            continue
        mode, ch, _lam, seq, _L = c.groups()
        prove = sum(m.get(k, 0.0) for k in pa.PROVE)
        verify = sum(m.get(k, 0.0) for k in pa.VERIFY)
        row = f"{NAME.get(model, model)} ({seq}) {mode}:{ch} B={bsz}"
        tb.rows.append(" & ".join([NAME.get(model, model), tokens(int(seq)), f"{mode}" + (" (int.)" if ch == "int" else ""),
                                   str(bsz), tb.value(row, "MB/query", m["bytes_total"] / bsz, cell, b),
                                   tb.value(row, "prove/query", prove / bsz, cell, t),
                                   tb.value(row, "verify/query", verify / bsz, cell, t)]) + r" \\")
    if not tb.rows:
        tb.missing.append("no batch cells")
    return tb


def tab_bench_memory(pa, new) -> Table:
    """F4 for the bench cells: peak GPU and host memory of every full-model LLM cell (bench.py's records)."""
    tb = Table("tab_memory_bench", f"from {[p.name for p in new]}: gpu_peak_memory and host_peak_rss of each cell (GB)")
    rows = defaultdict(dict)
    for d in new:
        path = d / "measured_summary.csv"
        if not path.exists():
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r["suite"] == "llm" and r["metric"] in ("gpu_peak_memory", "host_peak_rss") and not r.get("batch") \
                        and r["cell"].startswith("defence_") and r.get("cfg_n_layers") == r.get("cfg_n_layers_full"):
                    rows[(r["model"], r["cell"])][r["metric"]] = float(r["median"])
    for (model, cell), m in sorted(rows.items(), key=lambda kv: (model_rank(kv[0][0]), kv[0][1])):
        row = f"{model} {cell}"
        tb.rows.append(" & ".join([NAME.get(model, model), cell.replace("_", r"\_"),
                                   tb.value(row, "GPU GB", m.get("gpu_peak_memory"), cell, gb),
                                   tb.value(row, "host GB", m.get("host_peak_rss"), cell, gb)]) + r" \\")
    return tb


# ------------------------------------------------------------------------------------------ the run's logs
def reexec_full(recs, model, seq, device, variants=EXACT):
    """{variant: (full-model seconds measured at full depth, range, record)} of reexec.py on ``device``."""
    out = {}
    for r in by(recs, "reexec"):
        d = r.d
        if d.get("model") != model or d.get("seq") != seq or not str(d.get("device", "")).startswith(device):
            continue
        if d.get("variant") not in variants:
            continue
        full = str(d.get("n_layers"))
        if full not in d.get("measured_s", {}):      # only a measured full model counts (no extrapolation)
            continue
        v = d["measured_s"][full]
        if d["variant"] not in out or v < out[d["variant"]][0]:   # the faster allocator setting (c-rx-heap)
            out[d["variant"]] = (v, (d.get("range_s") or {}).get(full), r)
    return out


def perf_cell(recs, tier, model, seq, mode, fs=True, vdev="cpu", layers=None, label=None):
    """The perf.py record of one cell (the last one if several), or None."""
    hit = None
    for r in by(recs, "perf", tier):
        d = r.d
        if (d.get("model"), d.get("seq"), d.get("mode"), bool(d.get("fs")), d.get("verifier_device", "cpu"),
                d.get("layers")) == (model, seq, mode, fs, vdev, layers) and (label is None or d.get("label") == label):
            hit = r
    return hit


def tab_options(recs, macros) -> Table:
    """tab:options' measured rows and the abstract's N: per model and length, the CPU verifier of the full model (Kpre
    and C, Fiat-Shamir) against the strongest exact re-execution on the same threads (RX), the int8 GPU re-execution,
    and V1's byte and CPU-verifier ratios (cpuv's V1 cells, new verifier path).  N = the floor of the smallest
    Kpre ratio over the lengths, for Llama-2-7B (and per model in cells.csv)."""
    tb = Table("tab_options", "cpuv (perf.py --fs, reexec.py on the CPU), gpuv (reexec.py int on the GPU), cpuv V1 "
               "cells; ratios = strongest exact baseline / CPU verifier")
    ratios = defaultdict(dict)
    for m in sorted({r.d.get("model") for r in by(recs, "perf", "cpuv")} | {"llama2-7b"}, key=model_rank):
        for seq in (64, 2048):
            row = f"{NAME.get(m, m)} ({seq})"
            base = reexec_full(recs, m, seq, "cpu")
            best = min(base.items(), key=lambda kv: kv[1][0]) if base else None
            gpu = reexec_full(recs, m, seq, "cuda", ("int",)).get("int")
            k = perf_cell(recs, "cpuv", m, seq, "Kpre")
            c = perf_cell(recs, "cpuv", m, seq, "C")
            kv = k.d["verify_s"] if k else None
            ratio = best[1][0] / kv if best and kv else None
            if ratio:
                ratios[m][seq] = ratio
            v1 = next((r for r in by(recs, "v1", "cpuv") if (r.d.get("model"), r.d.get("seq"), r.d.get("mode"))
                       == (m, seq, "Kpre") and r.d.get("timed")), None)
            v1_bytes = v1.d["v0_total"] / v1.d["v1_total"] if v1 else None
            v1_time = None
            if v1:
                v0p, v0v, _, _ = v1_timed(v1.d, "v0")
                _, v1v, _, _ = v1_timed(v1.d, "v1")
                v1_time = v1v / v0v if v0v and v1v else None
            src = lambda r: os.path.basename(r.path) if r else ""     # noqa: E731
            tb.rows.append(" & ".join([
                NAME.get(m, m), tokens(seq),
                tb.value(row, "strongest exact re-execution (CPU)", best[1][0] if best else None,
                         f"{best[0]} {src(best[1][2])}" if best else "", t),
                tb.value(row, "int8 re-execution (GPU)", gpu[0] if gpu else None, src(gpu[2]) if gpu else "", t),
                tb.value(row, "C proof", c.d["proof_bytes"] if c else None, src(c), b),
                tb.value(row, "C CPU verifier", c.d["verify_s"] if c else None, src(c), t),
                tb.value(row, "Kpre proof", k.d["proof_bytes"] if k else None, src(k), b),
                tb.value(row, "Kpre CPU verifier", kv, src(k), t),
                tb.value(row, "ratio", ratio, "", x),
                tb.value(row, "V1 bytes ratio", v1_bytes, src(v1), x),
                tb.value(row, "V1 / v0 CPU verifier", v1_time, src(v1), x)]) + r" \\")
    for m, rs in ratios.items():
        if rs:
            macros[macro_name("CpuRatioMin", NAME.get(m, m))] = f"{min(rs.values()):.2f}"
            for seq, r in rs.items():
                macros[macro_name("CpuRatio", NAME.get(m, m), seq)] = f"{r:.1f}"
    if ratios.get("llama2-7b") and len(ratios["llama2-7b"]) == 2:
        macros["CpuRatioN"] = str(math.floor(min(ratios["llama2-7b"].values())))
    else:
        tb.missing.append("CpuRatioN (Llama-2-7B at 64 and 2,048 tokens, cpuv)")
    return tb


def tab_eval_reexec(recs) -> Table:
    """tab:eval-reexec: on the GPU, the integer forward pass with int8 GEMMs / the fp16 decoder (full models,
    measured); on the CPU (Llama-2-7B), our Kpre verifier and the fp32, integer and strongest exact re-execution."""
    tb = Table("tab_eval_reexec", "gpuv reexec.py (int / float = fp16 on the GPU, full models); cpuv perf.py and "
               "reexec.py (8 threads)")
    tb.rows.append(r"\multicolumn{4}{@{}l}{\emph{GPU: integer forward pass / fp16 decoder}} \\")
    for m in ["opt-1.3b", "opt-6.7b", "llama2-7b", "llama2-13b", "qwen3-4b"]:
        cells = []
        for seq in (64, 512, 2048):
            g = reexec_full(recs, m, seq, "cuda", ("int", "float"))
            i, f = g.get("int"), g.get("float")
            cells.append(f"{tb.value(f'{m} {seq}', 'GPU int', i[0] if i else None, 'gpuv', lambda v: f'{v:.2f}')} / "
                         f"{tb.value(f'{m} {seq}', 'GPU fp16', f[0] if f else None, 'gpuv', lambda v: f'{v:.3g}')}")
        tb.rows.append(" & ".join([NAME[m]] + cells) + r" \\")
    tb.rows.append(r"\midrule")
    tb.rows.append(r"\multicolumn{4}{@{}l}{\emph{CPU: Llama-2-7B}} \\")
    fmt = lambda v: f"{v:.3g}"      # noqa: E731
    ver = [perf_cell(recs, "cpuv", "llama2-7b", s, "Kpre") for s in (64, 2048)]
    tb.rows.append(" & ".join(["our $\\Kpre$ verifier (Fiat--Shamir)",
                               tb.value("cpu 64", "Kpre verifier", ver[0].d["verify_s"] if ver[0] else None, "cpuv", fmt),
                               "--", tb.value("cpu 2048", "Kpre verifier", ver[1].d["verify_s"] if ver[1] else None,
                                              "cpuv", fmt)]) + r" \\")
    for label, variants in (("fp32 re-execution", ("float",)), ("bf16 re-execution", ("bf16",)),
                            ("integer re-execution (int8 GEMMs)", ("int",)),
                            ("integer re-execution (fp32 GEMMs)", ("int_fp32gemm",)),
                            ("strongest exact re-execution", EXACT), ("dynamic int8 (not exact)", ("dynint8",))):
        vals = []
        for seq in (64, 2048):
            g = reexec_full(recs, "llama2-7b", seq, "cpu", variants)
            vals.append(min(v[0] for v in g.values()) if g else None)
        tb.rows.append(" & ".join([label, tb.value("cpu 64", label, vals[0], "cpuv", fmt), "--",
                                   tb.value("cpu 2048", label, vals[1], "cpuv", fmt)]) + r" \\")
    return tb


def tab_eval_v1(recs, macros) -> Table:
    """tab:eval-v1: V1 against v0 on the same queries (v1 tier, and V1 cells of other tiers when the v1 tier lacks
    the cell): bytes, ratio, prover and verifier v0 / V1 (median; Fiat-Shamir hashing in both parties' time); the
    bandwidth crossover per cell, 8 dB / dt bits per second, as a macro per cell."""
    tb = Table("tab_eval_v1", "v1_bytes.py --full --fs records; times: median over the timed queries")
    seen = {}
    for r in sorted(by(recs, "v1"), key=lambda r: r.tier != "v1"):
        d = r.d
        if d.get("codec_only") or d.get("real") or "v1_total" not in d:
            continue
        key = (d["seq"], d["model"], d["layers"], d["mode"])
        seen.setdefault(key, r)
    for (seq, m, layers, mode), r in sorted(seen.items(), key=lambda kv: (kv[0][0], model_rank(kv[0][1]), kv[0][3])):
        d = r.d
        row = f"{NAME.get(m, m)} {layers}/{FULL_LAYERS.get(m, '?')} T={seq} {mode}"
        p0, v0, _, _ = v1_timed(d, "v0")
        p1, v1, _, _ = v1_timed(d, "v1")
        two = lambda a, c: "--" if a is None or c is None else f"{a:.2f} / {c:.2f}"    # noqa: E731
        tb.value(row, "v0 prover", p0, r.path)
        tb.value(row, "V1 prover", p1, r.path)
        tb.value(row, "v0 verifier", v0, r.path)
        tb.value(row, "V1 verifier", v1, r.path)
        tb.rows.append(" & ".join([f"{NAME.get(m, m)}, {layers}/{FULL_LAYERS.get(m, '?')}", tokens(seq),
                                   "$\\sC$" if mode == "C" else "$\\Kpre$",
                                   tb.value(row, "v0 MB", d["v0_total"] / 1e6, r.path, lambda v: f"{v:,.1f}".replace(",", "{,}")),
                                   tb.value(row, "V1 MB", d["v1_total"] / 1e6, r.path, lambda v: f"{v:,.1f}".replace(",", "{,}")),
                                   tb.value(row, "ratio", d["v0_total"] / d["v1_total"], r.path, x),
                                   two(p0, p1), two(v0, v1)]) + r" \\")
        if None not in (p0, p1, v0, v1) and (p1 + v1) > (p0 + v0):
            gbps = 8 * (d["v0_total"] - d["v1_total"]) / ((p1 + v1) - (p0 + v0)) / 1e9
            macros[macro_name("Crossover", NAME.get(m, m), f"L{layers}", seq, mode)] = f"{gbps:.2f}"
    if not seen:
        tb.missing.append("no V1 records")
    return tb


def tab_serve(recs, macros) -> Table:
    """A7's table (extended version): per model, prompt length T and batch B, mode and batch proof: U2 (fp16), U1,
    the prover (ms, median [95% CI] in cells.csv), prover/U1, prover/U2, proof MB per token, the egress cap."""
    tb = Table("tab_serve", "serve_ab.py prove rows (U2: fp16 if measured, else the first dtype)")
    lines = [r for r in by(recs, "serve") if r.d.get("row") == "prove"]
    for r in sorted(lines, key=lambda r: (model_rank(r.d.get("model")), r.d.get("n_layers"), r.d["seq"], r.d["batch"],
                                          r.d["mode"], r.d["batch_proof"])):
        d = r.d
        row = f"{d['model']} L{d.get('n_layers')} T={d['seq']} B={d['batch']} {d['mode']}/{d['batch_proof']}"
        if "skipped" in d:
            tb.rows.append(f"% {row}: skipped ({d['skipped']})")
            continue
        u2k = "fp16" if "fp16" in (d.get("u2_s") or {}) else next(iter(d.get("u2_s") or {}), None)
        med = lambda rec, k="median": rec[k] if rec else None     # noqa: E731
        ms = lambda v: f"{v * 1e3:.4g}"                            # noqa: E731
        tb.rows.append(" & ".join([
            NAME.get(d["model"], d["model"]) + ("" if d.get("layers") is None else f" ({d['n_layers']} bl.)"),
            tokens(d["seq"]), str(d["batch"]), f"{d['mode']}, {d['batch_proof']}",
            tb.value(row, "U2 ms", med((d.get("u2_s") or {}).get(u2k)) if u2k else None, r.path, ms),
            tb.value(row, "U1 ms", med(d.get("u1_s")), r.path, ms),
            tb.value(row, "prover ms", med(d.get("prove_s")), r.path, ms),
            tb.value(row, "P/U1", med(d.get("prover_over_u1"), "ratio"), r.path, lambda v: f"{v:.2f}"),
            tb.value(row, "P/U2", med((d.get("prover_over_u2") or {}).get(u2k), "ratio") if u2k else None, r.path,
                     lambda v: f"{v:.1f}"),
            tb.value(row, "MB/token", d.get("proof_mb_per_token"), r.path, lambda v: f"{v:.3g}"),
            tb.value(row, "egress cap tok/s", d.get("egress_cap_tokens_per_s_per_100gbps"), r.path,
                     lambda v: f"{v:,.0f}".replace(",", "{,}"))]) + r" \\")
        if d.get("layers") is None and d["batch"] == 1:      # B = 1 (serve_ab.py labels it "single")
            name = macro_name("Serve", NAME.get(d["model"], d["model"]), d["seq"], d["mode"])
            macros[name + "OverUOne"] = f"{d['prover_over_u1']['ratio']:.1f}"
            if u2k:
                macros[name + "OverUTwo"] = f"{d['prover_over_u2'][u2k]['ratio']:.1f}"
    if not lines:
        tb.missing.append("no serve records")
    return tb


def tab_gen(recs, macros) -> Table:
    """tab:gen-full (final run): per model, mode, rule and G: prover, verifier (token rule included), proof, proof
    per generated token, and the two tamper cells' labels."""
    tb = Table("tab_gen", "gen_bench.py: P-token prompt, G-token response proved as one prefill; median [min-max]")
    for r in sorted(by(recs, "gen"), key=lambda r: (model_rank(r.d["model"]), r.d["mode"], r.d["decode"], r.d["gen"])):
        d = r.d
        row = f"{d['model']} {d['mode']} {d['decode']} P={d['prompt']} G={d['gen']}"
        tam = ", ".join(f"{k}: {v['label']}" for k, v in (d.get("tampers") or {}).items())
        tb.rows.append(" & ".join([
            NAME.get(d["model"], d["model"]), "$\\sC$" if d["mode"] == "C" else "$\\Kpre$", d["decode"], str(d["gen"]),
            tb.value(row, "prover", d["prove_s"], r.path, lambda v: spread(v, d.get("prove_s_range"))),
            tb.value(row, "verifier", d["verify_s"], r.path, lambda v: spread(v, d.get("verify_s_range"))),
            tb.value(row, "proof", d["proof_bytes"], r.path, b),
            tb.value(row, "MB/token", d["proof_mb_per_token"], r.path, lambda v: f"{v:.2f}")]) + rf" \\ % {tam}")
        if d.get("layers") is None:
            macros[macro_name("Gen", NAME.get(d["model"], d["model"]), d["mode"], d["decode"], "G", d["gen"],
                              "MsPerToken")] = f"{1e3 * d['prove_s'] / d['gen']:.1f}"
    if not tb.rows:
        tb.missing.append("no gen records")
    return tb


def tab_memory(recs) -> Table:
    """F4 (C-Q4): peak GPU and host memory of every timed cell of the run's logs (perf.py, v1_bytes.py, reexec.py,
    gen_bench.py), in GB."""
    tb = Table("tab_memory", "peak GPU (allocated) and host (VmHWM) memory per record, GB")
    for r in sorted(recs, key=lambda r: (r.tier, model_rank(r.d.get("model")), str(r.d.get("seq")), r.cell)):
        if r.kind not in ("perf", "v1", "reexec", "gen", "setup") or r.tier not in KINDS:
            continue
        d, p = r.d, r.d.get("peak") or {}
        what = {"perf": f"{d.get('mode')} verifier on {d.get('verifier_device')}", "v1": f"V1 {d.get('mode')}",
                "reexec": f"re-execution {d.get('variant')} on {d.get('device')}", "gen": f"gen {d.get('mode')}",
                "setup": "setup proof"}[r.kind]
        row = f"{r.tier} {d.get('model')} T={d.get('seq', d.get('prompt'))} {what}"
        tb.rows.append(" & ".join([r.tier, NAME.get(d.get("model"), str(d.get("model"))),
                                   str(d.get("seq", d.get("prompt", "--"))), what,
                                   tb.value(row, "GPU GB", p.get("gpu_peak_memory"), r.path, gb),
                                   tb.value(row, "host GB", p.get("host_peak_rss"), r.path, gb)]) + r" \\")
    return tb


def tab_real(recs) -> Table:
    """F3a: real-weight cells next to their random-weight twins (the real tier, or records labelled f3a-* / real*,
    e.g. F3a's 2080 Ti and Xeon cells of before the freeze given as ``*.jsonl`` in ``--logs``): v0 and V1 bytes,
    V1's ratio, prover and CPU verifier, exceptions."""
    tb = Table("tab_real", "real tier and f3a-* records: perf.py (v0) and v1_bytes.py (V1), real / random")
    pairs = defaultdict(dict)
    for r in recs:
        label = str(r.d.get("label") or "")
        if r.kind not in ("perf", "v1") or not (r.tier == "real" or label.startswith(("f3a", "real"))):
            continue
        if r.d.get("codec_only"):
            continue
        d = r.d
        kind = "real" if d.get("real") else "random"
        pairs[(d["model"], d["seq"], d["mode"], r.kind)][kind] = r
    for (m, seq, mode, kind), p in sorted(pairs.items(), key=lambda kv: (model_rank(kv[0][0]), kv[0][1:])):
        row = f"{m} T={seq} {mode} {kind}"
        cells = [NAME.get(m, m), tokens(seq), mode, "v0" if kind == "perf" else "V1"]
        for which in ("real", "random"):
            r = p.get(which)
            d = r.d if r else {}
            if kind == "perf":
                cells += [tb.value(row, f"v0 bytes {which}", d.get("proof_bytes"), r.path if r else "", b),
                          tb.value(row, f"prove {which}", d.get("prove_s"), r.path if r else "", t),
                          tb.value(row, f"CPU verify {which}", d.get("verify_s"), r.path if r else "", t)]
            else:
                cells += [tb.value(row, f"V1 bytes {which}", d.get("v1_total"), r.path if r else "", b),
                          tb.value(row, f"ratio {which}", d["v0_total"] / d["v1_total"] if d else None,
                                   r.path if r else "", x),
                          tb.value(row, f"exceptions {which}", d.get("n_exceptions"), r.path if r else "")]
        tb.rows.append(" & ".join(cells) + r" \\")
    if not pairs:
        tb.missing.append("no real-tier records")
    return tb


def tab_setup(recs, macros) -> Table:
    """D4: the setup proof's prover and verifier time, its size, and the per-query columns (honest / untrusted)."""
    tb = Table("tab_setup_time", "setup_timing.py: commit, setup prover (GPU) and verifier (CPU)")
    for r in sorted(by(recs, "setup"), key=lambda r: (model_rank(r.d["model"]), r.d["amortise"])):
        d = r.d
        row = f"{d['model']} a={d['amortise']}"
        tb.rows.append(" & ".join([
            NAME.get(d["model"], d["model"]), f"{d['amortise']:,}".replace(",", "{,}"),
            tb.value(row, "setup proof", d["proof_bytes"], r.path, b),
            tb.value(row, "commit", d["commit_s"], r.path, t),
            tb.value(row, "setup prover", d["prove_s"], r.path, t),
            tb.value(row, "setup verifier", d["verify_s"], r.path, t),
            tb.value(row, "columns/query honest", d["query_columns_bytes_honest"], r.path, b),
            tb.value(row, "columns/query untrusted", d["query_columns_bytes_untrusted"], r.path, b)]) + r" \\")
        if d.get("layers") is None:
            name = macro_name("Setup", NAME.get(d["model"], d["model"]), "A", d["amortise"])
            macros[name + "Prover"] = t(d["prove_s"])
            macros[name + "Verifier"] = t(d["verify_s"])
    if not tb.rows:
        tb.missing.append("no setup records")
    return tb


def tab_quality(recs) -> Table:
    """Quality records (ppl_2048.py, ppl_llama.py): per model and graph, the median perplexity ratio to the
    reference over the calibration draws (and its 95% interval where recorded) and the top-1 agreement."""
    tb = Table("tab_quality", "ppl records: ratio to the fp reference, median over draws")
    groups = defaultdict(list)
    ref = {}
    for r in by(recs, "ppl"):
        d = r.d
        key = (d.get("model"), d.get("seq"), d.get("windows"))
        if d.get("cell") in ("fp32", "reference"):
            ref[key] = d["ppl"]
        else:
            groups[key + (d["cell"],)].append(r)
    for (m, seq, win, cell), rs in sorted(groups.items(), key=lambda kv: tuple(map(str, kv[0]))):
        ratios = [r.d.get("ratio") or (r.d["ppl"] / ref[(m, seq, win)] if (m, seq, win) in ref else None) for r in rs]
        ratios = [v for v in ratios if v]
        top1 = [r.d["top1"] for r in rs if r.d.get("top1") is not None]
        row = f"{m} T={seq} {cell}"
        tb.rows.append(" & ".join([str(m).replace("_", r"\_"), tokens(seq) if seq else "--", str(win), cell,
                                   tb.value(row, "fp ppl", ref.get((m, seq, win)), rs[0].path, lambda v: f"{v:.2f}"),
                                   tb.value(row, "ratio", statistics.median(ratios) if ratios else None, rs[0].path,
                                            lambda v: f"$\\times${v:.3f}"),
                                   str(len(ratios)),
                                   tb.value(row, "top-1", statistics.median(top1) if top1 else None, rs[0].path,
                                            lambda v: f"{v:.3f}")]) + r" \\")
    return tb


def tab_long(recs, macros) -> Table:
    """V-LONG-b: per prompt length, the CPU verifier per block (3 blocks minus 2), proof bytes per block, and every
    re-execution baseline per block, then the fit t(T) = a T + b T^2 of the verifier per block."""
    tb = Table("tab_long", "long tier: perf.py --layers 2 and 3, reexec.py --layers 2 3 (per block = L3 - L2)")
    pts = []
    for seq in sorted({r.d.get("seq") for r in by(recs, "perf", "long")} - {None}):
        two = perf_cell(recs, "long", "llama2-7b", seq, "Kpre", layers=2)
        three = perf_cell(recs, "long", "llama2-7b", seq, "Kpre", layers=3)
        row = f"llama2-7b T={seq}"
        per = three.d["verify_s"] - two.d["verify_s"] if two and three else None
        per_b = three.d["proof_bytes"] - two.d["proof_bytes"] if two and three else None
        if per:
            pts.append((seq, per))
        base = []
        for v in ("float", "bf16", "int", "int_fp32gemm", "dynint8"):
            rr = [r for r in by(recs, "reexec", "long") if r.d.get("seq") == seq and r.d.get("variant") == v]
            ms = rr[-1].d.get("measured_s", {}) if rr else {}
            base.append(tb.value(row, f"{v} per block", ms["3"] - ms["2"] if "2" in ms and "3" in ms else None,
                                 rr[-1].path if rr else "", t))
        tb.rows.append(" & ".join([tokens(seq), tb.value(row, "verifier per block", per, "", t),
                                   tb.value(row, "bytes per block", per_b, "", b)] + base) + r" \\")
    if len(pts) >= 2:                  # least squares for t = a T + b T^2 (no constant: per block)
        s11 = sum(T * T for T, _ in pts)
        s12 = sum(T ** 3 for T, _ in pts)
        s22 = sum(T ** 4 for T, _ in pts)
        y1 = sum(T * v for T, v in pts)
        y2 = sum(T * T * v for T, v in pts)
        det = s11 * s22 - s12 * s12
        if det:
            a, bq = (y1 * s22 - y2 * s12) / det, (s11 * y2 - s12 * y1) / det
            macros["LongFitA"] = f"{a:.3e}"
            macros["LongFitB"] = f"{bq:.3e}"
            macros["LongFitMaxResidual"] = f"{max(abs(a * T + bq * T * T - v) / v for T, v in pts):.3f}"
    return tb


def tab_h2d(recs, macros) -> Table:
    tb = Table("tab_h2d", "h2d_bandwidth.py: median GB/s (10^9 B/s)")
    for r in by(recs, "h2d"):
        d = r.d
        row = f"{d['gpu']} {d['bytes'] >> 20} MiB {d['direction']} {d['memory']}"
        tb.rows.append(" & ".join([d["gpu"], str(d["bytes"] >> 20), d["direction"], d["memory"],
                                   tb.value(row, "GB/s", d["gb_s"], r.path, lambda v: f"{v:.1f}")]) + r" \\")
        if r.tier == "gpuv" and d["direction"] == "h2d" and d["memory"] == "pinned":
            macros["HtoDPinnedGBs"] = f"{d['gb_s']:.1f}"
    return tb


def tab_spread(recs) -> Table:
    """Every headline timing as median [min-max] (F6 part 2): the cpuv and gpuv perf.py cells and the v1 tier."""
    tb = Table("tab_spread", "perf.py prove_s/verify_s and v1_bytes.py timed medians with [min, max]")
    for r in sorted(recs, key=lambda r: (r.tier, model_rank(r.d.get("model")), str(r.d.get("seq")), r.cell)):
        d = r.d
        if r.kind == "perf" and r.tier in ("cpuv", "gpuv"):
            row = f"{r.tier} {d['model']} T={d['seq']} {d['mode']} {d.get('verifier_device')}"
            tb.rows.append(" & ".join([r.tier, NAME.get(d["model"], d["model"]), tokens(d["seq"]), d["mode"],
                                       tb.value(row, "prover", d["prove_s"], r.path,
                                                lambda v: spread(v, d.get("prove_s_range"))),
                                       tb.value(row, "verifier", d["verify_s"], r.path,
                                                lambda v: spread(v, d.get("verify_s_range"))),
                                       str(d.get("queries"))]) + r" \\")
        elif r.kind == "v1" and r.tier in ("v1", "cpuv") and d.get("timed"):
            for proto in ("v0", "v1"):
                p, v, pr, vr = v1_timed(d, proto)
                row = f"{r.tier} {d['model']} L{d['layers']} T={d['seq']} {d['mode']} {proto}"
                tb.rows.append(" & ".join([r.tier, f"{NAME.get(d['model'], d['model'])} {d['layers']} bl.",
                                           tokens(d["seq"]), f"{d['mode']} {proto}",
                                           tb.value(row, "prover", p, r.path, lambda q, pr=pr: spread(q, pr)),
                                           tb.value(row, "verifier", v, r.path, lambda q, vr=vr: spread(q, vr)),
                                           str(d["timed"].get("queries"))]) + r" \\")
    return tb


def outcome_counts(pa, names: list[str], policy: str, macros: dict, failed: list) -> Table:
    """The soundness and completeness counts of the run (evaluation.tex's "counts with the final code"): every
    honest query and attack of bench.py's records in ``raw_<name>/`` (``count_outcomes.count_runs``; the run's
    root holds only the package's cells, so no definition filter is needed), as ``\\FinalNHonest``,
    ``\\FinalNAttacks``, ``\\FinalNFreivalds`` and so on."""
    import count_outcomes as co
    tb = Table("tab_counts", f"count_outcomes.py over {names}")
    roots = [p for p in (pa.BASE / f"raw_{n}" for n in names) if p.is_dir()]
    if not roots:
        tb.missing.append("no raw roots")
        return tb
    c = co.count_runs(roots, policy)
    try:
        m = co.macros(c, "Final")
    except SystemExit as e:
        failed.append(f"CHECK FAILED outcome counts: {e}")
        return tb
    for k, v in m.items():
        macros[k] = f"{v:,}".replace(",", "{,}")
        tb.rows.append(f"{k} & {macros[k]} \\\\")
        tb.value(k, "count", v, ",".join(r.name for r in roots))
    return tb


# ================================================================================================ the paper
def paper_rows(tex: str, label: str) -> list[list[str]]:
    """The body rows of the table with ``\\label{label}`` in ``tex`` (cells split on ``&``)."""
    i = tex.find(f"\\label{{{label}}}")
    if i < 0:
        return []
    body = tex[tex.find("\\midrule", i):tex.find("\\bottomrule", i)]
    rows = []
    for line in body.splitlines():
        line = line.strip()
        if "&" not in line or line.startswith("%") or line.startswith("\\multicolumn"):
            continue
        rows.append([c.strip() for c in line.rstrip("\\").rstrip().rstrip("\\").split("&")])
    return rows


def compare_eval_llm(generated: Table, paper_dir: Path) -> tuple[int, list[str]]:
    """The base columns (C prove, C CPU and GPU verify, Kpre CPU verify) and the proofs of the regenerated
    tab:eval-llm against evaluation.tex's: ``(matching cells, mismatches)``.  A ``\\todo`` in the paper is skipped."""
    path = paper_dir / "evaluation.tex"
    if not path.exists():
        return 0, [f"no {path}"]
    paper = paper_rows(path.read_text(encoding="utf-8"), "tab:eval-llm")
    ours = [[c.strip() for c in r.rstrip("\\").rstrip().rstrip("\\").split("&")]
            for r in generated.rows if not r.startswith("\\multicolumn")]
    ok, bad = 0, []
    if len(paper) != len(ours):
        return 0, [f"tab:eval-llm has {len(paper)} rows in the paper, {len(ours)} regenerated"]
    for p, o in zip(paper, ours):
        for col in (1, 2, 4, 5, 7, 8, 9):
            if "\\todo" in p[col]:
                continue
            if p[col] == o[col]:
                ok += 1
            else:
                bad.append(f"{p[0]} column {col}: paper {p[col]!r}, regenerated {o[col]!r}")
    return ok, bad


QUANTITY = re.compile(r"([0-9][0-9.,{}]*)\\,(ms|s|min|kB|MB|GB)")


def compare_eval_compare(generated: Table, paper_dir: Path) -> tuple[int, list[str]]:
    """The "ours" columns (prover, verifier, proof) of the regenerated tab:eval-compare against evaluation.tex's:
    the first quantity of each cell (the factor and bold face are the paper's layout).  Maverick's row is the cell
    without the compact encoding; a ``\\todo`` is skipped."""
    path = paper_dir / "evaluation.tex"
    if not path.exists():
        return 0, [f"no {path}"]
    ours = {}
    for r in generated.rows:
        cells = [c.strip() for c in r.rstrip("\\").rstrip().rstrip("\\").split("&")]
        ours[cells[0].replace(", no encoding", "")] = cells[1:]
    ours["Qwen3-4B (8)"] = next((r.split("&")[1:] for r in generated.rows if r.startswith("Qwen3-4B (8), no encoding")),
                                ours.get("Qwen3-4B (8)", []))
    ok, bad = 0, []
    for p in paper_rows(path.read_text(encoding="utf-8"), "tab:eval-compare"):
        model, mine = p[1], ours.get(p[1])
        if mine is None:
            bad.append(f"{p[0]} {model}: no regenerated row")
            continue
        mine = [c.strip().rstrip("\\").strip() for c in mine]
        for col, label in ((3, "prover"), (5, "verifier"), (7, "proof")):
            if "\\todo" in p[col]:
                continue
            want = QUANTITY.search(p[col])
            got = mine[(col - 3) // 2] if len(mine) > (col - 3) // 2 else "--"
            if want and want.group(0) == got:
                ok += 1
            else:
                bad.append(f"{p[0]} {model} {label}: paper {want.group(0) if want else p[col]!r}, regenerated {got!r}")
    return ok, bad


# ================================================================================================ coverage
def expected_jobs(listing: Path, platform: str) -> list[str]:
    """The job names of a dry-run listing (``-J <name>``)."""
    names = []
    for line in listing.read_text(encoding="utf-8").splitlines():
        m = re.search(r" -J (\S+)", line)
        if m and line.startswith("sbatch"):
            names.append(m.group(1).replace("l40s_sp2027-", f"{platform}-"))
    return names


def job_coverage(names: list[str], logs: Path, recs: list[Rec]) -> list[str]:
    """The jobs of ``names`` whose log is missing or holds nothing (sp- jobs: no record; bench jobs: no 'done in')."""
    have = defaultdict(int)
    for r in recs:
        have[r.cell] += 1
    missing = []
    for name in names:
        if name.startswith("sp-"):
            if not have.get(name[3:]):
                missing.append(name)
        else:
            files = glob.glob(str(logs / f"{name}-*.out"))
            if not any("done in" in Path(f).read_text(encoding="utf-8", errors="replace") for f in files):
                missing.append(name)
    return missing


# ================================================================================================ main
def build(args) -> int:
    import paper_assets as pa
    pa.TABLES = pa.BASE / "tables_l40s"          # paper_assets' default run (its block counts for llm_cell)
    problems: list[str] = []
    if args.stored:
        files = stored_files()
        new_names, base_names = REPORT_RUNS.split(","), REPORT_RUNS.split(",")
        challenges, policy = "int", "auto"
    else:
        files = log_files(Path(args.logs))
        new_names, base_names = args.platform.split(","), args.base.split(",") if args.base else []
        challenges, policy = args.challenges, args.policy
    recs = read_logs(files, problems)
    new, base = bench_tables(pa, new_names), bench_tables(pa, base_names)
    failed = [f"CHECK FAILED {os.path.basename(r.path)}: {msg}" for r in recs for msg in checks(r)]
    macros: dict[str, str] = {}
    eval_llm, eval_llm_full = tab_eval_llm(pa, base, new, policy, challenges)
    tables = [eval_llm, eval_llm_full, tab_eval_compare(pa, new, policy, challenges),
              tab_eval_v1(recs, macros), tab_options(recs, macros), tab_eval_reexec(recs), tab_serve(recs, macros),
              tab_gen(recs, macros), tab_batch(pa, new), tab_memory(recs), tab_bench_memory(pa, new), tab_real(recs),
              tab_setup(recs, macros), tab_quality(recs), tab_long(recs, macros), tab_h2d(recs, macros),
              tab_spread(recs), outcome_counts(pa, new_names, policy, macros, failed)]
    if args.stored or base:
        tables.append(tab_eval_compare(pa, base, "auto", "fs", tag="base"))
        tables[-1].name = "tab_eval_compare_base"
    report = [f"records: {len(recs)} from {len(files)} files; bench tables: new {[p.name for p in new]}, "
              f"base {[p.name for p in base]}"]
    counts = defaultdict(lambda: defaultdict(int))
    for r in recs:
        counts[r.tier][r.kind] += 1
    for tier in sorted(counts):
        report.append(f"  tier {tier}: " + ", ".join(f"{k} {n}" for k, n in sorted(counts[tier].items())))
    for tb in tables:
        report.append(f"{tb.name}: {tb.filled} cells filled, {len(tb.missing)} missing, {len(tb.rows)} rows")
        if args.verbose and tb.missing:
            report += [f"    missing: {m}" for m in tb.missing]
    status = 0
    if args.stored or args.compare_paper:
        ok, bad = compare_eval_llm(eval_llm, Path(args.paper))
        report.append(f"tab:eval-llm against the paper: {ok} base cells and proofs identical, {len(bad)} differ")
        report += [f"    {m}" for m in bad]
        base_compare = next((tb for tb in tables if tb.name == "tab_eval_compare_base"), None)
        if base_compare is not None:
            ok2, bad2 = compare_eval_compare(base_compare, Path(args.paper))
            report.append(f"tab:eval-compare (ours) against the paper: {ok2} cells identical, {len(bad2)} differ")
            report += [f"    {m}" for m in bad2]
            bad = bad + bad2
        if args.stored:
            pairs = {(c["row"], c["column"]): c["value"] for c in eval_llm_full.cells}
            diff = [f"{row} / {col}" for (row, col), v in pairs.items() if col.endswith(" new")
                    and v != pairs.get((row, col[:-4] + " base"))
                    and (row, col[:-4] + " base") in pairs]
            report.append(f"tab:eval-llm new columns from the same runs: {len(diff)} cells differ from the base ones")
            report += [f"    {m}" for m in diff]
            status |= bool(diff)
        status |= bool(bad)
    if args.rehearsal:
        for tier, kinds in KINDS.items():
            if not any(counts[tier].get(k) for k in kinds):
                report.append(f"REHEARSAL: tier {tier} has no record of {sorted(kinds)}")
                status = 1
        if not new:
            report.append("REHEARSAL: no bench tables (run aggregate.py --platform <rehearsal platform>)")
            status = 1
    missing_jobs = []
    if args.expect:
        missing_jobs = job_coverage(expected_jobs(Path(args.expect), new_names[0]), Path(args.logs), recs)
        report.append(f"jobs of {args.expect}: {len(missing_jobs)} without results")
        report += [f"    {m}" for m in missing_jobs]
    report += problems + failed
    status |= bool(problems or failed)
    if args.strict and (missing_jobs or any(tb.missing for tb in tables)):
        status = 1
    print("\n".join(report))
    if not args.check:
        out = Path(args.out)
        out.mkdir(parents=True, exist_ok=True)
        for tb in tables:
            (out / f"{tb.name}.tex").write_text(
                f"% generated by experiments/5_comparison/sp2027_numbers.py: {tb.note}\n" + "\n".join(tb.rows) + "\n",
                encoding="utf-8", newline="\n")
        (out / "numbers.tex").write_text(
            "% generated by experiments/5_comparison/sp2027_numbers.py\n"
            + "".join(f"\\newcommand{{\\{k}}}{{{v}}}\n" for k, v in sorted(macros.items())), encoding="utf-8",
            newline="\n")
        with open(out / "cells.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=["table", "row", "column", "value", "source"])
            w.writeheader()
            for tb in tables:
                w.writerows(tb.cells)
        (out / "coverage.txt").write_text("\n".join(report) + "\n", encoding="utf-8", newline="\n")
        print(f"wrote {len(tables)} tables, {len(macros)} macros to {out}")
    return status


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--logs", default=str(REPO / "logs" / "l40s_sp2027"), help="the run's log folder")
    ap.add_argument("--platform", default="l40s_sp2027", help="bench tables of the run (comma-separated runs)")
    ap.add_argument("--base", default=REPORT_RUNS, help="bench tables of tab:eval-llm's base columns ('' for none)")
    ap.add_argument("--policy", default=os.environ.get("PVI_C_POLICY") or "cnn18c", help="mode C's plan of the run")
    ap.add_argument("--challenges", default="fs", choices=["fs", "int"], help="the run's v0 cells (default fs)")
    ap.add_argument("--out", default=str(REPO / "paper_sp2027" / "generated"))
    ap.add_argument("--paper", default=str(REPO / "paper_sp2027" / "sections"))
    ap.add_argument("--expect", default=None, help="a dry-run listing (PVI_DRYRUN=1 sp2027.sh all): job coverage")
    ap.add_argument("--check", action="store_true", help="parse and report; write nothing")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any missing cell or job")
    ap.add_argument("--rehearsal", action="store_true", help="a rehearsal's folder: every tier must have records")
    ap.add_argument("--stored", action="store_true", help="the repository's stored records (the parsers' check)")
    ap.add_argument("--compare-paper", action="store_true", help="compare tab:eval-llm's base columns with the paper")
    ap.add_argument("--verbose", action="store_true", help="list every missing cell")
    return build(ap.parse_args())


if __name__ == "__main__":
    sys.exit(main())

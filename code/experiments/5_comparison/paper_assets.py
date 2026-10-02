"""Every figure and table body of the report, built from the stored benchmark tables of the runs.

    python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved,l40s_improved2
        # -> ../report/figures/*.pdf, ../report/tables/*.tex
    python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved,l40s_improved2 \\
        --check --pending-csv pending.csv
        # write nothing: list the basic run's missing cells and the optimised numbers still pending

Nothing here runs a model; every number is read from the stored benchmark tables
(``aggregate.py``, ``literature.py``) and from zkLLM's run on our GPU
(``artifacts/results/zkllm_l40s/summary.csv``), so the report can be rebuilt without a GPU.

``--platform`` (default ``l40s``, or ``$PVI_PLATFORM``) is the *basic protocol*, measured with the
code of commit 9401431.  ``--optimised`` is a comma-separated list of runs of the *optimised protocol*
on the same machines; a run whose ``tables_<name>/`` does not exist yet is skipped with a warning, and
where two runs hold the same cell the later one in the list is used.  The report's results are the
optimised protocol: Tables 2, 3 and 5, Figure 4 and the filled points of Figure 3.  The basic protocol
appears in Table 4 (basic -> optimised, the same machines) and as the hollow points of Figure 3.

Which stored cell holds each optimised number is defined in one place, ``OPTIMISED`` (below).  A number
whose cell is not stored yet is *pending*: the basic protocol's value stands in for it, typeset
``\\pending{...}`` (red; main.tex defines the macro) in the tables and drawn with a red edge in the
figures (with a legend note).  ``--check`` lists the pending numbers, ``--pending-csv`` writes them, and
``--strict`` refuses to write while any is pending.  Once the cells are stored, regenerating turns every
table cell and figure point black by itself.

Figures: build them with the pinned matplotlib (``requirements.txt``).  Each panel is its own
file (sub-captions are set in LaTeX), saved on a fixed canvas equal to its printed width, so the
fonts print at their real size; ``save()`` refuses a figure whose width is off, whose text is
smaller than 7 pt, leaves the canvas or overlaps other text, or whose landmark label covers a
marker.  Fonts are embedded as TrueType (Type 42).  Figures 1 and 2 are schematics.
"""

from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from collections import defaultdict
from functools import lru_cache
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import patheffects  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch, Rectangle  # noqa: E402
from matplotlib.ticker import LogLocator, NullFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts" / "comparison"
PLATFORM = os.environ.get("PVI_PLATFORM") or "l40s"   # --platform's default: the report's basic numbers
TABLES = BASE / f"tables_{PLATFORM}"   # the basic run's tables; set by main() from --platform
# the optimised runs' tables, in order (a later run's cell replaces an earlier one's); set by main()
# from --optimised.  A single Path is accepted too.
OPT_TABLES: list[Path] | Path | None = None
ZKLLM_L40S = ROOT / "artifacts" / "results" / "zkllm_l40s" / "summary.csv"   # zkLLM's code on our GPU


def tables_dir(platform: str, required: bool = True) -> Path | None:
    """A platform's tables, written by ``aggregate.py --platform <platform>``."""
    path = BASE / f"tables_{platform}"
    if not path.is_dir():
        if not required:
            return None
        raise SystemExit(f"no {path} (aggregate.py --platform <name> writes one per re-measurement; "
                         f"the report's numbers are l40s and the optimised runs)")
    return path


def opt_dirs(tables=None) -> list[Path]:
    """The optimised runs' tables directories, in order of precedence (last wins)."""
    tables = OPT_TABLES if tables is None else tables
    if tables is None:
        return []
    return [tables] if isinstance(tables, Path) else list(tables)


def _platform_name(tables: Path) -> str:
    """tables_l40s_improved -> l40s_improved."""
    return tables.name[len("tables_"):] if tables.name.startswith("tables_") else tables.name


def hw_short(name) -> str:
    """'NVIDIA L40S' -> 'L40S', 'NVIDIA GeForce RTX 2080 Ti' -> 'RTX 2080 Ti'."""
    return (name or "?").replace("NVIDIA ", "").replace("GeForce ", "")


# the headline selection: Reed-Solomon rate 4, 8 verifier threads, untagged cells, lambda = 128
DEFAULT = {"rate": "4", "threads": "8", "variant": "", "lam": 128}
CNN_ORDER = ["mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar", "resnet18_224"]
# the timing parts that make up one proof and one verification (fs_hash is paid by both; prove_encode and
# verify_decode: the compact encoding of the proof, bench.py --wire; prove_lookups and verify_lookups: a
# commitment plan's lookup tables, bench.py --policy <c policy>)
PROVE = ("prove_forward", "prove_lookups", "prove_fold", "prove_open", "prove_encode", "fs_hash")
# verify_upload: a GPU client's host-to-device copy of the proof (absent for the CPU verifier);
# verify_total: the streaming verifier (bench.py --verifier-impl stream), which times its overlapped
# derive, uploads, lookup check, products and columns as one phase, recorded instead of those five (OVERLAPPED)
VERIFY = ("verify_decode", "verify_derive", "verify_lookups", "verify_fold", "verify_products", "verify_columns",
          "verify_upload", "verify_total", "fs_hash")
OVERLAPPED = ("verify_derive", "verify_lookups", "verify_products", "verify_columns", "verify_upload")


def _one_verifier(metrics, where: str) -> None:
    """The verify phases of one row come from one verifier: never ``verify_total`` with the
    phases it overlaps (they would be counted twice)."""
    if "verify_total" in metrics and any(k in metrics for k in OVERLAPPED):
        raise SystemExit(f"{where}: verify_total (the streaming verifier) together with {OVERLAPPED}")


LITERATURE = ("reported_curated.csv",)  # not measured by us: in tables/, shared by every platform


def _read(name: str, tables: Path | None = None) -> list[dict]:
    """One stored table: of ``tables`` (default: the basic run's), or the shared literature table."""
    path = (BASE / "tables" if name in LITERATURE else (tables or TABLES)) / name
    if not path.exists():
        return []
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


# ------------------------------------------------------------------------------ cell names and tags
# A cell's variant is bench.py's --tag (its parts in the runner's order) plus the suffixes bench.py adds
# itself (_prune, _lookups, _pol<name>).  Cells are matched by the SET of their tags, so a re-run that
# writes --tag _gpuv_wire instead of _wire_gpuv finds the same cell.
TAG_ORDER = ("thr1", "thr12", "nolean", "nofix", "tf32", "batch", "wire", "gpuv", "stream", "prune", "lookups")
OPT_TAGS = frozenset({"wire", "stream", "prune", "lookups"})   # what the optimised protocol adds (and pol*)


def tagset(variant: str) -> frozenset:
    """'_wire_gpuv_prune_polauto' -> {'wire', 'gpuv', 'prune', 'polauto'}."""
    return frozenset(t for t in (variant or "").split("_") if t)


def variant_of(tags) -> str:
    """The variant string bench.py writes for a tag set (its --tag parts in TAG_ORDER, then _pol<name>)."""
    tags = set(tags)
    known = [t for t in TAG_ORDER if t in tags]
    pol = sorted(t for t in tags if t.startswith("pol"))
    rest = sorted(tags - set(known) - set(pol))
    return "".join(f"_{t}" for t in known + rest + pol)


def basic_tags(tags) -> frozenset:
    """The basic protocol's counterpart of an optimised cell: the same settings without the optimisations."""
    return frozenset(t for t in tags if t not in OPT_TAGS and not t.startswith("pol"))


CNN_CELL = re.compile(r"^defence_(C|K|Kpre)_(int|fs)_lam(\d+)_rate(\d+)(.*)$")


def cnn_cell(mode, ch, lam, tags, rate="4") -> str:
    return f"defence_{mode}_{ch}_lam{lam}_rate{rate}{variant_of(tags)}"


class Measured:
    """Index over ``measured_summary.csv`` (median and mean per model, cell, metric) of one run, or of
    several runs merged (a list of tables directories; where two hold the same cell, the later one's
    rows replace the earlier one's, cell by cell)."""

    def __init__(self, tables: Path | list | None = None) -> None:
        dirs = [tables or TABLES] if not isinstance(tables, (list, tuple)) else list(tables)
        self.dirs = dirs
        self.tables = dirs[-1] if dirs else None
        self.idx, self.src = {}, {}
        by_cell: dict = {}
        for d in dirs:
            rows: dict = defaultdict(list)
            for r in _read("measured_summary.csv", d):
                rows[(r["suite"], r["model"], r["cell"])].append(r)
            for cell, rs in rows.items():
                by_cell[cell] = (_platform_name(d), rs)       # a later run replaces the whole cell
        for (suite, model, cell), (name, rs) in by_cell.items():
            for r in rs:
                key = (r["suite"], r["model"], r["cell"], r["metric"], r.get("batch", ""), r.get("attack", ""),
                       r.get("lam", ""), r.get("stage", ""), r.get("k", ""))
                self.idx[key] = r
                self.src[key] = name
        # one machine and one thread count for the timed CNN cells (cells tagged _thr<n> excepted: they
        # are a separate selection and never enter Tables 2-5 by default)
        timed = [r for k, r in self.idx.items() if k[0] == "cnn" and k[2].startswith("defence_")
                 and not any(t.startswith("thr") for t in tagset(CNN_CELL.match(k[2]).group(5)
                                                                     if CNN_CELL.match(k[2]) else ""))]
        hw = {r.get("prover_hw") for r in timed}
        if len(hw) > 1:
            raise SystemExit(f"CNN timings of several machines in {[d.name for d in dirs]}: {sorted(map(str, hw))}")
        self.prover_hw = next(iter(hw), None)
        # CNN rows are not filtered by thread count, so a CNN job run with another --threads would
        # enter Table 2 silently; the verifier string ('<cpu> x<n> threads') catches it
        vhw = {r.get("verifier_hw") for r in timed}
        if len(vhw) > 1:
            raise SystemExit(f"CNN verifier timings of several CPUs/thread counts: {sorted(map(str, vhw))}")
        self.verifier_hw = next(iter(vhw), None)
        self.cnn_cells = {}
        for (suite, model, cell, *_rest) in self.idx:
            m = CNN_CELL.match(cell) if suite == "cnn" else None
            if m:
                mode, ch, lam, rate, var = m.groups()
                self.cnn_cells[(model, mode, ch, int(lam), rate, tagset(var))] = cell

    def get(self, model, cell, metric, lam=""):
        """Median of one CNN metric (batch, attack, stage and k left empty)."""
        r = self.idx.get(("cnn", model, cell, metric, "", "", str(lam), "", ""))
        return _f(r["median"]) if r else None

    def platform(self, model, cell, metric="bytes_total"):
        return self.src.get(("cnn", model, cell, metric, "", "", "", "", ""))

    def find_cnn(self, model, mode, ch, lam, tags, rate="4"):
        """The stored name of an image model's defence cell with these settings and tags, or None."""
        return self.cnn_cells.get((model, mode, ch, int(lam), rate, frozenset(tags)))

    def total(self, model, cell, parts):
        vals = {k: self.get(model, cell, k) for k in parts}
        _one_verifier({k for k, v in vals.items() if v is not None}, f"cnn/{model}/{cell}")
        return None if all(v is None for v in vals.values()) else sum(v or 0 for v in vals.values())

    def cost(self, model, cell):
        """(prove seconds, verify seconds, proof bytes) of one CNN cell."""
        return self.total(model, cell, PROVE), self.total(model, cell, VERIFY), self.get(model, cell, "bytes_total")

    def attack_rate(self, model, cell, attack):
        """(fraction rejected, number of attempts) for one attack type."""
        vals = [(_f(r["mean"]), _f(r["n"])) for k, r in self.idx.items()
                if k[1] == model and k[2] == cell and k[3] == "rejected" and k[5] == attack]
        n = sum(v[1] for v in vals)
        return (sum(v[0] * v[1] for v in vals) / n, int(n)) if n else (None, 0)


@lru_cache(maxsize=None)
def _llm_table(dirs: tuple) -> tuple:
    """The rows of ``llm_full_model.csv`` of one or several runs; where two runs hold the same selection
    (model, length, setting, tags), the later run's rows replace the earlier one's."""
    merged: dict = {}
    for d in dirs:
        sel: dict = defaultdict(list)
        for r in _read("llm_full_model.csv", Path(d)):
            r = dict(r, _platform=_platform_name(Path(d)))
            sel[(r["model"], r["seq"], r["mode"], r["challenges"], r["lam"], r.get("threads", ""),
                 r.get("rate", ""), tagset(r.get("variant", "")), r.get("batch", ""))].append(r)
        merged.update(sel)
    return tuple(r for rows in merged.values() for r in rows)


def _llm_dirs(tables) -> tuple:
    if tables is None:
        return (str(TABLES),)
    if isinstance(tables, (list, tuple)):
        return tuple(str(t) for t in tables)
    return (str(tables),)


def llm_rows(mode="C", lam=DEFAULT["lam"], threads=DEFAULT["threads"], variant=DEFAULT["variant"],
             challenges="int", tables=None, tags=None) -> dict:
    """``{(model, seq): {metric: (value, provenance, params, prover_hw)}}`` from the ``llm_full_model.csv``
    of one run (``tables``: default the basic run) or several (a list).  The variant is matched by its tag
    set (``tags``, or the tags of ``variant``).  Where a run has both a full build and 1-2 block builds,
    the measured row is used."""
    want = frozenset(tags) if tags is not None else tagset(variant)
    out: dict = defaultdict(dict)
    for r in _llm_table(_llm_dirs(tables)):
        if (r["mode"], r["challenges"]) != (mode, challenges) or _f(r["lam"]) != lam or r.get("batch"):
            continue
        if (r.get("rate", ""), r.get("threads", "")) != (DEFAULT["rate"], str(threads)):
            continue
        if tagset(r.get("variant", "")) != want:
            continue
        key = (r["model"], int(float(r["seq"])))
        old = out[key].get(r["metric"])
        if old is not None:
            if old[1] == r["provenance"] or "measured" not in (old[1], r["provenance"]):
                raise SystemExit(f"two headline rows for {key} {r['metric']}")
            if old[1] == "measured":
                continue
        out[key][r["metric"]] = (float(r["value"]), r["provenance"], _f(r["params_full"]), r.get("prover_hw"))
    hws = {v[3] for d in out.values() for v in d.values()}
    if len(hws) > 1:
        raise SystemExit(f"LLM rows of several machines in one selection: {sorted(map(str, hws))}")
    return out


def _llm_cost(d: dict):
    """(prove seconds, verify seconds, proof bytes, parameters) of one LLM row."""
    _one_verifier(d, "an LLM row")
    prove = sum(d[k][0] for k in PROVE if k in d)
    verify = sum(d[k][0] for k in VERIFY if k in d) if any(k in d for k in ("verify_products", "verify_total")) \
        else None
    return prove, verify, d.get("bytes_total", (None,))[0], d["prove_forward"][2]


def _measured(d: dict) -> bool:
    """Every metric of one LLM row comes from a full-depth build (none extrapolated)."""
    return bool(d) and all(v[1] == "measured" for v in d.values())


def zkllm_l40s_whole_model_s(model: str = "llama2-7b", seq: int = 2048) -> float | None:
    """zkLLM's public code on our L40S: its per-layer proving time times the layer count, as
    ``artifacts/results/zkllm_l40s/summarise.py --csv`` stores it."""
    if not ZKLLM_L40S.exists():
        return None
    with open(ZKLLM_L40S, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if (r["model"], int(r["seq"])) == (model, seq):
                return float(r["whole_model_s"])
    return None


def _sel(sel: dict):
    """(mode, lam, threads, tags) of a selection dict (``variant`` or ``tags``)."""
    tags = frozenset(sel["tags"]) if "tags" in sel else tagset(sel.get("variant", DEFAULT["variant"]))
    return sel.get("mode", "C"), sel.get("lam", DEFAULT["lam"]), str(sel.get("threads", DEFAULT["threads"])), tags


def llm_cost(key, sel: dict, tables=None, challenges="int", full_only=False):
    """(prove, verify, bytes, params) of one (model, seq) in one selection of one run (or several), or None."""
    mode, lam, thr, tags = _sel(sel)
    d = llm_rows(mode, lam=lam, threads=thr, challenges=challenges, tables=tables, tags=tags).get(key)
    if not d or "prove_forward" not in d or (full_only and not _measured(d)):
        return None
    return _llm_cost(d)


def _llm_platform(key, sel, tables, challenges):
    mode, lam, thr, tags = _sel(sel)
    for r in _llm_table(_llm_dirs(tables)):
        if (r["model"], int(float(r["seq"])), r["mode"], r["challenges"]) == (*key, mode, challenges) and \
                _f(r["lam"]) == lam and r.get("threads", "") == thr and tagset(r.get("variant", "")) == tags \
                and not r.get("batch"):
            return r["_platform"]
    return "?"


# ============================================================ THE OPTIMISED PROTOCOL: ONE DEFINITION
# Which stored cell holds each optimised number, as the cell's settings and its set of variant tags.
# "tags": the defining cell; "acceptable": cells that stand in, in black, until the defining one is
# stored (listed with status 'acceptable' by --check); otherwise the number is pending and the basic
# protocol's cell (the same settings without OPT_TAGS and pol*) stands in, in red.
# This follows the re-run's definition (paper_final/rerun: bench.py at c8be5eb):
#   CNN, C (int/fs):      --policy auto --wire --tag _wire                 -> _wire_polauto
#   CNN, K/Kpre:          --wire --tag _wire                               -> _wire
#   LLM, C, CPU verifier: --policy auto --wire --prune-last --tag _wire    -> _wire_prune_polauto
#   LLM, C, GPU verifier: ... --verifier-device cuda --verifier-impl stream --tag _wire_gpuv_stream
#   LLM, Kpre, CPU:       --lookups --wire --prune-last --tag _wire         -> _wire_prune_lookups
#   Qwen3-4B (Maverick):  PVI_THREADS=1, Kpre:int lambda 40, --lookups --prune-last --wire --tag _thr1_wire
LONG = 2048   # prompts of this length or more use the "long" entries
OPTIMISED = {
    # image models with committed weights: planning rule + compact encoding (interactive and Fiat-Shamir,
    # lambda 40/80/128)
    ("cnn", "C"): dict(tags=("wire", "polauto")),
    # image models with known weights (no columns, so no planning rule): compact encoding.  No stand-in:
    # l40s_improved's defence_{K,Kpre}_*_rate4 cells are the released code without --wire, whose proof is
    # the basic one byte for byte, so the basic placeholder (red) is shown until the _wire cells are stored
    ("cnn", "Kpre"): dict(tags=("wire",)),
    ("cnn", "K"): dict(tags=("wire",)),
    # language models, committed weights: planning rule + compact encoding + the last block at the last
    # position only.  Llama-2-7B's 1- and 64-token runs without pruning stand in until the re-run (the
    # authors' one sanctioned stand-in).
    ("llm", "C", "short", "cpu"): dict(tags=("wire", "prune", "polauto"), acceptable=[("wire", "polauto")]),
    ("llm", "C", "long", "cpu"): dict(tags=("wire", "prune", "polauto")),
    # ... and with the streaming GPU verifier.  No stand-in: GPT-2's _wire_gpuv_prune_polauto cell has the
    # non-streaming GPU verifier, and the 2,048-token _gpuv_stream cells of l40s_improved send the basic proof
    ("llm", "C", "short", "gpu"): dict(tags=("wire", "gpuv", "stream", "prune", "polauto")),
    ("llm", "C", "long", "gpu"): dict(tags=("wire", "gpuv", "stream", "prune", "polauto")),
    # known weights: compact encoding + embedding look-ups + pruning
    ("llm", "Kpre", "short", "cpu"): dict(tags=("wire", "prune", "lookups")),
    ("llm", "Kpre", "long", "cpu"): dict(tags=("wire", "prune", "lookups")),
    ("llm", "Kpre", "short", "gpu"): dict(tags=("wire", "gpuv", "stream", "prune", "lookups")),
    ("llm", "Kpre", "long", "gpu"): dict(tags=("wire", "gpuv", "stream", "prune", "lookups")),
}
# settings that differ from the default (lambda 128, 8 verifier threads): Qwen3-4B in Maverick's setting
OPTIMISED_SPECIAL = {
    ("qwen3-4b", 8, "Kpre"): dict(lam=40, threads="1", tags=("thr1", "wire", "prune", "lookups")),
}
# Table 5's verifier column against a published system whose verifier ran on fewer threads: () compares
# our 8-thread verifier (the text states the caveat); ("thr1",) our one-thread run of the re-run's
# optional cells defence_C_<ch>_lam<l>_rate4_thr1_wire_polauto (platform l40s_improved2_thr1, listed in
# --optimised), pending (red, '--') until it is stored.  zkCNN's verifier ran on one core.
TABLE5_VERIFIER_EXTRA_TAGS = {"zkCNN": ()}
# Which cell each column of an LLM row comes from, in order of preference: the prover and the proof
# from the CPU-verifier cell (the same one at every prompt length, so Tables 3-5 and the text quote one
# prover time per configuration), the CPU verifier from it, the GPU verifier from the GPU cell.
COLUMN_CELLS = {"prove": ("cpu", "gpu"), "bytes": ("cpu", "gpu"), "verify": ("cpu",), "gpu": ("gpu",)}
LLM_METRIC = {"prove": 0, "verify": 1, "gpu": 1, "bytes": 2, "params": 3}
CNN_METRIC = {"prove": 0, "verify": 1, "bytes": 2}
# kept for scripts written against the previous version (the image models' defining cell)
CNN_OPT_CELL = "defence_C_{ch}_lam{lam}_rate4" + variant_of(OPTIMISED[("cnn", "C")]["tags"])


def definition_tagsets(acceptable: bool = False) -> dict:
    """{(suite, mode): {tag sets}} of the optimised definition's defence and tamper cells (and of the
    acceptable stand-ins too, if asked): what count_outcomes.py --definition counts.  A language model's
    CPU-verifier cell may also be a batch run (the batch tag added)."""
    out: dict = defaultdict(set)
    for key, spec in OPTIMISED.items():
        suite, mode = key[0], key[1]
        for tags in [spec["tags"]] + (list(spec.get("acceptable", [])) if acceptable else []):
            out[(suite, mode)].add(frozenset(tags))
            if suite == "llm" and "gpuv" not in tags:
                out[(suite, mode)].add(frozenset(tags) | {"batch"})
    for (_m, _s, mode), spec in OPTIMISED_SPECIAL.items():
        out[("llm", mode)].add(frozenset(spec["tags"]))
    return dict(out)


def llm_spec(model, seq, mode="C", device="cpu"):
    """The optimised cell of one language-model setting: dict(mode, lam, threads, tags, acceptable), or None."""
    special = OPTIMISED_SPECIAL.get((model, seq, mode))
    if special:
        return dict(mode=mode, acceptable=[], **special) if device == "cpu" else None
    spec = OPTIMISED.get(("llm", mode, "long" if seq >= LONG else "short", device))
    if spec is None:
        return None
    return dict(mode=mode, lam=DEFAULT["lam"], threads=DEFAULT["threads"], tags=spec["tags"],
                acceptable=spec.get("acceptable", []))


@lru_cache(maxsize=None)
def _n_layers() -> dict:
    out = {}
    for r in _read("llm_full_model.csv"):
        if r.get("n_layers_full"):
            out[r["model"]] = int(float(r["n_layers_full"]))
    return out


def llm_cell(model, seq, mode, ch, lam, tags) -> str:
    return f"defence_{mode}_{ch}_lam{lam}_T{seq}_L{_n_layers().get(model, '?')}{variant_of(tags)}"


# Every optimised number a builder needed and did not find (pending), or found only as an acceptable
# stand-in: --check prints them and --pending-csv writes them.
PENDING: list[dict] = []
PENDING_FIELDS = ("asset", "row", "column", "model", "optimised_cell", "basic_cell", "status")


def _note(where, model, opt_cell, shown_cell, status):
    """``shown_cell``: the cell whose value is shown instead ('<platform>:<cell>'; for a pending number the
    basic protocol's, for an acceptable one the stand-in optimised cell)."""
    if where is None or status == "measured":
        return
    asset, row, column = where
    PENDING.append(dict(asset=asset, row=row, column=column, model=model, optimised_cell=opt_cell,
                        basic_cell=shown_cell or "(none: --)", status=status))


_CACHE: dict = {}


def basic_measured() -> Measured:
    key = ("basic", str(TABLES))
    if key not in _CACHE:
        _CACHE[key] = Measured(TABLES)
    return _CACHE[key]


def opt_measured() -> Measured:
    dirs = opt_dirs()
    key = ("opt",) + tuple(map(str, dirs))
    if key not in _CACHE:
        _CACHE[key] = Measured(dirs)
    return _CACHE[key]


def _cnn_value(Mx: Measured, model, cell, metric):
    if cell is None:
        return None
    if metric == "bits":
        return Mx.get(model, cell, "soundness_bits")
    if Mx.get(model, cell, "prove_forward") is None:
        return None
    return Mx.cost(model, cell)[CNN_METRIC[metric]]


def opt_cnn(model, metric, mode="C", ch="int", lam=None, where=None, extra=()):
    """(value, pending) of one image model's optimised number: metric 'prove', 'verify', 'bytes' or 'bits'.
    Pending: the basic protocol's value of the same setting stands in (None if there is none).  ``extra``:
    tags added to the definition (a one-thread run: ('thr1',)), with no acceptable stand-in."""
    lam = DEFAULT["lam"] if lam is None else lam
    spec = OPTIMISED[("cnn", mode)]
    if extra:
        spec = dict(tags=tuple(spec["tags"]) + tuple(extra))
    defining = cnn_cell(mode, ch, lam, spec["tags"])
    MO = opt_measured()
    for status, tags in [("measured", spec["tags"])] + [("acceptable", a) for a in spec.get("acceptable", [])]:
        cell = MO.find_cnn(model, mode, ch, lam, tags)
        v = _cnn_value(MO, model, cell, metric)
        if v is not None:
            _note(where, model, defining, f"{MO.platform(model, cell)}:{cell}", status)
            return v, False
    M = basic_measured()
    cell = M.find_cnn(model, mode, ch, lam, basic_tags(spec["tags"]))
    v = _cnn_value(M, model, cell, metric)
    _note(where, model, defining, f"{_platform_name(TABLES)}:{cell}" if v is not None else None,
          "pending")
    return v, True


def basic_cnn(model, metric, mode="C", ch="int", lam=None):
    """The basic protocol's value of one image model (Table 4's left-hand side, Figure 3's hollow points)."""
    lam = DEFAULT["lam"] if lam is None else lam
    M = basic_measured()
    return _cnn_value(M, model, M.find_cnn(model, mode, ch, lam, ()), metric)


def _specs(model, seq, mode, metric):
    return [s for s in (llm_spec(model, seq, mode, d) for d in COLUMN_CELLS[metric]) if s]


def opt_llm(model, seq, metric, mode="C", ch="int", where=None, full_only=True):
    """(value, pending) of one language model's optimised number: 'prove', 'verify' (CPU verifier),
    'gpu' (GPU verifier), 'bytes' or 'params'.  Defining cells first (in COLUMN_CELLS order), then the
    acceptable ones; else pending, with the basic protocol's value of the same setting standing in."""
    if metric == "params":
        c = llm_cost((model, seq), {"mode": mode}, None, "int") or llm_cost((model, seq), {"mode": mode}, None, ch)
        return (c[3] if c else None), False
    specs = _specs(model, seq, mode, "gpu" if metric == "gpu" else metric)
    i = LLM_METRIC[metric]
    dirs = opt_dirs()
    for status in ("measured", "acceptable"):
        for spec in specs:
            for tags in ([spec["tags"]] if status == "measured" else spec["acceptable"]):
                sel = dict(spec, tags=tags)
                c = llm_cost((model, seq), sel, dirs, ch, full_only) if dirs else None
                if c is not None and c[i] is not None:
                    _note(where, model, llm_cell(model, seq, mode, ch, spec["lam"], specs[0]["tags"]),
                          f"{_llm_platform((model, seq), sel, dirs, ch)}:"
                          f"{llm_cell(model, seq, mode, ch, spec['lam'], tags)}", status)
                    return c[i], False
    defining = llm_cell(model, seq, mode, ch, specs[0]["lam"], specs[0]["tags"]) if specs else "?"
    for spec in specs:
        sel = dict(spec, tags=basic_tags(spec["tags"]))
        c = llm_cost((model, seq), sel, None, ch, full_only)
        if c is not None and c[i] is not None:
            _note(where, model, defining, f"{_platform_name(TABLES)}:"
                                          f"{llm_cell(model, seq, mode, ch, spec['lam'], sel['tags'])}", "pending")
            return c[i], True
    _note(where, model, defining, None, "pending")
    return None, True


def basic_llm(model, seq, metric, mode="C", ch="int", full_only=True):
    """The basic protocol's value of one language model, from the cells (in COLUMN_CELLS order) that are
    the basic counterparts of the optimised ones: the same cell the pending tables print."""
    for spec in _specs(model, seq, mode, metric):
        c = llm_cost((model, seq), dict(spec, tags=basic_tags(spec["tags"])), None, ch, full_only)
        if c is not None and c[LLM_METRIC[metric]] is not None:
            return c[LLM_METRIC[metric]]
    return None


def opt_cost(MO=None, model=None, seq=None, challenges="int", lam=None):
    """(prove, verify, bytes) of the optimised protocol on one model (seq None: an image model), or None
    if any of the three is pending.  (The verifier is the GPU one at 2,048 tokens, as in Table 5.)"""
    if seq is None:
        vals = [opt_cnn(model, k, "C", challenges, lam) for k in ("prove", "verify", "bytes")]
    else:
        vals = [opt_llm(model, seq, k, _llm_mode(model, seq), challenges)
                for k in ("prove", "gpu" if seq >= LONG else "verify", "bytes")]
    return None if any(p for _, p in vals) else tuple(v for v, _ in vals)


def _llm_mode(model, seq):
    return "Kpre" if (model, seq, "Kpre") in OPTIMISED_SPECIAL else "C"


# (published system, its model and seq in reported_curated.csv, our model, our prompt length or None
# for an image model, our setting): Table 5.  ZKTorch's 1-token Llama-2-7B is compared with our 1-token one.
MAVERICK = "Maverick (verif.-only, 1 thr.)"
OPT_MATCHES = [
    ("zkCNN", "LeNet-5 MNIST", "", "lenet5", None, "C"),
    ("Bionetta", "LeNet-5", "", "lenet5", None, "C"),
    ("zkCNN", "VGG-16 CIFAR-10", "", "vgg16", None, "C"),
    ("ZKML", "VGG-16 CIFAR-10", "", "vgg16", None, "C"),
    ("zkGPT", "GPT-2", "", "gpt2", 64, "C"),
    ("DeepProve", "GPT-2", "64", "gpt2", 64, "C"),
    ("DeepProve", "GPT-2", "512", "gpt2", 512, "C"),
    ("ZKTorch", "Llama-2-7B (1 token)", "1", "llama2-7b", 1, "C"),
    (MAVERICK, "Qwen3-4B", "8", "qwen3-4b", 8, "Kpre"),
    ("zkLLM", "OPT-125M", "2048", "opt-125m", 2048, "C"),
    ("zkLLM", "OPT-1.3B", "2048", "opt-1.3b", 2048, "C"),
    ("zkLLM", "OPT-6.7B", "2048", "opt-6.7b", 2048, "C"),
    ("zkLLM", "Llama-2-7B", "2048", "llama2-7b", 2048, "C"),
    ("zkLLM", "Llama-2-13B", "2048", "llama2-13b", 2048, "C"),
]
# non-interactive published systems: compared with our Fiat-Shamir proofs where those were run
NONINTERACTIVE = {"zkCNN", "Bionetta", "ZKML", "zkGPT", "DeepProve", "ZKTorch"}
# Maverick's verification-only client (Table 8 of Maverick, 1 client thread; catalogue row 90) spends
# 87.1 ms on the matrix checks (reported_curated.csv's verifier_s) and 37.4 ms replaying the
# non-linear layers, which our verifier's time includes, so Table 5 counts both.
MAVERICK_NONLINEAR_S = 0.0374
# published systems whose stated setting differs from the matched row: a footnote letter on their label
# (main.tex's note under Table 5 explains each), lettered in the order a reader meets them in Table 5.
# a: zkCNN's verifier ran on one core, ours on 8 threads; b: zkGPT states no prompt length, ours uses 64
# tokens; c: DeepProve's figures are its transparent (BaseFold) configuration, which proves every position
# (its HyperKZG one sends 9.4 MB at 64 tokens); d: Maverick (MAVERICK_MARK); e: zkLLM's code on our GPU
# (ZKLLM_OWN).  The interactive stand-in for a missing Fiat-Shamir cell is marked with an asterisk
# (INTERACTIVE_MARK), not a letter, so no letter changes when the second run removes it.
SYSTEM_MARK = {"zkCNN": "a", "zkGPT": "b", "DeepProve": "c"}
MAVERICK_MARK = "d"
INTERACTIVE_MARK = r"\ast"
ZKLLM_OWN = "zkLLM$^e$"   # Table 5's last row: zkLLM's public code on our L40S


def _marks(letters) -> str:
    """'$^d$' for one footnote mark, '$^{a,e}$' for several (letters in order, the asterisk last), '' for none."""
    letters = sorted(set(letters), key=lambda s: (s.startswith("\\"), s))
    return "" if not letters else (f"$^{letters[0]}$" if len(letters) == 1 else "$^{" + ",".join(letters) + "}$")

REPORT = ROOT.parent / "report"
FIGS = REPORT / "figures"
TABS = REPORT / "tables"
LAM = DEFAULT["lam"]

# ------------------------------------------------------------------------------------- the palette
OURS = "#1f5f99"        # ours, committed weights (C)
OURS_K = "#7fb2e5"      # ours, known weights (Kpre)
ANCH = "#e67e22"        # Anchuri et al., the path protocol
ANCH_LIGHT = "#f5cba7"
ATTACK = "#c0392b"      # tampered values only
ATTACK_LIGHT = "#f5d0cb"
THEM = "0.55"           # published systems: hollow grey markers
MSG = "0.25"            # protocol messages
PENDING_EDGE = "#ff0000"   # a basic placeholder for a pending optimised point: LaTeX's red, as \pending{}
MIN_PT = 7.0            # no text below 7 pt at print size
SF_C, SF_K = "\U0001d5a2", "\U0001d5aa"   # upright sans-serif C and K: the paper's \mathsf{C}, \mathsf{K}
KPRE = SF_K + r"$_{\mathrm{pre}}$"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                     "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 7,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "xtick.major.width": 0.6,
                     "ytick.major.width": 0.6, "xtick.major.size": 2.5, "ytick.major.size": 2.5,
                     "xtick.major.pad": 2, "ytick.major.pad": 2, "axes.labelpad": 2,
                     "savefig.bbox": None, "figure.dpi": 200,
                     "pdf.fonttype": 42, "ps.fonttype": 42})   # TrueType, not Type 3

SHORT = {"mlp_mnist": "MLP", "lenet5": "LeNet-5", "vgg11": "VGG-11", "vgg16": "VGG-16",
         "resnet18_cifar": "ResNet-18", "resnet18_224": "ResNet-18"}
DATA = {"mlp_mnist": "MNIST", "lenet5": "MNIST", "vgg11": "CIFAR", "vgg16": "CIFAR",
        "resnet18_cifar": "CIFAR", "resnet18_224": "224px"}
LLM_NAME = {"gpt2": "GPT-2", "opt-125m": "OPT-125M", "opt-350m": "OPT-350M", "opt-1.3b": "OPT-1.3B",
            "opt-2.7b": "OPT-2.7B", "opt-13b": "OPT-13B", "opt-6.7b": "OPT-6.7B",
            "llama2-7b": "Llama-2-7B", "llama2-13b": "Llama-2-13B", "qwen3-4b": "Qwen3-4B",
            "llama2-70b": "Llama-2-70B", "opt-30b": "OPT-30B", "opt-66b": "OPT-66B"}

# published systems grouped by proof family, for the cost figure (Plonk/Groth16 and the other ZK proofs
# share one marker)
FAMILY = {"zkCNN": "sumcheck", "zkLLM": "sumcheck", "zkGPT": "sumcheck", "DeepProve": "sumcheck",
          "zkPyTorch": "sumcheck", "Jolt": "sumcheck", "SLP": "sumcheck",
          "ZKML": "other", "EZKL": "other", "ZKTorch": "other", "Bionetta": "other", "vCNN": "other",
          "Mystique": "other", "LAMP": "other", "Maverick": "known"}
FAM_STYLE = {"sumcheck": ("^", "sum-check zkSNARKs"), "other": ("D", "Plonk/Groth16/other ZK"),
             "known": ("*", "Maverick")}
# a few landmark systems are named on each panel: (system, model[, seq]) -> (label, where, ha), where
# where is ("off", (dx, dy)) in points from the marker, or ("at", (x, y)) in data coordinates with a
# leader line.  zkCNN is named at its LeNet-5 point in (a) and (b), Table 5's bold comparison.
LABELS = {
    "prover": {("zkCNN", "LeNet-5 MNIST"): ("zkCNN", ("off", (5, 0)), "left"),
               ("zkLLM", "Llama-2-7B"): ("zkLLM", ("at", (2.6e9, 55)), "center"),
               ("DeepProve", "GPT-2", "64"): ("DeepProve", ("at", (9e7, 6)), "right")},
    "verifier": {("zkCNN", "LeNet-5 MNIST"): ("zkCNN", ("at", (2.6e5, 4e-2)), "center"),
                 ("zkLLM", "Llama-2-7B"): ("zkLLM", ("at", (3.0e9, 0.19)), "center"),
                 ("ZKTorch", "Llama-2-7B (1 token)"): ("ZKTorch", ("off", (0, -9)), "center")},
    "proof": {("zkCNN", "VGG-16 CIFAR-10"): ("zkCNN", ("off", (0, -8)), "center"),
              ("zkLLM", "Llama-2-7B"): ("zkLLM", ("at", (3e9, 3e4)), "center"),
              ("DeepProve", "GPT-2", "64"): ("DeepProve", ("at", (6e8, 3.2e6)), "center")},
}


def family(system: str):
    for k, v in FAMILY.items():
        if system.startswith(k):
            return v
    return None


# ------------------------------------------------------------------------------- layout checks
def _drawn_texts(fig):
    """Every non-empty text the figure draws: titles, labels, tick labels in view, legends, annotations."""
    out = list(fig.texts)
    for leg in fig.legends:
        out += leg.get_texts()
    for ax in fig.axes:
        out += [ax.title, ax._left_title, ax._right_title] + list(ax.texts)
        for axis in (ax.xaxis, ax.yaxis):
            if ax.axison and axis.get_visible():
                out.append(axis.label)
                for tick in axis._update_ticks():
                    out += [tick.label1, tick.label2]
        if ax.get_legend():
            out += ax.get_legend().get_texts()
    return [t for t in out if t.get_visible() and t.get_text().strip()]


def _marker_points(ax):
    """Display coordinates of every marker: scatter collections and marker-only lines."""
    pts = [p for coll in ax.collections for p in coll.get_transform().transform(coll.get_offsets())]
    for line in ax.lines:
        if line.get_marker() not in (None, "", "None", " ") and line.get_linestyle() in ("None", "", " "):
            pts += list(line.get_transform().transform(line.get_xydata()))
    return pts


def layout_problems(fig) -> list[str]:
    """Text below 7 pt, outside the canvas, overlapping other text, or covering a marker."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    texts = [(t, t.get_window_extent(r)) for t in _drawn_texts(fig)]
    bad = []
    for t, bb in texts:
        if t.get_fontsize() < MIN_PT - 1e-6:
            bad.append(f"{t.get_text()!r}: {t.get_fontsize():.1f} pt")
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > W + 0.5 or bb.y1 > H + 0.5:
            bad.append(f"{t.get_text()!r} leaves the canvas")
    for i, (t, a) in enumerate(texts):
        for u, c in texts[i + 1:]:
            if a.overlaps(c) and min(a.x1, c.x1) - max(a.x0, c.x0) > 1 and min(a.y1, c.y1) - max(a.y0, c.y0) > 1:
                bad.append(f"{t.get_text()!r} overlaps {u.get_text()!r}")
    for ax in fig.axes:   # a label never sits on a marker
        pts = _marker_points(ax)
        for t in ax.texts:
            bb = t.get_window_extent(r).expanded(1.0, 1.0)
            for x, y in pts:
                if bb.x0 + 1 < x < bb.x1 - 1 and bb.y0 + 1 < y < bb.y1 - 1:
                    bad.append(f"{t.get_text()!r} covers a marker")
                    break
    return bad


DRY = False     # --check: build everything, write nothing


def save(fig, name: str, width: float):
    """Write figures/<name>.pdf on its fixed canvas (no tight cropping: the canvas is the printed size)."""
    assert abs(fig.get_size_inches()[0] - width) < 0.01, (name, fig.get_size_inches(), width)
    if DRY:
        plt.close(fig)
        return
    bad = layout_problems(fig)
    if bad and not DRAFT:
        raise SystemExit(f"figure {name}: " + "; ".join(bad))
    for problem in bad:
        print(f"  DRAFT {name}: {problem}")
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf", metadata={"CreationDate": None}, bbox_inches=None)  # reproducible bytes
    plt.close(fig)
    print("figure", name)


def _log_axes(ax, x=True, y=True, numticks=5):
    """Major ticks only (minor ticks turn a log spine into a solid bar at this size)."""
    for on, axis in ((x, ax.xaxis), (y, ax.yaxis)):
        if on:
            axis.set_major_locator(LogLocator(base=10, numticks=numticks))
            axis.set_minor_locator(NullLocator())
            axis.set_minor_formatter(NullFormatter())


DRAFT = False   # --draft: write figures despite layout problems (to iterate on a style)
WHITE = [patheffects.withStroke(linewidth=2, foreground="white")]


# ----------------------------------------------------------------------------------- figures 1-2
WIDTHS = [5, 7, 7, 3]
POS = {(l, i): (l, (i - (w - 1) / 2) * 0.55) for l, w in enumerate(WIDTHS) for i in range(w)}
PATH = [(3, 1), (2, 4), (1, 2), (0, 3)]
TAMPERED = (1, 5)
OVERVIEW = (2.17, 1.45)   # printed at 0.31 of acmart sigconf's text width (7.006 in): fonts print at 7 pt


def _network(ax, colour_node):
    for l in range(3):
        for i in range(WIDTHS[l]):
            for j in range(WIDTHS[l + 1]):
                (x0, y0), (x1, y1) = POS[(l, i)], POS[(l + 1, j)]
                ax.plot([x0, x1], [y0, y1], color="0.87", lw=0.35, zorder=1)
    for node, (x, y) in POS.items():
        if node not in PATH:
            fc, ec = colour_node(node)
            ax.scatter(x, y, s=30, facecolor=fc, edgecolor=ec, lw=0.7, zorder=3)
    ax.text(0, -1.45, "input", ha="center", va="top", color="0.45")
    ax.text(3, -0.9, "output", ha="center", va="top", color="0.45")


def _path(ax, colour_node):
    """The checked path, drawn the same in (a) and (b): orange line and node edges, fill = node state."""
    for a, b in zip(PATH[:-1], PATH[1:]):
        (x0, y0), (x1, y1) = POS[a], POS[b]
        ax.plot([x0, x1], [y0, y1], color=ANCH, lw=1.6, zorder=2)
    for n in PATH:
        ax.scatter(*POS[n], s=30, facecolor=colour_node(n)[0], edgecolor=ANCH, lw=1.1, zorder=4)


def _finish(ax):
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-2.02, 2.75)
    ax.axis("off")
    ax.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)


def _attacked(n):
    """(fill, edge) of a node after the single-neuron attack: the tampered node red, every later node
    recomputed honestly (light red), the rest untouched."""
    if n == TAMPERED:
        return ATTACK, ATTACK
    return (ATTACK_LIGHT, "0.35") if n[0] > TAMPERED[0] else ("white", "0.35")


def fig_overview():
    """Figure 1: (a) one random path, (b) one tampered neuron the path misses, (c) the same attacked
    network, every layer checked: the box of the tampered layer is the check that fails."""
    white = lambda n: ("white", "0.35")   # noqa: E731
    fig, ax = plt.subplots(figsize=OVERVIEW)
    _network(ax, white)
    _path(ax, white)
    _finish(ax)
    save(fig, "overview_path", OVERVIEW[0])

    fig, ax = plt.subplots(figsize=OVERVIEW)
    _network(ax, _attacked)
    _path(ax, _attacked)
    ax.annotate("wrong value", xy=POS[TAMPERED], xytext=(0.2, 2.05), color=ATTACK,
                arrowprops=dict(arrowstyle="->", color=ATTACK, lw=0.6, shrinkB=3))
    ax.text(2.55, 2.05, "recomputed\nhonestly", color="0.35", ha="center", va="center", linespacing=1.0)
    _finish(ax)
    save(fig, "overview_attack", OVERVIEW[0])

    fig, ax = plt.subplots(figsize=OVERVIEW)
    _network(ax, _attacked)
    for n in PATH:
        fc, ec = _attacked(n)
        ax.scatter(*POS[n], s=30, facecolor=fc, edgecolor=ec, lw=0.7, zorder=3)
    for l in range(1, 4):
        h = (WIDTHS[l] - 1) * 0.55 / 2 + 0.28
        failing = l == TAMPERED[0]
        ax.add_patch(FancyBboxPatch((l - 0.2, -h), 0.4, 2 * h, boxstyle="round,pad=0.02", fc="none",
                                    ec=ATTACK if failing else OURS, lw=1.4 if failing else 1.0, zorder=2))
    ax.text(1.5, 2.45, "one random combination per layer", ha="center", va="center", color=OURS)
    _finish(ax)
    save(fig, "overview_defence", OVERVIEW[0])


def fig_protocol():
    """Figure 2: the messages of one query in setting C, numbered as in Section 4.2."""
    W_, H_ = 3.33, 2.1
    fig = plt.figure(figsize=(W_, H_))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W_)
    ax.set_ylim(H_, 0)                     # inches from the top
    ax.axis("off")
    xp, xv = 0.62, 2.0                     # prover and verifier lifelines
    ax.text(xp, 0.07, "prover (provider)", ha="center", va="center", weight="bold")
    ax.text(xv, 0.07, "verifier (client)", ha="center", va="center", weight="bold")
    ax.add_patch(Rectangle((0.03, 0.17), W_ - 0.06, 0.19, fc="0.91", ec="none"))
    ax.text(W_ / 2, 0.265, r"setup (once): owner publishes $C_M$ = Merkle roots of $\mathrm{Enc}(A_\ell)$",
            ha="center", va="center", color="0.2")
    top, bottom = 0.42, 2.07
    for x in (xp, xv):
        ax.plot([x, x], [top, bottom], color="0.6", lw=0.8)
    steps = [  # (y, direction, message, verifier's note)
        (0.58, "vp", r"query $x$", None),
        (0.82, "pv", r"$y$, claims $Z_\ell$", "range-check the claims,\nrecompute non-weight ops;\n"
                                               r"draw $\chi$"),
        (1.18, "vp", r"random $\chi$", None),
        (1.42, "pv", r"$u_\ell = \chi^\top\! A_\ell$", r"check $\chi^\top Z_\ell = u_\ell\,[X_\ell; 1]$;"
                                                         "\n" r"draw positions $c$"),
        (1.70, "vp", r"positions $c$", None),
        (1.94, "pv", r"columns $\hat A_{\ell,c}$ + Merkle proof", "check the Merkle paths and\n"
                                                                     r"$\chi^\top \hat A_{\ell,c} = \mathrm{Enc}(u_\ell)_c$"),
    ]
    for i, (y, d, msg, note) in enumerate(steps, 1):
        x0, x1 = (xv, xp) if d == "vp" else (xp, xv)
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>,head_length=0.5,head_width=0.22", color=MSG, lw=0.8,
                                    shrinkA=0, shrinkB=0))
        ax.text((xp + xv) / 2, y - 0.025, msg, ha="center", va="bottom")
        ax.text(xp - 0.05, y, f"({i})", ha="right", va="center", color=MSG)
        if note:
            ax.text(xv + 0.06, y - 0.07, note, ha="left", va="top", color=OURS, linespacing=1.05)
    # messages (3)-(6) are sent in setting C only (in K and Kpre the verifier computes u itself, so the
    # proof is message (2)); the bracket sits on the message side, clear of the verifier's notes
    xb, y0, y1 = 0.30, steps[2][0] - 0.1, steps[5][0] + 0.08
    ax.plot([xb + 0.04, xb, xb, xb + 0.04], [y0, y0, y1, y1], color="0.35", lw=0.6)
    ax.text(xb - 0.06, (y0 + y1) / 2, f"sent in {SF_C} only", rotation=90, ha="center", va="center",
            color="0.35")
    save(fig, "protocol", W_)


# ----------------------------------------------------------------------------------- figure 3
SECURITY_MODELS = ["mlp_mnist", "lenet5", "vgg16", "resnet18_224"]
# one marker shape per model (no per-model colours); hollow: basic, filled: optimised
SECURITY_MARKER = {"mlp_mnist": "o", "lenet5": "s", "vgg16": "^", "resnet18_224": "D"}
# each model's name: (text, anchor point, offset in points, ha, va); anchors: the optimised or basic
# point at a lambda
SECURITY_LABEL = {"mlp_mnist": ("MLP", ("basic", 128), (5, 0), "left", "center"),
                  "lenet5": ("LeNet-5", ("optimised", 128), (-5, 0), "right", "center"),
                  "vgg16": ("VGG-16", ("basic", 40), (5, 0), "left", "center"),
                  "resnet18_224": ("ResNet-18\n(224px)", ("basic", 128), (-1, 7), "right", "bottom")}
SECURITY_TITLE = "Optimised at 128 bits sends less than paths at 40 bits"
# models whose path curve is not drawn: the 224-pixel ResNet's (13.7 MB at 2^-40) lies 0.05 decades from
# VGG-16's (15.6 MB), so the two orange curves cannot be told apart; its blue points stay, and the title
# is still checked against its paths (fig_security)
SECURITY_NO_PATH = {"resnet18_224"}
PATH_MARK_S = 20   # the paths' 2^-40 markers on the 40-bit line (larger than the curves' markers)


def _path_curve(M, m):
    """(bytes, bits) of the path protocol with s = 1, 3, 10, ... paths, in order of s."""
    rows = [r for k, r in M.idx.items() if k[:4] == ("cnn", m, "sampling", "paths_bytes_shared_k")]
    rows.sort(key=lambda r: float(r["k"]))
    return [_f(r["median"]) for r in rows], [_f(r.get("bits")) for r in rows]


def fig_security(M=None, MO=None):
    """Figure 3: security bits against bytes per query, the path protocol against the basic (hollow) and
    the optimised (filled) protocol at lambda = 40, 80, 128.  The bits axis is linear: the path protocol
    stays near zero bits until it opens almost the whole model and trace."""
    M = M or basic_measured()
    size = (3.33, 2.2)
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.165, top=0.80)
    top = 190
    any_pending = False
    for m in SECURITY_MODELS:
        mk = SECURITY_MARKER[m]
        bx, by = _path_curve(M, m)
        x40 = M.get(m, "sampling", "paths_bytes_shared", lam=40)
        if bx and None not in by and m not in SECURITY_NO_PATH:
            cap = M.get(m, "sampling", "open_all_bytes")
            ax.plot(bx + [cap, cap], by + [max(by), top], ":", color=ANCH, lw=1.1, zorder=2)
            ax.scatter([x40], [40], s=PATH_MARK_S, marker=mk, facecolor="white", edgecolor=ANCH, lw=0.9, zorder=4)
        drawn = {}
        basic = {lam: (basic_cnn(m, "bytes", lam=lam), basic_cnn(m, "bits", lam=lam)) for lam in (40, 80, 128)}
        basic = {lam: p for lam, p in basic.items() if None not in p}
        opt = {}
        for lam in (40, 80, 128):
            where = ("Fig. 3", f"{SHORT[m]} ({DATA[m]})", f"optimised, lambda {lam}")
            (x, px), (y, py) = opt_cnn(m, "bytes", lam=lam, where=where), opt_cnn(m, "bits", lam=lam)
            if x is not None and y is not None:
                opt[lam] = (x, y, px or py)
        # the in-figure title (and Sec. 5.2's "on every model") claims the optimised proof at lambda 128 is
        # smaller than the paths' bytes for 2^-40: refuse to print it once a stored point makes it false
        if x40 is not None and 128 in opt and not opt[128][0] < x40:
            raise SystemExit(f"Fig. 3 title false for {m}: optimised lambda=128 sends {opt[128][0]:,.0f} B, "
                             f"the paths need {x40:,.0f} B for 2^-40 (change SECURITY_TITLE and Sec. 5.2)")
        for kind, pts, fill, lw in (("basic", {k: (*v, False) for k, v in basic.items()}, "white", 0.6),
                                    ("optimised", opt, OURS, 1.0)):
            if not pts:
                continue
            drawn[kind] = {lam: p[:2] for lam, p in pts.items()}
            xy = [pts[lam] for lam in sorted(pts)]
            ax.plot([p[0] for p in xy], [p[1] for p in xy], "-", color=OURS, lw=lw, zorder=3)
            for x, y, pend in xy:
                any_pending |= pend
                ax.scatter([x], [y], s=13, marker=mk, facecolor=fill, edgecolor=PENDING_EDGE if pend else OURS,
                           lw=0.9 if pend else 0.8, zorder=4.5 if kind == "optimised" else 4)
        text, (kind, lam), off, ha, va = SECURITY_LABEL[m]
        if kind not in drawn:          # no optimised point of this model: label its basic point
            kind = "basic"
        ax.annotate(text, drawn[kind][lam], xytext=off, textcoords="offset points", ha=ha, va=va,
                    color="0.15", linespacing=1.0, path_effects=WHITE, zorder=5)
    ax.axhline(40, color="0.6", lw=0.5, ls="--", zorder=1)
    ax.set_xscale("log")
    ax.set_xlim(2e3, 3e7)
    ax.set_ylim(-6, top)
    _log_axes(ax, y=False, numticks=6)
    ax.set_yticks([0, 40, 80, 128])
    ax.set_xlabel("bytes sent per query")
    ax.set_ylabel("security (bits)")
    ax.grid(True, which="major", lw=0.3, alpha=0.4)
    handles = [Line2D([], [], color=ANCH, ls=":", lw=1.1, marker="o", ms=3.6, mfc="white", mew=0.8,
                      label="path protocol"),
               Line2D([], [], color=OURS, ls="-", lw=0.6, marker="o", ms=3.6, mfc="white", mew=0.8,
                      label="basic"),
               Line2D([], [], color=OURS, ls="-", lw=1.0, marker="o", ms=3.6, mew=0.8, label="optimised")]
    if any_pending:   # a temporary fourth entry, red: gone once every optimised point is stored
        handles.append(Line2D([], [], ls="", marker="o", ms=3.6, mfc=OURS, mec=PENDING_EDGE, mew=0.9,
                              label="red edge: basic placeholder"))
    spacing = dict(handlelength=2.2, columnspacing=1.4, handletextpad=0.5) if not any_pending else         dict(handlelength=1.4, columnspacing=0.6, handletextpad=0.3)    # four entries in the same row
    leg = fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55 if not any_pending else 0.5, 0.905),
                     ncol=len(handles), frameon=False, borderaxespad=0, **spacing)
    for t in leg.get_texts():
        if t.get_text().startswith("red edge"):
            t.set_color(PENDING_EDGE)
    fig.text(0.015, 0.985, SECURITY_TITLE, ha="left", va="top", weight="bold")
    save(fig, "security_bits", size[0])


# ----------------------------------------------------------------------------------- figure 4
COST = (2.24, 1.75)       # printed at 0.32 of the text width
XLIM = (3e4, 4e10)        # one x-range for the three panels
# the models of Figure 4's language-model points: every full build of the basic run at these lengths
COST_SEQS = {"C": (64, 2048), "Kpre": (64,)}


def _ours_points(M=None):
    """Figure 4's points of ours: (params, [(value, pending)] for prover / verifier / proof, kind, colour)
    with kind 'cnn', 'T64' or 'T2048' and colour OURS (setting C) or OURS_K (Kpre); the CPU verifier
    (8 threads) throughout."""
    M = M or basic_measured()
    out = []
    for mode, colour in (("C", OURS), ("Kpre", OURS_K)):
        for m in CNN_ORDER:
            row = f"CNN, {mode}"
            vals = [opt_cnn(m, k, mode, where=("Fig. 4", row, k)) for k in ("prove", "verify", "bytes")]
            if vals[0][0] is not None:
                out.append((M.get(m, "facts", "n_params"), vals, "cnn", colour))
        for (model, seq), d in sorted(llm_rows(mode).items()):
            if seq not in COST_SEQS[mode] or "prove_forward" not in d or not _measured(d):
                continue
            row = f"LLM {seq} tokens, {mode}"
            vals = [opt_llm(model, seq, k, mode, where=("Fig. 4", row, k)) for k in ("prove", "verify", "bytes")]
            out.append((_llm_cost(d)[3], vals, f"T{seq}", colour))
    return out


def _anchuri_width():
    rows = [r for r in _read("analytic.csv", BASE / "tables") if r["model"] == "llama2-7b"]
    return int(rows[0]["anchuri_width"]) if rows else None


def _our_marker(ax, x, y, kind, colour, pending):
    edge = PENDING_EDGE if pending else colour
    lw = 0.9 if pending else 0.8
    if kind == "T2048":   # half-filled: hollow means "basic" in Figure 3
        ax.plot([x], [y], ls="", marker="s", ms=4, fillstyle="top", mfc=colour, mfcalt="white", mec=edge, mew=lw,
                zorder=4)
    else:
        ax.scatter(x, y, marker="o" if kind == "cnn" else "s", s=16, c=colour, edgecolors=edge, linewidths=lw,
                   zorder=4)


def fig_cost(M=None):
    """Figure 4 (optimised protocol): prover time, verifier time and proof size against model size."""
    rep = [r for r in _read("reported_curated.csv") if _f(r["params"])]
    ours = _ours_points(M)
    any_pending = any(p for _, vals, _, _ in ours for _, p in vals)
    panels = (("prover_s", 0, "prover", "prover time (s)"), ("verifier_s", 1, "verifier", "verifier time (s)"),
              ("proof_bytes", 2, "proof", "proof size (bytes)"))
    width = _anchuri_width()
    for key, col, name, ylabel in panels:
        fig, ax = plt.subplots(figsize=COST)
        fig.subplots_adjust(left=0.19, right=0.97, bottom=0.2, top=0.97)
        named = set()
        for r in rep:
            y, x = _f(r[key]), _f(r["params"])
            if y is None:
                continue
            if r["system"].startswith("Anchuri"):
                ax.scatter(x, y, marker="X", s=30, c=ANCH, zorder=6)
                if name == "prover" and width:
                    ax.annotate(f"one path: detects\n1 in {width:,}", (x, y), xytext=(2.2e10, 1.1e-3),
                                textcoords="data", ha="right", va="center", color=ANCH, linespacing=1.0,
                                path_effects=WHITE, zorder=7)
                continue
            fam = family(r["system"])
            if fam is None:
                continue
            # published markers are hollow and drawn above ours, so ours never hide them
            ax.scatter(x, y, marker=FAM_STYLE[fam][0], s=12 if fam != "known" else 24, c="none",
                       edgecolors=THEM, linewidths=0.6, zorder=5)
            lab = LABELS[name].get((r["system"], r["model"])) or LABELS[name].get((r["system"], r["model"], r["seq"]))
            if lab and lab[0] not in named:
                named.add(lab[0])
                text, (how, where), ha = lab
                kw = dict(xytext=where, textcoords="offset points") if how == "off" else \
                    dict(xytext=where, textcoords="data",
                         arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4, shrinkA=0, shrinkB=1.0))
                ax.annotate(text, (x, y), color="0.3", ha=ha, va="center", path_effects=WHITE, zorder=7, **kw)
        for n, vals, kind, colour in ours:
            v, pending = vals[col]
            if v is not None:
                _our_marker(ax, n, v, kind, colour, pending)
        if name == "proof":
            ax.plot(XLIM, XLIM, "k--", lw=0.5, zorder=1)
            ax.text(2.5e5, 7e6, "int8 model", color="0.3", rotation=31, ha="center", va="center",
                    path_effects=WHITE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlim(*XLIM)
        _log_axes(ax)
        ax.set_xlabel("model parameters")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", lw=0.3, alpha=0.4)
        save(fig, f"cost_{name}", COST[0])

    # one legend for the three panels, as its own strip: shape = workload, colour = setting, grey = published
    def mk(marker, ms=4, **kw):
        return Line2D([], [], marker=marker, ls="", ms=ms, **kw)
    row1 = [mk("o", color="0.3", label="CNN"), mk("s", color="0.3", label="LLM, 64 tokens"),
            mk("s", color="0.3", fillstyle="top", mfcalt="white", label="LLM, 2,048 tokens"),
            Patch(fc=OURS, ec=OURS, label=f"committed ({SF_C})"),
            Patch(fc=OURS_K, ec=OURS_K, label=f"known weights ({KPRE})")]
    row2 = [mk(FAM_STYLE[f][0], mfc="none", color=THEM, label=FAM_STYLE[f][1], ms=6 if f == "known" else 4)
            for f in ("sumcheck", "other", "known")]
    row2 += [mk("X", color=ANCH, label="Anchuri et al., one path", ms=5),
             mk("o", mfc=OURS, mec=PENDING_EDGE, mew=0.9, label="red edge: basic placeholder") if any_pending
             else Line2D([], [], ls="", label=" ")]
    handles = [h for col in zip(row1, row2) for h in col]   # a legend fills its columns top to bottom
    size = (7.0, 0.36)
    fig = plt.figure(figsize=size)
    leg = fig.legend(handles=handles, loc="center", ncol=5, frameon=False, handletextpad=0.4, columnspacing=1.6,
                     handlelength=1.1, labelspacing=0.3, borderaxespad=0)
    for t in leg.get_texts():
        if t.get_text().startswith("red edge"):
            t.set_color(PENDING_EDGE)
    save(fig, "cost_legend", size[0])


# ------------------------------------------------------------------------------------ tables
def t(sec):
    if sec is None:
        return "--"
    if sec < 1:
        return f"{sec * 1e3:.0f}\\,ms" if sec >= 0.0995 else f"{sec * 1e3:.1f}\\,ms"
    if sec < 100:
        return f"{sec:.1f}\\,s"
    return f"{sec / 60:.1f}\\,min"


def b(n):
    if n is None:
        return "--"
    for unit, k in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= k:
            v = n / k
            return (f"{v:.1f}" if v < 100 else f"{v:.0f}") + f"\\,{unit}"
    return f"{n:.0f}\\,B"


def red(s: str, pending: bool) -> str:
    """A table cell, wrapped in main.tex's \\pending{} (red) while its optimised value is a basic placeholder."""
    return rf"\pending{{{s}}}" if pending else s


def _num(v):
    """A number in a shared unit: 1{,}012, 375, 14.8, 0.30."""
    if v >= 99.5:
        return f"{v:,.0f}".replace(",", "{,}")
    return f"{v:.1f}" if v >= 1 else f"{v:.2f}"


def pair(a, b_, kind, pending=False):
    """'a$\\to$b unit' (Table 4), both in the unit ``t()``/``b()`` give the smaller value (1{,}012$\\to$375\\,ms);
    one value if both print the same (unless b is a pending placeholder, which is printed red)."""
    fmt = t if kind == "t" else b
    if b_ is None:
        return f"{fmt(a)}$\\to${red('--', pending)}"
    if fmt(a) == fmt(b_) and not pending:
        return fmt(a)
    lo = min(a, b_)
    if kind == "t":
        if lo >= 100:
            raise SystemExit("pair(): minutes are not used in Table 4")
        scale, unit = ((1e-3, "ms") if lo < 1 else (1, "s"))
    else:
        scale, unit = next((k, u) for u, k in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3), ("B", 1)) if lo >= k)
    return f"{_num(a / scale)}$\\to${red(_num(b_ / scale), pending)}\\,{unit}"


def params(n):
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e8:
        return f"{n / 1e6:.0f}M"
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def tokens(seq):
    return f"{seq:,}".replace(",", "{,}")


def write(name, lines):
    if DRY:
        return
    TABS.mkdir(parents=True, exist_ok=True)
    # the closing rule lives in the file: after \input, a \bottomrule in main.tex is a misplaced \noalign
    (TABS / f"{name}.tex").write_text("\n".join(lines + [r"\bottomrule"]) + "\n", encoding="utf-8", newline="\n")
    print("table", name)


def tab_cnn(M=None):
    """Table 2 (optimised protocol): Model | Params | acc. | C prove, verify, proof | Kpre verify, proof."""
    M = M or basic_measured()
    lines = []
    for m in CNN_ORDER:
        acc = M.get(m, "facts", "int8_accuracy")
        row = f"{SHORT[m]} ({DATA[m]})"
        label = row + (r"$^\dagger$" if m == "resnet18_224" else "")
        cells = []
        for mode, metrics in (("C", ("prove", "verify", "bytes")), ("Kpre", ("verify", "bytes"))):
            for k in metrics:
                v, p = opt_cnn(m, k, mode, where=("Table 2", row, f"{mode} {k}"))
                cells.append(red((b if k == "bytes" else t)(v), p))
        lines.append(" & ".join([label, params(M.get(m, "facts", "n_params")), f"{acc * 100:.1f}\\%"] + cells) + r" \\")
    write("cnn", lines)


# Table 3's rows by prompt length (the first block's heading row is in main.tex: after \midrule and
# \input a \multicolumn is not allowed, so only the second heading is written here)
TAB_LLM_BLOCKS = {64: ["gpt2", "qwen3-4b", "llama2-7b", "llama2-13b"],
                  2048: ["opt-125m", "opt-1.3b", "opt-6.7b", "llama2-7b", "llama2-13b"]}
TAB_LLM_PICK = [(m, seq) for seq, ms in TAB_LLM_BLOCKS.items() for m in ms]


def tab_llm():
    """Table 3 (optimised protocol): Model | Params | C prove, verify, GPU verify, proof | Kpre verify, proof."""
    lines = []
    for i, (seq, models) in enumerate(TAB_LLM_BLOCKS.items()):
        if i:
            lines.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{tokens(seq)}-token prompts}}}} \\")
        for model in models:
            row = f"{LLM_NAME[model]} ({seq})"
            cols = [("C", "prove", t), ("C", "verify", t), ("C", "gpu", t), ("C", "bytes", b),
                    ("Kpre", "verify", t), ("Kpre", "bytes", b)]
            cells = []
            for mode, k, fmt in cols:
                v, p = opt_llm(model, seq, k, mode, where=("Table 3", row, f"{mode} {k}"))
                cells.append(red(fmt(v), p))
            n, _ = opt_llm(model, seq, "params")
            lines.append(" & ".join([LLM_NAME[model], params(n)] + cells) + r" \\")
    write("llm", lines)
    return False   # no extrapolated row: Table 3 has full builds only


# Table 4: (label, our model, prompt length or None for an image model, setting, the verifier column)
TAB_OPT = [("LeNet-5 (MNIST)", "lenet5", None, "C", "verify"), ("VGG-16 (CIFAR)", "vgg16", None, "C", "verify"),
           ("ResNet-18 (CIFAR)", "resnet18_cifar", None, "C", "verify"),
           ("GPT-2 (64)", "gpt2", 64, "C", "verify"), ("GPT-2 (512)", "gpt2", 512, "C", "verify"),
           ("Llama-2-7B (1)", "llama2-7b", 1, "C", "verify"), ("Llama-2-7B (64)", "llama2-7b", 64, "C", "verify"),
           ("Llama-2-7B (2{,}048)$^g$", "llama2-7b", 2048, "C", "gpu"),
           ("Qwen3-4B (8)$^k$", "qwen3-4b", 8, "Kpre", "verify")]


def _plain(label: str) -> str:
    return re.sub(r"\$\^.\$|\$\^\\dagger\$", "", label).replace("{,}", ",")


def opt_pairs(M=None, MO=None):
    """Table 4's rows: (label, model, seq, basic (prove, verify, bytes), optimised (prove, verify, bytes),
    optimised pending flags).  The two sides come from the same settings; at 2,048 tokens the prover and
    the proof from the CPU-verifier runs (as in Table 3) and the verifier from the GPU-verifier runs."""
    out = []
    for label, model, seq, mode, vcol in TAB_OPT:
        cols = ("prove", vcol, "bytes")
        where = lambda k: ("Table 4", _plain(label), k)   # noqa: E731
        if seq is None:
            before = tuple(basic_cnn(model, k) for k in ("prove", "verify", "bytes"))
            after = [opt_cnn(model, k, where=where(k)) for k in ("prove", "verify", "bytes")]
        else:
            before = tuple(basic_llm(model, seq, k, mode) for k in cols)
            after = [opt_llm(model, seq, k, mode, where=where(k)) for k in cols]
        out.append((label, model, seq, before, tuple(v for v, _ in after), tuple(p for _, p in after)))
    return out


def tab_opt(M=None, MO=None):
    """Table 4: basic -> optimised protocol on the same machines, each cell 'a -> b'."""
    lines = []
    for label, model, seq, before, after, pend in opt_pairs(M, MO):
        if None in before:
            raise SystemExit(f"Table 4, {label}: the basic run lacks {before}")
        lines.append(" & ".join([label, pair(before[0], after[0], "t", pend[0]), pair(before[1], after[1], "t", pend[1]),
                                 pair(before[2], after[2], "b", pend[2])]) + r" \\")
    write("opt", lines)


def write_hardware(M=None, starred=False, MO=None):
    """Macros for the text: the headline machine, the Table 3 note on extrapolated rows, and the number of
    pending optimised numbers (main.tex can print its red note only while it is not zero)."""
    M = M or basic_measured()
    rows = llm_rows("C")
    hw = {v[3] for d in rows.values() for v in d.values()} | {M.prover_hw}
    verifier = {r.get("verifier_hw") for r in _read("llm_full_model.csv")
                if r.get("threads") == DEFAULT["threads"] and r.get("variant") == DEFAULT["variant"]} | {M.verifier_hw}
    dirs = opt_dirs()
    if dirs:   # the optimised runs must be "on the same machines" as the basic one
        MO = opt_measured()
        hw |= {MO.prover_hw} | {r.get("prover_hw") for r in _llm_table(_llm_dirs(dirs))}
        # (a GPU client's verifier string names the GPU first: '<gpu> (GPU client) + <cpu> x<n> threads')
        verifier |= {MO.verifier_hw} | {re.sub(r"^.* \(GPU client\) \+ ", "", r.get("verifier_hw", ""))
                                         for r in _llm_table(_llm_dirs(dirs))
                                         if r.get("threads") == DEFAULT["threads"]}
    if len(hw - {None}) != 1 or len(verifier - {None}) != 1:
        raise SystemExit(f"the tables mix machines: {sorted(map(str, hw))} / {sorted(map(str, verifier))}")
    gpu, cpu = next(iter(hw - {None})), next(iter(verifier - {None}))
    cpu_name, _, threads = cpu.partition(" x")
    cpu_name = re.sub(r"\(R\)|\(TM\)|\bCPU\b|@.*$|\d+-Core Processor", "", cpu_name)
    pending = pending_rows("pending")
    # a tie before the name's last word ('RTX 2080~Ti'), so the text never breaks a line inside it
    lines = [r"\newcommand{\ProverGPU}{" + re.sub(r" (\S+)$", r"~\1", hw_short(gpu)) + "}",
             r"\newcommand{\VerifierCPU}{" + re.sub(r"\s+", " ", cpu_name).strip() + "}",
             r"\newcommand{\VerifierThreads}{" + threads.split()[0] + "}",
             r"\newcommand{\LLMNote}{" + (r" ($^\ast$extrapolated from 1- and 2-block builds)" if starred else "") + "}",
             r"\newcommand{\NPendingCells}{" + str(len(pending)) + "}"]
    if DRY:
        return
    TABS.mkdir(parents=True, exist_ok=True)
    (TABS / "hardware.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("table hardware:", hw_short(gpu), "/", cpu, f"/ {len(pending)} pending")


RATIO_STYLE = "fraction"   # Table 5's cells: theirs/ours as 'n\times' or '1/n' ("arrows": n\times with up/down)


def fac(theirs, ours):
    """theirs/ours as Table 5 prints it, and the ratio.  'fraction': '81$\\times$' when ours is 81x faster or
    smaller, '1/72' when it is 72x slower or larger (the printed value is theirs/ours); 'arrows': '81$\\times$
    $\\uparrow$' / '72$\\times$$\\downarrow$'."""
    if not theirs or not ours:
        return "--", None
    q = theirs / ours
    if 0.99 < q < 1.01:
        return "same", q
    v = q if q >= 1 else 1 / q
    s = f"{v:,.0f}".replace(",", "{,}") if v >= 100 else (f"{v:.0f}" if v >= 10 else f"{v:.1f}")
    if RATIO_STYLE == "arrows":
        return s + r"$\times$" + (r"$\uparrow$" if q >= 1 else r"$\downarrow$"), q
    return (s + r"$\times$" if q >= 1 else "1/" + s), q


def _ours_t5(model, seq, ch, where, system=""):
    """Our (prove, verify, bytes) and pending flags of one Table 5 row: the GPU verifier at 2,048 tokens."""
    if seq is None:
        extra = {"verify": TABLE5_VERIFIER_EXTRA_TAGS.get(system, ())}
        vals = [opt_cnn(model, k, "C", ch, where=where and (*where, k), extra=extra.get(k, ()))
                for k in ("prove", "verify", "bytes")]
    else:
        mode = _llm_mode(model, seq)
        vals = [opt_llm(model, seq, k, mode, ch, where=where and (*where, k))
                for k in ("prove", "gpu" if seq >= LONG else "verify", "bytes")]
    return [v for v, _ in vals], [p for _, p in vals]


def ratio_rows(MO=None):
    """Table 5's rows: (system label, model label, [(theirs, ours) for prover, verifier, proof], interactive?,
    [pending?] per column).  Fiat-Shamir proofs against the non-interactive systems (where they are
    stored), the interactive protocol against zkLLM and Maverick; the last row is zkLLM's code on our GPU."""
    curated = {(r["system"], r["model"], r["seq"]): r for r in _read("reported_curated.csv")}
    out = []
    for system, pub, pub_seq, model, seq, mode in OPT_MATCHES:
        r = curated[(system, pub, pub_seq)]
        fs_ok = system in NONINTERACTIVE and not any(_ours_t5(model, seq, "fs", None)[1][::2])
        ch = "fs" if fs_ok else "int"
        what = SHORT[model] if seq is None else f"{LLM_NAME[model]} ({tokens(seq)})"
        name = system.split(" (")[0]
        us, pend = _ours_t5(model, seq, ch, ("Table 5", f"{name} / {what.replace('{,}', ',')}"), name)
        them = [_f(r["prover_s"]), _f(r["verifier_s"]), _f(r["proof_bytes"])]
        letters = [SYSTEM_MARK[name]] if name in SYSTEM_MARK else []
        if system == MAVERICK:
            them[1] += MAVERICK_NONLINEAR_S
            letters.append(MAVERICK_MARK)
        elif system in NONINTERACTIVE and ch == "int":
            letters.append(INTERACTIVE_MARK)   # interactive: no Fiat-Shamir cell of this model is stored
            spec = llm_spec(model, seq) if seq is not None else None
            _note(("Table 5", f"{name} / {what.replace('{,}', ',')}", "Fiat-Shamir (mark *)"), model,
                  llm_cell(model, seq, "C", "fs", spec["lam"], spec["tags"]) if spec else
                  cnn_cell("C", "fs", LAM, OPTIMISED[("cnn", "C")]["tags"]), "(the interactive cell, marked *)",
                  "acceptable")
        out.append((name + _marks(letters), what, list(zip(them, us)), ch == "int", pend))
    ours7, p7 = opt_llm("llama2-7b", 2048, "prove", where=("Table 5", "zkLLM^e / Llama-2-7B (2,048)", "prove"))
    out.append((ZKLLM_OWN, f"Llama-2-7B ({tokens(2048)})",
                [(zkllm_l40s_whole_model_s("llama2-7b", 2048), ours7), (None, None), (None, None)], True,
                [p7, False, False]))
    return out


def tab_ratios(MO=None):
    """Table 5: the optimised protocol against published systems (theirs/ours); bold: ours better on all three."""
    lines = []
    for system, what, cells, _, pend in ratio_rows(MO):
        shown = [fac(a, o) for a, o in cells]
        cols = [system, what] + [red(s, p) for (s, _), p in zip(shown, pend)]
        if all(q is not None and q >= 1 for _, q in shown):
            cols = [rf"\textbf{{{c}}}" for c in cols]
        lines.append(" & ".join(cols) + r" \\")
    write("ratios", lines)


def missing(M=None, MO=None) -> list[str]:
    """Every stored number of the BASIC run that the tables and figures need and its tables lack (the
    optimised numbers it lacks are pending, not missing: see PENDING)."""
    M = M or basic_measured()
    out = []
    for m in CNN_ORDER:
        need = [("facts", "n_params", ""), ("facts", "int8_accuracy", ""), ("sampling", "open_all_bytes", ""),
                ("sampling", "paths_bytes_shared", 40)]
        need += [(f"defence_C_int_lam{l}_rate4", k, "") for l in (40, 80, 128) for k in ("bytes_total", "soundness_bits")]
        need += [(f"defence_C_int_lam{LAM}_rate4", "prove_forward", ""),
                 (f"defence_Kpre_int_lam{LAM}_rate4", "prove_forward", "")]
        out += [f"cnn/{m}/{c} {k}" for c, k, lam in need if M.get(m, c, k, lam=lam) is None]
        if not _path_curve(M, m)[0]:
            out.append(f"cnn/{m}/sampling paths_bytes_shared_k")
    # Table 4's left-hand side and the placeholders of Table 3's C and Kpre CPU-verifier columns
    for label, model, seq, mode, vcol in TAB_OPT:
        if seq is not None and None in [basic_llm(model, seq, k, mode) for k in ("prove", vcol, "bytes")]:
            out.append(f"llm/{model} {mode} T{seq} (Table 4, basic side)")
    for model, seq in TAB_LLM_PICK:
        for mode in ("C", "Kpre"):
            if basic_llm(model, seq, "verify", mode) is None:
                out.append(f"llm/{model} {mode}:int lam{LAM} T{seq} threads {DEFAULT['threads']} (Table 3 placeholder)")
    if zkllm_l40s_whole_model_s("llama2-7b", 2048) is None:
        out.append(f"{ZKLLM_L40S.relative_to(ROOT)} llama2-7b T2048 (summarise.py --csv)")
    return out


def pending_rows(status=None) -> list[dict]:
    """PENDING without duplicates, in the order the builders met them."""
    seen, out = set(), []
    for p in PENDING:
        key = tuple(p[f] for f in PENDING_FIELDS)
        if key not in seen and (status is None or p["status"] == status):
            seen.add(key)
            out.append(p)
    return out


def write_pending_csv(path: Path) -> None:
    rows = pending_rows()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=PENDING_FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows: {sum(r['status'] == 'pending' for r in rows)} pending, "
          f"{sum(r['status'] == 'acceptable' for r in rows)} acceptable stand-ins)")


OLD_FIGURES = ("llm_zkllm", "security_detection")   # dropped from the report: deleted if present


def build_all(figures: bool = True):
    """Every figure and table (in --check mode: computed, not written)."""
    PENDING.clear()
    if figures:
        fig_overview()
        fig_protocol()
        fig_security()
        fig_cost()
    tab_cnn()
    starred = tab_llm()
    tab_opt()
    tab_ratios()
    write_hardware(starred=starred)


def main():
    global TABLES, OPT_TABLES, DRAFT, DRY
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=PLATFORM,
                    help="the basic protocol's run, tables_<platform>/ (default: $PVI_PLATFORM, else l40s)")
    ap.add_argument("--optimised", default="",
                    help="comma-separated runs of the optimised protocol on the same machines, in order (a later "
                         "run's cell replaces an earlier one's; a run not yet aggregated is skipped with a warning). "
                         "The report: l40s_improved,l40s_improved2")
    ap.add_argument("--check", action="store_true",
                    help="write nothing: list the basic run's missing cells and the pending optimised numbers")
    ap.add_argument("--pending-csv", type=Path, help="write the pending (and acceptable stand-in) numbers to this CSV")
    ap.add_argument("--strict", action="store_true", help="refuse to write while any optimised number is pending")
    ap.add_argument("--draft", action="store_true", help="write figures despite layout problems (printed)")
    args = ap.parse_args()
    DRAFT = args.draft
    TABLES = tables_dir(args.platform)
    dirs = []
    for name in [s.strip() for s in args.optimised.split(",") if s.strip()]:
        d = tables_dir(name, required=False)
        if d is None:
            print(f"warning: no tables_{name}/ (not yet run or aggregated): skipped; its numbers stay pending",
                  file=sys.stderr)
        else:
            dirs.append(d)
    OPT_TABLES = dirs
    gaps = missing()
    if gaps:   # before any file is written
        print(f"{TABLES.name}: {len(gaps)} missing", *gaps, sep="\n  ")
        raise SystemExit(1)
    DRY = True                 # first pass: compute everything, write nothing
    build_all()
    pend = pending_rows("pending")
    acc = pending_rows("acceptable")
    if args.pending_csv:
        write_pending_csv(args.pending_csv)
    if args.check or (args.strict and pend):
        print(f"{TABLES.name} + {[d.name for d in dirs]}: 0 missing basic cells; {len(pend)} pending optimised "
              f"numbers (a basic placeholder, red), {len(acc)} shown from an acceptable stand-in (black)")
        for p in pend + acc:
            print(f"  {p['status']:10s} {p['asset']:8s} {p['row']:30s} {p['model']:15s} {p['column']:12s} "
                  f"{p['optimised_cell']}  <- {p['basic_cell']}")
        raise SystemExit(1 if args.strict and pend else 0)
    DRY = False
    build_all()
    for name in OLD_FIGURES:
        old = FIGS / f"{name}.pdf"
        if old.exists():
            old.unlink()
            print("removed", old.name)


if __name__ == "__main__":
    main()

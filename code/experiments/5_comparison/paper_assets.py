"""Every figure and table body of the report, built from the stored benchmark tables of two runs.

    python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved   # the report
        # -> ../report/figures/*.pdf, ../report/tables/*.tex
    python experiments/5_comparison/paper_assets.py --platform l40s --optimised l40s_improved --check
        # only list the missing cells of both runs, write nothing

Nothing here runs a model; every number is read from the stored benchmark tables
(``aggregate.py``, ``literature.py``) and from zkLLM's run on our GPU
(``artifacts/results/zkllm_l40s/summary.csv``), so the report can be rebuilt without a GPU.
``--platform`` (default ``l40s``, or ``$PVI_PLATFORM``) is the *basic protocol*, measured with the
code of the report's commit: Tables 2-3, Figure 4 and the hollow points of Figure 3.
``--optimised`` (``l40s_improved``) is the *optimised protocol* on the same machines: the filled
points of Figure 3, the right-hand side of Table 4 and every ratio of Table 5.  Without
``--optimised`` only the basic assets are written.  Both runs must hold every cell the report
draws: ``--check`` lists the missing ones, and without it the script refuses before writing
anything.  Table 3 stars the rows extrapolated from 1- and 2-block builds (none on ``l40s``), and
Figure 4 and the text's ranges use the full-depth builds only.

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
from collections import defaultdict
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
OPT_TABLES: Path | None = None         # the optimised run's tables; set by main() from --optimised
ZKLLM_L40S = ROOT / "artifacts" / "results" / "zkllm_l40s" / "summary.csv"   # zkLLM's code on our GPU


def tables_dir(platform: str) -> Path:
    """A platform's tables, written by ``aggregate.py --platform <platform>``."""
    path = BASE / f"tables_{platform}"
    if not path.is_dir():
        raise SystemExit(f"no {path} (aggregate.py --platform <name> writes one per re-measurement; "
                         f"the report's numbers are l40s and l40s_improved)")
    return path


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


class Measured:
    """Index over one run's ``measured_summary.csv`` (median and mean per model, cell, metric)."""

    def __init__(self, tables: Path | None = None) -> None:
        self.tables = tables or TABLES
        self.idx = {}
        for r in _read("measured_summary.csv", self.tables):
            key = (r["suite"], r["model"], r["cell"], r["metric"], r.get("batch", ""), r.get("attack", ""),
                   r.get("lam", ""), r.get("stage", ""), r.get("k", ""))
            self.idx[key] = r
        timed = {r.get("prover_hw") for k, r in self.idx.items() if k[0] == "cnn" and k[2].startswith("defence_")}
        if len(timed) > 1:
            raise SystemExit(f"CNN timings of several machines in one tables directory: {sorted(map(str, timed))}")
        self.prover_hw = next(iter(timed), None)
        # CNN rows are not filtered by thread count, so a CNN job run with another --threads would
        # enter Table 2 silently; the verifier string ('<cpu> x<n> threads') catches it
        vtimed = {r.get("verifier_hw") for k, r in self.idx.items() if k[0] == "cnn" and k[2].startswith("defence_")}
        if len(vtimed) > 1:
            raise SystemExit(f"CNN verifier timings of several CPUs/thread counts: {sorted(map(str, vtimed))}")
        self.verifier_hw = next(iter(vtimed), None)

    def get(self, model, cell, metric, lam=""):
        """Median of one CNN metric (batch, attack, stage and k left empty)."""
        r = self.idx.get(("cnn", model, cell, metric, "", "", str(lam), "", ""))
        return _f(r["median"]) if r else None

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


def llm_rows(mode="C", lam=DEFAULT["lam"], threads=DEFAULT["threads"], variant=DEFAULT["variant"],
             challenges="int", tables: Path | None = None) -> dict:
    """``{(model, seq): {metric: (value, provenance, params, prover_hw)}}`` from one run's
    ``llm_full_model.csv``.  Where a run has both a full build and 1-2 block builds, the measured row is used."""
    out: dict = defaultdict(dict)
    for r in _read("llm_full_model.csv", tables):
        if (r["mode"], r["challenges"]) != (mode, challenges) or _f(r["lam"]) != lam or r.get("batch"):
            continue
        if (r.get("rate", ""), r.get("threads", ""), r.get("variant", "")) != (DEFAULT["rate"], threads, variant):
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
    verify = sum(d[k][0] for k in VERIFY if k in d)
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


# ------------------------------------------------------------------------------ the optimised protocol
# The optimised protocol of each (model, prompt length): the planning rule with the compact encoding
# ("_wire_polauto"; GPT-2 also prunes the last block to the last position, "_wire_prune_polauto"); at
# 2,048 tokens the streaming GPU verifier, whose proof is the basic one ("_gpuv_stream"); Qwen3-4B in
# Maverick's setting (Kpre, lambda 40, one verifier thread) with embedding look-ups.
CNN_OPT_CELL = "defence_C_{ch}_lam{lam}_rate4_wire_polauto"
OPT_LLM = {("gpt2", 64): dict(variant="_wire_prune_polauto"), ("gpt2", 512): dict(variant="_wire_prune_polauto"),
           ("llama2-7b", 1): dict(variant="_wire_polauto"), ("llama2-7b", 64): dict(variant="_wire_polauto"),
           ("qwen3-4b", 8): dict(mode="Kpre", lam=40, threads="1", variant="_thr1_wire_prune_lookups")}
OPT_LLM.update({(m, 2048): dict(variant="_gpuv_stream")
                for m in ("opt-125m", "opt-1.3b", "opt-6.7b", "llama2-7b", "llama2-13b")})
# ... and the basic protocol's cell it is compared with in Table 4 (same machines)
BASIC_LLM = {("qwen3-4b", 8): dict(mode="Kpre", lam=40, threads="1", variant="_thr1"),
             ("llama2-7b", 2048): dict(variant="_gpuv")}


def _sel(sel: dict):
    """(mode, lam, threads, variant) of an OPT_LLM / BASIC_LLM entry."""
    return sel.get("mode", "C"), sel.get("lam", DEFAULT["lam"]), sel.get("threads", DEFAULT["threads"]), \
        sel.get("variant", DEFAULT["variant"])


def llm_cost(key, sel: dict, tables: Path | None = None, challenges="int"):
    """(prove, verify, bytes, params) of one (model, seq) in one selection of one run, or None if absent."""
    mode, lam, thr, var = _sel(sel)
    d = llm_rows(mode, lam=lam, threads=thr, variant=var, challenges=challenges, tables=tables).get(key)
    return _llm_cost(d) if d and "prove_forward" in d else None


def opt_cost(MO: Measured, model, seq, challenges="int", lam=DEFAULT["lam"]):
    """(prove, verify, bytes) of the optimised protocol on one model (seq None: an image model)."""
    if seq is None:
        c = MO.cost(model, CNN_OPT_CELL.format(ch=challenges, lam=lam))
        return None if c[0] is None else c
    c = llm_cost((model, seq), OPT_LLM[(model, seq)], OPT_TABLES, challenges)
    return None if c is None else c[:3]


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
            "llama2-7b": "Llama-2-7B", "llama2-13b": "Llama-2-13B", "qwen3-4b": "Qwen3-4B"}

# published systems grouped by proof family, for the cost figure (Plonk/Groth16 and the other ZK proofs
# share one marker)
FAMILY = {"zkCNN": "sumcheck", "zkLLM": "sumcheck", "zkGPT": "sumcheck", "DeepProve": "sumcheck",
          "zkPyTorch": "sumcheck", "Jolt": "sumcheck", "SLP": "sumcheck",
          "ZKML": "other", "EZKL": "other", "ZKTorch": "other", "Bionetta": "other", "vCNN": "other",
          "Mystique": "other", "LAMP": "other", "Maverick": "known"}
FAM_STYLE = {"sumcheck": ("^", "sum-check zkSNARKs"), "other": ("D", "Plonk/Groth16/other ZK"),
             "known": ("*", "Maverick")}
# a few landmark systems are named on each panel: (system, model[, seq]) -> (label, offset in points, ha)
# where: ("off", (dx, dy)) in points from the marker, or ("at", (x, y)) in data coordinates with a leader line
LABELS = {
    "prover": {("zkCNN", "VGG-16 CIFAR-10"): ("zkCNN", ("at", (2e6, 150)), "center"),
               ("zkLLM", "Llama-2-7B"): ("zkLLM", ("off", (-4, 7)), "right"),
               ("DeepProve", "GPT-2", "64"): ("DeepProve", ("at", (9e7, 6)), "right")},
    "verifier": {("zkCNN", "VGG-16 CIFAR-10"): ("zkCNN", ("off", (4, 5)), "left"),
                 ("zkLLM", "Llama-2-7B"): ("zkLLM", ("at", (2.5e9, 0.22)), "center"),
                 ("ZKTorch", "Llama-2-7B (1 token)"): ("ZKTorch", ("off", (-4, 6)), "right")},
    "proof": {("zkCNN", "VGG-16 CIFAR-10"): ("zkCNN", ("off", (4, 5)), "left"),
              ("zkLLM", "Llama-2-7B"): ("zkLLM", ("at", (3e9, 3e4)), "center"),
              ("DeepProve", "GPT-2", "64"): ("DeepProve", ("off", (-4, 6)), "right")},
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


def layout_problems(fig) -> list[str]:
    """Text below 7 pt, outside the canvas, overlapping other text, or covering a scatter marker."""
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
        pts = [p for coll in ax.collections for p in coll.get_transform().transform(coll.get_offsets())]
        for t in ax.texts:
            bb = t.get_window_extent(r).expanded(1.0, 1.0)
            for x, y in pts:
                if bb.x0 + 1 < x < bb.x1 - 1 and bb.y0 + 1 < y < bb.y1 - 1:
                    bad.append(f"{t.get_text()!r} covers a marker")
                    break
    return bad


def save(fig, name: str, width: float):
    """Write figures/<name>.pdf on its fixed canvas (no tight cropping: the canvas is the printed size)."""
    assert abs(fig.get_size_inches()[0] - width) < 0.01, (name, fig.get_size_inches(), width)
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


def fig_overview():
    """Figure 1: (a) one random path, (b) one tampered neuron the path misses, (c) every layer checked."""
    white = lambda n: ("white", "0.35")   # noqa: E731
    fig, ax = plt.subplots(figsize=OVERVIEW)
    _network(ax, white)
    _path(ax, white)
    _finish(ax)
    save(fig, "overview_path", OVERVIEW[0])

    def attacked(n):
        if n == TAMPERED:
            return ATTACK, ATTACK
        return (ATTACK_LIGHT, "0.35") if n[0] > TAMPERED[0] else ("white", "0.35")

    fig, ax = plt.subplots(figsize=OVERVIEW)
    _network(ax, attacked)
    _path(ax, attacked)
    ax.annotate("wrong value", xy=POS[TAMPERED], xytext=(0.2, 2.05), color=ATTACK,
                arrowprops=dict(arrowstyle="->", color=ATTACK, lw=0.6, shrinkB=3))
    ax.text(2.55, 2.05, "recomputed\nhonestly", color="0.35", ha="center", va="center", linespacing=1.0)
    _finish(ax)
    save(fig, "overview_attack", OVERVIEW[0])

    fig, ax = plt.subplots(figsize=OVERVIEW)
    _network(ax, white)
    for n in PATH:
        ax.scatter(*POS[n], s=30, facecolor="white", edgecolor="0.35", lw=0.7, zorder=3)
    for l in range(1, 4):
        h = (WIDTHS[l] - 1) * 0.55 / 2 + 0.28
        ax.add_patch(FancyBboxPatch((l - 0.2, -h), 0.4, 2 * h, boxstyle="round,pad=0.02",
                                    fc="none", ec=OURS, lw=1.0, zorder=2))
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
    # messages (4)-(6) exist in setting C only
    xb, y0, y1 = W_ - 0.12, steps[3][0] - 0.13, steps[5][0] + 0.1
    ax.plot([xb - 0.04, xb, xb, xb - 0.04], [y0, y0, y1, y1], color="0.35", lw=0.6)
    ax.text(xb + 0.05, (y0 + y1) / 2, f"setting {SF_C} only", rotation=270, ha="center", va="center",
            color="0.35")
    save(fig, "protocol", W_)


# ----------------------------------------------------------------------------------- figure 3
SECURITY_MODELS = ["mlp_mnist", "lenet5", "vgg16", "resnet18_224"]
# one marker shape per model (no per-model colours); hollow: basic, filled: optimised
SECURITY_MARKER = {"mlp_mnist": "o", "lenet5": "s", "vgg16": "^", "resnet18_224": "D"}
# each model's name: (anchor point, offset in points, ha, va); anchors: the optimised or basic point at a lambda
SECURITY_LABEL = {"mlp_mnist": ("MLP", ("basic", 128), (0, 6), "center", "bottom"),
                  "lenet5": ("LeNet-5", ("optimised", 128), (-5, 0), "right", "center"),
                  "vgg16": ("VGG-16", ("basic", 128), (0, 6), "center", "bottom"),
                  "resnet18_224": ("ResNet-18\n(224px)", ("basic", 40), (-4, -14), "right", "top")}


def _path_curve(M, m):
    """(bytes, bits) of the path protocol with s = 1, 3, 10, ... paths, in order of s."""
    rows = [r for k, r in M.idx.items() if k[:4] == ("cnn", m, "sampling", "paths_bytes_shared_k")]
    rows.sort(key=lambda r: float(r["k"]))
    return [_f(r["median"]) for r in rows], [_f(r.get("bits")) for r in rows]


def _ours_curve(M, m, cell):
    """{lambda: (bytes, bits)} of one protocol at lambda = 40, 80, 128 (the cells that exist)."""
    pts = {lam: (M.get(m, cell.format(lam=lam), "bytes_total"), M.get(m, cell.format(lam=lam), "soundness_bits"))
           for lam in (40, 80, 128)}
    return {lam: p for lam, p in pts.items() if None not in p}


def fig_security(M, MO=None):
    """Figure 3: security bits against bytes per query, the path protocol against the basic and the
    optimised protocol (lambda = 40, 80, 128).  The bits axis is linear: the path protocol stays near
    zero bits until it opens almost the whole model and trace."""
    size = (3.33, 2.2)
    fig, ax = plt.subplots(figsize=size)
    fig.subplots_adjust(left=0.115, right=0.985, bottom=0.165, top=0.80)
    top = 190
    for m in SECURITY_MODELS:
        mk = SECURITY_MARKER[m]
        bx, by = _path_curve(M, m)
        if bx and None not in by:
            cap = M.get(m, "sampling", "open_all_bytes")
            ax.plot(bx + [cap, cap], by + [max(by), top], ":", color=ANCH, lw=1.1, zorder=2)
            x40 = M.get(m, "sampling", "paths_bytes_shared", lam=40)
            ax.scatter([x40], [40], s=13, marker=mk, facecolor="white", edgecolor=ANCH, lw=0.8, zorder=4)
        drawn = {}
        for kind, cell, fill, lw in (("basic", "defence_C_int_lam{lam}_rate4", "white", 0.6),
                                     ("optimised", CNN_OPT_CELL.format(ch="int", lam="{lam}"), OURS, 1.0)):
            pts = _ours_curve(M if kind == "basic" else MO, m, cell) if (kind == "basic" or MO) else {}
            if not pts:
                continue
            drawn[kind] = pts
            xy = [pts[lam] for lam in sorted(pts)]
            ax.plot(*zip(*xy), "-", color=OURS, lw=lw, zorder=3)
            ax.scatter(*zip(*xy), s=13, marker=mk, facecolor=fill, edgecolor=OURS, lw=0.8, zorder=4)
        text, (kind, lam), off, ha, va = SECURITY_LABEL[m]
        if kind not in drawn:          # no optimised run of this model: label its basic point
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
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.55, 0.905), ncol=3, frameon=False,
               handlelength=2.2, columnspacing=1.4, handletextpad=0.5, borderaxespad=0)
    fig.text(0.015, 0.985, "Optimised: 128 bits cost fewer bytes than the paths' 40", ha="left",
             va="top", weight="bold")
    save(fig, "security_bits", size[0])


# ----------------------------------------------------------------------------------- figure 4
def _ours_points(M):
    ours_c, ours_k = [], []
    for m in CNN_ORDER:
        n = M.get(m, "facts", "n_params")
        for cell, out in ((f"defence_C_int_lam{LAM}_rate4", ours_c), (f"defence_Kpre_int_lam{LAM}_rate4", ours_k)):
            if M.get(m, cell, "prove_forward") is not None:
                out.append((n, *M.cost(m, cell), "cnn"))
    for mode, out, seqs in (("C", ours_c, (64, 2048)), ("Kpre", ours_k, (64,))):
        for (model, seq), d in sorted(llm_rows(mode).items()):
            if "prove_forward" in d and seq in seqs and _measured(d):
                p, v, by, n = _llm_cost(d)
                out.append((n, p, v, by, f"T{seq}"))
    return ours_c, ours_k


COST = (2.24, 1.75)       # printed at 0.32 of the text width


def fig_cost(M):
    """Figure 4 (basic protocol): prover time, verifier time and proof size against model size."""
    rep = [r for r in _read("reported_curated.csv") if _f(r["params"])]
    ours_c, ours_k = _ours_points(M)
    panels = (("prover_s", 0, "prover", "prover time (s)"), ("verifier_s", 1, "verifier", "verifier time (s)"),
              ("proof_bytes", 2, "proof", "proof size (bytes)"))
    for key, col, name, ylabel in panels:
        fig, ax = plt.subplots(figsize=COST)
        fig.subplots_adjust(left=0.19, right=0.97, bottom=0.2, top=0.97)
        named = set()
        for r in rep:
            y, x = _f(r[key]), _f(r["params"])
            if y is None:
                continue
            if r["system"].startswith("Anchuri"):
                ax.scatter(x, y, marker="X", s=30, c=ANCH, zorder=3)
                continue
            fam = family(r["system"])
            if fam is None:
                continue
            ax.scatter(x, y, marker=FAM_STYLE[fam][0], s=12 if fam != "known" else 24, c="none",
                       edgecolors=THEM, linewidths=0.6, zorder=2)
            lab = LABELS[name].get((r["system"], r["model"])) or LABELS[name].get((r["system"], r["model"], r["seq"]))
            if lab and lab[0] not in named:
                named.add(lab[0])
                text, (how, where), ha = lab
                kw = dict(xytext=where, textcoords="offset points") if how == "off" else \
                    dict(xytext=where, textcoords="data",
                         arrowprops=dict(arrowstyle="-", color="0.5", lw=0.4, shrinkA=1.5, shrinkB=2.5))
                ax.annotate(text, (x, y), color="0.3", ha=ha, va="center", path_effects=WHITE, zorder=5, **kw)
        for pts, colour in ((ours_c, OURS), (ours_k, OURS_K)):
            for n, *vals, kind in pts:
                if vals[col] is None:
                    continue
                ax.scatter(n, vals[col], marker="o" if kind == "cnn" else "s", s=16,
                           c="white" if kind == "T2048" else colour, edgecolors=colour, linewidths=0.8, zorder=4)
        if name == "proof":
            ax.plot([1e4, 3e10], [1e4, 3e10], "k--", lw=0.5, zorder=1)
            ax.text(1.5e5, 4.2e6, "int8 model", color="0.3", rotation=31, ha="center", va="center",
                    path_effects=WHITE)
        ax.set_xscale("log")
        ax.set_yscale("log")
        _log_axes(ax)
        ax.set_xlabel("model parameters")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", lw=0.3, alpha=0.4)
        save(fig, f"cost_{name}", COST[0])

    # one legend for the three panels, as its own strip: shape = workload, colour = setting, grey = published
    def mk(marker, ms=4, **kw):
        return Line2D([], [], marker=marker, ls="", ms=ms, **kw)
    row1 = [mk("o", color="0.3", label="CNN"), mk("s", color="0.3", label="LLM, 64 tokens"),
            mk("s", color="0.3", mfc="white", label="LLM, 2,048 tokens"),
            Patch(fc=OURS, ec=OURS, label=f"committed ({SF_C})"),
            Patch(fc=OURS_K, ec=OURS_K, label=f"known weights ({KPRE})")]
    row2 = [mk(FAM_STYLE[f][0], mfc="none", color=THEM, label=FAM_STYLE[f][1], ms=6 if f == "known" else 4)
            for f in ("sumcheck", "other", "known")]
    row2 += [mk("X", color=ANCH, label="Anchuri et al., one path", ms=5), Line2D([], [], ls="", label=" ")]
    handles = [h for col in zip(row1, row2) for h in col]   # a legend fills its columns top to bottom
    size = (7.0, 0.36)
    fig = plt.figure(figsize=size)
    fig.legend(handles=handles, loc="center", ncol=5, frameon=False, handletextpad=0.4, columnspacing=1.6,
               handlelength=1.1, labelspacing=0.3, borderaxespad=0)
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


def _num(v):
    """A number in a shared unit: 1{,}012, 375, 14.8, 0.30."""
    if v >= 99.5:
        return f"{v:,.0f}".replace(",", "{,}")
    return f"{v:.1f}" if v >= 1 else f"{v:.2f}"


def pair(a, b_, kind):
    """'a$\\to$b unit' (Table 4), both in the unit ``t()``/``b()`` give the smaller value (1{,}012$\\to$375\\,ms);
    one value if both print the same."""
    fmt = t if kind == "t" else b
    if fmt(a) == fmt(b_):
        return fmt(a)
    lo = min(a, b_)
    if kind == "t":
        if lo >= 100:
            raise SystemExit("pair(): minutes are not used in Table 4")
        scale, unit = ((1e-3, "ms") if lo < 1 else (1, "s"))
    else:
        scale, unit = next((k, u) for u, k in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3), ("B", 1)) if lo >= k)
    return f"{_num(a / scale)}$\\to${_num(b_ / scale)}\\,{unit}"


def params(n):
    if n >= 1e9:
        return f"{n / 1e9:.1f}B"
    if n >= 1e8:
        return f"{n / 1e6:.0f}M"
    return f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K"


def tokens(seq):
    return f"{seq:,}".replace(",", "{,}")


def write(name, lines):
    TABS.mkdir(parents=True, exist_ok=True)
    # the closing rule lives in the file: after \input, a \bottomrule in main.tex is a misplaced \noalign
    (TABS / f"{name}.tex").write_text("\n".join(lines + [r"\bottomrule"]) + "\n", encoding="utf-8", newline="\n")
    print("table", name)


def tab_cnn(M):
    """Table 2 (basic protocol): Model | Params | int8 | C prove, verify, proof | Kpre verify, proof."""
    lines = []
    for m in CNN_ORDER:
        c, k = f"defence_C_int_lam{LAM}_rate4", f"defence_Kpre_int_lam{LAM}_rate4"
        acc = M.get(m, "facts", "int8_accuracy")
        label = f"{SHORT[m]} ({DATA[m]})" + (r"$^\dagger$" if m == "resnet18_224" else "")
        lines.append(" & ".join([
            label, params(M.get(m, "facts", "n_params")), f"{acc * 100:.1f}\\%",
            t(M.total(m, c, PROVE)), t(M.total(m, c, VERIFY)), b(M.get(m, c, "bytes_total")),
            t(M.total(m, k, VERIFY)), b(M.get(m, k, "bytes_total"))]) + r" \\")
    write("cnn", lines)


# Table 3's rows by prompt length (the first block's heading row is in main.tex: after \midrule and
# \input a \multicolumn is not allowed, so only the second heading is written here)
TAB_LLM_BLOCKS = {64: ["gpt2", "qwen3-4b", "llama2-7b", "llama2-13b"],
                  2048: ["opt-125m", "opt-1.3b", "opt-6.7b", "llama2-7b", "llama2-13b"]}
TAB_LLM_PICK = [(m, seq) for seq, ms in TAB_LLM_BLOCKS.items() for m in ms]


def tab_llm():
    """Table 3 (basic protocol): Model | Params | C prove, verify, GPU verify, proof | Kpre verify, proof."""
    c, k, g = llm_rows("C"), llm_rows("Kpre"), llm_rows("C", variant="_gpuv")
    lines, starred = [], False
    for i, (seq, models) in enumerate(TAB_LLM_BLOCKS.items()):
        if i:
            lines.append(rf"\multicolumn{{8}}{{l}}{{\emph{{{tokens(seq)}-token prompts}}}} \\")
        for model in models:
            p, v, by, n = _llm_cost(c[(model, seq)])
            _, kv, kb, _ = _llm_cost(k[(model, seq)])
            gd = g.get((model, seq), {})
            gv = _llm_cost(gd)[1] if "prove_forward" in gd and ("verify_products" in gd or "verify_total" in gd) \
                else None
            extra = not (_measured(c[(model, seq)]) and _measured(k[(model, seq)]))   # starred
            starred |= extra
            name = LLM_NAME[model] + (r"$^\ast$" if extra else "")
            lines.append(" & ".join([name, params(n), t(p), t(v), t(gv), b(by), t(kv), b(kb)]) + r" \\")
    write("llm", lines)
    return starred


# Table 4: (label, our model, prompt length or None for an image model)
TAB_OPT = [("LeNet-5", "lenet5", None), ("VGG-16", "vgg16", None), ("ResNet-18 (CIFAR)", "resnet18_cifar", None),
           ("GPT-2 (64)", "gpt2", 64), ("GPT-2 (512)", "gpt2", 512), ("Llama-2-7B (1)", "llama2-7b", 1),
           ("Llama-2-7B (64)", "llama2-7b", 64), ("Llama-2-7B (2{,}048)$^g$", "llama2-7b", 2048),
           ("Qwen3-4B (8)$^k$", "qwen3-4b", 8)]


def opt_pairs(M, MO):
    """Table 4's rows: (label, model, seq, basic (prove, verify, bytes), optimised (prove, verify, bytes))."""
    out = []
    for label, model, seq in TAB_OPT:
        if seq is None:
            before = M.cost(model, f"defence_C_int_lam{LAM}_rate4")
        else:
            before = llm_cost((model, seq), BASIC_LLM.get((model, seq), {}))[:3]
        out.append((label, model, seq, before, opt_cost(MO, model, seq)))
    return out


def tab_opt(M, MO):
    """Table 4: basic -> optimised protocol on the same machines, each cell 'a -> b'."""
    lines = []
    for label, model, seq, before, after in opt_pairs(M, MO):
        if seq == 2048 and abs(after[2] / before[2] - 1) > 1e-3:
            raise SystemExit(f"{model} T{seq}: the streaming verifier changed the proof ({before[2]} -> {after[2]})")
        lines.append(" & ".join([label, pair(before[0], after[0], "t"), pair(before[1], after[1], "t"),
                                 pair(before[2], after[2], "b")]) + r" \\")
    write("opt", lines)


def write_hardware(M, starred, MO=None):
    """Macros for the text: the headline machine, and the Table 3 note on extrapolated rows."""
    rows = llm_rows("C")
    hw = {v[3] for d in rows.values() for v in d.values()} | {M.prover_hw}
    verifier = {r.get("verifier_hw") for r in _read("llm_full_model.csv")
                if r.get("threads") == DEFAULT["threads"] and r.get("variant") == DEFAULT["variant"]} | {M.verifier_hw}
    if MO is not None:   # Tables 4-5 compare the two runs "on the same machines"
        hw |= {MO.prover_hw} | {r.get("prover_hw") for r in _read("llm_full_model.csv", OPT_TABLES)}
        # (a GPU client's verifier string names the GPU first: '<gpu> (GPU client) + <cpu> x<n> threads')
        verifier |= {MO.verifier_hw} | {re.sub(r"^.* \(GPU client\) \+ ", "", r.get("verifier_hw", ""))
                                         for r in _read("llm_full_model.csv", OPT_TABLES)
                                         if r.get("threads") == DEFAULT["threads"]}
    if len(hw - {None}) != 1 or len(verifier - {None}) != 1:
        raise SystemExit(f"the tables mix machines: {sorted(map(str, hw))} / {sorted(map(str, verifier))}")
    gpu, cpu = next(iter(hw - {None})), next(iter(verifier - {None}))
    cpu_name, _, threads = cpu.partition(" x")
    cpu_name = re.sub(r"\(R\)|\(TM\)|\bCPU\b|@.*$|\d+-Core Processor", "", cpu_name)
    # a tie before the name's last word ('RTX 2080~Ti'), so the text never breaks a line inside it
    lines = [r"\newcommand{\ProverGPU}{" + re.sub(r" (\S+)$", r"~\1", hw_short(gpu)) + "}",
             r"\newcommand{\VerifierCPU}{" + re.sub(r"\s+", " ", cpu_name).strip() + "}",
             r"\newcommand{\VerifierThreads}{" + threads.split()[0] + "}",
             r"\newcommand{\LLMNote}{" + (r" ($^\ast$extrapolated from 1- and 2-block builds)" if starred else "") + "}"]
    TABS.mkdir(parents=True, exist_ok=True)
    (TABS / "hardware.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("table hardware:", hw_short(gpu), "/", cpu)


def fac(theirs, ours):
    """theirs/ours as Table 5 prints it ('81$\\times$$\\uparrow$': ours 81x better), and the ratio."""
    if not theirs or not ours:
        return "--", None
    q = theirs / ours
    if 0.99 < q < 1.01:
        return "same", q
    v = q if q >= 1 else 1 / q
    s = f"{v:,.0f}".replace(",", "{,}") if v >= 100 else (f"{v:.0f}" if v >= 10 else f"{v:.1f}")
    arrow = r"$\uparrow$" if q >= 1 else r"$\downarrow$"
    return s + r"$\times$" + arrow, q


def ratio_rows(MO):
    """Table 5's rows: (system label, model label, [(theirs, ours) for prover, verifier, proof], interactive?).
    Fiat-Shamir proofs against the non-interactive systems (where we ran them), the interactive protocol
    against zkLLM and Maverick; the last row is zkLLM's code on our GPU."""
    curated = {(r["system"], r["model"], r["seq"]): r for r in _read("reported_curated.csv")}
    out = []
    for system, pub, pub_seq, model, seq, mode in OPT_MATCHES:
        r = curated[(system, pub, pub_seq)]
        ch = "fs" if system in NONINTERACTIVE and opt_cost(MO, model, seq, "fs") is not None else "int"
        us = opt_cost(MO, model, seq, ch)
        them = [_f(r["prover_s"]), _f(r["verifier_s"]), _f(r["proof_bytes"])]
        mark = ""
        if system == MAVERICK:
            them[1] += MAVERICK_NONLINEAR_S
            mark = "$^b$"
        elif system in NONINTERACTIVE and ch == "int":
            mark = "$^a$"   # interactive: no Fiat-Shamir cell of this model was run
        what = SHORT[model] if seq is None else f"{LLM_NAME[model]} ({tokens(seq)})"
        out.append((system.split(" (")[0] + mark, what, list(zip(them, us)), ch == "int"))
    ours7 = opt_cost(MO, "llama2-7b", 2048)
    out.append(("zkLLM$^c$", f"Llama-2-7B ({tokens(2048)})",
                [(zkllm_l40s_whole_model_s("llama2-7b", 2048), ours7[0]), (None, None), (None, None)], True))
    return out


def tab_ratios(MO):
    """Table 5: the optimised protocol against published systems (theirs/ours); bold: ours better on all three."""
    lines = []
    for system, what, cells, _ in ratio_rows(MO):
        shown = [fac(a, o) for a, o in cells]
        cols = [system, what] + [s for s, _ in shown]
        if all(q is not None and q >= 1 for _, q in shown):
            cols = [rf"\textbf{{{c}}}" for c in cols]
        lines.append(" & ".join(cols) + r" \\")
    write("ratios", lines)


def missing(M, MO=None) -> list[str]:
    """Every stored number the tables and figures need, that the two runs' tables lack."""
    out = []
    for m in CNN_ORDER:
        need = [("facts", "n_params", ""), ("facts", "int8_accuracy", ""), ("sampling", "open_all_bytes", ""),
                (f"defence_Kpre_int_lam{LAM}_rate4", "prove_forward", ""),
                (f"defence_Kpre_int_lam{LAM}_rate4", "bytes_total", "")]
        need += [(f"defence_C_int_lam{l}_rate4", k, "") for l in (40, 80, 128) for k in ("bytes_total", "soundness_bits")]
        need += [(f"defence_C_int_lam{LAM}_rate4", "prove_forward", "")]
        out += [f"cnn/{m}/{c} {k}" for c, k, lam in need if M.get(m, c, k, lam=lam) is None]
        if not _path_curve(M, m)[0]:
            out.append(f"cnn/{m}/sampling paths_bytes_shared_k")
    # Tables 3-4 (basic side)
    sel = {("C", LAM, "8", ""): set(TAB_LLM_PICK) | {(m, s) for _, m, s in TAB_OPT if s and (m, s) not in BASIC_LLM},
           ("Kpre", LAM, "8", ""): set(TAB_LLM_PICK)}
    for key, s in BASIC_LLM.items():
        sel.setdefault(_sel(s), set()).add(key)
    for (mode, lam, thr, var), keys in sel.items():
        rows = llm_rows(mode, lam=lam, threads=thr, variant=var)
        for model, seq in sorted(keys):
            d = rows.get((model, seq), {})
            verified = "verify_products" in d or "verify_total" in d
            if "prove_forward" not in d or not verified or "bytes_total" not in d:
                out.append(f"llm/{model} {mode}:int lam{lam} T{seq} threads {thr} variant '{var}'")
    if MO is None:
        return out
    # the optimised run: Figure 3's filled points, Table 4's right-hand side and Table 5
    for m in ("mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar"):
        for ch in ("int", "fs"):
            for lam in (40, 80, 128):
                cell = CNN_OPT_CELL.format(ch=ch, lam=lam)
                out += [f"optimised cnn/{m}/{cell} {k}" for k in ("prove_forward", "bytes_total", "soundness_bits")
                        if MO.get(m, cell, k) is None]
    chs = {("gpt2", 64): ("int", "fs"), ("gpt2", 512): ("int", "fs")}
    for (model, seq), s in OPT_LLM.items():
        mode, lam, thr, var = _sel(s)
        for ch in chs.get((model, seq), ("int",)):
            d = llm_rows(mode, lam=lam, threads=thr, variant=var, challenges=ch, tables=MO.tables).get((model, seq), {})
            if "prove_forward" not in d or not ("verify_products" in d or "verify_total" in d) or "bytes_total" not in d:
                out.append(f"optimised llm/{model} {mode}:{ch} lam{lam} T{seq} threads {thr} variant '{var}'")
        if seq == 2048:   # the known-weights streaming cells of the text
            d = llm_rows("Kpre", variant=var, tables=MO.tables).get((model, seq), {})
            if "prove_forward" not in d or "bytes_total" not in d:
                out.append(f"optimised llm/{model} Kpre:int lam{LAM} T{seq} variant '{var}'")
    if zkllm_l40s_whole_model_s("llama2-7b", 2048) is None:
        out.append(f"{ZKLLM_L40S.relative_to(ROOT)} llama2-7b T2048 (summarise.py --csv)")
    return out


OLD_FIGURES = ("llm_zkllm", "security_detection")   # dropped from the report: deleted if present


def main():
    global TABLES, OPT_TABLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=PLATFORM,
                    help="the basic protocol's run, tables_<platform>/ (default: $PVI_PLATFORM, else l40s)")
    ap.add_argument("--optimised", default="",
                    help="the optimised protocol's run on the same machines (the report: l40s_improved); "
                         "empty: the basic assets only")
    ap.add_argument("--check", action="store_true", help="only list the stored numbers that are missing")
    ap.add_argument("--draft", action="store_true", help="write figures despite layout problems (printed)")
    args = ap.parse_args()
    global DRAFT
    DRAFT = args.draft
    TABLES = tables_dir(args.platform)
    M = Measured()
    MO = None
    if args.optimised:
        OPT_TABLES = tables_dir(args.optimised)
        MO = Measured(OPT_TABLES)
    gaps = missing(M, MO)
    if gaps or args.check:   # before any file is written
        names = TABLES.name + (f" + {OPT_TABLES.name}" if MO else "")
        print(f"{names}: {len(gaps)} missing", *gaps, sep="\n  ")
        raise SystemExit(1 if gaps else 0)
    fig_overview()
    fig_protocol()
    fig_security(M, MO)
    fig_cost(M)
    tab_cnn(M)
    starred = tab_llm()
    write_hardware(M, starred, MO)
    if MO is not None:
        tab_opt(M, MO)
        tab_ratios(MO)
    for name in OLD_FIGURES:
        old = FIGS / f"{name}.pdf"
        if old.exists():
            old.unlink()
            print("removed", old.name)


if __name__ == "__main__":
    main()

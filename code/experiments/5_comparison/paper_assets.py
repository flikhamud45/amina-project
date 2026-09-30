"""Every figure and table body of the report, built from one platform's stored benchmark tables.

    python experiments/5_comparison/paper_assets.py --platform l40s   # the report's numbers
        # -> ../report/figures/*.pdf, ../report/tables/*.tex (from tables_l40s/)
    python experiments/5_comparison/paper_assets.py --platform <p> --compare <q>
        # headline numbers from tables_<p>/; Figure 5 also draws <q> (on the zkLLM models both measured)
    python experiments/5_comparison/paper_assets.py --platform l40s --check   # only list missing cells

Nothing here runs a model; every number is read from the stored benchmark tables,
so the report can be rebuilt without a GPU.  The report's numbers are platform
``l40s`` (``tables_l40s/``); ``rtx2080ti-v2`` rebuilds the report's previous version.
The earliest run (``tables/``, platform ``rtx2080ti``: the default when neither
``--platform`` nor ``$PVI_PLATFORM`` is set) is rebuilt exactly as submitted: no extrapolation markers or hatching, and no rows beyond the
submitted ones.  Any other platform marks extrapolated rows and bars, and its Figure 4 and
text ranges use measured (full-depth) rows only.  Build the committed figures with the
pinned matplotlib (``requirements.txt``): other versions change their sizes slightly.  Figures 1 and 2
are schematics.  Each panel is its own file (sub-captions are set in LaTeX); sizes match
the width they are printed at, so the fonts appear at their real size.
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
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch, Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "artifacts" / "comparison"
TABLES = BASE / "tables"  # the headline platform's tables; set by main() from --platform


def tables_dir(platform: str) -> Path:
    """'' (alias rtx2080ti) is the earlier RTX 2080 Ti run, tables/; otherwise tables_<platform>/."""
    return BASE / ("tables" if platform in ("", "rtx2080ti") else f"tables_{platform}")


def is_frozen(tables: Path | None = None) -> bool:
    """The earlier run's RTX 2080 Ti tables (``tables/``), to be rebuilt exactly as submitted."""
    return (tables or TABLES) == tables_dir("")


def hw_short(name) -> str:
    """'NVIDIA A100-SXM4-80GB' -> 'A100 80GB', 'NVIDIA GeForce RTX 2080 Ti' -> 'RTX 2080 Ti'."""
    s = (name or "?").replace("NVIDIA ", "").replace("GeForce ", "")
    s = re.sub(r"-SXM\d?-|-PCIE-|\bPCIe\b|\bHBM3e?\b|\bNVL\b", " ", s)
    return re.sub(r"\s+", " ", s).strip()

# Hardware is chosen with --platform (tables_<platform>/).  PVI_LLM_VARIANT selects a --tag variant
# inside that platform (e.g. "_nolean"); "" = the plain runs.
DEFAULT = {"rate": "4", "threads": "8", "variant": os.environ.get("PVI_LLM_VARIANT", ""), "lam": 128}
CNN_ORDER = ["mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar", "resnet18_224"]
# the timing parts that make up one proof and one verification (fs_hash is paid by both; prove_encode and
# verify_decode: the compact encoding of the proof, bench.py --wire)
PROVE = ("prove_forward", "prove_fold", "prove_open", "prove_encode", "fs_hash")
# verify_upload: a GPU client's host-to-device copy of the proof (absent for the CPU verifier);
# verify_total: the streaming verifier (bench.py --verifier-impl stream), which times its overlapped
# derive, uploads, products and columns as one phase, recorded instead of those four (OVERLAPPED)
VERIFY = ("verify_decode", "verify_derive", "verify_fold", "verify_products", "verify_columns", "verify_upload",
          "verify_total", "fs_hash")
OVERLAPPED = ("verify_derive", "verify_products", "verify_columns", "verify_upload")


def _one_verifier(metrics, where: str) -> None:
    """The verify phases of one row come from one verifier: never ``verify_total`` with the
    phases it overlaps (they would be counted twice)."""
    if "verify_total" in metrics and any(k in metrics for k in OVERLAPPED):
        raise SystemExit(f"{where}: verify_total (the streaming verifier) together with {OVERLAPPED}")


LITERATURE = ("reported_curated.csv", "analytic.csv")  # not measured by us: always in tables/


def _read(name: str, tables: Path | None = None) -> list[dict]:
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
    """Index over ``measured_summary.csv`` (median and mean per model, cell, metric)."""

    def __init__(self, tables: Path | None = None) -> None:
        self.idx = {}
        for r in _read("measured_summary.csv", tables):
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

    def attack_rate(self, model, cell, attack):
        """(fraction rejected, number of attempts) for one attack type."""
        vals = [(_f(r["mean"]), _f(r["n"])) for k, r in self.idx.items()
                if k[1] == model and k[2] == cell and k[3] == "rejected" and k[5] == attack]
        n = sum(v[1] for v in vals)
        return (sum(v[0] * v[1] for v in vals) / n, int(n)) if n else (None, 0)


def llm_rows(mode="C", lam=DEFAULT["lam"], threads=DEFAULT["threads"], variant=DEFAULT["variant"],
             tables: Path | None = None) -> dict:
    """``{(model, seq): {metric: (value, provenance, params, prover_hw)}}`` from ``llm_full_model.csv``.
    Where a platform has both a full build and 1-2 block builds, the measured row is used."""
    out: dict = defaultdict(dict)
    for r in _read("llm_full_model.csv", tables):
        if (r["mode"], r["challenges"]) != (mode, "int") or _f(r["lam"]) != lam or r.get("batch"):
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


def _llm_bytes(d: dict):
    """Proof bytes, with the Merkle term as a multiproof (native or expected; see aggregate.py)."""
    return d["bytes_total_multiproof"][0] if "bytes_total_multiproof" in d else d.get("bytes_total", (None,))[0]


def _llm_cost(d: dict):
    """(prove seconds, verify seconds, proof bytes, parameters) of one LLM row."""
    _one_verifier(d, "an LLM row")
    prove = sum(d[k][0] for k in PROVE if k in d)
    verify = sum(d[k][0] for k in VERIFY if k in d)
    return prove, verify, _llm_bytes(d), d["prove_forward"][2]


def _measured(d: dict) -> bool:
    """Every metric of one LLM row comes from a full-depth build (none extrapolated)."""
    return bool(d) and all(v[1] == "measured" for v in d.values())


# (published system, published model, our model, our prompt length or None for CNNs, our mode): Table 4
MATCHES = [
    ("zkCNN", "LeNet-5 MNIST", "lenet5", None, "C"),
    ("Bionetta", "LeNet-5", "lenet5", None, "C"),
    ("zkCNN", "VGG-16 CIFAR-10", "vgg16", None, "C"),
    ("ZKML", "VGG-16 CIFAR-10", "vgg16", None, "C"),
    ("zkGPT", "GPT-2", "gpt2", 64, "C"),
    ("DeepProve", "GPT-2", "gpt2", 64, "C"),
    ("DeepProve", "GPT-2", "gpt2", 512, "C"),
    ("zkLLM", "OPT-125M", "opt-125m", 2048, "C"),
    ("zkLLM", "OPT-6.7B", "opt-6.7b", 2048, "C"),
    ("zkLLM", "Llama-2-7B", "llama2-7b", 2048, "C"),
    ("ZKTorch", "Llama-2-7B (1 token)", "llama2-7b", 64, "C"),
    ("Maverick (verif.-only, 1 thr.)", "Qwen3-4B", "qwen3-4b", 8, "Kpre"),
]
# zkLLM's 2,048-token list (all eight are in reported_curated.csv).  Figure 5 draws those that every
# drawn platform measured; Table 4 adds the ones beyond MATCHES that the headline platform has
# (OPT-1.3B stays out of Table 4, as submitted: it is in Figure 5 and Table 3).
ZKLLM = [("opt-125m", "OPT-125M"), ("opt-350m", "OPT-350M"), ("opt-1.3b", "OPT-1.3B"), ("opt-2.7b", "OPT-2.7B"),
         ("opt-6.7b", "OPT-6.7B"), ("opt-13b", "OPT-13B"), ("llama2-7b", "Llama-2-7B"), ("llama2-13b", "Llama-2-13B")]


def zkllm_models(tables_list) -> list[tuple[str, str]]:
    """(our model, zkLLM's name) of zkLLM's 2,048-token models that every tables directory has."""
    zk = {r["model"] for r in _read("reported_curated.csv") if r["system"] == "zkLLM" and r["seq"] == "2048"}
    rows = [llm_rows("C", tables=t) for t in tables_list]
    return [(m, n) for m, n in ZKLLM if n in zk and all("prove_forward" in r.get((m, 2048), {}) for r in rows)]


def matches(c_llm: dict | None = None) -> list[tuple]:
    """Table 4's pairs for the headline rows ``c_llm``: MATCHES, with zkLLM's other 2,048-token models
    where the headline platform measured them (on the frozen platform without OPT-1.3B, as submitted),
    and ZKTorch at its own 1 token where that was run (otherwise at 64 tokens, as submitted)."""
    c_llm = llm_rows("C") if c_llm is None else c_llm
    listed = {ours for system, _, ours, _, _ in MATCHES if system == "zkLLM"}
    out, zk_done = [], False
    for system, pub, ours, seq, mode in MATCHES:
        if system == "zkLLM":
            if not zk_done:   # every zkLLM row here, in ZKLLM's order
                zk_done = True
                out += [("zkLLM", n, m, 2048, "C") for m, n in ZKLLM if m in listed or
                        ((m != "opt-1.3b" or not is_frozen()) and "prove_forward" in c_llm.get((m, 2048), {}))]
            continue
        if system == "ZKTorch" and "prove_forward" in c_llm.get((ours, 1), {}):
            seq = 1
        out.append((system, pub, ours, seq, mode))
    return out

REPORT = ROOT.parent / "report"
FIGS = REPORT / "figures"
TABS = REPORT / "tables"
LAM = DEFAULT["lam"]

OURS, OURS_K, THEM, ANCH = "#c0392b", "#2471a3", "0.5", "#e67e22"
plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"],
                     "mathtext.fontset": "stix", "font.size": 7, "axes.labelsize": 7, "legend.fontsize": 6.2,
                     "xtick.labelsize": 6.3, "ytick.labelsize": 6.3, "axes.spines.top": False,
                     "axes.spines.right": False, "axes.linewidth": 0.6, "savefig.bbox": "tight",
                     "savefig.pad_inches": 0.02, "figure.dpi": 200})

SHORT = {"mlp_mnist": "MLP", "lenet5": "LeNet-5", "vgg11": "VGG-11", "vgg16": "VGG-16",
         "resnet18_cifar": "ResNet-18", "resnet18_224": "ResNet-18†"}
DATA = {"mlp_mnist": "MNIST", "lenet5": "MNIST", "vgg11": "CIFAR-10", "vgg16": "CIFAR-10",
        "resnet18_cifar": "CIFAR-10", "resnet18_224": "224px"}
LLM_NAME = {"gpt2": "GPT-2", "opt-125m": "OPT-125M", "opt-350m": "OPT-350M", "opt-1.3b": "OPT-1.3B",
            "opt-2.7b": "OPT-2.7B", "opt-13b": "OPT-13B", "opt-6.7b": "OPT-6.7B",
            "llama2-7b": "Llama-2-7B", "llama2-13b": "Llama-2-13B", "qwen3-4b": "Qwen3-4B"}

# published systems grouped by proof family, for the cost figure
FAMILY = {"zkCNN": "gkr", "zkLLM": "gkr", "zkGPT": "gkr", "DeepProve": "gkr", "zkPyTorch": "gkr",
          "Jolt": "gkr", "ZKML": "plonk", "EZKL": "plonk", "ZKTorch": "plonk", "Bionetta": "plonk",
          "vCNN": "plonk", "SLP": "plonk", "Mystique": "other", "LAMP": "other", "Maverick": "known"}
FAM_STYLE = {"gkr": ("^", "sum-check zkSNARKs"), "plonk": ("D", "Plonk/Groth16 zkSNARKs"),
             "other": ("p", "other ZK proofs"), "known": ("*", "Maverick (knows weights)")}
# a few landmark systems are named on the chart; one name per panel at most once
LABELS = {
    "prover": {("zkCNN", "VGG-16 CIFAR-10"): "zkCNN", ("zkLLM", "Llama-2-7B"): "zkLLM",
               ("DeepProve", "GPT-2", "64"): "DeepProve"},
    "verifier": {("zkCNN", "VGG-16 CIFAR-10"): "zkCNN", ("zkLLM", "Llama-2-7B"): "zkLLM",
                 ("ZKTorch", "Llama-2-7B (1 token)"): "ZKTorch"},
    "proof": {("zkCNN", "VGG-16 CIFAR-10"): "zkCNN", ("zkLLM", "Llama-2-7B"): "zkLLM",
              ("DeepProve", "GPT-2", "64"): "DeepProve"},
}


def family(system: str):
    for k, v in FAMILY.items():
        if system.startswith(k):
            return v
    return None


def save(fig, name, tight=True):
    FIGS.mkdir(parents=True, exist_ok=True)
    box = "tight" if tight else None   # fixed canvases keep side-by-side panels aligned
    fig.savefig(FIGS / f"{name}.pdf", metadata={"CreationDate": None}, bbox_inches=box)  # reproducible bytes
    plt.close(fig)
    print("figure", name)


# ----------------------------------------------------------------------------------- figures 1-2
WIDTHS = [5, 7, 7, 3]
POS = {(l, i): (l, (i - (w - 1) / 2) * 0.55) for l, w in enumerate(WIDTHS) for i in range(w)}
PATH = [(3, 1), (2, 4), (1, 2), (0, 3)]
TAMPERED = (1, 5)


def _network(ax, colour_node):
    for l in range(3):
        for i in range(WIDTHS[l]):
            for j in range(WIDTHS[l + 1]):
                (x0, y0), (x1, y1) = POS[(l, i)], POS[(l + 1, j)]
                ax.plot([x0, x1], [y0, y1], color="0.87", lw=0.35, zorder=1)
    for node, (x, y) in POS.items():
        fc, ec = colour_node(node)
        ax.scatter(x, y, s=30, facecolor=fc, edgecolor=ec, lw=0.7, zorder=3)
    ax.text(0, -1.45, "input", ha="center", va="top", fontsize=6, color="0.45")
    ax.text(3, -0.9, "output", ha="center", va="top", fontsize=6, color="0.45")


def _path(ax, filled):
    for a, b in zip(PATH[:-1], PATH[1:]):
        (x0, y0), (x1, y1) = POS[a], POS[b]
        ax.plot([x0, x1], [y0, y1], color=OURS_K, lw=1.6, zorder=2)
    for n in PATH:
        ax.scatter(*POS[n], s=30, facecolor=OURS_K if filled else "white", edgecolor=OURS_K, lw=1.0, zorder=4)


def _finish(ax):
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-2.02, 2.75)
    ax.axis("off")
    ax.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)


def fig_overview():
    size = (2.2, 1.45)
    fig, ax = plt.subplots(figsize=size)
    _network(ax, lambda n: ("white", "0.35"))
    _path(ax, filled=True)
    _finish(ax)
    save(fig, "overview_path", tight=False)

    fig, ax = plt.subplots(figsize=size)
    _network(ax, lambda n: (OURS, OURS) if n == TAMPERED else
             (("#f5d0cb", "0.35") if n[0] > TAMPERED[0] else ("white", "0.35")))
    _path(ax, filled=False)
    ax.annotate("wrong value", xy=POS[TAMPERED], xytext=(0.2, 2.05), fontsize=6.5, color=OURS,
                arrowprops=dict(arrowstyle="->", color=OURS, lw=0.6))
    ax.text(2.55, 2.05, "recomputed\nhonestly", fontsize=6, color="0.35", ha="center", va="center")
    _finish(ax)
    save(fig, "overview_attack", tight=False)

    fig, ax = plt.subplots(figsize=size)
    _network(ax, lambda n: ("white", "0.35"))
    for l in range(1, 4):
        h = (WIDTHS[l] - 1) * 0.55 / 2 + 0.28
        ax.add_patch(FancyBboxPatch((l - 0.2, -h), 0.4, 2 * h, boxstyle="round,pad=0.02",
                                    fc="none", ec=OURS, lw=1.0, zorder=2))
    ax.text(1.5, 2.45, "$\\chi^\\top Z = (\\chi^\\top W)\\,X$ per layer", ha="center", va="center",
            fontsize=6.8, color=OURS)
    _finish(ax)
    save(fig, "overview_defence", tight=False)


def fig_protocol():
    """Message flow of one query (Sec. 3.3): what each side sends and checks."""
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    xp, xv = 0.1, 0.56                       # prover and verifier lifelines
    ax.text(xp, 1.0, "prover (cloud)", ha="center", va="bottom", fontsize=7, weight="bold")
    ax.text(xv, 1.0, "verifier (client)", ha="center", va="bottom", fontsize=7, weight="bold")
    ax.plot([xp, xp], [0.02, 0.98], color="0.6", lw=0.8)
    ax.plot([xv, xv], [0.02, 0.98], color="0.6", lw=0.8)
    steps = [  # (y, direction, message, verifier's note)
        (0.92, "pv", "query $x$", None),
        (0.80, "vp", "output $y$, claims $Z_\\ell$ (every layer)",
         "recompute all non-weight ops;\nrange-check claims; draw $\\chi$"),
        (0.60, "pv", "random $\\chi$", None),
        (0.48, "vp", "$u_\\ell = \\chi^\\top A_\\ell$",
         "check $\\chi^\\top Z_\\ell = u_\\ell\\,[X_\\ell; 1]$;\npick column positions $j$"),
        (0.28, "pv", "positions $j$", None),
        (0.16, "vp", "columns $\\hat A_{\\ell,j}$ + Merkle proof",
         "check Merkle paths and\n$\\chi^\\top \\hat A_{\\ell,j} = \\mathrm{Enc}(u_\\ell)_j$;\naccept if all checks pass"),
    ]
    for y, d, msg, note in steps:
        x0, x1 = (xv, xp) if d == "pv" else (xp, xv)
        ax.annotate("", xy=(x1, y), xytext=(x0, y),
                    arrowprops=dict(arrowstyle="-|>", color=OURS if d == "vp" else OURS_K, lw=0.9,
                                    shrinkA=0, shrinkB=0))
        ax.text((xp + xv) / 2, y + 0.018, msg, ha="center", va="bottom", fontsize=6.3)
        if note:
            ax.text(xv + 0.03, y - 0.03, note, ha="left", va="top", fontsize=5.8, color="0.3")
    ax.set_xlim(0, 1)
    ax.set_ylim(-0.1, 1.06)
    ax.axis("off")
    save(fig, "protocol")


# ----------------------------------------------------------------------------------- figure 3
def fig_security(M):
    models = [m for m in CNN_ORDER if M.get(m, "sampling", "p_detect_penultimate") is not None]

    fig, a = plt.subplots(figsize=(3.4, 1.85))
    xs = list(range(len(models)))
    one = [M.get(m, "sampling", "p_detect_penultimate") for m in models]
    worst = [M.get(m, "sampling", "p_detect_min_node") for m in models]
    ours = [M.attack_rate(m, "tamper_C_int_lam40", "penultimate_neuron")[0] for m in models]
    w = 0.27
    a.bar([x - w for x in xs], one, w, color=ANCH, label="Anchuri et al., attacked neuron")
    a.bar(xs, worst, w, color="#f5cba7", label="Anchuri et al., least-visited neuron")
    a.bar([x + w for x in xs], ours, w, color=OURS, label="ours (all rejected)")
    a.set_yscale("log")
    a.set_ylim(1e-8, 3e3)
    a.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    a.set_xticks(xs)
    a.set_xticklabels([f"{SHORT[m]}\n{DATA[m]}" for m in models], fontsize=5.8)
    a.set_ylabel("detection per query")
    a.legend(frameon=False, loc="upper left", ncol=2, fontsize=5.8, handlelength=1.0, columnspacing=0.8,
             bbox_to_anchor=(0, 1.04))
    a.grid(True, axis="y", lw=0.3, alpha=0.4)
    save(fig, "security_detection")

    fig, b = plt.subplots(figsize=(3.4, 1.85))
    curve = {}
    for key, r in M.idx.items():
        if key[3] == "paths_bytes_shared_k" and key[2] == "sampling":
            curve.setdefault(key[1], []).append(r)
    for i, m in enumerate(models):
        c = f"C{i}"
        rows = sorted(curve.get(m, []), key=lambda r: _f(r["median"]))
        bx, by = [_f(r["median"]) for r in rows], [_f(r.get("bits")) for r in rows]
        if bx and None not in by:
            b.plot(bx, by, ":", color=c, lw=1)
            cap = M.get(m, "sampling", "open_all_bytes")
            b.plot([cap, cap], [max(by), 200], ":", color=c, lw=0.6)
        pts = [(M.get(m, f"defence_C_int_lam{lam}_rate4", "bytes_total"),
                M.get(m, f"defence_C_int_lam{lam}_rate4", "soundness_bits")) for lam in (40, 80, 128)]
        pts = [p for p in pts if None not in p]
        b.plot(*zip(*pts), "-o", color=c, ms=2.5, lw=1, label=f"{SHORT[m]} ({DATA[m]})")
    b.axhline(40, color="0.6", lw=0.5, ls="--")
    b.text(2.3e3, 48, "$2^{-40}$", fontsize=6, color="0.4")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_ylim(1e-3, 300)
    b.set_xlim(2e3, 3e7)
    b.set_xlabel("bytes sent per query")
    b.set_ylabel("security (bits)")
    handles, labels = b.get_legend_handles_labels()
    handles += [Line2D([], [], color="0.3", ls="-", marker="o", ms=2.5, lw=1),
                Line2D([], [], color="0.3", ls=":", lw=1)]
    labels += ["ours ($\\lambda$ = 40, 80, 128)", "Anchuri et al., $s$ paths"]
    b.legend(handles, labels, frameon=False, fontsize=5.8, loc="upper center", bbox_to_anchor=(0.5, -0.3),
             ncol=3, handlelength=1.6, columnspacing=1.0)
    b.grid(True, lw=0.3, alpha=0.4)
    save(fig, "security_bits")


# ----------------------------------------------------------------------------------- figure 4
def _ours_points(M):
    ours_c, ours_k = [], []
    for m in CNN_ORDER:
        n = M.get(m, "facts", "n_params")
        for cell, out in ((f"defence_C_int_lam{LAM}_rate4", ours_c), (f"defence_Kpre_int_lam{LAM}_rate4", ours_k)):
            if M.get(m, cell, "prove_forward") is not None:
                out.append((n, M.total(m, cell, PROVE), M.total(m, cell, VERIFY), M.get(m, cell, "bytes_total"), "cnn"))
    frozen = is_frozen()   # the submitted figure: its extrapolated rows are drawn as they were
    for mode, out, seqs in (("C", ours_c, (64, 2048)), ("Kpre", ours_k, (64,))):
        for (model, seq), d in sorted(llm_rows(mode).items()):
            if "prove_forward" in d and seq in seqs and (frozen or _measured(d)):
                p, v, by, n = _llm_cost(d)
                out.append((n, p, v, by, f"T{seq}"))
    return ours_c, ours_k


def fig_cost(M):
    rep = [r for r in _read("reported_curated.csv") if _f(r["params"])]
    ours_c, ours_k = _ours_points(M)
    panels = (("prover_s", 0, "prover", "prover time (s)"), ("verifier_s", 1, "verifier", "verifier time (s)"),
              ("proof_bytes", 2, "proof", "proof size (bytes)"))
    for key, col, name, ylabel in panels:
        fig, ax = plt.subplots(figsize=(2.25, 1.75))
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
            if lab and lab not in named:
                named.add(lab)
                ax.annotate(lab, (x, y), xytext=(3, 0), textcoords="offset points", fontsize=5.5, color="0.3",
                            va="center")
        for pts, colour in ((ours_c, OURS), (ours_k, OURS_K)):
            for n, *vals, kind in pts:
                if vals[col] is None:
                    continue
                ax.scatter(n, vals[col], marker="o" if kind == "cnn" else "s", s=16,
                           c="none" if kind == "T2048" else colour, edgecolors=colour, linewidths=0.8, zorder=4)
        if name == "proof":
            ax.plot([1e4, 3e10], [1e4, 3e10], "k--", lw=0.5, zorder=1)
            ax.text(1.2e5, 3e6, "int8 model", fontsize=5.5, color="0.3", rotation=33)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("model parameters")
        ax.set_ylabel(ylabel)
        ax.grid(True, which="major", lw=0.3, alpha=0.4)
        save(fig, f"cost_{name}")

    # one legend for the three panels, as its own strip
    handles = [Line2D([], [], marker="o", ls="", color=OURS, ms=4, label="ours, committed: CNN"),
               Line2D([], [], marker="s", ls="", color=OURS, ms=4, label="ours, committed: LLM, 64 tokens"),
               Line2D([], [], marker="s", ls="", mfc="none", color=OURS, ms=4, label="ours, committed: LLM, 2048 tokens"),
               Line2D([], [], marker="o", ls="", color=OURS_K, ms=4, label="ours, known weights: CNN"),
               Line2D([], [], marker="s", ls="", color=OURS_K, ms=4, label="ours, known weights: LLM, 64 tokens"),
               Line2D([], [], marker="X", ls="", color=ANCH, ms=5, label="Anchuri et al., one path")]
    for fam in ("gkr", "plonk", "other", "known"):
        mk, lab = FAM_STYLE[fam]
        handles.append(Line2D([], [], marker=mk, ls="", mfc="none", color=THEM, ms=4 if fam != "known" else 6,
                              label=lab))
    fig = plt.figure(figsize=(7.0, 0.42))
    fig.legend(handles=handles, loc="center", ncol=5, frameon=False, fontsize=6.2, handletextpad=0.3,
               columnspacing=1.2)
    save(fig, "cost_legend")


# ----------------------------------------------------------------------------------- figure 5
def fig_llm(compare=()):
    """Prover time at 2,048 tokens: ours (headline GPU, and any --compare GPUs) against zkLLM
    (A100 40GB, as reported) on zkLLM's models that every drawn platform measured.  The ratios
    are against the headline GPU.  The earlier run's platform alone gives the submitted figure."""
    zk = {r["model"]: r for r in _read("reported_curated.csv") if r["system"] == "zkLLM" and r["seq"] == "2048"}
    tables_list = [tables_dir(p) for p in compare] + [TABLES]
    pairs = zkllm_models(tables_list)
    if not pairs:
        raise SystemExit("Figure 5: no zkLLM model at 2,048 tokens that every drawn platform has")
    plain = is_frozen() and not compare   # exactly the submitted figure
    series = []
    for p, tables in zip(list(compare) + [None], tables_list):
        rows = llm_rows("C", tables=tables)
        label = hw_short(rows[(pairs[0][0], 2048)]["prove_forward"][3])
        if p is not None and is_frozen(tables) and not is_frozen():
            label += ", as submitted"
            print(f"WARNING: Figure 5 --compare {p or 'rtx2080ti'} draws the earlier run (raw/, as submitted): its "
                  f"C-mode times include the committed weights' re-upload (fixed since) and predate --lean, "
                  f"so the difference is not all hardware (use a re-run of that GPU for a hardware ratio)")
        series.append(([_llm_cost(rows[(m, 2048)])[0] for m, _ in pairs], label,
                       [rows[(m, 2048)]["prove_forward"][1] != "measured" for m, _ in pairs]))
    fig, b = plt.subplots(figsize=(3.3, 1.3))
    xs = range(len(pairs))
    them = [_f(zk[name]["prover_s"]) for _, name in pairs]
    w = 0.8 / (len(series) + 1)
    shades = [plt.cm.Reds(0.3 + 0.4 * i / max(1, len(compare) - 1)) for i in range(len(compare))]
    # hatch extrapolated bars, but only where measured bars are drawn too (never in the submitted figure)
    hatch = not plain and (len(series) > 1 or not all(series[-1][2]))
    hatched = False
    for i, (ours, label, extra) in enumerate(series):
        bars = b.bar([x - 0.4 + w * (i + 0.5) for x in xs], ours, w * 0.95, color=(shades + [OURS])[i],
                     label=f"ours ({label})")
        for bar, e in zip(bars, extra):
            if e and hatch:
                bar.set_hatch("////")  # extrapolated from 1- and 2-block builds
                hatched = True
    zk_hw = "A100" if plain else hw_short(zk[pairs[0][1]]["hardware"])   # the submitted legend's wording
    b.bar([x + 0.4 - w / 2 for x in xs], them, w * 0.95, color=THEM, label=f"zkLLM ({zk_hw})")
    ours = series[-1][0]
    for x, o, t_ in zip(xs, ours, them):
        b.text(x, t_ * 1.3, f"{t_ / o:.0f}$\\times$", ha="center", fontsize=6.3)
    b.set_yscale("log")
    handles, labels = b.get_legend_handles_labels()
    if hatched:
        handles.append(Patch(facecolor="white", edgecolor="0.3", hatch="////", label="extrapolated from 1-2 blocks"))
        labels.append(handles[-1].get_label())
    ncol = len(handles) if len(handles) <= 2 else 2      # one row, or two columns
    top = 10.0 ** (3 + -(-len(handles) // ncol))          # 1e4 for one legend row: room above the bars
    floor = 0.5 if plain else min(0.5, min(min(s[0]) for s in series) / 3)  # a faster GPU needs a lower floor
    b.set_ylim(floor, top)
    b.set_xticks(list(xs))
    if len(pairs) <= 4:
        b.set_xticklabels([LLM_NAME[m] for m, _ in pairs])
    else:   # up to zkLLM's eight models, on two lines: OPT / 125M, Llama-2 / 13B
        b.set_xticklabels(["\n".join(LLM_NAME[m].rsplit("-", 1)) for m, _ in pairs], fontsize=5.8)
    b.set_ylabel("prover time (s)")
    if hatched:
        b.legend(handles, labels, frameon=False, loc="upper left", ncol=ncol, fontsize=6)
    else:
        b.legend(frameon=False, loc="upper left", ncol=ncol, fontsize=6)
    save(fig, "llm_zkllm")


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


def params(n):
    return f"{n / 1e9:.1f}B" if n >= 1e9 else (f"{n / 1e6:.1f}M" if n >= 1e6 else f"{n / 1e3:.0f}K")


def write(name, lines):
    TABS.mkdir(parents=True, exist_ok=True)
    # the closing rule lives in the file: after \input, a \bottomrule in main.tex is a misplaced \noalign
    (TABS / f"{name}.tex").write_text("\n".join(lines + [r"\bottomrule"]) + "\n", encoding="utf-8", newline="\n")
    print("table", name)


def tab_cnn(M):
    lines = []
    for m in CNN_ORDER:
        c, k = f"defence_C_int_lam{LAM}_rate4", f"defence_Kpre_int_lam{LAM}_rate4"
        acc = M.get(m, "facts", "int8_accuracy")
        label = SHORT[m].replace("†", r"$^\dagger$") + f" ({DATA[m]})"
        lines.append(" & ".join([
            label, params(M.get(m, "facts", "n_params")),
            f"{acc * 100:.1f}\\%",
            t(M.total(m, c, PROVE)), t(M.total(m, c, VERIFY)), b(M.get(m, c, "bytes_total")),
            t(M.total(m, k, VERIFY)), b(M.get(m, k, "bytes_total")),
            f"{M.get(m, 'sampling', 'paths_needed_penultimate', lam=40):,.0f}".replace(",", "{,}"),
            b(M.get(m, "sampling", "paths_bytes_shared", lam=40)),
            b(M.get(m, "defence_C_int_lam40_rate4", "bytes_total"))]) + r" \\")
    write("cnn", lines)


def tab_llm():
    c, k = llm_rows("C"), llm_rows("Kpre")
    # the largest model at zkLLM's 2,048 tokens, where a platform measured it (the 2080 Ti cannot hold it)
    pick = TAB_LLM_PICK + [key for key in (("llama2-13b", 2048),) if _measured(c.get(key)) and _measured(k.get(key))]
    lines, starred = [], False
    for model, seq in pick:
        p, v, by, n = _llm_cost(c[(model, seq)])
        _, kv, kb, _ = _llm_cost(k[(model, seq)])
        # the submitted table has no markers (the text says which rows are extrapolated)
        extra = not is_frozen() and not (_measured(c[(model, seq)]) and _measured(k[(model, seq)]))
        starred |= extra
        name = LLM_NAME[model] + (r"$^\ast$" if extra else "")
        lines.append(" & ".join([name, f"{seq:,}".replace(",", "{,}"), t(p), t(v), b(by), t(kv), b(kb)]) + r" \\")
    write("llm", lines)
    return starred


def write_hardware(M, starred):
    """Macros for the text: the headline machine, and the Table 3 note on extrapolated rows."""
    rows = llm_rows("C")
    hw = {v[3] for d in rows.values() for v in d.values()} | {M.prover_hw}
    verifier = {r.get("verifier_hw") for r in _read("llm_full_model.csv")
                if r.get("threads") == DEFAULT["threads"] and r.get("variant") == DEFAULT["variant"]} | {M.verifier_hw}
    if len(hw - {None}) != 1 or len(verifier - {None}) != 1:
        raise SystemExit(f"the headline tables mix machines: {sorted(map(str, hw))} / {sorted(map(str, verifier))}")
    gpu, cpu = next(iter(hw - {None})), next(iter(verifier - {None}))
    cpu_name, _, threads = cpu.partition(" x")
    # a tie before the name's last word ('RTX 2080~Ti'), so the text never breaks a line inside it
    lines = [r"\newcommand{\ProverGPU}{" + re.sub(r" (\S+)$", r"~\1", hw_short(gpu)) + "}",
             r"\newcommand{\VerifierCPU}{" + re.sub(r"\(R\)|\(TM\)|CPU|@.*$", "", cpu_name).split("  ")[0].strip() + "}",
             r"\newcommand{\VerifierThreads}{" + threads.split()[0] + "}",
             r"\newcommand{\LLMNote}{" + (r" ($^\ast$extrapolated from 1- and 2-block builds)" if starred else "") + "}"]
    TABS.mkdir(parents=True, exist_ok=True)
    (TABS / "hardware.tex").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    print("table hardware:", hw_short(gpu), "/", cpu)


def tab_ratios(M):
    curated = _read("reported_curated.csv")
    c_llm = llm_rows("C")
    k1 = llm_rows("Kpre", lam=40, threads="1", variant="_thr1" + DEFAULT["variant"])
    lines = []
    for system, model, ours, seq, mode in matches(c_llm):
        cand = [r for r in curated if (r["system"], r["model"]) == (system, model)]
        same = [r for r in cand if seq and r["seq"] == str(seq)]
        r = same[0] if same else cand[0]
        if seq is None:
            cell = f"defence_{mode}_int_lam{LAM}_rate4"
            us = (M.total(ours, cell, PROVE), M.total(ours, cell, VERIFY), M.get(ours, cell, "bytes_total"))
            what = SHORT[ours]
        else:
            us = _llm_cost((c_llm if mode == "C" else k1)[(ours, seq)])[:3]
            what = f"{LLM_NAME[ours]} ({seq:,})".replace(",", "{,}")
        them = (_f(r["prover_s"]), _f(r["verifier_s"]), _f(r["proof_bytes"]))

        def fac(x, y, good):
            if not x or not y:
                return "--"
            q = x / y
            if 0.99 < q < 1.01:
                return "same"
            v = q if q >= 1 else 1 / q
            s = f"{v:,.0f}".replace(",", "{,}") if v >= 100 else (f"{v:.0f}" if v >= 10 else f"{v:.1f}")
            arrow = r"$\uparrow$" if (q >= 1) == good else r"$\downarrow$"
            return f"{s}$\\times${arrow}"
        name = system.split(" (")[0]
        lines.append(" & ".join([name, what, fac(them[0], us[0], True), fac(them[1], us[1], True),
                                 fac(them[2], us[2], True)]) + r" \\")
    write("ratios", lines)


TAB_LLM_PICK = [("gpt2", 64), ("gpt2", 512), ("opt-125m", 2048), ("opt-1.3b", 2048), ("qwen3-4b", 64),
                ("opt-6.7b", 2048), ("llama2-7b", 64), ("llama2-7b", 2048), ("llama2-13b", 64)]


def missing(M) -> list[str]:
    """Every stored number the tables and figures need, that the headline tables lack."""
    out = []
    for m in CNN_ORDER:
        need = [("facts", "n_params", ""), ("facts", "int8_accuracy", ""), ("sampling", "p_detect_penultimate", ""),
                ("sampling", "p_detect_min_node", ""), ("sampling", "open_all_bytes", ""),
                ("sampling", "paths_needed_penultimate", 40), ("sampling", "paths_bytes_shared", 40),
                (f"defence_Kpre_int_lam{LAM}_rate4", "prove_forward", ""),
                (f"defence_Kpre_int_lam{LAM}_rate4", "bytes_total", "")]
        need += [(f"defence_C_int_lam{l}_rate4", k, "") for l in (40, 80, 128) for k in ("bytes_total", "soundness_bits")]
        need += [(f"defence_C_int_lam{LAM}_rate4", "prove_forward", "")]
        out += [f"cnn/{m}/{c} {k}" for c, k, lam in need if M.get(m, c, k, lam=lam) is None]
        if not M.attack_rate(m, "tamper_C_int_lam40", "penultimate_neuron")[1]:
            out.append(f"cnn/{m}/tamper_C_int_lam40 penultimate_neuron")
    v = DEFAULT["variant"]
    # gpt2 T128/T256: the DeepProve sentence of Sec. 4.4 ('faster than DeepProve up to 256 tokens')
    sel = {("C", LAM, "8", v): set(TAB_LLM_PICK) | {("opt-125m", 2048), ("opt-1.3b", 2048), ("opt-6.7b", 2048),
                                                    ("llama2-7b", 2048), ("gpt2", 64), ("gpt2", 128), ("gpt2", 256),
                                                    ("gpt2", 512), ("llama2-7b", 64)},
           ("Kpre", LAM, "8", v): set(TAB_LLM_PICK), ("Kpre", 40, "1", "_thr1" + v): {("qwen3-4b", 8)}}
    for (mode, lam, thr, var), keys in sel.items():
        rows = llm_rows(mode, lam=lam, threads=thr, variant=var)
        for model, seq in sorted(keys):
            d = rows.get((model, seq), {})
            verified = "verify_products" in d or "verify_total" in d
            if "prove_forward" not in d or not verified or not ("bytes_total" in d or "bytes_total_multiproof" in d):
                out.append(f"llm/{model} {mode}:int lam{lam} T{seq} threads {thr} variant '{var}'")
    return out


def main():
    global TABLES
    ap = argparse.ArgumentParser()
    ap.add_argument("--platform", default=os.environ.get("PVI_PLATFORM", ""),
                    help="headline platform: tables_<platform>/ ('' or rtx2080ti: tables/)")
    ap.add_argument("--compare", nargs="*", default=[], help="platforms also drawn in Figure 5, e.g. rtx2080ti")
    ap.add_argument("--check", action="store_true", help="only list the stored numbers that are missing")
    args = ap.parse_args()
    TABLES = tables_dir(args.platform)
    M = Measured()
    gaps = missing(M)
    if gaps or args.check:
        print(f"{TABLES.name}: {len(gaps)} missing", *gaps, sep="\n  ")
        raise SystemExit(1 if gaps else 0)
    fig_overview()
    fig_protocol()
    fig_security(M)
    fig_cost(M)
    fig_llm(args.compare)
    tab_cnn(M)
    starred = tab_llm()
    tab_ratios(M)
    write_hardware(M, starred)


if __name__ == "__main__":
    main()

"""Figures and table bodies for the report, built from ``code/artifacts/comparison/tables``.

    python report/make_assets.py      # -> report/figures/*.pdf, report/tables/*.tex

Nothing here runs a model; every number is read from the stored benchmark tables
(see code/COMPARISON_ANALYSIS.md), so the report can be rebuilt without the GPU.
"""

from __future__ import annotations

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "code" / "scripts"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from comparison_figures import (CNN_ORDER, DEFAULT, PROVE, VERIFY, Measured, _f, _llm_cost,  # noqa: E402
                                _read, llm_rows)

FIGS = HERE / "figures"
TABS = HERE / "tables"
LAM = DEFAULT["lam"]

OURS, OURS_K, THEM, ANCH = "#c0392b", "#2471a3", "0.45", "#e67e22"
plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
                     "mathtext.fontset": "stix", "font.size": 7.5, "axes.titlesize": 7.5,
                     "axes.labelsize": 7.5, "legend.fontsize": 6.3, "xtick.labelsize": 6.5,
                     "ytick.labelsize": 6.5, "axes.spines.top": False, "axes.spines.right": False,
                     "savefig.bbox": "tight", "savefig.pad_inches": 0.02, "figure.dpi": 200})

SHORT = {"mlp_mnist": "MLP", "lenet5": "LeNet-5", "vgg11": "VGG-11", "vgg16": "VGG-16",
         "resnet18_cifar": "ResNet-18", "resnet18_224": "ResNet-18†"}
DATA = {"mlp_mnist": "MNIST", "lenet5": "MNIST", "vgg11": "CIFAR-10", "vgg16": "CIFAR-10",
        "resnet18_cifar": "CIFAR-10", "resnet18_224": "224px"}
LLM_NAME = {"gpt2": "GPT-2", "opt-125m": "OPT-125M", "opt-1.3b": "OPT-1.3B", "opt-6.7b": "OPT-6.7B",
            "llama2-7b": "Llama-2-7B", "llama2-13b": "Llama-2-13B", "qwen3-4b": "Qwen3-4B"}

# published systems grouped by proof family, for the cost figure
FAMILY = {"zkCNN": "gkr", "zkLLM": "gkr", "zkGPT": "gkr", "DeepProve": "gkr", "zkPyTorch": "gkr",
          "Jolt": "gkr", "ZKML": "plonk", "EZKL": "plonk", "ZKTorch": "plonk", "Bionetta": "plonk",
          "vCNN": "plonk", "Mystique": "vole", "LAMP": "code", "SLP": "plonk", "Maverick": "known"}
FAM_STYLE = {"gkr": ("^", "sum-check / GKR zkSNARKs"), "plonk": ("D", "Plonkish / Groth16 zkSNARKs"),
             "vole": ("p", "VOLE-based ZK (Mystique)"), "code": ("h", "LAMP (one layer)"),
             "known": ("*", "Maverick (knows weights)")}
LABEL_PTS = {("zkLLM", "OPT-125M"): "zkLLM", ("zkCNN", "VGG-16 CIFAR-10"): "zkCNN",
             ("ZKML", "VGG-16 CIFAR-10"): "ZKML", ("DeepProve", "GPT-2"): "DeepProve",
             ("zkGPT", "GPT-2"): "zkGPT", ("ZKTorch", "Llama-2-7B (1 token)"): "ZKTorch",
             ("Maverick (verif.-only, 8 thr.)", "Qwen3-4B"): "Maverick",
             ("EZKL (run by Bionetta)", "MNIST MLP (2M)"): "EZKL"}


SKIP_LABEL = {0: ("zkGPT",), 1: ("zkGPT",), 2: ("zkGPT", "Maverick")}   # labels that would overlap
LEFT_LABEL = {0: ("zkCNN",)}   # per panel: labels drawn to the left of their point


def family(system: str):
    for k, v in FAMILY.items():
        if system.startswith(k):
            return v
    return None


def save(fig, name):
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIGS / f"{name}.pdf")
    fig.savefig(FIGS / f"{name}.png")
    plt.close(fig)
    print("figure", name)


# ----------------------------------------------------------------------------------- figures
def fig_overview():
    """Schematic: (a) one random path, (b) the single-neuron attack, (c) the whole-layer check."""
    widths = [5, 7, 7, 3]
    xs = [0, 1, 2, 3]
    pos = {(l, i): (xs[l], (i - (w - 1) / 2) * 0.55) for l, w in enumerate(widths) for i in range(w)}
    path = [(3, 1), (2, 4), (1, 2), (0, 3)]
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 1.9))
    titles = ["(a) Anchuri et al.: check one random path",
              "(b) Our attack: change one neuron",
              "(c) Our defence: check every layer at once"]
    for p, ax in enumerate(axes):
        for l in range(3):
            for i in range(widths[l]):
                for j in range(widths[l + 1]):
                    (x0, y0), (x1, y1) = pos[(l, i)], pos[(l + 1, j)]
                    ax.plot([x0, x1], [y0, y1], color="0.85", lw=0.35, zorder=1)
        tampered = (1, 5)
        for (l, i), (x, y) in pos.items():
            fc, ec = "white", "0.35"
            if p == 1 and (l, i) == tampered:
                fc, ec = OURS, OURS
            elif p == 1 and l > tampered[0]:
                fc = "#f5d0cb"
            ax.scatter(x, y, s=34, facecolor=fc, edgecolor=ec, lw=0.7, zorder=3)
        if p in (0, 1):
            for (a, b) in zip(path[:-1], path[1:]):
                (x0, y0), (x1, y1) = pos[a], pos[b]
                ax.plot([x0, x1], [y0, y1], color=OURS_K, lw=1.6, zorder=2)
            for n in path:
                ax.scatter(*pos[n], s=34, facecolor=OURS_K if p == 0 else "white", edgecolor=OURS_K, lw=1.0,
                           zorder=4)
        if p == 0:
            ax.text(1.5, -2.35, "opens $\\approx$1 row per layer\nmisses a node w.p. $1-1/N$",
                    ha="center", va="top", fontsize=6.3)
        if p == 1:
            ax.annotate("wrong value", xy=pos[tampered], xytext=(0.45, 2.3), fontsize=6.3, color=OURS,
                        arrowprops=dict(arrowstyle="->", color=OURS, lw=0.6))
            ax.text(3.0, -0.85, "above it:\nrecomputed\nhonestly", fontsize=5.5, color="0.3", ha="center", va="top")
            ax.text(1.5, -2.35, "the path misses it: accepted\nwith probability $1-1/N$ (99.8%)",
                    ha="center", va="top", fontsize=6.3)
        if p == 2:
            for l in range(1, 4):
                h = (widths[l] - 1) * 0.55 / 2 + 0.3
                ax.add_patch(FancyBboxPatch((xs[l] - 0.2, -h), 0.4, 2 * h, boxstyle="round,pad=0.02",
                                            fc="none", ec=OURS, lw=1.0, zorder=2))
            ax.text(1.5, 2.35, "$\\chi^\\top Z \\overset{?}{=} (\\chi^\\top W)\\,X$ for every layer",
                    ha="center", fontsize=6.6, color=OURS)
            ax.text(1.5, -2.35, "any wrong value is caught\nwith probability $1-2^{-\\lambda}$",
                    ha="center", va="top", fontsize=6.3)
        ax.set_xlim(-0.4, 3.4)
        ax.set_ylim(-3.1, 2.75)
        ax.set_title(titles[p], fontsize=7, pad=2)
        ax.axis("off")
    fig.subplots_adjust(wspace=0.05)
    save(fig, "overview")


def fig_security(M):
    """(a) detection per query of the single-neuron attack; (b) security bits vs bytes sent."""
    models = [m for m in CNN_ORDER if M.get(m, "sampling", "p_detect_penultimate") is not None]
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.0), gridspec_kw={"width_ratios": [1.2, 1]})
    xs = range(len(models))
    one = [M.get(m, "sampling", "p_detect_penultimate") for m in models]
    worst = [M.get(m, "sampling", "p_detect_min_node") for m in models]
    ours = [M.attack_rate(m, "tamper_C_int_lam40", "penultimate_neuron")[0] for m in models]
    w = 0.27
    a.bar([x - w for x in xs], one, w, color=ANCH, label="Anchuri et al., 1 path: attacked neuron")
    a.bar(list(xs), worst, w, color="#f5cba7", label="Anchuri et al., 1 path: least-visited")
    a.bar([x + w for x in xs], ours, w, color=OURS, label="ours (measured)")
    for x, o, m in zip(xs, ours, models):
        _, n = M.attack_rate(m, "tamper_C_int_lam40", "penultimate_neuron")
        a.text(x + w, 1.5, f"{round(o * n)}/{n}", ha="center", fontsize=4.4, color=OURS, va="bottom")
    a.set_yscale("log")
    a.set_ylim(1e-8, 5e4)
    a.set_yticks([1e-8, 1e-6, 1e-4, 1e-2, 1])
    a.set_xticks(list(xs))
    a.set_xticklabels([f"{SHORT[m]}\n{DATA[m]}" for m in models], fontsize=5.8)
    a.set_ylabel("P[attack caught] per query")
    a.legend(frameon=False, loc="upper left", fontsize=5.1, handlelength=1.0, ncol=2, columnspacing=0.6,
             handletextpad=0.3, bbox_to_anchor=(-0.02, 1.03))
    a.grid(True, axis="y", lw=0.3, alpha=0.4)
    a.set_title("(a) Catching the single-neuron attack", fontsize=7)

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
        pts = [(M.get(m, f"defence_C_int_lam{l}_rate4", "bytes_total"),
                M.get(m, f"defence_C_int_lam{l}_rate4", "soundness_bits")) for l in (40, 80, 128)]
        pts = [p for p in pts if None not in p]
        b.plot(*zip(*pts), "-o", color=c, ms=2.5, lw=1, label=f"{SHORT[m]} ({DATA[m]})")
    b.axhline(40, color="0.6", lw=0.5, ls="--")
    b.text(2.2e3, 46, "$2^{-40}$", fontsize=5.8, color="0.4")
    b.set_xscale("log")
    b.set_yscale("log")
    b.set_ylim(1e-3, 300)
    b.set_xlim(2e3, 3e7)
    b.set_xlabel("bytes sent per query")
    b.set_ylabel("security against the attack (bits)")
    b.set_title("(b) Security vs. bytes: ours (solid), Anchuri et al. with $k$ paths (dotted)", fontsize=7)
    b.legend(frameon=False, fontsize=5.3, loc="center left", bbox_to_anchor=(1.0, 0.5), ncol=1, handlelength=1.4)
    b.grid(True, lw=0.3, alpha=0.4)
    fig.tight_layout(w_pad=1.2)
    save(fig, "security")


def fig_cost(M):
    """Prover time, verifier time and proof size vs model size: ours vs published systems."""
    rep = [r for r in _read("reported_curated.csv") if _f(r["params"])]
    ours_c, ours_k = [], []
    for m in CNN_ORDER:
        n = M.get(m, "facts", "n_params")
        for cell, out in ((f"defence_C_int_lam{LAM}_rate4", ours_c), (f"defence_Kpre_int_lam{LAM}_rate4", ours_k)):
            if M.get(m, cell, "prove_forward") is not None:
                out.append((n, M.total(m, cell, PROVE), M.total(m, cell, VERIFY), M.get(m, cell, "bytes_total"), "cnn"))
    for (model, seq), d in sorted(llm_rows("C").items()):
        if "prove_forward" in d and seq in (64, 2048):
            p, v, b, n = _llm_cost(d)
            ours_c.append((n, p, v, b, f"T{seq}"))
    for (model, seq), d in sorted(llm_rows("Kpre").items()):
        if "prove_forward" in d and seq == 64:
            p, v, b, n = _llm_cost(d)
            ours_k.append((n, p, v, b, f"T{seq}"))
    fig, axes = plt.subplots(1, 3, figsize=(7.0, 2.25))
    panels = (("prover_s", 0, "(a) prover time (s)"), ("verifier_s", 1, "(b) verifier time (s)"),
              ("proof_bytes", 2, "(c) proof size (bytes)"))
    for ax, (key, col, title) in zip(axes, panels):
        for r in rep:
            y = _f(r[key])
            if y is None:
                continue
            x = _f(r["params"])
            if r["system"].startswith("Anchuri"):
                ax.scatter(x, y, marker="X", s=26, c=ANCH, zorder=3)
                continue
            fam = family(r["system"])
            if fam is None:
                continue
            ax.scatter(x, y, marker=FAM_STYLE[fam][0], s=12 if fam != "known" else 22, c="none",
                       edgecolors=THEM, linewidths=0.6, zorder=2)
            lab = LABEL_PTS.get((r["system"], r["model"]))
            if lab in SKIP_LABEL.get(col, ()):
                lab = None
            if lab and (lab != "DeepProve" or r["seq"] == "64"):
                left = lab in LEFT_LABEL.get(col, ())
                ax.annotate(lab, (x, y), xytext=(-3 if left else 3, 1), textcoords="offset points", fontsize=4.9,
                            color="0.3", ha="right" if left else "left")
        for pts, colour in ((ours_c, OURS), (ours_k, OURS_K)):
            for n, *vals, kind in pts:
                y = vals[col]
                if y is None:
                    continue
                ax.scatter(n, y, marker="o" if kind == "cnn" else "s", s=15,
                           c="none" if kind == "T2048" else colour, edgecolors=colour, linewidths=0.8, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("model parameters")
        ax.set_title(title, fontsize=7)
        ax.grid(True, which="major", lw=0.3, alpha=0.4)
    axes[2].plot([1e4, 3e10], [1e4, 3e10], "k--", lw=0.5, zorder=1)
    axes[2].text(3e5, 2.2e6, "size of the\nint8 model", fontsize=5, rotation=0, color="0.25")
    handles = [Line2D([], [], marker="o", ls="", color=OURS, ms=4, label="ours, committed weights (CNN)"),
               Line2D([], [], marker="s", ls="", color=OURS, ms=4, label="ours, committed (LLM, 64 tokens)"),
               Line2D([], [], marker="s", ls="", mfc="none", color=OURS, ms=4, label="ours, committed (LLM, 2048 tokens)"),
               Line2D([], [], marker="o", ls="", color=OURS_K, ms=4, label="ours, verifier knows weights (CNN)"),
               Line2D([], [], marker="s", ls="", color=OURS_K, ms=4, label="ours, knows weights (LLM, 64 tokens)"),
               Line2D([], [], marker="X", ls="", color=ANCH, ms=5, label="Anchuri et al., one path")]
    for fam in ("gkr", "plonk", "vole", "code", "known"):
        mk, lab = FAM_STYLE[fam]
        handles.append(Line2D([], [], marker=mk, ls="", mfc="none", color=THEM, ms=4 if fam != "known" else 6,
                              label=lab))
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.2),
               fontsize=5.9, handletextpad=0.3, columnspacing=1.0)
    fig.tight_layout(w_pad=0.8)
    save(fig, "cost")


def fig_llm(M):
    """GPT-2 cost against prompt length (ours vs DeepProve) and large models at 2048 tokens (ours vs zkLLM)."""
    rows = llm_rows("C")
    seqs = sorted(s for (m, s), d in rows.items() if m == "gpt2" and "prove_forward" in d)
    dp = sorted((r for r in _read("reported_curated.csv") if r["system"] == "DeepProve"), key=lambda r: _f(r["seq"]))
    fig, (a, b) = plt.subplots(1, 2, figsize=(3.4, 1.8), gridspec_kw={"width_ratios": [1, 1.1]})
    cost = [_llm_cost(rows[("gpt2", s)]) for s in seqs]
    a.plot(seqs, [c[0] for c in cost], "-o", color=OURS, ms=2.5, lw=1, label="ours: prove")
    a.plot(seqs, [c[1] for c in cost], "--o", color=OURS, ms=2.5, lw=1, mfc="white", label="ours: verify")
    a.plot([_f(r["seq"]) for r in dp], [_f(r["prover_s"]) for r in dp], "-^", color=THEM, ms=2.5, lw=1,
           label="DeepProve: prove")
    a.plot([_f(r["seq"]) for r in dp], [_f(r["verifier_s"]) for r in dp], "--^", color=THEM, ms=2.5, lw=1,
           mfc="white", label="DeepProve: verify")
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xticks(seqs)
    a.set_xticklabels([str(s) for s in seqs])
    a.set_xlabel("prompt length (tokens)")
    a.set_ylabel("seconds")
    a.set_title("(a) GPT-2", fontsize=7)
    a.legend(frameon=False, fontsize=4.6, loc="center", bbox_to_anchor=(0.5, 0.62), ncol=2, handlelength=1.6,
             columnspacing=0.6)
    a.grid(True, lw=0.3, alpha=0.4)
    zk = {r["model"]: r for r in _read("reported_curated.csv") if r["system"] == "zkLLM" and r["seq"] == "2048"}
    pairs = [("opt-125m", "OPT-125M"), ("opt-1.3b", "OPT-1.3B"), ("opt-6.7b", "OPT-6.7B"),
             ("llama2-7b", "Llama-2-7B")]
    xs = range(len(pairs))
    ours = [_llm_cost(rows[(m, 2048)])[0] for m, _ in pairs]
    them = [_f(zk[name]["prover_s"]) for _, name in pairs]
    b.bar([x - 0.2 for x in xs], ours, 0.38, color=OURS, label="ours (RTX 2080 Ti)")
    b.bar([x + 0.2 for x in xs], them, 0.38, color=THEM, label="zkLLM (A100)")
    for x, o, t in zip(xs, ours, them):
        b.text(x, max(o, t) * 1.25, f"{t / o:.0f}$\\times$", ha="center", fontsize=5.5)
    b.set_yscale("log")
    b.set_ylim(0.5, 4e4)
    b.set_xticks(list(xs))
    b.set_xticklabels([LLM_NAME[m] for m, _ in pairs], fontsize=5, rotation=20)
    b.set_ylabel("prover time (s)")
    b.set_title("(b) 2048 tokens: prover", fontsize=7)
    b.legend(frameon=False, fontsize=4.9, loc="upper left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "llm")


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
    (TABS / f"{name}.tex").write_text("\n".join(lines + [r"\bottomrule"]) + "\n", encoding="utf-8")
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
    pick = [("gpt2", 64), ("gpt2", 512), ("opt-125m", 2048), ("opt-1.3b", 2048), ("qwen3-4b", 64),
            ("opt-6.7b", 2048), ("llama2-7b", 64), ("llama2-7b", 2048), ("llama2-13b", 64)]
    lines = []
    for model, seq in pick:
        p, v, by, n = _llm_cost(c[(model, seq)])
        kp, kv, kb, _ = _llm_cost(k[(model, seq)])
        lines.append(" & ".join([LLM_NAME[model], f"{seq:,}".replace(",", "{,}"), t(p), t(v), b(by), t(kv), b(kb)]) + r" \\")
    write("llm", lines)


def tab_ratios():
    from comparison_report_tables import MATCHES, NAMES  # noqa: F401  (same pairs as COMPARISON_ANALYSIS.md)
    M = Measured()
    curated = _read("reported_curated.csv")
    c_llm = llm_rows("C")
    k1 = llm_rows("Kpre", lam=40, threads="1", variant="_thr1")
    keep = {("zkCNN", "LeNet-5 MNIST"), ("zkCNN", "VGG-16 CIFAR-10"), ("ZKML", "VGG-16 CIFAR-10"),
            ("Bionetta", "LeNet-5"), ("zkGPT", "GPT-2"), ("DeepProve", "GPT-2"), ("zkLLM", "OPT-125M"),
            ("zkLLM", "OPT-6.7B"), ("zkLLM", "Llama-2-7B"), ("ZKTorch", "Llama-2-7B (1 token)"),
            ("Maverick (verif.-only, 1 thr.)", "Qwen3-4B")}
    lines = []
    for system, model, ours, seq, mode, note in MATCHES:
        if (system, model) not in keep or (system == "DeepProve" and seq not in (64, 512)):
            continue
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


def main():
    M = Measured()
    fig_overview()
    fig_security(M)
    fig_cost(M)
    fig_llm(M)
    tab_cnn(M)
    tab_llm()
    tab_ratios()


if __name__ == "__main__":
    main()

"""Every figure and table body of the report, built from ``artifacts/comparison/tables``.

    python experiments/5_comparison/paper_assets.py   # -> ../report/figures/*.pdf, ../report/tables/*.tex

Nothing here runs a model; every number is read from the stored benchmark tables,
so the report can be rebuilt without a GPU.  Figures 1 and 2 are schematics.  Each panel is
its own file (sub-captions are set in LaTeX); sizes match the width they are printed
at, so the fonts appear at their real size.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

from common import CNN_ORDER, DEFAULT, PROVE, VERIFY, ROOT, Measured, _f, _llm_cost, _read, llm_rows  # noqa: E402
from report_tables import MATCHES  # noqa: E402  (the same published/ours pairs as report_tables.md)

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
LLM_NAME = {"gpt2": "GPT-2", "opt-125m": "OPT-125M", "opt-1.3b": "OPT-1.3B", "opt-6.7b": "OPT-6.7B",
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
    for mode, out, seqs in (("C", ours_c, (64, 2048)), ("Kpre", ours_k, (64,))):
        for (model, seq), d in sorted(llm_rows(mode).items()):
            if "prove_forward" in d and seq in seqs:
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
def fig_llm():
    """Prover time at 2,048 tokens: ours (RTX 2080 Ti) against zkLLM (A100) on the same models."""
    rows = llm_rows("C")
    zk = {r["model"]: r for r in _read("reported_curated.csv") if r["system"] == "zkLLM" and r["seq"] == "2048"}
    pairs = [("opt-125m", "OPT-125M"), ("opt-1.3b", "OPT-1.3B"), ("opt-6.7b", "OPT-6.7B"), ("llama2-7b", "Llama-2-7B")]
    fig, b = plt.subplots(figsize=(3.3, 1.3))
    xs = range(len(pairs))
    ours = [_llm_cost(rows[(m, 2048)])[0] for m, _ in pairs]
    them = [_f(zk[name]["prover_s"]) for _, name in pairs]
    b.bar([x - 0.2 for x in xs], ours, 0.38, color=OURS, label="ours (RTX 2080 Ti)")
    b.bar([x + 0.2 for x in xs], them, 0.38, color=THEM, label="zkLLM (A100)")
    for x, o, t_ in zip(xs, ours, them):
        b.text(x, t_ * 1.3, f"{t_ / o:.0f}$\\times$", ha="center", fontsize=6.3)
    b.set_yscale("log")
    b.set_ylim(0.5, 1e4)
    b.set_xticks(list(xs))
    b.set_xticklabels([LLM_NAME[m] for m, _ in pairs])
    b.set_ylabel("prover time (s)")
    b.legend(frameon=False, loc="upper left", ncol=2, fontsize=6)
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
    pick = [("gpt2", 64), ("gpt2", 512), ("opt-125m", 2048), ("opt-1.3b", 2048), ("qwen3-4b", 64),
            ("opt-6.7b", 2048), ("llama2-7b", 64), ("llama2-7b", 2048), ("llama2-13b", 64)]
    lines = []
    for model, seq in pick:
        p, v, by, n = _llm_cost(c[(model, seq)])
        kp, kv, kb, _ = _llm_cost(k[(model, seq)])
        lines.append(" & ".join([LLM_NAME[model], f"{seq:,}".replace(",", "{,}"), t(p), t(v), b(by), t(kv), b(kb)]) + r" \\")
    write("llm", lines)


def tab_ratios():
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
    fig_protocol()
    fig_security(M)
    fig_cost(M)
    fig_llm()
    tab_cnn(M)
    tab_llm()
    tab_ratios()


if __name__ == "__main__":
    main()

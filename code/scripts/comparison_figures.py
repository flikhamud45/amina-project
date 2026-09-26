"""Figures for the comparison, built from ``artifacts/comparison/tables`` only.

    python scripts/comparison_aggregate.py && python scripts/comparison_literature.py
    python scripts/comparison_figures.py            # -> artifacts/comparison/figures/*.pdf|png

Nothing here runs a model: every point is a row of ``measured_summary.csv``,
``llm_full_model.csv`` or ``reported_curated.csv``, so the figures can be
re-styled or re-selected without re-running the benchmark.  Headline settings
(``DEFAULT``) pick one configuration per point; nothing is chosen by sort order.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "artifacts" / "comparison" / "tables"
FIGS = ROOT / "artifacts" / "comparison" / "figures"

DEFAULT = {"rate": "4", "threads": "8", "variant": "", "lam": 128}

plt.rcParams.update({"font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.3,
                     "xtick.labelsize": 7, "ytick.labelsize": 7, "figure.dpi": 200, "savefig.bbox": "tight",
                     "font.family": "serif", "axes.spines.top": False, "axes.spines.right": False})

OURS = "#c0392b"
OURS_K = "#2471a3"
GREY = "0.35"
CNN_ORDER = ["mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar", "resnet18_224"]
CNN_LABEL = {"mlp_mnist": "MLP\n(MNIST)", "lenet5": "LeNet-5\n(MNIST)", "vgg11": "VGG-11\n(CIFAR-10)",
             "vgg16": "VGG-16\n(CIFAR-10)", "resnet18_cifar": "ResNet-18\n(CIFAR-10)",
             "resnet18_224": "ResNet-18\n(224px, [1])"}
LLM_LABEL = {"gpt2": "GPT-2", "opt-125m": "OPT-125M", "opt-1.3b": "OPT-1.3B", "opt-6.7b": "OPT-6.7B",
             "llama2-7b": "Llama-2-7B", "llama2-13b": "Llama-2-13B", "qwen3-4b": "Qwen3-4B"}
MARKERS = [("zkLLM", "s"), ("zkCNN", "^"), ("ZKML", "D"), ("zkGPT", "v"), ("DeepProve", "P"), ("ZKTorch", "X"),
           ("Bionetta", "<"), ("EZKL", ">"), ("Mystique", "p"), ("Maverick", "*"), ("vCNN", "d"),
           ("zkPyTorch", "h"), ("Jolt", "H"), ("SLP", "8"), ("LAMP", "o"), ("Anchuri", "X")]
PROVE = ("prove_forward", "prove_fold", "prove_open", "fs_hash")
VERIFY = ("verify_derive", "verify_fold", "verify_products", "verify_columns", "fs_hash")


def _read(name: str) -> list[dict]:
    path = TABLES / name
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
    """Index over ``measured_summary.csv``."""

    def __init__(self) -> None:
        self.idx = {}
        for r in _read("measured_summary.csv"):
            key = (r["suite"], r["model"], r["cell"], r["metric"], r.get("batch", ""), r.get("attack", ""),
                   r.get("lam", ""), r.get("stage", ""))
            self.idx[key] = r

    def get(self, model, cell, metric, stat="median", suite="cnn", batch="", attack="", lam="", stage=""):
        r = self.idx.get((suite, model, cell, metric, batch, attack, str(lam) if lam != "" else "", stage))
        return _f(r[stat]) if r else None

    def total(self, model, cell, parts, suite="cnn"):
        vals = [self.get(model, cell, k, suite=suite) for k in parts]
        return None if all(v is None for v in vals) else sum(v or 0 for v in vals)

    def attack_rate(self, model, cell, attack):
        vals = [(_f(r["mean"]), _f(r["n"])) for k, r in self.idx.items()
                if k[1] == model and k[2] == cell and k[3] == "rejected" and k[5] == attack]
        n = sum(v[1] for v in vals)
        return (sum(v[0] * v[1] for v in vals) / n, int(n)) if n else (None, 0)


def llm_rows(mode="C", lam=DEFAULT["lam"], chal="int") -> dict:
    """``{(model, seq): {metric: (value, provenance, params)}}`` at the headline settings."""
    out: dict = defaultdict(dict)
    for r in _read("llm_full_model.csv"):
        if r["metric"] == "commit_total":
            continue
        if (r["mode"], r["challenges"]) != (mode, chal) or _f(r["lam"]) != lam:
            continue
        if (r.get("rate", ""), r.get("threads", ""), r.get("variant", "")) != (
                DEFAULT["rate"], DEFAULT["threads"], DEFAULT["variant"]):
            continue
        key = (r["model"], int(float(r["seq"])))
        if r["metric"] in out[key]:
            raise SystemExit(f"two headline rows for {key} {r['metric']}")
        out[key][r["metric"]] = (float(r["value"]), r["provenance"], _f(r["params_full"]))
    return out


def _llm_bytes(d: dict, part: str = "total"):
    """Proof bytes, with the Merkle term as a multiproof (native or expected; see aggregate)."""
    key = f"bytes_{part}_multiproof"
    return d[key][0] if key in d else d.get(f"bytes_{part}", (None,))[0]


def _llm_cost(d: dict):
    prove = sum(d[k][0] for k in PROVE if k in d)
    verify = sum(d[k][0] for k in VERIFY if k in d)
    return prove, verify, _llm_bytes(d), d["prove_forward"][2]


def _marker(system: str) -> str:
    for k, m in MARKERS:
        if system.startswith(k):
            return m
    return "o"


def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(FIGS / f"{name}.{ext}")
    plt.close(fig)
    print("wrote", name)


# ------------------------------------------------------------------------------ figures
def fig_cost_vs_size(M: Measured) -> None:
    """Prover time, verifier time and proof size against model size: ours vs published."""
    rep = [r for r in _read("reported_curated.csv") if _f(r["params"])]
    ours_c, ours_k = [], []
    for m in CNN_ORDER:
        n = M.get(m, "facts", "n_params")
        cell = f"defence_C_int_lam{DEFAULT['lam']}_rate4"
        if n is None or M.get(m, cell, "prove_forward") is None:
            continue
        ours_c.append((n, M.total(m, cell, PROVE), M.total(m, cell, VERIFY), M.get(m, cell, "bytes_total"), "cnn"))
        kc = f"defence_Kpre_int_lam{DEFAULT['lam']}_rate4"
        if M.get(m, kc, "prove_forward") is not None:
            ours_k.append((n, M.total(m, kc, PROVE), M.total(m, kc, VERIFY), M.get(m, kc, "bytes_total"), "cnn"))
    for (model, seq), d in sorted(llm_rows("C").items()):
        if "prove_forward" in d and seq in (64, 2048):
            p, v, b, n = _llm_cost(d)
            ours_c.append((n, p, v, b, f"T{seq}"))
    for (model, seq), d in sorted(llm_rows("Kpre").items()):
        if "prove_forward" in d and seq == 64:
            p, v, b, n = _llm_cost(d)
            ours_k.append((n, p, v, b, f"T{seq}"))

    fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.45))
    panels = (("prover_s", 1, "prover time (s)"), ("verifier_s", 2, "verifier time (s)"),
              ("proof_bytes", 3, "proof / communication (bytes)"))
    for ax, (key, col, title) in zip(axes, panels):
        for r in rep:
            y = _f(r[key])
            if y is None:
                continue
            if r["system"].startswith("Anchuri"):   # the protocol we attack: highlighted
                ax.scatter(_f(r["params"]), y, marker="X", s=30, c="k", zorder=3)
                continue
            ax.scatter(_f(r["params"]), y, marker=_marker(r["system"]), s=13, c="none", edgecolors=GREY,
                       linewidths=0.6, zorder=2)
        for pts, colour in ((ours_c, OURS), (ours_k, OURS_K)):
            for n, *vals, kind in pts:
                y = vals[col - 1]
                if y is None:
                    continue
                hollow = kind == "T2048"
                ax.scatter(n, y, marker="o" if kind == "cnn" else "s", s=20,
                           c="none" if hollow else colour, edgecolors=colour, linewidths=0.9, zorder=4)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel("model parameters")
        ax.set_title(title)
        ax.grid(True, which="major", lw=0.3, alpha=0.4)
    xs = [1e4, 3e10]
    axes[2].plot(xs, xs, "k--", lw=0.6, zorder=1)
    axes[2].text(2e7, 6e7, "int8 model", fontsize=5.8, rotation=30)
    handles = [Line2D([], [], marker="o", ls="", color=OURS, label="ours, weights committed (CNN)"),
               Line2D([], [], marker="s", ls="", color=OURS, label="ours, committed (LLM, 64 tok.)"),
               Line2D([], [], marker="s", ls="", mfc="none", color=OURS, label="ours, committed (LLM, 2048 tok.)"),
               Line2D([], [], marker="o", ls="", color=OURS_K, label="ours, verifier knows weights (CNN)"),
               Line2D([], [], marker="s", ls="", color=OURS_K, label="ours, knows weights (LLM, 64 tok.)"),
               Line2D([], [], marker="X", ls="", color="k", label="Anchuri et al. [1], one path")]
    seen = set()
    for r in rep:
        name = r["system"].split(" (")[0]
        if name in seen or name.startswith("Anchuri"):
            continue
        seen.add(name)
        handles.append(Line2D([], [], marker=_marker(r["system"]), ls="", mfc="none", color=GREY, label=name))
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False, bbox_to_anchor=(0.5, -0.2))
    fig.tight_layout()
    _save(fig, "cost_vs_model_size")


def fig_detection(M: Measured) -> None:
    models = [m for m in CNN_ORDER if M.get(m, "sampling", "p_detect_penultimate") is not None]
    if not models:
        return
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    xs = list(range(len(models)))
    one = [M.get(m, "sampling", "p_detect_penultimate") for m in models]
    worst = [M.get(m, "sampling", "p_detect_min_node") for m in models]
    ours = [M.attack_rate(m, "tamper_C_int_lam40", "penultimate_neuron")[0] for m in models]
    ax.bar([x - 0.27 for x in xs], one, 0.26, color="0.55", label="[1], 1 path: attack neuron")
    ax.bar(xs, worst, 0.26, color="0.8", label="[1], 1 path: least-visited neuron")
    ax.bar([x + 0.27 for x in xs], [o if o is not None else 0 for o in ours], 0.26, color=OURS,
           label="ours (measured)")
    ax.set_yscale("log")
    ax.set_ylim(1e-5, 2)
    ax.set_xticks(xs)
    ax.set_xticklabels([CNN_LABEL[m] for m in models], fontsize=5.6)
    ax.set_ylabel("P[tamper detected] per query")
    ax.legend(frameon=False, loc="lower left", fontsize=5.8)
    ax.grid(True, axis="y", lw=0.3, alpha=0.4)
    _save(fig, "detection_single_neuron")


def fig_security(M: Measured) -> None:
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for i, m in enumerate(CNN_ORDER):
        lams, ours, samp = [], [], []
        for lam in (40, 80, 128):
            b = M.get(m, f"defence_C_int_lam{lam}_rate4", "bytes_total")
            s = M.get(m, "sampling", "paths_bytes_shared", lam=lam)
            if b is None or s is None:
                continue
            lams.append(lam)
            ours.append(b)
            samp.append(s)
        if not lams:
            continue
        colour = f"C{i}"
        ax.plot(lams, ours, "-o", color=colour, ms=2.8, lw=1, label=CNN_LABEL[m].replace("\n", " "))
        ax.plot(lams, samp, ":s", color=colour, ms=2.8, lw=1)
        cap = M.get(m, "sampling", "open_all_bytes")
        if cap:
            ax.axhline(cap, color=colour, lw=0.4, alpha=0.5)
    ax.set_yscale("log")
    ax.set_xlabel("security level λ (bits)")
    ax.set_ylabel("bytes per query")
    ax.set_title("solid: ours   dotted: [1] with k(λ) paths (shared openings)", fontsize=6.3)
    ax.legend(frameon=False, fontsize=5.4, ncol=2)
    ax.grid(True, lw=0.3, alpha=0.4)
    _save(fig, "bytes_vs_security")


def fig_seq_scaling() -> None:
    rows = llm_rows("C")
    seqs = sorted(s for (m, s), d in rows.items() if m == "gpt2" and "prove_forward" in d)
    if not seqs:
        return
    rep = [r for r in _read("reported_curated.csv") if r["system"] == "DeepProve"]
    zk = [r for r in _read("reported_curated.csv") if r["system"] == "zkGPT"]
    fig, axes = plt.subplots(1, 2, figsize=(3.5, 1.75))
    cost = [_llm_cost(rows[("gpt2", s)]) for s in seqs]
    axes[0].plot(seqs, [c[0] for c in cost], "-o", color=OURS, ms=3, label="ours (prove)")
    axes[0].plot(seqs, [c[1] for c in cost], "--o", color=OURS, ms=3, mfc="none", label="ours (verify)")
    axes[0].plot([_f(r["seq"]) for r in rep], [_f(r["prover_s"]) for r in rep], "-P", color=GREY, ms=3,
                 label="DeepProve (prove)")
    axes[0].plot([_f(r["seq"]) for r in rep], [_f(r["verifier_s"]) for r in rep], "--P", color=GREY, ms=3,
                 mfc="none", label="DeepProve (verify)")
    axes[1].plot(seqs, [c[2] for c in cost], "-o", color=OURS, ms=3)
    axes[1].plot([_f(r["seq"]) for r in rep], [_f(r["proof_bytes"]) for r in rep], "-P", color=GREY, ms=3)
    for r in zk:
        axes[0].scatter(64, _f(r["prover_s"]), marker="v", c="none", edgecolors=GREY, s=14)
        axes[1].scatter(64, _f(r["proof_bytes"]), marker="v", c="none", edgecolors=GREY, s=14)
    for ax, t in zip(axes, ("seconds", "proof bytes")):
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel("prompt length (tokens)")
        ax.set_title(t)
        ax.grid(True, lw=0.3, alpha=0.4)
    axes[0].legend(frameon=False, fontsize=5, loc="upper left")
    fig.tight_layout()
    _save(fig, "gpt2_seq_scaling")


def fig_breakdown(M: Measured) -> None:
    labels, parts = [], []
    for m in CNN_ORDER:
        cell = f"defence_C_int_lam{DEFAULT['lam']}_rate4"
        v = [M.get(m, cell, "bytes_" + k) for k in ("claims", "u", "columns", "paths")]
        if None not in v:
            labels.append(CNN_LABEL[m])
            parts.append(v)
    for (model, seq), d in sorted(llm_rows("C").items()):
        if seq == 64 and all(("bytes_" + k) in d for k in ("claims", "u", "columns", "paths")):
            labels.append(f"{LLM_LABEL.get(model, model)}\n(64 tok.)")
            parts.append([d["bytes_" + k][0] for k in ("claims", "u", "columns")] + [_llm_bytes(d, "paths")])
    if not parts:
        return
    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    names = ("claimed pre-activations", "folded rows u", "opened columns", "Merkle paths")
    colours = (OURS, OURS_K, "#58a55c", "0.6")
    for j in range(4):
        tot = [sum(p) for p in parts]
        ax.bar(range(len(parts)), [p[j] / t for p, t in zip(parts, tot)],
               bottom=[sum(p[:j]) / t for p, t in zip(parts, tot)], color=colours[j], label=names[j])
    ax.set_xticks(range(len(parts)))
    ax.set_xticklabels(labels, fontsize=5)
    ax.set_ylabel(f"share of proof (λ={DEFAULT['lam']})")
    ax.legend(frameon=False, fontsize=5.2, ncol=2, loc="upper center", bbox_to_anchor=(0.5, 1.28))
    _save(fig, "proof_breakdown")


def fig_maverick() -> None:
    """Qwen3-4B, 8-token prompt, verifier knows the weights: ours vs Maverick."""
    rep = {r["system"]: r for r in _read("reported_curated.csv") if r["system"].startswith("Maverick")}
    rows = {}
    for r in _read("llm_full_model.csv"):
        if r["model"] == "qwen3-4b" and int(float(r["seq"])) == 8 and r["mode"] == "Kpre" and _f(r["lam"]) == 40 \
                and r.get("rate", "") == "4":
            rows.setdefault(r.get("threads", ""), {})[r["metric"]] = float(r["value"])
    if not rows or not rep:
        return
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.7))
    labels, verify, comm, colours = [], [], [], []
    for thr, d in sorted(rows.items(), key=lambda kv: int(kv[0] or 0)):
        labels.append(f"ours\n{thr} thr.")
        verify.append(sum(d.get(k, 0) for k in VERIFY))
        comm.append(d.get("bytes_total", 0))
        colours.append(OURS_K)
    for name in ("Maverick (verif.-only, 1 thr.)", "Maverick (verif.-only, 8 thr.)"):
        if name in rep:
            labels.append("Maverick\n" + name.split(", ")[1].rstrip(")"))
            verify.append(_f(rep[name]["verifier_s"]))
            comm.append(_f(rep[name]["proof_bytes"]))
            colours.append("0.6")
    axes[0].bar(range(len(labels)), [v * 1e3 for v in verify], color=colours)
    axes[1].bar(range(len(labels)), [c / 1e6 for c in comm], color=colours)
    for ax, t in zip(axes, ("client verification (ms)", "communication (MB)")):
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, fontsize=5)
        ax.set_title(t)
    fig.tight_layout()
    _save(fig, "qwen3_4b_vs_maverick")


def main() -> None:
    M = Measured()
    fig_cost_vs_size(M)
    fig_detection(M)
    fig_security(M)
    fig_seq_scaling()
    fig_breakdown(M)
    fig_maverick()


if __name__ == "__main__":
    main()

"""Figures G3-a/b and Table G3 from the adaptive-game JSON files (plan G3-min, item 4).

Reads the JSON written by ``adaptive_game.py`` (and, if present, ``adaptive_game_llm.py``) and emits:

* **Figure G3-a** (one panel per model/layer): the best adversary detection per path (log y) against
  the cap percentile, one line per sampler plus the LP mixture, with a horizontal line at ``27.7/N``
  (the level at which ``2^-40`` needs fewer than ``N`` paths).
* **Figure G3-b**: detection against the number of paths ``s`` for uniform, the best single sampler
  and the mixture, at caps 100 and 95, using ``1 - (1 - d)^s``.
* **Table G3** (LaTeX): rows = model/layer, columns = uniform, best single value-aware sampler
  (named), LP mixture -- each as per-path ``d``, multiple of uniform and ``s_40`` -- plus the success
  rate and mean support at the 95th-percentile cap.

Usage
-----
    cd code && PYTHONPATH=src python experiments/3_sampling_fixes/adaptive_game_plots.py \
        --mnist artifacts/results/adaptive_game_mlp_mnist.json \
        --out-dir artifacts/results
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "artifacts" / "results"
CAP_ORDER = ["50", "75", "90", "95", "99", "100", "inf"]
CAP_LABEL = {"50": "50", "75": "75", "90": "90", "95": "95", "99": "99", "100": "100", "inf": r"$\infty$"}


def _cap_x(key: str) -> float:
    return 101.0 if key == "inf" else float(key)


def load_models(mnist: str | None, llm: str | None) -> list[dict]:
    """Flatten the JSON files into one row per (model, layer)."""
    rows: list[dict] = []
    if mnist and Path(mnist).exists():
        data = json.loads(Path(mnist).read_text())
        for layer, lo in data["layers"].items():
            rows.append({
                "label": f"MNIST MLP, layer {layer}",
                "width": lo["width"],
                "caps": lo["caps"],
                "source": "mnist",
            })
    if llm and Path(llm).exists():
        data = json.loads(Path(llm).read_text())
        for layer, lo in data["per_layer"].items():
            rows.append({
                "label": f"{data['model'].split('/')[-1]}, layer {layer}",
                "width": lo["d_ff"],
                "caps": lo["caps"],
                "source": "llm",
            })
    return rows


# --------------------------------------------------------------------------- #
# Figure G3-a
# --------------------------------------------------------------------------- #


FLOOR = 1e-5  # detections of exactly 0 (to rounding) are drawn here and labelled "0"


def figure_a(rows: list[dict], out: Path) -> None:
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4), squeeze=False)
    handles: dict[str, object] = {}
    for ax, row in zip(axes[0], rows):
        caps = row["caps"]
        cap_keys = [k for k in CAP_ORDER if k in caps and caps[k].get("samplers")]
        sampler_names = list(next(iter(caps.values()))["samplers"])
        xs = list(range(len(cap_keys)))  # evenly spaced: the caps are categories, not a scale

        def clip(d: float) -> float:
            return d if d > FLOOR else FLOOR

        for name in sampler_names:
            ys = [clip(caps[k]["samplers"][name]["detection_per_path"]) for k in cap_keys]
            style = "--" if name.endswith("*") else "-"
            (h,) = ax.plot(xs, ys, style, marker="o", ms=3, lw=1, alpha=0.8)
            handles[name] = h
        mix = [clip(caps[k]["mixture"]["detection_per_path"]) for k in cap_keys]
        (h,) = ax.plot(xs, mix, "k-", marker="s", ms=4, lw=2)
        handles["LP mixture (deployable)"] = h
        if all(caps[k].get("mixture_upper_bound_with_saliency") for k in cap_keys):
            mix_ub = [clip(caps[k]["mixture_upper_bound_with_saliency"]["detection_per_path"])
                      for k in cap_keys]
            (h,) = ax.plot(xs, mix_ub, color="grey", ls=":", marker="s", ms=3, lw=1.3)
            handles["LP mixture with saliency* (upper bound)"] = h
        h = ax.axhline(27.7 / row["width"], color="red", ls=":", lw=1)
        handles[r"$27.7/N$"] = h
        ax.set_yscale("log")
        ax.set_ylim(FLOOR / 2, 0.5)
        ticks = [FLOOR, 1e-4, 1e-3, 1e-2, 1e-1]
        ax.set_yticks(ticks)
        ax.set_yticklabels(["0"] + [f"$10^{{{int(math.log10(t))}}}$" for t in ticks[1:]])
        ax.set_xticks(xs)
        ax.set_xticklabels([CAP_LABEL[k] for k in cap_keys])
        ax.set_xlabel("cap percentile $p$")
        ax.set_ylabel("detection per path $d$")
        ax.set_title(f"{row['label']} ($N={row['width']}$)", fontsize=9)
        ax.grid(True, which="major", alpha=0.25)
    fig.legend(list(handles.values()), list(handles.keys()), loc="lower center", ncol=5, fontsize=6,
               frameon=False)
    fig.tight_layout(rect=(0, 0.16, 1, 1))
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# Figure G3-b
# --------------------------------------------------------------------------- #


def figure_b(rows: list[dict], out: Path) -> None:
    """Acceptance ``(1 - d)^s`` (log scale) against the number of paths ``s``.

    On a log axis the ``2^-40`` target is visible, which a linear detection axis hides (every curve is
    at 1 long before it); the vertical line at ``s = N`` marks where paths outnumber the layer's width.
    """
    import numpy as np

    s_values = np.logspace(0, 5, 400)
    target = 2.0 ** -40
    n = len(rows)
    fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 3.4), squeeze=False)
    colours = {"uniform": "tab:blue", "best": "tab:orange", "mix": "black"}
    for ax, row in zip(axes[0], rows):
        caps = row["caps"]
        for cap_key, dash in (("100", "-"), ("95", "--")):
            if cap_key not in caps or not caps[cap_key].get("samplers"):
                continue
            cell = caps[cap_key]
            best = cell.get("best_single_implementable", cell["best_single_sampler"])
            series = (
                ("uniform", f"uniform (p={cap_key})", cell["samplers"]["uniform"]["detection_per_path"]),
                ("best", f"{best} (p={cap_key})", cell["samplers"][best]["detection_per_path"]),
                ("mix", f"LP mixture (p={cap_key})", cell["mixture"]["detection_per_path"]),
            )
            for colour, name, d in series:
                acceptance = np.exp(s_values * np.log1p(-min(max(d, 0.0), 1.0 - 1e-12)))
                ax.plot(s_values, np.maximum(acceptance, 1e-16), dash, color=colours[colour], lw=1.2,
                        label=name)
        ax.axhline(target, color="red", ls=":", lw=1, label=r"$2^{-40}$")
        ax.axvline(row["width"], color="grey", ls="-.", lw=1, label=f"$s=N={row['width']}$")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylim(1e-15, 2.0)
        ax.set_xlabel("paths $s$")
        ax.set_ylabel(r"acceptance $(1-d)^s$")
        ax.set_title(row["label"], fontsize=9)
        ax.grid(True, which="major", alpha=0.25)
        ax.legend(fontsize=6, loc="lower left")
    fig.tight_layout()
    fig.savefig(out, format="pdf")
    plt.close(fig)
    print(f"wrote {out}")


# --------------------------------------------------------------------------- #
# Table G3
# --------------------------------------------------------------------------- #


def _fmt(d: float) -> str:
    if d <= 0:
        return "0"
    return f"{100 * d:.2f}\\%"


def _s40(cell: dict, key: str) -> str:
    s = cell.get("s_40") if key == "mixture" else cell["samplers"][key]["s_40"] if key in cell.get("samplers", {}) else None
    return "--" if s is None else f"{s:,}"


def table_g3(rows: list[dict], out: Path, cap_key: str = "95") -> None:
    lines = [
        "% Table G3: the adaptive game at the 95th-percentile cap (generated by adaptive_game_plots.py).",
        "% Each detection is the exact single-path probability against the adversary's best found",
        "% in-range attack (A1-A4), an UPPER BOUND on the sampler's true worst case.",
        "\\begin{tabular}{@{}lrrrrrr@{}}",
        "\\toprule",
        "Model / layer & $N$ & uniform $d$ & best single ($d$, $\\times$unif) & LP mix ($d$, $\\times$unif) & "
        "$s_{40}$ (unif/best/mix) & succ., $\\bar k$ \\\\",
        "\\midrule",
    ]
    for row in rows:
        caps = row["caps"]
        if cap_key not in caps or not caps[cap_key].get("samplers"):
            continue
        cell = caps[cap_key]
        unif = cell["samplers"]["uniform"]["detection_per_path"]
        best = cell.get("best_single_implementable", cell["best_single_sampler"])
        best_d = cell["samplers"][best]["detection_per_path"]
        mix_d = cell["mixture"]["detection_per_path"]
        ratio = (best_d / unif) if unif > 0 else float("inf")
        s_u = cell["samplers"]["uniform"]["s_40"]
        s_b = cell["samplers"][best]["s_40"]
        s_m = cell["mixture"]["s_40"]
        succ = cell.get("success_rate")
        kbar = cell.get("mean_support", cell.get("mean_min_support"))
        name = best.replace("_", "\\_").replace("*", "$^\\ast$")
        s_cell = "/".join("--" if s is None else f"{s:,}" for s in (s_u, s_b, s_m))
        succ_cell = (f"{succ:.0%}" if succ is not None else "--").replace("%", "\\%")
        kbar_cell = f"{kbar:.1f}" if kbar is not None else "--"

        def times(x: float) -> str:
            return "$\\infty$" if math.isinf(x) else f"{x:.1f}$\\times$"

        mix_ratio = (mix_d / unif) if unif > 0 else float("inf")
        lines.append(
            f"{row['label']} & {row['width']} & {_fmt(unif)} & {name} ({_fmt(best_d)}, {times(ratio)}) & "
            f"{_fmt(mix_d)}, {times(mix_ratio)} & {s_cell} & {succ_cell}, {kbar_cell} \\\\"
        )
    lines += ["\\bottomrule", "\\end{tabular}", ""]
    out.write_text("\n".join(lines))
    print(f"wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mnist", default=str(RESULTS / "adaptive_game_mlp_mnist.json"))
    ap.add_argument("--llm", default=str(RESULTS / "adaptive_game_opt-1.3b.json"))
    ap.add_argument("--out-dir", default=str(RESULTS))
    ap.add_argument("--cap-key", default="95", help="cap percentile for Table G3")
    args = ap.parse_args()

    rows = load_models(args.mnist, args.llm)
    if not rows:
        raise SystemExit("no input JSON found; run adaptive_game.py first")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    figure_a(rows, out_dir / "figure_g3a.pdf")
    figure_b(rows, out_dir / "figure_g3b.pdf")
    table_g3(rows, out_dir / "table_g3.tex", cap_key=args.cap_key)


if __name__ == "__main__":
    main()

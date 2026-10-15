"""The paper's figures that are drawn from code or from the run's records, written to ``paper_sp2027/figures/``.

* ``overview_attack.pdf`` and ``overview_defence.pdf`` (``fig:overview``, Sec. ``sec:intro``) and ``protocol.pdf``
  (``fig:protocol``, extended version): schematics, drawn from code only (:func:`schematics`).
* ``ev_breakeven.pdf`` (``fig:breakeven``, Sec. ``sec:eval-reexec``): the client's bytes against one download of the
  weights and a Kpre client's time against re-execution, by ``deployment/ev_breakeven.py``'s drawing on the
  crossovers that :func:`experiments.paper.macros.breakeven` computes from the run's records (:func:`breakeven`).

Each schematic is saved on a fixed canvas equal to its printed width (no tight cropping), so that its 7 pt text
prints at its real size; :func:`save` refuses a figure whose text is smaller than 7 pt, leaves the canvas, overlaps
other text or covers a marker.  Fonts are embedded as TrueType (Type 42) and the PDFs carry no creation date, so the
same matplotlib writes the same bytes.  Needs matplotlib (``requirements.txt``); seconds on a laptop.
"""
from __future__ import annotations

from pathlib import Path

MIN_PT = 7.0
"""No text below 7 pt at print size."""
OURS = "#1f5f99"        # the defence
ANCH = "#e67e22"        # the path protocol of Anchuri et al.
ATTACK = "#c0392b"      # tampered values
ATTACK_LIGHT = "#f5d0cb"
MSG = "0.25"            # protocol messages
SF_C = "\U0001d5a2"     # upright sans-serif C: the paper's \mathsf{C}
WIDTHS = [5, 7, 7, 3]
"""The neurons per layer of the schematic network."""
PATH = [(3, 1), (2, 4), (1, 2), (0, 3)]
"""The checked output-to-input path of the attack panel."""
TAMPERED = (1, 5)
OVERVIEW = (1.635, 1.2)
"""The overview panels' canvas in inches (printed at about half a column width: 7 pt text)."""


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.serif": ["STIXGeneral"], "mathtext.fontset": "stix",
                         "font.size": 7, "axes.labelsize": 7, "axes.titlesize": 7, "legend.fontsize": 7,
                         "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.spines.top": False,
                         "axes.spines.right": False, "axes.linewidth": 0.6, "savefig.bbox": None, "figure.dpi": 200,
                         "pdf.fonttype": 42, "ps.fonttype": 42})
    return plt


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
    w, h = fig.bbox.width, fig.bbox.height
    texts = [(t, t.get_window_extent(r)) for t in _drawn_texts(fig)]
    bad = []
    for t, bb in texts:
        if t.get_fontsize() < MIN_PT - 1e-6:
            bad.append(f"{t.get_text()!r}: {t.get_fontsize():.1f} pt")
        if bb.x0 < -0.5 or bb.y0 < -0.5 or bb.x1 > w + 0.5 or bb.y1 > h + 0.5:
            bad.append(f"{t.get_text()!r} leaves the canvas")
    for i, (t, a) in enumerate(texts):
        for u, c in texts[i + 1:]:
            if a.overlaps(c) and min(a.x1, c.x1) - max(a.x0, c.x0) > 1 and min(a.y1, c.y1) - max(a.y0, c.y0) > 1:
                bad.append(f"{t.get_text()!r} overlaps {u.get_text()!r}")
    for ax in fig.axes:
        pts = _marker_points(ax)
        for t in ax.texts:
            bb = t.get_window_extent(r).expanded(1.0, 1.0)
            if any(bb.x0 + 1 < x < bb.x1 - 1 and bb.y0 + 1 < y < bb.y1 - 1 for x, y in pts):
                bad.append(f"{t.get_text()!r} covers a marker")
    return bad


def save(fig, path: Path, width: float) -> Path:
    """Write ``path`` on the figure's fixed canvas after the layout checks."""
    import matplotlib.pyplot as plt
    if abs(fig.get_size_inches()[0] - width) > 0.01:
        raise ValueError(f"{path.name}: canvas {fig.get_size_inches()} is not {width} in wide")
    bad = layout_problems(fig)
    if bad:
        raise ValueError(f"figure {path.name}: " + "; ".join(bad))
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, metadata={"CreationDate": None}, bbox_inches=None)
    plt.close(fig)
    return path


def _pos():
    return {(layer, i): (layer, (i - (w - 1) / 2) * 0.55) for layer, w in enumerate(WIDTHS) for i in range(w)}


def _network(ax, colour_node):
    pos = _pos()
    for layer in range(3):
        for i in range(WIDTHS[layer]):
            for j in range(WIDTHS[layer + 1]):
                (x0, y0), (x1, y1) = pos[(layer, i)], pos[(layer + 1, j)]
                ax.plot([x0, x1], [y0, y1], color="0.87", lw=0.35, zorder=1)
    for node, (x, y) in pos.items():
        if node not in PATH:
            fc, ec = colour_node(node)
            ax.scatter(x, y, s=30, facecolor=fc, edgecolor=ec, lw=0.7, zorder=3)
    ax.text(0, -1.45, "input", ha="center", va="top", color="0.45")
    ax.text(3, -0.9, "output", ha="center", va="top", color="0.45")


def _attacked(node):
    """(fill, edge) of a node after the single-neuron attack: the tampered node red, every later node recomputed
    honestly (light red), the others untouched."""
    if node == TAMPERED:
        return ATTACK, ATTACK
    return (ATTACK_LIGHT, "0.35") if node[0] > TAMPERED[0] else ("white", "0.35")


def _finish(ax):
    ax.set_xlim(-0.35, 3.35)
    ax.set_ylim(-2.02, 2.95)
    ax.axis("off")
    ax.figure.subplots_adjust(left=0, right=1, bottom=0, top=1)


def overview(out: Path) -> list[Path]:
    """``fig:overview``: (a) one tampered neuron that a random path misses; (b) the same network with every layer
    checked, the box of the tampered layer the check that fails."""
    from matplotlib.patches import FancyBboxPatch
    plt, pos = _plt(), _pos()
    fig, ax = plt.subplots(figsize=OVERVIEW)
    _network(ax, _attacked)
    for a, b in zip(PATH[:-1], PATH[1:]):
        (x0, y0), (x1, y1) = pos[a], pos[b]
        ax.plot([x0, x1], [y0, y1], color=ANCH, lw=1.6, zorder=2)
    for n in PATH:
        ax.scatter(*pos[n], s=30, facecolor=_attacked(n)[0], edgecolor=ANCH, lw=1.1, zorder=4)
    ax.annotate("wrong value", xy=pos[TAMPERED], xytext=(0.2, 2.25), color=ATTACK,
                arrowprops=dict(arrowstyle="->", color=ATTACK, lw=0.6, shrinkB=3))
    ax.text(2.55, 2.25, "recomputed\nhonestly", color="0.35", ha="center", va="center", linespacing=1.0)
    _finish(ax)
    paths = [save(fig, out / "overview_attack.pdf", OVERVIEW[0])]
    fig, ax = plt.subplots(figsize=OVERVIEW)
    _network(ax, _attacked)
    for n in PATH:
        fc, ec = _attacked(n)
        ax.scatter(*pos[n], s=30, facecolor=fc, edgecolor=ec, lw=0.7, zorder=3)
    for layer in range(1, 4):
        h = (WIDTHS[layer] - 1) * 0.55 / 2 + 0.28
        failing = layer == TAMPERED[0]
        ax.add_patch(FancyBboxPatch((layer - 0.2, -h), 0.4, 2 * h, boxstyle="round,pad=0.02", fc="none",
                                    ec=ATTACK if failing else OURS, lw=1.4 if failing else 1.0, zorder=2))
    ax.text(1.5, 2.6, "one random combination per layer", ha="center", va="center", color=OURS)
    _finish(ax)
    return paths + [save(fig, out / "overview_defence.pdf", OVERVIEW[0])]


def protocol(out: Path) -> Path:
    """``fig:protocol``: the messages of one query in mode C, numbered."""
    from matplotlib.patches import Rectangle
    plt = _plt()
    w, h = 3.33, 1.82
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)                      # inches from the top
    ax.axis("off")
    xp, xv = 0.62, 1.88                    # the prover's and the verifier's lifelines
    ax.text(xp, 0.07, "prover (provider)", ha="center", va="center", weight="bold")
    ax.text(xv, 0.07, "verifier (client)", ha="center", va="center", weight="bold")
    ax.add_patch(Rectangle((0.03, 0.17), w - 0.06, 0.19, fc="0.91", ec="none"))
    ax.text(w / 2, 0.265, r"setup (once): owner publishes $C_M$ = Merkle roots of $\mathrm{Enc}(A_\ell)$",
            ha="center", va="center", color="0.2")
    for x in (xp, xv):
        ax.plot([x, x], [0.42, 1.80], color="0.6", lw=0.8)
    steps = [  # (y, direction, message, the verifier's note)
        (0.53, "vp", r"query $x$", None),
        (0.74, "pv", r"$y$, claims $Z_\ell$", "range-check the claims, recompute\n"
                                               r"non-weight ops; draw $\chi$"),
        (1.00, "vp", r"random $\chi$", None),
        (1.21, "pv", r"$u_\ell = \chi^\top\! A_\ell$", r"check $\chi^\top Z_\ell = u_\ell\,[X_\ell; 1]$;"
                                                         "\n" r"draw positions $c$"),
        (1.45, "vp", r"positions $c$", None),
        (1.66, "pv", r"columns $\hat A_{\ell,c}$ + Merkle proof", "check the Merkle paths and\n"
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
    # messages (3)-(6) are sent in mode C only (in K and Kpre the verifier computes u itself, so the proof is
    # message (2)); the bracket sits on the message side, clear of the verifier's notes
    xb, y0, y1 = 0.30, steps[2][0] - 0.1, steps[5][0] + 0.08
    ax.plot([xb + 0.04, xb, xb, xb + 0.04], [y0, y0, y1, y1], color="0.35", lw=0.6)
    ax.text(xb - 0.06, (y0 + y1) / 2, f"sent in {SF_C} only", rotation=90, ha="center", va="center", color="0.35")
    return save(fig, out / "protocol.pdf", w)


def schematics(out: Path) -> list[Path]:
    """The schematic figures (``fig:overview``, ``fig:protocol``)."""
    return overview(out) + [protocol(out)]


def breakeven(crossovers: dict, out: Path) -> Path:
    """``fig:breakeven`` from :func:`experiments.paper.macros.breakeven`'s crossovers of the run (the PDF dated at
    the epoch, ``SOURCE_DATE_EPOCH=0``, so that it carries neither the time nor the time zone of its drawing)."""
    import os
    from experiments.deployment import ev_breakeven as ev
    path = out / "ev_breakeven.pdf"
    before = os.environ.get("SOURCE_DATE_EPOCH")
    os.environ["SOURCE_DATE_EPOCH"] = "0"
    try:
        ev.draw(crossovers, {"reexec_s": {"2048": {"exact": crossovers["reexec_s"]}}}, [str(path)])
    finally:
        if before is None:
            del os.environ["SOURCE_DATE_EPOCH"]
        else:
            os.environ["SOURCE_DATE_EPOCH"] = before
    return path

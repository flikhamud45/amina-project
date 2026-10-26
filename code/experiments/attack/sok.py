"""The acceptance probabilities of ``tab:sok`` and ``tab:sok-derivation`` (Sec. ``sec:sok``; ``ext_attack.tex``), and
the path counts of ``tab:paths`` and ``tab:paths-acc`` (``app:paths``), from each scheme's own formulas and parameters.

An edit followed by honest recomputation is caught exactly when the sample covers it, so each scheme's acceptance
probability follows from its sampling alone (``tab:sok-derivation``):

* the path protocol of Anchuri et al.: one path accepts with ``1 - 1/N`` and ``s`` paths with ``(1 - 1/N)^s``
  (``N = 512``, ``s = 1`` and ``250``); ``2^-40`` needs ``s_40(1, N) = ceil(40 ln 2 / -ln(1 - 1/N))`` paths, for
  LeNet-5 (``N = 84``), VGG and ResNet (``N = 512``), Llama-2-7B (``N = 11,008``) and OPT-6.7B (``N = 16,384``);
  with ``k`` edited neurons in a layer of width ``N`` (Proposition ``prop:paths``) ``s_lambda(k, N) = ceil(lambda ln 2
  / -ln(1 - k/N))`` and ``s`` paths accept with ``(1 - k/N)^s``;
* SLP's 70B configuration (3 of its 161 non-anchor chunks proved per audit): one audit accepts with ``1 - 3/161``,
  and detection with probability ``P`` needs ``ceil(ln(1 - P) / ln(158/161))`` audits (``P = 50%`` and ``95%``);
  its TinyLlama configuration (5 of 45 chunks): ``1 - 5/45``;
* CommitLLM's routine audit (10 of 80 layers at one token): ``1 - 10/80`` when the token is opened, and
  ``1 - 0.05 * 10/80`` per one-token response at a 5% audit rate;
* NanoZK's audit mode with a uniform 50% budget: ``1 - |S|/L = 50%``.

:func:`values` gives each value with the decimals the paper prints; ``experiments/paper/macros.py`` (family ``sok``)
writes them as the macros ``\\Sok...`` at every build of the paper's numbers, and ``experiments/paper/tables.py``
builds ``tab:paths`` and ``tab:paths-acc`` with :func:`paths_needed` and :func:`accept_log10` for the edits that the
run's in-range attack measured.  Nothing is measured: the parameters are the schemes' stated ones (:data:`PATH_N`,
:data:`SLP_70B`, ...).

    python -m experiments.attack.sok                  # the derivation table: setting, formula, value
    python -m experiments.attack.sok --json           # one JSON line per value (the macro name, value, printed form)

Pure arithmetic: no GPU, no model, well under a second and a few MB of memory.
"""
from __future__ import annotations

import argparse
import json
import math

LAM = 40
"""The path protocol's soundness target ``2^-40`` (Sec. ``sec:dichotomy``)."""
PATH_N, PATH_S = 512, (1, 250)
"""The path protocol's MNIST setting of ``tab:sok``: a layer of 512 neurons, one path and 250 paths."""
PATH_WIDTHS = (("Lenet", "LeNet-5", 84), ("Wide", "VGG, ResNet", 512), ("Llama", "Llama-2-7B", 11_008),
               ("OptL", "OPT-6.7B", 16_384))
"""The widths whose ``2^-40`` path counts ``tab:sok-derivation`` lists: the penultimate layer of LeNet-5 and of the
VGG and ResNet models, and the FFN layers of Llama-2-7B and OPT-6.7B."""
SLP_70B, SLP_TINY = (3, 161), (5, 45)
"""SLP's proved chunks per audit and non-anchor chunks: its 70B and TinyLlama configurations."""
SLP_DETECT = (0.5, 0.95)
"""The detection probabilities whose audit counts ``tab:sok`` states."""
COMMITLLM_LAYERS, COMMITLLM_RATE = (10, 80), 0.05
"""CommitLLM's routine audit: layers checked of all layers at the opened token, and the audit rate."""
NANOZK_BUDGET = 0.5
"""NanoZK's audit mode with a uniform budget of half of the layers."""


def paths_needed(lam: float, k: float, n: int) -> int:
    """``s_lambda(k, N)``: the least number of uniform paths that catch an edit of ``k`` sites in a layer of width
    ``n`` with probability at least ``1 - 2^-lam`` (Proposition ``prop:paths``), computed as
    :func:`pvi.baseline.path_int.paths_for` computes it for ``k = 1``."""
    return math.ceil(lam / -math.log2(1.0 - k / n))


def accept_log10(k: float, n: int, s: float) -> float:
    """``log10`` of ``(1 - k/N)^s``, the probability that ``s`` uniform paths accept an edit of ``k`` sites in a layer
    of width ``n`` (Proposition ``prop:paths``); in the log domain, which never underflows."""
    return s * math.log1p(-k / n) / math.log(10)


def audits_for(detect: float, proved: int, chunks: int) -> int:
    """The least number of SLP audits, each proving ``proved`` of ``chunks`` chunks, that detect an edited chunk with
    probability at least ``detect``: ``ceil(ln(1 - P) / ln(1 - proved/chunks))``."""
    return math.ceil(math.log(1 - detect) / math.log(1 - proved / chunks))


def values() -> list[dict]:
    """Every value of ``tab:sok-derivation``, in its order: ``{name, setting, formula, value, printed}`` (``name``:
    the macro without ``Sok``; ``printed``: as the paper prints it, a percentage without its sign or a count)."""
    rows = []

    def put(name, setting, formula, value, printed):
        rows.append({"name": name, "setting": setting, "formula": formula, "value": value, "printed": printed})

    for s, name in zip(PATH_S, ("PathOne", "PathMany")):
        p = (1 - 1 / PATH_N) ** s
        put(name, f"Path, N={PATH_N}, s={s}", "(1-1/N)^s", 100 * p, f"{100 * p:.2f}" if s == 1 else f"{100 * p:.1f}")
    for name, label, n in PATH_WIDTHS:
        v = paths_needed(LAM, 1, n)
        put(f"Paths{name}", f"Path, 2^-{LAM}, {label} (N={n:,})", "ceil(40 ln2 / -ln(1-1/N))", v, f"{v:,}")
    proved, chunks = SLP_70B
    p = 1 - proved / chunks
    put("SlpAudit", f"SLP 70B, {proved} of {chunks} chunks", f"1-{proved}/{chunks}", 100 * p, f"{100 * p:.2f}")
    for detect, name in zip(SLP_DETECT, ("SlpAuditsHalf", "SlpAuditsMost")):
        v = audits_for(detect, proved, chunks)
        put(name, f"SLP 70B, audits for {100 * detect:g}%", f"ceil(ln(1-P)/ln({chunks - proved}/{chunks}))", v,
            f"{v:,}")
    proved, chunks = SLP_TINY
    p = 1 - proved / chunks
    put("SlpTiny", f"SLP TinyLlama, {proved} of {chunks} chunks", f"1-{proved}/{chunks}", 100 * p, f"{100 * p:.1f}")
    checked, layers = COMMITLLM_LAYERS
    p = 1 - checked / layers
    put("CommitOpened", "CommitLLM, token opened", f"1-{checked}/{layers}", 100 * p, f"{100 * p:.1f}")
    p = 1 - COMMITLLM_RATE * checked / layers
    put("CommitRate", f"CommitLLM, {100 * COMMITLLM_RATE:g}% audit rate", f"1-{COMMITLLM_RATE}*{checked}/{layers}",
        100 * p, f"{100 * p:.3f}")
    put("CommitRateAbout", f"CommitLLM, {100 * COMMITLLM_RATE:g}% audit rate (rounded)", "the same", 100 * p,
        f"{100 * p:.1f}")
    p = 1 - NANOZK_BUDGET
    put("NanoAudit", f"NanoZK audit, uniform {100 * NANOZK_BUDGET:g}%", "1-|S|/L", 100 * p, f"{100 * p:.0f}")
    return rows


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--json", action="store_true", help="one JSON line per value instead of the table")
    return ap


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    for row in values():
        if args.json:
            print(json.dumps(dict(row, name="Sok" + row["name"])))
        else:
            print(f"{row['setting']:<40} {row['formula']:<34} {row['printed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

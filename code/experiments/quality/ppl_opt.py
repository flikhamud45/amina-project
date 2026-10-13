"""Perplexity of the integer OPT models on WikiText-2 (raw): the OPT rows of ``tab:quality``, the cost of 8-bit
attention probabilities and the probability mass a rounded row keeps (Sec. ``sec:quality``), and the percentile recipe,
the recipe search, the per-row weight scales and cross-layer equalisation of the extended version
(Sec. ``sec:ext-quality``).  Every graph is ``pvi.models.hf_opt.build_opt_from_hf``'s, so the certified cells are the
certified models themselves (their digests are those of ``digest_check.py``).

    python -m experiments.quality.ppl_opt --model facebook/opt-1.3b --device cuda --out ppl_opt-1.3b.jsonl
    python -m experiments.quality.ppl_opt --model facebook/opt-6.7b --device cuda --ref-device cpu \\
        --ref-cache ref_opt-6.7b.pt --out ppl_opt-6.7b.jsonl
    python -m experiments.quality.ppl_opt --model facebook/opt-125m --check
    python -m experiments.quality.ppl_opt --model facebook/opt-1.3b --device cuda --split validation \\
        --graphs smooth --alphas 0.5 0.6 0.7 --pbits 11
    python -m experiments.quality.ppl_opt --model facebook/opt-1.3b --device cuda --recipe legacy \\
        --graphs smooth rows --draws 0 1 2
    python -m experiments.quality.ppl_opt --model facebook/opt-125m --calibration test-end --seq 128 --windows 16 \\
        --recipe legacy --graphs scalar smooth --gains 4 3 2 --alphas 0.55 0.5 --pbits 8

* Text: WikiText-2 (raw) ``--split`` -- ``test`` (the reported numbers) or ``validation`` (to check the recipe's
  alpha on text it was not chosen on) -- joined with blank lines (``"\\n\\n".join``, the convention of the published
  2,048-token OPT perplexities: 27.65, 14.62 and 10.86 for OPT-125M, OPT-1.3B and OPT-6.7B), tokenised once.
* Windows: every non-overlapping ``--seq``-token window of the split (``--windows`` limits them; 140 test windows at
  2,048 tokens), each scored on its own, all positions in one pass (``hf_opt.every_position``: what one query per
  prefix gives, since every op is per position but causal attention and no constant depends on the input).
* Calibration (``--calibration train``, the certified recipe's): draw ``d`` is ``--calib-windows`` consecutive
  ``--seq``-token windows of the train split's first 4,000,000 characters from window ``d * --calib-windows`` on
  (``checkpoints.certified_calibration``); no evaluated token is seen.  ``--calibration test-end`` is the setting of
  the first smoothing study (extended version): ``--windows`` windows from the start of the test split joined
  without a separator and one ``--seq``-token calibration window at its end, disjoint from them
  (``checkpoints.load_windows``; ``--draws`` does not apply).
* Graphs (``--graphs``), each at every ``--pbits`` (8: probabilities 0..255; 11: 0..2047, the certified models'):
  ``scalar``, one norm gain (``--gain``/``--gains``; default per model 4 / 3 / 2 for OPT-125M / 1.3B / 6.7B, the
  best of the scalar-gain search: ``tab:quality``'s "Scalar gain"); ``smooth``, SmoothQuant through the norm gains
  with ``--recipe``'s statistics, ``certified`` (``hf_opt.CERTIFIED_RECIPE``: channel maxima, no clipping, alpha
  0.6; ``tab:quality``'s "Certified") or ``legacy`` (the first recipe: the 99.9th percentile of each channel,
  99.99th-percentile clipping, alpha 0.55), ``--alphas`` replacing the recipe's alpha (the recipe search);
  ``rows``, ``smooth`` with per-row power-of-two weight scales (``--row-buckets`` per matrix; protocol v0 only).
  ``--cle-mlp`` adds cross-layer equalisation between fc1 and fc2 to the smoothed graphs.  ``--scalar-opts`` and
  ``--smooth-opts`` (JSON) override keyword arguments of ``build_opt_from_hf``.
* Probability mass: on the first ``--mass-windows`` windows, ``hf_opt.probability_mass`` records per block the mean
  share of the probability mass a rounded row keeps (``sec:quality``: as little as 56% at 8 bits, 97.5% at 11).
* Lean: the integer forward frees each tensor after its last reader and keeps no claim; the LM head's logits are
  reduced to NLL and argmax in chunks.  The calibration runs on the CPU, the integer forward on ``--device``, the
  fp32 reference on ``--ref-device``.
* ``--check``: the evaluation path against the library on one ``--seq``-token window (OPT-125M; see :func:`check`;
  ``--seq 256 --calib-windows 2`` is a quicker variant of the full-window default).

Output: one JSON line per cell on stdout (and appended to ``--out``): ``cell`` (``fp32`` for the reference, else the
graph, its gain or recipe and alpha, and ``-p<bits>``), ``graph``, ``recipe``, ``gain``, ``alpha``, ``pbits``,
``draw``, ``opts`` (the builder's keyword arguments), ``ppl``, ``fp32``, ``ratio`` with ``ratio_ci95`` (a 95%
bootstrap interval over windows), ``top1`` (agreement with fp32), ``mass_per_layer``/``mass_min``, ``nll`` per window
(to pool calibration draws), ``digest`` and ``weights_sha256`` of the graph, ``build_s``, ``seconds``, the peak memory
(``peaks``, ``peak_rss_GB``, ``peak_gpu_GB``) and ``env`` (``records.environment``); a short summary per cell on
stderr.

Hardware and time (RTX 2080 Ti, 11 GB, for the integer forward; 140 windows of 2,048 tokens, 4 x 2,048 calibration
tokens): per cell, OPT-125M builds in 0.5-2 min and evaluates in under a minute (6 GB host); OPT-1.3B builds in 4-15
min and evaluates in 1.5-4.5 min (14 GB host, 3 GB GPU); OPT-6.7B builds in 15-35 min and evaluates in 4-8 min (44 GB
host, 8 GB GPU), and its fp32 reference takes about 2 h on the CPU (``--ref-device cpu``; ``--ref-cache`` keeps it).
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

from experiments.common import checkpoints, records
from experiments.common.cli import add_runtime_arguments, apply_runtime
from experiments.common.stats import ppl_ratio_ci
from pvi.models import hf_opt

BEST_GAIN = {"opt-125m": 4.0, "opt-1.3b": 3.0, "opt-6.7b": 2.0}
"""The scalar graphs' norm gain per model: the best of the scalar-gain search (``tab:quality``'s "Scalar gain")."""
RECIPES = checkpoints.OPT_RECIPES
"""The smoothed graphs' recipes: the certified models' and the first one (alpha 0.55 with percentile statistics)."""


# ------------------------------------------------------------------ data
def load_data(tok, args) -> tuple[list[torch.Tensor], dict[int, torch.Tensor], str]:
    """``(evaluation windows [1, seq], {draw: calibration ids [n, seq]}, dataset label)`` of ``args``."""
    if args.calibration == "test-end":
        if args.split != "test" or args.windows <= 0:
            raise SystemExit("--calibration test-end scores --windows (> 0) windows of the test split")
        wins, calib = checkpoints.load_windows(tok, seq=args.seq, n=args.windows, calib_windows=1)
        return wins, {0: calib[0]}, "wikitext-2-raw-v1 test, ''.join"
    ids = tok(checkpoints.wikitext(args.split), return_tensors="pt").input_ids[0]
    calibs = {d: checkpoints.certified_calibration(tok, n=args.calib_windows, seq=args.seq, draw=d) for d in args.draws}
    return checkpoints.windows(ids, args.seq, args.windows), calibs, f"wikitext-2-raw-v1 {args.split}, '\\n\\n'.join"


def cells(args) -> list[dict]:
    """The integer graphs of a run, in order: ``{"cell", "graph", "recipe", "gain", "alpha", "pbits", "kw"}`` with
    ``kw`` the keyword arguments of ``build_opt_from_hf``."""
    gains = args.gains or [args.gain or BEST_GAIN.get(args.model.split("/")[-1], 4.0)]
    recipe = RECIPES[args.recipe]
    variants = []                                   # (label, graph, gain, alpha, kw)
    for g in args.graphs:
        if g == "scalar":
            extra = json.loads(args.scalar_opts)
            variants += [(f"scalar-g{x:g}", g, x, None, {"norm_gain": x, **extra}) for x in gains]
            continue
        for a in args.alphas or [recipe["smooth"]]:
            kw = {**recipe, "smooth": a, **json.loads(args.smooth_opts)}
            label = f"smooth-{args.recipe}-a{a:g}"
            if g == "rows":
                kw["row_pow2"] = args.row_buckets
                label = f"rows{args.row_buckets}-{args.recipe}-a{a:g}"
            if args.cle_mlp is not None:
                kw.update(cle_mlp=args.cle_mlp, cle_ratio=args.cle_ratio)
                label += f"-cle{args.cle_mlp:g}r{args.cle_ratio:g}"
            variants.append((label, g, None, a, kw))
    return [{"cell": f"{label}-p{pb}", "graph": g, "recipe": None if g == "scalar" else args.recipe, "gain": gain,
             "alpha": alpha, "pbits": pb, "kw": dict(kw, prob_max=(1 << pb) - 1)}
            for label, g, gain, alpha, kw in variants for pb in args.pbits]


# ------------------------------------------------------------------ evaluation
def nll_argmax(logits: torch.Tensor, scale: float, ids: torch.Tensor, chunk: int = 256):
    """Summed NLL and the argmaxes of ``logits [1, T, V]`` (integers times ``scale``) against ``ids [1, T]``."""
    lg = logits.reshape(-1, logits.shape[-1])[:-1]
    tgt = ids[0, 1:].to(lg.device)
    nll, am = 0.0, []
    for a in range(0, lg.shape[0], chunk):
        part = lg[a:a + chunk].double() * scale
        nll += float(F.cross_entropy(part, tgt[a:a + chunk], reduction="sum"))
        am.append(part.argmax(-1).cpu())
    return nll, torch.cat(am)


def evaluate(lean, scale: float, w: torch.Tensor, device: str, mass: bool = False):
    """``(NLL, argmaxes, per-block mass or None)`` of the all-positions graph ``lean`` on the window ``w``: with
    ``mass`` through ``hf_opt.probability_mass`` (the same integers), else the lean forward (no claim kept)."""
    x = w.to(device)
    if mass:
        out, m = hf_opt.probability_mass(lean, x)
    else:
        env, _ = lean.forward(x, free=True, send=lambda z: None)
        out, m = env.pop(lean.output_name), None
    nll, am = nll_argmax(out, scale, w)
    return nll, am, m


def reference(model, wins, device: str, cache: Path | None) -> dict:
    """The fp32 checkpoint's per-window NLL and argmaxes (from ``cache`` when it holds as many windows)."""
    if cache is not None and cache.exists():
        ref = torch.load(cache, weights_only=True)
        if ref["windows"] == len(wins):
            return ref
    t0 = time.perf_counter()
    m = model.to(device)
    nll, am = [], []
    with torch.no_grad():
        for w in wins:
            x, a = nll_argmax(m(w.to(device)).logits, 1.0, w)
            nll.append(x)
            am.append(a)
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ref = {"windows": len(wins), "nll": nll, "argmax": torch.stack(am), "seconds": time.perf_counter() - t0}
    if cache is not None:
        torch.save(ref, cache)
    return ref


def _emit(args, row: dict) -> None:
    records.emit(row, args.out)
    short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
             if k in ("cell", "draw", "ppl", "ratio", "ratio_ci95", "top1", "mass_min", "seconds")}
    print(json.dumps(short), file=sys.stderr, flush=True)


def _gb(v):
    return None if v is None else v / 2 ** 30


def run(model, tok, args) -> None:
    wins, calibs, dataset = load_data(tok, args)
    counts = [w.shape[1] - 1 for w in wins]
    ref = reference(model, wins, args.ref_device or args.device, Path(args.ref_cache) if args.ref_cache else None)
    fp32 = math.exp(sum(ref["nll"]) / sum(counts))
    env = records.environment(args.device)
    head = {"model": args.model, "tag": args.tag, "seq": args.seq, "windows": len(wins), "tokens": sum(counts),
            "dataset": dataset, "split": args.split, "calibration": args.calibration,
            "calib_windows": args.calib_windows if args.calibration == "train" else 1, "device": args.device,
            "host": env["host"], "gpu": env["gpu"], "env": env}
    _emit(args, dict(head, cell="fp32", ppl=fp32, nll=ref["nll"], seconds=ref["seconds"]))
    for draw, calib in calibs.items():
        for cell in cells(args):
            records.reset_peaks(args.device)
            t0 = time.perf_counter()
            with torch.no_grad():
                graph, info = hf_opt.build_opt_from_hf(model, calib, **cell["kw"])
            build_s = time.perf_counter() - t0
            digest, wsha = graph.digest().hex(), checkpoints.weights_sha(graph)
            lean, _ = hf_opt.every_position(graph).with_constants_on(args.device)
            t0 = time.perf_counter()
            nll, agree, mass = [], 0, []
            with torch.no_grad():
                for i, (w, r_am) in enumerate(zip(wins, ref["argmax"])):
                    x, am, m = evaluate(lean, info["logit_scale"], w, args.device, mass=i < args.mass_windows)
                    if m is not None:
                        mass.append(m)
                    nll.append(x)
                    agree += int((am == r_am).sum())
            ppl = math.exp(sum(nll) / sum(counts))
            layers = [sum(col) / len(col) for col in zip(*mass)] if mass else None
            peaks = records.peaks(args.device)
            _emit(args, dict(head, cell=cell["cell"], graph=cell["graph"], recipe=cell["recipe"], gain=cell["gain"],
                             alpha=cell["alpha"], pbits=cell["pbits"], draw=draw, opts=cell["kw"], ppl=ppl, fp32=fp32,
                             ratio=ppl / fp32, ratio_ci95=ppl_ratio_ci(nll, ref["nll"], counts),
                             top1=agree / sum(counts), mass_per_layer=layers, mass_min=min(layers) if layers else None,
                             nll=nll, digest=digest, weights_sha256=wsha, build_s=build_s,
                             seconds=time.perf_counter() - t0, peaks=peaks, peak_rss_GB=_gb(peaks["host_peak_rss"]),
                             peak_gpu_GB=_gb(peaks.get("gpu_peak_memory"))))
            del graph, lean, info
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ------------------------------------------------------------------ check
def check(model, w: torch.Tensor, calib: torch.Tensor, gain: float = 4.0) -> dict:
    """The evaluation path against the library on the window ``w [1, T]`` with the calibration ``calib``, for the
    scalar graph (``gain``), the certified and the legacy graph at 8 and 11 bits and the per-row graph at 11 bits:
    (a) the lean path's NLL and argmaxes are those of ``hf_opt.real_logits(all_positions=True)``; (b) the mass path
    (``hf_opt.probability_mass``) gives the same NLL and argmaxes, and a mass per block in ``(0, 1.01]``; (c) on a
    GPU, the lean graph gives the CPU's integers.  Returns the report (perplexity, smallest mass and digest per graph);
    ``report["ok"]`` says whether every check holds."""
    graphs = [(f"scalar-g{gain:g}", dict(norm_gain=gain))] + [
        (f"{r}", dict(RECIPES[r])) for r in ("certified", "legacy")]
    todo = [(f"{name}-p{pb}", dict(kw, prob_max=(1 << pb) - 1)) for name, kw in graphs for pb in (8, 11)]
    todo.append(("rows4-legacy-p11", dict(RECIPES["legacy"], row_pow2=4)))
    report, ok = {"seq": w.shape[1], "calib": list(calib.shape)}, True
    for name, kw in todo:
        with torch.no_grad():
            graph, info = hf_opt.build_opt_from_hf(model, calib, **kw)
            want = nll_argmax(hf_opt.real_logits(graph, info, w, all_positions=True), 1.0, w)
            lean = hf_opt.every_position(graph)
            lean_nll, lean_am, _ = evaluate(lean, info["logit_scale"], w, "cpu")
            mass_nll, mass_am, mass = evaluate(lean, info["logit_scale"], w, "cpu", mass=True)
            gpu = None
            if torch.cuda.is_available():
                env, _ = lean.forward(w, free=True, send=lambda z: None)
                dev, _ = lean.with_constants_on("cuda")
                env_gpu, _ = dev.forward(w.cuda(), free=True, send=lambda z: None)
                gpu = bool(torch.equal(env_gpu[lean.output_name].cpu(), env[lean.output_name]))
        row = {"lean_equals_real_logits": lean_nll == want[0] and bool(torch.equal(lean_am, want[1])),
               "mass_path_equals": mass_nll == lean_nll and bool(torch.equal(mass_am, lean_am)),
               "mass_in_range": all(0 < m <= 1.01 for m in mass), "gpu_equals_cpu": gpu,
               "ppl": math.exp(lean_nll / (w.shape[1] - 1)), "mass_min": min(mass), "digest": graph.digest().hex()}
        ok &= all(v for k, v in row.items() if k in ("lean_equals_real_logits", "mass_path_equals", "mass_in_range",
                                                       "gpu_equals_cpu") and v is not None)
        report[name] = row
    report["ok"] = ok
    return report


# ------------------------------------------------------------------ main
def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="facebook/opt-1.3b")
    ap.add_argument("--split", choices=["test", "validation"], default="test",
                    help="the evaluated split (validation: re-check a recipe's alpha on text it was not chosen on)")
    ap.add_argument("--seq", type=int, default=2048, help="tokens per window (and per calibration window)")
    ap.add_argument("--windows", type=int, default=0, help="evaluated windows (0: every window of the split)")
    ap.add_argument("--calibration", choices=["train", "test-end"], default="train",
                    help="train: the certified recipe's calibration draws; test-end: the first smoothing study's "
                         "setting (one window at the test split's end, evaluation windows from its start)")
    ap.add_argument("--calib-windows", type=int, default=4, help="calibration windows per draw (train calibration)")
    ap.add_argument("--draws", type=int, nargs="+", default=[0], help="calibration draws (train calibration)")
    ap.add_argument("--graphs", nargs="+", choices=["scalar", "smooth", "rows"], default=["scalar", "smooth"])
    ap.add_argument("--recipe", choices=sorted(RECIPES), default="certified", help="the smoothed graphs' recipe")
    ap.add_argument("--alphas", type=float, nargs="+", default=None,
                    help="SmoothQuant alphas of the smoothed graphs (default: the recipe's, 0.6 / 0.55)")
    ap.add_argument("--gain", type=float, default=None, help="the scalar graph's gain (default: the model's best)")
    ap.add_argument("--gains", type=float, nargs="+", default=None, help="several scalar gains (a gain search)")
    ap.add_argument("--row-buckets", type=int, default=4,
                    help="the rows graph's power-of-two weight scales per matrix (build_opt_from_hf(row_pow2=...))")
    ap.add_argument("--cle-mlp", type=float, default=None,
                    help="cross-layer equalisation between fc1 and fc2 with this alpha on the smoothed graphs")
    ap.add_argument("--cle-ratio", type=float, default=32.0, help="the equalisation's clamp around its median")
    ap.add_argument("--smooth-opts", default="{}", help="JSON keyword arguments of build_opt_from_hf replacing the "
                    "recipe's in the smooth and rows graphs (e.g. pct, pct_att, smooth_pct)")
    ap.add_argument("--scalar-opts", default="{}", help="JSON keyword arguments of build_opt_from_hf added to the "
                    "scalar graphs (e.g. pct)")
    ap.add_argument("--pbits", type=int, nargs="+", default=[8, 11],
                    help="attention probability bits (8: 0..255; 11: 0..2047, the certified models')")
    ap.add_argument("--mass-windows", type=int, default=2, help="windows on which the probability mass is recorded")
    ap.add_argument("--ref-device", default=None, help="where the fp32 reference runs (default: --device)")
    ap.add_argument("--ref-cache", default=None, help="a file keeping the fp32 reference between runs")
    ap.add_argument("--tag", default="", help="a label written into every record of this run")
    ap.add_argument("--out", default=None, help="a JSONL file the records are appended to")
    ap.add_argument("--check", action="store_true",
                    help="check the evaluation path on one --seq-token window of the split (calibration: draw 0 of "
                         "--calib-windows train windows), then stop")
    add_runtime_arguments(ap, threads=8, verifier_device=False)
    args = ap.parse_args()
    apply_runtime(args)
    model, tok = checkpoints.load_opt(args.model)
    if args.check:
        w = checkpoints.windows(tok(checkpoints.wikitext(args.split), return_tensors="pt").input_ids[0], args.seq, 1)[0]
        calib = checkpoints.certified_calibration(tok, n=args.calib_windows, seq=args.seq)
        report = check(model, w, calib, args.gain or BEST_GAIN.get(args.model.split("/")[-1], 4.0))
        records.emit(dict(report, model=args.model, env=records.environment("cpu")))
        print("CHECK", "PASSED" if report["ok"] else "FAILED", flush=True)
        raise SystemExit(0 if report["ok"] else 1)
    run(model, tok, args)


if __name__ == "__main__":
    main()

"""The integer OPT's WikiText-2 perplexity with and without SmoothQuant through the norm gains (plan F2).

    cd code && PYTHONPATH=src python experiments/9_quality/ppl_smooth.py --model facebook/opt-1.3b --seq 128 --windows 16

Same windows as ``experiments/2_attack/real_weights_ppl.py`` (WikiText-2 raw test, non-overlapping windows of
``--seq`` tokens from the start; calibration on the last window of the split). Rows: the fp32 checkpoint, the
graph with one scalar norm gain per model (``--gains``, the best of the earlier sweep is 4 for OPT-125M, 3 for
OPT-1.3B, 2 for OPT-6.7B), and the smoothed graph (``--alphas``, 0.55 measured best, with smooth_pct 99.9 and
99.99th-percentile clipping). One JSON line with every row: perplexity, top-1 agreement with fp32, seconds.
"""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Scorer, load_opt, load_windows          # noqa: E402

from pvi.fullcheck.real_weights import build_opt_from_hf, real_logits   # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--gains", type=float, nargs="*", default=[4.0, 3.0, 2.0])
    ap.add_argument("--alphas", type=float, nargs="*", default=[0.55, 0.5])
    ap.add_argument("--device", default="cpu", help="where the integer graph runs (the fp32 model stays on the CPU)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, tok = load_opt(args.model)
    wins, calib = load_windows(tok, seq=args.seq, windows=args.windows, calib_windows=1)
    sc = Scorer(wins)
    t0 = time.perf_counter()
    with torch.no_grad():
        fp32, _, am = sc.score(lambda w: model(w).logits)
    sc.ref_argmax = am
    rows = [{"variant": "fp32", "ppl": fp32, "top1": 1.0, "seconds": time.perf_counter() - t0}]
    variants = [("gain", dict(norm_gain=g)) for g in args.gains]
    variants += [("smooth", dict(norm_gain=4, smooth=a, smooth_pct=99.9, pct=99.99, pct_att=99.99)) for a in args.alphas]
    for kind, kw in variants:
        t0 = time.perf_counter()
        graph, info = build_opt_from_hf(model, calib[0].to(args.device), device=args.device, **kw)
        ppl, agree, _ = sc.score(lambda w: real_logits(graph, info, w.to(args.device), all_positions=True).cpu())
        rows.append({"variant": kind, **{k: v for k, v in kw.items()}, "ppl": ppl, "top1": agree,
                     "ratio": ppl / fp32, "seconds": time.perf_counter() - t0})
        del graph
        print(json.dumps(rows[-1]), file=sys.stderr, flush=True)
    print(json.dumps({"model": args.model, "dataset": "wikitext-2-raw-v1 test", "seq": args.seq,
                      "windows": args.windows, "rows": rows, "host": platform.node(),
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}), flush=True)


if __name__ == "__main__":
    main()

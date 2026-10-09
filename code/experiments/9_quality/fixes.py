"""Candidate fixes for the integer OPT's perplexity (F2), measured on the exact integer graph.

Every row builds ``opt_builder.build_opt`` with some options and scores the integer graph itself
(``int_logits``: the integers the verifier recomputes), on the windows of ``real_weights_ppl.py``.  Each row
also records the constraint report (claim bound, multiplier ranges, vector multipliers) and the difference
list against the benchmark graph (the LM head bias only).

    PYTHONPATH=src python experiments/9_quality/fixes.py --model facebook/opt-125m --sweep smooth
    PYTHONPATH=src python experiments/9_quality/fixes.py --configs '[{"norm_mode": "smooth", "alpha": 0.5}]'
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Scorer, load_opt, load_windows  # noqa: E402
from opt_builder import build_opt, graph_report, hidden_max, int_logits  # noqa: E402

from pvi.fullcheck.real_weights import matches_benchmark_graph  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def sweeps() -> dict[str, list[dict]]:
    sm = dict(norm_mode="smooth")
    best = dict(norm_mode="smooth", alpha=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)   # from the sweeps
    return {
        "scalar": [dict(norm_gain=g, pct=p) for p in (100.0, 99.9) for g in (3, 4, 6)],
        "smooth": [dict(sm, alpha=a) for a in (0.0, 0.25, 0.5, 0.65, 0.8, 1.0)],
        "smooth_clip": [dict(sm, alpha=a, smooth_pct=sp) for a in (0.5, 0.8) for sp in (99.99, 99.9)]
        + [dict(sm, alpha=a, smooth_target=t) for a in (0.5, 0.8) for t in (190.0, 254.0)],
        "rows": [dict(best, weight_scale="row"), dict(best, weight_scale="row_pow2", row_buckets=2),
                 dict(best, weight_scale="row_pow2", row_buckets=4), dict(best, weight_scale="row_pow2", row_buckets=8),
                 dict(weight_scale="row"), dict(weight_scale="row_pow2", row_buckets=4)],
        "probs": [dict(best, prob_bits=b, exp_res=r) for b, r in ((10, 16), (12, 16), (14, 16), (8, 64), (12, 64),
                                                                 (14, 64))]
        + [dict(prob_bits=b, exp_res=r) for b, r in ((12, 16), (14, 64))],
        "best": [best],
        "calib": [dict(best, pct=p) for p in (99.99, 99.9, 99.5)]
        + [dict(best, relu_calib=True), dict(best, pct_att=99.99), dict(best, pct_att=99.9)]
        + [dict(best, calib_windows=n) for n in (4, 8)],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--config", default=None, help="benchmark config name (default: from the model name)")
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--sweep", nargs="*", default=[])
    ap.add_argument("--configs", default=None, help="JSON list of option dicts (calib_windows is a row option)")
    ap.add_argument("--baseline", action="store_true", help="add the default graph (pct 100, norm gain 4)")
    ap.add_argument("--calib-offset", type=int, default=0,
                    help="use calibration windows offset .. offset+n-1 counted back from the end of the split")
    ap.add_argument("--lean", action="store_true",
                    help="after the fp32 reference, keep the checkpoint's weights in fp16 (OPT ships fp16 weights, so "
                         "this is lossless; the builder upcasts each matrix) and pass the calibration hidden-state "
                         "maxima precomputed in fp32: fits OPT-1.3B in a laptop's memory")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    tag = args.model.split("/")[-1]
    config = args.config or tag
    configs = ([{}] if args.baseline else []) + [c for s in args.sweep for c in sweeps()[s]]
    if args.configs:
        configs += json.loads(args.configs)
    out_path = Path(args.out or ROOT / "artifacts" / "results" / "9_quality" /
                    f"fixes_{tag}_{'_'.join(args.sweep) or 'custom'}_w{args.windows}.json")

    model, tok = load_opt(args.model)
    max_cw = max([c.get("calib_windows", 1) for c in configs] + [1])
    wins, calib = load_windows(tok, args.parquet, seq=args.seq, windows=args.windows,
                               calib_windows=max_cw + args.calib_offset)
    calib = calib[args.calib_offset:]
    sc = Scorer(wins)
    t0 = time.time()
    with torch.no_grad():
        ppl_fp, _, am = sc.score(lambda w: model(w).logits)
    sc.ref_argmax = am
    print(f"fp32 ppl {ppl_fp:.3f} ({time.time() - t0:.0f}s)", flush=True)
    hmax = {}
    if args.lean:
        for n_cal in sorted({c.get("calib_windows", 1) for c in configs}):
            hmax[n_cal] = hidden_max(model, torch.cat(calib[:n_cal], 0))
        for p_ in model.parameters():
            p_.data = p_.data.half()
        print("lean: hidden maxima", hmax, flush=True)
    rows = []
    for c in configs:
        c = dict(c)
        n_cal = c.pop("calib_windows", 1)
        t = time.time()
        extra = {"hidden_max": hmax[n_cal]} if args.lean else {}
        g, info = build_opt(model, torch.cat(calib[:n_cal], 0), **c, **extra)
        rep = graph_report(g, info)
        diffs = matches_benchmark_graph(g, config)
        ppl, agree, _ = sc.score(lambda w: int_logits(g, info, w))
        del g
        row = {"options": c, "calib_windows": n_cal, "calib_offset": args.calib_offset, "ppl": ppl, "top1_agreement_with_fp32": agree,
               "graph_differences": diffs, "report": rep, "seconds": round(time.time() - t, 1)}
        rows.append(row)
        print(json.dumps({k: row[k] for k in ("options", "calib_windows", "ppl", "top1_agreement_with_fp32")}),
              "| claims<2^29:", rep["claims_ok"], "log2 max claim", rep["max_claim_bound_log2"],
              "| mult range 2^", rep["linear_mult_min_log2"], "..", rep["linear_mult_max_log2"],
              "| outside P3:", len(rep["linear_mults_outside_P3"]),
              "| vec ops:", rep["vector_mult_ops"], "distinct max", rep["distinct_mults_per_vector_op_max"],
              "| G", rep["norm_gain_min"], "..", rep["norm_gain_max"], "| diffs", diffs, f"| {row['seconds']}s",
              flush=True)
        out = {"model": args.model, "windows": args.windows, "seq": args.seq, "tokens": args.windows * (args.seq - 1),
               "fp32_ppl": ppl_fp, "rows": rows}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(out, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()

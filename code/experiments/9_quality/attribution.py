"""Error attribution of the integer OPT (F2): which integer components cost how much perplexity.

Windows as in ``experiments/2_attack/real_weights_ppl.py`` (WikiText-2 raw test, ``--windows`` x ``--seq``,
calibration on the disjoint last window).  The integer graph is ``opt_builder.build_opt`` (with its defaults,
the graph of ``real_weights.build_opt_from_hf``: checked here op by op).  The hybrids (``hybrid.run``) run
the graph op by op with chosen components in float:

* leave-one-out: everything integer except one component (float) -> what fixing that component alone gains;
* add-one: everything float except one component (integer) -> what that component costs on its own.

Validation rows: the all-integer hybrid must give the graph's logits exactly, the all-float hybrid the
fp32 checkpoint's perplexity.

    PYTHONPATH=src python experiments/9_quality/attribution.py --model facebook/opt-125m --windows 16
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
from hybrid import FULL, GROUPS, float_cfg, int_only_cfg, run  # noqa: E402
from opt_builder import build_opt, graph_report, int_logits  # noqa: E402

from pvi.fullcheck.graph import CheapOp, MatOp  # noqa: E402
from pvi.fullcheck.real_weights import build_opt_from_hf, matches_benchmark_graph  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]


def same_graph(a, b) -> list[str]:
    """Differences between two graphs' weights, biases and cheap-op constants."""
    diffs = []
    for x, y in zip(a.ops, b.ops):
        if type(x) is not type(y) or x.name != y.name:
            diffs.append(f"{x.name}/{y.name}: kind")
        elif isinstance(x, MatOp):
            if not torch.equal(x.weight, y.weight):
                diffs.append(f"{x.name}: weights differ in {int((x.weight != y.weight).sum())} entries")
            if (x.bias is None) != (y.bias is None) or (x.bias is not None and not torch.equal(x.bias, y.bias)):
                diffs.append(f"{x.name}: bias")
        elif isinstance(x, CheapOp) and x.params != y.params:
            diffs.append(f"{x.name}: params {x.params} vs {y.params}")
    if len(a.ops) != len(b.ops):
        diffs.append("op count")
    return diffs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-125m")
    ap.add_argument("--config", default="opt-125m")
    ap.add_argument("--parquet", default=None)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--windows", type=int, default=16)
    ap.add_argument("--pct", type=float, default=100.0)
    ap.add_argument("--norm-gain", type=float, default=4.0)
    ap.add_argument("--skip-sub", action="store_true", help="skip the per-feature breakdown")
    ap.add_argument("--opts", default=None, help="JSON dict of further build_opt options (e.g. the best fix)")
    ap.add_argument("--tag", default="", help="suffix of the output file")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.manual_seed(0)
    tag = args.model.split("/")[-1]
    out_path = Path(args.out or ROOT / "artifacts" / "results" / "9_quality" /
                    f"attribution_{tag}_pct{args.pct:g}_g{args.norm_gain:g}{args.tag}.json")

    model, tok = load_opt(args.model)
    wins, calib = load_windows(tok, args.parquet, seq=args.seq, windows=args.windows)
    sc = Scorer(wins)
    t0 = time.time()
    with torch.no_grad():
        ppl_fp, _, am = sc.score(lambda w: model(w).logits)
    sc.ref_argmax = am
    print(f"fp32 ppl {ppl_fp:.3f} ({time.time() - t0:.0f}s)", flush=True)
    res = {"model": args.model, "windows": args.windows, "seq": args.seq, "tokens": args.windows * (args.seq - 1),
           "pct": args.pct, "norm_gain": args.norm_gain, "fp32_ppl": ppl_fp}

    # the graph, and the check that the copy of the builder reproduces build_opt_from_hf
    extra = json.loads(args.opts) if args.opts else {}
    res["opts"] = extra
    g, info = build_opt(model, calib[0], pct=args.pct, norm_gain=args.norm_gain, keep_float=True, **extra)
    g0, info0 = build_opt_from_hf(model, calib[0], pct=args.pct, norm_gain=args.norm_gain)
    res["builder_copy_differences"] = same_graph(g, g0) if not extra else "n/a (options differ)"
    res["benchmark_graph_differences"] = matches_benchmark_graph(g, args.config)
    res["graph_report"] = graph_report(g, info)
    del g0
    print("builder copy differences:", res["builder_copy_differences"], flush=True)
    print("benchmark differences:", res["benchmark_graph_differences"], flush=True)
    print("graph report:", res["graph_report"], flush=True)
    ppl_int, agree_int, _ = sc.score(lambda w: int_logits(g, info, w))
    res["int_ppl"], res["int_top1"] = ppl_int, agree_int
    print(f"integer graph ppl {ppl_int:.3f} top1 {agree_int:.4f}", flush=True)

    # hybrid validation
    w0 = wins[0]
    with torch.no_grad():
        exact = int_logits(g, info, w0)
        hyb = run(g, info, w0, {})
        res["hybrid_all_int_max_abs_logit_diff"] = float((exact - hyb).abs().max())
        hf = model(w0).logits.double()
        hfl = run(g, info, w0, int_only_cfg())
        res["hybrid_all_float_max_abs_logit_diff_vs_hf"] = float((hf - hfl).abs().max())
    print("hybrid all-int vs graph max |dlogit|", res["hybrid_all_int_max_abs_logit_diff"],
          "; all-float vs HF", res["hybrid_all_float_max_abs_logit_diff_vs_hf"], flush=True)

    def hyb_ppl(cfg):
        with torch.no_grad():
            p, a, _ = sc.score(lambda w: run(g, info, w, cfg))
        return p, a

    rows = []

    def add(label, mode, cfg):
        t = time.time()
        p, a = hyb_ppl(cfg)
        row = {"component": label, "mode": mode, "ppl": p, "top1": a, "seconds": round(time.time() - t, 1)}
        rows.append(row)
        print(json.dumps(row), flush=True)

    add("(none)", "all integer (hybrid)", {})
    add("(all)", "all float (hybrid)", int_only_cfg())
    for label, comps in GROUPS.items():
        add(label, "leave-one-out: this float, rest integer", float_cfg(*comps))
        add(label, "add-one: this integer, rest float", int_only_cfg(*comps))

    if not args.skip_sub:
        norms = ["norm1", "norm2", "normf"]
        rqs = ["rq_q", "rq_k", "rq_v", "rq_fc1"]
        subs = [
            ("LayerNorm outputs: clip only (no rounding)", {c: {"stats", "clip"} for c in norms}),
            ("LayerNorm outputs: rounding only (no clip)", {c: {"stats", "round"} for c in norms}),
            ("LayerNorm outputs: integer stats only", {c: {"stats"} for c in norms}),
            ("final LayerNorm: rounding only", {"normf": {"stats", "round"}}),
            ("final LayerNorm: clip only", {"normf": {"stats", "clip"}}),
            ("softmax: exp table only (float p)", {"softmax": {"exp"}}),
            ("softmax: 8-bit p only (exact exp)", {"softmax": {"pbits"}}),
            ("act requants: rounding only", {c: {"round"} for c in rqs}),
            ("act requants: clip only", {c: {"clip"} for c in rqs}),
            ("attention output: rounding only", {"att_out": {"round"}}),
            ("attention output: clip only", {"att_out": {"clip"}}),
            ("residual: rounding only", {"res": {"round"}}),
            ("residual: clamp only", {"res": {"clip"}}),
        ]
        for label, part in subs:
            # add-one: only these features integer, the rest float
            cfg = int_only_cfg()
            cfg.update(part)
            add(label, "add-one (feature): these features integer, rest float", cfg)
            # leave-one-out of the complementary feature: everything integer, the component restricted to `part`
            cfg = dict(part)
            add(label, "restricted: rest integer, this component with these features only", cfg)

    res["rows"] = rows
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(res, indent=2))
    print("wrote", out_path)


if __name__ == "__main__":
    main()

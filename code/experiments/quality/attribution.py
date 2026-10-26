"""The attribution of the integer OPT-125M's perplexity to the components of its integer graph: ``tab:quality-attr`` and
the paragraph "Where the perplexity goes" of the extended version (Sec. ``sec:ext-quality``), and the same paragraph of
Sec. ``sec:quality``.  A floating-point copy of the integer graph (``hybrid_opt.Hybrid``) switches each component
between its exact integer form and float.

    python -m experiments.quality.attribution --model facebook/opt-125m --device cpu --threads 8
    python -m experiments.quality.attribution --model facebook/opt-125m --windows 2 --skip-sub     # a short run
    python -m experiments.quality.attribution --model facebook/opt-125m --graph certified --pbits 11

* Text and windows: the WikiText-2 (raw) test split joined without a separator, ``--windows`` (16) windows of
  ``--seq`` (128) tokens from its start, and calibration on one ``--seq``-token window at its end
  (``checkpoints.load_windows``: the setting of ``ppl_opt.py --calibration test-end``).
* The graph (``--graph``): ``scalar`` (the default: one norm gain, ``--gain`` 4, no clipping), ``certified``
  (``hf_opt.CERTIFIED_RECIPE``) or ``legacy`` (``hf_opt.LEGACY_RECIPE``), with ``--pbits``-bit attention probabilities
  (8); ``--opts`` (JSON) replaces keyword arguments of ``build_opt_from_hf``.  The default graph is ``ppl_opt.py``'s
  cell ``scalar-g4-p8`` at 128 tokens with test-end calibration, the same graph digest (the paper's numbers check it).
* Rows, each a perplexity over the windows with its top-1 agreement with the fp32 checkpoint: the all-integer and the
  all-float copy; per group of components (``hybrid_opt.GROUPS``) leave-one-out (that group float, the rest integer)
  and add-one (only that group integer); unless ``--skip-sub``, per feature sub-row (``hybrid_opt.SUBROWS``) add-one
  (only those features integer, the rest float) and restricted (the rest integer, the component restricted to those
  features).
* Checks: on the first window, the all-integer copy gives the graph's logits at every position exactly
  (``hf_opt.real_logits``) and the all-float copy the checkpoint's within ``--float-tol``; every op of the graph has
  its role (``hybrid_opt.roles``) and the checkpoint's weights, folded and quantised as the builder does, are the
  graph's int8 weights, biases and real scales; the graph differs from the benchmark graph of ``--config`` only by
  the LM head's bias; over every window, the all-integer row's perplexity is the graph's.  A failed check prints a
  ``FAILED:`` line and the exit status is 1 (a failure before the rows stops the run).

Output: JSON lines on stdout (and appended to ``--out``), each with ``kind: "attribution"`` and the run's head
(``model``, ``tag``, ``seq``, ``windows``, ``tokens``, ``dataset``, ``split``, ``calibration``, ``calib_windows``,
``graph``, ``graph_cell``: ``ppl_opt.py``'s name of the graph, None with ``--opts``; ``gain``, ``pbits``, ``opts``: the
builder's keyword arguments; ``device``, ``threads``, ``env``): one per row (``part: "row"``: ``group``, ``label``,
``mode``: ``all_int``, ``all_float``, ``leave_one_out``, ``add_one``, ``add_one_feature`` or ``restricted``;
``features``: the integer features of each component that is not fully integer; ``ppl``, ``top1``, ``seconds``),
then the check record (``part: "check"``: ``fp32``, ``int_ppl``, ``int_top1``, ``all_int_exact``,
``all_int_max_abs_dlogit``, ``all_float_max_abs_dlogit``, ``float_tol``, ``all_int_row_equal``, ``weights_problems``,
``benchmark_config``, ``benchmark_differences``, ``graph_report``, ``digest``, ``weights_sha256``, ``build_s``,
``seconds``, ``ok``).  A short line per row goes to stderr.

Hardware and time: the CPU only.  OPT-125M, 62 rows of 16 windows: about 20 minutes on 6 threads of a laptop CPU (an
11th-generation Intel Core), of which 15 minutes are the rows, with 5 GB of host memory at its peak;
``--windows 2 --skip-sub``: about 7 minutes, most of them the build and the checks before the rows.
"""
from __future__ import annotations

import argparse
import json
import sys
import time

import torch

from experiments.common import checkpoints, records
from experiments.common.cli import add_runtime_arguments, apply_runtime
from experiments.quality.hybrid_opt import FULL, GROUPS, SUBROWS, Hybrid, float_cfg, int_only_cfg
from pvi.models import hf_opt
from pvi.models.decoder import CONFIGS

GRAPHS = ("scalar", "certified", "legacy")
UNSUPPORTED = ("row_pow2", "cle_mlp", "prune_last")
"""Builder options whose graphs the float copy does not reproduce (per-row weight scales, cross-layer equalisation,
a pruned last block)."""


def builder_kwargs(args) -> tuple[dict, str | None]:
    """The keyword arguments of ``build_opt_from_hf`` for ``--graph``, ``--gain``, ``--pbits`` and ``--opts``, and the
    graph's cell name in ``ppl_opt.py``'s terms (None with ``--opts``)."""
    if args.graph == "scalar":
        kw, cell = dict(norm_gain=args.gain), f"scalar-g{args.gain:g}-p{args.pbits}"
    else:
        recipe = checkpoints.OPT_RECIPES[args.graph]
        kw, cell = dict(recipe), f"smooth-{args.graph}-a{recipe['smooth']:g}-p{args.pbits}"
    kw["prob_max"] = (1 << args.pbits) - 1
    extra = json.loads(args.opts)
    kw.update(extra)
    return kw, None if extra else cell


def plan(skip_sub: bool) -> list[tuple[str, str, str, dict]]:
    """``(group, label, mode, configuration)`` of every row, in order."""
    rows = [("all", "all integer", "all_int", {}), ("all", "all float", "all_float", int_only_cfg())]
    for key, (label, comps) in GROUPS.items():
        rows.append((key, label, "leave_one_out", float_cfg(*comps)))
        rows.append((key, label, "add_one", int_only_cfg(*comps)))
    if not skip_sub:
        for key, (label, part) in SUBROWS.items():
            rows.append((key, label, "add_one_feature", {**int_only_cfg(), **part}))
            rows.append((key, label, "restricted", dict(part)))
    return rows


def features(cfg: dict) -> dict:
    """The integer features of every component that ``cfg`` does not leave fully integer."""
    return {c: sorted(cfg[c]) for c in FULL if c in cfg and set(cfg[c]) != FULL[c]}


def benchmark_config(args) -> str | None:
    """The benchmark configuration of the checkpoint's shape: ``--config``, else the model's name if it is one."""
    name = args.config or args.model.split("/")[-1]
    return name if name in CONFIGS else None


def _failed(reasons: list[str]) -> int:
    print("FAILED: " + "; ".join(reasons), flush=True)
    return 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="facebook/opt-125m", help="an OPT checkpoint")
    ap.add_argument("--config", default=None,
                    help="the benchmark configuration of its shape (default: the model's name, if it is one)")
    ap.add_argument("--seq", type=int, default=128, help="tokens per window and of the calibration window")
    ap.add_argument("--windows", type=int, default=16, help="evaluated windows from the start of the test split")
    ap.add_argument("--graph", choices=GRAPHS, default="scalar",
                    help="scalar (one norm gain, no clipping), certified or legacy (hf_opt's recipes)")
    ap.add_argument("--gain", type=float, default=4.0, help="the scalar graph's norm gain")
    ap.add_argument("--pbits", type=int, default=8, help="attention probability bits (8: 0..255; 11: 0..2047)")
    ap.add_argument("--opts", default="{}", help="JSON keyword arguments of build_opt_from_hf replacing the graph's")
    ap.add_argument("--skip-sub", action="store_true", help="leave out the feature sub-rows")
    ap.add_argument("--float-tol", type=float, default=1e-3,
                    help="the largest |logit| difference between the all-float copy and the checkpoint on the first "
                         "window")
    ap.add_argument("--tag", default="", help="a label written into every record of this run")
    ap.add_argument("--out", default=None, help="a JSONL file the records are appended to")
    add_runtime_arguments(ap, threads=8, verifier_device=False)
    ap.set_defaults(device="cpu")
    args = ap.parse_args()
    if args.device != "cpu":
        raise SystemExit("the attribution runs on the CPU (--device cpu)")
    kw, cell = builder_kwargs(args)
    if any(kw.get(k) for k in UNSUPPORTED):
        raise SystemExit(f"the float copy does not reproduce graphs built with {', '.join(UNSUPPORTED)}")
    apply_runtime(args)
    t_start = time.perf_counter()
    model, tok = checkpoints.load_opt(args.model)
    wins, calib = checkpoints.load_windows(tok, seq=args.seq, n=args.windows, calib_windows=1)
    head = {"kind": "attribution", "model": args.model, "tag": args.tag, "seq": args.seq, "windows": len(wins),
            "tokens": sum(w.shape[1] - 1 for w in wins), "dataset": "wikitext-2-raw-v1 test, ''.join",
            "split": "test", "calibration": "test-end", "calib_windows": 1, "graph": args.graph, "graph_cell": cell,
            "gain": args.gain if args.graph == "scalar" else None, "pbits": args.pbits, "opts": kw,
            "device": args.device, "threads": args.threads, "env": records.environment("cpu")}
    t0 = time.perf_counter()
    with torch.no_grad():
        graph, info = hf_opt.build_opt_from_hf(model, calib[0], **kw)
    check = {"build_s": time.perf_counter() - t0, "digest": graph.digest().hex(),
             "weights_sha256": checkpoints.weights_sha(graph), "float_tol": args.float_tol}
    config = benchmark_config(args)
    diffs = None if config is None else hf_opt.matches_benchmark_graph(graph, config)
    check.update(benchmark_config=config, benchmark_differences=diffs)
    reasons = []
    if diffs is not None and diffs != [f"{graph.ops[-1].name}: bias True vs benchmark False"]:
        reasons.append(f"the graph differs from the benchmark graph {config} by {diffs}")
    try:
        hyb = Hybrid(model, graph, info)
        check["weights_problems"] = hyb.problems
        reasons += [f"weights: {p}" for p in hyb.problems]
    except ValueError as e:
        hyb = None
        check["weights_problems"] = [f"role map: {e}"]
        reasons.append(f"the role map does not hold: {e}")
    # the claim bound, the multipliers outside V1's range and the norm gains, each op named by its component and block
    check["graph_report"] = checkpoints.graph_report(graph, None if hyb is None else {"meta": hyb.meta})

    # the reference, the integer graph and the two invariants of the copy
    sc = checkpoints.Scorer(wins)
    with torch.no_grad():
        fp32, _, argmax = sc.score(lambda w: model(w).logits)
        sc.ref_argmax = argmax
        int_ppl, int_top1, _ = sc.score(lambda w: hf_opt.real_logits(graph, info, w, all_positions=True))
        check.update(fp32=fp32, int_ppl=int_ppl, int_top1=int_top1)
        if hyb is not None:
            w0 = wins[0]
            exact = hf_opt.real_logits(graph, info, w0, all_positions=True)
            all_int = hyb.logits(w0, {})
            ref = model(w0).logits.double()
            all_float = hyb.logits(w0, int_only_cfg())
            check.update(all_int_exact=bool(torch.equal(exact, all_int)),
                         all_int_max_abs_dlogit=float((exact - all_int).abs().max()),
                         all_float_max_abs_dlogit=float((ref - all_float).abs().max()))
            if not check["all_int_exact"]:
                reasons.append(f"the all-integer copy is not the graph's logits (max |dlogit| "
                               f"{check['all_int_max_abs_dlogit']:.3g})")
            if not check["all_float_max_abs_dlogit"] <= args.float_tol:
                reasons.append(f"the all-float copy is {check['all_float_max_abs_dlogit']:.3g} from the checkpoint's "
                               f"logits (--float-tol {args.float_tol:g})")
            del exact, all_int, ref, all_float
    print(f"fp32 ppl {fp32:.4f}; integer graph ppl {int_ppl:.4f} (top-1 {int_top1:.4f}); all-integer copy max |dlogit| "
          f"{check.get('all_int_max_abs_dlogit')}, all-float copy {check.get('all_float_max_abs_dlogit')}",
          file=sys.stderr, flush=True)
    if reasons:
        records.emit(dict(head, part="check", **check, seconds=time.perf_counter() - t_start, ok=False), args.out)
        return _failed(reasons)
    del model

    # the rows
    ppl_of = {}
    for group, label, mode, cfg in plan(args.skip_sub):
        t = time.perf_counter()
        with torch.no_grad():
            ppl, top1, _ = sc.score(lambda w: hyb.logits(w, cfg))
        ppl_of[group, mode] = ppl
        row = dict(head, part="row", group=group, label=label, mode=mode, features=features(cfg), ppl=ppl, top1=top1,
                   seconds=round(time.perf_counter() - t, 1))
        records.emit(row, args.out)
        print(f"{label:48s} {mode:16s} ppl {ppl:9.4f}  top-1 {top1:.4f}  ({row['seconds']:.0f} s)", file=sys.stderr,
              flush=True)
    check["all_int_row_equal"] = ppl_of["all", "all_int"] == int_ppl
    if not check["all_int_row_equal"]:
        reasons.append(f"the all-integer row's perplexity {ppl_of['all', 'all_int']!r} is not the graph's {int_ppl!r}")
    records.emit(dict(head, part="check", **check, seconds=time.perf_counter() - t_start, ok=not reasons), args.out)
    return _failed(reasons) if reasons else 0


if __name__ == "__main__":
    sys.exit(main())

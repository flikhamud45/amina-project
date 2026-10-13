"""A6: the Kpre client's one-time precompute (``Verifier.precompute``: ``u = chi A`` for every weight matrix) with the
float64 limb products (``PVI_FOLD_FAST=0``) and with A6's paths (native field product on a CPU, int8 GEMMs on a GPU),
on the same challenges; the ``u`` must be identical.

    cd code && PYTHONPATH=src python experiments/6_improvements/fold_ab.py --model llama2-7b --layers 0 \\
        --devices cpu cuda --reps 3 --fs --threads 8

One JSON line per (device, path): the median precompute time over ``--reps`` runs, the number of weight matrices and
their entries, and whether every ``u`` equals the float64 path's.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time

import torch


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama2-7b")
    ap.add_argument("--layers", type=int, default=2, help="decoder blocks (0: the full model)")
    ap.add_argument("--devices", nargs="+", default=["cpu"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--fs", action="store_true", help="Fiat-Shamir's repetitions (7 instead of 5 at lambda 128)")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier
    from pvi.v0.params import params_for
    from pvi.models.decoder import CONFIGS, build_decoder
    cfg = CONFIGS[args.model]
    graph = build_decoder(cfg, n_layers=args.layers or cfg.n_layers, calib_tokens=8, seed=0, prune_last=True)
    params = params_for(128, len(graph.mat_ops), fiat_shamir=args.fs)
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    entries = sum(op.weight.numel() for op in graph.mat_ops)
    ok_all = True
    for dev in args.devices:
        if dev.startswith("cuda") and not torch.cuda.is_available():
            continue
        ref = None
        for path in ("0", "1"):
            os.environ["PVI_FOLD_FAST"] = path
            ts, pre = [], None
            for _ in range(args.reps + 1):                 # the first run warms up (uploads, the library build)
                v = Verifier(graph.public(), params, "Kpre", weights=weights, device=dev)
                t0 = time.perf_counter()
                v.precompute(Challenger(seed=5))
                ts.append(time.perf_counter() - t0)
                pre = {k: b.cpu() for k, (_, b) in v._pre.items()}
                del v
            if ref is None:
                ref = pre
            same = all(torch.equal(pre[k], ref[k]) for k in ref)
            ok_all &= same
            print(json.dumps({"model": cfg.name, "layers": args.layers or cfg.n_layers, "device": dev, "path": path,
                              "fs": args.fs, "reps": params.reps, "matrices": len(pre), "entries": entries,
                              "precompute_s": statistics.median(ts[1:]), "all_s": [round(t, 3) for t in ts],
                              "equal_to_float64": same, "threads": args.threads}), flush=True)
    os.environ.pop("PVI_FOLD_FAST", None)
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

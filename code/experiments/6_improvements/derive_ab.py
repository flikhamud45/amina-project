"""A/B of the CPU verifier's derive (step S1, the recomputed cheap ops) with the cheap ops under
``torch.compile`` (inductor's fused C++ loops; needs a C++ compiler on the host).

    cd code && PYTHONPATH=src python experiments/6_improvements/derive_ab.py --model llama2-7b --layers 2 --seq 2048

Builds the decoder as bench.py does (random weights, seed 0, last block pruned), computes one query's
claims, then times ``Verifier.derive`` (mode Kpre, CPU, ``--threads``) as it is, with only the int32
attention core compiled, and with every cheap op compiled, on the same claims, and checks that every
derived input is identical.  Prints one JSON line per variant (median of ``--reps``, compile time apart).
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import sys
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama2-7b")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=3)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from pvi.fullcheck import transformer as tr
    from pvi.fullcheck.graph import CheapOp
    from pvi.fullcheck.protocol import Verifier, params_for
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    cfg = CONFIGS[args.model]
    graph = build_decoder(cfg, n_layers=args.layers, calib_tokens=32, seed=0, prune_last=True)
    x = torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(1))
    _, claims = graph.forward(x)
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    params = params_for(128, len(graph.mat_ops))
    core = tr._attention_core
    fns = {op.name: op.fn for op in graph.ops if isinstance(op, CheapOp)}

    def variant(name):
        tr._attention_core = core
        for op in graph.ops:
            if isinstance(op, CheapOp):
                op.fn = fns[op.name]
        if name in ("core", "all"):
            tr._attention_core = torch.compile(core, dynamic=False)
        if name == "all":
            for op in graph.ops:
                if isinstance(op, CheapOp):
                    op.fn = torch.compile(fns[op.name], dynamic=False)
        return Verifier(graph.public(), params, "K", weights=weights)

    ref = None
    for name in ("eager", "core", "all"):
        try:
            v = variant(name)
            t0 = time.perf_counter()
            out = v.derive(x, claims)                  # the first call compiles
            first = time.perf_counter() - t0
            times = []
            for _ in range(args.reps):
                t0 = time.perf_counter()
                out = v.derive(x, claims)
                times.append(time.perf_counter() - t0)
            same = out is not None and (ref is None or all(torch.equal(out[k], ref[k]) for k in ref))
            ref = ref if ref is not None else out
            row = {"variant": name, "derive_s": statistics.median(times), "first_s": first, "identical": same}
        except Exception as exc:                         # e.g. no C++ compiler for inductor
            row = {"variant": name, "error": f"{type(exc).__name__}: {str(exc)[:300]}"}
        print(json.dumps({"model": args.model, "layers": args.layers, "seq": args.seq, "threads": args.threads,
                          **row, "host": platform.node(), "torch": torch.__version__}), flush=True)
    tr._attention_core = core


if __name__ == "__main__":
    sys.exit(main())

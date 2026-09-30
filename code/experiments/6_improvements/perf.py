"""Timing harness for the improvement work: the same queries on the old and the new code.

    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model lenet5 --modes C Kpre --queries 10
    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C --verifier-device cuda
    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C Kpre --wire

Builds the model exactly as bench.py does (trained CNN weights, random-weight decoders with seed 0),
commits it, runs one warm-up and ``--queries`` honest queries with ``run_query`` and prints one JSON
line per (model, mode): the median of every timing part and every byte part, the acceptance count,
and the git commit, so results of two checkouts can be compared line by line.  ``--wire`` sends the
proof in the compact encoding (``run_query(wire=True)``: encoded bytes, ``prove_encode`` and
``verify_decode``).
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

CODE = Path.cwd()   # run from code/


def _bench():
    spec = importlib.util.spec_from_file_location("bench", CODE / "experiments/4_defence_benchmark/bench.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: full model)")
    ap.add_argument("--modes", nargs="+", default=["C", "Kpre"])
    ap.add_argument("--lam", type=int, default=128)
    ap.add_argument("--rate", type=int, default=4)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verifier-device", default="cpu")
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--label", default="")
    ap.add_argument("--extra", default="{}", help="JSON dict of extra keyword arguments for Verifier (new options)")
    ap.add_argument("--wire", action="store_true", help="run_query(wire=True): the compact encoding of the proof")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    bench = _bench()
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query

    device = torch.device(args.device)
    t0 = time.perf_counter()
    if args.model in ("mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar", "resnet18_224"):
        from pvi.fullcheck.quantize import quantize_input, quantize_model
        model, data = bench._mlp_model() if args.model == "mlp_mnist" else bench._cnn_model(args.model)
        graph = quantize_model(model, data["train_x"])
        xs, _ = next(bench._test_batches(data, args.queries + 1))
        queries = [x for x in quantize_input(graph, xs).split(1)]
        n_checks = len(graph.mat_ops)
        seq = None
    else:
        from pvi.fullcheck.analytic import decoder_shapes
        from pvi.fullcheck.transformer import CONFIGS, build_decoder
        cfg = CONFIGS[args.model]
        n_layers = args.layers or cfg.n_layers
        graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(args.seq, 32), seed=0)
        tokens = torch.randint(0, cfg.vocab, (args.queries + 1, args.seq), generator=torch.Generator().manual_seed(1))
        queries = [t for t in tokens.split(1)]
        n_checks = len(decoder_shapes(cfg))   # (r, t) for the full model, as bench.py
        seq = args.seq
    build_s = time.perf_counter() - t0
    mats = graph.mat_ops
    coms, commit_s = None, None
    if "C" in args.modes:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        coms = commit_graph(graph, args.rate, device=device)
        if device.type == "cuda":
            torch.cuda.synchronize()
        commit_s = time.perf_counter() - t0
    lean = args.lean and seq is not None
    prover = Prover(graph, device=device, commitments=coms, lean=lean)
    weights = {op.name: (op.weight, op.bias) for op in mats}
    extra = json.loads(args.extra)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=CODE).stdout.strip()
    for mode in args.modes:
        params = params_for(args.lam, n_checks, rate=args.rate)
        if mode == "C":
            v = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()},
                         lean=lean, device=args.verifier_device, **extra)
        else:
            v = Verifier(graph.public(), params, mode, weights=weights, lean=lean, device=args.verifier_device, **extra)
            if mode == "Kpre":
                v.precompute(Challenger())
        run_query(prover, v, queries[0], wire=args.wire)  # warm-up
        runs = [run_query(prover, v, q, wire=args.wire) for q in queries[1:]]
        timing = {k: statistics.median(r["timings"].get(k, 0.0) for r in runs)
                  for k in sorted({k for r in runs for k in r["timings"]})}
        nbytes = {k: statistics.median(r["bytes"].get(k, 0) for r in runs)
                  for k in sorted({k for r in runs for k in r["bytes"]})}
        prove = sum(timing.get(k, 0.0) for k in ("prove_forward", "prove_fold", "prove_open", "prove_encode", "fs_hash"))
        verify = sum(v_ for k, v_ in timing.items() if k.startswith("verify_") or k == "fs_hash")
        out = {"label": args.label, "git": sha, "model": args.model, "seq": seq, "layers": args.layers or None,
               "mode": mode, "wire": args.wire, "lam": args.lam, "rate": args.rate, "reps": params.reps, "columns": params.columns,
               "device": args.device, "verifier_device": args.verifier_device, "threads": args.threads,
               "accepted": sum(bool(r["accepted"]) for r in runs), "queries": len(runs),
               "prove_s": prove, "verify_s": verify, "proof_bytes": sum(nbytes.values()),
               "timings": timing, "bytes": nbytes, "build_s": build_s, "commit_s": commit_s,
               "host": platform.node(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    sys.exit(main())

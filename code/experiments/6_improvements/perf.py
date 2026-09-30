"""Timing harness for the improvement work: the same queries on the old and the new code.

    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model lenet5 --modes C Kpre --queries 10
    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C --verifier-device cuda
    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model vgg16 --modes C --policy paper cnn17 R16

Builds the model exactly as bench.py does (trained CNN weights, random-weight decoders with seed 0;
``--random-init``: random CNN weights too, as ``ab_verifier.py`` builds them), commits it, runs one
warm-up and ``--queries`` honest queries with ``run_query`` and prints one JSON line per (model,
mode): the median of every timing part and every byte part, the acceptance count, and the git
commit, so results of two checkouts can be compared line by line.

``--policy`` names the commitment plans (``pvi.fullcheck.plans``) of mode C: each is committed (its
``commit_s`` is the setup time) and their queries are interleaved in this process, the order
rotating from query to query.  Each policy gets its own line, with ``vs_first``: its medians over
those of the first policy (e.g. ``paper``, the report).  Modes K and Kpre open no columns, so they
run once.
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


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, CODE / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # (its dataclasses look their module up)
    spec.loader.exec_module(mod)
    return mod


def _medians(runs: list[dict]) -> tuple[dict, dict, float, float]:
    """``(timings, bytes, prove_s, verify_s)``: the medians over ``runs`` of ``run_query``."""
    timing = {k: statistics.median(r["timings"].get(k, 0.0) for r in runs)
              for k in sorted({k for r in runs for k in r["timings"]})}
    nbytes = {k: statistics.median(r["bytes"].get(k, 0) for r in runs)
              for k in sorted({k for r in runs for k in r["bytes"]})}
    prove = sum(timing.get(k, 0.0) for k in ("prove_forward", "prove_fold", "prove_open", "fs_hash"))
    verify = sum(v_ for k, v_ in timing.items() if k.startswith("verify_") or k == "fs_hash")
    return timing, nbytes, prove, verify


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
    ap.add_argument("--policy", nargs="+", default=["paper"],
                    help="mode C: commitment plans (paper, tight, cnn<e>, R<rate>), timed interleaved in this process")
    ap.add_argument("--random-init", action="store_true",
                    help="CNNs: random weights (ab_verifier.py's models: the costs depend on the shapes only)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    bench = _load("bench", "experiments/4_defence_benchmark/bench.py")
    import pvi.fullcheck as fullcheck
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query

    device = torch.device(args.device)
    t0 = time.perf_counter()
    model_ops = None
    if args.model in ("mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar", "resnet18_224"):
        from pvi.fullcheck.quantize import quantize_input, quantize_model
        if args.random_init:
            ab = _load("ab_verifier", "experiments/6_improvements/ab_verifier.py")
            graph, x, _ = ab.build(fullcheck, ab.Case(args.model, "C", args.lam))
            xs = torch.randn(args.queries + 1, *x.shape[1:], generator=torch.Generator().manual_seed(2))
        else:
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
        model_ops = decoder_shapes(cfg)
        n_checks = len(model_ops)   # (r, t) for the full model, as bench.py
        seq = args.seq
    build_s = time.perf_counter() - t0
    mats = graph.mat_ops
    policies = args.policy if "C" in args.modes else []
    coms, commit_s = {}, {}
    for policy in policies:
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        coms[policy] = commit_graph(graph, args.rate, device=device, policy=policy, model_ops=model_ops)
        if device.type == "cuda":
            torch.cuda.synchronize()
        commit_s[policy] = time.perf_counter() - t0
    lean = args.lean and seq is not None
    weights = {op.name: (op.weight, op.bias) for op in mats}
    extra = json.loads(args.extra)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=CODE).stdout.strip()
    for mode in args.modes:
        pairs = {}
        for policy in (policies if mode == "C" else [None]):
            c = coms.get(policy)
            params = params_for(args.lam, n_checks, rate=args.rate, plan=None if c is None else c.plan)
            if mode == "C":
                v = Verifier(graph.public(), params, "C", publics=c.publics, groups=c.group_publics,
                             lean=lean, device=args.verifier_device, **extra)
            else:
                v = Verifier(graph.public(), params, mode, weights=weights, lean=lean, device=args.verifier_device,
                             **extra)
                if mode == "Kpre":
                    v.precompute(Challenger())
            prover = Prover(graph, device=device, commitments=c, lean=lean)
            run_query(prover, v, queries[0])  # warm-up
            pairs[policy] = (prover, v, [])
        order = list(pairs)
        for i, q in enumerate(queries[1:]):     # interleaved, the first policy rotating
            for policy in order[i % len(order):] + order[:i % len(order)]:
                prover, v, runs = pairs[policy]
                runs.append(run_query(prover, v, q))
        first = None
        for policy, (_, v, runs) in pairs.items():
            timing, nbytes, prove, verify = _medians(runs)
            out = {"label": args.label, "git": sha, "model": args.model, "seq": seq, "layers": args.layers or None,
                   "mode": mode, "policy": policy, "lam": args.lam, "rate": args.rate, "reps": v.params.reps,
                   "columns": v.params.columns,
                   **({"group_columns": dict(v.params.group_columns)} if v.groups else {}),
                   "device": args.device, "verifier_device": args.verifier_device, "threads": args.threads,
                   "accepted": sum(bool(r["accepted"]) for r in runs), "queries": len(runs),
                   "prove_s": prove, "verify_s": verify, "proof_bytes": sum(nbytes.values()),
                   "timings": timing, "bytes": nbytes, "build_s": build_s, "commit_s": commit_s.get(policy),
                   "host": platform.node(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
            if first is None:
                first = out
            elif len(pairs) > 1:
                out["vs_first"] = {k: out[k] / first[k] for k in ("prove_s", "verify_s", "proof_bytes", "commit_s")
                                   if first[k]}
                out["vs_first"].update({k: t / first["timings"][k] for k, t in timing.items()
                                        if first["timings"].get(k)})
            print(json.dumps(out), flush=True)


if __name__ == "__main__":
    sys.exit(main())

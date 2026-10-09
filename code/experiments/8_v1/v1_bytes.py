"""V1's proof size against today's (v0), on the same model, query and parameters.

    cd code && PYTHONPATH=src python experiments/8_v1/v1_bytes.py --model llama2-7b --layers 2 --seq 2048 \\
        --modes C Kpre --policy auto --prune-last --fs
    cd code && PYTHONPATH=src python experiments/8_v1/v1_bytes.py --model gpt2 --seq 64 --modes C --full

Builds the decoder as ``experiments/6_improvements/perf.py`` and ``bench.py`` do (random weights, seed 0; the
parameters ``(r, t)`` of the full model), commits it under each ``--policy`` (mode C), runs one v0 query on the
wire (``run_query(wire=True)``) and measures V1's bytes on the same query with
``cut_protocol.cut_proof_bytes`` (M1 encoded for real, the other messages sized exactly; ``--full`` also runs the
whole V1 query, whose eager GKR suits small models only, and checks its bytes).  One JSON line per variant: both
byte breakdowns, their ratio, the cut's plan (leaves, instances, table, exceptions) and the git commit.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

CODE = Path.cwd()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: full model)")
    ap.add_argument("--modes", nargs="+", default=["C", "Kpre"])
    ap.add_argument("--policy", nargs="+", default=["auto"])
    ap.add_argument("--lam", type=int, default=128)
    ap.add_argument("--fs", action="store_true", help="Fiat--Shamir parameters (grinding bits added)")
    ap.add_argument("--prune-last", action="store_true")
    ap.add_argument("--lmax", type=int, default=27)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--full", action="store_true", help="also run the whole V1 query (the GKR on the GPU with Triton)")
    ap.add_argument("--reps", type=int, default=1, help="with --full: timed queries of each protocol after a warm-up")
    ap.add_argument("--verifier-device", default="cpu")
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from pvi.fullcheck import cut_protocol as cp
    from pvi.fullcheck.analytic import decoder_shapes
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    cfg = CONFIGS[args.model]
    n_layers = args.layers or cfg.n_layers
    t0 = time.perf_counter()
    graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(args.seq, 32), seed=0, prune_last=args.prune_last)
    model_ops = decoder_shapes(cfg, prune_last=args.prune_last)
    build_s = time.perf_counter() - t0
    x = torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(1))
    n_checks = len(model_ops)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=CODE).stdout.strip()
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    variants = [("C", pol) for pol in args.policy if "C" in args.modes] + ([("Kpre", None)] if "Kpre" in args.modes else [])
    for mode, policy in variants:
        t0 = time.perf_counter()
        if mode == "C":
            coms = commit_graph(graph, 4, device=args.device, policy=policy, model_ops=model_ops)
            plan = coms.plan
            kw = dict(publics=coms.publics, groups=coms.group_publics, tables=coms.table_publics)
            prover = Prover(graph, device=args.device, commitments=coms, lean=args.lean)
        else:
            plan, kw = None, dict(weights=weights)
            prover = Prover(graph, device=args.device, lean=args.lean)
        setup_s = time.perf_counter() - t0
        p0 = params_for(args.lam, n_checks, fiat_shamir=args.fs, plan=plan)
        p1 = params_for(args.lam, n_checks, fiat_shamir=args.fs, plan=plan, cut=True, cut_lmax=args.lmax)
        v0 = Verifier(graph.public(), p0, mode, lean=args.lean, device=args.verifier_device, **kw)
        v1 = Verifier(graph.public(), p1, mode, lean=args.lean, device=args.verifier_device, **kw)
        if mode == "Kpre":
            v0.precompute(Challenger())
            v1._pre = v0._pre                        # the same secret rows (only F2 reads them in V1)
        t0 = time.perf_counter()
        ref = run_query(prover, v0, x, wire=True)
        v0_s = time.perf_counter() - t0
        t0 = time.perf_counter()
        b1 = cp.cut_proof_bytes(prover, v1, x, seed=7)
        v1_s = time.perf_counter() - t0
        stats = {k: b1.pop(k) for k in ("n_leaves", "n_instances", "table", "n_exceptions")}
        total0, total1 = sum(ref["bytes"].values()), sum(b1.values())
        out = {"label": args.label, "git": sha, "model": args.model, "layers": n_layers, "seq": args.seq, "mode": mode,
               "policy": policy, "chosen": None if plan is None else plan.policy, "fs": args.fs,
               "prune_last": args.prune_last, "lmax": args.lmax, "v0_accepted": ref["accepted"],
               "v0_bytes": ref["bytes"], "v1_bytes": b1, "v0_total": total0, "v1_total": total1,
               "ratio": total0 / total1, **stats, "reps": p1.reps, "columns": p1.columns,
               "build_s": build_s, "setup_s": setup_s, "v0_query_s": v0_s, "v1_bytes_s": v1_s,
               "host": platform.node(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        if args.full:
            t0 = time.perf_counter()
            full = cp.run_cut_query(prover, v1, x)
            out["v1_full"] = {"accepted": full["accepted"], "rejected_at": full["rejected_at"], "bytes": full["bytes"],
                              "timings": full["timings"], "s": time.perf_counter() - t0}
            if args.reps > 1:                              # medians of timed queries of both protocols, interleaved
                import statistics
                runs0, runs1 = [], []
                for i in range(args.reps):
                    q = torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(100 + i))
                    runs0.append(run_query(prover, v0, q, wire=True))
                    runs1.append(cp.run_cut_query(prover, v1, q))

                def med(runs):
                    keys = sorted({k for r in runs for k in r["timings"]})
                    t = {k: statistics.median(r["timings"].get(k, 0.0) for r in runs) for k in keys}
                    return {"accepted": sum(r["accepted"] for r in runs), "timings": t,
                            "prove_s": sum(v for k, v in t.items() if k.startswith("prove_")),
                            "verify_s": sum(v for k, v in t.items() if k.startswith("verify_")),
                            "fs_s": t.get("fs_hash", 0.0)}
                out["timed"] = {"queries": args.reps, "v0": med(runs0), "v1": med(runs1)}
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    sys.exit(main())

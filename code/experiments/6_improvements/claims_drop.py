"""What a CPU prover's wire query spends encoding its claims, and dropping them after it, in a checkout.

    cd code && PYTHONPATH=src python experiments/6_improvements/claims_drop.py --threads 4
    cd ../../base/code && PYTHONPATH=src python <this checkout>/code/experiments/6_improvements/claims_drop.py --threads 4

The benchmark's random-weight decoder (default Qwen3-4B, 8 blocks, 8 tokens), a Kpre verifier (lambda 40)
and a CPU prover, after one warm-up query.  Per query: ``run_query(wire=True)``'s ``prove_encode`` and the
time inside ``claimcodec.encode`` (their difference: what else the timer holds, e.g. dropping the claims);
then the same query's claims again (``Prover.claims``), dropped (``del``) without being encoded; then again,
encoded, then dropped, each step timed.  Works with any checkout's ``src`` on ``PYTHONPATH``.  Nothing here
writes to the benchmark's roots.
"""
from __future__ import annotations

import argparse
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--model", default="qwen3-4b")
    ap.add_argument("--layers", type=int, default=8)
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--queries", type=int, default=7)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from pvi.fullcheck import claimcodec as cc
    from pvi.fullcheck import protocol as proto
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    src = Path(cc.__file__).resolve().parents[2]
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=src).stdout.strip()
    cfg = CONFIGS[args.model]
    graph = build_decoder(cfg, n_layers=args.layers, calib_tokens=min(32, args.seq), seed=0)
    v = proto.Verifier(graph.public(), proto.params_for(40, 1), "Kpre",
                       weights={op.name: (op.weight, op.bias) for op in graph.mat_ops})
    v.precompute(proto.Challenger())
    prover = proto.Prover(graph, device="cpu")
    inside = []
    real = cc.encode

    def timed(*a, **k):
        t0 = time.perf_counter()
        out = real(*a, **k)
        inside.append(time.perf_counter() - t0)
        return out

    cc.encode = timed
    tokens = torch.randint(0, cfg.vocab, (args.queries + 1, args.seq), generator=torch.Generator().manual_seed(1))
    proto.run_query(prover, v, tokens[:1], wire=True)                 # warm-up
    rows = []
    for q in range(1, args.queries + 1):
        x = tokens[q:q + 1]
        r = proto.run_query(prover, v, x, wire=True)
        pe, enc = r["timings"]["prove_encode"], inside[-1]
        claims = prover.claims(x, to_host=False)
        t0 = time.perf_counter()
        del claims
        t1 = time.perf_counter()
        claims = prover.claims(x, to_host=False)
        zs = [claims[op.name] for op in graph.mat_ops]
        t2 = time.perf_counter()
        real(zs)
        t3 = time.perf_counter()
        del claims, zs
        t4 = time.perf_counter()
        rows.append((pe, enc, t1 - t0, t3 - t2, t4 - t3))
        print(f"{sha} threads {args.threads} q{q} accepted {r['accepted']}: prove_encode {1e3 * pe:.1f} ms, inside encode "
              f"{1e3 * enc:.1f}, the rest {1e3 * (pe - enc):.1f}; the claims again: dropped {1e3 * (t1 - t0):.1f}; "
              f"encoded {1e3 * (t3 - t2):.1f}, then dropped {1e3 * (t4 - t3):.1f}", flush=True)
    med = [1e3 * statistics.median(c) for c in zip(*rows)]
    print(f"{sha} threads {args.threads} medians: prove_encode {med[0]:.1f} ms, inside encode {med[1]:.1f}, the rest "
          f"{med[0] - med[1]:.1f}; the claims again: dropped {med[2]:.1f}; encoded {med[3]:.1f}, then dropped {med[4]:.1f}",
          flush=True)


if __name__ == "__main__":
    sys.exit(main())

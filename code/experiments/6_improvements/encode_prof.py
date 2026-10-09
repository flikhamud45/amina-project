"""Where the prover's claim encoding (``claimcodec.encode`` on its device) spends its time.

    cd code && PYTHONPATH=src python experiments/6_improvements/encode_prof.py --model opt-1.3b --layers 4 --seq 2048

Builds the decoder as bench.py does (random weights, seed 0, last block pruned), computes one query's
claims on the device as the lean wire prover keeps them (int32, ``claimcodec.narrow``), then times
``claimcodec.encode`` on them (median of ``--reps``) and prints a ``torch.profiler`` table of one more
call, sorted by CPU and by CUDA time.  ``--chunks`` times proofs of more than ``_ONE_PASS`` values with
other part sizes (``claimcodec._CHUNK``, default 2**23) and checks they give the same bytes; ``--no-profile``
skips the table.  ``--compare-torch`` encodes the same claims with the fused Triton encoder and with the torch
encoder (``claimcodec._FUSED`` on and off), checks that the bytes are the same and prints both medians.
"""
from __future__ import annotations

import argparse
import statistics
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="opt-1.3b")
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--rows", type=int, default=25)
    ap.add_argument("--chunks", type=int, nargs="*", default=[], help="log2 part sizes to time, e.g. 23 25 26")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--compare-torch", action="store_true")
    args = ap.parse_args()
    from pvi.fullcheck import claimcodec
    from pvi.fullcheck.protocol import Prover
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    device = torch.device("cuda")
    cfg = CONFIGS[args.model]
    graph = build_decoder(cfg, n_layers=args.layers, calib_tokens=32, seed=0, prune_last=True)
    prover = Prover(graph, device=device, lean=True)
    x = torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(1))
    claims = prover.claims(x, send=claimcodec.narrow, to_host=False)
    zs = [claims[op.name] for op in graph.mat_ops if op.name in claims]
    n = sum(z.numel() for z in zs)
    print(f"{args.model} {args.layers} blocks T={args.seq}: {len(zs)} ops, {n / 1e6:.1f} M claims, "
          f"devices {sorted({str(z.device) for z in zs})}, dtypes {sorted({str(z.dtype) for z in zs})}")
    if args.compare_torch:
        blobs, med = {}, {}
        for fused in (False, True, False, True):          # interleaved; the first of each warms up
            claimcodec._FUSED = fused
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            blob = claimcodec.encode(zs)
            torch.cuda.synchronize()
            dt = time.perf_counter() - t0
            if fused in blobs:
                med[fused] = dt
            blobs.setdefault(fused, blob)
            del blob
        same = blobs[True] == blobs[False]
        print(f"torch encoder {med[False]:.3f} s, fused {med[True]:.3f} s ({med[False] / med[True]:.1f}x), "
              f"same bytes: {same}, {len(blobs[True]) / 1e6:.1f} MB, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB",
              flush=True)
        return
    times, size = [], 0
    for _ in range(args.reps + 1):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        size = len(claimcodec.encode(zs))
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    med = statistics.median(times[1:])
    print(f"encode: median {med:.3f} s over {args.reps} ({n / med / 1e6:.0f} M claims/s), {size / 1e6:.1f} MB, "
          f"{8 * size / n:.2f} bits/claim; first call {times[0]:.3f} s; one pass up to {claimcodec._ONE_PASS} "
          f"values, parts of {claimcodec._CHUNK}", flush=True)
    if args.chunks:
        ref = claimcodec.encode(zs)
        default = claimcodec._CHUNK
        for lg in args.chunks:
            claimcodec._CHUNK = 1 << lg
            ts = []
            for _ in range(args.reps + 1):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                out = claimcodec.encode(zs)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
                same = out == ref
                del out
            med = statistics.median(ts[1:])
            print(f"parts of 2**{lg}: median {med:.3f} s ({n / med / 1e6:.0f} M claims/s), same bytes: {same}, "
                  f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        claimcodec._CHUNK = default
    if args.no_profile:
        return
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        claimcodec.encode(zs)
        torch.cuda.synchronize()
    ka = prof.key_averages()
    print(ka.table(sort_by="self_cpu_time_total", row_limit=args.rows))
    print(ka.table(sort_by="self_cuda_time_total", row_limit=args.rows))


if __name__ == "__main__":
    main()

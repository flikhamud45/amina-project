"""Microbenchmarks of V1's GKR prover on the GPU (``V1_SPEC.md`` Sec. 6.3, MB2-MB4): one instance of ``2^n`` leaves
built from random residues, proved with ``gkr_triton.prove_leaves``; the time per instance, per leaf and per
round, and the peak GPU memory.  ``--check`` also runs the eager prover (small ``n``) and compares transcripts.

    cd code && PYTHONPATH=src python experiments/8_v1/gkr_bench.py --n 20 22 24 --check 20
"""
from __future__ import annotations

import argparse
import json
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, nargs="+", default=[20, 22, 24])
    ap.add_argument("--T", type=int, default=2048)
    ap.add_argument("--check", type=int, nargs="*", default=[])
    ap.add_argument("--reps", type=int, default=2)
    args = ap.parse_args()
    from pvi.fullcheck import extfield as ef
    from pvi.fullcheck import gkr_triton as gt
    from pvi.fullcheck import logup, logup_gkr as gkr
    from pvi.fullcheck.field import P
    from pvi.fullcheck.protocol import Challenger

    def ch():
        c = Challenger(fiat_shamir=True)
        c.absorb(b"s", b"bench")
        return c

    for n in args.n:
        n_col = (args.T - 1).bit_length()
        rows = 1 << (n - n_col)
        g = torch.Generator().manual_seed(n)
        w = torch.randint(2, 1 << 12, (rows, 1), generator=g).expand(rows, args.T).contiguous()
        d = (torch.rand(rows, args.T, generator=g) * w).floor().to(torch.int64)
        alpha = torch.randint(0, P, (ef.D,), generator=g)
        times = []
        for _ in range(args.reps + 1):
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            t0 = time.perf_counter()
            tr = gt.prove_leaves(d, w, n_col, n, alpha, ch(), "cut/0/")
            torch.cuda.synchronize()
            times.append(time.perf_counter() - t0)
        best = min(times[1:])
        if gt._PROFILE:
            gt.PROFILE.clear()
            gt.prove_leaves(d, w, n_col, n, alpha, ch(), "cut/0/")
        out = {"n": n, "leaves": 1 << n, "T": args.T, "s": best, "ns_per_leaf": best / (1 << n) * 1e9,
               "profile": {k: round(v, 4) for k, v in gt.PROFILE.items()},
               "rounds": n * (n - 1) // 2, "first_s": times[0], "peak_gb": torch.cuda.max_memory_allocated() / 2**30,
               "gpu": torch.cuda.get_device_name(0)}
        if n in args.check:
            t0 = time.perf_counter()
            ref = gkr.prove(*logup.leaves(d, w, n_col, n, alpha), ch(), "cut/0/")
            out["eager_s"] = time.perf_counter() - t0
            out["same_transcript"] = ref.to_bytes() == tr.to_bytes()
        print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()

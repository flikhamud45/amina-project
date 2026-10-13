"""V1's GKR prover on the GPU on its own: the time per instance and per leaf, and the peak GPU memory.

Produces the extended version's GKR time per leaf (``app:eval-gkr-leaf`` and the V1 prover paragraph of
``ext_v1.tex``: "an instance of 2^25 leaves then takes 0.52 s (15.6 ns per leaf)" on the RTX 2080 Ti).

    python -m experiments.v1.gkr_bench --n 20 22 24 25 --check 20
    python -m experiments.v1.gkr_bench --n 22 --gkr-profile          # with the per-phase profile

For each ``n``, one instance of ``2^n`` leaves laid out as V1's cut ops lay them out (rows of ``--T`` positions, a
random window width per row, each value below its width) is proved with ``gkr_triton.prove_leaves`` under
Fiat--Shamir, ``--reps`` times after one untimed proof (the first proof also compiles the kernels: ``first_s``);
``s`` is the fastest timed proof and ``ns_per_leaf`` its time over ``2^n``.  With the implementation option
``--gkr-profile`` one more proof runs with every phase synchronised and timed (``profile``, seconds per phase); the
timed proofs never are.  ``--check n ...`` also runs the PyTorch prover (:func:`pvi.v1.gkr.prove`) on those sizes and
compares the transcripts byte for byte (``same_transcript``; small ``n`` only: it holds every layer in memory).

**Record** (one JSON line per ``n``, on stdout): ``n``, ``leaves``, ``T``, ``s``, ``ns_per_leaf``, ``first_s``,
``rounds``, ``profile``, ``peak_gb``, ``gpu``, ``env``; with ``--check`` ``eager_s`` and ``same_transcript``.

Hardware and time: a GPU with Triton (exit status 2 without one); seconds per size up to ``n = 25`` (5.8 GB at
``2^25`` leaves); the package's instances of ``2^27`` leaves need about 22 GB.
"""
from __future__ import annotations

import argparse
import sys
import time

import torch

from experiments.common import cli, records
from pvi import options


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, nargs="+", default=[20, 22, 24], help="instance sizes: 2^n leaves")
    ap.add_argument("--T", type=int, default=2048, help="positions per row (a query's tokens)")
    ap.add_argument("--check", type=int, nargs="*", default=[],
                    help="sizes n whose transcript is also computed by the PyTorch prover and compared")
    ap.add_argument("--reps", type=int, default=2, help="timed proofs per size, after one untimed proof")
    ap.add_argument("--seed", type=int, default=None, help="the leaves' seed (default: n)")
    cli.add_runtime_arguments(ap, threads=8, verifier_device=False)
    return ap


def main() -> int:
    args = build_parser().parse_args()
    if not (torch.cuda.is_available() and torch.device(args.device).type == "cuda"):
        print("the Triton GKR prover needs a GPU (--device cuda)", file=sys.stderr)
        return 2
    profile = args.gkr_profile
    args.gkr_profile = False                      # the timed proofs are never synchronised per phase
    cli.apply_runtime(args)
    from pvi.core import extfield as ef
    from pvi.core.field import P
    from pvi.v0.transcript import Challenger
    from pvi.v1 import gkr, logup
    from pvi.v1 import gkr_triton as gt

    dev = torch.device(args.device)
    env = records.environment(dev)

    def ch():
        c = Challenger(fiat_shamir=True)
        c.absorb(b"s", b"bench")
        return c

    for n in args.n:
        n_col = (args.T - 1).bit_length()
        rows = 1 << (n - n_col)
        g = torch.Generator().manual_seed(n if args.seed is None else args.seed)
        w = torch.randint(2, 1 << 12, (rows, 1), generator=g).expand(rows, args.T).contiguous()
        d = (torch.rand(rows, args.T, generator=g) * w).floor().to(torch.int64)
        alpha = torch.randint(0, P, (ef.D,), generator=g)
        times = []
        for _ in range(args.reps + 1):
            torch.cuda.synchronize(dev)
            torch.cuda.reset_peak_memory_stats(dev)
            t0 = time.perf_counter()
            tr = gt.prove_leaves(d, w, n_col, n, alpha, ch(), "cut/0/", device=dev)
            torch.cuda.synchronize(dev)
            times.append(time.perf_counter() - t0)
        best = min(times[1:]) if args.reps else times[0]
        peak = torch.cuda.max_memory_allocated(dev) / 2 ** 30
        gt.PROFILE.clear()
        if profile:
            with options.using(gkr_profile=True):
                gt.prove_leaves(d, w, n_col, n, alpha, ch(), "cut/0/", device=dev)
        out = {"n": n, "leaves": 1 << n, "T": args.T, "s": best, "ns_per_leaf": best / (1 << n) * 1e9,
               "profile": {k: round(v, 4) for k, v in gt.PROFILE.items()},
               "rounds": n * (n - 1) // 2, "first_s": times[0], "peak_gb": peak,
               "gpu": torch.cuda.get_device_name(dev), "env": env}
        if n in args.check:
            t0 = time.perf_counter()
            ref = gkr.prove(*logup.leaves(d, w, n_col, n, alpha), ch(), "cut/0/")
            out["eager_s"] = time.perf_counter() - t0
            out["same_transcript"] = ref.to_bytes() == tr.to_bytes()
        records.emit(out)
    return 0


if __name__ == "__main__":
    sys.exit(main())

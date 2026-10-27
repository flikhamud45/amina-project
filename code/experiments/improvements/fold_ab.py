"""The Kpre client's one-time precomputation (``Verifier.precompute``: ``u = chi A`` for every weight matrix): float64
limb products against the native exact field product (CPU) or int8 GEMMs (GPU), on the same challenges.

The client's key "12.7 s for Llama-2-7B on 8 Xeon threads" of ``sections/ext_discussion.tex`` (its float64
counterpart is the "before" of this A/B; the GPU path is measured alongside).

    python -m experiments.improvements.fold_ab --model llama2-7b --devices cpu cuda --reps 3
    python -m experiments.improvements.fold_ab --model gpt2 --layers 1 --reps 1 --threads 2      # a laptop check
    python -m experiments.improvements.fold_ab --model llama2-7b --real meta-llama/Llama-2-7b-hf --devices cpu cuda \\
        --reps 3

Builds the decoder as ``bench.py`` does (random weights, ``--seed``; ``--layers`` blocks, 0: all of them; the last one
pruned; 11-bit attention probabilities, ``--prob-max``, which the precomputation does not read) and runs
``Verifier.precompute`` in mode Kpre on each of ``--devices``, ``--reps`` times after one untimed run (uploads, the
native library's build), with the float64 limb products (the implementation option ``int8_fold`` of
:mod:`pvi.options` off) and with the default path (on: the native field product on a CPU, which needs the native
kernels, else the float64 products; int8 GEMMs on a GPU whose int8 GEMMs pass their self-check), on the same
challenges (``--challenge-seed``).  Every ``u`` must equal the float64 path's.

``--real`` (a Hugging Face checkpoint; ``--model`` names its shapes): instead of random weights, the integer graph of
the checkpoint's first ``--layers`` blocks (:mod:`experiments.common.real_weights`; OPT: the certified recipe, or
``--recipe legacy`` with ``--smooth`` and ``--calib-seq``, which it then needs: there is no prompt length to take it
from; a Llama-family checkpoint: ``hf_llama.MAIN_RECIPE`` with its fixed public prefix).  The precomputation reads the
weights and the challenges only, so no query is loaded.  Refused: shapes without an integer builder of a real
checkpoint (GPT-2) and ``--prob-max`` (the recipe fixes the scale).

Record (one JSON line per device and path ``float64`` / ``default``): ``model``, ``real`` (the checkpoint; None for
random weights), ``recipe``, ``smooth``, ``calib_seq`` (:func:`experiments.common.real_weights.fields`), ``layers``,
``precompute_s`` (the median), ``all_s``, the number of weight matrices and their entries, ``equal_to_float64``, the
repetitions ``reps`` of the security parameters, the threads, ``prob_max`` (the graph's attention probabilities'
scale), the implementation status (``native``) and ``env``.  The full Llama-2-7B holds its 6.7 GB of int8 weights in
host memory; its float64 path took 107 s per run on 8 threads of a Xeon Silver 4114, the native path 12.7 s.  With
``--real`` the checkpoint is loaded in fp32 and calibrated on the CPU first (host memory: its fp32 weights, 27 GB for
Llama-2-7B, next to the integer graph).
"""
from __future__ import annotations

import argparse
import statistics
import sys
import time

import torch

from experiments.common import cli, real_weights, records
from experiments.common.models import decoder_config
from experiments.common.paths import CODE
from pvi import options


def setup(args, cfg):
    """``(graph, real)``: the graph of ``--layers`` blocks and the checkpoint's load (None for random weights:
    ``bench.py``'s graph).  The precomputation needs no query: the checkpoint is loaded with none (and with the legacy
    recipe's ``--calib-seq`` as its window length)."""
    if args.real:
        real = real_weights.load(args, cfg, seq=args.calib_seq or 1, n_queries=0, layers=[args.layers])
        graph = real_weights.graph(real, args, args.layers, args.prune_last)
        del real.model
        return graph, real
    from pvi.models.decoder import build_decoder
    return build_decoder(cfg, n_layers=args.layers or cfg.n_layers, calib_tokens=cli.CALIB_TOKENS, seed=args.seed,
                         prune_last=args.prune_last, prob_max=args.prob_max), None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="llama2-7b", help="a decoder of pvi.models.decoder.CONFIGS")
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: the full model)")
    ap.add_argument("--devices", nargs="+", default=["cpu"], help="the verifier's devices (cpu, cuda)")
    ap.add_argument("--reps", type=int, default=3, help="timed runs per path, after one untimed run")
    ap.add_argument("--fs", action=argparse.BooleanOptionalAction, default=cli.PAPER.fiat_shamir,
                    help="Fiat-Shamir's repetitions (7 instead of 5 at lambda 128)")
    ap.add_argument("--lam", type=int, default=cli.PAPER.lam, help="security level in bits")
    ap.add_argument("--prune-last", action=argparse.BooleanOptionalAction, default=cli.PAPER.prune_last,
                    help="the last block at the last position only")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed")
    ap.add_argument("--challenge-seed", type=int, default=5, help="the secret challenges' seed")
    cli.add_prob_max_argument(ap)
    real_weights.add_arguments(ap)
    cli.add_runtime_arguments(ap, threads=8, verifier_device=False)
    args = ap.parse_args()
    cfg = decoder_config(args.model)
    real_weights.check(args, cfg, layers=[args.layers], prob_max=args.prob_max)
    cli.apply_runtime(args)
    from pvi.kernels import native
    from pvi.models.decoder import prob_max_of
    from pvi.v0.params import params_for
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier

    sha = records.git_sha(CODE)              # before the build, which can outlast a checkout update
    graph, real = setup(args, cfg)
    params = params_for(args.lam, len(graph.mat_ops), fiat_shamir=args.fs)
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    entries = sum(op.weight.numel() for op in graph.mat_ops)
    ok_all = True
    for dev in args.devices:
        if dev.startswith("cuda") and not torch.cuda.is_available():
            continue
        env = records.environment(dev)
        ref = None
        for path, opts in (("float64", {"int8_fold": False}), ("default", {})):
            ts, pre = [], None
            with options.using(**opts):
                for _ in range(args.reps + 1):                 # the first run warms up (uploads, the library build)
                    v = Verifier(graph.public(), params, "Kpre", weights=weights, device=dev)
                    t0 = time.perf_counter()
                    v.precompute(Challenger(seed=args.challenge_seed))
                    ts.append(time.perf_counter() - t0)
                    pre = {k: b.cpu() for k, (_, b) in v._pre.items()}
                    del v
            if ref is None:
                ref = pre
            same = all(torch.equal(pre[k], ref[k]) for k in ref)
            ok_all &= same
            records.emit({"git": sha, "model": cfg.name, **real_weights.fields(args, real),
                          "layers": args.layers or cfg.n_layers, "device": dev,
                          "path": path, "fs": args.fs, "reps": params.reps, "matrices": len(pre), "entries": entries,
                          "precompute_s": statistics.median(ts[1:]) if args.reps else None,
                          "all_s": [round(t, 3) for t in ts], "equal_to_float64": same, "threads": args.threads,
                          "prob_max": prob_max_of(graph),
                          "native": native.available(), "env": env})
    return 0 if ok_all else 1


if __name__ == "__main__":
    sys.exit(main())

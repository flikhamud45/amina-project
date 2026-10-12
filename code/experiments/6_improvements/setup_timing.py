"""Plan 2 F6 (D4's remainder): the time of the one-time setup proof of an untrusted commitment
(``pvi.fullcheck.setup_proof``), prover and verifier.  The ``setup`` tier of ``slurm/sp2027.sh``.

    cd code && PYTHONPATH=src python experiments/6_improvements/setup_timing.py --model llama2-7b --policy auto \\
        --amortise 100 --label setup
    cd code && PYTHONPATH=src python experiments/6_improvements/setup_timing.py --model tiny-llama --device cpu \\
        --threads 2                                                                       # a laptop check

Builds the decoder as ``perf.py --prune-last on`` does (random weights, seed 0; ``--layers`` blocks, 0: all of
them; ``tiny-*``/``small-*``: ``serve_ab.TINY``'s shapes), commits it under ``--policy`` on ``--device``
(``commit_s``), derives the setup proof's parameters for ``--amortise`` amortised queries (``setup_params``: ``s``,
the radii, ``t0`` and the per-query ``t`` of each tree, at ``--lam`` with the per-op column budget of
``plans.column_bits``, plus 64 bits under ``--fiat-shamir``), and times ``prove_setup`` (the committer: the ``s``
combinations of every committed matrix's rows on ``--device``, then ``t0`` opened columns and one multiproof per
tree) and ``verify_setup`` (the client, on the CPU: the paths and the code check of every opened column), each
``--reps`` times after one untimed run; every verification must accept.  The setup proof is non-interactive
(Fiat-Shamir over the graph digest and the roots), so its bytes and challenges are the same in every repetition.
One JSON line per ``--amortise`` value, ``"kind": "setup"``: ``commit_s``, ``prove_s`` / ``verify_s`` (median and
``[min, max]``), ``proof_bytes`` (``SetupProof.nbytes``), ``s``, ``t0`` and ``t`` per tree, the per-query column
bytes under the honest and the untrusted bound, peak host (VmHWM, Linux) and GPU memory.  The exit status is 1 if a
verification rejected.
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

CODE = Path(__file__).resolve().parents[2]   # code/
RATE = 4


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, CODE / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: the full model)")
    ap.add_argument("--policy", default="auto", help="the commitment plan (pvi.fullcheck.plans)")
    ap.add_argument("--lam", type=float, default=128)
    ap.add_argument("--amortise", type=int, nargs="+", default=[100])
    ap.add_argument("--fiat-shamir", action="store_true", help="the per-query budget of Fiat-Shamir queries")
    ap.add_argument("--reps", type=int, default=1, help="timed repetitions after an untimed one")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    gb = _load("gen_bench", "experiments/6_improvements/gen_bench.py")
    from pvi.fullcheck import setup_proof as sp
    from pvi.fullcheck.analytic import decoder_shapes
    from pvi.fullcheck.plans import column_bits, exact_columns
    from pvi.fullcheck.protocol import commit_graph
    from pvi.fullcheck.transformer import build_decoder

    device = torch.device(args.device)
    cfg = gb.config(args.model)
    n_layers = args.layers or cfg.n_layers
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=CODE).stdout.strip()
    gb.reset_peaks(device)
    t0 = time.perf_counter()
    graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=32, seed=0, prune_last=True)
    build_s = time.perf_counter() - t0
    shapes = decoder_shapes(cfg, prune_last=True)
    t0 = time.perf_counter()
    coms = commit_graph(graph, RATE, device=device, policy=args.policy, model_ops=shapes,
                        cache=False)                  # this tier measures the setup itself: never the setup cache
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    commit_s = time.perf_counter() - t0
    if coms.plan is None:
        raise SystemExit("--policy paper commits every op on a tree of its own; the setup proof is written for "
                         "commitment plans (their trees): pass a plan, e.g. auto")
    commit_peak = gb.peaks(device)
    digest = graph.digest()
    bits = column_bits(args.lam, len(shapes), args.fiat_shamir)
    ok = True
    for amortise in args.amortise:
        params = sp.setup_params(coms.publics, coms.group_publics, args.lam, bits, amortise=amortise)
        gb.reset_peaks(device)
        prove, verify, nbytes = [], [], None
        for rep in range(args.reps + 1):              # the first repetition is untimed
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            t0 = time.perf_counter()
            proof = sp.prove_setup(coms.groups, coms.publics, digest, params, device=device)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            tp = time.perf_counter() - t0
            t0 = time.perf_counter()
            good = sp.verify_setup(coms.publics, coms.group_publics, digest, params, proof)
            tv = time.perf_counter() - t0
            ok = ok and bool(good)
            nbytes = proof.nbytes()
            if rep:
                prove.append(tp)
                verify.append(tv)
            proof = None
        coded = {m.name: m for m in coms.plan.coded}
        rows = {g: sum(coded[m].n_rows for m in members) for g, members in coms.plan.groups}
        honest = {g: max(exact_columns(coded[m].row_length, coded[m].n_points, bits) for m in members)
                  for g, members in coms.plan.groups}
        line = {"kind": "setup", "label": args.label, "git": sha, "model": args.model, "layers": args.layers or None,
                "n_layers": n_layers, "policy": args.policy, "chosen": coms.plan.policy, "lam": args.lam,
                "fiat_shamir": args.fiat_shamir, "amortise": amortise, "device": args.device, "threads": args.threads,
                "build_s": build_s, "commit_s": commit_s, "reps": len(prove), "accepted": ok,
                "prove_s": statistics.median(prove) if prove else None,
                "prove_s_range": [min(prove), max(prove)] if prove else None,
                "verify_s": statistics.median(verify) if verify else None,
                "verify_s_range": [min(verify), max(verify)] if verify else None,
                "proof_bytes": nbytes, "s": params.s, "t0": dict(params.t0), "t": dict(params.t),
                "trees": len(rows), "query_columns_bytes_honest": sum(4 * honest[g] * rows[g] for g in rows),
                "query_columns_bytes_untrusted": sum(4 * dict(params.t)[g] * rows[g] for g in rows),
                "commit_peak": commit_peak, "peak": gb.peaks(device), "host": platform.node(),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
        print(json.dumps(line), flush=True)
    if not ok:
        print("FAILED: the honest setup proof was rejected", file=sys.stderr, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

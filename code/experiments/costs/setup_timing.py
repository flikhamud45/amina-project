"""The time of the one-time setup proof of an untrusted commitment (:mod:`pvi.commit.setup_proof`), prover and verifier.

The setup proof (Sec. ``sec:setup-proof``; its bytes are ``tab:setup``, from ``setup_bytes.py``) lets a client accept
a commitment it did not compute.  This script measures what the extended version reports next to ``tab:setup``: the
committer's and the client's time, on a random-weight decoder with the paper's shapes.  The run package's ``setup``
tier runs it on GPT-2 and Llama-2-7B.

    python -m experiments.costs.setup_timing --model llama2-7b --amortise 100 1000 --label setup
    python -m experiments.costs.setup_timing --model tiny-llama --device cpu --threads 2      # a laptop check

Builds the decoder as ``bench.py`` does (random weights, ``--seed``; ``--layers`` blocks, 0: all of them; the last
block pruned; ``tiny-*`` / ``small-*``: the shapes of :data:`experiments.common.models.TINY`), commits it under
``--policy`` (default the paper's plan ``cnn18c``) on ``--device`` (``commit_s``; never from the setup cache: this
measures the setup itself), derives the setup proof's parameters for each ``--amortise`` (``setup_params``: ``s``,
the radii, ``t0`` and the per-query ``t`` of each tree, at ``--lam`` with the per-op column budget of
``plans.column_bits``, 64 bits more with ``--fs``) and times ``prove_setup`` (the committer: the ``s`` combinations of
every committed matrix's rows on ``--device``, then ``t0`` opened columns and one multiproof per tree) and
``verify_setup`` (the client, on the CPU: the paths and the code check of every opened column), each ``--reps`` times
after one untimed run; every verification must accept.  The setup proof is non-interactive (Fiat--Shamir over the
graph digest and the roots), so its bytes and challenges are the same in every repetition.

Record (one JSON line per ``--amortise`` value, ``"kind": "setup"``): ``commit_s``, ``prove_s`` / ``verify_s``
(median and ``[min, max]``), ``proof_bytes`` (``SetupProof.nbytes``), ``s``, ``t0`` and ``t`` per tree, the per-query
column bytes under the honest and the untrusted bound, peak host (Linux) and GPU memory, and ``env``.  The exit status
is 1 if a verification rejected.  Hardware and time: a GPU for the committer; the package gives GPT-2 a job limit of one
hour and Llama-2-7B one of three hours with 128 GB of host memory.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time

import torch

from experiments.common import cli, records
from experiments.common.models import decoder_config
from experiments.common.paths import CODE


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: the full model)")
    ap.add_argument("--policy", default=cli.PAPER.policy, help="the commitment plan (pvi.commit.plans; not per-op)")
    ap.add_argument("--lam", type=float, default=cli.PAPER.lam, help="security level in bits")
    ap.add_argument("--rate", type=int, default=cli.RATE, help="Reed-Solomon rate of the weight commitment")
    ap.add_argument("--amortise", type=int, nargs="+", default=[100], help="queries the setup proof is amortised over")
    ap.add_argument("--fs", action=argparse.BooleanOptionalAction, default=False,
                    help="the per-query column budget of Fiat-Shamir queries (default: interactive, as tab:setup)")
    ap.add_argument("--reps", type=int, default=1, help="timed repetitions after an untimed one")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed")
    ap.add_argument("--calib-tokens", type=int, default=cli.CALIB_TOKENS, help="calibration tokens of the build")
    ap.add_argument("--label", default="", help="a label copied into every record")
    cli.add_runtime_arguments(ap, threads=8, verifier_device=False)
    args = ap.parse_args()
    cli.apply_runtime(args)
    from pvi.commit import setup_proof as sp
    from pvi.commit.cost import decoder_shapes
    from pvi.commit.plans import column_bits, exact_columns
    from pvi.models.decoder import build_decoder
    from pvi.v0.graph_commitment import commit_graph

    device = torch.device(args.device)
    cfg = decoder_config(args.model)
    n_layers = args.layers or cfg.n_layers
    sha = records.git_sha(CODE)
    records.reset_peaks(device)
    t0 = time.perf_counter()
    graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=args.calib_tokens, seed=args.seed, prune_last=True)
    build_s = time.perf_counter() - t0
    shapes = decoder_shapes(cfg, prune_last=True)
    t0 = time.perf_counter()
    coms = commit_graph(graph, args.rate, device=device, policy=args.policy, model_ops=shapes, cache=None)
    records.sync(device)
    commit_s = time.perf_counter() - t0
    if coms.plan is None:
        raise SystemExit("--policy per-op commits every op on a tree of its own; the setup proof is written for "
                         "commitment plans (their trees): pass a plan, e.g. cnn18c")
    commit_peak = records.peaks(device)
    env = records.environment(device)
    digest = graph.digest()
    bits = column_bits(args.lam, len(shapes), args.fs)
    ok = True
    for amortise in args.amortise:
        params = sp.setup_params(coms.publics, coms.group_publics, args.lam, bits, amortise=amortise)
        records.reset_peaks(device)
        prove, verify, nbytes = [], [], None
        for rep in range(args.reps + 1):              # the first repetition is untimed
            records.sync(device)
            t0 = time.perf_counter()
            proof = sp.prove_setup(coms.groups, coms.publics, digest, params, device=device)
            records.sync(device)
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
        records.emit({"kind": "setup", "label": args.label, "git": sha, "model": args.model,
                      "layers": args.layers or None, "n_layers": n_layers, "policy": args.policy,
                      "chosen": coms.plan.policy, "lam": args.lam, "fiat_shamir": args.fs, "amortise": amortise,
                      "device": args.device, "threads": args.threads, "build_s": build_s, "commit_s": commit_s,
                      "reps": len(prove), "accepted": ok,
                      "prove_s": statistics.median(prove) if prove else None,
                      "prove_s_range": [min(prove), max(prove)] if prove else None,
                      "verify_s": statistics.median(verify) if verify else None,
                      "verify_s_range": [min(verify), max(verify)] if verify else None,
                      "proof_bytes": nbytes, "s": params.s, "t0": dict(params.t0), "t": dict(params.t),
                      "trees": len(rows), "query_columns_bytes_honest": sum(4 * honest[g] * rows[g] for g in rows),
                      "query_columns_bytes_untrusted": sum(4 * dict(params.t)[g] * rows[g] for g in rows),
                      "commit_peak": commit_peak, "peak": records.peaks(device), "host": platform.node(),
                      "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "env": env})
    if not ok:
        print("FAILED: the honest setup proof was rejected", file=sys.stderr, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

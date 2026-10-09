"""Byte model of the untrusted-commitment setup proof (``pvi.fullcheck.setup_proof``) on the real model shapes.

    cd code && PYTHONPATH=src python experiments/6_improvements/setup_bytes.py --model gpt2 llama2-7b opt-6.7b llama2-13b

For each model's optimised commitment plan (``--policy``, default ``auto``) and security level, prints the
setup proof's size (the combinations ``s x m`` per matrix, ``t0`` opened columns per tree, a multiproof bound of
``t0 log2(n)`` hashes per tree), its column count per tree, and the per-query column bytes under the honest
bound and under the untrusted bound the setup proof certifies (``setup_params``: radii minimising
``t0 / amortise + t``).  No weights are needed: everything follows from the public shapes.
"""
from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace

from pvi.fullcheck import setup_proof as sp
from pvi.fullcheck.analytic import decoder_shapes
from pvi.fullcheck.plans import column_bits, exact_columns, plan_commitment
from pvi.fullcheck.transformer import CONFIGS


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", default=["gpt2", "llama2-7b", "opt-6.7b", "llama2-13b"])
    ap.add_argument("--policy", default="auto")
    ap.add_argument("--lam", type=float, default=128)
    ap.add_argument("--amortise", type=int, nargs="+", default=[1000])
    ap.add_argument("--fiat-shamir", action="store_true")
    args = ap.parse_args()
    for name in args.model:
        shapes = decoder_shapes(CONFIGS[name], prune_last=True)
        plan = plan_commitment(shapes, args.policy, rate=4)
        coded = {m.name: m for m in plan.coded}
        publics = {m.name: SimpleNamespace(tag=m.name.encode(), n_rows=m.n_rows, row_length=m.row_length,
                                           n_points=m.n_points) for m in coded.values()}
        groups = {g: SimpleNamespace(members=members, n_points=coded[members[0]].n_points, tag=g.encode(),
                                     root=bytes(32)) for g, members in plan.groups}
        bits = column_bits(args.lam, len(shapes), args.fiat_shamir)
        honest = {g: max(exact_columns(coded[m].row_length, coded[m].n_points, bits) for m in members)
                  for g, members in plan.groups}
        rows = {g: sum(coded[m].n_rows for m in members) for g, members in plan.groups}
        honest_bytes = sum(4 * honest[g] * rows[g] for g in rows)
        for amortise in args.amortise:
            p = sp.setup_params(publics, groups, args.lam, bits, amortise=amortise)
            t0, tq = dict(p.t0), dict(p.t)
            setup = sum(4 * p.s * coded[m].row_length for m in coded)
            setup += sum(4 * t0[g] * rows[g] + 32 * t0[g] * int(math.log2(groups[g].n_points)) for g in rows)
            query = sum(4 * tq[g] * rows[g] for g in rows)
            print(json.dumps({"model": name, "policy": plan.policy, "lam": args.lam, "fiat_shamir": args.fiat_shamir,
                              "amortise": amortise, "s": p.s, "trees": len(rows),
                              "setup_MB": round(setup / 1e6, 1), "t0": t0,
                              "query_columns_MB_honest": round(honest_bytes / 1e6, 2),
                              "query_columns_MB_untrusted": round(query / 1e6, 2),
                              "query_factor": round(query / honest_bytes, 3), "t_honest": honest, "t_untrusted": tq}),
                  flush=True)


if __name__ == "__main__":
    main()

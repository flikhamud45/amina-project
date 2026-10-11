"""Plan 2 TH-b: every number of the paper's concrete-security table (``tab:params``, ``sections/app_params.tex``).

    cd code && PYTHONPATH=src python experiments/6_improvements/params_table.py --model llama2-7b --policy auto

Everything follows from the public shapes (``analytic.decoder_shapes``, last block pruned) and the commitment plan,
through the functions the protocol itself uses (``plans.column_bits``, ``protocol.params_for``,
``setup_proof.setup_params``), so the table cannot drift from the code:

* v0 (Theorem ``thm:sound``): the checked weight operations ``|J|``; ``beta`` (interactive, and plus 64 under
  Fiat-Shamir); ``r``; the Freivalds term; the smallest tree's ``t``; the opened column bytes per query.
* V1 (Theorem ``thm:v1``) at ``--seq`` tokens: the lookup term ``(N_leaf + |V_tab|) / (p^8 - p^2)`` and the GKR terms
  ``sum_gamma (1 + sum_{k<n_gamma} (3k + 2)) / p^8``. ``N_leaf``, the instances and ``|V_tab|`` come from measured
  V1 cells (``--v1-cells``: ``v1_bytes.py`` records at two block counts), extended per block: every block has the
  same shapes, so the leaves of the full model are exact; the instance count assumes each instance holds at most
  ``2^lmax`` leaves, and ``|V_tab|`` grows by the per-block difference (an upper bound for the term).
* The setup proof (Theorem ``thm:untrusted``): ``beta_0`` over the coded matrices, ``s``, and per tree the radius
  fraction ``f``, ``e`` and ``t0`` at each ``--amortise``.

Prints one JSON object and, with ``--tex``, the table rows.
"""
from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace

from pvi.fullcheck import setup_proof as sp
from pvi.fullcheck.analytic import decoder_shapes
from pvi.fullcheck.field import P
from pvi.fullcheck.plans import column_bits, column_error_log2, exact_columns, plan_commitment
from pvi.fullcheck.protocol import params_for
from pvi.fullcheck.transformer import CONFIGS

LOG2_P = math.log2(P)


def _gkr_log2(n: int) -> float:
    """``log2`` of ``eps_GKR(n) = (1 + sum_{k=1}^{n-1} (3k + 2)) / p^8``."""
    return math.log2(1 + sum(3 * k + 2 for k in range(1, n))) - 8 * LOG2_P


def _v1_counts(path: str | None, n_layers: int, lmax: int, defaults: dict) -> dict:
    """``N_leaf``, instances and ``|V_tab|`` of the full model from two measured cells (``layers`` a < b)."""
    cells = defaults
    if path:
        cells = {}
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            if "n_leaves" in r:
                cells[r["layers"]] = (r["n_leaves"], r["n_instances"], r["table"])
    a, b = sorted(cells)[:2]
    (na, ia, ta), (nb, ib, tb) = cells[a], cells[b]
    k = (n_layers - b) / (b - a)
    leaves = nb + k * (nb - na)
    return {"cells": {a: cells[a], b: cells[b]}, "N_leaf": int(leaves),
            "instances": max(int(round(ib + k * (ib - ia))), math.ceil(leaves / 2 ** lmax)),
            "table": int(tb + k * (tb - ta))}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama2-7b")
    ap.add_argument("--policy", default="auto")
    ap.add_argument("--lam", type=float, default=128)
    ap.add_argument("--amortise", type=int, nargs="+", default=[100, 1000])
    ap.add_argument("--lmax", type=int, default=24)
    ap.add_argument("--v1-cells", default=None, help="v1_bytes.py records at two block counts (JSON lines)")
    ap.add_argument("--tex", action="store_true")
    args = ap.parse_args()
    cfg = CONFIGS[args.model]
    shapes = decoder_shapes(cfg, prune_last=True)
    plan = plan_commitment(shapes, args.policy, rate=4)
    coded = {m.name: m for m in plan.coded}
    rows = {g: sum(coded[m].n_rows for m in members) for g, members in plan.groups}
    out = {"model": args.model, "policy": plan.policy, "lam": args.lam, "ops": len(shapes), "coded": len(coded),
           "trees": len(rows)}

    # v0, interactive and Fiat-Shamir
    for fs in (False, True):
        key = "fs" if fs else "int"
        params = params_for(args.lam, len(shapes), plan=plan, fiat_shamir=fs)
        cols = dict(params.group_columns)
        out[key] = {"beta": round(column_bits(args.lam, len(shapes), fs), 2), "r": params.reps,
                    "freivalds_log2": round(-params.reps * LOG2_P, 1), "t_min": min(cols.values()),
                    "t_max": max(cols.values()),
                    "columns_MB": round(sum(4 * cols[g] * rows[g] for g in rows) / 1e6, 1)}

    # V1 at 2,048 tokens (Llama-2-7B's measured 2- and 3-block cells, job 1007816, by default)
    v1 = _v1_counts(args.v1_cells, cfg.n_layers, args.lmax, {2: (103809024, 6, 41247), 3: (190840832, 11, 58336)})
    lookup = math.log2(v1["N_leaf"] + v1["table"]) - math.log2(P ** 8 - P ** 2)
    gkr = math.log2(v1["instances"]) + _gkr_log2(args.lmax)
    out["v1"] = dict(v1, cells={str(k): v for k, v in v1["cells"].items()}, lookup_log2=round(lookup, 1),
                     gkr_log2=round(gkr, 1), ext_log2=round(8 * LOG2_P, 1))

    # the setup proof
    publics = {m.name: SimpleNamespace(tag=m.name.encode(), n_rows=m.n_rows, row_length=m.row_length,
                                       n_points=m.n_points) for m in coded.values()}
    groups = {g: SimpleNamespace(members=members, n_points=coded[members[0]].n_points, tag=g.encode(),
                                 root=bytes(32)) for g, members in plan.groups}
    qbits = column_bits(args.lam, len(shapes), False)
    out["setup"] = {"beta0": round(args.lam + 64 + math.log2(2 * len(coded)), 2)}
    for a in args.amortise:
        p = sp.setup_params(publics, groups, args.lam, qbits, amortise=a)
        e = dict(p.e)
        per_tree = {}
        for g, members in plan.groups:
            m0 = coded[members[0]]
            frac = 2 * e[members[0]] / max(1, m0.n_points - m0.row_length)
            per_tree[g] = {"n": m0.n_points, "k_max": max(coded[m].row_length for m in members),
                           "f": f"2^{round(math.log2(frac))}" if frac > 0 else "0",
                           "e": max(e[m] for m in members), "t0": dict(p.t0)[g], "t": dict(p.t)[g]}
        out["setup"][str(a)] = {"s": p.s, "trees": per_tree}
    print(json.dumps(out), flush=True)
    if args.tex:
        i, f, s = out["int"], out["fs"], out["setup"]
        print(f"% params_table.py --model {args.model} --policy {plan.policy}")
        print(f"checked weight operations |J| = {len(shapes)}; coded matrices |J_0| = {len(coded)}; trees {len(rows)}")
        print(f"beta = {i['beta']} / {f['beta']};  r = {i['r']} / {f['r']};  t (smallest tree) = {i['t_min']} / {f['t_min']};"
              f"  columns {i['columns_MB']} / {f['columns_MB']} MB")
        print(f"V1: lookup 2^{out['v1']['lookup_log2']}, GKR 2^{out['v1']['gkr_log2']} (N_leaf {v1['N_leaf']:,}, "
              f"{v1['instances']} instances, |V_tab| {v1['table']:,})")
        print(f"setup: beta_0 = {s['beta0']}; s = " + " / ".join(str(s[str(a)]['s']) for a in args.amortise))
        for a in args.amortise:
            print(f"  a = {a}: " + "; ".join(f"{g}: n={t['n']} f={t['f']} e={t['e']} t0={t['t0']}"
                                            for g, t in s[str(a)]["trees"].items()))


if __name__ == "__main__":
    main()

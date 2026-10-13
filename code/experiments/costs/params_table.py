"""Every value of the concrete-security table ``tab:params`` (``sections/app_params.tex``).

    python -m experiments.costs.params_table --tex        # tab:params: Llama-2-7B, the plan cnn18c, a = 100, 1,000

Everything follows from the public shapes (``cost.decoder_shapes``, last block pruned) and the commitment plan
(``--policy``, default the paper's ``cnn18c``), through the functions the protocol itself uses
(``plans.column_bits``, ``params_for``, ``setup_proof.setup_params``), so the table cannot drift from the code:

* v0 (Theorem ``thm:sound``): the checked weight operations ``|J|``; ``beta`` (interactive, and plus 64 under
  Fiat--Shamir); ``r``; the Freivalds term; the smallest and largest tree's ``t``; the opened column bytes per query.
* V1 (Theorem ``thm:v1``) at 2,048 tokens: the lookup term ``(N_leaf + |V_tab|) / (p^8 - p^2)`` and the GKR terms
  ``sum_gamma (1 + sum_{k<n_gamma} (3k + 2)) / p^8``.  ``N_leaf``, the instances and ``|V_tab|`` come from two measured
  V1 cells of the model (``--v1-counts``, default Llama-2-7B's 2- and 3-block cells at 2,048 tokens, or the
  ``v1/v1_bytes.py`` records of ``--v1-cells``), extended per block: every block has the same shapes, so the leaves
  of the full model are exact; the instance count assumes each instance holds at most ``2^lmax`` leaves, and
  ``|V_tab|`` grows by the per-block difference (an upper bound for the term).
* The setup proof (Theorem ``thm:untrusted``): ``beta_0`` over the coded matrices, ``s``, per tree the radius fraction
  ``f``, ``e``, ``t0`` and ``t`` at each ``--amortise``, and the setup proof's size and the untrusted per-query column
  bytes (:func:`experiments.costs.setup_bytes.setup_model`, interactive queries).

Prints one JSON object and, with ``--tex``, the table's values as text lines.  Runs in seconds on a laptop CPU.
"""
from __future__ import annotations

import argparse
import json
import math
import sys

from experiments.common import cli
from experiments.costs.setup_bytes import plan_publics, setup_model
from pvi.commit import setup_proof as sp
from pvi.commit.cost import decoder_shapes
from pvi.commit.plans import column_bits, plan_commitment
from pvi.core.field import P
from pvi.models.decoder import CONFIGS
from pvi.v1.cut import LMAX
from pvi.v0.params import params_for

LOG2_P = math.log2(P)
V1_COUNTS = ("2:103809024:6:41247", "3:190840832:11:58336")
"""Llama-2-7B's measured V1 cells at 2,048 tokens (``v1/v1_bytes.py``): ``blocks:N_leaf:instances:|V_tab|``."""


def _gkr_log2(n: int) -> float:
    """``log2`` of ``eps_GKR(n) = (1 + sum_{k=1}^{n-1} (3k + 2)) / p^8``."""
    return math.log2(1 + sum(3 * k + 2 for k in range(1, n))) - 8 * LOG2_P


def _counts(specs) -> dict:
    """``{blocks: (N_leaf, instances, |V_tab|)}`` of ``blocks:N_leaf:instances:|V_tab|`` strings."""
    out = {}
    for spec in specs:
        b, *rest = (int(x) for x in spec.split(":"))
        if len(rest) != 3:
            raise SystemExit(f"--v1-counts: blocks:N_leaf:instances:table, not {spec!r}")
        out[b] = tuple(rest)
    return out


def _v1_counts(path: str | None, counts: dict, n_layers: int, lmax: int) -> dict:
    """``N_leaf``, instances and ``|V_tab|`` of the full model from two measured cells (``layers`` a < b)."""
    cells = counts
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
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="llama2-7b", help="a decoder of pvi.models.decoder.CONFIGS")
    ap.add_argument("--policy", default=cli.PAPER.policy, help="the commitment plan (pvi.commit.plans)")
    ap.add_argument("--lam", type=float, default=cli.PAPER.lam, help="security level in bits")
    ap.add_argument("--rate", type=int, default=cli.RATE, help="the base Reed-Solomon rate of the plan")
    ap.add_argument("--amortise", type=int, nargs="+", default=[100, 1000],
                    help="queries the setup proof is amortised over")
    ap.add_argument("--lmax", type=int, default=LMAX,
                    help="V1: at most 2^lmax leaves per GKR instance (default the library's and the run's)")
    ap.add_argument("--v1-counts", nargs=2, default=list(V1_COUNTS), metavar="B:NLEAF:INST:TABLE",
                    help="V1: two measured cells of the model, blocks:N_leaf:instances:|V_tab| (default Llama-2-7B's "
                         "2- and 3-block cells at 2,048 tokens)")
    ap.add_argument("--v1-cells", default=None,
                    help="V1: v1_bytes.py records at two block counts (JSON lines) instead of --v1-counts")
    ap.add_argument("--tex", action="store_true", help="also print the table's values as text lines")
    args = ap.parse_args()
    if args.model != "llama2-7b" and not args.v1_cells and args.v1_counts == list(V1_COUNTS):
        print(f"warning: the V1 rows use Llama-2-7B's measured cells, not {args.model}'s (--v1-counts, --v1-cells)",
              file=sys.stderr)
    cfg = CONFIGS[args.model]
    shapes = decoder_shapes(cfg, prune_last=True)
    plan = plan_commitment(shapes, args.policy, rate=args.rate)
    coded = {m.name: m for m in plan.coded}
    rows = {g: sum(coded[m].n_rows for m in members) for g, members in plan.groups}
    out = {"model": args.model, "policy": plan.policy, "lam": args.lam, "ops": len(shapes), "coded": len(coded),
           "trees": len(rows)}

    # v0, interactive and Fiat-Shamir
    for fs in (False, True):
        key = "fs" if fs else "int"
        params = params_for(args.lam, len(shapes), rate=args.rate, plan=plan, fiat_shamir=fs)
        cols = dict(params.group_columns)
        out[key] = {"beta": round(column_bits(args.lam, len(shapes), fs), 2), "r": params.reps,
                    "freivalds_log2": round(-params.reps * LOG2_P, 1), "t_min": min(cols.values()),
                    "t_max": max(cols.values()),
                    "columns_MB": round(sum(4 * cols[g] * rows[g] for g in rows) / 1e6, 1)}

    # V1 at 2,048 tokens
    v1 = _v1_counts(args.v1_cells, _counts(args.v1_counts), cfg.n_layers, args.lmax)
    lookup = math.log2(v1["N_leaf"] + v1["table"]) - math.log2(P ** 8 - P ** 2)
    gkr = math.log2(v1["instances"]) + _gkr_log2(args.lmax)
    out["v1"] = dict(v1, cells={str(k): v for k, v in v1["cells"].items()}, lookup_log2=round(lookup, 1),
                     gkr_log2=round(gkr, 1), ext_log2=round(8 * LOG2_P, 1))

    # the setup proof
    publics, groups = plan_publics(plan)
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
        size = setup_model(shapes, args.policy, args.lam, a, rate=args.rate)
        out["setup"][str(a)] = {"s": p.s, "trees": per_tree, "setup_MB": size["setup_MB"],
                                "query_columns_MB_untrusted": size["query_columns_MB_untrusted"]}
    print(json.dumps(out), flush=True)
    if args.tex:
        i, f, s = out["int"], out["fs"], out["setup"]
        print(f"% python -m experiments.costs.params_table --model {args.model} --policy {plan.policy}")
        print(f"checked weight operations |J| = {len(shapes)}; coded matrices |J_0| = {len(coded)}; trees {len(rows)}")
        print(f"beta = {i['beta']} / {f['beta']};  r = {i['r']} / {f['r']};  "
              f"t (smallest tree) = {i['t_min']} / {f['t_min']};  columns {i['columns_MB']} / {f['columns_MB']} MB")
        print(f"V1: lookup 2^{out['v1']['lookup_log2']}, GKR 2^{out['v1']['gkr_log2']} (N_leaf {v1['N_leaf']:,}, "
              f"{v1['instances']} instances, |V_tab| {v1['table']:,})")
        print(f"setup: beta_0 = {s['beta0']}; s = " + " / ".join(str(s[str(a)]['s']) for a in args.amortise))
        for a in args.amortise:
            print(f"  a = {a}: " + "; ".join(f"{g}: n={t['n']} f={t['f']} e={t['e']} t0={t['t0']}"
                                            for g, t in s[str(a)]["trees"].items()))
        print("setup proof " + " / ".join(f"{s[str(a)]['setup_MB']:,} MB" for a in args.amortise)
              + "; untrusted columns per query " + " / ".join(f"{s[str(a)]['query_columns_MB_untrusted']} MB"
                                                              for a in args.amortise))


if __name__ == "__main__":
    main()

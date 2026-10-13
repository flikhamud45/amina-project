"""The byte model of the untrusted-commitment setup proof (:mod:`pvi.commit.setup_proof`) on the paper's model shapes.

Produces ``tab:setup`` (``sections/ext_defence.tex``: the setup proof at ``a = 100`` and ``1,000`` amortised queries
and the opened columns per query, honest and untrusted, for GPT-2, Llama-2-7B, OPT-6.7B and Llama-2-13B), the
Llama-2-7B setup rows of ``tab:params`` and the numbers of the paragraph after ``tab:setup`` (the setup proof's
breakdown into opened columns, combined rows and paths, and the columns a test of error ``2^-lam`` per matrix would
still open at the same radii).  No weights are needed: everything follows from the public shapes.

    python -m experiments.costs.setup_bytes                       # tab:setup: four models, cnn18c, a = 100, 1,000
    python -m experiments.costs.setup_bytes --model llama2-7b --amortise 100
    python -m experiments.costs.setup_bytes --fixed-bytes --policy cnn16c cnn18c R16c R64c \\
        --model gpt2 opt-1.3b llama2-7b llama2-13b qwen3-4b      # the fixed bytes of a mode-C query per plan

For each model (decoder shapes, last block pruned), commitment plan (``--policy``, default the paper's ``cnn18c``)
and ``--amortise`` value, one JSON line: the setup proof's size (the ``s`` combinations of every coded matrix's rows,
``t0`` opened columns per tree, a multiproof bound of ``t0 log2(n)`` hashes per tree), its column counts, the
per-query column bytes under the honest bound and under the untrusted bound the setup proof certifies
(``setup_params``: radii minimising ``t0 / amortise + t``), and two quantities of the remark on other proximity tests
(``rem:forced``): ``columns_MB_at_lam``, the setup's opened columns at the same radii with an error of ``2^-lam`` per
matrix instead of ``2^-beta0``, and per tree ``t0_with_replacement`` = ``beta0 / -log2((n - e - 1) / n)``, the count
of columns drawn with replacement, against ``t0``.  The per-query columns are those of interactive queries unless
``--fs`` (as ``tab:setup``).

``--fixed-bytes`` prints instead, per model, plan and challenge setting, the part of a mode-C proof that does not grow
with the prompt (``u``, the opened columns and the Merkle paths, from :func:`pvi.commit.cost.proof_bytes` at ``--seq``
tokens) and the plan's one-time encoded entries: the comparison behind the choice of the paper's plan (longer codewords
open fewer columns at the price of a longer one-time encoding).  ``--entries-per-s`` adds a setup-time estimate at a
measured encoding rate.  The paper reports neither number; the option documents the choice.

Runs in seconds on a laptop CPU.
"""
from __future__ import annotations

import argparse
import json
import math
from types import SimpleNamespace

from experiments.common import cli
from pvi.commit import setup_proof as sp
from pvi.commit.cost import decoder_shapes, proof_bytes
from pvi.commit.plans import column_bits, exact_columns, plan_commitment
from pvi.models.decoder import CONFIGS
from pvi.v0.params import params_for

MODELS = ("gpt2", "llama2-7b", "opt-6.7b", "llama2-13b")
"""The models of ``tab:setup``."""


def plan_publics(plan) -> tuple[dict, dict]:
    """``(publics, groups)``: the verifier's view of a plan's coded matrices and trees, as ``setup_params`` reads it
    (the shapes only; no roots)."""
    coded = {m.name: m for m in plan.coded}
    publics = {m.name: SimpleNamespace(tag=m.name.encode(), n_rows=m.n_rows, row_length=m.row_length,
                                       n_points=m.n_points) for m in coded.values()}
    groups = {g: SimpleNamespace(members=members, n_points=coded[members[0]].n_points, tag=g.encode(),
                                 root=bytes(32)) for g, members in plan.groups}
    return publics, groups


def setup_model(shapes, policy: str, lam: float, amortise: int, fs: bool = False, rate: int = cli.RATE) -> dict:
    """The setup proof's byte model for one plan and one amortisation (module docstring), in MB."""
    plan = plan_commitment(shapes, policy, rate=rate)
    if plan is None:
        raise SystemExit("--policy per-op commits every op on a tree of its own; the setup proof is written for "
                         "commitment plans")
    coded = {m.name: m for m in plan.coded}
    publics, groups = plan_publics(plan)
    bits = column_bits(lam, len(shapes), fs)
    honest = {g: max(exact_columns(coded[m].row_length, coded[m].n_points, bits) for m in members)
              for g, members in plan.groups}
    rows = {g: sum(coded[m].n_rows for m in members) for g, members in plan.groups}
    honest_bytes = sum(4 * honest[g] * rows[g] for g in rows)
    p = sp.setup_params(publics, groups, lam, bits, amortise=amortise)
    t0, tq, e = dict(p.t0), dict(p.t), dict(p.e)
    beta0 = lam + p.grinding_bits + math.log2(2 * len(publics))
    combined = sum(4 * p.s * coded[m].row_length for m in coded)          # the s combinations per matrix
    columns = sum(4 * t0[g] * rows[g] for g in rows)                      # t0 opened columns per tree
    paths = sum(32 * t0[g] * int(math.log2(groups[g].n_points)) for g in rows)   # multiproof bound
    at_lam = {g: max(sp.survivors(coded[m].n_points - e[m] - 1, coded[m].n_points, lam) for m in members)
              for g, members in plan.groups}
    with_replacement = {g: max(beta0 / -math.log2((coded[m].n_points - e[m] - 1) / coded[m].n_points)
                               for m in members) for g, members in plan.groups}
    query = sum(4 * tq[g] * rows[g] for g in rows)
    setup = combined + columns + paths
    return {"policy": plan.policy, "lam": lam, "fiat_shamir": fs, "amortise": amortise, "s": p.s,
            "beta0": round(beta0, 2), "trees": len(rows), "setup_MB": round(setup / 1e6, 1),
            "combined_rows_MB": round(combined / 1e6, 1), "columns_MB": round(columns / 1e6, 1),
            "paths_MB_bound": round(paths / 1e6, 1), "t0": t0,
            "query_columns_MB_honest": round(honest_bytes / 1e6, 2),
            "query_columns_MB_untrusted": round(query / 1e6, 2),
            "query_factor": round(query / honest_bytes, 3), "t_honest": honest, "t_untrusted": tq,
            "columns_MB_at_lam": round(sum(4 * at_lam[g] * rows[g] for g in rows) / 1e6, 1),
            "t0_with_replacement": {g: round(v, 1) for g, v in with_replacement.items()}}


def fixed_bytes(shapes, policy: str, lam: float, seq: int, fs: bool, rate: int = cli.RATE,
                entries_per_s: float | None = None) -> dict:
    """The part of a mode-C query that does not grow with the prompt, and the plan's one-time encoded entries."""
    plan = plan_commitment(shapes, policy, rate=rate, model_ops=shapes)
    entries = sum(m.n_points * m.n_rows for m in plan.matrices) if plan is not None else None
    params = params_for(lam, len(shapes), rate=rate, plan=plan, fiat_shamir=fs)
    b = proof_bytes(shapes, params, seq, plan=plan, rate=rate, mode="C")
    fixed = b["u"] + b["columns"] + b["paths"]
    return {"policy": policy, "fs": fs, "reps": params.reps, "fixed_MB": round(fixed / 1e6, 2),
            "u_MB": round(b["u"] / 1e6, 2), "columns_MB": round(b["columns"] / 1e6, 2),
            "paths_MB": round(b["paths"] / 1e6, 2), "setup_entries": entries,
            **({"setup_s_est": round(entries / entries_per_s) if entries else None} if entries_per_s else {})}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", nargs="+", default=list(MODELS), help="decoders (pvi.models.decoder.CONFIGS)")
    ap.add_argument("--policy", nargs="+", default=[cli.PAPER.policy], help="commitment plans (pvi.commit.plans)")
    ap.add_argument("--lam", type=float, default=cli.PAPER.lam, help="security level in bits")
    ap.add_argument("--rate", type=int, default=cli.RATE, help="the base Reed-Solomon rate of the plans")
    ap.add_argument("--amortise", type=int, nargs="+", default=[100, 1000],
                    help="queries the setup proof is amortised over")
    ap.add_argument("--fs", action=argparse.BooleanOptionalAction, default=False,
                    help="the per-query column budget of Fiat-Shamir queries (default: interactive, as tab:setup); "
                         "with --fixed-bytes both are printed")
    ap.add_argument("--fixed-bytes", action="store_true",
                    help="print the fixed bytes of a mode-C query per plan instead (module docstring)")
    ap.add_argument("--seq", type=int, default=64, help="with --fixed-bytes: the prompt tokens of the query")
    ap.add_argument("--entries-per-s", type=float, default=None,
                    help="with --fixed-bytes: a measured setup encoding rate (entries per second), for an estimate")
    args = ap.parse_args()
    for name in args.model:
        shapes = decoder_shapes(CONFIGS[name], prune_last=True)
        for policy in args.policy:
            if args.fixed_bytes:
                for fs in (False, True):
                    print(json.dumps({"model": name, **fixed_bytes(shapes, policy, args.lam, args.seq, fs, args.rate,
                                                                    args.entries_per_s)}), flush=True)
                continue
            for amortise in args.amortise:
                print(json.dumps({"model": name, **setup_model(shapes, policy, args.lam, amortise, args.fs,
                                                                 args.rate)}), flush=True)


if __name__ == "__main__":
    main()

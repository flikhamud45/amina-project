"""The prover's fold and column openings on exact int8 GEMMs against float64 products, on the same queries.

The rows "Opening and fold on int8 GEMMs (prover, C, T=64)" of ``tab:eval-improve`` (``sections/ext_evaluation.tex``)
and the body's "3.7--4.3x faster on Llama-2 blocks" (Sec. ``sec:eval-cost``).

    python -m experiments.improvements.open_ab --model llama2-7b --layers 1 4 --queries 7
    python -m experiments.improvements.open_ab --model llama2-13b --layers 1 --queries 7
    python -m experiments.improvements.open_ab --model gpt2 --queries 7

``WeightCommitment.fold`` and the ``columns_at`` of both layouts compute their products with
``field.weight_matmul_mod``: on a GPU whose ``torch._int_mm`` passes ``field.int8_ok``, as int8 GEMMs with
bit-identical results, else in float64 (``field.small_matmul_mod``).  This script builds the decoder as ``bench.py``
does (random weights, ``--seed``; ``--layers`` blocks of the model, the last one pruned), commits it under ``--policy``
(default the paper's plan ``cnn18c``) and runs ``--queries`` mode-C queries with ``run_query`` (the protocol flags of
:func:`experiments.common.cli.add_protocol_arguments`, default the paper's configuration), each once with the float64
products and once with the int8 ones, the order alternating; the first query warms both up.  Both runs of a query use
the same challenge seed, so they draw the same ``chi`` and the same columns: the script checks that they return the same
folded rows, opened columns and multiproofs, and that the verifier accepts.

The float64 path is the implementation option :data:`OPTION` of :mod:`pvi.options` turned off
(``options.using(int8_opening=False)``).

Record (one JSON line per model, block count and variant ``float64`` / ``int8``): the medians of the prover's parts
(``timings``: ``prove_forward``, ``prove_fold``, ``prove_open``, ...), their sum ``prove_s`` (the prover's total, the
table's value), ``open_fold_s`` (``prove_fold + prove_open``), ``vs_float64``, ``identical``, ``accepted`` of
``queries``, ``int8_ok``, ``prob_max``, the GPU and ``env``.  Needs a CUDA GPU with int8 GEMMs (an RTX 2080 Ti or
newer); minutes per build.
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys

import torch

from experiments.common import cli, records
from experiments.common.models import decoder_config
from experiments.common.paths import CODE
from pvi import options

OPTION = "int8_opening"
"""The implementation option that turns the prover's int8 fold and openings off (float64 products)."""
PARTS = ("prove_forward", "prove_lookups", "prove_fold", "prove_open", "prove_encode", "fs_hash")



def _same(a, b) -> bool:
    """The same folded rows ``{op: u}`` or openings ``{tree: (columns, multiproof)}``."""
    if a.keys() != b.keys():
        return False
    for k in a:
        x, y = a[k], b[k]
        if isinstance(x, tuple):
            if not (torch.equal(x[0], y[0]) and x[1] == y[1]):
                return False
        elif not torch.equal(x, y):
            return False
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True, help="a decoder of pvi.models.decoder.CONFIGS")
    ap.add_argument("--layers", type=int, nargs="+", default=[0], help="decoder blocks (0: the full model)")
    ap.add_argument("--seq", type=int, default=64, help="prompt tokens")
    ap.add_argument("--queries", type=int, default=7, help="queries per variant (the first warms up)")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed")
    ap.add_argument("--prompt-seed", type=int, default=1, help="the random prompts' seed")
    ap.add_argument("--challenge-seed", type=int, default=1000, help="query i draws its challenges with seed + i")
    cli.add_protocol_arguments(ap)
    cli.add_runtime_arguments(ap, threads=8, verifier_device=True)
    args = ap.parse_args()
    cli.apply_runtime(args)
    from pvi.commit.cost import decoder_shapes
    from pvi.core import field
    from pvi.models.decoder import build_decoder
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.verifier import Verifier

    variants = {"float64": {OPTION: False}, "int8": {}}
    device = torch.device(args.device)
    cfg = decoder_config(args.model)
    model_ops = decoder_shapes(cfg, prune_last=args.prune_last)
    sha, env = records.git_sha(CODE), records.environment(device)
    tokens = torch.randint(0, cfg.vocab, (args.queries + 1, args.seq),
                           generator=torch.Generator().manual_seed(args.prompt_seed))
    ok = True
    for n_layers in args.layers:
        graph = build_decoder(cfg, n_layers=n_layers or cfg.n_layers, calib_tokens=min(args.seq, cli.CALIB_TOKENS),
                              seed=args.seed, prune_last=args.prune_last, prob_max=args.prob_max)
        com = commit_graph(graph, args.rate, device=device, policy=args.policy, model_ops=model_ops,
                           cache=args.setup_cache)
        params = params_for(args.lam, len(model_ops), rate=args.rate, plan=com.plan, fiat_shamir=args.fs)
        v = Verifier(graph.public(), params, "C", publics=com.publics, groups=com.group_publics,
                     tables=com.table_publics, device=args.verifier_device, lean=args.lean)
        prover = Prover(graph, device=device, commitments=com, lean=args.lean)
        seen: dict[str, dict] = {}
        fold, open_ = prover.fold, prover.open

        def watched(fn, key):
            def call(*a, **kw):
                out = fn(*a, **kw)
                seen[key] = out
                return out
            return call

        prover.fold, prover.open = watched(fold, "fold"), watched(open_, "open")
        runs = {name: [] for name in variants}
        identical = True
        for i, q in enumerate(tokens.split(1)):
            order = list(variants) if i % 2 == 0 else list(variants)[::-1]
            got = {}
            for name in order:
                with options.using(**variants[name]):
                    out = run_query(prover, v, q, seed=args.challenge_seed + i, wire=args.wire)
                got[name] = dict(seen)
                if i:                                   # query 0 warms both variants up
                    runs[name].append(out)
            identical &= all(_same(got["float64"][k], got["int8"][k]) for k in ("fold", "open"))
        base = None
        for name, rs in runs.items():
            med = {k: statistics.median(r["timings"].get(k, 0.0) for r in rs) for k in PARTS}
            row = {"git": sha, "model": args.model, "layers": n_layers or cfg.n_layers, "seq": args.seq,
                   "policy": args.policy, "chosen": com.plan.policy if com.plan else None, "variant": name,
                   "fiat_shamir": args.fs, "wire": args.wire, "prob_max": args.prob_max, "queries": len(rs),
                   "accepted": sum(bool(r["accepted"]) for r in rs), "identical": identical,
                   "prove_s": sum(med.values()), "open_fold_s": med["prove_fold"] + med["prove_open"],
                   "timings": med, "device": str(device), "int8_ok": field.int8_ok(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "host": platform.node(), "env": env}
            if base is None:
                base = row
            else:
                row["vs_float64"] = {k: row["timings"][k] / base["timings"][k]
                                     for k in ("prove_fold", "prove_open") if base["timings"][k]}
                row["vs_float64"]["prove_s"] = row["prove_s"] / base["prove_s"]
                row["vs_float64"]["open_fold_s"] = row["open_fold_s"] / base["open_fold_s"]
            ok &= identical and row["accepted"] == row["queries"]
            records.emit(row)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

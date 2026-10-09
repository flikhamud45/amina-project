"""A/B of the prover's folds and column openings: float64 products against exact int8 GEMMs.

    cd code && PYTHONPATH=src python experiments/6_improvements/open_ab.py --model llama2-7b --layers 1 2 --queries 7
    cd code && PYTHONPATH=src python experiments/6_improvements/open_ab.py --model gpt2 --queries 7

``WeightCommitment.fold`` and the ``columns_at`` of both layouts compute their products with
``field.weight_matmul_mod``: on a GPU whose ``torch._int_mm`` passes ``field.int8_ok``, as int8 GEMMs
(``field.int8_weight_matmul``), else in float64 (``field.small_matmul_mod``, the code before).  This
script builds the decoder as bench.py does (random weights, seed 0, last block pruned), commits it
under ``--policy`` (default ``auto``, the paper's planning rule) and runs ``--queries`` mode-C queries
with ``run_query``, each once with the float64 products and once with the int8 ones, the order
alternating.  Both runs of a query use the same challenge seed, so they draw the same ``chi`` and
the same columns; the script checks that they return the same folded rows, opened columns and
multiproofs, and that the verifier accepts.  It prints one JSON line per (model, layers, variant):
the medians of ``prove_fold``, ``prove_open`` and the prover total, with ``vs_float64``.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import subprocess
import sys
from pathlib import Path

import torch

CODE = Path.cwd()   # run from code/


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, CODE / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[0], help="decoder blocks (0: full model)")
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--policy", default="auto")
    ap.add_argument("--lam", type=int, default=128)
    ap.add_argument("--fiat-shamir", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--queries", type=int, default=7)
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    bench = _load("bench", "experiments/4_defence_benchmark/bench.py")
    from pvi.fullcheck import commitment, field
    from pvi.fullcheck.analytic import decoder_shapes
    from pvi.fullcheck.protocol import Prover, Verifier, commit_graph, params_for, run_query
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    variants = {"float64": field.small_matmul_mod, "int8": field.weight_matmul_mod}
    device = torch.device(args.device)
    cfg = CONFIGS[args.model]
    model_ops = decoder_shapes(cfg, prune_last=True)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=CODE).stdout.strip()
    tokens = torch.randint(0, cfg.vocab, (args.queries + 1, args.seq), generator=torch.Generator().manual_seed(1))
    for n_layers in args.layers:
        graph = build_decoder(cfg, n_layers=n_layers or cfg.n_layers, calib_tokens=min(args.seq, 32), seed=0,
                              prune_last=True)
        com = commit_graph(graph, bench.RATE, device=device, policy=args.policy, model_ops=model_ops)
        params = params_for(args.lam, len(model_ops), rate=bench.RATE, plan=com.plan, fiat_shamir=args.fiat_shamir)
        v = Verifier(graph.public(), params, "C", publics=com.publics, groups=com.group_publics,
                     tables=com.table_publics, device="cpu")
        prover = Prover(graph, device=device, commitments=com)
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
                commitment.weight_matmul_mod = variants[name]
                out = run_query(prover, v, q, seed=1000 + i, wire=True)
                got[name] = dict(seen)
                if i:                                   # query 0 warms both variants up
                    runs[name].append(out)
            identical &= all(_same(got["float64"][k], got["int8"][k]) for k in ("fold", "open"))
        commitment.weight_matmul_mod = field.weight_matmul_mod
        base = None
        for name, rs in runs.items():
            med = {k: statistics.median(r["timings"].get(k, 0.0) for r in rs)
                   for k in ("prove_forward", "prove_lookups", "prove_fold", "prove_open", "prove_encode", "fs_hash")}
            row = {"git": sha, "model": args.model, "layers": n_layers or cfg.n_layers, "seq": args.seq,
                   "policy": args.policy, "chosen": com.plan.policy if com.plan else None, "variant": name,
                   "fiat_shamir": args.fiat_shamir, "queries": len(rs),
                   "accepted": sum(bool(r["accepted"]) for r in rs), "identical": identical,
                   "prove_s": sum(med.values()), "timings": med,
                   "device": str(device), "int8_ok": field.int8_ok(device),
                   "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                   "host": platform.node()}
            if base is None:
                base = row
            else:
                row["vs_float64"] = {k: row["timings"][k] / base["timings"][k]
                                     for k in ("prove_fold", "prove_open") if base["timings"][k]}
                row["vs_float64"]["prove_s"] = row["prove_s"] / base["prove_s"]
            print(json.dumps(row), flush=True)


if __name__ == "__main__":
    sys.exit(main())

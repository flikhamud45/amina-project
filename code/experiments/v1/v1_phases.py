"""V1's verifier phase by phase, under each of its implementations, on the same queries.

The breakdown behind the extended version's V1 verifier times (Sec. ``sec:eval-v1``, "Prover and verifier"): on
Llama-2-7B in mode Kpre at 2,048 tokens, 1.44 s per block with the optimised verifier against 2.49 s with the
baseline one, and the 45.4 s of the full model in ``tab:options`` and the body's V1 paragraph (the package's
``cpuv`` tier measures the totals with ``v1_bytes.py --v1-verifier optimised baseline``; this script shows where
the time goes).

    python -m experiments.v1.v1_phases --model llama2-7b --layers 2 --seq 2048 --mode Kpre --reps 3 \\
        --v1-verifier optimised baseline --device cuda --threads 8
    python -m experiments.v1.v1_phases --small 256 4 64 688 512 --layers 2 --seq 256 --mode Kpre --lmax 18 \\
        --reps 3 --v1-verifier optimised baseline --device cpu --threads 2          # a laptop check (seconds)
    python -m experiments.v1.v1_phases --micro 4096 2048 --reps 5 --threads 8

Builds a random-weight decoder as ``v1_bytes.py`` does (``--model``: a decoder of ``pvi.models.decoder.CONFIGS`` or
``experiments.common.models.TINY`` with ``--layers`` blocks; ``--small D H DH FF V``: a Llama-shaped decoder of that
width, heads, head size, FFN width and vocabulary), the prover on ``--device`` and the non-streaming verifier on
``--verifier-device``, in mode Kpre or C (``--policy``).  The protocol settings default to the paper's configuration
(:data:`experiments.common.cli.PAPER`: Fiat--Shamir, the lean prover and verifier, the last block at the last
position, mode Kpre's verifier reading the embedding rows itself; V1's proof is always in the compact encoding).
``--v1-verifier`` takes one or more of :data:`pvi.options.V1_VERIFIERS` (``optimised``: the window-table products,
the batched GKR check and the batch-inverted logUp identity; ``baseline``: float64 limb products and one check per
instance).  After one warm-up query per verifier (a random prompt, seed ``--prompt-seed``), ``--reps`` queries
(seeds ``--timed-seed + i``) each run once under every listed verifier (:func:`pvi.options.using`), interleaved, so
that every verifier sees the same queries and the same machine state.

Within ``verify_final`` the script also times its parts, by wrapping the functions for the run (the code paths are
the options'; the wrappers only time and count): ``final_xbar`` (``protocol._xbar``, of which ``final_xbar_products``
in ``products.xbar``), ``final_points`` (the eq tables, the prover's calls in ``prove_fold`` included),
``final_kpre_y``, ``final_check`` (``protocol._final_ok``, of which ``final_lw_products`` in ``products.lw_times``,
``final_lw_baseline`` in ``protocol._lw_e_old`` and ``final_dot`` in the inner products), and it counts the ops that
fall back from ``products.lw_times`` (``n_lw_products_fallback``: a residual op whose values span more than
``products.SPAN``) and those that run ``protocol._lw_e_old`` (``n_lw_baseline``).

**Record** (one JSON line per verifier, then a table on stdout): ``label``, ``git``, ``model``, ``layers``, ``seq``,
``mode``, ``policy``, ``fs``, ``lean``, ``lookups``, ``lmax``, ``threads``, ``path`` (the verifier's name),
``queries``, ``accepted``, ``cut`` (the cut plan's sizes), ``timings`` (the median of every phase), ``verify_s``
(the median of each query's ``verify_*`` total), ``env``.  Every query must be accepted, else the exit status is 1.

``--micro ROWS T``: no model and no prover; the final check's ``(L e, W e)`` of one requantisation op (int8 values, a
few exceptions) and of one residual op (int16 ``Delta`` spanning about 7,000 values, as the random-weight
decoders'), both ``[ROWS, T]``, and ``_xbar``'s ``X^T e`` of an int8 input ``[T, ROWS]``, at a random point, by each
listed verifier's products (``baseline``: ``protocol._lw_e_old`` and ``extfield.int_times``; ``optimised``:
``products.lw_times`` and ``products.xbar``), checked equal; medians in ns per entry (``ns_per_entry``), and the
per-block projection for Llama-2-7B (34,304 requantisation rows, 8,192 residual rows and 23,296 ``_xbar`` columns per
block).

Hardware and time: the extended version's numbers are 8 threads of a Xeon Silver 4114 with an RTX 2080 Ti prover
(Llama-2-7B, 2 blocks, ``--lmax 24``: minutes).  The laptop check takes seconds.
"""
from __future__ import annotations

import argparse
import contextlib
import statistics
import sys
import time

import torch

from experiments.common import cli, records
from experiments.common.models import decoder_config
from experiments.common.paths import CODE
from pvi import options
from pvi.v1.cut import LMAX


@contextlib.contextmanager
def _instrumented(parts: dict):
    """Time and count the final check's parts into ``parts`` (module docstring) while the block runs, then put the
    functions back."""
    from pvi.v1 import products
    from pvi.v1 import protocol as cp
    saved = []

    def wrap(module, name: str, key: str, count=None):
        fn = getattr(module, name)
        saved.append((module, name, fn))

        def timed(*args, **kw):
            t0 = time.perf_counter()
            try:
                out = fn(*args, **kw)
            finally:
                parts[key] = parts.get(key, 0.0) + time.perf_counter() - t0
            if count is not None and count[1](out):
                parts[count[0]] = parts.get(count[0], 0) + 1
            return out
        setattr(module, name, timed)

    wrap(cp, "_xbar", "final_xbar")
    wrap(cp, "_kpre_y_ok", "final_kpre_y")
    wrap(cp, "_final_ok", "final_check")
    wrap(cp, "_points", "final_points")
    wrap(products, "xbar", "final_xbar_products")
    wrap(products, "lw_times", "final_lw_products", ("n_lw_products_fallback", lambda r: r is None))
    wrap(cp, "_lw_e_old", "final_lw_baseline", ("n_lw_baseline", lambda r: True))
    wrap(cp, "_dot", "final_dot")
    try:
        yield
    finally:
        for module, name, fn in reversed(saved):
            setattr(module, name, fn)


LLAMA7_REQUANT, LLAMA7_RESIDUAL, LLAMA7_XBAR = 3 * 4096 + 2 * 11008, 2 * 4096, 4096 * 3 + 11008
"""Llama-2-7B per block: the requantisation cut ops' rows (q, k, v, gate, up), the residual ones' (o, down) and
``_xbar``'s columns (one per distinct input: q/k/v share one, gate/up one)."""


def micro(rows: int, t: int, reps: int, verifiers: list, threads: int) -> int:
    """The ``--micro`` measurement (module docstring)."""
    from pvi.core import extfield as ef
    from pvi.core.field import P
    from pvi.v1 import cut as cutmod
    from pvi.v1 import products
    from pvi.v1 import protocol as cp
    g = torch.Generator().manual_seed(0)
    e = torch.randint(0, P, (t, ef.D), generator=g)
    req = cutmod.CutOp("q", "requant", 3_000_017, 30, -127, 127, rows, 4096, False, "c", 0, 0, "q", "row")
    res = cutmod.CutOp("o", "residual", 70_001, 30, 0, 0, rows, 4096, False, "c", 0, 0, "o", "row")
    s8 = torch.randint(-127, 128, (rows, t), generator=g).to(torch.int8)
    d16 = torch.randint(-3000, 4000, (rows, t), generator=g).to(torch.int16)
    idx = torch.randperm(rows * t, generator=g)[:max(1, rows * t // 100_000)].sort().values
    s8.view(-1)[idx] = 127
    z = torch.randint(1 << 20, 1 << 28, (idx.numel(),), generator=g)
    s64, d64 = s8.to(torch.int64), d16.to(torch.int64)       # the baseline verifier decodes int64 values
    x = torch.randint(-128, 129, (t, rows), generator=g)      # derive's inputs are int64 for both
    e0 = torch.zeros(0, dtype=torch.int64)
    cases = {"final_requant": {"optimised": lambda: products.lw_times(s8, idx, z, req, e),
                               "baseline": lambda: cp._lw_e_old(s64, idx, z, req, e, t)},
             "final_residual": {"optimised": lambda: products.lw_times(d16, e0, e0, res, e),
                                "baseline": lambda: cp._lw_e_old(d64, e0, e0, res, e, t)},
             # the baseline _xbar: its size check, then the limb product
             "xbar": {"optimised": lambda: products.xbar(x, e),
                      "baseline": lambda: (int(x.abs().max()), ef.int_times(x.to(torch.float64).T, e, 31))[1]}}

    def timed(f):
        f()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = f()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts), out

    per = {}
    for name, impl in cases.items():
        per[name], ref = {}, None
        for v in verifiers:
            per[name][v], got = timed(impl[v])
            if ref is None:
                ref = got
            elif not (all(torch.equal(a, b) for a, b in zip(got, ref)) if isinstance(ref, tuple)
                      else torch.equal(got, ref)):
                print(f"FAILED: {name}: the {v} products differ from the {verifiers[0]} ones", file=sys.stderr)
                return 1
    n = rows * t
    records.emit({"micro": [rows, t], "threads": threads, "reps": reps,
                  "ns_per_entry": {k: {v: s / n * 1e9 for v, s in d.items()} for k, d in per.items()},
                  "env": records.environment("cpu")})
    print(f"\n{'ns per entry (median)':24s}" + "".join(f"{v:>12s}" for v in verifiers))
    for k, d in per.items():
        print(f"{k:24s}" + "".join(f"{s / n * 1e9:12.2f}" for s in d.values()))
    print(f"\nLlama-2-7B per block at T = {t}, the final check's products and _xbar (s):")
    for v in verifiers:
        sec = (per["final_requant"][v] * LLAMA7_REQUANT + per["final_residual"][v] * LLAMA7_RESIDUAL
               + per["xbar"][v] * LLAMA7_XBAR) / rows
        print(f"  {v:10s} {sec:8.3f}")
    return 0


NAMES = ["verify_decode", "verify_derive", "verify_cut_m1", "verify_gkr", "verify_products", "verify_final",
         "final_xbar", "final_xbar_products", "final_points", "final_kpre_y", "final_check", "final_lw_products",
         "final_lw_baseline", "final_dot", "n_lw_products_fallback", "n_lw_baseline", "verify_logup",
         "verify_columns", "verify_total"]
"""The rows of the printed table, in order."""


def build_parser() -> argparse.ArgumentParser:
    # "resolve": --v1-verifier below takes several values in place of add_runtime_arguments' single one
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], conflict_handler="resolve")
    ap.add_argument("--micro", type=int, nargs=2, metavar=("ROWS", "T"), default=None,
                    help="no model: the final check's products of one op of ROWS x T values")
    ap.add_argument("--model", default="", help="the decoder's shapes (pvi.models.decoder.CONFIGS, "
                    "experiments.common.models.TINY)")
    ap.add_argument("--small", type=int, nargs=5, metavar=("D", "H", "DH", "FF", "V"), default=None,
                    help="instead of --model: a Llama-shaped decoder of these sizes")
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: the full model)")
    ap.add_argument("--seq", type=int, default=256, help="prompt tokens")
    ap.add_argument("--mode", default="Kpre", choices=["Kpre", "C"])
    ap.add_argument("--lmax", type=int, default=LMAX, help="V1's GKR instances: the target size 2^lmax")
    ap.add_argument("--reps", type=int, default=3, help="timed queries (--micro: timed calls) per verifier")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed")
    ap.add_argument("--calib-tokens", type=int, default=cli.CALIB_TOKENS,
                    help="calibration tokens of the random-weight decoder (at most --seq)")
    ap.add_argument("--prompt-seed", type=int, default=1, help="the seed of the warm-up queries' random prompt")
    ap.add_argument("--timed-seed", type=int, default=100,
                    help="timed random prompt i is drawn with the seed --timed-seed + i")
    ap.add_argument("--label", default="", help="a label copied into every record")
    cli.add_protocol_arguments(ap)
    cli.add_runtime_arguments(ap, threads=8)
    ap.add_argument("--v1-verifier", nargs="+", choices=options.V1_VERIFIERS, default=list(options.V1_VERIFIERS),
                    help="V1's verifiers to time on the same queries (optimised, baseline)")
    return ap


def main() -> int:
    from pvi.models.decoder import DecoderConfig, build_decoder
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier
    from pvi.v1 import protocol as cp

    args = build_parser().parse_args()
    verifiers = list(dict.fromkeys(args.v1_verifier))
    args.v1_verifier = verifiers[0]
    cli.apply_runtime(args)
    if args.micro:
        return micro(*args.micro, args.reps, verifiers, args.threads)
    if not args.wire:
        raise SystemExit("--no-wire: V1's proof is always in the compact encoding")
    if args.small:
        d, h, dh, ff, vocab = args.small
        cfg = DecoderConfig(f"small-llama-{d}", d, max(1, args.layers), h, h, dh, ff, vocab, norm="rmsnorm",
                            mlp="swiglu", pos="rope", bias=False, tied=False)
    elif args.model:
        cfg = decoder_config(args.model)
    else:
        raise SystemExit("give --model, --small or --micro")
    n_layers = args.layers or cfg.n_layers
    graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(args.seq, args.calib_tokens), seed=args.seed,
                          prune_last=args.prune_last)
    if args.mode == "C":
        coms = commit_graph(graph, args.rate, device=args.device, policy=args.policy, cache=args.setup_cache)
        params = params_for(args.lam, len(graph.mat_ops), rate=args.rate, fiat_shamir=args.fs, plan=coms.plan,
                            cut=True, cut_lmax=args.lmax)
        verifier = Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                            tables=coms.table_publics, lean=args.lean, device=args.verifier_device)
        prover = Prover(graph, device=args.device, commitments=coms, lean=args.lean)
    else:
        params = params_for(args.lam, len(graph.mat_ops), rate=args.rate, fiat_shamir=args.fs, cut=True,
                            cut_lmax=args.lmax)
        verifier = Verifier(graph.public(), params, "Kpre", weights={op.name: (op.weight, op.bias)
                                                                      for op in graph.mat_ops},
                            lean=args.lean, device=args.verifier_device, lookups=args.lookups)
        verifier.precompute(Challenger())
        prover = Prover(graph, device=args.device, lean=args.lean)
    parts: dict = {}

    def query(name: str, x: torch.Tensor) -> dict:
        parts.clear()
        with options.using(v1_verifier=name), _instrumented(parts):
            out = cp.run_cut_query(prover, verifier, x)
        out["timings"].update(parts)
        return out

    def prompt(seed: int) -> torch.Tensor:
        return torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(seed))

    warm = {v: query(v, prompt(args.prompt_seed)) for v in verifiers}
    runs = {v: [] for v in verifiers}
    for i in range(args.reps):
        x = prompt(args.timed_seed + i)
        for v in verifiers:
            runs[v].append(query(v, x))
    sha = records.git_sha(CODE)
    env = records.environment(args.device)
    ok = all(w["accepted"] for w in warm.values())
    rows = {}
    for v, rs in runs.items():
        keys = sorted({k for r in rs for k in r["timings"]})
        med = {k: statistics.median(r["timings"].get(k, 0.0) for r in rs) for k in keys}
        verify = statistics.median(sum(t for k, t in r["timings"].items() if k.startswith("verify_")) for r in rs)
        accepted = sum(r["accepted"] for r in rs)
        ok = ok and accepted == len(rs)
        rows[v] = med | {"verify_total": verify}
        records.emit({"label": args.label, "git": sha, "model": cfg.name, "layers": n_layers, "seq": args.seq,
                      "mode": args.mode, "policy": args.policy if args.mode == "C" else None, "fs": args.fs,
                      "lean": args.lean, "lookups": args.lookups if args.mode != "C" else False, "lmax": args.lmax,
                      "threads": args.threads, "path": v, "queries": len(rs), "accepted": accepted,
                      "cut": rs[0]["cut"], "timings": med, "verify_s": verify, "env": env})
    print(f"\n{'phase (s, median)':24s}" + "".join(f"{v:>14s}" for v in verifiers))
    for k in NAMES:
        if any(k in r for r in rows.values()):
            print(f"{k:24s}" + "".join(f"{rows[v].get(k, 0.0):14.4f}" for v in verifiers))
    if not ok:
        print("FAILED: a query was rejected", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

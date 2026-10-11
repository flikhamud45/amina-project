"""V-V1N: V1's verifier phase by phase with today's code and V-V1N's paths, on the same queries.

    cd code && PYTHONPATH=src python experiments/8_v1/v1_phases.py --small 256 4 64 688 512 --layers 2 --seq 256 \\
        --mode Kpre --fs --lmax 18 --reps 3 --paths 0 1 scatter --threads 2
    cd code && PYTHONPATH=src python experiments/8_v1/v1_phases.py --model llama2-7b --layers 2 --seq 2048 \\
        --mode Kpre --fs --lean --lmax 24 --reps 3 --paths 0 1 --device cuda --threads 8

Builds a random-weight decoder as ``v1_bytes.py`` does (``--model``: a ``transformer.CONFIGS`` entry, with
``--layers`` blocks; ``--small D H DH FF V``: a Llama-shaped decoder of that width, heads, head size, FFN width and
vocabulary, for a laptop), its last block pruned, the prover on ``--device`` and the non-streaming verifier on the
CPU, in mode Kpre or C (``--policy``), Fiat--Shamir with ``--fs``.  After one warm-up query per path, ``--reps``
queries (``torch.randint`` prompts, seeds 100 ..) each run once under every ``PVI_V1_BUCKETS`` value of ``--paths``
(``0`` today's code, ``1`` the table products, ``scatter`` the buckets), interleaved, so every path sees the same
queries and the same machine state.  Within ``verify_final`` the script also times its parts (``final_xbar``:
``_xbar``, of which ``final_xbar_new`` in ``buckets.xbar``; ``final_points``: the eq tables; ``final_kpre_y``;
``final_check``: ``_final_ok``, of which ``final_lw_new`` in ``buckets.lw_times``, ``final_lw_old`` in today's
``_lw_e_old`` and ``final_dot`` in the inner products) and counts the ops that fall back from ``buckets.lw_times``
(``n_lw_new_fallback``) and that run ``_lw_e_old`` (``n_lw_old``), by wrapping the modules' functions.  One JSON line per
path: the median of every phase, the verifier's total (the ``verify_*`` phases) and the accepted count; then a
table.  Every query must be accepted (else the exit status is 1).

    cd code && PYTHONPATH=src python experiments/8_v1/v1_phases.py --micro 4096 2048 --reps 5 --threads 8

``--micro ROWS T``: no model and no prover; the final check's ``(L e, W e)`` of one requant op (int8 values, a few
exceptions) and of one residual op (int16 ``Delta`` spanning about 7,000 values, as the random-weight decoders'),
both ``[ROWS, T]``, and ``_xbar``'s ``X^T e`` of an int8 input ``[T, ROWS]``, at a random point, with today's code
(``cut_protocol._lw_e_old``, ``extfield.int_times``) and each path, checked equal; medians in ns per entry, and the
per-block projection for Llama-2-7B (34,304 requant rows, 8,192 residual rows and 23,296 ``_xbar`` columns per block).
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

CODE = Path.cwd()


def _wrap(module, name: str, key: str, acc: dict) -> None:
    """Time ``module.name`` into ``acc[key]`` (added to) on every call."""
    fn = getattr(module, name)

    def timed(*args, **kw):
        t0 = time.perf_counter()
        try:
            return fn(*args, **kw)
        finally:
            acc[key] = acc.get(key, 0.0) + time.perf_counter() - t0
    setattr(module, name, timed)


def _count(module, name: str, key: str, acc: dict, pred) -> None:
    """Count the calls of ``module.name`` whose result satisfies ``pred`` into ``acc[key]``."""
    fn = getattr(module, name)

    def counted(*args, **kw):
        out = fn(*args, **kw)
        if pred(out):
            acc[key] = acc.get(key, 0) + 1
        return out
    setattr(module, name, counted)


LLAMA7_REQUANT, LLAMA7_RESIDUAL, LLAMA7_XBAR = 3 * 4096 + 2 * 11008, 2 * 4096, 4096 * 3 + 11008
"""Llama-2-7B per block: the requant cut ops' rows (q, k, v, gate, up), the residual ones' (o, down) and ``_xbar``'s
columns (one per distinct input: q/k/v share one, gate/up one)."""


def micro(rows: int, t: int, reps: int, paths: list, threads: int) -> int:
    """The ``--micro`` measurement (module docstring)."""
    from pvi.fullcheck import buckets
    from pvi.fullcheck import cut as cutmod
    from pvi.fullcheck import cut_protocol as cp
    from pvi.fullcheck import extfield as ef
    from pvi.fullcheck.field import P
    g = torch.Generator().manual_seed(0)
    e = torch.randint(0, P, (t, ef.D), generator=g)
    req = cutmod.CutOp("q", "requant", 3_000_017, 30, -127, 127, rows, 4096, False, "c", 0, 0, "q", "row")
    res = cutmod.CutOp("o", "residual", 70_001, 30, 0, 0, rows, 4096, False, "c", 0, 0, "o", "row")
    s8 = torch.randint(-127, 128, (rows, t), generator=g).to(torch.int8)
    d16 = torch.randint(-3000, 4000, (rows, t), generator=g).to(torch.int16)
    idx = torch.randperm(rows * t, generator=g)[:max(1, rows * t // 100_000)].sort().values
    s8.view(-1)[idx] = 127
    z = torch.randint(1 << 20, 1 << 28, (idx.numel(),), generator=g)
    s64, d64 = s8.to(torch.int64), d16.to(torch.int64)       # today's verifier decodes int64 values
    x = torch.randint(-128, 129, (t, rows), generator=g)      # derive's inputs are int64 in both
    e0 = torch.zeros(0, dtype=torch.int64)
    def under(path, f):                                          # as a query dispatches: PVI_V1_BUCKETS = path
        saved = os.environ.get(buckets.ENV)
        os.environ[buckets.ENV] = path
        try:
            return f()
        finally:
            if saved is None:
                os.environ.pop(buckets.ENV, None)
            else:
                os.environ[buckets.ENV] = saved
    cases = {"final_requant": (lambda m: under(m, lambda: buckets.lw_times(s8, idx, z, req, e)),
                               lambda: cp._lw_e_old(s64, idx, z, req, e, t)),
             "final_residual": (lambda m: under(m, lambda: buckets.lw_times(d16, e0, e0, res, e)),
                                lambda: cp._lw_e_old(d64, e0, e0, res, e, t)),
             "xbar": (lambda m: under(m, lambda: buckets.xbar(x, e)),   # today's _xbar: its size check, then the limbs
                      lambda: (int(x.abs().max()), ef.int_times(x.to(torch.float64).T, e, 31))[1])}

    def timed(f):
        f()
        ts = []
        for _ in range(reps):
            t0 = time.perf_counter()
            out = f()
            ts.append(time.perf_counter() - t0)
        return statistics.median(ts), out
    per = {}
    for name, (new, old) in cases.items():
        t_old, ref = timed(old)
        per[name] = {"0": t_old}
        for p in paths:
            if p == "0":
                continue
            t_new, got = timed(lambda p=p: new(p))
            same = all(torch.equal(a, b) for a, b in zip(got, ref)) if isinstance(ref, tuple) else torch.equal(got, ref)
            if not same:
                print(f"FAILED: {name} path {p} differs from today's code", file=sys.stderr)
                return 1
            per[name][p] = t_new
    n = rows * t
    print(json.dumps({"micro": [rows, t], "threads": threads, "reps": reps,
                      "ns_per_entry": {k: {p: v / n * 1e9 for p, v in d.items()} for k, d in per.items()}}))
    print(f"\n{'ns per entry (median)':24s}" + "".join(f"{'path ' + p:>12s}" for p in per["xbar"]))
    for k, d in per.items():
        print(f"{k:24s}" + "".join(f"{v / n * 1e9:12.2f}" for v in d.values()))
    print(f"\nLlama-2-7B per block at T = {t}, the final check's products and _xbar (s):")
    for p in per["xbar"]:
        sec = (per["final_requant"][p] * LLAMA7_REQUANT + per["final_residual"][p] * LLAMA7_RESIDUAL
               + per["xbar"][p] * LLAMA7_XBAR) / rows
        print(f"  path {p:8s} {sec:8.3f}")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--micro", type=int, nargs=2, metavar=("ROWS", "T"), default=None)
    ap.add_argument("--model", default="")
    ap.add_argument("--small", type=int, nargs=5, metavar=("D", "H", "DH", "FF", "V"), default=None)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--seq", type=int, default=256)
    ap.add_argument("--mode", default="Kpre", choices=["Kpre", "C"])
    ap.add_argument("--policy", default="auto")
    ap.add_argument("--fs", action="store_true")
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--lmax", type=int, default=24)
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--paths", nargs="+", default=["0", "1"])
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    if args.micro:
        return micro(*args.micro, args.reps, args.paths, args.threads)
    from pvi.fullcheck import buckets
    from pvi.fullcheck import cut_protocol as cp
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for
    from pvi.fullcheck.transformer import CONFIGS, DecoderConfig, build_decoder

    if args.small:
        d, h, dh, ff, vocab = args.small
        cfg = DecoderConfig(f"small-llama-{d}", d, max(1, args.layers), h, h, dh, ff, vocab, norm="rmsnorm",
                            mlp="swiglu", pos="rope", bias=False, tied=False)
    else:
        cfg = CONFIGS[args.model]
    n_layers = args.layers or cfg.n_layers
    graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(args.seq, 32), seed=0, prune_last=True)
    if args.mode == "C":
        coms = commit_graph(graph, 4, device=args.device, policy=args.policy)
        params = params_for(128, len(graph.mat_ops), fiat_shamir=args.fs, plan=coms.plan, cut=True, cut_lmax=args.lmax)
        verifier = Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                            tables=coms.table_publics, lean=args.lean)
        prover = Prover(graph, device=args.device, commitments=coms, lean=args.lean)
    else:
        params = params_for(128, len(graph.mat_ops), fiat_shamir=args.fs, cut=True, cut_lmax=args.lmax)
        verifier = Verifier(graph.public(), params, "Kpre", weights={op.name: (op.weight, op.bias) for op in graph.mat_ops},
                            lean=args.lean)
        verifier.precompute(Challenger())
        prover = Prover(graph, device=args.device, lean=args.lean)
    parts: dict = {}
    _wrap(cp, "_xbar", "final_xbar", parts)
    _wrap(cp, "_kpre_y_ok", "final_kpre_y", parts)
    _wrap(cp, "_final_ok", "final_check", parts)
    # finer: inside _xbar and _final_ok (V-V1N's per-op profile)
    _wrap(cp, "_points", "final_points", parts)
    _wrap(buckets, "xbar", "final_xbar_new", parts)
    _wrap(buckets, "lw_times", "final_lw_new", parts)
    _wrap(cp, "_lw_e_old", "final_lw_old", parts)
    _wrap(cp, "_dot", "final_dot", parts)
    _count(buckets, "lw_times", "n_lw_new_fallback", parts, lambda r: r is None)
    _count(cp, "_lw_e_old", "n_lw_old", parts, lambda r: True)
    saved = os.environ.get(buckets.ENV)

    def query(path: str, x: torch.Tensor) -> dict:
        os.environ[buckets.ENV] = path
        parts.clear()
        try:
            out = cp.run_cut_query(prover, verifier, x)
        finally:
            if saved is None:
                os.environ.pop(buckets.ENV, None)
            else:
                os.environ[buckets.ENV] = saved
        out["timings"].update(parts)
        return out

    def prompt(seed: int) -> torch.Tensor:
        return torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(seed))

    warm = {p: query(p, prompt(1)) for p in args.paths}
    runs = {p: [] for p in args.paths}
    for i in range(args.reps):
        x = prompt(100 + i)
        for p in args.paths:
            runs[p].append(query(p, x))
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=CODE).stdout.strip()
    ok = all(w["accepted"] for w in warm.values())
    rows = {}
    for p, rs in runs.items():
        keys = sorted({k for r in rs for k in r["timings"]})
        med = {k: statistics.median(r["timings"].get(k, 0.0) for r in rs) for k in keys}
        verify = statistics.median(sum(v for k, v in r["timings"].items() if k.startswith("verify_")) for r in rs)
        accepted = sum(r["accepted"] for r in rs)
        ok = ok and accepted == len(rs)
        rows[p] = med | {"verify_total": verify}
        print(json.dumps({"label": args.label, "git": sha, "model": cfg.name, "layers": n_layers, "seq": args.seq,
                          "mode": args.mode, "fs": args.fs, "lmax": args.lmax, "threads": args.threads, "path": p,
                          "queries": len(rs), "accepted": accepted, "cut": rs[0]["cut"], "timings": med,
                          "verify_s": verify}), flush=True)
    names = ["verify_decode", "verify_derive", "verify_cut_m1", "verify_gkr", "verify_products", "verify_final",
             "final_xbar", "final_xbar_new", "final_points", "final_kpre_y", "final_check", "final_lw_new",
             "final_lw_old", "final_dot", "n_lw_new_fallback", "n_lw_old", "verify_logup", "verify_columns",
             "verify_total"]
    print(f"\n{'phase (s, median)':22s}" + "".join(f"{'path ' + p:>14s}" for p in args.paths))
    for k in names:
        if any(k in r for r in rows.values()):
            print(f"{k:22s}" + "".join(f"{rows[p].get(k, 0.0):14.4f}" for p in args.paths))
    if not ok:
        print("FAILED: a query was rejected", file=sys.stderr)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

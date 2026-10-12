"""Timing harness for the improvement work: the same queries on the old and the new code.

    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model lenet5 --modes C Kpre --queries 10
    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C --verifier-device cuda
    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model vgg16 --modes C --policy paper cnn17 R16
    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model gpt2 --seq 64 --modes C Kpre --wire
    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model lenet5 --modes C --policy paper cnn16 --wire off on
    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model gpt2 --modes C Kpre --policy R16 R16c \\
        --prune-last off on --lookups off on
    cd code && PYTHONPATH=src python experiments/6_improvements/perf.py --model opt-1.3b --real facebook/opt-1.3b \\
        --smooth 0.55 --calib-seq 2048 --seq 2048 --modes C Kpre --policy auto --queries 3 --lean --wire on \\
        --prune-last on --tampers 3

Builds the model exactly as bench.py does (trained CNN weights, random-weight decoders with seed 0;
``--random-init``: random CNN weights too, as ``ab_verifier.py`` builds them), commits it, runs one
warm-up and ``--queries`` honest queries with ``run_query`` and prints one JSON line per (model,
mode): the median of every timing part and every byte part, the ``[min, max]`` of each query's prover
and verifier totals (``prove_s_range``, ``verify_s_range``: the paper's median [min-max]), the
acceptance count, and the git commit, so results of two checkouts can be compared line by line.

``--policy`` names the commitment plans (``pvi.fullcheck.plans``) of mode C: each is committed (its
``commit_s`` is the setup time).  ``--wire`` sends the proof in the compact encoding
(``run_query(wire=True)``: encoded bytes, ``prove_encode`` and ``verify_decode``); ``--wire off on``
runs every policy both ways.  The queries of all these variants are interleaved in this process,
the order rotating from query to query, and each variant gets its own line, with ``vs_first``: its
medians over those of the first variant (e.g. ``paper`` without wire, the report).  Modes K and
Kpre open no columns, so they run once per wire setting.  ``--prune-last`` builds the decoder with its
last block at the last position (``build_decoder(..., prune_last=True)``; ``--prune-last off on``:
both graphs, each variant on each); ``--lookups`` (``off on``: both) gives the K and Kpre verifiers
the embedding rows to read themselves (``Verifier(lookups=True)``; in mode C a ``c`` policy's lookup
tables do that).

``--real`` (a Hugging Face checkpoint): the checkpoint's integer graph instead of the random-weight one; for
Llama-family checkpoints ``real_llama.build_llama_from_hf`` with ``MAIN_RECIPE`` and its fixed public prefix
(``v1_bytes.load_real_llama``), for OPT (``real_weights.build_opt_from_hf``; ``--smooth``: the paper's SmoothQuant through the norm gains), calibrated
on the CPU on one WikiText-2 window of ``--calib-seq`` tokens from the end of the test split and queried on
``--queries`` + 1 windows of ``--seq`` tokens from its start (``experiments/8_v1/v1_bytes.py``'s ``load_real``
and ``real_graph``; ``--model`` names its shapes).  Every block is built, and mode C's commitment plans are made
from the graph's own weight ops.  ``--tampers n`` (decoders): after the timed queries, every variant runs ``n``
of each of ``bench.py``'s LLM attacks on the first query (a random claim entry +-1..999, the top logit forged,
the last hidden coordinate set to ``Z_BOUND // 4``), each of which must be rejected (``tampers``).  ``peak``: the
host's peak resident set (Linux VmHWM) and the GPU's peak memory over a mode's warm-ups, queries and tampers
(all its variants); ``build_peak``: the same over the build and the commitments.  The exit status is 1 if an
honest query was rejected or a tampered one accepted.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch

CODE = Path.cwd()   # run from code/


def _load(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, CODE / path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod          # (its dataclasses look their module up)
    spec.loader.exec_module(mod)
    return mod


def _chosen(policy: str, plan) -> str:
    """The policy whose plan was committed, as ``plan_bytes.py`` records it: ``policy`` itself when it
    has no plan (``paper``), else ``plan.policy`` (``auto``: the c plan it picked)."""
    return policy if plan is None else plan.policy


def _peaks(device) -> dict:
    """Bytes since ``bench._reset_peaks``, named as ``bench.py`` records them: the host's peak resident set
    (``None`` where ``/proc`` has none) and on a GPU its peak allocated and reserved memory."""
    out = {"host_peak_rss": None}
    try:
        with open("/proc/self/status") as fh:
            out["host_peak_rss"] = 1024 * next(int(line.split()[1]) for line in fh if line.startswith("VmHWM:"))
    except (OSError, StopIteration):
        pass
    if device.type == "cuda":
        out["gpu_peak_memory"] = torch.cuda.max_memory_allocated(device)
        out["gpu_peak_reserved"] = torch.cuda.max_memory_reserved(device)
    return out


def _tampers(prover, v, x, n: int, wire: bool, seed: int = 123) -> dict:
    """``n`` attempts of each of ``bench.py``'s LLM attacks on the query ``x`` (its ``tamper_C_int_lam40`` cell):
    ``single_value``, a random entry of a random weight op's claim +-1..999; ``output_logit``, the smallest logit
    raised 1000 above the largest; ``last_hidden``, a random coordinate of the op feeding the LM head (the last
    block's down projection) at the last position set to ``Z_BOUND // 4``.  ``{kind: {attempts, rejected,
    labels}}``."""
    from pvi.fullcheck.protocol import Z_BOUND, run_query
    mats = prover.graph.mat_ops
    g = torch.Generator().manual_seed(seed)
    out = {k: {"attempts": 0, "rejected": 0, "labels": {}} for k in ("single_value", "output_logit", "last_hidden")}

    def attempt(kind, tamper):
        res = run_query(prover, v, x, wire=wire, forward_kwargs={"tamper": tamper})
        rec, label = out[kind], str(res["rejected_at"])
        rec["attempts"] += 1
        rec["rejected"] += int(not res["accepted"])
        rec["labels"][label] = rec["labels"].get(label, 0) + 1

    def t_logit(o, z):
        if o.name == mats[-1].name:
            z = z.clone()
            z[int(z[:, -1].argmin()), -1] = int(z[:, -1].max()) + 1000
        return z

    for i in range(n):
        op = mats[int(torch.randint(len(mats), (1,), generator=g))]
        pos = int(torch.randint(10 ** 9, (1,), generator=g))
        delta = int(torch.randint(1, 1000, (1,), generator=g)) * (1 if i % 2 else -1)
        j = int(torch.randint(mats[-2].n_rows, (1,), generator=g))

        def t_single(o, z, op=op, pos=pos, delta=delta):
            if o.name == op.name:
                z = z.clone()
                z.view(-1)[pos % z.numel()] += delta
            return z

        def t_last(o, z, j=j):
            if o.name == mats[-2].name:
                z = z.clone()
                z[j, -1] = Z_BOUND // 4
            return z

        attempt("single_value", t_single)
        attempt("output_logit", t_logit)
        attempt("last_hidden", t_last)
    return out


def _medians(runs: list[dict]) -> tuple[dict, dict, float, float]:
    """``(timings, bytes, prove_s, verify_s)``: the medians over ``runs`` of ``run_query``."""
    timing = {k: statistics.median(r["timings"].get(k, 0.0) for r in runs)
              for k in sorted({k for r in runs for k in r["timings"]})}
    nbytes = {k: statistics.median(r["bytes"].get(k, 0) for r in runs)
              for k in sorted({k for r in runs for k in r["bytes"]})}
    prove = sum(timing.get(k, 0.0) for k in ("prove_forward", "prove_lookups", "prove_fold", "prove_open",
                                                "prove_encode", "fs_hash"))
    verify = sum(v_ for k, v_ in timing.items() if k.startswith("verify_") or k == "fs_hash")
    return timing, nbytes, prove, verify


def _spread(runs: list[dict]) -> dict:
    """``[min, max]`` over ``runs`` of each query's own prover and verifier totals (the parts of :func:`_medians`),
    for the paper's median [min-max] (F6)."""
    prove = [sum(r["timings"].get(k, 0.0) for k in ("prove_forward", "prove_lookups", "prove_fold", "prove_open",
                                                     "prove_encode", "fs_hash")) for r in runs]
    verify = [sum(v_ for k, v_ in r["timings"].items() if k.startswith("verify_") or k == "fs_hash") for r in runs]
    return {"prove_s_range": [min(prove), max(prove)] if runs else None,
            "verify_s_range": [min(verify), max(verify)] if runs else None}


def main() -> int:
    ab = _load("ab_verifier", "experiments/6_improvements/ab_verifier.py")
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: full model)")
    ap.add_argument("--modes", nargs="+", default=["C", "Kpre"])
    ap.add_argument("--lam", type=int, default=128)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verifier-device", default="cpu")
    ap.add_argument("--queries", type=int, default=10)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--fs", action="store_true", help="Fiat-Shamir (non-interactive); each party's hashing is timed")
    ap.add_argument("--label", default="")
    ap.add_argument("--extra", default="{}", help="JSON dict of extra keyword arguments for Verifier (new options)")
    ap.add_argument("--policy", nargs="+", default=["paper"],
                    help="mode C: commitment plans (paper, tight, cnn<e>, R<rate>; with the suffix c, their col "
                         "layouts; auto: the c plan it picks, printed as 'chosen'), timed interleaved in this process")
    ap.add_argument("--wire", nargs="*", choices=["off", "on"], default=["off"],
                    help="run_query(wire=True), the compact encoding of the proof: '--wire' for every variant, "
                         "'--wire off on' for each policy both ways, timed interleaved")
    ap.add_argument("--prune-last", nargs="*", choices=["off", "on"], default=["off"],
                    help="decoders: the last block at the last position only (build_decoder(prune_last=True)); "
                         "'--prune-last off on': both graphs, timed interleaved")
    ap.add_argument("--lookups", nargs="*", choices=["off", "on"], default=["off"],
                    help="modes K and Kpre: the verifier reads the embedding rows itself (Verifier(lookups=True)); "
                         "'--lookups off on': both, timed interleaved")
    ap.add_argument("--real", default="", help="decoders: a Hugging Face OPT checkpoint (e.g. facebook/opt-1.3b), "
                    "its integer graph (real_weights.build_opt_from_hf) queried on WikiText-2 windows; --model names "
                    "its shapes")
    ap.add_argument("--smooth", type=float, default=None, help="with --real: SmoothQuant strength through the norm "
                    "gains (F2; 0.55 with smooth_pct 99.9, pct 99.99, pct_att 99.99)")
    ap.add_argument("--calib-seq", type=int, default=0, help="with --real: the calibration window's tokens (0: --seq)")
    ap.add_argument("--recipe", choices=["legacy", "certified"], default="legacy",
                    help="with --real: certified = real_weights.CERTIFIED_RECIPE (F2a's smooth-max alpha 0.6) calibrated "
                         "on common.certified_calib (four 2,048-token train windows; --smooth and --calib-seq unused)")
    ap.add_argument("--tampers", type=int, default=0, help="decoders, per variant after the timed queries: attempts of "
                    "each of bench.py's LLM attacks, each of which must be rejected")
    ap.add_argument("--random-init", action="store_true",
                    help="CNNs (" + ", ".join(ab.CNNS) + "): random weights and random queries, as ab_verifier.py "
                         "builds them (the costs depend on the shapes only)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    bench = _load("bench", "experiments/4_defence_benchmark/bench.py")
    import pvi.fullcheck as fullcheck
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query

    device = torch.device(args.device)
    prunes = [w == "on" for w in args.prune_last or ["on"]]      # a bare flag: on
    lookups = [w == "on" for w in args.lookups or ["on"]]
    bench._reset_peaks(device)
    t0 = time.perf_counter()
    graphs, model_ops = {}, {}
    if args.model in ab.CNNS:
        if prunes != [False]:
            raise SystemExit("--prune-last is for the decoders")
        if args.real or args.tampers:
            raise SystemExit("--real and --tampers are for the decoders")
        if args.random_init:
            graph, x, _ = ab.build(fullcheck, ab.Case(args.model, "C", args.lam))
            queries = ab.random_queries(fullcheck, graph, x, args.queries + 1)
        else:
            from pvi.fullcheck.quantize import quantize_input, quantize_model
            model, data = bench._mlp_model() if args.model == "mlp_mnist" else bench._cnn_model(args.model)
            graph = quantize_model(model, data["train_x"])
            xs, _ = next(bench._test_batches(data, args.queries + 1))
            queries = list(quantize_input(graph, xs).split(1))
        graphs[False], model_ops[False] = graph, None
        n_checks = len(graph.mat_ops)
        seq = None
    elif args.real:                 # the checkpoint's integer graph, every block; WikiText-2 windows as queries
        if args.layers:
            raise SystemExit("--real builds every block of the checkpoint")
        v1b = _load("v1_bytes", "experiments/8_v1/v1_bytes.py")
        model, queries, calib = v1b.load_real(args.real, args.seq, args.queries + 1, args.calib_seq, args.recipe)
        for prune in prunes:
            graphs[prune], _ = v1b.real_graph(model, calib, args.smooth, prune, args.recipe)
            model_ops[prune] = None    # planned from its own weight ops (OPT's LM head has a bias column)
        del model
        n_checks = len(graphs[prunes[0]].mat_ops)
        seq = args.seq
    else:
        from pvi.fullcheck.analytic import decoder_shapes
        from pvi.fullcheck.transformer import CONFIGS, build_decoder
        cfg = CONFIGS[args.model]
        n_layers = args.layers or cfg.n_layers
        for prune in prunes:
            graphs[prune] = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(args.seq, 32), seed=0,
                                          prune_last=prune)
            model_ops[prune] = decoder_shapes(cfg, prune_last=prune)
        tokens = torch.randint(0, cfg.vocab, (args.queries + 1, args.seq), generator=torch.Generator().manual_seed(1))
        queries = [t for t in tokens.split(1)]
        n_checks = len(model_ops[prunes[0]])   # (r, t) for the full model, as bench.py
        seq = args.seq
    build_s = time.perf_counter() - t0
    policies = args.policy if "C" in args.modes else []
    wires = [w == "on" for w in args.wire or ["on"]]     # a bare --wire: every variant
    coms, commit_s = {}, {}
    for prune, graph in graphs.items():
        for policy in policies:
            if device.type == "cuda":
                torch.cuda.synchronize()
            t0 = time.perf_counter()
            coms[prune, policy] = commit_graph(graph, bench.RATE, device=device, policy=policy,
                                               model_ops=model_ops[prune])
            if device.type == "cuda":
                torch.cuda.synchronize()
            commit_s[prune, policy] = time.perf_counter() - t0
    build_peak = _peaks(device)
    lean = args.lean and seq is not None
    extra = json.loads(args.extra)
    from pvi.results import git_sha
    sha = git_sha(CODE)
    ok = True
    for mode in args.modes:
        bench._reset_peaks(device)
        pairs = {}
        for prune, graph in graphs.items():
            weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
            for policy in (policies if mode == "C" else [None]):
                c = coms.get((prune, policy))
                params = params_for(args.lam, n_checks, rate=bench.RATE, plan=None if c is None else c.plan,
                                    fiat_shamir=args.fs)
                for looked in (lookups if mode != "C" else [False]):
                    for wire in wires:
                        if mode == "C":
                            v = Verifier(graph.public(), params, "C", publics=c.publics, groups=c.group_publics,
                                         tables=c.table_publics, lean=lean, device=args.verifier_device, **extra)
                        else:
                            v = Verifier(graph.public(), params, mode, weights=weights, lean=lean,
                                         device=args.verifier_device, lookups=looked, **extra)
                            if mode == "Kpre":
                                v.precompute(Challenger())
                        prover = Prover(graph, device=device, commitments=c, lean=lean)
                        run_query(prover, v, queries[0], wire=wire)  # warm-up
                        pairs[prune, policy, looked, wire] = (prover, v, [])
        order = list(pairs)
        for i, q in enumerate(queries[1:]):     # interleaved, the first variant rotating
            for key in order[i % len(order):] + order[:i % len(order)]:
                prover, v, runs = pairs[key]
                runs.append(run_query(prover, v, q, wire=key[3]))
        tampered = {key: _tampers(prover, v, queries[0], args.tampers, key[3])
                    for key, (prover, v, _) in pairs.items()} if args.tampers else {}
        peak = _peaks(device)
        first = None
        for (prune, policy, looked, wire), (_, v, runs) in pairs.items():
            timing, nbytes, prove, verify = _medians(runs)
            out = {"label": args.label, "git": sha, "model": args.model, "real": args.real or None,
                   "smooth": args.smooth if args.real else None,
                   "recipe": args.recipe if args.real else None,
                   "calib_seq": (args.calib_seq or args.seq) if args.real else None,
                   "seq": seq, "layers": args.layers or None,
                   "mode": mode, "policy": policy, "wire": wire,
                   **({"chosen": _chosen(policy, coms[prune, policy].plan)} if mode == "C" else {}),
                   "lam": args.lam, "rate": bench.RATE,
                   "prune_last": prune, "lookups": looked,
                   "reps": v.params.reps, "columns": v.params.columns,
                   **({"group_columns": dict(v.params.group_columns)} if v.groups else {}),
                   "device": args.device, "verifier_device": args.verifier_device, "threads": args.threads,
                   "accepted": sum(bool(r["accepted"]) for r in runs), "queries": len(runs),
                   "fs": args.fs, "prove_s": prove, "verify_s": verify, **_spread(runs),
                   "proof_bytes": sum(nbytes.values()),
                   "timings": timing, "bytes": nbytes, "build_s": build_s, "commit_s": commit_s.get((prune, policy)),
                   "setup_cache": getattr(coms.get((prune, policy)), "setup_cache", None),  # "hit": commit_s is a load
                   **({"tampers": tampered[prune, policy, looked, wire]} if tampered else {}),
                   "peak": peak, "build_peak": build_peak,
                   "host": platform.node(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
            if first is None:
                first = out
            elif len(pairs) > 1:
                out["vs_first"] = {k: out[k] / first[k] for k in ("prove_s", "verify_s", "proof_bytes", "commit_s")
                                   if first[k]}
                out["vs_first"].update({k: t / first["timings"][k] for k, t in timing.items()
                                        if first["timings"].get(k)})
            ok = ok and out["accepted"] == out["queries"] and all(
                t["rejected"] == t["attempts"] for t in out.get("tampers", {}).values())
            print(json.dumps(out), flush=True)
    if not ok:
        print("FAILED: an honest query was rejected or a tampered one accepted", file=sys.stderr, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

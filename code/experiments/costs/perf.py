"""The timing harness of protocol v0: one process times variants of the protocol on the same queries.

Produces the v0 cells of the run package's ``cpuv`` tier (the CPU verifier of the full models in modes Kpre and C
against re-execution on the same threads: ``tab:options`` and the abstract's ratio, ``tab:eval-reexec``'s verifier
row), its ``gpuv`` tier (the streaming GPU verifier), its ``real`` tier (real-weight OPT and Llama-2-7B next to their
random-weight twins: ``tab:eval-real``) and its ``long`` tier (2,048 to 16,384 tokens: ``tab:eval-long``); and, with
the implementation options of :mod:`pvi.options`, the GPU-verifier and CPU-verifier rows and the prover's fused
attention of ``tab:eval-improve`` (``experiments/improvements/README.md``); with ``--prob-max 255 2047`` the ``improve``
tier's A/B of 8- against 11-bit attention probabilities (``quality.tex``: the proof's size and the prover's and CPU
verifier's times on the same queries).

    python -m experiments.costs.perf --model llama2-7b --seq 2048 --modes Kpre C --queries 5 --verifier-device cpu \\
        --label cpuv --client-memory
    python -m experiments.costs.perf --model opt-1.3b --seq 2048 --modes Kpre --queries 3 --verifier-device cuda \\
        --stream --label gpuv
    python -m experiments.costs.perf --model opt-1.3b --real facebook/opt-1.3b --seq 2048 --modes Kpre C \\
        --queries 3 --tampers 3 --label real
    python -m experiments.costs.perf --model lenet5 --random-init --modes C --policy per-op cnn18c --wire off on
    python -m experiments.costs.perf --model llama2-7b --layers 2 --seq 2048 --modes Kpre C --prob-max 255 2047 \\
        --queries 5 --verifier-device cpu --label prob-bits
    python -m experiments.costs.perf --model tiny-gpt --seq 16 --modes C Kpre --queries 3 --device cpu --threads 2
        # a laptop check (seconds)

Builds the model as ``bench.py`` does (trained image models, or with ``--random-init`` random weights and random
queries; random-weight decoders with ``--seed`` and prompts with ``--prompt-seed``; ``--real``: a checkpoint's integer
graph), commits it, runs one warm-up and ``--queries`` honest queries with ``run_query`` per variant and prints one
JSON line per (mode, variant).  The protocol settings default to the paper's configuration
(:data:`experiments.common.cli.PAPER`: plan ``cnn18c``, Fiat--Shamir, the compact encoding, the lean prover and
verifier, the pruned last block, the verifier's own embedding rows in modes K and Kpre, 11-bit attention
probabilities); ``--policy``, ``--wire``, ``--lean``, ``--prune-last``, ``--lookups`` and ``--prob-max`` take several
values (``--wire off on``, ``--prob-max 255 2047``), and every combination is a variant, its queries interleaved with
the others' (the order rotating from query to query).  The scales of ``--prob-max`` are graphs of the same weights,
built with the same seeds, queried on the same prompts.  Modes K and Kpre open no columns, so they run once per
setting other than the plan; image models have no embedding, no pruned block, no lean path and no attention, and a
checkpoint's recipe fixes its probabilities' scale (``--prob-max`` is for random-weight decoders).

``--real`` (a Hugging Face checkpoint; ``--model`` names its shapes): every block of the checkpoint's integer graph
(:func:`experiments.common.checkpoints.real_graph`), queried on WikiText-2 windows of ``--seq`` tokens.  OPT: the
certified recipe (``hf_opt.CERTIFIED_RECIPE``) by default, or with ``--recipe legacy`` the percentile recipe (norm gain 4,
``--smooth``: SmoothQuant through the norm gains, calibrated on one window of ``--calib-seq`` tokens from the end of
the test split); a Llama-family checkpoint: ``hf_llama.MAIN_RECIPE`` with its fixed public prefix.  Mode C's plans are
made from the graph's own weight ops.  ``--tampers n`` (decoders): after the timed queries every variant runs ``n`` of
each of the trace-tampering attacks of :func:`experiments.common.tampers.decoder_tampers` on the first query, each of
which must be rejected.

``--client-memory`` (a CPU verifier, the compact encoding): the client's own memory.  The prover and the verifier share
this process, so its peak resident set (``peak``) is theirs together.  With this option a separate process
(:class:`experiments.common.client.ClientProcess`, spawned once per mode and variant after the warm-up) holds only what
the client holds -- the public graph and the parameters, mode C's roots, mode Kpre's key and the embedding rows it
reads itself -- and checks each timed proof again from its messages alone (:func:`pvi.v0.query.verify_wire`), right
after the timed query and while this process waits; its verdict must equal this process's.  The key and each proof
reach it through files in a temporary folder under ``--client-dir`` (default: the system's temporary folder), one
proof at a time; the folder is removed when the process ends, also on SIGTERM (a scheduler's time limit or requeue).

Record (one JSON line per mode and variant): ``label``, ``git``, ``model``, ``real``, ``smooth``, ``recipe``,
``calib_seq``, ``seq``, ``layers``, ``mode``, ``policy``, ``chosen`` (mode C: the plan committed, ``auto``'s pick),
``wire``, ``lean``, ``lam``, ``rate``, ``prune_last``, ``lookups``, ``prob_max`` (the graph's attention probabilities'
scale, :func:`pvi.models.decoder.prob_max_of`; None for an image model), ``reps``, ``columns``, ``group_columns``,
``device``, ``verifier_device``, ``stream``, ``threads``, ``accepted`` / ``queries``, ``fs``, ``prove_s`` and
``verify_s`` (sums of the per-part medians, Fiat--Shamir hashing counted for both), ``prove_s_range`` /
``verify_s_range`` (``[min, max]`` of each query's own totals), ``prove_s_all`` / ``verify_s_all`` and
``timings_all`` (every timed query's own totals and parts, in query order), ``transcripts`` (with Fiat--Shamir, each
timed query's transcript digest, :func:`pvi.v0.query.run_query`), ``warmup`` (the untimed warm-up query's totals,
verdict and transcript digest), ``graph_digest``, ``commit_roots_sha256`` (mode C: the SHA-256 of ``C_M``'s roots,
:func:`experiments.common.records.roots_sha256`), ``proof_bytes``, ``timings`` and ``bytes`` (per-part
medians), ``build_s``, ``commit_s``, ``setup_cache``, ``tampers``, ``peak`` / ``build_peak`` (host peak resident set
and GPU peak memory over the queries, and over the build and commitments), ``vs_first`` (several variants: the
medians over the first variant's), ``host``, ``gpu`` and ``env`` (:func:`experiments.common.records.environment`);
with ``--client-memory`` ``client_memory``: the client process's peak resident set ``peak_rss``, its resident set
after the imports ``baseline_rss`` and after loading the key ``key_rss`` (bytes; ``None`` without Linux's ``/proc``),
the proofs it checked (``queries``) and accepted (``accepted``), ``agree`` (every verdict and rejection label equal
to this process's), the median of its verifier's time ``verify_s`` and, if it failed, ``error``.  The exit status
is 1 if an honest query was rejected, a tampered one accepted or the client process's verdicts differ.

Hardware and time: the package runs it on an NVIDIA L40S prover with 8 CPU threads, with job limits of one to two
hours and up to 256 GB of host memory for a 7B decoder at 2,048 tokens (the CPU verifier holds the decoded claims);
a small decoder on a laptop CPU takes seconds.  ``--client-memory`` adds one CPU verification per timed query, the
client process's memory next to this process's (for a 7B decoder at 2,048 tokens about its proof and the decoded
claims) and temporary space for one proof (6.4 GB for Llama-2-7B at 2,048 tokens).
"""
from __future__ import annotations

import argparse
import contextlib
import platform
import signal
import statistics
import sys
import time

import torch

from experiments.common import checkpoints, cli, records
from experiments.common.client import ClientProcess
from experiments.common.models import decoder_config, test_batches, trained_cnn, trained_mlp
from experiments.common.paths import CODE
from experiments.common.tampers import decoder_tampers

PROVE_PARTS = ("prove_forward", "prove_lookups", "prove_fold", "prove_open", "prove_encode", "fs_hash")


def _chosen(policy: str, plan) -> str:
    """The policy whose plan was committed: ``policy`` itself when it has no plan (``per-op``), else
    ``plan.policy`` (``auto``: the ``c`` plan it picked)."""
    return policy if plan is None else plan.policy


def _medians(runs: list[dict]) -> tuple[dict, dict, float, float]:
    """``(timings, bytes, prove_s, verify_s)``: the medians over ``runs`` of ``run_query``."""
    timing = {k: statistics.median(r["timings"].get(k, 0.0) for r in runs)
              for k in sorted({k for r in runs for k in r["timings"]})}
    nbytes = {k: statistics.median(r["bytes"].get(k, 0) for r in runs)
              for k in sorted({k for r in runs for k in r["bytes"]})}
    prove = sum(timing.get(k, 0.0) for k in PROVE_PARTS)
    verify = sum(v_ for k, v_ in timing.items() if k.startswith("verify_") or k == "fs_hash")
    return timing, nbytes, prove, verify


def _totals(runs: list[dict]) -> tuple[list, list]:
    """Each query's own prover and verifier totals (the parts of :func:`_medians`)."""
    prove = [sum(r["timings"].get(k, 0.0) for k in PROVE_PARTS) for r in runs]
    verify = [sum(v_ for k, v_ in r["timings"].items() if k.startswith("verify_") or k == "fs_hash") for r in runs]
    return prove, verify


def _spread(runs: list[dict]) -> dict:
    """``[min, max]`` over ``runs`` of each query's own prover and verifier totals (:func:`_totals`), for the paper's
    median [min-max], and every query's own values (:func:`experiments.common.records.per_query`)."""
    prove, verify = _totals(runs)
    return {"prove_s_range": [min(prove), max(prove)] if runs else None,
            "verify_s_range": [min(verify), max(verify)] if runs else None,
            **records.per_query(runs, prove, verify)}


def _warmup(run: dict) -> dict:
    """The untimed warm-up query's own totals, verdict and transcript digest (its first-call costs: lazy
    initialisation, compilation)."""
    prove, verify = _totals([run])
    return {"prove_s": prove[0], "verify_s": verify[0], "accepted": bool(run["accepted"]),
            "transcript": run.get("transcript")}


def _start_client(stack: contextlib.ExitStack, verifier, folder: str | None, threads: int) -> dict:
    """A variant's client process (:class:`experiments.common.client.ClientProcess`, ended with ``stack``), and
    its verdicts and failure: ``{"process", "got", "error"}``."""
    try:
        return {"process": stack.enter_context(ClientProcess(verifier, folder, threads)), "got": [], "error": None}
    except Exception as e:                     # recorded as a failed check (client_memory.error)
        return {"process": None, "got": [], "error": f"{type(e).__name__}: {e}"}


def _terminated(signum, frame) -> None:
    """SIGTERM as an exit, so that the client processes' ``ExitStack`` removes their temporary folders."""
    raise SystemExit(128 + signum)


def _client_check(st: dict, x: torch.Tensor, proof) -> None:
    """The client process's verdict on one timed proof, appended to ``st["got"]`` (none once it failed)."""
    if st["error"] is None:
        try:
            st["got"].append(st["process"].verify(x, proof))
        except (RuntimeError, OSError, EOFError) as e:
            st["error"] = f"{type(e).__name__}: {e}"


def _client_memory(st: dict, runs: list[dict]) -> dict:
    """The record's ``client_memory``: the client process's peak resident set (it ends here), its resident set
    after the imports and after loading the key, its verdicts on the timed proofs and whether they are this
    process's (``runs``)."""
    c, got, peak = st["process"], st["got"], None
    if st["error"] is None:
        try:
            peak = c.close()["peak_rss"]
        except (RuntimeError, OSError, EOFError) as e:
            st["error"] = f"{type(e).__name__}: {e}"
    ready = c.ready if c is not None else {}
    agree = st["error"] is None and [(g["accepted"], g["rejected_at"]) for g in got] == [
        (r["accepted"], r["rejected_at"]) for r in runs]
    return {"peak_rss": peak, "baseline_rss": ready.get("baseline_rss"), "key_rss": ready.get("key_rss"),
            "queries": len(got), "accepted": sum(bool(g["accepted"]) for g in got), "agree": agree,
            "verify_s": statistics.median(g["verify_s"] for g in got) if got else None,
            **({"error": st["error"]} if st["error"] else {})}


def build_parser() -> argparse.ArgumentParser:
    from pvi.models.quantize import RANDOM_CNNS
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", required=True,
                    help="an image model (" + ", ".join(RANDOM_CNNS) + ") or a decoder (pvi.models.decoder.CONFIGS, "
                         "experiments.common.models.TINY)")
    ap.add_argument("--seq", type=int, default=64, help="decoders: prompt tokens")
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: the full model)")
    ap.add_argument("--modes", nargs="+", default=["C", "Kpre"], choices=["C", "K", "Kpre"])
    ap.add_argument("--queries", type=int, default=10, help="timed queries per variant, after one warm-up")
    ap.add_argument("--label", default="", help="a label copied into every record")
    ap.add_argument("--stream", action="store_true",
                    help="the streaming verifier (Verifier(stream=True); with --verifier-device cuda the paper's GPU "
                         "verifier): one verify_total per query instead of the verify phases it overlaps")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed (decoders)")
    ap.add_argument("--prompt-seed", type=int, default=1, help="the random prompts' seed (decoders)")
    ap.add_argument("--calib-tokens", type=int, default=cli.CALIB_TOKENS,
                    help="calibration tokens of a random-weight decoder (at most --seq)")
    ap.add_argument("--random-init", action="store_true",
                    help="image models: random weights and random queries (pvi.models.quantize.random_cnn; the costs "
                         "depend on the shapes only)")
    ap.add_argument("--real", default="", help="decoders: a Hugging Face checkpoint (e.g. facebook/opt-1.3b), its "
                    "integer graph queried on WikiText-2 windows; --model names its shapes")
    ap.add_argument("--recipe", choices=["certified", "legacy"], default="certified",
                    help="with an OPT --real: certified = hf_opt.CERTIFIED_RECIPE calibrated on four 2,048-token "
                         "train windows; legacy = the percentile recipe (norm gain 4, --smooth, --calib-seq)")
    ap.add_argument("--smooth", type=float, default=None, help="with --recipe legacy: SmoothQuant strength through "
                    "the norm gains (0.55 with experiments.common.checkpoints.SMOOTH's clipping)")
    ap.add_argument("--calib-seq", type=int, default=0,
                    help="with --recipe legacy: the calibration window's tokens (0: --seq)")
    ap.add_argument("--tampers", type=int, default=0, help="decoders, per variant after the timed queries: attempts "
                    "of each trace-tampering attack, each of which must be rejected")
    ap.add_argument("--client-memory", action="store_true",
                    help="check each timed proof again in a separate process that holds only what the client holds "
                         "(experiments.common.client) and record that process's peak resident memory "
                         "(client_memory); a CPU verifier, the variants with the compact encoding")
    ap.add_argument("--client-dir", default=None, metavar="DIR",
                    help="with --client-memory: the folder of the client's key and proofs (default: the system's "
                         "temporary folder)")
    cli.add_protocol_arguments(ap, variants=True)
    cli.add_runtime_arguments(ap, threads=8)
    return ap


def main() -> int:
    from pvi.models.decoder import prob_max_of
    from pvi.models.quantize import RANDOM_CNNS, random_cnn, random_inputs
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier

    args = build_parser().parse_args()
    if args.real and args.recipe == "certified" and (args.smooth is not None or args.calib_seq):
        raise SystemExit("--smooth and --calib-seq are the legacy recipe's: pass --recipe legacy")
    scales = list(dict.fromkeys(args.prob_max))
    if (args.real or args.model in RANDOM_CNNS) and scales != [cli.PAPER.prob_max]:
        raise SystemExit("--prob-max is for random-weight decoders: an image model has no attention, and a "
                         "checkpoint's recipe fixes its scale")
    if args.client_memory:
        if torch.device(args.verifier_device).type != "cpu" or args.stream:
            raise SystemExit("--client-memory measures the CPU verifier's process: --verifier-device cpu, no --stream")
        if "on" not in args.wire:
            raise SystemExit("--client-memory sends the client the proof's bytes: --wire on")
        if "C" in args.modes and not args.fs:
            raise SystemExit("--client-memory checks mode C's proofs from their Fiat-Shamir transcript: --fs")
        signal.signal(signal.SIGTERM, _terminated)
    cli.apply_runtime(args)
    sha = records.git_sha(CODE)              # before the build, which can outlast a checkout update
    device = torch.device(args.device)
    on = {k: [w == "on" for w in getattr(args, k)] for k in ("wire", "lean", "prune_last", "lookups")}
    records.reset_peaks(device)
    t0 = time.perf_counter()
    graphs, model_ops = {}, {}
    cnn = args.model in RANDOM_CNNS
    if cnn:                      # no embedding, no pruned block, no lean path: those settings do not apply
        if args.real or args.tampers:
            raise SystemExit("--real and --tampers are for the decoders")
        on.update(lean=[False], prune_last=[False], lookups=[False])
        if args.random_init:
            graph, x = random_cnn(args.model)
            queries = random_inputs(graph, x, args.queries + 1)
        else:
            from pvi.models.quantize import quantize_input, quantize_model
            model, data = trained_mlp() if args.model == "mlp_mnist" else trained_cnn(args.model)
            graph = quantize_model(model, data["train_x"])
            xs, _ = next(test_batches(data, args.queries + 1))
            queries = list(quantize_input(graph, xs).split(1))
        graphs[False, None], model_ops[False] = graph, None
        n_checks = len(graph.mat_ops)
        seq = None
    elif args.real:                 # the checkpoint's integer graph, every block; WikiText-2 windows as queries
        if args.layers:
            raise SystemExit("--real builds every block of the checkpoint")
        model, queries, calib = checkpoints.load_real(args.real, args.seq, args.queries + 1, args.calib_seq,
                                                      args.recipe)
        recipe_used = checkpoints.recipe_name(calib, args.recipe)
        for prune in on["prune_last"]:
            graph, _ = checkpoints.real_graph(model, calib, prune, args.recipe, args.smooth)
            graphs[prune, prob_max_of(graph)] = graph
            model_ops[prune] = None    # planned from its own weight ops (OPT's LM head has a bias column)
        del model
        n_checks = len(next(iter(graphs.values())).mat_ops)
        seq = args.seq
    else:
        from pvi.commit.cost import decoder_shapes
        from pvi.models.decoder import build_decoder
        cfg = decoder_config(args.model)
        n_layers = args.layers or cfg.n_layers
        for prune in on["prune_last"]:
            for scale in scales:
                graphs[prune, scale] = build_decoder(cfg, n_layers=n_layers,
                                                     calib_tokens=min(args.seq, args.calib_tokens), seed=args.seed,
                                                     prune_last=prune, prob_max=scale)
            model_ops[prune] = decoder_shapes(cfg, prune_last=prune)
        tokens = torch.randint(0, cfg.vocab, (args.queries + 1, args.seq),
                               generator=torch.Generator().manual_seed(args.prompt_seed))
        queries = list(tokens.split(1))
        n_checks = len(model_ops[on["prune_last"][0]])   # (r, t) for the full model, as bench.py
        seq = args.seq
    build_s = time.perf_counter() - t0
    policies = args.policy if "C" in args.modes else []
    coms, commit_s = {}, {}
    for (prune, scale), graph in graphs.items():
        for policy in policies:
            records.sync(device)
            t0 = time.perf_counter()
            coms[prune, scale, policy] = commit_graph(graph, args.rate, device=device, policy=policy,
                                                      model_ops=model_ops[prune], cache=args.setup_cache)
            records.sync(device)
            commit_s[prune, scale, policy] = time.perf_counter() - t0
    build_peak = records.peaks(device)
    digests = {key: graph.digest().hex() for key, graph in graphs.items()}
    roots = {key: records.roots_sha256(c)[0] for key, c in coms.items()}
    env = records.environment(device)
    ok = client_ok = True
    for mode in args.modes:
        records.reset_peaks(device)
        pairs, warm = {}, {}
        for (prune, scale), graph in graphs.items():
            weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
            for policy in (policies if mode == "C" else [None]):
                c = coms.get((prune, scale, policy))
                params = params_for(args.lam, n_checks, rate=args.rate, plan=None if c is None else c.plan,
                                    fiat_shamir=args.fs)
                for looked in (on["lookups"] if mode != "C" else [False]):
                    for lean in on["lean"]:
                        for wire in on["wire"]:
                            if mode == "C":
                                v = Verifier(graph.public(), params, "C", publics=c.publics, groups=c.group_publics,
                                             tables=c.table_publics, lean=lean, device=args.verifier_device,
                                             stream=args.stream)
                            else:
                                v = Verifier(graph.public(), params, mode, weights=weights, lean=lean,
                                             device=args.verifier_device, lookups=looked, stream=args.stream)
                                if mode == "Kpre":
                                    v.precompute(Challenger())
                            prover = Prover(graph, device=device, commitments=c, lean=lean)
                            key = (prune, scale, policy, looked, lean, wire)
                            warm[key] = _warmup(run_query(prover, v, queries[0], wire=wire))  # untimed
                            pairs[key] = (prover, v, [])
        order = list(pairs)
        with contextlib.ExitStack() as stack:
            clients = {key: _start_client(stack, v, args.client_dir, args.threads)
                       for key, (_, v, _) in pairs.items() if args.client_memory and key[5]}
            for i, q in enumerate(queries[1:]):     # interleaved, the first variant rotating
                for key in order[i % len(order):] + order[:i % len(order)]:
                    prover, v, runs = pairs[key]
                    runs.append(run_query(prover, v, q, wire=key[5], keep_proof=key in clients))
                    if key in clients:                # the client checks the proof again while this process waits
                        _client_check(clients[key], q, runs[-1].pop("proof"))
            memory = {key: _client_memory(st, pairs[key][2]) for key, st in clients.items()}
        tampered = {key: decoder_tampers(prover, v, queries[0], args.tampers, key[5])
                    for key, (prover, v, _) in pairs.items()} if args.tampers else {}
        peak = records.peaks(device)
        first = None
        for key, (_, v, runs) in pairs.items():
            prune, scale, policy, looked, lean, wire = key
            timing, nbytes, prove, verify = _medians(runs)
            legacy = args.real and args.recipe == "legacy"
            out = {"label": args.label, "git": sha, "model": args.model, "real": args.real or None,
                   "smooth": args.smooth if legacy else None,
                   "recipe": recipe_used if args.real else None,
                   "calib_seq": (args.calib_seq or args.seq) if legacy else None,
                   "seq": seq, "layers": args.layers or None,
                   "mode": mode, "policy": policy, "wire": wire, "lean": lean,
                   **({"chosen": _chosen(policy, coms[prune, scale, policy].plan)} if mode == "C" else {}),
                   "lam": args.lam, "rate": args.rate,
                   "prune_last": prune, "lookups": looked, "prob_max": scale,
                   "reps": v.params.reps, "columns": v.params.columns,
                   **({"group_columns": dict(v.params.group_columns)} if v.groups else {}),
                   "device": args.device, "verifier_device": args.verifier_device, "stream": args.stream,
                   "threads": args.threads,
                   "accepted": sum(bool(r["accepted"]) for r in runs), "queries": len(runs),
                   "fs": args.fs, "prove_s": prove, "verify_s": verify, **_spread(runs),
                   "proof_bytes": sum(nbytes.values()),
                   "timings": timing, "bytes": nbytes, "build_s": build_s,
                   "commit_s": commit_s.get((prune, scale, policy)),
                   "setup_cache": getattr(coms.get((prune, scale, policy)), "setup_cache", None),  # "hit": a load
                   "warmup": warm[key], "graph_digest": digests[prune, scale],
                   **({"commit_roots_sha256": roots[prune, scale, policy]} if (prune, scale, policy) in roots else {}),
                   **({"tampers": tampered[key]} if tampered else {}),
                   "peak": peak, "build_peak": build_peak,
                   **({"client_memory": memory[key]} if key in memory else {}),
                   "host": platform.node(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
                   "env": env}
            if first is None:
                first = out
            elif len(pairs) > 1:
                out["vs_first"] = {k: out[k] / first[k] for k in ("prove_s", "verify_s", "proof_bytes", "commit_s")
                                   if first[k]}
                out["vs_first"].update({k: t / first["timings"][k] for k, t in timing.items()
                                        if first["timings"].get(k)})
            ok = ok and out["accepted"] == out["queries"] and all(
                t["rejected"] == t["attempts"] for t in out.get("tampers", {}).values())
            client_ok = client_ok and (out.get("client_memory") or {}).get("agree", True)
            records.emit(out)
    if not ok:
        print("FAILED: an honest query was rejected or a tampered one accepted", file=sys.stderr, flush=True)
    if not client_ok:
        print("FAILED: the client process's verdicts differ from this process's, or it failed (client_memory)",
              file=sys.stderr, flush=True)
    return 0 if ok and client_ok else 1


if __name__ == "__main__":
    sys.exit(main())

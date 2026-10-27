"""V1 against protocol v0 on the same model, query and parameters: proof bytes, prover and verifier times.

Produces ``tab:eval-v1`` (Sec. ``sec:eval-v1``: V1's and v0's proofs, their ratio, and the prover and verifier
times), V1's real-weight cells (the real OPT and Llama-2-7B rows next to their random-weight twins, and V1's
1.7x on the real OPT-6.7B in the abstract), ``tab:options``' V1 columns and the V1 verifier's time before and after
its optimised paths (the run package's ``cpuv`` tier, ``--v1-verifier optimised baseline``; the extended version's
"1.44 s per block (2.49 s before)" and the 45.4 s of the full Llama-2-7B).

    python -m experiments.v1.v1_bytes --model llama2-7b --seq 2048 --modes Kpre C --full --reps 5
    python -m experiments.v1.v1_bytes --model llama2-7b --seq 2048 --modes Kpre --full --reps 5 \\
        --v1-verifier optimised baseline --label cpuv-v1
    python -m experiments.v1.v1_bytes --model opt-6.7b --real facebook/opt-6.7b --seq 2048 --modes Kpre --full \\
        --reps 3 --tampers 3 --label real
    python -m experiments.v1.v1_bytes --model opt-6.7b --real facebook/opt-6.7b --seq 2048 --codec-only --device cpu
    python -m experiments.v1.v1_bytes --model tiny-llama --seq 16 --modes C Kpre --full --reps 2 --tampers 2 \\
        --lmax 20 --device cpu --threads 2          # a laptop check (seconds)

Builds the decoder as ``costs/perf.py`` does (random weights, ``--seed``; the parameters ``(r, t)`` of the full
model; ``--layers`` blocks), commits it in mode C under ``--policy``, runs one v0 query (``run_query``) and measures
V1's bytes on the same query with :func:`pvi.v1.protocol.cut_proof_bytes` (M1 encoded for real, the other messages
sized exactly).  ``--full`` also runs the whole V1 query (:func:`pvi.v1.protocol.run_cut_query`, the GKR on the GPU
in Triton) and checks its bytes; with ``--reps n`` it then times ``n`` queries of each protocol, interleaved on the
same prompts, after one warm-up.  The protocol settings default to the paper's configuration
(:data:`experiments.common.cli.PAPER`: plan ``cnn18c``, Fiat--Shamir, the lean prover and verifier, the last block at
the last position, mode Kpre's verifier reading the embedding rows itself, 11-bit attention probabilities); V1
always sends its proof in the compact encoding, and ``--wire`` concerns v0's proof only.  ``--lmax`` (default
:data:`pvi.v1.cut.LMAX` = 27, the package's and the measured proofs' value) is the GKR instances' target size
``2^lmax``; an operation larger than that forms an instance of its own, so the record lists every instance's size
(``instance_bits``).

**Checks.** Every line checks the byte accounting (``accounting``, ``accounting_ok``): v0's ``u`` and opened columns
(and, off the wire, its claims) are :func:`pvi.commit.cost.proof_bytes` of the graph's weight ops exactly, with the
multiproofs' ratio to their expectation (``paths_ratio``; on the wire v0's claims are measured, so they are no
check); with ``--full``, every V1 message but the multiproofs is ``cut_proof_bytes``'s (V1's claims included: both
encode the same values); and the graph's weight-op shapes and claim columns are those of
:func:`pvi.commit.cost.decoder_shapes` / ``decoder_claim_columns`` (``shapes``; a real OPT checkpoint's LM head has
one more column, its folded bias).  ``--tampers n`` (with ``--full``): per variant, ``n`` trace-tampering V1 queries
(a cut op's claim moved by whole windows, everything after it recomputed honestly) and ``n`` single-value ones (any
weight op, any entry, +-1..999), each of which must be rejected.  A failed check is printed and makes the exit
status 1.

**V1's verifier.** ``--v1-verifier`` takes one or more of :data:`pvi.options.V1_VERIFIERS`: ``optimised`` (the
window-table products, the batched GKR check and the batch-inverted logUp identity; the default) and ``baseline``
(float64 limb products, one check per instance: the verifier before those changes).  The untimed V1 query, the byte
sizing and the tamper queries run under ``optimised`` when it is listed (else the first one given); with ``--full
--reps n`` every listed verifier checks the same timed queries, after one warm-up each (``timed.v1_paths``, by name;
``timed.v1`` is ``timed.v1_path``'s).  Both verifiers give the same verdicts on the same bytes.

``--real`` (a Hugging Face checkpoint; ``--model`` names its shapes): every block of the checkpoint's integer graph
(:func:`experiments.common.checkpoints.real_graph`), or with ``--layers n`` a Llama-family checkpoint's (Llama, Qwen3)
first ``n`` blocks with the same calibration and prefix (refused for OPT, whose builder takes every block; planned and
sized for the whole model, as a random-weight build), queried on WikiText-2 test windows of ``--seq`` tokens.  OPT:
the certified recipe (``hf_opt.CERTIFIED_RECIPE``, calibrated on four 2,048-token train windows) by default, or with
``--recipe legacy`` the percentile recipe (norm gain 4, ``--smooth``: SmoothQuant through the norm gains, calibrated on one
window of ``--calib-seq`` tokens from the end of the test split); a Llama-family checkpoint: ``hf_llama.MAIN_RECIPE``
with its fixed public prefix.  ``--codec-only``: no commitment and no proof -- one forward pass (``--device``) whose
claims are encoded op by op as they are produced (v0: ``PVC3`` of the claims; V1: of the cut ops' requantised values
and the other ops' claims, and the exceptions frame), summed per block: the byte profile of a model whose prover the
machine cannot hold.

**Record** (one JSON line per mode, on stdout): ``label``, ``git``, ``model``, ``real``, ``smooth``, ``recipe``,
``calib_seq``, ``layers``, ``seq``, ``prune_last``, ``prob_max`` (the graph's attention probabilities' scale:
``--prob-max``, or a checkpoint recipe's), ``lmax``, ``graph_digest``, ``shapes``, ``build_s``,
``build_peak``, ``host``, ``gpu``, ``env`` (:func:`experiments.common.records.environment`), ``mode``, ``policy``,
``chosen`` (mode C: the plan committed), ``fs``, ``wire``, ``lean``, ``lookups``, ``lam``, ``rate``,
``v1_verifier``, ``v0_accepted``, ``v0_transcript`` (with Fiat--Shamir, the untimed v0 query's transcript digest,
:func:`pvi.v0.query.run_query`), ``commit_roots_sha256`` (mode C: :func:`experiments.common.records.roots_sha256`),
``v0_bytes`` and ``v1_bytes`` (per message), ``v0_total``, ``v1_total``,
``ratio``, ``n_leaves``, ``n_instances``, ``instance_bits``, ``table``, ``n_exceptions``, ``reps``, ``columns``,
``setup_s``, ``setup_cache``, ``v0_query_s``, ``v1_bytes_s``; with ``--full`` ``v1_full`` (verdict, bytes, timings,
the transcript digest), ``timed`` (``queries``; ``v0``, ``v1`` and ``v1_paths``: the per-part medians, ``prove_s`` /
``verify_s`` with their ``[min, max]`` over the queries' own totals, ``fs_s``, and every query's own values
``prove_s_all``, ``verify_s_all``, ``timings_all`` and ``transcripts``) and ``tampers``; ``accounting``, ``accounting_ok``, ``peak``.
With ``--codec-only`` one line with ``codec_only``, ``blocks`` and ``total`` instead.

Hardware and time: the paper's cells ran on an NVIDIA L40S (48 GB) prover with 8 CPU threads for the verifier.  A
7B decoder at 2,048 tokens with ``--full`` takes one to three hours and up to 256 GB of host memory; GPT-2 at 64
tokens minutes; the laptop check above seconds on a CPU (the GKR then runs in PyTorch).
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time

import torch

from experiments.common import checkpoints, cli, real_cells, records
from experiments.common.models import decoder_config
from experiments.common.paths import CODE
from pvi import options
from pvi.v1.cut import LMAX


# ------------------------------------------------------------------ the accounting
def shape_check(graph, cfg, n_layers: int, prune_last: bool, real: bool, x: torch.Tensor) -> dict:
    """``graph``'s weight ops against ``cost.decoder_shapes(cfg, n_layers, prune_last)`` (rows and row length, in
    order; a real OPT checkpoint's LM head has one more column, its final LayerNorm's folded bias, while a real
    Llama-family head (final RMSNorm, no bias) has the benchmark's row length) and its claim columns on ``x`` against
    ``decoder_claim_columns``."""
    from pvi.commit.cost import decoder_claim_columns, decoder_shapes
    shapes = decoder_shapes(cfg, n_layers, prune_last)
    mats = graph.mat_ops
    if len(shapes) != len(mats):
        return {"problems": [f"{len(mats)} weight ops, the model {len(shapes)}"], "claim_columns": False}
    problems = []
    for s, op in zip(shapes, mats):
        folded = real and s.name == "head" and cfg.norm == "layernorm"
        want = (s.n_rows, s.row_length + (1 if folded else 0))
        if (op.n_rows, op.row_length) != want:
            problems.append(f"{op.name} ({s.name}): {(op.n_rows, op.row_length)}, the model {want}")
    cols = graph.claim_columns(x)
    return {"problems": problems,
            "claim_columns": cols == list(decoder_claim_columns(cfg, x.shape[1], n_layers, prune_last).values())}


def v0_accounting(graph, ref: dict, params, plan, mode: str, x: torch.Tensor, table_ids: dict, wire: bool) -> dict:
    """``run_query``'s bytes ``ref`` against ``cost.proof_bytes`` of ``graph``'s weight ops on ``x``: ``u`` and the
    opened columns exactly, and the claims too off the wire; the multiproofs' ratio to their expectation (exact for
    the ids ``table_ids`` a plan's lookup tables looked up).  On the wire the model takes the claims' measured
    ``PVC3`` size (``wire_claims``), so comparing them would hold by construction."""
    from pvi.commit.cost import proof_bytes
    cols = dict(zip((op.name for op in graph.mat_ops), graph.claim_columns(x)))
    want = proof_bytes(graph.mat_ops, params, cols, plan=plan, mode=mode,
                       wire_claims=ref["bytes"]["claims"] if wire else None, table_ids=table_ids)
    out = {k: ref["bytes"][k] == want[k] for k in (V0_EXACT if wire else ("claims",) + V0_EXACT)}
    out["paths_ratio"] = ref["bytes"]["paths"] / want["paths"] if want["paths"] else None
    return out


V0_EXACT = ("u", "columns")
V1_EXACT = ("claims", "cut_exc", "cut_mult", "cut_gkr", "u", "columns")


def v1_accounting(est: dict, full: dict) -> dict:
    """``cut_proof_bytes``'s ``est`` against a real V1 query's bytes: every message exactly but the multiproofs,
    whose ratio is reported."""
    out = {k: est[k] == full["bytes"][k] for k in V1_EXACT}
    out["paths_ratio"] = full["bytes"]["paths"] / est["paths"] if est["paths"] else None
    return out


# ------------------------------------------------------------------ tampering (V1)
def v1_tampers(prover, verifier, x: torch.Tensor, n: int, seed: int = 0) -> dict:
    """``n`` trace-tampering V1 queries -- a random entry of a random cut op's claim moved by 1-3 widths of its normal
    window (another requantised value), everything after it recomputed honestly -- and ``n`` single-value ones (a
    random entry of a random weight op, +-1..999): ``{kind: {attempts, rejected, labels}}``."""
    from pvi.v1 import protocol as cp
    plan = cp.cut_plan(verifier.graph, verifier.claim_columns(x), x.shape[1],
                       verifier.publics if verifier.groups else None, verifier.params.cut_lmax,
                       verifier.weights or None)
    mats = prover.graph.mat_ops
    g = torch.Generator().manual_seed(seed)
    out = {k: {"attempts": 0, "rejected": 0, "labels": {}} for k in ("trace", "single")}

    def draw(hi: int) -> int:
        return int(torch.randint(hi, (1,), generator=g))

    def attempt(kind, name, pos, delta):
        def tamper(op, z):
            if op.name != name:
                return z
            z = z.clone()
            z.view(-1)[pos % z.numel()] += delta
            return z
        res = cp.run_cut_query(prover, verifier, x, forward_kwargs={"tamper": tamper})
        rec = out[kind]
        rec["attempts"] += 1
        rec["rejected"] += int(not res["accepted"])
        label = str(res["rejected_at"])
        rec["labels"][label] = rec["labels"].get(label, 0) + 1

    for i in range(n):
        if plan.ops:
            o = plan.ops[draw(len(plan.ops))]
            attempt("trace", o.name, draw(1 << 30), (1 + i % 3) * o.widths[1] * (1 if i % 2 else -1))
        attempt("single", mats[draw(len(mats))].name, draw(1 << 30), (1 + draw(999)) * (1 if i % 2 else -1))
    return out


# ------------------------------------------------------------------ the codec-only byte profile
def _blocks(graph, n_layers: int) -> dict:
    """Each weight op's block: ``"embed"`` before the first norm, then ``b`` for the ops after norms ``2b`` and
    ``2b + 1`` (each block has two), ``"head"`` after the final norm."""
    out, norms = {}, 0
    for op in graph.ops:
        if getattr(op, "note", "") == "norm":
            norms += 1
        elif hasattr(op, "weight"):
            b = (norms - 1) // 2
            out[op.name] = "embed" if norms == 0 else "head" if b >= n_layers else b
    return out


def codec_profile(graph, x: torch.Tensor, n_layers: int, lmax: int, device) -> dict:
    """The claims of one forward pass on ``x``, encoded op by op as the pass produces them (no commitment, no
    proof), summed per block: entries and raw bytes (4 each), v0's ``PVC3`` bytes, V1's (the requantised values of
    the ops cut without a commitment plan, as in mode Kpre -- ``CutPlan.from_graph`` without col matrices, P4 from
    the graph's weights -- and the other ops' claims) and the cut ops' exceptions; the totals add V1's exceptions
    frame (``PVX1``).  Each op is encoded on its own (its own header, Rice parameters and exception list), so the
    sums differ slightly from one encoding of all the claims, which a protocol line (without ``--codec-only``) gives
    exactly."""
    from pvi.v0 import claimcodec
    from pvi.v1 import cut as cutmod
    from pvi.v1.protocol import pack_exceptions
    T = x.shape[1]
    plan = cutmod.CutPlan.from_graph(graph, graph.claim_columns(x), T, None, lmax=lmax)
    cut = {o.name: o for o in plan.ops}
    block_of = _blocks(graph, n_layers)
    rows: dict = {}
    exc: dict = {}
    mats = iter(graph.mat_ops)

    def send(z):
        op = next(mats)
        zc = z.to("cpu")
        rec = rows.setdefault(block_of[op.name], {"claims": 0, "raw": 0, "v0_claims": 0, "v1_values": 0,
                                                  "cut_claims": 0, "exceptions": 0})
        v0 = len(claimcodec.encode([zc]))
        rec["claims"] += zc.numel()
        rec["raw"] += 4 * zc.numel()
        rec["v0_claims"] += v0
        if op.name in cut:
            s, _, _, idx, ez = cutmod.split(zc, cut[op.name])
            exc[op.name] = (idx, ez)
            rec["v1_values"] += len(claimcodec.encode([s]))
            rec["cut_claims"] += zc.numel()
            rec["exceptions"] += int(idx.numel())
        else:
            rec["v1_values"] += v0
        return None

    t0 = time.perf_counter()
    with torch.no_grad():
        graph.forward(x.to(device), free=True, send=send)
    blocks = [dict(block=b, **rec) for b, rec in rows.items()]
    total = {k: sum(r[k] for r in blocks) for k in blocks[0] if k != "block"}
    total["v1_exceptions_frame"] = len(pack_exceptions(plan, exc))
    return {"blocks": blocks, "total": total, "n_cut_ops": len(plan.ops), "n_instances": len(plan.instances),
            "table": plan.table_size(), "forward_encode_s": time.perf_counter() - t0}


# ------------------------------------------------------------------ timed queries
def _medians(runs: list[dict]) -> dict:
    """The per-part medians of ``runs``, the prover's and the verifier's totals (sums of the medians) with the
    ``[min, max]`` of each query's own totals, the accepted count, the Fiat--Shamir hashing (``fs_s``) and every
    query's own values (:func:`experiments.common.records.per_query`: ``prove_s_all``, ``verify_s_all``,
    ``timings_all``, ``transcripts``)."""
    keys = sorted({k for r in runs for k in r["timings"]})
    t = {k: statistics.median(r["timings"].get(k, 0.0) for r in runs) for k in keys}
    pq = [sum(v for k, v in r["timings"].items() if k.startswith("prove_")) for r in runs]
    vq = [sum(v for k, v in r["timings"].items() if k.startswith("verify_")) for r in runs]
    return {"accepted": sum(r["accepted"] for r in runs), "timings": t,
            "prove_s": sum(v for k, v in t.items() if k.startswith("prove_")),
            "verify_s": sum(v for k, v in t.items() if k.startswith("verify_")),
            "prove_s_range": [min(pq), max(pq)], "verify_s_range": [min(vq), max(vq)],
            "fs_s": t.get("fs_hash", 0.0), **records.per_query(runs, pq, vq)}


def build_parser() -> argparse.ArgumentParser:
    # "resolve": --v1-verifier below takes several values in place of add_runtime_arguments' single one
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], conflict_handler="resolve")
    ap.add_argument("--model", required=True,
                    help="the decoder's shapes (pvi.models.decoder.CONFIGS, experiments.common.models.TINY)")
    ap.add_argument("--seq", type=int, default=64, help="prompt tokens")
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: the full model)")
    ap.add_argument("--modes", nargs="+", default=["C", "Kpre"], choices=["C", "Kpre"])
    ap.add_argument("--lmax", type=int, default=LMAX,
                    help="V1's GKR instances: the target size 2^lmax (an operation larger than that is an instance "
                         "of its own)")
    ap.add_argument("--full", action="store_true",
                    help="also run the whole V1 query (the GKR on the GPU in Triton) and check its bytes")
    ap.add_argument("--reps", type=int, default=1, help="with --full: timed queries of each protocol after a warm-up")
    ap.add_argument("--tampers", type=int, default=0, help="with --full: trace-tampering and single-value V1 queries "
                    "per mode, each of which must be rejected")
    ap.add_argument("--codec-only", action="store_true", help="no commitment and no proof: the claims of one forward "
                    "pass encoded op by op (v0 and V1), per block")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed")
    ap.add_argument("--calib-tokens", type=int, default=cli.CALIB_TOKENS,
                    help="calibration tokens of the random-weight decoder (at most --seq)")
    ap.add_argument("--prompt-seed", type=int, default=1, help="the seed of the random prompt of the untimed queries")
    ap.add_argument("--timed-seed", type=int, default=100,
                    help="timed random prompt i is drawn with the seed --timed-seed + i")
    ap.add_argument("--label", default="", help="a label copied into every record")
    ap.add_argument("--real", default="", help="a Hugging Face checkpoint (e.g. facebook/opt-1.3b): its integer graph, "
                    "queried on WikiText-2 windows; --model names its shapes")
    ap.add_argument("--recipe", choices=["certified", "legacy"], default="certified",
                    help="with an OPT --real: certified = hf_opt.CERTIFIED_RECIPE calibrated on four 2,048-token "
                         "train windows; legacy = the percentile recipe (norm gain 4, --smooth, --calib-seq)")
    ap.add_argument("--smooth", type=float, default=None, help="with --recipe legacy: SmoothQuant strength through "
                    "the norm gains (0.55 with experiments.common.checkpoints.SMOOTH's clipping)")
    ap.add_argument("--calib-seq", type=int, default=0,
                    help="with --recipe legacy: the calibration window's tokens (0: --seq)")
    cli.add_protocol_arguments(ap)
    cli.add_runtime_arguments(ap, threads=8)
    ap.add_argument("--v1-verifier", nargs="+", choices=options.V1_VERIFIERS, default=[options.current().v1_verifier],
                    help="V1's verifier: optimised (window-table products, batched GKR check, batch-inverted logUp "
                         "identity) or baseline (float64 limb products, one check per instance); with --full --reps "
                         "n each one given checks the same timed queries")
    return ap


def main() -> int:
    from pvi.commit.cost import decoder_shapes
    from pvi.models.decoder import build_decoder, prob_max_of
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier
    from pvi.v1 import cut as cutmod
    from pvi.v1 import protocol as cp

    args = build_parser().parse_args()
    if args.tampers and not args.full:
        raise SystemExit("--tampers runs V1 queries: it goes with --full")
    verifiers = list(dict.fromkeys(args.v1_verifier))
    if len(verifiers) > 1 and not (args.full and args.reps > 1):
        raise SystemExit("several --v1-verifier values time V1 queries: they go with --full --reps n (n > 1)")
    if args.real and args.recipe == "certified" and (args.smooth is not None or args.calib_seq):
        raise SystemExit("--smooth and --calib-seq are the legacy recipe's: pass --recipe legacy")
    if args.real and args.prob_max != cli.PAPER.prob_max:
        raise SystemExit("--prob-max is for random-weight decoders: a checkpoint's recipe fixes its scale")
    main_verifier = "optimised" if "optimised" in verifiers else verifiers[0]
    args.v1_verifier = main_verifier                 # the untimed V1 work runs under it
    cli.apply_runtime(args)
    sha = records.git_sha(CODE)
    cfg = decoder_config(args.model)
    n_layers = args.layers or cfg.n_layers
    records.reset_peaks(args.device)
    t0 = time.perf_counter()
    recipe_used, legacy = None, args.real and args.recipe == "legacy"
    if args.real:                     # the checkpoint's weights; queries are WikiText-2 windows
        part = real_cells.partial(cfg, args.layers)     # None: every block; else a Llama checkpoint's first ones
        model, wins, calib = checkpoints.load_real(args.real, args.seq, max(2, args.reps + 1), args.calib_seq,
                                                   args.recipe)
        graph, _ = checkpoints.real_graph(model, calib, args.prune_last, args.recipe, args.smooth, n_layers=part)
        recipe_used = checkpoints.recipe_name(calib, args.recipe)
        del model
        n_layers = len([op for op in graph.ops if getattr(op, "note", "") == "attention"])
        x = wins[0]
        real_queries = wins[1:]
    else:
        graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(args.seq, args.calib_tokens), seed=args.seed,
                              prune_last=args.prune_last, prob_max=args.prob_max)
        x = torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(args.prompt_seed))
        real_queries = None
    build_s = time.perf_counter() - t0
    head = {"label": args.label, "git": sha, "model": args.model, "real": args.real or None,
            "smooth": args.smooth if legacy else None, "recipe": recipe_used,
            "calib_seq": (args.calib_seq or args.seq) if legacy else None, "layers": n_layers, "seq": args.seq,
            "prune_last": args.prune_last, "prob_max": prob_max_of(graph), "lmax": args.lmax,
            "graph_digest": graph.digest().hex(),
            "shapes": shape_check(graph, cfg, n_layers, args.prune_last, bool(args.real), x), "build_s": build_s,
            "build_peak": records.peaks(args.device), "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "env": records.environment(args.device)}
    shapes_ok = not head["shapes"]["problems"] and head["shapes"]["claim_columns"]
    if args.codec_only:
        records.reset_peaks(args.device)
        records.emit({**head, "codec_only": True, "device": args.device,
                      **codec_profile(graph, x, n_layers, args.lmax, args.device), "peak": records.peaks(args.device)})
        return 0 if shapes_ok else 1
    ok = shapes_ok
    # a real checkpoint's full graph (e.g. OPT's LM head has a bias) is planned from its own ops; a build of a few
    # blocks (random weights, or a Llama-family checkpoint's first blocks) for the whole model's shapes
    model_ops = (None if args.real and n_layers == cfg.n_layers
                 else decoder_shapes(cfg, prune_last=args.prune_last))
    n_checks = len(model_ops) if model_ops is not None else len(graph.mat_ops)
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    for mode in args.modes:
        records.reset_peaks(args.device)
        t0 = time.perf_counter()
        if mode == "C":
            coms = commit_graph(graph, args.rate, device=args.device, policy=args.policy, model_ops=model_ops,
                                cache=args.setup_cache)
            plan = coms.plan
            kw = dict(publics=coms.publics, groups=coms.group_publics, tables=coms.table_publics)
            prover = Prover(graph, device=args.device, commitments=coms, lean=args.lean)
        else:
            coms, plan, kw = None, None, dict(weights=weights, lookups=args.lookups)
            prover = Prover(graph, device=args.device, lean=args.lean)
        setup_s = time.perf_counter() - t0
        p0 = params_for(args.lam, n_checks, rate=args.rate, fiat_shamir=args.fs, plan=plan)
        p1 = params_for(args.lam, n_checks, rate=args.rate, fiat_shamir=args.fs, plan=plan, cut=True,
                        cut_lmax=args.lmax)
        v0 = Verifier(graph.public(), p0, mode, lean=args.lean, device=args.verifier_device, **kw)
        v1 = Verifier(graph.public(), p1, mode, lean=args.lean, device=args.verifier_device, **kw)
        if mode == "Kpre":
            v0.precompute(Challenger())
            v1._pre = v0._pre                        # the same secret rows (V1 reads them for the clear ops)
        t0 = time.perf_counter()
        ref = run_query(prover, v0, x, wire=args.wire)
        v0_s = time.perf_counter() - t0
        # the ids the plan's lookup tables looked up on this query (the prover keeps them for its openings)
        accounting = {"v0": v0_accounting(graph, ref, p0, plan, mode, x, dict(prover._looked_up), args.wire)}
        t0 = time.perf_counter()
        b1 = cp.cut_proof_bytes(prover, v1, x, seed=7)
        v1_s = time.perf_counter() - t0
        stats = {k: b1.pop(k) for k in ("n_leaves", "n_instances", "table", "n_exceptions")}
        cut_plan = cp.cut_plan(v1.graph, v1.claim_columns(x), x.shape[1], v1.publics if v1.groups else None,
                               args.lmax, v1.weights or None)
        total0, total1 = sum(ref["bytes"].values()), sum(b1.values())
        out = {**head, "mode": mode, "policy": args.policy if mode == "C" else None,
               "chosen": None if mode != "C" else args.policy if plan is None else plan.policy, "fs": args.fs,
               "wire": args.wire, "lean": args.lean, "lookups": args.lookups if mode != "C" else False,
               "lam": args.lam, "rate": args.rate, "v1_verifier": main_verifier,
               "v0_accepted": ref["accepted"], "v0_transcript": ref.get("transcript"),
               **({"commit_roots_sha256": records.roots_sha256(coms)[0]} if coms is not None else {}),
               "v0_bytes": ref["bytes"], "v1_bytes": b1, "v0_total": total0,
               "v1_total": total1, "ratio": total0 / total1, **stats, "instance_bits": cutmod.instance_bits(cut_plan),
               "reps": p1.reps, "columns": p1.columns, "setup_s": setup_s,
               "setup_cache": coms.setup_cache if coms is not None else None, "v0_query_s": v0_s, "v1_bytes_s": v1_s}
        good = ref["accepted"] and all(v for k, v in accounting["v0"].items() if k != "paths_ratio")
        if args.full:
            t0 = time.perf_counter()
            full = cp.run_cut_query(prover, v1, x)
            out["v1_full"] = {"accepted": full["accepted"], "rejected_at": full["rejected_at"], "bytes": full["bytes"],
                              "timings": full["timings"], "s": time.perf_counter() - t0,
                              "transcript": full.get("transcript")}
            accounting["v1"] = v1_accounting(b1, full)
            good = good and full["accepted"] and all(accounting["v1"][k] for k in V1_EXACT)
            if args.reps > 1:                              # timed queries of both protocols, interleaved

                def v1_query(name, q):
                    with options.using(v1_verifier=name):
                        return cp.run_cut_query(prover, v1, q)
                for name in verifiers:                     # one warm-up each (the query above was main_verifier's)
                    if name != main_verifier:
                        good = good and v1_query(name, x)["accepted"]
                runs0, runs1 = [], {name: [] for name in verifiers}
                for i in range(args.reps):
                    q = (real_queries[i] if real_queries is not None else
                         torch.randint(0, cfg.vocab, (1, args.seq),
                                       generator=torch.Generator().manual_seed(args.timed_seed + i)))
                    runs0.append(run_query(prover, v0, q, wire=args.wire))
                    for name in verifiers:
                        runs1[name].append(v1_query(name, q))
                out["timed"] = {"queries": args.reps, "v0": _medians(runs0), "v1": _medians(runs1[main_verifier]),
                                "v1_paths": {name: _medians(rs) for name, rs in runs1.items()},
                                "v1_path": main_verifier}
                good = good and out["timed"]["v0"]["accepted"] == args.reps
                good = good and all(sum(r["accepted"] for r in rs) == args.reps for rs in runs1.values())
            if args.tampers:
                out["tampers"] = v1_tampers(prover, v1, x, args.tampers)
                good = good and all(t["rejected"] == t["attempts"] for t in out["tampers"].values())
        out["accounting"], out["accounting_ok"] = accounting, good and shapes_ok
        out["peak"] = records.peaks(args.device)
        ok = ok and good
        records.emit(out)
        prover = v0 = v1 = coms = None
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    if not ok:
        print("FAILED: a byte accounting, acceptance or tamper check (see accounting_ok)", file=sys.stderr, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

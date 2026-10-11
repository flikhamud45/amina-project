"""V1's proof size against today's (v0), on the same model, query and parameters.

    cd code && PYTHONPATH=src python experiments/8_v1/v1_bytes.py --model llama2-7b --layers 2 --seq 2048 \\
        --modes C Kpre --policy auto --prune-last --fs
    cd code && PYTHONPATH=src python experiments/8_v1/v1_bytes.py --model gpt2 --seq 64 --modes C --full
    cd code && PYTHONPATH=src python experiments/8_v1/v1_bytes.py --model opt-1.3b --real facebook/opt-1.3b \\
        --smooth 0.55 --calib-seq 2048 --prune-last --seq 2048 --modes Kpre --fs --lean --full --reps 3 --tampers 3
    cd code && PYTHONPATH=src python experiments/8_v1/v1_bytes.py --model opt-6.7b --real facebook/opt-6.7b \\
        --smooth 0.55 --prune-last --seq 2048 --codec-only --device cpu

Builds the decoder as ``experiments/6_improvements/perf.py`` and ``bench.py`` do (random weights, seed 0; the
parameters ``(r, t)`` of the full model), commits it under each ``--policy`` (mode C), runs one v0 query on the
wire (``run_query(wire=True)``) and measures V1's bytes on the same query with
``cut_protocol.cut_proof_bytes`` (M1 encoded for real, the other messages sized exactly; ``--full`` also runs the
whole V1 query, whose eager GKR suits small models only, and checks its bytes).  One JSON line per variant: both
byte breakdowns, their ratio, the cut's plan (leaves, instances, table, exceptions) and the git commit.

Every line also checks the byte accounting (T7; ``accounting``, ``accounting_ok``): v0's ``u`` and opened columns
are ``analytic.proof_bytes`` of the graph's weight ops exactly, with the multiproofs' ratio to their expectation
(``paths_ratio``; v0's claims travel in ``PVC3``, whose size depends on their values: the model takes the measured
size, so they are not a check); with ``--full``, V1's every message but the multiproofs is ``cut_proof_bytes``'s
(V1's claims included: both encode the same values); and the graph's weight-op shapes and claim columns are those of
``analytic.decoder_shapes`` / ``decoder_claim_columns`` (``shapes``; a real checkpoint's LM head has one more
column, its folded bias, which is expected).  A failed check is printed and makes the exit status 1.

``--real``: the checkpoint's weights (``real_weights.build_opt_from_hf``, calibrated on the CPU on one WikiText-2
window of ``--calib-seq`` tokens from the end of the test split, default ``--seq``, so the 64-token queries can use
the 2,048-token graph), queried on WikiText-2 windows from the start of the split.  ``--prune-last`` builds either
graph with its last block at the last position.  ``--tampers n`` (with ``--full``): per variant, n trace-tampering
queries (a cut op's claim moved by whole windows, the rest recomputed honestly) and n single-value ones (any
weight op, any entry, +-1..999), each of which must be rejected (else the exit status is 1).  ``--codec-only``: no
commitment and no proof -- one forward pass (``--device``) whose claims are encoded op by op as they are produced
(v0: ``PVC3`` of the claims; V1: of the cut ops' requantised values and the clear ops' claims, and the exceptions
frame), summed per block: the byte profile of a model whose prover the machine cannot hold.  Peak host memory
(VmHWM, Linux) and GPU memory are recorded per variant (``peak``).
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import torch

CODE = Path.cwd()
SMOOTH = dict(smooth_pct=99.9, pct=99.99, pct_att=99.99)
"""The paper's integer OPT (F2) with ``--smooth`` alpha: SmoothQuant through the norm gains, 99.99th-percentile
clipping; ``norm_gain`` 4 either way."""


# ------------------------------------------------------------------ real checkpoints
def load_real(name: str, seq: int, n_queries: int, calib_seq: int = 0):
    """``(model, queries, calib)``: the fp32 checkpoint ``name`` (``9_quality/common.load_opt``), ``n_queries``
    WikiText-2 windows of ``seq`` tokens from the start of the test split and one calibration window of
    ``calib_seq`` tokens (default ``seq``) from its end (``common.load_windows``).  The second call asks for as many
    ``calib_seq`` windows as cover the queries, so its own check that they and the calibration window fit the split
    keeps the calibration window disjoint from every query."""
    sys.path.insert(0, str(CODE / "experiments" / "9_quality"))
    from common import load_opt, load_windows
    model, tok = load_opt(name)
    cs = calib_seq or seq
    queries, _ = load_windows(tok, seq=seq, windows=n_queries, calib_windows=0)
    _, calib = load_windows(tok, seq=cs, windows=-(-n_queries * seq // cs), calib_windows=1)
    return model, queries, calib[0]


def real_graph(model, calib: torch.Tensor, smooth: float | None, prune_last: bool):
    """The integer OPT of ``model``, calibrated on ``calib`` on the CPU: ``build_opt_from_hf`` with norm gain 4,
    and with ``smooth`` the paper's smoothing (:data:`SMOOTH`)."""
    from pvi.fullcheck.real_weights import build_opt_from_hf
    kw = dict(norm_gain=4) if smooth is None else dict(norm_gain=4, smooth=smooth, **SMOOTH)
    return build_opt_from_hf(model, calib, prune_last=prune_last, **kw)


# ------------------------------------------------------------------ peaks
def reset_peaks(device) -> None:
    """A new peak window: the GPU's allocator statistics and the host's VmHWM (Linux)."""
    if torch.device(device).type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError:
        pass


def peaks(device) -> dict:
    """Bytes since :func:`reset_peaks`, named as ``bench.py`` records them: the host's peak resident set (``None``
    where ``/proc`` has none), and on a GPU its peak allocated and reserved memory."""
    out = {"host_peak_rss": None}
    try:
        with open("/proc/self/status") as fh:
            out["host_peak_rss"] = 1024 * next(int(line.split()[1]) for line in fh if line.startswith("VmHWM:"))
    except (OSError, StopIteration):
        pass
    if torch.device(device).type == "cuda":
        out["gpu_peak_memory"] = torch.cuda.max_memory_allocated(device)
        out["gpu_peak_reserved"] = torch.cuda.max_memory_reserved(device)
    return out


# ------------------------------------------------------------------ the accounting (T7)
def shape_check(graph, cfg, n_layers: int, prune_last: bool, real: bool, x: torch.Tensor) -> dict:
    """``graph``'s weight ops against ``analytic.decoder_shapes(cfg, n_layers, prune_last)`` (rows and row length,
    in order; a real checkpoint's LM head has one more column, the final norm's folded bias) and its claim columns
    on ``x`` against ``decoder_claim_columns``."""
    from pvi.fullcheck.analytic import decoder_claim_columns, decoder_shapes
    shapes = decoder_shapes(cfg, n_layers, prune_last)
    mats = graph.mat_ops
    if len(shapes) != len(mats):
        return {"problems": [f"{len(mats)} weight ops, the model {len(shapes)}"], "claim_columns": False}
    problems = []
    for s, op in zip(shapes, mats):
        want = (s.n_rows, s.row_length + (1 if real and s.name == "head" else 0))
        if (op.n_rows, op.row_length) != want:
            problems.append(f"{op.name} ({s.name}): {(op.n_rows, op.row_length)}, the model {want}")
    cols = graph.claim_columns(x)
    return {"problems": problems,
            "claim_columns": cols == list(decoder_claim_columns(cfg, x.shape[1], n_layers, prune_last).values())}


V0_EXACT = ("u", "columns")


def v0_accounting(graph, ref: dict, params, plan, mode: str, x: torch.Tensor, table_ids: dict) -> dict:
    """``run_query(wire=True)``'s bytes ``ref`` against ``analytic.proof_bytes`` of ``graph``'s weight ops on ``x``:
    ``u`` and the opened columns exactly (:data:`V0_EXACT`); the multiproofs' ratio to their expectation (exact for
    the ids ``table_ids`` a plan's lookup tables looked up).  The claims are not compared: on the wire the model
    takes their measured ``PVC3`` size (``wire_claims``), so the comparison would hold by construction
    (``tests/test_real_weights.py`` checks their unencoded size, which the shapes fix)."""
    from pvi.fullcheck.analytic import proof_bytes
    cols = dict(zip((op.name for op in graph.mat_ops), graph.claim_columns(x)))
    want = proof_bytes(graph.mat_ops, params, cols, plan=plan, mode=mode, wire_claims=ref["bytes"]["claims"],
                       table_ids=table_ids)
    out = {k: ref["bytes"][k] == want[k] for k in V0_EXACT}
    out["paths_ratio"] = ref["bytes"]["paths"] / want["paths"] if want["paths"] else None
    return out


V1_EXACT = ("claims", "cut_exc", "cut_mult", "cut_gkr", "u", "columns")


def v1_accounting(est: dict, full: dict) -> dict:
    """``cut_proof_bytes``'s ``est`` against a real V1 query's bytes (``tests/test_cut_protocol.py`` T7): every message
    exactly but the multiproofs (other columns), whose ratio is reported."""
    out = {k: est[k] == full["bytes"][k] for k in V1_EXACT}
    out["paths_ratio"] = full["bytes"]["paths"] / est["paths"] if est["paths"] else None
    return out


# ------------------------------------------------------------------ tampering (V1)
def v1_tampers(prover, verifier, x: torch.Tensor, n: int, seed: int = 0) -> dict:
    """``n`` trace-tampering V1 queries -- a random entry of a random cut op's claim moved by 1-3 widths of its normal
    window (another requantised value), everything after it recomputed honestly -- and ``n`` single-value ones (a
    random entry of a random weight op, +-1..999): ``{kind: {attempts, rejected, labels}}``."""
    from pvi.fullcheck import cut_protocol as cp
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
    the ops cut under the plan of one tree per op or Kpre -- ``CutPlan.from_graph`` without col matrices, P4 from the
    graph's weights -- and the other ops' claims) and the cut ops' exceptions; the totals add V1's exceptions frame
    (``PVX1``).  Each op is encoded on its own (its own header, Rice parameters and exception list), so the sums
    differ slightly from one encoding of all the claims, which a protocol line (without ``--codec-only``) gives
    exactly."""
    from pvi.fullcheck import claimcodec
    from pvi.fullcheck import cut as cutmod
    from pvi.fullcheck.cut_protocol import pack_exceptions
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: full model)")
    ap.add_argument("--modes", nargs="+", default=["C", "Kpre"])
    ap.add_argument("--policy", nargs="+", default=["auto"])
    ap.add_argument("--lam", type=int, default=128)
    ap.add_argument("--fs", action="store_true", help="Fiat--Shamir parameters (grinding bits added)")
    ap.add_argument("--prune-last", action="store_true")
    ap.add_argument("--lmax", type=int, default=27)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--lean", action="store_true")
    ap.add_argument("--full", action="store_true", help="also run the whole V1 query (the GKR on the GPU with Triton)")
    ap.add_argument("--reps", type=int, default=1, help="with --full: timed queries of each protocol after a warm-up")
    ap.add_argument("--verifier-device", default="cpu")
    ap.add_argument("--label", default="")
    ap.add_argument("--real", default="", help="a Hugging Face OPT checkpoint (e.g. facebook/opt-125m): its real "
                    "weights (real_weights.build_opt_from_hf), calibrated and queried on WikiText-2 windows")
    ap.add_argument("--smooth", type=float, default=None, help="with --real: SmoothQuant strength through the norm gains "
                    "(F2; 0.55 with smooth_pct 99.9, pct 99.99, pct_att 99.99)")
    ap.add_argument("--calib-seq", type=int, default=0, help="with --real: the calibration window's tokens (0: --seq)")
    ap.add_argument("--tampers", type=int, default=0, help="with --full: trace-tampering and single-value V1 queries "
                    "per variant, each of which must be rejected")
    ap.add_argument("--codec-only", action="store_true", help="no commitment and no proof: the claims of one forward "
                    "pass encoded op by op (v0 and V1), per block")
    args = ap.parse_args()
    if args.tampers and not args.full:
        raise SystemExit("--tampers runs V1 queries: it goes with --full")
    torch.set_num_threads(args.threads)
    from pvi.fullcheck import cut_protocol as cp
    from pvi.fullcheck.analytic import decoder_shapes
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    cfg = CONFIGS[args.model]
    n_layers = args.layers or cfg.n_layers
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=CODE).stdout.strip()
    reset_peaks(args.device)
    t0 = time.perf_counter()
    if args.real:                     # the checkpoint's weights; queries are WikiText-2 windows
        if args.layers:
            raise SystemExit("--real builds every block of the checkpoint")
        model, wins, calib = load_real(args.real, args.seq, max(2, args.reps + 1), args.calib_seq)
        graph, _ = real_graph(model, calib, args.smooth, args.prune_last)
        del model
        n_layers = len([op for op in graph.ops if getattr(op, "note", "") == "attention"])
        x = wins[0]
        real_queries = wins[1:]
    else:
        graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(args.seq, 32), seed=0,
                              prune_last=args.prune_last)
        x = torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(1))
        real_queries = None
    build_s = time.perf_counter() - t0
    head = {"label": args.label, "git": sha, "model": args.model, "real": args.real or None, "smooth": args.smooth,
            "calib_seq": (args.calib_seq or args.seq) if args.real else None, "layers": n_layers, "seq": args.seq,
            "prune_last": args.prune_last, "lmax": args.lmax, "graph_digest": graph.digest().hex(),
            "shapes": shape_check(graph, cfg, n_layers, args.prune_last, bool(args.real), x), "build_s": build_s,
            "build_peak": peaks(args.device), "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    shapes_ok = not head["shapes"]["problems"] and head["shapes"]["claim_columns"]
    if args.codec_only:
        reset_peaks(args.device)
        out = {**head, "codec_only": True, "device": args.device,
               **codec_profile(graph, x, n_layers, args.lmax, args.device), "peak": peaks(args.device)}
        print(json.dumps(out), flush=True)
        return 0 if shapes_ok else 1
    ok = shapes_ok
    # a real checkpoint's graph (e.g. OPT's LM head has a bias) is planned from its own ops
    model_ops = None if args.real else decoder_shapes(cfg, prune_last=args.prune_last)
    n_checks = len(model_ops) if model_ops is not None else len(graph.mat_ops)
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    variants = [("C", pol) for pol in args.policy if "C" in args.modes] + ([("Kpre", None)] if "Kpre" in args.modes else [])
    for mode, policy in variants:
        reset_peaks(args.device)
        t0 = time.perf_counter()
        if mode == "C":
            coms = commit_graph(graph, 4, device=args.device, policy=policy, model_ops=model_ops)
            plan = coms.plan
            kw = dict(publics=coms.publics, groups=coms.group_publics, tables=coms.table_publics)
            prover = Prover(graph, device=args.device, commitments=coms, lean=args.lean)
        else:
            plan, kw = None, dict(weights=weights)
            prover = Prover(graph, device=args.device, lean=args.lean)
        setup_s = time.perf_counter() - t0
        p0 = params_for(args.lam, n_checks, fiat_shamir=args.fs, plan=plan)
        p1 = params_for(args.lam, n_checks, fiat_shamir=args.fs, plan=plan, cut=True, cut_lmax=args.lmax)
        v0 = Verifier(graph.public(), p0, mode, lean=args.lean, device=args.verifier_device, **kw)
        v1 = Verifier(graph.public(), p1, mode, lean=args.lean, device=args.verifier_device, **kw)
        if mode == "Kpre":
            v0.precompute(Challenger())
            v1._pre = v0._pre                        # the same secret rows (only F2 reads them in V1)
        t0 = time.perf_counter()
        ref = run_query(prover, v0, x, wire=True)
        v0_s = time.perf_counter() - t0
        # the ids the plan's lookup tables looked up on this query (the prover keeps them for its openings)
        accounting = {"v0": v0_accounting(graph, ref, p0, plan, mode, x, dict(prover._looked_up))}
        t0 = time.perf_counter()
        b1 = cp.cut_proof_bytes(prover, v1, x, seed=7)
        v1_s = time.perf_counter() - t0
        stats = {k: b1.pop(k) for k in ("n_leaves", "n_instances", "table", "n_exceptions")}
        total0, total1 = sum(ref["bytes"].values()), sum(b1.values())
        out = {**head, "mode": mode, "policy": policy, "chosen": None if plan is None else plan.policy, "fs": args.fs,
               "v0_accepted": ref["accepted"], "v0_bytes": ref["bytes"], "v1_bytes": b1, "v0_total": total0,
               "v1_total": total1, "ratio": total0 / total1, **stats, "reps": p1.reps, "columns": p1.columns,
               "setup_s": setup_s, "v0_query_s": v0_s, "v1_bytes_s": v1_s}
        good = ref["accepted"] and all(accounting["v0"][k] for k in V0_EXACT)
        if args.full:
            t0 = time.perf_counter()
            full = cp.run_cut_query(prover, v1, x)
            out["v1_full"] = {"accepted": full["accepted"], "rejected_at": full["rejected_at"], "bytes": full["bytes"],
                              "timings": full["timings"], "s": time.perf_counter() - t0}
            accounting["v1"] = v1_accounting(b1, full)
            good = good and full["accepted"] and all(accounting["v1"][k] for k in V1_EXACT)
            if args.reps > 1:                              # medians of timed queries of both protocols, interleaved
                import statistics
                runs0, runs1 = [], []
                for i in range(args.reps):
                    q = (real_queries[i] if real_queries is not None else
                         torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(100 + i)))
                    runs0.append(run_query(prover, v0, q, wire=True))
                    runs1.append(cp.run_cut_query(prover, v1, q))

                def med(runs):
                    keys = sorted({k for r in runs for k in r["timings"]})
                    t = {k: statistics.median(r["timings"].get(k, 0.0) for r in runs) for k in keys}
                    return {"accepted": sum(r["accepted"] for r in runs), "timings": t,
                            "prove_s": sum(v for k, v in t.items() if k.startswith("prove_")),
                            "verify_s": sum(v for k, v in t.items() if k.startswith("verify_")),
                            "fs_s": t.get("fs_hash", 0.0)}
                out["timed"] = {"queries": args.reps, "v0": med(runs0), "v1": med(runs1)}
                good = good and out["timed"]["v0"]["accepted"] == out["timed"]["v1"]["accepted"] == args.reps
            if args.tampers:
                out["tampers"] = v1_tampers(prover, v1, x, args.tampers)
                good = good and all(t["rejected"] == t["attempts"] for t in out["tampers"].values())
        out["accounting"], out["accounting_ok"] = accounting, good and shapes_ok
        out["peak"] = peaks(args.device)
        ok = ok and good
        print(json.dumps(out), flush=True)
        prover = v0 = v1 = coms = None
        if torch.device(args.device).type == "cuda":
            torch.cuda.empty_cache()
    if not ok:
        print("FAILED: a byte accounting, acceptance or tamper check (see accounting_ok)", file=sys.stderr, flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

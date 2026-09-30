"""GPU A/B timing of the verifier: a baseline checkout (the paper code) against this one, in the
same process, on ``--verifier-device``.

    cd code && export PYTHONPATH=$PWD/src
    python experiments/6_improvements/gpu_ab.py --base ../../base/code/src --model llama2-7b --seq 2048 \\
        --layers 1,2 --modes Kpre,C --reps 5 --out $SCRATCH/gpu_ab_llama2048.json

For one decoder (random int8 weights, the real shapes), prompt length and block count, both
packages build the same graph (the same seeds) and both provers run on ``--device``: they must
send the same claims.  Then, per mode (Kpre with a fixed secret ``chi``; C with the committed
weights and the openings), ``--reps`` times each, alternating which goes first:

  base    the baseline verifier as ``run_query`` drives it: upload (int64 claims from pageable
          memory, and ``u`` in mode C), derive, check_products, check_columns (mode C)
  new     this checkout's verifier, the same calls (int32 attention, device-resident constants,
          deferred verdicts, int8 products, batched device column checks)
  stream  this checkout's streaming verifier (``Verifier.verify_streaming``) on the int32 wire
          claims in pinned memory: from the first upload to the verdict

Before timing, base and new must derive identical tensors (compared on the device), and one
tampered transcript per kind (a claim +1, a claim at ``Z_BOUND``; in mode C ``u`` +1, an opened
column +1, a forged Merkle path) must be rejected by all three at the same check.  Peak device
memory is recorded per implementation.  ``--compile`` also times this checkout's derive with the
int32 attention core under ``torch.compile`` (``--compile-all``: every cheap op as well) and
checks it bit-exact; ``--attn-budgets`` sweeps ``transformer.ATTN_BYTES``.  ``--impls stream``
times the streaming verifier alone (models whose other verifiers do not fit the GPU).  With
block counts 1 and 2 every stage is also extrapolated to the full model: ``t1 + (L - 1)(t2 - t1)``.

Writes JSON to ``--out`` (never into the stored benchmark roots).
"""

from __future__ import annotations

import argparse
import gc
import importlib
import json
import statistics
import sys
import time
from pathlib import Path

import torch

import pvi.fullcheck as N
from pvi.fullcheck import pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ab_verifier import Case, build, load_base, sub  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
IMPLS = ("base", "new", "stream")


def sync(dev) -> None:
    if torch.device(dev).type == "cuda":
        torch.cuda.synchronize(dev)


def timed(f, dev):
    sync(dev)
    t0 = time.perf_counter()
    out = f()
    sync(dev)
    return time.perf_counter() - t0, out


def reset_peak(dev) -> None:
    gc.collect()
    if torch.device(dev).type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(dev)


def peak(dev) -> int:
    return torch.cuda.max_memory_allocated(dev) if torch.device(dev).type == "cuda" else 0


# -- one query, as each implementation runs it --------------------------------------------------

def run_default(v, tr: dict, mode: str, dev, keep: bool = False):
    """``(timings, label, derived inputs if keep)`` with ``run_query``'s order of checks."""
    t = {}
    t["upload"], (x, claims, us) = timed(lambda: (tr["x"].to(dev), {k: z.to(dev) for k, z in tr["claims"].items()},
                                                  {k: u.to(dev) for k, u in tr["us"].items()}), dev)
    t["derive"], inputs = timed(lambda: v.derive(x, claims), dev)
    label = "range_or_shape" if inputs is None else None
    if inputs is not None:
        t["products"], ok = timed(lambda: v.check_products(claims, inputs, tr["chis"], us), dev)
        if not ok:
            label = "freivalds"
        elif mode == "C":
            t["columns"], label = timed(lambda: v.check_columns(tr["chis"], us, tr["cols"], tr["openings"]), dev)
    t["total"] = sum(t.values())
    return t, label, inputs if keep else None


def run_stream(v, tr: dict, mode: str, dev, keep: bool = False):
    total, label = timed(lambda: v.verify_streaming(tr["x"], tr["wire"], tr["chis"], tr["us"],
                                                    tr.get("cols"), tr.get("wire_openings")), dev)
    return {"total": total}, label, None


RUN = {"base": run_default, "new": run_default, "stream": run_stream}


# -- transcripts ------------------------------------------------------------------------------

def make_transcript(base, case: Case, mode: str, device, vdev, impls) -> tuple[dict, dict]:
    """The messages of one query (this checkout's prover on ``device``) and the verifiers."""
    pin = torch.device(vdev).type == "cuda"
    graph, x, n_ops = build(N, case)
    params = N.params_for(case.lam, n_ops)
    coms = N.commit_graph(graph, params.rate, device=device) if mode == "C" else {}
    prover = N.Prover(graph, device=device, commitments=coms, lean=True)
    tr: dict = {"x": x, "claims_identical": None}
    if impls != ["stream"]:
        tr["claims"] = prover.claims(x)
    if "stream" in impls:
        tr["wire"] = prover.claims(x, send=lambda z: pipeline.wire_claim(z, pin=pin))
    if "base" in impls:           # the baseline prover must send the same claims
        graph_b, x_b, _ = build(base, case)
        claims_b = base.Prover(graph_b, device=device, lean=True).claims(x_b)
        tr["claims_identical"] = claims_b.keys() == tr["claims"].keys() and all(
            torch.equal(claims_b[k], tr["claims"][k]) for k in claims_b)
        del claims_b
    publics = {k: c.public for k, c in coms.items()}
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    verifiers = {}
    if impls != ["base"]:
        verifiers["new"] = verifiers["stream"] = N.Verifier(graph.public(), params, mode, publics=publics,
                                                          weights=weights, lean=True, device=vdev)
    if "base" in impls:
        verifiers["base"] = base.Verifier(graph_b.public(), base.params_for(case.lam, n_ops), mode,
                                          publics=publics, weights=weights, lean=True, device=vdev)
    ch = N.Challenger(seed=3)
    mats = graph.mat_ops
    if mode == "Kpre":
        some = next(iter(verifiers.values()))
        some.precompute(N.Challenger(seed=0))
        for v in verifiers.values():
            v._pre = some._pre
        tr["chis"] = {op.name: some._pre[op.name][0] for op in mats}
        tr["us"] = {op.name: some._pre[op.name][1] for op in mats}
    else:
        tr["chis"] = {op.name: ch.folding(op.name, op.n_rows, params.reps).to(vdev) for op in mats}
        tr["us"] = prover.fold(tr["chis"])
        tr["cols"] = {op.name: ch.columns(op.name, publics[op.name].n_points, params.columns) for op in mats}
        tr["openings"] = prover.open(tr["cols"])
        tr["wire_openings"] = pipeline.wire_openings(tr["openings"])
    del prover, coms
    return tr, verifiers


def tampered(tr: dict, mode: str, pin: bool):
    """``(kind, transcript)`` pairs that every implementation must reject at the same check."""
    names = list(tr["chis"])
    mid = names[len(names) // 2]
    P, Z = N.P, sub(N, "protocol").Z_BOUND

    def claim(value):
        out = dict(tr)
        if "claims" in tr:
            z = tr["claims"][mid].clone()
            z.view(-1)[0] = value(int(z.view(-1)[0]))
            out["claims"] = dict(tr["claims"], **{mid: z})
        if "wire" in tr:
            z = tr["wire"][mid].to(torch.int64)
            z.view(-1)[0] = value(int(z.view(-1)[0]))
            out["wire"] = dict(tr["wire"], **{mid: pipeline.wire_claim(z, pin=pin)})
        return out

    yield "claim+1", claim(lambda v: v + 1)
    yield "claim=Z_BOUND", claim(lambda v: Z)
    if mode == "C":
        u = tr["us"][mid].clone()
        u[0, 0] = (u[0, 0] + 1) % P
        yield "u+1", dict(tr, us=dict(tr["us"], **{mid: u}))
        o, proof = tr["openings"][mid]
        o1 = o.clone()
        o1[0, 0] = (o1[0, 0] + 1) % P
        for kind, opened in (("opened+1", (o1, proof)), ("path forged", (o, [bytes(32)] + list(proof)[1:]))):
            openings = dict(tr["openings"], **{mid: opened})
            yield kind, dict(tr, openings=openings, wire_openings=pipeline.wire_openings(openings))


# -- one case ---------------------------------------------------------------------------------

def run_case(base, a, layers: int, mode: str) -> dict:
    case = Case(a.model, mode, a.lam, a.seq, layers, a.vocab)
    vdev = a.verifier_device
    impls = a.impls
    t0 = time.perf_counter()
    tr, vs = make_transcript(base, case, mode, a.device, vdev, impls)
    rec = {"model": a.model, "seq": a.seq, "layers": layers, "mode": mode, "lam": a.lam, "reps": a.reps,
           "impls": impls, "device": a.device, "verifier_device": vdev, "torch": torch.__version__,
           "cuda": torch.version.cuda, "threads": torch.get_num_threads(),
           "gpu": torch.cuda.get_device_name(torch.device(vdev)) if torch.device(vdev).type == "cuda" else None,
           "int8_gemm": sub(N, "field").int8_ok(vdev),
           "claims": sum(z.numel() for z in (tr.get("claims") or tr["wire"]).values()),
           "claims_identical": tr["claims_identical"], "setup_s": time.perf_counter() - t0}
    pin = torch.device(vdev).type == "cuda"
    # bit-exactness and verdicts, before any timing
    derived = {}
    for impl in impls:
        reset_peak(vdev)
        _, label, derived[impl] = RUN[impl](vs[impl], tr, mode, vdev, keep=impl != "stream")
        assert label is None, f"{impl} rejects the honest query at {label}"
    if "base" in derived and "new" in derived:
        rec["derive_bit_exact"] = derived["base"].keys() == derived["new"].keys() and all(
            torch.equal(derived["base"][k], derived["new"][k]) for k in derived["base"])
        assert rec["derive_bit_exact"], "derive differs"
    del derived
    rec["labels"] = {kind: [RUN[impl](vs[impl], bad, mode, vdev)[1] for impl in impls]
                     for kind, bad in tampered(tr, mode, pin)}
    for kind, got in rec["labels"].items():
        assert len(set(got)) == 1 and got[0] is not None, f"{kind}: {got}"
    # timing
    times = {impl: [] for impl in impls}
    peaks = {impl: 0 for impl in impls}
    for i in range(a.reps):
        for impl in impls if i % 2 == 0 else impls[::-1]:
            reset_peak(vdev)
            t, label, _ = RUN[impl](vs[impl], tr, mode, vdev)
            assert label is None
            times[impl].append(t)
            peaks[impl] = max(peaks[impl], peak(vdev))
    for impl in impls:
        rec[impl] = {stage: statistics.median(t[stage] for t in times[impl]) for stage in times[impl][0]}
    rec["peak_bytes"] = peaks
    if "base" in impls:
        for impl in ("new", "stream"):
            if impl in impls:
                rec[f"ratio_{impl}"] = {stage: rec["base"][stage] / s for stage, s in rec[impl].items()}
    if "new" in impls and "stream" in impls:
        rec["ratio_stream_vs_new_total"] = rec["new"]["total"] / rec["stream"]["total"]
    if "new" in impls and (a.attn_budgets or a.compile):
        rec.update(compile_and_budgets(vs["new"], tr, vdev, a))
    return rec


def compile_and_budgets(v, tr: dict, vdev, a) -> dict:
    """This checkout's derive with other attention budgets, and under torch.compile."""
    out = {}
    tr_mod = sub(N, "transformer")
    x = tr["x"].to(vdev)
    claims = {k: z.to(vdev) for k, z in tr["claims"].items()}
    want = v.derive(x, claims)

    def derive_times():
        ts, got = [], None
        for _ in range(a.reps):
            t, got = timed(lambda: v.derive(x, claims), vdev)
            ts.append(t)
        exact = got is not None and got.keys() == want.keys() and all(torch.equal(got[k], want[k]) for k in want)
        return statistics.median(ts), exact

    saved = tr_mod.ATTN_BYTES
    try:
        for budget in a.attn_budgets:
            tr_mod.ATTN_BYTES = budget
            out[f"derive_attn_{budget >> 20}MiB"], out[f"derive_attn_{budget >> 20}MiB_exact"] = derive_times()
    finally:
        tr_mod.ATTN_BYTES = saved
    if a.compile:
        importlib.import_module("torch._dynamo").config.cache_size_limit = 1024
        core = tr_mod._attention_core
        fns = [(op, op.fn) for op in v.graph.ops if isinstance(op, sub(N, "graph").CheapOp)]
        try:
            tr_mod._attention_core = torch.compile(core, dynamic=False)
            out["compile_attention_s"], _ = timed(lambda: v.derive(x, claims), vdev)
            out["derive_compiled_attention"], out["compiled_attention_exact"] = derive_times()
            if a.compile_all:
                for op, fn in fns:
                    op.fn = torch.compile(fn, dynamic=False)
                out["compile_all_s"], _ = timed(lambda: v.derive(x, claims), vdev)
                out["derive_compiled_all"], out["compiled_all_exact"] = derive_times()
        except Exception as exc:          # e.g. no C++ compiler or no Triton for this GPU
            out["compile_error"] = repr(exc)[:800]
        finally:
            tr_mod._attention_core = core
            for op, fn in fns:
                op.fn = fn
    return out


def extrapolate(rows: list[dict], full_layers: int) -> list[dict]:
    """Stage medians of the full model from the 1- and 2-block builds: ``t1 + (L - 1)(t2 - t1)``."""
    out = []
    for mode in sorted({r["mode"] for r in rows}):
        by = {r["layers"]: r for r in rows if r["mode"] == mode}
        if 1 not in by or 2 not in by:
            continue
        rec = {"model": by[1]["model"], "seq": by[1]["seq"], "mode": mode, "extrapolated_layers": full_layers}
        for impl in IMPLS:
            if impl in by[1]:
                rec[impl] = {s: by[1][impl][s] + (full_layers - 1) * (by[2][impl][s] - by[1][impl][s])
                             for s in by[1][impl]}
        for impl in ("new", "stream"):
            if "base" in rec and impl in rec:
                rec[f"ratio_{impl}"] = {s: rec["base"][s] / t for s, t in rec[impl].items()}
        out.append(rec)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--base", type=Path, required=True, help="the baseline checkout's code/src")
    ap.add_argument("--model", default="llama2-7b")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--layers", default="1,2", help="block counts, e.g. 1,2 or 12 (or 'full')")
    ap.add_argument("--vocab", type=int, default=None, help="override the vocabulary (the blocks are unchanged)")
    ap.add_argument("--modes", default="Kpre,C")
    ap.add_argument("--lam", type=int, default=128)
    ap.add_argument("--impls", default=",".join(IMPLS), help="any of base,new,stream")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--device", default="cuda", help="the provers' device")
    ap.add_argument("--verifier-device", default="cuda")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--compile", action="store_true", help="also time derive with the attention core compiled")
    ap.add_argument("--compile-all", action="store_true", help="with --compile: every cheap op compiled too")
    ap.add_argument("--attn-budgets", default="", help="comma-separated ATTN_BYTES values to sweep")
    ap.add_argument("--out", type=Path, required=True, help="JSON file to write")
    a = ap.parse_args()
    a.impls = [i for i in IMPLS if i in a.impls.split(",")]
    a.attn_budgets = [int(b) for b in a.attn_budgets.split(",") if b]
    out = a.out.resolve()
    stored = (ROOT / "artifacts" / "comparison").resolve()
    if out.is_relative_to(stored) and any(p.name.startswith("raw") for p in out.relative_to(stored).parents):
        raise SystemExit(f"{out}: the stored benchmark roots are frozen; write somewhere else")
    torch.set_num_threads(a.threads)
    torch.backends.cuda.matmul.allow_tf32 = False
    base = load_base(a.base.resolve())
    assert Path(base.__file__).resolve() != Path(N.__file__).resolve(), "--base is this checkout"
    cfg = sub(N, "transformer").CONFIGS[a.model]
    counts = [cfg.n_layers if n == "full" else int(n) for n in a.layers.split(",")]
    rows = []
    for layers in counts:
        for mode in a.modes.split(","):
            rec = run_case(base, a, layers, mode)
            print(json.dumps(rec), flush=True)
            rows.append(rec)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    rows += extrapolate(rows, cfg.n_layers)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rows, indent=1))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()

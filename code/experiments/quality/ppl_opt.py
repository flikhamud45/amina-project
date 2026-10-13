"""Plan 2, F2a: the certified integer OPT's quality at 2,048-token context on the full WikiText-2 test set.

    cd code && PYTHONPATH=src python experiments/9_quality/ppl_2048.py --model facebook/opt-1.3b --device cuda \\
        --graphs scalar smooth rows --pbits 8 11 --float-softmax --draws 0 1 2 --out results.jsonl
    cd code && PYTHONPATH=src python experiments/9_quality/ppl_2048.py --model facebook/opt-125m --check
    cd code && PYTHONPATH=src python experiments/9_quality/ppl_2048.py --model facebook/opt-1.3b --legacy

* Text: WikiText-2 (raw) joined with blank lines (``"\\n\\n".join``, the convention of the published 2,048-token
  OPT perplexities: 14.62 for OPT-1.3B, 10.86 for OPT-6.7B), tokenised once.
* Windows: every non-overlapping ``--seq``-token window of the test split (``--windows`` limits them), each scored on
  its own, all positions in one pass (as one query per prefix would give them).
* Calibration: draw ``d`` is ``--calib-windows`` consecutive ``--seq``-token windows of the *train* split starting at
  window ``d * --calib-windows``; no evaluated token is seen.
* Graphs (``opt_builder.build_opt``, the paper builder's arithmetic for 8-bit probabilities):
  ``scalar`` (one norm gain, ``--gain``; the best of the earlier sweep is 4 / 3 / 2 for OPT-125M / 1.3B / 6.7B),
  ``smooth`` (SmoothQuant through the norm gains, alpha ``--alphas``, smooth_pct 99.9, 99.99th-percentile clipping:
  the paper's model) and ``rows`` (``smooth`` plus per-row power-of-two weight scales, 4 buckets: v0 only).
  Each at every ``--pbits`` (8: today's probabilities; 11: ``P_MAX = 2047``, its own calibrated output multiplier).
* ``--float-softmax`` adds, per 8-bit graph, the same graph with an ideal softmax over the same integer scores
  (``255 softmax(-gap m_s / 2^30 / 16)`` in float, no exp table, no rounding of the probabilities).
* ``--mass-windows`` windows record, per layer, the mean share of probability mass a rounded row keeps.
* Lean: the integer forward frees each tensor after its last reader and keeps no claim; the LM head's logits are
  reduced to NLL and argmax in chunks.
* Output: one JSON line per cell (stdout, and appended to ``--out``): perplexity, ratio to fp32 with a 95% bootstrap
  interval over windows, top-1 agreement with fp32, per-layer mass, per-window NLL (to pool calibration draws),
  the graph digest, seconds, peak host and GPU memory.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import platform
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import opt_builder as ob                                   # noqa: E402
from common import find_parquet, load_opt, load_windows   # noqa: E402

from pvi.models import hf_opt as rw              # noqa: E402
from pvi.core.graph import CheapOp, IntGraph, OpFn, identity_op, op_fn, op_kind   # noqa: E402

BEST_GAIN = {"opt-125m": 4.0, "opt-1.3b": 3.0, "opt-6.7b": 2.0}
SMOOTH = dict(norm_mode="smooth", norm_gain=4.0, smooth_pct=99.9, pct=99.99, pct_att=99.99)


# ------------------------------------------------------------------ data

def _text(split: str) -> str:
    import pyarrow.parquet as pq
    if split == "test":
        path = find_parquet(None)
    else:
        from huggingface_hub import hf_hub_download
        path = hf_hub_download(repo_id="Salesforce/wikitext", filename="wikitext-2-raw-v1/train-00000-of-00001.parquet",
                               repo_type="dataset")
    return "\n\n".join(pq.read_table(path).column("text").to_pylist())


def _windows(ids: torch.Tensor, seq: int, n: int | None = None, start: int = 0) -> list[torch.Tensor]:
    total = ids.numel() // seq - start
    n = total if not n else min(n, total)
    return [ids[(start + i) * seq:(start + i + 1) * seq].unsqueeze(0) for i in range(n)]


# ------------------------------------------------------------------ evaluation

def _all_positions(graph: IntGraph) -> IntGraph:
    ops = [CheapOp(o.name, o.inputs, o.output, fn=op_fn(identity_op), note=o.note, params=o.params)
           if isinstance(o, CheapOp) and o.note == "last position" else o for o in graph.ops]
    return IntGraph(ops, graph.input_name, graph.output_name, dict(graph.meta))


def _nll_argmax(logits: torch.Tensor, scale: float, ids: torch.Tensor, chunk: int = 256):
    """Summed NLL and the argmaxes of ``logits [1, T, V]`` (integers times ``scale``) against ``ids [1, T]``."""
    lg = logits.reshape(-1, logits.shape[-1])[:-1]
    tgt = ids[0, 1:].to(lg.device)
    nll, am = 0.0, []
    for a in range(0, lg.shape[0], chunk):
        part = lg[a:a + chunk].double() * scale
        nll += float(F.cross_entropy(part, tgt[a:a + chunk], reduction="sum"))
        am.append(part.argmax(-1).cpu())
    return nll, torch.cat(am)


def _int_eval(graph: IntGraph, scale: float, w: torch.Tensor, device: str):
    env, _ = graph.forward(w.to(device), free=True, send=lambda z: None)
    out = _nll_argmax(env.pop(graph.output_name), scale, w)
    del env
    return out


def _float_softmax_attention(q, k, v, m_s, m_o, n_heads, n_kv, dh):
    """``transformer._attention`` with ``255 softmax(.)`` in place of the 8-bit probabilities: the scores are the same
    integers (exact in float64), the logits in nats ``-gap m_s / 2^30 / 16`` as the exp table reads them."""
    b, tq, _ = q.shape
    t = k.shape[1]
    rep = n_heads // n_kv
    qh = q.reshape(b, tq, n_heads, dh).transpose(1, 2)
    kh = k.reshape(b, t, n_kv, dh).transpose(1, 2).repeat_interleave(rep, 1)
    vh = v.reshape(b, t, n_kv, dh).transpose(1, 2).repeat_interleave(rep, 1)
    allowed = torch.ones(tq, t, dtype=torch.bool, device=q.device).tril(diagonal=t - tq)
    out = torch.empty(b, n_heads, tq, dh, dtype=torch.int64, device=q.device)
    for h0 in range(0, n_heads, ob.HEADS_PER_CHUNK):
        h1 = min(n_heads, h0 + ob.HEADS_PER_CHUNK)
        s = torch.matmul(qh[:, h0:h1].double(), kh[:, h0:h1].double().transpose(-1, -2)) * (m_s / float(1 << 30) / 16)
        s = s.masked_fill_(~allowed, float("-inf"))
        raw = torch.matmul(torch.softmax(s, -1).mul_(255.0), vh[:, h0:h1].double())
        del s
        out[:, h0:h1] = (raw.round() if m_o is None else
                         torch.floor((raw * m_o + float(1 << 29)) / float(1 << 30)).clamp_(-127, 127)).to(torch.int64)
    return out.transpose(1, 2).reshape(b, tq, n_heads * dh)


def _var255(q, k, v, m_s, m_o, n_heads, n_kv, dh):
    """``transformer._attention``'s integers through :func:`opt_builder.attention_var` (which records the mass)."""
    assert n_kv == n_heads
    return ob.attention_var(q, k, v, m_s, m_o, n_heads, dh, 255, ob.exp_table(16))


@op_kind("diagnostic/float_softmax_attention/1")
def _float_softmax_op(q, k, v, *, m_s, m_o, heads, kv_heads, dh, pmax):
    return _float_softmax_attention(q, k, v, m_s, m_o, heads, kv_heads, dh)


@op_kind("diagnostic/attention_var255/1")
def _var255_op(q, k, v, *, m_s, m_o, heads, kv_heads, dh, pmax):
    return _var255(q, k, v, m_s, m_o, heads, kv_heads, dh)


def _with_attention(graph: IntGraph, impl) -> IntGraph:
    """``graph`` with every default attention op (kind ``attention/1``) computed by the diagnostic kind ``impl`` with
    the same constants (``None``: ``graph`` itself)."""
    if impl is None:
        return graph
    ops = [dataclasses.replace(o, fn=OpFn(impl, o.fn.consts))
           if isinstance(o, CheapOp) and isinstance(o.fn, OpFn) and o.fn.kind == "attention/1" else o
           for o in graph.ops]
    return IntGraph(ops, graph.input_name, graph.output_name, dict(graph.meta))


def _bootstrap(a: list[float], b: list[float], n: list[int], reps: int = 2000) -> list[float]:
    g = torch.Generator().manual_seed(0)
    a, b = torch.tensor(a, dtype=torch.float64), torch.tensor(b, dtype=torch.float64)
    n = torch.tensor(n, dtype=torch.float64)
    idx = torch.randint(0, len(a), (reps, len(a)), generator=g)
    r = torch.exp(a[idx].sum(1) / n[idx].sum(1) - b[idx].sum(1) / n[idx].sum(1))
    return [float(r.quantile(0.025)), float(r.quantile(0.975))]


def _peak_rss_gb() -> float | None:
    try:
        import resource
    except ImportError:                                         # not on Windows
        return None
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2 ** 20


def _peak_gpu_gb() -> float | None:
    return torch.cuda.max_memory_allocated() / 2 ** 30 if torch.cuda.is_available() else None


def _weights_sha(graph: IntGraph) -> str:
    h = hashlib.sha256()
    for op in graph.mat_ops:
        h.update(op.name.encode() + op.weight.contiguous().numpy().tobytes())
        if op.bias is not None:
            h.update(op.bias.contiguous().numpy().tobytes())
    return h.hexdigest()


def _reference(model, wins, device: str, cache: Path | None):
    if cache is not None and cache.exists():
        ref = torch.load(cache, weights_only=True)
        if ref["windows"] == len(wins):
            return ref
    t0 = time.perf_counter()
    m = model.to(device)
    nll, am = [], []
    with torch.no_grad():
        for w in wins:
            x, a = _nll_argmax(m(w.to(device)).logits, 1.0, w)
            nll.append(x)
            am.append(a)
    model.to("cpu")
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    ref = {"windows": len(wins), "nll": nll, "argmax": torch.stack(am), "seconds": time.perf_counter() - t0}
    if cache is not None:
        torch.save(ref, cache)
    return ref


def _cells(args) -> list[dict]:
    short = args.model.split("/")[-1]
    out = []
    gains = args.gains or [args.gain or BEST_GAIN.get(short, 4.0)]
    for g in args.graphs:
        variants = [("gain", x) for x in gains] if g == "scalar" else [("alpha", a) for a in args.alphas]
        for what, value in variants:
            for pb in args.pbits:
                if g == "scalar":
                    kw = dict({"norm_gain": value}, **json.loads(args.scalar_opts))
                else:
                    kw = dict(SMOOTH, alpha=value, **json.loads(args.smooth_opts))
                if g == "rows":
                    kw.update(weight_scale="row_pow2", row_buckets=4)
                kw["prob_bits"] = pb
                out.append({"graph": g, "alpha": value if what == "alpha" else None, "pbits": pb, "kw": kw})
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="facebook/opt-1.3b")
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--windows", type=int, default=0, help="0: every window of the test split")
    ap.add_argument("--calib-windows", type=int, default=4)
    ap.add_argument("--draws", type=int, nargs="+", default=[0])
    ap.add_argument("--graphs", nargs="+", default=["scalar", "smooth", "rows"])
    ap.add_argument("--gain", type=float, default=None, help="the scalar graph's gain (default: the model's best)")
    ap.add_argument("--gains", type=float, nargs="+", default=None, help="several scalar gains (a recipe search)")
    ap.add_argument("--smooth-opts", default="{}", help="JSON overrides of the smoothed recipe (e.g. pct, pct_att, "
                    "smooth_pct, smooth_target), applied to the smooth and rows graphs")
    ap.add_argument("--scalar-opts", default="{}", help="JSON options added to the scalar graphs (e.g. pct)")
    ap.add_argument("--tag", default="", help="a label written into every record of this run")
    ap.add_argument("--alphas", type=float, nargs="+", default=[0.55])
    ap.add_argument("--pbits", type=int, nargs="+", default=[8, 11])
    ap.add_argument("--float-softmax", action="store_true")
    ap.add_argument("--mass-windows", type=int, default=2)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu",
                    help="where the integer forward runs (the calibration runs on the CPU)")
    ap.add_argument("--ref-device", default=None, help="where the fp32 reference runs (default: --device)")
    ap.add_argument("--ref-cache", default=None)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--out", default=None)
    ap.add_argument("--check", action="store_true", help="validation on one window (OPT-125M at T=256)")
    ap.add_argument("--legacy", action="store_true", help="the paper's 16 x 128 setting (calibration on the test split)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    model, tok = load_opt(args.model)
    if args.check:
        return _check(model, tok, args)
    if args.legacy:
        return _legacy(model, tok, args)

    test = tok(_text("test"), return_tensors="pt").input_ids[0]
    wins = _windows(test, args.seq, args.windows)
    train = tok(_text("train")[:4_000_000], return_tensors="pt").input_ids[0]
    counts = [w.shape[1] - 1 for w in wins]
    ref = _reference(model, wins, args.ref_device or args.device, Path(args.ref_cache) if args.ref_cache else None)
    fp32 = math.exp(sum(ref["nll"]) / sum(counts))
    head = {"model": args.model, "tag": args.tag, "seq": args.seq, "windows": len(wins), "tokens": sum(counts),
            "dataset": "wikitext-2-raw-v1 test, '\\n\\n'.join", "host": platform.node(),
            "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None, "device": args.device,
            "calib_windows": args.calib_windows}
    _emit(args, dict(head, cell="fp32", ppl=fp32, nll=ref["nll"], seconds=ref["seconds"]))
    for draw in args.draws:
        calib = torch.cat(_windows(train, args.seq, args.calib_windows, start=draw * args.calib_windows))
        for cell in _cells(args):
            t0 = time.perf_counter()
            with torch.no_grad():
                graph, info = ob.build_opt(model, calib, **cell["kw"])
            build_s = time.perf_counter() - t0
            digest = graph.digest().hex()
            lean, _ = _all_positions(graph).with_constants_on(args.device)
            softmaxes = ["int"] + (["float"] if args.float_softmax and cell["pbits"] == 8 else [])
            for sm in softmaxes:
                if torch.cuda.is_available():
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                nll, agree, mass = [], 0, []
                with torch.no_grad():
                    for i, (w, r_am) in enumerate(zip(wins, ref["argmax"])):
                        record = sm == "int" and i < args.mass_windows
                        ob.MASS = [] if record else None
                        impl = _float_softmax_op if sm == "float" else (_var255_op if record and cell["pbits"] == 8
                                                                         else None)
                        x, am = _int_eval(_with_attention(lean, impl), info["logit_scale"], w, args.device)
                        if record:
                            mass.append(ob.MASS)
                        nll.append(x)
                        agree += int((am == r_am).sum())
                ob.MASS = None
                ppl = math.exp(sum(nll) / sum(counts))
                layers = [sum(col) / len(col) for col in zip(*mass)] if mass else None
                _emit(args, dict(head, cell=f"{cell['graph']}-p{cell['pbits']}-{sm}", graph=cell["graph"],
                                 alpha=cell["alpha"], pbits=cell["pbits"], softmax=sm, draw=draw,
                                 opts=dict(cell["kw"]), ppl=ppl, fp32=fp32, ratio=ppl / fp32,
                                 ratio_ci95=_bootstrap(nll, ref["nll"], counts), top1=agree / sum(counts),
                                 mass_per_layer=layers, mass_min=min(layers) if layers else None,
                                 nll=nll, digest=digest, build_s=build_s, seconds=time.perf_counter() - t0,
                                 peak_rss_GB=_peak_rss_gb(), peak_gpu_GB=_peak_gpu_gb()))
            del graph, lean, info
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def _emit(args, row: dict) -> None:
    line = json.dumps(row)
    print(line, flush=True)
    short = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()
             if k in ("cell", "draw", "alpha", "ppl", "ratio", "ratio_ci95", "top1", "mass_min", "seconds")}
    print(json.dumps(short), file=sys.stderr, flush=True)
    if args.out:
        with open(args.out, "a", encoding="utf-8") as f:
            f.write(line + "\n")


def _legacy(model, tok, args) -> None:
    """``ppl_smooth.py``'s setting through this script's path: 16 x 128-token test windows, calibration on the
    split's last 128-token window (expected for OPT-1.3B: 32.79 fp32, 33.70 smoothed, 36.44 with gain 3)."""
    wins, calib = load_windows(tok, seq=128, windows=16, calib_windows=1)
    counts = [w.shape[1] - 1 for w in wins]
    ref = _reference(model, wins, "cpu", None)
    fp32 = math.exp(sum(ref["nll"]) / sum(counts))
    print(json.dumps({"cell": "fp32", "ppl": fp32}), flush=True)
    for cell in _cells(argparse.Namespace(**dict(vars(args), pbits=[8]))):
        with torch.no_grad():
            graph, info = ob.build_opt(model, calib[0], **cell["kw"])
        lean, _ = _all_positions(graph).with_constants_on(args.device)
        with torch.no_grad():
            nll = [_int_eval(lean, info["logit_scale"], w, args.device)[0] for w in wins]
        ppl = math.exp(sum(nll) / sum(counts))
        print(json.dumps({"cell": cell["graph"], "alpha": cell["alpha"], "ppl": ppl, "ratio": ppl / fp32}), flush=True)


def _check(model, tok, args) -> None:
    """(a) ``opt_builder`` at 8 bits is ``build_opt_from_hf`` bit for bit (every weight, every cheap op's parameters
    and constants, the logit scale, the logits), for the scalar and the smoothed recipe; (b) the lean path's NLL and
    argmaxes are those of ``real_logits``; (c) ``attention_var`` at 255 gives ``transformer._attention``'s integers;
    (d) on a GPU, the same integers as on the CPU."""
    seq = 256
    test = tok(_text("test"), return_tensors="pt").input_ids[0]
    w = _windows(test, seq, 1)[0]
    calib = torch.cat(_windows(tok(_text("train")[:200_000], return_tensors="pt").input_ids[0], seq, 2))
    report = {"model": args.model, "seq": seq}
    for name, kw_rw, kw_ob in (
            ("scalar", dict(norm_gain=4.0, prob_max=255), dict(norm_gain=4.0)),
            ("smooth", dict(norm_gain=4, smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99, prob_max=255),
             dict(SMOOTH, alpha=0.55))):
        with torch.no_grad():
            g_rw, i_rw = rw.build_opt_from_hf(model, calib, device="cpu", **kw_rw)
            g_ob, i_ob = ob.build_opt(model, calib, **kw_ob)
        same_params = len(g_rw.ops) == len(g_ob.ops) and all(
            type(a) is type(b) and a.name == b.name and a.inputs == b.inputs and a.output == b.output
            and (not isinstance(a, CheapOp) or (a.note == b.note and a.params == b.params and _consts(a) == _consts(b)))
            for a, b in zip(g_rw.ops, g_ob.ops))
        with torch.no_grad():
            nll_rw, am_rw = _nll_argmax(rw.real_logits(g_rw, i_rw, w, all_positions=True), 1.0, w)
            lean = _all_positions(g_ob)
            nll_lean, am_lean = _int_eval(lean, i_ob["logit_scale"], w, "cpu")
            ob.MASS = []
            nll_var, am_var = _int_eval(_with_attention(lean, _var255_op), i_ob["logit_scale"], w, "cpu")
            mass, ob.MASS = ob.MASS, None
            gpu_same = None
            if torch.cuda.is_available():
                env_cpu, _ = lean.forward(w, free=True, send=lambda z: None)
                g_dev, _ = lean.with_constants_on("cuda")
                env_gpu, _ = g_dev.forward(w.cuda(), free=True, send=lambda z: None)
                gpu_same = bool(torch.equal(env_gpu[lean.output_name].cpu(), env_cpu[lean.output_name]))
        # The digest names each cheap op's function by module and qualified name, so a copy of the builder cannot
        # match it; equal weights, constants and logits show the same function.
        report[name] = {"digest_equal (not expected)": g_rw.digest() == g_ob.digest(),
                        "weights_equal": _weights_sha(g_rw) == _weights_sha(g_ob),
                        "params_equal": same_params, "logit_scale_equal": i_rw["logit_scale"] == i_ob["logit_scale"],
                        "lean_equals_real_logits": nll_lean == nll_rw and bool(torch.equal(am_lean, am_rw)),
                        "attention_var_equals": nll_var == nll_lean and bool(torch.equal(am_var, am_lean)),
                        "gpu_equals_cpu": gpu_same, "ppl": math.exp(nll_lean / (seq - 1)), "mass_min": min(mass)}
    with torch.no_grad():
        g, i = ob.build_opt(model, calib, **dict(SMOOTH, alpha=0.55, prob_bits=11))
        ob.MASS = []
        nll, _ = _int_eval(_all_positions(g), i["logit_scale"], w, "cpu")
        report["smooth_p11"] = {"ppl": math.exp(nll / (seq - 1)), "mass_min": min(ob.MASS),
                                "digest": g.digest().hex()}
        ob.MASS = None
    print(json.dumps(report, indent=1), flush=True)
    ok = all(v for k in ("scalar", "smooth") for kk, v in report[k].items()
             if kk.endswith(("_equal", "_equals", "_real_logits", "equals_cpu")) and v is not None)
    print("CHECK", "PASSED" if ok else "FAILED", flush=True)


def _consts(op: CheapOp):
    """The constants of ``op.fn`` (an ``OpFn``) in name order, comparable with ``==``."""
    return [op.fn.kind] + [v.tolist() if torch.is_tensor(v) else v for _, v in sorted(op.fn.consts.items())]


if __name__ == "__main__":
    main()

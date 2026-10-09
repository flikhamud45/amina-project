"""Hybrid execution of an integer OPT graph (``opt_builder.build_opt(..., keep_float=True)``): every
component runs either as the exact integer op or in float, op by op along the graph.

Each tensor is carried as its real value (float64) and, when it lies on its integer grid, its integers.
An integer op whose inputs carry integers computes the graph's exact integers (the same functions the
verifier runs); an integer op fed by a float op applies the same integer rule to ``real / scale``
(rounding, clamping, table look-ups), so its quantisation error is the integer pipeline's.  A float op
computes OPT's real arithmetic from the (possibly quantised) real values it receives.

Components (``FULL`` lists each one's integer features; a component maps to the set of features that
are integer, the empty set meaning float):

* ``w_emb, w_qkv, w_out, w_fc1, w_fc2, w_head``: weight quantisation (per-matrix int8);
* ``norm1, norm2, normf``: the LayerNorm feeding attention, the MLP, and the LM head: ``stats`` (integer
  mean and isqrt sigma, no eps), ``round`` (to 1/G), ``clip`` (+-127/G);
* ``rq_q, rq_k, rq_v, rq_fc1``: activation requantisation to int8: ``round``, ``clip``;
* ``softmax``: ``exp`` (the 1/16-nat Q15 exp table) and ``pbits`` (probabilities rounded to 1/255);
* ``att_out``: the attention output requantisation to int8: ``round``, ``clip``;
* ``res``: residual stream: ``round`` (branch outputs and embeddings rounded to the residual scale),
  ``clip`` (+-RES_MAX);
* ``relu``: the ReLU look-up table (exact on int8 inputs, so it never adds error by itself).
"""

from __future__ import annotations

import math

import torch

from pvi.fullcheck.graph import CheapOp, MatOp
from pvi.fullcheck.transformer import RES_MAX, SHIFT, _lut

from opt_builder import _RELU

FULL = {
    "w_emb": {"q"}, "w_qkv": {"q"}, "w_out": {"q"}, "w_fc1": {"q"}, "w_fc2": {"q"}, "w_head": {"q"},
    "norm1": {"stats", "round", "clip"}, "norm2": {"stats", "round", "clip"}, "normf": {"stats", "round", "clip"},
    "rq_q": {"round", "clip"}, "rq_k": {"round", "clip"}, "rq_v": {"round", "clip"}, "rq_fc1": {"round", "clip"},
    "softmax": {"exp", "pbits"}, "att_out": {"round", "clip"}, "res": {"round", "clip"}, "relu": {"lut"},
}

GROUPS = {
    "weights (all matrices)": ["w_emb", "w_qkv", "w_out", "w_fc1", "w_fc2", "w_head"],
    "weights: embeddings": ["w_emb"],
    "weights: q/k/v": ["w_qkv"],
    "weights: out_proj": ["w_out"],
    "weights: fc1": ["w_fc1"],
    "weights: fc2": ["w_fc2"],
    "weights: LM head": ["w_head"],
    "LayerNorm outputs (all three)": ["norm1", "norm2", "normf"],
    "LayerNorm before attention": ["norm1"],
    "LayerNorm before MLP": ["norm2"],
    "final LayerNorm (LM head input)": ["normf"],
    "q/k/v requant": ["rq_q", "rq_k", "rq_v"],
    "fc1 requant": ["rq_fc1"],
    "softmax (exp table + 8-bit p)": ["softmax"],
    "attention output requant": ["att_out"],
    "residual stream (round + clamp)": ["res"],
    "ReLU table": ["relu"],
}


class Val:
    __slots__ = ("real", "ints")

    def __init__(self, real, ints=None):
        self.real = real
        self.ints = ints


def _rhu(x):
    """Round half up (the graph's rounding)."""
    return torch.floor(x + 0.5)


def _f64(m, key):
    """``m[key]`` in float64, converted once and cached in ``m``."""
    k = "_" + key + "64"
    if k not in m:
        m[k] = m[key].double()
    return m[k]


def _vec(s, like):
    return s.to(like.dtype) if torch.is_tensor(s) else s


def run(graph, info, ids: torch.Tensor, cfg: dict[str, set]) -> torch.Tensor:
    """Real logits ``[B, T, vocab]`` at every position with the components' features in ``cfg``
    (missing components: fully integer)."""
    feats = {c: set(cfg.get(c, FULL[c])) for c in FULL}
    meta = info["meta"]
    s_res = info["s_residual"]
    env: dict[str, Val] = {"x": Val(None, ids)}
    for op in graph.ops:
        m = meta[op.name]
        kind = m["kind"]
        if isinstance(op, MatOp):
            x = env[op.inputs[0]]
            integer = "q" in feats[m["comp"]]
            if op.layout == "embed":
                if integer:
                    z = op.fold(op.compute(x.ints), x.ints)
                    env[op.output] = Val(z.double() * m["s_z"], z)
                else:
                    env[op.output] = Val(_f64(m, "w_float").T[x.ints])
                continue
            if integer and x.ints is not None:
                z = op.fold(op.compute(x.ints), x.ints)
                env[op.output] = Val(z.double() * _vec(m["s_z"], z.double()), z)
            elif integer:
                out = x.real @ _f64(m, "w_deq").T
                env[op.output] = Val(out if m["b_deq"] is None else out + m["b_deq"])
            else:
                out = x.real @ _f64(m, "w_float").T
                env[op.output] = Val(out if m["b_float"] is None else out + m["b_float"])
            continue
        if kind == "pos":
            env[op.output] = Val(None, op.fn(env["x"].ints))
        elif kind == "scale":
            x = env[op.inputs[0]]
            env[op.output] = Val(x.real, None if x.ints is None else x.ints * m["c"])
        elif kind == "last":
            env[op.output] = env[op.inputs[0]]
        elif kind == "act":
            x = env[op.inputs[0]]
            if "lut" in feats["relu"] and x.ints is not None:
                y = _lut(x.ints, _RELU)
                env[op.output] = Val(x.real.clamp_min(0), y)          # y = relu(x.ints): the same real value
            else:
                env[op.output] = Val(x.real.clamp_min(0))
        elif kind == "res":
            env[op.output] = _residual(op, m, env[op.inputs[0]], env[op.inputs[1]], feats["res"], s_res)
        elif kind == "norm":
            env[op.output] = _norm(op, m, env[op.inputs[0]], feats[m["comp"]], s_res)
        elif kind == "rq":
            env[op.output] = _requant(op, m, env[op.inputs[0]], feats[m["comp"]])
        elif kind == "attn":
            env[op.output] = _attn(op, m, *[env[n] for n in op.inputs], feats["softmax"], feats["att_out"])
        else:
            raise ValueError(kind)
    return env[graph.output_name].real


def _residual(op, m, r: Val, z: Val, f: set, s_res: float) -> Val:
    if f == FULL["res"] and r.ints is not None and z.ints is not None:
        y = op.fn(r.ints, z.ints)
        return Val(y.double() * s_res, y)
    if not f:
        return Val(r.real + z.real)
    rr = r.ints.double() if r.ints is not None else r.real / s_res
    if z.ints is not None and torch.is_tensor(m["mult"]) is False:
        dz = z.ints.double() * m["mult"] / (1 << SHIFT)
    else:
        dz = z.real / s_res
    if "round" in f:
        rr, dz = _rhu(rr), _rhu(dz)
    out = rr + dz
    if "clip" in f:
        out = out.clamp(-RES_MAX, RES_MAX)
    ints = out.to(torch.int64) if "round" in f else None
    return Val(out * s_res, ints)


def _norm(op, m, r: Val, f: set, s_res: float) -> Val:
    G = m["G"]                                  # [d] float64, the exact gain g / 2^16
    if f == FULL["norm1"] and r.ints is not None:
        y = op.fn(r.ints)
        return Val(y.double() / G, y)
    d = G.numel()
    if "stats" in f:
        if r.ints is not None:
            x = r.ints
            mean = torch.div(x.sum(-1, keepdim=True) + d // 2, d, rounding_mode="floor")
            xc = x - mean
            var = torch.div((xc * xc).sum(-1, keepdim=True), d, rounding_mode="floor")
            sigma = torch.sqrt(var.double()).floor().clamp_min(1)
            # exact isqrt correction (float sqrt can be off by one near perfect squares)
            sigma = torch.where((sigma + 1) ** 2 <= var.double(), sigma + 1, sigma)
            sigma = torch.where(sigma ** 2 > var.double(), sigma - 1, sigma).clamp_min(1)
            n = xc.double() / sigma
        else:
            x = r.real / s_res
            mean = torch.floor((x.sum(-1, keepdim=True) + d // 2) / d)
            xc = x - mean
            sigma = torch.floor((xc * xc).sum(-1, keepdim=True) / d).sqrt().floor().clamp_min(1)
            n = xc / sigma
    else:
        h = r.real if r.ints is None else r.ints.double() * s_res
        mu = h.mean(-1, keepdim=True)
        hc = h - mu
        n = hc / torch.sqrt((hc * hc).mean(-1, keepdim=True) + 1e-5)
    y = n * G
    if "round" in f:
        y = _rhu(y)
    if "clip" in f:
        y = y.clamp(-127, 127)
    ints = y.to(torch.int64) if ("round" in f and "clip" in f) else None
    return Val(y / G, ints)


def _requant(op, m, z: Val, f: set) -> Val:
    s_out = m["s_out"]
    if f == FULL["rq_q"] and z.ints is not None:
        y = op.fn(z.ints)
        return Val(y.double() * s_out, y)
    if not f:
        return Val(z.real)
    if z.ints is not None and not torch.is_tensor(m["mult"]):
        u = z.ints.double() * m["mult"] / (1 << SHIFT)
    else:
        u = z.real / s_out
    if "round" in f:
        u = _rhu(u)
    if "clip" in f:
        u = u.clamp(-127, 127)
    ints = u.to(torch.int64) if ("round" in f and "clip" in f) else None
    return Val(u * s_out, ints)


def _attn(op, m, q: Val, k: Val, v: Val, fs: set, fo: set) -> Val:
    if fs == FULL["softmax"] and fo == FULL["att_out"] and q.ints is not None and k.ints is not None \
            and v.ints is not None:
        y = op.fn(q.ints, k.ints, v.ints)
        return Val(y.double() * m["s_att"], y)
    H, dh = m["n_heads"], m["dh"]
    b, t, _ = v.real.shape

    def heads(a):
        return a.reshape(b, t, H, dh).transpose(1, 2)

    mask = ~torch.ones(t, t, dtype=torch.bool).tril()
    vr = heads(v.real)
    raw_ints = None
    if not fs:
        sc = heads(q.real) @ heads(k.real).transpose(-1, -2) / math.sqrt(dh)
        prob = torch.softmax(sc.masked_fill(mask, -math.inf), -1)
    else:
        qi = q.ints.double() if q.ints is not None else q.real / m["sq"]
        ki = k.ints.double() if k.ints is not None else k.real / m["sk"]
        s = heads(qi) @ heads(ki).transpose(-1, -2)
        s = s.masked_fill(mask, -math.inf)
        gap = (s.amax(-1, keepdim=True) - s)
        if "exp" in fs:
            tab = m["table"].double()
            idx = torch.floor(gap.clamp(max=1e12) * m["m_s"] / (1 << SHIFT) + 0.5).clamp(0, tab.numel() - 1)
            e = tab[idx.to(torch.int64)].masked_fill(mask, 0.0)
        else:
            e = (32768.0 * torch.exp(-gap * m["sq"] * m["sk"] / math.sqrt(dh))).masked_fill(mask, 0.0)
        tot = e.sum(-1, keepdim=True)
        if "pbits" in fs:
            P = m["pmax"]
            p = torch.floor((tot + 2 * P * e) / (2 * tot))
            prob = p / P
            if "exp" in fs and v.ints is not None:
                raw_ints = p @ heads(v.ints.double())           # exact (sums below 2^53)
        else:
            prob = e / tot
    raw_real = prob @ vr                                         # [B, H, T, dh]
    if not fo:
        out = raw_real
        ints = None
    else:
        if raw_ints is not None:
            u = raw_ints * m["m_o"] / (1 << SHIFT)
        else:
            u = raw_real / m["s_att"]
        if "round" in fo:
            u = _rhu(u)
        if "clip" in fo:
            u = u.clamp(-127, 127)
        out = u * m["s_att"]
        ints = u if ("round" in fo and "clip" in fo) else None
    out = out.transpose(1, 2).reshape(b, t, H * dh)
    if ints is not None:
        ints = ints.transpose(1, 2).reshape(b, t, H * dh).to(torch.int64)
    return Val(out, ints)


def float_cfg(*comps) -> dict:
    """Everything integer except ``comps`` (float)."""
    return {c: set() for c in comps}


def int_only_cfg(*comps) -> dict:
    """Everything float except ``comps`` (integer)."""
    return {c: (FULL[c] if c in comps else set()) for c in FULL}

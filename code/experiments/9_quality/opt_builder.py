"""A copy of ``pvi.fullcheck.real_weights.build_opt_from_hf`` with quality options (the F2 experiments).

Every option changes only public constants of cheap ops or the offline weight quantisation; the graph
has the same ops, in the same order, with the same weight-op shapes, and stays exact integer arithmetic
(``matches_benchmark_graph`` still holds).  With the defaults the graph is the one ``build_opt_from_hf``
builds (same weights, multipliers and gains; checked by ``attribution.py --check``).

Options
-------
* ``norm_mode="scalar"`` (default, gain ``norm_gain`` for every channel) or ``"smooth"``: a PER-CHANNEL
  gain vector ``G_j`` for each LayerNorm (SmoothQuant-style migration).  The norm op is unchanged
  (``_norm_int`` already takes a gain vector of public constants); ``1/G_j`` is folded into column ``j``
  of the next weight matrices.  ``G_j = c / s_j`` with ``s_j = a_j^alpha / w_j^(1-alpha)``, ``a_j`` the
  ``smooth_pct`` percentile of ``|normalised_j|`` on calibration data, ``w_j = max_i |W_ij gamma_j|`` over
  the consumers, and ``c`` such that the largest ``a_j G_j`` is ``smooth_target`` (127 = no clipping of
  ``a_j``).
* ``weight_scale="tensor"`` (default, one scale per matrix), ``"row"`` (one scale per output row; the
  consumer's multiplier becomes a VECTOR -- not V1-compatible, reported separately) or ``"row_pow2"``
  (row scales ``s_tensor 2^-e`` with ``0 <= e < row_buckets``: ``row_buckets`` distinct multipliers per op).
  The LM head always keeps one scale (no cheap op follows it to absorb per-row scales).
* ``prob_bits`` (default 8) and ``exp_res`` (default 16, the exp table's index step is ``1/exp_res`` nat):
  the attention's probabilities ``p = round(P e / tot)`` with ``P = 2^prob_bits - 1``.  Anything but
  (8, 16) uses :func:`attention_var` (exact int64 torch), not ``transformer._attention``.
* ``pct`` (percentile clipping at q/k/v/fc1 requantisation), ``pct_att`` (attention output, default max),
  ``relu_calib`` (fc1's scale from its positive part only: the ReLU zeroes the negatives anyway),
  ``residual_headroom``; ``calib_ids`` may hold several windows ``[B, T]``.
"""

from __future__ import annotations

import math

import torch

from pvi.fullcheck.graph import INT8_MAX, CheapOp, IntGraph, MatOp, requant, requant_fn, residual_add
from pvi.fullcheck.transformer import RES_MAX, SHIFT, _attention, _isqrt, _lut, _norm_int

_RELU = torch.tensor([max(0, x) for x in range(-128, 128)], dtype=torch.int64)


# ------------------------------------------------------------------ generalised attention (exact int64)

def exp_table(res: int, q: int = 15) -> torch.Tensor:
    """``round(2^q exp(-i/res))`` up to and including its first zero entry."""
    vals = []
    i = 0
    while True:
        v = round((1 << q) * math.exp(-i / res))
        vals.append(v)
        if v == 0:
            break
        i += 1
    return torch.tensor(vals, dtype=torch.int64)


def attention_var(q, k, v, m_s: int, m_o: int | None, n_heads: int, dh: int, pmax: int, table: torch.Tensor):
    """Integer causal attention with ``pmax``-level probabilities and an arbitrary exp table.

    ``s = q k^T`` (exact), ``idx = clamp((gap m_s + 2^29) >> 30, 0, len-1)``, ``e = table[idx]``,
    ``p = floor((tot + 2 pmax e) / (2 tot))`` (round-half-up of ``pmax e / tot``), raw ``P V`` (exact: float64
    sums below ``2^53``), requantised by ``m_o`` (``None``: raw).  ``(pmax, table) = (255, the 1/16-nat
    table)`` gives ``transformer._attention``'s integers."""
    b, tq, _ = q.shape
    t = k.shape[1]
    qh = q.reshape(b, tq, n_heads, dh).transpose(1, 2).to(torch.float64)
    kh = k.reshape(b, t, n_heads, dh).transpose(1, 2).to(torch.float64)
    vh = v.reshape(b, t, n_heads, dh).transpose(1, 2).to(torch.float64)
    s = torch.matmul(qh, kh.transpose(-1, -2)).to(torch.int64)            # |s| <= 2^21: exact
    mask = ~torch.ones(t, t, dtype=torch.bool, device=q.device).tril()[t - tq:]
    s = s.masked_fill(mask, -(1 << 40))
    gap = s.amax(-1, keepdim=True) - s
    idx = ((gap.clamp(max=1 << 31) * m_s + (1 << (SHIFT - 1))) >> SHIFT).clamp(0, table.numel() - 1)
    e = table.to(q.device)[idx].masked_fill(mask, 0)
    tot = e.sum(-1, keepdim=True)
    p = (tot + 2 * pmax * e).div(2 * tot, rounding_mode="floor")
    raw = torch.matmul(p.to(torch.float64), vh).to(torch.int64)            # [B, H, Tq, dh]
    if m_o is not None:
        raw = requant(raw, torch.tensor(m_o, dtype=torch.int64), SHIFT, -INT8_MAX, INT8_MAX)
    return raw.transpose(1, 2).reshape(b, tq, n_heads * dh)


def _vec_requant_fn(mult: torch.Tensor, kind: str, res_max: int = RES_MAX):
    """Per-row (vector) multiplier requantisation: NOT V1-compatible (``requant_fn`` takes a scalar)."""
    if kind == "requant":
        def fn(a, m=mult):
            return requant(a, m.to(a.device), SHIFT, -INT8_MAX, INT8_MAX)
    else:
        def fn(a, b, m=mult):
            return residual_add(a, b, m.to(b.device), SHIFT, res_max)
    return fn


# ------------------------------------------------------------------ the builder

class _Builder:
    def __init__(self, tokens, o: dict) -> None:
        self.o = o
        self.ops: list = []
        self.env = {"x": tokens}
        self.scale: dict[str, object] = {}        # real scale: float, or a per-channel / per-row vector
        self.meta: dict[str, dict] = {}            # per-op information for the hybrid runner
        self._n = 0

    def name(self, p):
        self._n += 1
        return f"{p}{self._n}"

    def _run(self, op, scale, meta):
        self.ops.append(op)
        if isinstance(op, MatOp):
            xin = self.env[op.inputs[0]]
            self.env[op.output] = op.fold(op.compute(xin), xin)
        else:
            self.env[op.output] = op.fn(*[self.env[n] for n in op.inputs])
        self.scale[op.output] = scale
        self.meta[op.name] = meta
        return op.output

    # -- weight ops
    def mat(self, inp, w, b, s_in, comp, layer, layout="linear", s_w=None, per_row=None):
        """Real ``w [rows, cols]`` acting on the real input; ``s_in`` is the input's real scale (scalar or a
        per-column vector).  Per-row: ``s_z`` is a vector over the rows."""
        n = self.name("mm")
        if per_row is None:
            per_row = self.o["weight_scale"] if comp in self.o["row_ops"] else "tensor"
        vec_in = torch.is_tensor(s_in)
        if not vec_in and per_row == "tensor":       # build_opt_from_hf's arithmetic, bit for bit
            s_w = float(w.abs().max()) / INT8_MAX if s_w is None else s_w
            w_int = (w / s_w).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
            s_z = s_w * s_in
            b_int = None if b is None else (b.double() / s_z).round().to(torch.int64)
            w_deq = w_int.double() * s_w if self.o["keep_float"] else None   # in the real input's domain
        else:
            w_eff = w.double() * (s_in.double()[None, :] if vec_in else s_in)   # acts on the integer input
            if per_row == "tensor":
                s_z = float(w_eff.abs().max()) / INT8_MAX
                w_int = (w_eff / s_z).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
            else:
                row = w_eff.abs().amax(1).clamp_min(1e-30) / INT8_MAX
                if per_row == "row_pow2":
                    top = float(row.max())
                    e = torch.floor(torch.log2(top / row)).clamp(0, self.o["row_buckets"] - 1)
                    row = top * torch.pow(2.0, -e)
                s_z = row
                w_int = (w_eff / s_z[:, None]).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
            sz = s_z if torch.is_tensor(s_z) else torch.tensor(s_z, dtype=torch.float64)
            b_int = None if b is None else (b.double() / sz).round().to(torch.int64)
            del w_eff
            if self.o["keep_float"]:
                w_deq = w_int.double() * (sz[:, None] if sz.dim() else sz)
                w_deq = w_deq / (s_in.double()[None, :] if vec_in else s_in)
        meta = {"kind": "mat", "comp": comp, "layer": layer, "s_in": s_in, "s_z": s_z, "layout": layout}
        if self.o["keep_float"]:
            meta["w_float"] = w.float()
            meta["b_float"] = None if b is None else b.double()
            meta["w_deq"] = w_deq.float()
            meta["b_deq"] = None if b_int is None else b_int.double() * s_z
        return self._run(MatOp(n, (inp,), n + ".z", weight=w_int, bias=b_int, layout=layout), s_z, meta)

    def cheap(self, prefix, inputs, fn, scale, note="", params=None, meta=None):
        n = self.name(prefix)
        return self._run(CheapOp(n, tuple(inputs), n + ".y", fn=fn, note=note, params=params or {}), scale,
                         meta or {"kind": prefix})

    def requant_op(self, prefix, inputs, scale, note, meta, **params):
        mult = params["mult"]
        if torch.is_tensor(mult):           # per-row multipliers (not V1-compatible)
            fn = _vec_requant_fn(mult, params["kind"])
            return self.cheap(prefix, inputs, fn, scale, note, {"vec_requant": {"kind": params["kind"]}}, meta)
        return self.cheap(prefix, inputs, requant_fn(params), scale, note, {"requant": params}, meta)

    def _mult(self, s_z, s_out):
        if torch.is_tensor(s_z):
            return torch.round((1 << SHIFT) * s_z / s_out).to(torch.int64)
        return round((1 << SHIFT) * s_z / s_out)

    def to_int8(self, z, comp, layer, pct=None, relu=False):
        pct = self.o["pct"] if pct is None else pct
        sz = self.scale[z]
        sz_t = sz if torch.is_tensor(sz) else torch.tensor(sz, dtype=torch.float64)
        real = self.env[z].double() * sz_t
        real = real.clamp_min(0) if relu else real.abs()
        flat = real.flatten()
        if relu:
            flat = flat[flat > 0]
        if flat.numel() > 1 << 24:
            flat = flat[torch.randperm(flat.numel(), generator=torch.Generator().manual_seed(0))[:1 << 24]]
        top = float(torch.quantile(flat, pct / 100.0)) if pct < 100 else float(flat.max())
        s_out = max(top, 1e-12) / INT8_MAX
        m = self._mult(sz, s_out)
        meta = {"kind": "rq", "comp": comp, "layer": layer, "s_in": sz, "s_out": s_out, "mult": m}
        return self.requant_op("rq", [z], s_out, "requant", meta, kind="requant", mult=m, shift=SHIFT,
                               lo=-INT8_MAX, hi=INT8_MAX)

    def residual(self, r, z, layer, comp="res"):
        m = self._mult(self.scale[z], self.scale[r])
        meta = {"kind": "res", "comp": comp, "layer": layer, "s_res": self.scale[r], "s_in": self.scale[z],
                "mult": m}
        return self.requant_op("res", [r, z], self.scale[r], "residual add", meta, kind="residual", mult=m,
                               shift=SHIFT, res_max=RES_MAX)

    def norm(self, r, d, comp, layer, consumers):
        """``consumers``: ``[(W, gamma)]`` of the matrices this norm feeds (for the smooth gains)."""
        o = self.o
        if o["norm_mode"] == "scalar":
            G = torch.full((d,), float(o["norm_gain"]), dtype=torch.float64)
        else:
            x = self.env[r]
            mean = torch.div(x.sum(-1, keepdim=True) + d // 2, d, rounding_mode="floor")
            xc = x - mean
            sigma = _isqrt(torch.div((xc * xc).sum(-1, keepdim=True), d, rounding_mode="floor"))
            n = (xc.double() / sigma.double()).reshape(-1, d).abs()
            if o["smooth_pct"] < 100:
                a = torch.quantile(n[:min(n.shape[0], 1 << 14)], o["smooth_pct"] / 100.0, dim=0)
            else:
                a = n.amax(0)
            a = a.clamp_min(1e-3)
            wj = torch.stack([(w.double().abs() * g.double().abs()[None, :]).amax(0) for w, g in consumers]
                             ).amax(0).clamp_min(1e-8)
            alpha = o["alpha"]
            s = a.pow(alpha) / wj.pow(1 - alpha)
            c = o["smooth_target"] / float((a / s).max())
            G = (c / s).clamp(1.0 / 64, float(1 << 14))
        gain = torch.round(G * (1 << 16)).to(torch.int64)
        G_eff = gain.double() / (1 << 16)
        scale = 1.0 / float(o["norm_gain"]) if o["norm_mode"] == "scalar" else 1.0 / G_eff
        meta = {"kind": "norm", "comp": comp, "layer": layer, "gain": gain, "G": G_eff}
        return self.cheap("norm", [r], lambda a, g=gain: _norm_int(a, g, True), scale, "norm", meta=meta)

    def attention(self, q, k, v, n_heads, dh, layer):
        o = self.o
        sq, sk, sv = self.scale[q], self.scale[k], self.scale[v]
        default = o["prob_bits"] == 8 and o["exp_res"] == 16
        pmax = (1 << o["prob_bits"]) - 1
        table = exp_table(o["exp_res"])
        m_s = round((1 << SHIFT) * float(o["exp_res"]) * sq * sk / math.sqrt(dh))
        if default:
            raw = _attention(self.env[q], self.env[k], self.env[v], m_s, None, n_heads, n_heads, dh)
        else:
            raw = attention_var(self.env[q], self.env[k], self.env[v], m_s, None, n_heads, dh, pmax, table)
        s_raw = sv / pmax
        rabs = raw.double().abs().flatten() * s_raw
        pa = o["pct_att"]
        top = float(torch.quantile(rabs[:1 << 24], pa / 100.0)) if pa < 100 else float(rabs.max())
        s_att = max(top, 1e-12) / INT8_MAX
        m_o = round((1 << SHIFT) * s_raw / s_att)
        meta = {"kind": "attn", "comp": "attn", "layer": layer, "m_s": m_s, "m_o": m_o, "pmax": pmax,
                "table": table, "s_raw": s_raw, "s_att": s_att, "sq": sq, "sk": sk, "sv": sv,
                "n_heads": n_heads, "dh": dh}
        if default:
            fn = (lambda a, b_, c, ms=m_s, mo=m_o: _attention(a, b_, c, ms, mo, n_heads, n_heads, dh))
        else:
            fn = (lambda a, b_, c, ms=m_s, mo=m_o, P=pmax, tb=table: attention_var(a, b_, c, ms, mo, n_heads, dh, P,
                                                                                   tb))
        return self.cheap("attn", [q, k, v], fn, s_att, "attention", meta=meta)


def _fold(w, b, gamma, beta):
    w32 = w.double()
    wf = w32 * gamma.double()[None, :]
    bf = w32 @ beta.double() + (b.double() if b is not None else 0.0)
    return wf, bf


def _act_stats(model, calib_ids):
    """Per layer, per channel max |.| (fp32 checkpoint on the calibration windows) of the q, k, v projections'
    outputs, out_proj's input and fc2's input."""
    stats, hooks = {}, []

    def out_hook(key):
        def h(mod, inp, out):
            stats[key] = out.detach().abs().reshape(-1, out.shape[-1]).amax(0).double()
        return h

    def in_hook(key):
        def h(mod, inp, out):
            x = inp[0]
            stats[key] = x.detach().abs().reshape(-1, x.shape[-1]).amax(0).double()
        return h

    for i, layer in enumerate(model.model.decoder.layers):
        at = layer.self_attn
        hooks += [at.q_proj.register_forward_hook(out_hook((i, "q"))), at.k_proj.register_forward_hook(out_hook((i, "k"))),
                  at.v_proj.register_forward_hook(out_hook((i, "v"))), at.out_proj.register_forward_hook(in_hook((i, "o"))),
                  layer.fc2.register_forward_hook(in_hook((i, "h")))]
    try:
        with torch.no_grad():
            model(calib_ids)
    finally:
        for h in hooks:
            h.remove()
    return stats


def _eq_scale(act, r1, r2, alpha, ratio):
    """Per-channel ``s_j`` dividing channel ``j`` of an intermediate tensor (produced by rows of matrix 1, read by
    columns of matrix 2): ``s_j = sqrt(act_j^alpha r1_j^(1-alpha) / r2_j)``, so afterwards the producer's row range
    (alpha=0) or the activation's range (alpha=1) meets the reader's column range ``r2_j s_j`` (alpha=0 is
    data-free cross-layer equalisation), clamped to ``[median / ratio, median * ratio]``."""
    a = act.clamp_min(1e-8 * float(act.max()))
    s = (a.pow(alpha) * r1.double().clamp_min(1e-12).pow(1 - alpha) / r2.double().clamp_min(1e-12)).sqrt()
    return _clamp_ratio(s, ratio)


def _clamp_ratio(s, r):
    med = float(s.median())
    return s.clamp(med / r, med * r)


DEFAULTS = dict(pct=100.0, pct_att=100.0, norm_gain=4.0, norm_mode="scalar", alpha=0.5, smooth_pct=100.0,
                smooth_target=127.0, weight_scale="tensor", row_buckets=4, prob_bits=8, exp_res=16,
                relu_calib=False, residual_headroom=1.25, keep_float=False, emb_row=False,
                eq_qk=False, eq_vo=None, eq_mlp=None, eq_ratio=32.0,
                row_ops=("w_qkv", "w_out", "w_fc1", "w_fc2"), pct_fc1=None, hidden_max=None)


def hidden_max(model, calib_ids) -> float:
    """Largest |hidden state| of the fp checkpoint on the calibration windows (sets the residual scale)."""
    with torch.no_grad():
        hs = model(calib_ids, output_hidden_states=True).hidden_states
    return max(float(h.abs().max()) for h in hs)


def build_opt(model, calib_ids: torch.Tensor, **opts) -> tuple[IntGraph, dict]:
    """``calib_ids`` ``[B, T]`` (B calibration windows).  Returns ``(graph, info)``; ``info["meta"]`` has
    every op's scales (and, with ``keep_float``, the float and dequantised weights for the hybrids)."""
    o = dict(DEFAULTS)
    bad = set(opts) - set(o)
    assert not bad, bad
    o.update(opts)
    cfg = model.config
    assert cfg.do_layer_norm_before and cfg.word_embed_proj_dim == cfg.hidden_size
    dec = model.model.decoder
    d, n_heads = cfg.hidden_size, cfg.num_attention_heads
    dh = d // n_heads

    if o["hidden_max"] is None:
        o["hidden_max"] = hidden_max(model, calib_ids)
    res_max = o["hidden_max"] * o["residual_headroom"]
    s_res = max(res_max / RES_MAX, float(dec.embed_tokens.weight.abs().max()) / (INT8_MAX * 8))

    eq = o["eq_qk"] or o["eq_vo"] is not None or o["eq_mlp"] is not None
    stats = _act_stats(model, calib_ids) if eq else {}

    B = _Builder(calib_ids, o)
    embed_mult = max(1, round(float(dec.embed_tokens.weight.abs().max()) / INT8_MAX / s_res))
    tok = B.mat("x", dec.embed_tokens.weight.T.float(), None, 1.0, "w_emb", -1, layout="embed",
                s_w=embed_mult * s_res, per_row="tensor")
    pos_ids = B.cheap("pos", ["x"], lambda x: torch.arange(x.shape[1], device=x.device)
                      .expand(x.shape[0], -1).contiguous(), 1.0, "position ids", meta={"kind": "pos"})
    max_pos = cfg.max_position_embeddings
    pos_table = dec.embed_positions.weight[2:2 + max_pos].T.float()
    pos = B.mat(pos_ids, pos_table, None, 1.0, "w_emb", -1, layout="embed", per_row="tensor")
    sc = B.cheap("scale", [tok], lambda a, c=embed_mult: a * c, s_res, "embed to residual",
                 meta={"kind": "scale", "c": embed_mult})
    r = B.residual(sc, pos, -1, comp="res")

    for li, layer in enumerate(dec.layers):
        at = layer.self_attn
        ln1 = layer.self_attn_layer_norm
        projs = (at.q_proj, at.k_proj, at.v_proj)
        folded = [_fold(p.weight.float(), p.bias, ln1.weight, ln1.bias) for p in projs]
        wo, bo = at.out_proj.weight.float(), at.out_proj.bias
        ln2 = layer.final_layer_norm
        w1, b1 = _fold(layer.fc1.weight.float(), layer.fc1.bias, ln2.weight, ln2.bias)
        w2, b2 = layer.fc2.weight.float(), layer.fc2.bias
        # cross-layer equalisation (offline weight rescaling; the float function is unchanged):
        if o["eq_qk"]:        # q_j k_j is invariant under q_j / s_j, k_j s_j
            s_ = _clamp_ratio((stats[li, "q"] / stats[li, "k"].clamp_min(1e-8)).sqrt(), o["eq_ratio"])
            (wq, bq), (wk, bk) = folded[0], folded[1]
            folded[0] = (wq / s_[:, None], bq / s_)
            folded[1] = (wk * s_[:, None], bk * s_)
        if o["eq_vo"] is not None:   # out_proj(o) with o / s_j (via v's rows) and out_proj's columns * s_j
            a = o["eq_vo"]
            wv, bv = folded[2]
            s_ = _eq_scale(stats[li, "o"], wv.abs().amax(1), wo.double().abs().amax(0), a, o["eq_ratio"])
            folded[2] = (wv / s_[:, None], bv / s_)
            wo = (wo.double() * s_[None, :]).float()
        if o["eq_mlp"] is not None:  # relu(x / s_j) = relu(x) / s_j for s_j > 0; fc2's columns * s_j
            a = o["eq_mlp"]
            s_ = _eq_scale(stats[li, "h"], w1.abs().amax(1), w2.double().abs().amax(0), a, o["eq_ratio"])
            w1, b1 = w1 / s_[:, None], b1 / s_
            w2 = (w2.double() * s_[None, :]).float()
        h = B.norm(r, d, "norm1", li, [(wf, torch.ones(d)) for wf, _ in folded])
        qkv = []
        for (wf, bf), nm in zip(folded, ("q", "k", "v")):
            qkv.append(B.to_int8(B.mat(h, wf.float(), bf, B.scale[h], "w_qkv", li), "rq_" + nm, li))
        q, k, v = qkv
        att = B.attention(q, k, v, n_heads, dh, li)
        r = B.residual(r, B.mat(att, wo, bo, B.scale[att], "w_out", li), li)

        h = B.norm(r, d, "norm2", li, [(w1, torch.ones(d))])
        f1 = B.to_int8(B.mat(h, w1.float(), b1, B.scale[h], "w_fc1", li), "rq_fc1", li, relu=o["relu_calib"],
                       pct=o["pct_fc1"])
        act = B.cheap("act", [f1], lambda a, t=_RELU: _lut(a, t), B.scale[f1], "relu",
                      meta={"kind": "act", "comp": "relu", "layer": li})
        r = B.residual(r, B.mat(act, w2, b2, B.scale[act], "w_fc2", li), li)

    lnf = dec.final_layer_norm
    wf, bf = _fold(model.lm_head.weight.float(), None, lnf.weight, lnf.bias)
    h = B.norm(r, d, "normf", len(dec.layers), [(wf, torch.ones(d))])
    h = B.cheap("last", [h], lambda a: a[:, -1:, :].contiguous(), B.scale[h], "last position",
                meta={"kind": "last"})
    logits = B.mat(h, wf.float(), bf, B.scale[h], "w_head", len(dec.layers), per_row="tensor")
    info = {"s_residual": s_res, "residual_real_max": res_max, "logit_scale": B.scale[logits],
            "embed_mult": embed_mult, "opts": {k: v for k, v in o.items() if k != "keep_float"}, "meta": B.meta}
    return IntGraph(B.ops, "x", logits), info


def int_logits(graph: IntGraph, info: dict, ids: torch.Tensor) -> torch.Tensor:
    """Real logits of the integer graph at every position (``real_weights.real_logits(all_positions=True)``)."""
    ops = [CheapOp(o.name, o.inputs, o.output, fn=(lambda a: a), note=o.note)
           if isinstance(o, CheapOp) and o.note == "last position" else o for o in graph.ops]
    env, _ = IntGraph(ops, graph.input_name, graph.output_name, dict(graph.meta)).forward(ids)
    return env[graph.output_name].double() * info["logit_scale"]


def graph_report(graph: IntGraph, info: dict | None = None) -> dict:
    """Constraint checks: honest claim bound (< 2^29), multiplier ranges (V1's P3 wants 2^14 <= m <= 2^29 for a
    cut op; only requantisations of LINEAR weight ops can be cut, so the embedding's residual add is listed
    apart), vector multipliers (per-row), distinct multipliers per op, norm gain ranges."""
    from pvi.fullcheck.protocol import claim_bound
    worst_claim = max(claim_bound(op) for op in graph.mat_ops)
    producer = {op.output: op for op in graph.ops}
    mults, vec_ops, distinct, outside = [], 0, [], []
    for op in graph.ops:
        if not isinstance(op, CheapOp):
            continue
        z = producer.get(op.inputs[-1])
        linear = isinstance(z, MatOp) and z.layout == "linear"
        if "requant" in op.params:
            m = int(op.params["requant"]["mult"])
            if linear:
                mults.append(m)
            if linear and not (1 << 14 <= m <= 1 << 29):
                comp = info["meta"][op.name].get("comp") if info else ""
                outside.append(f"{op.name}({comp}, layer {info['meta'][op.name].get('layer') if info else ''}):"
                               f" 2^{math.log2(m):.2f}")
        elif "vec_requant" in op.params:
            vec_ops += 1
            mv = op.fn.__defaults__[0]
            distinct.append(int(torch.unique(mv).numel()))
            mults.extend([int(mv.min()), int(mv.max())])
            bad = int(((mv < (1 << 14)) | (mv > (1 << 29))).sum())
            if bad:
                comp = info["meta"][op.name].get("comp") if info else ""
                outside.append(f"{op.name}({comp}, vector): {bad} rows outside")
    gains = []
    if info:
        gains = [m["G"] for m in info["meta"].values() if m.get("kind") == "norm"]
    return {"max_claim_bound_log2": round(math.log2(worst_claim), 3), "claims_ok": worst_claim < (1 << 29),
            "linear_mult_min_log2": round(math.log2(max(1, min(mults))), 3),
            "linear_mult_max_log2": round(math.log2(max(mults)), 3),
            "linear_mults_outside_P3": outside, "vector_mult_ops": vec_ops,
            "distinct_mults_per_vector_op_max": max(distinct) if distinct else 1,
            "distinct_mults_per_vector_op_mean": (sum(distinct) / len(distinct)) if distinct else 1,
            "norm_gain_min": float(min(float(g.min()) for g in gains)) if gains else None,
            "norm_gain_max": float(max(float(g.max()) for g in gains)) if gains else None}

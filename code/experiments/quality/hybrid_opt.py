"""A floating-point copy of the integer OPT graph (``pvi.models.hf_opt.build_opt_from_hf``) whose components switch
between their exact integer form and float: the method of the attribution study (``attribution.py``,
``tab:quality-attr`` of the extended version).

:class:`Hybrid` runs the graph's own ops in order, at every position.  Each tensor is carried as its real value
(float64) and, when it lies on its integer grid, its integers.  An integer op whose inputs carry integers calls the
graph's own op, so it computes the graph's integers exactly; an integer op fed by a float value applies the same
integer rule to ``real / scale`` (rounding half up, clamping, the table look-ups), so its error is that of the
integer pipeline; a float op computes OPT's real arithmetic in float64 from the real values it receives: the
checkpoint's weights with the LayerNorm gains and biases folded in, a LayerNorm with eps 1e-5, an exact softmax and
ReLU.

Components (:data:`FULL` lists each one's integer features; a configuration maps a component to the set of its
features that are integer, the empty set meaning float and a component it leaves out fully integer):

* ``w_emb``, ``w_qkv``, ``w_out``, ``w_fc1``, ``w_fc2``, ``w_head``: the int8 weights (``q``);
* ``norm1``, ``norm2``, ``normf``: the LayerNorm before attention, before the MLP and before the LM head: ``stats``
  (the integer mean and square root, no eps), ``round`` (to steps of 1/G), ``clip`` (to +-127/G);
* ``rq_q``, ``rq_k``, ``rq_v``, ``rq_fc1``: the requantisations to int8 of q, k, v and fc1's output: ``round``,
  ``clip``;
* ``softmax``: ``exp`` (the exp table in steps of 1/16 nat) and ``pbits`` (the probabilities rounded to steps of
  1/pmax);
* ``att_out``: the requantisation of the attention output to int8: ``round``, ``clip``;
* ``res``: the residual stream: ``round`` (the branches and the embeddings rounded to the residual scale), ``clip``
  (to +-RES_MAX);
* ``relu``: the ReLU table (``lut``; exact on int8 inputs, so it adds no error by itself).

:func:`roles` maps every op of the graph to its component and checks the graph's op pattern; :class:`Hybrid` rebuilds
every weight op's float and dequantised weights from the checkpoint and the builder's real scales
(``info["scales"]``) and lists every weight, bias or scale that does not quantise to the graph's own
(:attr:`Hybrid.problems`).  Graphs with per-row weight scales, cross-layer equalisation or a pruned last block are
refused.
"""
from __future__ import annotations

import math

import torch

from pvi.core.graph import INT8_MAX, MatOp
from pvi.models import hf_opt
from pvi.models.decoder import EXP_TABLE, RES_MAX, SHIFT

__all__ = ["FULL", "GROUPS", "SUBROWS", "Hybrid", "roles", "float_cfg", "int_only_cfg"]

FULL = {
    "w_emb": {"q"}, "w_qkv": {"q"}, "w_out": {"q"}, "w_fc1": {"q"}, "w_fc2": {"q"}, "w_head": {"q"},
    "norm1": {"stats", "round", "clip"}, "norm2": {"stats", "round", "clip"}, "normf": {"stats", "round", "clip"},
    "rq_q": {"round", "clip"}, "rq_k": {"round", "clip"}, "rq_v": {"round", "clip"}, "rq_fc1": {"round", "clip"},
    "softmax": {"exp", "pbits"}, "att_out": {"round", "clip"}, "res": {"round", "clip"}, "relu": {"lut"},
}
"""Every component and its integer features."""
WEIGHTS = ("w_emb", "w_qkv", "w_out", "w_fc1", "w_fc2", "w_head")
NORMS = ("norm1", "norm2", "normf")
REQUANTS = ("rq_q", "rq_k", "rq_v", "rq_fc1")
GROUPS = {
    "weights": ("weights (all matrices)", WEIGHTS),
    "w_emb": ("weights: embeddings", ("w_emb",)),
    "w_qkv": ("weights: q/k/v", ("w_qkv",)),
    "w_out": ("weights: out_proj", ("w_out",)),
    "w_fc1": ("weights: fc1", ("w_fc1",)),
    "w_fc2": ("weights: fc2", ("w_fc2",)),
    "w_head": ("weights: LM head", ("w_head",)),
    "norm": ("LayerNorm outputs (all three)", NORMS),
    "norm1": ("LayerNorm before attention", ("norm1",)),
    "norm2": ("LayerNorm before the MLP", ("norm2",)),
    "normf": ("final LayerNorm (LM head input)", ("normf",)),
    "rq_qkv": ("q/k/v requantisation", ("rq_q", "rq_k", "rq_v")),
    "rq_fc1": ("fc1 requantisation", ("rq_fc1",)),
    "softmax": ("softmax (exp table and rounded probabilities)", ("softmax",)),
    "att_out": ("attention output requantisation", ("att_out",)),
    "res": ("residual stream (rounding and clamp)", ("res",)),
    "relu": ("ReLU table", ("relu",)),
}
"""The study's groups of components, by the key its records name (``group``): its label and its components."""
SUBROWS = {
    "norm_clip": ("LayerNorm outputs: clip only (no rounding)", {c: {"stats", "clip"} for c in NORMS}),
    "norm_round": ("LayerNorm outputs: rounding only (no clip)", {c: {"stats", "round"} for c in NORMS}),
    "norm_stats": ("LayerNorm outputs: integer statistics only", {c: {"stats"} for c in NORMS}),
    "normf_round": ("final LayerNorm: rounding only", {"normf": {"stats", "round"}}),
    "normf_clip": ("final LayerNorm: clip only", {"normf": {"stats", "clip"}}),
    "softmax_exp": ("softmax: exp table only (exact probabilities)", {"softmax": {"exp"}}),
    "softmax_pbits": ("softmax: rounded probabilities only (exact exp)", {"softmax": {"pbits"}}),
    "rq_round": ("activation requantisations: rounding only", {c: {"round"} for c in REQUANTS}),
    "rq_clip": ("activation requantisations: clip only", {c: {"clip"} for c in REQUANTS}),
    "att_out_round": ("attention output: rounding only", {"att_out": {"round"}}),
    "att_out_clip": ("attention output: clip only", {"att_out": {"clip"}}),
    "res_round": ("residual: rounding only", {"res": {"round"}}),
    "res_clip": ("residual: clamp only", {"res": {"clip"}}),
}
"""The feature sub-rows, by key: the label and the integer features of the components they restrict."""
LN_EPS = 1e-5
"""The epsilon of OPT's LayerNorm (the float norm)."""

_KINDS = {"positions/1": "pos", "scale/1": "scale", "residual/1": "res", "norm/1": "norm", "requant/1": "rq",
          "attention/1": "attn", "lut/1": "act", "last_position/1": "last", "identity/1": "last"}
HEAD = (("mat", "w_emb"), ("pos", None), ("mat", "w_emb"), ("scale", None), ("res", "res"))
BLOCK = (("norm", "norm1"), ("mat", "w_qkv"), ("rq", "rq_q"), ("mat", "w_qkv"), ("rq", "rq_k"), ("mat", "w_qkv"),
         ("rq", "rq_v"), ("attn", "softmax"), ("mat", "w_out"), ("res", "res"), ("norm", "norm2"), ("mat", "w_fc1"),
         ("rq", "rq_fc1"), ("act", "relu"), ("mat", "w_fc2"), ("res", "res"))
TAIL = (("norm", "normf"), ("last", None), ("mat", "w_head"))
"""The op pattern of an unpruned integer OPT graph: ``(kind, component)`` of the embeddings, of each block and of the
LM head."""


class Val:
    """A tensor of the hybrid: its real value (float64; None for the token and position ids) and, when it lies on its
    integer grid, its integers."""

    __slots__ = ("real", "ints")

    def __init__(self, real, ints=None):
        self.real = real
        self.ints = ints


def _rhu(x):
    """Round half up (the graph's rounding)."""
    return torch.floor(x + 0.5)


def op_kind(op) -> str | None:
    """An op's kind in :data:`BLOCK`'s terms (``mat``, ``norm``, ``rq``, ...); None for an op the hybrid does not copy
    (a requantisation with a per-row multiplier)."""
    if isinstance(op, MatOp):
        return "mat" if op.layout in ("linear", "embed") else None
    kind = _KINDS.get(getattr(op.fn, "kind", None))
    if kind == "last" and op.note != "last position":
        return None
    if kind in ("rq", "res") and "requant" not in op.params:
        return None
    return kind


def roles(graph, n_layers: int) -> list[tuple[str, str | None, int]]:
    """``(kind, component, layer)`` of every op of ``graph`` (layer -1: the embeddings; ``n_layers``: the LM head);
    ValueError if the graph's ops are not those of an unpruned integer OPT graph of ``n_layers`` blocks with one
    weight scale per matrix (:data:`HEAD`, :data:`BLOCK`, :data:`TAIL`)."""
    want = [*HEAD, *(BLOCK * n_layers), *TAIL]
    layers = [-1] * len(HEAD) + [i for i in range(n_layers) for _ in BLOCK] + [n_layers] * len(TAIL)
    if len(graph.ops) != len(want):
        raise ValueError(f"{len(graph.ops)} ops, where an unpruned OPT graph of {n_layers} blocks has {len(want)}")
    out = []
    for i, (op, (kind, comp), layer) in enumerate(zip(graph.ops, want, layers)):
        got = op_kind(op)
        if got != kind:
            raise ValueError(f"op {i} ({op.name}) is {got}, where the pattern has {kind} ({comp})")
        if kind == "mat" and (op.layout == "embed") != (comp == "w_emb"):
            raise ValueError(f"op {i} ({op.name}): layout {op.layout} for {comp}")
        out.append((kind, comp, layer))
    return out


def float_weights(model) -> list[tuple[str, torch.Tensor, torch.Tensor | None]]:
    """``(component, real weight [rows, cols] in float32, bias in float64 or None)`` of every weight op of
    ``build_opt_from_hf``'s graph, in op order: the builder's own folding of the LayerNorm gains and biases
    (``hf_opt._fold``)."""
    dec, cfg = model.model.decoder, model.config
    out = [("w_emb", dec.embed_tokens.weight.T.float(), None),
           ("w_emb", dec.embed_positions.weight[2:2 + cfg.max_position_embeddings].T.float(), None)]
    for layer in dec.layers:
        at, ln1, ln2 = layer.self_attn, layer.self_attn_layer_norm, layer.final_layer_norm
        for proj in (at.q_proj, at.k_proj, at.v_proj):
            wf, bf = hf_opt._fold(proj.weight.float(), proj.bias, ln1.weight, ln1.bias)
            out.append(("w_qkv", wf.float(), bf))
        out.append(("w_out", at.out_proj.weight.float(), at.out_proj.bias.double()))
        wf, bf = hf_opt._fold(layer.fc1.weight.float(), layer.fc1.bias, ln2.weight, ln2.bias)
        out.append(("w_fc1", wf.float(), bf))
        out.append(("w_fc2", layer.fc2.weight.float(), layer.fc2.bias.double()))
    lnf = dec.final_layer_norm
    wf, bf = hf_opt._fold(model.lm_head.weight.float(), None, lnf.weight, lnf.bias)
    out.append(("w_head", wf.float(), bf))
    return out


def _quantise(w: torch.Tensor, b: torch.Tensor | None, s_in, s_w: float | None = None) -> dict:
    """The builder's quantisation of the real weight ``w`` acting on an input of real scale ``s_in``
    (``hf_opt._RealBuilder.mat``, one scale per matrix): the int8 weights, the int64 bias, the product's real scale
    ``s_z``, and the dequantised weight and bias in the real input's domain."""
    if torch.is_tensor(s_in):                        # a smoothed norm's per-channel scales, folded into the columns
        w_eff = w.double() * s_in.double()[None, :]
        s_z = float(w_eff.abs().max()) / INT8_MAX
        w_int = (w_eff / s_z).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        del w_eff
    else:
        s_w = float(w.abs().max()) / INT8_MAX if s_w is None else s_w
        w_int = (w / s_w).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
        s_z = s_w * s_in
    b_int = None if b is None else (b.double() / s_z).round().to(torch.int64)
    return {"w_int": w_int, "b_int": b_int, "s_z": s_z, "s_w": s_w if not torch.is_tensor(s_in) else None}


class Hybrid:
    """The float copy of ``graph`` (``build_opt_from_hf(model, ...)``, with its ``info``).  :attr:`problems` lists
    every weight op whose rebuilt int8 weights, bias or real scale differ from the graph's (empty: the copy's float
    and dequantised weights are those the graph quantised)."""

    def __init__(self, model, graph, info: dict) -> None:
        if "scales" not in info:
            raise ValueError("the builder's info has no real scales (info['scales'])")
        if info.get("row_pow2") or info.get("cle_mlp") is not None or info.get("prune_last"):
            raise ValueError("the hybrid copies graphs with one weight scale per matrix, without cross-layer "
                             "equalisation and without a pruned last block")
        self.graph, self.s_res, self.last = graph, info["s_residual"], graph.last_use()
        scales = info["scales"]
        self.roles = roles(graph, model.config.num_hidden_layers)
        self.problems: list[str] = []
        self.meta: dict[str, dict] = {}
        weights = iter(float_weights(model))
        for op, (kind, comp, layer) in zip(graph.ops, self.roles):
            m = {"kind": kind, "comp": comp, "layer": layer}
            if kind == "mat":
                wcomp, w, b = next(weights)
                assert wcomp == comp, (wcomp, comp)
                s_in = 1.0 if op.layout == "embed" else scales[op.inputs[0]]
                s_w = info["embed_mult"] * info["s_residual"] if op.inputs[0] == graph.input_name else None
                q = _quantise(w, b, s_in, s_w)
                s_z = scales[op.output]
                if not torch.equal(q["w_int"], op.weight):
                    self.problems.append(f"{op.name} ({comp}, layer {layer}): the int8 weights differ in "
                                         f"{int((q['w_int'] != op.weight).sum())} entries")
                if (q["b_int"] is None) != (op.bias is None) or (op.bias is not None
                                                                  and not torch.equal(q["b_int"], op.bias)):
                    self.problems.append(f"{op.name} ({comp}, layer {layer}): the bias differs")
                if q["s_z"] != s_z:
                    self.problems.append(f"{op.name} ({comp}, layer {layer}): the real scale {q['s_z']!r} is not "
                                         f"the builder's {s_z!r}")
                m.update(s_in=s_in, s_z=s_z, s_w=q["s_w"], w=w, b=b,
                         b_deq=None if op.bias is None else op.bias.double() * s_z)
            elif kind == "res":
                m["mult"] = int(op.params["requant"]["mult"])
            elif kind == "rq":
                m.update(mult=int(op.params["requant"]["mult"]), s_out=scales[op.output])
            elif kind == "norm":
                m["G"] = op.fn.consts["gain"].double() / (1 << 16)
            elif kind == "attn":
                c = op.fn.consts
                if c["heads"] != c["kv_heads"]:
                    raise ValueError(f"{op.name}: grouped-query attention")
                m.update(m_s=c["m_s"], m_o=c["m_o"], pmax=c["pmax"], heads=c["heads"], dh=c["dh"],
                         sq=scales[op.inputs[0]], sk=scales[op.inputs[1]], s_att=scales[op.output])
            elif kind == "scale":
                m["c"] = op.fn.consts["c"]
            self.meta[op.name] = m

    # ---------------------------------------------------------------- the weights in float64
    # Only float32 copies are kept (the float64 copies of OPT-125M's weights alone take 2 GB); each product converts
    # its weight to float64, which is exact, so the products are those of float64 copies.
    @staticmethod
    def _w64(m: dict) -> torch.Tensor:
        """The real weight in float64."""
        return m["w"].double()

    @staticmethod
    def _deq64(op, m: dict) -> torch.Tensor:
        """The dequantised weight in the real input's domain, float64: ``w_int s_w`` (``w_int s_z / s_in`` for a
        smoothed input), rounded to float32 (the arithmetic of the reported attribution)."""
        if "deq" not in m:
            w_int = op.weight.double()
            if m["s_w"] is None:
                deq = w_int * m["s_z"] / m["s_in"].double()[None, :]
            else:
                deq = w_int * m["s_w"]
            m["deq"] = deq.float()
        return m["deq"].double()

    # ---------------------------------------------------------------- the forward pass
    def logits(self, ids: torch.Tensor, cfg: dict) -> torch.Tensor:
        """Real logits ``[B, T, vocab]`` at every position with the integer features ``cfg`` (component -> set;
        a component it leaves out is fully integer).  Each tensor is freed after its last reader."""
        feats = {c: set(cfg.get(c, FULL[c])) for c in FULL}
        env: dict[str, Val] = {self.graph.input_name: Val(None, ids)}
        for i, op in enumerate(self.graph.ops):
            m = self.meta[op.name]
            kind = m["kind"]
            if kind == "mat":
                env[op.output] = self._mat(op, m, env[op.inputs[0]], "q" in feats[m["comp"]])
            elif kind == "pos":
                env[op.output] = Val(None, op.fn(env[op.inputs[0]].ints))
            elif kind == "scale":
                x = env[op.inputs[0]]
                env[op.output] = Val(x.real, None if x.ints is None else x.ints * m["c"])
            elif kind == "last":
                env[op.output] = env[op.inputs[0]]                   # every position
            elif kind == "act":
                x = env[op.inputs[0]]
                if "lut" in feats["relu"] and x.ints is not None:
                    env[op.output] = Val(x.real.clamp_min(0), op.fn(x.ints))   # relu of the integers: the same real
                else:
                    env[op.output] = Val(x.real.clamp_min(0))
            elif kind == "res":
                env[op.output] = self._residual(op, m, env[op.inputs[0]], env[op.inputs[1]], feats["res"])
            elif kind == "norm":
                env[op.output] = self._norm(op, m, env[op.inputs[0]], feats[m["comp"]])
            elif kind == "rq":
                env[op.output] = self._requant(op, m, env[op.inputs[0]], feats[m["comp"]])
            else:
                env[op.output] = self._attn(op, m, *[env[n] for n in op.inputs], feats["softmax"], feats["att_out"])
            for n in op.inputs:
                if self.last[n] <= i:
                    env.pop(n, None)
        return env[self.graph.output_name].real

    def _mat(self, op, m: dict, x: Val, integer: bool) -> Val:
        if op.layout == "embed":
            if integer:
                z = op.fold(op.compute(x.ints), x.ints)
                return Val(z.double() * m["s_z"], z)
            return Val(m["w"].T[x.ints].double())
        if integer and x.ints is not None:
            z = op.fold(op.compute(x.ints), x.ints)
            return Val(z.double() * m["s_z"], z)
        if integer:
            out = x.real @ self._deq64(op, m).T
            return Val(out if m["b_deq"] is None else out + m["b_deq"])
        out = x.real @ self._w64(m).T
        return Val(out if m["b"] is None else out + m["b"])

    def _residual(self, op, m: dict, r: Val, z: Val, f: set) -> Val:
        s_res = self.s_res
        if f == FULL["res"] and r.ints is not None and z.ints is not None:
            y = op.fn(r.ints, z.ints)
            return Val(y.double() * s_res, y)
        if not f:
            return Val(r.real + z.real)
        rr = r.ints.double() if r.ints is not None else r.real / s_res
        dz = z.ints.double() * m["mult"] / (1 << SHIFT) if z.ints is not None else z.real / s_res
        if "round" in f:
            rr, dz = _rhu(rr), _rhu(dz)
        out = rr + dz
        if "clip" in f:
            out = out.clamp(-RES_MAX, RES_MAX)
        return Val(out * s_res, out.to(torch.int64) if "round" in f else None)

    def _norm(self, op, m: dict, r: Val, f: set) -> Val:
        G = m["G"]                                          # [d], the exact gain g / 2^16
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
                # the exact integer square root (a float root can be one off near a perfect square)
                sigma = torch.where((sigma + 1) ** 2 <= var.double(), sigma + 1, sigma)
                sigma = torch.where(sigma ** 2 > var.double(), sigma - 1, sigma).clamp_min(1)
                n = xc.double() / sigma
            else:
                x = r.real / self.s_res
                mean = torch.floor((x.sum(-1, keepdim=True) + d // 2) / d)
                xc = x - mean
                sigma = torch.floor((xc * xc).sum(-1, keepdim=True) / d).sqrt().floor().clamp_min(1)
                n = xc / sigma
        else:
            h = r.real if r.ints is None else r.ints.double() * self.s_res
            hc = h - h.mean(-1, keepdim=True)
            n = hc / torch.sqrt((hc * hc).mean(-1, keepdim=True) + LN_EPS)
        y = n * G
        if "round" in f:
            y = _rhu(y)
        if "clip" in f:
            y = y.clamp(-INT8_MAX, INT8_MAX)
        return Val(y / G, y.to(torch.int64) if ("round" in f and "clip" in f) else None)

    def _requant(self, op, m: dict, z: Val, f: set) -> Val:
        s_out = m["s_out"]
        if f == FULL["rq_q"] and z.ints is not None:
            y = op.fn(z.ints)
            return Val(y.double() * s_out, y)
        if not f:
            return Val(z.real)
        u = z.ints.double() * m["mult"] / (1 << SHIFT) if z.ints is not None else z.real / s_out
        if "round" in f:
            u = _rhu(u)
        if "clip" in f:
            u = u.clamp(-INT8_MAX, INT8_MAX)
        return Val(u * s_out, u.to(torch.int64) if ("round" in f and "clip" in f) else None)

    def _attn(self, op, m: dict, q: Val, k: Val, v: Val, fs: set, fo: set) -> Val:
        if fs == FULL["softmax"] and fo == FULL["att_out"] and q.ints is not None and k.ints is not None \
                and v.ints is not None:
            y = op.fn(q.ints, k.ints, v.ints)
            return Val(y.double() * m["s_att"], y)
        H, dh = m["heads"], m["dh"]
        b, t, _ = v.real.shape

        def heads(a):
            return a.reshape(b, t, H, dh).transpose(1, 2)

        mask = ~torch.ones(t, t, dtype=torch.bool).tril()
        raw_ints = None
        if not fs:
            sc = heads(q.real) @ heads(k.real).transpose(-1, -2) / math.sqrt(dh)
            prob = torch.softmax(sc.masked_fill(mask, -math.inf), -1)
        else:
            qi = q.ints.double() if q.ints is not None else q.real / m["sq"]
            ki = k.ints.double() if k.ints is not None else k.real / m["sk"]
            s = (heads(qi) @ heads(ki).transpose(-1, -2)).masked_fill(mask, -math.inf)
            gap = s.amax(-1, keepdim=True) - s
            if "exp" in fs:
                tab = EXP_TABLE.double()
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
                    raw_ints = p @ heads(v.ints.double())            # exact (sums below 2^53)
            else:
                prob = e / tot
        raw_real = prob @ heads(v.real)                             # [B, H, T, dh]
        ints = None
        if not fo:
            out = raw_real
        else:
            u = raw_ints * m["m_o"] / (1 << SHIFT) if raw_ints is not None else raw_real / m["s_att"]
            if "round" in fo:
                u = _rhu(u)
            if "clip" in fo:
                u = u.clamp(-INT8_MAX, INT8_MAX)
            out = u * m["s_att"]
            ints = u if ("round" in fo and "clip" in fo) else None
        out = out.transpose(1, 2).reshape(b, t, H * dh)
        if ints is not None:
            ints = ints.transpose(1, 2).reshape(b, t, H * dh).to(torch.int64)
        return Val(out, ints)


def float_cfg(*comps) -> dict:
    """Every component integer except ``comps`` (float)."""
    return {c: set() for c in comps}


def int_only_cfg(*comps) -> dict:
    """Every component float except ``comps`` (integer)."""
    return {c: (set(FULL[c]) if c in comps else set()) for c in FULL}

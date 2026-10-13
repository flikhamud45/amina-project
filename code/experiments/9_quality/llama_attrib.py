"""F2b-Q attribution: every intermediate tensor of the integer Llama/Qwen3 graph against the fp32 checkpoint's.

    cd code && PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --blocks 0
    cd code && PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base \\
        --blocks all --mode both --recipe smooth55 --prefix --out attrib.jsonl
    cd code && PYTHONPATH=src python experiments/9_quality/llama_attrib.py --model Qwen/Qwen3-0.6B-Base --smoke \\
        --blocks all --threads 2                                       # a tiny random model (laptop)

``llama_diag.py`` says after which block the integer residual stream leaves the checkpoint's; this script says at
which op. For one WikiText-2 test window (calibration on train windows, as ``llama_diag.py``), one recipe
(``scalar4``: norm gain 4; ``smooth55``: SmoothQuant alpha 0.55, the 99.9 / 99.99 percentiles), and the chosen
decoder layers (``--blocks``: HF's ``model.layers[i]``, 0-based; layer ``i``'s output is ``llama_diag.py``'s block
``i + 1``), every tensor of the block (``ROLES``, ``real_llama.ROLES`` with the raw attention output) is dequantised
with its real scale from the builder (``info["scales"]``; the head-equalisation factors ``info["head_beta"]``, the
q/k balance and a smoothed norm's per-channel scales included) and compared, in the checkpoint's units, three ways:

* ``vs_hf``: the graph's tensor against the checkpoint's activation (``HF_SOURCE``: forward hooks on the HF modules;
  RoPE, which no module computes, is HF's own ``apply_rotary_pos_emb`` on the hooked q/k with the hooked cos/sin);
* ``local``: the graph's tensor against the exact op (float64) applied to the graph's own dequantised inputs -- the
  error this op adds (``FLOAT_OP``);
* ``carried``: that exact op on the graph's inputs against the checkpoint -- the error its inputs bring.

Per comparison: the cosine over channels per position (mean, minimum and its position, mean over positions 1..),
the relative L2 error ``|a - b| / |b|`` (which also sees a wrong scale, unlike the cosine) and, for tensors with
heads, the worst head's mean cosine. Per tensor also: the fraction of int8 entries at +-127 and at 0, of residual
entries at +-RES_MAX, a claim's largest |value| (log2; int32 needs < 31) and its honest bound, and the real scale
(a float, or the range of a per-channel vector).

**The attention pattern** (``pattern_stats``): a cosine of q or k is dominated by its largest channels, which need
not be the ones the scores weigh (Qwen3's k norm has gains up to 96.5 on channels where q's gain is about 0), so the
scores ``q k / sqrt(head_dim)`` of the graph's q and k are compared with the checkpoint's: the RMS of the row-centred
score difference in nats (against the reference's own spread) and the total variation of the softmax rows. Per
stage of the q path (``in``, the block's input; ``attn_norm``, ``q_proj``, ``q``, ``q_norm``, ``q_rope``) the stage's
tensor is carried to RoPE's output by the exact ops and scored against the checkpoint's k (and the k path against
its q), so the increase from one stage to the next is the score error that op adds, and ``in``'s the error the
block's input brings (``score_steps``).

``--mode accumulated`` runs the graph on the window as is (errors accumulate from the embedding on);
``--mode teacher`` feeds each block the checkpoint's residual at its input, quantised on the graph's residual scale
(``round(h / s_res)``, clamped at RES_MAX), so each block's error is measured alone; ``both`` does the two from one
build. With ``--blocks i`` the graph is built with ``i + 1`` layers only: its first ``i + 1`` blocks are those of the
full graph, integer for integer (every scale of a block is calibrated on the blocks before it; ``s_res`` comes from
the checkpoint's full forward). Per block the output names the first op (in op order) whose own error exceeds
``--flag-rel`` (relative L2), whose mean cosine falls below ``--flag-cos``, or which adds more than ``--flag-nats``
to the scores' error (``first_flag``), and the op of the largest own error and of the largest score step.
``--no-qk-balance`` builds Qwen3's graph without the q/k balance (``real_llama``'s ``qk_balance``, on by default).

Self-checks: the role map (``check_roles``: every role is the output of an op of the expected kind and note, in op
order; a block's ops read nothing from outside it but its input) and, per block, the raw attention output
recomputed with the op's own constants requantises to the op's output, integer for integer (else the script
stops).

``--prefix [SPEC]``: as ``llama_diag.py`` (the reference runs on ``[prefix | window]``, its prefix positions
dropped; the attention's float op attends to the graph's int8 prefix constants, dequantised). Output: one JSON line
per (mode, block) on stdout (and appended to ``--out``), the table on stderr.
"""
from __future__ import annotations

import argparse
import importlib
import inspect
import json
import math
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

from pvi.fullcheck import real_llama as rl
from pvi.fullcheck.graph import INT8_MAX, CheapOp, MatOp, requant
from pvi.fullcheck.protocol import claim_bound
from pvi.fullcheck.transformer import PROB_MAX, RES_MAX, SHIFT, _attention

ROLES = tuple(r for r, *_ in rl.ROLES[:11]) + ("attn_raw",) + tuple(r for r, *_ in rl.ROLES[11:])
"""The compared tensors in op order: ``real_llama.ROLES`` and ``attn_raw`` (the attention's raw ``P V``, before its
requantisation; recomputed from the op's constants)."""
KIND = {r: k for r, _, k in rl.ROLES} | {"attn_raw": "raw"}
HEADS = {"q_proj": "q", "q": "q", "q_norm": "q", "q_rope": "q", "k_proj": "kv", "k": "kv", "k_norm": "kv",
         "k_rope": "kv", "v_proj": "kv", "v": "kv", "attn_raw": "q", "attn": "q"}
HF_SOURCE = {
    "in": "layer input (forward pre-hook of model.layers[i])",
    "attn_norm": "input_layernorm output",
    "q_proj": "self_attn.q_proj output", "q": "self_attn.q_proj output",
    "k_proj": "self_attn.k_proj output", "k": "self_attn.k_proj output",
    "v_proj": "self_attn.v_proj output", "v": "self_attn.v_proj output",
    "q_norm": "self_attn.q_norm output", "k_norm": "self_attn.k_norm output",
    "q_rope": "apply_rotary_pos_emb(q_norm output, or q_proj's for Llama; the hooked cos, sin)",
    "k_rope": "apply_rotary_pos_emb(k_norm output, or k_proj's for Llama; the hooked cos, sin)",
    "attn_raw": "self_attn.o_proj input", "attn": "self_attn.o_proj input",
    "o_proj": "self_attn.o_proj output",
    "res_attn": "post_attention_layernorm input",
    "mlp_norm": "post_attention_layernorm output",
    "gate_proj": "mlp.gate_proj output", "gate": "mlp.gate_proj output",
    "up_proj": "mlp.up_proj output", "up": "mlp.up_proj output",
    "silu": "mlp.act_fn output", "glu": "mlp.down_proj input", "down_proj": "mlp.down_proj output",
    "res_mlp": "layer output (forward hook of model.layers[i])"}
"""Each role's checkpoint activation. The graph's tensor is brought to the checkpoint's units first: times its real
scale; a norm's output times the norm's gamma (the builder folds gamma into the next matrices); q's and k's claims
and int8 values divided by the head-equalisation factor; a ``glu_exp`` product and its down projection times the
token's ``2^e``."""
FLOAT_OP = {
    "attn_norm": "RMSNorm(in) with input_layernorm's gamma and epsilon",
    "q_proj": "q_proj(attn_norm)", "q": "q_proj (identity: the requantisation's own error)",
    "k_proj": "k_proj(attn_norm)", "k": "k_proj", "v_proj": "v_proj(attn_norm)", "v": "v_proj",
    "q_norm": "per-head RMSNorm(q) with q_norm's gamma and epsilon", "k_norm": "the same for k",
    "q_rope": "apply_rotary_pos_emb(q_norm, or q)", "k_rope": "apply_rotary_pos_emb(k_norm, or k)",
    "attn_raw": "softmax(q_rope k_rope^T / sqrt(head_dim)) v, causal, grouped (with the prefix's constants)",
    "attn": "attn_raw (identity)", "o_proj": "o_proj(attn)", "res_attn": "in + o_proj",
    "mlp_norm": "RMSNorm(res_attn) with post_attention_layernorm's gamma and epsilon",
    "gate_proj": "gate_proj(mlp_norm)", "gate": "gate_proj", "up_proj": "up_proj(mlp_norm)", "up": "up_proj",
    "silu": "silu(gate)", "glu": "silu * up", "down_proj": "down_proj(glu)", "res_mlp": "res_attn + down_proj"}
"""Each role's exact float64 op on the graph's own dequantised inputs (the ``local`` reference)."""
RECIPES = {"scalar4": dict(norm_gain=4.0),
           "smooth55": dict(norm_gain=4.0, smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)}
"""``llama_diag.py``'s recipes."""


# ------------------------------------------------------------------ the checkpoint's activations

def hf_activations(model, ids: torch.Tensor, blocks, prefix=None) -> dict[int, dict[str, torch.Tensor]]:
    """Per decoder layer in ``blocks``: the checkpoint's activation of every role (``HF_SOURCE``) on ``ids [B, T]``
    (float64 on the CPU, ``[B, T, C]``), and ``in`` (the layer's input) and ``cos``, ``sin`` (``[B, T, head_dim]``,
    the rotary tables HF passes to the attention). With ``prefix`` (token ids) the checkpoint runs on
    ``[prefix | ids]`` and the prefix's positions are dropped, but for the prefix's keys and values
    (``k_rope_prefix``, ``v_proj_prefix``, ``[B, P, Hkv head_dim]``), which the queries attend to."""
    want = sorted(set(blocks))
    pre = None if prefix is None or not len(prefix) else torch.as_tensor(prefix, dtype=torch.int64).reshape(1, -1)
    n = 0 if pre is None else pre.shape[1]
    out: dict[int, dict[str, torch.Tensor]] = {b: {} for b in want}
    layers = list(model.model.layers)
    hooks = []

    def keep(store, key, t):
        t = t[0] if isinstance(t, tuple) else t
        t = t.detach()
        if t.dim() == 4:                                   # q_norm / k_norm: [B, T, heads, head_dim]
            t = t.reshape(*t.shape[:2], -1)
        if n and key in ("k_proj", "k_norm", "v_proj"):    # the prefix's keys and values (the reference's attention)
            store[key + "_prefix"] = t[:, :n].to("cpu", torch.float64)
        store[key] = t[:, n:].to("cpu", torch.float64)

    def fwd(store, key):
        return lambda mod, args, o: keep(store, key, o)

    def pre_in(store, key):
        return lambda mod, args, kwargs: keep(store, key, args[0] if args else kwargs["hidden_states"])

    def rotary(store):
        def hook(mod, args, kwargs):
            pe = kwargs.get("position_embeddings")
            pe = pe if pe is not None else args[1]
            for key, t in zip(("cos", "sin"), pe):
                store[key + "_all"] = t.detach().to("cpu", torch.float64)
                store[key] = store[key + "_all"][:, n:]
        return hook

    for b in want:
        layer, s = layers[b], out[b]
        at, mlp = layer.self_attn, layer.mlp
        hooks += [layer.register_forward_pre_hook(pre_in(s, "in"), with_kwargs=True),
                  layer.register_forward_hook(fwd(s, "res_mlp")),
                  layer.input_layernorm.register_forward_hook(fwd(s, "attn_norm")),
                  at.register_forward_pre_hook(rotary(s), with_kwargs=True),
                  at.q_proj.register_forward_hook(fwd(s, "q_proj")),
                  at.k_proj.register_forward_hook(fwd(s, "k_proj")),
                  at.v_proj.register_forward_hook(fwd(s, "v_proj")),
                  at.o_proj.register_forward_pre_hook(pre_in(s, "attn"), with_kwargs=True),
                  at.o_proj.register_forward_hook(fwd(s, "o_proj")),
                  layer.post_attention_layernorm.register_forward_pre_hook(pre_in(s, "res_attn"), with_kwargs=True),
                  layer.post_attention_layernorm.register_forward_hook(fwd(s, "mlp_norm")),
                  mlp.gate_proj.register_forward_hook(fwd(s, "gate_proj")),
                  mlp.up_proj.register_forward_hook(fwd(s, "up_proj")),
                  mlp.act_fn.register_forward_hook(fwd(s, "silu")),
                  mlp.down_proj.register_forward_pre_hook(pre_in(s, "glu"), with_kwargs=True),
                  mlp.down_proj.register_forward_hook(fwd(s, "down_proj"))]
        if hasattr(at, "q_norm"):
            hooks += [at.q_norm.register_forward_hook(fwd(s, "q_norm")),
                      at.k_norm.register_forward_hook(fwd(s, "k_norm"))]
    joint = ids if pre is None else torch.cat([pre.to(ids.device).expand(ids.shape[0], -1), ids], 1)
    try:
        with torch.no_grad():
            model.model(joint.to(model.device))
    finally:
        for h in hooks:
            h.remove()
    cfg = model.config
    hq, hkv = cfg.num_attention_heads, cfg.num_key_value_heads or cfg.num_attention_heads
    dh = getattr(cfg, "head_dim", None) or cfg.hidden_size // hq
    for b in want:
        s = out[b]
        for r in ("q", "k", "v", "gate", "up"):
            s[r] = s[r + "_proj"]
        s["attn_raw"] = s["attn"]
        rope = _hf_rope_fn(layers[b])
        src = ("q_norm", "k_norm") if "q_norm" in s else ("q_proj", "k_proj")
        s["q_rope"] = rope(s[src[0]], s["cos"], s["sin"], hq, dh)
        s["k_rope"] = rope(s[src[1]], s["cos"], s["sin"], hkv, dh)
        if n:                                              # the prefix's keys at the positions 0 .. P-1
            s["k_rope_prefix"] = rope(s[src[1] + "_prefix"], s["cos_all"][:, :n], s["sin_all"][:, :n], hkv, dh)
    return out


def _hf_rope_fn(layer):
    """``x [B, T, H dh] -> `` HF's ``apply_rotary_pos_emb`` of ``x`` (the attention module's own function, in
    float64), with ``cos``, ``sin [B, T', dh]`` of which the last ``T`` positions are ``x``'s."""
    apply = importlib.import_module(type(layer.self_attn).__module__).apply_rotary_pos_emb

    def rope(x, cos, sin, heads, dh):
        b, t, _ = x.shape
        xh = x.reshape(b, t, heads, dh).transpose(1, 2)
        got, _ = apply(xh, xh, cos[:, -t:].to(x.dtype), sin[:, -t:].to(x.dtype))
        return got.transpose(1, 2).reshape(b, t, heads * dh)
    return rope


# ------------------------------------------------------------------ the exact float ops

def _w(p) -> torch.Tensor:
    """A module parameter as float64 on the CPU."""
    return p.detach().to("cpu", torch.float64)


def _rms(x: torch.Tensor, norm) -> torch.Tensor:
    """HF's RMSNorm (``x / sqrt(mean x^2 + eps) gamma``) in float64."""
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon) * _w(norm.weight)


def _head_rms(x: torch.Tensor, norm, heads: int, dh: int) -> torch.Tensor:
    return _rms(x.reshape(*x.shape[:-1], heads, dh), norm).reshape(x.shape)


def _linear(x: torch.Tensor, lin) -> torch.Tensor:
    return F.linear(x, _w(lin.weight), None if lin.bias is None else _w(lin.bias))


def float_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, hq: int, hkv: int, dh: int) -> torch.Tensor:
    """Exact causal grouped-query attention in float64: ``q [B, Tq, hq dh]`` are the queries of the last ``Tq`` of
    the ``Tk`` positions of ``k, v [B, Tk, hkv dh]`` (``transformer._attention``'s convention); query head ``j``
    reads key/value head ``j // (hq / hkv)``; scores ``q k / sqrt(dh)`` (HF's ``scaling``)."""
    b, tq, tk = q.shape[0], q.shape[1], k.shape[1]
    qh = q.reshape(b, tq, hq, dh).transpose(1, 2)
    kh = k.reshape(b, tk, hkv, dh).transpose(1, 2).repeat_interleave(hq // hkv, 1)
    vh = v.reshape(b, tk, hkv, dh).transpose(1, 2).repeat_interleave(hq // hkv, 1)
    s = qh @ kh.transpose(-1, -2) / math.sqrt(dh)
    s = s.masked_fill(~torch.ones(tq, tk, dtype=torch.bool).tril(tk - tq), float("-inf"))
    return (torch.softmax(s, -1) @ vh).transpose(1, 2).reshape(b, tq, hq * dh)


# ------------------------------------------------------------------ the graph's tensors

def _defaults(fn) -> dict:
    return dict(fn.consts)


def block_ops(graph, info: dict, b: int) -> tuple[int, int]:
    """``(lo, hi)``: block ``b``'s ops are ``graph.ops[lo:hi]``, from the one after its input's producer through the
    producer of ``res_mlp``; every input they read is computed among them, or is the block's input or the token ids
    (a pruned q's RoPE reads their length)."""
    at = {op.output: i for i, op in enumerate(graph.ops)}
    roles = info["roles"][b]
    lo, hi = at[roles["in"]] + 1, at[roles["res_mlp"]] + 1
    made = {roles["in"], graph.input_name} | {op.output for op in graph.ops[lo:hi]}
    stray = [n for op in graph.ops[lo:hi] for n in op.inputs if n not in made]
    if stray:
        raise ValueError(f"block {b} reads tensors from outside it: {stray}")
    return lo, hi


def check_roles(graph, info: dict) -> list[str]:
    """The map's consistency: per block, the roles of ``real_llama.ROLES`` (no q/k norm without them), each the
    output of an op of the expected kind and note, in op order. Returns the problems (empty when it holds)."""
    by_out = {op.output: (i, op) for i, op in enumerate(graph.ops)}
    problems = []
    for b, roles in enumerate(info["roles"]):
        last = by_out[roles["in"]][0] if roles["in"] in by_out else -1
        for role, note, _ in rl.ROLES:
            if role not in roles:
                if role not in ("q_norm", "k_norm"):
                    problems.append(f"block {b}: no {role}")
                continue
            i, op = by_out[roles[role]]
            if note is None and not isinstance(op, MatOp):
                problems.append(f"block {b} {role}: {op.name} is no weight op")
            elif note is not None and not (isinstance(op, CheapOp) and op.note.startswith(note)):
                problems.append(f"block {b} {role}: {op.name} is '{getattr(op, 'note', '')}', not '{note}'")
            if i <= last:
                problems.append(f"block {b} {role}: out of op order")
            last = i
    return problems


def _run(graph, env: dict, lo: int, hi: int, keep: set, last: dict) -> None:
    """Run ``graph.ops[lo:hi]`` on ``env``, dropping each tensor after its last reader unless it is in ``keep``."""
    for i in range(lo, hi):
        op = graph.ops[i]
        if isinstance(op, MatOp):
            xin = env[op.inputs[0]]
            env[op.output] = op.fold(op.compute(xin), xin)
        else:
            env[op.output] = op.fn(*[env[n] for n in op.inputs])
        for n in op.inputs:
            if last.get(n, -1) <= i and n not in keep:
                env.pop(n, None)


def _attn_raw(graph, roles: dict, env: dict) -> torch.Tensor:
    """The attention op's raw ``P V`` (int64), recomputed with the op's own constants (``m_o = None``); its
    requantisation must be the op's output (checked)."""
    op = next(o for o in graph.ops if o.output == roles["attn"])
    d = _defaults(op.fn)
    q, k, v = (env[n] for n in op.inputs)
    p = d.get("pmax", PROB_MAX)
    if "k_prefix" in d:
        raw = rl.prefix_attention(q, k, v, d["k_prefix"], d["v_prefix"], d["m_s"], None, d["heads"], d["kv_heads"],
                                  d["dh"], p)
    else:
        raw = _attention(q, k, v, d["m_s"], None, d["heads"], d["kv_heads"], d["dh"], p)
    again = requant(raw, torch.tensor(d["m_o"], dtype=torch.int64), SHIFT, -INT8_MAX, INT8_MAX)
    if not torch.equal(again, env[roles["attn"]]):
        raise AssertionError("the recomputed attention does not requantise to the op's output")
    return raw


def _glu_factor(graph, roles: dict, env: dict):
    """``2^e_t [B, T, 1]`` of a ``glu_exp`` product (the per-token exponent ``real_llama._exponents``), else 1."""
    op = next(o for o in graph.ops if o.output == roles["glu"])
    d = _defaults(op.fn)
    if "emax" not in d:
        return 1.0
    a, u = (env[n].to(torch.int64) for n in op.inputs)
    return torch.pow(2.0, rl._exponents(a * u * d["mult"], d["emax"]).double().cpu())


def _real(t: torch.Tensor, scale) -> torch.Tensor:
    s = scale.double().cpu() if torch.is_tensor(scale) else scale
    return t.to("cpu", torch.float64) * s


# ------------------------------------------------------------------ statistics

def compare(a: torch.Tensor, b: torch.Tensor, heads: int | None = None) -> dict:
    """``a`` against the reference ``b`` (``[B, T, C]``, float64; ``b`` cut to ``a``'s last positions): the cosine
    over channels per position -- mean, minimum, the minimum's position, mean over positions 1.. -- and the
    relative L2 error ``|a - b| / |b|``; with ``heads``, the worst head's mean cosine and its index."""
    b = b[:, -a.shape[1]:]
    t, c = a.shape[1], a.shape[-1]
    cos = F.cosine_similarity(a.reshape(-1, c), b.reshape(-1, c), dim=-1).reshape(a.shape[0], t)
    i = int(cos.reshape(-1).argmin())
    out = {"cos_mean": float(cos.mean()), "cos_min": float(cos.min()), "cos_min_pos": i % t,
           "cos_rest": float(cos[:, 1:].mean()) if t > 1 else float(cos.mean()),
           "rel": float((a - b).norm() / b.norm().clamp_min(1e-300))}
    if heads:
        hc = F.cosine_similarity(a.reshape(-1, heads, c // heads), b.reshape(-1, heads, c // heads), dim=-1).mean(0)
        out["head_cos_min"], out["head_worst"] = float(hc.min()), int(hc.argmin())
    return out


def pattern_stats(qg: torch.Tensor, kg: torch.Tensor, qh: torch.Tensor, kh: torch.Tensor, hq: int, hkv: int,
                  dh: int) -> dict:
    """The attention pattern of the graph's ``q, k`` (``qg [B, Tq, hq dh]``, ``kg [B, Tk, hkv dh]``, real, RoPE
    applied; the prefix's keys in front) against the checkpoint's (``qh``, ``kh``), per head (query head ``j`` on
    key/value head ``j // (hq / hkv)``), over the causal entries: the RMS of the row-centred score difference
    (nats; a constant per row does not change the softmax) against the RMS spread of the checkpoint's row-centred
    scores, and the total variation ``1/2 sum |p - p'|`` of the softmax rows (mean, maximum, the worst head's mean).
    It sees what a per-tensor cosine of q or k can hide: errors in the channels the scores weigh most."""
    b, tq, tk = qg.shape[0], qg.shape[1], kg.shape[1]
    qh = qh[:, -tq:]                          # (a pruned block's q: the last position)
    rep = hq // hkv
    mask = torch.ones(tq, tk, dtype=torch.bool).tril(tk - tq)
    nvalid = mask.sum(-1, keepdim=True).double()
    err2 = spread2 = 0.0
    tv_head, tv_max = [], 0.0
    for j in range(hq):
        kv = j // rep
        s = []
        for q, k in ((qg, kg), (qh, kh)):
            qj = q.reshape(b, tq, hq, dh)[:, :, j]
            kj = k[:, -tk:].reshape(b, tk, hkv, dh)[:, :, kv]
            s.append((qj @ kj.transpose(-1, -2) / math.sqrt(dh)).masked_fill(~mask, float("-inf")))
        cen = [torch.where(mask, x - (x.masked_fill(~mask, 0).sum(-1, keepdim=True) / nvalid), 0.0) for x in s]
        err2 += float(((cen[0] - cen[1]) ** 2).sum())
        spread2 += float((cen[1] ** 2).sum())
        tv = 0.5 * (torch.softmax(s[0], -1) - torch.softmax(s[1], -1)).abs().sum(-1)
        tv_head.append(float(tv.mean()))
        tv_max = max(tv_max, float(tv.max()))
    n = float(mask.sum()) * b * hq
    return {"score_err_nats": math.sqrt(err2 / n), "score_spread_nats": math.sqrt(spread2 / n),
            "tv_mean": sum(tv_head) / hq, "tv_max": tv_max, "head_tv_max": max(tv_head),
            "head_worst": int(torch.tensor(tv_head).argmax())}


def _scale_summary(s):
    if torch.is_tensor(s):
        return [float(s.min()), float(s.max())]
    return float(s)


def _int_stats(kind: str, t: torch.Tensor, op) -> dict:
    t = t.to(torch.int64)
    if kind == "int8":
        return {"clip": float((t.abs() >= INT8_MAX).double().mean()), "zero": float((t == 0).double().mean())}
    if kind == "residual":
        return {"clip": float((t.abs() >= RES_MAX).double().mean())}
    if kind == "claim":
        top = int(t.abs().max())
        return {"claim_max_log2": round(math.log2(max(top, 1)), 2),
                "claim_bound_log2": round(math.log2(max(claim_bound(op), 1)), 2)}
    return {}


def attribute_block(graph, info: dict, model, b: int, env: dict, hf: dict) -> dict:
    """``{"rows", "pattern"}`` of block ``b``: per role (``ROLES``) the graph's tensor in ``env`` (from a run of the
    block's ops), dequantised into the checkpoint's units, against ``hf`` (:func:`hf_activations`' block ``b``) --
    ``vs_hf`` --, and against the exact op on the graph's own inputs (``FLOAT_OP``) -- ``local`` --, which against
    ``hf`` is ``carried``, with the integer statistics (:func:`_int_stats`); and the attention pattern of the
    graph's q and k against the checkpoint's (:func:`pattern_stats`)."""
    roles, sc = info["roles"][b], info["scales"]
    layer = list(model.model.layers)[b]
    at, mlp = layer.self_attn, layer.mlp
    cfg = info["config"]
    hq, hkv, dh = cfg.n_heads, cfg.n_kv_heads, cfg.head_dim
    beta_q, beta_k = info["head_beta"][b] or (1.0, 1.0)
    op_of = {o.output: o for o in graph.ops}
    glu_f = _glu_factor(graph, roles, env)
    raw = _attn_raw(graph, roles, env)
    g = {"in": _real(env[roles["in"]], sc[roles["in"]])}           # the graph's tensors in the checkpoint's units
    g["attn_norm"] = _real(env[roles["attn_norm"]], sc[roles["attn_norm"]]) * _w(layer.input_layernorm.weight)
    for r, beta in (("q_proj", beta_q), ("q", beta_q), ("k_proj", beta_k), ("k", beta_k)):
        g[r] = _real(env[roles[r]], sc[roles[r]]) / beta
    for r in ("v_proj", "v", "q_norm", "k_norm", "q_rope", "k_rope", "attn", "o_proj", "res_attn", "gate_proj",
              "gate", "up_proj", "up", "silu"):
        if r in roles:
            g[r] = _real(env[roles[r]], sc[roles[r]])
    s_raw = sc[roles["v"]] / float(_defaults(op_of[roles["attn"]].fn).get("pmax", PROB_MAX))     # s_v / P
    g["attn_raw"] = _real(raw, s_raw)
    g["mlp_norm"] = _real(env[roles["mlp_norm"]], sc[roles["mlp_norm"]]) * _w(layer.post_attention_layernorm.weight)
    g["glu"] = _real(env[roles["glu"]], sc[roles["glu"]]) * glu_f
    g["down_proj"] = _real(env[roles["down_proj"]], sc[roles["down_proj"]]) * glu_f
    g["res_mlp"] = _real(env[roles["res_mlp"]], sc[roles["res_mlp"]])

    def last(t, like):
        return t[:, -like.shape[1]:]

    qk = "q_norm" in roles
    rope = _hf_rope_fn(layer)
    f = {"attn_norm": _rms(g["in"], layer.input_layernorm)}           # the exact ops on the graph's inputs
    f["q_proj"], f["k_proj"], f["v_proj"] = (_linear(last(g["attn_norm"], g[r]), p) for r, p in
                                             (("q_proj", at.q_proj), ("k_proj", at.k_proj), ("v_proj", at.v_proj)))
    f["q"], f["k"], f["v"] = g["q_proj"], g["k_proj"], g["v_proj"]
    if qk:
        f["q_norm"] = _head_rms(g["q"], at.q_norm, hq, dh)
        f["k_norm"] = _head_rms(g["k"], at.k_norm, hkv, dh)
    f["q_rope"] = rope(g["q_norm" if qk else "q"], hf["cos"], hf["sin"], hq, dh)
    f["k_rope"] = rope(g["k_norm" if qk else "k"], hf["cos"], hf["sin"], hkv, dh)
    kk, vv = g["k_rope"], g["v"]
    if info.get("prefix_kv"):
        (kp, vp), (sk, sv) = info["prefix_kv"][b], info["prefix_scales"][b]
        kk = torch.cat([_real(kp, sk).expand(kk.shape[0], -1, -1), kk], 1)
        vv = torch.cat([_real(vp, sv).expand(vv.shape[0], -1, -1), vv], 1)
    f["attn_raw"] = float_attention(g["q_rope"], kk, vv, hq, hkv, dh)
    f["attn"] = g["attn_raw"]
    f["o_proj"] = _linear(g["attn"], at.o_proj)
    f["res_attn"] = last(g["in"], g["o_proj"]) + g["o_proj"]
    f["mlp_norm"] = _rms(g["res_attn"], layer.post_attention_layernorm)
    f["gate_proj"], f["up_proj"] = _linear(g["mlp_norm"], mlp.gate_proj), _linear(g["mlp_norm"], mlp.up_proj)
    f["gate"], f["up"] = g["gate_proj"], g["up_proj"]
    f["silu"] = F.silu(g["gate"])
    f["glu"] = g["silu"] * g["up"]
    f["down_proj"] = _linear(g["glu"], mlp.down_proj)
    f["res_mlp"] = g["res_attn"] + g["down_proj"]

    rows = {}
    for r in ROLES:
        if r not in g:
            continue
        heads = {"q": hq, "kv": hkv}.get(HEADS.get(r))
        name = roles["attn"] if r == "attn_raw" else roles[r]
        kind = KIND[r]
        t = raw if r == "attn_raw" else env[name]
        row = {"tensor": name, "kind": kind, "scale": _scale_summary(s_raw if r == "attn_raw" else sc[name]),
               "vs_hf": compare(g[r], hf[r], heads), "local": compare(g[r], f[r], heads),
               "carried": compare(f[r], hf[r], heads), **_int_stats(kind, t, op_of.get(name))}
        rows[r] = row
    # the attention pattern, and per stage of the q and k paths that stage's tensor carried to RoPE's output by the
    # exact ops (against the checkpoint's other side): the score error each op adds
    kh = hf["k_rope"] if "k_rope_prefix" not in hf else torch.cat([hf["k_rope_prefix"].expand(
        hf["k_rope"].shape[0], -1, -1), hf["k_rope"]], 1)
    pre_g = None if kk.shape[1] == g["k_rope"].shape[1] else kk[:, :kk.shape[1] - g["k_rope"].shape[1]]
    pre_h = None if kh.shape[1] == hf["k_rope"].shape[1] else kh[:, :kh.shape[1] - hf["k_rope"].shape[1]]

    def path(side):
        """The q (k) path's stages and, per stage, the exact op that follows it (the requantisation: identity)."""
        heads, proj, norm = (hq, at.q_proj, "q_norm") if side == "q" else (hkv, at.k_proj, "k_norm")
        steps = [("in", lambda x: _rms(x, layer.input_layernorm)),
                 ("attn_norm", lambda x: _linear(last(x, g[side]), proj)), (side + "_proj", lambda x: x)]
        if qk:
            steps += [(side, lambda x: _head_rms(x, getattr(at, norm), heads, dh)),
                      (side + "_norm", lambda x: rope(x, hf["cos"], hf["sin"], heads, dh))]
        else:
            steps += [(side, lambda x: rope(x, hf["cos"], hf["sin"], heads, dh))]
        return steps + [(side + "_rope", None)]

    def carry(stage, side):
        """The stage's tensor carried to RoPE's output by the exact ops after it."""
        steps = path(side)
        i = [s for s, _ in steps].index(stage)
        x = g[stage]
        for _, fn in steps[i:-1]:
            x = fn(x)
        return x

    def with_prefix(k, pre):
        return k if pre is None else torch.cat([pre, k], 1)

    pattern = {"qk": pattern_stats(g["q_rope"], kk, hf["q_rope"], kh, hq, hkv, dh), "q": {}, "k": {}}
    for side in ("q", "k"):
        for stage, _ in path(side):
            x = carry(stage, side)
            if side == "q":
                pattern["q"][stage] = pattern_stats(x, kh, hf["q_rope"], kh, hq, hkv, dh)
            else:                     # the graph's prefix constants enter as the attention's keys: at k_rope
                x = with_prefix(x, pre_g if stage == "k_rope" else pre_h)
                pattern["k"][stage] = pattern_stats(hf["q_rope"], x, hf["q_rope"], kh, hq, hkv, dh)
    return {"rows": rows, "pattern": pattern}


def score_steps(pattern: dict) -> dict:
    """Per stage of the q and k paths (``pattern["q"]``, ``pattern["k"]``): the score error (nats) it adds to the
    stage before it (the first stage, ``in``: the error the block's input brings); ``in`` and ``attn_norm``, on both
    paths, take the larger of their two."""
    out = {}
    for side in ("q", "k"):
        prev = 0.0
        for stage, p in pattern[side].items():
            out[stage] = max(out.get(stage, 0.0), p["score_err_nats"] - prev)
            prev = p["score_err_nats"]
    return out


def first_flag(res: dict, rel: float, cos: float, nats: float) -> str | None:
    """The first role (op order) whose own error (``local``) exceeds ``rel`` or whose mean cosine is below ``cos``,
    or, on the q and k paths, that adds more than ``nats`` to the attention scores' error (:func:`score_steps`)."""
    rows, steps = res["rows"], score_steps(res["pattern"])
    for r in ROLES:
        if r in rows and (rows[r]["local"]["rel"] > rel or rows[r]["local"]["cos_mean"] < cos
                          or steps.get(r, 0.0) > nats):
            return r
    return None


def worst(res: dict) -> dict:
    """The role of the largest own error (``local`` relative L2) and the q/k stage of the largest score step."""
    rows, steps = res["rows"], score_steps(res["pattern"])
    r = max(rows, key=lambda k: rows[k]["local"]["rel"])
    s = max(steps, key=steps.get)
    return {"local": r, "local_rel": rows[r]["local"]["rel"], "score_step": s, "score_step_nats": steps[s]}


def attribute(graph, info: dict, model, ids: torch.Tensor, blocks, mode: str, prefix=None,
              hf: dict | None = None) -> list[dict]:
    """Per block of ``blocks`` (decoder layers, 0-based): :func:`attribute_block`'s rows, ``mode`` ``accumulated``
    (the graph on ``ids`` as is) or ``teacher`` (the block's ops on the checkpoint's residual at its input,
    ``clamp(round(h / s_res), +-RES_MAX)``). ``hf``: :func:`hf_activations` of these blocks (computed if None)."""
    blocks = sorted(set(blocks))
    hf = hf_activations(model, ids, blocks, prefix) if hf is None else hf
    last = graph.last_use()
    out = []
    if mode == "accumulated":
        env, pos = {graph.input_name: ids}, 0
        for b in blocks:
            lo, hi = block_ops(graph, info, b)
            keep = set(info["roles"][b].values())
            _run(graph, env, pos, hi, keep, last)
            pos = hi
            out.append({"block": b, "mode": mode, **attribute_block(graph, info, model, b, env, hf[b])})
            for n in keep:
                if last.get(n, -1) < hi:
                    env.pop(n, None)
        return out
    if mode != "teacher":
        raise ValueError(f"mode {mode!r}")
    s_res = info["s_residual"]
    for b in blocks:
        lo, hi = block_ops(graph, info, b)
        roles = info["roles"][b]
        forced = (hf[b]["in"] / s_res).round().clamp(-RES_MAX, RES_MAX).to(torch.int64).to(ids.device)
        env = {graph.input_name: ids, roles["in"]: forced}
        _run(graph, env, lo, hi, set(roles.values()), last)
        out.append({"block": b, "mode": mode, **attribute_block(graph, info, model, b, env, hf[b])})
        del env
    return out


# ------------------------------------------------------------------ output

def _line(tag: str, r: str, row: dict) -> str:
    v, lo, ca = row["vs_hf"], row["local"], row["carried"]
    extra = []
    if "clip" in row:
        extra.append(f"clip {100 * row['clip']:.2f}%")
    if "zero" in row:
        extra.append(f"zero {100 * row['zero']:.1f}%")
    if "claim_max_log2" in row:
        extra.append(f"claim 2^{row['claim_max_log2']:.1f} (bound 2^{row['claim_bound_log2']:.1f})")
    if "head_cos_min" in lo:
        extra.append(f"worst head {lo['head_worst']} local {lo['head_cos_min']:.3f}")
    s = row["scale"]
    s = f"{s:.3g}" if isinstance(s, float) else f"{s[0]:.3g}..{s[1]:.3g}"
    return (f"  {tag} {r:<9} vs hf cos {v['cos_mean']:.4f} min {v['cos_min']:+.3f} rel {v['rel']:.3f} | local cos "
            f"{lo['cos_mean']:.4f} min {lo['cos_min']:+.3f} rel {lo['rel']:.3f} | carried rel {ca['rel']:.3f} | "
            + " ".join(extra) + f" | scale {s}")


def _pattern_lines(tag: str, pattern: dict) -> list[str]:
    p = pattern["qk"]
    out = [f"  {tag} pattern   score error {p['score_err_nats']:.3f} nats (reference spread "
           f"{p['score_spread_nats']:.2f}) | softmax TV mean {p['tv_mean']:.4f} max {p['tv_max']:.3f} | worst head "
           f"{p['head_worst']} TV {p['head_tv_max']:.4f}"]
    for side in ("q", "k"):
        chain = ", ".join(f"{s} {v['score_err_nats']:.3f} (TV {v['tv_mean']:.3f})" for s, v in pattern[side].items())
        out.append(f"  {tag} {side} path    score error (nats) carried from: {chain}")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B-Base")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--window", type=int, default=0, help="which test window")
    ap.add_argument("--calib-windows", type=int, default=4)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--recipe", default="smooth55", choices=sorted(RECIPES))
    ap.add_argument("--blocks", default="0", help="decoder layers, 0-based: 'all', or comma-separated")
    ap.add_argument("--mode", default="both", choices=["accumulated", "teacher", "both"])
    ap.add_argument("--no-head-eq", action="store_true")
    ap.add_argument("--no-qk-balance", action="store_true", help="Qwen3: no q/k balance per RoPE pair (the old graph)")
    ap.add_argument("--flag-rel", type=float, default=0.1)
    ap.add_argument("--flag-cos", type=float, default=0.99)
    ap.add_argument("--flag-nats", type=float, default=0.1)
    ap.add_argument("--smoke", action="store_true", help="a tiny random model of the family (laptop check)")
    ap.add_argument("--prefix", nargs="?", const="auto", default=None,
                    help="a fixed public prefix: auto (PrefixQuant's rule; a bare --prefix), bos, or ids 'a,b,..'")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import ppl_llama as pl                                  # noqa: E402 (data and checkpoints: the script only)

    model, tok = pl.smoke_model(args.model) if args.smoke else pl.load_llama(args.model)
    test = pl.token_ids(tok, pl.p2._text("test")[:400_000], args.prefix)
    train = pl.token_ids(tok, pl.p2._text("train")[:400_000], args.prefix)
    w = test[args.window * args.seq:(args.window + 1) * args.seq].unsqueeze(0)
    calib = torch.stack([train[i * args.seq:(i + 1) * args.seq] for i in range(args.calib_windows)])
    n_all = model.config.num_hidden_layers
    blocks = list(range(n_all)) if args.blocks == "all" else [int(s) for s in args.blocks.split(",")]
    with torch.no_grad():
        prefix, choice = pl.prefix_of(args.prefix, model, calib, tok)
        kw = dict(RECIPES[args.recipe], head_eq=not args.no_head_eq, qk_balance=not args.no_qk_balance,
                  prefix_tokens=prefix, n_layers=max(blocks) + 1)
        graph, info = rl.build_llama_from_hf(model, calib, **kw)
        problems = check_roles(graph, info)
        if problems:
            raise SystemExit(f"the role map does not hold: {problems}")
        hf = hf_activations(model, w, blocks, prefix)
        qk = info["arch"] == "qwen3"
        head = {"model": args.model, "smoke": args.smoke, "recipe": args.recipe, "seq": args.seq,
                "window": args.window, "prefix_tokens": prefix, "prefix_choice": choice,
                "head_eq": info["head_eq"], "qk_balance": info["qk_balance"], "s_residual": info["s_residual"],
                "n_layers_built": max(blocks) + 1}
        modes = ["accumulated", "teacher"] if args.mode == "both" else [args.mode]
        for mode in modes:
            for res in attribute(graph, info, model, w, blocks, mode, prefix, hf):
                flag, top = first_flag(res, args.flag_rel, args.flag_cos, args.flag_nats), worst(res)
                rec = dict(head, mode=mode, block=res["block"], first_flag=flag, worst=top, pattern=res["pattern"],
                           rows=res["rows"])
                line = json.dumps(rec)
                print(line, flush=True)
                if args.out:
                    with open(args.out, "a", encoding="utf-8") as fh:
                        fh.write(line + "\n")
                tag = (f"{args.recipe}{'+prefix' if prefix else ''}{'' if info['qk_balance'] or not qk else '-nobal'} "
                       f"{mode[:5]} L{res['block']:02d}")
                for r, row in res["rows"].items():
                    print(_line(tag, r, row), file=sys.stderr, flush=True)
                for s in _pattern_lines(tag, res["pattern"]):
                    print(s, file=sys.stderr, flush=True)
                out = res["rows"]["res_mlp"]["vs_hf"]
                print(f"  {tag} => block output vs hf cos {out['cos_mean']:.4f} (min {out['cos_min']:+.3f}); first op "
                      f"over local rel {args.flag_rel} / below cos {args.flag_cos} / adding {args.flag_nats} nats to "
                      f"the scores: {flag}; largest own error {top['local']} (rel {top['local_rel']:.3f}), largest "
                      f"score step {top['score_step']} ({top['score_step_nats']:.3f} nats)", file=sys.stderr,
                      flush=True)


if __name__ == "__main__":
    main()

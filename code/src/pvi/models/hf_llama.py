"""Integer RMSNorm / RoPE / SwiGLU / GQA decoders built from *real* pretrained weights: the Llama-family and Qwen3
models of Sec. ``sec:quality``.

:func:`pvi.models.hf_opt.build_opt_from_hf` does this for OPT; this module does it for the Llama family (Hugging
Face ``LlamaForCausalLM``: TinyLlama, OpenLLaMA, Llama-2) and for Qwen3 (``Qwen3ForCausalLM``:
Qwen3-0.6B/1.7B/4B-Base). The graph is :func:`pvi.models.decoder.build_decoder`'s for the same shape -- the same op
kinds, in the same order, with the same names, wiring and matrix shapes, so the cost tables measured on random weights
apply to it -- and only the *constants* change (``matches_benchmark_graph`` checks it; the diff is empty, see below):

* **Weights** are the real ones, quantised per tensor to int8 (one scale per matrix, ``s_w = max|W| / 127``).
* **RMSNorm gain gamma is folded** into the matrices the norm feeds (``W' = W diag(gamma)``); RMSNorm has no shift,
  so no bias appears. The proven norm op stays parameter-free -- integer mean square, integer square root, int8
  output ``G x (normalised value)`` -- with ``G`` a public constant: one scalar ``norm_gain``, or, with ``smooth``,
  SmoothQuant through the gain vector exactly as for OPT (channel ``j`` gets ``G_j = c / s_j``,
  ``s_j = a_j^alpha / w_j^(1-alpha)``; the next matrices take ``1/G_j`` into column ``j``).
* **Qwen3's q_norm / k_norm** (an RMSNorm over each head's ``head_dim`` values, weight ``gamma [head_dim]`` shared by
  the heads) are the benchmark's "q norm" / "k norm" ops with the gain vector ``round(G gamma_j 2^16)``: gamma cannot
  be folded into a matrix, because RoPE mixes channel ``j`` with channel ``j + head_dim/2`` after it, so it lives in
  the per-head gain, and ``G`` (one public scalar per op) maps the calibration range of ``gamma_j x_j / rms`` to 127.
  The norm is invariant to scaling a head's input, so each head's rows of ``W_q`` (``W_k``) are first multiplied by
  ``beta_h`` (``head_eq``), which brings every head's int8 values to the same range: the integer square root's error
  ``1/rms`` is then small for every head (a head of int8 RMS 10 would otherwise see its scores scaled by up to 10%).
  The float function is unchanged (up to RMSNorm's epsilon). The real gains are far from 1 and unlike between q and
  k (Qwen3-0.6B's layer-0 k norm: 96.5 on a channel where q's is 0.003), so ``qk_balance`` (default) gives each RoPE
  pair a factor ``c_p`` in q's gains and ``1/c_p`` in k's (:meth:`_Builder.qk_balance`): the scores are unchanged in
  real numbers, and k's int8 range is no longer set by a channel q ignores.
* **RoPE** uses the graph's Q14 tables (``pvi.models.decoder._rope``). Hugging Face's ``rotate_half`` gives
  ``[x1 cos - x2 sin | x2 cos + x1 sin]`` with the frequencies ``theta^(-2i/head_dim)`` -- the convention of
  ``_rope`` (``x [cos | cos] + [x2 | x1] [-sin | sin]``; ``tests/test_real_llama.py`` compares them on random
  values). The rotation preserves each pair's norm, so the scale before RoPE is set from the pairs' norms, which
  then never clip at 127.
* **GQA**: query head ``j`` reads key/value head ``j // (n_heads / n_kv)``, Hugging Face's ``repeat_kv``
  (``repeat_interleave``), which is :func:`pvi.models.decoder.integer_attention`'s grouping.
* **SwiGLU**: the SiLU table is built from the gate's real scale ``s_g`` (entry ``x`` is
  ``round(silu(x s_g) / s_a)``, ``s_a = max_x |silu(x s_g)| / 127`` over the table's 256 inputs, which is
  ``silu(127 s_g) / 127`` unless ``s_g`` is tiny), and the product ``act x up`` is requantised
  (:func:`pvi.core.graph.requant_mul`) with the multiplier ``round(2^30 s_a s_u / s_p)`` from real scales.
* **Requantisation multipliers come from true scales** (as in ``build_opt_from_hf``), and the residual stream's
  scale is set so the fp checkpoint's hidden states never clip at ``RES_MAX`` on the calibration windows.
* **Heads**: tied (Qwen3) and untied (Llama) LM heads are both a separate weight op ``W_head diag(gamma_final)``, as
  in the benchmark graph (which has its own LM head op whatever ``tied`` says). The final RMSNorm leaves no bias, so,
  unlike OPT, the LM head has the benchmark's row length.
* ``prune_last`` builds the last block at the last position only (but for its keys and values), calibrated on every
  position (``calibrate_all``), so its logits are the unpruned graph's last-position logits, integer for integer.

Evaluation: :func:`lean_logits` runs the graph with the lean forward (``free=True``, no claim kept) up to the LM
head and applies the head to a few positions at a time: the logits of a 151,936-entry vocabulary at 2,048 positions
(2.5 GB as int64) are never materialised, and every chunk is the same integers (a weight op's columns are
independent). :func:`real_logits` concatenates them; :func:`residual_states` and :func:`hf_residual_states` give the
residual stream at every block boundary for the per-block cosine against the checkpoint.

Activations stay per-tensor int8 with static scales, as proved. Qwen and Llama carry massive activations at the
down projection's input; ``experiments/quality/ppl_llama.py`` measures whether static per-tensor scales survive
them (``pct_glu`` is the product's clipping percentile). The opt-in ``glu_exp`` (an outlier handling outside the
main recipe) gives each token of that product its own power-of-two exponent instead, with the matching shift in
the next residual add; it is the one option that changes the graph's structure (two ops per block).

The fixed prefix (``prefix_tokens``, opt-in in the builder and part of the main recipe, :data:`MAIN_RECIPE`; after
PrefixQuant, Chen et al. 2024, arXiv:2410.05265) keeps the structure: the massive activations of these models sit at
the initial token (the attention sink) and at the first occurrences of a few delimiter tokens. A fixed public prefix
of ``P`` tokens is run once through the fp checkpoint; its keys and values per layer (Hugging Face's KV cache: after
the k norm and RoPE at the positions ``0 .. P-1``) become int8 constants on the scales of that layer's computed k and
v, default arguments of the attention op (:func:`prefix_attention` prepends them; every query attends to the whole
prefix), so the graph digest binds them. The query's RoPE positions start at ``P``. Every scale is calibrated on the
query positions only (the fp residual maximum behind ``s_res`` too), so the prefix's outliers never set one. The
weight ops, their shapes and claim columns are the benchmark graph's (the structural diff stays empty); the prefix
adds no claim.
:func:`prefixquant_tokens` is PrefixQuant's rule for choosing the prefix.
"""

from __future__ import annotations

import collections
import dataclasses
import math

import torch

from pvi.core.graph import INT8_MAX, CheapOp, IntGraph, MatOp, identity_op, mul_requant_op, op_fn, op_kind, requant_fn
from pvi.models.decoder import (CERTIFIED_PROB_MAX, CONFIGS, PROB_MAX, RES_MAX, SHIFT, DecoderConfig, integer_attention,
                                _isqrt, attention_op, build_decoder, head_norm_op, last_position_op, lut_op, norm_op,
                                rope_op, rope_tail_op, scale_op)

__all__ = ["build_llama_from_hf", "config_of", "same_shape", "matches_benchmark_graph", "lean_logits",
           "real_logits", "residual_states", "hf_residual_states", "block_cosines", "swiglu_exp", "residual_exp",
           "prefix_attention", "prefix_kv", "prefixquant_tokens", "ROLES"]

_ROWS_NORM = 1 << 14
"""Calibration rows read by the smoothing statistics (evenly spaced over all windows)."""
_ENTRIES = 1 << 24
"""Entries a percentile reads (``torch.quantile``'s limit): whole rows, evenly spaced over all windows."""
ROLES = (("attn_norm", "norm", "int8"), ("q_proj", None, "claim"), ("q", "requant", "int8"),
         ("k_proj", None, "claim"), ("k", "requant", "int8"), ("v_proj", None, "claim"), ("v", "requant", "int8"),
         ("q_norm", "q norm", "int8"), ("k_norm", "k norm", "int8"), ("q_rope", "rope", "int8"),
         ("k_rope", "rope", "int8"), ("attn", "attention", "int8"), ("o_proj", None, "claim"),
         ("res_attn", "residual add", "residual"), ("mlp_norm", "norm", "int8"), ("gate_proj", None, "claim"),
         ("gate", "requant", "int8"), ("up_proj", None, "claim"), ("up", "requant", "int8"), ("silu", "silu", "int8"),
         ("glu", "swiglu", "int8"), ("down_proj", None, "claim"), ("res_mlp", "residual add", "residual"))
"""A block's tensors in op order (``info["roles"]``): role, the note of the cheap op that computes it (None: a weight
op's claims), and what it holds (int8 values, int64 claims, the int32 residual stream). Llama has no ``q_norm`` /
``k_norm``; with ``glu_exp`` the notes of ``glu`` and ``res_mlp`` carry the per-token suffix."""


def _rows(t: torch.Tensor, cap: int) -> torch.Tensor:
    """``t [R, C]`` itself, or ``cap`` of its rows evenly spaced from the first to the last (every calibration
    window contributes; deterministic)."""
    if t.shape[0] <= cap:
        return t
    return t[torch.linspace(0, t.shape[0] - 1, cap, dtype=torch.float64).round().to(torch.int64).to(t.device)]


def _top(stat: torch.Tensor, pct: float) -> float:
    """The ``pct`` percentile of the non-negative ``stat [R, C]`` (whole rows, evenly spaced, at most ``2^24``
    entries), or its maximum (every entry) for ``pct >= 100``."""
    if pct >= 100:
        return float(stat.max())
    sub = _rows(stat, max(1, _ENTRIES // max(1, stat.shape[1])))
    return float(torch.quantile(sub.reshape(-1)[:_ENTRIES], pct / 100.0))


def _exponents(y: torch.Tensor, emax: int) -> torch.Tensor:
    """Per row of ``y [..., F]`` (int64): the smallest ``e`` in ``0 .. emax`` with every round-half-up
    ``(y + 2^(29+e)) >> (30+e)`` within ``+-127`` (``emax`` where none is), as ``[..., 1]``. A larger shift never
    gives a larger magnitude, so the condition is monotone in ``e``; only the row maximum and minimum are shifted."""
    hi, lo = y.amax(-1, keepdim=True), y.amin(-1, keepdim=True)
    e = torch.full_like(hi, emax)
    for k in range(emax - 1, -1, -1):
        sh = SHIFT + k
        ok = (((hi + (1 << (sh - 1))) >> sh) <= INT8_MAX) & (((lo + (1 << (sh - 1))) >> sh) >= -INT8_MAX)
        e = torch.where(ok, torch.full_like(e, k), e)
    return e


def swiglu_exp(a: torch.Tensor, b: torch.Tensor, mult: int, emax: int) -> torch.Tensor:
    """SwiGLU's product with a per-token power-of-two exponent (``glu_exp``): ``y = a b mult``, and token ``t``'s
    int8 values ``clamp((y + 2^(29+e_t)) >> (30+e_t), +-127)`` with ``e_t = _exponents(y)`` (its real scale is
    ``s_p 2^e_t``). Exact integer arithmetic on any device."""
    y = a.to(torch.int64) * b.to(torch.int64) * mult
    sh = _exponents(y, emax) + SHIFT
    return ((y + (torch.ones_like(sh) << (sh - 1))) >> sh).clamp_(-INT8_MAX, INT8_MAX)


def residual_exp(r: torch.Tensor, z: torch.Tensor, a: torch.Tensor, b: torch.Tensor, glu_mult: int, mult: int,
                 emax: int) -> torch.Tensor:
    """The residual add after a :func:`swiglu_exp` product's down projection: token ``t``'s claims ``z`` carry the
    product's ``2^e_t``, recomputed here from the same ``a, b`` (the verifier holds them), so the add is
    ``clamp(r + ((z mult + 2^(29-e_t)) >> (30-e_t)), +-RES_MAX)`` -- the round-half-up of ``z mult 2^e_t / 2^30``."""
    sh = SHIFT - _exponents(a.to(torch.int64) * b.to(torch.int64) * glu_mult, emax)
    out = (z.to(torch.int64) * mult + (torch.ones_like(sh) << (sh - 1))) >> sh
    return out.add_(r).clamp_(-RES_MAX, RES_MAX)


def prefix_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, kp: torch.Tensor, vp: torch.Tensor, m_s: int,
                     m_o: int | None, n_heads: int, n_kv: int, dh: int, pmax: int = PROB_MAX) -> torch.Tensor:
    """:func:`pvi.models.decoder.integer_attention` with a fixed prefix: the public int8 keys and values
    ``kp, vp [1, P, Hkv dh]`` of the positions ``0 .. P-1`` in front of the query's ``k, v [B, T, Hkv dh]`` (positions
    ``P .. P+T-1``). ``integer_attention`` takes ``q``'s ``Tq`` rows as the queries of the last ``Tq`` of the ``P + T``
    positions, so every query attends to the whole prefix and causally to the query's own positions. Exact integers
    on any device (the constants are copied to ``k``'s device and type)."""
    b = k.shape[0]
    k = torch.cat([kp.to(device=k.device, dtype=k.dtype).expand(b, -1, -1), k], 1)
    v = torch.cat([vp.to(device=v.device, dtype=v.dtype).expand(b, -1, -1), v], 1)
    return integer_attention(q, k, v, m_s, m_o, n_heads, n_kv, dh, pmax)


@op_kind("prefix_attention/1")
def prefix_attention_op(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, *, k_prefix: torch.Tensor,
                        v_prefix: torch.Tensor, m_s: int, m_o: int, heads: int, kv_heads: int, dh: int,
                        pmax: int) -> torch.Tensor:
    """Causal attention behind a fixed public prefix (:func:`prefix_attention`); the prefix's int8 keys and values
    are constants of the op, so the digest binds them."""
    return prefix_attention(q, k, v, k_prefix, v_prefix, m_s, m_o, heads, kv_heads, dh, pmax)


@op_kind("swiglu_exp/1")
def swiglu_exp_op(a: torch.Tensor, b: torch.Tensor, *, mult: int, emax: int) -> torch.Tensor:
    """SwiGLU's product with a per-token exponent (:func:`swiglu_exp`)."""
    return swiglu_exp(a, b, mult, emax)


@op_kind("residual_exp/1")
def residual_exp_op(r: torch.Tensor, z: torch.Tensor, a: torch.Tensor, b: torch.Tensor, *, glu_mult: int, mult: int,
                    emax: int) -> torch.Tensor:
    """The residual add after a per-token-exponent product's down projection (:func:`residual_exp`)."""
    return residual_exp(r, z, a, b, glu_mult, mult, emax)


def _to_int8(real: torch.Tensor, scale: float) -> torch.Tensor:
    """``clamp(round(real / scale), +-127)`` as int8 (a prefix's keys or values on their layer's scale)."""
    return (real.double() / scale).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)


def _pair_norms(real: torch.Tensor, dh: int) -> torch.Tensor:
    """``[..., H*dh]`` -> ``[rows, dh/2]``: the norm of each RoPE pair ``(x_i, x_{i+dh/2})`` of each head, the
    largest magnitude either value reaches under any rotation."""
    x = real.reshape(-1, dh)
    return torch.hypot(x[:, :dh // 2], x[:, dh // 2:])


class _Builder:
    """Like ``pvi.models.decoder._Builder``, but every tensor carries its real scale
    (``pvi.models.hf_opt._RealBuilder``'s arithmetic for the weight ops, requantisations and residual adds, extended
    to RMSNorm, q/k norm, RoPE and SwiGLU)."""

    prob_max = PROB_MAX     # the attention probabilities' scale (build_llama_from_hf(prob_max=...))
    prefix_len = 0          # P of a fixed prefix (build_llama_from_hf(prefix_tokens=...)): the query starts at P

    def __init__(self, tokens: torch.Tensor, *, pct: float, norm_gain: float, smooth: float | None,
                 smooth_pct: float, pct_att: float, pct_glu: float, glu_exp: int = 0) -> None:
        self.pct, self.norm_gain, self.smooth, self.smooth_pct = pct, norm_gain, smooth, smooth_pct
        self.pct_att, self.pct_glu, self.glu_exp = pct_att, pct_glu, glu_exp
        self.ops: list = []
        self.env = {"x": tokens}
        self.scale: dict[str, float | torch.Tensor] = {}
        self.chan: dict[str, torch.Tensor] = {}    # a balanced q/k (qk_balance): real scale = scale x chan per channel
        self.exp_of: dict[str, tuple] = {}         # a glu_exp product -> (act, up, multiplier, emax)
        self.prefix_consts: list[dict] = []        # per layer with a prefix: its int8 k, v, their scales and clipping
        self._n = 0

    def name(self, p):
        self._n += 1
        return f"{p}{self._n}"

    def _run(self, op, scale):
        self.ops.append(op)
        if isinstance(op, MatOp):
            xin = self.env[op.inputs[0]]
            self.env[op.output] = op.fold(op.compute(xin), xin)
        else:
            self.env[op.output] = op.fn(*[self.env[n] for n in op.inputs])
        self.scale[op.output] = scale
        return op.output

    def release(self, keep) -> None:
        """Drop every calibration tensor but ``keep`` (their scales stay)."""
        for n in [n for n in self.env if n not in keep]:
            del self.env[n]

    # -- weight ops
    def mat(self, inp, w: torch.Tensor, b: torch.Tensor | None, s_in, layout="linear", s_w: float | None = None):
        """Real ``w [rows, cols]`` -> int8; the product's real scale is ``s_w * s_in``. A per-channel input scale
        ``s_in`` (a smoothed norm's, a vector) is folded into ``w``'s columns first, then one scale per matrix."""
        n = self.name("mm")
        if torch.is_tensor(s_in):
            w_eff = w.double() * s_in.double().cpu()[None, :]           # acts on the integer input
            s_z = float(w_eff.abs().max()) / INT8_MAX
            w_int = (w_eff / s_z).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
            del w_eff
        else:
            s_w = float(w.abs().max()) / INT8_MAX if s_w is None else s_w
            w_int = (w.double() / s_w).round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)
            s_z = s_w * s_in
        b_int = None if b is None else (b.double() / s_z).round().to(torch.int64)
        return self._run(MatOp(n, (inp,), n + ".z", weight=w_int, bias=b_int, layout=layout), s_z)

    # -- cheap ops
    def cheap(self, prefix, inputs, fn, scale, note="", params=None):
        n = self.name(prefix)
        return self._run(CheapOp(n, tuple(inputs), n + ".y", fn=fn, note=note, params=params or {}), scale)

    def requant(self, prefix, inputs, scale, note, **params):
        """A requantising consumer of a weight op (:func:`pvi.core.graph.requant_fn`, so V1 can cut it)."""
        return self.cheap(prefix, inputs, requant_fn(params), scale, note, {"requant": params})

    def last(self, name, *, calibrate_all=False):
        """``pvi.models.decoder._Builder.last``: the last position of ``name``; with ``calibrate_all`` the calibration
        keeps every position (the op itself takes the last), so the ops after it get the multipliers of the graph
        without pruning."""
        if not calibrate_all:
            return self.cheap("last", [name], op_fn(last_position_op), self.scale[name], "last position")
        n = self.name("last")
        self.ops.append(CheapOp(n, (name,), n + ".y", fn=op_fn(last_position_op), note="last position"))
        self.env[n + ".y"] = self.env[name]
        self.scale[n + ".y"] = self.scale[name]
        return n + ".y"

    def to_int8(self, z, *, pairs: int | None = None):
        """Requantise ``z`` to int8 at the scale mapping the ``pct`` percentile of ``|real value|`` to 127; with
        ``pairs = head_dim`` (RoPE follows) the percentile is of the pairs' norms."""
        real = self.env[z].double() * self.scale[z]
        stat = _pair_norms(real, pairs) if pairs else real.abs().reshape(-1, real.shape[-1])
        del real
        s_out = max(_top(stat, self.pct), 1e-12) / INT8_MAX
        m = round((1 << SHIFT) * self.scale[z] / s_out)
        return self.requant("rq", [z], s_out, "requant", kind="requant", mult=m, shift=SHIFT, lo=-INT8_MAX,
                            hi=INT8_MAX)

    def residual(self, r, z):
        """r + z, with z brought to the residual's real scale."""
        m = round((1 << SHIFT) * self.scale[z] / self.scale[r])
        return self.requant("res", [r, z], self.scale[r], "residual add", kind="residual", mult=m, shift=SHIFT,
                            res_max=RES_MAX)

    def norm(self, r, d, consumers):
        """Integer RMSNorm (``_norm_int(.., center=False)``) with a scalar gain ``norm_gain`` (real scale
        ``1/G``), or with ``smooth`` the SmoothQuant gain vector of ``build_opt_from_hf`` (``consumers``: the
        gamma-folded matrices this norm feeds)."""
        if self.smooth is None:
            gain = torch.full((d,), round(self.norm_gain * (1 << 16)), dtype=torch.int64)
            return self.cheap("norm", [r], op_fn(norm_op, gain=gain, center=False), 1.0 / self.norm_gain, "norm")
        x = self.env[r]
        sigma = _isqrt(torch.div((x * x).sum(-1, keepdim=True), d, rounding_mode="floor"))
        n = _rows((x.double() / sigma.double()).reshape(-1, d), _ROWS_NORM).abs().cpu()
        a = (torch.quantile(n, self.smooth_pct / 100.0, dim=0) if self.smooth_pct < 100
             else n.amax(0)).clamp_min(1e-3)
        wj = torch.stack([w.double().abs().amax(0) for w in consumers]).amax(0).clamp_min(1e-8)
        s_ = a.pow(self.smooth) / wj.pow(1 - self.smooth)
        g_real = ((INT8_MAX / float((a / s_).max())) / s_).clamp(1.0 / 64, float(1 << 14))
        gain = torch.round(g_real * (1 << 16)).to(torch.int64)
        return self.cheap("norm", [r], op_fn(norm_op, gain=gain, center=False), (1 << 16) / gain.double(), "norm")

    def _head_normed(self, z, dh: int, gamma: torch.Tensor) -> torch.Tensor:
        """``gamma_j x_j / rms`` per head of the int8 ``z`` (float64 ``[rows, dh]``, the integer norm's ``rms``)."""
        x = self.env[z].reshape(-1, dh)
        sigma = _isqrt(torch.div((x * x).sum(-1, keepdim=True), dh, rounding_mode="floor"))
        return x.double() / sigma.double() * gamma.double().to(x.device)[None, :]

    def head_norm(self, z, heads: int, dh: int, gamma: torch.Tensor, prefix: str, note: str,
                  c: torch.Tensor | None = None):
        """Qwen3's q/k norm: the benchmark's per-head integer RMSNorm with the gain ``round(G gamma_j c_j 2^16)``
        (``c = 1`` but with ``qk_balance``: :meth:`qk_balance`); ``G`` maps the ``pct`` percentile of the RoPE pairs'
        norms of ``gamma_j c_j x_j / rms`` to 127, so the real scale of channel ``j`` is ``1 / (G c_j)``: ``1/G``
        in ``self.scale`` and ``1/c_j`` (per head) in ``self.chan``. The input's scale does not matter: the norm
        divides it out."""
        normed = self._head_normed(z, dh, gamma)
        if c is not None:
            normed *= c.to(normed.device)[None, :]
        G = INT8_MAX / max(_top(_pair_norms(normed, dh), self.pct), 1e-12)
        del normed
        g_real = gamma.double().cpu() * (1.0 if c is None else c.double().cpu())
        gain = torch.round(G * g_real * (1 << 16)).to(torch.int64)
        out = self.cheap(prefix, [z], op_fn(head_norm_op, gain=gain, heads=heads, dh=dh), 1.0 / G, note)
        if c is not None:
            self.chan[out] = (1.0 / c.double().cpu()).repeat(heads)
        return out

    def qk_balance(self, q, k, dh: int, gamma_q: torch.Tensor, gamma_k: torch.Tensor) -> torch.Tensor:
        """The per-RoPE-pair factor ``c [dh]`` (``c_j = c_{j + dh/2}``) of ``qk_balance``: q's channel ``j`` takes
        ``c_j`` into its norm's gain and k's channel ``j`` takes ``1/c_j``, with ``c_p = sqrt(A_k,p / A_q,p)``,
        ``A_p`` the ``pct`` percentile of pair ``p``'s norm of ``gamma_j x_j / rms`` over the calibration rows (all
        heads), clamped to ``[2^-8, 2^8]``. RoPE rotates each pair as a whole, so the factor commutes with it, and
        every score ``sum_j q_j k_j`` is unchanged in real numbers (the product of the two scales is ``1 / (G_q
        G_k)`` in every channel, which ``m_s`` uses). Both then span ``sqrt(A_q,p A_k,p)`` per pair, instead of k's
        int8 range being set by a channel whose q gain is near zero (Qwen3-0.6B's layer-0 k norm has gains 96.5,
        38.3 and 26.3 against a median of 2.3, on channels where q's gain is 0.00-0.18)."""
        stat = []
        for z, gamma in ((q, gamma_q), (k, gamma_k)):
            pn = _rows(_pair_norms(self._head_normed(z, dh, gamma), dh), max(1, _ENTRIES // (dh // 2)))
            stat.append(pn.amax(0) if self.pct >= 100 else torch.quantile(pn, self.pct / 100.0, dim=0))
            del pn
        a_q, a_k = (s.double().cpu() for s in stat)
        half = torch.where((a_q > 0) & (a_k > 0), (a_k / a_q.clamp_min(1e-30)).sqrt(), torch.ones_like(a_q))
        half = half.clamp(1.0 / 256, 256.0)
        return torch.cat([half, half])

    def rope(self, z, heads: int, dh: int, theta: float, *, pruned: bool = False):
        """The benchmark's RoPE op (the scale is kept: a rotation); ``pruned``: ``z`` holds the last positions,
        whose angles come from the query's length. With a prefix of ``P`` tokens the query's positions start at
        ``P`` (the prefix's keys were rotated at ``0 .. P-1`` offline). A per-channel factor (``self.chan``, the
        same for both channels of a pair) is kept too."""
        n = self.prefix_len
        if pruned:
            out = self.cheap("rope", [z, "x"], op_fn(rope_tail_op, heads=heads, dh=dh, theta=theta, base=n),
                             self.scale[z], "rope")
        else:
            out = self.cheap("rope", [z], op_fn(rope_op, heads=heads, dh=dh, theta=theta, offset=n), self.scale[z],
                             "rope")
        if z in self.chan:
            self.chan[out] = self.chan[z]
        return out

    def attention(self, q, k, v, hq: int, hkv: int, dh: int, prefix: tuple | None = None):
        """``build_opt_from_hf``'s attention, with grouped-query heads: the scores' multiplier is
        ``2^30 16 s_q s_k / sqrt(dh)`` (Hugging Face's ``head_dim^-0.5`` in units of 1/16 nat), the output's maps the
        ``pct_att`` percentile of the raw ``P V`` to 127. ``prefix``: the fixed prefix's real keys and values
        ``[1, P, Hkv dh]`` (:func:`prefix_kv`), quantised on ``s_k`` and ``s_v`` -- the scales of this layer's computed
        k and v, calibrated on the query positions alone -- and prepended by :func:`prefix_attention` (default
        arguments of the op's function, so the digest binds them); ``m_o`` is calibrated with them in place. With
        ``qk_balance`` q's and k's channels carry the factors ``self.chan`` (k's: ``c_j``, q's: ``1/c_j``), whose
        product is 1, so ``m_s`` is the same formula; the prefix's keys are quantised on ``s_k c_j``."""
        sq, sk, sv = self.scale[q], self.scale[k], self.scale[v]
        m_s = round((1 << SHIFT) * 16.0 * sq * sk / math.sqrt(dh))
        P = self.prob_max
        if prefix is not None:
            k_real = prefix[0] if k not in self.chan else prefix[0] / self.chan[k].to(prefix[0].dtype)
            kp, vp = _to_int8(k_real, sk), _to_int8(prefix[1], sv)
            self.prefix_consts.append({"k": kp, "v": vp, "s_k": sk if k not in self.chan else sk * self.chan[k],
                                       "s_v": sv,
                                       "k_clip": float(k_real.abs().max()) / (INT8_MAX * sk),
                                       "v_clip": float(prefix[1].abs().max()) / (INT8_MAX * sv)})
            del k_real
            raw = prefix_attention(self.env[q], self.env[k], self.env[v], kp, vp, m_s, None, hq, hkv, dh, P)
        else:
            raw = integer_attention(self.env[q], self.env[k], self.env[v], m_s, None, hq, hkv, dh, P)
        s_raw = sv / float(P)    # probabilities 0..P x int8 values (PROB_MAX = 255: 8 bits; CERTIFIED_PROB_MAX = 2047)
        top = _top(raw.abs().reshape(-1, raw.shape[-1]).double(), self.pct_att) * s_raw
        del raw
        s_att = max(top, 1e-12) / INT8_MAX
        m_o = round((1 << SHIFT) * s_raw / s_att)
        if prefix is not None:   # the prefix's keys and values are constants of the op: bound by the digest
            fn = op_fn(prefix_attention_op, k_prefix=kp, v_prefix=vp, m_s=m_s, m_o=m_o, heads=hq, kv_heads=hkv, dh=dh,
                       pmax=P)
        else:
            fn = op_fn(attention_op, m_s=m_s, m_o=m_o, heads=hq, kv_heads=hkv, dh=dh, pmax=P)
        return self.cheap("attn", [q, k, v], fn, s_att, "attention")

    def swiglu(self, gate, up):
        """``silu(gate) * up``: the SiLU table from the gate's real scale, then the product requantised from real
        scales (the ``pct_glu`` percentile of ``|silu(gate) up|`` maps to 127)."""
        s_g, s_u = self.scale[gate], self.scale[up]
        xs = torch.arange(-128, 128, dtype=torch.float64) * s_g
        silu = xs * torch.sigmoid(xs)
        s_a = float(silu.abs().max()) / INT8_MAX
        table = torch.round(silu / s_a).clamp(-INT8_MAX, INT8_MAX).to(torch.int64)
        act = self.cheap("silu", [gate], op_fn(lut_op, table=table), s_a, "silu")
        prod = (self.env[act] * self.env[up]).abs().reshape(-1, self.env[up].shape[-1]).double()
        if self.glu_exp:
            return self._swiglu_exp(act, up, prod, s_a, s_u)
        top = _top(prod, self.pct_glu)
        if top == 0:
            raise ValueError(f"the SwiGLU product is 0 at its {self.pct_glu} percentile on the calibration rows "
                             f"(the up projection's int8 rows vanish?); no requantisation multiplier exists")
        s_p = max(top * s_a * s_u, 1e-12) / INT8_MAX
        del prod
        mm = torch.tensor(round((1 << SHIFT) * s_a * s_u / s_p), dtype=torch.int64)
        return self.cheap("glu", [act, up], op_fn(mul_requant_op, mult=mm, shift=SHIFT, lo=-INT8_MAX, hi=INT8_MAX),
                          s_p, "swiglu")

    def _swiglu_exp(self, act, up, prod, s_a, s_u):
        """The product with a per-token exponent (:func:`swiglu_exp`): the ``pct_glu`` percentile of the tokens' largest
        ``|silu(gate) up|`` takes exponent ``emax``, so ``s_p = top / (127 2^emax)`` and the tokens below
        ``top / 2^emax`` use exponent 0. ``emax`` is lowered until ``|a b| mult < 2^62`` for every int8 ``a, b``."""
        top = max(_top(prod.amax(-1, keepdim=True), self.pct_glu) * s_a * s_u, 1e-12)
        del prod
        emax = self.glu_exp
        while True:
            s_p = top / (INT8_MAX << emax)
            mm = round((1 << SHIFT) * s_a * s_u / s_p)
            if mm * INT8_MAX * INT8_MAX < 1 << 62 or emax == 0:
                break
            emax -= 1
        out = self.cheap("glu", [act, up], op_fn(swiglu_exp_op, mult=mm, emax=emax), s_p, "swiglu, per-token exponent")
        self.exp_of[out] = (act, up, mm, emax)
        return out

    def residual_exp(self, r, z, prod):
        """``residual(r, z)`` for the down projection ``z`` of a :func:`swiglu_exp` product ``prod``: the per-token
        shift of :func:`residual_exp` (not a ``requant_fn`` op, so V1 leaves the down projection clear)."""
        act, up, mm, emax = self.exp_of[prod]
        m = round((1 << SHIFT) * self.scale[z] / self.scale[r])
        return self.cheap("res", [r, z, act, up], op_fn(residual_exp_op, glu_mult=mm, mult=m, emax=emax), self.scale[r],
                          "residual add, per-token shift")


def _head_eq(h_int: torch.Tensor, s_in, w: torch.Tensor, b: torch.Tensor | None, heads: int, dh: int, pct: float):
    """``(diag(beta) w, beta b, beta)``: each head's rows times ``beta_h = max_h' a_h' / a_h`` (at most ``2^8``),
    ``a_h`` the ``pct`` percentile of ``|w h|`` over head ``h``'s outputs on the calibration rows (``beta`` per row,
    float64 ``[heads dh]``). An RMSNorm over the head follows, which ``beta_h`` does not change."""
    x = _rows(h_int.reshape(-1, h_int.shape[-1]), _ROWS_NORM).double().cpu()
    w_eff = w.double() * (s_in.double().cpu()[None, :] if torch.is_tensor(s_in) else s_in)
    z = (x @ w_eff.T).abs().reshape(-1, heads, dh)
    del x, w_eff
    a = torch.tensor([max(_top(z[:, i], pct), 1e-30) for i in range(heads)], dtype=torch.float64)
    beta = (a.max() / a).clamp(1.0, 256.0).repeat_interleave(dh)
    return (w.double() * beta[:, None]).float(), (None if b is None else b.double() * beta), beta


MAIN_RECIPE = dict(smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99, glu_smooth=0.2)
"""The main recipe for Llama-family and Qwen3 checkpoints (Sec. ``sec:quality``): SmoothQuant through the norms'
gains at alpha 0.55 with the 99.9th-percentile channel statistic, requantisation at the 99.99th percentile, the
SwiGLU product balance (``glu_smooth``) at 0.2; with ``prefix_tokens`` from PrefixQuant's rule on the calibration
windows (:func:`prefixquant_tokens`) and four 2,048-token windows of the WikiText-2 train split as calibration
(``experiments.common.checkpoints.llama_calibration``). Constants only: the graph keeps the benchmark's structure.
``qk_balance`` (on by default) adds Qwen3's q/k balance; the per-token exponent (``glu_exp``) is not part of it.
The alpha and the product balance per checkpoint are :data:`MAIN_SETTINGS` (:func:`main_recipe`)."""

MAIN_SETTINGS = {"open_llama_7b_v2": (0.55, 0.2), "Llama-2-7b-hf": (0.55, 0.2), "Llama-2-13b-hf": (0.6, 0.2),
                 "TinyLlama_v1.1": (0.75, 0.2), "Qwen3-1.7B-Base": (0.45, 0.3), "Qwen3-4B-Base": (0.45, 0.5)}
"""The main recipe's smoothing alpha and SwiGLU product balance per checkpoint, chosen per model on the 2,048-token
windows of the WikiText-2 (raw) test split.  At these settings the measured perplexity ratios to fp32 on every such
window (``experiments/quality/ppl_llama.py``, calibration as in :data:`MAIN_RECIPE`) are x1.073 for OpenLLaMA-7B,
x1.122 for TinyLlama, x1.40 for Qwen3-1.7B and x5.3 for Qwen3-4B.  Llama-2-7B takes :data:`MAIN_RECIPE`'s
(OpenLLaMA-7B's, a model of the same shapes) and Llama-2-13B 0.6; any other checkpoint :data:`MAIN_RECIPE`'s.
Llama-2's two settings are fixed here before the final run and never chosen from its records: the run package's
quality tier measures each among its neighbours on the test and the validation split, and the paper reports where
it lies (``experiments/paper/macros.py``, ``LLAMA_GRIDS``)."""


def main_recipe(name: str, alpha: float | None = None, glu_smooth: float | None = None) -> dict:
    """:data:`MAIN_RECIPE` with the smoothing alpha and the SwiGLU product balance of the checkpoint ``name`` (a Hugging
    Face name; :data:`MAIN_SETTINGS`); ``alpha`` and ``glu_smooth`` replace them.  The fixed prefix is the builder's
    ``prefix_tokens``."""
    a, g = MAIN_SETTINGS.get(name.split("/")[-1], (MAIN_RECIPE["smooth"], MAIN_RECIPE["glu_smooth"]))
    a, g = a if alpha is None else alpha, g if glu_smooth is None else glu_smooth
    return dict(MAIN_RECIPE, smooth=a, glu_smooth=g)


def _glu_smooth(h_int: torch.Tensor, s_in, wg: torch.Tensor, wu: torch.Tensor, w_down: torch.Tensor,
                alpha: float) -> torch.Tensor:
    """``glu_smooth``'s factors ``g [ff]`` (float64): SmoothQuant's balance for the down projection's input, folded
    through the up projection (the product ``silu(gate) up`` is linear in ``up``). With ``A_j`` the largest
    ``|silu(gate) up|`` of channel ``j`` on the calibration rows (from the norm's integer output ``h_int`` and its scale
    ``s_in``, a scalar or a per-channel vector) and ``W_j`` the largest ``|w_down[:, j]|``,
    ``g_j = A_j^alpha / W_j^(1-alpha)``, divided by their geometric mean, then kept in ``[U_j / U_max, D_max / D_j]``
    (``U_j`` the largest ``|up row j|`` as the integer input sees it, ``D_j = W_j``) and in ``[2^-8, 2^8]``. Dividing
    up's row ``j`` by ``g_j`` and multiplying the down projection's column ``j`` by ``g_j`` leaves every real output
    unchanged; the product's one int8 scale is then no longer set by its few largest channels. The box keeps both
    weight matrices' per-tensor int8 scales from growing: without it one up row (a channel with a tiny product, so a
    small ``g_j``) can grow up to 256 times, set up's scale and round the other rows to zero, and the product's
    calibration percentile is then 0 (no requantisation multiplier exists: :meth:`_Builder.swiglu` raises)."""
    x = _rows(h_int.reshape(-1, h_int.shape[-1]), _ROWS_NORM).double().cpu()
    s = s_in.double().cpu()[None, :] if torch.is_tensor(s_in) else float(s_in)
    a = torch.nn.functional.silu(x @ (wg.double() * s).T) * (x @ (wu.double() * s).T)
    amax = a.abs().amax(0)
    del x, a
    wmax = w_down.double().abs().amax(0)
    umax = (wu.double() * s).abs().amax(1)                  # up's rows as mat() quantises them (input scale folded)
    ok = (amax > 0) & (wmax > 0) & (umax > 0)
    g = torch.ones_like(amax)
    g[ok] = amax[ok] ** alpha / wmax[ok] ** (1 - alpha)
    g = g / torch.exp(torch.log(g[ok]).mean()) if bool(ok.any()) else g
    lo = torch.where(ok, umax / umax.max(), torch.ones_like(g))           # <= 1: up's largest row stays the largest
    hi = torch.where(ok, wmax.max() / wmax.clamp_min(1e-300), torch.ones_like(g))   # >= 1: likewise down's column
    return torch.minimum(torch.maximum(g, lo), hi).clamp(1.0 / 256, 256.0)


def _with_prefix(ids: torch.Tensor, prefix: torch.Tensor | None) -> torch.Tensor:
    """``[prefix | ids]`` per row of ``ids [B, T]`` (``ids`` itself without a prefix)."""
    if prefix is None:
        return ids
    return torch.cat([prefix.to(ids.device).reshape(1, -1).expand(ids.shape[0], -1), ids], 1)


def _hidden_max(model, calib_ids: torch.Tensor, prefix: torch.Tensor | None = None) -> tuple[float, float]:
    """Largest ``|value|`` of the fp checkpoint's residual stream (the embedding output and every block's output)
    on the calibration windows, one window at a time, through forward hooks (no hidden state is kept): over the
    windows' positions, and (0.0 without one) over the positions of the fixed ``prefix [1, P]`` run in front of each
    window -- which sets no scale."""
    top = [0.0, 0.0]
    n = 0 if prefix is None else prefix.shape[1]

    def hook(mod, inp, out):
        out = (out[0] if isinstance(out, tuple) else out).detach().abs()
        top[0] = max(top[0], float(out[:, n:].max()))
        if n:
            top[1] = max(top[1], float(out[:, :n].max()))

    inner = model.model
    hooks = [inner.embed_tokens.register_forward_hook(hook)] + [layer.register_forward_hook(hook)
                                                                for layer in inner.layers]
    try:
        with torch.no_grad():
            for i in range(calib_ids.shape[0]):
                inner(_with_prefix(calib_ids[i:i + 1], prefix).to(model.device))
    finally:
        for h in hooks:
            h.remove()
    return top[0], top[1]


def prefix_kv(model, prefix_ids, n_layers: int | None = None) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The fp checkpoint's keys and values of the fixed prefix ``prefix_ids`` (``P`` token ids), per layer (the
    first ``n_layers``), as float64 ``[1, P, Hkv head_dim]`` on the CPU: Hugging Face's KV cache after one forward
    pass over the prefix alone -- the keys after the k norm (Qwen3) and RoPE at the positions ``0 .. P-1``, the
    values as projected. Causal attention makes them independent of the query that follows."""
    ids = torch.as_tensor(prefix_ids, dtype=torch.int64).reshape(1, -1)
    with torch.no_grad():
        cache = model.model(ids.to(model.device), use_cache=True).past_key_values
    out = []
    for i in range(len(list(model.model.layers)[:n_layers])):
        k, v = cache[i]                                        # [1, Hkv, P, head_dim]
        out.append(tuple(t.detach().transpose(1, 2).reshape(1, ids.shape[1], -1).to("cpu", torch.float64)
                         for t in (k, v)))
    return out


def _prefixquant_select(maxes: list[torch.Tensor], ids: torch.Tensor, bos: int, eta: float) -> tuple[list[int], dict]:
    """PrefixQuant's choice from the token-wise maxima ``maxes`` (per block, ``[W, T]``: ``max_j |x_tj|`` at the
    down projection's input of token ``t`` of window ``w``) of the windows ``ids [W, T]``: token ``t`` is an outlier
    token of its block when ``R_t = max_t / median_t' max_t' > eta`` (the median over its window); ``o`` is the
    ceiling of the largest, over blocks, mean number of outlier tokens per window; the prefix is the ``o - 1`` token
    ids found most often among the outlier tokens -- not counting a window's first position, the initial-token
    outlier every window has, nor ``bos`` -- by decreasing count (ties: smaller id first), then ``bos``."""
    per_block, freq = [], collections.Counter()
    for m in maxes:
        out = m / m.median(-1, keepdim=True).values.clamp_min(1e-30) > eta
        per_block.append(float(out.sum(-1).double().mean()))
        for w, t in out.nonzero().tolist():
            if t > 0 and int(ids[w, t]) != bos:
                freq[int(ids[w, t])] += 1
    o = math.ceil(max(per_block, default=0.0))
    extra = [tok for tok, _ in sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))][:max(0, o - 1)]
    return extra + [bos], {"rule": "prefixquant", "eta": eta, "o": o, "outliers_per_block": per_block,
                           "outlier_tokens": sorted(freq.items(), key=lambda kv: (-kv[1], kv[0]))[:16]}


def prefixquant_tokens(model, calib_ids: torch.Tensor, *, eta: float = 64.0,
                       bos: int | None = None) -> tuple[list[int], dict]:
    """The fixed prefix PrefixQuant prescribes (Chen et al. 2024, arXiv:2410.05265, Sec. 4; their ``eta_1 = 64``),
    from the fp checkpoint's down-projection inputs on the calibration windows ``calib_ids [W, T]``
    (:func:`_prefixquant_select`): the high-frequency outlier tokens, then the BOS token (their "[BOS] last, for
    consistency"; ``bos`` defaults to the configuration's ``bos_token_id``: ``<s>`` for Llama, ``<|endoftext|>``
    for Qwen3). A model whose only outlier is the initial token gets ``[bos]`` (their Llama-3-8B and Qwen-2-7B
    prefix); Llama-2-7B's is ``. \\n [BOS]``. Returns the token ids and the statistics behind them."""
    bos = model.config.bos_token_id if bos is None else bos
    if bos is None:
        raise ValueError("the model has no bos_token_id: give the prefix's token ids")
    layers = list(model.model.layers)
    maxes: list[list[torch.Tensor]] = [[] for _ in layers]

    def pre_hook(i):
        def hook(mod, args):
            maxes[i].append(args[0].detach().abs().amax(-1).to("cpu", torch.float64))
        return hook

    hooks = [layer.mlp.down_proj.register_forward_pre_hook(pre_hook(i)) for i, layer in enumerate(layers)]
    try:
        with torch.no_grad():
            for w in range(calib_ids.shape[0]):
                model.model(calib_ids[w:w + 1].to(model.device))
    finally:
        for h in hooks:
            h.remove()
    return _prefixquant_select([torch.cat(m) for m in maxes], calib_ids.cpu(), int(bos), eta)


def config_of(hf_config, name: str | None = None) -> DecoderConfig:
    """The ``DecoderConfig`` of a Hugging Face Llama or Qwen3 configuration's shape (named ``name``, by default the
    checkpoint's name)."""
    d, hq = hf_config.hidden_size, hf_config.num_attention_heads
    return DecoderConfig(
        name or getattr(hf_config, "_name_or_path", "") or "hf", d, hf_config.num_hidden_layers, hq,
        hf_config.num_key_value_heads or hq,
        getattr(hf_config, "head_dim", None) or d // hq, hf_config.intermediate_size, hf_config.vocab_size,
        norm="rmsnorm", mlp="swiglu", pos="rope", rope_theta=float(hf_config.rope_theta),
        bias=bool(getattr(hf_config, "attention_bias", False)), tied=bool(hf_config.tie_word_embeddings),
        qk_norm=hf_config.model_type == "qwen3", max_pos=hf_config.max_position_embeddings)


def same_shape(a: DecoderConfig, b: DecoderConfig) -> list[str]:
    """The fields in which two configurations differ, but for the name and ``max_pos`` (which a RoPE graph does
    not use)."""
    return [f"{k}: {getattr(a, k)} vs {getattr(b, k)}" for k in a.__dataclass_fields__
            if k not in ("name", "max_pos") and getattr(a, k) != getattr(b, k)]


def build_llama_from_hf(model, calib_ids: torch.Tensor, *, pct: float = 100.0, norm_gain: float = 4.0,
                        smooth: float | None = None, smooth_pct: float = 99.9, pct_att: float = 100.0,
                        prune_last: bool = False, device=None, pct_glu: float | None = None, head_eq: bool = True,
                        residual_headroom: float = 1.25, n_layers: int | None = None, glu_exp: int = 0,
                        free_calib: bool = True, prob_max: int = CERTIFIED_PROB_MAX,
                        prefix_tokens=None, qk_balance: bool = True,
                        glu_smooth: float | None = None) -> tuple[IntGraph, dict]:
    """An integer Llama / Qwen3 decoder with the checkpoint's real weights.

    ``calib_ids [B, T]`` (``B`` calibration windows) sets every activation scale. The knobs are public constants
    of cheap ops or the offline weight quantisation -- none changes a shape, so none changes the cost:

    * ``pct``: percentile of ``|value|`` mapped to 127 at each requantisation of q/k/v/gate/up and at the q/k
      norms' outputs (100 = never clip on calibration data); ``pct_att``: the same at the attention output;
      ``pct_glu`` (default ``pct``): the same at the SwiGLU product (the down projection's input);
    * ``norm_gain``: the scalar RMSNorm gain (output ``norm_gain x normalised``, range ``+-127/norm_gain`` RMS);
      ``smooth`` (alpha) with ``smooth_pct``: SmoothQuant gain vectors instead (``build_opt_from_hf``'s rule);
    * ``head_eq``: equalise the heads in front of Qwen3's q/k norms (module docstring); no effect on Llama;
    * ``prune_last``: the last block at the last position only (calibrated on every position);
    * ``n_layers``: build the first ``n_layers`` blocks only (then the final norm and the LM head), as
      ``build_decoder(..., n_layers=)``;
    * ``glu_exp`` (opt-in, default 0 = off): a per-token power-of-two exponent ``0 .. glu_exp`` at the SwiGLU
      product (:func:`swiglu_exp`) and the matching per-token shift in the next residual add (:func:`residual_exp`), for
      massive activations at the down projection's input. Both are cheap ops the verifier recomputes; the graph
      then differs from the benchmark's in exactly these two ops per block (their notes, and the residual add also
      reads the product's two inputs), and V1 leaves the down projection clear (its reader is no ``requant_fn``).
    * ``prefix_tokens`` (default None = off; the main recipe's comes from :func:`prefixquant_tokens`): a fixed
      public prefix of ``P`` token ids (e.g. the BOS alone) in front of every query.
      Its keys and values per layer (:func:`prefix_kv`, the fp checkpoint's) become int8 constants of the attention
      op on that layer's k and v scales, the query's RoPE positions start at ``P``, and every scale -- ``s_res``
      from the fp residual maximum included -- is calibrated on the query positions only (the calibration windows
      run after the prefix). The structure, the weight ops and the claim columns are the benchmark graph's;
      ``graph.meta`` records ``prefix_len`` and ``prefix_tokens``.
    * ``glu_smooth`` (default None = off; an alpha, the main recipe's per checkpoint in :data:`MAIN_SETTINGS`):
      SmoothQuant's balance for the down projection's input, folded through the up projection
      (:func:`_glu_smooth`): up's row ``j`` / ``g_j``, the down projection's column ``j`` * ``g_j``. Constants only
      (the structure is the benchmark graph's); ``info["glu_smooth_g"]`` keeps the factors.
    * ``qk_balance`` (default on; Qwen3 only, no effect on Llama): per RoPE pair, a factor ``c_p`` into the q norm's
      gains and ``1/c_p`` into the k norm's (:meth:`_Builder.qk_balance`), so a channel whose k gain is large and
      whose q gain is near zero no longer sets k's whole int8 range; the scores are unchanged in real numbers.
      Constants only: the structure is the benchmark graph's. ``False`` gives the graph without the balance.
      Measured by the attribution of ``experiments/quality/llama_attrib.py`` on a tiny Qwen3 with Qwen3-0.6B's
      layer-0 q/k-norm gains, the k norm alone adds 0.90 nats to the attention scores' error without the balance
      and about 0.11 with it.

    ``device`` is where the integer calibration pass runs (the fp model stays where it is). Returns the graph and
    ``info``: the real scales (``logit_scale``, ``s_residual``), the knobs, the residual tensors at the block
    boundaries (``embed``, ``blocks``) and the ``DecoderConfig`` of the model's shape (``config``); for the
    op-level attribution (``experiments/quality/llama_attrib.py``) ``roles`` (per block ``{role: tensor name}``,
    the roles of ``ROLES`` and ``in``, the block's input), ``scales`` (every tensor's real scale: a float, or a
    per-channel vector for a smoothed norm and a balanced q/k), ``head_beta`` (per block the head-equalisation
    factors ``(beta_q, beta_k)`` per output row, or None) and ``qk_balance_c`` (per block ``c [head_dim]`` or None);
    with a prefix also ``prefix_tokens``, ``prefix_kv`` (per layer the int8 ``(k, v)``, the graph's constants),
    ``prefix_scales`` (``(s_k, s_v)``; ``s_k`` per channel with ``qk_balance``), ``prefix_clip`` (``max |real| /
    (127 s)`` of the prefix's keys and values: above 1 they clip) and ``prefix_hidden_max`` (the fp residual
    stream's largest ``|value|`` at the prefix positions).
    """
    cfg = model.config
    arch = cfg.model_type
    if arch not in ("llama", "qwen3"):
        raise ValueError(f"model type {arch!r}: Llama and Qwen3 only")
    scaling = getattr(cfg, "rope_scaling", None)
    if scaling and scaling.get("rope_type", scaling.get("type", "default")) != "default":
        raise ValueError(f"RoPE scaling {scaling!r} is not handled")
    if cfg.hidden_act != "silu":
        raise ValueError(f"activation {cfg.hidden_act!r} is not SwiGLU's SiLU")
    if getattr(cfg, "use_sliding_window", False):
        raise ValueError("sliding-window attention is not handled")
    shape = config_of(cfg)
    d, hq, hkv, dh, theta = shape.d_model, shape.n_heads, shape.n_kv_heads, shape.head_dim, shape.rope_theta
    inner = model.model
    layers = list(inner.layers)[:n_layers]
    pct_glu = pct if pct_glu is None else pct_glu
    pre = None
    if prefix_tokens is not None and len(prefix_tokens):
        pre = torch.as_tensor(prefix_tokens, dtype=torch.int64).reshape(1, -1)
        if int(pre.min()) < 0 or int(pre.max()) >= cfg.vocab_size:
            raise ValueError(f"prefix token ids {pre[0].tolist()} outside the vocabulary")
    kv = None if pre is None else prefix_kv(model, pre, len(layers))

    # The residual stream's real scale: the fp checkpoint's hidden states never clip at RES_MAX on the calibration
    # windows (their own positions: a prefix runs in front of each and sets no scale); the embedding's int8 scale
    # is c x s_residual, c the graph's "embed to residual" multiplier.
    emb = inner.embed_tokens.weight
    res_max, pre_max = _hidden_max(model, calib_ids, pre)
    res_max *= residual_headroom
    s_res = max(res_max / RES_MAX, float(emb.abs().max()) / (INT8_MAX * 8))

    B = _Builder(calib_ids.to(device or calib_ids.device), pct=pct, norm_gain=norm_gain, smooth=smooth,
                 smooth_pct=smooth_pct, pct_att=pct_att, pct_glu=pct_glu, glu_exp=glu_exp)
    assert 1 <= prob_max <= 2047, prob_max
    B.prob_max = prob_max
    B.prefix_len = 0 if pre is None else pre.shape[1]
    embed_mult = max(1, round(float(emb.abs().max()) / INT8_MAX / s_res))
    # build_opt_from_hf's rounding: c rounds max|E| / (127 s_res) to the nearest integer, so for a ratio in
    # (k, k + 1/2), k >= 1, the largest entries clip at 127; ``embed_clip`` (> 1 then) records by how much
    embed_clip = float(emb.abs().max()) / (INT8_MAX * embed_mult * s_res)
    tok = B.mat("x", emb.T.float(), None, 1.0, layout="embed", s_w=embed_mult * s_res)
    r = B.cheap("scale", [tok], op_fn(scale_op, c=embed_mult), s_res, "embed to residual")
    embed_name, blocks = r, []

    def bias(lin):
        return None if lin.bias is None else lin.bias.detach()

    roles, head_beta, balance, glu_g = [], [], [], []
    for i, layer in enumerate(layers):
        prune = prune_last and i == len(layers) - 1
        at, mlp = layer.self_attn, layer.mlp
        role = {"in": r}                            # the block's tensors by role (ROLES; the attribution's map)
        g1 = layer.input_layernorm.weight.detach()
        wq, wk, wv = (p.weight.detach().float() * g1.float()[None, :] for p in (at.q_proj, at.k_proj, at.v_proj))
        bq, bk, bv = bias(at.q_proj), bias(at.k_proj), bias(at.v_proj)
        h = role["attn_norm"] = B.norm(r, d, [wq, wk, wv])
        qk_norm = hasattr(at, "q_norm")
        betas = None
        if qk_norm and head_eq:
            wq, bq, beta_q = _head_eq(B.env[h], B.scale[h], wq, bq, hq, dh, pct)
            wk, bk, beta_k = _head_eq(B.env[h], B.scale[h], wk, bk, hkv, dh, pct)
            betas = (beta_q, beta_k)
        head_beta.append(betas)
        rope_pairs = None if qk_norm else dh       # without q/k norm, RoPE reads the requantised values
        zq = role["q_proj"] = B.mat(B.last(h, calibrate_all=True) if prune else h, wq, bq, B.scale[h])
        q = role["q"] = B.to_int8(zq, pairs=rope_pairs)
        zk = role["k_proj"] = B.mat(h, wk, bk, B.scale[h])
        k = role["k"] = B.to_int8(zk, pairs=rope_pairs)
        zv = role["v_proj"] = B.mat(h, wv, bv, B.scale[h])
        v = role["v"] = B.to_int8(zv)
        del wq, wk, wv
        c = None
        if qk_norm:
            gq, gk = at.q_norm.weight.detach(), at.k_norm.weight.detach()
            if qk_balance:
                c = B.qk_balance(q, k, dh, gq, gk)
            q = role["q_norm"] = B.head_norm(q, hq, dh, gq, "qn", "q norm", c=c)
            k = role["k_norm"] = B.head_norm(k, hkv, dh, gk, "kn", "k norm", c=None if c is None else 1.0 / c)
        balance.append(c)
        q = role["q_rope"] = B.rope(q, hq, dh, theta, pruned=prune)
        k = role["k_rope"] = B.rope(k, hkv, dh, theta)
        att = role["attn"] = B.attention(q, k, v, hq, hkv, dh, prefix=None if kv is None else kv[i])
        r_in = B.last(r, calibrate_all=True) if prune else r
        zo = role["o_proj"] = B.mat(att, at.o_proj.weight.detach().float(), bias(at.o_proj), B.scale[att])
        r = role["res_attn"] = B.residual(r_in, zo)

        g2 = layer.post_attention_layernorm.weight.detach()
        wg, wu = (p.weight.detach().float() * g2.float()[None, :] for p in (mlp.gate_proj, mlp.up_proj))
        h = role["mlp_norm"] = B.norm(r, d, [wg, wu])
        w_down, b_up = mlp.down_proj.weight.detach().float(), bias(mlp.up_proj)
        g = None
        if glu_smooth is not None:          # the product's channel j / g_j, the down projection's column j * g_j
            g = _glu_smooth(B.env[h], B.scale[h], wg, wu, w_down, glu_smooth)
            wu = (wu.double() / g[:, None]).float()
            w_down = (w_down.double() * g[None, :]).float()
            b_up = None if b_up is None else b_up.double() / g
        glu_g.append(g)
        zg = role["gate_proj"] = B.mat(h, wg, bias(mlp.gate_proj), B.scale[h])
        gate = role["gate"] = B.to_int8(zg)
        zu = role["up_proj"] = B.mat(h, wu, b_up, B.scale[h])
        up = role["up"] = B.to_int8(zu)
        del wg, wu
        prod = role["glu"] = B.swiglu(gate, up)
        role["silu"] = B.ops[-1].inputs[0]          # the SiLU table's output, the product's first input
        if glu_exp and mlp.down_proj.bias is not None:
            raise ValueError("glu_exp scales the down projection's claims per token, which a bias would not follow")
        z = role["down_proj"] = B.mat(prod, w_down, bias(mlp.down_proj), B.scale[prod])
        del w_down
        r = role["res_mlp"] = B.residual_exp(r, z, prod) if glu_exp else B.residual(r, z)
        blocks.append(r)
        roles.append(role)
        if free_calib:
            B.release({"x", r})

    w_head = model.lm_head.weight.detach().float() * inner.norm.weight.detach().float()[None, :]
    h = B.last(B.norm(r, d, [w_head]))      # (in a pruned graph the one position of its last block)
    logits = B.mat(h, w_head, None, B.scale[h])
    del w_head
    info = {"arch": arch, "config": dataclasses.replace(shape, n_layers=len(layers)), "s_residual": s_res,
            "residual_real_max": res_max, "logit_scale": B.scale[logits], "embed_mult": embed_mult,
            "embed_clip": embed_clip, "embed": embed_name, "blocks": blocks, "pct": pct, "norm_gain": norm_gain,
            "smooth": smooth, "smooth_pct": smooth_pct, "pct_att": pct_att, "pct_glu": pct_glu,
            "head_eq": head_eq and arch == "qwen3", "prune_last": prune_last,
            "residual_headroom": residual_headroom, "glu_exp": glu_exp,
            "glu_emax": [e for *_, e in B.exp_of.values()], "prefix_tokens": None, "prefix_len": 0,
            "qk_balance": qk_balance and arch == "qwen3", "roles": roles, "head_beta": head_beta,
            "qk_balance_c": balance, "glu_smooth_g": glu_g, "prob_max": prob_max,
            "scales": {n: (s * B.chan[n] if n in B.chan else s) for n, s in B.scale.items()}}
    meta = {}
    if pre is not None:
        meta = {"prefix_len": pre.shape[1], "prefix_tokens": pre[0].tolist()}
        info.update(meta, prefix_kv=[(c["k"], c["v"]) for c in B.prefix_consts],
                    prefix_scales=[(c["s_k"], c["s_v"]) for c in B.prefix_consts],
                    prefix_clip=[(c["k_clip"], c["v_clip"]) for c in B.prefix_consts], prefix_hidden_max=pre_max)
    return IntGraph(B.ops, "x", logits, meta), info


# ------------------------------------------------------------------ evaluation

def lean_logits(graph: IntGraph, ids: torch.Tensor, *, all_positions: bool = True, chunk: int = 256):
    """Yield ``(start, z)``: the integer logits ``z [B, n, V]`` (int64) of the positions ``start .. start + n - 1``.

    The graph runs with the lean forward (``free=True``, no claim kept) up to the LM head's input, and the LM head
    then takes ``chunk`` positions at a time: the same integers as one pass (each column of a weight op depends on
    its own input column only), without the ``V x T`` logits. ``all_positions`` drops the "last position" slice in
    front of the head -- one pass then gives what one query per prefix would (every op is per position but causal
    attention, and no constant depends on the input); a pruned graph has its last position only."""
    head = graph.ops[-1]
    if not (isinstance(head, MatOp) and head.output == graph.output_name):
        raise ValueError("the graph does not end with its LM head")
    ops = list(graph.ops[:-1])
    if all_positions:
        lasts = [i for i, o in enumerate(ops) if isinstance(o, CheapOp) and o.note == "last position"]
        if len(lasts) != 1:
            raise ValueError("a pruned graph has logits at its last position only")
        o = ops[lasts[0]]
        ops[lasts[0]] = CheapOp(o.name, o.inputs, o.output, fn=op_fn(identity_op), note=o.note, params=o.params)
    body = IntGraph(ops, graph.input_name, head.inputs[0], dict(graph.meta))
    env, _ = body.forward(ids, free=True, send=lambda z: None)
    h = env.pop(head.inputs[0])
    del env
    for a in range(0, h.shape[1], chunk):
        hc = h[:, a:a + chunk]
        yield a, head.fold(head.compute(hc), hc)


def real_logits(graph: IntGraph, info: dict, ids: torch.Tensor, *, all_positions: bool = False,
                chunk: int = 256) -> torch.Tensor:
    """Real-valued logits ``[B, T or 1, V]`` (float64) of the integer graph (:func:`lean_logits`, concatenated)."""
    return torch.cat([z.double() for _, z in lean_logits(graph, ids, all_positions=all_positions, chunk=chunk)],
                     1) * info["logit_scale"]


def residual_states(graph: IntGraph, info: dict, ids: torch.Tensor) -> list[torch.Tensor]:
    """The integer residual stream at every block boundary -- after the embedding, then after each block -- in
    real units (float64 on the CPU, ``[B, T, d]``; a pruned graph's last block has its last position only). The ops
    run as in the lean forward and stop after the last block."""
    names = [info["embed"]] + list(info["blocks"])
    want, got = set(names), {}
    env = {graph.input_name: ids}
    last = graph.last_use()
    for i, op in enumerate(graph.ops):
        if isinstance(op, MatOp):
            xin = env[op.inputs[0]]
            env[op.output] = op.fold(op.compute(xin), xin)
        else:
            env[op.output] = op.fn(*[env[n] for n in op.inputs])
        if op.output in want:
            got[op.output] = env[op.output].to("cpu", torch.float64) * info["s_residual"]
            if len(got) == len(want):
                break
        for n in op.inputs:
            if last.get(n, -1) <= i:
                env.pop(n, None)
    return [got[n] for n in names]


def hf_residual_states(model, ids: torch.Tensor, n_layers: int | None = None, prefix=None) -> list[torch.Tensor]:
    """The fp checkpoint's residual stream at the same boundaries (the embedding's output, then each decoder
    layer's), float64 on the CPU, through forward hooks; ``n_layers``: the first blocks only. ``prefix`` (token
    ids): the checkpoint runs on ``[prefix | ids]`` and the states of ``ids``' positions are returned, the
    reference of a graph built with ``prefix_tokens=prefix``."""
    out: list[torch.Tensor] = []
    pre = None if prefix is None or not len(prefix) else torch.as_tensor(prefix, dtype=torch.int64).reshape(-1)
    n = 0 if pre is None else pre.numel()

    def hook(mod, inp, o):
        o = o[0] if isinstance(o, tuple) else o
        out.append(o.detach()[:, n:].to("cpu", torch.float64))

    inner = model.model
    hooks = [inner.embed_tokens.register_forward_hook(hook)] + [
        layer.register_forward_hook(hook) for layer in list(inner.layers)[:n_layers]]
    try:
        with torch.no_grad():
            inner(_with_prefix(ids, pre).to(model.device))
    finally:
        for h in hooks:
            h.remove()
    return out


def block_cosines(ours: list[torch.Tensor], ref: list[torch.Tensor]) -> list[dict]:
    """Per boundary: the mean and the minimum over positions of the cosine between the integer and the fp
    residual stream (a pruned last block is compared at the last position)."""
    out = []
    for a, b in zip(ours, ref):
        b = b[:, -a.shape[1]:]
        c = torch.nn.functional.cosine_similarity(a.reshape(-1, a.shape[-1]), b.reshape(-1, b.shape[-1]), dim=-1)
        out.append({"mean": float(c.mean()), "min": float(c.min())})
    return out


# ------------------------------------------------------------------ structure

def matches_benchmark_graph(graph: IntGraph, config: str | DecoderConfig, *, n_layers: int | None = None,
                            prune_last: bool = False) -> list[str]:
    """Differences from ``build_decoder(config, n_layers=, prune_last=)``'s graph: the op count, then per op its
    kind, name, inputs and output, a weight op's shape, bias and layout, a cheap op's note, and finally every weight
    op's column count on a 9-token query (``IntGraph.claim_columns``). Returns the list of mismatches (empty for a
    Llama or Qwen3 graph of ``config``'s shape: RMSNorm leaves no LM head bias)."""
    cfg = CONFIGS[config] if isinstance(config, str) else config
    ref = build_decoder(cfg, n_layers=n_layers, prune_last=prune_last, calib_tokens=8)
    if len(ref.ops) != len(graph.ops):
        return [f"op count {len(graph.ops)} vs benchmark {len(ref.ops)}"]
    problems = []
    for a, b in zip(graph.ops, ref.ops):
        if type(a) is not type(b):
            problems.append(f"{a.name}: {type(a).__name__} vs {type(b).__name__}")
        elif (a.name, a.inputs, a.output) != (b.name, b.inputs, b.output):
            problems.append(f"{a.name}: wiring {a.inputs} -> {a.output} vs {b.name}: {b.inputs} -> {b.output}")
        elif isinstance(a, MatOp) and (a.n_rows, a.n_in, a.layout) != (b.n_rows, b.n_in, b.layout):
            problems.append(f"{a.name}: shape {(a.n_rows, a.n_in, a.layout)} vs {(b.n_rows, b.n_in, b.layout)}")
        elif isinstance(a, MatOp) and a.has_bias != b.has_bias:
            problems.append(f"{a.name}: bias {a.has_bias} vs benchmark {b.has_bias}")
        elif isinstance(a, CheapOp) and a.note != b.note:
            problems.append(f"{a.name}: '{a.note}' vs '{b.note}'")
    if not problems:
        x = torch.zeros(1, 9, dtype=torch.int64)
        if graph.claim_columns(x) != ref.claim_columns(x):
            problems.append(f"claim columns {graph.claim_columns(x)} vs {ref.claim_columns(x)}")
    return problems

"""Integer RMSNorm / RoPE / SwiGLU / GQA decoders built from *real* pretrained weights (plan 2, item F2b-Q).

``real_weights.build_opt_from_hf`` does this for OPT; this module does it for the Llama family (Hugging Face
``LlamaForCausalLM``: TinyLlama, Llama-2) and for Qwen3 (``Qwen3ForCausalLM``: Qwen3-0.6B/1.7B/4B-Base). The graph
is ``transformer.build_decoder``'s for the same shape -- the same op kinds, in the same order, with the same names,
wiring and matrix shapes, so the cost tables measured on random weights apply to it -- and only the *constants*
change (``matches_benchmark_graph`` checks it; the diff is empty, see below):

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
  The float function is unchanged (up to RMSNorm's epsilon).
* **RoPE** uses the graph's Q14 tables (``transformer._rope``). Hugging Face's ``rotate_half`` gives
  ``[x1 cos - x2 sin | x2 cos + x1 sin]`` with the frequencies ``theta^(-2i/head_dim)`` -- the convention of
  ``_rope`` (``x [cos | cos] + [x2 | x1] [-sin | sin]``; tested on random values). The rotation preserves each
  pair's norm, so the scale before RoPE is set from the pairs' norms, which then never clip at 127.
* **GQA**: query head ``j`` reads key/value head ``j // (n_heads / n_kv)``, Hugging Face's ``repeat_kv``
  (``repeat_interleave``), which is ``transformer._attention``'s grouping.
* **SwiGLU**: the SiLU table is built from the gate's real scale ``s_g`` (entry ``x`` is
  ``round(silu(x s_g) / s_a)``, ``s_a = max_x |silu(x s_g)| / 127`` over the table's 256 inputs, which is
  ``silu(127 s_g) / 127`` unless ``s_g`` is tiny), and the product ``act x up`` is requantised
  (``graph.requant_mul``) with the multiplier ``round(2^30 s_a s_u / s_p)`` from real scales.
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
down projection's input; whether static per-tensor scales survive them is what ``experiments/9_quality/
ppl_llama.py`` measures (``pct_glu`` is the product's clipping percentile). The opt-in ``glu_exp`` (plan F2b-Q's
outlier handling) gives each token of that product its own power-of-two exponent instead, with the matching shift in
the next residual add; it is the one option that changes the graph's structure (two ops per block).
"""

from __future__ import annotations

import dataclasses
import math

import torch

from .graph import INT8_MAX, CheapOp, IntGraph, MatOp, requant_fn, requant_mul
from .transformer import (
    CONFIGS, PROB_MAX, RES_MAX, SHIFT, DecoderConfig, _attention, _isqrt, _last_position, _lut, _norm_int, _rope,
    build_decoder,
)

__all__ = ["build_llama_from_hf", "config_of", "same_shape", "matches_benchmark_graph", "lean_logits",
           "real_logits", "residual_states", "hf_residual_states", "block_cosines", "swiglu_exp", "residual_exp"]

_ROWS_NORM = 1 << 14
"""Calibration rows read by the smoothing statistics (evenly spaced over all windows)."""
_ENTRIES = 1 << 24
"""Entries a percentile reads (``torch.quantile``'s limit): whole rows, evenly spaced over all windows."""


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


def _pair_norms(real: torch.Tensor, dh: int) -> torch.Tensor:
    """``[..., H*dh]`` -> ``[rows, dh/2]``: the norm of each RoPE pair ``(x_i, x_{i+dh/2})`` of each head, the
    largest magnitude either value reaches under any rotation."""
    x = real.reshape(-1, dh)
    return torch.hypot(x[:, :dh // 2], x[:, dh // 2:])


class _Builder:
    """Like ``transformer._Builder``, but every tensor carries its real scale (``real_weights._RealBuilder``'s
    arithmetic for the weight ops, requantisations and residual adds, extended to RMSNorm, q/k norm, RoPE and
    SwiGLU)."""

    prob_max = PROB_MAX     # the attention probabilities' scale (build_llama_from_hf(prob_max=...))

    def __init__(self, tokens: torch.Tensor, *, pct: float, norm_gain: float, smooth: float | None,
                 smooth_pct: float, pct_att: float, pct_glu: float, glu_exp: int = 0) -> None:
        self.pct, self.norm_gain, self.smooth, self.smooth_pct = pct, norm_gain, smooth, smooth_pct
        self.pct_att, self.pct_glu, self.glu_exp = pct_att, pct_glu, glu_exp
        self.ops: list = []
        self.env = {"x": tokens}
        self.scale: dict[str, float | torch.Tensor] = {}
        self.exp_of: dict[str, tuple] = {}         # a glu_exp product -> (act, up, multiplier, emax)
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
        """A requantising consumer of a weight op (:func:`graph.requant_fn`, so V1 can cut it)."""
        return self.cheap(prefix, inputs, requant_fn(params), scale, note, {"requant": params})

    def last(self, name, *, calibrate_all=False):
        """``transformer._Builder.last``: the last position of ``name``; with ``calibrate_all`` the calibration
        keeps every position (the op itself takes the last), so the ops after it get the multipliers of the graph
        without pruning."""
        if not calibrate_all:
            return self.cheap("last", [name], _last_position, self.scale[name], "last position")
        n = self.name("last")
        self.ops.append(CheapOp(n, (name,), n + ".y", fn=_last_position, note="last position"))
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
            return self.cheap("norm", [r], lambda a, g=gain: _norm_int(a, g, False), 1.0 / self.norm_gain, "norm")
        x = self.env[r]
        sigma = _isqrt(torch.div((x * x).sum(-1, keepdim=True), d, rounding_mode="floor"))
        n = _rows((x.double() / sigma.double()).reshape(-1, d), _ROWS_NORM).abs().cpu()
        a = (torch.quantile(n, self.smooth_pct / 100.0, dim=0) if self.smooth_pct < 100
             else n.amax(0)).clamp_min(1e-3)
        wj = torch.stack([w.double().abs().amax(0) for w in consumers]).amax(0).clamp_min(1e-8)
        s_ = a.pow(self.smooth) / wj.pow(1 - self.smooth)
        g_real = ((INT8_MAX / float((a / s_).max())) / s_).clamp(1.0 / 64, float(1 << 14))
        gain = torch.round(g_real * (1 << 16)).to(torch.int64)
        return self.cheap("norm", [r], lambda a_, g=gain: _norm_int(a_, g, False), (1 << 16) / gain.double(),
                          "norm")

    def head_norm(self, z, heads: int, dh: int, gamma: torch.Tensor, prefix: str, note: str):
        """Qwen3's q/k norm: the benchmark's per-head integer RMSNorm with the gain ``round(G gamma_j 2^16)``;
        ``G`` maps the ``pct`` percentile of the RoPE pairs' norms of ``gamma_j x_j / rms`` to 127 (real scale
        ``1/G``). The input's scale does not matter: the norm divides it out."""
        x = self.env[z].reshape(-1, dh)
        sigma = _isqrt(torch.div((x * x).sum(-1, keepdim=True), dh, rounding_mode="floor"))
        normed = x.double() / sigma.double() * gamma.double().to(x.device)[None, :]
        G = INT8_MAX / max(_top(_pair_norms(normed, dh), self.pct), 1e-12)
        del normed
        gain = torch.round(G * gamma.double().cpu() * (1 << 16)).to(torch.int64)
        return self.cheap(prefix, [z], lambda a, g=gain, H=heads, dh=dh: _norm_int(
            a.reshape(*a.shape[:-1], H, dh), g, False).reshape(a.shape), 1.0 / G, note)

    def rope(self, z, heads: int, dh: int, theta: float, *, pruned: bool = False):
        """The benchmark's RoPE op (the scale is kept: a rotation); ``pruned``: ``z`` holds the last positions,
        whose angles come from the query's length."""
        if pruned:
            return self.cheap("rope", [z, "x"], lambda a, x, H=heads, dh=dh, th=theta: _rope(
                a.reshape(*a.shape[:-1], H, dh), th, x.shape[1] - a.shape[1]).reshape(a.shape), self.scale[z],
                "rope")
        return self.cheap("rope", [z], lambda a, H=heads, dh=dh, th=theta: _rope(
            a.reshape(*a.shape[:-1], H, dh), th).reshape(a.shape), self.scale[z], "rope")

    def attention(self, q, k, v, hq: int, hkv: int, dh: int):
        """``build_opt_from_hf``'s attention, with grouped-query heads: the scores' multiplier is
        ``2^30 16 s_q s_k / sqrt(dh)`` (Hugging Face's ``head_dim^-0.5`` in units of 1/16 nat), the output's maps the
        ``pct_att`` percentile of the raw ``P V`` to 127."""
        sq, sk, sv = self.scale[q], self.scale[k], self.scale[v]
        m_s = round((1 << SHIFT) * 16.0 * sq * sk / math.sqrt(dh))
        P = self.prob_max
        raw = _attention(self.env[q], self.env[k], self.env[v], m_s, None, hq, hkv, dh, P)
        s_raw = sv / float(P)    # probabilities 0..P x int8 values (transformer.PROB_MAX = 255; F2a's P_MAX)
        top = _top(raw.abs().reshape(-1, raw.shape[-1]).double(), self.pct_att) * s_raw
        del raw
        s_att = max(top, 1e-12) / INT8_MAX
        m_o = round((1 << SHIFT) * s_raw / s_att)
        if P == PROB_MAX:        # the default scale: no extra constant, so the digest is today's
            fn = lambda a, b_, c, ms=m_s, mo=m_o, H=hq, Hkv=hkv, dh=dh: _attention(a, b_, c, ms, mo, H, Hkv, dh)  # noqa: E731
        else:
            fn = lambda a, b_, c, ms=m_s, mo=m_o, H=hq, Hkv=hkv, dh=dh, P=P: _attention(  # noqa: E731
                a, b_, c, ms, mo, H, Hkv, dh, P)
        return self.cheap("attn", [q, k, v], fn, s_att, "attention")

    def swiglu(self, gate, up):
        """``silu(gate) * up``: the SiLU table from the gate's real scale, then the product requantised from real
        scales (the ``pct_glu`` percentile of ``|silu(gate) up|`` maps to 127)."""
        s_g, s_u = self.scale[gate], self.scale[up]
        xs = torch.arange(-128, 128, dtype=torch.float64) * s_g
        silu = xs * torch.sigmoid(xs)
        s_a = float(silu.abs().max()) / INT8_MAX
        table = torch.round(silu / s_a).clamp(-INT8_MAX, INT8_MAX).to(torch.int64)
        act = self.cheap("silu", [gate], lambda a, t=table: _lut(a, t), s_a, "silu")
        prod = (self.env[act] * self.env[up]).abs().reshape(-1, self.env[up].shape[-1]).double()
        if self.glu_exp:
            return self._swiglu_exp(act, up, prod, s_a, s_u)
        s_p = max(_top(prod, self.pct_glu) * s_a * s_u, 1e-12) / INT8_MAX
        del prod
        mm = torch.tensor(round((1 << SHIFT) * s_a * s_u / s_p), dtype=torch.int64)
        return self.cheap("glu", [act, up], lambda a, b_, m=mm: requant_mul(a, b_, m.to(a.device), SHIFT, -INT8_MAX,
                                                                            INT8_MAX), s_p, "swiglu")

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
        out = self.cheap("glu", [act, up], lambda a, b_, m=mm, E=emax: swiglu_exp(a, b_, m, E), s_p,
                         "swiglu, per-token exponent")
        self.exp_of[out] = (act, up, mm, emax)
        return out

    def residual_exp(self, r, z, prod):
        """``residual(r, z)`` for the down projection ``z`` of a :func:`swiglu_exp` product ``prod``: the per-token
        shift of :func:`residual_exp` (not a ``requant_fn`` op, so V1 leaves the down projection clear)."""
        act, up, mm, emax = self.exp_of[prod]
        m = round((1 << SHIFT) * self.scale[z] / self.scale[r])
        return self.cheap("res", [r, z, act, up], lambda a, b_, c, d_, gm=mm, m=m, E=emax: residual_exp(
            a, b_, c, d_, gm, m, E), self.scale[r], "residual add, per-token shift")


def _head_eq(h_int: torch.Tensor, s_in, w: torch.Tensor, b: torch.Tensor | None, heads: int, dh: int, pct: float):
    """``(diag(beta) w, beta b)``: each head's rows times ``beta_h = max_h' a_h' / a_h`` (at most ``2^8``), ``a_h``
    the ``pct`` percentile of ``|w h|`` over head ``h``'s outputs on the calibration rows. An RMSNorm over the head
    follows, which ``beta_h`` does not change."""
    x = _rows(h_int.reshape(-1, h_int.shape[-1]), _ROWS_NORM).double().cpu()
    w_eff = w.double() * (s_in.double().cpu()[None, :] if torch.is_tensor(s_in) else s_in)
    z = (x @ w_eff.T).abs().reshape(-1, heads, dh)
    del x, w_eff
    a = torch.tensor([max(_top(z[:, i], pct), 1e-30) for i in range(heads)], dtype=torch.float64)
    beta = (a.max() / a).clamp(1.0, 256.0).repeat_interleave(dh)
    return (w.double() * beta[:, None]).float(), (None if b is None else b.double() * beta)


def _hidden_max(model, calib_ids: torch.Tensor) -> float:
    """Largest ``|value|`` of the fp checkpoint's residual stream (the embedding output and every block's output)
    on the calibration windows, one window at a time, through forward hooks (no hidden state is kept)."""
    top = [0.0]

    def hook(mod, inp, out):
        out = out[0] if isinstance(out, tuple) else out
        top[0] = max(top[0], float(out.detach().abs().max()))

    inner = model.model
    hooks = [inner.embed_tokens.register_forward_hook(hook)] + [layer.register_forward_hook(hook)
                                                                for layer in inner.layers]
    try:
        with torch.no_grad():
            for i in range(calib_ids.shape[0]):
                inner(calib_ids[i:i + 1].to(model.device))
    finally:
        for h in hooks:
            h.remove()
    return top[0]


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
                        free_calib: bool = True, prob_max: int = PROB_MAX) -> tuple[IntGraph, dict]:
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

    ``device`` is where the integer calibration pass runs (the fp model stays where it is). Returns the graph and
    ``info``: the real scales (``logit_scale``, ``s_residual``), the knobs, the residual tensors at the block
    boundaries (``embed``, ``blocks``) and the ``DecoderConfig`` of the model's shape (``config``).
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

    # The residual stream's real scale: the fp checkpoint's hidden states never clip at RES_MAX on the calibration
    # windows; the embedding's int8 scale is c x s_residual, c the graph's "embed to residual" multiplier.
    emb = inner.embed_tokens.weight
    res_max = _hidden_max(model, calib_ids) * residual_headroom
    s_res = max(res_max / RES_MAX, float(emb.abs().max()) / (INT8_MAX * 8))

    B = _Builder(calib_ids.to(device or calib_ids.device), pct=pct, norm_gain=norm_gain, smooth=smooth,
                 smooth_pct=smooth_pct, pct_att=pct_att, pct_glu=pct_glu, glu_exp=glu_exp)
    assert 1 <= prob_max <= 2047, prob_max
    B.prob_max = prob_max
    embed_mult = max(1, round(float(emb.abs().max()) / INT8_MAX / s_res))
    # build_opt_from_hf's rounding: c rounds max|E| / (127 s_res) to the nearest integer, so for a ratio in
    # (k, k + 1/2), k >= 1, the largest entries clip at 127; ``embed_clip`` (> 1 then) records by how much
    embed_clip = float(emb.abs().max()) / (INT8_MAX * embed_mult * s_res)
    tok = B.mat("x", emb.T.float(), None, 1.0, layout="embed", s_w=embed_mult * s_res)
    r = B.cheap("scale", [tok], lambda a, c=embed_mult: a * c, s_res, "embed to residual")
    embed_name, blocks = r, []

    def bias(lin):
        return None if lin.bias is None else lin.bias.detach()

    for i, layer in enumerate(layers):
        prune = prune_last and i == len(layers) - 1
        at, mlp = layer.self_attn, layer.mlp
        g1 = layer.input_layernorm.weight.detach()
        wq, wk, wv = (p.weight.detach().float() * g1.float()[None, :] for p in (at.q_proj, at.k_proj, at.v_proj))
        bq, bk, bv = bias(at.q_proj), bias(at.k_proj), bias(at.v_proj)
        h = B.norm(r, d, [wq, wk, wv])
        qk_norm = hasattr(at, "q_norm")
        if qk_norm and head_eq:
            wq, bq = _head_eq(B.env[h], B.scale[h], wq, bq, hq, dh, pct)
            wk, bk = _head_eq(B.env[h], B.scale[h], wk, bk, hkv, dh, pct)
        rope_pairs = None if qk_norm else dh       # without q/k norm, RoPE reads the requantised values
        q = B.to_int8(B.mat(B.last(h, calibrate_all=True) if prune else h, wq, bq, B.scale[h]), pairs=rope_pairs)
        k = B.to_int8(B.mat(h, wk, bk, B.scale[h]), pairs=rope_pairs)
        v = B.to_int8(B.mat(h, wv, bv, B.scale[h]))
        del wq, wk, wv
        if qk_norm:
            q = B.head_norm(q, hq, dh, at.q_norm.weight.detach(), "qn", "q norm")
            k = B.head_norm(k, hkv, dh, at.k_norm.weight.detach(), "kn", "k norm")
        q = B.rope(q, hq, dh, theta, pruned=prune)
        k = B.rope(k, hkv, dh, theta)
        att = B.attention(q, k, v, hq, hkv, dh)
        r = B.residual(B.last(r, calibrate_all=True) if prune else r,
                       B.mat(att, at.o_proj.weight.detach().float(), bias(at.o_proj), B.scale[att]))

        g2 = layer.post_attention_layernorm.weight.detach()
        wg, wu = (p.weight.detach().float() * g2.float()[None, :] for p in (mlp.gate_proj, mlp.up_proj))
        h = B.norm(r, d, [wg, wu])
        gate = B.to_int8(B.mat(h, wg, bias(mlp.gate_proj), B.scale[h]))
        up = B.to_int8(B.mat(h, wu, bias(mlp.up_proj), B.scale[h]))
        del wg, wu
        prod = B.swiglu(gate, up)
        if glu_exp and mlp.down_proj.bias is not None:
            raise ValueError("glu_exp scales the down projection's claims per token, which a bias would not follow")
        z = B.mat(prod, mlp.down_proj.weight.detach().float(), bias(mlp.down_proj), B.scale[prod])
        r = B.residual_exp(r, z, prod) if glu_exp else B.residual(r, z)
        blocks.append(r)
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
            "glu_emax": [e for *_, e in B.exp_of.values()]}
    return IntGraph(B.ops, "x", logits), info


# ------------------------------------------------------------------ evaluation

def _identity(a):
    return a


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
        ops[lasts[0]] = CheapOp(o.name, o.inputs, o.output, fn=_identity, note=o.note, params=o.params)
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


def hf_residual_states(model, ids: torch.Tensor, n_layers: int | None = None) -> list[torch.Tensor]:
    """The fp checkpoint's residual stream at the same boundaries (the embedding's output, then each decoder
    layer's), float64 on the CPU, through forward hooks; ``n_layers``: the first blocks only."""
    out: list[torch.Tensor] = []

    def hook(mod, inp, o):
        o = o[0] if isinstance(o, tuple) else o
        out.append(o.detach().to("cpu", torch.float64))

    inner = model.model
    hooks = [inner.embed_tokens.register_forward_hook(hook)] + [
        layer.register_forward_hook(hook) for layer in list(inner.layers)[:n_layers]]
    try:
        with torch.no_grad():
            inner(ids.to(model.device))
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

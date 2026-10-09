"""Integer decoder-only transformers (GPT-2 / OPT / Llama / Qwen shapes).

These are used for *cost* measurements on the language models the literature
benchmarks.  Weights are random int8 -- the prover's and verifier's work, and
the proof size, depend only on the shapes, never on the weight values (the same
methodology Slalom used for its untrained ResNets).  Because the pretrained
checkpoints are not used, no fidelity (perplexity) claim is made for them.

Integer semantics (both parties compute these identically):

* residual stream: int32; each branch output is requantised into it and added;
* LayerNorm / RMSNorm: integer mean, integer square root, integer division;
* GELU / ReLU / SiLU: 256-entry look-up tables on int8 inputs;
* SwiGLU gate: int8 x int8 product, requantised;
* RoPE: Q14 cosine/sine tables, rounding shift;
* attention: exact ``q k^T``, integer softmax (exp via a 256-entry table in units
  of 1/16 nat, integer normalisation to 8-bit probabilities), exact ``P V``.

Only the weight products are ``MatOp`` s; everything else is recomputed by the
verifier, including attention, which costs it O(T^2 d) per layer.

``build_decoder(..., prune_last=True)`` (opt-in) builds the last block at the last position only,
but for its keys and values: only that position reaches the next-token logits, which are the same
integers.
"""

from __future__ import annotations

import dataclasses
import functools
import math
import os
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
import torch

from . import attention_kernels, native_kernels
from .graph import (INT8_MAX, CheapOp, IntGraph, MatOp, exact_matmul, int_scalar, mul_add_half, requant, requant_fn,
                    residual_add)

__all__ = ["DecoderConfig", "CONFIGS", "build_decoder", "decoder_param_count", "with_lm_positions", "greedy_tokens"]

SHIFT = 30
_FUSED_ATTN = os.environ.get("PVI_FUSED_ATTN", "1") != "0"
"""On a GPU with Triton, the int32 attention runs as one fused kernel (``attention_kernels``); 0: torch."""
RES_MAX = (1 << 22) - 1
ATTN_BYTES = 1 << 27
"""Budget for one ``[B, heads, T, T]`` score tensor (heads are grouped to fit): int32 scores
on the int32 path of :func:`_attention_heads`, int64 ones on its fallback."""
_EXP_TABLE = torch.tensor([round((1 << 15) * math.exp(-i / 16.0)) for i in range(256)], dtype=torch.int64)
_EXP_ZERO = 178
"""``_EXP_TABLE[i] == 0`` exactly for ``i >= _EXP_ZERO``."""
assert int(_EXP_TABLE[_EXP_ZERO - 1]) > 0 and not bool(_EXP_TABLE[_EXP_ZERO:].any())
_MASKED = -(1 << 30)
"""The int32 path's score of a masked entry: below every real score (``|s| <= 2**21``)."""
_LUT_MAX = 1 << 22


@dataclass(frozen=True)
class DecoderConfig:
    name: str
    d_model: int
    n_layers: int
    n_heads: int
    n_kv_heads: int
    head_dim: int
    d_ff: int
    vocab: int
    norm: str = "layernorm"      # or "rmsnorm"
    mlp: str = "gelu"            # "gelu" | "relu" | "swiglu"
    pos: str = "learned"         # "learned" | "rope"
    rope_theta: float = 10000.0
    bias: bool = True
    tied: bool = True
    qk_norm: bool = False
    max_pos: int = 2048
    embed_dim: int = 0           # OPT-350M: token embedding / LM head in 512 dims, projected to d_model


CONFIGS = {
    "gpt2": DecoderConfig("gpt2", 768, 12, 12, 12, 64, 3072, 50257, max_pos=1024),
    "opt-125m": DecoderConfig("opt-125m", 768, 12, 12, 12, 64, 3072, 50272, mlp="relu"),
    "opt-350m": DecoderConfig("opt-350m", 1024, 24, 16, 16, 64, 4096, 50272, mlp="relu", embed_dim=512),
    "opt-1.3b": DecoderConfig("opt-1.3b", 2048, 24, 32, 32, 64, 8192, 50272, mlp="relu"),
    "opt-2.7b": DecoderConfig("opt-2.7b", 2560, 32, 32, 32, 80, 10240, 50272, mlp="relu"),
    "opt-6.7b": DecoderConfig("opt-6.7b", 4096, 32, 32, 32, 128, 16384, 50272, mlp="relu"),
    "opt-13b": DecoderConfig("opt-13b", 5120, 40, 40, 40, 128, 20480, 50272, mlp="relu"),
    "llama2-7b": DecoderConfig("llama2-7b", 4096, 32, 32, 32, 128, 11008, 32000, norm="rmsnorm",
                               mlp="swiglu", pos="rope", bias=False, tied=False, max_pos=4096),
    "llama2-13b": DecoderConfig("llama2-13b", 5120, 40, 40, 40, 128, 13824, 32000, norm="rmsnorm",
                                mlp="swiglu", pos="rope", bias=False, tied=False, max_pos=4096),
    "qwen3-4b": DecoderConfig("qwen3-4b", 2560, 36, 32, 8, 128, 9728, 151936, norm="rmsnorm",
                              mlp="swiglu", pos="rope", rope_theta=1e6, bias=False, tied=True,
                              qk_norm=True, max_pos=40960),
    # the largest shapes of the sampled/zk literature (the report builds them with 1 and 2 blocks only)
    "opt-30b": DecoderConfig("opt-30b", 7168, 48, 56, 56, 128, 28672, 50272, mlp="relu"),
    "opt-66b": DecoderConfig("opt-66b", 9216, 64, 72, 72, 128, 36864, 50272, mlp="relu"),
    "llama2-70b": DecoderConfig("llama2-70b", 8192, 80, 64, 8, 128, 28672, 32000, norm="rmsnorm",
                                mlp="swiglu", pos="rope", bias=False, tied=False, max_pos=4096),
}


def decoder_param_count(cfg: DecoderConfig) -> int:
    """Weight-product parameters of the full model (norm gains excluded)."""
    d, f = cfg.d_model, cfg.d_ff
    q = cfg.n_heads * cfg.head_dim
    kv = cfg.n_kv_heads * cfg.head_dim
    b = 1 if cfg.bias else 0
    attn = d * (q + 2 * kv) + q * d + b * (q + 2 * kv + d)
    mlp = (3 if cfg.mlp == "swiglu" else 2) * d * f + b * (f + d)
    per_layer = attn + mlp
    e = cfg.embed_dim or d
    emb = cfg.vocab * e + (cfg.max_pos * d if cfg.pos == "learned" else 0)
    proj = 2 * e * d if cfg.embed_dim else 0
    head = 0 if cfg.tied else cfg.vocab * e
    return cfg.n_layers * per_layer + emb + proj + head


# The cheap ops below compute the integers of the straightforward formulas (kept in
# ``reference.py``) with fewer passes: fused multiply-adds, in-place steps on fresh
# tensors, flat gathers and cached tables.  Integer arithmetic is exact (int64 wraps
# identically in any order), so every output is the same tensor.

def _isqrt(s: torch.Tensor) -> torch.Tensor:
    """``max(1, floor(sqrt(s)))``: a float64 root, then one integer correction each way.

    The (tiny, one entry per row) CPU tensors go through numpy: the same correctly rounded
    IEEE root and the same corrections, at a third of the call cost.  A negative ``s`` (an
    int64 sum of squares that wrapped, which only adversarial claims reach) takes the torch
    steps, those of ``reference.isqrt``: its root is NaN, which numpy would warn about and
    cast to an integer by its own rules."""
    n = s.numpy() if s.device.type == "cpu" and s.dtype == torch.int64 else None
    if n is not None and (n.size == 0 or n.min() >= 0):
        r = np.floor(np.sqrt(n.astype(np.float64))).astype(np.int64)
        r += (r + 1) * (r + 1) <= n
        r -= r * r > n
        return torch.from_numpy(np.maximum(r, 1))
    r = torch.sqrt(s.to(torch.float64)).floor().to(torch.int64)
    r = torch.where((r + 1) * (r + 1) <= s, r + 1, r)
    r = torch.where(r * r > s, r - 1, r)
    return r.clamp_min(1)


def _norm_int(x: torch.Tensor, gain: torch.Tensor, center: bool, k: int = 16) -> torch.Tensor:
    """Integer LayerNorm (``center=True``) or RMSNorm over the last axis -> int8."""
    d = x.shape[-1]
    if center:
        mean = torch.div(x.sum(-1, keepdim=True) + d // 2, d, rounding_mode="floor")
        x = x - mean
    sigma = _isqrt(torch.div((x * x).sum(-1, keepdim=True), d, rounding_mode="floor"))
    g = gain.to(x.device)
    if x.dtype == g.dtype == torch.int64:
        num = torch.addcmul(sigma * (1 << (k - 1)), x, g)          # x * g + sigma * 2**(k-1)
    else:
        num = x * g + sigma * (1 << (k - 1))
    return num.div_(sigma << k, rounding_mode="floor").clamp_(-INT8_MAX, INT8_MAX)


def _lut(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    idx = x.clamp(-128, 127)
    idx += 128
    return torch.take(table.to(x.device), idx)          # table[idx] as one flat gather


def _rope_tables(t: int, dh: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    half = dh // 2
    inv = 1.0 / (theta ** (torch.arange(half, dtype=torch.float64) / half))
    ang = torch.arange(t, dtype=torch.float64)[:, None] * inv[None, :]
    return (torch.round(torch.cos(ang) * (1 << 14)).to(torch.int64),
            torch.round(torch.sin(ang) * (1 << 14)).to(torch.int64))


@lru_cache(maxsize=64)
def _rope_tables_full(t: int, dh: int, theta: float, device: str, offset: int = 0) -> tuple[torch.Tensor, torch.Tensor]:
    """``[cos | cos]`` and ``[-sin | sin]`` of the positions ``offset .. offset + T - 1`` as ``[1, T, 1, dh]``
    on ``device`` (cached, so a GPU gets them once, not per call; callers only read them)."""
    cos, sin = (a[offset:] for a in _rope_tables(offset + t, dh, theta))
    return (torch.cat([cos, cos], -1)[None, :, None, :].to(device),
            torch.cat([-sin, sin], -1)[None, :, None, :].to(device))


def _rope(x: torch.Tensor, theta: float, offset: int = 0) -> torch.Tensor:
    """x: ``[B, T, H, dh]`` int8 values at the positions ``offset .. offset + T - 1`` -> rotated int8
    values.

    ``[x1 cos - x2 sin | x2 cos + x1 sin] = x [cos | cos] + [x2 | x1] [-sin | sin]``."""
    t, dh = x.shape[1], x.shape[3]
    cos2, sin2 = _rope_tables_full(t, dh, theta, str(x.device), offset)
    h = dh // 2
    swapped = torch.cat([x[..., h:], x[..., :h]], -1)
    if x.dtype == torch.int64:
        out = torch.addcmul(int_scalar(1 << 13, str(x.device)), x, cos2).addcmul_(swapped, sin2)
    else:
        out = x * cos2 + swapped * sin2 + (1 << 13)
    out >>= 14
    return out.clamp_(-INT8_MAX, INT8_MAX)


@lru_cache(maxsize=32)
def _causal_notmask(t: int, rep: int, device: str, queries: int | None = None) -> torch.Tensor:
    """``~tril(ones(t, t))`` repeated ``rep`` times along the rows, on ``device`` (cached;
    callers only read it); with ``queries``, of its last ``queries`` rows only (the queries of
    the last positions)."""
    rows = t if queries is None else queries
    return (~torch.ones(t, t, dtype=torch.bool).tril())[t - rows:].repeat(rep, 1).to(device)


def _exp_cap(m_s: int) -> int:
    """The smallest score gap ``d`` whose exp-table index ``(d m_s + 2**29) >> 30`` reaches
    ``_EXP_ZERO``.  For ``m_s > 0`` the index grows with ``d``, so every larger gap looks up 0."""
    return -(-((_EXP_ZERO << SHIFT) - (1 << (SHIFT - 1))) // m_s)


def _int32_scores(m_s: int, dh: int, t: int) -> bool:
    """Whether the int32 path of :func:`_attention_heads` applies: ``|s| <= dh 128**2 <= 2**21``
    (``dh <= 128``); row sums ``tot <= T 2**15`` with ``2 tot`` and ``510 e + tot`` below
    ``2**31`` (``T < 2**15``); a gap table of at most ``2**22`` entries (so masked gaps, at
    least ``2**30 - 2**21``, lie past its end); and no int64 overflow in the fallback's
    ``d m_s`` (``d <= 2**22``)."""
    return 0 < m_s < 1 << 32 and dh <= 128 and t < 1 << 15 and _exp_cap(m_s) <= _LUT_MAX


@lru_cache(maxsize=256)
def _exp_lut(m_s: int, device: str) -> torch.Tensor:
    """``LUT[d] = EXP[clamp((d m_s + 2**29) >> 30, 0, 255)]`` for ``0 <= d <= _exp_cap(m_s)``
    (so ``LUT[-1] == 0``), int32 on ``device`` (cached; callers only read it)."""
    d = torch.arange(_exp_cap(m_s) + 1, dtype=torch.int64)
    idx = ((d * m_s + (1 << (SHIFT - 1))) >> SHIFT).clamp_(0, 255)
    return _EXP_TABLE[idx].to(device=device, dtype=torch.int32)


def _attention_core(q, k, v, lut: torch.Tensor, notmask: torch.Tensor) -> torch.Tensor:
    """The int32 path of :func:`_attention_heads` (raw ``P V``, int64).

    The same integers as the fallback, in half-width passes: the scores are exact in float32
    (``|s| <= 2**21``) and kept as int32; a masked score is ``_MASKED``, so its gap from the
    row maximum is past the end of ``lut`` and looks up the 0 that the fallback's second mask
    pass writes; the look-up ``LUT[gap]`` replaces the int64 ``gap * m_s`` steps; and ``P V``
    is ONE float32 GEMM, since a row of probabilities sums to at most ``255 + T/2`` and every
    partial sum is below ``(255 + T/2) 128 < 2**24``."""
    s = torch.matmul(q.to(torch.float32), k.to(torch.float32).transpose(-1, -2)).to(torch.int32)
    s.masked_fill_(notmask, _MASKED)
    d = torch.sub(s.amax(-1, keepdim=True), s, out=s)            # gaps, >= 0
    e = lut[d.clamp_(max=lut.shape[0] - 1)]                       # 0 .. 2**15
    del s, d
    tot = e.sum(-1, keepdim=True, dtype=torch.int32)              # >= 2**15: the row maximum
    p = torch.add(tot, e, alpha=510).div_(tot * 2, rounding_mode="floor")   # 0..255
    del e
    return torch.matmul(p.to(torch.float32), v.to(torch.float32)).to(torch.int64)


@lru_cache(maxsize=None)
def _exp_table(device: str) -> torch.Tensor:
    """``_EXP_TABLE`` on ``device`` (cached; callers only read it)."""
    return _EXP_TABLE.to(device)


def _attention_int64(q, k, v, m_s: int, notmask: torch.Tensor) -> torch.Tensor:
    """The fallback of :func:`_attention_heads` (raw ``P V``): int64 scores and gaps."""
    s = exact_matmul(q, k.transpose(-1, -2), max_w=128, max_x=128)       # [B,H,rep*T,T]
    s.masked_fill_(notmask, -(1 << 40))
    d = s.amax(-1, keepdim=True) - s
    del s
    idx = mul_add_half(d.clamp_(max=1 << 31), int_scalar(m_s, str(q.device)), SHIFT)
    del d
    idx >>= SHIFT
    e = torch.take(_exp_table(str(q.device)), idx.clamp_(0, 255)).masked_fill_(notmask, 0)
    del idx
    tot = e.sum(-1, keepdim=True)
    p = torch.addcmul(tot, e, int_scalar(510, str(q.device))).div_(2 * tot, rounding_mode="floor")   # 0..255
    del e
    return exact_matmul(p, v, max_w=256, max_x=128)                      # [B,H,rep*T,dh]


def _attention_heads(q, k, v, m_s: int, rep: int = 1) -> torch.Tensor:
    """Integer causal attention for a group of heads: the raw ``P V`` (int64), which
    :func:`_attention` requantises into its output.  ``k`` and ``v`` are ``[B, H, T, dh]``, and ``q``
    ``[B, H, rep*Tq, dh]``: the queries of the last ``Tq <= T`` positions (the query at position
    ``T - Tq + i`` attends to the keys ``0 .. T - Tq + i``), of the ``rep`` query heads that share each
    key/value head stacked along the rows (grouped-query attention), where every row is the same dot
    products and the same row-wise softmax as with ``k, v`` repeated.
    Runs in int32 (:func:`_attention_core`) where :func:`_int32_scores` allows, else in
    int64; both give the integers of ``reference.attention_heads(..., m_o=None)``."""
    t, dh = k.shape[2], k.shape[3]
    dev = str(q.device)
    if _int32_scores(m_s, dh, t):
        if q.is_cuda and _FUSED_ATTN and attention_kernels.available(q.device):   # the same integers, no T x T tensors
            return attention_kernels.attention_core(q, k, v, _exp_lut(m_s, dev), q.shape[2] // rep)
        if q.device.type == "cpu" and native_kernels.available():                # a CPU verifier: native, exact
            return native_kernels.attention_core(q, k, v, _exp_lut(m_s, dev), q.shape[2] // rep)
        return _attention_core(q, k, v, _exp_lut(m_s, dev), _causal_notmask(t, rep, dev, q.shape[2] // rep))
    return _attention_int64(q, k, v, m_s, _causal_notmask(t, rep, dev, q.shape[2] // rep))


def _attention(q, k, v, m_s: int, m_o: int | None, n_heads: int, n_kv: int, dh: int) -> torch.Tensor:
    """Integer causal attention.  q ``[B,Tq,Hq*dh]`` (the queries of the last ``Tq <= T`` positions),
    k/v ``[B,T,Hkv*dh]`` -> ``[B,Tq,Hq*dh]``.

    Heads are independent, so they are processed a few at a time to bound the
    ``Tq x T`` score tensors (0.5 GiB per 32 heads at T=2048 in int32); the result is
    the same integers as processing them all at once.  When all heads fit in one group,
    grouped-query attention stacks each key/value head's query heads instead of
    repeating ``k`` and ``v``.  The output requantisation writes straight into the
    final ``[B, Tq, heads, dh]`` layout.
    """
    b, tq, _ = q.shape
    t = k.shape[1]
    q = q.reshape(b, tq, n_heads, dh).transpose(1, 2)
    k = k.reshape(b, t, n_kv, dh).transpose(1, 2)
    v = v.reshape(b, t, n_kv, dh).transpose(1, 2)
    out = torch.empty(b, tq, n_heads, dh, dtype=torch.int64, device=q.device)
    out_h = out.transpose(1, 2)                                     # [B, heads, Tq, dh] view
    width = 4 if _int32_scores(m_s, dh, t) else 8                   # bytes per score
    group = max(1, ATTN_BYTES // (width * b * tq * t))              # heads per group
    if n_kv != n_heads and group >= n_heads:
        rep = n_heads // n_kv   # query heads j*rep .. j*rep+rep-1 read kv head j (as repeat_interleave)
        raw = _attention_heads(q.reshape(b, n_kv, rep * tq, dh), k, v, m_s, rep)
        _write_output(out_h, raw.reshape(b, n_heads, tq, dh), m_o)
    else:
        if n_kv != n_heads:
            k = k.repeat_interleave(n_heads // n_kv, 1)
            v = v.repeat_interleave(n_heads // n_kv, 1)
        for h in range(0, n_heads, group):
            _write_output(out_h[:, h:h + group],
                          _attention_heads(q[:, h:h + group], k[:, h:h + group], v[:, h:h + group], m_s), m_o)
    return out.reshape(b, tq, n_heads * dh)


def _write_output(dst: torch.Tensor, raw: torch.Tensor, m_o: int | None) -> None:
    """``dst[...] = requant(raw, m_o)``, or ``raw`` itself when ``m_o`` is None (calibration)."""
    if m_o is None:
        dst.copy_(raw)
    else:
        requant(raw, int_scalar(m_o, str(raw.device)), SHIFT, -INT8_MAX, INT8_MAX, out=dst)


def _last_position(a: torch.Tensor) -> torch.Tensor:
    """``[B, T, d] -> [B, 1, d]``: the last position."""
    return a[:, -1:, :].contiguous()


def _last_positions(a: torch.Tensor, n: int) -> torch.Tensor:
    """``[B, T, d] -> [B, n, d]``: the last ``n`` positions."""
    return a[:, -n:, :].contiguous()


def with_lm_positions(graph: IntGraph, n: int) -> IntGraph:
    """``graph`` (a decoder of :func:`build_decoder` or ``real_weights``) with its logits at the last
    ``n`` positions instead of the last one: every "last position" slice -- before the LM head, and in
    a pruned last block -- takes the last ``n``.  The ops, weights and multipliers are ``graph``'s.

    Every op but causal attention is per position and no constant depends on the input, so the logits
    at position ``T - n + j`` are those that ``graph`` gives on the prefix ``x[:, :T - n + j + 1]``.  A
    query is then a prompt followed by the ``n - 1`` tokens generated after it, proved as one prefill:
    ``meta["lm_positions"] = n`` makes the verifier check that each of those tokens is the greedy choice
    of the logits before it (:meth:`protocol.Verifier.check_tokens`), and the last logits give token
    ``n``."""
    if n < 1:
        raise ValueError("at least one position")
    ops = [dataclasses.replace(op, fn=functools.partial(_last_positions, n=n))
           if isinstance(op, CheapOp) and op.note == "last position" else op for op in graph.ops]
    return IntGraph(ops, graph.input_name, graph.output_name, {**graph.meta, "lm_positions": n})


def greedy_tokens(graph: IntGraph, prompt: torch.Tensor, steps: int) -> torch.Tensor:
    """``prompt [B, P]`` followed by the ``steps`` tokens ``graph`` (logits at the last position) generates
    greedily, each the first largest logit: ``[B, P + steps]``.  One forward pass per token, without a
    KV cache: the integers do not depend on how they are computed."""
    x = prompt
    for _ in range(steps):
        env, _ = graph.forward(x, free=True)
        nxt = env[graph.output_name][:, -1].argmax(-1, keepdim=True)
        x = torch.cat([x, nxt.to(x.device, x.dtype)], 1)
    return x


def _residual(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    """``(a + ((b * m + 2**29) >> 30)).clamp(-RES_MAX, RES_MAX)`` in one output buffer."""
    return residual_add(a, b, m.to(b.device), SHIFT, RES_MAX)


class _Builder:
    """Adds ops one at a time while running them on a calibration input, so
    every requantisation multiplier is set from real value statistics."""

    def __init__(self, tokens: torch.Tensor, seed: int) -> None:
        self.ops: list = []
        self.env = {"x": tokens}
        self.g = torch.Generator().manual_seed(seed)
        self._n = 0

    def name(self, p):
        self._n += 1
        return f"{p}{self._n}"

    def _run(self, op):
        self.ops.append(op)
        if isinstance(op, MatOp):
            xin = self.env[op.inputs[0]]
            self.env[op.output] = op.fold(op.compute(xin), xin)
        else:
            self.env[op.output] = op.fn(*[self.env[n] for n in op.inputs])
        return op.output

    def weight(self, rows, cols, std=40.0):
        w = torch.randn(rows, cols, generator=self.g) * std
        return w.round().clamp(-INT8_MAX, INT8_MAX).to(torch.int8)

    def mat(self, inp, rows, cols, bias, layout="linear"):
        n = self.name("mm")
        b = (torch.randint(-(1 << 12), 1 << 12, (rows,), generator=self.g, dtype=torch.int64)
             if bias else None)
        return self._run(MatOp(n, (inp,), n + ".z", weight=self.weight(rows, cols), bias=b, layout=layout))

    def cheap(self, prefix, inputs, fn, note="", params=None):
        n = self.name(prefix)
        return self._run(CheapOp(n, tuple(inputs), n + ".y", fn=fn, note=note, params=params or {}))

    def requant(self, prefix, inputs, note, **params):
        """A requantising consumer of a weight op (:func:`graph.requant_fn`, so V1 can cut it)."""
        return self.cheap(prefix, inputs, requant_fn(params), note, {"requant": params})

    def last(self, name, *, calibrate_all=False):
        """The last position of ``name``.  With ``calibrate_all`` the calibration keeps every position
        (the op itself takes the last), so the ops after it are calibrated -- get their multipliers --
        and draw their weights as in the graph without it."""
        if not calibrate_all:
            return self.cheap("last", [name], _last_position, "last position")
        n = self.name("last")
        self.ops.append(CheapOp(n, (name,), n + ".y", fn=_last_position, note="last position"))
        self.env[n + ".y"] = self.env[name]
        return n + ".y"

    def std(self, name):
        return max(float(self.env[name].double().std()), 1e-6)

    def to_int8(self, z, target=24.0):
        m = round((1 << SHIFT) * target / self.std(z))
        return self.requant("rq", [z], "requant", kind="requant", mult=m, shift=SHIFT, lo=-INT8_MAX, hi=INT8_MAX)

    def residual(self, r, z, target=1024.0):
        m = round((1 << SHIFT) * target / self.std(z))
        return self.requant("res", [r, z], "residual add", kind="residual", mult=m, shift=SHIFT, res_max=RES_MAX)

    def norm(self, r, d, center):
        gain = torch.full((d,), round(32 * (1 << 16)), dtype=torch.int64)  # gamma=1 at scale 1/32
        return self.cheap("norm", [r], lambda a, g=gain, c=center: _norm_int(a, g, c), "norm")


def build_decoder(cfg: DecoderConfig, *, n_layers: int | None = None, calib_tokens: int = 16,
                  seed: int = 0, prune_last: bool = False) -> IntGraph:
    """An integer decoder with ``n_layers`` blocks (default: all of them).

    The LM head is applied to the last position only: one query is a prompt and
    its answer is the next-token distribution.  ``prune_last`` (opt-in): the last block computes
    q, the attention output and its projection, the residuals and the MLP at the last position only
    -- k and v at every position, which its one query row attends to -- since only that position
    reaches the logits; those ops claim one column instead of ``T``.  The pruned block is
    calibrated on every position (:meth:`_Builder.last`), so the graph has the weights and
    multipliers of the graph without pruning, and gives the same logits (tested).
    """
    layers = cfg.n_layers if n_layers is None else n_layers
    g = torch.Generator().manual_seed(seed + 1)
    tokens = torch.randint(0, cfg.vocab, (1, calib_tokens), generator=g)
    B = _Builder(tokens, seed)
    d, dh, hq, hkv = cfg.d_model, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
    center = cfg.norm == "layernorm"

    e = cfg.embed_dim or d
    tok = B.mat("x", e, cfg.vocab, False, layout="embed")
    if cfg.embed_dim:  # OPT-350M's project_in (512 -> d_model)
        tok = B.to_int8(B.mat(tok, d, e, False))
    if cfg.pos == "learned":
        pos_ids = B.cheap("pos", ["x"], lambda x: torch.arange(x.shape[1], device=x.device)
                          .expand(x.shape[0], -1).contiguous(), "position ids")
        pos = B.mat(pos_ids, d, cfg.max_pos, False, layout="embed")
        r = B.residual(B.cheap("scale", [tok], lambda a: a * 8, "embed to residual"), pos)
    else:
        r = B.cheap("scale", [tok], lambda a: a * 8, "embed to residual")

    gelu = torch.tensor([round(24 * (x / 24) * 0.5 * (1 + math.erf((x / 24) / math.sqrt(2))))
                         for x in range(-128, 128)], dtype=torch.int64)
    relu = torch.tensor([max(0, x) for x in range(-128, 128)], dtype=torch.int64)
    silu = torch.tensor([round(24 * (x / 24) / (1 + math.exp(-(x / 24)))) for x in range(-128, 128)],
                        dtype=torch.int64)

    for i in range(layers):
        prune = prune_last and i == layers - 1
        h = B.norm(r, d, center)
        q = B.to_int8(B.mat(B.last(h, calibrate_all=True) if prune else h, hq * dh, d, cfg.bias))
        k = B.to_int8(B.mat(h, hkv * dh, d, cfg.bias))
        v = B.to_int8(B.mat(h, hkv * dh, d, cfg.bias))
        if cfg.qk_norm:
            gh = torch.full((dh,), round(32 * (1 << 16)), dtype=torch.int64)
            q = B.cheap("qn", [q], lambda a, g=gh, H=hq: _norm_int(a.reshape(*a.shape[:-1], H, dh), g, False)
                        .reshape(a.shape), "q norm")
            k = B.cheap("kn", [k], lambda a, g=gh, H=hkv: _norm_int(a.reshape(*a.shape[:-1], H, dh), g, False)
                        .reshape(a.shape), "k norm")
        if cfg.pos == "rope":
            th = cfg.rope_theta
            if prune:        # q at the last Tq positions: the angles of positions T - Tq .. T - 1
                q = B.cheap("rope", [q, "x"], lambda a, x, H=hq, th=th: _rope(
                    a.reshape(*a.shape[:-1], H, dh), th, x.shape[1] - a.shape[1]).reshape(a.shape), "rope")
            else:
                q = B.cheap("rope", [q], lambda a, H=hq, th=th: _rope(a.reshape(*a.shape[:-1], H, dh), th)
                            .reshape(a.shape), "rope")
            k = B.cheap("rope", [k], lambda a, H=hkv, th=th: _rope(a.reshape(*a.shape[:-1], H, dh), th)
                        .reshape(a.shape), "rope")
        # softmax temperature: logits of std ~1 nat, in units of 1/16 nat
        qs, ks = B.std(q), B.std(k)
        m_s = round((1 << SHIFT) * 16.0 / (qs * ks * math.sqrt(dh)))
        raw = _attention(B.env[q], B.env[k], B.env[v], m_s, None, hq, hkv, dh)
        m_o = round((1 << SHIFT) * 24.0 / max(float(raw.double().std()), 1e-6))
        att = B.cheap("attn", [q, k, v], lambda a, b_, c, ms=m_s, mo=m_o: _attention(a, b_, c, ms, mo, hq, hkv, dh),
                      "attention")
        r = B.residual(B.last(r, calibrate_all=True) if prune else r, B.mat(att, d, hq * dh, cfg.bias))
        h = B.norm(r, d, center)
        if cfg.mlp == "swiglu":
            gate = B.to_int8(B.mat(h, cfg.d_ff, d, cfg.bias))
            up = B.to_int8(B.mat(h, cfg.d_ff, d, cfg.bias))
            act = B.cheap("silu", [gate], lambda a, t=silu: _lut(a, t), "silu")
            mm = torch.tensor(round((1 << SHIFT) * 24.0 / max(B.std(act) * B.std(up), 1e-6)), dtype=torch.int64)
            prod = B.cheap("glu", [act, up], lambda a, b_, m=mm: requant(a * b_, m.to(a.device), SHIFT, -INT8_MAX,
                                                                         INT8_MAX), "swiglu")
        else:
            f1 = B.to_int8(B.mat(h, cfg.d_ff, d, cfg.bias))
            prod = B.cheap("act", [f1], lambda a, t=(gelu if cfg.mlp == "gelu" else relu): _lut(a, t), cfg.mlp)
        r = B.residual(r, B.mat(prod, d, cfg.d_ff, cfg.bias))

    h = B.last(B.norm(r, d, center))       # (in a pruned graph the one position of its last block)
    if cfg.embed_dim:  # OPT-350M's project_out (d_model -> 512)
        h = B.to_int8(B.mat(h, e, d, False))
    logits = B.mat(h, cfg.vocab, e, False)
    return IntGraph(B.ops, "x", logits)

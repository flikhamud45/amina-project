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
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from .graph import INT8_MAX, CheapOp, IntGraph, MatOp, exact_matmul, requant

__all__ = ["DecoderConfig", "CONFIGS", "build_decoder", "decoder_param_count"]

SHIFT = 30
RES_MAX = (1 << 22) - 1
ATTN_BYTES = 1 << 27
"""Budget for one int64 ``[B, heads, T, T]`` score tensor (heads are grouped to fit)."""
_EXP_TABLE = torch.tensor([round((1 << 15) * math.exp(-i / 16.0)) for i in range(256)], dtype=torch.int64)


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
    # beyond the report (strong-GPU plan): the largest shapes of the sampled/zk literature
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


def _isqrt(s: torch.Tensor) -> torch.Tensor:
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
    num = x * g + sigma * (1 << (k - 1))
    return torch.div(num, sigma << k, rounding_mode="floor").clamp(-INT8_MAX, INT8_MAX)


def _lut(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return table.to(x.device)[(x.clamp(-128, 127) + 128)]


def _rope_tables(t: int, dh: int, theta: float) -> tuple[torch.Tensor, torch.Tensor]:
    half = dh // 2
    inv = 1.0 / (theta ** (torch.arange(half, dtype=torch.float64) / half))
    ang = torch.arange(t, dtype=torch.float64)[:, None] * inv[None, :]
    return (torch.round(torch.cos(ang) * (1 << 14)).to(torch.int64),
            torch.round(torch.sin(ang) * (1 << 14)).to(torch.int64))


def _rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    """x: ``[B, T, H, dh]`` int8 values -> rotated int8 values."""
    t, dh = x.shape[1], x.shape[3]
    cos, sin = (a.to(x.device)[None, :, None, :] for a in _rope_tables(t, dh, theta))
    x1, x2 = x[..., : dh // 2], x[..., dh // 2:]
    half = 1 << 13
    y1 = (x1 * cos - x2 * sin + half) >> 14
    y2 = (x2 * cos + x1 * sin + half) >> 14
    return torch.cat([y1, y2], -1).clamp(-INT8_MAX, INT8_MAX)


def _attention_heads(q, k, v, m_s: int, m_o: int | None) -> torch.Tensor:
    """Integer causal attention for a group of heads: ``[B,H,T,dh]`` each."""
    t = q.shape[2]
    s = exact_matmul(q, k.transpose(-1, -2), max_w=128, max_x=128)       # [B,H,T,T]
    mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    s = s.masked_fill(~mask, -(1 << 40))
    d = (s.amax(-1, keepdim=True) - s).clamp(max=1 << 31)
    del s
    idx = ((d * m_s + (1 << (SHIFT - 1))) >> SHIFT).clamp(0, 255)
    del d
    e = _EXP_TABLE.to(q.device)[idx].masked_fill(~mask, 0)
    del idx
    tot = e.sum(-1, keepdim=True)
    p = torch.div(e * 510 + tot, 2 * tot, rounding_mode="floor")        # 0..255
    del e
    o = exact_matmul(p, v, max_w=256, max_x=128)                         # [B,H,T,dh]
    if m_o is not None:  # m_o=None returns the raw product, for calibration only
        o = requant(o, torch.tensor(m_o, device=q.device), SHIFT, -INT8_MAX, INT8_MAX)
    return o


def _attention(q, k, v, m_s: int, m_o: int | None, n_heads: int, n_kv: int, dh: int) -> torch.Tensor:
    """Integer causal attention.  q ``[B,T,Hq*dh]``, k/v ``[B,T,Hkv*dh]`` -> ``[B,T,Hq*dh]``.

    Heads are independent, so they are processed a few at a time to bound the
    ``T x T`` score tensors (1 GiB per 32 heads at T=2048); the result is the
    same integers as processing them all at once.
    """
    b, t, _ = q.shape
    q = q.reshape(b, t, n_heads, dh).transpose(1, 2)
    k = k.reshape(b, t, n_kv, dh).transpose(1, 2)
    v = v.reshape(b, t, n_kv, dh).transpose(1, 2)
    if n_kv != n_heads:
        rep = n_heads // n_kv
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
    group = max(1, ATTN_BYTES // (8 * b * t * t))       # heads per group
    outs = [_attention_heads(q[:, h:h + group], k[:, h:h + group], v[:, h:h + group], m_s, m_o)
            for h in range(0, n_heads, group)]
    o = torch.cat(outs, 1)
    return o.transpose(1, 2).reshape(b, t, n_heads * dh)


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

    def cheap(self, prefix, inputs, fn, note=""):
        n = self.name(prefix)
        return self._run(CheapOp(n, tuple(inputs), n + ".y", fn=fn, note=note))

    def std(self, name):
        return max(float(self.env[name].double().std()), 1e-6)

    def to_int8(self, z, target=24.0):
        m = torch.tensor(round((1 << SHIFT) * target / self.std(z)), dtype=torch.int64)
        return self.cheap("rq", [z], lambda a, m=m: requant(a, m.to(a.device), SHIFT, -INT8_MAX, INT8_MAX),
                          "requant")

    def residual(self, r, z, target=1024.0):
        m = torch.tensor(round((1 << SHIFT) * target / self.std(z)), dtype=torch.int64)
        half = 1 << (SHIFT - 1)
        return self.cheap("res", [r, z], lambda a, b, m=m: (a + ((b * m.to(b.device) + half) >> SHIFT))
                          .clamp(-RES_MAX, RES_MAX), "residual add")

    def norm(self, r, d, center):
        gain = torch.full((d,), round(32 * (1 << 16)), dtype=torch.int64)  # gamma=1 at scale 1/32
        return self.cheap("norm", [r], lambda a, g=gain, c=center: _norm_int(a, g, c), "norm")


def build_decoder(cfg: DecoderConfig, *, n_layers: int | None = None, calib_tokens: int = 16,
                  seed: int = 0) -> IntGraph:
    """An integer decoder with ``n_layers`` blocks (default: all of them).

    The LM head is applied to the last position only: one query is a prompt and
    its answer is the next-token distribution.
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

    for _ in range(layers):
        h = B.norm(r, d, center)
        q = B.to_int8(B.mat(h, hq * dh, d, cfg.bias))
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
        r = B.residual(r, B.mat(att, d, hq * dh, cfg.bias))
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

    h = B.norm(r, d, center)
    h = B.cheap("last", [h], lambda a: a[:, -1:, :].contiguous(), "last position")
    if cfg.embed_dim:  # OPT-350M's project_out (d_model -> 512)
        h = B.to_int8(B.mat(h, e, d, False))
    logits = B.mat(h, cfg.vocab, e, False)
    return IntGraph(B.ops, "x", logits)

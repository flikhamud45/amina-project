"""The straightforward definitions the fast verifier routines are tested against.

The verifier's modular products, codeword evaluation, right-hand sides, range checks,
leaf hashing, checks and cheap operations were rewritten for speed (fewer passes,
batched products, numpy for small operands).  This module keeps the code they replaced
as the specification (attention without grouping the heads, which gives the same
integers): ``tests/test_fast_verifier.py`` checks that every fast routine returns the
same integers, and the checks the same verdicts, on edge values and on real graphs.
Nothing in the protocol calls it.
"""

from __future__ import annotations

import torch

from .commitment import column_leaf, vandermonde_columns, verify_multiproof
from .field import P, small_matmul_mod, to_field
from .graph import INT8_MAX, MatOp, exact_matmul
from .transformer import _EXP_TABLE, RES_MAX, SHIFT, _rope_tables


# -- field and commitment -------------------------------------------------------------------

def field_matmul_mod(left: torch.Tensor, right: torch.Tensor, *, chunk: int = 1 << 10) -> torch.Tensor:
    """``left @ right mod P``: one float64 product per 11-bit limb of ``left`` and per block
    of ``chunk`` rows, each reduced and accumulated mod ``P``."""
    left = to_field(left)
    right = to_field(right)
    limbs = [(left >> (11 * i)) & 0x7FF for i in range(3)]
    k = left.shape[-1]
    total = None
    for i, limb in enumerate(limbs):
        acc = None
        for start in range(0, k, chunk):
            a = limb[..., start:start + chunk].to(torch.float64)
            b = right[start:start + chunk].to(torch.float64)
            part = torch.remainder((a @ b).round().to(torch.int64), P)
            acc = part if acc is None else (acc + part) % P
        acc = (acc * pow(2, 11 * i, P)) % P
        total = acc if total is None else (total + acc) % P
    return total


def codeword_at(u: torch.Tensor, n_points: int, columns: torch.Tensor) -> torch.Tensor:
    """``Enc(u)[:, columns]`` through the Vandermonde block of the opened columns."""
    return field_matmul_mod(u, vandermonde_columns(n_points, u.shape[-1], columns, u.device))


def column_leaves(tag: bytes, indices: list[int], opened: torch.Tensor) -> dict[int, bytes]:
    col_np = opened.to(torch.int64).cpu().numpy()
    return {c: column_leaf(tag, c, col_np[:, j]) for j, c in enumerate(indices)}


# -- protocol -------------------------------------------------------------------------------

def rhs(op: MatOp, u: torch.Tensor, xin: torch.Tensor) -> torch.Tensor:
    """``u^T [X ; 1]`` over the field, shape ``[r, M]``."""
    k = op.n_in
    if op.layout == "embed":
        out = u[:, xin.reshape(-1)]
    else:
        x = op.unfold(xin)  # [K, M] small ints
        out = small_matmul_mod(x.T.contiguous(), u[:, :k].T.contiguous()).T
    if op.has_bias:
        out = (out + u[:, k:k + 1]) % P
    return out


def in_range(z: torch.Tensor, bound: int) -> bool:
    return not bool(((z >= bound) | (z <= -bound)).any())


def in_field(a: torch.Tensor) -> bool:
    return not bool(((a < 0) | (a >= P)).any())


def check_products(verifier, claims, inputs, chis, us) -> bool:
    """``Verifier.check_products``, op by op."""
    for op in verifier.graph.mat_ops:
        u = us.get(op.name)
        if (u is None or u.dtype != torch.int64 or tuple(u.shape) != (verifier.params.reps, op.row_length)
                or not in_field(u)):
            return False
        lhs = field_matmul_mod(chis[op.name], to_field(claims[op.name]))
        if not torch.equal(lhs, rhs(op, us[op.name], inputs[op.name])):
            return False
    return True


def check_columns(verifier, chis, us, cols, openings) -> str | None:
    """``Verifier.check_columns``, op by op."""
    for op in verifier.graph.mat_ops:
        pub = verifier.publics[op.name]
        opened, proof = openings[op.name]
        idx = cols[op.name]
        if (tuple(opened.shape) != (pub.n_rows, len(idx)) or len(proof) > len(idx) * pub.depth
                or opened.dtype != torch.int64 or not in_field(opened)
                or us[op.name].shape[1] != pub.row_length):
            return "columns_shape"
        dev = us[op.name].device
        enc = field_matmul_mod(us[op.name], vandermonde_columns(pub.n_points, pub.row_length, idx, dev))
        if not torch.equal(field_matmul_mod(chis[op.name], opened.to(dev)), enc):
            return "columns_code"
        if not verify_multiproof(pub.root, pub.depth, column_leaves(pub.tag, idx.tolist(), opened), proof):
            return "columns_merkle"
    return None


# -- graph and cheap operations -----------------------------------------------------------------

def fold(op: MatOp, z: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``MatOp.fold`` with contiguous outputs."""
    if op.layout == "conv":
        k, s, p = op.conv
        b, _, h, w = x.shape
        ho, wo = (h + 2 * p - k) // s + 1, (w + 2 * p - k) // s + 1
        return z.reshape(op.n_rows, b, ho, wo).permute(1, 0, 2, 3).contiguous()
    if op.layout == "embed":
        return z.T.reshape(*x.shape, op.n_rows).contiguous()
    return z.T.reshape(*x.shape[:-1], op.n_rows).contiguous()


def requant(z: torch.Tensor, mult: torch.Tensor, shift: int, lo: int, hi: int) -> torch.Tensor:
    out = (z * mult + (1 << (shift - 1))) >> shift
    return out.clamp(lo, hi)


def isqrt(s: torch.Tensor) -> torch.Tensor:
    r = torch.sqrt(s.to(torch.float64)).floor().to(torch.int64)
    r = torch.where((r + 1) * (r + 1) <= s, r + 1, r)
    r = torch.where(r * r > s, r - 1, r)
    return r.clamp_min(1)


def norm_int(x: torch.Tensor, gain: torch.Tensor, center: bool, k: int = 16) -> torch.Tensor:
    d = x.shape[-1]
    if center:
        mean = torch.div(x.sum(-1, keepdim=True) + d // 2, d, rounding_mode="floor")
        x = x - mean
    sigma = isqrt(torch.div((x * x).sum(-1, keepdim=True), d, rounding_mode="floor"))
    g = gain.to(x.device)
    num = x * g + sigma * (1 << (k - 1))
    return torch.div(num, sigma << k, rounding_mode="floor").clamp(-INT8_MAX, INT8_MAX)


def lut(x: torch.Tensor, table: torch.Tensor) -> torch.Tensor:
    return table.to(x.device)[(x.clamp(-128, 127) + 128)]


def rope(x: torch.Tensor, theta: float) -> torch.Tensor:
    t, dh = x.shape[1], x.shape[3]
    cos, sin = (a.to(x.device)[None, :, None, :] for a in _rope_tables(t, dh, theta))
    x1, x2 = x[..., : dh // 2], x[..., dh // 2:]
    half = 1 << 13
    y1 = (x1 * cos - x2 * sin + half) >> 14
    y2 = (x2 * cos + x1 * sin + half) >> 14
    return torch.cat([y1, y2], -1).clamp(-INT8_MAX, INT8_MAX)


def attention_heads(q, k, v, m_s: int, m_o: int | None) -> torch.Tensor:
    t = q.shape[2]
    s = exact_matmul(q, k.transpose(-1, -2), max_w=128, max_x=128)       # [B,H,T,T]
    mask = torch.ones(t, t, dtype=torch.bool, device=q.device).tril()
    s = s.masked_fill(~mask, -(1 << 40))
    d = (s.amax(-1, keepdim=True) - s).clamp(max=1 << 31)
    idx = ((d * m_s + (1 << (SHIFT - 1))) >> SHIFT).clamp(0, 255)
    e = _EXP_TABLE.to(q.device)[idx].masked_fill(~mask, 0)
    tot = e.sum(-1, keepdim=True)
    p = torch.div(e * 510 + tot, 2 * tot, rounding_mode="floor")        # 0..255
    o = exact_matmul(p, v, max_w=256, max_x=128)                         # [B,H,T,dh]
    if m_o is not None:
        o = requant(o, torch.tensor(m_o, device=q.device), SHIFT, -INT8_MAX, INT8_MAX)
    return o


def attention(q, k, v, m_s: int, m_o: int | None, n_heads: int, n_kv: int, dh: int) -> torch.Tensor:
    b, t, _ = q.shape
    q = q.reshape(b, t, n_heads, dh).transpose(1, 2)
    k = k.reshape(b, t, n_kv, dh).transpose(1, 2)
    v = v.reshape(b, t, n_kv, dh).transpose(1, 2)
    if n_kv != n_heads:
        rep = n_heads // n_kv
        k = k.repeat_interleave(rep, 1)
        v = v.repeat_interleave(rep, 1)
    return attention_heads(q, k, v, m_s, m_o).transpose(1, 2).reshape(b, t, n_heads * dh)


def residual(a: torch.Tensor, b: torch.Tensor, m: torch.Tensor) -> torch.Tensor:
    return (a + ((b * m.to(b.device) + (1 << (SHIFT - 1))) >> SHIFT)).clamp(-RES_MAX, RES_MAX)

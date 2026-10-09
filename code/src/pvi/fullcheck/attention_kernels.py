"""A fused, exact integer causal attention on the GPU (Triton): the integers of
``transformer._attention_core`` without its ``T x T`` score tensors.

``_attention_core`` computes, for the queries of a group of heads, the scores ``s = q k^T`` (int8 inputs),
masks the keys after each query, takes each row's gaps ``d = max(s) - s`` to an exp table
(``e = LUT[min(d, len - 1)]``, 0 for a masked key), the row sums ``tot`` and the probabilities
``p = floor((tot + 510 e) / (2 tot))`` (0..255), and returns ``p v`` (int64).  This kernel computes the same
integers per block of query rows in three passes over the key blocks up to the block's last query (keys after
it are all masked and skipped):

1. the row maxima of the unmasked scores;
2. ``tot``, the row sums of ``e``;
3. ``p`` and ``p v``, accumulated over the key blocks.

**Exactness.** ``q``, ``k``, ``v`` are int8 and ``p <= 255``, all exact in fp16; ``tl.dot`` accumulates in
fp32, exact while every partial sum is an integer below ``2**24``.  A score is at most ``dh 127^2 <= 2**21``
(``dh <= 128``), and every partial sum of a row of ``p v`` is at most ``(sum p) 127 <= (255 + T/2) 127 <
2**24`` (``T < 2**15``; the same bound that lets ``_attention_core`` use one float32 GEMM).  The integer steps
(maximum, gaps, table, sums, floor division) are exact.  So the output is bit-identical (tested against
``_attention_core`` and ``reference.attention_heads``).
"""

from __future__ import annotations

import torch

from .codec_kernels import triton

if triton is not None:
    import triton.language as tl

__all__ = ["available", "attention_core"]


def available(device) -> bool:
    return triton is not None and torch.device(device).type == "cuda"


if triton is not None:
    @triton.jit
    def _scores(q, k_ptr, ks_t, ks_d, n0, T, DH, BN: tl.constexpr, BD: tl.constexpr):
        """The fp32 scores of the query rows ``q`` ([BM, BD] fp16) against keys ``[n0, n0 + BN)``."""
        offs_n = n0 + tl.arange(0, BN)
        offs_d = tl.arange(0, BD)
        kt = tl.load(k_ptr + offs_n[None, :] * ks_t + offs_d[:, None] * ks_d,
                     mask=(offs_n[None, :] < T) & (offs_d[:, None] < DH), other=0).to(tl.float16)
        return tl.dot(q, kt), offs_n

    @triton.jit
    def _attn_kernel(q_ptr, k_ptr, v_ptr, lut_ptr, out_ptr,
                     qs_b, qs_h, qs_r, qs_d, ks_b, ks_h, ks_t, ks_d, vs_b, vs_h, vs_t, vs_d,
                     os_b, os_h, os_r, os_d,
                     H, R, TQ, T, DH, L,
                     BM: tl.constexpr, BN: tl.constexpr, BD: tl.constexpr):
        pid_m = tl.program_id(0)
        bh = tl.program_id(1)
        b = bh // H
        h = bh % H
        offs_m = pid_m * BM + tl.arange(0, BM)
        offs_d = tl.arange(0, BD)
        rowok = offs_m < R
        qpos = T - TQ + offs_m % TQ                     # the stacked query heads' rows: position of each
        q = tl.load(q_ptr + b * qs_b + h * qs_h + offs_m[:, None] * qs_r + offs_d[None, :] * qs_d,
                    mask=rowok[:, None] & (offs_d[None, :] < DH), other=0).to(tl.float16)
        kp = k_ptr + b * ks_b + h * ks_h
        vp = v_ptr + b * vs_b + h * vs_h
        last = tl.max(tl.where(rowok, qpos, 0), axis=0)  # the block's last query: later keys are masked
        n_end = last + 1
        # 1. row maxima of the unmasked scores
        mx = tl.full([BM], -(1 << 30), tl.int32)
        for n0 in range(0, n_end, BN):
            s, offs_n = _scores(q, kp, ks_t, ks_d, n0, T, DH, BN, BD)
            keep = (offs_n[None, :] <= qpos[:, None]) & (offs_n[None, :] < T)
            mx = tl.maximum(mx, tl.max(tl.where(keep, s.to(tl.int32), -(1 << 30)), axis=1))
        # 2. row sums of the exp table at the gaps
        tot = tl.zeros([BM], tl.int32)
        for n0 in range(0, n_end, BN):
            s, offs_n = _scores(q, kp, ks_t, ks_d, n0, T, DH, BN, BD)
            keep = (offs_n[None, :] <= qpos[:, None]) & (offs_n[None, :] < T) & rowok[:, None]
            gap = tl.minimum(mx[:, None] - s.to(tl.int32), L - 1)
            e = tl.load(lut_ptr + tl.where(keep, gap, L - 1), mask=keep, other=0)
            tot += tl.sum(e, axis=1)
        tot = tl.where(rowok, tot, 1)
        # 3. probabilities and p v
        acc = tl.zeros([BM, BD], tl.float32)
        for n0 in range(0, n_end, BN):
            s, offs_n = _scores(q, kp, ks_t, ks_d, n0, T, DH, BN, BD)
            keep = (offs_n[None, :] <= qpos[:, None]) & (offs_n[None, :] < T) & rowok[:, None]
            gap = tl.minimum(mx[:, None] - s.to(tl.int32), L - 1)
            e = tl.load(lut_ptr + tl.where(keep, gap, L - 1), mask=keep, other=0)
            p = (tot[:, None] + 510 * e) // (2 * tot[:, None])
            p = tl.where(keep, p, 0).to(tl.float16)
            vt = tl.load(vp + offs_n[:, None] * vs_t + offs_d[None, :] * vs_d,
                         mask=(offs_n[:, None] < T) & (offs_d[None, :] < DH), other=0).to(tl.float16)
            acc += tl.dot(p, vt)
        tl.store(out_ptr + b * os_b + h * os_h + offs_m[:, None] * os_r + offs_d[None, :] * os_d,
                 acc.to(tl.int64), mask=rowok[:, None] & (offs_d[None, :] < DH))


def attention_core(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, lut: torch.Tensor, queries: int) -> torch.Tensor:
    """``_attention_core(q, k, v, lut, notmask)`` (raw ``p v``, int64 ``[B, H, R, dh]``) on the GPU for int8-valued
    ``q [B, H, R, dh]`` (``R = rep * queries`` stacked rows: row ``r`` is the query at position
    ``T - queries + r % queries``), ``k, v [B, H, T, dh]`` (any integer dtype, any strides) and the int32 table
    ``lut``; ``dh <= 128`` and ``T < 2**15``."""
    bsz, heads, rows, dh = q.shape
    t = k.shape[2]
    out = torch.empty(bsz, heads, rows, dh, dtype=torch.int64, device=q.device)
    bd = max(16, triton.next_power_of_2(dh))
    bm, bn = 64, 64
    grid = (triton.cdiv(rows, bm), bsz * heads)
    lut = lut.to(device=q.device, dtype=torch.int32).contiguous()
    _attn_kernel[grid](q, k, v, lut, out,
                       *q.stride(), *k.stride(), *v.stride(), *out.stride(),
                       heads, rows, queries, t, dh, lut.numel(),
                       BM=bm, BN=bn, BD=bd, num_warps=4)
    return out

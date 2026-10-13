"""Fused GPU kernels of the claim codec (Triton), for the device encoder of :mod:`claimcodec`.

The encoding's body is one slot stream per width ``B`` over the *global order* of the values (by width,
then segment order; a segment is an op's values, a centred op's residuals or its row means).  The device
encoder of :mod:`claimcodec` built each stream with about ten torch kernels per width plus index tables of
several bytes a value, in parts of ``_CHUNK`` values.  Here one kernel per width reads every value of a
block of the stream's columns straight from the claims, through a table of the stream's segments:

* the value at stream position ``v`` is found by a binary search of the segment starts, read from its op's
  claims (an int32 pointer per segment), minus its row's mean for a residual, minus the segment's base;
* its slot ``(x - lo) mod 2**B`` goes into lane ``v // G`` of column ``v % G`` (``G = ceil(n / 32)``): the
  ``B`` little-endian words of a column hold its 32 lanes' slots at bits ``[j B, j B + B)``, exactly as
  :func:`claimcodec.pack32`;
* a value outside ``[lo, lo + 2**B)`` is an exception: bit ``j`` of the column's flag word.

A second kernel recomputes the high parts ``(x - lo) >> B`` of the flagged positions.  The values, slots and
words are the integers of the torch encoder, so the bytes are the same (tested against it).

The decoder's kernel is the inverse: each program reads the words of a block of columns of a ``B``-bit stream
(``B <= 32``: also a Rice vector's low parts) and writes its 32 lanes' values in place, flagging an error if a
padding slot (a lane position at or past ``n``) is not zero.  Every offset and length is checked on the host
before the kernel reads anything (:func:`claimcodec.decode_device`).

Triton builds a small C launcher on first use; on hosts without the Python headers, ``PVI_PY_INCLUDE`` names
a directory holding a copy of them (``docs/improvements/I_server_setup.md``).
"""

from __future__ import annotations

import os
import sysconfig

import torch

_INCLUDE = os.environ.get("PVI_PY_INCLUDE")
if _INCLUDE:
    _paths = sysconfig.get_paths
    sysconfig.get_paths = lambda *a, **k: {**_paths(*a, **k), "include": _INCLUDE}

try:                                   # Triton ships with CUDA builds of torch on Linux
    import triton
    import triton.language as tl
except ImportError:                    # pragma: no cover - the CPU-only test environment
    triton = None

__all__ = ["available", "SegmentTable", "pack_stream", "high_parts", "unpack_stream"]

BLOCK = 64                             # columns per program (32 lanes each)


def available(device) -> bool:
    """Whether the fused kernels can run on ``device`` (a CUDA device and Triton)."""
    return triton is not None and torch.device(device).type == "cuda"


class SegmentTable:
    """The segments of the global order on the device: their starts (int64, with the end), their source
    addresses (an int32 pointer to the segment's first value: its op's claims, or the row means for a
    centred op's means), the index of their first row's mean for a residual (else -1), their row length
    (residuals; else 1) and their bases.  ``keep`` holds the tensors the addresses point into."""

    def __init__(self, starts, addresses, mean_rows, cols, bases, means: torch.Tensor, keep: list, device) -> None:
        def up(values, dtype):
            return torch.tensor(values, dtype=dtype).to(device)

        self.start = up(starts, torch.int64)
        self.src = up(addresses, torch.int64)
        self.mean = up(mean_rows, torch.int64)
        self.cols = up(cols, torch.int64)
        self.lo = up(bases, torch.int32)
        self.means = means
        self.keep = keep
        self.steps = max(1, (len(addresses) - 1).bit_length()) + 1


if triton is not None:
    @triton.jit
    def _values(e, ok, s0, s1, start_ptr, src_ptr, mean_ptr, cols_ptr, lo_ptr, mr_ptr, STEPS: tl.constexpr):
        """The value at global position ``e`` minus its segment's base (and a residual's row mean), int32: the
        segment by a binary search of the starts in ``[s0, s1)`` (the last one starting at or before ``e``)."""
        lo_s = tl.zeros(e.shape, dtype=tl.int64) + s0
        hi_s = tl.zeros(e.shape, dtype=tl.int64) + s1
        for _ in tl.static_range(STEPS):
            mid = (lo_s + hi_s) // 2
            active = (hi_s - lo_s) > 1
            st = tl.load(start_ptr + mid, mask=ok & active, other=0)
            right = st <= e
            lo_s = tl.where(active & right, mid, lo_s)
            hi_s = tl.where(active & (~right), mid, hi_s)
        seg = lo_s
        k = e - tl.load(start_ptr + seg, mask=ok, other=0)
        base = tl.load(src_ptr + seg, mask=ok, other=0).to(tl.pointer_type(tl.int32))
        x = tl.load(base + k, mask=ok, other=0)
        mb = tl.load(mean_ptr + seg, mask=ok, other=-1)
        cols = tl.load(cols_ptr + seg, mask=ok, other=1)
        has = ok & (mb >= 0)
        mean = tl.load(mr_ptr + tl.where(has, mb + k // cols, 0), mask=has, other=0)
        return x - mean - tl.load(lo_ptr + seg, mask=ok, other=0)

    @triton.jit
    def _stream_kernel(start_ptr, src_ptr, mean_ptr, cols_ptr, lo_ptr, mr_ptr, words_ptr, flags_ptr,
                       s0, s1, a, n, G, g0, m,
                       B: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
        """Columns ``[g0, g0 + m)`` of the stream of width ``B`` that starts at global position ``a`` and holds
        ``n`` values in ``G`` columns: its words ``[B, m]`` (row stride ``m``) and its flag words."""
        gl = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        colok = gl < m
        g = (g0 + gl).to(tl.int64)
        lane = tl.arange(0, 32).to(tl.int64)
        v = lane[:, None] * G + g[None, :]
        ok = colok[None, :] & (v < n)
        val = _values(a + v, ok, s0, s1, start_ptr, src_ptr, mean_ptr, cols_ptr, lo_ptr, mr_ptr, STEPS)
        flag = ok & ((val >> B) != 0)
        one = tl.full([32, 1], 1, tl.int64)
        bits = tl.sum(tl.where(flag, one << lane[:, None], 0), axis=0)
        tl.store(flags_ptr + g, bits.to(tl.int32), mask=colok)
        if B > 0:
            slot = tl.where(ok, val & ((1 << B) - 1), 0).to(tl.int64)
            bitpos = lane[:, None] * B
            wi = bitpos >> 5
            sh = bitpos & 31
            for i in tl.static_range(B):
                main = tl.where(wi == i, (slot << sh) & 0xFFFFFFFF, 0)
                spill = tl.where((wi + 1 == i) & (sh + B > 32), slot >> (32 - sh), 0)
                w = tl.sum(main + spill, axis=0)
                tl.store(words_ptr + i * m + gl, w.to(tl.int32), mask=colok)

    @triton.jit
    def _high_kernel(start_ptr, src_ptr, mean_ptr, cols_ptr, lo_ptr, mr_ptr, pos_ptr, out_ptr, n_pos, s0, s1,
                     B: tl.constexpr, STEPS: tl.constexpr, BLOCK: tl.constexpr):
        """The high parts ``(x - lo) >> B`` (int32) of the values at the global positions ``pos``."""
        i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
        ok = i < n_pos
        e = tl.load(pos_ptr + i, mask=ok, other=0)
        val = _values(e, ok, s0, s1, start_ptr, src_ptr, mean_ptr, cols_ptr, lo_ptr, mr_ptr, STEPS)
        tl.store(out_ptr + i, val >> B, mask=ok)


if triton is not None:
    @triton.jit
    def _unpack_kernel(words_ptr, out_ptr, err_ptr, n, G, B: tl.constexpr, BLOCK: tl.constexpr):
        """The values of a ``B``-bit stream of ``n`` values in ``G`` columns (words ``[B, G]``, int32) into
        ``out[:n]``; ``err`` is set to 1 if a padding slot holds a non-zero value."""
        g = (tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)).to(tl.int64)
        colok = g < G
        lane = tl.arange(0, 32).to(tl.int64)
        bitpos = lane[:, None] * B
        wi = bitpos >> 5
        sh = bitpos & 31
        spill = (sh + B) > 32
        low = tl.load(words_ptr + wi * G + g[None, :], mask=colok[None, :], other=0).to(tl.int64) & 0xFFFFFFFF
        high = tl.load(words_ptr + (wi + 1) * G + g[None, :], mask=colok[None, :] & spill, other=0)
        high = high.to(tl.int64) & 0xFFFFFFFF
        val = ((low >> sh) | (high << (32 - sh))) & ((1 << B) - 1)
        v = lane[:, None] * G + g[None, :]
        inside = colok[None, :] & (v < n)
        tl.store(out_ptr + v, val, mask=inside)
        bad = tl.where(colok[None, :] & (v >= n) & (val != 0), 1, 0)
        tl.atomic_max(err_ptr, tl.max(tl.max(bad, axis=1), axis=0))


def unpack_stream(words: torch.Tensor, out: torch.Tensor, err: torch.Tensor, n: int, b: int) -> None:
    """The ``n`` values of the ``b``-bit stream ``words`` (int32 ``[b, ceil(n / 32)]``, contiguous; ``1 <= b <= 32``)
    into ``out[:n]`` (int32 for slots, int64 for a Rice vector's low parts); ``err`` (int32 ``[1]``) becomes 1 on a
    non-zero padding slot."""
    g = -(-n // 32)
    grid = (triton.cdiv(g, BLOCK),)
    _unpack_kernel[grid](words, out, err, n, g, B=b, BLOCK=BLOCK)


def pack_stream(t: SegmentTable, s0: int, s1: int, a: int, n: int, b: int, g0: int, m: int,
                words: torch.Tensor, flags: torch.Tensor) -> None:
    """Columns ``[g0, g0 + m)`` of the stream of width ``b`` (segments ``[s0, s1)``, values ``[a, a + n)`` of
    the global order): ``words`` (int32 ``[b, m]``, contiguous; unused when ``b == 0``) and the stream's flag
    words ``flags`` (int32 ``[G]``)."""
    g = -(-n // 32)
    grid = (triton.cdiv(m, BLOCK),)
    _stream_kernel[grid](t.start, t.src, t.mean, t.cols, t.lo, t.means, words, flags,
                         s0, s1, a, n, g, g0, m, B=b, STEPS=t.steps, BLOCK=BLOCK)


def high_parts(t: SegmentTable, s0: int, s1: int, b: int, pos: torch.Tensor) -> torch.Tensor:
    """The high parts (int32) of the values at the global positions ``pos`` (int64), all in segments
    ``[s0, s1)`` of width ``b``."""
    out = torch.empty(pos.numel(), dtype=torch.int32, device=pos.device)
    if pos.numel():
        grid = (triton.cdiv(pos.numel(), 1024),)
        _high_kernel[grid](t.start, t.src, t.mean, t.cols, t.lo, t.means, pos, out, pos.numel(), s0, s1,
                           B=b, STEPS=t.steps, BLOCK=1024)
    return out

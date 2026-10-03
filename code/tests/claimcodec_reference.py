"""The ``PVC3`` encoder as commit 6c2059a wrote it, verbatim: the specification the fast encoders of
``pvi.fullcheck.claimcodec`` are tested against (same bytes for every input).  Not part of the package."""

from __future__ import annotations

import functools
import struct

import numpy as np
import torch

MAGIC = b"PVC3"
BMAX = 30
_X_LIMIT = 1 << 30
_Z_LIMIT = 1 << 29
_F_CENTRED = 1
_KMAX = 32
_QCAP = 15
_SMALL = 1 << 15
_PLAN_ROWS = 256
_PLAN_VALUES = 1 << 16


class Unencodable(ValueError):
    """As ``claimcodec.Unencodable``."""


@functools.lru_cache(maxsize=None)
def _lane_maps(k: int):
    """Per lane ``j``: first word, its shift and ``32 -`` it (as ``[32, 1]`` columns), the word
    receiving the spill (``k`` = a zero row), and whether the lane spills.  Read only."""
    bit = np.arange(32, dtype=np.int64) * k
    wi, sh = bit >> 5, bit & 31
    spill = (sh > 0) & (sh + k > 32)
    return wi, sh.astype(np.uint64)[:, None], (32 - sh).astype(np.uint64)[:, None], np.where(spill, wi + 1, k), spill


def _pack32_small(vv: np.ndarray, k: int, g: int) -> bytes:
    wi, sh, rsh, w2, spill = _lane_maps(k)
    v64 = vv.astype(np.uint64)                                           # [32, g]
    lo = (v64 << sh) & np.uint64(0xFFFFFFFF)
    hi = np.where(spill[:, None], v64 >> rsh, np.uint64(0))
    target = np.concatenate([wi, w2])                                    # the word of each part
    contrib = np.concatenate([lo, hi])
    order = np.argsort(target, kind="stable")
    target, contrib = target[order], contrib[order]
    starts = np.flatnonzero(np.r_[True, target[1:] != target[:-1]])
    words = np.zeros((k + 1, g), dtype=np.uint64)
    words[target[starts]] = np.add.reduceat(contrib, starts, axis=0)     # disjoint bits: sum == or
    return words[:k].astype("<u4").tobytes()


def pack32(v: np.ndarray, k: int) -> bytes:
    """``v`` (``uint32``, all ``< 2**k``, ``k <= 32``) at ``k`` bits each: ``4 k ceil(n / 32)`` bytes."""
    n = v.size
    if k == 0 or n == 0:
        return b""
    g = (n + 31) // 32
    vv = np.zeros(32 * g, dtype=np.uint32)
    vv[:n] = v
    vv = vv.reshape(32, g)
    if n <= _SMALL:
        return _pack32_small(vv, k, g)
    w = np.zeros((k + 1, g), dtype="<u4")
    tmp = np.empty(g, dtype=np.uint32)
    for j in range(32):
        wi, sh = (j * k) >> 5, (j * k) & 31
        np.left_shift(vv[j], np.uint32(sh), out=tmp)
        np.bitwise_or(w[wi], tmp, out=w[wi])
        if sh and sh + k > 32:
            np.right_shift(vv[j], np.uint32(32 - sh), out=tmp)
            np.bitwise_or(w[wi + 1], tmp, out=w[wi + 1])
    return w[:k].tobytes()


def _pack32_torch(v: torch.Tensor, k: int) -> bytes:
    """:func:`pack32` with torch ops on ``v``'s device, all 32 lanes at once (about ten kernels, so
    a GPU prover packs before the copy to the host)."""
    n = v.numel()
    if k == 0 or n == 0:
        return b""
    g = (n + 31) // 32
    dev = v.device
    vv = torch.zeros(32 * g, dtype=torch.int64, device=dev)
    vv[:n] = v.reshape(-1)
    vv = vv.view(32, g)
    wi, sh, rsh, w2, spill = (torch.from_numpy(np.asarray(t, dtype=np.int64)).to(dev) for t in _lane_maps(k))
    lo = (vv << sh) & 0xFFFFFFFF
    hi = torch.where(spill[:, None].bool(), vv >> rsh, torch.zeros((), dtype=torch.int64, device=dev))
    words = torch.zeros(k + 1, g, dtype=torch.int64, device=dev)
    words.index_add_(0, wi, lo)          # the bits of different lanes are disjoint: sum == or
    words.index_add_(0, w2, hi)
    # int64 -> int32 keeps the low 32 bits: the same little-endian bytes as the uint32 words
    return words[:k].to(torch.int32).cpu().numpy().astype("<i4", copy=False).tobytes()


def _pack_levels(q: np.ndarray) -> bytes:
    parts, active = [], q
    for level in range(1, _QCAP + 1):
        if active.size == 0:
            break
        bits = active >= level
        parts.append(np.packbits(bits, bitorder="little").tobytes())
        active = active[bits]
    return b"".join(parts)


def _rice_k(v: np.ndarray) -> int:
    """The ``k`` with the fewest bits, among those that keep every quotient ``<= _QCAP``."""
    kmin = max(0, int(v.max()).bit_length() - 4)
    best = None
    for k in range(kmin, _KMAX + 1):
        q = v >> k
        bits = v.size * k + int(q.sum()) + int((q < _QCAP).sum())
        if best is None or bits < best[0]:
            best = (bits, k)
        elif k > best[1] + 1:
            break
    return best[1]


def _rice_encode(v: np.ndarray) -> bytes:
    """Non-negative integers ``< 2**36``; an empty vector is empty (the decoder knows the length)."""
    v = np.asarray(v, dtype=np.int64)
    if v.size == 0:
        return b""
    if int(v.min()) < 0 or int(v.max()) >= 1 << (_KMAX + 4):
        raise ValueError("Rice values must be in [0, 2**36)")
    k = _rice_k(v)
    low = pack32((v & ((1 << k) - 1)).astype(np.uint32), k)
    return bytes([k]) + low + _pack_levels(v >> k)


def _width_costs(x: np.ndarray, c: int, scale: float) -> np.ndarray:
    """Estimated bits of the values ``x`` about ``c`` at every width ``B = 0..BMAX`` (``x`` is a
    sample of the segment, ``scale`` = segment size / sample size)."""
    n = x.size
    d = x.reshape(-1) - c
    zz = (d << 1) ^ (d >> 63)
    hist = np.bincount(np.frexp(zz.astype(np.float64))[1], minlength=64).astype(np.float64)
    ls = np.arange(64, dtype=np.float64)
    bs = np.arange(BMAX + 1, dtype=np.float64)
    ge = np.cumsum(hist[::-1])[::-1]                   # values whose zigzag has >= j bits
    gel = np.cumsum((hist * ls)[::-1])[::-1]
    e = ge[1:BMAX + 2]                                 # exceptions at width B: more than B bits
    payload = gel[1:BMAX + 2] - bs * e                 # about their high parts' bits
    exc = e * (np.log2(max(n, 1) / np.maximum(e, 1.0)) + 3.5) + payload
    return scale * (n * bs + exc)


def _choose(x: np.ndarray, c: int, scale: float = 1.0) -> tuple[int, float]:
    costs = _width_costs(x, c, scale)
    b = int(np.argmin(costs))
    return b, float(costs[b])


def _sample_rows(n_rows: int, m: int) -> np.ndarray:
    """Rows sampled to plan an ``[N, M]`` matrix: at most ``_PLAN_ROWS`` or ``~_PLAN_VALUES`` values."""
    step = max(1, n_rows // _PLAN_ROWS, (n_rows * m) // _PLAN_VALUES)
    return np.arange(0, n_rows, step)


def _median(x: np.ndarray) -> int:
    return int(np.round(np.median(x))) if x.size else 0


def _plan(zs: np.ndarray, scale: float, centre: bool) -> dict:
    """The cheapest coding of one ``[N, M]`` op from its sampled rows ``zs`` (``scale`` = N / rows
    sampled): the width and centre of its values, or (centred) of its residuals."""
    c = _median(zs)
    b, cost = _choose(zs, c, scale)
    best = {"centred": False, "B": b, "c": c}
    m = zs.shape[1]
    if centre and m >= 2 and zs.shape[0]:
        off = np.floor_divide(zs.sum(1) + m // 2, m)              # the sampled rows' integer means
        _, ocost = _choose(off, _median(off), scale)
        res = zs - off[:, None]
        cr = _median(res)
        br, rcost = _choose(res, cr, scale)
        if ocost + 40 + rcost < cost:                             # 40: the offsets' B and lo
            best = {"centred": True, "B": br, "c": cr}
    return best


def _base(c: int, b: int) -> int:
    """The base centring the ``2**b`` slots on ``c``, within the range the decoder accepts."""
    return min(max(c - (1 << b >> 1), 1 - _X_LIMIT), _X_LIMIT - (1 << b))


def _nonzero(mask: torch.Tensor) -> np.ndarray:
    if mask.device.type == "cpu":
        return np.flatnonzero(mask.numpy())          # numpy's is several times faster than torch's on a CPU
    return torch.nonzero(mask).reshape(-1).cpu().numpy()


def _host(tensors: list[torch.Tensor]) -> list[np.ndarray]:
    """The tensors on the host, with one copy per device (flattened, then split)."""
    out: list = [None] * len(tensors)
    by_dev: dict = {}
    for i, t in enumerate(tensors):
        by_dev.setdefault(t.device, []).append(i)
    for idx in by_dev.values():
        flat = torch.cat([tensors[i].reshape(-1) for i in idx]).cpu().numpy()
        o = 0
        for i in idx:
            c = tensors[i].numel()
            out[i] = flat[o:o + c].reshape(tuple(tensors[i].shape))
            o += c
    return out


def encode(claims: list, *, centre: bool = True) -> bytes:
    """``PVC3`` bytes of the claim matrices (``[N, M]`` integer tensors or arrays, in the graph's
    weight-op order, every ``|z| < 2**29``).  ``centre=False`` never centres (for measurements).

    Tensors are encoded on their own device: only the range extremes, a sample of rows, the row
    means of centred ops (each in one copy), the exceptions and the packed words come to the host."""
    zts = [z.detach() if torch.is_tensor(z) else torch.from_numpy(np.asarray(z)) for z in claims]
    if any(z.dim() != 2 for z in zts):
        raise ValueError("claims must be 2-D")
    ext = _host([torch.stack(torch.aminmax(z)).to(torch.int64) for z in zts if z.numel()])
    if ext and (max(int(e[1]) for e in ext) >= _Z_LIMIT or min(int(e[0]) for e in ext) <= -_Z_LIMIT):
        raise Unencodable("a claim outside the range check cannot be encoded")
    z32s = [z.to(torch.int32) for z in zts]            # every claim is < 2**29: the rest runs in int32
    rows = [_sample_rows(*z.shape) for z in zts]
    samples = _host([z[torch.from_numpy(r).to(z.device)] for z, r in zip(z32s, rows)])
    plans = [_plan(s.astype(np.int64), z.shape[0] / max(r.size, 1), centre)
             for z, s, r in zip(zts, samples, rows)]
    centred = [i for i, p in enumerate(plans) if p["centred"]]
    means = {i: torch.div(zts[i].to(torch.int64).sum(1) + zts[i].shape[1] // 2, zts[i].shape[1],
                          rounding_mode="floor").to(torch.int32) for i in centred}     # the integer row means
    means_h = dict(zip(centred, _host([means[i] for i in centred])))
    ops = []
    for i, (z32, p) in enumerate(zip(z32s, plans)):
        segs = []
        x = z32.reshape(-1)
        if p["centred"]:
            co = _median(means_h[i])
            bo, _ = _choose(means_h[i].astype(np.int64), co)
            segs.append((bo, _base(co, bo), means[i]))
            x = (z32 - means[i][:, None]).reshape(-1)                # |z - o| < 2**30
        segs.append((p["B"], _base(p["c"], p["B"]), x))
        ops.append((z32.shape[1], segs))
    return _assemble(ops)


def _assemble(ops: list) -> bytes:
    """The ``PVC3`` bytes of ``ops``: per op ``(M, segments)``, with one segment ``(B, lo, x)`` or two
    (the row means', then the residuals'), ``x`` a flat int32 tensor with ``|x| < 2**30``."""
    head = [MAGIC, struct.pack("<I", len(ops))]
    segs = []
    for m, op_segs in ops:
        head.append(struct.pack("<IB", m, _F_CENTRED if len(op_segs) == 2 else 0))
        for b, lo, x in op_segs:
            head.append(struct.pack("<Bi", b, lo))
            segs.append((b, lo, x))
    head = b"".join(head)
    parts = [head + bytes(-len(head) % 4)]
    exc_pos, exc_h, base = [], [], 0
    for b in sorted({s[0] for s in segs}):
        group = [(lo, x) for bb, lo, x in segs if bb == b]          # segment order within a width
        cnt = sum(x.numel() for _, x in group)
        d = torch.empty(cnt, dtype=torch.int32, device=group[0][1].device)
        o = 0
        for lo, x in group:                                          # |x - lo| < 2**31: no overflow
            torch.sub(x, lo, out=d[o:o + x.numel()])
            o += x.numel()
        e = _nonzero((d < 0) | (d >= (1 << b)))
        if e.size:
            exc_pos.append(e + base)
            exc_h.append((d[torch.from_numpy(e).to(d.device)] >> b).cpu().numpy().astype(np.int64))
        d &= (1 << b) - 1                                            # the slots, in place
        parts.append(pack32(d.numpy().view(np.uint32), b) if d.device.type == "cpu" else _pack32_torch(d, b))
        base += cnt
    pos = np.concatenate(exc_pos) if exc_pos else np.zeros(0, dtype=np.int64)
    h = np.concatenate(exc_h) if exc_h else np.zeros(0, dtype=np.int64)
    parts.append(struct.pack("<I", pos.size) + _rice_encode(np.diff(pos, prepend=-1) - 1)
                 + _rice_encode(((h << 1) ^ (h >> 63)) - 1))
    return b"".join(parts)

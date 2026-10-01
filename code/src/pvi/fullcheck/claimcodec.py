"""A compact, lossless wire encoding of the proof (``run_query(wire=True)``).

The protocol counts every claim (a pre-activation ``z``) and every field element as 4 bytes.
The claims of an int8 model are far below the range check ``|z| < 2**29``: the benchmark's
random-weight LLM claims need 18-21 signed bits and carry about 17 bits of entropy, embedding
look-ups 8.  In mode Kpre the proof is only the claims, so their encoding is the proof size.
Field elements (``u``, the opened columns) are below ``p < 2**31`` and travel at 31 bits
(:func:`pack_field`).  The claims of a commitment plan's lookup tables are rows of int8 weights
and travel as their bytes (:func:`pack_rows`), not in ``PVC3``.

Claims: format ``PVC3``
-----------------------
A frame of reference whose exceptions carry only their high part (as "NewPFD"); the prototype
``PVC2`` of branch research-claims without its raw (int32) ops, which are never smaller than
``B = 30``, and with median centres (smaller on the CNNs, the same on the decoders).  The verifier
knows every weight op's shape ``[N, M]``: ``N`` from the public graph, ``M`` from the query's shape
(``Verifier.claim_columns``).  The header repeats ``M``, and the decoder rejects any other before
it allocates anything, so a malformed input costs it no more memory than the honest claims.

Each op is one *segment*, or two when it is *centred*: a run of integers ``x`` with a width
``B <= 30`` and a base ``lo``.  Every value is stored as its ``B``-bit slot ``(x - lo) mod 2**B``;
a value outside ``[lo, lo + 2**B)`` is also an *exception*, whose high part
``h = (x - lo) >> B`` (non-zero, almost always +-1) goes to one list for the whole proof.  A
centred op sends its ``N`` integer row means ``o_i = floor((sum_j z_ij + M // 2) / M)`` as a
segment of its own, then the residuals ``z_ij - o_i`` (worth it when rows have different means:
ReLU or GELU inputs, correlated tokens).  The encoder picks centring and ``B`` per op by an
estimate of the bits (``B`` per value, about ``log2(n / e) + 3.5`` per exception plus its high
part) on a sample of rows.

Layout (little-endian), nothing before or after::

    b"PVC3"  u32 n_ops
    per op:  u32 M  u8 flags (bit 0: centred; the others zero)
             [centred: u8 B  i32 lo  of the row means]  u8 B  i32 lo
    zero bytes up to a multiple of 4
    one slot stream per width, in increasing B (widths with no value have none)
    u32 e  Rice(gaps)  Rice(payloads)        -- the e exceptions

The global value order is "by width, then segment order" (a centred op's row means before its
residuals, ops in graph order), so each stream and each segment is one contiguous block of it.
Exception ``i`` sits at global position ``pos_i``; its gap is ``pos_i - pos_{i-1} - 1``
(``pos_{-1} = -1``) and its payload ``zigzag(h) - 1``.

Slot streams (:func:`pack32`): ``n`` values of ``k`` bits in 32 lanes of ``G = ceil(n / 32)``
values; value ``j G + g`` occupies bits ``[j k, j k + k)`` of the ``k`` little-endian 32-bit words
``g, G + g, 2 G + g, ...``, padding values are zero.  Unpacking is 32 contiguous shift/or/and
passes (one vectorised step below ``_SMALL`` values).  With ``workers`` threads the decoder splits
the lanes of large streams, the exceptions and the ops (numpy releases the GIL in each pass), and
reads the exception list while the streams are unpacked: the offsets of both follow from the
header.

Rice vectors (:func:`_rice_encode`): a byte ``k <= 32``; the ``k``-bit low parts as a slot
stream; the quotients ``q = v >> k <= 15`` in bit-plane unary: level ``l = 1..15`` holds one bit
``[q >= l]`` for each value still active at that level (``bitorder="little"``, zero-padded to a
byte).  They hold values below ``2**36``.

Decoding (:func:`decode`) rejects with :class:`ClaimCodecError`, never with another exception,
any input that is not exactly such an encoding: a bad magic or op count, an ``M`` other than
the expected one, other flag bits, ``B > 30``, a base outside ``(-2**30, 2**30 - 2**B]``,
truncated or trailing bytes, non-zero padding, ``k > 32``, exception positions out of order or
range, and exceptions reaching ``|x| >= 2**30``.  So every segment value is ``|x| < 2**30`` and
every claim fits in int32.  The range check stays the verifier's (``|z| < 2**29`` in
``Verifier.derive``): the codec only guarantees well-formed matrices of the expected shapes.
The work and the memory are linear in the input and in the claims' size ``sum(N M)``.

Soundness.  Decoding is a deterministic function from bytes to claim matrices (or a rejection),
applied to what the prover sent before any challenge; with Fiat--Shamir the transcript absorbs
these bytes, which fixes the claims.  So Freivalds' and the column check's bounds are those of
the int32 claims.  The encoding is not canonical (other widths give other bytes for the same
claims): re-encoding is one more way to grind the Fiat--Shamir challenges, which the
``grinding_bits`` margin already covers.
"""

from __future__ import annotations

import functools
import itertools
import struct

import numpy as np
import torch

from .commitment import map_threaded

__all__ = ["ClaimCodecError", "Unencodable", "MAGIC", "BMAX", "encode", "decode", "decode_torch", "FIELD_BITS",
           "field_size", "pack_field", "unpack_field", "pack_rows", "unpack_rows"]

MAGIC = b"PVC3"
BMAX = 30                   # slot widths 0..30
_X_LIMIT = 1 << 30          # every segment value |x| < 2**30
_Z_LIMIT = 1 << 29          # the range check (protocol.Z_BOUND): the encoder refuses larger claims
_F_CENTRED = 1
_KMAX = 32                  # Rice low parts
_QCAP = 15                  # Rice quotients, so Rice values < 2**36
_SMALL = 1 << 15            # streams up to this many values: all 32 lanes in one vectorised step
_THREADED = 1 << 18         # streams of at least this many values: lanes split over the workers
_PLAN_ROWS = 256            # rows sampled to plan an op
_PLAN_VALUES = 1 << 16


class ClaimCodecError(ValueError):
    """Malformed encoded proof: the verifier rejects."""


class Unencodable(ValueError):
    """Values the encoding cannot carry -- a claim outside the range check, a looked-up row outside
    int8, a field element of 32 bits: no honest message holds them, and whatever a prover with such
    values sends instead, the verifier rejects or decodes into other values."""


# ---------------------------------------------------------------------------- slot streams
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


def _unpack32_small(w: np.ndarray, k: int, g: int) -> np.ndarray:
    wi, sh, rsh, w2, _ = _lane_maps(k)
    w64 = np.zeros((k + 1, g), dtype=np.uint64)
    w64[:k] = w
    v = (w64[wi] >> sh) | (w64[w2] << rsh)
    return (v & np.uint64((1 << k) - 1)).astype(np.uint32)             # [32, g]


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


def _unpack_small(w: np.ndarray, k: int, n: int, out: np.ndarray) -> None:
    v = _unpack32_small(w, k, w.shape[1]).reshape(-1)
    if v[n:].any():
        raise ClaimCodecError("non-zero padding in a slot stream")
    out[:n] = v[:n]


def _unpack_lanes(w: np.ndarray, k: int, n: int, out: np.ndarray, js) -> None:
    """Lanes ``js`` of the stream ``w`` of ``n`` values into ``out[:n]`` (lane ``j`` holds values
    ``[j G, (j + 1) G)``; the lane holding padding goes through a buffer, whose padding is checked)."""
    g = w.shape[1]
    tmp = np.empty(g, dtype=np.uint32)
    mask = np.uint32((1 << k) - 1)
    for j in js:
        wi, sh = (j * k) >> 5, (j * k) & 31
        a = j * g
        o = out[a:a + g] if a + g <= n else np.empty(g, dtype=np.uint32)
        np.right_shift(w[wi], np.uint32(sh), out=o)
        if sh and sh + k > 32:
            np.left_shift(w[wi + 1], np.uint32(32 - sh), out=tmp)
            np.bitwise_or(o, tmp, out=o)
        if sh + k != 32:                    # a value ending at a word boundary needs no mask
            np.bitwise_and(o, mask, out=o)
        if a + g > n:
            m = max(n - a, 0)
            if o[m:].any():
                raise ClaimCodecError("non-zero padding in a slot stream")
            out[a:a + m] = o[:m]


def _stream_jobs(buf, offset: int, n: int, k: int, out: np.ndarray, workers: int) -> tuple[list, int]:
    """Jobs that unpack the stream of ``n`` ``k``-bit values at ``buf[offset:]`` into exactly
    ``out[:n]`` (all 32 lanes in one vectorised step up to ``_SMALL`` values; from ``_THREADED``
    values, lanes split in ``workers`` jobs), and the offset after the stream."""
    g = (n + 31) // 32
    nbytes = 4 * k * g
    if offset + nbytes > len(buf):
        raise ClaimCodecError("truncated slot stream")
    if n == 0 or k == 0:
        out[:n] = 0
        return [], offset
    w = np.frombuffer(buf, dtype="<u4", count=k * g, offset=offset).reshape(k, g)
    if n <= _SMALL:
        return [functools.partial(_unpack_small, w, k, n, out)], offset + nbytes
    parts = min(workers, 32) if n >= _THREADED else 1
    return [functools.partial(_unpack_lanes, w, k, n, out, range(i, 32, parts)) for i in range(parts)], offset + nbytes


def _run(jobs: list, sizes: list[int], workers: int) -> None:
    """Every job, those of ``sizes >= LEAF_THREAD_BYTES`` on ``workers`` threads (numpy releases the
    GIL in each pass), the others meanwhile on this one."""
    map_threaded(lambda job: job(), jobs, sizes, workers)


def unpack32(buf, offset: int, n: int, k: int, out: np.ndarray | None = None,
             workers: int = 1) -> tuple[np.ndarray, int]:
    """Inverse of :func:`pack32` at ``buf[offset:]``: ``(values, offset after them)``, written into
    ``out[:n]`` (``uint32``) with ``workers`` threads."""
    out = np.empty(n, dtype=np.uint32) if out is None else out
    jobs, offset = _stream_jobs(buf, offset, n, k, out, workers)
    _run(jobs, [4 * n // len(jobs) if len(jobs) > 1 else 0] * len(jobs), workers)
    return out[:n], offset


# ---------------------------------------------------------------------------- Rice vectors
def _pack_levels(q: np.ndarray) -> bytes:
    parts, active = [], q
    for level in range(1, _QCAP + 1):
        if active.size == 0:
            break
        bits = active >= level
        parts.append(np.packbits(bits, bitorder="little").tobytes())
        active = active[bits]
    return b"".join(parts)


def _unpack_levels(buf, offset: int, n: int) -> tuple[np.ndarray, int]:
    q = np.zeros(n, dtype=np.int64)
    idx = None
    count = n
    for level in range(1, _QCAP + 1):
        if count == 0:
            break
        nbytes = (count + 7) // 8
        if offset + nbytes > len(buf):
            raise ClaimCodecError("truncated unary levels")
        raw = np.frombuffer(buf, dtype=np.uint8, count=nbytes, offset=offset)
        if count % 8 and raw[-1] >> (count % 8):
            raise ClaimCodecError("non-zero padding in unary levels")
        bits = np.unpackbits(raw, count=count, bitorder="little").view(bool)
        offset += nbytes
        idx = np.flatnonzero(bits) if idx is None else idx[bits]
        q[idx] = level
        count = idx.size
    return q, offset


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


def _rice_decode(buf, offset: int, n: int) -> tuple[np.ndarray, int]:
    if n == 0:
        return np.zeros(0, dtype=np.int64), offset
    if offset >= len(buf):
        raise ClaimCodecError("truncated Rice vector")
    k = buf[offset]
    if k > _KMAX:
        raise ClaimCodecError("Rice parameter out of range")
    low, offset = unpack32(buf, offset + 1, n, k)
    v, offset = _unpack_levels(buf, offset, n)
    if k:
        v <<= k
        v |= low
    return v, offset


# ---------------------------------------------------------------------------- the encoder
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


# ---------------------------------------------------------------------------- the decoder
def _need(buf, off: int, n: int) -> int:
    if off + n > len(buf):
        raise ClaimCodecError("truncated header")
    return off + n


def _patch(x: np.ndarray, gaps: np.ndarray, hz: np.ndarray, segs: list[tuple[int, int, int]], total: int,
           workers: int) -> None:
    """Adds the exceptions to the slots ``x``: value ``slot + lo + h 2**B`` stored minus its base
    ``lo``, which is added with the others.  ``segs`` are ``(start, B, lo)`` in the global order.
    Works in place on ``gaps`` and ``hz`` (few passes and no large temporaries: the exceptions are
    4-15% of the claims); from ``_THREADED`` exceptions in ``workers`` slices."""
    pos = gaps
    pos += 1
    np.cumsum(pos, out=pos)
    pos -= 1
    # each step is in [1, 2**36]: a sum that wrapped past 2**63 left a negative position
    if int(pos.min()) < 0 or int(pos[-1]) >= total:
        raise ClaimCodecError("exception position out of range")
    if int(hz.max()) >= 1 << 32:                 # then |h| <= 2**31 and h 2**B fits in int64
        raise ClaimCodecError("exception out of range")
    # positions are sorted and each segment is a block of the global order
    cuts = np.searchsorted(pos, [s for s, _, _ in segs] + [total])
    n = pos.size
    parts = workers if n >= _THREADED else 1
    ends = [n * i // parts for i in range(parts + 1)]
    _run([functools.partial(_patch_slice, x, pos, hz, segs, cuts, a, e) for a, e in zip(ends, ends[1:])],
         [8 * n // parts if parts > 1 else 0] * parts, workers)


def _patch_slice(x, pos, hz, segs, cuts, a: int, e: int) -> None:
    """:func:`_patch` for the exceptions ``[a, e)``."""
    h = hz[a:e]                                  # h = unzigzag(hz + 1)
    h += 1
    sign = h & 1
    h >>= 1
    np.negative(sign, out=sign)
    h ^= sign
    i = 0
    for b, run in itertools.groupby(segs, key=lambda s: s[1]):       # a block of the global order per width
        j = i + len(list(run))
        if b and max(cuts[i], a) < min(cuts[j], e):
            h[max(cuts[i], a) - a:min(cuts[j], e) - a] <<= b
        i = j
    p = pos[a:e]
    np.add(h, x[p], out=h)                       # slot + h 2**B
    first, last = np.clip(cuts[:-1], a, e), np.clip(cuts[1:], a, e)
    used = np.flatnonzero(first < last)          # the segments with exceptions here
    lo = np.array([segs[i][2] for i in used], dtype=np.int64)
    if ((np.minimum.reduceat(h, first[used] - a) + lo <= -_X_LIMIT).any()
            or (np.maximum.reduceat(h, first[used] - a) + lo >= _X_LIMIT).any()):
        raise ClaimCodecError("exception out of range")
    x[p] = h


def _add_bases(x: np.ndarray, ops: list) -> None:
    """Per op ``(start, N, M, lo, row means' start and base or None)``: its base (and row means)
    added to its values in place."""
    for s, n_rows, m, lo, means in ops:
        z = x[s:s + n_rows * m].reshape(n_rows, m)
        if means is not None:
            so, loo = means
            np.add(z, (x[so:so + n_rows] + np.int64(loo + lo)).astype(np.int32)[:, None], out=z)
        elif lo:
            np.add(z, np.int32(lo), out=z)


def _decode(buf, rows: list[int], cols: list[int], workers: int) -> tuple[np.ndarray, list]:
    """The value buffer (int32, every claim in place) and per op ``(start, N, M)`` in it."""
    if not isinstance(buf, bytes):
        raise ClaimCodecError("not bytes")
    if len(buf) < 8 or buf[:4] != MAGIC:
        raise ClaimCodecError("bad magic")
    if struct.unpack_from("<I", buf, 4)[0] != len(rows):
        raise ClaimCodecError("wrong number of ops")
    off = 8
    heads, segs = [], []          # per op (N, M, its row means' segment or None, its main segment); (B, lo, n)
    for n_rows, n_cols in zip(rows, cols, strict=True):
        off = _need(buf, off, 5)
        m, flags = struct.unpack_from("<IB", buf, off - 5)
        if m != n_cols:
            raise ClaimCodecError("a column count other than the query's")
        if flags & ~_F_CENTRED:
            raise ClaimCodecError("bad flags")
        oseg = None
        if flags & _F_CENTRED:
            off = _need(buf, off, 5)
            oseg = len(segs)
            segs.append(struct.unpack_from("<Bi", buf, off - 5) + (n_rows,))
        off = _need(buf, off, 5)
        heads.append((n_rows, m, oseg, len(segs)))
        segs.append(struct.unpack_from("<Bi", buf, off - 5) + (n_rows * m,))
    pad = -off % 4
    off = _need(buf, off, pad)
    if any(buf[off - pad:off]):
        raise ClaimCodecError("non-zero header padding")
    if any(b > BMAX or not -_X_LIMIT < lo <= _X_LIMIT - (1 << b) for b, lo, _ in segs):
        raise ClaimCodecError("segment width or base out of range")
    if sum(b * n for b, _, n in segs) > 8 * (len(buf) - off):
        raise ClaimCodecError("slot streams longer than the input")
    total = sum(n for _, _, n in segs)
    # 1. the slots: one stream per width, each a contiguous block of x, and meanwhile (on this
    # thread, while the others unpack the streams) the exception list after them
    x = np.empty(total, dtype=np.int32)
    widths = sorted({s[0] for s in segs})
    order = [i for b in widths for i, s in enumerate(segs) if s[0] == b]   # the global segment order
    start = [0] * len(segs)
    jobs, sizes, base = [], [], 0
    for b in widths:
        members = [i for i in order if segs[i][0] == b]
        cnt = sum(segs[i][2] for i in members)
        more, off = _stream_jobs(buf, off, cnt, b, x.view(np.uint32)[base:base + cnt], workers)
        jobs += more
        # small ones here; a width 0 has no job (its values are all 0 slots)
        sizes += [4 * cnt // len(more) if more and cnt >= _THREADED else 0] * len(more)
        for i in members:
            start[i] = base
            base += segs[i][2]
    exc = {}
    _run([functools.partial(_exceptions, buf, off, total, exc)] + jobs, [0] + sizes, workers)
    if exc["n"]:
        # 2. the exceptions: each value is slot + lo + h 2**B, range-checked, stored minus its base
        _patch(x, exc["gaps"], exc["hz"], [(start[i],) + segs[i][:2] for i in order], total, workers)
    # 3. per op: the base (and the row means) added in place, ops in about equal parts on the workers
    ops = [(start[main], n, m, segs[main][1], None if means is None else (start[means], segs[means][1]))
           for n, m, means, main in heads]
    parts = workers if total >= _THREADED else 1
    cum = np.cumsum([0] + [n * m for _, n, m, _, _ in ops])
    cut = [int(c) for c in np.searchsorted(cum, [cum[-1] * i // parts for i in range(parts)])] + [len(ops)]
    _run([functools.partial(_add_bases, x, ops[a:e]) for a, e in zip(cut, cut[1:])],
         [4 * int(cum[e] - cum[a]) if parts > 1 else 0 for a, e in zip(cut, cut[1:])], workers)
    return x, [(s, n, m) for s, n, m, _, _ in ops]


def _exceptions(buf, off: int, total: int, out: dict) -> None:
    """The exception list at ``buf[off:]``, the end of the input, into ``out``: count, gaps, payloads."""
    off = _need(buf, off, 4)
    (n_exc,) = struct.unpack_from("<I", buf, off - 4)
    if n_exc > total or 2 * ((n_exc + 7) // 8) > len(buf) - off:    # each Rice vector: n_exc bits at least
        raise ClaimCodecError("more exceptions than values or bytes")
    out["gaps"], off = _rice_decode(buf, off, n_exc)
    out["hz"], off = _rice_decode(buf, off, n_exc)
    if off != len(buf):
        raise ClaimCodecError("trailing bytes")
    out["n"] = n_exc


def decode(buf, rows: list[int], cols: list[int], *, workers: int = 1) -> list[np.ndarray]:
    """The claim matrices of :func:`encode`'s bytes, int32 ``[rows[i], cols[i]]`` in op order (views
    of one buffer); raises :class:`ClaimCodecError` on anything else.

    ``rows`` are the ops' row counts (public) and ``cols`` their column counts on the query
    (``Verifier.claim_columns``): a header with any other is rejected before anything is
    allocated.  ``workers`` threads share the work on large proofs."""
    x, ops = _decode(buf, rows, cols, workers)
    return [x[s:s + n * m].reshape(n, m) for s, n, m in ops]


def decode_torch(buf, rows: list[int], cols: list[int], *, dtype: torch.dtype = torch.int64, workers: int = 1,
                 pin: bool = False) -> list[torch.Tensor]:
    """:func:`decode` into torch tensors of ``dtype``: int64, what ``Verifier.derive`` takes (widened in
    one pass on torch's threads), or int32 (the decoded buffer itself; copied to pinned memory
    with ``pin``, for non-blocking uploads to a GPU)."""
    x, ops = _decode(buf, rows, cols, workers)
    flat = torch.from_numpy(x)
    if dtype != torch.int32:
        flat = flat.to(dtype)
    elif pin:
        flat = flat.pin_memory()
    return [flat[s:s + n * m].view(n, m) for s, n, m in ops]


# ---------------------------------------------------------------------------- field elements
FIELD_BITS = 31
"""BabyBear elements are ``< 2**31``: 31-bit packing saves 1/32 of ``u`` and the opened columns."""


def field_size(numel: int) -> int:
    """Bytes of :func:`pack_field` for ``numel`` elements."""
    return 4 * FIELD_BITS * ((numel + 31) // 32)


def pack_field(tensors: list[torch.Tensor]) -> bytes:
    """Field elements (integer tensors with values in ``[0, 2**31)``), flattened and concatenated,
    at 31 bits each (:func:`pack32`)."""
    flat = torch.cat([t.detach().reshape(-1).cpu().to(torch.int64) for t in tensors]) if tensors else \
        torch.zeros(0, dtype=torch.int64)
    if flat.numel() and (int(flat.min()) < 0 or int(flat.max()) >= 1 << FIELD_BITS):
        raise Unencodable("field elements must be in [0, 2**31)")
    return pack32(flat.to(torch.int32).numpy().view(np.uint32), FIELD_BITS)


def unpack_field(buf, numel: int, *, workers: int = 1) -> np.ndarray:
    """The ``numel`` elements of :func:`pack_field`'s bytes (``uint32``, each ``< 2**31``; the
    verifier checks they are in the field); raises :class:`ClaimCodecError` unless ``buf`` is
    exactly such bytes."""
    if not isinstance(buf, bytes):
        raise ClaimCodecError("not bytes")
    if len(buf) != field_size(numel):
        raise ClaimCodecError("wrong length of packed field elements")
    return unpack32(buf, 0, numel, FIELD_BITS, workers=workers)[0]


# ---------------------------------------------------------------------------- lookup rows
def pack_rows(claims: list[torch.Tensor]) -> bytes:
    """The claims ``[d, M]`` of lookup tables (int8 values: one looked-up row of ``d`` weights per
    column), claim after claim, each as its ``M`` rows of ``d`` bytes."""
    if not claims:
        return b""
    flat = torch.cat([z.detach().T.reshape(-1) for z in claims])
    if flat.numel() and (int(flat.min()) < -128 or int(flat.max()) > 127):
        raise Unencodable("looked-up rows must be int8")
    return flat.to(torch.int8).cpu().numpy().tobytes()


def unpack_rows(buf, shapes: list[tuple[int, int]], *, dtype: torch.dtype = torch.int64,
                pin: bool = False) -> list[torch.Tensor]:
    """The claims of :func:`pack_rows`'s bytes, as ``dtype`` tensors of ``shapes`` (``[d, M]`` each,
    known to the verifier; int32 in pinned memory with ``pin``); raises :class:`ClaimCodecError`
    unless ``buf`` is exactly that many bytes."""
    if not isinstance(buf, bytes):
        raise ClaimCodecError("not bytes")
    if len(buf) != sum(d * m for d, m in shapes):
        raise ClaimCodecError("wrong length of looked-up rows")
    flat = torch.from_numpy(np.frombuffer(buf, dtype=np.int8).copy()).to(dtype)
    if pin and dtype == torch.int32:
        flat = flat.pin_memory()
    out, o = [], 0
    for d, m in shapes:
        out.append(flat[o:o + d * m].view(m, d).T)
        o += d * m
    return out

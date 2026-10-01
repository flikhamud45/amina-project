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
decodes the exception list's two Rice vectors while the streams are unpacked: the streams' offsets
follow from the header, and the second vector's from the bits set in the first one's unary levels.

Encoders.  :func:`encode` plans each op on a sample of its rows and gives the same bytes on a host
and on a device (the per-op encoder that defined the format is ``tests/claimcodec_reference.py``;
both are tested against it).  The host's runs numpy on ``workers`` threads; a device's runs torch
where the claims are, with a fixed number of kernels plus about ten per distinct width (not per op)
and four copies back: every op's statistics at once (one sort), the exceptions' number, the Rice
parameters' statistics, and the encoding (about 2.3 bytes a claim).

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

import bisect
import functools
import itertools
import struct

import numpy as np
import torch

from .commitment import map_threaded

__all__ = ["ClaimCodecError", "Unencodable", "MAGIC", "BMAX", "encode", "decode", "decode_torch", "narrow",
           "FIELD_BITS", "field_size", "pack_field", "unpack_field", "pack_rows", "unpack_rows"]

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
_JOB = 1 << 18              # values per job of the host encoder's threads (packing, scanning, statistics)
_SPLIT = 1 << 16            # exceptions from which the decoder's workers decode both Rice vectors at once


class ClaimCodecError(ValueError):
    """Malformed encoded proof: the verifier rejects."""


class Unencodable(ValueError):
    """Values the encoding cannot carry -- a claim outside the range check, a looked-up row outside
    int8, a field element of 32 bits: no honest message holds them, and whatever a prover with such
    values sends instead, the verifier rejects or decodes into other values."""


def _parallel(fn, items: list, sizes: list[int], workers: int) -> list:
    """``[fn(x) for x in items]`` on up to ``workers`` threads (numpy releases the GIL in each pass), in at
    most ``workers`` groups of consecutive items of about equal ``sizes`` (bytes), each of at least
    ``_JOB`` values' worth: a smaller job costs the pool more than it saves."""
    total = sum(sizes)
    parts = min(workers, len(items), total // (4 * _JOB)) if workers > 1 else 1
    if parts <= 1:
        return [fn(x) for x in items]
    cum = np.cumsum(np.asarray(sizes, dtype=np.int64))
    inner = (int(np.searchsorted(cum, total * i // parts, side="right")) for i in range(1, parts))
    cut = sorted({0, len(items), *inner})
    groups = [(a, e) for a, e in zip(cut, cut[1:]) if e > a]
    done = map_threaded(lambda g: [fn(items[i]) for i in range(*g)], groups, [total // parts] * len(groups), workers)
    return [r for part in done for r in part]


def _cuts(n: int, parts: int) -> list[tuple[int, int]]:
    """``[0, n)`` in ``parts`` near-equal consecutive ranges (none empty unless ``n`` is 0)."""
    parts = max(1, min(parts, n))
    return [(n * i // parts, n * (i + 1) // parts) for i in range(parts)]


# ---------------------------------------------------------------------------- host <-> device
# The device encoder copies to and from the host only through these: ``_upload`` (pinned memory, no
# wait), ``_d2h`` and ``_nonzero_dev`` (each one wait for the device).

def _upload(a: np.ndarray, device: torch.device) -> torch.Tensor:
    """A host array on ``device``, through pinned memory and without waiting for the device (a copy from
    pageable memory synchronises with it)."""
    t = torch.from_numpy(np.ascontiguousarray(a))
    if device.type == "cpu":
        return t
    if device.type == "cuda":
        return t.pin_memory().to(device, non_blocking=True)
    return t.to(device)


def _d2h(t: torch.Tensor) -> np.ndarray:
    """``t`` on the host: one copy back, through pinned memory, and one wait for the device."""
    if t.device.type == "cpu":
        return t.numpy()
    if t.device.type == "cuda":
        host = torch.empty(t.shape, dtype=t.dtype, pin_memory=True)
        host.copy_(t, non_blocking=True)
        torch.cuda.current_stream(t.device).synchronize()
        return host.numpy()
    return t.cpu().numpy()


def _nonzero_dev(t: torch.Tensor) -> torch.Tensor:
    """The positions of ``t``'s non-zero entries (1-D): one wait for the device, for their number."""
    return torch.nonzero(t).view(-1)


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


def _pack_cols(v: np.ndarray, k: int, w: np.ndarray, c0: int, c1: int) -> None:
    """Columns ``[c0, c1)`` of the words ``w`` (``[k, G]``, zero) of :func:`pack32` of ``v`` (``uint32``,
    each ``< 2**k``), lane by lane: lane ``j`` (values ``[j G, (j + 1) G)``) is shifted into its first word
    (written, not or-ed, where it starts that word: no lower lane wrote it) and spills into the next
    one (which no lower lane wrote either)."""
    g, n = w.shape[1], v.size
    tmp = np.empty(c1 - c0, dtype=np.uint32)
    for j in range(32):
        m = min(c1, n - j * g) - c0              # lane j's values in these columns
        if m <= 0:
            break
        src = v[j * g + c0:j * g + c0 + m]
        wi, sh = (j * k) >> 5, (j * k) & 31
        row = w[wi, c0:c0 + m]
        if sh == 0:
            row[...] = src
            continue
        t = tmp[:m]
        np.left_shift(src, np.uint32(sh), out=t)
        np.bitwise_or(row, t, out=row)
        if sh + k > 32:
            np.right_shift(src, np.uint32(32 - sh), out=w[wi + 1, c0:c0 + m])


def _pack_into(v: np.ndarray, k: int, w: np.ndarray, workers: int = 1) -> None:
    """:func:`pack32` of ``v`` (``uint32``) into the zero words ``w`` (``[k, G]``), columns split over
    ``workers`` threads in jobs of about ``_JOB`` values."""
    g = w.shape[1]
    if k == 0 or g == 0:
        return
    jobs = _cuts(g, min(workers, max(1, v.size // _JOB))) if workers > 1 else [(0, g)]
    _parallel(lambda c: _pack_cols(v, k, w, *c), jobs, [4 * 32 * (e - a) for a, e in jobs], workers)


def pack32(v: np.ndarray, k: int, workers: int = 1) -> bytes:
    """``v`` (``uint32``, all ``< 2**k``, ``k <= 32``) at ``k`` bits each: ``4 k ceil(n / 32)`` bytes."""
    n = v.size
    if k == 0 or n == 0:
        return b""
    g = (n + 31) // 32
    if n <= _SMALL:
        vv = np.zeros(32 * g, dtype=np.uint32)
        vv[:n] = v
        return _pack32_small(vv.reshape(32, g), k, g)
    w = np.zeros((k, g), dtype="<u4")
    _pack_into(np.asarray(v, dtype=np.uint32), k, w, workers)
    return w.tobytes()


@functools.lru_cache(maxsize=None)
def _lane_maps_on(k: int, device: torch.device) -> tuple:
    """:func:`_lane_maps` as int64 tensors on ``device`` (uploaded once)."""
    wi, sh, rsh, w2, spill = _lane_maps(k)
    return tuple(_upload(np.asarray(a, dtype=np.int64), device) for a in (wi, sh, rsh, w2, spill[:, None]))


def _pack_dev(v: torch.Tensor, k: int) -> torch.Tensor:
    """The words ``[k, G]`` (int32: the little-endian bytes of :func:`pack32`'s) of the integers ``v``
    (each ``< 2**k``), all 32 lanes at once with about ten torch ops on ``v``'s device."""
    n, dev = v.numel(), v.device
    g = (n + 31) // 32
    wi, sh, rsh, w2, spill = _lane_maps_on(k, dev)
    vv = torch.zeros(32 * g, dtype=torch.int64, device=dev)
    vv[:n] = v.reshape(-1)
    vv = vv.view(32, g)
    words = torch.zeros(k + 1, g, dtype=torch.int64, device=dev)
    words.index_add_(0, wi, (vv << sh) & 0xFFFFFFFF)          # the bits of different lanes are disjoint: sum == or
    words.index_add_(0, w2, (vv >> rsh) * spill)
    return words[:k].to(torch.int32)                           # the low 32 bits: the uint32 words' bytes


def _pack32_torch(v: torch.Tensor, k: int) -> bytes:
    """:func:`pack32` with torch ops on ``v``'s device (:func:`_pack_dev`), the words copied back once."""
    if k == 0 or v.numel() == 0:
        return b""
    return _d2h(_pack_dev(v, k)).astype("<i4", copy=False).tobytes()


def _add_bases_in(out: np.ndarray, bases, a: int, e: int) -> None:
    """Adds to the slots ``out[a:e]`` (uint32, of one stream) the base of each segment over them: ``bases``
    are ``(starts, ends, bases)`` of the stream's segments in order (lists; a base as ``np.uint32``, or 0
    for none), or ``None``.  uint32 arithmetic wraps, and every value fits int32."""
    if bases is None or e <= a:
        return
    starts, ends, los = bases
    i = bisect.bisect_right(ends, a)
    while i < len(starts) and starts[i] < e:
        if los[i]:
            seg = out[max(starts[i], a):min(ends[i], e)]
            np.add(seg, los[i], out=seg)
        i += 1


def _unpack_small(w: np.ndarray, k: int, n: int, out: np.ndarray, bases=None) -> None:
    v = _unpack32_small(w, k, w.shape[1]).reshape(-1)
    if v[n:].any():
        raise ClaimCodecError("non-zero padding in a slot stream")
    if bases is None:
        out[:n] = v[:n]
        return
    for a, e, lo in zip(*bases):        # each segment's slots copied out with its base added, in one pass
        if lo:
            np.add(v[a:e], lo, out=out[a:e])
        else:
            out[a:e] = v[a:e]


def _unpack_lanes(w: np.ndarray, k: int, n: int, out: np.ndarray, js, bases=None) -> None:
    """Lanes ``js`` of the stream ``w`` of ``n`` values into ``out[:n]`` (lane ``j`` holds values
    ``[j G, (j + 1) G)``; the lane holding padding goes through a buffer, whose padding is checked),
    each lane's segments' bases added while it is in cache (:func:`_add_bases_in`)."""
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
        _add_bases_in(out, bases, a, min(a + g, n))


def _stream_jobs(buf, offset: int, n: int, k: int, out: np.ndarray, workers: int,
                 bases=None) -> tuple[list, int]:
    """Jobs that unpack the stream of ``n`` ``k``-bit values at ``buf[offset:]`` into exactly
    ``out[:n]`` (all 32 lanes in one vectorised step up to ``_SMALL`` values; from ``_THREADED``
    values, lanes split in ``workers`` jobs), adding the segments' ``bases`` (:func:`_add_bases_in`),
    and the offset after the stream."""
    g = (n + 31) // 32
    nbytes = 4 * k * g
    if offset + nbytes > len(buf):
        raise ClaimCodecError("truncated slot stream")
    if n == 0 or k == 0:
        out[:n] = 0
        _add_bases_in(out, bases, 0, n)
        return [], offset
    w = np.frombuffer(buf, dtype="<u4", count=k * g, offset=offset).reshape(k, g)
    if n <= _SMALL:
        return [functools.partial(_unpack_small, w, k, n, out, bases=bases)], offset + nbytes
    parts = min(workers, 32) if n >= _THREADED else 1
    if n < _THREADED:               # one job, the bases in one pass after it (a lane's call costs ~7 us)
        return [functools.partial(_unpack_stream, w, k, n, out, bases)], offset + nbytes
    return [functools.partial(_unpack_lanes, w, k, n, out, range(i, 32, parts), bases=bases)
            for i in range(parts)], offset + nbytes


def _unpack_stream(w: np.ndarray, k: int, n: int, out: np.ndarray, bases) -> None:
    _unpack_lanes(w, k, n, out, range(32))
    _add_bases_in(out, bases, 0, n)


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


def _rice_k_hist(n: int, kmin: int, hist) -> int:
    """The ``k`` with the fewest bits, among those that keep every quotient ``<= _QCAP`` (``k >= kmin``),
    from ``hist[x]``: the number of the ``n`` values ``v`` with ``v >> kmin == x`` (``x < 16``).  As
    ``v >> k == (v >> kmin) >> (k - kmin)``, the bits of each ``k`` are exactly those of the values."""
    xs = np.arange(16, dtype=np.int64)
    hist = np.asarray(hist, dtype=np.int64)
    best = None
    for k in range(kmin, _KMAX + 1):
        q = xs >> (k - kmin)
        bits = n * k + int(hist @ q) + int(hist[q < _QCAP].sum())
        if best is None or bits < best[0]:
            best = (bits, k)
        elif k > best[1] + 1:
            break
    return best[1]


def _rice_k(v: np.ndarray) -> int:
    """The ``k`` with the fewest bits, among those that keep every quotient ``<= _QCAP``."""
    kmin = max(0, int(v.max()).bit_length() - 4)
    return _rice_k_hist(v.size, kmin, np.bincount(v >> kmin, minlength=16))


def _rice_encode(v: np.ndarray, workers: int = 1) -> bytes:
    """Non-negative integers ``< 2**36``; an empty vector is empty (the decoder knows the length)."""
    v = np.asarray(v, dtype=np.int64)
    if v.size == 0:
        return b""
    if int(v.min()) < 0 or int(v.max()) >= 1 << (_KMAX + 4):
        raise ValueError("Rice values must be in [0, 2**36)")
    k = _rice_k(v)
    low = pack32((v & ((1 << k) - 1)).astype(np.uint32), k, workers)
    return bytes([k]) + low + _pack_levels((v >> k).astype(np.uint8))


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
# The plan.  Each op is planned on a sample of its rows (every ``step``-th row: at most
# ``_PLAN_ROWS`` rows or about ``_PLAN_VALUES`` values): the centre ``c`` (the rounded median) of the
# sampled values, of their residuals about the sampled rows' integer means and of those means, the
# histogram of the bit lengths of each sample's zigzagged deviations from its centre, and from them
# the estimated bits of every width (:func:`_costs`).  A centred op's row means (all ``N``) are
# planned the same way, without sampling.  Both encoders compute exactly these integers and do the
# same float64 arithmetic on them (one row per sample), so they choose the same widths and bases as
# the per-op encoder that defined the format (``tests/claimcodec_reference.py``) and give its bytes.
#
# The host's (:func:`_encode_host`, numpy, ``workers`` threads) takes CPU claims; it works in place on
# scratch buffers (a pass into a fresh large array pays its page faults).  A device's
# (:func:`_encode_device`, torch) runs where the claims are, with a number of kernels that grows with
# the number of distinct widths, not of ops, and four copies back: the statistics of all ops (min/max,
# medians and histograms: one sort of every sample, then two binary searches per bit length), the
# exceptions' number, the Rice parameters' statistics, and the encoding (packed words, Rice low parts,
# unary quotients: ~2.3 bytes a claim).

_BINS = 64                  # bit lengths of zigzagged deviations: < 34
_SPAN = 34                  # the bit lengths a zigzagged deviation of two int32 values can have: 0..33
_BS = np.arange(BMAX + 1, dtype=np.float64)
_LS = np.arange(_BINS, dtype=np.float64)


def _costs(hist, n, scale) -> np.ndarray:
    """Estimated bits at every width ``B = 0..BMAX`` (``[S, BMAX + 1]``) of ``S`` segments, from their
    samples: ``hist[s]`` the counts of the bit lengths of the zigzagged deviations from the centre,
    ``n[s]`` the sample's size and ``scale[s]`` the segment's over it (``B`` per value, about
    ``log2(n / e) + 3.5`` per exception plus its high part's bits)."""
    h = np.asarray(hist, dtype=np.float64).reshape(-1, _BINS)
    ge = np.cumsum(h[:, ::-1], axis=1)[:, ::-1]                # values whose zigzag has >= j bits
    gel = np.cumsum((h * _LS)[:, ::-1], axis=1)[:, ::-1]
    e = ge[:, 1:BMAX + 2]                                      # exceptions at width B: more than B bits
    payload = gel[:, 1:BMAX + 2] - _BS * e                     # about their high parts' bits
    nn = np.asarray(n, dtype=np.float64).reshape(-1, 1)
    exc = e * (np.log2(np.maximum(nn, 1.0) / np.maximum(e, 1.0)) + 3.5) + payload
    return np.asarray(scale, dtype=np.float64).reshape(-1, 1) * (nn * _BS + exc)


def _round_half(s):
    """``round(s / 2)`` with halves to even (``np.round``): the rounded median of two middle values whose
    sum is ``s`` (integer arrays or tensors; ``>>`` floors)."""
    q = s >> 1
    return q + ((s & 1) & (q & 1))


def _base(c, b):
    """The base centring the ``2**b`` slots on ``c``, within the range the decoder accepts (arrays too)."""
    return np.minimum(np.maximum(c - ((1 << b) >> 1), 1 - _X_LIMIT), _X_LIMIT - (1 << b))


def _cum0(a: np.ndarray) -> np.ndarray:
    out = np.zeros(a.size + 1, dtype=np.int64)
    np.cumsum(a, out=out[1:])
    return out


class _Layout:
    """What the shapes ``[(N, M), ...]`` of a query's claims fix (:func:`_layout`: once per shape): each
    op's place in the flat buffer of all claims, its sampled rows (every ``step``-th of its ``N``) and
    their place in the flat sample, and their copies on a device."""

    def __init__(self, shapes: tuple):
        sh = np.array(shapes, dtype=np.int64).reshape(-1, 2)
        rows, cols = sh[:, 0], sh[:, 1]
        self.n_ops = n = len(sh)
        self.rows, self.cols, self.size = rows, cols, rows * cols
        self.off = _cum0(self.size)                            # each op's values in the flat buffer
        self.total = int(self.off[-1])
        step = np.maximum(np.maximum(1, rows // _PLAN_ROWS), self.size // _PLAN_VALUES)
        r = np.where(rows > 0, (rows + step - 1) // step, 0)  # len(range(0, N, step))
        self.sampled, self.ssize = r, r * cols
        self.scale = rows / np.maximum(r, 1)                   # N / rows sampled, as Python divides
        self.soff, self.roff = _cum0(self.ssize), _cum0(r)     # each op's sample, its sampled rows
        self.op_of_row = np.repeat(np.arange(n, dtype=np.int64), r)
        self.row_m = cols[self.op_of_row]
        t = np.arange(self.roff[-1], dtype=np.int64) - self.roff[:-1][self.op_of_row]
        row_start = self.off[:-1][self.op_of_row] + t * step[self.op_of_row] * self.row_m
        self.srow = _cum0(self.row_m)                          # the sampled rows tile the sample
        self.sample_row = np.repeat(np.arange(self.roff[-1], dtype=np.int64), self.row_m)
        within = np.arange(self.soff[-1], dtype=np.int64) - self.srow[:-1][self.sample_row]
        self.sample_idx = row_start[self.sample_row] + within        # each sampled value's place in the buffer
        self.op_of_sample = self.op_of_row[self.sample_row]
        self.frow0 = _cum0(rows)                               # each op's first row among all rows
        self.eligible = (cols >= 2) & (r > 0)                  # may be centred
        self._dev: dict = {}

    def on(self, device: torch.device, centre: bool) -> dict:
        """The device tables of :func:`_encode_device` (uploaded once per device)."""
        key = (device, centre)
        if key not in self._dev:
            self._dev[key] = self._tables(device, centre)
        return self._dev[key]

    def _tables(self, device: torch.device, centre: bool) -> dict:
        n = self.n_ops
        up = functools.partial(_upload, device=device)
        t = {"sample_idx": up(self.sample_idx), "pow2": up(np.int64(1) << np.arange(41, dtype=np.int64))}
        kinds = [(self.op_of_sample, self.ssize)]
        if centre:
            op_of_frow = np.repeat(np.arange(n, dtype=np.int64), self.rows)
            frow_m = self.cols[op_of_frow]
            t.update(sample_row=up(self.sample_row), srow=up(self.srow), row_half=up(self.row_m // 2),
                     row_div=up(np.maximum(self.row_m, 1)), frow=up(_cum0(frow_m)), frow_half=up(frow_m // 2),
                     frow_div=up(np.maximum(frow_m, 1)))
            kinds += [(n + self.op_of_sample, self.ssize), (2 * n + self.op_of_row, self.sampled),
                      (3 * n + op_of_frow, self.rows)]
        # the sort key of each statistic's value v: (segment << 32) + v + 2**31 (v clamped to int32)
        t["segkey"] = up(np.concatenate([(seg << 32) + (1 << 31) for seg, _ in kinds]))
        size = np.concatenate([s for _, s in kinds])
        start = _cum0(size)
        last = max(int(start[-1]) - 1, 0)
        t["mid_lo"] = up(np.minimum(start[:-1] + np.maximum(size - 1, 0) // 2, last))
        t["mid_hi"] = up(np.minimum(start[:-1] + size // 2, last))
        t["nonempty"] = up((size > 0).astype(np.int64))
        t["segbase"] = up((np.arange(len(size), dtype=np.int64) << 32) + (1 << 31))
        half = np.r_[0, np.int64(1) << np.arange(_SPAN - 1, dtype=np.int64)]       # 2**(B-1), B = 0.._SPAN-1
        t["half"], t["half_hi"] = up(half[None, :]), up((half - (half > 0))[None, :])
        t["n_seg"] = len(size)
        return t


@functools.lru_cache(maxsize=4)          # each holds its device tables (Qwen3-4B, 36 blocks: ~70 MB)
def _layout(shapes: tuple) -> _Layout:
    return _Layout(shapes)


def _decide(lay: _Layout, c: np.ndarray, hist: np.ndarray, centre: bool):
    """Per op: whether it is centred, and its values' (or residuals') width and base, from the centres
    ``c`` and histograms ``hist`` of its samples: ``[n]`` of the values, then if ``centre`` ``[n]`` of the
    residuals and ``[n]`` of the sampled rows' means."""
    n = lay.n_ops
    kinds = 3 if centre else 1
    costs = _costs(hist[:kinds * n], np.tile(lay.ssize, kinds) if not centre else
                   np.concatenate([lay.ssize, lay.ssize, lay.sampled]), np.tile(lay.scale, kinds))
    b = np.argmin(costs, axis=1)
    cost = costs[np.arange(b.size), b]
    centred = np.zeros(n, dtype=bool)
    width, mid = b[:n], c[:n]
    if centre:
        centred = lay.eligible & ((cost[2 * n:] + 40) + cost[n:2 * n] < cost[:n])   # 40: the offsets' B and lo
        width, mid = np.where(centred, b[n:2 * n], width), np.where(centred, c[n:2 * n], mid)
    return centred, width, _base(mid, width)


def _decide_means(c: np.ndarray, hist: np.ndarray, rows: np.ndarray):
    """Width and base of each centred op's row means (all ``N`` of them: no sampling)."""
    b = np.argmin(_costs(hist, rows, np.ones(rows.size)), axis=1)
    return b, _base(c, b)


def _write_header(lay: _Layout, centred, seg_b, seg_lo) -> bytes:
    """The header and its padding: per op ``M``, the flags and each segment's ``B`` and ``lo``
    (``seg_b``, ``seg_lo`` in segment order)."""
    fmt, vals, j = ["<4sI"], [MAGIC, lay.n_ops], 0
    seg_b, seg_lo = seg_b.tolist(), seg_lo.tolist()
    for m, cen in zip(lay.cols.tolist(), centred.tolist()):
        if cen:
            fmt.append("IBBiBi")
            vals += [m, _F_CENTRED, seg_b[j], seg_lo[j], seg_b[j + 1], seg_lo[j + 1]]
            j += 2
        else:
            fmt.append("IBBi")
            vals += [m, 0, seg_b[j], seg_lo[j]]
            j += 1
    head = struct.pack("".join(fmt), *vals)
    return head + bytes(-len(head) % 4)


def _segments(lay: _Layout, centred, width, lo, means_b, means_lo):
    """The segments in segment order (per op: its row means' if centred, then its values' or
    residuals'): width, base, kind (0 values, 1 residuals, 2 row means), op and size."""
    n = lay.n_ops
    ci = np.flatnonzero(centred)
    main = np.arange(n, dtype=np.int64) + np.cumsum(centred, dtype=np.int64)
    means = main[ci] - 1
    ns = n + ci.size
    seg = {k: np.zeros(ns, dtype=np.int64) for k in ("b", "lo", "kind", "op", "size")}
    for k, a, v in (("b", main, width), ("lo", main, lo), ("kind", main, centred),
                    ("op", main, np.arange(n, dtype=np.int64)), ("size", main, lay.size),
                    ("b", means, means_b[ci]), ("lo", means, means_lo[ci]), ("kind", means, 2), ("op", means, ci),
                    ("size", means, lay.rows[ci])):
        seg[k][a] = v
    return seg


def encode(claims: list, *, centre: bool = True, impl: str | None = None, workers: int | None = None) -> bytes:
    """``PVC3`` bytes of the claim matrices (``[N, M]`` integer tensors or arrays, in the graph's
    weight-op order, every ``|z| < 2**29``).  ``centre=False`` never centres (for measurements).

    Tensors on a device are encoded there (``impl="device"``; :func:`_encode_device`): only the
    statistics, the exceptions' number, the Rice statistics and the encoding itself come to the host,
    each in one copy.  Host claims are encoded on ``workers`` threads (``impl="host"``; default
    ``torch.get_num_threads()``).  Both give the same bytes."""
    zts = [z.detach() if torch.is_tensor(z) else torch.from_numpy(np.asarray(z)) for z in claims]
    if any(z.dim() != 2 for z in zts):
        raise ValueError("claims must be 2-D")
    devices = {z.device for z in zts}
    if impl is None:
        impl = "device" if len(devices) == 1 and next(iter(devices)).type != "cpu" else "host"
    if impl not in ("host", "device"):
        raise ValueError(f"unknown encoder {impl!r}")
    if impl == "device" and zts:
        if len(devices) != 1:
            raise ValueError("the device encoder takes claims on one device")
        return _encode_device(zts, centre)
    return _encode_host([z.cpu() for z in zts], centre, workers or torch.get_num_threads())


def narrow(z: torch.Tensor) -> torch.Tensor:
    """A claim kept for :func:`encode` on its device as int32, clipped to the range check's limits (so a
    claim with ``|z| >= 2**29`` still has one, and the encoder still refuses it)."""
    return torch.clamp(z, -_Z_LIMIT, _Z_LIMIT).to(torch.int32)


# -- the host's encoder

def _as_numpy(z: torch.Tensor) -> np.ndarray:
    return z.numpy() if z.dtype != torch.bool else z.to(torch.int64).numpy()


def _medians(work: np.ndarray, starts) -> np.ndarray:
    """The rounded medians (``np.round(np.median(.))``; 0 if empty) of the runs ``work[starts[s]:starts[s + 1]]``,
    from their two middle values; partitions each run in place."""
    starts = starts.tolist()
    out = np.zeros(len(starts) - 1, dtype=np.int64)
    for s, (a, e) in enumerate(zip(starts, starts[1:])):
        n = e - a
        if n:
            k1, k2 = (n - 1) // 2, n // 2
            run = work[a:e]
            run.partition([k1, k2] if k1 != k2 else k1)
            out[s] = int(run[k1]) + int(run[k2])
    return _round_half(out)


def _hist(src: np.ndarray, starts, c: np.ndarray, bins: np.ndarray, n_seg: int, work: np.ndarray,
          tmp: np.ndarray) -> np.ndarray:
    """Per run ``src[starts[s]:starts[s + 1]]`` the counts of the bit lengths of its zigzagged deviations from
    ``c[s]`` (``[n_seg, _BINS]``), in place in the int64 scratch ``work`` and ``tmp`` (no allocation the size of
    ``src``: on a fresh one each pass pays its page faults); ``bins`` is ``64 run - 1022`` per value."""
    starts = starts.tolist()
    for s, (a, e) in enumerate(zip(starts, starts[1:])):
        np.subtract(src[a:e], c[s], out=work[a:e])
    np.right_shift(work, 63, out=tmp)
    np.left_shift(work, 1, out=work)
    np.bitwise_xor(work, tmp, out=work)                        # zz = zigzag(d) < 2**33
    f = tmp.view(np.float64)
    np.add(work, 0.5, out=f)                                   # exact; its exponent is zz's bit length + 1022
    e = f.view(np.int64)
    np.right_shift(e, 52, out=e)
    np.add(e, bins, out=e)
    return torch.bincount(torch.from_numpy(e), minlength=n_seg * _BINS).numpy().reshape(n_seg, _BINS)


def _stats_host(lay: _Layout, x: np.ndarray, o0: int, o1: int, centre: bool) -> tuple[np.ndarray, np.ndarray]:
    """``(c, hist)`` of the samples of ops ``[o0, o1)``: ``[kinds, k]`` and ``[kinds, k, _BINS]`` (the values;
    with ``centre`` also the residuals and the sampled rows' means)."""
    k, s0, s1, r0, r1 = o1 - o0, lay.soff[o0], lay.soff[o1], lay.roff[o0], lay.roff[o1]
    s = x[lay.sample_idx[s0:s1]]                               # int32
    starts = lay.soff[o0:o1 + 1] - s0
    bins = lay.op_of_sample[s0:s1] * _BINS + (-1022 - o0 * _BINS)
    work, tmp = np.empty(s.size, dtype=np.int64), np.empty(s.size, dtype=np.int64)
    np.copyto(work, s)
    c = [_medians(work, starts)]
    hist = [_hist(s, starts, c[0], bins, k, work, tmp)]
    if centre:
        cs = np.zeros(s.size + 1, dtype=np.int64)
        np.cumsum(s, dtype=np.int64, out=cs[1:])
        bounds = lay.srow[r0:r1 + 1] - s0
        m = lay.row_m[r0:r1]
        off = np.floor_divide(cs[bounds[1:]] - cs[bounds[:-1]] + m // 2, np.maximum(m, 1))   # sampled rows' means
        res = np.empty(s.size, dtype=np.int64)
        rst = (lay.roff[o0:o1 + 1] - r0).tolist()
        for i, (a, e) in enumerate(zip(starts.tolist(), starts.tolist()[1:])):
            if e > a:                                          # residuals: an op's rows have one length
                mm = int(lay.cols[o0 + i])
                np.subtract(s[a:e].reshape(-1, mm), off[rst[i]:rst[i + 1], None], out=res[a:e].reshape(-1, mm))
        np.copyto(work, res)
        c.append(_medians(work, starts))
        hist.append(_hist(res, starts, c[1], bins, k, work, tmp))
        rows = lay.roff[o0:o1 + 1] - r0
        ow = off.copy()
        c.append(_medians(ow, rows))
        hist.append(_hist(off, rows, c[2], lay.op_of_row[r0:r1] * _BINS + (-1022 - o0 * _BINS), k,
                          np.empty_like(off), np.empty_like(off)))
    return np.stack(c), np.stack(hist)


def _op_chunks(weights: np.ndarray, workers: int) -> list[tuple[int, int]]:
    """Consecutive op ranges of about equal ``weights`` (at least ``_JOB`` each), one per worker at most."""
    n = weights.size
    total = int(weights.sum())
    parts = max(1, min(workers, total // _JOB, n))
    if parts == 1:
        return [(0, n)]
    cum = np.cumsum(weights)
    cut = [0] + [int(np.searchsorted(cum, total * i // parts)) + 1 for i in range(1, parts)] + [n]
    cut = sorted(set(min(max(c, 0), n) for c in cut))
    return list(zip(cut, cut[1:]))


def _encode_host(zts: list, centre: bool, workers: int) -> bytes:
    """:func:`encode` of CPU claims with numpy on ``workers`` threads."""
    lay = _layout(tuple((int(z.shape[0]), int(z.shape[1])) for z in zts))
    n = lay.n_ops
    # 1. every claim clipped into one int32 buffer: a claim outside the range check stays outside
    x = np.empty(lay.total, dtype=np.int32)

    def clip(i):
        a, e = lay.off[i], lay.off[i + 1]
        if e > a:
            np.clip(_as_numpy(zts[i]), -_Z_LIMIT, _Z_LIMIT, out=x[a:e].reshape(zts[i].shape), casting="unsafe")

    _parallel(clip, list(range(n)), (8 * lay.size).tolist(), workers)
    if lay.total:
        lo, hi = (int(v) for v in torch.aminmax(torch.from_numpy(x)))
        if hi >= _Z_LIMIT or lo <= -_Z_LIMIT:
            raise Unencodable("a claim outside the range check cannot be encoded")
    # 2. the plan, from the samples of ops in chunks on the workers
    centre = bool(centre and lay.eligible.any())
    kinds = 3 if centre else 1
    chunks = _op_chunks(lay.ssize * kinds, workers)
    parts = _parallel(lambda ch: _stats_host(lay, x, *ch, centre), chunks,
                      [8 * kinds * int(lay.soff[e] - lay.soff[a]) for a, e in chunks], workers)
    c = np.concatenate([p[0] for p in parts], axis=1).reshape(-1)
    hist = np.concatenate([p[1] for p in parts], axis=1).reshape(-1, _BINS)
    centred, width, lo = _decide(lay, c, hist, centre)
    ci = np.flatnonzero(centred)
    means = {}
    means_b = np.zeros(n, dtype=np.int64)
    means_lo = np.zeros(n, dtype=np.int64)
    if ci.size:                    # 3. the row means of the centred ops (all their rows), and their plan
        def row_means(i):
            m = int(lay.cols[i])
            z = x[lay.off[i]:lay.off[i + 1]].reshape(-1, m)
            return np.floor_divide(z.sum(1, dtype=np.int64) + m // 2, m)

        mv = _parallel(row_means, ci.tolist(), (4 * lay.size[ci]).tolist(), workers)
        means = dict(zip(ci.tolist(), mv))
        flat = np.concatenate(mv)
        starts = _cum0(lay.rows[ci])
        cm = _medians(flat.copy(), starts)
        bins = np.repeat(np.arange(ci.size, dtype=np.int64) * _BINS - 1022, lay.rows[ci])
        hm = _hist(flat, starts, cm, bins, ci.size, np.empty_like(flat), np.empty_like(flat))
        means_b[ci], means_lo[ci] = _decide_means(cm, hm, lay.rows[ci])
    ops = []
    for i in range(n):
        z = x[lay.off[i]:lay.off[i + 1]]
        if centred[i]:
            ops.append((int(lay.cols[i]), [(int(means_b[i]), int(means_lo[i]), means[i]),
                                           (int(width[i]), int(lo[i]), (z.reshape(-1, int(lay.cols[i])), means[i]))]))
        else:
            ops.append((int(lay.cols[i]), [(int(width[i]), int(lo[i]), z)]))
    return _assemble(ops, workers)


def _values(x) -> np.ndarray | tuple:
    if isinstance(x, tuple):
        return x
    return _as_numpy(x).reshape(-1) if torch.is_tensor(x) else np.asarray(x).reshape(-1)


def _numel(x) -> int:
    return x[0].size if isinstance(x, tuple) else x.size


def _fill(d: np.ndarray, lo: int, x) -> None:
    """``d = x - lo`` (int32: ``|x - lo| < 2**31``); ``x = (Z, o)`` stands for the residuals ``Z - o[:, None]``."""
    if isinstance(x, tuple):
        z, o = x
        np.subtract(z, (o.astype(np.int64) + lo)[:, None], out=d.reshape(z.shape), casting="unsafe")
    else:
        np.subtract(x, np.int64(lo), out=d, casting="unsafe")


def _scan(d: np.ndarray, b: int, a: int, e: int) -> tuple[np.ndarray, np.ndarray]:
    """The exceptions among ``d[a:e]`` (values outside ``[0, 2**b)``: positions and high parts
    ``d >> b``), and those values cut to their slots in place."""
    part = d[a:e]
    idx = np.flatnonzero(part.view(np.uint32) >= (1 << b)) if b else np.flatnonzero(part)
    h = part[idx].astype(np.int64)
    if b:
        h >>= b
        np.bitwise_and(part, (1 << b) - 1, out=part)
    return idx + a, h


def _assemble(ops: list, workers: int = 1) -> bytes:
    """The ``PVC3`` bytes of ``ops``: per op ``(M, segments)``, with one segment ``(B, lo, x)`` or two (the row
    means', then the residuals'), ``x`` a flat int32 tensor or array with ``|x| < 2**30`` (or, for residuals,
    ``(Z [N, M], row means [N])``).  Each width's segments are cut to slots and scanned for exceptions in
    jobs of about ``_JOB`` values and packed by column ranges, on ``workers`` threads."""
    head = [MAGIC, struct.pack("<I", len(ops))]
    segs = []
    for m, op_segs in ops:
        head.append(struct.pack("<IB", m, _F_CENTRED if len(op_segs) == 2 else 0))
        for b, lo, x in op_segs:
            head.append(struct.pack("<Bi", b, lo))
            segs.append((int(b), int(lo), _values(x)))
    head = b"".join(head)
    widths = sorted({s[0] for s in segs})
    groups = {b: [(lo, x) for bb, lo, x in segs if bb == b] for b in widths}       # segment order within a width
    counts = {b: sum(_numel(x) for _, x in groups[b]) for b in widths}
    body = np.zeros(sum(4 * b * ((counts[b] + 31) // 32) for b in widths), dtype=np.uint8)
    exc_pos, exc_h, base, at = [], [], 0, 0
    for b in widths:
        cnt = counts[b]
        d = np.empty(cnt, dtype=np.int32)
        fills, o = [], 0
        for lo, x in groups[b]:
            fills.append((d[o:o + _numel(x)], lo, x))
            o += _numel(x)
        _parallel(lambda f: _fill(*f), fills, [4 * f[0].size for f in fills], workers)
        jobs = _cuts(cnt, max(1, cnt // _JOB)) if workers > 1 else [(0, cnt)]
        for p, h in _parallel(lambda j: _scan(d, b, *j), jobs, [4 * (e - a) for a, e in jobs], workers):
            if p.size:
                exc_pos.append(p + base)
                exc_h.append(h)
        if b and cnt:
            g = (cnt + 31) // 32
            _pack_into(d.view(np.uint32), b, body[at:at + 4 * b * g].view("<u4").reshape(b, g), workers)
            at += 4 * b * g
        base += cnt
    pos = np.concatenate(exc_pos) if exc_pos else np.zeros(0, dtype=np.int64)
    h = np.concatenate(exc_h) if exc_h else np.zeros(0, dtype=np.int64)
    tail = (struct.pack("<I", pos.size) + _rice_encode(np.diff(pos, prepend=-1) - 1, workers)
            + _rice_encode(((h << 1) ^ (h >> 63)) - 1, workers))
    return b"".join([head, bytes(-len(head) % 4), memoryview(body), tail])


# -- a device's encoder

def _encode_device(zts: list, centre: bool) -> bytes:
    """:func:`encode` with torch ops on the claims' device: a fixed number of kernels plus about ten per
    distinct width, and four waits for the device (:func:`_d2h`, :func:`_nonzero_dev`)."""
    dev = zts[0].device
    lay = _layout(tuple((int(z.shape[0]), int(z.shape[1])) for z in zts))
    if lay.total == 0:                                         # nothing to encode on the device
        return _encode_host([z.cpu() for z in zts], centre, 1)
    n = lay.n_ops
    centre = bool(centre and lay.eligible.any())
    tb = lay.on(dev, centre)
    # 1. the claims in one int32 buffer, then the statistics of every op's samples and (if centring)
    # of every op's row means, in one sort and one scatter; one copy back
    flat = [z.reshape(-1) for z in zts]
    if flat[0].dtype not in (torch.int32, torch.int64) or any(f.dtype != flat[0].dtype for f in flat):
        flat = [f.to(torch.int64) for f in flat]
    x = torch.cat(flat)
    ext = torch.stack(list(torch.aminmax(x))).to(torch.int64)
    x = x.to(torch.int32)            # out-of-range claims wrap: the extremes refuse them below
    s = x.index_select(0, tb["sample_idx"]).to(torch.int64)
    vals = [s]
    if centre:
        cs = torch.cat([s.new_zeros(1), torch.cumsum(s, 0)])
        off = torch.div(cs[tb["srow"][1:]] - cs[tb["srow"][:-1]] + tb["row_half"], tb["row_div"], rounding_mode="floor")
        csx = torch.cat([s.new_zeros(1), torch.cumsum(x, 0, dtype=torch.int64)])
        means = torch.div(csx[tb["frow"][1:]] - csx[tb["frow"][:-1]] + tb["frow_half"], tb["frow_div"],
                          rounding_mode="floor")
        vals += [s - off[tb["sample_row"]], off, means]
    keys = (torch.cat(vals) if len(vals) > 1 else s.clone()).clamp_(-(1 << 31), (1 << 31) - 1).add_(tb["segkey"])
    keys = torch.sort(keys).values                  # each segment's values, sorted, in its own key range
    value = lambda k: (k & 0xFFFFFFFF) - (1 << 31)    # noqa: E731
    mid = _round_half(value(keys[tb["mid_lo"]]) + value(keys[tb["mid_hi"]])) * tb["nonempty"]
    # zigzag(v - c) < 2**B  <=>  c - 2**(B-1) <= v < c + 2**(B-1): each bit length's count from the
    # sorted keys, at the bounds of these nested ranges (B = 0: v == c)
    lo_q = (mid[:, None] - tb["half"]).clamp_(-(1 << 31), (1 << 31) - 1).add_(tb["segbase"][:, None])
    hi_q = (mid[:, None] + tb["half_hi"]).clamp_(-(1 << 31), (1 << 31) - 1).add_(tb["segbase"][:, None])
    within = torch.searchsorted(keys, hi_q.view(-1), right=True) - torch.searchsorted(keys, lo_q.view(-1))
    st = _d2h(torch.cat([ext, mid, within]))
    if st[1] >= _Z_LIMIT or st[0] <= -_Z_LIMIT:
        raise Unencodable("a claim outside the range check cannot be encoded")
    n_seg = tb["n_seg"]
    c = st[2:2 + n_seg]
    h = np.zeros((n_seg, _BINS), dtype=np.int64)
    h[:, :_SPAN] = np.diff(st[2 + n_seg:].reshape(n_seg, _SPAN), axis=1, prepend=0)
    # 2. the plan, on the host
    centred, width, lo = _decide(lay, c, h, centre)
    means_b = means_lo = np.zeros(n, dtype=np.int64)
    if centre:
        means_b, means_lo = _decide_means(c[3 * n:], h[3 * n:], lay.rows)
    seg_t = _segments(lay, centred, width, lo, means_b, means_lo)
    head = _write_header(lay, centred, seg_t["b"], seg_t["lo"])
    # 3. every value in the global order (by width, then segment order) minus its base, from
    # xr = [claims, row means, 0] by index: a residual is its claim minus its row's mean
    order = np.argsort(seg_t["b"], kind="stable")
    g = {k: v[order] for k, v in seg_t.items()}
    gstart = _cum0(g["size"])
    total = int(gstart[-1])
    zero = lay.total + int(lay.frow0[-1])
    wide = max(total, zero + 1) >= (1 << 31) - 1     # int32 indices unless the buffers are that large
    src = np.where(g["kind"] == 2, lay.total + lay.frow0[g["op"]], lay.off[:-1][g["op"]])
    sub = np.where(g["kind"] == 1, lay.total + lay.frow0[g["op"]], zero)
    div = np.where(g["kind"] == 1, lay.cols[g["op"]], (1 << 62) if wide else (1 << 31) - 1)   # k // div == 0: the 0
    table = _upload(np.stack([g["size"], src - gstart[:-1], gstart[:-1], sub, div, g["lo"], g["b"],
                              (1 << g["b"]) - 1]), dev)
    it = torch.int64 if wide else torch.int32
    fields = table.to(it)
    s_e = torch.repeat_interleave(torch.arange(order.size, dtype=it, device=dev), table[0], output_size=total)

    def per_value(row: int) -> torch.Tensor:
        return fields[row].index_select(0, s_e)

    e = torch.arange(total, dtype=it, device=dev)
    if centred.any():
        xr = torch.cat([x, means.to(torch.int32), x.new_zeros(1)])
        k_e = e - per_value(2)
        val = xr.index_select(0, e.add_(per_value(1))) - xr.index_select(
            0, per_value(3).add_(torch.div(k_e, per_value(4), rounding_mode="floor")))
        del k_e
    else:
        val = x.index_select(0, e.add_(per_value(1)))
    del e
    val = val.to(torch.int32).sub_(per_value(5).to(torch.int32))
    b_e = per_value(6).to(torch.int32)
    high = torch.bitwise_right_shift(val, b_e)          # non-zero: outside [0, 2**B), an exception
    pos = _nonzero_dev(high)
    slots = val.bitwise_and_(per_value(7).to(torch.int32))
    hp = high[pos].to(torch.int64)
    del high, b_e, s_e
    # 4. each width's slots packed (about ten kernels a width), the exceptions' Rice parameters
    parts = []
    for b, a, z in _width_runs(g["b"], gstart):
        if b and z > a:
            parts.append(_pack_dev(slots[a:z], b).view(-1))
    n_exc = pos.numel()
    ks, lows, q8 = [], [], []
    if n_exc:
        rv = torch.stack([pos - torch.cat([pos.new_full((1,), -1), pos[:-1]]) - 1, ((hp << 1) ^ (hp >> 63)) - 1])
        vmin, vmax = rv.amin(1), rv.amax(1)
        kmin = ((vmax[:, None] >= tb["pow2"][None, :]).sum(1) - 4).clamp_(min=0)
        q = ((rv >> kmin[:, None]).clamp_(0, 15) + torch.arange(0, 32, 16, device=dev)[:, None]).view(-1)
        qh = torch.zeros(32, dtype=torch.int64, device=dev).index_add_(0, q, torch.ones_like(q))
        rs = _d2h(torch.cat([vmin, vmax, kmin, qh]))
        if rs[:2].min() < 0 or rs[2:4].max() >= 1 << (_KMAX + 4):
            raise ValueError("Rice values must be in [0, 2**36)")
        ks = [_rice_k_hist(n_exc, int(rs[4 + i]), rs[6 + 16 * i:22 + 16 * i]) for i in range(2)]
        lows = [_pack_dev(rv[i] & ((1 << k) - 1), k).view(-1) for i, k in enumerate(ks) if k]
        q8 = [torch.stack([rv[i] >> k for i, k in enumerate(ks)]).to(torch.uint8).view(-1)]
    # 5. one copy back: the words, the Rice low parts and quotients
    pieces = [p.view(torch.uint8) for p in parts + lows] + q8
    host = _d2h(torch.cat(pieces)) if pieces else np.zeros(0, dtype=np.uint8)
    mv = memoryview(host)
    at = 4 * sum(p.numel() for p in parts)
    out = [head, mv[:at], struct.pack("<I", n_exc)]
    if n_exc:
        quot = host[host.size - 2 * n_exc:].reshape(2, n_exc)
        for i, k in enumerate(ks):
            size = 4 * k * ((n_exc + 31) // 32)
            out += [bytes([k]), mv[at:at + size], _pack_levels(quot[i])]
            at += size
    return b"".join(out)


def _width_runs(widths: np.ndarray, start: np.ndarray):
    """``(B, first, end)`` of each width's run of the global order (``widths`` sorted)."""
    if widths.size == 0:
        return []
    cut = np.flatnonzero(np.r_[True, widths[1:] != widths[:-1]])
    ends = np.r_[cut[1:], widths.size]
    return [(int(widths[a]), int(start[a]), int(start[z])) for a, z in zip(cut, ends)]

# ---------------------------------------------------------------------------- the decoder
def _need(buf, off: int, n: int) -> int:
    if off + n > len(buf):
        raise ClaimCodecError("truncated header")
    return off + n


def _patch(x: np.ndarray, gaps: np.ndarray, hz: np.ndarray, segs: list[tuple[int, int, int]], total: int,
           workers: int) -> None:
    """Adds the exceptions to the values ``x``: each holds ``slot + lo``, but the residuals of a centred op
    ``slot`` (their base is added with the row means), so an exception becomes ``slot + lo + h 2**B`` less
    the base still pending, range-checked with it.  ``segs`` are ``(start, B, pending base)`` in the
    global order.
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
    np.add(h, x[p], out=h)                       # slot + lo + h 2**B, but for a base still pending
    first, last = np.clip(cuts[:-1], a, e), np.clip(cuts[1:], a, e)
    used = np.flatnonzero(first < last)          # the segments with exceptions here
    if used.size:
        pend = np.array([segs[i][2] for i in used], dtype=np.int64)
        if ((np.minimum.reduceat(h, first[used] - a) + pend <= -_X_LIMIT).any()
                or (np.maximum.reduceat(h, first[used] - a) + pend >= _X_LIMIT).any()):
            raise ClaimCodecError("exception out of range")
    x[p] = h.astype(np.int32)                    # range-checked; an int32 scatter is twice as fast


def _add_means(x: np.ndarray, ops: list) -> None:
    """Per centred op ``(start, N, M, row means' start, residuals' base)``: its row means and base added to
    its residuals in place, in one pass (the row means already hold their base)."""
    for s, n_rows, m, so, lo in ops:
        z = x[s:s + n_rows * m].reshape(n_rows, m)
        np.add(z, (x[so:so + n_rows] + np.int64(lo)).astype(np.int32)[:, None], out=z)


def _read_header(buf, rows: list[int], cols: list[int]) -> tuple[list, list, int]:
    """``(heads, segs, offset after the header)``: per op ``(N, M, its row means' segment or None, its main
    segment)``, per segment ``(B, lo, n)``; every field checked against the query's shapes."""
    if not isinstance(buf, bytes):
        raise ClaimCodecError("not bytes")
    if len(buf) < 8 or buf[:4] != MAGIC:
        raise ClaimCodecError("bad magic")
    if struct.unpack_from("<I", buf, 4)[0] != len(rows):
        raise ClaimCodecError("wrong number of ops")
    off = 8
    heads, segs = [], []
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
    return heads, segs, off


_POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.int64)


def _rice_end(buf, off: int, n: int) -> int:
    """Where the Rice vector of ``n`` values at ``buf[off:]`` ends: after its parameter, its low parts and
    its unary levels, each level as long as the bits set in the one before (counted, not decoded; the
    padding is checked where the levels are decoded)."""
    if n == 0:
        return off
    if off >= len(buf):
        raise ClaimCodecError("truncated Rice vector")
    k = buf[off]
    if k > _KMAX:
        raise ClaimCodecError("Rice parameter out of range")
    off += 1 + 4 * k * ((n + 31) // 32)
    count = n
    for _ in range(_QCAP):
        if count == 0:
            break
        nbytes = (count + 7) // 8
        if off + nbytes > len(buf):
            raise ClaimCodecError("truncated unary levels")
        raw = np.frombuffer(buf, dtype=np.uint8, count=nbytes, offset=off)
        last = int(raw[-1]) & (((1 << (count % 8)) - 1) if count % 8 else 0xFF)
        count = int(_POP8[raw[:-1]].sum()) + int(_POP8[last])
        off += nbytes
    return off


def _exceptions(buf, off: int, total: int, workers: int) -> tuple[int, int, int]:
    """``(e, start of the gaps, start of the payloads or -1)`` of the exception list at ``buf[off:]``, the end
    of the input; with several ``workers`` and at least ``_SPLIT`` exceptions the payloads' start too (so
    both vectors decode at once), and the input's end checked."""
    off = _need(buf, off, 4)
    (n_exc,) = struct.unpack_from("<I", buf, off - 4)
    if n_exc > total or 2 * ((n_exc + 7) // 8) > len(buf) - off:    # each Rice vector: n_exc bits at least
        raise ClaimCodecError("more exceptions than values or bytes")
    if workers == 1 or n_exc < _SPLIT:
        return n_exc, off, -1
    mid = _rice_end(buf, off, n_exc)
    if _rice_end(buf, mid, n_exc) != len(buf):
        raise ClaimCodecError("trailing bytes")
    return n_exc, off, mid


def _rice_job(buf, off: int, n: int, out: dict, key: str) -> None:
    out[key] = _rice_decode(buf, off, n)[0]


def _rice_pair(buf, off: int, n: int, out: dict) -> None:
    """Both vectors one after the other, the second where the first ends, which must be the input's end."""
    out["gaps"], off = _rice_decode(buf, off, n)
    out["hz"], off = _rice_decode(buf, off, n)
    if off != len(buf):
        raise ClaimCodecError("trailing bytes")


def _decode(buf, rows: list[int], cols: list[int], workers: int, *, alloc=None) -> tuple[np.ndarray, list]:
    """The value buffer (int32, every claim in place; allocated by ``alloc(n)`` if given) and per op
    ``(start, N, M)`` in it.

    The slot streams (lanes of large ones split over the ``workers``) and the two Rice vectors of the
    exceptions (where each starts follows from the header and their levels' bit counts) are decoded
    by parallel jobs, which add each segment's base to its slots as they write them; then the exceptions
    are patched in and the centred ops' row means added, in slices and op ranges on the workers."""
    heads, segs, off = _read_header(buf, rows, cols)
    total = sum(n for _, _, n in segs)
    # 1. the slots: one stream per width, each a contiguous block of x, and the exceptions' vectors
    x = np.empty(total, dtype=np.int32) if alloc is None else alloc(total)
    widths = sorted({s[0] for s in segs})
    order = [i for b in widths for i, s in enumerate(segs) if s[0] == b]   # the global segment order
    start = [0] * len(segs)
    jobs, sizes, base = [], [], 0
    residual = {main for _, _, means, main in heads if means is not None}
    pending = [segs[i][1] if i in residual else 0 for i in range(len(segs))]   # added with the row means
    for b in widths:
        members = [i for i in order if segs[i][0] == b]
        cnt = sum(segs[i][2] for i in members)
        ends = list(itertools.accumulate(segs[i][2] for i in members))
        bases = ([e - segs[i][2] for e, i in zip(ends, members)], ends,
                 [np.uint32(segs[i][1] & 0xFFFFFFFF) if segs[i][1] and i not in residual else 0 for i in members])
        more, off = _stream_jobs(buf, off, cnt, b, x.view(np.uint32)[base:base + cnt], workers, bases)
        jobs += more
        # small ones here; a width 0 has no job (its values are all 0 slots)
        sizes += [4 * cnt // len(more) if more and cnt >= _THREADED else 0] * len(more)
        for i in members:
            start[i] = base
            base += segs[i][2]
    n_exc, g_off, p_off = _exceptions(buf, off, total, workers)
    exc: dict = {}
    if p_off >= 0:
        jobs = [functools.partial(_rice_job, buf, g_off, n_exc, exc, "gaps"),
                functools.partial(_rice_job, buf, p_off, n_exc, exc, "hz")] + jobs
        sizes = [8 * n_exc] * 2 + sizes
    else:                               # on this thread while the workers unpack the streams
        jobs = [functools.partial(_rice_pair, buf, g_off, n_exc, exc)] + jobs
        sizes = [0] + sizes
    _run(jobs, sizes, workers)
    if n_exc:
        # 2. the exceptions: each value is slot + lo + h 2**B, range-checked
        _patch(x, exc["gaps"], exc["hz"], [(start[i], segs[i][0], pending[i]) for i in order], total, workers)
    # 3. the centred ops' row means and base added to their residuals in place, ops in about equal parts on
    # the workers (the other bases were added as the streams were unpacked)
    centred = [(start[main], n, m, start[means], segs[main][1]) for n, m, means, main in heads if means is not None]
    if centred:
        cum = np.cumsum([0] + [n * m for _, n, m, _, _ in centred])
        parts = workers if int(cum[-1]) >= _THREADED else 1
        cut = [int(c) for c in np.searchsorted(cum, [cum[-1] * i // parts for i in range(parts)])] + [len(centred)]
        sz = [4 * int(cum[e] - cum[a]) if parts > 1 else 0 for a, e in zip(cut, cut[1:])]
        _run([functools.partial(_add_means, x, centred[a:e]) for a, e in zip(cut, cut[1:])], sz, workers)
    return x, [(start[main], n, m) for n, m, _, main in heads]


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
    one pass on torch's threads), or int32 (the decoded buffer itself; with ``pin`` decoded straight
    into pinned memory, for non-blocking uploads to a GPU)."""
    held = {}

    def pinned(n: int) -> np.ndarray:
        held["t"] = torch.empty(n, dtype=torch.int32, pin_memory=True)
        return held["t"].numpy()

    x, ops = _decode(buf, rows, cols, workers, alloc=pinned if pin and dtype == torch.int32 else None)
    flat = held.get("t", torch.from_numpy(x))
    if dtype != torch.int32:
        flat = flat.to(dtype)
    return [flat[s:s + n * m].view(n, m) for s, n, m in ops]


# ---------------------------------------------------------------------------- field elements
FIELD_BITS = 31
"""BabyBear elements are ``< 2**31``: 31-bit packing saves 1/32 of ``u`` and the opened columns."""


def field_size(numel: int) -> int:
    """Bytes of :func:`pack_field` for ``numel`` elements."""
    return 4 * FIELD_BITS * ((numel + 31) // 32)


def pack_field(tensors: list[torch.Tensor], *, workers: int | None = None) -> bytes:
    """Field elements (integer tensors with values in ``[0, 2**31)``), flattened and concatenated,
    at 31 bits each (:func:`pack32`).  Tensors on one device are packed there (:func:`_pack_dev`) and
    come back, with their range, in one copy; host tensors are clipped into one buffer (an element
    outside the field stays outside) and packed on ``workers`` threads (default: torch's)."""
    ts = [t.detach() for t in tensors]
    n = sum(t.numel() for t in ts)
    if n == 0:
        return b""
    devices = {t.device for t in ts}
    if len(devices) == 1 and next(iter(devices)).type != "cpu":
        flat = torch.cat([t.reshape(-1).to(torch.int64) for t in ts])
        ext = torch.stack(list(torch.aminmax(flat)))
        words = _pack_dev(flat & ((1 << FIELD_BITS) - 1), FIELD_BITS)
        host = _d2h(torch.cat([ext.view(torch.int32), words.view(-1)]))
        lo, hi = host[:4].view(np.int64).tolist()
        if lo < 0 or hi >= 1 << FIELD_BITS:
            raise Unencodable("field elements must be in [0, 2**31)")
        return host[4:].astype("<i4", copy=False).tobytes()
    flat, o = np.empty(n, dtype=np.uint32), 0
    for t in ts:
        np.clip(_as_numpy(t.cpu()), -1, 1 << FIELD_BITS, out=flat[o:o + t.numel()].reshape(t.shape), casting="unsafe")
        o += t.numel()
    if int(flat.max()) >= 1 << FIELD_BITS:                    # -1 is 2**32 - 1 here
        raise Unencodable("field elements must be in [0, 2**31)")
    return pack32(flat, FIELD_BITS, workers or torch.get_num_threads())


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
    column), claim after claim, each as its ``M`` rows of ``d`` bytes (the bytes and their range come
    back from a device in one copy)."""
    if not claims:
        return b""
    flat = torch.cat([z.detach().T.reshape(-1) for z in claims])
    if flat.numel() == 0:
        return b""
    ext = torch.stack(list(torch.aminmax(flat))).to(torch.int64)
    host = _d2h(torch.cat([ext.view(torch.int8), flat.to(torch.int8)]))
    lo, hi = host[:16].view(np.int64).tolist()
    if lo < -128 or hi > 127:
        raise Unencodable("looked-up rows must be int8")
    return host[16:].tobytes()


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

"""The products of V1's verifier with the cut's values, computed by value.

The final check (F3) (``pvi.v1.protocol._final_ok``, rejection label ``cut_final``) needs, for every cut operation
of an instance, the products ``L e`` and ``W e`` ``[N, 8]`` of the operation's window matrices with the instance's
column point ``e = eq(r_col)[0:T]`` ``[T, 8]`` in ``F_{p^8} = F_p[xi]/(xi^8 - 11)`` (:mod:`pvi.core.extfield`):
``L = lo(s)`` and ``W = W(s)`` of the sent values ``s`` ``[N, T]`` (:mod:`pvi.v1.cut`), except at the listed
exceptions, whose window is their exact claim ``z`` (``L = z``, ``W = 1``).  ``pvi.v1.protocol._xbar`` needs
``X^T e`` ``[K, 8]`` for every cut operation's input ``X`` ``[T, K]`` (small integers: the input range check of (D1)
bounds ``|X| <= max_input + 1``).  The baseline verifier (option ``v1_verifier`` of :mod:`pvi.options`) gathers
``[L ; W]`` in float64 and multiplies it with ``e`` in three 11-bit limbs (:func:`pvi.v1.cut.public_lw_f64` and
:func:`pvi.core.extfield.int_times`: 24 columns, partial sums below ``2^52``).

Both matrices are functions of one small integer per entry: ``L`` and ``W`` of the value ``s`` (an int8 for a
requantisation, ``2^8`` values; a residual operation's ``Delta`` spans its own range ``[a, b]``), and ``X`` is that
integer itself.  ``W(v)`` takes the two values ``w0 = floor(2^sh / m)`` and ``w0 + 1``
(``W(v) = ceil(x + q) - ceil(x)`` with ``q = 2^sh / m``), so with ``D(v) = W(v) - w0`` in ``{0, 1}`` and
``C(v) = sum_{a <= u < v} D(u)`` in ``[0, v - a]``,

    ``lo(v) = lo(a) + w0 (v - a) + C(v)``,   ``W(v) = w0 + D(v)``,

and ``L e = lo(a) (1 . e) + w0 (S - a) e + C(S) e``, ``W e = w0 (1 . e) + D(S) e``.  The three matrices ``S - a``,
``C(S)``, ``D(S)`` have entries in ``[0, V)`` (``V = b - a + 1``), gathered from one ``[V, 3]`` table
(``torch.nn.functional.embedding``) and multiplied with ``e`` as **one** 31-bit limb per coefficient: 8 columns
instead of 24, exact over ``2^(22 - bits(V - 1))`` columns at a time.  ``X^T e`` is the same product with the
identity table, on ``X`` itself.  On a CPU with the native library the three sums come from
``pvi.kernels.native.lw_affine`` (one pass over the values with int64 accumulators over the 8 coefficients; option
``v1_native``).  Every path gives the canonical residues of the same integer vectors as the baseline, so the field
results are bit-identical (``tests/test_v1_products.py``).

**Exceptions** sit in their value's row like any other entry; ``(z - lo(s)) e_j`` and ``(1 - W(s)) e_j`` are then
added to their row (int64, each term reduced before it is summed).

**Exactness** (every bound is checked by the code or follows from its inputs): ``e`` is canonical, below
``p < 2^31``; a float64 product of entries below ``2^k`` with coefficients below ``2^31`` over ``n <= 2^(22 - k)``
columns has every partial sum an integer below ``2^53``, so it is exact in any summation order (as in
:func:`pvi.core.extfield.int_times`); every other step is int64 arithmetic on reduced values (products below
``2^62``).
"""

from __future__ import annotations

import functools

import torch

from pvi.core.extfield import D
from pvi.core.field import P, min_max

__all__ = ["optimised", "lw_times", "xbar", "table_times", "SPAN"]

SPAN = 1 << 15
"""Value spans up to this many values use these products (a residual operation's ``Delta`` may span more: it then
takes the baseline products)."""
_BLOCK_BYTES = 1 << 24
"""Bytes of one row block's gathered table (``[R, T, 3]`` float64)."""


def optimised() -> bool:
    """Whether V1's verifier runs its optimised paths (option ``v1_verifier``)."""
    from pvi import options
    return options.current().v1_verifier == "optimised"


def _affine(op, a: int, b: int):
    """``(lo(a), w0, tab [V, 3] float64, lo [V], W [V])``: the table products' decomposition of the windows of the
    values ``a .. b`` (module docstring): columns ``v - a``, ``C(v)``, ``D(v)``, each in ``[0, V)``.  The last 128
    are cached per ``(m, sh, a, b)`` (each at most about 1.3 MB, at ``V = SPAN``: 24 bytes per value for ``tab`` and
    16 for ``lo`` and ``W``; the tensors are read, never written)."""
    return _affine_cached(int(op.mult), int(op.sh), int(a), int(b))


@functools.lru_cache(maxsize=128)
def _affine_cached(mult: int, sh: int, a: int, b: int):
    from pvi.v1.cut import lo
    t = lo(torch.arange(a, b + 2, dtype=torch.int64), mult, sh)
    lo_v, w_v = t[:-1], t[1:] - t[:-1]
    w0 = (1 << sh) // mult
    d = w_v - w0
    v = torch.arange(b - a + 1, dtype=torch.int64)
    c = lo_v - lo_v[0] - w0 * v
    if bool((d < 0).any() or (d > 1).any() or (c < 0).any() or (c > v).any()):
        raise AssertionError("the windows are not affine plus a 0/1 step")       # (impossible: module docstring)
    return int(lo_v[0]), w0, torch.stack([v, c, d], 1).to(torch.float64), lo_v, w_v


def _chunk(max_entry: int) -> int:
    """Columns per exact float64 product of entries ``|x| <= max_entry`` with 31-bit coefficients: every partial sum
    below ``2^53``."""
    k = max(1, int(max_entry).bit_length())
    return 1 << max(0, 22 - k)


def _sum_e(e: torch.Tensor) -> torch.Tensor:
    """``1 . e = sum_j e_j`` ``[8]`` (canonical)."""
    return e.sum(0) % P


# ------------------------------------------------------------------ the products
def table_times(idx: torch.Tensor, tab: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
    """``sum_j tab[idx_ij, :] e_j`` ``[R, k, 8]`` (canonical) for indices ``idx`` ``[R, T]`` (int32 or int64, in
    ``[0, V)``) into a float64 table ``tab`` ``[V, k]`` of integers in ``[0, V)``: one gather, then batched float64
    products with ``e`` as one 31-bit limb over :func:`_chunk` columns at a time."""
    rows, t = idx.shape
    k = tab.shape[1]
    out = torch.zeros(rows, k, D, dtype=torch.int64, device=idx.device)
    if not rows or not t:
        return out
    ef64 = e.to(idx.device, torch.float64)
    step = _chunk(tab.shape[0] - 1)
    tab = tab.to(idx.device)
    for c0 in range(0, t, step):
        m = torch.nn.functional.embedding(idx[:, c0:c0 + step], tab)              # [R, n, k]
        g = torch.matmul(m.transpose(1, 2), ef64[c0:c0 + step])                     # [R, k, 8]: exact integers
        out = (out + g.to(torch.int64)) % P
    return out


def _native_lw(s: torch.Tensor, a: int, tab: torch.Tensor, e: torch.Tensor) -> torch.Tensor | None:
    """The table products' three sums ``[N, 3, 8]`` from the native kernel (``pvi.kernels.native.lw_affine``: one
    pass over the values, int64 accumulators over the 8 coefficients, the same integers), or ``None`` (no library, a
    GPU tensor, or option ``v1_native`` off): the torch path then runs."""
    try:
        from pvi.kernels import native as native_kernels
    except ImportError:                                                             # (pragma: no cover)
        return None
    return native_kernels.lw_affine(s, a, tab[:, 1:3].to(torch.int64), e)


def _rows_per_block(t: int, width: int) -> int:
    return max(1, _BLOCK_BYTES // max(1, 8 * t * width))


# ------------------------------------------------------------------ the final check's L e and W e
def lw_times(s: torch.Tensor, exc_idx: torch.Tensor, exc_z: torch.Tensor,
             op, e: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(L e, W e)`` ``[N, 8]`` each (canonical, on the CPU) of a cut operation's range-checked values ``s``
    ``[N, T]`` (any integer dtype, on any device) with its exceptions (flat indices ``exc_idx``, ascending, and
    claims ``exc_z``, checked by (D4) in ``pvi.v1.protocol.exceptions_ok``) at the column point ``e`` ``[T, 8]``: the
    integers of ``pvi.core.extfield.int_times(pvi.v1.cut.public_lw_f64(...), e, 30)``.  ``None`` if the values span
    more than :data:`SPAN` values (the caller then runs the baseline products).  A requantisation's values lie in its
    clamp ``[lo, hi]`` (checked by (D1)); a residual operation's span is found here."""
    n, t = s.shape
    dev = s.device
    if s.numel() == 0:
        z = torch.zeros(n, D, dtype=torch.int64)
        return z, z.clone()
    a, b = (op.lo_, op.hi) if op.kind == "requant" else min_max(s)
    if b - a + 1 > SPAN:
        return None
    lo_a, w0, tab, lo_v, w_v = _affine(op, a, b)
    e_dev = e.to(dev)
    je = _sum_e(e_dev)
    le = torch.empty(n, D, dtype=torch.int64, device=dev)
    we = torch.empty(n, D, dtype=torch.int64, device=dev)
    lo_a, w0 = lo_a % P, w0 % P
    g = _native_lw(s, a, tab, e) if dev.type == "cpu" else None                     # [N, 3, 8], or None
    if g is not None:
        le = ((lo_a * je) % P + (w0 * g[:, 0]) % P + g[:, 1]) % P
        we = ((w0 * je) % P + g[:, 2]) % P
    else:
        step = _rows_per_block(t, 3)
        for r0 in range(0, n, step):
            idx = s[r0:r0 + step].to(torch.int32)
            if a:
                idx = idx - a                   # out of place: in place, int32 values would shift the caller's tensor
            g = table_times(idx, tab, e_dev)                                            # [R, 3, 8]
            le[r0:r0 + step] = ((lo_a * je) % P + (w0 * g[:, 0]) % P + g[:, 1]) % P
            we[r0:r0 + step] = ((w0 * je) % P + g[:, 2]) % P
    le, we = le.cpu(), we.cpu()
    if exc_idx.numel():
        idx = exc_idx.to(torch.int64).cpu()
        rows, cols = idx // t, idx % t
        sv = s.reshape(-1)[exc_idx.to(dev)].cpu().to(torch.int64) - a
        ec = e.cpu()[cols]                                                              # [n_exc, 8]
        dl = torch.remainder(exc_z.cpu().to(torch.int64) - lo_v[sv], P)
        dw = torch.remainder(1 - w_v[sv], P)
        le.index_add_(0, rows, (dl[:, None] * ec) % P)
        we.index_add_(0, rows, (dw[:, None] * ec) % P)
        le, we = le % P, we % P
    return le, we


# ------------------------------------------------------------------ X^T e
def xbar(x: torch.Tensor, e: torch.Tensor) -> torch.Tensor | None:
    """``X^T e`` ``[K, 8]`` (canonical, on the CPU) of an integer input ``x`` ``[T, K]`` (any dtype, any device) at the
    column point ``e`` ``[T, 8]``: the integers of ``pvi.core.extfield.int_times(x.T, e, 31)``, by one float64 product
    with ``e`` as one 31-bit limb (or the native kernel).  ``None`` if ``x`` spans more than :data:`SPAN` values or
    has an entry of ``2^16`` or more in absolute value (the caller then runs the baseline product)."""
    t, k = x.shape
    if x.numel() == 0:
        return torch.zeros(k, D, dtype=torch.int64)
    a, b = min_max(x)
    if b - a + 1 > SPAN or max(abs(a), abs(b)) >= 1 << 16:
        return None
    dev = x.device
    if dev.type == "cpu":
        from pvi.kernels import native as native_kernels
        got = native_kernels.xbar(x, e)                                             # [K, 8], or None
        if got is not None:
            return got
    ef64 = e.to(dev, torch.float64)
    step = _chunk(max(abs(a), abs(b)))
    acc = torch.zeros(D, k, dtype=torch.int64, device=dev)
    for c0 in range(0, t, step):
        part = ef64[c0:c0 + step].T @ x[c0:c0 + step].to(torch.float64)               # [8, K]: exact integers
        acc = (acc + torch.remainder(part.to(torch.int64), P)) % P
    return acc.T.contiguous().cpu()

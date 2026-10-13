"""V-V1N parts 1-2: the V1 verifier's products with the cut, by value (``SP2027_PLAN2.md`` item V-V1N).

The final check (``cut_protocol._final_ok``, check F3 ``cut_final``) needs, for every cut op of an instance, the
products ``L e`` and ``W e`` ``[N, 8]`` of the op's window matrices with the instance's column point
``e = eq(r_col)[0:T]`` ``[T, 8]`` in ``F = F_{p^8}`` (:mod:`extfield`): ``L = lo(s)`` and ``W = W(s)`` of the sent
values ``s`` ``[N, T]`` (:mod:`cut`), except at the listed exceptions, whose window is their exact claim ``z``
(``L = z``, ``W = 1``).  ``cut_protocol._xbar`` needs ``X^T e`` ``[K, 8]`` for every cut op's input ``X`` ``[T, K]``
(int8-valued: derive checks ``|X| <= max_input + 1``).  Today's code (:func:`cut.public_lw_f64` and
:func:`extfield.int_times`) gathers ``[L ; W]`` in float64 and multiplies it with ``e`` in three 11-bit limbs (24
columns, partial sums below ``2^52``).

Both matrices are functions of one small integer per entry: ``L`` and ``W`` of the value ``s`` (an int8 for a
requant op, ``2^8`` values; a residual op's ``Delta`` spans its own range ``[a, b]``), ``X`` is that integer itself.
Two exact ways to use it are implemented, both giving the canonical residues of the same integer vectors as
today's code (so the field results are bit-identical, ``tests/test_v1_buckets.py``):

* **Buckets** (the spec's kernel, :func:`bucket_sums`): per row ``i`` and value ``v``,
  ``B_i[v] = sum_{j : S_ij = v} e_j`` (one 8-lane integer add per entry), then
  ``sum_j f(S_ij) e_j = sum_v f(v) B_i[v]`` for ``f`` = ``lo``, ``W`` or the identity.  This is the native and
  GPU kernel's algorithm (one add per entry, the buckets of a row in L1); in torch, eight ``scatter_add_`` passes
  cost about as much as today's GEMM (measured, ``docs/improvements/V_V1N_fast_v1_verifier.md``).
* **Table products** (the torch default): ``W(v)`` takes the two values ``w0 = floor(2^sh / m)`` and ``w0 + 1``
  (``W(v) = ceil(x + q) - ceil(x)`` with ``q = 2^sh / m``), so with ``D(v) = W(v) - w0`` in ``{0, 1}`` and
  ``C(v) = sum_{a <= u < v} D(u)`` in ``[0, v - a]``,

      ``lo(v) = lo(a) + w0 (v - a) + C(v)``,   ``W(v) = w0 + D(v)``,

  and ``L e = lo(a) (1 . e) + w0 (S - a) e + C(S) e``, ``W e = w0 (1 . e) + D(S) e``.  The three matrices
  ``S - a``, ``C(S)``, ``D(S)`` have entries in ``[0, V)`` (``V = b - a + 1``), gathered from one ``[V, 3]`` table
  (``torch.nn.functional.embedding``) and multiplied with ``e`` as **one** 31-bit limb per coefficient: 8 columns
  instead of 24, exact over ``2^(22 - bits(V - 1))`` columns at a time.  ``X^T e`` is the same product with the
  identity table, on ``X`` itself.

**Exceptions** sit in their value's row like any other entry; ``(z - lo(s)) e_j`` and ``(1 - W(s)) e_j`` are then
added to their row (int64, each term reduced before it is summed).

**Exactness** (every bound is checked by the code or follows from its inputs): ``e`` is canonical, below
``p < 2^31``; a bucket sums at most ``T < 2^32`` coefficients (int64); a float64 product of entries below ``2^k``
with coefficients below ``2^31`` over ``n <= 2^(22 - k)`` columns has every partial sum an integer below ``2^53``,
so it is exact in any summation order (as :func:`extfield.int_times`); every other step is int64 arithmetic on
reduced values (products below ``2^62``).

The switch ``PVI_V1_BUCKETS``, read at each call: ``1`` (default) the table products; ``scatter`` the buckets;
``0`` today's code everywhere in V1's verifier (parts 1-4 of V-V1N: these products, the batched GKR check and the
batch-inverted logUp identity, the int8 values; ``cut_protocol``).  Under ``1``, on a CPU with the native library, the
table products' three sums come from ``native_kernels.lw_affine`` (one pass over the values with int64 accumulators
over the 8 coefficients, the same integers; ``PVI_V1_NATIVE=0`` keeps the torch products).
"""

from __future__ import annotations

import functools
import os

import torch

from pvi.core.extfield import D
from pvi.core.field import P, min_max

__all__ = ["mode", "enabled", "lw_times", "xbar", "bucket_sums", "bucket_times", "table_times", "window_tables",
           "SPAN"]

ENV = "PVI_V1_BUCKETS"
SPAN = 1 << 15
"""Value spans up to this many values use the new products (a residual op's ``Delta`` may span more: today's code)."""
_BLOCK_BYTES = 1 << 24
"""Bytes of one row block's gathered table (``[R, T, 3]`` float64) or buckets (``[8, R, V]`` int64)."""


def mode() -> str:
    """``PVI_V1_BUCKETS``: ``"1"`` (default; the table products), ``"scatter"`` (the buckets) or ``"0"`` (today's
    code); any other value is ``"1"``."""
    v = os.environ.get(ENV, "1").strip().lower()
    return v if v in ("0", "scatter") else "1"


def enabled() -> bool:
    """Whether V1's verifier runs V-V1N's paths (``PVI_V1_BUCKETS`` not ``0``)."""
    return mode() != "0"


# ------------------------------------------------------------------ the window tables
def window_tables(op, a: int, b: int) -> tuple[torch.Tensor, torch.Tensor]:
    """``(lo(v), W(v))`` for ``v = a .. b`` (int64 ``[V]`` each) of the cut op ``op`` (:func:`cut.lo`)."""
    from pvi.v1.cut import lo
    t = lo(torch.arange(a, b + 2, dtype=torch.int64), op.mult, op.sh)
    return t[:-1], t[1:] - t[:-1]


def _affine(op, a: int, b: int):
    """``(lo(a), w0, tab [V, 3] float64, lo [V], W [V])``: the table products' decomposition of the windows of the
    values ``a .. b`` (module docstring): columns ``v - a``, ``C(v)``, ``D(v)``, each in ``[0, V)``; the last 128
    are cached per ``(m, sh, a, b)`` (at most about 1.3 MB each; the tensors are read, never written)."""
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
        raise AssertionError("the windows are not affine plus a 0/1 step")       # (impossible: see the docstring)
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


def bucket_sums(idx: torch.Tensor, n_values: int, e: torch.Tensor) -> torch.Tensor:
    """The buckets ``B[c, i, v] = sum_{j : idx_ij = v} e_jc`` ``[8, R, V]`` (int64, unreduced: below ``T p``) of
    indices ``idx`` ``[R, T]`` (int64, in ``[0, V)``): one ``scatter_add_`` per coefficient plane."""
    rows, t = idx.shape
    if t >= 1 << 32:
        raise ValueError("a bucket sums at most 2^32 coefficients")
    planes = e.to(idx.device, torch.int64).T.contiguous()                          # [8, T]
    b = torch.zeros(D, rows, n_values, dtype=torch.int64, device=idx.device)
    for c in range(D):
        b[c].scatter_add_(1, idx, planes[c].expand(rows, t))
    return b


def bucket_times(b: torch.Tensor, tabs: torch.Tensor) -> torch.Tensor:
    """``sum_v tabs[:, v] B[:, i, v]`` ``[R, k, 8]`` (canonical) of buckets ``b`` ``[8, R, V]`` and integer tables
    ``tabs`` ``[k, V]`` (any sign): the buckets reduced (below ``p``), the tables' residues in three 11-bit limbs, one
    float64 product per ``2^11`` values (partial sums below ``2^53``)."""
    d, rows, nv = b.shape
    k = tabs.shape[0]
    t = torch.remainder(tabs.to(b.device, torch.int64), P)
    limbs = torch.cat([(t >> (11 * l)) & 2047 for l in range(3)]).T.to(torch.float64)     # [V, 3k]
    br = torch.remainder(b, P).to(torch.float64).view(d * rows, nv)
    acc = torch.zeros(d * rows, 3 * k, dtype=torch.int64, device=b.device)
    for v0 in range(0, nv, 1 << 11):
        acc = (acc + (br[:, v0:v0 + (1 << 11)] @ limbs[v0:v0 + (1 << 11)]).to(torch.int64)) % P
    out = (acc[:, :k] + acc[:, k:2 * k] * (1 << 11) % P + acc[:, 2 * k:] * (1 << 22) % P) % P      # [8R, k]
    return out.view(d, rows, k).permute(1, 2, 0).contiguous()


def _native_lw(s: torch.Tensor, a: int, tab: torch.Tensor, e: torch.Tensor) -> torch.Tensor | None:
    """The table products' three sums ``[N, 3, 8]`` from the native kernel (``native_kernels.lw_affine``: one pass over
    the values, int64 accumulators over the 8 coefficients, the same integers), or None (no library, a GPU tensor,
    ``PVI_V1_NATIVE=0``): the torch path then runs."""
    try:
        from pvi.kernels import native as native_kernels
    except ImportError:                                                             # (pragma: no cover)
        return None
    return native_kernels.lw_affine(s, a, tab[:, 1:3].to(torch.int64), e)


def _method(n_values: int, t: int) -> str:
    """The products' method under :func:`mode` for ``n_values`` values on rows of ``t`` entries: the buckets only
    where they are not larger than the rows."""
    m = mode()
    return "scatter" if m == "scatter" and n_values <= max(256, t) else "1"


def _rows_per_block(t: int, width: int) -> int:
    return max(1, _BLOCK_BYTES // max(1, 8 * t * width))


# ------------------------------------------------------------------ the final check's L e and W e
def lw_times(s: torch.Tensor, exc_idx: torch.Tensor, exc_z: torch.Tensor, op, e: torch.Tensor,
             method: str | None = None) -> tuple[torch.Tensor, torch.Tensor] | None:
    """``(L e, W e)`` ``[N, 8]`` each (canonical, on the CPU) of a cut op's range-checked values ``s`` ``[N, T]`` (any
    integer dtype, on any device) with its exceptions (flat indices ``exc_idx``, ascending, and claims ``exc_z``,
    checked by ``cut_protocol.exceptions_ok``) at the column point ``e`` ``[T, 8]``: the integers of
    ``extfield.int_times(cut.public_lw_f64(...), e, 30)``.  ``method``: ``"1"`` (table products) or ``"scatter"``
    (buckets); default :func:`mode`, where ``scatter`` takes the buckets only for spans of at most ``max(256, T)``
    values (a row's buckets cost ``8 V`` on top of its ``8 T`` adds).  ``None`` if the values span more than
    :data:`SPAN` values (the caller then runs today's code).  A requant op's values are in its clamp ``[lo, hi]``
    (derive checked them); a residual op's span is found here."""
    n, t = s.shape
    dev = s.device
    if s.numel() == 0:
        z = torch.zeros(n, D, dtype=torch.int64)
        return z, z.clone()
    a, b = (op.lo_, op.hi) if op.kind == "requant" else min_max(s)
    if b - a + 1 > SPAN:
        return None
    method = method or _method(b - a + 1, t)
    lo_a, w0, tab, lo_v, w_v = _affine(op, a, b)
    e_dev = e.to(dev)
    je = _sum_e(e_dev)
    le = torch.empty(n, D, dtype=torch.int64, device=dev)
    we = torch.empty(n, D, dtype=torch.int64, device=dev)
    if method == "scatter":
        tabs = torch.stack([lo_v, w_v])
        step = _rows_per_block(b - a + 1, 8)
        for r0 in range(0, n, step):
            idx = s[r0:r0 + step].to(torch.int64) - a
            got = bucket_times(bucket_sums(idx, b - a + 1, e_dev), tabs)               # [R, 2, 8]
            le[r0:r0 + step], we[r0:r0 + step] = got[:, 0], got[:, 1]
    else:
        lo_a, w0 = lo_a % P, w0 % P
        g = _native_lw(s, a, tab, e) if dev.type == "cpu" else None                 # [N, 3, 8], or None
        if g is not None:
            le = ((lo_a * je) % P + (w0 * g[:, 0]) % P + g[:, 1]) % P
            we = ((w0 * je) % P + g[:, 2]) % P
        else:
            step = _rows_per_block(t, 3)
            for r0 in range(0, n, step):
                idx = s[r0:r0 + step].to(torch.int32)
                if a:
                    idx = idx - a                       # out of place: int32 values would be shifted in the caller's tensor
                g = table_times(idx, tab, e_dev)                                        # [R, 3, 8]
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
def xbar(x: torch.Tensor, e: torch.Tensor, method: str | None = None) -> torch.Tensor | None:
    """``X^T e`` ``[K, 8]`` (canonical, on the CPU) of an integer input ``x`` ``[T, K]`` (any dtype, any device) at the
    column point ``e`` ``[T, 8]``: the integers of ``extfield.int_times(x.T, e, 31)``.  ``method``: ``"1"`` (one float64
    product with ``e`` as one 31-bit limb) or ``"scatter"`` (buckets per feature, the identity table); default
    :func:`mode` (``scatter`` for spans of at most ``max(256, T)`` values).  ``None`` if ``x`` spans more than
    :data:`SPAN` values or has an entry of ``2^16`` or more in absolute value (the caller then runs today's code)."""
    t, k = x.shape
    if x.numel() == 0:
        return torch.zeros(k, D, dtype=torch.int64)
    a, b = min_max(x)
    if b - a + 1 > SPAN or max(abs(a), abs(b)) >= 1 << 16:
        return None
    method = method or _method(b - a + 1, t)
    dev = x.device
    if method == "scatter":
        vals = torch.arange(a, b + 1, dtype=torch.int64)[None]
        xt = x.T
        out = torch.empty(k, D, dtype=torch.int64, device=dev)
        step = _rows_per_block(b - a + 1, 8)
        for k0 in range(0, k, step):
            idx = xt[k0:k0 + step].to(torch.int64).contiguous() - a
            out[k0:k0 + step] = bucket_times(bucket_sums(idx, b - a + 1, e.to(dev)), vals)[:, 0]
        return out.cpu()
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

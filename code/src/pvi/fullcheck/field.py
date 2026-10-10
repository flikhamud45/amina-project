"""Arithmetic over the BabyBear prime field, ``p = 15 * 2**27 + 1``.

Why this field.  Every value we ever reduce is either a field element
(< 2**31) or a small signed integer (an int8 weight or activation, or an int32
pre-activation), so a product of two field elements fits in a signed 64-bit
integer and torch's ``int64`` arithmetic is exact.  The multiplicative group has
a subgroup of order ``2**27``, which is what the number-theoretic transform
(NTT) behind the Reed--Solomon code needs.  The same field is used by Maverick
(arXiv 2609.10264), the closest published relative of this defence.

Everything here works on torch tensors on any device and is deterministic:
the same inputs give bit-identical outputs on CPU and GPU.  Small CPU operands of
the modular products run the same integer steps in numpy, whose per-call cost is
a fraction of torch's; on a GPU with int8 tensor cores the verifier's products can
run as exact int8 GEMMs instead (``int8_*``).
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
import torch

__all__ = [
    "P",
    "LOG2_P",
    "GENERATOR",
    "TWO_ADICITY",
    "to_field",
    "min_max",
    "root_of_unity",
    "power_table",
    "ntt",
    "rs_encode",
    "small_matmul_mod",
    "exact_chunk",
    "limbs_f64",
    "combine_limbs",
    "exact_gemm_i64",
    "field_matmul_mod",
    "NP_SMALL",
    "np_field_matmul_mod",
    "INT8_TERMS",
    "int8_ok",
    "int8_bytes",
    "int8_gemm",
    "int8_left",
    "int8_field_matmul",
    "int8_right",
    "int8_small_matmul",
    "int8_weight_matmul",
    "weight_matmul_mod",
]

P = 2013265921
"""The BabyBear prime ``15 * 2**27 + 1``."""
LOG2_P = math.log2(P)
GENERATOR = 31
"""A generator of the multiplicative group of ``F_P``."""
TWO_ADICITY = 27


def to_field(values: torch.Tensor) -> torch.Tensor:
    """Reduce an integer tensor (any sign, |x| < 2**62) into ``[0, P)`` as int64."""
    return torch.remainder(values.to(torch.int64), P)


def min_max(a: torch.Tensor) -> tuple[int, int]:
    """Exact ``(min, max)`` of a non-empty integer tensor: numpy's reductions on the CPU
    (about 2x torch's there), ``aminmax`` elsewhere (one device-to-host copy)."""
    if a.device.type == "cpu":
        from . import native_kernels                  # one parallel pass where the native library is built
        got = native_kernels.min_max(a)
        if got is not None:
            return got
        n = a.numpy()
        return int(n.min()), int(n.max())
    lo, hi = torch.stack(torch.aminmax(a)).tolist()
    return lo, hi


def root_of_unity(n: int) -> int:
    """A primitive ``n``-th root of unity (``n`` a power of two up to ``2**27``)."""
    if n < 1 or n & (n - 1) or n > (1 << TWO_ADICITY):
        raise ValueError(f"NTT length must be a power of two <= 2**27, got {n}")
    return pow(GENERATOR, (P - 1) // n, P)


@lru_cache(maxsize=64)
def _power_table_cpu(base: int, length: int) -> torch.Tensor:
    table = torch.ones(length, dtype=torch.int64)
    if length > 1:
        table[1] = base % P
    filled = 2
    while filled < length:
        take = min(filled, length - filled)
        step = pow(base, filled, P)
        table[filled:filled + take] = (table[:take] * step) % P
        filled += take
    return table


@lru_cache(maxsize=128)
def _power_table_on(base: int, length: int, device: str) -> torch.Tensor:
    return _power_table_cpu(base, length).to(device)


def power_table(base: int, length: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """``[base**0, base**1, ..., base**(length-1)] mod P`` as int64 (cached per device, so a
    GPU gets it once; callers only read it)."""
    return _power_table_on(base, length, str(torch.device(device)))


@lru_cache(maxsize=32)
def _bit_reverse_cpu(n: int) -> torch.Tensor:
    bits = n.bit_length() - 1
    idx = torch.arange(n, dtype=torch.int64)
    rev = torch.zeros_like(idx)
    for b in range(bits):
        rev |= ((idx >> b) & 1) << (bits - 1 - b)
    return rev


def ntt(values: torch.Tensor) -> torch.Tensor:
    """Number-theoretic transform along the last axis (length a power of two).

    ``values`` must already be reduced into ``[0, P)``.  The forward transform
    evaluates the polynomial whose coefficients are ``values`` at the powers of
    a primitive root of unity.
    """
    n = values.shape[-1]
    device = values.device
    out = values.to(torch.int64)[..., _bit_reverse_cpu(n).to(device)]
    length = 2
    while length <= n:
        half = length // 2
        twiddles = power_table(root_of_unity(length), half, device)
        shaped = out.reshape(*out.shape[:-1], n // length, length)
        even = shaped[..., :half]
        odd = (shaped[..., half:] * twiddles) % P
        out = torch.cat(((even + odd) % P, (even - odd) % P), dim=-1).reshape(out.shape)
        length *= 2
    return out


def rs_encode(rows: torch.Tensor, n_points: int) -> torch.Tensor:
    """Reed--Solomon encode each row: coefficients -> evaluations at ``n_points``.

    A row of length ``k`` becomes a codeword of length ``n_points``.  Two
    different rows disagree on at least ``n_points - k + 1`` positions (the
    code's distance), which is what the column spot check relies on.
    """
    k = rows.shape[-1]
    if k > n_points:
        raise ValueError(f"row length {k} exceeds codeword length {n_points}")
    padded = torch.zeros(*rows.shape[:-1], n_points, dtype=torch.int64, device=rows.device)
    padded[..., :k] = to_field(rows)
    return ntt(padded)


def small_matmul_mod(small: torch.Tensor, field: torch.Tensor, *, chunk: int = 8192) -> torch.Tensor:
    """``small @ field mod P`` where ``|small| <= 2**8`` and ``field`` is in ``[0, P)``.

    Uses float64 matrix products, which are exact while every partial sum stays
    below ``2**53``: a product is below ``2**39`` and we sum at most ``chunk``
    (``2**13``) of them before reducing.  This is the workhorse for anything that
    multiplies an int8 weight matrix by field elements.
    """
    k = small.shape[-1]
    if field.shape[0] != k:
        raise ValueError(f"shape mismatch {tuple(small.shape)} @ {tuple(field.shape)}")
    out = None
    for start in range(0, k, chunk):
        a = small[..., start:start + chunk].to(torch.float64)
        b = field[start:start + chunk].to(torch.float64)
        part = torch.remainder((a @ b).round().to(torch.int64), P)
        out = part if out is None else (out + part) % P
    return out


# -- products of two field-sized operands -----------------------------------------------
# A field element has 31 bits, so ``left`` is split into three 11-bit limbs: a limb times
# a field element is below 2**42 and a float64 GEMM over them is exact while each sum of
# absolute terms stays <= 2**53.  Integer terms with such a sum are exact in any
# summation order, so no BLAS kernel or GPU can round them.

_LIMB_BITS = 11
_LIMB_MAX = (1 << _LIMB_BITS) - 1
_EXACT = 1 << 53


def exact_chunk(max_left: int, max_right: int) -> int:
    """Longest contraction ``c`` with ``c * max_left * max_right <= 2**53``: every float64
    sum of ``c`` integer products of those magnitudes is exact."""
    c = _EXACT // (max_left * max_right)
    if c < 1:
        raise ValueError("operands too large for an exact float64 product")
    return c


def _reduced(a: torch.Tensor) -> torch.Tensor:
    """``to_field(a)``; on the CPU the (integer-division) remainder pass is skipped when
    one min/max pass shows ``a`` is already in ``[0, P)``, where it is the identity."""
    if a.device.type == "cpu" and a.dtype == torch.int64 and a.numel():
        lo, hi = min_max(a)
        if lo >= 0 and hi < P:
            return a
    return to_field(a)


@lru_cache(maxsize=None)
def _limb_consts(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """Limb shifts ``(0, 11, 22)`` and weights ``2**(11 i)`` (all ``< P``), ``[3, 1, 1]`` (read-only)."""
    shifts = torch.tensor([0, _LIMB_BITS, 2 * _LIMB_BITS], dtype=torch.int64, device=device)
    return shifts[:, None, None], (torch.ones_like(shifts) << shifts)[:, None, None]


def limbs_f64(left: torch.Tensor) -> torch.Tensor:
    """Field elements ``[..., r, K]`` -> float64 ``[..., 3r, K]``: the three 11-bit limbs
    stacked along the rows (limb ``i`` of row ``j`` is row ``i*r + j``)."""
    left = _reduced(left)
    shifts, _ = _limb_consts(str(left.device))
    parts = left.unsqueeze(-3) >> shifts          # [..., 3, r, K]
    parts[..., :2, :, :] &= _LIMB_MAX             # the top limb is below 2**9 already (P < 2**31)
    return parts.reshape(*left.shape[:-2], 3 * left.shape[-2], left.shape[-1]).to(torch.float64)


def combine_limbs(acc: torch.Tensor, r: int) -> torch.Tensor:
    """Exact limb products ``[..., 3r, M]`` (int64) -> ``[..., r, M]`` in ``[0, P)``."""
    _, weights = _limb_consts(str(acc.device))
    acc = torch.remainder(acc, P).reshape(*acc.shape[:-2], 3, r, acc.shape[-1])
    return (acc * weights).sum(-3) % P            # each term < 2**31 * 2**22, the sum < 2**55


def exact_gemm_i64(a: torch.Tensor, b: torch.Tensor, chunk: int) -> torch.Tensor:
    """``a @ b`` as int64 for integer-valued float64 ``a`` ``[..., m, K]`` and integer ``b``
    ``[..., K, n]`` (batch dimensions broadcast), when every block of ``chunk`` terms is
    exact (:func:`exact_chunk`).  ``b`` is converted to float64 once; the full blocks go
    through ONE batched GEMM on strided views (no copies, no loop over blocks) and the tail
    through one more, and the block sums (each ``<= 2**53``) are added in int64.

    Below 512 full blocks the result is the exact product.  From 512 blocks on, each block
    sum is reduced mod ``P`` first (so that their sum stays below ``2**62``): the result is
    then only congruent to ``a @ b`` mod ``P``, which is all the callers use (every one of
    them reduces it)."""
    k = a.shape[-1]
    bf = b.to(torch.float64)
    if k <= chunk:
        return (a @ bf).to(torch.int64)
    nc = k // chunk
    full = nc * chunk
    blocks = torch.matmul(a[..., :full].reshape(*a.shape[:-1], nc, chunk).transpose(-3, -2),
                          bf[..., :full, :].reshape(*bf.shape[:-2], nc, chunk, bf.shape[-1]))  # [..., nc, m, n]
    blocks = blocks.to(torch.int64)
    if nc >= 512:                  # keep the sum of the block sums below 2**62
        blocks = torch.remainder(blocks, P)
    out = blocks.sum(-3)
    if full < k:
        out += (a[..., full:] @ bf[..., full:, :]).to(torch.int64)
    return out


def field_matmul_mod(left: torch.Tensor, right: torch.Tensor, *, right_bound: int | None = None) -> torch.Tensor:
    """``left @ right mod P`` for field elements ``left`` ``[..., r, K]`` and ``right`` ``[..., K, M]``.

    ``right`` holds field elements (reduced into ``[0, P)`` here), or, when the caller
    has checked ``|right| < right_bound``, integers used as they are: the verifier's
    range-checked claims (``|z| < 2**29``) need no reduction pass and allow 4x longer
    exact blocks.  ``left`` is split into limbs stacked into one ``[3r, K]`` operand, so every
    block is ONE GEMM.  The result is the canonical residue in ``[0, P)`` of the integer product.
    """
    chunk = exact_chunk(_LIMB_MAX, (P if right_bound is None else right_bound) - 1)
    if _np_small(left, right):
        return torch.from_numpy(np_field_matmul_mod(left.numpy(), right.numpy(), right_bound is None, chunk))
    if right_bound is None:
        right = _reduced(right)
    return combine_limbs(exact_gemm_i64(limbs_f64(left), right, chunk), left.shape[-2])


# -- the same products in numpy, for small CPU operands ------------------------------------
# A numpy call costs ~1-2 us against ~4-5 us for a torch call, and a small modular product
# is ~15 calls of pure overhead.  The steps (exact float64 limb GEMMs, int64 floor-mod) are
# those above, so the results are the same integers.  numpy's BLAS threads are not bound by
# torch.set_num_threads, which is one more reason to keep this to small operands.

NP_SMALL = 1 << 16
"""Operand entries (``3 * left + right``) up to which CPU products run in numpy."""
_NP_SHIFTS = np.array([0, _LIMB_BITS, 2 * _LIMB_BITS], dtype=np.int64)[:, None, None]
_NP_WEIGHTS = np.int64(1) << _NP_SHIFTS


def _np_small(left: torch.Tensor, right: torch.Tensor) -> bool:
    """Small int64 ``left`` and int64 or int32 ``right`` (the wire's opened rows) on the CPU."""
    return (left.device.type == "cpu" and right.device.type == "cpu" and left.dtype == torch.int64
            and right.dtype in (torch.int64, torch.int32) and 3 * left.numel() + right.numel() <= NP_SMALL)


def _np_reduced(a: np.ndarray) -> np.ndarray:
    if a.size and (a.min() < 0 or a.max() >= P):
        return np.remainder(a, P)          # floor-mod: the representative torch.remainder gives
    return a


def np_field_matmul_mod(left: np.ndarray, right: np.ndarray, reduce_right: bool, chunk: int) -> np.ndarray:
    """:func:`field_matmul_mod` on int64 arrays, ``right`` also int32 (``reduce_right=False``:
    ``right`` is used as it is, and ``chunk`` must keep its blocks exact)."""
    if reduce_right:
        right = _np_reduced(right)
    left = _np_reduced(left)
    r, k = left.shape[-2], left.shape[-1]
    parts = left[..., None, :, :] >> _NP_SHIFTS          # [..., 3, r, K]
    parts[..., :2, :, :] &= _LIMB_MAX
    lf = parts.reshape(*left.shape[:-2], 3 * r, k).astype(np.float64)
    rf = right.astype(np.float64)
    acc = 0
    for s in range(0, max(k, 1), chunk):   # at most a few blocks: operands here have <= 2**16 entries
        acc = acc + np.matmul(lf[..., s:s + chunk], rf[..., s:s + chunk, :]).astype(np.int64)
    acc = np.remainder(acc, P).reshape(*acc.shape[:-2], 3, r, acc.shape[-1])
    return np.remainder((acc * _NP_WEIGHTS).sum(-3), P)


# -- the same products on int8 GEMMs (GPU tensor cores) -----------------------------------
# Flipping the sign bit of bytes 0-2 of an int32 (XOR 0x00808080) makes each of its four
# bytes a signed int8 s_b with  x = sum_b 256**b (s_b + o_b),  o = (128, 128, 128, 0)  (the top
# byte keeps its sign), for EVERY int32 x.  A product of two such bytes is at most 2**14 in
# magnitude, so an int8 GEMM over INT8_TERMS of them is exact in its int32 accumulator, and the
# offsets o are exact int64 arithmetic mod P afterwards.  Operands are zero-padded to the sizes
# torch._int_mm needs; a zero byte adds nothing to any product or column sum.

INT8_TERMS = 1 << 16
"""Contraction block of one int8 GEMM (``2**16`` terms below ``2**14`` sum below ``2**30``)."""
_BYTE_FLIP = 0x00808080
_BYTE_OFFSET = 128 * (1 + (1 << 8) + (1 << 16))         # O = sum_b 256**b o_b
_INT8: dict[str, bool] = {}


def _int_mm_exact(device: str) -> bool:
    """``torch._int_mm`` runs on ``device`` and matches float64, at the int8 bounds and at the
    accumulation bound of :data:`INT8_TERMS` terms (every entry ``2**30``)."""
    try:
        g = torch.Generator().manual_seed(0)
        a = torch.randint(-128, 128, (24, 4096), generator=g, dtype=torch.int64).to(torch.int8)
        b = torch.randint(-128, 128, (4096, 40), generator=g, dtype=torch.int64).to(torch.int8)
        a[0], b[:, 0] = -128, -128
        want = (a.double() @ b.double()).to(torch.int32)
        if not torch.equal(torch._int_mm(a.to(device), b.to(device)).cpu(), want):
            return False
        full = torch.full((24, INT8_TERMS), -128, dtype=torch.int8, device=device)
        return bool((torch._int_mm(full, full[:8].T.contiguous()) == 1 << 30).all())
    except (RuntimeError, AttributeError):       # no int8 GEMM on this build or GPU
        return False


def int8_ok(device) -> bool:
    """Whether the verifier's products on ``device`` run as int8 GEMMs: a GPU whose
    ``torch._int_mm`` passes a self-check (once per device).  Off the GPU the float64 limb
    products are faster; there :func:`int8_gemm` is a float64 emulation, which the tests use."""
    d = torch.device(device)
    if d.type != "cuda":
        return False
    key = str(d if d.index is not None else torch.device("cuda", torch.cuda.current_device()))
    if key not in _INT8:
        _INT8[key] = _int_mm_exact(key)
    return _INT8[key]


def int8_bytes(x: torch.Tensor) -> torch.Tensor:
    """int32-valued ``x [..., m]`` -> int8 ``[..., 4m]``: the bytes ``s_b`` above, little-endian."""
    y = torch.bitwise_xor(x, _BYTE_FLIP) if x.dtype == torch.int32 else x.to(torch.int32).bitwise_xor_(_BYTE_FLIP)
    return y.contiguous().view(torch.int8)


def int8_gemm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Exact ``a @ b`` in int32 for int8 ``a [m, k]`` and ``b [k, n]`` (``m > 16``, ``k`` and ``n``
    multiples of 8, both contiguous): ``torch._int_mm`` where :func:`int8_ok`, else float64
    (exact: every sum is below ``2**31``).  A GPU whose int8 GEMM fails at run time falls back."""
    if a.is_cuda and int8_ok(a.device):
        try:
            return torch._int_mm(a, b)
        except RuntimeError:
            _INT8[str(a.device)] = False
    return (a.to(torch.float64) @ b.to(torch.float64)).to(torch.int32)


def _pad8(n: int) -> int:
    return -(-n // 8) * 8


@lru_cache(maxsize=None)
def _byte_consts(device: str) -> tuple[torch.Tensor, torch.Tensor]:
    """``256**b`` and the offsets ``o_b`` for ``b < 4`` (read-only)."""
    return (torch.tensor([1, 1 << 8, 1 << 16, 1 << 24], dtype=torch.int64, device=device),
            torch.tensor([128, 128, 128, 0], dtype=torch.int64, device=device))


def int8_left(left: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """The operand of :func:`int8_field_matmul` for field elements ``left [r, N]``: the ``4r`` byte
    rows ``t_c`` (row ``c r + i`` is byte ``c`` of row ``i``) and a row of ones, padded; and
    ``sum_c 256**c t_c 1`` per row, mod P."""
    r, n = left.shape
    cb = int8_bytes(left).reshape(r, n, 4)
    lb = torch.zeros(max(24, _pad8(4 * r + 1)), _pad8(n), dtype=torch.int8, device=left.device)
    lb[:4 * r, :n] = cb.permute(2, 0, 1).reshape(4 * r, n)
    lb[4 * r, :n] = 1
    return lb, (cb.sum(1, dtype=torch.int64) * _byte_consts(str(left.device))[0]).sum(-1) % P


def int8_field_matmul(operand: tuple[torch.Tensor, torch.Tensor], right: torch.Tensor) -> torch.Tensor:
    """``left @ right mod P`` for field elements ``left [r, N]`` (``operand = int8_left(left)``)
    and int32 values ``right [N, M]`` (int32, or int64 holding int32 values; any sign), on int8
    GEMMs.

    With ``left = sum_c 256**c (t_c + o_c)`` and ``right = sum_b 256**b (s_b + o_b)`` bytewise,
    ``left @ right = sum_{b,c} 256**(b+c) t_c s_b + O sum_b 256**b 1^T s_b + O sum_c 256**c t_c 1
    + N O**2``: ONE GEMM of the byte rows of ``left`` and a row of ones with the ``[N, 4M]``
    bytes of ``right`` gives every ``t_c s_b`` and every column sum.  The result is the
    canonical residue.
    """
    lb, row_sums = operand
    r, (n, m) = row_sums.shape[0], right.shape
    if m == 0:
        return torch.zeros(r, 0, dtype=torch.int64, device=right.device)
    zb = int8_bytes(right).reshape(n, 4 * m)
    if lb.shape[1] != n or 4 * m % 8:
        zb = torch.nn.functional.pad(zb, (0, _pad8(4 * m) - 4 * m, 0, lb.shape[1] - n))
    acc = None
    for s in range(0, lb.shape[1], INT8_TERMS):
        a = lb if lb.shape[1] <= INT8_TERMS else lb[:, s:s + INT8_TERMS].contiguous()
        part = int8_gemm(a, zb[s:s + INT8_TERMS]).to(torch.int64)
        acc = part if acc is None else acc.add_(part)
    acc = acc[:4 * r + 1, :4 * m] % P
    w = _byte_consts(str(right.device))[0]
    per_c = (acc[:4 * r].reshape(4, r, m, 4) * w).sum(-1) % P          # [c, i, column]: sum_b 256**b t_c s_b
    main = (per_c * w[:, None, None]).sum(0)                            # < 4 * 2**31 * 2**24
    ones = (acc[4 * r].reshape(m, 4) * w).sum(-1) % P                   # sum_b 256**b 1^T s_b
    o = _BYTE_OFFSET
    return (main + o * ones[None, :] + o * row_sums[:, None] + (n * o * o) % P) % P


def int8_right(field: torch.Tensor) -> tuple[torch.Tensor, int]:
    """The operand of :func:`int8_small_matmul` for field elements ``field [R, K]``: the ``[K, 4R]``
    bytes ``t_c`` (column ``c R + i`` is byte ``c`` of row ``i``) and a column of ones, padded; and ``R``."""
    rr, k = field.shape
    tb = int8_bytes(field).reshape(rr, k, 4)
    out = torch.zeros(_pad8(k), _pad8(4 * rr + 1), dtype=torch.int8, device=field.device)
    out[:k, :4 * rr] = tb.permute(1, 2, 0).reshape(k, 4 * rr)
    out[:k, 4 * rr] = 1
    return out, rr


def int8_small_matmul(operand: tuple[torch.Tensor, int], small: torch.Tensor) -> torch.Tensor:
    """``field @ small.T mod P`` (``[R, M]``) for field elements ``field [R, K]`` (``operand =
    int8_right(field)``) and int8 ``small [M, K]``, on int8 GEMMs: with ``field = sum_c 256**c
    (t_c + o_c)`` bytewise, ``small @ field.T = sum_c 256**c (small @ t_c.T + o_c (small 1))``:
    ONE GEMM of ``small`` with the bytes of ``field`` and a column of ones."""
    right, rr = operand
    m, k = small.shape
    if m == 0:
        return torch.zeros(rr, 0, dtype=torch.int64, device=small.device)
    rows = max(24, _pad8(m))
    if rows != m or right.shape[0] != k:
        small = torch.nn.functional.pad(small, (0, right.shape[0] - k, 0, rows - m))
    acc = None
    for s in range(0, right.shape[0], INT8_TERMS):
        part = int8_gemm(small[:, s:s + INT8_TERMS].contiguous(), right[s:s + INT8_TERMS]).to(torch.int64)
        acc = part if acc is None else acc.add_(part)
    acc = acc[:m, :4 * rr + 1] % P
    w, offsets = _byte_consts(str(small.device))
    per_c = (acc[:, :4 * rr].reshape(m, 4, rr) + offsets[:, None] * acc[:, 4 * rr, None, None]) % P
    return ((per_c * w[:, None]).sum(1) % P).T


def int8_weight_matmul(small: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """``small @ field mod P`` (``[M, R]``) for int8 ``small [M, K]`` and field elements ``field
    [K, R]``, on int8 GEMMs: :func:`int8_small_matmul` of ``field``'s ``R`` columns against
    ``small``.  The result is the canonical residue."""
    return int8_small_matmul(int8_right(field.T), small).T


def weight_matmul_mod(small: torch.Tensor, field: torch.Tensor) -> torch.Tensor:
    """:func:`small_matmul_mod` for int8 weights ``small``: on int8 GEMMs where :func:`int8_ok`
    (the prover's folds and column openings on a GPU), else in float64.  Both return the
    canonical residue, so the results are bit-identical."""
    if small.dtype == torch.int8 and small.is_cuda and int8_ok(small.device):
        return int8_weight_matmul(small, field)
    return small_matmul_mod(small, field)

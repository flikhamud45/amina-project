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
a fraction of torch's.
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


def power_table(base: int, length: int, device: torch.device | str = "cpu") -> torch.Tensor:
    """``[base**0, base**1, ..., base**(length-1)] mod P`` as int64."""
    return _power_table_cpu(base, length).to(device)


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
        n = a.numpy()
        if n.min() >= 0 and n.max() < P:
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
    """Exact ``a @ b`` as int64 for integer-valued float64 ``a`` ``[..., m, K]`` and integer
    ``b`` ``[..., K, n]`` (batch dimensions broadcast), when every block of ``chunk``
    terms is exact (:func:`exact_chunk`).  ``b`` is converted to float64 once; the full
    blocks go through ONE batched GEMM on strided views (no copies, no loop over
    blocks) and the tail through one more, and the block sums (each ``<= 2**53``) are
    added in int64."""
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


def field_matmul_mod(left: torch.Tensor, right: torch.Tensor, *, right_bound: int | None = None,
                     left_limbs: torch.Tensor | None = None) -> torch.Tensor:
    """``left @ right mod P`` for field elements ``left`` ``[..., r, K]`` and ``right`` ``[..., K, M]``.

    ``right`` holds field elements (reduced into ``[0, P)`` here), or, when the caller
    has checked ``|right| < right_bound``, integers used as they are: the verifier's
    range-checked claims (``|z| < 2**29``) need no reduction pass and allow 4x longer
    exact blocks.  ``left`` is split into limbs stacked into one ``[3r, K]`` operand
    (``left_limbs`` may pass :func:`limbs_f64` of it, precomputed), so every block is ONE
    GEMM.  The result is the canonical residue in ``[0, P)`` of the integer product.
    """
    chunk = exact_chunk(_LIMB_MAX, (P if right_bound is None else right_bound) - 1)
    if left_limbs is None and _np_small(left, right):
        return torch.from_numpy(np_field_matmul_mod(left.numpy(), right.numpy(), right_bound is None, chunk))
    if right_bound is None:
        right = _reduced(right)
    lf = limbs_f64(left) if left_limbs is None else left_limbs
    return combine_limbs(exact_gemm_i64(lf, right, chunk), left.shape[-2])


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
    return (left.device.type == "cpu" and right.device.type == "cpu" and left.dtype == right.dtype == torch.int64
            and 3 * left.numel() + right.numel() <= NP_SMALL)


def _np_reduced(a: np.ndarray) -> np.ndarray:
    if a.size and (a.min() < 0 or a.max() >= P):
        return np.remainder(a, P)          # floor-mod: the representative torch.remainder gives
    return a


def np_field_matmul_mod(left: np.ndarray, right: np.ndarray, reduce_right: bool, chunk: int) -> np.ndarray:
    """:func:`field_matmul_mod` on int64 arrays (``reduce_right=False``: ``right`` is used as
    it is, and ``chunk`` must keep its blocks exact)."""
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

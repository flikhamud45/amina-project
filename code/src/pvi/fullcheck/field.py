"""Arithmetic over the BabyBear prime field, ``p = 15 * 2**27 + 1``.

Why this field.  Every value we ever reduce is either a field element
(< 2**31) or a small signed integer (an int8 weight or activation, or an int32
pre-activation), so a product of two field elements fits in a signed 64-bit
integer and torch's ``int64`` arithmetic is exact.  The multiplicative group has
a subgroup of order ``2**27``, which is what the number-theoretic transform
(NTT) behind the Reed--Solomon code needs.  The same field is used by Maverick
(arXiv 2609.10264), the closest published relative of this defence.

Everything here works on torch tensors on any device and is deterministic:
the same inputs give bit-identical outputs on CPU and GPU.
"""

from __future__ import annotations

import math
from functools import lru_cache

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
    "intt",
    "rs_encode",
    "small_matmul_mod",
    "field_matmul_mod",
    "signed",
]

P = 2013265921
"""The BabyBear prime ``15 * 2**27 + 1``."""
LOG2_P = math.log2(P)
GENERATOR = 31
"""A generator of the multiplicative group of ``F_P``."""
TWO_ADICITY = 27

_HALF = P // 2


def to_field(values: torch.Tensor) -> torch.Tensor:
    """Reduce an integer tensor (any sign, |x| < 2**62) into ``[0, P)`` as int64."""
    return torch.remainder(values.to(torch.int64), P)


def signed(values: torch.Tensor) -> torch.Tensor:
    """Map field elements back to the symmetric range ``(-P/2, P/2]``."""
    values = values.to(torch.int64)
    return torch.where(values > _HALF, values - P, values)


def root_of_unity(n: int, inverse: bool = False) -> int:
    """A primitive ``n``-th root of unity (``n`` a power of two up to ``2**27``)."""
    if n < 1 or n & (n - 1) or n > (1 << TWO_ADICITY):
        raise ValueError(f"NTT length must be a power of two <= 2**27, got {n}")
    w = pow(GENERATOR, (P - 1) // n, P)
    return pow(w, P - 2, P) if inverse else w


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


def ntt(values: torch.Tensor, inverse: bool = False) -> torch.Tensor:
    """Number-theoretic transform along the last axis (length a power of two).

    ``values`` must already be reduced into ``[0, P)``.  The forward transform
    evaluates the polynomial whose coefficients are ``values`` at the powers of
    a primitive root of unity; the inverse undoes it exactly.
    """
    n = values.shape[-1]
    device = values.device
    out = values.to(torch.int64)[..., _bit_reverse_cpu(n).to(device)]
    length = 2
    while length <= n:
        half = length // 2
        twiddles = power_table(root_of_unity(length, inverse), half, device)
        shaped = out.reshape(*out.shape[:-1], n // length, length)
        even = shaped[..., :half]
        odd = (shaped[..., half:] * twiddles) % P
        out = torch.cat(((even + odd) % P, (even - odd) % P), dim=-1).reshape(out.shape)
        length *= 2
    if inverse:
        out = (out * pow(n, P - 2, P)) % P
    return out


def intt(values: torch.Tensor) -> torch.Tensor:
    return ntt(values, inverse=True)


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


def field_matmul_mod(left: torch.Tensor, right: torch.Tensor, *, chunk: int = 1 << 10) -> torch.Tensor:
    """``left @ right mod P`` for arbitrary field elements (both in ``[0, P)``).

    ``left`` is split into three 11-bit limbs so every product is below
    ``2**42`` and at most ``chunk`` (``2**10``) of them are summed in float64
    before reducing, keeping every partial sum below ``2**52``.
    Used by the verifier, whose operands are challenge vectors (``left``, a few
    rows) times claimed pre-activations reduced into the field (``right``).
    """
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

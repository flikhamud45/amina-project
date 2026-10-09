"""The extension field ``F = F_{p^8} = F_p[x] / (x^8 - 11)`` over BabyBear, on torch int64 tensors ``[..., 8]``.

Every challenge of V1's lookup argument (the logUp point ``alpha``, the GKR's sum-check and folding challenges)
lives in ``F``: its ``p^8 ~ 2^247`` elements keep each Schwartz--Zippel term far below ``2^-128`` even under
Fiat--Shamir, where the 31-bit base field would not (``docs/improvements/C2_v1_proof_size.md``).  ``x^8 - 11`` is
irreducible over ``F_p`` by the criterion for binomials (Lidl and Niederreiter, Thm. 3.75): 11 is a quadratic
non-residue mod ``p``, so its order has the full 2-adic part of ``p - 1``, and ``p = 1 mod 4`` (both checked by
``tests/test_extfield.py``).

An element is a vector of 8 coefficients in ``[0, p)``, coefficient ``i`` of ``x^i``; every function returns
canonical values.  Products are reduced before they are summed, so every intermediate fits int64
(``(p - 1)^2 < 2^62``, eight reduced terms ``< 2^34``).
"""

from __future__ import annotations

import torch

from .field import P

__all__ = ["D", "BETA", "const", "gen", "add", "sub", "neg", "mul", "scal", "lift", "inv", "inv_fermat", "frobenius",
           "batch_inv", "power",
           "eq_eval", "eq_table", "prefix_eq_sum", "lagrange4", "frac_sum", "small_times", "pack", "unpack", "equal"]

D = 8
BETA = 11


def const(c: int, device="cpu") -> torch.Tensor:
    """The constant ``c`` of ``F``."""
    v = torch.zeros(D, dtype=torch.int64, device=device)
    v[0] = c % P
    return v


def gen(device="cpu") -> torch.Tensor:
    """``x``, the generator of ``F`` over ``F_p``."""
    v = torch.zeros(D, dtype=torch.int64, device=device)
    v[1] = 1
    return v


def add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a + b) % P


def sub(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    return (a - b) % P


def neg(a: torch.Tensor) -> torch.Tensor:
    return (-a) % P


def mul(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``a b`` (broadcast over the leading axes): ``sum_i a_i x^i b``, where ``x^i b`` rotates ``b``'s coefficients
    by ``i`` and multiplies the ``i`` that wrap past ``x^7`` by 11 (``x^8 = 11``): with ``bb = [11 b, b]``,
    ``x^i b = bb[8 - i : 16 - i]``, all eight read as one ``[8, 8]`` window view."""
    a, b = torch.broadcast_tensors(a, b)
    bb = torch.cat([(b * BETA) % P, b], -1)
    rot = bb.unfold(-1, D, 1).flip(-2)[..., :D, :]                  # [..., i, k]: coefficient k of x^i b
    return ((a[..., :, None] * rot) % P).sum(-2) % P


def scal(a: torch.Tensor, s) -> torch.Tensor:
    """``s a`` for base-field scalars ``s`` (an int, or an int64 tensor broadcast over ``a``'s leading axes)."""
    s = (torch.as_tensor(s, dtype=torch.int64, device=a.device) % P)
    return (a * (s[..., None] if s.dim() else s)) % P


def lift(s) -> torch.Tensor:
    """Base-field integers ``[...]`` as elements of ``F`` ``[..., 8]``."""
    s = torch.as_tensor(s, dtype=torch.int64)
    out = torch.zeros(*s.shape, D, dtype=torch.int64, device=s.device)
    out[..., 0] = s % P
    return out


def power(a: torch.Tensor, e: int) -> torch.Tensor:
    """``a^e`` by square and multiply (``e >= 0``)."""
    out = torch.zeros_like(a)
    out[..., 0] = 1
    base = a.clone()
    while e:
        if e & 1:
            out = mul(out, base)
        base = mul(base, base)
        e >>= 1
    return out


_ORDER = P ** D - 1


_ZETA = pow(BETA, (P - 1) // D, P)              # x^p = 11^((p-1)/8) x, since x^8 = 11 and 8 | p - 1
_FROB = torch.tensor([[pow(_ZETA, i * k, P) for k in range(D)] for i in range(D)], dtype=torch.int64)


def frobenius(a: torch.Tensor, i: int = 1) -> torch.Tensor:
    """``a^(p^i)``: coefficient ``k`` times ``zeta^(i k)`` (the Frobenius map fixes ``F_p`` and sends ``x`` to
    ``zeta x``)."""
    return (a * _FROB[i % D].to(a.device)) % P


def inv(a: torch.Tensor) -> torch.Tensor:
    """``a^-1`` (0 maps to 0) through the norm: ``b = a^(p + p^2 + ... + p^7)`` is a product of Frobenius images,
    ``N(a) = a b`` lies in ``F_p``, and ``a^-1 = b / N(a)``: six products and one base-field inversion."""
    b = frobenius(a, 1)
    for i in range(2, D):
        b = mul(b, frobenius(a, i))
    norm = mul(a, b)[..., 0]                       # coefficients 1..7 are zero
    return scal(b, torch.as_tensor([pow(int(v), P - 2, P) for v in norm.reshape(-1).tolist()],
                                   dtype=torch.int64, device=a.device).view(norm.shape))


def inv_fermat(a: torch.Tensor) -> torch.Tensor:
    """``a^-1`` by Fermat (``a^(p^8 - 2)``): the reference :func:`inv` is tested against."""
    return power(a, _ORDER - 1)


def batch_inv(a: torch.Tensor) -> torch.Tensor:
    """The inverses of the rows of ``a`` ``[n, 8]`` (all non-zero) with one :func:`inv` (Montgomery's trick)."""
    n = a.shape[0]
    if n == 0:
        return a.clone()
    prefix = [a[0]]
    for i in range(1, n):
        prefix.append(mul(prefix[-1], a[i]))
    acc = inv(prefix[-1])
    out = [None] * n
    for i in range(n - 1, 0, -1):
        out[i] = mul(acc, prefix[i - 1])
        acc = mul(acc, a[i])
    out[0] = acc
    return torch.stack(out)


def equal(a: torch.Tensor, b: torch.Tensor) -> bool:
    return bool(torch.equal(a % P, b % P))


def eq_eval(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``eq(a, b) = prod_i (a_i b_i + (1 - a_i)(1 - b_i))`` for points ``a, b`` ``[k, 8]``."""
    one = const(1, a.device)
    out = one
    for i in range(a.shape[0]):
        out = mul(out, add(mul(a[i], b[i]), mul(sub(one, a[i]), sub(one, b[i]))))
    return out


def eq_table(r: torch.Tensor) -> torch.Tensor:
    """``[2^k, 8]``: entry ``y`` is ``eq(r, bits(y))``, bit ``i`` of ``y`` against ``r[i]`` (LSB first)."""
    one = const(1, r.device)
    t = one[None]
    for i in range(r.shape[0]):
        t = torch.cat([mul(t, sub(one, r[i])), mul(t, r[i])])
    return t


def prefix_eq_sum(r: torch.Tensor, n: int) -> torch.Tensor:
    """``sum_{y < n} eq(r, bits(y))`` in ``O(k)`` from the binary digits of ``n`` (``r`` ``[k, 8]``)."""
    one = const(1, r.device)
    k = r.shape[0]
    if n >= 1 << k:
        return one
    total, acc = const(0, r.device), one
    for i in range(k - 1, -1, -1):
        if (n >> i) & 1:
            total = add(total, mul(acc, sub(one, r[i])))     # bit i = 0, the lower bits free (their eq sums to 1)
            acc = mul(acc, r[i])
        else:
            acc = mul(acc, sub(one, r[i]))
    return total


def lagrange4(g0: torch.Tensor, g1: torch.Tensor, g2: torch.Tensor, g3: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    """The cubic through ``(0, g0), (1, g1), (2, g2), (3, g3)``, evaluated at ``r``."""
    pts = (0, 1, 2, 3)
    out = const(0, r.device)
    for i, gi in enumerate((g0, g1, g2, g3)):
        num, den = const(1, r.device), 1
        for j in pts:
            if j != i:
                num = mul(num, sub(r, const(j, r.device)))
                den *= i - j
        out = add(out, scal(mul(num, gi), pow(den % P, P - 2, P)))
    return out


def frac_sum(num: torch.Tensor, den: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``sum_i num_i / den_i`` as one fraction ``(n, d)``, by a pairwise tree (no inversion)."""
    zero, one = const(0, num.device), const(1, num.device)
    if num.shape[0] == 0:
        return zero, one
    while num.shape[0] > 1:
        if num.shape[0] % 2:
            num = torch.cat([num, zero[None]])
            den = torch.cat([den, one[None]])
        num, den = add(mul(num[0::2], den[1::2]), mul(num[1::2], den[0::2])), mul(den[0::2], den[1::2])
    return num[0], den[0]


def small_times(mat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """``mat @ vec`` in ``F`` for an integer matrix ``mat`` ``[R, C]`` (``|entries| < 2^31``) and ``vec`` ``[C, 8]``:
    one exact base-field product per coefficient plane."""
    from .field import field_matmul_mod, to_field
    m = to_field(mat)
    return field_matmul_mod(vec.T.contiguous(), m.T.contiguous()).T.contiguous()   # [R, 8]


def pack(a: torch.Tensor) -> bytes:
    """The coefficients of ``a`` ``[..., 8]`` in 31-bit packing (``claimcodec.pack_field``)."""
    from .claimcodec import pack_field
    return pack_field([a.reshape(-1)])


def unpack(buf, count: int) -> torch.Tensor:
    """``count`` elements of ``F`` from :func:`pack`'s bytes (raises ``claimcodec.ClaimCodecError`` on other bytes
    or on a coefficient outside ``[0, p)``)."""
    from .claimcodec import ClaimCodecError, unpack_field
    out = torch.from_numpy(unpack_field(buf, count * D).astype("int64")).view(count, D)
    if bool((out >= P).any()):
        raise ClaimCodecError("an extension-field coefficient outside the field")
    return out

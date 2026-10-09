"""The extension field ``F_{p^8}`` of V1's lookup argument (``pvi.fullcheck.extfield``) and its challenges.

The field axioms on random elements, ``x^8 = 11``, the irreducibility criterion of ``x^8 - 11``, inverses, the
eq tables and their prefix sums, the cubic interpolation, the fraction tree, the integer-matrix products, the
31-bit packing, and ``Challenger.ext`` (uniform, deterministic under Fiat--Shamir, ``nonbase`` outside
``F_p + x F_p``).
"""

from __future__ import annotations

import itertools

import pytest
import torch

from pvi.fullcheck import extfield as ef
from pvi.fullcheck import protocol as proto
from pvi.fullcheck.field import P


def _rand(g, *shape):
    return torch.randint(0, P, (*shape, ef.D), generator=g, dtype=torch.int64)


def _naive_mul(a, b):
    out = [0] * ef.D
    for i, j in itertools.product(range(ef.D), repeat=2):
        t = int(a[i]) * int(b[j])
        if i + j < ef.D:
            out[i + j] += t
        else:
            out[i + j - ef.D] += ef.BETA * t
    return torch.tensor([v % P for v in out], dtype=torch.int64)


def test_the_modulus_is_irreducible():
    # x^t - a is irreducible over F_p iff every prime factor of t divides ord(a) but not (p-1)/ord(a), and
    # p = 1 mod 4 when 4 | t (Lidl and Niederreiter, Thm. 3.75): here t = 8 and 11 is a non-residue
    assert pow(ef.BETA, (P - 1) // 2, P) == P - 1 and P % 4 == 1


def test_the_field_axioms_hold():
    g = torch.Generator().manual_seed(1)
    a, b, c = _rand(g, 2000), _rand(g, 2000), _rand(g, 2000)
    a[0] = P - 1
    b[1] = 0
    one, zero = ef.const(1), ef.const(0)
    assert torch.equal(ef.mul(a, b), ef.mul(b, a))
    assert torch.equal(ef.mul(ef.mul(a, b), c), ef.mul(a, ef.mul(b, c)))
    assert torch.equal(ef.mul(a, ef.add(b, c)), ef.add(ef.mul(a, b), ef.mul(a, c)))
    assert torch.equal(ef.mul(a, one), a) and torch.equal(ef.add(a, zero), a)
    assert torch.equal(ef.add(a, ef.neg(a)), torch.zeros_like(a))
    for i in range(20):
        assert torch.equal(ef.mul(a[i], b[i]), _naive_mul(a[i], b[i]))
    assert torch.equal(ef.power(ef.gen(), 8), ef.const(ef.BETA))          # x^8 = 11
    assert torch.equal(ef.scal(a, 7), ef.mul(a, ef.const(7)))
    assert torch.equal(ef.scal(a, torch.full((2000,), 7)), ef.mul(a, ef.const(7)))


def test_inverses():
    g = torch.Generator().manual_seed(2)
    a = _rand(g, 6)
    for x in a:
        assert torch.equal(ef.mul(x, ef.inv(x)), ef.const(1))
    inv = ef.batch_inv(a)
    assert torch.equal(ef.mul(a, inv), ef.const(1).expand(6, ef.D))


@pytest.mark.parametrize("k", [0, 1, 3, 6])
def test_eq_tables_and_prefix_sums(k):
    g = torch.Generator().manual_seed(k)
    r, b = _rand(g, k), _rand(g, k)
    t = ef.eq_table(r)
    assert t.shape == (1 << k, ef.D)
    total = ef.const(0)
    for y in range(1 << k):
        bits = torch.stack([ef.const((y >> i) & 1) for i in range(k)]) if k else torch.zeros(0, ef.D, dtype=torch.int64)
        assert torch.equal(t[y], ef.eq_eval(r, bits))
        total = ef.add(total, t[y])
        assert torch.equal(ef.prefix_eq_sum(r, y + 1), total)
    assert torch.equal(total, ef.const(1))
    assert torch.equal(ef.prefix_eq_sum(r, 0), ef.const(0))
    assert torch.equal(ef.eq_eval(r, b), ef.eq_eval(b, r))


def test_cubic_interpolation():
    g = torch.Generator().manual_seed(3)
    coef = _rand(g, 4)
    r = _rand(g, 1)[0]

    def f(x):
        out, xp = ef.const(0), ef.const(1)
        for c in coef:
            out = ef.add(out, ef.mul(c, xp))
            xp = ef.mul(xp, x)
        return out

    vals = [f(ef.const(i)) for i in range(4)]
    assert torch.equal(ef.lagrange4(*vals, r), f(r))


def test_the_fraction_tree():
    g = torch.Generator().manual_seed(4)
    for n in (0, 1, 2, 7, 33):
        num, den = _rand(g, n), _rand(g, n)
        fn, fd = ef.frac_sum(num, den)
        want = ef.const(0)
        for i in range(n):
            want = ef.add(want, ef.mul(num[i], ef.inv(den[i])))
        assert torch.equal(ef.mul(fn, ef.inv(fd)), want)


def test_integer_matrix_products_and_packing():
    g = torch.Generator().manual_seed(5)
    mat = torch.randint(-(1 << 30), 1 << 30, (9, 13), generator=g)
    vec = _rand(g, 13)
    got = ef.small_times(mat, vec)
    for i in range(9):
        want = ef.const(0)
        for j in range(13):
            want = ef.add(want, ef.scal(vec[j], int(mat[i, j]) % P))
        assert torch.equal(got[i], want)
    assert torch.equal(ef.unpack(ef.pack(vec), 13), vec)


def test_extension_challenges():
    a = proto.Challenger(fiat_shamir=True)
    b = proto.Challenger(fiat_shamir=True)
    for ch in (a, b):
        ch.absorb(b"x", b"statement")
    x, y = a.ext("alpha", 3, nonbase=True), b.ext("alpha", 3, nonbase=True)
    assert torch.equal(x, y) and x.shape == (3, ef.D) and bool(((x >= 0) & (x < P)).all())
    assert bool(x[:, 2:].any(1).all())                       # outside F_p + x F_p
    assert not torch.equal(a.ext("beta", 3), x)
    inter = proto.Challenger(seed=7)
    assert inter.ext("alpha").shape == (1, ef.D)

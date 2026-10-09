"""The windowed logUp argument of V1 (``pvi.fullcheck.logup``; ``V1_SPEC.md`` tests T3 and kill criterion K1).

* A two-instance toy end to end: the honest proof passes D5, every GKR, the final claims (``P_hat = I``,
  ``Q_hat = alpha I - sum eq v + (1 - I)``) and F4.
* Multiplicities: a count ``>= p`` with compensating counts, and a total off by one, fail D5; a count moved
  between bins fails F4; an off-table leaf with counts that still sum to ``N`` fails F4.
* Lemma 1 by brute force on ``F_49 = F_7[x] / (x^2 - 3)``: whenever ``0 <= m_t < 7`` and ``sum m = N``, the
  rational identity holds only if every leaf is in the table; dropping either condition admits an off-table leaf.
* K1 (``PVI_SLOW=1``): an ``n = 20`` instance, honest accepted, every element tampered rejected.
"""

from __future__ import annotations

import itertools
import os

import numpy as np
import pytest
import torch

from pvi.fullcheck import extfield as ef
from pvi.fullcheck import logup
from pvi.fullcheck import logup_gkr as gkr
from pvi.fullcheck import protocol as proto
from pvi.fullcheck.field import P


def _chs():
    a, b = proto.Challenger(fiat_shamir=True), proto.Challenger(fiat_shamir=True)
    for c in (a, b):
        c.absorb(b"statement", b"logup test")
    return a, b


WIDTHS = (1, 3, 4)
SHAPES = ((5, 6), (3, 6))          # two instances: [R, T]


def _entries(seed):
    g = torch.Generator().manual_seed(seed)
    out = []
    for rows, cols in SHAPES:
        w = torch.tensor(WIDTHS[1:])[torch.randint(0, 2, (rows, 1), generator=g)].expand(rows, cols).clone()
        w[0, :2] = 1                                     # two exceptions (W = 1, delta = 0)
        d = (torch.rand(rows, cols, generator=g) * w).floor().to(torch.int64)
        out.append((d, w))
    return out


def _run(entries, m=None, seed=0):
    """Prover and verifier of the toy; the verdict of each check."""
    tau, offs = logup.table(WIDTHS)
    if m is None:
        m = sum(logup.multiplicities(d, w, offs, tau.shape[0]) for d, w in entries)
    total = sum(d.numel() for d, _ in entries)
    cp, cv = _chs()
    for c in (cp, cv):
        c.absorb(b"cut/mult", m.numpy().tobytes())
    alpha = cp.ext("cut/alpha", nonbase=True)[0]
    assert torch.equal(alpha, cv.ext("cut/alpha", nonbase=True)[0])
    layout = []
    for d, _ in entries:
        rows, cols = d.shape
        n_col, n_row = max(1, (cols - 1).bit_length()), max(0, (rows - 1).bit_length())
        layout.append((n_col, n_col + n_row))
    insts = [logup.leaves(d, w, nc, n, alpha) for (d, w), (nc, n) in zip(entries, layout)]
    trs = logup.prove_instances(insts, cp)
    verdict = {"mult": logup.multiplicities_ok(m, tau.shape[0], total)}
    outs = logup.verify_instances(trs, [n for _, n in layout], cv)
    verdict["gkr"] = outs is not None
    if outs is None:
        return verdict
    final = True
    for (d, w), (nc, n), (p0, q0, rho, p_hat, q_hat) in zip(entries, layout, outs):
        rows, cols = d.shape
        i_b = logup.indicator(rho, rows, cols, nc)
        eq = ef.eq_table(rho).view(1 << (n - nc), 1 << nc, ef.D)[:rows, :cols]
        v = ef.lift(d) + ef.scal(ef.gen(), w) % P
        sum_ev = ef.mul(eq, v % P).reshape(-1, ef.D).sum(0) % P
        want_q = ef.add(ef.sub(ef.mul(alpha, i_b), sum_ev), ef.sub(ef.const(1), i_b))
        final &= ef.equal(p_hat, i_b) and ef.equal(q_hat, want_q)
    verdict["final"] = final
    verdict["logup"] = logup.logup_ok([(o[0], o[1]) for o in outs], m, alpha, tau)
    return verdict


def test_the_table():
    tau, offs = logup.table((4, 1, 3, 3))
    assert offs == {1: 0, 3: 1, 4: 4} and tau.shape == (8, ef.D)
    assert tau[5].tolist() == [1, 4, 0, 0, 0, 0, 0, 0]          # 1 + 4x
    with pytest.raises(ValueError):
        logup.table((0, 2))


def test_honest_toy_passes_every_check():
    assert _run(_entries(1)) == {"mult": True, "gkr": True, "final": True, "logup": True}


def test_multiplicity_cheats():
    entries = _entries(2)
    tau, offs = logup.table(WIDTHS)
    m = sum(logup.multiplicities(d, w, offs, tau.shape[0]) for d, w in entries)
    big = m.clone()
    i, j = int(torch.nonzero(m)[0]), int(torch.nonzero(m)[1])
    assert logup.multiplicities_ok(m, tau.shape[0], 48)
    big[i] += P                                        # m_i >= p (p = 0 in F, so the fractions do not see it)
    assert not logup.multiplicities_ok(big, tau.shape[0], 48)
    big[j] -= P                                        # compensated in another bin: the total is right again
    assert not logup.multiplicities_ok(big, tau.shape[0], 48)
    for off in (1, -1):
        bad = m.clone()
        bad[i] += off
        assert not logup.multiplicities_ok(bad, tau.shape[0], 48)
    assert not logup.multiplicities_ok(m[:-1], tau.shape[0], 48)
    assert not logup.multiplicities_ok(m.to(torch.float64), tau.shape[0], 48)
    moved = m.clone()
    moved[i] -= 1
    moved[(i + 1) % m.numel()] += 1
    v = _run(entries, moved)
    assert v["mult"] and v["gkr"] and v["final"] and not v["logup"]


def test_an_off_table_leaf_is_caught():
    entries = _entries(3)
    d, w = entries[0]
    tau, offs = logup.table(WIDTHS)
    m = sum(logup.multiplicities(dd, ww, offs, tau.shape[0]) for dd, ww in entries)
    for bad in (int(w[2, 3]), -1):                     # delta = W (one past the window) and delta = -1
        d2 = d.clone()
        old = int(d2[2, 3])
        d2[2, 3] = bad
        m2 = m.clone()
        m2[offs[int(w[2, 3])] + old] -= 1              # drop the honest count, keep the total in another bin
        m2[offs[int(w[2, 3])] + (old + 1) % int(w[2, 3])] += 1
        v = _run([(d2, w), entries[1]], m2)
        assert v["mult"] and v["gkr"] and v["final"] and not v["logup"]


# ------------------------------------------------------------------ Lemma 1 on F_49 = F_7[x]/(x^2 - 3)
Q7 = 7


def _gf49_all():
    return [(a, b) for a in range(Q7) for b in range(Q7)]


def _mul49(u, v):
    return ((u[0] * v[0] + 3 * u[1] * v[1]) % Q7, (u[0] * v[1] + u[1] * v[0]) % Q7)


def _inv49(u):
    return next(v for v in _gf49_all() if _mul49(u, v) == (1, 0))


def test_lemma_1_by_brute_force():
    assert pow(3, (Q7 - 1) // 2, Q7) == Q7 - 1          # x^2 - 3 is irreducible over F_7
    table = [(t, w) for w in (1, 2) for t in range(w)]   # t + x w
    off = [(2, 2), (3, 0)]                               # off the table: t = W, and a tag-0 value
    univ = table + off
    xs = [x for x in _gf49_all() if x not in univ]
    # inv[X, v] = 1 / (X - v): Phi(X) = sum_v d_v / (X - v) with d_v = c_v - m_v (m_v = 0 off the table)
    inv = np.array([[_inv49(((x[0] - v[0]) % Q7, (x[1] - v[1]) % Q7)) for v in univ] for x in xs])   # [|X|, 5, 2]

    def identity(d):                                     # d [..., 5] integer coefficients
        return ~np.any(np.einsum("...v,xvc->...xc", d % Q7, inv) % Q7, axis=(-1, -2))

    for n in range(1, 10):
        cs = np.array([np.bincount(c, minlength=len(univ)) for c in itertools.combinations_with_replacement(range(len(univ)), n)])
        ms = np.array([m for m in itertools.product(range(Q7), repeat=len(table)) if sum(m) == n])
        if not ms.size:
            continue
        mfull = np.concatenate([ms, np.zeros((len(ms), len(off)), dtype=ms.dtype)], 1)
        holds = identity(cs[:, None, :] - mfull[None, :, :])                 # [multisets, m]
        has_off = cs[:, len(table):].sum(1) > 0
        assert not np.any(holds & has_off[:, None]), n
    # each condition is load-bearing: 7 copies of an off-table value
    seven = np.zeros(len(univ), dtype=np.int64)
    seven[len(table)] = Q7
    m_big = np.zeros(len(univ), dtype=np.int64)
    m_big[0] = Q7                                        # m_t = p, sum = N: the identity holds
    assert identity(seven - m_big)
    assert identity(seven)                               # m = 0, sum m != N: the identity holds


@pytest.mark.skipif(os.environ.get("PVI_SLOW") != "1", reason="K1: about a minute on a laptop (PVI_SLOW=1)")
def test_k1_a_large_instance_accepts_honest_and_rejects_every_tamper():
    n = 20
    g = torch.Generator().manual_seed(0)
    p = torch.zeros(1 << n, ef.D, dtype=torch.int64)
    p[:, 0] = 1
    q = torch.randint(0, P, (1 << n, ef.D), generator=g)
    tr = gkr.prove(p, q, _chs()[0], "cut/0/")
    out = gkr.verify(tr, _chs()[1], "cut/0/")
    assert out is not None
    num, den = ef.frac_sum(p, q)
    assert ef.equal(ef.mul(out[0], den), ef.mul(num, out[1]))
    flat = tr.flat()
    for i in range(flat.shape[0]):
        bad = flat.clone()
        bad[i, i % ef.D] = (bad[i, i % ef.D] + 1) % P
        forged = gkr.Transcript.from_bytes(bytes([n]) + ef.pack(bad), n)
        res = gkr.verify(forged, _chs()[1], "cut/0/")
        if res is not None:                              # only the top or the last values can pass the rounds
            assert not (ef.equal(ef.mul(res[0], den), ef.mul(num, res[1]))
                        and ef.equal(res[3], ef.mul(ef.eq_table(res[2]), p).sum(0) % P)), i

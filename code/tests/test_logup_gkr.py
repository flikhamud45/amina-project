"""The fractional-sum GKR of V1's logUp argument (``pvi.fullcheck.logup_gkr``), eager (``V1_SPEC.md`` test T2).

For ``n = 1 .. 14`` and random dense lengths (real leaves ``(1, alpha - v)``, padding ``(0, 1)``): the honest
transcript is accepted, its root is the sum of the leaves' fractions and its final claims are the leaf layer's
multilinear extensions at ``rho``; the vectorised verifier agrees with the round-by-round reference; every
transcript element tampered in turn is rejected or ends in a final mismatch the caller detects; the transcript
bytes round-trip and malformed bytes raise.
"""

from __future__ import annotations

import pytest
import torch

from pvi.fullcheck import extfield as ef
from pvi.fullcheck import logup_gkr as gkr
from pvi.fullcheck import protocol as proto
from pvi.fullcheck.claimcodec import ClaimCodecError, field_size
from pvi.fullcheck.field import P


def _rand(g, *shape):
    return torch.randint(0, P, (*shape, ef.D), generator=g, dtype=torch.int64)


def _leaves(n, dense, seed):
    g = torch.Generator().manual_seed(seed)
    alpha = _rand(g, 1)[0]
    vals = torch.randint(-(1 << 20), 1 << 20, (dense,), generator=g)
    p = ef.lift(torch.zeros(1 << n, dtype=torch.int64))
    q = ef.lift(torch.ones(1 << n, dtype=torch.int64))
    p[:dense] = ef.const(1)
    q[:dense] = ef.sub(alpha, ef.lift(vals))
    return p, q


def _pair(fs, seed=0):
    if fs:
        a, b = proto.Challenger(fiat_shamir=True), proto.Challenger(fiat_shamir=True)
        for c in (a, b):
            c.absorb(b"statement", b"gkr test")
        return a, b
    return proto.Challenger(seed=seed), proto.Challenger(seed=seed)


def _mle(leaves, rho):
    return ef.mul(ef.eq_table(rho), leaves).sum(0) % P


def _sound(out, p, q):
    """The verifier's output is consistent with the leaves (what the caller checks)."""
    if out is None:
        return False
    p0, q0, rho, p_hat, q_hat = out
    num, den = ef.frac_sum(p, q)
    return (ef.equal(ef.mul(p0, den), ef.mul(num, q0)) and ef.equal(p_hat, _mle(p, rho))
            and ef.equal(q_hat, _mle(q, rho)))


@pytest.mark.parametrize("n", range(1, 15))
@pytest.mark.parametrize("fs", [True, False])
def test_honest_transcripts_are_accepted(n, fs):
    g = torch.Generator().manual_seed(n)
    dense = int(torch.randint(1, (1 << n) + 1, (1,), generator=g))
    p, q = _leaves(n, dense, seed=100 + n)
    cp, cv = _pair(fs, seed=n)
    tr = gkr.prove(p, q, cp, "cut/0/")
    assert tr.n == n and tr.flat().shape == (gkr.n_elements(n), ef.D)
    out = gkr.verify(gkr.Transcript.from_bytes(tr.to_bytes(), n), cv, "cut/0/")
    assert _sound(out, p, q)
    if n <= 10:
        ref = gkr.verify_rounds(tr, _pair(fs, seed=n)[1], "cut/0/")
        assert all(torch.equal(x, y) for x, y in zip(out, ref))


@pytest.mark.parametrize("n", [1, 2, 3, 5])
def test_every_tampered_element_is_caught(n):
    p, q = _leaves(n, (1 << n) - 1, seed=7)
    tr = gkr.prove(p, q, _pair(True)[0], "cut/0/")
    flat = tr.flat()
    for i in range(flat.shape[0]):
        for coef in (0, 5):
            bad = flat.clone()
            bad[i, coef] = (bad[i, coef] + 1) % P
            forged = gkr.Transcript.from_bytes(bytes([n]) + ef.pack(bad), n)
            assert not _sound(gkr.verify(forged, _pair(True)[1], "cut/0/"), p, q), (i, coef)


def test_a_wrong_sum_cannot_be_proved_by_the_honest_algorithm_on_other_leaves():
    # the prover commits to leaves whose sum differs from the caller's: the final claims expose it
    p, q = _leaves(6, 50, seed=3)
    q2 = q.clone()
    q2[17] = ef.add(q2[17], ef.const(1))
    tr = gkr.prove(p, q2, _pair(True)[0], "cut/0/")
    out = gkr.verify(tr, _pair(True)[1], "cut/0/")
    assert out is not None and _sound(out, p, q2) and not _sound(out, p, q)


def test_labels_separate_instances():
    p, q = _leaves(4, 16, seed=5)
    tr = gkr.prove(p, q, _pair(True)[0], "cut/0/")
    assert gkr.verify(tr, _pair(True)[1], "cut/1/") is None


def test_transcript_bytes():
    p, q = _leaves(5, 20, seed=9)
    tr = gkr.prove(p, q, _pair(True)[0], "cut/0/")
    blob = tr.to_bytes()
    assert len(blob) == 1 + field_size(8 * gkr.n_elements(5))       # 31 B per element, in groups of 4
    assert gkr.n_elements(27) == 1161
    back = gkr.Transcript.from_bytes(blob, 5)
    assert torch.equal(back.flat(), tr.flat())
    with pytest.raises(ValueError):
        gkr.Transcript.from_bytes(blob, 6)
    with pytest.raises(ClaimCodecError):
        gkr.Transcript.from_bytes(blob[:-1], 5)
    big = tr.flat().clone()
    big[3, 2] = P                                     # 31 bits, but not a field element
    with pytest.raises(ClaimCodecError):
        gkr.Transcript.from_bytes(bytes([5]) + ef.pack(big), 5)

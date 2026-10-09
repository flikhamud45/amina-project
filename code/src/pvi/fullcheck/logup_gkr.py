"""The fractional-sum GKR of V1's logUp argument (``V1_SPEC.md`` Sec. 2.4.1; the toy reference
``sp2027_research/v1_design/scratch/spec_toy.py``).

An *instance* has ``2^n`` leaves, each a fraction ``p/q`` over ``F = F_{p^8}`` (:mod:`extfield`): ``(1, alpha - v)``
for a real leaf of value ``v``, ``(0, 1)`` for padding.  The fraction tree combines children ``2y, 2y + 1`` into
``(p_L q_R + p_R q_L, q_L q_R)``; the root is the sum of all leaves' fractions.  The prover sends the root's two
children and then, layer by layer, a sum-check that reduces a claim on layer ``k`` (``P + lambda Q`` at a point)
to a claim on layer ``k + 1``, binding the point's bits least significant first; the verifier checks each round
and ends with claims ``(P_hat, Q_hat)`` on the leaf layer's multilinear extensions at a random point ``rho``
(column bits first, as the caller lays the leaves out), which the caller checks against its own data.

Rounds send ``g(0), g(2), g(3)``; ``g(1) = c - g(0)``.  Every message is absorbed before the next challenge (a
:class:`protocol.Challenger`), interactive or Fiat--Shamir.  ``prove`` and ``verify`` here are the eager reference
(dense torch tensors); a Triton prover must produce identical transcripts.

**Soundness** (standard): a wrong root survives a layer's sum-check with probability at most ``3k/|F|`` over its
``k`` degree-3 rounds plus ``1/|F|`` for ``lambda`` and ``mu``: ``eps(n) <= (1 + sum_{k<n} (3k + 2)) / p^8``.
"""

from __future__ import annotations

import struct
from dataclasses import dataclass, field

import torch

from . import extfield as ef
from .field import P

__all__ = ["Transcript", "fraction_layers", "prove", "verify", "verify_rounds", "n_elements"]


def n_elements(n: int) -> int:
    """Field elements in an instance's transcript: 4 at the top, then per layer ``k < n`` its ``k`` rounds of 3 and
    4 child values."""
    return 4 + sum(3 * k + 4 for k in range(1, n))


@dataclass
class Transcript:
    """One instance: ``top`` (``[4, 8]``: ``p_1(0), p_1(1), q_1(0), q_1(1)``), per layer ``k = 1 .. n - 1`` its rounds
    ``[k, 3, 8]`` and the four child values ``[4, 8]``."""
    n: int
    top: torch.Tensor
    rounds: list = field(default_factory=list)
    vals: list = field(default_factory=list)
    point: torch.Tensor | None = None          # the prover's own copy of the final point rho (not sent)

    def flat(self) -> torch.Tensor:
        parts = [self.top.reshape(-1, ef.D)]
        for r, v in zip(self.rounds, self.vals):
            parts += [r.reshape(-1, ef.D), v.reshape(-1, ef.D)]
        return torch.cat(parts)

    def to_bytes(self) -> bytes:
        return struct.pack("<B", self.n) + ef.pack(self.flat())

    @classmethod
    def from_bytes(cls, buf, n: int) -> "Transcript":
        """Raises ``claimcodec.ClaimCodecError`` (malformed) or ``ValueError`` (another ``n``)."""
        if len(buf) < 1 or buf[0] != n:
            raise ValueError("an instance of another size")
        flat = ef.unpack(bytes(buf[1:]), n_elements(n))
        top, at = flat[:4], 4
        tr = cls(n, top)
        for k in range(1, n):
            tr.rounds.append(flat[at:at + 3 * k].view(k, 3, ef.D))
            at += 3 * k
            tr.vals.append(flat[at:at + 4])
            at += 4
        return tr


def fraction_layers(p: torch.Tensor, q: torch.Tensor) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """The fraction tree's layers from the leaves ``p, q`` ``[2^n, 8]``: ``layers[k]`` holds ``2^k`` nodes."""
    layers = [(p, q)]
    while layers[-1][0].shape[0] > 1:
        a, b = layers[-1]
        layers.append((ef.add(ef.mul(a[0::2], b[1::2]), ef.mul(a[1::2], b[0::2])), ef.mul(b[0::2], b[1::2])))
    return layers[::-1]


def _absorb(ch, label: str, v: torch.Tensor) -> None:
    ch.absorb(label.encode(), v.reshape(-1).to(torch.int64).contiguous().cpu().numpy().tobytes())


def _at(a: torch.Tensor, t: int) -> torch.Tensor:
    """The pairs ``(a[2y], a[2y + 1])`` at ``t``: ``a[2y] + t (a[2y + 1] - a[2y])``."""
    return ef.add(a[0::2], ef.scal(ef.sub(a[1::2], a[0::2]), t))


def _fold(a: torch.Tensor, r: torch.Tensor) -> torch.Tensor:
    return ef.add(a[0::2], ef.mul(ef.sub(a[1::2], a[0::2]), r))


def prove(p: torch.Tensor, q: torch.Tensor, ch, label: str) -> Transcript:
    """The prover's transcript for the leaves ``p, q`` ``[2^n, 8]`` (``n >= 1``), absorbing into ``ch``."""
    layers = fraction_layers(p, q)
    n = len(layers) - 1
    top = torch.stack([layers[1][0][0], layers[1][0][1], layers[1][1][0], layers[1][1][1]])
    tr = Transcript(n, top)
    _absorb(ch, label + "top", top)
    rho = [ch.ext(label + "mu0")[0]]
    for k in range(1, n):
        lam = ch.ext(label + f"lam{k}")[0]
        a, b = layers[k + 1]
        pl, pr, ql, qr = a[0::2], a[1::2], b[0::2], b[1::2]
        e = ef.eq_table(torch.stack(rho))
        rounds, rs = [], []
        for j in range(k):
            g = []
            for t in (0, 2, 3):
                E, PL, PR, QL, QR = (_at(x, t) for x in (e, pl, pr, ql, qr))
                term = ef.add(ef.add(ef.mul(PL, QR), ef.mul(PR, QL)), ef.mul(lam, ef.mul(QL, QR)))
                g.append(ef.mul(E, term).sum(0) % P)
            g = torch.stack(g)
            _absorb(ch, label + f"g{k}.{j}", g)
            rounds.append(g)
            r = ch.ext(label + f"r{k}.{j}")[0]
            rs.append(r)
            e, pl, pr, ql, qr = (_fold(x, r) for x in (e, pl, pr, ql, qr))
        vals = torch.stack([pl[0], pr[0], ql[0], qr[0]])
        _absorb(ch, label + f"v{k}", vals)
        tr.rounds.append(torch.stack(rounds))
        tr.vals.append(vals)
        rho = [ch.ext(label + f"mu{k}")[0]] + rs
    tr.point = torch.stack(rho)
    return tr


def _lagrange_basis(r: torch.Tensor) -> tuple[torch.Tensor, ...]:
    """``l_0(r) .. l_3(r)``, the Lagrange basis on ``{0, 1, 2, 3}`` at the points ``r`` ``[..., 8]``."""
    one = ef.const(1, r.device)
    m1, m2, m3 = ef.sub(r, one), ef.sub(r, ef.const(2, r.device)), ef.sub(r, ef.const(3, r.device))
    inv2, inv6 = pow(2, P - 2, P), pow(6, P - 2, P)
    return (ef.scal(ef.mul(ef.mul(m1, m2), m3), P - inv6), ef.scal(ef.mul(ef.mul(r, m2), m3), inv2),
            ef.scal(ef.mul(ef.mul(r, m1), m3), P - inv2), ef.scal(ef.mul(ef.mul(r, m1), m2), inv6))


def verify(tr: Transcript, ch, label: str):
    """``(p0, q0, rho [n, 8], P_hat, Q_hat)``: the root fraction and the claims on the leaf layer's multilinear
    extensions at ``rho`` (bit ``i`` of a leaf index against ``rho[i]``), or ``None`` if a round fails.

    The challenges depend on the messages only, so they are drawn first, in transcript order; the checks are then
    vectorised over the ``n - 1`` layers (``V1_SPEC.md`` Sec. 2.4.1, implementation note): within a layer the
    claim after round ``j`` is affine in the claim before it, ``c' = l_1(r) c + (l_0(r) - l_1(r)) g(0) + l_2(r) g(2)
    + l_3(r) g(3)``, and each layer starts from the previous layer's child values, so every layer's chain runs at
    once, ``n - 1`` steps of batched products (layers shorter than ``n - 1`` rounds are padded with identities)."""
    n = tr.n
    if n < 1 or tuple(tr.top.shape) != (4, ef.D) or len(tr.rounds) != n - 1 or len(tr.vals) != n - 1:
        return None
    for k in range(1, n):
        if tuple(tr.rounds[k - 1].shape) != (k, 3, ef.D) or tuple(tr.vals[k - 1].shape) != (4, ef.D):
            return None
    _absorb(ch, label + "top", tr.top)
    mus, lams, rs = [ch.ext(label + "mu0")[0]], [], []
    for k in range(1, n):
        lams.append(ch.ext(label + f"lam{k}")[0])
        for j in range(k):
            _absorb(ch, label + f"g{k}.{j}", tr.rounds[k - 1][j])
            rs.append(ch.ext(label + f"r{k}.{j}")[0])
        _absorb(ch, label + f"v{k}", tr.vals[k - 1])
        mus.append(ch.ext(label + f"mu{k}")[0])

    p10, p11, q10, q11 = tr.top
    p0, q0 = ef.add(ef.mul(p10, q11), ef.mul(p11, q10)), ef.mul(q10, q11)
    mu = torch.stack(mus)                                           # [n, 8]
    if n == 1:
        return p0, q0, mu, ef.add(p10, ef.mul(mu[0], ef.sub(p11, p10))), ef.add(q10, ef.mul(mu[0], ef.sub(q11, q10)))
    L = n - 1
    one = ef.const(1)
    vals = torch.stack(tr.vals)                                     # [L, 4, 8]: layer k at k - 1
    parents = torch.cat([tr.top[None], vals[:-1]])                 # the values layer k's claim comes from
    pc = ef.add(parents[:, 0], ef.mul(mu[:-1], ef.sub(parents[:, 1], parents[:, 0])))
    qc = ef.add(parents[:, 2], ef.mul(mu[:-1], ef.sub(parents[:, 3], parents[:, 2])))
    lam = torch.stack(lams)                                         # [L, 8]
    claim = ef.add(pc, ef.mul(lam, qc))

    # padded [L, L] grids: layer k's round j at (k - 1, j); its point rho_k = (mu_{k-1}, r_{k-1, 0 .. k-2})
    r = one.repeat(L, L, 1)
    g = torch.zeros(L, L, 3, ef.D, dtype=torch.int64)
    rho = one.repeat(L, L, 1)
    live = torch.zeros(L, L, dtype=torch.bool)
    at = 0
    for k in range(1, n):
        r[k - 1, :k] = torch.stack(rs[at:at + k])
        g[k - 1, :k] = tr.rounds[k - 1]
        live[k - 1, :k] = True
        rho[k - 1, 0] = mu[k - 1]
        if k > 1:
            rho[k - 1, 1:k] = r[k - 2, :k - 1]
        at += k
    l0, l1, l2, l3 = _lagrange_basis(r)
    a = torch.where(live[..., None], l1, one)
    b = ef.add(ef.add(ef.mul(ef.sub(l0, l1), g[:, :, 0]), ef.mul(l2, g[:, :, 1])), ef.mul(l3, g[:, :, 2]))
    b = torch.where(live[..., None], b, torch.zeros_like(b))
    term = ef.add(ef.mul(rho, r), ef.mul(ef.sub(one, rho), ef.sub(one, r)))
    term = torch.where(live[..., None], term, one.expand_as(term))
    ev = one.repeat(L, 1)
    for j in range(L):
        claim = ef.add(ef.mul(a[:, j], claim), b[:, j])
        ev = ef.mul(ev, term[:, j])
    pl, pr, ql, qr = vals.unbind(1)
    lhs = ef.mul(ev, ef.add(ef.add(ef.mul(pl, qr), ef.mul(pr, ql)), ef.mul(lam, ef.mul(ql, qr))))
    if not ef.equal(lhs, claim):
        return None
    last = vals[-1]
    p_hat = ef.add(last[0], ef.mul(mu[-1], ef.sub(last[1], last[0])))
    q_hat = ef.add(last[2], ef.mul(mu[-1], ef.sub(last[3], last[2])))
    return p0, q0, torch.cat([mu[-1:], r[-1, :L]]), p_hat, q_hat


def verify_rounds(tr: Transcript, ch, label: str):
    """:func:`verify` round by round, exactly as ``spec_toy.gkr_verify`` (the reference the tests compare it with)."""
    n = tr.n
    p10, p11, q10, q11 = tr.top
    _absorb(ch, label + "top", tr.top)
    p0 = ef.add(ef.mul(p10, q11), ef.mul(p11, q10))
    q0 = ef.mul(q10, q11)
    mu = ch.ext(label + "mu0")[0]
    pc, qc = ef.add(p10, ef.mul(mu, ef.sub(p11, p10))), ef.add(q10, ef.mul(mu, ef.sub(q11, q10)))
    rho = [mu]
    one = ef.const(1)
    for k in range(1, n):
        lam = ch.ext(label + f"lam{k}")[0]
        rounds, vals = tr.rounds[k - 1], tr.vals[k - 1]
        claim, rs = ef.add(pc, ef.mul(lam, qc)), []
        for j in range(k):
            g0, g2, g3 = rounds[j]
            _absorb(ch, label + f"g{k}.{j}", rounds[j])
            r = ch.ext(label + f"r{k}.{j}")[0]
            rs.append(r)
            claim = ef.lagrange4(g0, ef.sub(claim, g0), g2, g3, r)
        pl, pr, ql, qr = vals
        _absorb(ch, label + f"v{k}", vals)
        ev = one
        for ri, si in zip(rs, rho):
            ev = ef.mul(ev, ef.add(ef.mul(si, ri), ef.mul(ef.sub(one, si), ef.sub(one, ri))))
        if not ef.equal(ef.mul(ev, ef.add(ef.add(ef.mul(pl, qr), ef.mul(pr, ql)), ef.mul(lam, ef.mul(ql, qr)))), claim):
            return None
        mu = ch.ext(label + f"mu{k}")[0]
        pc, qc = ef.add(pl, ef.mul(mu, ef.sub(pr, pl))), ef.add(ql, ef.mul(mu, ef.sub(qr, ql)))
        rho = [mu] + rs
    return p0, q0, torch.stack(rho), pc, qc

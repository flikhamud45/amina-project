"""A one-time public proximity proof for a weight commitment (plan item D4, the trust report's Option B).

Theorem 3.2 assumes the commitment ``C_M`` is computed honestly: each committed matrix's columns under its
Merkle root are those of the Reed--Solomon encodings of its rows.  A committer can instead bind a matrix ``E``
that is not a codeword, for example one that encodes model ``A`` on half of the columns and model ``A'`` on the
rest (the *split commitment*): a prover then answers with either model and passes the per-query column check
with probability about ``2**-t`` (``tests/test_setup_proof.py``).

This module removes the assumption.  Once per model, published with ``C_M``, a non-interactive proof (Fiat--Shamir
over the graph's digest and the roots) shows that every committed matrix ``E_j`` (``R_j`` encoded rows of message
length ``m_j`` at ``n_j`` points) is ``e_j``-close to a matrix of codewords, i.e. that it differs from
``Enc(A_j*)`` on at most ``e_j`` columns for a unique ``A_j*`` (``e_j < (n_j - m_j + 1) / 2``):

1. for each tree, ``s`` uniform base-field combinations ``psi_j`` (``s x R_j``) of each member's rows are drawn;
   the committer sends ``psi_j A_j*`` (``s x m_j``), the combinations of the messages;
2. ``t0`` uniform columns of each tree are drawn; the committer opens them (one multiproof per tree);
3. the verifier checks the Merkle paths and, for every member, ``Enc(psi_j A_j*)[c] = psi_j E_j[:, c]``.

**Soundness.**  The ``s`` base-field rows checked at the same columns are one uniform combination over
``K = F_{p^s}`` (Fact 1 of the trust report).  If ``E_j`` is not ``e_j``-close, then by correlated agreement in the
unique-decoding regime (Ben-Sasson et al. 2020, Thm. 1.6; Ligero's Lemma 4.2 for ``e_j < d_j / 4``) the
combination ``psi_j E_j`` is ``e_j``-far from every codeword except with probability ``n_j / p^s``; then any
message's codeword disagrees with it on more than ``e_j`` columns, and ``t0`` distinct uniform columns all miss
them with probability at most ``prod_{i < t0} (n_j - e_j - 1 - i) / (n_j - i)``.  With Fiat--Shamir both terms
are multiplied by the number of random-oracle queries (``2**grinding_bits``).  ``setup_params`` picks ``s`` and
``t0`` so that the sum over the matrices is at most ``2**-lambda``.

**Per query.**  Once every ``E_j`` is ``e_j``-close, the per-query column check of matrix ``j`` fails to catch a
wrong folded row only if all ``t`` columns land among the at most ``m_j - 1 + e_j`` positions where ``Enc(u)``
agrees with ``u``'s committed combination or ``E_j`` is corrupted: probability
``prod_{i<t} (m_j - 1 + e_j - i) / (n_j - i)``, the honest bound with ``m_j`` replaced by ``m_j + e_j``
(:func:`setup_params` computes it).  The committed model is then ``A*`` (evaluated in the field; see the trust report
for the semantics when ``A*`` is not int8).
"""

from __future__ import annotations

import math
import struct
from dataclasses import dataclass

import numpy as np
import torch

from .commitment import (CommitmentPublic, GroupCommitment, GroupPublic, TransposedCommitment, WeightCommitment,
                         codeword_at, column_leaf, group_leaf, verify_multiproof)
from .field import LOG2_P, P, field_matmul_mod, to_field, weight_matmul_mod

__all__ = ["SetupParams", "SetupProof", "survivors", "setup_params", "query_group_columns", "prove_setup",
           "verify_setup"]


@dataclass(frozen=True)
class SetupParams:
    """``s`` base-field rows per combination; per committed matrix its closeness radius ``e``; per tree the
    columns ``t0`` the setup proof opens and the columns ``t`` each query then opens; the security level and
    grinding bits the setup proof was sized for."""
    s: int
    e: tuple[tuple[str, int], ...]
    t0: tuple[tuple[str, int], ...]
    t: tuple[tuple[str, int], ...]
    lam: float
    grinding_bits: int = 64

    def e_of(self, matrix: str) -> int:
        return dict(self.e)[matrix]

    def t0_of(self, group: str) -> int:
        return dict(self.t0)[group]


@dataclass
class SetupProof:
    """The combinations ``{matrix: psi A* (int64, s x m)}`` and per tree ``(column indices, the opened columns
    [sum of its members' rows, t0] (int64), multiproof)``."""
    messages: dict[str, torch.Tensor]
    openings: dict[str, tuple[torch.Tensor, torch.Tensor, list[bytes]]]

    def nbytes(self) -> int:
        """Field elements at 4 bytes (31 bits on the wire), hashes at 32; the column indices follow from the
        transcript and are not counted."""
        fields = sum(m.numel() for m in self.messages.values()) + sum(o.numel() for _, o, _ in self.openings.values())
        return 4 * fields + 32 * sum(len(p) for _, _, p in self.openings.values())


def survivors(a: int, n: int, bits: float) -> int:
    """The smallest ``t`` with ``prod_{i<t} (a - i) / (n - i) <= 2**-bits``: how many distinct uniform columns of
    ``n`` must be opened so that all of them land in a fixed set of ``a`` columns with probability at most
    ``2**-bits`` (at most ``a + 1``, when the product reaches 0).  One pass, linear in ``t``."""
    acc = 0.0
    for t in range(1, n + 1):
        if a - (t - 1) <= 0:
            return t
        acc += math.log2((a - (t - 1)) / (n - (t - 1)))
        if acc <= -bits:
            return t
    return n


FRACTIONS = tuple(2.0 ** -j for j in range(0, 11))
"""The radii tried, as fractions of the unique-decoding radius ``(n - m) / 2``."""


def setup_params(publics: dict[str, CommitmentPublic], groups: dict[str, GroupPublic], lam: float,
                 query_bits: float, *, amortise: int = 1000, grinding_bits: int = 64) -> SetupParams:
    """Parameters of the setup proof and of the queries after it.

    The setup proof's ``2 J`` terms (``J`` matrices) are each at most ``2**-beta``, ``beta = lam + grinding_bits +
    log2(2 J)``, so ``2**grinding_bits`` random-oracle queries succeed with probability at most ``2**-lam``: ``s``
    with ``n / p**s <= 2**-beta``, and ``t0`` columns with ``prod_{i<t0} (n - e - 1 - i) / (n - i) <= 2**-beta``.
    A query's column term per matrix is ``prod_{i<t} (m + e - 1 - i) / (n - i) <= 2**-query_bits`` (the
    per-op budget of :func:`protocol.params_for`).  A larger radius ``e`` makes the setup proof shorter and every
    query longer: each tree takes the radius ``e_j = floor(f (n - m_j) / 2)`` (``f`` in :data:`FRACTIONS`, the
    same for its members) that minimises ``t0 / amortise + t``, its columns' bytes over ``amortise`` queries."""
    beta = lam + grinding_bits + math.log2(2 * max(1, len(publics)))
    n_max = max(g.n_points for g in groups.values())
    s = math.ceil((beta + math.log2(n_max)) / LOG2_P)          # n / p**s <= 2**-beta
    e, t0, tq = {}, {}, {}
    for name, g in groups.items():
        best = None
        for f in FRACTIONS:
            radius = {m: int(f * (publics[m].n_points - publics[m].row_length) / 2) for m in g.members}
            if any(2 * radius[m] >= publics[m].n_points - publics[m].row_length + 1 for m in g.members):
                continue
            a0 = max(survivors(publics[m].n_points - radius[m] - 1, publics[m].n_points, beta) for m in g.members)
            a1 = max(survivors(publics[m].row_length + radius[m] - 1, publics[m].n_points, query_bits)
                     for m in g.members)
            cost = a0 / amortise + a1
            if best is None or cost < best[0]:
                best = (cost, radius, a0, a1)
        _, radius, t0[name], tq[name] = best
        e.update(radius)
    return SetupParams(s, tuple(sorted(e.items())), tuple(sorted(t0.items())), tuple(sorted(tq.items())), lam,
                       grinding_bits)


def query_group_columns(params: SetupParams, groups: dict[str, GroupPublic]) -> tuple[tuple[str, int], ...]:
    """The per-query ``(group, t)`` once the setup proof certified the radii, in the groups' order (as
    ``SecurityParams.group_columns``)."""
    t = dict(params.t)
    return tuple((name, t[name]) for name in groups)


def _transcript(graph_digest: bytes, publics: dict[str, CommitmentPublic], groups: dict[str, GroupPublic],
                params: SetupParams):
    from .protocol import Challenger
    ch = Challenger(fiat_shamir=True)
    ch.absorb(b"setup/params", struct.pack("<IdI", params.s, params.lam, params.grinding_bits))
    ch.absorb(b"setup/graph", graph_digest)
    for name, g in groups.items():
        blob = struct.pack("<H", len(g.tag)) + g.tag + g.root + struct.pack("<QQQ", g.n_points, params.t0_of(name),
                                                                           len(g.members))
        for m in g.members:
            pub = publics[m]
            blob += (struct.pack("<H", len(pub.tag)) + pub.tag
                     + struct.pack("<QQQQ", pub.n_rows, pub.row_length, pub.n_points, params.e_of(m)))
        ch.absorb(b"setup/group/" + name.encode(), blob)
    return ch


def _challenges_rows(ch, publics, groups, params) -> dict[str, torch.Tensor]:
    return {m: ch.folding("setup/" + m, publics[m].n_rows, params.s) for g in groups.values() for m in g.members}


def _absorb_messages(ch, groups, messages) -> None:
    for g in groups.values():
        for m in g.members:
            msg = messages.get(m)
            ch.absorb(b"setup/msg/" + m.encode(), b"" if msg is None else
                      struct.pack("<II", *msg.shape) + msg.to(torch.int64).contiguous().numpy().tobytes())


def _combine(member: WeightCommitment | TransposedCommitment, psi: torch.Tensor, device) -> torch.Tensor:
    """``psi A*`` for an honest member: its rows' combinations (``s x row_length``), on ``device``."""
    if isinstance(member, WeightCommitment):
        return member.fold(psi, device)
    k = member.weights[0].shape[1]
    parts = []
    for i, w in enumerate(member.weights):          # the rows of A' are W_i^T (and the biases' row): psi A' per op
        part = weight_matmul_mod(w.to(device), psi[:, :k].T.contiguous().to(device)).T
        if member.biases is not None:
            b = to_field(member.biases[i].to(device))
            part = (part + field_matmul_mod(psi[:, k:k + 1].to(device), b[None, :])) % P
        parts.append(part)
    return torch.cat(parts, 1).cpu()


def prove_setup(groups: dict[str, GroupCommitment], publics: dict[str, CommitmentPublic], graph_digest: bytes,
                params: SetupParams, device="cpu") -> SetupProof:
    """The committer's setup proof for honest group commitments (``GraphCommitment.groups``)."""
    gp = {name: g.public for name, g in groups.items()}
    ch = _transcript(graph_digest, publics, gp, params)
    psis = _challenges_rows(ch, publics, gp, params)
    messages = {m: _combine(c, psis[m], device) for g in groups.values() for m, c in g.members.items()}
    _absorb_messages(ch, gp, messages)
    openings = {}
    for name, g in groups.items():
        cols = ch.columns("setup/" + name, g.n_points, params.t0_of(name))
        openings[name] = (cols, *g.open(cols, device))
    return SetupProof(messages, openings)


def verify_setup(publics: dict[str, CommitmentPublic], groups: dict[str, GroupPublic], graph_digest: bytes,
                 params: SetupParams, proof: SetupProof) -> bool:
    """Whether ``proof`` shows every committed matrix of ``groups`` ``e_j``-close (see the module docstring)."""
    ch = _transcript(graph_digest, publics, groups, params)
    psis = _challenges_rows(ch, publics, groups, params)
    msgs = proof.messages
    for g in groups.values():
        for m in g.members:
            x = msgs.get(m)
            if (not torch.is_tensor(x) or x.dtype != torch.int64 or tuple(x.shape) != (params.s, publics[m].row_length)
                    or bool(((x < 0) | (x >= P)).any())):
                return False
    _absorb_messages(ch, groups, msgs)
    for name, g in groups.items():
        cols = ch.columns("setup/" + name, g.n_points, params.t0_of(name))
        item = proof.openings.get(name)
        if item is None or len(item) != 3:
            return False
        sent_cols, opened, mp = item
        rows = sum(publics[m].n_rows for m in g.members)
        if (not torch.equal(torch.as_tensor(sent_cols), cols) or not torch.is_tensor(opened)
                or opened.dtype != torch.int64 or tuple(opened.shape) != (rows, cols.numel())
                or bool(((opened < 0) | (opened >= P)).any())):
            return False
        at, digests = 0, []
        for m in g.members:                          # each member's columns: the code check, then its digests
            pub = publics[m]
            e = opened[at:at + pub.n_rows]
            at += pub.n_rows
            want = field_matmul_mod(psis[m], e)
            if not torch.equal(codeword_at(msgs[m], pub.n_points, cols), want):
                return False
            rows_u32 = e.T.contiguous().numpy().astype("<u4")
            digests.append({int(c): column_leaf(pub.tag, int(c), rows_u32[j]) for j, c in enumerate(cols.tolist())})
        leaves = {int(c): group_leaf(g.tag, int(c), [d[int(c)] for d in digests]) for c in cols.tolist()}
        if not verify_multiproof(g.root, int(math.log2(g.n_points)), leaves, mp):
            return False
    return True

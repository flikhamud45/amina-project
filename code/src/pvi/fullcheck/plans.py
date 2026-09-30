"""Commitment plans for mode C: each weight op's codeword length, the Merkle trees the
matrices share, and the exact number of columns each tree opens.

In the report every matrix ``A_l = [W_l | b_l]`` (``N_l`` rows of length ``k_l``) is encoded at
``n_l = rate * next_pow2(k_l)`` under a tree of its own, and every op opens ``t = ceil(beta /
log2 rate)`` columns, ``beta = lambda + log2(2L)`` (plus the grinding bits under Fiat--Shamir):
each op's column term ``((k_l-1)/n_l)**t <= rate**-t`` and its Freivalds term ``p**-r`` are
both ``<= 2**-beta``, so the union bound over the ``L`` ops is ``2L 2**-beta = 2**-lambda``.
A plan keeps exactly that per-op budget and changes three things:

* **exact t**: a tree opens the smallest ``t_l`` whose exact column error
  :func:`column_error_log2` (``t`` distinct uniform columns all among the at most ``k_l - 1``
  zeros of a nonzero codeword) is ``<= 2**-beta`` -- the same bound the report reaches through
  ``(k-1)/n <= 1/rate``, which the power-of-two padding makes loose;
* **shared trees**: the matrices of one group (one codeword length) are committed under ONE
  Merkle tree whose leaf ``c`` binds column ``c`` of every member, open one set of ``t_g =
  max_l t_l`` indices and send one multiproof.  Each member still gets ``t_g >= t_l`` distinct
  uniform columns of its own codeword, drawn after ``u``; the union bound never needed the
  ops' index sets to be independent;
* **per-op codeword lengths** ``n_l``, chosen by the policy.

Policies (``bench.py --policy``; the plan is public, part of the verifier's key):

* ``paper`` -- no plan: the report's commitment and parameters (the default everywhere);
* ``tight`` -- the report's codeword lengths (``rate * next_pow2(k)``);
* ``cnn<e>`` -- every matrix at ``n = max(2**e, 2 next_pow2(k))``: one length for all but the
  longest rows (a CNN's matrices can then share one tree);
* ``R<R>`` -- rate ``R`` (a power of two) for every matrix but the embedding tables, which keep
  the base rate (their rows are the vocabulary, so a higher rate multiplies the setup, while
  their columns hold only ``d`` entries).

Every policy opens exact ``t`` and shares trees between matrices of one length: the matrices of
a length, ordered by their ``t``, are split into the runs whose groups minimise the expected
proof bytes -- ``4 t_g sum N`` of columns plus the multiproof -- at ``REFERENCE_LAMBDA``
(interactive).  A tall matrix then opens no extra columns for a group's larger ``t`` (a
decoder's groups are one per length and ``t``), while matrices of few rows share one multiproof
(a CNN's are one or a few trees).  The commitment is made once for every ``lambda``; at another
``lambda`` a group opens its members' largest ``t`` for that ``lambda``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from functools import cached_property, lru_cache

from .analytic import expected_multiproof_nodes
from .field import TWO_ADICITY

__all__ = ["MAX_N", "REFERENCE_LAMBDA", "PlannedOp", "CommitmentPlan", "column_error_log2", "exact_columns",
           "column_bits", "next_pow2", "plan_commitment"]

MAX_N = 1 << TWO_ADICITY
"""The longest codeword the field's NTT supports."""
REFERENCE_LAMBDA = 128
"""The security level (interactive) at which a plan splits equal-length groups by ``t``."""


def next_pow2(x: int) -> int:
    return 1 << max(0, (x - 1).bit_length())


def column_bits(lam: float, n_checks: int, fiat_shamir: bool = False, grinding_bits: int = 64) -> float:
    """``beta``: every op's Freivalds and column terms are each ``<= 2**-beta``, so the union bound
    over ``n_checks`` ops is ``<= 2**-lam`` (``2**-(lam + grinding_bits)`` under Fiat--Shamir)."""
    return lam + math.log2(2 * max(1, n_checks)) + (grinding_bits if fiat_shamir else 0)


def column_error_log2(k: int, n: int, t: int) -> float:
    """``log2`` of the probability that ``t`` distinct uniform columns of a length-``n``
    Reed--Solomon code of dimension ``k`` all miss a nonzero codeword: ``prod_{i<t} (k-1-i) /
    (n-i) <= ((k-1)/n)**t`` (``-inf`` for ``k = 1``, and practically ``-inf`` once ``t >= k``)."""
    t = min(t, n)
    return sum(math.log2(max(k - 1 - i, 1e-300) / (n - i)) for i in range(t)) if k > 1 else -math.inf


@lru_cache(maxsize=1 << 14)
def exact_columns(k: int, n: int, bits: float) -> int:
    """The smallest ``t >= 1`` with ``column_error_log2(k, n, t) <= -bits`` (at most ``k``: ``k``
    distinct points pin a polynomial of degree ``< k``, so no nonzero codeword survives them).

    The search starts at the ``t`` of the looser bound ``((k-1)/n)**t`` and steps with
    :func:`column_error_log2` itself (strictly decreasing in ``t``), so the result is exactly the
    smallest ``t`` that :func:`protocol.soundness_bits` accepts."""
    if k <= 1:
        return 1
    t = min(k, max(1, math.ceil(bits / -math.log2((k - 1) / n))))
    while t < k and column_error_log2(k, n, t) > -bits:
        t += 1
    while t > 1 and column_error_log2(k, n, t - 1) <= -bits:
        t -= 1
    return t


@dataclass(frozen=True)
class PlannedOp:
    name: str
    n_rows: int
    row_length: int
    n_points: int


@dataclass(frozen=True)
class CommitmentPlan:
    """Public: each weight op's codeword length, and the groups, each with its members in the
    order their column digests enter the group's leaves."""

    policy: str
    ops: tuple[PlannedOp, ...]
    groups: tuple[tuple[str, tuple[str, ...]], ...]

    @cached_property
    def _by_name(self) -> dict[str, PlannedOp]:
        return {o.name: o for o in self.ops}

    def op(self, name: str) -> PlannedOp:
        return self._by_name[name]

    def group_columns(self, bits: float) -> tuple[tuple[str, int], ...]:
        """``(group, t_g)`` for column bits ``beta``: ``t_g`` is its members' largest exact ``t``."""
        return tuple((g, max(exact_columns(o.row_length, o.n_points, bits) for o in map(self.op, members)))
                     for g, members in self.groups)

    def op_columns(self, group_columns) -> list[int]:
        """The columns each op opens (its group's ``t``), in op order."""
        t = dict(group_columns)
        of = {m: g for g, members in self.groups for m in members}
        return [t[of[o.name]] for o in self.ops]

    def shapes(self) -> list[tuple[int, int]]:
        """``(k, n)`` per op, in op order (as :func:`protocol.soundness_bits` takes them)."""
        return [(o.row_length, o.n_points) for o in self.ops]


def _lengths(ops, policy: str, rate: int) -> dict[str, int]:
    """Each op's codeword length under ``policy`` (one spelling per plan: no leading zeros)."""
    m = re.fullmatch(r"(tight)|cnn([1-9][0-9]*)|R([1-9][0-9]*)", policy)
    if m is None:
        raise ValueError(f"unknown commitment policy {policy!r}: expected paper, tight, cnn<e> or R<rate> "
                         "(decimal, no leading zero)")
    if m.group(1):
        n_of = {op.name: rate * next_pow2(op.row_length) for op in ops}
    elif m.group(2):
        e = int(m.group(2))
        if not 1 <= e <= TWO_ADICITY:
            raise ValueError(f"policy {policy!r}: 2**{e} is not a codeword length of this field")
        n_of = {op.name: max(1 << e, 2 * next_pow2(op.row_length)) for op in ops}
    else:
        high = int(m.group(3))
        if high < 2 or high & (high - 1):
            raise ValueError(f"policy {policy!r}: the rate must be a power of two >= 2")
        n_of = {op.name: (rate if op.layout == "embed" else high) * next_pow2(op.row_length) for op in ops}
    for op in ops:
        if not op.row_length <= n_of[op.name] <= MAX_N:
            raise ValueError(f"policy {policy!r}: {op.name} (row length {op.row_length}) gets codeword length "
                             f"{n_of[op.name]}")
    return n_of


def _runs(n: int, ts: list[int], rows: list[int]) -> list[tuple[int, int]]:
    """The split of the distinct column counts ``ts`` (ascending; ``rows[j]``: the rows of the
    matrices with ``ts[j]``) of one length ``n`` into runs ``[a, b)`` of least expected bytes, a
    run's group opening ``ts[b-1]`` columns of all its rows with one multiproof (dynamic
    programming over the run ends)."""
    best, cut = [0.0] + [math.inf] * len(ts), [0] * (len(ts) + 1)
    for b in range(1, len(ts) + 1):
        for a in range(b):
            cost = best[a] + 4 * ts[b - 1] * sum(rows[a:b]) + 32 * expected_multiproof_nodes(n, ts[b - 1])
            if cost < best[b]:
                best[b], cut[b] = cost, a
    out, b = [], len(ts)
    while b:
        out.append((cut[b], b))
        b = cut[b]
    return out[::-1]


def plan_commitment(ops, policy: str, *, rate: int = 4, model_ops=None) -> CommitmentPlan | None:
    """The plan of ``policy`` for weight ops ``ops`` (objects with ``name``, ``n_rows``,
    ``row_length`` and ``layout``: a graph's ``MatOp``s or ``analytic.decoder_shapes``), in
    their order; ``None`` for ``"paper"``.  ``rate`` is the base rate (the report's, 4).

    ``model_ops``: the whole model's ops when ``ops`` are those of a build of a few of its
    decoder blocks.  The runs are then chosen on the whole model (at its op count), so the
    build's groups are the whole model's restricted to the built ops, and its costs extrapolate
    over blocks."""
    if policy == "paper":
        return None
    ops = list(ops)
    model = ops if model_ops is None else list(model_ops)
    bits = column_bits(REFERENCE_LAMBDA, len(model))
    n_of, n_model = _lengths(ops, policy, rate), _lengths(model, policy, rate)
    runs = {}                        # (n, t) -> the (n, smallest t) of its run
    for n in dict.fromkeys(n_model.values()):
        members = [op for op in model if n_model[op.name] == n]
        t_of = [exact_columns(op.row_length, n, bits) for op in members]
        ts = sorted(set(t_of))
        rows = [sum(op.n_rows for op, t_ in zip(members, t_of) if t_ == t) for t in ts]
        for a, b in _runs(n, ts, rows):
            runs.update(((n, t), (n, ts[a])) for t in ts[a:b])
    classes: dict[tuple, list[str]] = {}
    for op in ops:                   # groups in the order of their first member, members in op order
        key = (n_of[op.name], exact_columns(op.row_length, n_of[op.name], bits))
        if key not in runs:
            raise ValueError(f"{op.name}: (codeword length, t) = {key} is not a shape of the model")
        classes.setdefault(runs[key], []).append(op.name)
    groups = tuple((f"g{i}_n{key[0]}", tuple(names)) for i, (key, names) in enumerate(classes.items()))
    return CommitmentPlan(policy, tuple(PlannedOp(op.name, op.n_rows, op.row_length, n_of[op.name]) for op in ops),
                          groups)

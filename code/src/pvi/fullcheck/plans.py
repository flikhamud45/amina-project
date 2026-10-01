"""Commitment plans for mode C: each weight op's committed matrix and its layout, its codeword
length, the Merkle trees the matrices share, and the exact number of columns each tree opens.

In the report every matrix ``A_l = [W_l | b_l]`` (``N_l`` rows of length ``k_l``) is encoded at
``n_l = rate * next_pow2(k_l)`` under a tree of its own, and every op opens ``t = ceil(beta /
log2 rate)`` columns, ``beta = lambda + log2(2L)`` (plus the grinding bits under Fiat--Shamir):
each op's column term ``((k_l-1)/n_l)**t <= rate**-t`` and its Freivalds term ``p**-r`` are
both ``<= 2**-beta``, so the union bound over the ``L`` ops is ``2L 2**-beta = 2**-lambda``.
A plan keeps exactly that per-op budget and changes four things:

* **exact t**: a tree opens the smallest ``t`` whose exact column error
  :func:`column_error_log2` (``t`` distinct uniform columns all among the at most ``m - 1``
  zeros of a nonzero codeword of message length ``m``) is ``<= 2**-beta`` -- the same bound the
  report reaches through ``(k-1)/n <= 1/rate``, which the power-of-two padding makes loose;
* **shared trees**: the matrices of one group (one codeword length) are committed under ONE
  Merkle tree whose leaf ``c`` binds column ``c`` of every member, open one set of ``t_g =
  max t`` indices and send one multiproof.  Each member still gets ``t_g`` distinct uniform
  columns of its own codeword, drawn after the messages its check follows; the union bound never
  needed the matrices' index sets to be independent;
* **per-op codeword lengths** ``n``, chosen by the policy;
* **per-op layouts** (the policies with the suffix ``c``), below.

Layouts of a weight op ``Z = A [X ; 1]`` (``X``: ``k' x M``, one column per pixel or token):

* ``row`` (the report's): the rows of ``A`` are encoded (message length ``k``), a leaf holds a
  column of length ``N``.  The prover sends ``u = chi A`` (``chi``: ``r x N``) and the verifier
  checks ``chi Z == u [X ; 1]`` (Freivalds) and ``chi . E[:, c] == Enc(u)[c]``: ``r k + t N``
  field elements.  Its error: ``p**-r + prod_{i<t} (k-1-i)/(n-i)``.
* ``col``: the rows of ``A^T`` (one per input coordinate, of length ``N``) are encoded, a leaf
  holds a column of length ``k``.  After the claims the verifier draws ``chi'`` (``r x M``; the
  identity when ``M <= r``) and computes ``z_i = Z chi'_i^T`` and ``w_i = [X ; 1] chi'_i^T``
  itself; the prover opens ``t`` columns ``E'[:, c]`` and the verifier checks ``w_i . E'[:, c] ==
  Enc(z_i)[c]``: ``t k`` field elements, no ``u``.  Linear ops reading one tensor (q/k/v,
  gate/up) with one row length may be committed as ONE col matrix ``[A_1 ; A_2 ; ...]^T`` (their
  rows stacked: message length ``sum N``), checked with one ``chi'``, one ``w`` and one set of
  columns of length ``k``.  Its error: ``p**-r + prod_{i<t} (N-1-i)/(n-i)`` with ``N`` the
  stacked rows (no Freivalds term when ``chi' = I``).  Proof: take a wrong claim of the matrix's
  ops, whose input ``X`` the verifier derived from correct claims, so ``Delta = Z - A [X ; 1] !=
  0``.  ``Delta chi'^T = 0`` with probability ``<= p**-r`` (never for ``chi' = I``); otherwise
  some ``z_i != A w_i``, and ``Enc(z_i)`` and ``w_i^T E' = Enc(A w_i)`` are distinct codewords of
  dimension ``N``, which agree on at most ``N - 1`` of the ``n`` positions; the opened columns are
  those of ``E'`` (the Merkle root binds them) and are drawn after the claims, independently of
  ``chi'``, so all ``t`` of them fall where the two agree with probability at most the product.
* ``lookup`` (embedding ops without a bias: ``W`` ``[d, V]`` is the table, ``X`` the token ids, and
  the claim at id ``j`` is ``W[:, j]``): the table is committed as a Merkle tree over its ``V`` rows
  (int8, padded to a power of two), no code.  The claims are the looked-up rows, one column of ``d``
  int8 values per position, and the prover opens the tree at the distinct ids with ONE multiproof,
  sent with the claims.  The verifier checks that every id names a row of the table, that the claims
  are int8 (so a leaf's bytes are the claim, one to one), that repeated ids carry equal columns, and
  the multiproof: no challenge, no ``u``, no columns, ``d M`` bytes of claims instead of ``4 d M``.
  Its error is 0: accepting any other claim means other bytes for a leaf the root binds, a SHA-256
  collision (the assumption of every Merkle tree here, not a term of the statistical bound).  The
  tree binds ``W`` alone, so an embedding with a bias (claims ``W[:, j] + b``) keeps the row layout.

Policies (``bench.py --policy``; the plan is public, part of the verifier's key):

* ``paper`` -- no plan: the report's commitment and parameters (the default everywhere);
* ``tight`` -- the report's codeword lengths (``rate * next_pow2(m)`` for message length ``m``);
* ``cnn<e>`` -- every matrix at ``n = max(2**e, 2 next_pow2(m))``: one length for all but the
  longest messages (a CNN's matrices can then share one tree);
* ``R<R>`` -- rate ``R`` (a power of two) for every matrix but the embedding tables, which keep
  the base rate (their rows are the vocabulary, so a higher rate multiplies the setup, while
  their columns hold only ``d`` entries);
* the same with the suffix ``c`` (``tightc``, ``cnn18c``, ``R64c``, ...): every embedding table
  without a bias takes the lookup layout, and a linear op may take the col layout, alone or with the
  other linear ops that read its input with its row length in one col matrix.  A lookup costs less
  than the table's row layout: it drops ``u`` (``4 r V`` bytes) and the columns (``4 t d`` per
  tree), and its claims take one byte each (four as int32, about one in ``PVC3``), for one
  multiproof of at most one path (``32 log2 V`` bytes) per position -- less than the three bytes
  per claim it saves on int32 claims for every benchmark decoder (``d >= 512``), and less than ``u``
  alone with wire.
  The linear ops of a set (of one input and row length) take one of three options: every op in
  the row layout, every op transposed on its own, or all in one col matrix; sets of equal shapes
  (the blocks of a decoder) take the same.  The options are those of least expected proof bytes of
  the whole plan at ``REFERENCE_LAMBDA`` (interactive): ``u`` (row: ``4 r k``), the opened
  columns (row: ``4 t N``, col: ``4 t k``) and the multiproofs of the shared trees they end up
  in, found by descent from the base policy's plan, so a ``c`` plan never costs more than its base
  policy's in that model.  Convolutions keep the row layout (a convolution's ``w`` would need the
  unfolded input, and it opens no fewer bytes transposed: its ``N`` is small against its ``k``).
* ``auto`` -- one of :data:`AUTO_CANDIDATES` (:func:`auto_policy`): on the whole model's shapes, at
  ``REFERENCE_LAMBDA`` (interactive), the candidate of fewest expected non-claim bytes
  (:func:`plan_overhead`: ``u``, opened columns, the encoded trees' multiproofs) whose setup
  (``analytic.setup_size``'s encoded entries) is within a budget, by default ``max(2 x the paper's
  setup, 2**34)``; ties go to the first candidate, and with no candidate within the budget the one of
  least setup.  The plan is exactly that candidate's (``plan.policy``, which the Fiat--Shamir
  statement binds through its trees as for the concrete policy), with ``plan.requested = "auto"``.

Every policy opens exact ``t`` and shares trees between the encoded matrices of one length: the
matrices of a length, ordered by their ``t``, are split into the runs whose groups minimise the
expected proof bytes -- ``4 t_g sum n_rows`` of columns plus the multiproof -- at ``REFERENCE_LAMBDA``.
A tall matrix then opens no extra columns for a group's larger ``t`` (a decoder's groups are one
per length and ``t``), while matrices of few rows share one multiproof (a CNN's are one or a
few trees).  Row and col matrices of one length may share a tree: every tree's columns are drawn
in one round, after ``u`` (and ``chi'`` with ``chi``, right after the claims).  A col matrix's
check needs only its ``chi'`` and columns to follow the claims, so drawing its columns after ``u``
too costs nothing interactively; under Fiat--Shamir they then also depend on ``u``, which a
prover could vary to draw them again -- a grinding attack, which the 64 grinding bits bound as
for every other challenge.  The commitment is made once for every ``lambda``; at another
``lambda`` a group opens its members' largest ``t`` for that ``lambda``.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, replace
from functools import cached_property, lru_cache

from .analytic import expected_multiproof_nodes, setup_size
from .commitment import next_pow2
from .field import LOG2_P, TWO_ADICITY

__all__ = ["MAX_N", "REFERENCE_LAMBDA", "AUTO_CANDIDATES", "AUTO_MIN_BUDGET", "PlannedMatrix", "CommitmentPlan",
           "column_error_log2", "exact_columns", "column_bits", "next_pow2", "plan_commitment", "col_name",
           "plan_overhead", "auto_budget", "auto_policy"]

MAX_N = 1 << TWO_ADICITY
"""The longest codeword the field's NTT supports."""
REFERENCE_LAMBDA = 128
"""The security level (interactive) at which a plan chooses layouts and splits equal-length groups by ``t``."""
AUTO_CANDIDATES = ("tightc", "cnn16c", "cnn17c", "cnn18c", "R16c", "R64c")
"""The policies ``auto`` chooses from, in tie-break order (a ``c`` plan never costs more than its base's)."""
AUTO_MIN_BUDGET = 1 << 34
"""The least default setup budget of ``auto`` (encoded entries)."""


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


def col_name(ops) -> str:
    """The name of the col matrix of the weight ops ``ops`` (in their stacking order)."""
    return "+".join(ops) + "^T"


@dataclass(frozen=True)
class PlannedMatrix:
    """One committed matrix, whose ``n_rows`` rows are encoded (message length ``row_length``) at
    ``n_points``: a weight op's ``[W | b]`` (the row layout, named after the op), or with
    ``members`` the transposed stack ``[A_1 ; A_2 ; ...]^T`` of these weight ops (the col layout:
    ``n_rows`` is their row length, ``row_length`` their rows together); or with ``lookup`` an
    embedding op's table ``W`` ``[d, V]``, not encoded: a tree over its ``V`` rows of ``d`` entries
    (``n_rows = d``, ``row_length = V``, ``n_points``: the tree's ``next_pow2(V)`` leaves)."""

    name: str
    n_rows: int
    row_length: int
    n_points: int
    members: tuple[str, ...] = ()
    lookup: bool = False

    @property
    def layout(self) -> str:
        return "lookup" if self.lookup else "col" if self.members else "row"

    @property
    def ops(self) -> tuple[str, ...]:
        """The weight ops this matrix checks."""
        return self.members or (self.name,)


@dataclass(frozen=True)
class CommitmentPlan:
    """Public: the committed matrices (in the order of their first ops), and the groups of the
    encoded ones, each with its matrices in the order their column digests enter the group's
    leaves (a lookup table is a tree of its own)."""

    policy: str
    matrices: tuple[PlannedMatrix, ...]
    groups: tuple[tuple[str, tuple[str, ...]], ...]
    requested: str | None = None    # "auto" when ``policy`` is the concrete policy it chose

    @cached_property
    def _by_name(self) -> dict[str, PlannedMatrix]:
        return {m.name: m for m in self.matrices}

    @cached_property
    def _by_op(self) -> dict[str, PlannedMatrix]:
        return {op: m for m in self.matrices for op in m.ops}

    @cached_property
    def coded(self) -> tuple[PlannedMatrix, ...]:
        """The encoded matrices (row and col layouts), each in one group: those with a column check."""
        return tuple(m for m in self.matrices if not m.lookup)

    @cached_property
    def tables(self) -> tuple[PlannedMatrix, ...]:
        """The lookup tables."""
        return tuple(m for m in self.matrices if m.lookup)

    def matrix(self, name: str) -> PlannedMatrix:
        return self._by_name[name]

    def matrix_of(self, op: str) -> PlannedMatrix:
        """The matrix that checks weight op ``op``."""
        return self._by_op[op]

    def group_columns(self, bits: float) -> tuple[tuple[str, int], ...]:
        """``(group, t_g)`` for column bits ``beta``: ``t_g`` is its members' largest exact ``t``."""
        return tuple((g, max(exact_columns(m.row_length, m.n_points, bits) for m in map(self.matrix, members)))
                     for g, members in self.groups)

    def matrix_columns(self, group_columns) -> list[int]:
        """The columns each encoded matrix opens (its group's ``t``), in matrix order."""
        t = dict(group_columns)
        of = {m: g for g, members in self.groups for m in members}
        return [t[of[m.name]] for m in self.coded]

    def shapes(self) -> list[tuple[int, int]]:
        """``(message length, n)`` per encoded matrix, in matrix order: one check each (as
        :func:`protocol.soundness_bits` takes them in mode C; a lookup table adds no term)."""
        return [(m.row_length, m.n_points) for m in self.coded]


def _policy(policy: str, rate: int):
    """``(length, col)``: the codeword length ``length(m, embed)`` of a message of length ``m``
    (``embed``: an embedding table's rows) under ``policy``, and whether it takes the layouts of the
    suffix ``c`` -- lookup tables, and the col layout for linear ops (one spelling per plan: no
    leading zeros)."""
    m = re.fullmatch(r"(?:(tight)|cnn([1-9][0-9]*)|R([1-9][0-9]*))(c?)", policy)
    if m is None:
        raise ValueError(f"unknown commitment policy {policy!r}: expected paper, auto, tight, cnn<e> or R<rate> "
                         "(decimal, no leading zero), the last three optionally with the suffix c")
    if m.group(1):
        def length(k, embed):
            return rate * next_pow2(k)
    elif m.group(2):
        e = int(m.group(2))
        if not 1 <= e <= TWO_ADICITY:
            raise ValueError(f"policy {policy!r}: 2**{e} is not a codeword length of this field")

        def length(k, embed):
            return max(1 << e, 2 * next_pow2(k))
    else:
        high = int(m.group(3))
        if high < 2 or high & (high - 1):
            raise ValueError(f"policy {policy!r}: the rate must be a power of two >= 2")

        def length(k, embed):
            return (rate if embed else high) * next_pow2(k)
    return length, bool(m.group(4))


def _sets(ops, col: bool) -> list[list]:
    """``ops`` in the sets whose layouts are chosen together, in the order of their first ops: under a
    ``c`` policy the linear ops that read one tensor with one row length (they may share a col
    matrix), every other op alone."""
    sets: dict = {}
    for i, op in enumerate(ops):
        shared = col and op.layout == "linear" and op.inputs[0] is not None
        sets.setdefault((op.inputs[0], op.row_length) if shared else i, []).append(op)
    return list(sets.values())


def _options(members, policy: str, rate: int) -> list[list[PlannedMatrix]]:
    """The ways to commit one set of ops: every op in the row layout (the base policy's), and under a
    ``c`` policy for linear ops every op transposed on its own and, for several, all of them in one
    col matrix (an option whose codeword would be longer than the field's NTT is left out); under a
    ``c`` policy an embedding table without a bias is looked up, its one way."""
    length, col = _policy(policy, rate)
    if col and members[0].layout == "embed" and not any(op.has_bias for op in members):
        return [[PlannedMatrix(op.name, op.n_rows, op.row_length, next_pow2(op.row_length), lookup=True)
                 for op in members]]

    def row(op):
        m = PlannedMatrix(op.name, op.n_rows, op.row_length, length(op.row_length, op.layout == "embed"))
        if not op.row_length <= m.n_points <= MAX_N:
            raise ValueError(f"policy {policy!r}: {op.name} (row length {op.row_length}) gets codeword length "
                             f"{m.n_points}")
        return m

    def stack(ops):
        rows = sum(op.n_rows for op in ops)
        return PlannedMatrix(col_name(op.name for op in ops), ops[0].row_length, rows, length(rows, False),
                             tuple(op.name for op in ops))

    options = [[row(op) for op in members]]
    if col and members[0].layout == "linear":
        options += [[stack([op]) for op in members]] + ([[stack(members)]] if len(members) > 1 else [])
    return [ms for ms in options if all(m.n_points <= MAX_N for m in ms)]


def _signature(members) -> tuple:
    """What a set's options depend on: its ops' shapes and biases, in order."""
    return tuple((op.n_rows, op.row_length, op.layout, op.has_bias) for op in members)


def _runs(n: int, ts: list[int], rows: list[int]) -> tuple[float, list[tuple[int, int]]]:
    """The split of the distinct column counts ``ts`` (ascending; ``rows[j]``: the rows of the
    matrices with ``ts[j]``) of one length ``n`` into runs ``[a, b)`` of least expected bytes, a
    run's group opening ``ts[b-1]`` columns of all its rows with one multiproof (dynamic
    programming over the run ends), and those bytes."""
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
    return best[-1], out[::-1]


def _shapes(matrices, bits: float, weights=None) -> dict[int, dict[int, int]]:
    """``{n: {t: rows}}``: the rows of the encoded matrices of each codeword length and exact ``t``
    (each matrix counted ``weights[i]`` times; lookup tables open no columns)."""
    out: dict = {}
    for i, m in enumerate(matrices):
        if m.lookup:
            continue
        at = out.setdefault(m.n_points, {})
        t = exact_columns(m.row_length, m.n_points, bits)
        at[t] = at.get(t, 0) + m.n_rows * (1 if weights is None else weights[i])
    return out


def _choose(model, policy: str, rate: int, bits: float) -> dict[tuple, int]:
    """The option (:func:`_options`) each signature of the model's sets takes: the plan of least
    expected proof bytes -- ``u``, and the opened columns and multiproofs of the groups
    (:func:`_runs`) -- found by descent over the signatures from the base policy's plan (every
    op in the row layout), each step taking another option of one signature when it lowers the
    whole plan's bytes; so a ``c`` policy never costs more than its base policy in this model."""
    reps = max(1, math.ceil(bits / LOG2_P))
    classes: dict[tuple, list] = {}           # signature -> [options, sets of that signature]
    for members in _sets(model, _policy(policy, rate)[1]):
        classes.setdefault(_signature(members), [_options(members, policy, rate), 0])[1] += 1

    def cost(choice):
        mats, weights = [], []
        for sig, (options, count) in classes.items():
            mats += options[choice[sig]]
            weights += [count] * len(options[choice[sig]])
        u = sum(w * reps * m.row_length for m, w in zip(mats, weights) if m.layout == "row")
        return 4 * u + sum(_runs(n, sorted(at), [at[t] for t in sorted(at)])[0]
                           for n, at in _shapes(mats, bits, weights).items())

    choice = {sig: 0 for sig in classes}
    best, better = cost(choice), True
    while better:
        better = False
        for sig, (options, _) in classes.items():
            for i in range(len(options)):
                trial = {**choice, sig: i}
                if i != choice[sig] and (c := cost(trial)) < best:
                    best, choice, better = c, trial, True
    return choice


def _matrices(ops, policy: str, rate: int, choice: dict) -> list[PlannedMatrix]:
    """The committed matrices of ``ops``, their sets in the options ``choice`` gives their
    signatures, in the order of their first ops."""
    out = []
    for members in _sets(ops, _policy(policy, rate)[1]):
        if _signature(members) not in choice:
            raise ValueError(f"{members[0].name}: its ops' shapes are not a set of the model's")
        out += _options(members, policy, rate)[choice[_signature(members)]]
    order = {op.name: i for i, op in enumerate(ops)}
    return sorted(out, key=lambda m: order[m.ops[0]])


def plan_overhead(model_ops, plan: CommitmentPlan | None, *, rate: int = 4) -> float:
    """The expected non-claim bytes of one query of the whole model ``model_ops`` at
    ``REFERENCE_LAMBDA`` (interactive), as ``analytic.proof_bytes`` counts them: ``u`` of the row-layout
    ops, the opened columns and the encoded trees' multiproofs (``plan``: ``None`` for the report's).
    A ``c`` plan's lookup tables are left out: their multiproofs depend on the prompt, and every ``c``
    policy looks the same tables up."""
    ops = list(model_ops)
    bits = column_bits(REFERENCE_LAMBDA, len(ops))
    reps = max(1, math.ceil(bits / LOG2_P))
    if plan is None:
        columns = max(1, math.ceil(bits / math.log2(rate)))
        trees = [(rate * next_pow2(op.row_length), op.n_rows, columns) for op in ops]     # (n, rows, t)
        u = sum(op.row_length for op in ops)
    else:
        group_t = dict(plan.group_columns(bits))
        trees = [(plan.matrix(ms[0]).n_points, sum(plan.matrix(m).n_rows for m in ms), group_t[g])
                 for g, ms in plan.groups]
        u = sum(m.row_length for m in plan.coded if m.layout == "row")
    return (4 * reps * u + 4 * sum(min(t, n) * rows for n, rows, t in trees)
            + 32 * sum(expected_multiproof_nodes(n, t) for n, _, t in trees))


def auto_budget(model_ops, *, rate: int = 4) -> int:
    """``auto``'s default setup budget: ``max(2 x the paper's encoded entries, AUTO_MIN_BUDGET)``."""
    ops = list(model_ops)
    paper = sum(op.n_rows * rate * next_pow2(op.row_length) for op in ops)
    return max(2 * paper, AUTO_MIN_BUDGET)


def auto_policy(model_ops, *, rate: int = 4, setup_budget: float | None = None) -> str:
    """The policy ``auto`` resolves to for the whole model ``model_ops``: of :data:`AUTO_CANDIDATES`
    (those whose codewords fit the field), the one of least :func:`plan_overhead` among those whose
    encoded entries are ``<= setup_budget`` (default :func:`auto_budget`), the first on a tie; if
    none fits, the one of least setup (the first on a tie)."""
    ops = list(model_ops)
    if not ops:
        return AUTO_CANDIDATES[0]
    budget = auto_budget(ops, rate=rate) if setup_budget is None else setup_budget
    costs = []                                   # (policy, non-claim bytes, encoded entries)
    for policy in AUTO_CANDIDATES:
        try:
            plan = plan_commitment(ops, policy, rate=rate)
        except ValueError:                       # a codeword longer than the field's NTT
            continue
        costs.append((policy, plan_overhead(ops, plan, rate=rate), setup_size(ops, plan=plan)["encoded_entries"]))
    if not costs:
        raise ValueError("policy 'auto': no candidate policy fits this model")
    within = [c for c in costs if c[2] <= budget]
    return (min(within, key=lambda c: c[1]) if within else min(costs, key=lambda c: c[2]))[0]


def plan_commitment(ops, policy: str, *, rate: int = 4, model_ops=None,
                    setup_budget: float | None = None) -> CommitmentPlan | None:
    """The plan of ``policy`` for weight ops ``ops`` (objects with ``name``, ``n_rows``,
    ``row_length``, ``layout``, ``inputs`` and ``has_bias``: a graph's ``MatOp``s or
    ``analytic.decoder_shapes``), in their order; ``None`` for ``"paper"``.  ``rate`` is the base
    rate (the report's, 4).

    ``model_ops``: the whole model's ops when ``ops`` are those of a build of a few of its
    decoder blocks.  The layouts are chosen, and the runs split, on the whole model (at its op
    count), so the build's matrices and groups are the whole model's restricted to the built ops,
    and its costs extrapolate over blocks.

    ``policy="auto"`` takes the plan of :func:`auto_policy` on the whole model (``setup_budget``: its
    budget in encoded entries), recorded as ``plan.policy`` with ``plan.requested = "auto"``."""
    if setup_budget is not None and policy != "auto":
        raise ValueError("setup_budget: only for policy 'auto'")
    if policy == "auto":
        chosen = auto_policy(ops if model_ops is None else model_ops, rate=rate, setup_budget=setup_budget)
        return replace(plan_commitment(ops, chosen, rate=rate, model_ops=model_ops), requested="auto")
    if policy == "paper":
        return None
    ops = list(ops)
    model = ops if model_ops is None else list(model_ops)
    bits = column_bits(REFERENCE_LAMBDA, len(model))
    choice = _choose(model, policy, rate, bits)
    mats, model_mats = _matrices(ops, policy, rate, choice), _matrices(model, policy, rate, choice)
    runs = {}                        # (n, t) -> the (n, smallest t) of its run
    for n, at in _shapes(model_mats, bits).items():
        ts = sorted(at)
        for a, b in _runs(n, ts, [at[t] for t in ts])[1]:
            runs.update(((n, t), (n, ts[a])) for t in ts[a:b])
    classes: dict[tuple, list[str]] = {}
    for m in (m for m in mats if not m.lookup):   # groups in the order of their first member, members in order
        key = (m.n_points, exact_columns(m.row_length, m.n_points, bits))
        if key not in runs:
            raise ValueError(f"{m.name}: (codeword length, t) = {key} is not a shape of the model")
        classes.setdefault(runs[key], []).append(m.name)
    groups = tuple((f"g{i}_n{key[0]}", tuple(names)) for i, (key, names) in enumerate(classes.items()))
    return CommitmentPlan(policy, tuple(mats), groups)

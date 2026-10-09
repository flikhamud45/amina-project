"""The whole-network check: every weight product is verified, nothing is sampled.

Message flow for one query ``x`` (``mode="C"``, the verifier holds only the
32-byte commitment of each weight matrix -- the setting of Anchuri et al.):

1. prover -> verifier: the claimed pre-activations ``Z_l`` of every weight op.
2. verifier: recomputes every cheap op (requant, ReLU, pooling, residual adds,
   norms, softmax, attention, ...) from ``x`` and the claims, which gives it the
   input ``X_l`` of every weight op without trusting the prover for any of them.
   It then draws ``r`` random rows ``chi_l`` per weight op.
3. prover -> verifier: ``u_l = chi_l^T [W_l | b_l]``.
4. verifier: checks ``chi_l^T Z_l == u_l^T [X_l ; 1]`` for every op and every
   column of ``Z_l`` (Freivalds), then draws ``t`` column indices per op (under a
   commitment plan, per shared tree: see below).
5. prover -> verifier: the opened columns and their Merkle paths; the verifier
   checks each against its commitment and against ``Enc(u_l)``.

``mode="K"`` is the setting of SafetyNets, Slalom and Maverick, where the
verifier holds the weights: it computes ``u_l`` itself (fresh ``chi`` per query)
and steps 3 and 5 disappear.  ``mode="Kpre"`` additionally precomputes ``u_l``
for a secret ``chi`` once, so a query costs the verifier only step 2 and the two
inner products; the proof is just the claims (Slalom's and Maverick's
"preprocessing" mode).

Soundness (a false claim accepted), per weight op, is at most ``p**-r``
(Freivalds: a nonzero column of ``Z - A X`` survives a uniformly random row with
probability ``1/p``) plus, in mode C, ``((k-1)/n)**t`` (a wrong ``u`` survives
``t`` distinct columns of a Reed--Solomon code of distance ``n-k+1``), with a
union bound over ops.  A commitment plan (:mod:`pvi.fullcheck.plans`, opt-in) keeps each op's
budget and meets it with the op's exact ``t``, trees shared by matrices of one codeword length
and per-op codeword lengths -- and under its ``c`` policies commits some linear ops transposed
(the col layout, alone or several reading one tensor in one matrix).  A col-layout op sends no
``u`` and is not checked in step 4: step 2 also draws ``chi'`` over its claims' columns, the
verifier computes both sides of its code check itself (``w = chi' [X ; 1]^T`` and ``z = chi'
Z^T``), and step 5 opens columns of its transposed matrix, which must satisfy ``w . E'[:, c] ==
Enc(z)[c]``.  All trees' columns are drawn in step 4, so row and col matrices of one codeword
length share trees.  The ``c`` policies also commit every embedding table without a bias as a
lookup table (a Merkle tree over its rows): its claims are the looked-up rows, sent as int8 in step
1 with one multiproof over the distinct ids, and checked once derive has the ids, before any
challenge -- int8 and ids of the table (in derive), equal rows for equal ids
(``lookup_consistency``), the multiproof (``lookup_merkle``) -- with no ``u`` and no columns.  In
modes K and Kpre a verifier with ``lookups=True`` (opt-in) computes the embedding ops from its own
weights (the rows at the ids, plus the bias of an embedding that has one), and the prover sends no
claims for them.  Two preconditions make the *integer* claim, not just its residue, the thing
that is checked: claims are range-checked to ``|z| < 2**29`` (so two in-range integers with equal
residues are equal, since ``2 * 2**29 < p``), and every weight op is checked at commitment time to
have honest outputs inside that range (:func:`claim_bound`).  Completeness is exact: every
quantity is an integer and both parties compute cheap ops with the same integer rules.
"""

from __future__ import annotations

import hashlib
import math
import secrets
import struct
import time
import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

from . import claimcodec, contention, hostmem
from .commitment import (HASH_BYTES, CommitmentPublic, GroupCommitment, GroupPublic, TableCommitment, TablePublic,
                         TransposedCommitment, WeightCommitment, codeword_at, column_rows, group_leaf, map_threaded,
                         row_leaves, table_leaves, verify_multiproof, verify_multiproofs)
from .field import (_LIMB_MAX, LOG2_P, P, combine_limbs, exact_chunk, exact_gemm_i64, field_matmul_mod,
                    int8_field_matmul, int8_left, int8_ok, int8_right, int8_small_matmul, limbs_f64, min_max,
                    small_matmul_mod, to_field)
from .graph import IntGraph, MatOp
from .pipeline import ClaimUploads, wire_claim, wire_openings, wire_rows
from .plans import CommitmentPlan, column_bits, column_error_log2, plan_commitment

__all__ = [
    "SecurityParams",
    "params_for",
    "soundness_bits",
    "claim_bound",
    "Challenger",
    "Prover",
    "Verifier",
    "run_query",
    "Z_BOUND",
    "GraphCommitment",
    "commit_graph",
]

Z_BOUND = 1 << 29
"""Claims must satisfy ``-2**29 < z < 2**29``.  Because ``2 * 2**29 < p``, two
in-range integers with the same residue mod ``p`` are equal, so checking
``Z = A X`` mod ``p`` pins down the integer claim the verifier then uses."""


@dataclass(frozen=True)
class SecurityParams:
    lam: float
    reps: int
    rate: int
    columns: int
    fiat_shamir: bool = False
    grinding_bits: int = 64
    group_columns: tuple[tuple[str, int], ...] = ()
    """Under a commitment plan, ``(group, t)`` per group: its members' largest exact ``t``
    (``rate`` is then 0 -- each op's is in the plan -- and ``columns`` the largest ``t``)."""

    def columns_for(self, group: str) -> int:
        return dict(self.group_columns)[group]


def params_for(lam: float, n_checks: int, *, rate: int = 4, fiat_shamir: bool = False,
               grinding_bits: int = 64, plan: CommitmentPlan | None = None) -> SecurityParams:
    """Smallest ``(r, t)`` whose union bound over ``n_checks`` ops is ``<= 2**-lam``.

    With Fiat--Shamir the prover can grind offline, so ``grinding_bits`` more are
    required (security against ``2**grinding_bits`` hash evaluations).  Under a commitment
    ``plan`` (:mod:`pvi.fullcheck.plans`) ``r`` is the same, and each group opens its members'
    largest exact ``t`` for the same per-op budget ``2**-beta``.
    """
    bits = column_bits(lam, n_checks, fiat_shamir, grinding_bits)
    reps = max(1, math.ceil(bits / LOG2_P))
    if plan is None:
        columns = max(1, math.ceil(bits / math.log2(rate)))
        return SecurityParams(lam, reps, rate, columns, fiat_shamir, grinding_bits)
    groups = plan.group_columns(bits)
    return SecurityParams(lam, reps, 0, max(t for _, t in groups), fiat_shamir, grinding_bits, groups)


def soundness_bits(params: SecurityParams, shapes: list[tuple[int, int]], mode: str = "C",
                   columns: list[int] | None = None) -> float:
    """``-log2`` of the total soundness error: the sum over the checks of ``p**-r`` (Freivalds) plus,
    in mode C, the column term ``prod_{i<t} (m-1-i)/(n-i)``.  ``shapes`` holds ``(m, n)`` per check --
    each op's ``(k, n)`` for the report's commitment, and one entry per committed matrix of a plan
    (``plan.shapes()``: a row matrix checks its op, message length ``k``; a col matrix every op it
    stacks, message length their rows together) -- and ``columns`` each check's ``t`` (default:
    ``params.columns`` for every one).  Under a plan's parameters mode C needs
    ``columns=plan.matrix_columns(params.group_columns)``: ``params.columns`` is only the largest
    group's ``t``, which would credit every matrix with more columns than its tree opens.  In modes K
    and Kpre every op is a check of its own (Freivalds): pass one shape per op.  A col matrix whose
    claims have ``M <= r`` columns is checked on each of them (``chi' = I``), without the Freivalds
    term this sum still counts for it (an upper bound).  A lookup is no check of chance and adds no
    term: a plan's lookup table (its claims are the rows the root binds; ``plan.shapes()`` leaves the
    tables out), or in modes K and Kpre an embedding op whose rows the verifier reads itself
    (``lookups``: leave its shape out)."""
    if mode == "C" and params.group_columns and columns is None:
        raise ValueError("a commitment plan's matrices open their groups' t: "
                         "pass columns=plan.matrix_columns(params.group_columns)")
    total = 0.0
    for i, (k, n) in enumerate(shapes):
        err = 2.0 ** (-params.reps * LOG2_P)
        if mode == "C":
            err += 2.0 ** column_error_log2(k, n, params.columns if columns is None else columns[i])
        total += err
    bits = -math.log2(total)
    return bits - (params.grinding_bits if params.fiat_shamir else 0)


def claim_bound(op: MatOp) -> int:
    """Largest ``|z|`` an honest execution of ``op`` can produce."""
    w = op.weight.to(torch.int64)
    bound = w.abs().amax(1) if op.layout == "embed" else (op.max_input + 1) * w.abs().sum(1)
    if op.bias is not None:
        bound = bound + op.bias.abs()
    return int(bound.max())


class GraphCommitment(dict):
    """``{op name: WeightCommitment}`` for every weight op; under a commitment plan
    ``{matrix name: its commitment}`` for every encoded matrix (a row-layout op's
    ``WeightCommitment`` under the op's name, a col-layout matrix's ``TransposedCommitment``), the
    ``plan``, its ``groups`` (``{name: GroupCommitment}``), whose trees bind the members' columns,
    and its lookup ``tables`` (``{op name: TableCommitment}``)."""

    def __init__(self, members: dict[str, WeightCommitment | TransposedCommitment], plan: CommitmentPlan | None = None,
                 groups: dict[str, GroupCommitment] | None = None,
                 tables: dict[str, TableCommitment] | None = None) -> None:
        super().__init__(members)
        self.plan = plan
        self.groups = groups or {}
        self.tables = tables or {}

    @property
    def publics(self) -> dict[str, CommitmentPublic]:
        return {name: c.public for name, c in self.items()}

    @property
    def group_publics(self) -> dict[str, GroupPublic]:
        return {name: g.public for name, g in self.groups.items()}

    @property
    def table_publics(self) -> dict[str, TablePublic]:
        return {name: t.public for name, t in self.tables.items()}


def commit_graph(graph: IntGraph, rate: int, device="cpu", *, policy: str = "paper",
                 model_ops=None) -> GraphCommitment:
    """Commit every weight op, after checking honest claims fit the range check: on a tree of its
    own at ``rate`` (``policy="paper"``, the report), or as the commitment plan of ``policy`` lays
    the ops out (:func:`plans.plan_commitment`, with ``rate`` its base rate; ``model_ops``: the
    whole model's op shapes when ``graph`` is a build of a few of its blocks)."""
    for op in graph.mat_ops:
        b = claim_bound(op)
        if b >= Z_BOUND:
            raise ValueError(f"{op.name}: honest claims can reach {b} >= Z_BOUND = {Z_BOUND}")
    plan = plan_commitment(graph.mat_ops, policy, rate=rate, model_ops=model_ops)
    if plan is None:
        return GraphCommitment({op.name: WeightCommitment.build(op.name.encode(), op.weight, op.bias, rate=rate,
                                                                device=device) for op in graph.mat_ops})
    ops = {op.name: op for op in graph.mat_ops}

    def member(m):
        if m.layout == "col":
            return TransposedCommitment.member(m.name.encode(), {o: (ops[o].weight, ops[o].bias) for o in m.members},
                                               m.n_points)
        return WeightCommitment.member(m.name.encode(), ops[m.name].weight, ops[m.name].bias, m.n_points)

    groups = {g: GroupCommitment.build(g.encode(), {m: member(plan.matrix(m)) for m in members}, device=device)
              for g, members in plan.groups}
    members = {m: c for g in groups.values() for m, c in g.members.items()}
    tables = {m.name: TableCommitment.build(m.name.encode(), ops[m.name].weight) for m in plan.tables}
    return GraphCommitment({m.name: members[m.name] for m in plan.coded}, plan, groups, tables)


class Challenger:
    """Verifier randomness.

    Interactive: every challenge is expanded with SHAKE-256 from a fresh 256-bit
    key drawn from the operating system's CSPRNG (``secrets``) at the moment the
    challenge is issued.  In particular the column indices are keyed only after
    the prover has sent ``u``, so nothing the prover saw earlier predicts them.
    Fiat--Shamir: the key is read from the running SHAKE-256 transcript of the statement
    (the graph's digest, parameters, commitments, input) and of every prover message so far:
    each challenge absorbs its label into a copy of the transcript and reads 32 bytes, so the
    challenges are random-oracle outputs of prefix-free inputs without a length-extension argument.
    ``seed`` makes interactive keys reproducible (tests only).
    """

    def __init__(self, *, fiat_shamir: bool = False, seed: int | None = None) -> None:
        self.fiat_shamir = fiat_shamir
        self._state = hashlib.shake_256(b"pvi/fullcheck/v3")
        self._seed = seed
        self._count = 0

    def absorb(self, label: bytes, blob: bytes) -> None:
        """Length-prefixed, labelled absorption (an injective transcript encoding)."""
        self._state.update(len(label).to_bytes(2, "big") + label + len(blob).to_bytes(8, "big"))
        self._state.update(blob)

    def _key(self, label: str) -> bytes:
        if self.fiat_shamir:
            h = self._state.copy()
            h.update(b"challenge/" + label.encode())
            return h.digest(32)
        if self._seed is None:
            return secrets.token_bytes(32)
        self._count += 1
        return hashlib.sha256(f"test-seed/{self._seed}/{self._count}/{label}".encode()).digest()

    @staticmethod
    def _stream(key: bytes, label: str, n_bytes: int) -> bytes:
        return hashlib.shake_256(key + label.encode()).digest(n_bytes)

    def folding(self, name: str, n_rows: int, reps: int) -> torch.Tensor:
        """``reps x n_rows`` uniform field elements (rejection sampling, no bias)."""
        n = reps * n_rows
        key = self._key("fold/" + name)
        parts, have, block = [], 0, 0
        while have < n:
            want = int((n - have) * 1.08) + 64
            words = np.frombuffer(self._stream(key, f"fold/{name}/{block}", 4 * want), dtype="<u4")
            words = words.astype(np.int64)
            words = words[words < 2 * P] % P
            parts.append(words)
            have += words.size
            block += 1
        return torch.from_numpy(np.concatenate(parts)[:n].reshape(reps, n_rows).copy())

    def columns(self, name: str, n_points: int, t: int) -> torch.Tensor:
        """``t`` distinct uniform indices in ``[0, n_points)``, sorted."""
        t = min(t, n_points)
        key = self._key("cols/" + name)
        limit = (1 << 64) - ((1 << 64) % n_points)
        chosen: set[int] = set()
        block = 0
        while len(chosen) < t:
            vals = np.frombuffer(self._stream(key, f"cols/{name}/{block}", 8 * (4 * t + 64)), dtype="<u8")
            for v in vals.tolist():
                if v < limit:
                    chosen.add(v % n_points)
                    if len(chosen) == t:
                        break
            block += 1
        return torch.tensor(sorted(chosen), dtype=torch.int64)


def _tensor_blob(t: torch.Tensor) -> bytes:
    shape = struct.pack(f"<{t.dim()}Q", *t.shape)
    return struct.pack("<B", t.dim()) + shape + t.detach().to("cpu", torch.int64).contiguous().numpy().tobytes()


class Prover:
    def __init__(self, graph: IntGraph, *, device="cpu", commitments: dict[str, WeightCommitment] | None = None,
                 lean: bool = False):
        self.graph = graph
        self.device = torch.device(device)
        self.commitments = commitments or {}
        self.lean = lean  # free dead activations and stream each claim to the host during the forward pass
        self._ops = {op.name: op for op in graph.mat_ops}
        self._looked_up: dict[str, list[int]] = {}    # a plan's lookup tables: the ids of the last query

    def claims(self, x: torch.Tensor, *, send=None, to_host: bool = True, **forward_kwargs) -> dict[str, torch.Tensor]:
        """The claims of query ``x``, on the host (``to_host=False``: where the prover keeps them, its
        device, or the host in lean mode).  ``send(z)`` (see ``IntGraph.forward``), if given, is
        applied to each claim on the prover's device as it is computed, and its results are returned
        as they are.  The ids each lookup table of a commitment plan looks up are kept for
        :meth:`open_tables`."""
        if self.lean:
            forward_kwargs = dict(forward_kwargs, free=True, claims_device="cpu")
        tables = getattr(self.commitments, "tables", {})
        self._looked_up = {}
        if tables:
            def watch(op, xin):
                if op.name in tables:
                    self._looked_up[op.name] = xin.reshape(-1).tolist()
            forward_kwargs = dict(forward_kwargs, watch=watch)
        _, claims = self.graph.forward(x.to(self.device), send=send, **forward_kwargs)
        return claims if send is not None or not to_host else {k: v.to("cpu") for k, v in claims.items()}

    def _weight(self, name: str) -> torch.Tensor:
        # the int8 weights the forward pass already keeps on the device
        return self._ops[name]._weights_on(self.device)[0]

    def open_tables(self) -> dict[str, list[bytes]]:
        """The multiproof of every lookup table of a commitment plan for the rows the last
        :meth:`claims` looked up (sent with the claims)."""
        return {name: t.open(self._looked_up[name]) for name, t in getattr(self.commitments, "tables", {}).items()}

    def fold(self, chis: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {name: self.commitments[name].fold(chi, self.device, weight=self._weight(name))
                for name, chi in chis.items()}

    def open(self, cols: dict[str, torch.Tensor]):
        """The opened columns and multiproof of every tree: each weight op's, or under a commitment
        plan each group's (its members' columns stacked in member order), computed from the weights
        on the prover's device."""
        groups = getattr(self.commitments, "groups", None)
        if groups:
            plan = self.commitments.plan
            return {g: groups[g].open(idx, self.device, weights=[self._weights_of(plan.matrix(m))
                                                                 for m in groups[g].members])
                    for g, idx in cols.items()}
        return {name: self.commitments[name].open(idx, self.device, weight=self._weight(name))
                for name, idx in cols.items()}

    def _weights_of(self, matrix):
        """The device weights of a committed matrix: its op's, or each of a col-layout matrix's ops'."""
        return [self._weight(o) for o in matrix.members] if matrix.layout == "col" else self._weight(matrix.name)


_RHS_CHUNK = 8192
"""Contraction block of the right-hand sides: that of ``small_matmul_mod``, which the
straightforward ``u^T X`` uses (int8-sized ``X`` times field elements: every block sum is
below ``2**53``), so the products below are exact on exactly the same inputs."""
_IM2COL_MAX = 1 << 16
"""Convolutions whose unfolded input has at most this many entries use im2col and one GEMM
(fewest calls); larger ones the unfold-free shifted GEMM (less memory traffic)."""
_BATCH_BYTES = 1 << 25
"""Budget (float64 operand bytes) of one batched product in the verifier's checks."""


def _conv_rhs(op: MatOp, u: torch.Tensor, xin: torch.Tensor) -> torch.Tensor:
    """``u^T unfold(x)`` without unfolding: ONE GEMM of the ``r k^2`` filter rows of ``u``
    with the padded input (``[r k^2, C] @ [C, B Hp Wp]``), then ``k^2`` shifted strided
    slices are added.  Each output is the sum of the same ``C k^2`` integer terms as in
    the unfolded product, exact in float64 when ``C k^2 <= _RHS_CHUNK``."""
    kk, s, p = op.conv
    b, c, h, w = xin.shape
    r = u.shape[0]
    ho, wo = (h + 2 * p - kk) // s + 1, (w + 2 * p - kk) // s + 1
    filt = u[:, :c * kk * kk].to(torch.float64).reshape(r, c, kk, kk).permute(0, 2, 3, 1).reshape(r * kk * kk, c)
    xf = torch.nn.functional.pad(xin.to(torch.float64), (p, p, p, p))
    hp, wp = h + 2 * p, w + 2 * p
    y = (filt @ xf.transpose(0, 1).reshape(c, b * hp * wp)).view(r, kk, kk, b, hp, wp)
    acc = y[:, 0, 0, :, :s * (ho - 1) + 1:s, :s * (wo - 1) + 1:s].clone()
    for dy in range(kk):
        for dx in range(kk):
            if dy or dx:
                acc += y[:, dy, dx, :, dy:dy + s * (ho - 1) + 1:s, dx:dx + s * (wo - 1) + 1:s]
    return torch.remainder(acc.reshape(r, b * ho * wo).to(torch.int64), P)


def _rhs(op: MatOp, u: torch.Tensor, xin: torch.Tensor) -> torch.Tensor:
    """``u^T [X ; 1]`` over the field, shape ``[r, M]``.

    Linear layers multiply ``u`` with the ``[M, K]`` input as it is (no int64 unfold, no
    transposed copies); convolutions with ``C k^2 <= _RHS_CHUNK`` use im2col (small) or
    :func:`_conv_rhs`.  Every product keeps the ``_RHS_CHUNK`` blocks of the
    straightforward product, so both are exact on the same inputs.
    """
    k = op.n_in
    if op.layout == "embed":
        out = u[:, xin.reshape(-1)]
    elif op.layout == "linear":
        out = torch.remainder(exact_gemm_i64(u[:, :k].to(torch.float64), xin.reshape(-1, k).T, _RHS_CHUNK), P)
    elif k > _RHS_CHUNK:
        x = op.unfold(xin)  # [K, M] small ints
        out = small_matmul_mod(x.T.contiguous(), u[:, :k].T.contiguous()).T
    elif k * op.n_cols(xin) <= _IM2COL_MAX:
        kk, s, p = op.conv
        cols = torch.nn.functional.unfold(xin.to(torch.float64), kk, padding=p, stride=s)   # [B, K, L]
        y = (u[:, :k].to(torch.float64) @ cols).transpose(0, 1).reshape(u.shape[0], -1)
        out = torch.remainder(y.to(torch.int64), P)
    else:
        out = _conv_rhs(op, u, xin)
    if op.has_bias:
        out = (out + u[:, k:k + 1]) % P
    return out


def _in_bounds(a: torch.Tensor, lo: int, hi: int) -> bool:
    """``lo <= a <= hi`` everywhere (a claim's range check: ``-Z_BOUND < z < Z_BOUND`` without abs(),
    which overflows on INT64_MIN)."""
    if a.numel() == 0:
        return True
    amin, amax = min_max(a)
    return amin >= lo and amax <= hi


def _in_field(a: torch.Tensor) -> bool:
    """``0 <= a < P`` everywhere."""
    if a.numel() == 0:
        return True
    lo, hi = min_max(a)
    return lo >= 0 and hi < P


def _stamp(t: torch.Tensor) -> tuple:
    """Identifies ``t`` as it is now, without keeping it alive: a weak reference and the
    version counter that every in-place change of ``t`` (or of a view of it) increments.

    A write that bypasses the counter (through ``.numpy()``, ``.data`` or DLPack) is not
    seen.  What is stamped is the verifier's own state: the claims it received, which must not
    change while one query is checked in any case (derive's inputs were computed from them; a
    verifier fed over a wire deserialises them into tensors nobody else holds), and in mode
    Kpre its secret ``chi`` and ``u``."""
    return weakref.ref(t), t._version


def _same(stamp: tuple, t: torch.Tensor) -> bool:
    """``t`` is the very tensor ``stamp`` was taken of, and it has not been modified since."""
    return stamp[0]() is t and stamp[1] == t._version


def _leading_passes(ops: list, check) -> tuple[int, Exception | None]:
    """How many leading ``ops`` pass ``check``, and the exception the next one raised (if any)."""
    for i, op in enumerate(ops):
        try:
            if not check(op):
                return i, None
        except Exception as exc:   # raised by the caller once the ops before it are decided
            return i, exc
    return len(ops), None


# The checks do not copy a verdict back per op (on a GPU each copy waits for the device): every
# field check and comparison leaves a boolean where it was computed, and all of them come back
# with ONE copy once the work is queued (on the CPU that copy costs nothing).  Only derive's
# range checks of claims are read at once on the CPU, where a rejected claim then skips the
# rest of derive.

def _defer(device) -> bool:
    """Whether the checks on ``device`` keep their work there: derive leaves the range checks of
    claims for the one copy, the field checks of ``u`` stay on the device, and the column code
    checks run on opened columns uploaded as the int32 rows the leaves hash.  Everywhere but on
    the CPU (the tests run these forms on the CPU too)."""
    return torch.device(device).type != "cpu"


def _outside(a: torch.Tensor, lo: int, hi: int) -> torch.Tensor:
    """Some entry of ``a`` outside ``[lo, hi]``, as a device boolean (``False`` if ``a`` is empty)."""
    if a.numel() == 0:
        return torch.zeros((), dtype=torch.bool, device=a.device)
    amin, amax = torch.aminmax(a)
    return (amin < lo) | (amax > hi)


def _check_bounds(a: torch.Tensor, lo: int, hi: int, pending: list) -> bool:
    """``lo <= a <= hi`` everywhere (:func:`_in_bounds`), or where the checks defer (:func:`_defer`)
    ``True``, with the verdict left in ``pending`` (:func:`_outside`)."""
    if not _defer(a.device):
        return _in_bounds(a, lo, hi)
    if a.numel():
        pending.append(_outside(a, lo, hi))
    return True


def _disagrees(checks: list[tuple]) -> torch.Tensor:
    """Some ``(u, lhs, rhs)`` of ``checks`` (one or more ops, on one device) with ``u`` not in the
    field or ``lhs = chi^T Z != rhs = u^T [X ; 1]``: their Freivalds checks as ONE boolean, left
    where it was computed.  On a device a few calls check all the ops at once; on the CPU, where
    a value costs nothing to read, they are checked one by one (numpy's min/max and
    ``torch.equal``, the fastest there) up to the first failure."""
    if not _defer(checks[0][1].device):
        return torch.tensor(not all(_in_field(u) and torch.equal(lhs, rhs) for u, lhs, rhs in checks))
    lhs = torch.cat([left.reshape(-1) for _, left, _ in checks])
    bad = (lhs != torch.cat([right.reshape(-1) for _, _, right in checks])).any()
    bounds = [torch.aminmax(u) for u, _, _ in checks if u.numel()]
    if bounds:
        bad |= (torch.stack([lo for lo, _ in bounds]) < 0).any() | (torch.stack([hi for _, hi in bounds]) >= P).any()
    return bad


def _unit_codes(units: list, failed: list[bool]) -> list[bool]:
    """Per tree of ``units``, whether the code check of every member passed (``failed``: one flag
    per member op, in unit order)."""
    out, i = [], 0
    for _, members in units:
        out.append(not any(failed[i:i + len(members)]))
        i += len(members)
    return out


def _columns_verdict(codes: list[bool], merkle: list, n_ok: int, exc: Exception | None, n_trees: int) -> str | None:
    """The column check's verdict, tree by tree -- shape, code, Merkle -- for the ``n_ok`` leading
    trees (weight ops, or groups) with well-formed openings (``exc``: what the next one raised):
    ``codes[i]`` and ``merkle[i] = (ok, exception)`` of tree ``i``."""
    for i in range(n_ok):
        if not codes[i]:
            return "columns_code"
        ok, err = merkle[i]
        if err is not None:
            raise err
        if not ok:
            return "columns_merkle"
    if exc is not None:
        raise exc
    return "columns_shape" if n_ok < n_trees else None


def _to_host(flags: list[torch.Tensor]) -> list[bool]:
    """Device booleans (0-d or 1-d, on one device) as a flat list, with ONE device-to-host copy."""
    return torch.cat([f.reshape(-1) for f in flags]).tolist() if flags else []


_NOT_INT8 = object()     # the streaming verifier: a weight op's input was not int8-valued, verify again


def _to_device(t: torch.Tensor, device) -> torch.Tensor:
    """``t`` on ``device``; a host tensor bound for a GPU goes through pinned memory, so the copy
    does not wait for the device."""
    if torch.device(device).type != "cuda" or t.device.type != "cpu":
        return t.to(device)
    return hostmem.pinned(t).to(device, non_blocking=True)


def _host_copy(t: torch.Tensor) -> torch.Tensor:
    """``t`` on the host: from a CUDA device a non-blocking copy into pinned memory, which holds ``t``
    once the device has run what was queued before it (after any later copy back that waits)."""
    if t.device.type != "cuda":
        return t
    return hostmem.empty(t.shape, t.dtype).copy_(t, non_blocking=True)


def _upload_rows(rows: dict[str, np.ndarray], device) -> dict[str, torch.Tensor]:
    """Host int32 arrays on ``device``: per shape, one staging buffer (pinned) and one
    non-blocking copy (a copy from pageable memory would wait for the device each time)."""
    by_shape: dict = {}
    for name, a in rows.items():
        by_shape.setdefault(a.shape, []).append(name)
    out = {}
    for shape, names in by_shape.items():
        staged = hostmem.empty((len(names), *shape), torch.int32, pin=torch.cuda.is_available())
        for j, name in enumerate(names):
            staged[j].numpy()[...] = rows[name]
        out.update(zip(names, staged.to(device, non_blocking=True)))
    return out


@dataclass
class Verifier:
    graph: IntGraph
    params: SecurityParams
    mode: str = "C"
    publics: dict[str, CommitmentPublic] = field(default_factory=dict)
    weights: dict[str, tuple[torch.Tensor, torch.Tensor | None]] = field(default_factory=dict)
    lean: bool = False       # free every recomputed tensor once no later op reads it
    device: str = "cpu"      # "cuda": a client on a GPU, the _gpuv cells (Merkle hashing stays on the CPU)
    stream: bool = False     # run_query checks with verify_streaming (the same verdicts and labels)
    groups: dict[str, GroupPublic] = field(default_factory=dict)   # a commitment plan's trees (mode C)
    tables: dict[str, TablePublic] = field(default_factory=dict)   # a plan's lookup tables (mode C; kept in graph order)
    lookups: bool = False    # modes K and Kpre: the verifier reads the embedding rows itself (none are sent)
    _pre: dict = field(default_factory=dict)
    _wdev: dict = field(default_factory=dict)   # modes K and Kpre on a GPU: the weights they read stay there
    _ranged: dict = field(default_factory=dict)  # name -> (weakref, _version) of the claim derive() range-checked
    _kept: dict = field(default_factory=dict)    # Kpre: stacked operands of the fixed chi and u
    _consts: list = field(default_factory=list)  # a GPU client: the cheap ops' constants (source, copy, versions)
    _columns: dict = field(default_factory=dict)  # query shape -> claim_columns
    _digest: bytes = b""                          # the public graph's digest, once (Fiat--Shamir absorbs it)

    def __post_init__(self) -> None:
        if self.groups or self.tables:
            self._check_plan()
            # one order of the tables, the graph's, for their rows on the wire, the transcript and the checks
            self.tables = {op.name: self.tables[op.name] for op in self.graph.mat_ops if op.name in self.tables}
        if self.lookups and self.mode == "C":
            raise ValueError("lookups=True is for modes K and Kpre, whose verifier holds the weights "
                             "(mode C looks rows up in a commitment plan's tables: its c policies)")
        if self.params.fiat_shamir:
            self._digest = self.graph.digest()
        if torch.device(self.device).type != "cpu":   # the cheap ops' constants live on the device
            self.graph, pairs = self.graph.with_constants_on(self.device)
            self._consts = [(src, dst, (src._version, dst._version)) for src, dst in pairs]

    def _check_plan(self) -> None:
        """A key of groups and tables (a commitment plan) fits the graph and the parameters: mode C;
        every weight op checked by exactly one committed matrix of its shape -- its own (named after
        it: the row layout), a col-layout matrix of linear ops that read one tensor with one row
        length, or for an embedding op a lookup table of its ``d`` and ``V``; every encoded matrix in
        exactly one group of its codeword length; and a column count for every group.  A table binds
        the rows of ``W`` alone: an embedding op with a bias cannot be one."""
        if self.mode != "C":
            raise ValueError(f"commitment groups are a mode-C key, not one of mode {self.mode}")
        ops = {op.name: op for op in self.graph.mat_ops}
        reads = _input_bindings(self.graph)
        checked = []
        for name, pub in self.publics.items():
            of = [ops.get(o) for o in pub.members or (name,)]
            if None in of:
                raise ValueError(f"committed matrix {name} names no weight op of this graph")
            if pub.members and (any(op.layout != "linear" for op in of) or len({reads[op.name] for op in of}) != 1
                                or len({(op.n_in, op.has_bias) for op in of}) != 1):
                raise ValueError(f"col matrix {name}: its ops must be linear ops reading one tensor, of one row length")
            shape = (of[0].row_length, sum(op.n_rows for op in of)) if pub.members else (of[0].n_rows,
                                                                                         of[0].row_length)
            if (pub.n_rows, pub.row_length) != shape:
                raise ValueError(f"committed matrix {name} does not have the shape of its ops")
            checked += [op.name for op in of]
        for name, table in self.tables.items():
            op = ops.get(name)
            if op is None or op.layout != "embed":
                raise ValueError(f"lookup table {name} names no embedding op of this graph")
            if (table.n_rows, table.n_tokens) != (op.n_rows, op.n_in):
                raise ValueError(f"lookup table {name} does not have the shape of its op")
            if op.has_bias:
                raise ValueError(f"lookup table {name}: its op has a bias, which a tree over the table's rows "
                                 "does not bind")
            checked.append(name)
        if sorted(checked) != sorted(ops):
            raise ValueError("every weight op must be checked by exactly one committed matrix or table")
        members = [m for g in self.groups.values() for m in g.members]
        if sorted(members) != sorted(self.publics):
            raise ValueError("every committed matrix must be in exactly one commitment group")
        columns = dict(self.params.group_columns)
        for name, g in self.groups.items():
            if any(self.publics[m].n_points != g.n_points for m in g.members):
                raise ValueError(f"group {name} holds a matrix of another codeword length")
            if not 1 <= columns.get(name, 0) <= g.n_points:
                raise ValueError(f"the parameters give group {name} no column count: use params_for(..., plan=)")

    def _refresh_constants(self) -> None:
        """Copy again every device constant whose source (or copy) was modified since."""
        for i, (src, dst, versions) in enumerate(self._consts):
            if (src._version, dst._version) != versions:
                dst.copy_(src)
                self._consts[i] = (src, dst, (src._version, dst._version))

    def precompute(self, challenger: Challenger) -> None:
        """Mode Kpre: fix a secret ``chi`` per op and precompute ``u``."""
        for op in self._row_ops():
            chi = challenger.folding(op.name, op.n_rows, self.params.reps).to(self.device)
            self._pre[op.name] = (chi, self._fold_local(op, chi))
        self._wdev.clear()   # Kpre never folds again: no device copy of the model (but the tables of lookups)
        _sync(self.device)   # the precompute is timed by its callers

    def _weights_on(self, op: MatOp, device) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Modes K and Kpre: op's weights on ``device`` (uploaded once, not on every query)."""
        w, b = self.weights[op.name]
        if torch.device(device).type != "cpu":
            key = (op.name, str(device))
            if key not in self._wdev:
                self._wdev[key] = (w.to(device), None if b is None else b.to(device))
            w, b = self._wdev[key]
        return w, b

    def _fold_local(self, op: MatOp, chi: torch.Tensor) -> torch.Tensor:
        w, b = self._weights_on(op, chi.device)
        u = small_matmul_mod(w.T.to(torch.int64).contiguous(), chi.T.contiguous()).T
        if b is not None:
            u = torch.cat([u, field_matmul_mod(chi, to_field(b)[:, None])], 1)
        return u

    def _col_matrices(self) -> dict[str, list[MatOp]]:
        """A commitment plan's col-layout matrices, each with the weight ops it stacks (in order)."""
        if not self.groups:
            return {}
        ops = {op.name: op for op in self.graph.mat_ops}
        return {name: [ops[o] for o in pub.members] for name, pub in self.publics.items() if pub.members}

    def _row_ops(self) -> list[MatOp]:
        """The weight ops checked with Freivalds (step 4): every one but a plan's col-layout ops and
        lookup tables, and the embedding ops whose rows a verifier with ``lookups`` reads itself."""
        skip = {op.name for members in self._col_matrices().values() for op in members}
        skip |= set(self.tables) | {op.name for op in self._own_rows()}
        return [op for op in self.graph.mat_ops if op.name not in skip]

    def _own_rows(self) -> list[MatOp]:
        """Modes K and Kpre with ``lookups``: the embedding ops, whose claims the verifier computes from
        its own weights (the prover sends none)."""
        return [op for op in self.graph.mat_ops if op.layout == "embed"] if self.lookups else []

    def _sent_ops(self) -> list[MatOp]:
        """The weight ops whose claims the prover sends: all but a verifier's own rows."""
        own = {op.name for op in self._own_rows()}
        return [op for op in self.graph.mat_ops if op.name not in own]

    def _own_claim(self, op: MatOp, xin: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        """``W[:, ids] + b`` of embedding op ``op`` (``W[:, ids]`` without a bias) from the verifier's
        own weights, as ``dtype`` (the ids clamped to the table: derive rejects a query with any other)."""
        w, b = self._weights_on(op, xin.device)
        z = w[:, xin.reshape(-1).clamp(0, op.n_in - 1)].to(dtype)
        return z if b is None else z + b[:, None].to(dtype)

    def fold_challenges(self, ch: Challenger, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Step 2's challenges, drawn after the claims: ``r`` rows ``chi`` over the outputs of every op
        checked with Freivalds, and for every col-layout matrix ``chi'`` over its claims' ``M``
        columns -- ``r`` rows, or the identity when ``M <= r`` (each column is then checked as it is)."""
        reps = self.params.reps
        chis = {op.name: ch.folding(op.name, op.n_rows, reps) for op in self._row_ops()}
        stacked = self._col_matrices()
        if stacked:
            m_of = dict(zip((op.name for op in self.graph.mat_ops), self.claim_columns(x) or ()))
            for name, members in stacked.items():
                m = m_of.get(members[0].name)
                if m is not None:          # (else ``x`` is malformed, and derive rejects the query)
                    chis[name] = torch.eye(m, dtype=torch.int64) if m <= reps else ch.folding(name, m, reps)
        return chis

    def claim_columns(self, x: torch.Tensor) -> list[int] | None:
        """Every weight op's column count on the query ``x`` (``None``: ``x`` is malformed), from the
        graph's shapes alone (:meth:`IntGraph.claim_columns`), once per query shape."""
        key = (tuple(x.shape), x.dtype)
        if key not in self._columns:
            self._columns[key] = self.graph.claim_columns(x)
        return self._columns[key]

    def check_tokens(self, x: torch.Tensor, claims: dict) -> bool:
        """The token rule of a graph with its logits at ``n = meta["lm_positions"] > 1`` positions
        (:func:`transformer.with_lm_positions`): its queries are a prompt and the ``n - 1`` tokens
        generated after it, so each of the first ``n - 1`` columns of the claimed logits must have its
        first largest entry at the next input token.  The claims are then checked as any others: a
        wrong token needs wrong logits.  Claims of another shape pass here (:meth:`derive` rejects
        them)."""
        n = self.graph.meta.get("lm_positions", 1)
        if n <= 1:
            return True
        head = next(op for op in self.graph.mat_ops if op.output == self.graph.output_name)
        z = claims.get(head.name)
        if not torch.is_tensor(z) or x.dim() != 2 or x.shape[1] < n or tuple(z.shape) != (head.n_rows, x.shape[0] * n):
            return True
        chosen = z.reshape(head.n_rows, x.shape[0], n)[:, :, :-1].argmax(0)          # [B, n - 1]
        return torch.equal(chosen.cpu(), x[:, x.shape[1] - n + 1:].cpu().to(chosen.dtype))

    def derive(self, x: torch.Tensor, claims: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        """Recompute every cheap op; return each weight op's input, or ``None`` to reject.

        The range checks of claims on a GPU are deferred: they come back with one copy at
        the end instead of one per weight op.  A lookup table's claims must be int8, and every op
        that looks rows up (a table, or the verifier's own rows) must look up ids of its table."""
        out = self._derive(x, claims)
        if out is None or any(_to_host(out[1])):
            return None
        self._ranged = out[2]
        return out[0]

    def _derive(self, x: torch.Tensor, claims, *, dtype: torch.dtype = torch.int64, visit=None):
        """The derive loop: ``None`` to reject, else ``(inputs, pending, ranged)``.

        ``claims.get(name)`` gives each claim, of ``dtype``.  A claim on the CPU is range-checked
        at once; a claim on a device leaves its (deferred) verdict in ``pending``, and the ops
        after it run on whatever it holds: every cheap op is a total function of int64
        tensors, so they only compute values that are never used, and an exception there is
        reported as the rejection it follows.  ``visit(op, z, x)``, if given, is called for
        each weight op once its claim has the right shape; the op's input is then not kept
        (``inputs`` stays empty) and every tensor is freed after its last reader.
        """
        env = {self.graph.input_name: x}
        inputs: dict[str, torch.Tensor] = {}
        pending: list[torch.Tensor] = []
        ranged: dict[str, tuple] = {}
        self._ranged = {}
        self._refresh_constants()
        free = self.lean or visit is not None
        last = self.graph.last_use() if free else None
        own = {op.name for op in self._own_rows()}
        looked = own | set(self.tables)
        try:
            for i, op in enumerate(self.graph.ops):
                if free and i > 0:  # free what no later op reads (weight-op inputs stay in ``inputs``)
                    for n in self.graph.ops[i - 1].inputs:
                        if last.get(n, -1) <= i - 1:
                            env.pop(n, None)
                if isinstance(op, MatOp):
                    xin = env[op.inputs[0]]
                    m = op.n_cols(xin)
                    if op.name in own:
                        z = None if m is None else self._own_claim(op, xin, dtype)
                    else:
                        z = claims.get(op.name)
                    if z is None or m is None or z.dtype != dtype or tuple(z.shape) != (op.n_rows, m):
                        return None
                    if op.name not in own:          # a table's rows of int8 weights, else the range check
                        lo, hi = (-128, 127) if op.name in self.tables else (1 - Z_BOUND, Z_BOUND - 1)
                        if not _check_bounds(z, lo, hi, pending):
                            return None
                    if op.name in looked and not _check_bounds(xin, 0, op.n_in - 1, pending):   # ids of the table
                        return None
                    ranged[op.name] = _stamp(z)
                    if visit is None:
                        inputs[op.name] = xin
                    else:
                        visit(op, z, xin)
                    y = op.fold(z, xin)
                    env[op.output] = y if y.dtype == torch.int64 else y.to(torch.int64)
                else:
                    env[op.output] = op.fn(*[env[n] for n in op.inputs])
        except Exception:
            if any(_to_host(pending)):
                return None
            raise
        return inputs, pending, ranged

    # -- the checks ---------------------------------------------------------------------
    # The checks of different weight ops are independent, and a transformer repeats a few
    # shapes in every block, so same-shape ops are checked with ONE batched product instead
    # of ~15 small torch calls each.  Each op's result is the same integers as its own
    # product, and the verdict (and any exception) is that of checking op by op in order.

    def _kept_or_built(self, key, tensors: list, build):
        """``build(tensors)``; in mode Kpre, where ``chi`` and ``u`` are the verifier's own fixed
        values, kept across queries while exactly these, unmodified, tensors come back."""
        if self.mode != "Kpre":
            return build(tensors)
        hit = self._kept.get(key)
        if hit is not None and len(hit[0]) == len(tensors) and all(map(_same, hit[0], tensors)):
            return hit[1]
        out = build(tensors)
        self._kept[key] = ([_stamp(t) for t in tensors], out)
        return out

    def _field_products(self, pairs: dict, bound: int) -> dict:
        """``{name: left @ right mod P}`` for ``pairs[name] = (left, right)``, every ``|right| <
        bound``; same-shape pairs go through one batched limb product."""
        out, groups = {}, {}
        for name, (left, right) in pairs.items():
            groups.setdefault((left.shape, right.shape, left.device, right.device), []).append(name)
        chunk = exact_chunk(_LIMB_MAX, bound - 1)
        for (lshape, rshape, _, dev), names in groups.items():
            if len(names) == 1 and self.mode != "Kpre":        # nothing to share or to keep
                out[names[0]] = field_matmul_mod(*pairs[names[0]], right_bound=bound)
                continue
            step = max(1, _BATCH_BYTES // (8 * (rshape.numel() + 3 * lshape.numel())))
            for i in range(0, len(names), step):
                part = tuple(names[i:i + step])
                limbs = self._kept_or_built(("limbs",) + part, [pairs[n][0] for n in part],
                                            lambda ts: limbs_f64(torch.stack(ts)))
                rf = torch.empty((len(part),) + tuple(rshape), dtype=torch.float64, device=dev)
                for j, name in enumerate(part):
                    rf[j].copy_(pairs[name][1])
                out.update(zip(part, combine_limbs(exact_gemm_i64(limbs, rf, chunk), lshape[0])))
        return out

    def _lhs(self, mats, claims, chis) -> dict:
        """``chi_l Z_l mod P`` of every op of ``mats``."""
        return self._times_claims({op.name: (chis[op.name], op.name) for op in mats}, claims)

    def _times_claims(self, pairs: dict, claims, *, transposed: bool = False) -> dict:
        """``{key: chi @ Z mod P}`` for ``pairs[key] = (chi, op name)``, ``Z`` the op's claim (with
        ``transposed``: its transpose).  A claim that derive() range-checked (this very tensor, not
        modified since) is multiplied as a signed integer; any other is reduced first."""
        ranged, other = {}, {}
        for key, (chi, name) in pairs.items():
            z, seen = claims[name], self._ranged.get(name)
            if seen is not None and _same(seen, z):
                ranged[key] = (chi, z.T if transposed else z)
            else:
                z = to_field(z)
                other[key] = (chi, z.T if transposed else z)
        return {**self._signed_lhs(ranged), **self._field_products(other, P)}

    def _signed_lhs(self, pairs: dict) -> dict:
        """``{name: chi @ Z mod P}`` for ``pairs[name] = (chi, Z)`` with signed claims ``|Z| < 2**29``:
        int8 GEMMs where :func:`int8_ok` (exact for every int32 claim), else limb products."""
        if not pairs or not int8_ok(next(iter(pairs.values()))[1].device):
            return self._field_products(pairs, Z_BOUND)
        return {name: int8_field_matmul(self._kept_or_built(("chi8", name), [chi], lambda ts: int8_left(ts[0])), z)
                for name, (chi, z) in pairs.items()}

    def _rhs_all(self, mats, inputs, us, *, int8: bool = True) -> tuple[dict, list]:
        """``u_l^T [X_l ; 1]``, and a device flag per input the product took to be int8-valued.

        Linear ops reading the same input (q, k, v; gate, up) share one product with their ``u``
        rows stacked: an int8 GEMM of the input where :func:`int8_ok` (unless ``int8=False``),
        whose flag is set if the input is not int8-valued (the caller then computes again with
        ``int8=False``); else a limb product, with inputs of the same shape batched."""
        out, shared = {}, {}
        use8 = int8 and bool(mats) and int8_ok(us[mats[0].name].device)
        for op in mats:
            xin = inputs[op.name]
            if op.layout == "linear" and (use8 or op.n_in <= _RHS_CHUNK) and us[op.name].device == xin.device:
                shared.setdefault(id(xin), (xin, []))[1].append(op)
            else:
                out[op.name] = _rhs(op, us[op.name], xin)
        if use8:
            return out, [self._rhs_int8(xin, ops, us, out) for xin, ops in shared.values()]
        groups = {}
        for xin, ops in shared.values():
            key = (ops[0].n_in, xin.numel() // ops[0].n_in, tuple(us[op.name].shape[0] for op in ops), xin.device)
            groups.setdefault(key, []).append((xin, ops))
        for (k, m, rows, dev), items in groups.items():
            step = max(1, _BATCH_BYTES // (8 * k * (m + sum(rows))))
            for i in range(0, len(items), step):
                part = items[i:i + step]
                names = tuple(op.name for _, ops in part for op in ops)
                uf = self._kept_or_built(("u",) + names, [us[n] for n in names], lambda ts, g=len(part): torch.cat(
                    [t[:, :k] for t in ts]).to(torch.float64).reshape(g, sum(rows), k))
                # X = x^T [K, M] per input; the batch buffer takes the layout the inputs are stored
                # in (row- or column-major: a linear layer's output is a transposed view of its
                # claim, see MatOp.fold), so the copy reads memory in order
                xs = [xin.reshape(m, k).T for xin, _ in part]
                by_rows = not xs[0].is_contiguous()
                xf = torch.empty(len(part), *((m, k) if by_rows else (k, m)), dtype=torch.float64, device=dev)
                for j, x in enumerate(xs):
                    xf[j].copy_(x.T if by_rows else x)
                y = torch.remainder((uf @ (xf.transpose(1, 2) if by_rows else xf)).to(torch.int64), P)   # exact
                for j, (_, ops) in enumerate(part):
                    for op, part_rows in zip(ops, y[j].split(rows)):
                        out[op.name] = (part_rows + us[op.name][:, k:]) % P if op.has_bias else part_rows
        return out, []

    def _rhs_int8(self, xin: torch.Tensor, ops: list, us, out: dict) -> torch.Tensor:
        """Fills ``out`` with ``u^T [X ; 1]`` of the linear ``ops`` reading ``xin`` (one int8 GEMM);
        returns the device flag "``X`` is not int8-valued", which makes this product wrong."""
        k = ops[0].n_in
        x = xin.reshape(-1, k)
        names = tuple(op.name for op in ops)
        operand = self._kept_or_built(("u8",) + names, [us[n] for n in names],
                                      lambda ts: int8_right(torch.cat([t[:, :k] for t in ts])))
        y, r0 = int8_small_matmul(operand, x.to(torch.int8)), 0
        for op in ops:
            r = us[op.name].shape[0]
            out[op.name] = (y[r0:r0 + r] + us[op.name][:, k:]) % P if op.has_bias else y[r0:r0 + r]
            r0 += r
        return _outside(x, -128, 127)

    def _u_ok(self, op: MatOp, us) -> bool:
        """``u`` has the dtype and shape of the op's fold (whether it is in the field is left to
        :func:`_disagrees`)."""
        u = us.get(op.name)
        return u is not None and u.dtype == torch.int64 and tuple(u.shape) == (self.params.reps, op.row_length)

    def check_lookups(self, claims, inputs, proofs: dict) -> str | None:
        """A plan's lookup tables: ``None`` if every table op's claim is the table's rows at the ids it
        looks up, else the failing check, table by table -- ``lookup_consistency`` (two positions of one
        id with different rows), then ``lookup_merkle`` (the rows of the distinct ids and
        ``proofs[name]``, their one multiproof, do not give the table's root).  ``claims`` and
        ``inputs``: the ones derive() accepted, so every row is int8 (its bytes are the claim, one to
        one) and every id names a row."""
        return self._lookup_verdict(claims, {name: inputs[name] for name in self.tables}, proofs)

    def _lookup_verdict(self, claims, ids: dict, proofs) -> str | None:
        """:meth:`check_lookups` on the table ops' ``claims`` and ``ids`` (on any device)."""
        proofs = proofs or {}
        for name, table in self.tables.items():
            rows = claims[name].T.to(torch.int8).cpu().numpy()             # [M, d]
            uniq, first, inverse = np.unique(ids[name].reshape(-1).cpu().numpy(), return_index=True,
                                             return_inverse=True)
            if not np.array_equal(rows, rows[first][inverse.reshape(-1)]):
                return "lookup_consistency"
            proof = proofs.get(name)
            if (not isinstance(proof, (list, tuple)) or len(proof) > len(uniq) * table.depth
                    or not all(isinstance(h, bytes) and len(h) == HASH_BYTES for h in proof)
                    or not verify_multiproof(table.root, table.depth,
                                             table_leaves(table.tag, uniq.tolist(), rows[first]), list(proof))):
                return "lookup_merkle"
        return None

    def check_products(self, claims, inputs, chis, us) -> bool:
        """Freivalds for every weight op checked with it (all but a plan's col-layout ops; ``claims``
        are the ones derive() accepted): the verdict of checking op by op -- ``u`` well formed and
        in the field, then ``chi^T Z == u^T [X ; 1]``.  The field check of every ``u`` and every
        comparison come back with one copy."""
        mats = self._row_ops()
        n_ok, exc = _leading_passes(mats, lambda op: self._u_ok(op, us))
        good = mats[:n_ok]
        lhs = self._lhs(good, claims, chis)

        def disagree(rhs):
            return [_disagrees([(us[op.name], lhs[op.name], rhs[op.name]) for op in good])] if good else []

        rhs, unchecked = self._rhs_all(good, inputs, us)
        flags, n = _to_host(unchecked + disagree(rhs)), len(unchecked)
        if any(flags[:n]):                # an input that is not int8-valued (int8 GEMMs only)
            flags, n = _to_host(disagree(self._rhs_all(good, inputs, us, int8=False)[0])), 0
        if any(flags[n:]):
            return False
        if exc is not None:
            raise exc
        return n_ok == len(mats)

    def column_operands(self, claims, inputs, chis, us) -> tuple[dict, dict]:
        """The two sides of every committed matrix's code check, by matrix name: a row matrix's
        ``chi`` and ``u`` (``chis``, ``us``: every weight op's for the report's commitment), and a
        col-layout matrix's ``w = chi' [X ; 1]^T`` and ``z = chi' Z^T`` (``Z``: its ops' claims
        stacked, ``X`` their input), which the verifier computes itself from the ``claims`` and
        ``inputs`` derive() accepted -- exactly, same-shape products batched."""
        stacked = self._col_matrices()
        if not stacked:
            return chis, us
        zs = self._times_claims({op.name: (chis[name], op.name) for name, members in stacked.items() for op in members},
                                claims, transposed=True)
        lefts = dict(chis, **self._col_inputs(stacked, inputs, chis))
        return lefts, dict(us, **{name: torch.cat([zs[op.name] for op in members], 1)
                                  for name, members in stacked.items()})

    def _col_inputs(self, stacked: dict, inputs, chis) -> dict:
        """``w = chi' [X ; 1]^T`` of every col-layout matrix of ``stacked`` (``X^T``: its ops' input as
        ``[M, K]`` small integers), exact: blocks of ``_RHS_CHUNK`` terms, as in ``_rhs``; same-shape
        products batched."""
        out, batches = {}, {}
        for name, members in stacked.items():
            chi = chis[name]
            batches.setdefault((tuple(chi.shape), members[0].n_in, chi.device), []).append(name)
        for (shape, k, _), names in batches.items():
            step = max(1, _BATCH_BYTES // (8 * shape[1] * (shape[0] + k)))
            for i in range(0, len(names), step):
                part = names[i:i + step]
                left = torch.stack([chis[n] for n in part]).to(torch.float64)
                right = torch.stack([inputs[stacked[n][0].name].reshape(-1, k) for n in part])
                w = torch.remainder(exact_gemm_i64(left, right, _RHS_CHUNK), P)
                for n, wn in zip(part, w):
                    out[n] = torch.cat([wn, chis[n].sum(1, keepdim=True) % P], 1) if stacked[n][0].has_bias else wn
        return out

    # A column check opens trees: each weight op's own (the paper), or each group's of a commitment
    # plan, whose opening stacks its members' columns and whose index set they share.  Every member
    # is then checked against its own codeword -- a row matrix's Enc(u), a col matrix's Enc(z) --
    # exactly as a matrix on its own tree, and the verdict is that of checking tree by tree.

    def column_challenges(self, ch: Challenger) -> dict[str, torch.Tensor]:
        """Step 5's column indices, drawn after ``u``: ``t`` distinct columns of every tree."""
        if not self.groups:
            return {op.name: ch.columns(op.name, self.publics[op.name].n_points, self.params.columns)
                    for op in self.graph.mat_ops}
        return {name: ch.columns(name, g.n_points, self.params.columns_for(name)) for name, g in self.groups.items()}

    def _column_units(self) -> list[tuple[str, list[str]]]:
        """The trees, in order, each with the committed matrices it binds, in leaf order."""
        if not self.groups:
            return [(op.name, [op.name]) for op in self.graph.mat_ops]
        return [(name, list(g.members)) for name, g in self.groups.items()]

    def _tree(self, name: str):
        """The public of tree ``name`` (``tag``, ``root``, ``depth``): an op's own or a group's."""
        return self.groups[name] if self.groups else self.publics[name]

    def _offsets(self, members: list[str]) -> list[int]:
        """Where each member's rows start in its tree's stacked columns."""
        out, off = [], 0
        for m in members:
            out.append(off)
            off += self.publics[m].n_rows
        return out

    def _member_columns(self, units, opened: dict, *, rows: bool = False) -> dict:
        """Each member matrix's opened columns ``[n_rows, t]`` (views): ``opened`` holds each tree's
        columns ``[sum n_rows, t]``, or with ``rows`` its int32 rows ``[t, sum n_rows]``."""
        out = {}
        for name, members in units:
            o = opened[name].T if rows else opened[name]
            for m, off in zip(members, self._offsets(members)):
                out[m] = o if len(members) == 1 else o[off:off + self.publics[m].n_rows]
        return out

    def _columns_shape_ok(self, unit, us, cols, openings, *, wire: bool = False) -> bool:
        """The opening of tree ``unit = (name, members)`` is well formed: int64 columns ``[N, t]``
        (``N``: its members' rows together), or with ``wire`` the int32 rows ``[t, N]`` of the
        streaming verifier (``None``: not representable); in the field; with ``u`` of every row
        matrix among its members of its row length (a col matrix's ``z`` is the verifier's own)."""
        name, members = unit
        tree = self._tree(name)
        n_rows = sum(self.publics[m].n_rows for m in members)
        opened, proof = openings[name]
        idx = cols[name]
        if wire and opened is None:
            return False
        shape, dtype = ((len(idx), n_rows), torch.int32) if wire else ((n_rows, len(idx)), torch.int64)
        return not (tuple(opened.shape) != shape or len(proof) > len(idx) * tree.depth
                    or opened.dtype != dtype or not _in_field(opened)
                    or any(us[m].shape[1] != self.publics[m].row_length for m in members
                           if not self.publics[m].members))

    def _codewords(self, names, us, cols) -> dict:
        """``Enc(u)[columns]`` per matrix (``us``: the sources of :meth:`column_operands`).  ``Enc(u)``
        is only needed at the ``t`` opened columns: it is evaluated there directly
        (:func:`codeword_at`), same-shape matrices at once."""
        enc, groups = {}, {}
        for name in names:
            u = us[name]
            groups.setdefault((u.shape, self.publics[name].n_points, len(cols[name]), u.device), []).append(name)
        for (ushape, n_points, t, dev), same in groups.items():
            step = max(1, _BATCH_BYTES // (8 * ushape[-1] * (3 * ushape[0] + 2 * t)))
            for i in range(0, len(same), step):
                part = same[i:i + step]
                if len(part) == 1:
                    enc[part[0]] = codeword_at(us[part[0]], n_points, _to_device(cols[part[0]], dev))
                else:
                    res = codeword_at(torch.stack([us[n] for n in part]), n_points,
                                      _to_device(torch.stack([cols[n] for n in part]), dev))
                    enc.update(zip(part, res))
        return enc

    @staticmethod
    def _members_of(units, cols) -> tuple[list[str], dict]:
        """The member matrices of ``units``, in order, and each one's column indices (its tree's)."""
        return ([m for _, members in units for m in members],
                {m: cols[name] for name, members in units for m in members})

    def _code_flags(self, names, cols, chis, us, opened: dict) -> list[torch.Tensor]:
        """``chi^T (opened columns) != Enc(u)[columns]`` of every matrix of ``names`` (``chis``, ``us``:
        the two sides of :meth:`column_operands`), as booleans left where they were computed, which
        :func:`_to_host` reads as one per matrix: ``opened`` holds each one's opened columns ``[N, t]``
        in the field, next to its ``u`` (int64, or a view of the int32 rows ``[t, N]`` uploaded to a
        device).  On the CPU, where a value costs nothing to read, ``torch.equal`` (the fastest
        there) gives all of them in one tensor."""
        enc = self._codewords(names, us, cols)
        prod = self._field_products({m: (chis[m], opened[m]) for m in names}, P)
        if not _defer(self.device):
            return [torch.tensor([not torch.equal(prod[m], enc[m]) for m in names], dtype=torch.bool)]
        return [(prod[m] != enc[m]).any() for m in names]

    def _merkle(self, units, cols, openings, rows: dict | None = None) -> list[tuple[bool, Exception | None]]:
        """``(multiproof verified, exception)`` per tree: each member's opened columns (``rows``, or
        cast from ``openings``) hashed into its column digests, large ones on threads -- the leaves of
        an op's own tree, or through :func:`group_leaf` those of a group's -- then its multiproof."""
        if self.groups and rows is None:       # a group's columns are cast once, then sliced per member
            rows = {name: column_rows(openings[name][0]) for name, _ in units}
        parts = [(name, m, off) for name, members in units for m, off in zip(members, self._offsets(members))]

        def digests(part):
            name, m, off = part
            pub = self.publics[m]
            r = rows[name] if rows is not None else column_rows(openings[name][0])
            return row_leaves(pub.tag, cols[name].tolist(), r[:, off:off + pub.n_rows] if self.groups else r)

        workers = torch.get_num_threads()
        hashed = iter(map_threaded(digests, parts, [4 * self.publics[m].n_rows for _, m, _ in parts], workers))
        jobs = []
        for name, members in units:
            tree, own = self._tree(name), [next(hashed) for _ in members]
            leaves = {c: group_leaf(tree.tag, c, [d[c] for d in own]) for c in own[0]} if self.groups else own[0]
            jobs.append((tree.root, tree.depth, leaves, openings[name][1]))
        return verify_multiproofs(jobs, workers)

    def _opened(self, good, openings, *, wire: bool) -> tuple[dict | None, dict]:
        """``(rows, opened)`` of the well-formed trees ``good``: the int32 rows ``[t, N]`` the leaves
        hash (``None``: the leaves cast the columns, on threads), and each member's opened columns
        for the code checks.  On a device the rows are cast once (with ``wire``, they are the rows
        received) and uploaded; on the CPU the openings are used as they are."""
        received = {name: openings[name][0] for name, _ in good}
        if wire:
            rows = {name: o.numpy() for name, o in received.items()}
        elif _defer(self.device):
            rows = {name: column_rows(o) for name, o in received.items()}
        else:
            return None, self._member_columns(good, received)
        return rows, self._member_columns(good, _upload_rows(rows, self.device) if _defer(self.device) else received,
                                          rows=True)

    def check_columns(self, chis, us, cols, openings, *, wire: bool = False) -> str | None:
        """``None`` if every opened column is consistent, else the failing check: the verdict
        (and any exception) of checking tree by tree -- shape, then code (of every member), then
        Merkle.  ``chis`` and ``us``: the two sides of every matrix's code check
        (:meth:`column_operands`; each op's ``chi`` and ``u`` for the report's commitment).  The code
        checks are queued first (on a GPU they run there while the host checks the Merkle paths)
        and come back with one copy.  With ``wire`` the openings are the int32 rows ``[t, N]`` of
        :func:`pipeline.wire_openings` (``None``: not representable), which the checks take as they
        are."""
        units = self._column_units()
        n_ok, exc = _leading_passes(units, lambda unit: self._columns_shape_ok(unit, us, cols, openings, wire=wire))
        good = units[:n_ok]
        rows, opened = self._opened(good, openings, wire=wire)
        flags = self._code_flags(*self._members_of(good, cols), chis, us, opened)
        merkle = self._merkle(good, cols, openings, rows)
        return _columns_verdict(_unit_codes(good, _to_host(flags)), merkle, n_ok, exc, len(units))

    # -- the streaming verifier -------------------------------------------------------------
    # The same checks, arranged for a GPU client (the wire formats and the claim uploads are in
    # pvi.fullcheck.pipeline).  Each weight op is checked as soon as its claim is on the device:
    # its range check, both sides of Freivalds' check (chi^T Z, and u^T [X ; 1] once per input
    # tensor shared by q/k/v or gate/up; int8 tensor cores where int8_ok), then its fold into the
    # next op's input.  Nothing is kept for a later products pass and every tensor is freed after
    # its last reader, so the device holds a window of claims and one block's activations, not
    # the whole proof.  In mode C the column code checks are queued on the device first and the
    # Merkle checks run on a host thread while the device derives.  Every verdict stays on the
    # device until ONE copy decides the query; a lookup table's ids come back with it (queued
    # before it), and its rows are hashed on the host from the claims as received.  Only the
    # verifier's local computations are reordered, each a function of messages it already holds.

    def verify_streaming(self, x: torch.Tensor, claims: dict, chis: dict, us: dict,
                         cols: dict | None = None, openings: dict | None = None,
                         table_proofs: dict | None = None) -> str | None:
        """``None`` to accept, else the check that rejects, with ``run_query``'s labels: the verdict
        of :meth:`check_tokens` (``"token"``), :meth:`derive`, :meth:`check_lookups` (``table_proofs``: the multiproofs of a plan's lookup
        tables), :meth:`check_products` and (with ``openings``) :meth:`check_columns` on these
        messages.  ``claims`` from :func:`pipeline.wire_claim` (a lookup table's from
        :func:`pipeline.wire_rows`), ``openings`` from :func:`pipeline.wire_openings`.  In mode C the
        opened columns are part of the proof: without them every column check would be skipped, so a
        call without ``openings`` is refused rather than accepted on Freivalds' check alone."""
        if self.mode == "C" and openings is None:
            raise ValueError("mode C: verify_streaming needs the opened columns (openings)")
        if not self.check_tokens(x, claims):
            return "token"
        reason = self._streamed(x, claims, chis, us, cols, openings, table_proofs, int8=True)
        if reason is _NOT_INT8:        # (the graphs here clamp every weight op's input to int8)
            reason = self._streamed(x, claims, chis, us, cols, openings, table_proofs, int8=False)
        return reason

    def _streamed(self, x, claims, chis, us, cols, openings, table_proofs, *, int8: bool):
        dev = torch.device(self.device)
        mats = self._row_ops()
        stacked = self._col_matrices()
        matrix_of = {op.name: name for name, members in stacked.items() for op in members}
        index = {op.name: i for i, op in enumerate(mats)}
        us = {k: _to_device(v, dev) if torch.is_tensor(v) else v for k, v in us.items()}
        n_u, u_exc = _leading_passes(mats, lambda op: self._u_ok(op, us))
        reads = _input_bindings(self.graph)
        sharing: dict[tuple, list[MatOp]] = {}     # linear ops reading one input tensor: one right-hand side
        for op in mats[:n_u]:
            if op.layout == "linear":
                sharing.setdefault(reads[op.name], []).append(op)
        right: dict = {}
        x_flags: list = []
        flags: list = []
        lefts: dict = {}                           # col-layout matrices: w, and chi' Z^T of each of their ops
        parts: dict = {}
        looked: dict = {}                          # lookup tables: the ids each looks up
        elsewhere = set(self.tables) | {op.name for op in self._own_rows()}

        def visit(op: MatOp, z: torch.Tensor, xin: torch.Tensor) -> None:
            if op.name in elsewhere:           # a table (checked on the host after derive) or the verifier's own rows
                if op.name in self.tables:
                    looked[op.name] = xin
                return
            name = matrix_of.get(op.name)
            if name is not None:               # both sides of its matrix's code check, if it is checked
                if columns is not None and name in chis:     # (no chi': a malformed x, which derive rejects)
                    if name not in lefts:
                        lefts.update(self._col_inputs({name: stacked[name]}, {stacked[name][0].name: xin}, chis))
                    parts[op.name] = self._signed_lhs({op.name: (chis[name], z.T)})[op.name]
                return
            if index[op.name] >= n_u:          # the verdict no longer depends on this product
                return
            if op.name not in right:
                group = sharing.get(reads[op.name], [op]) if op.layout == "linear" else [op]
                out, unchecked = self._rhs_all(group, {o.name: xin for o in group}, us, int8=int8)
                right.update(out)
                x_flags.extend(unchecked)
            left = self._signed_lhs({op.name: (chis[op.name], z)})[op.name]
            flags.append(_disagrees([(us[op.name], left, right.pop(op.name))]))

        with ThreadPoolExecutor(1) as host:
            # the column checks compare with Enc(u): only started when every u is well formed (with
            # one that is not, the verdict is freivalds or u_exc, and the columns are never looked at)
            well_formed = openings is not None and n_u == len(mats)
            columns = self._start_columns(chis, us, cols, openings, host) if well_formed else None
            uploads = ClaimUploads(claims, [op.name for op in self._sent_ops()], dev, rows=set(self.tables))
            derived = self._derive(_to_device(x, dev), uploads, dtype=torch.int32, visit=visit)
            if derived is None:
                return "range_or_shape"
            pending = derived[1]
            ids = {name: _host_copy(xin) for name, xin in looked.items()}   # read after the one round trip
            code = [] if columns is None else columns.flags
            if columns is not None and columns.later:    # the col matrices' code checks: derive gave w and z
                sources = {name: torch.cat([parts[op.name] for op in stacked[name]], 1) for name in columns.later}
                code = code + self._code_flags(columns.later, columns.cols, lefts, sources, columns.opened)
            got = _to_host(pending + x_flags + flags + code)       # the one round trip
            a, b, c = len(pending), len(pending) + len(x_flags), len(pending) + len(x_flags) + len(flags)
            if any(got[:a]):
                return "range_or_shape"
            reason = self._lookup_verdict(claims, ids, table_proofs)       # (the rows on the host, as received)
            if reason is not None:
                return reason
            if any(got[a:b]):
                return _NOT_INT8
            if any(got[b:c]) or (u_exc is None and n_u < len(mats)):
                return "freivalds"
            if u_exc is not None:
                raise u_exc
            if columns is None:
                return None
            failed = dict(zip(columns.now + columns.later, got[c:]))
            codes = _unit_codes(columns.units[:columns.n_ok], [failed[m] for m in columns.names])
            return _columns_verdict(codes, columns.merkle.result(), columns.n_ok, columns.exc, len(columns.units))

    def _start_columns(self, chis, us, cols, openings, host: ThreadPoolExecutor) -> "_StartedColumns":
        """The column checks of the leading trees whose openings are well formed, started: the code
        checks of their row matrices queued on the device, and their Merkle checks on ``host``."""
        units = self._column_units()
        n_ok, exc = _leading_passes(units, lambda unit: self._columns_shape_ok(unit, us, cols, openings, wire=True))
        good = units[:n_ok]
        rows, opened = self._opened(good, openings, wire=True)
        names, idx = self._members_of(good, cols)
        now = [m for m in names if not self.publics[m].members]
        later = [m for m in names if self.publics[m].members]
        return _StartedColumns(units, n_ok, exc, names, idx, opened, now, later,
                               self._code_flags(now, idx, chis, us, opened),
                               host.submit(self._merkle, good, cols, openings, rows))


class _StartedColumns(NamedTuple):
    """The streaming verifier's column checks, started (:meth:`Verifier._start_columns`): every tree
    (``units``), the ``n_ok`` leading ones with well-formed openings and what the next one raised
    (``exc``), their matrices (``names``, in tree order) with their column indices and opened columns
    (on the device), the row matrices (``now``) whose code checks were queued (``flags``) and the col
    ones (``later``) whose checks wait for their ``w`` and ``z``, and the Merkle checks' future."""

    units: list
    n_ok: int
    exc: Exception | None
    names: list
    cols: dict
    opened: dict
    now: list
    later: list
    flags: list
    merkle: Future


def _input_bindings(graph: IntGraph) -> dict[str, tuple[str, int]]:
    """Each weight op's input as ``(name, index of the op that last bound it, or -1 for the
    query)``: the tensor it reads, as a key (a graph may bind one name more than once)."""
    writer: dict[str, int] = {}
    out = {}
    for i, op in enumerate(graph.ops):
        if isinstance(op, MatOp):
            out[op.name] = (op.inputs[0], writer.get(op.inputs[0], -1))
        writer[op.output] = i
    return out


def _sync(device) -> None:
    device = torch.device(device)
    if device.type == "cuda":
        torch.cuda.synchronize(device)   # the verifier's GPU, not necessarily the current one


def _phase_done(out: dict, name: str, start) -> None:
    """``out["contention"][name]``: the contention evidence of the timed phase ``name`` that
    ``start = contention.begin()`` opened (:mod:`pvi.fullcheck.contention`; measurement only, both
    snapshots outside the timed window; a phase timed in parts sums them)."""
    contention.add(out.setdefault("contention", {}), name, start)


def _absorb_statement(ch: Challenger, verifier: Verifier, x: torch.Tensor) -> None:
    p = verifier.params
    ch.absorb(b"graph", verifier._digest or verifier.graph.digest())    # every op and cheap-op constant
    ch.absorb(b"params", struct.pack("<IIIi", p.reps, p.columns, p.rate, p.grinding_bits) + verifier.mode.encode())
    for op in verifier.graph.mat_ops:
        pub = verifier.publics.get(op.name)
        blob = struct.pack("<QQ", op.n_rows, op.row_length)
        if pub is not None:
            blob += pub.tag + pub.root + struct.pack("<Q", pub.n_points)
        ch.absorb(b"op/" + op.name.encode(), blob)
    for name, pub in verifier.publics.items() if verifier.groups else ():
        if pub.members:            # a plan's col-layout matrix: its code and the ops it stacks (in order)
            blob = struct.pack("<QQQH", pub.n_rows, pub.row_length, pub.n_points, len(pub.tag)) + pub.tag
            ch.absorb(b"col/" + name.encode(), blob + b"".join(struct.pack("<H", len(m)) + m.encode()
                                                                for m in pub.members))
    for name, g in verifier.groups.items():      # a commitment plan: its trees, members (in order) and t
        blob = struct.pack("<H", len(g.tag)) + g.tag + g.root + struct.pack("<QQQ", g.n_points, p.columns_for(name),
                                                                            len(g.members))
        ch.absorb(b"group/" + name.encode(), blob + b"".join(struct.pack("<H", len(m)) + m.encode() for m in g.members))
    for name, table in verifier.tables.items():   # a plan's lookup tables: the root over each one's rows
        ch.absorb(b"table/" + name.encode(), struct.pack("<H", len(table.tag)) + table.tag + table.root
                  + struct.pack("<QQ", table.n_rows, table.n_tokens))
    if verifier.lookups:                          # modes K and Kpre: the embedding ops the verifier reads itself
        ch.absorb(b"lookups", b"".join(struct.pack("<H", len(op.name)) + op.name.encode()
                                       for op in verifier._own_rows()))
    ch.absorb(b"x", _tensor_blob(x.cpu()))


# -- one query ---------------------------------------------------------------------------
# The prover's messages, each challenge drawn after the message it follows (under Fiat--Shamir
# from the transcript that absorbed it), each message's proof bytes counted and its prover time
# recorded in ``out``.  run_query checks each message as it arrives and stops asking at the
# first failing check; the streaming verifier receives them all, then checks.  Both parties run
# in one process, so what one hands the other is passed as a copy (:func:`_passed`): the query, the
# challenges and the column indices the prover receives, and every message of the prover's that
# the verifier reads after the prover has run again (the claims, ``u`` and the tables'
# multiproofs) -- as a wire, a decoder or a GPU client's upload gives each party its own tensors.

def _passed(message, pin: bool = False):
    """``message`` (a tensor, or a dict of them) as the other party receives it: a copy of every
    tensor (with ``pin``, in pinned host memory: a GPU client's) and of every list among the values,
    so that neither party can change in place, when it runs again, what the other holds."""
    def copy(t):
        return hostmem.empty(t.shape, t.dtype).copy_(t) if pin else t.clone()

    if torch.is_tensor(message):
        return copy(message)
    return {k: copy(v) if torch.is_tensor(v) else list(v) if isinstance(v, list) else v for k, v in message.items()}


def _encoded(encode, values) -> bytes:
    """``encode(values)``, the bytes of a message in the compact encoding; none for values it cannot
    carry (:class:`claimcodec.Unencodable`), which a prover can only replace with other bytes: no
    bytes, a message the verifier rejects as malformed at that message's check (``range_or_shape``,
    ``freivalds``, ``columns_shape``), as it rejects those values without wire."""
    try:
        return encode(values)
    except claimcodec.Unencodable:
        return b""


def _proof_hashes(proof) -> list[bytes]:
    """The hashes of a multiproof as they travel (nothing if it is not a list of byte strings)."""
    ok = isinstance(proof, (list, tuple)) and all(isinstance(h, bytes) for h in proof)
    return list(proof) if ok else []


def _claims_message(prover: Prover, verifier: Verifier, x: torch.Tensor, ch: Challenger, out: dict,
                    forward_kwargs: dict, send=None, wire: bool = False) -> tuple[dict | bytes, bytes | None, dict]:
    """``(claims, rows, proofs)``: the claims of every weight op but those whose rows the verifier reads
    itself (``send``: the wire format they travel in; ``wire``: the ``PVC3`` bytes of all but a plan's
    lookup tables, encoded where the prover keeps them, and ``rows`` the tables' as int8 bytes, else
    ``None``) and the multiproof of every lookup table, absorbed with the statement.  A table's claims
    count one byte each."""
    t = out["timings"]
    query = _passed(x)
    if wire and send is None and prover.lean and prover.device.type != "cpu":
        # a lean GPU prover keeps each claim on its device as int32 (not the host's int64) and encodes
        # them there: about 2.3 bytes a claim come back instead of 8
        send = claimcodec.narrow
    _sync(prover.device)
    c0 = contention.begin()
    t0 = time.perf_counter()
    claims = prover.claims(query, send=send, to_host=not wire, **forward_kwargs)
    _sync(prover.device)
    t["prove_forward"] = time.perf_counter() - t0
    _phase_done(out, "prove_forward", c0)
    sent = {op.name for op in verifier._sent_ops()}
    if len(sent) < len(claims):
        claims = {k: z for k, z in claims.items() if k in sent}
    proofs, rows = {}, None
    if verifier.tables:
        c0 = contention.begin()
        t0 = time.perf_counter()
        proofs = prover.open_tables()
        t["prove_lookups"] = time.perf_counter() - t0
        _phase_done(out, "prove_lookups", c0)
        proofs = _passed(proofs)
    if wire:
        c0 = contention.begin()
        t0 = time.perf_counter()
        if verifier.tables:
            rows = _encoded(claimcodec.pack_rows, [claims[name] for name in verifier.tables])
        claims = _encoded(claimcodec.encode, [claims[op.name] for op in prover.graph.mat_ops
                                              if op.name in sent and op.name not in verifier.tables])
        _sync(prover.device)
        t["prove_encode"] = time.perf_counter() - t0
        _phase_done(out, "prove_encode", c0)
    size = len(claims) + len(rows or b"") if wire else sum(z.numel() * (1 if k in verifier.tables else 4)
                                                          for k, z in claims.items())
    out["bytes"] = {"claims": size, "u": 0, "columns": 0,
                    "paths": HASH_BYTES * sum(len(_proof_hashes(p)) for p in proofs.values())}
    c0 = contention.begin()
    t0 = time.perf_counter()
    if verifier.params.fiat_shamir:
        _absorb_statement(ch, verifier, x)
        if wire:
            ch.absorb(b"claims/" + claimcodec.MAGIC, claims)
            if rows is not None:
                ch.absorb(b"rows/I8", rows)
        else:
            for k in sorted(claims):       # int32 wire claims hash as the int64 ones
                ch.absorb(b"claim/" + k.encode(), _tensor_blob(claims[k]))
        for name in verifier.tables:       # the tables' multiproofs, in table order
            hashes = _proof_hashes(proofs.get(name))
            ch.absorb(b"lookup/" + name.encode(), struct.pack("<Q", len(hashes)) + b"".join(hashes))
    t["fs_hash"] = time.perf_counter() - t0
    _phase_done(out, "fs_hash", c0)
    return claims, rows, proofs


def _decoded_claims(verifier: Verifier, blob: bytes, rows: bytes | None, x: torch.Tensor, out: dict,
                    dtype: torch.dtype, pin: bool = False) -> dict | None:
    """The verifier's claims from their ``PVC3`` bytes and a plan's lookup tables' from their int8
    ``rows`` (``None``: malformed), as ``dtype`` host tensors (int32 in pinned memory with ``pin``),
    timed as ``verify_decode``.  Each claim must have the shape the query gives it, which the decoders
    check before they allocate the claims."""
    sent = verifier._sent_ops()
    mats = [op for op in sent if op.name not in verifier.tables]
    tables = [op for op in sent if op.name in verifier.tables]
    c0 = contention.begin()
    t0 = time.perf_counter()
    cols = verifier.claim_columns(x)
    zs = None
    if cols is not None:
        m_of = dict(zip((op.name for op in verifier.graph.mat_ops), cols))
        try:
            zs = dict(zip((op.name for op in mats), claimcodec.decode_torch(
                blob, [op.n_rows for op in mats], [m_of[op.name] for op in mats], dtype=dtype,
                workers=torch.get_num_threads(), pin=pin)))
            if tables:
                zs.update(zip((op.name for op in tables), claimcodec.unpack_rows(
                    rows, [(op.n_rows, m_of[op.name]) for op in tables], dtype=dtype, pin=pin)))
        except claimcodec.ClaimCodecError:
            zs = None
    out["timings"]["verify_decode"] = out["timings"].get("verify_decode", 0.0) + time.perf_counter() - t0
    _phase_done(out, "verify_decode", c0)
    return zs


def _field_message(blob: bytes, shapes: list[tuple[int, int]], out: dict, dtype: torch.dtype) -> list | None:
    """The field elements of a 31-bit packed message as ``dtype`` host tensors of ``shapes``
    (``None``: malformed), timed as ``verify_decode``."""
    c0 = contention.begin()
    t0 = time.perf_counter()
    try:
        flat = claimcodec.unpack_field(blob, sum(a * b for a, b in shapes), workers=torch.get_num_threads())
    except claimcodec.ClaimCodecError:
        flat = None
    parts = None
    if flat is not None:
        flat = torch.from_numpy(flat.view(np.int32)).to(dtype)
        parts, o = [], 0
        for a, b in shapes:
            parts.append(flat[o:o + a * b].view(a, b))
            o += a * b
    out["timings"]["verify_decode"] = out["timings"].get("verify_decode", 0.0) + time.perf_counter() - t0
    _phase_done(out, "verify_decode", c0)
    return parts


def _fold_message(prover: Prover, verifier: Verifier, ch: Challenger, x: torch.Tensor, out: dict,
                  wire: bool = False) -> tuple[dict, dict, bytes | None]:
    """``(chi, u, the bytes of u with wire)``: fresh challenges (:meth:`Verifier.fold_challenges`:
    under a commitment plan also each col-layout matrix's ``chi'``) and the prover's ``u`` of every
    op checked with Freivalds (mode C; with ``wire`` 31-bit packed, and ``{}`` if the packed ``u``
    is malformed), the verifier's own (K), or the precomputed pair (Kpre)."""
    p, t = verifier.params, out["timings"]
    mats = verifier._row_ops()
    if verifier.mode == "Kpre":
        t["prove_fold"] = t["verify_fold"] = 0.0
        return ({op.name: verifier._pre[op.name][0] for op in mats},
                {op.name: verifier._pre[op.name][1] for op in mats}, None)
    vdev = torch.device(verifier.device)
    chis = {name: chi.to(vdev) for name, chi in verifier.fold_challenges(ch, x).items()}
    sent = _passed({op.name: chis[op.name] for op in mats}) if verifier.mode == "C" else None
    _sync(prover.device)
    c0 = contention.begin()
    t0 = time.perf_counter()
    blob = None
    if verifier.mode == "C":
        us = prover.fold(sent)
        _sync(prover.device)
        t["prove_fold"] = time.perf_counter() - t0
        _phase_done(out, "prove_fold", c0)
        t["verify_fold"] = 0.0
        out["bytes"]["u"] = sum(u.numel() for u in us.values()) * 4
        if wire:
            c0 = contention.begin()
            t0 = time.perf_counter()
            blob = _encoded(claimcodec.pack_field, [us[op.name] for op in mats])
            t["prove_encode"] += time.perf_counter() - t0
            _phase_done(out, "prove_encode", c0)
            out["bytes"]["u"] = len(blob)
            parts = _field_message(blob, [(p.reps, op.row_length) for op in mats], out, torch.int64)
            us = {} if parts is None else {op.name: u for op, u in zip(mats, parts)}
        else:
            us = _passed(us)
    else:  # K: the verifier folds its own copy of the weights
        us = {op.name: verifier._fold_local(op, chis[op.name]) for op in mats}
        _sync(vdev)
        t["verify_fold"] = time.perf_counter() - t0
        _phase_done(out, "verify_fold", c0)
        t["prove_fold"] = 0.0
    return chis, us, blob


def _open_message(prover: Prover, verifier: Verifier, ch: Challenger, us: dict, out: dict,
                  package=None, u_blob: bytes | None = None) -> tuple[dict, dict]:
    """Mode C: ``u`` absorbed (its bytes ``u_blob`` with wire), the column indices, and the openings
    (``package``: the wire format they travel in).  With ``u_blob`` the opened columns of every tree
    (each weight op's, or each group's: its members' rows side by side) travel 31-bit packed, tree
    after tree in the order of the challenges and column by column, and arrive as the int32 rows
    ``[t, N]`` of :func:`pipeline.wire_openings` (``None`` if the packed columns are malformed)."""
    p, t = verifier.params, out["timings"]
    if p.fiat_shamir:
        c0 = contention.begin()
        t0 = time.perf_counter()
        if u_blob is not None:
            ch.absorb(b"u/F31", u_blob)
        else:
            for k in sorted(us):
                ch.absorb(b"u/" + k.encode(), _tensor_blob(us[k]))
        t["fs_hash"] += time.perf_counter() - t0
        _phase_done(out, "fs_hash", c0)
    cols = verifier.column_challenges(ch)
    sent = _passed(cols)
    _sync(prover.device)
    c0 = contention.begin()
    t0 = time.perf_counter()
    opened = prover.open(sent)
    _sync(prover.device)
    if u_blob is None:
        openings = opened if package is None else package(opened)
        t["prove_open"] = time.perf_counter() - t0
        _phase_done(out, "prove_open", c0)
        out["bytes"]["columns"] = sum(o[0].numel() for o in opened.values()) * 4
    else:
        t["prove_open"] = time.perf_counter() - t0
        _phase_done(out, "prove_open", c0)
        c0 = contention.begin()
        t0 = time.perf_counter()
        blob = _encoded(claimcodec.pack_field, [opened[name][0].T for name in cols])
        t["prove_encode"] += time.perf_counter() - t0
        _phase_done(out, "prove_encode", c0)
        out["bytes"]["columns"] = len(blob)
        units = verifier._column_units()
        rows = _field_message(blob, [(len(cols[name]), sum(verifier.publics[m].n_rows for m in members))
                                     for name, members in units], out, torch.int32)
        openings = {name: (None if rows is None else rows[i], opened[name][1]) for i, (name, _) in enumerate(units)}
    out["bytes"]["paths"] += sum(len(proof) * HASH_BYTES for _, proof in opened.values())
    return cols, openings


def run_query(prover: Prover, verifier: Verifier, x: torch.Tensor, *, seed: int | None = None,
              forward_kwargs: dict | None = None, wire: bool = False) -> dict:
    """One full interaction.  Returns acceptance, the rejecting check, timings (s)
    and proof bytes.  With Fiat--Shamir, ``fs_hash`` is paid by both parties.

    A verifier with ``stream=True`` receives every message (the claims and openings in the wire
    formats of :mod:`pvi.fullcheck.pipeline`), then checks them with :meth:`Verifier.verify_streaming`:
    the same verdict and label, its overlapped work timed as one ``verify_total``.  An accepted
    query has the same proof bytes and Fiat--Shamir transcript; a rejected one has also received
    (counted, and absorbed) the messages this function no longer asks for once a check fails --
    in mode C, ``u`` and the openings -- so the transcript here is a prefix of the streaming one.

    ``wire=True`` sends the proof in the compact encoding of :mod:`pvi.fullcheck.claimcodec`: the
    claims as ``PVC3`` bytes, encoded where the prover keeps them (``prove_encode``), and ``u`` and
    the opened columns 31-bit packed.  The verifier decodes them (``verify_decode``) before it
    checks anything and rejects a malformed message at the check that rejects a malformed message
    of the same kind (``range_or_shape`` for the claims, ``freivalds`` for ``u``, ``columns_shape``
    for the columns) -- so too a prover's values that the encoding cannot carry (a claim outside the
    range check, a looked-up row outside int8, a field element of 32 bits), which travel as no
    bytes; ``bytes`` counts the encoded sizes, and with Fiat--Shamir the transcript absorbs these
    bytes.  A GPU client uploads the decoded claims as int32 and widens them there."""
    ch = Challenger(fiat_shamir=verifier.params.fiat_shamir, seed=seed)
    out = {"accepted": False, "rejected_at": None, "timings": {}, "contention": {}}
    if verifier.stream:
        out["rejected_at"] = _run_streaming(prover, verifier, x, ch, out, forward_kwargs or {}, wire)
        out["accepted"] = out["rejected_at"] is None
        return out
    t = out["timings"]
    claims, rows, proofs = _claims_message(prover, verifier, x, ch, out, forward_kwargs or {}, wire=wire)

    vdev = torch.device(verifier.device)
    x_v, claims_v = x.cpu(), claims
    if wire:                      # a GPU client uploads int32 claims
        claims_v = _decoded_claims(verifier, claims, rows, x, out, torch.int64 if vdev.type == "cpu" else torch.int32)
        if claims_v is None:
            out["rejected_at"] = "range_or_shape"
            return out
    else:                         # the prover's own tensors, which the checks after its fold read again
        claims_v = _passed(claims)
    if not verifier.check_tokens(x_v, claims_v):
        out["rejected_at"] = "token"
        return out
    if vdev.type != "cpu":        # a GPU client: receiving the proof includes uploading it
        _sync(vdev)
        c0 = contention.begin()
        t0 = time.perf_counter()
        x_v, claims_v = x.to(vdev), {k: v.to(vdev) for k, v in claims_v.items()}
        if wire:
            claims_v = {k: v.to(torch.int64) for k, v in claims_v.items()}
        _sync(vdev)
        t["verify_upload"] = time.perf_counter() - t0
        _phase_done(out, "verify_upload", c0)

    c0 = contention.begin()
    t0 = time.perf_counter()
    inputs = verifier.derive(x_v, claims_v)
    _sync(vdev)
    t["verify_derive"] = time.perf_counter() - t0
    _phase_done(out, "verify_derive", c0)
    if inputs is None:
        out["rejected_at"] = "range_or_shape"
        return out
    if verifier.tables:
        c0 = contention.begin()
        t0 = time.perf_counter()
        reason = verifier.check_lookups(claims_v, inputs, proofs)
        t["verify_lookups"] = time.perf_counter() - t0
        _phase_done(out, "verify_lookups", c0)
        if reason is not None:
            out["rejected_at"] = reason
            return out

    chis, us, u_blob = _fold_message(prover, verifier, ch, x, out, wire)
    if verifier.mode == "C" and vdev.type != "cpu":
        c0 = contention.begin()
        t0 = time.perf_counter()
        us = {k: v.to(vdev) for k, v in us.items()}
        _sync(vdev)
        t["verify_upload"] += time.perf_counter() - t0
        _phase_done(out, "verify_upload", c0)

    c0 = contention.begin()
    t0 = time.perf_counter()
    ok = verifier.check_products(claims_v, inputs, chis, us)
    if ok and verifier.mode == "C":     # the code checks' two sides (a col-layout matrix's: the verifier's own)
        lefts, sources = verifier.column_operands(claims_v, inputs, chis, us)
    _sync(vdev)
    t["verify_products"] = time.perf_counter() - t0
    _phase_done(out, "verify_products", c0)
    if not ok:
        out["rejected_at"] = "freivalds"
        return out

    if verifier.mode == "C":
        cols, openings = _open_message(prover, verifier, ch, us, out, u_blob=u_blob)
        c0 = contention.begin()
        t0 = time.perf_counter()
        reason = verifier.check_columns(lefts, sources, cols, openings, wire=wire)
        _sync(vdev)
        t["verify_columns"] = time.perf_counter() - t0
        _phase_done(out, "verify_columns", c0)
        if reason is not None:
            out["rejected_at"] = reason
            return out
    out["accepted"] = True
    return out


def _run_streaming(prover: Prover, verifier: Verifier, x: torch.Tensor, ch: Challenger, out: dict,
                   forward_kwargs: dict, wire: bool) -> str | None:
    """:func:`run_query` with ``stream=True``: every message, then :meth:`Verifier.verify_streaming`.
    ``verify_total`` times the verifier's work from the moment it holds the messages (its uploads
    included; with ``wire``, after ``verify_decode``, which gives it the int32 claims and rows of
    :mod:`pvi.fullcheck.pipeline`'s wire formats)."""
    vdev = torch.device(verifier.device)
    pin = vdev.type == "cuda"
    claims, rows, proofs = _claims_message(prover, verifier, x, ch, out, forward_kwargs, wire=wire,
                                           send=None if wire else lambda z: wire_claim(z, pin=pin))
    if not wire:   # the verifier's own copies (read after the prover runs again); a lookup table's as its int8 rows
        claims = {k: wire_rows(z, pin=pin) if k in verifier.tables else z for k, z in _passed(claims, pin).items()}
    chis, us, u_blob = _fold_message(prover, verifier, ch, x, out, wire)
    cols = openings = None
    if verifier.mode == "C":
        cols, openings = _open_message(prover, verifier, ch, us, out, package=wire_openings, u_blob=u_blob)
    if wire:
        claims = _decoded_claims(verifier, claims, rows, x, out, torch.int32, pin=vdev.type == "cuda")
        if claims is None:
            return "range_or_shape"
    _sync(vdev)
    c0 = contention.begin()
    t0 = time.perf_counter()
    reason = verifier.verify_streaming(x, claims, chis, us, cols, openings, proofs)
    _sync(vdev)
    out["timings"]["verify_total"] = time.perf_counter() - t0
    _phase_done(out, "verify_total", c0)
    return reason

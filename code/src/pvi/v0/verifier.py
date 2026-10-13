"""The verifier in modes C, K and Kpre: derive every weight op's input from the claims, Freivalds' check
of every weight op, and in mode C the column checks against the commitment; its streaming form
checks messages as they arrive."""

from __future__ import annotations

import weakref
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

from pvi.commit.commitment import (HASH_BYTES, CommitmentPublic, GroupPublic, TablePublic, codeword_at, column_rows,
                                   group_leaf, map_threaded, row_leaves, table_leaves, verify_multiproof,
                                   verify_multiproofs)
from pvi.core import hostmem
from pvi import options
from pvi.core.field import (_LIMB_MAX, P, combine_limbs, exact_chunk, exact_gemm_i64, field_matmul_mod,
                            int8_field_matmul, int8_left, int8_ok, int8_right, int8_small_matmul, int8_weight_matmul,
                            limbs_f64, min_max, small_matmul_mod, to_field)
from pvi.core.graph import CheapOp, IntGraph, MatOp
from pvi.kernels import native as native_kernels
from pvi.v0.params import _CUT_NARROW, Z_BOUND, SecurityParams
from pvi.v0.streaming import ClaimUploads
from pvi.v0.transcript import Challenger
from pvi.v1.cut import delta_range as cut_delta_range

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
    _cut_plan: object = None                      # V1: the query's cut.CutPlan (set by cut_protocol, per query)
    sampler: object = None                        # E2: the query's token_sampler.TokenSampler (set by run_query)

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

    def precompute(self, challenger: Challenger | None = None) -> None:
        """Mode Kpre: fix a secret ``chi`` per op and precompute ``u``. The vectors must stay secret, so the
        challenger is interactive: by default a fresh one (keys from ``secrets``); a seeded one is for tests only;
        a Fiat--Shamir one, whose draws anyone can recompute from public data, is refused."""
        challenger = Challenger() if challenger is None else challenger
        if challenger.fiat_shamir:
            raise ValueError("Kpre's vectors must be secret: precompute needs an interactive Challenger")
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
        """``u = chi A`` (``[r, K]``, canonical) of op's int8 weights (with the option ``int8_fold`` off, the float64
        path alone): on a CPU the native exact field product (``native_kernels.field_matmul``, the weights read as int32,
        no transposed int64 copy), on a GPU int8 GEMMs (``weight_matmul_mod``), else the float64 limb products; all
        give the canonical residue of the same integers."""
        w, b = self._weights_on(op, chi.device)
        u = None
        if options.current().int8_fold and w.dtype == torch.int8:
            if chi.device.type == "cpu":
                u = native_kernels.field_matmul(chi.to(torch.int64).contiguous(), w.to(torch.int32))
            elif chi.is_cuda and int8_ok(chi.device):
                u = int8_weight_matmul(w.T.contiguous(), chi.T.contiguous()).T
        if u is None:
            u = small_matmul_mod(w.T.to(torch.int64).contiguous(), chi.T.contiguous()).T
        if b is not None:
            u = torch.cat([u, field_matmul_mod(chi, to_field(b)[:, None])], 1)
        return u

    def _col_matrices(self) -> dict[str, list[MatOp]]:
        """A commitment plan's col-layout matrices, each with the weight ops it stacks (in order)."""
        if not self.groups:
            return {}
        ops = {op.name: op for op in self.graph.mat_ops}
        cut = self._cut_names()
        return {name: [ops[o] for o in pub.members] for name, pub in self.publics.items()
                if pub.members and not cut.intersection(pub.members)}

    def _cut_names(self) -> frozenset:
        """V1: the weight ops of the query's cut set (none outside a cut query)."""
        return self._cut_plan.names() if self._cut_plan is not None else frozenset()

    def _row_ops(self) -> list[MatOp]:
        """The weight ops checked with Freivalds (step 4): every one but a plan's col-layout ops and
        lookup tables, and the embedding ops whose rows a verifier with ``lookups`` reads itself."""
        skip = {op.name for members in self._col_matrices().values() for op in members}
        skip |= set(self.tables) | {op.name for op in self._own_rows()} | self._cut_names()
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
        wrong token needs wrong logits.  A query shorter than ``n`` positions is rejected (its tokens
        could not all be checked); claims of another shape pass here (:meth:`derive` rejects them)."""
        n = self.graph.meta.get("lm_positions", 1)
        if n <= 1:
            return True
        if x.dim() != 2 or x.shape[1] < n:
            return False
        head = next(op for op in self.graph.mat_ops if op.output == self.graph.output_name)
        z = claims.get(head.name)
        if not torch.is_tensor(z) or tuple(z.shape) != (head.n_rows, x.shape[0] * n):
            return True
        if self.sampler is not None:      # E2: each token is the seeded sampler's choice from the logits before it
            if x.shape[0] != 1:
                return False
            t = x.shape[1]
            want = [self.sampler.choose(z[:, j], t - n + j) for j in range(n - 1)]
            return x[0, t - n + 1:].cpu().tolist() == want
        chosen = z.reshape(head.n_rows, x.shape[0], n)[:, :, :-1].argmax(0)          # [B, n - 1]
        return torch.equal(chosen.cpu(), x[:, x.shape[1] - n + 1:].cpu().to(chosen.dtype))

    def derive(self, x: torch.Tensor, claims: dict[str, torch.Tensor]) -> dict[str, torch.Tensor] | None:
        """Recompute every cheap op; return each weight op's input, or ``None`` to reject.

        The range checks of claims on a GPU are deferred: they come back with one copy at
        the end instead of one per weight op.  A lookup table's claims must be int8, and every op
        that looks rows up (a table, or the verifier's own rows) must look up ids of its table."""
        dtype = torch.int64
        if x.device.type == "cpu" and native_kernels.cheap_enabled(x):
            native_kernels.tune_malloc()                          # (once per process; see its docstring)
        if x.device.type == "cpu" and native_kernels.int32_claims() and any(
                torch.is_tensor(z) and z.dtype == torch.int32 for z in claims.values()):
            dtype = torch.int32                                   # int32 claims, read by the native kernels
        out = self._derive(x, claims, dtype=dtype)
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
        cut = self._cut_plan
        cut_ops = {o.name: o for o in cut.ops} if cut is not None else {}
        consumers = {o.consumer: o for o in cut.ops} if cut is not None else {}
        # a CPU verifier's int32 claim stays int32 when only requantisations read it (as z: their first input, or a
        # residual addition's second), which the native kernels compute in int64 from it; else it is widened
        keep32 = (dtype == torch.int32 and x.device.type == "cpu" and visit is None and native_kernels.int32_claims())
        readers = self.graph.readers() if keep32 else {}

        def requant_only(name: str) -> bool:
            ops = readers.get(name, ())
            return bool(ops) and name != self.graph.output_name and all(
                isinstance(r, CheapOp) and r.name not in consumers and getattr(r.fn, "_pvi_requant", None) is not None
                and r.inputs.index(name) == len(r.inputs) - 1 and r.inputs.count(name) == 1 for r in ops)
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
                    # V1 (V-V1N part 4): a cut op's sent values may also be int8 or int16 (cut_protocol keeps them
                    # so once they are in range); they are range-checked below and widened for their readers
                    if z is None or m is None or tuple(z.shape) != (op.n_rows, m) or (
                            z.dtype != dtype and not (op.name in cut_ops and z.dtype in _CUT_NARROW)):
                        return None
                    if op.name not in own:          # a table's rows of int8 weights, else the range check
                        lo, hi = (-128, 127) if op.name in self.tables else (1 - Z_BOUND, Z_BOUND - 1)
                        if op.name in cut_ops:      # V1: the sent requantised values (a, or the residual's Delta)
                            c = cut_ops[op.name]
                            lo, hi = (c.lo_, c.hi) if c.kind == "requant" else cut_delta_range(c.mult, c.sh)
                        if not _check_bounds(z, lo, hi, pending):
                            return None
                    if op.name in looked and not _check_bounds(xin, 0, op.n_in - 1, pending):   # ids of the table
                        return None
                    if op.name in cut_ops and not _check_bounds(xin, -op.max_input - 1, op.max_input + 1, pending):
                        return None              # V1: the input of a cut op within the claim bound's (defence in depth)
                    ranged[op.name] = _stamp(z)
                    if visit is None:
                        inputs[op.name] = xin
                    else:
                        visit(op, z, xin)
                    y = op.fold(z, xin)
                    keep = keep32 and op.name not in cut_ops and op.layout != "conv" and requant_only(op.output)
                    env[op.output] = y if y.dtype == torch.int64 or keep else y.to(torch.int64)
                elif op.name in consumers:          # V1: the requantisation the cut op's values already are
                    if consumers[op.name].kind == "requant":
                        env[op.output] = env[op.inputs[0]]
                    else:
                        res_max = int(op.params["requant"]["res_max"])
                        env[op.output] = (env[op.inputs[0]] + env[op.inputs[1]]).clamp_(-res_max, res_max)
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
            out = {}
            for name, (chi, z) in pairs.items():       # a CPU verifier: one native pass over each claim
                got = native_kernels.field_matmul(chi, z)
                if got is not None:
                    out[name] = got
            return {**out, **self._field_products({k: v for k, v in pairs.items() if k not in out}, Z_BOUND)}
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
        native = (not use8 and bool(mats) and us[mats[0].name].device.type == "cpu"
                  and native_kernels.cheap_enabled(us[mats[0].name]))        # any input width (exact for every K)
        for op in mats:
            xin = inputs[op.name]
            if (op.layout == "linear" and (use8 or native or op.n_in <= _RHS_CHUNK)
                    and us[op.name].device == xin.device):
                shared.setdefault(id(xin), (xin, []))[1].append(op)
            else:
                out[op.name] = _rhs(op, us[op.name], xin)
        if use8:
            return out, [self._rhs_int8(xin, ops, us, out) for xin, ops in shared.values()]
        for key, (xin, ops) in list(shared.items()):    # a CPU verifier: one native pass over each input
            k = ops[0].n_in
            names = tuple(op.name for op in ops)
            ut = self._kept_or_built(("u64",) + names, [us[n] for n in names],
                                     lambda ts, k=k: torch.cat([t[:, :k] for t in ts]))
            y = native_kernels.field_matmul(ut, xin.reshape(-1, k).T)
            if y is None:
                continue
            for op, part_rows in zip(ops, y.split([us[op.name].shape[0] for op in ops])):
                out[op.name] = (part_rows + us[op.name][:, k:]) % P if op.has_bias else part_rows
            del shared[key]
        for key, (xin, ops) in list(shared.items()):    # wider than the batched limb products allow: one by one
            if ops[0].n_in > _RHS_CHUNK:
                for op in ops:
                    out[op.name] = _rhs(op, us[op.name], xin)
                del shared[key]
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
    # pvi.v0.streaming).  Each weight op is checked as soon as its claim is on the device:
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

"""V1's cut: which weight operations are cut, their windows, and the GKR instances.

A weight operation ``l`` whose output ``Z = A [X; 1]`` is read only by a requantisation is *cut*: the prover sends
the requantised values ``s`` (the int8 ``a = rq(Z)`` of a requantisation, or the ``Delta = shift(Z)`` that a
residual addition adds) instead of the claims ``Z``, and proves with the windowed logUp of :mod:`pvi.v1.logup` that
every entry lies in its window, ``lo(s) <= Z < lo(s) + W(s)``.  Here ``shift(z) = (m z + 2^(sh-1)) >> sh`` with the
public multiplier ``m`` and shift ``sh`` (30 in the paper), ``lo(v) = ceil((v 2^sh - 2^(sh-1)) / m)`` is the smallest
``z`` with ``shift(z) = v``, and ``W(v) = lo(v + 1) - lo(v)``.  A clamped entry (``a = +-127`` with ``Z`` outside the
window of ``a``) is an *exception*: the prover lists its exact ``Z``, and its window is that single value (width 1).

The cut set is a public function of the graph, the commitment plan and the query's column counts
(:meth:`CutPlan.from_graph`), absorbed into the statement through :meth:`CutPlan.digest`.  An operation is cut if it
satisfies the plan rules:

* (P1) it is a linear weight operation (not an embedding, and not the one that computes the graph's output) whose
  output is read by exactly one cheap operation built by :func:`pvi.core.graph.requant_fn` with
  ``params["requant"]``: a requantisation (``kind`` ``"requant"``) whose clamp lies within ``[-128, 127]``, or a
  residual addition (``kind`` ``"residual"``) that reads it as its second input only;
* (P2) it claims all ``T`` columns (the single-column operations of the pruned last block and the head-side
  operations of a generation query stay clear);
* (P3) its window widths lie in ``[2, 2^16]``: ``floor(2^sh / m) >= 2`` and ``ceil(2^sh / m) <= 2^16``;
* (P4) its honest claims are below ``2^29`` in magnitude (``pvi.v0.graph_commitment.commit_graph`` refuses every
  committed operation that fails it; it is evaluated here when the graph, or a mode K or Kpre verifier, holds the
  weights);
* (P5) each window width is shared by fewer than ``2^29`` cut entries, so that honest multiplicities stay below
  ``p``; while a width is shared by more, the operation with the most rows among those with that width is made
  clear;
* under a commitment plan, the members of a transposed (``col``) matrix are cut together or not at all.

**Instances.** The units are, in graph order, the transposed matrices (with all their members) and the other cut
operations one by one (in Kpre, which has no commitment plan, every cut operation); consecutive units are packed
greedily into instances of at most ``2^lmax`` entries (a larger unit forms an instance of its own).  An instance
stacks its operations by rows; the entry in row ``row`` and column ``col`` is the leaf ``row 2^n_col + col``, with
``n_col = ceil(log2 T)``.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property

import torch

from pvi.core import extfield as ef
from pvi.core.graph import CheapOp, IntGraph, MatOp, canonical
from pvi.v1 import logup

__all__ = ["CutOp", "CutPlan", "lo", "windows", "shift", "requant_values", "split", "split_counts", "public_lw",
           "public_lw_f64",
           "delta_range",
           "instance_bits", "LMAX", "CLAIM_LIMIT"]

LMAX = 27                             # the default instance bound: at most 2^27 leaves
CLAIM_LIMIT = 1 << 29                 # pvi.v0.params.Z_BOUND: the claim bound of (D1) and (P4)
_COUNT_LIMIT = 1 << 29                # (P5)
_VERSION = b"pvi/cut/v1"


# ------------------------------------------------------------------ windows (exact int64)
def shift(z: torch.Tensor, m: int, sh: int = 30) -> torch.Tensor:
    """``floor((z m + 2^(sh-1)) / 2^sh)`` (int64)."""
    return (z.to(torch.int64) * m + (1 << (sh - 1))) >> sh


def lo(v: torch.Tensor, m: int, sh: int = 30) -> torch.Tensor:
    """``ceil((v 2^sh - 2^(sh-1)) / m)``: the smallest ``z`` with ``shift(z) = v``."""
    return -((-((v.to(torch.int64) << sh) - (1 << (sh - 1)))) // m)


_TABLE_SPAN = 1 << 20
"""Values spanning at most this many integers take their windows from a table of ``lo`` over their range."""


def windows(s: torch.Tensor, m: int, sh: int = 30,
            span: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(L, W)``: ``lo(s)`` and ``lo(s + 1) - lo(s)``, the windows of the values ``s``.  Values spanning at most
    ``2^20`` integers (a requantisation's int8 values, and typically a residual operation's ``Delta``) are looked up
    in a table of ``lo`` over their range instead of being divided entry by entry: the same integers.  ``span``:
    bounds the values are known to lie in (a requantisation's clamp), which saves finding them."""
    if s.numel():
        a, b = span if span is not None else (int(v) for v in torch.aminmax(s))
        if b - a <= _TABLE_SPAN:
            tab = lo(torch.arange(a, b + 2, dtype=torch.int64, device=s.device), m, sh)
            i = s.to(torch.int64) - a
            lower = tab[i]
            return lower, tab[i + 1] - lower
    lower = lo(s, m, sh)
    return lower, lo(s.to(torch.int64) + 1, m, sh) - lower


def delta_range(m: int, sh: int = 30) -> tuple[int, int]:
    """``R_Delta = [shift(1 - 2^29), shift(2^29 - 1)]``: the shifted values of claims in ``(-2^29, 2^29)``, the
    range that (D1) checks for a residual operation's sent values."""
    t = torch.tensor([1 - CLAIM_LIMIT, CLAIM_LIMIT - 1])
    a, b = shift(t, m, sh).tolist()
    return a, b


@dataclass(frozen=True)
class CutOp:
    """One cut operation: an ``N x K`` matrix (``N x (K + 1)`` with ``has_bias``) whose output the cheap operation
    ``consumer`` requantises (``kind``, multiplier ``mult``, shift ``sh``, and for a requantisation the clamp
    ``[lo_, hi]``); its ``N T`` entries occupy the rows from ``row_offset`` of instance ``instance``; it is committed
    in ``matrix``, with layout ``"row"`` or the transposed ``"col"`` (``"row"`` without a commitment plan)."""
    name: str
    kind: str
    mult: int
    sh: int
    lo_: int
    hi: int
    n_rows: int
    n_in: int
    has_bias: bool
    consumer: str
    instance: int
    row_offset: int
    matrix: str
    layout: str

    @property
    def widths(self) -> tuple[int, int]:
        """``(floor(2^sh / m), ceil(2^sh / m))``: the widths of the normal windows."""
        return (1 << self.sh) // self.mult, -(-(1 << self.sh) // self.mult)


@dataclass(frozen=True)
class CutPlan:
    """The cut plan of one query: ``T`` columns, the instance bound ``lmax``, the cut operations in row order, the
    table's widths and every instance's operation names."""
    T: int
    lmax: int
    ops: tuple[CutOp, ...]
    widths: tuple[int, ...]                       # the table's widths, ascending, with 1
    instances: tuple[tuple[str, ...], ...]       # operation names per instance, in row order

    # -------------------------------------------------------------- construction
    @classmethod
    def from_graph(cls, graph: IntGraph, columns: list[int] | None, T: int, plan=None, *,
                   lmax: int = LMAX, weights: dict | None = None) -> "CutPlan":
        """The cut plan of ``graph`` on a query of ``T`` tokens.  ``columns``: every weight operation's column count,
        in operation order, as ``claim_columns`` of the graph or of the verifier gives them (``None``: ``T`` for
        every operation).  ``plan``: the commitment plan, as a :class:`pvi.commit.plans.CommitmentPlan` or as its
        transposed matrices ``{name: members}`` (what a verifier's public data give), or ``None`` (Kpre, or one tree
        per operation).  ``weights`` (``{op: (W, b)}``, a mode K or Kpre verifier's): (P4) is evaluated from them
        when the graph is a public one, without weights in its operations; in mode C the verifier holds no weights
        and (P4) is a property of the commitment (``pvi.v0.graph_commitment.commit_graph`` refuses every operation
        that fails it)."""
        col_of = _col_matrices(plan)
        mats = graph.mat_ops
        cols = dict(zip((op.name for op in mats), columns if columns is not None else [T] * len(mats)))
        readers = graph.readers()
        cand: dict[str, tuple[MatOp, CheapOp, dict]] = {}
        for op in mats:
            c = _consumer(op, readers, graph)
            if c is None or cols.get(op.name) != T:                                     # (P1), (P2)
                continue
            params = c.params["requant"]
            m, sh = int(params["mult"]), int(params["shift"])
            if m <= 0 or (1 << sh) // m < 2 or -(-(1 << sh) // m) > 1 << 16:          # (P3)
                continue
            # (P1): with (P3), a clamp within int8 keeps |lo(s)| <= 129 * 2^16 (the no-wrap lemma's |L| < 2^29 + 2^16)
            if params["kind"] == "requant" and not -128 <= int(params["lo"]) <= int(params["hi"]) <= 127:
                continue
            w_b = (op.weight, op.bias) if op.weight is not None else (weights or {}).get(op.name)
            if w_b is not None and _claim_bound(op, *w_b) >= CLAIM_LIMIT:              # (P4)
                continue
            cand[op.name] = (op, c, params)

        def closed(names):
            """Drop the members of every transposed matrix that is not cut with all its members, until every such
            matrix is cut whole or not at all."""
            names = dict(names)
            while True:
                drop = {o for members in col_of.values() if any(o not in names for o in members)
                        for o in members if o in names}
                if not drop:
                    return names
                for o in drop:
                    del names[o]

        cand = closed(cand)
        while True:                                                                    # (P5)
            count: dict[int, int] = {}
            for op, c, params in cand.values():
                for w in set(_widths(params)):
                    count[w] = count.get(w, 0) + op.n_rows * T
            bad = [w for w, n in count.items() if n >= _COUNT_LIMIT]
            if not bad:
                break
            w = min(bad)
            worst = max((o for o, (op, c, p) in cand.items() if w in _widths(p)),
                        key=lambda o: (cand[o][0].n_rows, o))
            del cand[worst]
            cand = closed(cand)

        # the units in graph order, then the instances
        units, placed = [], set()
        for op in mats:
            if op.name not in cand or op.name in placed:
                continue
            mtx = next((m for m, members in col_of.items() if op.name in members), None)
            if mtx is not None:
                units.append((mtx, "col", list(col_of[mtx])))
                placed.update(col_of[mtx])
            else:
                units.append((op.name, "row", [op.name]))
                placed.add(op.name)
        instances, cur, cur_n = [], [], 0
        for unit in units:
            n = sum(cand[o][0].n_rows for o in unit[2]) * T
            if cur and cur_n + n > 1 << lmax:
                instances.append(cur)
                cur, cur_n = [], 0
            cur.append(unit)
            cur_n += n
        if cur:
            instances.append(cur)
        ops, names = [], []
        for b, inst in enumerate(instances):
            row, inst_names = 0, []
            for matrix, layout, members in inst:
                for o in members:
                    op, c, p = cand[o]
                    ops.append(CutOp(o, p["kind"], int(p["mult"]), int(p["shift"]), int(p.get("lo", 0)),
                                     int(p.get("hi", 0)), op.n_rows, op.n_in, op.has_bias, c.name, b, row,
                                     matrix, layout))
                    row += op.n_rows
                    inst_names.append(o)
            names.append(tuple(inst_names))
        widths = sorted({1} | {w for o in ops for w in o.widths})
        return cls(T, lmax, tuple(ops), tuple(widths), tuple(names))

    # -------------------------------------------------------------- public quantities
    def op(self, name: str) -> CutOp:
        return self._by_name[name]

    @cached_property
    def _by_name(self) -> dict[str, CutOp]:
        return {o.name: o for o in self.ops}

    def names(self) -> frozenset[str]:
        return frozenset(o.name for o in self.ops)

    def instance_ops(self, beta: int) -> list[CutOp]:
        return [o for o in self.ops if o.instance == beta]

    def rows(self, beta: int) -> int:
        return sum(o.n_rows for o in self.instance_ops(beta))

    def n_vars(self, beta: int) -> tuple[int, int]:
        """``(n_row, n_col)`` of instance ``beta`` (at least one variable in all)."""
        n_col = max(0, (self.T - 1).bit_length())
        n_row = max(0, (self.rows(beta) - 1).bit_length())
        if n_row + n_col == 0:
            n_row = 1
        return n_row, n_col

    def n_leaves(self) -> int:
        """Real leaves: every cut entry."""
        return sum(o.n_rows for o in self.ops) * self.T

    def table_size(self) -> int:
        return sum(self.widths)

    def table(self) -> tuple[torch.Tensor, dict[int, int]]:
        """The table values ``[|T|, 8]`` and each width's offset (:func:`pvi.v1.logup.table`)."""
        return logup.table(self.widths)

    def digest(self) -> bytes:
        """What the transcript absorbs as ``b"cut"``: the version, the extension field's parameters, ``lmax``,
        ``T``, the table's widths and every cut operation with its placement."""
        out = [_VERSION]
        canonical([ef.D, ef.BETA, self.lmax, self.T, list(self.widths),
                    [[o.name, o.kind, o.mult, o.sh, o.lo_, o.hi, o.n_rows, o.n_in, o.has_bias, o.consumer, o.instance,
                      o.row_offset, o.matrix, o.layout] for o in self.ops],
                    [list(i) for i in self.instances]], out)
        return hashlib.sha256(b"".join(out)).digest()


def _col_matrices(plan) -> dict[str, tuple[str, ...]]:
    """``{transposed matrix: its members}`` of a commitment plan, of such a dict, or of ``None``."""
    if plan is None:
        return {}
    if hasattr(plan, "matrices"):
        return {m.name: tuple(m.members) for m in plan.matrices if m.layout == "col"}
    return {name: tuple(members) for name, members in plan.items()}


def _widths(params: dict) -> tuple[int, int]:
    m, sh = int(params["mult"]), int(params["shift"])
    return (1 << sh) // m, -(-(1 << sh) // m)


def _claim_bound(op: MatOp, weight: torch.Tensor, bias: torch.Tensor | None) -> int:
    """``pvi.v0.params.claim_bound`` of a linear operation with these weights: the largest honest ``|z|``."""
    w = weight.to(torch.int64)
    bound = (op.max_input + 1) * w.abs().sum(1)
    if bias is not None:
        bound = bound + bias.to(torch.int64).abs()
    return int(bound.max())


def _consumer(op: MatOp, readers: dict, graph: IntGraph) -> CheapOp | None:
    """(P1): the one requantising reader of ``op``'s output, built by :func:`pvi.core.graph.requant_fn`, else
    ``None``."""
    if op.layout != "linear" or op.output == graph.output_name:
        return None
    rs = readers.get(op.output, [])
    if len(rs) != 1 or not isinstance(rs[0], CheapOp):
        return None
    c = rs[0]
    params = c.params.get("requant")
    if not isinstance(params, dict) or getattr(c.fn, "_pvi_requant", None) != params:
        return None
    if params.get("kind") == "requant":
        ok = c.inputs == (op.output,)
    elif params.get("kind") == "residual":
        ok = len(c.inputs) == 2 and c.inputs[1] == op.output and c.inputs[0] != op.output
    else:
        ok = False
    return c if ok else None


# ------------------------------------------------------------------ the witness and the public windows
def requant_values(z: torch.Tensor, op: CutOp) -> torch.Tensor:
    """``s``: ``clamp(shift(z), lo, hi)`` for a requantisation, ``shift(z)`` (``Delta``) for a residual
    addition."""
    s = shift(z, op.mult, op.sh)
    return s.clamp(op.lo_, op.hi) if op.kind == "requant" else s


def split(z: torch.Tensor, op: CutOp):
    """The prover's view of an operation's claims ``z`` ``[N, T]``: ``(s, delta, width, exc_idx, exc_z)``, the sent
    values, every entry's residue ``z - L`` and window width (``0 <= delta < width``), and the exceptions (flat
    indices in row-major order, ascending, and their ``z``): the entries of a requantisation whose ``z`` lies outside
    the normal window of their clamped value."""
    z = z.to(torch.int64)
    s = requant_values(z, op)
    lower, width = windows(s, op.mult, op.sh)
    exc = (z < lower) | (z >= lower + width)
    idx = torch.nonzero(exc.reshape(-1)).reshape(-1)
    exc_z = z.reshape(-1)[idx]
    lower = torch.where(exc, z, lower)
    width = torch.where(exc, torch.ones_like(width), width)
    return s, z - lower, width, idx, exc_z


def split_counts(z: torch.Tensor, op: CutOp, offsets: dict[int, int], size: int):
    """:func:`split` on ``z``'s device, and the operation's honest table counts ``[size]`` (int64, same device): the
    key of an entry is ``delta`` plus the table offset of its width, one of the two normal widths or 1."""
    s, delta, width, idx, exc_z = split(z, op)
    w_lo, w_hi = op.widths
    off = torch.where(width == 1, offsets[1], torch.where(width == w_lo, offsets[w_lo], offsets[w_hi]))
    counts = torch.bincount((delta + off).reshape(-1), minlength=size)
    return s, delta, width, idx, exc_z, counts


def public_lw_f64(s: torch.Tensor, exc_idx: torch.Tensor, exc_z: torch.Tensor, op: CutOp) -> torch.Tensor:
    """:func:`public_lw` for range-checked values, as one float64 matrix ``[L ; W]`` ``[2N, T]`` (exact: every entry
    is below ``2^31``), gathered from the window table directly in float64 (what the baseline final check (F3)
    multiplies)."""
    if s.numel() == 0:
        return torch.zeros(2 * s.shape[0], s.shape[1], dtype=torch.float64, device=s.device)
    a, b = (op.lo_, op.hi) if op.kind == "requant" else (int(v) for v in torch.aminmax(s))
    if b - a > _TABLE_SPAN:
        lower, width = public_lw(s, exc_idx, exc_z, op)
        return torch.cat([lower, width]).to(torch.float64)
    tab = lo(torch.arange(a, b + 2, dtype=torch.int64, device=s.device), op.mult, op.sh)
    ltab, wtab = tab[:-1].to(torch.float64), (tab[1:] - tab[:-1]).to(torch.float64)
    i = s.to(torch.int64) - a
    out = torch.cat([ltab[i], wtab[i]])
    if exc_idx.numel():
        n = s.numel()
        flat = out.view(-1)
        flat[exc_idx] = exc_z.to(torch.float64)
        flat[n + exc_idx] = 1.0
    return out


def public_lw(s: torch.Tensor, exc_idx: torch.Tensor, exc_z: torch.Tensor, op: CutOp, *, in_range: bool = False):
    """The verifier's ``(L, W)`` from the sent values and the listed exceptions (already checked by (D4)).
    ``in_range``: the values are known to lie in the operation's range (checked by (D1)), so a requantisation's
    table spans its clamp."""
    span = (op.lo_, op.hi) if in_range and op.kind == "requant" else None
    lower, width = windows(s, op.mult, op.sh, span)
    if exc_idx.numel():
        lower = lower.reshape(-1).clone()
        width = width.reshape(-1).clone()
        lower[exc_idx] = exc_z.to(torch.int64)
        width[exc_idx] = 1
        lower, width = lower.view(s.shape), width.view(s.shape)
    return lower, width


def instance_bits(plan: CutPlan) -> list[int]:
    """Every instance's variable count ``n = n_row + n_col``."""
    return [sum(plan.n_vars(b)) for b in range(len(plan.instances))]

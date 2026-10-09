"""V1's int8 cut (``V1_SPEC.md`` Sec. 2.1-2.3): which weight ops are cut, their windows, and the GKR instances.

A weight op ``l`` whose output ``Z = A [X; 1]`` only feeds a requantisation is *cut*: the prover sends the
requantised values ``s = requant(Z)`` (int8, or the residual's shifted ``Delta``) instead of the int32 ``Z``, and
proves with the windowed logUp of :mod:`logup` that every entry lies in its window,
``lo(s) <= Z < lo(s) + W(s)`` with ``lo(v) = ceil((v 2^30 - 2^29) / m)`` the smallest ``z`` with
``shift(z) = (z m + 2^29) >> 30 = v``.  Clamped entries (``a = +-127`` with ``Z`` outside the normal window) are
*exceptions*: the prover lists their exact ``Z`` and their window is that single value (width 1).

The cut set is a public function of the graph, the commitment plan and the query's columns (:meth:`CutPlan.from_graph`):

* **P1** the op is a linear weight op (not an embedding) whose output is read by exactly one cheap op, built by
  :func:`graph.requant_fn` with ``params["requant"]`` (``kind`` ``requant`` or ``residual``; a residual reads it
  as its second input only);
* **P2** it claims ``T`` columns (the pruned block's one-column ops and generation's head-side ops stay clear);
* **P3** its windows have ``2 <= floor(2^30 / m)`` and ``ceil(2^30 / m) <= 2^16``;
* **P4** honest claims are below ``2^29`` (``protocol.commit_graph`` enforces it for every committed op; checked
  here when the graph carries weights);
* **P5** per window width, the cut entries with that width number fewer than ``2^29`` (honest multiplicities stay
  below ``p``); else the largest ops with it are made clear;
* under a commitment plan, a col-layout matrix is cut only with all its members.

**Instances.** Units are the committed matrices in graph order (in Kpre every cut op); an instance is a maximal
run of consecutive units with at most ``2^lmax`` entries.  Its ops are stacked by rows; leaf
``row 2^n_col + col`` (``n_col = ceil(log2 T)``).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import cached_property

import torch

from . import extfield as ef
from . import logup
from .graph import CheapOp, IntGraph, MatOp, _canonical

__all__ = ["CutOp", "CutPlan", "lo", "windows", "shift", "requant_values", "split", "split_counts", "public_lw",
           "delta_range",
           "instance_bits", "LMAX", "CLAIM_LIMIT"]

LMAX = 27
CLAIM_LIMIT = 1 << 29                 # protocol.Z_BOUND
_COUNT_LIMIT = 1 << 29                # P5
_VERSION = b"pvi/cut/v1"


# ------------------------------------------------------------------ windows (exact int64)
def shift(z: torch.Tensor, m: int, sh: int = 30) -> torch.Tensor:
    """``floor((z m + 2^(sh-1)) / 2^sh)`` (int64)."""
    return (z.to(torch.int64) * m + (1 << (sh - 1))) >> sh


def lo(v: torch.Tensor, m: int, sh: int = 30) -> torch.Tensor:
    """``ceil((v 2^sh - 2^(sh-1)) / m)``: the smallest ``z`` with ``shift(z) = v``."""
    return -((-((v.to(torch.int64) << sh) - (1 << (sh - 1)))) // m)


_TABLE_SPAN = 1 << 20


def windows(s: torch.Tensor, m: int, sh: int = 30, span: tuple[int, int] | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """``(L, W)``: ``lo(s)`` and ``lo(s + 1) - lo(s)``, the window of the values ``s``.  Values spanning at most
    ``2^20`` (every requant op's int8 values; most residual ops' Delta) are looked up in a table of ``lo`` over their
    range instead of dividing per entry: the same integers.  ``span``: bounds the values are known to lie in (a
    requant op's clamp), which saves finding them."""
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
    """``R_Delta``: the shifted values of claims in ``(-2^29, 2^29)``."""
    t = torch.tensor([1 - CLAIM_LIMIT, CLAIM_LIMIT - 1])
    a, b = shift(t, m, sh).tolist()
    return a, b


@dataclass(frozen=True)
class CutOp:
    """One cut weight op: ``N x (K + 1)`` (``has_bias``), requantised by ``consumer`` (``kind``, multiplier
    ``mult``, shift ``sh``, clamp ``[lo_, hi]`` for requant); entries ``N T`` at rows ``row_offset ..`` of
    ``instance``; committed in ``matrix`` (layout ``row`` or ``col``; ``row`` without a plan)."""
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
        """``(floor(2^sh / m), ceil(2^sh / m))``: the normal windows' widths."""
        return (1 << self.sh) // self.mult, -(-(1 << self.sh) // self.mult)


@dataclass(frozen=True)
class CutPlan:
    T: int
    lmax: int
    ops: tuple[CutOp, ...]
    widths: tuple[int, ...]                       # the table's widths, ascending, with 1
    instances: tuple[tuple[str, ...], ...]       # op names per instance, in row order

    # -------------------------------------------------------------- construction
    @classmethod
    def from_graph(cls, graph: IntGraph, columns: list[int] | None, T: int, plan=None, *,
                   lmax: int = LMAX) -> "CutPlan":
        """The cut plan of ``graph`` on a query of ``T`` tokens (``columns``: every weight op's column count, in
        op order, as ``IntGraph.claim_columns`` gives them; ``None`` takes ``T`` for every op) under the commitment
        plan ``plan``: a ``plans.CommitmentPlan``, or its col matrices as ``{name: members}`` (what a verifier's
        publics give), or ``None`` (Kpre, or one tree per op)."""
        col_of = _col_matrices(plan)
        mats = graph.mat_ops
        cols = dict(zip((op.name for op in mats), columns if columns is not None else [T] * len(mats)))
        readers = graph.readers()
        cand: dict[str, tuple[MatOp, CheapOp, dict]] = {}
        for op in mats:
            c = _consumer(op, readers, graph)
            if c is None or cols.get(op.name) != T:                                     # P1, P2
                continue
            params = c.params["requant"]
            m, sh = int(params["mult"]), int(params["shift"])
            if m <= 0 or (1 << sh) // m < 2 or -(-(1 << sh) // m) > 1 << 16:          # P3
                continue
            if op.weight is not None and _claim_bound(op) >= CLAIM_LIMIT:              # P4
                continue
            cand[op.name] = (op, c, params)

        def closed(names):
            """Drop the members of col matrices not cut with all their members, until stable."""
            names = dict(names)
            while True:
                drop = {o for members in col_of.values() if any(o not in names for o in members)
                        for o in members if o in names}
                if not drop:
                    return names
                for o in drop:
                    del names[o]

        cand = closed(cand)
        while True:                                                                    # P5
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

        # units in graph order, then instances
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
        """The table values ``[|T|, 8]`` and each width's offset (:func:`logup.table`)."""
        return logup.table(self.widths)

    def digest(self) -> bytes:
        """What the transcript absorbs as ``b"cut"``: the version, ``F``'s parameters, ``lmax``, ``T``, the table's
        widths and every cut op with its placement."""
        out = [_VERSION]
        _canonical([ef.D, ef.BETA, self.lmax, self.T, list(self.widths),
                    [[o.name, o.kind, o.mult, o.sh, o.lo_, o.hi, o.n_rows, o.n_in, o.has_bias, o.consumer, o.instance,
                      o.row_offset, o.matrix, o.layout] for o in self.ops],
                    [list(i) for i in self.instances]], out)
        return hashlib.sha256(b"".join(out)).digest()


def _col_matrices(plan) -> dict[str, tuple[str, ...]]:
    """``{col matrix: its members}`` of a commitment plan, of such a dict, or of ``None``."""
    if plan is None:
        return {}
    if hasattr(plan, "matrices"):
        return {m.name: tuple(m.members) for m in plan.matrices if m.layout == "col"}
    return {name: tuple(members) for name, members in plan.items()}


def _widths(params: dict) -> tuple[int, int]:
    m, sh = int(params["mult"]), int(params["shift"])
    return (1 << sh) // m, -(-(1 << sh) // m)


def _claim_bound(op: MatOp) -> int:
    w = op.weight.to(torch.int64)
    bound = (op.max_input + 1) * w.abs().sum(1)
    if op.bias is not None:
        bound = bound + op.bias.abs()
    return int(bound.max())


def _consumer(op: MatOp, readers: dict, graph: IntGraph) -> CheapOp | None:
    """P1: the one requantising reader of ``op``'s output, built by ``graph.requant_fn``, else ``None``."""
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
    """``s``: ``clamp(shift(z), lo, hi)`` for a requant op, ``shift(z)`` (Delta) for a residual."""
    s = shift(z, op.mult, op.sh)
    return s.clamp(op.lo_, op.hi) if op.kind == "requant" else s


def split(z: torch.Tensor, op: CutOp):
    """The prover's view of an op's claims ``z`` ``[N, T]``: ``(s, delta, width, exc_idx, exc_z)``: the sent
    values, every entry's residue ``z - L`` and window width (``0 <= delta < width``), and the exceptions (flat
    indices in row-major order, ascending, and their ``z``): requant entries whose ``z`` is outside the normal
    window of their clamped value."""
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
    """:func:`split` on ``z``'s device, plus the op's honest table counts ``[size]`` (int64, same device): the key of
    an entry is ``delta`` plus the table offset of its width, one of the op's two normal widths or 1."""
    s, delta, width, idx, exc_z = split(z, op)
    w_lo, w_hi = op.widths
    off = torch.where(width == 1, offsets[1], torch.where(width == w_lo, offsets[w_lo], offsets[w_hi]))
    counts = torch.bincount((delta + off).reshape(-1), minlength=size)
    return s, delta, width, idx, exc_z, counts


def public_lw(s: torch.Tensor, exc_idx: torch.Tensor, exc_z: torch.Tensor, op: CutOp, *, in_range: bool = False):
    """The verifier's ``(L, W)`` from the sent values and the listed exceptions (already checked).  ``in_range``:
    the values are known to be in the op's range (derive checked them), so a requant op's table spans its clamp."""
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

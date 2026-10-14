"""Security parameters of a query: Freivalds repetitions and opened columns for a target level, the
soundness bound they give, and the bound on an honest claim."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch

from pvi.commit.plans import CommitmentPlan, column_bits, column_error_log2
from pvi.core.field import LOG2_P
from pvi.core.graph import MatOp

Z_BOUND = 1 << 29
"""Claims must satisfy ``-2**29 < z < 2**29``.  Because ``2 * 2**29 < p``, two
in-range integers with the same residue mod ``p`` are equal, so checking
``Z = A X`` mod ``p`` pins down the integer claim the verifier then uses."""
CUT_NARROW = (torch.int8, torch.int16)
"""The dtypes besides derive's own in which a V1 query's cut ops may hold their sent values
(:mod:`pvi.v1.protocol`)."""


@dataclass(frozen=True)
class SecurityParams:
    """A query's security parameters: the target level ``lam`` (bits), the Freivalds repetitions ``reps``
    (``r``), the code's ``rate`` (a weight op's codeword length is ``rate * next_pow2(k)``) and the opened
    ``columns`` (``t``), Fiat--Shamir with its ``grinding_bits`` margin, and V1's options."""
    lam: float
    reps: int
    rate: int
    columns: int
    fiat_shamir: bool = False
    grinding_bits: int = 64
    group_columns: tuple[tuple[str, int], ...] = ()
    """Under a commitment plan, ``(group, t)`` per group: its members' largest exact ``t``
    (``rate`` is then 0 -- each op's is in the plan -- and ``columns`` the largest ``t``)."""
    cut: bool = False
    """V1's int8 cut (:mod:`pvi.v1.protocol`): requantised values sent, windows proved by logUp-GKR."""
    cut_lmax: int = 27
    """V1: the target size of a GKR instance, ``2**cut_lmax`` leaves: units are packed into instances of at most this
    size, and a unit larger than it is an instance of its own (the default is :data:`pvi.v1.cut.LMAX`; a smaller value
    bounds the prover's memory)."""

    def columns_for(self, group: str) -> int:
        return dict(self.group_columns)[group]


def params_for(lam: float, n_checks: int, *, rate: int = 4, fiat_shamir: bool = False,
               grinding_bits: int = 64, plan: CommitmentPlan | None = None, cut: bool = False,
               cut_lmax: int = 27) -> SecurityParams:
    """Smallest ``(r, t)`` whose union bound over ``n_checks`` ops is ``<= 2**-lam``.

    With Fiat--Shamir the prover can grind offline, so ``grinding_bits`` more are
    required (security against ``2**grinding_bits`` hash evaluations).  Under a commitment
    ``plan`` (:mod:`pvi.commit.plans`) ``r`` is the same, and each group opens its members'
    largest exact ``t`` for the same per-op budget ``2**-beta``.  With ``cut`` (V1) two more terms share the
    budget: the logUp point and the GKR instances (the paper's Theorem ``thm:v1``), each far below it.
    """
    bits = column_bits(lam, n_checks + (2 if cut else 0), fiat_shamir, grinding_bits)
    reps = max(1, math.ceil(bits / LOG2_P))
    extra = {"cut": cut, "cut_lmax": cut_lmax}
    if plan is None:
        columns = max(1, math.ceil(bits / math.log2(rate)))
        return SecurityParams(lam, reps, rate, columns, fiat_shamir, grinding_bits, **extra)
    groups = plan.group_columns(bits)
    return SecurityParams(lam, reps, 0, max(t for _, t in groups), fiat_shamir, grinding_bits, groups, **extra)


def soundness_bits(params: SecurityParams, shapes: list[tuple[int, int]], mode: str = "C",
                   columns: list[int] | None = None) -> float:
    """``-log2`` of the total soundness error: the sum over the checks of ``p**-r`` (Freivalds) plus,
    in mode C, the column term ``prod_{i<t} (m-1-i)/(n-i)``.  ``shapes`` holds ``(m, n)`` per check --
    each op's ``(k, n)`` under the per-op commitment, and one entry per committed matrix of a plan
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
    """A bound on the ``|z|`` an honest execution of ``op`` can produce: per output row ``i``,
    ``(max_input + 1) sum_j |w_ij| + |b_i|`` (every input has ``|x| <= max_input + 1``), and for an
    embedding op the row's largest ``|w_ij|`` plus ``|b_i|``."""
    w = op.weight.to(torch.int64)
    bound = w.abs().amax(1) if op.layout == "embed" else (op.max_input + 1) * w.abs().sum(1)
    if op.bias is not None:
        bound = bound + op.bias.abs()
    return int(bound.max())

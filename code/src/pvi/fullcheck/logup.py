"""The windowed logUp argument of V1 (``V1_SPEC.md`` Sec. 2.2-2.5): the table, the multiplicities, the leaves of a
GKR instance, the padding indicator, and the final identity (checks D5 ``cut_multiplicities``, C1/M2 ``cut_gkr``
and F4 ``cut_logup``).

A cut entry with residue ``delta`` and window width ``W`` is the leaf ``v = delta + x W`` of ``F = F_{p^8}``
(:mod:`extfield`).  It lies in the table ``T = {t + x w : w in widths, 0 <= t < w}`` iff ``0 <= delta < W``
(``1`` and ``x`` are independent over ``F_p``; an honest input bounds ``|delta| < 2^30 + 2^16``, so a negative
``delta`` maps to ``p + delta > 2^16 >= W``).  The prover sends integer
multiplicities ``m_t`` and proves, with one fractional-sum GKR per instance (:mod:`logup_gkr`), the roots of
``sum_i 1 / (alpha - v_i)``; the verifier checks ``sum_beta p0 / q0 = sum_t m_t / (alpha - tau_t)``.

Because a proof can hold more than ``p`` leaves, plain logUp would let ``p`` copies of an off-table value cancel.
**Lemma 1** (``V1_SPEC.md`` Sec. 3, tested by brute force on a small field): if ``0 <= m_t < p`` and
``sum_t m_t = N`` as integers, the identity of rational functions forces every leaf into the table.  So
:func:`multiplicities_ok` checks both conditions; both are load-bearing.
"""

from __future__ import annotations

import torch

from . import extfield as ef
from . import logup_gkr as gkr
from .field import P

__all__ = ["table", "multiplicities", "multiplicities_ok", "leaves", "indicator", "table_fraction", "logup_ok",
           "prove_instances", "verify_instances"]


def table(widths) -> tuple[torch.Tensor, dict[int, int]]:
    """The table values ``[|T|, 8]`` in the order (``w`` ascending, ``t`` ascending) and each width's offset."""
    ws = sorted(set(int(w) for w in widths))
    if not ws or ws[0] < 1:
        raise ValueError("table widths must be positive")
    offsets, at, parts = {}, 0, []
    for w in ws:
        offsets[w] = at
        tag = torch.zeros(w, ef.D, dtype=torch.int64)
        tag[:, 0] = torch.arange(w)
        tag[:, 1] = w
        parts.append(tag)
        at += w
    return torch.cat(parts), offsets


def multiplicities(delta: torch.Tensor, width: torch.Tensor, offsets: dict[int, int], size: int) -> torch.Tensor:
    """The honest counts ``[size]`` (int64) of the entries ``delta``, ``width`` (integer tensors of one shape,
    every ``0 <= delta < width``, every width a key of ``offsets``)."""
    keys = torch.zeros_like(delta, dtype=torch.int64)
    for w, off in offsets.items():
        keys = torch.where(width == w, delta.to(torch.int64) + off, keys)
    return torch.bincount(keys.reshape(-1), minlength=size)


def multiplicities_ok(m: torch.Tensor, size: int, total: int) -> bool:
    """Check D5: ``size`` counts, each ``0 <= m_t < p``, summing to ``total`` as integers."""
    if tuple(m.shape) != (size,) or m.dtype not in (torch.int32, torch.int64):
        return False
    m = m.to(torch.int64)
    if bool((m < 0).any()) or bool((m >= P).any()):
        return False
    return int(m.sum()) == int(total)                   # < size * p < 2^63


def leaves(delta: torch.Tensor, width: torch.Tensor, n_col: int, n: int, alpha: torch.Tensor):
    """The leaf layer ``(p, q)`` ``[2^n, 8]`` of an instance whose real entries are ``delta``, ``width``
    ``[R, T]`` (its ops stacked by rows): leaf ``row 2^n_col + col`` is ``(1, alpha - (delta + x width))`` for
    ``row < R``, ``col < T`` and ``(0, 1)`` elsewhere."""
    rows, cols = delta.shape
    if cols > 1 << n_col or rows > 1 << (n - n_col):
        raise ValueError("the entries do not fit the instance")
    v = torch.zeros(1 << (n - n_col), 1 << n_col, ef.D, dtype=torch.int64)
    v[:rows, :cols, 0] = delta.to(torch.int64) % P
    v[:rows, :cols, 1] = width.to(torch.int64) % P
    real = torch.zeros(1 << (n - n_col), 1 << n_col, dtype=torch.bool)
    real[:rows, :cols] = True
    v, real = v.reshape(-1, ef.D), real.reshape(-1)
    p = torch.zeros_like(v)
    p[real, 0] = 1
    q = torch.where(real[:, None], ef.sub(alpha, v), ef.const(1))
    return p, q


def indicator(rho: torch.Tensor, rows: int, cols: int, n_col: int) -> torch.Tensor:
    """``I(rho) = sum over the real leaves of eq(rho, leaf)``: ``I_rows(r_row) I_cols(r_col)`` with
    ``rho = (r_col || r_row)``, in ``O(n)``."""
    return ef.mul(ef.prefix_eq_sum(rho[n_col:], rows), ef.prefix_eq_sum(rho[:n_col], cols))


def table_fraction(m: torch.Tensor, alpha: torch.Tensor, tau: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``sum_t m_t / (alpha - tau_t)`` as one fraction (no inversion); zero counts are skipped."""
    nz = torch.nonzero(m).reshape(-1)
    return ef.frac_sum(ef.lift(m[nz]), ef.sub(alpha, tau[nz]))


def logup_ok(roots, m: torch.Tensor, alpha: torch.Tensor, tau: torch.Tensor) -> bool:
    """Check F4: ``sum_beta p0 / q0 = sum_t m_t / (alpha - tau_t)`` by cross-multiplication (every ``q0`` is
    non-zero, which :func:`verify_instances` has checked)."""
    if not roots:
        return not bool(m.any())
    num, den = ef.frac_sum(torch.stack([r[0] for r in roots]), torch.stack([r[1] for r in roots]))
    tn, td = table_fraction(m, alpha, tau)
    return ef.equal(ef.mul(num, td), ef.mul(tn, den))


def prove_instances(instances, ch, prefix: str = "cut/") -> list:
    """One :func:`logup_gkr.prove` per instance ``(p, q)``, in order, under the labels ``prefix + f"{beta}/"``."""
    return [gkr.prove(p, q, ch, f"{prefix}{b}/") for b, (p, q) in enumerate(instances)]


def verify_instances(trs, ns, ch, prefix: str = "cut/"):
    """The outputs of :func:`logup_gkr.verify` for every instance (``ns``: their variable counts), or ``None`` if
    one fails, has another size, or has a zero root denominator (check ``cut_gkr``)."""
    if len(trs) != len(ns):
        return None
    outs = []
    for b, (tr, n) in enumerate(zip(trs, ns)):
        if tr.n != n:
            return None
        out = gkr.verify(tr, ch, f"{prefix}{b}/")
        if out is None or not bool(out[1].any()):
            return None
        outs.append(out)
    return outs

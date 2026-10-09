"""CPU stand-ins for the Triton kernels of ``pvi.fullcheck.gkr_triton`` (for tests on hosts without CUDA/Triton).

Each kernel's job is restated with ``extfield`` on canonical values: the planes ``[8, n]`` hold Montgomery values
``v R mod p`` (``R = 2^32``), converted with ``R^-1`` on the way in and ``R`` on the way out.  A partial-sum kernel puts
its whole sum in block 0 (the host sums the blocks).  ``install(module)`` swaps these into the module, so its host code
-- the round logic, the bottom-layer path, the transcript -- runs unchanged on the CPU.
"""

from __future__ import annotations

import types

import torch

from pvi.fullcheck import extfield as ef
from pvi.fullcheck.field import P

R = (1 << 32) % P
R_INV = pow(1 << 32, -1, P)


def _canon(planes: torch.Tensor) -> torch.Tensor:
    """``[8, n]`` Montgomery int32 -> ``[n, 8]`` canonical int64."""
    return (planes.to(torch.int64).T % P) * R_INV % P


def _mont(v: torch.Tensor) -> torch.Tensor:
    """``[n, 8]`` canonical -> ``[8, n]`` Montgomery int64."""
    return (v.to(torch.int64) % P * R % P).T


def _scal(vals) -> torch.Tensor:
    return torch.tensor([int(x) % P * R_INV % P for x in vals], dtype=torch.int64)


def _store(dst: torch.Tensor, cols, src_mont: torch.Tensor) -> None:
    dst[:, cols] = src_mont.to(torch.int32)


class _K:
    def __init__(self, fn):
        self.fn = fn

    def __getitem__(self, grid):
        return self.fn


def _leaf(d, w, p, q, *args, **kw):
    al = _scal(args[:8])
    rows, T, n_col, total, stride = args[8:13]
    i = torch.arange(total)
    row, col = i >> n_col, i & ((1 << n_col) - 1)
    real = (row < rows) & (col < T)
    src = (row * T + col).clamp(max=max(rows * T - 1, 0))
    dv = torch.where(real, d.reshape(-1).to(torch.int64)[src], torch.zeros_like(i))
    wv = torch.where(real, w.reshape(-1).to(torch.int64)[src], torch.zeros_like(i))
    qv = torch.zeros(total, 8, dtype=torch.int64)
    qv[:, 0] = torch.where(real, (al[0] - dv) % P, torch.ones_like(i))
    qv[:, 1] = torch.where(real, (al[1] - wv) % P, torch.zeros_like(i))
    for k in range(2, 8):
        qv[:, k] = torch.where(real, al[k].expand(total), torch.zeros_like(i))
    pv = torch.zeros(total, 8, dtype=torch.int64)
    pv[:, 0] = real.to(torch.int64)
    p[:, :total] = _mont(pv).to(torch.int32)
    q[:, :total] = _mont(qv).to(torch.int32)


def _combine(pi, qi, po, qo, m_out, s_in, s_out, **kw):
    pl, pr = _canon(pi[:, 0:2 * m_out:2]), _canon(pi[:, 1:2 * m_out:2])
    ql, qr = _canon(qi[:, 0:2 * m_out:2]), _canon(qi[:, 1:2 * m_out:2])
    po[:, :m_out] = _mont(ef.add(ef.mul(pl, qr), ef.mul(pr, ql))).to(torch.int32)
    qo[:, :m_out] = _mont(ef.mul(ql, qr)).to(torch.int32)


def _eq(ti, to, *args, **kw):
    r, s_ = _scal(args[:8]), _scal(args[8:16])
    m, s_in, s_out, inter = args[16:20]
    t = _canon(ti[:, :m])
    lo, hi = _mont(ef.mul(t, s_)), _mont(ef.mul(t, r))
    y = torch.arange(m)
    lo_at, hi_at = (2 * y, 2 * y + 1) if inter == 1 else (y, y + m)
    _store(to, lo_at, lo)
    _store(to, hi_at, hi)


def _pairs(a, half):
    return _canon(a[:, 0:2 * half:2]), _canon(a[:, 1:2 * half:2])


def _at(a, b, t):
    return a if t == 0 else b if t == 1 else ef.add(b, ef.sub(b, a))


def _round2(e, pl, pr, ql, qr, out, *args, **kw):
    lam = _scal(args[:8])
    half = args[8]
    E = _canon(e[:, :half])
    arrs = [_pairs(x, half) for x in (pl, pr, ql, qr)]
    out.zero_()
    for ti, t in enumerate((0, 1, 2)):
        PL, PR, QL, QR = (_at(a, b, t) for a, b in arrs)
        F = ef.add(ef.mul(PL, QR), ef.mul(QL, ef.add(PR, ef.mul(lam, QR))))
        out[0, ti] = _mont(ef.mul(E, F).sum(0, keepdim=True) % P)[:, 0]


def _round_bot(e, ql, qr, out, *args, **kw):
    lam = _scal(args[:8])
    half = args[8]
    E = _canon(e[:, :half])
    arrs = [_pairs(x, half) for x in (ql, qr)]
    out.zero_()
    for ti, t in enumerate((0, 1, 2)):
        QL, QR = (_at(a, b, t) for a, b in arrs)
        F = ef.add(ef.add(QL, QR), ef.mul(lam, ef.mul(QL, QR)))
        out[0, ti] = _mont(ef.mul(E, F).sum(0, keepdim=True) % P)[:, 0]
    out[0, 3] = _mont(E.sum(0, keepdim=True) % P)[:, 0]


def _fold(n_arrays):
    def run(*args, **kw):
        ins, outs = args[:n_arrays], args[n_arrays:2 * n_arrays]
        r = _scal(args[2 * n_arrays:2 * n_arrays + 8])
        half = args[2 * n_arrays + 8]
        for a, o in zip(ins, outs):
            lo, hi = _pairs(a, half)
            o[:, :half] = _mont(ef.add(lo, ef.mul(ef.sub(hi, lo), r))).to(torch.int32)
    return run


def install(gt, monkeypatch) -> None:
    """Swap the mocks into the module ``gt`` (``pvi.fullcheck.gkr_triton``) for one test."""
    monkeypatch.setattr(gt, "triton", types.SimpleNamespace(cdiv=lambda a, b: -(-a // b)), raising=False)
    for name, fn in (("_leaf_kernel", _leaf), ("_combine_kernel", _combine), ("_eq_kernel", _eq),
                     ("_round2_kernel", _round2), ("_round_bot_kernel", _round_bot),
                     ("_fold4_kernel", _fold(4)), ("_fold2_kernel", _fold(2))):
        monkeypatch.setattr(gt, name, _K(fn), raising=False)

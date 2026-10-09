"""The GPU GKR prover's host logic (``pvi.fullcheck.gkr_triton``) on the CPU, its kernels replaced by exact CPU
stand-ins (``gkr_triton_mock``): the round structure (``h(0), h(1), h(2)``, the eq factor taken out), the suffix eq
tables, and the bottom layer's path (full rows, ``T = 2^n_col``) give the eager prover's transcript byte for byte.
The kernels themselves are checked against the same transcripts on a CUDA host (``test_gkr_triton.py``)."""

from __future__ import annotations

import pytest
import torch

from pvi.fullcheck import extfield as ef
from pvi.fullcheck import gkr_triton as gt
from pvi.fullcheck import logup
from pvi.fullcheck import logup_gkr as gkr
from pvi.fullcheck import protocol as proto
from pvi.fullcheck.field import P

import gkr_triton_mock


@pytest.fixture
def mocked(monkeypatch):
    gkr_triton_mock.install(gt, monkeypatch)
    return gt


def _ch():
    c = proto.Challenger(fiat_shamir=True)
    c.absorb(b"statement", b"gkr triton cpu")
    return c


@pytest.mark.parametrize("n", [1, 2, 3, 6, 9])
def test_uploaded_leaves(mocked, n):
    g = torch.Generator().manual_seed(n)
    p = torch.randint(0, P, (1 << n, ef.D), generator=g)
    q = torch.randint(0, P, (1 << n, ef.D), generator=g)
    a = gkr.prove(p, q, _ch(), "cut/0/")
    b = mocked.prove(p, q, _ch(), "cut/0/", device="cpu")
    assert a.to_bytes() == b.to_bytes() and torch.equal(a.point, b.point)


@pytest.mark.parametrize("rows,cols,n_col", [(1, 4, 2), (3, 8, 3), (5, 16, 4), (8, 16, 4), (13, 32, 5),
                                             (7, 9, 4), (2, 2, 1), (6, 64, 6)])
def test_instance_leaves_with_and_without_the_bottom_path(mocked, rows, cols, n_col):
    # cols == 2^n_col (n_col >= 2): the bottom layer's column rounds on the real rows only; otherwise the generic path
    g = torch.Generator().manual_seed(rows * 100 + cols)
    w = torch.randint(1, 1 << 12, (rows, 1), generator=g).expand(rows, cols).clone()
    d = (torch.rand(rows, cols, generator=g) * w).floor().to(torch.int64)
    alpha = torch.randint(0, P, (ef.D,), generator=g)
    n = n_col + max(0, (rows - 1).bit_length())
    pq = logup.leaves(d, w, n_col, n, alpha)
    a = gkr.prove(*pq, _ch(), "cut/2/")
    b = mocked.prove_leaves(d, w, n_col, n, alpha, _ch(), "cut/2/", device="cpu")
    assert a.to_bytes() == b.to_bytes() and torch.equal(a.point, b.point)

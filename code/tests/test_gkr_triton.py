"""The Triton GKR prover (``pvi.fullcheck.gkr_triton``) gives the eager prover's transcript byte for byte
(``V1_SPEC.md`` test T2 on the GPU), from uploaded leaves and from leaves built on the device; skipped without CUDA
and Triton."""

from __future__ import annotations

import pytest
import torch

from pvi.fullcheck import extfield as ef
from pvi.fullcheck import gkr_triton as gt
from pvi.fullcheck import logup
from pvi.fullcheck import logup_gkr as gkr
from pvi.fullcheck import protocol as proto
from pvi.fullcheck.field import P

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and gt.available("cuda")), reason="needs CUDA and Triton")


def _ch():
    c = proto.Challenger(fiat_shamir=True)
    c.absorb(b"statement", b"gkr triton")
    return c


@pytest.mark.parametrize("n", [1, 2, 3, 5, 8, 11, 14])
def test_random_leaves_give_the_eager_transcript(n):
    g = torch.Generator().manual_seed(n)
    p = torch.randint(0, P, (1 << n, ef.D), generator=g)
    q = torch.randint(0, P, (1 << n, ef.D), generator=g)
    a = gkr.prove(p, q, _ch(), "cut/0/")
    b = gt.prove(p, q, _ch(), "cut/0/")
    assert a.to_bytes() == b.to_bytes()
    assert torch.equal(a.point, b.point.cpu())


@pytest.mark.parametrize("rows,cols,n_col", [(1, 1, 1), (5, 6, 3), (13, 64, 6), (100, 33, 6), (7, 2048, 11)])
def test_device_leaves_give_the_eager_transcript(rows, cols, n_col):
    g = torch.Generator().manual_seed(rows * cols)
    w = torch.randint(1, 1 << 16, (rows, 1), generator=g).expand(rows, cols).clone()
    d = (torch.rand(rows, cols, generator=g) * w).floor().to(torch.int64)
    alpha = torch.randint(0, P, (ef.D,), generator=g)
    n = n_col + max(0, (rows - 1).bit_length())
    if n == 0:
        n = 1
    pq = logup.leaves(d, w, n_col, n, alpha)
    a = gkr.prove(*pq, _ch(), "cut/3/")
    b = gt.prove_leaves(d, w, n_col, n, alpha, _ch(), "cut/3/")
    assert a.to_bytes() == b.to_bytes()
    out = gkr.verify(gkr.Transcript.from_bytes(b.to_bytes(), n), _ch(), "cut/3/")
    assert out is not None

"""The Triton GKR prover (``pvi.v1.gkr_triton``) gives the eager prover's transcript byte for byte
(``tests/test_logup_gkr.py`` on the GPU), from uploaded leaves and from leaves built on the device (among them an
instance whose rows fill it, no padding row, as the full-size instances of OPT-1.3B at 2,048 tokens); skipped
without CUDA and Triton."""

from __future__ import annotations

import pytest
import torch

import pvi.v0 as proto
from pvi.core import extfield as ef
from pvi.core.field import P
from pvi.v1 import gkr, logup
from pvi.v1 import gkr_triton as gt

pytestmark = pytest.mark.skipif(not (torch.cuda.is_available() and gt.available("cuda")),
                                reason="needs CUDA and Triton")


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


@pytest.mark.parametrize("rows,cols,n_col", [(1, 1, 1), (5, 6, 3), (13, 64, 6), (16, 64, 6), (100, 33, 6),
                                             (7, 2048, 11)])
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


@pytest.mark.parametrize("rows", [512, 300])
def test_the_low_memory_path_gives_the_same_transcript_at_96_bytes_per_leaf(rows):
    # 2^20 leaves, full rows of 2,048 columns (300 rows: padding rows up to 512); the default path peaks at 148 bytes
    # per leaf (pvi.v1.protocol._TREE_BYTES), the low-memory path at 96
    n_col, n = 11, 20
    g = torch.Generator().manual_seed(rows)
    w = torch.randint(1, 1 << 16, (rows, 1), generator=g, dtype=torch.int32).expand(rows, 1 << n_col).contiguous()
    d = (torch.rand(rows, 1 << n_col, generator=g) * w).floor().to(torch.int32)
    alpha = torch.randint(0, P, (ef.D,), generator=g)
    out, peaks = {}, {}
    for low in (False, True):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        base = torch.cuda.memory_allocated()
        out[low] = gt.prove_leaves(d, w, n_col, n, alpha, _ch(), "cut/5/", low_memory=low).to_bytes()
        torch.cuda.synchronize()
        peaks[low] = (torch.cuda.max_memory_allocated() - base) / (1 << n)
    assert out[False] == out[True]
    assert peaks[True] <= 97 and peaks[False] <= 149

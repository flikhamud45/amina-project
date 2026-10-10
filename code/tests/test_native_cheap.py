"""The native kernels of the CPU verifier's recomputed operations (plan B5) give the torch ops' integers.

Each case runs the public helper twice, with the native kernels and with ``PVI_NATIVE_CHEAP`` off (the torch steps),
on contiguous operands and on the transposed claim views that a weight op's fold hands its readers, with values at
and beyond the ranges the protocol checks (int64 arithmetic wraps the same way in both)."""
import pytest
import torch

from pvi.fullcheck import graph as gr
from pvi.fullcheck import native_kernels
from pvi.fullcheck import transformer as tr

pytestmark = pytest.mark.skipif(not native_kernels.available(), reason="no C++ compiler (or not Linux)")


def _both(fn, monkeypatch):
    got = fn()
    monkeypatch.setattr(native_kernels, "_CHEAP", False)
    want = fn()
    monkeypatch.setattr(native_kernels, "_CHEAP", True)
    return got, want


def _claim_view(g, rows, cols, lo, hi):
    """A weight op's fold output: the transposed view ``Z.T`` of a contiguous claim ``Z [rows, cols]``, as
    ``[1, cols, rows]``."""
    z = torch.randint(lo, hi, (rows, cols), generator=g, dtype=torch.int64)
    return z.T.reshape(1, cols, rows)


@pytest.mark.parametrize("rows,cols", [(4096, 2048), (130, 7), (1, 1), (11008, 65)])
@pytest.mark.parametrize("transposed", [True, False])
def test_requant_and_residual(rows, cols, transposed, monkeypatch):
    g = torch.Generator().manual_seed(rows + cols)
    z = _claim_view(g, rows, cols, -(1 << 29), 1 << 29)
    if not transposed:
        z = z.contiguous()
    m = torch.tensor(2_000_000_123, dtype=torch.int64)
    got, want = _both(lambda: gr.requant(z, m, 30, -127, 127), monkeypatch)
    assert torch.equal(got, want) and got.is_contiguous()
    a = torch.randint(-(1 << 22), 1 << 22, z.shape, generator=g, dtype=torch.int64)
    got, want = _both(lambda: gr.residual_add(a, z, m, 30, (1 << 22) - 1), monkeypatch)
    assert torch.equal(got, want)
    huge = z * (1 << 40)                                                  # products that wrap int64
    got, want = _both(lambda: gr.requant(huge, m, 30, -(1 << 40), 1 << 40), monkeypatch)
    assert torch.equal(got, want)


def test_requant_fn_and_glu_product(monkeypatch):
    g = torch.Generator().manual_seed(3)
    a = torch.randint(-127, 128, (1, 300, 640), generator=g, dtype=torch.int64)
    b = torch.randint(-127, 128, (1, 300, 640), generator=g, dtype=torch.int64)
    m = torch.tensor(123_456_789, dtype=torch.int64)
    got, want = _both(lambda: gr.requant(a * b, m, 30, -127, 127), monkeypatch)
    assert torch.equal(got, want)
    fn = gr.requant_fn(dict(kind="residual", mult=987_654_321, shift=30, res_max=(1 << 22) - 1))
    zt = _claim_view(g, 640, 300, -(1 << 29), 1 << 29)
    got, want = _both(lambda: fn(a, zt), monkeypatch)
    assert torch.equal(got, want)


@pytest.mark.parametrize("center", [False, True])
@pytest.mark.parametrize("shape", [(1, 2048, 4096), (1, 7, 2048, 128), (1, 1, 768)])
def test_norm(center, shape, monkeypatch):
    g = torch.Generator().manual_seed(sum(shape) + center)
    x = torch.randint(-(1 << 22), 1 << 22, shape, generator=g, dtype=torch.int64)
    x[..., 0, :] = 0                                                      # a zero row: sigma = 1
    x[..., -1, :3] = (1 << 22) - 1
    gain = torch.randint(1, 1 << 23, (shape[-1],), generator=g, dtype=torch.int64)
    got, want = _both(lambda: tr._norm_int(x, gain, center), monkeypatch)
    assert torch.equal(got, want)
    wrapped = torch.full((1, 3, 64), 1 << 40, dtype=torch.int64)          # sums of squares wrap: torch's rules
    got, want = _both(lambda: tr._norm_int(wrapped, gain[:64], center), monkeypatch)
    assert torch.equal(got, want)


@pytest.mark.parametrize("transposed", [True, False])
def test_lut(transposed, monkeypatch):
    g = torch.Generator().manual_seed(5)
    x = _claim_view(g, 11008, 33, -300, 300)
    if not transposed:
        x = x.contiguous()
    table = torch.randint(-127, 128, (256,), generator=g, dtype=torch.int64)
    got, want = _both(lambda: tr._lut(x, table), monkeypatch)
    assert torch.equal(got, want)


@pytest.mark.parametrize("t,heads,dh,offset", [(2048, 32, 128, 0), (5, 8, 64, 0), (1, 32, 128, 2047), (3, 4, 80, 10)])
def test_rope(t, heads, dh, offset, monkeypatch):
    g = torch.Generator().manual_seed(t + heads + dh)
    x = torch.randint(-127, 128, (1, t, heads, dh), generator=g, dtype=torch.int64)
    x[0, 0, 0] = 127
    got, want = _both(lambda: tr._rope(x, 10000.0, offset), monkeypatch)
    assert torch.equal(got, want)

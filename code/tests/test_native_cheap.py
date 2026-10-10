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


@pytest.mark.parametrize("r,n,m", [(5, 4096, 2048), (7, 70_000, 3), (5, 11008, 1), (1, 1, 1), (3, (1 << 15) + 1, 65)])
@pytest.mark.parametrize("transposed", [False, True])
def test_field_matmul(r, n, m, transposed):
    from pvi.fullcheck.field import P, field_matmul_mod
    g = torch.Generator().manual_seed(r * n + m)
    chi = torch.randint(0, P, (r, n), generator=g, dtype=torch.int64)
    chi[0, :5] = P - 1
    z = torch.randint(-(1 << 31) + 1, 1 << 31, (m, n) if transposed else (n, m), generator=g, dtype=torch.int64)
    z.view(-1)[:3] = (1 << 31) - 1
    zz = z.T if transposed else z
    want = field_matmul_mod(chi, zz.contiguous(), right_bound=1 << 31)
    assert torch.equal(native_kernels.field_matmul(chi, zz), want)
    zz = zz.clone() if not transposed else zz
    bad = zz.contiguous()
    bad.view(-1)[-1] = 1 << 31                                            # out of range: the caller's products
    assert native_kernels.field_matmul(chi, bad) is None
    chi[-1, -1] = P
    assert native_kernels.field_matmul(chi, zz) is None


@pytest.mark.parametrize("shapes", [[(4096, 2048), (11008, 2048)], [(5, 70_000), (300, 1)], [(768, 3000)]])
def test_native_unpack_decodes_like_numpy(shapes, monkeypatch):
    from pvi.fullcheck import claimcodec
    g = torch.Generator().manual_seed(len(shapes))
    claims = [(torch.randn(n, m, generator=g) * 40_000).round().to(torch.int64)
              + torch.randint(-3000, 3000, (n, 1), generator=g) for n, m in shapes]
    claims[0][0, :5] = (1 << 29) - 1
    buf = claimcodec.encode(claims)
    rows, cols = [n for n, _ in shapes], [m for _, m in shapes]
    got = claimcodec.decode_torch(buf, rows, cols, workers=8)
    monkeypatch.setattr(native_kernels, "_DECODE", False)
    want = claimcodec.decode_torch(buf, rows, cols, workers=8)
    assert all(torch.equal(a, b) and torch.equal(a, c) for a, b, c in zip(got, want, claims))

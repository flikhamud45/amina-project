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


def test_int32_claims_give_the_int64_results(monkeypatch):
    """A CPU verifier's int32 claims (and their transposed fold views) through requantisation, residual addition
    and the field products give the integers of the same claims as int64."""
    from pvi.fullcheck.field import P
    g = torch.Generator().manual_seed(11)
    z64 = torch.randint(-(1 << 29) + 1, 1 << 29, (4096, 600), generator=g, dtype=torch.int64)
    z64[0, :4] = (1 << 29) - 1
    z32 = z64.to(torch.int32)
    m = torch.tensor(1_987_654_321, dtype=torch.int64)
    for view in (lambda z: z.T.reshape(1, 600, 4096), lambda z: z):
        want = gr.requant(view(z64), m, 30, -127, 127)
        assert torch.equal(gr.requant(view(z32), m, 30, -127, 127), want)
        a = torch.randint(-(1 << 22), 1 << 22, tuple(view(z64).shape), generator=g, dtype=torch.int64)
        assert torch.equal(gr.residual_add(a, view(z32), m, 30, (1 << 22) - 1),
                           gr.residual_add(a, view(z64), m, 30, (1 << 22) - 1))
    chi = torch.randint(0, P, (5, 4096), generator=g, dtype=torch.int64)
    assert torch.equal(native_kernels.field_matmul(chi, z32), native_kernels.field_matmul(chi, z64))
    assert torch.equal(native_kernels.field_matmul(chi[:, :600], z32.T), native_kernels.field_matmul(chi[:, :600], z64.T))
    monkeypatch.setattr(native_kernels, "_CHEAP", False)                  # the torch steps widen int32 claims
    assert torch.equal(gr.requant(z32.T, m, 30, -127, 127), gr.requant(z64.T, m, 30, -127, 127))


def test_native_unpack_on_corrupted_proofs_matches_numpy(monkeypatch):
    """Random bit flips: the native unpack gives the numpy decoder's claims or its rejection, never another
    exception."""
    import numpy as np
    from pvi.fullcheck import claimcodec
    g = torch.Generator().manual_seed(21)
    shapes = [(300, 700), (64, 2000), (5, 1)]
    claims = [(torch.randn(n, m, generator=g) * 3000).round().to(torch.int64) for n, m in shapes]
    blob = claimcodec.encode(claims)
    rows, cols = [n for n, _ in shapes], [m for _, m in shapes]
    rng = np.random.default_rng(5)

    def run(b):
        try:
            return claimcodec.decode(b, rows, cols, workers=4)
        except claimcodec.ClaimCodecError:
            return None

    for _ in range(400):
        b = bytearray(blob)
        for _ in range(int(rng.integers(1, 4))):
            b[int(rng.integers(0, len(b)))] ^= 1 << int(rng.integers(0, 8))
        b = bytes(b)
        got = run(b)
        monkeypatch.setattr(native_kernels, "_DECODE", False)
        want = run(b)
        monkeypatch.setattr(native_kernels, "_DECODE", True)
        assert (got is None) == (want is None)
        assert got is None or all(np.array_equal(x, y) for x, y in zip(got, want))



@pytest.mark.parametrize("hkv,rep,t,tq,dh", [(32, 1, 2048, 2048, 128), (8, 4, 777, 777, 128), (4, 2, 300, 1, 64),
                                             (6, 1, 513, 7, 80)])
def test_native_attention_on_strided_views(hkv, rep, t, tq, dh):
    """The attention kernels on the layouts _attention passes them: [B, T, H, dh] tensors seen as [B, H, T, dh]."""
    g = torch.Generator().manual_seed(t + dh)
    qb = torch.randint(-127, 128, (1, rep * tq, hkv, dh), generator=g, dtype=torch.int64)
    kb = torch.randint(-127, 128, (1, t, hkv, dh), generator=g, dtype=torch.int64)
    vb = torch.randint(-127, 128, (1, t, hkv, dh), generator=g, dtype=torch.int64)
    q, k, v = (x.transpose(1, 2) for x in (qb, kb, vb))
    lut = tr._exp_lut(1 << 20, "cpu")
    want = tr._attention_core(q, k, v, lut, tr._causal_notmask(t, rep, "cpu", tq))
    assert torch.equal(native_kernels.attention_core(q, k, v, lut, tq), want)
    assert torch.equal(native_kernels.attention_core(q.contiguous(), k.contiguous(), v.contiguous(), lut, tq), want)


def test_min_max_and_the_norm_with_large_denominators(monkeypatch):
    from pvi.fullcheck.field import min_max
    g = torch.Generator().manual_seed(17)
    for dtype in (torch.int32, torch.int64):
        a = torch.randint(-(1 << 30), 1 << 30, (300_001,), generator=g).to(dtype)
        a[123] = torch.iinfo(dtype).min
        a[-1] = torch.iinfo(dtype).max
        assert native_kernels.min_max(a) == (int(a.min()), int(a.max())) == min_max(a)
    x = torch.randint(-(1 << 30), 1 << 30, (1, 3, 4), generator=g, dtype=torch.int64)   # sigma << 16 >= 2**40
    gain = torch.randint(1, 1 << 23, (4,), generator=g, dtype=torch.int64)
    got, want = _both(lambda: tr._norm_int(x, gain, False), monkeypatch)
    assert torch.equal(got, want)


@pytest.mark.parametrize("hq,hkv,t,tq,dh", [(32, 32, 2048, 2048, 128), (32, 8, 600, 600, 128), (8, 2, 300, 1, 64),
                                            (6, 6, 513, 7, 80), (12, 4, 64, 64, 64)])
def test_fused_attention_and_output_requantisation(hq, hkv, t, tq, dh, monkeypatch):
    """transformer._attention on a CPU verifier (one AVX-512 pass that also requantises the output into the final
    layout) against the per-group path (native attention core, then the torch requantisation)."""
    g = torch.Generator().manual_seed(hq + t + tq)
    q = torch.randint(-127, 128, (1, tq, hq * dh), generator=g, dtype=torch.int64)
    k = torch.randint(-127, 128, (1, t, hkv * dh), generator=g, dtype=torch.int64)
    v = torch.randint(-127, 128, (1, t, hkv * dh), generator=g, dtype=torch.int64)
    m_s, m_o = 1 << 20, 1_234_567
    got = tr._attention(q, k, v, m_s, m_o, hq, hkv, dh)
    monkeypatch.setattr(native_kernels, "_AVX", False)
    want = tr._attention(q, k, v, m_s, m_o, hq, hkv, dh)
    assert torch.equal(got, want) and got.shape == (1, tq, hq * dh)

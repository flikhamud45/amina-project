"""The GPU verifier's code paths give the integers and verdicts of the reference code.

* The int32 attention (``transformer._attention_core``) against ``reference.attention_heads``, on
  both sides of its fallback conditions, and its exp look-up table against the formula.
* The int8 GEMM products (``field.int8_*``) against ``reference.field_matmul_mod`` on edge values
  (every int32, blocks straddling ``2**16`` terms), in their float64 emulation on the CPU and with
  ``torch._int_mm`` on a GPU, and the fallbacks when an int8 GEMM is missing, inexact or fails.

Tests taking ``device`` also run on CUDA when it is available; there they exercise the int8
tensor cores.
"""

from __future__ import annotations

import pytest
import torch

from pvi.fullcheck import field as fld
from pvi.fullcheck import reference as ref
from pvi.fullcheck import transformer as tr

P = fld.P
INT32_MIN, INT32_MAX = -(1 << 31), (1 << 31) - 1
cuda_only = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    return request.param


def _rand(g, lo, hi, shape):
    return torch.randint(lo, hi, shape, generator=g, dtype=torch.int64)


# -- the int32 attention ------------------------------------------------------------------------

def _qkv(g, b, h, t, dh, pattern):
    q, k, v = (_rand(g, -127, 128, (b, h, t, dh)) for _ in range(3))
    if pattern == "peaked":                      # a few large scores per row
        k[:, :, ::3] = q[:, :, ::3]
    elif pattern == "flat":                      # every score 0: the largest exp sums (tot)
        q.zero_()
        v.fill_(127)
    elif pattern == "extreme":                   # |s| = dh * 127**2, the largest gaps
        q.fill_(127)
        k[..., ::2], k[..., 1::2] = 127, -127
        v.fill_(-127)
    return q, k, v


_CAP_MS = -(-((tr._EXP_ZERO << 30) - (1 << 29)) // tr._LUT_MAX)   # smallest m_s with a table of <= 2**22 gaps
M_S = [1, 1 << 12, _CAP_MS - 1, _CAP_MS, 50_000, 2_600_000, 1 << 24, 1 << 30, (1 << 32) - 1, 1 << 32]


@pytest.mark.parametrize("t", [1, 2, 17, 64, 257])
@pytest.mark.parametrize("m_s", M_S)
def test_attention_heads_match_the_reference_on_both_paths(t, m_s, device):
    g = torch.Generator().manual_seed(t * 31 + m_s % 97)
    for pattern in ("random", "peaked", "flat", "extreme"):
        q, k, v = _qkv(g, 2, 3, t, 16, pattern)
        for m_o in (None, 1 << 22, 123_456_789):
            got = tr._attention_heads(q.to(device), k.to(device), v.to(device), m_s, m_o).cpu()
            assert torch.equal(got, ref.attention_heads(q, k, v, m_s, m_o)), (pattern, m_o)


def test_the_int32_path_applies_exactly_within_its_bounds():
    ok = tr._int32_scores
    assert ok(_CAP_MS, 128, (1 << 15) - 1) and not ok(_CAP_MS - 1, 128, 5)
    assert not ok(2_600_000, 129, 5) and not ok(2_600_000, 16, 1 << 15)
    assert not ok(0, 16, 5) and not ok(-3, 16, 5) and not ok(1 << 32, 16, 5) and ok((1 << 32) - 1, 16, 5)


@pytest.mark.parametrize("m_s", [_CAP_MS, 50_000, 2_600_000, 1 << 24, (1 << 30) + 5, (1 << 32) - 1])
def test_the_exp_lut_matches_the_formula_on_every_gap(m_s):
    lut = tr._exp_lut(m_s, "cpu")
    assert lut.dtype == torch.int32 and int(lut[-1]) == 0 and lut.numel() - 1 == tr._exp_cap(m_s)
    d = torch.arange(lut.numel() + 5000)
    idx = ((d * m_s + (1 << 29)) >> 30).clamp(0, 255)
    assert torch.equal(tr._EXP_TABLE[idx].to(torch.int32), lut[d.clamp(max=lut.numel() - 1)])
    assert (1 << 30) - (1 << 21) >= lut.numel() - 1         # every masked gap looks up LUT[-1] = 0


def test_a_row_of_probabilities_sums_to_at_most_255_plus_half_t():
    # the bound that makes ONE float32 P V GEMM exact: (255 + T/2) * 128 < 2**24 for T < 2**15
    g = torch.Generator().manual_seed(3)
    for t in (1, 5, 300, 4096):
        e = _rand(g, 0, (1 << 15) + 1, (50, t))
        e[:, 0] = 1 << 15
        e[1] = 1 << 15                                       # a flat row: the bound is tight
        tot = e.sum(-1, keepdim=True)
        p = torch.div(e * 510 + tot, 2 * tot, rounding_mode="floor")
        assert int(p.sum(-1).max()) <= 255 + t / 2
    assert (255 + ((1 << 15) - 1) / 2) * 128 < 1 << 24


def test_attention_at_a_real_head_dim_and_prompt_length(device):
    g = torch.Generator().manual_seed(7)
    q, k, v = _qkv(g, 1, 2, 1024, 128, "peaked")
    for m_s in (2_600_000, 1 << 12):                         # the int32 path, the fallback
        got = tr._attention_heads(q.to(device), k.to(device), v.to(device), m_s, None).cpu()
        assert torch.equal(got, ref.attention_heads(q, k, v, m_s, None))


@pytest.mark.parametrize("hq,hkv", [(6, 6), (8, 2), (4, 1)])
@pytest.mark.parametrize("m_s", [2_600_000, 1_000])
def test_grouped_attention_matches_the_reference_on_both_paths(hq, hkv, m_s, device, monkeypatch):
    g = torch.Generator().manual_seed(hq * 10 + hkv)
    b, t, dh = 2, 40, 8
    q = _rand(g, -127, 128, (b, t, hq * dh))
    k, v = (_rand(g, -127, 128, (b, t, hkv * dh)) for _ in range(2))
    want = ref.attention(q, k, v, m_s, 1 << 21, hq, hkv, dh)
    for budget in (tr.ATTN_BYTES, 4 * b * t * t * 3, 8 * b * t * t):   # one group; groups of 3 (int32) or 1
        monkeypatch.setattr(tr, "ATTN_BYTES", budget)
        got = tr._attention(q.to(device), k.to(device), v.to(device), m_s, 1 << 21, hq, hkv, dh).cpu()
        assert torch.equal(got, want), budget


# -- int8 GEMMs -------------------------------------------------------------------------------

@pytest.mark.parametrize("r,n,m", [(5, 8, 8), (5, 1000, 3), (4, 4097, 64), (1, 9000, 1), (5, 70_000, 2), (3, 7, 0),
                                   (2, 1 << 16, 5), (2, (1 << 16) + 1, 1)])
def test_int8_field_matmul_is_exact_for_every_int32(r, n, m, device):
    g = torch.Generator().manual_seed(r * 7 + n + m)
    chi = _rand(g, 0, P, (r, n))
    chi[0], chi[-1, :3] = P - 1, 0
    z = _rand(g, INT32_MIN, INT32_MAX + 1, (n, m))
    if z.numel():
        z[0], z[-1] = INT32_MAX, INT32_MIN
    want = ref.field_matmul_mod(chi, z)
    operand = fld.int8_left(chi.to(device))
    for claims in (z.to(torch.int32), z):                    # int32, or int64 holding int32 values
        assert torch.equal(fld.int8_field_matmul(operand, claims.to(device)).cpu(), want)


@pytest.mark.parametrize("rr,k,m", [(5, 768, 64), (15, 4096, 9), (5, 70_000, 3), (2, 7, 1), (3, 1 << 16, 17),
                                    (1, (1 << 16) + 3, 2), (4, 5, 0)])
def test_int8_small_matmul_is_exact(rr, k, m, device):
    g = torch.Generator().manual_seed(rr + k + m)
    u = _rand(g, 0, P, (rr, k))
    u[0] = P - 1
    x = _rand(g, -128, 128, (m, k))
    if x.numel():
        x[0], x[-1] = -128, 127
    got = fld.int8_small_matmul(fld.int8_right(u.to(device)), x.to(torch.int8).to(device)).cpu()
    assert torch.equal(got, ref.field_matmul_mod(u, x.T))


def test_int8_bytes_decompose_every_int32():
    x = torch.tensor([INT32_MIN, INT32_MIN + 1, -129, -128, -1, 0, 1, 127, 128, 255, 256, P - 1, INT32_MAX])
    x = torch.cat([x, torch.randint(INT32_MIN, INT32_MAX, (1000,), generator=torch.Generator().manual_seed(1))])
    x32 = x.to(torch.int32)
    b = fld.int8_bytes(x32).reshape(-1, 4).to(torch.int64)
    assert torch.equal((b + torch.tensor([128, 128, 128, 0])) @ torch.tensor([1, 1 << 8, 1 << 16, 1 << 24]), x)
    assert torch.equal(fld.int8_bytes(x), fld.int8_bytes(x32))      # int64 holding int32 values
    assert torch.equal(x32, x.to(torch.int32))                       # the input is not modified


def test_the_int8_self_check_rejects_a_missing_or_inexact_gemm(monkeypatch):
    assert fld.int8_ok("cpu") is False                                # never off the GPU
    monkeypatch.setattr(torch, "_int_mm", lambda a, b: (a.double() @ b.double()).to(torch.int32))
    assert fld._int_mm_exact("cpu")
    monkeypatch.setattr(torch, "_int_mm", lambda a, b: (a.double() @ b.double()).clamp(max=(1 << 30) - 1).int())
    assert not fld._int_mm_exact("cpu")                               # saturates at the accumulation bound

    def missing(a, b):
        raise RuntimeError("no int8 GEMM here")

    monkeypatch.setattr(torch, "_int_mm", missing)
    assert not fld._int_mm_exact("cpu")


@cuda_only
def test_int8_gemm_is_checked_once_per_gpu_and_falls_back_when_it_fails(monkeypatch):
    key = f"cuda:{torch.cuda.current_device()}"
    assert fld.int8_ok("cuda") == fld.int8_ok(key) == fld._INT8[key]
    monkeypatch.setitem(fld._INT8, key, True)

    def fails(a, b):
        raise RuntimeError("no int8 GEMM for this layout")

    monkeypatch.setattr(torch, "_int_mm", fails)
    g = torch.Generator().manual_seed(2)
    a, b = _rand(g, -128, 128, (24, 64)).to(torch.int8), _rand(g, -128, 128, (64, 16)).to(torch.int8)
    assert torch.equal(fld.int8_gemm(a.cuda(), b.cuda()).cpu(), (a.double() @ b.double()).to(torch.int32))
    assert fld._INT8[key] is False and not fld.int8_ok(key)          # the products use float64 from now on

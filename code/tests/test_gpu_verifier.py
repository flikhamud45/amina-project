"""The GPU verifier's code paths give the integers and verdicts of the reference code.

* The int32 attention (``transformer._attention_core``) against ``reference.attention_heads``, on
  both sides of its fallback conditions, and its exp look-up table against the formula.

Tests taking ``device`` also run on CUDA when it is available.
"""

from __future__ import annotations

import pytest
import torch

from pvi.fullcheck import reference as ref
from pvi.fullcheck import transformer as tr


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

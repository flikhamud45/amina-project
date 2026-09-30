"""The GPU verifier's code paths give the integers and verdicts of the reference code.

* The int32 attention (``transformer._attention_core``) against ``reference.attention_heads``, on
  both sides of its fallback conditions, and its exp look-up table against the formula.
* The int8 GEMM products (``field.int8_*``) against ``reference.field_matmul_mod`` on edge values
  (every int32, blocks straddling ``2**16`` terms), in their float64 emulation on the CPU and with
  ``torch._int_mm`` on a GPU, and the fallbacks when an int8 GEMM is missing, inexact or fails.
* A GPU verifier's device-resident constants (``IntGraph.with_constants_on``): the prover's graph
  is not modified, and a modified constant is copied again.
* Deferred range checks (an exception after an out-of-range claim is its rejection) and inputs
  that are not int8-valued (the int8 right-hand side falls back).

Tests taking ``device`` also run on CUDA when it is available; there they exercise the int8
tensor cores and the device-resident constants.
"""

from __future__ import annotations

import pytest
import torch

from pvi.fullcheck import field as fld
from pvi.fullcheck import protocol as proto
from pvi.fullcheck import reference as ref
from pvi.fullcheck import transformer as tr
from pvi.fullcheck.graph import CheapOp, IntGraph, MatOp
from pvi.fullcheck.transformer import DecoderConfig, build_decoder

P = fld.P
Z = proto.Z_BOUND
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


# -- a GPU verifier's constants ---------------------------------------------------------------

_TINY = {
    "gpt2": DecoderConfig("tiny-gpt2", 64, 2, 4, 4, 16, 256, 97),
    "llama": DecoderConfig("tiny-llama", 64, 2, 4, 4, 16, 160, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                           bias=False, tied=False),
    "qwen": DecoderConfig("tiny-qwen", 64, 2, 8, 2, 16, 160, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                          rope_theta=1e6, bias=False, qk_norm=True),
}


def _decoder(kind="qwen"):
    graph = build_decoder(_TINY[kind], calib_tokens=12, seed=3)
    return graph, torch.randint(0, 97, (1, 12), generator=torch.Generator().manual_seed(5))


def _tensor_args(fn):
    return [v for v in (*(fn.__defaults__ or ()), *(c.cell_contents for c in fn.__closure__ or ()))
            if torch.is_tensor(v)]


def test_constants_are_copied_to_the_device_once_without_touching_the_prover_graph():
    graph, _ = _decoder()
    public = graph.public()
    before = [(op, op.fn, [t.clone() for t in _tensor_args(op.fn)]) for op in graph.ops if isinstance(op, CheapOp)]
    moved, pairs = public.with_constants_on("meta")
    assert [type(a) for a in moved.ops] == [type(b) for b in public.ops]
    for a, b in zip(public.ops, moved.ops):
        if isinstance(a, MatOp):
            assert a is b
        else:
            assert a is not b and (a.fn is not b.fn) == bool(_tensor_args(a.fn))
            assert all(t.device.type == "meta" for t in _tensor_args(b.fn))
    assert pairs and all(s.device.type == "cpu" and d.device.type == "meta" for s, d in pairs)
    assert len({id(s) for s, _ in pairs}) == len(pairs)                   # each constant copied once
    for op, fn, saved in before:                                          # the prover's graph is as it was
        assert op.fn is fn and all(torch.equal(a, b) for a, b in zip(_tensor_args(fn), saved))
        assert all(t.device.type == "cpu" for t in _tensor_args(fn))


def test_constants_in_closures_are_moved_too():
    t = torch.arange(3)

    def make():
        return lambda a: a + t.to(a.device)

    fn = make()
    moved, pairs = IntGraph([CheapOp("c", ("x",), "c.y", fn=fn)], "x", "c.y").with_constants_on("meta")
    assert fn.__closure__[0].cell_contents is t and len(pairs) == 1
    assert moved.ops[0].fn(torch.zeros(3, device="meta")).device.type == "meta"


def test_a_modified_constant_is_copied_again():
    graph, _ = _decoder()
    v = proto.Verifier(graph.public(), proto.params_for(40, len(graph.mat_ops)), "Kpre")
    src = torch.arange(4)
    dst = src.clone()
    v._consts = [(src, dst, (src._version, dst._version))]
    v._refresh_constants()
    src[0] = 7
    v._refresh_constants()
    assert int(dst[0]) == 7
    dst[1] = 9                                     # the copy itself modified: copied again
    v._refresh_constants()
    assert torch.equal(dst, torch.tensor([7, 1, 2, 3]))


@cuda_only
def test_a_gpu_verifier_keeps_its_constants_on_the_gpu():
    graph, _ = _decoder()
    v = proto.Verifier(graph.public(), proto.params_for(40, len(graph.mat_ops)), "Kpre", device="cuda")
    moved = [t for op in v.graph.ops if isinstance(op, CheapOp) for t in _tensor_args(op.fn)]
    assert moved and all(t.is_cuda for t in moved)
    assert all(not t.is_cuda for op in graph.ops if isinstance(op, CheapOp) for t in _tensor_args(op.fn))


# -- deferred range checks, inputs that are not int8-valued -------------------------------------

def _raise_on_large(a):
    if int(a.abs().max()) >= 1 << 20:
        raise ValueError("large input")
    return a.clamp(-127, 127)


def _guarded_graph():
    """fc1 -> a cheap op that raises on large values -> fc2."""
    g = torch.Generator().manual_seed(4)
    fc1 = MatOp("fc1", ("x",), "fc1.z", weight=_rand(g, -127, 128, (4, 3)).to(torch.int8), layout="linear")
    fc2 = MatOp("fc2", ("g.y",), "fc2.z", weight=_rand(g, -127, 128, (2, 4)).to(torch.int8), layout="linear")
    guard = CheapOp("g", ("fc1.z",), "g.y", fn=_raise_on_large)
    return IntGraph([fc1, guard, fc2], "x", "fc2.z")


@pytest.mark.parametrize("deferred", [False, True])
def test_an_exception_after_an_out_of_range_claim_is_its_rejection(deferred, monkeypatch):
    graph = _guarded_graph()
    x = torch.tensor([[5, -7, 100]])
    _, claims = graph.forward(x)
    v = proto.Verifier(graph.public(), proto.params_for(40, 2), "Kpre")
    monkeypatch.setattr(proto, "_defer", lambda device: deferred)
    assert v.derive(x, claims) is not None
    for value, want in ((Z, None), (-(1 << 62), None), (1 << 21, ValueError)):
        bad = dict(claims, fc1=claims["fc1"].clone())
        bad["fc1"][0, 0] = value                   # out of range; or in range, but the guard raises
        if want is None:
            assert v.derive(x, bad) is None and not v._ranged
        else:
            with pytest.raises(want):
                v.derive(x, bad)


def _wide_input_verifier(mode="Kpre"):
    """One linear op whose input is NOT int8-valued (|x| up to 300)."""
    g = torch.Generator().manual_seed(6)
    w = _rand(g, -127, 128, (5, 16)).to(torch.int8)
    b = _rand(g, -1000, 1000, (5,))
    op = MatOp("fc", ("x",), "fc.z", weight=w, bias=b, layout="linear")
    graph = IntGraph([op], "x", "fc.z")
    x = _rand(g, -300, 301, (2, 3, 16))
    z = w.to(torch.int64) @ x.reshape(-1, 16).T + b[:, None]         # the exact claim
    v = proto.Verifier(graph.public(), proto.params_for(40, 1), mode, weights={"fc": (w, b)})
    v.precompute(proto.Challenger(seed=1))
    chis, us = {"fc": v._pre["fc"][0]}, {"fc": v._pre["fc"][1]}
    return v, x, {"fc": z}, chis, us


@pytest.mark.parametrize("deferred", [False, True])
def test_products_of_inputs_that_are_not_int8_valued_fall_back(deferred, monkeypatch):
    monkeypatch.setattr(proto, "_defer", lambda device: deferred)
    monkeypatch.setattr(proto, "int8_ok", lambda device: True)
    v, x, claims, chis, us = _wide_input_verifier()
    inputs = v.derive(x, claims)
    _, unchecked = v._rhs_all(v.graph.mat_ops, inputs, us)
    assert proto._to_host(unchecked) == [True]
    assert v.check_products(claims, inputs, chis, us) is ref.check_products(v, claims, inputs, chis, us) is True
    bad = {"fc": claims["fc"].clone()}
    bad["fc"][2, 3] += 1
    inputs = v.derive(x, bad)
    assert v.check_products(bad, inputs, chis, us) is ref.check_products(v, bad, inputs, chis, us) is False

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
* The streaming verifier (``Verifier.verify_streaming``, with the wire formats of
  ``pvi.fullcheck.pipeline``): the verdicts and labels of ``run_query``
  on honest and tampered queries, its proof bytes on accepted ones, and its Fiat--Shamir
  transcript.

Tests taking ``device`` also run on CUDA when it is available; there they exercise the int8
tensor cores, the side stream, the events and the pinned memory, and check that the GPU
verifier copies nothing back but its one set of verdicts per check.
"""

from __future__ import annotations

import contextlib
import hashlib

import pytest
import torch

from pvi.fullcheck import field as fld
from pvi.fullcheck import pipeline
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
        raw = tr._attention_heads(q.to(device), k.to(device), v.to(device), m_s)
        assert torch.equal(raw.cpu(), ref.attention_heads(q, k, v, m_s, None)), pattern
        for m_o in (1 << 22, 123_456_789):           # requantised into a strided view, as _attention writes it
            dst = torch.empty(2, t, 3, 16, dtype=torch.int64, device=device).transpose(1, 2)
            tr._write_output(dst, raw, m_o)
            assert torch.equal(dst.cpu(), ref.attention_heads(q, k, v, m_s, m_o)), (pattern, m_o)


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
        got = tr._attention_heads(q.to(device), k.to(device), v.to(device), m_s).cpu()
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


# -- the fused attention kernel (attention_kernels, Triton) ----------------------------------------------------

@cuda_only
@pytest.mark.parametrize("hq,hkv,t,tq,dh", [(4, 4, 64, 64, 64), (8, 2, 300, 300, 128), (4, 4, 2048, 2048, 128),
                                            (8, 8, 513, 1, 80), (6, 2, 1000, 7, 64), (2, 1, 97, 33, 16)])
@pytest.mark.parametrize("pattern", ["random", "peaked", "extreme"])
@pytest.mark.parametrize("m_s", [1 << 20, 2_600_000])
def test_the_fused_attention_kernel_gives_the_int32_cores_integers(hq, hkv, t, tq, dh, pattern, m_s):
    from pvi.fullcheck import attention_kernels
    if not attention_kernels.available("cuda"):
        pytest.skip("no Triton")
    if not tr._int32_scores(m_s, dh, t):
        pytest.skip("the int64 fallback applies")
    g = torch.Generator().manual_seed(hq * t + tq + dh)
    rep = hq // hkv
    q = _rand(g, -127, 128, (1, hkv, rep * tq, dh))
    k, v = _rand(g, -127, 128, (1, hkv, t, dh)), _rand(g, -127, 128, (1, hkv, t, dh))
    if pattern == "peaked":                       # a few large scores per row
        k[:, :, ::3] = 127
        q[..., : dh // 2] = 127
    elif pattern == "extreme":                    # every score at its bound, the largest p v sums
        q.fill_(127), k.fill_(-127), v.fill_(-127)
    q, k, v = q.cuda(), k.cuda(), v.cuda()
    lut = tr._exp_lut(m_s, "cuda")
    want = tr._attention_core(q, k, v, lut, tr._causal_notmask(t, rep, "cuda", tq))
    got = attention_kernels.attention_core(q, k, v, lut, tq)
    assert torch.equal(got, want)
    # strided inputs, as transformer._attention passes them (a transposed [B, T, H, dh] view)
    qs = q.transpose(1, 2).contiguous().transpose(1, 2)
    assert torch.equal(attention_kernels.attention_core(qs, k.transpose(1, 2).contiguous().transpose(1, 2), v, lut, tq),
                       want)


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


@pytest.mark.parametrize("n,k,m", [(24, 768, 64), (17, 4096, 1), (4096, 64, 9), (40, 70_000, 3), (33, 1 << 16, 8)])
def test_int8_product_is_the_exact_weight_product(n, k, m, device):
    # the prover's forward on int8 GEMMs: int8 weights and inputs at the extremes, M padded to 8, K in blocks
    from pvi.fullcheck.graph import int8_product
    g = torch.Generator().manual_seed(n + k + m)
    w, x = _rand(g, -128, 128, (n, k)), _rand(g, -128, 128, (k, m))
    w[0], x[:, 0] = -128, -128
    want = (w.double() @ x.double()).to(torch.int64) if k < 1 << 16 else w @ x
    got = int8_product(w.to(torch.int8).to(device), x.to(torch.int8).to(device))
    assert torch.equal(got.cpu(), want)


def test_int8_product_declines_shapes_the_int8_gemm_cannot_take():
    w8 = torch.ones(16, 64, dtype=torch.int8)
    assert fld.int8_ok("cpu") is False
    from pvi.fullcheck.graph import int8_product
    assert int8_product(w8, torch.ones(64, 8, dtype=torch.int8)) is None              # N <= 16
    assert int8_product(torch.ones(24, 60, dtype=torch.int8), torch.ones(60, 8, dtype=torch.int8)) is None  # K % 8


@pytest.mark.parametrize("m,k,rr", [(24, 768, 5), (4096, 64, 133), (3, 70_000, 2), (1, 7, 1), (17, 1 << 16, 3),
                                    (5, (1 << 16) + 3, 2)])
def test_int8_weight_matmul_is_exact(m, k, rr, device):
    # the prover's products: an opening W @ V[:, C], and a fold chi @ W (W transposed, not contiguous)
    g = torch.Generator().manual_seed(m + k + rr)
    w = _rand(g, -128, 128, (m, k))
    w[0], w[-1] = -128, 127
    v, chi = _rand(g, 0, P, (k, rr)), _rand(g, 0, P, (rr, m))
    v[0], chi[:, 0] = P - 1, P - 1
    w8 = w.to(torch.int8).to(device)
    opened, folded = ref.field_matmul_mod(v.T, w.T).T, ref.field_matmul_mod(chi, w)
    for fn in (fld.int8_weight_matmul, fld.weight_matmul_mod):     # weight_matmul_mod: int8 where int8_ok
        assert torch.equal(fn(w8, v.to(device)).cpu(), opened)
        assert torch.equal(fn(w8.T, chi.T.contiguous().to(device)).T.cpu(), folded)


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
    wire = {"fc": pipeline.wire_claim(claims["fc"])}
    assert v.verify_streaming(x, wire, chis, us) is None
    assert v.verify_streaming(x, {"fc": pipeline.wire_claim(bad["fc"])}, chis, us) == "freivalds"


# -- the streaming verifier ------------------------------------------------------------------------

def test_the_wire_formats():
    g = torch.Generator().manual_seed(8)
    z = _rand(g, 1 - Z, Z, (6, 5))
    w = pipeline.wire_claim(z)
    assert w.dtype == torch.int32 and torch.equal(w.to(torch.int64), z)
    assert pipeline.wire_claim(torch.zeros(3, 0, dtype=torch.int64)).shape == (3, 0)
    wide = z.clone()
    wide[0, 0] = 1 << 31
    for bad in (z.to(torch.int32), wide):          # not narrowed: int64, the claim's size and blob
        got = pipeline.wire_claim(bad)
        assert got.dtype == torch.int64 and torch.equal(got, bad.to(torch.int64))
        assert got.numel() == bad.numel() and proto._tensor_blob(got) == proto._tensor_blob(bad)
    for bad in ("not a tensor", None):
        assert pipeline.wire_claim(bad) is None
    o = _rand(g, 0, P, (7, 3))
    far = o.clone()
    far[0, 0] = (1 << 32) + 5                      # would wrap to 5 in int32
    opened = {"a": (o, ["p"]), "b": (o.to(torch.int32), []), "c": (o[:, 0], []), "d": (far, []), "e": None,
              "f": (o, [], "extra"), "g": [o, ["p"]]}
    got = pipeline.wire_openings(opened)
    for k in "ag":                                 # a tuple or a list of two, as run_query unpacks them
        assert got[k][0].dtype == torch.int32 and got[k][0].is_contiguous() and torch.equal(got[k][0], o.T.int())
        assert got[k][1] == ["p"]
    assert all(got[k][0] is None for k in "bcd")
    assert got["e"] is None and got["f"] == opened["f"]
    rows = _rand(g, -128, 128, (4, 9))                 # a lookup table's claim: int8 rows, a byte per value
    for honest in (rows, rows.to(torch.int32)):
        got = pipeline.wire_rows(honest)
        assert got.dtype == torch.int8 and torch.equal(got.to(torch.int64), rows)
    outside = rows.clone()
    outside[1, 2] = 128
    for bad in (outside, outside.to(torch.int32), rows.to(torch.float32)):   # not narrowed: int64, its size and blob
        got = pipeline.wire_rows(bad)
        assert got.dtype == torch.int64 and proto._tensor_blob(got) == proto._tensor_blob(bad)
    assert pipeline.wire_rows(None) is None
    uploads = pipeline.ClaimUploads({"t": pipeline.wire_rows(rows), "u": pipeline.wire_rows(rows), "z": w},
                                    ["t", "u", "z"], torch.device("cpu"), rows={"t", "z"})
    t, u, zz = (uploads.get(n) for n in "tuz")         # a table's int8 rows widened to int32 after the upload
    assert t.dtype == torch.int32 and torch.equal(t.to(torch.int64), rows) and u.dtype == torch.int8 and zz is w


def _claim_attacks(mats):
    mid = mats[len(mats) // 2]

    def at(op, change):
        def tamper(o, z):
            return change(z.clone()) if o.name == op.name else z
        return {"tamper": tamper}

    def setting(index, value):
        def change(z):
            z.view(-1)[index] = value
            return z
        return change

    def plus_one(z):
        z.view(-1)[z.numel() // 3] += 1
        return z

    return {"honest": {}, "first+1": at(mats[0], plus_one), "mid+1": at(mid, plus_one),
            "last+1": at(mats[-1], plus_one), "Z_BOUND": at(mid, setting(0, Z)),
            "1-Z_BOUND": at(mid, setting(-1, 1 - Z)), "beyond int32": at(mid, setting(0, 1 << 40)),
            "int32 dtype": at(mid, lambda z: z.to(torch.int32))}


def _setup(graph, mode, fiat_shamir=False, device="cpu"):
    params = proto.params_for(40, len(graph.mat_ops), fiat_shamir=fiat_shamir)
    coms = proto.commit_graph(graph, params.rate) if mode == "C" else {}
    prover = proto.Prover(graph, commitments=coms)
    kw = dict(publics={k: c.public for k, c in coms.items()}, weights={op.name: (op.weight, op.bias)
                                                                      for op in graph.mat_ops})
    pair = [proto.Verifier(graph.public(), params, mode, **kw),
            proto.Verifier(graph.public(), params, mode, stream=True, device=device, **kw)]
    if mode == "Kpre":
        for v in pair:
            v.precompute(proto.Challenger(seed=9))
    return prover, pair


def _same_outcome(prover, pair, x, seed, **kw):
    a, b = (proto.run_query(prover, v, x, seed=seed, **kw) for v in pair)
    labels = (a["rejected_at"], b["rejected_at"])
    assert (a["accepted"], a["rejected_at"]) == (b["accepted"], b["rejected_at"]), labels
    assert b["bytes"]["claims"] == a["bytes"]["claims"]
    if a["accepted"]:
        assert b["bytes"] == a["bytes"]
    else:        # the streaming verifier also received the messages run_query no longer asks for
        assert all(a["bytes"][k] in (0, b["bytes"][k]) for k in a["bytes"]), (a["bytes"], b["bytes"])
    assert "verify_total" in b["timings"]
    return a


@pytest.mark.parametrize("kind", list(_TINY))
@pytest.mark.parametrize("mode,fiat_shamir", [("C", False), ("C", True), ("K", False), ("Kpre", False)])
def test_streaming_gives_the_verdicts_of_run_query_on_tampered_claims(kind, mode, fiat_shamir, device):
    graph, x = _decoder(kind)
    prover, pair = _setup(graph, mode, fiat_shamir, device)
    labels = {name: _same_outcome(prover, pair, x, seed=i, forward_kwargs=kw)["rejected_at"]
              for i, (name, kw) in enumerate(_claim_attacks(graph.mat_ops).items())}
    assert labels["honest"] is None and {labels["first+1"], labels["mid+1"], labels["last+1"]} == {"freivalds"}
    assert {labels[k] for k in ("Z_BOUND", "beyond int32", "int32 dtype")} == {"range_or_shape"}
    assert labels["1-Z_BOUND"] == "freivalds"


def _fold_and_open_attacks(graph):
    mats = graph.mat_ops
    victim, other = mats[1].name, mats[-2].name

    def fold(change):
        def patched(fold_):
            return lambda chis: {k: (change(u) if k == victim else u) for k, u in fold_(chis).items()}
        return ("fold", patched)

    def opening(changes):
        def patched(open_):
            return lambda cols: {k: (changes[k](*o) if k in changes else o) for k, o in open_(cols).items()}
        return ("open", patched)

    def plus_one(o, pr):
        o = o.clone()
        o[0, 0] = (o[0, 0] + 1) % P
        return o, pr

    def setting(value):
        def change(o, pr):
            o = o.clone()
            o[0, 0] = value
            return o, pr
        return change

    return {"u+1": fold(lambda u: (u + (torch.arange(u.numel()).reshape(u.shape) == 0)) % P),
            "u=P": fold(lambda u: torch.where(torch.arange(u.numel()).reshape(u.shape) == 0, P, u)),
            "u narrow": fold(lambda u: u[:, :-1]),
            "opened+1": opening({victim: plus_one}), "opened=P": opening({victim: setting(P)}),
            "opened=-1": opening({victim: setting(-1)}), "opened>int32": opening({victim: setting((1 << 32) + 5)}),
            "opened narrow": opening({victim: lambda o, pr: (o[:, :-1], pr)}),
            "opened int32": opening({victim: lambda o, pr: (o.to(torch.int32), pr)}),
            "path forged": opening({victim: lambda o, pr: (o, [bytes(32)] + pr[1:])}),
            "path short": opening({victim: lambda o, pr: (o, pr[:-1])}),
            "path long": opening({victim: lambda o, pr: (o, pr + [bytes(32)])}),
            "as a list": opening({victim: lambda o, pr: [o, pr]}),
            "merkle then code": opening({victim: lambda o, pr: (o, [bytes(32)] + pr[1:]), other: plus_one}),
            "code then narrow": opening({victim: plus_one, other: lambda o, pr: (o[:, :-1], pr)})}


@pytest.mark.parametrize("kind", ["gpt2", "qwen"])
def test_streaming_gives_the_verdicts_of_run_query_on_forged_folds_and_openings(kind, device):
    graph, x = _decoder(kind)
    prover, pair = _setup(graph, "C", device=device)
    real = {"fold": prover.fold, "open": prover.open}
    labels = {}
    for i, (name, (what, patched)) in enumerate(_fold_and_open_attacks(graph).items()):
        setattr(prover, what, patched(real[what]))
        try:
            labels[name] = _same_outcome(prover, pair, x, seed=i)["rejected_at"]
        finally:
            setattr(prover, what, real[what])
    assert labels["u+1"] == labels["u=P"] == labels["u narrow"] == "freivalds"
    assert labels["opened+1"] == "columns_code"
    shape_labels = {labels[k] for k in ("opened=P", "opened=-1", "opened>int32", "opened narrow", "opened int32")}
    assert shape_labels == {"columns_shape"}
    assert labels["path forged"] == labels["path short"] == labels["path long"] == "columns_merkle"
    assert labels["as a list"] is None
    assert labels["merkle then code"] == "columns_merkle" and labels["code then narrow"] == "columns_code"
    # an exception is raised where run_query raises it (a proof entry that is not bytes)
    prover.open = lambda cols: {k: ((o, [0] + pr[1:]) if k == graph.mat_ops[2].name else (o, pr))
                                for k, (o, pr) in real["open"](cols).items()}
    for v in pair:
        with pytest.raises(TypeError):
            proto.run_query(prover, v, x, seed=99)


@pytest.mark.parametrize("fiat_shamir", [False, True])            # u of 2 and of 4 rows
@pytest.mark.parametrize("change", ["extra row", "one row fewer", "3-d", "3-d wide"])
def test_streaming_rejects_a_u_of_the_wrong_shape_at_freivalds(change, fiat_shamir, device):
    # no column check may run on a u whose rows cannot be compared with the opened columns
    graph, x = _decoder("qwen")
    prover, pair = _setup(graph, "C", fiat_shamir, device)
    wrong = {"extra row": lambda u: torch.cat([u, u[:1]]), "one row fewer": lambda u: u[:-1],
             "3-d": lambda u: u[:, :, None], "3-d wide": lambda u: torch.stack([u, u], -1)}[change]
    fold = prover.fold
    for victim in (graph.mat_ops[0].name, graph.mat_ops[len(graph.mat_ops) // 2].name):
        prover.fold = lambda chis, victim=victim: {k: wrong(u) if k == victim else u for k, u in fold(chis).items()}
        assert _same_outcome(prover, pair, x, seed=1)["rejected_at"] == "freivalds"


def test_streaming_gives_the_verdicts_of_run_query_on_a_cnn(device):
    from pvi.fullcheck.models import build_float_model
    from pvi.fullcheck.quantize import quantize_input, quantize_model

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    graph = quantize_model(build_float_model("lenet5", 10).eval(), torch.randn(16, 1, 28, 28, generator=g))
    x = quantize_input(graph, torch.randn(1, 1, 28, 28, generator=g))
    for mode in ("C", "Kpre"):
        prover, pair = _setup(graph, mode, device=device)
        for i, (name, kw) in enumerate(_claim_attacks(graph.mat_ops).items()):
            _same_outcome(prover, pair, x, seed=i, forward_kwargs=kw)


def _rebinding_graph():
    """in: x -> h, A: h -> A.z, rq: A.z -> h (the name h bound again), B: h -> B.z."""
    g = torch.Generator().manual_seed(3)
    wa, wb = (_rand(g, -127, 128, (8, 8)).to(torch.int8) for _ in range(2))
    return IntGraph([CheapOp("in", ("x",), "h", fn=lambda x: x.clamp(-127, 127)),
                     MatOp("A", ("h",), "A.z", weight=wa, layout="linear"),
                     CheapOp("rq", ("A.z",), "h", fn=lambda z: (z >> 7).clamp(-127, 127)),
                     MatOp("B", ("h",), "B.z", weight=wb, layout="linear")], "x", "B.z")


@pytest.mark.parametrize("mode", ["C", "K", "Kpre"])
def test_streaming_checks_each_op_against_its_own_input_when_a_name_is_bound_twice(mode, device):
    graph = _rebinding_graph()
    x = _rand(torch.Generator().manual_seed(4), -127, 128, (1, 5, 8))
    prover, pair = _setup(graph, mode, device=device)
    assert _same_outcome(prover, pair, x, seed=1)["accepted"]
    forged = graph.mat_ops[1].weight.to(torch.int64) @ x.reshape(-1, 8).T      # W_B times A's input
    kw = {"tamper": lambda op, z: forged if op.name == "B" else z}
    assert _same_outcome(prover, pair, x, seed=2, forward_kwargs=kw)["rejected_at"] == "freivalds"


@pytest.mark.parametrize("attack", ["honest", "mid+1", "beyond int32", "int32 dtype"])
def test_streaming_keeps_the_fiat_shamir_transcript(attack, monkeypatch):
    absorbed = []
    absorb = proto.Challenger.absorb
    monkeypatch.setattr(proto.Challenger, "absorb", lambda self, label, blob: (
        absorbed.append((label, hashlib.sha256(blob).digest())), absorb(self, label, blob)))
    graph, x = _decoder("llama")
    prover, pair = _setup(graph, "C", fiat_shamir=True)
    transcripts = []
    for v in pair:
        absorbed.clear()
        out = proto.run_query(prover, v, x, forward_kwargs=_claim_attacks(graph.mat_ops)[attack])
        assert out["accepted"] == (attack == "honest")
        transcripts.append(list(absorbed))
    a, b = transcripts
    # the same claims absorbed; a rejected streaming query has also absorbed the u run_query never asked for
    assert any(label.startswith(b"claim/") for label, _ in a)
    assert b[:len(a)] == a and (len(b) > len(a)) == (attack != "honest")


@pytest.mark.parametrize("paths", ["int8", "deferred_int8"])
def test_streaming_with_int8_products_gives_the_verdicts_of_run_query(paths, monkeypatch):
    if paths == "deferred_int8":
        monkeypatch.setattr(proto, "_defer", lambda device: True)
    monkeypatch.setattr(proto, "int8_ok", lambda device: True)
    graph, x = _decoder("qwen")
    for mode in ("C", "Kpre"):
        prover, pair = _setup(graph, mode)
        for i, (name, kw) in enumerate(_claim_attacks(graph.mat_ops).items()):
            _same_outcome(prover, pair, x, seed=i, forward_kwargs=kw)


@cuda_only
@pytest.mark.parametrize("ahead", [0, 1, 1000])
def test_streaming_on_a_gpu_with_any_upload_window(ahead, monkeypatch):
    monkeypatch.setattr(pipeline, "AHEAD", ahead)
    graph, x = _decoder("llama")
    for mode in ("C", "Kpre"):
        prover, pair = _setup(graph, mode, device="cuda")
        for i, (name, kw) in enumerate(_claim_attacks(graph.mat_ops).items()):
            _same_outcome(prover, pair, x, seed=i, forward_kwargs=kw)


# -- a GPU verifier copies nothing back but its verdicts -------------------------------------------

@contextlib.contextmanager
def _only_verdicts_come_back(monkeypatch):
    """CUDA's sync debug mode set to "error" (any host round trip raises) except inside
    ``_to_host``, whose calls are counted."""
    calls = []
    real = proto._to_host

    def counted(flags):
        torch.cuda.set_sync_debug_mode(0)
        try:
            calls.append(len(flags))
            return real(flags)
        finally:
            torch.cuda.set_sync_debug_mode("error")

    monkeypatch.setattr(proto, "_to_host", counted)
    torch.cuda.set_sync_debug_mode("error")
    try:
        yield calls
    finally:
        torch.cuda.set_sync_debug_mode(0)


@cuda_only
@pytest.mark.parametrize("kind", list(_TINY))
def test_a_gpu_verifier_makes_one_round_trip_per_check(kind, monkeypatch):
    graph, x = _decoder(kind)
    params = proto.params_for(40, len(graph.mat_ops))
    coms = proto.commit_graph(graph, params.rate)
    prover = proto.Prover(graph, commitments=coms)
    v = proto.Verifier(graph.public(), params, "C", publics={k: c.public for k, c in coms.items()}, device="cuda")
    ch = proto.Challenger(seed=3)
    claims = {k: z.cuda() for k, z in prover.claims(x).items()}
    chis = {op.name: ch.folding(op.name, op.n_rows, params.reps).cuda() for op in graph.mat_ops}
    us = {k: u.cuda() for k, u in prover.fold(chis).items()}
    cols = {op.name: ch.columns(op.name, v.publics[op.name].n_points, params.columns) for op in graph.mat_ops}
    openings = prover.open(cols)
    xd = x.cuda()
    for _ in range(2):                  # the first query fills the caches (tables, masks, constants)
        inputs = v.derive(xd, claims)
        assert v.check_products(claims, inputs, chis, us) and v.check_columns(chis, us, cols, openings) is None
    with _only_verdicts_come_back(monkeypatch) as calls:
        inputs = v.derive(xd, claims)
        assert v.check_products(claims, inputs, chis, us) and v.check_columns(chis, us, cols, openings) is None
    assert len(calls) == 3
    wire = {k: pipeline.wire_claim(z.cpu(), pin=True) for k, z in claims.items()}
    wired = pipeline.wire_openings(openings)
    assert v.verify_streaming(xd, wire, chis, us, cols, wired) is None
    with _only_verdicts_come_back(monkeypatch) as calls:
        assert v.verify_streaming(xd, wire, chis, us, cols, wired) is None
    assert len(calls) == 1

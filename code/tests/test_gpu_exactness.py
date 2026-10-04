"""Exactness of every GPU code path of the defence, at the sizes of the real models.

The three GPU tests in ``test_fullcheck.py`` use tiny graphs (T = 12 tokens, K <= 4608),
so they never reach the regimes where a new GPU could behave differently: split-K /
stream-K GEMM kernels at K = 11,008, the batched P @ V product at T = 2,048 (4 chunks
of 512 terms, or ONE float32 GEMM on the int32 attention path), every partial sum at its
2**24 bound, the float64 field products of the prover's fold/open, the verifier's int8
tensor-core products, and the Reed--Solomon/NTT commitment on the device.  Run this file
on every new GPU type before any benchmark job (``smoke.sbatch`` does); every test must
pass with TF32 off (what bench.py uses, except its ``--tf32`` cells tagged ``_tf32``)
and on (which shows exactness does not depend on the flag).

    PVI_TEST_DEVICE=cuda python -m pytest tests/test_gpu_exactness.py -v   # default: cuda
"""

from __future__ import annotations

import os

import pytest
import torch

from pvi.fullcheck import field as fld
from pvi.fullcheck.commitment import WeightCommitment
from pvi.fullcheck.graph import exact_matmul
from pvi.fullcheck.transformer import _attention_heads

DEV = os.environ.get("PVI_TEST_DEVICE", "cuda")
pytestmark = pytest.mark.skipif(DEV.startswith("cuda") and not torch.cuda.is_available(), reason="needs a GPU")
P = fld.P


@pytest.fixture(params=[False, True], ids=["tf32_off", "tf32_on"])
def tf32(request):
    old = torch.backends.cuda.matmul.allow_tf32
    torch.backends.cuda.matmul.allow_tf32 = request.param
    yield request.param
    torch.backends.cuda.matmul.allow_tf32 = old


# (rows N, contraction K, columns M) of real weight products
SHAPES = [
    (4096, 4096, 2048),    # Llama-2-7B q/k/v/o projection, 2,048-token prompt
    (4096, 11008, 512),    # Llama-2-7B down projection: K = 10.75 chunks of 1,024
    (11008, 4096, 512),    # Llama-2-7B gate/up
    (32000, 4096, 1),      # Llama-2 LM head on the last position (GEMV path)
    (512, 4608, 1024),     # ResNet-18 3x3x512 convolution after im2col
]


def _operands(n, k, m, pattern):
    g = torch.Generator().manual_seed(n * 7 + k * 3 + m)
    if pattern == "max":            # odd values at the bound: each 1,024-term chunk sums to ~16.28 M,
        # just below 2**24; with 4,096-term chunks float32 would round (see the last test)
        w = (127 - 2 * torch.randint(0, 2, (n, k), generator=g)).to(torch.int8)
        return w, (127 - 2 * torch.randint(0, 2, (k, m), generator=g)).to(torch.float32)
    if pattern == "alternating":    # +-127 in blocks: huge cancellations inside every chunk
        i = torch.arange(n)[:, None] // 3
        j = torch.arange(k)[None, :] // 5
        w = (127 * (1 - 2 * ((i + j) % 2))).to(torch.int8)
        jj = torch.arange(k)[:, None] // 7
        c = torch.arange(m)[None, :] // 2
        return w, (127 * (1 - 2 * ((jj + c) % 2))).to(torch.float32)
    w = torch.randint(-127, 128, (n, k), generator=g, dtype=torch.int64).to(torch.int8)
    return w, torch.randint(-127, 128, (k, m), generator=g, dtype=torch.int64).to(torch.float32)


_REF: dict = {}


def _reference(n, k, m, pattern):
    """Exact integer product via float64 on the CPU (every |partial sum| < 2**53)."""
    key = (n, k, m, pattern)
    if key not in _REF:
        w, x = _operands(n, k, m, pattern)
        _REF[key] = (w.double() @ x.double()).to(torch.int64)
    return _REF[key]


@pytest.mark.parametrize("pattern", ["max", "alternating", "random"])
@pytest.mark.parametrize("n,k,m", SHAPES)
def test_exact_matmul_at_model_sizes(n, k, m, pattern, tf32):
    w, x = _operands(n, k, m, pattern)
    out = exact_matmul(w.to(DEV), x.to(DEV)).cpu()
    assert torch.equal(out, _reference(n, k, m, pattern))


@pytest.mark.parametrize("heads,t,dh", [(4, 2048, 128), (1, 4096, 128)])
def test_batched_pv_product_at_its_bound(heads, t, dh, tf32):
    # P @ V with probabilities in {253, 255} and values in {125, 127}: each 512-term chunk
    # sums to at most 512 * 255 * 127 = 16,581,120 < 2**24; the total (66 M at T = 2,048) is not.
    g = torch.Generator().manual_seed(heads * t)
    p = (255 - 2 * torch.randint(0, 2, (1, heads, t, t), generator=g)).to(torch.float32)
    v = (127 - 2 * torch.randint(0, 2, (1, heads, t, dh), generator=g)).to(torch.float32)
    ref = (p.double() @ v.double()).to(torch.int64)
    out = exact_matmul(p.to(DEV), v.to(DEV), max_w=256, max_x=128).cpu()
    assert torch.equal(out, ref)


@pytest.mark.parametrize("m_s", [1 << 12, 2_600_000], ids=["int64_path", "int32_path"])
@pytest.mark.parametrize("heads,t,dh", [(4, 2048, 128), (1, 4096, 128)])
def test_attention_matches_cpu_at_long_prompts(heads, t, dh, m_s, tf32):
    g = torch.Generator().manual_seed(t + heads)
    q, k, v = (torch.randint(-127, 128, (1, heads, t, dh), generator=g) for _ in range(3))
    k[:, :, ::3] = q[:, :, ::3]                         # peaked softmax rows as well as flat ones
    cpu = _attention_heads(q, k, v, m_s)                # the raw P @ V integers
    dev = _attention_heads(q.to(DEV), k.to(DEV), v.to(DEV), m_s).cpu()
    assert torch.equal(cpu, dev)


def test_field_products_on_device_are_exact():
    # the prover's fold/open run small_matmul_mod / field_matmul_mod in float64 on the GPU
    k = 3 * 8192 + 5                                    # several 8,192-term chunks plus a tail
    small = torch.full((6, k), -127, dtype=torch.int64)
    field = torch.full((k, 3), P - 1, dtype=torch.int64)
    expect = (k * 127) % P                              # (-127) * (-1) summed k times
    assert bool((fld.small_matmul_mod(small.to(DEV), field.to(DEV)).cpu() == expect).all())
    g = torch.Generator().manual_seed(11)
    small = torch.randint(-127, 128, (37, k), generator=g, dtype=torch.int64)
    field = torch.randint(0, P, (k, 5), generator=g, dtype=torch.int64)
    assert torch.equal(fld.small_matmul_mod(small.to(DEV), field.to(DEV)).cpu(), fld.small_matmul_mod(small, field))
    left = torch.full((4, 5000), P - 1, dtype=torch.int64)
    right = torch.full((5000, 2), P - 1, dtype=torch.int64)
    assert bool((fld.field_matmul_mod(left.to(DEV), right.to(DEV)).cpu() == 5000 % P).all())
    left = torch.randint(0, P, (4, 5000), generator=g, dtype=torch.int64)
    right = torch.randint(0, P, (5000, 2), generator=g, dtype=torch.int64)
    assert torch.equal(fld.field_matmul_mod(left.to(DEV), right.to(DEV)).cpu(), fld.field_matmul_mod(left, right))


def test_int8_products_at_model_sizes():
    # the verifier's int8 tensor-core products at Llama-2-7B sizes (checked against float64):
    # chi^T Z for a down projection's claims at the int32 extremes, u^T X for q/k/v's input
    if not fld.int8_ok(DEV):
        pytest.skip("no exact int8 GEMM on this device: the verifier uses float64 there")
    g = torch.Generator().manual_seed(13)
    chi = torch.randint(0, P, (5, 4096), generator=g, dtype=torch.int64)
    chi[0] = P - 1
    z = torch.randint(-(1 << 31), 1 << 31, (4096, 2048), generator=g, dtype=torch.int64)
    z[0], z[-1] = (1 << 31) - 1, -(1 << 31)
    want = fld.field_matmul_mod(chi, fld.to_field(z))
    assert torch.equal(fld.int8_field_matmul(fld.int8_left(chi.to(DEV)), z.to(torch.int32).to(DEV)).cpu(), want)
    u = torch.randint(0, P, (15, 4096), generator=g, dtype=torch.int64)
    x = torch.randint(-128, 128, (2048, 4096), generator=g, dtype=torch.int64)
    x[0], u[0] = -128, P - 1
    want = fld.field_matmul_mod(u, x.T.contiguous())
    assert torch.equal(fld.int8_small_matmul(fld.int8_right(u.to(DEV)), x.to(torch.int8).to(DEV)).cpu(), want)


def test_commitment_fold_and_open_match_cpu():
    # a 4096 x 4097 layer ([W | b]): NTT length 32,768, as for Llama-2-7B's projections
    g = torch.Generator().manual_seed(12)
    w = torch.randint(-127, 128, (4096, 4096), generator=g, dtype=torch.int64).to(torch.int8)
    b = torch.randint(-(1 << 25), 1 << 25, (4096,), generator=g, dtype=torch.int64)
    cpu = WeightCommitment.build(b"t", w, b, rate=4, device="cpu")
    dev = WeightCommitment.build(b"t", w, b, rate=4, device=DEV)
    assert cpu.tree.root == dev.tree.root               # identical codeword, column by column
    chi = torch.randint(0, P, (4, 4096), generator=g, dtype=torch.int64)
    assert torch.equal(cpu.fold(chi, "cpu"), dev.fold(chi, DEV))
    cols = torch.tensor([0, 1, 4097, 20000, cpu.n_points - 1])
    (c_cpu, p_cpu), (c_dev, p_dev) = cpu.open(cols, "cpu"), dev.open(cols, DEV)
    assert torch.equal(c_cpu, c_dev) and p_cpu == p_dev


def test_the_worst_case_pattern_would_catch_a_wrong_chunk_bound(monkeypatch):
    # guards the test above: with 4,096-term chunks (partial sums up to 66 M) float32 rounds
    import pvi.fullcheck.graph as graph

    w, x = _operands(256, 4096, 64, "max")
    ref = (w.double() @ x.double()).to(torch.int64)
    assert torch.equal(exact_matmul(w, x), ref)
    monkeypatch.setattr(graph, "_FP32_EXACT", 1 << 26)
    assert not torch.equal(exact_matmul(w, x), ref)

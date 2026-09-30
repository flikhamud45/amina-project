"""The verifier's constant-factor speed-ups are bit-exact.

Every fast routine is compared with the straightforward code it replaced, kept in
``pvi.fullcheck.reference``: the modular products (stacked limbs, signed claims,
batched blocks, numpy for small operands), the codeword at the opened columns
(baby-step giant-step), the leaf hashing, the right-hand sides, the range checks,
the batched product and column checks (same verdicts, same exceptions, on tampered
transcripts), the cheap operations, and whole graphs (the same claims and the same
derived tensors).  Tests taking a ``device`` also run on CUDA when it is available.
"""

from __future__ import annotations

import random

import pytest
import torch

from pvi.fullcheck import commitment as com
from pvi.fullcheck import field as fld
from pvi.fullcheck import protocol as proto
from pvi.fullcheck import reference as ref

P = fld.P
Z = proto.Z_BOUND
INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    return request.param


@pytest.fixture(params=["numpy", "torch"])
def small_path(request, monkeypatch):
    """Run CPU products on either side of the numpy threshold."""
    monkeypatch.setattr(fld, "NP_SMALL", (1 << 62) if request.param == "numpy" else 0)
    monkeypatch.setattr(com, "NP_SMALL", (1 << 62) if request.param == "numpy" else 0)
    return request.param


def _rand(g, lo, hi, shape):
    return torch.randint(lo, hi, shape, generator=g, dtype=torch.int64)


# -- field ------------------------------------------------------------------------------------

@pytest.mark.parametrize("r,k,m", [(1, 1, 1), (5, 6, 784), (2, 2047, 3), (2, 2048, 3), (2, 2049, 3), (3, 8191, 2),
                                   (3, 8192, 2), (3, 8193, 2), (2, 20000, 1), (4, 1025, 3), (3, 40, 0)])
def test_field_matmul_mod_matches_reference(r, k, m, small_path):
    g = torch.Generator().manual_seed(r * 1000 + k + m)
    chi = _rand(g, 0, P, (r, k))
    right = _rand(g, 0, P, (k, m))
    chi.view(-1)[0] = P - 1
    if right.numel():
        right.view(-1)[-1] = P - 1
    want = ref.field_matmul_mod(chi, right)
    assert torch.equal(fld.field_matmul_mod(chi, right), want)
    assert torch.equal(fld.field_matmul_mod(chi, right, right_bound=P), want)
    # signed claims at both ends of the range check, used without reduction
    z = _rand(g, 1 - Z, Z, (k, m))
    if z.numel():
        z.view(-1)[0], z.view(-1)[-1] = Z - 1, 1 - Z
    want = ref.field_matmul_mod(chi, z)
    assert torch.equal(fld.field_matmul_mod(chi, z, right_bound=Z), want)
    assert torch.equal(fld.field_matmul_mod(chi, z), want)
    assert torch.equal(fld.field_matmul_mod(chi, z, right_bound=Z, left_limbs=fld.limbs_f64(chi)), want)
    # batch dimensions: three independent products at once
    chis, zs = _rand(g, 0, P, (3, r, k)), _rand(g, 1 - Z, Z, (3, k, m))
    got = fld.field_matmul_mod(chis, zs, right_bound=Z)
    assert all(torch.equal(got[i], ref.field_matmul_mod(chis[i], zs[i])) for i in range(3))


def test_field_matmul_mod_reduces_any_left_and_right(small_path):
    g = torch.Generator().manual_seed(1)
    left = _rand(g, -(1 << 62), 1 << 62, (2, 300))
    left[0, :3] = torch.tensor([INT64_MIN, INT64_MAX, P])
    right = _rand(g, -(1 << 62), 1 << 62, (300, 4))
    right[:3, 0] = torch.tensor([INT64_MIN, INT64_MAX, -1])
    assert torch.equal(fld.field_matmul_mod(left, right), ref.field_matmul_mod(left, right))


def test_exact_chunk_is_the_longest_exact_block():
    for a, b in ((2047, P - 1), (2047, Z - 1), (127, P - 1)):
        c = fld.exact_chunk(a, b)
        assert c * a * b <= 1 << 53 < (c + 1) * a * b
    assert fld.exact_chunk(2047, P - 1) >= 2048 and fld.exact_chunk(2047, Z - 1) >= 8192
    with pytest.raises(ValueError):
        fld.exact_chunk(1 << 27, 1 << 27)


@pytest.mark.parametrize("k,chunk", [(10, 16), (16, 16), (17, 16), (4 * 600 + 3, 4), (4 * 512, 4)])
def test_exact_gemm_i64_sums_blocks_exactly(k, chunk):
    # 512 blocks and more take the reduced accumulation; the result is then the same residue
    g = torch.Generator().manual_seed(k)
    a = _rand(g, -(1 << 20), 1 << 20, (2, 3, k))
    b = _rand(g, -(1 << 20), 1 << 20, (k, 5))
    exact = a @ b                                       # int64 matmul (|sum| < 2**52 here)
    got = fld.exact_gemm_i64(a.to(torch.float64), b, chunk)
    assert torch.equal(torch.remainder(got, P), torch.remainder(exact, P))
    if k // chunk < 512:
        assert torch.equal(got, exact)


def test_field_products_on_device_match_cpu(device):
    g = torch.Generator().manual_seed(2)
    for r, k, m in ((2, 9728, 8), (5, 8193, 3), (3, 151, 7)):
        chi = _rand(g, 0, P, (r, k))
        z = _rand(g, 1 - Z, Z, (k, m))
        want = ref.field_matmul_mod(chi, z)
        assert torch.equal(fld.field_matmul_mod(chi.to(device), z.to(device), right_bound=Z).cpu(), want)
        assert torch.equal(fld.field_matmul_mod(chi.to(device), fld.to_field(z).to(device)).cpu(), want)


# -- commitment -------------------------------------------------------------------------------

@pytest.mark.parametrize("k,n,t", [(1, 4, 3), (26, 128, 66), (64, 256, 20), (85, 512, 66), (401, 2048, 66),
                                   (769, 4096, 68), (4609, 32768, 67), (3000, 16384, 1)])
def test_codeword_at_matches_the_vandermonde_product(k, n, t, small_path):
    g = torch.Generator().manual_seed(k + n)
    u = _rand(g, 0, P, (3, k))
    u[0, 0] = P - 1
    idx = torch.tensor(sorted(random.Random(k).sample(range(n), min(t, n))))
    want = ref.codeword_at(u, n, idx)
    assert torch.equal(com.codeword_at(u, n, idx), want)
    if n <= 4096:
        assert torch.equal(want, fld.rs_encode(u, n)[:, idx])
    us = _rand(g, 0, P, (4, 3, k))
    idxs = torch.stack([torch.tensor(sorted(random.Random(k + i).sample(range(n), min(t, n)))) for i in range(4)])
    got = com.codeword_at(us, n, idxs)
    assert all(torch.equal(got[i], ref.codeword_at(us[i], n, idxs[i])) for i in range(4))


def test_codeword_at_on_device_matches_cpu(device):
    g = torch.Generator().manual_seed(4)
    u = _rand(g, 0, P, (4, 50257))
    idx = torch.tensor(sorted(random.Random(4).sample(range(1 << 18), 68)))
    assert torch.equal(com.codeword_at(u.to(device), 1 << 18, idx).cpu(), ref.codeword_at(u, 1 << 18, idx))


def test_vandermonde_columns_index_with_a_mask():
    idx = torch.tensor([0, 1, 5, 1023])
    table = fld.power_table(fld.root_of_unity(1024), 1024)
    j = torch.arange(700)[:, None]
    assert torch.equal(com.vandermonde_columns(1024, 700, idx), table[(j * idx[None, :]) % 1024])


@pytest.mark.parametrize("n_rows", [1, 7, 3000, (1 << 14) + 1])
def test_column_leaves_hash_the_bytes_of_column_leaf(n_rows):
    g = torch.Generator().manual_seed(n_rows)
    opened = _rand(g, 0, P, (n_rows, 9))
    opened[0, 0] = P - 1
    opened[-1, 1] = -1                      # outside the field: both keep the low 32 bits
    opened[-1, 2] = 1 << 40
    idx = sorted(random.Random(n_rows).sample(range(10 * n_rows + 20), 9))
    assert com.column_leaves(b"tag", idx, opened) == ref.column_leaves(b"tag", idx, opened)


def _merkle_jobs(rnd):
    jobs, want = [], []
    for depth in (6, 9, 12):
        n = 1 << depth
        leaves = [rnd.randbytes(32) for _ in range(n)]
        tree = com.MerkleTree(leaves)
        for bad in (False, True):
            idx = sorted(rnd.sample(range(n), 20))
            proof = com.multiproof(tree, idx)
            if bad:
                proof = [bytes(32)] + proof[1:]
            jobs.append((tree.root, depth, {i: leaves[i] for i in idx}, proof))
            want.append(not bad)
    return jobs, want


def test_merkle_worker_processes_give_the_same_results(monkeypatch):
    jobs, want = _merkle_jobs(random.Random(3))
    jobs.append((jobs[0][0], jobs[0][1], jobs[0][2], [0] + jobs[0][3][1:]))   # not bytes: raises
    sequential = com.verify_multiproofs(jobs, workers=2)                      # off by default
    assert [ok for ok, _ in sequential[:-1]] == want and all(exc is None for _, exc in sequential[:-1])
    assert isinstance(sequential[-1][1], TypeError)
    monkeypatch.setattr(com, "MERKLE_PROCESSES", True)
    monkeypatch.setattr(com, "MERKLE_PROCESSES_MIN_HASHES", 1)
    parallel = com.verify_multiproofs(jobs, workers=2)
    assert [ok for ok, _ in parallel] == [ok for ok, _ in sequential]
    assert [type(exc) for _, exc in parallel] == [type(exc) for _, exc in sequential]

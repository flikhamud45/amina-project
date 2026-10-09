"""The verifier's constant-factor speed-ups are bit-exact.

Every fast routine is compared with the straightforward code it replaced, kept in
``pvi.fullcheck.reference``: the modular products (stacked limbs, signed claims,
batched blocks, numpy for small operands), the codeword at the opened columns
(baby-step giant-step), the leaf hashing, the right-hand sides, the range checks,
the batched product and column checks (same verdicts, same exceptions, on tampered
transcripts), the cheap operations, and whole graphs (the same claims and the same
derived tensors).  Tests taking a ``device`` also run on CUDA when it is available; tests
taking ``check_paths`` also run the checks in the forms a GPU verifier uses (deferred
verdicts, int8 GEMMs), on the CPU.
"""

from __future__ import annotations

import random
import warnings

import numpy as np
import pytest
import torch

from pvi.fullcheck import commitment as com
from pvi.fullcheck import field as fld
from pvi.fullcheck import protocol as proto
from pvi.fullcheck import reference as ref
from pvi.fullcheck.pipeline import wire_openings

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


@pytest.fixture(params=["at_once", "deferred", "int8", "deferred_int8"])
def check_paths(request, monkeypatch):
    """The verifier's checks as the CPU runs them, with the verdicts left on the "device" and
    copied back once (``_defer``, as on a GPU), and with the products on int8 GEMMs (``int8_ok``,
    emulated in float64 off the GPU)."""
    if "deferred" in request.param:
        monkeypatch.setattr(proto, "_defer", lambda device: True)
    if "int8" in request.param:
        monkeypatch.setattr(proto, "int8_ok", lambda device: True)
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
    # int32 operands, as transposed views (the wire's opened rows, the streaming verifier's claims)
    for right_, bound in ((right, P), (z, Z)):
        view = right_.to(torch.int32).T.contiguous().T
        want = ref.field_matmul_mod(chi, right_)
        assert torch.equal(fld.field_matmul_mod(chi, view, right_bound=bound), want)
        assert torch.equal(fld.field_matmul_mod(chi, view), want)
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
def test_row_leaves_hash_the_bytes_of_column_leaf(n_rows):
    g = torch.Generator().manual_seed(n_rows)
    opened = _rand(g, 0, P, (n_rows, 9))
    opened[0, 0] = P - 1
    opened[-1, 1] = -1                      # outside the field: both keep the low 32 bits
    opened[-1, 2] = 1 << 40
    idx = sorted(random.Random(n_rows).sample(range(10 * n_rows + 20), 9))
    assert com.row_leaves(b"tag", idx, com.column_rows(opened)) == ref.column_leaves(b"tag", idx, opened)


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
    # proof entries as memoryviews hash as bytes do, but cannot be pickled to a worker
    views = jobs + [(root, depth, lv, [memoryview(p) for p in pr]) for root, depth, lv, pr in jobs[:-1]]
    sequential_views = com.verify_multiproofs(views, workers=2)
    assert [ok for ok, _ in sequential_views] == want + [False] + want
    monkeypatch.setattr(com, "MERKLE_PROCESSES", True)
    monkeypatch.setattr(com, "MERKLE_PROCESSES_MIN_HASHES", 1)
    for some, want_some in ((jobs, sequential), (views, sequential_views)):    # in the workers; checked here
        parallel = com.verify_multiproofs(some, workers=2)
        assert [ok for ok, _ in parallel] == [ok for ok, _ in want_some]
        assert [type(exc) for _, exc in parallel] == [type(exc) for _, exc in want_some]


def test_merkle_worker_processes_fall_back_when_the_pool_breaks(monkeypatch):
    from concurrent.futures import Future
    from concurrent.futures.process import BrokenProcessPool

    jobs, want = _merkle_jobs(random.Random(4))

    class Broken:                  # a pool whose workers died: before or after taking the batch
        def __init__(self):
            self.calls = 0

        def submit(self, fn, jobs):
            self.calls += 1
            if self.calls == 1:
                raise BrokenProcessPool("a worker died")
            fut = Future()
            fut.set_exception(BrokenProcessPool("a worker died"))
            return fut

    monkeypatch.setattr(com, "MERKLE_PROCESSES", True)
    monkeypatch.setattr(com, "MERKLE_PROCESSES_MIN_HASHES", 1)
    monkeypatch.setitem(com._PROCESS_POOLS, 3, Broken())
    assert com.verify_multiproofs(jobs, workers=3) == [(ok, None) for ok in want]
    assert 3 not in com._PROCESS_POOLS                        # the next call starts new workers


# -- protocol kernels --------------------------------------------------------------------------

@pytest.mark.parametrize("deferred", [False, True])
def test_range_and_field_checks_match_reference(deferred, monkeypatch):
    monkeypatch.setattr(proto, "_defer", lambda device: deferred)
    for z in (torch.zeros(0, 3, dtype=torch.int64), torch.tensor([[Z - 1, 1 - Z]]), torch.tensor([[Z]]),
              torch.tensor([[-Z]]), torch.tensor([[INT64_MIN]]), torch.tensor([[INT64_MAX]]),
              torch.randint(1 - Z, Z, (50, 7)), torch.tensor([[0, INT64_MIN], [1, 2]]).T):
        pending = []                                   # derive's range check of a claim (deferred: in pending)
        ok = proto._check_bounds(z, 1 - Z, Z - 1, pending)
        assert (ok and not any(proto._to_host(pending))) == ref.in_range(z, Z)
        assert deferred or not pending
    for a in (torch.tensor([[0, P - 1]]), torch.tensor([[P]]), torch.tensor([[-1]]), torch.tensor([[INT64_MIN]]),
              torch.zeros(2, 0, dtype=torch.int64), torch.randint(0, P, (9, 4))):
        assert proto._in_field(a) == ref.in_field(a)


@pytest.mark.parametrize("deferred", [False, True])
def test_disagrees_flags_any_op_that_fails_its_freivalds_check(deferred, monkeypatch):
    monkeypatch.setattr(proto, "_defer", lambda device: deferred)
    g = torch.Generator().manual_seed(21)
    checks = []
    for m in (3, 0, 7):                                  # an op with no columns too
        lhs = _rand(g, 0, P, (2, m))
        checks.append((_rand(g, 0, P, (2, 5)), lhs, lhs.clone()))
    assert proto._to_host([proto._disagrees(checks)]) == [False]
    for i, (u, lhs, rhs) in enumerate(checks):
        bad_u = [u.clone() for _ in range(3)]
        for b, value in zip(bad_u, (P, -1, INT64_MIN)):
            b[1, -1] = value
        bad = [(b, lhs, rhs) for b in bad_u]
        if lhs.numel():
            bad.append((u, lhs, (rhs + 1) % P))
        for one in bad:
            changed = checks[:i] + [one] + checks[i + 1:]
            assert proto._to_host([proto._disagrees(changed)]) == [True]
            assert proto._to_host([proto._disagrees([one])]) == [True]


def _mat_op(layout, n_in, bias, conv=None, rows=4):
    from pvi.fullcheck.graph import MatOp

    return MatOp("op", ("x",), "op.z", weight=torch.zeros(rows, n_in, dtype=torch.int8),
                 bias=torch.zeros(rows, dtype=torch.int64) if bias else None, layout=layout, conv=conv)


@pytest.mark.parametrize("conv,shape,bias", [((5, 1, 2), (1, 1, 28, 28), True), ((3, 1, 1), (2, 16, 8, 8), True),
                                             ((3, 2, 1), (1, 8, 9, 9), False), ((1, 2, 0), (3, 12, 7, 7), True),
                                             ((7, 2, 3), (1, 3, 20, 20), False), ((3, 1, 1), (1, 64, 32, 32), True),
                                             ((3, 1, 1), (2, 911, 5, 5), True), ((1, 1, 0), (1, 8200, 2, 2), False)])
def test_conv_rhs_matches_reference(conv, shape, bias, device):
    # im2col (small), the shifted GEMM (C k^2 <= 8192) and the unfold fallback (C k^2 > 8192)
    g = torch.Generator().manual_seed(sum(shape))
    k = conv[0]
    op = _mat_op("conv", shape[1] * k * k, bias, conv)
    x = _rand(g, -128, 128, shape)
    x.view(-1)[:2] = torch.tensor([-128, 127])
    u = _rand(g, 0, P, (5, op.row_length))
    u.view(-1)[0] = P - 1
    assert torch.equal(proto._rhs(op, u.to(device), x.to(device)).cpu(), ref.rhs(op, u, x))


@pytest.mark.parametrize("k,shape", [(768, (1, 64)), (8192, (3,)), (8193, (2,)), (9728, (1, 8)), (20000, (1, 1)),
                                     (5, (2, 3))])
def test_linear_and_embedding_rhs_match_reference(k, shape, device):
    g = torch.Generator().manual_seed(k)
    for bias in (True, False):
        op = _mat_op("linear", k, bias)
        x = _rand(g, -128, 128, shape + (k,))
        u = _rand(g, 0, P, (4, op.row_length))
        u.view(-1)[-1] = P - 1
        assert torch.equal(proto._rhs(op, u.to(device), x.to(device)).cpu(), ref.rhs(op, u, x))
    op = _mat_op("embed", k, False)
    ids = _rand(g, 0, k, shape)
    u = _rand(g, 0, P, (4, k))
    assert torch.equal(proto._rhs(op, u.to(device), ids.to(device)).cpu(), ref.rhs(op, u, ids))


# -- the batched checks give the op-by-op verdicts -------------------------------------------------

def _decoder_setup(mode):
    from pvi.fullcheck.transformer import DecoderConfig, build_decoder

    cfg = DecoderConfig("tiny", 32, 2, 4, 2, 8, 64, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                        bias=False, tied=False, qk_norm=True, max_pos=64)
    graph = build_decoder(cfg, calib_tokens=6, seed=2)
    params = proto.params_for(20, len(graph.mat_ops))
    coms = proto.commit_graph(graph, params.rate) if mode == "C" else {}
    prover = proto.Prover(graph, commitments=coms)
    v = proto.Verifier(graph.public(), params, mode, publics={k: c.public for k, c in coms.items()},
                       weights={op.name: (op.weight, op.bias) for op in graph.mat_ops})
    if mode == "Kpre":
        v.precompute(proto.Challenger(seed=5))
    x = torch.randint(0, cfg.vocab, (1, 6), generator=torch.Generator().manual_seed(0))
    return graph, prover, v, x


def _transcript(mode, prover, v, x, mats):
    ch = proto.Challenger(seed=9)
    claims = prover.claims(x)
    if mode == "Kpre":
        chis = {op.name: v._pre[op.name][0] for op in mats}
        us = {op.name: v._pre[op.name][1] for op in mats}
    else:
        chis = {op.name: ch.folding(op.name, op.n_rows, v.params.reps) for op in mats}
        us = prover.fold(chis)
    return ch, claims, chis, us


def _with(d, name, value):
    return dict(d, **{name: value})


@pytest.mark.parametrize("mode", ["C", "Kpre"])
def test_batched_products_give_the_per_op_verdicts(mode, check_paths):
    graph, prover, v, x = _decoder_setup(mode)
    mats = graph.mat_ops
    _, claims, chis, us = _transcript(mode, prover, v, x, mats)
    inputs = v.derive(x, claims)
    assert v.check_products(claims, inputs, chis, us) is True
    kept = {k: val[1] for k, val in v._kept.items()}
    assert bool(kept) == (mode == "Kpre")
    for _ in range(2):        # Kpre: the kept stacks are reused from the second query on
        assert v.check_products(claims, inputs, chis, us) is True
        assert all(v._kept[k][1] is val for k, val in kept.items())
    rnd = random.Random(1)
    for op in mats:
        z = claims[op.name].clone()
        z.view(-1)[rnd.randrange(z.numel())] += 1
        bad = _with(claims, op.name, z)
        inp = v.derive(x, bad)
        assert v.check_products(bad, inp, chis, us) is ref.check_products(v, bad, inp, chis, us) is False
        u = us[op.name].clone()
        u[0, rnd.randrange(u.shape[1])] += 1
        u %= P
        bad_u = _with(us, op.name, u)     # (an embedding's unread u columns are not in its product)
        inputs = v.derive(x, claims)
        assert v.check_products(claims, inputs, chis, bad_u) == ref.check_products(v, claims, inputs, chis, bad_u)
    # malformed u: the first failing op decides, as op by op
    first, later = mats[1].name, mats[-2].name
    z = claims[first].clone()
    z.view(-1)[0] += 1
    bad = _with(claims, first, z)
    inp = v.derive(x, bad)
    for broken in ("not a tensor", us[later].to(torch.int32), us[later][:, :-1], us[later] + P, None):
        bad_u = _with(us, later, broken)
        assert v.check_products(bad, inp, chis, bad_u) is ref.check_products(v, bad, inp, chis, bad_u) is False
    inputs = v.derive(x, claims)
    assert v.check_products(claims, inputs, chis, _with(us, later, us[later] + P)) is False
    for impl in (v.check_products, lambda *args: ref.check_products(v, *args)):
        with pytest.raises(AttributeError):
            impl(claims, inputs, chis, _with(us, later, "not a tensor"))


def test_claims_modified_after_derive_are_not_trusted(check_paths):
    graph, prover, v, x = _decoder_setup("Kpre")
    mats = graph.mat_ops
    _, claims, chis, us = _transcript("Kpre", prover, v, x, mats)
    inputs = v.derive(x, claims)
    assert v.check_products(claims, inputs, chis, us)
    name = mats[3].name
    claims[name].view(-1)[0] += P          # same residue, out of range: not multiplied as a signed claim
    assert v.check_products(claims, inputs, chis, us) == ref.check_products(v, claims, inputs, chis, us)
    claims[name].view(-1)[0] += 1 - P
    assert v.check_products(claims, inputs, chis, us) is False
    claims[name].view(-1)[0] -= 1
    assert v.check_products(claims, inputs, chis, us) is True
    u = us[mats[1].name]
    u.view(-1)[0] = (u.view(-1)[0] + 1) % P      # the kept stack of u must not be reused
    assert v.check_products(claims, inputs, chis, us) is ref.check_products(v, claims, inputs, chis, us) is False


def test_the_verifier_keeps_no_query_tensors_alive(check_paths):
    import gc
    import weakref

    graph, prover, v, x = _decoder_setup("Kpre")
    mats = graph.mat_ops
    _, claims, chis, us = _transcript("Kpre", prover, v, x, mats)
    inputs = v.derive(x, claims)
    other_u = {k: t.clone() for k, t in us.items()}        # not the verifier's own tensors
    assert v.check_products(claims, inputs, chis, other_u) is True
    refs = [weakref.ref(t) for t in [*claims.values(), *other_u.values()]]
    del claims, inputs, other_u
    gc.collect()
    assert all(r() is None for r in refs)


def test_batched_columns_give_the_per_op_verdicts(check_paths):
    graph, prover, v, x = _decoder_setup("C")
    mats = graph.mat_ops
    ch, claims, chis, us = _transcript("C", prover, v, x, mats)
    cols = {op.name: ch.columns(op.name, v.publics[op.name].n_points, v.params.columns) for op in mats}
    openings = prover.open(cols)
    assert v.check_columns(chis, us, cols, openings) is None
    variants = []
    for i in (0, len(mats) // 2, len(mats) - 1):
        name = mats[i].name
        o, pr = openings[name]
        o1 = o.clone()
        o1[0, 0] = (o1[0, 0] + 1) % P
        o2 = o.clone()
        o2[0, 0] = P
        variants += [{name: (o1, pr)}, {name: (o, [bytes(32)] + pr[1:])}, {name: (o[:, :-1], pr)},
                     {name: (o2, pr)}, {name: (o, pr[:-1])}, {name: (o, pr + [bytes(32)])},
                     {name: (o.to(torch.int32), pr)}]
    # two defects in different ops: the earlier op's verdict wins, even over a later exception
    a, b = mats[1].name, mats[-2].name
    (oa, pa), (ob, pb) = openings[a], openings[b]
    ob1 = ob.clone()
    ob1[0, 0] = (ob1[0, 0] + 1) % P
    variants += [{a: (oa, [bytes(32)] + pa[1:]), b: (ob1, pb)}, {a: (oa[:, :-1], pa), b: (ob1, pb)},
                 {b: (oa, pa)}, {a: (ob1, pb), b: (oa[:, :-1], pa)},
                 {a: (oa, [bytes(32)] + pa[1:]), b: (ob, [0] + pb[1:])},
                 {a: (ob1, pb), b: (ob, [0] + pb[1:])}, {a: (oa, [bytes(32)] + pa[1:]), b: None}]
    for var in variants:
        opened = dict(openings, **var)
        want = ref.check_columns(v, chis, us, cols, opened)
        assert v.check_columns(chis, us, cols, opened) == want
        rows = wire_openings(opened)          # the int32 rows run_query(wire=True) takes as they are
        assert v.check_columns(chis, us, cols, rows, wire=True) == ref.check_columns(v, chis, us, cols, rows,
                                                                                    wire=True) == want
    bad_u = _with(us, mats[2].name, (us[mats[2].name] + 1) % P)
    assert v.check_columns(chis, bad_u, cols, openings) == ref.check_columns(v, chis, bad_u, cols, openings) \
        == v.check_columns(chis, bad_u, cols, wire_openings(openings), wire=True) == "columns_code"
    # an exception is raised where op by op would raise it
    missing = dict(openings)
    del missing[mats[-1].name]
    for broken, exc in ((missing, KeyError), (_with(openings, b, (ob, [0] + pb[1:])), TypeError),
                        (_with(openings, b, None), TypeError)):
        for impl in (v.check_columns, lambda *args, **kw: ref.check_columns(v, *args, **kw)):
            with pytest.raises(exc):
                impl(chis, us, cols, broken)
            with pytest.raises(exc):
                impl(chis, us, cols, wire_openings(broken), wire=True)


# -- cheap operations ------------------------------------------------------------------------------

def test_requant_and_mul_add_half_match_reference(device):
    from pvi.fullcheck.graph import mul_add_half, requant

    g = torch.Generator().manual_seed(11)
    z = _rand(g, 1 - Z, Z, (2, 5, 33))
    z.view(-1)[:2] = torch.tensor([Z - 1, 1 - Z])
    for m in (torch.tensor(123456789), _rand(g, 1, 1 << 31, (1, 5, 1)), torch.tensor(1 << 40)):
        for lo, hi in ((-127, 127), (0, 127)):
            assert torch.equal(requant(z.to(device), m.to(device), 30, lo, hi).cpu(), ref.requant(z, m, 30, lo, hi))
    wide = torch.tensor([[INT64_MAX, INT64_MIN, -1]])          # int64 wraps the same way in both
    assert torch.equal(mul_add_half(wide.to(device), torch.tensor(3, device=device), 30).cpu(),
                       wide * 3 + (1 << 29))
    out = torch.empty(4, 6, dtype=torch.int64, device=device)
    small = _rand(g, -1000, 1000, (6, 4)).to(device)
    assert mul_add_half(small.T, torch.tensor(7, device=device), 12, out=out) is out
    assert torch.equal(out, small.T * 7 + (1 << 11))
    z32 = z.to(torch.int32)                                    # other dtypes keep the plain formula
    assert torch.equal(requant(z32, 3, 30, -127, 127), ref.requant(z32, 3, 30, -127, 127))


def test_isqrt_matches_reference():
    import math

    from pvi.fullcheck.transformer import _isqrt

    g = torch.Generator().manual_seed(5)
    s = torch.cat([torch.arange(0, 2000), _rand(g, 0, 1 << 50, (5000,)),
                   torch.tensor([(1 << 26) ** 2, (1 << 26) ** 2 - 1, (1 << 26) ** 2 + 1, (3 ** 19) ** 2,
                                 (1 << 62) + 12345, (1 << 53) + 1, INT64_MAX])]).reshape(-1, 1)
    assert torch.equal(_isqrt(s), ref.isqrt(s))
    small = s[s.view(-1) < (1 << 52)].reshape(-1, 1)
    assert torch.equal(_isqrt(small), torch.tensor([[max(1, math.isqrt(int(v)))] for v in small.view(-1)]))
    # a sum of squares that wrapped (adversarial claims only): the reference's steps, silently
    wrapped = torch.cat([s[:5], torch.tensor([[-1], [INT64_MIN], [-(1 << 40)], [3]])])
    empty = torch.zeros(0, 1, dtype=torch.int64)
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        assert torch.equal(_isqrt(wrapped), ref.isqrt(wrapped)) and torch.equal(_isqrt(empty), ref.isqrt(empty))


def test_derive_is_silent_on_claims_whose_norm_sums_wrap(monkeypatch):
    # an in-range embedding claim, scaled x8 into the residual stream, makes x * x wrap in the norm
    from pvi.fullcheck import transformer as tr

    graph, prover, v, x = _decoder_setup("Kpre")
    claims = prover.claims(x)
    emb = graph.mat_ops[0].name
    bad = dict(claims, **{emb: torch.full_like(claims[emb], Z - 1)})
    with warnings.catch_warnings(), np.errstate(all="raise"):
        warnings.simplefilter("error")
        got = v.derive(x, bad)
    monkeypatch.setattr(tr, "_isqrt", ref.isqrt)
    want = v.derive(x, bad)
    assert got.keys() == want.keys() and all(torch.equal(got[k], want[k]) for k in got)


def test_transformer_ops_match_reference(device):
    from pvi.fullcheck import transformer as tr

    g = torch.Generator().manual_seed(12)
    r = _rand(g, -(1 << 22), 1 << 22, (2, 9, 64))
    gain = _rand(g, 1 << 19, 1 << 22, (64,))
    for center in (True, False):
        assert torch.equal(tr._norm_int(r.to(device), gain, center).cpu(), ref.norm_int(r, gain, center))
    x = _rand(g, -127, 128, (1, 9, 4, 16))
    for theta in (1e4, 1e6):
        assert torch.equal(tr._rope(x.to(device), theta).cpu(), ref.rope(x, theta))
        assert torch.equal(tr._rope(x.to(device), theta).cpu(), ref.rope(x, theta))     # the cached tables
    table = _rand(g, -127, 128, (256,))
    xi = _rand(g, -300, 300, (3, 7, 11))
    assert torch.equal(tr._lut(xi.to(device), table).cpu(), ref.lut(xi, table))
    a = _rand(g, -(1 << 22), 1 << 22, (1, 9, 64))
    b = _rand(g, -(1 << 29), 1 << 29, (1, 9, 64))
    m = torch.tensor(123456789)
    assert torch.equal(tr._residual(a.to(device), b.to(device), m).cpu(), ref.residual(a, b, m))
    assert torch.equal(tr._residual(a.to(device), b.to(device).transpose(1, 2).contiguous().transpose(1, 2), m).cpu(),
                       ref.residual(a, b, m))
    for t in (1, 5, 40):
        q, k, v = (_rand(g, -127, 128, (1, 3, t, 16)) for _ in range(3))
        q[..., 0, :] = 127
        raw = tr._attention_heads(q.to(device), k.to(device), v.to(device), 1 << 20)
        assert torch.equal(raw.cpu(), ref.attention_heads(q, k, v, 1 << 20, None))
        out = torch.empty(1, t, 3, 16, dtype=torch.int64, device=device).transpose(1, 2)
        tr._write_output(out, raw, 1 << 22)
        assert torch.equal(out.cpu(), ref.attention_heads(q, k, v, 1 << 20, 1 << 22))


def test_the_real_weight_residual_is_the_reference_residual():
    from pvi.fullcheck.real_weights import _RealBuilder

    g = torch.Generator().manual_seed(13)
    b = _RealBuilder(torch.zeros(1, 4, dtype=torch.int64), 100.0, 4.0)
    b.env.update(r=_rand(g, -(1 << 22), 1 << 22, (1, 9, 64)), z=_rand(g, -(1 << 29), 1 << 29, (1, 9, 64)))
    b.scale.update(r=0.01, z=0.0037)
    out = b.residual("r", "z")
    m = torch.tensor(round((1 << 30) * 0.0037 / 0.01))
    assert torch.equal(b.env[out], ref.residual(b.env["r"], b.env["z"], m))


@pytest.mark.parametrize("m_s", [1 << 20, 1000])          # the int32 path and its int64 fallback
@pytest.mark.parametrize("hq,hkv,t,group_heads", [(8, 2, 1, None), (8, 2, 7, None), (4, 4, 9, None), (6, 1, 5, None),
                                                  (32, 8, 8, None), (8, 8, 12, 3), (8, 2, 12, 3)])
def test_attention_matches_reference(hq, hkv, t, group_heads, m_s, device, monkeypatch):
    from pvi.fullcheck import transformer as tr

    g = torch.Generator().manual_seed(hq * 100 + hkv * 10 + t)
    dh = 16
    q = _rand(g, -127, 128, (2, t, hq * dh))
    k = _rand(g, -127, 128, (2, t, hkv * dh))
    v = _rand(g, -127, 128, (2, t, hkv * dh))
    if group_heads:          # a few heads per group, the last group shorter
        width = 4 if tr._int32_scores(m_s, dh, t) else 8                 # bytes per score
        monkeypatch.setattr(tr, "ATTN_BYTES", width * 2 * t * t * group_heads)
    heads = []
    attention_heads = tr._attention_heads
    monkeypatch.setattr(tr, "_attention_heads",
                        lambda q, *args: heads.append(q.shape[1]) or attention_heads(q, *args))
    for m_o in (None, 1 << 21):
        got = tr._attention(q.to(device), k.to(device), v.to(device), m_s, m_o, hq, hkv, dh).cpu()
        assert torch.equal(got, ref.attention(q, k, v, m_s, m_o, hq, hkv, dh))
    # heads per call: groups of group_heads query heads, else all at once (the key/value heads, stacked)
    want = [group_heads] * (hq // group_heads) + [hq % group_heads] if group_heads else [hkv]
    assert heads == 2 * want


@pytest.mark.parametrize("layout,shape", [("linear", (2, 3, 10)), ("linear", (5, 10)), ("embed", (2, 3)),
                                          ("conv", (2, 3, 6, 6))])
def test_fold_gives_the_reference_values(layout, shape):
    g = torch.Generator().manual_seed(7)
    op = _mat_op(layout, 3 * 9 if layout == "conv" else 10, False, conv=(3, 1, 1), rows=4)
    x = _rand(g, 0, 10, shape)
    z = _rand(g, -1000, 1000, (4, op.n_cols(x)))
    assert torch.equal(op.fold(z, x), ref.fold(op, z, x))


# -- whole graphs: the same claims, derived tensors and verdicts as the reference code ----------------

def _use_reference_code(monkeypatch):
    """Route every graph closure, fold and check through ``pvi.fullcheck.reference``."""
    from pvi.fullcheck import graph, quantize, transformer

    monkeypatch.setattr(quantize, "requant", ref.requant)
    for name, fn in (("requant", ref.requant), ("_norm_int", ref.norm_int), ("_lut", ref.lut), ("_rope", ref.rope),
                     ("_attention", ref.attention), ("_residual", ref.residual)):
        monkeypatch.setattr(transformer, name, fn)
    monkeypatch.setattr(graph, "requant", ref.requant)                  # requant_fn's closures
    monkeypatch.setattr(graph, "residual_add", lambda a, b, m, shift, res_max: ref.residual(a, b, m))
    monkeypatch.setattr(graph.MatOp, "fold", ref.fold)
    monkeypatch.setattr(proto.Verifier, "check_products", ref.check_products)
    monkeypatch.setattr(proto.Verifier, "check_columns", ref.check_columns)


def _graph(kind):
    import dataclasses

    from pvi.fullcheck.models import VGG, LeNet5, ResNet18
    from pvi.fullcheck.quantize import quantize_input, quantize_model
    from pvi.fullcheck.transformer import CONFIGS, DecoderConfig, build_decoder

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    cnns = {"lenet5": (LeNet5, (1, 28, 28)), "vgg11": (lambda n: VGG("vgg11", n), (3, 32, 32)),
            "resnet18": (ResNet18, (3, 32, 32))}
    if kind in cnns:
        make, shape = cnns[kind]
        graph = quantize_model(make(10).eval(), torch.randn(16, *shape, generator=g))
        return graph, quantize_input(graph, torch.randn(2, *shape, generator=g))
    small = {"gpt2": dataclasses.replace(CONFIGS["gpt2"], vocab=500),
             "opt-350m": dataclasses.replace(CONFIGS["opt-350m"], vocab=500),
             "qwen3-4b": dataclasses.replace(CONFIGS["qwen3-4b"], vocab=300),
             "llama-like": dataclasses.replace(CONFIGS["llama2-7b"], d_model=256, n_heads=4, n_kv_heads=4,
                                               head_dim=64, d_ff=8200, vocab=300),
             "gqa-tiny": DecoderConfig("tiny", 32, 2, 4, 2, 8, 64, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                                       bias=False, tied=False, qk_norm=True, max_pos=64)}[kind]
    graph = build_decoder(small, n_layers=min(2, small.n_layers) if kind != "qwen3-4b" else 1, calib_tokens=8)
    return graph, torch.randint(0, small.vocab, (1, 9), generator=g)


@pytest.mark.parametrize("check_paths", ["at_once", "deferred"], indirect=True)
@pytest.mark.parametrize("kind", ["lenet5", "vgg11", "resnet18", "gpt2", "opt-350m", "qwen3-4b", "llama-like"])
def test_graphs_give_the_reference_claims_and_derived_tensors(kind, monkeypatch, check_paths):
    graph, x = _graph(kind)
    _, claims = graph.forward(x)
    params = proto.params_for(40, len(graph.mat_ops))
    derived = [proto.Verifier(graph.public(), params, "Kpre", lean=lean).derive(x, claims) for lean in (False, True)]
    _use_reference_code(monkeypatch)
    _, claims_ref = graph.forward(x)
    assert claims.keys() == claims_ref.keys() and all(torch.equal(claims[k], claims_ref[k]) for k in claims)
    derived_ref = proto.Verifier(graph.public(), params, "Kpre").derive(x, claims)
    for got in derived:
        assert got is not None and got.keys() == derived_ref.keys()
        assert all(torch.equal(got[k], derived_ref[k]) for k in got)


def _queries(graph, x, mode, fiat_shamir, device="cpu"):
    """``run_query`` on an honest query and on tampered claims, ``u`` and openings."""
    params = proto.params_for(40, len(graph.mat_ops), fiat_shamir=fiat_shamir)
    coms = proto.commit_graph(graph, params.rate) if mode == "C" else {}
    prover = proto.Prover(graph, commitments=coms)
    v = proto.Verifier(graph.public(), params, mode, publics={k: c.public for k, c in coms.items()},
                       weights={op.name: (op.weight, op.bias) for op in graph.mat_ops}, device=device)
    if mode == "Kpre":
        v.precompute(proto.Challenger(seed=5))
    mats = graph.mat_ops
    out = []
    for i, op in enumerate((None, mats[0], mats[len(mats) // 2], mats[-1])):
        def tamper(o, z, op=op):
            return z + 1 if op is not None and o.name == op.name else z

        out.append(proto.run_query(prover, v, x, seed=i, forward_kwargs={"tamper": tamper}))
    if mode == "C":
        victim = mats[len(mats) // 2].name
        fold, open_ = prover.fold, prover.open
        prover.fold = lambda chis: {k: (u + (k == victim)) % P for k, u in fold(chis).items()}
        out.append(proto.run_query(prover, v, x, seed=7))
        prover.fold = fold
        for bad in (lambda o, pr: ((o + 1) % P, pr), lambda o, pr: (o, pr[:-1]),
                    lambda o, pr: (o, [bytes(32)] + pr[1:])):
            prover.open = lambda cols, bad=bad: {k: (bad(*o) if k == victim else o) for k, o in open_(cols).items()}
            out.append(proto.run_query(prover, v, x, seed=8))
    return [(r["accepted"], r["rejected_at"], r["bytes"]) for r in out]


@pytest.mark.parametrize("kind,mode,fiat_shamir", [("lenet5", "C", False), ("lenet5", "C", True),
                                                   ("resnet18", "Kpre", False), ("gqa-tiny", "C", True),
                                                   ("gqa-tiny", "K", False), ("gqa-tiny", "Kpre", False)])
def test_queries_give_the_reference_verdicts_proof_bytes_and_transcript(kind, mode, fiat_shamir, monkeypatch,
                                                                         check_paths):
    import hashlib

    absorbed = []
    absorb = proto.Challenger.absorb
    monkeypatch.setattr(proto.Challenger, "absorb", lambda self, label, blob: (
        absorbed.append((label, hashlib.sha256(blob).digest())), absorb(self, label, blob)))
    graph, x = _graph(kind)
    x = x[:1]
    got = _queries(graph, x, mode, fiat_shamir)
    assert got[0][0] and not any(r[0] for r in got[1:])
    labels = {"freivalds", "columns_code", "columns_merkle"} if mode == "C" else {"freivalds"}
    assert {r[1] for r in got[1:]} == labels
    transcript, absorbed[:] = list(absorbed), []
    assert bool(transcript) == fiat_shamir
    _use_reference_code(monkeypatch)
    assert got == _queries(graph, x, mode, fiat_shamir)
    assert transcript == absorbed


@pytest.mark.parametrize("kind,mode", [("lenet5", "C"), ("lenet5", "Kpre"), ("gqa-tiny", "C"), ("gqa-tiny", "K"),
                                       ("gqa-tiny", "Kpre")])
def test_a_gpu_client_gives_the_cpu_verdicts(kind, mode, device):
    graph, x = _graph(kind)
    assert _queries(graph, x[:1], mode, False, device) == _queries(graph, x[:1], mode, False)

"""V-V1N parts 1-4 (``SP2027_PLAN2.md``): the fast V1 verifier gives today's results bit for bit.

* Parts 1-2 (:mod:`pvi.v1.products`): the window tables are affine plus a 0/1 step at every window width
  ``2 .. 2^16`` that a multiplier gives (the identity the table products rest on); ``lw_times`` (table products and buckets) equals today's
  ``[L ; W]`` float64 GEMM at random points of ``F_{p^8}``, with exceptions, across the window widths, for requant
  and residual ops, in every value dtype and across row blocks and column chunks; ``xbar`` equals
  ``extfield.int_times``; spans past :data:`buckets.SPAN` fall back.
* The final check's extensions (``cut_protocol._final_claims``) and ``_xbar`` are the same field elements with
  ``PVI_V1_BUCKETS`` = 0, 1 and scatter at random points, in mode C with row and col layouts, in Kpre, and on a
  model with many exceptions.
* Part 3: ``logup_gkr.verify_batch`` gives ``verify``'s outputs and transcript instance by instance and rejects
  every tampered coefficient as ``verify`` does; one challenge (``Challenger._ext_one``) is today's, also on crafted
  streams that hit both rejection rules; ``extfield.inv_many``; the batch-inverted logUp identity's verdict.
* Part 4: the cut ops' values are kept as int8 (requant) and int16 (residual); out-of-range values are rejected
  with today's label, also when the token rule fails first.
* Every tamper of ``test_cut_protocol`` (and random trace-tampering and single-value ones) gets the same verdict
  and label with ``PVI_V1_BUCKETS`` = 0, 1 and scatter.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from pvi.v1 import products as buckets
from pvi.v1 import cut as cutmod
from pvi.v1 import protocol as cp
from pvi.core import extfield as ef
from pvi.v1 import logup
from pvi.v1 import gkr
import pvi.v0 as proto
from pvi.core.field import P
from pvi.models.decoder import build_decoder
import test_cut_protocol as tcp
from test_plans import _TINY

MODES = ("0", "1", "scatter")
E0 = torch.zeros(0, dtype=torch.int64)


def _rand_ext(g, n):
    return torch.randint(0, P, (n, ef.D), generator=g, dtype=torch.int64)


def _op(kind, mult, lo=-128, hi=127, n_rows=1):
    return cutmod.CutOp("t", kind, int(mult), 30, lo, hi, n_rows, 4, False, "c", 0, 0, "t", "row")


def _multipliers(w):
    """The multipliers ``m`` with ``floor(2^30 / m) = w``: the smallest and the largest (none if empty)."""
    hi, lo = (1 << 30) // w, (1 << 30) // (w + 1) + 1
    return sorted({lo, hi}) if lo <= hi else []


# ------------------------------------------------------------------ parts 1-2: the products
def test_the_windows_are_affine_plus_a_step_at_every_width():
    # lo(v) = lo(a) + w0 (v - a) + C(v), W(v) = w0 + D(v), D in {0, 1}, 0 <= C(v) <= v - a, at every window width
    # floor(2^30 / m) in 2 .. 2^16 that a multiplier gives (both ends of its multipliers: 76,294) over the int8 values
    v = torch.arange(-128, 129, dtype=torch.int64)
    ms = torch.tensor([m for w in range(2, (1 << 16) + 1) for m in _multipliers(w)], dtype=torch.int64)
    assert ms.numel() == 76294
    for c0 in range(0, ms.numel(), 4096):
        m = ms[c0:c0 + 4096, None]
        lo = -((-((v << 30) - (1 << 29))) // m)                       # cut.lo, broadcast over the multipliers
        w = lo[:, 1:] - lo[:, :-1]
        w0 = (1 << 30) // m
        d = w - w0
        c = lo[:, :-1] - lo[:, :1] - w0 * (v[:-1] + 128)
        assert bool(((d == 0) | (d == 1)).all())
        assert bool((c >= 0).all() and (c <= v[:-1] + 128).all())
    # the module's own tables on a sample (it asserts the identity itself), and a residual span
    for w in (2, 3, 5, 255, 256, 257, 4097, 65535, 65536):
        for m in _multipliers(w):
            lo_a, w0, tab, lo_v, w_v = buckets._affine(_op("requant", m), -128, 127)
            assert w0 == w and torch.equal(lo_v, cutmod.lo(torch.arange(-128, 128), m))
            assert torch.equal(lo_a + w0 * tab[:, 0].long() + tab[:, 1].long(), lo_v)
            assert torch.equal(w0 + tab[:, 2].long(), w_v)
    buckets._affine(_op("residual", 3 << 20), -6000, 6000)


def _widths():
    ws = set(range(2, 70)) | {(1 << k) + d for k in range(6, 17) for d in (-1, 0, 1)}
    g = torch.Generator().manual_seed(11)
    ws |= set(torch.randint(2, (1 << 16) + 1, (60,), generator=g).tolist())
    return sorted(w for w in ws if 2 <= w <= 1 << 16)


def _case(g, op, n, t, a, b, n_exc):
    """Values in ``[a, b]`` ``[n, t]`` with ``n_exc`` exceptions (clamped values with an exact ``z``)."""
    s = torch.randint(a, b + 1, (n, t), generator=g)
    k = min(n_exc, n * t)
    idx = torch.randperm(n * t, generator=g)[:k].sort().values
    s.view(-1)[idx[: k // 2]] = op.hi if op.kind == "requant" else b
    s.view(-1)[idx[k // 2:]] = op.lo_ if op.kind == "requant" else a
    z = torch.randint(1 - (1 << 29), 1 << 29, (k,), generator=g)
    return s, idx, z


@pytest.mark.parametrize("method", ["1", "scatter"])
def test_lw_times_equals_the_float64_gemm_at_every_tested_width(method):
    g = torch.Generator().manual_seed(1)
    for w in _widths():
        for m in _multipliers(w):
            for lo, hi in ((-128, 127), (-100, 90)):
                op = _op("requant", m, lo, hi, 5)
                s, idx, z = _case(g, op, 5, 7, lo, hi, 3)
                e = _rand_ext(g, 7)
                want = cp._lw_e_old(s, idx, z, op, e, 7)
                got = buckets.lw_times(s, idx, z, op, e, method)
                assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]), (w, m, lo, hi)


@pytest.mark.parametrize("method", ["1", "scatter"])
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.int16, torch.int8])
def test_lw_times_across_blocks_chunks_kinds_and_dtypes(method, dtype, monkeypatch):
    g = torch.Generator().manual_seed(2)
    monkeypatch.setattr(buckets, "_BLOCK_BYTES", 1 << 12)            # many row blocks
    cases = [("requant", 3_000_017, -128, 127, 9, 20000, 40),          # past one column chunk (2^14 at V = 256)
             ("requant", 1 << 16, -7, 9, 33, 50, 25),                  # an exact power of two: W constant
             ("residual", 70_001, -3000, 4000, 6, 1500, 0)]            # V = 7001: chunks of 512 columns
    for kind, m, a, b, n, t, n_exc in cases:
        if dtype == torch.int8 and kind == "residual":
            continue
        op = _op(kind, m, a if kind == "requant" else -128, b if kind == "requant" else 127, n)
        s, idx, z = _case(g, op, n, t, a, b, n_exc)
        if kind == "residual":
            idx, z = E0, E0                                            # (residual ops list no exceptions)
        e = _rand_ext(g, t)
        want = cp._lw_e_old(s, idx, z, op, e, t)
        sv = s.to(dtype)
        before = sv.clone()
        got = buckets.lw_times(sv, idx, z, op, e, method)
        assert torch.equal(got[0], want[0]) and torch.equal(got[1], want[1]), (kind, m)
        assert torch.equal(sv, before), "lw_times changed its input"
    wide = torch.tensor([[0, buckets.SPAN]])                           # a residual span past SPAN: today's code
    assert buckets.lw_times(wide, E0, E0, _op("residual", 70_001, n_rows=1), _rand_ext(g, 2), method) is None


@pytest.mark.skipif(not __import__("pvi.kernels.native", fromlist=["available"]).available(),
                    reason="the native library needs a C++ compiler (Linux)")
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.int16, torch.int8])
def test_native_lw_equals_the_torch_table_products(dtype, monkeypatch):
    """V-V1N's native kernel (``native_kernels.lw_affine``, used by ``lw_times`` on a CPU) gives the torch table
    products' integers, and declines out-of-range values and malformed operands."""
    from pvi.kernels import native as nk
    g = torch.Generator().manual_seed(5)
    cases = [("requant", 3_000_017, -128, 127, 37, 2048, 40), ("requant", 1 << 16, -7, 9, 33, 50, 25),
             ("residual", 70_001, -3000, 4000, 6, 1500, 0), ("residual", 1 << 20, -40, 40, 5, 300, 0)]
    for kind, m, a, b, n, t, n_exc in cases:
        if dtype == torch.int8 and kind == "residual" and b > 127:
            continue
        op = _op(kind, m, a if kind == "requant" else -128, b if kind == "requant" else 127, n)
        s, idx, z = _case(g, op, n, t, a, b, n_exc)
        if kind == "residual":
            idx, z = E0, E0
        e = _rand_ext(g, t)
        monkeypatch.setattr(nk, "_LW", True)
        got = buckets.lw_times(s.to(dtype), idx, z, op, e, "1")
        monkeypatch.setattr(nk, "_LW", False)
        want = buckets.lw_times(s.to(dtype), idx, z, op, e, "1")
        old = cp._lw_e_old(s, idx, z, op, e, t)
        for x, y, w in zip(got, want, old):
            assert torch.equal(x, y) and torch.equal(x, w), (kind, m, dtype)
    monkeypatch.setattr(nk, "_LW", True)
    lo_a, w0, tab, lo_v, w_v = buckets._affine(_op("requant", 3_000_017), -128, 127)
    cd = tab[:, 1:3].to(torch.int64)
    e = _rand_ext(g, 4)
    ok = torch.tensor([[-128, 0, 5, 127]], dtype=dtype)
    assert nk.lw_affine(ok, -128, cd, e) is not None
    if dtype != torch.int8:
        bad = torch.tensor([[-128, 0, 5, 128]], dtype=dtype)                        # past the table
        assert nk.lw_affine(bad, -128, cd, e) is None
    assert nk.lw_affine(ok, -128, cd, e[:3]) is None                                 # e of the wrong length
    e_bad = e.clone()
    e_bad[0, 0] = P                                                                  # a non-canonical coefficient
    assert nk.lw_affine(ok, -128, cd, e_bad) is None


@pytest.mark.skipif(not __import__("pvi.kernels.native", fromlist=["available"]).available(),
                    reason="the native library needs a C++ compiler (Linux)")
@pytest.mark.parametrize("dtype", [torch.int64, torch.int32, torch.int16, torch.int8])
def test_native_xbar_equals_int_times(dtype, monkeypatch):
    """V-V1N's native ``X^T e`` (``native_kernels.xbar``, used by ``buckets.xbar`` on a CPU) is
    ``extfield.int_times``'s product, at the int extremes and across column blocks."""
    from pvi.kernels import native as nk
    monkeypatch.setattr(nk, "_LW", True)
    monkeypatch.delenv(buckets.ENV, raising=False)
    g = torch.Generator().manual_seed(7)
    hi = {torch.int8: 127, torch.int16: (1 << 15) - 1}.get(dtype, (1 << 16) - 1)
    for t, k in ((1, 1), (5, 300), (2048, 600)):
        x = torch.randint(-hi, hi + 1, (t, k), generator=g)
        x[0, :] = hi
        if t > 1:
            x[1, :] = -hi
        e = _rand_ext(g, t)
        e[0] = P - 1
        want = ef.int_times(x.to(torch.float64).T, e, 31)
        assert torch.equal(nk.xbar(x.to(dtype), e), want), (dtype, t, k)        # the kernel at the extremes
        xs = x.clamp(-(buckets.SPAN // 2) + 1, buckets.SPAN // 2 - 1)            # buckets.xbar takes spans <= SPAN
        got = buckets.xbar(xs.to(dtype), e, "1")
        assert torch.equal(got, ef.int_times(xs.to(torch.float64).T, e, 31)), (dtype, t, k)
    assert nk.xbar(torch.zeros(3, 2, dtype=torch.int64), _rand_ext(g, 2)) is None        # e of the wrong length


@pytest.mark.parametrize("method", ["1", "scatter"])
def test_xbar_equals_int_times(method, monkeypatch):
    g = torch.Generator().manual_seed(3)
    monkeypatch.setattr(buckets, "_BLOCK_BYTES", 1 << 12)
    for t, k, bound in ((1, 5, 128), (33, 17, 128), (20000, 3, 128), (300, 4, (1 << 14) - 1), (64, 6, 3)):
        x = torch.randint(-bound, bound + 1, (t, k), generator=g)
        e = _rand_ext(g, t)
        want = ef.int_times(x.to(torch.float64).T, e, 31)
        assert torch.equal(buckets.xbar(x, e, method), want), (t, k, bound)
        if bound <= 128:                                               # an int8 input as it is
            xc = x.clamp(-128, 127)
            assert torch.equal(buckets.xbar(xc.to(torch.int8), e, method),
                               ef.int_times(xc.to(torch.float64).T, e, 31))
    assert buckets.xbar(torch.tensor([[1 << 16]]), _rand_ext(g, 1), method) is None
    assert buckets.xbar(torch.tensor([[0], [buckets.SPAN]]), _rand_ext(g, 2), method) is None


# ------------------------------------------------------------------ the final check, end to end
def _clamping_gpt(mode):
    """``test_cut_protocol``'s model that clamps often (a requant multiplier 8x): many exceptions."""
    from pvi.core.graph import CheapOp, requant_fn
    g = build_decoder(_TINY["gpt"], calib_tokens=8, seed=1)
    ops = list(g.ops)
    i = next(k for k, o in enumerate(ops)
             if isinstance(o, CheapOp) and o.params.get("requant", {}).get("kind") == "requant")
    p = dict(ops[i].params["requant"])
    p["mult"] = p["mult"] * 8
    ops[i] = dataclasses.replace(ops[i], fn=requant_fn(p), params={"requant": p})
    g = dataclasses.replace(g, ops=ops)
    params = proto.params_for(40, len(g.mat_ops), fiat_shamir=True, cut=True)
    if mode == "C":
        coms = proto.commit_graph(g, 4)
        return g, proto.Prover(g, commitments=coms), proto.Verifier(g.public(), params, "C", publics=coms.publics)
    v = proto.Verifier(g.public(), params, "Kpre", weights={op.name: (op.weight, op.bias) for op in g.mat_ops})
    v.precompute(proto.Challenger())
    return g, proto.Prover(g), v


def _captured(prover, v, x, monkeypatch):
    """One accepted V1 query's final-check arguments and derive's inputs, with ``PVI_V1_BUCKETS=1`` (so the cut
    values are the narrow ones part 4 keeps)."""
    got = {}
    real_final = cp._final_ok
    real_derive = v.derive

    def final(*args):
        got["args"] = args
        return real_final(*args)

    def derive(xq, claims):
        got["inputs"] = real_derive(xq, claims)
        return got["inputs"]
    monkeypatch.setenv(buckets.ENV, "1")
    monkeypatch.setattr(cp, "_final_ok", final)
    v.derive = derive
    try:
        out = cp.run_cut_query(prover, v, x)
    finally:
        monkeypatch.setattr(cp, "_final_ok", real_final)
        del v.derive                                                   # the class's method again
    assert out["accepted"], out["rejected_at"]
    return got["args"], got["inputs"]


@pytest.mark.parametrize("setup", [("gpt", "C", None), ("llama", "C", "cnn16c"), ("opt", "Kpre", None),
                                   ("clamping", "C", None), ("clamping", "Kpre", None)])
def test_final_claims_and_xbar_are_the_same_field_elements_in_every_mode(setup, monkeypatch):
    kind, mode, policy = setup
    if kind == "clamping":
        g, prover, v = _clamping_gpt(mode)
        x = tcp._x(30, seed=9)
    else:
        g, prover, v = tcp._setup(kind, mode, policy, True, kind == "llama")
        x = tcp._x()
    args, inputs = _captured(prover, v, x, monkeypatch)
    verifier, plan, outs, alpha, values, exc, cut_folds, items, ys, xbars, points = args
    layouts = {o.layout for o in plan.ops}
    if kind == "llama":
        assert "col" in layouts                                       # cnn16c: col matrices are cut
    if kind == "clamping":
        assert sum(int(i.numel()) for i, _ in exc.values()) > 20
    assert any(z.dtype == torch.int8 for z in values.values())         # part 4: the values as decoded
    mats = {op.name: op for op in verifier.graph.mat_ops}
    gen = torch.Generator().manual_seed(17)
    for trial in range(2):                                             # the query's own point, then random ones
        if trial == 0:
            outs_r, alpha_r = outs, alpha
        else:
            outs_r = [(p0, q0, _rand_ext(gen, rho.shape[0]), ph, qh) for p0, q0, rho, ph, qh in outs]
            alpha_r = _rand_ext(gen, 1)[0]
        points_r = [cp._points(plan, b, o[2]) for b, o in enumerate(outs_r)]
        results = {}
        for m in MODES:
            for widen in (False, True):
                vals = {k: z.to(torch.int64) for k, z in values.items()} if widen else values
                monkeypatch.setenv(buckets.ENV, m)
                cache = {}
                xb = {o.name: cp._xbar(mats[o.name], inputs[o.name], points_r[b][0], cache)
                      for b in range(len(outs_r)) for o in plan.instance_ops(b)}
                claims = list(cp._final_claims(verifier, plan, outs_r, alpha_r, vals, exc, cut_folds, items, ys,
                                               xb, points_r))
                results[m, widen] = (xb, claims)
        ref_xb, ref_claims = results["0", True]
        for key, (xb, claims) in results.items():
            assert xb.keys() == ref_xb.keys() and all(torch.equal(xb[k], ref_xb[k]) for k in xb), key
            assert len(claims) == len(ref_claims) == len(outs)
            for (i1, q1), (i2, q2) in zip(claims, ref_claims):
                assert torch.equal(i1, i2) and torch.equal(q1, q2), key
        if trial == 0:                                                 # honest: the claims are the GKR's
            assert all(torch.equal(i, o[3]) and torch.equal(q, o[4]) for (i, q), o in zip(ref_claims, outs))
    monkeypatch.delenv(buckets.ENV, raising=False)


# ------------------------------------------------------------------ part 3: the GKR and the logUp identity
def _instances(seed, ns):
    g = torch.Generator().manual_seed(seed)
    alpha = _rand_ext(g, 1)[0]
    out = []
    for n in ns:
        dense = int(torch.randint(1, (1 << n) + 1, (1,), generator=g))
        vals = torch.randint(-(1 << 20), 1 << 20, (dense,), generator=g)
        p = ef.lift(torch.zeros(1 << n, dtype=torch.int64))
        q = ef.lift(torch.ones(1 << n, dtype=torch.int64))
        p[:dense] = ef.const(1)
        q[:dense] = ef.sub(alpha, ef.lift(vals))
        out.append((p, q))
    return out


def _proved(fs, seed, ns):
    """Transcripts (as the verifier parses them) of instances of ``ns`` variables, and two verifier challengers in
    the prover's starting state."""
    if fs:
        chs = [proto.Challenger(fiat_shamir=True) for _ in range(3)]
        for c in chs:
            c.absorb(b"statement", b"v1n test")
    else:
        chs = [proto.Challenger(seed=seed) for _ in range(3)]
    trs = [gkr.prove(p, q, chs[0], f"cut/{b}/") for b, (p, q) in enumerate(_instances(seed, ns))]
    return [gkr.Transcript.from_bytes(tr.to_bytes(), tr.n) for tr in trs], chs[1], chs[2]


@pytest.mark.parametrize("fs", [True, False])
def test_verify_batch_gives_verify_outputs(fs):
    for seed, ns in enumerate([(1,), (2,), (5,), (1, 3, 2), (6, 1, 4, 4, 2), (3,) * 5]):
        trs, ch1, ch2 = _proved(fs, seed, ns)
        one = [gkr.verify(tr, ch1, f"cut/{b}/") for b, tr in enumerate(trs)]
        many = gkr.verify_batch(trs, ch2, [f"cut/{b}/" for b in range(len(trs))])
        assert all(o is not None for o in one) and many is not None
        for a, b in zip(one, many):
            assert all(torch.equal(x, y) for x, y in zip(a, b))
        if fs:                                                         # the same transcript afterwards
            assert ch1._state.digest(32) == ch2._state.digest(32)
        a, b = logup.verify_instances(trs, list(ns), _proved(fs, seed, ns)[1], batched=False), \
            logup.verify_instances(trs, list(ns), _proved(fs, seed, ns)[2], batched=True)
        assert all(torch.equal(x, y) for oa, ob in zip(a, b) for x, y in zip(oa, ob))


@pytest.mark.parametrize("fs", [True, False])
def test_verify_batch_rejects_every_tampered_coefficient_as_verify_does(fs):
    ns = (3, 1, 4)
    trs, _, _ = _proved(fs, 5, ns)
    for b, tr in enumerate(trs):
        flat = tr.flat()
        for i in range(flat.shape[0]):
            for c in (0, 7):
                bad = flat.clone()
                bad[i, c] = (bad[i, c] + 1) % P
                tampered = list(trs)
                tampered[b] = gkr.Transcript.from_bytes(struct_bytes(tr.n, bad), tr.n)
                _, ch1, ch2 = _proved(fs, 5, ns)
                seq = logup.verify_instances(tampered, list(ns), ch1, batched=False)
                bat = logup.verify_instances(tampered, list(ns), ch2, batched=True)
                if seq is None or bat is None:
                    assert seq is None and bat is None, (b, i, c)
                else:                                                  # an accepted tamper ends in other outputs
                    assert all(torch.equal(x, y) for oa, ob in zip(seq, bat) for x, y in zip(oa, ob))
    assert logup.verify_instances(trs, [3, 1, 5], _proved(fs, 5, ns)[2], batched=True) is None
    assert gkr.verify_batch(trs[:2], _proved(fs, 5, ns)[2], ["cut/0/"]) is None


def struct_bytes(n, flat):
    import struct
    return struct.pack("<B", n) + ef.pack(flat)


def test_one_challenge_is_drawn_as_today(monkeypatch):
    # Challenger._ext_one (count 1, the GKR's rounds) against today's loop: random keys and labels, interactive
    # and Fiat-Shamir, both nonbase settings; then crafted streams that hit both rejection rules and the next block
    for fs in (True, False):
        for nonbase in (False, True):
            a = proto.Challenger(fiat_shamir=fs, seed=3)
            b = proto.Challenger(fiat_shamir=fs, seed=3)
            for i in range(400):
                for c in (a, b):
                    c.absorb(b"m", i.to_bytes(4, "little"))
                monkeypatch.setenv(buckets.ENV, "1")
                x = a.ext(f"cut/0/r{i}", nonbase=nonbase)
                monkeypatch.setenv(buckets.ENV, "0")
                y = b.ext(f"cut/0/r{i}", nonbase=nonbase)
                assert torch.equal(x, y) and x.dtype == y.dtype == torch.int64 and tuple(x.shape) == (1, ef.D)
            assert torch.equal(a.ext("many", 3), b.ext("many", 3))
    import struct
    bad = 0xFFFFFFFF                                                   # above 2p: rejected

    def crafted(blocks):
        def stream(key, label, n):
            w = blocks[int(label.rsplit("/", 1)[1])]
            return struct.pack(f"<{len(w)}I", *w)[:n].ljust(n, b"\xff")
        return staticmethod(stream)
    cases = [[[1, 2, 0, 0, 0, 0, 0, 0, bad, 3, 4, 5, 6, 7, 8, 9, 10]],        # a base-field group, then a full one
             [[bad] * 48, [P + 1, 2 * P - 1, 3, 4, 5, 6, 7, 8]],                # nothing accepted, then reductions
             [[1, bad, 2, 3, 4, 5, 6, 7], [9, 9, 9, 0, 0, 0, 0, 1]],            # 7 accepted words: the next block
             [[5, 6, 0, 0, 0, 0, 0, 0] * 6, [1, 1, 0, 0, 0, 0, 0, 0, 2, 2, 2, 2, 2, 2, 2, 2]]]
    for blocks in cases:
        monkeypatch.setattr(proto.Challenger, "_stream", crafted(blocks))
        for nonbase in (False, True):
            ch = proto.Challenger(fiat_shamir=True)
            key = ch._key("ext/x")
            assert torch.equal(ch._ext_one(key, "x", nonbase), ch._ext_draw(key, "x", 1, nonbase)), (blocks, nonbase)
    monkeypatch.undo()


def test_inv_many():
    g = torch.Generator().manual_seed(4)
    for n in (1, 2, 3, 7, 64, 129):
        a = _rand_ext(g, n)
        a[a.sum(-1) == 0, 0] = 1
        assert torch.equal(ef.inv_many(a), torch.stack([ef.inv(x) for x in a]))
        assert torch.equal(ef.inv_many(a), ef.batch_inv(a))
    assert ef.inv_many(torch.zeros(0, ef.D, dtype=torch.int64)).shape == (0, ef.D)


def test_the_batch_inverted_logup_identity_has_the_cross_multiplied_verdict():
    g = torch.Generator().manual_seed(6)
    widths = (1, 3, 4, 300)
    tau, offs = logup.table(widths)
    alpha = proto.Challenger(fiat_shamir=True).ext("cut/alpha", nonbase=True)[0]
    w = torch.tensor(widths)[torch.randint(0, len(widths), (2000,), generator=g)]
    d = (torch.rand(2000, generator=g) * w).floor().to(torch.int64)
    m = logup.multiplicities(d, w, offs, tau.shape[0])
    v = torch.stack([d % P, w % P] + [torch.zeros_like(d)] * 6, 1)
    q = ef.sub(alpha, v)
    roots = []
    for part in torch.split(torch.arange(2000), [700, 1000, 300]):     # three instances' roots, honest
        roots.append(ef.frac_sum(ef.lift(torch.ones(part.numel(), dtype=torch.int64)), q[part]))
    cases = {"honest": m}
    moved = m.clone()
    i = int(torch.nonzero(moved)[0])
    moved[i] -= 1
    moved[i + 1] += 1
    cases["moved"] = moved
    zero = m.clone()
    zero[zero > 0] = 0
    cases["zero"] = zero
    for name, mm in cases.items():
        a, b = logup.logup_ok(roots, mm, alpha, tau, batched=False), logup.logup_ok(roots, mm, alpha, tau, batched=True)
        assert a == b == (name == "honest"), name
    assert logup.logup_ok([], torch.zeros(4, dtype=torch.int64), alpha, tau[:4], batched=True)


# ------------------------------------------------------------------ part 4 and every tamper in every mode
def _labels(scenarios, monkeypatch):
    out = {}
    for m in MODES:
        monkeypatch.setenv(buckets.ENV, m)
        out[m] = [f() for f in scenarios]
    monkeypatch.delenv(buckets.ENV, raising=False)
    return out


def test_out_of_range_values_are_rejected_with_the_same_label(monkeypatch):
    g, prover, v = tcp._setup("gpt", "C", None, True)
    x = tcp._x()
    plan, o = tcp._first_cut(v, x)
    res = next(op for op in plan.ops if op.kind == "residual")
    lo, hi = cutmod.delta_range(res.mult, res.sh)

    def edit(name, value):
        def cheat(values, witness, exc, mult):
            values[name] = values[name].clone()
            values[name][0, 0] = value
        return lambda: cp.run_cut_query(prover, v, x, cheat={"witness": cheat})["rejected_at"]
    scenarios = [edit(o.name, o.hi + 1), edit(o.name, o.lo_ - 1), edit(o.name, 1 << 20), edit(res.name, hi + 1),
                 edit(res.name, (1 << 15) + 3), lambda: cp.run_cut_query(prover, v, x)["rejected_at"]]
    got = _labels(scenarios, monkeypatch)
    assert got["0"] == got["1"] == got["scatter"]
    assert got["0"][:4] == ["range_or_shape"] * 4
    assert got["0"][4] == "cut_final" and got["0"][5] is None          # in range, past int16: the windows differ
    # the token rule comes first, in every mode
    g, prover, v = tcp._setup("llama", "C", "cnn16c", True, True, gen=True)
    base = build_decoder(_TINY["llama"], calib_tokens=8, seed=1, prune_last=True)
    from pvi.models.decoder import greedy_tokens
    xg = greedy_tokens(base, tcp._x(10, seed=4), 2)
    plan, o = tcp._first_cut(v, xg)

    def both(values, witness, exc, mult):
        values[o.name] = values[o.name].clone()
        values[o.name][0, 0] = o.hi + 5
        head = next(op for op in g.mat_ops if op.output == g.output_name)
        values[head.name] = values[head.name].clone()
        values[head.name][:, 0] = 0
        values[head.name][(int(xg[0, -2]) + 1) % values[head.name].shape[0], 0] = 1 << 20    # another argmax
    got = _labels([lambda: cp.run_cut_query(prover, v, xg, cheat={"witness": both})["rejected_at"]], monkeypatch)
    assert got["0"] == got["1"] == got["scatter"] == ["token"]


def test_every_tamper_has_the_same_label_in_every_mode(monkeypatch):
    g, prover, v = tcp._setup("gpt", "C", None, True)
    gk, prover_k, vk = tcp._setup("gpt", "Kpre", None, True)
    x = tcp._x()
    plan, o = tcp._first_cut(v, x)
    offs = plan.table()[1]

    def query(pr, ver, **kw):
        return lambda: cp.run_cut_query(pr, ver, x, **kw)["rejected_at"]

    def tamper(op_name, at, delta):
        def f(op, z):
            if op.name != op_name:
                return z
            z = z.clone()
            z.view(-1)[at % z.numel()] += delta
            return z
        return f

    def flip(blobs):
        b = bytearray(blobs[0])
        b[40] ^= 1
        return [bytes(b)] + blobs[1:]

    def fold(parts):
        parts = [t.clone() for t in parts]
        parts[0][0, 0] = (parts[0][0, 0] + 1) % P
        return parts

    def off_by_one(values, witness, exc, mult):
        mult[0] += 1

    def moved(values, witness, exc, mult):
        i = int(torch.nonzero(mult)[0])
        mult[i] -= 1
        mult[i + 1] += 1
    scenarios = [query(prover, v, forward_kwargs={"tamper": tamper(o.name, 7, 3 * o.widths[1])}),
                 query(prover_k, vk, forward_kwargs={"tamper": tamper(o.name, 7, 3 * o.widths[1])}),
                 query(prover, v, cheat={"witness": tcp._cheat_exception(o, offs, True)}),
                 query(prover, v, cheat={"witness": tcp._cheat_exception(o, offs, False)}),
                 query(prover_k, vk, cheat={"witness": off_by_one}), query(prover_k, vk, cheat={"witness": moved}),
                 query(prover_k, vk, cheat={"gkr": flip}), query(prover_k, vk, cheat={"fold": fold}),
                 query(prover, v), query(prover_k, vk)]
    gen = torch.Generator().manual_seed(8)
    mats = g.mat_ops
    for i in range(6):                                  # v1_bytes.py's random trace-tampering and single-value ones
        c = plan.ops[int(torch.randint(len(plan.ops), (1,), generator=gen))]
        at = int(torch.randint(1 << 30, (1,), generator=gen))
        scenarios.append(query(prover, v, forward_kwargs={"tamper": tamper(c.name, at, (1 + i % 3) * c.widths[1])}))
        op = mats[int(torch.randint(len(mats), (1,), generator=gen))]
        d = int(torch.randint(1, 1000, (1,), generator=gen)) * (1 if i % 2 else -1)
        scenarios.append(query(prover_k if i % 2 else prover, vk if i % 2 else v,
                               forward_kwargs={"tamper": tamper(op.name, at, d)}))
    got = _labels(scenarios, monkeypatch)
    assert got["0"] == got["1"] == got["scatter"], got
    assert got["0"][:8] == ["cut_final", "kpre_y", "cut_logup", "cut_final", "cut_multiplicities", "cut_logup",
                            got["0"][6], "kpre_y"] and got["0"][6] in ("cut_gkr", "cut_final")
    assert got["0"][8] is None and got["0"][9] is None
    assert all(label is not None for label in got["0"][10:])

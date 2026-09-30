"""Commitment plans (``pvi.fullcheck.plans``): exact per-op column counts, shared Merkle trees and
per-op codeword lengths, opt-in by policy name (``commit_graph(..., policy=)``, ``params_for(...,
plan=)``, a verifier with ``groups``).  The default -- ``policy="paper"`` -- is the report's
commitment and parameters, which the rest of the suite checks bit for bit.

Every plan must keep the report's guarantee (``soundness_bits >= lambda`` for every model, mode
and challenge kind), honest queries must be accepted, and every forgery must be rejected at the
check that catches it, by ``run_query``, by the streaming verifier and in the forms a GPU
verifier runs, interactive and under Fiat--Shamir -- with the proof in its default form and in
the compact wire encoding (``run_query(wire=True)``, :mod:`pvi.fullcheck.claimcodec`), whose
malformed messages are rejected too.
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import math
import random

import numpy as np
import pytest
import torch

from pvi.fullcheck import analytic
from pvi.fullcheck import claimcodec as cc
from pvi.fullcheck import field as fld
from pvi.fullcheck import protocol as proto
from pvi.fullcheck import reference as ref
from pvi.fullcheck.commitment import (GroupCommitment, MerkleTree, WeightCommitment, column_leaf, column_leaves,
                                      group_leaf, verify_multiproof)
from pvi.fullcheck.models import build_float_model
from pvi.fullcheck.plans import (MAX_N, REFERENCE_LAMBDA, column_bits, column_error_log2, exact_columns, next_pow2,
                                 plan_commitment)
from pvi.fullcheck.quantize import quantize_input, quantize_model
from pvi.fullcheck.transformer import CONFIGS, DecoderConfig, build_decoder

P = fld.P
POLICIES = ("tight", "cnn12", "R8")          # small codeword lengths: the tiny models commit in milliseconds


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("needs a GPU")
    return request.param


# -- the commitment layer ---------------------------------------------------------------------------

def _report_leaves(tag, weight, bias, n_points):
    """The report's ``WeightCommitment.build``, verbatim, for the equality checks."""
    n_rows = weight.shape[0]
    cols = np.empty((n_points, n_rows), dtype="<u4")
    for r0 in range(0, n_rows, 256):
        rows = weight[r0:r0 + 256].to(torch.int64)
        if bias is not None:
            rows = torch.cat([rows, bias[r0:r0 + 256].to(torch.int64)[:, None]], 1)
        cols[:, r0:r0 + rows.shape[0]] = fld.rs_encode(rows, n_points).to(torch.int32).numpy().T
    return [column_leaf(tag, c, cols[c]) for c in range(n_points)]


def _matrix(seed, rows, cols, bias=True):
    g = torch.Generator().manual_seed(seed)
    w = torch.randint(-127, 128, (rows, cols), generator=g).to(torch.int8)
    return w, (torch.randint(-(1 << 20), 1 << 20, (rows,), generator=g) if bias else None)


@pytest.mark.parametrize("bias", [True, False])
def test_streamed_column_digests_equal_the_reports(bias):
    w, b = _matrix(1, 37, 50, bias)
    ref = _report_leaves(b"m", w, b, 256)
    assert column_leaves(b"m", w, b, 256) == ref
    assert column_leaves(b"m", w, b, 256, max_host_bytes=0, row_chunk=5) == ref            # one row block at a time
    assert column_leaves(b"m", w, b, 256, max_host_bytes=4 * 256 * 12, row_chunk=5) == ref  # blocks of 3 chunks
    com = WeightCommitment.build(b"m", w, b, rate=4)
    assert com.tree.root == MerkleTree(ref).root and com.n_points == 256 and com.public.root == com.tree.root


def test_a_group_tree_binds_every_members_own_column_digests():
    mats = {"a": _matrix(2, 5, 30), "b": _matrix(3, 9, 20, bias=False), "c": _matrix(4, 3, 61)}
    n = 256
    group = GroupCommitment.build(b"g", mats, n, max_host_bytes=64)          # every member hashed in blocks
    own = {name: _report_leaves(name.encode(), w, b, n) for name, (w, b) in mats.items()}
    assert group.tree.root == MerkleTree([group_leaf(b"g", c, [own[m][c] for m in mats]) for c in range(n)]).root
    # a member's column digests are those of its own tree at this codeword length
    assert own["a"] == WeightCommitment.build(b"a", *mats["a"], rate=8).tree.levels[0]
    pub = group.public
    assert (pub.members, pub.n_points, pub.depth, pub.root) == (("a", "b", "c"), n, 8, group.tree.root)
    assert all(m.tree is None and m.public.root == b"" for m in group.members.values())
    idx = torch.tensor([0, 7, 100, 255])
    cols, proof = group.open(idx)
    stacked = torch.cat([fld.rs_encode(torch.cat([w.to(torch.int64)] + ([b[:, None]] if b is not None else []), 1),
                                       n)[:, idx] for w, b in mats.values()])
    assert torch.equal(cols, stacked)
    leaves, off = {}, 0
    for j, c in enumerate(idx.tolist()):
        digests, off = [], 0
        for name, (w, _) in mats.items():
            digests.append(column_leaf(name.encode(), c, cols[off:off + w.shape[0], j].numpy()))
            off += w.shape[0]
        leaves[c] = group_leaf(b"g", c, digests)
    assert verify_multiproof(pub.root, pub.depth, leaves, proof)


# -- exact column counts and plans -------------------------------------------------------------------

def test_exact_columns_are_the_smallest_that_meet_the_bound():
    rnd = random.Random(0)
    for _ in range(300):
        k = rnd.choice([1, 2, 3, rnd.randrange(2, 70000)])
        n = next_pow2(k) * rnd.choice([2, 4, 16, 64])
        bits = rnd.uniform(20, 220)
        t = exact_columns(k, n, bits)
        assert 1 <= t <= max(k, 1)
        assert t == k or column_error_log2(k, n, t) <= -bits
        assert t == 1 or column_error_log2(k, n, t - 1) > -bits
        # never more than the report's t = ceil(bits / log2 rate) at the report's lengths
        assert exact_columns(k, 4 * next_pow2(k), bits) <= math.ceil(bits / 2)


def _shapes(*ops):
    return [analytic.OpShape(*o) for o in ops]


def test_policies_give_their_codeword_lengths():
    ops = _shapes(("embed", 64, 50257, "embed"), ("q", 64, 65), ("down", 64, 3073), ("head", 50257, 64))
    lengths = {p: [o.n_points for o in plan_commitment(ops, p).ops] for p in ("tight", "cnn17", "R16", "R64")}
    assert lengths["tight"] == [4 * 65536, 4 * 128, 4 * 4096, 4 * 64]
    assert lengths["cnn17"] == [2 * 65536, 1 << 17, 1 << 17, 1 << 17]
    assert lengths["R16"] == [4 * 65536, 16 * 128, 16 * 4096, 16 * 64]        # the embedding keeps the base rate
    assert lengths["R64"] == [4 * 65536, 64 * 128, 64 * 4096, 64 * 64]
    assert [o.n_points for o in plan_commitment(ops, "tight", rate=8).ops][0] == 8 * 65536
    assert plan_commitment(ops, "paper") is None
    for bad in ("cnn0", "cnn28", "R3", "R1", "rate4", "cnn", "R", "Paper", "tight "):
        with pytest.raises(ValueError):
            plan_commitment(ops, bad)
    with pytest.raises(ValueError):                  # longer than the field's NTT
        plan_commitment(_shapes(("x", 4, 1 << 24)), "R16")


def _check_partition(plan, ops, bits):
    names = [o.name for o in ops]
    members = [m for _, ms in plan.groups for m in ms]
    assert sorted(members) == sorted(names)                              # each op in exactly one group
    for _, ms in plan.groups:
        assert len({plan.op(m).n_points for m in ms}) == 1               # one codeword length per tree
        assert [m for m in names if m in ms] == list(ms)                 # members in op order
    # groups of one length are runs of t: no member of a later run has a smaller t
    for n in {o.n_points for o in plan.ops}:
        runs = [sorted(exact_columns(plan.op(m).row_length, n, bits) for m in ms)
                for _, ms in plan.groups if plan.op(ms[0]).n_points == n]
        runs.sort()
        assert all(a[-1] < b[0] for a, b in zip(runs, runs[1:]))


def test_groups_are_runs_of_one_length_chosen_for_bytes():
    for cfg in (CONFIGS["gpt2"], CONFIGS["llama2-7b"], CONFIGS["qwen3-4b"], CONFIGS["opt-350m"]):
        ops = analytic.decoder_shapes(cfg)
        bits = column_bits(REFERENCE_LAMBDA, len(ops))
        for policy in ("tight", "cnn16", "R16"):
            plan = plan_commitment(ops, policy)
            _check_partition(plan, ops, bits)
            if policy != "cnn16":        # tall decoder matrices: a group never over-opens (one t per group)
                assert all(len({exact_columns(plan.op(m).row_length, plan.op(m).n_points, bits) for m in ms}) == 1
                           for _, ms in plan.groups)
            assert len(plan.groups) <= 6
    lenet = [op for op in _cnn("lenet5")[0].mat_ops]
    assert len(plan_commitment(lenet, "cnn16").groups) == 1          # few rows: one tree for the whole CNN
    _check_partition(plan_commitment(lenet, "tight"), lenet, column_bits(REFERENCE_LAMBDA, len(lenet)))


def test_a_build_of_a_few_blocks_gets_the_whole_models_groups():
    cfg = CONFIGS["opt-350m"]
    full = analytic.decoder_shapes(cfg)
    for policy in ("tight", "cnn16", "R16"):
        whole = plan_commitment(full, policy)
        of = {m: g for g, ms in whole.groups for m in ms}
        for layers in (1, 2):
            built = analytic.decoder_shapes(cfg, layers)
            plan = plan_commitment(built, policy, model_ops=full)
            names = {s.name for s in built}
            assert plan.groups == tuple((g, tuple(m for m in ms if m in names)) for g, ms in whole.groups
                                        if any(m in names for m in ms))
            assert all(plan.op(m).n_points == whole.op(m).n_points for m in names) and set(of) >= names


# -- the guarantee, for every model, policy, mode and challenge kind ----------------------------------

def _cnn(name):
    from torch import nn

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    if name == "mlp_mnist":
        model, shape = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                     nn.Linear(256, 10)), (784,)
    else:
        model, shape = build_float_model(name, 10), ((1, 28, 28) if name == "lenet5" else (3, 32, 32))
    graph = quantize_model(model.eval(), torch.randn(8, *shape, generator=g))
    return graph, quantize_input(graph, torch.randn(1, *shape, generator=g))


@pytest.fixture(scope="module")
def all_models():
    out = {name: _cnn(name)[0].mat_ops for name in ("mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar")}
    out.update({name: analytic.decoder_shapes(cfg) for name, cfg in CONFIGS.items()})
    return out


@pytest.mark.parametrize("policy", ["tight", "cnn16", "cnn17", "cnn18", "R8", "R16", "R64"])
def test_every_plan_meets_lambda_for_every_model_and_mode(policy, all_models):
    for name, ops in all_models.items():
        plan = plan_commitment(ops, policy)
        assert plan.shapes() == [(op.row_length, plan.op(op.name).n_points) for op in ops]
        for lam in (40, 80, 128):
            for fs in (False, True):
                paper = proto.params_for(lam, len(ops), fiat_shamir=fs)
                params = proto.params_for(lam, len(ops), fiat_shamir=fs, plan=plan)
                assert (params.reps, params.rate, params.fiat_shamir) == (paper.reps, 0, fs)
                cols = plan.op_columns(params.group_columns)
                bits = column_bits(lam, len(ops), fs)
                for (g, t), (g2, ms) in zip(params.group_columns, plan.groups):
                    assert g == g2 and t == max(exact_columns(plan.op(m).row_length, plan.op(m).n_points, bits)
                                                for m in ms)
                for mode in ("C", "K"):          # (Kpre: the bound of K, Freivalds alone)
                    got = proto.soundness_bits(params, plan.shapes(), mode, columns=cols)
                    assert got >= lam, (name, policy, lam, fs, mode, got)
                if policy == "tight":            # the report's lengths: exact t never opens more
                    assert max(cols) <= paper.columns
                with pytest.raises(ValueError, match="columns=plan.op_columns"):
                    proto.soundness_bits(params, plan.shapes(), "C")    # would credit every op the largest t
                assert proto.soundness_bits(params, plan.shapes(), "K") == proto.soundness_bits(
                    params, plan.shapes(), "K", columns=cols)


# -- the protocol with a plan ---------------------------------------------------------------------------

_TINY = {
    "gpt": DecoderConfig("tiny-gpt", 64, 2, 4, 4, 16, 128, 97, max_pos=64),
    "llama": DecoderConfig("tiny-llama", 64, 2, 4, 4, 16, 160, 97, norm="rmsnorm", mlp="swiglu",
                           pos="rope", bias=False, tied=False),
    "opt": DecoderConfig("tiny-opt", 64, 2, 4, 4, 16, 128, 97, mlp="relu", max_pos=64, embed_dim=32),
}


def _model(kind):
    if kind == "lenet5":
        return _cnn(kind)
    graph = build_decoder(_TINY[kind], calib_tokens=8, seed=1)
    return graph, torch.randint(0, _TINY[kind].vocab, (1, 9), generator=torch.Generator().manual_seed(2))


_BUILT: dict = {}


def _planned(kind, policy):
    """``(graph, x, commitment)`` for a tiny model under ``policy`` (built once per test session)."""
    if (kind, policy) not in _BUILT:
        graph, x = _model(kind)
        _BUILT[kind, policy] = graph, x, proto.commit_graph(graph, 4, policy=policy)
    return _BUILT[kind, policy]


def _verifiers(graph, coms, fiat_shamir, device="cpu", lam=40):
    params = proto.params_for(lam, len(graph.mat_ops), fiat_shamir=fiat_shamir, plan=coms.plan)
    kw = dict(publics=coms.publics, groups=coms.group_publics)
    return (proto.Verifier(graph.public(), params, "C", device=device, **kw),
            proto.Verifier(graph.public(), params, "C", stream=True, device=device, **kw))


@pytest.mark.parametrize("kind", ["lenet5", "gpt", "llama", "opt"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("fiat_shamir", [False, True])
@pytest.mark.parametrize("wire", [False, True])
def test_honest_queries_are_accepted_and_sized_as_planned(kind, policy, fiat_shamir, wire):
    graph, x, coms = _planned(kind, policy)
    prover = proto.Prover(graph, commitments=coms)
    claims = [prover.claims(x)[op.name] for op in graph.mat_ops]
    rows = {op.name: op.n_rows for op in graph.mat_ops}
    size = cc.field_size if wire else (lambda numel: 4 * numel)
    for v in _verifiers(graph, coms, fiat_shamir):
        res = proto.run_query(prover, v, x, seed=None if fiat_shamir else 1, wire=wire)
        assert res["accepted"], res["rejected_at"]
        assert res["bytes"]["claims"] == (len(cc.encode(claims)) if wire else 4 * sum(z.numel() for z in claims))
        assert res["bytes"]["u"] == size(v.params.reps * sum(op.row_length for op in graph.mat_ops))
        assert res["bytes"]["columns"] == size(sum(v.params.columns_for(g) * sum(rows[m] for m in ms)
                                                   for g, ms in coms.plan.groups))
        assert res["bytes"]["paths"] <= 32 * sum(v.params.columns_for(g) * gp.depth for g, gp in v.groups.items())
        if wire and not fiat_shamir:     # the same challenges as without wire: the same multiproofs
            assert res["bytes"]["paths"] == proto.run_query(prover, v, x, seed=1)["bytes"]["paths"]
    # modes K and Kpre take the plan's parameters (they open no columns)
    params = proto.params_for(40, len(graph.mat_ops), fiat_shamir=fiat_shamir, plan=coms.plan)
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    victim = graph.mat_ops[len(graph.mat_ops) // 2].name
    for mode in ("K", "Kpre"):
        v = proto.Verifier(graph.public(), params, mode, weights=weights)
        if mode == "Kpre":
            v.precompute(proto.Challenger(seed=3))
        assert proto.run_query(proto.Prover(graph), v, x, seed=2, wire=wire)["accepted"]
        tampered = proto.run_query(proto.Prover(graph), v, x, seed=3, wire=wire,
                                   forward_kwargs={"tamper": lambda op, z: z + 1 if op.name == victim else z})
        assert tampered["rejected_at"] == "freivalds"


def _kernel_shift(chi):
    """A nonzero ``d`` with ``chi d = 0 mod P``: an opened column shifted by it passes the code check."""
    r = chi.shape[0]
    a = [[int(v) % P for v in chi[i, :r].tolist()] + [(-int(chi[i, r])) % P] for i in range(r)]
    for col in range(r):
        piv = next(i for i in range(col, r) if a[i][col])
        a[col], a[piv] = a[piv], a[col]
        inv = pow(a[col][col], P - 2, P)
        a[col] = [(v * inv) % P for v in a[col]]
        for i in range(r):
            if i != col and a[i][col]:
                f = a[i][col]
                a[i] = [(vi - f * vc) % P for vi, vc in zip(a[i], a[col])]
    d = torch.zeros(chi.shape[1], dtype=torch.int64)
    d[:r] = torch.tensor([a[i][r] for i in range(r)])
    d[r] = 1
    return d


def _attacks(graph, coms):
    """``{name: (tamper, fold patch, open patch)}``: each forgery a test expects to be rejected."""
    mats = graph.mat_ops
    groups = coms.plan.groups
    big = max(groups, key=lambda g: len(g[1]))           # a group with the most members
    victim = big[1][-1]                                  # its last member (a slice inside the stacked columns)
    off = sum(op.n_rows for op in mats if op.name in big[1][:-1])
    rows = next(op.n_rows for op in mats if op.name == victim)
    final = mats[-1]
    key = next(g for g, ms in groups if victim in ms)
    state: dict = {}

    def tamper_mid(op, z):
        return z + 1 if op.name == mats[len(mats) // 2].name else z

    def tamper_final(op, z):            # one logit of a one-column claim raised by 1000
        if op.name == final.name:
            z = z.clone()
            z[0, 0] += 1000
        return z

    def fold_forged(fold):              # ... and the final op's u shifted so that Freivalds passes on it
        def patched(chis):
            us = fold(chis)
            u = us[final.name].clone()
            u[:, final.n_in] = (u[:, final.n_in] + chis[final.name][:, 0] * 1000) % P   # the bias coordinate
            us[final.name] = u
            return us
        return patched

    def fold_capture(fold):
        def patched(chis):
            state["chi"] = chis[victim]
            return fold(chis)
        return patched

    def at_victim(change):              # the victim's rows of its group's opening
        def patch(open_):
            def patched(cols):
                out = open_(cols)
                out[key] = change(*out[key])
                return out
            return patched
        return patch

    def kernel(o, pr):
        o = o.clone()
        o[off:off + rows, 0] = (o[off:off + rows, 0] + _kernel_shift(state["chi"])) % P
        return o, pr

    def misplaced(o, pr):               # the victim's first two opened columns swapped
        o = o.clone()
        o[off:off + rows, [0, 1]] = o[off:off + rows, [1, 0]]
        return o, pr

    def plus_one(o, pr):
        o = o.clone()
        o[off, 0] = (o[off, 0] + 1) % P
        return o, pr

    def setting(value):
        def change(o, pr):
            o = o.clone()
            o[off, 0] = value
            return o, pr
        return change

    out = {"claim+1": (tamper_mid, None, None),
           "forged kernel column": (None, fold_capture, at_victim(kernel)),
           "misplaced columns": (None, None, at_victim(misplaced)),
           "opened+1": (None, None, at_victim(plus_one)),
           "opened=P": (None, None, at_victim(setting(P))),
           "opened=-1": (None, None, at_victim(setting(-1))),
           "opened>int32": (None, None, at_victim(setting((1 << 32) + 5))),     # no int32 wire form
           "opened narrow": (None, None, at_victim(lambda o, pr: (o[:, :-1], pr))),
           "opened short": (None, None, at_victim(lambda o, pr: (o[:-1], pr))),
           "opened int32": (None, None, at_victim(lambda o, pr: (o.to(torch.int32), pr))),
           "path forged": (None, None, at_victim(lambda o, pr: (o, [bytes(32)] + pr[1:]))),
           "path short": (None, None, at_victim(lambda o, pr: (o, pr[:-1]))),
           "path long": (None, None, at_victim(lambda o, pr: (o, pr + [bytes(32)]))),
           "u+1": (None, lambda fold: lambda chis: {k: (u + (k == victim)) % P for k, u in fold(chis).items()}, None)}
    if final.has_bias and final.layout == "linear":      # (a CNN's classifier: one claim column)
        out["forged fold"] = (tamper_final, fold_forged, None)
    return out


EXPECTED = {"claim+1": "freivalds", "u+1": "freivalds", "forged fold": "columns_code",
            "forged kernel column": "columns_merkle", "misplaced columns": "columns_code",
            "opened+1": "columns_code", "opened=P": "columns_shape", "opened=-1": "columns_shape",
            "opened>int32": "columns_shape",
            "opened narrow": "columns_shape", "opened short": "columns_shape", "opened int32": "columns_shape",
            "path forged": "columns_merkle", "path short": "columns_merkle", "path long": "columns_merkle"}


TENSOR_FORMS = ("opened=-1", "opened>int32", "opened narrow", "opened short", "opened int32")
"""Forgeries of the opened columns' tensor (an entry outside ``[0, 2**31)``, another shape or dtype).
With ``wire`` the columns travel as one run of 31-bit field elements instead, which cannot carry
them (:func:`claimcodec.pack_field` refuses such entries, and no dtype is sent):
:func:`test_malformed_wire_messages_are_rejected_under_a_plan` forges that message itself."""


def _labels(graph, x, coms, verifiers, seed=0, wire=False):
    """``{attack: the check that rejects it}``; every verifier of ``verifiers`` must agree."""
    prover = proto.Prover(graph, commitments=coms)
    real = {"fold": prover.fold, "open": prover.open}
    out = {}
    for i, (name, (tamper, fold, open_)) in enumerate(_attacks(graph, coms).items()):
        if wire and name in TENSOR_FORMS:
            continue
        prover.fold = fold(real["fold"]) if fold else real["fold"]
        prover.open = open_(real["open"]) if open_ else real["open"]
        try:
            got = [proto.run_query(prover, v, x, seed=seed + i, wire=wire,
                                   forward_kwargs={"tamper": tamper} if tamper else None) for v in verifiers]
        finally:
            prover.fold, prover.open = real["fold"], real["open"]
        assert all(not r["accepted"] for r in got), name
        labels = {r["rejected_at"] for r in got}
        assert len(labels) == 1, (name, labels)            # run_query and the streaming verifier agree
        out[name] = labels.pop()
    return out


@pytest.mark.parametrize("kind", ["lenet5", "gpt", "llama"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("fiat_shamir", [False, True])
@pytest.mark.parametrize("wire", [False, True])
def test_every_forgery_is_rejected_at_its_check(kind, policy, fiat_shamir, wire):
    graph, x, coms = _planned(kind, policy)
    labels = _labels(graph, x, coms, _verifiers(graph, coms, fiat_shamir), wire=wire)
    expected = {k: v for k, v in EXPECTED.items() if not (wire and k in TENSOR_FORMS)}
    assert labels == {k: expected[k] for k in labels} and len(labels) >= len(expected) - 1


@pytest.mark.parametrize("paths", ["deferred", "int8", "deferred_int8"])
@pytest.mark.parametrize("wire", [False, True])
def test_the_gpu_forms_of_the_checks_give_the_same_verdicts(paths, wire, monkeypatch):
    if "deferred" in paths:
        monkeypatch.setattr(proto, "_defer", lambda device: True)
    if "int8" in paths:
        monkeypatch.setattr(proto, "int8_ok", lambda device: True)
    for kind, policy in (("lenet5", "cnn12"), ("llama", "R8")):
        graph, x, coms = _planned(kind, policy)
        labels = _labels(graph, x, coms, _verifiers(graph, coms, False), seed=10, wire=wire)
        assert labels == {k: EXPECTED[k] for k in labels}


@pytest.mark.parametrize("wire", [False, True])
def test_a_gpu_client_and_prover_give_the_cpu_verdicts(device, wire):
    graph, x, coms = _planned("llama", "R8")
    cpu = _labels(graph, x, coms, _verifiers(graph, coms, False), seed=20, wire=wire)
    assert _labels(graph, x, coms, _verifiers(graph, coms, False, device=device), seed=20, wire=wire) == cpu
    if device == "cuda":
        on_device = proto.Prover(graph, device="cuda", commitments=coms)
        for v in _verifiers(graph, coms, False):
            a, b = (proto.run_query(p, v, x, seed=1, wire=wire) for p in (proto.Prover(graph, commitments=coms),
                                                                            on_device))
            assert a["accepted"] and (b["accepted"], b["bytes"]) == (True, a["bytes"])


@pytest.mark.parametrize("paths", ["at_once", "deferred", "int8", "deferred_int8"])
def test_grouped_column_checks_give_the_reference_verdicts(paths, monkeypatch):
    """The batched check of group openings against ``reference.check_columns``, group by group
    (the verdicts and the exceptions), on single and double defects."""
    if "deferred" in paths:
        monkeypatch.setattr(proto, "_defer", lambda device: True)
    if "int8" in paths:
        monkeypatch.setattr(proto, "int8_ok", lambda device: True)
    for kind, policy in (("lenet5", "tight"), ("gpt", "R8")):
        graph, x, coms = _planned(kind, policy)
        v = _verifiers(graph, coms, False)[0]
        prover = proto.Prover(graph, commitments=coms)
        ch = proto.Challenger(seed=6)
        chis = {op.name: ch.folding(op.name, op.n_rows, v.params.reps) for op in graph.mat_ops}
        us = prover.fold(chis)
        cols = v.column_challenges(ch)
        openings = prover.open(cols)
        assert v.check_columns(chis, us, cols, openings) is ref.check_columns(v, chis, us, cols, openings) is None
        names = list(v.groups)
        variants = []
        for g in (names[0], names[-1]):
            o, pr = openings[g]
            last = o.shape[0] - 1                       # a row of the group's last member
            for row in (0, last):
                o1 = o.clone()
                o1[row, 0] = (o1[row, 0] + 1) % P
                variants.append({g: (o1, pr)})
            o2 = o.clone()
            o2[last, -1] = P
            variants += [{g: (o2, pr)}, {g: (o, [bytes(32)] + pr[1:])}, {g: (o, pr[:-1])}, {g: (o, pr + [bytes(32)])},
                         {g: (o[:-1], pr)}, {g: (o[:, :-1], pr)}, {g: (o.to(torch.int32), pr)}]
        a, b = names[0], names[-1]
        (oa, pa), (ob, pb) = openings[a], openings[b]
        ob1 = ob.clone()
        ob1[0, 0] = (ob1[0, 0] + 1) % P
        variants += [{a: (oa, [bytes(32)] + pa[1:]), b: (ob1, pb)}, {a: (oa[:, :-1], pa), b: (ob1, pb)},
                     {a: (oa, [bytes(32)] + pa[1:]), b: (ob, [0] + pb[1:])}, {a: (oa, pa[:-1]), b: None}]
        for var in variants:
            opened = dict(openings, **var)
            assert v.check_columns(chis, us, cols, opened) == ref.check_columns(v, chis, us, cols, opened), var.keys()
        bad_u = dict(us, **{graph.mat_ops[1].name: (us[graph.mat_ops[1].name] + 1) % P})
        assert v.check_columns(chis, bad_u, cols, openings) == ref.check_columns(v, chis, bad_u, cols, openings)             == "columns_code"
        for broken, exc in ((dict(openings, **{b: (ob, [0] + pb[1:])}), TypeError), (dict(openings, **{b: None}),
                                                                                     TypeError)):
            for impl in (v.check_columns, lambda *args: ref.check_columns(v, *args)):
                with pytest.raises(exc):
                    impl(chis, us, cols, broken)


def test_malformed_openings_fail_as_the_default_fails():
    graph, x, coms = _planned("gpt", "R8")
    v = _verifiers(graph, coms, False)[0]
    prover = proto.Prover(graph, commitments=coms)
    ch = proto.Challenger(seed=4)
    claims = prover.claims(x)
    inputs = v.derive(x, claims)
    chis = {op.name: ch.folding(op.name, op.n_rows, v.params.reps) for op in graph.mat_ops}
    us = prover.fold(chis)
    assert v.check_products(claims, inputs, chis, us)
    cols = v.column_challenges(ch)
    assert set(cols) == set(v.groups) and all(len(c) == v.params.columns_for(g) for g, c in cols.items())
    openings = prover.open(cols)
    assert v.check_columns(chis, us, cols, openings) is None
    first, last = list(v.groups)[0], list(v.groups)[-1]
    missing = dict(openings)
    del missing[last]
    with pytest.raises(KeyError):
        v.check_columns(chis, us, cols, missing)
    with pytest.raises(TypeError):
        v.check_columns(chis, us, cols, dict(openings, **{last: None}))
    with pytest.raises(TypeError):                   # a proof entry that is not bytes
        v.check_columns(chis, us, cols, dict(openings, **{last: (openings[last][0], [0] + openings[last][1][1:])}))
    # an earlier tree's verdict wins over a later exception
    bad = (openings[first][0], [bytes(32)] + openings[first][1][1:])
    assert v.check_columns(chis, us, cols, dict(openings, **{first: bad, last: None})) == "columns_merkle"
    # another group's opening in its place
    swapped = dict(openings, **{first: openings[last], last: openings[first]})
    assert v.check_columns(chis, us, cols, swapped) in ("columns_shape", "columns_code")


def test_the_fiat_shamir_transcript_absorbs_the_plan(monkeypatch):
    graph, x, coms = _planned("llama", "R8")
    absorbed = []
    absorb = proto.Challenger.absorb
    monkeypatch.setattr(proto.Challenger, "absorb", lambda self, label, blob: (
        absorbed.append((label, hashlib.sha256(blob).digest())), absorb(self, label, blob)))
    v = _verifiers(graph, coms, True)[0]
    prover = proto.Prover(graph, commitments=coms)
    assert proto.run_query(prover, v, x)["accepted"]
    labels = [label for label, _ in absorbed]
    assert [label for label in labels if label.startswith(b"group/")] == [b"group/" + g.encode() for g in v.groups]
    # every public parameter of the plan changes the challenges: a group's root, its t, its member order
    # (with the same messages), and each op's codeword length (absorbed in its op record)

    def first_columns(verifier):
        ch = proto.Challenger(fiat_shamir=True)
        proto._absorb_statement(ch, verifier, x)
        return ch.columns("probe", 1 << 20, 8)

    def variant(**fields):
        w = copy.copy(v)
        for k, val in fields.items():
            setattr(w, k, val)
        return w

    g0 = next(iter(v.groups))
    multi = next(g for g, p in v.groups.items() if len(p.members) > 1)
    name = graph.mat_ops[0].name
    changed = [variant(groups=dict(v.groups, **{g0: dataclasses.replace(v.groups[g0], root=bytes(32))})),
               variant(params=dataclasses.replace(v.params, group_columns=tuple(
                   (g, t + (g == g0)) for g, t in v.params.group_columns))),
               variant(groups=dict(v.groups, **{multi: dataclasses.replace(
                   v.groups[multi], members=v.groups[multi].members[::-1])})),
               variant(publics=dict(v.publics, **{name: dataclasses.replace(
                   v.publics[name], n_points=2 * v.publics[name].n_points)}))]
    base = first_columns(v)
    for w in changed:
        assert not torch.equal(first_columns(w), base)


def _flip_lowest_bit(packed: bytes, e: int, numel: int) -> bytes:
    """``packed`` (``numel`` field elements at 31 bits) with the lowest bit of element ``e`` flipped:
    element ``j G + g`` starts at bit ``31 j`` of the words ``g, G + g, ...`` (:func:`claimcodec.pack32`)."""
    lanes = (numel + 31) // 32
    j, g = divmod(e, lanes)
    word, bit = ((31 * j) >> 5) * lanes + g, (31 * j) & 31
    out = bytearray(packed)
    out[4 * word + bit // 8] ^= 1 << (bit % 8)
    return bytes(out)


def _unpacked(tensors: list, packed: bytes) -> list:
    """The tensors of ``tensors``' shapes that the verifier unpacks from ``packed``."""
    flat = torch.from_numpy(cc.unpack_field(packed, sum(t.numel() for t in tensors)).view(np.int32)).long()
    return [a.view(t.shape) for a, t in zip(flat.split([t.numel() for t in tensors]), tensors)]


@pytest.mark.parametrize("kind,policy", [("lenet5", "cnn12"), ("gpt", "R8"), ("llama", "tight")])
@pytest.mark.parametrize("fiat_shamir", [False, True])
def test_malformed_wire_messages_are_rejected_under_a_plan(kind, policy, fiat_shamir, monkeypatch):
    """The encoded claims, ``u`` and the opened columns of every tree (its members' rows side by side),
    malformed or altered, are rejected where the default flow rejects a malformed message of that
    kind -- a well-formed message of other elements where it rejects those elements -- by both
    verifiers."""
    graph, x, coms = _planned(kind, policy)
    verifiers = _verifiers(graph, coms, fiat_shamir)
    params = verifiers[0].params
    prover = proto.Prover(graph, commitments=coms)
    real = {"fold": prover.fold, "open": prover.open}
    u_shapes = [(params.reps, op.row_length) for op in graph.mat_ops]
    pack, encode = cc.pack_field, cc.encode

    def label(seed, wire=True):
        got = {proto.run_query(prover, v, x, seed=seed, wire=wire)["rejected_at"] for v in verifiers}
        assert len(got) == 1, got
        return got.pop()

    def sent(which, change):            # the prover's packed u or opened columns, as ``change`` makes them
        def patched(tensors):
            is_u = [tuple(t.shape) for t in tensors] == u_shapes
            return change(pack(tensors)) if is_u == (which == "u") else pack(tensors)
        monkeypatch.setattr(proto.claimcodec, "pack_field", patched)

    for i, change in enumerate((lambda b: b + b"\0", lambda b: b[:-1], lambda b: b"PVC0" + b[4:])):
        monkeypatch.setattr(proto.claimcodec, "encode", lambda zs, change=change: change(encode(zs)))
        assert label(i) == "range_or_shape"
        monkeypatch.undo()
    rows = {op.name: op.n_rows for op in graph.mat_ops}
    opened = [(params.columns_for(g), sum(rows[m] for m in ms)) for g, ms in coms.plan.groups]
    n_opened = sum(t * n for t, n in opened)
    cases = [("u", lambda b: b + b"\0", "freivalds"), ("u", lambda b: b[:-4], "freivalds"),
             ("columns", lambda b: b + b"\0", "columns_shape"), ("columns", lambda b: b[:-4], "columns_shape")]
    if n_opened % 32:                   # then the last element of the last lane is padding: its top bit set
        cases.append(("columns", lambda b: b[:-1] + bytes([b[-1] ^ 0x80]), "columns_shape"))
    for i, (which, change, want) in enumerate(cases):
        sent(which, change)
        assert label(10 + i) == want, (which, want)
        monkeypatch.undo()
    # one element changed: that of the middle op's u, and in the opened columns the first entry of the
    # last member of a tree of several (the first tree, if none has several)
    tree = next((i for i, (_, ms) in enumerate(coms.plan.groups) if len(ms) > 1), 0)
    members = coms.plan.groups[tree][1]
    targets = {"u": (params.reps * sum(op.row_length for op in graph.mat_ops[:len(graph.mat_ops) // 2]),
                     params.reps * sum(op.row_length for op in graph.mat_ops)),
               "columns": (sum(t * n for t, n in opened[:tree]) + sum(rows[m] for m in members[:-1]), n_opened)}
    for i, (which, (e, numel)) in enumerate(targets.items()):
        flip = lambda b, e=e, numel=numel: _flip_lowest_bit(b, e, numel)
        sent(which, flip)
        got = label(20 + i)
        monkeypatch.undo()

        def fold(chis):                 # the same elements, sent without wire
            us = real["fold"](chis)
            return dict(zip(us, _unpacked(list(us.values()), flip(pack(list(us.values()))))))

        def open_(cols):
            out = real["open"](cols)
            columns = [o.T for o, _ in out.values()]
            changed = _unpacked(columns, flip(pack(columns)))
            assert sum(int((a != b).sum()) for a, b in zip(changed, columns)) == 1
            return {k: (c.T, pr) for (k, (_, pr)), c in zip(out.items(), changed)}

        prover.fold, prover.open = (fold, real["open"]) if which == "u" else (real["fold"], open_)
        try:
            assert got is not None and got == label(20 + i, wire=False), which
        finally:
            prover.fold, prover.open = real["fold"], real["open"]
    # an entry outside [0, 2**31) has no wire form: the prover cannot even encode it
    g0 = coms.plan.groups[0][0]

    def outside(cols):
        out = real["open"](cols)
        o = out[g0][0].clone()
        o[0, 0] = -1
        return dict(out, **{g0: (o, out[g0][1])})

    prover.open = outside
    try:
        with pytest.raises(ValueError, match="field elements"):
            proto.run_query(prover, verifiers[0], x, seed=30, wire=True)
    finally:
        prover.open = real["open"]


def test_the_wire_transcript_absorbs_the_plan_and_the_encoded_bytes(monkeypatch):
    graph, x, coms = _planned("llama", "R8")
    absorbed = []
    absorb = proto.Challenger.absorb
    monkeypatch.setattr(proto.Challenger, "absorb", lambda self, label, blob: (
        absorbed.append((label, bytes(blob))), absorb(self, label, blob)))
    verifiers = _verifiers(graph, coms, True)
    prover = proto.Prover(graph, commitments=coms)
    transcripts = []
    for v in verifiers:
        absorbed.clear()
        assert proto.run_query(prover, v, x, wire=True)["accepted"]
        transcripts.append(list(absorbed))
    assert transcripts[0] == transcripts[1]         # the streaming verifier's is the batched one's
    labels = [label for label, _ in transcripts[0]]
    assert labels == [b"params", *(b"op/" + op.name.encode() for op in graph.mat_ops),
                      *(b"group/" + g.encode() for g in verifiers[0].groups), b"x", b"claims/PVC3", b"u/F31"]
    assert transcripts[0][-2][1] == cc.encode([prover.claims(x)[op.name] for op in graph.mat_ops])
    reps = verifiers[0].params.reps
    assert len(transcripts[0][-1][1]) == cc.field_size(reps * sum(op.row_length for op in graph.mat_ops))
    # the statement -- the plan's trees, their members and t -- is absorbed as without wire
    absorbed.clear()
    assert proto.run_query(prover, verifiers[0], x)["accepted"]
    assert absorbed[:len(labels) - 2] == transcripts[0][:-2]


def test_a_verifier_refuses_a_key_that_does_not_fit():
    graph, x, coms = _planned("gpt", "R8")
    params = proto.params_for(40, len(graph.mat_ops), plan=coms.plan)
    kw = dict(publics=coms.publics, groups=coms.group_publics)
    proto.Verifier(graph.public(), params, "C", **kw)
    with pytest.raises(ValueError, match="mode-C"):
        proto.Verifier(graph.public(), params, "K", **kw)
    with pytest.raises(ValueError, match="no column count"):
        proto.Verifier(graph.public(), proto.params_for(40, len(graph.mat_ops)), "C", **kw)
    groups = dict(coms.group_publics)
    g0, g1 = list(groups)[:2]
    moved = dict(groups, **{g0: dataclasses.replace(groups[g0], members=groups[g0].members + groups[g1].members[:1])})
    with pytest.raises(ValueError, match="exactly one"):
        proto.Verifier(graph.public(), params, "C", publics=coms.publics, groups=moved)
    other = dict(groups, **{g0: dataclasses.replace(groups[g0], n_points=2 * groups[g0].n_points)})
    with pytest.raises(ValueError, match="another codeword length"):
        proto.Verifier(graph.public(), params, "C", publics=coms.publics, groups=other)


def test_the_default_policy_is_the_reports_commitment():
    graph, x = _model("gpt")
    coms = proto.commit_graph(graph, 4)
    assert isinstance(coms, dict) and coms.plan is None and coms.groups == {} and coms.group_publics == {}
    for op in graph.mat_ops:
        assert coms[op.name].tree.root == MerkleTree(_report_leaves(op.name.encode(), op.weight, op.bias,
                                                                     4 * next_pow2(op.row_length))).root
    assert coms.publics == {k: c.public for k, c in coms.items()}
    params = proto.params_for(40, len(graph.mat_ops))
    bits = 40 + math.log2(2 * len(graph.mat_ops))
    assert params == proto.SecurityParams(40, math.ceil(bits / fld.LOG2_P), 4, math.ceil(bits / 2))
    assert params.group_columns == ()


# -- the proof-size model ----------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["lenet5", "gpt", "opt"])
@pytest.mark.parametrize("policy", ("paper",) + POLICIES)
def test_the_proof_size_model_is_what_run_query_sends(kind, policy):
    graph, x = _model(kind) if policy == "paper" else _planned(kind, policy)[:2]
    coms = proto.commit_graph(graph, 4) if policy == "paper" else _planned(kind, policy)[2]
    prover = proto.Prover(graph, commitments=coms)
    _, claims = graph.forward(x)
    cols = {k: z.shape[1] for k, z in claims.items()}
    if kind != "lenet5":
        assert list(cols.values()) == list(analytic.decoder_claim_columns(_TINY[kind], x.shape[1]).values())
    for fs in (False, True):
        params = proto.params_for(40, len(graph.mat_ops), fiat_shamir=fs, plan=coms.plan)
        v = proto.Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics)
        want = analytic.proof_bytes(graph.mat_ops, params, cols, plan=coms.plan)
        encoded = len(cc.encode([claims[op.name] for op in graph.mat_ops]))
        wired = analytic.proof_bytes(graph.mat_ops, params, cols, plan=coms.plan, wire_claims=encoded)
        assert wired["paths"] == want["paths"] and wired["claims"] == encoded
        paths = []
        for seed in range(12):
            got = proto.run_query(prover, v, x, seed=seed)["bytes"]
            assert all(got[k] == want[k] for k in ("claims", "u", "columns"))
            paths.append(got["paths"])
            if seed < 2:             # with wire (interactive: the same challenges, so the same multiproofs)
                sent = proto.run_query(prover, v, x, seed=seed, wire=True)["bytes"]
                assert sent == dict(wired, paths=sent["paths"] if fs else got["paths"])
        assert abs(np.mean(paths) - want["paths"]) < 0.1 * want["paths"]
        k_modes = analytic.proof_bytes(graph.mat_ops, params, cols, plan=coms.plan, mode="Kpre")
        assert k_modes == {"claims": want["claims"], "u": 0, "columns": 0, "paths": 0.0}
        k_wired = analytic.proof_bytes(graph.mat_ops, params, cols, plan=coms.plan, mode="Kpre", wire_claims=encoded)
        assert k_wired == {"claims": encoded, "u": 0, "columns": 0, "paths": 0.0}
    setup = analytic.setup_size(graph.mat_ops, plan=coms.plan)
    assert setup["encoded_entries"] == sum(c.weight.shape[0] * c.n_points for c in coms.values())
    assert setup["trees"] == (len(coms.groups) or len(coms))
    assert setup["max_n"] <= MAX_N

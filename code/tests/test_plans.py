"""Commitment plans (``pvi.fullcheck.plans``): exact per-op column counts, shared Merkle trees,
per-op codeword lengths and, under the policies with the suffix ``c``, per-op layouts (the col
layout, with the ops reading one tensor fused, and lookup tables for the embeddings), opt-in by
policy name (``commit_graph(..., policy=)``, ``params_for(..., plan=)``, a verifier with ``groups``
and ``tables``), also on decoders built with their last block pruned (``prune_last``).  The default --
``policy="paper"`` -- is the report's commitment and parameters, which the rest of the suite
checks bit for bit.

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
                                      group_leaf, multiproof_size, verify_multiproof)
from pvi.fullcheck.models import build_float_model
from pvi.fullcheck.pipeline import wire_openings
from pvi.fullcheck.plans import (MAX_N, REFERENCE_LAMBDA, col_name, column_bits, column_error_log2, exact_columns,
                                 next_pow2, plan_commitment)
from pvi.fullcheck.quantize import quantize_input, quantize_model
from pvi.fullcheck.transformer import CONFIGS, DecoderConfig, build_decoder

P = fld.P
POLICIES = ("tight", "cnn12", "R8", "tightc", "cnn12c", "R8c")   # small codeword lengths: the tiny models
                                                                    # commit in milliseconds


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
    group = GroupCommitment.build(b"g", {name: WeightCommitment.member(name.encode(), w, b, n)   # hashed in blocks
                                         for name, (w, b) in mats.items()}, max_host_bytes=64)
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
    lengths = {p: [o.n_points for o in plan_commitment(ops, p).matrices] for p in ("tight", "cnn17", "R16", "R64")}
    assert lengths["tight"] == [4 * 65536, 4 * 128, 4 * 4096, 4 * 64]
    assert lengths["cnn17"] == [2 * 65536, 1 << 17, 1 << 17, 1 << 17]
    assert lengths["R16"] == [4 * 65536, 16 * 128, 16 * 4096, 16 * 64]        # the embedding keeps the base rate
    assert lengths["R64"] == [4 * 65536, 64 * 128, 64 * 4096, 64 * 64]
    assert [o.n_points for o in plan_commitment(ops, "tight", rate=8).matrices][0] == 8 * 65536
    assert plan_commitment(ops, "paper") is None
    for bad in ("cnn0", "cnn28", "R3", "R1", "rate4", "cnn", "R", "Paper", "tight ", "cnn016", "R016", "R08"):
        with pytest.raises(ValueError):
            plan_commitment(ops, bad)
    with pytest.raises(ValueError):                  # longer than the field's NTT
        plan_commitment(_shapes(("x", 4, 1 << 24)), "R16")


def _check_partition(plan, ops, bits):
    names = [o.name for o in ops]
    members = [m for _, ms in plan.groups for m in ms]
    assert sorted(members) == sorted(names)                              # each op in exactly one group
    for _, ms in plan.groups:
        assert len({plan.matrix(m).n_points for m in ms}) == 1               # one codeword length per tree
        assert [m for m in names if m in ms] == list(ms)                 # members in op order
    # groups of one length are runs of t: no member of a later run has a smaller t
    for n in {o.n_points for o in plan.matrices}:
        runs = [sorted(exact_columns(plan.matrix(m).row_length, n, bits) for m in ms)
                for _, ms in plan.groups if plan.matrix(ms[0]).n_points == n]
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
                assert all(len({exact_columns(plan.matrix(m).row_length, plan.matrix(m).n_points, bits)
                                for m in ms}) == 1 for _, ms in plan.groups)
            assert len(plan.groups) <= 6
    lenet = [op for op in _cnn("lenet5")[0].mat_ops]
    assert len(plan_commitment(lenet, "cnn16").groups) == 1          # few rows: one tree for the whole CNN
    _check_partition(plan_commitment(lenet, "tight"), lenet, column_bits(REFERENCE_LAMBDA, len(lenet)))


@pytest.mark.parametrize("model", ["opt-350m", "llama2-7b"])
@pytest.mark.parametrize("prune_last", [False, True])
def test_a_build_of_a_few_blocks_gets_the_whole_models_groups(model, prune_last):
    """A build's matrices and groups are the whole model's restricted to the built ops -- its blocks
    being the model's last ones (with ``prune_last`` the last is the pruned one), under their names."""
    cfg = CONFIGS[model]
    full = analytic.decoder_shapes(cfg, prune_last=prune_last)
    for policy in ("tight", "cnn16", "R16", "tightc", "cnn16c", "R16c"):
        whole = plan_commitment(full, policy)
        of = {m: g for g, ms in whole.groups for m in ms}
        for layers in (1, 2):
            built = analytic.decoder_shapes(cfg, layers, prune_last=prune_last)
            first = next(i for i, op in enumerate(built) if op.name == "q0")
            skip = (len(full) - len(built))                  # the ops of the model's first L - layers blocks
            rename = {op.name: full[i if i < first else i + skip].name for i, op in enumerate(built)}

            def named(m):
                return dataclasses.replace(m, name=col_name(rename[o] for o in m.members) if m.members
                                           else rename[m.name], members=tuple(rename[o] for o in m.members))

            plan = plan_commitment(built, policy, model_ops=full)
            matrices = [named(m) for m in plan.matrices]
            names = {m.name for m in matrices}
            assert matrices == [m for m in whole.matrices if m.name in names]
            assert set(of) | {m.name for m in whole.tables} >= names
            wanted = tuple((g, tuple(m for m in ms if m in names)) for g, ms in whole.groups if set(ms) & names)
            got = tuple((g, tuple(named(plan.matrix(m)).name for m in ms)) for g, ms in plan.groups)
            # the same trees (members and lengths; a pruned build meets them in another order)
            assert sorted((g.split("_n")[1], ms) for g, ms in got) == sorted((g.split("_n")[1], ms) for g, ms in wanted)
            if not prune_last:           # every block alike: the groups come in the model's order, named alike
                assert got == wanted


# -- the guarantee, for every model, policy, mode and challenge kind ----------------------------------

def _cnn(name):
    from torch import nn

    torch.manual_seed(0)
    g = torch.Generator().manual_seed(1)
    if name == "mlp_mnist":
        model, shape = nn.Sequential(nn.Linear(784, 512), nn.ReLU(), nn.Linear(512, 256), nn.ReLU(),
                                     nn.Linear(256, 10)), (784,)
    elif name == "mlp_wide":             # a hidden layer far wider than its input: the col layout's case
        model, shape = nn.Sequential(nn.Linear(784, 32), nn.ReLU(), nn.Linear(32, 256), nn.ReLU(),
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


@pytest.mark.parametrize("policy", [p + c for p in ("tight", "cnn16", "cnn17", "cnn18", "R8", "R16", "R64")
                                    for c in ("", "c")])
def test_every_plan_meets_lambda_for_every_model_and_mode(policy, all_models):
    for name, ops in all_models.items():
        plan = plan_commitment(ops, policy)
        of = {op.name: op for op in ops}
        # one check per encoded matrix, of its message length: a row matrix's k, a col matrix's stacked rows;
        # under a c policy every embedding table is a lookup table, which adds no term
        assert plan.shapes() == [(sum(of[o].n_rows for o in m.members) if m.members else of[m.name].row_length,
                                  m.n_points) for m in plan.coded]
        assert sorted(o for m in plan.matrices for o in m.ops) == sorted(of)
        assert policy.endswith("c") or all(m.layout == "row" for m in plan.matrices)
        assert [m.name for m in plan.tables] == ([op.name for op in ops if op.layout == "embed"]
                                                 if policy.endswith("c") else [])
        assert all(m.n_points == next_pow2(m.row_length) and m.n_rows == of[m.name].n_rows for m in plan.tables)
        per_op = [(op.row_length, 0) for op in ops]      # modes K and Kpre: one Freivalds check per op
        own_rows = [(op.row_length, 0) for op in ops if op.layout != "embed"]     # K, Kpre: lookups=True
        for lam in (40, 80, 128):
            for fs in (False, True):
                paper = proto.params_for(lam, len(ops), fiat_shamir=fs)
                params = proto.params_for(lam, len(ops), fiat_shamir=fs, plan=plan)
                assert (params.reps, params.rate, params.fiat_shamir) == (paper.reps, 0, fs)
                cols = plan.matrix_columns(params.group_columns)
                bits = column_bits(lam, len(ops), fs)
                for (g, t), (g2, ms) in zip(params.group_columns, plan.groups):
                    assert g == g2 and t == max(exact_columns(plan.matrix(m).row_length, plan.matrix(m).n_points, bits)
                                                for m in ms)
                got = proto.soundness_bits(params, plan.shapes(), "C", columns=cols)
                assert got >= lam, (name, policy, lam, fs, got)
                # each matrix meets its own budget: p**-r and its exact column term, each <= 2**-beta
                for (m_len, n), t in zip(plan.shapes(), cols):
                    assert column_error_log2(m_len, n, t) <= -bits and params.reps * fld.LOG2_P >= bits
                assert proto.soundness_bits(params, per_op, "K") >= lam      # (Kpre: the bound of K, Freivalds alone)
                assert proto.soundness_bits(params, own_rows, "K") >= proto.soundness_bits(params, per_op, "K")
                if policy == "tight":            # the report's lengths: exact t never opens more
                    assert max(cols) <= paper.columns
                with pytest.raises(ValueError, match="columns=plan.matrix_columns"):
                    proto.soundness_bits(params, plan.shapes(), "C")    # would credit every matrix the largest t
                assert proto.soundness_bits(params, per_op, "K") == proto.soundness_bits(
                    params, per_op, "K", columns=[0] * len(ops))


# -- the protocol with a plan ---------------------------------------------------------------------------

_TINY = {
    "gpt": DecoderConfig("tiny-gpt", 64, 2, 4, 4, 16, 128, 97, max_pos=64),
    "llama": DecoderConfig("tiny-llama", 64, 2, 4, 4, 16, 160, 97, norm="rmsnorm", mlp="swiglu",
                           pos="rope", bias=False, tied=False),
    "opt": DecoderConfig("tiny-opt", 64, 2, 4, 4, 16, 128, 97, mlp="relu", max_pos=64, embed_dim=32),
    "qwen": DecoderConfig("tiny-qwen", 64, 2, 8, 2, 16, 160, 97, norm="rmsnorm", mlp="swiglu",       # GQA, q/k norm
                          pos="rope", rope_theta=1e6, bias=False, qk_norm=True),
}


def _model(kind):
    """A tiny model and a query: a CNN, or a decoder of :data:`_TINY` (``<decoder>-pruned``: built with
    its last block pruned)."""
    if kind in ("lenet5", "mlp_wide"):
        return _cnn(kind)
    base, _, variant = kind.partition("-")
    graph = build_decoder(_TINY[base], calib_tokens=8, seed=1, prune_last=variant == "pruned")
    return graph, torch.randint(0, _TINY[base].vocab, (1, 9), generator=torch.Generator().manual_seed(2))


_BUILT: dict = {}


def _planned(kind, policy):
    """``(graph, x, commitment)`` for a tiny model under ``policy`` (built once per test session)."""
    if (kind, policy) not in _BUILT:
        graph, x = _model(kind)
        _BUILT[kind, policy] = graph, x, proto.commit_graph(graph, 4, policy=policy)
    return _BUILT[kind, policy]


def _verifiers(graph, coms, fiat_shamir, device="cpu", lam=40):
    params = proto.params_for(lam, len(graph.mat_ops), fiat_shamir=fiat_shamir, plan=coms.plan)
    kw = dict(publics=coms.publics, groups=coms.group_publics, tables=coms.table_publics)
    return (proto.Verifier(graph.public(), params, "C", device=device, **kw),
            proto.Verifier(graph.public(), params, "C", stream=True, device=device, **kw))


def _folded(graph, coms):
    """The weight ops checked with Freivalds (row layout) under ``coms``' plan."""
    return [op for op in graph.mat_ops if coms.plan.matrix_of(op.name).layout == "row"]


def _claim_bytes(graph, tables, claims: dict, wire: bool) -> int:
    """The claims' bytes as they are sent: 4 each, a lookup table's 1, or with ``wire`` the ``PVC3`` bytes
    of all but the tables' and their int8 rows."""
    if wire:
        return (len(cc.encode([claims[op.name] for op in graph.mat_ops if op.name in claims and op.name not in tables]))
                + len(cc.pack_rows([claims[name] for name in tables])))
    return sum((1 if name in tables else 4) * z.numel() for name, z in claims.items())


def _looked_up(graph, x, tables) -> dict[str, list[int]]:
    """The ids each lookup table of ``tables`` looks up on query ``x``."""
    ids = {}
    graph.forward(x, watch=lambda op, xin: ids.update({op.name: xin.reshape(-1).tolist()}) if op.name in tables
                  else None)
    return ids


DECODERS = ("gpt", "llama", "opt", "qwen")
PRUNED = tuple(k + "-pruned" for k in DECODERS)


@pytest.mark.parametrize("kind", ["lenet5", "mlp_wide", *DECODERS, *PRUNED])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("fiat_shamir", [False, True])
@pytest.mark.parametrize("wire", [False, True])
def test_honest_queries_are_accepted_and_sized_as_planned(kind, policy, fiat_shamir, wire):
    graph, x, coms = _planned(kind, policy)
    prover = proto.Prover(graph, commitments=coms)
    claims = prover.claims(x)
    rows = {name: c.public.n_rows for name, c in coms.items()}         # each encoded matrix's column length
    size = cc.field_size if wire else (lambda numel: 4 * numel)
    ids = _looked_up(graph, x, coms.tables)
    assert bool(coms.tables) == (policy.endswith("c") and kind not in ("lenet5", "mlp_wide"))
    table_paths = 32 * sum(multiproof_size(set(ids[name]), t.public.depth) for name, t in coms.tables.items())
    for v in _verifiers(graph, coms, fiat_shamir):
        res = proto.run_query(prover, v, x, seed=None if fiat_shamir else 1, wire=wire)
        assert res["accepted"], res["rejected_at"]
        assert res["bytes"]["claims"] == _claim_bytes(graph, coms.tables, claims, wire)
        assert res["bytes"]["u"] == size(v.params.reps * sum(op.row_length for op in _folded(graph, coms)))
        assert res["bytes"]["columns"] == size(sum(v.params.columns_for(g) * sum(rows[m] for m in ms)
                                                   for g, ms in coms.plan.groups))
        assert table_paths < res["bytes"]["paths"] <= table_paths + 32 * sum(
            v.params.columns_for(g) * gp.depth for g, gp in v.groups.items())
        if wire and not fiat_shamir:     # the same challenges as without wire: the same multiproofs
            assert res["bytes"]["paths"] == proto.run_query(prover, v, x, seed=1)["bytes"]["paths"]
    # modes K and Kpre take the plan's parameters (they open no columns); with lookups the verifier reads
    # the embedding rows itself and the prover sends no claims for them
    params = proto.params_for(40, len(graph.mat_ops), fiat_shamir=fiat_shamir, plan=coms.plan)
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    victim = graph.mat_ops[len(graph.mat_ops) // 2].name
    for mode in ("K", "Kpre"):
        for lookups in (False, True):
            v = proto.Verifier(graph.public(), params, mode, weights=weights, lookups=lookups)
            if mode == "Kpre":
                v.precompute(proto.Challenger(seed=3))
            res = proto.run_query(proto.Prover(graph), v, x, seed=2, wire=wire)
            own = {op.name for op in graph.mat_ops if lookups and op.layout == "embed"}
            assert res["accepted"] and res["bytes"] == {"claims": _claim_bytes(graph, (), {
                k: z for k, z in claims.items() if k not in own}, wire), "u": 0, "columns": 0, "paths": 0}
            tampered = proto.run_query(proto.Prover(graph), v, x, seed=3, wire=wire,
                                       forward_kwargs={"tamper": lambda op, z: z + 1 if op.name == victim else z})
            assert tampered["rejected_at"] == "freivalds"


def _kernel_shift(left):
    """A nonzero ``d`` with ``left d = 0 mod P`` (``left``: of rank below its column count): an
    opened column shifted by it passes the code check that folds the column with ``left``."""
    a = [[int(v) % P for v in row] for row in left.tolist()]
    pivots: list[int] = []
    for col in range(len(a[0])):
        piv = next((i for i in range(len(pivots), len(a)) if a[i][col]), None)
        if piv is None:
            continue
        r = len(pivots)
        a[r], a[piv] = a[piv], a[r]
        inv = pow(a[r][col], P - 2, P)
        a[r] = [(v * inv) % P for v in a[r]]
        for i in range(len(a)):
            if i != r and a[i][col]:
                f = a[i][col]
                a[i] = [(vi - f * vr) % P for vi, vr in zip(a[i], a[r])]
        pivots.append(col)
        if len(pivots) == len(a):
            break
    free = next(c for c in range(len(a[0])) if c not in pivots)
    d = torch.zeros(len(a[0]), dtype=torch.int64)
    d[free] = 1
    for i, c in enumerate(pivots):
        d[c] = (-a[i][free]) % P
    return d


def _attacks(graph, coms, x):
    """``{name: (tamper, fold patch, open patch)}``: each forgery a test expects to be rejected, on query ``x``."""
    mats = graph.mat_ops
    stacked = [m.name for m in coms.plan.matrices if m.layout == "col"]
    groups = [g for g in coms.plan.groups if set(g[1]) & set(stacked)] or coms.plan.groups
    key, members = max(groups, key=lambda g: len(g[1]))  # a group with the most members (and a col matrix)
    victim = [m for m in members if m in stacked or not stacked][-1]     # its last (col) matrix: a slice of
    matrix = coms.plan.matrix(victim)                                   # the stacked columns
    off = sum(coms[m].public.n_rows for m in members[:members.index(victim)])
    rows = coms[victim].public.n_rows
    final = mats[-1]
    folded = _folded(graph, coms)
    # u + 1 (every entry) passes Freivalds where [X ; 1] sums to 0 in every column -- which a normed
    # input of one column (a pruned block's q) can: a folded op of several claim columns then
    wide = [op.name for op in folded if dict(zip((o.name for o in mats), graph.claim_columns(x)))[op.name] > 1]
    wide = wide or [op.name for op in folded]
    u_victim = victim if matrix.layout == "row" else wide[len(wide) // 2]
    state: dict = {}
    hit = off                           # a row of the victim's block that its code check reads
    if matrix.layout == "col":          # its check folds the columns with w = chi' [X ; 1]^T: a shift in the kernel
        first = next(op for op in mats if op.name == matrix.members[0])     # of [X ; 1]^T passes it for every chi'
        xt = first.unfold(graph.forward(x)[0][first.inputs[0]]).T
        state["left"] = torch.cat([xt, torch.ones(xt.shape[0], 1, dtype=torch.int64)], 1) if first.has_bias else xt
        hit = off + int(state["left"].abs().sum(0).nonzero()[0])     # (w is 0 on an input coordinate X never uses)

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

    def fold_capture(fold):             # a row victim's check folds the columns with its chi
        def patched(chis):
            if matrix.layout == "row":
                state["left"] = chis[victim]
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
        o[off:off + rows, 0] = (o[off:off + rows, 0] + _kernel_shift(state["left"])) % P
        return o, pr

    def misplaced(o, pr):               # the victim's first two opened columns swapped
        o = o.clone()
        o[off:off + rows, [0, 1]] = o[off:off + rows, [1, 0]]
        return o, pr

    def misplaced_member(o, pr):        # the victim's rows sent first (last, if they are first)
        rest = torch.cat([o[:off], o[off + rows:]])
        return torch.cat([rest, o[off:off + rows]] if off == 0 else [o[off:off + rows], rest]), pr

    def wrong_column(open_):            # the victim's first column: the honest one at another index
        def patched(cols):
            out = open_(cols)
            other = open_({g: (idx + 1) % coms.groups[g].n_points for g, idx in cols.items()})
            o = out[key][0].clone()
            o[off:off + rows, 0] = other[key][0][off:off + rows, 0]
            out[key] = (o, out[key][1])
            return out
        return patched

    def plus_one(o, pr):
        o = o.clone()
        o[hit, 0] = (o[hit, 0] + 1) % P
        return o, pr

    def setting(value):
        def change(o, pr):
            o = o.clone()
            o[off, 0] = value
            return o, pr
        return change

    out = {"claim+1": (tamper_mid, None, None),
           "logit+1000": (tamper_final, None, None),
           "forged kernel column": (None, fold_capture, at_victim(kernel)),
           "misplaced columns": (None, None, at_victim(misplaced)),
           "wrong column": (None, None, wrong_column),
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
           "u+1": (None, lambda fold: lambda chis: {k: (u + (k == u_victim)) % P for k, u in fold(chis).items()},
                   None)}
    if len(members) > 1:
        out["misplaced member"] = (None, None, at_victim(misplaced_member))
    if final.has_bias and final.layout == "linear" and coms.plan.matrix_of(final.name).layout == "row":
        out["forged fold"] = (tamper_final, fold_forged, None)          # (a CNN's classifier: one claim column)
    return out


EXPECTED = {"claim+1": "freivalds", "logit+1000": "freivalds", "u+1": "freivalds", "forged fold": "columns_code",
            "forged kernel column": "columns_merkle", "misplaced columns": "columns_code",
            "wrong column": "columns_code", "misplaced member": "columns_code",
            "opened+1": "columns_code", "opened=P": "columns_shape", "opened=-1": "columns_shape",
            "opened>int32": "columns_shape",
            "opened narrow": "columns_shape", "opened short": "columns_shape", "opened int32": "columns_shape",
            "path forged": "columns_merkle", "path short": "columns_merkle", "path long": "columns_merkle"}


def _expected(graph, coms) -> dict:
    """:data:`EXPECTED` under ``coms``' plan: a wrong claim of a col-layout op is caught by the code check."""
    caught = {"row": "freivalds", "col": "columns_code"}
    mats = graph.mat_ops
    return dict(EXPECTED, **{attack: caught[coms.plan.matrix_of(op.name).layout]
                             for attack, op in (("claim+1", mats[len(mats) // 2]), ("logit+1000", mats[-1]))})


TENSOR_FORMS = ("opened narrow", "opened short", "opened int32")
"""Forgeries of the opened columns' tensor (another shape or dtype).  With ``wire`` the columns travel
as one run of 31-bit field elements instead, which carries neither a dtype nor the trees' shapes:
:func:`test_malformed_wire_messages_are_rejected_under_a_plan` forges that message itself.  (An entry
outside ``[0, 2**31)``, which the run cannot carry either, travels as no bytes: ``columns_shape``.)"""


def _labels(graph, x, coms, verifiers, seed=0, wire=False):
    """``{attack: the check that rejects it}``; every verifier of ``verifiers`` must agree."""
    prover = proto.Prover(graph, commitments=coms)
    real = {"fold": prover.fold, "open": prover.open}
    out = {}
    for i, (name, (tamper, fold, open_)) in enumerate(_attacks(graph, coms, x).items()):
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


@pytest.mark.parametrize("kind", ["lenet5", "mlp_wide", *DECODERS, "opt-pruned", "qwen-pruned"])
@pytest.mark.parametrize("policy", POLICIES)
@pytest.mark.parametrize("fiat_shamir", [False, True])
@pytest.mark.parametrize("wire", [False, True])
def test_every_forgery_is_rejected_at_its_check(kind, policy, fiat_shamir, wire):
    graph, x, coms = _planned(kind, policy)
    labels = _labels(graph, x, coms, _verifiers(graph, coms, fiat_shamir), wire=wire)
    expected = {k: v for k, v in _expected(graph, coms).items() if not (wire and k in TENSOR_FORMS)}
    assert labels == {k: expected[k] for k in labels} and len(labels) >= len(expected) - 2


@pytest.mark.parametrize("paths", ["deferred", "int8", "deferred_int8"])
@pytest.mark.parametrize("wire", [False, True])
def test_the_gpu_forms_of_the_checks_give_the_same_verdicts(paths, wire, monkeypatch):
    if "deferred" in paths:
        monkeypatch.setattr(proto, "_defer", lambda device: True)
    if "int8" in paths:
        monkeypatch.setattr(proto, "int8_ok", lambda device: True)
    for kind, policy in (("lenet5", "cnn12"), ("llama", "R8"), ("mlp_wide", "cnn12c"), ("qwen", "R8c"),
                         ("gpt-pruned", "tightc")):
        graph, x, coms = _planned(kind, policy)
        labels = _labels(graph, x, coms, _verifiers(graph, coms, False), seed=10, wire=wire)
        assert labels == {k: _expected(graph, coms)[k] for k in labels}


@pytest.mark.parametrize("policy", ["R8", "R8c"])
@pytest.mark.parametrize("wire", [False, True])
def test_a_gpu_client_and_prover_give_the_cpu_verdicts(device, policy, wire):
    graph, x, coms = _planned("llama", policy)
    cpu = _labels(graph, x, coms, _verifiers(graph, coms, False), seed=20, wire=wire)
    assert _labels(graph, x, coms, _verifiers(graph, coms, False, device=device), seed=20, wire=wire) == cpu
    if device == "cuda":
        on_device = proto.Prover(graph, device="cuda", commitments=coms)
        for v in _verifiers(graph, coms, False):
            a, b = (proto.run_query(p, v, x, seed=1, wire=wire) for p in (proto.Prover(graph, commitments=coms),
                                                                            on_device))
            assert a["accepted"] and (b["accepted"], b["bytes"]) == (True, a["bytes"])


# -- in-process messages: each party holds its own copy --------------------------------------------------

class _Scribbler(proto.Prover):
    """An honest prover that, once it has used them, overwrites in place every tensor it was handed or
    handed over -- the query, its claims and the tables' multiproofs (in ``fold``), ``chi`` (in
    ``fold``), ``u`` and the column indices (in ``open``)."""

    def claims(self, x, **kw):
        out = super().claims(x, **kw)
        x.zero_()
        self.sent = out
        return out

    def open_tables(self):
        self.proofs = super().open_tables()
        return self.proofs

    def fold(self, chis):
        self.us = super().fold(chis)
        for t in [*chis.values(), *(z for z in self.sent.values() if torch.is_tensor(z))]:
            t.add_(1)
        for proof in getattr(self, "proofs", {}).values():
            proof.clear()
        return self.us

    def open(self, cols):
        out = super().open(cols)
        for t in [*cols.values(), *self.us.values()]:
            t.copy_(t.flip(-1))
        return out


@pytest.mark.parametrize("policy", ["paper", "R8c"])
@pytest.mark.parametrize("fiat_shamir", [False, True])
def test_a_party_cannot_change_what_it_handed_the_other(policy, fiat_shamir):
    """The query and the challenges the prover receives, and the claims, multiproofs and ``u`` the
    verifier reads after the prover has run again, are each party's own copies: an honest prover that
    then overwrites all of them in place is still accepted, by every verifier (batched and streaming,
    with and without wire; modes C, K and Kpre)."""
    graph, x, coms = _planned("gpt", policy)
    for v in _verifiers(graph, coms, fiat_shamir):
        for wire in (False, True):
            res = proto.run_query(_Scribbler(graph, commitments=coms), v, x.clone(), seed=None if fiat_shamir else 1,
                                  wire=wire)
            assert res["accepted"], (v.stream, wire, res["rejected_at"])
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    for mode in ("K", "Kpre"):
        for stream in (False, True):
            v = proto.Verifier(graph.public(), proto.params_for(40, len(graph.mat_ops), fiat_shamir=fiat_shamir),
                               mode, weights=weights, stream=stream)
            if mode == "Kpre":
                v.precompute(proto.Challenger(seed=3))
            assert proto.run_query(_Scribbler(graph), v, x.clone(), seed=None if fiat_shamir else 1)["accepted"]


@pytest.mark.parametrize("policy", ["paper", "R8c"])
def test_a_prover_cannot_change_its_messages_or_the_challenges_in_place(policy):
    """Forgeries that would pass if the parties shared tensors, each rejected by every verifier
    (interactive and Fiat--Shamir, with and without wire): a claim forged (re-propagated) and restored
    in place inside ``fold``, after derive read it (a col-layout op's under ``R8c``); a bias shift of a
    row op's claims with the ``u`` that passes Freivalds for it, put back to the honest ``u`` in place
    inside ``open``, after Freivalds read it; every ``chi`` zeroed in place inside ``fold``, with ``u =
    0``; and the query rewritten in place (another prompt, proved honestly)."""
    graph, x, coms = _planned("gpt", policy)
    mats = graph.mat_ops
    restored = mats[len(mats) // 2].name if coms.plan is None else next(
        m for m in coms.plan.matrices if m.layout == "col").members[0]
    shifted = next(op for op in reversed(mats) if op.has_bias and (coms.plan is None
                                                                  or coms.plan.matrix_of(op.name).layout == "row"))
    kept = {}

    def forge(op, z):
        if op.name == restored:
            z = z.clone()
            z[:, -1] += 40000
            kept["z"] = z
        elif op.name == shifted.name:
            z = z.clone()
            z[0] += 40000                                      # as if the bias of row 0 were larger
        return z

    class Restorer(proto.Prover):
        def fold(self, chis):
            kept["z"][:, -1] -= 40000
            return super().fold(chis)

    class Unshifter(proto.Prover):
        def fold(self, chis):
            us = super().fold(chis)
            kept["honest"] = us[shifted.name].clone()
            us[shifted.name][:, shifted.n_in] = (us[shifted.name][:, shifted.n_in]
                                                 + 40000 * chis[shifted.name][:, 0]) % P
            kept["u"] = us[shifted.name]
            return us

        def open(self, cols):
            kept["u"].copy_(kept["honest"])
            return super().open(cols)

    class Zeroer(proto.Prover):
        def fold(self, chis):
            for chi in chis.values():
                chi.zero_()
            return {k: torch.zeros_like(u) for k, u in super().fold(chis).items()}

    class Rewriter(proto.Prover):
        def claims(self, x, **kw):
            x.copy_((x + 1) % _TINY["gpt"].vocab)
            return super().claims(x, **kw)

    for fiat_shamir in (False, True):
        for v in _verifiers(graph, coms, fiat_shamir):
            for wire in (False, True):
                for prover, kw in ((Restorer, {"tamper": forge}), (Unshifter, {"tamper": forge}), (Zeroer, None),
                                   (Rewriter, None)):
                    res = proto.run_query(prover(graph, commitments=coms), v, x.clone(),
                                          seed=None if fiat_shamir else 2, wire=wire, forward_kwargs=kw)
                    assert not res["accepted"], (prover.__name__, v.stream, wire)


@pytest.mark.parametrize("paths", ["at_once", "deferred", "int8", "deferred_int8"])
def test_grouped_column_checks_give_the_reference_verdicts(paths, monkeypatch):
    """The batched check of group openings against ``reference.check_columns``, group by group
    (the verdicts and the exceptions), on single and double defects."""
    if "deferred" in paths:
        monkeypatch.setattr(proto, "_defer", lambda device: True)
    if "int8" in paths:
        monkeypatch.setattr(proto, "int8_ok", lambda device: True)
    for kind, policy in (("lenet5", "tight"), ("gpt", "R8"), ("mlp_wide", "tightc"), ("qwen", "R8c")):
        graph, x, coms = _planned(kind, policy)
        v = _verifiers(graph, coms, False)[0]
        prover = proto.Prover(graph, commitments=coms)
        chis, us = _operands(v, prover, graph, x, proto.Challenger(seed=6))
        ch = proto.Challenger(seed=7)
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
            want = ref.check_columns(v, chis, us, cols, opened)
            assert v.check_columns(chis, us, cols, opened) == want, var.keys()
            rows = wire_openings(opened)      # the int32 rows run_query(wire=True) takes as they are
            assert v.check_columns(chis, us, cols, rows, wire=True) == ref.check_columns(
                v, chis, us, cols, rows, wire=True) == want, var.keys()
        for m in (_folded(graph, coms)[-1].name, next((m.name for m in coms.plan.matrices if m.layout == "col"), None)):
            if m is not None:           # a wrong u, or a wrong z (a col matrix's codeword source)
                bad = dict(us, **{m: (us[m] + 1) % P})
                assert v.check_columns(chis, bad, cols, openings) == ref.check_columns(v, chis, bad, cols, openings) \
                    == "columns_code"
        for broken, exc in ((dict(openings, **{b: (ob, [0] + pb[1:])}), TypeError), (dict(openings, **{b: None}),
                                                                                     TypeError)):
            for impl in (v.check_columns, lambda *args, **kw: ref.check_columns(v, *args, **kw)):
                with pytest.raises(exc):
                    impl(chis, us, cols, broken)
                with pytest.raises(exc):
                    impl(chis, us, cols, wire_openings(broken), wire=True)


def _operands(v, prover, graph, x, ch):
    """The two sides of every committed matrix's code check on an honest query ``x`` (as run_query gives
    them to ``check_columns``), which must equal ``reference.column_operands``'."""
    claims = prover.claims(x)
    inputs = v.derive(x, claims)
    chis = v.fold_challenges(ch, x)
    us = prover.fold({op.name: chis[op.name] for op in v._row_ops()})
    assert v.check_products(claims, inputs, chis, us) and ref.check_products(v, claims, inputs, chis, us)
    lefts, sources = v.column_operands(claims, inputs, chis, us)
    for mine, theirs in zip((lefts, sources), ref.column_operands(v, claims, inputs, chis, us)):
        assert mine.keys() == theirs.keys() and all(torch.equal(mine[k], theirs[k]) for k in mine)
    return lefts, sources


@pytest.mark.parametrize("policy", ["R8", "R8c"])
def test_malformed_openings_fail_as_the_default_fails(policy):
    graph, x, coms = _planned("gpt", policy)
    v = _verifiers(graph, coms, False)[0]
    prover = proto.Prover(graph, commitments=coms)
    ch = proto.Challenger(seed=4)
    chis, us = _operands(v, prover, graph, x, ch)
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


@pytest.mark.parametrize("policy", ["R8", "R8c"])
def test_the_fiat_shamir_transcript_absorbs_the_plan(policy, monkeypatch):
    graph, x, coms = _planned("llama", policy)
    events = []                         # every absorbed message (bytes labels) and challenge key (str labels)
    absorb, key = proto.Challenger.absorb, proto.Challenger._key
    monkeypatch.setattr(proto.Challenger, "absorb", lambda self, label, blob: (
        events.append(label), absorb(self, label, blob)))
    monkeypatch.setattr(proto.Challenger, "_key", lambda self, label: (events.append(label), key(self, label))[1])
    v = _verifiers(graph, coms, True)[0]
    prover = proto.Prover(graph, commitments=coms)
    assert proto.run_query(prover, v, x)["accepted"]
    labels = [label for label in events if isinstance(label, bytes)]
    assert [label for label in labels if label.startswith(b"group/")] == [b"group/" + g.encode() for g in v.groups]
    stacked = [name for name, pub in v.publics.items() if pub.members]
    assert [label for label in labels if label.startswith(b"col/")] == [b"col/" + m.encode() for m in stacked]
    assert bool(stacked) == policy.endswith("c") == bool(v.tables)
    assert [label for label in labels if label.startswith(b"table/")] == [b"table/" + t.encode() for t in v.tables]
    # every challenge follows the message it must: chi and chi' (drawn when M > r) the claims and the lookup
    # tables' multiproofs (sent with them), the columns u

    def at(prefix):
        return [i for i, e in enumerate(events) if type(e) is type(prefix) and e.startswith(prefix)]

    folds, columns = at("fold/"), at("cols/")
    wide = [m for m in stacked if v.claim_columns(x)[graph.mat_ops.index(
        next(op for op in graph.mat_ops if op.name == v.publics[m].members[0]))] > v.params.reps]
    assert sorted(events[i] for i in folds) == sorted("fold/" + n for n in [op.name for op in v._row_ops()] + wide)
    assert max(at(b"claim/")) < min(folds) and max(folds) < min(at(b"u/")) <= max(at(b"u/")) < min(columns)
    assert len(columns) == len(v.groups)
    if v.tables:
        assert max(at(b"claim/")) < min(at(b"lookup/")) <= max(at(b"lookup/")) < min(folds)
        assert [events[i] for i in at(b"lookup/")] == [b"lookup/" + t.encode() for t in v.tables]
    # every public parameter of the plan changes the challenges: a group's root, its t, its member order
    # (with the same messages), each op's codeword length (absorbed in its op record) and a col matrix's
    # (in its col record), and the ops a col matrix stacks and their order

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
                   v.publics[name], n_points=2 * v.publics[name].n_points)})) if name in v.publics else
               variant(tables=dict(v.tables, **{name: dataclasses.replace(v.tables[name], root=bytes(32))}))]
    for m in stacked:
        pub = v.publics[m]
        changed.append(variant(publics=dict(v.publics, **{m: dataclasses.replace(pub, n_points=2 * pub.n_points)})))
        for members in ((pub.members[:1], pub.members[::-1]) if len(pub.members) > 1 else ()):
            changed.append(variant(publics=dict(v.publics, **{m: dataclasses.replace(pub, members=members)})))
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


@pytest.mark.parametrize("kind,policy", [("lenet5", "cnn12"), ("gpt", "R8"), ("llama", "tight"), ("gpt", "R8c"),
                                         ("qwen", "tightc"), ("mlp_wide", "cnn12c")])
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
    folded = _folded(graph, coms)
    u_shapes = [(params.reps, op.row_length) for op in folded]
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
    rows = {name: c.public.n_rows for name, c in coms.items()}
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
    targets = {"u": (params.reps * sum(op.row_length for op in folded[:len(folded) // 2]),
                     params.reps * sum(op.row_length for op in folded)),
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
    # values the encoding cannot carry (a claim outside the range check, an entry of u or of an opened
    # column outside [0, 2**31)) travel as no bytes: rejected where those values are without wire
    g0 = coms.plan.groups[0][0]
    victim = graph.mat_ops[len(graph.mat_ops) // 2].name

    def huge(op, z):
        if op.name == victim:
            z = z.clone()
            z[0, 0] = proto.Z_BOUND
        return z

    def outside_u(chis):
        us = real["fold"](chis)
        u = next(iter(us))
        return dict(us, **{u: torch.full_like(us[u], -1)})

    def outside_columns(cols):
        out = real["open"](cols)
        o = out[g0][0].clone()
        o[0, 0] = -1
        return dict(out, **{g0: (o, out[g0][1])})

    for i, (fold, open_, kw, want) in enumerate(((real["fold"], real["open"], {"tamper": huge}, "range_or_shape"),
                                                 (outside_u, real["open"], None, "freivalds"),
                                                 (real["fold"], outside_columns, None, "columns_shape"))):
        prover.fold, prover.open = fold, open_
        try:
            for wire in (True, False):
                got = {proto.run_query(prover, v, x, seed=30 + i, wire=wire, forward_kwargs=kw)["rejected_at"]
                       for v in verifiers}
                assert got == {want}, (want, wire, got)
        finally:
            prover.fold, prover.open = real["fold"], real["open"]


@pytest.mark.parametrize("policy", ["R8", "R8c"])
def test_the_wire_transcript_absorbs_the_plan_and_the_encoded_bytes(policy, monkeypatch):
    graph, x, coms = _planned("llama", policy)
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
    tables = list(verifiers[0].tables)
    statement = [b"params", *(b"op/" + op.name.encode() for op in graph.mat_ops),
                 *(b"col/" + m.name.encode() for m in coms.plan.matrices if m.layout == "col"),
                 *(b"group/" + g.encode() for g in verifiers[0].groups), *(b"table/" + t.encode() for t in tables),
                 b"x"]
    assert labels == [*statement, b"claims/PVC3", *([b"rows/I8"] if tables else []),
                      *(b"lookup/" + t.encode() for t in tables), b"u/F31"]
    blobs = dict(transcripts[0])
    claims = prover.claims(x)
    assert blobs[b"claims/PVC3"] == cc.encode([claims[op.name] for op in graph.mat_ops if op.name not in tables])
    assert not tables or blobs[b"rows/I8"] == cc.pack_rows([claims[t] for t in tables])
    reps = verifiers[0].params.reps
    assert len(blobs[b"u/F31"]) == cc.field_size(reps * sum(op.row_length for op in _folded(graph, coms)))
    # the statement -- the plan's trees, their members and t, its tables -- is absorbed as without wire,
    # and so are the tables' multiproofs
    absorbed.clear()
    assert proto.run_query(prover, verifiers[0], x)["accepted"]
    assert absorbed[:len(statement)] == transcripts[0][:len(statement)]
    assert [a for a in absorbed if a[0].startswith(b"lookup/")] == [a for a in transcripts[0]
                                                                    if a[0].startswith(b"lookup/")]


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


def test_a_verifier_refuses_a_col_key_that_does_not_fit():
    """A col matrix must stack linear ops that read one tensor, with its shape, and every op must be
    checked by exactly one matrix."""
    graph, x, coms = _planned("gpt", "R8c")
    params = proto.params_for(40, len(graph.mat_ops), plan=coms.plan)
    groups, tables = coms.group_publics, coms.table_publics
    proto.Verifier(graph.public(), params, "C", publics=coms.publics, groups=groups, tables=tables)
    fused = next(m for m in coms.plan.matrices if len(m.members) > 1)       # q, k and v
    pub = coms.publics[fused.name]
    ops = {op.name: op for op in graph.mat_ops}
    embed = next(op.name for op in graph.mat_ops if op.layout == "embed")
    elsewhere = next(op.name for op in graph.mat_ops if op.layout == "linear" and op.row_length == pub.n_rows
                     and op.name not in fused.members)                        # e.g. an fc1: another input

    def refused(match, **changes):
        publics = dict(coms.publics, **changes)
        with pytest.raises(ValueError, match=match):
            proto.Verifier(graph.public(), params, "C", publics=publics, groups=groups, tables=tables)

    one = dataclasses.replace(pub, members=pub.members[:1])                    # k and v checked by nothing
    refused("shape of its ops", **{fused.name: one})
    refused("exactly one committed matrix", **{fused.name: dataclasses.replace(
        one, row_length=ops[pub.members[0]].n_rows)})
    refused("reading one tensor", **{fused.name: dataclasses.replace(pub, members=pub.members[:2] + (elsewhere,))})
    refused("linear ops", **{fused.name: dataclasses.replace(pub, members=pub.members + (embed,))})
    refused("no weight op", **{fused.name: dataclasses.replace(pub, members=pub.members[:2] + ("nothing",))})
    first = ops[pub.members[0]]                                                 # q, also on a row matrix
    refused("exactly one committed matrix", **{first.name: dataclasses.replace(
        pub, members=(), n_rows=first.n_rows, row_length=first.row_length)})


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


def test_bench_names_a_policys_cells_once_and_only_under_it(monkeypatch):
    """``--policy`` adds the ``_pol<name>`` suffix: a ``--tag`` with one would record other cells
    under a policy's names, and a second spelling of a policy (``cnn016``) a second name."""
    import importlib.util
    import sys
    from pathlib import Path

    path = Path(__file__).resolve().parents[1] / "experiments" / "4_defence_benchmark" / "bench.py"
    spec = importlib.util.spec_from_file_location("bench_policy_test", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    for argv, needs in ((["--tag", "_polcnn16"], "_pol"), (["--policy", "paper", "--tag", "_polpaper"], "_pol"),
                        (["--policy", "cnn16", "--tag", "_thr1_polcnn16"], "_pol"),
                        (["--policy", "cnn016"], "--policy"), (["--policy", "R016"], "--policy")):
        monkeypatch.setattr(sys, "argv", ["bench.py", "cnn", "--model", "lenet5", *argv])
        with pytest.raises(SystemExit, match=needs):
            bench.main()


def _improvements_script(name: str):
    """``experiments/6_improvements/<name>.py`` as a module (its directory on the path, as when run)."""
    import importlib.util
    import sys
    from pathlib import Path

    folder = Path(__file__).resolve().parents[1] / "experiments" / "6_improvements"
    if str(folder) not in sys.path:
        sys.path.insert(0, str(folder))
    spec = importlib.util.spec_from_file_location(name, folder / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def test_plan_bytes_runs_distinct_cnn_queries(monkeypatch):
    """``--run <CNN> --queries N``: N distinct inputs, so the Fiat--Shamir rows are medians of N
    transcripts (one input repeated gave N copies of one)."""
    import itertools

    pb = _improvements_script("plan_bytes")
    seen = []
    monkeypatch.setattr(pb, "_build_rows", lambda graph, cases, *a, **kw: seen.extend(cases) or [])
    pb.run_rows("lenet5", 40, ["cnn16"], 3)
    ((_, xs, cols),) = seen
    assert len(xs) == 3 and not any(torch.equal(a, b) for a, b in itertools.combinations(xs, 2))
    graph, x, _ = pb.cnn_graph("lenet5")
    assert all(q.shape == x.shape and q.dtype == x.dtype for q in xs)
    assert cols == {k: z.shape[1] for k, z in graph.forward(xs[1])[1].items()}


def test_perf_random_init_builds_every_cnn_it_accepts(monkeypatch, capsys):
    """``perf.py --random-init`` takes its CNNs from ``ab_verifier.CNNS``, which ``build`` makes:
    ResNet-18-224 too (it fell through to the decoder branch: ``KeyError``)."""
    import json
    import sys
    from pathlib import Path

    perf = _improvements_script("perf")
    monkeypatch.chdir(Path(__file__).resolve().parents[1])
    monkeypatch.setattr(sys, "argv", ["perf.py", "--model", "resnet18_224", "--random-init", "--modes", "Kpre",
                                      "--queries", "1", "--device", "cpu", "--threads", str(torch.get_num_threads())])
    perf.main()
    out = json.loads(capsys.readouterr().out.strip().splitlines()[-1])
    assert (out["model"], out["accepted"], out["queries"]) == ("resnet18_224", 1, 1)
    assert "resnet18_224" in _improvements_script("ab_verifier").CNNS


# -- the proof-size model ----------------------------------------------------------------------------

@pytest.mark.parametrize("kind", ["lenet5", "mlp_wide", "gpt", "opt", "qwen", "gpt-pruned", "qwen-pruned"])
@pytest.mark.parametrize("policy", ("paper",) + POLICIES)
def test_the_proof_size_model_is_what_run_query_sends(kind, policy):
    graph, x = _model(kind) if policy == "paper" else _planned(kind, policy)[:2]
    coms = proto.commit_graph(graph, 4) if policy == "paper" else _planned(kind, policy)[2]
    prover = proto.Prover(graph, commitments=coms)
    _, claims = graph.forward(x)
    cols = {k: z.shape[1] for k, z in claims.items()}
    base, _, variant = kind.partition("-")
    if base in _TINY:
        assert list(cols.values()) == list(analytic.decoder_claim_columns(
            _TINY[base], x.shape[1], prune_last=variant == "pruned").values())
    ids = _looked_up(graph, x, coms.tables)
    for fs in (False, True):
        params = proto.params_for(40, len(graph.mat_ops), fiat_shamir=fs, plan=coms.plan)
        v = proto.Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                           tables=coms.table_publics)
        want = analytic.proof_bytes(graph.mat_ops, params, cols, plan=coms.plan, table_ids=ids)
        encoded = _claim_bytes(graph, coms.tables, claims, wire=True)
        wired = analytic.proof_bytes(graph.mat_ops, params, cols, plan=coms.plan, wire_claims=encoded,
                                     table_ids=ids)
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
        # modes K and Kpre send the claims only, all as integers (a plan's tables are mode C's), and with
        # lookups none of the embedding ops'
        for lookups in (False, True):
            own = {op.name for op in graph.mat_ops if lookups and op.layout == "embed"}
            sent = {k: z for k, z in claims.items() if k not in own}
            k_modes = analytic.proof_bytes(graph.mat_ops, params, cols, plan=coms.plan, mode="Kpre", lookups=lookups)
            assert k_modes == {"claims": _claim_bytes(graph, (), sent, False), "u": 0, "columns": 0, "paths": 0.0}
            k_wired = analytic.proof_bytes(graph.mat_ops, params, cols, plan=coms.plan, mode="Kpre",
                                           wire_claims=_claim_bytes(graph, (), sent, True), lookups=lookups)
            assert k_wired == {"claims": _claim_bytes(graph, (), sent, True), "u": 0, "columns": 0, "paths": 0.0}
    with pytest.raises(ValueError, match="lookups"):
        analytic.proof_bytes(graph.mat_ops, params, cols, plan=coms.plan, lookups=True)
    setup = analytic.setup_size(graph.mat_ops, plan=coms.plan)
    assert setup["encoded_entries"] == sum(c.public.n_rows * c.n_points for c in coms.values())
    assert setup["trees"] == (len(coms.groups) + len(coms.tables) or len(coms))
    assert setup["leaves"] == (sum(g.n_points for g in coms.groups.values()) + sum(
        next_pow2(t.n_tokens) for t in coms.tables.values()) or sum(c.n_points for c in coms.values()))
    assert setup["max_n"] <= MAX_N

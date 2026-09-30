"""The col layout and fusion of the commitment plans (``pvi.fullcheck.plans``, the policies with the
suffix ``c``): a col-layout matrix commits ``[A_1 ; A_2 ; ...]^T`` for linear ops reading one
tensor; the verifier draws ``chi'`` over the claims' columns (the identity when ``M <= r``) with
``chi``, computes ``w = chi' [X ; 1]^T`` and ``z = chi' Z^T`` itself and checks the opened columns
``E'[:, c]`` against ``Enc(z)[c]``.  The protocol-level checks of every plan (honest queries and
their bytes, every forgery at its check, the GPU forms, the wire encoding, the transcript) run in
``tests/test_plans.py`` with the ``c`` policies among its ``POLICIES``; this file holds what is
particular to the layouts."""

from __future__ import annotations

import pytest
import torch

from pvi.fullcheck import analytic
from pvi.fullcheck import field as fld
from pvi.fullcheck import protocol as proto
from pvi.fullcheck import reference as ref
from pvi.fullcheck.commitment import (GroupCommitment, MerkleTree, TransposedCommitment, WeightCommitment,
                                      column_leaf, group_leaf, transposed_leaves, verify_multiproof)
from pvi.fullcheck.pipeline import wire_claim, wire_openings
from pvi.fullcheck.plans import REFERENCE_LAMBDA, col_name, next_pow2, plan_commitment
from pvi.fullcheck.transformer import CONFIGS, build_decoder

from test_plans import _TINY, _cnn, _folded, _matrix, _planned, _verifiers   # the plan tests' models, helpers

P = fld.P


# -- the commitment layer -------------------------------------------------------------------------------

def _transposed(mats):
    """``[A_1 ; A_2 ; ...]^T`` (``A_i = [W_i | b_i]``) as int64 ``[K (+1), sum N_i]``."""
    a = torch.cat([torch.cat([w.to(torch.int64)] + ([b[:, None]] if b is not None else []), 1) for w, b in mats])
    return a.T.contiguous()


@pytest.mark.parametrize("bias", [True, False])
def test_a_col_matrix_commits_the_transposed_stack(bias):
    mats = {"q": _matrix(1, 24, 40, bias), "k": _matrix(2, 8, 40, bias), "v": _matrix(3, 8, 40, bias)}
    n = 128                                           # >= 2 next_pow2(40 rows of the stack)
    at = _transposed(mats.values())                   # [K (+1), 40]
    enc = fld.rs_encode(at, n)
    want = [column_leaf(b"t", c, enc[:, c].to(torch.int32).numpy()) for c in range(n)]
    pairs = list(mats.values())
    assert transposed_leaves(b"t", pairs, n) == want
    assert transposed_leaves(b"t", pairs, n, max_host_bytes=0, row_chunk=3) == want        # one row block at a time
    assert transposed_leaves(b"t", pairs, n, max_host_bytes=4 * n * 7, row_chunk=3) == want  # blocks of 3 chunks
    com = TransposedCommitment.member(b"t", mats, n)
    pub = com.public
    assert (pub.n_rows, pub.row_length, pub.n_points, pub.members, pub.root) == (41 if bias else 40, 40, n,
                                                                                  ("q", "k", "v"), b"")
    assert com.leaves(max_host_bytes=0) == want
    idx = torch.tensor([0, 5, 77, 127])
    v = fld.power_table(fld.root_of_unity(n), n)[(torch.arange(40)[:, None] * idx[None, :]) % n]
    assert torch.equal(com.columns_at(v), enc[:, idx])                                    # recomputed from W
    # w . E'[:, c] == Enc(z)[c] for z = A w, the identity the code check tests
    w = torch.randint(0, P, (at.shape[0],), generator=torch.Generator().manual_seed(4))
    z = ref.field_matmul_mod(w[None, :], fld.to_field(at))                              # [1, sum N] = (A w)^T
    assert torch.equal(ref.field_matmul_mod(w[None, :], enc[:, idx]), ref.codeword_at(z, n, idx))
    with pytest.raises(ValueError, match="one row length"):
        TransposedCommitment.member(b"t", {"a": _matrix(1, 4, 40, True), "b": _matrix(2, 4, 40, False)}, n)


def test_a_group_binds_row_and_col_matrices_alike():
    n = 256
    row = {"fc2": _matrix(5, 12, 70)}
    col = {"q": _matrix(6, 20, 30), "k": _matrix(7, 4, 30)}
    members = {"fc2": WeightCommitment.member(b"fc2", *row["fc2"], n),
               "q+k^T": TransposedCommitment.member(b"q+k^T", col, n)}
    group = GroupCommitment.build(b"g", members, max_host_bytes=64)
    own = {"fc2": members["fc2"].leaves(), "q+k^T": members["q+k^T"].leaves()}
    assert group.tree.root == MerkleTree([group_leaf(b"g", c, [own["fc2"][c], own["q+k^T"][c]])
                                          for c in range(n)]).root
    idx = torch.tensor([3, 64, 200])
    cols, proof = group.open(idx)
    enc_row = fld.rs_encode(torch.cat([row["fc2"][0].to(torch.int64), row["fc2"][1][:, None]], 1), n)[:, idx]
    enc_col = fld.rs_encode(_transposed(col.values()), n)[:, idx]
    assert torch.equal(cols, torch.cat([enc_row, enc_col]))                     # [12 + 31, t]
    leaves = {c: group_leaf(b"g", c, [column_leaf(b"fc2", c, cols[:12, j].numpy()),
                                      column_leaf(b"q+k^T", c, cols[12:, j].numpy())])
              for j, c in enumerate(idx.tolist())}
    assert verify_multiproof(group.tree.root, group.public.depth, leaves, proof)
    with pytest.raises(ValueError, match="one codeword length"):
        GroupCommitment.build(b"g", dict(members, other=WeightCommitment.member(b"o", *_matrix(8, 3, 9), 2 * n)))


# -- the planner ----------------------------------------------------------------------------------------

def test_col_policy_names():
    ops = analytic.decoder_shapes(CONFIGS["gpt2"], 1)
    for good in ("tightc", "cnn16c", "R16c", "R64c"):
        plan = plan_commitment(ops, good)
        assert plan.policy == good and any(m.layout == "col" for m in plan.matrices)
    for bad in ("c", "paperc", "tightcc", "cnn16C", "cnnc", "Rc", "R16cc", "R3c", "cnn0c", "R016c", "tight c"):
        with pytest.raises(ValueError):
            plan_commitment(ops, bad)


def _length(policy, m, embed=False):
    """The codeword length of a message of length ``m`` under ``policy`` (the policies' definitions)."""
    if policy.startswith("cnn"):
        return max(1 << int(policy[3:].rstrip("c")), 2 * next_pow2(m))
    rate = 4 if policy.startswith("tight") or embed else int(policy[1:].rstrip("c"))
    return rate * next_pow2(m)


def _opened_bytes(ops, plan) -> float:
    """What a plan's layouts and groups were chosen by: ``u``, the opened columns and the expected
    multiproofs at ``REFERENCE_LAMBDA``, interactive (the claims do not depend on the plan)."""
    b = analytic.proof_bytes(ops, proto.params_for(REFERENCE_LAMBDA, len(ops), plan=plan), 1, plan=plan)
    return b["u"] + b["columns"] + b["paths"]


@pytest.mark.parametrize("model", ["gpt2", "opt-350m", "llama2-7b", "qwen3-4b"])
@pytest.mark.parametrize("policy", ["tightc", "cnn18c", "R16c", "R64c"])
def test_a_col_policy_fuses_the_ops_of_one_input_and_opens_less(model, policy):
    """Every set of linear ops reading one tensor with one row length is committed all in rows, each
    op transposed or all in one col matrix (q/k/v, gate/up: one matrix, the head transposed), every
    other op in rows; each matrix at the policy's codeword length of its message length; and the
    plan opens fewer bytes than its base policy's."""
    cfg = CONFIGS[model]
    ops = analytic.decoder_shapes(cfg)
    plan = plan_commitment(ops, policy)
    for m in plan.matrices:
        assert m.n_points == _length(policy, m.row_length, m.layout == "row" and m.name in ("embed", "pos"))
    sets: dict = {}
    for op in ops:
        sets.setdefault((op.input, op.row_length) if op.layout == "linear" else op.name, []).append(op)
    for members in sets.values():
        names = [op.name for op in members]
        chosen = [m for m in plan.matrices if set(m.ops) & set(names)]
        assert sorted(o for m in chosen for o in m.ops) == sorted(names)
        layouts = [m.layout for m in chosen]
        assert layouts == ["row"] * len(names) or layouts == ["col"] * len(chosen) and len(chosen) in (1, len(names))
        assert members[0].layout == "linear" or layouts == ["row"]
    for i in range(cfg.n_layers):
        for names in ((f"q{i}", f"k{i}", f"v{i}"),) + (((f"gate{i}", f"up{i}"),) if cfg.mlp == "swiglu" else ()):
            m = plan.matrix(col_name(names))
            assert (m.members, m.n_rows) == (names, ops[[op.name for op in ops].index(names[0])].row_length)
    assert plan.matrix_of("head").layout == "col" and plan.matrix_of("embed").layout == "row"
    assert _opened_bytes(ops, plan) < 0.6 * _opened_bytes(ops, plan_commitment(ops, policy[:-1]))


@pytest.mark.parametrize("model", ["mlp_mnist", "lenet5", "vgg16", "resnet18_cifar", "llama2-7b", "opt-1.3b"])
def test_a_col_policy_never_opens_more_than_its_base_policy(model):
    """The layouts are chosen for the whole plan, shared trees included: a col matrix that saves its
    ``u`` but ends up on a tree of its own may cost more than it saves (LeNet-5's classifier under
    ``cnn16c``), and is then not taken."""
    ops = _cnn(model)[0].mat_ops if model not in CONFIGS else analytic.decoder_shapes(CONFIGS[model])
    for policy in ("tight", "cnn16", "cnn17", "cnn18", "R8", "R16", "R64"):
        base, col = plan_commitment(ops, policy), plan_commitment(ops, policy + "c")
        assert _opened_bytes(ops, col) <= _opened_bytes(ops, base) + 1e-6, (model, policy)
    assert all(m.layout == "row" for m in plan_commitment(_cnn("lenet5")[0].mat_ops, "cnn16c").matrices)


def test_the_byte_and_setup_models_of_a_col_plan():
    ops = analytic.decoder_shapes(CONFIGS["gpt2"], 2)
    plan = plan_commitment(ops, "R16c")
    params = proto.params_for(128, len(ops), plan=plan)
    got = analytic.proof_bytes(ops, params, 64, plan=plan)
    folded = [op for op in ops if plan.matrix_of(op.name).layout == "row"]
    assert {op.name[:3] for op in folded} <= {"emb", "pos", "fc2", "o0", "o1"}        # u only for the row ops
    assert got["u"] == 4 * params.reps * sum(op.row_length for op in folded)
    rows = {m.name: m.n_rows for m in plan.matrices}                                   # a col matrix opens k rows
    assert got["columns"] == 4 * sum(params.columns_for(g) * sum(rows[m] for m in ms) for g, ms in plan.groups)
    assert analytic.setup_size(ops, plan=plan)["encoded_entries"] == sum(m.n_rows * m.n_points for m in plan.matrices)


# -- the verifier's challenges and checks ---------------------------------------------------------------

def test_chi_prime_is_the_identity_on_at_most_r_claim_columns():
    graph, x, coms = _planned("gpt", "R8c")
    v = _verifiers(graph, coms, False)[0]
    chis = v.fold_challenges(proto.Challenger(seed=1), x)
    stacked = {m.name: m for m in coms.plan.matrices if m.layout == "col"}
    head = coms.plan.matrix_of(graph.mat_ops[-1].name).name                 # the LM head: M = 1
    assert torch.equal(chis[head], torch.eye(1, dtype=torch.int64))
    for name, m in stacked.items():
        if name != head:
            assert chis[name].shape == (v.params.reps, x.shape[1])             # M = 9 prompt tokens > r
    one = x[:, :1]                                    # a one-token prompt: every claim has one column
    chis = v.fold_challenges(proto.Challenger(seed=1), one)
    assert all(torch.equal(chis[name], torch.eye(1, dtype=torch.int64)) for name in stacked)
    prover = proto.Prover(graph, commitments=coms)
    for w in _verifiers(graph, coms, False):
        assert proto.run_query(prover, w, one, seed=2)["accepted"]
        victim = next(iter(stacked.values())).members[0]
        bad = proto.run_query(prover, w, one, seed=2, forward_kwargs={
            "tamper": lambda op, z: z + (op.name == victim)})
        assert bad["rejected_at"] == "columns_code"


@pytest.mark.parametrize("paths", ["at_once", "deferred", "int8", "deferred_int8"])
def test_the_col_operands_are_the_references(paths, monkeypatch):
    """``w`` and ``z`` of every col matrix, batched (and on the GPU forms), equal the straightforward
    products, on honest claims and on claims derive() did not range-check (reduced first)."""
    if "deferred" in paths:
        monkeypatch.setattr(proto, "_defer", lambda device: True)
    if "int8" in paths:
        monkeypatch.setattr(proto, "int8_ok", lambda device: True)
    for kind, policy in (("gpt", "R8c"), ("qwen", "tightc"), ("mlp_wide", "cnn12c"), ("opt", "R8c")):
        graph, x, coms = _planned(kind, policy)
        v = _verifiers(graph, coms, False)[0]
        prover = proto.Prover(graph, commitments=coms)
        claims = prover.claims(x)
        inputs = v.derive(x, claims)
        chis = v.fold_challenges(proto.Challenger(seed=3), x)
        us = prover.fold({op.name: chis[op.name] for op in v._row_ops()})
        for cl in (claims, {k: z.clone() for k, z in claims.items()}):            # (copies: not range-checked)
            got, want = v.column_operands(cl, inputs, chis, us), ref.column_operands(v, cl, inputs, chis, us)
            for mine, theirs in zip(got, want):
                assert mine.keys() == theirs.keys() and all(torch.equal(mine[k], theirs[k]) for k in mine)
        assert v.column_operands(claims, inputs, chis, us)[0].keys() == set(chis)


@pytest.mark.parametrize("kind,policy", [("llama", "R8c"), ("mlp_wide", "tightc")])
def test_a_streaming_verifier_rejects_a_query_its_claims_do_not_fit(kind, policy):
    graph, x, coms = _planned(kind, policy)
    prover = proto.Prover(graph, commitments=coms)
    _, stream = _verifiers(graph, coms, False)
    chis = stream.fold_challenges(proto.Challenger(seed=1), x)
    us = prover.fold({op.name: chis[op.name] for op in stream._row_ops()})
    cols = stream.column_challenges(proto.Challenger(seed=2))
    openings = wire_openings(prover.open(cols))
    claims = {k: wire_claim(z) for k, z in prover.claims(x).items()}
    assert stream.verify_streaming(x, claims, chis, us, cols, openings) is None
    # another query: chi' of its claim columns (none for a malformed one), and derive rejects these claims
    other = x[:, :5] if kind in _TINY else x[:, :700]
    folded = {op.name for op in stream._row_ops()}
    chis = stream.fold_challenges(proto.Challenger(seed=1), other)
    assert chis.keys() == (folded if kind == "mlp_wide" else folded | {m.name for m in coms.plan.matrices
                                                                       if m.layout == "col"})
    assert stream.verify_streaming(other, claims, chis, us, cols, openings) == "range_or_shape"


@pytest.mark.parametrize("kind", ["gpt", "qwen"])
@pytest.mark.parametrize("mode", ["C", "K", "Kpre"])
def test_a_lean_prover_and_a_lean_verifier_under_a_col_plan(kind, mode):
    graph, x, coms = _planned(kind, "R8c")
    params = proto.params_for(40, len(graph.mat_ops), plan=coms.plan)
    prover = proto.Prover(graph, commitments=coms, lean=True)
    if mode == "C":
        v = proto.Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics, lean=True)
    else:
        v = proto.Verifier(graph.public(), params, mode, weights={o.name: (o.weight, o.bias) for o in graph.mat_ops},
                           lean=True)
        if mode == "Kpre":
            v.precompute(proto.Challenger(seed=4))
    assert proto.run_query(prover, v, x, seed=1)["accepted"]
    victim = next(m for m in coms.plan.matrices if m.layout == "col").members[-1]
    bad = proto.run_query(prover, v, x, seed=2, forward_kwargs={"tamper": lambda op, z: z + (op.name == victim)})
    assert bad["rejected_at"] == ("columns_code" if mode == "C" else "freivalds")


def test_a_col_plan_opens_fewer_bytes_than_its_row_policy_on_a_decoder():
    for kind in ("gpt", "llama", "opt", "qwen"):
        graph, x = _planned(kind, "R8")[:2]
        sizes = {}
        for policy in ("R8", "R8c"):
            g, _, coms = _planned(kind, policy)
            v = _verifiers(g, coms, False)[0]
            res = proto.run_query(proto.Prover(g, commitments=coms), v, x, seed=1)
            assert res["accepted"]
            sizes[policy] = res["bytes"]["u"] + res["bytes"]["columns"]
            assert policy == "R8" or res["bytes"]["u"] == 4 * v.params.reps * sum(
                op.row_length for op in _folded(g, coms))
        assert sizes["R8c"] < sizes["R8"], (kind, sizes)


def test_grouped_query_attention_fuses_q_with_its_narrower_k_and_v():
    cfg = _TINY["qwen"]
    assert cfg.n_kv_heads < cfg.n_heads and cfg.qk_norm
    graph = build_decoder(cfg, calib_tokens=8, seed=1)
    rows = {op.name: op.n_rows for op in graph.mat_ops}
    fused = [m for m in plan_commitment(graph.mat_ops, "R8c").matrices if len(m.members) == 3]
    assert len(fused) == cfg.n_layers
    for m in fused:
        q, k, v = (rows[o] for o in m.members)
        assert q == cfg.n_heads * cfg.head_dim > k == v == cfg.n_kv_heads * cfg.head_dim
        assert (m.n_rows, m.row_length) == (cfg.d_model, q + k + v)

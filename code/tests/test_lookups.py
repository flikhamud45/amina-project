"""Embedding lookups and last-block pruning.

* Lookup tables (the policies with the suffix ``c``, mode C): an embedding op's table ``W`` ``[d, V]``
  is committed as a Merkle tree over its ``V`` rows; its claims are the looked-up rows, sent as int8
  with one multiproof over the distinct ids, and checked with no challenge -- int8 and ids of the
  table (derive: ``range_or_shape``), equal rows for equal ids (``lookup_consistency``), the
  multiproof (``lookup_merkle``).
* In modes K and Kpre a verifier with ``lookups=True`` reads the embedding rows from its own weights,
  and the prover sends no claims for them.
The protocol tests of every plan (honest queries and their bytes, the forgeries of the columns, the
GPU forms, the wire encoding, the transcript, the byte model) run in ``tests/test_plans.py``, with
lookup tables under its ``c`` policies and on pruned decoders too; this file holds what is particular
to the lookups (and the experiment scripts' options for them and for pruning).
"""

from __future__ import annotations

import copy
import dataclasses
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

from pvi.fullcheck import analytic
from pvi.fullcheck import claimcodec as cc
from pvi.fullcheck import protocol as proto
from pvi.fullcheck import reference as ref
from pvi.fullcheck.commitment import MerkleTree, TableCommitment, multiproof_size, table_leaves, verify_multiproof
from pvi.fullcheck.plans import next_pow2, plan_commitment
from pvi.fullcheck.transformer import CONFIGS

from test_plans import _TINY, _planned, _verifiers   # the plan tests' models and helpers

KINDS = ("gpt", "opt", "llama", "qwen")


def _query(kind, t=9):
    """A prompt in which one token comes three times (a table's repeated id)."""
    x = torch.randint(0, _TINY[kind.partition("-")[0]].vocab, (1, t), generator=torch.Generator().manual_seed(7))
    x[0, t - 3] = x[0, t - 1] = x[0, 1]
    return x


# -- the commitment ---------------------------------------------------------------------------------------

def test_a_lookup_table_is_a_merkle_tree_over_its_rows():
    w = torch.randint(-128, 128, (5, 11), generator=torch.Generator().manual_seed(1)).to(torch.int8)   # d 5, V 11
    com = TableCommitment.build(b"emb", w)
    rows = w.T.contiguous().numpy()
    leaves = [hashlib.sha256(b"pvi/row" + b"emb" + j.to_bytes(8, "big") + rows[j].tobytes()).digest()
              for j in range(11)]
    leaves += [hashlib.sha256(b"pvi/pad" + b"emb" + j.to_bytes(8, "big")).digest() for j in range(11, 16)]
    pub = com.public
    assert (pub.root, pub.n_rows, pub.n_tokens, pub.depth) == (MerkleTree(leaves).root, 5, 11, 4)
    ids = [3, 10, 3, 0, 3]
    proof = com.open(ids)
    assert len(proof) == multiproof_size([0, 3, 10], 4)
    assert verify_multiproof(pub.root, pub.depth, table_leaves(b"emb", [0, 3, 10], rows[[0, 3, 10]]), proof)
    # another row at an index, or a row moved to another index, does not verify
    other = rows[[0, 3, 10]].copy()
    other[1, 0] ^= 1
    assert not verify_multiproof(pub.root, pub.depth, table_leaves(b"emb", [0, 3, 10], other), proof)
    assert not verify_multiproof(pub.root, pub.depth, table_leaves(b"emb", [0, 3, 10], rows[[3, 0, 10]]), proof)
    assert table_leaves(b"other", [0], rows[:1]) != table_leaves(b"emb", [0], rows[:1])     # the tag is bound


# -- the plans ----------------------------------------------------------------------------------------------

@pytest.mark.parametrize("model", ["gpt2", "opt-350m", "llama2-7b", "qwen3-4b"])
@pytest.mark.parametrize("prune_last", [False, True])
def test_the_c_policies_look_up_every_embedding_table(model, prune_last):
    """Under a ``c`` policy every embedding table (the tokens', a learned position table) is a lookup
    table: a tree over its rows, in no group, with no column check; the LM head -- a table of its own
    even where the model ties it to the embedding -- keeps an encoded commitment; the row policies
    look nothing up."""
    cfg = CONFIGS[model]
    ops = analytic.decoder_shapes(cfg, prune_last=prune_last)
    embeds = [op for op in ops if op.layout == "embed"]
    assert [op.name for op in embeds] == (["embed", "pos"] if cfg.pos == "learned" else ["embed"])
    for policy in ("tightc", "cnn16c", "R64c"):
        plan = plan_commitment(ops, policy)
        assert [(m.name, m.layout, m.n_rows, m.row_length, m.n_points) for m in plan.tables] == [
            (op.name, "lookup", op.n_rows, op.row_length, next_pow2(op.row_length)) for op in embeds]
        assert not {m for _, ms in plan.groups for m in ms} & {op.name for op in embeds}
        assert len(plan.shapes()) == len(plan.coded) == len(plan.matrices) - len(embeds)
        assert plan.matrix_of("head").layout in ("row", "col")
        assert not plan_commitment(ops, policy[:-1]).tables


# -- the protocol ---------------------------------------------------------------------------------------------

def _lookup_attacks(graph, coms, x):
    """``{name: (tamper, open_tables patch, expected label)}``: forgeries of the looked-up rows and of
    the tables' multiproofs (the ops after a tampered table re-propagate honestly, so only the lookup
    check can catch it)."""
    tables = list(coms.tables)
    token = next(op.name for op in graph.mat_ops if op.name in coms.tables and op.inputs[0] == graph.input_name)
    ids = x.reshape(-1)
    repeated = (ids == ids[1]).nonzero().reshape(-1)                  # the positions of the repeated token

    def at(name, change):
        def tamper(op, z):
            return change(z.clone()) if op.name == name else z
        return tamper

    def shifted(z, cols):                     # entry 0 of those columns moved by one, staying int8
        z[0, cols] = torch.where(z[0, cols] > -128, z[0, cols] - 1, z[0, cols] + 1)
        return z

    def another_row(z):                       # the repeated token's columns: the row of the token at position 0
        z[:, repeated] = z[:, :1]
        return z

    def outside(z):
        z[0, 0] = 128
        return z

    def proofs(change):
        return lambda real: lambda: change(real())

    out = {"row forged": (at(token, lambda z: shifted(z, repeated)), None, "lookup_merkle"),
           "row of another token": (at(token, another_row), None, "lookup_merkle"),
           "repeat inconsistent": (at(token, lambda z: shifted(z, repeated[-1:])), None, "lookup_consistency"),
           "row outside int8": (at(token, outside), None, "range_or_shape"),
           "path forged": (None, proofs(lambda p: dict(p, **{token: [bytes(32)] + p[token][1:]})), "lookup_merkle"),
           "path short": (None, proofs(lambda p: dict(p, **{token: p[token][:-1]})), "lookup_merkle"),
           "path long": (None, proofs(lambda p: dict(p, **{token: p[token] + [bytes(32)]})), "lookup_merkle"),
           "path not bytes": (None, proofs(lambda p: dict(p, **{token: [0] + p[token][1:]})), "lookup_merkle"),
           "no proof": (None, proofs(lambda p: {k: v for k, v in p.items() if k != token}), "lookup_merkle")}
    if len(tables) > 1:                        # a learned position table: its row at position 2, and the proofs swapped
        pos = next(t for t in tables if t != token)
        out["position row forged"] = (at(pos, lambda z: shifted(z, [2])), None, "lookup_merkle")
        out["proofs swapped"] = (None, proofs(lambda p: dict(p, **{token: p[pos], pos: p[token]})), "lookup_merkle")
    return out


def _rejections(graph, coms, x, verifiers, wire, seed=0):
    """``{attack: (label, expected)}`` of every lookup forgery (every verifier of ``verifiers`` agreeing)."""
    prover = proto.Prover(graph, commitments=coms)
    real = prover.open_tables
    out = {}
    for i, (name, (tamper, patch, want)) in enumerate(_lookup_attacks(graph, coms, x).items()):
        if wire and name == "row outside int8":           # no int8 wire form: the prover cannot even send it
            with pytest.raises(ValueError, match="int8"):
                proto.run_query(prover, verifiers[0], x, seed=seed, wire=True, forward_kwargs={"tamper": tamper})
            continue
        prover.open_tables = patch(real) if patch else real
        try:
            got = {proto.run_query(prover, v, x, seed=seed + i, wire=wire,
                                   forward_kwargs={"tamper": tamper} if tamper else None)["rejected_at"]
                   for v in verifiers}
        finally:
            prover.open_tables = real
        assert len(got) == 1, (name, got)             # run_query and the streaming verifier agree
        out[name] = (got.pop(), want)
    return out


@pytest.mark.parametrize("kind", [*KINDS, "gpt-pruned", "qwen-pruned"])
@pytest.mark.parametrize("fiat_shamir", [False, True])
@pytest.mark.parametrize("wire", [False, True])
def test_the_lookup_tables_accept_honest_rows_and_reject_every_forgery(kind, fiat_shamir, wire):
    graph, _, coms = _planned(kind, "R8c")
    x = _query(kind)
    verifiers = _verifiers(graph, coms, fiat_shamir)
    prover = proto.Prover(graph, commitments=coms)
    for v in verifiers:                               # a repeated token: the claims repeat its row
        res = proto.run_query(prover, v, x, seed=None if fiat_shamir else 1, wire=wire)
        assert res["accepted"], res["rejected_at"]
    for name, (got, want) in _rejections(graph, coms, x, verifiers, wire).items():
        assert got == want, name


@pytest.mark.parametrize("paths", ["deferred", "int8", "deferred_int8"])
def test_the_gpu_forms_check_the_lookups_alike(paths, monkeypatch):
    if "deferred" in paths:
        monkeypatch.setattr(proto, "_defer", lambda device: True)
    if "int8" in paths:
        monkeypatch.setattr(proto, "int8_ok", lambda device: True)
    for kind in ("opt", "qwen-pruned"):
        graph, _, coms = _planned(kind, "R8c")
        x = _query(kind)
        verifiers = _verifiers(graph, coms, False)
        assert all(proto.run_query(proto.Prover(graph, commitments=coms), v, x, seed=1)["accepted"] for v in verifiers)
        for wire in (False, True):
            for name, (got, want) in _rejections(graph, coms, x, verifiers, wire, seed=5).items():
                assert got == want, (kind, wire, name)
        bad = x.clone()
        bad[0, 0] = -1                              # an id outside the table (the prover's gather wraps it)
        assert {proto.run_query(proto.Prover(graph, commitments=coms), v, bad, seed=2)["rejected_at"]
                for v in verifiers} == {"range_or_shape"}
        # modes K and Kpre with lookups: the verifier's own rows, gathered at clamped ids on a device
        weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
        embed = next(op.name for op in graph.mat_ops if op.layout == "embed")
        for mode in ("K", "Kpre"):
            for stream in (False, True):
                v = proto.Verifier(graph.public(), proto.params_for(40, len(graph.mat_ops)), mode, weights=weights,
                                   lookups=True, stream=stream)
                if mode == "Kpre":
                    v.precompute(proto.Challenger(seed=3))
                for wire in (False, True):
                    assert proto.run_query(proto.Prover(graph), v, x, seed=1, wire=wire)["accepted"]
                    assert proto.run_query(proto.Prover(graph), v, x, seed=1, wire=wire, forward_kwargs={
                        "tamper": lambda op, z: _bump(z) if op.name == embed else z})["rejected_at"] == "freivalds"
                    rejected = proto.run_query(proto.Prover(graph), v, bad, seed=1, wire=wire)["rejected_at"]
                    assert rejected == "range_or_shape"


def test_the_lookup_check_gives_the_reference_verdicts():
    """The vectorised check against ``reference.check_lookups``, on the rows and proofs of every forgery."""
    for kind in ("gpt", "llama"):
        graph, _, coms = _planned(kind, "R8c")
        x = _query(kind)
        v = _verifiers(graph, coms, False)[0]
        prover = proto.Prover(graph, commitments=coms)
        for name, (tamper, patch, _) in _lookup_attacks(graph, coms, x).items():
            claims = prover.claims(x, tamper=tamper) if tamper else prover.claims(x)
            proofs = (patch(prover.open_tables) if patch else prover.open_tables)()
            inputs = v.derive(x, claims)
            if inputs is None:                      # (out of int8: derive's range check rejects it)
                assert name == "row outside int8"
                continue
            assert v.check_lookups(claims, inputs, proofs) == ref.check_lookups(v, claims, inputs, proofs), name
        honest = prover.claims(x)
        assert v.check_lookups(honest, v.derive(x, honest), prover.open_tables()) is None


def test_an_id_outside_the_table_is_a_malformed_query():
    for kind in ("gpt", "llama"):
        graph, _, coms = _planned(kind, "R8c")
        weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
        params = proto.params_for(40, len(graph.mat_ops))
        x = _query(kind)
        x[0, 3] = -1
        verifiers = [*_verifiers(graph, coms, False)] + [proto.Verifier(graph.public(), params, "K", weights=weights,
                                                                     lookups=True, stream=s) for s in (False, True)]
        for v in verifiers:
            res = proto.run_query(proto.Prover(graph, commitments=coms), v, x, seed=1)
            assert res["rejected_at"] == "range_or_shape"
        claims = proto.Prover(graph).claims(_query(kind))
        x_long = _query(kind)
        x_long[0, 0] = _TINY[kind].vocab                                  # past the last row: no claim computed
        for v in verifiers[:1] + verifiers[2:3]:
            assert v.derive(x_long, claims) is None


@pytest.mark.parametrize("fiat_shamir", [False, True])
def test_malformed_lookup_rows_on_the_wire_are_rejected(fiat_shamir, monkeypatch):
    graph, _, coms = _planned("opt", "R8c")
    x = _query("opt")
    verifiers = _verifiers(graph, coms, fiat_shamir)
    prover = proto.Prover(graph, commitments=coms)
    pack = cc.pack_rows
    for i, change in enumerate((lambda b: b + b"\0", lambda b: b[:-1], lambda b: b"", lambda b: None)):
        monkeypatch.setattr(proto.claimcodec, "pack_rows", lambda zs, change=change: change(pack(zs)))
        got = {proto.run_query(prover, v, x, seed=i, wire=True)["rejected_at"] for v in verifiers}
        assert got == {"range_or_shape"}
        monkeypatch.undo()
    # one entry of a token's row changed at every position of the token, on the wire: rejected where the
    # same claim without wire is
    real = prover.claims
    same = (x.reshape(-1) == x[0, 0]).nonzero().reshape(-1)

    def one_off(z):
        z = z.clone()
        z[0, same] += 1 if z[0, 0] < 127 else -1
        return z

    token = next(iter(coms.tables))
    prover.claims = lambda *a, **kw: {k: one_off(z) if k == token else z for k, z in real(*a, **kw).items()}
    try:
        got = {proto.run_query(prover, v, x, seed=9, wire=wire)["rejected_at"] for v in verifiers for wire in (0, 1)}
    finally:
        prover.claims = real
    assert got == {"lookup_merkle"}


def test_the_transcript_absorbs_the_tables_and_their_multiproofs(monkeypatch):
    graph, _, coms = _planned("gpt", "R8c")
    x = _query("gpt")
    events = []
    absorb, key = proto.Challenger.absorb, proto.Challenger._key
    monkeypatch.setattr(proto.Challenger, "absorb", lambda self, label, blob: (
        events.append((label, bytes(blob))), absorb(self, label, blob)))
    monkeypatch.setattr(proto.Challenger, "_key", lambda self, label: (events.append(label), key(self, label))[1])
    v = _verifiers(graph, coms, True)[0]
    prover = proto.Prover(graph, commitments=coms)
    assert proto.run_query(prover, v, x)["accepted"]
    labels = [e[0] if isinstance(e, tuple) else e for e in events]
    tables = list(v.tables)
    assert [lb for lb in labels if isinstance(lb, bytes) and lb.startswith(b"table/")] == [
        b"table/" + t.encode() for t in tables]
    looked = [i for i, lb in enumerate(labels) if isinstance(lb, bytes) and lb.startswith(b"lookup/")]
    claims = [i for i, lb in enumerate(labels) if isinstance(lb, bytes) and lb.startswith(b"claim/")]
    folds = [i for i, lb in enumerate(labels) if isinstance(lb, str) and lb.startswith("fold/")]
    assert max(claims) < min(looked) and max(looked) < min(folds)
    proofs = prover.open_tables()
    assert [events[i][1] for i in looked] == [len(proofs[t]).to_bytes(8, "little") + b"".join(proofs[t])
                                              for t in tables]

    def first_columns(verifier):
        ch = proto.Challenger(fiat_shamir=True)
        proto._absorb_statement(ch, verifier, x)
        return ch.columns("probe", 1 << 20, 8)

    base = first_columns(v)
    for t in tables:                              # a table's root, rows or tokens change the challenges
        table = v.tables[t]
        for change in (dict(root=bytes(32)), dict(n_rows=table.n_rows + 1), dict(n_tokens=table.n_tokens + 1)):
            w = copy.copy(v)
            w.tables = dict(v.tables, **{t: dataclasses.replace(v.tables[t], **change)})
            assert not torch.equal(first_columns(w), base)


def test_a_verifier_refuses_a_table_key_that_does_not_fit():
    graph, _, coms = _planned("gpt", "R8c")
    params = proto.params_for(40, len(graph.mat_ops), plan=coms.plan)
    kw = dict(publics=coms.publics, groups=coms.group_publics)
    tables = coms.table_publics
    proto.Verifier(graph.public(), params, "C", tables=tables, **kw)
    token = next(iter(tables))
    linear = next(op for op in graph.mat_ops if op.layout == "linear")

    def refused(match, tables, mode="C", **extra):
        with pytest.raises(ValueError, match=match):
            proto.Verifier(graph.public(), params, mode, tables=tables, **dict(kw, **extra))

    refused("exactly one committed matrix or table", {})                       # an embedding op checked by nothing
    refused("names no embedding op", dict(tables, **{linear.name: tables[token]}))
    refused("shape of its op", dict(tables, **{token: dataclasses.replace(tables[token], n_rows=3)}))
    refused("shape of its op", dict(tables, **{token: dataclasses.replace(tables[token], n_tokens=5)}))
    refused("mode-C", tables, mode="K")
    row = proto.commit_graph(graph, 4, policy="R8")
    refused("exactly one committed matrix or table", tables,                  # the table's op also in rows
            publics=dict(coms.publics, **{token: row.publics[token]}))
    with pytest.raises(ValueError, match="modes K and Kpre"):
        proto.Verifier(graph.public(), params, "C", lookups=True, tables=tables, **kw)


@pytest.mark.parametrize("kind", [*KINDS, "llama-pruned"])
@pytest.mark.parametrize("mode", ["K", "Kpre"])
def test_a_verifier_with_lookups_reads_the_embedding_rows_itself(kind, mode):
    """Modes K and Kpre with ``lookups``: no claims of the embedding ops travel, the verifier derives
    them from its own weights (the same tensors), and a prover whose embedding is wrong is caught at the
    ops after it (its claims follow from the wrong embedding; one entry: a shift of every entry is
    invisible to a LayerNorm)."""
    graph, _, _ = _planned(kind, "R8c")
    x = _query(kind)
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    params = proto.params_for(40, len(graph.mat_ops))
    embeds = [op.name for op in graph.mat_ops if op.layout == "embed"]
    honest = proto.Prover(graph).claims(x)
    for stream in (False, True):
        v = proto.Verifier(graph.public(), params, mode, weights=weights, lookups=True, stream=stream)
        if mode == "Kpre":
            v.precompute(proto.Challenger(seed=3))
            assert set(v._pre) == {op.name for op in graph.mat_ops} - set(embeds)
        assert [op.name for op in v._own_rows()] == embeds and not set(embeds) & {op.name for op in v._row_ops()}
        inputs = v.derive(x, {k: z for k, z in honest.items() if k not in embeds})
        assert inputs is not None
        for wire in (False, True):
            res = proto.run_query(proto.Prover(graph), v, x, seed=1, wire=wire)
            assert res["accepted"], res["rejected_at"]
            bad = proto.run_query(proto.Prover(graph), v, x, seed=2, wire=wire, forward_kwargs={
                "tamper": lambda op, z: _bump(z) if op.name == embeds[0] else z})
            assert bad["rejected_at"] == "freivalds"


@pytest.mark.parametrize("stream", [False, True])
def test_malformed_claims_without_the_embeddings_are_rejected(stream, monkeypatch):
    """A verifier with ``lookups`` decodes the ``PVC3`` claims of every op but the embeddings: the claims
    of all ops (the default message), or any malformed message, are rejected at ``range_or_shape``."""
    graph, _, _ = _planned("qwen", "R8c")
    x = _query("qwen")
    v = proto.Verifier(graph.public(), proto.params_for(40, len(graph.mat_ops)), "K",
                       weights={op.name: (op.weight, op.bias) for op in graph.mat_ops}, lookups=True, stream=stream)
    prover = proto.Prover(graph)
    encode = cc.encode
    everything = encode(list(prover.claims(x).values()))
    for i, change in enumerate((lambda b: b + b"\0", lambda b: b[:-1], lambda b: everything)):
        monkeypatch.setattr(proto.claimcodec, "encode", lambda zs, change=change: change(encode(zs)))
        assert proto.run_query(prover, v, x, seed=i, wire=True)["rejected_at"] == "range_or_shape"
        monkeypatch.undo()
    assert proto.run_query(prover, v, x, seed=5, wire=True)["accepted"]


def _bump(z):
    z = z.clone()
    z[0, 0] += 1
    return z


def test_the_transcript_of_a_verifier_with_lookups(monkeypatch):
    graph, _, _ = _planned("opt", "R8c")
    x = _query("opt")
    absorbed = []
    absorb = proto.Challenger.absorb
    monkeypatch.setattr(proto.Challenger, "absorb", lambda self, label, blob: (
        absorbed.append((label, bytes(blob))), absorb(self, label, blob)))
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    params = proto.params_for(40, len(graph.mat_ops), fiat_shamir=True)
    embeds = [op.name for op in graph.mat_ops if op.layout == "embed"]
    for lookups in (False, True):
        absorbed.clear()
        v = proto.Verifier(graph.public(), params, "K", weights=weights, lookups=lookups)
        assert proto.run_query(proto.Prover(graph), v, x)["accepted"]
        records = dict(absorbed)
        assert (b"lookups" in records) == lookups
        assert lookups or not any(label == b"lookups" for label, _ in absorbed)
        claims = {label[len(b"claim/"):].decode() for label, _ in absorbed if label.startswith(b"claim/")}
        assert claims == {op.name for op in graph.mat_ops} - (set(embeds) if lookups else set())
        if lookups:
            assert records[b"lookups"] == b"".join(len(n).to_bytes(2, "little") + n.encode() for n in embeds)


def test_the_lookup_byte_model():
    """``analytic.proof_bytes`` of a plan's tables: their claims a byte each, their multiproofs exact for
    the ids they look up (``multiproof_size``) or, without ids, the expectation for distinct uniform ids
    (``expected_lookup_nodes``, checked against sampling); and the setup's trees and leaves."""
    import random

    for v, m in ((97, 9), (1024, 64), (50257, 64), (300, 250), (5, 3)):
        n = next_pow2(v)
        rnd = random.Random(v)
        sampled = np.mean([multiproof_size(rnd.sample(range(v), m), n.bit_length() - 1) for _ in range(3000)])
        assert abs(analytic.expected_lookup_nodes(v, m) - sampled) < 0.02 * sampled + 0.05
    assert analytic.expected_lookup_nodes(1024, 64) == pytest.approx(analytic.expected_multiproof_nodes(1024, 64))
    cfg = CONFIGS["gpt2"]
    ops = analytic.decoder_shapes(cfg, 2)
    plan = plan_commitment(ops, "R16c")
    params = proto.params_for(128, len(ops), plan=plan)
    b = analytic.proof_bytes(ops, params, 64, plan=plan)
    exact = analytic.proof_bytes(ops, params, 64, plan=plan, table_ids={"pos": range(64)})
    row_plan = plan_commitment(ops, "R16")
    rows = analytic.proof_bytes(ops, proto.params_for(128, len(ops), plan=row_plan), 64, plan=row_plan)
    embed_claims = sum(op.n_rows * 64 for op in ops if op.layout == "embed")
    assert b["claims"] == rows["claims"] - 3 * embed_claims                  # int8 rows instead of int32
    assert exact["paths"] - b["paths"] == pytest.approx(32 * (multiproof_size(range(64), 10)
                                                              - analytic.expected_lookup_nodes(1024, 64)))
    setup = analytic.setup_size(ops, plan=plan)
    trees = {g: plan.matrix(ms[0]).n_points for g, ms in plan.groups}
    assert setup["trees"] == len(trees) + 2 and setup["leaves"] == sum(trees.values()) + 65536 + 1024


# -- the experiment scripts ---------------------------------------------------------------------------------

def _script(folder: str, name: str):
    import importlib.util

    path = Path(__file__).resolve().parents[1] / "experiments" / folder / f"{name}.py"
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(f"{name}_lookups_test", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_bench_names_the_pruned_and_lookup_cells_once(monkeypatch):
    """``--prune-last`` and ``--lookups`` add their suffixes (never a --tag's), run the decoders only, and
    ``--lookups`` the K and Kpre cells only (``--policy`` the mode-C ones)."""
    bench = _script("4_defence_benchmark", "bench")
    for argv, needs in ((["llm", "--model", "gpt2", "--tag", "_prune"], "_prune"),
                        (["llm", "--model", "gpt2", "--tag", "_x_lookups"], "_lookups"),
                        (["cnn", "--model", "lenet5", "--prune-last"], "decoders"),
                        (["cnn", "--model", "lenet5", "--lookups"], "decoders"),
                        (["llm", "--model", "gpt2", "--lookups", "--policy", "R8c"], "one at a time")):
        monkeypatch.setattr(sys, "argv", ["bench.py", *argv])
        with pytest.raises(SystemExit, match=needs):
            bench.main()
    seen = {}
    monkeypatch.setattr(bench, "env_info", lambda: {"pvi_path": "", "torch_threads": 1})
    monkeypatch.setattr(bench, "claim_platform", lambda env: None)       # (nothing is written)
    monkeypatch.setattr(bench, "suite_llm", lambda args, env: seen.update(tag=bench.TAG, args=args))
    monkeypatch.setattr(sys, "argv", ["bench.py", "llm", "--model", "gpt2", "--prune-last", "--lookups",
                                      "--wire", "--tag", "_wire", "--platform", "test"])
    bench.main()
    assert seen["tag"] == "_wire_prune_lookups" and seen["args"].prune_last and seen["args"].lookups
    args = bench.build_parser().parse_args(["llm", "--model", "gpt2", "--lookups"])
    args.force = True
    assert bench._todo(args, "llm", "gpt2", "defence_Kpre_int_lam40_T8_L1")
    assert not bench._todo(args, "llm", "gpt2", "defence_C_int_lam40_T8_L1")
    assert not bench._todo(args, "llm", "gpt2", "commit_T8_L1")


def test_plan_bytes_rows_of_pruned_builds_with_lookups():
    """``plan_bytes.run_rows`` on 1- and 2-block pruned builds (claims only): the measured claims of the
    c policy's tables and of the Kpre verifier's own rows are the model's, and the whole model's rows are
    the model on the extrapolated claims."""
    pb = _script("6_improvements", "plan_bytes")
    rows = pb.run_rows("gpt2:6:1,2:256", 40, ["R8c"], 2, wires=(False, True), claims_only=True, prune_last=True,
                       kpre=(False, True))
    whole = [r for r in rows if r["layers"] == 12]
    assert {(r["mode"], r["lookups"], r["wire"]) for r in whole} == {
        ("C", False, False), ("C", False, True), ("Kpre", False, False), ("Kpre", False, True),
        ("Kpre", True, False), ("Kpre", True, True)}
    cfg = dataclasses.replace(CONFIGS["gpt2"], vocab=256)
    ops = analytic.decoder_shapes(cfg, prune_last=True)
    cols = analytic.decoder_claim_columns(cfg, 6, prune_last=True)
    for r in whole:
        if not r["wire"]:
            plan = plan_commitment(ops, "R8c") if r["mode"] == "C" else None
            want = analytic.proof_bytes(ops, proto.params_for(40, len(ops), fiat_shamir=r["challenges"] == "fs",
                                                              plan=plan), cols, plan=plan, mode=r["mode"],
                                        lookups=r["lookups"], table_ids={"pos": range(6)})
            assert r["bytes_claims"] == want["claims"] and r["bytes_total"] == pytest.approx(sum(want.values()))
    kpre = {r["lookups"]: r["bytes_claims"] for r in whole if r["mode"] == "Kpre" and not r["wire"]}
    assert kpre[False] - kpre[True] == 4 * 6 * 768 * 2                  # the token and position rows, not sent

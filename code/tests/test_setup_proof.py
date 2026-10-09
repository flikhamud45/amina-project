"""The one-time setup proximity proof (``pvi.fullcheck.setup_proof``) and the split-commitment attack it stops.

* An honest commitment plan (row- and col-layout members of shared trees) passes its setup proof; a tampered
  message, opened column, multiproof, graph digest or root fails it.
* The split commitment: a committer binds, in one tree, model ``A``'s encoded columns on a secret half of the
  positions and model ``A'``'s (one weight row changed) on the other half.  **Without** a setup proof, a prover
  that computes with ``A'`` passes a query exactly when all of the tree's ``t`` opened columns avoid ``A``'s
  half: about ``2**-t`` per query, measured here with a small ``t``.  **With** it, the committer's setup proof is
  rejected (its combinations of ``A``'s rows disagree with ``A'``'s columns).
* The per-query column counts after a setup proof are those of the untrusted bound: ``m + e`` for ``m``.
"""

from __future__ import annotations

import dataclasses
import math

import pytest
import torch

from pvi.fullcheck import protocol as proto
from pvi.fullcheck import setup_proof as sp
from pvi.fullcheck.commitment import GroupCommitment, MerkleTree, multiproof
from pvi.fullcheck.plans import column_bits, column_error_log2
from pvi.fullcheck.transformer import build_decoder

from test_plans import _TINY


def _committed(kind: str, policy: str = "auto"):
    graph = build_decoder(_TINY[kind], calib_tokens=8, seed=1, prune_last=True)
    return graph, proto.commit_graph(graph, 4, policy=policy)


def _params(graph, coms, lam=128):
    return sp.setup_params(coms.publics, coms.group_publics, lam, column_bits(lam, len(graph.mat_ops)))


@pytest.mark.parametrize("kind,policy", [("gpt", "auto"), ("llama", "auto"), ("qwen", "cnn16c"), ("opt", "R16c")])
def test_an_honest_commitment_passes_its_setup_proof(kind, policy):
    graph, coms = _committed(kind, policy)
    params = _params(graph, coms)
    proof = sp.prove_setup(coms.groups, coms.publics, graph.digest(), params)
    assert sp.verify_setup(coms.publics, coms.group_publics, graph.digest(), params, proof)
    assert proof.nbytes() > 0 and params.s >= 7
    # every query then opens at least the honest bound's columns (m + e for m)
    honest = dict(proto.params_for(128, len(graph.mat_ops), plan=coms.plan).group_columns)
    assert all(t >= honest[g] for g, t in sp.query_group_columns(params, coms.group_publics))


def test_a_tampered_setup_proof_is_rejected():
    graph, coms = _committed("qwen", "cnn16c")                      # a plan with col-layout members
    params = _params(graph, coms)
    digest, publics, groups = graph.digest(), coms.publics, coms.group_publics
    proof = sp.prove_setup(coms.groups, publics, digest, params)
    assert sp.verify_setup(publics, groups, digest, params, proof)

    def changed(fn):
        p = sp.SetupProof({k: v.clone() for k, v in proof.messages.items()},
                          {k: (c.clone(), o.clone(), list(mp)) for k, (c, o, mp) in proof.openings.items()})
        fn(p)
        return p

    name = next(iter(proof.messages))
    tree = next(iter(proof.openings))
    bad = [changed(lambda p: p.messages[name].__setitem__((0, 0), (p.messages[name][0, 0] + 1) % proto.P)),
           changed(lambda p: p.openings[tree][1].__setitem__((0, 0), (p.openings[tree][1][0, 0] + 1) % proto.P)),
           changed(lambda p: p.openings[tree][2].pop()),
           changed(lambda p: p.messages.pop(name))]
    for p in bad:
        assert not sp.verify_setup(publics, groups, digest, params, p)
    other = build_decoder(_TINY["qwen"], calib_tokens=8, seed=2, prune_last=True).digest()
    assert not sp.verify_setup(publics, groups, other, params, proof)            # another graph's digest
    g0 = next(iter(groups))
    moved = dict(groups, **{g0: dataclasses.replace(groups[g0], root=bytes(32))})
    assert not sp.verify_setup(publics, moved, digest, params, proof)             # another root


# -- the split commitment --------------------------------------------------------------------------------

class _Split(GroupCommitment):
    """A tree whose column ``c`` is ``a``'s for ``c`` in ``in_a`` and ``b``'s otherwise (same members, tags and
    codeword length); it opens each column from the model that the leaf binds."""

    def __init__(self, a: GroupCommitment, b: GroupCommitment, in_a: set[int]) -> None:
        leaves = [a.tree.levels[0][c] if c in in_a else b.tree.levels[0][c] for c in range(a.n_points)]
        super().__init__(tag=a.tag, members=a.members, n_points=a.n_points, tree=MerkleTree(leaves))
        self.a, self.b, self.in_a = a, b, in_a

    def open(self, columns, device="cpu", weights=None):
        ca, _ = self.a.open(columns, device)
        cb, _ = self.b.open(columns, device)
        pick = torch.tensor([int(c) in self.in_a for c in columns.tolist()])
        return torch.where(pick[None, :], ca, cb), multiproof(self.tree, columns.tolist())


def _split_setup(seed: int = 0):
    """``(graph A', its prover's commitment, the split group, the split coms)``: one row of one row-layout op's
    weights changed in A', the tree holding it split between A and A' (half the columns each)."""
    graph_a, coms_a = _committed("gpt", "auto")
    graph_b = build_decoder(_TINY["gpt"], calib_tokens=8, seed=1, prune_last=True)
    target = next(m for g in coms_a.groups.values() for m, c in g.members.items()
                  if type(c).__name__ == "WeightCommitment" and c.weight.shape[0] > 1)
    op = next(o for o in graph_b.mat_ops if o.name == target)
    op.weight[0] = (op.weight[0].to(torch.int16) + 1).clamp(-127, 127).to(torch.int8)   # A' differs in one row
    coms_b = proto.commit_graph(graph_b, 4, policy="auto")
    gname = next(g for g, grp in coms_a.groups.items() if target in grp.members)
    n = coms_a.groups[gname].n_points
    rng = torch.Generator().manual_seed(seed)
    in_a = set(torch.randperm(n, generator=rng)[: n // 2].tolist())
    split = _Split(coms_a.groups[gname], coms_b.groups[gname], in_a)
    groups = dict(coms_a.groups, **{gname: split})
    coms_s = proto.GraphCommitment(dict(coms_b), coms_b.plan, groups, coms_b.tables)
    return graph_a, graph_b, coms_s, gname


def test_without_a_setup_proof_the_split_commitment_passes_about_two_to_the_minus_t_of_the_queries():
    graph_a, graph_b, coms_s, gname = _split_setup()
    params = proto.params_for(6, len(graph_b.mat_ops), plan=coms_s.plan)        # a small t, to see it pass
    t = params.columns_for(gname)
    v = proto.Verifier(graph_b.public(), params, "C", publics=coms_s.publics, groups=coms_s.group_publics,
                       tables=coms_s.table_publics)
    prover = proto.Prover(graph_b, commitments=coms_s)                         # computes with A'
    n = coms_s.groups[gname].n_points
    expect = 2 ** column_error_log2(n // 2 + 1, n, t)                           # all t columns in A''s half
    trials, passed = 300, 0
    for i in range(trials):
        x = torch.randint(0, _TINY["gpt"].vocab, (1, 9), generator=torch.Generator().manual_seed(100 + i))
        out = proto.run_query(prover, v, x, seed=i)
        passed += out["accepted"]
        assert out["accepted"] or out["rejected_at"].startswith("columns")
    assert passed > 0                                                           # the gap is real ...
    assert abs(passed / trials - expect) < 4 * math.sqrt(expect / trials) + 0.02, (passed, expect, t)   # ... ~2**-t


def test_the_setup_proof_rejects_the_split_commitment():
    graph_a, graph_b, coms_s, gname = _split_setup(seed=1)
    params = _params(graph_b, coms_s)
    # the committer's best setup proof: combinations of A's rows (it cannot know a codeword matching both halves)
    proof = sp.prove_setup(coms_s.groups, coms_s.publics, graph_b.digest(), params)
    assert not sp.verify_setup(coms_s.publics, coms_s.group_publics, graph_b.digest(), params, proof)


def test_the_parameters_meet_their_bounds():
    graph, coms = _committed("llama", "auto")
    lam, qb = 128, column_bits(128, len(graph.mat_ops))
    params = sp.setup_params(coms.publics, coms.group_publics, lam, qb)
    beta = lam + params.grinding_bits + math.log2(2 * len(coms.publics))
    assert params.s * proto.LOG2_P >= beta + math.log2(max(g.n_points for g in coms.group_publics.values()))
    for g, grp in coms.group_publics.items():
        for m in grp.members:
            pub, e = coms.publics[m], params.e_of(m)
            assert 2 * e < pub.n_points - pub.row_length + 1                     # unique decoding
            assert column_error_log2(pub.n_points - e, pub.n_points, params.t0_of(g)) <= -beta
            assert column_error_log2(pub.row_length + e, pub.n_points, dict(params.t)[g]) <= -qb

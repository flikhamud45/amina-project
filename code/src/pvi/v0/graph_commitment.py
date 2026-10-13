"""The commitment to a graph's weights: one Reed--Solomon/Merkle commitment per weight op, or the
matrices and shared trees of a commitment plan, with the prover's setup cache."""

from __future__ import annotations

from pvi.commit import setup_cache
from pvi.commit.commitment import (CommitmentPublic, GroupCommitment, GroupPublic, MerkleTree, TableCommitment,
                                   TablePublic, TransposedCommitment, WeightCommitment, next_pow2)
from pvi.commit.plans import CommitmentPlan, plan_commitment
from pvi.core.graph import IntGraph
from pvi.v0.params import Z_BOUND, claim_bound


class GraphCommitment(dict):
    """``{op name: WeightCommitment}`` for every weight op; under a commitment plan
    ``{matrix name: its commitment}`` for every encoded matrix (a row-layout op's
    ``WeightCommitment`` under the op's name, a col-layout matrix's ``TransposedCommitment``), the
    ``plan``, its ``groups`` (``{name: GroupCommitment}``), whose trees bind the members' columns,
    and its lookup ``tables`` (``{op name: TableCommitment}``)."""

    def __init__(self, members: dict[str, WeightCommitment | TransposedCommitment], plan: CommitmentPlan | None = None,
                 groups: dict[str, GroupCommitment] | None = None,
                 tables: dict[str, TableCommitment] | None = None) -> None:
        super().__init__(members)
        self.plan = plan
        self.groups = groups or {}
        self.tables = tables or {}
        self.setup_cache = "off"        # commit_graph: "off", "miss" (computed and stored) or "hit" (rebuilt)

    @property
    def publics(self) -> dict[str, CommitmentPublic]:
        return {name: c.public for name, c in self.items()}

    @property
    def group_publics(self) -> dict[str, GroupPublic]:
        return {name: g.public for name, g in self.groups.items()}

    @property
    def table_publics(self) -> dict[str, TablePublic]:
        return {name: t.public for name, t in self.tables.items()}


def commit_graph(graph: IntGraph, rate: int, device="cpu", *, policy: str = "paper",
                 model_ops=None, cache=None, cache_checks: int = 4) -> GraphCommitment:
    """Commit every weight op, after checking honest claims fit the range check: on a tree of its
    own at ``rate`` (``policy="paper"``, the report), or as the commitment plan of ``policy`` lays
    the ops out (:func:`plans.plan_commitment`, with ``rate`` its base rate; ``model_ops``: the
    whole model's op shapes when ``graph`` is a build of a few of its blocks).

    ``cache``: the setup cache's folder (:mod:`setup_cache`); ``None`` reads ``PVI_SETUP_CACHE`` (unset: no cache),
    ``False`` never uses one. On a hit every tree is rebuilt from stored leaf digests, after recomputing
    ``cache_checks`` random leaves of each from the weights; the result is the same commitment (same roots,
    openings and proofs). The result's ``setup_cache`` says ``"off"``, ``"miss"`` (computed, then stored) or
    ``"hit"``."""
    for op in graph.mat_ops:
        b = claim_bound(op)
        if b >= Z_BOUND:
            raise ValueError(f"{op.name}: honest claims can reach {b} >= Z_BOUND = {Z_BOUND}")
    plan = plan_commitment(graph.mat_ops, policy, rate=rate, model_ops=model_ops)
    where = setup_cache.folder(cache)
    if where is not None:
        k = setup_cache.key(graph, rate, plan)
        stored = setup_cache.read(where, k)
        if stored is not None:
            coms = _commit_from_leaves(graph, rate, device, plan, stored, cache_checks)
            if coms is not None:
                coms.setup_cache = "hit"
                return coms
    coms = _commit_fresh(graph, rate, device, plan)
    if where is not None:
        setup_cache.save(where, k, _tree_leaves(coms))
        coms.setup_cache = "miss"
    return coms


def _member_of(m, ops):
    """The (tree-less) member commitment of the planned matrix ``m``."""
    if m.layout == "col":
        return TransposedCommitment.member(m.name.encode(), {o: (ops[o].weight, ops[o].bias) for o in m.members},
                                           m.n_points)
    return WeightCommitment.member(m.name.encode(), ops[m.name].weight, ops[m.name].bias, m.n_points)


def _commit_fresh(graph: IntGraph, rate: int, device, plan) -> GraphCommitment:
    """:func:`commit_graph`'s encoding and hashing (no cache)."""
    if plan is None:
        return GraphCommitment({op.name: WeightCommitment.build(op.name.encode(), op.weight, op.bias, rate=rate,
                                                                device=device) for op in graph.mat_ops})
    ops = {op.name: op for op in graph.mat_ops}
    groups = {g: GroupCommitment.build(g.encode(), {m: _member_of(plan.matrix(m), ops) for m in members},
                                       device=device)
              for g, members in plan.groups}
    members = {m: c for g in groups.values() for m, c in g.members.items()}
    tables = {m.name: TableCommitment.build(m.name.encode(), ops[m.name].weight) for m in plan.tables}
    return GraphCommitment({m.name: members[m.name] for m in plan.coded}, plan, groups, tables)


def _tree_leaves(coms: GraphCommitment) -> dict[str, list[bytes]]:
    """``{tree name: its leaf digests}`` of a fresh commitment, for :func:`setup_cache.save`."""
    if coms.plan is None:
        return {f"op:{name}": c.tree.levels[0] for name, c in coms.items()}
    out = {f"group:{g}": gc.tree.levels[0] for g, gc in coms.groups.items()}
    out.update({f"table:{t}": tc.tree.levels[0] for t, tc in coms.tables.items()})
    return out


def _commit_from_leaves(graph: IntGraph, rate: int, device, plan, stored: dict[str, list[bytes]],
                        checks: int) -> GraphCommitment | None:
    """The commitment rebuilt from stored leaves, or ``None`` if a tree is missing or a spot check fails."""
    if plan is None:
        out = {}
        for op in graph.mat_ops:
            leaves = stored.get(f"op:{op.name}")
            n_points = rate * next_pow2(op.weight.shape[1] + (1 if op.bias is not None else 0))
            c = WeightCommitment.member(op.name.encode(), op.weight, op.bias, n_points)
            if leaves is None or not setup_cache.check_matrix(c, leaves, device, checks):
                return None
            c.tree = MerkleTree(leaves)
            out[op.name] = c
        return GraphCommitment(out)
    ops = {op.name: op for op in graph.mat_ops}
    groups = {}
    for g, names in plan.groups:
        mem = {m: _member_of(plan.matrix(m), ops) for m in names}
        n_points = next(iter(mem.values())).n_points
        leaves = stored.get(f"group:{g}")
        if (leaves is None or any(m.n_points != n_points for m in mem.values())
                or not setup_cache.check_group(g.encode(), mem, n_points, leaves, device, checks)):
            return None
        groups[g] = GroupCommitment(tag=g.encode(), members=mem, n_points=n_points, tree=MerkleTree(leaves))
    tables = {}
    for m in plan.tables:
        w = ops[m.name].weight
        leaves = stored.get(f"table:{m.name}")
        if leaves is None or not setup_cache.check_table(m.name.encode(), w, leaves, checks):
            return None
        tables[m.name] = TableCommitment(tag=m.name.encode(), n_rows=w.shape[0], n_tokens=w.shape[1],
                                         tree=MerkleTree(leaves))
    members = {m: c for gc in groups.values() for m, c in gc.members.items()}
    return GraphCommitment({m.name: members[m.name] for m in plan.coded}, plan, groups, tables)

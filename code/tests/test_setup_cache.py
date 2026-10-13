"""The setup cache (``commit_graph(cache=...)``, :mod:`pvi.commit.setup_cache`).

A commitment rebuilt from stored leaves is the fresh commitment: the same roots and trees, and the same proofs and
verdicts; the key separates different weights and plans; a damaged file is a miss, never a wrong commitment; and
without a cache folder (the default) nothing is read or written."""
from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

import pvi.v0 as proto
from pvi.commit import setup_cache
from pvi.commit.plans import plan_commitment
from pvi.models.decoder import DecoderConfig, build_decoder

_TINY = {
    "gpt": DecoderConfig("tiny-gpt", 64, 2, 4, 4, 16, 128, 97, max_pos=64),
    "llama": DecoderConfig("tiny-llama", 64, 2, 4, 4, 16, 160, 97, norm="rmsnorm", mlp="swiglu",
                           pos="rope", bias=False, tied=False),
    "qwen": DecoderConfig("tiny-qwen", 64, 2, 8, 2, 16, 160, 97, norm="rmsnorm", mlp="swiglu",
                          pos="rope", rope_theta=1e6, bias=False, qk_norm=True),
}


def _model(kind):
    graph = build_decoder(_TINY[kind], calib_tokens=8, seed=1)
    return graph, torch.randint(0, _TINY[kind].vocab, (1, 9), generator=torch.Generator().manual_seed(2))


def _trees(coms):
    if coms.plan is None:
        return {n: c.tree.levels for n, c in coms.items()}
    out = {f"g:{g}": gc.tree.levels for g, gc in coms.groups.items()}
    out.update({f"t:{t}": tc.tree.levels for t, tc in coms.tables.items()})
    return out


def _query(graph, coms, x, fiat_shamir):
    params = proto.params_for(40, len(graph.mat_ops), fiat_shamir=fiat_shamir, plan=coms.plan)
    kw = {} if coms.plan is None else dict(groups=coms.group_publics, tables=coms.table_publics)
    v = proto.Verifier(graph.public(), params, "C", publics=coms.publics, **kw)
    return proto.run_query(proto.Prover(graph, commitments=coms), v, x, seed=None if fiat_shamir else 1)


@pytest.mark.parametrize("kind,policy", [("gpt", "per-op"), ("llama", "R8"), ("llama", "R8c"), ("qwen", "tightc"),
                                         ("gpt", "cnn12c")])
def test_a_cached_commitment_is_the_fresh_one(tmp_path, kind, policy):
    graph, x = _model(kind)
    fresh = proto.commit_graph(graph, 4, policy=policy, cache=False)
    assert fresh.setup_cache == "off" and not list(tmp_path.iterdir())
    first = proto.commit_graph(graph, 4, policy=policy, cache=tmp_path)
    second = proto.commit_graph(graph, 4, policy=policy, cache=tmp_path)
    assert (first.setup_cache, second.setup_cache) == ("miss", "hit")
    assert len(list(tmp_path.glob("*.npz"))) == 1
    for c in (first, second):
        assert c.publics == fresh.publics
        assert c.group_publics == fresh.group_publics and c.table_publics == fresh.table_publics
        assert _trees(c) == _trees(fresh)
    for fs in (False, True):
        a, b = _query(graph, fresh, x, fs), _query(graph, second, x, fs)
        assert a["accepted"] and b["accepted"] and a["bytes"] == b["bytes"]


def test_the_key_separates_weights_biases_and_plans(tmp_path):
    graph, _ = _model("gpt")
    plan = lambda p: plan_commitment(graph.mat_ops, p, rate=4)                    # noqa: E731
    k = setup_cache.key(graph, 4, plan("R8c"))
    assert k == setup_cache.key(graph, 4, plan("R8c"))
    assert k != setup_cache.key(graph, 4, plan("R8")) != setup_cache.key(graph, 4, None)
    assert k != setup_cache.key(graph, 8, plan("R8c"))
    other = copy.deepcopy(graph)
    other.mat_ops[1].weight.view(-1)[5] += 1
    assert setup_cache.key(other, 4, plan("R8c")) != k
    biased = next(op for op in other.mat_ops if op.bias is not None)
    other2 = copy.deepcopy(graph)
    next(op for op in other2.mat_ops if op.name == biased.name).bias.view(-1)[0] += 1
    assert setup_cache.key(other2, 4, plan("R8c")) != k
    # a different weight with the same key cannot happen; the same weights under a new name (tag) differ too
    proto.commit_graph(graph, 4, policy="R8c", cache=tmp_path)
    assert proto.commit_graph(other, 4, policy="R8c", cache=tmp_path).setup_cache == "miss"


def test_damaged_files_are_misses_not_wrong_commitments(tmp_path):
    graph, x = _model("llama")
    fresh = proto.commit_graph(graph, 4, policy="R8c", cache=False)
    proto.commit_graph(graph, 4, policy="R8c", cache=tmp_path)
    (path,) = tmp_path.glob("*.npz")
    k = path.stem
    trees = setup_cache.read(tmp_path, k)
    name = next(n for n in trees if n.startswith("group:"))
    trees[name] = [bytes(32 - len(b"x")) + b"x" for _ in trees[name]]            # every leaf of one tree wrong
    setup_cache.save(tmp_path, k, trees)
    again = proto.commit_graph(graph, 4, policy="R8c", cache=tmp_path)
    assert again.setup_cache == "miss" and again.group_publics == fresh.group_publics
    assert _query(graph, again, x, False)["accepted"]
    assert proto.commit_graph(graph, 4, policy="R8c", cache=tmp_path).setup_cache == "hit"   # rewritten
    path.write_bytes(path.read_bytes()[:100])                                     # truncated
    assert proto.commit_graph(graph, 4, policy="R8c", cache=tmp_path).setup_cache == "miss"
    trees = setup_cache.read(tmp_path, k)
    del trees[name]                                                               # a missing tree
    setup_cache.save(tmp_path, k, trees)
    assert proto.commit_graph(graph, 4, policy="R8c", cache=tmp_path).setup_cache == "miss"


def test_off_by_default_and_on_with_a_folder(tmp_path):
    graph, _ = _model("gpt")
    assert proto.commit_graph(graph, 4, policy="R8").setup_cache == "off"
    assert proto.commit_graph(graph, 4, policy="R8", cache=tmp_path / "cache").setup_cache == "miss"
    assert proto.commit_graph(graph, 4, policy="R8", cache=tmp_path / "cache").setup_cache == "hit"
    assert proto.commit_graph(graph, 4, policy="R8", cache=None).setup_cache == "off"


def test_the_files_hold_only_digests(tmp_path):
    """No pickle and no weights: one ``uint8 [n, 32]`` array per tree and a JSON name list."""
    graph, _ = _model("qwen")
    proto.commit_graph(graph, 4, policy="R8c", cache=tmp_path)
    (path,) = tmp_path.glob("*.npz")
    with np.load(path, allow_pickle=False) as z:
        arrays = [z[n] for n in z.files if n != "meta"]
    assert arrays and all(a.dtype == np.uint8 and a.ndim == 2 and a.shape[1] == 32 for a in arrays)
    assert sum(a.nbytes for a in arrays) < sum(op.weight.numel() for op in graph.mat_ops) * 4

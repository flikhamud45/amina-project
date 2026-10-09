"""V1's cut (``pvi.fullcheck.cut``) and the requantising ops it reads (``graph.requant_fn``): ``V1_SPEC.md`` tests
T0 (windows) and T9 (the builders' graphs are unchanged in value), and the cut plan's rules P1-P5.
"""

from __future__ import annotations

import dataclasses

import pytest
import torch

from pvi.fullcheck import cut
from pvi.fullcheck import protocol as proto
from pvi.fullcheck.graph import CheapOp, mul_add_half, requant, requant_fn
from pvi.fullcheck.plans import plan_commitment
from pvi.fullcheck.transformer import RES_MAX, SHIFT, DecoderConfig, build_decoder, with_lm_positions

# W from 2 to 2^16, powers of two, and W = floor / W + 1 = ceil
MULTS = [1 << 29, (1 << 29) - 1, 1 << 14, (1 << 14) + 1, 1 << 20, (1 << 30) // 3, 12345679, 87654321]

_TINY = {
    "gpt": DecoderConfig("tiny-gpt", 64, 2, 4, 4, 16, 128, 97, max_pos=64),
    "llama": DecoderConfig("tiny-llama", 64, 2, 4, 4, 16, 160, 97, norm="rmsnorm", mlp="swiglu",
                           pos="rope", bias=False, tied=False),
    "opt": DecoderConfig("tiny-opt", 64, 2, 4, 4, 16, 128, 97, mlp="relu", max_pos=64, embed_dim=32),
}


def _op(m, kind="requant"):
    return cut.CutOp("w", kind, m, 30, -127 if kind == "requant" else 0, 127 if kind == "requant" else 0,
                     4, 8, False, "c", 0, 0, "w", "row")


def test_the_claim_limit_is_the_protocols():
    assert cut.CLAIM_LIMIT == proto.Z_BOUND


@pytest.mark.parametrize("m", MULTS)
def test_windows_exhaustively(m):
    # every z from lo(-127) to lo(128): shift(z) = s exactly on [lo(s), lo(s + 1)), and lo(s) is the least such z
    first, last = (int(v) for v in cut.lo(torch.tensor([-127, 128]), m))
    z = torch.arange(first, last, dtype=torch.int64)
    s = cut.shift(z, m)
    assert int(s[0]) == -127 and int(s[-1]) == 127 and bool((s[1:] - s[:-1] <= 1).all())
    lower, width = cut.windows(s, m)
    assert bool(((lower <= z) & (z < lower + width)).all())
    assert bool((cut.shift(lower - 1, m) == s - 1).all()) and bool((cut.shift(lower + width, m) == s + 1).all())
    w_lo, w_hi = _op(m).widths
    assert set(width.unique().tolist()) <= {w_lo, w_hi} and 2 <= w_lo and w_hi <= 1 << 16
    assert torch.equal(cut.shift(z, m), mul_add_half(z, torch.tensor(m), SHIFT) >> SHIFT)


@pytest.mark.parametrize("m", MULTS)
def test_residual_windows_at_the_edges(m):
    a, b = cut.delta_range(m)
    for v, z in ((a, 1 - cut.CLAIM_LIMIT), (b, cut.CLAIM_LIMIT - 1)):
        lower, width = cut.windows(torch.tensor([v]), m)
        assert int(lower) <= z < int(lower + width)
    assert cut.lo(torch.tensor([a]), m) <= 1 - cut.CLAIM_LIMIT


@pytest.mark.parametrize("m", MULTS)
@pytest.mark.parametrize("kind", ["requant", "residual"])
def test_honest_split_and_public_windows(m, kind):
    g = torch.Generator().manual_seed(m % 1000)
    z = torch.randint(-(1 << 22), 1 << 22, (16, 40), generator=g) * torch.randint(1, 64, (16, 40), generator=g)
    z[0, :5] = torch.tensor([cut.CLAIM_LIMIT - 1, 1 - cut.CLAIM_LIMIT, 0, 1 << 28, -(1 << 28)])   # deep clamps
    z = z.clamp(1 - cut.CLAIM_LIMIT, cut.CLAIM_LIMIT - 1)
    op = _op(m, kind)
    s, delta, width, idx, exc_z = cut.split(z, op)
    if kind == "requant":
        assert torch.equal(s, requant(z, torch.tensor(m), SHIFT, -127, 127))
        assert bool((s.reshape(-1)[idx].abs() == 127).all())                   # exceptions only where clamped
    else:
        assert torch.equal(s, mul_add_half(z, torch.tensor(m), SHIFT) >> SHIFT) and idx.numel() == 0
        a, b = cut.delta_range(m)
        assert bool(((a <= s) & (s <= b)).all())
    assert bool(((0 <= delta) & (delta < width)).all())
    assert torch.equal(exc_z, z.reshape(-1)[idx]) and bool((idx[1:] > idx[:-1]).all())
    lower, w2 = cut.public_lw(s, idx, exc_z, op)
    assert torch.equal(z - lower, delta) and torch.equal(w2, width)
    # Lemma 0: another sent value puts the entry outside its window
    lower, w3 = cut.public_lw(s + 1, idx, exc_z, op)
    moved = torch.ones_like(s, dtype=torch.bool).reshape(-1)
    moved[idx] = False
    d3 = (z - lower).reshape(-1)[moved]
    assert not bool(((0 <= d3) & (d3 < w3.reshape(-1)[moved])).any())


def test_requant_fn_matches_the_old_closures_and_survives_device_copies():
    g = torch.Generator().manual_seed(3)
    a = torch.randint(-(1 << 20), 1 << 20, (2, 5, 7), generator=g)
    r = torch.randint(-(1 << 21), 1 << 21, (2, 5, 7), generator=g)
    p = {"kind": "requant", "mult": 777777, "shift": SHIFT, "lo": -127, "hi": 127}
    assert torch.equal(requant_fn(p)(a), requant(a, torch.tensor(777777), SHIFT, -127, 127))
    pr = {"kind": "residual", "mult": 1 << 25, "shift": SHIFT, "res_max": RES_MAX}
    want = (r + ((a * (1 << 25) + (1 << 29)) >> 30)).clamp(-RES_MAX, RES_MAX)
    assert torch.equal(requant_fn(pr)(r, a), want)
    with pytest.raises(ValueError):
        requant_fn({**p, "kind": "other"})
    graph = build_decoder(_TINY["gpt"], calib_tokens=8, seed=1)
    moved, _ = graph.with_constants_on("cpu")
    for op in moved.ops:
        if isinstance(op, CheapOp) and op.params.get("requant"):
            assert op.fn._pvi_requant == op.params["requant"]


def _old_style(graph):
    """The graph with every requantising op's function rebuilt as the builders wrote it before ``requant_fn``."""
    from pvi.fullcheck import transformer as tr

    ops = []
    for op in graph.ops:
        p = op.params.get("requant") if isinstance(op, CheapOp) else None
        if p is None:
            ops.append(op)
            continue
        m = torch.tensor(p["mult"], dtype=torch.int64)
        if p["kind"] == "requant":
            fn = lambda a, m=m: requant(a, m.to(a.device), SHIFT, -127, 127)        # noqa: E731
        else:
            fn = lambda a, b, m=m: tr._residual(a, b, m)                              # noqa: E731
        ops.append(dataclasses.replace(op, fn=fn, params={}))
    return dataclasses.replace(graph, ops=ops)


@pytest.mark.parametrize("kind", sorted(_TINY))
def test_t9_builders_give_the_same_logits(kind):
    graph = build_decoder(_TINY[kind], calib_tokens=8, seed=1)
    x = torch.randint(0, 97, (1, 9), generator=torch.Generator().manual_seed(2))
    old = _old_style(graph)
    a, _ = graph.forward(x)
    b, _ = old.forward(x)
    assert torch.equal(a[graph.output_name], b[old.output_name])
    assert graph.digest() != old.digest()
    rq = [op for op in graph.ops if isinstance(op, CheapOp) and op.params.get("requant")]
    assert rq and all(op.note in ("requant", "residual add") for op in rq)


def _columns(graph, x):
    return graph.claim_columns(x)


def _expected(graph, x):
    """P1 and P2 by hand: linear ops read only by a requantising op, claiming every token's column."""
    readers = graph.readers()
    out = set()
    for op, c in zip(graph.mat_ops, graph.claim_columns(x)):
        r = readers.get(op.output, [])
        rq = len(r) == 1 and isinstance(r[0], CheapOp) and bool(r[0].params.get("requant"))
        if op.layout == "linear" and rq and op.output != graph.output_name and c == x.shape[1]:
            out.add(op.name)
    return out


@pytest.mark.parametrize("kind", sorted(_TINY))
def test_the_cut_set_of_a_decoder(kind):
    graph = build_decoder(_TINY[kind], calib_tokens=8, seed=1)
    x = torch.randint(0, 97, (1, 9), generator=torch.Generator().manual_seed(2))
    plan = cut.CutPlan.from_graph(graph, _columns(graph, x), 9)
    assert plan.names() == _expected(graph, x) and len(plan.ops) >= 4 * _TINY[kind].n_layers
    assert graph.output_name not in {o.name + ".z" for o in plan.ops}
    assert plan.widths[0] == 1 and list(plan.widths) == sorted(set(plan.widths))
    assert plan.table_size() == sum(plan.widths) and plan.table()[0].shape == (plan.table_size(), 8)
    assert plan.n_leaves() == sum(op.n_rows for op in plan.ops) * 9
    assert len(plan.instances) == 1 and plan.n_vars(0) == ((plan.rows(0) - 1).bit_length(), 4)
    rows = 0
    for o in plan.ops:
        assert o.row_offset == rows and o.layout == "row"
        rows += o.n_rows
    # a smaller lmax splits it into consecutive instances of at most 2^lmax entries (a larger unit alone)
    small = cut.CutPlan.from_graph(graph, _columns(graph, x), 9, lmax=10)
    assert len(small.instances) > 1 and [o.name for o in small.ops] == [o.name for o in plan.ops]
    for b in range(len(small.instances)):
        ops = small.instance_ops(b)
        assert len(ops) == 1 or small.rows(b) * 9 <= 1 << 10
    assert small.digest() != plan.digest() and plan.digest() == cut.CutPlan.from_graph(graph, _columns(graph, x), 9).digest()


def test_pruned_and_generation_ops_stay_clear():
    cfg = _TINY["gpt"]
    full = build_decoder(cfg, calib_tokens=8, seed=1)
    pruned = build_decoder(cfg, calib_tokens=8, seed=1, prune_last=True)
    x = torch.randint(0, 97, (1, 9), generator=torch.Generator().manual_seed(2))
    a = cut.CutPlan.from_graph(full, full.claim_columns(x), 9)
    b = cut.CutPlan.from_graph(pruned, pruned.claim_columns(x), 9)
    one_col = {op.name for op, c in zip(pruned.mat_ops, pruned.claim_columns(x)) if c == 1}
    assert one_col and not (one_col & b.names()) and b.names() == _expected(pruned, x)
    assert len(b.ops) == len(a.ops) - 4                    # the last block's q, o, fc1, fc2 (its k and v stay cut)
    gen = with_lm_positions(full, 4)
    c = cut.CutPlan.from_graph(gen, gen.claim_columns(x), 9)
    assert c.names() == _expected(gen, x) and c.names() <= a.names()


def test_window_and_count_rules():
    graph = build_decoder(_TINY["gpt"], calib_tokens=8, seed=1)
    ops = list(graph.ops)
    i = next(k for k, op in enumerate(ops) if isinstance(op, CheapOp) and op.params.get("requant", {}).get("kind") == "requant")
    p = dict(ops[i].params["requant"], mult=(1 << 29) + 1)                     # floor(2^30 / m) = 1: P3 fails
    ops[i] = dataclasses.replace(ops[i], fn=requant_fn(p), params={"requant": p})
    g2 = dataclasses.replace(graph, ops=ops)
    victim = ops[i].inputs[0][:-2]
    assert victim not in cut.CutPlan.from_graph(g2, None, 9).names()
    # a function that is not requant_fn's, or params it does not carry: P1 fails
    ops[i] = dataclasses.replace(ops[i], fn=lambda a: a, params={"requant": p})
    assert victim not in cut.CutPlan.from_graph(dataclasses.replace(graph, ops=ops), None, 9).names()
    # P5: with T large enough, a width's count reaches 2^29 and the largest ops become clear
    big = cut.CutPlan.from_graph(graph, None, 1 << 22)
    counts = {}
    for o in big.ops:
        for w in set(o.widths):
            counts[w] = counts.get(w, 0) + o.n_rows * big.T
    assert all(n < 1 << 29 for n in counts.values()) and len(big.ops) < len(cut.CutPlan.from_graph(graph, None, 9).ops)


@pytest.mark.parametrize("policy", ["cnn16c", "tightc"])
def test_col_matrices_are_cut_whole(policy):
    graph = build_decoder(_TINY["llama"], calib_tokens=8, seed=1)
    cplan = plan_commitment(graph.mat_ops, policy)
    plan = cut.CutPlan.from_graph(graph, None, 9, cplan)
    names = plan.names()
    for mtx in cplan.matrices:
        if mtx.layout == "col":
            inside = [o in names for o in mtx.members]
            assert all(inside) or not any(inside), mtx.name
            if all(inside):
                ops = [plan.op(o) for o in mtx.members]
                assert {o.matrix for o in ops} == {mtx.name} and {o.layout for o in ops} == {"col"}
                assert [o.row_offset for o in ops] == sorted(o.row_offset for o in ops)
                assert len({o.instance for o in ops}) == 1
    # a member made clear takes its whole matrix with it
    col = next(m for m in cplan.matrices if m.layout == "col" and all(o in names for o in m.members))
    ops = list(graph.ops)
    victim = col.members[0]
    i = next(k for k, op in enumerate(ops) if isinstance(op, CheapOp) and op.inputs and victim + ".z" in op.inputs)
    p = dict(ops[i].params["requant"], mult=(1 << 29) + 1)
    ops[i] = dataclasses.replace(ops[i], fn=requant_fn(p), params={"requant": p})
    plan2 = cut.CutPlan.from_graph(dataclasses.replace(graph, ops=ops), None, 9, cplan)
    assert not (set(col.members) & plan2.names())


def test_p4_from_a_verifiers_weights_and_the_int8_clamp():
    # review fixes: a K/Kpre verifier's public graph has no weights, so P4 is evaluated from the weights it holds;
    # a requant op whose clamp is wider than int8 is never cut (the proof's |L| < 2^29 + 2^16)
    graph = build_decoder(_TINY["gpt"], calib_tokens=8, seed=1)
    pub = graph.public()
    names = cut.CutPlan.from_graph(pub, None, 9).names()
    victim = sorted(names)[0]
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    assert cut.CutPlan.from_graph(pub, None, 9, weights=weights).names() == names
    op = next(o for o in graph.mat_ops if o.name == victim)
    weights[victim] = (op.weight, torch.full((op.n_rows,), cut.CLAIM_LIMIT, dtype=torch.int64))   # |z| >= 2^29
    assert victim not in cut.CutPlan.from_graph(pub, None, 9, weights=weights).names()
    ops = list(graph.ops)
    i = next(k for k, o in enumerate(ops) if isinstance(o, CheapOp) and o.params.get("requant", {}).get("kind") == "requant")
    p = dict(ops[i].params["requant"], lo=-1000)
    ops[i] = dataclasses.replace(ops[i], fn=requant_fn(p), params={"requant": p})
    assert ops[i].inputs[0][:-2] not in cut.CutPlan.from_graph(dataclasses.replace(graph, ops=ops), None, 9).names()

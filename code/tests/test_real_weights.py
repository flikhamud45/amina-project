"""pvi.models.hf_opt (the int8 OPT of the report, Section 4.1) on a tiny random OPT,
so no download is needed.

Also the real graph with its last block pruned (``build_opt_from_hf(..., prune_last=True)``, plan 2 F3a): the
weights and constants of the graph without pruning, the same next-token logits, the shapes of
``build_decoder(..., prune_last=True)``, accepted by v0 (modes C and Kpre) and V1 (the cut protocol), tampering
rejected, the byte accounting (T7) exact; and the F3a scripts (``v1_bytes.py --real --prune-last``, ``--codec-only``,
``perf.py --real``) on it, with the checkpoint and WikiText-2 loaders replaced by the tiny model and random windows.
"""

import dataclasses
import importlib.util
import json
import sys
from pathlib import Path

import pytest
import torch

transformers = pytest.importorskip("transformers")

from pvi.commit import cost as analytic                                            # noqa: E402
from pvi.v1 import protocol as cp                                  # noqa: E402
from pvi.v0 import protocol as proto                                   # noqa: E402
from pvi.models import decoder as transformer                                         # noqa: E402
from pvi.core.graph import CheapOp, MatOp                                # noqa: E402
from pvi.models.hf_opt import build_opt_from_hf, is_pruned, matches_benchmark_graph, real_logits  # noqa: E402
from pvi.models.decoder import DecoderConfig                           # noqa: E402

CODE = Path(__file__).resolve().parents[1]
_SMOOTH = dict(norm_gain=4, smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)
_TINY_CFG = DecoderConfig("tiny-opt", 32, 2, 4, 4, 8, 64, 97, mlp="relu", max_pos=64)
"""The ``build_decoder`` shapes of the tiny OPT below."""


def _tiny_model():
    torch.manual_seed(0)
    cfg = transformers.OPTConfig(vocab_size=97, hidden_size=32, num_hidden_layers=2, ffn_dim=64,
                                 num_attention_heads=4, max_position_embeddings=64,
                                 word_embed_proj_dim=32, do_layer_norm_before=True)
    return transformers.OPTForCausalLM(cfg).eval()


@pytest.fixture(scope="module")
def tiny_opt():
    model = _tiny_model()
    ids = torch.randint(3, 97, (1, 24))
    graph, info = build_opt_from_hf(model, ids, norm_gain=4)
    return model, graph, info, ids


def test_all_positions_equals_one_query_per_prefix(tiny_opt):
    model, graph, info, ids = tiny_opt
    every = real_logits(graph, info, ids, all_positions=True)[0]
    for t in (1, 7, ids.shape[1]):
        last = real_logits(graph, info, ids[:, :t])[0, -1]
        assert torch.equal(every[t - 1], last)


def test_graph_is_integer_and_tracks_the_float_model(tiny_opt):
    model, graph, info, ids = tiny_opt
    assert all(m.weight.dtype == torch.int8 for m in graph.ops if isinstance(m, MatOp))
    assert sum(isinstance(o, CheapOp) and o.note == "last position" for o in graph.ops) == 1
    with torch.no_grad():
        ref = model(ids).logits[0].double()
    got = real_logits(graph, info, ids, all_positions=True)[0]
    cos = torch.nn.functional.cosine_similarity(got, ref, dim=-1)
    assert cos.min() > 0.9, cos


def test_defence_accepts_honest_queries_on_real_weights(tiny_opt):
    from pvi.v0.protocol import Prover, Verifier, commit_graph, params_for, run_query

    _, graph, _, ids = tiny_opt
    params = params_for(20, len(graph.mat_ops))
    commitments = commit_graph(graph, params.rate)        # also checks the claim range bound
    prover = Prover(graph, commitments=commitments)
    verifier = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in commitments.items()})
    for seed in range(2):
        assert run_query(prover, verifier, ids[:, :16], seed=seed)["accepted"]


def test_smoothed_norm_gains_keep_the_graph_and_the_protocol(tiny_opt):
    # plan F2: SmoothQuant through the norms' per-channel gains -- the same ops and shapes, scalar requant
    # multipliers (requant_fn), an integer graph that tracks the float model, accepted by v0 and by V1
    from pvi.v0.protocol import Prover, Verifier, commit_graph, params_for, run_query

    model, base, _, ids = tiny_opt
    graph, info = build_opt_from_hf(model, ids, **_SMOOTH)
    assert [type(o) for o in graph.ops] == [type(o) for o in base.ops]
    assert all(a.weight.shape == b.weight.shape for a, b in zip(graph.mat_ops, base.mat_ops))
    gains = [o for o in graph.ops if isinstance(o, CheapOp) and o.note == "norm"]
    assert len({int(v) for v in gains[0].fn.consts["gain"]}) > 1             # a gain per channel
    assert all(isinstance(o.params["requant"]["mult"], int) for o in graph.ops
               if isinstance(o, CheapOp) and o.params.get("requant"))
    with torch.no_grad():
        ref = model(ids).logits[0].double()
    got = real_logits(graph, info, ids, all_positions=True)[0]
    assert torch.nn.functional.cosine_similarity(got, ref, dim=-1).min() > 0.9
    params = params_for(20, len(graph.mat_ops), cut=True)
    commitments = commit_graph(graph, params.rate)
    prover = Prover(graph, commitments=commitments)
    verifier = Verifier(graph.public(), params, "C", publics={k: c.public for k, c in commitments.items()})
    assert run_query(prover, verifier, ids[:, :16], seed=1)["accepted"]
    assert cp.run_cut_query(prover, verifier, ids[:, :16], seed=1)["accepted"]


# -- the pruned last block (plan 2 F3a) ------------------------------------------------------------------------------

def _without_slices(graph):
    """``graph``'s ops but the "last position" slices."""
    return [o for o in graph.ops if not (isinstance(o, CheapOp) and o.note == "last position")]


def _constants(op):
    """A cheap op's public constants: its parameters and every tensor or number it captured."""
    consts = [op.fn.consts[k] for k in sorted(op.fn.consts)]
    return op.params, [c for c in consts if torch.is_tensor(c) or isinstance(c, (int, float))]


def _same_constants(a, b) -> bool:
    (pa, ca), (pb, cb) = _constants(a), _constants(b)
    return pa == pb and len(ca) == len(cb) and all(
        torch.equal(x, y) if torch.is_tensor(x) else x == y for x, y in zip(ca, cb))


@pytest.mark.parametrize("smooth", [False, True])
def test_a_pruned_real_graph_has_the_constants_and_logits_of_the_whole_one(tiny_opt, smooth):
    """The last block at the last position: the same weights, multipliers and norm gains (the pruned block is
    calibrated on every position), the ops of ``build_decoder(..., prune_last=True)``, and bit-identical next-token
    logits on prefixes of every length; its q, o, fc1 and fc2 claim one column, k and v every column."""
    model, _, _, ids = tiny_opt
    kw = _SMOOTH if smooth else dict(norm_gain=4)
    whole, info_w = build_opt_from_hf(model, ids, **kw)
    pruned, info_p = build_opt_from_hf(model, ids, prune_last=True, **kw)
    assert is_pruned(pruned) and not is_pruned(whole)
    assert {k: v for k, v in info_p.items() if k != "prune_last"} == {k: v for k, v in info_w.items()
                                                                      if k != "prune_last"}
    assert info_p["prune_last"] and not info_w["prune_last"]
    a, b = _without_slices(whole), _without_slices(pruned)
    assert len(pruned.ops) == len(whole.ops) + 2 and [type(o) for o in a] == [type(o) for o in b]
    for x, y in zip(a, b):
        if isinstance(x, MatOp):
            assert torch.equal(x.weight, y.weight) and (x.bias is None) == (y.bias is None)
            assert x.bias is None or torch.equal(x.bias, y.bias)
        else:
            assert x.note == y.note and _same_constants(x, y), x.note
    # the shapes and op kinds of the benchmark's pruned decoder; only the LM head's folded bias differs
    for g in (whole, pruned):
        (problem,) = matches_benchmark_graph(g, _TINY_CFG)
        assert "bias" in problem and g.mat_ops[-1].name in problem
    assert matches_benchmark_graph(pruned, _TINY_CFG, prune_last=False)[0].startswith("op count")
    gen = torch.Generator().manual_seed(5)
    for t in (1, 2, 9, 24):
        for x in (ids[:, :t], torch.randint(3, 97, (1, t), generator=gen)):
            (env_w, claims_w), (env_p, claims_p) = whole.forward(x), pruned.forward(x)
            assert torch.equal(env_w[whole.output_name], env_p[pruned.output_name])
            assert torch.equal(real_logits(whole, info_w, x), real_logits(pruned, info_p, x))
            cols = analytic.decoder_claim_columns(_TINY_CFG, t, prune_last=True)
            shapes = analytic.decoder_shapes(_TINY_CFG, prune_last=True)
            assert [z.shape for z in claims_p.values()] == [(s.n_rows, cols[s.name]) for s in shapes]
            assert pruned.claim_columns(x) == list(cols.values())
            assert (sum(z.numel() for z in claims_p.values()) < sum(z.numel() for z in claims_w.values())) == (t > 1)


def test_a_pruned_real_graph_gives_every_position(tiny_opt):
    """``real_logits(all_positions=True)`` on a pruned graph skips the slices of its last block too: the logits
    at position t are those of the pruned graph on the prefix 0..t, and those of the graph without pruning."""
    model, _, _, ids = tiny_opt
    whole, info_w = build_opt_from_hf(model, ids, **_SMOOTH)
    pruned, info_p = build_opt_from_hf(model, ids, prune_last=True, **_SMOOTH)
    every = real_logits(pruned, info_p, ids, all_positions=True)[0]
    assert every.shape[0] == ids.shape[1]
    assert torch.equal(every, real_logits(whole, info_w, ids, all_positions=True)[0])
    for t in (1, 5, ids.shape[1]):
        assert torch.equal(every[t - 1], real_logits(pruned, info_p, ids[:, :t])[0, -1])


_PRUNED: dict = {}


def _pruned_setup(model, ids, mode: str):
    """The paper's smoothed graph with its last block pruned, a prover and a verifier for ``mode`` (C: the auto
    commitment plan, with lookup tables and col matrices; Kpre: precomputed secret rows), cut-ready parameters."""
    if mode in _PRUNED:
        return _PRUNED[mode]
    graph, _ = build_opt_from_hf(model, ids, prune_last=True, **_SMOOTH)
    if mode == "C":
        coms = proto.commit_graph(graph, 4, policy="auto")
        params = proto.params_for(20, len(graph.mat_ops), plan=coms.plan, cut=True)
        v = proto.Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                           tables=coms.table_publics)
        prover = proto.Prover(graph, commitments=coms)
    else:
        params = proto.params_for(20, len(graph.mat_ops), cut=True)
        weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
        v = proto.Verifier(graph.public(), params, mode, weights=weights)
        v.precompute(proto.Challenger(seed=0))
        prover = proto.Prover(graph)
    _PRUNED[mode] = graph, prover, v
    return _PRUNED[mode]


def _bump(name: str, index: int, delta: int):
    """A trace-tampering edit: entry ``index`` (flat, modulo the claim's size) of op ``name``'s claim moved by
    ``delta``; everything after it is recomputed honestly."""
    def tamper(op, z):
        if op.name != name:
            return z
        z = z.clone()
        z.view(-1)[index % z.numel()] += delta
        return z
    return tamper


@pytest.mark.parametrize("mode", ["C", "Kpre"])
def test_a_pruned_real_graph_is_accepted_by_v0_and_v1_and_rejects_tampering(tiny_opt, mode):
    model, _, _, ids = tiny_opt
    graph, prover, v = _pruned_setup(model, ids, mode)
    x = ids[:, :16]
    mats = graph.mat_ops
    assert graph.claim_columns(x)[-7:] == [1, 16, 16, 1, 1, 1, 1]       # the last block: q k v o fc1 fc2; the head
    assert proto.run_query(prover, v, x, seed=1)["accepted"]
    assert proto.run_query(prover, v, x, seed=2, wire=True)["accepted"]
    out = cp.run_cut_query(prover, v, x, seed=3)
    assert out["accepted"], out["rejected_at"]
    assert v._cut_plan is None
    coms_plan = getattr(prover.commitments, "plan", None)

    def caught_at(name):          # a col-layout matrix's claims are checked through its columns (chi' = I)
        col = coms_plan is not None and coms_plan.matrix_of(name).layout == "col"
        return "columns_code" if col else "freivalds"

    # v0: a claim of the pruned block (its out-projection's one column), the top logit, a k claim
    for name, index, delta in ((mats[-4].name, 5, 1), (mats[-1].name, 0, 1000), (mats[-6].name, 37, -3)):
        assert proto.run_query(prover, v, x, seed=4, forward_kwargs={"tamper": _bump(name, index, delta)})[
            "rejected_at"] == caught_at(name), name
    # V1: the trace-tampering attack on a cut op (another requantised value, consistent downstream), and on a
    # clear op of the pruned block (one column: not cut)
    plan = cp.cut_plan(v.graph, v.claim_columns(x), x.shape[1], v.publics if v.groups else None, v.params.cut_lmax,
                       v.weights or None)
    assert plan.ops and mats[-4].name not in plan.names()
    # the first cut op of each layout: a row op's values meet A [X; 1] through u in the final check, a col
    # matrix's fold y = A xbar is checked through its opened columns; Kpre's y disagrees with the secret rows
    firsts = {o.layout: o for o in reversed(plan.ops)}
    assert set(firsts) == ({"row", "col"} if mode == "C" else {"row"})
    for layout, o in firsts.items():
        got = cp.run_cut_query(prover, v, x, seed=5, forward_kwargs={"tamper": _bump(o.name, 41, 3 * o.widths[1])})
        assert got["rejected_at"] == ("kpre_y" if mode == "Kpre" else "cut_final" if layout == "row"
                                      else "columns_code"), (o.name, layout)
    got = cp.run_cut_query(prover, v, x, seed=6, forward_kwargs={"tamper": _bump(mats[-4].name, 3, 1)})
    assert got["rejected_at"] == caught_at(mats[-4].name)


@pytest.mark.parametrize("mode", ["C", "Kpre"])
def test_the_byte_accounting_is_exact_on_a_pruned_real_graph(tiny_opt, mode):
    """T7 on real weights: v0's claims (unencoded; on the wire their ``PVC3`` size is measured, not modelled), ``u``
    and opened columns are ``analytic.proof_bytes`` of the graph's weight ops, and V1's every message but the
    multiproofs is ``cut_proof_bytes``."""
    model, _, _, ids = tiny_opt
    graph, prover, v = _pruned_setup(model, ids, mode)
    x = ids[:, :16]
    cols = dict(zip((op.name for op in graph.mat_ops), graph.claim_columns(x)))
    plan = getattr(prover.commitments, "plan", None)
    for wire in (False, True):
        got = proto.run_query(prover, v, x, seed=7, wire=wire)
        assert got["accepted"]
        want = analytic.proof_bytes(graph.mat_ops, v.params, cols, plan=plan, mode=mode,
                                    wire_claims=got["bytes"]["claims"] if wire else None,
                                    table_ids=dict(prover._looked_up))
        keys = ("u", "columns") if wire else ("claims", "u", "columns")
        assert all(got["bytes"][k] == want[k] for k in keys), (got["bytes"], want)
        if mode == "C":
            assert 0.5 < got["bytes"]["paths"] / want["paths"] < 2
    real = cp.run_cut_query(prover, v, x)
    est = cp.cut_proof_bytes(prover, v, x, seed=3)
    for k in ("claims", "cut_exc", "cut_mult", "cut_gkr", "u", "columns"):
        assert est[k] == real["bytes"][k], k
    assert est["n_leaves"] == real["cut"]["n_leaves"]


# -- the F3a scripts on the tiny model ---------------------------------------------------------------------------------

def _script(rel: str, name: str):
    spec = importlib.util.spec_from_file_location(name, CODE / rel)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture
def fake_checkpoint(monkeypatch):
    """``9_quality/common``'s loaders replaced: ``load_opt`` gives the tiny OPT, ``load_windows`` random windows
    (the calibration windows from another stream), so the ``--real`` paths run without a download."""
    pytest.importorskip("pyarrow")
    folder = str(CODE / "experiments" / "9_quality")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    import common

    def windows(tok, parquet=None, *, seq=128, windows=16, calib_windows=1):
        g, gc = torch.Generator().manual_seed(seq), torch.Generator().manual_seed(1000 + seq)
        return ([torch.randint(3, 97, (1, seq), generator=g) for _ in range(windows)],
                [torch.randint(3, 97, (1, seq), generator=gc) for _ in range(calib_windows)])

    monkeypatch.setattr(common, "load_opt", lambda name: (_tiny_model(), None))
    monkeypatch.setattr(common, "load_windows", windows)
    monkeypatch.chdir(CODE)
    return common


def _lines(capsys) -> list[dict]:
    return [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]


def test_v1_bytes_real_pruned_checks_the_accounting_and_rejects_tampers(fake_checkpoint, monkeypatch, capsys):
    v1b = _script("experiments/8_v1/v1_bytes.py", "v1_bytes")
    monkeypatch.setitem(transformer.CONFIGS, "opt-tiny", dataclasses.replace(_TINY_CFG, name="opt-tiny"))
    monkeypatch.setattr(sys, "argv", ["v1_bytes.py", "--model", "opt-tiny", "--real", "tiny", "--smooth", "0.55",
                                      "--prune-last", "--seq", "16", "--calib-seq", "24", "--modes", "C", "Kpre",
                                      "--policy", "auto", "--fs", "--full", "--reps", "2", "--tampers", "2",
                                      "--device", "cpu", "--threads", str(torch.get_num_threads())])
    assert v1b.main() == 0
    rows = _lines(capsys)
    assert [r["mode"] for r in rows] == ["C", "Kpre"]
    for r in rows:
        assert r["prune_last"] and r["real"] == "tiny" and r["v0_accepted"] and r["v1_full"]["accepted"]
        assert r["accounting_ok"] and r["shapes"] == {"problems": [], "claim_columns": True}
        assert set(r["accounting"]["v0"]) == {"u", "columns", "paths_ratio"}
        assert r["accounting"]["v0"]["u"] and r["accounting"]["v0"]["columns"]
        assert all(r["accounting"]["v1"][k] for k in ("claims", "cut_exc", "cut_mult", "cut_gkr", "u", "columns"))
        assert r["timed"]["v0"]["accepted"] == r["timed"]["v1"]["accepted"] == 2
        assert {k: (t["attempts"], t["rejected"]) for k, t in r["tampers"].items()} == {"trace": (2, 2),
                                                                                         "single": (2, 2)}
        assert "host_peak_rss" in r["peak"]


def test_v1_bytes_fails_when_the_accounting_disagrees(fake_checkpoint, monkeypatch, capsys):
    """The accounting check is not vacuous: a byte model off by one ``u`` byte fails the line and the exit status."""
    v1b = _script("experiments/8_v1/v1_bytes.py", "v1_bytes")
    monkeypatch.setitem(transformer.CONFIGS, "opt-tiny", dataclasses.replace(_TINY_CFG, name="opt-tiny"))
    honest = analytic.proof_bytes
    monkeypatch.setattr(analytic, "proof_bytes", lambda *a, **kw: dict(honest(*a, **kw), u=honest(*a, **kw)["u"] + 1))
    monkeypatch.setattr(sys, "argv", ["v1_bytes.py", "--model", "opt-tiny", "--real", "tiny", "--prune-last",
                                      "--seq", "16", "--modes", "Kpre", "--device", "cpu",
                                      "--threads", str(torch.get_num_threads())])
    assert v1b.main() == 1
    (r,) = _lines(capsys)
    assert r["v0_accepted"] and not r["accounting"]["v0"]["u"] and not r["accounting_ok"]


def test_v1_bytes_codec_only_profiles_every_block(fake_checkpoint, monkeypatch, capsys):
    v1b = _script("experiments/8_v1/v1_bytes.py", "v1_bytes")
    monkeypatch.setitem(transformer.CONFIGS, "opt-tiny", dataclasses.replace(_TINY_CFG, name="opt-tiny"))
    monkeypatch.setattr(sys, "argv", ["v1_bytes.py", "--model", "opt-tiny", "--real", "tiny", "--smooth", "0.55",
                                      "--prune-last", "--seq", "16", "--codec-only", "--device", "cpu",
                                      "--threads", str(torch.get_num_threads())])
    assert v1b.main() == 0
    (r,) = _lines(capsys)
    assert r["codec_only"] and [b["block"] for b in r["blocks"]] == ["embed", 0, 1, "head"]
    cols = analytic.decoder_claim_columns(_TINY_CFG, 16, prune_last=True)
    rows = {s.name: s.n_rows for s in analytic.decoder_shapes(_TINY_CFG, prune_last=True)}
    assert r["total"]["claims"] == sum(rows[k] * c for k, c in cols.items())
    assert r["blocks"][2]["claims"] == sum(rows[k] * cols[k] for k in ("q1", "k1", "v1", "o1", "fc11", "fc21"))
    assert 0 < r["total"]["v1_values"] < r["total"]["v0_claims"] < r["total"]["raw"]
    # the same graph, query and parameters as the byte line of the protocol (Kpre): V1's values and exceptions
    graph, _ = v1b.real_graph(_tiny_model(), fake_checkpoint.load_windows(None, seq=16, windows=0)[1][0], 0.55, True)
    assert r["graph_digest"] == graph.digest().hex()


def test_perf_real_pruned_times_the_queries_and_rejects_tampers(fake_checkpoint, monkeypatch, capsys):
    folder = str(CODE / "experiments" / "6_improvements")
    if folder not in sys.path:
        sys.path.insert(0, folder)
    perf = _script("experiments/6_improvements/perf.py", "perf")
    monkeypatch.setitem(transformer.CONFIGS, "opt-tiny", dataclasses.replace(_TINY_CFG, name="opt-tiny"))
    monkeypatch.setattr(sys, "argv", ["perf.py", "--model", "opt-tiny", "--real", "tiny", "--smooth", "0.55",
                                      "--calib-seq", "24", "--seq", "16", "--modes", "Kpre", "C", "--policy", "auto",
                                      "--queries", "2", "--wire", "on", "--prune-last", "on", "--tampers", "1",
                                      "--device", "cpu", "--threads", str(torch.get_num_threads())])
    assert perf.main() == 0
    rows = _lines(capsys)
    assert [r["mode"] for r in rows] == ["Kpre", "C"]
    for r in rows:
        assert r["real"] == "tiny" and r["prune_last"] and (r["accepted"], r["queries"]) == (2, 2)
        assert {k: (t["attempts"], t["rejected"]) for k, t in r["tampers"].items()} == {
            "single_value": (1, 1), "output_logit": (1, 1), "last_hidden": (1, 1)}
        assert r["peak"]["host_peak_rss"] is None or r["peak"]["host_peak_rss"] > 0

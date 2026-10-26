"""The attribution study of the integer OPT (``experiments.quality.attribution``) and its floating-point copy of the
integer graph (``experiments.quality.hybrid_opt``) on tiny random OPTs, so no download is needed.

* The copy's invariants, for the scalar graph with 8-bit probabilities (the study's graph), the certified recipe's
  (smoothed norm gains, 11-bit probabilities) and the legacy recipe's (percentile clipping): every op has its role,
  the checkpoint's weights, folded and quantised as the builder does, are the graph's int8 weights, biases and real
  scales, the all-integer copy gives the graph's logits at every position exactly, the all-float copy the
  checkpoint's, and the ReLU table alone adds nothing.
* The builder's real scales (``info["scales"]``) are those its multipliers were computed from; a changed weight is
  caught; graphs the copy does not reproduce (a pruned last block, per-row weight scales) are refused.
* The script end to end on a fake checkpoint (``test_hf_opt.fake_checkpoint``) with two windows: one record of kind
  ``attribution`` per row and the check record, and a failed check (``FAILED:``, exit status 1).
"""

import json
import sys

import pytest
import torch

transformers = pytest.importorskip("transformers")

from test_hf_opt import _run, fake_checkpoint  # noqa: E402,F401  (the fake checkpoint and the script runner)
from test_recipe_equivalence import CASES, _ids, tiny_opt  # noqa: E402

from experiments.paper import records as R  # noqa: E402
from experiments.paper import tables as T  # noqa: E402
from experiments.quality import attribution  # noqa: E402
from experiments.quality.hybrid_opt import FULL, GROUPS, SUBROWS, Hybrid, int_only_cfg, roles  # noqa: E402
from pvi.core.graph import CheapOp, MatOp  # noqa: E402
from pvi.models import hf_opt  # noqa: E402
from pvi.models.decoder import SHIFT  # noqa: E402

GRAPHS = ("gain4-p8", "certified", "legacy-p8")
"""The study's scalar graph, the certified recipe's and the legacy recipe's (``test_recipe_equivalence.CASES``)."""


@pytest.fixture(scope="module")
def setup():
    return tiny_opt(), _ids(1, (2, 48)), _ids(2, (1, 30))


def _build(setup, **kw):
    model, calib, _ = setup
    with torch.no_grad():
        return hf_opt.build_opt_from_hf(model, calib, **kw)


@pytest.mark.parametrize("case", GRAPHS)
def test_the_copy_reproduces_the_integer_graph_and_the_checkpoint(setup, case):
    model, _, x = setup
    graph, info = _build(setup, **CASES[case])
    hyb = Hybrid(model, graph, info)
    assert hyb.problems == []
    assert len(hyb.roles) == len(graph.ops)
    assert {c for _, c, _ in hyb.roles} - {None} == set(FULL) - {"att_out"}    # (the attention op holds att_out)
    with torch.no_grad():
        exact = hf_opt.real_logits(graph, info, x, all_positions=True)
        assert torch.equal(hyb.logits(x, {}), exact)                       # every component integer: the graph
        ref = model(x).logits.double()
        all_float = hyb.logits(x, int_only_cfg())
        assert float((all_float - ref).abs().max()) <= 1e-4 * (1 + float(ref.abs().max()))
        assert torch.equal(hyb.logits(x, int_only_cfg("relu")), all_float)  # the table is exact on int8 inputs
        # the LayerNorm outputs in float, the rest integer: no longer the graph's logits
        assert not torch.equal(hyb.logits(x, {"norm1": set(), "norm2": set(), "normf": set()}), exact)


def test_the_builders_scales_are_those_of_its_multipliers(setup):
    for case in GRAPHS:
        graph, info = _build(setup, **CASES[case])
        scales = info["scales"]
        assert set(scales) == {op.output for op in graph.ops}
        for op in graph.ops:
            if isinstance(op, CheapOp) and "requant" in op.params:
                z = op.inputs[-1]
                assert op.params["requant"]["mult"] == round((1 << SHIFT) * scales[z] / scales[op.output]), op.name
        assert scales[graph.output_name] == info["logit_scale"]


def test_a_changed_weight_is_caught_and_unsupported_graphs_are_refused(setup):
    model = setup[0]
    graph, info = _build(setup, **CASES["gain4-p8"])
    op = [o for o in graph.ops if isinstance(o, MatOp)][3]                # the k projection of the first block
    op.weight[0, 0] = op.weight[0, 0] - 1 if op.weight[0, 0] > 0 else op.weight[0, 0] + 1
    problems = Hybrid(model, graph, info).problems
    assert len(problems) == 1 and problems[0].startswith(f"{op.name} (w_qkv, layer 0): the int8 weights differ")
    for kw in (dict(prune_last=True), dict(row_pow2=4)):
        g, i = _build(setup, **CASES["gain4-p8"], **kw)
        with pytest.raises(ValueError):
            Hybrid(model, g, i)
    with pytest.raises(ValueError, match="ops"):
        roles(graph, model.config.num_hidden_layers + 1)


def test_the_script_writes_the_rows_and_the_check(fake_checkpoint, monkeypatch, capsys):
    status, lines = _run(attribution, monkeypatch, capsys, "--model", "opt-tiny", "--seq", "16", "--windows", "2")
    assert status == 0
    assert all(R.kind_of(r) == "attribution" for r in lines)
    rows, check = lines[:-1], lines[-1]
    assert len(rows) == 2 + 2 * len(GROUPS) + 2 * len(SUBROWS) and check["part"] == "check"
    assert check["ok"] and check["all_int_exact"] and check["all_int_max_abs_dlogit"] == 0.0
    assert check["all_int_row_equal"] and check["weights_problems"] == []
    assert check["graph_report"]["norm_gain_min"] == check["graph_report"]["norm_gain_max"] == 4.0
    assert check["graph_report"]["claims_ok"]
    assert check["benchmark_differences"] == [f"{check['benchmark_differences'][0].split(':')[0]}: bias True vs "
                                              "benchmark False"]
    for r in lines:
        assert (r["graph"], r["graph_cell"], r["seq"], r["windows"], r["tokens"]) == ("scalar", "scalar-g4-p8", 16, 2,
                                                                                    30)
        assert r["opts"] == {"norm_gain": 4.0, "prob_max": 255} and r["calibration"] == "test-end"
    by = {(r["group"], r["mode"]): r for r in rows}
    assert by["all", "all_int"]["ppl"] == check["int_ppl"] and by["all", "all_int"]["features"] == {}
    assert set(by["all", "all_float"]["features"]) == set(FULL)
    assert by["norm", "leave_one_out"]["features"] == {"norm1": [], "norm2": [], "normf": []}
    assert by["norm_round", "restricted"]["features"] == {c: ["round", "stats"] for c in ("norm1", "norm2", "normf")}
    assert {m for _, m in by} == {"all_int", "all_float", "leave_one_out", "add_one", "add_one_feature", "restricted"}
    # the paper's selection names the script's groups (experiments.paper does not import the script)
    assert T.ATTRIBUTION_GROUPS == tuple(GROUPS) and {g for _, g in T.ATTRIBUTION_ROWS} <= set(GROUPS)
    assert {"norm_round", "norm_clip"} <= set(SUBROWS)
    status, lines = _run(attribution, monkeypatch, capsys, "--model", "opt-tiny", "--seq", "16", "--windows", "2",
                         "--skip-sub", "--graph", "certified", "--pbits", "11")
    assert status == 0 and len(lines) == 1 + 2 + 2 * len(GROUPS) and lines[-1]["ok"]
    assert lines[-1]["graph_cell"] == "smooth-certified-a0.6-p11"


def test_a_failed_check_stops_the_script(fake_checkpoint, monkeypatch, capsys):
    threads = torch.get_num_threads()
    monkeypatch.setattr(sys, "argv", ["attribution.py", "--model", "opt-tiny", "--seq", "16", "--windows", "2",
                                      "--float-tol", "0", "--device", "cpu", "--threads", str(threads)])
    try:
        status = attribution.main()
    finally:
        torch.set_num_threads(threads)
    out = capsys.readouterr().out.splitlines()
    lines = [json.loads(s) for s in out if s.startswith("{")]
    assert status == 1 and len(lines) == 1 and lines[0]["part"] == "check" and not lines[0]["ok"]
    assert out[-1].startswith("FAILED: the all-float copy is ")
    assert R.checks(R.Rec("quality", "q-opt125-attr", "1", "x.out", 1, "attribution", lines[0]))
    with pytest.raises(SystemExit, match="does not reproduce"):
        _run(attribution, monkeypatch, capsys, "--model", "opt-tiny", "--opts", '{"row_pow2": 4}')

"""The report's tables count the streaming verifier's time (``verify_total``), and only once.

``bench.py --verifier-impl stream`` records one ``verify_total`` per query instead of the
verify phases it overlaps; ``aggregate.py`` and ``paper_assets.py`` must carry it into the
tables and refuse a row that has both.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytest.importorskip("matplotlib")
pytestmark = pytest.mark.filterwarnings("ignore:.enablePackrat. deprecated")   # matplotlib's own pyparsing call
SCRIPTS = Path(__file__).resolve().parents[1] / "experiments" / "5_comparison"


def _script(name: str):
    spec = importlib.util.spec_from_file_location(f"comparison_{name}", SCRIPTS / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_verify_total_is_carried_into_the_tables():
    assert "verify_total" in _script("aggregate").ADDITIVE
    assert "verify_total" in _script("validate_extrapolation").METRICS


def test_every_timing_of_run_query_is_carried_into_the_tables():
    """Every timing ``run_query`` records -- modes C, K and Kpre, a ``c`` policy's lookup tables and a K
    verifier's own rows, with and without wire, batched and streaming -- is additive in aggregate.py,
    checked by validate_extrapolation.py and part of paper_assets.py's prover or verifier time (once:
    never both), and the batched verify phases the streaming verifier's ``verify_total`` covers are
    those ``OVERLAPPED`` names."""
    from pvi.fullcheck import protocol as proto
    from test_plans import _planned, _verifiers

    graph, x, coms = _planned("gpt", "R8c")
    weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
    verifiers = [*_verifiers(graph, coms, False), *_verifiers(graph, coms, True)]
    for mode in ("K", "Kpre"):
        for lookups in (False, True):
            for stream in (False, True):
                v = proto.Verifier(graph.public(), proto.params_for(40, len(graph.mat_ops)), mode, weights=weights,
                                   lookups=lookups, stream=stream)
                if mode == "Kpre":
                    v.precompute(proto.Challenger(seed=1))
                verifiers.append(v)
    keys = {False: set(), True: set()}                 # by stream
    for v in verifiers:
        for wire in (False, True):
            res = proto.run_query(proto.Prover(graph, commitments=coms), v, x, seed=None if v.params.fiat_shamir
                                  else 1, wire=wire)
            assert res["accepted"]
            keys[v.stream] |= set(res["timings"])
    emitted = keys[False] | keys[True]
    assert {"prove_lookups", "verify_lookups", "verify_total"} <= emitted
    pa = _script("paper_assets")
    assert emitted <= _script("aggregate").ADDITIVE and emitted <= _script("validate_extrapolation").METRICS
    assert emitted <= set(pa.PROVE) | set(pa.VERIFY) and set(pa.PROVE) & set(pa.VERIFY) == {"fs_hash"}
    batched_only = {k for k in keys[False] - keys[True] if k.startswith("verify_")}
    assert batched_only <= set(pa.OVERLAPPED) and not keys[True] & set(pa.OVERLAPPED)
    assert set(pa.OVERLAPPED) - batched_only == {"verify_upload"}    # (a GPU client's: none on the CPU)


def test_the_streaming_verifier_is_timed_once():
    pa = _script("paper_assets")
    row = {"prove_forward": (2.0, "measured", 7e6, "gpu"), "prove_fold": (0.5, "measured", 7e6, "gpu"),
           "fs_hash": (0.25, "measured", 7e6, "gpu"), "bytes_total": (1000.0, "measured", 7e6, "gpu")}
    stream = dict(row, verify_total=(3.0, "measured", 7e6, "gpu"))
    assert pa._llm_cost(stream) == (2.75, 3.25, 1000.0, 7e6)
    phases = dict(row, verify_derive=(1.0, "measured", 7e6, "gpu"), verify_products=(1.5, "measured", 7e6, "gpu"))
    assert pa._llm_cost(phases) == (2.75, 2.75, 1000.0, 7e6)
    with pytest.raises(SystemExit):
        pa._llm_cost(dict(phases, verify_total=(3.0, "measured", 7e6, "gpu")))
    m = pa.Measured.__new__(pa.Measured)             # a CNN cell of the summary table, without the files
    m.idx = {("cnn", "lenet5", "cell", k, "", "", "", "", ""): {"median": str(v)}
             for k, v in (("verify_total", 0.5), ("fs_hash", 0.25))}
    assert m.total("lenet5", "cell", pa.VERIFY) == 0.75
    m.idx[("cnn", "lenet5", "cell", "verify_products", "", "", "", "", "")] = {"median": "0.1"}
    with pytest.raises(SystemExit):
        m.total("lenet5", "cell", pa.VERIFY)


def test_table4_cells_share_one_unit():
    pa = _script("paper_assets")
    assert pa.pair(1.0117, 0.3747, "t") == r"1{,}012$\to$375\,ms"      # the smaller value's unit
    assert pa.pair(0.0043, 0.0053, "t") == r"4.3$\to$5.3\,ms"
    assert pa.pair(2.7, 1.48, "t") == r"2.7$\to$1.5\,s"
    assert pa.pair(131e3, 49.5e3, "b") == r"131$\to$49.5\,kB"
    assert pa.pair(11.586e9, 11.586e9, "b") == r"11.6\,GB"         # printed once when unchanged


def test_table5_follows_the_setting_rule():
    """Fiat-Shamir against the non-interactive systems where it is stored (else interactive and marked),
    the interactive protocol against zkLLM and Maverick, Maverick's non-linear replay in its verifier
    time, bold exactly where ours is better on all three costs, and theirs/ours printed as n-times or 1/n."""
    pa = _script("paper_assets")
    pa.OPT_TABLES = [pa.tables_dir("l40s_improved")]
    rows = {(s, w): (cells, interactive, pend) for s, w, cells, interactive, pend in pa.ratio_rows()}
    assert not rows[("zkCNN$^a$", "LeNet-5")][1] and not rows[("DeepProve$^c$", "GPT-2 (64)")][1]
    assert rows[(r"ZKTorch$^\ast$", "Llama-2-7B (1)")][1]               # no Fiat-Shamir Llama-2-7B cell
    assert ("zkGPT$^b$", "GPT-2 (64)") in rows                          # zkGPT states no prompt length
    assert ("zkCNN$^a$", "VGG-16") in rows and ("DeepProve$^c$", "GPT-2 (512)") in rows   # their caveats
    assert pa._marks(["e", "a"]) == "$^{a,e}$" and pa._marks([]) == ""
    assert pa._marks([pa.INTERACTIVE_MARK, "a"]) == r"$^{a,\ast}$"     # letters first, the asterisk last
    # the letters follow Table 5's reading order: the first mark of each kind a reader meets
    order = [m for s, *_ in pa.ratio_rows() for m in __import__("re").findall(r"\^\{?([a-z])", s)]
    assert sorted(dict.fromkeys(order)) == list(dict.fromkeys(order))
    assert all(i for (s, w), (c, i, _) in rows.items() if s.startswith(("zkLLM", "Maverick")))
    them, ours = rows[("Maverick$^d$", "Qwen3-4B (8)")][0][1]
    assert abs(them - (0.0871 + pa.MAVERICK_NONLINEAR_S)) < 1e-9
    bold = {k for k, (cells, _, _) in rows.items() if all(a and o and a / o >= 1 for a, o in cells)}
    assert bold == {("zkCNN$^a$", "LeNet-5"), ("DeepProve$^c$", "GPT-2 (64)")}
    # the 2,048-token rows need the second optimised run: pending, with the basic protocol standing in
    assert all(all(p) for (s, w), (c, i, p) in rows.items() if s == "zkLLM")
    assert not any(any(p) for (s, w), (c, i, p) in rows.items() if s.startswith(("zkCNN", "DeepProve", "Maverick")))
    assert pa.fac(844.0, 5.39)[0] == r"157$\times$" and pa.fac(0.0871, 0.155)[0] == "1/1.8"
    pa.RATIO_STYLE = "arrows"
    assert pa.fac(844.0, 5.39)[0] == r"157$\times$$\uparrow$" and pa.fac(0.0871, 0.155)[0] == r"1.8$\times$$\downarrow$"


def test_cells_are_matched_by_their_tag_set():
    pa = _script("paper_assets")
    assert pa.tagset("_wire_gpuv_prune_polauto") == pa.tagset("_gpuv_wire_polauto_prune")
    assert pa.variant_of({"polauto", "prune", "gpuv", "stream", "wire"}) == "_wire_gpuv_stream_prune_polauto"
    assert pa.variant_of({"thr1", "lookups", "prune", "wire"}) == "_thr1_wire_prune_lookups"
    assert pa.basic_tags({"wire", "gpuv", "stream", "prune", "polauto"}) == {"gpuv"}
    assert pa.basic_tags({"thr1", "wire", "prune", "lookups"}) == {"thr1"}
    # one definition: every optimised spec adds only OPT_TAGS (and pol*) to its basic counterpart
    for spec in list(pa.OPTIMISED.values()) + list(pa.OPTIMISED_SPECIAL.values()):
        extra = set(spec["tags"]) - pa.basic_tags(spec["tags"])
        assert extra <= pa.OPT_TAGS | {"polauto"}


def test_a_missing_optimised_cell_is_pending_with_the_basic_value():
    """Without the optimised runs every optimised number is pending and the basic protocol's value stands
    in; with l40s_improved the image models it covers are final, the 224-pixel ResNet stays pending,
    and Llama-2-7B's unpruned run stands in (acceptable, black) for the pruned definition."""
    pa = _script("paper_assets")
    pa.OPT_TABLES = []
    v, p = pa.opt_cnn("lenet5", "bytes")
    assert p and v == pa.basic_cnn("lenet5", "bytes")
    pa.OPT_TABLES = [pa.tables_dir("l40s_improved")]
    v, p = pa.opt_cnn("lenet5", "bytes", where=("t", "r", "c"))
    assert not p and abs(v - 49.5e3) < 1e3
    v, p = pa.opt_cnn("resnet18_224", "prove", where=("t", "r224", "c"))
    assert p and v == pa.basic_cnn("resnet18_224", "prove")
    v, p = pa.opt_llm("llama2-7b", 64, "bytes", where=("t", "l64", "c"))
    assert not p and v < 0.5 * pa.basic_llm("llama2-7b", 64, "bytes")
    # no stand-in for the known-weights image models (the released code's Kpre cells send the basic proof)
    # or for the GPU verifier at 64 tokens (GPT-2's stored GPU run is not the streaming one)
    v, p = pa.opt_cnn("resnet18_cifar", "bytes", "Kpre", where=("t", "kpre", "c"))
    assert p and v == pa.basic_cnn("resnet18_cifar", "bytes", "Kpre")
    v, p = pa.opt_llm("gpt2", 64, "gpu", where=("t", "g64", "c"))
    assert p and v == pa.basic_llm("gpt2", 64, "gpu")
    status = {r["row"]: r["status"] for r in pa.PENDING}
    assert status == {"r224": "pending", "l64": "acceptable", "kpre": "pending", "g64": "pending"}
    # at 2,048 tokens the stored _gpuv_stream cells send the basic proof: not the definition, so pending;
    # the placeholder is the basic run's, the prover from the CPU-verifier run (as Table 3 prints it)
    v, p = pa.opt_llm("llama2-7b", 2048, "prove")
    assert p and v == pa.llm_cost(("llama2-7b", 2048), {})[0]
    v, p = pa.opt_llm("llama2-13b", 2048, "gpu")      # no basic GPU-verifier run: no placeholder
    assert p and v is None


def test_pending_cells_are_red_in_the_tables():
    pa = _script("paper_assets")
    assert pa.red(r"27.4\,ms", True) == r"\pending{27.4\,ms}" and pa.red("27.4", False) == "27.4"
    assert pa.pair(17.6, 17.6, "t", pending=True) == r"17.6$\to$\pending{17.6}\,s"   # not printed once
    assert pa.pair(11.586e9, None, "b", pending=True) == r"11.6\,GB$\to$\pending{--}"


def test_later_optimised_runs_replace_earlier_cells(tmp_path):
    pa = _script("paper_assets")
    head = "suite,model,cell,metric,batch,attack,lam,stage,k,n,median,mean,prover_hw,verifier_hw\n"
    cell = "defence_C_int_lam128_rate4_wire_polauto"
    rows = {"a": [f"cnn,lenet5,{cell},bytes_total,,,,,,1,100,100,g,c x8 threads",
                  f"cnn,lenet5,{cell},prove_forward,,,,,,1,1,1,g,c x8 threads",
                  f"cnn,vgg16,{cell},bytes_total,,,,,,1,7,7,g,c x8 threads"],
            "b": [f"cnn,lenet5,{cell},bytes_total,,,,,,1,200,200,g,c x8 threads"]}
    dirs = []
    for name, lines in rows.items():
        d = tmp_path / f"tables_{name}"
        d.mkdir()
        (d / "measured_summary.csv").write_text(head + "\n".join(lines) + "\n", encoding="utf-8")
        dirs.append(d)
    M = pa.Measured(dirs)
    assert M.get("lenet5", cell, "bytes_total") == 200 and M.platform("lenet5", cell) == "b"
    assert M.get("lenet5", cell, "prove_forward") is None          # the whole cell is replaced
    assert M.get("vgg16", cell, "bytes_total") == 7
    assert M.find_cnn("lenet5", "C", "int", 128, {"polauto", "wire"}) == cell


def test_the_figure_checks_catch_small_overlapping_and_clipped_text():
    pa = _script("paper_assets")
    fig = pa.plt.figure(figsize=(2.0, 1.0))
    fig.text(0.1, 0.5, "small", fontsize=6)
    fig.text(0.1, 0.5, "on top")
    fig.text(0.98, 0.5, "clipped at the edge")
    bad = " ".join(pa.layout_problems(fig))
    assert "6.0 pt" in bad and "overlaps" in bad and "leaves the canvas" in bad
    pa.plt.close(fig)


def test_count_macros_refuse_an_accepted_attack():
    co = _script("count_outcomes")
    from collections import Counter
    c = {"honest": Counter({("cnn", 1): 3}), "attacks": Counter({("cnn", 1): 2}), "stages": Counter({"freivalds": 2}),
         "by_type": Counter(), "policy": Counter(), "partial": Counter(), "records": 5, "rejected": []}
    m = co.macros(c, "Opt")
    assert m["OptNHonest"] == 3 and m["OptNAttacks"] == 2 and "OptNAttacksPlans" not in m
    c["attacks"][("llm", 0)] += 1
    with pytest.raises(SystemExit):
        co.macros(c, "")


def test_count_macros_of_a_run_still_to_come_are_pending(tmp_path):
    co = _script("count_outcomes")
    from collections import Counter
    one = {"honest": Counter({("cnn", 1): 3}), "attacks": Counter({("cnn", 1): 2}), "stages": Counter({"freivalds": 2}),
           "by_type": Counter(), "policy": Counter(), "partial": Counter(), "records": 5, "rejected": []}
    c = co.merge([one, one])
    assert co.macros(c, "Opt")["OptNAttacks"] == 4
    out = tmp_path / "counts.tex"
    co.write_tex(out, "raw_a", co.macros(c, "Opt"), pending="raw_b")
    assert r"\newcommand{\OptNAttacks}{\pending{4}}" in out.read_text(encoding="utf-8")
    co.write_tex(out, "raw_a", {"OptNAttacks": 4973})
    assert r"\newcommand{\OptNAttacks}{4{,}973}" in out.read_text(encoding="utf-8")


def test_text_numbers_compares_and_knows_pending():
    tn = _script("text_numbers")
    text = tn.normalise(r"is proved in \pending{17.6}\,s instead of 17.6\,s, and \pendingclaim{by 2.0--4.2$\times$}"
                        r" and \pending{21--26}$\times$ and 49.5\,kB--\pending{12.1\,MB}")
    tok = lambda v, st, u="": tn.Tok([(v, st)], u)   # noqa: E731
    assert tn.judge(tok("17.6", tn.PENDING, "s"), text)[0] == "ok"
    assert tn.judge(tok("17.6", tn.FINAL, "s"), text)[0] == "ok"            # 'instead of 17.6 s'
    assert tn.judge(tn.Tok([("2.0", tn.FINAL), ("4.2", tn.FINAL)]), text)[0] == "ok"   # \pendingclaim: transparent
    assert tn.judge(tn.Tok([("21", tn.PENDING), ("26", tn.PENDING)]), text)[0] == "ok"
    assert tn.judge(tn.Tok([("21", tn.FINAL), ("26", tn.FINAL)]), text)[0] == "UNWRAP"
    assert tn.judge(tn.Tok([("2.0", tn.PENDING), ("4.2", tn.PENDING)]), "by 2.0–4.2×")[0] == "UNMARKED"
    assert tn.judge(tok("5.4", tn.FINAL, "s"), text)[0] == "MISMATCH"
    assert tn.judge(tok("5.4", tn.PENDING, "s"), text)[0] == "placeholder"   # a different red value: a warning
    # a pending endpoint left black is UNMARKED; an endpoint a final element attains may stay black
    assert tn.judge(tn.Tok([("49.5", tn.PENDING), ("12.1", tn.PENDING)], "MB", mid="kB"), text)[0] == "UNMARKED"
    assert tn.judge(tn.Tok([("49.5", tn.ACCEPTABLE), ("12.1", tn.PENDING)], "MB", mid="kB"), text)[0] == "ok"
    assert tn.judge(tn.Tok([("21", tn.ACCEPTABLE), ("26", tn.PENDING)]), text)[0] == "ok"   # ⟦21–26⟧ is split
    assert tn.judge(tn.Tok([("21", tn.PENDING), ("26", tn.PENDING)]), "by 21–⟦26⟧×")[0] == "UNMARKED"
    assert tn.judge(tok("1.7", tn.ACCEPTABLE, "s"), "in 1.7s and ⟦1.7⟧s")[0] == "ok"
    assert tn.x2(32334) == "32,000" and tn.x2(5195) == "5,200" and tn.x2(157.2) == "157"
    # rng(): in a pending range only the endpoints that pending elements alone supply must be red
    lo, mid_, hi = tn.V(3.8, tn.FINAL), tn.V(20.6, tn.FINAL), tn.V(27.4, tn.PENDING)
    (r,) = tn.rng([lo, mid_, hi])
    assert r.segs == [("3.8", tn.ACCEPTABLE), ("27", tn.PENDING)]
    (r,) = tn.rng([lo, mid_])
    assert r.segs == [("3.8", tn.FINAL), ("21", tn.FINAL)]
    a, b = tn.ends([tn.V(30.0, tn.PENDING), tn.V(32334.0, tn.FINAL)], tn.x2)
    assert a.segs == [("30", tn.PENDING)] and b.segs == [("32,000", tn.ACCEPTABLE)]
    assert tn.judge(a, "by ⟦30⟧× (zkLLM) to 32,000× (ZKML)")[0] == "ok"
    assert tn.judge(b, "by ⟦30⟧× (zkLLM) to 32,000× (ZKML)")[0] == "ok"


def test_counts_can_keep_the_definition_cells_only():
    co, pa = _script("count_outcomes"), _script("paper_assets")
    d = pa.definition_tagsets()
    assert co.cell_tags("tamper_C_int_lam40_T64_L32_wire_prune_polauto") == ("C", {"wire", "prune", "polauto"})
    assert co.cell_tags("commit_T64_L12") == (None, None)
    keep = lambda suite, cell: co._kept(suite, cell, (), d)   # noqa: E731
    assert keep("cnn", "defence_C_int_lam40_rate4_wire_polauto") and keep("cnn", "tamper_C_int_lam40_wire_polauto")
    assert keep("cnn", "defence_Kpre_int_lam128_rate4_wire") and not keep("cnn", "defence_C_int_lam40_rate4_wire")
    assert keep("llm", "defence_C_int_lam128_T2048_L32_wire_gpuv_stream_prune_polauto")
    assert keep("llm", "defence_Kpre_int_lam40_T8_L36_thr1_wire_prune_lookups")
    assert keep("llm", "defence_C_int_lam128_T64_L32_wire_batch_prune_polauto")
    assert not keep("llm", "defence_C_int_lam128_T64_L32_wire_polauto")          # a stand-in, not the definition
    assert not keep("llm", "defence_C_int_lam128_T2048_L32_gpuv_stream")         # the basic proof format

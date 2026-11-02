"""The absorption-checked prefix rule (``hf_llama.absorb_tokens``, ``--prefix absorb``) and its tooling, on synthetic
statistics and tiny random Llama models (no download):

* the greedy search starts from PrefixQuant's choice, keeps it when it absorbs, extends it by the next most frequent
  outlier token in front of the BOS until a candidate absorbs, stops at the longest length or when no outlier token is
  left (then keeping the candidate of the smallest ratio, recorded as not absorbed), and records every candidate;
* PrefixQuant's rule (``auto``, the default) gives what it gave before, and its record has no field of the new rule;
* a prefix the rule chose builds the same graph, digest included, as the same token ids given by hand;
* ``prefix_check.py`` measures the maxima ``_hidden_max`` measures, and prints every candidate and the rule's pick;
* the paper's tooling names the rule of the Llama-family records it reads; under the absorption rule, per
  checkpoint, a cell of PrefixQuant's rule whose prefix absorbs by its own record counts as the rule's, one whose
  prefix does not fails the text's claim when another cell of the checkpoint is the rule's and is a warning when all
  are PrefixQuant's; ``llama_prefix`` selects by the rule.
"""

from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments.common import checkpoints  # noqa: E402
from experiments.paper import bench as B  # noqa: E402
from experiments.paper import macros as M  # noqa: E402
from experiments.paper import records as R  # noqa: E402
from experiments.paper import tables as T  # noqa: E402
from experiments.quality import ppl_llama, prefix_check  # noqa: E402
from pvi.models import hf_llama as rl  # noqa: E402

BOS = 1


def _tiny(seed: int = 0):
    torch.manual_seed(seed)
    cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   rms_norm_eps=1e-6, tie_word_embeddings=False, initializer_range=0.15)
    model = transformers.LlamaForCausalLM(cfg)
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name:
                p.uniform_(0.5, 1.5)
    return model.eval()


def _statistics():
    """Windows ``[4, 10]`` and three blocks' token-wise maxima: the initial token is an outlier of every window in
    blocks 1 and 2, token 7 at its first occurrence in three windows (blocks 1 and 2), token 9 in one window of block 2.
    Outliers per window: 0, 1.75, 2.0, so ``o = 2``: PrefixQuant's prefix is ``[7, BOS]`` and 9 is the next outlier
    token."""
    g = torch.Generator().manual_seed(5)
    ids = torch.randint(10, 50, (4, 10), generator=g)
    ids[:3, 3], ids[0, 5] = 7, 9
    maxes = [torch.rand(4, 10, generator=g, dtype=torch.float64) + 1 for _ in range(3)]
    for m in maxes[1:]:
        m[:, 0], m[:3, 3] = 1000.0, 500.0
    maxes[2][0, 5] = 300.0
    return ids, maxes


def _fake_hidden(absorbing: int | None, calls: list):
    """A stand-in for ``_hidden_max``: the windows' maximum 100 (10 once the prefix holds ``absorbing``), the
    prefix's 50 plus its length (so that a longer prefix of no absorbing token has the smaller ratio)."""
    def hidden(model, calib, prefix=None):
        tokens = prefix[0].tolist()
        calls.append(tokens)
        return (10.0 if absorbing in tokens else 100.0), 50.0 + len(tokens)
    return hidden


@pytest.fixture
def stubbed(monkeypatch):
    ids, maxes = _statistics()
    monkeypatch.setattr(rl, "_down_maxes", lambda model, calib: [m.clone() for m in maxes])
    model = SimpleNamespace(config=SimpleNamespace(bos_token_id=BOS))
    return model, ids, maxes


# ------------------------------------------------------------------ the rule
def test_auto_is_unchanged_and_absorb_starts_from_it(stubbed):
    model, ids, maxes = stubbed
    auto, stats = rl.prefixquant_tokens(model, ids)
    assert auto == [7, BOS] and stats == {"rule": "prefixquant", "eta": 64.0, "o": 2,
                                          "outliers_per_block": [0.0, 1.75, 2.0], "outlier_tokens": [(7, 6), (9, 1)]}
    assert rl._prefixquant_select(maxes, ids, BOS, 64.0) == (auto, stats)


def test_absorb_extends_until_a_candidate_absorbs(stubbed, monkeypatch):
    model, ids, _ = stubbed
    calls: list = []
    monkeypatch.setattr(rl, "_hidden_max", _fake_hidden(9, calls))
    prefix, rec = rl.absorb_tokens(model, ids)
    assert prefix == [7, 9, BOS] and calls == [[7, BOS], [7, 9, BOS]]
    assert rec["rule"] == "absorb" and rec["auto_tokens"] == [7, BOS] and rec["absorbed"] and rec["stop"] == "absorbed"
    assert (rec["margin"], rec["max_len"]) == (rl.ABSORB_MARGIN, rl.ABSORB_MAX_LEN) == (1.0, 6)
    assert rec["o"] == 2 and rec["outlier_tokens"] == [(7, 6), (9, 1)]          # PrefixQuant's statistics kept
    assert rec["candidates"] == [
        {"tokens": [7, BOS], "window_max": 100.0, "prefix_max": 52.0, "ratio": 100 / 52, "absorbed": False},
        {"tokens": [7, 9, BOS], "window_max": 10.0, "prefix_max": 53.0, "ratio": 10 / 53, "absorbed": True}]
    json.dumps(rec)                                          # the record goes into the JSON lines as it is


def test_absorb_keeps_the_auto_prefix_that_absorbs(stubbed, monkeypatch):
    model, ids, _ = stubbed
    calls: list = []
    monkeypatch.setattr(rl, "_hidden_max", _fake_hidden(7, calls))
    prefix, rec = rl.absorb_tokens(model, ids)
    assert prefix == rl.prefixquant_tokens(model, ids)[0] == [7, BOS] and calls == [[7, BOS]]
    assert rec["absorbed"] and len(rec["candidates"]) == 1


def test_absorb_stops_at_the_longest_prefix_or_the_last_outlier_token(stubbed, monkeypatch):
    model, ids, _ = stubbed
    calls: list = []
    monkeypatch.setattr(rl, "_hidden_max", _fake_hidden(None, calls))
    prefix, rec = rl.absorb_tokens(model, ids, max_len=2)         # PrefixQuant's choice already has 2 tokens
    assert prefix == [7, BOS] and calls == [[7, BOS]] and not rec["absorbed"] and rec["stop"] == "max_len"
    calls.clear()
    prefix, rec = rl.absorb_tokens(model, ids, max_len=6)         # one outlier token left, none absorbs
    assert calls == [[7, BOS], [7, 9, BOS]] and rec["stop"] == "no outlier token left"
    assert prefix == [7, 9, BOS] and not rec["absorbed"]          # the smaller ratio, recorded as not absorbed
    assert [c["absorbed"] for c in rec["candidates"]] == [False, False]


def test_the_search_itself():
    measure = {(5, 1): (3.0, 2.0), (5, 8, 1): (2.5, 2.0), (5, 8, 4, 1): (1.5, 2.0), (5, 8, 4, 6, 1): (0.1, 9.0)}
    tried = []

    def m(tokens):
        tried.append(tuple(tokens))
        return measure[tuple(tokens)]

    assert rl._absorb_search(m, [5, 1], [8, 4, 6], margin=1.0, max_len=6)[0] == [5, 8, 4, 1]
    # a margin of 1.5: 1.5 x 1.5 > 2 does not absorb; the next candidate does
    out, rec = rl._absorb_search(m, [5, 1], [8, 4, 6], margin=1.5, max_len=6)
    assert out == [5, 8, 4, 6, 1] and rec["absorbed"] and len(rec["candidates"]) == 4
    out, rec = rl._absorb_search(m, [5, 1], [8, 4, 6], margin=1.5, max_len=4)
    assert out == [5, 8, 4, 1] and not rec["absorbed"] and rec["stop"] == "max_len"
    assert rl._absorb_search(lambda t: (1.0, 0.0), [1], [], margin=1.0, max_len=6)[1]["candidates"][0]["ratio"] is None
    for bad in (dict(margin=0.0, max_len=6), dict(margin=1.0, max_len=0)):
        with pytest.raises(ValueError):
            rl._absorb_search(m, [5, 1], [], **bad)


# ------------------------------------------------------------------ on a model: the records and the graph
def test_prefix_of_records_and_the_graph_of_an_absorbed_prefix():
    model = _tiny()
    g = torch.Generator().manual_seed(1)
    calib = torch.randint(0, 97, (2, 32), generator=g)
    tok = SimpleNamespace(decode=lambda ids: f"<{ids[0]}>")
    auto, a_rec = checkpoints.prefix_of("auto", model, calib, tok, margin=3.0, max_len=2)
    assert set(a_rec) == {"rule", "eta", "o", "outliers_per_block", "outlier_tokens", "spec", "text"}
    assert (auto, a_rec["rule"]) == (rl.prefixquant_tokens(model, calib)[0], "prefixquant")
    chosen, rec = checkpoints.prefix_of("absorb", model, calib, tok, margin=1.0, max_len=3)
    assert rec["spec"] == "absorb" and rec["rule"] == "absorb" and rec["auto_tokens"] == auto
    assert rec["candidates"][0]["tokens"] == auto and all("text" in c for c in rec["candidates"])
    assert rec["text"] == [f"<{i}>" for i in chosen] and len(chosen) <= 3
    first = rec["candidates"][0]
    assert (first["window_max"], first["prefix_max"]) == rl._hidden_max(model, calib, torch.tensor([auto]))
    # the same constants and digest as the same ids given by hand: the protocol is that of any fixed prefix
    given, _ = checkpoints.prefix_of(",".join(map(str, chosen)), model, calib)
    kw = dict(smooth=0.55, smooth_pct=99.9, pct=99.99, pct_att=99.99)
    g1, i1 = rl.build_llama_from_hf(model, calib, prefix_tokens=chosen, **kw)
    g2, _ = rl.build_llama_from_hf(model, calib, prefix_tokens=given, **kw)
    assert g1.digest() == g2.digest() and g1.meta["prefix_tokens"] == chosen
    assert rl.matches_benchmark_graph(g1, rl.config_of(model.config, "tiny")) == []
    assert i1["prefix_hidden_max"] == rl._hidden_max(model, calib, torch.tensor([chosen]))[1]


def test_the_scripts_take_the_rules_parameters():
    import argparse
    ap = argparse.ArgumentParser()
    ppl_llama.add_absorb_arguments(ap)
    assert ppl_llama.absorb_options(ap.parse_args([])) == {"margin": 1.0, "max_len": 6}
    assert ppl_llama.absorb_options(ap.parse_args(["--absorb-margin", "1.5", "--absorb-max-len", "4"])) == \
        {"margin": 1.5, "max_len": 4}
    assert ppl_llama.prefix_spec("absorb") == "absorb" and ppl_llama.prefix_spec("none") is None


# ------------------------------------------------------------------ prefix_check.py
def test_prefix_check_measures_what_the_rule_measures(monkeypatch, capsys, tmp_path):
    model = _tiny()
    g = torch.Generator().manual_seed(2)
    calib = torch.randint(0, 97, (2, 24), generator=g)
    r = prefix_check.where_max(model, calib, [5, BOS], per_block=True)
    assert (r["window_max"], r["prefix_max"]) == pytest.approx(rl._hidden_max(model, calib, torch.tensor([[5, BOS]])))
    at = r["at"]
    assert len(r["blocks"]) == 3 and max(b["window_max"] for b in r["blocks"]) == r["window_max"]
    assert at["token"] == int(calib[at["window"], at["position"]])
    cands = prefix_check.candidates("meta-llama/Llama-2-7b-hf", BOS, [13, BOS],
                                    {"candidates": [{"tokens": [13, BOS]}, {"tokens": [13, 29889, BOS]}]}, True,
                                    ["Llama-2-7b-hf:29889,13,1", "other:4,1"])
    assert cands == [([13, BOS], "auto"), ([13, 29889, BOS], "absorb candidate 1"), ([BOS], "bos"),
                     ([29889, 13, BOS], "extra"), ([1, 13], "extra")]
    # the script end to end on a tiny model in place of the checkpoint
    tok = SimpleNamespace(decode=lambda ids: f"<{ids[0]}>")
    monkeypatch.setattr(prefix_check, "smoke_model", lambda name: (model, tok))
    monkeypatch.setattr(prefix_check, "wikitext", lambda split: "text")
    monkeypatch.setattr(prefix_check, "token_ids", lambda tok_, text, spec: torch.randint(2, 97, (200,), generator=g))
    out = tmp_path / "pc.jsonl"
    monkeypatch.setattr(sys, "argv", ["prefix_check", "--models", "tiny/model", "--smoke", "--seq", "24",
                                      "--calib-windows", "2", "--device", "cpu", "--threads", "1",
                                      "--prefixes", "model:5,1", "--out", str(out)])
    threads = torch.get_num_threads()
    try:
        prefix_check.main()
    finally:
        torch.set_num_threads(threads)
    text = capsys.readouterr().out
    assert text.startswith("model tiny/model") and "prefix [5, 1]" in text and "(given)" in text
    assert "absorb (margin 1, at most 6 tokens) picks" in text
    assert not any(line.startswith(("{", "FAILED")) for line in text.splitlines())   # no record in the job log
    rows = [json.loads(x) for x in out.read_text().splitlines()]
    assert rows[-1]["summary"] and rows[-1]["absorb_choice"]["rule"] == "absorb"
    assert {r["source"] for r in rows[:-1]} >= {"auto", "given"}
    assert all({"prefix_max", "window_max", "ratio", "at", "absorbed"} <= set(r) for r in rows[:-1])


# ------------------------------------------------------------------ the paper's tooling
def _llama(model, cell, ratio, spec, job, *, window=None, prefix=None, **kw):
    """A ``ppl_llama.py`` record of ``spec``'s prefix rule; ``window``, ``prefix``: the fp residual maxima over the
    windows and the prefix (``residual_real_max`` holds the former times the builder's headroom)."""
    d = {"model": model, "smoke": False, "config": "x", "split": kw.get("split", "test"), "seq": 2048,
         "windows": 160, "ref_dtype": "float32", "cell": cell, "graph": "main", "alpha": kw.get("alpha"),
         "glu_smooth": kw.get("balance"), "glu_exp": 0, "draw": 0, "ppl": 6.0 * ratio, "ref_ppl": 6.0,
         "ratio": ratio, "ratio_ci95": [ratio, ratio], "top1": 0.9, "prefix_spec": spec}
    if window is not None:
        d.update(residual_real_max=window * R.RESIDUAL_HEADROOM, prefix_hidden_max=prefix)
    if spec == "absorb":
        d["prefix_choice"] = {"rule": "absorb", "margin": kw.get("margin", 1.0)}
    return R.Rec("quality", f"q-x{job}", str(job), f"sp-q-x-{job}.out", 1, R.kind_of(d), d)


def _macros(recs, warnings=None, **run_kw):
    """``(printed macros, problems)`` of a build of ``recs``; ``warnings`` (a list): extended by the run's
    ``WARNING`` lines."""
    problems: list = []
    run = R.Run(recs, **run_kw)
    m, _ = M.build(T.Ctx(run, B.Bench(None), {}, byte_model=False), problems)
    if warnings is not None:
        warnings += [w for w in run.warnings if w.startswith("WARNING prefix")]
    return {k: x.value for k, x in m.items()}, problems


def test_the_paper_reads_one_prefix_rule():
    """Only the checkpoints whose prefix the absorption rule changes are measured again: a cell of PrefixQuant's rule
    whose prefix absorbs by its own record counts as the rule's; one whose prefix does not fails the text's claim
    once a cell of its checkpoint is the rule's; a checkpoint whose cells are all PrefixQuant's is accepted with a
    warning when its prefix does not absorb (also with ``llama_prefix="absorb"``).  The latest job's record is read,
    and ``llama_prefix="absorb"`` prefers the records that agree with the rule."""
    tl, l7 = "TinyLlama/TinyLlama_v1.1", "meta-llama/Llama-2-7b-hf"
    assert (R.prefix_rule({"prefix_spec": "auto"}), R.prefix_rule({"prefix_spec": None}),
            R.prefix_rule({"prefix_spec": "13,1"}), R.prefix_rule({"cell": "x"})) == ("auto", "none", "ids", None)
    assert R.ABSORB_MARGIN == rl.ABSORB_MARGIN
    tiny = _llama(tl, "main-a0.75-glus0.2", 1.10, "auto", 1, window=900.0, prefix=1000.0)       # absorbs (0.9)
    seven = _llama(l7, "main-a0.55-glus0.2", 265.0, "auto", 2, window=2461.0, prefix=1130.0)    # does not (2.18)
    assert R.prefix_absorbs(tiny.d) and not R.prefix_absorbs(seven.d) and R.prefix_absorbs({"cell": "x"}) is None
    assert not R.prefix_absorbs(tiny.d, 1.2) and R.absorb_consistent(tiny.d) and not R.absorb_consistent(seven.d)
    v, problems = _macros([tiny, seven])
    assert not problems and v["LlamaSevenRatio"] == "265.000"            # PrefixQuant's rule throughout: no check
    warned: list = []
    v, problems = _macros([tiny, seven], warned, llama_prefix="absorb")  # no cell of the rule: a warning only
    assert not problems and len(warned) == 1 and l7 in warned[0] and tl not in warned[0]
    # a cell of the rule next to a PrefixQuant cell of the same checkpoint that does not absorb fails
    mixed = [tiny, seven, _llama(l7, "main-a0.55-glus0.2", 1.08, "absorb", 3, split="validation", window=900.0,
                                 prefix=1500.0)]
    v, problems = _macros(mixed)
    assert len(problems) == 1 and problems[0].startswith(M.TEXT_CLAIM) and "does not absorb" in problems[0]
    assert l7 in problems[0] and tl not in problems[0]
    # Llama-2-7B measured again with the rule (a later job): read, and TinyLlama's absorbing cell agrees
    again = [tiny, seven, _llama(l7, "main-a0.55-glus0.2", 1.08, "absorb", 3, window=900.0, prefix=1500.0)]
    v, problems = _macros(again)
    assert not problems and v["LlamaSevenRatio"] == "1.080"
    assert {k: len(x) for k, x in M.llama_prefix_rules(R.Run(again)).items()} == {"auto": 1, "absorb": 1}
    # a margin of 1.2 (named by the rule's records, or given): TinyLlama's prefix no longer absorbs, a warning (its
    # cells are all PrefixQuant's)
    warned = []
    v, problems = _macros(again, warned, absorb_margin=1.2)
    assert not problems and len(warned) == 1 and tl in warned[0] and "margin 1.2" in warned[0]
    warned = []
    v, problems = _macros(again[:2] + [_llama(l7, "main-a0.55-glus0.2", 1.08, "absorb", 3, margin=1.2)], warned)
    assert not problems and len(warned) == 1 and tl in warned[0]
    # a later job of PrefixQuant's rule: read by default (every cell read is then PrefixQuant's), passed over with
    # llama_prefix="absorb"
    late = again + [_llama(l7, "main-a0.55-glus0.2", 265.0, "auto", 5, window=2461.0, prefix=1130.0)]
    v, problems = _macros(late)
    assert v["LlamaSevenRatio"] == "265.000" and not problems
    v, problems = _macros(late, llama_prefix="absorb")
    assert not problems and v["LlamaSevenRatio"] == "1.080"
    v, problems = _macros(late, llama_prefix="auto")
    assert not problems and v["LlamaSevenRatio"] == "265.000"


def test_the_rule_is_checked_per_checkpoint():
    """Llama-2-7B measured again with the absorption rule, Qwen3-4B left with PrefixQuant's prefix (the rule cannot
    absorb its massive activation within its longest prefix): the build passes, with a warning naming Qwen3-4B; a
    cell of the rule whose own search found no prefix that absorbs is a warning too."""
    l7, qw = "meta-llama/Llama-2-7b-hf", "Qwen/Qwen3-4B-Base"
    seven = _llama(l7, "main-a0.55-glus0.2", 1.08, "absorb", 3, window=900.0, prefix=1500.0)
    seven.d["prefix_choice"]["absorbed"] = True
    qwen = _llama(qw, "main-a0.45-glus0.5", 1.05, "auto", 2, window=8803.0, prefix=2300.0)
    assert R.prefix_absorbs(qwen.d) is False
    warned: list = []
    v, problems = _macros([seven, qwen], warned)
    assert not problems, problems
    assert v["LlamaSevenRatio"] == "1.080"
    assert len(warned) == 1 and qw in warned[0] and l7 not in warned[0]
    warned = []
    v, problems = _macros([seven, qwen], warned, llama_prefix="absorb")
    assert not problems and len(warned) == 1 and qw in warned[0]
    # the rule's own search ended without a prefix that absorbs: still the rule's cell, with a warning
    short = _llama(l7, "main-a0.55-glus0.2", 1.08, "absorb", 4, window=900.0, prefix=800.0)
    short.d["prefix_choice"]["absorbed"] = False
    warned = []
    v, problems = _macros([short, qwen], warned)
    assert not problems and len(warned) == 2
    assert any(l7 in w and "no prefix that absorbs" in w for w in warned)

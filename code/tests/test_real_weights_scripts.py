"""The real-weight option (``--real``) of the improvement scripts and of ``experiments.v1.v1_phases``
(:mod:`experiments.common.real_cells`), on a tiny random OPT and a tiny random Llama, with the checkpoint and
WikiText-2 loaders of ``experiments.common.checkpoints`` replaced (no download).

The refusals; the graph of a checkpoint's first blocks (the full graph's first blocks, op for op, the residual scale
set on the whole checkpoint, for OPT and Llama-family alike); the whole model's op shapes as a real graph has them; ``open_ab.py``, ``fold_ab.py`` and
``v1_phases.py`` end to end on the CPU, every variant on the same real graph and the same windows; and the
graph-building part of the GPU-only scripts (``forward_ab.py``, ``encode_ab.py``, ``gpu_profile.py``), which is what
their GPU paths run.
"""
from __future__ import annotations

import dataclasses
import json
import sys
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments.common import checkpoints, cli, real_cells  # noqa: E402
from pvi.models import decoder as transformer  # noqa: E402
from pvi.models import hf_llama  # noqa: E402
from pvi.models.decoder import DecoderConfig  # noqa: E402
from pvi.core.graph import MatOp  # noqa: E402
from pvi.models.hf_opt import CERTIFIED_RECIPE, build_opt_from_hf  # noqa: E402

OPT_CFG = DecoderConfig("opt-tiny", 32, 2, 4, 4, 8, 64, 97, mlp="relu", max_pos=64)
"""The shapes of the tiny OPT below (``--model opt-tiny``)."""
LLAMA_CFG = DecoderConfig("llama-tiny", 64, 2, 4, 2, 16, 160, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                          bias=False, tied=False, max_pos=64)
"""The shapes of the tiny Llama below (``--model llama-tiny``)."""
PREFIX = [5, 17]


def _tiny_opt(layers: int = 2):
    torch.manual_seed(0)
    cfg = transformers.OPTConfig(vocab_size=97, hidden_size=32, num_hidden_layers=layers, ffn_dim=64,
                                 num_attention_heads=4, max_position_embeddings=64, word_embed_proj_dim=32,
                                 do_layer_norm_before=True)
    return transformers.OPTForCausalLM(cfg).eval()


def _tiny_llama():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   rms_norm_eps=1e-6, tie_word_embeddings=False, initializer_range=0.15)
    return transformers.LlamaForCausalLM(cfg).eval()


def _windows(tok, parquet=None, *, seq=128, n=16, calib_windows=1):
    g, gc = torch.Generator().manual_seed(seq), torch.Generator().manual_seed(1000 + seq)
    return ([torch.randint(3, 97, (1, seq), generator=g) for _ in range(n)],
            [torch.randint(3, 97, (1, seq), generator=gc) for _ in range(calib_windows)])


@pytest.fixture
def fake_opt(monkeypatch):
    """The OPT loaders: ``load_opt`` the tiny OPT, ``load_windows`` random windows, ``certified_calibration`` two
    random 24-token windows."""
    monkeypatch.setattr(checkpoints, "is_opt", lambda name: True)
    monkeypatch.setattr(checkpoints, "load_opt", lambda name: (_tiny_opt(), None))
    monkeypatch.setattr(checkpoints, "load_windows", _windows)
    monkeypatch.setattr(checkpoints, "certified_calibration",
                        lambda tok: torch.randint(3, 97, (2, 24), generator=torch.Generator().manual_seed(7)))
    monkeypatch.setitem(transformer.CONFIGS, "opt-tiny", OPT_CFG)


@pytest.fixture
def fake_llama(monkeypatch):
    """The Llama loaders: ``load_llama`` the tiny Llama, the test split a random token stream, and the main recipe's
    calibration two random 24-token windows with the fixed prefix :data:`PREFIX`."""
    monkeypatch.setattr(checkpoints, "is_opt", lambda name: False)
    monkeypatch.setattr(checkpoints, "load_llama", lambda name: (_tiny_llama(), None))
    monkeypatch.setattr(checkpoints, "wikitext", lambda split, path=None, sep="\n\n": "")
    monkeypatch.setattr(checkpoints, "token_ids", lambda tok, text, spec: torch.randint(
        3, 97, (4000,), generator=torch.Generator().manual_seed(3)))
    monkeypatch.setattr(checkpoints, "llama_calibration", lambda model, tok, name="": SimpleNamespace(
        ids=torch.randint(3, 97, (2, 24), generator=torch.Generator().manual_seed(1)), prefix=PREFIX,
        prefix_info={}, recipe=hf_llama.MAIN_RECIPE))
    monkeypatch.setitem(transformer.CONFIGS, "llama-tiny", LLAMA_CFG)


def _args(**kw) -> SimpleNamespace:
    return SimpleNamespace(**{"real": "tiny", "recipe": "certified", "smooth": None, "calib_seq": 0, **kw})


def _run(script, monkeypatch, capsys, *argv) -> tuple[int, list[dict]]:
    """``script.main()`` with ``argv`` on the CPU: its exit status and JSON lines."""
    threads = torch.get_num_threads()
    monkeypatch.setattr(sys, "argv", ["script", *argv, "--device", "cpu", "--threads", str(threads)])
    try:
        status = script.main()
    finally:
        torch.set_num_threads(threads)
    return status, [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.startswith("{")]


# -- refusals ---------------------------------------------------------------------------------------------------------

@pytest.mark.parametrize("cfg, kw, says", [
    (transformer.CONFIGS["gpt2"], {}, "no integer builder"),
    (transformer.CONFIGS["opt-350m"], {}, "no integer builder"),
    (transformer.CONFIGS["llama2-7b"], {"recipe": "legacy", "calib_seq": 64}, "OPT recipe"),
    (transformer.CONFIGS["opt-1.3b"], {"smooth": 0.55}, "legacy recipe's"),
    (transformer.CONFIGS["opt-1.3b"], {"recipe": "legacy"}, "--calib-seq"),
    (transformer.CONFIGS["opt-1.3b"], {"layers": [25]}, "24 blocks"),
    (transformer.CONFIGS["opt-1.3b"], {"seqs": [4096]}, "trained context of 2048"),
    (transformer.CONFIGS["opt-6.7b"], {"layers": [33]}, "has 32 blocks"),
    (transformer.CONFIGS["llama2-7b"], {"seqs": [8192]}, "trained context of 4096"),
    (transformer.CONFIGS["llama2-7b"], {"prob_max": 255}, "--prob-max"),
])
def test_check_refuses_what_has_no_real_graph(cfg, kw, says):
    args = _args(**{k: v for k, v in kw.items() if k in ("recipe", "smooth", "calib_seq")})
    with pytest.raises(SystemExit, match=says):
        real_cells.check_args(args, cfg, layers=kw.get("layers", [0]), seqs=kw.get("seqs", [1024, 2048]),
                              prob_max=kw.get("prob_max"))


def test_check_passes_the_paper_cells_and_random_weights():
    for name, layers, seq in (("llama2-7b", [1, 4], 2048), ("llama2-13b", [1], 2048), ("opt-1.3b", [24], 2048),
                              ("opt-6.7b", [0], 2048), ("qwen3-4b", [2], 2048), ("llama2-7b", [2], 4096)):
        real_cells.check_args(_args(), transformer.CONFIGS[name], layers=layers, seqs=[seq], prob_max=2047)
    real_cells.check_args(_args(real=""), transformer.CONFIGS["gpt2"], layers=[2], seqs=[64], prob_max=255)
    with pytest.raises(SystemExit, match="pass --real"):
        real_cells.check_args(_args(real="", recipe="legacy"), transformer.CONFIGS["opt-1.3b"])


def test_load_refuses_a_checkpoint_of_other_shapes_or_a_longer_prompt(fake_opt, fake_llama, monkeypatch):
    with pytest.raises(SystemExit, match="vocab: 97 vs 32000"):     # a tiny Llama is not Llama-2-7B
        real_cells.load("tiny", transformer.CONFIGS["llama2-7b"], seq=16, n_queries=1)
    with pytest.raises(SystemExit, match="model type 'llama'"):
        real_cells.load("tiny", OPT_CFG, seq=16, n_queries=1)
    # 63 tokens fit the tiny Llama's 64 positions, but not after the fixed prefix's two
    with pytest.raises(SystemExit, match="behind its 2-token fixed prefix exceed its trained context of 64"):
        real_cells.load("tiny", LLAMA_CFG, seq=63, n_queries=1)


@pytest.mark.parametrize("module, argv", [
    ("experiments.improvements.open_ab", ["--model", "gpt2", "--real", "gpt2"]),
    ("experiments.improvements.forward_ab", ["--model", "gpt2", "--real", "gpt2"]),
    ("experiments.improvements.forward_ab", ["--model", "opt-1.3b", "--real", "facebook/opt-1.3b", "--seq", "64",
                                             "128"]),
    ("experiments.improvements.encode_ab", ["--model", "opt-1.3b", "--real", "facebook/opt-1.3b", "--seq", "4096"]),
    ("experiments.improvements.fold_ab", ["--model", "llama2-7b", "--real", "meta-llama/Llama-2-7b-hf",
                                          "--prob-max", "255"]),
    ("experiments.improvements.gpu_profile", ["--model", "gpt2", "--real", "gpt2"]),
    ("experiments.v1.v1_phases", ["--small", "256", "4", "64", "688", "512", "--real", "x"]),
    ("experiments.v1.v1_phases", ["--model", "llama2-7b", "--real", "x", "--layers", "33"]),
])
def test_the_scripts_refuse_before_loading_anything(module, argv, monkeypatch, capsys):
    import importlib
    monkeypatch.setattr(checkpoints, "load_real", lambda *a, **kw: pytest.fail("loaded a checkpoint"))
    with pytest.raises(SystemExit):
        _run(importlib.import_module(module), monkeypatch, capsys, *argv)


# -- the graph of the first blocks ------------------------------------------------------------------------------------

def test_first_blocks_of_a_llama_is_the_builders_n_layers(fake_llama):
    real = real_cells.load("tiny", LLAMA_CFG, seq=16, n_queries=2)
    got = real.graph(True, 1)[0]
    want, _ = hf_llama.build_llama_from_hf(real.model, real.calib.ids, prune_last=True, prefix_tokens=PREFIX,
                                           n_layers=1, **hf_llama.MAIN_RECIPE)
    assert got.digest() == want.digest()
    assert len(real.model.model.layers) == real.model.config.num_hidden_layers == 2      # put back
    assert real.graph(True, 0)[0].digest() == real.graph(True, 2)[0].digest()
    assert (real.recipe, real.prefix_len) == ("main", 2)


def _key(op):
    """An op's name and values (a weight op's weights and bias, a cheap op's parameters)."""
    if isinstance(op, MatOp):
        return op.name, op.weight.tolist(), None if op.bias is None else op.bias.tolist()
    return op.name, op.note, repr(op.params)


def test_first_blocks_of_an_opt_are_the_full_graphs_first_blocks(fake_opt, monkeypatch):
    big = _tiny_opt(layers=3)
    with torch.no_grad():                            # the second block's output sets the residual stream's scale
        big.model.decoder.layers[1].fc2.bias.fill_(1000.0)
    monkeypatch.setattr(checkpoints, "load_opt", lambda name: (big, None))
    monkeypatch.setitem(transformer.CONFIGS, "opt-tiny", dataclasses.replace(OPT_CFG, n_layers=3))
    real = real_cells.load("tiny", transformer.CONFIGS["opt-tiny"], seq=16, n_queries=2)
    full, _ = real.graph(False)
    part, info = real.graph(False, 1)
    head = 3                                         # the final norm, the last position and the LM head
    assert [_key(o) for o in part.ops[:-head]] == [_key(o) for o in full.ops[:len(part.ops) - head]]
    one = _tiny_opt(layers=1)                        # a checkpoint of the first block alone has another scale
    one.load_state_dict(big.state_dict(), strict=False)
    cut, cut_info = build_opt_from_hf(one, real.calib, prune_last=False, **CERTIFIED_RECIPE)
    assert cut_info["s_residual"] < info["s_residual"] and cut.digest() != part.digest()
    want, _ = build_opt_from_hf(big, real.calib, prune_last=True, n_layers=1, **CERTIFIED_RECIPE)
    assert real.graph(True, 1)[0].digest() == want.digest()
    assert real.recipe == "certified" and real.calib_seq is None


@pytest.mark.parametrize("family", ["opt", "llama"])
def test_model_ops_are_the_real_graphs_shapes(family, request):
    request.getfixturevalue("fake_" + family)
    cfg = OPT_CFG if family == "opt" else LLAMA_CFG
    real = real_cells.load("tiny", cfg, seq=16, n_queries=1)
    graph = real.graph(True, 0)[0]
    shape = lambda o: (o.n_rows, o.row_length, o.layout, o.has_bias)  # noqa: E731
    assert [shape(o) for o in graph.mat_ops] == [shape(s) for s in real_cells.model_ops(cfg, True, True)]
    if family == "opt":                              # the folded bias of the LM head
        assert real_cells.model_ops(cfg, True, True)[-1].row_length == cfg.d_model + 1
    else:
        assert real_cells.model_ops(cfg, True, True) == real_cells.model_ops(cfg, True, False)


# -- the scripts on the CPU -------------------------------------------------------------------------------------------

def _watch_queries(monkeypatch, module, name: str) -> list:
    """Record ``(graph digest, query)`` of every call of ``module.name`` (``run_query`` or ``run_cut_query``)."""
    seen, fn = [], getattr(module, name)

    def watched(prover, verifier, x, *a, **kw):
        seen.append((prover.graph.digest(), x.clone()))
        return fn(prover, verifier, x, *a, **kw)
    monkeypatch.setattr(module, name, watched)
    return seen


@pytest.mark.parametrize("family", ["opt", "llama"])
def test_open_ab_real_both_variants_on_the_same_graph_and_windows(family, request, monkeypatch, capsys):
    import pvi.v0.query as query
    from experiments.improvements import open_ab
    request.getfixturevalue("fake_" + family)
    model = "opt-tiny" if family == "opt" else "llama-tiny"
    seen = _watch_queries(monkeypatch, query, "run_query")
    status, rows = _run(open_ab, monkeypatch, capsys, "--model", model, "--real", "tiny", "--layers", "1", "2",
                        "--seq", "16", "--queries", "2", "--policy", "auto")
    assert status == 0
    assert [(r["layers"], r["variant"]) for r in rows] == [(1, "float64"), (1, "int8"), (2, "float64"), (2, "int8")]
    for r in rows:
        assert r["real"] == "tiny" and r["recipe"] == ("certified" if family == "opt" else "main")
        assert r["smooth"] is None and r["calib_seq"] is None and r["prob_max"] == 2047
        assert r["identical"] and r["accepted"] == r["queries"] == 2
    windows = real_cells.load("tiny", OPT_CFG if family == "opt" else LLAMA_CFG, seq=16, n_queries=3).queries
    assert len(seen) == 2 * 2 * 3                    # two block counts, two variants, a warm-up and two queries
    for block in (seen[:6], seen[6:]):
        assert len({d for d, _ in block}) == 1       # one graph per block count
        for i, w in enumerate(windows):              # both variants of query i on window i
            assert all(torch.equal(x, w) for _, x in block[2 * i:2 * i + 2])
    assert seen[0][0] != seen[6][0]


def test_open_ab_random_records_say_random(monkeypatch, capsys):
    from experiments.improvements import open_ab
    status, rows = _run(open_ab, monkeypatch, capsys, "--model", "tiny-opt", "--layers", "1", "--seq", "8",
                        "--queries", "1")
    assert status == 0
    assert all((r["real"], r["recipe"], r["smooth"], r["calib_seq"], r["prob_max"]) == (None, None, None, None, 2047)
               for r in rows)


def test_fold_ab_real_on_the_cpu(fake_llama, monkeypatch, capsys):
    from experiments.improvements import fold_ab
    status, rows = _run(fold_ab, monkeypatch, capsys, "--model", "llama-tiny", "--real", "tiny", "--layers", "1",
                        "--devices", "cpu", "--reps", "1")
    assert status == 0
    assert [r["path"] for r in rows] == ["float64", "default"]
    real = real_cells.load("tiny", LLAMA_CFG, seq=1, n_queries=0)
    graph = real.graph(True, 1)[0]
    for r in rows:
        assert (r["real"], r["recipe"], r["layers"], r["equal_to_float64"]) == ("tiny", "main", 1, True)
        assert r["matrices"] == len(graph.mat_ops)
        assert r["entries"] == sum(op.weight.numel() for op in graph.mat_ops)


def test_fold_ab_legacy_recipe_takes_its_window_from_calib_seq(fake_opt, monkeypatch, capsys):
    from experiments.improvements import fold_ab
    status, rows = _run(fold_ab, monkeypatch, capsys, "--model", "opt-tiny", "--real", "tiny", "--recipe", "legacy",
                        "--smooth", "0.55", "--calib-seq", "24", "--devices", "cpu", "--reps", "1")
    assert status == 0
    assert all((r["recipe"], r["smooth"], r["calib_seq"], r["layers"]) == ("legacy", 0.55, 24, 2) for r in rows)


def test_v1_phases_real_every_verifier_on_the_same_windows(fake_opt, monkeypatch, capsys):
    from experiments.v1 import v1_phases
    from pvi.v1 import protocol as cp
    seen = _watch_queries(monkeypatch, cp, "run_cut_query")
    status, rows = _run(v1_phases, monkeypatch, capsys, "--model", "opt-tiny", "--real", "tiny", "--layers", "1",
                        "--seq", "16", "--mode", "Kpre", "--reps", "2", "--lmax", "20",
                        "--v1-verifier", "optimised", "baseline")
    assert status == 0
    assert [r["path"] for r in rows] == ["optimised", "baseline"]
    for r in rows:
        assert (r["real"], r["recipe"], r["layers"], r["accepted"], r["queries"]) == ("tiny", "certified", 1, 2, 2)
    windows = real_cells.load("tiny", OPT_CFG, seq=16, n_queries=3).queries
    assert len(seen) == 6 and len({d for d, _ in seen}) == 1
    for i, w in enumerate(windows):                  # the warm-up, then the timed windows, under each verifier
        assert all(torch.equal(x, w) for _, x in seen[2 * i:2 * i + 2])


# -- the GPU scripts' graphs and queries ------------------------------------------------------------------------------

def test_forward_ab_setup_builds_every_block_count_from_one_load(fake_opt, monkeypatch):
    from experiments.improvements import forward_ab
    loads = []
    load = checkpoints.load_real
    monkeypatch.setattr(checkpoints, "load_real", lambda *a, **kw: loads.append(a) or load(*a, **kw))
    args = _args(seq=[16], queries=2, layers=[1, 2], prune_last=True)
    graphs, queries, real = forward_ab.setup(args, OPT_CFG)
    assert len(loads) == 1 and real.recipe == "certified"
    ref = real_cells.load("tiny", OPT_CFG, seq=16, n_queries=3)
    for n in (1, 2):
        assert graphs(n).digest() == ref.graph(True, n)[0].digest()
    assert all(torch.equal(a, b) for a, b in zip(queries(16), ref.queries)) and len(queries(16)) == 3


def test_encode_ab_and_gpu_profile_setups_are_the_real_graph(fake_llama):
    from experiments.improvements import encode_ab, gpu_profile
    ref = real_cells.load("tiny", LLAMA_CFG, seq=16, n_queries=2)
    want = ref.graph(True, 1)[0].digest()
    graph, x, real = encode_ab.setup(_args(seq=16, layers=1, prune_last=True), LLAMA_CFG)
    assert graph.digest() == want and torch.equal(x, ref.queries[0]) and real.recipe == "main"
    graph, xs, real = gpu_profile.setup(_args(seq=16, layers=1, prune_last=True), LLAMA_CFG)
    assert graph.digest() == want and len(xs) == 2 and all(torch.equal(a, b) for a, b in zip(xs, ref.queries))
    assert real_cells.recipe_fields(real, _args()) == {"real": "tiny", "recipe": "main", "smooth": None,
                                                  "calib_seq": None}


def test_gpu_scripts_with_random_weights_are_unchanged():
    from experiments.improvements import encode_ab, forward_ab, gpu_profile
    from pvi.models.decoder import build_decoder
    cfg = transformer.CONFIGS["gpt2"]
    cfg = dataclasses.replace(cfg, d_model=32, n_heads=2, n_kv_heads=2, head_dim=16, d_ff=64,
                              vocab=97, n_layers=2)
    common = dict(real="", seed=0, prompt_seed=1, prob_max=2047, prune_last=True)
    graph, x, real = encode_ab.setup(SimpleNamespace(seq=8, layers=1, **common), cfg)
    assert real is None and graph.digest() == build_decoder(cfg, n_layers=1, calib_tokens=cli.CALIB_TOKENS,
                                                            seed=0, prune_last=True, prob_max=2047).digest()
    assert torch.equal(x, torch.randint(0, 97, (1, 8), generator=torch.Generator().manual_seed(1)))
    graphs, queries, real = forward_ab.setup(SimpleNamespace(seq=[8], layers=[1], queries=2, **common), cfg)
    assert real is None and graphs(1).digest() == graph.digest() and len(queries(8)) == 3
    graph, xs, real = gpu_profile.setup(SimpleNamespace(seq=8, layers=1, **common), cfg)
    assert real is None and len(xs) == 3

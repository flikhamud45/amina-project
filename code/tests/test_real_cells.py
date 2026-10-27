"""The real-weight cells of ``costs/bench.py``, ``costs/setup_timing.py``, ``deployment/serve_ab.py``, ``costs/perf.py``,
``v1/v1_bytes.py`` and ``generation/gen_bench.py`` (``--real``, :mod:`experiments.common.real_cells`) on a tiny random OPT and a tiny random Llama built in this process: the loaders
of :mod:`experiments.common.checkpoints` are replaced, so nothing is downloaded.

Each script builds the checkpoint's integer graph as ``perf.py --real`` does (``checkpoints.real_graph``: the
certified OPT recipe, the main Llama recipe with its fixed prefix; a build of ``n`` blocks is the checkpoint's first
``n``), queries it on the test split's windows from its start, records ``real`` and ``recipe``, and every honest
query is accepted; the random-weight cells keep their names; and the settings that cannot be built or would be
mislabelled are refused before anything is loaded.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments.common import checkpoints, real_cells  # noqa: E402
from experiments.costs import bench, setup_timing  # noqa: E402
from experiments.deployment import serve_ab  # noqa: E402
from pvi.models import decoder as transformer  # noqa: E402
from pvi.models import hf_llama  # noqa: E402
from pvi.models.decoder import DecoderConfig  # noqa: E402

PREFIX = [5, 17]
"""The tiny Llama's fixed public prefix (the main recipe's, chosen by PrefixQuant's rule on a real checkpoint)."""
OPT_CFG = DecoderConfig("opt-tiny", 32, 2, 4, 4, 8, 64, 97, mlp="relu", max_pos=64)
LLAMA_CFG = DecoderConfig("llama-tiny", 64, 2, 4, 2, 16, 160, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                          bias=False, tied=False, max_pos=64)
"""The ``build_decoder`` shapes of the tiny checkpoints below (``--model``)."""
TEST = torch.randint(3, 97, (4096,), generator=torch.Generator().manual_seed(11))
"""The tiny Llama's test split, tokenised."""


def _tiny_opt():
    torch.manual_seed(0)
    cfg = transformers.OPTConfig(vocab_size=97, hidden_size=32, num_hidden_layers=2, ffn_dim=64,
                                 num_attention_heads=4, max_position_embeddings=64,
                                 word_embed_proj_dim=32, do_layer_norm_before=True)
    return transformers.OPTForCausalLM(cfg).eval()


def _tiny_llama():
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=64,
                                   rms_norm_eps=1e-6, tie_word_embeddings=False, initializer_range=0.15)
    return transformers.LlamaForCausalLM(cfg).eval()


def _llama_calibration(model, tok, n=4, seq=2048, name=""):
    calib = torch.randint(0, 97, (2, 32), generator=torch.Generator().manual_seed(1))
    return SimpleNamespace(ids=calib, prefix=PREFIX, prefix_info={}, recipe=hf_llama.main_recipe(name))


@pytest.fixture
def fake(monkeypatch):
    """Both tiny checkpoints behind :mod:`experiments.common.checkpoints`' loaders: ``tiny-opt`` (any name with
    ``opt``) and ``tiny-llama``; the shapes ``opt-tiny`` and ``llama-tiny`` in ``CONFIGS``; ``loads`` counts the
    checkpoint loads."""
    loads = []

    def windows(tok, parquet=None, *, seq=128, n=16, calib_windows=1):
        g, gc = torch.Generator().manual_seed(seq), torch.Generator().manual_seed(1000 + seq)
        return ([torch.randint(3, 97, (1, seq), generator=g) for _ in range(n)],
                [torch.randint(3, 97, (1, seq), generator=gc) for _ in range(calib_windows)])

    monkeypatch.setattr(checkpoints, "is_opt", lambda name: "opt" in name)
    monkeypatch.setattr(checkpoints, "load_opt", lambda name: (loads.append(name), (_tiny_opt(), None))[1])
    monkeypatch.setattr(checkpoints, "load_windows", windows)
    monkeypatch.setattr(checkpoints, "certified_calibration",
                        lambda tok: torch.randint(3, 97, (2, 24), generator=torch.Generator().manual_seed(7)))
    monkeypatch.setattr(checkpoints, "load_llama", lambda name: (loads.append(name), (_tiny_llama(), None))[1])
    monkeypatch.setattr(checkpoints, "wikitext", lambda split, path=None, sep="\n\n": "")
    monkeypatch.setattr(checkpoints, "token_ids", lambda tok, text, prefix_spec: TEST)
    monkeypatch.setattr(checkpoints, "llama_calibration", _llama_calibration)
    for cfg in (OPT_CFG, LLAMA_CFG):
        monkeypatch.setitem(transformer.CONFIGS, cfg.name, cfg)
    return loads


def _main(script, monkeypatch, capsys, *argv) -> tuple[int, list[dict]]:
    """``script.main()`` with ``argv`` on the CPU (this process's threads kept): its exit status and JSON lines."""
    threads = torch.get_num_threads()
    monkeypatch.setattr(sys, "argv", ["script", *argv, "--device", "cpu", "--threads", str(threads)])
    try:
        status = script.main()
    finally:
        torch.set_num_threads(threads)
    return status, [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]


# ------------------------------------------------------------------------------------------ the shared part
def test_the_prompts_are_perf_s_queries(fake):
    """The ``i``-th prompt of ``T`` tokens is the test split's ``i``-th window of ``T`` tokens (``load_real``'s
    queries), whatever longer prompts the job loaded the stream for; the graph is ``real_graph``'s."""
    ck = real_cells.load("tiny-llama", LLAMA_CFG, seq=32, n_queries=4)
    for seq, n in ((32, 4), (16, 8), (8, 3)):
        assert torch.equal(real_cells.prompts(ck, seq, n), torch.cat(checkpoints.windows(TEST, seq, n)))
    with pytest.raises(ValueError):
        real_cells.prompts(ck, 32, 5)
    assert ck.recipe == "main" and real_cells.fields(ck) == {"real": "tiny-llama", "recipe": "main"}
    assert real_cells.fields(None) == {"real": None, "recipe": None}
    graph, _ = ck.graph(True, 1)
    direct, _ = hf_llama.build_llama_from_hf(_tiny_llama(), ck.calib.ids, prune_last=True, prefix_tokens=PREFIX,
                                             n_layers=1, **hf_llama.main_recipe("tiny-llama"))
    assert graph.digest() == direct.digest()
    full, _ = ck.graph(True, 2)
    assert full.digest() == checkpoints.real_graph(_tiny_llama(), ck.calib, True)[0].digest() != graph.digest()
    ck.free()
    with pytest.raises(RuntimeError):
        ck.graph(True)


def test_a_checkpoint_of_other_shapes_or_context_is_refused(fake):
    with pytest.raises(SystemExit, match="model type 'opt'"):
        real_cells.load("tiny-opt", LLAMA_CFG, seq=16, n_queries=1)
    with pytest.raises(SystemExit, match="trained context of 64 positions"):
        real_cells.load("tiny-llama", LLAMA_CFG, seq=63, n_queries=1)     # 63 tokens behind a 2-token prefix
    with pytest.raises(SystemExit, match="random weights only"):
        real_cells.check(transformer.CONFIGS["gpt2"])
    with pytest.raises(SystemExit, match="trained context of 4096"):
        real_cells.check(transformer.CONFIGS["llama2-7b"], seqs=[4097])
    with pytest.raises(SystemExit, match="has 32 blocks"):
        real_cells.check(transformer.CONFIGS["opt-6.7b"], layers=[33])
    with pytest.raises(SystemExit, match="--prob-max"):
        real_cells.check(transformer.CONFIGS["qwen3-4b"], prob_max=255)
    assert real_cells.check(transformer.CONFIGS["qwen3-4b"], seqs=[2048], layers=[2]) == "llama"
    assert real_cells.check(transformer.CONFIGS["opt-6.7b"], seqs=[2048], layers=[2, 32]) == "opt"


# ------------------------------------------------------------------------------------------ setup_timing.py
@pytest.mark.parametrize("model,real,layers,recipe", [("opt-tiny", "tiny-opt", "0", "certified"),
                                                      ("opt-tiny", "tiny-opt", "1", "certified"),
                                                      ("llama-tiny", "tiny-llama", "1", "main")])
def test_setup_timing_real(fake, monkeypatch, capsys, model, real, layers, recipe):
    status, rows = _main(setup_timing, monkeypatch, capsys, "--model", model, "--real", real, "--layers", layers,
                         "--policy", "R8c", "--amortise", "10", "--reps", "1")
    assert status == 0 and fake == [real]
    (r,) = rows
    assert (r["real"], r["recipe"], r["accepted"], r["reps"]) == (real, recipe, True, 1)
    assert r["n_layers"] == (int(layers) or 2)


def test_setup_timing_random_records_name_no_checkpoint(fake, monkeypatch, capsys):
    status, (r,) = _main(setup_timing, monkeypatch, capsys, "--model", "llama-tiny", "--policy", "R8c",
                         "--amortise", "10", "--reps", "1")
    assert status == 0 and fake == [] and (r["real"], r["recipe"]) == (None, None)


@pytest.mark.parametrize("argv,match", [(["--model", "gpt2", "--real", "gpt2"], "random weights only"),
                                        (["--model", "opt-tiny", "--real", "tiny-opt", "--layers", "3"],
                                         "has 2 blocks")])
def test_setup_timing_refuses(fake, monkeypatch, capsys, argv, match):
    with pytest.raises(SystemExit, match=match):
        _main(setup_timing, monkeypatch, capsys, *argv, "--amortise", "10")
    assert fake == []


# ------------------------------------------------------------------------------------------ serve_ab.py
@pytest.mark.parametrize("model,real,layers,recipe", [("opt-tiny", "tiny-opt", "0", "certified"),
                                                      ("opt-tiny", "tiny-opt", "1", "certified"),
                                                      ("llama-tiny", "tiny-llama", "1", "main")])
def test_serve_ab_real(fake, monkeypatch, capsys, model, real, layers, recipe):
    seen = []
    cell = serve_ab.Cell.__init__

    def init(self, *a, **kw):
        cell(self, *a, **kw)
        seen.append(self.x)

    monkeypatch.setattr(serve_ab.Cell, "__init__", init)
    status, rows = _main(serve_ab, monkeypatch, capsys, "--model", model, "--real", real, "--layers", layers,
                         "--cells", "16x1", "16x2", "--batch-proof", "seq", "joint", "--policy", "per-op",
                         "--reps", "2", "--boot", "50", "--energy-s", "0")
    assert status == 0 and fake == [real]
    prove = [r for r in rows if r["row"] == "prove"]
    assert len(prove) == 2 * (1 + 2) and all(r["ok"] for r in prove)
    for r in rows:
        assert (r["real"], r["recipe"], r["n_layers"]) == (real, recipe, int(layers) or 2)
        assert r["load_s"] > 0 and r["build_s"] > 0
    stream = torch.cat(checkpoints.windows(TEST, 16, 2)) if recipe == "main" else None
    assert torch.equal(seen[1][:1], seen[0])                      # prompt 0 of every cell is the B = 1 prompt
    if stream is not None:
        assert torch.equal(seen[1], stream)


@pytest.mark.parametrize("argv,match", [
    (["--model", "gpt2", "--real", "gpt2"], "random weights only"),
    (["--model", "opt-tiny", "llama-tiny", "--real", "tiny-opt"], None),
    (["--model", "llama-tiny", "--real", "tiny-llama", "--cells", "65x1"], "trained context"),
    (["--model", "opt-tiny", "--real", "tiny-opt", "--layers", "3", "--cells", "16x1"], "has 2 blocks"),
    (["--model", "llama-tiny", "--real", "tiny-llama", "--prob-max", "255", "--cells", "16x1"], "--prob-max")])
def test_serve_ab_refuses(fake, monkeypatch, capsys, argv, match):
    with pytest.raises(SystemExit, match=match):
        _main(serve_ab, monkeypatch, capsys, *argv)
    assert fake == []


# ------------------------------------------------------------------------------------------ bench.py
ENV = {"device": "cpu", "cpu": "test", "torch_threads": 1}


def _bench(tmp_path, monkeypatch, *argv) -> dict:
    """``bench.py llm`` with ``argv`` into ``tmp_path``: ``{cell: rows}``."""
    monkeypatch.setattr(bench, "RAW", tmp_path)
    args = bench.build_parser().parse_args(["llm", *argv, "--lams", "40", "--queries", "2", "--policy", "R8c",
                                            "--platform", "test"])
    bench.check(args)
    bench.suite_llm(args, dict(ENV))
    return {p.stem: [json.loads(x) for x in p.read_text().splitlines()] for p in tmp_path.rglob("*.jsonl")}


@pytest.mark.parametrize("model,real,builds,recipe", [("opt-tiny", "tiny-opt", "full", "certified"),
                                                      ("opt-tiny", "tiny-opt", "1,full", "certified"),
                                                      ("llama-tiny", "tiny-llama", "1,full", "main")])
def test_bench_real_cells(fake, tmp_path, monkeypatch, model, real, builds, recipe):
    cells = _bench(tmp_path, monkeypatch, "--model", model, "--real", real, "--seq", "16", "--builds", builds,
                   "--modes", "C:fs,Kpre:fs", "--llm-tampers", "1", "--batches", "2")
    assert fake == [real]                                          # loaded once for every build
    blocks = [1, 2] if builds == "1,full" else [2]
    want = {f"{c}_T16_L{b}_wire_batch_prune_{tags}_real" for b in blocks
            for c, tags in (("commit", "pmax2047_polR8c"), ("defence_C_fs_lam40", "pmax2047_polR8c"),
                            ("defence_Kpre_fs_lam40", "lookups_pmax2047"))}
    assert set(cells) == want | {"tamper_C_int_lam40_T16_L2_wire_batch_prune_pmax2047_polR8c_real"}
    for cell, rows in cells.items():
        assert cell.endswith("_real")
        data = [r for r in rows if r["metric"] != "env"]
        assert all((r["config"]["real"], r["config"]["recipe"]) == (real, recipe) for r in data)
        assert all(r["value"] == 1 for r in data if r["metric"] in ("accepted", "tamper_rejected", "rejected"))
    commits = [rows for cell, rows in cells.items() if cell.startswith("commit_")]
    assert len(commits) == len(blocks)
    loads = [r for rows in commits for r in rows if r["metric"] == "load_checkpoint"]
    assert len(loads) == 1 and loads[0]["value"] > 0


def test_bench_random_cells_keep_their_names(fake, tmp_path, monkeypatch):
    cells = _bench(tmp_path, monkeypatch, "--model", "llama-tiny", "--seq", "16", "--builds", "full",
                   "--modes", "Kpre:fs")
    assert fake == [] and set(cells) == {"defence_Kpre_fs_lam40_T16_L2_wire_prune_lookups_pmax2047"}
    assert all("real" not in r["config"] for r in cells["defence_Kpre_fs_lam40_T16_L2_wire_prune_lookups_pmax2047"])


@pytest.mark.parametrize("argv,match", [
    (["--model", "gpt2", "--real", "gpt2", "--builds", "full"], "random weights only"),
    (["--model", "opt-tiny", "--real", "tiny-opt", "--builds", "3,full"], "has 2 blocks"),
    (["--model", "llama-tiny", "--real", "tiny-llama", "--seq", "60", "--gen", "8"], "trained context"),
    (["--model", "llama-tiny", "--real", "tiny-llama", "--prob-max", "255"], "--prob-max")])
def test_bench_refuses(fake, tmp_path, monkeypatch, argv, match):
    with pytest.raises(SystemExit, match=match):
        _bench(tmp_path, monkeypatch, *argv, "--modes", "Kpre:fs")
    assert fake == []
    with pytest.raises(SystemExit, match="decoders"):
        bench.check(bench.build_parser().parse_args(["cnn", "--model", "lenet5", "--real", "x", "--platform", "t"]))


# ------------------------------------------------------------------------- perf.py, v1_bytes.py, gen_bench.py
# ``--layers n`` with ``--real``: the checkpoint's first ``n`` blocks (OPT and Llama-family alike), with the full
# model's prompts and calibration.
@pytest.fixture
def load_calls(fake, monkeypatch):
    """The arguments of every ``checkpoints.load_real`` call (the prompts and calibration a script asks for)."""
    calls, load_real = [], checkpoints.load_real

    def spy(*a, **kw):
        calls.append((a, kw))
        return load_real(*a, **kw)

    monkeypatch.setattr(checkpoints, "load_real", spy)
    return calls


def _one_block(prune: bool = True):
    """The tiny Llama's first block under the main recipe, calibrated as its full model."""
    return checkpoints.real_graph(_tiny_llama(), _llama_calibration(None, None, name="tiny-llama"), prune,
                                  n_layers=1)[0]


def test_partial_takes_the_first_blocks_of_either_family():
    assert real_cells.partial(transformer.CONFIGS["qwen3-4b"], 0) is None
    assert real_cells.partial(transformer.CONFIGS["llama2-7b"], 32) is None
    assert real_cells.partial(transformer.CONFIGS["qwen3-4b"], 2) == 2
    assert real_cells.partial(transformer.CONFIGS["opt-6.7b"], 2) == 2
    with pytest.raises(SystemExit, match="has 32 blocks"):
        real_cells.partial(transformer.CONFIGS["llama2-7b"], 40)


def test_perf_real_first_blocks(load_calls, monkeypatch, capsys):
    from experiments.costs import perf
    argv = ["--model", "llama-tiny", "--real", "tiny-llama", "--seq", "16", "--modes", "Kpre", "C", "--policy", "R8c",
            "--queries", "2", "--tampers", "1"]
    status, rows = _main(perf, monkeypatch, capsys, *argv, "--layers", "1")
    assert status == 0 and [r["mode"] for r in rows] == ["Kpre", "C"]
    for r in rows:
        assert (r["real"], r["recipe"], r["layers"], r["seq"]) == ("tiny-llama", "main", 1, 16)
        assert (r["accepted"], r["queries"]) == (2, 2) and all(t["rejected"] == t["attempts"]
                                                               for t in r["tampers"].values())
        assert r["graph_digest"] == _one_block().digest().hex()
    status, full = _main(perf, monkeypatch, capsys, *argv)
    assert status == 0 and all(r["layers"] is None for r in full)
    assert full[0]["graph_digest"] != rows[0]["graph_digest"]
    assert load_calls[0] == load_calls[1]                    # the same prompts and calibration as the full model


def test_v1_bytes_real_first_blocks(load_calls, monkeypatch, capsys):
    from experiments.v1 import v1_bytes
    status, rows = _main(v1_bytes, monkeypatch, capsys, "--model", "llama-tiny", "--real", "tiny-llama", "--seq",
                         "16", "--layers", "1", "--modes", "C", "Kpre", "--policy", "R8c", "--full", "--reps", "2",
                         "--lmax", "20")
    assert status == 0 and [r["mode"] for r in rows] == ["C", "Kpre"]
    for r in rows:
        assert (r["real"], r["recipe"], r["layers"]) == ("tiny-llama", "main", 1)
        assert r["shapes"] == {"problems": [], "claim_columns": True} and r["accounting_ok"]
        assert r["v0_accepted"] and r["v1_full"]["accepted"]
        assert r["graph_digest"] == _one_block().digest().hex()
    status, (codec,) = _main(v1_bytes, monkeypatch, capsys, "--model", "llama-tiny", "--real", "tiny-llama", "--seq",
                             "16", "--layers", "1", "--codec-only")
    assert status == 0 and codec["layers"] == 1
    status, (whole,) = _main(v1_bytes, monkeypatch, capsys, "--model", "llama-tiny", "--real", "tiny-llama", "--seq",
                             "16", "--codec-only")
    assert status == 0 and whole["layers"] == 2 and load_calls[1] == load_calls[2]   # the full model's prompts


def test_gen_bench_real_first_blocks(load_calls, monkeypatch, capsys):
    from experiments.generation import gen_bench
    argv = ["--model", "llama-tiny", "--real", "tiny-llama", "--prompt", "8", "--gen", "3", "--decode", "greedy",
            "sample", "--modes", "C", "Kpre", "--policy", "R8c", "--queries", "1"]
    status, rows = _main(gen_bench, monkeypatch, capsys, *argv, "--layers", "1")
    assert status == 0 and len(rows) == 4
    for r in rows:
        assert (r["real"], r["layers"], r["n_layers"], r["ok"]) == ("tiny-llama", 1, 1, True)
    status, _ = _main(gen_bench, monkeypatch, capsys, *argv)
    assert status == 0 and load_calls[0] == load_calls[1]


@pytest.mark.parametrize("script,argv", [
    ("experiments.costs.perf", ["--seq", "16", "--modes", "Kpre", "C", "--policy", "R8c", "--queries", "1"]),
    ("experiments.v1.v1_bytes", ["--seq", "16", "--modes", "C", "--policy", "R8c", "--lmax", "20"]),
    ("experiments.generation.gen_bench", ["--prompt", "8", "--gen", "3", "--modes", "C", "--policy", "R8c",
                                          "--queries", "1"])])
def test_first_blocks_of_an_opt_checkpoint(load_calls, monkeypatch, capsys, script, argv):
    """An OPT checkpoint's first block: one load, the records of a one-block build, every query accepted."""
    import importlib
    status, rows = _main(importlib.import_module(script), monkeypatch, capsys, "--model", "opt-tiny", "--real",
                         "tiny-opt", "--layers", "1", *argv)
    assert status == 0 and rows and len(load_calls) == 1
    for r in rows:
        assert r["real"] == "tiny-opt" and r["layers"] == 1

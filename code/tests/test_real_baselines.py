"""The re-execution and decoding baselines on a real checkpoint (``--real``), on tiny in-process Hugging Face models
(no download: the loaders are stood in for):

* ``reexec.py --real``: the integer variants run the very graph of the verifier's real cells
  (``checkpoints.real_graph``: the main recipe with its prefix for a Llama-family checkpoint, the certified recipe for
  OPT), on the first ``T`` tokens of the checkpoint's first WikiText-2 window, the full model only; the floating-point
  variants run the checkpoint's own weights with the norms folded (on one token, where no position embedding acts,
  the checkpoint's logits); every record names the checkpoint and the integer variants' recipe, and a random-weight
  record names none;
* refused before anything is loaded: GPT-2 and every checkpoint without an integer graph, other shapes, several
  models, ``--layers`` short of the full model, ``--max-pos``, ``--prob-max``, a ``--seq`` beyond the checkpoint's
  trained context (and, after the calibration, beyond it with the recipe's prefix);
* ``decode_baseline.py --real``: the checkpoint decodes WikiText-2 windows of the real cells' tokenisation, and its
  records name it; another architecture, other shapes, ``--layers`` and a response beyond the context are refused.
"""
from __future__ import annotations

import json
import sys
from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from experiments.common import checkpoints, real_cells  # noqa: E402
from experiments.generation import decode_baseline as db  # noqa: E402
from experiments.reexec import reexec  # noqa: E402
from pvi import options  # noqa: E402
from pvi.models.decoder import prob_max_of  # noqa: E402

PREFIX = [5, 17]


def _llama(max_pos: int = 2048, eps: float = 1e-6):
    """A tiny ``LlamaForCausalLM`` of ``tiny-llama``'s shapes (random norm gains)."""
    torch.manual_seed(0)
    cfg = transformers.LlamaConfig(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                                   num_attention_heads=4, num_key_value_heads=4, max_position_embeddings=max_pos,
                                   rms_norm_eps=eps, tie_word_embeddings=False, initializer_range=0.15)
    model = transformers.LlamaForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5)
    return model


def _qwen():
    """A tiny ``Qwen3ForCausalLM`` of ``tiny-qwen``'s shapes."""
    torch.manual_seed(1)
    cfg = transformers.Qwen3Config(vocab_size=97, hidden_size=64, intermediate_size=160, num_hidden_layers=2,
                                   num_attention_heads=8, num_key_value_heads=2, head_dim=16, rope_theta=1e6,
                                   max_position_embeddings=512, tie_word_embeddings=True, max_window_layers=2,
                                   use_sliding_window=False, sliding_window=None, initializer_range=0.15)
    return transformers.Qwen3ForCausalLM(cfg).eval()


def _opt():
    """A tiny ``OPTForCausalLM`` of ``tiny-opt``'s shapes (random LayerNorm gains and biases)."""
    torch.manual_seed(2)
    cfg = transformers.OPTConfig(vocab_size=97, hidden_size=64, num_hidden_layers=2, ffn_dim=128,
                                 num_attention_heads=4, max_position_embeddings=256, word_embed_proj_dim=64,
                                 do_layer_norm_before=True, init_std=0.15)
    model = transformers.OPTForCausalLM(cfg).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "layer_norm" in n:
                p.uniform_(0.5, 1.5) if n.endswith("weight") else p.uniform_(-0.2, 0.2)
    return model


WINDOW = torch.randint(3, 97, (1, 40), generator=torch.Generator().manual_seed(4))
"""The stand-in checkpoint's first WikiText-2 window."""


def _stand_in(monkeypatch, model, calib, loaded=None):
    """``reexec``'s checkpoint loaders on ``model``: its configuration, and ``load_real``'s ``(model, [WINDOW],
    calib)`` (its calls noted in ``loaded``)."""
    monkeypatch.setattr(real_cells, "hf_config", lambda name: model.config)

    def load_real(name, seq, n_queries, calib_seq=0, recipe="certified"):
        if loaded is not None:
            loaded.append((name, seq, n_queries))
        return model, [WINDOW[:, :seq]], calib

    monkeypatch.setattr(checkpoints, "load_real", load_real)


def _main(monkeypatch, capsys, *argv) -> list[dict]:
    monkeypatch.setattr(sys, "argv", ["reexec.py", *argv])
    saved, threads = options.current(), torch.get_num_threads()
    try:
        assert reexec.main() == 0
    finally:
        options.configure(saved)
        torch.set_num_threads(threads)
    return [json.loads(s) for s in capsys.readouterr().out.splitlines() if s.startswith("{")]


def _llama_calib():
    return SimpleNamespace(ids=torch.randint(3, 97, (2, 32), generator=torch.Generator().manual_seed(1)),
                           prefix=PREFIX)


ARGS = ("--device", "cpu", "--threads", "1", "--reps", "1")


@pytest.mark.parametrize("family", ["llama", "opt"])
def test_the_integer_variants_run_the_real_cells_graph_on_their_window(family, monkeypatch, capsys):
    model, calib, name = ((_llama(), _llama_calib(), "tiny-llama") if family == "llama" else
                          (_opt(), torch.randint(3, 97, (2, 32), generator=torch.Generator().manual_seed(1)),
                           "tiny-opt"))
    loaded = []
    _stand_in(monkeypatch, model, calib, loaded)
    args = reexec.build_parser().parse_args(["--model", name, "--real", f"x/{family}", "--seq", "8", "16", *ARGS])
    got = reexec.load_checkpoint(args, reexec.check_real(args, reexec.decoder_config(name)), True, True)
    want, _ = checkpoints.real_graph(model, calib, True)
    assert got["graph"].digest() == want.digest() and loaded == [(f"x/{family}", 16, 1)]
    assert got["recipe"] == ("main" if family == "llama" else "certified") and got["prob_max"] == 2047
    assert torch.equal(got["prompt"], WINDOW[:, :16])
    recs = _main(monkeypatch, capsys, "--model", name, "--real", f"x/{family}", "--seq", "8", "16", "--variants",
                 "int", "int_fp32gemm", "float", "bf16", "dynint8", *ARGS)
    assert [(r["seq"], r["variant"]) for r in recs] == [(s, v) for s in (8, 16)
                                                       for v in ("int", "int_fp32gemm", "float", "bf16", "dynint8")]
    for r in recs:
        integer = r["variant"].startswith("int")
        assert r["real"] == f"x/{family}" and list(r["measured_s"]) == ["2"] and r["full_s"] == r["measured_s"]["2"]
        assert r["recipe"] == (got["recipe"] if integer else None)
        assert r["prob_max"] == (prob_max_of(want) if integer else None)
        assert r["max_pos"] == model.config.max_position_embeddings


def test_a_random_weight_record_names_no_checkpoint(monkeypatch, capsys):
    recs = _main(monkeypatch, capsys, "--model", "tiny-llama", "--seq", "8", "--variants", "int", "float", *ARGS)
    assert [(r["variant"], r["real"], r["recipe"], sorted(r["measured_s"])) for r in recs] == \
        [("int", None, None, ["1", "2"]), ("float", None, None, ["1", "2"])]


@pytest.mark.parametrize("family", ["llama", "opt", "qwen"])
def test_the_float_decoder_runs_the_checkpoints_weights(family):
    """The folded weights give the checkpoint's own logits on one token (no position embedding acts there: rotary
    position 0 is the identity, and OPT's learned positions are zeroed here); Qwen3's q/k norms are not in the float
    decoder, so only its shapes are checked."""
    model = {"llama": _llama, "opt": _opt, "qwen": _qwen}[family]()
    if family == "opt":
        with torch.no_grad():
            model.model.decoder.embed_positions.weight.zero_()
    cfg = reexec.decoder_config(f"tiny-{family}")
    weights = reexec.hf_float_weights(model)
    x = torch.tensor([[11]])
    for prune in (False, True):
        dec = reexec.FloatDecoder(cfg, cfg.n_layers, torch.float32, torch.device("cpu"), prune_last=prune,
                                  weights=weights)
        with torch.no_grad():
            got, want = dec(x), model(x).logits[:, -1]
        assert got.shape == want.shape == (1, 97)
        if family != "qwen":
            assert torch.allclose(got, want, atol=1e-4, rtol=1e-4), (got - want).abs().max()
    dyn = reexec.DynInt8Decoder(cfg, cfg.n_layers, torch.device("cpu"), prune_last=True, weights=weights)
    with torch.no_grad():
        assert dyn(torch.randint(0, 97, (1, 8))).shape == (1, 97)


@pytest.mark.parametrize("argv, match", [
    (("--model", "gpt2", "--real", "x/gpt2"), "no integer builder"),
    (("--model", "tiny-llama", "--real", "x/llama", "--layers", "1"), "full graph"),
    (("--model", "tiny-llama", "--real", "x/llama", "--max-pos", "4096"), "--max-pos"),
    (("--model", "tiny-llama", "--real", "x/llama", "--prob-max", "255"), "--prob-max"),
    (("--model", "tiny-llama", "tiny-qwen", "--real", "x/llama"), "one --model"),
    (("--model", "tiny-qwen", "--real", "x/llama"), "not of --model tiny-qwen's shapes"),
    (("--model", "tiny-llama", "--real", "x/llama", "--seq", "64", "65"), "trained context of 64"),
])
def test_real_refusals_come_before_anything_is_loaded(argv, match, monkeypatch, capsys):
    model = (transformers.GPT2LMHeadModel(transformers.GPT2Config(n_embd=64, n_layer=2, n_head=4, vocab_size=97))
             if "gpt2" in argv else _llama(max_pos=64))
    monkeypatch.setattr(real_cells, "hf_config", lambda name: model.config)
    monkeypatch.setattr(checkpoints, "load_real", lambda *a, **k: pytest.fail("loaded"))
    monkeypatch.setattr(sys, "argv", ["reexec.py", *argv, "--variants", "int", "float", *ARGS])
    with pytest.raises(SystemExit, match=match):
        reexec.main()


def test_a_prompt_beyond_the_context_with_the_recipes_prefix_is_refused(monkeypatch):
    model = _llama(max_pos=64)
    _stand_in(monkeypatch, model, _llama_calib())
    monkeypatch.setattr(checkpoints, "real_graph", lambda *a, **k: pytest.fail("built"))
    args = reexec.build_parser().parse_args(["--model", "tiny-llama", "--real", "x/llama", "--seq", "63", *ARGS])
    cfg = reexec.check_real(args, reexec.decoder_config("tiny-llama"))
    with pytest.raises(SystemExit, match="2-token fixed prefix"):
        reexec.load_checkpoint(args, cfg, True, False)


# ------------------------------------------------------------------ decode_baseline.py --real
class _Tok:
    """A stand-in tokenizer: one id per character (mod the vocabulary, above the special ids)."""

    def __call__(self, text, return_tensors="pt", add_special_tokens=True):
        ids = [3 + ord(c) % 94 for c in text]
        return SimpleNamespace(input_ids=torch.tensor([([1] if add_special_tokens else []) + ids]))


def test_decode_runs_the_checkpoint_on_wikitext_windows(monkeypatch, capsys):
    model = _llama(max_pos=64)
    text = "Some test text of WikiText, long enough for every window. " * 4
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda name, **k: model.config)
    monkeypatch.setattr(checkpoints, "load_llama", lambda name, dtype: (model.to(dtype), _Tok()))
    monkeypatch.setattr(checkpoints, "wikitext", lambda split, path=None, sep="\n\n": text)
    got, prompts = db.load("x/llama", torch.float32, torch.device("cpu"), 8, 6)
    ids = _Tok()(text, add_special_tokens=False).input_ids[0]
    assert torch.equal(prompts, ids[:48].view(6, 8)) and got.config._attn_implementation == "sdpa"
    threads = torch.get_num_threads()
    try:
        status = db.main(["--model", "tiny-llama", "--real", "x/llama", "--prompt", "8", "--gen", "2", "3",
                          "--reps", "2", "--device", "cpu", "--dtype", "fp32", "--threads", "1"])
    finally:
        torch.set_num_threads(threads)
    recs = [json.loads(s) for s in capsys.readouterr().out.splitlines() if s.startswith("{")]
    assert status == 0 and [(r["real"], r["gen"], r["ok"]) for r in recs] == [("x/llama", 2, True),
                                                                              ("x/llama", 3, True)]


@pytest.mark.parametrize("argv, match", [
    (("--model", "tiny-llama", "--real", "x/opt"), "model type 'opt'"),
    (("--model", "tiny-qwen", "--real", "x/llama"), "not of --model tiny-qwen's shapes"),
    (("--model", "tiny-llama", "--real", "x/llama", "--layers", "1"), "every block"),
    (("--model", "tiny-llama", "--real", "x/llama", "--prompt", "60", "--gen", "6"), "> its 64"),
])
def test_decode_refusals_come_before_the_weights_are_loaded(argv, match, monkeypatch):
    model = _opt() if "x/opt" in argv else _llama(max_pos=64)
    monkeypatch.setattr(transformers.AutoConfig, "from_pretrained", lambda name, **k: model.config)
    monkeypatch.setattr(checkpoints, "load_llama", lambda *a, **k: pytest.fail("loaded"))
    threads = torch.get_num_threads()
    try:
        with pytest.raises(SystemExit, match=match):
            db.main([*argv, "--device", "cpu", "--dtype", "fp32", "--threads", "1", "--reps", "1"])
    finally:
        torch.set_num_threads(threads)

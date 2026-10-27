"""``experiments/reexec/llamacpp.py``: the llama.cpp CPU baseline, without building llama.cpp.

* The tensor plan of ``llama2-7b`` has Llama-2-7B's tensor names and shapes and its 6.74e9 parameters (the weight
  products of ``pvi.models.decoder`` and the norms), and the vocabulary has Llama-2's layout;
* a tiny GGUF written by the script's writer (a test-only small Llama config) reads back with gguf-py's reader: its
  metadata, tensor types and shapes, bf16 values of the drawn spread and norms of one (skipped without a gguf-py:
  an installed ``gguf`` or a llama.cpp clone under ``logs/<platform>/llamacpp/src``);
* a stand-in for llama-bench that prints JSON lines among other output gives records of kind ``llamacpp`` with the
  median and range of its samples, marked inexact and a different model, which pass the paper's checks; a test other
  than the one asked for, or a failing llama-bench, is refused;
* llama-bench's command line: prompt processing only, the given threads, the weights in memory, no NUMA mode or CPU
  mask, micro-batches capped at the prompt;
* a model other than Llama-2's architecture is refused before anything is built;
* ``--real``, on a tiny ``LlamaForCausalLM`` saved here with a SentencePiece tokenizer trained here (fp16 in one file
  with Transformers' configuration, and bf16 shards with a configuration like Llama-2-7B's): the q and k permutation
  turns Transformers' rotary embedding into llama.cpp's; the written file holds the checkpoint's tensors (permuted
  where they must be, rounded to bf16), hyperparameters and vocabulary, and is byte for byte the release's own
  ``convert_hf_to_gguf.py``'s but for ``general.*`` (with a clone of the release: ``$PVI_LLAMACPP_SRC`` or
  ``logs/<platform>/llamacpp/src``); other tensors, shapes or configurations, a file unlike the plan and a prompt
  beyond the checkpoint's context are refused; the command writes, checks, quantises and records real weights.
"""
from __future__ import annotations

import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import pytest

from experiments.paper import records as R
from experiments.reexec import llamacpp as L
from pvi.models.decoder import CONFIGS, DecoderConfig, decoder_param_count

CODE = Path(__file__).resolve().parents[1]
SMALL = DecoderConfig("test-llama", 64, 2, 4, 2, 16, 96, 300, norm="rmsnorm", mlp="swiglu", pos="rope", bias=False,
                      tied=False, max_pos=256)
"""A small Llama-architecture decoder (grouped key-value heads, a vocabulary of 300 tokens) for the writer's test."""


def test_the_tensor_plan_has_llama_2_7bs_names_shapes_and_parameters():
    cfg = CONFIGS["llama2-7b"]
    plan = L.tensor_plan(cfg, cfg.n_layers)
    shapes = dict(plan)
    assert len(plan) == len(shapes) == 3 + 9 * 32
    assert plan[0] == ("token_embd.weight", (32000, 4096)) and plan[-1] == ("output.weight", (32000, 4096))
    assert plan[-2] == ("output_norm.weight", (4096,))
    for i in (0, 31):
        b = f"blk.{i}."
        assert [shapes[b + n] for n in ("attn_norm.weight", "attn_q.weight", "attn_k.weight", "attn_v.weight",
                                        "attn_output.weight", "ffn_norm.weight", "ffn_gate.weight", "ffn_up.weight",
                                        "ffn_down.weight")] == \
            [(4096,), (4096, 4096), (4096, 4096), (4096, 4096), (4096, 4096), (4096,), (11008, 4096),
             (11008, 4096), (4096, 11008)]
    assert "blk.32.attn_q.weight" not in shapes
    params = sum(L._size(s) for _, s in plan)
    assert params == 6_738_415_616 and round(params / 1e9, 2) == 6.74            # Llama-2-7B's count
    assert sum(L._size(s) for _, s in plan if len(s) == 2) == decoder_param_count(cfg)
    assert sum(L._size(s) for _, s in L.tensor_plan(CONFIGS["llama2-13b"], 40)) == 13_015_864_320
    # the files: 13.5 GB in bf16, 7.2 GB in Q8_0 (34 bytes per 32 weights); one block of the rehearsal: 1.4 GB both
    assert round(L.file_bytes(cfg, 32, "bf16") / 1e9, 1) == 13.5
    assert round(L.file_bytes(cfg, 32, "q8_0") / 1e9, 1) == 7.2
    assert round((L.file_bytes(cfg, 1, "bf16") + L.file_bytes(cfg, 1, "q8_0")) / 1e9, 1) == 1.4


def test_the_vocabulary_has_llama_2s_layout():
    tokens, scores, types = L.vocabulary(32000)
    assert len(tokens) == len(set(tokens)) == len(scores) == len(types) == 32000
    assert tokens[:4] == ["<unk>", "<s>", "</s>", "<0x00>"] and tokens[3 + 0x0A] == "<0x0A>" and tokens[258] == "<0xFF>"
    assert types[:4] == [2, 3, 3, 6] and set(types[3:259]) == {6} and set(types[259:]) == {1}
    with pytest.raises(ValueError):
        L.vocabulary(L.SPECIAL - 1)


@pytest.mark.parametrize("model", ["opt-1.3b", "gpt2", "qwen3-4b", "tiny-llama"])
def test_a_model_that_is_not_llama_2s_architecture_is_refused(model, monkeypatch):
    for name in ("llama2-7b", "llama2-13b", "small-llama"):
        L.check_llama(L.decoder_config(name))
    with pytest.raises(SystemExit, match="not Llama-2's architecture"):
        L.check_llama(L.decoder_config(model))
    # the command refuses it before it clones, builds or writes anything
    monkeypatch.setattr(sys, "argv", ["llamacpp.py", "--model", model, "--llamacpp-dir", "/nonexistent/llamacpp"])
    with pytest.raises(SystemExit, match="not Llama-2's architecture"):
        L.main()


def _release_src() -> Path | None:
    """A clone of the release: ``$PVI_LLAMACPP_SRC``, else one under ``logs/<platform>/llamacpp/src``; else None."""
    found = ([Path(os.environ["PVI_LLAMACPP_SRC"])] if os.environ.get("PVI_LLAMACPP_SRC") else []) + \
        sorted((CODE.parent / "logs").glob("*/llamacpp/src"))
    return next((s for s in found if (s / "gguf-py" / "gguf" / "__init__.py").is_file()), None)


def _gguf():
    """gguf-py: an installed ``gguf``, else a llama.cpp clone's (:func:`_release_src`); else skip."""
    if importlib.util.find_spec("gguf") is not None:
        import gguf
        return gguf
    if _release_src() is not None:
        return L.import_gguf(_release_src())
    pytest.skip("no gguf-py (an installed gguf, or a llama.cpp clone: $PVI_LLAMACPP_SRC or "
                "logs/<platform>/llamacpp/src)")


def test_a_missing_dependency_of_gguf_py_is_a_failed_step(tmp_path, monkeypatch):
    """gguf-py imports numpy and PyYAML: without one of them the step fails with its name, not a traceback."""
    pkg = tmp_path / "gguf-py" / "gguf"
    pkg.mkdir(parents=True)
    (pkg / "__init__.py").write_text("import pvi_no_such_module_yaml_stand_in\n", encoding="utf-8")
    monkeypatch.setattr(sys, "path", list(sys.path))
    monkeypatch.delitem(sys.modules, "gguf", raising=False)
    with pytest.raises(L.LlamaCppError, match="PyYAML") as e:
        L.import_gguf(tmp_path)
    assert e.value.step == "gguf-py"


def test_a_tiny_gguf_written_one_tensor_at_a_time_reads_back(tmp_path):
    gguf = _gguf()
    L.check_llama(SMALL)
    path = tmp_path / "small.gguf"
    wrote = L.write_gguf(gguf, path, SMALL, 2, seed=3)
    plan = L.tensor_plan(SMALL, 2)
    assert wrote["tensors"] == len(plan) and wrote["params"] == sum(L._size(s) for _, s in plan)
    assert wrote["bytes"] == path.stat().st_size >= L.file_bytes(SMALL, 2, "bf16")
    reader = gguf.GGUFReader(str(path))
    field = lambda k: reader.get_field(k).contents()                  # noqa: E731
    assert field("general.architecture") == "llama" and field("tokenizer.ggml.model") == "llama"
    assert (field("llama.block_count"), field("llama.embedding_length"), field("llama.feed_forward_length"),
            field("llama.attention.head_count"), field("llama.attention.head_count_kv"),
            field("llama.rope.dimension_count"), field("llama.context_length")) == (2, 64, 96, 4, 2, 16, 256)
    assert field("general.file_type") == int(gguf.LlamaFileType.MOSTLY_BF16)
    assert field("llama.attention.layer_norm_rms_epsilon") == pytest.approx(L.RMS_EPS)
    tokens = reader.get_field("tokenizer.ggml.tokens")
    assert len(tokens.data) == 300 and [tokens.contents(i) for i in (0, 1, 2, 3, 259)] == \
        ["<unk>", "<s>", "</s>", "<0x00>", "t0"]
    assert (field("tokenizer.ggml.bos_token_id"), field("tokenizer.ggml.eos_token_id"),
            field("tokenizer.ggml.add_bos_token")) == (1, 2, True)
    got = {t.name: t for t in reader.tensors}
    assert list(got) == [name for name, _ in plan]
    for name, shape in plan:
        t = got[name]
        x = gguf.quants.dequantize(t.data, t.tensor_type)
        assert x.shape == shape, name
        if len(shape) == 1:
            assert t.tensor_type == gguf.GGMLQuantizationType.F32 and (x == 1).all(), name
        else:
            assert t.tensor_type == gguf.GGMLQuantizationType.BF16, name
            assert 0.015 < float(x.std()) < 0.025 and abs(float(x.mean())) < 0.005, name
    del reader, got


STUB = """import json, sys
args = sys.argv[1:]
seq = int(args[args.index("-p") + 1])
threads = int(args[args.index("-t") + 1])
reps = int(args[args.index("-r") + 1])
print("llama-bench: a line that is not a record")
for fa, base in ((0, 2.0), (1, 1.5)):
    samples = [int((base + 0.1 * i) * 1e9) for i in range(reps)]
    print(json.dumps({"build_commit": "d812350", "build_number": 11429, "cpu_info": "AMD EPYC 9334", "backends": "CPU",
                      "model_type": "llama 7B Q8_0", "model_size": 7160348672, "model_n_params": 6738415616,
                      "n_batch": seq, "n_ubatch": 512, "n_threads": threads, "cpu_mask": "0x0", "cpu_strict": False,
                      "poll": 50, "type_k": "f16", "type_v": "f16", "n_gpu_layers": 0, "flash_attn": fa,
                      "load_mode": "none", "repack": True, "n_prompt": seq, "n_gen": 0, "n_depth": 0,
                      "test_time": "2026-10-26T00:00:00Z", "avg_ns": sum(samples) // reps, "stddev_ns": 1,
                      "samples_ns": samples}))
print("llama-bench: done", file=sys.stderr)
"""


def _args(**kw):
    return Namespace(**{"threads": 8, "reps": 5, "ubatch": [512, 2048], "flash_attn": ["off", "on"], **kw})


def test_llama_benchs_command_line():
    argv = L.bench_argv(["llama-bench"], Path("m.gguf"), 2048, _args())
    text = " ".join(argv)
    assert text.startswith("llama-bench -m m.gguf -p 2048 -n 0 -t 8 -r 5 -b 2048 -ub 512,2048 -fa off,on -lm none")
    assert argv[-2:] == ["-o", "jsonl"] and "--numa" not in argv and "-C" not in argv and "--cpu-mask" not in argv
    assert " -ub 64 " in " ".join(L.bench_argv(["llama-bench"], Path("m.gguf"), 64, _args()))   # capped at the prompt
    assert L.ubatches(1000, [512, 2048]) == [512, 1000]
    assert L.c_compiler("/usr/bin/g++-12") == str(Path("/usr/bin/gcc-12")) and L.c_compiler("clang++") == "clang"


def test_a_stand_in_llama_bench_gives_inexact_records_of_the_median_and_range(tmp_path):
    stub = tmp_path / "bench_stub.py"
    stub.write_text(STUB, encoding="utf-8")
    argv = L.bench_argv([sys.executable, str(stub)], tmp_path / "m.gguf", 2048, _args(ubatch=[512]))
    lines, _ = L.run_bench(argv, tmp_path, "bench")
    assert [b["flash_attn"] for b in lines] == [0, 1]                 # the other line is not read
    recs = []
    for b in lines:
        L.check_line(b, seq=2048, threads=8, reps=5)
        recs.append(L.record(b, model="llama2-7b", n_layers=32, layers=32, wtype="q8_0", threads=8, reps=5,
                             llamacpp={"tag": L.TAG, "argv": argv}, extra={"env": {"cpu_affinity": 8}}))
    off, on = recs
    assert R.kind_of(json.loads(json.dumps(off))) == "llamacpp"
    assert off["measured_s"] == {"32": pytest.approx(2.2)} and off["range_s"]["32"] == pytest.approx([2.0, 2.4])
    assert on["measured_s"]["32"] == pytest.approx(1.7) and on["flash_attn"] == "on" and off["flash_attn"] == "off"
    assert off["full_s"] == pytest.approx(2.2) and off["samples_s"] == pytest.approx([2.0, 2.1, 2.2, 2.3, 2.4])
    assert (off["variant"], off["exact"], off["same_model"], off["weights"]) == \
        ("llamacpp_q8_0", False, False, "random")
    assert off["dtype"] == "llama.cpp Q8_0 (not exact, a different model)" and off["gguf_bytes"] == 7160348672
    assert off["llamacpp"]["n_threads"] == 8 and off["llamacpp"]["tag"] == "v0.6.0" and off["llamacpp"]["argv"] == argv
    rec = R.Rec("cpuv", "c-llamacpp-llama2-7b", "7", "sp-c-llamacpp-llama2-7b-7.out", 1, "llamacpp", off)
    assert R.checks(rec) == []
    # a one-block build is keyed by its block: it never stands for the full model
    one = L.record(lines[0], model="llama2-7b", n_layers=32, layers=1, wtype="bf16", threads=8, reps=5)
    assert one["measured_s"] == {"1": pytest.approx(2.2)} and one["full_s"] is None
    assert one["variant"] == "llamacpp_bf16" and one["dtype"].startswith("llama.cpp bf16")
    # a test other than the one asked for is refused
    for kw in ({"seq": 64}, {"threads": 4}, {"reps": 3}):
        with pytest.raises(L.LlamaCppError):
            L.check_line(lines[0], **{"seq": 2048, "threads": 8, "reps": 5, **kw})
    with pytest.raises(L.LlamaCppError, match="build"):
        L.check_line(dict(lines[0], build_number=1), seq=2048, threads=8, reps=5)
    # a llama-bench that fails is reported with the end of its error output
    fail = tmp_path / "fail_stub.py"
    fail.write_text("import sys\nprint('error: failed to load model', file=sys.stderr)\nsys.exit(1)\n",
                    encoding="utf-8")
    with pytest.raises(L.LlamaCppError, match="failed to load model"):
        L.run_bench([sys.executable, str(fail)], tmp_path, "fail")


def test_a_clones_commit_is_read_without_git(tmp_path):
    src = tmp_path / "src"
    (src / ".git" / "refs" / "tags").mkdir(parents=True)
    (src / ".git" / "HEAD").write_text(L.COMMIT + "\n")                    # the shallow clone's detached HEAD
    assert L._head(src, "no-such-git") == L.COMMIT
    (src / ".git" / "HEAD").write_text("ref: refs/tags/x\n")
    (src / ".git" / "refs" / "tags" / "x").write_text("a" * 40 + "\n")    # a loose ref
    assert L._head(src, "no-such-git") == "a" * 40
    (src / ".git" / "refs" / "tags" / "x").unlink()
    (src / ".git" / "packed-refs").write_text(f"# pack-refs\n{'b' * 40} refs/tags/x\n")
    assert L._head(src, "no-such-git") == "b" * 40
    assert L._head(tmp_path / "none", "no-such-git") == ""


# ------------------------------------------------------------------ --real: a checkpoint's weights
LLAMA2_7B_CONFIG = {"architectures": ["LlamaForCausalLM"], "bos_token_id": 1, "eos_token_id": 2,
                    "hidden_act": "silu", "hidden_size": 4096, "initializer_range": 0.02, "intermediate_size": 11008,
                    "max_position_embeddings": 4096, "model_type": "llama", "num_attention_heads": 32,
                    "num_hidden_layers": 32, "num_key_value_heads": 32, "pretraining_tp": 1, "rms_norm_eps": 1e-05,
                    "rope_scaling": None, "tie_word_embeddings": False, "torch_dtype": "float16",
                    "vocab_size": 32000}
"""meta-llama/Llama-2-7b-hf's ``config.json`` (no ``head_dim``, no ``rope_theta``)."""


def _sentencepiece(path: Path) -> None:
    """A small byte-fallback SentencePiece model in Llama's layout (``<unk>``, ``<s>``, ``</s>``, the 256 bytes, then
    the merges), trained here."""
    import io
    spm = pytest.importorskip("sentencepiece")
    text = [f"the quick brown fox {i} jumps over the lazy dog, said Alice to Bob." for i in range(200)]
    buf = io.BytesIO()
    spm.SentencePieceTrainer.train(sentence_iterator=iter(text), model_writer=buf, vocab_size=320, model_type="bpe",
                                   byte_fallback=True, unk_id=0, bos_id=1, eos_id=2, pad_id=-1,
                                   hard_vocab_limit=False, character_coverage=1.0, minloglevel=2)
    path.write_bytes(buf.getvalue())


@pytest.fixture(scope="module", params=["fp16", "bf16-llama2-config"])
def tiny_checkpoint(request, tmp_path_factory):
    """``(folder, model, cfg)``: a tiny ``LlamaForCausalLM`` (grouped key-value heads) saved as safetensors with its
    SentencePiece tokenizer, and its ``DecoderConfig``.  ``fp16``: one file, Transformers' configuration (with
    ``head_dim`` and ``rope_theta``); ``bf16-llama2-config``: bf16 shards and a configuration without them, as
    Llama-2-7B's."""
    torch = pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    folder = tmp_path_factory.mktemp(f"ckpt-{request.param}")
    _sentencepiece(folder / "sp.model")
    tok = transformers.LlamaTokenizer(vocab_file=str(folder / "sp.model"), legacy=False)
    hf = transformers.LlamaConfig(vocab_size=len(tok), hidden_size=64, intermediate_size=96, num_hidden_layers=2,
                                  num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
                                  rms_norm_eps=1e-5, tie_word_embeddings=False)
    torch.manual_seed(0)
    model = transformers.LlamaForCausalLM(hf).eval()
    with torch.no_grad():
        for n, p in model.named_parameters():
            if "norm" in n:
                p.uniform_(0.5, 1.5)
    bf16 = request.param.startswith("bf16")
    model = model.to(torch.bfloat16 if bf16 else torch.float16)
    model.save_pretrained(folder, safe_serialization=True, max_shard_size="60KB" if bf16 else "1GB")
    tok.save_pretrained(folder)
    (folder / "sp.model").unlink()
    if bf16:
        conf = json.loads((folder / "config.json").read_text(encoding="utf-8"))
        conf.pop("head_dim", None)
        conf.pop("rope_theta", None)
        (folder / "config.json").write_text(json.dumps(conf), encoding="utf-8")
        assert (folder / "model.safetensors.index.json").is_file()
    cfg = DecoderConfig("tiny-ckpt", 64, 2, 4, 2, 16, 96, len(tok), norm="rmsnorm", mlp="swiglu", pos="rope",
                        bias=False, tied=False, max_pos=256)
    return folder, model, cfg


def _rope_hf(x, theta: float = 10000.0, pos: int = 3):
    """Transformers' rotary embedding of one head's vector ``x`` (rotate-half: the pairs ``(i, i + d/2)``)."""
    import numpy as np
    d = x.shape[0]
    a = pos * theta ** (-np.arange(0, d, 2) / d)
    c, s = np.cos(np.concatenate([a, a])), np.sin(np.concatenate([a, a]))
    return x * c + np.concatenate([-x[d // 2:], x[:d // 2]]) * s


def _rope_llamacpp(x, theta: float = 10000.0, pos: int = 3):
    """llama.cpp's ``llama`` rotary embedding of one head's vector ``x`` (its normal mode: adjacent pairs
    ``(2i, 2i + 1)``)."""
    import numpy as np
    a = pos * theta ** (-np.arange(0, x.shape[0], 2) / x.shape[0])
    e, o = x[0::2], x[1::2]
    out = np.empty_like(x)
    out[0::2], out[1::2] = e * np.cos(a) - o * np.sin(a), e * np.sin(a) + o * np.cos(a)
    return out


def test_the_q_and_k_permutation_turns_transformers_rotary_into_llamacpps():
    """``permute`` (the release's ``LlamaModel.permute``) reorders each head's rows so that llama.cpp's adjacent-pair
    rotation of the permuted projection is the permuted rotate-half rotation of Transformers: the same queries and
    keys, so the same attention scores; k is reordered per key-value head."""
    import numpy as np
    rng = np.random.default_rng(0)
    heads, kv_heads, dh, d = 4, 2, 16, 64
    for n_rows, n_kv in ((heads * dh, heads), (kv_heads * dh, kv_heads)):
        w = rng.standard_normal((n_rows, d)).astype(np.float32)
        x = rng.standard_normal(d).astype(np.float32)
        p = L.permute(w, heads, n_kv)
        assert p.shape == w.shape and sorted(map(tuple, p)) == sorted(map(tuple, w))
        want = L.permute(np.concatenate([_rope_hf(h) for h in np.split(w @ x, n_kv)]), n_kv, n_kv)
        got = np.concatenate([_rope_llamacpp(h) for h in np.split(p @ x, n_kv)])
        assert np.allclose(got, want, atol=1e-5)
        # within a head: row i of each half goes to rows 2i and 2i + 1
        assert np.array_equal(p[1], w[dh // 2]) and np.array_equal(p[2], w[1])
    q = rng.standard_normal((heads * dh, d))
    assert np.array_equal(L.permute(q, heads, heads), L.permute(q, heads, None))


def test_a_tiny_checkpoint_converts_one_tensor_at_a_time(tiny_checkpoint, tmp_path):
    """The checkpoint's GGUF: the plan's tensors under their names, q and k permuted, every matrix the checkpoint's
    rounded to bf16, the norms in fp32; the configuration's hyperparameters; the tokenizer's pieces, scores and
    types (the special tokens of ``added_tokens_decoder`` as control tokens) and its special ids."""
    import numpy as np
    import torch
    gguf = _gguf()
    folder, model, cfg = tiny_checkpoint
    path = tmp_path / "real.gguf"
    wrote = L.write_gguf_real(gguf, path, folder, cfg, 2, "tiny")
    assert wrote["tensors"] == len(L.tensor_plan(cfg, 2)) == 21 and wrote["vocab"] == cfg.vocab
    assert L.check_gguf(gguf, path, cfg, 2) == {"tensors": 21, "vocab": cfg.vocab}
    reader = gguf.GGUFReader(str(path))
    field = lambda k: reader.get_field(k).contents() if reader.get_field(k) is not None else None  # noqa: E731
    assert (field("llama.context_length"), field("llama.block_count"), field("llama.vocab_size"),
            field("llama.attention.head_count_kv")) == (256, 2, cfg.vocab, 2)
    assert field("llama.attention.layer_norm_rms_epsilon") == pytest.approx(1e-5)
    assert field("tokenizer.ggml.bos_token_id") == 1 and field("tokenizer.ggml.eos_token_id") == 2
    import sentencepiece
    sp = sentencepiece.SentencePieceProcessor(model_file=str(folder / "tokenizer.model"))
    tokens = reader.get_field("tokenizer.ggml.tokens")
    assert [tokens.contents(i) for i in range(sp.vocab_size())] == [sp.IdToPiece(i) for i in range(sp.vocab_size())]
    types = field("tokenizer.ggml.token_type")
    assert types[:4] == [3, 3, 3, 6] and types[258] == 6 and types[259] == 1      # <unk> <s> </s>: special
    sd = {k: v.float() for k, v in model.state_dict().items()}
    got = {t.name: t for t in reader.tensors}
    for name, shape in L.tensor_plan(cfg, 2):
        want = sd[L.hf_name(name)].numpy()
        if name.endswith("attn_q.weight"):
            want = L.permute(want, 4, 4)
        elif name.endswith("attn_k.weight"):
            want = L.permute(want, 4, 2)
        x = gguf.quants.dequantize(got[name].data, got[name].tensor_type).reshape(shape)
        if len(shape) == 1:
            assert got[name].tensor_type == gguf.GGMLQuantizationType.F32 and np.array_equal(x, want), name
        else:
            assert got[name].tensor_type == gguf.GGMLQuantizationType.BF16, name
            # the checkpoint's fp16 or bf16 values rounded to bf16 (exact for bf16)
            ref = torch.from_numpy(np.ascontiguousarray(want)).to(torch.bfloat16).float().numpy()
            assert np.array_equal(x, ref), name
    del reader, got


def test_the_conversion_is_the_release_converters(tiny_checkpoint, tmp_path):
    """The release's own ``convert_hf_to_gguf.py --outtype bf16`` on the same checkpoint writes the same tensors byte
    for byte (names, types, shapes, data) and the same metadata but ``general.*`` (the file's name and size label)
    (a clone of the release: ``$PVI_LLAMACPP_SRC`` or ``logs/<platform>/llamacpp/src``; else skipped)."""
    import subprocess

    import numpy as np
    src = _release_src()
    if src is None or not (src / "convert_hf_to_gguf.py").is_file():
        pytest.skip("no clone of the release ($PVI_LLAMACPP_SRC or logs/<platform>/llamacpp/src)")
    pytest.importorskip("torch")
    gguf = _gguf()
    folder, _, cfg = tiny_checkpoint
    ref, ours = tmp_path / "release.gguf", tmp_path / "ours.gguf"
    out = subprocess.run([sys.executable, str(src / "convert_hf_to_gguf.py"), str(folder), "--outtype", "bf16",
                          "--outfile", str(ref)], capture_output=True, text=True, timeout=600, cwd=tmp_path)
    assert out.returncode == 0, out.stderr[-2000:]
    L.write_gguf_real(gguf, ours, folder, cfg, 2, "tiny")
    a, b = gguf.GGUFReader(str(ref)), gguf.GGUFReader(str(ours))
    meta = lambda r: {k: (tuple(f.types), f.contents()) for k, f in r.fields.items()         # noqa: E731
                      if not k.startswith(("general.", "GGUF."))}
    assert meta(a) == meta(b)
    ta, tb = ({t.name: (t.tensor_type, tuple(t.shape), np.asarray(t.data).tobytes()) for t in r.tensors}
              for r in (a, b))
    assert ta.keys() == tb.keys() and len(ta) == 21
    assert [n for n in ta if ta[n] != tb[n]] == []
    del a, b


def test_a_checkpoint_of_other_tensors_or_configuration_is_refused(tiny_checkpoint):
    folder, _, cfg = tiny_checkpoint
    index = L.safetensors_index(folder)
    L.check_tensors(index, cfg, 2, folder)
    L.check_tensors(index, cfg, 1, folder)                        # its first block alone
    import dataclasses
    with pytest.raises(L.LlamaCppError, match="not the plan's tensors"):
        L.check_tensors(index, dataclasses.replace(cfg, d_ff=128), 2, folder)
    with pytest.raises(L.LlamaCppError, match="no model.layers.1.mlp.up_proj.weight"):
        L.check_tensors({k: v for k, v in index.items() if k != "model.layers.1.mlp.up_proj.weight"}, cfg, 2, folder)
    with pytest.raises(L.LlamaCppError, match="unexpected"):
        L.check_tensors({**index, "model.layers.0.self_attn.q_proj.bias": index["model.norm.weight"]}, cfg, 2,
                        folder)
    hp = L.checkpoint_config(folder)
    L.check_checkpoint(hp, cfg)
    for change in ({"tie_word_embeddings": True}, {"rope_scaling": {"rope_type": "linear", "factor": 2.0}},
                   {"num_key_value_heads": 4}, {"architectures": ["MistralForCausalLM"]}, {"attention_bias": True}):
        with pytest.raises(SystemExit, match="not a LlamaForCausalLM"):
            L.check_checkpoint({**hp, **change}, cfg)
    L.check_checkpoint(LLAMA2_7B_CONFIG, CONFIGS["llama2-7b"])
    with pytest.raises(SystemExit, match="not a LlamaForCausalLM of llama2-13b"):
        L.check_checkpoint(LLAMA2_7B_CONFIG, CONFIGS["llama2-13b"])


def test_a_file_unlike_the_plan_fails_the_read_back(tmp_path):
    import dataclasses
    gguf = _gguf()
    path = tmp_path / "small.gguf"
    L.write_gguf(gguf, path, SMALL, 2, seed=3)
    assert L.check_gguf(gguf, path, SMALL, 2) == {"tensors": 21, "vocab": 300}
    with pytest.raises(L.LlamaCppError, match="tensors, not"):
        L.check_gguf(gguf, path, SMALL, 1)
    with pytest.raises(L.LlamaCppError, match="vocabulary"):
        L.check_gguf(gguf, path, dataclasses.replace(SMALL, vocab=301), 2)
    with pytest.raises(L.LlamaCppError, match="ffn_up"):
        L.check_gguf(gguf, path, dataclasses.replace(SMALL, d_ff=128), 2)


def test_real_refuses_a_prompt_beyond_the_checkpoints_context_before_building(tmp_path, monkeypatch):
    """With a local checkpoint folder (no download), ``--seq`` beyond its ``max_position_embeddings`` and a
    configuration of other shapes are refused before llama.cpp is cloned, built or anything written; the checkpoint's
    commit is read from a cache snapshot's path."""
    folder = tmp_path / "snapshots" / ("c" * 40)
    folder.mkdir(parents=True)
    (folder / "config.json").write_text(json.dumps(LLAMA2_7B_CONFIG), encoding="utf-8")
    assert L.checkpoint_folder(str(folder)) == (folder, "c" * 40)
    assert L.checkpoint_folder(str(tmp_path))[1] is None
    argv = ["llamacpp.py", "--model", "llama2-7b", "--real", str(folder), "--llamacpp-dir", "/nonexistent/llamacpp"]
    monkeypatch.setattr(sys, "argv", argv + ["--seq", "64", "4097"])
    with pytest.raises(SystemExit, match="--seq: 1 to 4096 tokens"):
        L.main()
    (folder / "config.json").write_text(json.dumps({**LLAMA2_7B_CONFIG, "max_position_embeddings": 2048}),
                                        encoding="utf-8")
    monkeypatch.setattr(sys, "argv", argv + ["--seq", "2049"])
    with pytest.raises(SystemExit, match="--seq: 1 to 2048 tokens"):
        L.main()
    monkeypatch.setattr(sys, "argv", argv[:2] + ["llama2-13b"] + argv[3:] + ["--seq", "64"])
    with pytest.raises(SystemExit, match="not a LlamaForCausalLM of llama2-13b"):
        L.main()


def test_a_real_weight_record_names_its_checkpoint(tmp_path):
    stub = tmp_path / "bench_stub.py"
    stub.write_text(STUB, encoding="utf-8")
    argv = L.bench_argv([sys.executable, str(stub)], tmp_path / "m.gguf", 2048, _args(ubatch=[512]))
    lines, _ = L.run_bench(argv, tmp_path, "bench")
    extra = {"weights": "real", "real": "meta-llama/Llama-2-7b-hf", "revision": "0" * 40, "seed": None,
             "max_pos": 4096, "env": {"cpu_affinity": 8}}
    rec = L.record(lines[0], model="llama2-7b", n_layers=32, layers=32, wtype="q8_0", threads=8, reps=5,
                   llamacpp={"tag": L.TAG}, extra=extra)
    assert (rec["weights"], rec["real"], rec["revision"], rec["seed"]) == \
        ("real", "meta-llama/Llama-2-7b-hf", "0" * 40, None)
    assert rec["exact"] is False and rec["same_model"] is False
    assert R.checks(R.Rec("cpuv", "c-llamacpp-llama2-7b", "7", "sp-c-llamacpp-llama2-7b-7.out", 1, "llamacpp",
                          json.loads(json.dumps(rec)))) == []


def test_the_command_with_a_real_checkpoint_writes_checks_quantises_and_records(tiny_checkpoint, tmp_path, monkeypatch,
                                                                                 capsys):
    """``main`` with ``--real`` (a local folder, no download), the build and the quantisation stood in for and
    llama-bench a stand-in: the checkpoint's file is written and read back, and each record says real weights, the
    checkpoint and its context; the model files are removed."""
    gguf = _gguf()
    folder, _, cfg = tiny_checkpoint
    stub = tmp_path / "bench_stub.py"
    stub.write_text(STUB, encoding="utf-8")
    written = []

    def quantize(step, argv, log, cwd=None):
        assert step == "quantize" and argv[3] == "Q8_0"
        written.append(Path(argv[1]).stat().st_size)
        Path(argv[2]).write_bytes(Path(argv[1]).read_bytes())

    bench_argv = L.bench_argv
    monkeypatch.setattr(L, "decoder_config", lambda name: cfg)
    monkeypatch.setattr(L, "checkout", lambda args, work: tmp_path)
    monkeypatch.setattr(L, "build", lambda args, src, cpu: {"stamp": {"cmake": [], "cmake_version": None,
                                                                       "cxx_version": None, "cc_version": None},
                                                             "bench": "bench", "quantize": "quantize",
                                                             "system_info": None})
    monkeypatch.setattr(L, "import_gguf", lambda src: gguf)
    monkeypatch.setattr(L, "_step", quantize)
    monkeypatch.setattr(L, "bench_argv", lambda bench, model, seq, args: bench_argv([sys.executable, str(stub)],
                                                                                    model, seq, args))
    monkeypatch.setattr(sys, "argv", ["llamacpp.py", "--model", "tiny-ckpt", "--real", str(folder), "--seq", "64",
                                      "--types", "q8_0", "--reps", "2", "--threads", "2", "--ubatch", "512",
                                      "--llamacpp-dir", str(tmp_path / "lcpp"), "--tmp-dir", str(tmp_path)])
    assert L.main() == 0
    recs = [json.loads(x) for x in capsys.readouterr().out.splitlines() if x.startswith("{")]
    assert written and len(recs) == 2
    for r in recs:
        assert (r["weights"], r["real"], r["revision"], r["seed"], r["max_pos"], r["variant"]) == \
            ("real", str(folder), None, None, 256, "llamacpp_q8_0")
    assert not list(tmp_path.glob("pvi-llamacpp-*"))

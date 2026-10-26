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
* a model other than Llama-2's architecture is refused before anything is built.
"""
from __future__ import annotations

import importlib.util
import json
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


def _gguf():
    """gguf-py: an installed ``gguf``, else a llama.cpp clone's (``logs/<platform>/llamacpp/src``); else skip."""
    if importlib.util.find_spec("gguf") is not None:
        import gguf
        return gguf
    for src in sorted((CODE.parent / "logs").glob("*/llamacpp/src")):
        if (src / "gguf-py" / "gguf" / "__init__.py").is_file():
            return L.import_gguf(src)
    pytest.skip("no gguf-py (an installed gguf, or a llama.cpp clone under logs/<platform>/llamacpp/src)")


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

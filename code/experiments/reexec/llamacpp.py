"""llama.cpp on the CPU: a production inference engine's prefill of a model of Llama-2-7B's shapes on the same 8
threads as the other re-execution baselines.  It is not exact and it computes a different model (llama.cpp's own
Q8_0 or bf16 weights), so it can confirm no claim; it is reported next to exact re-execution, never in its place.

Produces the llama.cpp rows of ``tab:options`` and ``tab:eval-reexec`` and the text's llama.cpp numbers (the package's
tier ``cpuv``; ``experiments/paper/numbers.py``).

    python -m experiments.reexec.llamacpp --model llama2-7b --seq 64 2048 --types q8_0 bf16 --reps 5 --threads 8 \\
        --llamacpp-dir ../logs/final/llamacpp
    python -m experiments.reexec.llamacpp --model llama2-7b --layers 1 --seq 64 --types q8_0 bf16 --reps 2 \\
        --threads 8 --llamacpp-dir ../logs/rehearsal/llamacpp                 # the rehearsal's cell

**Build.**  llama.cpp's release v0.6.0 (build b11429, commit ``d8123504``), cloned (``git clone --depth 1``) into
``<--llamacpp-dir>/src`` unless ``--llamacpp-src`` names a checkout; its commit must be the release's.  CMake
(``--cmake``) configures a CPU-only Release build for this host's CPU (``GGML_NATIVE``, i.e. ``-march=native``; no
CUDA, OpenSSL, server, tests or examples; OpenMP and llamafile's GEMMs at llama.cpp's defaults) with the compilers
``--cxx`` and ``--cc`` and builds ``llama-bench`` and ``llama-quantize`` with ``--jobs`` jobs into
``<--llamacpp-dir>/build``.  A stamp file there records the release, the commit, the CMake flags, the compilers and
the CPU; a later run whose stamp matches reuses the build.  The build's ``llama_print_system_info`` (the CPU features
it was compiled for, e.g. ``AVX512_VNNI`` and ``AVX512_BF16``) goes into every record.

**Weights.**  Random weights of the model's exact shapes (``pvi.models.decoder.CONFIGS``; ``--layers N``: its first
``N`` blocks), as every cost cell of the paper uses: normal values of standard deviation 0.02 for the embedding, the
projections and the LM head and ones for the norms (``--seed``), so that no activation is denormal.  They are written
as a bf16 GGUF by the clone's own ``gguf-py``, one tensor at a time (memory holds one tensor), with a vocabulary of
Llama-2's SentencePiece layout (``<unk>``, ``<s>``, ``</s>``, the 256 byte tokens and unique fillers);
``llama-quantize`` then writes the Q8_0 file (the norms stay fp32).  This step needs numpy, PyYAML (which ``gguf-py``
imports) and the clone's ``gguf-py`` only (no checkpoint, sentencepiece, protobuf or torch).  The files go to a new
folder under ``--tmp-dir`` (default the system's temporary folder, ``$TMPDIR``), deleted at exit.  Only Llama-2's
architecture (RMSNorm, SwiGLU, rotary positions, no biases, an untied LM head) is accepted.

**Measurement.**  ``llama-bench`` times prompt processing only (``-p T -n 0``) on ``--threads`` threads (``-t``;
never its default, every physical core of the machine): ``--reps`` passes after one untimed warm-up pass of the same
prompt (random token ids; the KV cache is cleared before each pass), with the weights read into the process's memory
(``-lm none``, not mapped from the page cache), the default CPU mask and no ``--numa``, so that its OpenMP threads
stay on the CPUs that ``run.sh`` pinned the job to.  The prompt is one batch (``-b T``) processed in micro-batches of
each of ``--ubatch`` tokens (at most ``T``), with each of ``--flash-attn``'s attention kernels; the tables use the
fastest configuration per weight type and prompt length.  A pass computes the next-token logits at the last position
only; unlike the verifier's pruned graph, llama.cpp computes the last block's attention at every position.
``llama-bench``'s output is read by this script and never copied to the job's log.

**Record** (one JSON line per weight type, prompt length and configuration, on stdout): ``kind`` (``llamacpp``),
``variant`` (``llamacpp_q8_0``, ``llamacpp_bf16``), ``dtype``, ``exact`` and ``same_model`` (false), ``weights``
(``random``), ``seed``, ``model``, ``seq``, ``device``, ``threads``, ``reps``, ``layers`` (the blocks built),
``n_layers`` (the model's), ``measured_s`` (the median pass, keyed by the blocks built: only a full build fills a
table cell), ``range_s`` (its ``[min, max]``), ``samples_s``, ``full_s`` (the median of a full build, else null),
``prune_last`` (false), ``flash_attn``, ``n_batch``, ``n_ubatch``, ``gguf_bytes`` and ``gguf_params`` (the weights'
bytes and parameters), ``gguf_file_bytes``, ``max_pos``, ``peak`` (the peak resident set of llama-bench's run of
that prompt length, every configuration), ``git``,
``cpu``, ``host``, ``env`` (:func:`experiments.common.records.environment`) and ``llamacpp``: the release, commit and
build number, the CMake flags, CMake and the compilers, the system information, llama-bench's command line, the
OpenMP variables set, and llama-bench's own fields (its CPU description, threads, CPU mask, KV-cache types, load mode,
the samples in ns).  A failed step prints ``FAILED: llama.cpp <step>: ...`` and the script exits with status 1.

Hardware, time, memory and disk: 8 threads of the verifier's CPU.  The first build takes about 5 to 10 minutes with 8
jobs (later runs reuse it); writing Llama-2-7B's 13.5 GB bf16 file and quantising it take a few minutes; one pass at
2,048 tokens takes roughly 30 to 70 s on 8 Zen 4 cores, so the full cell (two weight types at 64 and 2,048 tokens, two
attention kernels and two micro-batch sizes, one warm-up and 5 timed passes each) takes about one to one and a half
hours (the package's limit: 3 h).  Peak memory: the bf16 weights in memory (13.5 GB) and the compute buffers, within
the job's 64 GB.  Disk: about 21 GB under ``--tmp-dir`` for Llama-2-7B (the bf16 and the Q8_0 file; 1.4 GB for one
block), and about 1 GB under ``--llamacpp-dir`` (the clone and the build).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import signal
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path

from experiments.common.models import decoder_config
from experiments.common.paths import CODE

TAG = "v0.6.0"
"""The llama.cpp release built and timed (2026-10-05)."""
COMMIT = "d81235049384534c167caea52b85a694f6103d14"
"""The release's commit."""
BUILD_NUMBER = 11429
"""The release's build number (``b11429``; a shallow clone cannot count it, so CMake is given it)."""
REPOSITORY = "https://github.com/ggml-org/llama.cpp.git"
CMAKE_FLAGS = ("-DCMAKE_BUILD_TYPE=Release", "-DGGML_NATIVE=ON", "-DGGML_CUDA=OFF", "-DGGML_CCACHE=OFF",
               "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
               "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_APP=OFF", "-DLLAMA_BUILD_IS_DEV=OFF",
               f"-DLLAMA_BUILD_NUMBER={BUILD_NUMBER}")
"""The CMake configuration: a Release build for this CPU (``-march=native``), CPU only, without HTTPS, the server,
the tests and the examples; OpenMP and llamafile's GEMMs keep llama.cpp's defaults (on)."""
TARGETS = ("llama-bench", "llama-quantize")
TYPES = {"q8_0": "llama.cpp Q8_0", "bf16": "llama.cpp bf16"}
"""The weight types (``--types``) and the record's ``dtype`` stem of each."""
QUANTIZE = {"q8_0": "Q8_0"}
"""``llama-quantize``'s name of each type written from the bf16 file."""
FLASH_ATTN = {0: "off", 1: "on", -1: "auto"}
"""llama-bench's ``flash_attn`` field (the requested setting) as ``-fa`` names it."""
SPECIAL = 3 + 256
"""The vocabulary's leading tokens: ``<unk>``, ``<s>``, ``</s>`` and the 256 byte tokens."""
STD = 0.02
"""The random weights' standard deviation."""
RMS_EPS = 1e-5
"""Llama-2's RMSNorm epsilon."""
BENCH_FIELDS = ("build_commit", "build_number", "cpu_info", "backends", "model_type", "model_size", "model_n_params",
                "n_batch", "n_ubatch", "n_threads", "cpu_mask", "cpu_strict", "poll", "type_k", "type_v",
                "n_gpu_layers", "flash_attn", "load_mode", "repack", "n_prompt", "n_gen", "n_depth", "test_time",
                "avg_ns", "stddev_ns", "samples_ns")
"""The fields of llama-bench's JSON line that a record keeps (under ``llamacpp``)."""
SYSTEM_INFO = ("import ctypes, sys; lib = ctypes.CDLL(sys.argv[1]); f = lib.llama_print_system_info; "
               "f.restype = ctypes.c_char_p; print(f().decode())")
"""Prints the build's ``llama_print_system_info`` (run in a child process, so that this one never loads the build's
OpenMP runtime)."""


class LlamaCppError(RuntimeError):
    """A step that failed: ``step`` names it (the clone, the build, the weights, the quantisation, the benchmark)."""

    def __init__(self, step: str, message: str):
        super().__init__(message)
        self.step = step


# ================================================================================================ the model
def check_llama(cfg) -> None:
    """Refuse a decoder that llama.cpp's ``llama`` architecture does not compute with the same shapes: it needs
    Llama-2's RMSNorm, SwiGLU MLP, rotary positions, no biases, an untied LM head, no q/k norms, heads of
    ``d_model / n_heads``, whole groups of key-value heads, and a vocabulary that holds the :data:`SPECIAL` tokens."""
    need = (("RMSNorm", cfg.norm == "rmsnorm"), ("a SwiGLU MLP", cfg.mlp == "swiglu"),
            ("rotary positions", cfg.pos == "rope"), ("no biases", not cfg.bias), ("an untied LM head", not cfg.tied),
            ("no q/k norms", not cfg.qk_norm), ("no embedding projection", not cfg.embed_dim),
            ("heads of d_model / n_heads", cfg.n_heads * cfg.head_dim == cfg.d_model),
            ("whole groups of key-value heads", cfg.n_heads % cfg.n_kv_heads == 0),
            (f"a vocabulary of at least {SPECIAL} tokens", cfg.vocab >= SPECIAL))
    missing = [what for what, ok in need if not ok]
    if missing:
        raise SystemExit(f"--model {cfg.name}: not Llama-2's architecture, which llama.cpp's 'llama' models need "
                         f"({', '.join(missing)})")


def tensor_plan(cfg, layers: int) -> list[tuple[str, tuple[int, ...]]]:
    """``[(GGUF tensor name, numpy shape)]`` of a Llama decoder of ``cfg``'s shapes with ``layers`` blocks, in the
    order they are written: the embedding, each block's norms and projections, the final norm and the LM head.  A
    weight matrix is ``(out, in)`` (GGUF stores the dimensions reversed); a norm is one-dimensional."""
    d, f, dh = cfg.d_model, cfg.d_ff, cfg.head_dim
    q, kv = cfg.n_heads * dh, cfg.n_kv_heads * dh
    plan = [("token_embd.weight", (cfg.vocab, d))]
    for i in range(layers):
        b = f"blk.{i}."
        plan += [(b + "attn_norm.weight", (d,)), (b + "attn_q.weight", (q, d)), (b + "attn_k.weight", (kv, d)),
                 (b + "attn_v.weight", (kv, d)), (b + "attn_output.weight", (d, q)), (b + "ffn_norm.weight", (d,)),
                 (b + "ffn_gate.weight", (f, d)), (b + "ffn_up.weight", (f, d)), (b + "ffn_down.weight", (d, f))]
    plan += [("output_norm.weight", (d,)), ("output.weight", (cfg.vocab, d))]
    return plan


def vocabulary(n: int) -> tuple[list[str], list[float], list[int]]:
    """``(tokens, scores, token types)`` of a vocabulary of ``n`` tokens in Llama-2's SentencePiece layout:
    ``<unk>`` (0), ``<s>`` (1), ``</s>`` (2), the byte tokens ``<0x00>`` to ``<0xFF>``, then unique fillers.  The
    types are gguf's ``TokenType`` values (unknown 2, control 3, byte 6, normal 1); llama-bench feeds random token
    ids, so no text is ever tokenised."""
    if n < SPECIAL:
        raise ValueError(f"a vocabulary of {n} tokens cannot hold the {SPECIAL} special and byte tokens")
    tokens = ["<unk>", "<s>", "</s>"] + [f"<0x{b:02X}>" for b in range(256)]
    tokens += [f"t{i}" for i in range(n - SPECIAL)]
    scores = [0.0] * SPECIAL + [-float(i) for i in range(n - SPECIAL)]
    types = [2, 3, 3] + [6] * 256 + [1] * (n - SPECIAL)
    return tokens, scores, types


def file_bytes(cfg, layers: int, wtype: str) -> int:
    """The tensor bytes of the model's GGUF file in ``wtype`` (bf16: 2 bytes per weight; Q8_0: 34 bytes per 32
    weights; the norms in fp32)."""
    per = {"bf16": 2.0, "q8_0": 34 / 32}[wtype]
    return int(sum((4 if len(s) == 1 else per) * _size(s) for _, s in tensor_plan(cfg, layers)))


def _size(shape) -> int:
    n = 1
    for x in shape:
        n *= x
    return n


def import_gguf(src: Path):
    """The clone's own ``gguf-py`` (``<src>/gguf-py``), first on the path; a missing dependency of it (numpy,
    PyYAML) is a failed step."""
    path = Path(src) / "gguf-py"
    if not (path / "gguf" / "__init__.py").is_file():
        raise LlamaCppError("gguf-py", f"{path} holds no gguf package")
    sys.path.insert(0, str(path))
    try:
        import gguf
    except ImportError as e:
        raise LlamaCppError("gguf-py", f"{e} (gguf-py needs numpy and PyYAML)") from e
    if Path(gguf.__file__).resolve().parent != (path / "gguf").resolve():
        raise LlamaCppError("gguf-py", f"another gguf package was imported ({gguf.__file__}), not {path}")
    return gguf


def write_gguf(gguf, path: Path, cfg, layers: int, seed: int) -> dict:
    """Write the bf16 GGUF of ``cfg``'s shapes with ``layers`` blocks and random weights (:data:`STD`, norms one;
    ``seed``) with gguf-py's writer, one tensor at a time: the header and the tensors' descriptions first, then each
    tensor's data as it is drawn.  Returns ``{"tensors", "params", "bytes", "s"}``."""
    import numpy as np

    t0 = time.perf_counter()
    plan = tensor_plan(cfg, layers)
    bf16 = gguf.GGMLQuantizationType.BF16
    w = gguf.GGUFWriter(str(path), "llama")
    w.add_name(f"random-weight {cfg.name}, {layers} of {cfg.n_layers} blocks")
    w.add_file_type(gguf.LlamaFileType.MOSTLY_BF16)
    w.add_quantization_version(gguf.GGML_QUANT_VERSION)
    w.add_context_length(cfg.max_pos)
    w.add_embedding_length(cfg.d_model)
    w.add_block_count(layers)
    w.add_feed_forward_length(cfg.d_ff)
    w.add_head_count(cfg.n_heads)
    w.add_head_count_kv(cfg.n_kv_heads)
    w.add_layer_norm_rms_eps(RMS_EPS)
    w.add_rope_dimension_count(cfg.head_dim)
    w.add_rope_freq_base(cfg.rope_theta)
    w.add_vocab_size(cfg.vocab)
    tokens, scores, types = vocabulary(cfg.vocab)
    w.add_tokenizer_model("llama")
    w.add_token_list(tokens)
    w.add_token_scores(scores)
    w.add_token_types(types)
    w.add_unk_token_id(0)
    w.add_bos_token_id(1)
    w.add_eos_token_id(2)
    w.add_add_bos_token(True)
    w.add_add_eos_token(False)
    for name, shape in plan:
        if len(shape) == 1:
            w.add_tensor_info(name, shape, np.dtype(np.float32), 4 * _size(shape))
        else:
            byte_shape = gguf.quant_shape_to_byte_shape(shape, bf16)
            w.add_tensor_info(name, byte_shape, np.dtype(np.uint8), _size(byte_shape), raw_dtype=bf16)
    w.write_header_to_file()
    w.write_kv_data_to_file()
    w.write_ti_data_to_file()
    rng = np.random.default_rng(seed)
    params = 0
    for _, shape in plan:
        if len(shape) == 1:
            w.write_tensor_data(np.ones(shape, dtype=np.float32))
        else:
            x = rng.standard_normal(shape, dtype=np.float32)
            x *= STD
            w.write_tensor_data(gguf.quants.quantize(x, bf16))
            del x
        params += _size(shape)
    w.close()
    return {"tensors": len(plan), "params": params, "bytes": path.stat().st_size, "s": time.perf_counter() - t0}


# ================================================================================================ processes
def _run(argv: list[str], out: Path, err: Path, cwd: Path | None = None) -> tuple[int, int | None]:
    """Run ``argv`` with its standard output and error in the files ``out`` and ``err``: ``(exit status, peak
    resident set in bytes)`` (the peak where the platform reports a child's, else None)."""
    with open(out, "wb") as fo, open(err, "wb") as fe:
        p = subprocess.Popen(argv, stdout=fo, stderr=fe, cwd=cwd)
        if hasattr(os, "wait4"):
            _, status, usage = os.wait4(p.pid, 0)
            p.returncode = os.waitstatus_to_exitcode(status)
            return p.returncode, usage.ru_maxrss * 1024
        return p.wait(), None


def _tail(path: Path, n: int = 12) -> str:
    try:
        lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return ""
    return " | ".join(x.strip()[:200] for x in lines[-n:] if x.strip())


def _step(step: str, argv: list[str], log: Path, cwd: Path | None = None) -> None:
    """One build or conversion command, its output in ``log`` (``.out``/``.err``); a failure raises."""
    status, _ = _run(argv, log.with_suffix(".out"), log.with_suffix(".err"), cwd)
    if status != 0:
        raise LlamaCppError(step, f"{' '.join(argv)} exited with status {status}: "
                                  f"{_tail(log.with_suffix('.err')) or _tail(log.with_suffix('.out'))}")


def _first_line(argv: list[str]) -> str | None:
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=60)
        return (out.stdout.strip() or out.stderr.strip()).splitlines()[0] if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError, IndexError):
        return None


def c_compiler(cxx: str) -> str:
    """The C compiler next to the C++ compiler ``cxx``: ``g++-12`` -> ``gcc-12``, ``clang++`` -> ``clang``,
    ``c++`` -> ``cc`` (in the same folder)."""
    p = Path(cxx)
    for cpp, c in (("clang++", "clang"), ("g++", "gcc"), ("c++", "cc")):
        if cpp in p.name:
            return str(p.with_name(p.name.replace(cpp, c)))
    raise SystemExit(f"--cc: no C compiler known next to {cxx}; pass --cc <path>")


def _tool(name: str, what: str, step: str = "build") -> str:
    path = shutil.which(name)
    if path is None:
        raise LlamaCppError(step, f"{what} '{name}' not found")
    return path


# ================================================================================================ the build
def checkout(args, logs: Path) -> Path:
    """The release's sources: ``--llamacpp-src``, else ``<--llamacpp-dir>/src`` (cloned when missing); their commit
    must be :data:`COMMIT`."""
    git = _tool(args.git, "--git", "clone")
    src = Path(args.llamacpp_src) if args.llamacpp_src else Path(args.llamacpp_dir) / "src"
    if not src.exists():
        if args.llamacpp_src:
            raise LlamaCppError("clone", f"--llamacpp-src {src} does not exist")
        print(f"llama.cpp: cloning {TAG} into {src}", file=sys.stderr, flush=True)
        _step("clone", [git, "-c", "advice.detachedHead=false", "clone", "--depth", "1", "--branch", TAG,
                        "--single-branch", REPOSITORY, str(src)], logs / "clone.log")
    head = subprocess.run([git, "-c", "safe.directory=*", "-C", str(src), "rev-parse", "HEAD"], capture_output=True,
                          text=True, timeout=60).stdout.strip()
    if head != COMMIT:
        raise LlamaCppError("clone", f"{src} is at commit {head or '(none)'}, not {TAG}'s {COMMIT} (remove it, or "
                                     "clone the release there on a host with network access)")
    return src


def build(args, src: Path, cpu: str | None) -> dict:
    """``llama-bench`` and ``llama-quantize`` of the release, built for this host's CPU into
    ``<--llamacpp-dir>/build``, or reused when the stamp there matches (the release, the commit, the CMake flags,
    CMake, the compilers and the CPU).  Returns the build's description, with its system information."""
    cmake = _tool(args.cmake, "--cmake")
    cxx = _tool(args.cxx, "--cxx")
    cc = _tool(args.cc or c_compiler(args.cxx), "--cc")
    out = Path(args.llamacpp_dir) / "build"
    flags = [*CMAKE_FLAGS, f"-DCMAKE_C_COMPILER={cc}", f"-DCMAKE_CXX_COMPILER={cxx}"]
    stamp = {"tag": TAG, "commit": COMMIT, "build_number": BUILD_NUMBER, "src": str(src.resolve()), "cmake": flags,
             "cmake_version": _first_line([cmake, "--version"]), "cxx": cxx,
             "cxx_version": _first_line([cxx, "--version"]), "cc": cc, "cc_version": _first_line([cc, "--version"]),
             "cpu": cpu}
    stamp_file = out / "pvi_build.json"
    bins = {t: out / "bin" / t for t in TARGETS}
    try:
        old = json.loads(stamp_file.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        old = None
    if old == stamp and all(p.is_file() for p in bins.values()):
        print(f"llama.cpp: reusing the build in {out}", file=sys.stderr, flush=True)
    else:
        if out.exists():
            shutil.rmtree(out)
        out.mkdir(parents=True)
        t0 = time.perf_counter()
        print(f"llama.cpp: building {', '.join(TARGETS)} of {TAG} in {out} with {args.jobs} jobs", file=sys.stderr,
              flush=True)
        _step("build", [cmake, "-S", str(src), "-B", str(out), *flags], out / "configure.log")
        for t in TARGETS:            # one target per call (CMake 3.14 builds one target at a time)
            _step("build", [cmake, "--build", str(out), "--config", "Release", "-j", str(args.jobs), "--target", t],
                  out / f"build-{t}.log")
        missing = [str(p) for p in bins.values() if not p.is_file()]
        if missing:
            raise LlamaCppError("build", f"the build wrote no {', '.join(missing)}")
        stamp_file.write_text(json.dumps(stamp, indent=1), encoding="utf-8")
        print(f"llama.cpp: built in {time.perf_counter() - t0:.0f} s", file=sys.stderr, flush=True)
    info = None
    lib = next((p for p in (out / "bin" / "libllama.so", out / "bin" / "libllama.dylib", out / "bin" / "llama.dll")
                if p.is_file()), None)
    if lib is not None:
        got = subprocess.run([sys.executable, "-c", SYSTEM_INFO, str(lib)], capture_output=True, text=True,
                             timeout=120)
        info = (got.stdout.strip() or None) if got.returncode == 0 else None
    if info is None:
        print(f"llama.cpp: no system information from {lib or out / 'bin'}", file=sys.stderr, flush=True)
    return {"stamp": stamp, "bench": str(bins["llama-bench"]), "quantize": str(bins["llama-quantize"]),
            "system_info": info}


# ================================================================================================ the benchmark
def ubatches(seq: int, sizes) -> list[int]:
    """The micro-batch sizes of a prompt of ``seq`` tokens: each of ``sizes`` capped at ``seq`` (llama.cpp caps it
    there itself), each once."""
    return sorted({min(int(u), seq) for u in sizes})


def bench_argv(bench: list[str], model: Path, seq: int, args) -> list[str]:
    """llama-bench's command line for one prompt length: prompt processing only, ``--threads`` threads, ``--reps``
    passes after its warm-up, the whole prompt as one batch in each micro-batch size, each attention kernel, the
    weights in process memory, JSON lines on stdout."""
    return [*bench, "-m", str(model), "-p", str(seq), "-n", "0", "-t", str(args.threads), "-r", str(args.reps),
            "-b", str(seq), "-ub", ",".join(map(str, ubatches(seq, args.ubatch))), "-fa", ",".join(args.flash_attn),
            "-lm", "none", "-ngl", "0", "-o", "jsonl"]


def run_bench(argv: list[str], work: Path, name: str) -> tuple[list[dict], int | None]:
    """Run llama-bench (``argv``) with its output in files under ``work``: ``(its JSON lines, its peak resident set
    in bytes)``.  Every other line of its output is ignored; a nonzero exit status raises."""
    out, err = work / f"{name}.jsonl", work / f"{name}.err"
    status, peak = _run(argv, out, err)
    if status != 0:
        raise LlamaCppError("bench", f"llama-bench exited with status {status}: {_tail(err)}")
    lines = [json.loads(x) for x in out.read_text(encoding="utf-8").splitlines() if x.startswith("{")]
    if not lines:
        raise LlamaCppError("bench", f"llama-bench wrote no JSON line: {_tail(err)}")
    return lines, peak


def check_line(b: dict, *, seq: int, threads: int, reps: int) -> None:
    """Raise unless llama-bench's line is the test asked for: a prompt of ``seq`` tokens and no generation, on
    ``threads`` threads, ``reps`` samples, built from the release."""
    bad = []
    if b.get("n_prompt") != seq or b.get("n_gen") != 0 or b.get("n_depth", 0) != 0:
        bad.append(f"a test of {b.get('n_prompt')} prompt and {b.get('n_gen')} generated tokens, not {seq} and 0")
    if b.get("n_threads") != threads:
        bad.append(f"{b.get('n_threads')} threads, not {threads}")
    if len(b.get("samples_ns") or []) != reps:
        bad.append(f"{len(b.get('samples_ns') or [])} samples, not {reps}")
    if b.get("build_number") != BUILD_NUMBER or not COMMIT.startswith(str(b.get("build_commit") or "-")):
        bad.append(f"build {b.get('build_number')} ({b.get('build_commit')}), not {TAG}'s {BUILD_NUMBER} "
                   f"({COMMIT[:7]})")
    if bad:
        raise LlamaCppError("bench", "; ".join(bad))


def record(b: dict, *, model: str, n_layers: int, layers: int, wtype: str, threads: int, reps: int,
           llamacpp: dict | None = None, extra: dict | None = None) -> dict:
    """The record of one llama-bench line ``b``: the median of its samples keyed by the blocks built, their
    ``[min, max]``, the configuration, ``llamacpp`` (the build and the command) and ``extra`` (the run's fields)."""
    samples = [ns / 1e9 for ns in b["samples_ns"]]
    median = statistics.median(samples)
    key = str(layers)
    rec = {"kind": "llamacpp", "engine": "llama.cpp", "variant": f"llamacpp_{wtype}", "wtype": wtype,
           "dtype": f"{TYPES[wtype]} (not exact, a different model)", "exact": False, "same_model": False,
           "weights": "random", "model": model, "seq": b["n_prompt"], "device": "cpu", "threads": threads,
           "reps": reps, "layers": layers, "n_layers": n_layers, "measured_s": {key: median},
           "range_s": {key: [min(samples), max(samples)]}, "samples_s": samples,
           "full_s": median if layers == n_layers else None, "prune_last": False,
           "flash_attn": FLASH_ATTN.get(b.get("flash_attn"), b.get("flash_attn")), "n_batch": b.get("n_batch"),
           "n_ubatch": b.get("n_ubatch"), "gguf_bytes": b.get("model_size"), "gguf_params": b.get("model_n_params")}
    rec.update(extra or {})
    rec["llamacpp"] = {**(llamacpp or {}), **{k: b.get(k) for k in BENCH_FIELDS}}
    return rec


# ================================================================================================ the command
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="llama2-7b",
                    help="the decoder's shapes (pvi.models.decoder.CONFIGS; Llama-2's architecture only)")
    ap.add_argument("--layers", type=int, default=0, help="blocks built (default: the model's, the full model)")
    ap.add_argument("--seq", type=int, nargs="+", default=[64, 2048], help="prompt tokens (each one a cell)")
    ap.add_argument("--types", nargs="+", default=list(TYPES), choices=list(TYPES), help="the weights' types")
    ap.add_argument("--reps", type=int, default=5, help="timed passes per configuration, after one warm-up pass")
    ap.add_argument("--threads", type=int, default=8, help="llama-bench's threads (-t)")
    ap.add_argument("--flash-attn", nargs="+", default=["off", "on"], choices=["off", "on"],
                    help="llama-bench's attention kernels (-fa): each one a configuration")
    ap.add_argument("--ubatch", type=int, nargs="+", default=[512, 2048],
                    help="micro-batch sizes (-ub, each capped at the prompt): each one a configuration")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed")
    ap.add_argument("--llamacpp-dir", default=str(CODE.parent / "logs" / "llamacpp"),
                    help="the clone (src/) and the cached build (build/); run.sh passes logs/<platform>/llamacpp")
    ap.add_argument("--llamacpp-src", default=None,
                    help=f"a checkout of llama.cpp {TAG} to build instead of cloning one into --llamacpp-dir")
    ap.add_argument("--tmp-dir", default=None,
                    help="where the model files are written (default the system's temporary folder, $TMPDIR)")
    ap.add_argument("--jobs", type=int, default=8, help="parallel build jobs")
    ap.add_argument("--cmake", default=os.environ.get("CMAKE") or "cmake", help="CMake 3.14 or later (default $CMAKE)")
    ap.add_argument("--cxx", default=os.environ.get("CXX") or "g++", help="the C++ compiler (default $CXX)")
    ap.add_argument("--cc", default=os.environ.get("CC") or None,
                    help="the C compiler (default $CC, else the one next to --cxx)")
    ap.add_argument("--git", default="git", help="git, for the clone and its commit")
    return ap


def _terminate(signum, frame):
    raise SystemExit(128 + signum)              # unwinds main's finally: the model files are removed


def main() -> int:
    from experiments.common import records

    args = build_parser().parse_args()
    cfg = decoder_config(args.model)
    check_llama(cfg)
    layers = args.layers or cfg.n_layers
    if not 1 <= layers <= cfg.n_layers:
        raise SystemExit(f"--layers {args.layers}: 1 to {cfg.n_layers} for {args.model}")
    if max(args.seq) > cfg.max_pos or min(args.seq) < 1:
        raise SystemExit(f"--seq: 1 to {cfg.max_pos} tokens for {args.model}")
    if args.threads < 1 or args.reps < 1:
        raise SystemExit("--threads and --reps: at least 1")
    signal.signal(signal.SIGTERM, _terminate)
    env = records.environment("cpu")
    sha = records.git_sha(CODE)
    root = Path(args.llamacpp_dir)
    root.mkdir(parents=True, exist_ok=True)
    work = Path(tempfile.mkdtemp(prefix="pvi-llamacpp-", dir=args.tmp_dir))
    status = 0
    try:
        try:
            src = checkout(args, work)
            made = build(args, src, env.get("cpu"))
            gguf = import_gguf(src)
        except LlamaCppError as e:
            print(f"FAILED: llama.cpp {e.step}: {e}", file=sys.stderr, flush=True)
            return 1
        stamp = made["stamp"]
        build_info = {"tag": TAG, "commit": COMMIT, "release_build_number": BUILD_NUMBER, "cmake": stamp["cmake"],
                      "cmake_version": stamp["cmake_version"], "cxx_version": stamp["cxx_version"],
                      "cc_version": stamp["cc_version"], "system_info": made["system_info"],
                      "env_vars": {k: v for k, v in sorted(os.environ.items())
                                   if k.startswith(("OMP_", "GOMP_", "KMP_"))}}
        need = file_bytes(cfg, layers, "bf16") + sum(file_bytes(cfg, layers, t) for t in args.types if t != "bf16")
        free = shutil.disk_usage(work).free
        if free < need + (1 << 30):
            print(f"FAILED: llama.cpp disk: the model files need {need / 1e9:.1f} GB under {work.parent}, which has "
                  f"{free / 1e9:.1f} GB free (--tmp-dir)", file=sys.stderr, flush=True)
            return 1
        files = {"bf16": work / f"{args.model}-L{layers}-bf16.gguf"}
        try:
            wrote = write_gguf(gguf, files["bf16"], cfg, layers, args.seed)
        except (OSError, ValueError) as e:
            print(f"FAILED: llama.cpp weights: {e}", file=sys.stderr, flush=True)
            return 1
        print(f"llama.cpp: wrote {wrote['tensors']} tensors ({wrote['params']:,} weights, "
              f"{wrote['bytes'] / 1e9:.2f} GB) in {wrote['s']:.0f} s", file=sys.stderr, flush=True)
        for t in args.types:
            if t in QUANTIZE:
                files[t] = work / f"{args.model}-L{layers}-{t}.gguf"
                try:
                    _step("quantize", [made["quantize"], str(files["bf16"]), str(files[t]), QUANTIZE[t],
                                       str(args.threads)], work / f"quantize-{t}.log")
                except LlamaCppError as e:
                    print(f"FAILED: llama.cpp {e.step}: {e}", file=sys.stderr, flush=True)
                    status = 1
                    files.pop(t)
        if "bf16" not in args.types:
            files.pop("bf16").unlink()
        for t in args.types:
            if t not in files:
                continue
            for seq in args.seq:
                argv = bench_argv([made["bench"]], files[t], seq, args)
                try:
                    lines, peak = run_bench(argv, work, f"bench-{t}-{seq}")
                    for b in lines:
                        check_line(b, seq=seq, threads=args.threads, reps=args.reps)
                except (LlamaCppError, ValueError) as e:
                    print(f"FAILED: llama.cpp {getattr(e, 'step', 'bench')} ({t}, {seq} tokens): {e}",
                          file=sys.stderr, flush=True)
                    status = 1
                    continue
                extra = {"seed": args.seed, "max_pos": cfg.max_pos, "gguf_file_bytes": files[t].stat().st_size,
                         "peak": {"host_peak_rss": peak}, "git": sha,
                         "cpu": platform.processor() or platform.machine(), "host": platform.node(), "env": env}
                for b in lines:
                    records.emit(record(b, model=args.model, n_layers=cfg.n_layers, layers=layers, wtype=t,
                                        threads=args.threads, reps=args.reps,
                                        llamacpp={**build_info, "argv": argv}, extra=extra))
            files[t].unlink()                   # every file was written before the first benchmark
    finally:
        shutil.rmtree(work, ignore_errors=True)
    return status


if __name__ == "__main__":
    sys.exit(main())

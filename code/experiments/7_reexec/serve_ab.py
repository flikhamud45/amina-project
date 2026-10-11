"""Prover overhead against unverified serving, on the same GPU and the same token ids, with batching (plan 2, A7).

    cd code && PYTHONPATH=src python experiments/7_reexec/serve_ab.py --model llama2-7b --workload a b --device cuda
    cd code && PYTHONPATH=src python experiments/7_reexec/serve_ab.py --model llama2-7b --layers 2 --cells 64x1 2048x4 \\
        --batch-proof seq joint --device cuda
    cd code && PYTHONPATH=src python experiments/7_reexec/serve_ab.py --model tiny-llama --cells 32x1 32x4 --device cpu \\
        --threads 2
    cd code && PYTHONPATH=src python experiments/7_reexec/serve_ab.py --table ../logs/l40s_sp2027/sp-serve-*.out

A cell is a model, a prompt length ``T`` and a batch ``B``: ``B`` prompts of ``T`` tokens, random ids drawn with seed 1
as ``perf.py`` and ``reexec.py`` draw them (so prompt 0 of every cell is the B = 1 prompt of its ``T``).  The decoders
are those of ``pvi.fullcheck.transformer.CONFIGS`` with random weights (the costs depend on the shapes only), built as
``perf.py --prune-last`` builds them (seed 0, 32 calibration tokens), on ``--layers`` blocks (0: the full model);
the ``tiny-*`` and ``small-*`` names of :data:`TINY` are small shapes for a laptop and the tests.  Every cell times,
on the same token ids, shapes, device and host:

* **U1**, our exact integer model served without proofs: ``IntGraph.forward(free=True)`` of the pruned graph, every
  claim dropped as it is computed (``send`` keeps nothing).  It computes the integers the prover claims -- the logits
  are checked equal -- so prover/U1 is the cost of the proof itself.
* **U2**, an eager floating-point decoder of the same shapes: ``reexec.FloatDecoder`` (``F.linear`` and
  ``scaled_dot_product_attention``) in each ``--u2-dtype`` (fp16 and/or bf16 on a GPU; fp32 on a CPU), one batched
  prefill of the B prompts, its last block pruned like the integer graph's.  It skips the rotary embedding (an
  element-wise cost), so it is if anything faster than a complete implementation.  U2's model is built and timed
  first, and freed before the integer model goes to the device: the two need not fit at once.
* **the prover** in each of ``--modes`` (C, Kpre) as ``perf.py --fs --wire --lean --prune-last`` runs it: under
  Fiat--Shamir, the proof in the compact wire encoding, the lean prover, mode C under ``--policy``'s commitment plan,
  Kpre with ``--lookups`` (the verifier reads the embedding rows itself; on by default, as in the cpuv and gpuv
  tiers).  :func:`prove` runs the prover's side of ``run_query`` alone -- ``protocol._claims_message``,
  ``_fold_message`` and ``_open_message``, the functions ``run_query`` runs, so the messages, the byte counts and the
  phase timings are the same -- and returns the proof as its byte strings.  No check of the verifier's is in the
  timed window; one untimed ``run_query`` per cell (``--accept``) shows that the verifier accepts the same proof.

Batches (B > 1).  U1 and U2 serve the B prompts as one ``[B, T]`` forward pass.  The prover (``--batch-proof``):

* ``seq``: the B prompts are B independent queries (e.g. of B users).  The prover proves one statement per query
  today, so it proves them one after another, each proof the B = 1 proof of its prompt; a batched forward pass with
  per-query proofs (plan C4c) is not implemented;
* ``joint``: the B prompts as one statement, one client's batch -- the claims of all of them, one ``u`` and one
  opening for all (``run_query`` on ``[B, T]``, plan C4a's path).

Timing.  One warm-up of every variant (untimed; it also runs the checks), then ``--reps`` repetitions with the
variants interleaved, the order rotating from repetition to repetition (as ``perf.py``).  On a GPU each call is timed
with CUDA events, each recorded after a device synchronisation (the end event is waited for); on a CPU with
``perf_counter``.  A
time is reported as its median with a 95% percentile-bootstrap interval (:data:`BOOT` resamples, seeded), a ratio
(prover/U1, prover/U2, U1/U2) as the ratio of the medians, with an interval from the same bootstrap of both samples.

Phases.  The prover's timings are ``run_query``'s: ``prove_forward`` (the forward pass that computes the claims),
``prove_lookups`` (a commitment plan's lookup-table multiproofs), ``prove_encode`` (the PVC3 claims, the packed ``u``
and columns), ``fs_hash`` (the Fiat--Shamir absorption; none in Kpre, FS1), ``prove_fold`` (``u``) and ``prove_open``
(the opened columns and their paths), plus ``challenges``: the prover's derivation of the Fiat--Shamir challenges
from the transcript, which ``run_query`` books to neither party (:class:`TimedChallenges`).  The in-process run
also decodes ``u`` and the columns for the verifier (``verify_decode``), which is not the prover's work and is
subtracted: ``prove_s`` is the measured window minus ``verify_decode``.  ``residual`` is ``prove_s`` minus the sum
of the phases (Python, copies; the per-phase contention snapshots are off, :func:`quiet_contention`), and
``phases_ok`` holds when ``|residual| <= 2%`` of ``prove_s`` in the median repetition.  The proof-only overhead
``prove_s - U1`` is split into ``forward_extra`` (the prover's forward pass(es) minus U1: narrowing and keeping the
claims, and with ``seq`` B single-prompt passes against one batched pass), ``encode``, ``hash``, ``fold``, ``open``,
``lookups``, ``challenges`` and ``residual`` (medians: they add up to the overhead approximately).

Checks in every cell (a failure is recorded in the line, and the script exits with status 1 at the end):

* ``u1_equals_claims``: U1's logits (warm-up and every repetition) are ``torch.equal`` to the LM head's claims decoded
  from the proof's PVC3 bytes (``protocol._decoded_claims``, as the verifier decodes them); with ``seq``, each
  prompt's proof against its row of the batched U1, so batched serving and per-query proving give the same integers;
* ``deterministic``: proving the same prompts again gives byte-identical proofs (SHA-256 over every message: the
  warm-up's proofs against the first repetition's, prompt 0's with ``seq``).  Claims are a deterministic function of
  the model and the prompt and, under Fiat--Shamir, so are the challenges and every later message;
* ``accepted``: ``run_query`` accepts the proof of prompt 0 (``seq``) or of the batch, with the byte count of the
  proof timed here (``--no-accept`` skips it, and Kpre then needs no precompute);
* ``bytes_consistent``: the proof's byte strings add up to ``run_query``'s byte count.

Output: one JSON line per (cell, mode, batch proof), ``row = "prove"``, with the medians and intervals, the ratios,
the phases and the overhead split, the proof bytes, the proving time and proof MB per token, the egress cap (proved
tokens per second that 100 Gb/s carries: ``12.5e9`` bytes/s over the bytes per token) and the checks; and one line
per cell, ``row = "on_request"``: serving costs U1 (and its ratio to U2), and a proof produced later, only for the
queries a client asks about, costs ``prove_s`` and is byte-identical (``deterministic``, with its SHA-256).  A
variant that does not fit the GPU is recorded as ``skipped`` (out of memory), never extrapolated.  ``--table`` prints
such lines as a table, and for a (model, T, B) measured on two block counts the two-point extrapolation of U1 and U2 to
the full model (to check U2 against F1's numbers only; never a reported cell).
"""
from __future__ import annotations

import argparse
import contextlib
import dataclasses
import gc
import glob
import hashlib
import importlib.util
import json
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

from pvi.fullcheck import claimcodec, contention, protocol
from pvi.fullcheck.analytic import decoder_shapes
from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query
from pvi.fullcheck.transformer import CONFIGS, DecoderConfig, build_decoder

HERE = Path(__file__).resolve().parent
CODE = HERE.parents[1]          # code/

RATE = 4
"""The Reed--Solomon rate of the commitments (``bench.RATE``)."""
CALIB_TOKENS = 32
"""Calibration tokens of every build (``perf.py``'s ``min(seq, 32)`` for ``T >= 32``): one graph for every cell."""
WORKLOADS = {"a": [(64, 1), (512, 1), (2048, 1)],
             "b": [(64, 1), (64, 4), (64, 16), (64, 64), (2048, 1), (2048, 4)]}
"""``(T, B)`` of the plan's workloads: (a) prefill at B = 1; (b) batched prefill."""
EGRESS_BYTES_S = 100e9 / 8
"""100 Gb/s in bytes per second."""
RESIDUAL_MAX = 0.02
BOOT = 10_000
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
TINY = {
    "tiny-gpt": DecoderConfig("tiny-gpt", 64, 2, 4, 4, 16, 128, 97, max_pos=256),
    "tiny-opt": DecoderConfig("tiny-opt", 64, 2, 4, 4, 16, 128, 97, mlp="relu", max_pos=256),
    "tiny-llama": DecoderConfig("tiny-llama", 64, 2, 4, 4, 16, 160, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                                bias=False, tied=False),
    "tiny-qwen": DecoderConfig("tiny-qwen", 64, 2, 8, 2, 16, 160, 97, norm="rmsnorm", mlp="swiglu", pos="rope",
                               rope_theta=1e6, bias=False, qk_norm=True),
    "small-llama": DecoderConfig("small-llama", 256, 4, 8, 4, 32, 688, 4096, norm="rmsnorm", mlp="swiglu",
                                 pos="rope", bias=False, tied=False),
}
"""Small shapes for a laptop and the tests (``tiny-*`` as the test suites build them)."""
_OOM_TEXT = ("CUDA out of memory", "CUDA error: out of memory", "CUBLAS_STATUS_ALLOC_FAILED")
"""What a ``RuntimeError`` says when the GPU ran out of memory in an allocation that is not torch's caching
allocator's (a library's own: cuBLAS, a kernel launch); the allocator raises ``torch.cuda.OutOfMemoryError``."""
_BEGIN = contention.begin


def _is_oom(e: BaseException) -> bool:
    """Whether ``e`` is the GPU running out of memory (any other error propagates)."""
    return isinstance(e, torch.cuda.OutOfMemoryError) or (isinstance(e, RuntimeError)
                                                           and any(s in str(e) for s in _OOM_TEXT))


def config(name: str) -> DecoderConfig:
    return TINY[name] if name in TINY else CONFIGS[name]


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


# -- measurement ---------------------------------------------------------------------------------

def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed(fn, device: torch.device):
    """``(fn(), seconds, wait_frac)``: on a GPU between two CUDA events, the first recorded after a synchronisation
    and the second after ``fn`` returns and the device has finished every stream's work, waited for (so the window
    ends when the GPU has finished ``fn``'s work and holds the host work ``fn`` did meanwhile); on a CPU with
    ``perf_counter``.  ``wait_frac`` comes from a contention snapshot pair outside the window (``None`` where the
    kernel reports no scheduler statistics)."""
    c0 = _BEGIN()
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        result = fn()
        torch.cuda.synchronize(device)      # (work fn left on another stream; the end event sees its own only)
        end.record()
        end.synchronize()
        seconds = start.elapsed_time(end) / 1e3
    else:
        t0 = time.perf_counter()
        result = fn()
        seconds = time.perf_counter() - t0
    store: dict = {}
    contention.add(store, "window", c0)
    return result, seconds, (store.get("window") or {}).get("wait_frac")


def boot(xs, ys=None, n: int = BOOT, seed: int = 0) -> dict:
    """The median of ``xs`` (``{"median", "ci95"}``) or, with ``ys``, the ratio of the medians of ``xs`` and ``ys``
    (``{"ratio", "ci95"}``), with the 2.5% and 97.5% percentiles of ``n`` bootstrap resamples (each sample
    resampled with replacement on its own; seeded, so a line is reproducible from its times)."""
    rng = np.random.default_rng(seed)
    x = np.asarray(xs, dtype=np.float64)
    mx = np.median(x[rng.integers(0, x.size, (n, x.size))], axis=1)
    if ys is None:
        lo, hi = np.percentile(mx, [2.5, 97.5])
        return {"median": float(np.median(x)), "ci95": [float(lo), float(hi)]}
    y = np.asarray(ys, dtype=np.float64)
    my = np.median(y[rng.integers(0, y.size, (n, y.size))], axis=1)
    lo, hi = np.percentile(mx / my, [2.5, 97.5])
    return {"ratio": float(np.median(x) / np.median(y)), "ci95": [float(lo), float(hi)]}


@contextlib.contextmanager
def quiet_contention():
    """``contention.begin`` returns ``None`` inside (so ``contention.add`` records nothing): the per-phase snapshots
    of ``run_query``'s helpers read ``/proc`` outside each phase's window but inside the whole proof's.  They are
    measurement only; every timed call gets one snapshot pair of its own, outside its window (:func:`timed`)."""
    saved = contention.begin
    contention.begin = lambda: None
    try:
        yield
    finally:
        contention.begin = saved


def _gpu_peak_reset(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)


def _gpu_peak_mb(device: torch.device) -> float | None:
    return torch.cuda.max_memory_allocated(device) / 2 ** 20 if device.type == "cuda" else None


def _host_maxrss_mb() -> float | None:
    """The process's peak resident memory so far: ``ru_maxrss`` (KiB on Linux), on Windows the peak working set."""
    try:
        import resource
        return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    except ImportError:
        pass
    try:
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):       # PROCESS_MEMORY_COUNTERS
            _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                (f, ctypes.c_size_t) for f in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                               "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                               "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

        kernel32, psapi = ctypes.windll.kernel32, ctypes.windll.psapi
        kernel32.GetCurrentProcess.restype = wintypes.HANDLE
        psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        c = Counters()
        c.cb = ctypes.sizeof(Counters)
        if psapi.GetProcessMemoryInfo(kernel32.GetCurrentProcess(), ctypes.byref(c), c.cb):
            return c.PeakWorkingSetSize / 2 ** 20
    except (OSError, AttributeError):
        pass
    return None


def _free(device: torch.device) -> None:
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


# -- the three ways of answering a batch ----------------------------------------------------------

def _drop(z: torch.Tensor) -> None:
    """``IntGraph.forward``'s ``send``: nothing of a claim is kept (served without a proof)."""
    return None


def serve_u1(graph, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """U1: the logits of the integer graph on ``x`` (host token ids, moved to ``device`` inside, as the prover's are),
    every claim dropped as it is computed."""
    with torch.no_grad():
        env, _ = graph.forward(x.to(device), free=True, send=_drop)
    return env[graph.output_name]


def serve_u2(model, x: torch.Tensor, device: torch.device) -> torch.Tensor:
    """U2: the floating-point decoder's next-token logits on ``x`` (host token ids, moved to ``device`` inside)."""
    with torch.no_grad():
        return model(x.to(device))


@dataclasses.dataclass
class TimedChallenges(Verifier):
    """A :class:`Verifier` that also times its challenge derivations (``fold_challenges``, ``column_challenges``)
    into ``challenge_s``.  Under Fiat--Shamir the prover derives the same challenges from the transcript itself;
    ``run_query`` books that work to neither party."""

    challenge_s: float = 0.0

    def fold_challenges(self, ch: Challenger, x: torch.Tensor) -> dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        try:
            return super().fold_challenges(ch, x)
        finally:
            self.challenge_s += time.perf_counter() - t0

    def column_challenges(self, ch: Challenger) -> dict[str, torch.Tensor]:
        t0 = time.perf_counter()
        try:
            return super().column_challenges(ch)
        finally:
            self.challenge_s += time.perf_counter() - t0


def prove(prover: Prover, verifier: Verifier, x: torch.Tensor) -> tuple[list, dict | None, dict]:
    """The prover's side of ``run_query(prover, verifier, x, wire=True)`` alone: ``(parts, openings, out)``.

    ``parts`` are the proof's byte strings in the order they travel -- the PVC3 claims, a commitment plan's
    lookup-table rows (int8) and their multiproofs' hashes, and in mode C the packed ``u``; ``openings`` mode C's
    opened columns as the verifier decoded them, with their paths (their packed bytes are rebuilt by
    :func:`messages`, outside any timed window; ``None`` in Kpre); ``out`` the timings and byte counts as
    ``run_query`` records them, plus ``challenges`` (with a :class:`TimedChallenges` verifier).  The challenges are
    those ``run_query`` draws: they depend only on the transcript, which the verifier's checks do not touch.  Mode C
    needs Fiat--Shamir (a proof produced without a verifier); Kpre's one message reads no challenge."""
    if verifier.mode not in ("C", "Kpre"):
        raise ValueError(f"mode {verifier.mode}: the prover sends only the claims and, in mode C, u and the columns")
    if verifier.mode == "C" and not verifier.params.fiat_shamir:
        raise ValueError("mode C proves without a verifier only under Fiat-Shamir")
    ch = Challenger(fiat_shamir=verifier.params.fiat_shamir)
    out = {"timings": {}, "contention": {}}
    if isinstance(verifier, TimedChallenges):
        verifier.challenge_s = 0.0
    claims, rows, proofs = protocol._claims_message(prover, verifier, x, ch, out, {}, wire=True)
    hashes = b"".join(h for name in verifier.tables for h in protocol._proof_hashes(proofs.get(name)))
    parts, openings = [claims, rows or b"", hashes], None
    if verifier.mode == "C":
        _, us, u_blob = protocol._fold_message(prover, verifier, ch, x, out, wire=True)
        _, openings = protocol._open_message(prover, verifier, ch, us, out, u_blob=u_blob)
        parts.append(u_blob)
    out["timings"]["challenges"] = getattr(verifier, "challenge_s", 0.0)
    return parts, openings, out


def messages(parts: list, openings: dict | None, verifier: Verifier) -> list:
    """The proof's messages as byte strings, in the order they travel: ``parts`` and, in mode C, the opened columns
    31-bit packed -- ``claimcodec.pack_field`` of the decoded rows, tree after tree: the packing is a fixed function
    of the values, so these are the bytes ``_open_message`` sent -- and the hashes of their paths."""
    if openings is None:
        return list(parts)
    units = [name for name, _ in verifier._column_units()]
    cols = claimcodec.pack_field([openings[name][0] for name in units])
    paths = b"".join(h for name in units for h in protocol._proof_hashes(openings[name][1]))
    return [*parts, cols, paths]


def digest(msgs: list) -> str:
    """SHA-256 of the messages, each prefixed with its length."""
    h = hashlib.sha256()
    for m in msgs:
        h.update(len(m).to_bytes(8, "little"))
        h.update(m)
    return h.hexdigest()


def head_op(graph):
    return next(op for op in graph.mat_ops if op.output == graph.output_name)


def claimed_logits(verifier: Verifier, graph, parts: list, x: torch.Tensor) -> torch.Tensor:
    """The logits a proof claims: the LM head's claim decoded from the proof's PVC3 bytes (and a plan's rows) as
    the verifier decodes them (``protocol._decoded_claims``), as int64 ``[B * positions, vocab]``."""
    zs = protocol._decoded_claims(verifier, parts[0], parts[1], x, {"timings": {}, "contention": {}}, torch.int32)
    if zs is None:
        raise AssertionError("the proof's claims do not decode")
    return zs[head_op(graph).name].T.to("cpu", torch.int64)


def logit_rows(logits: torch.Tensor) -> torch.Tensor:
    """U1's logits ``[B, positions, vocab]`` as int64 host rows ``[B * positions, vocab]``."""
    if logits.dtype.is_floating_point:
        raise AssertionError("U1's logits must be integers")
    return logits.reshape(-1, logits.shape[-1]).to("cpu", torch.int64)


def queries(x: torch.Tensor, how: str) -> list[torch.Tensor]:
    """The statements a batch ``x [B, T]`` is proved as: one (``single``, ``joint``) or one per prompt (``seq``)."""
    return [x[i:i + 1] for i in range(x.shape[0])] if how == "seq" else [x]


def prove_batch(prover: Prover, verifier: Verifier, x: torch.Tensor, how: str, keep: int = 0):
    """``(outs, kept)``: the proofs of the batch ``x`` as ``how`` proves it (:func:`queries`), one after another --
    every proof's ``out`` and the ``(parts, openings)`` of the first ``keep`` (the others are dropped as soon as
    they are made)."""
    outs, kept = [], []
    for i, q in enumerate(queries(x, how)):
        parts, openings, out = prove(prover, verifier, q)
        outs.append(out)
        if i < keep:
            kept.append((parts, openings))
        del parts, openings
    return outs, kept


# -- one model -------------------------------------------------------------------------------------

PHASES = {"forward": "prove_forward", "lookups": "prove_lookups", "encode": "prove_encode", "hash": "fs_hash",
          "fold": "prove_fold", "open": "prove_open", "challenges": "challenges"}
"""The prover's phases (``run_query``'s timing names, and :func:`prove`'s ``challenges``)."""


def _phase_sums(outs: list[dict]) -> dict:
    """Every phase summed over the proofs of one call, and ``harness``: their ``verify_decode``."""
    s = {k: sum(o["timings"].get(v, 0.0) for o in outs) for k, v in PHASES.items()}
    s["harness"] = sum(o["timings"].get("verify_decode", 0.0) for o in outs)
    return s


def _bytes(outs: list[dict]) -> dict:
    keys = sorted({k for o in outs for k in o["bytes"]})
    return {k: sum(o["bytes"].get(k, 0) for o in outs) for k in keys}


def _cells(args) -> list[tuple[int, int]]:
    if args.cells:
        out = []
        for c in args.cells:
            t, _, b = c.lower().partition("x")
            out.append((int(t), int(b or 1)))
    else:
        out = [c for w in args.workload for c in WORKLOADS[w]]
    return list(dict.fromkeys(out))


def _tokens(cfg, seq: int, bsz: int) -> torch.Tensor:
    return torch.randint(0, cfg.vocab, (bsz, seq), generator=torch.Generator().manual_seed(1))


def _why(e: BaseException) -> str:
    """An exception as one short line, for a ``skipped`` field."""
    return f"{type(e).__name__}: {(str(e).splitlines() or [''])[0][:200]}"


def time_u2(cfg, layers: int, cells, dtypes: list[str], device: torch.device, reps: int) -> dict:
    """U2's times ``{(T, B): {dtype: {"times", "wait_frac", "gpu_peak_mb"} or {"skipped": why}}}``: per dtype one
    model, a warm-up and ``reps`` timed batched prefills per cell; the model is freed afterwards.  A dtype or cell
    the device cannot run (out of memory; bf16 GEMMs on a GPU without them) is recorded as skipped."""
    reexec = _load("reexec", HERE / "reexec.py")
    out: dict = {c: {} for c in cells}
    for name in dtypes:
        model, why = None, None
        try:
            model = reexec.FloatDecoder(cfg, layers, DTYPES[name], device, prune_last=True)
        except RuntimeError as e:          # (torch.cuda.OutOfMemoryError is one)
            why = _why(e)
        for seq, bsz in cells:
            if seq > cfg.max_pos:
                continue
            if model is None:
                out[seq, bsz][name] = {"skipped": why}
                continue
            x = _tokens(cfg, seq, bsz)
            failed = None
            try:
                _gpu_peak_reset(device)
                timed(lambda: serve_u2(model, x, device), device)
                peak = _gpu_peak_mb(device)
                times, waits = [], []
                for _ in range(reps):
                    _, s, w = timed(lambda: serve_u2(model, x, device), device)
                    times.append(s)
                    waits.append(w)
                out[seq, bsz][name] = {"times": times, "wait_frac": waits, "gpu_peak_mb": peak}
            except RuntimeError as e:
                failed = _why(e)
            if failed:
                out[seq, bsz][name] = {"skipped": failed}
                _free(device)
        del model
        _free(device)
    return out


@dataclasses.dataclass
class Built:
    """The integer model of one ``(model, layers)``, its provers and verifiers per mode, and its setup times."""

    name: str
    cfg: DecoderConfig
    layers: int
    graph: object
    provers: dict
    verifiers: dict
    setup: dict


def build(name: str, layers: int, args, device: torch.device) -> Built:
    """The pruned integer decoder of ``name`` on ``layers`` blocks (0: all of them), its provers and verifiers."""
    cfg = config(name)
    layers = layers or cfg.n_layers
    t0 = time.perf_counter()
    graph = build_decoder(cfg, n_layers=layers, calib_tokens=CALIB_TOKENS, seed=0, prune_last=True)
    setup = {"build_s": time.perf_counter() - t0}
    shapes = decoder_shapes(cfg, prune_last=True)
    n_checks = len(shapes)          # (r, t) for the full model's op count, as bench.py and perf.py
    provers, verifiers = {}, {}
    for mode in args.modes:
        if mode == "C":
            _sync(device)
            t0 = time.perf_counter()
            com = commit_graph(graph, RATE, device=device, policy=args.policy, model_ops=shapes)
            _sync(device)
            setup["commit_s"] = time.perf_counter() - t0
            params = params_for(args.lam, n_checks, rate=RATE, plan=com.plan, fiat_shamir=True)
            verifiers[mode] = TimedChallenges(graph.public(), params, "C", publics=com.publics,
                                              groups=com.group_publics, tables=com.table_publics, lean=True,
                                              device=args.verifier_device)
            provers[mode] = Prover(graph, device=device, commitments=com, lean=True)
            setup["chosen_policy"] = args.policy if com.plan is None else com.plan.policy
        else:
            params = params_for(args.lam, n_checks, rate=RATE, fiat_shamir=True)
            weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
            v = TimedChallenges(graph.public(), params, "Kpre", weights=weights, lean=True,
                                device=args.verifier_device, lookups=args.lookups)
            if args.accept:              # the verifier's secret chi and u (only the acceptance check reads them)
                t0 = time.perf_counter()
                v.precompute(Challenger())
                setup["precompute_s"] = time.perf_counter() - t0
            verifiers[mode], provers[mode] = v, Prover(graph, device=device, lean=True)
        setup[f"reps_{mode}"], setup[f"columns_{mode}"] = verifiers[mode].params.reps, verifiers[mode].params.columns
    return Built(name, cfg, layers, graph, provers, verifiers, setup)


class Cell:
    """One ``(T, B)`` of a built model: the variants ``"u1"`` and ``(mode, how)``, their warm-ups with the checks,
    the interleaved repetitions and the records."""

    def __init__(self, m: Built, seq: int, bsz: int, args, device: torch.device):
        self.m, self.seq, self.bsz, self.args, self.device = m, seq, bsz, args, device
        self.x = _tokens(m.cfg, seq, bsz)
        hows = ["single"] if bsz == 1 else list(args.batch_proof)
        self.variants = ["u1"] + [(mode, how) for mode in args.modes for how in hows]
        self.times = {k: [] for k in self.variants}
        self.waits = {k: [] for k in self.variants}
        self.phases = {k: [] for k in self.variants if k != "u1"}
        self.skipped: dict = {}
        self.peak: dict = {}
        self.checks = {k: {} for k in self.variants if k != "u1"}
        self.digests: dict = {}     # (mode, how) -> the warm-up's per-proof SHA-256
        self.nbytes: dict = {}      # (mode, how) -> the warm-up's per-proof byte counts (run_query's)
        self.byte_parts: dict = {}  # (mode, how) -> the warm-up's byte counts by message, summed over its proofs
        self.u1_rows = None
        self.u1_equal: list[bool] = []   # per timed repetition of U1: its logits are the warm-up's (the claimed ones)

    # the calls -----------------------------------------------------------------------------------
    def _call(self, key, keep: int = 0):
        m = self.m
        if key == "u1":
            return serve_u1(m.graph, self.x, self.device)
        mode, how = key
        return prove_batch(m.provers[mode], m.verifiers[mode], self.x, how, keep)

    def _run(self, key, keep: int = 0):
        """One timed call of ``key`` (``None`` and the variant dropped if it runs out of GPU memory)."""
        oom = False
        try:
            with quiet_contention():
                return timed(lambda: self._call(key, keep), self.device)
        except RuntimeError as e:            # (the traceback, and the tensors its frames hold, go with the block)
            if not _is_oom(e):
                raise
            oom = True
        if oom:
            self.skipped[key] = "out of memory"
            _free(self.device)
        return None

    # the warm-up, with the checks ------------------------------------------------------------------
    def warm_up(self) -> None:
        for key in self.variants:
            if key != "u1" and "u1" in self.skipped:
                self.skipped[key] = "U1 did not fit, so there is nothing to check its logits against"
                continue
            _gpu_peak_reset(self.device)
            if key == "u1":
                r = self._run(key)
                if r is not None:
                    self.u1_rows = logit_rows(r[0])
                    self.peak[key] = _gpu_peak_mb(self.device)
                continue
            if not self._warm_prove(key):
                continue
            self.peak[key] = _gpu_peak_mb(self.device)

    def _warm_prove(self, key) -> bool:
        """The warm-up of a prover variant, untimed, proof by proof: each proof's claimed logits against U1's rows,
        its SHA-256 and byte counts kept, the proof dropped before the next one."""
        mode, how = key
        m, v = self.m, self.m.verifiers[key[0]]
        equal, consistent, digests, counts, outs = True, True, [], [], []
        oom = False
        try:
            with quiet_contention():
                row = 0
                for q in queries(self.x, how):
                    parts, openings, out = prove(m.provers[mode], v, q)
                    claimed = claimed_logits(v, m.graph, parts, q)
                    equal &= torch.equal(claimed, self.u1_rows[row:row + claimed.shape[0]])
                    row += claimed.shape[0]
                    msgs = messages(parts, openings, v)
                    consistent &= sum(len(b) for b in msgs) == sum(out["bytes"].values())
                    digests.append(digest(msgs))
                    counts.append(sum(out["bytes"].values()))
                    outs.append(out)
                    del parts, openings, msgs, claimed
                equal &= row == self.u1_rows.shape[0]
        except RuntimeError as e:
            if not _is_oom(e):
                raise
            oom = True
        if oom:
            self.skipped[key] = "out of memory"
            _free(self.device)
            return False
        self.checks[key].update(u1_equals_claims=bool(equal), bytes_consistent=bool(consistent))
        self.digests[key], self.nbytes[key], self.byte_parts[key] = digests, counts, _bytes(outs)
        return True

    # the repetitions -------------------------------------------------------------------------------
    def repeat(self) -> None:
        active = [k for k in self.variants if k not in self.skipped]
        for i in range(self.args.reps):
            for key in active[i % len(active):] + active[:i % len(active)]:
                if key in self.skipped:
                    continue
                r = self._run(key, keep=1 if (i == 0 and key != "u1") else 0)
                if r is None:
                    continue
                result, seconds, wait = r
                del r                                # (the kept proof must not outlive this repetition)
                self.times[key].append(seconds)
                self.waits[key].append(wait)
                if key == "u1":                      # every repetition's logits: still the claimed ones
                    self.u1_equal.append(torch.equal(logit_rows(result), self.u1_rows))
                    del result
                    continue
                outs, kept = result
                self.phases[key].append(_phase_sums(outs))
                if kept:
                    msgs = messages(*kept[0], self.m.verifiers[key[0]])
                    self.checks[key]["deterministic"] = digest(msgs) == self.digests[key][0]
                    del msgs
                del outs, kept, result
        for key in self.checks:
            if key not in self.skipped:
                self.checks[key]["u1_equals_claims"] = (self.checks[key].get("u1_equals_claims", False)
                                                        and bool(self.u1_equal) and all(self.u1_equal))

    def accept(self) -> None:
        """One untimed ``run_query`` per prover variant: the proof of prompt 0 (``seq``) or of the batch is
        accepted, with the byte count of the proof timed here."""
        for key in self.checks:
            if key in self.skipped:
                continue
            mode, how = key
            q = queries(self.x, how)[0]
            oom = False
            try:
                res = run_query(self.m.provers[mode], self.m.verifiers[mode], q, wire=True)
            except RuntimeError as e:
                if not _is_oom(e):
                    raise
                oom = True
            if oom:
                self.checks[key]["accepted"] = None
                self.checks[key]["accept_skipped"] = "out of memory"
                _free(self.device)
                continue
            self.checks[key]["accepted"] = bool(res["accepted"])
            self.checks[key]["accept_rejected_at"] = res["rejected_at"]
            self.checks[key]["accept_same_bytes"] = sum(res["bytes"].values()) == self.nbytes[key][0]
            del res


def _ok(checks: dict) -> bool:
    """Every correctness check of a variant passed (``accepted`` may be absent: ``--no-accept``)."""
    need = ("u1_equals_claims", "bytes_consistent", "deterministic")
    good = all(checks.get(k) is True for k in need)
    if "accepted" in checks:
        good &= checks["accepted"] is True and checks.get("accept_same_bytes") is True
    return good


def _lines(cell: Cell, u2: dict, base: dict) -> list[dict]:
    """The cell's ``prove`` lines and its ``on_request`` line."""
    tokens = cell.seq * cell.bsz
    u1 = cell.times["u1"]
    u2_ok = {k: v["times"] for k, v in u2.items() if "times" in v}
    common = {**base, "seq": cell.seq, "batch": cell.bsz, "tokens": tokens}
    u1_rec = boot(u1) if u1 else None
    lines = []
    for key in cell.variants[1:]:
        mode, how = key
        line = {**common, "row": "prove", "mode": mode, "batch_proof": how,
                "proofs": cell.bsz if how == "seq" else 1}
        if key in cell.skipped or "u1" in cell.skipped or not cell.times[key] or not u1:
            line["skipped"] = cell.skipped.get(key) or cell.skipped.get("u1") or "no repetition"
            lines.append(line)
            continue
        ph = cell.phases[key]
        total = cell.times[key]
        prove_s = [t - p["harness"] for t, p in zip(total, ph)]
        summed = [sum(p[k] for k in PHASES) for p in ph]
        resid = [(s - q) / s for s, q in zip(prove_s, summed)]
        med = {k: float(np.median([p[k] for p in ph])) for k in (*PHASES, "harness")}
        p_med = float(np.median(prove_s))
        nbytes = _bytes_of(cell, key)
        per_tok = nbytes / tokens
        line.update({
            "u1_s": u1_rec, "u2_s": {k: boot(v) for k, v in u2_ok.items()},
            "window_s": boot(total), "prove_s": boot(prove_s),
            "prover_over_u1": boot(prove_s, u1),
            "prover_over_u2": {k: boot(prove_s, v) for k, v in u2_ok.items()},
            "u1_over_u2": {k: boot(u1, v) for k, v in u2_ok.items()},
            "phases_s": {**{k: med[k] for k in PHASES}, "harness_decode": med["harness"],
                         "residual": float(np.median([s - q for s, q in zip(prove_s, summed)]))},
            "residual_frac": {"median": float(np.median(resid)), "max_abs": float(np.max(np.abs(resid)))},
            "phases_ok": bool(abs(float(np.median(resid))) <= RESIDUAL_MAX),
            "overhead_s": p_med - float(np.median(u1)),
            "overhead_split_s": {"forward_extra": med["forward"] - float(np.median(u1)),
                                 **{k: med[k] for k in PHASES if k != "forward"},
                                 "residual": float(np.median([s - q for s, q in zip(prove_s, summed)]))},
            "proof_bytes": nbytes, "bytes": cell.byte_parts.get(key),
            "proof_mb_per_token": per_tok / 1e6, "prove_s_per_token": p_med / tokens,
            "proved_tokens_per_s": tokens / p_med,
            "served_tokens_per_s": {"u1": tokens / float(np.median(u1)),
                                    **{k: tokens / float(np.median(v)) for k, v in u2_ok.items()}},
            "egress_cap_tokens_per_s_per_100gbps": EGRESS_BYTES_S / per_tok,
            "checks": cell.checks[key], "ok": _ok(cell.checks[key]),
            "proof_sha256": cell.digests[key][0],
            "gpu_peak_mb": {"u1": cell.peak.get("u1"), "prover": cell.peak.get(key),
                            **{k: v.get("gpu_peak_mb") for k, v in u2.items()}},
            "wait_frac_max": {"u1": _max(cell.waits["u1"]), "prover": _max(cell.waits[key])},
            "times_s": {"u1": u1, "window": total, "prove": prove_s, **{k: v for k, v in u2_ok.items()}},
        })
        lines.append(line)
    req = {**common, "row": "on_request", "serve_s": u1_rec,
           "serve_over_u2": {k: boot(u1, v) for k, v in u2_ok.items()} if u1 else {},
           "u2_s": {k: boot(v) for k, v in u2_ok.items()},
           "served_tokens_per_s": tokens / float(np.median(u1)) if u1 else None, "later_proof": {}}
    for line in lines:
        if "skipped" in line:
            continue
        req["later_proof"][f"{line['mode']}/{line['batch_proof']}"] = {
            "prove_s": line["prove_s"], "proof_bytes": line["proof_bytes"], "sha256": line["proof_sha256"],
            "deterministic": line["checks"].get("deterministic"), "prover_over_u1": line["prover_over_u1"]}
    u2_skip = {k: v["skipped"] for k, v in u2.items() if "skipped" in v}
    if u2_skip:
        req["u2_skipped"] = u2_skip
    return lines + [req]


def _bytes_of(cell: Cell, key) -> int:
    return int(sum(cell.nbytes[key]))


def _max(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def run_model(name: str, args, device: torch.device, cells, base: dict) -> bool:
    """Every cell of one ``(model, layers)``: U2 first (its model freed), then the integer model's U1 and provers.
    Prints the lines; ``False`` if a correctness check failed."""
    cfg = config(name)
    layers = args.layers or cfg.n_layers
    cells = [c for c in cells if c[0] <= cfg.max_pos]
    u2 = time_u2(cfg, layers, cells, args.u2_dtype, device, args.reps)
    m = build(name, layers, args, device)
    good = True
    base = {**base, "model": name, "layers": args.layers or None, "n_layers": layers, "n_layers_full": cfg.n_layers,
            "policy": args.policy, "lookups": args.lookups, **m.setup}
    for seq, bsz in cells:
        cell = Cell(m, seq, bsz, args, device)
        cell.warm_up()
        cell.repeat()
        if args.accept:
            cell.accept()
        for line in _lines(cell, u2.get((seq, bsz), {}), base):
            line["host_maxrss_mb"] = _host_maxrss_mb()
            print(json.dumps(line), flush=True)
            if line["row"] == "prove" and "skipped" not in line:
                good &= line["ok"]
                if not line["phases_ok"]:
                    print(f"warning: {name} T={seq} B={bsz} {line['mode']}/{line['batch_proof']}: the phases leave "
                          f"{100 * line['residual_frac']['median']:.1f}% of the prover's time unaccounted",
                          file=sys.stderr)
        del cell
        _free(device)
    del m
    _free(device)
    return good


# -- the table -------------------------------------------------------------------------------------

def _fmt(r: dict | None, key: str = "median", scale: float = 1.0, digits: int = 3) -> str:
    if not r:
        return "-"
    lo, hi = r["ci95"]
    return f"{r[key] * scale:.{digits}g} [{lo * scale:.{digits}g}, {hi * scale:.{digits}g}]"


def table(paths: list[str]) -> None:
    """The ``prove`` lines of serve_ab.py logs as a table (times in ms), then the two-point extrapolations of U1 and
    U2 to the full model where a (model, T, B) was measured on two block counts."""
    lines = []
    for pattern in paths:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, encoding="utf-8") as fh:
                for raw in fh:
                    if raw.startswith("{"):
                        rec = json.loads(raw)
                        if rec.get("row") in ("prove", "on_request"):
                            lines.append(rec)
    head = ("model", "L", "T", "B", "mode", "how", "U2 ms", "U1 ms", "prover ms", "P/U1", "P/U2", "MB/tok",
            "cap tok/s", "ok", "phases")
    print("\t".join(head))
    for r in lines:
        if r["row"] != "prove":
            continue
        if "skipped" in r:
            print("\t".join(map(str, (r["model"], r["n_layers"], r["seq"], r["batch"], r["mode"], r["batch_proof"],
                                      "skipped: " + r["skipped"]))))
            continue
        d = next(iter(r["u2_s"]), None)
        print("\t".join(map(str, (
            r["model"], r["n_layers"], r["seq"], r["batch"], r["mode"], r["batch_proof"],
            f"{d}: {_fmt(r['u2_s'].get(d), scale=1e3)}" if d else "-", _fmt(r["u1_s"], scale=1e3),
            _fmt(r["prove_s"], scale=1e3), _fmt(r["prover_over_u1"], "ratio"),
            _fmt(r["prover_over_u2"].get(d), "ratio") if d else "-", f"{r['proof_mb_per_token']:.3g}",
            f"{r['egress_cap_tokens_per_s_per_100gbps']:.4g}", r["ok"],
            f"{100 * r['residual_frac']['median']:+.2f}%"))))
    by = {}
    for r in lines:
        if r["row"] == "on_request" and r.get("serve_s"):
            by.setdefault((r["model"], r["seq"], r["batch"]), {})[r["n_layers"]] = r
    for (model, seq, bsz), runs in sorted(by.items()):
        if len(runs) < 2:
            continue
        (l1, a), (l2, b) = sorted(runs.items())[:2]
        full = a["n_layers_full"]

        def ext(t1, t2):
            per = (t2 - t1) / (l2 - l1)
            return t1 + (full - l1) * per

        parts = [f"U1 {1e3 * ext(a['serve_s']['median'], b['serve_s']['median']):.4g} ms"]
        for k in a.get("u2_s", {}):
            if k in b.get("u2_s", {}):
                parts.append(f"U2 {k} {1e3 * ext(a['u2_s'][k]['median'], b['u2_s'][k]['median']):.4g} ms")
        print(f"extrapolated (check only) {model} T={seq} B={bsz} from {l1} and {l2} to {full} blocks: "
              + ", ".join(parts))


# -- main ------------------------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--model", nargs="+", default=[], help="CONFIGS names, or " + ", ".join(TINY))
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: the full model)")
    ap.add_argument("--workload", nargs="+", choices=sorted(WORKLOADS), default=["a", "b"],
                    help="(a) T = 64, 512, 2048 at B = 1; (b) T = 64 with B = 1, 4, 16, 64 and T = 2048 with B = 1, 4")
    ap.add_argument("--cells", nargs="+", default=[], help="TxB cells instead of --workload, e.g. 64x1 64x4")
    ap.add_argument("--modes", nargs="+", choices=["C", "Kpre"], default=["C", "Kpre"])
    ap.add_argument("--batch-proof", nargs="+", choices=["seq", "joint"], default=["seq"],
                    help="B > 1: B independent proofs one after another (seq) and/or one statement (joint)")
    ap.add_argument("--policy", default="paper", help="mode C's commitment plan (pvi.fullcheck.plans), as perf.py")
    ap.add_argument("--lookups", action=argparse.BooleanOptionalAction, default=True,
                    help="Kpre: the verifier reads the embedding rows itself (Verifier(lookups=True))")
    ap.add_argument("--lam", type=int, default=128)
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--u2-dtype", nargs="+", choices=sorted(DTYPES), default=None,
                    help="U2's dtypes (default fp16 on a GPU, fp32 on a CPU)")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--verifier-device", default="cpu", help="the decoding and acceptance checks' verifier")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--accept", action=argparse.BooleanOptionalAction, default=True,
                    help="one untimed run_query per prover variant and cell (Kpre: the verifier's precompute)")
    ap.add_argument("--label", default="")
    ap.add_argument("--table", nargs="+", default=None, help="print serve_ab.py logs (JSON lines) as a table")
    args = ap.parse_args()
    if args.table:
        table(args.table)
        return 0
    if not args.model:
        ap.error("--model is required")
    if args.reps < 2:
        ap.error("--reps must be at least 2 (the first repetition is compared with the warm-up)")
    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    args.u2_dtype = args.u2_dtype or (["fp16"] if device.type == "cuda" else ["fp32"])
    cells = _cells(args)
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True,
                         cwd=CODE).stdout.strip()
    base = {"label": args.label, "git": sha, "device": str(device), "verifier_device": args.verifier_device,
            "threads": args.threads, "reps": args.reps, "warmup": 1, "lam": args.lam, "rate": RATE, "fs": True,
            "wire": True, "lean": True, "prune_last": True,
            "timer": "cuda_events" if device.type == "cuda" else "perf_counter",
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cpu": platform.processor() or platform.machine(), "host": platform.node(), "torch": torch.__version__}
    good = True
    for name in args.model:
        good &= run_model(name, args, device, cells, base)
    if not good:
        print("A CORRECTNESS CHECK FAILED (see the lines whose ok is false)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""The prover against unverified serving: what proving costs a provider on the GPU that serves the model.

Produces the serving overhead of the evaluation (Sec. ``sec:eval-cost``, "Base costs": proving a 2,048-token prompt
against our exact integer forward pass without proofs and against an fp16 one, and 64 prompts proved together), the
provider's cost over unverified serving in note f of ``tab:options``, and the extended version's proof on request
(``ext:disc-options``: a provider serves with the exact model and proves later, with the same proof, the queries a
client asks about); the run package's ``serve`` tier.

    python -m experiments.deployment.serve_ab --model llama2-7b --workload a b --batch-proof seq joint \\
        --u2-dtype fp16 bf16 --label serve
    python -m experiments.deployment.serve_ab --model llama2-7b --layers 2 --cells 64x1 2048x4 --batch-proof seq joint
    python -m experiments.deployment.serve_ab --model llama2-7b --real meta-llama/Llama-2-7b-hf --workload a b \\
        --batch-proof seq joint --label serve-real
    python -m experiments.deployment.serve_ab --model tiny-llama --cells 32x1 32x4 --device cpu --threads 2
        # a laptop check (seconds)
    python -m experiments.deployment.serve_ab --table logs/<platform>/sp-serve-*.out

A cell is a model, a prompt length ``T`` and a batch ``B``: ``B`` prompts of ``T`` tokens, random ids drawn with
``--prompt-seed`` as ``costs/perf.py`` and ``reexec/reexec.py`` draw them (so prompt 0 of every cell is the B = 1
prompt of its ``T``).  The decoders have the shapes of ``pvi.models.decoder.CONFIGS`` and random weights (the costs
depend on the shapes only), built as ``perf.py`` builds them (``--seed``, ``--calib-tokens`` calibration tokens: one
graph for every cell), on ``--layers`` blocks (0: the full model); the small shapes of
:data:`experiments.common.models.TINY` are for a laptop and the tests.  With ``--real`` (a Hugging Face checkpoint;
one ``--model``, which names its shapes) the integer model is the checkpoint's integer graph instead, built as
``perf.py --real`` builds it (:mod:`experiments.common.real_cells`: an OPT checkpoint with
``hf_opt.CERTIFIED_RECIPE``, a Llama-family checkpoint, Qwen3 included, with ``hf_llama.MAIN_RECIPE`` and its fixed
public prefix; ``--layers n``: the checkpoint's first ``n`` blocks, the full graph's first ``n``), and the prompts are the WikiText-2 test split's windows from its start (``perf.py --real``'s queries: the
``B`` prompts of a cell are the first ``B`` windows of ``T`` tokens, so prompt 0 is again the B = 1 prompt).  The
checkpoint is loaded in fp32, calibrated and turned into the integer graph before anything is timed, and its fp32
model freed.  U2 keeps random weights of the checkpoint's shapes (its dense kernels' time depends on the shapes
only) and runs no fixed prefix, whose keys and values the real Llama-family graph's attention also reads.  Refused
with ``--real``: GPT-2 (no integer builder of a real checkpoint), a cell longer than the checkpoint's trained context,
an OPT model of fewer blocks than the checkpoint's, and an attention probabilities' scale other than the paper's.
Every cell times, on the same token ids, shapes, device and host:

* **U1**, our exact integer model served without proofs: ``IntGraph.forward(free=True)`` of the integer graph, every
  claim dropped as it is computed (``send`` keeps nothing).  It computes the integers the prover claims -- the logits
  are checked equal -- so prover/U1 is the cost of the proof itself.
* **U2**, an eager floating-point decoder of the same shapes: ``reexec.FloatDecoder`` (``F.linear`` and
  ``scaled_dot_product_attention``) in each ``--u2-dtype`` (fp16 and/or bf16 on a GPU; fp32 on a CPU), one batched
  prefill of the B prompts, its last block pruned like the integer graph's.  It skips the rotary embedding (an
  element-wise cost), so it is if anything faster than a complete implementation.  U2's model is built and timed
  first, and freed before the integer model goes to the device: the two need not fit at once.
* **the prover** in each of ``--modes`` (C, Kpre) under the protocol flags, which default to the paper's
  configuration (:data:`experiments.common.cli.PAPER`): Fiat--Shamir, the compact wire encoding, the lean prover, the
  last block pruned to the last position, mode C under the commitment plan ``cnn18c``, Kpre with the verifier
  reading the embedding rows itself, 11-bit attention probabilities (``--prob-max``, also U1's graph).
  :func:`prove` runs the prover's side of ``run_query`` alone -- ``v0.query._claims_message``, ``_fold_message`` and
  ``open_message``, the functions ``run_query`` runs, so the messages, the byte counts and the phase timings are the
  same -- and returns the proof's messages.  No check of the
  verifier's is in the timed window; one untimed ``run_query`` per cell (``--accept``) shows that the verifier
  accepts the same proof.  A proof that no verifier attends needs non-interactive challenges, so mode C needs
  Fiat--Shamir (``--no-fs`` is for Kpre, whose one message reads no challenge).

Batches (B > 1).  U1 and U2 serve the B prompts as one ``[B, T]`` forward pass.  The prover (``--batch-proof``):

* ``seq``: the B prompts are B independent queries (e.g. of B users).  The prover proves one statement per query, so
  it proves them one after another, each proof the B = 1 proof of its prompt (a batched forward pass with per-query
  proofs is not implemented);
* ``joint``: the B prompts as one statement, one client's batch -- the claims of all of them, one ``u`` and one
  opening for all (``run_query`` on ``[B, T]``).

Timing.  One warm-up of every variant (untimed; it also runs the checks), then ``--reps`` repetitions with the
variants interleaved, the order rotating from repetition to repetition (as ``perf.py``), each call timed by
:func:`experiments.common.records.timed` (CUDA events on a GPU, ``perf_counter`` on a CPU).  A time is reported as
its median with a 95% percentile-bootstrap interval (:func:`experiments.common.stats.median_ci`, ``--boot``
resamples, seeded), a ratio (prover/U1, prover/U2, U1/U2) as the ratio of the medians, with an interval from the same
bootstrap of both samples.

Phases.  The prover's timings are ``run_query``'s: ``prove_forward`` (the forward pass that computes the claims),
``prove_lookups`` (a commitment plan's lookup-table multiproofs), ``prove_encode`` (the compact encoding of the
claims, ``u`` and the columns), ``fs_hash`` (the Fiat--Shamir absorption; none in Kpre, whose one message to a
designated verifier reads no challenge), ``prove_fold`` (``u``) and ``prove_open`` (the opened columns and their
paths), plus ``challenges``: the prover's derivation of the Fiat--Shamir challenges from the transcript, which
``run_query`` books to neither party (:class:`TimedChallenges`).  The in-process run also decodes ``u`` and the
columns for the verifier (``verify_decode``), which is not the prover's work and is subtracted: ``prove_s`` is the
measured window minus ``verify_decode``.  ``residual`` is ``prove_s`` minus the sum of the phases (Python, copies; the
per-phase contention snapshots are off, :func:`pvi.core.contention.paused`), and ``phases_ok`` holds when ``|residual|`` is at
most ``--residual-max`` (2%) of ``prove_s`` in the median repetition.  The proof-only overhead ``prove_s - U1`` is
split into ``forward_extra`` (the prover's forward pass(es) minus U1: narrowing and keeping the claims, and with
``seq`` B single-prompt passes against one batched pass), ``encode``, ``hash``, ``fold``, ``open``, ``lookups``,
``challenges`` and ``residual`` (medians: they add up to the overhead approximately).

Checks in every cell (a failure is recorded in the line, and the script exits with status 1 at the end):

* ``u1_equals_claims``: U1's logits (warm-up and every repetition) are ``torch.equal`` to the LM head's claims as the
  verifier reads them from the proof (decoded from the compact encoding with ``v0.query.decoded_claims``); with
  ``seq``, each prompt's proof against its row of the batched U1, so batched serving and per-query proving give the
  same integers;
* ``deterministic``: proving the same prompts again gives byte-identical proofs (SHA-256 over every message: the
  warm-up's proofs against the first repetition's, prompt 0's with ``seq``).  Claims are a deterministic function of
  the model and the prompt and, under Fiat--Shamir, so are the challenges and every later message;
* ``accepted``: ``run_query`` accepts the proof of prompt 0 (``seq``) or of the batch, with the byte count of the
  proof timed here (``--no-accept`` skips it, and Kpre then needs no precompute);
* ``bytes_consistent``: the proof's messages add up to ``run_query``'s byte count.

Output: one JSON line per (cell, mode, batch proof), ``row = "prove"``, with the medians and intervals (``u1_s``,
``u2_s``, ``window_s``, ``prove_s``), the ratios (``prover_over_u1``, ``prover_over_u2``, ``u1_over_u2``), the phases
(``phases_s``, ``residual_frac``, ``phases_ok``) and the overhead split (``overhead_s``, ``overhead_split_s``), the
proof bytes (``proof_bytes``, ``bytes`` by message), ``proof_mb_per_token``, ``prove_s_per_token``, the proved and
served tokens per second, the egress cap (``egress_cap_tokens_per_s_per_100gbps``: proved tokens per second that
100 Gb/s carries, :data:`EGRESS_BYTES_S` over the bytes per token), the checks and ``ok``, ``proof_sha256``, the peak
GPU memory of each variant (``gpu_peak_mb``), the largest share of a timed window spent waiting for a CPU
(``wait_frac_max``), every time (``times_s``), the GPU's energy per call of each variant (``energy_j``: U1, the
prover's window and each U2 dtype, ``{"joules", "method", "calls"}`` from NVML's energy counter,
:mod:`experiments.common.energy`: over at least ``--energy-s`` seconds of back-to-back calls for a call shorter than
that, untimed and after the repetitions, else the median of the timed repetitions' own differences; None without a
counter, and the record's ``energy_counter`` says why) and the host's peak resident memory (``host_maxrss_mb``); and one line
per cell, ``row = "on_request"``: serving costs U1 (``serve_s``, and its ratio to U2), and a proof produced later,
only for the queries a client asks about, costs ``prove_s`` and is byte-identical (``later_proof``: ``deterministic``,
with its SHA-256).  Every line carries the run's settings, ``real`` (the checkpoint; None for random weights) and
``recipe`` (``certified`` or ``main``; None for random weights), the setup times (``build_s``, ``commit_s``,
``setup_cache``, ``precompute_s``; with ``--real`` also ``load_s``, loading the checkpoint in fp32 and its
calibration, which ``build_s``, the integer graph's own calibration and build, does not include) and ``env``
(:func:`experiments.common.records.environment`).  A variant that does not fit the GPU is recorded as ``skipped`` (out
of memory), never extrapolated.  ``--table`` prints such lines as a
table, and for a (model, T, B) measured on two block counts the two-point extrapolation of U1 and U2 to the full model
(a consistency check against ``reexec.py``'s extrapolated re-execution; never a reported cell).

Hardware and time: the package runs every table model on an NVIDIA L40S (48 GB) with 8 CPU threads, one job per
model with limits of 3 hours (OPT-1.3B) to 8 hours (Llama-2-13B) and 128-256 GB of host memory: a joint proof of
4 x 2,048 tokens holds tens of GB of claims, and its CPU acceptance check the decoded claims and every op's input.
A ``tiny-*`` cell on a laptop CPU takes seconds.  With ``--real`` the job also holds the checkpoint in fp32 while
its graph is built (about 4 bytes per parameter: 27 GB for Llama-2-7B and OPT-6.7B, 52 GB for Llama-2-13B, 16 GB for
Qwen3-4B), and its calibration's forward passes on the CPU add tens of minutes to a 7B or 13B job.
"""
from __future__ import annotations

import argparse
import dataclasses
import gc
import glob
import hashlib
import json
import platform
import sys
import time

import numpy as np
import torch

from experiments.common import checkpoints, cli, real_cells, records
from experiments.common.energy import EnergyMeter, joules_per_call
from experiments.common.models import TINY, decoder_config
from experiments.common.paths import CODE
from experiments.common.stats import median_ci
from experiments.reexec import reexec
from pvi.commit.cost import decoder_shapes
from pvi.core import contention
from pvi.models.decoder import DecoderConfig, build_decoder
from pvi.v0 import claimcodec
from pvi.v0 import query as v0_query
from pvi.v0.graph_commitment import commit_graph
from pvi.v0.params import params_for
from pvi.v0.prover import Prover
from pvi.v0.query import run_query
from pvi.v0.transcript import Challenger
from pvi.v0.verifier import Verifier

WORKLOADS = {"a": [(64, 1), (512, 1), (2048, 1)],
             "b": [(64, 1), (64, 4), (64, 16), (64, 64), (2048, 1), (2048, 4)]}
"""``(T, B)`` of the two workloads (``--workload``): (a) prefill at B = 1; (b) batched prefill."""
EGRESS_BYTES_S = 100e9 / 8
"""100 Gb/s in bytes per second: the link of the egress cap (``egress_cap_tokens_per_s_per_100gbps``)."""
DTYPES = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
_OOM_TEXT = ("CUDA out of memory", "CUDA error: out of memory", "CUBLAS_STATUS_ALLOC_FAILED")
"""What a ``RuntimeError`` says when the GPU ran out of memory in an allocation that is not torch's caching
allocator's (a library's own: cuBLAS, a kernel launch); the allocator raises ``torch.cuda.OutOfMemoryError``."""


def _is_oom(e: BaseException) -> bool:
    """Whether ``e`` is the GPU running out of memory (any other error propagates)."""
    return isinstance(e, torch.cuda.OutOfMemoryError) or (isinstance(e, RuntimeError)
                                                           and any(s in str(e) for s in _OOM_TEXT))


# -- measurement ---------------------------------------------------------------------------------

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


def prove(prover: Prover, verifier: Verifier, x: torch.Tensor, wire: bool = True) -> tuple[list, dict | None, dict]:
    """The prover's side of ``run_query(prover, verifier, x, wire=wire)`` alone: ``(parts, openings, out)``.

    ``parts`` are the proof's messages in the order they travel -- the claims (with ``wire`` their compact encoding,
    and a commitment plan's lookup-table rows as int8 bytes; without it the claims themselves and no rows), the
    hashes of the lookup tables' multiproofs, and in mode C ``u`` (packed with ``wire``, else the tensors);
    ``openings`` mode C's opened columns (with ``wire`` as the verifier decoded them) with their paths (their bytes
    are rebuilt by :func:`messages`, outside any timed window; ``None`` in Kpre); ``out`` the timings and byte counts
    as ``run_query`` records them, plus ``challenges`` (with a :class:`TimedChallenges` verifier).  The challenges
    are those ``run_query`` draws: they depend only on the transcript, which the verifier's checks do not touch.
    Mode C needs Fiat--Shamir (a proof produced without a verifier); Kpre's one message reads no challenge."""
    if verifier.mode not in ("C", "Kpre"):
        raise ValueError(f"mode {verifier.mode}: the prover sends only the claims and, in mode C, u and the columns")
    if verifier.mode == "C" and not verifier.params.fiat_shamir:
        raise ValueError("mode C proves without a verifier only under Fiat-Shamir")
    ch = Challenger(fiat_shamir=verifier.params.fiat_shamir)
    out = {"timings": {}, "contention": {}}
    if isinstance(verifier, TimedChallenges):
        verifier.challenge_s = 0.0
    claims, rows, proofs = v0_query._claims_message(prover, verifier, x, ch, out, {}, wire=wire)
    hashes = b"".join(h for name in verifier.tables for h in v0_query.proof_hashes(proofs.get(name)))
    parts, openings = [claims, rows or b"", hashes], None
    if verifier.mode == "C":
        _, us, u_blob = v0_query._fold_message(prover, verifier, ch, x, out, wire=wire)
        _, openings = v0_query.open_message(prover, verifier, ch, us, out, u_blob=u_blob)
        parts.append(u_blob if wire else us)
    out["timings"]["challenges"] = getattr(verifier, "challenge_s", 0.0)
    return parts, openings, out


def _raw(z: torch.Tensor, dtype: torch.dtype = torch.int32) -> bytes:
    """A tensor's values as the fixed-width integers that ``run_query`` counts them as without the compact encoding."""
    return z.to("cpu", dtype).contiguous().numpy().tobytes()


def messages(parts: list, openings: dict | None, verifier: Verifier) -> list:
    """The proof's messages as byte strings, in the order they travel: ``parts`` and, in mode C, the opened columns
    and the hashes of their paths.  In the compact encoding the columns travel 31-bit packed -- ``claimcodec.
    pack_field`` of the decoded rows, tree after tree: the packing is a fixed function of the values, so these are
    the bytes ``open_message`` sent.  Without it (``parts[0]`` the claims themselves) every claim, ``u`` and column
    entry is written as a 4-byte integer and a lookup table's rows as 1-byte ones, as ``run_query`` counts them."""
    raw = isinstance(parts[0], dict)
    head = list(parts)
    if raw:
        head[0] = b"".join(_raw(z, torch.int8 if k in verifier.tables else torch.int32) for k, z in parts[0].items())
        if len(parts) > 3:
            head[3] = b"".join(_raw(parts[3][op.name]) for op in verifier._row_ops())
    if openings is None:
        return head
    units = [name for name, _ in verifier._column_units()]
    cols = (b"".join(_raw(openings[name][0]) for name in units) if raw
            else claimcodec.pack_field([openings[name][0] for name in units]))
    paths = b"".join(h for name in units for h in v0_query.proof_hashes(openings[name][1]))
    return [*head, cols, paths]


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
    """The logits a proof claims: the LM head's claim as the verifier reads it (from the compact encoding, and a
    plan's rows, with ``v0.query.decoded_claims``), as int64 ``[B * positions, vocab]``."""
    if isinstance(parts[0], dict):
        return parts[0][head_op(graph).name].T.to("cpu", torch.int64)
    zs = v0_query.decoded_claims(verifier, parts[0], parts[1], x, {"timings": {}, "contention": {}}, torch.int32)
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


def prove_batch(prover: Prover, verifier: Verifier, x: torch.Tensor, how: str, keep: int = 0, wire: bool = True):
    """``(outs, kept)``: the proofs of the batch ``x`` as ``how`` proves it (:func:`queries`), one after another --
    every proof's ``out`` and the ``(parts, openings)`` of the first ``keep`` (the others are dropped as soon as
    they are made)."""
    outs, kept = [], []
    for i, q in enumerate(queries(x, how)):
        parts, openings, out = prove(prover, verifier, q, wire=wire)
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


def _tokens(cfg, seq: int, bsz: int, seed: int = 1) -> torch.Tensor:
    return torch.randint(0, cfg.vocab, (bsz, seq), generator=torch.Generator().manual_seed(seed))


def _prompts(cfg, seq: int, bsz: int, args) -> torch.Tensor:
    """A cell's ``[B, T]`` prompts: the checkpoint's WikiText-2 windows with ``--real``
    (:func:`experiments.common.real_cells.prompts`), else random ids drawn with ``--prompt-seed``."""
    ck = getattr(args, "checkpoint", None)
    return real_cells.prompts(ck, seq, bsz) if ck is not None else _tokens(cfg, seq, bsz, args.prompt_seed)


def _why(e: BaseException) -> str:
    """An exception as one short line, for a ``skipped`` field."""
    return f"{type(e).__name__}: {(str(e).splitlines() or [''])[0][:200]}"


def _joules(args) -> float | None:
    """The GPU's energy counter (joules), or None without one (:mod:`experiments.common.energy`)."""
    return args.meter.read() if getattr(args, "meter", None) is not None else None


def _delta(e0: float | None, e1: float | None) -> float | None:
    return None if e0 is None or e1 is None else e1 - e0


def _energy(per_call: list, loop: dict | None) -> dict | None:
    """A variant's GPU energy per call: ``{"joules", "method", "calls"}`` from back-to-back calls of at least
    ``--energy-s`` seconds (``loop``, for a call shorter than that) or else the median over the timed repetitions'
    own counter differences (``timed``, with every difference in ``joules_all``); None without an energy counter."""
    if loop is not None:
        return {"joules": loop["joules"], "method": "loop", "calls": loop["calls"]}
    xs = [e for e in per_call if e is not None]
    if not xs:
        return None
    return {"joules": float(np.median(xs)), "method": "timed", "calls": len(xs), "joules_all": xs}


def time_u2(cfg, layers: int, cells, args, device: torch.device) -> dict:
    """U2's times ``{(T, B): {dtype: {"times", "wait_frac", "gpu_peak_mb"} or {"skipped": why}}}``: per dtype one
    model, a warm-up and ``--reps`` timed batched prefills per cell; the model is freed afterwards.  A dtype or cell
    the device cannot run (out of memory; bf16 GEMMs on a GPU without them) is recorded as skipped."""
    out: dict = {c: {} for c in cells}
    for name in args.u2_dtype:
        model, why = None, None
        try:
            model = reexec.FloatDecoder(cfg, layers, DTYPES[name], device, seed=args.seed, prune_last=args.prune_last)
        except RuntimeError as e:          # (torch.cuda.OutOfMemoryError is one)
            why = _why(e)
        for seq, bsz in cells:
            if seq > cfg.max_pos:
                continue
            if model is None:
                out[seq, bsz][name] = {"skipped": why}
                continue
            x = _prompts(cfg, seq, bsz, args)
            failed = None
            try:
                _gpu_peak_reset(device)
                records.timed(lambda: serve_u2(model, x, device), device)
                peak = _gpu_peak_mb(device)
                times, waits, joules = [], [], []
                for _ in range(args.reps):
                    e0 = _joules(args)
                    _, s, w = records.timed(lambda: serve_u2(model, x, device), device)
                    joules.append(_delta(e0, _joules(args)))
                    times.append(s)
                    waits.append(w)
                loop = (joules_per_call(args.meter, lambda: serve_u2(model, x, device), lambda: records.sync(device),
                                        args.energy_s) if float(np.median(times)) < args.energy_s else None)
                out[seq, bsz][name] = {"times": times, "wait_frac": waits, "gpu_peak_mb": peak,
                                       "energy": _energy(joules, loop)}
            except RuntimeError as e:
                failed = _why(e)
            if failed:
                out[seq, bsz][name] = {"skipped": failed}
                _free(device)
        model = None                         # freed before the next dtype's model is built
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


def build(name: str, layers: int, args, device: torch.device, real: tuple | None = None) -> Built:
    """The integer decoder of ``name`` on ``layers`` blocks (0: all of them), its provers and verifiers under the
    protocol flags; ``real``: ``(graph, setup, family)`` of a checkpoint's integer graph already built
    (:func:`real_graph`) instead of random weights."""
    cfg = decoder_config(name)
    layers = layers or cfg.n_layers
    shapes = decoder_shapes(cfg, prune_last=args.prune_last)
    if real is not None:
        graph, setup, fam = real[0], dict(real[1]), real[2]
        plan_ops = (None if fam == "opt" and layers == cfg.n_layers   # a full OPT graph: from its own ops
                    else real_cells.model_ops(cfg, args.prune_last))
    else:
        t0 = time.perf_counter()
        graph = build_decoder(cfg, n_layers=layers, calib_tokens=args.calib_tokens, seed=args.seed,
                              prune_last=args.prune_last, prob_max=args.prob_max)
        setup = {"build_s": time.perf_counter() - t0}
        plan_ops = shapes
    n_checks = len(shapes)          # (r, t) for the full model's op count, as bench.py and perf.py
    provers, verifiers = {}, {}
    for mode in args.modes:
        if mode == "C":
            records.sync(device)
            t0 = time.perf_counter()
            com = commit_graph(graph, args.rate, device=device, policy=args.policy, model_ops=plan_ops,
                               cache=args.setup_cache)
            records.sync(device)
            setup["commit_s"] = time.perf_counter() - t0
            setup["setup_cache"] = com.setup_cache              # "hit": commit_s is a load from the setup cache
            params = params_for(args.lam, n_checks, rate=args.rate, plan=com.plan, fiat_shamir=args.fs)
            verifiers[mode] = TimedChallenges(graph.public(), params, "C", publics=com.publics,
                                              groups=com.group_publics, tables=com.table_publics, lean=args.lean,
                                              device=args.verifier_device)
            provers[mode] = Prover(graph, device=device, commitments=com, lean=args.lean)
            setup["chosen_policy"] = args.policy if com.plan is None else com.plan.policy
        else:
            params = params_for(args.lam, n_checks, rate=args.rate, fiat_shamir=args.fs)
            weights = {op.name: (op.weight, op.bias) for op in graph.mat_ops}
            v = TimedChallenges(graph.public(), params, "Kpre", weights=weights, lean=args.lean,
                                device=args.verifier_device, lookups=args.lookups)
            if args.accept:              # the verifier's secret chi and u (only the acceptance check reads them)
                t0 = time.perf_counter()
                v.precompute(Challenger())
                setup["precompute_s"] = time.perf_counter() - t0
            verifiers[mode], provers[mode] = v, Prover(graph, device=device, lean=args.lean)
        setup[f"reps_{mode}"], setup[f"columns_{mode}"] = verifiers[mode].params.reps, verifiers[mode].params.columns
    return Built(name, cfg, layers, graph, provers, verifiers, setup)


def real_graph(name: str, layers: int, args, cells) -> tuple:
    """``(graph, setup, family)``: the integer graph of the checkpoint ``--real`` on ``layers`` blocks (0: all of
    them), with ``load_s`` (the checkpoint in fp32 and its calibration) and ``build_s`` (the graph's own calibration
    and build) in ``setup``; the checkpoint's WikiText-2 prompts for ``cells`` become ``args.checkpoint``'s, and its
    fp32 model is freed."""
    cfg = decoder_config(name)
    fam = real_cells.family(cfg)
    t0 = time.perf_counter()
    seq = max(t for t, _ in cells)
    ck = real_cells.load(args.real, cfg, seq=seq, n_queries=max(1, -(-max(t * b for t, b in cells) // seq)))
    load_s = time.perf_counter() - t0
    t0 = time.perf_counter()
    graph, _ = ck.graph(args.prune_last, layers or cfg.n_layers)
    build_s = time.perf_counter() - t0
    ck.free()
    args.checkpoint = ck
    return graph, {"load_s": load_s, "build_s": build_s}, fam


class Cell:
    """One ``(T, B)`` of a built model: the variants ``"u1"`` and ``(mode, how)``, their warm-ups with the checks,
    the interleaved repetitions and the records."""

    def __init__(self, m: Built, seq: int, bsz: int, args, device: torch.device):
        self.m, self.seq, self.bsz, self.args, self.device = m, seq, bsz, args, device
        self.x = _prompts(m.cfg, seq, bsz, args)
        hows = ["single"] if bsz == 1 else list(args.batch_proof)
        self.variants = ["u1"] + [(mode, how) for mode in args.modes for how in hows]
        self.times = {k: [] for k in self.variants}
        self.waits = {k: [] for k in self.variants}
        self.joules = {k: [] for k in self.variants}      # the GPU's energy counter over each timed repetition
        self.energy: dict = {}                             # key -> _energy()
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
        with contention.paused():
            if key == "u1":
                return serve_u1(m.graph, self.x, self.device)
            mode, how = key
            return prove_batch(m.provers[mode], m.verifiers[mode], self.x, how, keep, wire=self.args.wire)

    def _run(self, key, keep: int = 0):
        """One timed call of ``key`` (``None`` and the variant dropped if it runs out of GPU memory)."""
        oom = False
        try:
            return records.timed(lambda: self._call(key, keep), self.device)
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
            with contention.paused():
                row = 0
                for q in queries(self.x, how):
                    parts, openings, out = prove(m.provers[mode], v, q, wire=self.args.wire)
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
                e0 = _joules(self.args)
                r = self._run(key, keep=1 if (i == 0 and key != "u1") else 0)
                if r is None:
                    continue
                self.joules[key].append(_delta(e0, _joules(self.args)))
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
        self.measure_energy()

    def measure_energy(self) -> None:
        """Each variant's GPU energy per call (:func:`_energy`): a variant whose median repetition is shorter than
        ``--energy-s`` is called back to back for that long after the timed repetitions (untimed, outside them);
        the others use their timed repetitions' counter differences."""
        for key in self.variants:
            if key in self.skipped or not self.times[key]:
                continue
            loop = None
            if float(np.median(self.times[key])) < self.args.energy_s:
                oom = False
                try:
                    loop = joules_per_call(self.args.meter, lambda: self._call(key),
                                           lambda: records.sync(self.device), self.args.energy_s)
                except RuntimeError as e:          # out of memory: the timed repetitions' differences stand
                    if not _is_oom(e):
                        raise
                    oom = True
                if oom:
                    _free(self.device)
            self.energy[key] = _energy(self.joules[key], loop)

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
                res = run_query(self.m.provers[mode], self.m.verifiers[mode], q, wire=self.args.wire)
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
    def ci(xs, ys=None):
        return median_ci(xs, ys, n=cell.args.boot)

    tokens = cell.seq * cell.bsz
    u1 = cell.times["u1"]
    u2_ok = {k: v["times"] for k, v in u2.items() if "times" in v}
    common = {**base, "seq": cell.seq, "batch": cell.bsz, "tokens": tokens}
    u1_rec = ci(u1) if u1 else None
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
        nbytes = int(sum(cell.nbytes[key]))
        per_tok = nbytes / tokens
        line.update({
            "u1_s": u1_rec, "u2_s": {k: ci(v) for k, v in u2_ok.items()},
            "window_s": ci(total), "prove_s": ci(prove_s),
            "prover_over_u1": ci(prove_s, u1),
            "prover_over_u2": {k: ci(prove_s, v) for k, v in u2_ok.items()},
            "u1_over_u2": {k: ci(u1, v) for k, v in u2_ok.items()},
            "phases_s": {**{k: med[k] for k in PHASES}, "harness_decode": med["harness"],
                         "residual": float(np.median([s - q for s, q in zip(prove_s, summed)]))},
            "residual_frac": {"median": float(np.median(resid)), "max_abs": float(np.max(np.abs(resid)))},
            "phases_ok": bool(abs(float(np.median(resid))) <= cell.args.residual_max),
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
            "energy_j": {"u1": cell.energy.get("u1"), "prover": cell.energy.get(key),
                         **{k: v.get("energy") for k, v in u2.items() if "times" in v}},
        })
        lines.append(line)
    req = {**common, "row": "on_request", "serve_s": u1_rec,
           "serve_over_u2": {k: ci(u1, v) for k, v in u2_ok.items()} if u1 else {},
           "u2_s": {k: ci(v) for k, v in u2_ok.items()},
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


def _max(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return max(xs) if xs else None


def run_model(name: str, args, device: torch.device, cells, base: dict) -> bool:
    """Every cell of one ``(model, layers)``: U2 first (its model freed), then the integer model's U1 and provers.
    Prints the lines; ``False`` if a correctness check failed."""
    cfg = decoder_config(name)
    layers = args.layers or cfg.n_layers
    cells = [c for c in cells if c[0] <= cfg.max_pos]
    real = real_graph(name, layers, args, cells) if args.real else None   # before anything is timed
    u2 = time_u2(cfg, layers, cells, args, device)
    m = build(name, layers, args, device, real)
    good = True
    base = {**base, "model": name, "layers": args.layers or None, "n_layers": layers, "n_layers_full": cfg.n_layers,
            "policy": args.policy, "lookups": args.lookups, "real": args.real or None,
            "recipe": None if real is None else ("main" if real[2] == "llama" else "certified"), **m.setup}
    if real is not None:              # the checkpoint's revision in the records' environment
        base["env"] = {**base["env"], "sources": {**(base["env"].get("sources") or {}), **checkpoints.SOURCES}}
    for seq, bsz in cells:
        cell = Cell(m, seq, bsz, args, device)
        cell.warm_up()
        cell.repeat()
        if args.accept:
            cell.accept()
        for line in _lines(cell, u2.get((seq, bsz), {}), base):
            line["host_maxrss_mb"] = _host_maxrss_mb()
            records.emit(line)
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
    """The ``prove`` lines of this script's logs as a table (times in ms), then the two-point extrapolations of U1
    and U2 to the full model where a (model, T, B) was measured on two block counts."""
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

def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", nargs="+", default=[],
                    help="decoders: pvi.models.decoder.CONFIGS names or the small shapes " + ", ".join(TINY))
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: the full model)")
    ap.add_argument("--workload", nargs="+", choices=sorted(WORKLOADS), default=["a", "b"],
                    help="(a) T = 64, 512, 2048 at B = 1; (b) T = 64 with B = 1, 4, 16, 64 and T = 2048 with B = 1, 4")
    ap.add_argument("--cells", nargs="+", default=[], help="TxB cells instead of --workload, e.g. 64x1 64x4")
    ap.add_argument("--modes", nargs="+", choices=["C", "Kpre"], default=["C", "Kpre"])
    ap.add_argument("--batch-proof", nargs="+", choices=["seq", "joint"], default=["seq"],
                    help="B > 1: B independent proofs one after another (seq) and/or one statement (joint)")
    ap.add_argument("--reps", type=int, default=10, help="timed repetitions of every variant, after one warm-up")
    ap.add_argument("--u2-dtype", nargs="+", choices=sorted(DTYPES), default=None,
                    help="U2's dtypes (default fp16 on a GPU, fp32 on a CPU)")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed (the integer model's and U2's)")
    ap.add_argument("--prompt-seed", type=int, default=1, help="the random prompts' seed")
    ap.add_argument("--calib-tokens", type=int, default=cli.CALIB_TOKENS,
                    help="calibration tokens of the integer model (one graph for every cell)")
    ap.add_argument("--accept", action=argparse.BooleanOptionalAction, default=True,
                    help="one untimed run_query per prover variant and cell (Kpre: the verifier's precompute)")
    ap.add_argument("--boot", type=int, default=10_000, help="bootstrap resamples of every interval")
    ap.add_argument("--residual-max", type=float, default=0.02,
                    help="phases_ok: the largest share of the prover's time the phases may leave unaccounted")
    ap.add_argument("--energy-s", type=float, default=2.0,
                    help="the GPU's energy per call (NVML's energy counter, experiments.common.energy): a variant "
                         "whose median call is shorter than this many seconds is also called back to back for this "
                         "long after its timed repetitions; 0: no energy (default 2)")
    real_cells.add_arguments(ap, real_cells.HELP + "; one --model")
    ap.add_argument("--label", default="", help="a label copied into every record")
    ap.add_argument("--table", nargs="+", default=None, help="print this script's logs (JSON lines) as a table")
    cli.add_protocol_arguments(ap)
    cli.add_runtime_arguments(ap, threads=8)
    return ap


def main() -> int:
    ap = build_parser()
    args = ap.parse_args()
    if args.table:
        table(args.table)
        return 0
    if not args.model:
        ap.error("--model is required")
    if args.reps < 2:
        ap.error("--reps must be at least 2 (the first repetition is compared with the warm-up)")
    if "C" in args.modes and not args.fs:
        ap.error("mode C is proved without a verifier only under Fiat-Shamir: --no-fs needs --modes Kpre")
    cells = _cells(args)
    if args.real:
        if len(args.model) != 1:
            ap.error("--real names one checkpoint: give one --model, its shapes")
        cfg = decoder_config(args.model[0])
        real_cells.check(cfg, seqs=[t for t, _ in cells], layers=[args.layers or cfg.n_layers],
                         prob_max=args.prob_max)
    cli.apply_runtime(args)
    device = torch.device(args.device)
    if device.type == "cuda" and device.index is not None:
        torch.cuda.set_device(device)
    args.u2_dtype = args.u2_dtype or (["fp16"] if device.type == "cuda" else ["fp32"])
    base = {"label": args.label, "git": records.git_sha(CODE), "device": str(device),
            "verifier_device": args.verifier_device, "threads": args.threads, "reps": args.reps, "warmup": 1,
            "lam": args.lam, "rate": args.rate, "fs": args.fs, "wire": args.wire, "lean": args.lean,
            "prune_last": args.prune_last, "prob_max": args.prob_max,
            "timer": "cuda_events" if device.type == "cuda" else "perf_counter",
            "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
            "cpu": platform.processor() or platform.machine(), "host": platform.node(), "torch": torch.__version__,
            "env": records.environment(device)}
    args.meter = EnergyMeter() if device.type == "cuda" and args.energy_s > 0 else None
    base["energy_counter"] = None if args.meter is None else (True if args.meter.available else args.meter.why)
    good = True
    for name in args.model:
        good &= run_model(name, args, device, cells, base)
    if not good:
        print("A CORRECTNESS CHECK FAILED (see the lines whose ok is false)", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())

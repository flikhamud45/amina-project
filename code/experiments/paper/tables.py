"""One builder per table of the paper: the body rows of each table, in the paper's layout, from one run.

Each builder returns a :class:`Table`: the rows that sit between the table's ``\\midrule`` and ``\\bottomrule`` in
the paper (its header and caption stay in the paper's source), a header comment that states the run's protocol
settings and sources (so that the caption can be made true), and one entry per value for ``cells.csv`` with the
record it comes from.  A value that the run does not provide prints as ``--`` and is listed as missing.

The tables and their sources (the run package's tiers, ``experiments/package/run.sh``):

==================== ==================================================================================
``tab:eval-llm``      ``bench``: modes C (CPU and streaming GPU verifier) and Kpre, Fiat--Shamir, the plan
``tab:eval-v1``       ``v1``: ``v1_bytes.py --full`` on the CPU verifier, random weights
``tab:options``       ``cpuv`` (perf.py, reexec.py, llamacpp.py, V1's CPU verifier), ``gpuv`` (reexec.py on the GPU)
``tab:eval-compare``  ``bench`` and the published systems (``reported_curated.csv``)
``tab:quality``       ``quality``: ``ppl_opt.py`` (the certified recipe, alpha 0.6, 11-bit probabilities)
``tab:params``        the library's parameters and byte model; V1's counts from the ``v1`` tier
``tab:inrange``       ``attack``: ``inrange_llm.py``'s documents
``tab:setup``         the setup proof's byte model (``costs/setup_bytes.py``)
``tab:eval-improve``  ``improve``: the A/B scripts and perf.py with the options of each row
``tab:memory``        ``cpuv``: perf.py ``--client-memory`` (the client's own process), the prover's GPU peak
``tab:eval-long``     ``long``: 2-, 3- and 4-block cells, extrapolated to the full Llama-2-7B
``tab:eval-long-check`` ``long``, ``cpuv``, ``gpuv``: the full Llama-2-7B measured against the extrapolation
``tab:eval-long-gpu`` ``gpuv``: the streaming GPU verifier and GPU re-execution at long prompts (2 and 3 blocks)
``tab:eval-v1-long`` ``long`` (V1 on 2 blocks) and ``v1`` (the full model's codec profile)
``tab:eval-real``     ``real``: OPT-1.3B, real weights against the random-weight twins (V1 cells, v0 timed)
``tab:eval-reexec``   ``gpuv`` and ``cpuv``: reexec.py, llamacpp.py, the Kpre verifier
``tab:eval-cpu``      ``cpux`` (2 and 4 threads, no AVX-512 attention, Llama-2-13B) and ``cpuv``: perf.py, reexec.py
``tab:eval-plans``    ``cpux``: perf.py with mode C's six candidate plans side by side
``tab:eval-lambda``   ``bench``: lambda 40 and 80 (``b-lam-*``) and 128
``tab:eval-compare-v1`` ``v1`` (mode C) and the published systems
``tab:v1``            ``v1``: the timed queries and the GKR's share of V1's prover
``tab:gen``           ``gen``: greedy responses, both modes
``tab:gen-full``      ``gen``: every cell, greedy and sampled
``tab:quality-attr``  ``quality``: ``attribution.py`` (OPT-125M, the scalar graph with 8-bit probabilities)
``tab:zeroshot``      ``quality``: ``zeroshot.py`` (the certified recipe)
``tab:paths``         Proposition ``prop:paths`` (``attack/sok.py``) at the ``attack`` tier's in-range edit sizes
``tab:paths-acc``     the same
==================== ==================================================================================

Tables that no record produces are not built: the analysis and literature tables ``tab:sok``, ``tab:sok-derivation``
and ``tab:aligned`` (the derived values of ``tab:sok`` and ``tab:sok-derivation`` are the macros ``\\Sok...`` of
``attack/sok.py``).
"""
from __future__ import annotations

import functools
import math
import statistics
from dataclasses import dataclass, field

from experiments.paper import format as f
from experiments.paper.bench import DEFAULT_POLICY, Bench
from experiments.paper.records import ATTRIBUTION_CELL, Rec, Run, v1_timed

NAME = {"gpt2": "GPT-2", "opt-125m": "OPT-125M", "opt-1.3b": "OPT-1.3B", "opt-6.7b": "OPT-6.7B",
        "llama2-7b": "Llama-2-7B", "llama2-13b": "Llama-2-13B", "qwen3-4b": "Qwen3-4B"}
FULL_LAYERS = {"gpt2": 12, "opt-125m": 12, "opt-1.3b": 24, "opt-6.7b": 32, "llama2-7b": 32, "qwen3-4b": 36,
               "llama2-13b": 40}
"""The decoder blocks of the paper's models (``pvi.models.decoder.CONFIGS``)."""
EXACT = ("int", "int_fp32gemm", "int_mm")
"""``reexec.py``'s exact variants: the integer forward pass, bit-exact with the claims, with its native int8 GEMMs,
with exact fp32 GEMMs, or with its int8 GEMMs through ``torch._int_mm`` (oneDNN).  Exact re-execution is the fastest
of them, computing only what the pruned graph computes; it is the only re-execution that checks what the verifier
checks."""
CHECKED_EXACT = ("int_mm",)
"""The exact variants that count as exact only when their record says so: ``exact`` true when the run found their
logits equal to the integer reference's bit for bit (:func:`exact_record`)."""
LLAMACPP = ("llamacpp_q8_0", "llamacpp_bf16")
"""``llamacpp.py``'s variants: llama.cpp's prefill with Q8_0 and with bf16 weights of the same shapes (another model,
not exact); each one's time is its fastest configuration (attention kernel, micro-batch)."""
INEXACT = ("float", "bf16", "dynint8") + LLAMACPP
"""The inexact re-executions: ``reexec.py``'s floating-point decoder of the same shapes in fp32 and bf16 and torch's
dynamic int8 engine, and llama.cpp (:data:`LLAMACPP`); what an off-the-shelf stack would run, not bit-exact with the
provider's model."""
REEXEC_NAME = {"int": "int8 GEMMs", "int_fp32gemm": "exact fp32 GEMMs", "int_mm": "oneDNN int8 GEMMs",
               "float": "fp32", "bf16": "bf16", "dynint8": "dynamic int8", "llamacpp_q8_0": r"llama.cpp Q8\_0",
               "llamacpp_bf16": "llama.cpp bf16"}
LLAMACPP_FORMAT = {"llamacpp_q8_0": r"Q8\_0", "llamacpp_bf16": "bf16"}
"""The weight format of each llama.cpp variant, as the text names it."""
MODE = {"C": r"$\sC$", "Kpre": r"$\Kpre$"}


@dataclass
class Table:
    """The body rows of one table, its header comment and its values."""

    name: str
    label: str
    note: str
    rows: list = field(default_factory=list)
    cells: list = field(default_factory=list)
    missing: list = field(default_factory=list)

    def put(self, row: str, col: str, value, src, fmt=str) -> str:
        """One value: recorded for ``cells.csv`` (``src``: a record, or a description of the source) and counted
        as filled or missing; returns its printed form."""
        source = src.source if isinstance(src, Rec) else str(src or "")
        printed = f.DASH if value is None else fmt(value)
        self.cells.append({"table": self.label, "row": row, "column": col, "value": value, "printed": printed,
                           "source": source})
        if value is None:
            self.missing.append(f"{row} / {col}")
        return printed

    @property
    def filled(self) -> int:
        return sum(c["value"] is not None for c in self.cells)

    def text(self) -> str:
        """The file: the header comment, then the rows."""
        head = "".join(f"% {line}\n" for line in self.note.splitlines())
        return f"% generated by experiments/paper/numbers.py: {self.label}\n" + head + "\n".join(self.rows) + "\n"


@dataclass
class Ctx:
    """What the builders read: the run's records, the benchmark's tables, the published systems' values, mode C's
    plan and the platform."""

    run: Run
    bench: Bench
    lit: dict
    policy: str = DEFAULT_POLICY
    platform: str = ""
    byte_model: bool = True
    xcheck: list = field(default_factory=list)
    """The ``digests`` records of other platforms' runs of the same cells (``numbers.py --xcheck-logs``)."""


def protocol(ctx: Ctx, extra: str = "") -> str:
    """The header comment's protocol line."""
    return (f"protocol: Fiat-Shamir (unless stated), lambda 128, mode C under the plan {ctx.policy}, the compact "
            f"encoding, the pruned last block, the verifier's own embedding rows in Kpre, decoders' attention "
            f"probabilities at prob_max {ctx.bench.prob_max}; {ctx.platform}"
            + (f"; {extra}" if extra else ""))


def exact_record(variant: str, rec: Rec | None) -> bool:
    """Whether a re-execution record of ``variant`` may stand for exact re-execution: always for the integer forward
    pass's own GEMM paths, and for a :data:`CHECKED_EXACT` variant only when the record's ``exact`` is true."""
    return variant not in CHECKED_EXACT or bool(rec is not None and rec.d.get("exact") is True)


def fastest(run: Run, tier: str, model: str, seq: int, variants=EXACT):
    """``(seconds, variant, record)`` of the fastest of ``variants`` (default: exact re-execution) on the CPU, or
    ``(None, None, None)``; a :data:`CHECKED_EXACT` variant whose record does not say it was exact is left out."""
    got = run.reexec(tier, model, seq, "cpu", variants, keep=exact_record)
    if not got:
        return None, None, None
    v, (s, r) = min(got.items(), key=lambda kv: kv[1][0])
    return s, v, r


def option(rec: Rec | None, name: str):
    """An implementation option in force for a record (``env.implementation.options``; None where not recorded)."""
    env = (rec.d.get("env") or {}) if rec is not None else {}
    return ((env.get("implementation") or {}).get("options") or {}).get(name)


def cpu_point(run: Run, tier: str, model: str, seq: int, threads: int = 8, avx512: bool = True, label: str = "cpuv"):
    """``(verifier, exact, ratio)`` of one CPU setting: the Kpre CPU verifier's ``perf.py`` record of the full
    ``model`` (``label``, ``threads``), the fastest exact re-execution on as many threads, ``(seconds, variant,
    record)`` or None, and the ratio of the two (None if either is missing).  ``avx512`` False: both with the option
    ``native_avx512`` off (records that state no options count as on)."""
    def same(r):
        return r.d.get("threads", 8) == threads and (option(r, "native_avx512") is not False) == avx512
    ver = run.perf(tier, model, seq, "Kpre", vdev="cpu", label=label, threads=threads)
    ver = ver if ver is not None and same(ver) else None
    got = run.reexec(tier, model, seq, "cpu", EXACT, keep=lambda v, r: exact_record(v, r) and same(r))
    best = None
    if got:
        v, (sec, r) = min(got.items(), key=lambda kv: kv[1][0])
        best = (sec, v, r)
    ratio = best[0] / ver.d["verify_s"] if best and ver and ver.d.get("verify_s") else None
    return ver, best, ratio


# ================================================================================================ evaluation.tex
LLM_ROWS = {64: ("gpt2", "qwen3-4b", "llama2-7b", "llama2-13b"),
            2048: ("opt-125m", "opt-1.3b", "opt-6.7b", "llama2-7b", "llama2-13b")}
"""``tab:eval-llm``'s rows."""


def tab_eval_llm(ctx: Ctx) -> Table:
    """``tab:eval-llm``: per model and prompt length, mode C's prover, CPU and GPU verifier and proof, and Kpre's CPU
    verifier and proof, all of the final code (one column per metric)."""
    tb = Table("tab_eval_llm", "tab:eval-llm",
               f"{protocol(ctx)}; bench tables {ctx.bench.name}\n"
               "columns: Model & Params & C prove & C CPU verify & C GPU verify (streaming) & C proof & "
               "Kpre CPU verify & Kpre proof (the prover and the proof of mode C from the CPU-verifier cell)")
    b = ctx.bench
    for seq, models in LLM_ROWS.items():
        tb.rows.append(rf"\multicolumn{{8}}{{@{{}}l}}{{\emph{{{f.tokens(seq)}-token prompts}}}} \\")
        for m in models:
            row = f"{NAME[m]} ({seq})"
            cc, cg, kc = b.llm_cost(m, seq, "c_cpu"), b.llm_cost(m, seq, "c_gpu"), b.llm_cost(m, seq, "kpre_cpu")
            src = lambda c: f"{b.name} {c['cell']}" if c else ""     # noqa: E731
            first = cc or cg
            n = next((c["params"] for c in (cc, cg, kc) if c and c["params"]), None)
            tb.rows.append(" & ".join([
                NAME[m], tb.put(row, "Params", n, src(first), f.params),
                tb.put(row, "C prove", first and first["prove"], src(first), f.t),
                tb.put(row, "C CPU verify", cc and cc["verify"], src(cc), f.t),
                tb.put(row, "C GPU verify", cg and cg["verify"], src(cg), f.t),
                tb.put(row, "C proof", first and first["bytes"], src(first), f.b),
                tb.put(row, "Kpre CPU verify", kc and kc["verify"], src(kc), f.t),
                tb.put(row, "Kpre proof", kc and kc["bytes"], src(kc), f.b)]) + r" \\")
    return tb


V1_ROWS = ((64, "gpt2"), (512, "gpt2"), (2048, "opt-125m"), (2048, "opt-1.3b"), (2048, "llama2-7b"),
           (2048, "llama2-13b"))
"""``tab:eval-v1``'s rows (and ``tab:v1``'s)."""


def v1_cell(run: Run, model: str, seq: int, mode: str) -> Rec | None:
    """The v1 tier's record of one ``tab:eval-v1`` cell: random weights, the CPU verifier, no label."""
    return run.v1("v1", model, seq, mode, vdev="cpu", label="", real=None)


def v1_shape_model(model: str, seq: int, lmax: int, policy: str | None = None, layers: int | None = None) -> dict:
    """V1's cut of ``model`` (``layers`` blocks, default all; the last block pruned) on a ``seq``-token query, from
    its shapes alone, when every weight operation that rule (P2) of :mod:`pvi.v1.cut` allows is cut (as with random
    weights): every linear operation with a claim per position but the LM head.  The instances are packed as
    :meth:`pvi.v1.cut.CutPlan.from_graph` packs them (in graph order, at most ``2^lmax`` leaves each; under the plan
    ``policy`` of the full model, as mode C commits it, a transposed matrix is one unit, cut whole or not at all;
    ``None``: mode Kpre).  Returns ``n_leaves``, ``n_instances``, ``instance_leaves`` (each instance's cut entries),
    ``max_bits`` (the largest instance's variables) and the two new terms of Theorem ``thm:v1`` as ``log2``: the GKR
    terms of the instances and the lookup term with the largest table the window widths allow (two widths of at most
    ``2^16`` per cut operation, and width 1)."""
    from pvi.commit.cost import decoder_claim_columns, decoder_shapes
    from pvi.commit.plans import plan_commitment
    from pvi.core.field import P
    from pvi.models.decoder import CONFIGS
    cfg = CONFIGS[model]
    full = decoder_shapes(cfg, prune_last=True)
    shapes = decoder_shapes(cfg, layers, prune_last=True)
    cols = decoder_claim_columns(cfg, seq, layers, prune_last=True)
    cut = {o.name: o for o in shapes if o.layout == "linear" and o.name != "head" and cols[o.name] == seq}
    col_of = {}
    if policy:
        plan = plan_commitment(shapes, policy, rate=4, model_ops=full)
        col_of = {m.name: tuple(m.members) for m in plan.matrices if m.layout == "col"}
    for members in col_of.values():
        if any(o not in cut for o in members):
            for o in members:
                cut.pop(o, None)
    units, placed = [], set()
    for o in shapes:
        if o.name in cut and o.name not in placed:
            names = next((ms for ms in col_of.values() if o.name in ms), (o.name,))
            placed.update(names)
            units.append(sum(cut[n].n_rows for n in names))
    rows, cur = [], 0
    for n in units:
        if cur and (cur + n) * seq > 1 << lmax:
            rows.append(cur)
            cur = 0
        cur += n
    rows += [cur] if cur else []
    n_col = (seq - 1).bit_length()
    bits = [max(1, (r - 1).bit_length() + n_col) for r in rows]
    log2p = math.log2(P)
    leaves = sum(o.n_rows for o in cut.values()) * seq
    gkr = [_gkr_log2(n, log2p) for n in bits]
    top = max(gkr) if gkr else 0.0
    return {"n_leaves": leaves, "n_instances": len(rows), "instance_leaves": [r * seq for r in rows],
            "max_bits": max(bits, default=0),
            "gkr_log2": top + math.log2(sum(2 ** (x - top) for x in gkr)) if gkr else None,
            "lookup_log2": math.log2(leaves + 1 + 2 * len(cut) * (1 << 16)) - math.log2(P ** 8 - P ** 2)}


def tab_eval_v1(ctx: Ctx) -> Table:
    """``tab:eval-v1``: V1 against v0 on the same queries: both proofs, their ratio, and the prover and CPU verifier
    of each (median over the timed queries, each party's Fiat--Shamir hashing in its time)."""
    tb = Table("tab_eval_v1", "tab:eval-v1",
               f"{protocol(ctx, 'V1 under Fiat-Shamir, its optimised verifier; v1 tier (v1_bytes.py --full)')}\n"
               "columns: Build (b/L) & mode & v0 MB & V1 MB & ratio & prover v0 / V1 (s) & verifier v0 / V1 (s)")
    last = None
    for seq, m in V1_ROWS:
        if seq != last:
            tb.rows.append(rf"\multicolumn{{7}}{{@{{}}l}}{{\emph{{$T{{=}}{f.tokens(seq)}$}}}} \\")
            last = seq
        for i, mode in enumerate(("C", "Kpre")):
            r = v1_cell(ctx.run, m, seq, mode)
            row = f"{NAME[m]} T={seq} {mode}"
            d = r.d if r else {}
            build = f"{NAME[m]}, {d.get('layers', '?')}/{FULL_LAYERS[m]}" if r else f"{NAME[m]}, --"
            p0, v0 = v1_timed(r, "v0")
            p1, v1 = v1_timed(r, "v1", "optimised")
            two = lambda a, c: f.DASH if a is None or c is None else f"{a:.2f} / {c:.2f}"   # noqa: E731
            for col, val in (("v0 prover", p0), ("V1 prover", p1), ("v0 verifier", v0), ("V1 verifier", v1)):
                tb.put(row, col, val, r)
            tb.rows.append(" & ".join([
                build if i == 0 else "", MODE[mode],
                tb.put(row, "v0 MB", d.get("v0_total"), r, lambda v: f.mb(v)),
                tb.put(row, "V1 MB", d.get("v1_total"), r, lambda v: f.mb(v)),
                tb.put(row, "ratio", d["v0_total"] / d["v1_total"] if r else None, r, lambda v: f.x(v, 2)),
                two(p0, p1), two(v0, v1)]) + r" \\")
    return tb


def tab_options(ctx: Ctx) -> Table:
    """``tab:options``: a client's options for one Llama-2-7B prompt: exact re-execution on the CPU (the fastest of
    the integer forward pass's exact variants, :data:`EXACT`) and on the GPU (the integer forward pass on int8 tensor
    cores), llama.cpp's inexact Q8_0 prefill on the same CPU (its fastest configuration; its bytes: the Q8_0 weights),
    the published options, and our modes C and Kpre and V1 (the CPU verifier: bytes and time)."""
    run, m = ctx.run, "llama2-7b"
    tb = Table("tab_options", "tab:options",
               f"{protocol(ctx)}; cpuv tier (perf.py --verifier-device cpu, reexec.py and llamacpp.py on 8 threads "
               "(llama.cpp: Q8_0, random weights, the fastest of its attention kernels and micro-batches), V1's "
               "optimised CPU verifier), gpuv tier (reexec.py int on the GPU); every model in full")
    weights = ctx.bench.llm_cost(m, 64, "c_cpu") or ctx.bench.llm_cost(m, 64, "kpre_cpu")
    wb = weights["params"] if weights else None
    once = rf"\multicolumn{{2}}{{c}}{{{f.b(wb) if wb else f.DASH} once}}"
    rx = [fastest(run, "cpuv", m, s) for s in (64, 2048)]
    gx = [run.reexec("gpuv", m, s, "cuda", ("int",)).get("int") for s in (64, 2048)]
    tb.put("weights", "bytes", wb, f"{ctx.bench.name} params_full", f.b)
    tb.rows.append(" & ".join([r"Re-exec., CPU$^a$", "weights", once,
                               tb.put("Re-exec. CPU", "64", rx[0][0], rx[0][2], f.secs),
                               tb.put("Re-exec. CPU", "2048", rx[1][0], rx[1][2], f.secs), "exact"]) + r" \\")
    tb.rows.append(" & ".join([r"Re-exec., GPU$^a$", "weights", once,
                               tb.put("Re-exec. GPU", "64", gx[0] and gx[0][0], gx[0] and gx[0][1], f.secs),
                               tb.put("Re-exec. GPU", "2048", gx[1] and gx[1][0], gx[1] and gx[1][1], f.secs),
                               "exact"]) + r" \\")
    lx = [fastest(run, "cpuv", m, s, ("llamacpp_q8_0",)) for s in (64, 2048)]
    lr = lx[1][2] or lx[0][2]
    lb = lr and lr.d.get("gguf_bytes")
    held = tb.put("llama.cpp", "bytes", lb, lr, f.b) + (r" Q8\_0 once" if lb else "")
    tb.rows.append(" & ".join([r"llama.cpp~\cite{llamacpp}$^g$", "weights", rf"\multicolumn{{2}}{{c}}{{{held}}}",
                               tb.put("llama.cpp", "64", lx[0][0], lx[0][2], f.secs),
                               tb.put("llama.cpp", "2048", lx[1][0], lx[1][2], f.secs), "inexact"]) + r" \\")
    tb.rows.append(r"GPU TEE~\cite{gpucc} & -- & -- & -- & -- & -- & vendor \\")
    tb.rows.append(r"CommitLLM$^e$ & key & \multicolumn{2}{c}{144\,kB per token} & "
                   r"\multicolumn{2}{c}{0.66\,ms per token} & sampled$^b$ \\")
    z = ctx.lit.get(("zkLLM", "Llama-2-7B", "2048"))
    zb, zv = (f.to_float(z["proof_bytes"]), f.to_float(z["verifier_s"])) if z else (None, None)
    tb.rows.append(" & ".join([r"zkLLM~\cite{zkllm}", "comm.", "--", f.b(zb) if zb else f.DASH, "--",
                               f.secs(zv) if zv else f.DASH, "full, ZK"]) + r" \\")
    for mode, label, held, guarantee in (("C", r"Ours, $\sC$", "roots", "full"),
                                         ("Kpre", r"Ours, $\Kpre$", "key", "full$^c$")):
        rs = [run.perf("cpuv", m, s, mode, vdev="cpu", label="cpuv") for s in (64, 2048)]
        tb.rows.append(" & ".join([label, held] + [
            tb.put(f"Ours {mode}", f"bytes {s}", r and r.d["proof_bytes"], r, f.b) for s, r in zip((64, 2048), rs)] + [
            tb.put(f"Ours {mode}", f"time {s}", r and r.d["verify_s"], r, f.secs) for s, r in zip((64, 2048), rs)]
            + [guarantee]) + r" \\")
    vs = [run.v1("cpuv", m, s, "Kpre", vdev="cpu", label="cpuv-v1") for s in (64, 2048)]
    tb.rows.append(" & ".join([r"\quad with V1$^d$", "key"] + [
        tb.put("with V1", f"bytes {s}", r and r.d["v1_total"], r, f.b) for s, r in zip((64, 2048), vs)] + [
        tb.put("with V1", f"time {s}", v1_timed(r, "v1", "optimised")[1], r, f.secs) for s, r in zip((64, 2048), vs)]
        + ["full"]) + r" \\")
    return tb


COMPARE = (("zkCNN$^a$", ("zkCNN", "VGG-16 CIFAR-10", ""), "VGG-16", ("cnn", "vgg16")),
           ("ZKML", ("ZKML", "VGG-16 CIFAR-10", ""), "VGG-16", ("cnn", "vgg16")),
           None,
           ("zkGPT$^b$", ("zkGPT", "GPT-2", ""), "GPT-2 (64)", ("llm", "gpt2", 64)),
           ("DeepProve$^c$", ("DeepProve", "GPT-2", "64"), "GPT-2 (64)", ("llm", "gpt2", 64)),
           ("DeepProve$^c$", ("DeepProve", "GPT-2", "512"), "GPT-2 (512)", ("llm", "gpt2", 512)),
           ("ZKTorch", ("ZKTorch", "Llama-2-7B (1 token)", "1"), "Llama-2-7B (1)", ("llm", "llama2-7b", 1)),
           ("Maverick$^d$", ("Maverick (verif.-only, 1 thr.)", "Qwen3-4B", "8"), "Qwen3-4B (8)", ("maverick",)),
           None,
           ("zkLLM", ("zkLLM", "OPT-125M", "2048"), "OPT-125M (2{,}048)", ("zkllm", "opt-125m")),
           ("zkLLM", ("zkLLM", "OPT-1.3B", "2048"), "OPT-1.3B (2{,}048)", ("zkllm", "opt-1.3b")),
           ("zkLLM", ("zkLLM", "Llama-2-7B", "2048"), "Llama-2-7B (2{,}048)", ("zkllm", "llama2-7b")),
           ("zkLLM", ("zkLLM", "Llama-2-13B", "2048"), "Llama-2-13B (2{,}048)", ("zkllm", "llama2-13b")))
"""``tab:eval-compare``'s rows: (system label, its row in ``reported_curated.csv``, the model label, ours);
None: a rule between groups."""


def compare_ours(ctx: Ctx, which: tuple):
    """``(prove, verify, bytes, source)`` of our side of one ``tab:eval-compare`` row: the image model under
    Fiat--Shamir, the language models under Fiat--Shamir with the CPU verifier, Maverick's setting (Kpre, lambda
    40, one thread, interactive, without the compact encoding), and zkLLM's rows interactive with the prover and
    the proof of the CPU-verifier cell and the streaming GPU verifier."""
    b = ctx.bench
    if which[0] == "cnn":
        cell = b.cnn_cell(which[1], "C", "fs")
        c = b.cnn_cost(which[1], cell)
        return (c["prove"], c["verify"], c["bytes"], f"{b.name} {cell}") if c else (None, None, None, "")
    if which[0] == "llm":
        c = b.llm_cost(which[1], which[2], "c_cpu")
        return (c["prove"], c["verify"], c["bytes"], f"{b.name} {c['cell']}") if c else (None, None, None, "")
    if which[0] == "maverick":
        c = b.llm_cost("qwen3-4b", 8, "maverick")
        return (c["prove"], c["verify"], c["bytes"], f"{b.name} {c['cell']}") if c else (None, None, None, "")
    cpu, gpu = b.llm_cost(which[1], 2048, "c_cpu_int"), b.llm_cost(which[1], 2048, "c_gpu_int")
    first = cpu or gpu
    return (first and first["prove"], gpu and gpu["verify"], first and first["bytes"],
            " + ".join(f"{b.name} {c['cell']}" for c in (cpu, gpu) if c))


def compare_cell(theirs, ours, fmt) -> str:
    """Our value with the factor against theirs, in bold where ours is lower."""
    if ours is None:
        return f.DASH
    text, q = f.factor(theirs, ours)
    if q is None:
        return fmt(ours)
    cell = f"{fmt(ours)} ({text})"
    return rf"\textbf{{{cell}}}" if q >= 1.01 else cell


def tab_eval_compare(ctx: Ctx) -> Table:
    """``tab:eval-compare``: our prover, verifier and proof against the published systems on matched models."""
    tb = Table("tab_eval_compare", "tab:eval-compare",
               f"{protocol(ctx)}; bench tables {ctx.bench.name}; ours under Fiat-Shamir with the CPU verifier, "
               "except Maverick's row (Kpre, lambda 40, one thread, interactive, no compact encoding) and zkLLM's "
               "(interactive; the streaming GPU verifier); theirs: artifacts/comparison/tables/reported_curated.csv")
    for row in COMPARE:
        if row is None:
            tb.rows.append(r"\midrule")
            continue
        label, key, model, which = row
        lit = ctx.lit.get(key) or {}
        theirs = [f.to_float(lit.get(k)) for k in ("prover_s", "verifier_s", "proof_bytes")]
        p, v, nb, src = compare_ours(ctx, which)
        name = f"{key[0]} {model}"
        cells = [label, model]
        for (t_val, o_val, col, fmt) in ((theirs[0], p, "prover", f.t), (theirs[1], v, "verifier", f.t),
                                         (theirs[2], nb, "proof", f.b)):
            tb.put(name, f"ours {col}", o_val, src)
            cells += [fmt(t_val) if t_val else f.DASH, compare_cell(t_val, o_val, fmt)]
        tb.rows.append(" & ".join(cells) + r" \\")
    return tb


# ================================================================================================ quality.tex
QUALITY = (("facebook/opt-125m", "OPT-125M", 4), ("facebook/opt-1.3b", "OPT-1.3B", 3), ("facebook/opt-6.7b",
                                                                                         "OPT-6.7B", 2))
"""``tab:quality``'s rows: the checkpoint, its label and the best scalar norm gain (``ppl_opt.BEST_GAIN``)."""
CERTIFIED_CELL = "smooth-certified-a0.6-p11"
"""``ppl_opt.py``'s cell of the certified model (alpha 0.6, 11-bit probabilities)."""


def scalar_cell(gain) -> str:
    return f"scalar-g{gain:g}-p11"


def tab_quality(ctx: Ctx) -> Table:
    """``tab:quality``: the fp32 perplexity, the scalar-gain graph's ratio, the certified model's perplexity and
    ratio and its top-1 agreement, on the whole WikiText-2 test split at 2,048 tokens (calibration draw 0 of the
    train split)."""
    tb = Table("tab_quality", "tab:quality", "quality tier, ppl_opt.py: test split, 2,048-token windows, calibration "
               f"on four train windows (draw 0); cells fp32, scalar-g<gain>-p11 and {CERTIFIED_CELL}")
    for model, label, gain in QUALITY:
        ref, sc, ce = (ctx.run.ppl(model, c) for c in ("fp32", scalar_cell(gain), CERTIFIED_CELL))
        tb.rows.append(" & ".join([
            label, tb.put(label, "fp32", ref and ref.d["ppl"], ref, lambda v: f.fixed(v, 2)),
            tb.put(label, "scalar gain", sc and sc.d["ratio"], sc, lambda v: rf"$\times${v:.3f}"),
            (f"{f.fixed(ce.d['ppl'], 2)} ($\\times${ce.d['ratio']:.3f})" if ce else f.DASH),
            tb.put(label, "top-1", ce and ce.d.get("top1"), ce, lambda v: f.fixed(v, 2))]) + r" \\")
        tb.put(label, "certified ppl", ce and ce.d["ppl"], ce)
        tb.put(label, "certified ratio", ce and ce.d["ratio"], ce)
    return tb


# ================================================================================================ app_params.tex
def _gkr_log2(n: int, log2p: float) -> float:
    """``log2`` of ``(1 + sum_{k=1}^{n-1} (3k + 2)) / p^8``, one GKR instance of ``2^n`` leaves (Theorem thm:v1)."""
    return math.log2(1 + sum(3 * k + 2 for k in range(1, n))) - 8 * log2p


@functools.lru_cache(maxsize=None)
def setup_of(model: str, policy: str, amortise: int):
    """``(size, params, plan)`` of the setup proof of one of the paper's decoders (last block pruned) under the plan
    ``policy`` at lambda 128 with interactive queries, amortised over ``amortise`` queries: ``size`` is
    ``costs/setup_bytes.setup_model``'s byte model, ``params`` :func:`pvi.commit.setup_proof.setup_params`' radii and
    column counts.  Each takes up to half a minute on a laptop CPU, so a build computes each once."""
    from experiments.costs.setup_bytes import plan_publics, setup_model
    from pvi.commit import setup_proof as sp
    from pvi.commit.cost import decoder_shapes
    from pvi.commit.plans import column_bits, plan_commitment
    from pvi.models.decoder import CONFIGS
    shapes = decoder_shapes(CONFIGS[model], prune_last=True)
    plan = plan_commitment(shapes, policy, rate=4)
    publics, groups = plan_publics(plan)
    params = sp.setup_params(publics, groups, 128, column_bits(128, len(shapes), False), amortise=amortise)
    return setup_model(shapes, policy, 128, amortise), params, plan


def radius_log2(params, plan) -> dict:
    """``{tree: log2 f}`` of a setup proof's radii ``e_j = floor(f (n_j - k_j) / 2)`` (None for radius 0)."""
    coded, e = {m.name: m for m in plan.coded}, dict(params.e)
    out = {}
    for g, members in plan.groups:
        m0 = coded[members[0]]
        frac = 2 * e[members[0]] / max(1, m0.n_points - m0.row_length)
        out[g] = round(math.log2(frac)) if frac > 0 else None
    return out


def params_values(policy: str, v1: dict | None, model: str = "llama2-7b", amortise=(100, 1000)) -> dict:
    """``tab:params``'s values from the library (the shapes of ``model`` with the last block pruned and the plan
    ``policy``), as ``costs/params_table.py`` computes them; V1's counts ``v1`` (``n_leaves``, ``n_instances``,
    ``table``, ``lmax`` of a full-model record at 2,048 tokens) give the V1 rows."""
    from pvi.commit.cost import decoder_shapes
    from pvi.commit.plans import column_bits, plan_commitment
    from pvi.core.field import P
    from pvi.models.decoder import CONFIGS
    from pvi.v0.params import params_for

    log2p = math.log2(P)
    shapes = decoder_shapes(CONFIGS[model], prune_last=True)
    plan = plan_commitment(shapes, policy, rate=4)
    coded = {m.name: m for m in plan.coded}
    rows = {g: sum(coded[m].n_rows for m in members) for g, members in plan.groups}
    out = {"log2p": log2p, "ops": len(shapes), "coded": len(coded), "trees": len(rows)}
    for fs in (False, True):
        p = params_for(128, len(shapes), rate=4, plan=plan, fiat_shamir=fs)
        cols = dict(p.group_columns)
        out["fs" if fs else "int"] = {"beta": column_bits(128, len(shapes), fs), "r": p.reps,
                                      "freivalds_log2": -p.reps * log2p, "t_min": min(cols.values()),
                                      "columns_bytes": sum(4 * cols[g] * rows[g] for g in rows)}
    out["ext_log2"] = 8 * log2p
    if v1:
        out["lookup_log2"] = math.log2(v1["n_leaves"] + v1["table"]) - math.log2(P ** 8 - P ** 2)
        out["gkr_log2"] = math.log2(v1["n_instances"]) + _gkr_log2(v1["lmax"], log2p)
    out["beta0"] = 128 + 64 + math.log2(2 * len(coded))
    for a in amortise:
        size, p, splan = setup_of(model, policy, a)
        fracs = set(radius_log2(p, splan).values())
        out[a] = {"s": p.s, "f": sorted(fracs, key=lambda v: (v is None, v)), "t0": sorted(dict(p.t0).values()),
                  "setup_bytes": size["setup_MB"] * 1e6,
                  "untrusted_bytes": size["query_columns_MB_untrusted"] * 1e6}
    return out


def params_v1_counts(run: Run, model: str = "llama2-7b") -> tuple[dict | None, Rec | None]:
    """V1's counts of the full ``model`` at 2,048 tokens from the v1 tier (mode C, else Kpre)."""
    for mode in ("C", "Kpre"):
        r = v1_cell(run, model, 2048, mode)
        if r and r.d.get("layers") == FULL_LAYERS[model] and "n_leaves" in r.d:
            return {k: r.d[k] for k in ("n_leaves", "n_instances", "table", "lmax")}, r
    return None, None


def tab_params(ctx: Ctx) -> Table:
    """``tab:params``: every term of the concrete security of Llama-2-7B at lambda 128 under the run's plan, from
    the library; V1's lookup and GKR terms from the v1 tier's full-model cell."""
    counts, rec = params_v1_counts(ctx.run)
    tb = Table("tab_params", "tab:params",
               f"the library's parameters (pvi.v0.params.params_for, pvi.commit.setup_proof.setup_params, "
               f"costs/setup_bytes.setup_model) for Llama-2-7B under the plan {ctx.policy}, lambda 128; V1's counts "
               f"from {rec.source if rec else 'no v1-tier record (missing)'}")
    v = params_values(ctx.policy, counts) if ctx.byte_model else None
    src = f"library ({ctx.policy})"

    def g(path, fmt=str):
        node = v
        for k in path:
            node = None if node is None else node.get(k) if isinstance(node, dict) else None
        return tb.put("tab:params", "/".join(map(str, path)), node, src, fmt)

    def pair(path_a, path_b, fmt):
        return f"{g(path_a, fmt)} / {g(path_b, fmt)}"

    def lg(x):
        return f"$2^{{{x:.1f}}}$"

    t0 = [v and v.get(a, {}).get("t0") for a in (100, 1000)] if v else [None, None]
    tb.put("tab:params", "t0", t0[0] and t0[1] and (t0[0], t0[1]), src)
    t0s = " / ".join(f.rng(t[0], t[-1], f.tokens) if t else f.DASH for t in t0)
    fr = [v and v.get(a, {}).get("f") for a in (100, 1000)] if v else [None, None]
    fexp = [x[0] if x and len(x) == 1 and x[0] is not None else None for x in fr]
    tb.put("tab:params", "f", fexp[0] is not None and fexp[1] is not None and tuple(fexp) or None, src)
    radius = (f"$f=2^{{{fexp[0]}}}$ / $2^{{{fexp[1]}}}$$^c$" if None not in fexp else f.DASH)

    def le(x):                     # a bound 2^x: the exponent rounded up, so that the printed bound holds
        return f"$\\le2^{{{math.ceil(x * 10 - 1e-9) / 10:.1f}}}{{}}^b$"

    lookup = tb.put("tab:params", "lookup", v and v.get("lookup_log2"), rec or src, le)
    gkr = tb.put("tab:params", "gkr", v and v.get("gkr_log2"), rec or src, le)
    tb.rows += [
        r"\multicolumn{2}{@{}l}{\emph{Per query, v0 (Theorem~\ref{thm:sound}, Section~\ref{sec:fs})}} \\",
        rf"Field $\F$, $p=15\cdot2^{{27}}+1$ & $\log_2p\approx{g(('log2p',), lambda x: f'{x:.1f}')}$ \\",
        rf"Weight operations $n_{{\mathrm{{op}}}}\ge|\mathcal J|$ & {g(('ops',))} \\",
        r"Target $\beta=\lambda+\log_2(2n_{\mathrm{op}})$, plus 64 under Fiat--Shamir & "
        + pair(("int", "beta"), ("fs", "beta"), lambda x: f"{x:.1f}") + r" \\",
        r"Freivalds vectors $r=\lceil\beta/\log_2p\rceil$ & " + pair(("int", "r"), ("fs", "r"), str) + r" \\",
        r"Freivalds term $p^{-r}$ per matrix & " + pair(("int", "freivalds_log2"), ("fs", "freivalds_log2"), lg)
        + r" \\",
        r"Columns $t_j$: least $t$ with $\prod_{i<t}\frac{k_j-1-i}{n_j-i}\le2^{-\beta}$ &",
        "  " + pair(("int", "t_min"), ("fs", "t_min"), str) + r"$^a$ \\",
        r"Opened columns per query & " + pair(("int", "columns_bytes"), ("fs", "columns_bytes"),
                                              lambda x: f.fixed(x / 1e6, 1)) + r"\,MB \\",
        r"Random-oracle queries $Q$ (64 bits added to $\beta$ and $\beta_0$) & -- / $\le2^{64}$ \\",
        r"SHA-256 Merkle trees, 32-byte roots; SHAKE-256 transcript, 256-bit keys &",
        r"  $O(Q^22^{-256})$ \\",
        r"Total: $\varepsilon_{\sC}$ / $Q\varepsilon_{\sC}$, plus collisions for $Q\le2^{63}$ & $\le2^{-128}$ \\",
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\emph{V1 (Theorem~\ref{thm:v1})}} \\",
        r"Extension $\Fext=\F[\xi]/(\xi^8-11)$; Fiat--Shamir needs degree $\ge8$ for the lookup term &",
        f"  $|\\Fext|\\approx2^{{{g(('ext_log2',), lambda x: f'{x:.1f}')}}}$ \\\\",
        rf"Lookup term $(\Nleaf+|\Vtab|)/(p^8-p^2)$ & {lookup} \\",
        rf"GKR terms $\sum_\gamma\varepsilon_{{\mathrm{{GKR}}}}(n_\gamma)$ & {gkr} \\",
        r"Folds: 8 base-field rows per cut operation & $t_j$ unchanged \\",
        r"\midrule",
        r"\multicolumn{2}{@{}l}{\emph{Setup proof (Theorem~\ref{thm:untrusted})}} \\",
        rf"Target $\beta_0=\lambda+64+\log_2(2|\mathcal J_0|)$, $|\mathcal J_0|={g(('coded',))}$ coded & "
        + g(("beta0",), lambda x: f"{x:.1f}") + r" \\",
        r"Combinations $s=\lceil(\beta_0+\log_2\max_jn_j)/\log_2p\rceil$ & " + g((100, "s")) + r" \\",
        r"Radius $e_j=\lfloor f(n_j-k_j)/2\rfloor$, $f\in\{2^0,\dots,2^{-10}\}$ minimising",
        rf"  $t^0_j/a+t_j$ & {radius} \\",
        r"Setup columns $t^0_j$: least with $\prod_{i<t^0_j}\frac{n_j-e_j-1-i}{n_j-i}\le2^{-\beta_0}$ &",
        f"  {t0s}$^c$ \\\\",
        r"Opened columns per query ($t_j$ with $k_j+e_j$ for $k_j$) & "
        + pair((100, "untrusted_bytes"), (1000, "untrusted_bytes"), lambda x: f.fixed(x / 1e6, 1)) + r"\,MB$^c$ \\",
        r"Setup proof & " + pair((100, "setup_bytes"), (1000, "setup_bytes"), lambda x: f.fixed(x / 1e9, 1))
        + r"\,GB$^c$ \\",
        r"Total $\varepsilon_0$ & $\le2^{-128}+O(Q^22^{-256})$ \\"]
    return tb


# ================================================================================================ ext_attack.tex
INRANGE = (("OPT-1.3B", "opt-1.3b", (6, 12, 18, 23)), ("OPT-6.7B", "opt-6.7b", (8, 16, 24, 31)))
"""``tab:inrange``'s blocks: the model, its documents' name and the attacked layers."""


def inrange_cell(doc: Rec | None, layer: int, rng: str) -> dict | None:
    """``{changed, k_median, k_max, p_uniform, p_contribution}`` of one layer and range of an ``inrange_llm.py``
    document (medians over the prompts whose next token changed)."""
    if doc is None:
        return None
    rs = [r for r in doc.d.get("records", []) if r.get("layer") == layer and r.get("range") == rng]
    if not rs:
        return None
    ok = [r for r in rs if r.get("success")]
    med = lambda k: statistics.median([r[k] for r in ok]) if ok else None     # noqa: E731
    return {"changed": len(ok), "prompts": len(rs), "k_median": med("k"),
            "k_max": max(r["k"] for r in ok) if ok else None, "p_uniform": med("p_uniform"),
            "p_contribution": med("p_contribution"), "k_all": [r["k"] for r in ok],
            "p_uniform_all": [r["p_uniform"] for r in ok]}


def tab_inrange(ctx: Ctx) -> Table:
    """``tab:inrange``: in-range edits on OPT-1.3B and OPT-6.7B (``inrange_llm.py``): prompts whose next token
    changed, the median and largest number of edited neurons, the median detection per path under uniform and
    contribution-weighted sampling, for edits to either bound of the natural maximum or of the 99.9th percentile,
    and edits that only lower activations."""
    tb = Table("tab_inrange", "tab:inrange", "attack tier, inrange_llm.py documents (attack/inrange_llm_<model>_"
               "{any,down}.json); medians over the prompts whose next token changed; fp32")
    for i, (label, key, layers) in enumerate(INRANGE):
        if i:
            tb.rows.append(r"\midrule")
        any_, down = ctx.run.file(f"inrange_llm_{key}_any.json"), ctx.run.file(f"inrange_llm_{key}_down.json")
        first = True
        for kind, doc, rng in (("any, max range", any_, "max"), (r"any, 99.9\% range", any_, "p999"),
                               ("down only", down, "max")):
            for j, layer in enumerate(layers):
                c = inrange_cell(doc, layer, rng) or {}
                row = f"{label} {kind} {layer}"
                tb.rows.append(" & ".join([
                    label if first else "", kind if j == 0 else "", str(layer),
                    tb.put(row, "changed", c.get("changed"), doc),
                    tb.put(row, "k med.", c.get("k_median"), doc, lambda v: f"{v:g}"),
                    tb.put(row, "k max", c.get("k_max"), doc),
                    tb.put(row, "uniform", c.get("p_uniform"), doc, f.sci),
                    tb.put(row, "contribution", c.get("p_contribution"), doc, lambda v: f.sig(v, 2))]) + r" \\")
                first = False
    return tb


INRANGE_LAYERS = (6, 12, 18, 23)
"""The attacked layers of OPT-1.3B whose in-range edits the text and ``tab:paths`` quote."""


def inrange_ks(run: Run) -> dict:
    """The sizes of OPT-1.3B's in-range edits that the text (``\\InrangeKMedian``, ``\\InrangeDownKMedian``,
    ``\\InrangeKMax``) and ``tab:paths`` quote: ``median``, the median number of edited neurons over the successful
    edits of either direction (both ranges, :data:`INRANGE_LAYERS`); ``max``, the largest of them; ``down``, the
    median of the lowering-only edits (the natural maximum's range).  Each median is the lower one of an even count,
    so that it is an edit size that occurred (a whole number of neurons).  None where a cell is missing or no edit
    succeeded."""
    any_, down = run.file("inrange_llm_opt-1.3b_any.json"), run.file("inrange_llm_opt-1.3b_down.json")
    cells = [inrange_cell(any_, layer, rng) for layer in INRANGE_LAYERS for rng in ("max", "p999")]
    ks = [k for x in cells if x for k in x["k_all"]] if all(cells) else []
    dc = [inrange_cell(down, layer, "max") for layer in INRANGE_LAYERS]
    dk = [k for x in dc if x for k in x["k_all"]] if all(dc) else []
    return {"median": statistics.median_low(ks) if ks else None, "max": max(ks) if ks else None,
            "down": statistics.median_low(dk) if dk else None, "source": any_, "source_down": down}


PATHS_MODELS = ("opt-1.3b", "qwen3-4b", "llama2-7b", "llama2-13b", "opt-6.7b")
"""``tab:paths``' rows: the FFN layer's width ``N`` is the model's ``d_ff`` (``pvi.models.decoder.CONFIGS``)."""
PATHS_ACC_MODELS, PATHS_ACC_S = ("llama2-7b", "opt-6.7b"), (3, 4, 5)
"""``tab:paths-acc``' rows: the models and the paths ``s = 10^e``."""
PATHS_K_ILLUSTRATIVE = 16
"""The one edit size of ``tab:paths`` and ``tab:paths-acc`` that no measurement gives."""
PATHS_LAM = 40
"""The soundness target ``2^-40`` of ``tab:paths``."""


def paths_columns(run: Run) -> list[tuple[str, float | None, object]]:
    """The edit sizes ``k`` of ``tab:paths``' and ``tab:paths-acc``' columns, in the order of the paper's header:
    ``(column, k, source)``: one neuron (the edit measured on OPT-6.7B), the median in-range edit, the median
    lowering-only edit, :data:`PATHS_K_ILLUSTRATIVE` and the largest in-range edit (:func:`inrange_ks`)."""
    ks = inrange_ks(run)
    return [("k=1", 1, "the single-neuron edit"), ("k median", ks["median"], ks["source"]),
            ("k down", ks["down"], ks["source_down"]),
            (f"k={PATHS_K_ILLUSTRATIVE}", PATHS_K_ILLUSTRATIVE, "illustrative"), ("k max", ks["max"], ks["source"])]


def accept_text(log10p: float) -> str:
    """An acceptance probability of ``tab:paths-acc`` from its ``log10``: three decimals down to 0.01
    (``0.913``), then scientific notation with two digits (``$4.2\\cdot10^{-3}$``), and below ``10^-100`` the bound
    ``$<10^{-E}$`` with ``E`` a multiple of 50 (``$<10^{-200}$``)."""
    if log10p >= -2:
        return f"{10 ** log10p:.3f}"
    if log10p >= -100:
        return f.sci(10 ** log10p)
    return f"$<10^{{-{50 * math.floor(-log10p / 50)}}}$"


def tab_paths(ctx: Ctx) -> Table:
    """``tab:paths``: the uniform paths ``s_40(k, N)`` that catch an edit of ``k`` neurons in one FFN layer of width
    ``N`` with probability ``1 - 2^-40`` (Proposition ``prop:paths``), for the edit sizes of :func:`paths_columns`:
    the run's in-range edits of OPT-1.3B (the attack tier's ``inrange_llm.py`` documents) and two fixed sizes."""
    from experiments.attack.sok import paths_needed
    from pvi.models.decoder import CONFIGS
    cols = paths_columns(ctx.run)
    tb = Table("tab_paths", "tab:paths", "Proposition prop:paths, s_40(k, N) = ceil(40 / -log2(1 - k/N)); N = d_ff; "
               "the columns k = 1, the median in-range edit, the median lowering-only edit, 16, the largest in-range "
               "edit (attack tier, inrange_llm_opt-1.3b_{any,down}.json): "
               + ", ".join(f"{name} {'--' if k is None else f'{k:g}'}" for name, k, _ in cols))
    for m in PATHS_MODELS:
        n = CONFIGS[m].d_ff
        tb.rows.append(" & ".join([NAME[m], f.num(n)] + [
            tb.put(NAME[m], name, None if k is None else paths_needed(PATHS_LAM, k, n), src, f.num)
            for name, k, src in cols]) + r" \\")
    return tb


def tab_paths_acc(ctx: Ctx) -> Table:
    """``tab:paths-acc``: the probability ``(1 - k/N)^s`` that ``s`` uniform paths accept an edit of ``k`` neurons in
    one FFN layer (Proposition ``prop:paths``), for the edit sizes of :func:`paths_columns`."""
    from experiments.attack.sok import accept_log10
    from pvi.models.decoder import CONFIGS
    cols = paths_columns(ctx.run)
    tb = Table("tab_paths_acc", "tab:paths-acc", "Proposition prop:paths, (1 - k/N)^s; N = d_ff; the columns of "
               "tab:paths: " + ", ".join(f"{name} {'--' if k is None else f'{k:g}'}" for name, k, _ in cols))
    for i, m in enumerate(PATHS_ACC_MODELS):
        if i:
            tb.rows.append(r"\midrule")
        n = CONFIGS[m].d_ff
        for j, e in enumerate(PATHS_ACC_S):
            row = f"{NAME[m]} s=10^{e}"
            tb.rows.append(" & ".join([NAME[m] if j == 0 else "", f"$10^{e}$"] + [
                tb.put(row, name, None if k is None else accept_log10(k, n, 10 ** e), src, accept_text)
                for name, k, src in cols]) + r" \\")
    return tb


# ================================================================================================ ext_defence.tex
SETUP_MODELS = (("gpt2", "GPT-2"), ("llama2-7b", "Llama-2-7B"), ("opt-6.7b", "OPT-6.7B"), ("llama2-13b", "Llama-2-13B"))


def setup_values(policy: str) -> dict:
    """``{model: {a: setup_model(...)}}`` of ``costs/setup_bytes.py`` for ``tab:setup``'s models."""
    return {m: {a: setup_of(m, policy, a)[0] for a in (100, 1000)} for m, _ in SETUP_MODELS}


def tab_setup(ctx: Ctx) -> Table:
    """``tab:setup``: the one-time setup proof at ``a = 100`` and ``1,000`` amortised queries and the opened columns
    per query, honest and untrusted (the byte model of ``costs/setup_bytes.py``, interactive queries; the setup
    tier measures the times, which the text's ``Setup...`` macros carry)."""
    tb = Table("tab_setup", "tab:setup", f"the byte model of costs/setup_bytes.py (setup_model) under the plan "
               f"{ctx.policy}, lambda 128, interactive queries")
    v = setup_values(ctx.policy) if ctx.byte_model else {}
    src = f"costs/setup_bytes.setup_model ({ctx.policy})"
    mb2 = lambda x: f"{x:.2f}" if x < 10 else f"{x:.1f}"     # noqa: E731
    for m, label in SETUP_MODELS:
        s = v.get(m, {})
        a, c = s.get(100, {}), s.get(1000, {})
        get = lambda d, k, scale=1.0: None if k not in d else d[k] * scale    # noqa: E731
        tb.rows.append(" & ".join([
            label, tb.put(label, "setup a=100", get(a, "setup_MB", 1e6), src, f.b),
            tb.put(label, "setup a=1000", get(c, "setup_MB", 1e6), src, f.b),
            tb.put(label, "honest", get(a, "query_columns_MB_honest"), src, mb2),
            f"{tb.put(label, 'untrusted a=100', get(a, 'query_columns_MB_untrusted'), src, mb2)} / "
            f"{tb.put(label, 'untrusted a=1000', get(c, 'query_columns_MB_untrusted'), src, mb2)}"]) + r" \\")
    return tb


# ================================================================================================ ext_evaluation.tex
IMPROVE = (
    (r"Opening and fold on int8 GEMMs (prover, $\sC$, $T{=}64$)",
     (("llama2-7b", 1), ("llama2-7b", 4), ("llama2-13b", 1), ("gpt2", 12)), "open"),
    ("Weight products on int8 GEMMs (forward pass)", (("llama2-7b", 2),), "forward"),
    ("Threaded assembly of the encoded claims (encoder)", (("opt-1.3b", 24), ("llama2-7b", 2)), "encode"),
    ("Fused exact attention in the prover (forward pass)", (("opt-1.3b", 12), ("llama2-7b", 2)), "attn"),
    (r"GPU claim decoder and fused attention (GPU verifier, $\Kpre$)", (("opt-1.3b", 12), ("llama2-7b", 2)), "gpuv"),
    (r"Native exact kernels for every phase (CPU verifier, $\Kpre$)", (("llama2-7b", 2),), "cpuv"))
"""``tab:eval-improve``'s groups: heading, (model, blocks) rows and the A/B."""


def improve_pair(run: Run, kind: str, model: str, layers: int):
    """``(before, after, record before, record after)`` seconds of one ``tab:eval-improve`` row."""
    if kind == "open":
        a, b = (run.first("open_ab", "improve", model=model, layers=layers, variant=v) for v in ("float64", "int8"))
        return a and a.d["prove_s"], b and b.d["prove_s"], a, b
    if kind == "forward":
        a, b = (run.first("forward_ab", "improve", model=model, layers=layers, variant=v) for v in ("float32", "int8"))
        return a and a.d["prove_forward"], b and b.d["prove_forward"], a, b
    if kind == "encode":
        a, b = (run.first("encode_ab", "improve", model=model, layers=layers, variant=v)
                for v in ("one_thread", "threaded"))
        return a and a.d["encode_s"], b and b.d["encode_s"], a, b
    vdev = "cpu" if kind == "cpuv" else "cuda"
    a, b = (run.perf("improve", model, 2048, "Kpre", vdev=vdev, layers=layers, label=f"{kind}-{w}")
            for w in ("before", "after"))
    if kind == "attn":
        return (a and a.d["timings"].get("prove_forward"), b and b.d["timings"].get("prove_forward"), a, b)
    return a and a.d["verify_s"], b and b.d["verify_s"], a, b


def tab_eval_improve(ctx: Ctx) -> Table:
    """``tab:eval-improve``: each implementation improvement against the code path it replaces, on the same queries
    (the improve tier; the options of ``pvi.options`` select the path)."""
    tb = Table("tab_eval_improve", "tab:eval-improve", f"{protocol(ctx)}; improve tier: open_ab.py, forward_ab.py, "
               "encode_ab.py and perf.py with the labels <row>-before / <row>-after")
    for head, rows, kind in IMPROVE:
        tb.rows.append(rf"\multicolumn{{4}}{{@{{}}l}}{{\emph{{{head}}}}} \\")
        for model, layers in rows:
            before, after, ra, rb = improve_pair(ctx.run, kind, model, layers)
            row = f"{kind} {NAME[model]} {layers}/{FULL_LAYERS[model]}"
            tb.rows.append(" & ".join([
                f"{NAME[model]}, {layers}/{FULL_LAYERS[model]}", tb.put(row, "before", before, ra, f.t),
                tb.put(row, "after", after, rb, f.t),
                tb.put(row, "gain", before / after if before and after else None, rb, lambda v: f.x(v, 1))])
                + r" \\")
    return tb


MEMORY = (("Kpre", 64), ("Kpre", 2048), ("C", 64), ("C", 2048))
"""``tab:memory``'s rows: the cpuv tier's perf.py cells of the full Llama-2-7B."""


def tab_memory(ctx: Ctx) -> Table:
    """``tab:memory``: the memory of one query on the full Llama-2-7B (``cpuv`` tier, ``perf.py --client-memory``),
    in GB: the CPU client's own process, which holds only its key and embedding tables (Kpre) or the roots (C), the
    public graph and one proof (its peak resident set over the timed proofs, and its resident set after the
    imports), the prover's GPU peak over the timed queries, and the peak resident set of the measuring process,
    which runs the prover and the in-process verifier together."""
    tb = Table("tab_memory", "tab:memory", f"{protocol(ctx)}; cpuv tier: perf.py --verifier-device cpu "
               "--client-memory (label cpuv); client: client_memory.peak_rss and .baseline_rss; prover GPU: "
               "peak.gpu_peak_memory; process: peak.host_peak_rss; GB")
    gb = lambda v: f.fixed(v / 1e9, 1)                 # noqa: E731
    last = None
    for mode, seq in MEMORY:
        r = ctx.run.perf("cpuv", "llama2-7b", seq, mode, vdev="cpu", label="cpuv")
        client, peak = ((r.d.get("client_memory") or {}), (r.d.get("peak") or {})) if r else ({}, {})
        row = f"{mode} T={seq}"
        tb.rows.append(" & ".join([
            MODE[mode] if mode != last else "", f.tokens(seq),
            tb.put(row, "client", client.get("peak_rss"), r, gb),
            tb.put(row, "client after imports", client.get("baseline_rss"), r, gb),
            tb.put(row, "prover GPU", peak.get("gpu_peak_memory"), r, gb),
            tb.put(row, "process", peak.get("host_peak_rss"), r, gb)]) + r" \\")
        last = mode
    return tb


LONG = (2048, 4096, 8192, 16384)
"""``tab:eval-long``'s prompt lengths."""
LONG_BLOCKS = (2, 3, 4)
"""The long tier's builds (blocks) on the CPU."""
LONG_GPU_BLOCKS = (2, 3)
"""The gpuv tier's builds at long prompts."""
LONG_CHECK = (("cpu", 2048), ("cpu", 8192), ("cuda", 2048))
"""``tab:eval-long-check``'s rows: the verifier's device and the prompt length of a full model measured next to the
long prompts' fit (the cpuv tier's full model at 2,048 tokens, the long tier's at 8,192, the gpuv tier's at 2,048)."""
V1_LONG = (2048, 4096, 8192)
"""``tab:eval-v1-long``'s prompt lengths (V1 on 2 blocks, whole queries); the codec profile of the full model also
has 16,384 tokens (``\\VOneClaimsRatioSixteenK``)."""


def block_fit(points: dict, n_layers: int, blocks=None) -> tuple[float, float] | None:
    """The least-squares line ``t(L) = a + b L`` through the builds of ``points`` (``{blocks: value}``, keys as
    numbers or strings) smaller than the full model's ``n_layers`` (only those in ``blocks`` if given):
    ``(b, a + n_layers b)``, the per-block increment and the full model, or None with fewer than two builds."""
    pts = sorted((int(k), float(v)) for k, v in points.items()
                 if v is not None and int(k) < n_layers and (blocks is None or int(k) in blocks))
    if len({n for n, _ in pts}) < 2:
        return None
    mn = sum(n for n, _ in pts) / len(pts)
    mv = sum(v for _, v in pts) / len(pts)
    slope = sum((n - mn) * (v - mv) for n, v in pts) / sum((n - mn) ** 2 for n, _ in pts)
    return slope, mv + (n_layers - mn) * slope


def long_point(run: Run, seq: int, n_layers: int = 32, *, gpu: bool = False, blocks=None) -> dict:
    """The long prompts at one length: the verifier (Kpre) and the proof of Llama-2-7B's shapes on the builds of
    :data:`LONG_BLOCKS` (the long tier, the CPU verifier; ``gpu``: the gpuv tier's streaming GPU verifier on
    :data:`LONG_GPU_BLOCKS`, label ``long-gpuv``), extrapolated to the full model by a least-squares line over the blocks (:func:`block_fit`;
    ``blocks``: only those builds); the same for each re-execution variant on the same device (``variants``:
    ``{variant: (seconds, record)}``; ``reexec``: the fastest exact one, ``(seconds, variant, record)``; on the GPU
    the integer pass, ``int``, and the floating-point one, ``float``, in fp16); ``cells``: the perf records by
    blocks, ``src`` the largest build's."""
    tier, vdev, label = ("gpuv", "cuda", "long-gpuv") if gpu else ("long", "cpu", "long")
    cells = {n: run.perf(tier, "llama2-7b", seq, "Kpre", vdev=vdev, layers=n, label=label)
             for n in (LONG_GPU_BLOCKS if gpu else LONG_BLOCKS)}
    cells = {n: r for n, r in cells.items() if r and (blocks is None or n in blocks)}
    out = {"cells": cells, "src": cells[max(cells)] if cells else None}
    v = block_fit({n: r.d["verify_s"] for n, r in cells.items()}, n_layers)
    b = block_fit({n: r.d["proof_bytes"] for n, r in cells.items()}, n_layers)
    if v and b:
        out["verify_block"], out["verify"] = v
        out["bytes_block"], out["bytes"] = b
    best = None
    for var in (("int", "float") if gpu else EXACT):
        ms, r = run.reexec_blocks(tier, "llama2-7b", seq, "cuda" if gpu else "cpu", var,
                                  keep=None if gpu else exact_record)
        fit = block_fit(ms, n_layers, blocks)
        if fit is None:
            continue
        out.setdefault("variants", {})[var] = (fit[1], r)
        if var in EXACT and (best is None or fit[1] < best[0]):
            best = (fit[1], var, r)
    out["reexec"] = best
    return out


def long_measured(run: Run, device: str, seq: int) -> dict:
    """The full Llama-2-7B measured at one of :data:`LONG_CHECK`'s points: ``perf`` the verifier's record (Kpre),
    ``reexec`` the fastest exact re-execution on the same device, ``(seconds, variant, record)`` (on the CPU of the
    records with glibc's allocator defaults, as the long tier's)."""
    m = "llama2-7b"
    if device == "cuda":
        perf = run.perf("gpuv", m, seq, "Kpre", vdev="cuda", label="gpuv", stream=True)
        got = run.reexec("gpuv", m, seq, "cuda", ("int",))
    elif seq == 2048:
        perf = run.perf("cpuv", m, seq, "Kpre", vdev="cpu", label="cpuv")
        got = run.reexec("cpuv", m, seq, "cpu", EXACT,
                         keep=lambda v, r: exact_record(v, r) and not r.d.get("malloc_tuned"))
    else:
        perf = run.perf("long", m, seq, "Kpre", vdev="cpu", layers=None, label="long")
        got = run.reexec("long", m, seq, "cpu", EXACT, keep=exact_record)
    best = min(((s, v, r) for v, (s, r) in got.items()), key=lambda x: x[0]) if got else None
    return {"perf": perf, "reexec": best}


def long_errors(run: Run) -> dict:
    """``{(device, seq): {"verify", "reexec", "bytes": (measured, extrapolated, relative error) or None, "src"}}``
    at the points of :data:`LONG_CHECK`, and ``("cpu", seq, "two-three")``: the verifier's ``(measured,
    extrapolated, relative error)`` when only the 2- and 3-block builds are fitted (the earlier extrapolation)."""
    out: dict = {}
    for dev, seq in LONG_CHECK:
        p, m = long_point(run, seq, gpu=dev == "cuda"), long_measured(run, dev, seq)
        pr, rx = m["perf"], m["reexec"]
        pairs = {"verify": (pr and pr.d["verify_s"], p.get("verify")),
                 "reexec": (rx and rx[0], p["reexec"] and p["reexec"][0]),
                 "bytes": (pr and pr.d["proof_bytes"], p.get("bytes"))}
        out[(dev, seq)] = {k: (a, e, (e - a) / a) if a and e is not None else None for k, (a, e) in pairs.items()}
        out[(dev, seq)]["src"] = pr
        if dev == "cpu":
            old, a = long_point(run, seq, blocks=(2, 3)), pr and pr.d["verify_s"]
            out[(dev, seq, "two-three")] = ((a, old["verify"], (old["verify"] - a) / a)
                                            if a and old.get("verify") is not None else None)
    return out


def tab_eval_long(ctx: Ctx) -> Table:
    """``tab:eval-long``: long prompts on the full Llama-2-7B, extrapolated from 2, 3 and 4 blocks: the CPU verifier
    (Kpre, Fiat--Shamir) against exact re-execution on the same 8 threads, and the proof."""
    tb = Table("tab_eval_long", "tab:eval-long", f"{protocol(ctx)}; long tier: perf.py --layers 2, 3 and 4 (label "
               "long), reexec.py --layers 2 3 4; full model = the least-squares line through the blocks at 32 "
               "blocks; re-execution: the fastest exact variant of the integer forward pass")
    for seq in LONG:
        p = long_point(ctx.run, seq)
        rx = p["reexec"]
        row = f"T={seq}"
        ratio = rx[0] / p["verify"] if rx and p.get("verify") else None
        tb.rows.append(" & ".join([
            f.tokens(seq), tb.put(row, "verifier", p.get("verify"), p["src"], lambda v: f.sig(v, 3)),
            tb.put(row, "re-execution", rx and rx[0], rx and rx[2], lambda v: f.num(v, 1)),
            tb.put(row, "ratio", ratio, p["src"], lambda v: f.x(v, 1)),
            tb.put(row, "proof GB", p.get("bytes"), p["src"], lambda v: f.fixed(v / 1e9, 1))]) + r" \\")
    return tb


def tab_eval_long_check(ctx: Ctx) -> Table:
    """``tab:eval-long-check``: the full Llama-2-7B measured against the long prompts' fit (:func:`long_point`) at
    the points of :data:`LONG_CHECK`: the verifier, the fastest exact re-execution and the proof, each as ``measured /
    extrapolated``."""
    tb = Table("tab_eval_long_check", "tab:eval-long-check",
               f"{protocol(ctx)}; measured: perf.py and reexec.py on the full model (the cpuv tier at 2,048 tokens, "
               "the long tier at 8,192, the gpuv tier's streaming GPU verifier and GPU re-execution at 2,048); "
               "extrapolated: the fit of tab:eval-long (CPU: 2, 3 and 4 blocks) and tab:eval-long-gpu (GPU: 2 and "
               "3)\ncolumns: device & T & verifier (s) measured / extrapolated & exact re-execution (s) measured / "
               "extrapolated & proof (GB) measured / extrapolated")
    err = long_errors(ctx.run)
    last = None
    for dev, seq in LONG_CHECK:
        e = err[(dev, seq)]
        row = f"{dev} T={seq}"
        name = "CPU" if dev == "cpu" else "GPU"
        cells = [name if name != last else "", f.tokens(seq)]
        last = name
        for key, fmt in (("verify", lambda v: f.sig(v, 3)), ("reexec", lambda v: f.sig(v, 3)),
                         ("bytes", lambda v: f.fixed(v / 1e9, 2))):
            pair = e[key]
            cells.append(tb.put(row, f"{key} measured", pair and pair[0], e["src"], fmt) + " / "
                         + tb.put(row, f"{key} extrapolated", pair and pair[1], e["src"], fmt))
        tb.rows.append(" & ".join(cells) + r" \\")
    return tb


def tab_eval_long_gpu(ctx: Ctx) -> Table:
    """``tab:eval-long-gpu``: the streaming GPU verifier at long prompts against re-execution on the same GPU (the
    integer pass with int8 GEMMs, exact, and fp16), the full Llama-2-7B extrapolated from 2 and 3 blocks, and the
    process's GPU peak with 3 blocks (the prover and the verifier share the card)."""
    tb = Table("tab_eval_long_gpu", "tab:eval-long-gpu",
               f"{protocol(ctx)}; gpuv tier: perf.py --layers 2 and 3 --verifier-device cuda --stream (label "
               "long-gpuv), reexec.py --device cuda --layers 2 3 (int; float, in fp16); full model = the line "
               "through the blocks at 32 blocks; GPU peak: the 3-block cell's process (prover and verifier)\n"
               "columns: T & GPU verifier (s) & int8 re-execution (s) & fp16 re-execution (s) & ratio int8 / "
               "verifier & GPU peak (GB)")
    for seq in LONG:
        p = long_point(ctx.run, seq, gpu=True)
        var = p.get("variants") or {}
        i8, f16 = var.get("int"), var.get("float")
        row = f"T={seq}"
        three = p["cells"].get(3)
        peak = three and (three.d.get("peak") or {}).get("gpu_peak_memory")
        tb.rows.append(" & ".join([
            f.tokens(seq), tb.put(row, "GPU verifier", p.get("verify"), p["src"], lambda v: f.sig(v, 3)),
            tb.put(row, "int8", i8 and i8[0], i8 and i8[1], lambda v: f.sig(v, 3)),
            tb.put(row, "fp16", f16 and f16[0], f16 and f16[1], lambda v: f.sig(v, 3)),
            tb.put(row, "ratio", i8 and p.get("verify") and i8[0] / p["verify"], p["src"], lambda v: f.x(v, 1)),
            tb.put(row, "GPU peak", peak, three, lambda v: f.fixed(v / 1e9, 1))]) + r" \\")
    return tb


def v1_long(run: Run, seq: int) -> tuple[Rec | None, Rec | None]:
    """V1 at a long prompt: the long tier's 2-block record (``v1_bytes.py --full``, label ``long-v1``) and the v1
    tier's codec profile of the full model (label ``v1-codec``)."""
    two = run.v1("long", "llama2-7b", seq, "Kpre", vdev="cpu", label="long-v1", layers=2)
    return two, run.codec("v1", "llama2-7b", seq, layers=FULL_LAYERS["llama2-7b"], label="v1-codec")


def codec_ratio(rec: Rec | None) -> float | None:
    """A codec profile's claims ratio: v0's claims over V1's values (the cut ops' requantised values and the other
    ops' claims; the exceptions are a frame of their own), as the break-even's V1 ratio counts them."""
    t = (rec.d.get("total") or {}) if rec else {}
    return t["v0_claims"] / t["v1_values"] if t.get("v0_claims") and t.get("v1_values") else None


def tab_eval_v1_long(ctx: Ctx) -> Table:
    """``tab:eval-v1-long``: V1 at long prompts on 2 Llama-2-7B blocks (whole queries: v0's and V1's proofs, V1's
    prover and CPU verifier), and the claims ratio of the full model from its codec profile."""
    tb = Table("tab_eval_v1_long", "tab:eval-v1-long",
               f"{protocol(ctx)}; long tier: v1_bytes.py --full --layers 2 (label long-v1; the CPU verifier, the GKR "
               "on the GPU); v1 tier: v1_bytes.py --codec-only on the full model (label v1-codec)\n"
               "columns: T & v0 proof (MB) & V1 proof (MB) & ratio & V1 prove (s) & V1 verify (s) & full model: "
               "claims ratio (codec profile)")
    for seq in V1_LONG:
        two, codec = v1_long(ctx.run, seq)
        p1, v1s = v1_timed(two, "v1")
        row = f"T={seq}"
        tb.rows.append(" & ".join([
            f.tokens(seq), tb.put(row, "v0 MB", two and two.d["v0_total"], two, lambda v: f.num(v / 1e6, 0)),
            tb.put(row, "V1 MB", two and two.d["v1_total"], two, lambda v: f.num(v / 1e6, 0)),
            tb.put(row, "ratio", two and two.d["v0_total"] / two.d["v1_total"], two, lambda v: f.x(v, 2)),
            tb.put(row, "V1 prove", p1, two, lambda v: f.sig(v, 3)),
            tb.put(row, "V1 verify", v1s, two, lambda v: f.sig(v, 3)),
            tb.put(row, "full claims ratio", codec_ratio(codec), codec, lambda v: f.x(v, 2))]) + r" \\")
    return tb


def real_pair(run: Run, model: str, seq: int, mode: str, checkpoint: str):
    """The real tier's V1 records of one cell: real weights and the random-weight twin."""
    real = run.v1("real", model, seq, mode, vdev="cpu", label="real", real=checkpoint)
    rnd = run.v1("real", model, seq, mode, vdev="cpu", label="real-random", real=None)
    return real, rnd


def tab_eval_real(ctx: Ctx) -> Table:
    """``tab:eval-real``: the certified OPT-1.3B (full model, last block pruned) against its random-weight twin:
    v0's and V1's proofs and v0's prover and CPU verifier on the same queries (``v1_bytes.py --full``'s timed v0
    queries)."""
    tb = Table("tab_eval_real", "tab:eval-real", f"{protocol(ctx)}; real tier: v1_bytes.py --full on the certified "
               "facebook/opt-1.3b (label real) and its random twin (label real-random); prove/verify: v0's timed "
               "queries")
    for mode in ("C", "Kpre"):
        for seq in (64, 2048):
            real, rnd = real_pair(ctx.run, "opt-1.3b", seq, mode, "facebook/opt-1.3b")
            row = f"{mode} T={seq}"
            tim = [v1_timed(r, "v0") for r in (real, rnd)]
            cells = [MODE[mode], f.tokens(seq)]
            for key, fmt in (("v0_total", lambda v: f.sig(v / 1e6, 4)), ("v1_total", lambda v: f.sig(v / 1e6, 4))):
                cells.append(" / ".join(tb.put(row, f"{key} {w}", r and r.d.get(key), r, fmt)
                                        for w, r in (("real", real), ("random", rnd))))
            for i, name in ((0, "prove"), (1, "verify")):
                cells.append(" / ".join(tb.put(row, f"{name} {w}", t[i], r, lambda v: f.fixed(v, 2))
                                        for w, t, r in (("real", tim[0], real), ("random", tim[1], rnd))))
            tb.rows.append(" & ".join(cells) + r" \\")
    return tb


REEXEC_GPU = ("opt-1.3b", "opt-6.7b", "llama2-7b", "llama2-13b", "qwen3-4b")
REEXEC_CPU = (64, 512, 2048)
"""``tab:eval-reexec``'s prompt lengths of the CPU rows (Llama-2-7B, the cpuv tier)."""


def tab_eval_reexec(ctx: Ctx) -> Table:
    """``tab:eval-reexec``: re-executing the full model on the GPU (the integer forward pass on int8 tensor cores /
    the fp16 decoder) and on the CPU (Llama-2-7B: our Kpre verifier, fp32 and exact re-execution, the fastest of
    :data:`EXACT`, and llama.cpp with Q8_0 and bf16 weights, each its fastest configuration)."""
    run = ctx.run
    tb = Table("tab_eval_reexec", "tab:eval-reexec", f"{protocol(ctx)}; gpuv tier: reexec.py --device cuda (int, "
               "float = fp16), full models measured; cpuv tier: perf.py Kpre, reexec.py and llamacpp.py (random "
               "weights, the fastest of llama.cpp's attention kernels and micro-batches) on 8 threads")
    tb.rows.append(r"\multicolumn{4}{@{}l}{\emph{GPU: integer forward pass / fp16 decoder}} \\")
    for m in REEXEC_GPU:
        cells = [NAME[m]]
        for seq in (64, 512, 2048):
            g = run.reexec("gpuv", m, seq, "cuda", ("int", "float"))
            i, fl = g.get("int"), g.get("float")
            a = tb.put(f"{m} {seq}", "GPU int", i and i[0], i and i[1], lambda v: f.fixed(v, 2))
            c = tb.put(f"{m} {seq}", "GPU fp16", fl and fl[0], fl and fl[1], lambda v: f.sig(v, 2))
            cells.append(f.DASH if i is None and fl is None else f"{a} / {c}")
        tb.rows.append(" & ".join(cells) + r" \\")
    tb.rows += [r"\midrule", r"\multicolumn{4}{@{}l}{\emph{CPU: Llama-2-7B}} \\"]
    ver = [run.perf("cpuv", "llama2-7b", s, "Kpre", vdev="cpu", label="cpuv") for s in REEXEC_CPU]
    vals = [tb.put(f"cpu {s}", "Kpre verifier", r and r.d["verify_s"], r, lambda v: f.fixed(v, 1))
            for s, r in zip(REEXEC_CPU, ver)]
    tb.rows.append(" & ".join([r"our $\Kpre$ verifier", *vals]) + r" \\")
    for label, variants in (("fp32 re-execution", ("float",)), ("integer re-execution", EXACT),
                            (r"llama.cpp Q8\_0", ("llamacpp_q8_0",)), ("llama.cpp bf16", ("llamacpp_bf16",))):
        vals = []
        for s in REEXEC_CPU:
            got = run.reexec("cpuv", "llama2-7b", s, "cpu", variants, keep=exact_record)
            best = min(got.values(), key=lambda v: v[0]) if got else None
            vals.append(tb.put(f"cpu {s}", label, best and best[0], best and best[1], lambda v: f.fixed(v, 1)))
        tb.rows.append(" & ".join([label, *vals]) + r" \\")
    return tb


COMPARE_V1 = (("zkGPT", ("zkGPT", "GPT-2", ""), "GPT-2 (64)", "gpt2", 64),
              ("DeepProve", ("DeepProve", "GPT-2", "64"), "GPT-2 (64)", "gpt2", 64),
              ("DeepProve", ("DeepProve", "GPT-2", "512"), "GPT-2 (512)", "gpt2", 512),
              ("zkLLM", ("zkLLM", "OPT-125M", "2048"), "OPT-125M (2{,}048)", "opt-125m", 2048),
              ("zkLLM", ("zkLLM", "OPT-1.3B", "2048"), "OPT-1.3B (2{,}048)", "opt-1.3b", 2048),
              ("zkLLM", ("zkLLM", "Llama-2-7B", "2048"), "Llama-2-7B (2{,}048)", "llama2-7b", 2048))
"""``tab:eval-compare-v1``'s rows: (system, its row in ``reported_curated.csv``, the model label, ours).  Every row is
a full model of the v1 tier (Llama-2-13B's V1 cell is a build of 2 of its 40 blocks, so it has no row)."""


def tab_eval_compare_v1(ctx: Ctx) -> Table:
    """``tab:eval-compare-v1``: V1's proof against the published systems (mode C, the full model, Fiat--Shamir; v0
    on the same query)."""
    tb = Table("tab_eval_compare_v1", "tab:eval-compare-v1", f"{protocol(ctx)}; v1 tier, mode C, full models only "
               "(v0 and V1 on the same query); theirs: reported_curated.csv")
    for system, key, label, m, seq in COMPARE_V1:
        r = v1_cell(ctx.run, m, seq, "C")
        if r and r.d.get("layers") != FULL_LAYERS[m]:
            r = None
        lit = ctx.lit.get(key) or {}
        theirs = f.to_float(lit.get("proof_bytes"))
        tb.rows.append(" & ".join([system, label, f.b(theirs) if theirs else f.DASH,
                                   tb.put(label, "ours (v0)", r and r.d["v0_total"], r, f.b),
                                   tb.put(label, "V1", r and r.d["v1_total"], r, f.b)]) + r" \\")
    return tb


CPU_ROWS = (("Llama-2-7B", "2 threads", "cpux", "llama2-7b", 2, True, "cpux"),
            ("", "4 threads", "cpux", "llama2-7b", 4, True, "cpux"),
            ("", "8 threads", "cpuv", "llama2-7b", 8, True, "cpuv"),
            ("", "8, no AVX-512 attention", "cpux", "llama2-7b", 8, False, "cpux-noavx512"),
            ("Llama-2-13B", "8 threads", "cpux", "llama2-13b", 8, True, "cpux"))
"""``tab:eval-cpu``'s rows: (model label, setting, tier, model, threads, the AVX-512 attention kernel, perf.py's
label), each at 2,048 tokens (:func:`cpu_point`); the 8-thread Llama-2-7B row is the cpuv tier's (``CpuRatio``)."""


def tab_eval_cpu(ctx: Ctx) -> Table:
    """``tab:eval-cpu``: the Kpre CPU verifier against the fastest exact re-execution on the same cores, at 2,048
    tokens: Llama-2-7B on 2, 4 and 8 threads (each pinned to as many cores of one NUMA node), on 8 threads without
    the AVX-512 attention kernel (in both), and Llama-2-13B on 8 threads."""
    tb = Table("tab_eval_cpu", "tab:eval-cpu", f"{protocol(ctx)}; cpux tier: perf.py Kpre --threads 2/4 (label cpux), "
               "--no-native-avx512 (label cpux-noavx512), Llama-2-13B (label cpux), reexec.py's exact variants on the "
               "same threads and options; the 8-thread Llama-2-7B row: the cpuv tier's cells")
    for name, setting, tier, m, threads, avx, label in CPU_ROWS:
        ver, rx, ratio = cpu_point(ctx.run, tier, m, 2048, threads, avx, label)
        row = f"{NAME[m]} {setting}"
        tb.rows.append(" & ".join([
            name, setting, tb.put(row, "verifier", ver and ver.d["verify_s"], ver, lambda v: f.sig(v, 3)),
            tb.put(row, "re-execution", rx and rx[0], rx and rx[2], lambda v: f.sig(v, 3)),
            tb.put(row, "ratio", ratio, ver, lambda v: f.x(v, 1))]) + r" \\")
    return tb


PLANS = ("tightc", "cnn16c", "cnn17c", "cnn18c", "R16c", "R64c")
"""``tab:eval-plans``' rows: mode C's candidate plans (``pvi.commit.plans.AUTO_CANDIDATES``)."""
PLAN_NAME = {"tightc": "tight", "cnn16c": r"$n\ge2^{16}$", "cnn17c": r"$n\ge2^{17}$", "cnn18c": r"$n\ge2^{18}$",
             "R16c": "rate 16", "R64c": "rate 64"}
"""Each candidate plan as the paper names it (every one with the transposed layout and lookup tables)."""
PLANS_LABEL = "plans"
"""The label of the cpux tier's ``perf.py`` cell that times every candidate plan side by side: its records name no
plan of the run (``numbers.record_plans``)."""


def plan_cell(run: Run, policy: str, model: str = "llama2-7b", seq: int = 2048) -> dict | None:
    """One plan of the cpux tier's side-by-side cell (``perf.py --policy ...``, label ``plans``): ``{trees, columns,
    commit, proof, prove, verify, record}`` with ``trees`` the Merkle trees of encoded matrices (``group_columns``),
    ``columns`` the columns they open together and ``commit`` the commitment's time (None if it came from the setup
    cache, which skips the encoding); None without the record."""
    r = run.perf("cpux", model, seq, "C", vdev="cpu", label=PLANS_LABEL, policy=policy)
    if r is None:
        return None
    groups = r.d.get("group_columns") or {}
    return {"trees": len(groups) or None, "columns": sum(groups.values()) if groups else r.d.get("columns"),
            "commit": r.d.get("commit_s") if r.d.get("setup_cache") != "hit" else None,
            "proof": r.d.get("proof_bytes"), "prove": r.d.get("prove_s"), "verify": r.d.get("verify_s"), "record": r}


def tab_eval_plans(ctx: Ctx) -> Table:
    """``tab:eval-plans``: mode C's candidate plans on the same Llama-2-7B queries at 2,048 tokens (the CPU verifier
    on 8 threads): the trees and the columns they open, the commitment's one-time encoding, the proof and the times."""
    tb = Table("tab_eval_plans", "tab:eval-plans", f"{protocol(ctx)}; cpux tier: perf.py --modes C --policy "
               f"{' '.join(PLANS)} (label plans) on the same queries, no setup cache")
    for p in PLANS:
        c = plan_cell(ctx.run, p) or {}
        r = c.get("record")
        tb.rows.append(" & ".join([
            PLAN_NAME[p], tb.put(p, "trees", c.get("trees"), r, str),
            tb.put(p, "columns", c.get("columns"), r, lambda v: f.num(v)),
            tb.put(p, "commit s", c.get("commit"), r, lambda v: f.sig(v, 3)),
            tb.put(p, "proof GB", c.get("proof"), r, lambda v: f.fixed(v / 1e9, 2)),
            tb.put(p, "prover s", c.get("prove"), r, lambda v: f.sig(v, 3)),
            tb.put(p, "verifier s", c.get("verify"), r, lambda v: f.sig(v, 3))]) + r" \\")
    return tb


LAMBDA_ROWS = ((64, 40), (64, 80), (64, 128), (2048, 40), (2048, 80), (2048, 128))
"""``tab:eval-lambda``'s rows: Llama-2-7B's prompt length and the security level (lambda 128: ``tab:eval-llm``'s
cells)."""


def tab_eval_lambda(ctx: Ctx) -> Table:
    """``tab:eval-lambda``: the full Llama-2-7B at lambda 40, 80 and 128 (the bench tier, Fiat--Shamir, the CPU
    verifier on 8 threads): prover, verifier and proof in modes C and Kpre."""
    tb = Table("tab_eval_lambda", "tab:eval-lambda", f"{protocol(ctx)} (lambda as the row states); bench tier: "
               "costs/bench.py llm --lams 40 80 (b-lam-*) and the lambda-128 cells, modes C:fs and Kpre:fs")
    for seq, lam in LAMBDA_ROWS:
        cells = [f.tokens(seq) if lam == LAMBDA_ROWS[0][1] else "", str(lam)]
        for mode, base in (("C", "c_cpu"), ("Kpre", "kpre_cpu")):
            cell = base if lam == 128 else f"{base}_lam{lam}"
            cost = ctx.bench.llm_cost("llama2-7b", seq, cell) or {}
            row, src = f"T={seq} lambda={lam}", cost.get("cell") or ctx.bench.name
            cells += [tb.put(row, f"{mode} prover", cost.get("prove"), src, lambda v: f.sig(v, 3)),
                      tb.put(row, f"{mode} verifier", cost.get("verify"), src, lambda v: f.sig(v, 3)),
                      tb.put(row, f"{mode} proof", cost.get("bytes"), src, f.b)]
        tb.rows.append(" & ".join(cells) + r" \\")
    return tb


# ================================================================================================ ext_v1.tex
def tab_v1(ctx: Ctx) -> Table:
    """``tab:v1``: V1's and v0's prover and CPU verifier on the timed cells of ``tab:eval-v1``, with the GKR's share
    of V1's prover (median of ``prove_gkr``)."""
    tb = Table("tab_v1", "tab:v1", f"{protocol(ctx, 'V1 with its optimised verifier')}; v1 tier, timed queries")
    for seq, m in V1_ROWS[1:]:
        for i, mode in enumerate(("C", "Kpre")):
            r = v1_cell(ctx.run, m, seq, mode)
            row = f"{NAME[m]} T={seq} {mode}"
            p0, v0 = v1_timed(r, "v0")
            p1, v1 = v1_timed(r, "v1", "optimised")
            q = ((r.d.get("timed") or {}).get("v1_paths") or {}).get("optimised") if r else None
            gkr = (q or {}).get("timings", {}).get("prove_gkr") if q else None
            two = lambda v: f.fixed(v, 2)     # noqa: E731
            tb.rows.append(" & ".join([NAME[m] if i == 0 else "", str(seq) if i == 0 else "", MODE[mode],
                                       tb.put(row, "v0 prover", p0, r, two), tb.put(row, "V1 prover", p1, r, two),
                                       tb.put(row, "GKR", gkr, r, two), tb.put(row, "v0 verifier", v0, r, two),
                                       tb.put(row, "V1 verifier", v1, r, two)]) + r" \\")
    return tb


# ================================================================================================ ext_generation.tex
GEN = (("gpt2", 64, (256,), ("greedy",)), ("llama2-7b", 512, (64, 256), ("greedy", "sample")),
       ("qwen3-4b", 512, (64, 256), ("greedy", "sample")))
"""The generation cells of the final run (the gen tier): per model the prompt's tokens ``P``, the responses' tokens
``G`` and the decoding rules (GPT-2: the per-token comparison with zkAgent; Llama-2-7B and Qwen3-4B: a 512-token
prompt, greedy and sampled)."""


def tab_gen(ctx: Ctx) -> Table:
    """``tab:gen``: a G-token greedy response as one prefill: prover, verifier and proof in modes C and Kpre."""
    tb = Table("tab_gen", "tab:gen", f"{protocol(ctx)}; gen tier (gen_bench.py, label gen): GPT-2 after a 64-token "
               "prompt, Llama-2-7B and Qwen3-4B after a 512-token prompt, greedy, random weights, full models; medians "
               "over the timed queries; the verifier's time includes the token rule")
    for m, prompt, gs, _ in GEN:
        for j, g in enumerate(gs):
            cells = [rf"{NAME[m]} ($P{{=}}{prompt}$)" if j == 0 else "", str(g)]
            for mode in ("C", "Kpre"):
                r = ctx.run.gen(m, mode, "greedy", g, prompt=prompt)
                row = f"{NAME[m]} G={g} {mode}"
                cells += [tb.put(row, "P", r and r.d["prove_s"], r, lambda v: f.fixed(v, 2)),
                          tb.put(row, "V", r and r.d["verify_s"], r, lambda v: f.fixed(v, 2)),
                          tb.put(row, "MB", r and r.d["proof_bytes"], r, lambda v: f.sig(v / 1e6, 3))]
            tb.rows.append(" & ".join(cells) + r" \\")
    return tb


def tab_gen_full(ctx: Ctx) -> Table:
    """``tab:gen-full``: every generation cell of the final run, greedy and sampled (the client's seed), with the
    proof per generated token."""
    tb = Table("tab_gen_full", "tab:gen-full", f"{protocol(ctx)}; gen tier (gen_bench.py, label gen): GPT-2 after a "
               "64-token prompt (greedy), Llama-2-7B and Qwen3-4B after a 512-token prompt, random weights, full "
               "models, 3 timed queries per cell; rows ', sampled': the integer sampler")
    for i, (m, prompt, gs, decodes) in enumerate(GEN):
        if i:
            tb.rows.append(r"\midrule")
        for mode in ("C", "Kpre"):
            for decode in decodes:
                for j, g in enumerate(gs):
                    r = ctx.run.gen(m, mode, decode, g, prompt=prompt)
                    row = f"{NAME[m]} {mode} {decode} G={g}"
                    label = f"{NAME[m]}, {MODE[mode]}" + (", sampled" if decode == "sample" else "")
                    tb.rows.append(" & ".join([
                        label if j == 0 else "", str(g),
                        tb.put(row, "prover", r and r.d["prove_s"], r, lambda v: f.fixed(v, 2)),
                        tb.put(row, "verifier", r and r.d["verify_s"], r, lambda v: f.fixed(v, 2)),
                        tb.put(row, "proof MB", r and r.d["proof_bytes"], r, lambda v: f.sig(v / 1e6, 3)),
                        tb.put(row, "MB/token", r and r.d["proof_mb_per_token"], r,
                               lambda v: f.fixed(v, 2) if v < 10 else f.sig(v, 3))]) + r" \\")
    return tb


ATTRIBUTION_MODEL = "facebook/opt-125m"
ATTRIBUTION_ROWS = (("LayerNorm outputs (all)", "norm"), (r"\quad before attention", "norm1"),
                    (r"\quad before the MLP", "norm2"), ("All weight matrices", "weights"),
                    (r"\quad q/k/v weights", "w_qkv"), ("fc1 requantisation", "rq_fc1"),
                    ("Attention output requantisation", "att_out"), ("Softmax (exp table, 8-bit prob.)", "softmax"),
                    ("Residual stream", "res"), ("ReLU table", "relu"))
"""``tab:quality-attr``'s rows: the label and ``attribution.py``'s group (``hybrid_opt.GROUPS``)."""
ATTRIBUTION_GROUPS = ("weights", "w_emb", "w_qkv", "w_out", "w_fc1", "w_fc2", "w_head", "norm", "norm1", "norm2",
                      "normf", "rq_qkv", "rq_fc1", "softmax", "att_out", "res", "relu")
"""Every group of ``attribution.py`` (the keys of ``hybrid_opt.GROUPS``, in its order)."""


def tab_quality_attr(ctx: Ctx) -> Table:
    """``tab:quality-attr``: the attribution of OPT-125M's perplexity, per group of components the leave-one-out
    perplexity (that group float, the rest integer) and the add-one perplexity (only that group integer) of the
    float copy of the integer graph (``attribution.py``: the scalar graph with norm gain 4 and 8-bit probabilities,
    16 windows of 128 tokens, calibration on one 128-token window at the end of the test split)."""
    check = ctx.run.attribution(ATTRIBUTION_MODEL, part="check")
    tb = Table("tab_quality_attr", "tab:quality-attr", "quality tier, attribution.py: WikiText-2 test split joined "
               "without a separator, 16 windows of 128 tokens, calibration on one 128-token window at its end; graph "
               f"{ATTRIBUTION_CELL}" + (f", digest {check.d.get('digest')}" if check else ""))
    for label, group in ATTRIBUTION_ROWS:
        loo, add = (ctx.run.attribution(ATTRIBUTION_MODEL, group, m) for m in ("leave_one_out", "add_one"))
        tb.rows.append(" & ".join([label, tb.put(label, "leave-one-out", loo and loo.d["ppl"], loo,
                                                 lambda v: f.fixed(v, 1)),
                                   tb.put(label, "add-one", add and add.d["ppl"], add, lambda v: f.fixed(v, 1))])
                       + r" \\")
    return tb


ZEROSHOT = (("lambada_openai", "LAMBADA"), ("hellaswag", "HellaSwag"), ("piqa", "PIQA"), ("arc_easy", "ARC-e"),
            ("arc_challenge", "ARC-c"), ("winogrande", "WinoGrande"))
ZEROSHOT_MODELS = ("facebook/opt-125m", "facebook/opt-1.3b", "facebook/opt-6.7b")


def zeroshot_cells(run: Run, model: str, recipe: str = "certified") -> dict:
    """``{task: (fp16 acc, int - fp16, 95% interval, same picks, int record)}`` of ``zeroshot.py`` (accuracy)."""
    out = {}
    for task, _ in ZEROSHOT:
        ref, got = run.zeroshot(model, task, "ref"), run.zeroshot(model, task, "int", recipe)
        if ref is None or got is None:
            out[task] = (None, None, None, None, got or ref)
            continue
        out[task] = (ref.d["acc"], got.d.get("diff_acc", got.d["acc"] - ref.d["acc"]), got.d.get("diff_acc_ci95"),
                     got.d.get("same_pick"), got)
    return out


def tab_zeroshot(ctx: Ctx) -> Table:
    """``tab:zeroshot``: the certified OPT models' zero-shot accuracy (%): fp16, and integer minus fp16 with its 95%
    paired bootstrap interval; the mean over the six tasks."""
    tb = Table("tab_zeroshot", "tab:zeroshot", "quality tier, zeroshot.py (the certified recipe; accuracy, acc): "
               "fp16 and integer minus fp16 with the paired bootstrap interval over documents")
    per = {m: zeroshot_cells(ctx.run, m) for m in ZEROSHOT_MODELS}
    for task, label in ZEROSHOT:
        cells = [label]
        for m in ZEROSHOT_MODELS:
            acc, diff, ci, _, r = per[m][task]
            row = f"{m} {task}"
            a = tb.put(row, "fp16", acc, r, lambda v: f.fixed(100 * v, 1))
            d = tb.put(row, "delta", diff, r, lambda v: f.signed(100 * v, 1))
            interval = (f" {{\\scriptsize[{f.signed(100 * ci[0], 1)}, {f.signed(100 * ci[1], 1)}]}}"
                        if ci and diff is not None else "")
            cells += [a, d + interval]
        tb.rows.append(" & ".join(cells) + r" \\")
    tb.rows.append(r"\midrule")
    cells = ["Mean"]
    for m in ZEROSHOT_MODELS:
        accs = [per[m][t][0] for t, _ in ZEROSHOT]
        diffs = [per[m][t][1] for t, _ in ZEROSHOT]
        ok = None not in accs and None not in diffs
        cells += [tb.put(f"{m} mean", "fp16", statistics.mean(accs) if ok else None, "mean of the six tasks",
                         lambda v: f.fixed(100 * v, 1)),
                  tb.put(f"{m} mean", "delta", statistics.mean(diffs) if ok else None, "mean of the six tasks",
                         lambda v: f.signed(100 * v, 2))]
    tb.rows.append(" & ".join(cells) + r" \\")
    return tb


BUILDERS = (tab_eval_llm, tab_eval_v1, tab_options, tab_eval_compare, tab_quality, tab_params, tab_inrange, tab_setup,
            tab_eval_improve, tab_memory, tab_eval_long, tab_eval_long_check, tab_eval_long_gpu, tab_eval_v1_long,
            tab_eval_real, tab_eval_reexec, tab_eval_cpu, tab_eval_plans, tab_eval_lambda, tab_eval_compare_v1, tab_v1,
            tab_gen, tab_gen_full, tab_quality_attr, tab_zeroshot, tab_paths, tab_paths_acc)
"""Every table builder, in the paper's order."""
TABLE_FILES = tuple(b.__name__ for b in BUILDERS)
"""The tables' file names (``<name>.tex``), as ``paper_sp2027/main.tex`` lists them."""


def build_all(ctx: Ctx, problems: list | None = None) -> list[Table]:
    """Every table.  A builder that fails on a malformed record is a ``PARSE`` problem and leaves its table empty
    (every cell missing)."""
    out = []
    for b in BUILDERS:
        try:
            out.append(b(ctx))
        except (KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError, IndexError) as e:
            if problems is None:
                raise
            problems.append(f"PARSE {b.__name__}: {type(e).__name__}: {e}")
            label = "tab:" + b.__name__[4:].replace("_", "-")
            out.append(Table(b.__name__, label, f"the builder failed: {type(e).__name__}: {e}",
                             missing=[f"the builder failed: {e}"]))
    return out

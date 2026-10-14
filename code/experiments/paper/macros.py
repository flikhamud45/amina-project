"""The paper's text numbers: one LaTeX macro per measured number of the running text, from the same records as the
tables.

``numbers.tex`` defines each macro with ``\\newcommand``; the paper writes ``\\Name{}`` where the number stands, and
``paper_sp2027/fallback/numbers.tex`` holds the values the paper prints today.  A macro whose number the run does not
provide is written as a visible ``\\todo`` and listed as missing.  The ``Todo...`` macros carry the ``\\todo`` markers
of numbers that the final run fills: their fallback is the marker, their generated value is empty.

The families, with the record each number comes from:

* the platform (the records' ``env``: the GPU that proves and verifies, the CPU of the CPU verifier) and mode C's
  plan (``--policy``), which the captions and the sentences on how a number was measured state;
* the CPU client (``cpuv`` tier): ``CpuRatioN`` is the abstract's "at least N times": the
  strongest exact re-execution (the fastest of the exact variants, :data:`tables.EXACT`, each computing only what the
  pruned graph computes) over the Kpre CPU verifier on the same 8 threads, for the full Llama-2-7B at 2,048 tokens,
  one value (rounded down to one decimal); ``CpuRatio`` the same ratio rounded, ``CpuRatioShort`` at 64 tokens;
* the outcome counts (``bench`` and ``batch`` tiers, ``comparison/count_outcomes.py`` over ``raw_<platform>/``);
* the cost of the defence, batches against one opening, V1, serving, the comparison with published systems,
  re-execution, real weights, long prompts, the improvements, generation (their tiers' records, through the table
  selections of :mod:`experiments.paper.tables`);
* the break-even (``deployment/ev_breakeven.py``'s formulas on the run's claims, verifier and re-execution times);
* quality and zero-shot accuracy (``quality`` tier), the sampler's distance (``tv`` tier) and the attack (``attack``
  tier's documents);
* values that follow from the code alone, under the run's plan: the security parameters, the setup proof's size,
  breakdown and radii (``costs/setup_bytes.py``), the split commitment's columns per query, and V1's cut of a model
  from its shapes (:func:`experiments.paper.tables.v1_shape_model`).
"""
from __future__ import annotations

import math
import re
import statistics
from dataclasses import dataclass

from experiments.paper import format as f
from experiments.paper import tables as T
from experiments.paper.records import Rec, v1_timed


@dataclass
class Macro:
    """One text number: its printed value (None if the run does not provide it) and its source."""

    name: str
    value: str | None
    raw: object = None
    source: str = ""


class Collector:
    """The macros of one build, in the order they are defined."""

    def __init__(self) -> None:
        self.m: dict[str, Macro] = {}

    def put(self, name: str, raw, src=None, fmt=str) -> None:
        """One number: ``raw`` printed by ``fmt`` (None, NaN or an empty container: not in the run)."""
        if name in self.m:
            raise ValueError(f"macro {name} defined twice")
        source = src.source if isinstance(src, Rec) else str(src or "")
        empty = raw is None or (isinstance(raw, float) and math.isnan(raw)) or (
            isinstance(raw, (dict, list, tuple, str)) and len(raw) == 0)
        self.m[name] = Macro(name, None if empty else fmt(raw), None if empty else raw, source)

    def todo(self, name: str) -> None:
        """A ``\\todo`` marker of numbers that the run fills: empty once generated."""
        self.m[name] = Macro(name, "", None, "todo marker")


def _floor1(v: float) -> str:
    return f"{math.floor(v * 10 + 1e-9) / 10:.1f}"


def ceil_pct(v: float) -> str:
    """A fraction as a whole percentage rounded up, at least 1, for "within N%" and "less than N%" (a float error of
    the ratio never rounds 5% up to 6%)."""
    return f"{max(1, math.ceil(100 * v - 1e-9)):.0f}"


def ceil_fixed(v: float, decimals: int = 1) -> str:
    """``v`` rounded up to ``decimals`` decimals, for a bound ("at most N", "within N"): 0.34 -> ``0.4``."""
    k = 10 ** decimals
    return f"{math.ceil(v * k - 1e-9) / k:.{decimals}f}"


def ceil_sig(v: float, digits: int = 2) -> str:
    """``v`` rounded up to ``digits`` significant digits, trailing zeros dropped, for a bound: 0.8349 -> ``0.84``."""
    if v <= 0:
        return f.sig(v, digits)
    decimals = max(0, digits - 1 - math.floor(math.log10(v)))
    return f.sig(math.ceil(v * 10 ** decimals - 1e-9) / 10 ** decimals, digits + 1)


def _range(c: Collector, name: str, vals, src, fmt) -> None:
    """A ``lo--hi`` macro over ``vals`` (None if any is missing)."""
    ok = None if not vals or any(v is None for v in vals) else (min(vals), max(vals))
    c.put(name, ok, src, lambda p: f.rng(p[0], p[1], fmt))


def build_text(layers, model: str) -> str:
    """A build as the text names it: ``the full model``, or ``b of L blocks``."""
    full = T.FULL_LAYERS[model]
    return "the full model" if layers in (None, full) else f"{layers} of {full} blocks"


def _ratio_text(v: float) -> str:
    """A factor without the times sign: 1{,}213, 78, 2.1."""
    return f.thousands(f"{v:,.0f}") if v >= 100 else (f"{v:.0f}" if v >= 10 else f"{v:.1f}")


# ================================================================================================ the platform
def cpu_name(name: str) -> str:
    """A CPU's model name as the text prints it: ``AMD EPYC 9334 32-Core Processor`` -> ``AMD EPYC 9334``,
    ``Intel(R) Xeon(R) Silver 4114 CPU @ 2.20GHz`` -> ``Intel Xeon Silver 4114``."""
    name = re.sub(r"\((R|TM)\)", "", name)
    name = re.sub(r"\s+CPU\s*@.*$|\s+\d+-Core Processor$|\s+Processor$", "", name.strip())
    return " ".join(name.split())


def platform(c: Collector, ctx: T.Ctx) -> None:
    """The run's GPU (the prover and the GPU verifier) and CPU (the CPU verifier and re-execution), from the records'
    environments (else the benchmark tables' prover and client), and mode C's plan: the settings that the captions and
    the sentences on how a number was measured state."""
    env = ctx.run.environment()
    provers, clients = ctx.bench.hardware()
    gpu = env["gpu"] or next(iter(sorted(provers)), None)
    cpu = env["cpu"] or next(iter(sorted(re.sub(r" x\d+ threads$", "", x) for x in clients)), None)
    c.put("PlatformGpu", gpu, "the records' env.gpu", lambda v: " ".join(v.split()))
    c.put("PlatformCpu", cpu, "the records' env.cpu", cpu_name)
    m = re.fullmatch(r"cnn(\d+)c?", ctx.policy or "")
    c.put("PlanLogPoints", m and int(m.group(1)), f"the plan {ctx.policy}: codewords of at least 2^e points")


# ================================================================================================ the CPU client
def cpu_client(c: Collector, ctx: T.Ctx) -> None:
    run, m = ctx.run, "llama2-7b"
    k = {s: run.perf("cpuv", m, s, "Kpre", vdev="cpu", label="cpuv") for s in (64, 2048)}
    rx = {s: T.strongest(run, "cpuv", m, s) for s in (64, 2048)}
    ratio = {s: rx[s][0] / k[s].d["verify_s"] if k[s] and rx[s][0] else None for s in (64, 2048)}
    c.put("CpuRatioN", ratio[2048], k[2048], _floor1)
    c.put("CpuRatio", ratio[2048], k[2048], lambda v: f"{v:.1f}")
    c.put("CpuRatioShort", ratio[64], k[64], lambda v: f"{v:.1f}")
    c.put("CpuReexecName", rx[2048][1], rx[2048][2], lambda v: T.REEXEC_NAME[v])
    c.put("CpuReexecNameShort", rx[64][1], rx[64][2], lambda v: T.REEXEC_NAME[v])
    tm = (k[2048].d.get("timings") or {}) if k[2048] else {}
    c.put("CpuDeriveShareLong", tm.get("verify_derive") / k[2048].d["verify_s"] if tm.get("verify_derive") else None,
          k[2048], f.pct)
    c.put("CpuReexecLong", rx[2048][0], rx[2048][2], f.sig)
    c.put("CpuReexecShort", rx[64][0], rx[64][2], f.sig)
    for s, name in ((64, "Short"), (2048, "Long")):
        ints = run.reexec("cpuv", m, s, "cpu", ("int", "int_fp32gemm"))
        best = min(ints.values(), key=lambda v: v[0]) if ints else None
        c.put(f"CpuReexecInt{name}", best and best[0], best and best[1], f.sig)
        fl = run.reexec("cpuv", m, s, "cpu", ("float",)).get("float")
        c.put(f"CpuFloat{name}", fl and fl[0], fl and fl[1], f.sig)
        dyn = run.reexec("cpuv", m, s, "cpu", ("dynint8",)).get("dynint8")
        c.put(f"DynInt{name}", dyn and dyn[0], dyn and dyn[1], lambda v: f.sig(v, 2))
        c.put(f"CpuVerify{name}", k[s] and k[s].d["verify_s"], k[s], f.sig)
        g = run.reexec("gpuv", m, s, "cuda", ("int", "float"))
        c.put(f"GpuInt{name}", g.get("int") and g["int"][0], g.get("int") and g["int"][1], f.sig)
        c.put(f"GpuFloat{name}", g.get("float") and g["float"][0], g.get("float") and g["float"][1],
              lambda v: f.sig(v, 2))
    c.todo("TodoCpuRatioIntro")
    opt = run.perf("gpuv", "opt-1.3b", 2048, "Kpre", vdev="cuda", label="gpuv")
    c.put("OptMGpuVerify", opt and opt.d["verify_s"], opt, lambda v: f.sig(v, 2))
    fl = run.reexec("gpuv", "opt-1.3b", 2048, "cuda", ("float",)).get("float")
    c.put("OptMGpuFloat", fl and fl[0], fl and fl[1], lambda v: f.sig(v, 2))
    c.todo("TodoGpuOneGpu")


# ================================================================================================ outcome counts
COUNTS = ("NHonest", "NAttacks", "NFreivalds", "NColumnsCode", "NColumnsMerkle", "NRange")


def counts(c: Collector, ctx: T.Ctx, problems: list) -> None:
    """The completeness and soundness counts of the run's benchmark records (every honest query accepted, every
    attack rejected, and the check that rejected each) over the defining cells, as ``comparison/count_outcomes.py
    --platform <name> --definition --policy <plan>`` counts them."""
    from experiments.comparison import count_outcomes as co
    from experiments.paper.bench import definition_tagsets
    root = ctx.bench.dir.parent / ctx.bench.dir.name.replace("tables_", "raw_", 1) if ctx.bench.dir else None
    m = None
    if root is not None and root.is_dir():
        try:
            m = co.macros(co.count_runs([root], ctx.policy, (), definition_tagsets(ctx.policy)), "Final")
        except SystemExit as e:                     # an honest query rejected or an attack accepted
            problems.append(f"CHECK FAILED outcome counts ({root.name}): {e}")
    for k in COUNTS:
        c.put("Final" + k, m and m.get("Final" + k), root.name if root else "", lambda v: f.num(v))
    c.todo("TodoCounts")


# ================================================================================================ cost of the defence
CNN_MODELS = ("mlp_mnist", "lenet5", "vgg11", "vgg16", "resnet18_cifar", "resnet18_224")
TREE_CHUNK = 1 << 22
"""The chunk of a message that the Fiat--Shamir transcript hashes into one leaf (``pvi.v0.transcript``'s
``absorb_large``; ``ext_proofs.tex``)."""


def defence_cost(c: Collector, ctx: T.Ctx) -> None:
    b = ctx.bench
    costs, ratios = [], []
    for m in CNN_MODELS:
        cell = b.cnn_cell(m, "C", "fs")
        cost = b.cnn_cost(m, cell)
        n = next((b.cnn_metric(m, x, "n_params") for x in b.cnn_cells(m, "facts")), None)
        costs.append(cost)
        ratios.append(n / cost["bytes"] if cost and n and cost["bytes"] else None)
    src = f"{b.name} defence_C_fs_lam128_rate4_wire_pol{ctx.policy}"
    change = []
    for m in CNN_MODELS:
        cell = next(iter(b.cnn_cells(m, "facts")), None)
        fl, it = ((b.cnn_metric(m, cell, k) for k in ("float_accuracy", "int8_accuracy")) if cell else (None, None))
        change.append(abs(it - fl) if fl is not None and it is not None else None)
    c.put("CnnAccuracyChange", None if None in change else max(change), f"{b.name} facts",
          lambda v: ceil_fixed(100 * v, 1))
    _range(c, "CnnProveRange", [x and x["prove"] * 1e3 for x in costs], src, lambda v: f.fixed(v, 1))
    _range(c, "CnnVerifyRange", [x and x["verify"] * 1e3 for x in costs], src, lambda v: f.fixed(v, 1))
    _range(c, "CnnProofRatio", ratios, src, lambda v: f.fixed(v, 1))
    l64, l2k = b.llm_cost("llama2-7b", 64, "c_cpu"), b.llm_cost("llama2-7b", 2048, "c_cpu")
    g64 = b.llm_cost("llama2-7b", 64, "c_gpu")
    src64, src2k = (f"{b.name} {x['cell']}" if x else "" for x in (l64, l2k))
    c.put("ProofRatioLlamaShort", l64 and l64["params"] / l64["bytes"], src64, lambda v: f"{v:.0f}")
    c.put("ProofLlamaShortC", l64 and l64["bytes"], src64, f.b)
    c.put("VerifyLlamaShortCCpu", l64 and l64["verify"], src64, f.t)
    c.put("VerifyLlamaShortCGpu", g64 and g64["verify"], g64 and f"{b.name} {g64['cell']}", f.t)
    c.put("ColumnsLlamaShort", b.llm_metric("llama2-7b", 64, "c_cpu", "bytes_columns"), src64, f.b)
    c.put("ProofLlamaLongC", l2k and l2k["bytes"], src2k, f.b)
    c.todo("TodoProofIntro")
    c.put("ProofLlamaLongCExact", l2k and l2k["bytes"], src2k, lambda v: f"{v / 1e9:.2f}\\,GB")
    claims = b.llm_metric("llama2-7b", 2048, "c_cpu", "bytes_claims")
    c.put("ClaimsShareLlama", claims / l2k["bytes"] if claims and l2k else None, src2k, f.pct)
    # the transcript's tree absorption of that claims message: one 32-byte leaf per chunk of TREE_CHUNK bytes
    leaves = math.ceil(claims / TREE_CHUNK) if claims else None
    c.put("TreeLeavesLlama", leaves, src2k, lambda v: f.about(v, 2))
    c.put("TreeKBLlama", leaves and 32 * leaves, src2k, lambda v: f.about(v / 1e3, 1))
    c.put("TransferLlamaLong", l2k and 8 * l2k["bytes"] / 10e9, src2k, lambda v: f"{v:.1f}")
    k2k = b.llm_cost("llama2-7b", 2048, "kpre_cpu")
    c.put("ProofLlamaLongK", k2k and k2k["bytes"], k2k and f"{b.name} {k2k['cell']}", f.b)
    o67 = b.llm_cost("opt-6.7b", 2048, "c_cpu")
    c.put("ProofOptLLong", o67 and o67["bytes"], o67 and f"{b.name} {o67['cell']}", f.b)
    c.put("VerifyOptLLong", o67 and o67["verify"], o67 and f"{b.name} {o67['cell']}", f.t)
    factors = []
    for m in T.LLM_ROWS[2048]:
        cc, cg = b.llm_cost(m, 2048, "c_cpu"), b.llm_cost(m, 2048, "c_gpu")
        factors.append(cc["verify"] / cg["verify"] if cc and cg and cc["verify"] and cg["verify"] else None)
    _range(c, "GpuVerifyFactor", [v for v in factors if v is not None] or [None], b.name, lambda v: f"{v:.0f}")
    weights = l64 and l64["params"]
    c.put("WeightsLlama", weights, src64, f.b)
    c.put("WeightsLlamaGB", weights, src64, lambda v: f"{v / 1e9:.2f}")


BATCH = 16
"""The batch size the text quotes for B 64-token Llama-2-7B prompts against one opening (the batch tier's B = 4, 16,
64)."""


def batching(c: Collector, ctx: T.Ctx) -> None:
    """One opening and one fold for B prompts (the batch tier, mode C, Llama-2-7B, 64 tokens): the proof per prompt at
    :data:`BATCH`, the single prompt's, the part that every prompt of a batch shares (the single proof without its
    claims) and how closely ``claims + shared / B`` predicts the proof per prompt at every measured ``B``."""
    b, m = ctx.bench, "llama2-7b"
    one = b.llm_cost(m, 64, "c_cpu")
    claims = b.llm_metric(m, 64, "c_cpu", "bytes_claims")
    per = b.llm_batches(m, 64, "c_cpu")
    src = f"{b.name} {m} T64 c_cpu and its batch cells"
    shared = one["bytes"] - claims if one and one["bytes"] and claims else None
    c.put("BatchLlamaSize", BATCH if BATCH in per else None, src)
    c.put("BatchLlamaPerQuery", per.get(BATCH, {}).get("bytes"), src, f.mb)
    c.put("BatchLlamaSingle", one and one["bytes"], src, f.mb)
    c.put("BatchLlamaShared", shared, src, f.mb)
    dev = [abs(v["bytes"] - (claims + shared / bsz)) / (claims + shared / bsz) for bsz, v in per.items()] \
        if shared and per else []
    c.put("BatchLlamaWithin", max(dev) if dev else None, src, ceil_pct)


# ================================================================================================ serving
def serving(c: Collector, ctx: T.Ctx) -> None:
    run = ctx.run
    r = {mode: run.serve("opt-1.3b", 2048, 1, mode, "seq") for mode in ("Kpre", "C")}

    def med(rec, k, sub=None):
        if rec is None:
            return None
        v = rec.d.get(k)
        v = (v or {}).get(sub) if sub else v
        return None if not v else v.get("median", v.get("ratio"))

    for mode, name in (("Kpre", "Kpre"), ("C", "C")):
        c.put(f"ServeOptM{name}", med(r[mode], "prove_s"), r[mode], lambda v: f"{v:.1f}")
        c.put(f"ServeOptM{name}OverUOne", med(r[mode], "prover_over_u1"), r[mode], lambda v: f"{v:.1f}")
        c.put(f"ServeOptM{name}OverUTwo", med(r[mode], "prover_over_u2", "fp16"), r[mode], lambda v: f"{v:.0f}")
    _range(c, "ServeOptMOverUOneRange", [med(r[m], "prover_over_u1") for m in r], r["C"], lambda v: f"{v:.1f}")
    _range(c, "ServeOptMOverUTwoRange", [med(r[m], "prover_over_u2", "fp16") for m in r], r["C"],
           lambda v: f"{v:.0f}")
    bat = run.serve("opt-1.3b", 64, 64, "Kpre", "joint")
    c.put("ServeBatchKpre", med(bat, "prove_s"), bat, lambda v: f"{v:.1f}")
    c.put("ServeBatchOverUOne", med(bat, "prover_over_u1"), bat, lambda v: f"{v:.1f}")
    c.put("ServeBatchOverUTwo", med(bat, "prover_over_u2", "fp16"), bat, lambda v: f"{v:.0f}")
    c.todo("TodoServing")
    c.todo("TodoServingExt")
    one = run.serve("opt-1.3b", 64, 1, "C", "seq")
    batch_c = run.serve("opt-1.3b", 64, 64, "C", "joint")
    c.put("ServeSingleMBPerTokenC", one and one.d.get("proof_mb_per_token"), one, lambda v: f"{v:.2f}")
    c.put("ServeBatchMBPerTokenC", batch_c and batch_c.d.get("proof_mb_per_token"), batch_c, lambda v: f"{v:.2f}")
    cells_a = ((64, 1), (512, 1), (2048, 1))
    for mode, name, fmt in (("Kpre", "Kpre", lambda v: f"{v:.1f}"), ("C", "C", lambda v: f"{v:.0f}")):
        rs = [run.serve("llama2-7b", s, bsz, mode, "seq") for s, bsz in cells_a]
        _range(c, f"ServeLlama{name}OverUTwo", [med(x, "prover_over_u2", "fp16") for x in rs], rs[-1], fmt)
    rs = [run.serve("opt-1.3b", s, bsz, "Kpre", "seq") for s, bsz in cells_a]
    _range(c, "ServeOptMUOneOverUTwo", [med(x, "u1_over_u2", "fp16") for x in rs], rs[-1], lambda v: f"{v:.0f}")


# ================================================================================================ V1
def v1_numbers(c: Collector, ctx: T.Ctx) -> None:
    run = ctx.run
    cell = lambda m, s, mode: T.v1_cell(run, m, s, mode)     # noqa: E731
    ratio = lambda r: r and r.d["v0_total"] / r.d["v1_total"]   # noqa: E731
    long_ = [cell(m, 2048, mode) for _, m in T.V1_ROWS if _ == 2048 for mode in ("C", "Kpre")]
    multi = [r for r in long_ if r and r.d.get("layers", 0) > 1]
    rs = [ratio(r) for r in multi] or [None]
    _range(c, "VOneRatioRandom", rs, "v1 tier, 2,048 tokens, builds of more than one block", lambda v: f"{v:.1f}")
    _range(c, "VOneRatioRandomExact", rs, "v1 tier, 2,048 tokens, builds of more than one block", lambda v: f"{v:.2f}")
    vals = [r.d["v0_bytes"]["claims"] / r.d["v1_bytes"]["claims"] for r in multi
            if r.d.get("v0_bytes", {}).get("claims") and r.d.get("v1_bytes", {}).get("claims")] or [None]
    _range(c, "VOneValuesRatio", vals, "v1 tier, claims against V1's values", lambda v: f"{v:.1f}")
    real = run.v1("real", "opt-6.7b", 2048, "Kpre", vdev="cpu", label="real", real="facebook/opt-6.7b")
    c.put("VOneRatioRealOptL", ratio(real), real, lambda v: f"{v:.1f}")
    c.put("VOneRatioRealOptLExact", ratio(real), real, lambda v: f"{v:.2f}")
    c.put("VOneRealOptLVZeroGB", real and real.d["v0_total"], real, lambda v: f"{v / 1e9:.2f}")
    c.put("VOneRealOptLVOneGB", real and real.d["v1_total"], real, lambda v: f"{v / 1e9:.2f}")
    c.put("VOneRealOptLExceptions", real and real.d.get("n_exceptions"), real, f.num)
    c.put("VOneRealOptLExceptionsMB", real and (real.d.get("v1_bytes") or {}).get("cut_exc"), real,
          lambda v: f"{v / 1e6:.1f}")
    c.put("VOneRealOptLExceptionRate", real and real.d.get("n_leaves") and real.d.get("n_exceptions") is not None
          and real.d["n_exceptions"] / real.d["n_leaves"] or None, real,
          lambda v: f"$10^{{{round(math.log10(v))}}}$")
    c.put("VOneRealOptLTable", real and real.d.get("table"), real, lambda v: f.sci(v).strip("$"))
    tier = [r for r in run.kind("v1", "v1") if not r.d.get("codec_only") and not r.d.get("real")]
    exc = [r.d["n_exceptions"] for r in tier if r.d.get("n_exceptions") is not None]
    c.put("VOneRandomExceptionsMax", max(exc) if exc else None, "v1 tier, random weights", f.num)
    # the multiplicities and GKR transcripts, which do not grow with the prompt, of every measured V1 proof (the
    # exceptions grow with the values and are counted apart)
    extras = [sum((r.d.get("v1_bytes") or {}).get(k, 0) for k in ("cut_mult", "cut_gkr")) for r in run.kind("v1")
              if not r.d.get("codec_only")]
    extras = [x for x in extras if x]
    c.put("VOneExtrasMax", max(extras) if extras else None, "every V1 record: V1's multiplicities and GKR transcripts",
          lambda v: ceil_sig(v / 1e6, 2))
    dev = [abs(sum(r.d["v1_full"]["bytes"].values()) - r.d["v1_total"]) / r.d["v1_total"] for r in tier
           if (r.d.get("v1_full") or {}).get("bytes") and r.d.get("v1_total")]
    c.put("VOneAccountingWithin", max(dev) if dev else None, "v1 tier: the full V1 queries against the accounting",
          lambda v: ceil_fixed(100 * v, 1))
    llama = cell("llama2-7b", 2048, "Kpre")
    c.put("VOneRatioLlama", ratio(llama), llama, lambda v: f"{v:.1f}")
    llama_c = cell("llama2-7b", 2048, "C")
    c.put("VOneLlamaCRatio", ratio(llama_c), llama_c, lambda v: f"{v:.2f}")
    c.put("VOneLlamaCVZeroGB", llama_c and llama_c.d["v0_total"], llama_c, lambda v: f"{v / 1e9:.2f}")
    c.put("VOneLlamaCVOneGB", llama_c and llama_c.d["v1_total"], llama_c, lambda v: f"{v / 1e9:.2f}")
    big = cell("llama2-13b", 2048, "C")
    c.put("VOneRatioLargeC", ratio(big), big, lambda v: f"{v:.2f}")
    short = [cell("gpt2", 64, mode) for mode in ("C", "Kpre")]
    _range(c, "VOneRatioShort", [ratio(r) for r in short], short[0], lambda v: f"{v:.2f}")
    c.put("VOneRatioShortC", ratio(short[0]), short[0], lambda v: f"{v:.2f}")
    c.put("VOneRatioShortK", ratio(short[1]), short[1], lambda v: f"{v:.2f}")
    medium = [cell("gpt2", 512, mode) for mode in ("C", "Kpre")]
    _range(c, "VOneRatioGptMedium", [ratio(r) for r in medium], medium[0], lambda v: f"{v:.2f}")
    for m, name in (("opt-1.3b", "OptM"),):
        for mode, mn in (("C", "C"), ("Kpre", "K")):
            r = cell(m, 2048, mode)
            c.put(f"VOne{name}{mn}VZeroGB", r and r.d["v0_total"], r, lambda v: f"{v / 1e9:.2f}")
            c.put(f"VOne{name}{mn}VOneGB", r and r.d["v1_total"], r, lambda v: f"{v / 1e9:.2f}")
    o125 = cell("opt-125m", 2048, "C")
    p0, _ = v1_timed(o125, "v0")
    p1, _ = v1_timed(o125, "v1", "optimised")
    q = ((o125.d.get("timed") or {}).get("v1_paths") or {}).get("optimised") if o125 else None
    c.put("VOneOptSProveVZero", p0, o125, lambda v: f"{v:.2f}")
    c.put("VOneOptSProveVOne", p1, o125, lambda v: f"{v:.2f}")
    c.put("VOneOptSGkr", (q or {}).get("timings", {}).get("prove_gkr") if q else None, o125, lambda v: f"{v:.2f}")
    for seq, name in ((2048, "Long"), (512, "Medium")):
        rows = [cell(m, s, mode) for s, m in T.V1_ROWS if s == seq for mode in ("C", "Kpre")]
        vr = []
        for r in rows:
            _, v0 = v1_timed(r, "v0")
            _, v1 = v1_timed(r, "v1", "optimised")
            if v0 and v1:
                vr.append(v1 / v0)
        _range(c, f"VOneVerifyRatio{name}", vr or [None], f"v1 tier, timed {seq}-token cells", lambda v: f"{v:.1f}")
    timed = [r for r in long_ if r and r.d.get("timed")]
    share, per_leaf, prove, pratio = [], [], [], []
    for r in timed:
        p0, _ = v1_timed(r, "v0")
        p1, _ = v1_timed(r, "v1", "optimised")
        gkr = (((r.d["timed"].get("v1_paths") or {}).get("optimised") or {}).get("timings") or {}).get("prove_gkr")
        if p1 and gkr:
            share.append(gkr / p1)
            if r.d.get("n_leaves"):
                per_leaf.append(p1 / r.d["n_leaves"] * 1e9)
        if p1:
            prove.append(p1)
        if p0 and p1:
            pratio.append(p1 / p0)
    _range(c, "VOneProverShare", share or [None], "v1 tier, timed 2,048-token cells", f.pct)
    c.put("VOneNsPerClaim", statistics.median(per_leaf) if per_leaf else None, "v1 tier, timed 2,048-token cells",
          lambda v: f"{v:.0f}")
    _range(c, "VOneProverLong", prove or [None], "v1 tier, timed 2,048-token cells", lambda v: f"{v:.1f}")
    _range(c, "VOneProverRatioLong", pratio or [None], "v1 tier, timed 2,048-token cells", lambda v: f"{v:.1f}")
    c.put("VOneProverRatioAbout", statistics.median(pratio) if pratio else None, "v1 tier, timed 2,048-token cells",
          lambda v: f.about(v, 1))
    cv = run.v1("cpuv", "llama2-7b", 2048, "Kpre", vdev="cpu", label="cpuv-v1")
    _, v0 = v1_timed(cv, "v0")
    _, v1 = v1_timed(cv, "v1", "optimised")
    c.put("VOneVerifyLlama", v1, cv, lambda v: f"{v:.1f}")
    c.put("VOneVZeroVerifyLlama", v0, cv, lambda v: f"{v:.1f}")
    c.put("VOneVerifyRatioLlama", v1 / v0 if v0 and v1 else None, cv, lambda v: f"{v:.1f}")
    c.put("VOneVerifyLlamaBaseline", v1_timed(cv, "v1", "baseline")[1], cv, lambda v: f"{v:.1f}")
    c.todo("TodoVOneTimings")
    c.todo("TodoVOneIntro")
    for (m, seq, mode), name in ((("gpt2", 512, "C"), "CrossoverGptMediumC"), (("gpt2", 512, "Kpre"),
                                                                             "CrossoverGptMediumK"),
                                 (("opt-125m", 2048, "C"), "CrossoverOptSLongC"),
                                 (("opt-125m", 2048, "Kpre"), "CrossoverOptSLongK"),
                                 (("llama2-7b", 2048, "Kpre"), "CrossoverLlamaLongK")):
        r = cell(m, seq, mode)
        p0, v0 = v1_timed(r, "v0")
        p1, v1 = v1_timed(r, "v1", "optimised")
        gbps = None
        if None not in (p0, p1, v0, v1) and (p1 + v1) > (p0 + v0):
            gbps = 8 * (r.d["v0_total"] - r.d["v1_total"]) / ((p1 + v1) - (p0 + v0)) / 1e9
        c.put(name, gbps, r, lambda v: f"{v:.2f}")
    r = cell("llama2-7b", 2048, "Kpre")
    c.put("CrossoverLlamaLongKBuild", r and r.d.get("layers"), r, lambda v: build_text(v, "llama2-7b"))
    c.todo("TodoCrossover")
    gk = run.first("gkr", "improve", n=25)
    c.put("GkrNsPerLeaf", gk and gk.d.get("ns_per_leaf"), gk, lambda v: f"{v:.1f}")
    c.put("GkrSecondsLarge", gk and gk.d.get("s"), gk, lambda v: f"{v:.2f}")
    # V1 on shapes alone (no run needed): the down projections' claims that a per-token exponent would leave outside
    # the cut, and the cut of the largest model at 2,048 tokens under the run's plan and instance bound
    from pvi.commit.cost import decoder_claim_columns, decoder_shapes
    from pvi.models.decoder import CONFIGS
    cols = decoder_claim_columns(CONFIGS["llama2-7b"], 2048, prune_last=True)
    down = sum(o.n_rows * 2048 for o in decoder_shapes(CONFIGS["llama2-7b"], prune_last=True)
               if o.name.startswith("down") and cols[o.name] == 2048)
    c.put("VOneDownClaims", down, "decoder_shapes(llama2-7b), 2,048 tokens", lambda v: f.sci(v).strip("$"))
    from pvi.v1.cut import LMAX
    counts_, _ = T.params_v1_counts(run)
    lmax = counts_["lmax"] if counts_ else LMAX
    big = T.v1_shape_model("llama2-13b", 2048, lmax, ctx.policy)
    src = f"v1_shape_model(llama2-13b, 2,048 tokens, lmax {lmax}, plan {ctx.policy})"
    c.put("VOneLargeLeaves", big["n_leaves"], src, lambda v: f.sci(v).strip("$"))
    c.put("VOneLargeInstances", big["n_instances"], src)
    c.put("VOneLargeBits", big["max_bits"], src)
    c.put("VOneLargeGkrLog", big["gkr_log2"], src, lambda v: f"{v:.1f}")
    c.put("VOneLargeLookupLog", big["lookup_log2"], src, lambda v: f"{v:.1f}")
    for m, name in (("opt-125m", "OptS"), ("opt-1.3b", "OptM")):
        r = cell(m, 2048, "C")
        c.put(f"CompareVOne{name}", r and r.d["v0_total"], r, lambda v: f.mb(v))


# ================================================================================================ comparison
COUNT_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten")


def rows_text(n: int) -> str:
    """A count of table rows in words: ``no row``, ``one row``, ``two rows``, ..., ``12 rows``."""
    word = COUNT_WORDS[n] if n < len(COUNT_WORDS) else str(n)
    return f"{word} row" + ("" if n == 1 else "s")


def count_word(v: float) -> str:
    """A count rounded to a whole number, in words from one to ten (``five``), else in digits (``14``)."""
    n = round(v)
    return COUNT_WORDS[n] if 1 <= n < len(COUNT_WORDS) else f.num(n)


def comparison(c: Collector, ctx: T.Ctx) -> None:
    prover, proof, slower = [], [], []
    for row in T.COMPARE:
        if row is None:
            continue
        label, key, model, which = row
        lit = ctx.lit.get(key) or {}
        p, v, nb, src = T.compare_ours(ctx, which)
        tp, tb = f.to_float(lit.get("prover_s")), f.to_float(lit.get("proof_bytes"))
        tv = f.to_float(lit.get("verifier_s"))
        slower.append(None if v is None else bool(tv and f.factor(tv, v)[1] < 0.99))
        if which[0] == "maverick":
            c.put("CompareMaverick", tp / p if tp and p else None, src, lambda v: f"{v:.1f}")
            continue
        prover.append(tp / p if tp and p else None)
        proof.append(nb / tb if tb and nb else None)
    known = [v for v in prover if v is not None]
    c.put("CompareProverMin", min(known) if known and None not in prover else None, ctx.bench.name,
          lambda v: f"{v:.0f}")
    c.put("CompareProverMax", max(known) if known and None not in prover else None, ctx.bench.name, f.about)
    known = [v for v in proof if v is not None]
    c.put("CompareProofMax", max(known) if known and None not in proof else None, ctx.bench.name, f.about)
    c.put("CompareVerifierSlowerRows", None if None in slower else sum(slower), ctx.bench.name, rows_text)
    c.todo("TodoCompareRange")
    z = (ctx.lit.get(("zkLLM", "OPT-1.3B", "2048")) or {}).get("proof_bytes")
    r = T.v1_cell(ctx.run, "opt-1.3b", 2048, "C")
    zb = f.to_float(z)
    c.put("CompareVOneBefore", r and zb and r.d["v0_total"] / zb, r, f.about)
    c.put("CompareVOneAfter", r and zb and r.d["v1_total"] / zb, r, f.about)
    zl = f.to_float((ctx.lit.get(("zkLLM", "Llama-2-13B", "2048")) or {}).get("prover_s"))
    ours = ctx.bench.llm_cost("llama2-13b", 2048, "c_cpu_int") or ctx.bench.llm_cost("llama2-13b", 2048, "c_gpu_int")
    c.put("ZkllmProverFactor", zl and ours and zl / ours["prove"], ours and f"{ctx.bench.name} {ours['cell']}",
          lambda v: f"{v:.0f}")
    c.todo("TodoZkllmFactor")
    b = ctx.bench
    cell = b.cnn_cell("lenet5", "C", "fs")
    le = b.cnn_cost("lenet5", cell)
    src = f"{b.name} {cell}"
    c.put("LenetProve", le and le["prove"] * 1e3, src, lambda v: f"{v:.1f}")
    c.put("LenetVerify", le and le["verify"] * 1e3, src, lambda v: f"{v:.1f}")
    c.put("LenetProof", le and le["bytes"] / 1e3, src, lambda v: f"{v:.1f}")
    for sys_, key in (("Zkcnn", ("zkCNN", "LeNet-5 MNIST", "")), ("Bionetta", ("Bionetta", "LeNet-5", ""))):
        lit = ctx.lit.get(key) or {}
        tp, tv, tb = (f.to_float(lit.get(k)) for k in ("prover_s", "verifier_s", "proof_bytes"))
        c.put(f"LenetProveVs{sys_}", le and tp and tp / le["prove"], src, _ratio_text)
        c.put(f"LenetVerifyVs{sys_}", le and tv and tv / le["verify"], src, _ratio_text)
        q = le and tb and (tb / le["bytes"] if sys_ == "Zkcnn" else le["bytes"] / tb)
        c.put(f"LenetProofVs{sys_}", q, src, _ratio_text)


# ================================================================================================ improvements
def improvements(c: Collector, ctx: T.Ctx) -> None:
    run = ctx.run
    gain = lambda kind, m, layers: (lambda p: p[0] / p[1] if p[0] and p[1] else None)(   # noqa: E731
        T.improve_pair(run, kind, m, layers))
    _range(c, "ImproveOpenRange", [gain("open", m, n) for m, n in (("llama2-7b", 1), ("llama2-7b", 4),
                                                                   ("llama2-13b", 1))], "improve tier, open_ab.py",
           lambda v: f"{v:.1f}")
    fl = run.first("open_ab", "improve", model="llama2-7b", layers=4, variant="float64")
    opening = fl and (fl.d.get("timings") or {}).get("prove_open")
    c.put("ImproveOpenShare", opening and fl.d["prove_s"] and opening / fl.d["prove_s"], fl, f.pct)
    c.put("ImproveOpenFloat", opening, fl, lambda v: f"{v:.3f}")
    c.put("ImproveProveFloat", fl and fl.d["prove_s"], fl, lambda v: f"{v:.2f}")
    _range(c, "ImproveGpuvRange", [gain("gpuv", m, n) for m, n in (("opt-1.3b", 12), ("llama2-7b", 2))],
           "improve tier, perf.py gpuv-before/after", lambda v: f"{v:.1f}")
    c.put("ImproveGpuvOptM", gain("gpuv", "opt-1.3b", 12), "improve tier, perf.py gpuv-before/after",
          lambda v: f"{v:.1f}")
    _range(c, "ImproveEncodeRange", [gain("encode", m, n) for m, n in (("opt-1.3b", 24), ("llama2-7b", 2))],
           "improve tier, encode_ab.py", lambda v: f"{v:.1f}")
    c.put("ImproveCpuv", gain("cpuv", "llama2-7b", 2), "improve tier, perf.py cpuv-before/after",
          lambda v: f"{v:.1f}")
    fused = run.first("encode_ab", "improve", model="opt-1.3b", layers=24, variant="fused")
    threaded = run.first("encode_ab", "improve", model="opt-1.3b", layers=24, variant="threaded")
    c.put("ImproveFusedEncoder", fused and fused.d["encode_s"], fused, lambda v: f"{v:.2f}")
    c.put("ImproveThreadedEncoder", threaded and threaded.d["encode_s"], threaded, lambda v: f"{v:.2f}")
    prof = run.first("gpu_profile", "improve", model="opt-1.3b", layers=12)
    shares = (prof.d.get("shares") or {}) if prof else {}
    c.put("ProfileCopies", (shares.get("aten::copy_") or {}).get("share"), prof, f.pct)
    c.put("ProfileIntMM", (shares.get("aten::_int_mm") or {}).get("share"), prof, lambda v: f.pct(v, 1))
    fold = [r for r in run.kind("fold_ab", "improve") if r.d.get("model") == "llama2-7b"
            and r.d.get("device") == "cpu" and r.d.get("path") != "float64"]
    c.put("KpreKeyLlama", fold[-1].d.get("precompute_s") if fold else None, fold[-1] if fold else None,
          lambda v: f"{v:.1f}")


# ================================================================================================ real weights
def real_weights(c: Collector, ctx: T.Ctx) -> None:
    run = ctx.run
    prove, verify, smaller = [], [], []
    for mode in ("C", "Kpre"):
        for seq in (64, 2048):
            real, rnd = T.real_pair(run, "opt-1.3b", seq, mode, "facebook/opt-1.3b")
            (p_r, v_r), (p_n, v_n) = v1_timed(real, "v0"), v1_timed(rnd, "v0")
            prove.append(abs(p_r / p_n - 1) if p_r and p_n else None)
            verify.append(abs(v_r / v_n - 1) if v_r and v_n else None)
            smaller.append(1 - real.d["v0_total"] / rnd.d["v0_total"] if real and rnd else None)
    src = "real tier, OPT-1.3B (label real against real-random)"
    c.put("RealProveWithin", None if None in prove else max(prove), src, f.pct)
    c.put("RealVerifyWithin", None if None in verify else max(verify), src, f.pct)
    _range(c, "RealProofSmaller", smaller, src, f.pct)
    real, rnd = T.real_pair(run, "opt-1.3b", 2048, "Kpre", "facebook/opt-1.3b")
    c.put("RealOptMKLongReal", real and real.d["v0_total"], real, lambda v: f"{v / 1e9:.2f}")
    c.put("RealOptMKLongRandom", rnd and rnd.d["v0_total"], rnd, lambda v: f"{v / 1e9:.2f}")
    for seq, length in ((64, "Short"), (2048, "Long")):
        for mode, name in (("C", "C"), ("Kpre", "K")):
            real, rnd = T.real_pair(run, "opt-1.3b", seq, mode, "facebook/opt-1.3b")
            c.put(f"VOneRealOptM{length}{name}", real and real.d["v0_total"] / real.d["v1_total"], real,
                  lambda v: f"{v:.2f}")
            c.put(f"VOneRandomOptM{length}{name}", rnd and rnd.d["v0_total"] / rnd.d["v1_total"], rnd,
                  lambda v: f"{v:.2f}")
    real, rnd = T.real_pair(run, "opt-1.3b", 64, "C", "facebook/opt-1.3b")
    c.put("VOneRealOptMShortExceptions", real and real.d.get("n_exceptions"), real, f.num)
    c.put("VOneRandomOptMShortExceptions", rnd and rnd.d.get("n_exceptions"), rnd, f.num)
    c.put("VOneRealOptMShortExceptionsMB", real and (real.d.get("v1_bytes") or {}).get("cut_exc"), real,
          lambda v: f"{v / 1e6:.2f}")
    # the real Llama-2-7B (the real tier's cells with --llama-licence) against its random-weight twins
    ckpt = "meta-llama/Llama-2-7b-hf"
    lr = run.perf("real", "llama2-7b", 2048, "Kpre", vdev="cpu", label="real", real=ckpt)
    ln = run.perf("real", "llama2-7b", 2048, "Kpre", vdev="cpu", label="real-random")
    tam = (lr.d.get("tampers") or {}) if lr else {}
    c.put("RealLlamaAccepted", lr and lr.d["accepted"], lr)
    c.put("RealLlamaQueries", lr and lr.d["queries"], lr)
    c.put("RealLlamaTampersRejected", sum(t.get("rejected", 0) for t in tam.values()) if tam else None, lr)
    c.put("RealLlamaTampers", sum(t.get("attempts", 0) for t in tam.values()) if tam else None, lr)
    c.put("RealLlamaProof", lr and lr.d["proof_bytes"], lr, lambda v: f"{v / 1e9:.2f}")
    c.put("RealLlamaSmaller", lr and ln and 1 - lr.d["proof_bytes"] / ln.d["proof_bytes"], lr, f.pct)
    c.put("RealLlamaVerify", lr and lr.d["verify_s"], lr, f.sig)
    c.put("RndLlamaVerify", ln and ln.d["verify_s"], ln, f.sig)
    short = [(run.perf("real", "llama2-7b", 64, mode, vdev="cpu", label="real", real=ckpt),
              run.perf("real", "llama2-7b", 64, mode, vdev="cpu", label="real-random")) for mode in ("Kpre", "C")]
    _range(c, "RealLlamaShortSmaller", [a and b and 1 - a.d["proof_bytes"] / b.d["proof_bytes"] for a, b in short],
           "real tier, Llama-2-7B, 64 tokens, Kpre and C", f.pct)
    within = [a and b and abs(a.d["verify_s"] / b.d["verify_s"] - 1) for a, b in short]
    c.put("RealLlamaShortVerifyWithin", None if None in within else max(within),
          "real tier, Llama-2-7B, 64 tokens, Kpre and C", ceil_pct)
    v1l = run.v1("real", "llama2-7b", 2048, "Kpre", vdev="cpu", label="real", real=ckpt)
    c.put("VOneRealLlamaRatio", v1l and v1l.d["v0_total"] / v1l.d["v1_total"], v1l, lambda v: f"{v:.2f}")
    r = run.perf("real", "opt-6.7b", 2048, "Kpre", vdev="cpu", label="real", real="facebook/opt-6.7b")
    n = run.perf("real", "opt-6.7b", 2048, "Kpre", vdev="cpu", label="real-random")
    tam = (r.d.get("tampers") or {}) if r else {}
    c.put("RealOptLAccepted", r and r.d["accepted"], r)
    c.put("RealOptLQueries", r and r.d["queries"], r)
    c.put("RealOptLTampersRejected", sum(t.get("rejected", 0) for t in tam.values()) if tam else None, r)
    c.put("RealOptLTampers", sum(t.get("attempts", 0) for t in tam.values()) if tam else None, r)
    c.put("RealOptLProof", r and r.d["proof_bytes"], r, lambda v: f"{v / 1e9:.2f}")
    c.put("RealOptLProve", r and r.d["prove_s"], r, f.sig)
    c.put("RealOptLVerify", r and r.d["verify_s"], r, f.sig)
    tm = (r.d.get("timings") or {}) if r else {}
    for key, name in (("verify_decode", "Decode"), ("verify_derive", "Recompute"), ("verify_products", "Products")):
        c.put(f"RealOptL{name}", tm.get(key), r, lambda v: f"{v:.1f}")
    c.put("RndOptLProve", n and n.d["prove_s"], n, f.sig)
    c.put("RndOptLVerify", n and n.d["verify_s"], n, f.sig)
    c.put("RndOptLProof", n and n.d["proof_bytes"], n, lambda v: f"{v / 1e9:.2f}")
    v1r = run.v1("real", "opt-6.7b", 2048, "Kpre", vdev="cpu", label="real-random", real=None)
    c.put("RndOptLVOneRatio", v1r and v1r.d["v0_total"] / v1r.d["v1_total"], v1r, lambda v: f"{v:.2f}")
    within = [abs(a / b - 1) for a, b in ((r and r.d["prove_s"], n and n.d["prove_s"]),
                                           (r and r.d["verify_s"], n and n.d["verify_s"])) if a and b]
    c.put("RealOptLTimeWithin", max(within) if len(within) == 2 else None, "real tier, OPT-6.7B", f.pct)
    c.put("RealOptLVZeroSmaller", r and n and 1 - r.d["proof_bytes"] / n.d["proof_bytes"], "real tier, OPT-6.7B",
          f.pct)
    v1real = run.v1("real", "opt-6.7b", 2048, "Kpre", vdev="cpu", label="real", real="facebook/opt-6.7b")
    c.put("RealOptLVOneLarger", v1real and v1r and v1real.d["v1_total"] / v1r.d["v1_total"] - 1,
          "real tier, OPT-6.7B", f.pct)


# ================================================================================================ long prompts
def long_prompts(c: Collector, ctx: T.Ctx) -> None:
    pts = {s: T.long_point(ctx.run, s) for s in T.LONG}
    fit = [(s, p["verify_block"]) for s, p in pts.items() if s <= 8192 and p.get("verify_block")]
    resid = None
    if len(fit) >= 3:            # least squares for t = a T + b T^2 per block (no constant)
        s11 = sum(t * t for t, _ in fit)
        s12 = sum(t ** 3 for t, _ in fit)
        s22 = sum(t ** 4 for t, _ in fit)
        y1 = sum(t * v for t, v in fit)
        y2 = sum(t * t * v for t, v in fit)
        det = s11 * s22 - s12 * s12
        if det:
            a, bq = (y1 * s22 - y2 * s12) / det, (s11 * y2 - s12 * y1) / det
            resid = max(abs(a * t + bq * t * t - v) / v for t, v in fit)
    c.put("LongFitResidual", resid, "long tier, 2,048 to 8,192 tokens", lambda v: f"{math.ceil(100 * v):.0f}")
    p8 = pts[8192]
    c.put("LongBytesPerBlock", p8.get("bytes_block"), p8["three"], lambda v: f"{v / 1e9:.2f}")
    p16 = pts[16384]
    rx = p16["reexec"]
    c.put("LongRatioLong", rx and p16.get("verify") and rx[0] / p16["verify"], p16["three"], lambda v: f"{v:.0f}")
    c.put("LongProofLong", p16.get("bytes"), p16["three"], lambda v: f"{v / 1e9:.0f}")


# ================================================================================================ break-even
def breakeven(c: Collector, ctx: T.Ctx) -> dict | None:
    """The crossovers of ``deployment/ev_breakeven.py`` (Llama-2-7B, Fiat--Shamir, the run's plan) on the run's
    records: the claims and the Kpre CPU verifier (cpuv perf.py), V1's claims ratio and verifier (cpuv V1 cell),
    the strongest exact re-execution and the int8 weights.  Returns the inputs and crossovers (for the figure)."""
    run, b, m = ctx.run, ctx.bench, "llama2-7b"
    k = {s: run.perf("cpuv", m, s, "Kpre", vdev="cpu", label="cpuv") for s in (64, 2048)}
    rx = {s: T.strongest(run, "cpuv", m, s)[0] for s in (64, 2048)}
    v1 = run.v1("cpuv", m, 2048, "Kpre", vdev="cpu", label="cpuv-v1")
    weights = (b.llm_cost(m, 64, "c_cpu") or b.llm_cost(m, 64, "kpre_cpu") or {}).get("params")
    names = ("BreakevenQueriesShort", "BreakevenQueriesLong", "BreakevenQueriesVOne", "BreakevenLinkRange",
             "BreakevenLinkLong", "BreakevenLinkVOne", "BreakevenVOneFaster", "BreakevenTokens", "BreakevenTokensVOne")
    v1_claims = v1 and (v1.d.get("v0_bytes") or {}).get("claims") and (v1.d.get("v1_bytes") or {}).get("claims")
    ready = (all(k[s] and (k[s].d.get("bytes") or {}).get("claims") for s in k) and all(rx.values())
             and v1_claims and weights and v1_timed(v1, "v1", "optimised")[1] and ctx.byte_model)
    if not ready:
        for n in names:
            c.put(n, None, "")
        return None
    from experiments.deployment import ev_breakeven as ev
    claims = {s: k[s].d["bytes"]["claims"] for s in k}
    ratio = v1.d["v0_bytes"]["claims"] / v1.d["v1_bytes"]["claims"]
    t_v = {s: k[s].d["verify_s"] for s in k}
    t_v1 = v1_timed(v1, "v1", "optimised")[1]
    fixed = {s: ev.fixed_part(m, ctx.policy, True, s)["fixed"] for s in (64, 2048)}
    proof = {s: claims[s] + fixed[s] for s in claims}
    proof_v1 = claims[2048] / ratio + fixed[2048]
    out = {"weights_bytes": weights, "policies": [ctx.policy],
           "panel1": [{"seq": 64, "protocol": "v0", "policy": ctx.policy, "fs": True, "proof_bytes": proof[64]},
                      {"seq": 2048, "protocol": "v0", "policy": ctx.policy, "fs": True, "proof_bytes": proof[2048]},
                      {"seq": 2048, "protocol": "v1", "policy": ctx.policy, "fs": True, "proof_bytes": proof_v1}],
           "panel2": [{"seq": 2048, "protocol": "v0 Kpre", "fs": True, "proof_bytes": claims[2048],
                       "verify_s": t_v[2048]},
                      {"seq": 2048, "protocol": "V1 Kpre", "fs": True, "proof_bytes": claims[2048] / ratio,
                       "verify_s": t_v1}],
           "reexec_s": rx[2048]}
    src = "deployment/ev_breakeven.py formulas on the cpuv records"
    c.put("BreakevenQueriesShort", ev.crossover_queries(weights, proof[64]), src, lambda v: f"{v:.0f}")
    c.put("BreakevenQueriesLong", ev.crossover_queries(weights, proof[2048]), src, lambda v: f"{v:.1f}")
    c.put("BreakevenQueriesVOne", ev.crossover_queries(weights, proof_v1), src, lambda v: f"{v:.1f}")
    links = [ev.breakeven_link_gbps(claims[s], t_v[s], rx[s]) for s in (64, 2048)]
    _range(c, "BreakevenLinkRange", links, src, lambda v: f"{v:.2f}")
    c.put("BreakevenLinkLong", links[1], src, lambda v: f"{v:.2f}")
    c.put("BreakevenLinkVOne", ev.breakeven_link_gbps(claims[2048] / ratio, t_v1, rx[2048]), src,
          lambda v: f"{v:.2f}")
    c.put("BreakevenVOneFaster", ev.breakeven_link_gbps(claims[2048] - claims[2048] / ratio, t_v[2048], t_v1), src,
          lambda v: f"{v:.1f}")
    c.put("BreakevenTokens", ev.crossover_tokens(weights, (64, proof[64]), (2048, proof[2048])), src, f.about)
    c.put("BreakevenTokensVOne", ev.crossover_tokens(weights, (64, claims[64] / ratio + fixed[64]),
                                                     (2048, proof_v1)), src, f.about)
    return out


# ================================================================================================ quality
OPTS = (("facebook/opt-125m", "OptS", 4), ("facebook/opt-1.3b", "OptM", 3), ("facebook/opt-6.7b", "OptL", 2))
LLAMAS = (("openlm-research/open_llama_7b_v2", "LlamaOpen", "main-a0.55-glus0.2", 3),
          ("TinyLlama/TinyLlama_v1.1", "TinyLlama", "main-a0.75-glus0.2", 2),
          ("Qwen/Qwen3-1.7B-Base", "QwenM", "main-a0.45-glus0.3", 2),
          ("Qwen/Qwen3-4B-Base", "QwenL", "main-a0.45-glus0.5", 1))
"""The Llama-family checkpoints of the text, their ``ppl_llama.py`` main cell (``ppl_llama.MAIN_SETTINGS``) and the
decimals the text prints."""
ALPHAS_TEST = (0.3, 0.4, 0.5, 0.6, 0.7)
ALPHAS_VALIDATION = (0.5, 0.6, 0.7)
"""The smoothing alphas of the quality tier's alpha studies on the test split (``q-opt13-recipes``,
``q-opt125-recipes``) and the validation split (``q-opt-validation``)."""


def quality(c: Collector, ctx: T.Ctx) -> None:
    run = ctx.run
    first = run.ppl("facebook/opt-125m", "fp32")
    c.put("QualityWindows", first and first.d.get("windows"), first, f.num)
    c.put("QualityTokens", first and first.d.get("tokens"), first, f.num)
    p8, p11, cost = [], [], []
    for model, name, gain in OPTS:
        ref = run.ppl(model, "fp32")
        sc = run.ppl(model, T.scalar_cell(gain))
        ce = run.ppl(model, T.CERTIFIED_CELL)
        c.put(f"QualityRef{name}", ref and ref.d["ppl"], ref, lambda v: f"{v:.2f}")
        c.put(f"QualityRatio{name}", ce and ce.d["ratio"], ce, lambda v: f"{v:.3f}")
        c.put(f"QualityScalar{name}", sc and sc.d["ratio"], sc, lambda v: f"{v:.2f}")
        c.put(f"QualityScalarExcess{name}", sc and sc.d["ratio"] - 1, sc, f.pct)
        ci = ce and ce.d.get("ratio_ci95")
        c.put(f"QualityCI{name}", ci, ce, lambda v: f"{v[0]:.3f}, {v[1]:.3f}")
        for cell11 in (T.scalar_cell(gain), T.CERTIFIED_CELL):
            a, b11 = run.ppl(model, cell11.replace("-p11", "-p8")), run.ppl(model, cell11)
            cost.append(a.d["ppl"] / b11.d["ppl"] - 1 if a and b11 else None)
            p8.append(a and a.d.get("mass_min"))
        p11.append(ce and ce.d.get("mass_min"))
    _range(c, "QualityProbCost", cost, "quality tier, the -p8 against the -p11 cells", f.pct)
    c.put("QualityMassMinEight", None if None in p8 else min(p8), "quality tier, -p8 cells", f.pct)
    c.put("QualityMassMinEleven", None if None in p11 else min(p11), "quality tier, certified cells",
          lambda v: f.pct(v, 1))
    ce = run.ppl("facebook/opt-125m", T.CERTIFIED_CELL)
    c.put("QualityMissOptS", ce and ce.d["ratio"] - 1.06, ce, lambda v: f"{v:.3f}")
    legacy = [run.ppl("facebook/opt-1.3b", "smooth-legacy-a0.55-p11", draw=d) for d in (0, 1, 2)]
    lr = [r.d["ratio"] for r in legacy if r]
    c.put("QualityLegacyLo", min(lr) if len(lr) == 3 else None, legacy[0], lambda v: f"{v:.2f}")
    c.put("QualityLegacyHi", max(lr) if len(lr) == 3 else None, legacy[0], lambda v: f"{v:.2f}")
    draws = [run.ppl("facebook/opt-1.3b", T.CERTIFIED_CELL, draw=d) for d in (0, 1)]
    dr = [r.d["ratio"] for r in draws if r]
    c.put("QualityDrawSpread", max(dr) - min(dr) if len(dr) == 2 else None, draws[1], lambda v: f"{v:.3f}")
    old = run.ppl("facebook/opt-125m", "smooth-legacy-a0.55-p11")
    c.put("QualityLegacyOptS", old and old.d["ratio"], old, lambda v: f"{v:.1f}")
    # the certified recipe's alpha against its neighbours: on the test split (where it was chosen) and on the
    # validation split
    for model, name, split, alphas, key in (
            ("facebook/opt-125m", "OptS", "test", ALPHAS_TEST, "QualityAlphaGap"),
            ("facebook/opt-1.3b", "OptM", "test", ALPHAS_TEST, "QualityAlphaGap"),
            ("facebook/opt-125m", "OptS", "validation", ALPHAS_VALIDATION, "QualityAlphaValGap"),
            ("facebook/opt-1.3b", "OptM", "validation", ALPHAS_VALIDATION, "QualityAlphaValGap")):
        rs = {a: run.ppl(model, f"smooth-certified-a{a:g}-p11", split=split) for a in alphas}
        ok = all(rs.values())
        c.put(f"{key}{name}", rs[0.6].d["ratio"] - min(r.d["ratio"] for r in rs.values()) if ok else None, rs[0.6],
              lambda v: ceil_fixed(v, 3))
    rs = {a: run.ppl("facebook/opt-6.7b", f"smooth-certified-a{a:g}-p11", split="validation") for a in (0.5, 0.6)}
    c.put("QualityAlphaCostOptL", rs[0.6].d["ratio"] - rs[0.5].d["ratio"] if all(rs.values()) else None, rs[0.6],
          lambda v: f"{v:.3f}")
    small = [run.ppl("facebook/opt-125m", f"smooth-legacy-a{a}-p8", calibration="test-end", seq=128)
             for a in ("0.55", "0.5")]
    sr = [r.d["ratio"] for r in small if r]
    c.put("LegacySmallLo", min(sr) if len(sr) == 2 else None, small[0], lambda v: f"{v:.2f}")
    c.put("LegacySmallHi", max(sr) if len(sr) == 2 else None, small[0], lambda v: f"{v:.2f}")
    rows = [run.ppl("facebook/opt-125m", f"rows4-legacy-a{a}-p8", calibration="test-end", seq=128)
            for a in ("0.55", "0.5")]
    rr = [r for r in rows if r]
    c.put("RowScalesLo", min(r.d["ratio"] for r in rr) if len(rr) == 2 else None, rows[0], lambda v: f"{v:.2f}")
    c.put("RowScalesHi", max(r.d["ratio"] for r in rr) if len(rr) == 2 else None, rows[0], lambda v: f"{v:.2f}")
    c.put("RowScalesTopOneLo", min(r.d["top1"] for r in rr) if len(rr) == 2 else None, rows[0],
          lambda v: f"{v:.2f}")
    c.put("RowScalesTopOneHi", max(r.d["top1"] for r in rr) if len(rr) == 2 else None, rows[0],
          lambda v: f"{v:.2f}")
    cle = run.ppl("facebook/opt-125m", "smooth-legacy-a0.55-cle0r2-p8", calibration="test-end", seq=128)
    base = run.ppl("facebook/opt-125m", "smooth-legacy-a0.55-p8", calibration="test-end", seq=128)
    c.put("CleWith", cle and cle.d["ppl"], cle, lambda v: f"{v:.1f}")
    c.put("CleWithout", base and base.d["ppl"], base, lambda v: f"{v:.1f}")
    for model, name, cell, dec in LLAMAS:
        r = run.ppl(model, cell)
        c.put(f"{name}Ratio", r and r.d["ratio"], r, lambda v, dec=dec: f"{v:.{dec}f}")
    for model, name, cell in (("Qwen/Qwen3-1.7B-Base", "QwenMExpRatio", "main-a0.45-glus0.4-exp8"),
                              ("Qwen/Qwen3-4B-Base", "QwenLExpRatio", "main-a0.45-glus0.5-exp8")):
        r = run.ppl(model, cell)
        c.put(name, r and r.d["ratio"], r, lambda v, name=name: f"{v:.2f}" if name == "QwenMExpRatio" else f"{v:.1f}")
    picks = []
    for model, name, _ in OPTS:
        cells = T.zeroshot_cells(run, model)
        diffs = [cells[t][1] for t, _ in T.ZEROSHOT]
        mean = statistics.mean(diffs) if None not in diffs else None
        src = cells["lambada_openai"][4]
        c.put(f"ZeroshotMean{name}", mean, src, (lambda v: f.signed(100 * v, 2)) if name == "OptL"
              else (lambda v: f"{abs(100 * v):.1f}"))
        lam = cells["lambada_openai"][1]
        c.put(f"ZeroshotLambada{name}", lam, src, lambda v: f"{abs(100 * v):.1f}")
        picks += [cells[t][3] for t, _ in T.ZEROSHOT]
    _range(c, "ZeroshotSamePick", picks, "quality tier, zeroshot.py same_pick", f.pct)
    legacy = T.zeroshot_cells(run, "facebook/opt-1.3b", "legacy")
    ld = [legacy[t][1] for t, _ in T.ZEROSHOT]
    c.put("ZeroshotLegacyMean", statistics.mean(ld) if None not in ld else None, legacy["lambada_openai"][4],
          lambda v: f"{abs(100 * v):.1f}")


# ================================================================================================ the sampler
def sampler(c: Collector, ctx: T.Ctx) -> None:
    doc = ctx.run.file("sampler_tv_opt-1.3b.json")
    summary = (doc.d.get("summary") or []) if doc else []

    def pick(key, tau, top_k, table):
        for s in summary:
            if (s.get("tau"), s.get("top_k"), str(s.get("table"))) == (tau, top_k, table):
                return (s.get(key) or {}).get("median")
        return None

    c.put("TvPositions", doc and len(doc.d.get("positions") or []) or None, doc, f.num)
    c.put("TvTopK", pick("tv_sampler", 1.0, 50, "16"), doc, lambda v: f.sig(v, 2))
    c.put("TvFull", pick("tv_sampler", 1.0, 0, "16"), doc, lambda v: f"{v:.3f}")
    c.put("TvModel", pick("tv_model", 1.0, 0, "16"), doc, lambda v: f.sig(v, 2))
    c.put("TvTwenty", pick("tv_sampler", 1.0, 0, "20"), doc, lambda v: f.sig(v, 2))


# ================================================================================================ the attack
def _pct2(v):
    """A detection probability as a percentage with two significant digits: 48, 6.8, 0.23."""
    return f.sig(100 * v, 2)


def _paths(p, lam=40):
    return None if not p else math.ceil(lam * math.log(2) / -math.log1p(-p))


def attack(c: Collector, ctx: T.Ctx) -> None:
    run, b = ctx.run, ctx.bench
    opt = run.file("real_llm_attack.json")
    auto = run.file("real_llm_attack_auto.json")
    u = (opt.d.get("untargeted") or {}) if opt else {}
    c.put("AttackOptLSucceeded", u.get("succeeded"), opt)
    c.put("AttackOptLPrompts", u.get("attempted"), opt)
    w = u.get("success_wilson95")
    c.put("AttackOptLWilson", w, opt, lambda v: f"{f.sig(100 * v[0], 3)}--{f.sig(100 * v[1], 3)}")
    for doc, name in ((auto, "Back"), (opt, "Hacked")):
        bd = (doc.d.get("backdoor") or {}) if doc else {}
        c.put(f"AttackOptLBackdoor{name}", bd and (bd.get("succeeded"), bd.get("attempted")), doc,
              lambda v: f"{v[0]}/{v[1]}")
    c.put("AttackOptLPaths", opt and (opt.d.get("path_test") or {}).get("paths_for_2^-40"), opt, f.num)
    reach = ((opt.d.get("reachability") or {}) if opt else {})
    last = reach.get(str(max(map(int, reach)))) if reach else None
    c.put("ReachNeurons", last and last.get("neurons"), opt, f.num)
    c.put("ReachTokens", last and last.get("distinct_forceable_tokens"), opt, f.num)
    c.put("ReachVocab", last and last.get("vocab"), opt, f.num)
    an = run.file("analytic.csv")
    row = next((r for r in (an.d["rows"] if an else []) if r.get("model") == "llama2-7b"), None)
    paths = f.to_float(row and row.get("anchuri_paths"))
    c.put("PathsLlama", paths, an, f.num)
    c.put("PathsLlamaAbout", paths, an, lambda v: f.about(v, 3))
    c.put("WidthLlama", f.to_float(row and row.get("anchuri_width")), an, f.num)
    c.put("OpenAllLlama", f.to_float(row and row.get("anchuri_open_all_bytes")), an, lambda v: f"{v / 1e9:.1f}")
    # MNIST (attack_mnist.py, samplers.py, floor_sampler.py, adaptive_game.py, reproduction.py)
    am = run.file("attack.json")
    base = (am.d.get("baselines") or {}) if am else {}
    single = [v.get("acceptance_mean") for k, v in base.items() if k.startswith("single-neuron")]
    _range(c, "MnistSinglePass", single or [None], am, lambda v: f"{100 * v:.1f}")
    bd = (am.d.get("backdoor") or {}) if am else {}
    c.put("MnistBackdoorPass", bd.get("triggered_acceptance_empirical"), am, lambda v: f"{100 * v:.2f}")
    c.put("MnistBackdoorRuns", bd.get("triggered_runs"), am, f.num)
    c.put("MnistBackdoorPredicted", bd.get("triggered_acceptance_exact"), am, lambda v: f"{100 * v:.1f}")
    de = run.file("defence.json")
    sweep = (de.d.get("sweep") or {}) if de else {}
    det = lambda k: (sweep.get(k) or {}).get("detection_evasive")     # noqa: E731
    c.put("SamplerUniform", det("uniform"), de, lambda v: f"{100 * v:.2f}")
    c.put("SamplerContribution", det("local-contribution"), de, lambda v: f"{100 * v:.1f}")
    c.put("SamplerStatic", det("static-importance"), de, lambda v: f"{100 * v:.2f}")
    c.put("SamplerSaliency", det("gradient-saliency"), de, _pct2)
    z = (de.d.get("zero_blind_spot") or {}) if de else {}
    c.put("ZeroingNeurons", z.get("mean_support"), de, lambda v: f"{v:.0f}")
    c.put("ZeroingZeroShare", z.get("natural_zero_fraction_at_chosen_neurons"), de, f.pct)
    st = ((am.d.get("stealth") or {}).get("p95.0") or {}) if am else {}
    c.put("MnistStealthNeurons", st.get("mean_support"), am, count_word)
    c.put("MnistStealthPass", st.get("acceptance_mean"), am, f.pct)
    c.put("ZeroingAccepts", z.get("protocol_accepts"), de, f.num)
    c.put("ZeroingRuns", z.get("protocol_runs"), de, f.num)
    ones = [v.get("support_mean") for k, v in base.items() if k.startswith("single-neuron")]
    c.put("MnistForgeryNodes", ones and z.get("mean_support") and (min(ones), z["mean_support"]), de,
          lambda v: f.rng(v[0], v[1], lambda x: f"{x:.0f}"))
    fl = run.file("floor_sampler.json")
    samp = (fl.d.get("samplers") or {}) if fl else {}
    best = lambda k: (samp.get(k) or {}).get("adversary_best")     # noqa: E731
    c.put("FloorBest", best("eps=1"), fl, f.pct)
    others = [best(k) for k in samp if k.startswith("eps=") and k not in ("eps=0", "eps=1")]
    uni = best("uniform")
    c.put("FloorVsUniform", uni and best("eps=1") and best("eps=1") / uni, fl, lambda v: f"{v:.0f}")
    _range(c, "FloorOthers", [o / uni for o in others] if uni and others and None not in others else [None], fl,
           lambda v: f.sig(v, 1))
    ag = run.file("adaptive_game_mlp_mnist.json")
    layers = (ag.d.get("layers") or {}) if ag else {}
    for li, name in (("1", "One"), ("2", "Two")):
        caps = (layers.get(li) or {}).get("caps") or {}
        mix = [(v.get("mixture") or {}).get("detection_per_path") for k, v in caps.items() if k != "inf"]
        c.put(f"MixtureLayer{name}", max(mix) if mix and None not in mix else None, ag, lambda v: f"{100 * v:.1f}")
        g1 = (((caps.get("inf") or {}).get("G1e") or {}).get("greedy") or {}).get("top_q_hit_rate") or {}
        c.put(f"Locator{name}", g1.get("1"), ag, f.pct)
    cfg = (ag.d.get("config") or {}) if ag else {}
    c.put("LocatorQueries", cfg.get("queries"), ag)
    c.put("StealthCalib", cfg.get("calib_size"), ag, f.num)
    c95 = ((layers.get("1") or {}).get("caps") or {}).get("95") or {}
    c.put("StealthNeurons", c95.get("mean_support"), ag, lambda v: f"{v:.1f}")
    c.put("StealthPass", c95.get("mean_support") and 1 - c95["mean_support"] / c95.get("width", 512), ag,
          lambda v: f"{100 * v:.1f}")
    c.put("StealthExceed", 1 - c95["R2_max_pass_rate"] if c95.get("R2_max_pass_rate") is not None else None, ag,
          f.pct)
    rp = run.file("reproduction.json")
    setups = (rp.d.get("setups") or {}) if rp else {}
    runs = {(s.get("correctness") or {}).get("runs") for s in setups.values()}
    c.put("ReproRuns", runs.pop() if len(runs) == 1 else None, rp, f.num)
    caught = [(s.get("other_model") or {}).get("detection_rate") for s in setups.values()]
    _range(c, "ReproCaught", caught or [None], rp, lambda v: f.sig(100 * v, 3))
    zero = [(s.get("correctness") or {}).get("zero_tolerance_acceptance_rate") for s in setups.values()]
    _range(c, "ReproZeroTolerance", zero or [None], rp, f.pct)
    c.put("ReproQueries", ((rp.d.get("config") or {}).get("queries")) if rp else None, rp)
    # the image models (bench.py's attack_float and sampling cells)
    pd, need, minnode = {}, [], []
    for m in CNN_MODELS:
        cell = next(iter(b.cnn_cells(m, "sampling")), None)
        if cell is None:
            pd[m] = None
            continue
        pd[m] = b.cnn_metric(m, cell, "p_detect_penultimate")
        need.append(b.cnn_metric(m, cell, "paths_needed_penultimate"))
        minnode.append(b.cnn_metric(m, cell, "p_detect_min_node"))
    ok = [v for v in pd.values() if v]
    c.put("CnnPathLenet", pd.get("lenet5"), b.name, lambda v: f"1/{round(1 / v)}")
    c.put("CnnPathMax", min(ok) if ok else None, b.name, lambda v: f"1/{round(1 / v)}")
    _range(c, "CnnPathsRange", need if need and None not in need else [None], b.name, f.num)
    cell = next(iter(b.cnn_cells("resnet18_224", "sampling")), None)
    c.put("CnnPathsResNet", cell and b.cnn_metric("resnet18_224", cell, "paths_needed_penultimate"), b.name, f.num)
    c.put("CnnMinNode", min(minnode) if minnode and None not in minnode else None, b.name,
          lambda v: f"{1 / v / 1e6:.0f}")
    c.put("PathPassResNet", pd.get("resnet18_224"), b.name, lambda v: f"${round(1 / v) - 1}/{round(1 / v)}$")
    inrange(c, ctx)


def all_of(v) -> str:
    """``(changed, prompts)`` over the cells of a document as the text counts the prompts: ``all 40`` when every cell
    changed the next token of every prompt, else ``at least 38 of the 40`` (the fewest of any cell)."""
    n, total = v
    return f"all {total}" if n == total else f"at least {n} of the {total}"


def inrange(c: Collector, ctx: T.Ctx) -> None:
    run = ctx.run
    any_, down = run.file("inrange_llm_opt-1.3b_any.json"), run.file("inrange_llm_opt-1.3b_down.json")
    layers = (6, 12, 18, 23)
    args = (any_.d.get("args") or {}) if any_ else {}
    c.put("InrangeCalibTokens", args and args.get("calib_windows", 0) * args.get("calib_seq", 0) or None, any_, f.num)
    cells = [T.inrange_cell(any_, layer, rng) for layer in layers for rng in ("max", "p999")]
    ok = all(cells)
    c.put("InrangeChanged", (min(x["changed"] for x in cells), max(x["prompts"] for x in cells)) if ok else None,
          any_, all_of)
    c.put("InrangePrompts", cells[0]["prompts"] if ok else None, any_)
    ks = [k for x in cells if x for k in x["k_all"]]
    c.put("InrangeKMedian", statistics.median(ks) if ok and ks else None, any_, lambda v: f"{v:g}")
    c.put("InrangeKMax", max(ks) if ok and ks else None, any_)
    c.put("InrangePMedianMax", max(x["p_uniform"] for x in cells) if ok else None, any_, f.sci)
    ps = [p for x in cells if x for p in x["p_uniform_all"]]
    c.put("InrangePMax", max(ps) if ok and ps else None, any_, f.sci)
    dc = [T.inrange_cell(down, layer, "max") for layer in layers]
    c.put("InrangeDown", [x["changed"] for x in dc] if all(dc) else None, down,
          lambda v: ", ".join(map(str, v[:-1])) + f" and {v[-1]}")
    c.put("InrangeDownLast", dc[-1]["changed"] if dc[-1] else None, down)
    dk = [k for x in dc if x for k in x["k_all"]]
    c.put("InrangeDownKMedian", statistics.median(dk) if all(dc) and dk else None, down, lambda v: f"{v:g}")
    allc = cells + dc
    paths = [_paths(x["p_uniform"]) for x in allc] if all(allc) else [None]
    _range(c, "InrangePathsOptM", paths, any_, lambda v: f.about(v, 3))
    a67, d67 = run.file("inrange_llm_opt-6.7b_any.json"), run.file("inrange_llm_opt-6.7b_down.json")
    l67 = (8, 16, 24, 31)
    c67 = [T.inrange_cell(a67, layer, rng) for layer in l67 for rng in ("max", "p999")]
    ok67 = all(c67)
    c.put("InrangeOptLChanged", (min(x["changed"] for x in c67), max(x["prompts"] for x in c67)) if ok67 else None,
          a67, all_of)
    c.put("InrangeOptLPMedianMax", max(x["p_uniform"] for x in c67) if ok67 else None, a67, f.sci)
    _range(c, "InrangeOptLKRange", [x["k_median"] for x in c67] if ok67 else [None], a67, lambda v: f"{v:g}")
    d67c = [T.inrange_cell(d67, layer, "max") for layer in l67]
    paths = [_paths(x["p_uniform"]) for x in c67 + d67c] if ok67 and all(d67c) else [None]
    _range(c, "InrangePathsOptL", paths, a67, lambda v: f.about(v, 3))
    game = run.file("adaptive_game_opt-1.3b.json")
    per = (game.d.get("per_layer") or {}) if game else {}
    caps = ("90", "95", "99", "100")
    mix = lambda layer, cap: (((per.get(str(layer)) or {}).get("caps") or {}).get(cap, {}).get("mixture")   # noqa
                              or {}).get("detection_per_path")
    uni = lambda layer, cap: ((((per.get(str(layer)) or {}).get("caps") or {}).get(cap, {}).get("samplers")  # noqa
                               or {}).get("uniform") or {}).get("detection_per_path")
    for layer, name in ((6, "Six"), (12, "Twelve"), (18, "Eighteen"), (23, "TwentyThree")):
        c.put(f"GameLayer{name}", mix(layer, "100"), game, _pct2)
        vals = [mix(layer, cap) for cap in caps]
        c.put(f"GameCaps{name}", None if None in vals else vals, game,
              lambda v: ", ".join(_pct2(x) + r"\%" for x in v[:-1]) + f" and {_pct2(v[-1])}")
    d_ff = next(((v or {}).get("d_ff") for v in per.values()), None)
    c.put("GameThreshold", d_ff and 27.7 / d_ff, game, _pct2)
    u100 = [uni(layer, "100") for layer in layers]
    _range(c, "GameUniform", u100 if per else [None], game, _pct2)
    uall = [uni(layer, cap) for layer in layers for cap in caps]
    _range(c, "GameUniformAll", uall if per else [None], game, _pct2)


# ================================================================================================ the rest
def setup_and_params(c: Collector, ctx: T.Ctx) -> None:
    run = ctx.run
    for model, name in (("llama2-7b", "Llama"), ("gpt2", "Gpt")):
        r = run.setup(model, 100)
        c.put(f"SetupProver{name}", r and r.d.get("prove_s"), r, f.t)
        c.put(f"SetupVerifier{name}", r and r.d.get("verify_s"), r, f.t)
    counts_, rec = T.params_v1_counts(run)
    c.put("ParamsLeaves", counts_ and counts_["n_leaves"], rec, lambda v: f.sci(v).strip("$"))
    c.put("ParamsInstances", counts_ and counts_["n_instances"], rec)
    c.put("ParamsLmax", counts_ and counts_["lmax"], rec)
    c.put("ParamsTable", counts_ and counts_["table"], rec, lambda v: f.sci(v).strip("$"))
    from pvi.v1.cut import LMAX
    n = counts_["lmax"] if counts_ else LMAX
    c.put("GkrTranscriptBits", n, rec or "pvi.v1.cut.LMAX")
    c.put("GkrTranscriptKB", 31 * (4 + sum(3 * k + 4 for k in range(1, n))), rec or "pvi.v1.cut.LMAX",
          lambda v: f"{v / 1e3:.1f}")
    c.todo("TodoGenFull")


RADIUS_WORDS = {0: "the whole", -1: "half", -2: "a quarter", -3: "an eighth"}


def split_columns(t_bits: float, n: int) -> int:
    """The least ``t`` with ``prod_{i<t} (n/2 - i) / (n - i) <= 2^-t_bits``: the columns per query that catch the
    split commitment of ``eq:split`` (half of a tree's positions bound to another model) at the query's target."""
    s, t = 0.0, 0
    while s > -t_bits:
        s += math.log2((n / 2 - t) / (n - t))
        t += 1
    return t


def split_factor(model: str, policy: str) -> float:
    """The column bytes per query that testing at every query against the split commitment would take, over the
    honest ones (interactive queries, lambda 128): ``sum_j t'_j rows_j / sum_j t_j rows_j`` over the plan's trees,
    ``t'_j`` from :func:`split_columns` at the column target ``beta``."""
    from pvi.commit.cost import decoder_shapes
    from pvi.commit.plans import column_bits, exact_columns, plan_commitment
    from pvi.models.decoder import CONFIGS
    shapes = decoder_shapes(CONFIGS[model], prune_last=True)
    plan = plan_commitment(shapes, policy, rate=4)
    coded = {m.name: m for m in plan.coded}
    bits = column_bits(128, len(shapes), False)
    rows = {g: sum(coded[m].n_rows for m in members) for g, members in plan.groups}
    honest = {g: max(exact_columns(coded[m].row_length, coded[m].n_points, bits) for m in members)
              for g, members in plan.groups}
    split = {g: split_columns(bits, coded[members[0]].n_points) for g, members in plan.groups}
    return sum(split[g] * rows[g] for g in rows) / sum(honest[g] * rows[g] for g in rows)


def setup_proof(c: Collector, ctx: T.Ctx) -> None:
    """The untrusted commitment (Sec. ``sec:setup-proof``, ``ext_defence.tex``, ``appendix_defence.tex``): the split
    commitment's columns per query and the column bytes that testing at every query would take, the setup proof's size,
    its breakdown and its radii, and the quantities of the remark on other proximity tests (``rem:forced``), all from
    the library under the run's plan (``costs/setup_bytes.py``; ``A``: ``a = 100`` amortised queries, ``B``:
    ``1,000``); the per-query growth over the run's 2,048-token Llama-2-7B proof."""
    on, pol = ctx.byte_model, ctx.policy
    src = f"the library under the plan {pol}"
    v = T.params_values(pol, None) if on else None
    c.put("ParamsColumnsMinInt", v and v["int"]["t_min"], src)
    c.put("ParamsColumnsMinFs", v and v["fs"]["t_min"], src)
    c.put("ParamsTrees", v and v["trees"], src)
    models = [m for m, _ in T.SETUP_MODELS]
    _range(c, "SplitColumnFactorRange", [split_factor(m, pol) for m in models] if on else [None], src,
           lambda x: f"{x:.1f}")
    got = {m: {a: T.setup_of(m, pol, a) for a in (100, 1000)} for m in models} if on else {}
    size = {m: {a: s[0] for a, s in per.items()} for m, per in got.items()}
    ll = size.get("llama2-7b", {})
    for a, w in ((100, "A"), (1000, "B")):
        s = ll.get(a)
        mb = (lambda key, s=s: s and s[key])          # noqa: E731
        c.put(f"SetupGBLlama{w}", mb("setup_MB"), src, lambda x: f"{x / 1e3:.1f}")
        c.put(f"SetupMBLlama{w}", mb("setup_MB"), src, lambda x: f.num(x, 1))
        c.put(f"SetupUntrustedMBLlama{w}", mb("query_columns_MB_untrusted"), src, lambda x: f"{x:.1f}")
        c.put(f"SetupColumnsMBLlama{w}", mb("columns_MB"), src, lambda x: f.num(x, 1))
        c.put(f"SetupColumnsShareLlama{w}", s and s["columns_MB"] / s["setup_MB"], src, lambda x: f.pct(x, 1))
        c.put(f"SetupCombinedShareLlama{w}", s and s["combined_rows_MB"] / s["setup_MB"], src, lambda x: f.pct(x, 1))
        c.put(f"SetupPathsMBLlama{w}", mb("paths_MB_bound"), src, lambda x: ceil_fixed(x, 1))
        c.put(f"SetupPathsShareLlama{w}", s and s["paths_MB_bound"] / s["setup_MB"], src, lambda x: f.pct(x, 2))
        c.put(f"SetupNonColumnsShareLlama{w}", s and 1 - s["columns_MB"] / s["setup_MB"], src, lambda x: f.pct(x, 1))
        others = [1 - size[m][a]["columns_MB"] / size[m][a]["setup_MB"] for m in size]
        c.put(f"SetupNonColumnsShareMax{w}", max(others) if others else None, src, lambda x: ceil_fixed(100 * x, 1))
        c.put(f"SetupAtLamMBLlama{w}", mb("columns_MB_at_lam"), src, lambda x: f.num(x, 1))
        c.put(f"SetupAtLamShareLlama{w}", s and s["columns_MB_at_lam"] / s["setup_MB"], src, lambda x: f.pct(x, 1))
        excess = [s["t0_with_replacement"][g] / s["t0"][g] - 1 for g in s["t0"]] if s else [None]
        _range(c, f"SetupReplacementExcess{w}", excess, src, lambda x: f.pct(x, 1))
        growth = s and s["query_factor"] - 1
        c.put(f"SetupQueryGrowthLlama{w}", growth, src, f.pct)
        proof = ctx.bench.llm_cost("llama2-7b", 2048, "c_cpu")
        extra = s and proof and proof["bytes"] and (s["query_columns_MB_untrusted"] - s["query_columns_MB_honest"]) \
            * 1e6 / proof["bytes"]
        c.put(f"SetupQueryShareLlama{w}", extra, proof and f"{src}; {ctx.bench.name} {proof['cell']}",
              ceil_pct)
    c.put("SetupHonestMBLlama", ll.get(100) and ll[100]["query_columns_MB_honest"], src, lambda x: f"{x:.1f}")
    c.put("SetupCombinedMBLlama", ll.get(100) and ll[100]["combined_rows_MB"], src, lambda x: f"{x:.1f}")
    shares = [size[m][a]["columns_MB_at_lam"] / size[m][a]["setup_MB"] for m in size for a in size[m]]
    _range(c, "SetupAtLamShareRange", shares or [None], src, lambda x: f.pct(x, 1))
    if "llama2-7b" in got:
        _, p, plan = got["llama2-7b"][1000]
        n = {g: next(m.n_points for m in plan.coded if m.name == members[0]) for g, members in plan.groups}
        c.put("SetupTZeroMaxShareB", max(t / n[g] for g, t in dict(p.t0).items()), src, lambda x: f.pct(x, 1))
    else:
        c.put("SetupTZeroMaxShareB", None, src)
    radii = [x for per in got.values() for _, p, plan in per.values() for x in T.radius_log2(p, plan).values()
             if x is not None]
    c.put("SetupRadiusMax", max(radii) if radii else None, src,
          lambda x: RADIUS_WORDS.get(x, f"$2^{{{x}}}$"))


GEN_P, GEN_G = 512, 512
"""The prompt and response of the text's per-prefix comparison (Sec. ``sec:gen-prefill``)."""


def generation(c: Collector, ctx: T.Ctx) -> None:
    """A generated response as one prefill (the gen tier): GPT-2's prover per generated token (the comparison with
    zkAgent), the 256-token greedy responses of Llama-2-7B and Qwen3-4B after a 512-token prompt, the proof per token
    and the cost of sampling against greedy decoding; and the proofs of per-prefix queries against one prefill for
    Llama-2-7B with ``P = G = 512`` (mode C: the cpuv tier's claims per position at 64 and 2,048 tokens, linear in
    the positions, plus the fixed part of the byte model under the run's plan)."""
    run = ctx.run
    g = run.gen("gpt2", "C", "greedy", 256, prompt=64)
    c.put("GenGptProveC", g and g.d["prove_s"], g, lambda v: f"{v:.2f}")
    c.put("GenGptMsPerToken", g and g.d["prove_s"] / g.d["gen"] * 1e3, g, lambda v: f"{v:.1f}")
    for m, name in (("llama2-7b", "Llama"), ("qwen3-4b", "Qwen")):
        for mode, mn in (("C", "C"), ("Kpre", "K")):
            r = run.gen(m, mode, "greedy", 256, prompt=512)
            c.put(f"Gen{name}Prove{mn}", r and r.d["prove_s"], r, f.sig)
            c.put(f"Gen{name}Verify{mn}", r and r.d["verify_s"], r, f.sig)
            c.put(f"Gen{name}Proof{mn}", r and r.d["proof_bytes"], r, f.b)
    for gg, name in ((64, "Short"), (256, "Long")):
        r = run.gen("llama2-7b", "C", "greedy", gg, prompt=512)
        c.put(f"GenLlamaMBPerToken{name}", r and r.d["proof_mb_per_token"], r,
              lambda v: f"{v:.2f}" if v < 10 else f.sig(v, 3))
    within = []
    for m in ("llama2-7b", "qwen3-4b"):
        for mode in ("C", "Kpre"):
            for gg in (64, 256):
                a, b = (run.gen(m, mode, d, gg, prompt=512) for d in ("greedy", "sample"))
                within += [abs(b.d[k] / a.d[k] - 1) for k in ("prove_s", "verify_s")] if a and b else [None]
    c.put("GenSampledWithin", None if None in within else max(within), "gen tier, sampled against greedy",
          ceil_pct)
    kc = {s: run.perf("cpuv", "llama2-7b", s, "C", vdev="cpu", label="cpuv") for s in (64, 2048)}
    claims = {s: (r.d.get("bytes") or {}).get("claims") if r else None for s, r in kc.items()}
    single = prefixes = None
    if ctx.byte_model and all(claims.values()):
        from experiments.deployment import ev_breakeven as ev
        per_pos = (claims[2048] - claims[64]) / (2048 - 64)
        fixed = ev.fixed_part("llama2-7b", ctx.policy, True, GEN_P)["fixed"]

        def proof(t):
            return claims[64] + per_pos * (t - 64) + fixed

        single = proof(GEN_P + GEN_G - 1)
        prefixes = sum(proof(GEN_P + i - 1) for i in range(1, GEN_G + 1))
    src = "cpuv tier's mode-C claims of Llama-2-7B at 64 and 2,048 tokens and the byte model's fixed part"
    c.put("GenPrefillGB", single, src, lambda v: f"{v / 1e9:.1f}")
    c.put("GenPerPrefixTB", prefixes, src, lambda v: f"{v / 1e12:.1f}")


FAMILIES = (platform, cpu_client, defence_cost, batching, serving, v1_numbers, comparison, improvements, real_weights,
            long_prompts, quality, sampler, attack, setup_and_params, setup_proof, generation)


def build(ctx: T.Ctx, problems: list) -> tuple[dict[str, Macro], dict | None]:
    """Every macro of the run (``(macros, the break-even's inputs or None)``).  A family that fails on a malformed
    record is a ``PARSE`` problem; its macros are then missing, so that the macro set never changes."""
    c, be = Collector(), None
    for fam in FAMILIES + (counts, breakeven):
        try:
            got = fam(c, ctx, problems) if fam is counts else fam(c, ctx)
            be = got if fam is breakeven else be
        except (KeyError, TypeError, ValueError, ZeroDivisionError, AttributeError, IndexError) as e:
            problems.append(f"PARSE macros {fam.__name__}: {type(e).__name__}: {e}")
    for name in names():
        c.m.setdefault(name, Macro(name, None, None, ""))
    return c.m, be


_NAMES: list[str] = []


def names() -> list[str]:
    """Every macro name (the macros of a run without records)."""
    if not _NAMES:
        from experiments.paper.bench import Bench
        from experiments.paper.records import Run
        c = Collector()
        ctx = T.Ctx(Run([]), Bench(None), {}, byte_model=False)
        for fam in FAMILIES:
            fam(c, ctx)
        counts(c, ctx, [])
        breakeven(c, ctx)
        _NAMES.extend(sorted(c.m))
    return list(_NAMES)


def text(macros: dict[str, Macro], header: str) -> str:
    """``numbers.tex``: one ``\\newcommand`` per macro; a missing number prints as a visible ``\\todo``."""
    lines = [f"% generated by experiments/paper/numbers.py: the text numbers ({header})"]
    for name in sorted(macros):
        mac = macros[name]
        value = mac.value if mac.value is not None else rf"\todo{{{name}: not in the run}}"
        lines.append(rf"\newcommand{{\{name}}}{{{value}}}" + (f"  % {mac.source}" if mac.source else ""))
    return "\n".join(lines) + "\n"

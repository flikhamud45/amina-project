"""The cost of re-running the model: the baselines against which a verifier's time is read.

Produces the re-execution rows of ``tab:options`` (exact re-execution on the CPU, the integer forward
pass on the GPU) and of ``tab:eval-reexec`` (extended version: the GPU's integer forward pass and fp16 decoder, the
CPU's fp32, bf16 and integer re-execution of Llama-2-7B), the abstract's and Sec. ``sec:eval-reexec``'s ratio of the
CPU verifier to the fastest exact re-execution, and the re-execution side of ``tab:eval-long`` (the package's tiers
``cpuv``, ``gpuv`` and ``long``).

    python -m experiments.reexec.reexec --model llama2-7b --seq 64 2048 --layers 2 32 \\
        --variants int int_fp32gemm int_mm float bf16 dynint8 --device cpu --threads 8 --reps 5
    python -m experiments.reexec.reexec --model llama2-7b opt-6.7b --seq 64 512 2048 --variants int float \\
        --device cuda
    python -m experiments.reexec.reexec --model llama2-7b --seq 8192 --layers 2 3 --max-pos 16384 --device cpu
    python -m experiments.reexec.reexec --model tiny-llama --seq 32 --layers 1 2 --device cpu --threads 2 \\
        --variants int int_fp32gemm int_mm float bf16 dynint8          # a laptop check (seconds)

A client that holds the weights (mode K), or a Kpre client that could keep them instead of its key, could check
an answer by computing it again.  The script times that for the decoders of ``pvi.models.decoder.CONFIGS`` (random
weights, ``--seed``: the cost depends on the shapes only), one prompt of ``T`` tokens (``--seq``; seed
``--prompt-seed``) per variant:

* ``int``: our integer model's forward pass (``IntGraph.forward`` of ``build_decoder``, with the attention
  probabilities' scale ``--prob-max``, default 2047: 11 bits, as the cost measurements' graphs), which the prover runs
  and which is bit-exact with the claims, so it is a complete check in itself (the int8 GEMMs of :mod:`pvi.options`'
  ``native_gemm`` on a CPU, ``torch._int_mm`` on a GPU);
* ``int_fp32gemm`` (CPU): the same integer forward with the option ``native_gemm`` off (its weight products as exact
  fp32 GEMMs over 1,024-term chunks);
* ``int_mm`` (CPU): the same integer forward with the int8 x int8 -> int32 GEMMs of its linear layers (those the
  native GEMM computes) on ``torch._int_mm`` (oneDNN, with VNNI's ``vpdpbusd`` where the CPU has it).  It is exact
  only where its record says so (``exact``): on CPUs without VNNI oneDNN adds pairs of u8 x s8 products in int16
  (``vpmaddubsw``), which saturates.  Every run therefore compares the GEMM with int64 products at the int8 bounds,
  and at each block count the logits of its warm-up with those of an untimed pass of the same graph's own forward
  (``int``'s), bit for bit.  Where torch has no ``_int_mm`` on the CPU (e.g. torch 2.2), no oneDNN, or refuses a
  shape, the variant is reported unavailable on stderr (no record) and the other variants run;
* ``float``: a plain floating-point decoder of the same shapes (fp16 on a GPU, fp32 on the CPU): ``F.linear`` and
  ``scaled_dot_product_attention`` with a causal mask, the norms and the MLP of the architecture, the LM head at the
  last position.  It is what a client would run with an off-the-shelf stack; it adds no position embedding (rotary
  or learned: an element-wise cost) and is not bit-exact with the provider's model;
* ``bf16`` (CPU): the floating-point decoder in bf16 (fast on CPUs with AVX512_BF16, e.g. Zen 4);
* ``dynint8`` (CPU): the floating-point decoder with every linear layer as torch's dynamic int8 Linear (fbgemm,
  activations quantised per call): a standard int8 engine, not exact.

**Pruning.**  A next-token query needs the last block at the last position only, so with ``--prune-last`` (the
default) every variant computes what ``build_decoder(..., prune_last=True)`` computes: the last block's keys and
values at every position, its query, attention, output projection, residuals and MLP at the last position only, and
the LM head at the last position (``--no-prune-last``: the last block at every position, the LM head still at the
last one).  The integer and the floating-point variants thus do the same work per block.

Each variant is timed on ``--layers`` blocks (``--reps`` runs after one warm-up, median); with ``t(L) = a + L b`` the
script extrapolates to the full model's ``n_layers``.  A ``--layers`` list that includes ``n_layers`` measures the
full model, whose ``measured_s`` the paper's tables use (no extrapolation).  ``--max-pos N`` raises a rotary-position
model's context to ``N`` tokens (the shapes and weights do not depend on it: the long-prompt cells); a model with
learned positions is refused, since its position table would change.

**Allocator.**  glibc's defaults unless ``--malloc-tune``, which applies the CPU verifier's allocator setting
(:func:`pvi.kernels.native.tune_malloc`: freed memory stays in the process) to this whole process on a CPU, so that
re-execution can be timed under either setting.

**Record** (one JSON line per model, ``T`` and variant, on stdout): ``git``, ``model``, ``seq``, ``variant``,
``dtype``, ``device``, ``threads``, ``reps``, ``prune_last``, ``prob_max`` (the integer variants' scale; None for
the floating-point ones), ``measured_s`` (median seconds per measured block
count), ``range_s`` (their ``[min, max]``), ``times_s`` (every timed repetition, per block count), ``per_block_s``, ``fixed_s``, ``n_layers``, ``full_s`` (the linear fit at
the full depth), ``max_pos``, ``peak`` (host peak resident set and GPU peak memory over the variant's builds and
runs), ``malloc_tuned`` (whether this process applied ``--malloc-tune``'s allocator setting; glibc's ``MALLOC_*``
environment variables, which set the same parameters, are in ``env``'s ``env_vars``), ``gpu``, ``cpu``, ``host``,
``torch``, ``env`` (:func:`experiments.common.records.environment`); ``int_mm`` also ``exact`` (true only when both
checks pass) and ``exact_check`` (``bounds``: the GEMM at the int8 bounds; ``logits``: by block count, whether the
logits equal the integer forward pass's).

Hardware and time: the CPU rows run on 8 threads of the verifier's CPU, the GPU rows on the prover's GPU.  One run of
the full Llama-2-7B at 2,048 tokens took 93 s (fp32) and 142 s (integer) on 8 threads of a Xeon Silver 4114; the
package gives the CPU variants of one model (64 and 2,048 tokens, 2 blocks and the full model) a job limit of one to
three hours (OPT-1.3B 1 h, Qwen3-4B 2 h, Llama-2-7B 3 h) with 256 GB of host memory.  A 2-block build takes seconds.
"""
from __future__ import annotations

import argparse
import dataclasses
import platform
import statistics
import sys
import warnings

import torch
import torch.nn.functional as F

from experiments.common import cli, records
from experiments.common.models import decoder_config
from experiments.common.paths import CODE
from pvi import options
from pvi.core.field import INT8_TERMS
from pvi.core.graph import INT8_MAX
from pvi.kernels import native as native_kernels


def _timed(fn, device: torch.device, reps: int, warm_up: bool = True) -> list[float]:
    """The seconds of ``reps`` calls of ``fn`` after one untimed call (``warm_up=False``: the caller made it)."""
    if warm_up:
        fn()
    return [records.timed(fn, device)[1] for _ in range(reps)]


class IntMMUnavailable(RuntimeError):
    """``torch._int_mm`` cannot run the ``int_mm`` variant here (not on this device or torch build, without oneDNN,
    or not on a shape)."""


def int_mm_product(w: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """``w @ x.T`` (int64 ``[N, M]``, the native GEMM's layout) for int8 ``w [N, K]`` and int8 ``x [M, K]`` on the
    CPU's ``torch._int_mm``, in blocks of ``INT8_TERMS`` terms (each int32 sum below ``2**30``); ``x.T`` is passed as
    a transposed view, not copied.  Raises :class:`IntMMUnavailable` when torch refuses."""
    k = w.shape[1]
    out = None
    try:
        for s in range(0, k, INT8_TERMS):
            part = torch._int_mm(w[:, s:s + INT8_TERMS], x[:, s:s + INT8_TERMS].T).to(torch.int64)
            out = part if out is None else out.add_(part)
    except (RuntimeError, AttributeError) as e:          # NotImplementedError (no CPU kernel) is a RuntimeError
        first = (str(e).splitlines() or [""])[0]           # (torch's dispatcher error lists every backend after it)
        raise IntMMUnavailable(f"torch._int_mm on [{w.shape[0]}, {k}] x [{k}, {x.shape[0]}]: {first}") from e
    return out


def _fell_back(caught) -> None:
    """Raise :class:`IntMMUnavailable` if torch warned that ``_int_mm``'s oneDNN path failed (``caught``: the
    warnings of the calls), after which it loops over the entries one by one."""
    for w in caught:
        if "int_mm" in str(w.message):
            raise IntMMUnavailable(f"oneDNN failed: {w.message}")


def int_mm_bounds_exact(device: torch.device) -> bool:
    """Whether ``torch._int_mm`` on the CPU gives the int64 products of random int8 operands with rows and columns
    of 127 and -128 (with one operand offset to u8, a pair of such products overflows int16, so a GEMM that adds
    pairs in int16 saturates there).  Raises :class:`IntMMUnavailable` off the CPU, without oneDNN, or where it does
    not run or falls back."""
    if device.type != "cpu":
        raise IntMMUnavailable("a CPU variant (the int variant's GEMMs on a GPU are torch._int_mm already)")
    if not (torch.backends.mkldnn.is_available() and torch.backends.mkldnn.enabled):
        raise IntMMUnavailable("torch runs without oneDNN here")
    g = torch.Generator().manual_seed(0)
    w = torch.randint(-128, 128, (40, 4096), generator=g).to(torch.int8)
    x = torch.randint(-128, 128, (24, 4096), generator=g).to(torch.int8)
    w[0], w[1], x[0], x[1] = 127, -128, 127, -128
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = int_mm_product(w, x)
    _fell_back(caught)
    return torch.equal(got, (w.double() @ x.double().T).to(torch.int64))


class IntMMOp:
    """A weight op of the ``int_mm`` variant: :meth:`pvi.core.graph.MatOp.compute`'s product on ``torch._int_mm``
    (:func:`int_mm_product`) instead of the native GEMM, plus the op's bias; ``IntGraph.forward(weights_override=...)``
    runs it in the op's place (the op still folds the product)."""

    def __init__(self, op):
        self.op = op
        self.bias = None if op.bias is None else op.bias.to(torch.int64)

    def compute(self, x: torch.Tensor) -> torch.Tensor:
        z = int_mm_product(self.op.weight, x.reshape(-1, self.op.n_in).to(torch.int8))
        return z if self.bias is None else z + self.bias[:, None]


def int_mm_ops(graph) -> dict:
    """``{op name: IntMMOp}`` for the weight ops whose products run on the native int8 GEMM on a CPU (linear layers
    with int8 weights and int8 inputs; ``MatOp.compute``)."""
    return {op.name: IntMMOp(op) for op in graph.mat_ops
            if op.layout == "linear" and op.max_input <= INT8_MAX and op.weight.dtype == torch.int8}


def _same_logits(graph, tokens: torch.Tensor, run) -> bool:
    """Whether ``run()`` (the ``int_mm`` forward pass, untimed: also its warm-up) gives the logits of ``graph``'s own
    forward pass bit for bit.  Raises :class:`IntMMUnavailable` if a GEMM fell back (:func:`_fell_back`)."""
    with torch.no_grad():
        want = graph.forward(tokens, free=True)[0][graph.output_name]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        got = run()
    _fell_back(caught)
    return torch.equal(got, want)


class FloatDecoder:
    """A floating-point decoder with the shapes of ``cfg`` and ``n_layers`` blocks (random weights).

    Tokens ``[B, T]`` are one batched prefill (``B`` prompts of ``T`` tokens, no padding); the LM head is applied at
    the last position.  ``prune_last`` (as ``build_decoder(..., prune_last=True)``): the last block computes k and v
    at every position and q, the attention, its projection, the residuals and the MLP at the last position only,
    whose one query attends to every key (no causal mask); only that position reaches the logits.  The last block
    then holds its q and k/v projections as two weights (``"q"``, ``"kv"``: row blocks of the fused q/k/v weight
    drawn as for an unpruned block), so a subclass that replaces ``_lin`` (:class:`DynInt8Decoder`) prunes too."""

    def __init__(self, cfg, n_layers: int, dtype: torch.dtype, device: torch.device, seed: int = 0,
                 prune_last: bool = False):
        g = torch.Generator().manual_seed(seed)
        d, f, dh, hq, hkv = cfg.d_model, cfg.d_ff, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        self.cfg, self.dtype, self.device, self.prune_last = cfg, dtype, device, prune_last

        def w(n, k):
            return (torch.randn(n, k, generator=g) / k ** 0.5).to(dtype).to(device)

        def b(n):
            return torch.zeros(n, dtype=dtype, device=device) if cfg.bias else None

        self.blocks = []
        for i in range(n_layers):
            blk = {"qkv": (w((hq + 2 * hkv) * dh, d), b((hq + 2 * hkv) * dh)), "o": (w(d, hq * dh), b(d))}
            if prune_last and i == n_layers - 1:      # q at the last position, k and v at every one
                wq, bq = blk.pop("qkv")
                blk["q"] = (wq[:hq * dh], None if bq is None else bq[:hq * dh])
                blk["kv"] = (wq[hq * dh:], None if bq is None else bq[hq * dh:])
            if cfg.mlp == "swiglu":
                blk["up"], blk["down"] = (w(2 * f, d), None), (w(d, f), None)
            else:
                blk["up"], blk["down"] = (w(f, d), b(f)), (w(d, f), b(d))
            self.blocks.append(blk)
        self.embed = (torch.randn(cfg.vocab, d, generator=g) * 0.02).to(dtype).to(device)
        self.head = self.embed if cfg.tied else (torch.randn(cfg.vocab, d, generator=g) / d ** 0.5).to(dtype).to(device)

    def _lin(self, x, wb):
        return F.linear(x, *wb)

    def _norm(self, x):
        if self.cfg.norm == "rmsnorm":
            return x * torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + 1e-6).to(x.dtype)
        return F.layer_norm(x, x.shape[-1:])

    def __call__(self, tokens: torch.Tensor) -> torch.Tensor:
        cfg = self.cfg
        dh, hq, hkv = cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        x = self.embed[tokens]                                    # [B, T, d]
        bsz, t, _ = x.shape
        for blk in self.blocks:
            h = self._norm(x)
            if "q" in blk:                                        # the pruned last block
                q = self._lin(h[:, -1:], blk["q"])
                k, v = self._lin(h, blk["kv"]).split([hkv * dh, hkv * dh], -1)
                x, tq, causal = x[:, -1:], 1, False
            else:
                q, k, v = self._lin(h, blk["qkv"]).split([hq * dh, hkv * dh, hkv * dh], -1)
                tq, causal = t, True
            q = q.view(bsz, tq, hq, dh).transpose(1, 2)
            k = k.view(bsz, t, hkv, dh).transpose(1, 2)
            v = v.view(bsz, t, hkv, dh).transpose(1, 2)
            if hkv != hq:
                k, v = k.repeat_interleave(hq // hkv, 1), v.repeat_interleave(hq // hkv, 1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=causal)
            x = x + self._lin(a.transpose(1, 2).reshape(bsz, tq, hq * dh), blk["o"])
            h = self._lin(self._norm(x), blk["up"])
            if cfg.mlp == "swiglu":
                gate, up = h.chunk(2, -1)
                h = F.silu(gate) * up
            else:
                h = F.relu(h) if cfg.mlp == "relu" else F.gelu(h)
            x = x + self._lin(h, blk["down"])
        return F.linear(self._norm(x[:, -1]), self.head)          # next-token logits


class DynInt8Decoder(FloatDecoder):
    """:class:`FloatDecoder` (fp32) with every block's linear layer as torch's dynamic int8 Linear (fbgemm, per-call
    activation scales): a standard int8 CPU engine, not exact."""

    def __init__(self, cfg, n_layers: int, device: torch.device, seed: int = 0, prune_last: bool = False):
        super().__init__(cfg, n_layers, torch.float32, device, seed, prune_last=prune_last)
        self._q = {}
        for blk in self.blocks:
            for w, b in blk.values():
                lin = torch.nn.Linear(w.shape[1], w.shape[0], bias=b is not None)
                with torch.no_grad():
                    lin.weight.copy_(w)
                    if b is not None:
                        lin.bias.copy_(b)
                q = torch.ao.quantization.quantize_dynamic(torch.nn.Sequential(lin), {torch.nn.Linear},
                                                           dtype=torch.qint8)
                self._q[id(w)] = q

    def _lin(self, x, wb):
        return self._q[id(wb[0])](x)


VARIANTS = ("int", "int_fp32gemm", "int_mm", "float", "bf16", "dynint8")
INTEGER = ("int", "int_fp32gemm", "int_mm")
"""The variants that run the integer graph of ``build_decoder``."""
DTYPE = {"int": "int8 GEMM", "int_fp32gemm": "int (fp32 GEMM)", "int_mm": "int8 GEMM (torch._int_mm)",
         "bf16": "bfloat16", "dynint8": "dynamic int8 (not exact)"}
"""The record's ``dtype`` of each variant (``float``: fp16 on a GPU, fp32 on the CPU; ``int_mm``: with
``(not exact)`` where a check failed)."""


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", nargs="+", required=True,
                    help="decoders' shapes (pvi.models.decoder.CONFIGS, experiments.common.models.TINY)")
    ap.add_argument("--seq", type=int, nargs="+", default=[64, 512, 2048], help="prompt tokens (each one a cell)")
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2],
                    help="block counts measured (the full model's count: measured directly)")
    ap.add_argument("--variants", nargs="+", default=["int", "float"], choices=VARIANTS)
    ap.add_argument("--reps", type=int, default=5, help="timed runs per block count, after one warm-up")
    ap.add_argument("--max-pos", type=int, default=0, help="rotary-position models: a longer context")
    ap.add_argument("--prune-last", action=argparse.BooleanOptionalAction, default=cli.PAPER.prune_last,
                    help="every variant computes the last block at the last position only (as the verifier's graph)")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed")
    ap.add_argument("--prompt-seed", type=int, default=1, help="the random prompt's seed")
    ap.add_argument("--calib-tokens", type=int, default=cli.CALIB_TOKENS,
                    help="calibration tokens of the integer decoder (at most --seq)")
    cli.add_prob_max_argument(ap, note="; the integer variants")
    cli.add_runtime_arguments(ap, threads=8, verifier_device=False)
    ap.set_defaults(malloc_tune=False)               # glibc's defaults unless --malloc-tune (module docstring)
    return ap


def main() -> int:
    from pvi.models.decoder import build_decoder

    args = build_parser().parse_args()
    cfgs = {}
    for name in args.model:                          # refused before anything is built
        cfg = decoder_config(name)
        if args.max_pos > cfg.max_pos:
            if cfg.pos != "rope":
                raise SystemExit(f"--max-pos: {name} has learned positions (a longer context is another model)")
            cfg = dataclasses.replace(cfg, max_pos=args.max_pos)
        cfgs[name] = cfg
    cli.apply_runtime(args)
    device = torch.device(args.device)
    malloc_tuned = device.type == "cpu" and native_kernels.tune_malloc()      # --malloc-tune, for the whole process
    fdtype = torch.float16 if device.type == "cuda" else torch.float32
    sha = records.git_sha(CODE)
    env = records.environment(device)
    variants, bounds = list(args.variants), None
    if "int_mm" in variants:
        try:
            bounds = int_mm_bounds_exact(device)
        except IntMMUnavailable as e:
            print(f"reexec: int_mm unavailable: {e}", file=sys.stderr)
            variants.remove("int_mm")
    for name, cfg in cfgs.items():
        for seq in args.seq:
            if seq > cfg.max_pos:                     # e.g. GPT-2's 1,024 learned positions
                continue
            tokens = torch.randint(0, cfg.vocab, (1, seq), generator=torch.Generator().manual_seed(args.prompt_seed))
            tokens = tokens.to(device)
            for variant in variants:
                measured, spread, same, raw = {}, {}, {}, {}
                records.reset_peaks(device)
                try:
                    for n_layers in args.layers:
                        warm_up = True
                        if variant in INTEGER:
                            graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(seq, args.calib_tokens),
                                                  seed=args.seed, prune_last=args.prune_last,
                                                  prob_max=args.prob_max)

                            # int_fp32gemm: the same forward pass with the native int8 GEMM switched off; int_mm:
                            # also off, its products on torch._int_mm
                            gemm = options.current().native_gemm and variant == "int"
                            override = int_mm_ops(graph) if variant == "int_mm" else None

                            def run(graph=graph, gemm=gemm, override=override):      # -> the logits
                                with options.using(native_gemm=gemm), torch.no_grad():
                                    out, _ = graph.forward(tokens, free=True, weights_override=override)
                                return out[graph.output_name]

                            if override is not None:
                                same[n_layers], warm_up = _same_logits(graph, tokens, run), False
                        else:
                            model = (DynInt8Decoder(cfg, n_layers, device, args.seed, prune_last=args.prune_last)
                                     if variant == "dynint8" else
                                     FloatDecoder(cfg, n_layers, torch.bfloat16 if variant == "bf16" else fdtype,
                                                  device, args.seed, prune_last=args.prune_last))

                            def run(model=model):
                                with torch.no_grad():
                                    model(tokens)
                        times = _timed(run, device, args.reps, warm_up=warm_up)
                        measured[n_layers] = statistics.median(times)
                        spread[n_layers] = [min(times), max(times)]
                        raw[n_layers] = list(times)
                        del run
                        if device.type == "cuda":
                            torch.cuda.empty_cache()
                except IntMMUnavailable as e:
                    print(f"reexec: int_mm unavailable for {name} at {seq} tokens: {e}", file=sys.stderr)
                    continue
                ls = sorted(measured)
                per_block = ((measured[ls[-1]] - measured[ls[0]]) / (ls[-1] - ls[0]) if len(ls) > 1
                             else measured[ls[0]] / ls[0])
                fixed = measured[ls[0]] - ls[0] * per_block
                rec = {
                    "git": sha, "model": name, "seq": seq, "variant": variant,
                    "dtype": DTYPE.get(variant, str(fdtype).replace("torch.", "")),
                    "device": str(device), "threads": args.threads, "reps": args.reps, "prune_last": args.prune_last,
                    "prob_max": args.prob_max if variant in INTEGER else None,
                    "measured_s": {str(k): v for k, v in measured.items()},
                    "range_s": {str(k): v for k, v in spread.items()},
                    "times_s": {str(k): v for k, v in raw.items()}, "peak": records.peaks(device),
                    "max_pos": cfg.max_pos,
                    "per_block_s": per_block, "fixed_s": fixed, "n_layers": cfg.n_layers,
                    "full_s": fixed + cfg.n_layers * per_block,
                    "malloc_tuned": malloc_tuned,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "cpu": platform.processor() or platform.machine(), "host": platform.node(),
                    "torch": torch.__version__, "env": env}
                if variant == "int_mm":
                    rec["exact"] = bool(bounds) and all(same.values())
                    rec["exact_check"] = {"bounds": bounds, "logits": {str(k): v for k, v in same.items()}}
                    if not rec["exact"]:
                        rec["dtype"] += " (not exact)"
                records.emit(rec)
    return 0


if __name__ == "__main__":
    sys.exit(main())

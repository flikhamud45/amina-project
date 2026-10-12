"""The cost of re-running the model: the baseline against which a verifier's time is read.

    cd code && PYTHONPATH=src python experiments/7_reexec/reexec.py --model llama2-7b --seq 64 512 2048 --device cuda
    cd code && PYTHONPATH=src python experiments/7_reexec/reexec.py --model opt-1.3b --seq 64 2048 --device cpu --threads 8

A client that holds the weights (settings K and Kpre) could check an answer by computing it again.
This script times that, for the decoders of ``pvi.fullcheck.transformer.CONFIGS`` (random weights:
the cost depends on the shapes only), in two variants:

* ``int``: our integer model's forward pass (``IntGraph.forward`` of ``build_decoder(...,
  prune_last=True)``), which the prover runs and which is bit-exact with the claims, so it is a
  complete check in itself;
* ``int_fp32gemm`` (CPU): the same integer forward with its weight products as exact fp32 GEMMs over 1,024-term
  chunks (the path before the native int8 GEMM), for comparison;
* ``bf16`` (CPU): the floating-point decoder below in bf16 (fast on CPUs with AVX512_BF16, e.g. Zen 4);
* ``dynint8`` (CPU): the floating-point decoder with every linear layer as torch's dynamic int8 Linear (fbgemm):
  a standard int8 engine, NOT exact (activations quantised per call);
* ``float``: a plain floating-point decoder of the same shapes (fp16 on a GPU, fp32 on the CPU):
  ``torch.nn.functional.linear`` and ``scaled_dot_product_attention`` with a causal mask, the norms
  and the MLP of the architecture, the LM head at the last position.  It is what a client would run
  with an off-the-shelf stack; it skips the rotary embedding (an element-wise cost) and is not
  bit-exact with the provider's model.

Each variant is timed on ``--layers`` blocks (default 1 and 2); with ``t(L) = a + L b`` the script
extrapolates to the full model's ``n_layers`` and prints one JSON line per (model, T, variant), with
the measured medians and ``full_s``.  Large models fit a small GPU this way; a ``--layers`` list that
includes ``n_layers`` measures the full model, whose ``full_s`` is then that measurement (F6: no
extrapolation).  Each line also has the ``[min, max]`` of every measured block count (``range_s``) and the
variant's peak host (VmHWM, Linux) and GPU memory over its builds and runs (``peak``, ``bench.py``'s names).
``--max-pos N`` raises a rotary-position model's context to ``N`` tokens (the shapes and weights do not depend
on it; V-LONG-b's long prompts); a model with learned positions is refused, since its position table would change.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import platform
import statistics
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

CODE = Path.cwd()   # run from code/


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn, device: torch.device, reps: int) -> list[float]:
    """The seconds of ``reps`` calls of ``fn`` after one untimed call."""
    fn()                                         # warm-up
    times = []
    for _ in range(reps):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return times


def _reset_peaks(device: torch.device) -> None:
    """A new peak window: the GPU allocator's statistics and the host's VmHWM (Linux)."""
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    try:
        with open("/proc/self/clear_refs", "w") as fh:
            fh.write("5")
    except OSError:
        pass


def _peaks(device: torch.device) -> dict:
    """Bytes since :func:`_reset_peaks`, named as ``bench.py`` records them (``host_peak_rss`` is ``None`` where
    ``/proc`` has none)."""
    out = {"host_peak_rss": None}
    try:
        with open("/proc/self/status") as fh:
            out["host_peak_rss"] = 1024 * next(int(line.split()[1]) for line in fh if line.startswith("VmHWM:"))
    except (OSError, StopIteration):
        pass
    if device.type == "cuda":
        out["gpu_peak_memory"] = torch.cuda.max_memory_allocated(device)
        out["gpu_peak_reserved"] = torch.cuda.max_memory_reserved(device)
    return out


class FloatDecoder:
    """A floating-point decoder with the shapes of ``cfg`` and ``n_layers`` blocks (random weights).

    Tokens ``[B, T]`` are one batched prefill (``B`` prompts of ``T`` tokens, no padding).  ``prune_last``
    (opt-in, as ``build_decoder(..., prune_last=True)``): the last block computes k and v at every position
    and q, the attention, its projection, the residuals and the MLP at the last position only, whose one
    query attends to every key (no causal mask); only that position reaches the logits.  Its q and k/v
    projections are ``F.linear`` on row slices of the fused weight, so a subclass that replaces ``_lin``
    (:class:`DynInt8Decoder`) does not prune."""

    def __init__(self, cfg, n_layers: int, dtype: torch.dtype, device: torch.device, seed: int = 0,
                 prune_last: bool = False):
        if prune_last and type(self)._lin is not FloatDecoder._lin:
            raise ValueError("prune_last slices the fused q/k/v weight, which a replaced _lin would not see")
        g = torch.Generator().manual_seed(seed)
        d, f, dh, hq, hkv = cfg.d_model, cfg.d_ff, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        self.cfg, self.dtype, self.device, self.prune_last = cfg, dtype, device, prune_last

        def w(n, k):
            return (torch.randn(n, k, generator=g) / k ** 0.5).to(dtype).to(device)

        def b(n):
            return torch.zeros(n, dtype=dtype, device=device) if cfg.bias else None

        self.blocks = []
        for _ in range(n_layers):
            blk = {"qkv": (w((hq + 2 * hkv) * dh, d), b((hq + 2 * hkv) * dh)), "o": (w(d, hq * dh), b(d))}
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
        for i, blk in enumerate(self.blocks):
            h = self._norm(x)
            if self.prune_last and i == len(self.blocks) - 1:     # q at the last position, k and v at every one
                w, b = blk["qkv"]
                q = F.linear(h[:, -1:], w[:hq * dh], None if b is None else b[:hq * dh])
                k, v = F.linear(h, w[hq * dh:], None if b is None else b[hq * dh:]).split([hkv * dh, hkv * dh], -1)
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

    def __init__(self, cfg, n_layers: int, device: torch.device, seed: int = 0):
        super().__init__(cfg, n_layers, torch.float32, device, seed)
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--seq", type=int, nargs="+", default=[64, 512, 2048])
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--variants", nargs="+", default=["int", "float"],
                    choices=["int", "int_fp32gemm", "float", "bf16", "dynint8"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--max-pos", type=int, default=0, help="rotary-position models: a longer context (V-LONG-b)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    device = torch.device(args.device)
    fdtype = torch.float16 if device.type == "cuda" else torch.float32
    from pvi.results import git_sha
    sha = git_sha(CODE)
    for name in args.model:
        cfg = CONFIGS[name]
        if args.max_pos > cfg.max_pos:
            if cfg.pos != "rope":
                raise SystemExit(f"--max-pos: {name} has learned positions (a longer context is another model)")
            cfg = dataclasses.replace(cfg, max_pos=args.max_pos)
        for seq in args.seq:
            if seq > cfg.max_pos:                     # e.g. GPT-2's 1,024 learned positions
                continue
            tokens =torch.randint(0, cfg.vocab, (1, seq), generator=torch.Generator().manual_seed(1)).to(device)
            for variant in args.variants:
                measured, spread = {}, {}
                _reset_peaks(device)
                for n_layers in args.layers:
                    from pvi.fullcheck import native_kernels
                    if variant in ("int", "int_fp32gemm"):
                        graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(seq, 32), seed=0,
                                              prune_last=True)

                        def run(graph=graph, native=variant == "int"):
                            saved = native_kernels._GEMM
                            native_kernels._GEMM = native and saved
                            try:
                                with torch.no_grad():
                                    graph.forward(tokens, free=True)
                            finally:
                                native_kernels._GEMM = saved
                    elif variant == "dynint8":
                        model = DynInt8Decoder(cfg, n_layers, device)

                        def run(model=model):
                            with torch.no_grad():
                                model(tokens)
                    else:
                        model = FloatDecoder(cfg, n_layers, torch.bfloat16 if variant == "bf16" else fdtype, device)

                        def run(model=model):
                            with torch.no_grad():
                                model(tokens)
                    times = _timed(run, device, args.reps)
                    measured[n_layers] = statistics.median(times)
                    spread[n_layers] = [min(times), max(times)]
                    del run
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                ls = sorted(measured)
                per_block = ((measured[ls[-1]] - measured[ls[0]]) / (ls[-1] - ls[0]) if len(ls) > 1
                             else measured[ls[0]] / ls[0])
                fixed = measured[ls[0]] - ls[0] * per_block
                print(json.dumps({
                    "git": sha, "model": name, "seq": seq, "variant": variant,
                    "dtype": {"int": "int8 GEMM", "int_fp32gemm": "int (fp32 GEMM)", "bf16": "bfloat16",
                              "dynint8": "dynamic int8 (not exact)"}.get(variant, str(fdtype).replace("torch.", "")),
                    "device": str(device), "threads": args.threads, "reps": args.reps,
                    "measured_s": {str(k): v for k, v in measured.items()},
                    "range_s": {str(k): v for k, v in spread.items()}, "peak": _peaks(device),
                    "max_pos": cfg.max_pos,
                    "per_block_s": per_block, "fixed_s": fixed, "n_layers": cfg.n_layers,
                    "full_s": fixed + cfg.n_layers * per_block,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "cpu": platform.processor() or platform.machine(), "host": platform.node(),
                    "torch": torch.__version__}), flush=True)


if __name__ == "__main__":
    sys.exit(main())

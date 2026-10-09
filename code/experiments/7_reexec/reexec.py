"""The cost of re-running the model: the baseline against which a verifier's time is read.

    cd code && PYTHONPATH=src python experiments/7_reexec/reexec.py --model llama2-7b --seq 64 512 2048 --device cuda
    cd code && PYTHONPATH=src python experiments/7_reexec/reexec.py --model opt-1.3b --seq 64 2048 --device cpu --threads 8

A client that holds the weights (settings K and Kpre) could check an answer by computing it again.
This script times that, for the decoders of ``pvi.fullcheck.transformer.CONFIGS`` (random weights:
the cost depends on the shapes only), in two variants:

* ``int``: our integer model's forward pass (``IntGraph.forward`` of ``build_decoder(...,
  prune_last=True)``), which the prover runs and which is bit-exact with the claims, so it is a
  complete check in itself;
* ``float``: a plain floating-point decoder of the same shapes (fp16 on a GPU, fp32 on the CPU):
  ``torch.nn.functional.linear`` and ``scaled_dot_product_attention`` with a causal mask, the norms
  and the MLP of the architecture, the LM head at the last position.  It is what a client would run
  with an off-the-shelf stack; it skips the rotary embedding (an element-wise cost) and is not
  bit-exact with the provider's model.

Each variant is timed on ``--layers`` blocks (default 1 and 2); with ``t(L) = a + L b`` the script
extrapolates to the full model's ``n_layers`` and prints one JSON line per (model, T, variant), with
the measured medians and ``full_s``.  Large models fit a small GPU this way.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

CODE = Path.cwd()   # run from code/


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _timed(fn, device: torch.device, reps: int) -> float:
    fn()                                         # warm-up
    times = []
    for _ in range(reps):
        _sync(device)
        t0 = time.perf_counter()
        fn()
        _sync(device)
        times.append(time.perf_counter() - t0)
    return statistics.median(times)


class FloatDecoder:
    """A floating-point decoder with the shapes of ``cfg`` and ``n_layers`` blocks (random weights)."""

    def __init__(self, cfg, n_layers: int, dtype: torch.dtype, device: torch.device, seed: int = 0):
        g = torch.Generator().manual_seed(seed)
        d, f, dh, hq, hkv = cfg.d_model, cfg.d_ff, cfg.head_dim, cfg.n_heads, cfg.n_kv_heads
        self.cfg, self.dtype, self.device = cfg, dtype, device

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
            qkv = F.linear(self._norm(x), *blk["qkv"])
            q, k, v = qkv.split([hq * dh, hkv * dh, hkv * dh], -1)
            q = q.view(bsz, t, hq, dh).transpose(1, 2)
            k = k.view(bsz, t, hkv, dh).transpose(1, 2)
            v = v.view(bsz, t, hkv, dh).transpose(1, 2)
            if hkv != hq:
                k, v = k.repeat_interleave(hq // hkv, 1), v.repeat_interleave(hq // hkv, 1)
            a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            x = x + F.linear(a.transpose(1, 2).reshape(bsz, t, hq * dh), *blk["o"])
            h = F.linear(self._norm(x), *blk["up"])
            if cfg.mlp == "swiglu":
                gate, up = h.chunk(2, -1)
                h = F.silu(gate) * up
            else:
                h = F.relu(h) if cfg.mlp == "relu" else F.gelu(h)
            x = x + F.linear(h, *blk["down"])
        return F.linear(self._norm(x[:, -1]), self.head)          # next-token logits


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", nargs="+", required=True)
    ap.add_argument("--seq", type=int, nargs="+", default=[64, 512, 2048])
    ap.add_argument("--layers", type=int, nargs="+", default=[1, 2])
    ap.add_argument("--variants", nargs="+", default=["int", "float"], choices=["int", "float"])
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--reps", type=int, default=5)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    device = torch.device(args.device)
    fdtype = torch.float16 if device.type == "cuda" else torch.float32
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=CODE).stdout.strip()
    for name in args.model:
        cfg = CONFIGS[name]
        for seq in args.seq:
            if seq > cfg.max_pos:                     # e.g. GPT-2's 1,024 learned positions
                continue
            tokens =torch.randint(0, cfg.vocab, (1, seq), generator=torch.Generator().manual_seed(1)).to(device)
            for variant in args.variants:
                measured = {}
                for n_layers in args.layers:
                    if variant == "int":
                        graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=min(seq, 32), seed=0,
                                              prune_last=True)

                        def run(graph=graph):
                            with torch.no_grad():
                                graph.forward(tokens, free=True)
                    else:
                        model = FloatDecoder(cfg, n_layers, fdtype, device)

                        def run(model=model):
                            with torch.no_grad():
                                model(tokens)
                    measured[n_layers] = _timed(run, device, args.reps)
                    del run
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
                ls = sorted(measured)
                per_block = ((measured[ls[-1]] - measured[ls[0]]) / (ls[-1] - ls[0]) if len(ls) > 1
                             else measured[ls[0]] / ls[0])
                fixed = measured[ls[0]] - ls[0] * per_block
                print(json.dumps({
                    "git": sha, "model": name, "seq": seq, "variant": variant,
                    "dtype": "int" if variant == "int" else str(fdtype).replace("torch.", ""),
                    "device": str(device), "threads": args.threads, "reps": args.reps,
                    "measured_s": {str(k): v for k, v in measured.items()},
                    "per_block_s": per_block, "fixed_s": fixed, "n_layers": cfg.n_layers,
                    "full_s": fixed + cfg.n_layers * per_block,
                    "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
                    "cpu": platform.processor() or platform.machine(), "host": platform.node(),
                    "torch": torch.__version__}), flush=True)


if __name__ == "__main__":
    sys.exit(main())

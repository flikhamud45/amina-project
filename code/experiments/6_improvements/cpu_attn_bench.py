"""The CPU verifier's exact attention alone (plan B5): torch's int32 path, the tiled native kernel and the hybrid
(float32 GEMMs + native integer softmax), against fp32 attention as re-execution computes it, on one block's heads.

    cd code && PYTHONPATH=src python experiments/6_improvements/cpu_attn_bench.py --heads 32 --kv 32 --dh 128 --seq 2048

Every exact variant is checked against torch's int32 path (bit-identical) before it is timed. One JSON line:
the median seconds of ``--reps`` runs after a warm-up, per variant.
"""
from __future__ import annotations

import argparse
import json
import platform
import statistics
import time

import torch
import torch.nn.functional as F

from pvi.fullcheck import native_kernels
from pvi.fullcheck import transformer as tr


def _time(fn, reps: int) -> float:
    fn()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--heads", type=int, default=32)
    ap.add_argument("--kv", type=int, default=32)
    ap.add_argument("--dh", type=int, default=128)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--queries", type=int, default=None, help="query rows (default: --seq; 1 is a pruned last block)")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--group", type=int, default=8, help="query heads per torch call (as transformer._attention)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    t, tq, dh, rep = args.seq, args.queries or args.seq, args.dh, args.heads // args.kv
    g = torch.Generator().manual_seed(0)
    q = torch.randint(-127, 128, (1, args.kv, rep * tq, dh), generator=g)
    k = torch.randint(-127, 128, (1, args.kv, t, dh), generator=g)
    v = torch.randint(-127, 128, (1, args.kv, t, dh), generator=g)
    m_s = 1 << 20
    lut = tr._exp_lut(m_s, "cpu")
    kvg = max(1, args.group // rep)                                  # key/value heads per call

    def torch_int32():
        return torch.cat([tr._attention_core(q[:, i:i + kvg], k[:, i:i + kvg], v[:, i:i + kvg], lut,
                                             tr._causal_notmask(t, rep, "cpu", tq)) for i in range(0, args.kv, kvg)], 1)

    want = torch_int32()
    variants = {"torch_int32": torch_int32}
    if native_kernels.available():
        variants["native_tiled"] = lambda: native_kernels.attention_core(q, k, v, lut, tq)
        variants["native_hybrid"] = lambda: native_kernels.attention_core_hybrid(q, k, v, lut, tq)
        for name in ("native_tiled", "native_hybrid"):
            assert torch.equal(variants[name](), want), name
    qf = (q.view(1, args.kv, rep, tq, dh).reshape(1, args.heads, tq, dh).float() / 64)
    kf = k.float().repeat_interleave(rep, 1) / 64
    vf = v.float().repeat_interleave(rep, 1) / 64
    variants["fp32_sdpa"] = lambda: F.scaled_dot_product_attention(qf, kf, vf, is_causal=tq == t)
    rows = {name: _time(fn, args.reps) for name, fn in variants.items()}
    print(json.dumps({"heads": args.heads, "kv": args.kv, "dh": dh, "seq": t, "queries": tq, "threads": args.threads,
                      "seconds": rows, "host": platform.node(), "cpu": platform.processor(),
                      "torch": torch.__version__}), flush=True)


if __name__ == "__main__":
    main()

"""Where the CPU verifier spends its derive (plan B5): one query after a warm-up, with every recomputed operation
(``CheapOp.fn``) and every weight op's fold timed, grouped by the op's name with the block index removed, then the
torch operators ranked by self CPU time (``torch.profiler``).

    cd code && PYTHONPATH=src python experiments/6_improvements/cpu_profile.py --model llama2-7b --layers 2 --seq 2048
"""
from __future__ import annotations

import argparse
import collections
import re
import time

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from torch.profiler import ProfilerActivity, profile

    from pvi.fullcheck.analytic import decoder_shapes
    from pvi.fullcheck.graph import CheapOp, MatOp
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, params_for, run_query
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    cfg = CONFIGS[args.model]
    graph = build_decoder(cfg, n_layers=args.layers, calib_tokens=32, seed=0, prune_last=True)
    ops = decoder_shapes(cfg, prune_last=True)
    params = params_for(128, len(ops))
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    v = Verifier(graph.public(), params, "Kpre", weights={op.name: (op.weight, op.bias) for op in graph.mat_ops},
                 lean=True, device="cpu", lookups=True)
    v.precompute(Challenger())
    prover = Prover(graph, device=dev, lean=True)
    xs = torch.randint(0, cfg.vocab, (3, args.seq), generator=torch.Generator().manual_seed(1)).split(1)
    warm = run_query(prover, v, xs[0], wire=True)
    print("warm-up", warm["accepted"], {k: round(t, 3) for k, t in warm["timings"].items() if t > 0.005})

    spent = collections.defaultdict(float)
    count = collections.Counter()

    def timed(key, fn):
        def run(*a, **k):          # the prover's and the verifier's graphs share their ops: split by device
            dev = next((x.device.type for x in a if torch.is_tensor(x)), "cpu")
            t0 = time.perf_counter()
            out = fn(*a, **k)
            spent[f"{key} {dev}"] += time.perf_counter() - t0
            count[f"{key} {dev}"] += 1
            return out
        return run

    for op in v.graph.ops:
        key = re.sub(r"\d+", "#", op.name)
        if isinstance(op, CheapOp) and op.fn is not None:
            object.__setattr__(op, "fn", timed(f"{key} [{op.note or 'cheap'}]", op.fn))
        elif isinstance(op, MatOp):
            object.__setattr__(op, "fold", timed(f"{key} [fold]", op.fold))
    out = run_query(prover, v, xs[1], wire=True)
    print("timed", out["accepted"], {k: round(t, 3) for k, t in out["timings"].items() if t > 0.005})
    total = sum(spent.values())
    print(f"== derive ops: {total:.3f} s in {sum(count.values())} calls")
    for key, s in sorted(spent.items(), key=lambda kv: -kv[1])[: args.top]:
        print(f"{s:8.3f} s {100 * s / total:5.1f}%  x{count[key]:<3d} {key}")
    with profile(activities=[ProfilerActivity.CPU]) as prof:
        run_query(prover, v, xs[2], wire=True)
    print("== torch operators by self CPU time")
    print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=args.top, max_name_column_width=50))


if __name__ == "__main__":
    main()

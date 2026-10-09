"""Where the streaming GPU verifier spends its time (plan B1): one profiled query after a warm-up, the CUDA kernels
and the CPU-side operators ranked by total time (``torch.profiler``), and the query's own phase timings.

    cd code && PYTHONPATH=src python experiments/6_improvements/gpu_profile.py --model llama2-7b --layers 2 --seq 2048
"""
from __future__ import annotations

import argparse

import torch


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, default=0)
    ap.add_argument("--seq", type=int, default=2048)
    ap.add_argument("--mode", default="Kpre")
    ap.add_argument("--policy", default="auto")
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    torch.set_num_threads(8)
    from torch.profiler import ProfilerActivity, profile

    from pvi.fullcheck.analytic import decoder_shapes
    from pvi.fullcheck.protocol import Challenger, Prover, Verifier, commit_graph, params_for, run_query
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    cfg = CONFIGS[args.model]
    graph = build_decoder(cfg, n_layers=args.layers or cfg.n_layers, calib_tokens=32, seed=0, prune_last=True)
    ops = decoder_shapes(cfg, prune_last=True)
    if args.mode == "C":
        coms = commit_graph(graph, 4, device="cuda", policy=args.policy, model_ops=ops)
        params = params_for(128, len(ops), plan=coms.plan)
        v = Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                     tables=coms.table_publics, lean=True, device="cuda", stream=True)
        prover = Prover(graph, device="cuda", commitments=coms, lean=True)
    else:
        params = params_for(128, len(ops))
        v = Verifier(graph.public(), params, "Kpre", weights={op.name: (op.weight, op.bias) for op in graph.mat_ops},
                     lean=True, device="cuda", stream=True, lookups=True)
        v.precompute(Challenger())
        prover = Prover(graph, device="cuda", lean=True)
    xs = torch.randint(0, cfg.vocab, (3, args.seq), generator=torch.Generator().manual_seed(1)).split(1)
    warm = run_query(prover, v, xs[0], wire=True)
    print("warm-up", warm["accepted"], {k: round(t, 3) for k, t in warm["timings"].items() if t > 0.005})
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = run_query(prover, v, xs[1], wire=True)
        torch.cuda.synchronize()
    print("profiled", out["accepted"], {k: round(t, 3) for k, t in out["timings"].items() if t > 0.005})
    ka = prof.key_averages()
    print("== by CUDA time")
    print(ka.table(sort_by="cuda_time_total", row_limit=args.top, max_name_column_width=60))
    print("== by CPU time (self)")
    print(ka.table(sort_by="self_cpu_time_total", row_limit=args.top, max_name_column_width=60))


if __name__ == "__main__":
    main()

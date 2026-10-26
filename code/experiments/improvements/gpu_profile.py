"""Where a query's CUDA time goes with the streaming GPU verifier: one profiled query after a warm-up.

The shares of ``sections/ext_evaluation.tex`` after ``tab:eval-improve`` ("copies take 32% of the CUDA time of an
OPT-1.3B query (12 blocks) and int8 GEMMs 4.4%").

    python -m experiments.improvements.gpu_profile --model opt-1.3b --layers 12 --seq 2048
    python -m experiments.improvements.gpu_profile --model llama2-7b --layers 2 --seq 2048 --top 40

Builds the decoder as ``bench.py`` does (random weights, ``--seed``; ``--layers`` blocks, 0: all of them), proves and
checks one query in ``--mode`` (default Kpre, with the verifier's own embedding rows) with the prover and the streaming
verifier on the GPU, under the protocol flags of :func:`experiments.common.cli.add_protocol_arguments` (default the
paper's configuration), then profiles a second query with ``torch.profiler``.  It prints the query's phase timings,
the operators ranked by CUDA time and by CPU time, and one JSON line (``kind`` ``gpu_profile``): ``cuda_total_s`` (the
sum of the device-side events' own time, as the profiler's table totals it) and, for each ``--share`` operator
(default ``aten::copy_``, the host-to-device and device-to-host copies; ``aten::_int_mm``, the int8 GEMMs;
``aten::index_select``; and ``_attn_kernel``, the fused attention kernel), its CUDA time and its share of the total,
with the timings, the acceptance, ``prob_max``, the GPU and ``env``.  Needs a CUDA GPU; minutes.
"""
from __future__ import annotations

import argparse
import platform
import sys

import torch

from experiments.common import cli, records
from experiments.common.models import decoder_config
from experiments.common.paths import CODE

SHARES = ("aten::copy_", "aten::_int_mm", "aten::index_select", "_attn_kernel")


def _device_time(event, own: bool) -> float:
    """An averaged profiler event's CUDA time in microseconds (``own``: without its children's)."""
    for name in (("self_device_time_total", "self_cuda_time_total") if own else
                 ("device_time_total", "cuda_time_total")):
        if hasattr(event, name):
            return float(getattr(event, name))
    return 0.0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="opt-1.3b", help="a decoder of pvi.models.decoder.CONFIGS")
    ap.add_argument("--layers", type=int, default=12, help="decoder blocks (0: the full model)")
    ap.add_argument("--seq", type=int, default=2048, help="prompt tokens")
    ap.add_argument("--mode", default="Kpre", choices=["C", "Kpre"])
    ap.add_argument("--top", type=int, default=30, help="rows of the profiler tables")
    ap.add_argument("--share", nargs="*", default=list(SHARES), help="operators whose share of the CUDA time to print")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed")
    ap.add_argument("--prompt-seed", type=int, default=1, help="the random prompts' seed")
    cli.add_protocol_arguments(ap)
    cli.add_runtime_arguments(ap, threads=8, verifier_device=False)
    args = ap.parse_args()
    cli.apply_runtime(args)
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    from pvi.commit.cost import decoder_shapes
    from pvi.models.decoder import build_decoder
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("this profiles the GPU verifier: --device cuda")
    cfg = decoder_config(args.model)
    graph = build_decoder(cfg, n_layers=args.layers or cfg.n_layers, calib_tokens=min(args.seq, cli.CALIB_TOKENS),
                          seed=args.seed, prune_last=args.prune_last, prob_max=args.prob_max)
    ops = decoder_shapes(cfg, prune_last=args.prune_last)
    if args.mode == "C":
        coms = commit_graph(graph, args.rate, device=device, policy=args.policy, model_ops=ops, cache=args.setup_cache)
        params = params_for(args.lam, len(ops), rate=args.rate, plan=coms.plan, fiat_shamir=args.fs)
        v = Verifier(graph.public(), params, "C", publics=coms.publics, groups=coms.group_publics,
                     tables=coms.table_publics, lean=args.lean, device=device, stream=True)
        prover = Prover(graph, device=device, commitments=coms, lean=args.lean)
    else:
        params = params_for(args.lam, len(ops), rate=args.rate, fiat_shamir=args.fs)
        v = Verifier(graph.public(), params, "Kpre", weights={op.name: (op.weight, op.bias) for op in graph.mat_ops},
                     lean=args.lean, device=device, stream=True, lookups=args.lookups)
        v.precompute(Challenger())
        prover = Prover(graph, device=device, lean=args.lean)
    xs = torch.randint(0, cfg.vocab, (3, args.seq), generator=torch.Generator().manual_seed(args.prompt_seed)).split(1)
    warm = run_query(prover, v, xs[0], wire=args.wire)
    print("warm-up", warm["accepted"], {k: round(t, 3) for k, t in warm["timings"].items() if t > 0.005})
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        out = run_query(prover, v, xs[1], wire=args.wire)
        torch.cuda.synchronize()
    print("profiled", out["accepted"], {k: round(t, 3) for k, t in out["timings"].items() if t > 0.005})
    ka = prof.key_averages()
    print("== by CUDA time")
    print(ka.table(sort_by="cuda_time_total", row_limit=args.top, max_name_column_width=60))
    print("== by CPU time (self)")
    print(ka.table(sort_by="self_cpu_time_total", row_limit=args.top, max_name_column_width=60))
    # the total as the profiler's table counts it: the device-side events' own time (kernels, copies)
    total = sum(_device_time(e, own=True) for e in ka if getattr(e, "device_type", None) == DeviceType.CUDA)
    shares = {}
    for name in args.share:
        t = sum(_device_time(e, own=False) for e in ka if e.key == name)
        shares[name] = {"cuda_s": t / 1e6, "share": t / total if total else None}
    records.emit({"kind": "gpu_profile", "git": records.git_sha(CODE), "model": args.model,
                  "layers": args.layers or cfg.n_layers, "seq": args.seq, "mode": args.mode, "fs": args.fs,
                  "wire": args.wire, "prob_max": args.prob_max, "accepted": bool(warm["accepted"] and out["accepted"]),
                  "timings": out["timings"], "cuda_total_s": total / 1e6, "shares": shares,
                  "gpu": torch.cuda.get_device_name(device), "host": platform.node(),
                  "env": records.environment(device)})
    return 0 if warm["accepted"] and out["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())

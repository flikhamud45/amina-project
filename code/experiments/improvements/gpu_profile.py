"""Where a query's CUDA time goes with the streaming GPU verifier: one profiled query after a warm-up.

The shares of ``sections/ext_evaluation.tex`` after ``tab:eval-improve`` ("copies take 32% of the CUDA time of an
OPT-1.3B query (12 blocks) and int8 GEMMs 4.4%").

    python -m experiments.improvements.gpu_profile --model opt-1.3b --layers 12 --seq 2048
    python -m experiments.improvements.gpu_profile --model llama2-7b --layers 2 --seq 2048 --top 40
    python -m experiments.improvements.gpu_profile --model opt-1.3b --real facebook/opt-1.3b --layers 12 --seq 2048

Builds the decoder as ``bench.py`` does (random weights, ``--seed``; ``--layers`` blocks, 0: all of them), proves and
checks one query in ``--mode`` (default Kpre, with the verifier's own embedding rows) with the prover and the streaming
verifier on the GPU, under the protocol flags of :func:`experiments.common.cli.add_protocol_arguments` (default the
paper's configuration), then profiles a second query with ``torch.profiler``.  It prints the query's phase timings,
the operators ranked by CUDA time and by CPU time, and one JSON line (``kind`` ``gpu_profile``): ``cuda_total_s`` (the
sum of the device-side events' own time, as the profiler's table totals it) and, for each ``--share`` operator
(default ``aten::copy_``, the host-to-device and device-to-host copies; ``aten::_int_mm``, the int8 GEMMs;
``aten::index_select``; and ``_attn_kernel``, the fused attention kernel), its CUDA time and its share of the total,
with ``model``, ``real`` (the checkpoint; None for random weights), ``recipe``, ``smooth``, ``calib_seq``
(:func:`experiments.common.real_cells.fields`), ``layers``, ``seq``, the timings, the acceptance, ``prob_max`` (the
graph's attention probabilities' scale), the GPU and ``env``.  Needs a CUDA GPU; minutes.

``--real`` (a Hugging Face checkpoint; ``--model`` names its shapes): instead of random weights and prompts, the integer
graph of the checkpoint's first ``--layers`` blocks (:mod:`experiments.common.real_cells`; OPT: the certified recipe,
or ``--recipe legacy`` with ``--smooth`` and ``--calib-seq``; a Llama-family checkpoint: ``hf_llama.MAIN_RECIPE`` with
its fixed public prefix), its warm-up and profiled queries the first two WikiText-2 test windows of ``--seq`` tokens;
mode C's plan is laid out on the whole model's weight ops as the real graph has them.  Refused: shapes without an
integer builder of a real checkpoint (GPT-2), ``--prob-max`` (the recipe fixes the scale) and a ``--seq`` beyond the
checkpoint's trained context.  The checkpoint is loaded in fp32 and calibrated on the CPU first (host memory: its fp32
weights, 5.3 GB for OPT-1.3B and 27 GB for Llama-2-7B, next to the integer graph).
"""
from __future__ import annotations

import argparse
import platform
import sys

import torch

from experiments.common import cli, real_cells, records
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


def setup(args, cfg):
    """``(graph, xs, real)``: the graph of ``--layers`` blocks, the warm-up and profiled queries and the checkpoint's
    load (None for random weights: ``bench.py``'s graph and random prompts)."""
    if args.real:
        real = real_cells.load(args.real, cfg, seq=args.seq, n_queries=2, recipe=args.recipe,
                              calib_seq=args.calib_seq)
        graph = real.graph(args.prune_last, args.layers, args.smooth)[0]
        real.free()
        return graph, real.queries, real
    from pvi.models.decoder import build_decoder
    graph = build_decoder(cfg, n_layers=args.layers or cfg.n_layers, calib_tokens=min(args.seq, cli.CALIB_TOKENS),
                          seed=args.seed, prune_last=args.prune_last, prob_max=args.prob_max)
    xs = torch.randint(0, cfg.vocab, (3, args.seq), generator=torch.Generator().manual_seed(args.prompt_seed)).split(1)
    return graph, xs, None


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
    real_cells.add_arguments(ap, recipes=True)
    cli.add_protocol_arguments(ap)
    cli.add_runtime_arguments(ap, threads=8, verifier_device=False)
    args = ap.parse_args()
    cfg = decoder_config(args.model)
    real_cells.check_args(args, cfg, layers=[args.layers], seqs=[args.seq], prob_max=args.prob_max)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("this profiles the GPU verifier: --device cuda")
    cli.apply_runtime(args)
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    from pvi.models.decoder import prob_max_of
    from pvi.v0.graph_commitment import commit_graph
    from pvi.v0.params import params_for
    from pvi.v0.prover import Prover
    from pvi.v0.query import run_query
    from pvi.v0.transcript import Challenger
    from pvi.v0.verifier import Verifier

    sha = records.git_sha(CODE)              # before the build, which can outlast a checkout update
    graph, xs, real = setup(args, cfg)
    ops = real_cells.model_ops(cfg, args.prune_last, bool(args.real))
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
    records.emit({"kind": "gpu_profile", "git": sha, "model": args.model, **real_cells.recipe_fields(real, args),
                  "layers": args.layers or cfg.n_layers, "seq": args.seq, "mode": args.mode, "fs": args.fs,
                  "wire": args.wire, "prob_max": prob_max_of(graph),
                  "accepted": bool(warm["accepted"] and out["accepted"]),
                  "timings": out["timings"], "cuda_total_s": total / 1e6, "shares": shares,
                  "gpu": torch.cuda.get_device_name(device), "host": platform.node(),
                  "env": records.environment(device)})
    return 0 if warm["accepted"] and out["accepted"] else 1


if __name__ == "__main__":
    sys.exit(main())

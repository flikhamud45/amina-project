"""The prover's forward pass: weight products on exact int8 GEMMs against float32 products, on the same queries.

The row "Weight products on int8 GEMMs (forward pass)" of ``tab:eval-improve`` (``sections/ext_evaluation.tex``).

    python -m experiments.improvements.forward_ab --model llama2-7b --layers 2 --seq 2048
    python -m experiments.improvements.forward_ab --model llama2-7b --real meta-llama/Llama-2-7b-hf --layers 2 \
        --seq 2048

``MatOp.compute`` runs a weight op on a GPU whose ``torch._int_mm`` passes ``field.int8_ok`` as int8 GEMMs
(``graph.int8_product``), else as float32 GEMMs over chunks of 1,024 terms; both give the same integers.  This script
builds the decoder as ``bench.py`` does (random weights, ``--seed``; ``--layers`` blocks, the last one pruned;
11-bit attention probabilities, ``--prob-max``), then
times the lean prover's forward pass with the claims kept for the compact encoding (``Prover.claims`` with
``claimcodec.narrow``: what ``run_query``'s ``prove_forward`` times) both ways on the same queries, alternating, and
checks that the claims are identical; the first query warms both up.  The float32 path is the implementation
option :data:`OPTION` of :mod:`pvi.options` turned off (``options.using(int8_forward=False)``).  A prompt length beyond
the model's trained context is skipped.

``--real`` (a Hugging Face checkpoint; ``--model`` names its shapes): instead of random weights and prompts, the integer
graph of the checkpoint's first ``--layers`` blocks (:mod:`experiments.common.real_weights`; OPT: the certified recipe,
or ``--recipe legacy`` with ``--smooth`` and ``--calib-seq``; a Llama-family checkpoint: ``hf_llama.MAIN_RECIPE`` with
its fixed public prefix), queried on WikiText-2 test windows of the one ``--seq`` (its calibration and windows come from
one load of the checkpoint); every block count and both variants take the same windows.  Refused: shapes without an
integer builder of a real checkpoint (GPT-2), ``--prob-max`` (the recipe fixes the scale), several ``--seq`` and a
``--seq`` beyond the checkpoint's trained context.

Record (one JSON line per model, block count, prompt length and variant ``float32`` / ``int8``): ``model``, ``real``
(the checkpoint; None for random weights), ``recipe``, ``smooth``, ``calib_seq``
(:func:`experiments.common.real_weights.fields`), ``layers``, ``seq``, ``prove_forward`` (the median),
``vs_float32``, ``identical``, ``queries``, ``int8_ok``, ``prob_max`` (the graph's attention probabilities' scale),
the GPU and ``env``.  Needs a CUDA GPU with int8 GEMMs; minutes per build.  With ``--real`` the checkpoint is loaded
in fp32 and calibrated on the CPU first (host memory: its fp32 weights, 27 GB for Llama-2-7B, next to the integer
graph).
"""
from __future__ import annotations

import argparse
import platform
import statistics
import sys
import time

import torch

from experiments.common import cli, real_weights, records
from experiments.common.models import decoder_config
from experiments.common.paths import CODE
from pvi import options

OPTION = "int8_forward"
"""The implementation option that turns the forward pass's int8 GEMMs off (float32 products)."""



def setup(args, cfg):
    """``(graphs, queries, real)``: the graph of ``n`` blocks (``graphs(n)``), the ``--queries + 1`` prompts of
    length ``seq`` (``queries(seq)``; the first warms up) and the checkpoint's load (None for random weights).  With
    ``--real`` every block count's graph is built here, from one load of the checkpoint, and the windows are its."""
    from pvi.models.decoder import build_decoder
    if args.real:
        real = real_weights.load(args, cfg, seq=args.seq[0], n_queries=args.queries + 1, layers=args.layers)
        built = {n: real_weights.graph(real, args, n, args.prune_last) for n in dict.fromkeys(args.layers)}
        del real.model
        return built.__getitem__, lambda seq: real.queries, real

    def graph(n):
        return build_decoder(cfg, n_layers=n or cfg.n_layers, calib_tokens=cli.CALIB_TOKENS, seed=args.seed,
                             prune_last=args.prune_last, prob_max=args.prob_max)

    def queries(seq):
        return list(torch.randint(0, cfg.vocab, (args.queries + 1, seq),
                                  generator=torch.Generator().manual_seed(args.prompt_seed)).split(1))
    return graph, queries, None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="llama2-7b", help="a decoder of pvi.models.decoder.CONFIGS")
    ap.add_argument("--layers", type=int, nargs="+", default=[2], help="decoder blocks (0: the full model)")
    ap.add_argument("--seq", type=int, nargs="+", default=[2048], help="prompt lengths")
    ap.add_argument("--queries", type=int, default=5, help="timed queries per variant, after one warm-up")
    ap.add_argument("--prune-last", action=argparse.BooleanOptionalAction, default=cli.PAPER.prune_last,
                    help="the last block at the last position only")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed")
    ap.add_argument("--prompt-seed", type=int, default=1, help="the random prompts' seed")
    cli.add_prob_max_argument(ap)
    real_weights.add_arguments(ap)
    cli.add_runtime_arguments(ap, threads=8, verifier_device=False)
    args = ap.parse_args()
    cfg = decoder_config(args.model)
    real_weights.check(args, cfg, layers=args.layers, seqs=args.seq, prob_max=args.prob_max)
    if args.real and len(set(args.seq)) > 1:
        raise SystemExit("--real takes one --seq: its calibration and windows come from one load of the checkpoint")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("the int8 forward pass runs on a GPU: --device cuda")
    cli.apply_runtime(args)
    sha = records.git_sha(CODE)              # before the build, which can outlast a checkout update
    graphs, queries, real = setup(args, cfg)
    env = records.environment(device)
    from pvi.core import field
    from pvi.models.decoder import prob_max_of
    from pvi.v0 import claimcodec
    from pvi.v0.prover import Prover

    variants = {"float32": {OPTION: False}, "int8": {}}
    ok = True
    for n_layers in args.layers:
        graph = graphs(n_layers)
        prover = Prover(graph, device=device, lean=True)
        for seq in args.seq:
            if seq > cfg.max_pos:
                continue
            times = {name: [] for name in variants}
            identical = True
            for i, q in enumerate(queries(seq)):
                got = {}
                for name in (list(variants) if i % 2 == 0 else list(variants)[::-1]):
                    with options.using(**variants[name]):
                        torch.cuda.synchronize()
                        t0 = time.perf_counter()
                        claims = prover.claims(q, send=claimcodec.narrow, to_host=False)
                        torch.cuda.synchronize()
                    if i:                                   # query 0 warms both up
                        times[name].append(time.perf_counter() - t0)
                    got[name] = claims
                identical &= all(torch.equal(got["float32"][k], got["int8"][k]) for k in got["float32"])
                del got, claims
            ok &= identical
            base = statistics.median(times["float32"])
            for name, ts in times.items():
                med = statistics.median(ts)
                records.emit({"git": sha, "model": args.model, **real_weights.fields(args, real),
                              "layers": n_layers or cfg.n_layers, "seq": seq,
                              "variant": name, "queries": len(ts), "identical": identical, "prove_forward": med,
                              "vs_float32": med / base, "int8_ok": field.int8_ok(device),
                              "prob_max": prob_max_of(graph),
                              "gpu": torch.cuda.get_device_name(device), "host": platform.node(), "env": env})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

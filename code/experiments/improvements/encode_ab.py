"""The prover's claim encoder (``claimcodec.encode`` on its device): the threaded assembly of the encoded bytes
against a single-threaded one, and the fused Triton encoder against the torch encoder, on the same claims.

The rows "Threaded assembly of the encoded claims (encoder)" of ``tab:eval-improve`` and the negative result of the
fused Triton encoder ("1.74 against 1.54 s on OPT-1.3B") in ``sections/ext_evaluation.tex``.

    python -m experiments.improvements.encode_ab --model opt-1.3b --layers 24 --seq 2048
    python -m experiments.improvements.encode_ab --model llama2-7b --layers 2 --seq 2048 --fused
    python -m experiments.improvements.encode_ab --model opt-1.3b --layers 4 --profile      # a torch.profiler table

Builds the decoder as ``bench.py`` does (random weights, ``--seed``; ``--layers`` blocks, the last one pruned),
computes one query's claims on the device as the lean prover keeps them for the compact encoding (int32,
``claimcodec.narrow``), then encodes them ``--reps`` times per variant after one untimed call each, the variants
interleaved, and checks that every variant gives the same bytes:

* ``one_thread``: the encoded pieces are assembled into one buffer on a single thread (torch's thread count set to 1
  for the call: ``claimcodec`` copies them on torch's threads);
* ``threaded``: on ``--threads`` threads (the default path);
* ``fused`` (with ``--fused``): the fused Triton encoder (the implementation option ``fused_encoder`` of
  :mod:`pvi.options`), on ``--threads`` threads.

Record (one JSON line per variant): ``encode_s`` (the median), ``all_s``, ``vs_one_thread``, ``same_bytes``, ``bytes``,
``claims``, ``ops``, the thread count, the GPU, peak GPU memory and ``env``; with ``--profile``, a ``torch.profiler``
table of one more threaded call, sorted by CPU and by CUDA time.  Needs a CUDA GPU (the device encoder); the full
OPT-1.3B at 2,048 tokens gives a proof of about 2 GB.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import platform
import statistics
import sys
import time

import torch

from experiments.common import cli, records
from experiments.common.models import decoder_config
from experiments.common.paths import CODE
from pvi import options


@contextlib.contextmanager
def _threads(n: int):
    saved = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(saved)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--model", default="opt-1.3b", help="a decoder of pvi.models.decoder.CONFIGS")
    ap.add_argument("--layers", type=int, default=0, help="decoder blocks (0: the full model)")
    ap.add_argument("--seq", type=int, default=2048, help="prompt tokens")
    ap.add_argument("--reps", type=int, default=3, help="timed encodings per variant, after one untimed call")
    ap.add_argument("--fused", action="store_true", help="also time the fused Triton encoder")
    ap.add_argument("--profile", action="store_true", help="print a torch.profiler table of one threaded call")
    ap.add_argument("--rows", type=int, default=25, help="rows of the profiler tables")
    ap.add_argument("--prune-last", action=argparse.BooleanOptionalAction, default=cli.PAPER.prune_last,
                    help="the last block at the last position only")
    ap.add_argument("--seed", type=int, default=0, help="the random weights' seed")
    ap.add_argument("--prompt-seed", type=int, default=1, help="the random prompt's seed")
    cli.add_runtime_arguments(ap, threads=8, verifier_device=False)
    args = ap.parse_args()
    cli.apply_runtime(args)
    from pvi.models.decoder import build_decoder
    from pvi.v0 import claimcodec
    from pvi.v0.prover import Prover

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("the device encoder runs on a GPU: --device cuda")
    cfg = decoder_config(args.model)
    sha, env = records.git_sha(CODE), records.environment(device)
    n_layers = args.layers or cfg.n_layers
    graph = build_decoder(cfg, n_layers=n_layers, calib_tokens=cli.CALIB_TOKENS, seed=args.seed,
                          prune_last=args.prune_last)
    prover = Prover(graph, device=device, lean=True)
    x = torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(args.prompt_seed))
    claims = prover.claims(x, send=claimcodec.narrow, to_host=False)
    zs = [claims[op.name] for op in graph.mat_ops if op.name in claims]
    n = sum(z.numel() for z in zs)
    variants = {"one_thread": dict(threads=1, opts={}), "threaded": dict(threads=args.threads, opts={})}
    if args.fused:
        variants["fused"] = dict(threads=args.threads, opts={"fused_encoder": True})
    times = {name: [] for name in variants}
    blobs = {}
    for rep in range(args.reps + 1):                  # the first round is untimed
        for name in (list(variants) if rep % 2 == 0 else list(variants)[::-1]):
            v = variants[name]
            with _threads(v["threads"]), options.using(**v["opts"]):
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                blob = claimcodec.encode(zs)
                torch.cuda.synchronize()
                dt = time.perf_counter() - t0
            if rep:
                times[name].append(dt)
            if name not in blobs:                     # (a digest: a proof's bytes can take gigabytes)
                blobs[name] = (hashlib.sha256(blob).hexdigest(), len(blob))
            del blob
    same = len(set(blobs.values())) == 1
    base = statistics.median(times["one_thread"])
    for name, ts in times.items():
        med = statistics.median(ts)
        records.emit({"git": sha, "model": args.model, "layers": n_layers, "seq": args.seq, "variant": name,
                      "threads": variants[name]["threads"], "encode_s": med, "all_s": [round(t, 4) for t in ts],
                      "vs_one_thread": med / base, "same_bytes": same, "bytes": blobs[name][1], "claims": n,
                      "ops": len(zs), "gpu_peak_memory": torch.cuda.max_memory_allocated(device),
                      "gpu": torch.cuda.get_device_name(device), "host": platform.node(), "env": env})
    if args.profile:
        from torch.profiler import ProfilerActivity, profile
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            claimcodec.encode(zs)
            torch.cuda.synchronize()
        ka = prof.key_averages()
        print(ka.table(sort_by="self_cpu_time_total", row_limit=args.rows))
        print(ka.table(sort_by="self_cuda_time_total", row_limit=args.rows))
    return 0 if same else 1


if __name__ == "__main__":
    sys.exit(main())

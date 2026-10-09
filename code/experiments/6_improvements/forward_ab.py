"""A/B of the prover's forward pass: weight products as float32 chunks against exact int8 GEMMs.

    cd code && PYTHONPATH=src python experiments/6_improvements/forward_ab.py --model llama2-7b --layers 2 --seq 64 2048

``MatOp.compute`` runs a weight op on a GPU whose ``torch._int_mm`` passes ``field.int8_ok`` as int8
GEMMs (``graph.int8_product``), else as float32 GEMMs over chunks of 1,024 terms (the code before).
This script builds the decoder as bench.py does (random weights, seed 0, last block pruned), then
times the lean wire prover's forward (``Prover.claims`` with ``claimcodec.narrow``: what bench.py's
``prove_forward`` times) both ways on the same queries, alternating, and checks that the claims are
identical.  One JSON line per (model, layers, T, variant), with ``vs_float32``.
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

CODE = Path.cwd()   # run from code/


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--layers", type=int, nargs="+", default=[2], help="decoder blocks (0: full model)")
    ap.add_argument("--seq", type=int, nargs="+", default=[64, 2048])
    ap.add_argument("--queries", type=int, default=5)
    args = ap.parse_args()
    from pvi.fullcheck import claimcodec, field, graph as graph_mod
    from pvi.fullcheck.protocol import Prover
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    device = torch.device("cuda")
    cfg = CONFIGS[args.model]
    sha = subprocess.run(["git", "rev-parse", "--short", "HEAD"], capture_output=True, text=True, cwd=CODE).stdout.strip()
    variants = {"float32": lambda d: False, "int8": field.int8_ok}
    for n_layers in args.layers:
        graph = build_decoder(cfg, n_layers=n_layers or cfg.n_layers, calib_tokens=32, seed=0, prune_last=True)
        prover = Prover(graph, device=device, lean=True)
        for seq in args.seq:
            if seq > cfg.max_pos:
                continue
            tokens = torch.randint(0, cfg.vocab, (args.queries + 1, seq), generator=torch.Generator().manual_seed(1))
            times = {name: [] for name in variants}
            identical = True
            for i, q in enumerate(tokens.split(1)):
                got = {}
                for name in (list(variants) if i % 2 == 0 else list(variants)[::-1]):
                    graph_mod.int8_ok = variants[name]
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    claims = prover.claims(q, send=claimcodec.narrow, to_host=False)
                    torch.cuda.synchronize()
                    if i:                                   # query 0 warms both up
                        times[name].append(time.perf_counter() - t0)
                    got[name] = claims
                identical &= all(torch.equal(got["float32"][k], got["int8"][k]) for k in got["float32"])
                del got, claims
            graph_mod.int8_ok = field.int8_ok
            base = statistics.median(times["float32"])
            for name, ts in times.items():
                med = statistics.median(ts)
                print(json.dumps({"git": sha, "model": args.model, "layers": n_layers or cfg.n_layers, "seq": seq,
                                  "variant": name, "queries": len(ts), "identical": identical, "prove_forward": med,
                                  "vs_float32": med / base, "int8_ok": field.int8_ok(device),
                                  "gpu": torch.cuda.get_device_name(device), "host": platform.node()}), flush=True)


if __name__ == "__main__":
    sys.exit(main())

"""PS2 (c): the setup cache on a real model size, one JSON line per step.

    PYTHONPATH=src python experiments/6_improvements/setup_cache_check.py --model llama2-7b --layers 2 \\
        --policy cnn18c --cache <empty folder> [--device cuda]

Builds the random-weight graph (``--layers`` blocks, the full model's op shapes for the plan, the last block
pruned), then commits it three times: without the cache (the setup's time), into the empty ``--cache`` (a miss: the
same work plus writing the leaves) and from it (a hit: reading, the spot check and rebuilding the trees). It checks
that the three commitments have the same roots and that a Fiat-Shamir query proves and verifies with the same bytes
under the fresh and the cached commitment."""
from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from pathlib import Path

import torch

CODE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(CODE / "src"))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="llama2-7b")
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--seq", type=int, default=64)
    ap.add_argument("--policy", default="cnn18c")
    ap.add_argument("--cache", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--skip-fresh", action="store_true", help="no uncached commitment (the miss times the setup)")
    args = ap.parse_args()
    torch.set_num_threads(args.threads)
    from pvi.fullcheck import protocol as proto
    from pvi.fullcheck.analytic import decoder_shapes
    from pvi.fullcheck.transformer import CONFIGS, build_decoder

    cache = Path(args.cache)
    if cache.exists() and any(cache.iterdir()):
        raise SystemExit(f"{cache} is not empty: the miss must be a miss")
    device = torch.device(args.device)
    cfg = CONFIGS[args.model]
    head = {"model": args.model, "layers": args.layers, "seq": args.seq, "policy": args.policy, "device": str(device),
            "host": platform.node(), "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    t0 = time.perf_counter()
    graph = build_decoder(cfg, n_layers=args.layers, calib_tokens=32, seed=0, prune_last=True)
    shapes = decoder_shapes(cfg, prune_last=True)
    print(json.dumps({**head, "step": "build", "s": round(time.perf_counter() - t0, 3)}), flush=True)

    def commit(cache_arg):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t = time.perf_counter()
        c = proto.commit_graph(graph, 4, device=device, policy=args.policy, model_ops=shapes, cache=cache_arg)
        if device.type == "cuda":
            torch.cuda.synchronize()
        return c, time.perf_counter() - t

    runs = {}
    if not args.skip_fresh:
        runs["fresh"] = commit(False)
    runs["miss"] = commit(cache)
    runs["hit"] = commit(cache)
    ref = runs["fresh"][0] if "fresh" in runs else runs["miss"][0]
    files = list(cache.glob("*.npz"))
    for name, (c, s) in runs.items():
        same = (c.publics == ref.publics and c.group_publics == ref.group_publics
                and c.table_publics == ref.table_publics)
        print(json.dumps({**head, "step": name, "setup_cache": c.setup_cache, "s": round(s, 3), "same_roots": same,
                          "plan": c.plan.policy if c.plan else "paper", "trees": len(c.groups)}), flush=True)
    print(json.dumps({**head, "step": "files", "n": len(files), "mb": round(sum(f.stat().st_size for f in files) / 1e6, 2)}),
          flush=True)
    x = torch.randint(0, cfg.vocab, (1, args.seq), generator=torch.Generator().manual_seed(1))
    params = proto.params_for(128, len(shapes), rate=4, fiat_shamir=True, plan=ref.plan)
    out = {}
    for name in ("miss", "hit"):
        c = runs[name][0]
        v = proto.Verifier(graph.public(), params, "C", publics=c.publics, groups=c.group_publics,
                           tables=c.table_publics, lean=True)
        res = proto.run_query(proto.Prover(graph, device=device, commitments=c, lean=True), v, x, wire=True)
        out[name] = res
    ok = (all(r["accepted"] for r in out.values()) and out["miss"]["bytes"] == out["hit"]["bytes"]
          and all(runs[k][0].publics == ref.publics for k in runs) and runs["hit"][0].setup_cache == "hit")
    print(json.dumps({**head, "step": "query", "accepted": {k: r["accepted"] for k, r in out.items()},
                      "bytes_equal": out["miss"]["bytes"] == out["hit"]["bytes"],
                      "bytes_total": sum(out["hit"]["bytes"].values()), "ok": ok}), flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
